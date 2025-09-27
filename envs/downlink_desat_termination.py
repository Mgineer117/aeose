import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw, world
from bsk_rl.utils.orbital import random_orbit, rv2HN


bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from bsk_rl.data.base import GlobalReward, NoDataStore
from bsk_rl.data.unique_image_data import UniqueImageReward


class Density(obs.Observation):
    def __init__(
        self,
        interval_duration=60 * 3,
        intervals=10,
        norm=3,
    ):
        self.satellite: "sats.AccessSatellite"
        super().__init__()
        self.interval_duration = interval_duration
        self.intervals = intervals
        self.norm = norm

    def get_obs(self):
        if self.intervals == 0:
            return []

        self.satellite.calculate_additional_windows(
            self.simulator.sim_time
            + (self.intervals + 1) * self.interval_duration
            - self.satellite.window_calculation_time
        )
        soonest = self.satellite.upcoming_opportunities_dict(types="target")
        rewards = np.array([opportunity.priority for opportunity in soonest])
        times = np.array([opportunities[0][1] for opportunities in soonest.values()])
        time_bins = np.floor((times - self.simulator.sim_time) / self.interval_duration)
        densities = [sum(rewards[time_bins == i]) for i in range(self.intervals)]
        return np.array(densities) / self.norm

#Removing this function and using in-built wheel_speeds_fraction observation instead to help with wheel desat action 
#def wheel_speed_3(sat):
#    return np.array(sat.dynamics.wheel_speeds[0:3]) / 630


def s_hat_H(sat):
    r_SN_N = (
        sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
            sat.simulator.world.sun_index
        ]
        .read()
        .PositionVector
    )
    r_BN_N = sat.dynamics.r_BN_N
    r_SB_N = np.array(r_SN_N) - np.array(r_BN_N)
    r_SB_H = rv2HN(r_BN_N, sat.dynamics.v_BN_N) @ r_SB_N
    return r_SB_H / np.linalg.norm(r_SB_H)


def power_sat_generator(n_ahead=32, include_time=False):
    class PowerSat(sats.ImagingSatellite):
        action_spec = [act.Image(n_ahead_image=n_ahead), act.Charge(), act.Desat(), act.Downlink()] #included Desat and downlink action
        observation_spec = [
            obs.SatProperties(
                dict(prop="omega_BH_H", norm=0.03),
                dict(prop="c_hat_H"),
                dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                dict(prop="v_BN_P", norm=7616.5),
                dict(prop="battery_charge_fraction"),
                #dict(prop="wheel_speed_3", fn=wheel_speed_3), #removed to use in-built wheel_speeds_fraction observation instead to help with wheel desat action
                dict(prop="wheel_speeds_fraction"), # wheel speeds normalized by max to help with wheel desat action
                dict(prop="s_hat_H", fn=s_hat_H),
                dict(prop="data_buffer_fraction"), # Added for downlink awareness
            ),

            obs.OpportunityProperties(
                dict(prop="priority"),
                dict(prop="r_LB_H", norm=800 * 1e3),
                dict(prop="target_angle", norm=np.pi / 2),
                dict(prop="target_angle_rate", norm=0.03),
                dict(prop="opportunity_open", norm=300.0),
                dict(prop="opportunity_close", norm=300.0),
                type="target",
                n_ahead_observe=n_ahead,
            ),

            # Ground station opportunities for downlink
            obs.OpportunityProperties(
                dict(prop="opportunity_open", norm=300.0),
                dict(prop="opportunity_close", norm=300.0),
                type="ground_station",
                n_ahead_observe=n_ahead,
            ),
            obs.Eclipse(norm=5700),
            Density(intervals=20, norm=5),
        ]

        if include_time:
            observation_spec.append(obs.Time())

        fsw_type = fsw.SteeringImagerFSWModel

    return PowerSat


SAT_ARGS = dict(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    batteryStorageCapacity=80.0 * 3600 * 100,
    storedCharge_Init=80.0 * 3600 * 100.0,
    dataStorageCapacity=200 * 8e6 * 100,
    u_max=0.4,
    imageTargetMinimumElevation=np.arctan(800 / 500),
    K1=0.25,
    K3=3.0,
    omega_max=np.radians(5),
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_orbit(alt=800),
)

SAT_ARGS_POWER = {}
SAT_ARGS_POWER.update(SAT_ARGS)
SAT_ARGS_POWER.update(
    dict(
        batteryStorageCapacity=120.0 * 3600,
        storedCharge_Init=lambda: 120.0 * 3600 * np.random.uniform(0.4, 1.0),
        rwBasePower=20.4,
        instrumentPowerDraw=-10,
        thrusterPowerDraw=-30,
        nHat_B=np.array([0, 0, -1]),

        maxWheelSpeed=6000.0, # ~630 rad/s defining max wheel speed to help with wheel_speeds_fraction observation #https://www.aac-clyde.space/what-we-do/space-products-components/adcs/rw400 
        wheelSpeeds=lambda: np.random.uniform(-2000, 2000, 3),
        desatAttitude="nadir",
        
        storageInit=lambda: np.random.randint(0, int(0.01 * SAT_ARGS["dataStorageCapacity"])),
        transmitterBaudRate=- 50 * 8e6,      # bits/s  (NEGATIVE drains buffer during Downlink)
        transmitterPowerDraw=- 25.0,         # W       (power draw while Downlinking
        instrumentBaudRate = 5 * 8e6,   # bits/s produced while imaging (e.g., 5 MB/s)
        basePowerDraw = -10.0,   # W always-on loads (negative = consumption)
        panelArea     = 0.25,    # m^2 of solar array (tune as needed)
    )
)

class DownlinkReward(GlobalReward):
    def __init__(self, data_value_per_bit=1e-6, storage_penalty_threshold=0.95, opp_type="ground_station"):
        super().__init__()
        self.data_value_per_bit = data_value_per_bit
        self.storage_penalty_threshold = storage_penalty_threshold
        self.previous_storage = {}
        self.opp_type = opp_type

    def _in_downlink(self, sat):
        # true if a GS contact is open "now"
        try:
            windows = sat.current_opportunities(types=self.opp_type)
            return bool(windows)
        except Exception:
            return False

    def _get_storage(self, dyn):
        # survive minor naming differences across bsk_rl versions
        for name in ("storage_level", "data_storage_level", "buffer_level"):
            if hasattr(dyn, name):
                return getattr(dyn, name)
        return 0.0

    def _get_storage_frac(self, dyn):
        for name in ("storage_level_fraction", "data_buffer_fraction", "buffer_fraction"):
            if hasattr(dyn, name):
                return getattr(dyn, name)
        return 0.0

    def calculate_reward(self, new_data_dict):
        rewards = {}
        for sat_id in self.simulator.satellites.keys():
            sat = self.simulator.satellites[sat_id]
            dyn = sat.dynamics
            current_storage = self._get_storage(dyn)
            current_frac    = self._get_storage_frac(dyn)

            r = 0.0
            if sat_id in self.previous_storage and self._in_downlink(sat):
                data_downlinked = max(0.0, self.previous_storage[sat_id] - current_storage)
                r += data_downlinked * self.data_value_per_bit

            if current_frac > self.storage_penalty_threshold:
                r -= 5.0 * (current_frac - self.storage_penalty_threshold)

            self.previous_storage[sat_id] = current_storage
            rewards[sat_id] = r
        return rewards

class TerminationGuard(GlobalReward):
    #Adds failure/termination only; contributes zero reward
    data_store_type = NoDataStore  # any store type works; we don't use new data

    def calculate_reward(self, new_data_dict):
        # No reward contribution; leave imaging reward to UniqueImageReward
        return {sat_id: 0.0 for sat_id in new_data_dict.keys()}

    def is_terminated(self, satellite) -> bool:
        dyn = satellite.dynamics
        if hasattr(dyn, "battery_valid") and not dyn.battery_valid():
            return True
        if hasattr(dyn, "rw_speeds_valid") and not dyn.rw_speeds_valid():
            return True
        frac = getattr(dyn, "storage_level_fraction", None)
        if frac is not None and frac >= 0.98:
            return True
        return False

duration = 5700.0 * 5  # 5 orbits
target_distribution = "uniform"
n_targets = 3000
n_ahead = 32

if target_distribution == "uniform":
    targets = scene.UniformTargets(n_targets)
elif target_distribution == "cities":
    targets = scene.CityTargets(n_targets)


def get_env():
    env = SatelliteTasking(
        satellite=power_sat_generator(n_ahead=32, include_time=False)(
            name="EO1-power",
            sat_args=SAT_ARGS_POWER,
        ),
        scenario=targets,
        rewarder=(UniqueImageReward(), DownlinkReward(data_value_per_bit=1e-6), TerminationGuard()),
        sim_rate=0.5,
        max_step_duration=300.0,
        time_limit=duration,
        failure_penalty=-10.0,
        terminate_on_time_limit=True,
        log_level="ERROR",
        world_type=world.GroundStationWorldModel,  # Verified from cloud environment example
        world_args=world.GroundStationWorldModel.default_world_args(),  # Verified from cloud environment example
    )

    return env
