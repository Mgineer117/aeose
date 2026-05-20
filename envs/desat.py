import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion
from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_orbit
from envs import build_targets, decision_interval, duration, n_ahead, orbit_alt_km

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

# Import GlobalReward and provide a compatibility fallback for NoDataStore
from bsk_rl.data.base import DataStore, GlobalReward
from bsk_rl.data.no_data import NoDataStore
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


def power_sat_generator(n_ahead=32, include_time=False):
    class PowerSat(sats.ImagingSatellite):
        action_spec = [
            act.Image(n_ahead_image=n_ahead),
            act.Charge(),
            act.Desat(),
        ]  # included Desat action
        observation_spec = [
            obs.SatProperties(
                dict(prop="omega_BH_H", norm=0.03),
                dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                dict(prop="v_BN_P", norm=7616.5),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
                dict(prop="storage_level_fraction"),
            ),
            obs.OpportunityProperties(
                dict(prop="priority"),
                dict(prop="r_LB_H", norm=800 * 1e3),
                type="target",
                n_ahead_observe=n_ahead,
            ),
            obs.Eclipse(norm=5700),
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
    oe=lambda: random_orbit(alt=orbit_alt_km),
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
        maxWheelSpeed=6000.0,  # ~630 rad/s defining max wheel speed to help with wheel_speeds_fraction observation #https://www.aac-clyde.space/what-we-do/space-products-components/adcs/rw400
        wheelSpeeds=lambda: np.random.uniform(-2000, 2000, 3),
        desatAttitude="nadir",  # feel like we shouold desat to sun-pointing to help with power generation during desat
        storageInit=lambda: np.random.randint(
            0, int(0.01 * SAT_ARGS["dataStorageCapacity"])
        ),
        # transmitterBaudRate=-50 * 8e6,      # bits/s  (NEGATIVE drains buffer during Downlink)
        # transmitterPowerDraw=-25.0,         # W       (power draw while Downlinking
        instrumentBaudRate=+5 * 8e6,  # bits/s produced while imaging (e.g., 5 MB/s)
        basePowerDraw=-10.0,  # W always-on loads (negative = consumption)
        panelArea=0.25,  # m^2 of solar array (tune as needed)
    )
)


class TerminationGuard(GlobalReward):
    # Adds failure/termination only; contributes zero reward
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

targets = build_targets(scene)


def get_desat_env(n_ahead=n_ahead):
    env = SatelliteTasking(
        satellite=power_sat_generator(n_ahead=n_ahead, include_time=False)(
            name="EO1-power",
            sat_args=SAT_ARGS_POWER,
        ),
        scenario=targets,
        rewarder=(data.UniqueImageReward(), TerminationGuard()),
        sim_rate=0.5,
        max_step_duration=decision_interval,
        time_limit=duration,
        failure_penalty=-10.0,
        terminate_on_time_limit=True,
        log_level="ERROR",
    )

    return env
