import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, obs, sats, scene
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.orbital import random_orbit, rv2HN

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.data.no_data import NoDataStore

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


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


class PowerSatDyn(dyn.GroundStationDynModel, dyn.ImagingDynModel):
    """Imaging dynamics + ground-station access hooks."""

    pass


def power_sat_generator(n_ahead=32, include_time=False):
    class PowerSat(sats.ImagingSatellite):
        # charge, desat, downlink, image one of the upcoming targets.
        action_spec = [
            act.Charge(),
            act.Desat(),
            act.Downlink(),
            act.Image(n_ahead_image=n_ahead),
        ]

        observation_spec = [
            obs.SatProperties(
                dict(prop="omega_BH_H", norm=0.03),
                dict(prop="c_hat_H"),
                dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                dict(prop="v_BN_P", norm=7616.5),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speeds_fraction"),
                dict(prop="s_hat_H", fn=s_hat_H),
                dict(prop="storage_level_fraction"),
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
        dyn_type = PowerSatDyn

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
        maxWheelSpeed=6000.0,
        wheelSpeeds=lambda: np.random.uniform(-2000, 2000, 3),
        desatAttitude="nadir",
        # Changed from random initial storage to empty storage.
        # This is important for target-based downlink reward:
        # if the buffer starts with random bits, those bits do not correspond
        # to known imaged targets, so they cannot be rewarded like paper.
        storageInit=0,
        # Imaging and downlink data rates.
        transmitterBaudRate=-50 * 8e6,
        transmitterPowerDraw=-25.0,
        instrumentBaudRate=5 * 8e6,
        basePowerDraw=-10.0,
        panelArea=0.25,
    )
)


class AgileEOSData(Data):
    """
    Per-step and cumulative target-lifecycle data.

    imaged:
        Targets imaged during this step or cumulatively.

    downlinked:
        Targets downlinked during this step or cumulatively.

    known:
        Targets known to the satellite/scenario.

    onboard:
        Ordered list of imaged targets that are currently assumed to be
        waiting in the onboard data buffer.
    """

    def __init__(
        self,
        imaged=None,
        downlinked=None,
        known=None,
        onboard=None,
    ):
        if imaged is None:
            imaged = []
        if downlinked is None:
            downlinked = []
        if known is None:
            known = set()
        if onboard is None:
            onboard = []

        self.imaged = set(imaged)
        self.downlinked = set(downlinked)
        self.known = set(known)
        self.onboard = list(onboard)

    def __add__(self, other: "AgileEOSData") -> "AgileEOSData":
        imaged = self.imaged | other.imaged
        downlinked = self.downlinked | other.downlinked
        known = self.known | other.known

        onboard = list(self.onboard)

        # Add newly imaged targets to the onboard queue if they have not
        # already been downlinked.
        for target in other.onboard:
            if target not in onboard and target not in downlinked:
                onboard.append(target)

        # Remove targets that were downlinked this step.
        onboard = [target for target in onboard if target not in other.downlinked]

        return AgileEOSData(
            imaged=imaged,
            downlinked=downlinked,
            known=known,
            onboard=onboard,
        )


class AgileEOSDataStore(DataStore):
    """
    DataStore that detects:
      1. newly imaged targets from storage increase + satellite.latest_target
      2. newly downlinked targets from storage decrease during Downlink mode

    This replaces the old approach of rewarding raw storage drain.
    """

    data_type = AgileEOSData

    def get_log_state(self):
        msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()

        # storedData[0] is the image buffer level in BSK-RL's imaging setup.
        # storage_level is the total storage level exposed by ImagingDynModel.
        return dict(
            image_buffer=float(msg.storedData[0]),
            total_storage=float(self.satellite.dynamics.storage_level),
        )

    def _last_action_key(self):
        try:
            return self.satellite.action_builder.prev_action_key
        except Exception:
            return ""

    def _in_ground_station_contact(self):
        try:
            return len(self.satellite.current_opportunities(types="ground_station")) > 0
        except Exception:
            return bool(getattr(self.satellite.dynamics, "gs_in_view", False))

    def _image_size_bits(self):
        # In ImagingDynModel, transmitter.packetSize defaults to image size
        # if transmitterPacketSize is not explicitly specified.
        try:
            return abs(float(self.satellite.dynamics.transmitter.packetSize))
        except Exception:
            return abs(
                float(self.satellite.sat_args.get("instrumentBaudRate", 5 * 8e6))
            )

    def compare_log_states(self, old_state, new_state) -> AgileEOSData:
        old_buffer = old_state["image_buffer"]
        new_buffer = new_state["image_buffer"]

        delta_buffer = new_buffer - old_buffer

        new_imaged = []
        new_downlinked = []
        new_onboard = []

        # Image event: buffer increased.
        if delta_buffer > 0:
            if self.satellite.latest_target is not None:
                target = self.satellite.latest_target
                new_imaged.append(target)
                new_onboard.append(target)

        # Downlink event: buffer decreased while the selected mode is downlink
        # and a ground-station contact is available.
        if (
            delta_buffer < 0
            and self._last_action_key() == "action_downlink"
            and self._in_ground_station_contact()
        ):
            drained_bits = abs(delta_buffer)
            image_size_bits = self._image_size_bits()

            # The transmitter packet size is set to the image size, so this
            # converts buffer decrease into a count of complete target images
            # downlinked during this step.
            n_packets = int(np.floor((drained_bits + 1e-6) / image_size_bits))

            if n_packets > 0:
                onboard_queue = list(self.data.onboard)
                new_downlinked = onboard_queue[:n_packets]

        return AgileEOSData(
            imaged=new_imaged,
            downlinked=new_downlinked,
            onboard=new_onboard,
        )


class AgileEOSReward(GlobalReward):
    """
    Paper-aligned agile EOS reward.

    For each first-time imaged target:
        reward += 0.1 / (priority * n_intervals)

    For each first-time downlinked target:
        reward += 1.0 / (priority * n_intervals)

    Failure reward is not included here. Failure is handled by
    TerminationGuard + failure_penalty.
    """

    data_store_type = AgileEOSDataStore

    def __init__(self, n_intervals, priority_epsilon=1e-9):
        super().__init__()
        self.n_intervals = int(n_intervals)
        self.priority_epsilon = priority_epsilon

    def initial_data(self, satellite):
        return self.data_type(known=self.scenario.targets)

    def create_data_store(self, satellite):
        super().create_data_store(satellite)

        # Same idea as UniqueImageReward: do not keep offering targets that
        # have already been imaged.
        def unique_target_filter(opportunity):
            if opportunity["type"] == "target":
                return opportunity["object"] not in satellite.data_store.data.imaged
            return True

        satellite.add_access_filter(unique_target_filter)

    def _priority_weight(self, target):
        priority = float(getattr(target, "priority", 1.0))
        return 1.0 / max(priority, self.priority_epsilon)

    def calculate_reward(self, new_data_dict):
        rewards = {}

        for sat_id, new_data in new_data_dict.items():
            rewards[sat_id] = 0.0

            # First-time imaging reward:
            # paper term = 0.1 / |I| * H(w_j)
            # where H(w_j) = 1 / p_j if not previously imaged and now imaged.
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    rewards[sat_id] += (
                        0.1 / self.n_intervals * self._priority_weight(target)
                    )

            # First-time downlink reward:
            # paper term = 1 / |I| * H(d_j)
            # where H(d_j) = 1 / p_j if not previously downlinked and now downlinked.
            for target in new_data.downlinked:
                if target not in self.data.downlinked:
                    rewards[sat_id] += (
                        1.0 / self.n_intervals * self._priority_weight(target)
                    )

        return rewards


class TerminationGuard(GlobalReward):
    """
    Adds failure/termination only; contributes zero reward.

    Failure conditions:
      - battery invalid
      - reaction wheel speeds invalid
      - storage buffer nearly full / overflow condition
    """

    data_store_type = NoDataStore

    def calculate_reward(self, new_data_dict):
        return {sat_id: 0.0 for sat_id in new_data_dict.keys()}

    def is_terminated(self, satellite) -> bool:
        dyn_model = satellite.dynamics

        if hasattr(dyn_model, "battery_valid") and not dyn_model.battery_valid():
            return True

        if hasattr(dyn_model, "rw_speeds_valid") and not dyn_model.rw_speeds_valid():
            return True

        frac = getattr(dyn_model, "storage_level_fraction", None)
        if frac is not None and frac >= 0.98:
            return True

        return False


class EclipseGroundStationWorld(
    world.EclipseWorldModel, world.GroundStationWorldModel, world.AtmosphereWorldModel
):
    """Combined world that provides eclipse, ground-station, and atmosphere."""

    @classmethod
    def default_world_args(cls, **kwargs):
        # Merge defaults from eclipse, ground-station, and atmosphere world models.
        e_defaults = world.EclipseWorldModel.default_world_args()
        g_defaults = world.GroundStationWorldModel.default_world_args()
        a_defaults = world.AtmosphereWorldModel.default_world_args()
        merged = e_defaults.copy()
        merged.update(a_defaults)
        merged.update(g_defaults)
        for k, v in kwargs.items():
            merged[k] = v
        return merged


duration = 5700.0 * 5  # 5 orbits
max_step_duration = 300.0

target_distribution = "uniform"
n_targets = 135
n_ahead = 3

if target_distribution == "uniform":
    targets = scene.UniformTargets(n_targets)
elif target_distribution == "cities":
    targets = scene.CityTargets(n_targets)
else:
    raise ValueError(f"Unknown target_distribution: {target_distribution}")

n_intervals = int(np.ceil(duration / max_step_duration))


def get_downlink_env():
    env = SatelliteTasking(
        satellite=power_sat_generator(n_ahead=n_ahead, include_time=False)(
            name="EO1-power",
            sat_args=SAT_ARGS_POWER,
        ),
        scenario=targets,
        rewarder=(
            AgileEOSReward(n_intervals=n_intervals),
            TerminationGuard(),
        ),
        sim_rate=0.5,
        max_step_duration=max_step_duration,
        time_limit=duration,
        failure_penalty=-1.0,
        # Keeping previous time-limit behavior:
        # the episode time limit is treated separately from real failures.
        terminate_on_time_limit=False,
        log_level="ERROR",
        world_type=EclipseGroundStationWorld,
        world_args=EclipseGroundStationWorld.default_world_args(),
    )

    return env
