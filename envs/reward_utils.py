import numpy as np
from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.data.no_data import NoDataStore

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
            sim_time=self.satellite.simulator.sim_time,
        )

    def _last_action_key(self):
        try:
            return self.satellite.action_builder.prev_action_key
        except Exception:
            return ""

    def _in_ground_station_contact(self, old_time=None, new_time=None):
        if old_time is None:
            old_time = self.satellite.simulator.sim_time - 360.0
        if new_time is None:
            new_time = self.satellite.simulator.sim_time

        try:
            for opp in self.satellite.opportunities:
                if opp["type"] == "ground_station":
                    window = opp["window"]
                    if window[0] <= new_time and window[1] >= old_time:
                        return True
        except AttributeError:
            pass
        return False

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
        old_time = old_state.get("sim_time", 0.0)
        new_time = new_state.get("sim_time", 0.0)
        if (
            delta_buffer < 0
            and self._last_action_key() == "action_downlink"
            and self._in_ground_station_contact(old_time, new_time)
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
            # paper term = 0.1 * H(w_j)
            # where H(w_j) = 1 / p_j if not previously imaged and now imaged.
            for target in new_data.imaged:
                if target not in self.data.imaged:
                    rewards[sat_id] += (
                        0.1 * self._priority_weight(target)
                    )

            # First-time downlink reward:
            # paper term = sum H(d_j)
            # where H(d_j) = 1 / p_j if not previously downlinked and now downlinked.
            for target in new_data.downlinked:
                if target not in self.data.downlinked:
                    rewards[sat_id] += (
                        1.0 * self._priority_weight(target)
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
