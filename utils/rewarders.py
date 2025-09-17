import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_orbit, rv2HN
from bsk_rl.sim import world

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

import bsk_rl
from bsk_rl import act
from bsk_rl.data.base import Data, DataStore, GlobalReward

for name in ["Charge", "Desat", "Downlink", "Image"]:
    print(name, hasattr(act, name))


"""
# =============================================== #
# THIS IS AN EXAMPLE REWARDER TO GET YOU STARTED. #
# =============================================== #
"""

# class NoData(Data):
#     """Holds no data."""

#     def __init__(self, *args, **kwargs):
#         """Holds no data."""
#         return super().__init__(*args, **kwargs)

#     def __add__(self, other):
#         """Add nothing to nothing."""
#         return self.__class__()


# class NoDataStore(DataStore):
#     """DataStore for no data."""

#     data_type = NoData

#     def __init__(self, *args, **kwargs):
#         """Stores and generates no data."""
#         return super().__init__(*args, **kwargs)

#     def compare_log_states(self, old_state, new_state):
#         """Always returns no data."""
#         return self.data_type()


# class NoReward(GlobalReward):
#     """GlobalReward for no data."""

#     data_store_type = NoDataStore

#     def __init__(self, *args, **kwargs):
#         """Returns zero reward at every step.

#         This reward system is useful for debugging environments, but is not useful for
#         training, since reward is always zero for every satellite.
#         """
#         return super().__init__(*args, **kwargs)

#     def calculate_reward(self, new_data_dict):
#         """Reward nothing."""
#         print(new_data_dict)
#         return {sat: 0.0 for sat in new_data_dict.keys()}


# class UniqueImageReward(GlobalReward):
#     """GlobalReward for rewarding unique images."""

#     data_store_type = UniqueImageStore

#     def __init__(
#         self,
#         reward_fn: Callable = lambda p: p,
#     ) -> None:
#         """GlobalReward for rewarding unique images.

#         This data system should be used with the :class:`~bsk_rl.sats.ImagingSatellite` and
#         a scenario that generates targets, such as :class:`~bsk_rl.scene.UniformTargets` or
#         :class:`~bsk_rl.scene.CityTargets`.

#         The satellites all start with complete knowledge of the targets in the scenario.
#         Each target can only give one satellite a reward once; if any satellite has imaged
#         a target, reward will never again be given for that target. The satellites filter
#         known imaged targets from consideration for imaging to prevent duplicates.
#         Communication can transmit information about what targets have been imaged in order
#         to prevent reimaging.


#         Args:
#             scenario: GlobalReward.scenario
#             reward_fn: Reward as function of priority.
#         """
#         super().__init__()
#         self.reward_fn = reward_fn

#     def initial_data(self, satellite: "Satellite") -> "UniqueImageData":
#         """Furnish data to the scenario.

#         Currently, it is assumed that all targets are known a priori, so the initial data
#         given to the data store is the list of all targets.
#         """
#         return self.data_type(known=self.scenario.targets)

#     def create_data_store(self, satellite: "Satellite") -> None:
#         """Override the access filter in addition to creating the data store."""
#         super().create_data_store(satellite)

#         def unique_target_filter(opportunity):
#             if opportunity["type"] == "target":
#                 return opportunity["object"] not in satellite.data_store.data.imaged
#             return True

#         satellite.add_access_filter(unique_target_filter)

#     def calculate_reward(
#         self, new_data_dict: dict[str, UniqueImageData]
#     ) -> dict[str, float]:
#         """Reward each new unique image once.

#         Reward is evaluated based on ``self.reward_fn(target.priority)``.

#         Args:
#             new_data_dict: Record of new images for each satellite

#         Returns:
#             reward: Cumulative reward across satellites for one step
#         """
#         reward = {}
#         imaged_targets = sum(
#             [new_data.imaged for new_data in new_data_dict.values()], []
#         )
#         for sat_id, new_data in new_data_dict.items():
#             reward[sat_id] = 0.0
#             for target in new_data.imaged:
#                 if target not in self.data.imaged:
#                     reward[sat_id] += self.reward_fn(
#                         target.priority
#                     ) / imaged_targets.count(target)

#         return reward


class PiecewiseReward(GlobalReward):
    """
    Reward:
      - Terminal failure: -10 if battery empty OR any wheel >= max OR storage full
      - If action == Image(c_j): +0.1 for first-time image of j
      - Else: 0
    """

    def __init__(self):
        super().__init__()

    # def some functions

    def calculate_reward(self, new_data_dict: dict[str]) -> dict[str, float]:
        sat = env.satellite

        # ----- Failure checks (Eq. 4) -----
        battery_empty = sat.dynamics.battery_charge_fraction <= 0.0
        wheel_max = getattr(sat.dynamics, "maxWheelSpeed", 630.0)
        wheels_over = any(abs(w) >= wheel_max for w in sat.dynamics.wheel_speeds[:3])

        # storage_level_fraction is exposed by your SatProperties
        storage_full = (
            getattr(sat, "storage_level_fraction", None) is not None
            and sat.storage_level_fraction >= 1.0
        )

        if battery_empty or wheels_over or storage_full:
            return -10.0

        # ----- Action-dependent reward (Eq. 3) -----
        a = info.get(
            "action_name"
        )  # SatelliteTasking typically fills this; if not, you can set it in a wrapper
        r = 0.0
        if a == "image":
            tid = info.get("imaged_target_id")
            if tid is not None and tid not in self.imaged_once:
                r += 0.1
                self.imaged_once.add(tid)
        return r


class PiecewiseDownlinkReward(GlobalReward):
    """
    Reward:
      - Terminal failure: -10 if battery empty OR any wheel >= max OR storage full
      - If action == Downlink: sum (1/priority_j) for first-time downlinks this step
      - If action == Image(c_j): +0.1 for first-time image of j
      - Else: 0
    """

    # def __init__(self, env):
    def __init__(self):
        super().__init__()
        # self.env = env
        # self.imaged_once = set()  # target IDs imaged at least once
        # self.downlinked_once = set()  # target IDs ever delivered

    def calculate_reward(
        self, new_data_dict: dict[str, UniqueImageData]
    ) -> dict[str, float]:
        reward = {}

        # sat = env.satellite

        # ----- Failure checks (Eq. 4) -----
        battery_empty = sat.dynamics.battery_charge_fraction <= 0.0
        wheel_max = getattr(sat.dynamics, "maxWheelSpeed", 630.0)
        wheels_over = any(abs(w) >= wheel_max for w in sat.dynamics.wheel_speeds[:3])

        # storage_level_fraction is exposed by your SatProperties
        storage_full = (
            getattr(sat, "storage_level_fraction", None) is not None
            and sat.storage_level_fraction >= 1.0
        )

        if battery_empty or wheels_over or storage_full:
            return -10.0

        # ----- Action-dependent reward (Eq. 3) -----
        a = info.get(
            "action_name"
        )  # SatelliteTasking typically fills this; if not, you can set it in a wrapper
        r = 0.0
        if a == "downlink":
            # If your env provides a list of (target_id, priority) delivered this step, use it:
            for tid, prio in info.get("downlinked_targets", []):
                if tid not in self.downlinked_once:
                    r += 1.0 / max(1.0, float(prio))
                    self.downlinked_once.add(tid)
        elif a == "image":
            tid = info.get("imaged_target_id")
            if tid is not None and tid not in self.imaged_once:
                r += 0.1
                self.imaged_once.add(tid)
        return r
