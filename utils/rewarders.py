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
        self, new_data_dict: dict[str, data.UniqueImageData]
    ) -> dict[str, float]:
        reward = {}

        # sat = env.satellite

        # ----- Failure checks (Eq. 4) -----
        battery_empty = self.sat.dynamics.battery_charge_fraction <= 0.0
        wheel_max = getattr(self.sat.dynamics, "maxWheelSpeed", 630.0)
        wheels_over = any(
            abs(w) >= wheel_max for w in self.sat.dynamics.wheel_speeds[:3]
        )

        # storage_level_fraction is exposed by your SatProperties
        storage_full = (
            getattr(self.sat, "storage_level_fraction", None) is not None
            and self.sat.storage_level_fraction >= 1.0
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
