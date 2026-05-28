import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, obs, sats, scene
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.orbital import random_orbit, rv2HN

from envs.reward_utils import AgileEOSReward, TerminationGuard
from envs import build_targets, decision_interval, duration, n_ahead, orbit_alt_km, SAT_ARGS_POWER

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
        dyn_type = PowerSatDyn

        def is_alive(self, log_failure=True) -> bool:
            if not super().is_alive(log_failure=log_failure):
                return False
            frac = getattr(self.dynamics, "storage_level_fraction", None)
            if frac is not None and frac >= 0.98:
                if log_failure:
                    self.dynamics.logger.warning(
                        f"Satellite {self.name} failed: storage level fraction {frac:.4f} >= 0.98"
                    )
                return False
            return True

    return PowerSat








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


max_step_duration = decision_interval

targets = build_targets(scene)

n_intervals = int(np.ceil(duration / max_step_duration))


def get_downlink_env(n_ahead=n_ahead):
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
        terminate_on_time_limit=True,
        log_level="ERROR",
        world_type=EclipseGroundStationWorld,
        world_args=EclipseGroundStationWorld.default_world_args(),
    )

    return env
