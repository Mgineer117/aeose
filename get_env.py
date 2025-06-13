import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_orbit, rv2HN

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)


def satellite_generator(observation, n_ahead=32, include_time=False):
    """_summary_

    Args:
        observation: Pick from "S1", "S2", "S3"
        n_ahead: Number of requests to include in the observation and action spaces
        include_time: Whether to include time through episode in the observation
    """

    assert observation in ["S1", "S2", "S3"]

    class CustomSatellite(sats.ImagingSatellite):
        action_spec = [act.Image(n_ahead_image=n_ahead)]
        if observation == "S1":
            observation_spec = [
                obs.SatProperties(
                    dict(prop="omega_BP_P", norm=0.03),
                    dict(prop="c_hat_P"),
                    dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    dict(prop="v_BN_P", norm=7616.5),
                ),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="r_LP_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    type="target",
                    n_ahead_observe=n_ahead,
                ),
            ]
        elif observation == "S2":
            observation_spec = [
                obs.SatProperties(
                    dict(prop="omega_BH_H", norm=0.03),
                    dict(prop="c_hat_H"),
                    dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    dict(prop="v_BN_P", norm=7616.5),
                ),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="r_LB_H", norm=orbitalMotion.REQ_EARTH * 1e3),
                    type="target",
                    n_ahead_observe=n_ahead,
                ),
            ]
        elif observation == "S3":
            observation_spec = [
                obs.SatProperties(
                    dict(prop="omega_BH_H", norm=0.03),
                    dict(prop="c_hat_H"),
                    dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                    dict(prop="v_BN_P", norm=7616.5),
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
            ]

        if include_time:
            observation_spec.append(obs.Time())
        fsw_type = fsw.SteeringImagerFSWModel

    return CustomSatellite


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


def wheel_speed_3(sat):
    return np.array(sat.dynamics.wheel_speeds[0:3]) / 630


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
        action_spec = [act.Image(n_ahead_image=n_ahead), act.Charge()]
        observation_spec = [
            obs.SatProperties(
                dict(prop="omega_BH_H", norm=0.03),
                dict(prop="c_hat_H"),
                dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                dict(prop="v_BN_P", norm=7616.5),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speed_3", fn=wheel_speed_3),
                dict(prop="s_hat_H", fn=s_hat_H),
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
        wheelSpeeds=lambda: np.random.uniform(-2000, 2000, 3),
        desatAttitude="nadir",
    )
)

duration = 5700.0 * 5  # 5 orbits
target_distribution = "uniform"
n_targets = 3000
n_ahead = 32

if target_distribution == "uniform":
    targets = scene.UniformTargets(n_targets)
elif target_distribution == "cities":
    targets = scene.CityTargets(n_targets)


def get_env():
    env_args = dict(
        satellite=power_sat_generator(n_ahead=32, include_time=False)(
            name="EO1-power",
            sat_args=SAT_ARGS_POWER,
        ),
        scenario=targets,
        rewarder=data.UniqueImageReward(),
        sim_rate=0.5,
        max_step_duration=300.0,
        time_limit=duration,
        failure_penalty=0.0,
        terminate_on_time_limit=True,
        log_level="INFO",
    )

    env = SatelliteTasking(**env_args)

    return env
