import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_orbit, rv2HN
from bsk_rl.sim import world

bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

import bsk_rl, bsk_rl.actions as act
print(bsk_rl.__version__)
for name in ["Charge","Desat","Downlink","Image"]:
    print(name, hasattr(act, name))


class Density(obs.Observation):       #custom Density bins (a forward-looking “how crowded” feature) not part of paper
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


def s_hat_H(sat):      #Sun direction not part of paper
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

#Attitude helpers
def mrp_error_norm(sat):   #signma_B/R- norm of Modified Rodrigues Parameter (MRP) attitude error
    sigma = sat.fsw.attitude_error_mrp()
    return(np.linalg.norm(sigma))

def omega_norm(sat): # omega_B/N- norm of angular attitude rate vector
    w = getattr(sat.dynamics, 'omega_BN_B', None)
    if w is None:
        w = sat.dynamics.omega_BN_H #fallback
    return float(np.linalg.norm(w))

# Data buffer storage helper
def buffer_level(sat):      #Data buffer fill fraction in [0, 1]
    cap = max(1.0, float(sat.data.capacity_bytes))
    return float(sat.data.buffer_bytes) / cap

def downlinked_this_step(sat): #Data downlinked this step, normalized by buffer capacity
    cap = max(1.0, float(sat.data.capacity_bytes))
    return float(getattr(sat.data, "downlinked_last_step_bytes", 0.0)) / cap





def power_sat_generator(n_ahead=5, include_time=False):
    class PowerSat(sats.ImagingSatellite):
        action_spec = [act.Image(n_ahead_image=n_ahead), act.Charge(), act.Downlink(), act.Desat()]
        observation_spec = [
            obs.SatProperties(
                dict(prop="omega_BH_H", norm=0.03),    #no error norm or rate norm
                dict(prop="c_hat_H"),
                dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
                dict(prop="v_BN_P", norm=7616.5),
                dict(prop="battery_charge_fraction"),
                dict(prop="wheel_speed_3", fn=wheel_speed_3),
                dict(prop="s_hat_H", fn=s_hat_H),
                
                
                dict(prop="mrp_error_norm", fn=mrp_error_norm, norm=0.1), #Attitude error norm
                dict(prop="omega_norm",     fn=omega_norm,     norm=0.03), #Attitude rate norm
                dict(prop="buffer_level",   fn=buffer_level),  #Data buffer level
                dict(prop="downlink_amount", fn=downlinked_this_step), #Data downlinked this step (normalized by buffer capacity)
            ),
            obs.OpportunityProperties(
                dict(prop="priority"),
                dict(prop="r_LB_H", norm=500 * 1e3),
                dict(prop="target_angle", norm=np.pi / 2),
                dict(prop="target_angle_rate", norm=0.03),
                dict(prop="opportunity_open", norm=360.0),
                dict(prop="opportunity_close", norm=360.0),
                type="target",
                n_ahead_observe=n_ahead,
                type="ground_station",
                n_ahead_observe=1,

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
    imageTargetMinimumElevation=np.arctan(800 / 500),    #what's happening here? what's the 800? Check back
    K1=0.25,
    K3=3.0,
    omega_max=np.radians(5),
    servo_Ki=5.0,
    servo_P=150 / 5,
    oe=lambda: random_orbit(alt=500),
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
        desatAttitude="nadir",      #A ground-station pass model and a Downlink mode that consumes buffer and returns reward when passes occur
        storageInit=lambda: np.random.randint(0, int(0.01 * SAT_ARGS["dataStorageCapacity"])),
        transmitterBaudRate=-50 * 8e6,      # bits/s  (NEGATIVE drains buffer during Downlink)
        transmitterPowerDraw=-25.0,         # W       (power draw while Downlinking
        maxWheelSpeed=1500.0,               # RPM


    )
)

duration = 5700.0 * 3  # 3 orbits #5700 -> 95 minutes
target_distribution = "uniform"
n_targets = 135 #135 targets for 3 orbits- but should we just keep 3000 if MJ is able to get good results with that?
n_ahead = 5 # 5 n_ahead for 3 orbits - also should we keep 32 if MJ is able to get good results with that?

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
        rewarder=data.UniqueImageReward(),      #Implement the piecewise reward with downlink-driven returns and 0.1 image bonus; wire in resource-failure terminal penalty and episode termination
        sim_rate=0.5,   #change to 1?
        max_step_duration=360.0, #Updated from 300 to 360 to accomodate 6 minute steps
        time_limit=duration,
        failure_penalty=0.0,
        terminate_on_time_limit=True,
        log_level="ERROR",
        world_type=world.GroundStationWorldModel,
        world_args=world.GroundStationWorldModel.default_world_args(),
    )

    return env
