import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion

from bsk_rl import SatelliteTasking, act, data, obs, sats, scene
from bsk_rl.sim import fsw
from bsk_rl.utils.orbital import random_orbit, rv2HN
from bsk_rl.sim import world
from bsk_rl.data.base import Data, DataStore
from bsk_rl.data import GlobalReward
from typing import Optional


n_ahead = 32
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

import bsk_rl
from bsk_rl import act

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
        if not soonest:
            return np.zeros(self.intervals, dtype=float)

        # soonest is a dict: {target_id: [(open_time, close_time, opp_obj), ...], ...}
        # Take the first window for each entry
        opens = []
        rewards = []
        for _, windows in soonest.items():
            open_t, _, opp = windows[0]
            opens.append(open_t)
            # priority is on the opportunity (or fall back to target priority)
            rewards.append(getattr(opp, "priority", getattr(opp, "target_priority", 1.0)))

        opens = np.asarray(opens, dtype=float)
        rewards = np.asarray(rewards, dtype=float)

        time_bins = np.floor((opens - self.simulator.sim_time) / self.interval_duration).astype(int)
        densities = [float(rewards[time_bins == i].sum()) for i in range(self.intervals)]
        return np.asarray(densities, dtype=float) / self.norm


def wheel_speed_3(sat):
    d = sat.dynamics
    max_rpm = float(getattr(d, "maxWheelSpeed", 5000.0))
    ws = np.array(getattr(d, "wheel_speeds", [0.0, 0.0, 0.0]), dtype=float)[:3]
    return ws / max_rpm


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
#def mrp_error_norm(sat):   #signma_B/R- norm of Modified Rodrigues Parameter (MRP) attitude error
    #sigma = sat.fsw.attitude_error_mrp()
    #return(np.linalg.norm(sigma))

def omega_norm(sat): # omega_B/N- norm of angular attitude rate vector
    w = getattr(sat.dynamics, 'omega_BN_B', None)
    if w is None:
        w = sat.dynamics.omega_BH_H 
    return float(np.linalg.norm(w))

# Data buffer storage helper
def storage_fill_fraction(sat):
    msg = sat.dynamics.storageUnit.storageUnitDataOutMsg.read()
    used = float(getattr(msg, "storageLevel", 0.0))
    cap  = float(getattr(msg, "storageCapacity", 1.0))
    return 0.0 if cap <= 0 else max(0.0, min(1.0, used / cap))


#ground station helper
from Basilisk.simulation import groundLocation

def _register_ground_tracking(env):
    """Attach the spacecraft to all Basilisk GroundLocation modules in the world."""
    sc = env.satellite.dynamics.scObject
    w  = env.satellite.simulator.world

    # Try common containers if your world model exposes them
    for name in ("groundStations", "groundLocations", "groundStationList"):
        if hasattr(w, name):
            cont = getattr(w, name)
            items = cont if isinstance(cont, (list, tuple, set)) else [cont]
            for gl in items:
                if isinstance(gl, groundLocation.GroundLocation):
                    gl.addSpacecraftToModel(sc)

    # Generic sweep as a fallback (handles custom attribute names)
    for attr in dir(w):
        try:
            obj = getattr(w, attr)
        except Exception:
            continue
        if isinstance(obj, groundLocation.GroundLocation):
            obj.addSpacecraftToModel(sc)
        elif isinstance(obj, (list, tuple, set)):
            for it in obj:
                if isinstance(it, groundLocation.GroundLocation):
                    it.addSpacecraftToModel(sc)


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
                
                
                #dict(prop="mrp_error_norm", fn=mrp_error_norm, norm=0.1), #Attitude error norm
                dict(prop="omega_norm",     fn=omega_norm,     norm=0.03), #Attitude rate norm
                dict(prop="storage_fill_fraction", fn=storage_fill_fraction),

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
            ),

            # Next ground-station pass (so the policy can time Downlink)
            obs.OpportunityProperties(
                dict(prop="opportunity_open",  norm=360.0),
                dict(prop="opportunity_close", norm=360.0),
                type="ground_station",
                n_ahead_observe=1,
            ),
            obs.OpportunityProperties(
                dict(prop="opportunity_open",  norm=360.0),
                dict(prop="opportunity_close", norm=360.0),
                type="downlink",
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

MAX_WHEEL_RPM = 5000.0   # rated limit in RPM for your RW model
INIT_FRAC = 0.05         # start wheels within ±5% of max so episodes are stable
# Safety guard-bands
CHARGE_ENTRY_GUARD = 0.10   # if battery <= 10% you MUST be in Charge
CHARGE_EXIT_GUARD  = 0.25   # leave forced Charge once >= 25%

DESAT_ENTRY_FRAC   = 0.90   # if |wheel| >= 90% max you MUST be in Desat
DESAT_EXIT_FRAC    = 0.60   # leave forced Desat once <= 60% max


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
        wheelSpeeds   = (lambda maxrpm=MAX_WHEEL_RPM, frac=INIT_FRAC:np.random.uniform(-frac*maxrpm, +frac*maxrpm, 3).astype(float)),
        maxWheelSpeed=MAX_WHEEL_RPM,               # RPM- https://www.aac-clyde.space/what-we-do/space-products-components/adcs/rw400
        desatAttitude="sun",      #A ground-station pass model and a Downlink mode that consumes buffer and returns reward when passes occur
        storageInit=lambda: np.random.randint(0, int(0.01 * SAT_ARGS["dataStorageCapacity"])),
        transmitterBaudRate=-50 * 8e6,      # bits/s  (NEGATIVE drains buffer during Downlink)
        transmitterPowerDraw=-25.0,         # W       (power draw while Downlinking
        instrumentBaudRate = +5 * 8e6,   # bits/s produced while imaging (e.g., 5 MB/s)
        basePowerDraw = -10.0,   # W always-on loads (negative = consumption)
        panelArea     = 0.25,    # m^2 of solar array (tune as needed)

    )
)

#Rewards bsk_rl style
class ImageOnceReward(data.UniqueImageReward):
    
    #reward = 0.1 * (1 / priority) for the FIRST image of a target.
    
    def __init__(self):
        super().__init__(reward_fn=lambda p: 0.1 * (1.0 / float(p)))


class DownlinkOnceData(Data):
    #Data unit: which targets became fully downlinked at this step
    def __init__(self, downlinked: Optional[list] = None):
        if downlinked is None:
            downlinked = []
        self.downlinked = list(set(downlinked))

    def __add__(self, other: "DownlinkOnceData") -> "DownlinkOnceData":
        return self.__class__(downlinked=list(set(self.downlinked + other.downlinked)))
    

class DownlinkOnceStore(DataStore):
    
    #DataStore that detects when a target's buffer partition goes to zero (fully downlinked) between steps,
    #using the same log-diff pattern as UniqueImageStore (see docs). 
    
    data_type = DownlinkOnceData

    def initial_data(self) -> DownlinkOnceData:
        # Make sure this store knows the scenario's targets
        self.known = list(getattr(self.data, "known", []))
        return DownlinkOnceData()

    def get_log_state(self):
        
        #Return a per-partition vector representing current stored data per target(reads storageUnitDataOutMsg arrays).
        
        msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        # storedData is bytes per partition; cast to np.array for diff checks
        import numpy as np
        return np.array(msg.storedData, dtype=float)

    def compare_log_states(self, old_state, new_state) -> DownlinkOnceData:
        
        #Identify partitions that went from >0 to ==0 this step -> fully downlinked targets. Uses 'storedDataName' to map buffer index back to target id, like the examples.
    
        EPS = 8.0  # bytes; one byte-word or less => treated as empty this step
        emptied = np.where((old_state > EPS) & (new_state <= EPS))[0].tolist()
        if not emptied:
            return DownlinkOnceData()

        # map partition index -> target object by id
        msg = self.satellite.dynamics.storageUnit.storageUnitDataOutMsg.read()
        names = list(getattr(msg, "storedDataName", []))
        mapped = []
        for i in emptied:
            raw = names[i]
            name = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            mapped.append(name)

        # Build id->Target dict once per call
        by_id = {str(getattr(t, "id")): t for t in getattr(self.data, "known", [])}
        targets = [by_id[n] for n in mapped if n in by_id]
        return DownlinkOnceData(downlinked=targets)


class DownlinkOnceReward(GlobalReward):
    
    #Downlink term:reward = sum(1 / priority_j) for targets that became fully downlinked for the FIRST time this step (any satellite).

    data_store_type = DownlinkOnceStore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cum_reward = {}          # re quired by bsk_rl.data.composition
        self._downlinked_once = set() # local memory; also set in reset_overwrite_previous

    def reset_overwrite_previous(self) -> None:
        # tracks which targets have already yielded downlink reward
        self._downlinked_once = set()

    def initial_data(self, satellite) -> DownlinkOnceData:
        # give the store the full known target list, like UniqueImageReward does
        if not hasattr(self, "known_targets"):
            # the GlobalReward has scenario linked; scenario.targets is the list
            self.known_targets = list(getattr(self.scenario, "targets", []))
        d = DownlinkOnceData()
        d.known = self.known_targets  # attach 'known' so datastore can map names -> targets
        return d
    
    def is_terminated(self, satellite) -> bool:
        batt_frac = float(satellite.dynamics.battery_charge_fraction)
        batt_empty = (batt_frac <= 1e-6)

        limit = float(getattr(satellite.dynamics, "maxWheelSpeed", 5000.0))
        ws = np.array(getattr(satellite.dynamics, "wheel_speeds", [0.0, 0.0, 0.0]), dtype=float)[:3]
        max_abs_w = float(np.max(np.abs(ws)))
        wheels_over = (max_abs_w >= 0.98 * limit)

        storage_full = (storage_fill_fraction(satellite) >= 1.0)

        if batt_empty or wheels_over or storage_full:
            return True

        env = satellite.simulator.env
        if not hasattr(env, "user_state") or env.user_state is None:
            env.user_state = {}
        st = env.user_state
        last_action = st.get("last_action", None)

        if batt_frac <= CHARGE_ENTRY_GUARD:
            st["force_charge"] = True
        elif batt_frac >= CHARGE_EXIT_GUARD:
            st["force_charge"] = False

        if max_abs_w >= DESAT_ENTRY_FRAC * limit:
            st["force_desat"] = True
        elif max_abs_w <= DESAT_EXIT_FRAC * limit:
            st["force_desat"] = False

        must_charge = bool(st.get("force_charge", False))
        must_desat  = bool(st.get("force_desat", False))

        if must_charge and (last_action != "Charge"):
            return True
        if must_desat and (last_action != "Desat"):
            return True

        return False

    def calculate_reward(self, new_data_dict: dict[str, DownlinkOnceData]) -> dict[str, float]:
        # Aggregate step reward across sats per the BSK-RL API (return dict per sat)
        rewards = {sid: 0.0 for sid in new_data_dict.keys()}
        for sid, d in new_data_dict.items():
            step_sum = 0.0
            for tgt in getattr(d, "downlinked", []):
                if tgt.id not in self._downlinked_once:
                    p = float(getattr(tgt, "priority", 1.0)) or 1.0
                    step_sum += 1.0 / p
                    self._downlinked_once.add(tgt.id)
            rewards[sid] = step_sum
            self.cum_reward[sid] = self.cum_reward.get(sid, 0.0) + step_sum
        return rewards
    
class _LogLastAction(GlobalReward):
    def __call__(self, env, info):
        idx = info.get("action_index", None)
        name = info.get("action_name", None)
        if name is None and idx is not None and 0 <= idx < len(env.satellite.action_spec):
            name = env.satellite.action_spec[idx].__class__.__name__
        if not hasattr(env, "user_state") or env.user_state is None:
            env.user_state = {}
        env.user_state["last_action"] = name

    def calculate_reward(self, new_data_dict):
        # zero reward, just a logger
        return {sid: 0.0 for sid in new_data_dict.keys()}

duration = 5700.0 * 5  # 3 orbits #5700 -> 95 minutes
target_distribution = "uniform"
n_targets = 3000 #135 targets for 3 orbits- but should we just keep 3000 if MJ is able to get good results with that?
n_ahead = 32 # 5 n_ahead for 3 orbits - also should we keep 32 if MJ is able to get good results with that?

if target_distribution == "uniform":
    targets = scene.UniformTargets(n_targets)
elif target_distribution == "cities":
    targets = scene.CityTargets(n_targets)

def get_env():
    env = SatelliteTasking(
        satellite=power_sat_generator(n_ahead=5, include_time=False)(
            name="EO1-power",
            sat_args=SAT_ARGS_POWER,
        ),
        scenario=targets,
        rewarder=(
            _LogLastAction(),
            data.UniqueImageStore(),  # tracks first images
            ImageOnceReward(),        # +0.1*(1/p) on first image
            DownlinkOnceStore(),      # tracks first full downlinks
            DownlinkOnceReward(),     # +1*(1/p) on first full downlink
        ), 
        sim_rate=1.0,
        max_step_duration=300.0,          # paper uses 6-minute modes
        time_limit=duration,
        failure_penalty=-10.0,            # paper's failure penalty
        terminate_on_time_limit=True,
        log_level="ERROR",
        # Make sure downlink passes exist:
        world_type=world.GroundStationWorldModel,
        world_args=world.GroundStationWorldModel.default_world_args(),
    )
    env.user_state = {"last_action": None, "force_charge": False, "force_desat": False}
    _register_ground_tracking(env)

    return env