# env_sanity.py
import numpy as np
from pprint import pprint
from desat_only_finalver import get_env

def _sat(env):
    return getattr(env, "satellite", None) or env.satellites[0]

def _finite(x):
    try:
        x = np.asarray(x, float)
        return np.all(np.isfinite(x))
    except Exception:
        return False

def _wheel_norm_rpm(sat):
    ws = np.array(getattr(sat.dynamics, "wheel_speeds", [0,0,0]), float)[:3]
    return float(np.linalg.norm(ws))

def _wheel_norm_normed(sat):
    lim = float(getattr(sat.dynamics, "maxWheelSpeed", 5000.0))
    return _wheel_norm_rpm(sat) / max(lim, 1.0)

def _battery_frac(sat):
    return float(getattr(sat.dynamics, "battery_charge_fraction", np.nan))

def _storage_frac(sat):
    lvl = getattr(sat, "storage_level_fraction", None)
    if lvl is not None:
        return float(lvl)
    msg = sat.dynamics.storageUnit.storageUnitDataOutMsg.read()
    used = float(getattr(msg, "storageLevel", 0.0))
    cap  = float(getattr(msg, "storageCapacity", 1.0))
    return used / cap if cap > 0 else 0.0

def smoke_test(steps=120):
    env = get_env()
    obs, info = env.reset()
    assert _finite(obs), "NaN/Inf in initial observation"
    R = 0.0
    done = trunc = False
    for k in range(steps):
        a = env.action_space.sample()
        obs, r, done, trunc, info = env.step(a)
        assert _finite(obs), f"NaN/Inf at step {k}"
        R += float(r)
        if done or trunc:
            break
    return {
        "finished": "done" if done else ("truncated" if trunc else "step_limit"),
        "reason": info.get("termination_reason", ""),
        "total_reward": round(R, 3)
    }

def action_probe(steps_per=12):
    """
    Run each discrete action (0,1,2) from fresh resets and measure deltas:
      Charge  -> Δbattery > 0
      Desat   -> Δ||wheel|| < 0
      Image   -> Δstorage > 0
    """
    env = get_env()
    out = []
    for aidx in range(3):
        obs, _ = env.reset()
        sat = _sat(env)
        b0, w0, s0 = _battery_frac(sat), _wheel_norm_rpm(sat), _storage_frac(sat)
        R = 0.0
        done = trunc = False
        for _ in range(steps_per):
            obs, r, done, trunc, info = env.step(int(aidx))
            R += float(r)
            if done or trunc:
                break
        sat = _sat(env)
        out.append({
            "action_idx": aidx,
            "Δbattery": round(_battery_frac(sat) - b0, 4),
            "Δwheel_rpm": round(_wheel_norm_rpm(sat) - w0, 1),
            "Δstorage": round(_storage_frac(sat) - s0, 4),
            "reward_sum": round(R, 3),
            "terminated": bool(done or trunc),
            "reason": info.get("termination_reason", "")
        })
    # Heuristic labels
    labels = {}
    for r in out:
        a = r["action_idx"]; db, dw, ds = r["Δbattery"], r["Δwheel_rpm"], r["Δstorage"]
        if ds > 0.01: labels[a] = "Image"
        elif dw < -5.0: labels[a] = "Desat"
        elif db > 0.01: labels[a] = "Charge"
        else: labels[a] = "Unclear"
    return out, labels

def desat_sequence():
    """
    Prove that: imaging spins wheels up, then desat reduces wheel norm.
    """
    env = get_env()
    obs, _ = env.reset()
    sat = _sat(env)

    # Map action indices by name if available; otherwise we infer later
    names = getattr(getattr(env, "action_builder", None), "actions", [])
    name_to_idx = {getattr(a, "name", str(i)): i for i, a in enumerate(names)} if names else {}

    # If we don't know which is which, infer quickly:
    probe, labels = action_probe(steps_per=6)
    idx_by_label = {v:k for k,v in labels.items() if v in ("Image","Desat")}
    a_img = idx_by_label.get("Image", name_to_idx.get("Image", 2))
    a_dst = idx_by_label.get("Desat", name_to_idx.get("Desat", 1))

    # Burst of imaging
    for _ in range(6):
        obs, r, done, trunc, info = env.step(int(a_img))
        if done or trunc: break
    w_after_img = _wheel_norm_rpm(_sat(env))

    # Desat until cool or stop
    for _ in range(24):
        obs, r, done, trunc, info = env.step(int(a_dst))
        if done or trunc: break
    w_after_desat = _wheel_norm_rpm(_sat(env))

    return {
        "wheel_after_imaging_rpm": round(w_after_img, 1),
        "wheel_after_desat_rpm": round(w_after_desat, 1),
        "desat_helped": bool(w_after_desat <= w_after_img + 1e-6)
    }

def invariants_check():
    """
    Quick invariants on normalized ranges & vectors that should hold most of the time.
    """
    env = get_env()
    obs, _ = env.reset()
    sat = _sat(env)
    results = {
        "battery_in_[0,1]": 0.0 - 1e-6 <= _battery_frac(sat) <= 1.0 + 1e-6,
        "storage_in_[0,1]": 0.0 - 1e-6 <= _storage_frac(sat) <= 1.0 + 1e-6,
        "wheel_normed_reasonable(<=1.2)": (_wheel_norm_normed(sat) <= 1.2),
    }
    return results

if __name__ == "__main__":
    np.random.seed(0)
    print("== SMOKE ==")
    pprint(smoke_test())

    print("\n== ACTION PROBE ==")
    out, labels = action_probe()
    pprint(out); print("labels:", labels)

    print("\n== DESAT SEQUENCE ==")
    pprint(desat_sequence())

    print("\n== INVARIANTS ==")
    pprint(invariants_check())
    print("\nSanity checks complete.")
