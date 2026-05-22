#!/usr/bin/env python3
"""
Verification script for unified reward function and failure logic across all environments.
"""

import os
import sys
import numpy as np

# Add the bsk_rl source code to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "bsk_rl", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.get_env import get_env
from envs.reward_utils import AgileEOSReward, TerminationGuard

def verify_env(env_name):
    print(f"\n==========================================")
    print(f"VERIFYING ENVIRONMENT: {env_name.upper()}")
    print(f"==========================================")
    
    # 1. Instantiate environment
    print(f"[*] Instantiating {env_name}...")
    env = get_env(env_name)
    
    # 2. Check failure_penalty
    print(f"[*] Checking failure penalty...")
    assert env.failure_penalty == -10.0, f"Expected failure penalty to be -10.0, got {env.failure_penalty}"
    print(f"  - failure_penalty: {env.failure_penalty} (OK)")
    
    # 3. Check rewarder types
    print(f"[*] Checking rewarder configuration...")
    rewarders = env.rewarder
    if type(rewarders).__name__ == "ComposedReward":
        rewarders = rewarders.rewarders
    elif not isinstance(rewarders, tuple):
        rewarders = (rewarders,)
    
    has_agile_reward = False
    has_termination_guard = False
    for r in rewarders:
        if isinstance(r, AgileEOSReward):
            has_agile_reward = True
        elif isinstance(r, TerminationGuard):
            has_termination_guard = True
            
    assert has_agile_reward, f"AgileEOSReward not found in {env_name} rewarders: {rewarders}"
    assert has_termination_guard, f"TerminationGuard not found in {env_name} rewarders: {rewarders}"
    print(f"  - Rewarders: {[type(r).__name__ for r in rewarders]} (OK)")
    
    # 5. Reset and check initial state
    print(f"[*] Resetting environment...")
    obs, info = env.reset(seed=0)
    print("  - Reset complete (OK)")
    
    # 4. Check SAT_ARGS_POWER values
    print(f"[*] Checking standardized SAT_ARGS_POWER attributes...")
    sat = env.satellite
    sat_args = sat.sat_args
    
    expected_args = {
        "batteryStorageCapacity": 150.0 * 3600,
        "maxWheelSpeed": 2000.0,
        "storageInit": 0,
        "transmitterBaudRate": -50 * 8e6,
        "transmitterPowerDraw": -25.0,
        "instrumentBaudRate": 5 * 8e6,
        "basePowerDraw": -10.0,
        "panelArea": 0.5,
    }
    
    for key, expected_val in expected_args.items():
        assert key in sat_args, f"Missing key {key} in satellite sat_args"
        val = sat_args[key]
        if callable(val):
            # If it's a lambda or function, we skip exact check or evaluate
            pass
        else:
            assert abs(val - expected_val) < 1e-6, f"For {key}, expected {expected_val}, got {val}"
    print("  - Satellite parameters standardized (OK)")
    
    # 6. Test is_alive override
    print(f"[*] Verifying is_alive() override and storage overflow detection...")
    assert sat.is_alive(log_failure=False), "Satellite should be alive initially"
    
    # Force storage_level_fraction >= 0.98 and check is_alive
    actual_class = sat.dynamics.__class__
    original_prop = actual_class.storage_level_fraction
    actual_class.storage_level_fraction = property(lambda self: 0.99)
    
    is_alive_res = sat.is_alive(log_failure=True)
    print(f"  - is_alive with storage fraction 0.99: {is_alive_res} (Expected: False)")
    assert not is_alive_res, "Satellite should be dead when storage fraction >= 0.98"
    
    # Check that termination guard is_terminated returns True
    for r in rewarders:
        if isinstance(r, TerminationGuard):
            assert r.is_terminated(sat), "TerminationGuard should flag termination when storage >= 0.98"
            
    # Restore fraction
    actual_class.storage_level_fraction = original_prop
    sat._is_alive = True  # reset cached flag
    assert sat.is_alive(log_failure=False), "Satellite should be alive again after restoring storage fraction"
    print("  - Aliveness and Termination checks for storage overflow (OK)")
    
    # 7. Take a step
    print(f"[*] Taking a test step...")
    # Just choose action 0 (usually Charge)
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  - Step complete. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated} (OK)")
    
    # Clean up environment to free resources
    env.close()
    print(f"[*] {env_name.upper()} VERIFICATION PASSED!\n")

def main():
    envs_to_test = ["charge", "desat", "downlink", "resource"]
    passed = []
    failed = []
    
    for name in envs_to_test:
        try:
            verify_env(name)
            passed.append(name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[!] Verification failed for {name}: {e}")
            failed.append(name)
            
    print("==========================================")
    print("VERIFICATION SUMMARY")
    print("==========================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
