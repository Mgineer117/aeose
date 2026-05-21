import os
import sys
import numpy as np

# Add the bsk_rl source code to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "bsk_rl", "src")))

from utils.get_env import get_env

def run_debug():
    print("Initializing downlink env...")
    env = get_env("downlink")
    obs, info = env.reset(seed=0)
    
    sat = env.satellite
    print("Initial storage level:", sat.dynamics.storage_level)
    msg = sat.dynamics.storageUnit.storageUnitDataOutMsg.read()
    print("Initial storedData:", msg.storedData)
    print("Initial storedData buffer 0:", msg.storedData[0])
    
    # Run steps
    for step in range(10):
        print(f"\n--- Step {step} ---")
        opps = sat.upcoming_opportunities
        print(f"Upcoming opps: {[{'type': o['type'], 'object': o['object'].name if hasattr(o['object'], 'name') else str(o['object']), 'window': o['window']} for o in opps[:3]]}")
        
        # Let's find an action that images a target
        action_idx = None
        for act_desc in sat.action_builder.action_spec:
            print("Action spec:", act_desc.name, "n_actions:", act_desc.n_actions)
        
        # Action spec for downlink env is:
        # action_spec = [
        #     act.Charge(),
        #     act.Desat(),
        #     act.Downlink(),
        #     act.Image(n_ahead_image=3),
        # ]
        # Gym actions map:
        # 0: Charge
        # 1: Desat
        # 2: Downlink
        # 3, 4, 5: Image target 0, 1, 2
        
        # Let's just try imaging target 0 (action index 3)
        action_idx = 3
        print(f"Taking action: {action_idx}")
        obs, reward, terminated, truncated, info = env.step(action_idx)
        
        print("After step storage level:", sat.dynamics.storage_level)
        msg = sat.dynamics.storageUnit.storageUnitDataOutMsg.read()
        print("After step storedData array:", msg.storedData)
        print("After step storedData[0]:", msg.storedData[0])
        print("Reward:", reward)
        print("Terminated:", terminated, "Truncated:", truncated)
        if terminated:
            break

if __name__ == "__main__":
    run_debug()
