import os
import sys

# Add the bsk_rl source code to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "bsk_rl", "src")))

from utils.get_env import get_env

def scan_windows():
    env = get_env("downlink")
    env.reset(seed=0)
    sat = env.satellite
    
    # Calculate all windows for the entire simulation time
    sat.calculate_additional_windows(17100.0)
    
    print("=== TARGET WINDOWS ===")
    for opp in sat.opportunities:
        if opp["type"] == "target":
            obj = opp["object"]
            w = opp["window"]
            print(f"Target: {obj.name:<8} | Priority: {obj.priority:.1f} | Window: [{w[0]:.1f}s, {w[1]:.1f}s]")
            
    print("\n=== GROUND STATION WINDOWS ===")
    for opp in sat.opportunities:
        if opp["type"] == "ground_station":
            obj = opp["object"]
            w = opp["window"]
            gs_name = obj.ModelTag.replace("GroundStation", "") if not isinstance(obj, str) else obj
            print(f"Station: {gs_name:<8} | Window: [{w[0]:.1f}s, {w[1]:.1f}s]")

if __name__ == "__main__":
    scan_windows()
