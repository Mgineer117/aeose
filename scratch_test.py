#!/usr/bin/env python3
"""
Interactive simulation tool for BSK-RL Satellite Tasking environments.
Allows a human user to act as the RL agent by selecting actions step-by-step
and viewing the human-interpretable state, action outcomes, and rewards.
"""

import os
import sys
import numpy as np

# Add the bsk_rl source code to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "bsk_rl", "src")))

from utils.get_env import get_env

# ANSI color codes for premium aesthetic
C_HEADER = "\033[1;36m"    # Bold Cyan
C_SECTION = "\033[1;34m"   # Bold Blue
C_ACCENT = "\033[1;35m"    # Bold Magenta
C_GREEN = "\033[32m"       # Green
C_YELLOW = "\033[33m"      # Yellow
C_RED = "\033[31m"         # Red
C_BOLD = "\033[1m"         # Bold
C_RESET = "\033[0m"        # Reset

def make_progress_bar(fraction, length=20, fill_char="█", empty_char="░"):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * length))
    bar = fill_char * filled + empty_char * (length - filled)
    percent = fraction * 100
    
    # Color the progress bar based on fraction if desired
    if fraction > 0.8:
        color = C_GREEN
    elif fraction > 0.3:
        color = C_YELLOW
    else:
        color = C_RED
    return f"[{color}{bar}{C_RESET}] {percent:.1f}%"

def get_eclipse_status(shadow_factor):
    if shadow_factor >= 0.99:
        return f"{C_GREEN}Sunlight{C_RESET}"
    elif shadow_factor <= 0.01:
        return f"{C_RED}Eclipse (Shadow){C_RESET}"
    else:
        return f"{C_YELLOW}Penumbra (Shadow Factor: {shadow_factor:.2f}){C_RESET}"

def get_action_descriptions(sat):
    """Dynamic action mapping based on satellite action specification."""
    try:
        # Find next 3 targets to map to Image actions
        targets = sat.find_next_opportunities(n=3, types="target")
    except Exception:
        targets = []
        
    descriptions = []
    index = 0
    for act in sat.action_builder.action_spec:
        if act.name == "action_image":
            for sub_idx in range(act.n_actions):
                gym_action = index + sub_idx
                if sub_idx < len(targets):
                    tgt = targets[sub_idx]["object"]
                    start_t, end_t = targets[sub_idx]["window"]
                    sim_time = sat.simulator.sim_time
                    is_open = start_t <= sim_time <= end_t
                    window_str = f"[{start_t:.1f}s, {end_t:.1f}s]"
                    if is_open:
                        status_str = f"{C_GREEN}OPEN{C_RESET}"
                    else:
                        status_str = f"opens in {start_t - sim_time:.1f}s"
                    descriptions.append(
                        (gym_action, f"Image target {C_BOLD}{tgt.name}{C_RESET} (Priority {tgt.priority:.1f}) - Window: {window_str} ({status_str})")
                    )
                else:
                    descriptions.append(
                        (gym_action, "Image (No target opportunity available)")
                    )
            index += act.n_actions
        elif act.name == "action_charge":
            descriptions.append((index, "Charge battery (Point solar panels to Sun)"))
            index += 1
        elif act.name == "action_desat":
            descriptions.append((index, "Desaturate reaction wheels (Fire thrusters to dump momentum)"))
            index += 1
        elif act.name == "action_downlink":
            descriptions.append((index, "Downlink data (Point antenna to ground station to transmit image buffer)"))
            index += 1
        else:
            descriptions.append((index, f"{act.name}"))
            index += 1
            
    return descriptions

def display_dashboard(env, step_num, cum_reward, last_reward=None, last_action_desc=None):
    sat = env.satellite
    sim_time = env.simulator.sim_time
    time_limit = env.time_limit
    
    time_frac = sim_time / time_limit if time_limit > 0 else 0.0
    orbit_frac = sim_time / 5700.0  # 1 orbit = 5700s
    
    print("\n" + "=" * 80)
    print(f"{C_HEADER}--- STATE INFO: STEP {step_num} (Time: {sim_time:.1f}s / {time_limit:.1f}s, Orbit: {orbit_frac:.2f}) ---{C_RESET}")
    print("=" * 80)
    
    # 1. Environment and Time Status
    print(f"{C_SECTION}Simulation Environment Status:{C_RESET}")
    print(f"  Step Number:       {C_BOLD}{step_num}{C_RESET}")
    print(f"  Mission Time Bar:  {make_progress_bar(time_frac, length=30, fill_char='▰', empty_char='▱')}")
    print(f"  Cumulative Reward: {C_GREEN}{cum_reward:.4f}{C_RESET}")
    if last_reward is not None:
        print(f"  Last Step Action:  {last_action_desc}")
        print(f"  Last Step Reward:  {C_GREEN if last_reward >= 0 else C_RED}{last_reward:+.4f}{C_RESET}")
    print("-" * 80)
    
    # 2. Battery Status
    try:
        charge = sat.dynamics.battery_charge
        capacity = sat.dynamics.powerMonitor.storageCapacity
        # Convert W*s to W-hr
        charge_wh = charge / 3600.0
        capacity_wh = capacity / 3600.0
        fraction = charge / capacity if capacity > 0 else 0.0
        
        # Color specific formatting
        bar_str = make_progress_bar(fraction, length=20)
        print(f"{C_SECTION}Battery Power System:{C_RESET}")
        print(f"  Charge Level:      {bar_str} ({charge_wh:.2f} / {capacity_wh:.2f} W-hr)")
    except Exception as e:
        print(f"{C_SECTION}Battery Power System:{C_RESET} {C_RED}Error reading battery: {e}{C_RESET}")
    
    # 3. Data Storage
    try:
        storage = sat.dynamics.storage_level
        capacity = sat.dynamics.storageUnit.storageCapacity
        # Convert bits to Megabits (Mb)
        storage_mb = storage / 1e6
        capacity_mb = capacity / 1e6
        fraction = storage / capacity if capacity > 0 else 0.0
        
        # In storage, high fraction is red (buffer full), low fraction is green
        filled = int(round(fraction * 20))
        bar = "█" * filled + "░" * (20 - filled)
        if fraction > 0.8:
            bar_color = C_RED
        elif fraction > 0.5:
            bar_color = C_YELLOW
        else:
            bar_color = C_GREEN
        bar_str = f"[{bar_color}{bar}{C_RESET}] {fraction*100:.1f}%"
        
        print(f"{C_SECTION}Onboard Data Storage Buffer:{C_RESET}")
        print(f"  Storage Level:     {bar_str} ({storage_mb:.2f} / {capacity_mb:.2f} Mb)")
    except Exception as e:
        print(f"{C_SECTION}Onboard Data Storage Buffer:{C_RESET} {C_RED}Error reading storage: {e}{C_RESET}")
        
    # 4. Reaction Wheels
    try:
        speeds_rad = sat.dynamics.wheel_speeds
        speeds_rpm = speeds_rad * 30.0 / np.pi
        max_speed = sat.dynamics.maxWheelSpeed
        
        print(f"{C_SECTION}Reaction Wheels (Attitude Control):{C_RESET}")
        if max_speed == np.inf or max_speed <= 0:
            print(f"  Speeds (RPM):      RW1: {speeds_rpm[0]:.1f} | RW2: {speeds_rpm[1]:.1f} | RW3: {speeds_rpm[2]:.1f} (Limit: N/A)")
        else:
            fractions = speeds_rpm / max_speed
            rw_lines = []
            for i, (rpm, frac) in enumerate(zip(speeds_rpm, fractions)):
                filled = int(round(abs(frac) * 10))
                bar_char = "█"
                bar = bar_char * filled + "░" * (10 - filled)
                sign_str = "+" if frac >= 0 else "-"
                
                if abs(frac) > 0.8:
                    color = C_RED
                elif abs(frac) > 0.5:
                    color = C_YELLOW
                else:
                    color = C_GREEN
                    
                rw_lines.append(f"RW{i+1}: {sign_str}[{color}{bar}{C_RESET}] {abs(frac)*100:.1f}% ({rpm:+.1f} RPM)")
            print(f"  Speeds Fraction:   " + "  ".join(rw_lines))
            print(f"  Maximum Speed:     {max_speed:.1f} RPM")
    except Exception as e:
        print(f"{C_SECTION}Reaction Wheels (Attitude Control):{C_RESET} {C_RED}Error reading wheels: {e}{C_RESET}")
        
    # 5. Celestial and Contact Status
    print(f"{C_SECTION}Environmental External Conditions:{C_RESET}")
    # Eclipse
    try:
        shadow_factor = sat.dynamics.world.eclipseObject.eclipseOutMsgs[sat.dynamics.eclipse_index].read().shadowFactor
        print(f"  Solar Visibility:  {get_eclipse_status(shadow_factor)}")
    except Exception:
        print("  Solar Visibility:  N/A")
        
    # Ground Station Contact
    visible_gs = []
    if hasattr(env.simulator.world, "groundStations"):
        for gs in env.simulator.world.groundStations:
            if len(gs.accessOutMsgs) > 0 and gs.accessOutMsgs[-1].read().hasAccess:
                visible_gs.append(gs.ModelTag.replace("GroundStation", ""))
    if visible_gs:
        print(f"  Ground Stations:   {C_GREEN}In contact with: {', '.join(visible_gs)}{C_RESET}")
    else:
        print(f"  Ground Stations:   {C_YELLOW}No ground station in view{C_RESET}")
        
    print("-" * 80)
    
    # 6. Targets and Opportunities
    print(f"{C_SECTION}Upcoming Mission Opportunities:{C_RESET}")
    try:
        # Get targets
        opps = [opp for opp in sat.upcoming_opportunities if opp["type"] == "target"]
        if opps:
            for opp in opps[:3]:
                tgt = opp["object"]
                start_t, end_t = opp["window"]
                is_open = start_t <= sim_time <= end_t
                status = f"{C_GREEN}[OPEN FOR IMAGING]{C_RESET}" if is_open else f"Opens in {start_t - sim_time:.1f}s"
                if sim_time > end_t:
                    status = f"{C_RED}[PASSED]{C_RESET}"
                print(f"  • Target: {C_BOLD}{tgt.name:<12}{C_RESET} | Priority: {tgt.priority:.1f} | Window: [{start_t:.1f}s, {end_t:.1f}s] ({status})")
        else:
            print("  No target opportunities remaining.")
            
        # Get ground stations (if downlink env)
        gs_opps = [opp for opp in sat.upcoming_opportunities if opp["type"] == "ground_station"]
        if gs_opps:
            print()
            for opp in gs_opps[:2]:
                gs = opp["object"]
                start_t, end_t = opp["window"]
                is_open = start_t <= sim_time <= end_t
                status = f"{C_GREEN}[IN CONTACT]{C_RESET}" if is_open else f"In view in {start_t - sim_time:.1f}s"
                if sim_time > end_t:
                    status = f"{C_RED}[PASSED]{C_RESET}"
                if isinstance(gs, str):
                    gs_name = gs.replace("GroundStation", "")
                else:
                    gs_name = gs.ModelTag.replace("GroundStation", "")
                print(f"  • Station: {C_BOLD}{gs_name:<11}{C_RESET} | Contact Window: [{start_t:.1f}s, {end_t:.1f}s] ({status})")
    except Exception as e:
        print(f"  Error reading opportunities: {e}")
        
    print("=" * 80)
    
    # 7. Warning Alerts
    alerts = []
    # Battery low alert
    try:
        if sat.dynamics.battery_charge_fraction < 0.2:
            alerts.append(f"{C_RED}{C_BOLD}▲ WARNING: CRITICAL BATTERY POWER ({sat.dynamics.battery_charge_fraction*100:.1f}%){C_RESET}")
    except Exception:
        pass
    # Wheel speed alert
    try:
        if max_speed != np.inf and max_speed > 0:
            fractions = abs(sat.dynamics.wheel_speeds_fraction)
            if any(fractions > 0.85):
                alerts.append(f"{C_RED}{C_BOLD}▲ WARNING: REACTION WHEELS APPROACHING SATURATION LIMIT (Max: {max(fractions)*100:.1f}%){C_RESET}")
            elif any(fractions > 0.7):
                alerts.append(f"{C_YELLOW}▲ CAUTION: High reaction wheel speeds (Max: {max(fractions)*100:.1f}%){C_RESET}")
    except Exception:
        pass
    # Storage buffer full alert
    try:
        if sat.dynamics.storage_level_fraction >= 0.8:
            alerts.append(f"{C_RED}{C_BOLD}▲ WARNING: ONBOARD BUFFER ALMOST FULL ({sat.dynamics.storage_level_fraction*100:.1f}%) - INCOMING IMAGES WILL CAUSE CRITICAL FAILURE{C_RESET}")
    except Exception:
        pass
        
    if alerts:
        print(f"{C_BOLD}ALERTS & NOTIFICATIONS:{C_RESET}")
        for alert in alerts:
            print(f"  {alert}")
        print("=" * 80)

def main():
    while True:
        os.system("clear" if os.name == "posix" else "")
        print("=" * 80)
        print(f"{C_HEADER}   ___  ___________ _____   ___            _                  _ ")
        print("  / _ \\ |  ___|  _  /  ___| / _ \\          | |                | |")
        print(" / /_\\ \\| |__ | | | \\ `--. / /_\\ \\__ _  ___| |__   ___  _ __  | |")
        print(" |  _  ||  __|| | | |`--. \\|  _  / _` |/ _ \\ '_ \\ / _ \\| '_ \\ | |")
        print(" | | | || |___\\ \\_/ /\\__/ /| | | (_| |  __/ |_) | (_) | | | ||_|")
        print(f" \\_| |_/\\____/ \\___/\\____/ \\_| |_/\\__, |\\___|_.__/ \\___/|_| |_/(_)")
        print(f"                                   __/ |                        ")
        print(f"                                  |___/                         {C_RESET}")
        print("========================================================================")
        print(f"           {C_BOLD}Interactive Agent Simulation for Satellite Tasking{C_RESET}")
        print("========================================================================")
        print("Select the tasking environment to run:")
        print(f"  {C_BOLD}1{C_RESET}. charge    - Image targets; recharge battery when low.")
        print(f"  {C_BOLD}2{C_RESET}. desat     - Image, charge, and desaturate wheels before saturation.")
        print(f"  {C_BOLD}3{C_RESET}. downlink  - Downlink data to ground stations to avoid buffer overflow.")
        print(f"  {C_BOLD}4{C_RESET}. resource  - Balance resource level and imaging targets for reward.")
        print(f"  {C_BOLD}q{C_RESET}. Exit Program")
        print("=" * 80)
        
        choice = input("Enter choice (1-4 or q): ").strip().lower()
        if choice == "q":
            print("\nExiting interactive simulation. Goodbye!")
            sys.exit(0)
            
        env_map = {
            "1": "charge",
            "2": "desat",
            "3": "downlink",
            "4": "resource"
        }
        
        if choice not in env_map:
            input(f"\n{C_RED}Invalid option.{C_RESET} Press Enter to try again.")
            continue
            
        env_name = env_map[choice]
        print(f"\nInitializing {C_BOLD}{env_name}{C_RESET} environment (this may take a few seconds)...")
        
        try:
            env = get_env(env_name)
            obs, info = env.reset(seed=0)
        except Exception as e:
            input(f"\n{C_RED}Failed to initialize environment: {e}{C_RESET}\nPress Enter to return to menu.")
            continue
            
        step_num = 0
        cum_reward = 0.0
        last_reward = None
        last_action_desc = None
        
        # Main step loop
        try:
            while True:
                sat = env.satellite
                action_descs = get_action_descriptions(sat)
                
                # Render dashboard (State Info)
                display_dashboard(env, step_num, cum_reward, last_reward, last_action_desc)
                
                # Output Action choices
                print(f"{C_SECTION}Action Control Interface:{C_RESET}")
                for idx, desc in action_descs:
                    print(f"  [{C_BOLD}{idx}{C_RESET}] {desc}")
                print(f"  [{C_BOLD}q{C_RESET}] Quit simulation (return to environment menu)")
                print("-" * 80)
                
                action_input = input(f"Select action index (0-{len(action_descs)-1} or q): ").strip().lower()
                
                if action_input == "q":
                    print("\nQuitting current environment...")
                    env.close()
                    break
                    
                # Parse action index
                try:
                    action_idx = int(action_input)
                    if action_idx < 0 or action_idx >= len(action_descs):
                        raise ValueError()
                except ValueError:
                    print(f"\n{C_RED}Invalid input!{C_RESET} Please choose an integer action from the menu or 'q'.")
                    input("Press Enter to continue...")
                    continue
                    
                # Look up chosen action description for the dashboard next turn
                chosen_desc = next(desc for idx, desc in action_descs if idx == action_idx)
                
                # Step the simulation
                print(f"\n{C_ACCENT}>>> Transitioning state with action {action_idx}: {chosen_desc}...{C_RESET}")
                obs, reward, terminated, truncated, info = env.step(action_idx)
                
                step_num += 1
                last_reward = reward
                cum_reward += reward
                last_action_desc = f"{C_BOLD}{chosen_desc}{C_RESET}"
                
                print(f"{C_GREEN}>>> Transition complete. Step Reward: {reward:+.4f} | Cumulative Reward: {cum_reward:.4f}{C_RESET}")
                
                # Handle end of episode
                if terminated or truncated:
                    # Final render of dashboard before ending
                    display_dashboard(env, step_num, cum_reward, last_reward, last_action_desc)
                    
                    if terminated:
                        print(f"\n{C_RED}{C_BOLD}--- ENVIRONMENT TERMINATED ---{C_RESET}")
                        dyn = sat.dynamics
                        # Diagnosing termination reason
                        reasons = []
                        if hasattr(dyn, "battery_valid") and not dyn.battery_valid():
                            reasons.append("Battery depleted (charge <= 0)")
                        if hasattr(dyn, "rw_speeds_valid") and not dyn.rw_speeds_valid():
                            reasons.append("Reaction wheel speed exceeded the maximum limit")
                        if hasattr(dyn, "storage_level_fraction") and dyn.storage_level_fraction >= 0.98:
                            reasons.append("Onboard data storage buffer overflow (>= 98% full)")
                        if hasattr(dyn, "altitude_valid") and not dyn.altitude_valid():
                            reasons.append("Altitude invalid (spacecraft has deorbit)")
                        
                        if reasons:
                            print(f"Reason(s): {', '.join(reasons)}")
                        else:
                            print("Reason: Satellite aliveness checker failed or critical boundary violated.")
                            
                    elif truncated:
                        print(f"\n{C_GREEN}{C_BOLD}--- ENVIRONMENT TRUNCATED ---{C_RESET}")
                        print(f"Reason: Successfully reached simulation time limit of {env.time_limit:.1f}s.")
                        
                    print(f"Final Episode Reward: {C_GREEN if cum_reward >= 0 else C_RED}{cum_reward:.4f}{C_RESET}")
                    input("\nPress Enter to return to main menu...")
                    env.close()
                    break
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
            try:
                env.close()
            except Exception:
                pass
            input("Press Enter to return to main menu...")
        except Exception as e:
            print(f"\n{C_RED}An error occurred during simulation step: {e}{C_RESET}")
            import traceback
            traceback.print_exc()
            try:
                env.close()
            except Exception:
                pass
            input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    main()
