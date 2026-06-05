import sys
import subprocess
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep ID to run (optional, creates new if empty)")
    parser.add_argument("--agents", type=int, default=4, help="Number of parallel agents to spawn")
    parser.add_argument("--count", type=int, default=10, help="Trials per agent")
    parser.add_argument("--project", type=str, default="AEOS-SWEEP", help="WandB project")
    args = parser.parse_args()

    sweep_id = args.sweep_id
    
    if not sweep_id:
        print(f"No sweep_id provided. Initializing a new sweep in project '{args.project}'...")
        # Run search.py with --count 0 to just create the sweep and exit
        result = subprocess.run(
            [sys.executable, "search.py", "--count", "0", "--project", args.project], 
            capture_output=True, 
            text=True
        )
        
        match = re.search(r"Created NEW wandb sweep with ID:\s+([a-zA-Z0-9]+)", result.stdout)
        if match:
            sweep_id = match.group(1)
            print(f"Successfully created new sweep with ID: {sweep_id}\n")
        else:
            print("Failed to automatically generate sweep ID. Output from search.py was:\n")
            print(result.stdout)
            if result.stderr:
                print("ERRORS:\n", result.stderr)
            sys.exit(1)

    print(f"Spawning {args.agents} parallel wandb agents for sweep {sweep_id}...")
    
    processes = []
    for i in range(args.agents):
        cmd = [
            sys.executable, "search.py", 
            "--sweep_id", sweep_id, 
            "--count", str(args.count),
            "--project", args.project
        ]
        # Start the process without blocking
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Started Agent {i+1} (PID: {p.pid})")

    print(f"\nAll {args.agents} agents are now running in parallel in the background.")
    print("Press Ctrl+C to stop all agents.\n")

    try:
        # Wait for all to finish
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all agents...")
        for p in processes:
            p.terminate()
        print("All agents stopped.")
