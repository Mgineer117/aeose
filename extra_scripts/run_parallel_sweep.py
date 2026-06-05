import sys
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True, help="Sweep ID to run")
    parser.add_argument("--agents", type=int, default=4, help="Number of parallel agents to spawn")
    parser.add_argument("--count", type=int, default=10, help="Trials per agent")
    parser.add_argument("--project", type=str, default="AEOS-SWEEP", help="WandB project")
    args = parser.parse_args()

    print(f"Spawning {args.agents} parallel wandb agents for sweep {args.sweep_id}...")
    
    processes = []
    for i in range(args.agents):
        cmd = [
            sys.executable, "search.py", 
            "--sweep_id", args.sweep_id, 
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
