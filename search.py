import datetime
import random
import uuid
import wandb
import torch

from utils.get_args import get_args
from main import run

def train():
    # wandb.agent automatically calls wandb.init() behind the scenes,
    # but calling it explicitly allows us to access wandb.config locally.
    wandb.init()
    config = wandb.config
    
    # Get standard args
    args = get_args()
    
    # Override args with sweep config
    if "target_kl" in config:
        args.target_kl = config.target_kl
    if "learning_rate" in config:
        args.actor_lr = config.learning_rate
        args.critic_lr = config.learning_rate
    if "gae" in config:
        args.gae = config.gae
    if "entropy_scaler" in config:
        args.entropy_scaler = config.entropy_scaler
        
    # Setup for the trial run
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")
    
    # Pick a random seed for this trial
    seed = random.randint(1, 10000)
    args.seed = seed
    
    # We set num_runs to 1 as we only want one execution per sweep combination
    args.num_runs = 1
    
    print(f"-------------------------------------------------------")
    print(f"      Sweep Trial ID: {unique_id}")
    print(f"      Seed: {seed}")
    print(f"      Time Begun: {exp_time}")
    print(f"-------------------------------------------------------")
    
    # Run training
    run(args, seed, unique_id, exp_time, args.run_id)

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Sweep Launcher")
    parser.add_argument("--sweep_id", type=str, default=None, help="WandB sweep ID to join")
    parser.add_argument("--count", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--project", type=str, default="AEOS-SWEEP", help="WandB project name")
    
    # Parse only known search-specific args
    search_args, remaining_args = parser.parse_known_args()
    
    # Overwrite sys.argv so get_args() in train() doesn't fail on unknown args
    sys.argv = [sys.argv[0]] + remaining_args

    torch.set_default_dtype(torch.float32)

    if search_args.sweep_id is None:
        # Define the hyperparameter search space
        sweep_config = {
            "method": "bayes",  # Supports "bayes", "random", or "grid"
            "metric": {
                "name": "max_eval_return",
                "goal": "maximize"
            },
            "parameters": {
                "target_kl": {
                    "min": 0.003,
                    "max": 0.03
                },
                "learning_rate": {
                    "min": 1e-5,
                    "max": 1e-3,
                    # "distribution": "log_uniform_values"
                },
                "gae": {
                    "min": 0.8,
                    "max": 1.0
                },
                "entropy_scaler": {
                    "min": 1e-4,
                    "max": 1e-1,
                    # "distribution": "log_uniform_values"
                }
            }
        }
        
        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=search_args.project) 
        print(f"\n=======================================================")
        print(f"Created NEW wandb sweep with ID: {sweep_id}")
        print(f"To run additional agents in parallel, run:")
        print(f"python search.py --sweep_id {sweep_id}")
        print(f"=======================================================\n")
        
        if search_args.count == 0:
            sys.exit(0)
    else:
        sweep_id = search_args.sweep_id
        print(f"\nJoining EXISTING wandb sweep with ID: {sweep_id}\n")
    
    print(f"Starting wandb agent for sweep {sweep_id}")
    # Count controls how many trials this specific agent will run
    wandb.agent(sweep_id, train, count=search_args.count, project=search_args.project) 
