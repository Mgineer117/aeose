import json
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from log.wandb_logger import WandbLogger
from utils.wrapper import FetchWrapper, ObsNormWrapper, PointMazeWrapper


def setup_logger(args, unique_id, exp_time, seed, wandb_id=None):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Group/log dir must be deterministic across cluster job chunks so a
    # resumed job finds the previous chunk's latest_model.pth. Keying it on
    # exp_time (a fresh timestamp each launch) would put every chunk in a new
    # folder and silently disable checkpoint resume, so we key on unique_id
    # (env + architecture) only.
    if args.group is None:
        args.group = unique_id

    if args.name is None:
        # Fold the outer seed group (CLI --seed) into the name so two launches
        # whose derived per-run seeds happen to coincide don't share a log dir.
        cli_seed = getattr(args, "cli_seed", None)
        seed_tag = f"g{cli_seed}_seed:{seed}" if cli_seed is not None else f"seed:{seed}"
        args.name = "-".join((args.algo_name, unique_id, seed_tag))

    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)
    args.unique_id = unique_id

    # W&B id resolution: if this run has an existing checkpoint, reuse the id
    # stored next to it so the resumed job continues the SAME W&B run. If there
    # is no checkpoint, start a fresh *random* run (wandb_id=None -> WandbLogger
    # generates a uuid). This avoids the deterministic-id pitfalls: a deleted
    # run can't be re-attached, and to start over you just delete the local
    # checkpoint folder instead of juggling seeds or deleting cloud runs.
    resume_state_path = os.path.join(args.logdir, args.name, "resume_state.json")
    resolved_wandb_id = None
    if os.path.exists(resume_state_path):
        try:
            with open(resume_state_path) as f:
                saved_state = json.load(f)
            # Prefer the stored id; fall back to the passed-in (legacy
            # deterministic) id for checkpoints written before ids were saved.
            resolved_wandb_id = saved_state.get("wandb_id") or wandb_id
        except Exception:
            resolved_wandb_id = wandb_id

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
        wandb_id=resolved_wandb_id,
    )
    logger.save_config(default_cfg, verbose=True)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    # exist_ok=True: on a resumed run log_dir (and tensorboard/) already exist.
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def temp_seed(seed, pid):
    """
    This saves current seed info and calls after stochastic action selection.
    -------------------------------------------------------------------------
    This is to introduce the stochacity in each multiprocessor.
    Without this, the samples from each multiprocessor will be same since the seed was fixed
    """
    rand_int = random.randint(0, 1_000_000)  # create a random integer

    # Set the temporary seed
    torch.manual_seed(seed + pid + rand_int)
    np.random.seed(seed + pid + rand_int)
    random.seed(seed + pid + rand_int)


def concat_csv_columnwise_and_delete(folder_path, output_file="output.csv"):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate column-wise (axis=1)
    combined_df = pd.concat(dataframes, axis=1)

    # Save to output file
    output_file = os.path.join(folder_path, output_file)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

    # Delete original CSV files
    for file in csv_files:
        os.remove(os.path.join(folder_path, file))

    print("Original CSV files deleted.")
