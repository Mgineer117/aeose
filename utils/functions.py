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


def setup_logger(args, unique_id, exp_time, seed):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join((args.algo_name, unique_id, "seed:" + str(seed)))

    if args.project is None:
        args.project = args.task

    args.logdir = os.path.join(args.logdir, args.group)

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
    )
    logger.save_config(default_cfg, verbose=True)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.mkdir(tensorboard_path)
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
