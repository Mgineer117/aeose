import argparse
import json
from copy import deepcopy

import torch


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--env-name", type=str, default="charge", help="Environment name."
    )
    parser.add_argument("--algo-name", type=str, default="ppo", help="Algorithm name.")
    parser.add_argument("--seed", type=int, default=42, help="Batch size.")
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of samples for training."
    )
    parser.add_argument(
        "--run-id", type=int, default=0, help="Unique identifier for the run."
    )

    parser.add_argument(
        "--actor-lr", type=float, default=1e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4, help="Base learning rate."
    )
    parser.add_argument(
        "--eps-clip", type=float, default=0.2, help="Base learning rate."
    )
    parser.add_argument(
        "--actor-fc-dim",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument(
        "--target-actor-fc-dim", type=int, nargs="+", default=[1024, 1024]
    )
    parser.add_argument("--critic-fc-dim", type=int, nargs="+", default=[256, 256])

    parser.add_argument(
        "--timesteps", type=int, default=int(1e7), help="Number of training epochs."
    )
    parser.add_argument(
        "--epochs", type=int, default=int(100000), help="Number of training epochs."
    )

    parser.add_argument(
        "--log-interval", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--num-minibatch", type=int, default=4, help="")
    parser.add_argument("--minibatch-size", type=int, default=4096, help="")
    parser.add_argument("--batch-size", type=int, default=256, help="")
    parser.add_argument("--K-epochs", type=int, default=5, help="")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="Upper bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Lower bound of the eigenvalue of the dual metric.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Base learning rate.")
    parser.add_argument(
        "--warmup-samples", type=int, default=1000, help="Base learning rate."
    )
    # ---- Sampler tuning (PPO) ----
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Max sampler worker processes. 0 = auto cap (min(cpu_count-1, 8)).",
    )
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=0,
        help="Episodes each sampler worker runs per collect_samples call. "
        "0 = auto-derive from batch_size, total workers, and episode_len.",
    )
    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="Number of training epochs."
    )
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Path to a directory for storing the log.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Path to a directory for storing the log.",
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx, verbose=True)
    args.str_actor_fc_dim = str(tuple(args.actor_fc_dim))
    args.str_target_actor_fc_dim = str(tuple(args.target_actor_fc_dim))

    return args


def select_device(gpu_idx=0, verbose=False):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
