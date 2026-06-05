import argparse
import json
from copy import deepcopy

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.1),
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


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
        "--actor-lr", type=float, default=3e-4, help="Base learning rate."
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
        default=[1024, 1024],
    )
    parser.add_argument(
        "--target-actor-fc-dim", type=int, nargs="+", default=[1024, 1024]
    )
    parser.add_argument("--critic-fc-dim", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--activation",
        type=str,
        default="leakyrelu",
        choices=list(_ACTIVATIONS.keys()),
        help="Activation function used in the actor and critic MLPs.",
    )

    parser.add_argument(
        "--timesteps", type=int, default=int(1e7), help="Number of training epochs."
    )
    parser.add_argument(
        "--epochs", type=int, default=int(100000), help="Number of training epochs."
    )

    parser.add_argument(
        "--log-interval", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--num-minibatch", type=int, default=4, help="")
    parser.add_argument("--minibatch-size", type=int, default=1024, help="")
    parser.add_argument("--batch-size", type=int, default=512, help="")
    parser.add_argument("--K-epochs", type=int, default=10, help="")
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
        "--entropy-scaler", type=float, default=1e-2, help="Base learning rate."
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Base learning rate.")
    parser.add_argument(
        "--warmup-samples", type=int, default=1000, help="Base learning rate."
    )
    # ---- Sampler tuning (PPO) ----
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Legacy multiprocessing knob. The sampler now uses in-process vectorized envs.",
    )
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=0,
        help="Deprecated compatibility option. The vectorized sampler does not use workers.",
    )
    parser.add_argument(
        "--envs-per-worker",
        default=4,
        help="Number of envs batched together in the in-process vectorized sampler.",
    )
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default="forkserver",
        choices=["vectorized", "async_vectorized", "forkserver", "spawn"],
        help="Sampler backend: in-process serial vectorized, async vectorized workers, forkserver workers, or spawn workers.",
    )
    parser.add_argument(
        "--async-sampling",
        action="store_true",
        help="Overlap sampler rollouts with the SGD update by dispatching "
        "the next batch before policy.learn(). Introduces 1 step of policy "
        "staleness (rollout uses pre-update weights); safe under PPO's "
        "target_kl bound but a small deviation from strict on-policy.",
    )
    parser.add_argument(
        "--first-round-timeout",
        type=int,
        default=7200,
        help="Seconds to wait for the first sampler batch before respawning "
        "workers. First round pays the Basilisk/SPICE import cost; raise on "
        "cluster runs with cold NFS-backed SPICE kernels.",
    )
    parser.add_argument(
        "--steady-timeout",
        type=int,
        default=2400,
        help="Seconds to wait for each subsequent sampler batch.",
    )
    parser.add_argument(
        "--ppo-buffer-strategy",
        type=str,
        default="diverse_recent",
        choices=["fifo", "diverse_recent"],
        help="Replay-buffer retention policy used when PPO snapshots are saved.",
    )
    parser.add_argument(
        "--ppo-buffer-recent-fraction",
        type=float,
        default=0.35,
        help="Fraction of the PPO replay buffer reserved for the newest policy rollouts.",
    )
    parser.add_argument(
        "--ppo-buffer-diversity-weight",
        type=float,
        default=1.0,
        help="Weight assigned to novelty when PPO retains saved buffer samples.",
    )
    parser.add_argument(
        "--ppo-buffer-recency-weight",
        type=float,
        default=0.35,
        help="Weight assigned to recency when PPO retains saved buffer samples.",
    )
    parser.add_argument(
        "--ppo-buffer-random-swap-prob",
        type=float,
        default=0.02,
        help="Small probability of random replacement to avoid a stale diverse PPO archive.",
    )
    parser.add_argument(
        "--ppo-buffer-candidate-pool-size",
        type=int,
        default=256,
        help="Number of random stored transitions compared during PPO buffer replacement.",
    )
    parser.add_argument(
        "--ppo-buffer-recency-horizon",
        type=int,
        default=4096,
        help="Decay horizon, in insertions, for PPO replay-buffer recency bias.",
    )
    parser.add_argument(
        "--ppo-buffer-projection-dim",
        type=int,
        default=32,
        help="Random-projection dimension used to estimate PPO buffer diversity.",
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

    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=None,
        help=(
            "Optional cap on decision-interval steps per episode. "
            "If set, overrides the env-derived decision-step count (useful to "
            "force a 6-interval episode regardless of env time_limit)."
        ),
    )
    parser.add_argument(
        "--pd-buffer-mode",
        type=str,
        default="teacher",
        choices=["teacher", "student", "mixed"],
        help="Policy-distillation data source: teacher buffer only, student rollouts only, or a hybrid of both.",
    )
    parser.add_argument(
        "--pd-buffer-capacity",
        type=int,
        default=50000,
        help="Max number of teacher-side states retained for policy distillation.",
    )
    parser.add_argument(
        "--pd-student-buffer-capacity",
        type=int,
        default=50000,
        help="Max number of student-collected states retained for policy distillation.",
    )
    parser.add_argument(
        "--pd-batch-size",
        type=int,
        default=1024,
        help="Mini-batch size used by policy distillation.",
    )
    parser.add_argument(
        "--pd-student-rollout-steps",
        type=int,
        default=256,
        help="How many states to collect from the current student policy before each PD update.",
    )
    parser.add_argument(
        "--pd-student-rollout-deterministic",
        action="store_true",
        help="Collect student distillation states using greedy student actions instead of sampled actions.",
    )
    parser.add_argument(
        "--pd-mixed-student-ratio",
        type=float,
        default=0.5,
        help="In mixed PD mode, fraction of each batch drawn from student-side states.",
    )
    parser.add_argument(
        "--pd-mixed-update",
        type=str,
        default="dagger",
        choices=["random", "dagger"],
        help="How mixed PD mode merges new student states into the aggregate buffer.",
    )

    # ---- MCTS-specific arguments ----
    parser.add_argument(
        "--mcts-mode",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help=(
            "MCTS training mode: 'online' = tree search + learned critic online; "
            "'offline' = pure tree search data generation → supervised network training"
        ),
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo Tree Search simulations per decision.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="Exploration constant (UCB coefficient) for MCTS tree selection.",
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx, verbose=True)
    args.str_actor_fc_dim = str(tuple(args.actor_fc_dim))
    args.str_target_actor_fc_dim = str(tuple(args.target_actor_fc_dim))
    # Keep the string for logging; resolve to a fresh nn.Module instance at
    # the call site (each network gets its own instance).
    args.activation_name = args.activation
    args.activation = _ACTIVATIONS[args.activation]

    return args


def select_device(gpu_idx=0, verbose=False):
    device = torch.device("cpu")
    if torch.cuda.is_available() and gpu_idx is not None:
        device = torch.device("cuda:" + str(gpu_idx))
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    if verbose:
        print(
            "============================================================================================"
        )
        if device.type == "cuda":
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print(f"Device set to : {device.type}")
        print(
            "============================================================================================"
        )
    return device
