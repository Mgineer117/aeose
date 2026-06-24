import datetime
import os
import random
import uuid

import torch
import torch.nn as nn
import wandb
from utils.functions import concat_csv_columnwise_and_delete, seed_all, setup_logger
from utils.get_args import get_args
from utils.get_env import get_env


def run(args, seed, unique_id, exp_time, run_id, wandb_id=None):
    # fix seed
    seed_all(seed)

    # get env
    env = get_env(args.env_name)

    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.n
    args.episode_len = getattr(env, "max_steps", None)
    args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    logger, writer = setup_logger(args, unique_id, exp_time, seed, wandb_id=wandb_id)

    # run algorithm
    if args.algo_name == "ppo":
        from algorithms.ppo import PPO_Algorithm

        algo = PPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "ddpg":
        from algorithms.ddpg import DDPG_Algorithm

        algo = DDPG_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "pd":
        from algorithms.pd import PD_Algorithm

        algo = PD_Algorithm(
            env=env, logger=logger, writer=writer, args=args, run_id=run_id
        )
    elif args.algo_name == "mcts":
        from algorithms.mcts import MCTS_Algorithm

        algo = MCTS_Algorithm(env=env, logger=logger, writer=writer, args=args)
    else:
        raise NotImplementedError(f"{args.algo_name} is not implemented.")

    result = algo.begin_training()

    if result and "max_eval_return" in result:
        wandb.log({"max_eval_return": result["max_eval_return"]})

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    wandb.finish()

    return result


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # NOTE: the sampler runs in-process vectorized env batching; no global
    # multiprocessing start method is set here.

    init_args = get_args()
    
    # Generate a deterministic unique_id based on environment and architecture to allow resuming
    arch_str = "".join(map(str, init_args.actor_fc_dim))
    unique_id = f"{init_args.env_name}_{arch_str}"
    
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    # The CLI --seed is the *outer* seed group: it deterministically generates
    # the per-run seeds below. Two launches with different --seed can draw an
    # overlapping per-run seed, so the outer seed is folded into the run
    # identity (name + wandb_id) to keep concurrent runs from colliding on the
    # same checkpoint folder / W&B run.
    cli_seed = init_args.seed
    random.seed(cli_seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = get_args()
        args.seed = seed
        args.cli_seed = cli_seed  # outer seed group, for collision-free identity

        # Legacy deterministic wandb_id. Only used as a fallback when resuming a
        # checkpoint written before the run id was persisted; new runs get a
        # random id and resumes reuse the id saved in resume_state.json (see
        # setup_logger).
        wandb_id = f"aeos_{args.env_name}_{args.algo_name}_{arch_str}_g{cli_seed}_s{seed}"

        run(args, seed, unique_id, exp_time, args.run_id, wandb_id=wandb_id)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
