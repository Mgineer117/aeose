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


def run(args, seed, unique_id, exp_time, run_id):
    # fix seed
    seed_all(seed)

    # get env
    env = get_env(args.env_name)
    # bsk_rl envs end internally via `time_limit / max_step_duration` (≈95
    # decision steps for the 5-orbit scenarios). We give the sampler some
    # headroom above that as a safety-net so it never cuts the episode
    # before the env itself does — but no longer the 1000 we had, which
    # was 10× the real cap and just wasted preallocated buffer memory.
    env_time_limit = getattr(env, "time_limit", None)
    env_step_dur = getattr(env, "max_step_duration", None)
    if env_time_limit is not None and env_step_dur:
        env.max_steps = int(env_time_limit / env_step_dur) + 10
    else:
        env.max_steps = 1000
    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.n
    args.episode_len = env.max_steps
    args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    logger, writer = setup_logger(args, unique_id, exp_time, seed)

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
    else:
        raise NotImplementedError(f"{args.algo_name} is not implemented.")

    algo.begin_training()

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    wandb.finish()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # NOTE: the sampler runs in-process vectorized env batching; no global
    # multiprocessing start method is set here.

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = get_args()
        args.seed = seed

        run(args, seed, unique_id, exp_time, args.run_id)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
