import datetime
import os
import random
import uuid

import torch
import torch.nn as nn

import wandb
from get_env import get_env
from utils.functions import concat_csv_columnwise_and_delete, seed_all, setup_logger
from utils.get_args import get_args


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    env = get_env()
    env.max_steps = 100
    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.n
    args.episode_len = env.max_steps
    args.is_discrete = env.action_space.__class__.__name__ == "Discrete"

    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    # run algorithm
    if args.algo_name == "ppo":
        from algorithms.ppo import PPO_Algorithm

        algo = PPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    else:
        raise NotImplementedError(f"{args.algo_name} is not implemented.")

    algo.begin_training()

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    wandb.finish()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

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

        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
