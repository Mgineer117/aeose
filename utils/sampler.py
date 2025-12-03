import random
import time
from datetime import date
from math import ceil, floor
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from utils.functions import temp_seed
from utils.get_env import get_env

today = date.today()


class Base:
    def __init__(self, **kwargs):
        """
        Base class for the sampler.
        """
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.episode_len = kwargs.get("episode_len")
        self.batch_size = kwargs.get("batch_size")

    def get_reset_data(self):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        np.nan makes it easy to debug
        """
        batch_size = (self.num_episodes_per_worker + 1) * self.episode_len
        data = dict(
            states=np.full(((batch_size,) + self.state_dim), np.nan, dtype=np.float32),
            next_states=np.full(
                ((batch_size,) + self.state_dim), np.nan, dtype=np.float32
            ),
            actions=np.full((batch_size, self.action_dim), np.nan, dtype=np.float32),
            rewards=np.full((batch_size, 1), np.nan, dtype=np.float32),
            terminals=np.full((batch_size, 1), np.nan, dtype=np.float32),
            logprobs=np.full((batch_size, 1), np.nan, dtype=np.float32),
            entropys=np.full((batch_size, 1), np.nan, dtype=np.float32),
        )
        return data


class OnlineSampler(Base):
    def __init__(
        self,
        env_name: str,
        state_dim: tuple,
        action_dim: int,
        episode_len: int,
        batch_size: int,
        verbose: bool = True,
    ) -> None:
        """
        Monte Carlo-based sampler for online RL training. Dynamically schedules
        worker processes based on CPU availability and the desired batch size.

        Each worker collects 2 trajectories per round. The class adjusts sampling
        load over multiple rounds when cores are insufficient.

        Args:
            state_dim (tuple): Shape of state space.
            action_dim (int): Dimensionality of action space.
            episode_len (int): Maximum episode length.
            batch_size (int): Desired sample batch size.
            cpu_preserve_rate (float): Fraction of CPU to keep free.
            num_cores (int | None): Override for max cores to use.
            verbose (bool): Whether to print initialization info.
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
        )

        self.env_name = env_name
        self.num_episodes_per_worker = 4  # fixed
        self.total_num_worker = ceil(
            batch_size / (self.num_episodes_per_worker * episode_len)
        )

        # self.envs = [get_env() for _ in range(self.total_num_worker)]

        if verbose:
            print("Sampling Parameters:")
            print(f"Total number of workers: {self.total_num_worker}")

        torch.set_num_threads(1)  # Avoid CPU oversubscription

    def collect_samples(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
        use_mp: bool = True,
    ):
        """
        Collect samples either in parallel (multiprocessing) or sequentially.

        Args:
            policy: Policy to sample actions from.
            seed (int | None): Seed for reproducibility.
            deterministic (bool): Whether to use deterministic policy.
            random_init_pos (bool): Randomize initial position in env reset.
            use_mp (bool): If True, use multiprocessing. If False, run sequentially.

        Returns:
            memory (dict): Sampled batch.
            duration (float): Time taken to collect.
        """
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))
        policy.to_device(torch.device("cpu"))

        memory = {}

        if use_mp:
            # === Multiprocessing path ===
            processes = []
            queue = mp.Queue()
            # barrier = mp.Barrier(self.total_num_worker)
            worker_memories = [None] * self.total_num_worker

            for i in range(self.total_num_worker):
                # args = (i, queue, barrier, policy, seed, deterministic)
                args = (i, queue, policy, seed, deterministic)
                p = mp.Process(target=self.collect_trajectory, args=args)
                processes.append(p)
                p.start()

            # ✅ Wait for just the subprocess workers of this round
            expected = len(processes)
            collected = 0
            retry_counts = {pid: 0 for pid in range(expected)}
            max_retries = 2
            while collected < expected:
                try:
                    pid, data = queue.get(timeout=300)
                    if worker_memories[pid] is None:
                        worker_memories[pid] = data
                        collected += 1
                        retry_counts[pid] = 0  # reset retry count
                except Empty:
                    print(
                        f"[Warning] Queue timeout. Retrying... ({collected}/{expected})"
                    )
                    missing = [
                        pid for pid in range(expected) if worker_memories[pid] is None
                    ]
                    for pid in missing:
                        retry_counts[pid] += 1
                        if retry_counts[pid] >= max_retries:
                            print(f"[Error] Worker {pid} did not respond. Skipping.")
                            worker_memories[pid] = None
                            collected += 1

            # ✅ Cleanup processes
            start_time = time.time()
            for p in processes:
                p.join(timeout=max(0.1, 10 - (time.time() - start_time)))
                if p.is_alive():
                    p.terminate()
                    p.join()

            # ✅ Merge memory from workers
            for wm in worker_memories:
                if wm is not None:
                    for key, val in wm.items():
                        if key in memory:
                            memory[key] = np.concatenate((memory[key], val), axis=0)
                        else:
                            memory[key] = val

        else:
            # === Single-process path ===
            worker_memories = []
            for i in range(self.total_num_worker):
                wm = self.collect_trajectory(i, None, policy, seed, deterministic)
                worker_memories.append(wm)

            for wm in worker_memories:
                if wm is not None:
                    for key, val in wm.items():
                        if key in memory:
                            memory[key] = np.concatenate((memory[key], val), axis=0)
                        else:
                            memory[key] = val

        t_end = time.time()
        policy.to_device(device)
        return memory, t_end - t_start

    def collect_trajectory(
        self,
        pid,
        queue,
        policy: nn.Module,
        seed: int,
        deterministic: bool = False,
    ):
        env = get_env(self.env_name)

        # assign per-worker seed
        worker_seed = seed + pid
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        # estimate the batch size to hava a large batch
        data = self.get_reset_data()  # allocate memory

        current_time = 0

        # env initialization
        for i in range(self.num_episodes_per_worker):
            state, _ = env.reset(seed=worker_seed + i)
            for t in range(self.episode_len):
                with torch.no_grad():
                    a, metaData = policy(state, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                # env stepping
                next_state, rew, term, trunc, infos = env.step(np.argmax(a))

                if t == self.episode_len - 1:
                    trunc = True  # force truncation at the end of episode
                done = term or trunc

                # saving the data
                data["states"][current_time + t] = state
                data["next_states"][current_time + t] = next_state
                data["actions"][current_time + t] = a
                data["rewards"][current_time + t] = rew
                data["terminals"][current_time + t] = done
                data["logprobs"][current_time + t] = (
                    metaData["logprobs"].cpu().detach().numpy()
                )
                data["entropys"][current_time + t] = (
                    metaData["entropy"].cpu().detach().numpy()
                )

                if done:
                    current_time += t + 1
                    break

                state = next_state

        env.close()

        for k in data:
            data[k] = data[k][:current_time]

        if queue is not None:
            queue.put([pid, data])
        else:
            return data
