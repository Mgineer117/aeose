import random
import time
from datetime import date
from math import ceil, floor
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import time
import threading
from queue import Queue, Empty

from get_env import get_env
from utils.functions import temp_seed

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
        batch_size = 2 * self.episode_len
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

        self.total_num_worker = ceil(batch_size / episode_len)

        if verbose:
            print("Sampling Parameters:")
            print(f"Total number of workers: {self.total_num_worker}")

        torch.set_num_threads(1)  # Avoid CPU oversubscription

    def collect_samples(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
        random_init_pos: bool = False,
        use_threads: bool = True,  # ✅ new option to toggle threading vs sequential
    ):
        """
        Collect samples in parallel using threading (safe for Basilisk/SPICE).

        Args:
            policy: Policy to sample actions from.
            seed (int | None): Seed for reproducibility.
            deterministic (bool): Whether to use deterministic policy.
            random_init_pos (bool): Randomize initial position in env reset.
            use_threads (bool): If True, collect in parallel with threads.
                                If False, collect sequentially.

        Returns:
            memory (dict): Sampled batch.
            duration (float): Time taken to collect.
        """
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))

        # Run policy on CPU inside workers
        policy.to_device(torch.device("cpu"))

        # Shared queue between workers
        queue = Queue()
        worker_memories = [None] * self.total_num_worker
        threads = []

        if use_threads:
            # ✅ Spawn threads
            for i in range(self.total_num_worker):
                args = (i, queue, policy, seed, deterministic, random_init_pos)
                t = threading.Thread(target=self.collect_trajectory, args=args)
                threads.append(t)
                t.start()

            # ✅ Collect results
            expected = len(threads)
            collected = 0
            while collected < expected:
                try:
                    pid, data = queue.get(timeout=300)
                    if worker_memories[pid] is None:
                        worker_memories[pid] = data
                        collected += 1
                except Empty:
                    print(f"[Warning] Queue timeout. Collected {collected}/{expected}")
                    break

            # ✅ Join threads
            for t in threads:
                t.join(timeout=5)

        else:
            # ✅ Sequential collection (debug mode, safer fallback)
            for i in range(self.total_num_worker):
                _, data = self.collect_trajectory(
                    i, queue, policy, seed, deterministic, random_init_pos
                )
                worker_memories[i] = data

        # ✅ Merge memory
        memory = {}
        for wm in worker_memories:
            if wm is not None:
                for key, val in wm.items():
                    if key in memory:
                        memory[key] = np.concatenate((memory[key], wm[key]), axis=0)
                    else:
                        memory[key] = wm[key]

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
        random_init_pos: bool = False,
    ):
        # assign per-worker seed
        worker_seed = seed + pid
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        # estimate the batch size to hava a large batch
        env = get_env()
        data = self.get_reset_data()  # allocate memory

        current_time = 0
        while current_time < self.episode_len:
            # env initialization
            options = {"random_init_pos": random_init_pos}
            state, _ = env.reset(seed=worker_seed, options=options)

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

        for k in data:
            data[k] = data[k][:current_time]

        if queue is not None:
            queue.put([pid, data])
        else:
            return data
