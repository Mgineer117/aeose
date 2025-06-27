import random
import time
from datetime import date
from math import ceil, floor
from get_env import get_env
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from queue import Empty
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
        self.batch_size_for_worker = kwargs.get("batch_size_for_worker")

    def get_reset_data(self):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        np.nan makes it easy to debug
        """
        batch_size = self.batch_size_for_worker + self.episode_len
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
            batch_size_for_worker=episode_len,
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
    ):
        """
        Collect samples in parallel using multiprocessing.

        Args:
            env: The environment to interact with.
            policy: Policy to sample actions from.
            seed (int | None): Seed for reproducibility.
            deterministic (bool): Whether to use deterministic policy.
            random_init_pos (bool): Randomize initial position in env reset.

        Returns:
            memory (dict): Sampled batch.
            duration (float): Time taken to collect.
        """
        t_start = time.time()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))

        policy.to_device(torch.device("cpu"))

        processes = []
        queue = mp.Manager().Queue()
        worker_memories = [None] * self.total_num_worker
        for i in range(self.total_num_worker):
            args = (i, queue, policy, seed, deterministic)
            p = mp.Process(target=self.collect_trajectory, args=args)
            processes.append(p)
            p.start()

        # ✅ Wait for just the subprocess workers of this round
        expected = len(processes)
        collected = 0
        while collected < expected:
            try:
                pid, data = queue.get(timeout=600)
                if worker_memories[pid] is None:
                    worker_memories[pid] = data
                    collected += 1
            except Empty:
                print(f"[Warning] Queue timeout. Retrying... ({collected}/{expected})")

        start_time = time.time()
        for p in processes:
            p.join(timeout=max(0.1, 10 - (time.time() - start_time)))
            if p.is_alive():
                p.terminate()
                p.join()  # Force cleanup
            # if p.exitcode != 0:
            #     print(f"[Error] Process {p.pid} exited with code {p.exitcode}")

        # ✅ Merge memory
        memory = {}
        for wm in worker_memories:
            if wm is None:
                raise RuntimeError("One or more workers failed to return data.")
            for key, val in wm.items():
                if key in memory:
                    memory[key] = np.concatenate((memory[key], wm[key]), axis=0)
                else:
                    memory[key] = wm[key]

        # ✅ Truncate to desired batch size
        for k in memory:
            memory[k] = memory[k][: self.batch_size]

        t_end = time.time()
        policy.to_device(device)

        return memory, t_end - t_start

    def collect_trajectory(
        self, pid, queue, policy: nn.Module, seed: int, deterministic: bool = False
    ):
        # assign per-worker seed
        worker_seed = seed + pid
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(worker_seed)

        # estimate the batch size to hava a large batch
        data = self.get_reset_data()  # allocate memory

        # env initialization
        env = get_env()
        state, _ = env.reset(seed=seed)        
        for t in range(self.episode_len):
            with torch.no_grad():
                a, metaData = policy(state, deterministic=deterministic)
                a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

            # env stepping
            next_state, rew, term, trunc, infos = env.step(np.argmax(a))
            if t == self.episode_len - 1:
                # safe truncation
                trunc = True

            done = term or trunc

            # saving the data
            data["states"][t] = state
            data["next_states"][t] = next_state
            data["actions"][t] = a
            data["rewards"][t] = rew
            data["terminals"][t] = done
            data["logprobs"][t] = (
                metaData["logprobs"].cpu().detach().numpy()
            )
            data["entropys"][t] = (
                metaData["entropy"].cpu().detach().numpy()
            )

            state = next_state

            if done:
                break

        for k in data:
            data[k] = data[k][:t+1]

        if queue is not None:
            queue.put([pid, data])
        else:
            return data
