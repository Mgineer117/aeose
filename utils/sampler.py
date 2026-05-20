import random
import time
from math import ceil
import multiprocessing as mp

import numpy as np
import torch

from utils.get_env import get_env


class Base:
    def __init__(self, **kwargs):
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.episode_len = kwargs.get("episode_len")
        self.batch_size = kwargs.get("batch_size")


def _build_buffer(num_samples_per_worker, state_dim, action_dim):
    """Pre-allocate buffer for a fixed number of samples (not episodes)."""
    # Add extra capacity to handle edge cases
    size = num_samples_per_worker + 100
    return dict(
        states=np.full(((size,) + state_dim), np.nan, dtype=np.float32),
        next_states=np.full(((size,) + state_dim), np.nan, dtype=np.float32),
        actions=np.full((size, action_dim), np.nan, dtype=np.float32),
        rewards=np.full((size, 1), np.nan, dtype=np.float32),
        # terminations: env said the episode actually ended (failure / task done).
        # truncations:  artificial cutoff (env time_limit, or our safety-net at episode_len).
        # `terminals` is preserved as the OR of the two for any consumer that
        # just wants "episode is done here".
        terminations=np.full((size, 1), np.nan, dtype=np.float32),
        truncations=np.full((size, 1), np.nan, dtype=np.float32),
        terminals=np.full((size, 1), np.nan, dtype=np.float32),
        dones=np.full((size, 1), np.nan, dtype=np.float32),
        logprobs=np.full((size, 1), np.nan, dtype=np.float32),
        entropys=np.full((size, 1), np.nan, dtype=np.float32),
    )


def _run_episodes(
    policy,
    env,
    seed,
    deterministic,
    num_samples_per_worker,
    episode_len,
    state_dim,
    action_dim,
):
    """Collect exactly num_samples_per_worker samples by running full episodes,
    continuing across episode boundaries until the sample count is reached.
    Returns (data, timings) where `timings` breaks down where wall-clock
    inside the worker actually went."""
    data = _build_buffer(num_samples_per_worker, state_dim, action_dim)
    current_time = 0
    ep = 0
    t_policy = 0.0
    t_step = 0.0
    t_reset = 0.0
    t_buffer = 0.0
    n_resets = 0
    t_worker_start = time.time()

    while current_time < num_samples_per_worker:
        _t = time.time()
        state, _ = env.reset(seed=seed + ep)
        t_reset += time.time() - _t
        n_resets += 1
        for t in range(episode_len):
            # Stop if we've collected enough samples
            if current_time >= num_samples_per_worker:
                break

            _t = time.time()
            with torch.no_grad():
                a, metaData = policy(state, deterministic=deterministic)
                a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]
            t_policy += time.time() - _t

            # Wrap env.step so a SPICE/Basilisk numeric blow-up doesn't kill
            # the whole worker. We end the episode as a truncation in that
            # case so it still bootstraps cleanly.
            _t = time.time()
            try:
                next_state, rew, term, trunc, _ = env.step(np.argmax(a))
            except Exception as exc:
                print(
                    f"[sampler] env.step raised: {exc!r}. Ending episode as truncated."
                )
                next_state = state
                rew = 0.0
                term = False
                trunc = True
            t_step += time.time() - _t

            # Safety-net truncation if the env didn't already signal the end.
            if t == episode_len - 1 and not (term or trunc):
                trunc = True

            done = bool(term) or bool(trunc)
            _t = time.time()
            idx = current_time
            data["states"][idx] = state
            data["next_states"][idx] = next_state
            data["actions"][idx] = a
            data["rewards"][idx] = rew
            data["terminations"][idx] = float(bool(term))
            data["truncations"][idx] = float(bool(trunc))
            data["terminals"][idx] = float(done)
            data["dones"][idx] = float(done)
            data["logprobs"][idx] = metaData["logprobs"].cpu().detach().numpy()
            data["entropys"][idx] = metaData["entropy"].cpu().detach().numpy()
            t_buffer += time.time() - _t

            current_time += 1
            if not done:
                state = next_state
            else:
                break  # Episode ended, will start a new one on next loop iteration

        ep += 1

    # Trim buffer to exact number of samples collected
    for k in data:
        data[k] = data[k][:num_samples_per_worker]

    timings = {
        "policy": t_policy,
        "step": t_step,
        "reset": t_reset,
        "buffer": t_buffer,
        "worker_wall": time.time() - t_worker_start,
        "n_envs": 1,
        "n_steps": current_time,
        "n_resets": n_resets,
    }
    return data, timings


def _run_episodes_vec(
    policy,
    envs,
    seed,
    deterministic,
    num_samples_per_worker,
    episode_len,
    state_dim,
    action_dim,
):
    """Vectorized rollout: batches policy inference across `len(envs)` envs
    living in the same worker process. env.step runs serially per env (no
    threading — Basilisk does not benefit from it in practice). Each env
    keeps its own trajectory and is reset on done. We collect
    `num_samples_per_worker` total transitions across all envs in this
    worker (sampling order is interleaved by env).
    """
    n = len(envs)
    data = _build_buffer(num_samples_per_worker, state_dim, action_dim)
    t_policy = 0.0
    t_step = 0.0
    t_reset = 0.0
    t_buffer = 0.0
    n_resets = 0
    t_worker_start = time.time()

    # Per-env state and episode bookkeeping.
    states = np.zeros(((n,) + state_dim), dtype=np.float32)
    ep_counts = [0] * n
    ep_steps = [0] * n
    for i, env in enumerate(envs):
        _t = time.time()
        s, _ = env.reset(seed=seed + i * 1009)
        t_reset += time.time() - _t
        n_resets += 1
        states[i] = s

    current_time = 0
    while current_time < num_samples_per_worker:
        # One batched forward pass across all envs in this worker.
        _t = time.time()
        with torch.no_grad():
            a_tensor, meta = policy(states, deterministic=deterministic)
            a_np = a_tensor.cpu().numpy()
            lp_np = meta["logprobs"].cpu().detach().numpy()
            ent_np = meta["entropy"].cpu().detach().numpy()
        t_policy += time.time() - _t

        for i, env in enumerate(envs):
            if current_time >= num_samples_per_worker:
                break

            _t = time.time()
            try:
                ns, rew, term, trunc, _ = env.step(int(np.argmax(a_np[i])))
            except Exception as exc:
                print(
                    f"[sampler] env[{i}].step raised: {exc!r}. "
                    f"Ending episode as truncated."
                )
                ns = states[i]
                rew = 0.0
                term = False
                trunc = True
            t_step += time.time() - _t

            if ep_steps[i] == episode_len - 1 and not (term or trunc):
                trunc = True
            done = bool(term) or bool(trunc)

            _t = time.time()
            idx = current_time
            data["states"][idx] = states[i]
            data["next_states"][idx] = ns
            data["actions"][idx] = a_np[i]
            data["rewards"][idx] = rew
            data["terminations"][idx] = float(bool(term))
            data["truncations"][idx] = float(bool(trunc))
            data["terminals"][idx] = float(done)
            data["dones"][idx] = float(done)
            data["logprobs"][idx] = lp_np[i]
            data["entropys"][idx] = ent_np[i]
            t_buffer += time.time() - _t

            current_time += 1
            ep_steps[i] += 1

            if done:
                ep_counts[i] += 1
                ep_steps[i] = 0
                _t = time.time()
                ns, _ = env.reset(seed=seed + i * 1009 + ep_counts[i] * 7919)
                t_reset += time.time() - _t
                n_resets += 1

            states[i] = ns

    for k in data:
        data[k] = data[k][:num_samples_per_worker]

    timings = {
        "policy": t_policy,
        "step": t_step,
        "reset": t_reset,
        "buffer": t_buffer,
        "worker_wall": time.time() - t_worker_start,
        "n_envs": n,
        "n_steps": current_time,
        "n_resets": n_resets,
    }
    return data, timings


def _forkserver_env_worker(conn, env_name, episode_len):
    env = None
    t_step = 0.0
    t_reset = 0.0
    t_wait = 0.0
    n_resets = 0
    n_steps = 0
    ep_step = 0

    try:
        env = get_env(env_name)
        conn.send(("ready", None))
        while True:
            t_recv = time.time()
            try:
                msg = conn.recv()
            except EOFError:
                break
            t_wait += time.time() - t_recv

            if not msg:
                continue

            cmd = msg[0]
            if cmd == "reset":
                seed = int(msg[1])
                _t = time.time()
                obs, _ = env.reset(seed=seed)
                t_reset += time.time() - _t
                n_resets += 1
                ep_step = 0
                conn.send(("obs", obs))
            elif cmd == "step":
                action = int(msg[1])
                _t = time.time()
                try:
                    next_obs, rew, term, trunc, _ = env.step(action)
                except Exception as exc:
                    print(
                        f"[sampler/forkserver] env.step raised: {exc!r}. "
                        "Ending episode as truncated."
                    )
                    next_obs = None
                    rew = 0.0
                    term = False
                    trunc = True
                t_step += time.time() - _t
                n_steps += 1

                ep_step += 1
                if ep_step >= episode_len and not (term or trunc):
                    trunc = True

                done = bool(term) or bool(trunc)
                if done:
                    _t = time.time()
                    reset_obs, _ = env.reset()
                    t_reset += time.time() - _t
                    n_resets += 1
                    ep_step = 0
                else:
                    reset_obs = next_obs

                conn.send(("transition", next_obs, rew, bool(term), bool(trunc), reset_obs))
            elif cmd == "stats":
                conn.send(
                    {
                        "policy": 0.0,
                        "step": t_step,
                        "reset": t_reset,
                        "buffer": 0.0,
                        "worker_wall": t_step + t_reset + t_wait,
                        "n_envs": 1,
                        "n_steps": n_steps,
                        "n_resets": n_resets,
                    }
                )
                t_step = 0.0
                t_reset = 0.0
                t_wait = 0.0
                n_resets = 0
                n_steps = 0
            elif cmd == "close":
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def _spawn_env_worker(conn, env_name, episode_len):
    """Spawn worker process for environment sampling (same as forkserver but using spawn context)."""
    env = None
    t_step = 0.0
    t_reset = 0.0
    t_wait = 0.0
    n_resets = 0
    n_steps = 0
    ep_step = 0

    try:
        env = get_env(env_name)
        conn.send(("ready", None))
        while True:
            t_recv = time.time()
            try:
                msg = conn.recv()
            except EOFError:
                break
            t_wait += time.time() - t_recv

            if not msg:
                continue

            cmd = msg[0]
            if cmd == "reset":
                seed = int(msg[1])
                _t = time.time()
                obs, _ = env.reset(seed=seed)
                t_reset += time.time() - _t
                n_resets += 1
                ep_step = 0
                conn.send(("obs", obs))
            elif cmd == "step":
                action = int(msg[1])
                _t = time.time()
                try:
                    next_obs, rew, term, trunc, _ = env.step(action)
                except Exception as exc:
                    print(
                        f"[sampler/spawn] env.step raised: {exc!r}. "
                        "Ending episode as truncated."
                    )
                    next_obs = None
                    rew = 0.0
                    term = False
                    trunc = True
                t_step += time.time() - _t
                n_steps += 1

                ep_step += 1
                if ep_step >= episode_len and not (term or trunc):
                    trunc = True

                done = bool(term) or bool(trunc)
                if done:
                    _t = time.time()
                    reset_obs, _ = env.reset()
                    t_reset += time.time() - _t
                    n_resets += 1
                    ep_step = 0
                else:
                    reset_obs = next_obs

                conn.send(("transition", next_obs, rew, bool(term), bool(trunc), reset_obs))
            elif cmd == "stats":
                conn.send(
                    {
                        "policy": 0.0,
                        "step": t_step,
                        "reset": t_reset,
                        "buffer": 0.0,
                        "worker_wall": t_step + t_reset + t_wait,
                        "n_envs": 1,
                        "n_steps": n_steps,
                        "n_resets": n_resets,
                    }
                )
                t_step = 0.0
                t_reset = 0.0
                t_wait = 0.0
                n_resets = 0
                n_steps = 0
            elif cmd == "close":
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def _async_vectorized_worker(conn, env_name, episode_len):
    """Async vectorized worker process (identical to forkserver/spawn, used in AsyncVectorEnv collection)."""
    env = None
    t_step = 0.0
    t_reset = 0.0
    t_wait = 0.0
    n_resets = 0
    n_steps = 0
    ep_step = 0

    try:
        env = get_env(env_name)
        conn.send(("ready", None))
        while True:
            t_recv = time.time()
            try:
                msg = conn.recv()
            except EOFError:
                break
            t_wait += time.time() - t_recv

            if not msg:
                continue

            cmd = msg[0]
            if cmd == "reset":
                seed = int(msg[1])
                _t = time.time()
                obs, _ = env.reset(seed=seed)
                t_reset += time.time() - _t
                n_resets += 1
                ep_step = 0
                conn.send(("obs", obs))
            elif cmd == "step":
                action = int(msg[1])
                _t = time.time()
                try:
                    next_obs, rew, term, trunc, _ = env.step(action)
                except Exception as exc:
                    print(
                        f"[sampler/async_vec] env.step raised: {exc!r}. "
                        "Ending episode as truncated."
                    )
                    next_obs = None
                    rew = 0.0
                    term = False
                    trunc = True
                t_step += time.time() - _t
                n_steps += 1

                ep_step += 1
                if ep_step >= episode_len and not (term or trunc):
                    trunc = True

                done = bool(term) or bool(trunc)
                if done:
                    _t = time.time()
                    reset_obs, _ = env.reset()
                    t_reset += time.time() - _t
                    n_resets += 1
                    ep_step = 0
                else:
                    reset_obs = next_obs

                conn.send(("transition", next_obs, rew, bool(term), bool(trunc), reset_obs))
            elif cmd == "stats":
                conn.send(
                    {
                        "policy": 0.0,
                        "step": t_step,
                        "reset": t_reset,
                        "buffer": 0.0,
                        "worker_wall": t_step + t_reset + t_wait,
                        "n_envs": 1,
                        "n_steps": n_steps,
                        "n_resets": n_resets,
                    }
                )
                t_step = 0.0
                t_reset = 0.0
                t_wait = 0.0
                n_resets = 0
                n_steps = 0
            elif cmd == "close":
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class OnlineSampler(Base):
    def __init__(
        self,
        env_name: str,
        state_dim: tuple,
        action_dim: int,
        episode_len: int,
        batch_size: int,
        num_workers: int = 0,
        episodes_per_worker: int = 0,
        envs_per_worker: int = 1,
        sampler_mode: str = "vectorized",
        first_round_timeout: int = 3600,
        steady_timeout: int = 1200,
        verbose: bool = True,
    ) -> None:
        """
                Sampler with selectable parallelization mode.

                Modes:
                    - vectorized: in-process env batching
                    - forkserver: one env per forkserver worker process
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
        )

        self.env_name = env_name
        self.sampler_mode = str(sampler_mode).lower()
        if self.sampler_mode not in {"vectorized", "async_vectorized", "forkserver", "spawn"}:
            raise ValueError(
                f"Unknown sampler_mode={sampler_mode!r}. "
                "Expected one of {'vectorized', 'async_vectorized', 'forkserver', 'spawn'}."
            )

        self.envs_per_worker = max(2, int(envs_per_worker))
        self.num_workers = max(1, int(num_workers))
        self.total_num_worker = 1
        self.num_samples_per_worker = int(batch_size)

        if verbose:
            print("Sampling Parameters:")
            print(f"Sampler mode:                {self.sampler_mode}")
            print(f"Vectorized envs:             {self.envs_per_worker}")
            print(f"Workers:                     {self.num_workers}")
            print(f"Samples per call:            {self.num_samples_per_worker}")

        torch.set_num_threads(1)  # avoid CPU oversubscription in parent

        self._envs = []
        self._fs_ctx = None
        self._fs_conns = []
        self._fs_workers = []
        self._fs_states = None
        self._spawn_ctx = None
        self._spawn_conns = []
        self._spawn_workers = []
        self._spawn_states = None
        self._async_ctx = None
        self._async_conns = []
        self._async_workers = []
        self._async_states = None
        if self.sampler_mode == "vectorized":
            self._envs = [get_env(self.env_name) for _ in range(self.envs_per_worker)]
        elif self.sampler_mode == "async_vectorized":
            self._init_async_vectorized_workers()
        elif self.sampler_mode == "forkserver":
            self._init_forkserver_workers()
        elif self.sampler_mode == "spawn":
            self._init_spawn_workers()

        self._iter_idx = 0
        self._pending_batch = None
        self._pending_first_round = False
        self._pending_t_start = None
        self._has_pending = False

    def _init_forkserver_workers(self):
        self._fs_ctx = mp.get_context("forkserver")
        self._fs_conns = []
        self._fs_workers = []
        for _ in range(self.num_workers):
            parent_conn, child_conn = self._fs_ctx.Pipe()
            proc = self._fs_ctx.Process(
                target=_forkserver_env_worker,
                args=(child_conn, self.env_name, self.episode_len),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()
            self._fs_conns.append(parent_conn)
            self._fs_workers.append(proc)

        for conn in self._fs_conns:
            msg = conn.recv()
            if not (isinstance(msg, tuple) and msg[0] == "ready"):
                raise RuntimeError(f"Forkserver worker failed to start: {msg!r}")

        self._fs_states = np.zeros((self.num_workers,) + self.state_dim, dtype=np.float32)

    def _init_spawn_workers(self):
        """Initialize spawn-based worker processes (similar to forkserver but using spawn context)."""
        self._spawn_ctx = mp.get_context("spawn")
        self._spawn_conns = []
        self._spawn_workers = []
        for _ in range(self.num_workers):
            parent_conn, child_conn = self._spawn_ctx.Pipe()
            proc = self._spawn_ctx.Process(
                target=_spawn_env_worker,
                args=(child_conn, self.env_name, self.episode_len),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()
            self._spawn_conns.append(parent_conn)
            self._spawn_workers.append(proc)

        for conn in self._spawn_conns:
            msg = conn.recv()
            if not (isinstance(msg, tuple) and msg[0] == "ready"):
                raise RuntimeError(f"Spawn worker failed to start: {msg!r}")

        self._spawn_states = np.zeros((self.num_workers,) + self.state_dim, dtype=np.float32)

    def _init_async_vectorized_workers(self):
        """Initialize async vectorized worker processes (true parallelization like SB3 AsyncVectorEnv)."""
        self._async_ctx = mp.get_context("forkserver")
        self._async_conns = []
        self._async_workers = []
        for _ in range(self.num_workers):
            parent_conn, child_conn = self._async_ctx.Pipe()
            proc = self._async_ctx.Process(
                target=_async_vectorized_worker,
                args=(child_conn, self.env_name, self.episode_len),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()
            self._async_conns.append(parent_conn)
            self._async_workers.append(proc)

        for conn in self._async_conns:
            msg = conn.recv()
            if not (isinstance(msg, tuple) and msg[0] == "ready"):
                raise RuntimeError(f"Async vectorized worker failed to start: {msg!r}")

        self._async_states = np.zeros((self.num_workers,) + self.state_dim, dtype=np.float32)

    def dispatch(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """Compatibility shim for the trainer's async path.

        There is no background worker anymore, so dispatch computes the next
        vectorized batch immediately and stores it for gather().
        """
        if self._has_pending:
            raise RuntimeError(
                "OnlineSampler.dispatch called while a previous dispatch "
                "is still pending. Call gather() (or drain()) first."
            )

        if self.sampler_mode == "forkserver":
            self._pending_batch = self._collect_forkserver(
                policy, seed=seed, deterministic=deterministic
            )
        else:
            self._pending_batch = self._collect_vectorized(
                policy, seed=seed, deterministic=deterministic
            )
        self._has_pending = True

    def gather(self, policy):
        """Return the most recent dispatch result."""
        if not self._has_pending:
            raise RuntimeError(
                "OnlineSampler.gather called with no pending dispatch."
            )

        memory, wall = self._pending_batch
        self._pending_batch = None
        self._has_pending = False
        self._pending_t_start = None
        return memory, wall

    def _collect_vectorized(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        seed_val = 0 if seed is None else int(seed)
        self._iter_idx += 1
        worker_seed = seed_val + self._iter_idx * 31

        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )

        try:
            policy.eval()
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            memory, timings = _run_episodes_vec(
                policy,
                self._envs,
                worker_seed,
                deterministic,
                self.num_samples_per_worker,
                self.episode_len,
                self.state_dim,
                self.action_dim,
            )
            wall = timings["worker_wall"]
            self._print_timing_breakdown([timings], wall, 0.0)
            return memory, wall
        finally:
            policy.to_device(original_device)

    def _collect_forkserver(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        seed_val = 0 if seed is None else int(seed)
        self._iter_idx += 1
        worker_seed = seed_val + self._iter_idx * 31

        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )

        try:
            policy.eval()
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            for i, conn in enumerate(self._fs_conns):
                conn.send(("reset", worker_seed + i * 1009))
            for i, conn in enumerate(self._fs_conns):
                msg = conn.recv()
                if not (isinstance(msg, tuple) and msg[0] == "obs"):
                    raise RuntimeError(f"Unexpected reset response from worker {i}: {msg!r}")
                self._fs_states[i] = msg[1]

            data = _build_buffer(
                self.num_samples_per_worker,
                self.state_dim,
                self.action_dim,
            )
            t_policy = 0.0
            t_buffer = 0.0
            t_worker_wait = 0.0
            current_time = 0

            while current_time < self.num_samples_per_worker:
                _t = time.time()
                with torch.no_grad():
                    a_tensor, meta = policy(self._fs_states, deterministic=deterministic)
                    a_np = a_tensor.cpu().numpy()
                    lp_np = meta["logprobs"].cpu().detach().numpy()
                    ent_np = meta["entropy"].cpu().detach().numpy()
                t_policy += time.time() - _t

                for i, conn in enumerate(self._fs_conns):
                    conn.send(("step", int(np.argmax(a_np[i]))))

                _t = time.time()
                for i, conn in enumerate(self._fs_conns):
                    msg = conn.recv()
                    if not (isinstance(msg, tuple) and msg[0] == "transition"):
                        raise RuntimeError(
                            f"Unexpected step response from worker {i}: {msg!r}"
                        )

                    step_next_state, rew, term, trunc, next_policy_state = msg[1:]

                    if current_time < self.num_samples_per_worker:
                        _tb = time.time()
                        idx = current_time
                        done = bool(term) or bool(trunc)
                        data["states"][idx] = self._fs_states[i]
                        data["next_states"][idx] = step_next_state
                        data["actions"][idx] = a_np[i]
                        data["rewards"][idx] = rew
                        data["terminations"][idx] = float(bool(term))
                        data["truncations"][idx] = float(bool(trunc))
                        data["terminals"][idx] = float(done)
                        data["dones"][idx] = float(done)
                        data["logprobs"][idx] = lp_np[i]
                        data["entropys"][idx] = ent_np[i]
                        t_buffer += time.time() - _tb
                        current_time += 1

                    self._fs_states[i] = next_policy_state
                t_worker_wait += time.time() - _t

            for k in data:
                data[k] = data[k][: self.num_samples_per_worker]

            worker_timings = []
            for conn in self._fs_conns:
                conn.send(("stats",))
            for conn in self._fs_conns:
                msg = conn.recv()
                if isinstance(msg, dict):
                    worker_timings.append(msg)

            wall = t_policy + t_worker_wait + t_buffer
            if worker_timings:
                for wt in worker_timings:
                    wt["policy"] = t_policy / max(1, len(worker_timings))
                    wt["buffer"] = t_buffer / max(1, len(worker_timings))
                self._print_timing_breakdown(worker_timings, wall, 0.0)

            return data, wall
        finally:
            policy.to_device(original_device)

    def _collect_spawn(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """Collect samples using spawn-based worker processes (functionally identical to forkserver)."""
        seed_val = 0 if seed is None else int(seed)
        self._iter_idx += 1
        worker_seed = seed_val + self._iter_idx * 31

        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )

        try:
            policy.eval()
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            for i, conn in enumerate(self._spawn_conns):
                conn.send(("reset", worker_seed + i * 1009))
            for i, conn in enumerate(self._spawn_conns):
                msg = conn.recv()
                if not (isinstance(msg, tuple) and msg[0] == "obs"):
                    raise RuntimeError(f"Unexpected reset response from worker {i}: {msg!r}")
                self._spawn_states[i] = msg[1]

            data = _build_buffer(
                self.num_samples_per_worker,
                self.state_dim,
                self.action_dim,
            )
            t_policy = 0.0
            t_buffer = 0.0
            t_worker_wait = 0.0
            current_time = 0

            while current_time < self.num_samples_per_worker:
                _t = time.time()
                with torch.no_grad():
                    a_tensor, meta = policy(self._spawn_states, deterministic=deterministic)
                    a_np = a_tensor.cpu().numpy()
                    lp_np = meta["logprobs"].cpu().detach().numpy()
                    ent_np = meta["entropy"].cpu().detach().numpy()
                t_policy += time.time() - _t

                for i, conn in enumerate(self._spawn_conns):
                    conn.send(("step", int(np.argmax(a_np[i]))))

                _t = time.time()
                for i, conn in enumerate(self._spawn_conns):
                    msg = conn.recv()
                    if not (isinstance(msg, tuple) and msg[0] == "transition"):
                        raise RuntimeError(
                            f"Unexpected step response from worker {i}: {msg!r}"
                        )

                    step_next_state, rew, term, trunc, next_policy_state = msg[1:]

                    if current_time < self.num_samples_per_worker:
                        _tb = time.time()
                        idx = current_time
                        done = bool(term) or bool(trunc)
                        data["states"][idx] = self._spawn_states[i]
                        data["next_states"][idx] = step_next_state
                        data["actions"][idx] = a_np[i]
                        data["rewards"][idx] = rew
                        data["terminations"][idx] = float(bool(term))
                        data["truncations"][idx] = float(bool(trunc))
                        data["terminals"][idx] = float(done)
                        data["dones"][idx] = float(done)
                        data["logprobs"][idx] = lp_np[i]
                        data["entropys"][idx] = ent_np[i]
                        t_buffer += time.time() - _tb
                        current_time += 1

                    self._spawn_states[i] = next_policy_state
                t_worker_wait += time.time() - _t

            for k in data:
                data[k] = data[k][: self.num_samples_per_worker]

            worker_timings = []
            for conn in self._spawn_conns:
                conn.send(("stats",))
            for conn in self._spawn_conns:
                msg = conn.recv()
                if isinstance(msg, dict):
                    worker_timings.append(msg)

            wall = t_policy + t_worker_wait + t_buffer
            if worker_timings:
                for wt in worker_timings:
                    wt["policy"] = t_policy / max(1, len(worker_timings))
                    wt["buffer"] = t_buffer / max(1, len(worker_timings))
                self._print_timing_breakdown(worker_timings, wall, 0.0)

            return data, wall
        finally:
            policy.to_device(original_device)

    @staticmethod
    def _print_timing_breakdown(worker_timings, wall, t_concat):
        """Print where the most recent sampling call spent its wall-clock.
        Per-worker times are summed across the worker's own loop and then
        averaged so the breakdown is per-worker (not per-call), since
        workers run in parallel and 'total per worker' is the meaningful
        comparison against `wall`."""
        good = [t for t in worker_timings if t is not None]
        if not good:
            print(f"[Sampler] wall={wall:.2f}s  (no per-worker timings — workers timed out)")
            return

        n = len(good)
        agg = {
            "policy": sum(t["policy"] for t in good) / n,
            "step":   sum(t["step"]   for t in good) / n,
            "reset":  sum(t["reset"]  for t in good) / n,
            "buffer": sum(t["buffer"] for t in good) / n,
            "worker_wall": sum(t["worker_wall"] for t in good) / n,
        }
        n_steps = sum(t["n_steps"] for t in good)
        n_resets = sum(t["n_resets"] for t in good)
        n_envs = good[0]["n_envs"]

        # Wall - max(per-worker wall) ≈ overhead from queues / dispatch /
        # workers finishing at slightly different times.
        max_worker = max(t["worker_wall"] for t in good)
        overhead = max(0.0, wall - max_worker)
        per_step_step = (agg["step"] / max(1, sum(t["n_steps"] for t in good) / n))

        print(
            f"[Sampler] wall={wall:.2f}s  workers={n}  envs/worker={n_envs}  "
            f"total_steps={n_steps}  resets={n_resets}\n"
            f"   per-worker avg: env.step={agg['step']:.2f}s ({100*agg['step']/agg['worker_wall']:.0f}%)  "
            f"policy={agg['policy']:.2f}s ({100*agg['policy']/agg['worker_wall']:.0f}%)  "
            f"env.reset={agg['reset']:.2f}s ({100*agg['reset']/agg['worker_wall']:.0f}%)  "
            f"buffer={agg['buffer']:.3f}s\n"
            f"   per-step: env.step≈{1000*per_step_step:.1f}ms  "
            f"worker_wall={agg['worker_wall']:.2f}s  "
            f"max_worker={max_worker:.2f}s  "
            f"overhead(wall-max_worker)={overhead:.2f}s  "
            f"concat={t_concat:.3f}s"
        )

    def drain(self, policy):
        """Consume and discard any in-flight dispatched batch."""
        if not self._has_pending:
            return
        self._pending_batch = None
        self._has_pending = False
        self._pending_t_start = None

    def _collect_async_vectorized(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """Collect samples using async vectorized workers (true parallelization like SB3 AsyncVectorEnv).
        
        Key difference from forkserver/spawn: sends ALL actions at once (non-blocking),
        then collects ALL results in parallel, enabling true environment step parallelization.
        """
        seed_val = 0 if seed is None else int(seed)
        self._iter_idx += 1
        worker_seed = seed_val + self._iter_idx * 31

        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )

        try:
            policy.eval()
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            # Initial reset: send all resets in one go
            for i, conn in enumerate(self._async_conns):
                conn.send(("reset", worker_seed + i * 1009))
            for i, conn in enumerate(self._async_conns):
                msg = conn.recv()
                if not (isinstance(msg, tuple) and msg[0] == "obs"):
                    raise RuntimeError(f"Unexpected reset response from worker {i}: {msg!r}")
                self._async_states[i] = msg[1]

            data = _build_buffer(
                self.num_samples_per_worker,
                self.state_dim,
                self.action_dim,
            )
            t_policy = 0.0
            t_buffer = 0.0
            t_worker_wait = 0.0
            current_time = 0

            while current_time < self.num_samples_per_worker:
                # Batched policy forward pass
                _t = time.time()
                with torch.no_grad():
                    a_tensor, meta = policy(self._async_states, deterministic=deterministic)
                    a_np = a_tensor.cpu().numpy()
                    lp_np = meta["logprobs"].cpu().detach().numpy()
                    ent_np = meta["entropy"].cpu().detach().numpy()
                t_policy += time.time() - _t

                # Send ALL actions to workers in one go (non-blocking)
                for i, conn in enumerate(self._async_conns):
                    conn.send(("step", int(np.argmax(a_np[i]))))

                # Collect ALL results in parallel (workers can run simultaneously)
                _t = time.time()
                for i, conn in enumerate(self._async_conns):
                    msg = conn.recv()
                    if not (isinstance(msg, tuple) and msg[0] == "transition"):
                        raise RuntimeError(
                            f"Unexpected step response from worker {i}: {msg!r}"
                        )

                    step_next_state, rew, term, trunc, next_policy_state = msg[1:]

                    if current_time < self.num_samples_per_worker:
                        _tb = time.time()
                        idx = current_time
                        done = bool(term) or bool(trunc)
                        data["states"][idx] = self._async_states[i]
                        data["next_states"][idx] = step_next_state
                        data["actions"][idx] = a_np[i]
                        data["rewards"][idx] = rew
                        data["terminations"][idx] = float(bool(term))
                        data["truncations"][idx] = float(bool(trunc))
                        data["terminals"][idx] = float(done)
                        data["dones"][idx] = float(done)
                        data["logprobs"][idx] = lp_np[i]
                        data["entropys"][idx] = ent_np[i]
                        t_buffer += time.time() - _tb
                        current_time += 1

                    self._async_states[i] = next_policy_state
                t_worker_wait += time.time() - _t

            for k in data:
                data[k] = data[k][: self.num_samples_per_worker]

            worker_timings = []
            for conn in self._async_conns:
                conn.send(("stats",))
            for conn in self._async_conns:
                msg = conn.recv()
                if isinstance(msg, dict):
                    worker_timings.append(msg)

            wall = t_policy + t_worker_wait + t_buffer
            if worker_timings:
                for wt in worker_timings:
                    wt["policy"] = t_policy / max(1, len(worker_timings))
                    wt["buffer"] = t_buffer / max(1, len(worker_timings))
                self._print_timing_breakdown(worker_timings, wall, 0.0)

            return data, wall
        finally:
            policy.to_device(original_device)

    def collect_samples(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
        use_mp: bool = True,
    ):
        """Synchronous one-shot collect using the configured sampler backend."""
        _ = use_mp
        if self.sampler_mode == "forkserver":
            return self._collect_forkserver(policy, seed=seed, deterministic=deterministic)
        elif self.sampler_mode == "spawn":
            return self._collect_spawn(policy, seed=seed, deterministic=deterministic)
        elif self.sampler_mode == "async_vectorized":
            return self._collect_async_vectorized(policy, seed=seed, deterministic=deterministic)
        return self._collect_vectorized(policy, seed=seed, deterministic=deterministic)

    def _close_forkserver_workers(self):
        for conn in self._fs_conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for conn in self._fs_conns:
            try:
                conn.close()
            except Exception:
                pass
        for worker in self._fs_workers:
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
        self._fs_conns = []
        self._fs_workers = []
        self._fs_states = None

    def _close_spawn_workers(self):
        """Close spawn worker processes (similar to forkserver cleanup)."""
        for conn in self._spawn_conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for conn in self._spawn_conns:
            try:
                conn.close()
            except Exception:
                pass
        for worker in self._spawn_workers:
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
        self._spawn_conns = []
        self._spawn_workers = []
        self._spawn_states = None

    def _close_async_vectorized_workers(self):
        """Close async vectorized worker processes."""
        for conn in self._async_conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for conn in self._async_conns:
            try:
                conn.close()
            except Exception:
                pass
        for worker in self._async_workers:
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
        self._async_conns = []
        self._async_workers = []
        self._async_states = None

    def close(self):
        if self.sampler_mode == "forkserver":
            self._close_forkserver_workers()
        elif self.sampler_mode == "spawn":
            self._close_spawn_workers()
        elif self.sampler_mode == "async_vectorized":
            self._close_async_vectorized_workers()
        if hasattr(self, "_envs"):
            for env in self._envs:
                try:
                    env.close()
                except Exception:
                    pass
            self._envs = []
        self._pending_batch = None
        self._has_pending = False
        self._pending_t_start = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
