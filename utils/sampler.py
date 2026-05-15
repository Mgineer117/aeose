import os
import random
import time
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp

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


def _worker_loop(
    pid,
    task_q,
    result_q,
    policy,
    env_name,
    num_samples_per_worker,
    episode_len,
    state_dim,
    action_dim,
    envs_per_worker,
):
    """Persistent fork worker: creates env(s) once and processes rollout
    requests until exit. When envs_per_worker > 1 the worker hosts a small
    batch of envs and uses vectorized policy inference."""
    # Workers must stay CPU-only. The CUDA context inherited via fork is not
    # valid in the child — any CUDA call (incl. cuda.manual_seed_all) would
    # raise cudaErrorInvalidResourceHandle.
    torch.set_num_threads(1)
    envs_per_worker = max(1, int(envs_per_worker))
    envs = [get_env(env_name) for _ in range(envs_per_worker)]

    while True:
        try:
            msg = task_q.get()
        except (KeyboardInterrupt, EOFError):
            break
        if msg is None:
            break

        state_dict, seed, deterministic, iter_idx = msg
        try:
            policy.load_state_dict(state_dict)
            policy.eval()

            worker_seed = int(seed) + pid * 9173 + iter_idx * 31
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

            if envs_per_worker == 1:
                data, timings = _run_episodes(
                    policy,
                    envs[0],
                    worker_seed,
                    deterministic,
                    num_samples_per_worker,
                    episode_len,
                    state_dim,
                    action_dim,
                )
            else:
                data, timings = _run_episodes_vec(
                    policy,
                    envs,
                    worker_seed,
                    deterministic,
                    num_samples_per_worker,
                    episode_len,
                    state_dim,
                    action_dim,
                )
            result_q.put((pid, data, timings))
        except Exception:
            import traceback

            traceback.print_exc()
            result_q.put((pid, None, None))

    for env in envs:
        try:
            env.close()
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
        first_round_timeout: int = 3600,
        steady_timeout: int = 1200,
        verbose: bool = True,
    ) -> None:
        """
        Persistent multiprocessing sampler with a forkserver-first start
        method. Workers are spawned once (each builds its own Basilisk env
        fresh) and reused across iterations — every collect_samples call
        just broadcasts a state_dict over a queue and waits for rollouts.

        Worker count is capped (see __init__) because each worker is a
        whole Basilisk+SPICE process; we'd rather have a handful of workers
        running more episodes than dozens of workers fighting over SPICE
        kernel files at import time.
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
        )

        self.env_name = env_name
        self.envs_per_worker = max(1, int(envs_per_worker))

        # Cap workers. Each worker holds a full Basilisk + SPICE process (and
        # now `envs_per_worker` Basilisk envs inside that process), so
        # spawning dozens of them at once causes import storms, file-handle
        # contention on shared SPICE kernels, and OOM.
        cpu_cap = max(1, (os.cpu_count() or 4) - 1)
        if num_workers and num_workers > 0:
            max_workers = max(1, num_workers)
        else:
            max_workers = min(cpu_cap, 8)

        # Calculate samples per worker (independent of episode boundaries).
        # Total samples target is batch_size; distribute across workers.
        self.total_num_worker = max(1, min(ceil(batch_size / episode_len), max_workers))
        self.num_samples_per_worker = max(
            episode_len, ceil(batch_size / self.total_num_worker)
        )

        if verbose:
            print("Sampling Parameters:")
            print(f"Total number of workers:     {self.total_num_worker}")
            print(f"Envs per worker:             {self.envs_per_worker}")
            print(f"Samples per worker / call:   {self.num_samples_per_worker}")
            print(
                f"Approx total samples/call:   {self.total_num_worker * self.num_samples_per_worker}"
            )

        torch.set_num_threads(1)  # avoid CPU oversubscription in parent

        # Use forkserver instead of fork: the parent has already imported
        # Basilisk and initialized SPICE kernels (via the eval env built in
        # main.py), and plain fork inherits all of that into the worker.
        # When the worker then re-initializes its own sim, SPICE handles
        # collide with inherited state and SPKE02 trips with INVALIDRADIUS.
        # forkserver starts from a cold child process, so each worker imports
        # Basilisk and initializes SPICE fresh.
        for method in ("forkserver", "spawn", "fork"):
            try:
                self._ctx = mp.get_context(method)
                self._start_method = method
                break
            except ValueError:
                continue
        if verbose:
            print(f"Sampler start method: {self._start_method}")

        self._workers = []
        self._task_queues = []
        self._result_queue = None
        self._started = False
        self._iter_idx = 0
        # First call pays the Basilisk-import cost in every worker; later
        # calls are just a state_dict broadcast + rollout. Generous timeout
        # for round 1, tighter steady-state timeout afterwards. Both are
        # configurable so cluster runs with cold SPICE/NFS can extend them.
        self._first_round_timeout = int(first_round_timeout)
        self._steady_timeout = int(steady_timeout)
        # Async-sampling state: tracks whether a dispatch is in flight that
        # gather() still owes a response on.
        self._pending_first_round = False
        self._pending_t_start = None
        self._has_pending = False

    def _spawn_one(self, i, policy):
        task_q = self._ctx.Queue()
        p = self._ctx.Process(
            target=_worker_loop,
            args=(
                i,
                task_q,
                self._result_queue,
                policy,
                self.env_name,
                self.num_samples_per_worker,
                self.episode_len,
                self.state_dim,
                self.action_dim,
                self.envs_per_worker,
            ),
            daemon=True,
        )
        p.start()
        return p, task_q

    def _start_workers(self, policy):
        if self._started:
            return

        # Snapshot policy on CPU so the pickle handed to the spawned workers
        # carries no CUDA tensors. (`to_device` also walks any attached
        # optimizers — without that, Adam state stays on CUDA and unpickle
        # raises cudaErrorInvalidResourceHandle in the worker.)
        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )
        policy.to_device(torch.device("cpu"))

        # Detach the replay buffer (≈200 MB of numpy arrays) for the duration
        # of the spawn. Workers do inference only; they don't need it, and
        # pickling it for every worker is expensive.
        stashed_buf = getattr(policy, "replay_buffer", None)
        if stashed_buf is not None:
            policy.replay_buffer = None

        self._result_queue = self._ctx.Queue()
        self._workers = [None] * self.total_num_worker
        self._task_queues = [None] * self.total_num_worker
        for i in range(self.total_num_worker):
            p, task_q = self._spawn_one(i, policy)
            self._workers[i] = p
            self._task_queues[i] = task_q

        if stashed_buf is not None:
            policy.replay_buffer = stashed_buf
        policy.to_device(original_device)
        self._started = True

    def _respawn_worker(self, i, policy):
        """Replace a dead/stuck worker in slot i."""
        old = self._workers[i]
        if old is not None and old.is_alive():
            old.terminate()
            old.join(timeout=2)
            if old.is_alive():
                old.kill()
                old.join()

        original_device = next(
            (p.device for p in policy.parameters()), torch.device("cpu")
        )
        policy.to_device(torch.device("cpu"))
        stashed_buf = getattr(policy, "replay_buffer", None)
        if stashed_buf is not None:
            policy.replay_buffer = None

        p, task_q = self._spawn_one(i, policy)
        self._workers[i] = p
        self._task_queues[i] = task_q

        if stashed_buf is not None:
            policy.replay_buffer = stashed_buf
        policy.to_device(original_device)

    def dispatch(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """Non-blocking: snapshot policy weights, push a rollout task to
        every worker, return immediately. Pair with `gather()` to receive
        the batch. Used by the async sampling path so the trainer can
        overlap policy updates with worker rollouts."""
        if self._has_pending:
            raise RuntimeError(
                "OnlineSampler.dispatch called while a previous dispatch "
                "is still pending. Call gather() (or drain()) first."
            )

        first_round = not self._started
        if first_round:
            self._start_workers(policy)

        sd = {
            k: v.detach().to("cpu", copy=True)
            for k, v in policy.state_dict().items()
        }
        self._iter_idx += 1
        seed_val = 0 if seed is None else int(seed)

        for q in self._task_queues:
            q.put((sd, seed_val, deterministic, self._iter_idx))

        self._pending_first_round = first_round
        self._pending_t_start = time.time()
        self._has_pending = True

    def gather(self, policy):
        """Blocking: wait for the workers to finish the most recently
        dispatched rollout and return (memory, sample_time). `policy` is
        only used to respawn dead workers on timeout."""
        if not self._has_pending:
            raise RuntimeError(
                "OnlineSampler.gather called with no pending dispatch."
            )

        t_start = self._pending_t_start
        first_round = self._pending_first_round
        timeout = self._first_round_timeout if first_round else self._steady_timeout

        worker_memories = [None] * self.total_num_worker
        worker_timings = [None] * self.total_num_worker
        collected = 0
        while collected < self.total_num_worker:
            try:
                pid, data, timings = self._result_queue.get(timeout=timeout)
                worker_memories[pid] = data
                worker_timings[pid] = timings
                collected += 1
            except Empty:
                missing = [
                    i
                    for i in range(self.total_num_worker)
                    if worker_memories[i] is None
                ]
                print(
                    f"[Warning] Sampler queue timeout after "
                    f"{timeout}s ({collected}/{self.total_num_worker}). "
                    f"Respawning workers {missing} — likely SPICE/Basilisk "
                    f"hang in those processes."
                )
                for i in missing:
                    self._respawn_worker(i, policy)
                break

        t_concat_start = time.time()
        memory = {}
        for wm in worker_memories:
            if wm is None:
                continue
            for key, val in wm.items():
                memory[key] = (
                    np.concatenate((memory[key], val), axis=0)
                    if key in memory
                    else val
                )
        t_concat = time.time() - t_concat_start

        wall = time.time() - t_start
        self._print_timing_breakdown(worker_timings, wall, t_concat)

        self._has_pending = False
        self._pending_first_round = False
        self._pending_t_start = None
        return memory, wall

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
        """Consume and discard any in-flight dispatched batch. Used at the
        end of training to avoid leaking a queue full of results."""
        if not self._has_pending:
            return
        try:
            self.gather(policy)
        except Exception:
            self._has_pending = False
            self._pending_first_round = False
            self._pending_t_start = None

    def collect_samples(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
        use_mp: bool = True,
    ):
        """Synchronous one-shot collect: dispatch + gather. Preserved as the
        default API so the rest of the codebase is unchanged."""
        if use_mp:
            self.dispatch(policy, seed=seed, deterministic=deterministic)
            return self.gather(policy)
        else:
            t_start = time.time()
            memory = {}
            # Sequential fallback: build a single env in-process and roll out.
            self._iter_idx += 1
            seed_val = 0 if seed is None else int(seed)
            original_device = next(
                (p.device for p in policy.parameters()), torch.device("cpu")
            )
            policy.to_device(torch.device("cpu"))
            env = get_env(self.env_name)
            try:
                for i in range(self.total_num_worker):
                    worker_seed = seed_val + i * 9173 + self._iter_idx * 31
                    np.random.seed(worker_seed)
                    random.seed(worker_seed)
                    torch.manual_seed(worker_seed)
                    wm, _ = _run_episodes(
                        policy,
                        env,
                        worker_seed,
                        deterministic,
                        self.num_samples_per_worker,
                        self.episode_len,
                        self.state_dim,
                        self.action_dim,
                    )
                    for key, val in wm.items():
                        memory[key] = (
                            np.concatenate((memory[key], val), axis=0)
                            if key in memory
                            else val
                        )
            finally:
                env.close()
                policy.to_device(original_device)

        return memory, time.time() - t_start

    def close(self):
        if not self._started:
            return
        for q in self._task_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join()
        self._workers = []
        self._task_queues = []
        self._result_queue = None
        self._started = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
