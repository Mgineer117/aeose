import random
import time
from math import ceil

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
        Persistent vectorized sampler.

        The sampler keeps a small batch of envs alive in the same process and
        batches policy inference across them. No fork, no spawn, no forkserver.
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            episode_len=episode_len,
            batch_size=batch_size,
        )

        self.env_name = env_name
        self.envs_per_worker = max(2, int(envs_per_worker))
        self.total_num_worker = 1
        self.num_samples_per_worker = int(batch_size)

        if verbose:
            print("Sampling Parameters:")
            print(f"Vectorized envs:             {self.envs_per_worker}")
            print(f"Samples per call:            {self.num_samples_per_worker}")

        torch.set_num_threads(1)  # avoid CPU oversubscription in parent

        self._envs = [get_env(self.env_name) for _ in range(self.envs_per_worker)]
        self._iter_idx = 0
        self._pending_batch = None
        self._pending_first_round = False
        self._pending_t_start = None
        self._has_pending = False

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

    def collect_samples(
        self,
        policy,
        seed: int | None = None,
        deterministic: bool = False,
        use_mp: bool = True,
    ):
        """Synchronous one-shot collect using the in-process vectorized path."""
        _ = use_mp
        return self._collect_vectorized(policy, seed=seed, deterministic=deterministic)

    def close(self):
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
