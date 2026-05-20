"""Benchmark serial versus vectorized sampling on the downlink environment.

This script compares compute time for two sampling styles on the real downlink
env:
    1. Serial sampling: one env, one policy forward, one env.step at a time
    2. Vectorized sampling: multiple envs in one process, batched policy forward

Run:
    python tests/bench_sampler.py

The benchmark reports wall time, policy inference time, and env.step time so
you can see the compute-time impact of the sampling method directly.
"""

import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.get_env import get_env
import multiprocessing as mp


class _SimplePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        return self.fc(self.cnn(x))


def _obs_to_tensor(obs, device="cpu"):
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).view(1, -1).to(device)


def _policy_forward(policy, obs_batch, device="cpu"):
    x = torch.from_numpy(np.asarray(obs_batch, dtype=np.float32)).view(len(obs_batch), -1)
    x = x.to(device)
    with torch.no_grad():
        logits = policy(x)
    return torch.argmax(logits, dim=-1).cpu().numpy()


def _build_vector_envs(env_name: str, n_envs: int):
    return [get_env(env_name) for _ in range(n_envs)]


def bench_serial(env_name, policy, total_steps, device="cpu"):
    env = get_env(env_name)
    t_reset = 0.0
    resets = 0
    _t = time.time()
    obs, _ = env.reset(seed=0)
    t_reset += time.time() - _t
    resets += 1
    t0 = time.time()
    t_policy = 0.0
    t_step = 0.0

    steps = 0
    while steps < total_steps:
        _t = time.time()
        action = _policy_forward(policy, [obs], device=device)[0]
        t_policy += time.time() - _t

        _t = time.time()
        obs, _, terminated, truncated, _ = env.step(int(action))
        t_step += time.time() - _t
        steps += 1

        if terminated or truncated:
            _t = time.time()
            obs, _ = env.reset(seed=steps)
            t_reset += time.time() - _t
            resets += 1

    total = time.time() - t0
    try:
        env.close()
    except Exception:
        pass
    return {"total": total, "policy": t_policy, "step": t_step, "reset": t_reset, "steps": steps, "resets": resets}


def bench_vectorized(env_name, policy, total_steps, num_envs, device="cpu"):
    envs = _build_vector_envs(env_name, num_envs)
    obs_batch = []
    t_reset = 0.0
    resets = 0
    for idx, env in enumerate(envs):
        _t = time.time()
        obs, _ = env.reset(seed=idx)
        t_reset += time.time() - _t
        resets += 1
        obs_batch.append(obs)

    t0 = time.time()
    t_policy = 0.0
    t_step = 0.0
    steps = 0

    while steps < total_steps:
        _t = time.time()
        actions = _policy_forward(policy, obs_batch, device=device)
        t_policy += time.time() - _t

        next_obs_batch = []
        _t = time.time()
        for idx, env in enumerate(envs):
            if steps >= total_steps:
                break
            obs, _, terminated, truncated, _ = env.step(int(actions[idx]))
            steps += 1
            if terminated or truncated:
                _rt = time.time()
                obs, _ = env.reset(seed=steps + idx)
                t_reset += time.time() - _rt
                resets += 1
            next_obs_batch.append(obs)
        t_step += time.time() - _t
        obs_batch = next_obs_batch if next_obs_batch else obs_batch

    total = time.time() - t0
    for env in envs:
        try:
            env.close()
        except Exception:
            pass
    return {"total": total, "policy": t_policy, "step": t_step, "reset": t_reset, "steps": steps, "resets": resets}


def _forkserver_worker(conn, env_name, seed_offset=0):
    try:
        env = get_env(env_name)
    except Exception as exc:
        conn.send({'error': str(exc)})
        conn.close()
        return

    t_step = 0.0
    t_reset = 0.0
    n_resets = 0

    # initial reset
    _t = time.time()
    try:
        obs, _ = env.reset(seed=seed_offset)
    except Exception:
        obs = None
    t_reset += time.time() - _t
    n_resets += 1
    conn.send(('obs', obs))

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if not msg:
            continue
        cmd = msg[0]
        if cmd == 'act':
            action = msg[1]
            _t = time.time()
            try:
                obs, _, term, trunc, _ = env.step(int(action))
            except Exception:
                obs = None
                term = False
                trunc = True
            t_step += time.time() - _t
            if term or trunc:
                _rt = time.time()
                try:
                    obs, _ = env.reset(seed=seed_offset + n_resets)
                except Exception:
                    obs = None
                t_reset += time.time() - _rt
                n_resets += 1
            conn.send(('obs', obs, bool(term), bool(trunc)))
        elif cmd == 'close':
            break

    try:
        env.close()
    except Exception:
        pass

    # send back timings
    try:
        conn.send({'step': t_step, 'reset': t_reset, 'n_resets': n_resets})
    except Exception:
        pass
    conn.close()


def _spawn_worker(conn, env_name, seed_offset=0):
    """Spawn worker process for benchmarking (identical to forkserver but using spawn context)."""
    try:
        env = get_env(env_name)
    except Exception as exc:
        conn.send({'error': str(exc)})
        conn.close()
        return

    t_step = 0.0
    t_reset = 0.0
    n_resets = 0

    # initial reset
    _t = time.time()
    try:
        obs, _ = env.reset(seed=seed_offset)
    except Exception:
        obs = None
    t_reset += time.time() - _t
    n_resets += 1
    conn.send(('obs', obs))

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if not msg:
            continue
        cmd = msg[0]
        if cmd == 'act':
            action = msg[1]
            _t = time.time()
            try:
                obs, _, term, trunc, _ = env.step(int(action))
            except Exception:
                obs = None
                term = False
                trunc = True
            t_step += time.time() - _t
            if term or trunc:
                _rt = time.time()
                try:
                    obs, _ = env.reset(seed=seed_offset + n_resets)
                except Exception:
                    obs = None
                t_reset += time.time() - _rt
                n_resets += 1
            conn.send(('obs', obs, bool(term), bool(trunc)))
        elif cmd == 'close':
            break

    try:
        env.close()
    except Exception:
        pass

    # send back timings
    try:
        conn.send({'step': t_step, 'reset': t_reset, 'n_resets': n_resets})
    except Exception:
        pass
    conn.close()


def bench_spawn(env_name, policy, total_steps, num_envs, device="cpu"):
    """Benchmark using spawn multiprocessing (more portable across platforms than forkserver)."""
    ctx = mp.get_context('spawn')
    conns = []
    procs = []
    for i in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_spawn_worker, args=(child_conn, env_name, i))
        p.daemon = True
        p.start()
        child_conn.close()
        conns.append(parent_conn)
        procs.append(p)

    # receive initial observations
    obs_batch = []
    for conn in conns:
        msg = conn.recv()
        if isinstance(msg, dict) and msg.get('error'):
            raise RuntimeError(f"Worker error: {msg['error']}")
        if msg[0] == 'obs':
            obs_batch.append(msg[1])

    t0 = time.time()
    t_policy = 0.0
    t_roundtrip = 0.0
    steps = 0

    while steps < total_steps:
        _t = time.time()
        actions = _policy_forward(policy, obs_batch, device=device)
        t_policy += time.time() - _t

        # send actions and collect responses
        _t = time.time()
        for idx, conn in enumerate(conns):
            if steps >= total_steps:
                break
            conn.send(('act', int(actions[idx])))
        # gather
        next_obs = []
        for conn in conns:
            if steps >= total_steps:
                # drain any remaining replies
                try:
                    _ = conn.recv()
                except Exception:
                    pass
                continue
            try:
                msg = conn.recv()
            except Exception:
                msg = ('obs', None)
            if msg[0] == 'obs':
                # could be ( 'obs', obs ) or ('obs', obs, term, trunc)
                if len(msg) >= 4:
                    obs, term, trunc = msg[1], msg[2], msg[3]
                else:
                    obs = msg[1]
                    term = False
                    trunc = False
                next_obs.append(obs)
                steps += 1
        t_roundtrip += time.time() - _t
        obs_batch = next_obs if next_obs else obs_batch

    total = time.time() - t0

    # close workers and collect timings
    worker_timings = []
    for conn, p in zip(conns, procs):
        try:
            conn.send(('close',))
        except Exception:
            pass
    for conn, p in zip(conns, procs):
        try:
            t = conn.recv()
            if isinstance(t, dict):
                worker_timings.append(t)
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=1.0)
        except Exception:
            pass

    t_step = sum(w.get('step', 0.0) for w in worker_timings)
    t_reset = sum(w.get('reset', 0.0) for w in worker_timings)

    return {"total": total, "policy": t_policy, "roundtrip": t_roundtrip, "step": t_step, "reset": t_reset, "steps": steps}


def bench_forkserver(env_name, policy, total_steps, num_envs, device="cpu"):
    ctx = mp.get_context('forkserver')
    conns = []
    procs = []
    for i in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_forkserver_worker, args=(child_conn, env_name, i))
        p.daemon = True
        p.start()
        child_conn.close()
        conns.append(parent_conn)
        procs.append(p)

    # receive initial observations
    obs_batch = []
    for conn in conns:
        msg = conn.recv()
        if isinstance(msg, dict) and msg.get('error'):
            raise RuntimeError(f"Worker error: {msg['error']}")
        if msg[0] == 'obs':
            obs_batch.append(msg[1])

    t0 = time.time()
    t_policy = 0.0
    t_roundtrip = 0.0
    steps = 0

    while steps < total_steps:
        _t = time.time()
        actions = _policy_forward(policy, obs_batch, device=device)
        t_policy += time.time() - _t

        # send actions and collect responses
        _t = time.time()
        for idx, conn in enumerate(conns):
            if steps >= total_steps:
                break
            conn.send(('act', int(actions[idx])))
        # gather
        next_obs = []
        for conn in conns:
            if steps >= total_steps:
                # drain any remaining replies
                try:
                    _ = conn.recv()
                except Exception:
                    pass
                continue
            try:
                msg = conn.recv()
            except Exception:
                msg = ('obs', None)
            if msg[0] == 'obs':
                # could be ( 'obs', obs ) or ('obs', obs, term, trunc)
                if len(msg) >= 4:
                    obs, term, trunc = msg[1], msg[2], msg[3]
                else:
                    obs = msg[1]
                    term = False
                    trunc = False
                next_obs.append(obs)
                steps += 1
        t_roundtrip += time.time() - _t
        obs_batch = next_obs if next_obs else obs_batch

    total = time.time() - t0

    # close workers and collect timings
    worker_timings = []
    for conn, p in zip(conns, procs):
        try:
            conn.send(('close',))
        except Exception:
            pass
    for conn, p in zip(conns, procs):
        try:
            t = conn.recv()
            if isinstance(t, dict):
                worker_timings.append(t)
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=1.0)
        except Exception:
            pass

    t_step = sum(w.get('step', 0.0) for w in worker_timings)
    t_reset = sum(w.get('reset', 0.0) for w in worker_timings)

    return {"total": total, "policy": t_policy, "roundtrip": t_roundtrip, "step": t_step, "reset": t_reset, "steps": steps}


def bench_async_vectorized(env_name, policy, total_steps, num_envs, device="cpu"):
    """Benchmark using async vectorization (true parallelization with send-all then recv-all)."""
    ctx = mp.get_context('forkserver')
    conns = []
    procs = []
    for i in range(num_envs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(target=_spawn_worker, args=(child_conn, env_name, i))  # Use same worker protocol
        p.daemon = True
        p.start()
        child_conn.close()
        conns.append(parent_conn)
        procs.append(p)

    # receive initial observations
    obs_batch = []
    for conn in conns:
        msg = conn.recv()
        if isinstance(msg, dict) and msg.get('error'):
            raise RuntimeError(f"Worker error: {msg['error']}")
        if msg[0] == 'obs':
            obs_batch.append(msg[1])

    t0 = time.time()
    t_policy = 0.0
    t_roundtrip = 0.0
    steps = 0

    while steps < total_steps:
        _t = time.time()
        actions = _policy_forward(policy, obs_batch, device=device)
        t_policy += time.time() - _t

        # KEY DIFFERENCE: send ALL actions to workers simultaneously (non-blocking)
        _t = time.time()
        for idx, conn in enumerate(conns):
            if steps >= total_steps:
                break
            conn.send(('act', int(actions[idx])))
        
        # Then collect ALL results in parallel (workers can run while we collect)
        next_obs = []
        for conn in conns:
            if steps >= total_steps:
                try:
                    _ = conn.recv()
                except Exception:
                    pass
                continue
            try:
                msg = conn.recv()
            except Exception:
                msg = ('obs', None)
            if msg[0] == 'obs':
                if len(msg) >= 4:
                    obs, term, trunc = msg[1], msg[2], msg[3]
                else:
                    obs = msg[1]
                    term = False
                    trunc = False
                next_obs.append(obs)
                steps += 1
        t_roundtrip += time.time() - _t
        obs_batch = next_obs if next_obs else obs_batch

    total = time.time() - t0

    # close workers and collect timings
    worker_timings = []
    for conn, p in zip(conns, procs):
        try:
            conn.send(('close',))
        except Exception:
            pass
    for conn, p in zip(conns, procs):
        try:
            t = conn.recv()
            if isinstance(t, dict):
                worker_timings.append(t)
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=1.0)
        except Exception:
            pass

    t_step = sum(w.get('step', 0.0) for w in worker_timings)
    t_reset = sum(w.get('reset', 0.0) for w in worker_timings)

    return {"total": total, "policy": t_policy, "roundtrip": t_roundtrip, "step": t_step, "reset": t_reset, "steps": steps}


def main():
    ENV_NAME = "downlink"
    TOTAL_STEPS = 128
    NUM_ENVS = 4
    NUM_TRIALS = 10

    print("=" * 70)
    print("BENCHMARK: serial vs vectorized sampling on downlink")
    print("=" * 70)
    print(f"  Env:            {ENV_NAME}")
    print(f"  Total steps:    {TOTAL_STEPS}")
    print(f"  Vector envs:    {NUM_ENVS}")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"  Device:         {device}")
    print()

    probe_env = get_env(ENV_NAME)
    obs, _ = probe_env.reset(seed=0)
    obs_dim = int(np.asarray(obs).size)
    action_dim = int(probe_env.action_space.n)
    try:
        probe_env.close()
    except Exception:
        pass

    policy = _SimplePolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256)
    policy = policy.to(device)
    policy.eval()

    print("Warming up...")
    _ = _policy_forward(policy, [obs], device=device)
    print()

    print("─" * 70)
    print("[1] Serial sampling — running multiple trials")
    print("─" * 70)
    serial_runs = []
    for _ in range(NUM_TRIALS):
        r = bench_serial(ENV_NAME, policy, TOTAL_STEPS, device=device)
        serial_runs.append(r)

    serial_totals = np.array([r['total'] for r in serial_runs])
    serial_policies = np.array([r['policy'] for r in serial_runs])
    serial_steps_time = np.array([r['step'] for r in serial_runs])
    serial_resets = np.array([r.get('reset', 0.0) for r in serial_runs])
    serial_steps = np.array([r['steps'] for r in serial_runs])

    serial_sps = serial_steps.sum() / serial_totals.sum()
    print(f"  Trials:              {NUM_TRIALS}")
    print(f"  Total time (mean±std):      {serial_totals.mean():.3f}s ± {serial_totals.std():.3f}s")
    print(f"    ├─ Policy time (mean):     {serial_policies.mean():.3f}s")
    print(f"    ├─ Env.step time (mean):   {serial_steps_time.mean():.3f}s")
    print(f"    └─ Reset time (mean):      {serial_resets.mean():.3f}s")
    print(f"  Steps total:          {serial_steps.sum()}  Steps/sec overall: {serial_sps:.1f}")
    print()

    print("─" * 70)
    print("[2] Vectorized + batched policy inference — running multiple trials")
    print("─" * 70)
    vector_runs = []
    for _ in range(NUM_TRIALS):
        r = bench_vectorized(ENV_NAME, policy, TOTAL_STEPS, NUM_ENVS, device=device)
        vector_runs.append(r)

    vector_totals = np.array([r['total'] for r in vector_runs])
    vector_policies = np.array([r['policy'] for r in vector_runs])
    vector_steps_time = np.array([r['step'] for r in vector_runs])
    vector_resets = np.array([r.get('reset', 0.0) for r in vector_runs])
    vector_steps = np.array([r['steps'] for r in vector_runs])

    vector_sps = vector_steps.sum() / vector_totals.sum()
    print(f"  Trials:              {NUM_TRIALS}")
    print(f"  Total time (mean±std):      {vector_totals.mean():.3f}s ± {vector_totals.std():.3f}s")
    print(f"    ├─ Policy time (mean):     {vector_policies.mean():.3f}s")
    print(f"    ├─ Env.step time (mean):   {vector_steps_time.mean():.3f}s")
    print(f"    └─ Reset time (mean):      {vector_resets.mean():.3f}s")
    print(f"  Steps total:          {vector_steps.sum()}  Steps/sec overall: {vector_sps:.1f}")
    print()

    print("─" * 70)
    print("[3] Spawn worker baseline — parent policy, worker env.step")
    print("─" * 70)
    spawn_runs = []
    for _ in range(NUM_TRIALS):
        r = bench_spawn(ENV_NAME, policy, TOTAL_STEPS, NUM_ENVS, device=device)
        spawn_runs.append(r)

    spawn_totals = np.array([r['total'] for r in spawn_runs])
    spawn_policies = np.array([r['policy'] for r in spawn_runs])
    spawn_roundtrips = np.array([r.get('roundtrip', 0.0) for r in spawn_runs])
    spawn_steps_time = np.array([r['step'] for r in spawn_runs])
    spawn_resets = np.array([r.get('reset', 0.0) for r in spawn_runs])
    spawn_steps = np.array([r['steps'] for r in spawn_runs])

    spawn_sps = spawn_steps.sum() / spawn_totals.sum()
    print(f"  Trials:              {NUM_TRIALS}")
    print(f"  Total time (mean±std):      {spawn_totals.mean():.3f}s ± {spawn_totals.std():.3f}s")
    print(f"    ├─ Policy time (mean):     {spawn_policies.mean():.3f}s")
    print(f"    ├─ Roundtrip (send/recv)    {spawn_roundtrips.mean():.3f}s")
    print(f"    ├─ Env.step time (sum mean):{spawn_steps_time.mean():.3f}s")
    print(f"    └─ Reset time (mean):      {spawn_resets.mean():.3f}s")
    print(f"  Steps total:          {spawn_steps.sum()}  Steps/sec overall: {spawn_sps:.1f}")
    print()

    print("─" * 70)
    print("[4] Forkserver worker baseline — parent policy, worker env.step")
    print("─" * 70)
    fork_runs = []
    for _ in range(NUM_TRIALS):
        r = bench_forkserver(ENV_NAME, policy, TOTAL_STEPS, NUM_ENVS, device=device)
        fork_runs.append(r)

    fork_totals = np.array([r['total'] for r in fork_runs])
    fork_policies = np.array([r['policy'] for r in fork_runs])
    fork_roundtrips = np.array([r.get('roundtrip', 0.0) for r in fork_runs])
    fork_steps_time = np.array([r['step'] for r in fork_runs])
    fork_resets = np.array([r.get('reset', 0.0) for r in fork_runs])
    fork_steps = np.array([r['steps'] for r in fork_runs])

    fork_sps = fork_steps.sum() / fork_totals.sum()
    print(f"  Trials:              {NUM_TRIALS}")
    print(f"  Total time (mean±std):      {fork_totals.mean():.3f}s ± {fork_totals.std():.3f}s")
    print(f"    ├─ Policy time (mean):     {fork_policies.mean():.3f}s")
    print(f"    ├─ Roundtrip (send/recv)    {fork_roundtrips.mean():.3f}s")
    print(f"    ├─ Env.step time (sum mean):{fork_steps_time.mean():.3f}s")
    print(f"    └─ Reset time (mean):      {fork_resets.mean():.3f}s")
    print(f"  Steps total:          {fork_steps.sum()}  Steps/sec overall: {fork_sps:.1f}")
    print()

    print("─" * 70)
    print("[5] Async Vectorized worker — true parallelization (send-all then recv-all)")
    print("─" * 70)
    async_runs = []
    for _ in range(NUM_TRIALS):
        r = bench_async_vectorized(ENV_NAME, policy, TOTAL_STEPS, NUM_ENVS, device=device)
        async_runs.append(r)

    async_totals = np.array([r['total'] for r in async_runs])
    async_policies = np.array([r['policy'] for r in async_runs])
    async_roundtrips = np.array([r.get('roundtrip', 0.0) for r in async_runs])
    async_steps_time = np.array([r['step'] for r in async_runs])
    async_resets = np.array([r.get('reset', 0.0) for r in async_runs])
    async_steps = np.array([r['steps'] for r in async_runs])

    async_sps = async_steps.sum() / async_totals.sum()
    print(f"  Trials:              {NUM_TRIALS}")
    print(f"  Total time (mean±std):      {async_totals.mean():.3f}s ± {async_totals.std():.3f}s")
    print(f"    ├─ Policy time (mean):     {async_policies.mean():.3f}s")
    print(f"    ├─ Roundtrip (send/recv)    {async_roundtrips.mean():.3f}s")
    print(f"    ├─ Env.step time (sum mean):{async_steps_time.mean():.3f}s")
    print(f"    └─ Reset time (mean):      {async_resets.mean():.3f}s")
    print(f"  Steps total:          {async_steps.sum()}  Steps/sec overall: {async_sps:.1f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Method':<42} {'Steps/s':>8} {'Speedup':>8}")
    print(f"  {'─' * 42} {'─' * 8} {'─' * 8}")
    print(f"  {'[1] Serial sampling':<42} {serial_sps:>8.1f} {'1.0x':>8}")
    print(f"  {'[2] Vectorized sampling':<42} {vector_sps:>8.1f} {vector_sps / serial_sps:>7.2f}x")
    print(f"  {'[3] Spawn worker sampling':<42} {spawn_sps:>8.1f} {spawn_sps / serial_sps:>7.2f}x")
    print(f"  {'[4] Forkserver worker sampling':<42} {fork_sps:>8.1f} {fork_sps / serial_sps:>7.2f}x")
    print(f"  {'[5] Async vectorized (true parallel)':<42} {async_sps:>8.1f} {async_sps / serial_sps:>7.2f}x")
    print()

    print("Key takeaway:")
    print("  Vectorized sampling should reduce policy compute time by batching")
    print("  multiple env states in one forward pass on the selected device.")


if __name__ == "__main__":
    main()
