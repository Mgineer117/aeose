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

import numpy as np
import torch
import torch.nn as nn

from utils.get_env import get_env


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
    obs, _ = env.reset(seed=0)
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
            obs, _ = env.reset(seed=steps)

    total = time.time() - t0
    env.close()
    return {"total": total, "policy": t_policy, "step": t_step, "steps": steps}


def bench_vectorized(env_name, policy, total_steps, num_envs, device="cpu"):
    envs = _build_vector_envs(env_name, num_envs)
    obs_batch = []
    for idx, env in enumerate(envs):
        obs, _ = env.reset(seed=idx)
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
                obs, _ = env.reset(seed=steps + idx)
            next_obs_batch.append(obs)
        t_step += time.time() - _t
        obs_batch = next_obs_batch if next_obs_batch else obs_batch

    total = time.time() - t0
    for env in envs:
        env.close()
    return {"total": total, "policy": t_policy, "step": t_step, "steps": steps}


if __name__ == "__main__":
    ENV_NAME = "downlink"
    TOTAL_STEPS = 128
    NUM_ENVS = 4

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
    probe_env.close()

    policy = _SimplePolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256)
    policy.eval()

    print("Warming up...")
    _ = _policy_forward(policy, [obs], device=device)
    print()

    print("─" * 70)
    print("[1] Serial sampling")
    print("─" * 70)
    serial_result = bench_serial(ENV_NAME, policy, TOTAL_STEPS, device=device)
    serial_sps = serial_result["steps"] / serial_result["total"]
    print(f"  Total time:          {serial_result['total']:.3f}s")
    print(f"    ├─ Policy time:    {serial_result['policy']:.3f}s")
    print(f"    └─ Env.step time:  {serial_result['step']:.3f}s")
    print(f"  Steps/sec:           {serial_sps:.0f}")
    print()

    print("─" * 70)
    print("[2] Vectorized + batched policy inference")
    print("─" * 70)
    vector_result = bench_vectorized(ENV_NAME, policy, TOTAL_STEPS, NUM_ENVS, device=device)
    vector_sps = vector_result["steps"] / vector_result["total"]
    print(f"  Total time:          {vector_result['total']:.3f}s")
    print(f"    ├─ Policy time:    {vector_result['policy']:.3f}s")
    print(f"    └─ Env.step time:  {vector_result['step']:.3f}s")
    print(f"  Steps/sec:           {vector_sps:.0f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Method':<42} {'Steps/s':>8} {'Speedup':>8}")
    print(f"  {'─' * 42} {'─' * 8} {'─' * 8}")
    print(f"  {'[1] Serial sampling':<42} {serial_sps:>8.0f} {'1.0x':>8}")
    print(f"  {'[2] Vectorized sampling':<42} {vector_sps:>8.0f} {vector_sps / serial_sps:>7.1f}x")
    print()

    print("Key takeaway:")
    print("  Vectorized sampling should reduce policy compute time by batching")
    print("  multiple env states in one forward pass on the selected device.")