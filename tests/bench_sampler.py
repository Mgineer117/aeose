"""Benchmark fork-based sampling versus vectorized batched sampling.

This script simulates the Atari-style pipeline to measure overhead from:
  1. Process spawn/join
  2. mp.Queue serialization of numpy arrays
  3. Per-frame serial CNN encoding in forked workers
  4. Batched encoding in a single process

Run:
  python tests/bench_sampler.py

The point is to compare only two paths:
  - Forked workers: env.step() + encode each frame on CPU in child processes
  - Vectorized: batch env.step() simulation + batched encoding on one device

On CUDA, forked workers cannot safely re-initialize CUDA after fork(), so they
are effectively CPU-only for the encoding step. The vectorized path can use the
GPU for batched inference.
"""

import time
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


class _SimpleEncoder(nn.Module):
    def __init__(self, input_chw=(1, 210, 160), encoder_dim=256):
        super().__init__()
        c, h, w = input_chw
        self.encoder_dim = encoder_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_dim = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(flat_dim, encoder_dim)

    def forward(self, x):
        return self.fc(self.cnn(x))


ENV_STEP_MS = 0.4


def _sim_env_step(frame_shape):
    """Simulate ALE env.step(): produce a new frame plus a small delay."""
    time.sleep(ENV_STEP_MS / 1000.0)
    return np.random.randint(0, 256, size=frame_shape, dtype=np.uint8)


def _encode_single(encoder, raw_frame: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(raw_frame.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        feat = encoder(t)
    return feat.squeeze(0).cpu().numpy()


def _encode_batch(encoder, raw_frames: np.ndarray, device="cpu") -> np.ndarray:
    t = torch.from_numpy(raw_frames.astype(np.float32) / 255.0)
    t = t.unsqueeze(1).to(device)
    with torch.no_grad():
        feat = encoder(t)
    return feat.cpu().numpy()


def _worker(pid, queue, encoder, n_steps, frame_shape):
    encoded = np.zeros((n_steps, encoder.encoder_dim), dtype=np.float32)
    for i in range(n_steps):
        raw = _sim_env_step(frame_shape)
        encoded[i] = _encode_single(encoder, raw)
    queue.put((pid, encoded))


def bench_fork(encoder, total_steps, num_workers, frame_shape):
    steps_per_worker = ceil(total_steps / num_workers)
    t0 = time.time()

    t_spawn = time.time()
    procs, queue = [], mp.Queue()
    for worker_id in range(num_workers):
        proc = mp.Process(
            target=_worker,
            args=(worker_id, queue, encoder, steps_per_worker, frame_shape),
        )
        procs.append(proc)
        proc.start()
    t_spawn = time.time() - t_spawn

    t_collect = time.time()
    results = [None] * num_workers
    collected = 0
    while collected < num_workers:
        try:
            pid, data = queue.get(timeout=120)
            results[pid] = data
            collected += 1
        except Empty:
            break
    t_collect = time.time() - t_collect

    t_join = time.time()
    for proc in procs:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join()
        proc.close()
    queue.close()
    t_join = time.time() - t_join

    total = time.time() - t0
    steps = sum(result.shape[0] for result in results if result is not None)
    return {
        "total": total,
        "spawn": t_spawn,
        "collect": t_collect,
        "join": t_join,
        "steps": steps,
    }


def bench_vectorized(encoder, total_steps, batch_size, frame_shape, device="cpu"):
    enc = encoder.to(device)
    t0 = time.time()

    all_enc = np.zeros((total_steps, enc.encoder_dim), dtype=np.float32)
    step = 0
    while step < total_steps:
        n = min(batch_size, total_steps - step)

        # Simulate one vectorized environment step across N envs.
        time.sleep(ENV_STEP_MS / 1000.0)
        raw = np.random.randint(0, 256, (n, *frame_shape), dtype=np.uint8)

        all_enc[step : step + n] = _encode_batch(enc, raw, device=device)
        step += n

    total = time.time() - t0
    return {"total": total, "steps": step}


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    FRAME_SHAPE = (210, 160)
    TOTAL_STEPS = 2048
    NUM_WORKERS = 4
    ENCODER_DIM = 256

    print("=" * 70)
    print("BENCHMARK: forked workers vs vectorized batched sampling")
    print("=" * 70)
    print(f"  Frame shape:    {FRAME_SHAPE}")
    print(f"  Total steps:    {TOTAL_STEPS}")
    print(f"  Num workers:    {NUM_WORKERS}")
    print(f"  Encoder dim:    {ENCODER_DIM}")
    print(f"  Env step sim:   {ENV_STEP_MS}ms per step")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"  Device:         {device}")
    print()

    encoder = _SimpleEncoder(input_chw=(1, *FRAME_SHAPE), encoder_dim=ENCODER_DIM)
    encoder.eval()

    print("Warming up...")
    _encode_single(encoder.cpu(), np.random.randint(0, 256, FRAME_SHAPE, dtype=np.uint8))
    _encode_batch(encoder.cpu(), np.random.randint(0, 256, (4,) + FRAME_SHAPE, dtype=np.uint8))
    print()

    print("─" * 70)
    print("[1] mp.Process fork: serial env.step + serial encode on CPU")
    print("    Forked workers cannot use CUDA safely after fork().")
    print("─" * 70)
    encoder_cpu = encoder.cpu()
    fork_result = bench_fork(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE)
    fork_sps = fork_result["steps"] / fork_result["total"]
    print(f"  Total time:          {fork_result['total']:.3f}s")
    print(f"    ├─ Process spawn:  {fork_result['spawn']:.3f}s")
    print(f"    ├─ Queue collect:  {fork_result['collect']:.3f}s")
    print(f"    └─ Process join:   {fork_result['join']:.3f}s")
    print(f"  Steps collected:     {fork_result['steps']}")
    print(f"  Steps/sec:           {fork_sps:.0f}")
    print()

    print("─" * 70)
    print("[2] Vectorized + batched encoding")
    print("─" * 70)
    vector_result = bench_vectorized(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE, device=device)
    vector_sps = vector_result["steps"] / vector_result["total"]
    print(f"  Total time:          {vector_result['total']:.3f}s")
    print(f"  Steps/sec:           {vector_sps:.0f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Method':<42} {'Steps/s':>8} {'Speedup':>8}")
    print(f"  {'─' * 42} {'─' * 8} {'─' * 8}")
    print(f"  {'[1] Fork + serial CPU encode':<42} {fork_sps:>8.0f} {'1.0x':>8}")
    print(f"  {'[2] Vectorized + batch encode':<42} {vector_sps:>8.0f} {vector_sps / fork_sps:>7.1f}x")
    print()

    print("Key takeaway:")
    print("  Forked workers keep encoding on CPU after fork, while the vectorized")
    print("  path can batch frames and use the selected device for inference.")