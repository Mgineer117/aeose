import torch
import torch.nn as nn
import numpy as np

from utils.sampler import OnlineSampler


class RunningMeanStd:
    """Numerically-stable running mean / variance for observation normalization.

    Usage:
        rms = RunningMeanStd(shape)
        rms.update(batch)   # batch: (N, dim) numpy or torch
        x_norm = (x - rms.mean) / sqrt(rms.var + eps)
    """

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x):
        # Accept torch tensor or numpy array
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.array(x)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]

        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count



def estimate_advantages(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GAE with separate handling of terminations and truncations.

    - Terminations are *real* episode ends: V(s') is **not** bootstrapped.
    - Truncations are artificial cutoffs (env time_limit or rollout buffer
      full): V(s') **is** bootstrapped, but we still reset the GAE
      accumulator because s' belongs to a different trajectory.

    Masks:
        not_term   = 1 - terminations  -> used on the value bootstrap term
        not_done   = 1 - (terminations | truncations) -> used on the GAE
                    recurrence (prev_advantage carry).

    Shapes:
        rewards / terminations / truncations / values: [T, 1]
    """
    device = rewards.device

    T = rewards.size(0)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_value = torch.tensor(0.0, device=device)
    prev_advantage = torch.tensor(0.0, device=device)

    # done = OR of terminations and truncations (clamped so float inputs
    # behave like booleans).
    dones = torch.clamp(terminations + truncations, max=1.0)

    for t in reversed(range(T)):
        not_term = 1.0 - terminations[t]
        not_done = 1.0 - dones[t]
        deltas[t] = rewards[t] + gamma * prev_value * not_term - values[t]
        advantages[t] = deltas[t] + gamma * gae * prev_advantage * not_done

        prev_value = values[t]
        prev_advantage = advantages[t]

    returns = values + advantages
    return advantages, returns
