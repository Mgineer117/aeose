import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


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
