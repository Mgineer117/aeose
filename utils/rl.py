import torch
import torch.nn as nn

from utils.sampler import OnlineSampler


def estimate_advantages(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate advantages and returns using Generalized Advantage Estimation (GAE),
    while keeping all operations on the original device.

    Args:
        rewards (Tensor): Reward at each timestep, shape [T, 1]
        terminals (Tensor): Binary terminal indicators (1 if done), shape [T, 1]
        values (Tensor): Value function estimates, shape [T, 1]
        gamma (float): Discount factor.
        gae (float): GAE lambda.

    Returns:
        advantages (Tensor): Estimated advantages, shape [T, 1]
        returns (Tensor): Estimated returns (value targets), shape [T, 1]
    """
    device = rewards.device  # Infer device from input tensor

    T = rewards.size(0)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_value = torch.tensor(0.0, device=device)
    prev_advantage = torch.tensor(0.0, device=device)

    for t in reversed(range(T)):
        non_terminal = 1.0 - terminals[t]
        deltas[t] = rewards[t] + gamma * prev_value * non_terminal - values[t]
        advantages[t] = deltas[t] + gamma * gae * prev_advantage * non_terminal

        prev_value = values[t]
        prev_advantage = advantages[t]

    returns = values + advantages
    return advantages, returns
