import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.base import Base
from policy.layers.building_blocks import MLP


class TD3_Actor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        action_scale: tuple,
        activation: nn.Module = nn.ReLU(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)  # modify based on Base's signature

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.low_action_scale = torch.from_numpy(action_scale[0]).to(self.device)
        self.high_action_scale = torch.from_numpy(action_scale[1]).to(self.device)

        self.action_max = torch.max(
            torch.stack(
                [
                    torch.abs(self.low_action_scale).max(),
                    torch.abs(self.high_action_scale).max(),
                ]
            )
        )

        self.is_discrete = False

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )

        self.device = device
        self._dummy = torch.tensor(1e-8)
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool,
    ):
        logits = self.model(state)
        action = self.action_max * F.tanh(logits)

        if not deterministic:
            # Add small exploration noise for action selection (not training!)
            mean = torch.zeros(action.size()).to(self.device)
            var = 0.1 * torch.ones(action.size()).to(self.device)
            noise = torch.normal(mean, var)
            action += noise

        action = torch.min(action, self.high_action_scale)
        action = torch.max(action, self.low_action_scale)

        return action, {
            "dist": self._dummy,
            "probs": self._dummy,
            "logprobs": self._dummy,
            "entropy": self._dummy,
        }


class TD3_Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.input_dim = np.prod(state_dim) + np.prod(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = MLP(
            self.input_dim,
            hidden_dim,
            1,
            activation=activation,
            initialization="critic",
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value


class TD3_Actor_From_Critic(nn.Module):
    """
    Treats a TD3 critic as a policy by evaluating discrete actions
    and selecting the best (or sampling epsilon-greedy if stochastic).

    Args:
        TD3_Critic (nn.Module): inherited Q-network for state-action value
    """

    def __init__(
        self,
        critic: TD3_Critic,
        epsilon: float = 0.1,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.input_dim = critic.input_dim
        self.state_dim = critic.state_dim
        self.action_dim = critic.action_dim

        self.is_discrete = True

        self.critic = critic

        # Generate all one-hot discrete actions once
        self.register_buffer(
            "all_actions",
            F.one_hot(
                torch.arange(self.action_dim), num_classes=self.action_dim
            ).float(),
        )

        self._dummy = torch.tensor(1e-8)

    def forward(self, state: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        Choose an action using the critic Q-values.

        Args:
            state (torch.Tensor): [batch_size, state_dim]
            deterministic (bool): use argmax if True, else epsilon-greedy

        Returns:
            torch.Tensor: [batch_size, action_dim] one-hot encoded actions
        """
        batch_size = state.shape[0]

        # Repeat state for each action
        state_expanded = state.unsqueeze(1).expand(
            -1, self.action_dim, -1
        )  # [B, A, state_dim]
        actions = self.all_actions.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, A, action_dim]

        # Concatenate and evaluate Q-values
        sa_input = torch.cat(
            [state_expanded, actions], dim=-1
        )  # [B, A, state_dim + action_dim]
        q_input = sa_input.view(-1, self.input_dim)
        q_values = self.critic(q_input).view(batch_size, self.action_dim)  # [B, A]

        if deterministic:
            best_idx = torch.argmax(q_values, dim=1)
        else:
            rand_idx = torch.randint(
                0, self.action_dim, (batch_size,), device=state.device
            )
            greedy_idx = torch.argmax(q_values, dim=1)
            coin = torch.rand(batch_size, device=state.device) < self.epsilon
            best_idx = torch.where(coin, rand_idx, greedy_idx)

        # Convert selected index to one-hot action
        return F.one_hot(best_idx, num_classes=self.action_dim).float(), {
            "dist": self._dummy,
            "probs": self._dummy,
            "logprobs": self._dummy,
            "entropy": self._dummy,
        }
