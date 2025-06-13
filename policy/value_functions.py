from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.base import Base
from utils.rl import estimate_advantages


class Critic_Learner(Base):
    def __init__(
        self,
        critic: nn.Module,
        critic_lr: float,
        gamma: float,
        gae: float,
        l2_reg: float = 1e-8,
        device=torch.device("cpu"),
    ):
        super(Critic_Learner, self).__init__()

        self.name = "IGTPO"
        self.critic = deepcopy(critic)
        self.num_minibatch = 4
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.device = device

        self.optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        value = self.critic(x)
        return value

    def learn(self, batch: dict):
        self.train()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])

        with torch.no_grad():
            values = self.critic(states)
            _, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        batch_size = states.shape[0]
        minibatch_size = batch_size // self.num_minibatch

        value_loss_list = []
        l2_loss_list = []
        for _ in range(self.num_minibatch):
            indices = torch.randperm(batch_size)[:minibatch_size]
            mb_states, mb_returns = states[indices], returns[indices]

            value_loss, l2_loss = self.critic_loss(mb_states, mb_returns)

            value_loss_list.append(value_loss.item())
            l2_loss_list.append(l2_loss.item())

            loss = value_loss + l2_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.critic],
                [f"critic"],
                dir=f"{self.name}",
                device=self.device,
            )
            self.optimizer.step()

        loss_dict = {
            f"{self.name}/loss/value_loss": np.mean(value_loss_list),
            f"{self.name}/loss/l2_loss": np.mean(l2_loss_list),
        }
        norm_dict = self.compute_weight_norm(
            [self.critic],
            [f"critic"],
            dir=f"{self.name}",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        return loss_dict, values.mean()

    def critic_loss(self, states: torch.Tensor, returns: torch.Tensor):
        values = self.critic(states)
        value_loss = self.mse_loss(values, returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critic.parameters()) * self.l2_reg
        )

        return value_loss, l2_loss


class Critics_Learner(Base):
    def __init__(
        self,
        critic: nn.Module,
        critic_lr: float,
        num: int,
        gamma: float,
        gae: float,
        l2_reg: float = 1e-8,
        device=torch.device("cpu"),
    ):
        super(Critics_Learner, self).__init__()

        self.name = "IGTPO"
        self.critics = nn.ModuleList([deepcopy(critic) for _ in range(num)])
        self.num_minibatch = 4
        self.gamma = gamma
        self.gae = gae
        self.l2_reg = l2_reg
        self.device = device

        self.optimizers = [
            torch.optim.Adam(critic.parameters(), lr=critic_lr)
            for critic in self.critics
        ]

        self.to(self.device)

    def forward(self, x: torch.Tensor, idx: int):
        value = self.critics[idx](x)
        return value

    def learn(self, batch: dict, idx: int, prefix: str):
        self.train()

        # Ingredients: Convert batch data to tensors
        states = self.preprocess_state(batch["states"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])

        with torch.no_grad():
            values = self.critics[idx](states)
            _, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        batch_size = states.shape[0]
        minibatch_size = batch_size // self.num_minibatch

        value_loss_list = []
        l2_loss_list = []
        for _ in range(self.num_minibatch):
            indices = torch.randperm(batch_size)[:minibatch_size]
            mb_states, mb_returns = states[indices], returns[indices]

            value_loss, l2_loss = self.critic_loss(mb_states, mb_returns, idx)

            value_loss_list.append(value_loss.item())
            l2_loss_list.append(l2_loss.item())

            loss = value_loss + l2_loss

            self.optimizers[idx].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[idx].parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.critics[idx]],
                [f"critic"],
                dir=f"{self.name}-{prefix}",
                device=self.device,
            )
            self.optimizers[idx].step()

        loss_dict = {
            f"{self.name}-{prefix}/loss/value_loss": np.mean(value_loss_list),
            f"{self.name}-{prefix}/loss/l2_loss": np.mean(l2_loss_list),
        }
        norm_dict = self.compute_weight_norm(
            [self.critics[idx]],
            [f"critic"],
            dir=f"{self.name}-{prefix}",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(grad_dict)

        return loss_dict

    def critic_loss(self, states: torch.Tensor, returns: torch.Tensor, idx: int):
        values = self.critics[idx](states)
        value_loss = self.mse_loss(values, returns)
        l2_loss = (
            sum(param.pow(2).sum() for param in self.critics[idx].parameters())
            * self.l2_reg
        )

        return value_loss, l2_loss
