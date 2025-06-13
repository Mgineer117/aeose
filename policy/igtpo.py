import os
import pickle
import time
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import inverse, matmul, transpose
from torch.autograd import grad

from policy.base import Base
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.rl import estimate_advantages

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class IGTPO_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        actor_lr: float = 3e-4,
        batch_size: int = 256,
        eps_clip: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(IGTPO_Learner, self).__init__()

        # constants
        self.name = "IGTPO"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        self.batch_size = batch_size
        self.entropy_scaler = entropy_scaler
        self.gamma = gamma
        self.gae = gae
        self.K = K
        self.l2_reg = l2_reg
        self.target_kl = target_kl
        self.eps_clip = eps_clip

        # trainable networks
        self.actor = actor
        self.actor_lr = actor_lr

        # self.divider = len(list(self.actor.parameters()))

        #
        self.to(self.dtype).to(self.device)

    def forward(self, state: np.ndarray, deterministic: bool = False):
        state = self.preprocess_state(state)
        a, metaData = self.actor(state, deterministic=deterministic)

        return a, {
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "dist": metaData["dist"],
        }

    def learn(self, critic: nn.Module, batch: dict, prefix: str):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # 0. Prepare ingredients
        states = self.preprocess_state(batch["states"])
        actions = self.preprocess_state(batch["actions"])
        rewards = self.preprocess_state(batch["rewards"])
        terminals = self.preprocess_state(batch["terminals"])
        old_logprobs = self.preprocess_state(batch["logprobs"])

        # 1. Compute advantages and returns
        with torch.no_grad():
            values = critic(states)
            advantages, _ = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self.gamma,
                gae=self.gae,
            )

        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # 3. actor Loss
        actor_loss, entropy_loss, clip_fraction, kl_div = self.actor_loss(
            states, actions, old_logprobs, normalized_advantages
        )

        # 4. Total loss
        loss = actor_loss - entropy_loss

        # 5. Compute gradients (example)
        gradients = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        gradients = self.clip_grad_norm(gradients, max_norm=1.0)

        # 6. Manual SGD update (structured, not flat)
        actor_clone = deepcopy(self.actor)
        with torch.no_grad():
            for p, g in zip(actor_clone.parameters(), gradients):
                p -= self.actor_lr * g

        # 7. create a new policy
        new_policy = deepcopy(self)
        new_policy.actor = actor_clone

        # 8. Logging
        actor_grad_norm = torch.sqrt(
            sum(g.pow(2).sum() for g in gradients if g is not None)
        )

        loss_dict = {
            f"{self.name}-{prefix}/loss/loss": loss.item(),
            f"{self.name}-{prefix}/loss/actor_loss": actor_loss.item(),
            f"{self.name}-{prefix}/loss/entropy_loss": entropy_loss.item(),
            f"{self.name}-{prefix}/grad/actor": actor_grad_norm.item(),
            f"{self.name}-{prefix}/analytics/avg_rewards": torch.mean(rewards).item(),
        }
        norm_dict = self.compute_weight_norm(
            [self.actor],
            ["actor"],
            dir=f"{self.name}-{prefix}",
            device=self.device,
        )
        loss_dict.update(norm_dict)

        self.eval()

        timesteps = self.batch_size
        update_time = time.time() - t0

        return (
            loss_dict,
            timesteps,
            update_time,
            new_policy,
            gradients,
            values.mean().cpu().numpy(),
        )

    def actor_loss(
        self,
        mb_states: torch.Tensor,
        mb_actions: torch.Tensor,
        mb_old_logprobs: torch.Tensor,
        mb_advantages: torch.Tensor,
    ):
        _, metaData = self.actor(mb_states)
        logprobs = self.actor.log_prob(metaData["dist"], mb_actions)
        entropy = self.actor.entropy(metaData["dist"])
        ratios = torch.exp(logprobs - mb_old_logprobs)

        surr1 = ratios * mb_advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = self.entropy_scaler * entropy.mean()

        # Compute clip fraction (for logging)
        clip_fraction = torch.mean(
            (torch.abs(ratios - 1) > self.eps_clip).float()
        ).item()

        # Check if KL divergence exceeds target KL for early stopping
        kl_div = torch.mean(mb_old_logprobs - logprobs)

        return actor_loss, entropy_loss, clip_fraction, kl_div

    def clip_grad_norm(self, grads, max_norm, eps=1e-6):
        # Compute total norm
        total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
        clip_coef = max_norm / (total_norm + eps)

        if clip_coef < 1:
            grads = tuple(g * clip_coef for g in grads)

        return grads
