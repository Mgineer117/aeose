import time

import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from policy.base import Base
from policy.layers.ppo_networks import PPO_Actor

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class PD_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        target_actor: PPO_Actor,
        data: torch.Tensor,
        actor_lr: float,
        target_kl: float,
        gamma: float,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        # constants
        self.name = "PD"
        self.device = device

        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim

        # trainable networks
        self.actor = actor
        self.target_actor = target_actor
        self.data = data

        self.minibatch_size = 1024

        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.target_kl = target_kl

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

    def learn(self):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        indices = np.random.choice(len(self.data), self.minibatch_size, replace=False)
        states = self.data[indices].to(self.device)

        # given states, generate distribution by the target_actor
        with torch.no_grad():
            _, infos = self.target_actor(states)
            target_dist = infos["dist"]

        _, infos = self.actor(states)
        dist = infos["dist"]

        # find the kl loss between the distribution
        kl_loss = kl_divergence(target_dist, dist).mean()

        # Update critic parameters
        self.optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.actor],
            ["actor"],
            dir=f"{self.name}",
            device=self.device,
        )
        self.optimizer.step()

        # Logging
        loss_dict = {
            f"{self.name}/loss/kl_loss": kl_loss.item(),
        }
        loss_dict.update(grad_dict)

        update_time = time.time() - t0

        self.eval()

        # return the additional info
        termination = True if kl_loss <= self.target_kl else False
        infos = {"termination": termination}

        return loss_dict, update_time, infos
