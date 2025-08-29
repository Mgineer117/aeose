import time

import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from policy.base import Base
from policy.layers.ppo_networks import PPO_Actor

# from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.replay_buffer import ReplayBuffer
from utils.rl import estimate_advantages

# from models.layers.ppo_networks import PPO_Policy, PPO_Critic


class PD_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        target_actor: PPO_Actor,
        actor_lr: float = 3e-4,
        gamma: float = 0.99,
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

        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma

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

    def learn(self, replay_buffer: ReplayBuffer):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        states, _, _, _, _ = replay_buffer.sample()

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
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
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

        return loss_dict, update_time
