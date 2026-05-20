import time

import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from policy.base import Base
from policy.layers.ppo_networks import PPO_Actor


class StateBuffer:
    def __init__(self, state_dim, capacity: int, device: str = "cpu", seed: int = 0):
        self.capacity = max(1, int(capacity))
        self.device = device
        self.storage = torch.zeros((self.capacity,) + tuple(state_dim), dtype=torch.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, states: torch.Tensor | np.ndarray):
        if states is None:
            return 0
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).to(torch.float32)
        elif isinstance(states, torch.Tensor):
            states = states.detach().cpu().to(torch.float32)
        else:
            raise ValueError("Unsupported state type for StateBuffer.add().")

        if states.ndim == len(self.storage.shape) - 1:
            states = states.unsqueeze(0)
        states = states.view(states.shape[0], *self.storage.shape[1:])

        for idx in range(states.shape[0]):
            self.storage[self.ptr] = states[idx]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        return states.shape[0]

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty state buffer.")
        replace = self.size < batch_size
        indices = self.rng.choice(self.size, size=batch_size, replace=replace)
        return self.storage[indices].to(self.device)


class PD_Learner(Base):
    def __init__(
        self,
        actor: PPO_Actor,
        target_actor: PPO_Actor,
        teacher_states: torch.Tensor,
        actor_lr: float,
        target_kl: float,
        gamma: float,
        buffer_mode: str = "teacher",
        teacher_buffer_capacity: int = 50000,
        student_buffer_capacity: int = 50000,
        minibatch_size: int = 1024,
        mixed_student_ratio: float = 0.5,
        mixed_update: str = "dagger",
        seed: int = 0,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        self.name = "PD"
        self.device = device
        self.state_dim = actor.state_dim
        self.action_dim = actor.action_dim
        self.gamma = gamma
        self.target_kl = target_kl
        self.buffer_mode = buffer_mode
        self.minibatch_size = int(minibatch_size)
        self.mixed_student_ratio = float(np.clip(mixed_student_ratio, 0.0, 1.0))
        self.mixed_update = mixed_update
        self.rng = np.random.default_rng(seed)

        self.actor = actor
        self.target_actor = target_actor

        teacher_capacity = max(int(teacher_buffer_capacity), len(teacher_states))
        self.teacher_buffer = StateBuffer(
            state_dim=self.state_dim,
            capacity=teacher_capacity,
            device=self.device,
            seed=seed,
        )
        self.teacher_buffer.add(teacher_states)

        self.student_buffer = StateBuffer(
            state_dim=self.state_dim,
            capacity=max(1, int(student_buffer_capacity)),
            device=self.device,
            seed=seed + 1,
        )

        mixed_capacity = max(
            1, min(self.teacher_buffer.capacity + self.student_buffer.capacity, 200000)
        )
        self.mixed_buffer = StateBuffer(
            state_dim=self.state_dim,
            capacity=mixed_capacity,
            device=self.device,
            seed=seed + 2,
        )
        self.mixed_buffer.add(teacher_states)

        self.optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=actor_lr)
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

    def ingest_student_states(self, states: torch.Tensor | np.ndarray):
        if states is None:
            return 0
        added = self.student_buffer.add(states)
        if self.buffer_mode != "mixed":
            return added

        if self.mixed_update == "dagger":
            self.mixed_buffer.add(states)
            return added

        states_t = states
        if isinstance(states_t, np.ndarray):
            states_t = torch.from_numpy(states_t).to(torch.float32)
        else:
            states_t = states_t.detach().cpu().to(torch.float32)
        if states_t.ndim == len(self.teacher_buffer.storage.shape) - 1:
            states_t = states_t.unsqueeze(0)

        for idx in range(states_t.shape[0]):
            if len(self.mixed_buffer) < self.mixed_buffer.capacity:
                self.mixed_buffer.add(states_t[idx : idx + 1])
            else:
                replace_idx = int(self.rng.integers(0, self.mixed_buffer.capacity))
                self.mixed_buffer.storage[replace_idx] = states_t[idx]
        self.mixed_buffer.size = min(
            max(self.mixed_buffer.size, states_t.shape[0]), self.mixed_buffer.capacity
        )
        return added

    def _sample_teacher_batch(self, batch_size: int) -> torch.Tensor:
        return self.teacher_buffer.sample(batch_size)

    def _sample_student_batch(self, batch_size: int) -> torch.Tensor:
        if len(self.student_buffer) == 0:
            return self.teacher_buffer.sample(batch_size)
        return self.student_buffer.sample(batch_size)

    def _sample_mixed_batch(self, batch_size: int) -> torch.Tensor:
        teacher_n = batch_size
        student_n = 0

        if len(self.student_buffer) > 0:
            student_n = int(round(batch_size * self.mixed_student_ratio))
            student_n = min(student_n, batch_size)
            teacher_n = batch_size - student_n

        parts = []
        if self.mixed_update == "random":
            if len(self.mixed_buffer) > 0:
                return self.mixed_buffer.sample(batch_size)

        if teacher_n > 0:
            parts.append(self.teacher_buffer.sample(teacher_n))
        if student_n > 0:
            parts.append(self.student_buffer.sample(student_n))

        if not parts:
            return self.teacher_buffer.sample(batch_size)

        if len(parts) == 1:
            return parts[0]

        states = torch.cat(parts, dim=0)
        perm = torch.randperm(states.size(0), device=states.device)
        return states[perm]

    def _sample_training_batch(self):
        batch_size = self.minibatch_size
        if self.buffer_mode == "teacher":
            return self._sample_teacher_batch(batch_size), "teacher"
        if self.buffer_mode == "student":
            source = "student" if len(self.student_buffer) > 0 else "teacher_fallback"
            return self._sample_student_batch(batch_size), source
        return self._sample_mixed_batch(batch_size), "mixed"

    def learn(self):
        self.train()
        t0 = time.time()

        states, batch_source = self._sample_training_batch()

        with torch.no_grad():
            _, infos = self.target_actor(states)
            target_dist = infos["dist"]

        _, infos = self.actor(states)
        dist = infos["dist"]
        kl_loss = kl_divergence(target_dist, dist).mean()

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

        loss_dict = {
            f"{self.name}/loss/kl_loss": kl_loss.item(),
            f"{self.name}/buffer/teacher_size": float(len(self.teacher_buffer)),
            f"{self.name}/buffer/student_size": float(len(self.student_buffer)),
            f"{self.name}/buffer/mixed_size": float(len(self.mixed_buffer)),
            f"{self.name}/buffer/student_ratio": self.mixed_student_ratio,
            f"{self.name}/buffer/source_is_student": float(batch_source == "student"),
            f"{self.name}/buffer/source_is_teacher": float(batch_source == "teacher"),
            f"{self.name}/buffer/source_is_mixed": float(batch_source == "mixed"),
        }
        loss_dict.update(grad_dict)

        update_time = time.time() - t0
        self.eval()

        termination = True if kl_loss <= self.target_kl else False
        infos = {"termination": termination, "batch_source": batch_source}
        return loss_dict, update_time, infos
