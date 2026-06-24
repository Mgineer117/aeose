import numpy as np
import torch
import json


class ReplayBuffer:
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        dtype=torch.float32,
        device=torch.device("cpu"),
        retention_strategy: str = "fifo",
        recent_fraction: float = 0.0,
        diversity_weight: float = 1.0,
        recency_weight: float = 0.35,
        random_swap_prob: float = 0.02,
        candidate_pool_size: int = 256,
        recency_horizon: int = 4096,
        projection_dim: int = 32,
        seed: int = 0,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terminal = np.zeros((buffer_size, 1), dtype=np.float32)
        self.truncation = np.zeros((buffer_size, 1), dtype=np.float32)

        self.dtype = dtype
        self.device = device
        self.retention_strategy = retention_strategy
        self.recent_fraction = float(np.clip(recent_fraction, 0.0, 1.0))
        self.diversity_weight = float(diversity_weight)
        self.recency_weight = float(recency_weight)
        self.random_swap_prob = float(np.clip(random_swap_prob, 0.0, 1.0))
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.recency_horizon = max(1, int(recency_horizon))
        self.projection_dim = max(4, int(projection_dim))
        self.rng = np.random.default_rng(seed)
        self.insert_counter = 0

        self.embedding = np.zeros((buffer_size, self.projection_dim), dtype=np.float32)
        self.novelty = np.zeros(buffer_size, dtype=np.float32)
        self.insert_id = np.full(buffer_size, -1, dtype=np.int64)
        self._proj_matrix = None

        self.recent_capacity = 0
        self.recent_size = 0
        self.recent_ptr = 0
        self.diverse_start = 0
        self.diverse_capacity = buffer_size
        self.diverse_size = 0
        self.diverse_ptr = 0

        if self.retention_strategy == "diverse_recent":
            if self.buffer_size <= 1:
                self.recent_capacity = self.buffer_size
            else:
                proposed_recent = int(round(self.buffer_size * self.recent_fraction))
                self.recent_capacity = min(self.buffer_size - 1, max(1, proposed_recent))
            self.diverse_start = self.recent_capacity
            self.diverse_capacity = self.buffer_size - self.recent_capacity
        self._refresh_size()

    def pre_process(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def append(self, state, action, next_state, reward, terminal, truncation=0.0):
        state = self.pre_process(state)
        action = self.pre_process(action)
        next_state = self.pre_process(next_state)
        reward = self.pre_process(reward)
        terminal = self.pre_process(terminal)
        truncation = self.pre_process(truncation)

        if self.retention_strategy == "diverse_recent":
            self._append_diverse_recent(state, action, next_state, reward, terminal, truncation)
            return

        self._write_at(
            self.ptr,
            state,
            action,
            next_state,
            reward,
            terminal,
            truncation=truncation,
            update_stats=False,
        )
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def _refresh_size(self):
        if self.retention_strategy == "diverse_recent":
            self.size = self.recent_size + self.diverse_size
        else:
            self.size = min(self.size, self.buffer_size)

    def _sanitize_array(self, x):
        arr = np.asarray(x, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _transition_feature(self, state, action, next_state, reward, terminal):
        state = self._sanitize_array(state).reshape(-1)
        action = self._sanitize_array(action).reshape(-1)
        next_state = self._sanitize_array(next_state).reshape(-1)
        reward = self._sanitize_array(reward).reshape(-1)
        terminal = self._sanitize_array(terminal).reshape(-1)
        delta = next_state - state
        return np.concatenate([state, action, delta, reward, terminal], axis=0)

    def _embed_transition(self, state, action, next_state, reward, terminal):
        feature = self._transition_feature(state, action, next_state, reward, terminal)
        if self._proj_matrix is None or self._proj_matrix.shape[0] != feature.size:
            scale = np.sqrt(max(1, feature.size))
            self._proj_matrix = (
                self.rng.standard_normal((feature.size, self.projection_dim)).astype(np.float32)
                / scale
            )
        embedding = feature @ self._proj_matrix
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def _write_at(
        self,
        idx,
        state,
        action,
        next_state,
        reward,
        terminal,
        truncation=0.0,
        embedding=None,
        novelty=None,
        insert_id=None,
        update_stats=True,
    ):
        self.state[idx] = self._sanitize_array(state)
        self.action[idx] = self._sanitize_array(action)
        self.next_state[idx] = self._sanitize_array(next_state)
        self.reward[idx] = self._sanitize_array(reward)
        self.terminal[idx] = self._sanitize_array(terminal)
        self.truncation[idx] = self._sanitize_array(truncation)
        if update_stats or embedding is not None:
            if embedding is None:
                embedding = self._embed_transition(state, action, next_state, reward, terminal)
            if novelty is None:
                novelty = 0.0
            if insert_id is None:
                insert_id = self.insert_counter
            self.embedding[idx] = embedding
            self.novelty[idx] = float(novelty)
            self.insert_id[idx] = int(insert_id)

    def _recent_indices(self):
        if self.recent_size == 0:
            return np.empty(0, dtype=np.int64)
        if self.recent_size < self.recent_capacity:
            return np.arange(self.recent_size, dtype=np.int64)
        head = np.arange(self.recent_ptr, self.recent_capacity, dtype=np.int64)
        tail = np.arange(0, self.recent_ptr, dtype=np.int64)
        return np.concatenate([head, tail], axis=0)

    def _diverse_indices(self):
        if self.diverse_size == 0:
            return np.empty(0, dtype=np.int64)
        if self.diverse_size < self.diverse_capacity:
            return self.diverse_start + np.arange(self.diverse_size, dtype=np.int64)
        head = self.diverse_start + np.arange(
            self.diverse_ptr, self.diverse_capacity, dtype=np.int64
        )
        tail = self.diverse_start + np.arange(0, self.diverse_ptr, dtype=np.int64)
        return np.concatenate([head, tail], axis=0)

    def _valid_indices(self):
        if self.retention_strategy == "diverse_recent":
            recent = self._recent_indices()
            diverse = self._diverse_indices()
            if recent.size == 0:
                return diverse
            if diverse.size == 0:
                return recent
            return np.concatenate([recent, diverse], axis=0)
        return np.arange(self.size, dtype=np.int64)

    def _estimate_novelty(self, embedding, candidate_indices):
        if candidate_indices.size == 0:
            return 1.0
        pool_size = min(self.candidate_pool_size, candidate_indices.size)
        subset = self.rng.choice(candidate_indices, size=pool_size, replace=False)
        dists = np.linalg.norm(self.embedding[subset] - embedding[None, :], axis=1)
        if dists.size == 0:
            return 1.0
        k = min(8, dists.size)
        nearest = np.partition(dists, k - 1)[:k]
        return float(np.mean(nearest))

    def _recency_score(self, insert_ids):
        age = np.maximum(0, self.insert_counter - insert_ids)
        return np.exp(-age / float(self.recency_horizon))

    def _append_diverse_recent(self, state, action, next_state, reward, terminal, truncation):
        embedding = self._embed_transition(state, action, next_state, reward, terminal)
        insert_id = self.insert_counter
        self.insert_counter += 1

        if self.recent_capacity > 0:
            recent_idx = self.recent_ptr
            recent_novelty = self._estimate_novelty(embedding, self._recent_indices())
            self._write_at(
                recent_idx,
                state,
                action,
                next_state,
                reward,
                terminal,
                truncation=truncation,
                embedding=embedding,
                novelty=recent_novelty,
                insert_id=insert_id,
            )
            self.recent_ptr = (self.recent_ptr + 1) % max(1, self.recent_capacity)
            self.recent_size = min(self.recent_size + 1, self.recent_capacity)

        if self.diverse_capacity > 0:
            if self.diverse_size < self.diverse_capacity:
                diverse_idx = self.diverse_start + self.diverse_size
                novelty = self._estimate_novelty(embedding, self._diverse_indices())
                self._write_at(
                    diverse_idx,
                    state,
                    action,
                    next_state,
                    reward,
                    terminal,
                    truncation=truncation,
                    embedding=embedding,
                    novelty=novelty,
                    insert_id=insert_id,
                )
                self.diverse_size += 1
            else:
                valid_diverse = self._diverse_indices()
                novelty = self._estimate_novelty(embedding, valid_diverse)
                pool_size = min(self.candidate_pool_size, valid_diverse.size)
                subset = self.rng.choice(valid_diverse, size=pool_size, replace=False)
                keep_scores = (
                    self.diversity_weight * self.novelty[subset]
                    + self.recency_weight * self._recency_score(self.insert_id[subset])
                )
                weakest_local = int(np.argmin(keep_scores))
                weakest_idx = int(subset[weakest_local])
                weakest_score = float(keep_scores[weakest_local])
                candidate_score = self.diversity_weight * novelty + self.recency_weight

                should_replace = candidate_score > weakest_score
                if not should_replace and self.random_swap_prob > 0.0:
                    should_replace = bool(self.rng.random() < self.random_swap_prob)

                if should_replace:
                    dists = np.linalg.norm(self.embedding[subset] - embedding[None, :], axis=1)
                    self.novelty[subset] = np.minimum(self.novelty[subset], dists.astype(np.float32))
                    self._write_at(
                        weakest_idx,
                        state,
                        action,
                        next_state,
                        reward,
                        terminal,
                        truncation=truncation,
                        embedding=embedding,
                        novelty=novelty,
                        insert_id=insert_id,
                    )
                    self.diverse_ptr = (weakest_idx - self.diverse_start + 1) % max(
                        1, self.diverse_capacity
                    )

        self._refresh_size()

    def sample(self):
        valid_indices = self._valid_indices()
        if valid_indices.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        replace = valid_indices.size < self.batch_size
        ind = self.rng.choice(valid_indices, size=self.batch_size, replace=replace)

        return (
            torch.from_numpy(self.state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.action[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.next_state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.reward[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.terminal[ind]).to(self.device).to(self.dtype),
        )

    def save_to_json(self, filepath):
        valid_indices = self._valid_indices()
        # Convert numpy arrays to lists for JSON serialization
        buffer_dict = {
            "state": self.state[valid_indices].tolist(),
            "action": self.action[valid_indices].tolist(),
            "next_state": self.next_state[valid_indices].tolist(),
            "reward": self.reward[valid_indices].tolist(),
            "terminal": self.terminal[valid_indices].tolist(),
            "truncation": self.truncation[valid_indices].tolist(),
            "meta": {
                "retention_strategy": self.retention_strategy,
                "size": int(valid_indices.size),
                "recent_fraction": self.recent_fraction,
                "diversity_weight": self.diversity_weight,
                "recency_weight": self.recency_weight,
            },
        }

        with open(filepath, "w") as f:
            json.dump(buffer_dict, f)

    def load_from_json(self, filepath):
        """Repopulate the buffer from a save_to_json() dump. Transitions are
        re-inserted through append() so retention bookkeeping (recent/diverse
        slots, embeddings, counters) stays consistent. Counterpart to
        save_to_json; used by the checkpoint-restart resume path."""
        with open(filepath, "r") as f:
            buffer_dict = json.load(f)

        states = np.asarray(buffer_dict.get("state", []), dtype=np.float32)
        actions = np.asarray(buffer_dict.get("action", []), dtype=np.float32)
        next_states = np.asarray(buffer_dict.get("next_state", []), dtype=np.float32)
        rewards = np.asarray(buffer_dict.get("reward", []), dtype=np.float32)
        terminals = np.asarray(buffer_dict.get("terminal", []), dtype=np.float32)
        if buffer_dict.get("truncation") is not None:
            truncations = np.asarray(buffer_dict["truncation"], dtype=np.float32)
        else:
            truncations = np.zeros_like(rewards)

        n = int(states.shape[0]) if states.ndim >= 1 else 0
        for i in range(n):
            self.append(
                states[i],
                actions[i],
                next_states[i],
                rewards[i],
                terminals[i],
                truncations[i],
            )
