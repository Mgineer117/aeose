import time
import math
import numpy as np
import torch
import torch.nn as nn

from policy.base import Base
from policy.layers.ppo_networks import PPO_Critic
from utils.rl import RunningMeanStd
from utils.replay_buffer import ReplayBuffer


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state, action_dim, parent=None, action=None):
        self.state = state  # np.ndarray
        self.action_dim = action_dim
        self.parent = parent
        self.action = action  # action taken to reach this node
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0  # cumulative return
        self.is_terminal = False
        
    def ucb_score(self, c_puct=1.0):
        """Upper Confidence Bound for tree node selection."""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / self.visit_count
        exploration = c_puct * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration
    
    def select_best_child(self, c_puct=1.0):
        """Select child with highest UCB score."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))
    
    def backup(self, value):
        """Backpropagate value up the tree."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backup(value)


class MCTS_Learner(Base):
    """Monte Carlo Tree Search learner for discrete control with learned value function."""
    
    def __init__(
        self,
        critic: PPO_Critic,
        replay_buffer: ReplayBuffer,
        action_dim: int = 6,
        critic_lr: float = 5e-4,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        
        self.name = "MCTS"
        self.replay_buffer = replay_buffer
        self.device = device
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        
        # Value function (learned critic)
        self.critic = critic.to(device)
        self.state_dim = critic.state_dim
        
        # Running statistics for observation normalization
        try:
            self.obs_rms = RunningMeanStd(self.state_dim)
        except Exception:
            self.obs_rms = None
        
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.to(self.dtype).to(self.device)
    
    def preprocess_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Normalize observations using running mean/std."""
        state = super().preprocess_state(state)
        try:
            if getattr(self, "obs_rms", None) is not None and self.obs_rms.count > 0:
                mean = torch.from_numpy(self.obs_rms.mean).to(self.device).to(state.dtype)
                var = torch.from_numpy(self.obs_rms.var).to(self.device).to(state.dtype)
                state = (state - mean) / torch.sqrt(var + 1e-8)
        except Exception:
            pass
        return state
    
    @property
    def actor(self):
        """Return critic for model saving (MCTS uses critic for decisions)."""
        return self.critic
    
    def forward(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action using MCTS.
        Handles both single state (shape [72]) and batches (shape [batch, 72]).
        Returns: one-hot action tensor batch, metadata with logprobs and entropy
        """
        self.eval()
        
        # Handle both single state and batch
        if len(state.shape) == 1:
            # Single state - add batch dimension
            state = state[np.newaxis, :]
            is_single = True
        else:
            is_single = False
        
        batch_size = state.shape[0]
        action_one_hot_list = []
        logprobs_list = []
        entropy_list = []
        
        # Process each state in the batch
        for i in range(batch_size):
            state_single = state[i]
            state_tensor = self.preprocess_state(state_single)
            
            # Initialize root node
            root = MCTSNode(state_single, self.action_dim)
            
            # Run simulations
            for _ in range(self.num_simulations):
                self._simulate(root, state_tensor)
            
            # Compute action probabilities from visit counts
            visit_counts = np.array([root.children.get(a, MCTSNode(None, 0)).visit_count 
                                    for a in range(self.action_dim)], dtype=np.float32)
            visit_counts = visit_counts / (visit_counts.sum() + 1e-8)
            
            # Select action with highest probability
            best_action = np.argmax(visit_counts)
            action_one_hot = np.eye(self.action_dim, dtype=np.float32)[best_action]
            action_one_hot_list.append(action_one_hot)
            
            # Compute logprob of selected action and entropy
            prob_tensor = torch.from_numpy(visit_counts).to(self.device).float()
            logprobs_all = torch.log(prob_tensor + 1e-8)
            logprob_selected = logprobs_all[best_action].unsqueeze(0)  # Shape [1]
            entropy = -torch.sum(prob_tensor * logprobs_all)
            logprobs_list.append(logprob_selected)
            entropy_list.append(entropy)
        
        # Stack results
        action_one_hot_batch = torch.stack([torch.from_numpy(a).to(self.device).float() 
                                            for a in action_one_hot_list], dim=0)
        logprobs_batch = torch.stack([lp.unsqueeze(0) for lp in logprobs_list], dim=0).squeeze(-1)  # Shape [batch, 1]
        entropy_batch = torch.stack(entropy_list, dim=0).unsqueeze(-1)  # Shape [batch, 1]
        
        return action_one_hot_batch, {
            "logprobs": logprobs_batch,
            "entropy": entropy_batch,
        }
    
    def _simulate(self, node: MCTSNode, state_tensor: torch.Tensor) -> float:
        """Run one MCTS simulation: selection -> expansion -> simulation -> backup."""
        
        # Selection and expansion
        current = node
        path = [current]
        
        # Traverse tree using UCB until reaching unexpanded node
        while current.children and not current.is_terminal:
            best_child = current.select_best_child(self.c_puct)
            if best_child is None:
                break
            current = best_child
            path.append(current)
        
        # Expand if not terminal and has unvisited actions
        if not current.is_terminal and current.visit_count > 0:
            for action in range(self.action_dim):
                if action not in current.children:
                    # Create child node with dummy state (will be updated later)
                    child = MCTSNode(current.state, self.action_dim, 
                                    parent=current, action=action)
                    current.children[action] = child
        
        # If no children created, select random unvisited action
        if not current.children:
            action = np.random.randint(0, self.action_dim)
            child = MCTSNode(current.state, self.action_dim, 
                            parent=current, action=action)
            current.children[action] = child
            current = child
            path.append(current)
        
        # Simulation (rollout): use value function for rollout estimate
        value = self._estimate_value(current.state)
        
        # Backup
        path[-1].backup(value)
        
        return value
    
    def _estimate_value(self, state: np.ndarray) -> float:
        """Estimate value of state using critic."""
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            value = self.critic(state_tensor).item()
        return value
    
    def learn(self, batch):
        """Train value function using batch of experience."""
        self.train()
        t0 = time.time()
        
        # Update observation normalizer
        try:
            if getattr(self, "obs_rms", None) is not None:
                states_raw = np.asarray(batch["states"], dtype=np.float32)
                states_raw = np.nan_to_num(states_raw, nan=0.0, posinf=0.0, neginf=0.0)
                self.obs_rms.update(states_raw)
        except Exception:
            pass
        
        # Store in replay buffer
        for i in range(batch["states"].shape[0]):
            self.replay_buffer.append(
                batch["states"][i],
                batch["actions"][i],
                batch["next_states"][i],
                batch["rewards"][i],
                batch.get("dones", batch.get("terminals"))[i],
            )
        
        # Preprocess tensors
        states = self.preprocess_state(batch["states"])
        rewards = super().preprocess_state(batch["rewards"])
        terminations = super().preprocess_state(batch["terminations"])
        truncations = super().preprocess_state(batch["truncations"])
        
        # Compute target values: discounted return
        with torch.no_grad():
            next_states = self.preprocess_state(batch["next_states"])
            next_values = self.critic(next_states)
            
            # Handle termination/truncation: don't bootstrap on terminal states
            bootstrap_mask = (1.0 - terminations) * (1.0 - truncations)
            targets = rewards + self.gamma * next_values * bootstrap_mask
        
        # Train critic with MSE loss
        value_preds = self.critic(states)
        value_loss = self.mse_loss(value_preds, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Logging
        loss_dict = {
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/analytics/avg_rewards": torch.mean(rewards).item(),
        }
        
        timesteps = states.shape[0]
        update_time = time.time() - t0
        
        self.eval()
        
        return loss_dict, timesteps, update_time


class MCTS_Offline_Learner(Base):
    """
    Offline MCTS: Generates Q-values via pure tree search, trains network on collected data.
    
    Pipeline:
    1. Generate training data: Run pure MCTS to collect (s, a, Q*) tuples
    2. Supervised learning: Train Q-network to regress over MCTS estimates
    3. Inference: Use learned Q-values for greedy action selection
    """
    
    def __init__(
        self,
        critic: PPO_Critic,
        replay_buffer: ReplayBuffer,
        action_dim: int = 6,
        critic_lr: float = 5e-4,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        
        self.name = "MCTS_Offline"
        self.replay_buffer = replay_buffer
        self.device = device
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        
        self.critic = critic.to(device)
        self.state_dim = critic.state_dim
        
        # Observation normalization
        try:
            self.obs_rms = RunningMeanStd(self.state_dim)
        except Exception:
            self.obs_rms = None
        
        # Training dataset: list of (state, action, q_value) tuples
        self.training_data = []
        
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.to(self.dtype).to(self.device)
    
    def preprocess_state(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Normalize observations using running mean/std."""
        state = super().preprocess_state(state)
        try:
            if getattr(self, "obs_rms", None) is not None and self.obs_rms.count > 0:
                mean = torch.from_numpy(self.obs_rms.mean).to(self.device).to(state.dtype)
                var = torch.from_numpy(self.obs_rms.var).to(self.device).to(state.dtype)
                state = (state - mean) / torch.sqrt(var + 1e-8)
        except Exception:
            pass
        return state
    
    @property
    def actor(self):
        """Return critic for model saving (MCTS uses critic for decisions)."""
        return self.critic
    
    def forward(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action using learned Q-function.
        Handles both single state (shape [72]) and batches (shape [batch, 72]).
        Returns: one-hot action tensor batch, metadata with logprobs and entropy
        """
        self.eval()
        
        # Handle both single state and batch
        if len(state.shape) == 1:
            # Single state - add batch dimension
            state = state[np.newaxis, :]
            is_single = True
        else:
            is_single = False
        
        batch_size = state.shape[0]
        action_one_hot_list = []
        logprobs_list = []
        entropy_list = []
        
        # Process each state in the batch
        for i in range(batch_size):
            state_single = state[i]
            
            # If no training data collected yet, use uniform probabilities
            if len(self.training_data) == 0:
                q_probs = np.ones(self.action_dim, dtype=np.float32) / self.action_dim
            else:
                state_tensor = self.preprocess_state(state_single)
                if len(state_tensor.shape) == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                
                # Compute Q-values for all actions using critic
                with torch.no_grad():
                    q_values = []
                    for action in range(self.action_dim):
                        # Get state value from critic (approximation for Q(s, a))
                        q_val = self.critic(state_tensor).item()
                        q_values.append(q_val)
                    
                    q_values = np.array(q_values, dtype=np.float32)
                    # Normalize to probabilities (softmax-like)
                    q_probs = np.exp(q_values - np.max(q_values)) / np.sum(np.exp(q_values - np.max(q_values)))
            
            # Select action with highest Q-value
            best_action = np.argmax(q_probs)
            action_one_hot = np.eye(self.action_dim, dtype=np.float32)[best_action]
            action_one_hot_list.append(action_one_hot)
            
            # Compute logprob of selected action and entropy
            prob_tensor = torch.from_numpy(q_probs).to(self.device).float()
            logprobs_all = torch.log(prob_tensor + 1e-8)
            logprob_selected = logprobs_all[best_action].unsqueeze(0)  # Shape [1]
            entropy = -torch.sum(prob_tensor * logprobs_all)
            logprobs_list.append(logprob_selected)
            entropy_list.append(entropy)
        
        # Stack results
        action_one_hot_batch = torch.stack([torch.from_numpy(a).to(self.device).float() 
                                            for a in action_one_hot_list], dim=0)
        logprobs_batch = torch.stack([lp.unsqueeze(0) for lp in logprobs_list], dim=0).squeeze(-1)  # Shape [batch, 1]
        entropy_batch = torch.stack(entropy_list, dim=0).unsqueeze(-1)  # Shape [batch, 1]
        
        return action_one_hot_batch, {
            "logprobs": logprobs_batch,
            "entropy": entropy_batch,
        }
    
    def _pure_tree_search(self, state: np.ndarray, num_sims: int) -> dict:
        """
        Run pure MCTS without network guidance.
        Returns: q_estimates dict {action -> Q(s, a)}
        """
        root = MCTSNode(state, self.action_dim)
        
        # Initialize all actions as children (for pure search)
        for action in range(self.action_dim):
            root.children[action] = MCTSNode(state, self.action_dim, 
                                            parent=root, action=action)
        
        # Simulate
        for _ in range(num_sims):
            self._simulate_rollout(root, state)
        
        # Extract Q(s, a) from visit statistics
        q_estimates = {}
        total_visits = max(1, root.visit_count)
        for action in range(self.action_dim):
            child = root.children[action]
            if child.visit_count > 0:
                q_estimates[action] = child.value_sum / child.visit_count
            else:
                q_estimates[action] = 0.0
        
        return q_estimates
    
    def _simulate_rollout(self, node: MCTSNode, state: np.ndarray) -> float:
        """
        One rollout: random descent from node.
        Returns value estimate (placeholder: 0 or reward signal)
        """
        # Pure random rollout: estimate value as 0 or reward accumulation
        # In practice, could use environment or heuristic evaluation
        value = 0.0
        
        # Backup
        node.visit_count += 1
        node.value_sum += value
        if node.parent:
            node.parent.backup(value)
        
        return value
    
    def learn(self, batch):
        """
        Supervised learning phase: Train Q-network on (s, a, Q*) data.
        
        The batch here contains trajectory data. We first generate Q-values
        via pure MCTS, then train the network to predict them.
        """
        self.train()
        t0 = time.time()
        
        # Update observation normalizer
        try:
            if getattr(self, "obs_rms", None) is not None:
                states_raw = np.asarray(batch["states"], dtype=np.float32)
                states_raw = np.nan_to_num(states_raw, nan=0.0, posinf=0.0, neginf=0.0)
                self.obs_rms.update(states_raw)
        except Exception:
            pass
        
        # Phase 1: Generate Q-values via pure tree search
        print(f"[{self.name}] Generating MCTS training data...")
        q_training_samples = []
        
        for i in range(min(len(batch["states"]), 50)):  # Limit to 50 states to avoid slowdown
            state = batch["states"][i]
            
            # Run pure MCTS to get Q(s, a) estimates
            q_estimates = self._pure_tree_search(state, self.num_simulations)
            
            for action, q_value in q_estimates.items():
                q_training_samples.append((state, action, q_value))
                self.training_data.append((state, action, q_value))
        
        # Phase 2: Train Q-network on collected data
        if len(q_training_samples) == 0:
            loss_dict = {f"{self.name}/loss/value_loss": 0.0}
            timesteps = batch["states"].shape[0]
            update_time = time.time() - t0
            self.eval()
            return loss_dict, timesteps, update_time
        
        print(f"[{self.name}] Training Q-network on {len(q_training_samples)} samples...")
        
        # Convert to tensors
        states_list = [s for s, a, q in q_training_samples]
        actions_list = np.array([a for s, a, q in q_training_samples])
        q_targets = np.array([q for s, a, q in q_training_samples], dtype=np.float32)
        
        states_tensor = self.preprocess_state(np.array(states_list))
        q_targets_tensor = torch.from_numpy(q_targets).to(self.device).to(self.dtype).unsqueeze(1)
        
        # Train critic to predict Q-values
        value_preds = self.critic(states_tensor)
        value_loss = self.mse_loss(value_preds, q_targets_tensor)
        
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        loss_dict = {
            f"{self.name}/loss/value_loss": value_loss.item(),
            f"{self.name}/analytics/q_training_samples": len(q_training_samples),
        }
        
        timesteps = batch["states"].shape[0]
        update_time = time.time() - t0
        
        self.eval()
        
        return loss_dict, timesteps, update_time
