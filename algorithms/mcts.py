import torch
import torch.nn as nn

from base_trainer import Trainer
from policy.layers.ppo_networks import PPO_Critic
from policy.mcts import MCTS_Learner, MCTS_Offline_Learner
from utils.sampler import OnlineSampler
from utils.replay_buffer import ReplayBuffer


class MCTS_Algorithm(nn.Module):
    """Monte Carlo Tree Search algorithm for discrete control."""
    
    def __init__(self, env, logger, writer, args):
        super(MCTS_Algorithm, self).__init__()
        
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args
    
    def begin_training(self):
        """Initialize policy, sampler, and trainer for MCTS."""
        
        # Define critic (value function) network
        self.define_policy()
        
        # Sampler configuration
        total_batch = int(self.args.minibatch_size * self.args.num_minibatch)
        
        if getattr(self.env, "max_steps", None) is None and getattr(
            self.args, "episode_len", None
        ) is None:
            episode_len = int(self.args.minibatch_size)
            num_workers = max(1, int(getattr(self.args, "num_minibatch", 1)))
            envs_per_worker = 1
        else:
            episode_len = int(getattr(self.env, "max_steps", self.args.episode_len))
            num_workers = max(1, int(getattr(self.args, "num_workers", 3)))
            envs_per_worker = max(2, int(getattr(self.args, "envs_per_worker", 4)))
        
        # Sampler
        sampler = OnlineSampler(
            env_name=self.args.env_name,
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=episode_len,
            batch_size=total_batch,
            num_workers=num_workers,
            envs_per_worker=envs_per_worker,
            sampler_mode=getattr(self.args, "sampler_mode", "vectorized"),
        )
        
        # Trainer
        trainer = Trainer(
            env=self.env,
            policy=self.mcts_learner,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            episode_len=episode_len,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            async_sampling=getattr(self.args, "async_sampling", False),
        )
        
        trainer.train()
    
    def define_policy(self):
        """Define MCTS learner with critic network."""
        
        # Replay buffer
        replay_buffer = ReplayBuffer(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            buffer_size=200_000,
            batch_size=self.args.batch_size,
            device=self.args.device,
        )
        
        # Critic network
        critic = PPO_Critic(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.critic_fc_dim,
            activation=self.args.activation(),
        )
        
        # Instantiate correct MCTS learner based on mode
        mcts_mode = getattr(self.args, "mcts_mode", "online")
        
        if mcts_mode == "online":
            print(f"[MCTS] Initializing Online MCTS (tree search + learned critic)")
            self.mcts_learner = MCTS_Learner(
                critic=critic,
                replay_buffer=replay_buffer,
                action_dim=self.args.action_dim,
                critic_lr=self.args.critic_lr,
                num_simulations=getattr(self.args, "num_simulations", 100),
                c_puct=getattr(self.args, "c_puct", 1.0),
                gamma=self.args.gamma,
                device=self.args.device,
            )
        elif mcts_mode == "offline":
            print(f"[MCTS] Initializing Offline MCTS (pure tree search → supervised learning)")
            self.mcts_learner = MCTS_Offline_Learner(
                critic=critic,
                replay_buffer=replay_buffer,
                action_dim=self.args.action_dim,
                critic_lr=self.args.critic_lr,
                num_simulations=getattr(self.args, "num_simulations", 100),
                c_puct=getattr(self.args, "c_puct", 1.0),
                gamma=self.args.gamma,
                device=self.args.device,
            )
        else:
            raise ValueError(f"Unknown MCTS mode: {mcts_mode}. Choose 'online' or 'offline'.")

