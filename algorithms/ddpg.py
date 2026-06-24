import torch
import torch.nn as nn

from policy.ddpg import DDPG_Learner
from policy.layers.td3_network import TD3_Actor, TD3_Actor_From_Critic, TD3_Critic
from trainers.offpolicy_trainer import OffPolicyTrainer
from utils.replay_buffer import ReplayBuffer


class DDPG_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(DDPG_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.args.nupdates = args.timesteps // args.batch_size

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        replay_buffer = ReplayBuffer(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            buffer_size=100_000,
            batch_size=self.args.batch_size,
            device=self.args.device,
        )
        import os
        import json
        latest_model_path = os.path.join(self.logger.log_dir, "latest_model.pth")
        resume_state_path = os.path.join(self.logger.log_dir, "resume_state.json")
        latest_buffer_path = os.path.join(self.logger.log_dir, "latest_buffer.json")
        
        if os.path.exists(latest_model_path) and os.path.exists(resume_state_path):
            print(f"Resuming from latest checkpoint at {latest_model_path}!")
            checkpoint = torch.load(latest_model_path, map_location=self.args.device)
            self.policy.actor.load_state_dict(checkpoint)
            
            if os.path.exists(latest_buffer_path):
                replay_buffer.load_from_json(latest_buffer_path)
                
            with open(resume_state_path, "r") as f:
                state = json.load(f)
                self.args.init_timesteps = state.get("step", 0)

        trainer = OffPolicyTrainer(
            env=self.env,
            policy=self.policy,
            replay_buffer=replay_buffer,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=getattr(self.args, "init_timesteps", 0),
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            warmup_samples=self.args.warmup_samples,
            rendering=self.args.rendering,
            seed=self.args.seed,
            checkpoint_interval=getattr(self.args, "checkpoint_interval_sec", 1800.0),
        )

        return trainer.train()

    def define_policy(self):
        if self.args.is_discrete:
            critic = TD3_Critic(
                self.args.state_dim,
                self.args.action_dim,
                hidden_dim=self.args.critic_fc_dim,
            )
            actor = TD3_Actor_From_Critic(critic)
        else:
            action_scale = (self.env.action_space.low, self.env.action_space.high)
            actor = TD3_Actor(
                input_dim=self.args.state_dim,
                hidden_dim=self.args.actor_fc_dim,
                action_dim=self.args.action_dim,
                action_scale=action_scale,
                device=self.args.device,
            )
            critic = TD3_Critic(
                self.args.state_dim,
                self.args.action_dim,
                hidden_dim=self.args.critic_fc_dim,
            )

        self.policy = DDPG_Learner(
            actor=actor,
            critic=critic,
            nupdates=self.args.nupdates,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            gamma=self.args.gamma,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.grid = self.env.get_grid()
