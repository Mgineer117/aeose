import os
import torch
import torch.nn as nn

from policy.pd import PD_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from offpolicy_trainer import OffPolicyTrainer
from utils.replay_buffer import ReplayBuffer


class PD_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args, run_id):
        super(PD_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args
        self.run_id = run_id

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # === Sampler === #
        replay_buffer = ReplayBuffer(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            buffer_size=100_000,
            batch_size=self.args.batch_size,
            device=self.args.device,
        )
        trainer = OffPolicyTrainer(
            env=self.env,
            policy=self.policy,
            replay_buffer=replay_buffer,
            logger=self.logger,
            writer=self.writer,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            warmup_samples=self.args.warmup_samples,
            rendering=self.args.rendering,
            seed=self.args.seed,
        )

        trainer.train()

    def define_policy(self):
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )

        target_actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.target_actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )

        model_path = f"model/model({'_'.join(str(x) for x in self.args.target_actor_fc_dim)})_{self.run_id}.pth"
        if os.path.exists(model_path):
            target_actor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            print(f"target model at {model_path} is loaded!")
        else:
            raise FileNotFoundError(f"target model at {model_path} is not found!")

        self.policy = PD_Learner(
            actor=actor,
            target_actor=target_actor,
            actor_lr=self.args.actor_lr,
            target_kl=0.01,
            gamma=self.args.gamma,
            device=self.args.device,
        )
