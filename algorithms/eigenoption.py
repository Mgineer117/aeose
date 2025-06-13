from copy import deepcopy

import torch
import torch.nn as nn

from policy.eigenoption import EigenOption_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner
from policy.uniform_random import UniformRandom
from trainer.eigenoption_trainer import EigenOptionTrainer
from trainer.extractor_trainer import ExtractorTrainer
from utils.rl import get_extractor, get_vector
from utils.sampler import HLSampler, OnlineSampler


class EigenOption(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(EigenOption, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0

    def begin_training(self):
        # === Define extractor === #
        self.define_extractor()
        self.define_eigenvectors()

        # === Define policy === #
        self.define_policy()

        hl_sampler = HLSampler(
            state_dim=self.args.state_dim,
            action_dim=int(self.args.num_options + 1),
            episode_len=self.env.max_steps,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            max_option_len=10,
            gamma=self.args.gamma,
            verbose=False,
        )

        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=int(self.args.minibatch_size * self.args.num_minibatch),
            verbose=False,
        )

        trainer = EigenOptionTrainer(
            env=self.env,
            hl_policy=self.hl_policy,
            policies=self.policies,
            extractor=self.extractor,
            eigenvectors=self.eigenvectors,
            hl_sampler=hl_sampler,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            seed=self.args.seed,
        )

        # design hl_policy and hl_sampler and trainer

        trainer.train()

    def define_extractor(self):
        extractor = get_extractor(self.args)

        # if not os.path.exists(model_path):
        self.uniform_random_policy = UniformRandom(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=16384,
            verbose=False,
        )
        trainer = ExtractorTrainer(
            env=self.env,
            random_policy=self.uniform_random_policy,
            extractor=extractor,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            epochs=self.args.extractor_epochs,
            batch_size=self.args.batch_size,
        )
        final_timesteps = trainer.train()

        self.current_timesteps += final_timesteps

        self.extractor = extractor

    def define_policy(self):
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        policy = PPO_Learner(
            actor=actor,
            critic=critic,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )
        policy.name = "EigenOptionPolicy"

        self.policies = nn.ModuleList(
            [deepcopy(policy) for _ in range(self.args.num_options)]
        )
        self.policies.append(self.uniform_random_policy)

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=int(self.args.num_options + 1),
            is_discrete=True,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.hl_policy = EigenOption_Learner(
            actor=actor,
            critic=critic,
            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            num_minibatch=self.args.num_minibatch,
            minibatch_size=self.args.minibatch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        self.eigenvectors, heatmaps = get_vector(self.env, self.extractor, self.args)
        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )
