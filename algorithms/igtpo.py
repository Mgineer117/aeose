import os

import torch
import torch.nn as nn

from policy.igtpo import IGTPO_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner
from policy.uniform_random import UniformRandom
from policy.value_functions import Critic_Learner, Critics_Learner
from trainer.base_trainer import Trainer
from trainer.extractor_trainer import ExtractorTrainer
from trainer.meta_trainer import MetaTrainer
from utils.rl import get_extractor, get_vector
from utils.sampler import OnlineSampler


class IGTPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IGTPO_Algorithm, self).__init__()

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

        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
        )

        # === Meta-train using options === #'
        self.define_meta_policy()
        trainer = MetaTrainer(
            env=self.env,
            policy=self.policy,
            extractor=self.extractor,
            meta_critic=self.critic,
            task_critics=self.task_critics,
            subtask_critics=self.subtask_critics,
            eigenvectors=self.eigenvectors,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            num_local_updates=self.args.num_local_updates,
            init_timesteps=self.current_timesteps,
            timesteps=int(0.9 * self.args.timesteps),
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            marker=self.args.marker,
            seed=self.args.seed,
        )
        final_steps = trainer.train()
        self.current_timesteps += final_steps

        # === Fine-tune === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=int(self.args.num_minibatch * self.args.minibatch_size),
        )

        self.define_ppo_policy()
        trainer = Trainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=int(0.1 * self.args.timesteps),
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            seed=self.args.seed,
        )

        trainer.train()

    def define_extractor(self):
        model_path = f"model/{self.args.env_name}-feature_network.pth"
        extractor = get_extractor(self.args)

        # if not os.path.exists(model_path):
        uniform_random_policy = UniformRandom(
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
            random_policy=uniform_random_policy,
            extractor=extractor,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            epochs=self.args.extractor_epochs,
            batch_size=self.args.batch_size,
        )
        final_timesteps = trainer.train()
        # torch.save(extractor.state_dict(), model_path)

        self.current_timesteps += final_timesteps
        # else:
        #     extractor.load_state_dict(
        #         torch.load(model_path, map_location=self.args.device)
        #     )
        #     extractor.to(self.args.device)

        self.extractor = extractor

    def define_eigenvectors(self):
        # === Define eigenvectors === #
        self.eigenvectors, heatmaps = get_vector(self.env, self.extractor, self.args)
        self.logger.write_images(
            step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
        )

    def define_meta_policy(self):
        # === Define policy === #
        self.actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        self.critic = Critic_Learner(
            critic=critic,
            critic_lr=self.args.critic_lr,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )
        self.task_critics = Critics_Learner(
            critic=critic,
            critic_lr=self.args.critic_lr,
            num=self.eigenvectors.shape[0],
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )
        self.subtask_critics = Critics_Learner(
            critic=critic,
            critic_lr=self.args.critic_lr,
            num=self.eigenvectors.shape[0],
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )

        self.policy = IGTPO_Learner(
            actor=self.actor,
            actor_lr=self.args.meta_actor_lr,
            batch_size=self.args.batch_size,
            eps_clip=self.args.eps_clip,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            K=self.args.K_epochs,
            device=self.args.device,
        )

    def define_ppo_policy(self):
        self.policy = PPO_Learner(
            actor=self.actor,
            critic=self.critic,
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
