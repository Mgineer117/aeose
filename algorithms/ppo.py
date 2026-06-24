import torch
import torch.nn as nn

from trainers.base_trainer import Trainer
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.ppo import PPO_Learner
from utils.sampler import OnlineSampler
from utils.replay_buffer import ReplayBuffer


class PPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(PPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

    def begin_training(self):
        # === Define policy === #
        self.define_policy()

        # PPO should use the vectorized rollout path by default.
        # The sampler switches from serial env stepping to batched stepping
        # when envs_per_worker > 1.
        # If the environment does not expose a decision-step `max_steps`,
        # fall back to spawning `num_minibatch` workers (or vectorized groups)
        # and let each worker collect up to `minibatch_size` steps. This
        # produces a total batch of `minibatch_size * num_minibatch` as
        # expected by the learner.
        total_batch = int(self.args.minibatch_size * self.args.num_minibatch)

        if getattr(self.env, "max_steps", None) is None and getattr(
            self.args, "episode_len", None
        ) is None:
            episode_len = int(self.args.minibatch_size)
            # spawn one env per minibatch (workers collect minibatch_size each)
            num_workers = max(1, int(getattr(self.args, "num_minibatch", 1)))
            envs_per_worker = 1
        else:
            episode_len = int(getattr(self.env, "max_steps", self.args.episode_len))
            num_workers = max(1, int(getattr(self.args, "num_workers", 3)))
            envs_per_worker = max(2, int(getattr(self.args, "envs_per_worker", 4)))

        # === Sampler === #
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

        trainer = Trainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            episode_len=episode_len,
            timesteps=self.args.timesteps,
            init_timesteps=getattr(self.args, "init_timesteps", 0),
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            async_sampling=getattr(self.args, "async_sampling", False),
            checkpoint_interval=getattr(self.args, "checkpoint_interval_sec", 1800.0),
        )

        return trainer.train()

    def define_policy(self):
        replay_buffer = ReplayBuffer(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            buffer_size=200_000,
            batch_size=self.args.batch_size,
            device=self.args.device,
            retention_strategy=self.args.ppo_buffer_strategy,
            recent_fraction=self.args.ppo_buffer_recent_fraction,
            diversity_weight=self.args.ppo_buffer_diversity_weight,
            recency_weight=self.args.ppo_buffer_recency_weight,
            random_swap_prob=self.args.ppo_buffer_random_swap_prob,
            candidate_pool_size=self.args.ppo_buffer_candidate_pool_size,
            recency_horizon=self.args.ppo_buffer_recency_horizon,
            projection_dim=self.args.ppo_buffer_projection_dim,
            seed=self.args.seed,
        )

        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=self.args.activation(),
            device=self.args.device,
        )
        critic = PPO_Critic(
            self.args.state_dim,
            hidden_dim=self.args.critic_fc_dim,
            activation=self.args.activation(),
        )

        self.policy = PPO_Learner(
            actor=actor,
            critic=critic,
            replay_buffer=replay_buffer,
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

        import os
        import json
        latest_model_path = os.path.join(self.logger.log_dir, "latest_model.pth")
        resume_state_path = os.path.join(self.logger.log_dir, "resume_state.json")
        latest_buffer_path = os.path.join(self.logger.log_dir, "latest_buffer.json")
        
        if os.path.exists(latest_model_path) and os.path.exists(resume_state_path):
            print(f"Resuming from latest checkpoint at {latest_model_path}!")
            checkpoint = torch.load(latest_model_path, map_location=self.args.device)
            # Restore the full training state (actor+critic+optimizer+obs_rms),
            # not just the actor, so training continues rather than restarting
            # the critic/optimizer/normalizer from scratch.
            if hasattr(self.policy, "load_training_state"):
                self.policy.load_training_state(checkpoint)
            else:
                self.policy.actor.load_state_dict(checkpoint)

            if os.path.exists(latest_buffer_path) and hasattr(self.policy, "replay_buffer"):
                self.policy.replay_buffer.load_from_json(latest_buffer_path)
                
            with open(resume_state_path, "r") as f:
                state = json.load(f)
                self.args.init_timesteps = state.get("step", 0)

        elif self.args.load_model:
            model_path = (
                f"model/model({'_'.join(str(x) for x in self.args.actor_fc_dim)}).pth"
            )

            checkpoint = torch.load(model_path, map_location=self.args.device)
            self.policy.actor.load_state_dict(checkpoint)
            print(f"model at {model_path} is loaded!")
