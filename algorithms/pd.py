import json
import os

import torch
import torch.nn as nn

from trainers.pd_trainer import PDTrainer
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from policy.pd import PD_Learner


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

        trainer = PDTrainer(
            env=self.env,
            policy=self.policy,
            logger=self.logger,
            writer=self.writer,
            timesteps=self.args.epochs,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            student_rollout_steps=self.args.pd_student_rollout_steps,
            student_rollout_deterministic=self.args.pd_student_rollout_deterministic,
        )

        return trainer.train()

    def define_policy(self):
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=nn.LeakyReLU(),
            device=self.args.device,
        )

        target_actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.target_actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            activation=nn.Tanh(),
            device=self.args.device,
        )

        path = f"model/{self.args.env_name}/({'_'.join(str(x) for x in self.args.target_actor_fc_dim)})/{self.run_id}/"

        # find # of model_#.pth where # is highest in path dir
        iter_number = max(
            (
                int(f[len("model_") : -4])
                for f in os.listdir(path)
                if f.startswith("model_")
                and f.endswith(".pth")
                and f[len("model_") : -4].isdigit()
            ),
            default=0,
        )

        model_path = path + f"model_{iter_number}.pth"
        data_path = path + f"replay_buffer_{iter_number}.json"
        if os.path.exists(model_path):
            target_actor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            print(f"target model at {model_path} is loaded!")
        else:
            raise FileNotFoundError(f"target model at {model_path} is not found!")

        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                states = json.load(f)["state"]
                states = torch.tensor(states, dtype=torch.float32)
            print(f"replay buffer at {data_path} is loaded!")
        else:
            raise FileNotFoundError(f"replay buffer at {data_path} is not found!")

        self.policy = PD_Learner(
            actor=actor,
            target_actor=target_actor,
            teacher_states=states,
            actor_lr=self.args.actor_lr,
            target_kl=0.003,
            gamma=self.args.gamma,
            buffer_mode=self.args.pd_buffer_mode,
            teacher_buffer_capacity=self.args.pd_buffer_capacity,
            student_buffer_capacity=self.args.pd_student_buffer_capacity,
            minibatch_size=self.args.pd_batch_size,
            mixed_student_ratio=self.args.pd_mixed_student_ratio,
            mixed_update=self.args.pd_mixed_update,
            seed=self.args.seed,
            device=self.args.device,
        )
