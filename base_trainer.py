import os
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log.wandb_logger import WandbLogger
from policy.base import Base
from utils.rl import estimate_advantages
from utils.sampler import OnlineSampler


# model-free policy trainer
class Trainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        episode_len: int,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        rendering: bool = False,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sampler = sampler
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.episode_len = episode_len
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        self.last_max_return_mean = -1e10
        self.last_min_return_std = 1e10

        self.rendering = rendering
        self.seed = seed

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)
        self.last_return_std = deque(maxlen=5)

        # Train loop
        eval_idx = 0
        total_clock_time = 0
        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                batch, sample_time = self.sampler.collect_samples(
                    policy=self.policy, seed=self.seed
                )
                if "states" in batch:
                    loss_dict, timesteps, update_time = self.policy.learn(batch)

                    # Calculate expected remaining time
                    pbar.update(timesteps)

                    elapsed_time = time.time() - start_time
                    avg_time_per_iter = elapsed_time / step
                    remaining_time = avg_time_per_iter * (self.timesteps - step)

                    total_clock_time += sample_time
                    total_clock_time += update_time

                    # Update environment steps and calculate time metrics
                    loss_dict[f"{self.policy.name}/analytics/timesteps"] = (
                        step + timesteps
                    )
                    loss_dict[f"{self.policy.name}/analytics/total_clock_time (s)"] = (
                        total_clock_time
                    )
                    loss_dict[f"{self.policy.name}/analytics/sample_time"] = sample_time
                    loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                    loss_dict[f"{self.policy.name}/analytics/remaining_time (hr)"] = (
                        remaining_time / 3600
                    )  # Convert to hours
                    loss_dict[f"{self.policy.name}/analytics/discounted_return"] = (
                        self.average_discounted_return(
                            batch["rewards"], batch["terminals"], self.policy.gamma
                        )
                    )

                    self.write_log(loss_dict, step=step)

                    #### EVALUATIONS ####
                    if step >= self.eval_interval * eval_idx:
                        ### Eval Loop
                        self.policy.eval()
                        eval_idx += 1

                        eval_dict, running_video = self.evaluate()

                        # Manual logging
                        self.write_log(eval_dict, step=step, eval_log=True)
                        self.write_video(
                            running_video,
                            step=step,
                            logdir=f"videos",
                            name="running_video",
                        )

                        self.last_return_mean.append(eval_dict[f"eval/return_mean"])
                        self.last_return_std.append(eval_dict[f"eval/return_std"])

                        self.save_model(step)

                torch.cuda.empty_cache()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward = []

            # Env initialization
            state, _ = self.env.reset(seed=self.seed)

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, _ = self.policy(state, deterministic=True)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(np.argmax(a))
                if t == self.episode_len - 1:
                    # safe truncation
                    trunc = True
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    discounted_return = self.discounted_return(
                        ep_reward, self.policy.gamma
                    )
                    ep_buffer.append(
                        {
                            "return": discounted_return,
                            "avg_reward": np.mean(ep_reward),
                            "episode_length": t + 1,
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        avg_reward_list = [ep_info["avg_reward"] for ep_info in ep_buffer]
        episode_length_list = [ep_info["episode_length"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        avg_reward_mean, avg_reward_std = np.mean(avg_reward_list), np.std(
            avg_reward_list
        )
        epi_len_mean, epi_len_std = np.mean(episode_length_list), np.std(
            episode_length_list
        )

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
            f"eval/avg_reward_mean": avg_reward_mean,
            f"eval/avg_reward_std": avg_reward_std,
            f"eval/epi_len_mean": epi_len_mean,
            f"eval/epi_len_std": epi_len_std,
        }

        return eval_dict, image_array

    def average_discounted_return(self, rewards, terminals, gamma):
        """
        Computes the average discounted return across all episodes, resetting at terminals.

        Args:
            rewards (list or np.array): Sequence of rewards.
            terminals (list or np.array): Sequence of terminal flags (bool or 0/1).
            gamma (float): Discount factor.

        Returns:
            float: Average episodic discounted return.
        """
        episode_returns = []
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            if terminals[t]:
                episode_returns.append(G)
                G = 0.0  # reset for the next episode

        if not episode_returns:
            return 0.0
        return sum(episode_returns) / len(episode_returns)

    def discounted_return(self, rewards, gamma):
        G = 0.0
        for i, r in enumerate(reversed(rewards)):
            G = float(r) + gamma * G
        return G

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, image: np.ndarray, step: int, logdir: str, name: str):
        image_list = [image]
        image_path = os.path.join(logdir, name)
        self.logger.write_images(step=step, images=image_list, logdir=image_path)

    def write_video(self, image: list, step: int, logdir: str, name: str):
        if len(image) > 0:
            tensor = np.stack(image, axis=0)
            video_path = os.path.join(logdir, name)
            self.logger.write_videos(step=step, images=tensor, logdir=video_path)

    def save_model(self, e):
        ### save checkpoint
        name = f"model_{e}.pth"
        path = os.path.join(self.logger.checkpoint_dir, name)

        model = self.policy.actor

        if model is not None:
            model = deepcopy(model).to("cpu")
            torch.save(model.state_dict(), path)

            # save the best model
            if (
                np.mean(self.last_return_mean) >= self.last_max_return_mean
                and np.mean(self.last_return_std) <= self.last_min_return_std
            ):
                name = f"best_model.pth"
                path = os.path.join(self.logger.log_dir, name)
                torch.save(model.state_dict(), path)

                self.last_max_return_mean = np.mean(self.last_return_mean)
                self.last_min_return_std = np.mean(self.last_return_std)
        else:
            raise ValueError("Error: Model is not identifiable!!!")
