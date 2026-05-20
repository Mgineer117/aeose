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


# model-free policy trainer
class PDTrainer:
    def __init__(
        self,
        env: gym.Env,
        policy: Base,
        logger: WandbLogger,
        writer: SummaryWriter,
        init_timesteps: int = 0,
        timesteps: int = 1e6,
        log_interval: int = 100,
        eval_num: int = 10,
        rendering: bool = False,
        seed: int = 0,
        student_rollout_steps: int = 0,
        student_rollout_deterministic: bool = False,
    ) -> None:
        self.env = env
        self.policy = policy
        self.eval_num = eval_num

        self.logger = logger
        self.writer = writer

        # training parameters
        self.init_timesteps = init_timesteps
        self.timesteps = timesteps

        self.log_interval = log_interval
        self.eval_interval = int(self.timesteps / self.log_interval)

        # initialize the essential training components
        # Use -inf as the sentinel so best_model tracks the *highest* return.
        self.last_max_return_mean = -1e10

        self.rendering = rendering
        self.seed = seed
        self.student_rollout_steps = max(0, int(student_rollout_steps))
        self.student_rollout_deterministic = student_rollout_deterministic

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=5)

        # === Pre-training baseline eval (no checkpoint save). ===
        self._run_eval(step=self.init_timesteps, save=False)

        # Train loop
        eval_idx = 0
        policy_infos = {"termination": False}
        with tqdm(
            total=self.timesteps + self.init_timesteps,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < self.timesteps + self.init_timesteps:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                student_added = 0
                if self.student_rollout_steps > 0 and hasattr(
                    self.policy, "ingest_student_states"
                ):
                    states = self.collect_student_states(
                        max_steps=self.student_rollout_steps,
                        deterministic=self.student_rollout_deterministic,
                    )
                    student_added = self.policy.ingest_student_states(states)

                loss_dict, update_time, policy_infos = self.policy.learn()
                loss_dict[f"{self.policy.name}/buffer/student_added"] = student_added

                pbar.update(1)

                # Update environment steps and calculate time metrics
                loss_dict[f"{self.policy.name}/analytics/epochs"] = step
                loss_dict[f"{self.policy.name}/analytics/update_time"] = update_time
                self.write_log(loss_dict, step=step)

                #### Periodic evaluation ####
                if step >= self.eval_interval * (eval_idx + 1):
                    eval_idx += 1
                    self._run_eval(step=step, save=True, eval_idx=eval_idx)

                # terminate the training loop
                if policy_infos["termination"]:
                    break

                torch.cuda.empty_cache()

        # === Final evaluation — always checkpoints. ===
        final_step = pbar.n
        self._run_eval(
            step=final_step, save=True, eval_idx=eval_idx + 1, force_save=True
        )

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )

    def collect_student_states(self, max_steps: int, deterministic: bool = False):
        if max_steps <= 0:
            return None

        states = []
        state, _ = self.env.reset(seed=self.seed)
        episode_seed = self.seed

        while len(states) < max_steps:
            states.append(np.asarray(state, dtype=np.float32).copy())
            with torch.no_grad():
                action, _ = self.policy(state, deterministic=deterministic)
                action_np = (
                    action.cpu().numpy().squeeze(0) if action.shape[-1] > 1 else [action.item()]
                )

            next_state, _, term, trunc, _ = self.env.step(np.argmax(action_np))
            state = next_state

            if term or trunc:
                episode_seed += 1
                state, _ = self.env.reset(seed=episode_seed)

        return np.asarray(states, dtype=np.float32)

    def _run_eval(self, step, save=False, eval_idx=None, force_save=False):
        """Single evaluation pass shared by baseline / periodic / final hooks."""
        self.policy.eval()
        eval_dict, running_video = self.evaluate()

        if self.policy.state_visitation is not None:
            visitation_map = self.policy.state_visitation
            vmin, vmax = visitation_map.min(), visitation_map.max()
            visitation_map = (visitation_map - vmin) / (vmax - vmin + 1e-8)
            visitation_map = self.visitation_to_rgb(visitation_map)
            self.write_image(
                image=visitation_map, step=step,
                logdir="Image", name="visitation map",
            )

        self.write_log(eval_dict, step=step, eval_log=True)
        self.write_video(
            running_video, step=step, logdir="videos", name="running_video"
        )
        self.last_return_mean.append(eval_dict["eval/return_mean"])

        if save:
            self.save_model(step, eval_idx=eval_idx, force=force_save)

    def evaluate(self):
        ep_buffer = []
        image_array = []
        for num_episodes in range(self.eval_num):
            ep_reward, ep_inf = [], []

            # Env initialization — vary seed per episode for diverse rollouts.
            state, infos = self.env.reset(seed=self.seed + num_episodes)

            for t in range(self.env.max_steps):
                with torch.no_grad():
                    t0 = time.time()
                    a, _ = self.policy(state)
                    t1 = time.time()
                    ep_inf.append(t1 - t0)
                    a = a.cpu().numpy().squeeze(0) if a.shape[-1] > 1 else [a.item()]

                if num_episodes == 0 and self.rendering:
                    image = self.env.render()
                    image_array.append(image)

                next_state, rew, term, trunc, infos = self.env.step(np.argmax(a))
                if t == self.env.max_steps - 1:
                    # safe truncation
                    trunc = True
                done = term or trunc

                state = next_state
                ep_reward.append(rew)

                if done:
                    ep_buffer.append(
                        {
                            "return": self.discounted_return(
                                ep_reward, self.policy.gamma
                            ),
                            "inf_time": np.mean(ep_inf),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        inf_time_list = [ep_info["inf_time"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        inf_time_mean, inf_time_std = np.mean(inf_time_list), np.std(inf_time_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
            f"eval/inf_time_mean": inf_time_mean,
            f"eval/inf_time_std": inf_time_std,
        }

        return eval_dict, image_array

    def discounted_return(self, rewards, gamma):
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
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

    def save_model(self, e, eval_idx=None, force=False):
        model = self.policy.actor
        if model is None:
            raise ValueError("Error: Model is not identifiable!!!")

        model_cpu = deepcopy(model).to("cpu")

        # best_model: higher return is better.
        if np.mean(self.last_return_mean) >= self.last_max_return_mean:
            best_path = os.path.join(self.logger.log_dir, "best_model.pth")
            torch.save(model_cpu.state_dict(), best_path)
            self.last_max_return_mean = np.mean(self.last_return_mean)

        # Periodic step-keyed checkpoint: every 10th eval, or forced (final).
        periodic_due = (
            eval_idx is not None and eval_idx > 0 and eval_idx % 10 == 0
        )
        if not (force or periodic_due):
            return

        torch.save(
            model_cpu.state_dict(),
            os.path.join(self.logger.checkpoint_dir, f"model_{e}.pth"),
        )

    def visitation_to_rgb(self, visitation_map: np.ndarray) -> np.ndarray:
        visitation_map = np.squeeze(visitation_map)  # Make sure it's 2D
        H, W = visitation_map.shape

        rgb_map = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # Zero visitation → gray
        zero_mask = visitation_map == 0
        rgb_map[zero_mask] = [0.5, 0.5, 0.5]

        # Nonzero visitation → white → blue gradient
        nonzero_mask = visitation_map > 0
        blue_intensity = visitation_map[nonzero_mask]

        rgb_map[nonzero_mask] = np.stack(
            [
                1.0 - blue_intensity,  # Red
                1.0 - blue_intensity,  # Green
                np.ones_like(blue_intensity),  # Blue
            ],
            axis=-1,
        )

        return rgb_map
