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
        async_sampling: bool = False,
        checkpoint_interval: float = 1800.0,
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

        self.rendering = rendering
        self.seed = seed
        self.async_sampling = async_sampling
        self.checkpoint_interval = checkpoint_interval

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_return_mean = deque(maxlen=1)

        # === Pre-training baseline evaluation ===
        # Run an eval on the freshly-initialized policy so we have a step-0
        # baseline in the logs. We do NOT save a checkpoint for this — the
        # model is untrained and not worth keeping.
        self._run_eval(step=self.init_timesteps, save=True)

        # `timesteps` is the absolute budget. We resume by starting the bar at
        # init_timesteps and training the remainder, NOT by adding another full
        # budget on top of the resumed step. If a chained job resumes from a
        # checkpoint that already hit the budget, there's nothing left to do.
        total = self.timesteps
        if self.init_timesteps >= total:
            self.logger.print(
                f"{self.policy.name}: resumed at {self.init_timesteps} >= target "
                f"{total} timesteps; training already complete, skipping."
            )
            if hasattr(self.sampler, "close"):
                self.sampler.close()
            return {"max_eval_return": self.last_max_return_mean}

        # Train loop
        # Skip evals that already fired in earlier chunks so a resume doesn't
        # replay a burst of catch-up evaluations.
        eval_idx = self.init_timesteps // self.eval_interval if self.eval_interval > 0 else 0
        total_clock_time = 0
        last_ckpt_time = time.time()
        # Kick off the first rollout before entering the loop so the very
        # first `gather()` returns something. Subsequent dispatches are
        # issued right after gather() but before learn(), which lets the
        # workers run the next rollout concurrently with the SGD update at
        # the cost of one step of policy staleness.
        if self.async_sampling:
            self.sampler.dispatch(self.policy, seed=self.seed)
        with tqdm(
            total=total,
            initial=self.init_timesteps,
            desc=f"{self.policy.name} Training (Timesteps)",
        ) as pbar:
            while pbar.n < total:
                step = pbar.n + 1  # + 1 to avoid zero division
                self.policy.train()

                if self.async_sampling:
                    batch, sample_time = self.sampler.gather(self.policy)
                    # Estimate whether we'll need at least one more batch;
                    # if so, dispatch now so workers roll while we learn.
                    expected_per_call = (
                        self.sampler.total_num_worker
                        * self.sampler.num_samples_per_worker
                    )
                    if pbar.n + expected_per_call < total:
                        self.sampler.dispatch(self.policy, seed=self.seed)
                else:
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
                            batch["rewards"], batch.get("dones", batch.get("terminals")), self.policy.gamma
                        )
                    )

                    self.write_log(loss_dict, step=step)

                    #### Periodic evaluation ####
                    if step >= self.eval_interval * (eval_idx + 1):
                        eval_idx += 1
                        # Run eval without saving; only final eval saves checkpoint
                        self._run_eval(
                            step=step, save=True, eval_idx=eval_idx, force_save=True
                        )

                    # Rolling resume checkpoint on a wall-clock interval, kept
                    # separate from eval cadence so a timeout loses at most
                    # checkpoint_interval of training, not a full eval window.
                    if time.time() - last_ckpt_time >= self.checkpoint_interval:
                        self._save_latest_checkpoint(step)
                        last_ckpt_time = time.time()

                torch.cuda.empty_cache()

        # === Final evaluation (always saves checkpoint + replay buffer) ===
        final_step = pbar.n
        self._run_eval(
            step=final_step, save=True, eval_idx=eval_idx + 1, force_save=True
        )

        # Drain any in-flight async-dispatched batch before shutting workers
        # down, otherwise the result_queue can hold an orphaned rollout.
        if self.async_sampling and hasattr(self.sampler, "drain"):
            try:
                self.sampler.drain(self.policy)
            except Exception:
                pass

        # Shut down sampler workers so they don't leak past training.
        if hasattr(self.sampler, "close"):
            self.sampler.close()

        self.logger.print(
            f"Total {self.policy.name} training time: {(time.time() - start_time) / 3600} hours"
        )
        return {"max_eval_return": self.last_max_return_mean}

    def _run_eval(self, step, save=False, eval_idx=None, force_save=False):
        """Run one evaluation pass and (optionally) checkpoint."""
        self.policy.eval()
        eval_dict, running_video = self.evaluate()
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

            # Env initialization (vary seed per episode so we don't get
            # identical rollouts that artificially zero out the std).
            state, _ = self.env.reset(seed=self.seed + num_episodes)

            for t in range(self.episode_len):
                with torch.no_grad():
                    t0 = time.time()
                    a, _ = self.policy(state)  # , deterministic=True)
                    t1 = time.time()
                    ep_inf.append(t1 - t0)
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
                            "inf_time": np.mean(ep_inf),
                            "image_success": infos.get("image_success_rate", 0),
                            "desat_success": infos.get("desat_success_rate", 0),
                            "downlink_success": infos.get("downlink_success_rate", 0),
                        }
                    )

                    break

        return_list = [ep_info["return"] for ep_info in ep_buffer]
        avg_reward_list = [ep_info["avg_reward"] for ep_info in ep_buffer]
        episode_length_list = [ep_info["episode_length"] for ep_info in ep_buffer]
        inf_time_list = [ep_info["inf_time"] for ep_info in ep_buffer]
        return_mean, return_std = np.mean(return_list), np.std(return_list)
        avg_reward_mean, avg_reward_std = np.mean(avg_reward_list), np.std(
            avg_reward_list
        )
        epi_len_mean, epi_len_std = np.mean(episode_length_list), np.std(
            episode_length_list
        )
        inf_time_mean, inf_time_std = np.mean(inf_time_list), np.std(inf_time_list)

        eval_dict = {
            f"eval/return_mean": return_mean,
            f"eval/return_std": return_std,
            f"eval/avg_reward_mean": avg_reward_mean,
            f"eval/avg_reward_std": avg_reward_std,
            f"eval/epi_len_mean": epi_len_mean,
            f"eval/epi_len_std": epi_len_std,
            f"eval/inf_time_mean": inf_time_mean,
            f"eval/inf_time_std": inf_time_std,
            f"eval/image_success": np.mean([ep["image_success"] for ep in ep_buffer]),
            f"eval/desat_success": np.mean([ep["desat_success"] for ep in ep_buffer]),
            f"eval/downlink_success": np.mean([ep["downlink_success"] for ep in ep_buffer]),
        }

        return eval_dict, image_array

    def average_discounted_return(self, rewards, terminals, gamma):
        """
        Computes the average discounted return across all completed episodes.

        The previous reverse-pass implementation accumulated rewards from
        a later episode into an earlier episode's return (it never reset G
        before crossing a terminal). This version groups rewards per
        episode and discounts each independently.
        """
        episode_returns = []
        ep_rewards = []
        for r, term in zip(rewards, terminals):
            ep_rewards.append(float(r))
            if term:
                G = 0.0
                for rr in reversed(ep_rewards):
                    G = rr + gamma * G
                episode_returns.append(G)
                ep_rewards = []
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

    def _save_latest_checkpoint(self, step):
        """Persist the FULL training state for a faithful resume — actor +
        critic weights, the optimizer's Adam moments, and the observation
        normalizer (via policy.get_training_state()) — plus the replay buffer
        and current timestep. Called both at each eval and on a wall-clock
        interval from the train loop, so a timeout loses at most one interval
        of progress AND resume continues *training* (not just inference).
        Falls back to actor-only for policies without get_training_state."""
        import json

        if hasattr(self.policy, "get_training_state"):
            train_state = self.policy.get_training_state()
        elif getattr(self.policy, "actor", None) is not None:
            train_state = self.policy.actor.state_dict()
        else:
            return

        torch.save(
            train_state, os.path.join(self.logger.log_dir, "latest_model.pth")
        )
        if hasattr(self.policy, "replay_buffer") and self.policy.replay_buffer is not None:
            self.policy.replay_buffer.save_to_json(
                os.path.join(self.logger.log_dir, "latest_buffer.json")
            )
        # Persist the W&B run id so a resumed job re-attaches to the same run.
        state = {"step": int(step)}
        run = getattr(self.logger, "wandb_run", None)
        if run is not None and getattr(run, "id", None):
            state["wandb_id"] = run.id
        with open(os.path.join(self.logger.log_dir, "resume_state.json"), "w") as f:
            json.dump(state, f)

    def save_model(self, e, eval_idx=None, force=False):
        model = self.policy.actor
        if model is None:
            raise ValueError("Error: Model is not identifiable!!!")

        model_cpu = deepcopy(model).to("cpu")

        # Always keep best_model.pth and best_buffer.json aligned.
        if np.mean(self.last_return_mean) >= self.last_max_return_mean:
            best_path = os.path.join(self.logger.log_dir, "best_model.pth")
            torch.save(model_cpu.state_dict(), best_path)
            if (
                hasattr(self.policy, "replay_buffer")
                and self.policy.replay_buffer is not None
            ):
                best_buffer_path = os.path.join(self.logger.log_dir, "best_buffer.json")
                self.policy.replay_buffer.save_to_json(best_buffer_path)
            self.last_max_return_mean = np.mean(self.last_return_mean)

        # Always refresh the full-state 'latest' checkpoint used for resuming.
        self._save_latest_checkpoint(e)

        # Only save checkpoint + replay buffer when forced (final eval).
        if not force:
            return

        torch.save(
            model_cpu.state_dict(),
            os.path.join(self.logger.checkpoint_dir, f"model_{e}.pth"),
        )
        if hasattr(self.policy, "replay_buffer") and self.policy.replay_buffer is not None:
            self.policy.replay_buffer.save_to_json(
                os.path.join(self.logger.checkpoint_dir, f"replay_buffer_{e}.json")
            )
