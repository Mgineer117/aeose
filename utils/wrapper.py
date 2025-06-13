from io import BytesIO

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        tot_count = self.count + count

        new_mean = self.mean + delta * count / tot_count
        m_a = self.var * self.count
        m_b = var * count
        M2 = m_a + m_b + np.square(delta) * self.count * count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ObsNormWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip_obs=10.0, epsilon=1e-8):
        super().__init__(env)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        obs_shape = self.observation_space.shape
        self.rms = RunningMeanStd(shape=obs_shape)

    def observation(self, obs):
        self.rms.update(obs[np.newaxis, ...])
        norm_obs = (obs - self.rms.mean) / (np.sqrt(self.rms.var) + self.epsilon)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)

    def __getattr__(self, name):
        # Forward any unknown attribute to the inner environment
        return getattr(self.env, name)


class FetchWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, episode_len: int, seed: int):
        super(FetchWrapper, self).__init__(env)

        self.max_steps = episode_len
        self.seed = seed

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        observation = np.concatenate(
            (
                # observation_dict["observation"],
                observation_dict["desired_goal"],
                observation_dict["achieved_goal"],
            )
        )

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                # observation_dict["observation"],
                observation_dict["desired_goal"],
                observation_dict["achieved_goal"],
            )
        )
        return observation, reward, termination, truncation, info

    def get_rewards_heatmap(self, extractor: torch.nn.Module, eigenvectors: np.ndarray):

        state, _ = self.reset(seed=self.seed)
        dg = state[-6:-3]  # desired goal pos
        del state
        self.close()

        X = [0.5, 1.5]
        Y = [0, 1.0]
        Z = [0.5, 1.0]

        # Define the ranges and number of increments per dimension
        x_vals = np.linspace(X[0], X[1], num=10)
        y_vals = np.linspace(Y[0], Y[1], num=10)
        z_vals = np.linspace(Z[0], Z[1], num=10)

        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

        # Flatten to create a batch of shape (N, 3)
        features = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        # goal_states = np.repeat(dg.reshape(1, 3), current_states.shape[0], axis=0)
        # features = np.concatenate([goal_states, current_states], axis=1)

        images = []
        for idx, vector in enumerate(eigenvectors):
            # Compute reward as dot product
            rewards = features @ vector  # shape: (N,)
            # rewards = achieved_goals @ vector  # shape: (N,)
            neg_idx = rewards < 0
            pos_idx = rewards >= 0

            # Normalize positive values to [0, 1]
            if np.any(pos_idx):
                pos_max, pos_min = (
                    rewards[pos_idx].max(),
                    rewards[pos_idx].min(),
                )
                if pos_max != pos_min:
                    rewards[pos_idx] = (rewards[pos_idx] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )

            # Normalize negative values to [-1, 0]
            if np.any(neg_idx):
                neg_max, neg_min = (
                    rewards[neg_idx].max(),
                    rewards[neg_idx].min(),
                )
                if neg_max != neg_min:
                    rewards[neg_idx] = (rewards[neg_idx] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0

            # Create 3D scatter plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(
                features[:, 0],
                features[:, 1],
                features[:, 2],
                c=rewards,
                cmap=cm.seismic,
                s=30,
            )

            # Mark desired goal
            ax.scatter(
                dg[0],
                dg[1],
                dg[2],
                color="red",
                edgecolors="green",
                s=150,
                marker="*",
                label="Desired Goal",
            )

            ax.set_title(f"Rewards for Eigenvector {idx}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.colorbar(sc, ax=ax, label="Normalized Reward")
            ax.legend()

            # Convert plot to image
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            images.append(np.array(image))
            buf.close()

        return images


class PointMazeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, maze_map: list, episode_len: int, seed: int):
        super(PointMazeWrapper, self).__init__(env)

        self.maze_map = maze_map
        self.max_steps = episode_len
        self.seed = seed

    def reset(self, **kwargs):
        observation_dict, info = self.env.reset(**kwargs)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["desired_goal"],
                observation_dict["achieved_goal"],
            )
        )

        return observation, info

    def step(self, action):
        # Call the original step method
        observation_dict, reward, termination, truncation, info = self.env.step(action)
        observation = np.concatenate(
            (
                observation_dict["observation"],
                observation_dict["desired_goal"],
                observation_dict["achieved_goal"],
            )
        )
        return observation, reward, termination, truncation, info

    def get_rewards_heatmap(
        self,
        extractor: torch.nn.Module,
        eigenvectors: np.ndarray,
    ):
        # Get desired goal (2D)
        state, _ = self.reset(seed=self.seed)
        dg = state[-4:-2]
        del state
        self.close()

        # Maze size
        example_map = self.maze_map
        maze_height = len(example_map)
        maze_width = len(example_map[0])

        # Spatial bounds and discretization
        cell_size = 1.0
        x_low, x_high = 0, maze_width * cell_size
        y_low, y_high = 0, maze_height * cell_size
        resolution = 80

        x_vals = np.linspace(x_low, x_high, num=resolution)
        y_vals = np.linspace(y_low, y_high, num=resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing="ij")
        features = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

        self.width = resolution
        self.height = resolution

        # Create wall/goal/agent masks
        wall_mask = np.zeros((resolution, resolution), dtype=bool)
        goal_mask = np.zeros((resolution, resolution), dtype=bool)
        agent_mask = np.zeros((resolution, resolution), dtype=bool)

        for i in range(maze_height):
            for j in range(maze_width):
                val = example_map[i][j]
                x_start = j * cell_size
                x_end = (j + 1) * cell_size
                y_start = (maze_height - 1 - i) * cell_size
                y_end = (maze_height - i) * cell_size

                region_mask = (
                    (X_grid >= x_start)
                    & (
                        (X_grid < x_end)
                        if j < maze_width - 1
                        else (X_grid <= (x_end + 1e-4))
                    )
                    & (Y_grid >= y_start)
                    & ((Y_grid < y_end) if i < maze_height - 1 else (Y_grid <= y_end))
                )

                if val == 1:
                    wall_mask |= region_mask
                elif val == "g":
                    goal_mask |= region_mask
                elif val == "r":
                    agent_mask |= region_mask

        # Expand wall_mask for channel-wise masking
        valid_mask = ~wall_mask

        # Loop over each eigenvector to generate heatmaps
        images = []
        for idx, vector in enumerate(eigenvectors):
            rewards = features @ vector

            neg_idx = rewards < 0
            pos_idx = rewards >= 0

            # Normalize positive/negative separately
            if np.any(pos_idx):
                pos_max, pos_min = rewards[pos_idx].max(), rewards[pos_idx].min()
                if pos_max != pos_min:
                    rewards[pos_idx] = (rewards[pos_idx] - pos_min) / (
                        pos_max - pos_min + 1e-4
                    )
                else:
                    rewards[pos_idx] = 1
            if np.any(neg_idx):
                neg_max, neg_min = rewards[neg_idx].max(), rewards[neg_idx].min()
                if neg_max != neg_min:
                    rewards[neg_idx] = (rewards[neg_idx] - neg_min) / (
                        neg_max - neg_min + 1e-4
                    ) - 1.0
                else:
                    rewards[neg_idx] = -1

            reward_map = rewards.reshape(resolution, resolution)

            # Convert reward map to RGB using your function
            rgb_img = self.reward_map_to_rgb(reward_map, valid_mask)
            images.append(rgb_img)

        return images

    def reward_map_to_rgb(self, reward_map: np.ndarray, mask) -> np.ndarray:
        rgb_img = np.zeros((self.width, self.height, 3), dtype=np.float32)

        pos_mask = np.logical_and(mask, (reward_map >= 0))
        neg_mask = np.logical_and(mask, (reward_map < 0))

        # Blue for negative: map [-1, 0] → [1, 0]
        rgb_img[neg_mask, 2] = -reward_map[neg_mask]  # blue channel

        # Red for positive: map [0, 1] → [0, 1]
        rgb_img[pos_mask, 0] = reward_map[pos_mask]  # red channel

        # rgb_img.flatten()[mask] to grey
        rgb_img[~mask, :] = 0.5

        return rgb_img
