import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        buffer_size: int,
        batch_size: int,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size,) + state_dim, dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terminal = np.zeros((buffer_size, 1), dtype=np.float32)

        self.dtype = dtype
        self.device = device

    def pre_process(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def append(self, state, action, next_state, reward, terminal):
        self.state[self.ptr] = self.pre_process(state)
        self.action[self.ptr] = self.pre_process(action)
        self.next_state[self.ptr] = self.pre_process(next_state)
        self.reward[self.ptr] = self.pre_process(reward)
        self.terminal[self.ptr] = self.pre_process(terminal)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.from_numpy(self.state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.action[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.next_state[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.reward[ind]).to(self.device).to(self.dtype),
            torch.from_numpy(self.terminal[ind]).to(self.device).to(self.dtype),
        )
