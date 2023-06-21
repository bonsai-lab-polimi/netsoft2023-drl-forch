from typing import NamedTuple

import gym.spaces
import numpy as np
import torch


class ReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    obs_next: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: str,
        buffer_size: int = 1000000,
    ):
        self.buffer_size = buffer_size
        self.device = device

        self.pos = 0
        self.full = False

        self.obs_shape = np.prod(observation_space.shape)
        self.action_dim = action_space.n
        self.observations = np.zeros(shape=(self.buffer_size, self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, 1), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
        (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array, dtype=torch.float).to(self.device)
        return torch.as_tensor(array, dtype=torch.float).to(self.device)

    def add(self, obs, action, reward, next_obs, done):
        """
        Add a sample to the replay buffer.
        """
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Get batch_size random samples from the replay buffer.
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        """
        Get the samples corresponding to batch_inds from the buffer.
        """
        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds],
            self.rewards[batch_inds],
            self.next_observations[batch_inds, :],
            self.dones[batch_inds],
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
