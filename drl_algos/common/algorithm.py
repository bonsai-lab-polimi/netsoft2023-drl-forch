from abc import ABC, abstractmethod
from typing import Optional

import gym
import numpy as np
import numpy.typing as npt
from torch.utils.tensorboard import SummaryWriter


class Algorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        seed: int,
        device: str,
        tensorboard_log: str,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.writer = SummaryWriter(log_dir=tensorboard_log)

    @abstractmethod
    def learn(self, num_timesteps: int) -> None:
        """
        Executes the training loop for num_timesteps iterations.
        """

    @abstractmethod
    def predict(self, obs: npt.NDArray, masks: Optional[npt.NDArray] = None) -> npt.NDArray:
        """
        Predict the policy action from an observation
        """
