from typing import Optional

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPAgent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.critic(x)

    def get_action(self, x: torch.Tensor, deterministic=True) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if deterministic:
            return dist.mode()
        return dist.sample()

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.flatten(x, start_dim=1)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


class EquivariantLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.Gamma = nn.Linear(in_channels, out_channels, bias=False)
        self.Lambda = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        xm, _ = torch.max(x, dim=1, keepdim=True)
        return self.Lambda(x) - self.Gamma(xm)


class EquivariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ReLU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        return torch.squeeze(self.net(x), dim=-1)


class InvariantDeepSet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.psi = nn.Sequential(
            EquivariantLayer(in_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels),
            nn.ELU(),
            EquivariantLayer(hidden_channels, hidden_channels),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, n_elements, in_channels)
        # return: (batch_size, n_elements)
        x = torch.mean(self.psi(x), dim=1)
        return torch.squeeze(self.rho(x), dim=-1)


class DeepSetAgent(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv) -> None:
        super().__init__()
        in_channels = envs.observation_space.shape[1]

        # Actor outputs pi(a|s)
        self.actor = EquivariantDeepSet(in_channels)

        # Critic outputs V(s)
        self.critic = InvariantDeepSet(in_channels)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(self, x: torch.Tensor, masks: Optional[torch.Tensor] = None, deterministic: bool = True) -> torch.Tensor:
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        dist = Categorical(logits=logits)
        if deterministic:
            return dist.mode
        return dist.sample()

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None, masks: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        if masks is not None:
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype)
            logits = torch.where(masks, logits, HUGE_NEG)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)
