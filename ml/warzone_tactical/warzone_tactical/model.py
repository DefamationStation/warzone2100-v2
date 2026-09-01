"""Fixed Contract V1 actor and a training-only critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import OBSERVATION_SIZE, q15_to_float, quantize_policy_output


class TacticalActor(nn.Module):
    """The only actor shape that the native V1 runtime accepts."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 12),
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.dtype == torch.int16:
            observation = q15_to_float(observation)
        return self.layers(observation)

    @torch.no_grad()
    def deterministic_action(self, observation_q15: Tensor, target_mask: Tensor) -> Tensor:
        return quantize_policy_output(self(observation_q15), target_mask)


class CentralCritic(nn.Module):
    """A training-only centralized value function for up to 16 units."""

    def __init__(self, max_units: int = 16) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE * max_units, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, team_observations: Tensor) -> Tensor:
        if team_observations.dtype == torch.int16:
            team_observations = q15_to_float(team_observations)
        return self.value(team_observations.flatten(start_dim=-2)).squeeze(-1)
