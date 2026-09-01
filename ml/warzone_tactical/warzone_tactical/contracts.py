"""Frozen integer action and observation contracts for Contract V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

OBSERVATION_SIZE = 200
SELF_OFFSET = 0
RAY_OFFSET = 20
ALLY_OFFSET = 28
ENEMY_OFFSET = 98
RECOVERY_OFFSET = 186
CONTEXT_OFFSET = 190
ALLY_SLOTS = 7
ENEMY_SLOTS = 8
Q15_SCALE = 1 << 15


@dataclass(frozen=True, slots=True)
class QuantizedAction:
    """An action that can enter synchronized game state."""

    heading_delta_q: int
    speed_fraction_q: int
    target_slot_q: int
    fire_q: int

    def validate(self) -> None:
        if not -2048 <= self.heading_delta_q <= 2047:
            raise ValueError("heading_delta_q is outside Contract V1")
        if not 0 <= self.speed_fraction_q <= 256:
            raise ValueError("speed_fraction_q is outside Contract V1")
        if not -1 <= self.target_slot_q < ENEMY_SLOTS:
            raise ValueError("target_slot_q is outside Contract V1")
        if self.fire_q not in (0, 1):
            raise ValueError("fire_q is outside Contract V1")


def q15_to_float(observation_q15: Tensor) -> Tensor:
    """Convert signed Q15 with the exact power-of-two model scale."""

    if observation_q15.dtype != torch.int16:
        raise TypeError("observation must use torch.int16")
    return observation_q15.to(torch.float32) * (2.0**-15)


def _lround(value: Tensor) -> Tensor:
    """Match C++ std::lround half-away-from-zero behavior."""

    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def quantize_policy_output(
    output: Tensor, target_mask: Tensor, heading_output_scale: int = 2048
) -> Tensor:
    """Convert 12 actor outputs to [heading, speed, target, fire] integers."""

    if output.shape[-1] != 12:
        raise ValueError("the Contract V1 actor must have 12 outputs")
    if target_mask.shape != output.shape[:-1] + (ENEMY_SLOTS,):
        raise ValueError("target mask has the wrong shape")

    if not 1 <= heading_output_scale <= 2048:
        raise ValueError("heading output scale must be in [1, 2048]")
    heading = _lround(torch.tanh(output[..., 0]) * heading_output_scale).to(torch.int64)
    heading.clamp_(-2048, 2047)
    speed = _lround((torch.tanh(output[..., 1]) + 1.0) * 128.0).to(torch.int64)
    speed.clamp_(0, 256)

    target_logits = output[..., 2:11]
    none_logit = target_logits[..., :1]
    enemy_logits = target_logits[..., 1:].masked_fill(~target_mask, -torch.inf)
    target = torch.cat((none_logit, enemy_logits), dim=-1).argmax(dim=-1).to(torch.int64) - 1
    fire = (output[..., 11] > 0).to(torch.int64)
    fire = fire.masked_fill(target < 0, 0)
    return torch.stack((heading, speed, target, fire), dim=-1)


def action_mask_from_alive_visible_hostile(
    alive: Tensor, visible: Tensor, hostile: Tensor
) -> Tensor:
    """Build the exact V1 mask for eight fixed enemy slots."""

    return alive.to(torch.bool) & visible.to(torch.bool) & hostile.to(torch.bool)
