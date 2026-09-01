"""Frozen stage, rehearsal, and promotion definitions for Contract V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class StageGate:
    stage: int
    name: str
    minimum_win_or_success: float
    maximum_unfinished: float | None = None
    extra_gate: str = ""
    time_limit_ticks: int | None = None


STAGE_GATES = (
    StageGate(0, "engine_plumbing", 1.00, 0.00, "All integration gates pass."),
    StageGate(1, "stationary_target_clear", 0.95, None, "This is a valid-target and fire plumbing gate. It is not evidence of complex learning."),
    StageGate(
        2,
        "approach_and_hold_range",
        0.70,
        None,
        "The metric is mean episode time in the open-closed band (short_range, long_range]. Each seed must reach 0.65. Mean normalized band violation must be at most 0.10.",
        30_000,
    ),
    StageGate(3, "fixed_1v1", 0.60, 0.10, "Use a 60 second kill limit.", 60_000),
    StageGate(4, "shared_area", 0.90, None, "Ally collision failures are at most 5 percent.", 90_000),
    StageGate(5, "two_vs_strong_one", 0.80, None, "Both-unit survival is at least 60 percent."),
    StageGate(6, "recovery_choice", 0.80, None, "Healthy recovery use is below 10 percent."),
    StageGate(7, "fixed_2v2", 0.55, 0.10, "All retention gates pass."),
)


def sample_stage_ids(current_stage: int, episodes: int, generator: torch.Generator) -> Tensor:
    """Return the fixed 80/20 current-stage and earlier-stage mixture."""

    if current_stage < 0 or current_stage > 7:
        raise ValueError("stage is outside Contract V1")
    if current_stage < 2:
        return torch.full((episodes,), current_stage, dtype=torch.int64)
    current_count = round(episodes * 0.8)
    stages = torch.full((episodes,), current_stage, dtype=torch.int64)
    stages[current_count:] = torch.randint(0, current_stage, (episodes - current_count,), generator=generator)
    permutation = torch.randperm(episodes, generator=generator)
    return stages[permutation]


def shaping_weight(stage: int, stage_fraction: float, initial_weight: float) -> float:
    """Anneal shaping from Stage 5 through the final quarter of Stage 7."""

    if stage < 5:
        return initial_weight
    progress = ((stage - 5) + min(max(stage_fraction, 0.0), 1.0)) / 2.75
    return initial_weight * max(0.0, 1.0 - progress)


def potential_shaping(current: Tensor, next_value: Tensor, gamma: float, terminal: Tensor) -> Tensor:
    terminal_next = torch.where(terminal, torch.zeros_like(next_value), next_value)
    return gamma * terminal_next - current
