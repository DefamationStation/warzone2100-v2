from __future__ import annotations

import torch

from warzone_tactical.environment import BatchedEpisodeEnvironment


def test_shared_episode_clock_staggers_only_the_initial_reset() -> None:
    first = BatchedEpisodeEnvironment(
        64,
        300,
        torch.device("cpu"),
        19,
        stagger_initial_episode_phase=True,
    )
    second = BatchedEpisodeEnvironment(
        64,
        300,
        torch.device("cpu"),
        19,
        stagger_initial_episode_phase=True,
    )
    mask = torch.ones(64, dtype=torch.bool)
    first.reset_episode_clock(mask, initial=True)
    second.reset_episode_clock(mask, initial=True)

    assert torch.equal(first.episode_step, second.episode_step)
    assert torch.unique(first.episode_step).numel() > 1
    assert int(first.episode_step.min()) >= 0
    assert int(first.episode_step.max()) < 300

    reset_mask = torch.zeros(64, dtype=torch.bool)
    reset_mask[:2] = True
    first.reset_episode_clock(reset_mask)
    assert first.episode_step[:2].tolist() == [0, 0]
    assert torch.equal(first.episode_step[2:], second.episode_step[2:])
