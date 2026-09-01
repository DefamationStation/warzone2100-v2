"""Shared batched-environment state for tactical training stages."""

from __future__ import annotations

import torch
from torch import Tensor


class BatchedEpisodeEnvironment:
    """Provide a reproducible episode clock for batched training environments."""

    def __init__(
        self,
        arenas: int,
        max_episode_steps: int,
        device: torch.device,
        seed: int,
        *,
        stagger_initial_episode_phase: bool,
    ) -> None:
        self.arenas = arenas
        self.max_episode_steps = max_episode_steps
        self.device = device
        self.stagger_initial_episode_phase = stagger_initial_episode_phase
        self.episode_step = torch.zeros(arenas, dtype=torch.int64, device=device)
        self._episode_phase_generator = torch.Generator(device=device)
        self._episode_phase_generator.manual_seed(seed)

    def reset_episode_clock(self, mask: Tensor, *, initial: bool = False) -> None:
        """Reset selected clocks, with a seeded phase spread on the first reset."""

        if initial and self.stagger_initial_episode_phase:
            phases = torch.randint(
                0,
                self.max_episode_steps,
                (self.arenas,),
                generator=self._episode_phase_generator,
                device=self.device,
            )
            self.episode_step[mask] = phases[mask]
        else:
            self.episode_step[mask] = 0
