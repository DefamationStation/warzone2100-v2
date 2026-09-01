from __future__ import annotations

import torch

from warzone_tactical.curriculum import potential_shaping, sample_stage_ids, shaping_weight
from warzone_tactical.evaluation import Results, passes_incumbent_gate, passes_three_seed_gate


def test_rehearsal_mix_is_80_20() -> None:
    stages = sample_stage_ids(2, 1000, torch.Generator().manual_seed(4))
    assert int((stages == 2).sum()) == 800
    assert int((stages < 2).sum()) == 200


def test_stage1_does_not_rehearse_an_earlier_stage() -> None:
    stages = sample_stage_ids(1, 1000, torch.Generator().manual_seed(4))
    assert torch.equal(stages, torch.ones(1000, dtype=torch.int64))


def test_shaping_reaches_zero_in_final_quarter() -> None:
    assert shaping_weight(5, 0.0, 0.1) == 0.1
    assert shaping_weight(7, 0.75, 0.1) == 0.0
    current = torch.tensor([1.0, 2.0])
    next_value = torch.tensor([3.0, 4.0])
    terminal = torch.tensor([False, True])
    assert torch.allclose(potential_shaping(current, next_value, 0.5, terminal), torch.tensor([0.5, -2.0]))


def test_three_seed_gate_keeps_five_point_floor() -> None:
    good = (Results(560, 440, 0, 0), Results(550, 450, 0, 0), Results(540, 460, 0, 0))
    bad_seed = (Results(650, 350, 0, 0), Results(650, 350, 0, 0), Results(490, 510, 0, 0))
    assert passes_three_seed_gate(good, 0.55, 0.10)
    assert not passes_three_seed_gate(bad_seed, 0.55, 0.10)


def test_incumbent_gate_uses_500_engine_matches_per_seed() -> None:
    side_a = Results(140, 100, 0, 10)
    side_b = Results(135, 105, 0, 10)
    assert passes_incumbent_gate(((side_a, side_b), (side_a, side_b), (side_a, side_b)))
