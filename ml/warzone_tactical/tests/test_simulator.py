from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch

from warzone_tactical.simulator import (
    GAME_TICKS_PER_SEC,
    UPDATE_TICKS,
    ResolvedStats,
    TacticalSimulator,
    integer_floor_sqrt,
    quantise_fraction,
)

FIXTURE = Path(__file__).parent / "fixtures" / "resolved_stats_v1.json"


def _single_shooter(simulator: TacticalSimulator, distance: int) -> tuple[torch.Tensor, torch.Tensor]:
    simulator.alive[:] = False
    simulator.alive[0, 0] = True
    simulator.alive[0, 2] = True
    simulator.body[0, 0] = simulator.stats.health
    simulator.body[0, 2] = simulator.stats.health
    simulator.position_x[0, 0] = 1000
    simulator.position_y[0, 0] = 1000
    simulator.position_x[0, 2] = 1000 + distance
    simulator.position_y[0, 2] = 1000
    simulator.direction[0, 0] = 65536 // 4
    simulator.move_direction[0, 0] = 65536 // 4
    simulator.turret_direction[0, 0] = 0
    target_pitch, _ = simulator._target_pitch_and_required(
        simulator.position_x[:, [2]].expand_as(simulator.position_x),
        simulator.position_y[:, [2]].expand_as(simulator.position_y),
        simulator.position_z[:, [2]].expand_as(simulator.position_z),
    )
    simulator.turret_pitch[0, 0] = target_pitch[0, 0]
    simulator.speed[0, 0] = 0
    action = torch.zeros((1, simulator.max_units, 4), dtype=torch.int64)
    action[..., 2] = -1
    action[0, 0, 2] = 0
    action[0, 0, 3] = 1
    forced_roll = torch.zeros((1, simulator.max_units), dtype=torch.int64)
    return action, forced_roll


def test_quantise_fraction_carries_absolute_tick_phase() -> None:
    numerator = torch.tensor([-11, 11, 19], dtype=torch.int64)
    old_time = torch.tensor([7, 7, 7], dtype=torch.int64)
    new_time = old_time + UPDATE_TICKS
    actual = quantise_fraction(numerator, GAME_TICKS_PER_SEC, new_time, old_time)
    expected = torch.tensor([-1, 1, 2], dtype=torch.int64)
    assert torch.equal(actual, expected)


def test_displayed_75_percent_hit_uses_76_integer_rolls() -> None:
    rolls = torch.arange(100)
    assert int((rolls <= 75).sum()) == 76


@pytest.mark.parametrize("device", ["cpu"])
def test_simulator_is_repeatable(device: str) -> None:
    stats = ResolvedStats.load(FIXTURE)
    first = TacticalSimulator(4, stats, device=device, seed=9)
    second = TacticalSimulator(4, stats, device=device, seed=9)
    action = torch.zeros((4, 16, 4), dtype=torch.int64)
    action[..., 1] = 256
    action[..., 2] = 0
    action[..., 3] = 1
    forced_roll = torch.zeros((4, 16), dtype=torch.int64)
    for _ in range(100):
        first.step(action.clone(), forced_roll)
        second.step(action.clone(), forced_roll)
    for name in (
        "game_time",
        "position_x",
        "position_y",
        "direction",
        "move_direction",
        "speed",
        "turret_direction",
        "body",
        "alive",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name)), name


def test_empty_mask_reset_does_not_change_simulator_state() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(4, stats, device="cpu", seed=9)
    before = {
        name: getattr(simulator, name).clone()
        for name in ("game_time", "position_x", "position_y", "direction", "body", "alive")
    }

    simulator.reset(torch.zeros(4, dtype=torch.bool))

    for name, expected in before.items():
        assert torch.equal(getattr(simulator, name), expected), name


def test_target_slots_do_not_compact_after_a_death() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=2)
    before, before_mask = simulator._enemy_indices_and_mask()
    first_enemy = before[0, 0, 0].item()
    simulator.alive[0, first_enemy] = False
    after, after_mask = simulator._enemy_indices_and_mask()
    assert before_mask[0, 0, 0]
    assert after[0, 0, 0].item() == first_enemy
    assert not after_mask[0, 0, 0]


def test_out_of_range_fire_does_not_change_weapon_or_target() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=3)
    action, forced_roll = _single_shooter(simulator, stats.long_range + 1)
    _, target_mask = simulator.build_observation()
    assert target_mask[0, 0, 0]
    old_last_fired = simulator.last_fired[0, 0].item()
    old_target_body = simulator.body[0, 2].item()

    simulator.step(action, forced_roll)

    assert simulator.last_fired[0, 0].item() == old_last_fired
    assert simulator.body[0, 2].item() == old_target_body
    assert not (simulator.projectile_impact_time[0, 0] >= 0).any()


def test_below_minimum_range_fire_is_blocked() -> None:
    stats = replace(ResolvedStats.load(FIXTURE), min_range=100)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=4)
    action, forced_roll = _single_shooter(simulator, stats.min_range - 1)
    old_last_fired = simulator.last_fired[0, 0].item()

    simulator.step(action, forced_roll)

    assert simulator.last_fired[0, 0].item() == old_last_fired


def test_last_fired_uses_engine_fire_time() -> None:
    stats = replace(ResolvedStats.load(FIXTURE), fire_pause=5)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=5)
    action, forced_roll = _single_shooter(simulator, stats.min_range + 1)
    simulator.game_time[:] = 500
    simulator.last_fired[0, 0] = 0

    simulator.step(action, forced_roll)

    assert simulator.last_fired[0, 0].item() == 401


def test_turret_pitch_must_align_before_fire() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=5)
    action, forced_roll = _single_shooter(simulator, stats.min_range + 1)
    simulator.turret_pitch[0, 0] = 0
    old_last_fired = simulator.last_fired[0, 0].item()

    simulator.step(action, forced_roll)

    assert simulator.last_fired[0, 0].item() == old_last_fired
    assert simulator.turret_pitch[0, 0].item() != 0


def test_projectile_overflow_is_counted() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=6)
    action, forced_roll = _single_shooter(simulator, stats.min_range + 1)
    simulator.projectile_impact_time[0, 0] = 1_000_000

    simulator.step(action, forced_roll)

    assert simulator.projectile_drop_count[0].item() == 1


def test_projectile_damage_is_visible_on_the_next_ml_update() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=6)
    simulator.alive[:] = False
    simulator.alive[0, 2] = True
    simulator.body[0, 2] = stats.health
    simulator.projectile_impact_time[0, 0, 0] = 500
    simulator.projectile_target[0, 0, 0] = 2
    simulator.projectile_damage[0, 0, 0] = stats.hit_damage
    simulator.game_time[:] = 500
    action = torch.zeros((1, simulator.max_units, 4), dtype=torch.int64)
    action[..., 2] = -1

    simulator.step(action)
    assert simulator.body[0, 2].item() == stats.health

    simulator.step(action)
    assert simulator.body[0, 2].item() == stats.health - stats.hit_damage


def test_step_does_not_mutate_the_callers_action() -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=7)
    action = torch.tensor([[[4096, 999, 99, 7]] * simulator.max_units], dtype=torch.int64)
    original = action.clone()

    simulator.step(action)

    assert torch.equal(action, original)


def test_enemy_slot_data_is_built_once_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = ResolvedStats.load(FIXTURE)
    simulator = TacticalSimulator(1, stats, device="cpu", seed=8)
    action = torch.zeros((1, simulator.max_units, 4), dtype=torch.int64)
    calls = 0
    original = simulator._enemy_indices_and_mask

    def counted() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(simulator, "_enemy_indices_and_mask", counted)
    simulator.step(action)
    assert calls == 1


def test_integer_floor_sqrt_matches_math_isqrt() -> None:
    generator = torch.Generator().manual_seed(9)
    values = torch.randint(0, 1 << 33, (10_000,), dtype=torch.int64, generator=generator)
    squares = torch.tensor([0, 1, 4, 9, 65535**2, 65536**2, 92681**2], dtype=torch.int64)
    values = torch.cat((values, squares, squares + 1, torch.clamp(squares - 1, min=0)))
    expected = torch.tensor([math.isqrt(int(value)) for value in values], dtype=torch.int64)
    assert torch.equal(integer_floor_sqrt(values), expected)
