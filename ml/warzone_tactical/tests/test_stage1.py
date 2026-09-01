from __future__ import annotations

import torch

from warzone_tactical.model import TacticalActor
from warzone_tactical.simulator import ResolvedStats
from warzone_tactical.stage1 import (
    Stage1Environment,
    compute_gae,
    masked_target_logits,
    policy_action,
)


def _stats() -> ResolvedStats:
    return ResolvedStats(
        base_speed=127,
        calculated_max_speed=125,
        acceleration=250,
        deceleration=800,
        skid_deceleration=600,
        turn_speed=60,
        spin_speed=136,
        spin_angle=65,
        health=290,
        armour=10,
        damage=22,
        hit_damage=16,
        short_range=512,
        long_range=1088,
        min_range=128,
        short_hit_chance=40,
        long_hit_chance=50,
        fire_pause=1000,
        projectile_speed=2250,
        turret_rotation_rate=3276,
        turret_pitch_rate=1638,
        muzzle_connector_x=0,
        muzzle_connector_y=10,
        muzzle_connector_z=23,
        min_elevation=-10922,
        max_elevation=16384,
        direct_weapon=True,
        fire_on_move=True,
    )


def test_stage1_reset_has_two_or_three_id_sorted_target_slots() -> None:
    environment = Stage1Environment(64, _stats(), torch.device("cpu"), seed=7)
    simulator = environment.simulator
    _observation, mask = environment.observation()
    target_count = mask.sum(dim=-1)
    assert bool(((target_count == 2) | (target_count == 3)).all())
    ids = torch.gather(simulator.unit_id, 1, simulator.enemy_slots[:, 0, :3])
    sentinel = torch.full_like(ids, 1 << 30)
    ids = torch.where(mask[:, :3], ids, sentinel)
    assert bool((ids[:, 1:] >= ids[:, :-1]).all())
    assert not bool((simulator.speed != 0).any())
    active_body = torch.where(
        simulator.alive[:, 1:4], simulator.body[:, 1:4], -torch.ones_like(simulator.body[:, 1:4])
    )
    for row in active_body.tolist():
        health = [value for value in row if value >= 0]
        assert len(health) == len(set(health))


def test_stage1_scenario_seed_is_independent_of_batch_size() -> None:
    first_seeds = torch.arange(8, dtype=torch.int64)
    small = Stage1Environment(
        8, _stats(), torch.device("cpu"), seed=3, scenario_seeds=first_seeds
    )
    large = Stage1Environment(
        16,
        _stats(),
        torch.device("cpu"),
        seed=9,
        scenario_seeds=torch.arange(16, dtype=torch.int64),
    )
    for name in (
        "game_time",
        "position_x",
        "position_y",
        "direction",
        "body",
        "alive",
        "unit_id",
        "enemy_slots",
        "enemy_slot_assigned",
    ):
        assert torch.equal(
            getattr(small.simulator, name), getattr(large.simulator, name)[:8]
        )
    assert torch.equal(small.random_state, large.random_state[:8])


def test_stage1_invalid_target_logits_are_masked() -> None:
    output = torch.zeros((2, 12))
    output[:, 3:11] = 100
    mask = torch.zeros((2, 8), dtype=torch.bool)
    logits = masked_target_logits(output, mask)
    assert bool(torch.isneginf(logits[:, 1:]).all())
    actor = TacticalActor()
    observation = torch.zeros((2, 200), dtype=torch.int16)
    action, _log_probability, _entropy = policy_action(
        actor, observation, mask, deterministic=True
    )
    assert action[:, 0].tolist() == [-1, -1]
    assert action[:, 1].tolist() == [0, 0]


def test_stage1_gae_stops_at_completed_episodes() -> None:
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    done = torch.tensor([[False], [True], [False]])
    value = torch.zeros_like(reward)
    advantage, returns = compute_gae(
        reward, done, value, torch.zeros(1), gamma=1.0, gae_lambda=1.0
    )
    assert advantage[:, 0].tolist() == [3.0, 2.0, 3.0]
    assert torch.equal(advantage, returns)
