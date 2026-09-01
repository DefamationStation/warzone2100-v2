from __future__ import annotations

from warzone_tactical.fidelity import ACCEPTED_V1_DIVERGENCES, FidelityResult, compare_rows


def test_bump_time_divergence_is_reported() -> None:
    result = FidelityResult(
        collision_free_exact=True,
        turret_exact=True,
        cooldown_exact=True,
        damage_exact=True,
        masks_exact=True,
        maximum_collision_position_error=0.0,
        projectile_timing_error_ticks=0,
        engine_hit_rate=0.76,
        simulator_hit_rate=0.76,
    )
    assert result.accepted_divergences == ACCEPTED_V1_DIVERGENCES
    assert "bumpTime" in result.accepted_divergences[0]


def test_combat_order_measurements_are_reported() -> None:
    shared = {
        "position": [0, 0, 0],
        "body_heading": 0,
        "turret_heading": 0,
        "cooldown": 0,
        "body": 100,
        "target_mask": [True, False],
    }
    engine = shared | {"turret_aligned": True, "shot_fired": True, "target_distance": 100}
    simulator = shared | {"turret_aligned": False, "shot_fired": False, "target_distance": 103}

    result = compare_rows([engine], [simulator], collision=False)

    assert result.combat_order_alignment_mismatches == 1
    assert result.combat_order_shot_mismatches == 1
    assert result.maximum_target_distance_error == 3


def test_unadjudicated_combat_order_mismatch_blocks_promotion() -> None:
    common = dict(
        collision_free_exact=True,
        turret_exact=True,
        cooldown_exact=True,
        damage_exact=True,
        masks_exact=True,
        maximum_collision_position_error=0.0,
        projectile_timing_error_ticks=0,
        engine_hit_rate=0.76,
        simulator_hit_rate=0.76,
        combat_order_shot_mismatches=1,
    )

    assert not FidelityResult(**common).passes
    accepted = FidelityResult(
        **common,
        accepted_divergences=ACCEPTED_V1_DIVERGENCES
        + ("V1-COMBAT-002: measured sequential combat bound",),
    )
    assert accepted.passes
