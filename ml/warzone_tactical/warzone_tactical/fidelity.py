"""Promotion checks for matched engine and simulator traces."""

from __future__ import annotations

from dataclasses import dataclass


ACCEPTED_V1_DIVERGENCES = (
    "V1-COMBAT-001: bumpTime-gated target lead is not simulated; track it before mixed weapons or splash damage",
    "V1-COMBAT-003: center-distance projectile timing may lag engine surface collision by at most one 100 ms update; simulator lead is forbidden",
)


@dataclass(frozen=True, slots=True)
class FidelityResult:
    collision_free_exact: bool
    turret_exact: bool
    cooldown_exact: bool
    damage_exact: bool
    masks_exact: bool
    maximum_collision_position_error: float
    projectile_timing_error_ticks: int
    engine_hit_rate: float
    simulator_hit_rate: float
    combat_order_alignment_mismatches: int = 0
    combat_order_shot_mismatches: int = 0
    maximum_target_distance_error: float = 0.0
    accepted_divergences: tuple[str, ...] = ACCEPTED_V1_DIVERGENCES

    @property
    def passes(self) -> bool:
        ordering_clean = (
            self.combat_order_alignment_mismatches == 0
            and self.combat_order_shot_mismatches == 0
        )
        ordering_adjudicated = any(
            divergence.startswith("V1-COMBAT-002")
            for divergence in self.accepted_divergences
        )
        timing_adjudicated = self.projectile_timing_error_ticks == 0 or any(
            divergence.startswith("V1-COMBAT-003")
            for divergence in self.accepted_divergences
        )
        return (
            self.collision_free_exact
            and self.turret_exact
            and self.cooldown_exact
            and self.damage_exact
            and self.masks_exact
            and self.maximum_collision_position_error < 32.0
            and self.projectile_timing_error_ticks <= 100
            and timing_adjudicated
            and abs(self.engine_hit_rate - self.simulator_hit_rate) < 0.01
            and (ordering_clean or ordering_adjudicated)
        )


def compare_rows(
    engine_rows: list[dict[str, object]],
    simulator_rows: list[dict[str, object]],
    *,
    collision: bool,
    engine_hits: int = 0,
    simulator_hits: int = 0,
    shots: int = 100_000,
) -> FidelityResult:
    if len(engine_rows) != len(simulator_rows):
        raise ValueError("matched traces must have the same number of rows")
    exact_position = True
    turret_exact = True
    cooldown_exact = True
    damage_exact = True
    masks_exact = True
    maximum_error = 0.0
    maximum_target_distance_error = 0.0
    projectile_error = 0
    alignment_mismatches = 0
    shot_mismatches = 0
    for engine, simulator in zip(engine_rows, simulator_rows, strict=True):
        engine_position = engine["position"]
        simulator_position = simulator["position"]
        error = max(abs(float(a) - float(b)) for a, b in zip(engine_position[:2], simulator_position[:2], strict=True))
        maximum_error = max(maximum_error, error)
        exact_position &= error == 0 and engine["body_heading"] == simulator["body_heading"]
        turret_exact &= engine["turret_heading"] == simulator["turret_heading"] and engine["turret_aligned"] == simulator["turret_aligned"]
        cooldown_exact &= engine.get("cooldown") == simulator.get("cooldown")
        damage_exact &= engine.get("body") == simulator.get("body")
        masks_exact &= engine["target_mask"] == simulator["target_mask"]
        alignment_mismatches += int(engine["turret_aligned"] != simulator["turret_aligned"])
        if "shot_fired" in engine and "shot_fired" in simulator:
            shot_mismatches += int(engine["shot_fired"] != simulator["shot_fired"])
        if "target_distance" in engine and "target_distance" in simulator:
            maximum_target_distance_error = max(
                maximum_target_distance_error,
                abs(float(engine["target_distance"]) - float(simulator["target_distance"])),
            )
        projectile_error = max(
            projectile_error,
            abs(int(engine.get("impact_time", 0)) - int(simulator.get("impact_time", 0))),
        )
    return FidelityResult(
        collision_free_exact=exact_position if not collision else True,
        turret_exact=turret_exact,
        cooldown_exact=cooldown_exact,
        damage_exact=damage_exact,
        masks_exact=masks_exact,
        maximum_collision_position_error=maximum_error if collision else 0.0,
        projectile_timing_error_ticks=projectile_error,
        engine_hit_rate=engine_hits / shots,
        simulator_hit_rate=simulator_hits / shots,
        combat_order_alignment_mismatches=alignment_mismatches,
        combat_order_shot_mismatches=shot_mismatches,
        maximum_target_distance_error=maximum_target_distance_error,
    )
