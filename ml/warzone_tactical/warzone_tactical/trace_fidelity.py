"""Replay engine traces through the integer tactical simulator."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .simulator import ResolvedStats, TacticalSimulator, angle_delta, integer_floor_sqrt


@dataclass(frozen=True, slots=True)
class MovementReport:
    ticks_checked: int
    rows_checked: int
    position_exact: bool
    body_heading_exact: bool
    speed_exact: bool
    body_exact: bool
    damage_exact: bool
    weapon_time_exact: bool
    maximum_position_error: int
    maximum_body_heading_error: int
    maximum_speed_error: int
    maximum_body_error: int
    maximum_damage_timing_error_ticks: int
    maximum_simulator_damage_lead_ticks: int
    maximum_simulator_damage_lag_ticks: int
    engine_damage_events: int
    simulator_damage_events: int
    engine_damage_events_by_droid: dict[int, int]
    simulator_damage_events_by_droid: dict[int, int]
    maximum_weapon_time_error: int
    first_mismatch: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class CombatReport:
    ticks_checked: int
    rows_checked: int
    turret_exact: bool
    cooldown_exact: bool
    shot_eligibility_exact: bool
    masks_exact: bool
    combat_order_alignment_mismatches: int
    combat_order_shot_mismatches: int
    maximum_target_distance_error: int
    first_mismatch: dict[str, object] | None


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Load one engine JSONL action trace."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def group_ticks(
    rows: list[dict[str, Any]], maximum_ticks: int
) -> list[list[dict[str, Any]]]:
    """Group rows by engine game time and keep file order inside each tick."""

    grouped: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        game_time = int(row["game_time"])
        grouped.setdefault(game_time, []).append(row)
    ticks = list(grouped.values())[:maximum_ticks]
    if not ticks:
        raise ValueError("the engine trace has no ticks")
    expected_ids = {int(row["droid_id"]) for row in ticks[0]}
    for tick in ticks:
        ids = [int(row["droid_id"]) for row in tick]
        if len(ids) != len(set(ids)):
            raise ValueError("an engine tick contains a duplicate droid ID")
        if set(ids) != expected_ids:
            raise ValueError("the droid set changed inside the requested trace window")
    return ticks


def _clear_simulator(simulator: TacticalSimulator) -> None:
    simulator.game_time.zero_()
    simulator.position_x.zero_()
    simulator.position_y.zero_()
    simulator.position_z.zero_()
    simulator.direction.zero_()
    simulator.move_direction.zero_()
    simulator.speed.zero_()
    simulator.turret_direction.zero_()
    simulator.turret_pitch.zero_()
    simulator.turret_aligned.zero_()
    simulator.body.zero_()
    simulator.previous_body.zero_()
    simulator.alive.zero_()
    simulator.team.zero_()
    simulator.unit_id.zero_()
    simulator.last_fired.fill_(-simulator.stats.fire_pause)
    simulator.projectile_impact_time.fill_(-1)
    simulator.projectile_target.fill_(-1)
    simulator.projectile_damage.zero_()
    simulator.projectile_drop_count.zero_()
    simulator.previous_action.zero_()
    simulator.enemy_slots.zero_()
    simulator.enemy_slot_assigned.zero_()
    simulator.ally_slots.zero_()
    simulator.ally_slot_assigned.zero_()


def _set_slots(
    simulator: TacticalSimulator,
    arena: int,
    observer: int,
    state: dict[str, Any],
    id_to_index: dict[int, int],
) -> None:
    for slot, entry in enumerate(state["enemies"]):
        assigned_id = int(entry["assigned_id"])
        if assigned_id in id_to_index:
            simulator.enemy_slots[arena, observer, slot] = id_to_index[assigned_id]
            simulator.enemy_slot_assigned[arena, observer, slot] = True
    for slot, entry in enumerate(state["allies"]):
        assigned_id = int(entry["assigned_id"])
        if assigned_id in id_to_index:
            simulator.ally_slots[arena, observer, slot] = id_to_index[assigned_id]
            simulator.ally_slot_assigned[arena, observer, slot] = True


def _initialize_tick(simulator: TacticalSimulator, tick: list[dict[str, Any]]) -> list[int]:
    """Initialize one simulator arena from a complete engine tick snapshot."""

    _clear_simulator(simulator)
    ordered = sorted(tick, key=lambda row: int(row["droid_id"]))
    ids = [int(row["droid_id"]) for row in ordered]
    id_to_index = {droid_id: index for index, droid_id in enumerate(ids)}
    simulator.game_time[0, 0] = int(ordered[0]["golden_state"]["game_time"])
    for index, row in enumerate(ordered):
        state = row["golden_state"]
        own = state["self"]
        if "move_direction" not in own:
            raise ValueError("the engine trace has no move_direction; rebuild and record it again")
        simulator.alive[0, index] = True
        simulator.unit_id[0, index] = int(own["id"])
        simulator.team[0, index] = int(own["player"])
        simulator.position_x[0, index] = int(own["position"][0])
        simulator.position_y[0, index] = int(own["position"][1])
        simulator.position_z[0, index] = int(own["position"][2])
        simulator.direction[0, index] = int(own["body_heading"])
        simulator.move_direction[0, index] = int(own["move_direction"])
        simulator.speed[0, index] = int(own["speed"])
        simulator.turret_direction[0, index] = int(own["turret_heading"])
        simulator.turret_pitch[0, index] = int(own.get("turret_pitch", 0))
        simulator.body[0, index] = int(own["body"])
        simulator.previous_body[0, index] = int(own["previous_body"])
        simulator.last_fired[0, index] = int(own["weapon_last_fired"])
        simulator.previous_action[0, index] = torch.tensor(
            state["previous_action"], dtype=torch.int64, device=simulator.device
        )
        _set_slots(simulator, 0, index, state, id_to_index)
    return ids


def replay_movement(
    rows: list[dict[str, Any]], stats: ResolvedStats, maximum_ticks: int = 100
) -> MovementReport:
    """Replay one action trace continuously and compare post-update movement."""

    ticks = group_ticks(rows, maximum_ticks)
    unit_count = len(ticks[0])
    simulator = TacticalSimulator(1, stats, device="cpu", max_units=unit_count, seed=0)
    ids = _initialize_tick(simulator, ticks[0])
    first_by_id = {int(row["droid_id"]): row for row in ticks[0]}
    last_engine_body = {
        droid_id: int(first_by_id[droid_id]["golden_state"]["self"]["body"])
        for droid_id in ids
    }
    last_simulator_body = dict(last_engine_body)
    engine_damage_times = {droid_id: [] for droid_id in ids}
    simulator_damage_times = {droid_id: [] for droid_id in ids}
    position_exact = True
    body_heading_exact = True
    speed_exact = True
    body_exact = True
    weapon_time_exact = True
    maximum_position_error = 0
    maximum_body_heading_error = 0
    maximum_speed_error = 0
    maximum_body_error = 0
    maximum_weapon_time_error = 0
    first_mismatch: dict[str, object] | None = None

    for tick_index, tick in enumerate(ticks):
        row_by_id = {int(row["droid_id"]): row for row in tick}
        action = torch.zeros((1, unit_count, 4), dtype=torch.int64)
        action[..., 2] = -1
        forced_roll = torch.full((1, unit_count), 99, dtype=torch.int64)
        for unit, droid_id in enumerate(ids):
            row = row_by_id[droid_id]
            action[0, unit] = torch.tensor(row["action"], dtype=torch.int64)
            if int(row.get("shot_hit", -1)) == 1:
                forced_roll[0, unit] = 0
        simulator.step(action, forced_roll)

        for unit, droid_id in enumerate(ids):
            engine = row_by_id[droid_id]
            engine_x, engine_y = map(int, engine["position"][:2])
            simulator_x = int(simulator.position_x[0, unit])
            simulator_y = int(simulator.position_y[0, unit])
            position_error = max(abs(engine_x - simulator_x), abs(engine_y - simulator_y))
            heading_error = abs(
                int(angle_delta(torch.tensor(int(engine["body_heading"]) - int(simulator.direction[0, unit]))))
            )
            speed_error = abs(int(engine["speed"]) - int(simulator.speed[0, unit]))
            body_error = abs(int(engine["body"]) - int(simulator.body[0, unit]))
            engine_body = int(engine["body"])
            simulator_body = int(simulator.body[0, unit])
            if engine_body < last_engine_body[droid_id]:
                engine_damage_times[droid_id].extend(
                    [int(engine["game_time"])]
                    * (last_engine_body[droid_id] - engine_body)
                )
            if simulator_body < last_simulator_body[droid_id]:
                simulator_damage_times[droid_id].extend(
                    [int(engine["game_time"])]
                    * (last_simulator_body[droid_id] - simulator_body)
                )
            last_engine_body[droid_id] = engine_body
            last_simulator_body[droid_id] = simulator_body
            weapon_time_error = abs(
                int(engine["weapon_last_fired"]) - int(simulator.last_fired[0, unit])
            )
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_body_heading_error = max(maximum_body_heading_error, heading_error)
            maximum_speed_error = max(maximum_speed_error, speed_error)
            maximum_body_error = max(maximum_body_error, body_error)
            maximum_weapon_time_error = max(maximum_weapon_time_error, weapon_time_error)
            position_exact &= position_error == 0
            body_heading_exact &= heading_error == 0
            speed_exact &= speed_error == 0
            body_exact &= body_error == 0
            weapon_time_exact &= weapon_time_error == 0
            if first_mismatch is None and (
                position_error or heading_error or speed_error or body_error or weapon_time_error
            ):
                first_mismatch = {
                    "tick_index": tick_index,
                    "game_time": int(engine["game_time"]),
                    "droid_id": droid_id,
                    "engine_position": [engine_x, engine_y],
                    "simulator_position": [simulator_x, simulator_y],
                    "position_error": position_error,
                    "body_heading_error": heading_error,
                    "speed_error": speed_error,
                    "body_error": body_error,
                    "engine_body": engine_body,
                    "simulator_body": simulator_body,
                    "weapon_time_error": weapon_time_error,
                }

    damage_exact = all(
        len(engine_damage_times[droid_id]) == len(simulator_damage_times[droid_id])
        for droid_id in ids
    )
    maximum_damage_timing_error = 0
    maximum_simulator_damage_lead = 0
    maximum_simulator_damage_lag = 0
    if damage_exact:
        for droid_id in ids:
            for engine_time, simulator_time in zip(
                engine_damage_times[droid_id],
                simulator_damage_times[droid_id],
                strict=True,
            ):
                signed_error = engine_time - simulator_time
                maximum_damage_timing_error = max(
                    maximum_damage_timing_error, abs(signed_error)
                )
                maximum_simulator_damage_lead = max(
                    maximum_simulator_damage_lead, signed_error
                )
                maximum_simulator_damage_lag = max(
                    maximum_simulator_damage_lag, -signed_error
                )
    return MovementReport(
        ticks_checked=len(ticks),
        rows_checked=len(ticks) * unit_count,
        position_exact=position_exact,
        body_heading_exact=body_heading_exact,
        speed_exact=speed_exact,
        body_exact=body_exact,
        damage_exact=damage_exact,
        weapon_time_exact=weapon_time_exact,
        maximum_position_error=maximum_position_error,
        maximum_body_heading_error=maximum_body_heading_error,
        maximum_speed_error=maximum_speed_error,
        maximum_body_error=maximum_body_error,
        maximum_damage_timing_error_ticks=maximum_damage_timing_error,
        maximum_simulator_damage_lead_ticks=maximum_simulator_damage_lead,
        maximum_simulator_damage_lag_ticks=maximum_simulator_damage_lag,
        engine_damage_events=sum(map(len, engine_damage_times.values())),
        simulator_damage_events=sum(map(len, simulator_damage_times.values())),
        engine_damage_events_by_droid={
            droid_id: len(engine_damage_times[droid_id]) for droid_id in ids
        },
        simulator_damage_events_by_droid={
            droid_id: len(simulator_damage_times[droid_id]) for droid_id in ids
        },
        maximum_weapon_time_error=maximum_weapon_time_error,
        first_mismatch=first_mismatch,
    )


def _snapshot_units(state: dict[str, Any]) -> list[dict[str, Any]]:
    units: dict[int, dict[str, Any]] = {int(state["self"]["id"]): state["self"]}
    for block in (state["allies"], state["enemies"]):
        for entry in block:
            unit = entry["unit"]
            if unit is not None:
                units[int(unit["id"])] = unit
    return [units[droid_id] for droid_id in sorted(units)]


def replay_combat_order(
    rows: list[dict[str, Any]], stats: ResolvedStats, maximum_ticks: int = 100
) -> CombatReport:
    """Replay each observer snapshot and measure sequential combat differences."""

    ticks = group_ticks(rows, maximum_ticks)
    selected_rows = [row for tick in ticks for row in tick]
    unit_count = len(ticks[0])
    simulator = TacticalSimulator(
        len(selected_rows), stats, device="cpu", max_units=unit_count, seed=0
    )
    _clear_simulator(simulator)
    actions = torch.zeros((len(selected_rows), unit_count, 4), dtype=torch.int64)
    actions[..., 2] = -1
    observer_indices: list[int] = []

    for arena, row in enumerate(selected_rows):
        state = row["golden_state"]
        units = _snapshot_units(state)
        id_to_index = {int(unit["id"]): index for index, unit in enumerate(units)}
        observer = id_to_index[int(row["droid_id"])]
        observer_indices.append(observer)
        simulator.game_time[arena, 0] = int(state["game_time"])
        for index, unit in enumerate(units):
            simulator.alive[arena, index] = True
            simulator.unit_id[arena, index] = int(unit["id"])
            simulator.team[arena, index] = int(unit["player"])
            simulator.position_x[arena, index] = int(unit["position"][0])
            simulator.position_y[arena, index] = int(unit["position"][1])
            simulator.position_z[arena, index] = int(unit["position"][2])
            simulator.direction[arena, index] = int(unit["body_heading"])
            simulator.move_direction[arena, index] = int(unit.get("move_direction", unit["body_heading"]))
            simulator.speed[arena, index] = int(unit["speed"])
            simulator.body[arena, index] = int(unit["body"])
            simulator.previous_body[arena, index] = int(unit.get("previous_body", unit["body"]))
            simulator.last_fired[arena, index] = int(
                unit.get("weapon_last_fired", state["game_time"] - stats.fire_pause)
            )
        own = state["self"]
        simulator.turret_direction[arena, observer] = int(own["turret_heading"])
        simulator.turret_pitch[arena, observer] = int(own.get("turret_pitch", 0))
        simulator.last_fired[arena, observer] = int(own["weapon_last_fired"])
        simulator.previous_action[arena, observer] = torch.tensor(
            state["previous_action"], dtype=torch.int64
        )
        _set_slots(simulator, arena, observer, state, id_to_index)
        actions[arena, observer] = torch.tensor(row["action"], dtype=torch.int64)

    enemy_indices, target_mask = simulator._enemy_indices_and_mask()
    before_last_fired = simulator.last_fired.clone()
    requested = actions[..., 2]
    slot = requested.clamp(0, 7)
    targets = torch.gather(enemy_indices, 2, slot.unsqueeze(-1)).squeeze(-1)
    target_x = torch.gather(simulator.position_x, 1, targets)
    target_y = torch.gather(simulator.position_y, 1, targets)
    target_distance = integer_floor_sqrt(
        (target_x - simulator.position_x) ** 2 + (target_y - simulator.position_y) ** 2
    )
    simulator._combat(
        actions,
        torch.full((len(selected_rows), unit_count), 99, dtype=torch.int64),
        enemy_indices,
        target_mask,
    )

    turret_exact = True
    cooldown_exact = True
    shot_eligibility_exact = True
    masks_exact = True
    alignment_mismatches = 0
    shot_mismatches = 0
    maximum_target_distance_error = 0
    first_mismatch: dict[str, object] | None = None
    for arena, (row, observer) in enumerate(zip(selected_rows, observer_indices, strict=True)):
        selected_slot = int(row["action"][2])
        slot_valid = 0 <= selected_slot < 8 and bool(target_mask[arena, observer, selected_slot])
        aligned = slot_valid and bool(simulator.turret_aligned[arena, observer])
        shot_fired = bool(
            simulator.last_fired[arena, observer] != before_last_fired[arena, observer]
        )
        remaining_cooldown = max(
            0,
            stats.fire_pause
            - (int(simulator.game_time[arena, 0]) - int(simulator.last_fired[arena, observer])),
        )
        shot_eligible = remaining_cooldown == 0
        row_mask = [bool(value) for value in row["target_mask"]]
        simulator_mask = [bool(value) for value in target_mask[arena, observer].tolist()]
        row_turret_exact = (
            int(row["turret_heading"])
            == int(simulator.turret_direction[arena, observer])
            and int(row.get("turret_pitch", 0))
            == int(simulator.turret_pitch[arena, observer])
        )
        row_cooldown_exact = int(row["cooldown"]) == remaining_cooldown
        row_eligibility_exact = bool(row["shot_eligible"]) == shot_eligible
        row_mask_exact = row_mask == simulator_mask
        row_alignment_mismatch = bool(row["turret_aligned"]) != aligned
        row_shot_mismatch = bool(row["shot_fired"]) != shot_fired
        distance_error = 0
        if slot_valid and int(row["target_distance"]) >= 0:
            distance_error = abs(
                int(row["target_distance"]) - int(target_distance[arena, observer])
            )
        turret_exact &= row_turret_exact
        cooldown_exact &= row_cooldown_exact
        shot_eligibility_exact &= row_eligibility_exact
        masks_exact &= row_mask_exact
        alignment_mismatches += int(row_alignment_mismatch)
        shot_mismatches += int(row_shot_mismatch)
        maximum_target_distance_error = max(maximum_target_distance_error, distance_error)
        if first_mismatch is None and not all(
            (
                row_turret_exact,
                row_cooldown_exact,
                row_eligibility_exact,
                row_mask_exact,
                not row_alignment_mismatch,
                not row_shot_mismatch,
                distance_error == 0,
            )
        ):
            first_mismatch = {
                "game_time": int(row["game_time"]),
                "droid_id": int(row["droid_id"]),
                "turret_exact": row_turret_exact,
                "cooldown_exact": row_cooldown_exact,
                "shot_eligibility_exact": row_eligibility_exact,
                "mask_exact": row_mask_exact,
                "alignment_mismatch": row_alignment_mismatch,
                "shot_mismatch": row_shot_mismatch,
                "target_distance_error": distance_error,
            }

    return CombatReport(
        ticks_checked=len(ticks),
        rows_checked=len(selected_rows),
        turret_exact=turret_exact,
        cooldown_exact=cooldown_exact,
        shot_eligibility_exact=shot_eligibility_exact,
        masks_exact=masks_exact,
        combat_order_alignment_mismatches=alignment_mismatches,
        combat_order_shot_mismatches=shot_mismatches,
        maximum_target_distance_error=maximum_target_distance_error,
        first_mismatch=first_mismatch,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--collision", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_trace(args.trace)
    stats = ResolvedStats.load(args.stats)
    movement = replay_movement(rows, stats, args.ticks)
    combat = replay_combat_order(rows, stats, args.ticks)
    result = {
        "contract": "warzone-tactical-v1",
        "trace": str(args.trace),
        "collision_trace": args.collision,
        "movement": asdict(movement),
        "combat_order": asdict(combat),
        "combat_order_resolution": {
            "status": "eliminated_by_engine_tick_start_snapshot",
            "verification": "zero mismatches verifies the snapshot fix; it does not show that live sequential ordering was equal",
            "engine_ml_behavior": "aim, range, turret alignment, and fire use the observation-batch target position",
            "stock_asymmetry": "stock droids use live sequential target positions during the incumbent benchmark",
            "instant_hit_warning": "re-audit the temporary target-position override before an instant-hit weapon is used",
        },
        "movement_gate_passes": (
            movement.maximum_position_error < 32
            if args.collision
            else movement.position_exact and movement.body_heading_exact
        )
        and movement.speed_exact
        and movement.weapon_time_exact,
        "projectile_timing_gate_passes": (
            movement.damage_exact
            and movement.maximum_simulator_damage_lead_ticks == 0
            and movement.maximum_simulator_damage_lag_ticks <= 100
        ),
        "combat_order_gate_passes": (
            combat.turret_exact
            and combat.cooldown_exact
            and combat.shot_eligibility_exact
            and combat.masks_exact
            and combat.combat_order_alignment_mismatches == 0
            and combat.combat_order_shot_mismatches == 0
            and combat.maximum_target_distance_error == 0
        ),
    }
    result["fidelity_gate_passes"] = bool(
        result["movement_gate_passes"]
        and result["projectile_timing_gate_passes"]
        and result["combat_order_gate_passes"]
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
