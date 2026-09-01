"""Run the complete compiled-engine Contract V1 validation sequence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .engine_harness import run_acceptance, run_one
from .golden import check_native_actions, load_golden_rows
from .hit_rate import compare_hit_rates
from .trace_fidelity import load_trace, replay_combat_order, replay_movement
from .simulator import ResolvedStats


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fidelity(trace: Path, stats_path: Path, ticks: int = 100) -> dict[str, object]:
    rows = load_trace(trace)
    stats = ResolvedStats.load(stats_path)
    movement = replay_movement(rows, stats, ticks)
    combat = replay_combat_order(rows, stats, ticks)
    movement_passes = (
        movement.position_exact
        and movement.body_heading_exact
        and movement.speed_exact
        and movement.weapon_time_exact
    )
    projectile_passes = (
        movement.damage_exact
        and movement.maximum_simulator_damage_lead_ticks == 0
        and movement.maximum_simulator_damage_lag_ticks <= 100
    )
    combat_passes = (
        combat.turret_exact
        and combat.cooldown_exact
        and combat.shot_eligibility_exact
        and combat.masks_exact
        and combat.combat_order_alignment_mismatches == 0
        and combat.combat_order_shot_mismatches == 0
        and combat.maximum_target_distance_error == 0
    )
    return {
        "movement": asdict(movement),
        "combat_order": asdict(combat),
        "combat_order_resolution": "eliminated_by_engine_tick_start_snapshot",
        "movement_gate_passes": movement_passes,
        "projectile_timing_gate_passes": projectile_passes,
        "combat_order_gate_passes": combat_passes,
        "fidelity_gate_passes": movement_passes and projectile_passes and combat_passes,
    }


def run_revalidation(
    executable: Path,
    repository: Path,
    output: Path,
    battles: int = 100,
    trained_manifest: Path | None = None,
    golden_required: int = 10_000,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    acceptance_dir = output / "acceptance"
    acceptance = run_acceptance(executable, repository, acceptance_dir, battles)

    trace_paths = sorted(acceptance_dir.glob("seed-*/actions-*.jsonl"))
    rows = load_golden_rows(trace_paths, required=golden_required)
    fixture_model = acceptance_dir / "native-fixture" / "policy.wzml"
    fixture_rows = [
        json.loads(line)
        for line in (acceptance_dir / "native-loader" / "actions-0.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    check_native_actions(fixture_rows, fixture_model)
    golden = {
        "golden_states": len(rows),
        "observation_q15_exact": True,
        "fixture_action_rows": len(fixture_rows),
        "fixture_actions_exact": True,
    }
    _write(output / "golden-parity.json", golden)

    forced = run_one(
        executable,
        repository,
        output / "forced-hit",
        0,
        spec_overrides={
            "duration_ticks": 10_000,
            "forced_hit_roll": 0,
            "fire_stop_ticks": 8_000,
            "paired_combat": True,
        },
    )
    fidelity = _fidelity(
        output / "forced-hit" / "actions-0.jsonl",
        output / "forced-hit" / "resolved-stats.json",
    )
    _write(output / "fidelity.json", fidelity)

    hit_rates: dict[str, object] = {}
    for band in ("short", "long"):
        hit_run = run_one(
            executable,
            repository,
            output / f"hit-rate-{band}",
            0,
            spec_overrides={
                "duration_ticks": 100,
                "hit_rate_samples": 100_000,
                "hit_rate_band": band,
            },
        )
        hit_rates[band] = compare_hit_rates(hit_run.report)
    _write(output / "hit-rates.json", hit_rates)

    trained_actions_exact: bool | None = None
    if trained_manifest is not None:
        manifest = json.loads(trained_manifest.read_text(encoding="utf-8"))
        model_path = Path(str(manifest["wzml_path"]))
        trained_run = run_one(
            executable,
            repository,
            output / "trained-model",
            0,
            policy=f"native:{trained_manifest.resolve()}",
        )
        trained_rows = list(trained_run.trace_rows)
        check_native_actions(trained_rows, model_path)
        trained_actions_exact = True

    summary = {
        "contract": "warzone-tactical-v1",
        "acceptance": acceptance,
        "golden": golden,
        "fidelity_gate_passes": fidelity["fidelity_gate_passes"],
        "hit_rates": hit_rates,
        "trained_actions_exact": trained_actions_exact,
        "passes": bool(
            acceptance["same_seed_actions_identical"]
            and acceptance["same_seed_crc_identical"]
            and golden["observation_q15_exact"]
            and golden["fixture_actions_exact"]
            and fidelity["fidelity_gate_passes"]
            and all(bool(value["passes"]) for value in hit_rates.values())
            and trained_actions_exact is not False
        ),
    }
    _write(output / "revalidation-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--trained-manifest", type=Path)
    parser.add_argument("--golden-required", type=int, default=10_000)
    args = parser.parse_args()
    print(
        json.dumps(
            run_revalidation(
                args.executable,
                args.repository,
                args.output,
                args.battles,
                args.trained_manifest,
                args.golden_required,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
