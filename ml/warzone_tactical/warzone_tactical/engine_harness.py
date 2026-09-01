"""Headless engine acceptance harness for the fixed 2v2 integration test."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class EngineRun:
    seed: int
    report: dict[str, object]
    action_trace: tuple[tuple[int, ...], ...]
    trace_rows: tuple[dict[str, object], ...]
    crc_trace: bytes


def _read_actions(path: Path) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.append((int(row["game_time"]), int(row["droid_id"]), *map(int, row["action"])))
    return tuple(rows)


def _read_trace(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def _read_scenario_crc(path: Path, start_game_time: int, end_game_time: int) -> bytes:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        game_time = int(line.split(maxsplit=1)[0])
        if start_game_time <= game_time <= end_game_time:
            lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _expect_engine_failure(**kwargs: object) -> None:
    try:
        run_one(**kwargs)
    except RuntimeError:
        return
    raise AssertionError("the engine accepted a fatal ML contract error")


def run_one(
    executable: Path,
    repository: Path,
    output_directory: Path,
    seed: int,
    policy: str = "scripted",
    spec_overrides: dict[str, object] | None = None,
) -> EngineRun:
    output_directory.mkdir(parents=True, exist_ok=True)
    spec = output_directory / f"scenario-{seed}.json"
    report = output_directory / f"report-{seed}.json"
    actions = output_directory / f"actions-{seed}.jsonl"
    crc = output_directory / f"crc-{seed}.txt"
    resolved = output_directory / "resolved-stats.json"
    spec_data: dict[str, object] = {
        "seed": seed,
        "duration_ticks": 90_000,
        "report_path": str(report),
    }
    if spec_overrides:
        spec_data.update(spec_overrides)
    spec.write_text(json.dumps(spec_data), encoding="utf-8")
    command = [
        str(executable),
        "--autogame",
        "--headless",
        "--nosound",
        "--skirmish=ml-v1.json",
        f"--ml-policy={policy}",
        "--ml-players=0,1",
        f"--ml-combat-test={spec}",
        f"--ml-trace={actions}",
        f"--ml-resolved-stats={resolved}",
        f"--gamestate-crc-trace={crc}",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"engine seed {seed} failed with exit {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not report.is_file() or not actions.is_file() or not crc.is_file():
        raise RuntimeError(f"engine seed {seed} did not write all acceptance artifacts")
    report_data = json.loads(report.read_text(encoding="utf-8"))
    start_game_time = int(report_data["start_game_time"])
    end_game_time = int(report_data["end_game_time"])
    trace_rows = tuple(
        row for row in _read_trace(actions) if int(row["game_time"]) <= end_game_time
    )
    for row in trace_rows:
        if int(row["target_id"]) < 0 and int(row["action"][3]) != 0:
            raise AssertionError(f"seed {seed} requested fire with no valid target")
    return EngineRun(
        seed=seed,
        report=report_data,
        action_trace=tuple(
            (int(row["game_time"]), int(row["droid_id"]), *map(int, row["action"]))
            for row in trace_rows
        ),
        trace_rows=trace_rows,
        crc_trace=_read_scenario_crc(crc, start_game_time, end_game_time),
    )


def run_acceptance(executable: Path, repository: Path, output: Path, battles: int = 100) -> dict[str, object]:
    runs = [run_one(executable, repository, output / f"seed-{seed}", seed) for seed in range(battles)]
    repeat_a = run_one(executable, repository, output / "repeat-a", 0)
    repeat_b = run_one(executable, repository, output / "repeat-b", 0)
    if repeat_a.action_trace != repeat_b.action_trace:
        raise AssertionError("same-seed quantized action traces differ")
    if repeat_a.crc_trace != repeat_b.crc_trace:
        raise AssertionError("same-seed game-state CRC traces differ")
    turret_values = {(int(row["droid_id"]), int(row["turret_heading"])) for row in repeat_a.trace_rows}
    if len(turret_values) <= 4:
        raise AssertionError("turret movement is not visible in the action trace")
    if not any(int(row["weapon_last_fired"]) > 0 for row in repeat_a.trace_rows):
        raise AssertionError("weapon firing delay is not visible in the action trace")
    range_run = run_one(
        executable,
        repository,
        output / "range-invariant",
        0,
        spec_overrides={"duration_ticks": 200, "range_invariant": True},
    )
    blocked_rows = [
        row
        for row in range_run.trace_rows
        if bool(row["requested_fire"])
        and int(row["target_distance"]) == int(row["long_range"]) + 1
    ]
    if not blocked_rows:
        raise AssertionError("the engine range fixture did not request fire at long_range + 1")
    for row in blocked_rows:
        if bool(row["shot_fired"]):
            raise AssertionError("the engine fired beyond long_range")
        if int(row["weapon_last_fired"]) != int(row["weapon_last_fired_before"]):
            raise AssertionError("out-of-range fire changed last_fired")
        if int(row["target_body_after"]) != int(row["target_body_before"]):
            raise AssertionError("out-of-range fire changed target body")
    from .wzml import HEADER, export_artifacts, make_fixture_actor

    fixture_directory = output / "native-fixture"
    manifest = export_artifacts(
        make_fixture_actor(), fixture_directory, output / "seed-0" / "resolved-stats.json"
    )
    native_run = run_one(
        executable,
        repository,
        output / "native-loader",
        0,
        policy=f"native:{fixture_directory / 'policy.manifest.json'}",
    )
    if not native_run.action_trace:
        raise AssertionError("native fixture did not complete the action path")
    stop_turn_actor = make_fixture_actor()
    with torch.no_grad():
        final_layer = stop_turn_actor.layers[-1]
        final_layer.bias.zero_()
        final_layer.bias[0] = 4.0
        final_layer.bias[1] = -4.0
        final_layer.bias[2] = 1.0
        final_layer.bias[3] = -1.0
        final_layer.bias[11] = -1.0
    stop_turn_directory = output / "stop-turn-fixture"
    export_artifacts(
        stop_turn_actor,
        stop_turn_directory,
        output / "seed-0" / "resolved-stats.json",
    )
    stop_turn_run = run_one(
        executable,
        repository,
        output / "stop-turn",
        0,
        policy=f"native:{stop_turn_directory / 'policy.manifest.json'}",
        spec_overrides={"duration_ticks": 500},
    )
    first_droid = int(stop_turn_run.trace_rows[0]["droid_id"])
    stop_turn_rows = [row for row in stop_turn_run.trace_rows if int(row["droid_id"]) == first_droid]
    if len({tuple(row["position"][:2]) for row in stop_turn_rows}) != 1:
        raise AssertionError("the zero-speed turn fixture moved the droid")
    if len({int(row["body_heading"]) for row in stop_turn_rows}) <= 1:
        raise AssertionError("a stopped ML droid did not turn")
    valid_manifest_path = fixture_directory / "policy.manifest.json"
    valid_manifest = json.loads(valid_manifest_path.read_text(encoding="utf-8"))
    mismatch_manifest = dict(valid_manifest)
    mismatch_manifest["resolved_stats_sha256"] = "0" * 64
    mismatch_manifest_path = fixture_directory / "policy.stats-mismatch.manifest.json"
    mismatch_manifest_path.write_text(json.dumps(mismatch_manifest, indent=2) + "\n", encoding="utf-8")
    _expect_engine_failure(
        executable=executable,
        repository=repository,
        output_directory=output / "stats-mismatch",
        seed=0,
        policy=f"native:{mismatch_manifest_path}",
    )

    model_bytes = bytearray((fixture_directory / "policy.wzml").read_bytes())
    final_bias_offset = HEADER.size + (200 * 256 + 256 + 256 * 256 + 256 + 256 * 12) * 4
    model_bytes[final_bias_offset : final_bias_offset + 4] = struct.pack("<f", float("nan"))
    nonfinite_model_path = fixture_directory / "policy.nonfinite.wzml"
    nonfinite_model_path.write_bytes(model_bytes)
    nonfinite_manifest = dict(valid_manifest)
    nonfinite_manifest["wzml_path"] = str(nonfinite_model_path.resolve())
    nonfinite_manifest["wzml_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    nonfinite_manifest_path = fixture_directory / "policy.nonfinite.manifest.json"
    nonfinite_manifest_path.write_text(json.dumps(nonfinite_manifest, indent=2) + "\n", encoding="utf-8")
    _expect_engine_failure(
        executable=executable,
        repository=repository,
        output_directory=output / "nonfinite-model",
        seed=0,
        policy=f"native:{nonfinite_manifest_path}",
    )
    results: dict[str, int] = {
        "team_0_win": 0,
        "team_1_win": 0,
        "draw": 0,
        "unfinished": 0,
    }
    for run in runs:
        result = str(run.report["result"])
        results[result] = results.get(result, 0) + 1
    summary: dict[str, object] = {
        "contract": "warzone-tactical-v1",
        "battles": battles,
        "results": results,
        "same_seed_actions_identical": True,
        "same_seed_crc_identical": True,
        "turret_motion_visible": True,
        "weapon_delay_visible": True,
        "out_of_range_fire_blocked": True,
        "invalid_target_fire_requests": 0,
        "native_fixture_loaded": True,
        "zero_speed_body_turns": True,
        "stats_mismatch_fails_loudly": True,
        "nonfinite_model_fails_loudly": True,
        "native_fixture_wzml_sha256": manifest["wzml_sha256"],
    }
    (output / "acceptance-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--battles", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_acceptance(args.executable, args.repository, args.output, args.battles), indent=2))


if __name__ == "__main__":
    main()
