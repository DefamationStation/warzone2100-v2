"""Read-only fixed evaluation for the three Stage 2 training conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from .diagnose_stage2_eval import evaluate_checkpoint, far_spawn_trajectory
from .simulator import ResolvedStats


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "minimum": min(values), "maximum": max(values)}


def _last_training_metric(seed_dir: Path) -> float:
    last_line = seed_dir.joinpath("training.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    return float(json.loads(last_line)["mean_completed_time_in_band_fraction"])


@torch.no_grad()
def evaluate_condition(
    label: str,
    run: Path,
    stats: ResolvedStats,
    device: torch.device,
) -> dict[str, object]:
    report = json.loads(run.joinpath("stage2-report.json").read_text(encoding="utf-8"))
    report_by_seed = {int(row["seed"]): row for row in report["seeds"]}
    rows: list[dict[str, object]] = []
    for seed in range(12):
        seed_dir = run / f"seed-{seed}"
        checkpoint = seed_dir / "checkpoint.pt"
        deterministic_stored = float(
            report_by_seed[seed]["evaluation"]["mean_time_in_band_fraction"]
        )
        deterministic_recomputed = evaluate_checkpoint(
            checkpoint, stats, device, deterministic=True
        )
        stochastic = evaluate_checkpoint(
            checkpoint, stats, device, deterministic=False, action_seed=0
        )
        rows.append(
            {
                "seed": seed,
                "deterministic": deterministic_stored,
                "deterministic_recomputed": deterministic_recomputed,
                "deterministic_exact": deterministic_stored == deterministic_recomputed,
                "stochastic": stochastic,
                "final_on_policy": _last_training_metric(seed_dir),
                "stage1_retention": float(report_by_seed[seed]["stage1_clear_rate_after"]),
            }
        )
    deterministic = [float(row["deterministic"]) for row in rows]
    stochastic = [float(row["stochastic"]) for row in rows]
    on_policy = [float(row["final_on_policy"]) for row in rows]
    retention = [float(row["stage1_retention"]) for row in rows]
    return {
        "label": label,
        "run": str(run),
        "deterministic": _summary(deterministic),
        "stochastic": _summary(stochastic),
        "deterministic_seed_count_above_50_percent": sum(value > 0.5 for value in deterministic),
        "final_on_policy": _summary(on_policy),
        "stage1_retention": _summary(retention),
        "all_deterministic_recomputations_exact": all(
            bool(row["deterministic_exact"]) for row in rows
        ),
        "seeds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--condition", nargs=2, action="append", metavar=("LABEL", "RUN"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("fixed checkpoint evaluation needs CUDA")
    device = torch.device("cuda")
    stats = ResolvedStats.load(args.resolved_stats)
    conditions = [
        evaluate_condition(label, Path(run), stats, device)
        for label, run in args.condition
    ]
    best_condition = max(conditions, key=lambda row: float(row["deterministic"]["mean"]))
    best_seed_row = max(best_condition["seeds"], key=lambda row: float(row["deterministic"]))
    best_run = Path(str(best_condition["run"]))
    trajectory = far_spawn_trajectory(
        best_run / f"seed-{best_seed_row['seed']}" / "checkpoint.pt",
        stats,
        device,
    )
    result = {
        "conditions": conditions,
        "best_condition": best_condition["label"],
        "best_seed": best_seed_row["seed"],
        "trajectory": trajectory,
    }
    encoded = json.dumps(result, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
