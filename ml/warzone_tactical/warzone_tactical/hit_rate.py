"""Compare engine and simulator integer hit-roll rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compare_hit_rates(report: dict[str, object], seed: int = 0) -> dict[str, object]:
    samples = int(report["hit_rate_samples"])
    if samples <= 0:
        raise ValueError("the engine report has no hit-rate samples")
    engine_hits = int(report["hit_rate_hits"])
    chance = int(report["hit_rate_chance"])
    if chance < 0 or chance > 99:
        raise ValueError("the engine report has an invalid resolved hit chance")
    generator = torch.Generator().manual_seed(seed)
    simulator_hits = int(
        (torch.randint(0, 100, (samples,), generator=generator) <= chance).sum()
    )
    engine_rate = engine_hits / samples
    simulator_rate = simulator_hits / samples
    difference = abs(engine_rate - simulator_rate)
    return {
        "samples": samples,
        "range_band": str(report.get("hit_rate_band", "")),
        "displayed_hit_chance": chance,
        "successful_integer_rolls": chance + 1,
        "engine_hits": engine_hits,
        "simulator_hits": simulator_hits,
        "engine_rate": engine_rate,
        "simulator_rate": simulator_rate,
        "rate_difference": difference,
        "passes": difference < 0.01,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = compare_hit_rates(report, args.seed)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
