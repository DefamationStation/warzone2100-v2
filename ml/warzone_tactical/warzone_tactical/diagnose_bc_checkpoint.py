"""Read-only fixed diagnostics for the Stage 2 behaviour-cloning checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import TacticalActor
from .simulator import ResolvedStats
from .stage2 import Stage2Environment, _distance, drive_policy_action, range_metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the BC checkpoint diagnostic needs CUDA")

    device = torch.device("cuda")
    stats = ResolvedStats.load(args.resolved_stats)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = TacticalActor().to(device).eval()
    actor.load_state_dict(checkpoint["actor"], strict=True)
    heading_scale = int(checkpoint["config"]["heading_output_scale"])
    log_std = checkpoint["drive_log_std"].to(device)

    scenario_seeds = torch.arange(1000, dtype=torch.int64, device=device)
    environment = Stage2Environment(1000, stats, device, 0, 300, scenario_seeds)
    observation = environment.observation()
    initial_distance = _distance(environment.simulator)
    for _tick in range(300):
        drive, _latent, _log_probability, _entropy = drive_policy_action(
            actor,
            log_std,
            observation,
            deterministic=True,
            heading_output_scale=heading_scale,
        )
        observation, _reward, _done = environment.step(drive, auto_reset=False)
    fraction = environment.in_band_steps / 300
    final_distance = _distance(environment.simulator).to(torch.float32)
    masks = {
        "near": initial_distance <= stats.short_range,
        "in_band": (initial_distance > stats.short_range)
        & (initial_distance <= stats.long_range),
        "far": initial_distance > stats.long_range,
    }
    split = {
        name: {
            "episodes": int(mask.sum().item()),
            "mean_time_in_band_fraction": float(fraction[mask].mean().item()),
            "mean_final_distance": float(final_distance[mask].mean().item()),
        }
        for name, mask in masks.items()
    }

    seed = torch.tensor([894], dtype=torch.int64, device=device)
    trajectory_environment = Stage2Environment(1, stats, device, 0, 300, seed)
    observation = trajectory_environment.observation()
    rows: list[dict[str, int | bool]] = []
    for tick in range(1, 301):
        distance = _distance(trajectory_environment.simulator)
        in_band, _violation = range_metrics(distance, stats.short_range, stats.long_range)
        drive, _latent, _log_probability, _entropy = drive_policy_action(
            actor,
            log_std,
            observation,
            deterministic=True,
            heading_output_scale=heading_scale,
        )
        rows.append(
            {
                "tick": tick,
                "distance": int(distance[0].item()),
                "heading_delta_q": int(drive[0, 0].item()),
                "speed_fraction_q": int(drive[0, 1].item()),
                "in_band": bool(in_band[0].item()),
            }
        )
        observation, _reward, _done = trajectory_environment.step(
            drive, auto_reset=False
        )

    result = {
        "split": split,
        "trajectory": {"scenario_seed": 894, "rows": rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
