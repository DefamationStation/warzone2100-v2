"""Read-only checkpoint diagnosis for the Stage 2 train/evaluation gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import TacticalActor
from .simulator import ResolvedStats
from .stage2 import Stage2Environment, _distance, drive_policy_action, range_metrics


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    stats: ResolvedStats,
    device: torch.device,
    *,
    deterministic: bool,
    episodes: int = 1000,
    action_seed: int = 0,
) -> float:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = TacticalActor().to(device).eval()
    actor.load_state_dict(checkpoint["actor"], strict=True)
    log_std = checkpoint["drive_log_std"].to(device)
    heading_scale = int(checkpoint["config"]["heading_output_scale"])
    scenario_seeds = torch.arange(episodes, dtype=torch.int64, device=device)
    environment = Stage2Environment(
        episodes,
        stats,
        device,
        0,
        300,
        scenario_seeds,
    )
    observation = environment.observation()
    torch.manual_seed(action_seed)
    torch.cuda.manual_seed_all(action_seed)
    for _tick in range(300):
        drive, _latent, _log_probability, _entropy = drive_policy_action(
            actor,
            log_std,
            observation,
            deterministic=deterministic,
            heading_output_scale=heading_scale,
        )
        observation, _reward, _done = environment.step(drive, auto_reset=False)
    return float((environment.in_band_steps / 300).mean().item())


@torch.no_grad()
def far_spawn_trajectory(
    checkpoint_path: Path,
    stats: ResolvedStats,
    device: torch.device,
    episodes: int = 1000,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = TacticalActor().to(device).eval()
    actor.load_state_dict(checkpoint["actor"], strict=True)
    log_std = checkpoint["drive_log_std"].to(device)
    heading_scale = int(checkpoint["config"]["heading_output_scale"])
    scenario_seeds = torch.arange(episodes, dtype=torch.int64, device=device)
    environment = Stage2Environment(
        episodes,
        stats,
        device,
        0,
        300,
        scenario_seeds,
    )
    observation = environment.observation()
    initial_distance = _distance(environment.simulator)
    far_index = int(initial_distance.argmax().item())
    rows: list[dict[str, int | bool]] = []
    for tick in range(1, 101):
        distance = _distance(environment.simulator)
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
                "distance": int(distance[far_index].item()),
                "heading_delta_q": int(drive[far_index, 0].item()),
                "speed_fraction_q": int(drive[far_index, 1].item()),
                "in_band": bool(in_band[far_index].item()),
            }
        )
        observation, _reward, _done = environment.step(drive, auto_reset=False)
    return {
        "scenario_seed": far_index,
        "initial_distance": int(initial_distance[far_index].item()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint diagnosis needs CUDA")
    device = torch.device("cuda")
    stats = ResolvedStats.load(args.resolved_stats)
    results: list[dict[str, object]] = []
    for seed in (4, 17, 31):
        seed_dir = args.run / f"seed-{seed}"
        final_training = json.loads(
            seed_dir.joinpath("training.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )["mean_completed_time_in_band_fraction"]
        checkpoint_path = seed_dir / "checkpoint.pt"
        deterministic = evaluate_checkpoint(
            checkpoint_path, stats, device, deterministic=True
        )
        stochastic = evaluate_checkpoint(
            checkpoint_path, stats, device, deterministic=False
        )
        results.append(
            {
                "seed": seed,
                "training": final_training,
                "deterministic": deterministic,
                "stochastic": stochastic,
            }
        )
    trajectory = far_spawn_trajectory(args.run / "seed-31" / "checkpoint.pt", stats, device)
    print(json.dumps({"results": results, "trajectory": trajectory}, indent=2))


if __name__ == "__main__":
    main()
