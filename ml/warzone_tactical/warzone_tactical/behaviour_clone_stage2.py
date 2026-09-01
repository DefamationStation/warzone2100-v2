"""Supervised Stage 2 behaviour-cloning probe for scripted_rangekeeper_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .diagnose_stage2_eval import evaluate_checkpoint
from .model import TacticalActor
from .simulator import ResolvedStats
from .stage2 import Stage2Environment, _distance, drive_policy_action, range_metrics, scripted_drive


@torch.no_grad()
def trajectory_for_seed(
    actor: TacticalActor,
    stats: ResolvedStats,
    device: torch.device,
    scenario_seed: int,
    heading_output_scale: int,
    ticks: int = 100,
) -> list[dict[str, int | bool]]:
    seeds = torch.tensor([scenario_seed], dtype=torch.int64, device=device)
    environment = Stage2Environment(1, stats, device, 0, 300, seeds)
    observation = environment.observation()
    log_std = torch.zeros(2, device=device)
    rows: list[dict[str, int | bool]] = []
    for tick in range(1, ticks + 1):
        distance = _distance(environment.simulator)
        in_band, _violation = range_metrics(distance, stats.short_range, stats.long_range)
        drive, _latent, _log_probability, _entropy = drive_policy_action(
            actor,
            log_std,
            observation,
            deterministic=True,
            heading_output_scale=heading_output_scale,
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
        observation, _reward, _done = environment.step(drive, auto_reset=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--arenas", type=int, default=2048)
    parser.add_argument("--heading-output-scale", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the behaviour-cloning probe needs CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    stats = ResolvedStats.load(args.resolved_stats)
    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    actor = TacticalActor().to(device)
    actor.load_state_dict(stage1["actor"], strict=True)
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    scenario_seeds = 100_000 + torch.arange(args.arenas, dtype=torch.int64, device=device)
    environment = Stage2Environment(
        args.arenas,
        stats,
        device,
        args.seed,
        scenario_seeds=scenario_seeds,
    )
    observation = environment.observation()
    observations: list[torch.Tensor] = []
    teacher_actions: list[torch.Tensor] = []
    with torch.no_grad():
        for _tick in range(environment.max_episode_steps):
            teacher_drive = scripted_drive(environment)
            observations.append(observation.clone())
            teacher_actions.append(teacher_drive.clone())
            observation, _reward, _done = environment.step(teacher_drive, auto_reset=False)
    dataset_observation = torch.stack(observations).reshape(-1, 200)
    dataset_action = torch.stack(teacher_actions).reshape(-1, 2)
    dataset_size = dataset_observation.shape[0]
    loss_curve: list[dict[str, float | int]] = []

    scale = float(args.heading_output_scale)
    for iteration in range(1, args.iterations + 1):
        index = torch.randint(0, dataset_size, (args.batch_size,), device=device)
        observation_batch = dataset_observation[index]
        teacher_drive = dataset_action[index]
        heading_target = teacher_drive[:, 0].clamp(
            -args.heading_output_scale, args.heading_output_scale
        ).to(torch.float32) / scale
        speed_target = teacher_drive[:, 1].to(torch.float32) / 128.0 - 1.0

        output = actor(observation_batch)
        heading_loss = F.mse_loss(torch.tanh(output[:, 0]), heading_target)
        speed_loss = F.mse_loss(torch.tanh(output[:, 1]), speed_target)
        loss = heading_loss + speed_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            quantized, _latent, _log_probability, _entropy = drive_policy_action(
                actor,
                torch.zeros(2, device=device),
                observation_batch,
                deterministic=True,
                heading_output_scale=args.heading_output_scale,
            )
            target_quantized = torch.stack(
                (
                    teacher_drive[:, 0].clamp(
                        -args.heading_output_scale, args.heading_output_scale
                    ),
                    teacher_drive[:, 1],
                ),
                dim=-1,
            )
            exact_action_fraction = (quantized == target_quantized).all(dim=-1).to(
                torch.float32
            ).mean()
            loss_curve.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.item()),
                    "heading_loss": float(heading_loss.item()),
                    "speed_loss": float(speed_loss.item()),
                    "exact_quantized_action_fraction": float(exact_action_fraction.item()),
                }
            )

    actor.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "checkpoint.pt"
    torch.save(
        {
            "contract": "warzone-tactical-v1",
            "stage": "stage2_behaviour_cloning_probe",
            "actor": actor.state_dict(),
            "drive_log_std": torch.zeros(2),
            "config": {"heading_output_scale": args.heading_output_scale},
        },
        checkpoint_path,
    )
    fixed_evaluation = evaluate_checkpoint(
        checkpoint_path,
        stats,
        device,
        deterministic=True,
        episodes=1000,
    )
    trajectory = trajectory_for_seed(
        actor,
        stats,
        device,
        scenario_seed=894,
        heading_output_scale=args.heading_output_scale,
    )
    report = {
        "kind": "stage2_behaviour_cloning_probe",
        "training": {
            "iterations": args.iterations,
            "arenas": args.arenas,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "dataset_states": dataset_size,
            "heading_output_scale": args.heading_output_scale,
            "loss_curve": loss_curve,
        },
        "deterministic_fixed_evaluation": {
            "episodes": 1000,
            "mean_time_in_band_fraction": fixed_evaluation,
        },
        "trajectory": {
            "scenario_seed": 894,
            "rows": trajectory,
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
