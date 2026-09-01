"""Profile one warmed Stage 2 iteration without changing training settings."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from .contracts import q15_to_float
from .model import TacticalActor
from .simulator import ResolvedStats
from .stage1 import Stage1Critic, compute_gae
from .stage2 import (
    RunningReturnNormalizer,
    Stage2Config,
    Stage2Environment,
    _ppo_update,
    _rehearsal_data,
    drive_policy_action,
)


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _iteration(
    actor: TacticalActor,
    critic: Stage1Critic,
    log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    environment: Stage2Environment,
    observation: Tensor,
    rehearsal_observation: Tensor,
    rehearsal_output: Tensor,
    return_normalizer: RunningReturnNormalizer,
    config: Stage2Config,
    *,
    profile: bool,
) -> tuple[Tensor, dict[str, object]]:
    device = observation.device
    timing: dict[str, object] = {}
    _synchronize(device)
    iteration_started = time.perf_counter()
    section_started = iteration_started
    buffers = {
        "observation": torch.empty(
            (config.rollout_steps, config.arenas, 200), dtype=torch.int16, device=device
        ),
        "latent": torch.empty((config.rollout_steps, config.arenas, 2), device=device),
        "log_probability": torch.empty((config.rollout_steps, config.arenas), device=device),
        "reward": torch.empty((config.rollout_steps, config.arenas), device=device),
        "done": torch.empty(
            (config.rollout_steps, config.arenas), dtype=torch.bool, device=device
        ),
        "value": torch.empty((config.rollout_steps, config.arenas), device=device),
    }
    for step in range(config.rollout_steps):
        with torch.no_grad():
            drive, latent, log_probability, _entropy = drive_policy_action(
                actor, log_std, observation, deterministic=False
            )
            value = return_normalizer.denormalize(critic(q15_to_float(observation)))
        buffers["observation"][step] = observation
        buffers["latent"][step] = latent
        buffers["log_probability"][step] = log_probability
        buffers["value"][step] = value
        observation, reward, done = environment.step(
            drive,
            range_objective_coefficient=config.range_objective_coefficient,
            range_progress_coefficient=config.range_progress_coefficient,
            heading_alignment_coefficient=config.heading_alignment_coefficient,
            speed_settling_coefficient=config.speed_settling_coefficient,
        )
        buffers["reward"][step] = reward
        buffers["done"][step] = done
    _synchronize(device)
    timing["rollout_ms"] = (time.perf_counter() - section_started) * 1000

    section_started = time.perf_counter()
    with torch.no_grad():
        last_value = return_normalizer.denormalize(critic(q15_to_float(observation)))
    advantage, returns = compute_gae(
        buffers["reward"],
        buffers["done"],
        buffers["value"],
        last_value,
        config.gamma,
        config.gae_lambda,
    )
    buffers["advantage"] = advantage
    buffers["returns"] = returns
    return_normalizer.update(returns)
    _synchronize(device)
    timing["gae_ms"] = (time.perf_counter() - section_started) * 1000

    update_profile: dict[str, object] | None = {} if profile else None
    section_started = time.perf_counter()
    metrics = _ppo_update(
        actor,
        critic,
        log_std,
        optimizer,
        buffers,
        rehearsal_observation,
        rehearsal_output,
        return_normalizer,
        config,
        profile=update_profile,
    )
    _synchronize(device)
    timing["ppo_update_ms"] = (time.perf_counter() - section_started) * 1000
    if update_profile is not None:
        timing["ppo_detail"] = update_profile

    section_started = time.perf_counter()
    completed_episodes, completed_time_sum, completed_violation_sum = environment.completed_metrics()
    row = {
        "completed_episodes": completed_episodes,
        "mean_completed_time_in_band_fraction": completed_time_sum / max(completed_episodes, 1),
        "mean_completed_normalized_band_violation": completed_violation_sum
        / max(completed_episodes, 1),
        "mean_reward": float(buffers["reward"].mean().item()),
        "drive_log_std": [float(value) for value in log_std.detach().cpu().tolist()],
        **metrics,
    }
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        stream.flush()
    _synchronize(device)
    timing["metrics_logging_ms"] = (time.perf_counter() - section_started) * 1000
    timing["total_ms"] = (time.perf_counter() - iteration_started) * 1000
    timing["section_sum_ms"] = sum(
        float(timing[name])
        for name in ("rollout_ms", "gae_ms", "ppo_update_ms", "metrics_logging_ms")
    )
    return observation, timing


def profile_iteration(stats_path: Path, stage1_checkpoint: Path, output: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 profiling needs CUDA")
    device = torch.device("cuda")
    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    stats = ResolvedStats.load(stats_path)
    config = Stage2Config(arenas=2048, rollout_steps=8, target_kl=1_000_000.0)
    checkpoint = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    actor = TacticalActor().to(device)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    rehearsal_observation, rehearsal_output = _rehearsal_data(actor, stats, device, seed)
    critic = Stage1Critic().to(device)
    return_normalizer = RunningReturnNormalizer(device)
    log_std = nn.Parameter(torch.full((2,), config.initial_log_std, device=device))
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()) + [log_std],
        lr=config.learning_rate,
        eps=1.0e-5,
        weight_decay=0.0,
        fused=True,
    )
    environment = Stage2Environment(2048, stats, device, seed, compile_simulator=True)
    observation = environment.observation()
    observation, _warmup = _iteration(
        actor,
        critic,
        log_std,
        optimizer,
        environment,
        observation,
        rehearsal_observation,
        rehearsal_output,
        return_normalizer,
        config,
        profile=False,
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as operator_profile:
        _observation, timing = _iteration(
            actor,
            critic,
            log_std,
            optimizer,
            environment,
            observation,
            rehearsal_observation,
            rehearsal_output,
            return_normalizer,
            config,
            profile=True,
        )
    timing["operator_table"] = operator_profile.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    return timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(profile_iteration(args.resolved_stats, args.stage1_checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
