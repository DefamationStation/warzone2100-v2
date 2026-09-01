"""CUDA throughput report for the fixed simulator and actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .model import TacticalActor
from .simulator import ResolvedStats, TacticalSimulator


@dataclass(frozen=True, slots=True)
class Measurement:
    arenas: int
    environment_updates_per_second: float
    wall_steps_per_second: float
    wall_ms_per_step: float
    unit_decisions_per_second: float
    hours_per_billion_decisions: float
    gpu_memory_bytes: int
    simulator_ms: float
    policy_forward_ms: float
    ppo_update_ms: float
    projectile_drops: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _actor_backward_smoke(
    actor: TacticalActor, optimizer: torch.optim.Optimizer, sample: torch.Tensor
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = actor(sample).square().mean()
    loss.backward()
    optimizer.step()


def benchmark_size(
    arenas: int,
    stats: ResolvedStats,
    device: torch.device,
    steps: int = 200,
    *,
    compile_simulator: bool = True,
) -> Measurement:
    simulator = TacticalSimulator(arenas, stats, device=device, max_units=4, seed=7)
    actor = TacticalActor().to(device).eval()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    action = torch.zeros((arenas, 4, 4), dtype=torch.int64, device=device)
    action[..., 1] = 256
    action[..., 2] = 0
    action[..., 3] = 1
    forced_roll = torch.empty((arenas, 4), dtype=torch.int64, device=device)
    simulator_step = (
        torch.compile(
            simulator.step,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=False,
        )
        if compile_simulator
        else simulator.step
    )

    for _ in range(10):
        forced_roll.random_(0, 100, generator=simulator.generator)
        observation, mask = simulator_step(action, forced_roll)
        action = actor.deterministic_action(observation, mask)
    _synchronize(device)

    simulator_seconds = 0.0
    forward_seconds = 0.0
    start_total = time.perf_counter()
    for _ in range(steps):
        start = time.perf_counter()
        forced_roll.random_(0, 100, generator=simulator.generator)
        observation, mask = simulator_step(action, forced_roll)
        _synchronize(device)
        simulator_seconds += time.perf_counter() - start
        start = time.perf_counter()
        action = actor.deterministic_action(observation, mask)
        _synchronize(device)
        forward_seconds += time.perf_counter() - start
    total_seconds = time.perf_counter() - start_total

    sample = observation[: min(32, arenas)].to(torch.float32) / 32768.0
    # Discard the first backward pass. It includes one-time CUDA setup costs.
    _actor_backward_smoke(actor, optimizer, sample)
    _synchronize(device)
    start = time.perf_counter()
    _actor_backward_smoke(actor, optimizer, sample)
    _synchronize(device)
    ppo_seconds = time.perf_counter() - start

    updates_per_second = steps * arenas / total_seconds
    decisions_per_second = updates_per_second * 4
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return Measurement(
        arenas=arenas,
        environment_updates_per_second=updates_per_second,
        wall_steps_per_second=steps / total_seconds,
        wall_ms_per_step=total_seconds * 1000 / steps,
        unit_decisions_per_second=decisions_per_second,
        hours_per_billion_decisions=1_000_000_000 / decisions_per_second / 3600,
        gpu_memory_bytes=memory,
        simulator_ms=simulator_seconds * 1000 / steps,
        policy_forward_ms=forward_seconds * 1000 / steps,
        ppo_update_ms=ppo_seconds * 1000,
        projectile_drops=int(simulator.projectile_drop_count.sum().item()),
    )


def benchmark_all(
    stats_path: Path,
    output: Path,
    steps: int = 200,
    *,
    compile_simulator: bool = True,
    arena_sizes: tuple[int, ...] = (512, 1024, 2048, 4096, 8192),
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("the throughput gate needs CUDA")
    device = torch.device("cuda")
    stats = ResolvedStats.load(stats_path)
    measurements = [
        benchmark_size(size, stats, device, steps, compile_simulator=compile_simulator)
        for size in arena_sizes
    ]
    total_memory = torch.cuda.get_device_properties(device).total_memory
    eligible = [item for item in measurements if item.gpu_memory_bytes < total_memory * 0.85]
    selected = max(eligible, key=lambda item: item.unit_decisions_per_second)
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "resolved_stats_path": stats_path.as_posix(),
        "resolved_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "memory_limit_bytes": int(total_memory * 0.85),
        "gate_decisions_per_second": 300_000,
        "selected_arenas": selected.arenas,
        "gate_passed": selected.unit_decisions_per_second >= 300_000,
        "ppo_update_kind": "single_actor_backward_smoke",
        "compiled_simulator": compile_simulator,
        "measurements": [asdict(item) for item in measurements],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--arenas", type=int, nargs="+", default=(512, 1024, 2048, 4096, 8192))
    parser.add_argument("--eager-simulator", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_all(
                args.resolved_stats,
                args.output,
                args.steps,
                compile_simulator=not args.eager_simulator,
                arena_sizes=tuple(args.arenas),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
