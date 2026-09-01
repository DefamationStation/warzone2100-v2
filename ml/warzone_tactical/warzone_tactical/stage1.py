"""Stage 1 target-selection and fire PPO sanity training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical

from .contracts import q15_to_float
from .environment import BatchedEpisodeEnvironment
from .model import TacticalActor
from .simulator import Q15_MAX, TILE_UNITS, ResolvedStats, TacticalSimulator
from .wzml import export_artifacts


@dataclass(frozen=True, slots=True)
class Stage1Config:
    arenas: int = 2048
    rollout_steps: int = 32
    iterations: int = 24
    max_episode_steps: int = 300
    learning_rate: float = 3.0e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5
    stagger_initial_episode_phase: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    clears: int
    unfinished: int
    clear_rate: float
    median_clear_ticks: float | None
    mean_clear_ticks: float | None
    p90_clear_ticks: float | None
    projectile_drops: int


class Stage1Critic(nn.Module):
    """Use the controlled droid observation as the Stage 1 value input."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(200, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.layers(observation).squeeze(-1)


def masked_target_logits(output: Tensor, target_mask: Tensor) -> Tensor:
    """Return the none logit and eight masked enemy logits."""

    if output.shape[-1] != 12 or target_mask.shape != output.shape[:-1] + (8,):
        raise ValueError("Stage 1 policy tensors have an invalid shape")
    return torch.cat(
        (output[..., 2:3], output[..., 3:11].masked_fill(~target_mask, -torch.inf)),
        dim=-1,
    )


def policy_action(
    actor: TacticalActor,
    observation_q15: Tensor,
    target_mask: Tensor,
    *,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select the two active Stage 1 heads and return their log probability."""

    output = actor(observation_q15)
    target_distribution = Categorical(logits=masked_target_logits(output, target_mask))
    fire_distribution = Bernoulli(logits=output[..., 11])
    if deterministic:
        target_category = target_distribution.logits.argmax(dim=-1)
        fire = (output[..., 11] > 0).to(torch.float32)
    else:
        target_category = target_distribution.sample()
        fire = fire_distribution.sample()
    has_target = target_category != 0
    fire = fire * has_target
    log_probability = target_distribution.log_prob(target_category)
    log_probability += fire_distribution.log_prob(fire) * has_target
    entropy = target_distribution.entropy() + fire_distribution.entropy() * has_target
    action = torch.stack((target_category.to(torch.int64) - 1, fire.to(torch.int64)), dim=-1)
    return action, log_probability, entropy


def evaluate_policy_action(
    actor: TacticalActor,
    observation_q15: Tensor,
    target_mask: Tensor,
    action: Tensor,
) -> tuple[Tensor, Tensor]:
    """Recalculate a recorded Stage 1 action for PPO."""

    output = actor(observation_q15)
    target_distribution = Categorical(logits=masked_target_logits(output, target_mask))
    fire_distribution = Bernoulli(logits=output[..., 11])
    category = action[..., 0] + 1
    fire = action[..., 1].to(torch.float32)
    has_target = category != 0
    log_probability = target_distribution.log_prob(category)
    log_probability += fire_distribution.log_prob(fire) * has_target
    entropy = target_distribution.entropy() + fire_distribution.entropy() * has_target
    return log_probability, entropy


class Stage1Environment(BatchedEpisodeEnvironment):
    """Create one stationary shooter and two or three stationary targets."""

    def __init__(
        self,
        arenas: int,
        stats: ResolvedStats,
        device: torch.device,
        seed: int,
        max_episode_steps: int = 100,
        scenario_seeds: Tensor | None = None,
        stagger_initial_episode_phase: bool = False,
    ) -> None:
        super().__init__(
            arenas,
            max_episode_steps,
            device,
            seed,
            stagger_initial_episode_phase=stagger_initial_episode_phase,
        )
        self.simulator = TacticalSimulator(arenas, stats, device=device, max_units=4, seed=seed)
        self.stats = stats
        self.episode_return = torch.zeros(arenas, dtype=torch.float32, device=device)
        self.initial_enemy_health = torch.ones(arenas, dtype=torch.float32, device=device)
        self.random_state = torch.zeros(arenas, dtype=torch.int64, device=device)
        self.next_scenario_seed = (
            100_000_000
            + seed * 10_000_000
            + torch.arange(arenas, dtype=torch.int64, device=device)
        )
        self.completed_clears = 0
        self.completed_unfinished = 0
        self.clear_ticks: list[int] = []
        self.reset(
            torch.ones(arenas, dtype=torch.bool, device=device),
            scenario_seeds=scenario_seeds,
            initial=True,
        )

    def _draw(self, arena: Tensor, columns: int, low: int, high: int) -> Tensor:
        """Draw independent per-arena integers from the scenario seed."""

        if high <= low:
            raise ValueError("the random integer range is empty")
        state = self.random_state[arena]
        values = torch.empty((arena.numel(), columns), dtype=torch.int64, device=self.device)
        for column in range(columns):
            state = torch.bitwise_and(state * 1_103_515_245 + 12_345, 0x7FFFFFFF)
            values[:, column] = low + torch.remainder(state, high - low)
        self.random_state[arena] = state
        return values

    def reset(
        self,
        mask: Tensor,
        scenario_seeds: Tensor | None = None,
        *,
        initial: bool = False,
    ) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if not mask.any():
            return
        self.simulator.reset(mask)
        arena = mask.nonzero(as_tuple=False).squeeze(-1)
        count = arena.numel()
        simulator = self.simulator
        if scenario_seeds is None:
            seeds = self.next_scenario_seed[arena]
            self.next_scenario_seed[arena] += self.arenas
        else:
            seeds = scenario_seeds.to(device=self.device, dtype=torch.int64)
            if seeds.shape != (count,):
                raise ValueError("scenario seeds must have one value for each reset arena")
        self.random_state[arena] = torch.bitwise_and(seeds, 0x7FFFFFFF)
        simulator.game_time[arena] = self._draw(arena, 1, 0, 1000)

        opponent_count = self._draw(arena, 1, 2, 4).squeeze(-1)
        active_enemy = torch.arange(3, device=self.device).view(1, 3) < opponent_count.view(-1, 1)
        simulator.alive[arena] = False
        simulator.alive[arena, 0] = True
        simulator.alive[arena, 1:4] = active_enemy
        simulator.team[arena] = torch.tensor([0, 1, 1, 1], device=self.device)
        simulator.body[arena] = 0
        simulator.body[arena, 0] = self.stats.health
        # Require one, two, and four successful hits. Randomize which physical
        # target receives each value. This prevents the one-hit saturation of
        # the first Stage 1 sanity task.
        health_key = self._draw(arena, 3, 0, 0x7FFFFFFF)
        health_order = torch.argsort(health_key, dim=-1)
        required_hits = torch.zeros((count, 3), dtype=torch.int64, device=self.device)
        required_hits.scatter_(
            1,
            health_order,
            torch.tensor([1, 2, 4], dtype=torch.int64, device=self.device)
            .view(1, 3)
            .expand(count, -1),
        )
        enemy_health = required_hits * self.stats.hit_damage
        simulator.body[arena, 1:4] = torch.where(active_enemy, enemy_health, 0)
        simulator.previous_body[arena] = simulator.body[arena]

        center = 32 * TILE_UNITS
        simulator.position_x[arena, 0] = center
        simulator.position_y[arena, 0] = center
        simulator.position_z[arena] = TILE_UNITS
        phase = self._draw(arena, 1, 0, 65536)
        jitter = self._draw(arena, 3, -2048, 2049)
        bearing = phase + torch.arange(3, device=self.device).view(1, 3) * (65536 // 3) + jitter
        near_low = max(3 * TILE_UNITS, self.stats.min_range + 1)
        near_high = max(near_low + 1, min(self.stats.short_range, self.stats.long_range) + 1)
        far_low = min(max(near_high, self.stats.short_range + 1), self.stats.long_range)
        far_mid = max(far_low + 1, (far_low + self.stats.long_range + 1) // 2)
        far_mid = min(far_mid, self.stats.long_range)
        distance = torch.cat(
            (
                self._draw(arena, 1, near_low, near_high),
                self._draw(arena, 1, far_low, far_mid + 1),
                self._draw(arena, 1, far_mid, self.stats.long_range + 1),
            ),
            dim=1,
        )
        simulator.position_x[arena, 1:4] = center + simulator.trig.sin_r(bearing, distance)
        simulator.position_y[arena, 1:4] = center + simulator.trig.cos_r(bearing, distance)
        simulator.direction[arena] = self._draw(arena, 4, 0, 65536)
        simulator.move_direction[arena] = simulator.direction[arena]
        simulator.speed[arena] = 0
        simulator.turret_direction[arena] = 0
        simulator.turret_pitch[arena] = 0
        simulator.turret_aligned[arena] = False
        simulator.last_fired[arena] = simulator.game_time[arena] - self.stats.fire_pause
        simulator.projectile_impact_time[arena] = -1
        simulator.projectile_target[arena] = -1
        simulator.projectile_damage[arena] = 0
        simulator.projectile_drop_count[arena] = 0
        simulator.previous_action[arena] = 0
        simulator.previous_action[arena, :, 2] = -1

        random_key = self._draw(arena, 3, 0, 0x7FFFFFFF)
        random_key.masked_fill_(~active_enemy, 0x7FFFFFFF)
        slot_indices = torch.argsort(random_key, dim=-1) + 1
        simulator.unit_id[arena, 0] = 1
        simulator.unit_id[arena, 1:4] = 0
        ascending_ids = torch.arange(2, 5, device=self.device).view(1, 3).expand(count, -1)
        simulator.unit_id[arena.view(-1, 1), slot_indices] = ascending_ids
        simulator.enemy_slots[arena] = 0
        simulator.enemy_slot_assigned[arena] = False
        simulator.enemy_slots[arena, 0, :3] = slot_indices
        simulator.enemy_slot_assigned[arena, 0, :3] = (
            torch.arange(3, device=self.device).view(1, 3) < opponent_count.view(-1, 1)
        )
        simulator.ally_slots[arena] = 0
        simulator.ally_slot_assigned[arena] = False

        self.reset_episode_clock(mask, initial=initial)
        self.episode_return[arena] = 0
        self.initial_enemy_health[arena] = simulator.body[arena, 1:4].sum(dim=1).to(torch.float32)

    def observation(self) -> tuple[Tensor, Tensor]:
        observation, target_mask = self.simulator.build_observation()
        return observation[:, 0], target_mask[:, 0]

    def step(
        self,
        policy_action_value: Tensor,
        *,
        auto_reset: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        full_action = torch.zeros(
            (self.arenas, 4, 4), dtype=torch.int64, device=self.device
        )
        full_action[..., 2] = -1
        full_action[:, 0, 2:] = policy_action_value
        enemy_health_before = self.simulator.body[:, 1:4].sum(dim=1)
        all_arenas = torch.arange(self.arenas, device=self.device)
        forced_roll = self._draw(all_arenas, 4, 0, 100)
        observation, target_mask = self.simulator.step(full_action, forced_roll=forced_roll)
        enemy_health_after = self.simulator.body[:, 1:4].sum(dim=1)
        damage = (enemy_health_before - enemy_health_after).to(torch.float32)
        reward = damage / self.initial_enemy_health - 0.001
        self.episode_step += 1
        clear = ~self.simulator.alive[:, 1:4].any(dim=1)
        timeout = (self.episode_step >= self.max_episode_steps) & ~clear
        done = clear | timeout
        reward += clear.to(torch.float32)
        self.episode_return += reward
        clear_tick = self.episode_step * 100
        if done.any():
            self.completed_clears += int(clear.sum().item())
            self.completed_unfinished += int(timeout.sum().item())
            self.clear_ticks.extend(int(value) for value in clear_tick[clear].tolist())
        if auto_reset:
            self.reset(done)
            if done.any():
                observation, target_mask = self.simulator.build_observation()
        return observation[:, 0], target_mask[:, 0], reward, done, clear


def compute_gae(
    reward: Tensor,
    done: Tensor,
    value: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute generalized advantages for independently reset arenas."""

    advantage = torch.zeros_like(reward)
    next_advantage = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(reward.shape[0] - 1, -1, -1):
        continuing = ~done[step]
        delta = reward[step] + gamma * next_value * continuing - value[step]
        next_advantage = delta + gamma * gae_lambda * continuing * next_advantage
        advantage[step] = next_advantage
        next_value = value[step]
    return advantage, advantage + value


def _ppo_update(
    actor: TacticalActor,
    critic: Stage1Critic,
    optimizer: torch.optim.Optimizer,
    buffers: dict[str, Tensor],
    config: Stage1Config,
) -> dict[str, float]:
    sample_count = buffers["reward"].numel()
    if sample_count % config.minibatches:
        raise ValueError("the rollout size must divide into the PPO minibatches")
    observation = buffers["observation"].reshape(sample_count, 200)
    target_mask = buffers["target_mask"].reshape(sample_count, 8)
    action = buffers["action"].reshape(sample_count, 2)
    old_log_probability = buffers["log_probability"].reshape(sample_count)
    advantage = buffers["advantage"].reshape(sample_count)
    returns = buffers["returns"].reshape(sample_count)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1.0e-8)
    batch_size = sample_count // config.minibatches
    totals = torch.zeros(6, device=observation.device)
    updates = 0
    for _epoch in range(config.epochs):
        permutation = torch.randperm(sample_count, device=observation.device)
        for start in range(0, sample_count, batch_size):
            index = permutation[start : start + batch_size]
            new_log_probability, entropy = evaluate_policy_action(
                actor, observation[index], target_mask[index], action[index]
            )
            new_value = critic(q15_to_float(observation[index]))
            ratio = torch.exp(new_log_probability - old_log_probability[index])
            unclipped = ratio * advantage[index]
            clipped = ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * advantage[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (new_value - returns[index]).square().mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm
            )
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip).to(torch.float32).mean()
                totals += torch.stack(
                    (policy_loss, value_loss, entropy_mean, approximate_kl, clip_fraction, grad_norm)
                )
            updates += 1
    totals /= updates
    names = ("policy_loss", "value_loss", "entropy", "approximate_kl", "clip_fraction", "grad_norm")
    return {name: float(value.item()) for name, value in zip(names, totals)}


def train_stage1_seed(
    seed: int,
    stats: ResolvedStats,
    config: Stage1Config,
    device: torch.device,
    output: Path,
) -> TacticalActor:
    """Train one Stage 1 sanity policy and write iteration telemetry."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actor = TacticalActor().to(device)
    critic = Stage1Critic().to(device)
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=config.learning_rate,
        eps=1.0e-5,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    environment = Stage1Environment(
        config.arenas,
        stats,
        device,
        seed,
        max_episode_steps=config.max_episode_steps,
        stagger_initial_episode_phase=config.stagger_initial_episode_phase,
    )
    observation, target_mask = environment.observation()
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / "training.jsonl"
    global_steps = 0
    started = time.perf_counter()
    with telemetry_path.open("w", encoding="utf-8") as telemetry:
        for iteration in range(1, config.iterations + 1):
            buffers = {
                "observation": torch.empty(
                    (config.rollout_steps, config.arenas, 200), dtype=torch.int16, device=device
                ),
                "target_mask": torch.empty(
                    (config.rollout_steps, config.arenas, 8), dtype=torch.bool, device=device
                ),
                "action": torch.empty(
                    (config.rollout_steps, config.arenas, 2), dtype=torch.int64, device=device
                ),
                "log_probability": torch.empty(
                    (config.rollout_steps, config.arenas), device=device
                ),
                "reward": torch.empty((config.rollout_steps, config.arenas), device=device),
                "done": torch.empty(
                    (config.rollout_steps, config.arenas), dtype=torch.bool, device=device
                ),
                "value": torch.empty((config.rollout_steps, config.arenas), device=device),
            }
            for step in range(config.rollout_steps):
                with torch.no_grad():
                    action, log_probability, _entropy = policy_action(
                        actor, observation, target_mask, deterministic=False
                    )
                    value = critic(q15_to_float(observation))
                buffers["observation"][step] = observation
                buffers["target_mask"][step] = target_mask
                buffers["action"][step] = action
                buffers["log_probability"][step] = log_probability
                buffers["value"][step] = value
                observation, target_mask, reward, done, _clear = environment.step(action)
                buffers["reward"][step] = reward
                buffers["done"][step] = done
            with torch.no_grad():
                last_value = critic(q15_to_float(observation))
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
            metrics = _ppo_update(actor, critic, optimizer, buffers, config)
            global_steps += config.rollout_steps * config.arenas
            row = {
                "iteration": iteration,
                "environment_steps": global_steps,
                "completed_clears": environment.completed_clears,
                "completed_unfinished": environment.completed_unfinished,
                "clear_rate": environment.completed_clears
                / max(environment.completed_clears + environment.completed_unfinished, 1),
                "mean_reward": float(buffers["reward"].mean().item()),
                "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            }
            telemetry.write(json.dumps(row, separators=(",", ":")) + "\n")
            telemetry.flush()
            print(json.dumps({"seed": seed, **row}, separators=(",", ":")), flush=True)
    torch.save(
        {
            "contract": "warzone-tactical-v1",
            "stage": 1,
            "training_seed": seed,
            "config": asdict(config),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "environment_steps": global_steps,
        },
        output / "checkpoint.pt",
    )
    return actor


@torch.no_grad()
def evaluate_stage1(
    actor: TacticalActor | None,
    stats: ResolvedStats,
    episodes: int,
    seed: int,
    device: torch.device,
    max_episode_steps: int,
) -> EvaluationResult:
    """Run one fixed episode per arena for a learned or scripted policy."""

    scenario_seeds = torch.arange(seed, seed + episodes, dtype=torch.int64, device=device)
    environment = Stage1Environment(
        episodes,
        stats,
        device,
        seed,
        max_episode_steps,
        scenario_seeds=scenario_seeds,
    )
    observation, target_mask = environment.observation()
    done = torch.zeros(episodes, dtype=torch.bool, device=device)
    clear = torch.zeros_like(done)
    clear_step = torch.zeros(episodes, dtype=torch.int64, device=device)
    for step in range(1, max_episode_steps + 1):
        if actor is None:
            enemy_indices = environment.simulator.enemy_slots[:, 0]
            target_x = torch.gather(environment.simulator.position_x, 1, enemy_indices)
            target_y = torch.gather(environment.simulator.position_y, 1, enemy_indices)
            dx = target_x - environment.simulator.position_x[:, :1]
            dy = target_y - environment.simulator.position_y[:, :1]
            distance_squared = dx * dx + dy * dy
            distance_squared.masked_fill_(~target_mask, torch.iinfo(torch.int64).max)
            slot = distance_squared.argmin(dim=-1)
            slot = torch.where(target_mask.any(dim=-1), slot, -1)
            action = torch.stack((slot, (slot >= 0).to(torch.int64)), dim=-1)
        else:
            action, _log_probability, _entropy = policy_action(
                actor, observation, target_mask, deterministic=True
            )
        action[done, 0] = -1
        action[done, 1] = 0
        observation, target_mask, _reward, step_done, step_clear = environment.step(
            action, auto_reset=False
        )
        new_clear = step_clear & ~done
        clear_step[new_clear] = step
        clear |= new_clear
        done |= step_done
        if done.all():
            break
    clear_ticks = (clear_step[clear] * 100).tolist()
    sorted_clear_ticks = sorted(clear_ticks)
    p90_index = math.ceil(len(sorted_clear_ticks) * 0.90) - 1
    return EvaluationResult(
        episodes=episodes,
        clears=int(clear.sum().item()),
        unfinished=int((~clear).sum().item()),
        clear_rate=float(clear.to(torch.float32).mean().item()),
        median_clear_ticks=float(statistics.median(clear_ticks)) if clear_ticks else None,
        mean_clear_ticks=float(statistics.fmean(clear_ticks)) if clear_ticks else None,
        p90_clear_ticks=(
            float(sorted_clear_ticks[p90_index]) if sorted_clear_ticks else None
        ),
        projectile_drops=int(environment.simulator.projectile_drop_count.sum().item()),
    )


def clear_time_distribution_passes(
    evaluation: EvaluationResult, baseline: EvaluationResult
) -> bool:
    """Require the mean and slow tail to stay within 10 percent of baseline."""

    return (
        evaluation.mean_clear_ticks is not None
        and baseline.mean_clear_ticks is not None
        and evaluation.p90_clear_ticks is not None
        and baseline.p90_clear_ticks is not None
        and evaluation.mean_clear_ticks <= baseline.mean_clear_ticks * 1.10
        and evaluation.p90_clear_ticks <= baseline.p90_clear_ticks * 1.10
    )


def run_three_seed_sanity(
    stats_path: Path,
    output: Path,
    config: Stage1Config,
    episodes: int = 1000,
    training_seeds: tuple[int, int, int] = (0, 1, 2),
    promotion: bool = False,
) -> dict[str, object]:
    """Train seeds 0, 1, and 2, then apply the Stage 1 sanity gate."""

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 sanity training needs CUDA")
    device = torch.device("cuda")
    stats = ResolvedStats.load(stats_path)
    output.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_stage1(None, stats, episodes, 0, device, config.max_episode_steps)
    seed_rows: list[dict[str, object]] = []
    for seed in training_seeds:
        seed_output = output / f"seed-{seed}"
        actor = train_stage1_seed(seed, stats, config, device, seed_output)
        evaluation = evaluate_stage1(actor, stats, episodes, 0, device, config.max_episode_steps)
        export_artifacts(actor.cpu().eval(), seed_output / "artifacts", stats_path)
        clear_time_gate = clear_time_distribution_passes(evaluation, baseline)
        seed_rows.append(
            {
                "seed": seed,
                "evaluation": asdict(evaluation),
                "clear_time_gate_passed": clear_time_gate,
                "projectile_gate_passed": evaluation.projectile_drops == 0,
            }
        )
    report = _build_sanity_report(
        stats_path,
        config,
        episodes,
        baseline,
        seed_rows,
        training_seeds=training_seeds,
        promotion=promotion,
    )
    (output / "stage1-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _build_sanity_report(
    stats_path: Path,
    config: Stage1Config,
    episodes: int,
    baseline: EvaluationResult,
    seed_rows: list[dict[str, object]],
    training_seeds: tuple[int, int, int] = (0, 1, 2),
    promotion: bool = False,
) -> dict[str, object]:
    clear_rates = [
        float(evaluation["clear_rate"])
        for row in seed_rows
        if isinstance((evaluation := row.get("evaluation")), dict)
    ]
    if len(clear_rates) != 3:
        raise ValueError("the sanity report needs three seed evaluations")
    mean_clear_rate = statistics.fmean(clear_rates)
    lowest_clear_rate = min(clear_rates)
    sanity_passed = (
        mean_clear_rate >= 0.95
        and lowest_clear_rate >= 0.90
        and all(bool(row["clear_time_gate_passed"]) for row in seed_rows)
        and all(bool(row["projectile_gate_passed"]) for row in seed_rows)
    )
    report: dict[str, object] = {
        "contract": "warzone-tactical-v1",
        "stage": 1,
        "kind": "three_seed_promotion" if promotion else "three_seed_learning_sanity",
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "config": asdict(config),
        "sanity_episodes_per_seed": episodes,
        "evaluation_episodes_per_seed": episodes,
        "scenario_seed_first": 0,
        "scenario_seed_last_inclusive": episodes - 1,
        "per_arena_random_stream": True,
        "target_required_hits": [1, 2, 4],
        "target_range_layout": "one short-range band and two distinct long-range bands",
        "scripted_nearest_v1": asdict(baseline),
        "seeds": seed_rows,
        "mean_clear_rate": mean_clear_rate,
        "lowest_seed_clear_rate": lowest_clear_rate,
        "mean_gate_passed": mean_clear_rate >= 0.95,
        "five_point_floor_passed": lowest_clear_rate >= 0.90,
        "sanity_gate_passed": sanity_passed,
        "promotion_gate_passed": sanity_passed if promotion else None,
        "contract_v1_promotion_evaluated": promotion,
        "promotion_training_seeds": [4, 17, 31],
        "training_seeds": list(training_seeds),
    }
    return report


def evaluate_existing_sanity(
    stats_path: Path,
    output: Path,
    episodes: int = 1000,
) -> dict[str, object]:
    """Re-evaluate saved seed-0/1/2 actors on exact scenario seeds."""

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 sanity evaluation needs CUDA")
    device = torch.device("cuda")
    stats = ResolvedStats.load(stats_path)
    first_checkpoint = torch.load(
        output / "seed-0" / "checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = Stage1Config(**first_checkpoint["config"])
    baseline = evaluate_stage1(None, stats, episodes, 0, device, config.max_episode_steps)
    seed_rows: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        checkpoint = torch.load(
            output / f"seed-{seed}" / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        if int(checkpoint["training_seed"]) != seed:
            raise ValueError("the Stage 1 checkpoint seed does not match its directory")
        actor = TacticalActor().to(device)
        actor.load_state_dict(checkpoint["actor"], strict=True)
        actor.eval()
        evaluation = evaluate_stage1(actor, stats, episodes, 0, device, config.max_episode_steps)
        clear_time_gate = clear_time_distribution_passes(evaluation, baseline)
        seed_rows.append(
            {
                "seed": seed,
                "evaluation": asdict(evaluation),
                "clear_time_gate_passed": clear_time_gate,
                "projectile_gate_passed": evaluation.projectile_drops == 0,
            }
        )
    report = _build_sanity_report(stats_path, config, episodes, baseline, seed_rows)
    (output / "stage1-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arenas", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--training-seeds", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--promotion", action="store_true")
    parser.add_argument("--evaluate-existing", action="store_true")
    args = parser.parse_args()
    config = Stage1Config(
        arenas=args.arenas,
        iterations=args.iterations,
        rollout_steps=args.rollout_steps,
    )
    if args.evaluate_existing:
        report = evaluate_existing_sanity(args.resolved_stats, args.output, args.episodes)
    else:
        report = run_three_seed_sanity(
            args.resolved_stats,
            args.output,
            config,
            args.episodes,
            training_seeds=tuple(args.training_seeds),
            promotion=args.promotion,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
