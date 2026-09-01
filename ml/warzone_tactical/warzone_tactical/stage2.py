"""Stage 2 approach and range-hold PPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from .contracts import _lround, q15_to_float
from .environment import BatchedEpisodeEnvironment
from .model import TacticalActor
from .simulator import Q15_MAX, TILE_UNITS, ResolvedStats, TacticalSimulator, angle_delta, integer_floor_sqrt
from .stage1 import Stage1Critic, Stage1Environment, compute_gae, evaluate_stage1
from .wzml import export_artifacts


@dataclass(frozen=True, slots=True)
class Stage2Config:
    arenas: int = 2048
    rollout_steps: int = 8
    iterations: int = 1500
    max_episode_steps: int = 300
    learning_rate: float = 3.0e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.015
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.002
    retention_coefficient: float = 0.10
    rehearsal_fraction: float = 0.20
    range_objective_coefficient: float = 1.0
    range_progress_coefficient: float = 0.1
    heading_alignment_coefficient: float = 0.02
    range_progress_start_coefficient: float = 0.1
    heading_alignment_start_coefficient: float = 0.02
    approach_anneal_iterations: int = 0
    speed_settling_coefficient: float = 0.02
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5
    initial_log_std: float = -0.35
    final_log_std: float | None = None
    log_std_anneal_iterations: int = 0
    clip_level_violation: bool = False
    heading_output_scale: int = 2048
    compile_simulator: bool = True
    stagger_initial_episode_phase: bool = True
    training_in_band_spawn_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class Stage2Evaluation:
    episodes: int
    mean_time_in_band_fraction: float
    median_time_in_band_fraction: float
    p10_time_in_band_fraction: float
    mean_normalized_band_violation: float
    median_normalized_band_violation: float
    mean_final_distance: float


class RunningReturnNormalizer:
    """Track scalar return statistics without changing the reward contract."""

    def __init__(self, device: torch.device) -> None:
        self.mean = torch.zeros((), dtype=torch.float64, device=device)
        self.variance = torch.ones((), dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    @property
    def standard_deviation(self) -> Tensor:
        return torch.sqrt(self.variance.clamp_min(1.0e-8)).to(torch.float32)

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.mean.to(torch.float32)) / self.standard_deviation

    def denormalize(self, value: Tensor) -> Tensor:
        return value * self.standard_deviation + self.mean.to(torch.float32)

    @torch.no_grad()
    def update(self, value: Tensor) -> None:
        value64 = value.to(torch.float64)
        batch_count = torch.tensor(value.numel(), dtype=torch.float64, device=value.device)
        batch_mean = value64.mean()
        batch_variance = value64.var(unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        combined_m2 = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta.square() * self.count * batch_count / total
        )
        self.mean.copy_(new_mean)
        self.variance.copy_(combined_m2 / total)
        self.count.copy_(total)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "variance": self.variance.detach().cpu(),
            "count": self.count.detach().cpu(),
        }


def explained_variance(returns: Tensor, values: Tensor) -> Tensor:
    """Measure critic fit in the original, unnormalized return scale."""

    return_variance = returns.var(unbiased=False)
    residual_variance = (returns - values).var(unbiased=False)
    return torch.where(
        return_variance > 1.0e-8,
        1.0 - residual_variance / return_variance,
        torch.zeros_like(return_variance),
    )


def _distance(simulator: TacticalSimulator) -> Tensor:
    dx = simulator.position_x[:, 1] - simulator.position_x[:, 0]
    dy = simulator.position_y[:, 1] - simulator.position_y[:, 0]
    return integer_floor_sqrt(dx * dx + dy * dy)


def range_metrics(distance: Tensor, short_range: int, long_range: int) -> tuple[Tensor, Tensor]:
    """Use the open-closed Rocket-Pod band: short < distance <= long."""

    in_band = (distance > short_range) & (distance <= long_range)
    violation = torch.where(
        distance <= short_range,
        short_range + 1 - distance,
        torch.where(distance > long_range, distance - long_range, 0),
    )
    width = max(long_range - short_range, 1)
    return in_band, violation.to(torch.float32) / width


def stage2_reward(
    in_band: Tensor,
    violation: Tensor,
    range_progress: Tensor,
    heading_alignment: Tensor,
    speed_quality: Tensor,
    *,
    range_objective_coefficient: float = 1.0,
    range_progress_coefficient: float = 0.1,
    heading_alignment_coefficient: float = 0.02,
    speed_settling_coefficient: float = 0.02,
    clip_level_violation: bool = False,
) -> Tensor:
    """Reward range control. Holding the band is the main objective."""

    in_band_float = in_band.to(torch.float32)
    level_violation = violation.clamp(max=1.0) if clip_level_violation else violation
    return (
        range_objective_coefficient * (in_band_float - level_violation)
        + range_progress_coefficient * range_progress
        + heading_alignment_coefficient * heading_alignment * (~in_band).to(torch.float32)
        + speed_settling_coefficient * (2.0 * speed_quality - 1.0)
    )


def approach_reward_coefficients(config: Stage2Config, iteration: int) -> tuple[float, float]:
    """Linearly anneal approach shaping to the current Stage 2 coefficients."""

    if config.approach_anneal_iterations <= 1:
        return config.range_progress_coefficient, config.heading_alignment_coefficient
    if iteration >= config.approach_anneal_iterations:
        return config.range_progress_coefficient, config.heading_alignment_coefficient
    progress = min(max(iteration - 1, 0), config.approach_anneal_iterations - 1) / (
        config.approach_anneal_iterations - 1
    )
    range_progress = config.range_progress_start_coefficient + progress * (
        config.range_progress_coefficient - config.range_progress_start_coefficient
    )
    heading_alignment = config.heading_alignment_start_coefficient + progress * (
        config.heading_alignment_coefficient - config.heading_alignment_start_coefficient
    )
    return range_progress, heading_alignment


def scheduled_log_std(config: Stage2Config, iteration: int) -> float | None:
    """Return the fixed state-independent exploration schedule when enabled."""

    if config.final_log_std is None or config.log_std_anneal_iterations <= 0:
        return None
    if config.log_std_anneal_iterations == 1 or iteration >= config.log_std_anneal_iterations:
        return config.final_log_std
    progress = (iteration - 1) / (config.log_std_anneal_iterations - 1)
    return config.initial_log_std + progress * (config.final_log_std - config.initial_log_std)


def quantize_drive(latent: Tensor, heading_output_scale: int = 2048) -> Tensor:
    if latent.shape[-1] != 2:
        raise ValueError("the Stage 2 drive latent must have two values")
    heading = _lround(torch.tanh(latent[..., 0]) * heading_output_scale).to(torch.int64)
    speed = _lround((torch.tanh(latent[..., 1]) + 1.0) * 128.0).to(torch.int64)
    return torch.stack((heading.clamp(-2048, 2047), speed.clamp(0, 256)), dim=-1)


def _wrapped_normal_log_probability(mean: Tensor, standard_deviation: Tensor, value: Tensor) -> Tensor:
    offsets = torch.arange(-3, 4, device=value.device, dtype=value.dtype) * 2.0
    expanded_distribution = Normal(mean.unsqueeze(-1), standard_deviation)
    return torch.logsumexp(
        expanded_distribution.log_prob(value.unsqueeze(-1) + offsets), dim=-1
    )


def drive_policy_action(
    actor: TacticalActor,
    log_std: Tensor,
    observation_q15: Tensor,
    *,
    deterministic: bool,
    heading_output_scale: int = 2048,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output = actor(observation_q15)
    heading_mean = torch.tanh(output[..., 0])
    heading_distribution = Normal(heading_mean, log_std[0].exp())
    speed_distribution = Normal(output[..., 1], log_std[1].exp())
    if deterministic:
        heading = heading_mean
        speed_latent = output[..., 1]
    else:
        heading = torch.remainder(heading_distribution.sample() + 1.0, 2.0) - 1.0
        speed_latent = speed_distribution.sample()
    heading_q = _lround(heading * heading_output_scale).to(torch.int64).clamp(-2048, 2047)
    speed_q = _lround((torch.tanh(speed_latent) + 1.0) * 128.0).to(torch.int64).clamp(0, 256)
    latent = torch.stack((heading, speed_latent), dim=-1)
    heading_log_probability = _wrapped_normal_log_probability(
        heading_mean, log_std[0].exp(), heading
    )
    return (
        torch.stack((heading_q, speed_q), dim=-1),
        latent,
        heading_log_probability + speed_distribution.log_prob(speed_latent),
        heading_distribution.entropy() + speed_distribution.entropy(),
    )


class Stage2Environment(BatchedEpisodeEnvironment):
    """Create one controlled droid and one stationary, non-firing enemy."""

    def __init__(
        self,
        arenas: int,
        stats: ResolvedStats,
        device: torch.device,
        seed: int,
        max_episode_steps: int = 300,
        scenario_seeds: Tensor | None = None,
        compile_simulator: bool = True,
        stagger_initial_episode_phase: bool = False,
        training_in_band_spawn_fraction: float | None = None,
    ) -> None:
        if stats.short_range >= stats.long_range:
            raise ValueError("Stage 2 needs short_range below long_range")
        super().__init__(
            arenas,
            max_episode_steps,
            device,
            seed,
            stagger_initial_episode_phase=stagger_initial_episode_phase,
        )
        self.simulator = TacticalSimulator(arenas, stats, device=device, max_units=2, seed=seed)
        self._forced_roll = torch.zeros((arenas, 2), dtype=torch.int64, device=device)
        self._compiled_simulator_step = (
            torch.compile(
                self.simulator.step,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )
            if compile_simulator and device.type == "cuda"
            else None
        )
        self.stats = stats
        self.training_in_band_spawn_fraction = training_in_band_spawn_fraction
        if training_in_band_spawn_fraction is not None:
            inside_buckets = round(training_in_band_spawn_fraction * 100)
            if not math.isclose(
                training_in_band_spawn_fraction * 100,
                inside_buckets,
                abs_tol=1.0e-9,
            ):
                raise ValueError("the training in-band spawn fraction must use one-percent steps")
            outside_buckets = 100 - inside_buckets
            if inside_buckets <= 0 or outside_buckets <= 0 or outside_buckets % 2:
                raise ValueError("the training spawn mix must split the remainder equally")
            self._training_near_spawn_buckets = outside_buckets // 2
            self._training_inside_spawn_buckets = inside_buckets
        self.in_band_steps = torch.zeros(arenas, dtype=torch.float32, device=device)
        self.violation_sum = torch.zeros(arenas, dtype=torch.float32, device=device)
        self.random_state = torch.zeros(arenas, dtype=torch.int64, device=device)
        self.next_scenario_seed = 200_000_000 + seed * 10_000_000 + torch.arange(
            arenas, dtype=torch.int64, device=device
        )
        self.completed_episodes_t = torch.zeros((), dtype=torch.int64, device=device)
        self.completed_time_in_band_sum_t = torch.zeros((), dtype=torch.float64, device=device)
        self.completed_violation_sum_t = torch.zeros((), dtype=torch.float64, device=device)
        self._host_episode_step = 0
        self.reset(
            torch.ones(arenas, dtype=torch.bool, device=device),
            scenario_seeds,
            initial=True,
        )

    def _draw(self, mask: Tensor, columns: int, low: int, high: int) -> Tensor:
        if high <= low:
            raise ValueError("the random integer range is empty")
        state = self.random_state
        values = torch.empty((self.arenas, columns), dtype=torch.int64, device=self.device)
        for column in range(columns):
            state = torch.bitwise_and(state * 1_103_515_245 + 12_345, 0x7FFFFFFF)
            values[:, column] = low + torch.remainder(state, high - low)
        self.random_state.copy_(torch.where(mask, state, self.random_state))
        return values

    def completed_metrics(self) -> tuple[int, float, float]:
        """Copy the three episode counters to the host in one operation."""

        values = torch.stack(
            (
                self.completed_episodes_t.to(torch.float64),
                self.completed_time_in_band_sum_t,
                self.completed_violation_sum_t,
            )
        ).cpu().tolist()
        return int(values[0]), float(values[1]), float(values[2])

    def reset(
        self,
        mask: Tensor,
        scenario_seeds: Tensor | None = None,
        *,
        initial: bool = False,
    ) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if scenario_seeds is None:
            seeds = self.next_scenario_seed.clone()
            self.next_scenario_seed.add_(mask.to(torch.int64) * self.arenas)
        else:
            seeds = scenario_seeds.to(device=self.device, dtype=torch.int64)
            if seeds.shape != (self.arenas,):
                raise ValueError("scenario seeds must have one value for each arena")
        self.random_state.copy_(
            torch.where(mask, torch.bitwise_and(seeds, 0x7FFFFFFF), self.random_state)
        )

        simulator = self.simulator
        simulator.game_time[mask] = self._draw(mask, 1, 0, 1000)[mask]
        simulator.alive[mask] = True
        simulator.team[mask] = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        simulator.unit_id[mask] = torch.tensor([1, 2], dtype=torch.int64, device=self.device)
        simulator.body[mask] = self.stats.health
        simulator.previous_body[mask] = simulator.body[mask]

        center = 32 * TILE_UNITS
        simulator.position_x[mask, 1] = center
        simulator.position_y[mask, 1] = center
        simulator.position_z[mask] = TILE_UNITS
        if self.training_in_band_spawn_fraction is None:
            start_band = self._draw(mask, 1, 0, 3).squeeze(-1)
        else:
            start_bucket = self._draw(mask, 1, 0, 100).squeeze(-1)
            near_limit = self._training_near_spawn_buckets
            inside_limit = near_limit + self._training_inside_spawn_buckets
            start_band = torch.where(
                start_bucket < near_limit,
                0,
                torch.where(start_bucket < inside_limit, 1, 2),
            )
        near = self._draw(mask, 1, max(self.stats.min_range + 32, 160), self.stats.short_range + 1).squeeze(-1)
        inside = self._draw(mask, 1, self.stats.short_range + 1, self.stats.long_range + 1).squeeze(-1)
        far = self._draw(mask, 1, self.stats.long_range + 1, self.stats.long_range + 8 * TILE_UNITS + 1).squeeze(-1)
        distance = torch.where(start_band == 0, near, torch.where(start_band == 1, inside, far))
        bearing = self._draw(mask, 1, 0, 65536).squeeze(-1)
        simulator.position_x[mask, 0] = (center + simulator.trig.sin_r(bearing, distance))[mask]
        simulator.position_y[mask, 0] = (center + simulator.trig.cos_r(bearing, distance))[mask]
        simulator.direction[mask] = self._draw(mask, 2, 0, 65536)[mask]
        simulator.move_direction[mask] = simulator.direction[mask]
        simulator.speed[mask] = 0
        simulator.turret_direction[mask] = 0
        simulator.turret_pitch[mask] = 0
        simulator.turret_aligned[mask] = False
        simulator.last_fired[mask] = simulator.game_time[mask] - self.stats.fire_pause
        simulator.projectile_impact_time[mask] = -1
        simulator.projectile_target[mask] = -1
        simulator.projectile_damage[mask] = 0
        simulator.projectile_drop_count[mask] = 0
        simulator.previous_action[mask] = 0
        simulator.previous_action[mask, :, 2] = -1
        simulator.enemy_slots[mask] = 0
        simulator.enemy_slot_assigned[mask] = False
        simulator.enemy_slots[mask, 0, 0] = 1
        simulator.enemy_slots[mask, 1, 0] = 0
        simulator.enemy_slot_assigned[mask, :, 0] = True
        simulator.ally_slots[mask] = 0
        simulator.ally_slot_assigned[mask] = False
        self.reset_episode_clock(mask, initial=initial)
        self.in_band_steps[mask] = 0
        self.violation_sum[mask] = 0
        self._host_episode_step = 0

    def observation(self) -> Tensor:
        observation, _mask = self.simulator.build_observation()
        return observation[:, 0]

    def step(
        self,
        drive: Tensor,
        *,
        auto_reset: bool = True,
        range_objective_coefficient: float = 1.0,
        range_progress_coefficient: float = 0.1,
        heading_alignment_coefficient: float = 0.02,
        speed_settling_coefficient: float = 0.02,
        clip_level_violation: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        before_distance = _distance(self.simulator)
        _before_in_band, before_violation = range_metrics(
            before_distance, self.stats.short_range, self.stats.long_range
        )
        full_action = torch.zeros((self.arenas, 2, 4), dtype=torch.int64, device=self.device)
        full_action[..., 2] = -1
        full_action[:, 0, :2] = drive
        staggered_reset = auto_reset and self.stagger_initial_episode_phase
        reset_due = (
            auto_reset
            and not self.stagger_initial_episode_phase
            and self._host_episode_step + 1 >= self.max_episode_steps
        )
        if staggered_reset:
            if self._compiled_simulator_step is None:
                step_result = self.simulator.step(
                    full_action,
                    self._forced_roll,
                    build_observation=False,
                )
            else:
                step_result = self._compiled_simulator_step(
                    full_action,
                    self._forced_roll,
                    build_observation=False,
                )
        elif reset_due or self._compiled_simulator_step is None:
            step_result = self.simulator.step(
                full_action,
                self._forced_roll,
                build_observation=not reset_due,
            )
        else:
            step_result = self._compiled_simulator_step(full_action, self._forced_roll)
        self.episode_step += 1
        self._host_episode_step += 1
        in_band, violation = range_metrics(_distance(self.simulator), self.stats.short_range, self.stats.long_range)
        dx = self.simulator.position_x[:, 1] - self.simulator.position_x[:, 0]
        dy = self.simulator.position_y[:, 1] - self.simulator.position_y[:, 0]
        toward = self.simulator.trig.atan2(dx, dy)
        desired = torch.where(_distance(self.simulator) <= self.stats.short_range, toward + 32768, toward)
        heading_error = angle_delta(desired - self.simulator.direction[:, 0])
        heading_alignment = self.simulator.trig.cos(heading_error).to(torch.float32) / 65536.0
        speed_fraction = self.simulator.speed[:, 0].to(torch.float32) / max(
            self.stats.calculated_max_speed, 1
        )
        speed_quality = torch.where(in_band, 1.0 - speed_fraction, speed_fraction)
        self.in_band_steps += in_band.to(torch.float32)
        self.violation_sum += violation
        range_progress = before_violation - violation
        reward = stage2_reward(
            in_band,
            violation,
            range_progress,
            heading_alignment,
            speed_quality,
            range_objective_coefficient=range_objective_coefficient,
            range_progress_coefficient=range_progress_coefficient,
            heading_alignment_coefficient=heading_alignment_coefficient,
            speed_settling_coefficient=speed_settling_coefficient,
            clip_level_violation=clip_level_violation,
        )
        done = self.episode_step >= self.max_episode_steps
        done_float = done.to(torch.float32)
        steps = self.episode_step.clamp_min(1).to(torch.float32)
        self.completed_episodes_t += done.sum()
        self.completed_time_in_band_sum_t += (
            (self.in_band_steps / steps) * done_float
        ).sum().to(torch.float64)
        self.completed_violation_sum_t += (
            (self.violation_sum / steps) * done_float
        ).sum().to(torch.float64)
        if auto_reset:
            if staggered_reset:
                self.reset(done)
            elif reset_due:
                self.reset(done)
        if staggered_reset or reset_due:
            observation, _mask = self.simulator.build_observation()
        else:
            if step_result is None:
                raise RuntimeError("the simulator did not build the Stage 2 observation")
            observation, _mask = step_result
        return observation[:, 0], reward, done


def _evaluate_drive_action(
    actor: TacticalActor,
    log_std: Tensor,
    observation: Tensor,
    latent: Tensor,
) -> tuple[Tensor, Tensor]:
    output = actor(observation)
    heading_distribution = Normal(torch.tanh(output[..., 0]), log_std[0].exp())
    speed_distribution = Normal(output[..., 1], log_std[1].exp())
    heading_log_probability = _wrapped_normal_log_probability(
        torch.tanh(output[..., 0]), log_std[0].exp(), latent[..., 0]
    )
    log_probability = heading_log_probability + speed_distribution.log_prob(latent[..., 1])
    entropy = heading_distribution.entropy() + speed_distribution.entropy()
    return log_probability, entropy


def _ppo_update(
    actor: TacticalActor,
    critic: Stage1Critic,
    log_std: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    buffers: dict[str, Tensor],
    rehearsal_observation: Tensor,
    rehearsal_output: Tensor,
    return_normalizer: RunningReturnNormalizer,
    config: Stage2Config,
    profile: dict[str, object] | None = None,
) -> dict[str, float]:
    if config.rehearsal_fraction <= 0.0 or config.rehearsal_fraction >= 1.0:
        raise ValueError("the rehearsal fraction must be above zero and below one")
    sample_count = buffers["reward"].numel()
    observation = buffers["observation"].reshape(sample_count, 200)
    latent = buffers["latent"].reshape(sample_count, 2)
    old_log_probability = buffers["log_probability"].reshape(sample_count)
    advantage = buffers["advantage"].reshape(sample_count)
    returns = buffers["returns"].reshape(sample_count)
    normalized_returns = return_normalizer.normalize(returns)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1.0e-8)
    batch_size = sample_count // config.minibatches
    totals = torch.zeros(9, device=observation.device)
    updates = 0
    early_stopped = False
    stop_kl = 0.0
    epoch_profiles: list[dict[str, object]] = []
    for epoch in range(config.epochs):
        if profile is not None:
            torch.cuda.synchronize(observation.device)
            epoch_started = time.perf_counter()
            minibatch_profiles: list[dict[str, float]] = []
        permutation = torch.randperm(sample_count, device=observation.device)
        for minibatch, start in enumerate(range(0, sample_count, batch_size), start=1):
            if profile is not None:
                torch.cuda.synchronize(observation.device)
                minibatch_started = time.perf_counter()
            index = permutation[start : start + batch_size]
            new_log_probability, entropy = _evaluate_drive_action(
                actor, log_std, observation[index], latent[index]
            )
            value = critic(q15_to_float(observation[index]))
            ratio = torch.exp(new_log_probability - old_log_probability[index])
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip).to(torch.float32).mean()
            if profile is not None:
                kl_check_started = time.perf_counter()
            exceeds_kl = bool(approximate_kl > config.target_kl)
            kl_check_ms = (
                (time.perf_counter() - kl_check_started) * 1000
                if profile is not None
                else 0.0
            )
            if exceeds_kl:
                early_stopped = True
                stop_kl = float(approximate_kl.item())
                break
            policy_loss = -torch.minimum(
                ratio * advantage[index],
                ratio.clamp(1.0 - config.clip, 1.0 + config.clip) * advantage[index],
            ).mean()
            value_loss = 0.5 * (value - normalized_returns[index]).square().mean()
            entropy_mean = entropy.mean()
            rehearsal_count = round(
                batch_size
                * config.rehearsal_fraction
                / (1.0 - config.rehearsal_fraction)
            )
            rehearsal_index = torch.randint(
                0,
                rehearsal_observation.shape[0],
                (rehearsal_count,),
                device=observation.device,
            )
            retained = actor(rehearsal_observation[rehearsal_index])[..., 2:]
            retention_loss = F.mse_loss(retained, rehearsal_output[rehearsal_index, 2:])
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
                + config.retention_coefficient * retention_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            final_layer = actor.layers[-1]
            final_layer_gradient_norm = torch.sqrt(
                final_layer.weight.grad.square().sum()
                + final_layer.bias.grad.square().sum()
            )
            actor_grad_norm = nn.utils.clip_grad_norm_(
                list(actor.parameters()) + [log_std], config.max_grad_norm
            )
            critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                totals += torch.stack(
                    (
                        policy_loss,
                        value_loss,
                        entropy_mean,
                        retention_loss,
                        approximate_kl,
                        clip_fraction,
                        actor_grad_norm,
                        critic_grad_norm,
                        final_layer_gradient_norm,
                    )
                )
            updates += 1
            if profile is not None:
                torch.cuda.synchronize(observation.device)
                minibatch_profiles.append(
                    {
                        "minibatch": float(minibatch),
                        "total_ms": (time.perf_counter() - minibatch_started) * 1000,
                        "kl_check_ms": kl_check_ms,
                    }
                )
        if profile is not None:
            torch.cuda.synchronize(observation.device)
            epoch_profiles.append(
                {
                    "epoch": epoch + 1,
                    "total_ms": (time.perf_counter() - epoch_started) * 1000,
                    "minibatches": minibatch_profiles,
                }
            )
        if early_stopped:
            break
    totals /= updates
    names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "retention_loss",
        "approximate_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "final_policy_layer_grad_norm",
    )
    metrics = {name: float(value.item()) for name, value in zip(names, totals)}
    metrics.update(
        {
            "ppo_optimizer_updates": float(updates),
            "kl_early_stopped": float(early_stopped),
            "kl_stop_value": stop_kl,
        }
    )
    if profile is not None:
        profile["epochs"] = epoch_profiles
    return metrics


def _rehearsal_data(
    actor: TacticalActor, stats: ResolvedStats, device: torch.device, seed: int, count: int = 4096
) -> tuple[Tensor, Tensor]:
    environment = Stage1Environment(count, stats, device, seed, max_episode_steps=300)
    observation, _mask = environment.observation()
    with torch.no_grad():
        output = actor(observation)
    return observation, output


def scripted_drive(environment: Stage2Environment) -> Tensor:
    simulator = environment.simulator
    dx = simulator.position_x[:, 1] - simulator.position_x[:, 0]
    dy = simulator.position_y[:, 1] - simulator.position_y[:, 0]
    distance = integer_floor_sqrt(dx * dx + dy * dy)
    toward = simulator.trig.atan2(dx, dy)
    desired = torch.where(distance <= environment.stats.short_range, toward + 32768, toward)
    relative = angle_delta(desired - simulator.direction[:, 0])
    heading = torch.clamp(torch.div(relative, 16, rounding_mode="trunc"), -2048, 2047)
    move = (distance <= environment.stats.short_range) | (distance > environment.stats.long_range)
    return torch.stack((heading, move.to(torch.int64) * 256), dim=-1)


def train_stage2_seed(
    seed: int,
    stats: ResolvedStats,
    stats_path: Path,
    stage1_checkpoint: Path,
    config: Stage2Config,
    device: torch.device,
    output: Path,
    actor_initial_checkpoint: Path | None = None,
) -> tuple[TacticalActor, Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    initial_checkpoint = (
        checkpoint
        if actor_initial_checkpoint is None
        else torch.load(actor_initial_checkpoint, map_location="cpu", weights_only=False)
    )
    actor = TacticalActor().to(device)
    actor.load_state_dict(initial_checkpoint["actor"], strict=True)
    if actor_initial_checkpoint is None:
        rehearsal_actor = actor
    else:
        with torch.random.fork_rng(devices=[]):
            rehearsal_actor = TacticalActor().to(device)
        rehearsal_actor.load_state_dict(checkpoint["actor"], strict=True)
    rehearsal_observation, rehearsal_output = _rehearsal_data(
        rehearsal_actor, stats, device, seed
    )
    critic = Stage1Critic().to(device)
    return_normalizer = RunningReturnNormalizer(device)
    use_log_std_schedule = config.final_log_std is not None
    log_std = nn.Parameter(
        torch.full((2,), config.initial_log_std, device=device),
        requires_grad=not use_log_std_schedule,
    )
    optimizer_parameters = list(actor.parameters()) + list(critic.parameters())
    if not use_log_std_schedule:
        optimizer_parameters.append(log_std)
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=config.learning_rate,
        eps=1.0e-5,
        weight_decay=0.0,
        fused=device.type == "cuda",
    )
    environment = Stage2Environment(
        config.arenas,
        stats,
        device,
        seed,
        config.max_episode_steps,
        compile_simulator=config.compile_simulator,
        stagger_initial_episode_phase=config.stagger_initial_episode_phase,
        training_in_band_spawn_fraction=config.training_in_band_spawn_fraction,
    )
    observation = environment.observation()
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / "training.jsonl"
    global_steps = 0
    kl_stop_count = 0
    started = time.perf_counter()
    with telemetry_path.open("w", encoding="utf-8") as telemetry:
        for iteration in range(1, config.iterations + 1):
            scheduled_value = scheduled_log_std(config, iteration)
            if scheduled_value is not None:
                with torch.no_grad():
                    log_std.fill_(scheduled_value)
            range_progress_coefficient, heading_alignment_coefficient = (
                approach_reward_coefficients(config, iteration)
            )
            buffers = {
                "observation": torch.empty((config.rollout_steps, config.arenas, 200), dtype=torch.int16, device=device),
                "latent": torch.empty((config.rollout_steps, config.arenas, 2), device=device),
                "log_probability": torch.empty((config.rollout_steps, config.arenas), device=device),
                "reward": torch.empty((config.rollout_steps, config.arenas), device=device),
                "done": torch.empty((config.rollout_steps, config.arenas), dtype=torch.bool, device=device),
                "value": torch.empty((config.rollout_steps, config.arenas), device=device),
                "drive": torch.empty(
                    (config.rollout_steps, config.arenas, 2), dtype=torch.int64, device=device
                ),
            }
            for step in range(config.rollout_steps):
                with torch.no_grad():
                    drive, latent, log_probability, _entropy = drive_policy_action(
                        actor,
                        log_std,
                        observation,
                        deterministic=False,
                        heading_output_scale=config.heading_output_scale,
                    )
                    value = return_normalizer.denormalize(
                        critic(q15_to_float(observation))
                    )
                buffers["observation"][step] = observation
                buffers["latent"][step] = latent
                buffers["log_probability"][step] = log_probability
                buffers["value"][step] = value
                buffers["drive"][step] = drive
                observation, reward, done = environment.step(
                    drive,
                    range_objective_coefficient=config.range_objective_coefficient,
                    range_progress_coefficient=range_progress_coefficient,
                    heading_alignment_coefficient=heading_alignment_coefficient,
                    speed_settling_coefficient=config.speed_settling_coefficient,
                    clip_level_violation=config.clip_level_violation,
                )
                buffers["reward"][step] = reward
                buffers["done"][step] = done
            with torch.no_grad():
                last_value = return_normalizer.denormalize(
                    critic(q15_to_float(observation))
                )
                actor_output = actor(buffers["observation"].reshape(-1, 200))[:, :2]
                critic_raw_output = critic(
                    q15_to_float(buffers["observation"].reshape(-1, 200))
                )
                output_mean = actor_output.mean(dim=0)
                output_std = actor_output.std(dim=0, unbiased=False)
                output_saturation = (torch.tanh(actor_output).abs() > 0.99).to(
                    torch.float32
                ).mean(dim=0)
                commanded_speed_mean = buffers["drive"][..., 1].to(torch.float32).mean()
                commanded_heading_magnitude_mean = buffers["drive"][..., 0].abs().to(
                    torch.float32
                ).mean()
                critic_raw_output_mean = critic_raw_output.mean()
                critic_raw_output_std = critic_raw_output.std(unbiased=False)
                normalizer_mean_before_update = return_normalizer.mean.clone()
                normalizer_variance_before_update = return_normalizer.variance.clone()
                normalizer_count_before_update = return_normalizer.count.clone()
            advantage, returns = compute_gae(
                buffers["reward"], buffers["done"], buffers["value"], last_value, config.gamma, config.gae_lambda
            )
            critic_explained_variance = explained_variance(returns, buffers["value"])
            raw_return_mean = returns.mean()
            raw_return_std = returns.std(unbiased=False)
            return_normalizer.update(returns)
            normalized_value_target = return_normalizer.normalize(returns)
            normalized_value_target_mean = normalized_value_target.mean()
            normalized_value_target_std = normalized_value_target.std(unbiased=False)
            buffers["advantage"] = advantage
            buffers["returns"] = returns
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
            )
            kl_stop_count += int(metrics["kl_early_stopped"])
            global_steps += config.rollout_steps * config.arenas
            completed_episodes, completed_time_sum, completed_violation_sum = (
                environment.completed_metrics()
            )
            row = {
                "iteration": iteration,
                "environment_steps": global_steps,
                "completed_episodes": completed_episodes,
                "mean_completed_time_in_band_fraction": completed_time_sum / max(completed_episodes, 1),
                "mean_completed_normalized_band_violation": completed_violation_sum / max(completed_episodes, 1),
                "mean_reward": float(buffers["reward"].mean().item()),
                "explained_variance": float(critic_explained_variance.item()),
                "return_running_mean": float(return_normalizer.mean.item()),
                "return_running_std": float(return_normalizer.standard_deviation.item()),
                "return_running_variance": float(return_normalizer.variance.item()),
                "return_running_count": float(return_normalizer.count.item()),
                "return_running_mean_before_update": float(
                    normalizer_mean_before_update.item()
                ),
                "return_running_variance_before_update": float(
                    normalizer_variance_before_update.item()
                ),
                "return_running_count_before_update": float(
                    normalizer_count_before_update.item()
                ),
                "raw_return_mean": float(raw_return_mean.item()),
                "raw_return_std": float(raw_return_std.item()),
                "normalized_value_target_mean": float(
                    normalized_value_target_mean.item()
                ),
                "normalized_value_target_std": float(normalized_value_target_std.item()),
                "critic_raw_output_mean": float(critic_raw_output_mean.item()),
                "critic_raw_output_std": float(critic_raw_output_std.item()),
                "cumulative_kl_stop_count": kl_stop_count,
                "drive_log_std": [float(value) for value in log_std.detach().cpu().tolist()],
                "actor_output_mean": [float(value) for value in output_mean.cpu().tolist()],
                "actor_output_std": [float(value) for value in output_std.cpu().tolist()],
                "actor_output_saturation_fraction": [
                    float(value) for value in output_saturation.cpu().tolist()
                ],
                "mean_commanded_speed_fraction_q": float(commanded_speed_mean.item()),
                "mean_absolute_heading_delta_q": float(
                    commanded_heading_magnitude_mean.item()
                ),
                "elapsed_seconds": time.perf_counter() - started,
                **metrics,
            }
            telemetry.write(json.dumps(row, separators=(",", ":")) + "\n")
            telemetry.flush()
            if iteration == 1 or iteration % 8 == 0 or iteration == config.iterations:
                print(json.dumps({"seed": seed, **row}, separators=(",", ":")), flush=True)
    torch.save(
        {
            "contract": "warzone-tactical-v1",
            "stage": 2,
            "training_seed": seed,
            "stage1_checkpoint": str(stage1_checkpoint),
            "actor_initial_checkpoint": (
                None if actor_initial_checkpoint is None else str(actor_initial_checkpoint)
            ),
            "config": asdict(config),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "return_normalizer": return_normalizer.state_dict(),
            "drive_log_std": log_std.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "environment_steps": global_steps,
            "cumulative_kl_stop_count": kl_stop_count,
        },
        output / "checkpoint.pt",
    )
    export_artifacts(actor.cpu().eval(), output / "artifacts", stats_path)
    return actor.to(device).eval(), log_std.detach()


@torch.no_grad()
def evaluate_stage2(
    actor: TacticalActor | None,
    log_std: Tensor | None,
    stats: ResolvedStats,
    episodes: int,
    scenario_seed: int,
    device: torch.device,
    max_episode_steps: int = 300,
    controller: Literal["scripted", "do_nothing"] = "scripted",
    heading_output_scale: int = 2048,
) -> Stage2Evaluation:
    seeds = torch.arange(scenario_seed, scenario_seed + episodes, dtype=torch.int64, device=device)
    environment = Stage2Environment(episodes, stats, device, scenario_seed, max_episode_steps, seeds)
    observation = environment.observation()
    for _step in range(max_episode_steps):
        if actor is None:
            if controller == "scripted":
                drive = scripted_drive(environment)
            elif controller == "do_nothing":
                drive = torch.zeros((episodes, 2), dtype=torch.int64, device=device)
            else:
                raise ValueError(f"unknown Stage 2 controller: {controller}")
        else:
            if log_std is None:
                raise ValueError("a learned Stage 2 actor needs drive log standard deviation")
            drive, _latent, _log_probability, _entropy = drive_policy_action(
                actor,
                log_std,
                observation,
                deterministic=True,
                heading_output_scale=heading_output_scale,
            )
        observation, _reward, _done = environment.step(drive, auto_reset=False)
    fractions = environment.in_band_steps / max_episode_steps
    violation = environment.violation_sum / max_episode_steps
    final_distance = _distance(environment.simulator).to(torch.float32)
    sorted_fraction = fractions.sort().values
    p10_index = max(0, math.ceil(episodes * 0.10) - 1)
    return Stage2Evaluation(
        episodes=episodes,
        mean_time_in_band_fraction=float(fractions.mean().item()),
        median_time_in_band_fraction=float(fractions.median().item()),
        p10_time_in_band_fraction=float(sorted_fraction[p10_index].item()),
        mean_normalized_band_violation=float(violation.mean().item()),
        median_normalized_band_violation=float(violation.median().item()),
        mean_final_distance=float(final_distance.mean().item()),
    )


def _append_seed_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_seed_results(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pending_training_seeds(
    path: Path, training_seeds: tuple[int, ...]
) -> tuple[int, ...]:
    if not path.exists():
        path.write_text("", encoding="utf-8")
        return training_seeds
    rows = _load_seed_results(path)
    completed = [int(row["seed"]) for row in rows]
    if len(completed) != len(set(completed)):
        raise ValueError(f"duplicate completed seeds in {path}")
    unexpected = sorted(set(completed) - set(training_seeds))
    if unexpected:
        raise ValueError(f"completed seeds not requested by this run: {unexpected}")
    completed_set = set(completed)
    return tuple(seed for seed in training_seeds if seed not in completed_set)


def run_three_seed_stage2(
    stats_path: Path,
    stage1_output: Path,
    output: Path,
    config: Stage2Config,
    episodes: int,
    training_seeds: tuple[int, ...],
    promotion: bool,
    stage1_checkpoint_seed: int | None = None,
    actor_initial_checkpoint: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 training needs CUDA")
    if promotion and episodes != 10_000:
        raise ValueError("a Stage 2 promotion run needs exactly 10,000 episodes per seed")
    device = torch.device("cuda")
    stats = ResolvedStats.load(stats_path)
    if (stats.short_range, stats.long_range) != (512, 1088):
        raise ValueError("Contract V1 Stage 2 requires the resolved range band (512, 1088]")
    output.mkdir(parents=True, exist_ok=True)
    seed_results_path = output / "seed-results.jsonl"
    pending_seeds = _pending_training_seeds(seed_results_path, training_seeds)
    baseline = evaluate_stage2(None, None, stats, episodes, 0, device, config.max_episode_steps)
    do_nothing = evaluate_stage2(
        None,
        None,
        stats,
        episodes,
        0,
        device,
        config.max_episode_steps,
        controller="do_nothing",
    )
    for seed in pending_seeds:
        checkpoint_seed = seed if stage1_checkpoint_seed is None else stage1_checkpoint_seed
        stage1_checkpoint = stage1_output / f"seed-{checkpoint_seed}" / "checkpoint.pt"
        before_checkpoint = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        before_actor = TacticalActor().to(device)
        before_actor.load_state_dict(before_checkpoint["actor"], strict=True)
        retention_episodes = min(episodes, 1000)
        stage1_before = evaluate_stage1(before_actor, stats, retention_episodes, 0, device, 300)
        actor, log_std = train_stage2_seed(
            seed,
            stats,
            stats_path,
            stage1_checkpoint,
            config,
            device,
            output / f"seed-{seed}",
            actor_initial_checkpoint,
        )
        checkpoint = torch.load(
            output / f"seed-{seed}" / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        evaluation = evaluate_stage2(
            actor,
            log_std,
            stats,
            episodes,
            0,
            device,
            config.max_episode_steps,
            heading_output_scale=config.heading_output_scale,
        )
        stage1_after = evaluate_stage1(actor, stats, retention_episodes, 0, device, 300)
        _append_seed_result(
            seed_results_path,
            {
                "seed": seed,
                "evaluation": asdict(evaluation),
                "stage1_clear_rate_before": stage1_before.clear_rate,
                "stage1_clear_rate_after": stage1_after.clear_rate,
                "stage1_regression": stage1_before.clear_rate - stage1_after.clear_rate,
                "prior_stage_retention": {
                    "stage": 1,
                    "clear_rate_before": stage1_before.clear_rate,
                    "clear_rate_after": stage1_after.clear_rate,
                    "regression": stage1_before.clear_rate - stage1_after.clear_rate,
                    "maximum_allowed_regression": 0.05,
                    "gate_passed": stage1_after.clear_rate
                    >= stage1_before.clear_rate - 0.05,
                },
                "time_in_band_above_do_nothing": (
                    evaluation.mean_time_in_band_fraction
                    - do_nothing.mean_time_in_band_fraction
                ),
                "continuous_seed_floor_passed": evaluation.mean_time_in_band_fraction >= 0.65,
                "retention_gate_passed": stage1_after.clear_rate >= stage1_before.clear_rate - 0.05,
                "cumulative_kl_stop_count": int(checkpoint["cumulative_kl_stop_count"]),
            },
        )
    seed_rows = _load_seed_results(seed_results_path)
    time_fractions = [float(row["evaluation"]["mean_time_in_band_fraction"]) for row in seed_rows]
    violations = [float(row["evaluation"]["mean_normalized_band_violation"]) for row in seed_rows]
    mean_time_fraction = statistics.fmean(time_fractions)
    mean_violation = statistics.fmean(violations)
    passed = (
        mean_time_fraction >= 0.70
        and min(time_fractions) >= 0.65
        and mean_violation <= 0.10
        and all(bool(row["retention_gate_passed"]) for row in seed_rows)
    )
    report = {
        "contract": "warzone-tactical-v1",
        "stage": 2,
        "kind": "three_seed_promotion" if promotion else "three_seed_screening",
        "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "config": asdict(config),
        "stage1_checkpoint_seed": stage1_checkpoint_seed,
        "actor_initial_checkpoint": (
            None if actor_initial_checkpoint is None else str(actor_initial_checkpoint)
        ),
        "evaluation_episodes_per_seed": episodes,
        "evaluation_tier": (
            "promotion" if episodes == 10_000 else "screening" if episodes == 1_000 else "diagnostic"
        ),
        "scenario_seed_first": 0,
        "scenario_seed_last_inclusive": episodes - 1,
        "range_band": {"lower": 512, "lower_inclusive": False, "upper": 1088, "upper_inclusive": True},
        "primary_metric": "episode_time_in_band_fraction",
        "scripted_rangekeeper_v1": asdict(baseline),
        "do_nothing_v1": asdict(do_nothing),
        "seeds": seed_rows,
        "mean_time_in_band_fraction": mean_time_fraction,
        "mean_time_in_band_above_do_nothing": (
            mean_time_fraction - do_nothing.mean_time_in_band_fraction
        ),
        "lowest_seed_time_in_band_fraction": min(time_fractions),
        "mean_normalized_band_violation": mean_violation,
        "mean_gate_passed": mean_time_fraction >= 0.70,
        "five_point_seed_floor_passed": min(time_fractions) >= 0.65,
        "violation_gate_passed": mean_violation <= 0.10,
        "retention_gate_passed": all(bool(row["retention_gate_passed"]) for row in seed_rows),
        "screening_gate_passed": passed,
        "promotion_gate_passed": passed if promotion else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stage2-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--stage1-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arenas", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--rehearsal-fraction", type=float, default=0.20)
    parser.add_argument("--range-progress-start", type=float, default=0.1)
    parser.add_argument("--heading-alignment-start", type=float, default=0.02)
    parser.add_argument("--approach-anneal-iterations", type=int, default=0)
    parser.add_argument("--stage1-checkpoint-seed", type=int)
    parser.add_argument("--actor-initial-checkpoint", type=Path)
    parser.add_argument("--heading-output-scale", type=int, default=2048)
    parser.add_argument("--initial-log-std", type=float, default=-0.35)
    parser.add_argument("--final-log-std", type=float)
    parser.add_argument("--log-std-anneal-iterations", type=int, default=0)
    parser.add_argument("--clip-level-violation", action="store_true")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=(4, 17, 31))
    parser.add_argument("--promotion", action="store_true")
    parser.add_argument("--eager-simulator", action="store_true")
    parser.add_argument("--training-in-band-spawn-fraction", type=float)
    args = parser.parse_args()
    config = Stage2Config(
        arenas=args.arenas,
        iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        target_kl=args.target_kl,
        rehearsal_fraction=args.rehearsal_fraction,
        range_progress_start_coefficient=args.range_progress_start,
        heading_alignment_start_coefficient=args.heading_alignment_start,
        approach_anneal_iterations=args.approach_anneal_iterations,
        initial_log_std=args.initial_log_std,
        final_log_std=args.final_log_std,
        log_std_anneal_iterations=args.log_std_anneal_iterations,
        clip_level_violation=args.clip_level_violation,
        heading_output_scale=args.heading_output_scale,
        compile_simulator=not args.eager_simulator,
        training_in_band_spawn_fraction=args.training_in_band_spawn_fraction,
    )
    report = run_three_seed_stage2(
        args.resolved_stats,
        args.stage1_output,
        args.output,
        config,
        args.episodes,
        tuple(args.training_seeds),
        args.promotion,
        args.stage1_checkpoint_seed,
        args.actor_initial_checkpoint,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
