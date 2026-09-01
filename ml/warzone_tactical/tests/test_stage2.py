from __future__ import annotations

from pathlib import Path

import torch

from warzone_tactical.contracts import quantize_policy_output
from warzone_tactical.model import TacticalActor
from warzone_tactical.stage2 import (
    RunningReturnNormalizer,
    Stage2Config,
    Stage2Environment,
    _append_seed_result,
    _distance,
    _load_seed_results,
    _pending_training_seeds,
    approach_reward_coefficients,
    drive_policy_action,
    explained_variance,
    range_metrics,
    stage2_reward,
    scheduled_log_std,
)
from warzone_tactical.simulator import ResolvedStats


FIXTURE = Path(__file__).parent / "fixtures" / "resolved_stats_v1.json"


def test_return_normalizer_round_trips_and_tracks_combined_batches() -> None:
    normalizer = RunningReturnNormalizer(torch.device("cpu"))
    normalizer.update(torch.tensor([1.0, 2.0, 3.0]))
    normalizer.update(torch.tensor([4.0, 5.0]))
    values = torch.tensor([1.0, 3.0, 5.0])

    assert normalizer.mean == torch.tensor(3.0, dtype=torch.float64)
    assert torch.allclose(
        normalizer.variance, torch.tensor(2.0, dtype=torch.float64)
    )
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(values)), values)


def test_explained_variance_uses_unnormalized_returns() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert explained_variance(returns, returns) == 1.0
    assert explained_variance(returns, torch.zeros_like(returns)) == 0.0


def test_stage2_training_phase_stagger_is_seeded_and_evaluation_stays_at_zero() -> None:
    stats = ResolvedStats.load(FIXTURE)
    first = Stage2Environment(
        64,
        stats,
        torch.device("cpu"),
        7,
        compile_simulator=False,
        stagger_initial_episode_phase=True,
    )
    second = Stage2Environment(
        64,
        stats,
        torch.device("cpu"),
        7,
        compile_simulator=False,
        stagger_initial_episode_phase=True,
    )
    evaluation = Stage2Environment(
        64,
        stats,
        torch.device("cpu"),
        7,
        scenario_seeds=torch.arange(64),
        compile_simulator=False,
    )

    assert torch.equal(first.episode_step, second.episode_step)
    assert int(first.episode_step.min()) >= 0
    assert int(first.episode_step.max()) < 300
    assert torch.unique(first.episode_step).numel() > 1
    assert torch.equal(evaluation.episode_step, torch.zeros(64, dtype=torch.int64))


def test_stage2_training_spawn_mix_keeps_fixed_evaluation_unchanged() -> None:
    stats = ResolvedStats.load(FIXTURE)
    seeds = torch.arange(12)
    evaluation = Stage2Environment(
        12,
        stats,
        torch.device("cpu"),
        7,
        scenario_seeds=seeds,
        compile_simulator=False,
    )

    assert _distance(evaluation.simulator).tolist() == [
        876,
        378,
        1038,
        485,
        624,
        240,
        913,
        347,
        1076,
        454,
        662,
        208,
    ]


def test_stage2_training_spawn_mix_uses_ten_percent_in_band() -> None:
    stats = ResolvedStats.load(FIXTURE)
    environment = Stage2Environment(
        1000,
        stats,
        torch.device("cpu"),
        7,
        scenario_seeds=torch.arange(1000),
        compile_simulator=False,
        training_in_band_spawn_fraction=0.10,
    )
    distance = _distance(environment.simulator)
    in_band, _violation = range_metrics(distance, stats.short_range, stats.long_range)
    near = distance <= stats.short_range
    far = distance > stats.long_range

    assert 0.43 <= float(near.to(torch.float32).mean()) <= 0.47
    assert 0.09 <= float(in_band.to(torch.float32).mean()) <= 0.11
    assert 0.43 <= float(far.to(torch.float32).mean()) <= 0.47


def test_stage2_staggered_reset_resets_only_completed_arenas() -> None:
    stats = ResolvedStats.load(FIXTURE)
    environment = Stage2Environment(
        2,
        stats,
        torch.device("cpu"),
        7,
        compile_simulator=False,
        stagger_initial_episode_phase=True,
    )
    environment.episode_step[:] = torch.tensor([299, 10])

    _observation, _reward, done = environment.step(torch.zeros((2, 2), dtype=torch.int64))

    assert done.tolist() == [True, False]
    assert environment.episode_step.tolist() == [0, 11]


def test_stage2_range_band_is_open_then_closed() -> None:
    distance = torch.tensor([512, 513, 1088, 1089])
    in_band, violation = range_metrics(distance, 512, 1088)

    assert in_band.tolist() == [False, True, True, False]
    assert violation[0] > 0
    assert violation[1] == 0
    assert violation[2] == 0
    assert violation[3] > 0


def test_stage2_deterministic_drive_uses_contract_quantization() -> None:
    torch.manual_seed(1)
    actor = TacticalActor()
    observation = torch.zeros((8, 200), dtype=torch.int16)
    mask = torch.ones((8, 8), dtype=torch.bool)
    drive, _latent, _log_probability, _entropy = drive_policy_action(
        actor, torch.zeros(2), observation, deterministic=True
    )
    full_action = quantize_policy_output(actor(observation), mask)

    assert torch.equal(drive, full_action[:, :2])


def test_stage2_sampled_heading_wraps_at_one_turn() -> None:
    torch.manual_seed(3)
    actor = TacticalActor()
    observation = torch.zeros((1024, 200), dtype=torch.int16)
    drive, latent, log_probability, _entropy = drive_policy_action(
        actor, torch.tensor([1.0, 0.0]), observation, deterministic=False
    )

    assert torch.all(latent[:, 0] >= -1.0)
    assert torch.all(latent[:, 0] < 1.0)
    assert torch.all(drive[:, 0] >= -2048)
    assert torch.all(drive[:, 0] <= 2047)
    assert torch.isfinite(log_probability).all()


def test_stage2_reward_makes_band_dwell_dominate_approach() -> None:
    reward = stage2_reward(
        in_band=torch.tensor([True, False]),
        violation=torch.tensor([0.0, 0.0]),
        range_progress=torch.tensor([0.0, 12.5 / 576.0]),
        heading_alignment=torch.tensor([0.0, 1.0]),
        speed_quality=torch.tensor([1.0, 1.0]),
    )

    assert reward[0] == torch.tensor(1.02)
    assert reward[0] > reward[1]


def test_stage2_default_has_no_scripted_action_guidance() -> None:
    config = Stage2Config()

    assert config.iterations == 1500
    assert config.rollout_steps == 8
    assert config.compile_simulator
    assert config.target_kl == 0.015
    assert config.rehearsal_fraction == 0.20
    assert not hasattr(config, "action_guidance_coefficient")
    assert not hasattr(config, "behavior_clone_updates")


def test_stage2_approach_shaping_anneals_to_current_values() -> None:
    config = Stage2Config(
        range_progress_start_coefficient=1.0,
        heading_alignment_start_coefficient=0.2,
        approach_anneal_iterations=500,
    )

    assert approach_reward_coefficients(config, 1) == (1.0, 0.2)
    assert approach_reward_coefficients(config, 500) == (0.1, 0.02)
    assert approach_reward_coefficients(config, 501) == (0.1, 0.02)


def test_stage2_log_std_schedule_has_exact_endpoints() -> None:
    config = Stage2Config(
        initial_log_std=-2.0,
        final_log_std=-3.0,
        log_std_anneal_iterations=1000,
    )

    assert scheduled_log_std(config, 1) == -2.0
    assert scheduled_log_std(config, 1000) == -3.0
    assert scheduled_log_std(config, 1001) == -3.0


def test_stage2_violation_clip_changes_only_level_term() -> None:
    arguments = {
        "in_band": torch.tensor([False]),
        "violation": torch.tensor([2.0]),
        "range_progress": torch.tensor([0.5]),
        "heading_alignment": torch.tensor([0.25]),
        "speed_quality": torch.tensor([0.75]),
    }

    unclipped = stage2_reward(**arguments)
    clipped = stage2_reward(**arguments, clip_level_violation=True)

    assert torch.equal(clipped - unclipped, torch.tensor([1.0]))


def test_completed_seed_results_survive_an_interrupted_run(tmp_path) -> None:
    path = tmp_path / "seed-results.jsonl"
    first = {"seed": 4, "evaluation": {"value": 1}}
    second = {"seed": 17, "evaluation": {"value": 2}}

    _append_seed_result(path, first)
    _append_seed_result(path, second)

    assert _load_seed_results(path) == [first, second]


def test_completed_seed_results_are_skipped_when_run_resumes(tmp_path) -> None:
    path = tmp_path / "seed-results.jsonl"
    _append_seed_result(path, {"seed": 4})
    _append_seed_result(path, {"seed": 17})

    assert _pending_training_seeds(path, (4, 17, 31)) == (31,)
