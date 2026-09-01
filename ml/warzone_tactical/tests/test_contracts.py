from __future__ import annotations

import torch

from warzone_tactical.contracts import _lround, q15_to_float, quantize_policy_output


def test_q15_conversion_uses_power_of_two_scale() -> None:
    values = torch.tensor([-32767, -1, 0, 1, 32767], dtype=torch.int16)
    actual = q15_to_float(values)
    expected = values.to(torch.float32) / 32768.0
    assert torch.equal(actual, expected)


def test_lround_uses_half_away_from_zero() -> None:
    values = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    expected = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    assert torch.equal(_lround(values), expected)


def test_invalid_target_logits_are_ignored() -> None:
    output = torch.zeros((2, 12))
    output[:, 3:11] = 100.0
    mask = torch.zeros((2, 8), dtype=torch.bool)
    action = quantize_policy_output(output, mask)
    assert torch.equal(action[:, 2], torch.tensor([-1, -1]))
    assert torch.equal(action[:, 3], torch.tensor([0, 0]))


def test_quantized_action_bounds() -> None:
    output = torch.tensor([[100.0, 100.0] + [0.0] * 10, [-100.0, -100.0] + [0.0] * 10])
    action = quantize_policy_output(output, torch.zeros((2, 8), dtype=torch.bool))
    assert action[0, 0].item() == 2047
    assert action[1, 0].item() == -2048
    assert action[0, 1].item() == 256
    assert action[1, 1].item() == 0


def test_quantized_action_can_limit_policy_heading_range() -> None:
    output = torch.zeros((2, 12))
    output[:, 0] = torch.tensor([100.0, -100.0])
    mask = torch.zeros((2, 8), dtype=torch.bool)

    action = quantize_policy_output(output, mask, heading_output_scale=128)

    assert action[:, 0].tolist() == [128, -128]
