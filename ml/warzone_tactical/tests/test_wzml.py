from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from warzone_tactical.contracts import quantize_policy_output
from warzone_tactical.model import TacticalActor
from warzone_tactical.wzml import WzmlActor, export_artifacts, make_fixture_actor

FIXTURE = Path(__file__).parent / "fixtures" / "resolved_stats_v1.json"


def test_wzml_matches_pytorch_quantized_actions(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    torch.manual_seed(17)
    actor = TacticalActor().eval()
    manifest = export_artifacts(actor, tmp_path, FIXTURE)
    compact = WzmlActor(Path(str(manifest["wzml_path"])))
    generator = np.random.default_rng(31)
    observation_q15 = generator.integers(-32767, 32768, size=(10_000, 200), dtype=np.int16)
    target_mask_np = generator.integers(0, 2, size=(10_000, 8), dtype=np.int8).astype(np.bool_)
    target_mask_np[:, 0] = True
    observation = observation_q15.astype(np.float32) * np.float32(2.0**-15)
    target_mask = torch.from_numpy(target_mask_np)
    with torch.no_grad():
        expected = actor.deterministic_action(torch.from_numpy(observation_q15), target_mask)
    actual_output = torch.from_numpy(compact(observation))
    actual = quantize_policy_output(actual_output, target_mask)
    assert torch.equal(actual, expected)
    loaded_manifest = json.loads((tmp_path / "policy.manifest.json").read_text())
    assert loaded_manifest["network"] == [200, 256, 256, 12]


def test_export_rejects_nonfinite_parameters(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    actor = make_fixture_actor()
    with torch.no_grad():
        actor.layers[-1].bias[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        export_artifacts(actor, tmp_path, FIXTURE)
