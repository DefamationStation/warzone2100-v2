"""Export and validate the restricted native WZML V1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor

from .model import TacticalActor

MAGIC = b"WZMLV1\0\0"
HEADER = struct.Struct("<8sIIIII")
CONTRACT = "warzone-tactical-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_layers(actor: TacticalActor) -> list[torch.nn.Linear]:
    layers = [module for module in actor.layers if isinstance(module, torch.nn.Linear)]
    if [(layer.in_features, layer.out_features) for layer in layers] != [
        (200, 256),
        (256, 256),
        (256, 12),
    ]:
        raise ValueError("actor does not match the frozen V1 shape")
    return layers


def validate_restricted_onnx(path: Path) -> None:
    """Reject any ONNX graph that the native V1 runtime cannot represent."""

    graph = onnx.load(path).graph
    raw_operations = [node.op_type for node in graph.node]
    if any(operation not in {"Identity", "Gemm", "Tanh"} for operation in raw_operations):
        raise ValueError(f"unsupported ONNX operations: {raw_operations}")
    operations = [operation for operation in raw_operations if operation != "Identity"]
    if operations != ["Gemm", "Tanh", "Gemm", "Tanh", "Gemm"]:
        raise ValueError(f"unsupported ONNX operation sequence: {operations}")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("V1 ONNX must have one input and one output")
    onnx.checker.check_model(onnx.load(path), full_check=True)


def export_artifacts(
    actor: TacticalActor,
    output_directory: Path,
    resolved_stats_path: Path,
) -> dict[str, object]:
    """Write ONNX, WZML, and the strict deployment manifest."""

    output_directory.mkdir(parents=True, exist_ok=True)
    resolved_stats_path = resolved_stats_path.resolve()
    if not resolved_stats_path.is_file():
        raise FileNotFoundError(resolved_stats_path)

    actor = actor.to("cpu").eval()
    layers = _linear_layers(actor)
    for layer_index, layer in enumerate(layers):
        if not torch.isfinite(layer.weight).all() or not torch.isfinite(layer.bias).all():
            raise ValueError(f"layer {layer_index} contains a non-finite parameter")
    onnx_path = (output_directory / "policy.onnx").resolve()
    wzml_path = (output_directory / "policy.wzml").resolve()
    manifest_path = (output_directory / "policy.manifest.json").resolve()

    torch.onnx.export(
        actor,
        torch.zeros((1, 200), dtype=torch.float32),
        onnx_path,
        input_names=["observation"],
        output_names=["policy_output"],
        dynamic_axes={"observation": {0: "batch"}, "policy_output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    validate_restricted_onnx(onnx_path)

    with wzml_path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, 1, 200, 256, 256, 12))
        for layer in layers:
            weight = layer.weight.detach().contiguous().numpy().astype("<f4", copy=False)
            bias = layer.bias.detach().contiguous().numpy().astype("<f4", copy=False)
            output.write(weight.tobytes(order="C"))
            output.write(bias.tobytes(order="C"))

    manifest: dict[str, object] = {
        "contract": CONTRACT,
        "format_version": 1,
        "observation": {"size": 200, "storage": "signed_q15", "model_scale": "2^-15"},
        "action": {
            "heading_steps_per_turn": 4096,
            "speed_steps": 256,
            "enemy_slots": 8,
            "none_target": -1,
        },
        "network": [200, 256, 256, 12],
        "onnx_path": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "wzml_path": str(wzml_path),
        "wzml_sha256": _sha256(wzml_path),
        "resolved_stats_path": str(resolved_stats_path),
        "resolved_stats_sha256": _sha256(resolved_stats_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class WzmlActor:
    """A Python parity reader with the same fixed layer evaluation order."""

    def __init__(self, path: Path) -> None:
        data = path.read_bytes()
        magic, version, input_size, hidden1, hidden2, output_size = HEADER.unpack_from(data)
        if (magic, version, input_size, hidden1, hidden2, output_size) != (
            MAGIC,
            1,
            200,
            256,
            256,
            12,
        ):
            raise ValueError("unsupported WZML header")
        offset = HEADER.size
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        for inputs, outputs in ((200, 256), (256, 256), (256, 12)):
            count = inputs * outputs
            weight = np.frombuffer(data, dtype="<f4", count=count, offset=offset).reshape(outputs, inputs)
            offset += count * 4
            bias = np.frombuffer(data, dtype="<f4", count=outputs, offset=offset)
            offset += outputs * 4
            self.layers.append((weight.copy(), bias.copy()))
        if offset != len(data):
            raise ValueError("WZML file has trailing or missing data")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        for index, (weight, bias) in enumerate(self.layers):
            value64 = value.astype(np.float64) @ weight.T.astype(np.float64)
            value64 += bias.astype(np.float64)
            value = (np.tanh(value64) if index < 2 else value64).astype(np.float32)
        return value


def make_fixture_actor() -> TacticalActor:
    """Create a fixed model that proves the full native loader path."""

    actor = TacticalActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        final = _linear_layers(actor)[-1]
        final.bias[1] = 4.0  # Full forward speed.
        final.bias[2] = -1.0  # Prefer a target over none.
        final.bias[3] = 1.0  # Select enemy slot zero when it is valid.
        final.bias[11] = 1.0  # Request fire.
    return actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-stats", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()

    actor = make_fixture_actor() if args.fixture else TacticalActor()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        actor.load_state_dict(state["actor"] if "actor" in state else state)
    export_artifacts(actor, args.output, args.resolved_stats)


if __name__ == "__main__":
    main()
