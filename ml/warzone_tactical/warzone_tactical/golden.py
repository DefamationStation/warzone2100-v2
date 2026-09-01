"""Checks for C++ observation goldens and native action parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .contracts import quantize_policy_output
from .wzml import WzmlActor

Q15_MAX = 32767
TILE_UNITS = 128
MAX_PLAYERS = 11

_INDEX = np.arange(0x4001, dtype=np.float64)
_SIN_TABLE = (65536.0 * np.sin(_INDEX * (np.pi / 0x8000)) + 0.5).astype(np.int64)
_SIN_TABLE -= _INDEX != 0
_ATAN_INDEX = np.arange(0x2001, dtype=np.float64)
_ATAN_TABLE = (0x8000 / np.pi * np.arctan(_ATAN_INDEX / 0x2000) + 0.5).astype(np.int64)


def _trunc_div(numerator: int, denominator: int) -> int:
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def _q15_ratio(value: int, scale: int) -> int:
    if scale <= 0:
        return 0
    return max(-Q15_MAX, min(Q15_MAX, _trunc_div(value * Q15_MAX, scale)))


def _q15_unsigned(value: int, maximum: int) -> int:
    if maximum <= 0:
        return 0
    return min(Q15_MAX, value * Q15_MAX // maximum)


def _angle_delta(angle: int) -> int:
    wrapped = angle & 0xFFFF
    return wrapped - 0x10000 if wrapped >= 0x8000 else wrapped


def _isin(angle: int) -> int:
    angle &= 0xFFFF
    quadrant = angle >> 14
    remainder = angle & 0x3FFF
    reverse = quadrant in (1, 3)
    table_index = 0x4000 - remainder if reverse else remainder
    sign = 1 if quadrant < 2 else -1
    return sign * (int(_SIN_TABLE[table_index]) + int(table_index != 0))


def _icos(angle: int) -> int:
    return _isin(angle + 0x4000)


def _isin_r(angle: int, radius: int) -> int:
    return _trunc_div(radius * _isin(angle), 65536)


def _icos_r(angle: int, radius: int) -> int:
    return _trunc_div(radius * _icos(angle), 65536)


def _iatan2(sine: int, cosine: int) -> int:
    if sine == 0 and cosine == 0:
        return 0
    case = (int(sine < 0) << 1) + int(cosine < 0)
    if case == 0:
        j, k, base = sine, cosine, 0
    elif case == 1:
        j, k, base = -cosine, sine, 0x4000
    elif case == 2:
        j, k, base = cosine, -sine, 0xC000
    else:
        j, k, base = -sine, -cosine, 0x8000
    if j < k:
        return (base + int(_ATAN_TABLE[(j * 0x2000 + k // 2) // k])) & 0xFFFF
    return (base + 0x4000 - int(_ATAN_TABLE[(k * 0x2000 + j // 2) // j])) & 0xFFFF


def _put_unit(output: np.ndarray, offset: int, observer: dict[str, object], entry: dict[str, object], width: int) -> None:
    unit = entry["unit"]
    if unit is None:
        return
    position = unit["position"]
    observer_position = observer["position"]
    dx = int(position[0]) - int(observer_position[0])
    dy = int(position[1]) - int(observer_position[1])
    heading = int(observer["body_heading"])
    local_x = _icos_r(heading, dx) - _isin_r(heading, dy)
    local_y = _isin_r(heading, dx) + _icos_r(heading, dy)
    output[offset + 0] = _q15_ratio(local_x, 16 * TILE_UNITS)
    output[offset + 1] = _q15_ratio(local_y, 16 * TILE_UNITS)
    output[offset + 2] = _q15_unsigned(int(unit["body"]), max(int(unit["original_body"]), 1))
    output[offset + 3] = _q15_ratio(_angle_delta(int(unit["body_heading"]) - heading), 32768)
    output[offset + 4] = _q15_unsigned(int(unit["speed"]), max(int(unit["base_speed"]), 1))
    output[offset + 5] = Q15_MAX
    if width > 6:
        output[offset + 6] = _q15_ratio(int(position[2]) - int(observer_position[2]), 4 * TILE_UNITS)
    if width > 7:
        output[offset + 7] = Q15_MAX if int(unit["weapon_count"]) > 0 else 0
    if width > 8:
        output[offset + 8] = Q15_MAX if unit["visible"] else 0
    if width > 9:
        output[offset + 9] = Q15_MAX
    if width > 10:
        output[offset + 10] = _q15_unsigned(int(unit["id"]) & 0x7FFF, 0x7FFF)


def build_observation_from_state(state: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild one C++ golden observation without using saved output values."""

    output = np.zeros(200, dtype=np.int16)
    mask = np.zeros(8, dtype=np.uint8)
    own = state["self"]
    game_time = int(state["game_time"])
    previous_action = [int(value) for value in state["previous_action"]]
    enemies = state["enemies"]

    selected = None
    if 0 <= previous_action[2] < 8:
        candidate = enemies[previous_action[2]]["unit"]
        if candidate is not None and candidate["visible"] and int(candidate["player"]) != int(own["player"]):
            selected = candidate
    turret_error = 0
    turret_aligned = False
    if selected is not None:
        target_position = selected["position"]
        own_position = own["position"]
        target_heading = _iatan2(
            int(target_position[0]) - int(own_position[0]),
            int(target_position[1]) - int(own_position[1]),
        )
        turret_error = _angle_delta(
            target_heading - int(own["body_heading"]) - int(own["turret_heading"])
        )
        turret_aligned = turret_error == 0
        if own.get("direct_weapon", False) and "muzzle_base_position" in own:
            own_position = own["position"]
            target_position = selected["position"]
            minimum_range = int(own.get("weapon_minimum_range", 0))
            object_dx = int(target_position[0]) - int(own_position[0])
            object_dy = int(target_position[1]) - int(own_position[1])
            if object_dx * object_dx + object_dy * object_dy > minimum_range * minimum_range:
                muzzle = own["muzzle_base_position"]
                pitch_dx = int(target_position[0]) - int(muzzle[0])
                pitch_dy = int(target_position[1]) - int(muzzle[1])
                pitch_distance = math.isqrt(pitch_dx * pitch_dx + pitch_dy * pitch_dy)
                target_pitch = _angle_delta(
                    _iatan2(int(target_position[2]) - int(muzzle[2]), pitch_distance)
                )
                target_pitch = max(
                    int(own["minimum_elevation"]) * 65536 // 360,
                    min(int(own["maximum_elevation"]) * 65536 // 360, target_pitch),
                )
                turret_aligned = turret_aligned and target_pitch == _angle_delta(
                    int(own.get("turret_pitch", 0))
                )

    fire_pause = int(own["fire_pause"])
    since_fired = game_time - int(own["weapon_last_fired"])
    output[0] = _q15_unsigned(int(own["body"]), max(int(own["original_body"]), 1))
    output[1] = _q15_unsigned(int(own["speed"]), max(int(own["base_speed"]), 1))
    output[2] = _q15_unsigned(int(own["calculated_speed"]), max(int(own["base_speed"]), 1))
    output[3] = _isin_r(int(own["body_heading"]), Q15_MAX)
    output[4] = _icos_r(int(own["body_heading"]), Q15_MAX)
    output[5] = _q15_unsigned(min(since_fired, fire_pause), max(fire_pause, 1))
    output[6] = Q15_MAX if since_fired >= fire_pause else 0
    output[7] = _q15_ratio(_angle_delta(int(own["turret_heading"])), 32768)
    output[8] = _q15_ratio(turret_error, 32768)
    output[9] = Q15_MAX if selected is not None and turret_aligned else 0
    output[10] = _q15_unsigned(
        max(0, int(own["previous_body"]) - int(own["body"])), max(int(own["original_body"]), 1)
    )
    turn_speed = int(own["turn_speed"])
    spin_speed = int(own["spin_speed"])
    output[11] = _q15_unsigned(turn_speed, max(turn_speed, spin_speed))
    output[12] = _q15_unsigned(spin_speed, max(turn_speed, spin_speed))
    output[13] = _q15_unsigned(int(own["weapon_long_range"]), 16 * TILE_UNITS)
    output[14] = _q15_unsigned(int(own["weapon_damage"]), 1000)
    output[15] = _q15_unsigned(int(own["armour"]), 1000)
    output[16] = Q15_MAX if own["fire_on_move"] else 0
    output[17] = _q15_ratio(_angle_delta(int(own["pitch"])), 32768)
    output[18] = _q15_ratio(_angle_delta(int(own["roll"])), 32768)
    output[19] = Q15_MAX

    world_width = int(state["map"]["width"]) * TILE_UNITS
    world_height = int(state["map"]["height"]) * TILE_UNITS
    position = own["position"]
    for ray in range(8):
        direction = int(own["body_heading"]) + ray * 8192
        dx = _isin_r(direction, 16 * TILE_UNITS)
        dy = _icos_r(direction, 16 * TILE_UNITS)
        fraction = Q15_MAX
        if dx > 0:
            fraction = min(fraction, (world_width - 1 - int(position[0])) * Q15_MAX // dx)
        if dx < 0:
            fraction = min(fraction, (int(position[0]) - 1) * Q15_MAX // -dx)
        if dy > 0:
            fraction = min(fraction, (world_height - 1 - int(position[1])) * Q15_MAX // dy)
        if dy < 0:
            fraction = min(fraction, (int(position[1]) - 1) * Q15_MAX // -dy)
        output[20 + ray] = max(0, min(Q15_MAX, fraction))

    for slot, entry in enumerate(state["allies"]):
        _put_unit(output, 28 + slot * 10, own, entry, 10)
    for slot, entry in enumerate(enemies):
        _put_unit(output, 98 + slot * 11, own, entry, 11)
        unit = entry["unit"]
        mask[slot] = int(
            unit is not None and unit["visible"] and int(unit["player"]) != int(own["player"])
        )

    output[190] = _q15_unsigned(game_time % 1000, 999)
    output[191] = _q15_ratio(previous_action[0], 2048)
    output[192] = _q15_unsigned(previous_action[1], 256)
    output[193] = -Q15_MAX if previous_action[2] < 0 else _q15_unsigned(previous_action[2] + 1, 8)
    output[194] = Q15_MAX if previous_action[3] else 0
    phase = (game_time * 65536 // 90000) & 0xFFFF
    output[195] = _isin_r(phase, Q15_MAX)
    output[196] = _icos_r(phase, Q15_MAX)
    output[197] = _q15_unsigned(int(own["player"]), MAX_PLAYERS - 1)
    output[198] = _q15_unsigned(int(own["id"]) & 0x7FFF, 0x7FFF)
    output[199] = Q15_MAX
    return output, mask


def load_golden_rows(paths: list[Path], required: int = 10_000) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            observation = np.asarray(row["observation_q15"], dtype="<i2")
            if observation.shape != (200,):
                raise AssertionError("golden observation does not have 200 values")
            if hashlib.sha256(observation.tobytes()).hexdigest() != row["observation_sha256"]:
                raise AssertionError("golden observation hash does not match its values")
            if len(row["target_mask"]) != 8:
                raise AssertionError("golden target mask does not have eight enemy slots")
            rebuilt, rebuilt_mask = build_observation_from_state(row["golden_state"])
            if not np.array_equal(rebuilt, observation):
                mismatch = int(np.flatnonzero(rebuilt != observation)[0])
                raise AssertionError(
                    f"Python observation mismatch at index {mismatch}: {rebuilt[mismatch]} != {observation[mismatch]}"
                )
            if not np.array_equal(rebuilt_mask, np.asarray(row["target_mask"], dtype=np.uint8)):
                raise AssertionError("Python target mask does not match the C++ target mask")
            rows.append(row)
            if len(rows) == required:
                return rows
    raise AssertionError(f"only {len(rows)} golden states were available; {required} are required")


def check_native_actions(rows: list[dict[str, object]], model_path: Path) -> None:
    model = WzmlActor(model_path)
    observations = np.asarray([row["observation_q15"] for row in rows], dtype=np.int16)
    masks = torch.tensor([row["target_mask"] for row in rows], dtype=torch.bool)
    outputs = torch.from_numpy(model(observations.astype(np.float32) * (2.0**-15)))
    actions = quantize_policy_output(outputs, masks).numpy()
    expected = np.asarray([row["action"] for row in rows], dtype=np.int64)
    if not np.array_equal(actions, expected):
        mismatch = int(np.flatnonzero(np.any(actions != expected, axis=1))[0])
        raise AssertionError(
            f"native action mismatch at golden row {mismatch}: {actions[mismatch]} != {expected[mismatch]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--required", type=int, default=10_000)
    parser.add_argument("--wzml", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_golden_rows(args.traces, args.required)
    if args.wzml:
        check_native_actions(rows, args.wzml)
    result = {
        "contract": "warzone-tactical-v1",
        "golden_states": len(rows),
        "observation_q15_exact": True,
        "native_actions_checked": bool(args.wzml),
        "native_actions_exact": True if args.wzml else None,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
