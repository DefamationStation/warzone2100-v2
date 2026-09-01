from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from warzone_tactical.golden import build_observation_from_state, load_golden_rows


def test_golden_loader_checks_q15_bytes(tmp_path: Path) -> None:
    state = {
        "game_time": 100,
        "map": {"width": 64, "height": 64},
        "self": {
            "id": 1,
            "player": 0,
            "position": [4096, 4096, 128],
            "body": 300,
            "original_body": 300,
            "previous_body": 300,
            "base_speed": 450,
            "calculated_speed": 450,
            "speed": 0,
            "body_heading": 16384,
            "pitch": 0,
            "roll": 0,
            "turret_heading": 0,
            "weapon_last_fired": 0,
            "fire_pause": 200,
            "turn_speed": 40,
            "spin_speed": 80,
            "weapon_long_range": 768,
            "weapon_damage": 20,
            "armour": 10,
            "fire_on_move": True,
        },
        "previous_action": [0, 0, -1, 0],
        "allies": [{"assigned_id": 0, "unit": None} for _ in range(7)],
        "enemies": [{"assigned_id": 0, "unit": None} for _ in range(8)],
    }
    observation, mask = build_observation_from_state(state)
    row = {
        "observation_q15": observation.tolist(),
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "target_mask": mask.tolist(),
        "action": [0, 0, -1, 0],
        "golden_state": state,
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_golden_rows([trace], required=1) == [row]
