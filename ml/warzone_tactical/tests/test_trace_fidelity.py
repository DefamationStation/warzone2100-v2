from __future__ import annotations

import pytest

from warzone_tactical.trace_fidelity import group_ticks


def test_trace_rows_are_grouped_by_first_game_time_order() -> None:
    rows = [
        {"game_time": 102, "droid_id": 2},
        {"game_time": 102, "droid_id": 1},
        {"game_time": 202, "droid_id": 2},
        {"game_time": 202, "droid_id": 1},
        {"game_time": 302, "droid_id": 2},
        {"game_time": 302, "droid_id": 1},
    ]

    grouped = group_ticks(rows, maximum_ticks=2)

    assert [[row["droid_id"] for row in tick] for tick in grouped] == [[2, 1], [2, 1]]


def test_trace_window_rejects_a_changed_droid_set() -> None:
    rows = [
        {"game_time": 102, "droid_id": 1},
        {"game_time": 102, "droid_id": 2},
        {"game_time": 202, "droid_id": 1},
        {"game_time": 202, "droid_id": 3},
    ]

    with pytest.raises(ValueError, match="droid set changed"):
        group_ticks(rows, maximum_ticks=2)
