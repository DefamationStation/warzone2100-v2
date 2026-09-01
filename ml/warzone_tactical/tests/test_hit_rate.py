from __future__ import annotations

from warzone_tactical.hit_rate import compare_hit_rates


def test_hit_rate_comparison_records_the_inclusive_short_roll_contract() -> None:
    result = compare_hit_rates(
        {
            "hit_rate_samples": 100_000,
            "hit_rate_hits": 41_072,
            "hit_rate_band": "short",
            "hit_rate_chance": 40,
        },
        seed=0,
    )

    assert result["successful_integer_rolls"] == 41
    assert result["range_band"] == "short"
    assert result["passes"]


def test_hit_rate_comparison_records_the_inclusive_long_roll_contract() -> None:
    result = compare_hit_rates(
        {
            "hit_rate_samples": 100_000,
            "hit_rate_hits": 50_900,
            "hit_rate_band": "long",
            "hit_rate_chance": 50,
        },
        seed=0,
    )

    assert result["successful_integer_rolls"] == 51
    assert result["range_band"] == "long"
    assert result["passes"]
