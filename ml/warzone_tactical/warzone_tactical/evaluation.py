"""Fixed benchmark results and non-self-referential promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

FIXED_BENCHMARKS = (
    "scripted_nearest_v1",
    "scripted_rangekeeper_v1",
    "wz_stock_attack_v1",
    "ml_anchor_1_v1",
    "ml_anchor_2_v1",
    "ml_anchor_3_v1",
)


@dataclass(frozen=True, slots=True)
class Results:
    wins: int
    losses: int
    draws: int
    unfinished: int
    finish_times: tuple[int, ...] = ()

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws + self.unfinished

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def unfinished_rate(self) -> float:
        return self.unfinished / self.total if self.total else 0.0

    def report(self) -> dict[str, object]:
        result = asdict(self)
        result["win_rate"] = self.win_rate
        result["unfinished_rate"] = self.unfinished_rate
        return result


def passes_three_seed_gate(
    seed_results: tuple[Results, Results, Results], minimum_win_rate: float, maximum_unfinished: float
) -> bool:
    mean_win_rate = sum(result.win_rate for result in seed_results) / 3
    mean_unfinished = sum(result.unfinished_rate for result in seed_results) / 3
    if mean_win_rate < minimum_win_rate or mean_unfinished > maximum_unfinished:
        return False
    return all(result.win_rate >= minimum_win_rate - 0.05 for result in seed_results)


def passes_incumbent_gate(seed_results: tuple[tuple[Results, Results], ...]) -> bool:
    """Check 250 engine matches from each side for all three seeds."""

    if len(seed_results) != 3:
        return False
    for first_side, second_side in seed_results:
        if first_side.total != 250 or second_side.total != 250:
            return False
        combined = Results(
            first_side.wins + second_side.wins,
            first_side.losses + second_side.losses,
            first_side.draws + second_side.draws,
            first_side.unfinished + second_side.unfinished,
        )
        if combined.win_rate < 0.55 or combined.unfinished_rate > 0.10:
            return False
        if first_side.win_rate < 0.50 or second_side.win_rate < 0.50:
            return False
    return True
