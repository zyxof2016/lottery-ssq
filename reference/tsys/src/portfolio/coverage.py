from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterable


def red_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def subset_coverage(tickets: list[dict], subset_size: int) -> set[tuple[int, ...]]:
    covered: set[tuple[int, ...]] = set()
    for ticket in tickets:
        covered.update(combinations(sorted(ticket["red"]), int(subset_size)))
    return covered


def marginal_subset_gain(selected: list[dict], candidate: dict) -> float:
    if not selected:
        return 1.0
    gains: list[float] = []
    for size in (2, 3, 4):
        existing = subset_coverage(selected, size)
        candidate_subsets = set(combinations(sorted(candidate["red"]), size))
        gains.append(len(candidate_subsets - existing) / max(len(candidate_subsets), 1))
    return float(sum(gains) / len(gains))


def coverage_summary(tickets: list[dict]) -> dict[str, float | int | dict]:
    if not tickets:
        return {
            "ticket_count": 0,
            "red_coverage": 0,
            "blue_coverage": 0,
            "blue_group_coverage_ratio": 0.0,
            "average_red_overlap": 0.0,
            "average_jaccard": 0.0,
            "subsets": {},
        }
    overlaps: list[int] = []
    jaccards: list[float] = []
    for left, right in combinations(tickets, 2):
        intersection = len(set(left["red"]) & set(right["red"]))
        overlaps.append(intersection)
        jaccards.append(red_jaccard(left["red"], right["red"]))
    subset_metrics: dict[str, dict[str, float | int]] = {}
    for size in (2, 3, 4):
        covered = subset_coverage(tickets, size)
        possible_selected = len(tickets) * comb(6, size)
        subset_metrics[str(size)] = {
            "unique": len(covered),
            "uniqueness_ratio": len(covered) / max(possible_selected, 1),
            "universe_ratio": len(covered) / comb(33, size),
        }
    blue_coverage = len({int(ticket["blue"]) for ticket in tickets})
    return {
        "ticket_count": len(tickets),
        "red_coverage": len({number for ticket in tickets for number in ticket["red"]}),
        "blue_coverage": blue_coverage,
        "blue_group_coverage_ratio": float(blue_coverage / 16.0),
        "average_red_overlap": float(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
        "average_jaccard": float(sum(jaccards) / len(jaccards)) if jaccards else 0.0,
        "subsets": subset_metrics,
    }


__all__ = ["coverage_summary", "marginal_subset_gain", "red_jaccard", "subset_coverage"]
