from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.portfolio.coverage import red_jaccard


def _gaussian_similarity(value: float, mean: float, std: float) -> float:
    z_score = (value - mean) / max(std, 0.5)
    return float(math.exp(-0.5 * z_score * z_score))


def structure_score(red: tuple[int, ...], profile: dict[str, tuple[float, float]]) -> float:
    ordered = tuple(sorted(int(number) for number in red))
    values = {
        "sum": sum(ordered),
        "odd_count": sum(number % 2 for number in ordered),
        "big_count": sum(number > 16 for number in ordered),
        "zone1": sum(number <= 11 for number in ordered),
        "zone2": sum(12 <= number <= 22 for number in ordered),
        "zone3": sum(number >= 23 for number in ordered),
        "span": ordered[-1] - ordered[0],
        "consecutive_pairs": sum((right - left) == 1 for left, right in zip(ordered, ordered[1:])),
    }
    return float(np.mean([
        _gaussian_similarity(values[name], *profile[name])
        for name in values
    ]))


def number_score(red: tuple[int, ...], blue: int, probabilities: dict[str, np.ndarray]) -> float:
    red_values = np.asarray(probabilities["red"], dtype=float)[np.array(red) - 1] * 33.0
    blue_value = float(probabilities["blue"][blue - 1]) * 16.0
    relative_values = np.concatenate([red_values, np.array([blue_value])])
    return float(np.exp(np.mean(np.log(np.clip(relative_values, 1e-12, None)))))


def base_quality(
    red_number_value: float,
    blue_number_value: float,
    structure_value: float,
    config: dict[str, Any],
) -> float:
    if any(key in config for key in ("red_number_weight", "blue_number_weight", "structure_weight")):
        red_weight = float(config.get("red_number_weight", 0.55))
        blue_weight = float(config.get("blue_number_weight", 0.10))
        structure_weight = float(config.get("structure_weight", 0.35))
    else:
        red_weight = float(config.get("number_weight", 0.65))
        blue_weight = 0.0
        structure_weight = float(config.get("structure_weight", 0.35))
    return (
        red_weight * float(red_number_value)
        + blue_weight * float(blue_number_value)
        + structure_weight * float(structure_value)
    )


def selection_score(candidate: dict, selected: list[dict], config: dict[str, Any]) -> float:
    overlap_penalty = max(
        (red_jaccard(candidate["red"], item["red"]) for item in selected),
        default=0.0,
    )
    return float(candidate["quality_score"] - float(config.get("diversity_penalty", 0.25)) * overlap_penalty)
