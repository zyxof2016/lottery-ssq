from __future__ import annotations

from typing import Any

import numpy as np


def normalize_distribution(values: np.ndarray, size: int = 16) -> np.ndarray:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if scores.shape != (int(size),):
        scores = np.resize(scores, int(size)).astype(float)
    total = float(scores.sum())
    return scores / total if total > 0 else np.full(int(size), 1.0 / int(size))


def multinomial_probability(values: np.ndarray, *, size: int = 16, alpha: float = 1.0) -> np.ndarray:
    observed = np.asarray(values, dtype=int)
    observed = observed[(1 <= observed) & (observed <= int(size))]
    counts = np.bincount(observed - 1, minlength=int(size)).astype(float)
    return normalize_distribution(counts + float(alpha), int(size))


def symmetric_kl(left: np.ndarray, right: np.ndarray) -> float:
    left_p = np.clip(normalize_distribution(left, len(left)), 1e-12, 1.0)
    right_p = np.clip(normalize_distribution(right, len(left)), 1e-12, 1.0)
    return float(
        0.5 * np.sum(left_p * np.log(left_p / right_p))
        + 0.5 * np.sum(right_p * np.log(right_p / left_p))
    )


def detect_change_points(
    values: np.ndarray,
    *,
    min_segment: int = 80,
    max_points: int = 3,
    penalty: float = 0.018,
    size: int = 16,
) -> list[int]:
    """Lightweight PELT-style change-point detector for categorical draws.

    It keeps the runtime predictable by evaluating a coarse candidate grid and
    accepting only splits whose two-sided distribution divergence clears an
    explicit penalty.  The returned indexes are segment starts for the right
    side, never including 0 or len(values).
    """
    series = np.asarray(values, dtype=int)
    length = len(series)
    if length < int(min_segment) * 2:
        return []
    step = max(10, int(min_segment) // 2)
    candidates = range(int(min_segment), length - int(min_segment), step)
    scored: list[tuple[float, int]] = []
    for point in candidates:
        left = series[:point]
        right = series[point:]
        score = symmetric_kl(
            multinomial_probability(left, size=size),
            multinomial_probability(right, size=size),
        )
        if score >= float(penalty):
            recency_bonus = 1.0 + 0.35 * point / max(length - 1, 1)
            scored.append((float(score * recency_bonus), int(point)))
    selected: list[int] = []
    for _, point in sorted(scored, reverse=True):
        if all(abs(point - existing) >= int(min_segment) for existing in selected):
            selected.append(int(point))
        if len(selected) >= int(max_points):
            break
    return sorted(selected)


def regime_weight_profile(
    length: int,
    change_points: list[int],
    *,
    half_life: int = 520,
    pre_change_decay: float = 0.42,
) -> np.ndarray:
    if int(length) <= 0:
        return np.asarray([], dtype=float)
    age = np.arange(int(length) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / max(float(half_life), 1.0))
    if change_points:
        last_change = int(change_points[-1])
        weights[:last_change] *= float(pre_change_decay)
    total = float(weights.sum())
    return weights / total if total > 0 else np.full(int(length), 1.0 / int(length))


def weighted_recent_distribution(
    values: np.ndarray,
    *,
    change_points: list[int] | None = None,
    half_life: int = 520,
    pre_change_decay: float = 0.42,
    size: int = 16,
) -> np.ndarray:
    series = np.asarray(values, dtype=int)
    if len(series) == 0:
        return np.full(int(size), 1.0 / int(size))
    weights = regime_weight_profile(
        len(series),
        list(change_points or []),
        half_life=half_life,
        pre_change_decay=pre_change_decay,
    )
    scores = np.ones(int(size), dtype=float)
    valid = (1 <= series) & (series <= int(size))
    np.add.at(scores, series[valid] - 1, weights[valid] * len(series))
    return normalize_distribution(scores, int(size))


def regime_summary(values: np.ndarray, *, size: int = 16) -> dict[str, Any]:
    series = np.asarray(values, dtype=int)
    points = detect_change_points(series, size=size)
    probability = weighted_recent_distribution(series, change_points=points, size=size)
    return {
        "change_points": [int(point) for point in points],
        "last_change_point": int(points[-1]) if points else None,
        "weighted_probability": probability,
        "mode": "weighted_pelt_regime",
    }
