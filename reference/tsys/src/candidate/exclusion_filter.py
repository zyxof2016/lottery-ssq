from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_handler.cleaner import RED_COLUMNS
from src.feature_engineer.base_stats import occurrence_matrix


def _normalize_probability(values: np.ndarray, expected_size: int) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.shape != (int(expected_size),):
        raise ValueError(f"概率长度必须为 {expected_size}")
    if not np.all(np.isfinite(probability)):
        raise ValueError("概率包含 NaN 或无穷值")
    probability = np.clip(probability, 0.0, None)
    total = float(probability.sum())
    if total <= 0.0:
        return np.full(int(expected_size), 1.0 / int(expected_size), dtype=float)
    return probability / total


def _current_gaps(matrix: np.ndarray) -> np.ndarray:
    periods, number_count = matrix.shape
    last_seen = np.full(number_count, -1, dtype=int)
    for row_index, row in enumerate(matrix):
        last_seen[np.flatnonzero(row)] = row_index
    return np.where(last_seen >= 0, periods - 1 - last_seen, periods).astype(float)


def _gap_typicality(matrix: np.ndarray) -> np.ndarray:
    periods, number_count = matrix.shape
    gaps = _current_gaps(matrix)
    result = np.ones(number_count, dtype=float)
    for number in range(number_count):
        seen = np.flatnonzero(matrix[:, number])
        if len(seen) < 3:
            result[number] = 0.65
            continue
        historical_gaps = np.diff(seen) - 1
        mean_gap = float(historical_gaps.mean()) if len(historical_gaps) else float(periods / max(matrix[:, number].sum(), 1))
        std_gap = max(float(historical_gaps.std(ddof=0)) if len(historical_gaps) else mean_gap * 0.75, 1.0)
        z_score = abs(float(gaps[number]) - mean_gap) / std_gap
        result[number] = float(np.exp(-0.5 * min(z_score, 5.0) ** 2))
    return np.clip(result, 0.0, 1.0)


def _decayed_frequency(matrix: np.ndarray, decay: float) -> np.ndarray:
    if len(matrix) == 0:
        return np.zeros(matrix.shape[1], dtype=float)
    ages = np.arange(len(matrix) - 1, -1, -1, dtype=float)
    weights = np.power(float(decay), ages)
    weighted = weights @ matrix.astype(float)
    return weighted / max(float(weights.sum()), 1e-12)


def _rank_support(model_rankings: dict[str, dict[str, Any]] | None, *, color: str, number_count: int) -> np.ndarray:
    support = np.zeros(number_count, dtype=float)
    key = "top_red" if color == "red" else "top_blue"
    score_key = "red_weight" if color == "red" else "blue_weight"
    for model, summary in (model_rankings or {}).items():
        weight = max(float(summary.get(score_key, summary.get("weight", 0.0)) or 0.0), 0.03)
        top = [int(value) for value in summary.get(key, [])]
        if not top:
            continue
        limit = 18 if color == "red" else 8
        for rank, number in enumerate(top[:limit]):
            if 1 <= number <= number_count:
                support[number - 1] += weight * (1.0 - rank / max(limit, 1))
    maximum = float(support.max()) if support.size else 0.0
    return support / maximum if maximum > 0.0 else support


def _score_numbers(
    probability: np.ndarray,
    matrix: np.ndarray,
    *,
    base_rate: float,
    model_support: np.ndarray,
    decay: float,
) -> np.ndarray:
    periods = len(matrix)
    recent30 = matrix[-min(periods, 30):].mean(axis=0) if periods else np.zeros(matrix.shape[1])
    recent100 = matrix[-min(periods, 100):].mean(axis=0) if periods else np.zeros(matrix.shape[1])
    recent300 = matrix[-min(periods, 300):].mean(axis=0) if periods else np.zeros(matrix.shape[1])
    decayed = _decayed_frequency(matrix, decay)
    typicality = _gap_typicality(matrix)
    probability_relative = probability / max(base_rate, 1e-12)
    score = (
        0.34 * np.clip(probability_relative, 0.0, 2.5) / 2.5
        + 0.14 * np.clip(recent30 / max(base_rate, 1e-12), 0.0, 2.5) / 2.5
        + 0.13 * np.clip(recent100 / max(base_rate, 1e-12), 0.0, 2.5) / 2.5
        + 0.12 * np.clip(recent300 / max(base_rate, 1e-12), 0.0, 2.5) / 2.5
        + 0.12 * np.clip(decayed / max(base_rate, 1e-12), 0.0, 2.5) / 2.5
        + 0.08 * typicality
        + 0.07 * model_support
    )
    return np.clip(score, 0.0, None)


def _select_exclusions(
    score: np.ndarray,
    probability: np.ndarray,
    *,
    base_rate: float,
    max_excluded: int,
    min_remaining: int,
    score_quantile: float,
) -> list[int]:
    number_count = int(len(score))
    quota = min(max(0, int(max_excluded)), max(0, number_count - int(min_remaining)))
    if quota <= 0:
        return []
    threshold = float(np.quantile(score, min(max(float(score_quantile), 0.0), 1.0)))
    candidates = [
        index for index in range(number_count)
        if float(score[index]) <= threshold and float(probability[index]) <= float(base_rate)
    ]
    ordered = sorted(candidates, key=lambda idx: (float(score[idx]), float(probability[idx]), idx))
    return [int(index + 1) for index in ordered[:quota]]


def apply_low_confidence_exclusion(
    probabilities: dict[str, np.ndarray],
    history: pd.DataFrame,
    config: dict[str, Any],
    *,
    model_rankings: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Remove or downweight numbers with repeated low-confidence evidence.

    This is deliberately conservative: it does not claim a legal number is
    impossible.  It only zeroes a small quota when several left-truncated
    signals agree that the number is currently low-confidence; otherwise it
    merely returns the original normalized distribution.
    """
    exclusion_config = config.get("exclusion_filter", {})
    if not bool(exclusion_config.get("enabled", True)):
        return {
            "red": _normalize_probability(probabilities["red"], 33),
            "blue": _normalize_probability(probabilities["blue"], 16),
        }, {"enabled": False, "reason": "disabled"}
    if history is None or history.empty:
        return {
            "red": _normalize_probability(probabilities["red"], 33),
            "blue": _normalize_probability(probabilities["blue"], 16),
        }, {"enabled": False, "reason": "empty_history"}

    red_probability = _normalize_probability(probabilities["red"], 33)
    blue_probability = _normalize_probability(probabilities["blue"], 16)
    red_matrix = occurrence_matrix(history, "red").astype(float)
    blue_matrix = occurrence_matrix(history, "blue").astype(float)
    decay = float(exclusion_config.get("decay_factor", config.get("features", {}).get("decay_factor", 0.98)))

    red_score = _score_numbers(
        red_probability,
        red_matrix,
        base_rate=6.0 / 33.0,
        model_support=_rank_support(model_rankings, color="red", number_count=33),
        decay=decay,
    )
    blue_score = _score_numbers(
        blue_probability,
        blue_matrix,
        base_rate=1.0 / 16.0,
        model_support=_rank_support(model_rankings, color="blue", number_count=16),
        decay=decay,
    )
    red_low_confidence = _select_exclusions(
        red_score,
        red_probability,
        base_rate=6.0 / 33.0,
        max_excluded=int(exclusion_config.get("red_max_excluded", 5)),
        min_remaining=int(exclusion_config.get("red_min_remaining", 24)),
        score_quantile=float(exclusion_config.get("red_score_quantile", 0.18)),
    )
    blue_low_confidence = _select_exclusions(
        blue_score,
        blue_probability,
        base_rate=1.0 / 16.0,
        max_excluded=int(exclusion_config.get("blue_max_excluded", 2)),
        min_remaining=int(exclusion_config.get("blue_min_remaining", 13)),
        score_quantile=float(exclusion_config.get("blue_score_quantile", 0.16)),
    )

    adjusted_red = red_probability.copy()
    adjusted_blue = blue_probability.copy()
    hard_exclusion_enabled = bool(exclusion_config.get("hard_exclusion_enabled", False))
    red_excluded = list(red_low_confidence) if hard_exclusion_enabled else []
    blue_excluded = list(blue_low_confidence) if hard_exclusion_enabled else []
    for number in red_excluded:
        adjusted_red[int(number) - 1] = 0.0
    for number in blue_excluded:
        adjusted_blue[int(number) - 1] = 0.0
    adjusted_red = _normalize_probability(adjusted_red, 33)
    adjusted_blue = _normalize_probability(adjusted_blue, 16)
    mode = (
        "low_confidence_exclusion_before_candidate_generation"
        if hard_exclusion_enabled
        else "low_confidence_audit_only_no_hard_exclusion"
    )
    return {
        "red": adjusted_red,
        "blue": adjusted_blue,
    }, {
        "enabled": True,
        "mode": mode,
        "hard_exclusion_enabled": hard_exclusion_enabled,
        "red_excluded": red_excluded,
        "blue_excluded": blue_excluded,
        "red_low_confidence_candidates": red_low_confidence,
        "blue_low_confidence_candidates": blue_low_confidence,
        "red_remaining": int(33 - len(red_excluded)),
        "blue_remaining": int(16 - len(blue_excluded)),
        "red_scores": {
            f"{number:02d}": float(red_score[number - 1])
            for number in red_low_confidence
        },
        "blue_scores": {
            f"{number:02d}": float(blue_score[number - 1])
            for number in blue_low_confidence
        },
    }
