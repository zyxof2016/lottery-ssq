from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np


BLUE_POLICY_VERSION = "blue_adaptive_policy_v1"


def _blue_array(history: Iterable[Any]) -> np.ndarray:
    values: list[int] = []
    for value in history:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 16:
            values.append(number)
    return np.asarray(values, dtype=int)


def _normalize(values: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(scores.sum())
    if total <= 0:
        return np.full(16, 1.0 / 16.0)
    return scores / total


def _frequency(history: np.ndarray, window: int) -> np.ndarray:
    values = history[-max(1, min(int(window), len(history))):]
    return _normalize(np.bincount(values - 1, minlength=16) + 1.0)


def _time_decay(history: np.ndarray, decay: float = 0.985) -> np.ndarray:
    if len(history) == 0:
        return np.full(16, 1.0 / 16.0)
    # The tail contains essentially all of the usable mass while keeping the
    # rolling OOS selector cheap enough to run for every historical target.
    history = history[-300:]
    scores = np.ones(16, dtype=float)
    weights = float(decay) ** np.arange(len(history) - 1, -1, -1)
    for blue, weight in zip(history, weights):
        scores[int(blue) - 1] += float(weight)
    return _normalize(scores)


def _policy_probabilities(history: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "frequency_60": _frequency(history, 60),
        "frequency_100": _frequency(history, 100),
        "frequency_300": _frequency(history, 300),
        "time_decay": _time_decay(history),
        "frequency_decay_hybrid": _normalize(
            0.50 * _frequency(history, 100) + 0.50 * _time_decay(history)
        ),
    }


def _topk(probability: np.ndarray, count: int, excluded: set[int]) -> list[int]:
    order = [int(index + 1) for index in np.argsort(probability)[::-1]]
    result = [number for number in order if number not in excluded]
    return result[: max(1, min(int(count), len(result)))]


def _evaluate_policy(
    history: np.ndarray,
    builder: Callable[[np.ndarray], np.ndarray],
    *,
    count: int,
    window: int,
    recency_decay: float,
) -> tuple[float, int]:
    if len(history) < max(12, int(window) + 1):
        return -1.0, 0
    start = max(1, len(history) - int(window))
    hits: list[float] = []
    weights: list[float] = []
    for index in range(start, len(history)):
        train = history[:index]
        actual = int(history[index])
        selected = _topk(builder(train), count, set())
        hits.append(1.0 if actual in selected else 0.0)
        weights.append(float(recency_decay) ** float(len(history) - 1 - index))
    if not hits:
        return -1.0, 0
    score = float(np.average(np.asarray(hits), weights=np.asarray(weights)))
    return score, len(hits)


def select_adaptive_blue_set(
    history: Iterable[Any],
    *,
    count: int,
    probability: np.ndarray | None = None,
    evaluation_window: int = 60,
    minimum_history: int = 120,
    excluded_blues: set[int] | None = None,
    recency_decay: float = 0.985,
    fallback_policy: str = "frequency_100",
) -> tuple[list[int], dict[str, Any]]:
    """Select a finite blue set using only information available before the target draw.

    The selector is deliberately small and auditable. It compares a fixed pool of
    historical policies on completed pre-target observations, then applies the
    winning policy to the current history. Model probability is only a tie-breaker
    when two policies are effectively tied; it cannot manufacture future evidence.
    """
    values = _blue_array(history)
    count = max(1, min(int(count), 16))
    excluded = {int(v) for v in (excluded_blues or set()) if 1 <= int(v) <= 16}
    policies = _policy_probabilities
    builders = {
        name: (lambda h, n=name: policies(h)[n])
        for name in policies(values)
    }
    fallback = str(fallback_policy)
    if fallback not in builders:
        fallback = "frequency_100"

    scores: dict[str, float] = {}
    samples: dict[str, int] = {}
    for name, builder in builders.items():
        score, sample = _evaluate_policy(
            values,
            builder,
            count=count,
            window=int(evaluation_window),
            recency_decay=float(recency_decay),
        )
        scores[name] = float(score)
        samples[name] = int(sample)

    eligible = [
        name for name in builders
        if samples.get(name, 0) >= max(12, min(int(evaluation_window), int(minimum_history) if len(values) >= int(minimum_history) else 12))
        and scores.get(name, -1.0) >= 0.0
    ]
    if not eligible:
        selected_policy = fallback
        selection_reason = "insufficient_pre_target_history_for_policy_selection"
    else:
        selected_policy = max(
            eligible,
            key=lambda name: (scores.get(name, -1.0), 1 if name == fallback else 0, -len(name)),
        )
        selection_reason = "rolling_oos_policy_selection"

    selected_probability = builders[selected_policy](values)
    if probability is not None:
        model_probability = _normalize(np.asarray(probability, dtype=float))
        # Only break near-ties. Historical OOS policy remains the primary signal.
        best_score = scores.get(selected_policy, 0.0)
        tied = [
            name for name in eligible
            if abs(scores.get(name, -1.0) - best_score) <= 0.015
        ]
        if len(tied) > 1:
            candidate_pool = np.mean([builders[name](values) for name in tied], axis=0)
            selected_probability = _normalize(0.85 * selected_probability + 0.15 * candidate_pool + 0.0 * model_probability)

    selected = _topk(selected_probability, count, excluded)
    if len(selected) < count:
        model_order = _topk(_normalize(probability) if probability is not None else selected_probability, 16, excluded)
        for number in model_order:
            if number not in selected:
                selected.append(number)
            if len(selected) >= count:
                break

    return selected[:count], {
        "schema": BLUE_POLICY_VERSION,
        "selected_policy": selected_policy,
        "selection_reason": selection_reason,
        "evaluation_window": int(evaluation_window),
        "minimum_history": int(minimum_history),
        "history_length": int(len(values)),
        "policy_scores": {key: float(value) for key, value in sorted(scores.items())},
        "policy_samples": samples,
        "selected_blue": [int(v) for v in selected[:count]],
        "excluded_blue_numbers": sorted(excluded),
        "count": int(count),
        "objective": f"rolling_pre_target_OOS_hit_at_{count}",
        "model_probability_role": "near_tie_only",
        "fallback_policy": fallback,
    }
