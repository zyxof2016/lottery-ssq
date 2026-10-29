from __future__ import annotations

from math import comb
from typing import Iterable

import numpy as np
from scipy.stats import t, ttest_rel


BLUE_SINGLE_TICKET_BASELINE = 1.0 / 16.0


def theoretical_ticket_baseline(ticket_count: int) -> dict[str, float | dict[str, float]]:
    """Exact random-draw baseline for a 6-number ticket and diversified blues."""
    denominator = comb(33, 6)
    red_distribution = {
        str(hits): float(comb(6, hits) * comb(27, 6 - hits) / denominator)
        for hits in range(7)
    }
    unique_blue_count = min(max(int(ticket_count), 0), 16)
    return {
        "single_ticket_expected_red_hits": float(36.0 / 33.0),
        "single_ticket_red_hit_distribution": red_distribution,
        "single_ticket_red_ge_3_probability": float(
            sum(red_distribution[str(hits)] for hits in range(3, 7))
        ),
        "single_ticket_red_ge_4_probability": float(
            sum(red_distribution[str(hits)] for hits in range(4, 7))
        ),
        "single_ticket_blue_hit_probability": BLUE_SINGLE_TICKET_BASELINE,
        "diversified_group_ticket_count": unique_blue_count,
        "diversified_group_any_blue_hit_probability": float(unique_blue_count / 16.0),
    }


def evaluate_tickets(tickets: list[dict], actual_red: set[int], actual_blue: int) -> dict[str, float]:
    if not tickets:
        raise ValueError("待评估组合不能为空")
    red_hits = np.array([len(set(ticket["red"]) & actual_red) for ticket in tickets], dtype=float)
    blue_hits = np.array([int(ticket["blue"] == actual_blue) for ticket in tickets], dtype=float)
    return {
        "average_red_hits": float(red_hits.mean()),
        "red_ge3_ratio": float(np.mean(red_hits >= 3)),
        "red_ge4_ratio": float(np.mean(red_hits >= 4)),
        "any_red_ge3": float(np.max(red_hits) >= 3),
        "any_red_ge4": float(np.max(red_hits) >= 4),
        "blue_hit_ratio": float(blue_hits.mean()),
        "any_blue_hit": float(blue_hits.max()),
        "blue_unique_count": float(len({int(ticket["blue"]) for ticket in tickets})),
        "best_red_hits": float(red_hits.max()),
    }


def validate_tickets(tickets: list[dict], actual_red: set[int], actual_blue: int) -> list[dict]:
    validated: list[dict] = []
    for ticket in tickets:
        red = [int(number) for number in ticket["red"]]
        matched_red = sorted(set(red) & actual_red)
        blue_hit = bool(int(ticket["blue"]) == int(actual_blue))
        validated.append({
            **ticket,
            "red": red,
            "red_hits": int(len(matched_red)),
            "matched_red": matched_red,
            "blue_hit": blue_hit,
            "matched_blue": int(actual_blue) if blue_hit else None,
        })
    return validated


def blue_ranking_metrics(probability: Iterable[float], actual_blue: int, *, top_k: int = 5) -> dict[str, float | int]:
    """Evaluate a 16-way blue probability distribution against Top-K goals."""
    values = np.asarray(list(probability), dtype=float).reshape(-1)
    if values.size != 16 or not np.all(np.isfinite(values)):
        values = np.full(16, 1.0 / 16.0, dtype=float)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    values = values / total if total > 0 else np.full(16, 1.0 / 16.0, dtype=float)
    actual = int(actual_blue)
    if not 1 <= actual <= 16:
        raise ValueError("actual_blue 必须在 1-16 范围内")
    ordered = [int(index) + 1 for index in np.argsort(values)[::-1]]
    rank = int(ordered.index(actual) + 1)
    top1 = int(ordered[0])
    top_k = max(1, min(int(top_k), 16))
    dcg = 1.0 / np.log2(rank + 1.0) if rank <= top_k else 0.0
    return {
        "hit_at_1": float(rank <= 1),
        "hit_at_3": float(rank <= 3),
        "hit_at_5": float(rank <= 5),
        "mrr": float(1.0 / rank),
        "ndcg_at_5": float(dcg),
        "mean_rank": float(rank),
        "rank": rank,
        "actual_probability": float(values[actual - 1]),
        "top1": top1,
        "top1_probability": float(values[top1 - 1]),
    }


def expected_calibration_error(
    probabilities: Iterable[float],
    outcomes: Iterable[bool],
    *,
    bins: int = 10,
) -> dict[str, float | int]:
    """Expected calibration error for binary probability/outcome rows."""
    probs = np.asarray(list(probabilities), dtype=float).reshape(-1)
    hits = np.asarray(list(outcomes), dtype=float).reshape(-1)
    if probs.shape != hits.shape:
        raise ValueError("probabilities 与 outcomes 长度必须一致")
    mask = np.isfinite(probs) & np.isfinite(hits)
    probs = np.clip(probs[mask], 0.0, 1.0)
    hits = np.clip(hits[mask], 0.0, 1.0)
    if probs.size == 0:
        return {"samples": 0, "bins": int(bins), "ece": 0.0, "mce": 0.0}
    bins = max(2, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        left = edges[index]
        right = edges[index + 1]
        if index == bins - 1:
            bucket = (probs >= left) & (probs <= right)
        else:
            bucket = (probs >= left) & (probs < right)
        if not np.any(bucket):
            continue
        confidence = float(np.mean(probs[bucket]))
        accuracy = float(np.mean(hits[bucket]))
        gap = abs(confidence - accuracy)
        ece += float(np.mean(bucket)) * gap
        mce = max(mce, gap)
    return {
        "samples": int(probs.size),
        "bins": bins,
        "ece": float(ece),
        "mce": float(mce),
    }


def _sign_flip_permutation_p_value(
    difference: np.ndarray,
    *,
    repetitions: int,
    random_seed: int,
) -> float | None:
    if len(difference) < 2:
        return None
    observed = abs(float(difference.mean()))
    if np.allclose(difference, 0.0):
        return 1.0
    rng = np.random.default_rng(int(random_seed))
    extreme = 0
    remaining = int(repetitions)
    batch_size = 5000
    while remaining > 0:
        size = min(batch_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(difference)))
        permuted = np.abs((signs * difference).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        remaining -= size
    return float((extreme + 1) / (int(repetitions) + 1))


def paired_significance(
    model_values: Iterable[float],
    random_values: Iterable[float],
    *,
    confidence_level: float = 0.95,
    permutation_repetitions: int = 20000,
    random_seed: int = 2026,
) -> dict[str, float | bool | None | str | list[float | None]]:
    model = np.asarray(list(model_values), dtype=float)
    random = np.asarray(list(random_values), dtype=float)
    if model.shape != random.shape:
        raise ValueError("模型与随机基线必须按同一期一一对应")
    confidence_level = float(confidence_level)
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 (0.5, 1.0) 范围内")
    if int(permutation_repetitions) < 1000:
        raise ValueError("permutation_repetitions 必须至少为1000")
    difference = model - random
    sample_size = int(len(difference))
    mean_difference = float(difference.mean()) if sample_size else 0.0
    standard_deviation = float(difference.std(ddof=1)) if sample_size > 1 else None
    if sample_size > 1 and standard_deviation is not None:
        standard_error = standard_deviation / float(np.sqrt(sample_size))
        margin = float(t.ppf(0.5 + confidence_level / 2.0, sample_size - 1)) * standard_error
        confidence_interval = [mean_difference - margin, mean_difference + margin]
        effect_size = mean_difference / standard_deviation if standard_deviation > 0 else None
    else:
        confidence_interval = [None, None]
        effect_size = None
    if len(difference) < 2 or np.allclose(difference, difference[0]):
        paired_t_p_value = None
    else:
        paired_t_p_value = float(ttest_rel(model, random).pvalue)
    permutation_p_value = _sign_flip_permutation_p_value(
        difference,
        repetitions=int(permutation_repetitions),
        random_seed=int(random_seed),
    )
    return {
        "sample_size": sample_size,
        "mean_difference": mean_difference,
        "difference_standard_deviation": standard_deviation,
        "confidence_level": confidence_level,
        "confidence_interval": confidence_interval,
        "confidence_interval_95": confidence_interval,
        "paired_effect_size": effect_size,
        "test_method": "paired_sign_flip_permutation",
        "permutation_repetitions": int(permutation_repetitions),
        "permutation_p_value": permutation_p_value,
        "paired_t_p_value": paired_t_p_value,
        "p_value": permutation_p_value,
        "significant_at_0_05": bool(permutation_p_value is not None and permutation_p_value < 0.05),
    }


def apply_holm_correction(results: dict[str, dict], significance_key: str = "significance") -> None:
    available = [
        (name, float(detail[significance_key]["p_value"]))
        for name, detail in results.items()
        if detail.get(significance_key, {}).get("p_value") is not None
    ]
    ordered = sorted(available, key=lambda item: item[1])
    running_max = 0.0
    count = len(ordered)
    adjusted: dict[str, float] = {}
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    for name, detail in results.items():
        significance = detail.setdefault(significance_key, {})
        adjusted_value = adjusted.get(name)
        significance["holm_adjusted_p_value"] = adjusted_value
        significance["significant_after_holm_0_05"] = bool(
            adjusted_value is not None
            and adjusted_value < 0.05
            and float(significance.get("mean_difference", 0.0)) > 0
        )


def window_volatility(values: Iterable[float], windows: Iterable[int] = (3, 5, 10)) -> dict[str, float | None]:
    series = np.asarray(list(values), dtype=float)
    result: dict[str, float | None] = {}
    for window in windows:
        size = int(window)
        if len(series) < size:
            result[str(size)] = None
            continue
        rolling = np.convolve(series, np.ones(size) / size, mode="valid")
        result[str(size)] = float(np.std(rolling, ddof=0))
    return result
