from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def calibration_error(
    probabilities: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    bins: int = 10,
) -> float:
    """Expected calibration error for binary probability forecasts."""
    probs = np.asarray(list(probabilities), dtype=float)
    obs = np.asarray(list(outcomes), dtype=float)
    if probs.shape != obs.shape:
        raise ValueError("probabilities and outcomes must have the same length")
    if probs.size == 0:
        return 0.0
    if bins <= 0:
        raise ValueError("bins must be positive")
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if np.any(~np.isfinite(obs)) or np.any((obs < 0.0) | (obs > 1.0)):
        raise ValueError("outcomes must be finite binary-like values in [0, 1]")

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        error += weight * abs(float(np.mean(probs[mask])) - float(np.mean(obs[mask])))
    return float(error)


def reliability_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    blue_probs = []
    blue_hits = []
    for record in period_records:
        ranking = (record.get("fused_validation") or {}).get("blue_ranking") or {}
        if ranking.get("top1_probability") is not None and ranking.get("hit_at_1") is not None:
            blue_probs.append(float(ranking.get("top1_probability")))
            blue_hits.append(float(ranking.get("hit_at_1")))
    bins = []
    if blue_probs:
        probs = np.asarray(blue_probs)
        hits = np.asarray(blue_hits)
        edges = np.linspace(0.0, 1.0, 11)
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (probs >= left) & (probs < right if right < 1.0 else probs <= right)
            if np.any(mask):
                bins.append({
                    "left": float(left),
                    "right": float(right),
                    "count": int(mask.sum()),
                    "mean_probability": float(probs[mask].mean()),
                    "empirical_hit_rate": float(hits[mask].mean()),
                })
    return {
        "schema": "probability_reliability_report_v7",
        "blue_top1": {
            "periods": len(blue_probs),
            "bins": bins,
        },
    }


def uncertainty_calibration_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    report = reliability_report(period_records)
    return {
        "schema": "probability_calibration_api_v7",
        "reliability": report,
        "tracked_metrics": ["Brier", "LogLoss", "ECE", "Entropy", "Rank Stability", "Top-K Edge"],
    }
