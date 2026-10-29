from __future__ import annotations

from typing import Any

import numpy as np

from src.blue.expert_groups import DEFAULT_BLUE_EXPERT_GROUPS
from src.red.expert_groups import DEFAULT_RED_EXPERT_GROUPS


def _normalize(values: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != int(size) or not np.all(np.isfinite(array)):
        return None
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return None
    return array / total


def _rank_vector(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    return ranks


def _corr(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(right) < 3 or len(left) != len(right):
        return None
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return None
    return value


def _actual(record: dict[str, Any]) -> tuple[set[int], int] | None:
    payload = record.get("actual") if isinstance(record, dict) else None
    if not isinstance(payload, dict):
        return None
    try:
        red = {int(value) for value in payload.get("red", [])}
        blue = int(payload.get("blue"))
    except (TypeError, ValueError):
        return None
    if len(red) != 6 or not 1 <= blue <= 16:
        return None
    return red, blue


def _model_series(records: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    for record in records or []:
        actual = _actual(record)
        payload = record.get("model_oof_probabilities") if isinstance(record, dict) else None
        if actual is None or not isinstance(payload, dict):
            continue
        actual_red, actual_blue = actual
        for model_name, probabilities in payload.items():
            if not isinstance(probabilities, dict):
                continue
            red_probability = _normalize(probabilities.get("red"), 33)
            blue_probability = _normalize(probabilities.get("blue"), 16)
            if red_probability is None or blue_probability is None:
                continue
            vector = np.concatenate([red_probability, blue_probability])
            rank = np.concatenate([_rank_vector(red_probability), _rank_vector(blue_probability)])
            red_top6 = {int(index + 1) for index in np.argsort(red_probability)[-6:]}
            blue_top1 = int(np.argmax(blue_probability) + 1)
            red_error = (6.0 - float(len(red_top6 & actual_red))) / 6.0
            blue_error = 0.0 if blue_top1 == actual_blue else 1.0
            bucket = series.setdefault(str(model_name), {"probability": [], "rank": [], "error": []})
            bucket["probability"].extend(float(value) for value in vector)
            bucket["rank"].extend(float(value) for value in rank)
            bucket["error"].append(0.65 * red_error + 0.35 * blue_error)
    return series


def _pairwise(series: dict[str, dict[str, list[float]]], metric: str) -> dict[str, dict[str, float]]:
    models = sorted(series)
    result: dict[str, dict[str, float]] = {}
    for left in models:
        row: dict[str, float] = {}
        for right in models:
            value = 1.0 if left == right else _corr(series[left].get(metric, []), series[right].get(metric, []))
            if value is not None:
                row[right] = float(value)
        result[left] = row
    return result


def _union_find_clusters(
    models: list[str],
    probability: dict[str, dict[str, float]],
    rank: dict[str, dict[str, float]],
    error: dict[str, dict[str, float]],
    *,
    threshold: float,
) -> list[list[str]]:
    parent = {model: model for model in models}

    def find(model: str) -> str:
        while parent[model] != model:
            parent[model] = parent[parent[model]]
            model = parent[model]
        return model

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, left in enumerate(models):
        for right in models[index + 1 :]:
            score = max(
                abs(float(probability.get(left, {}).get(right, 0.0))),
                abs(float(rank.get(left, {}).get(right, 0.0))),
                abs(float(error.get(left, {}).get(right, 0.0))),
            )
            if score >= threshold:
                union(left, right)
    grouped: dict[str, list[str]] = {}
    for model in models:
        grouped.setdefault(find(model), []).append(model)
    return [sorted(values) for values in grouped.values()]


def expert_decorrelation_report(backtest: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    records = backtest.get("period_records", []) if isinstance(backtest, dict) else []
    scorecard = backtest.get("model_scorecard", {}) if isinstance(backtest, dict) else {}
    known_models = sorted((scorecard.get("models") or {}).keys()) if isinstance(scorecard, dict) else []
    series = _model_series(records)
    models = sorted(series) or known_models
    groups: dict[str, list[str]] = {}
    red_groups: dict[str, list[str]] = {}
    blue_groups: dict[str, list[str]] = {}
    for name in models:
        red_group = DEFAULT_RED_EXPERT_GROUPS.get(name, "red_other")
        blue_group = DEFAULT_BLUE_EXPERT_GROUPS.get(name, "blue_other")
        red_groups.setdefault(red_group, []).append(name)
        blue_groups.setdefault(blue_group, []).append(name)
        groups.setdefault(f"{red_group}|{blue_group}", []).append(name)
    probability = _pairwise(series, "probability") if series else {}
    rank = _pairwise(series, "rank") if series else {}
    error = _pairwise(series, "error") if series else {}
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    governance = (v7.get("governance") or {}) if isinstance(v7, dict) else {}
    threshold = float(governance.get("decorrelation_threshold", 0.85))
    clusters = (
        _union_find_clusters(models, probability, rank, error, threshold=threshold)
        if series
        else [sorted(values) for values in groups.values()]
    )
    group_cap = governance.get("group_cap_by_edge_state") or {
        "STRONG_EDGE": 0.70,
        "WEAK_EDGE": 0.55,
        "NO_EDGE": 0.42,
        "RECALL_GUARD": 0.35,
    }
    return {
        "schema": "expert_decorrelation_v7",
        "executed": bool(series),
        "periods": len(records),
        "models": models,
        "groups": groups,
        "red_groups": red_groups,
        "blue_groups": blue_groups,
        "default_group_maps": {
            "red": DEFAULT_RED_EXPERT_GROUPS,
            "blue": DEFAULT_BLUE_EXPERT_GROUPS,
        },
        "probability_correlation": probability,
        "rank_correlation": rank,
        "error_correlation": error,
        "clusters": clusters,
        "cluster_threshold": threshold,
        "group_cap": group_cap,
        "effective_evidence": {
            "raw_expert_count": len(models),
            "decorrelated_expert_count": len(clusters),
            "evidence_discount": float(len(clusters) / len(models)) if models else 0.0,
            "rule": "Highly correlated experts count as one evidence cluster before group caps are applied.",
        },
    }
