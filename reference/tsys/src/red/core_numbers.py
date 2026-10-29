from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np


DEFAULT_CORE_WEIGHTS = {
    "fusion_probability": 0.35,
    "model_consensus": 0.30,
    "ticket_frequency": 0.20,
    "global_coverage_support": 0.15,
}


def _normalize(values: dict[int, float], number_range: range) -> dict[int, float]:
    cleaned = {
        int(number): max(0.0, float(values.get(int(number), 0.0) or 0.0))
        for number in number_range
    }
    maximum = max(cleaned.values(), default=0.0)
    if maximum <= 0:
        return {number: 0.0 for number in number_range}
    return {number: value / maximum for number, value in cleaned.items()}


def _probability_component(probabilities: np.ndarray, number_range: range) -> dict[int, float]:
    values = np.asarray(probabilities, dtype=float)
    return _normalize(
        {
            number: float(values[number - 1])
            for number in number_range
            if 0 <= number - 1 < len(values)
        },
        number_range,
    )


def _model_consensus_component(
    model_summaries: dict[str, Any],
    *,
    color: str,
    top_k: int,
    number_range: range,
) -> tuple[dict[int, float], dict[int, list[str]]]:
    support: dict[int, list[str]] = {number: [] for number in number_range}
    key = "top_red" if color == "red" else "top_blue"
    for model_name, summary in (model_summaries or {}).items():
        if not isinstance(summary, dict):
            continue
        values = [
            int(value)
            for value in list(summary.get(key, []) or [])[:top_k]
            if str(value).isdigit() and int(value) in number_range
        ]
        for number in values:
            support[int(number)].append(str(model_name))
    max_models = max(1, len(model_summaries or {}))
    component = {
        number: min(1.0, len(models) / float(max_models))
        for number, models in support.items()
    }
    return component, support


def _ticket_frequency_component(
    tickets: list[dict[str, Any]],
    *,
    color: str,
    number_range: range,
) -> tuple[dict[int, float], dict[int, int]]:
    counter: Counter[int] = Counter()
    for ticket in tickets or []:
        if color == "red":
            for value in ticket.get("red", []) or []:
                if str(value).isdigit() and int(value) in number_range:
                    counter[int(value)] += 1
        else:
            value = ticket.get("blue")
            if str(value).isdigit() and int(value) in number_range:
                counter[int(value)] += 1
    return _normalize(counter, number_range), {number: int(counter.get(number, 0)) for number in number_range}


def _global_support_component(
    method_tickets: dict[str, list[dict[str, Any]]],
    *,
    color: str,
    number_range: range,
) -> tuple[dict[int, float], dict[int, int]]:
    tickets = list((method_tickets or {}).get("global_coverage", []) or [])
    return _ticket_frequency_component(tickets, color=color, number_range=number_range)


def _weighted_score(
    number_range: range,
    weights: dict[str, float],
    components: dict[str, dict[int, float]],
) -> dict[int, float]:
    return {
        number: float(sum(float(weights.get(name, 0.0)) * component.get(number, 0.0) for name, component in components.items()))
        for number in number_range
    }


def _zone(number: int) -> int:
    if int(number) <= 11:
        return 1
    if int(number) <= 22:
        return 2
    return 3


def _structure_score(red: tuple[int, ...]) -> float:
    if len(red) != 6:
        return 0.0
    red_sum = sum(red)
    span = max(red) - min(red)
    odd = sum(number % 2 for number in red)
    zones = Counter(_zone(number) for number in red)
    sum_score = 1.0 - min(abs(red_sum - 102.0) / 55.0, 1.0)
    span_score = 1.0 - min(abs(span - 23.0) / 18.0, 1.0)
    odd_score = 1.0 - min(abs(odd - 3.0) / 3.0, 1.0)
    zone_score = 1.0 - min(sum(max(0, count - 3) for count in zones.values()) / 3.0, 1.0)
    return float(0.34 * sum_score + 0.28 * span_score + 0.20 * odd_score + 0.18 * zone_score)


def _concentrated_red_combo(
    red_scores: dict[int, float],
    candidate_pool_size: int = 20,
    *,
    sum_range: tuple[int, int] = (70, 140),
    span_range: tuple[int, int] = (16, 31),
    odd_range: tuple[int, int] = (1, 5),
) -> tuple[int, ...]:
    pool = sorted(red_scores, key=lambda number: (red_scores[number], -number), reverse=True)[: max(6, candidate_pool_size)]
    best_combo: tuple[int, ...] | None = None
    best_score = -1.0
    for combo in combinations(pool, 6):
        combo = tuple(sorted(int(value) for value in combo))
        red_sum = sum(combo)
        span = max(combo) - min(combo)
        odd = sum(number % 2 for number in combo)
        if not (int(sum_range[0]) <= red_sum <= int(sum_range[1])):
            continue
        if not (int(span_range[0]) <= span <= int(span_range[1])):
            continue
        if not (int(odd_range[0]) <= odd <= int(odd_range[1])):
            continue
        score = float(np.mean([red_scores[number] for number in combo])) + 0.18 * _structure_score(combo)
        if score > best_score:
            best_score = score
            best_combo = combo
    return best_combo or tuple(sorted(pool[:6]))


def _detail_rows(
    *,
    scores: dict[int, float],
    components: dict[str, dict[int, float]],
    consensus_support: dict[int, list[str]],
    ticket_counts: dict[int, int],
    global_counts: dict[int, int],
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(scores, key=lambda number: (scores[number], -number), reverse=True)[:limit]
    return [
        {
            "number": int(number),
            "score": float(scores[number]),
            "grade": "high" if scores[number] >= 0.80 else "medium" if scores[number] >= 0.60 else "reference",
            "model_consensus_count": int(len(consensus_support.get(number, []))),
            "model_consensus_models": list(consensus_support.get(number, [])),
            "ticket_frequency": int(ticket_counts.get(number, 0)),
            "global_coverage_frequency": int(global_counts.get(number, 0)),
            "components": {
                name: float(values.get(number, 0.0))
                for name, values in components.items()
            },
        }
        for number in ordered
    ]


def compute_core_numbers(
    *,
    probabilities: dict[str, np.ndarray],
    model_pipeline: dict[str, Any],
    final_tickets: list[dict[str, Any]],
    method_tickets: dict[str, list[dict[str, Any]]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate auditable core-number suggestions from existing model evidence."""
    core_config = (config or {}).get("core_numbers", {})
    enabled = bool(core_config.get("enabled", True))
    if not enabled:
        return {"enabled": False, "reason": "core_numbers_disabled"}
    weights = {**DEFAULT_CORE_WEIGHTS, **(core_config.get("weights", {}) or {})}
    total_weight = sum(max(0.0, float(value)) for value in weights.values())
    if total_weight <= 0:
        weights = DEFAULT_CORE_WEIGHTS.copy()
        total_weight = sum(weights.values())
    weights = {name: max(0.0, float(value)) / total_weight for name, value in weights.items()}
    model_summaries = model_pipeline.get("models", {}) if isinstance(model_pipeline, dict) else {}

    red_range = range(1, 34)
    blue_range = range(1, 17)
    red_consensus, red_support = _model_consensus_component(
        model_summaries,
        color="red",
        top_k=int(core_config.get("red_model_top_k", 6)),
        number_range=red_range,
    )
    blue_consensus, blue_support = _model_consensus_component(
        model_summaries,
        color="blue",
        top_k=int(core_config.get("blue_model_top_k", 3)),
        number_range=blue_range,
    )
    red_ticket, red_ticket_counts = _ticket_frequency_component(final_tickets, color="red", number_range=red_range)
    blue_ticket, blue_ticket_counts = _ticket_frequency_component(final_tickets, color="blue", number_range=blue_range)
    red_global, red_global_counts = _global_support_component(method_tickets, color="red", number_range=red_range)
    blue_global, blue_global_counts = _global_support_component(method_tickets, color="blue", number_range=blue_range)
    red_components = {
        "fusion_probability": _probability_component(probabilities.get("red", np.full(33, 1.0 / 33.0)), red_range),
        "model_consensus": red_consensus,
        "ticket_frequency": red_ticket,
        "global_coverage_support": red_global,
    }
    blue_components = {
        "fusion_probability": _probability_component(probabilities.get("blue", np.full(16, 1.0 / 16.0)), blue_range),
        "model_consensus": blue_consensus,
        "ticket_frequency": blue_ticket,
        "global_coverage_support": blue_global,
    }
    red_scores = _weighted_score(red_range, weights, red_components)
    blue_scores = _weighted_score(blue_range, weights, blue_components)
    red_limit = max(6, int(core_config.get("red_output_count", 6)))
    blue_limit = max(1, int(core_config.get("blue_output_count", 3)))
    red_rows = _detail_rows(
        scores=red_scores,
        components=red_components,
        consensus_support=red_support,
        ticket_counts=red_ticket_counts,
        global_counts=red_global_counts,
        limit=red_limit,
    )
    blue_rows = _detail_rows(
        scores=blue_scores,
        components=blue_components,
        consensus_support=blue_support,
        ticket_counts=blue_ticket_counts,
        global_counts=blue_global_counts,
        limit=blue_limit,
    )
    concentrated_red = _concentrated_red_combo(
        red_scores,
        candidate_pool_size=int(core_config.get("concentrated_candidate_pool_size", 20)),
        sum_range=tuple(core_config.get("concentrated_sum_range", [70, 140])),
        span_range=tuple(core_config.get("concentrated_span_range", [16, 31])),
        odd_range=tuple(core_config.get("concentrated_odd_range", [1, 5])),
    )
    top_blue = int(blue_rows[0]["number"]) if blue_rows else int(np.argmax(probabilities.get("blue", np.full(16, 1.0 / 16.0))) + 1)
    return {
        "enabled": True,
        "method": "fusion_probability_plus_model_consensus_plus_ticket_frequency_plus_global_support",
        "weights": weights,
        "red": red_rows,
        "blue": blue_rows,
        "core_red": [int(item["number"]) for item in red_rows[:6]],
        "core_blue": [int(item["number"]) for item in blue_rows[:3]],
        "consensus_red": [
            int(number)
            for number, models in red_support.items()
            if len(models) >= int(core_config.get("red_consensus_min_models", 5))
        ],
        "frequent_red": [
            int(number)
            for number, count in red_ticket_counts.items()
            if int(count) >= int(core_config.get("ticket_frequency_min_count", 2))
        ],
        "concentrated_ticket": {
            "red": [int(number) for number in concentrated_red],
            "blue": top_blue,
            "structure_score": _structure_score(concentrated_red),
            "usage": "high_concentration_reference_only",
        },
        "usage_suggestions": {
            "conservative": "直接查看最终5注覆盖组合。",
            "aggressive": "关注核心红球Top-6与核心蓝球Top-1形成的集中型参考单注。",
            "balanced": "将核心红球中高分号码作为胆码候选，再结合预算内胆拖方案。",
        },
        "risk_note": "核心号码是历史数据实验摘要，不保证命中，也不构成投注建议。",
    }
