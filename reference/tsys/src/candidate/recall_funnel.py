from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _normalize(values: Iterable[float], expected_size: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size != int(expected_size) or not np.all(np.isfinite(array)):
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    return array / total


def _top_numbers(probability: Iterable[float], expected_size: int, top_k: int) -> list[int]:
    values = _normalize(probability, expected_size)
    limit = max(1, min(int(top_k), int(expected_size)))
    return [int(index) + 1 for index in np.argsort(values)[::-1][:limit]]


def _model_red_top_sets(
    model_probabilities: dict[str, dict[str, Any]] | None,
    *,
    top6: int = 6,
    top10: int = 10,
) -> tuple[set[int], set[int], dict[int, dict[str, list[dict[str, Any]]]]]:
    top6_set: set[int] = set()
    top10_set: set[int] = set()
    support: dict[int, dict[str, list[dict[str, Any]]]] = {
        number: {"top6_models": [], "top10_models": []}
        for number in range(1, 34)
    }
    for model_name, prediction in (model_probabilities or {}).items():
        probability = _normalize((prediction or {}).get("red", []), 33)
        if bool(np.allclose(probability, probability[0], rtol=1e-10, atol=1e-12)):
            continue
        model_top6 = _top_numbers(probability, 33, top6)
        model_top10 = _top_numbers(probability, 33, top10)
        top6_set.update(model_top6)
        top10_set.update(model_top10)
        for rank, number in enumerate(model_top6, start=1):
            support[int(number)]["top6_models"].append({
                "model": str(model_name),
                "rank": int(rank),
                "probability": float(probability[int(number) - 1]),
            })
        for rank, number in enumerate(model_top10, start=1):
            support[int(number)]["top10_models"].append({
                "model": str(model_name),
                "rank": int(rank),
                "probability": float(probability[int(number) - 1]),
            })
    return top6_set, top10_set, support


def _candidate_union(pool: list[dict[str, Any]], *, color: str) -> set[int]:
    if color == "red":
        return {
            int(number)
            for ticket in pool
            for number in ticket.get("red", []) or []
            if 1 <= int(number) <= 33
        }
    return {
        int(ticket.get("blue", 0) or 0)
        for ticket in pool
        if 1 <= int(ticket.get("blue", 0) or 0) <= 16
    }


def _method_red_counts(selected_by_method: dict[str, list[dict[str, Any]]]) -> dict[int, int]:
    counts: dict[int, int] = {number: 0 for number in range(1, 34)}
    seen: set[tuple[str, int]] = set()
    for method, tickets in (selected_by_method or {}).items():
        for ticket in tickets or []:
            for number in ticket.get("red", []) or []:
                value = int(number)
                if not 1 <= value <= 33:
                    continue
                key = (str(method), value)
                if key in seen:
                    continue
                counts[value] += 1
                seen.add(key)
    return counts


def _ranking_candidate_set(
    red_probability: Iterable[float],
    selected_by_method: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 18,
    rescue_numbers: Iterable[int] | None = None,
) -> tuple[set[int], dict[str, Any]]:
    values = _normalize(red_probability, 33)
    order = _top_numbers(values, 33, 33)
    ranks = {number: index for index, number in enumerate(order, start=1)}
    top12 = set(order[:12])
    top16 = set(order[:16])
    top18 = set(order[:18])
    method_counts = _method_red_counts(selected_by_method)
    method_consensus = {
        number for number, count in method_counts.items()
        if int(count) >= 2
    }
    independent_rescue = {
        int(number)
        for number in (rescue_numbers or [])
        if 1 <= int(number) <= 33
    }
    raw = top18 | method_consensus | independent_rescue
    mandatory = list(order[:18])
    ranked_extras = sorted(
        raw - set(mandatory),
        key=lambda number: (
            number in independent_rescue,
            number in top12,
            number in top16,
            number in top18,
            int(method_counts.get(number, 0)),
            float(values[number - 1]),
            -int(ranks.get(number, 99)),
        ),
        reverse=True,
    )
    selected_order = list(dict.fromkeys(mandatory + ranked_extras))
    selected = set(selected_order[: max(18, min(int(limit), 33))])
    return selected, {
        "schema": "red_candidate_set_v8",
        "definition": "Fusion Top18 is mandatory; remaining slots are method consensus and independent rescue. Raw Monte Carlo pool union is diagnostic only.",
        "limit": int(limit),
        "top12": sorted(top12),
        "top16": sorted(top16),
        "top18": sorted(top18),
        "method_consensus": sorted(method_consensus),
        "independent_model_rescue": sorted(independent_rescue),
        "selected": sorted(selected),
        "method_support_counts": {
            str(number): int(count)
            for number, count in method_counts.items()
            if int(count) > 0
        },
    }


def _final_union(tickets: list[dict[str, Any]], *, color: str) -> set[int]:
    return _candidate_union(tickets, color=color)


def evaluate_candidate_recall(
    *,
    fused_probability: dict[str, Iterable[float]],
    model_probabilities: dict[str, dict[str, Any]] | None = None,
    recall_protection: dict[str, Any] | None = None,
    candidate_pool: list[dict[str, Any]],
    selected_by_method: dict[str, list[dict[str, Any]]],
    final_tickets: list[dict[str, Any]],
    portfolio_tickets: list[dict[str, Any]] | None = None,
    red_candidate_limit: int = 22,
    actual_red: set[int],
    actual_blue: int,
) -> dict[str, Any]:
    """Measure where the real draw was lost: model Top-K, pool, method layer or final layer.

    This is intentionally descriptive rather than predictive.  It gives the
    V7 pipeline a common funnel for deciding whether a miss belongs to the
    probability model, candidate generation, optimizer selection, or final
    compression/rebalance.
    """
    actual_red = {int(number) for number in actual_red}
    actual_blue = int(actual_blue)
    red_top6 = set(_top_numbers(fused_probability.get("red", []), 33, 6))
    red_top10 = set(_top_numbers(fused_probability.get("red", []), 33, 10))
    red_top12 = set(_top_numbers(fused_probability.get("red", []), 33, 12))
    red_top16 = set(_top_numbers(fused_probability.get("red", []), 33, 16))
    red_top18 = set(_top_numbers(fused_probability.get("red", []), 33, 18))
    independent_top6, independent_top10, independent_support = _model_red_top_sets(
        model_probabilities,
        top6=6,
        top10=10,
    )
    blue_top1 = set(_top_numbers(fused_probability.get("blue", []), 16, 1))
    blue_top3 = set(_top_numbers(fused_probability.get("blue", []), 16, 3))
    blue_top5 = set(_top_numbers(fused_probability.get("blue", []), 16, 5))

    raw_pool_red = _candidate_union(candidate_pool, color="red")
    pool_red, red_candidate_set = _ranking_candidate_set(
        fused_probability.get("red", []),
        selected_by_method,
        limit=red_candidate_limit,
        rescue_numbers=(recall_protection or {}).get("protected_numbers", []),
    )
    pool_blue = _candidate_union(candidate_pool, color="blue")
    method_red = _candidate_union(
        [ticket for tickets in selected_by_method.values() for ticket in tickets],
        color="red",
    )
    method_blue = _candidate_union(
        [ticket for tickets in selected_by_method.values() for ticket in tickets],
        color="blue",
    )
    portfolio_red = _final_union(portfolio_tickets or final_tickets, color="red")
    portfolio_blue = _final_union(portfolio_tickets or final_tickets, color="blue")
    final_red = _final_union(final_tickets, color="red")
    final_blue = _final_union(final_tickets, color="blue")
    portfolio_available = portfolio_tickets is not None

    def red_hit_count(values: set[int]) -> int:
        return int(len(actual_red & values))

    def red_missed(values: set[int]) -> list[int]:
        return sorted(actual_red - values)

    candidate_true_found = actual_red & pool_red
    method_true_retained = candidate_true_found & method_red
    portfolio_true_retained = candidate_true_found & portfolio_red
    final_true_retained = candidate_true_found & final_red
    true_found_count = len(candidate_true_found)
    candidate_to_method_loss = int(true_found_count - len(method_true_retained))
    candidate_to_method_retention = (
        float(len(method_true_retained) / true_found_count)
        if true_found_count
        else None
    )
    true_survival = {
        "schema": "candidate_to_final_true_survival_v8",
        "candidate_found_numbers": sorted(candidate_true_found),
        "method_retained_numbers": sorted(method_true_retained),
        "portfolio_retained_numbers": sorted(portfolio_true_retained),
        "final_retained_numbers": sorted(final_true_retained),
        "final_lost_numbers": sorted(candidate_true_found - final_true_retained),
        "candidate_found_count": int(true_found_count),
        "method_retained_count": int(len(method_true_retained)),
        "portfolio_retained_count": int(len(portfolio_true_retained)),
        "final_retained_count": int(len(final_true_retained)),
        "candidate_to_method_loss": candidate_to_method_loss,
        "candidate_to_method_retention": candidate_to_method_retention,
        "candidate_to_final_loss": int(true_found_count - len(final_true_retained)),
        "candidate_to_final_retention": (
            float(len(final_true_retained) / true_found_count)
            if true_found_count
            else None
        ),
        "definition": "只统计Candidate Set已经找到的真实红球，观察这些真实红球是否继续存活到Final 5。",
    }

    def survival_stage(number: int) -> str:
        value = int(number)
        if value in final_red:
            return "final"
        if value in portfolio_red:
            return "portfolio"
        if value in method_red:
            return "method"
        if value in pool_red:
            return "candidate"
        if value in red_top18:
            return "fusion_top18"
        if value in independent_top6:
            return "independent_model_top6"
        if value in independent_top10:
            return "independent_model_top10"
        return "not_found"

    topk_survival = [
        {
            "number": int(number),
            "in_independent_model_top6": int(number) in independent_top6,
            "in_independent_model_top10": int(number) in independent_top10,
            "in_fusion_top6": int(number) in red_top6,
            "in_fusion_top10": int(number) in red_top10,
            "in_fusion_top18": int(number) in red_top18,
            "in_candidate": int(number) in pool_red,
            "in_method": int(number) in method_red,
            "in_portfolio": int(number) in portfolio_red,
            "in_final": int(number) in final_red,
            "last_survival_stage": survival_stage(int(number)),
            "independent_model_support": independent_support.get(int(number), {}),
        }
        for number in sorted(actual_red)
    ]

    return {
        "schema": "candidate_recall_funnel_v7",
        "red": {
            "actual": sorted(actual_red),
            "independent_model_top6_hits": red_hit_count(independent_top6),
            "independent_model_top10_hits": red_hit_count(independent_top10),
            "independent_model_top6_numbers": sorted(independent_top6),
            "independent_model_top10_numbers": sorted(independent_top10),
            "model_top6_hits": red_hit_count(red_top6),
            "model_top10_hits": red_hit_count(red_top10),
            "model_top12_hits": red_hit_count(red_top12),
            "model_top16_hits": red_hit_count(red_top16),
            "model_top18_hits": red_hit_count(red_top18),
            "candidate_pool_hits": red_hit_count(pool_red),
            "candidate_set_hits": red_hit_count(pool_red),
            "raw_pool_union_hits": red_hit_count(raw_pool_red),
            "candidate_set": red_candidate_set,
            "method_layer_hits": red_hit_count(method_red),
            "portfolio_layer_hits": red_hit_count(portfolio_red),
        "final_layer_hits": red_hit_count(final_red),
        "true_survival": true_survival,
        "final5_drop_regret": int(len(candidate_true_found - final_true_retained)),
        "final5_failure_attribution": (
            "final5_combination_failure"
            if candidate_true_found - final_true_retained
            else "retained"
            if candidate_true_found
            else "model_or_candidate_miss"
        ),
        "final5_drop_regret_detail": {
            "schema": "final5_drop_regret_v11_7",
            "candidate_hit": int(true_found_count),
            "final5_hit": int(len(final_true_retained)),
            "drop_regret": int(len(candidate_true_found - final_true_retained)),
            "dropped_numbers": sorted(candidate_true_found - final_true_retained),
            "attribution": (
                "Final5组合优化失败"
                if candidate_true_found - final_true_retained
                else "已保留"
                if candidate_true_found
                else "模型或候选未找到"
            ),
            "definition": "Candidate已经包含的真实红球如果没有进入最终5注，记为Final5组合优化失败，不再记成模型没有预测到。",
        },
            "candidate_to_method_true_loss": true_survival["candidate_to_method_loss"],
            "candidate_to_method_true_retention": true_survival["candidate_to_method_retention"],
            "candidate_to_final_true_loss": true_survival["candidate_to_final_loss"],
            "candidate_to_final_true_retention": true_survival["candidate_to_final_retention"],
            "topk_survival": topk_survival,
            "missed_at_model_top18": red_missed(red_top18),
            "missed_at_candidate_pool": red_missed(pool_red),
            "missed_at_method_layer": red_missed(method_red),
            "missed_at_portfolio_layer": red_missed(portfolio_red),
            "missed_at_final_layer": red_missed(final_red),
            "portfolio_stage_available": bool(portfolio_available),
            "diagnosis": (
                "model_probability"
                if red_hit_count(red_top18) <= 2
                else "candidate_generation"
                if red_hit_count(pool_red) < red_hit_count(red_top18)
                else "optimizer_method_selection"
                if red_hit_count(method_red) < red_hit_count(pool_red)
                else "portfolio_optimization"
                if red_hit_count(portfolio_red) < red_hit_count(method_red)
                else "final_portfolio_compression"
                if red_hit_count(final_red) < red_hit_count(portfolio_red)
                else "retained"
            ),
        },
        "blue": {
            "actual": actual_blue,
            "model_hit_at_1": bool(actual_blue in blue_top1),
            "model_hit_at_3": bool(actual_blue in blue_top3),
            "model_hit_at_5": bool(actual_blue in blue_top5),
            "candidate_pool_hit": bool(actual_blue in pool_blue),
            "method_layer_hit": bool(actual_blue in method_blue),
            "portfolio_layer_hit": bool(actual_blue in portfolio_blue),
            "final_layer_hit": bool(actual_blue in final_blue),
            "portfolio_stage_available": bool(portfolio_available),
            "diagnosis": (
                "model_top5"
                if actual_blue not in blue_top5
                else "candidate_generation"
                if actual_blue not in pool_blue
                else "optimizer_method_selection"
                if actual_blue not in method_blue
                else "blue_portfolio_optimization"
                if actual_blue not in portfolio_blue
                else "final_blue_compression"
                if actual_blue not in final_blue
                else "retained"
            ),
        },
    }


def summarize_candidate_recall_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    funnels = [
        record.get("candidate_recall_funnel", {})
        for record in records
        if isinstance(record.get("candidate_recall_funnel"), dict)
    ]
    if not funnels:
        return {"schema": "candidate_recall_funnel_summary_v7", "periods": 0}

    def mean_red(key: str) -> float:
        return float(np.mean([float((item.get("red") or {}).get(key, 0.0)) for item in funnels]))

    def mean_red_optional(key: str) -> float | None:
        values = [
            (item.get("red") or {}).get(key)
            for item in funnels
            if (item.get("red") or {}).get(key) is not None
        ]
        return float(np.mean([float(value) for value in values])) if values else None

    red_diagnoses: dict[str, int] = {}
    blue_diagnoses: dict[str, int] = {}
    for item in funnels:
        red_diag = str((item.get("red") or {}).get("diagnosis", "unknown"))
        blue_diag = str((item.get("blue") or {}).get("diagnosis", "unknown"))
        red_diagnoses[red_diag] = red_diagnoses.get(red_diag, 0) + 1
        blue_diagnoses[blue_diag] = blue_diagnoses.get(blue_diag, 0) + 1

    return {
        "schema": "candidate_recall_funnel_summary_v7",
        "periods": len(funnels),
        "red": {
            "avg_independent_model_top6_hits": mean_red("independent_model_top6_hits"),
            "avg_independent_model_top10_hits": mean_red("independent_model_top10_hits"),
            "avg_model_top6_hits": mean_red("model_top6_hits"),
            "avg_model_top10_hits": mean_red("model_top10_hits"),
            "avg_model_top12_hits": mean_red("model_top12_hits"),
            "avg_model_top16_hits": mean_red("model_top16_hits"),
            "avg_model_top18_hits": mean_red("model_top18_hits"),
            "avg_candidate_pool_hits": mean_red("candidate_pool_hits"),
            "avg_candidate_set_hits": mean_red("candidate_set_hits"),
            "avg_raw_pool_union_hits": mean_red("raw_pool_union_hits"),
            "avg_method_layer_hits": mean_red("method_layer_hits"),
            "avg_portfolio_layer_hits": mean_red("portfolio_layer_hits"),
            "avg_final_layer_hits": mean_red("final_layer_hits"),
            "avg_candidate_to_method_true_loss": mean_red("candidate_to_method_true_loss"),
            "avg_candidate_to_method_true_retention": mean_red_optional("candidate_to_method_true_retention"),
            "avg_candidate_to_final_true_loss": mean_red("candidate_to_final_true_loss"),
            "avg_candidate_to_final_true_retention": mean_red_optional("candidate_to_final_true_retention"),
            "avg_final5_drop_regret": mean_red_optional("final5_drop_regret"),
            "recall_rates": {
                "independent_model_top6": mean_red("independent_model_top6_hits") / 6.0,
                "independent_model_top10": mean_red("independent_model_top10_hits") / 6.0,
                "fusion_top6": mean_red("model_top6_hits") / 6.0,
                "fusion_top10": mean_red("model_top10_hits") / 6.0,
                "fusion_top18": mean_red("model_top18_hits") / 6.0,
                "candidate": mean_red("candidate_set_hits") / 6.0,
                "method": mean_red("method_layer_hits") / 6.0,
                "portfolio": mean_red("portfolio_layer_hits") / 6.0,
                "final": mean_red("final_layer_hits") / 6.0,
            },
            "stage_retention": {
                "candidate_to_method": mean_red_optional("candidate_to_method_true_retention"),
                "candidate_to_final": mean_red_optional("candidate_to_final_true_retention"),
                "method_to_final_hit_delta": (mean_red("final_layer_hits") - mean_red("method_layer_hits")) / 6.0,
            },
            "diagnosis_counts": red_diagnoses,
        },
        "blue": {
            "hit_at_1": float(np.mean([bool((item.get("blue") or {}).get("model_hit_at_1")) for item in funnels])),
            "hit_at_3": float(np.mean([bool((item.get("blue") or {}).get("model_hit_at_3")) for item in funnels])),
            "hit_at_5": float(np.mean([bool((item.get("blue") or {}).get("model_hit_at_5")) for item in funnels])),
            "candidate_pool_hit": float(np.mean([bool((item.get("blue") or {}).get("candidate_pool_hit")) for item in funnels])),
            "method_layer_hit": float(np.mean([bool((item.get("blue") or {}).get("method_layer_hit")) for item in funnels])),
            "portfolio_layer_hit": float(np.mean([bool((item.get("blue") or {}).get("portfolio_layer_hit")) for item in funnels])),
            "final_layer_hit": float(np.mean([bool((item.get("blue") or {}).get("final_layer_hit")) for item in funnels])),
            "diagnosis_counts": blue_diagnoses,
        },
    }
