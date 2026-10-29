from __future__ import annotations

from itertools import combinations
from statistics import mean
from typing import Any


DEFAULT_RED_METHOD_ABLATION_METHODS = (
    ("A", "red_ge3_objective", "红球3+目标优化"),
    ("B", "greedy", "贪心覆盖"),
    ("C", "pareto", "Pareto多目标"),
    ("D", "historical_hit", "历史回放命中优化"),
    ("E", "global_coverage", "整数规划全局覆盖"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _red_union(tickets: list[dict[str, Any]]) -> set[int]:
    numbers: set[int] = set()
    for ticket in tickets or []:
        numbers.update(_safe_int(number) for number in ticket.get("red", []) or [])
    return {number for number in numbers if 1 <= number <= 33}


def _mean_pairwise_jaccard(tickets: list[dict[str, Any]]) -> float:
    sets = [
        {number for number in (_safe_int(value) for value in ticket.get("red", []) or []) if 1 <= number <= 33}
        for ticket in tickets or []
    ]
    values: list[float] = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 0.0)
    return float(mean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_red_method_ablation(
    period_records: list[dict[str, Any]],
    *,
    methods: tuple[tuple[str, str, str], ...] = DEFAULT_RED_METHOD_ABLATION_METHODS,
) -> dict[str, Any]:
    """Compare red portfolio optimizers on each period's identical candidate pool.

    Each method is evaluated against the candidate recall already measured for
    that strict walk-forward period. This answers whether a method preserved the
    true red numbers that the shared Candidate Pool had already surfaced.
    """

    rows: dict[str, dict[str, Any]] = {}
    period_rows: list[dict[str, Any]] = []
    for code, method, label in methods:
        method_periods: list[dict[str, Any]] = []
        for record in period_records or []:
            method_payload = (record.get("methods") or {}).get(method) or {}
            tickets = list(method_payload.get("tickets") or [])
            if not tickets:
                continue
            actual_red = {
                _safe_int(number)
                for number in ((record.get("actual") or {}).get("red") or [])
            }
            actual_red = {number for number in actual_red if 1 <= number <= 33}
            funnel = (record.get("candidate_recall_funnel") or {}).get("red") or {}
            candidate_set = set(
                _safe_int(number)
                for number in (((funnel.get("candidate_set") or {}).get("selected")) or [])
            )
            candidate_set = {number for number in candidate_set if 1 <= number <= 33}
            candidate_found = actual_red & candidate_set
            final_union = _red_union(tickets)
            retained_found = candidate_found & final_union
            if candidate_set:
                candidate_hits = float(len(candidate_found))
                final_hits = float(len(retained_found))
                retention = final_hits / candidate_hits if candidate_hits > 0 else 0.0
                compression_regret = max(candidate_hits - final_hits, 0.0)
            else:
                candidate_hits = _safe_float(funnel.get("candidate_pool_hits"))
                final_hits = float(len(final_union & actual_red))
                retention = final_hits / candidate_hits if candidate_hits > 0 else 0.0
                compression_regret = max(candidate_hits - final_hits, 0.0)
            metrics = method_payload.get("model") or {}
            method_period = {
                "validation_period": str(record.get("validation_period", "")),
                "training_through_period": str(record.get("training_through_period", "")),
                "candidate_red_hits": candidate_hits,
                "final_red_union_hits": final_hits,
                "candidate_to_final_retention": float(retention),
                "compression_regret": float(compression_regret),
                "candidate_found_numbers": sorted(candidate_found),
                "final_retained_candidate_numbers": sorted(retained_found),
                "final_lost_candidate_numbers": sorted(candidate_found - retained_found),
                "best_red_hit": _safe_float(metrics.get("best_red_hits")),
                "any_3_plus": _safe_float(metrics.get("any_red_ge3")),
                "any_4_plus": _safe_float(metrics.get("any_red_ge4")),
                "average_red_hits": _safe_float(metrics.get("average_red_hits")),
                "portfolio_union": float(len(_red_union(tickets))),
                "jaccard": _mean_pairwise_jaccard(tickets),
                "ticket_count": len(tickets),
            }
            method_periods.append(method_period)
            period_rows.append({"code": code, "method": method, **method_period})

        def avg(key: str) -> float:
            values = [_safe_float(item.get(key)) for item in method_periods]
            return float(mean(values)) if values else 0.0

        rows[method] = {
            "code": code,
            "method": method,
            "label": label,
            "periods": len(method_periods),
            "same_candidate_pool": True,
            "candidate_pool_source": "period_records[].candidate_recall_funnel + period_records[].methods[*].tickets",
            "candidate_recall": avg("candidate_red_hits") / 6.0,
            "average_candidate_red_hits": avg("candidate_red_hits"),
            "final_recall": avg("final_red_union_hits") / 6.0,
            "average_final_red_union_hits": avg("final_red_union_hits"),
            "candidate_to_final_retention": avg("candidate_to_final_retention"),
            "compression_regret": avg("compression_regret"),
            "average_best_red_hit": avg("best_red_hit"),
            "any_3_plus": avg("any_3_plus"),
            "any_4_plus": avg("any_4_plus"),
            "average_red_hits": avg("average_red_hits"),
            "portfolio_union": avg("portfolio_union"),
            "jaccard": avg("jaccard"),
            "retention_p10": _percentile(
                [_safe_float(item.get("candidate_to_final_retention")) for item in method_periods],
                0.10,
            ),
            "retention_p50": _percentile(
                [_safe_float(item.get("candidate_to_final_retention")) for item in method_periods],
                0.50,
            ),
            "retention_p90": _percentile(
                [_safe_float(item.get("candidate_to_final_retention")) for item in method_periods],
                0.90,
            ),
        }

    comparable = [row for row in rows.values() if int(row.get("periods", 0)) > 0]
    ranked = sorted(
        comparable,
        key=lambda row: (
            _safe_float(row.get("candidate_to_final_retention")),
            -_safe_float(row.get("compression_regret")),
            _safe_float(row.get("any_3_plus")),
            _safe_float(row.get("average_best_red_hit")),
            _safe_float(row.get("portfolio_union")),
            -_safe_float(row.get("jaccard")),
        ),
        reverse=True,
    )
    leader = ranked[0]["method"] if ranked else None
    red_ge3 = rows.get("red_ge3_objective", {})
    leader_row = rows.get(str(leader), {}) if leader else {}
    return {
        "schema": "red_method_portfolio_ablation_v1",
        "comparison_question": "red_ge3_objective 是否比 greedy / pareto / historical_hit / global_coverage 更能保留同一 Candidate Pool 中已发现的真实红球",
        "production_changed": False,
        "same_candidate_pool_enforced": True,
        "methods": rows,
        "ranking": [row["method"] for row in ranked],
        "leader": leader,
        "red_ge3_rank": (
            [row["method"] for row in ranked].index("red_ge3_objective") + 1
            if "red_ge3_objective" in [row["method"] for row in ranked]
            else None
        ),
        "leader_vs_red_ge3": {
            "leader": leader,
            "candidate_to_final_retention_delta": _safe_float(
                leader_row.get("candidate_to_final_retention")
            ) - _safe_float(red_ge3.get("candidate_to_final_retention")),
            "compression_regret_delta": _safe_float(
                leader_row.get("compression_regret")
            ) - _safe_float(red_ge3.get("compression_regret")),
            "any_3_plus_delta": _safe_float(leader_row.get("any_3_plus")) - _safe_float(red_ge3.get("any_3_plus")),
            "any_4_plus_delta": _safe_float(leader_row.get("any_4_plus")) - _safe_float(red_ge3.get("any_4_plus")),
        },
        "period_rows": sorted(period_rows, key=lambda row: (row["validation_period"], row["code"])),
    }
