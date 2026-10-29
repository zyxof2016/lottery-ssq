from __future__ import annotations

from typing import Any

from src.blue.adaptive_topk import adaptive_blue_topk_policy
from src.blue.edge_detector import blue_no_edge_report


def _blue_rank_rows(period_records: list[dict[str, Any]]) -> list[int]:
    ranks: list[int] = []
    for record in period_records:
        ranking = (record.get("fused_validation") or {}).get("blue_ranking") or {}
        rank = ranking.get("rank") or ranking.get("mean_rank")
        try:
            value = int(round(float(rank)))
        except (TypeError, ValueError):
            continue
        if 1 <= value <= 16:
            ranks.append(value)
    return ranks


def _search_prediction_width(
    ranks: list[int],
    policy: dict[str, Any],
) -> dict[str, Any]:
    grid = sorted(set(int(value) for value in policy.get("candidate_k_grid", [3, 4, 5, 6, 8, 10, 12, 14, 16])))
    allowed = sorted(set(int(value) for value in policy.get("candidate_topk", grid)))
    if not ranks:
        fallback = max(allowed or grid or [12])
        return {
            "selected_width": int(fallback),
            "rows": 0,
            "candidates": [],
            "reason": "no_inner_oos_rows_for_width_search",
        }
    scored = []
    for k in grid:
        coverage = sum(1 for rank in ranks if rank <= k) / float(len(ranks))
        baseline = k / 16.0
        edge = coverage - baseline
        width_penalty = 0.015 * (k / 16.0)
        state_bonus = 0.0 if k in allowed else -0.04
        score = edge - width_penalty + state_bonus
        scored.append({
            "k": int(k),
            "coverage": float(coverage),
            "baseline": float(baseline),
            "edge": float(edge),
            "score": float(score),
            "allowed_by_state": bool(k in allowed),
        })
    viable = [item for item in scored if item["allowed_by_state"]]
    best = max(viable or scored, key=lambda item: (item["score"], item["coverage"], -item["k"]))
    return {
        "selected_width": int(best["k"]),
        "rows": int(len(ranks)),
        "candidates": scored,
        "reason": "inner_oos_candidate_k_search",
    }


def blue_prediction_set_report(period_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    edge = blue_no_edge_report(period_records, config)
    policy = adaptive_blue_topk_policy(edge, config)
    widths = policy.get("candidate_topk", [10, 12, 14, 16])
    search = (
        _search_prediction_width(_blue_rank_rows(period_records), policy)
        if bool(policy.get("adaptive_k_search", True))
        else {"selected_width": max(widths) if widths else 12, "rows": 0, "candidates": [], "reason": "adaptive_k_search_disabled"}
    )
    return {
        "schema": "blue_prediction_set_v7",
        "edge_state": edge.get("final_state"),
        "blue_mode": edge.get("blue_mode"),
        "candidate_width_range": widths,
        "selected_width": int(search.get("selected_width", max(widths) if widths else 12)),
        "inner_k_search": search,
        "policy": policy,
        "edge_report": edge,
    }
