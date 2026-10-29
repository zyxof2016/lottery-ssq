from __future__ import annotations

from typing import Any


def adaptive_blue_topk_policy(
    edge_report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    blue = v7.get("blue", {}) if isinstance(v7, dict) else {}
    grid = [max(1, min(16, int(value))) for value in blue.get("candidate_k_grid", [3, 4, 5, 6, 8, 10, 12, 14, 16])]
    state = str(edge_report.get("final_state") or edge_report.get("state") or "NO_EDGE")
    if state == "STRONG_EDGE":
        k_values = list(blue.get("strong_edge_topk", [3, 4, 5, 6]))
        coverage_priority = "probability_first"
    elif state == "WEAK_EDGE":
        k_values = list(blue.get("weak_edge_topk", [6, 8, 10, 12]))
        coverage_priority = "balanced_probability_coverage"
    else:
        k_values = list(blue.get("no_edge_topk", [10, 12, 14, 16]))
        coverage_priority = "coverage_first"
    k_values = [value for value in [max(1, min(16, int(value))) for value in k_values] if value in grid]
    if not k_values:
        k_values = grid
    return {
        "schema": "blue_adaptive_topk_policy_v7",
        "edge_state": state,
        "adaptive_k_search": bool(blue.get("adaptive_k_search", True)),
        "candidate_k_grid": sorted(set(grid)),
        "candidate_topk": sorted(set(k_values)),
        "portfolio_unique_blue_floor": int(blue.get("unique_blue_floor", 5)),
        "coverage_priority": coverage_priority,
        "reason": "V7 maps blue evidence state to an Inner-OOS-searchable Prediction Set width.",
    }
