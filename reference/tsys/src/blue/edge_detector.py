from __future__ import annotations

from typing import Any

import numpy as np


def _safe_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _window_records(period_records: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if not period_records:
        return []
    return period_records[-min(max(int(window), 1), len(period_records)) :]


def _blue_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        ranking = (record.get("fused_validation") or {}).get("blue_ranking")
        if isinstance(ranking, dict):
            rows.append(ranking)
    return rows


def blue_edge_state(
    period_records: list[dict[str, Any]],
    *,
    window: int = 100,
    minimum_periods: int | None = None,
    hit5_baseline: float = 5.0 / 16.0,
    strong_edge_margin: float = 0.06,
    weak_edge_margin: float = 0.015,
    minimum_stability: float = 0.55,
) -> dict[str, Any]:
    """Classify blue-ball evidence from completed OOS rows only.

    V7 treats blue prediction as defensive unless the already-finished
    validation rows show a stable Top-K edge.  The state drives Top-K and
    shrinkage, but never claims a formal edge by itself.
    """
    records = _window_records(period_records, window)
    rows = _blue_rows(records)
    effective_minimum = max(1, int(minimum_periods if minimum_periods is not None else min(30, max(1, int(window)))))
    if not rows:
        return {
            "schema": "blue_edge_state_v7",
            "state": "NO_EDGE",
            "mode": "DEFENSIVE",
            "periods": 0,
            "minimum_periods": effective_minimum,
            "reason": "no_completed_oos_blue_rows",
            "alpha": 0.0,
        }
    if len(rows) < effective_minimum:
        return {
            "schema": "blue_edge_state_v7",
            "state": "NO_EDGE",
            "mode": "DEFENSIVE",
            "periods": len(rows),
            "window": int(window),
            "minimum_periods": effective_minimum,
            "reason": "insufficient_completed_oos_blue_rows",
            "alpha": 0.08,
        }
    hit1 = _safe_mean([float(row.get("hit_at_1", 0.0) or 0.0) for row in rows])
    hit3 = _safe_mean([float(row.get("hit_at_3", 0.0) or 0.0) for row in rows])
    hit5 = _safe_mean([float(row.get("hit_at_5", 0.0) or 0.0) for row in rows])
    mrr = _safe_mean([float(row.get("mrr", 0.0) or 0.0) for row in rows])
    mean_rank = _safe_mean([float(row.get("mean_rank", 0.0) or 0.0) for row in rows])
    hit5_values = [float(row.get("hit_at_5", 0.0) or 0.0) for row in rows]
    block_size = max(10, min(25, len(hit5_values) // 4 or 10))
    block_rates = [
        float(np.mean(hit5_values[index : index + block_size]))
        for index in range(0, len(hit5_values), block_size)
        if hit5_values[index : index + block_size]
    ]
    positive_blocks = sum(1 for value in block_rates if value >= hit5_baseline)
    stability = positive_blocks / max(len(block_rates), 1)
    edge = float((hit5 or 0.0) - hit5_baseline)
    if edge >= strong_edge_margin and stability >= minimum_stability:
        state = "STRONG_EDGE"
        mode = "AGGRESSIVE"
        alpha = 0.70
    elif edge >= weak_edge_margin and stability >= max(0.40, minimum_stability - 0.15):
        state = "WEAK_EDGE"
        mode = "BALANCED"
        alpha = 0.22
    else:
        state = "NO_EDGE"
        mode = "DEFENSIVE"
        alpha = 0.08
    return {
        "schema": "blue_edge_state_v7",
        "state": state,
        "mode": mode,
        "periods": len(rows),
        "window": int(window),
        "minimum_periods": effective_minimum,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "hit5_baseline": float(hit5_baseline),
        "hit5_edge": edge,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "block_size": int(block_size),
        "block_hit5_rates": block_rates,
        "stability": float(stability),
        "alpha": float(alpha),
        "reason": "completed_oos_topk_edge_and_block_stability",
    }


def blue_no_edge_report(
    period_records: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    blue_config = v7.get("blue", {}) if isinstance(v7, dict) else {}
    windows = blue_config.get("edge_windows", [30, 100, 300])
    states = {
        str(window): blue_edge_state(
            period_records,
            window=int(window),
            minimum_periods=int(blue_config.get("minimum_edge_periods", min(30, int(window)))),
            strong_edge_margin=float(blue_config.get("strong_edge_margin", 0.06)),
            weak_edge_margin=float(blue_config.get("weak_edge_margin", 0.015)),
            minimum_stability=float(blue_config.get("minimum_stability", 0.55)),
        )
        for window in windows
    }
    ordered = [states[str(window)] for window in windows if str(window) in states]
    strong_count = sum(1 for item in ordered if item.get("state") == "STRONG_EDGE")
    weak_or_better = sum(1 for item in ordered if item.get("state") in {"STRONG_EDGE", "WEAK_EDGE"})
    if strong_count >= 2:
        final_state = "STRONG_EDGE"
    elif weak_or_better >= 2:
        final_state = "WEAK_EDGE"
    else:
        final_state = "NO_EDGE"
    return {
        "schema": "blue_no_edge_report_v7",
        "enabled": bool(blue_config.get("no_edge_detector", True)),
        "final_state": final_state,
        "blue_mode": "DEFENSIVE" if final_state == "NO_EDGE" else ("AGGRESSIVE" if final_state == "STRONG_EDGE" else "BALANCED"),
        "windows": states,
        "decision_rule": "two_window_consensus_over_completed_oos_only",
    }
