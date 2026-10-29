from __future__ import annotations

from typing import Any

import numpy as np


def _window_records(period_records: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    if not period_records:
        return []
    return period_records[-min(max(int(window), 1), len(period_records)) :]


def _safe_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _red_record_metrics(record: dict[str, Any]) -> dict[str, float] | None:
    methods = record.get("methods") if isinstance(record, dict) else {}
    consensus = ((methods or {}).get("consensus") or {}).get("model") or {}
    if consensus:
        average = consensus.get("average_red_hits")
        ge3 = consensus.get("red_ge3") or consensus.get("any_red_ge3")
        best = consensus.get("best_red_hits")
    else:
        validation = record.get("fused_validation") or {}
        average = validation.get("red_hits")
        ge3 = 1.0 if float(average or 0.0) >= 3.0 else 0.0
        best = average
    if average is None:
        return None
    matched = record.get("matched_no_signal") or record.get("matched_random") or {}
    matched_average = matched.get("average_red_hits") if isinstance(matched, dict) else None
    return {
        "average_red_hits": float(average or 0.0),
        "red_ge3": float(ge3 or 0.0),
        "best_red_hits": float(best or average or 0.0),
        "matched_average_red_hits": float(matched_average if matched_average is not None else 6.0 * 6.0 / 33.0),
    }


def red_edge_state(
    period_records: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Classify red-ball evidence from completed OOS rows only.

    Red and blue have different objectives and baselines.  This detector is the
    only source of red Portfolio edge state in V7, so blue Top-K evidence can no
    longer control red budget allocation.
    """
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    windows = [int(value) for value in red_cfg.get("edge_windows", [30, 100, 300])]
    effective_window = int(window or (windows[0] if windows else 100))
    minimum_periods = max(1, int(red_cfg.get("minimum_edge_periods", min(30, effective_window))))
    rows = [
        metrics
        for record in _window_records(period_records, effective_window)
        if (metrics := _red_record_metrics(record)) is not None
    ]
    if len(rows) < minimum_periods:
        return {
            "schema": "red_edge_state_v7",
            "state": "NO_EDGE",
            "mode": "DEFENSIVE",
            "periods": len(rows),
            "window": effective_window,
            "minimum_periods": minimum_periods,
            "reason": "insufficient_completed_oos_red_rows",
        }
    averages = [row["average_red_hits"] for row in rows]
    baselines = [row["matched_average_red_hits"] for row in rows]
    ge3_values = [row["red_ge3"] for row in rows]
    edge = float((_safe_mean(averages) or 0.0) - (_safe_mean(baselines) or 0.0))
    ge3_rate = float(_safe_mean(ge3_values) or 0.0)
    block_size = max(10, min(25, len(rows) // 4 or 10))
    block_edges = []
    for index in range(0, len(rows), block_size):
        block = rows[index : index + block_size]
        if not block:
            continue
        block_edges.append(
            float(np.mean([item["average_red_hits"] for item in block]) - np.mean([item["matched_average_red_hits"] for item in block]))
        )
    positive_blocks = sum(1 for value in block_edges if value >= 0.0)
    stability = float(positive_blocks / max(len(block_edges), 1))
    strong_margin = float(red_cfg.get("strong_edge_margin", 0.08))
    weak_margin = float(red_cfg.get("weak_edge_margin", 0.015))
    minimum_stability = float(red_cfg.get("minimum_stability", 0.55))
    if edge >= strong_margin and stability >= minimum_stability:
        state = "STRONG_EDGE"
        mode = "AGGRESSIVE"
    elif edge >= weak_margin and stability >= max(0.40, minimum_stability - 0.15):
        state = "WEAK_EDGE"
        mode = "BALANCED"
    else:
        state = "NO_EDGE"
        mode = "DEFENSIVE"
    return {
        "schema": "red_edge_state_v7",
        "state": state,
        "mode": mode,
        "periods": len(rows),
        "window": effective_window,
        "minimum_periods": minimum_periods,
        "average_red_hits": _safe_mean(averages),
        "matched_average_red_hits": _safe_mean(baselines),
        "edge_vs_matched": edge,
        "red_ge3_rate": ge3_rate,
        "block_size": block_size,
        "block_edges": block_edges,
        "stability": stability,
        "reason": "completed_oos_red_edge_and_block_stability",
    }


def red_edge_report(period_records: list[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    windows = [int(value) for value in red_cfg.get("edge_windows", [30, 100, 300])]
    states = {
        str(window): red_edge_state(period_records, config, window=int(window))
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
        "schema": "red_edge_report_v7",
        "enabled": bool(red_cfg.get("edge_detector", True)),
        "final_state": final_state,
        "red_mode": "DEFENSIVE" if final_state == "NO_EDGE" else ("AGGRESSIVE" if final_state == "STRONG_EDGE" else "BALANCED"),
        "windows": states,
        "decision_rule": "two_window_consensus_over_completed_oos_only",
    }
