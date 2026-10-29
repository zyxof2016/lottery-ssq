from __future__ import annotations

from typing import Any

import numpy as np


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _series(backtest: dict[str, Any], method: str, key: str) -> list[float]:
    rows: list[float] = []
    for record in backtest.get("period_records", []) or []:
        try:
            rows.append(float(record["methods"][method]["model"][key]))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def evidence_score_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    model_avg = float(metrics.get("model_average_red_hits", 0.0) or 0.0)
    matched_avg = float(metrics.get("matched_no_signal_average_red_hits", 0.0) or 0.0)
    random_avg = float(metrics.get("random_average_red_hits", 0.0) or 0.0)
    model_blue = float(metrics.get("period_any_blue_hit_ratio", 0.0) or 0.0)
    matched_blue = float(metrics.get("matched_no_signal_period_any_blue_hit_ratio", 0.0) or 0.0)
    p_value = float((metrics.get("significance") or {}).get("holm_adjusted_p_value", 1.0) or 1.0)
    red_edge = model_avg - max(matched_avg, random_avg)
    blue_edge = model_blue - matched_blue
    statistical = max(0.0, min(1.0, 1.0 - p_value))
    raw_score = 0.55 * red_edge + 0.45 * blue_edge
    score = _clip(raw_score)
    confidence = float(max(0.0, min(1.0, statistical * min(1.0, abs(raw_score) * 4.0 + 0.15))))
    return {
        "schema": "evidence_score_v7",
        "score": score,
        "confidence": confidence,
        "red_edge_vs_baseline": red_edge,
        "blue_edge_vs_matched": blue_edge,
        "holm_adjusted_p_value": p_value,
        "decision": "TRUST" if score > 0.03 and confidence >= 0.35 else ("REJECT" if score < -0.03 else "NO_EDGE"),
    }


def _window_method_metrics(records: list[dict[str, Any]], method: str, window: int) -> dict[str, Any]:
    rows = records[-min(max(1, int(window)), len(records)) :] if records else []
    red_hits: list[float] = []
    blue_hits: list[float] = []
    matched_red: list[float] = []
    matched_blue: list[float] = []
    for record in rows:
        methods = record.get("methods", {}) if isinstance(record, dict) else {}
        model = ((methods.get(method) or {}).get("model") or {}) if isinstance(methods, dict) else {}
        if model:
            if model.get("average_red_hits") is not None:
                red_hits.append(float(model.get("average_red_hits") or 0.0))
            if model.get("any_blue_hit") is not None:
                blue_hits.append(float(model.get("any_blue_hit") or 0.0))
        matched = record.get("matched_no_signal") or record.get("matched_random") or {}
        if isinstance(matched, dict):
            if matched.get("average_red_hits") is not None:
                matched_red.append(float(matched.get("average_red_hits") or 0.0))
            if matched.get("any_blue_hit") is not None:
                matched_blue.append(float(matched.get("any_blue_hit") or 0.0))
    return {
        "periods": len(rows),
        "red_mean": _mean(red_hits),
        "blue_hit_rate": _mean(blue_hits),
        "matched_red_mean": _mean(matched_red),
        "matched_blue_hit_rate": _mean(matched_blue),
    }


def _card_decision(delta: float | None, confidence: float, *, minimum_delta: float = 0.0) -> str:
    if delta is None:
        return "NO_EDGE"
    if float(delta) > minimum_delta and confidence >= 0.35:
        return "TRUST"
    if float(delta) < -abs(minimum_delta) and confidence >= 0.35:
        return "REJECT"
    return "NO_EDGE"


def build_evidence_cards(backtest: dict[str, Any], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build V7 EvidenceCard rows by color, method, and OOS window."""
    summary = backtest.get("summary", {}) if isinstance(backtest, dict) else {}
    records = list(backtest.get("period_records", []) or []) if isinstance(backtest, dict) else []
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    evidence_cfg = v7.get("evidence", {}) if isinstance(v7, dict) else {}
    windows = [int(value) for value in evidence_cfg.get("windows", [30, 100, 300]) if int(value) > 0]
    if not windows:
        windows = [30, 100, 300]
    cards: list[dict[str, Any]] = []
    for method, metrics in summary.items():
        if not isinstance(metrics, dict):
            continue
        score = evidence_score_from_metrics(metrics)
        for window in windows:
            metrics_window = _window_method_metrics(records, method, window)
            red_delta = None
            if metrics_window.get("red_mean") is not None:
                baseline = metrics_window.get("matched_red_mean")
                if baseline is None:
                    baseline = metrics.get("matched_no_signal_average_red_hits", metrics.get("random_average_red_hits", 0.0))
                red_delta = float(metrics_window["red_mean"] or 0.0) - float(baseline or 0.0)
            blue_delta = None
            if metrics_window.get("blue_hit_rate") is not None:
                baseline = metrics_window.get("matched_blue_hit_rate")
                if baseline is None:
                    baseline = metrics.get("matched_no_signal_period_any_blue_hit_ratio", 5.0 / 16.0)
                blue_delta = float(metrics_window["blue_hit_rate"] or 0.0) - float(baseline or 0.0)
            for color, delta in (("red", red_delta), ("blue", blue_delta)):
                cards.append({
                    "schema": "evidence_card_v7",
                    "color": color,
                    "method": str(method),
                    "window": int(window),
                    "metric_set": {
                        "oos_delta": delta,
                        "calibration": None,
                        "stability": (backtest.get("v7_diagnostics") or {}).get("stability"),
                        "recent_degradation": None,
                    },
                    "oos_delta": delta,
                    "bootstrap_ci": None,
                    "p_value": (metrics.get("significance") or {}).get("p_value"),
                    "adjusted_p_value": (metrics.get("significance") or {}).get("holm_adjusted_p_value"),
                    "stability": None,
                    "recent_degradation": None,
                    "correlation_group": None,
                    "complexity_penalty": 0.0,
                    "decision": _card_decision(delta, float(score.get("confidence", 0.0) or 0.0), minimum_delta=0.01 if color == "red" else 0.0),
                })
    return cards


def build_evidence_report(backtest: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = backtest.get("summary", {}) if isinstance(backtest, dict) else {}
    method_scores = {
        method: evidence_score_from_metrics(metrics)
        for method, metrics in summary.items()
        if isinstance(metrics, dict)
    }
    ranked = sorted(method_scores.items(), key=lambda item: (item[1]["score"], item[1]["confidence"]), reverse=True)
    records = backtest.get("period_records", []) if isinstance(backtest, dict) else []
    stability = {}
    for method in summary:
        values = _series(backtest, method, "average_red_hits")
        if values:
            half = max(1, len(values) // 2)
            stability[method] = {
                "first_half_mean": _mean(values[:half]),
                "second_half_mean": _mean(values[half:]),
                "recent_degradation": (
                    float((_mean(values[half:]) or 0.0) - (_mean(values[:half]) or 0.0))
                    if len(values) > 1 else None
                ),
            }
    evidence_cards = build_evidence_cards(backtest, config)
    return {
        "schema": "evidence_report_v7",
        "periods": len(records),
        "method_scores": method_scores,
        "evidence_cards": evidence_cards,
        "card_decision_counts": {
            decision: sum(1 for card in evidence_cards if card.get("decision") == decision)
            for decision in ("TRUST", "NO_EDGE", "REJECT")
        },
        "best_method": ranked[0][0] if ranked else None,
        "best_score": ranked[0][1] if ranked else None,
        "stability": stability,
        "rule": "生产信任只来自严格样本外、匹配随机、稳定性与多重检验后的综合证据。",
    }
