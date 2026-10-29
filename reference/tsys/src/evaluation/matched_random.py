from __future__ import annotations

from typing import Any


def matched_random_report(backtest: dict[str, Any]) -> dict[str, Any]:
    summary = backtest.get("summary", {}) if isinstance(backtest, dict) else {}
    rows = {}
    for method, metrics in summary.items():
        rows[method] = {
            "model_average_red_hits": metrics.get("model_average_red_hits"),
            "matched_no_signal_average_red_hits": metrics.get("matched_no_signal_average_red_hits"),
            "random_average_red_hits": metrics.get("random_average_red_hits"),
            "model_any_blue_hit": metrics.get("period_any_blue_hit_ratio"),
            "matched_any_blue_hit": metrics.get("matched_no_signal_period_any_blue_hit_ratio"),
            "random_any_blue_hit": metrics.get("random_period_any_blue_hit_ratio"),
        }
    return {
        "schema": "matched_random_report_v7",
        "baseline": backtest.get("matched_no_signal_baseline", {}),
        "methods": rows,
    }
