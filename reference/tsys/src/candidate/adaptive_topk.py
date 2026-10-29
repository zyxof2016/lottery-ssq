from __future__ import annotations

from typing import Any

from src.uncertainty.prediction_set import blue_prediction_set_report


def adaptive_topk_report(period_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    blue = blue_prediction_set_report(period_records, config)
    return {
        "schema": "adaptive_topk_report_v7",
        "blue_prediction_set": blue,
        "rule": "Top-K 是候选集合宽度，不是强行购买数量；最终覆盖由 Portfolio Engine 决定。",
    }
