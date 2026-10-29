from __future__ import annotations

from typing import Any

from src.red.survival_tracker import red_survival_report


def number_survival_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "number_survival_v7",
        "red": red_survival_report(period_records),
        "blue": {
            "note": "蓝球生存路径由 blue_prediction_set、final blue portfolio 与 validation_trace 联合定位。",
        },
    }
