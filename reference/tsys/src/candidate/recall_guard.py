from __future__ import annotations

from typing import Any

from src.red.recall_guard import red_recall_guard_report


def candidate_recall_guard_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    red = red_recall_guard_report(period_records)
    return {
        "schema": "candidate_recall_guard_v7",
        "red": red,
        "rule": "Candidate Recall 优先于组合外观；若最终层压缩损失大，应降低重叠/结构/历史命中等硬约束。",
    }
