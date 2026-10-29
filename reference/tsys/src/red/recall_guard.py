from __future__ import annotations

from typing import Any

import numpy as np


def red_recall_guard_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        (record.get("candidate_recall_funnel") or {}).get("red", {})
        for record in period_records
        if isinstance((record.get("candidate_recall_funnel") or {}).get("red", {}), dict)
    ]
    if not rows:
        return {"schema": "red_recall_guard_report_v7", "periods": 0, "active": False}
    candidate_hits = [float(row.get("candidate_pool_hits", 0.0) or 0.0) for row in rows]
    method_hits = [float(row.get("method_layer_hits", 0.0) or 0.0) for row in rows]
    final_hits = [float(row.get("final_layer_hits", 0.0) or 0.0) for row in rows]
    model_top18 = [float(row.get("model_top18_hits", 0.0) or 0.0) for row in rows]
    true_losses: list[float] = []
    true_retentions: list[float] = []
    for row in rows:
        survival = row.get("true_survival") if isinstance(row, dict) else None
        if not isinstance(survival, dict):
            continue
        found = float(survival.get("candidate_found_count", 0.0) or 0.0)
        retained = float(survival.get("final_retained_count", 0.0) or 0.0)
        loss = float(survival.get("candidate_to_final_loss", max(0.0, found - retained)) or 0.0)
        true_losses.append(max(0.0, loss))
        if found > 0:
            true_retentions.append(max(0.0, min(1.0, retained / found)))
    if true_losses:
        compression_loss = float(np.mean(true_losses))
        retention = float(np.mean(true_retentions)) if true_retentions else None
    else:
        legacy_loss = float(np.mean(candidate_hits) - np.mean(final_hits))
        compression_loss = float(max(0.0, legacy_loss))
        retention = None
    return {
        "schema": "red_recall_guard_report_v7",
        "periods": len(rows),
        "candidate_pool_average_hits": float(np.mean(candidate_hits)),
        "model_top18_average_hits": float(np.mean(model_top18)),
        "method_layer_average_hits": float(np.mean(method_hits)),
        "final_layer_average_hits": float(np.mean(final_hits)),
        "candidate_to_final_loss": compression_loss,
        "candidate_to_final_true_retention": retention,
        "definition": "压缩损失优先采用真实号码生存口径：Candidate Set 已找到的真实红球在 Final 5 中丢失了几个。",
        "active": compression_loss >= 1.0,
        "reason": "protect candidate recall when final portfolio compression destroys too many true red numbers",
    }
