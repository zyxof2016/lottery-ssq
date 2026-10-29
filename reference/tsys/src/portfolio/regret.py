from __future__ import annotations

from collections import Counter
from typing import Any


def portfolio_regret_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    candidate_hits_total = 0.0
    method_hits_total = 0.0
    final_hits_total = 0.0
    final_regret_total = 0.0
    information_retention_rows: list[float] = []
    compression_regret_rows: list[float] = []
    for record in period_records or []:
        funnel = (record.get("candidate_recall_funnel") or {}).get("red") or {}
        actual = [int(value) for value in funnel.get("actual", []) or []]
        candidate_miss = set(int(value) for value in funnel.get("missed_at_candidate_pool", []) or [])
        method_miss = set(int(value) for value in funnel.get("missed_at_method_layer", []) or [])
        final_miss = set(int(value) for value in funnel.get("missed_at_final_layer", []) or [])
        candidate_hits = float(funnel.get("candidate_pool_hits", 0.0) or 0.0)
        method_hits = float(funnel.get("method_layer_hits", 0.0) or 0.0)
        final_hits = float(funnel.get("final_layer_hits", 0.0) or 0.0)
        true_survival = funnel.get("true_survival") if isinstance(funnel.get("true_survival"), dict) else {}
        true_found = float(true_survival.get("candidate_found_count", candidate_hits) or 0.0)
        true_final_retained = float(true_survival.get("final_retained_count", min(candidate_hits, final_hits)) or 0.0)
        true_loss = float(true_survival.get("candidate_to_final_loss", max(0.0, true_found - true_final_retained)) or 0.0)
        true_retention = true_survival.get("candidate_to_final_retention")
        if true_retention is None:
            true_retention = true_final_retained / true_found if true_found > 0 else None
        else:
            true_retention = float(true_retention)
        if true_retention is not None:
            true_retention = max(0.0, min(1.0, true_retention))
        candidate_hits_total += candidate_hits
        method_hits_total += method_hits
        final_hits_total += final_hits
        final_regret_total += max(0.0, true_loss)
        if true_retention is not None:
            information_retention_rows.append(float(true_retention))
            compression_regret_rows.append(1.0 - float(true_retention))
        for number in actual:
            if number in candidate_miss:
                stage = "model_or_feature_loss"
            elif number in method_miss:
                stage = "method_optimizer_loss"
            elif number in final_miss:
                stage = "final_portfolio_loss"
            else:
                stage = "protected_to_final"
            stage_counter[stage] += 1
        if len(examples) < 8 and (candidate_miss or method_miss or final_miss):
            examples.append({
                "period": record.get("period"),
                "candidate_loss": sorted(candidate_miss),
                "method_loss": sorted(method_miss),
                "final_portfolio_loss": sorted(final_miss),
            })
    total = sum(stage_counter.values())
    periods = len(period_records or [])
    avg_candidate_hits = candidate_hits_total / float(periods) if periods else 0.0
    avg_final_hits = final_hits_total / float(periods) if periods else 0.0
    hit_retention = (
        float(sum(information_retention_rows) / len(information_retention_rows))
        if information_retention_rows
        else 0.0
    )
    hit_retention = max(0.0, min(1.0, hit_retention))
    return {
        "schema": "regret_report_v7",
        "periods": periods,
        "stage_counts": dict(stage_counter),
        "stage_ratios": {key: value / total for key, value in stage_counter.items()} if total else {},
        "information_retention_rate": float(hit_retention),
        "compression_regret": float(max(0.0, 1.0 - hit_retention)),
        "average_candidate_pool_hits": float(avg_candidate_hits),
        "average_method_layer_hits": float(method_hits_total / float(periods)) if periods else 0.0,
        "average_final_layer_hits": float(avg_final_hits),
        "average_final_compression_loss": float(final_regret_total / float(periods)) if periods else 0.0,
        "period_information_retention_rate": (
            float(sum(information_retention_rows) / len(information_retention_rows))
            if information_retention_rows else 0.0
        ),
        "period_compression_regret": (
            float(sum(compression_regret_rows) / len(compression_regret_rows))
            if compression_regret_rows else 0.0
        ),
        "examples": examples,
        "interpretation": "IRR=Final保留的候选池真实命中/候选池真实命中；Compression Regret重点衡量候选池已有信息在最终Portfolio压缩中丢失多少。",
    }
