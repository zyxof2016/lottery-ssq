from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.backtest.metrics import paired_significance
from src.fusion.model_fusion import random_baseline_brier_components

BLUE_TOPK_LADDER = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)


def _humanize_blue_reason(reason: str | None) -> str:
    text = str(reason or "").strip()
    mapping = {
        "multiscale_weak_signal_with_relative_tilt_shrunk_to_uniform": "多窗口蓝球信号偏弱，已向均匀基线收缩",
        "multiscale_relative_loss_tilt_shrunk_to_uniform": "蓝球相对损失倾斜后仍偏弱，已向均匀基线收缩",
        "relative_loss_tilt_shrunk_to_uniform": "蓝球损失优势不足，已向均匀基线收缩",
        "uniform": "蓝球侧退化为均匀基线",
    }
    return mapping.get(text, text or "未记录")


def _blue_period_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for record in records:
        actual = record.get("actual", {})
        actual_blue = actual.get("blue")
        if actual_blue is None:
            continue
        fused_prediction = record.get("fused_prediction", {})
        top_blue_list = list(fused_prediction.get("top_blue", []))
        fused_top1 = top_blue_list[0] if top_blue_list else None
        top3 = top_blue_list[:3]
        top5 = top_blue_list[:5]
        fused_validation = record.get("fused_validation", {})
        top1_hit = bool(fused_validation.get("blue_hit"))
        top3_hit = int(actual_blue) in {int(value) for value in top3}
        top5_hit = int(actual_blue) in {int(value) for value in top5}
        consensus_tickets = record.get("methods", {}).get("consensus", {}).get("tickets", [])
        consensus_blues = [int(ticket["blue"]) for ticket in consensus_tickets if ticket.get("blue") is not None]
        consensus_any_hit = any(bool(ticket.get("blue_hit")) for ticket in consensus_tickets)
        gate_reason = str(record.get("calibration_adjustment", {}).get("parameter_gate_reason", "")).strip()
        blue_reason_code = str(record.get("weight_reason_by_color", {}).get("blue", "")).strip()
        blue_reason = _humanize_blue_reason(blue_reason_code)
        if top1_hit:
            reason = "融合首选命中"
        elif consensus_any_hit:
            reason = "组合覆盖命中但首选失准"
        elif top5_hit:
            reason = "Prediction Set/Hit@5已覆盖但首位排序失准"
        elif "不足" in gate_reason:
            reason = "验证样本或门控不足"
        elif "冷却期" in gate_reason:
            reason = "调参冷却期内"
        elif "均匀分布" in gate_reason or "shrunk_to_uniform" in blue_reason_code or "弱" in blue_reason:
            reason = "蓝球信号弱并回缩基线"
        else:
            reason = "前序排序未覆盖实际蓝球"
        reason_counter[reason] += 1
        rows.append(
            {
                "validation_period": str(record.get("validation_period") or record.get("period") or ""),
                "training_through_period": str(record.get("training_through_period") or ""),
                "actual_blue": int(actual_blue),
                "fused_top1": int(fused_top1) if fused_top1 is not None else None,
                "fused_top3": [int(value) for value in top3],
                "fused_top5": [int(value) for value in top5],
                "consensus_blues": consensus_blues,
                "top1_hit": bool(top1_hit),
                "top3_hit": bool(top3_hit),
                "top5_hit": bool(top5_hit),
                "consensus_any_hit": bool(consensus_any_hit),
                "reason": reason,
                "gate_reason": gate_reason or "无",
                "blue_reason": blue_reason,
            }
        )
    periods = len(rows)
    top1_hits = sum(1 for row in rows if row["top1_hit"])
    top3_hits = sum(1 for row in rows if row["top3_hit"])
    top5_hits = sum(1 for row in rows if row["top5_hit"])
    consensus_hits = sum(1 for row in rows if row["consensus_any_hit"])
    reason_rows = [
        {
            "reason": reason,
            "count": count,
            "ratio": float(count / periods) if periods else None,
        }
        for reason, count in reason_counter.most_common()
    ]
    return {
        "periods": periods,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits,
        "consensus_hits": consensus_hits,
        "top1_hit_ratio": float(top1_hits / periods) if periods else None,
        "top3_hit_ratio": float(top3_hits / periods) if periods else None,
        "top5_hit_ratio": float(top5_hits / periods) if periods else None,
        "consensus_hit_ratio": float(consensus_hits / periods) if periods else None,
        "reason_rows": reason_rows,
        "rows": rows,
    }


def _blue_randomness_summary(randomness_audit: dict[str, Any]) -> dict[str, Any]:
    families = randomness_audit.get("families", {})
    details = [
        detail
        for family in families.values()
        for detail in family.get("details", [])
        if detail.get("color") == "blue"
    ]
    globally_significant = [
        detail for detail in details if detail.get("significant_after_global_holm")
    ]
    strongest = sorted(
        details,
        key=lambda detail: (
            detail.get("global_holm_adjusted_p_value")
            if detail.get("global_holm_adjusted_p_value") is not None
            else 1.0
        ),
    )[:5]
    return {
        "tested_series": len(details),
        "global_holm_significant_count": len(globally_significant),
        "strongest": [
            {
                "label": item.get("label"),
                "global_holm_adjusted_p_value": item.get("global_holm_adjusted_p_value"),
            }
            for item in strongest
        ],
        "conclusion": (
            "发现需复核的蓝球历史异常；它只提示数据质量或开奖机制复核，不能作为规律或号码评分依据。"
            if globally_significant
            else "未发现经全局 Holm 校正后仍显著的蓝球频率、游程、自相关或周期性证据。"
        ),
    }


def _ratio(values: np.ndarray, predicate: Any) -> float | None:
    return float(np.mean(predicate(values))) if len(values) else None


def _blue_sequence_patterns(
    history: pd.DataFrame | None,
    trend_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe blue distribution, movement and transitions without leakage."""
    if history is None or history.empty or "blue" not in history:
        return {"available": False, "reason": "缺少可分析的蓝球历史序列"}
    settings = trend_config or {}
    analysis_window = min(
        len(history), max(10, int(settings.get("analysis_window", 100)))
    )
    recent_window = min(
        analysis_window, max(10, int(settings.get("recent_window", 30)))
    )
    values = history["blue"].astype(int).to_numpy()[-analysis_window:]
    recent = values[-recent_window:]
    steps = np.abs(np.diff(values))
    recent_steps = np.abs(np.diff(recent))

    uniform_mean_step = float((16**2 - 1) / (3 * 16))
    uniform_large_step = float(
        np.mean([
            abs(left - right) >= 8
            for left in range(1, 17)
            for right in range(1, 17)
        ])
    )
    metric_rows = [
        {
            "metric": "小号占比（01–08）",
            "recent": _ratio(recent, lambda item: item <= 8),
            "analysis": _ratio(values, lambda item: item <= 8),
            "reference": 0.5,
            "interpretation": "观察分布，均匀基线为 50%",
        },
        {
            "metric": "奇数占比",
            "recent": _ratio(recent, lambda item: item % 2 == 1),
            "analysis": _ratio(values, lambda item: item % 2 == 1),
            "reference": 0.5,
            "interpretation": "观察分布，均匀基线为 50%",
        },
        {
            "metric": "相邻期重号率",
            "recent": _ratio(recent_steps, lambda item: item == 0),
            "analysis": _ratio(steps, lambda item: item == 0),
            "reference": 1.0 / 16.0,
            "interpretation": "相邻两期蓝球相同；仅作历史描述",
        },
        {
            "metric": "相邻期邻号率",
            "recent": _ratio(recent_steps, lambda item: item == 1),
            "analysis": _ratio(steps, lambda item: item == 1),
            "reference": 30.0 / 256.0,
            "interpretation": "相邻两期绝对差为 1",
        },
        {
            "metric": "相邻期平均跨度",
            "recent": float(np.mean(recent_steps)) if len(recent_steps) else None,
            "analysis": float(np.mean(steps)) if len(steps) else None,
            "reference": uniform_mean_step,
            "interpretation": "跨度定义为相邻两期蓝球的绝对差",
        },
        {
            "metric": "大跨度占比（≥8）",
            "recent": _ratio(recent_steps, lambda item: item >= 8),
            "analysis": _ratio(steps, lambda item: item >= 8),
            "reference": uniform_large_step,
            "interpretation": "观察蓝球跨区移动，不直接当作预测规则",
        },
    ]

    transitions = np.zeros((16, 16), dtype=int)
    for previous, current in zip(values[:-1], values[1:]):
        transitions[int(previous) - 1, int(current) - 1] += 1
    latest = int(values[-1])
    row = transitions[latest - 1]
    alpha = 1.0
    probabilities = (row + alpha) / (row.sum() + alpha * 16.0)
    transition_rows = [
        {
            "previous": latest,
            "next": int(index + 1),
            "observed_count": int(row[index]),
            "laplace_probability": float(probabilities[index]),
            "uniform_probability": 1.0 / 16.0,
        }
        for index in sorted(
            range(16), key=lambda item: (row[item], -item), reverse=True
        )[:5]
    ]

    cycle_rows = []
    for number in range(1, 17):
        positions = np.flatnonzero(values == number)
        intervals = np.diff(positions)
        cycle_rows.append({
            "number": number,
            "occurrences": int(len(positions)),
            "mean_interval": float(np.mean(intervals)) if len(intervals) else None,
            "maximum_interval": int(np.max(intervals)) if len(intervals) else None,
            "current_omission": (
                int(len(values) - 1 - positions[-1]) if len(positions) else int(len(values))
            ),
        })
    return {
        "available": True,
        "analysis_periods": int(len(values)),
        "recent_periods": int(len(recent)),
        "latest_blue": latest,
        "metric_rows": metric_rows,
        "transition_rows": transition_rows,
        "cycle_rows": cycle_rows,
        "recent_step_distribution": {
            str(step): int(np.count_nonzero(recent_steps == step)) for step in range(16)
        },
        "analysis_step_distribution": {
            str(step): int(np.count_nonzero(steps == step)) for step in range(16)
        },
        "conclusion": (
            "蓝球已按大小、奇偶、相邻期跨度、重号、邻号、转移和遗漏周期分别统计。"
            "这些历史偏离只有在严格左截断样本外检验中持续优于 1/16 均匀基线时，才会进入发布证据。"
        ),
    }


def _blue_ablation_summary(records: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    periods = len(records)
    stacking_applied = sum(1 for record in records if bool(record.get("stacking", {}).get("applied")))
    stacking_gate_passed = sum(
        1
        for record in records
        if bool(record.get("stacking", {}).get("blue_gate", {}).get("passed"))
    )
    base_hit1 = [
        float(record.get("fused_validation", {}).get("blue_ranking", {}).get("hit_at_1"))
        for record in records
        if record.get("fused_validation", {}).get("blue_ranking", {}).get("hit_at_1") is not None
    ]
    base_hit5 = [
        float(record.get("fused_validation", {}).get("blue_ranking", {}).get("hit_at_5"))
        for record in records
        if record.get("fused_validation", {}).get("blue_ranking", {}).get("hit_at_5") is not None
    ]
    return {
        "periods": periods,
        "primary_objective": "Prediction Set + Hit@5；Top1只保留为诊断，不作为蓝球生产主目标。",
        "stages": [
            {
                "stage": "base_probability_fusion",
                "description": "一层模型融合后的蓝球概率排序",
                "hit_at_1": float(np.mean(base_hit1)) if base_hit1 else None,
                "hit_at_5": float(np.mean(base_hit5)) if base_hit5 else None,
            },
            {
                "stage": "oof_stacking",
                "description": "二层OOF Stacking，仅满足门槛后参与",
                "applied_periods": int(stacking_applied),
                "blue_gate_passed_periods": int(stacking_gate_passed),
            },
            {
                "stage": "final_coverage_portfolio",
                "description": "最终5注/覆盖层蓝球命中",
                "coverage_hit_ratio": audit.get("consensus_hit_ratio"),
                "coverage_hits": audit.get("consensus_hits"),
            },
        ],
    }


def summarize_blue_evidence(
    backtest: dict[str, Any],
    randomness_audit: dict[str, Any],
    *,
    ticket_count: int,
    report_config: dict[str, Any],
    random_seed: int,
    history: pd.DataFrame | None = None,
    trend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report prequential blue-ball evidence without treating patterns as a rule."""
    records = list(backtest.get("period_records", []))
    ranked_records = [
        record for record in records
        if record.get("fused_validation", {}).get("blue_hit") is not None
    ]
    top1_hits = int(sum(bool(record["fused_validation"]["blue_hit"]) for record in ranked_records))
    top3_hits = int(sum(
        int(record["actual"]["blue"]) in {
            int(value) for value in list(record["fused_prediction"].get("top_blue", []))[:3]
        }
        for record in ranked_records
    ))
    ranking_rows = [
        record.get("fused_validation", {}).get("blue_ranking", {})
        for record in ranked_records
        if isinstance(record.get("fused_validation", {}).get("blue_ranking"), dict)
    ]
    periods = len(ranked_records)
    ranking_periods = len(ranking_rows)
    topk_ladder = []
    for top_k in BLUE_TOPK_LADDER:
        hits = int(sum(int(row.get("rank", 999)) <= int(top_k) for row in ranking_rows))
        uniform_probability = float(top_k) / 16.0
        topk_ladder.append({
            "k": int(top_k),
            "hits": hits,
            "periods": ranking_periods,
            "hit_ratio": float(hits / ranking_periods) if ranking_periods else None,
            "uniform_probability": uniform_probability,
            "edge_vs_uniform": (
                float(hits / ranking_periods) - uniform_probability
                if ranking_periods
                else None
            ),
            "exact_one_sided_p_value": (
                float(binomtest(hits, ranking_periods, uniform_probability, alternative="greater").pvalue)
                if ranking_periods
                else None
            ),
        })
    baseline = random_baseline_brier_components()["blue"]
    blue_losses = [
        float(record["fused_brier_components"]["blue"])
        for record in records
    ]
    options = {
        "confidence_level": float(report_config.get("confidence_level", 0.95)),
        "permutation_repetitions": int(report_config.get("permutation_repetitions", 20000)),
        "random_seed": int(random_seed) + 900_000,
    }
    brier_significance = paired_significance(
        [float(baseline)] * len(blue_losses), blue_losses, **options
    ) if blue_losses else {"p_value": None, "confidence_interval": [None, None]}
    average_loss = float(np.mean(blue_losses)) if blue_losses else None
    model_diagnostics = backtest.get("model_diagnostics", {}).get("by_color", {}).get("blue", {})
    model_rows = [
        {
            "model": name,
            "brier_skill_score": detail.get("brier_skill_score"),
            "top1_hit_ratio": detail.get("top1_hit_ratio"),
            "holm_adjusted_p_value": detail.get("significance", {}).get("holm_adjusted_p_value"),
            "top1_holm_adjusted_p_value": detail.get("top1_significance", {}).get("holm_adjusted_p_value"),
            "proven_better_than_uniform": bool(detail.get("proven_better_than_uniform")),
            "top1_proven_better_than_uniform": bool(detail.get("top1_proven_better_than_uniform")),
        }
        for name, detail in model_diagnostics.items()
    ]
    audit = _blue_period_audit(ranked_records)
    brier_confirmed = bool(
        brier_significance.get("p_value") is not None
        and float(brier_significance["p_value"]) < 0.05
        and float(brier_significance.get("mean_difference", 0.0)) > 0
    )
    return {
        "periods": periods,
        "top1": {
            "hits": top1_hits,
            "hit_ratio": float(top1_hits / periods) if periods else None,
            "uniform_probability": 1.0 / 16.0,
            "exact_one_sided_p_value": float(binomtest(top1_hits, periods, 1.0 / 16.0, alternative="greater").pvalue) if periods else None,
        },
        "top3": {
            "hits": top3_hits,
            "hit_ratio": float(top3_hits / periods) if periods else None,
            "uniform_probability": 3.0 / 16.0,
            "exact_one_sided_p_value": float(binomtest(top3_hits, periods, 3.0 / 16.0, alternative="greater").pvalue) if periods else None,
        },
        "ranking": {
            "hit_at_1": float(np.mean([row["hit_at_1"] for row in ranking_rows])) if ranking_rows else None,
            "hit_at_3": float(np.mean([row["hit_at_3"] for row in ranking_rows])) if ranking_rows else None,
            "hit_at_5": float(np.mean([row["hit_at_5"] for row in ranking_rows])) if ranking_rows else None,
            "mrr": float(np.mean([row["mrr"] for row in ranking_rows])) if ranking_rows else None,
            "ndcg_at_5": float(np.mean([row["ndcg_at_5"] for row in ranking_rows])) if ranking_rows else None,
            "mean_rank": float(np.mean([row["mean_rank"] for row in ranking_rows])) if ranking_rows else None,
        },
        "prediction_set": {
            "primary_metric": "hit_at_5",
            "hit_at_5": float(np.mean([row["hit_at_5"] for row in ranking_rows])) if ranking_rows else None,
            "uniform_probability": 5.0 / 16.0,
            "edge_vs_uniform": (
                float(np.mean([row["hit_at_5"] for row in ranking_rows])) - 5.0 / 16.0
                if ranking_rows
                else None
            ),
            "interpretation": "蓝球按候选集合覆盖评估；Top1不再作为主要KPI。",
        },
        "topk_ladder": topk_ladder,
        "brier": {
            "average": average_loss,
            "uniform_baseline": float(baseline),
            "skill_score": (float(1.0 - average_loss / baseline) if average_loss is not None else None),
            "significance": brier_significance,
            "confirmed_better_than_uniform": brier_confirmed,
        },
        "coverage": {
            "ticket_count": min(max(int(ticket_count), 0), 16),
            "any_blue_hit_probability_with_distinct_blues": float(min(max(int(ticket_count), 0), 16) / 16.0),
        },
        "audit": audit,
        "ablation": _blue_ablation_summary(ranked_records, audit),
        "model_rows": model_rows,
        "patterns": _blue_sequence_patterns(history, trend_config),
        "randomness": _blue_randomness_summary(randomness_audit),
        "conclusion": (
            "融合蓝球概率在独立样本外同时通过概率损失检验；仍需继续以开奖前留痕复验。"
            if brier_confirmed
            else "当前没有足够的独立样本外证据证明蓝球存在可用规律；系统只保留不同蓝球的组合覆盖，不把遗漏、热度或短期命中当作规则。"
        ),
    }
