from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


RED_STAGE_KEYS = (
    ("model_top18", "model_top18_hits"),
    ("candidate_pool", "candidate_pool_hits"),
    ("method_layer", "method_layer_hits"),
    ("portfolio_layer", "portfolio_layer_hits"),
    ("final_layer", "final_layer_hits"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _bootstrap_mean_ci(values: list[float], *, seed: int = 2026, repeats: int = 1000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "samples": 0}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(int(repeats), len(array)))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "samples": int(len(array)),
    }


def _blue_rank(record: dict[str, Any]) -> int | None:
    ranking = (record.get("fused_validation") or {}).get("blue_ranking") or {}
    rank = ranking.get("rank") or ranking.get("mean_rank")
    try:
        value = int(round(float(rank)))
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 16 else None


def _prediction_set_width(record: dict[str, Any]) -> int:
    runtime = record.get("evidence_runtime_control") or record.get("runtime_control") or {}
    prediction = runtime.get("blue_prediction_set") if isinstance(runtime, dict) else {}
    width = None
    if isinstance(prediction, dict):
        width = prediction.get("selected_width")
        if width is None and isinstance(prediction.get("policy"), dict):
            topk = prediction["policy"].get("candidate_topk")
            if isinstance(topk, list) and topk:
                width = max(topk)
    return max(5, min(16, _safe_int(width, 10)))


def _consensus_blue_set(record: dict[str, Any]) -> set[int]:
    tickets = (((record.get("methods") or {}).get("consensus") or {}).get("tickets") or [])
    result: set[int] = set()
    for ticket in tickets:
        blue = _safe_int(ticket.get("blue"))
        if 1 <= blue <= 16:
            result.add(blue)
    return result


def _red_number_survival(record: dict[str, Any]) -> list[dict[str, Any]]:
    funnel = (record.get("candidate_recall_funnel") or {}).get("red") or {}
    actual = [_safe_int(value) for value in funnel.get("actual", []) or []]
    missed = {
        "model_top18": set(_safe_int(value) for value in funnel.get("missed_at_model_top18", []) or []),
        "candidate_pool": set(_safe_int(value) for value in funnel.get("missed_at_candidate_pool", []) or []),
        "method_layer": set(_safe_int(value) for value in funnel.get("missed_at_method_layer", []) or []),
        "portfolio_layer": set(_safe_int(value) for value in funnel.get("missed_at_portfolio_layer", []) or []),
        "final_layer": set(_safe_int(value) for value in funnel.get("missed_at_final_layer", []) or []),
    }
    rows: list[dict[str, Any]] = []
    for number in actual:
        survival = {
            "model_top18": number not in missed["model_top18"],
            "candidate_pool": number not in missed["candidate_pool"],
            "method_layer": number not in missed["method_layer"],
            "portfolio_layer": number not in missed["portfolio_layer"],
            "final_layer": number not in missed["final_layer"],
        }
        lost_at = "retained"
        for stage in ("model_top18", "candidate_pool", "method_layer", "portfolio_layer", "final_layer"):
            if not survival[stage]:
                lost_at = stage
                break
        rows.append({
            "number": number,
            "survival": survival,
            "lost_at": lost_at,
            "is_final_hit": bool(survival["final_layer"]),
        })
    return rows


def _blue_survival(record: dict[str, Any]) -> dict[str, Any]:
    funnel = (record.get("candidate_recall_funnel") or {}).get("blue") or {}
    actual = _safe_int(funnel.get("actual"))
    rank = _blue_rank(record)
    width = _prediction_set_width(record)
    final_blues = _consensus_blue_set(record)
    survival = {
        "top1": bool(funnel.get("model_hit_at_1")),
        "top3": bool(funnel.get("model_hit_at_3")),
        "top5": bool(funnel.get("model_hit_at_5")),
        "prediction_set": bool(rank is not None and rank <= width),
        "candidate_pool": bool(funnel.get("candidate_pool_hit")),
        "method_layer": bool(funnel.get("method_layer_hit")),
        "portfolio_layer": bool(funnel.get("portfolio_layer_hit")),
        "final_layer": bool(funnel.get("final_layer_hit")),
    }
    lost_at = "retained"
    for stage in ("prediction_set", "candidate_pool", "method_layer", "portfolio_layer", "final_layer"):
        if not survival[stage]:
            lost_at = stage
            break
    return {
        "number": actual,
        "rank": rank,
        "prediction_set_width": width,
        "final_blue_count": len(final_blues),
        "coverage_efficiency": len(final_blues) / float(width or 1),
        "survival": survival,
        "lost_at": lost_at,
    }


def _counterfactual_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_counter: Counter[str] = Counter()
    red_hit_rows: dict[str, list[float]] = {}
    blue_hit_rows: dict[str, list[float]] = {}
    gain_rows: dict[str, list[float]] = {}
    for record in records:
        experiment = record.get("compression_experiment") or {}
        strategies = experiment.get("strategies") if isinstance(experiment, dict) else None
        if not isinstance(strategies, dict):
            continue
        current_red = _safe_float(((strategies.get("current_final") or {}).get("metrics") or {}).get("best_red_hits"))
        for name, row in strategies.items():
            metrics = row.get("metrics") or {}
            red_hits = _safe_float(metrics.get("best_red_hits"))
            blue_hit = _safe_float(metrics.get("any_blue_hit"))
            red_hit_rows.setdefault(str(name), []).append(red_hits)
            blue_hit_rows.setdefault(str(name), []).append(blue_hit)
            gain_rows.setdefault(str(name), []).append(red_hits - current_red)
        best = max(
            strategies,
            key=lambda key: _safe_float(((strategies[key] or {}).get("metrics") or {}).get("best_red_hits")),
        )
        strategy_counter[str(best)] += 1
    return {
        "schema": "counterfactual_portfolio_audit_v1",
        "periods": sum(strategy_counter.values()),
        "best_red_strategy_counts": dict(strategy_counter),
        "strategy_rows": [
            {
                "strategy": strategy,
                "avg_best_red_hits": _mean(red_hit_rows.get(strategy, [])),
                "avg_any_blue_hit": _mean(blue_hit_rows.get(strategy, [])),
                "avg_red_gain_vs_current": _mean(gain_rows.get(strategy, [])),
            }
            for strategy in sorted(red_hit_rows)
        ],
    }


def compression_audit_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit where upstream information is lost before the final finite portfolio."""

    rows = [record for record in records or [] if isinstance(record.get("candidate_recall_funnel"), dict)]
    red_stage_values: dict[str, list[float]] = {name: [] for name, _ in RED_STAGE_KEYS}
    red_regrets = {
        "model_to_candidate": [],
        "candidate_to_method": [],
        "method_to_portfolio": [],
        "portfolio_to_final": [],
        "candidate_to_final": [],
    }
    survival_counter: Counter[str] = Counter()
    survival_examples: list[dict[str, Any]] = []
    blue_lost_counter: Counter[str] = Counter()
    blue_prediction_hits: list[float] = []
    blue_top5_hits: list[float] = []
    blue_final_hits: list[float] = []
    blue_efficiency: list[float] = []
    evidence_retention_rows: list[float] = []
    candidate_recall_proxy_rows: list[float] = []
    survival_value_retention_rows: list[float] = []
    portfolio_composite_irr_rows: list[float] = []
    true_candidate_to_final_retention_rows: list[float] = []
    true_candidate_to_final_loss_rows: list[float] = []

    for record in rows:
        red = (record.get("candidate_recall_funnel") or {}).get("red") or {}
        for stage, key in RED_STAGE_KEYS:
            red_stage_values[stage].append(_safe_float(red.get(key)) / 6.0)
        model_hits = _safe_float(red.get("model_top18_hits"))
        candidate_hits = _safe_float(red.get("candidate_pool_hits"))
        method_hits = _safe_float(red.get("method_layer_hits"))
        portfolio_hits = _safe_float(red.get("portfolio_layer_hits", red.get("final_layer_hits")))
        final_hits = _safe_float(red.get("final_layer_hits"))
        true_survival = red.get("true_survival") if isinstance(red.get("true_survival"), dict) else {}
        true_retention = true_survival.get("candidate_to_final_retention")
        true_loss = true_survival.get("candidate_to_final_loss")
        if true_retention is not None:
            true_candidate_to_final_retention_rows.append(max(0.0, min(1.0, _safe_float(true_retention))))
        if true_loss is not None:
            true_candidate_to_final_loss_rows.append(_safe_float(true_loss))
        red_regrets["model_to_candidate"].append(max(0.0, model_hits - candidate_hits))
        red_regrets["candidate_to_method"].append(max(0.0, candidate_hits - method_hits))
        red_regrets["method_to_portfolio"].append(max(0.0, method_hits - portfolio_hits))
        red_regrets["portfolio_to_final"].append(max(0.0, portfolio_hits - final_hits))
        red_regrets["candidate_to_final"].append(
            _safe_float(true_loss)
            if true_loss is not None
            else max(0.0, candidate_hits - final_hits)
        )
        number_rows = _red_number_survival(record)
        for item in number_rows:
            survival_counter[str(item["lost_at"])] += 1
        if len(survival_examples) < 12 and any(item["lost_at"] != "retained" for item in number_rows):
            survival_examples.append({
                "period": record.get("period"),
                "actual_red": red.get("actual", []),
                "numbers": number_rows,
            })
        blue = _blue_survival(record)
        blue_lost_counter[str(blue["lost_at"])] += 1
        blue_prediction_hits.append(1.0 if bool((blue.get("survival") or {}).get("prediction_set")) else 0.0)
        blue_top5_hits.append(1.0 if bool((blue.get("survival") or {}).get("top5")) else 0.0)
        blue_final_hits.append(1.0 if bool((blue.get("survival") or {}).get("final_layer")) else 0.0)
        blue_efficiency.append(_safe_float(blue.get("coverage_efficiency")))
        v7_portfolio = record.get("v7_portfolio") if isinstance(record.get("v7_portfolio"), dict) else {}
        retention = v7_portfolio.get("information_retention") if isinstance(v7_portfolio, dict) else {}
        if isinstance(retention, dict):
            if retention.get("evidence_retention") is not None:
                evidence_retention_rows.append(_safe_float(retention.get("evidence_retention")))
            if retention.get("candidate_recall_proxy") is not None:
                candidate_recall_proxy_rows.append(_safe_float(retention.get("candidate_recall_proxy")))
            if retention.get("survival_value_retention") is not None:
                survival_value_retention_rows.append(_safe_float(retention.get("survival_value_retention")))
            if retention.get("information_retention_rate") is not None:
                portfolio_composite_irr_rows.append(_safe_float(retention.get("information_retention_rate")))

    candidate_recall = _mean(red_stage_values["candidate_pool"])
    final_recall = _mean(red_stage_values["final_layer"])
    legacy_candidate_to_final_recall = min(1.0, final_recall / candidate_recall) if candidate_recall > 0 else 0.0
    candidate_to_final_recall = (
        _mean(true_candidate_to_final_retention_rows)
        if true_candidate_to_final_retention_rows
        else legacy_candidate_to_final_recall
    )
    candidate_to_final_recall = max(0.0, min(1.0, candidate_to_final_recall))
    blue_final_minus_random = [value - 5.0 / 16.0 for value in blue_final_hits]
    blue_final_minus_top5 = [
        final - top5
        for final, top5 in zip(blue_final_hits, blue_top5_hits)
    ]
    return {
        "schema": "compression_audit_v1",
        "periods": len(rows),
        "red": {
            "stage_recall": {
                stage: {
                    "mean": _mean(values),
                    "p10": _percentile(values, 10),
                    "p50": _percentile(values, 50),
                    "p90": _percentile(values, 90),
                }
                for stage, values in red_stage_values.items()
            },
            "evidence_retention": {
                "mean": _mean(evidence_retention_rows),
                "p10": _percentile(evidence_retention_rows, 10),
                "p50": _percentile(evidence_retention_rows, 50),
                "p90": _percentile(evidence_retention_rows, 90),
                "source": "portfolio_candidate_evidence",
            },
            "candidate_recall_proxy": {
                "mean": _mean(candidate_recall_proxy_rows),
                "p10": _percentile(candidate_recall_proxy_rows, 10),
                "p50": _percentile(candidate_recall_proxy_rows, 50),
                "p90": _percentile(candidate_recall_proxy_rows, 90),
                "source": "portfolio_candidate_number_survival_proxy",
            },
            "portfolio_composite_information_retention": {
                "mean": _mean(portfolio_composite_irr_rows),
                "p10": _percentile(portfolio_composite_irr_rows, 10),
                "p50": _percentile(portfolio_composite_irr_rows, 50),
                "p90": _percentile(portfolio_composite_irr_rows, 90),
            },
            "survival_value_retention": {
                "mean": _mean(survival_value_retention_rows),
                "p10": _percentile(survival_value_retention_rows, 10),
                "p50": _percentile(survival_value_retention_rows, 50),
                "p90": _percentile(survival_value_retention_rows, 90),
            },
            "candidate_to_final_recall": float(candidate_to_final_recall),
            "information_retention_rate": float(candidate_to_final_recall),
            "compression_regret": float(max(0.0, 1.0 - candidate_to_final_recall)),
            "true_survival": {
                "schema": "candidate_to_final_true_survival_summary_v8",
                "mean_retention": _mean(true_candidate_to_final_retention_rows),
                "mean_lost_true_numbers": _mean(true_candidate_to_final_loss_rows),
                "source": "candidate_recall_funnel.red.true_survival",
            },
            "regret_decomposition": {
                key: {
                    "mean_lost_true_numbers": _mean(values),
                    "total_lost_true_numbers": float(sum(values)),
                }
                for key, values in red_regrets.items()
            },
            "true_number_survival_counts": dict(survival_counter),
            "survival_examples": survival_examples,
        },
        "blue": {
            "prediction_set_hit": _mean(blue_prediction_hits),
            "top5_hit": _mean(blue_top5_hits),
            "final_set_hit": _mean(blue_final_hits),
            "coverage_efficiency": _mean(blue_efficiency),
            "benchmarks": {
                "random5": 5.0 / 16.0,
                "final5_minus_random5": _bootstrap_mean_ci(blue_final_minus_random, seed=2026),
                "final5_minus_top5": _bootstrap_mean_ci(blue_final_minus_top5, seed=2027),
                "decision": (
                    "PROVEN_EDGE"
                    if blue_final_minus_random
                    and _bootstrap_mean_ci(blue_final_minus_random, seed=2026)["ci95"][0] is not None
                    and _bootstrap_mean_ci(blue_final_minus_random, seed=2026)["ci95"][0] > 0
                    else "NO_PROVEN_EDGE"
                ),
            },
            "lost_stage_counts": dict(blue_lost_counter),
            "primary_metric": "Prediction Set Coverage + Final Set Hit; Top1 is diagnostic only.",
        },
        "counterfactual": _counterfactual_summary(rows),
        "interpretation": "该审计器专门定位候选层到最终有限注数之间的信息损失；命中低时先看哪一级Regret最大，再调对应约束。",
    }
