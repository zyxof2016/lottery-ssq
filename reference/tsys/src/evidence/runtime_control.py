from __future__ import annotations

from typing import Any

import numpy as np

from src.blue.edge_detector import blue_no_edge_report
from src.blue.probability_shrinkage import alpha_for_blue_edge_state
from src.portfolio.budget_allocator import allocate_portfolio_budget
from src.portfolio.compression_audit import compression_audit_report
from src.portfolio.regret import portfolio_regret_report
from src.red.edge_detector import red_edge_report
from src.red.recall_guard import red_recall_guard_report
from src.red.survival_tracker import red_survival_report
from src.uncertainty.prediction_set import blue_prediction_set_report


def _control_window(config: dict[str, Any]) -> int:
    v7 = (config.get("v7") or {}) if isinstance(config, dict) else {}
    evidence = (v7.get("evidence") or {}) if isinstance(v7, dict) else {}
    windows = [int(value) for value in evidence.get("windows", [30, 100, 300])]
    return max(1, min(max(windows or [100]), 300))


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    clean: dict[str, float] = {}
    passthrough: dict[str, Any] = {}
    for key, value in weights.items():
        if str(key).startswith("__"):
            passthrough[str(key)] = value
            continue
        try:
            clean[str(key)] = max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            passthrough[str(key)] = value
    total = float(sum(clean.values()))
    if total <= 0:
        return {**clean, **passthrough}
    return {**{key: float(value / total) for key, value in clean.items()}, **passthrough}


def _cap_single_weight(weights: dict[str, float], cap: float) -> dict[str, float]:
    cap = float(np.clip(cap, 0.0, 1.0))
    if cap <= 0:
        return _renormalize(weights)
    numeric: dict[str, float] = {}
    passthrough: dict[str, Any] = {}
    for key, value in weights.items():
        if str(key).startswith("__"):
            passthrough[str(key)] = value
            continue
        try:
            numeric[str(key)] = float(value or 0.0)
        except (TypeError, ValueError):
            passthrough[str(key)] = value
    safety_anchor = "uniform_baseline"
    capped = {
        key: (max(0.0, value) if key == safety_anchor else min(max(0.0, value), cap))
        for key, value in numeric.items()
    }
    overflow = max(0.0, 1.0 - sum(capped.values()))
    expandable = {
        key: max(0.0, cap - value)
        for key, value in capped.items()
        if key != safety_anchor and value < cap and numeric.get(key, 0.0) > 0.0
    }
    expandable_total = sum(expandable.values())
    if overflow > 0 and expandable_total > 0:
        redistributed = min(overflow, expandable_total)
        for key, room in expandable.items():
            capped[key] += redistributed * room / expandable_total
    overflow = max(0.0, 1.0 - sum(capped.values()))
    if overflow > 0:
        capped[safety_anchor] = max(0.0, capped.get(safety_anchor, 0.0)) + overflow
    return _renormalize({**capped, **passthrough})


def _mean_blue_hit_at(records: list[dict[str, Any]], k: int) -> float | None:
    values: list[float] = []
    key = f"hit_at_{int(k)}"
    for record in records or []:
        try:
            values.append(float(record["fused_validation"]["blue_ranking"][key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return float(np.mean(values))


def _red_final_recall(records: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for record in records or []:
        try:
            hits = float(record["candidate_recall_funnel"]["red"]["final_layer_hits"])
            values.append(hits / 6.0)
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return float(np.mean(values))


def _red_retention_stats(records: list[dict[str, Any]]) -> dict[str, float | None]:
    candidate_hits: list[float] = []
    final_hits: list[float] = []
    true_found_hits: list[float] = []
    true_final_hits: list[float] = []
    true_losses: list[float] = []
    period_retention: list[float] = []
    for record in records or []:
        try:
            red = record["candidate_recall_funnel"]["red"]
            candidate = float(red.get("candidate_pool_hits", 0.0) or 0.0)
            final = float(red.get("final_layer_hits", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        candidate_hits.append(candidate)
        final_hits.append(final)
        survival = red.get("true_survival") if isinstance(red, dict) else None
        if isinstance(survival, dict):
            true_found = float(survival.get("candidate_found_count", 0.0) or 0.0)
            true_final = float(survival.get("final_retained_count", 0.0) or 0.0)
            true_loss = float(survival.get("candidate_to_final_loss", max(0.0, true_found - true_final)) or 0.0)
            true_retention = survival.get("candidate_to_final_retention")
            true_found_hits.append(true_found)
            true_final_hits.append(true_final)
            true_losses.append(max(0.0, true_loss))
            if true_retention is not None:
                try:
                    period_retention.append(max(0.0, min(1.0, float(true_retention))))
                except (TypeError, ValueError):
                    pass
        elif candidate > 0:
            # Legacy rows had only aggregate hit counts.  Clamp here because
            # final union hits can exceed candidate hits when Final finds
            # true numbers outside the explicit Candidate Set; that is not
            # Candidate→Final survival and must not create >100% retention.
            period_retention.append(max(0.0, min(1.0, final / candidate)))
    candidate_mean = float(np.mean(candidate_hits)) if candidate_hits else None
    final_mean = float(np.mean(final_hits)) if final_hits else None
    if true_found_hits and sum(true_found_hits) > 0:
        retention = max(0.0, min(1.0, float(sum(true_final_hits) / sum(true_found_hits))))
        compression_regret = float(np.mean(true_losses)) / 6.0 if true_losses else float(1.0 - retention)
        true_candidate_mean = float(np.mean(true_found_hits))
        true_final_mean = float(np.mean(true_final_hits))
    elif candidate_hits and sum(candidate_hits) > 0:
        retention = max(0.0, min(1.0, float(sum(min(c, f) for c, f in zip(candidate_hits, final_hits)) / sum(candidate_hits))))
        compression_regret = float(1.0 - retention)
        true_candidate_mean = None
        true_final_mean = None
    else:
        retention = None
        compression_regret = None
        true_candidate_mean = None
        true_final_mean = None
    return {
        "candidate_recall": None if candidate_mean is None else candidate_mean / 6.0,
        "final_recall": None if final_mean is None else final_mean / 6.0,
        "true_candidate_recall": None if true_candidate_mean is None else true_candidate_mean / 6.0,
        "true_final_retained_recall": None if true_final_mean is None else true_final_mean / 6.0,
        "candidate_to_final_retention": retention,
        "compression_regret": compression_regret,
        "period_retention_mean": float(np.mean(period_retention)) if period_retention else None,
    }


def _regret_diagnostic(regret: dict[str, Any]) -> dict[str, Any]:
    ratios = regret.get("stage_ratios") if isinstance(regret, dict) else {}
    model = float((ratios or {}).get("model_or_feature_loss", 0.0) or 0.0)
    method = float((ratios or {}).get("method_optimizer_loss", 0.0) or 0.0)
    portfolio = float((ratios or {}).get("final_portfolio_loss", 0.0) or 0.0)
    dominant = max(
        (("model_regret", model), ("method_regret", method), ("portfolio_regret", portfolio)),
        key=lambda item: item[1],
    )[0]
    return {
        "schema": "regret_diagnostic_v7",
        "model_regret": model,
        "method_regret": method,
        "portfolio_regret": portfolio,
        "dominant": dominant,
        "control_mode": "diagnostic_only",
        "would_have_actions": {
            "model_regret": "would_lower_single_expert_cap_and_expand_prediction_set",
            "method_regret": "would_prefer_portfolio_strategy_with_more_coverage_slots",
            "portfolio_regret": "would_activate_recall_guard_tighten_jaccard_and_protect_lost_numbers",
        },
        "rule": "Regret is recorded for diagnosis only and does not automatically change model weights or runtime strategy.",
    }


def _strategy_candidates(
    config: dict[str, Any],
    *,
    blue_prediction: dict[str, Any],
    fallback_state: str,
) -> list[dict[str, Any]]:
    v7 = (config.get("v7") or {}) if isinstance(config, dict) else {}
    nested = (v7.get("nested_oos") or {}) if isinstance(v7, dict) else {}
    configured = nested.get("strategy_candidates")
    if isinstance(configured, list) and configured:
        return [dict(item) for item in configured if isinstance(item, dict)]
    widths = [int(value) for value in (blue_prediction.get("candidate_width_range") or [10, 12, 14, 16])]
    wide = max(widths or [16])
    mid = min(widths or [10], key=lambda value: abs(value - 10))
    narrow = min(widths or [5], key=lambda value: abs(value - 5))
    return [
        {
            "name": "Strategy A - coverage_guard",
            "prediction_set_k": wide,
            "blue_alpha": 0.08,
            "red_budget_state": "RECALL_GUARD",
            "jaccard_max_overlap": 3,
            "expert_cap": 0.35,
            "portfolio_strategy": "pareto_recall_guard",
        },
        {
            "name": "Strategy B - balanced",
            "prediction_set_k": mid,
            "blue_alpha": 0.22,
            "red_budget_state": "NO_EDGE" if fallback_state == "NO_EDGE" else "WEAK_EDGE",
            "jaccard_max_overlap": 4,
            "expert_cap": 0.42,
            "portfolio_strategy": "pareto_balanced",
        },
        {
            "name": "Strategy C - probability",
            "production_eligible": False,
            "lifecycle": "shadow_challenger",
            "prediction_set_k": narrow,
            "blue_alpha": 0.45,
            "red_budget_state": "STRONG_EDGE",
            "jaccard_max_overlap": 5,
            "expert_cap": 0.55,
            "portfolio_strategy": "pareto_probability",
        },
        {
            "name": "Strategy D - exploration",
            "prediction_set_k": wide,
            "blue_alpha": 0.12,
            "red_budget_state": "NO_EDGE",
            "jaccard_max_overlap": 3,
            "expert_cap": 0.35,
            "portfolio_strategy": "pareto_coverage",
        },
    ]


def _red_budget_rank(state: str) -> int:
    return {
        "NO_EDGE": 0,
        "RECALL_GUARD": 1,
        "WEAK_EDGE": 2,
        "STRONG_EDGE": 3,
    }.get(str(state or "NO_EDGE").upper(), 0)


def _is_red_budget_allowed(edge_state: str, budget_state: str) -> bool:
    edge = str(edge_state or "NO_EDGE").upper()
    budget = str(budget_state or "NO_EDGE").upper()
    if budget == "RECALL_GUARD":
        return True
    if edge == "NO_EDGE":
        return budget == "NO_EDGE"
    if edge == "WEAK_EDGE":
        return budget in {"NO_EDGE", "WEAK_EDGE"}
    if edge == "STRONG_EDGE":
        return budget in {"NO_EDGE", "WEAK_EDGE", "STRONG_EDGE"}
    return _red_budget_rank(budget) <= _red_budget_rank(edge)


def _clamp_red_budget_to_edge(
    edge_state: str,
    budget_state: str,
    *,
    guard_active: bool,
) -> tuple[str, dict[str, Any]]:
    edge = str(edge_state or "NO_EDGE").upper()
    requested = str(budget_state or "NO_EDGE").upper()
    if _is_red_budget_allowed(edge, requested):
        return requested, {
            "active": False,
            "red_edge_state": edge,
            "requested_red_budget_state": requested,
            "final_red_budget_state": requested,
            "reason": "red_budget_state_allowed_by_red_edge_state",
        }
    final_state = "RECALL_GUARD" if guard_active else ("WEAK_EDGE" if edge == "WEAK_EDGE" else "NO_EDGE")
    return final_state, {
        "active": True,
        "red_edge_state": edge,
        "requested_red_budget_state": requested,
        "final_red_budget_state": final_state,
        "reason": "red_budget_state_was_more_aggressive_than_red_edge_state_and_was_clamped",
    }


def _score_strategy(
    records: list[dict[str, Any]],
    strategy: dict[str, Any],
    regret_diagnostic: dict[str, Any],
    counterfactual_preference: dict[str, Any] | None = None,
    red_edge_state: str = "NO_EDGE",
    blue_edge_state: str = "NO_EDGE",
    allowed_widths: list[int] | None = None,
) -> dict[str, Any]:
    k = int(strategy.get("prediction_set_k", 12) or 12)
    blue_hit = _mean_blue_hit_at(records, k)
    red_recall = _red_final_recall(records)
    baseline = float(k) / 16.0
    if blue_hit is None:
        blue_score = 0.0
    else:
        # Penalise oversized sets. K=16 must not win merely because it covers
        # every actual blue in the prediction set.
        blue_score = (float(blue_hit) - baseline) / float(np.sqrt(max(k, 1)))
    blue_score -= 0.010 * max(0, k - 10)
    red_score = float(red_recall if red_recall is not None else 0.0)
    jaccard = int(strategy.get("jaccard_max_overlap", 4) or 4)
    coverage_bonus = 0.04 if jaccard <= 3 else 0.0
    probability_bonus = 0.025 if "probability" in str(strategy.get("portfolio_strategy", "")) else 0.0
    red_budget_state = str(strategy.get("red_budget_state", "NO_EDGE")).upper()
    rejected_reasons: list[str] = []
    if strategy.get("production_eligible") is False:
        rejected_reasons.append("strategy_lifecycle_shadow_challenger_only")
    if not _is_red_budget_allowed(red_edge_state, red_budget_state):
        rejected_reasons.append("red_budget_state_exceeds_red_edge_state")
    if str(blue_edge_state).upper() == "NO_EDGE":
        if allowed_widths and int(strategy.get("prediction_set_k", k) or k) not in set(allowed_widths):
            rejected_reasons.append("blue_no_edge_prediction_set_k_outside_coverage_range")
        if float(strategy.get("blue_alpha", 0.0) or 0.0) > 0.22:
            rejected_reasons.append("blue_no_edge_alpha_too_aggressive")
    if rejected_reasons:
        total = -999.0
        return {
            "name": str(strategy.get("name", "strategy")),
            "score": float(total),
            "blue_component": float(blue_score),
            "red_component": float(red_score),
            "regret_bonus": 0.0,
            "counterfactual_bonus": 0.0,
            "counterfactual_strategy": "",
            "regret_mode": str(regret_diagnostic.get("control_mode", "diagnostic_only")),
            "regret_dominant_observed_only": str(regret_diagnostic.get("dominant", "unknown")),
            "strategy": dict(strategy),
            "eligible": False,
            "rejected_reasons": rejected_reasons,
        }
    mapped_counterfactual = {
        "pareto_recall_guard": "recall_first",
        "pareto_coverage": "coverage_first",
        "pareto_balanced": "portfolio_stage",
        "pareto_probability": "probability_first",
    }.get(str(strategy.get("portfolio_strategy", "")), "")
    cf_rows = (counterfactual_preference or {}).get("strategy_gain", {}) if isinstance(counterfactual_preference, dict) else {}
    counterfactual_bonus = float(cf_rows.get(mapped_counterfactual, 0.0) or 0.0)
    counterfactual_bonus = max(-0.08, min(0.08, 0.035 * counterfactual_bonus))
    regret_bonus = 0.0
    total = 0.46 * blue_score + 0.42 * red_score + coverage_bonus + probability_bonus + counterfactual_bonus
    return {
        "name": str(strategy.get("name", "strategy")),
        "score": float(total),
        "blue_component": float(blue_score),
        "red_component": float(red_score),
        "regret_bonus": float(regret_bonus),
        "counterfactual_bonus": float(counterfactual_bonus),
        "counterfactual_strategy": mapped_counterfactual,
        "regret_mode": str(regret_diagnostic.get("control_mode", "diagnostic_only")),
        "regret_dominant_observed_only": str(regret_diagnostic.get("dominant", "unknown")),
        "strategy": dict(strategy),
        "eligible": True,
        "rejected_reasons": [],
    }


def _counterfactual_preference(records: list[dict[str, Any]]) -> dict[str, Any]:
    gains: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    for record in records or []:
        experiment = record.get("compression_experiment") if isinstance(record, dict) else None
        strategies = experiment.get("strategies") if isinstance(experiment, dict) else None
        if not isinstance(strategies, dict):
            continue
        current = (strategies.get("current_final") or {}).get("metrics") or {}
        current_red = float(current.get("best_red_hits", 0.0) or 0.0)
        best_name = None
        best_value = -1.0
        for name, row in strategies.items():
            metrics = (row or {}).get("metrics") or {}
            red_hits = float(metrics.get("best_red_hits", 0.0) or 0.0)
            blue_hit = float(metrics.get("any_blue_hit", 0.0) or 0.0)
            value = red_hits + 0.18 * blue_hit
            gains.setdefault(str(name), []).append(red_hits - current_red)
            if value > best_value:
                best_value = value
                best_name = str(name)
        if best_name:
            wins[best_name] = wins.get(best_name, 0) + 1
    strategy_gain = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in gains.items()
    }
    best = max(strategy_gain, key=lambda name: (strategy_gain[name], wins.get(name, 0)), default=None)
    return {
        "schema": "counterfactual_preference_v7_1",
        "periods": sum(len(values) for values in gains.values()),
        "strategy_gain": strategy_gain,
        "win_counts": wins,
        "best_strategy": best,
        "rule": "Counterfactual只影响下一窗口Portfolio策略选择，不反向修改模型权重。",
    }


def _inner_oos_control_search(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    red_budget_state: str,
    blue_state: str,
    blue_prediction: dict[str, Any],
    red_guard_active: bool,
    regret_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose runtime knobs from completed inner OOS rows only.

    This is intentionally conservative: the search uses already-completed walk-forward
    records and only outputs controls for the next target period. Outer OOS rows remain
    evaluation-only.
    """
    v7 = (config.get("v7") or {}) if isinstance(config, dict) else {}
    nested = (v7.get("nested_oos") or {}) if isinstance(v7, dict) else {}
    if not bool(nested.get("inner_search", True)):
        selected_width = int(blue_prediction.get("selected_width", 12) or 12)
        return {
            "schema": "nested_inner_oos_control_v7",
            "active": False,
            "reason": "inner_search_disabled",
            "selected": {
                "prediction_set_k": selected_width,
                "blue_alpha": alpha_for_blue_edge_state(blue_state, config),
                "red_budget_state": red_budget_state,
                "jaccard_max_overlap": 4,
                "expert_cap": 0.42,
                "portfolio_strategy": "pareto_coverage",
            },
        }
    window = max(1, int(nested.get("selection_window", 100)))
    inner_rows = list(records or [])[-window:]
    red_retention = _red_retention_stats(inner_rows)
    diagnostic = regret_diagnostic or {"dominant": "unknown", "control_mode": "diagnostic_only"}
    counterfactual_preference = _counterfactual_preference(inner_rows)
    configured_width_grid = [
        int(value)
        for value in nested.get("prediction_set_k_grid", (blue_prediction.get("candidate_width_range") or [12]))
        if 1 <= int(value) <= 16
    ] or [int(blue_prediction.get("selected_width", 12) or 12)]
    allowed_widths = [
        int(value)
        for value in (blue_prediction.get("candidate_width_range") or [])
        if 1 <= int(value) <= 16
    ]
    width_grid = [value for value in configured_width_grid if not allowed_widths or value in allowed_widths]
    if not width_grid:
        width_grid = allowed_widths or configured_width_grid
    alpha_grid = [float(value) for value in nested.get("blue_alpha_grid", [0.08, 0.22, 0.45, 0.70])]
    selected_width = int(blue_prediction.get("selected_width", max(width_grid)) or max(width_grid))
    best_width = selected_width if selected_width in width_grid else max(width_grid)
    best_width_score = -1.0
    for width in width_grid:
        hit = _mean_blue_hit_at(inner_rows, width)
        if hit is None:
            continue
        coverage_penalty = 0.006 * float(width - 5)
        baseline = float(width) / 16.0
        score = (float(hit) - baseline) / float(np.sqrt(max(width, 1))) - max(0.0, coverage_penalty)
        if score > best_width_score:
            best_width = int(width)
            best_width_score = score
    minimum_blue_periods = int(((v7.get("blue") or {}).get("minimum_edge_periods", 30)) if isinstance(v7, dict) else 30)
    if len(inner_rows) < minimum_blue_periods and str(blue_state).upper() == "NO_EDGE":
        best_width = max(width_grid)
        best_width_score = -1.0
    blue_hit5 = _mean_blue_hit_at(inner_rows, 5)
    no_edge_blue = str(blue_state).upper() == "NO_EDGE"
    if blue_hit5 is None:
        selected_alpha = alpha_for_blue_edge_state(blue_state, config)
    elif no_edge_blue or float(blue_hit5) <= (5.0 / 16.0):
        selected_alpha = min(alpha_grid)
    else:
        selected_alpha = min(alpha_grid, key=lambda value: abs(value - alpha_for_blue_edge_state(blue_state, config)))
    fallback_state = "RECALL_GUARD" if red_guard_active else str(red_budget_state or "NO_EDGE").upper()
    red_edge_final_state = str(
        red_edge_report(inner_rows, config).get(
            "final_state",
            "RECALL_GUARD" if str(red_budget_state).upper() == "RECALL_GUARD" else "NO_EDGE",
        )
    ).upper()
    strategy_scores = [
        _score_strategy(
            inner_rows,
            strategy,
            diagnostic,
            counterfactual_preference,
            red_edge_state=red_edge_final_state,
            blue_edge_state=blue_state,
            allowed_widths=allowed_widths,
        )
        for strategy in _strategy_candidates(config, blue_prediction=blue_prediction, fallback_state=fallback_state)
    ]
    eligible_strategy_scores = [item for item in strategy_scores if item.get("eligible", True)]
    best_strategy = max(eligible_strategy_scores, key=lambda item: item["score"], default=None)
    if not inner_rows and not red_guard_active:
        cold_start_candidates = [
            item for item in eligible_strategy_scores
            if str((item.get("strategy") or {}).get("red_budget_state", "")).upper() == "NO_EDGE"
            and "coverage" in str((item.get("strategy") or {}).get("portfolio_strategy", ""))
        ]
        if cold_start_candidates:
            best_strategy = max(cold_start_candidates, key=lambda item: item["score"], default=best_strategy)
    red_cfg = ((v7.get("red") or {}) if isinstance(v7, dict) else {})
    retention_floor = float(red_cfg.get("retention_guard_minimum", 0.80) or 0.80)
    compression_ceiling = float(red_cfg.get("retention_guard_compression_regret_max", 1.0 - retention_floor) or (1.0 - retention_floor))
    red_final_floor = float(red_cfg.get("retention_guard_final_recall_minimum", 0.50) or 0.50)
    retention_low = (
        red_retention.get("candidate_to_final_retention") is not None
        and float(red_retention.get("candidate_to_final_retention") or 0.0) < retention_floor
    )
    compression_high = (
        red_retention.get("compression_regret") is not None
        and float(red_retention.get("compression_regret") or 0.0) > compression_ceiling
    )
    final_recall_low = (
        red_retention.get("final_recall") is not None
        and float(red_retention.get("final_recall") or 0.0) < red_final_floor
    )
    retention_guard_forced = bool(red_guard_active or retention_low or compression_high or final_recall_low)
    if retention_guard_forced:
        coverage_candidates = [
            item for item in strategy_scores
            if str((item.get("strategy") or {}).get("red_budget_state", "")).upper() == "RECALL_GUARD"
            or "coverage" in str((item.get("strategy") or {}).get("portfolio_strategy", ""))
            or "recall" in str((item.get("strategy") or {}).get("portfolio_strategy", ""))
        ]
        coverage_candidates = [item for item in coverage_candidates if item.get("eligible", True)]
        best_strategy = max(coverage_candidates, key=lambda item: item["score"], default=best_strategy)
    selected_strategy = dict((best_strategy or {}).get("strategy") or {})
    state = str(selected_strategy.get("red_budget_state") or fallback_state).upper()
    state, budget_clamp = _clamp_red_budget_to_edge(
        red_edge_final_state,
        state,
        guard_active=retention_guard_forced,
    )
    jaccard_map = nested.get("red_jaccard_max_overlap_by_state") or {}
    cap_map = nested.get("expert_cap_by_state") or {}
    strategy_map = nested.get("portfolio_strategy_by_state") or {}
    if jaccard_map and state not in jaccard_map:
        state = fallback_state if fallback_state in jaccard_map else str(next(iter(jaccard_map)))
    red_recall = _red_final_recall(inner_rows)
    jaccard = int(selected_strategy.get("jaccard_max_overlap", jaccard_map.get(state, jaccard_map.get("NO_EDGE", 4))))
    if jaccard_map:
        allowed_jaccard = {int(value) for value in jaccard_map.values()}
        if jaccard not in allowed_jaccard:
            jaccard = int(jaccard_map.get(state, min(allowed_jaccard)))
    if red_recall is not None and red_recall < red_final_floor:
        jaccard = min(jaccard, 3)
        state = "RECALL_GUARD"
    if retention_guard_forced:
        jaccard = min(jaccard, 3)
        state = "RECALL_GUARD"
    chosen_width = int(selected_strategy.get("prediction_set_k", best_width) or best_width)
    if chosen_width not in width_grid:
        chosen_width = best_width
    if len(inner_rows) < minimum_blue_periods and str(blue_state).upper() == "NO_EDGE":
        chosen_width = max(width_grid)
    return {
        "schema": "nested_inner_oos_control_v7",
        "active": True,
        "inner_rows": len(inner_rows),
        "selection_window": window,
        "metrics": {
            "blue_hit_at_5": blue_hit5,
            "red_final_recall": red_recall,
            "red_retention": red_retention,
            "best_prediction_set_k_score": None if best_width_score < 0 else float(best_width_score),
        },
        "searched": {
            "prediction_set_k": width_grid,
            "blue_alpha": alpha_grid,
            "red_budget": ["STRONG_EDGE", "WEAK_EDGE", "NO_EDGE", "RECALL_GUARD"],
            "jaccard_penalty": sorted({int(value) for value in jaccard_map.values()} or {3, 4, 5}),
            "expert_cap": sorted({float(value) for value in cap_map.values()} or {0.35, 0.42, 0.55, 0.70}),
            "portfolio_strategy": sorted({str(value) for value in strategy_map.values()} or {"pareto_coverage"}),
            "production_rule": (
                "red_budget_state不得强于red_edge_state；标记为shadow_challenger的集中策略只参与对照评分，"
                "不得成为生产最终Portfolio。"
            ),
        },
        "strategy_comparison": {
            "executed": True,
            "rule": "Strategy A/B/C/D are scored on completed Inner OOS rows; the best controls the next window.",
            "scores": strategy_scores,
            "best": best_strategy,
            "selected": (best_strategy or {}).get("name"),
            "counterfactual_preference": counterfactual_preference,
            "selection_reason": {
                "score_rule": "0.46*blue_component + 0.42*red_component + coverage/probability strategy bonus",
                "blue_component": "(Hit@K - K/16) / sqrt(K) with explicit large-set penalty",
                "counterfactual_usage": "completed Inner OOS compression experiments can add a small next-window portfolio strategy bonus",
                "regret_usage": "diagnostic_only_not_scoring_input",
                "red_retention_guard": {
                    "forced": retention_guard_forced,
                    "retention_floor": retention_floor,
                    "compression_ceiling": compression_ceiling,
                    "final_recall_floor": red_final_floor,
                    "retention_low": retention_low,
                    "compression_high": compression_high,
                    "final_recall_low": final_recall_low,
                    "rule": "Compression Regret高或Final Recall低时，优先切Coverage/RECALL_GUARD；集中策略只能作为Inner OOS证明后的候选。",
                },
                "red_budget_clamp": budget_clamp,
            },
        },
        "selected": {
            "prediction_set_k": int(chosen_width),
            "blue_alpha": float(selected_strategy.get("blue_alpha", selected_alpha)),
            "red_budget_state": state,
            "jaccard_max_overlap": int(jaccard),
            "expert_cap": float(selected_strategy.get("expert_cap", cap_map.get(state, cap_map.get("NO_EDGE", 0.42)))),
            "portfolio_strategy": str(selected_strategy.get("portfolio_strategy", strategy_map.get(state, strategy_map.get("NO_EDGE", "pareto_coverage")))),
        },
        "outer_oos_rule": "Outer OOS rows are never used here; they are reserved for release evidence.",
    }


def build_evidence_runtime_control(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    ticket_count: int,
) -> dict[str, Any]:
    """Single V7 runtime controller for weights, alpha, widths, and budget."""
    records = list(period_records or [])[-_control_window(config) :]
    v7 = (config.get("v7") or {}) if isinstance(config, dict) and isinstance(config.get("v7"), dict) else {}
    performance = config.get("performance", {}) if isinstance(config.get("performance", {}), dict) else {}
    refresh = performance.get("research_refresh", {}) if isinstance(performance.get("research_refresh", {}), dict) else {}
    refresh_enabled = bool(refresh.get("enabled", False))
    refresh_interval = max(1, int(refresh.get("nested_inner_oos", 1) or 1))
    previous_control = (
        ((records[-1].get("stacking") or {}).get("v7_runtime_control") or {})
        if records and isinstance(records[-1], dict)
        else {}
    )
    previous_nested = previous_control.get("nested_inner_oos") if isinstance(previous_control, dict) else None
    use_locked_nested = bool(
        refresh_enabled
        and isinstance(previous_nested, dict)
        and previous_nested.get("active")
        and len(records) > 0
        and len(records) % refresh_interval != 0
    )
    red_edge = red_edge_report(records, config)
    blue_edge = blue_no_edge_report(records, config)
    blue_prediction = blue_prediction_set_report(records, config)
    red_guard = red_recall_guard_report(records)
    regret = portfolio_regret_report(records)
    compression_audit = compression_audit_report(records)
    regret_diagnostic = _regret_diagnostic(regret)
    survival = red_survival_report(records)
    from src.governance.expert_decorrelation import expert_decorrelation_report

    decorrelation = expert_decorrelation_report({"period_records": records}, config)
    stage_ratios = regret.get("stage_ratios") or {}
    final_loss_ratio = float(stage_ratios.get("final_portfolio_loss", 0.0) or 0.0)
    red_cfg = ((config.get("v7") or {}).get("red") or {}) if isinstance(config, dict) else {}
    guard_threshold = float(red_cfg.get("recall_guard_final_loss_ratio", 0.20))
    compression_threshold = float(red_cfg.get("recall_guard_compression_loss", 0.85))
    guard_active = bool(
        red_guard.get("active")
        or final_loss_ratio >= guard_threshold
        or float(red_guard.get("candidate_to_final_loss", 0.0) or 0.0) >= compression_threshold
    )
    red_budget_state = "RECALL_GUARD" if guard_active else str(red_edge.get("final_state", "NO_EDGE"))
    protected_numbers = [
        int(item["number"])
        for item in (survival.get("top_lost_numbers") or [])[: int(red_cfg.get("survival_protected_count", 6))]
        if int(item.get("final_miss_count", 0) or 0) > 0
    ]
    blue_state = str(blue_edge.get("final_state", "NO_EDGE"))
    if use_locked_nested:
        nested_control = {
            **dict(previous_nested or {}),
            "research_refresh_lock": {
                "enabled": True,
                "interval": refresh_interval,
                "records_seen": len(records),
                "source": "previous_completed_window_control",
                "rule": "性能实验开关：非刷新点复用上一已完成窗口的Inner OOS选择；默认关闭，需A/B验证后才能作为生产默认。",
            },
        }
    else:
        nested_control = _inner_oos_control_search(
            records,
            config,
            red_budget_state=red_budget_state,
            blue_state=blue_state,
            blue_prediction=blue_prediction,
            red_guard_active=guard_active,
            regret_diagnostic=regret_diagnostic,
        )
    selected_control = nested_control.get("selected") if isinstance(nested_control.get("selected"), dict) else {}
    red_edge_state = str(red_edge.get("final_state", "NO_EDGE"))
    requested_red_budget_state = str(selected_control.get("red_budget_state", red_budget_state))
    red_budget_state, final_budget_clamp = _clamp_red_budget_to_edge(
        red_edge_state,
        requested_red_budget_state,
        guard_active=guard_active,
    )
    effective_guard_active = bool(guard_active or str(red_budget_state).upper() == "RECALL_GUARD")
    guard_sources = []
    if guard_active:
        guard_sources.append("red_recall_or_compression_guard_triggered")
    if str(red_budget_state).upper() == "RECALL_GUARD":
        guard_sources.append("inner_oos_selected_recall_guard_budget")
    red_budget = allocate_portfolio_budget(red_budget_state, ticket_count, config)
    production_blue_width = int(selected_control.get("prediction_set_k", blue_prediction.get("selected_width", 12)) or 12)
    production_blue_prediction = {
        **blue_prediction,
        "selected_width": production_blue_width,
        "policy": {
            **(blue_prediction.get("policy") or {}),
            "selected_width": production_blue_width,
            "runtime_source": "nested_inner_oos",
        },
    }
    base_expert_cap = float(selected_control.get("expert_cap", 0.42))
    evidence_discount = float((decorrelation.get("effective_evidence") or {}).get("evidence_discount", 1.0) or 1.0)
    decorrelated_cap = min(base_expert_cap, max(0.30, base_expert_cap * (0.70 + 0.30 * evidence_discount)))
    red_state_different = red_edge_state.upper() != red_budget_state.upper()
    red_state_mismatch = _red_budget_rank(red_budget_state) > _red_budget_rank(red_edge_state)
    red_state_alignment = {
        "schema": "red_edge_budget_alignment_v7_1",
        "red_edge_state": red_edge_state,
        "red_budget_state": red_budget_state,
        "requested_red_budget_state": requested_red_budget_state,
        "final_budget_clamp": final_budget_clamp,
        "state_different": red_state_different,
        "mismatch": red_state_mismatch,
        "problematic_mismatch": red_state_mismatch,
        "interpretation": (
            "Budget强于Edge，属于风险错配；必须查看Inner OOS分数、Clamp和选择原因。"
            if red_state_mismatch
            else "Edge与Budget不同，但Budget未强于Edge，属于保守降级/覆盖优先。"
            if red_state_different
            else "红球证据状态与预算状态一致。"
        ),
        "inner_oos_strategy_scores": (
            ((nested_control.get("strategy_comparison") or {}).get("scores") or [])
            if red_state_different
            else []
        ),
        "selection_reason": (
            ((nested_control.get("strategy_comparison") or {}).get("selection_reason") or {})
            if red_state_different
            else {}
        ),
    }
    return {
        "schema": "evidence_runtime_control_v7",
        "periods": len(records),
        "red_edge": red_edge,
        "blue_edge": blue_edge,
        "red_edge_state": red_edge_state,
        "blue_edge_state": blue_state,
        "blue_alpha": float(selected_control.get("blue_alpha", alpha_for_blue_edge_state(blue_state, config))),
        "blue_prediction_set": production_blue_prediction,
        "blue_candidate_width": production_blue_width,
        "red_recall_guard": red_guard,
        "regret": regret,
        "compression_audit": compression_audit,
        "regret_diagnostic": regret_diagnostic,
        "regret_feedback": {
            **regret_diagnostic,
            "schema": "regret_learning_feedback_v7_deprecated_diagnostic_only",
        },
        "survival": survival,
        "red_guard_active": effective_guard_active,
        "red_guard_trigger_sources": guard_sources or ["not_active"],
        "red_budget_state": red_budget_state,
        "red_budget": red_budget,
        "red_state_alignment": red_state_alignment,
        "red_jaccard_max_overlap": int(selected_control.get("jaccard_max_overlap", 4) or 4),
        "portfolio_strategy": str(selected_control.get("portfolio_strategy", "pareto_coverage")),
        "expert_decorrelation": decorrelation,
        "expert_cap_by_color": {
            "red": float(decorrelated_cap),
            "blue": float(decorrelated_cap),
        },
        "red_protected_numbers": protected_numbers,
        "final_v7_guard": dict(((v7.get("final_guard") or {}) if isinstance(v7, dict) else {})),
        "final_v7_guard_mode": str(((v7.get("final_guard") or {}) if isinstance(v7, dict) else {}).get("mode", "corrective")),
        "final_v7_guard_hard_constraints_only": bool(
            ((v7.get("final_guard") or {}) if isinstance(v7, dict) else {}).get("hard_constraints_only", False)
        ),
        "nested_inner_oos": nested_control,
        "performance_policy": {
            "safe_mode": bool(performance.get("safe_mode", True)),
            "research_refresh_enabled": refresh_enabled,
            "nested_inner_oos_refresh_interval": refresh_interval,
            "nested_inner_oos_locked_this_round": use_locked_nested,
            "rule": "默认关闭会改变研究策略刷新节奏的优化；开启前需用prediction_fingerprint与300期OOS对照确认。",
        },
        "prediction_set_contract": {
            "edge_state_to_search_range": blue_prediction.get("candidate_width_range"),
            "prediction_set_final_k": production_blue_width,
            "rule": "Edge State only defines the search range; Prediction Set selected_width is the sole production K.",
        },
        "reason": "Evidence Engine controls runtime model weight caps, blue alpha, blue candidate width and red portfolio budget; red compression guard can force Coverage Backbone, while regret remains diagnostic-only for model weights.",
    }


def apply_evidence_weight_control(
    weights_by_color: dict[str, dict[str, float]],
    runtime_control: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    v7 = (config.get("v7") or {}) if isinstance(config, dict) else {}
    evidence = (v7.get("evidence") or {}) if isinstance(v7, dict) else {}
    caps = evidence.get("single_model_cap_by_edge_state") or {
        "STRONG_EDGE": 0.70,
        "WEAK_EDGE": 0.55,
        "NO_EDGE": 0.42,
        "RECALL_GUARD": 0.35,
    }
    runtime_caps = runtime_control.get("expert_cap_by_color") if isinstance(runtime_control, dict) else {}
    result = {color: dict(values) for color, values in weights_by_color.items()}
    applied: dict[str, Any] = {"schema": "evidence_weight_control_v7", "applied": False, "colors": {}}
    for color, state_key in (("red", "red_budget_state"), ("blue", "blue_edge_state")):
        state = str(runtime_control.get(state_key, "NO_EDGE"))
        cap = float((runtime_caps or {}).get(color, caps.get(state, caps.get("NO_EDGE", 0.42))))
        before = result.get(color, {})
        after = _cap_single_weight(before, cap)
        result[color] = after
        applied["colors"][color] = {
            "state": state,
            "single_model_cap": cap,
            "changed": after != before,
        }
        applied["applied"] = bool(applied["applied"] or after != before)
    return result, applied
