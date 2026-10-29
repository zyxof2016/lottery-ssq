from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.blue.specialist_gateway import finalize_blue_portfolio
from src.blue.probability_shrinkage import shrink_blue_probability_to_uniform
from src.backtest.research_benchmark import build_research_benchmark
from src.backtest.metrics import (
    apply_holm_correction,
    blue_ranking_metrics,
    evaluate_tickets,
    expected_calibration_error,
    paired_significance,
    validate_tickets,
    window_volatility,
)
from src.backtest.adaptive_calibration import (
    adjust_after_validation,
    assess_validation,
    most_similar_prior_context,
    parameter_snapshot,
)
from src.combination_gen.monte_carlo import (
    generate_candidates,
    generate_core_candidates,
)
from src.combination_gen.dantuo import build_dantuo_plan
from src.combination_gen.hit_coherence import (
    apply_hit_coherence_final5,
    evaluate_final5_counterfactual,
    red_draws_from_frame,
    summarize_final5_records,
)
from src.utils.idle_sleep import pulse_idle_sleep_block
from src.combination_gen.optimizer import (
    _ensure_minimum_blue_diversity,
    adverse_guard_coverage_optimize,
    consensus_optimize,
    historical_hit_optimize,
    optimize_candidates,
    rotational_coverage_optimize,
)
from src.candidate.recall_funnel import evaluate_candidate_recall, summarize_candidate_recall_records
from src.candidate.recall_protection import (
    apply_red_fusion_recall_channel,
    apply_red_candidate_recall_protection,
    enforce_red_candidate_recall_slots,
)
from src.data_handler.cleaner import RED_COLUMNS, assert_left_truncated
from src.portfolio.final_rebalance import rebalance_final_portfolio
from src.portfolio.red_method_ablation import summarize_red_method_ablation
from src.candidate.exclusion_filter import apply_low_confidence_exclusion
from src.fusion.model_fusion import (
    ANCHOR_HEAD_SHARE_KEY,
    ANCHOR_MIN_SIGNAL_KEY,
    ANCHOR_MODE_KEY,
    ANCHOR_TOP_K_KEY,
    ENTROPY_SHARPEN_MIN_SIGNAL_KEY,
    ENTROPY_SHARPEN_POWER_KEY,
    ENTROPY_SHARPEN_THRESHOLD_KEY,
    MODEL_NAMES,
    adaptive_weights,
    brier_components,
    build_models,
    cap_model_weight_shares,
    fuse_predictions,
    multiscale_shrinkage_weights,
    random_baseline_brier_components,
    rank_prediction,
)
from src.portfolio.rolling_strategy import apply_rolling_portfolio_memory
from src.governance.pipeline_guards import (
    merge_candidate_pools as _merge_candidate_pools,
    red_guard_stoploss_decision as _red_guard_stoploss_decision,
    temper_with_uniform as _temper_with_uniform,
    uniform_probabilities as _uniform_probabilities,
)
from src.utils.seed_offsets import seed_with_offset
from src.runtime.logging_compact import compact_sampling_log
from src.fusion.stacking_meta import (
    blend_with_stacking,
    serialize_model_probabilities,
    stacking_meta_predict,
)
from src.blue.expert_groups import cap_blue_expert_groups
from src.evidence.runtime_control import apply_evidence_weight_control, build_evidence_runtime_control
from src.governance.scorecard import model_scorecard
from src.governance.v7_diagnostics import build_v7_diagnostics
from src.governance.final_v7_guard import apply_final_v7_guard
from src.red.portfolio import select_red_portfolio

MODEL_LABELS = {
    "frequency": "频率模型",
    "time_decay": "时间衰减模型",
    "bayesian": "贝叶斯模型",
    "logistic_regression": "逻辑回归模型",
    "xgboost": "XGBoost模型",
    "lightgbm_ranker": "LightGBM排序模型",
    "mlp": "浅层MLP模型",
    "catboost": "CatBoost模型",
    "pattern_memory": "形态记忆模型",
}
METHOD_LABELS = {
    "greedy": "贪心覆盖",
    "genetic": "遗传算法",
    "pareto": "Pareto多目标",
    "historical_hit": "历史回放命中优化",
    "historical_hit_ablation_no_history": "历史回放消融（无历史权重）",
    "red_ge3_objective": "红球3+目标优化",
    "global_coverage": "整数规划全局覆盖",
    "consensus": "综合筛选3注",
}
ML_MODEL_NAMES = {"logistic_regression", "xgboost", "lightgbm_ranker", "mlp", "catboost"}


def _log_detail_enabled(config: dict[str, Any], key: str, *, default: str = "summary") -> bool:
    value = str(config.get("logging", {}).get(key, default)).strip().lower()
    return value in {"full", "verbose", "debug"}


def _compact_runtime_log(payload: dict[str, Any], *, max_list: int = 12) -> dict[str, Any]:
    """Keep long walk-forward logs small while retaining decision evidence.

    Full diagnostics remain in the report and trace artifacts.  The runtime log
    should only carry the fields needed to spot strategy, guard, and failure
    changes during a long 300-period run.
    """
    if not isinstance(payload, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "schema",
        "active",
        "version",
        "production_strategy",
        "red_ge3_lifecycle",
        "runtime_fast_path",
        "research_skipped",
        "research_cache_hit",
        "elapsed_seconds",
        "blue_edge_state",
        "blue_path",
        "passed",
        "mode",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    if "budget" in payload and isinstance(payload.get("budget"), dict):
        budget = payload["budget"]
        summary["budget"] = {
            "edge_state": budget.get("edge_state"),
            "slots": budget.get("slots"),
        }
    if "red_pools" in payload and isinstance(payload.get("red_pools"), dict):
        pools = payload["red_pools"]
        summary["red_pools"] = {
            key: list(pools.get(key, []) or [])[:max_list]
            for key in ("consensus_core", "probability_support", "model_rescue", "exploration")
        }
        summary["candidate_count"] = len(pools.get("candidate_set", []) or [])
    if "retention_guard" in payload and isinstance(payload.get("retention_guard"), dict):
        guard = payload["retention_guard"]
        summary["retention_guard"] = {
            "selected": guard.get("selected"),
            "controlled_passes": guard.get("controlled_passes"),
            "backbone_irr": guard.get("backbone_information_retention_rate"),
            "controlled_irr": guard.get("controlled_information_retention_rate"),
        }
    if "model_rescue" in payload and isinstance(payload.get("model_rescue"), dict):
        rescue = payload["model_rescue"]
        summary["model_rescue"] = {
            "active": rescue.get("active"),
            "entered_final": rescue.get("entered_final"),
            "replaced": rescue.get("replaced"),
            "rescue_numbers": rescue.get("rescue_numbers"),
        }
    info = payload.get("information_retention")
    if isinstance(info, dict):
        summary["information_retention"] = {
            "candidate_number_count": info.get("candidate_number_count"),
            "final_union_count": info.get("final_union_count"),
            "information_retention_rate": info.get("information_retention_rate"),
            "compression_regret": info.get("compression_regret"),
            "dropped_candidate_numbers": info.get("dropped_candidate_numbers"),
        }
    if "blue_prediction_set" in payload and isinstance(payload.get("blue_prediction_set"), dict):
        prediction = payload["blue_prediction_set"]
        summary["blue_prediction_set"] = {
            "edge_state": prediction.get("edge_state"),
            "selected_width": prediction.get("selected_width"),
            "candidate_width_range": prediction.get("candidate_width_range"),
        }
    if "unique_blues" in payload:
        summary["unique_blues"] = payload.get("unique_blues")
    if "violations" in payload:
        summary["violations"] = payload.get("violations")
    if "events" in payload:
        summary["events"] = payload.get("events")
    return summary


def _uniform_model_prediction() -> dict[str, np.ndarray]:
    return {
        "red": np.full(33, 1.0 / 33.0, dtype=float),
        "blue": np.full(16, 1.0 / 16.0, dtype=float),
    }


def _calibration_log_status(
    assessment: dict[str, Any],
    *,
    allow_parameter_change: bool,
    adjustment: dict[str, Any],
) -> str:
    if assessment.get("abstained"):
        return "abstained_no_signal"
    if assessment.get("calibration_objective_passed", assessment.get("passed", False)):
        return "objective_passed" if assessment.get("passed") else "objective_passed_hit_check_failed"
    if not allow_parameter_change:
        return "objective_failed_gate_blocked"
    if adjustment.get("changes"):
        return "objective_failed_parameters_adjusted"
    return "objective_failed_weights_refreshed"


def _fit_and_predict(
    history: pd.DataFrame,
    config: dict[str, Any],
    target_idx: int,
    models: dict | None = None,
    *,
    refit_ml: bool = False,
) -> tuple[dict, dict]:
    # 滑动窗口训练：只使用最近N期数据，减少特征工程和训练时间
    training_window = int(config.get("backtest", {}).get("training_window", 0))
    if training_window > 0 and len(history) > training_window:
        original_len = len(history)
        original_target_idx = int(target_idx)
        history = history.iloc[-training_window:].copy()
        # 重新设置idx从0开始连续递增，满足特征工程的assert_clean_sequence要求
        # 保留原始period列用于期数标识，idx仅用于特征工程的位置计算
        history["idx"] = np.arange(len(history), dtype=int)
        history = history.reset_index(drop=True)
        # 同步调整target_idx：新的target_idx = 原始target_idx - 截去的期数
        # 截去的期数 = original_len - training_window
        truncated_periods = original_len - len(history)
        target_idx = original_target_idx - truncated_periods
        logging.info(
            "滑动窗口训练 | 事件=SLIDING_WINDOW_APPLIED | 原始期数=%s | 窗口大小=%s | 实际使用期数=%s | 截去老数据=%s期 | idx已重置为0-%s | 原始target_idx=%s | 新target_idx=%s",
            original_len,
            training_window,
            len(history),
            truncated_periods,
            len(history) - 1,
            original_target_idx,
            target_idx,
        )
    if models is None:
        models = build_models(config)
        refit_ml = True
    else:
        _sync_cached_statistical_models(models, config)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    max_attempts = max(1, int(config.get("models", {}).get("fit_max_retries", 2)))
    # 增量训练配置
    incremental_enabled = bool(config.get("backtest", {}).get("incremental_training", False))
    incremental_tree_ratio = float(config.get("backtest", {}).get("incremental_tree_ratio", 0.2))
    # 支持增量训练的模型
    incremental_models = {"xgboost", "lightgbm_ranker"}
    model_detail_enabled = _log_detail_enabled(config, "model_detail")
    for name, model in models.items():
        if bool(getattr(model, "disabled_", False)):
            predictions[name] = model.predict(history, target_idx)
            setattr(model, "fit_failed_", False)
            setattr(model, "last_fit_error_", None)
            logging.debug(
                "模型已按配置禁用，使用均匀占位 | 事件=MODEL_SKIPPED_DISABLED | 模型=%s(%s) | 目标索引=%s | 原因=%s",
                MODEL_LABELS.get(name, name),
                name,
                target_idx,
                getattr(model, "reason", "disabled_by_config"),
            )
            continue
        is_ml_model = name in ML_MODEL_NAMES
        has_ml_state = bool(getattr(model, "models_", None)) if is_ml_model else False
        has_statistical_state = hasattr(model, "scores_")
        fit_required = (
            (not is_ml_model)
            or bool(refit_ml)
            or (is_ml_model and not has_ml_state)
            or ((not is_ml_model) and not has_statistical_state)
        )
        # 判断是否使用增量训练：启用增量 + 支持增量的模型 + 非完整重训期 + 已有模型状态
        use_incremental = (
            incremental_enabled
            and name in incremental_models
            and not bool(refit_ml)
            and has_ml_state
            and fit_required
        )
        started = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                if fit_required:
                    (logging.info if model_detail_enabled else logging.debug)(
                        "模型训练开始 | 事件=MODEL_FIT_START | 模型=%s(%s) | 历史期数=%s | 目标索引=%s | 尝试=%s/%s | 增量训练=%s",
                        MODEL_LABELS.get(name, name), name, len(history), target_idx, attempt, max_attempts, use_incremental,
                    )
                    if use_incremental and hasattr(model, "fit"):
                        # 调用支持增量训练的fit方法
                        model.fit(history, incremental=True, incremental_tree_ratio=incremental_tree_ratio)
                    else:
                        model.fit(history)
                predictions[name] = model.predict(history, target_idx)
                setattr(model, "fit_failed_", False)
                setattr(model, "last_fit_error_", None)
                break
            except Exception as exc:
                if attempt >= max_attempts:
                    logging.exception(
                        "模型训练或预测失败，已降级为均匀分布 | 事件=MODEL_FIT_FALLBACK | 模型=%s(%s) | 目标索引=%s | 错误=%s",
                        MODEL_LABELS.get(name, name),
                        name,
                        target_idx,
                        exc,
                    )
                    setattr(model, "fit_failed_", True)
                    setattr(model, "last_fit_error_", str(exc))
                    predictions[name] = _uniform_model_prediction()
                else:
                    logging.warning(
                        "模型训练或预测失败，准备重试 | 事件=MODEL_FIT_RETRY | 模型=%s(%s) | 目标索引=%s | 尝试=%s/%s | 错误=%s",
                        MODEL_LABELS.get(name, name),
                        name,
                        target_idx,
                        attempt,
                        max_attempts,
                        exc,
                    )
        (logging.info if model_detail_enabled else logging.debug)(
            "模型评分完成 | 事件=MODEL_PREDICT_DONE | 模型=%s(%s) | 本轮是否重训=%s | 耗时=%.3f秒 | 训练Brier=%s | 红球前6=%s | 蓝球首选=%s",
            MODEL_LABELS.get(name, name),
            name,
            fit_required,
            time.perf_counter() - started,
            getattr(model, "training_brier_", None),
            (np.argsort(predictions[name]["red"])[-6:] + 1).tolist(),
            int(np.argmax(predictions[name]["blue"]) + 1),
        )
    return models, predictions


def _sync_cached_statistical_models(models: dict[str, Any], config: dict[str, Any]) -> None:
    """Refresh cheap non-ML model parameters without invalidating ML caches.

    The retrain interval governs expensive supervised models.  Statistical
    models are deliberately re-fitted every period because their estimates are
    direct functions of the left-truncated history.  When online calibration
    changes cheap statistical parameters between ML retrain points, update only
    those cheap models here; changing an already-fitted ML model's feature
    config would risk train/predict feature mismatch, so ML changes take effect
    at the next scheduled retrain.
    """
    features = config.get("features", {})
    if "time_decay" in models and hasattr(models["time_decay"], "decay_factor"):
        models["time_decay"].decay_factor = float(features.get("decay_factor", 0.98))
    if "pattern_memory" in models:
        pattern = models["pattern_memory"]
        if hasattr(pattern, "feature_config"):
            pattern.feature_config = dict(features)
        if hasattr(pattern, "model_config"):
            pattern.model_config = dict(config.get("models", {}).get("pattern_memory", {}))


def _rolling_losses(loss_history: dict[str, list[float]], window: int) -> dict[str, list[float]]:
    return {name: values[-int(window):] for name, values in loss_history.items()}


def _mean_uniform_ticket_metrics(
    *,
    count: int,
    repeats: int,
    actual_red: set[int],
    actual_blue: int,
    random_seed: int,
) -> dict[str, float]:
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if int(repeats) <= 0:
        raise ValueError("repeats 必须大于0")

    # A uniformly random双色球 ticket is symmetric with respect to the concrete
    # opened numbers, so the random baseline does not need repeated simulation.
    # Computing the exact expectation removes Monte-Carlo noise and saves a
    # large amount of research-mode time without changing any model prediction.
    ticket_count = int(count)
    total_red_combinations = comb(33, 6)

    def red_hit_probability(hit_count: int) -> float:
        if hit_count < 0 or hit_count > 6:
            return 0.0
        if 6 - hit_count > 27:
            return 0.0
        return float(comb(6, hit_count) * comb(27, 6 - hit_count) / total_red_combinations)

    red_distribution = [red_hit_probability(hit_count) for hit_count in range(7)]
    single_red_ge3 = float(sum(red_distribution[3:]))
    single_red_ge4 = float(sum(red_distribution[4:]))
    cumulative = np.cumsum(np.asarray(red_distribution, dtype=float))
    expected_best_red = float(
        sum(1.0 - float(cumulative[threshold - 1]) ** ticket_count for threshold in range(1, 7))
    )
    blue_miss_probability = (15.0 / 16.0) ** ticket_count
    return {
        "average_red_hits": float(6.0 * 6.0 / 33.0),
        "red_ge3_ratio": single_red_ge3,
        "red_ge4_ratio": single_red_ge4,
        "any_red_ge3": float(1.0 - (1.0 - single_red_ge3) ** ticket_count),
        "any_red_ge4": float(1.0 - (1.0 - single_red_ge4) ** ticket_count),
        "blue_hit_ratio": float(1.0 / 16.0),
        "any_blue_hit": float(1.0 - blue_miss_probability),
        "blue_unique_count": float(16.0 * (1.0 - (15.0 / 16.0) ** ticket_count)),
        "best_red_hits": expected_best_red,
    }


def _json_trace_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _validation_trace_path(config: dict[str, Any]) -> Path:
    configured = str(
        config.get("paths", {}).get("validation_trace", "data/validation_trace.jsonl")
    )
    path = Path(configured)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _write_validation_trace(path: Path, payload: dict[str, Any], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append and path.exists() and path.stat().st_size > 0:
        archive_dir = path.parent / f"{path.stem}_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
        counter = 1
        while archive_path.exists():
            archive_path = archive_dir / f"{path.stem}_{timestamp}_{counter}{path.suffix}"
            counter += 1
        archive_path.write_bytes(path.read_bytes())
        logging.info(
            "逐期验证追踪旧文件已归档 | 事件=VALIDATION_TRACE_ARCHIVED | 来源=%s | 归档=%s",
            path,
            archive_path,
        )
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_trace_default))
        handle.write("\n")


def _probability_rank_lookup(values: np.ndarray) -> dict[int, dict[str, float | int]]:
    probabilities = np.asarray(values, dtype=float)
    order = np.argsort(-probabilities, kind="stable")
    return {
        int(index + 1): {
            "rank": int(rank + 1),
            "score": float(probabilities[index]),
        }
        for rank, index in enumerate(order)
    }


def _is_uniform_placeholder_prediction(values: np.ndarray, *, tolerance: float = 1e-12) -> bool:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.size == 0:
        return True
    return bool(np.nanmax(probabilities) - np.nanmin(probabilities) <= float(tolerance))


def _model_number_reasons(
    predictions: dict[str, dict[str, np.ndarray]],
    weights_by_color: dict[str, dict[str, float]],
    *,
    color: str,
    number: int,
    rank_cutoff: int,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    for model_name, prediction in predictions.items():
        if color not in prediction:
            continue
        lookup = _probability_rank_lookup(prediction[color])
        detail = lookup.get(int(number))
        if not detail:
            continue
        weight = float(weights_by_color.get(color, {}).get(model_name, 0.0))
        if weight <= 0.0:
            continue
        if int(detail["rank"]) <= int(rank_cutoff) or weight > 0.0:
            reasons.append({
                "model": model_name,
                "color_weight": weight,
                "rank": int(detail["rank"]),
                "score": float(detail["score"]),
                "reason": (
                    f"{model_name}在{color}侧排名第{int(detail['rank'])}，"
                    f"当前融合权重={weight:.4f}"
                ),
            })
    return sorted(
        reasons,
        key=lambda item: (int(item["rank"]), -float(item["color_weight"])),
    )


def _method_number_support(
    selected_by_method: dict[str, list[dict]],
    *,
    color: str,
    number: int,
) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    for method, tickets in selected_by_method.items():
        indices: list[int] = []
        for index, ticket in enumerate(tickets, start=1):
            if color == "red" and int(number) in {int(item) for item in ticket.get("red", [])}:
                indices.append(index)
            if color == "blue" and int(ticket.get("blue", 0)) == int(number):
                indices.append(index)
        if indices:
            support.append({
                "method": method,
                "ticket_indices": indices,
                "reason": f"{method}方法第{','.join(str(item) for item in indices)}注包含该{color}号码",
            })
    return support


def _candidate_pool_number_summary(pool: list[dict], *, color: str, number: int) -> dict[str, Any]:
    appearances: list[dict[str, Any]] = []
    for index, candidate in enumerate(pool, start=1):
        included = (
            int(number) in {int(item) for item in candidate.get("red", [])}
            if color == "red"
            else int(candidate.get("blue", 0)) == int(number)
        )
        if not included:
            continue
        appearances.append({
            "candidate_rank": index,
            "red": [int(item) for item in candidate.get("red", [])],
            "blue": int(candidate.get("blue", 0)),
            "quality_score": float(candidate.get("quality_score", 0.0)),
            "red_number_score": float(candidate.get("red_number_score", candidate.get("number_score", 0.0))),
            "blue_number_score": float(candidate.get("blue_number_score", candidate.get("number_score", 0.0))),
            "selection_role": candidate.get("selection_role"),
        })
    best = max(appearances, key=lambda item: float(item["quality_score"])) if appearances else None
    return {
        "entered_candidate_pool": bool(appearances),
        "candidate_pool_support_count": len(appearances),
        "best_candidate": best,
    }


def _ticket_key_tuple(ticket: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    return (
        tuple(sorted(int(item) for item in ticket.get("red", []) or [])),
        int(ticket.get("blue", 0) or 0),
    )


def _build_recommended_ticket_trace(
    tickets: list[dict],
    *,
    selected_by_method: dict[str, list[dict]],
    predictions: dict[str, dict[str, np.ndarray]],
    fused: dict[str, np.ndarray],
    weights_by_color: dict[str, dict[str, float]],
    consensus_strategy: str,
    rebalance_info: dict[str, Any],
    rolling_memory_info: dict[str, Any],
) -> list[dict[str, Any]]:
    method_ticket_sources: dict[tuple[tuple[int, ...], int], list[str]] = {}
    for method, method_tickets in selected_by_method.items():
        for ticket in method_tickets:
            method_ticket_sources.setdefault(_ticket_key_tuple(ticket), []).append(method)
    fused_lookup = {
        "red": _probability_rank_lookup(fused["red"]),
        "blue": _probability_rank_lookup(fused["blue"]),
    }
    traced: list[dict[str, Any]] = []
    for index, ticket in enumerate(tickets, start=1):
        red = [int(item) for item in ticket.get("red", [])]
        blue = int(ticket.get("blue", 0))
        ticket_sources = method_ticket_sources.get(_ticket_key_tuple(ticket), [])
        red_reasons = {}
        for number in red:
            red_reasons[str(number)] = {
                "fused_rank": fused_lookup["red"].get(number),
                "model_reasons": _model_number_reasons(
                    predictions,
                    weights_by_color,
                    color="red",
                    number=number,
                    rank_cutoff=12,
                ),
                "method_support": _method_number_support(
                    selected_by_method,
                    color="red",
                    number=number,
                ),
                "final_reason": (
                    f"最终组合策略={consensus_strategy}；"
                    f"该号码出现在最终第{index}注"
                ),
            }
        traced.append({
            "ticket_index": index,
            "red": red,
            "blue": blue,
            "matched_red": ticket.get("matched_red", []),
            "blue_hit": bool(ticket.get("blue_hit", False)),
            "ticket_selection_role": ticket.get("selection_role"),
            "ticket_selection_reason": ticket.get("selection_reason"),
            "method_ticket_sources": ticket_sources,
            "candidate_scores": {
                "quality_score": float(ticket.get("quality_score", 0.0)),
                "red_number_score": float(ticket.get("red_number_score", ticket.get("number_score", 0.0))),
                "blue_number_score": float(ticket.get("blue_number_score", ticket.get("number_score", 0.0))),
                "structure_score": float(ticket.get("structure_score", 0.0)),
                "blue_signal_bonus": float(ticket.get("blue_signal_bonus", 0.0)),
            },
            "red_number_reasons": red_reasons,
            "blue_number_reason": {
                "blue_selection_role": ticket.get("blue_selection_role"),
                "blue_selection_reason": ticket.get("blue_selection_reason"),
                "final_decision_function": "finalize_blue_portfolio -> apply_blue_specialist_portfolio -> _policy_optimized_blues -> _coverage_optimized_blues",
                "fused_rank": fused_lookup["blue"].get(blue),
                "model_reasons": _model_number_reasons(
                    predictions,
                    weights_by_color,
                    color="blue",
                    number=blue,
                    rank_cutoff=5,
                ),
                "method_support": _method_number_support(
                    selected_by_method,
                    color="blue",
                    number=blue,
                ),
                "final_reason": (
                    f"最终组合策略={consensus_strategy}；"
                    f"该蓝球出现在最终第{index}注"
                ),
            },
            "postprocess": {
                "rebalance": rebalance_info,
                "rolling_memory": rolling_memory_info,
            },
        })
    return traced


def _infer_drop_stage(
    *,
    color: str,
    number: int,
    selected_final: bool,
    candidate_summary: dict[str, Any],
    method_support: list[dict[str, Any]],
    exclusion_diagnostics: dict[str, Any],
) -> str:
    if selected_final:
        return "retained_in_final_recommendation"
    excluded_key = "red_excluded" if color == "red" else "blue_excluded"
    excluded = {int(item) for item in exclusion_diagnostics.get(excluded_key, []) or []}
    if int(number) in excluded:
        return "low_confidence_exclusion_filter"
    if not candidate_summary.get("entered_candidate_pool"):
        return "candidate_generation_or_candidate_retention"
    if not method_support:
        return "optimizer_method_selection"
    return "final_consensus_rebalance_or_blue_specialist"


def _infer_eliminated_by(
    *,
    color: str,
    selected_final: bool,
    drop_stage: str,
    candidate_summary: dict[str, Any],
    method_support: list[dict[str, Any]],
    selected_by_method: dict[str, list[dict]],
    consensus_strategy: str,
    final_tickets: list[dict],
    in_ranking_candidate: bool = False,
) -> str:
    if selected_final:
        return "已通过候选筛选：进入最终筛选前的候选集合"
    if in_ranking_candidate:
        return (
            "Final5组合优化失败：该号已进入Candidate，但最终5注没有保留。"
            "这不是模型没有预测到，而是Final5把已经找到的号码拆散或丢掉了。"
        )
    if drop_stage == "low_confidence_exclusion_filter":
        return "未进入预算胆拖：在低置信剔除器阶段被排除，未进入候选生成。判断：模型共识或历史支持不足。"
    if not candidate_summary.get("entered_candidate_pool"):
        return "未进入预算胆拖：在候选生成/候选池保留阶段被挡住。判断：模型概率、抽样结果或质量排序不足，未进入可选候选池。"
    support_methods = [
        str(item.get("method"))
        for item in method_support
        if item.get("method")
    ]
    attempted_methods = [
        str(method)
        for method in selected_by_method
        if str(method) != "consensus"
    ]
    if not support_methods:
        methods_text = "/".join(attempted_methods) if attempted_methods else "优化方法"
        return f"未进入预算胆拖：该号进入候选池，但未被{methods_text}选中。判断：组合质量、结构约束或覆盖目标不占优。"
    best = candidate_summary.get("best_candidate") or {}
    support_count = int(candidate_summary.get("candidate_pool_support_count", 0) or 0)
    best_rank = best.get("candidate_rank")
    method_text = "、".join(support_methods)
    strategy = consensus_strategy or "最终综合层"
    rank_text = f"候选池最佳第{best_rank}名，出现{support_count}次" if best_rank else f"候选池出现{support_count}次"
    if color == "blue":
        return (
            f"未进入预算胆拖：该蓝球虽被{method_text}选中过，且{rank_text}；"
            f"但在{strategy}阶段被压缩掉。判断：系统优先保留蓝球唯一性、覆盖轮换和整票综合分更高的蓝球。"
        )
    best_red = {int(item) for item in best.get("red", []) or []}
    final_sets = [
        {int(item) for item in ticket.get("red", []) or []}
        for ticket in final_tickets
    ]
    max_overlap = max((len(best_red & red_set) for red_set in final_sets), default=0)
    if max_overlap <= 2 and support_count >= 20:
        detail = "含该号的高支持候选票与最终保留红球交集不足3个，属于“整票信号被拆散”；预算胆拖压缩层会优先桥接这类候选票。"
    elif max_overlap >= 5:
        detail = "主要原因是含该号候选与已保留组合高度重叠，最终多样性/去重再平衡优先替换。"
    elif best_rank and int(best_rank) > max(10, len(final_tickets) * 3):
        detail = "主要原因是含该号候选虽然进池，但综合排序靠后，未挤进最终保留集合。"
    elif support_count >= 20:
        detail = "该号并非完全无信号，主要是在最终共识投票、重叠约束和覆盖目标之间被挤掉。"
    else:
        detail = "主要原因是候选支持次数偏少，最终组合层优先保留跨方法支持更稳定的票。"
    return (
        f"未进入预算胆拖：该号虽被{method_text}选中过，且{rank_text}；"
        f"但在{strategy}阶段被压缩掉。证据：含该号候选与已保留候选最高重叠{max_overlap}个红球。判断：{detail}"
    )


def _build_actual_number_trace(
    *,
    actual_red: set[int],
    actual_blue: int,
    final_tickets: list[dict],
    pool: list[dict],
    selected_by_method: dict[str, list[dict]],
    predictions: dict[str, dict[str, np.ndarray]],
    fused: dict[str, np.ndarray],
    weights_by_color: dict[str, dict[str, float]],
    exclusion_diagnostics: dict[str, Any],
    consensus_strategy: str,
    candidate_recall_funnel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_red = {int(number) for ticket in final_tickets for number in ticket.get("red", [])}
    final_blues = {int(ticket.get("blue", 0)) for ticket in final_tickets}
    fused_lookup = {
        "red": _probability_rank_lookup(fused["red"]),
        "blue": _probability_rank_lookup(fused["blue"]),
    }
    funnel = candidate_recall_funnel or {}
    red_candidate_set = (((funnel.get("red") or {}).get("candidate_set") or {}).get("selected") or [])
    red_channels = (funnel.get("red") or {}).get("candidate_set") or {}
    red_channel_sets = {
        "independent_model_top6": {int(value) for value in (funnel.get("red") or {}).get("independent_model_top6_numbers", []) or []},
        "independent_model_top10": {int(value) for value in (funnel.get("red") or {}).get("independent_model_top10_numbers", []) or []},
        "model_top12": {int(value) for value in red_channels.get("top12", []) or []},
        "model_top16": {int(value) for value in red_channels.get("top16", []) or []},
        "model_top18": {int(value) for value in red_channels.get("top18", []) or []},
        "method_consensus": {int(value) for value in red_channels.get("method_consensus", []) or []},
        "wide_candidate": {int(value) for value in red_candidate_set},
    }

    def first_seen_stage(color: str, number: int, candidate_summary: dict[str, Any], method_support: list[dict[str, Any]]) -> str:
        rank = (fused_lookup[color].get(int(number)) or {}).get("rank")
        try:
            rank_value = int(rank)
        except (TypeError, ValueError):
            rank_value = 999
        if color == "red":
            if int(number) in red_channel_sets["independent_model_top6"]:
                return "independent_model_top6"
            if int(number) in red_channel_sets["independent_model_top10"]:
                return "independent_model_top10"
            if rank_value <= 6:
                return "fusion_top6"
            if rank_value <= 10:
                return "fusion_top10"
            if int(number) in red_channel_sets["model_top18"]:
                return "fusion_top18"
            if int(number) in red_channel_sets["method_consensus"]:
                return "method_consensus"
            if candidate_summary.get("entered_candidate_pool"):
                return "wide_candidate_pool"
            if method_support:
                return "optimizer_method"
            return "not_found_before_final"
        if rank_value <= 1:
            return "blue_top1"
        if rank_value <= 3:
            return "blue_top3"
        if rank_value <= 5:
            return "blue_top5"
        if candidate_summary.get("entered_candidate_pool"):
            return "blue_candidate_pool"
        if method_support:
            return "blue_method_layer"
        return "not_found_before_final"

    def candidate_channel(color: str, number: int, candidate_summary: dict[str, Any], method_support: list[dict[str, Any]]) -> list[str]:
        channels: list[str] = []
        if color == "red":
            for name, values in red_channel_sets.items():
                if int(number) in values:
                    channels.append(name)
        if candidate_summary.get("entered_candidate_pool"):
            channels.append("raw_candidate_pool")
        if method_support:
            channels.append("optimizer_method_support")
        return list(dict.fromkeys(channels))

    def trace_one(color: str, number: int) -> dict[str, Any]:
        selected_final = (
            int(number) in final_red if color == "red" else int(number) in final_blues
        )
        candidate_summary = _candidate_pool_number_summary(pool, color=color, number=number)
        method_support = _method_number_support(selected_by_method, color=color, number=number)
        drop_stage = _infer_drop_stage(
            color=color,
            number=int(number),
            selected_final=bool(selected_final),
            candidate_summary=candidate_summary,
            method_support=method_support,
            exclusion_diagnostics=exclusion_diagnostics,
        )
        first_stage = first_seen_stage(color, int(number), candidate_summary, method_support)
        channels = candidate_channel(color, int(number), candidate_summary, method_support)
        return {
            "number": int(number),
            "selected_in_final": bool(selected_final),
            "hit_status": "hit" if selected_final else "missed",
            "first_seen_stage": first_stage,
            "candidate_channel": channels,
            "elimination_stage": "retained" if selected_final else drop_stage,
            "final_survival": {
                "entered_before_final": first_stage != "not_found_before_final",
                "selected_in_final": bool(selected_final),
                "lost_after_being_found": bool((first_stage != "not_found_before_final") and not selected_final),
            },
            "candidate_pool": candidate_summary,
            "method_support": method_support,
            "fused_rank": fused_lookup[color].get(int(number)),
            "model_reasons": _model_number_reasons(
                predictions,
                weights_by_color,
                color=color,
                number=int(number),
                rank_cutoff=12 if color == "red" else 5,
            ),
            "drop_stage_inference": drop_stage,
            "eliminated_by": _infer_eliminated_by(
                color=color,
                selected_final=bool(selected_final),
                drop_stage=drop_stage,
                candidate_summary=candidate_summary,
                method_support=method_support,
                selected_by_method=selected_by_method,
                consensus_strategy=consensus_strategy,
                final_tickets=final_tickets,
                in_ranking_candidate=bool(color == "red" and int(number) in red_channel_sets["wide_candidate"]),
            ),
            "drop_stage_note": (
                "该字段基于流水线阶段做可复盘归因：剔除器→候选池→各优化方法→最终consensus/再平衡/蓝球专家。"
            ),
        }

    return {
        "red": [trace_one("red", number) for number in sorted(actual_red)],
        "blue": trace_one("blue", int(actual_blue)),
        "missed_red": [
            int(number) for number in sorted(actual_red)
            if int(number) not in final_red
        ],
        "missed_blue": None if int(actual_blue) in final_blues else int(actual_blue),
    }


def _build_validation_trace_record(
    *,
    target_period: str,
    training_period: str,
    target_idx: int,
    history_max_idx: int,
    actual_red: set[int],
    actual_blue: int,
    final_tickets: list[dict],
    selected_by_method: dict[str, list[dict]],
    methods: dict[str, Any],
    pool: list[dict],
    sampling: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    fused: dict[str, np.ndarray],
    fused_ranking: dict[str, Any],
    weights_by_color: dict[str, dict[str, float]],
    weight_reason_by_color: dict[str, str],
    consensus_strategy: str,
    rebalance_info: dict[str, Any],
    rolling_memory_info: dict[str, Any],
    assessment: dict[str, Any],
    adjustment: dict[str, Any],
    candidate_recall_funnel: dict[str, Any] | None = None,
    dantuo_config: dict[str, Any] | None = None,
    blue_history: Iterable[Any] | None = None,
    final5_hit_coherence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dantuo_config = dantuo_config or {}
    method_tickets = {
        method: [
            {
                "ticket_index": index,
                "red": [int(item) for item in ticket.get("red", [])],
                "blue": int(ticket.get("blue", 0)),
                "matched_red": ticket.get("matched_red", []),
                "blue_hit": bool(ticket.get("blue_hit", False)),
                "selection_role": ticket.get("selection_role"),
            }
            for index, ticket in enumerate(value.get("tickets", []), start=1)
        ]
        for method, value in methods.items()
    }
    final_recommendation = _build_recommended_ticket_trace(
        final_tickets,
        selected_by_method=selected_by_method,
        predictions=predictions,
        fused=fused,
        weights_by_color=weights_by_color,
        consensus_strategy=consensus_strategy,
        rebalance_info=rebalance_info,
        rolling_memory_info=rolling_memory_info,
    )
    dantuo_source_tickets = [
        {
            "red": ticket.get("red", []),
            "blue": ticket.get("blue"),
            "consensus_score": (ticket.get("candidate_scores") or {}).get("quality_score", 0.0),
            "selection_role": ticket.get("ticket_selection_role")
            or ",".join(str(item) for item in ticket.get("method_ticket_sources", []) or []),
            "blue_selection_role": (ticket.get("blue_number_reason") or {}).get("final_reason", ""),
        }
        for ticket in final_recommendation
    ]
    method_limit = max(1, int(dantuo_config.get("method_ticket_limit_per_method", 5)))
    for method, tickets in method_tickets.items():
        if str(method) == "consensus":
            continue
        for ticket in (tickets or [])[:method_limit]:
            dantuo_source_tickets.append({
                "red": ticket.get("red", []),
                "blue": ticket.get("blue"),
                "consensus_score": 0.0,
                "selection_role": f"dantuo_method_source_{method}",
                "blue_selection_role": str(ticket.get("selection_role", "") or ""),
            })
    actual_payload = {
        "red": sorted(int(item) for item in actual_red),
        "blue": int(actual_blue),
    }
    dantuo_plan = build_dantuo_plan(
        dantuo_source_tickets,
        final_recommendation=final_recommendation,
        candidate_pool=pool,
        candidate_pool_ticket_limit=int(dantuo_config.get("candidate_pool_ticket_limit", 80)),
        fusion=fused_ranking,
        actual=actual_payload,
        blue_history=blue_history,
        red_priority_numbers=(
            ((candidate_recall_funnel or {}).get("red") or {}).get("independent_model_top10_numbers")
            or ((candidate_recall_funnel or {}).get("red") or {}).get("independent_model_top6_numbers")
            or []
        ),
        max_amount=int(dantuo_config.get("max_amount", 20)),
        unit_price=int(dantuo_config.get("unit_price", 2)),
        preferred_red_dan_count=int(dantuo_config.get("preferred_red_dan_count", 2)),
        preferred_red_tuo_count=int(dantuo_config.get("preferred_red_tuo_count", 5)),
        preferred_blue_count=int(dantuo_config.get("preferred_blue_count", 2)),
        prefer_compound_when_coverage_like=bool(
            dantuo_config.get("prefer_compound_when_coverage_like", True)
        ),
        preserve_final_portfolio_evidence=bool(
            dantuo_config.get("preserve_final_portfolio_evidence", True)
        ),
        final_to_dantuo_audit_enabled=bool(
            dantuo_config.get("final_to_dantuo_audit_enabled", True)
        ),
        compound_red_count=int(dantuo_config.get("compound_red_count", 7)),
        compound_blue_count=int(dantuo_config.get("compound_blue_count", 1)),
        blue_recent_gap_weight=float(dantuo_config.get("blue_recent_gap_weight", 1.00)),
        blue_recent_frequency_weight=float(dantuo_config.get("blue_recent_frequency_weight", 0.00)),
        blue_evidence_weight=float(dantuo_config.get("blue_evidence_weight", 1.00)),
        blue_specialist_weight=float(dantuo_config.get("blue_specialist_weight", 0.20)),
        blue_omission_balance_weight=float(dantuo_config.get("blue_omission_balance_weight", 0.80)),
        blue_immediate_repeat_penalty=float(dantuo_config.get("blue_immediate_repeat_penalty", 0.35)),
        blue_recent_windows=dantuo_config.get("blue_recent_windows", [20, 50, 100]),
        auto_cap_blue_to_budget=bool(dantuo_config.get("auto_cap_blue_to_budget", True)),
    )
    dantuo_red_set = {
        int(number)
        for number in list(dantuo_plan.get("red_dan", []) or [])
        + list(dantuo_plan.get("red_tuo", []) or [])
    } if dantuo_plan else set()
    dantuo_blue_set = {int(number) for number in dantuo_plan.get("blue", []) or []} if dantuo_plan else set()
    actual_red_set = {int(number) for number in actual_payload["red"]}
    dantuo_actual_summary = {
        "covered_red": sorted(actual_red_set & dantuo_red_set),
        "missed_red": sorted(actual_red_set - dantuo_red_set),
        "covered_blue": int(actual_blue) if int(actual_blue) in dantuo_blue_set else None,
        "missed_blue": None if int(actual_blue) in dantuo_blue_set else int(actual_blue),
        "red_hit_count": len(actual_red_set & dantuo_red_set),
        "blue_hit": int(actual_blue) in dantuo_blue_set,
        "blue_selected_count": len(dantuo_blue_set),
        "blue_metric": f"dantuo_blue_hit_at_{len(dantuo_blue_set)}",
        "blue_theoretical_random_hit_probability": len(dantuo_blue_set) / 16.0,
    } if dantuo_plan else None
    return {
        "event": "validation_period_trace",
        "validation_period": str(target_period),
        "training_through_period": str(training_period),
        "target_idx": int(target_idx),
        "history_max_idx": int(history_max_idx),
        "strict_left_truncation_passed": int(history_max_idx) + 1 == int(target_idx),
        "actual": actual_payload,
        "final_recommendation": final_recommendation,
        "dantuo_plan": dantuo_plan,
        "dantuo_actual_summary": dantuo_actual_summary,
        "candidate_recall_funnel": candidate_recall_funnel or {},
        "final5_hit_coherence": final5_hit_coherence or {},
        "actual_number_trace": _build_actual_number_trace(
            actual_red=actual_red,
            actual_blue=actual_blue,
            final_tickets=final_tickets,
            pool=pool,
            selected_by_method=selected_by_method,
            predictions=predictions,
            fused=fused,
            weights_by_color=weights_by_color,
            exclusion_diagnostics=sampling.get("exclusion_filter", {}),
            consensus_strategy=consensus_strategy,
            candidate_recall_funnel=candidate_recall_funnel,
        ),
        "method_recommendations": method_tickets,
        "recommendation_context": {
            "consensus_strategy": consensus_strategy,
            "sampling": sampling,
            "candidate_pool_size": len(pool),
            "fused_prediction": fused_ranking,
            "weights_by_color": weights_by_color,
            "weight_reason_by_color": weight_reason_by_color,
            "validation_assessment": assessment,
            "calibration_adjustment": adjustment,
        },
    }


def _build_compression_experiment(
    *,
    pool: list[dict],
    selected_by_method: dict[str, list[dict]],
    final_tickets: list[dict],
    portfolio_tickets: list[dict] | None = None,
    fused_probability: dict[str, np.ndarray],
    actual_red: set[int],
    actual_blue: int,
    count: int,
    optimizer_config: dict[str, Any],
    random_seed: int,
    red_candidate_limit: int = 22,
) -> dict[str, Any]:
    """Evaluate real compression strategies on the same candidate pool.

    This is intentionally run inside each strict walk-forward period after the
    candidate pool exists and before any future outcome can affect later rounds.
    It separates "which compression policy loses fewer true numbers" from the
    normal final recommendation path.
    """

    effective_count = min(max(int(count), 1), max(len(pool), 1))
    max_red_overlap = int(optimizer_config.get("signal_max_red_overlap", optimizer_config.get("max_red_overlap", 5)))
    min_blue_unique = min(
        effective_count,
        int(optimizer_config.get(
            "signal_minimum_blue_unique_count",
            optimizer_config.get("minimum_blue_unique_count", effective_count),
        )),
        16,
    )

    def clean(tickets: list[dict] | None) -> list[dict]:
        return [dict(ticket) for ticket in (tickets or [])[:effective_count]]

    probability_first = clean(
        selected_by_method.get("greedy")
        or selected_by_method.get("red_ge3_objective")
        or selected_by_method.get("pareto")
        or pool[:effective_count]
    )
    coverage_first = clean(selected_by_method.get("global_coverage"))
    if not coverage_first and pool:
        coverage_first = rotational_coverage_optimize(
            pool,
            effective_count,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=min_blue_unique,
            random_seed=random_seed,
        )

    low_jaccard = rotational_coverage_optimize(
        pool,
        effective_count,
        max_red_overlap=max(2, max_red_overlap - 1),
        minimum_blue_unique_count=min_blue_unique,
        random_seed=random_seed + 17,
    ) if pool else []
    low_probability_constraint = rotational_coverage_optimize(
        sorted(pool, key=lambda ticket: float(ticket.get("quality_score", ticket.get("score", 0.0)) or 0.0)),
        effective_count,
        max_red_overlap=max_red_overlap,
        minimum_blue_unique_count=min_blue_unique,
        random_seed=random_seed + 31,
    ) if pool else []
    no_structure_constraint = sorted(
        pool,
        key=lambda ticket: float(
            (ticket.get("candidate_scores") or {}).get(
                "red_number_score",
                ticket.get("quality_score", ticket.get("score", 0.0)),
            ) or 0.0
        ),
        reverse=True,
    )[:effective_count]
    recall_first = rotational_coverage_optimize(
        no_structure_constraint or pool,
        effective_count,
        max_red_overlap=min(6, max_red_overlap + 1),
        minimum_blue_unique_count=min_blue_unique,
        random_seed=random_seed + 47,
    ) if pool else []

    raw_strategies = {
        "current_final": clean(final_tickets),
        "portfolio_stage": clean(portfolio_tickets),
        "probability_first": probability_first,
        "coverage_first": coverage_first,
        "low_jaccard": clean(low_jaccard),
        "low_probability_constraint": clean(low_probability_constraint),
        "no_structure_constraint": clean(no_structure_constraint),
        "recall_first": clean(recall_first),
    }
    strategy_rows: dict[str, Any] = {}
    for name, tickets in raw_strategies.items():
        metrics = evaluate_tickets(tickets, actual_red, actual_blue) if tickets else {}
        recall = evaluate_candidate_recall(
            fused_probability=fused_probability,
            candidate_pool=pool,
            selected_by_method={name: tickets},
            final_tickets=tickets,
            portfolio_tickets=tickets,
            red_candidate_limit=red_candidate_limit,
            actual_red=actual_red,
            actual_blue=actual_blue,
        ) if tickets else {}
        strategy_rows[name] = {
            "description": {
                "current_final": "正式最终链路输出，含consensus、再平衡、滚动记忆与蓝球最终链路。",
                "portfolio_stage": "Portfolio Optimizer刚完成后的快照，用于和Final Guard/滚动/蓝球改写后的最终结果分离比较。",
                "probability_first": "偏模型概率/质量的压缩路线，优先采用贪心或红球3+目标组合。",
                "coverage_first": "偏全局覆盖的压缩路线，优先采用整数规划全局覆盖或轮转覆盖兜底。",
                "low_jaccard": "低重叠路线，用于识别Jaccard是否过度挤掉有效号码。",
                "low_probability_constraint": "降低单票概率优先级、提高覆盖互补性的压缩路线。",
                "no_structure_constraint": "弱化结构美观/历史典型度，直接看模型证据候选的压缩路线。",
                "recall_first": "候选召回优先路线，允许更宽重叠以减少高证据号码丢失。",
            }.get(name, name),
            "metrics": metrics,
            "candidate_recall_funnel": recall,
            "tickets": validate_tickets(tickets, actual_red, actual_blue),
        }
    best_red = max(
        strategy_rows,
        key=lambda key: float(strategy_rows[key].get("metrics", {}).get("best_red_hits", 0.0)),
    ) if strategy_rows else None
    best_blue = max(
        strategy_rows,
        key=lambda key: float(strategy_rows[key].get("metrics", {}).get("any_blue_hit", 0.0)),
    ) if strategy_rows else None
    return {
        "schema": "compression_experiment_v1",
        "strategies": strategy_rows,
        "best_red_strategy": best_red,
        "best_blue_strategy": best_blue,
    }


def _aggregate_color_weights(weights_by_color: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([weights_by_color[color].get(name, 0.0) for color in ("red", "blue")]))
        for name in MODEL_NAMES
    }


def _attach_structured_anchor_metadata(
    weights: dict[str, float],
    *,
    color: str,
    fusion_config: dict[str, Any],
) -> dict[str, float]:
    enriched = dict(weights)
    mode = str(fusion_config.get("structured_anchor_mode_by_color", {}).get(color, "uniform"))
    if mode != "tiered_head_tail":
        return enriched
    enriched[ANCHOR_MODE_KEY] = mode
    enriched[ANCHOR_TOP_K_KEY] = int(
        fusion_config.get("structured_anchor_top_k_by_color", {}).get(color, 0)
    )
    enriched[ANCHOR_HEAD_SHARE_KEY] = float(
        fusion_config.get("structured_anchor_head_share_by_color", {}).get(color, 0.0)
    )
    enriched[ANCHOR_MIN_SIGNAL_KEY] = float(
        fusion_config.get("structured_anchor_min_signal_by_color", {}).get(color, 1.0)
    )
    enriched[ENTROPY_SHARPEN_THRESHOLD_KEY] = float(
        fusion_config.get("entropy_sharpen_threshold_by_color", {}).get(color, 1.01)
    )
    enriched[ENTROPY_SHARPEN_POWER_KEY] = float(
        fusion_config.get("entropy_sharpen_power_by_color", {}).get(color, 1.0)
    )
    enriched[ENTROPY_SHARPEN_MIN_SIGNAL_KEY] = float(
        fusion_config.get("entropy_sharpen_min_signal_by_color", {}).get(color, 1.0)
    )
    return enriched


def _adaptive_color_weights(
    red_losses: dict[str, list[float]],
    blue_losses: dict[str, list[float]],
    config: dict[str, Any],
    invalid_models_by_color: dict[str, set[str]],
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    fusion = config["fusion"]
    baselines = random_baseline_brier_components()
    options = {
        "minimum_observations": int(fusion["minimum_observations"]),
        "uncertainty_z": float(fusion.get("uncertainty_z", 0.0)),
        "model_groups": fusion.get("model_groups"),
    }
    shrinkage_enabled = bool(fusion.get("enable_multiscale_shrinkage", True))
    if shrinkage_enabled:
        skill_windows = fusion.get("skill_windows", [30, 100, 200])
        weak_signal_min_windows_by_color = fusion.get("weak_signal_min_windows_by_color", {})
        weak_signal_max_strength_by_color = fusion.get("weak_signal_max_strength_by_color", {})
        weak_signal_min_strength_by_color = fusion.get("weak_signal_min_strength_by_color", {})
        max_uniform_share_if_signal_by_color = fusion.get(
            "max_uniform_share_if_signal_by_color", {}
        )
        relative_tilt_max_strength_by_color = fusion.get("relative_tilt_max_strength_by_color", {})
        relative_tilt_min_strength_by_color = fusion.get("relative_tilt_min_strength_by_color", {})
        relative_tilt_consistency_boost_by_color = fusion.get(
            "relative_tilt_consistency_boost_by_color", {}
        )
        relative_tilt_recovery_max_strength_by_color = fusion.get(
            "relative_tilt_recovery_max_strength_by_color", {}
        )
        relative_tilt_recovery_consistency_threshold_by_color = fusion.get(
            "relative_tilt_recovery_consistency_threshold_by_color", {}
        )
        red_weights, red_reason, _ = multiscale_shrinkage_weights(
            red_losses,
            baseline_loss=baselines["red"],
            invalid_models=invalid_models_by_color.get("red", set()),
            **options,
            skill_windows=skill_windows,
            weak_signal_min_windows=int(
                weak_signal_min_windows_by_color.get("red", fusion.get("weak_signal_min_windows", 2))
            ),
            weak_signal_max_strength=float(
                weak_signal_max_strength_by_color.get("red", fusion.get("weak_signal_max_strength", 0.25))
            ),
            weak_signal_min_strength=float(
                weak_signal_min_strength_by_color.get("red", fusion.get("weak_signal_min_strength", 0.0))
            ),
            max_uniform_share_if_signal=float(
                max_uniform_share_if_signal_by_color.get(
                    "red", fusion.get("max_uniform_share_if_signal", 1.0)
                )
            ),
            relative_tilt_max_strength=float(
                relative_tilt_max_strength_by_color.get("red", fusion.get("relative_tilt_max_strength", 0.0))
            ),
            relative_tilt_min_strength=float(
                relative_tilt_min_strength_by_color.get("red", fusion.get("relative_tilt_min_strength", 0.0))
            ),
            relative_tilt_consistency_boost=float(
                relative_tilt_consistency_boost_by_color.get(
                    "red", fusion.get("relative_tilt_consistency_boost", 0.0)
                )
            ),
            relative_tilt_recovery_max_strength=float(
                relative_tilt_recovery_max_strength_by_color.get(
                    "red", fusion.get("relative_tilt_recovery_max_strength", 0.0)
                )
            ),
            relative_tilt_recovery_consistency_threshold=float(
                relative_tilt_recovery_consistency_threshold_by_color.get(
                    "red", fusion.get("relative_tilt_recovery_consistency_threshold", 1.0)
                )
            ),
            weak_signal_recent_window_power=float(
                fusion.get("weak_signal_recent_window_power_by_color", {}).get(
                    "red", fusion.get("weak_signal_recent_window_power", 0.0)
                )
            ),
            relative_tilt_recent_window_power=float(
                fusion.get("relative_tilt_recent_window_power_by_color", {}).get(
                    "red", fusion.get("relative_tilt_recent_window_power", 0.0)
                )
            ),
        )
        blue_weights, blue_reason, _ = multiscale_shrinkage_weights(
            blue_losses,
            baseline_loss=baselines["blue"],
            invalid_models=invalid_models_by_color.get("blue", set()),
            **options,
            skill_windows=skill_windows,
            weak_signal_min_windows=int(
                weak_signal_min_windows_by_color.get("blue", fusion.get("weak_signal_min_windows", 2))
            ),
            weak_signal_max_strength=float(
                weak_signal_max_strength_by_color.get("blue", fusion.get("weak_signal_max_strength", 0.25))
            ),
            weak_signal_min_strength=float(
                weak_signal_min_strength_by_color.get("blue", fusion.get("weak_signal_min_strength", 0.0))
            ),
            max_uniform_share_if_signal=float(
                max_uniform_share_if_signal_by_color.get(
                    "blue", fusion.get("max_uniform_share_if_signal", 1.0)
                )
            ),
            relative_tilt_max_strength=float(
                relative_tilt_max_strength_by_color.get("blue", fusion.get("relative_tilt_max_strength", 0.0))
            ),
            relative_tilt_min_strength=float(
                relative_tilt_min_strength_by_color.get("blue", fusion.get("relative_tilt_min_strength", 0.0))
            ),
            relative_tilt_consistency_boost=float(
                relative_tilt_consistency_boost_by_color.get(
                    "blue", fusion.get("relative_tilt_consistency_boost", 0.0)
                )
            ),
            relative_tilt_recovery_max_strength=float(
                relative_tilt_recovery_max_strength_by_color.get(
                    "blue", fusion.get("relative_tilt_recovery_max_strength", 0.0)
                )
            ),
            relative_tilt_recovery_consistency_threshold=float(
                relative_tilt_recovery_consistency_threshold_by_color.get(
                    "blue", fusion.get("relative_tilt_recovery_consistency_threshold", 1.0)
                )
            ),
            weak_signal_recent_window_power=float(
                fusion.get("weak_signal_recent_window_power_by_color", {}).get(
                    "blue", fusion.get("weak_signal_recent_window_power", 0.0)
                )
            ),
            relative_tilt_recent_window_power=float(
                fusion.get("relative_tilt_recent_window_power_by_color", {}).get(
                    "blue", fusion.get("relative_tilt_recent_window_power", 0.0)
                )
            ),
        )
    else:
        validation_window = max(1, int(fusion.get("validation_periods", 30)))
        red_losses = _rolling_losses(red_losses, validation_window)
        blue_losses = _rolling_losses(blue_losses, validation_window)
        red_weights, red_reason = adaptive_weights(
            red_losses,
            baseline_loss=baselines["red"],
            invalid_models=invalid_models_by_color.get("red", set()),
            **options,
        )
        blue_weights, blue_reason = adaptive_weights(
            blue_losses,
            baseline_loss=baselines["blue"],
            invalid_models=invalid_models_by_color.get("blue", set()),
            **options,
        )
    caps_by_color = fusion.get("model_weight_caps_by_color", {})
    uniform_caps_by_color = fusion.get("model_uniform_caps_by_color", {})
    max_uniform_share_if_signal_by_color = fusion.get(
        "max_uniform_share_if_signal_by_color", {}
    )
    red_weights, red_cap_diagnostics = cap_model_weight_shares(
        red_weights,
        caps_by_color.get("red", {}),
        max_uniform_share=float(
            uniform_caps_by_color.get(
                "red",
                max_uniform_share_if_signal_by_color.get(
                    "red", fusion.get("max_uniform_share_if_signal", 1.0)
                ),
            )
        ),
        invalid_models=invalid_models_by_color.get("red", set()),
    )
    blue_weights, blue_cap_diagnostics = cap_model_weight_shares(
        blue_weights,
        caps_by_color.get("blue", {}),
        max_uniform_share=float(
            uniform_caps_by_color.get(
                "blue",
                max_uniform_share_if_signal_by_color.get(
                    "blue", fusion.get("max_uniform_share_if_signal", 1.0)
                ),
            )
        ),
        invalid_models=invalid_models_by_color.get("blue", set()),
    )
    if red_cap_diagnostics.get("applied"):
        red_reason = f"{red_reason};single_model_cap_applied"
    if blue_cap_diagnostics.get("applied"):
        blue_reason = f"{blue_reason};single_model_cap_applied"
    v7_config = config.get("v7") or {}
    v7_blue = v7_config.get("blue") or {}
    v7_governance = v7_config.get("governance") or {}
    if bool(v7_config.get("enabled", True)) and bool(v7_governance.get("expert_decorrelation", False)):
        blue_group_caps = {
            "machine_learning": float(v7_blue.get("max_model_weight", 0.35)),
            "statistical": float(v7_blue.get("max_statistical_weight", 0.60)),
            "baseline": float(v7_blue.get("max_baseline_weight", 1.0)),
        }
        blue_weights, blue_group_diagnostics = cap_blue_expert_groups(
            blue_weights,
            group_caps=blue_group_caps,
        )
        if blue_group_diagnostics.get("events"):
            blue_reason = f"{blue_reason};v7_blue_group_cap_applied"
    return {
        "red": _attach_structured_anchor_metadata(red_weights, color="red", fusion_config=fusion),
        "blue": _attach_structured_anchor_metadata(blue_weights, color="blue", fusion_config=fusion),
    }, {"red": red_reason, "blue": blue_reason}


def _rolling_red_adverse_guard(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    guard_config = config.get("red_adverse_guard", {})
    if not bool(guard_config.get("enabled", True)):
        return {"active": False, "reason": "disabled"}
    minimum_periods = int(
        guard_config.get(
            "backtest_minimum_periods",
            min(12, int(guard_config.get("minimum_periods", 20))),
        )
    )
    window = int(guard_config.get("backtest_window", max(minimum_periods, 12)))
    records = period_records[-window:]
    if len(records) < minimum_periods:
        return {
            "active": False,
            "reason": "insufficient_rolling_backtest_periods",
            "periods": len(records),
            "minimum_periods": minimum_periods,
        }
    average_margin = float(guard_config.get("backtest_red_average_margin", guard_config.get("red_average_margin", 0.03)))
    period_any_margin = float(
        guard_config.get(
            "backtest_period_any_margin",
            guard_config.get("period_any_red_ge3_margin", guard_config.get("period_any_margin", 0.03)),
        )
    )
    severe_average_multiplier = float(guard_config.get("backtest_severe_average_margin_multiplier", 2.0))
    severe_period_multiplier = float(guard_config.get("backtest_severe_period_any_margin_multiplier", 2.0))
    require_consensus_adverse = bool(guard_config.get("backtest_require_consensus_adverse", True))
    activate_on_soft_adverse = bool(guard_config.get("backtest_activate_on_soft_adverse", True))
    persist_periods = max(0, int(guard_config.get("backtest_persist_periods", 0)))
    required_methods = int(guard_config.get("backtest_minimum_adverse_methods", guard_config.get("minimum_adverse_methods", 2)))
    persist_require_consensus_adverse = bool(
        guard_config.get("backtest_persist_require_consensus_adverse", True)
    )
    persist_minimum_adverse_methods = int(
        guard_config.get("backtest_persist_minimum_adverse_methods", required_methods)
    )
    methods = [
        method
        for method in [*config["optimizer"]["methods"], "consensus"]
        if all(method in record.get("methods", {}) for record in records)
    ]
    diagnostics: dict[str, Any] = {}
    adverse_methods = 0
    severe_methods = 0
    for method in methods:
        model_average = float(np.mean([
            record["methods"][method]["model"]["average_red_hits"]
            for record in records
        ]))
        matched_average = float(np.mean([
            record["methods"][method]["matched_no_signal"]["average_red_hits"]
            for record in records
        ]))
        random_average = float(np.mean([
            record["methods"][method]["random"]["average_red_hits"]
            for record in records
        ]))
        model_any = float(np.mean([
            record["methods"][method]["model"]["any_red_ge3"]
            for record in records
        ]))
        matched_any = float(np.mean([
            record["methods"][method]["matched_no_signal"]["any_red_ge3"]
            for record in records
        ]))
        random_any = float(np.mean([
            record["methods"][method]["random"]["any_red_ge3"]
            for record in records
        ]))
        reference_average = max(matched_average, random_average)
        reference_any = max(matched_any, random_any)
        average_shortfall = reference_average - model_average
        period_any_shortfall = reference_any - model_any
        average_failed = average_shortfall >= average_margin
        period_failed = period_any_shortfall >= period_any_margin
        severe_average_failed = average_shortfall >= average_margin * severe_average_multiplier
        severe_period_failed = (
            period_any_shortfall >= period_any_margin * severe_period_multiplier
            and average_shortfall > 0
        )
        adverse = bool(
            (average_failed and period_failed)
            or severe_period_failed
        )
        severe = bool(
            (severe_average_failed and period_failed)
            or severe_period_failed
        )
        adverse_methods += int(adverse)
        severe_methods += int(severe)
        diagnostics[method] = {
            "adverse": adverse,
            "severe_adverse": severe,
            "average_failed": average_failed,
            "period_any_failed": period_failed,
            "severe_average_failed": severe_average_failed,
            "severe_period_any_failed": severe_period_failed,
            "model_average_red_hits": model_average,
            "reference_average_red_hits": reference_average,
            "average_shortfall": average_shortfall,
            "model_period_any_red_ge3_ratio": model_any,
            "reference_period_any_red_ge3_ratio": reference_any,
            "period_any_shortfall": period_any_shortfall,
        }
    consensus_adverse = bool(diagnostics.get("consensus", {}).get("adverse", False))
    consensus_severe = bool(diagnostics.get("consensus", {}).get("severe_adverse", False))
    severe_required_methods = int(guard_config.get("backtest_minimum_severe_methods", required_methods))
    soft_active = adverse_methods >= required_methods and (not require_consensus_adverse or consensus_adverse)
    severe_active = (
        soft_active
        and severe_methods >= severe_required_methods
        and (not require_consensus_adverse or consensus_severe)
    )
    active = bool(severe_active or (activate_on_soft_adverse and soft_active))
    persisted_from_recent_guard = False
    if not active and persist_periods > 0:
        recent_guards = [
            record.get("red_adverse_guard", {})
            for record in period_records[-persist_periods:]
        ]
        recent_severe_guard = any(
            bool(guard.get("severe_active")) for guard in recent_guards
        )
        current_still_adverse = adverse_methods >= persist_minimum_adverse_methods
        current_consensus_allows_persist = (
            not persist_require_consensus_adverse or consensus_adverse
        )
        persisted_from_recent_guard = bool(
            recent_severe_guard
            and current_still_adverse
            and current_consensus_allows_persist
        )
        active = bool(persisted_from_recent_guard)
    if active:
        if persisted_from_recent_guard:
            reason = "rolling_red_guard_persisted_after_recent_activation"
        elif severe_active:
            reason = f"rolling_red_severely_underperformed_in_{severe_methods}_methods"
        else:
            reason = f"rolling_red_soft_adverse_activated_in_{adverse_methods}_methods"
    elif soft_active:
        reason = "rolling_red_soft_adverse_diagnostic_only"
    elif adverse_methods >= required_methods and require_consensus_adverse and not consensus_adverse:
        reason = "rolling_red_guard_blocked_consensus_not_adverse"
    else:
        reason = "rolling_red_not_adverse_enough"
    return {
        "active": active,
        "reason": reason,
        "periods": len(records),
        "window": window,
        "required_methods": required_methods,
        "adverse_methods": adverse_methods,
        "severe_methods": severe_methods,
        "severe_required_methods": severe_required_methods,
        "soft_active": soft_active,
        "severe_active": severe_active,
        "activate_on_soft_adverse": activate_on_soft_adverse,
        "persisted_from_recent_guard": persisted_from_recent_guard,
        "persist_periods": persist_periods,
        "persist_require_consensus_adverse": persist_require_consensus_adverse,
        "persist_minimum_adverse_methods": persist_minimum_adverse_methods,
        "require_consensus_adverse": require_consensus_adverse,
        "methods": diagnostics,
    }


def _rolling_red_emergency_coverage_guard(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Short-window stop-loss for obvious red-side collapse.

    The longer adverse guard is deliberately conservative because it changes the
    final portfolio construction.  The recent backtests exposed another failure
    mode: several consecutive periods can be dramatically worse than the
    same-pipeline no-signal baseline before the long-window guard has enough
    evidence to activate.  This guard is therefore narrow and short-memory: it
    only reacts to repeated near-zero consensus output with a clear reference
    shortfall.
    """
    guard_config = config.get("red_adverse_guard", {})
    if not bool(guard_config.get("emergency_enabled", True)):
        return {"active": False, "reason": "disabled"}

    window = max(1, int(guard_config.get("emergency_window", 3)))
    minimum_periods = max(1, int(guard_config.get("emergency_minimum_periods", 2)))
    records = period_records[-window:]
    if len(records) < minimum_periods:
        return {
            "active": False,
            "reason": "insufficient_emergency_periods",
            "periods": len(records),
            "minimum_periods": minimum_periods,
        }

    consensus_records = [
        record.get("methods", {}).get("consensus", {})
        for record in records
        if "consensus" in record.get("methods", {})
    ]
    if len(consensus_records) < minimum_periods:
        return {
            "active": False,
            "reason": "missing_consensus_history",
            "periods": len(consensus_records),
            "minimum_periods": minimum_periods,
        }

    low_average_threshold = float(
        guard_config.get("emergency_low_average_red_hits", 0.55)
    )
    low_best_threshold = float(
        guard_config.get("emergency_low_best_red_hits", 1.05)
    )
    shortfall_threshold = float(
        guard_config.get("emergency_reference_shortfall", 0.45)
    )
    required_low_periods = max(
        1,
        int(guard_config.get("emergency_required_low_periods", minimum_periods)),
    )

    per_period: list[dict[str, Any]] = []
    low_periods = 0
    shortfall_periods = 0
    for record in consensus_records:
        model = record["model"]
        matched = record["matched_no_signal"]
        random = record["random"]
        model_average = float(model["average_red_hits"])
        model_best = float(model["best_red_hits"])
        reference_average = max(
            float(matched["average_red_hits"]),
            float(random["average_red_hits"]),
        )
        shortfall = reference_average - model_average
        low = bool(
            model_average <= low_average_threshold
            or model_best <= low_best_threshold
        )
        clear_shortfall = bool(shortfall >= shortfall_threshold)
        low_periods += int(low)
        shortfall_periods += int(clear_shortfall)
        per_period.append({
            "model_average_red_hits": model_average,
            "model_best_red_hits": model_best,
            "reference_average_red_hits": reference_average,
            "shortfall": shortfall,
            "low": low,
            "clear_shortfall": clear_shortfall,
        })

    mean_model_average = float(np.mean([
        item["model_average_red_hits"] for item in per_period
    ]))
    mean_reference_average = float(np.mean([
        item["reference_average_red_hits"] for item in per_period
    ]))
    active = bool(
        low_periods >= required_low_periods
        and shortfall_periods >= max(1, required_low_periods - 1)
    )
    return {
        "active": active,
        "reason": (
            "recent_consensus_red_collapse_vs_reference"
            if active
            else "recent_consensus_not_collapsed"
        ),
        "periods": len(consensus_records),
        "window": window,
        "required_low_periods": required_low_periods,
        "low_periods": low_periods,
        "shortfall_periods": shortfall_periods,
        "mean_model_average_red_hits": mean_model_average,
        "mean_reference_average_red_hits": mean_reference_average,
        "mean_shortfall": mean_reference_average - mean_model_average,
        "period_details": per_period,
    }


def _rolling_best_method_preference(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    optimizer = config.get("optimizer", {})
    if not bool(optimizer.get("rolling_best_method_enabled", True)):
        return {"active": False, "reason": "disabled"}
    minimum_periods = int(optimizer.get("rolling_best_method_min_periods", 6))
    window = int(optimizer.get("rolling_best_method_window", max(8, minimum_periods)))
    records = period_records[-window:]
    if len(records) < minimum_periods:
        return {
            "active": False,
            "reason": "insufficient_rolling_periods",
            "periods": len(records),
            "minimum_periods": minimum_periods,
        }
    methods = [
        method
        for method in [*optimizer.get("methods", []), "consensus"]
        if all(method in record.get("methods", {}) for record in records)
    ]
    if "consensus" not in methods:
        return {"active": False, "reason": "missing_consensus_history", "periods": len(records)}

    stats: dict[str, dict[str, float]] = {}
    for method in methods:
        model_average = float(np.mean([
            record["methods"][method]["model"]["average_red_hits"]
            for record in records
        ]))
        period_any = float(np.mean([
            record["methods"][method]["model"]["any_red_ge3"]
            for record in records
        ]))
        best_red = float(np.mean([
            record["methods"][method]["model"]["best_red_hits"]
            for record in records
        ]))
        blue_any = float(np.mean([
            record["methods"][method]["model"]["any_blue_hit"]
            for record in records
        ]))
        reference_average = float(np.mean([
            max(
                record["methods"][method]["matched_no_signal"]["average_red_hits"],
                record["methods"][method]["random"]["average_red_hits"],
            )
            for record in records
        ]))
        score = model_average + 0.22 * period_any + 0.08 * blue_any + 0.05 * best_red
        stats[method] = {
            "score": score,
            "model_average_red_hits": model_average,
            "period_any_red_ge3_ratio": period_any,
            "average_best_red_hits": best_red,
            "period_any_blue_hit_ratio": blue_any,
            "reference_average_red_hits": reference_average,
            "average_shortfall_to_reference": reference_average - model_average,
        }

    alternatives = [method for method in methods if method != "consensus"]
    if not alternatives:
        return {"active": False, "reason": "no_alternative_methods", "periods": len(records), "stats": stats}
    best_method = max(
        alternatives,
        key=lambda method: (
            stats[method]["score"],
            stats[method]["model_average_red_hits"],
            stats[method]["average_best_red_hits"],
        ),
    )
    score_margin = stats[best_method]["score"] - stats["consensus"]["score"]
    average_margin = (
        stats[best_method]["model_average_red_hits"]
        - stats["consensus"]["model_average_red_hits"]
    )
    period_any_margin = (
        stats[best_method]["period_any_red_ge3_ratio"]
        - stats["consensus"]["period_any_red_ge3_ratio"]
    )
    reference_advantage = (
        stats["consensus"]["average_shortfall_to_reference"]
        - stats[best_method]["average_shortfall_to_reference"]
    )
    require_reference_advantage = bool(
        optimizer.get("rolling_best_method_require_reference_advantage", True)
    )
    active = bool(
        (
            score_margin >= float(optimizer.get("rolling_best_method_score_margin", 0.08))
            and average_margin >= float(optimizer.get("rolling_best_method_average_margin", 0.12))
            and period_any_margin >= float(optimizer.get("rolling_best_method_period_any_margin", 0.03))
        )
        and (not require_reference_advantage or reference_advantage > 0.0)
    )
    return {
        "active": active,
        "reason": "rolling_method_outperformed_consensus" if active else "consensus_not_worse_than_alternatives",
        "method": best_method if active else None,
        "periods": len(records),
        "window": window,
        "score_margin": score_margin,
        "average_margin": average_margin,
        "period_any_margin": period_any_margin,
        "reference_advantage": reference_advantage,
        "stats": stats,
    }


def _model_status(
    models: dict,
    losses: dict[str, list[float]],
    red_losses: dict[str, list[float]],
    blue_losses: dict[str, list[float]],
    *,
    minimum_observations: int,
    gap_threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    status: dict[str, dict[str, Any]] = {}
    invalid_by_color = {"red": set(), "blue": set()}
    for name, model in models.items():
        model_disabled = bool(getattr(model, "disabled_", False))
        model_fit_failed = bool(getattr(model, "fit_failed_", False))
        model_has_no_training_metric = bool(
            name in ML_MODEL_NAMES and getattr(model, "training_brier_", None) is None
        )
        training = getattr(model, "training_brier_", None)
        walk_forward = float(np.mean(losses[name])) if losses.get(name) else None
        gap = walk_forward - float(training) if training is not None and walk_forward is not None else None
        training_by_color = getattr(model, "training_brier_by_color_", {})
        color_status: dict[str, dict[str, float | bool | None]] = {}
        for color, color_losses in (("red", red_losses), ("blue", blue_losses)):
            color_training = training_by_color.get(color)
            color_walk = (
                float(np.mean(color_losses[name]))
                if color_losses.get(name)
                else None
            )
            color_gap = (
                color_walk - float(color_training)
                if color_training is not None and color_walk is not None
                else None
            )
            color_invalid = bool(
                model_disabled
                or model_fit_failed
                or model_has_no_training_metric
                or (
                    color_gap is not None
                    and len(color_losses.get(name, [])) >= int(minimum_observations)
                    and color_gap > float(gap_threshold)
                )
            )
            if color_invalid:
                invalid_by_color[color].add(name)
            color_status[color] = {
                "training_brier": float(color_training) if color_training is not None else None,
                "walk_forward_brier": color_walk,
                "generalization_gap": color_gap,
                "invalid_for_overfit": color_invalid,
                "disabled": model_disabled,
                "fit_failed": model_fit_failed,
                "missing_training_metric": model_has_no_training_metric,
            }
        is_invalid = any(color_status[color]["invalid_for_overfit"] for color in ("red", "blue"))
        status[name] = {
            "training_brier": float(training) if training is not None else None,
            "walk_forward_brier": walk_forward,
            "generalization_gap": gap,
            "invalid_for_overfit": is_invalid,
            "disabled": model_disabled,
            "fit_failed": model_fit_failed,
            "missing_training_metric": model_has_no_training_metric,
            "last_fit_error": getattr(model, "last_fit_error_", None),
            "by_color": color_status,
        }
    return status, invalid_by_color


def _model_diagnostics(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    baselines = random_baseline_brier_components()
    report_config = config.get("report", {})
    significance_options = {
        "confidence_level": float(report_config.get("confidence_level", 0.95)),
        "permutation_repetitions": int(report_config.get("permutation_repetitions", 20000)),
    }
    by_color: dict[str, dict[str, Any]] = {}
    for color_index, color in enumerate(("red", "blue")):
        color_results: dict[str, dict[str, Any]] = {}
        for model_index, name in enumerate(MODEL_NAMES):
            losses = [
                float(record["model_validation"][name]["brier"][color])
                for record in period_records
            ]
            baseline_values = [float(baselines[color])] * len(losses)
            significance = paired_significance(
                baseline_values,
                losses,
                random_seed=seed_with_offset(
                    seed,
                    "blue_diagnostic_color",
                    color_index * 100_000,
                    model_index * 10_000,
                ),
                **significance_options,
            )
            average_loss = float(np.mean(losses))
            result = {
                "periods": len(losses),
                "average_brier": average_loss,
                "uniform_baseline_brier": float(baselines[color]),
                "brier_skill_score": float(1.0 - average_loss / float(baselines[color])),
                "brier_improvement": float(significance["mean_difference"]),
                "significance": significance,
            }
            if color == "red":
                result["average_top6_hits"] = float(np.mean([
                    record["model_validation"][name]["red_hits"]
                    for record in period_records
                ]))
            else:
                top1_hits = int(sum(
                    bool(record["model_validation"][name]["blue_hit"])
                    for record in period_records
                ))
                top1_hit_ratio = float(top1_hits / len(period_records)) if period_records else 0.0
                result["top1_hits"] = top1_hits
                result["top1_hit_ratio"] = top1_hit_ratio
                result["top1_significance"] = {
                    "sample_size": len(period_records),
                    "uniform_probability": 1.0 / 16.0,
                    "mean_difference": float(top1_hit_ratio - 1.0 / 16.0),
                    "test_method": "exact_binomial_greater",
                    "p_value": float(
                        binomtest(
                            top1_hits,
                            len(period_records),
                            1.0 / 16.0,
                            alternative="greater",
                        ).pvalue
                    ) if period_records else None,
                }
            color_results[name] = result
        apply_holm_correction(color_results)
        if color == "blue":
            apply_holm_correction(color_results, "top1_significance")
        for result in color_results.values():
            significance = result["significance"]
            result["proven_better_than_uniform"] = bool(
                significance["significant_after_holm_0_05"]
                and significance["mean_difference"] > 0
            )
            if color == "blue":
                top1_significance = result.get("top1_significance", {})
                result["top1_proven_better_than_uniform"] = bool(
                    top1_significance.get("significant_after_holm_0_05")
                    and top1_significance.get("mean_difference", 0.0) > 0
                )
        by_color[color] = color_results
    return {
        "primary_metric": "out_of_sample_brier_skill_vs_uniform",
        "multiple_testing": "Holm correction across five models per color",
        "by_color": by_color,
    }


def _pipeline_diagnostics(
    period_records: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Evaluate the frozen fusion procedure as one pre-specified system.

    The release decision concerns the procedure that produces the final
    combination, not whether one of its component models happens to win every
    historical slice.  Weight values may change only through the already
    specified, left-truncated online update rule.
    """
    baselines = random_baseline_brier_components()
    report_config = config.get("report", {})
    significance_options = {
        "confidence_level": float(report_config.get("confidence_level", 0.95)),
        "permutation_repetitions": int(report_config.get("permutation_repetitions", 20000)),
    }
    by_color: dict[str, dict[str, Any]] = {}
    for color_index, color in enumerate(("red", "blue")):
        losses = [
            float(record["fused_brier_components"][color])
            for record in period_records
        ]
        significance = paired_significance(
            [float(baselines[color])] * len(losses),
            losses,
            random_seed=seed_with_offset(seed, "blue_diagnostic_pipeline", color_index * 100_000),
            **significance_options,
        )
        result: dict[str, Any] = {
            "periods": len(losses),
            "average_brier": float(np.mean(losses)) if losses else None,
            "uniform_baseline_brier": float(baselines[color]),
            "brier_skill_score": (
                float(1.0 - float(np.mean(losses)) / float(baselines[color]))
                if losses else None
            ),
            "brier_improvement": float(significance["mean_difference"]),
            "significance": significance,
        }
        if color == "red":
            red_hits = [
                record["fused_validation"]["red_hits"]
                for record in period_records
            ]
            result["average_top6_hits"] = (
                float(np.mean([int(value) for value in red_hits if value is not None]))
                if any(value is not None for value in red_hits)
                else None
            )
            result["signal_periods"] = int(sum(value is not None for value in red_hits))
        else:
            blue_hits = [
                record["fused_validation"]["blue_hit"]
                for record in period_records
            ]
            signal_periods = int(sum(value is not None for value in blue_hits))
            top1_hits = int(sum(bool(value) for value in blue_hits))
            # Abstained periods are measured as coverage, not silently counted
            # as wrong ranked forecasts.  Top-1 accuracy is evaluated only for
            # periods in which the frozen procedure actually emitted a rank.
            top1_sample_size = signal_periods
            top1_hit_ratio = float(top1_hits / top1_sample_size) if top1_sample_size else 0.0
            result.update({
                "signal_periods": signal_periods,
                "top1_hits": top1_hits,
                "top1_hit_ratio": top1_hit_ratio,
                "top1_significance": {
                    "sample_size": top1_sample_size,
                    "uniform_probability": 1.0 / 16.0,
                    "mean_difference": float(top1_hit_ratio - 1.0 / 16.0),
                    "test_method": "exact_binomial_greater",
                    "p_value": float(
                        binomtest(
                            top1_hits,
                            top1_sample_size,
                            1.0 / 16.0,
                            alternative="greater",
                        ).pvalue
                    ) if top1_sample_size else None,
                },
            })
        by_color[color] = {"adaptive_fusion": result}
    for color, results in by_color.items():
        apply_holm_correction(results)
        if color == "blue":
            apply_holm_correction(results, "top1_significance")
        result = results["adaptive_fusion"]
        evidence = result["significance"]
        result["proven_better_than_uniform"] = bool(
            evidence["significant_after_holm_0_05"]
            and evidence["mean_difference"] > 0
        )
        if color == "blue":
            top1 = result["top1_significance"]
            result["top1_proven_better_than_uniform"] = bool(
                top1["significant_after_holm_0_05"]
                and top1["mean_difference"] > 0
            )
    return {
        "primary_metric": "frozen_adaptive_fusion_out_of_sample_brier_vs_uniform",
        "multiple_testing": "Holm correction across the pre-specified fusion procedure",
        "by_color": by_color,
    }


def run_walk_forward(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    periods: int | None = None,
) -> dict[str, Any]:
    backtest = config["backtest"]
    generation = config["generation"]
    optimizer_config = {
        **config["optimizer"],
        "_walk_forward_context": True,
    }
    scoring_config = config["scoring"]
    consensus_scoring_config = {
        **scoring_config,
        "signal_red_consensus_weight": float(
            optimizer_config.get("signal_red_consensus_weight", 0.0)
        ),
        "signal_red_consensus_pool_size": int(
            optimizer_config.get("signal_red_consensus_pool_size", 20)
        ),
        "signal_blue_consensus_pool_size": int(
            optimizer_config.get("signal_blue_consensus_pool_size", 20)
        ),
        "allow_blue_unique_relaxation": bool(
            optimizer_config.get("allow_blue_unique_relaxation", False)
        ),
        "blue_unique_relaxation_min_strength": float(
            optimizer_config.get("blue_unique_relaxation_min_strength", 0.85)
        ),
        "blue_unique_relaxation_floor": int(
            optimizer_config.get("blue_unique_relaxation_floor", 4)
        ),
    }
    minimum_train = int(backtest["min_train_size"])
    calibration = config.get("calibration", {})
    requested_periods = int(periods if periods is not None else calibration.get("holdout_periods", backtest["periods"]))
    red_candidate_limit = int(
        ((config.get("v7") or {}).get("red") or {}).get(
            "wide_candidate_limit",
            ((config.get("v7") or {}).get("red") or {}).get("candidate_set_limit", 22),
        )
        or 22
    )
    if requested_periods < 1:
        raise ValueError("验证期数必须为正整数")
    if len(frame) <= minimum_train:
        raise ValueError(f"严格回测至少需要 {minimum_train + 1} 期数据")
    available_holdout = len(frame) - minimum_train
    if available_holdout < requested_periods:
        raise ValueError(
            f"当前历史数据共 {len(frame)} 期，最小训练窗口为 {minimum_train} 期；"
            f"本次最多可验证 {available_holdout} 期，输入为 {requested_periods} 期"
        )
    start = max(minimum_train, len(frame) - requested_periods)
    target_indices = list(range(start, len(frame)))
    release_config = config.get("release", {})
    development_requested = max(0, int(release_config.get("development_periods", 0)))
    confirmatory_requested = max(0, int(release_config.get("confirmatory_periods", 0)))
    two_window_requested = bool(release_config.get("require_two_window_confirmation", False))
    release_windows_available = bool(
        two_window_requested
        and development_requested >= 1
        and confirmatory_requested >= 1
        and requested_periods >= development_requested + confirmatory_requested
    )
    freeze_confirmatory_fusion = bool(release_config.get("freeze_confirmatory_fusion_weights", True))
    frozen_confirmatory_weights_by_color = None
    confirmatory_start_round = (
        requested_periods - confirmatory_requested
        if release_windows_available
        else None
    )
    seed = int(config["project"]["random_seed"])
    output_count = int(generation["output_count"])
    consensus_count = int(generation.get("consensus_output_count", 3))
    matched_baseline_enabled = bool(backtest.get("matched_no_signal_baseline", True))
    loss_history = {name: [] for name in MODEL_NAMES}
    red_loss_history = {name: [] for name in MODEL_NAMES}
    blue_loss_history = {name: [] for name in MODEL_NAMES}
    fused_component_history = {"red": [], "blue": [], "combined": []}
    period_records: list[dict[str, Any]] = []
    final5_policy_history: list[dict[str, Any]] = []
    leakage_audit: list[dict[str, Any]] = []
    validation_trace_path = _validation_trace_path(config)
    _write_validation_trace(
        validation_trace_path,
        {
            "event": "validation_trace_run_start",
            "requested_holdout_periods": requested_periods,
            "completed_holdout_periods": 0,
            "first_validation_period": str(frame.iloc[target_indices[0]]["period"]),
            "last_validation_period": str(frame.iloc[target_indices[-1]]["period"]),
            "minimum_train_size": minimum_train,
            "output_count": output_count,
            "consensus_output_count": consensus_count,
            "trace_schema": "validation_period_trace_v1",
            "note": "后续每行对应一个严格Walk-forward验证期；用于复盘推荐原因与实际号码丢失阶段。",
        },
        append=False,
    )
    models_cache: dict | None = None
    last_fitted_models: dict | None = None
    calibrated_config = copy.deepcopy(config)
    failure_streak = 0
    last_parameter_adjustment_round = -10_000
    retrain_interval = max(1, int(backtest.get("retrain_interval", 1)))
    validation_window = max(1, int(config["fusion"].get("validation_periods", 30)))
    calibration_window = max(
        validation_window,
        int(calibration.get("rolling_objective_periods", validation_window)),
    )
    minimum_observations = int(config["fusion"]["minimum_observations"])
    gap_threshold = float(config["fusion"].get("overfit_gap_threshold", 0.05))
    warmup_requested = max(0, int(config["fusion"].get("warmup_periods", 0)))
    warmup_start = max(minimum_train, start - warmup_requested)
    warmup_indices = list(range(warmup_start, start))
    warmup_audit: list[dict[str, Any]] = []
    warmup_models_cache: dict | None = None
    if warmup_indices:
        logging.info(
            "融合预热开始 | 事件=FUSION_WARMUP_START | 预热期数=%s | 起始期=%s | 结束期=%s | 最终保留期未使用=True",
            len(warmup_indices),
            frame.iloc[warmup_indices[0]]["period"],
            frame.iloc[warmup_indices[-1]]["period"],
        )
        for warmup_number, target_position in enumerate(warmup_indices):
            history = frame.iloc[:target_position].copy()
            target = frame.iloc[target_position]
            target_idx = int(target["idx"])
            assert_left_truncated(history, target_idx)
            if warmup_models_cache is None or warmup_number % retrain_interval == 0:
                warmup_models_cache, predictions = _fit_and_predict(
                    history,
                    calibrated_config,
                    target_idx,
                    refit_ml=True,
                )
            else:
                warmup_models_cache, predictions = _fit_and_predict(
                    history,
                    calibrated_config,
                    target_idx,
                    warmup_models_cache,
                )
            actual_red = {int(target[column]) for column in RED_COLUMNS}
            actual_blue = int(target["blue"])
            for name, prediction in predictions.items():
                components = brier_components(prediction, actual_red, actual_blue)
                loss_history[name].append(components["combined"])
                red_loss_history[name].append(components["red"])
                blue_loss_history[name].append(components["blue"])
            audit = {
                "period": str(target["period"]),
                "target_idx": target_idx,
                "history_max_idx": int(history["idx"].max()),
                "is_before_holdout": target_position < start,
                "passed": int(history["idx"].max()) + 1 == target_idx and target_position < start,
            }
            warmup_audit.append(audit)
            logging.info(
                "融合预热完成 | 事件=FUSION_WARMUP_ROUND | 验证期=%s | 进度=%s/%s | 防泄漏=%s",
                target["period"],
                warmup_number + 1,
                len(warmup_indices),
                audit["passed"],
            )
    if warmup_models_cache is not None:
        models_cache = warmup_models_cache
    logging.info(
        "严格回测开始 | 事件=BACKTEST_START | 验证期数=%s | 融合预热期数=%s | 起始索引=%s | 结束索引=%s | 最小训练期数=%s | 随机对照次数=%s | 重训间隔=%s | 验证顺序=%s",
        len(target_indices),
        len(warmup_indices),
        target_indices[0],
        target_indices[-1],
        minimum_train,
        backtest["random_repeats"],
        retrain_interval,
        calibration.get("validation_order", "chronological_oldest_to_newest"),
    )
    if release_windows_available:
        logging.info(
            "发布证据窗口 | 事件=RELEASE_WINDOWS | 开发期数=%s | 确认期数=%s | 确认窗口从第%s个保留期开始，冻结超参数=%s",
            development_requested,
            confirmatory_requested,
            int(confirmatory_start_round) + 1,
            bool(release_config.get("require_frozen_confirmatory_parameters", True)),
        )

    for round_number, target_position in enumerate(target_indices):
        pulse_idle_sleep_block()
        round_started = time.perf_counter()
        history = frame.iloc[:target_position].copy()
        target = frame.iloc[target_position]
        target_idx = int(target["idx"])
        is_confirmatory_window = bool(
            release_windows_available
            and confirmatory_start_round is not None
            and round_number >= confirmatory_start_round
        )
        assert_left_truncated(history, target_idx)
        parameters_used = parameter_snapshot(calibrated_config)
        similar_context = most_similar_prior_context(
            frame,
            target_position=target_position,
            window=int(calibration.get("similarity_window", 30)),
        )
        if models_cache is None or round_number % retrain_interval == 0:
            models_cache, predictions = _fit_and_predict(
                history,
                calibrated_config,
                target_idx,
                refit_ml=True,
            )
        else:
            models_cache, predictions = _fit_and_predict(
                history,
                calibrated_config,
                target_idx,
                models_cache,
                refit_ml=False,
            )
        last_fitted_models = models_cache
        rolling_losses = _rolling_losses(loss_history, validation_window)
        rolling_red_losses = _rolling_losses(red_loss_history, validation_window)
        rolling_blue_losses = _rolling_losses(blue_loss_history, validation_window)
        pre_status, invalid_models_by_color = _model_status(
            models_cache,
            rolling_losses,
            rolling_red_losses,
            rolling_blue_losses,
            minimum_observations=minimum_observations,
            gap_threshold=gap_threshold,
        )
        weights_by_color, weight_reason_by_color = _adaptive_color_weights(
            red_loss_history,
            blue_loss_history,
            calibrated_config,
            invalid_models_by_color,
        )
        runtime_control = build_evidence_runtime_control(
            period_records,
            calibrated_config,
            ticket_count=int(optimizer_config.get("consensus_count", generation.get("ticket_count", 5))),
        )
        weights_by_color, evidence_weight_control = apply_evidence_weight_control(
            weights_by_color,
            runtime_control,
            calibrated_config,
        )
        if is_confirmatory_window and freeze_confirmatory_fusion:
            if frozen_confirmatory_weights_by_color is None:
                frozen_confirmatory_weights_by_color = copy.deepcopy(weights_by_color)
                logging.info(
                    "确认窗口融合冻结 | 事件=CONFIRMATORY_FUSION_FREEZE | 验证期=%s | 红球权重=%s | 蓝球权重=%s",
                    target["period"],
                    frozen_confirmatory_weights_by_color["red"],
                    frozen_confirmatory_weights_by_color["blue"],
                )
            weights_by_color = copy.deepcopy(frozen_confirmatory_weights_by_color)
            weight_reason_by_color = {
                "red": "confirmatory_frozen_weights",
                "blue": "confirmatory_frozen_weights",
            }
        weight_reason_by_color = {
            **weight_reason_by_color,
            "red": f"{weight_reason_by_color['red']};evidence_control={runtime_control.get('red_budget_state')}",
            "blue": f"{weight_reason_by_color['blue']};evidence_control={runtime_control.get('blue_edge_state')}",
        }
        weights = _aggregate_color_weights(weights_by_color)
        weight_reason = f"red={weight_reason_by_color['red']};blue={weight_reason_by_color['blue']}"
        if _log_detail_enabled(config, "model_detail"):
            logging.info(
                "融合评分 | 事件=FUSION_ROUND | 验证期=%s | 目标索引=%s | 红球权重原因=%s | 蓝球权重原因=%s | 红球权重=%s | 蓝球权重=%s | 失效模型=%s",
                target["period"],
                target_idx,
                weight_reason_by_color["red"],
                weight_reason_by_color["blue"],
                weights_by_color["red"],
                weights_by_color["blue"],
                {
                    color: sorted(names)
                    for color, names in invalid_models_by_color.items()
                },
            )
        else:
            logging.debug(
            "融合评分 | 事件=FUSION_ROUND | 验证期=%s | 目标索引=%s | 红球权重原因=%s | 蓝球权重原因=%s | 红球权重=%s | 蓝球权重=%s | 失效模型=%s",
            target["period"],
            target_idx,
            weight_reason_by_color["red"],
            weight_reason_by_color["blue"],
            weights_by_color["red"],
            weights_by_color["blue"],
            {
                color: sorted(names)
                for color, names in invalid_models_by_color.items()
            },
            )
        fused_base = fuse_predictions(predictions, weights_by_color)
        stacking_prediction, stacking_info = stacking_meta_predict(
            period_records,
            predictions,
            calibrated_config,
        )
        fused = blend_with_stacking(
            fused_base,
            stacking_prediction,
            calibrated_config,
            stacking_info=stacking_info,
        )
        if bool((config.get("v7") or {}).get("enabled", True)) and bool(
            ((config.get("v7") or {}).get("uncertainty") or {}).get("probability_shrinkage", True)
        ):
            alpha = float(runtime_control.get("blue_alpha", 0.5))
            shrunk_blue, shrinkage_info = shrink_blue_probability_to_uniform(
                fused["blue"],
                alpha=alpha,
                reason=f"v7_runtime_{runtime_control.get('blue_edge_state', 'NO_EDGE')}",
            )
            fused = {**fused, "blue": shrunk_blue}
            stacking_info["v7_blue_shrinkage"] = {
                **shrinkage_info,
                "edge_report": runtime_control.get("blue_edge", {}),
                "topk_policy": runtime_control.get("blue_prediction_set", {}).get("policy", {}),
                "production_candidate_width": runtime_control.get("blue_candidate_width"),
            }
        if bool((config.get("v7") or {}).get("enabled", True)):
            fused, red_fusion_recall_channel = apply_red_fusion_recall_channel(
                fused,
                predictions,
                config=calibrated_config,
            )
            stacking_info["v8_red_fusion_recall_channel"] = red_fusion_recall_channel
            if red_fusion_recall_channel.get("applied"):
                logging.info(
                    "红球融合召回通道已执行 | 事件=RED_FUSION_RECALL_CHANNEL | 验证期=%s | 诊断=%s",
                    target["period"],
                    {
                        "alpha": red_fusion_recall_channel.get("alpha"),
                        "added_to_top18": red_fusion_recall_channel.get("added_to_top18"),
                        "removed_from_top18": red_fusion_recall_channel.get("removed_from_top18"),
                    },
                )
        stacking_info["v7_runtime_control"] = runtime_control
        stacking_info["v7_evidence_weight_control"] = evidence_weight_control
        if stacking_info.get("applied"):
            weight_reason = f"{weight_reason};stacking=out_of_fold_meta_blend"
        actual_red = {int(target[column]) for column in RED_COLUMNS}
        actual_blue = int(target["blue"])
        model_rankings = {name: rank_prediction(prediction) for name, prediction in predictions.items()}
        model_blue_ranking_metrics = {
            name: blue_ranking_metrics(prediction["blue"], actual_blue)
            for name, prediction in predictions.items()
        }
        base_fused_ranking = rank_prediction(fused_base, weights_by_color=weights_by_color)
        base_fused_components = brier_components(fused_base, actual_red, actual_blue)
        base_fusion_validation = {
            "red_hits": (
                None
                if base_fused_ranking["red_is_uniform"]
                else int(len(set(base_fused_ranking["top_red"]) & actual_red))
            ),
            "blue_hit": (
                None
                if base_fused_ranking["blue_is_uniform"]
                else bool(base_fused_ranking["top_blue"][0] == actual_blue)
            ),
            "blue_ranking": blue_ranking_metrics(fused_base["blue"], actual_blue),
            "brier": base_fused_components,
        }
        fused_ranking = rank_prediction(fused, weights_by_color=weights_by_color)
        fused_blue_ranking_metrics = blue_ranking_metrics(fused["blue"], actual_blue)
        fused_validation = {
            "red_hits": (
                None
                if fused_ranking["red_is_uniform"]
                else int(len(set(fused_ranking["top_red"]) & actual_red))
            ),
            "blue_hit": (
                None
                if fused_ranking["blue_is_uniform"]
                else bool(fused_ranking["top_blue"][0] == actual_blue)
            ),
            "blue_ranking": fused_blue_ranking_metrics,
        }
        stacking_info["base_vs_final"] = {
            "base_red_hits": base_fusion_validation["red_hits"],
            "final_red_hits": fused_validation["red_hits"],
            "base_blue_rank": base_fusion_validation["blue_ranking"]["rank"],
            "final_blue_rank": fused_validation["blue_ranking"]["rank"],
            "base_blue_hit_at_5": base_fusion_validation["blue_ranking"]["hit_at_5"],
            "final_blue_hit_at_5": fused_validation["blue_ranking"]["hit_at_5"],
            "base_blue_brier": base_fused_components["blue"],
            "final_blue_brier": None,
        }

        red_round_guard = _rolling_red_adverse_guard(period_records, config)
        red_emergency_guard = _rolling_red_emergency_coverage_guard(period_records, config)
        emergency_shadow_only = bool(optimizer_config.get("red_emergency_guard_shadow_only", True))
        active_red_coverage_guard = (
            red_round_guard
            if red_round_guard.get("active")
            else (
                red_emergency_guard
                if red_emergency_guard.get("active") and not emergency_shadow_only
                else {
                    "active": False,
                    "reason": "red_emergency_guard_shadow_only",
                    "shadow_emergency_guard": red_emergency_guard,
                }
            )
        )
        generation_probabilities = fused
        if red_round_guard.get("active"):
            logging.info(
                "滚动红球逆向保护触发 | 事件=ROLLING_RED_ADVERSE_GUARD | 验证期=%s | 诊断=%s",
                target["period"],
                red_round_guard,
            )

        if red_emergency_guard.get("active") and not red_round_guard.get("active"):
            logging.info(
                "RED_EMERGENCY_GUARD shadow | period=%s | shadow_only=%s | diagnostics=%s",
                target["period"],
                emergency_shadow_only,
                red_emergency_guard,
            )

        red_candidate_recall_protection: dict[str, Any] = {
            "schema": "candidate_recall_protection_v8",
            "enabled": False,
            "reason": "not_evaluated",
        }
        candidate_generation_probabilities = generation_probabilities
        if active_red_coverage_guard.get("active"):
            emergency_active = bool(
                red_emergency_guard.get("active") and not red_round_guard.get("active")
            )
            red_relief = float(
                optimizer_config.get(
                    "red_emergency_guard_uniform_relief",
                    optimizer_config.get("red_adverse_guard_uniform_relief", 0.45),
                )
                if emergency_active
                else optimizer_config.get("red_adverse_guard_uniform_relief", 0.45)
            )
            guard_probabilities = {
                "red": _temper_with_uniform(generation_probabilities["red"], red_relief),
                "blue": generation_probabilities["blue"],
            }
            guard_probabilities, exclusion_diagnostics = apply_low_confidence_exclusion(
                guard_probabilities,
                history,
                calibrated_config,
                model_rankings=model_rankings,
            )
            guard_probabilities, red_candidate_recall_protection = apply_red_candidate_recall_protection(
                guard_probabilities,
                predictions,
                model_rankings=model_rankings,
                config=calibrated_config,
            )
            candidate_generation_probabilities = guard_probabilities
            red_guard_pool_size = int(generation.get(
                "red_adverse_guard_candidate_pool_size",
                max(int(generation["candidate_pool_size"]), 3000),
            ))
            guarded_pool, guarded_sampling = generate_candidates(
                guard_probabilities,
                history,
                samples=int(
                    generation.get(
                        "red_adverse_guard_monte_carlo_samples",
                        max(int(generation["monte_carlo_samples"]), 8000),
                    )
                ),
                pool_size=red_guard_pool_size,
                scoring_config=scoring_config,
                random_seed=seed_with_offset(seed, "red_guard_monte_carlo", round_number),
            )
            include_uniform_coverage = bool(generation.get(
                "red_adverse_guard_include_uniform_coverage_pool",
                False,
            ))
            coverage_pool: list[dict[str, Any]] = []
            coverage_sampling: dict[str, Any] = {
                "enabled": False,
                "reason": "uniform_coverage_pool_kept_as_baseline_only",
            }
            if include_uniform_coverage:
                coverage_samples = int(generation.get(
                    "red_adverse_guard_coverage_samples",
                    max(int(generation["monte_carlo_samples"]), int(generation.get("red_adverse_guard_monte_carlo_samples", 8000)) // 2),
                ))
                coverage_pool_size = int(generation.get(
                    "red_adverse_guard_coverage_pool_size",
                    max(int(generation["candidate_pool_size"]), red_guard_pool_size // 2),
                ))
                coverage_pool, coverage_sampling = generate_candidates(
                    _uniform_probabilities(),
                    history,
                    samples=coverage_samples,
                    pool_size=coverage_pool_size,
                    scoring_config=scoring_config,
                    random_seed=seed_with_offset(seed, "uniform_coverage_pool", round_number),
                )
            pool = _merge_candidate_pools(guarded_pool, coverage_pool)
            guard_mode = (
                (
                    "rolling_red_adverse_guard_hybrid_signal_plus_coverage"
                    if include_uniform_coverage
                    else "rolling_red_adverse_guard_tempered_signal_only"
                )
                if red_round_guard.get("active")
                else (
                    "rolling_red_emergency_guard_fast_coverage"
                    if include_uniform_coverage
                    else "rolling_red_emergency_guard_tempered_signal_only"
                )
            )
            sampling = {
                **guarded_sampling,
                "mode": guard_mode,
                "reason": (
                    "red_models_underperformed_matched_baseline; "
                    f"red_signal_tempered_toward_uniform_relief={red_relief:.2f}; "
                    + (
                        "merged_independent_uniform_coverage_candidates; "
                        if include_uniform_coverage
                        else "uniform_coverage_pool_used_as_baseline_not_candidates; "
                    )
                    + "blue_signal_is_retained"
                ),
                "red_uniform_relief": red_relief,
                "guarded_pool_candidates": len(guarded_pool),
                "coverage_pool_candidates": len(coverage_pool),
                "merged_candidate_pool": len(pool),
                "coverage_sampling": coverage_sampling,
                "red_adverse_guard": red_round_guard,
                "red_emergency_guard": red_emergency_guard,
                "red_candidate_recall_protection": red_candidate_recall_protection,
                "exclusion_filter": exclusion_diagnostics,
                "exclusion_actual_removed": {
                    "red": sorted(set(int(number) for number in exclusion_diagnostics.get("red_excluded", [])) & actual_red),
                    "blue": int(actual_blue) in set(int(number) for number in exclusion_diagnostics.get("blue_excluded", [])),
                },
            }
        else:
            generation_probabilities, exclusion_diagnostics = apply_low_confidence_exclusion(
                generation_probabilities,
                history,
                calibrated_config,
                model_rankings=model_rankings,
            )
            generation_probabilities, red_candidate_recall_protection = apply_red_candidate_recall_protection(
                generation_probabilities,
                predictions,
                model_rankings=model_rankings,
                config=calibrated_config,
            )
            candidate_generation_probabilities = generation_probabilities
            pool, sampling = generate_candidates(
                generation_probabilities,
                history,
                samples=int(generation["monte_carlo_samples"]),
                pool_size=int(generation["candidate_pool_size"]),
                scoring_config=scoring_config,
                random_seed=seed_with_offset(seed, "monte_carlo", round_number),
            )
            sampling = {
                **sampling,
                "red_candidate_recall_protection": red_candidate_recall_protection,
                "exclusion_filter": exclusion_diagnostics,
                "exclusion_actual_removed": {
                    "red": sorted(set(int(number) for number in exclusion_diagnostics.get("red_excluded", [])) & actual_red),
                    "blue": int(actual_blue) in set(int(number) for number in exclusion_diagnostics.get("blue_excluded", [])),
                },
            }
        pool, slot_enforcement = enforce_red_candidate_recall_slots(
            pool,
            candidate_generation_probabilities,
            history,
            red_candidate_recall_protection,
            pool_size=max(int(generation["candidate_pool_size"]), len(pool)),
            scoring_config=scoring_config,
        )
        red_candidate_recall_protection = {
            **red_candidate_recall_protection,
            "candidate_slot_enforcement": slot_enforcement,
        }
        sampling = {
            **sampling,
            "red_candidate_recall_protection": red_candidate_recall_protection,
        }
        logging.info(
            "候选池生成完成 | 事件=BACKTEST_SAMPLING | 验证期=%s | 统计=%s",
            target["period"],
            compact_sampling_log(sampling),
        )
        matched_pool: list[dict] = []
        matched_sampling: dict[str, int] | None = None
        if matched_baseline_enabled:
            matched_pool, matched_sampling = generate_candidates(
                _uniform_probabilities(),
                history,
                samples=int(generation["monte_carlo_samples"]),
                pool_size=int(generation["candidate_pool_size"]),
                scoring_config=scoring_config,
                random_seed=seed_with_offset(seed, "matched_no_signal_pool", round_number),
            )
            logging.info(
                "同流程无信号候选池完成 | 事件=MATCHED_BASELINE_SAMPLING | 验证期=%s | 统计=%s",
                target["period"],
                compact_sampling_log(matched_sampling),
            )
        output_random_mean = _mean_uniform_ticket_metrics(
            count=output_count,
            repeats=int(backtest["random_repeats"]),
            actual_red=actual_red,
            actual_blue=actual_blue,
            random_seed=seed_with_offset(seed, "output_random_mean", round_number * 1000),
        )
        methods: dict[str, Any] = {}
        selected_by_method: dict[str, list[dict]] = {}
        matched_selected_by_method: dict[str, list[dict]] = {}
        for method in optimizer_config["methods"]:
            selected = optimize_candidates(
                pool,
                output_count,
                method,
                optimizer_config,
                scoring_config,
                seed_with_offset(seed, "method_optimizer", round_number),
                history_frame=history,
            )
            selected_by_method[method] = selected
            model_metrics = evaluate_tickets(selected, actual_red, actual_blue)
            validated_tickets = validate_tickets(selected, actual_red, actual_blue)
            matched_selected = (
                optimize_candidates(
                    matched_pool,
                    output_count,
                    method,
                    optimizer_config,
                    scoring_config,
                    seed_with_offset(seed, "matched_optimizer_pool", round_number),
                    history_frame=history,
                )
                if matched_baseline_enabled
                else []
            )
            matched_selected_by_method[method] = matched_selected
            matched_metrics = (
                evaluate_tickets(matched_selected, actual_red, actual_blue)
                if matched_selected
                else output_random_mean
            )
            methods[method] = {
                "model": model_metrics,
                "matched_no_signal": matched_metrics,
                "random": output_random_mean,
                "tickets": validated_tickets,
                "matched_no_signal_tickets": validate_tickets(
                    matched_selected, actual_red, actual_blue
                ) if matched_selected else [],
            }
            (logging.info if _log_detail_enabled(config, "per_period_method_detail") else logging.debug)(
                "覆盖方法验证 | 事件=BACKTEST_METHOD | 验证期=%s | 方法=%s(%s) | 模型结果=%s | 同流程无信号=%s | 纯随机=%s | 各注命中=%s",
                target["period"],
                METHOD_LABELS.get(method, method),
                method,
                model_metrics,
                matched_metrics,
                output_random_mean,
                [(ticket["red_hits"], ticket["blue_hit"]) for ticket in validated_tickets],
            )

        if bool(optimizer_config.get("historical_hit_ablation_enabled", True)):
            ablation_method = "historical_hit_ablation_no_history"
            ablation_selected = historical_hit_optimize(
                pool,
                output_count,
                history_frame=history,
                window=int(optimizer_config.get("historical_hit_window", 300)),
                scoring_config=scoring_config,
                max_red_overlap=int(
                    optimizer_config.get("signal_max_red_overlap", optimizer_config.get("max_red_overlap", 5))
                ),
                minimum_blue_unique_count=int(
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", 5),
                    )
                ),
                history_weight=float(optimizer_config.get("historical_hit_ablation_history_weight", 0.0)),
                current_signal_weight=float(
                    optimizer_config.get("historical_hit_ablation_current_signal_weight", 0.55)
                ),
                coverage_weight=float(optimizer_config.get("historical_hit_ablation_coverage_weight", 0.45)),
            )
            ablation_metrics = evaluate_tickets(ablation_selected, actual_red, actual_blue)
            ablation_validated = validate_tickets(ablation_selected, actual_red, actual_blue)
            matched_ablation_selected = (
                historical_hit_optimize(
                    matched_pool,
                    output_count,
                    history_frame=history,
                    window=int(optimizer_config.get("historical_hit_window", 300)),
                    scoring_config=scoring_config,
                    max_red_overlap=int(
                        optimizer_config.get("signal_max_red_overlap", optimizer_config.get("max_red_overlap", 5))
                    ),
                    minimum_blue_unique_count=int(
                        optimizer_config.get(
                            "signal_minimum_blue_unique_count",
                            optimizer_config.get("minimum_blue_unique_count", 5),
                        )
                    ),
                    history_weight=float(optimizer_config.get("historical_hit_ablation_history_weight", 0.0)),
                    current_signal_weight=float(
                        optimizer_config.get("historical_hit_ablation_current_signal_weight", 0.55)
                    ),
                    coverage_weight=float(optimizer_config.get("historical_hit_ablation_coverage_weight", 0.45)),
                )
                if matched_baseline_enabled and matched_pool
                else []
            )
            methods[ablation_method] = {
                "model": ablation_metrics,
                "matched_no_signal": (
                    evaluate_tickets(matched_ablation_selected, actual_red, actual_blue)
                    if matched_ablation_selected
                    else output_random_mean
                ),
                "random": output_random_mean,
                "selection_strategy": "historical_hit_weight_ablation_no_history_signal_plus_coverage_only",
                "tickets": ablation_validated,
                "matched_no_signal_tickets": validate_tickets(
                    matched_ablation_selected, actual_red, actual_blue
                ) if matched_ablation_selected else [],
            }
            logging.info(
                "历史回放权重消融验证 | 事件=BACKTEST_METHOD_ABLATION | 验证期=%s | 方法=%s | 结果=%s | 各注命中=%s",
                target["period"],
                METHOD_LABELS.get(ablation_method, ablation_method),
                ablation_metrics,
                [(ticket["red_hits"], ticket["blue_hit"]) for ticket in ablation_validated],
            )

        if active_red_coverage_guard.get("active"):
            emergency_active = bool(
                red_emergency_guard.get("active") and not red_round_guard.get("active")
            )
            consensus_selected = adverse_guard_coverage_optimize(
                pool,
                selected_by_method,
                consensus_count,
                max_red_overlap=int(
                    optimizer_config.get(
                        "red_emergency_guard_max_red_overlap",
                        optimizer_config.get("red_adverse_guard_max_red_overlap", 4),
                    )
                    if emergency_active
                    else optimizer_config.get("red_adverse_guard_max_red_overlap", 4)
                ),
                minimum_blue_unique_count=int(
                    optimizer_config.get("red_adverse_guard_minimum_blue_unique_count", consensus_count)
                ),
                minimum_red_coverage=int(
                    optimizer_config.get(
                        "red_emergency_guard_minimum_red_coverage",
                        optimizer_config.get("red_adverse_guard_minimum_red_coverage", max(8, consensus_count + 3)),
                    )
                    if emergency_active
                    else optimizer_config.get("red_adverse_guard_minimum_red_coverage", max(8, consensus_count + 3))
                ),
                red_signal_weight=float(
                    optimizer_config.get(
                        "red_emergency_guard_red_signal_weight",
                        optimizer_config.get("red_adverse_guard_red_signal_weight", 0.08),
                    )
                    if emergency_active
                    else optimizer_config.get("red_adverse_guard_red_signal_weight", 0.08)
                ),
                blue_signal_weight=float(
                    optimizer_config.get("red_adverse_guard_blue_signal_weight", 0.55)
                ),
                method_anchor_count=int(
                    optimizer_config.get(
                        "red_emergency_guard_method_anchor_count",
                        optimizer_config.get("red_adverse_guard_method_anchor_count", 2),
                    )
                    if emergency_active
                    else optimizer_config.get("red_adverse_guard_method_anchor_count", 2)
                ),
                method_anchor_top_k_per_method=int(
                    optimizer_config.get("red_adverse_guard_method_anchor_top_k_per_method", 2)
                ),
                random_seed=seed_with_offset(seed, "weak_signal_monte_carlo", round_number),
            )
            consensus_strategy = (
                "rolling_red_emergency_guard_coverage_matrix"
                if emergency_active
                else "rolling_red_adverse_guard_coverage_matrix"
            )
        elif fused_ranking["is_uniform"]:
            consensus_selected = rotational_coverage_optimize(
                pool,
                consensus_count,
                max_red_overlap=int(optimizer_config.get("max_red_overlap", 4)),
                minimum_blue_unique_count=int(
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", 3),
                    )
                ),
                random_seed=seed_with_offset(seed, "weak_signal_core", round_number),
            )
            consensus_strategy = "no_signal_rotational_coverage"
        else:
            consensus_selected = consensus_optimize(
                pool,
                selected_by_method,
                consensus_count,
                scoring_config=consensus_scoring_config,
                max_red_overlap=int(
                    optimizer_config.get("signal_max_red_overlap", optimizer_config.get("max_red_overlap", 4))
                ),
                blue_diversity_weight=float(
                    optimizer_config.get("signal_blue_diversity_weight", optimizer_config.get("blue_diversity_weight", 0.30))
                ),
                minimum_blue_unique_count=int(
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", 3),
                    )
                ),
                concentration_weight=float(optimizer_config.get("signal_concentration_weight", 0.0)),
                minimum_unique_red_sets=int(
                    optimizer_config.get("signal_minimum_unique_red_sets", consensus_count)
                ),
                minimum_red_coverage=int(
                    optimizer_config.get("signal_minimum_red_coverage", max(6, consensus_count + 5))
                ),
                preserve_top_k=int(
                    optimizer_config.get("signal_preserve_top_ticket_count", 0)
                ),
                red_coverage_relief_max=float(
                    optimizer_config.get("signal_red_coverage_relief_max", 0.0)
                ),
                red_set_relief_max=float(
                    optimizer_config.get("signal_red_set_relief_max", 0.0)
                ),
                blue_focus_bonus_weight=float(
                    optimizer_config.get("signal_blue_focus_bonus_weight", 0.0)
                ),
                blue_unique_relief_max=float(
                    optimizer_config.get("signal_blue_unique_relief_max", 0.0)
                ),
                force_method_anchor_count=int(
                    optimizer_config.get("signal_force_method_anchor_count", 0)
                ),
                force_method_anchor_top_k_per_method=int(
                    optimizer_config.get("signal_force_method_anchor_top_k_per_method", 1)
                ),
                force_method_anchor_min_support=int(
                    optimizer_config.get("signal_force_method_anchor_min_support", 1)
                ),
                method_agreement_bonus_weight=float(
                    scoring_config.get("signal_method_agreement_bonus_weight", 0.0)
                ),
                red_core_bonus_weight=float(
                    optimizer_config.get("signal_red_core_bonus_weight", 0.0)
                ),
                red_core_top_k=int(
                    optimizer_config.get("signal_red_core_top_k", 6)
                ),
                top_consensus_margin_threshold=float(
                    optimizer_config.get("signal_top_consensus_margin_threshold", 0.0)
                ),
                top_consensus_overlap_bonus_weight=float(
                    optimizer_config.get("signal_top_consensus_overlap_bonus_weight", 0.0)
                ),
                allow_concentrated_portfolio=bool(
                    optimizer_config.get("signal_allow_concentrated_portfolio", False)
                ),
                concentrated_ticket_count=int(
                    optimizer_config.get("signal_concentrated_ticket_count", max(1, consensus_count // 2))
                ),
            )
            consensus_strategy = "model_signal_consensus"
        red_guard_stoploss = _red_guard_stoploss_decision(
            active_red_coverage_guard,
            optimizer_config,
        )
        if (
            red_guard_stoploss.get("active")
            and matched_pool
        ):
            consensus_selected = rotational_coverage_optimize(
                matched_pool,
                consensus_count,
                max_red_overlap=int(optimizer_config.get("max_red_overlap", 4)),
                minimum_blue_unique_count=int(
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", 3),
                    )
                ),
                random_seed=seed_with_offset(seed, "weak_release_pool", round_number),
            )
            consensus_selected = [
                {
                    **ticket,
                    "selection_role": "red_guard_no_signal_stoploss",
                    "selection_reason": red_guard_stoploss,
                }
                for ticket in consensus_selected
            ]
            consensus_strategy = "red_guard_no_signal_stoploss"
            logging.info(
                "红球逆信号止损最终层接管 | 事件=RED_GUARD_STOPLOSS_SELECTED | 验证期=%s | 原因=%s | 注数=%s",
                target["period"],
                red_guard_stoploss,
                len(consensus_selected),
            )
        elif (
            active_red_coverage_guard.get("active")
            and bool(optimizer_config.get("red_guard_prefer_method_portfolio", True))
        ):
            fallback_method = str(optimizer_config.get("red_guard_default_method", "greedy"))
            if fallback_method in selected_by_method and selected_by_method.get(fallback_method):
                consensus_selected = [
                    {
                        **ticket,
                        "selection_role": f"red_guard_method_portfolio_{fallback_method}",
                        "selection_reason": {
                            "reason": "red_guard_prefers_existing_method_portfolio_over_new_coverage_matrix",
                            "guard": active_red_coverage_guard,
                        },
                    }
                    for ticket in selected_by_method[fallback_method][:consensus_count]
                ]
                consensus_strategy = f"red_guard_method_portfolio_{fallback_method}"
                logging.info(
                    "红球保护最终层采用既有方法组合 | 事件=RED_GUARD_METHOD_PORTFOLIO_SELECTED | 验证期=%s | 方法=%s | 诊断=%s",
                    target["period"],
                    fallback_method,
                    active_red_coverage_guard,
                )
        rolling_preference = _rolling_best_method_preference(period_records, config)
        if (
            rolling_preference.get("active")
            and rolling_preference.get("method") in selected_by_method
            and selected_by_method.get(str(rolling_preference.get("method")))
        ):
            preferred_method = str(rolling_preference["method"])
            consensus_selected = [
                {
                    **ticket,
                    "selection_role": f"rolling_best_method_{preferred_method}",
                    "selection_reason": rolling_preference,
                }
                for ticket in selected_by_method[preferred_method][:consensus_count]
            ]
            consensus_strategy = f"rolling_best_method_{preferred_method}"
            logging.info(
                "滚动择优最终层接管 | 事件=ROLLING_BEST_METHOD_SELECTED | 验证期=%s | 方法=%s | 诊断=%s",
                target["period"],
                preferred_method,
                rolling_preference,
            )
        v7_portfolio_info: dict[str, Any] = {"active": False, "reason": "v7_portfolio_disabled"}
        if bool(config.get("v7", {}).get("enabled", True)) and bool(
            config.get("v7", {}).get("red", {}).get("dynamic_portfolio", True)
        ) and pool:
            protected_numbers = sorted({
                *[
                    int(number)
                    for number in runtime_control.get("red_protected_numbers", []) or []
                    if 1 <= int(number) <= 33
                ],
                *[
                    int(number)
                    for number in red_candidate_recall_protection.get("protected_numbers", []) or []
                    if 1 <= int(number) <= 33
                ],
            })
            portfolio_config = {
                **config,
                "v7": {
                    **(config.get("v7") or {}),
                    "current_red_edge_state": runtime_control.get("red_budget_state", runtime_control.get("red_edge_state", "NO_EDGE")),
                    "red_budget": runtime_control.get("red_budget", {}),
                    "red_protected_numbers": protected_numbers,
                    "red_candidate_recall_protection": red_candidate_recall_protection,
                    "red_runtime_control": runtime_control,
                },
            }
            consensus_selected, v7_portfolio_info = select_red_portfolio(
                pool,
                selected_by_method,
                consensus_count,
                portfolio_config,
            )
            consensus_strategy = f"{consensus_strategy}_v7_portfolio"
            logging.info(
                "V7红球Portfolio已接管最终层 | 事件=V7_RED_PORTFOLIO | 验证期=%s | 诊断=%s",
                target["period"],
                _compact_runtime_log(v7_portfolio_info),
            )
        if v7_portfolio_info.get("active"):
            rebalance_info = {"active": False, "reason": "v7_portfolio_is_final_layer"}
        else:
            consensus_selected, rebalance_info = rebalance_final_portfolio(
                pool,
                consensus_selected,
                count=consensus_count,
                optimizer_config=optimizer_config,
                random_seed=seed_with_offset(seed, "final_diversity_rebalance", round_number),
                reason_prefix="rolling_final_diversity_rebalance",
            )
        if rebalance_info.get("active"):
            consensus_strategy = "rolling_final_diversity_rebalance"
            logging.info(
                "逐期最终组合覆盖再平衡接管 | 事件=FINAL_DIVERSITY_REBALANCE | 验证期=%s | 诊断=%s",
                target["period"],
                rebalance_info,
            )
        portfolio_stage_selected = [dict(ticket) for ticket in consensus_selected]
        consensus_selected, rolling_memory_info = apply_rolling_portfolio_memory(
            consensus_selected,
            pool,
            period_records,
            config=optimizer_config,
        )
        if rolling_memory_info.get("active"):
            consensus_strategy = f"{consensus_strategy}_with_rolling_memory"
            logging.info(
                "逐期滚动组合记忆已参与最终组合 | 事件=ROLLING_PORTFOLIO_MEMORY_APPLIED | 验证期=%s | 诊断=%s",
                target["period"],
                rolling_memory_info,
            )
        period_red_candidate_limit = red_candidate_limit
        final5_prepared = None
        final5_applied = False
        hit_cfg = ((config.get("v11_7") or {}).get("hit_coherence") or {})
        if bool(hit_cfg.get("enabled", True)) and consensus_selected:
            final5_started = time.perf_counter()
            hit_width = int(hit_cfg.get("default_width", period_red_candidate_limit) or period_red_candidate_limit)
            coherence_tickets, coherence_info = apply_hit_coherence_final5(
                consensus_selected,
                fusion_red=fused.get("red"),
                history_draws=red_draws_from_frame(history),
                method_tickets=selected_by_method,
                pool=pool,
                model_probabilities=predictions,
                rescue_numbers=red_candidate_recall_protection.get("protected_numbers", []),
                width=hit_width,
                count=consensus_count,
                policy_history=final5_policy_history,
                min_periods=int(hit_cfg.get("min_periods", 8)),
                recall_drop_limit=float(hit_cfg.get("recall_drop_limit", 0.05)),
            )
            final5_prepared = coherence_info.get("prepared")
            final5_applied = bool(hit_cfg.get("apply_to_final5", True) and coherence_info.get("active"))
            if final5_applied:
                consensus_selected = coherence_tickets
                period_red_candidate_limit = int(coherence_info.get("width") or period_red_candidate_limit)
                consensus_strategy = f"{consensus_strategy}_v11_7_{coherence_info.get('strategy')}"
                logging.info(
                    "Final5改为命中共现组票 | 事件=FINAL5_HIT_COHERENCE_SELECTED | 验证期=%s | 策略=%s | 宽度=%s | 原因=%s | 组票耗时=%.3f秒 | 共现组=%s",
                    target["period"],
                    coherence_info.get("strategy"),
                    coherence_info.get("width"),
                    coherence_info.get("reason"),
                    time.perf_counter() - final5_started,
                    coherence_info.get("groups"),
                )
        effective_blue_upgrade = config.get("upgrade", {})
        if bool((config.get("v7") or {}).get("enabled", True)) and bool(
            ((config.get("v7") or {}).get("blue") or {}).get("adaptive_top_k", True)
        ):
            blue_edge_state_for_policy = str(runtime_control.get("blue_edge_state", "NO_EDGE")).upper()
            blue_mode_by_state = (
                ((config.get("v7") or {}).get("blue") or {}).get("edge_coverage_mode_by_state")
                or {}
            )
            effective_blue_upgrade = {
                **effective_blue_upgrade,
                "blue_coverage_candidate_top_n": int(runtime_control.get("blue_candidate_width", effective_blue_upgrade.get("blue_coverage_candidate_top_n", 10))),
                "blue_coverage_policy": str(
                    blue_mode_by_state.get(
                        blue_edge_state_for_policy,
                        "no_edge_coverage_only"
                        if blue_edge_state_for_policy == "NO_EDGE"
                        else "weak_edge_mixed_coverage"
                        if blue_edge_state_for_policy == "WEAK_EDGE"
                        else "probability_optimized",
                    )
                ),
                "blue_no_edge_fast_runtime_path": True,
                "_v7_blue_prediction_set": runtime_control.get("blue_prediction_set", {}),
                "_v7_blue_topk_policy": (runtime_control.get("blue_prediction_set", {}) or {}).get("policy", {}),
                "_v7_blue_edge_report": runtime_control.get("blue_edge", {}),
                "_v7_runtime_control": runtime_control,
            }
        consensus_selected, blue_finalize_info = finalize_blue_portfolio(
            consensus_selected,
            history,
            optimizer_config={
                **optimizer_config,
                "_blue_research_cache_context": "backtest",
                "_blue_final_exposure_history": optimizer_config.setdefault("_blue_final_exposure_history", []),
                "_blue_specialist_research_cache": optimizer_config.setdefault("_blue_specialist_research_cache", {}),
            },
            upgrade_config=effective_blue_upgrade,
            exclusion_diagnostics=exclusion_diagnostics,
        )
        if blue_finalize_info.get("active"):
            logging.info(
                "蓝球最终链路已执行 | 事件=BLUE_FINALIZE_PORTFOLIO_SELECTED | 验证期=%s | 蓝球=%s | 诊断=%s",
                target["period"],
                [ticket.get("blue") for ticket in consensus_selected],
                _compact_runtime_log(blue_finalize_info),
            )
        final_blue_target = int(
            optimizer_config.get(
                "signal_minimum_blue_unique_count",
                optimizer_config.get("minimum_blue_unique_count", consensus_count),
            )
        )
        consensus_selected = _ensure_minimum_blue_diversity(
            consensus_selected,
            pool,
            minimum_unique_blues=min(consensus_count, final_blue_target, 16),
        )
        consensus_selected, final_v7_guard_info = apply_final_v7_guard(
            consensus_selected,
            pool,
            runtime_control=runtime_control,
            minimum_unique_blues=min(consensus_count, final_blue_target, 16),
            v7_portfolio_info=v7_portfolio_info,
        )
        if final_v7_guard_info.get("active"):
            logging.info(
                "Final V7 Guard已执行 | 事件=FINAL_V7_GUARD | 验证期=%s | 诊断=%s",
                target["period"],
                _compact_runtime_log(final_v7_guard_info),
            )
        consensus_metrics = evaluate_tickets(consensus_selected, actual_red, actual_blue)
        consensus_validated = validate_tickets(consensus_selected, actual_red, actual_blue)
        candidate_recall_funnel = evaluate_candidate_recall(
            fused_probability=fused,
            model_probabilities=predictions,
            recall_protection=red_candidate_recall_protection,
            candidate_pool=pool,
            selected_by_method=selected_by_method,
            portfolio_tickets=portfolio_stage_selected,
            final_tickets=consensus_selected,
            red_candidate_limit=period_red_candidate_limit,
            actual_red=actual_red,
            actual_blue=actual_blue,
        )
        compression_experiment = _build_compression_experiment(
            pool=pool,
            selected_by_method=selected_by_method,
            portfolio_tickets=portfolio_stage_selected,
            final_tickets=consensus_selected,
            fused_probability=fused,
            actual_red=actual_red,
            actual_blue=actual_blue,
            count=consensus_count,
            optimizer_config=optimizer_config,
            random_seed=seed_with_offset(seed, "compression_experiment", round_number),
            red_candidate_limit=period_red_candidate_limit,
        )
        matched_consensus = (
            rotational_coverage_optimize(
                matched_pool,
                consensus_count,
                max_red_overlap=int(optimizer_config.get("max_red_overlap", 4)),
                minimum_blue_unique_count=int(
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", 3),
                    )
                ),
                random_seed=seed_with_offset(seed, "matched_consensus_pool", round_number),
            )
            if matched_baseline_enabled
            else []
        )
        matched_consensus_metrics = (
            evaluate_tickets(matched_consensus, actual_red, actual_blue)
            if matched_consensus
            else _mean_uniform_ticket_metrics(
                count=consensus_count,
                repeats=int(backtest["random_repeats"]),
                actual_red=actual_red,
                actual_blue=actual_blue,
                random_seed=seed_with_offset(seed, "random_baseline", round_number * 1000),
            )
        )
        consensus_random_mean = _mean_uniform_ticket_metrics(
            count=consensus_count,
            repeats=int(backtest["random_repeats"]),
            actual_red=actual_red,
            actual_blue=actual_blue,
            random_seed=seed_with_offset(seed, "consensus_random_baseline", round_number * 1000),
        )
        methods["consensus"] = {
            "model": consensus_metrics,
            "matched_no_signal": matched_consensus_metrics,
            "random": consensus_random_mean,
            "selection_strategy": consensus_strategy,
            "matched_selection_strategy": "no_signal_rotational_coverage",
            "tickets": consensus_validated,
            "matched_no_signal_tickets": validate_tickets(
                matched_consensus, actual_red, actual_blue
            ) if matched_consensus else [],
        }
        (logging.info if _log_detail_enabled(config, "per_period_method_detail") else logging.debug)(
            "最终组合验证 | 事件=BACKTEST_METHOD | 验证期=%s | 策略=%s | 模型结果=%s | 同流程轮转覆盖基线=%s | 纯随机=%s | 各注命中=%s",
            target["period"],
            consensus_strategy,
            consensus_metrics,
            matched_consensus_metrics,
            consensus_random_mean,
            [(ticket["red_hits"], ticket["blue_hit"]) for ticket in consensus_validated],
        )

        model_validation: dict[str, dict[str, Any]] = {}
        for name, prediction in predictions.items():
            components = brier_components(prediction, actual_red, actual_blue)
            loss_history[name].append(components["combined"])
            red_loss_history[name].append(components["red"])
            blue_loss_history[name].append(components["blue"])
            ranking = model_rankings[name]
            model_validation[name] = {
                "brier": components,
                "red_hits": int(len(set(ranking["top_red"]) & actual_red)),
                "blue_hit": bool(ranking["top_blue"][0] == actual_blue),
            }
        fused_components = brier_components(fused, actual_red, actual_blue)
        stacking_info.setdefault("base_vs_final", {})["final_blue_brier"] = fused_components["blue"]
        fused_brier = fused_components["combined"]
        for color, value in fused_components.items():
            fused_component_history[color].append(float(value))
        post_rolling_losses = _rolling_losses(loss_history, validation_window)
        post_red_losses = _rolling_losses(red_loss_history, validation_window)
        post_blue_losses = _rolling_losses(blue_loss_history, validation_window)
        post_status, post_invalid_models_by_color = _model_status(
            models_cache,
            post_rolling_losses,
            post_red_losses,
            post_blue_losses,
            minimum_observations=minimum_observations,
            gap_threshold=gap_threshold,
        )
        next_weights_by_color, next_weight_reason_by_color = _adaptive_color_weights(
            red_loss_history,
            blue_loss_history,
            calibrated_config,
            post_invalid_models_by_color,
        )
        next_weights = _aggregate_color_weights(next_weights_by_color)
        next_weight_reason = (
            f"red={next_weight_reason_by_color['red']};blue={next_weight_reason_by_color['blue']}"
        )
        rolling_fused_components = {
            color: float(np.mean(values[-calibration_window:]))
            for color, values in fused_component_history.items()
        }
        assessment = assess_validation(
            fused_brier=fused_brier,
            red_hits=fused_validation["red_hits"],
            blue_hit=fused_validation["blue_hit"],
            config=calibrated_config,
            rolling_brier_components=rolling_fused_components,
            rolling_periods=min(len(fused_component_history["combined"]), calibration_window),
        )
        failure_streak = (
            0
            if assessment["calibration_objective_passed"] or assessment.get("abstained")
            else failure_streak + 1
        )
        minimum_adjustment_observations = int(calibration.get("minimum_adjustment_observations", 10))
        minimum_failure_streak = int(calibration.get("minimum_failure_streak", 3))
        adjustment_cooldown = int(calibration.get("adjustment_cooldown_periods", 5))
        effective_adjustment_observations = len(warmup_indices) + round_number + 1
        enough_observations = effective_adjustment_observations >= minimum_adjustment_observations
        sustained_failure = failure_streak >= minimum_failure_streak
        cooldown_complete = round_number - last_parameter_adjustment_round >= adjustment_cooldown
        red_evidence_available = not bool(fused_ranking["red_is_uniform"])
        hyperparameter_adjustment_enabled = bool(
            calibration.get("enable_hyperparameter_adjustment", False)
        )
        allow_parameter_change = bool(
            hyperparameter_adjustment_enabled
            and not is_confirmatory_window
            and
            not assessment["calibration_objective_passed"]
            and red_evidence_available
            and enough_observations
            and sustained_failure
            and cooldown_complete
        )
        if is_confirmatory_window:
            parameter_gate_reason = "最终确认窗口冻结超参数，仅允许按既往损失更新融合权重"
        elif not hyperparameter_adjustment_enabled:
            parameter_gate_reason = "最终保留期冻结超参数，仅允许按既往损失更新融合权重"
        elif assessment["calibration_objective_passed"]:
            parameter_gate_reason = "滚动样本外概率损失未劣于基线，仅刷新权重"
        elif not red_evidence_available:
            parameter_gate_reason = "红球融合已退化为均匀分布，禁止通过调参拟合随机波动"
        elif not enough_observations:
            parameter_gate_reason = (
                f"预热与累计验证合计不足{minimum_adjustment_observations}期，"
                f"当前仅 {effective_adjustment_observations} 期，暂不调整超参数"
            )
        elif not sustained_failure:
            parameter_gate_reason = f"连续失败不足{minimum_failure_streak}期，避免追逐单期波动"
        elif not cooldown_complete:
            parameter_gate_reason = f"距上次调参不足{adjustment_cooldown}期，处于冷却期"
        else:
            parameter_gate_reason = "滚动概率损失持续劣于基线且样本、冷却条件满足，允许受限调参"
        calibrated_config, adjustment = adjust_after_validation(
            calibrated_config,
            assessment=assessment,
            rolling_losses=post_rolling_losses,
            model_status=post_status,
            rolling_losses_by_color={
                "red": post_red_losses,
                "blue": post_blue_losses,
            },
            allow_parameter_change=allow_parameter_change,
            gate_reason=parameter_gate_reason,
        )
        adjustment["next_weights"] = next_weights
        adjustment["next_weight_reason"] = next_weight_reason
        adjustment["next_weights_by_color"] = next_weights_by_color
        adjustment["next_weight_reason_by_color"] = next_weight_reason_by_color
        adjustment["failure_streak"] = failure_streak
        adjustment["effective_adjustment_observations"] = effective_adjustment_observations
        if adjustment["changes"]:
            last_parameter_adjustment_round = round_number
            logging.info(
                "校准参数已更新 | 事件=CALIBRATION_PARAMETERS_UPDATED | 验证期=%s | 变更数=%s | ML模型将在下一个计划重训点生效 | 重训间隔=%s",
                target["period"],
                len(adjustment["changes"]),
                retrain_interval,
            )
        logged_status = _calibration_log_status(
            assessment,
            allow_parameter_change=allow_parameter_change,
            adjustment=adjustment,
        )
        logging.info(
            "逐期校准 | 事件=CALIBRATION_ROUND | 验证期=%s | 验证状态=%s | Brier比值=%.6f | 参数门控=%s | 动作=%s | 变更=%s | 下一期红球权重=%s | 下一期蓝球权重=%s",
            target["period"],
            logged_status,
            assessment["brier_ratio"],
            parameter_gate_reason,
            adjustment["action"],
            adjustment["changes"],
            next_weights_by_color["red"],
            next_weights_by_color["blue"],
        )
        if final5_prepared is not None:
            final5_report = evaluate_final5_counterfactual(
                final5_prepared,
                actual_red,
                applied=final5_applied,
            )
            observation = final5_report.get("policy_observation")
            if observation:
                final5_policy_history.append(observation)
            production = final5_report.get("production") or {}
            logging.info(
                "Final5反事实对照 | 事件=FINAL5_DROP_REGRET | 验证期=%s | 策略=%s | 宽度=%s | CandidateHit=%s | Final5Hit=%s | DropRegret=%s | Any3=%s | Any4=%s | Best=%s | 归因=%s",
                target["period"],
                production.get("strategy"),
                (final5_report.get("policy") or {}).get("width"),
                production.get("candidate_hit"),
                production.get("final5_hit"),
                production.get("final5_drop_regret"),
                production.get("any3_plus"),
                production.get("any4_plus"),
                production.get("best_ticket_hit"),
                production.get("attribution"),
            )
            final5_prepared = None
        else:
            final5_report = {"schema": "final5_hit_coherence_v11_7", "active": False}
        period_record = {
            "period": str(target["period"]),
            "validation_period": str(target["period"]),
            "is_confirmatory_window": is_confirmatory_window,
            "target_idx": target_idx,
            "history_max_idx": int(history["idx"].max()),
            "training_periods": int(len(history)),
            "training_through_period": str(history.iloc[-1]["period"]),
            "weight_reason": weight_reason,
            "weights": weights,
            "weight_reason_by_color": weight_reason_by_color,
            "weights_by_color": weights_by_color,
            "runtime_control": runtime_control,
            "evidence_weight_control": evidence_weight_control,
            "parameters_used": parameters_used,
            "model_status_before_target": pre_status,
            "model_predictions": model_rankings,
            "model_blue_ranking_metrics": model_blue_ranking_metrics,
            "model_oof_probabilities": serialize_model_probabilities(predictions),
            "model_validation": model_validation,
            "fused_prediction": fused_ranking,
            "stacking": stacking_info,
            "actual": {
                "red": sorted(actual_red),
                "blue": actual_blue,
            },
            "base_fusion_validation": base_fusion_validation,
            "fused_validation": fused_validation,
            "fused_brier_components": fused_components,
            "red_adverse_guard": red_round_guard,
            "red_emergency_guard": red_emergency_guard,
            "red_candidate_recall_protection": red_candidate_recall_protection,
            "rolling_best_method_preference": rolling_preference,
            "rolling_portfolio_memory": rolling_memory_info,
            "final_v7_guard": final_v7_guard_info,
            "validation_assessment": assessment,
            "calibration_logged_status": logged_status,
            "calibration_adjustment": adjustment,
            "most_similar_prior_context": similar_context,
            "sampling": sampling,
            "matched_no_signal_sampling": matched_sampling,
            "methods": methods,
            "candidate_recall_funnel": candidate_recall_funnel,
            "compression_experiment": compression_experiment,
            "v7_portfolio": v7_portfolio_info,
            "final5_hit_coherence": final5_report,
        }
        period_records.append(period_record)
        _write_validation_trace(
            validation_trace_path,
            _build_validation_trace_record(
                target_period=str(target["period"]),
                training_period=str(history.iloc[-1]["period"]),
                target_idx=target_idx,
                history_max_idx=int(history["idx"].max()),
                actual_red=actual_red,
                actual_blue=actual_blue,
                final_tickets=consensus_validated,
                selected_by_method=selected_by_method,
                methods=methods,
                pool=pool,
                sampling=sampling,
                predictions=predictions,
                fused=fused,
                fused_ranking=fused_ranking,
                weights_by_color=weights_by_color,
                weight_reason_by_color=weight_reason_by_color,
                consensus_strategy=consensus_strategy,
                rebalance_info=rebalance_info,
                rolling_memory_info=rolling_memory_info,
                assessment=assessment,
                adjustment=adjustment,
                candidate_recall_funnel=candidate_recall_funnel,
                dantuo_config=config.get("dantuo", {}),
                blue_history=history["blue"].tolist(),
                final5_hit_coherence=final5_report,
            ),
            append=True,
        )
        leakage_audit.append({
            "period": str(target["period"]),
            "target_idx": target_idx,
            "history_max_idx": int(history["idx"].max()),
            "is_next_period": int(history["idx"].max()) + 1 == target_idx,
            "passed": int(history["idx"].max()) + 1 == target_idx,
        })
        logging.info(
            "单期验证完成 | 事件=BACKTEST_ROUND_DONE | 验证期=%s | 训练截止=%s | 进度=%s/%s | 耗时=%.3f秒 | 防泄漏=%s | 融合命中=%s | 各模型损失=%s",
            target["period"],
            history.iloc[-1]["period"],
            round_number + 1,
            len(target_indices),
            time.perf_counter() - round_started,
            leakage_audit[-1]["passed"],
            fused_validation,
            {name: values[-1] for name, values in loss_history.items()},
        )

    _write_validation_trace(
        validation_trace_path,
        {
            "event": "validation_trace_run_complete",
            "requested_holdout_periods": requested_periods,
            "completed_holdout_periods": len(period_records),
            "first_validation_period": period_records[0]["validation_period"] if period_records else None,
            "last_validation_period": period_records[-1]["validation_period"] if period_records else None,
            "strict_left_truncation_passed": all(item["passed"] for item in leakage_audit),
            "trace_schema": "validation_period_trace_v1",
        },
        append=True,
    )

    summary: dict[str, Any] = {}
    evaluation_methods = [*optimizer_config["methods"]]
    if period_records and all(
        "historical_hit_ablation_no_history" in record.get("methods", {})
        for record in period_records
    ):
        evaluation_methods.append("historical_hit_ablation_no_history")
    evaluation_methods.append("consensus")
    report_config = config.get("report", {})
    significance_options = {
        "confidence_level": float(report_config.get("confidence_level", 0.95)),
        "permutation_repetitions": int(report_config.get("permutation_repetitions", 20000)),
    }
    for method_index, method in enumerate(evaluation_methods):
        model_red = [record["methods"][method]["model"]["average_red_hits"] for record in period_records]
        matched_red = [
            record["methods"][method]["matched_no_signal"]["average_red_hits"]
            for record in period_records
        ]
        random_red = [record["methods"][method]["random"]["average_red_hits"] for record in period_records]
        model_best_red = [record["methods"][method]["model"]["best_red_hits"] for record in period_records]
        matched_best_red = [
            record["methods"][method]["matched_no_signal"]["best_red_hits"]
            for record in period_records
        ]
        summary[method] = {
            "periods": len(period_records),
            "ticket_count_per_period": consensus_count if method == "consensus" else output_count,
            "primary_baseline": "matched_no_signal_same_pipeline",
            "model_average_red_hits": float(np.mean(model_red)),
            "matched_no_signal_average_red_hits": float(np.mean(matched_red)),
            "random_average_red_hits": float(np.mean(random_red)),
            "red_ge3_ratio": float(np.mean([record["methods"][method]["model"]["red_ge3_ratio"] for record in period_records])),
            "red_ge4_ratio": float(np.mean([record["methods"][method]["model"]["red_ge4_ratio"] for record in period_records])),
            "random_red_ge3_ratio": float(np.mean([record["methods"][method]["random"]["red_ge3_ratio"] for record in period_records])),
            "random_red_ge4_ratio": float(np.mean([record["methods"][method]["random"]["red_ge4_ratio"] for record in period_records])),
            "matched_no_signal_red_ge3_ratio": float(np.mean([
                record["methods"][method]["matched_no_signal"]["red_ge3_ratio"] for record in period_records
            ])),
            "matched_no_signal_red_ge4_ratio": float(np.mean([
                record["methods"][method]["matched_no_signal"]["red_ge4_ratio"] for record in period_records
            ])),
            "period_any_red_ge3_ratio": float(np.mean([
                record["methods"][method]["model"]["any_red_ge3"] for record in period_records
            ])),
            "matched_no_signal_period_any_red_ge3_ratio": float(np.mean([
                record["methods"][method]["matched_no_signal"]["any_red_ge3"] for record in period_records
            ])),
            "random_period_any_red_ge3_ratio": float(np.mean([
                record["methods"][method]["random"]["any_red_ge3"] for record in period_records
            ])),
            "average_best_red_hits": float(np.mean(model_best_red)),
            "matched_no_signal_average_best_red_hits": float(np.mean(matched_best_red)),
            "blue_hit_ratio": float(np.mean([
                record["methods"][method]["model"]["blue_hit_ratio"] for record in period_records
            ])),
            "random_blue_hit_ratio": float(np.mean([
                record["methods"][method]["random"]["blue_hit_ratio"] for record in period_records
            ])),
            "period_any_blue_hit_ratio": float(np.mean([
                record["methods"][method]["model"]["any_blue_hit"] for record in period_records
            ])),
            "random_period_any_blue_hit_ratio": float(np.mean([
                record["methods"][method]["random"]["any_blue_hit"] for record in period_records
            ])),
            "matched_no_signal_period_any_blue_hit_ratio": float(np.mean([
                record["methods"][method]["matched_no_signal"]["any_blue_hit"] for record in period_records
            ])),
            "average_blue_coverage": float(np.mean([
                record["methods"][method]["model"]["blue_unique_count"] for record in period_records
            ])),
            "volatility": window_volatility(model_red),
            "significance": paired_significance(
                model_red,
                matched_red,
                random_seed=seed_with_offset(seed, "method_significance", method_index * 10_000),
                **significance_options,
            ),
            "pure_random_significance": paired_significance(
                model_red,
                random_red,
                random_seed=seed_with_offset(seed, "pure_random_significance", method_index * 10_000),
                **significance_options,
            ),
            "best_red_hits_significance": paired_significance(
                model_best_red,
                matched_best_red,
                random_seed=seed_with_offset(seed, "best_red_significance", method_index * 10_000),
                **significance_options,
            ),
        }
    apply_holm_correction(summary)
    apply_holm_correction(summary, "pure_random_significance")
    apply_holm_correction(summary, "best_red_hits_significance")
    model_diagnostics = _model_diagnostics(period_records, config, seed)
    release_evidence: dict[str, Any] = {
        "two_window_requested": two_window_requested,
        "development_periods_requested": development_requested,
        "confirmatory_periods_requested": confirmatory_requested,
        "available": release_windows_available,
        "selection_rule": "development_window_precedes_confirmatory_window; no overlap",
        "confirmatory_fusion_weights_frozen": bool(
            release_config.get("freeze_confirmatory_fusion_weights", True)
        ) if release_windows_available else False,
        "windows": {},
    }
    if release_windows_available:
        confirmatory_records = period_records[-confirmatory_requested:]
        development_end = len(period_records) - confirmatory_requested
        development_start = development_end - development_requested
        development_records = period_records[development_start:development_end]
        confirmatory_parameter_snapshots = {
            repr(record["parameters_used"])
            for record in confirmatory_records
        }
        release_evidence["windows"] = {
            "development": {
                "periods": len(development_records),
                "first_period": development_records[0]["validation_period"],
                "last_period": development_records[-1]["validation_period"],
                "model_diagnostics": _model_diagnostics(
                    development_records,
                    config,
                    seed_with_offset(seed, "development_model_diagnostics"),
                ),
                "pipeline_diagnostics": _pipeline_diagnostics(
                    development_records,
                    config,
                    seed_with_offset(seed, "development_pipeline_diagnostics"),
                ),
            },
            "confirmation": {
                "periods": len(confirmatory_records),
                "first_period": confirmatory_records[0]["validation_period"],
                "last_period": confirmatory_records[-1]["validation_period"],
                "model_diagnostics": _model_diagnostics(
                    confirmatory_records,
                    config,
                    seed_with_offset(seed, "confirmatory_model_diagnostics"),
                ),
                "pipeline_diagnostics": _pipeline_diagnostics(
                    confirmatory_records,
                    config,
                    seed_with_offset(seed, "confirmatory_pipeline_diagnostics"),
                ),
                "hyperparameters_frozen": len(confirmatory_parameter_snapshots) <= 1,
                "parameter_snapshot_count": len(confirmatory_parameter_snapshots),
                "frozen_pipeline_rule": "adaptive_color_fusion_with_left_truncated_online_weights",
                "online_weight_updates": "allowed_from_completed_prior_validation_only",
            },
        }
    final_rolling_losses = _rolling_losses(loss_history, validation_window)
    final_red_losses = _rolling_losses(red_loss_history, validation_window)
    final_blue_losses = _rolling_losses(blue_loss_history, validation_window)
    final_status, final_invalid_models_by_color = _model_status(
        last_fitted_models or {},
        final_rolling_losses,
        final_red_losses,
        final_blue_losses,
        minimum_observations=minimum_observations,
        gap_threshold=gap_threshold,
    )
    final_weights_by_color, final_weight_reason_by_color = _adaptive_color_weights(
        red_loss_history,
        blue_loss_history,
        calibrated_config,
        final_invalid_models_by_color,
    )
    final_runtime_control = build_evidence_runtime_control(
        period_records,
        calibrated_config,
        ticket_count=int(config.get("optimizer", {}).get("consensus_count", config.get("generation", {}).get("ticket_count", 5))),
    )
    final_weights_by_color, final_evidence_weight_control = apply_evidence_weight_control(
        final_weights_by_color,
        final_runtime_control,
        calibrated_config,
    )
    final_weight_reason_by_color = {
        **final_weight_reason_by_color,
        "red": f"{final_weight_reason_by_color['red']};evidence_control={final_runtime_control.get('red_budget_state')}",
        "blue": f"{final_weight_reason_by_color['blue']};evidence_control={final_runtime_control.get('blue_edge_state')}",
    }
    final_weights = _aggregate_color_weights(final_weights_by_color)
    final_signal_strength_by_color = {
        color: float(sum(final_weights_by_color[color].get(name, 0.0) for name in MODEL_NAMES))
        for color in ("red", "blue")
    }
    final_weight_reason = (
        f"red={final_weight_reason_by_color['red']};blue={final_weight_reason_by_color['blue']}"
    )
    ranked_red_hits = [
        record["fused_validation"]["red_hits"]
        for record in period_records
        if record["fused_validation"]["red_hits"] is not None
    ]
    ranked_blue_hits = [
        record["fused_validation"]["blue_hit"]
        for record in period_records
        if record["fused_validation"]["blue_hit"] is not None
    ]
    blue_ranking_rows = [
        record["fused_validation"].get("blue_ranking", {})
        for record in period_records
        if isinstance(record["fused_validation"].get("blue_ranking"), dict)
    ]
    blue_top1_probabilities = [
        float(row.get("top1_probability", 0.0))
        for row in blue_ranking_rows
    ]
    blue_top1_outcomes = [
        bool(record["fused_validation"]["blue_hit"])
        for record in period_records
        if record["fused_validation"].get("blue_hit") is not None
    ]
    fusion_validation_summary = {
        "periods": len(period_records),
        "ranked_red_periods": len(ranked_red_hits),
        "ranked_blue_periods": len(ranked_blue_hits),
        "average_red_hits": float(np.mean(ranked_red_hits)) if ranked_red_hits else None,
        "blue_hit_ratio": float(np.mean(ranked_blue_hits)) if ranked_blue_hits else None,
        "blue_hit_at_1": float(np.mean([row["hit_at_1"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_hit_at_3": float(np.mean([row["hit_at_3"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_hit_at_5": float(np.mean([row["hit_at_5"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_mrr": float(np.mean([row["mrr"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_ndcg_at_5": float(np.mean([row["ndcg_at_5"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_mean_rank": float(np.mean([row["mean_rank"] for row in blue_ranking_rows])) if blue_ranking_rows else None,
        "blue_top1_ece": expected_calibration_error(
            blue_top1_probabilities,
            blue_top1_outcomes,
        ) if blue_top1_probabilities and len(blue_top1_probabilities) == len(blue_top1_outcomes) else None,
    }
    calibration_summary = {
        "requested_holdout_periods": requested_periods,
        "completed_holdout_periods": len(period_records),
        "validation_order": calibration.get("validation_order", "chronological_oldest_to_newest"),
        "initial_training_periods": int(target_indices[0]),
        "initial_training_through_period": str(frame.iloc[target_indices[0] - 1]["period"]),
        "first_validation_period": str(frame.iloc[target_indices[0]]["period"]),
        "last_validation_period": str(frame.iloc[target_indices[-1]]["period"]),
        "quality_gate_passes": int(sum(record["validation_assessment"]["passed"] for record in period_records)),
        "quality_gate_failures": int(sum(
            record["validation_assessment"].get("status") == "failed"
            for record in period_records
        )),
        "calibration_objective_passes": int(sum(
            record["validation_assessment"].get("calibration_objective_passed", False)
            for record in period_records
        )),
        "calibration_objective_failures": int(sum(
            not record["validation_assessment"].get("calibration_objective_passed", False)
            and not record["validation_assessment"].get("abstained", False)
            for record in period_records
        )),
        "no_signal_abstentions": int(sum(
            record["validation_assessment"].get("status") == "abstained_no_signal"
            for record in period_records
        )),
        "parameter_adjustment_rounds": int(
            sum(bool(record["calibration_adjustment"]["changes"]) for record in period_records)
        ),
        "weight_refresh_rounds": len(period_records),
        "fusion_warmup_requested_periods": warmup_requested,
        "fusion_warmup_completed_periods": len(warmup_audit),
        "fusion_warmup_last_period": (
            str(frame.iloc[warmup_indices[-1]]["period"]) if warmup_indices else None
        ),
        "hyperparameters_frozen_during_holdout": not bool(
            calibration.get("enable_hyperparameter_adjustment", False)
        ),
        "confirmatory_periods": int(
            release_evidence.get("windows", {}).get("confirmation", {}).get("periods", 0)
        ),
        "confirmatory_hyperparameters_frozen": release_evidence.get(
            "windows", {}
        ).get("confirmation", {}).get("hyperparameters_frozen"),
        "future_validation_rows_used_for_adjustment": False,
        "final_parameters": parameter_snapshot(calibrated_config),
    }
    logging.info(
        "回测汇总 | 事件=BACKTEST_SUMMARY | 结果=%s | 红球最终权重=%s | 蓝球最终权重=%s | 权重原因=%s | 模型状态=%s",
        summary,
        final_weights_by_color["red"],
        final_weights_by_color["blue"],
        final_weight_reason,
        final_status,
    )
    try:
        research_benchmark = build_research_benchmark(
            period_records,
            random_seed=seed_with_offset(seed, "research_benchmark"),
            windows=(30, 50, 100, 200, 300, 500),
        )
    except Exception as error:
        logging.exception("研究基准构建失败 | 事件=RESEARCH_BENCHMARK_FAILED")
        research_benchmark = {
            "schema": "tsys_research_benchmark_v1",
            "available": False,
            "error": str(error),
            "windows_requested": [30, 50, 100, 200, 300, 500],
        }
    result = {
        "strict_left_truncation": (
            all(item["passed"] for item in leakage_audit)
            and all(item["passed"] for item in warmup_audit)
        ),
        "minimum_train_size": minimum_train,
        "random_repeats": int(backtest["random_repeats"]),
        "matched_no_signal_baseline": {
            "enabled": matched_baseline_enabled,
            "description": "均匀号码分布经过相同蒙特卡洛、结构评分和覆盖优化，用于隔离模型信号贡献",
        },
        "period_records": period_records,
        "validation_trace_path": str(validation_trace_path),
        "summary": summary,
        "fusion_validation_summary": fusion_validation_summary,
        "model_diagnostics": model_diagnostics,
        "model_scorecard": model_scorecard(period_records),
        "runtime_control": final_runtime_control,
        "evidence_weight_control": final_evidence_weight_control,
        "pipeline_diagnostics": _pipeline_diagnostics(
            period_records,
            config,
            seed_with_offset(seed, "final_pipeline_diagnostics"),
        ),
        "candidate_recall_funnel": summarize_candidate_recall_records(period_records),
        "final5_hit_coherence": summarize_final5_records(period_records),
        "red_method_ablation": summarize_red_method_ablation(period_records),
        "release_evidence": release_evidence,
        "research_benchmark": research_benchmark,
        "calibration_summary": calibration_summary,
        "model_brier_history": loss_history,
        "model_brier_history_by_color": {
            "red": red_loss_history,
            "blue": blue_loss_history,
        },
        "fused_brier_history_by_color": fused_component_history,
        "final_weights": final_weights,
        "final_weights_by_color": final_weights_by_color,
        "final_signal_strength_by_color": final_signal_strength_by_color,
        "final_weight_reason": final_weight_reason,
        "final_weight_reason_by_color": final_weight_reason_by_color,
        "final_parameters": parameter_snapshot(calibrated_config),
        "final_model_status": final_status,
        "fusion_warmup_audit": warmup_audit,
        "leakage_audit": leakage_audit,
    }
    result["v7_diagnostics"] = build_v7_diagnostics(result, config)
    return result
