from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from src.data_handler.cleaner import RED_COLUMNS
from src.fusion.model_fusion import random_baseline_brier, random_baseline_brier_components


def parameter_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the parameters that the online calibration loop may change."""
    return {
        "features": {
            "decay_factor": float(config["features"]["decay_factor"]),
            "training_lookback": int(config["features"]["training_lookback"]),
        },
        "logistic_regression": {
            "c": float(config["models"]["logistic_regression"]["c"]),
            "blue_c": float(config["models"]["logistic_regression"].get("blue_c", 0.05)),
        },
        "xgboost": {
            "max_depth": int(config["models"]["xgboost"]["max_depth"]),
            "learning_rate": float(config["models"]["xgboost"]["learning_rate"]),
            "reg_alpha": float(config["models"]["xgboost"]["reg_alpha"]),
            "reg_lambda": float(config["models"]["xgboost"]["reg_lambda"]),
        },
    }


def apply_parameter_snapshot(config: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["features"].update(snapshot.get("features", {}))
    result["models"]["logistic_regression"].update(snapshot.get("logistic_regression", {}))
    result["models"]["xgboost"].update(snapshot.get("xgboost", {}))
    return result


def assess_validation(
    *,
    fused_brier: float,
    red_hits: int | None,
    blue_hit: bool | None,
    config: dict[str, Any],
    rolling_brier_components: dict[str, float] | None = None,
    rolling_periods: int = 1,
) -> dict[str, Any]:
    calibration = config.get("calibration", {})
    baseline = random_baseline_brier()
    maximum_ratio = float(calibration.get("maximum_brier_ratio", 1.0))
    minimum_red_hits = int(calibration.get("minimum_red_hits", 1))
    require_blue_hit = bool(calibration.get("require_blue_hit", False))
    rolling = rolling_brier_components or {"combined": float(fused_brier)}
    rolling_combined = float(rolling.get("combined", fused_brier))
    brier_passed = rolling_combined <= baseline * maximum_ratio
    red_passed = red_hits is not None and int(red_hits) >= minimum_red_hits
    blue_passed = not require_blue_hit or blue_hit is True
    abstained = red_hits is None and blue_hit is None
    passed = bool(not abstained and brier_passed and red_passed and blue_passed)
    reasons: list[str] = []
    if abstained:
        reasons.append("no_signal_abstention")
    else:
        if not brier_passed:
            reasons.append("rolling_brier_above_threshold")
        if not red_passed:
            reasons.append("red_hits_below_minimum")
        if not blue_passed:
            reasons.append("blue_required_but_missed")
    return {
        "status": "abstained_no_signal" if abstained else "passed" if passed else "failed",
        "reason": "passed" if passed else ",".join(reasons) if reasons else "unknown",
        "failure_reasons": reasons,
        "abstained": abstained,
        "passed": passed,
        "fused_brier": float(fused_brier),
        "rolling_brier_components": {
            color: float(value) for color, value in rolling.items()
        },
        "rolling_brier_periods": int(rolling_periods),
        "rolling_random_baseline_brier_components": random_baseline_brier_components(),
        "random_baseline_brier": float(baseline),
        "brier_ratio": float(rolling_combined / baseline),
        "maximum_brier_ratio": maximum_ratio,
        "red_hits": red_hits,
        "minimum_red_hits": minimum_red_hits,
        "blue_hit": blue_hit,
        "require_blue_hit": require_blue_hit,
        "checks": {
            "brier": bool(brier_passed),
            "red_hits": bool(red_passed),
            "blue_hit": bool(blue_passed),
        },
        # Automatic parameter changes are based on the rolling proper-score
        # objective below, never on a single draw's hit count.
        "calibration_objective_passed": bool(brier_passed),
    }


def adjust_after_validation(
    config: dict[str, Any],
    *,
    assessment: dict[str, Any],
    rolling_losses: dict[str, list[float]],
    model_status: dict[str, dict[str, Any]],
    rolling_losses_by_color: dict[str, dict[str, list[float]]] | None = None,
    allow_parameter_change: bool = True,
    gate_reason: str = "quality_gate_failed",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adjust bounded hyperparameters using completed validations only.

    Fusion weights are refreshed elsewhere after every validation. Hyperparameters
    are changed only after the caller has failed the rolling out-of-sample proper
    score gate; hit counts are intentionally excluded from this decision.
    """
    updated = copy.deepcopy(config)
    calibration = config.get("calibration", {})
    before = parameter_snapshot(config)
    changes: list[dict[str, Any]] = []

    def change(path: str, old: float | int, new: float | int, reason: str) -> None:
        if new != old:
            changes.append({"parameter": path, "before": old, "after": new, "reason": reason})

    objective_passed = bool(
        assessment.get("calibration_objective_passed", assessment.get("passed", False))
    )
    if not objective_passed and allow_parameter_change:
        features = updated["features"]
        old_lookback = int(features["training_lookback"])
        new_lookback = min(
            int(calibration.get("maximum_training_lookback", 1200)),
            old_lookback + int(calibration.get("training_lookback_step", 100)),
        )
        features["training_lookback"] = new_lookback
        change(
            "features.training_lookback",
            old_lookback,
            new_lookback,
            "validation_failed_increase_training_stability",
        )

        frequency_losses = rolling_losses.get("frequency", [])
        decay_losses = rolling_losses.get("time_decay", [])
        if frequency_losses and decay_losses:
            frequency_loss = float(np.mean(frequency_losses))
            decay_loss = float(np.mean(decay_losses))
            old_decay = float(features["decay_factor"])
            step = float(calibration.get("decay_step", 0.005))
            tolerance = float(calibration.get("loss_tolerance", 1e-5))
            if decay_loss > frequency_loss + tolerance:
                new_decay = min(float(calibration.get("maximum_decay_factor", 0.995)), old_decay + step)
                reason = "time_decay_underperformed_reduce_recency_bias"
            else:
                new_decay = max(float(calibration.get("minimum_decay_factor", 0.95)), old_decay - step)
                reason = "time_decay_competitive_test_more_recency"
            features["decay_factor"] = round(new_decay, 6)
            change("features.decay_factor", old_decay, features["decay_factor"], reason)

        baselines = random_baseline_brier_components()
        combined_baseline = random_baseline_brier()
        color_losses = rolling_losses_by_color or {}
        lr_status = model_status.get("logistic_regression", {})
        lr_color_status = lr_status.get("by_color", {})
        lr_red_losses = color_losses.get("red", {}).get("logistic_regression", [])
        lr_blue_losses = color_losses.get("blue", {}).get("logistic_regression", [])
        lr_red_invalid = bool(lr_color_status.get("red", {}).get("invalid_for_overfit"))
        lr_blue_invalid = bool(lr_color_status.get("blue", {}).get("invalid_for_overfit"))
        if lr_red_invalid or (
            lr_red_losses and float(np.mean(lr_red_losses)) >= float(baselines["red"])
        ):
            lr = updated["models"]["logistic_regression"]
            old_c = float(lr["c"])
            new_c = max(float(calibration.get("minimum_lr_c", 0.05)), old_c * 0.8)
            lr["c"] = round(new_c, 6)
            change(
                "models.logistic_regression.c",
                old_c,
                lr["c"],
                "lr_red_failed_or_overfit_strengthen_regularization",
            )
        if lr_blue_invalid or (
            lr_blue_losses and float(np.mean(lr_blue_losses)) >= float(baselines["blue"])
        ):
            lr = updated["models"]["logistic_regression"]
            old_blue_c = float(lr.get("blue_c", 0.05))
            new_blue_c = max(
                float(calibration.get("minimum_lr_blue_c", calibration.get("minimum_lr_c", 0.02))),
                old_blue_c * 0.8,
            )
            lr["blue_c"] = round(new_blue_c, 6)
            change(
                "models.logistic_regression.blue_c",
                old_blue_c,
                lr["blue_c"],
                "lr_blue_failed_or_overfit_strengthen_regularization",
            )

        xgb_losses = rolling_losses.get("xgboost", [])
        xgb_invalid = bool(model_status.get("xgboost", {}).get("invalid_for_overfit"))
        if xgb_invalid or (xgb_losses and float(np.mean(xgb_losses)) >= combined_baseline):
            xgb = updated["models"]["xgboost"]
            old_depth = int(xgb["max_depth"])
            xgb["max_depth"] = max(int(calibration.get("minimum_xgb_depth", 2)), old_depth - 1)
            change("models.xgboost.max_depth", old_depth, xgb["max_depth"], "xgboost_failed_or_overfit_reduce_depth")
            old_rate = float(xgb["learning_rate"])
            xgb["learning_rate"] = round(max(float(calibration.get("minimum_xgb_learning_rate", 0.02)), old_rate * 0.9), 6)
            change("models.xgboost.learning_rate", old_rate, xgb["learning_rate"], "xgboost_failed_or_overfit_reduce_step")
            old_alpha = float(xgb["reg_alpha"])
            xgb["reg_alpha"] = round(min(float(calibration.get("maximum_xgb_reg_alpha", 2.0)), old_alpha * 1.2), 6)
            change("models.xgboost.reg_alpha", old_alpha, xgb["reg_alpha"], "xgboost_failed_or_overfit_strengthen_l1")
            old_lambda = float(xgb["reg_lambda"])
            xgb["reg_lambda"] = round(min(float(calibration.get("maximum_xgb_reg_lambda", 10.0)), old_lambda * 1.2), 6)
            change("models.xgboost.reg_lambda", old_lambda, xgb["reg_lambda"], "xgboost_failed_or_overfit_strengthen_l2")

    after = parameter_snapshot(updated)
    return updated, {
        "quality_gate_passed": bool(assessment["passed"]),
        "calibration_objective_passed": objective_passed,
        "action": "weights_refreshed_parameters_adjusted" if changes else "weights_refreshed_parameters_retained",
        "parameter_change_allowed": bool(allow_parameter_change),
        "parameter_gate_reason": str(gate_reason),
        "changes": changes,
        "parameters_before": before,
        "parameters_after": after,
        "uses_only_completed_validation": True,
    }


def most_similar_prior_context(
    frame: pd.DataFrame,
    *,
    target_position: int,
    window: int = 30,
) -> dict[str, Any] | None:
    """Find a prior pre-draw context without looking at the target or future rows."""
    window = int(window)
    if target_position <= window:
        return None
    available = frame.iloc[:target_position].reset_index(drop=True)
    red = np.zeros((len(available), 33), dtype=float)
    blue = np.zeros((len(available), 16), dtype=float)
    for row_position, row in available.iterrows():
        for column in RED_COLUMNS:
            red[row_position, int(row[column]) - 1] = 1.0
        blue[row_position, int(row["blue"]) - 1] = 1.0
    target_profile = np.concatenate([
        red[target_position - window:target_position].mean(axis=0),
        blue[target_position - window:target_position].mean(axis=0),
    ])
    target_norm = float(np.linalg.norm(target_profile))
    best: tuple[float, int] | None = None
    for candidate_position in range(window, target_position):
        profile = np.concatenate([
            red[candidate_position - window:candidate_position].mean(axis=0),
            blue[candidate_position - window:candidate_position].mean(axis=0),
        ])
        denominator = target_norm * float(np.linalg.norm(profile))
        similarity = float(np.dot(target_profile, profile) / denominator) if denominator else 0.0
        candidate = (similarity, candidate_position)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    similarity, candidate_position = best
    return {
        "period": str(available.iloc[candidate_position]["period"]),
        "idx": int(available.iloc[candidate_position]["idx"]),
        "history_through_period": str(available.iloc[candidate_position - 1]["period"]),
        "similarity": similarity,
        "window": window,
        "search_max_position": int(target_position - 1),
        "future_rows_used": False,
    }
