from __future__ import annotations

from math import erf, sqrt
from typing import Any

import numpy as np

from src.probability_model.bayes_model import BayesianModel
from src.probability_model.catboost_model import CatBoostProbabilityModel
from src.probability_model.fallback import DisabledProbabilityModel
from src.probability_model.freq_model import FrequencyModel
from src.probability_model.lr_model import LogisticRegressionModel
from src.probability_model.lightgbm_ranker import LightGBMRankerModel
from src.probability_model.mlp_model import MLPProbabilityModel
from src.probability_model.pattern_memory import PatternMemoryModel
from src.probability_model.time_decay import TimeDecayModel
from src.probability_model.xgb_model import XGBoostModel


MODEL_NAMES = [
    "frequency",
    "time_decay",
    "bayesian",
    "logistic_regression",
    "xgboost",
    "lightgbm_ranker",
    "mlp",
    "catboost",
    "pattern_memory",
]
DEFAULT_MODEL_GROUPS = (("frequency", "bayesian"),)
UNIFORM_ANCHOR = "uniform_baseline"
_MODEL_PRIOR_RANK = {
    "xgboost": 6.0,
    "lightgbm_ranker": 6.5,
    "mlp": 5.5,
    "catboost": 5.8,
    "logistic_regression": 5.0,
    "pattern_memory": 4.0,
    "time_decay": 3.0,
    "frequency": 2.0,
    "bayesian": 2.0,
}
ANCHOR_MODE_KEY = "__anchor_mode__"
ANCHOR_TOP_K_KEY = "__anchor_top_k__"
ANCHOR_HEAD_SHARE_KEY = "__anchor_head_share__"
ANCHOR_MIN_SIGNAL_KEY = "__anchor_min_signal__"
ENTROPY_SHARPEN_THRESHOLD_KEY = "__entropy_sharpen_threshold__"
ENTROPY_SHARPEN_POWER_KEY = "__entropy_sharpen_power__"
ENTROPY_SHARPEN_MIN_SIGNAL_KEY = "__entropy_sharpen_min_signal__"


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    features = {
        **config["features"],
        "cross_red_blue_feature": bool(
            config.get("upgrade", {}).get("cross_red_blue_feature", False)
        ),
    }
    seed = int(config["project"]["random_seed"])
    lr_config = {**config["models"]["logistic_regression"], "random_seed": seed}
    xgb_config = {**config["models"]["xgboost"], "random_seed": seed}
    lgbm_config = {**config["models"].get("lightgbm_ranker", {}), "random_seed": seed}
    mlp_config = {**config["models"].get("mlp", {}), "random_seed": seed}
    catboost_config = {**config["models"].get("catboost", {}), "random_seed": seed}
    pattern_config = config["models"].get("pattern_memory", {})
    mlp_model = (
        MLPProbabilityModel(features, mlp_config)
        if bool(mlp_config.get("enabled", False))
        else DisabledProbabilityModel("mlp", "mlp_disabled_by_config")
    )
    catboost_model = (
        CatBoostProbabilityModel(features, catboost_config)
        if bool(catboost_config.get("enabled", False))
        else DisabledProbabilityModel("catboost", "catboost_disabled_by_config")
    )
    return {
        "frequency": FrequencyModel(),
        "time_decay": TimeDecayModel(features["decay_factor"]),
        "bayesian": BayesianModel(),
        "logistic_regression": LogisticRegressionModel(features, lr_config),
        "xgboost": XGBoostModel(features, xgb_config),
        "lightgbm_ranker": LightGBMRankerModel(features, lgbm_config),
        "mlp": mlp_model,
        "catboost": catboost_model,
        "pattern_memory": PatternMemoryModel(features, pattern_config),
    }


def brier_components(
    prediction: dict[str, np.ndarray],
    actual_red: set[int],
    actual_blue: int,
) -> dict[str, float]:
    red_labels = np.array([number in actual_red for number in range(1, 34)], dtype=float)
    blue_labels = np.array([number == actual_blue for number in range(1, 17)], dtype=float)
    red_probability = np.clip(np.asarray(prediction["red"]) * 6.0, 0.0, 1.0)
    blue_probability = np.clip(np.asarray(prediction["blue"]), 0.0, 1.0)
    red_loss = float(np.mean((red_probability - red_labels) ** 2))
    blue_loss = float(np.mean((blue_probability - blue_labels) ** 2))
    return {"red": red_loss, "blue": blue_loss, "combined": (red_loss + blue_loss) / 2.0}


def random_baseline_brier() -> float:
    return random_baseline_brier_components()["combined"]


def random_baseline_brier_components() -> dict[str, float]:
    red_labels = np.array([1.0] * 6 + [0.0] * 27)
    blue_labels = np.array([1.0] + [0.0] * 15)
    red_loss = float(np.mean((6.0 / 33.0 - red_labels) ** 2))
    blue_loss = float(np.mean((1.0 / 16.0 - blue_labels) ** 2))
    return {"red": red_loss, "blue": blue_loss, "combined": (red_loss + blue_loss) / 2.0}


def adaptive_weights(
    loss_history: dict[str, list[float]] | None,
    *,
    minimum_observations: int = 3,
    invalid_models: set[str] | None = None,
    baseline_loss: float | None = None,
    uncertainty_z: float = 0.0,
    model_groups: tuple[tuple[str, ...], ...] | list[list[str]] | None = None,
) -> tuple[dict[str, float], str]:
    invalid = invalid_models or set()
    if not loss_history or min((len(values) for values in loss_history.values()), default=0) < minimum_observations:
        valid = [name for name in MODEL_NAMES if name not in invalid]
        equal = 1.0 / max(len(valid), 1)
        return {name: equal if name in valid else 0.0 for name in MODEL_NAMES}, "bootstrap_equal_until_history_available"

    baseline = float(random_baseline_brier() if baseline_loss is None else baseline_loss)
    skills = {
        name: 0.0 if name in invalid else _conservative_skill(
            loss_history.get(name, []),
            baseline,
            float(uncertainty_z),
        )
        for name in MODEL_NAMES
    }
    groups = model_groups if model_groups is not None else DEFAULT_MODEL_GROUPS
    grouped_names: set[str] = set()
    adjusted_skills = dict(skills)
    for group in groups:
        members = [name for name in group if name in adjusted_skills]
        if len(members) < 2:
            continue
        grouped_names.update(members)
        group_skill = max(skills[name] for name in members)
        positive = [name for name in members if skills[name] > 0]
        share = group_skill / len(positive) if positive else 0.0
        for name in members:
            adjusted_skills[name] = share if name in positive else 0.0
    skills = adjusted_skills
    total = sum(skills.values())
    if total <= 0:
        return {name: 0.0 for name in MODEL_NAMES}, "all_models_below_random_use_uniform_lottery"
    reason = "conservative_positive_brier_skill_with_group_deduplication" if grouped_names else "conservative_positive_brier_skill"
    return {name: value / total for name, value in skills.items()}, reason


def multiscale_shrinkage_weights(
    loss_history: dict[str, list[float]] | None,
    *,
    minimum_observations: int = 20,
    baseline_loss: float | None = None,
    uncertainty_z: float = 1.645,
    skill_windows: tuple[int, ...] | list[int] = (30, 100, 200),
    weak_signal_min_windows: int = 2,
    weak_signal_max_strength: float = 0.25,
    weak_signal_min_strength: float = 0.0,
    relative_tilt_max_strength: float = 0.0,
    relative_tilt_min_strength: float = 0.0,
    relative_tilt_consistency_boost: float = 0.0,
    relative_tilt_recovery_max_strength: float = 0.0,
    relative_tilt_recovery_consistency_threshold: float = 1.0,
    weak_signal_recent_window_power: float = 0.0,
    relative_tilt_recent_window_power: float = 0.0,
    max_uniform_share_if_signal: float = 1.0,
    invalid_models: set[str] | None = None,
    model_groups: tuple[tuple[str, ...], ...] | list[list[str]] | None = None,
) -> tuple[dict[str, float], str, dict[str, Any]]:
    """Estimate color-specific weights across several completed OOS windows.

    Strong evidence keeps the conservative Brier-skill rule.  Repeated but
    uncertain positive skill is retained only as a bounded mixture with the
    uniform distribution.  This avoids an unstable all-or-nothing switch while
    preserving an explicit uniform anchor and never turns negative skill into
    model weight.
    """
    invalid = invalid_models or set()
    baseline = float(random_baseline_brier() if baseline_loss is None else baseline_loss)
    history = loss_history or {}
    available_count = min(
        (len(history.get(name, [])) for name in MODEL_NAMES if len(history.get(name, [])) > 0),
        default=0,
    )
    if available_count < int(minimum_observations):
        bootstrap, reason = adaptive_weights(
            history,
            minimum_observations=int(minimum_observations),
            invalid_models=invalid,
            baseline_loss=baseline,
            uncertainty_z=float(uncertainty_z),
            model_groups=model_groups,
        )
        return bootstrap, reason, {
            "mode": "bootstrap",
            "signal_strength": float(sum(bootstrap.values())),
            "available_windows": [],
            "models": {},
        }

    windows = sorted({
        int(window)
        for window in skill_windows
        if int(minimum_observations) <= int(window) <= available_count
    })
    if not windows:
        windows = [available_count]

    evidence: dict[str, Any] = {}
    strong_scores: dict[str, float] = {}
    weak_scores: dict[str, float] = {}
    weak_confidences: dict[str, float] = {}
    window_losses_by_model: dict[int, dict[str, float]] = {int(window): {} for window in windows}
    window_importance = {
        int(window): _window_importance(
            int(window),
            minimum_window=max(1, min(windows, default=available_count)),
            power=float(weak_signal_recent_window_power),
        )
        for window in windows
    }
    relative_window_importance = {
        int(window): _window_importance(
            int(window),
            minimum_window=max(1, min(windows, default=available_count)),
            power=float(relative_tilt_recent_window_power),
        )
        for window in windows
    }
    required_positive = max(1, int(weak_signal_min_windows))
    for name in MODEL_NAMES:
        if name in invalid or len(history.get(name, [])) < int(minimum_observations):
            evidence[name] = {"invalid": True, "windows": []}
            strong_scores[name] = 0.0
            weak_scores[name] = 0.0
            weak_confidences[name] = 0.0
            continue
        window_evidence: list[dict[str, float | int]] = []
        for window in windows:
            values = np.asarray(history.get(name, [])[-window:], dtype=float)
            mean_loss = float(values.mean())
            window_losses_by_model[int(window)][name] = mean_loss
            raw_skill = float(baseline - mean_loss)
            standard_error = (
                float(values.std(ddof=1)) / float(np.sqrt(values.size))
                if values.size > 1 else 0.0
            )
            lower_skill = float(raw_skill - float(uncertainty_z) * standard_error)
            if standard_error > 0:
                probability_positive = 0.5 * (
                    1.0 + erf(raw_skill / standard_error / sqrt(2.0))
                )
            else:
                probability_positive = float(raw_skill > 0)
            confidence = max(0.0, min(1.0, 2.0 * (probability_positive - 0.5)))
            window_evidence.append({
                "window": int(window),
                "mean_loss": mean_loss,
                "raw_skill": raw_skill,
                "standard_error": standard_error,
                "lower_skill": lower_skill,
                "positive_confidence": confidence,
            })
        strong = [max(0.0, float(item["lower_skill"])) for item in window_evidence]
        positive = [item for item in window_evidence if float(item["raw_skill"]) > 0]
        stable_weak = len(windows) >= required_positive and len(positive) >= required_positive
        strong_scores[name] = float(np.mean(strong))
        weak_scores[name] = (
            float(np.average(
                [
                    float(item["raw_skill"]) * float(item["positive_confidence"])
                    for item in positive
                ],
                weights=[
                    float(window_importance.get(int(item["window"]), 1.0))
                    for item in positive
                ],
            ))
            if stable_weak else 0.0
        )
        weak_confidences[name] = (
            float(np.average(
                [float(item["positive_confidence"]) for item in positive],
                weights=[
                    float(window_importance.get(int(item["window"]), 1.0))
                    for item in positive
                ],
            ))
            if stable_weak else 0.0
        )
        evidence[name] = {
            "invalid": False,
            "positive_windows": len(positive),
            "required_positive_windows": required_positive,
            "windows": window_evidence,
        }

    groups = model_groups if model_groups is not None else DEFAULT_MODEL_GROUPS
    strong_scores = _deduplicate_group_skills(strong_scores, groups)
    weak_scores = _deduplicate_group_skills(weak_scores, groups)
    strong_total = float(sum(strong_scores.values()))
    if strong_total > 0:
        weights = {name: strong_scores[name] / strong_total for name in MODEL_NAMES}
        return weights, "multiscale_conservative_positive_brier_skill", {
            "mode": "strong_model_signal",
            "signal_strength": 1.0,
            "available_windows": windows,
            "models": evidence,
        }

    relative_scores = _relative_loss_tilt_scores(
        window_losses_by_model,
        window_weights=relative_window_importance,
    )
    relative_scores = _deduplicate_group_skills(relative_scores, groups)
    relative_total = float(sum(relative_scores.values()))
    relative_signal_strength = 0.0
    relative_consistency = 0.0
    relative_support_mass = 0.0
    relative_recovery_bonus = 0.0
    concentration = 0.0
    if relative_total > 0 and float(relative_tilt_max_strength) > 0:
        leader_share = max(
            (
                relative_scores[name] / relative_total
                for name in MODEL_NAMES
                if relative_scores[name] > 0
            ),
            default=0.0,
        )
        valid_count = max(1, sum(1 for name in MODEL_NAMES if name not in invalid))
        uniform_share = 1.0 / valid_count
        concentration = 0.0 if leader_share <= uniform_share else (
            (leader_share - uniform_share) / max(1e-12, 1.0 - uniform_share)
        )
        positive_window_ratios = _relative_window_support_ratios(
            window_losses_by_model,
            window_weights=relative_window_importance,
        )
        relative_consistency = float(np.average(
            [positive_window_ratios[name] for name in MODEL_NAMES if relative_scores[name] > 0],
            weights=[relative_scores[name] for name in MODEL_NAMES if relative_scores[name] > 0],
        ))
        relative_support_mass = min(1.0, max(0.0, relative_total * max(1, valid_count - 1)))
        base_strength = max(
            float(relative_tilt_min_strength),
            max(0.0, relative_support_mass * float(relative_tilt_max_strength)),
            max(0.0, concentration * float(relative_tilt_max_strength)),
        )
        relative_signal_strength = min(
            float(relative_tilt_max_strength),
            base_strength + relative_consistency * max(0.0, float(relative_tilt_consistency_boost)),
        )
        recovery_threshold = min(
            1.0,
            max(0.0, float(relative_tilt_recovery_consistency_threshold)),
        )
        recovery_max_strength = max(0.0, float(relative_tilt_recovery_max_strength))
        if recovery_max_strength > 0.0 and relative_consistency > recovery_threshold:
            consistency_excess = (relative_consistency - recovery_threshold) / max(
                1e-12,
                1.0 - recovery_threshold,
            )
            recovery_score = max(0.0, consistency_excess) * max(0.0, concentration) * min(
                1.0,
                max(0.0, relative_support_mass),
            )
            relative_recovery_bonus = min(
                recovery_max_strength,
                recovery_score * recovery_max_strength,
            )
            relative_signal_strength = min(1.0, relative_signal_strength + relative_recovery_bonus)

    weak_total = float(sum(weak_scores.values()))
    if weak_total > 0 and float(weak_signal_max_strength) > 0:
        confidence = float(np.average(
            [weak_confidences[name] for name in MODEL_NAMES if weak_scores[name] > 0],
            weights=[weak_scores[name] for name in MODEL_NAMES if weak_scores[name] > 0],
        ))
        weak_signal_strength = min(
            float(weak_signal_max_strength),
            max(
                float(weak_signal_min_strength),
                max(0.0, confidence * float(weak_signal_max_strength)),
            ),
        )
        weak_distribution = {
            name: weak_scores[name] / weak_total
            for name in MODEL_NAMES
        }
        if relative_signal_strength > 0 and relative_total > 0:
            relative_distribution = {
                name: relative_scores[name] / relative_total
                for name in MODEL_NAMES
            }
            support_overlap = float(sum(
                min(weak_distribution[name], relative_distribution[name])
                for name in MODEL_NAMES
            ))
            relative_uplift = min(
                1.0 - weak_signal_strength,
                max(0.0, relative_signal_strength * support_overlap),
            )
            weights = {
                name: weak_signal_strength * weak_distribution[name]
                + relative_uplift * relative_distribution[name]
                for name in MODEL_NAMES
            }
            weights[UNIFORM_ANCHOR] = 1.0 - weak_signal_strength - relative_uplift
            weights = _cap_uniform_anchor(
                weights,
                max_uniform_share=_adaptive_uniform_cap(
                    float(max_uniform_share_if_signal),
                    signal_strength=weak_signal_strength + relative_uplift,
                    concentration=max(concentration, relative_support_mass),
                    consistency=relative_consistency,
                    minimum_cap=max(0.12, float(max_uniform_share_if_signal) - 0.30),
                ),
            )
            return weights, "multiscale_weak_signal_with_relative_tilt_shrunk_to_uniform", {
                "mode": "weak_plus_relative_shrunk_signal",
                "signal_strength": weak_signal_strength + relative_uplift,
                "weak_signal_strength": weak_signal_strength,
                "relative_signal_strength": relative_signal_strength,
                "relative_uplift": relative_uplift,
                "relative_overlap": support_overlap,
                "available_windows": windows,
                "models": evidence,
                "relative_scores": relative_scores,
                "relative_consistency": relative_consistency,
                "relative_support_mass": relative_support_mass,
                "relative_recovery_bonus": relative_recovery_bonus,
                "relative_window_importance": relative_window_importance,
            }
        weights = {
            name: weak_signal_strength * weak_distribution[name]
            for name in MODEL_NAMES
        }
        weights[UNIFORM_ANCHOR] = 1.0 - weak_signal_strength
        weights = _cap_uniform_anchor(
            weights,
            max_uniform_share=_adaptive_uniform_cap(
                float(max_uniform_share_if_signal),
                signal_strength=weak_signal_strength,
                concentration=max(weak_distribution.values(), default=0.0),
                consistency=confidence,
                minimum_cap=max(0.14, float(max_uniform_share_if_signal) - 0.24),
            ),
        )
        return weights, "multiscale_weak_signal_shrunk_to_uniform", {
            "mode": "weak_shrunk_signal",
            "signal_strength": weak_signal_strength,
            "available_windows": windows,
            "models": evidence,
            "window_importance": window_importance,
        }

    if relative_signal_strength > 0:
            weights = {
                name: relative_signal_strength * relative_scores[name] / relative_total
                for name in MODEL_NAMES
            }
            weights[UNIFORM_ANCHOR] = 1.0 - relative_signal_strength
            weights = _cap_uniform_anchor(
                weights,
                max_uniform_share=_adaptive_uniform_cap(
                    float(max_uniform_share_if_signal),
                    signal_strength=relative_signal_strength,
                    concentration=max(concentration, relative_support_mass),
                    consistency=relative_consistency,
                    minimum_cap=max(0.10, float(max_uniform_share_if_signal) - 0.32),
                ),
            )
            return weights, "multiscale_relative_loss_tilt_shrunk_to_uniform", {
                "mode": "relative_loss_tilt_signal",
                "signal_strength": relative_signal_strength,
                "available_windows": windows,
                "models": evidence,
                "relative_scores": relative_scores,
                "relative_consistency": relative_consistency,
                "relative_support_mass": relative_support_mass,
                "relative_recovery_bonus": relative_recovery_bonus,
                "relative_window_importance": relative_window_importance,
            }

    weights = {name: 0.0 for name in MODEL_NAMES}
    weights[UNIFORM_ANCHOR] = 1.0
    return weights, "multiscale_no_positive_skill_use_uniform", {
        "mode": "uniform_no_signal",
        "signal_strength": 0.0,
        "available_windows": windows,
        "models": evidence,
    }


def cap_model_weight_shares(
    weights: dict[str, float],
    caps: dict[str, float] | None,
    *,
    max_uniform_share: float | None = None,
    invalid_models: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Limit any single model from dominating a color-specific fusion.

    The cap is intentionally applied after evidence weighting.  This keeps a
    weak but noisy leader from turning the whole red/blue distribution into
    one model's short-window story.  Excess weight is first offered to other
    uncapped model signals; if no signal has room left, it is moved to the
    explicit uniform anchor.
    """
    invalid = set(invalid_models or set())
    configured_caps = {
        name: min(1.0, max(0.0, float(limit)))
        for name, limit in (caps or {}).items()
        if name in MODEL_NAMES and name not in invalid
    }
    capped = {name: float(value) for name, value in weights.items()}
    for name in invalid:
        if name in MODEL_NAMES:
            capped[name] = 0.0
    capped.setdefault(UNIFORM_ANCHOR, 0.0)
    uniform_cap_diagnostics: dict[str, Any] = {
        "configured": max_uniform_share,
        "applied": False,
    }
    if not configured_caps:
        normalized = _normalize_weights(capped)
        normalized, uniform_cap_diagnostics = _redistribute_uniform_excess(
            normalized,
            configured_caps,
            max_uniform_share=max_uniform_share,
            invalid_models=invalid,
        )
        return normalized, {
            "applied": bool(uniform_cap_diagnostics.get("applied")),
            "caps": {},
            "uniform_cap": uniform_cap_diagnostics,
        }

    capped_models: dict[str, dict[str, float]] = {}
    for _ in range(len(MODEL_NAMES) + 1):
        excess = 0.0
        for name, limit in configured_caps.items():
            value = float(capped.get(name, 0.0))
            if value > limit + 1e-12:
                capped_models[name] = {"before": value, "cap": limit}
                capped[name] = limit
                excess += value - limit
        if excess <= 1e-12:
            break

        receivers = [
            name
            for name in MODEL_NAMES
            if float(capped.get(name, 0.0)) > 0.0
            and float(capped.get(name, 0.0)) < configured_caps.get(name, 1.0) - 1e-12
        ]
        capacity = {
            name: max(0.0, configured_caps.get(name, 1.0) - float(capped.get(name, 0.0)))
            for name in receivers
        }
        total_capacity = float(sum(capacity.values()))
        if receivers and total_capacity > 1e-12:
            signal_mass = float(sum(float(capped.get(name, 0.0)) for name in receivers))
            for name in receivers:
                if signal_mass > 1e-12:
                    proposed = excess * float(capped.get(name, 0.0)) / signal_mass
                else:
                    proposed = excess * capacity[name] / total_capacity
                capped[name] += min(capacity[name], proposed)
            remainder = 1.0 - sum(float(capped.get(name, 0.0)) for name in MODEL_NAMES)
            capped[UNIFORM_ANCHOR] = max(0.0, float(capped.get(UNIFORM_ANCHOR, 0.0)) + max(0.0, remainder - float(capped.get(UNIFORM_ANCHOR, 0.0))))
        else:
            capped[UNIFORM_ANCHOR] = float(capped.get(UNIFORM_ANCHOR, 0.0)) + excess
            break

    normalized = _normalize_weights(capped)
    normalized, uniform_cap_diagnostics = _redistribute_uniform_excess(
        normalized,
        configured_caps,
        max_uniform_share=max_uniform_share,
        invalid_models=invalid,
    )
    for name, record in capped_models.items():
        record["after"] = float(normalized.get(name, 0.0))
    return normalized, {
        "applied": bool(capped_models) or bool(uniform_cap_diagnostics.get("applied")),
        "caps": configured_caps,
        "capped_models": capped_models,
        "uniform_cap": uniform_cap_diagnostics,
    }


def _redistribute_uniform_excess(
    weights: dict[str, float],
    caps: dict[str, float] | None,
    *,
    max_uniform_share: float | None,
    invalid_models: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Keep the explicit uniform anchor from re-absorbing capped model signal.

    Model caps are useful, but their excess should not silently become a large
    random anchor.  This helper first moves uniform excess back into already
    active non-baseline models.  If those models are saturated, it can open a
    small bridge to cap-eligible backup models so a single capped signal does
    not collapse into a mostly-random distribution.
    """
    if max_uniform_share is None:
        return weights, {"configured": None, "applied": False}

    maximum = min(1.0, max(0.0, float(max_uniform_share)))
    adjusted = dict(weights)
    before_uniform = float(adjusted.get(UNIFORM_ANCHOR, 0.0))
    if before_uniform <= maximum + 1e-12:
        return adjusted, {
            "configured": maximum,
            "applied": False,
            "before": before_uniform,
            "after": before_uniform,
        }

    invalid = set(invalid_models or set())
    configured_caps = {
        name: min(1.0, max(0.0, float(limit)))
        for name, limit in (caps or {}).items()
        if name in MODEL_NAMES and name not in invalid
    }
    excess = before_uniform - maximum
    original_signal_mass = float(sum(float(adjusted.get(name, 0.0)) for name in MODEL_NAMES))
    if original_signal_mass <= 1e-12:
        return adjusted, {
            "configured": maximum,
            "applied": False,
            "before": before_uniform,
            "after": before_uniform,
            "reason": "no_model_signal",
        }

    def _eligible_capacity(name: str) -> float:
        return max(0.0, configured_caps.get(name, 1.0) - float(adjusted.get(name, 0.0)))

    def _release_to(receivers: list[str], remaining_excess: float) -> float:
        capacities = {name: _eligible_capacity(name) for name in receivers}
        total_capacity = float(sum(capacities.values()))
        released = min(remaining_excess, total_capacity)
        if released <= 1e-12:
            return 0.0
        signal_mass = float(sum(float(adjusted.get(name, 0.0)) for name in receivers))
        remaining = released
        for index, name in enumerate(receivers):
            if index == len(receivers) - 1:
                add = min(capacities[name], remaining)
            elif signal_mass > 1e-12:
                add = min(capacities[name], released * float(adjusted.get(name, 0.0)) / signal_mass)
            else:
                add = min(capacities[name], released * capacities[name] / total_capacity)
            adjusted[name] = float(adjusted.get(name, 0.0)) + add
            remaining -= add
        return released - max(0.0, remaining)

    active_receivers = [
        name
        for name in MODEL_NAMES
        if name not in invalid
        and float(adjusted.get(name, 0.0)) > 0.0
        and _eligible_capacity(name) > 1e-12
    ]
    released = _release_to(active_receivers, excess)
    remaining_excess = max(0.0, excess - released)
    backup_receivers: list[str] = []
    backup_released = 0.0
    if remaining_excess > 1e-12:
        backup_receivers = sorted(
            [
                name
                for name in MODEL_NAMES
                if name not in invalid
                and name not in active_receivers
                and _eligible_capacity(name) > 1e-12
            ],
            key=lambda name: (
                configured_caps.get(name, 1.0),
                _MODEL_PRIOR_RANK.get(name, 0.0),
                name,
            ),
            reverse=True,
        )
        backup_released = _release_to(backup_receivers, remaining_excess)
        released += backup_released

    if released <= 1e-12:
        return adjusted, {
            "configured": maximum,
            "applied": False,
            "before": before_uniform,
            "after": before_uniform,
            "reason": "zero_model_capacity",
        }

    adjusted[UNIFORM_ANCHOR] = before_uniform - released
    normalized = _normalize_weights(adjusted)
    return normalized, {
        "configured": maximum,
        "applied": True,
        "before": before_uniform,
        "after": float(normalized.get(UNIFORM_ANCHOR, 0.0)),
        "released": before_uniform - float(normalized.get(UNIFORM_ANCHOR, 0.0)),
        "active_receivers": active_receivers,
        "backup_receivers": backup_receivers,
        "backup_released": backup_released,
    }


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    preserved = {
        key: value
        for key, value in weights.items()
        if key not in MODEL_NAMES and key != UNIFORM_ANCHOR
    }
    numeric = {
        name: max(0.0, float(weights.get(name, 0.0)))
        for name in [*MODEL_NAMES, UNIFORM_ANCHOR]
    }
    total = float(sum(numeric.values()))
    if total <= 0.0:
        numeric = {name: 0.0 for name in MODEL_NAMES}
        numeric[UNIFORM_ANCHOR] = 1.0
    else:
        numeric = {name: value / total for name, value in numeric.items()}
    return {**numeric, **preserved}


def _cap_uniform_anchor(
    weights: dict[str, float],
    *,
    max_uniform_share: float,
) -> dict[str, float]:
    capped = dict(weights)
    maximum = min(1.0, max(0.0, float(max_uniform_share)))
    current_uniform = float(capped.get(UNIFORM_ANCHOR, 0.0))
    if current_uniform <= maximum:
        return capped
    signal_names = [name for name in MODEL_NAMES if float(capped.get(name, 0.0)) > 0.0]
    if not signal_names:
        return capped
    released = current_uniform - maximum
    signal_mass = float(sum(float(capped.get(name, 0.0)) for name in signal_names))
    if signal_mass <= 0.0:
        return capped
    capped[UNIFORM_ANCHOR] = maximum
    for name in signal_names:
        share = float(capped.get(name, 0.0)) / signal_mass
        capped[name] = float(capped.get(name, 0.0)) + released * share
    return capped


def _adaptive_uniform_cap(
    base_cap: float,
    *,
    signal_strength: float,
    concentration: float = 0.0,
    consistency: float = 0.0,
    minimum_cap: float = 0.0,
) -> float:
    """Lower the effective uniform cap when non-baseline signal is coherent."""
    base = min(1.0, max(0.0, float(base_cap)))
    floor = min(base, max(0.0, float(minimum_cap)))
    strength = min(1.0, max(0.0, float(signal_strength)))
    concentration = min(1.0, max(0.0, float(concentration)))
    consistency = min(1.0, max(0.0, float(consistency)))
    synergy = max(0.0, concentration - 0.45) * max(0.0, consistency - 0.50)
    coherence = min(
        1.0,
        0.45 * strength
        + 0.35 * concentration
        + 0.20 * consistency
        + 0.35 * synergy,
    )
    return max(floor, base - (base - floor) * coherence)


def _deduplicate_group_skills(
    skills: dict[str, float],
    groups: tuple[tuple[str, ...], ...] | list[list[str]],
) -> dict[str, float]:
    adjusted = dict(skills)
    for group in groups:
        members = [name for name in group if name in adjusted]
        positive = [name for name in members if adjusted[name] > 0]
        if len(members) < 2 or not positive:
            continue
        shared = max(adjusted[name] for name in positive) / len(positive)
        for name in members:
            adjusted[name] = shared if name in positive else 0.0
    return adjusted


def _relative_loss_tilt_scores(
    window_losses_by_model: dict[int, dict[str, float]],
    *,
    window_weights: dict[int, float] | None = None,
) -> dict[str, float]:
    scores = {name: 0.0 for name in MODEL_NAMES}
    counted_weight = 0.0
    for window, losses in window_losses_by_model.items():
        valid = {
            name: float(loss)
            for name, loss in losses.items()
            if np.isfinite(float(loss))
        }
        if len(valid) < 2:
            continue
        ordered = np.array([valid[name] for name in valid], dtype=float)
        best = float(ordered.min())
        scale = float(max(np.std(ordered), 1e-6))
        centered = {
            name: np.exp(-(float(loss) - best) / scale)
            for name, loss in valid.items()
        }
        total = float(sum(centered.values()))
        if total <= 0:
            continue
        weight = float((window_weights or {}).get(int(window), 1.0))
        if weight <= 0:
            continue
        counted_weight += weight
        uniform_share = 1.0 / len(valid)
        for name in valid:
            share = centered[name] / total
            scores[name] += weight * max(0.0, float(share - uniform_share))
    if counted_weight <= 0:
        return scores
    return {
        name: float(value / counted_weight)
        for name, value in scores.items()
    }


def _relative_window_support_ratios(
    window_losses_by_model: dict[int, dict[str, float]],
    *,
    window_weights: dict[int, float] | None = None,
) -> dict[str, float]:
    support_counts = {name: 0.0 for name in MODEL_NAMES}
    counted_weight = 0.0
    for window, losses in window_losses_by_model.items():
        valid = {
            name: float(loss)
            for name, loss in losses.items()
            if np.isfinite(float(loss))
        }
        if len(valid) < 2:
            continue
        ordered = np.array([valid[name] for name in valid], dtype=float)
        best = float(ordered.min())
        scale = float(max(np.std(ordered), 1e-6))
        centered = {
            name: np.exp(-(float(loss) - best) / scale)
            for name, loss in valid.items()
        }
        total = float(sum(centered.values()))
        if total <= 0:
            continue
        weight = float((window_weights or {}).get(int(window), 1.0))
        if weight <= 0:
            continue
        counted_weight += weight
        uniform_share = 1.0 / len(valid)
        for name in valid:
            share = centered[name] / total
            if share > uniform_share:
                support_counts[name] += weight
    if counted_weight <= 0:
        return support_counts
    return {
        name: float(value / counted_weight)
        for name, value in support_counts.items()
    }


def _window_importance(window: int, *, minimum_window: int, power: float) -> float:
    if power <= 0:
        return 1.0
    ratio = max(float(window), 1.0) / max(float(minimum_window), 1.0)
    return float(1.0 / (ratio ** power))


def _conservative_skill(values: list[float], baseline: float, uncertainty_z: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    uncertainty = 0.0
    if array.size > 1 and uncertainty_z > 0:
        uncertainty = float(uncertainty_z) * float(array.std(ddof=1)) / float(np.sqrt(array.size))
    return max(0.0, float(baseline) - float(array.mean()) - uncertainty)


def _entropy_sharpen(values: np.ndarray, *, threshold: float, power: float, min_signal: float, signal_strength: float) -> np.ndarray:
    """Mildly amplify rankings when a weighted model mixture is almost flat.

    The fusion layer can have high non-uniform model weight while the actual
    probability vector is still nearly uniform.  In that situation Top-K output
    is unstable: tiny numeric differences decide the rank.  This optional,
    color-specific sharpening keeps the same order but increases separation a
    little, so the downstream candidate generator does not behave like a fully
    random sampler while still preserving normalization.
    """
    probability = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(probability.sum())
    if total <= 0:
        return probability
    probability = probability / total
    if float(power) <= 1.0 or float(signal_strength) < float(min_signal):
        return probability
    if _normalized_entropy(probability) < float(threshold):
        return probability
    sharpened = np.power(np.clip(probability, 1e-12, None), float(power))
    sharpened_total = float(sharpened.sum())
    return sharpened / sharpened_total if sharpened_total > 0 else probability


def fuse_predictions(
    predictions: dict[str, dict[str, np.ndarray]],
    weights: dict[str, float] | dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    fused: dict[str, np.ndarray] = {}
    for color, number_count in (("red", 33), ("blue", 16)):
        color_weights = weights.get(color, weights)
        if not isinstance(color_weights, dict):
            raise TypeError("融合权重必须是模型权重字典，或按 red/blue 分组的权重字典")
        uniform_weight = float(color_weights.get(UNIFORM_ANCHOR, 0.0))
        model_weight = float(sum(float(color_weights.get(name, 0.0)) for name in MODEL_NAMES))
        model_values = np.zeros(number_count, dtype=float)
        for name in MODEL_NAMES:
            model_values += float(color_weights.get(name, 0.0)) * np.asarray(predictions[name][color], dtype=float)
        if model_weight > 1e-12:
            normalized_model_values = model_values / model_weight
            anchor_distribution = _structured_anchor_distribution(
                normalized_model_values,
                top_k=int(color_weights.get(ANCHOR_TOP_K_KEY, 0)),
                head_share=float(color_weights.get(ANCHOR_HEAD_SHARE_KEY, 0.0)),
                min_signal=float(color_weights.get(ANCHOR_MIN_SIGNAL_KEY, 1.0)),
                signal_strength=model_weight,
                mode=str(color_weights.get(ANCHOR_MODE_KEY, "uniform")),
            )
            values = model_values + uniform_weight * anchor_distribution
        else:
            values = np.full(number_count, uniform_weight / number_count, dtype=float)
        values = _entropy_sharpen(
            values,
            threshold=float(color_weights.get(ENTROPY_SHARPEN_THRESHOLD_KEY, 1.01)),
            power=float(color_weights.get(ENTROPY_SHARPEN_POWER_KEY, 1.0)),
            min_signal=float(color_weights.get(ENTROPY_SHARPEN_MIN_SIGNAL_KEY, 1.0)),
            signal_strength=model_weight,
        )
        total = float(values.sum())
        fused[color] = values / total if total > 0 else np.full(number_count, 1.0 / number_count)
    return fused


def rank_prediction(
    prediction: dict[str, np.ndarray],
    *,
    red_count: int = 6,
    blue_count: int = 3,
    weights_by_color: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    red_values = np.asarray(prediction["red"], dtype=float)
    blue_values = np.asarray(prediction["blue"], dtype=float)
    red_order = np.argsort(-red_values, kind="stable")[: int(red_count)]
    blue_order = np.argsort(-blue_values, kind="stable")[: int(blue_count)]
    red_uniform = bool(np.allclose(red_values, red_values[0], rtol=1e-10, atol=1e-12))
    blue_uniform = bool(np.allclose(blue_values, blue_values[0], rtol=1e-10, atol=1e-12))
    red_entropy = _normalized_entropy(red_values)
    blue_entropy = _normalized_entropy(blue_values)
    signal_strength_by_color = None
    signal_tier = None
    prediction_mode = "uniform_no_signal_baseline" if red_uniform and blue_uniform else "model_weighted_signal"
    if weights_by_color is not None:
        signal_strength_by_color = {
            color: float(sum(float(weights_by_color.get(color, {}).get(name, 0.0)) for name in MODEL_NAMES))
            for color in ("red", "blue")
        }
        maximum_strength = max(signal_strength_by_color.values())
        if maximum_strength <= 1e-12:
            signal_tier = "no_signal"
            prediction_mode = "uniform_no_signal_baseline"
        elif any(value < 1.0 - 1e-12 for value in signal_strength_by_color.values()):
            signal_tier = "weak_shrunk_signal"
            prediction_mode = "shrunk_weak_model_signal"
        else:
            signal_tier = "full_model_signal"
            prediction_mode = "model_weighted_signal"
    result = {
        "top_red": [int(index + 1) for index in red_order],
        "top_red_scores": [float(red_values[index]) for index in red_order],
        "top_blue": [int(index + 1) for index in blue_order],
        "top_blue_scores": [float(blue_values[index]) for index in blue_order],
        "red_is_uniform": red_uniform,
        "blue_is_uniform": blue_uniform,
        "is_uniform": red_uniform and blue_uniform,
        "red_normalized_entropy": red_entropy,
        "blue_normalized_entropy": blue_entropy,
        "prediction_mode": prediction_mode,
    }
    if signal_strength_by_color is not None:
        result["signal_strength_by_color"] = signal_strength_by_color
        result["signal_tier"] = signal_tier
    return result


def _normalized_entropy(values: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(probabilities.sum())
    if total <= 0 or len(probabilities) <= 1:
        return 0.0
    probabilities = probabilities / total
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)) / np.log(len(probabilities)))


def _structured_anchor_distribution(
    model_values: np.ndarray,
    *,
    top_k: int,
    head_share: float,
    min_signal: float,
    signal_strength: float,
    mode: str,
) -> np.ndarray:
    normalized = np.asarray(model_values, dtype=float)
    total = float(normalized.sum())
    if total <= 0:
        return np.full(len(normalized), 1.0 / max(len(normalized), 1), dtype=float)
    normalized = normalized / total
    if (
        mode != "tiered_head_tail"
        or int(top_k) <= 0
        or len(normalized) <= int(top_k)
        or float(signal_strength) < float(min_signal)
    ):
        return np.full(len(normalized), 1.0 / len(normalized), dtype=float)

    clamped_top_k = max(1, min(int(top_k), len(normalized) - 1))
    clamped_head_share = min(1.0, max(0.0, float(head_share)))
    head_indices = np.argsort(-normalized, kind="stable")[:clamped_top_k]
    tail_indices = np.setdiff1d(np.arange(len(normalized)), head_indices, assume_unique=True)
    head_total = float(normalized[head_indices].sum())
    if head_total <= 0 or tail_indices.size == 0:
        return np.full(len(normalized), 1.0 / len(normalized), dtype=float)

    anchor = np.zeros(len(normalized), dtype=float)
    anchor[head_indices] = clamped_head_share * (normalized[head_indices] / head_total)
    tail_share = max(0.0, 1.0 - clamped_head_share)
    anchor[tail_indices] = tail_share / float(tail_indices.size)
    anchor_total = float(anchor.sum())
    if anchor_total <= 0:
        return np.full(len(normalized), 1.0 / len(normalized), dtype=float)
    return anchor / anchor_total
