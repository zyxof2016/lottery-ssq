from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineer.window_feat import (
    build_prediction_features,
    build_training_samples,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return values / values.sum()


class LogisticRegressionModel:
    name = "logistic_regression"

    def __init__(self, feature_config: dict[str, Any], model_config: dict[str, Any]):
        self.feature_config = feature_config
        self.model_config = model_config
        self.models_: dict[str, Pipeline] = {}

    def fit(self, history: pd.DataFrame) -> "LogisticRegressionModel":
        training_losses: dict[str, float] = {}
        red_features, red_labels = build_training_samples(
            history,
            color="red",
            windows=self.feature_config["windows"],
            decay_factor=self.feature_config["decay_factor"],
            min_history=self.feature_config["min_history"],
            lookback=self.feature_config.get("training_lookback"),
            include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
            include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
            cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
            cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
        )
        red_model = self._pipeline(float(self.model_config.get("c", 0.5)))
        red_model.fit(red_features, red_labels)
        red_probability = red_model.predict_proba(red_features)[:, 1].reshape(-1, 33)
        red_probability = red_probability / np.maximum(red_probability.sum(axis=1, keepdims=True), 1e-12)
        red_probability = np.clip(red_probability * 6.0, 0.0, 1.0)
        training_losses["red"] = float(
            np.mean((red_probability - red_labels.reshape(-1, 33)) ** 2)
        )
        self.models_["red"] = red_model

        blue_features, blue_labels = build_training_samples(
            history,
            color="blue",
            windows=self.feature_config["windows"],
            decay_factor=self.feature_config["decay_factor"],
            min_history=self.feature_config["min_history"],
            lookback=self.feature_config.get("training_lookback"),
            include_blue_sequence=bool(
                self.model_config.get("use_blue_sequence_features", True)
            ),
            include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
            include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
            cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
            cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
        )
        blue_scaler, blue_coefficients, blue_probability = _fit_conditional_multinomial(
            blue_features,
            blue_labels,
            alternatives=16,
            c_value=float(self.model_config.get("blue_c", 0.05)),
            max_iter=int(self.model_config.get("max_iter", 300)),
        )
        blue_targets = blue_labels.reshape(-1, 16)
        training_losses["blue"] = float(np.mean((blue_probability - blue_targets) ** 2))
        self.models_["blue"] = {
            "scaler": blue_scaler,
            "coefficients": blue_coefficients,
        }
        self.model_modes_ = {
            "red": "binary_marginal",
            "blue": "conditional_multinomial_16_class",
        }
        self.blue_sequence_features_ = bool(
            self.model_config.get("use_blue_sequence_features", True)
        )
        self.training_brier_by_color_ = training_losses
        self.training_brier_ = float(np.mean(list(training_losses.values())))
        return self

    def _pipeline(self, c_value: float) -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=float(c_value),
                max_iter=int(self.model_config.get("max_iter", 300)),
                random_state=int(self.model_config.get("random_seed", 2026)),
                class_weight=self.model_config.get("red_class_weight", "balanced"),
            )),
        ])

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        red_model = self.models_["red"]
        red_features = build_prediction_features(
            history,
            color="red",
            target_idx=target_idx,
            windows=self.feature_config["windows"],
            decay_factor=self.feature_config["decay_factor"],
            include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
            include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
            lookback=int(self.feature_config.get("prediction_lookback", 900)),
            cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
            cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
        )
        result["red"] = _normalize(red_model.predict_proba(red_features)[:, 1])

        blue_model = self.models_["blue"]
        blue_features = build_prediction_features(
            history,
            color="blue",
            target_idx=target_idx,
            windows=self.feature_config["windows"],
            decay_factor=self.feature_config["decay_factor"],
            include_blue_sequence=bool(
                self.model_config.get("use_blue_sequence_features", True)
            ),
            include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
            include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
            lookback=int(self.feature_config.get("prediction_lookback", 900)),
            cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
            cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
        )
        scaled = blue_model["scaler"].transform(blue_features)
        scores = scaled @ blue_model["coefficients"]
        scores -= float(np.max(scores))
        result["blue"] = _normalize(np.exp(np.clip(scores, -50.0, 50.0)))
        return result


def _fit_conditional_multinomial(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    alternatives: int,
    c_value: float,
    max_iter: int,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit one shared utility function under an exact one-of-K softmax.

    Unlike a flattened high-dimensional draw classifier, the model has only
    one coefficient per audited number feature.  This preserves the mutually
    exclusive blue outcome while sharply limiting overfit.
    """
    alternative_count = int(alternatives)
    feature_count = int(features.shape[1])
    if len(features) % alternative_count:
        raise ValueError("条件多分类样本不能按候选数完整分组")
    periods = len(features) // alternative_count
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features).reshape(periods, alternative_count, feature_count)
    targets = np.asarray(labels, dtype=float).reshape(periods, alternative_count)
    regularization = 1.0 / max(float(c_value) * periods, 1e-12)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        scores = np.einsum("tkf,f->tk", scaled, coefficients)
        scores -= scores.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(scores, -50.0, 50.0))
        probability = exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-12)
        loss = -float(np.mean(np.sum(targets * np.log(np.clip(probability, 1e-12, 1.0)), axis=1)))
        loss += 0.5 * regularization * float(coefficients @ coefficients)
        gradient = np.einsum("tk,tkf->f", probability - targets, scaled) / periods
        gradient += regularization * coefficients
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(feature_count, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(max_iter), "ftol": 1e-10},
    )
    coefficients = np.asarray(fitted.x, dtype=float)
    scores = np.einsum("tkf,f->tk", scaled, coefficients)
    scores -= scores.max(axis=1, keepdims=True)
    probability = np.exp(np.clip(scores, -50.0, 50.0))
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
    return scaler, coefficients, probability
