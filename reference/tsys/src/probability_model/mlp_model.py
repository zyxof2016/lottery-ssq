from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineer.window_feat import build_prediction_features, build_training_samples
from src.probability_model.feature_selection import StableFeatureSelector


def _normalize(values: np.ndarray, size: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    return values / total if total > 0 else np.full(int(size), 1.0 / int(size), dtype=float)


class MLPProbabilityModel:
    """Small neural tabular expert used for model-diversity, not deep learning theatre.

    The dataset is small, so this is intentionally shallow and heavily bounded.
    It adds a different inductive bias from trees and linear models while
    keeping the standard runtime profile close to the 30-minute target.
    """

    name = "mlp"

    def __init__(self, feature_config: dict[str, Any], model_config: dict[str, Any]):
        self.feature_config = feature_config
        self.model_config = model_config
        self.models_: dict[str, Pipeline] = {}
        self.feature_selectors_: dict[str, StableFeatureSelector] = {}

    def fit(self, history: pd.DataFrame) -> "MLPProbabilityModel":
        training_losses: dict[str, float] = {}
        for color, size in (("red", 33), ("blue", 16)):
            features, labels = build_training_samples(
                history,
                color=color,
                windows=self.feature_config["windows"],
                decay_factor=self.feature_config["decay_factor"],
                min_history=self.feature_config["min_history"],
                lookback=self.feature_config.get("training_lookback"),
                include_blue_sequence=bool(
                    color == "blue" and self.model_config.get("use_blue_sequence_features", True)
                ),
                include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
                include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
                cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
                cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
            )
            selector = StableFeatureSelector.from_config(
                self.feature_config,
                color=color,
                random_seed=int(self.model_config.get("random_seed", 2026)),
            )
            selected_features = selector.fit_transform(features, labels)
            if len(selected_features) != len(labels):
                raise ValueError(
                    f"MLP训练样本数不一致: color={color}, "
                    f"features={len(selected_features)}, labels={len(labels)}"
                )
            if len(selected_features) % int(size) != 0:
                raise ValueError(
                    f"MLP训练样本不能按号码数分组: color={color}, "
                    f"features={len(selected_features)}, group_size={size}"
                )
            self.feature_selectors_[color] = selector
            model = self._pipeline()
            model.fit(selected_features, labels)
            probability = model.predict_proba(selected_features)[:, 1].reshape(-1, size)
            if color == "red":
                marginal = np.clip(probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12) * 6.0, 0.0, 1.0)
                target = labels.reshape(-1, size)
                training_losses[color] = float(np.mean((marginal - target) ** 2))
            else:
                normalized = probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
                training_losses[color] = float(np.mean((normalized - labels.reshape(-1, size)) ** 2))
            self.models_[color] = model
        self.model_modes_ = {"red": "shallow_mlp_binary_marginal", "blue": "shallow_mlp_binary_rank"}
        self.training_brier_by_color_ = training_losses
        self.training_brier_ = float(np.mean(list(training_losses.values())))
        self.feature_importance_ = {}
        return self

    def _pipeline(self) -> Pipeline:
        hidden = self.model_config.get("hidden_layer_sizes", [64, 32])
        hidden_tuple = tuple(int(value) for value in hidden)
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=hidden_tuple,
                activation=str(self.model_config.get("activation", "relu")),
                alpha=float(self.model_config.get("alpha", 0.003)),
                learning_rate_init=float(self.model_config.get("learning_rate_init", 0.001)),
                max_iter=int(self.model_config.get("max_iter", 80)),
                early_stopping=bool(self.model_config.get("early_stopping", True)),
                n_iter_no_change=int(self.model_config.get("n_iter_no_change", 8)),
                random_state=int(self.model_config.get("random_seed", 2026)),
            )),
        ])

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for color, size in (("red", 33), ("blue", 16)):
            features = build_prediction_features(
                history,
                color=color,
                target_idx=target_idx,
                windows=self.feature_config["windows"],
                decay_factor=self.feature_config["decay_factor"],
                include_blue_sequence=bool(
                    color == "blue" and self.model_config.get("use_blue_sequence_features", True)
                ),
                include_cross_feature=bool(self.feature_config.get("cross_red_blue_feature", False)),
                include_calendar_feature=bool(self.feature_config.get("calendar_features_enabled", False)),
                lookback=int(self.feature_config.get("prediction_lookback", 900)),
                cache_enabled=bool(self.feature_config.get("feature_cache_enabled", True)),
                cache_max_entries=int(self.feature_config.get("feature_cache_max_entries", 24)),
            )
            selector = self.feature_selectors_.get(color, StableFeatureSelector(enabled=False))
            features = selector.transform(features)
            result[color] = _normalize(self.models_[color].predict_proba(features)[:, 1], size)
        return result
