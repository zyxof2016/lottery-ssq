from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineer.window_feat import build_prediction_features, build_training_samples
from src.probability_model.feature_selection import StableFeatureSelector, summarize_feature_importance


def _normalize(values: np.ndarray, size: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    return values / total if total > 0 else np.full(int(size), 1.0 / int(size), dtype=float)


class CatBoostProbabilityModel:
    """Optional ordered-boosting tabular expert.

    CatBoost is kept as a separately configurable model because it is heavier
    than the ordinary models.  When enabled, it gives the ensemble a different
    tree-building bias from XGBoost/LightGBM; when disabled the registry uses a
    uniform placeholder from ``fallback.py``.
    """

    name = "catboost"

    def __init__(self, feature_config: dict[str, Any], model_config: dict[str, Any]):
        self.feature_config = feature_config
        self.model_config = model_config
        self.models_: dict[str, Any] = {}
        self.feature_selectors_: dict[str, StableFeatureSelector] = {}

    def fit(self, history: pd.DataFrame) -> "CatBoostProbabilityModel":
        from catboost import CatBoostClassifier

        training_losses: dict[str, float] = {}
        feature_importance: dict[str, Any] = {}
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
            self.feature_selectors_[color] = selector
            model = CatBoostClassifier(
                iterations=int(self.model_config.get("iterations", 90)),
                depth=int(self.model_config.get("depth", 5)),
                learning_rate=float(self.model_config.get("learning_rate", 0.035)),
                l2_leaf_reg=float(self.model_config.get("l2_leaf_reg", 8.0)),
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=int(self.model_config.get("random_seed", 2026)),
                verbose=False,
                allow_writing_files=False,
                thread_count=int(self.model_config.get("thread_count", 1)),
                auto_class_weights=str(self.model_config.get("auto_class_weights", "Balanced")),
            )
            model.fit(selected_features, labels)
            probability = model.predict_proba(selected_features)[:, 1].reshape(-1, size)
            if color == "red":
                marginal = np.clip(probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12) * 6.0, 0.0, 1.0)
                training_losses[color] = float(np.mean((marginal - labels.reshape(-1, size)) ** 2))
            else:
                normalized = probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)
                training_losses[color] = float(np.mean((normalized - labels.reshape(-1, size)) ** 2))
            self.models_[color] = model
            feature_importance[color] = summarize_feature_importance(
                columns=selector.columns_,
                importances=model.get_feature_importance(),
            )
        self.model_modes_ = {"red": "catboost_binary_marginal", "blue": "catboost_binary_rank"}
        self.training_brier_by_color_ = training_losses
        self.training_brier_ = float(np.mean(list(training_losses.values())))
        self.feature_importance_ = feature_importance
        return self

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
            selected = selector.transform(features)
            result[color] = _normalize(self.models_[color].predict_proba(selected)[:, 1], size)
        return result
