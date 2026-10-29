from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.feature_engineer.window_feat import (
    build_prediction_features,
    build_training_samples,
)
from src.probability_model.feature_selection import StableFeatureSelector, summarize_feature_importance

class XGBoostModel:
    name = "xgboost"

    def __init__(self, feature_config: dict[str, Any], model_config: dict[str, Any]):
        self.feature_config = feature_config
        self.model_config = model_config
        self.models_: dict[str, XGBRanker] = {}
        self.feature_selectors_: dict[str, StableFeatureSelector] = {}

    def fit(self, history: pd.DataFrame, *, incremental: bool = False, incremental_tree_ratio: float = 0.2) -> "XGBoostModel":
        training_losses: dict[str, float] = {}
        # 增量训练：如果已有模型且启用增量训练，使用续训模式
        use_incremental = bool(incremental) and bool(self.models_)
        if use_incremental:
            logging.info("XGBoost增量训练 | 事件=XGB_INCREMENTAL_FIT | 已有模型将作为初始模型继续训练 | 增量树比例=%.2f", incremental_tree_ratio)
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
        red_selector = StableFeatureSelector.from_config(
            self.feature_config,
            color="red",
            random_seed=int(self.model_config.get("random_seed", 2026)),
        )
        red_features = red_selector.fit_transform(red_features, red_labels)
        self.feature_selectors_["red"] = red_selector
        # 增量训练：减少n_estimators，并使用已有模型作为初始模型
        red_params = self._common_parameters()
        red_init_model = None
        if use_incremental and "red" in self.models_:
            red_params["n_estimators"] = max(10, int(red_params["n_estimators"] * incremental_tree_ratio))
            red_init_model = self.models_["red"]
        red_model = XGBRanker(
            **red_params,
            objective=str(self.model_config.get("red_objective", self.model_config.get("objective", "rank:ndcg"))),
            eval_metric="ndcg@6",
        )
        red_groups = np.full(len(red_labels) // 33, 33, dtype=int)
        if red_init_model is not None:
            red_model.fit(red_features, red_labels, group=red_groups, xgb_model=red_init_model)
        else:
            red_model.fit(red_features, red_labels, group=red_groups)
        red_scores = red_model.predict(red_features).reshape(-1, 33)
        red_probability = _row_softmax(
            red_scores,
            temperature=float(self.model_config.get("rank_temperature", 2.0)),
        )
        red_marginal = np.clip(red_probability * 6.0, 0.0, 1.0)
        training_losses["red"] = float(
            np.mean((red_marginal - red_labels.reshape(-1, 33)) ** 2)
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
        blue_selector = StableFeatureSelector.from_config(
            self.feature_config,
            color="blue",
            random_seed=int(self.model_config.get("random_seed", 2026)),
        )
        blue_features = blue_selector.fit_transform(blue_features, blue_labels)
        self.feature_selectors_["blue"] = blue_selector
        # 增量训练：减少n_estimators，并使用已有模型作为初始模型
        blue_params = self._common_parameters()
        blue_init_model = None
        if use_incremental and "blue" in self.models_:
            blue_params["n_estimators"] = max(10, int(blue_params["n_estimators"] * incremental_tree_ratio))
            blue_init_model = self.models_["blue"]
        blue_model = XGBRanker(
            **blue_params,
            objective=str(self.model_config.get("blue_objective", self.model_config.get("objective", "rank:ndcg"))),
            eval_metric="ndcg@1",
        )
        blue_groups = np.full(len(blue_labels) // 16, 16, dtype=int)
        if blue_init_model is not None:
            blue_model.fit(blue_features, blue_labels, group=blue_groups, xgb_model=blue_init_model)
        else:
            blue_model.fit(blue_features, blue_labels, group=blue_groups)
        blue_scores = blue_model.predict(blue_features).reshape(-1, 16)
        blue_probability = _row_softmax(
            blue_scores,
            temperature=float(self.model_config.get("rank_temperature", 2.0)),
        )
        blue_targets = blue_labels.reshape(-1, 16)
        training_losses["blue"] = float(
            np.mean((blue_probability - blue_targets) ** 2)
        )
        self.models_["blue"] = blue_model
        self.blue_sequence_features_ = bool(
            self.model_config.get("use_blue_sequence_features", True)
        )
        self.model_modes_ = {"red": "pairwise_rank_6_of_33", "blue": "grouped_rank_1_of_16"}
        self.feature_importance_ = {
            "red": summarize_feature_importance(
                columns=self.feature_selectors_.get("red").columns_ if self.feature_selectors_.get("red") else None,
                importances=getattr(red_model, "feature_importances_", []),
            ),
            "blue": summarize_feature_importance(
                columns=self.feature_selectors_.get("blue").columns_ if self.feature_selectors_.get("blue") else None,
                importances=getattr(blue_model, "feature_importances_", []),
            ),
        }
        self.training_brier_by_color_ = training_losses
        self.training_brier_ = float(np.mean(list(training_losses.values())))
        return self

    def _common_parameters(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.model_config.get("n_estimators", 60)),
            "max_depth": int(self.model_config.get("max_depth", 3)),
            "learning_rate": float(self.model_config.get("learning_rate", 0.05)),
            "subsample": float(self.model_config.get("subsample", 0.8)),
            "colsample_bytree": float(self.model_config.get("colsample_bytree", 0.8)),
            "reg_alpha": float(self.model_config.get("reg_alpha", 0.2)),
            "reg_lambda": float(self.model_config.get("reg_lambda", 2.0)),
            "n_jobs": int(self.model_config.get("n_jobs", 2)),
            "random_state": int(self.model_config.get("random_seed", 2026)),
        }

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
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
        red_features = self.feature_selectors_.get("red", StableFeatureSelector(enabled=False)).transform(red_features)
        red_scores = np.asarray(self.models_["red"].predict(red_features), dtype=float)
        result["red"] = _row_softmax(
            red_scores.reshape(1, -1),
            temperature=float(self.model_config.get("rank_temperature", 2.0)),
        )[0]

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
        blue_features = self.feature_selectors_.get("blue", StableFeatureSelector(enabled=False)).transform(blue_features)
        blue_scores = np.asarray(self.models_["blue"].predict(blue_features), dtype=float)
        result["blue"] = _row_softmax(
            blue_scores.reshape(1, -1),
            temperature=float(self.model_config.get("rank_temperature", 2.0)),
        )[0]
        return result


def _row_softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(np.clip(scaled, -50.0, 50.0))
    return exponential / np.maximum(exponential.sum(axis=1, keepdims=True), 1e-12)
