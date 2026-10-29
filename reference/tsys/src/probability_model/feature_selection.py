from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class StableFeatureSelector:
    """Low-noise filter for compact lottery feature matrices.

    It first removes near-constant columns, then keeps the top mutual-information
    columns.  The selector is fitted only on the left-truncated training matrix
    and reused for prediction, so it does not introduce future leakage.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_features: int | None = None,
        min_variance: float = 1e-10,
        random_seed: int = 2026,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_features = None if max_features is None else int(max_features)
        self.min_variance = float(min_variance)
        self.random_seed = int(random_seed)
        self.columns_: list[str] | None = None
        self.scores_: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], *, color: str, random_seed: int) -> "StableFeatureSelector":
        enabled = bool(config.get("feature_selection_enabled", True))
        default_cap = 80 if str(color) == "red" else 64
        cap_config = config.get("feature_selection_max_features_by_color", {})
        max_features = int(cap_config.get(str(color), config.get("feature_selection_max_features", default_cap)))
        return cls(
            enabled=enabled,
            max_features=max_features,
            min_variance=float(config.get("feature_selection_min_variance", 1e-10)),
            random_seed=random_seed,
        )

    def fit_transform(self, features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        if not self.enabled or self.max_features is None or int(self.max_features) <= 0:
            self.columns_ = list(features.columns)
            return features
        variances = features.var(axis=0).fillna(0.0)
        viable = [column for column in features.columns if float(variances[column]) > self.min_variance]
        if not viable:
            self.columns_ = list(features.columns)
            return features
        if len(viable) <= int(self.max_features):
            self.columns_ = viable
            return features.loc[:, self.columns_]
        values = features.loc[:, viable].to_numpy(dtype=float)
        targets = np.asarray(labels, dtype=int)
        try:
            scores = mutual_info_classif(
                values,
                targets,
                discrete_features=False,
                random_state=self.random_seed,
            )
        except Exception:
            scores = np.asarray([float(variances[column]) for column in viable], dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(-scores, kind="stable")[: int(self.max_features)]
        self.columns_ = [viable[int(index)] for index in order]
        self.scores_ = {
            viable[int(index)]: float(scores[int(index)])
            for index in order
        }
        return features.loc[:, self.columns_]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_:
            return features
        missing = [column for column in self.columns_ if column not in features.columns]
        if missing:
            raise ValueError(f"预测特征缺少训练期保留列: {missing[:5]}")
        return features.loc[:, self.columns_]


def summarize_feature_importance(
    *,
    columns: list[str] | None,
    importances: Any,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    if not columns:
        return []
    values = np.asarray(importances, dtype=float)
    if values.size != len(columns):
        return []
    total = float(np.sum(np.clip(values, 0.0, None)))
    denominator = total if total > 0 else 1.0
    order = np.argsort(-values, kind="stable")[: max(1, int(top_k))]
    return [
        {
            "feature": str(columns[int(index)]),
            "importance": float(values[int(index)]),
            "share": float(max(values[int(index)], 0.0) / denominator),
        }
        for index in order
        if np.isfinite(values[int(index)])
    ]
