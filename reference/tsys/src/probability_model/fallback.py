from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DisabledProbabilityModel:
    """Uniform placeholder for configured-but-disabled optional models.

    The fusion layer expects every registered model name to produce a red and
    blue probability vector each round.  Returning a transparent uniform
    placeholder keeps the pipeline shape stable without letting an optional
    heavyweight model silently affect rankings.
    """

    def __init__(self, name: str, reason: str = "disabled_by_config"):
        self.name = str(name)
        self.reason = str(reason)
        self.model_modes_ = {"red": self.reason, "blue": self.reason}
        self.models_ = {"red": None, "blue": None}
        self.disabled_ = True
        self.fit_failed_ = False
        self.training_brier_ = None
        self.training_brier_by_color_ = {}
        self.feature_importance_ = {}

    def fit(self, history: pd.DataFrame) -> "DisabledProbabilityModel":
        del history
        return self

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        del history, target_idx
        return {
            "red": np.full(33, 1.0 / 33.0, dtype=float),
            "blue": np.full(16, 1.0 / 16.0, dtype=float),
        }
