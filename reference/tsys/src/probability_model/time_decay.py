from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineer.base_stats import occurrence_matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


class TimeDecayModel:
    name = "time_decay"

    def __init__(self, decay_factor: float = 0.98):
        self.decay_factor = float(decay_factor)

    def fit(self, history: pd.DataFrame) -> "TimeDecayModel":
        ages = np.arange(len(history) - 1, -1, -1)
        weights = np.power(self.decay_factor, ages)
        self.scores_ = {
            color: _normalize(weights @ occurrence_matrix(history, color) + 1e-9)
            for color in ("red", "blue")
        }
        return self

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        del history, target_idx
        return {key: value.copy() for key, value in self.scores_.items()}
