from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineer.base_stats import occurrence_matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


class FrequencyModel:
    name = "frequency"

    def fit(self, history: pd.DataFrame) -> "FrequencyModel":
        self.scores_ = {
            color: _normalize(occurrence_matrix(history, color).sum(axis=0) + 1e-9)
            for color in ("red", "blue")
        }
        return self

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        del history, target_idx
        return {key: value.copy() for key, value in self.scores_.items()}
