from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.feature_engineer.base_stats import occurrence_matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / len(values))


class BayesianModel:
    name = "bayesian"

    def __init__(self, alpha: float = 1.0, beta_prior: float = 1.0):
        self.alpha = float(alpha)
        self.beta_prior = float(beta_prior)

    def fit(self, history: pd.DataFrame) -> "BayesianModel":
        self.scores_: dict[str, np.ndarray] = {}
        self.intervals_: dict[str, np.ndarray] = {}
        for color in ("red", "blue"):
            matrix = occurrence_matrix(history, color)
            successes = matrix.sum(axis=0)
            failures = len(history) - successes
            posterior_alpha = successes + self.alpha
            posterior_beta = failures + self.beta_prior
            means = posterior_alpha / (posterior_alpha + posterior_beta)
            self.scores_[color] = _normalize(means)
            self.intervals_[color] = np.column_stack([
                beta.ppf(0.025, posterior_alpha, posterior_beta),
                beta.ppf(0.975, posterior_alpha, posterior_beta),
            ])
        return self

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        del history, target_idx
        return {key: value.copy() for key, value in self.scores_.items()}

    def uncertainty(self, color: str) -> np.ndarray:
        return self.intervals_[color].copy()
