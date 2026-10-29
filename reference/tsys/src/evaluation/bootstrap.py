from __future__ import annotations

from typing import Iterable

import numpy as np


def block_bootstrap_ci(
    values: Iterable[float],
    *,
    block_size: int = 10,
    repetitions: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 2026,
) -> dict[str, float | int | None]:
    series = np.asarray([float(value) for value in values], dtype=float)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {"schema": "block_bootstrap_ci_v7", "sample_size": 0, "mean": None, "lower": None, "upper": None}
    rng = np.random.default_rng(int(random_seed))
    block_size = max(1, min(int(block_size), len(series)))
    starts = np.arange(0, len(series) - block_size + 1)
    estimates = []
    for _ in range(max(1, int(repetitions))):
        sampled = []
        while len(sampled) < len(series):
            start = int(rng.choice(starts))
            sampled.extend(series[start : start + block_size])
        estimates.append(float(np.mean(sampled[: len(series)])))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "schema": "block_bootstrap_ci_v7",
        "sample_size": int(len(series)),
        "block_size": int(block_size),
        "repetitions": int(repetitions),
        "mean": float(np.mean(series)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }
