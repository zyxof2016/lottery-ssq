from __future__ import annotations

import numpy as np


def normalized_entropy(probability, *, size: int | None = None) -> float:
    values = np.asarray(probability, dtype=float)
    if size is None:
        size = int(values.size)
    if values.size == 0 or size <= 1:
        return 0.0
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return 1.0
    p = values / total
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / np.log(size))
