from __future__ import annotations

from typing import Any

import numpy as np


def alpha_for_blue_edge_state(edge_state: str, config: dict[str, Any] | None = None) -> float:
    """Map V7 blue evidence state to model-probability share alpha."""
    v7 = ((config or {}).get("v7") or {}) if isinstance(config, dict) else {}
    uncertainty = (v7.get("uncertainty") or {}) if isinstance(v7, dict) else {}
    blue = (v7.get("blue") or {}) if isinstance(v7, dict) else {}
    alpha_by_state = {
        "STRONG_EDGE": float(uncertainty.get("strong_edge_alpha", blue.get("strong_edge_alpha", 0.70))),
        "WEAK_EDGE": float(uncertainty.get("weak_edge_alpha", blue.get("weak_edge_alpha", 0.22))),
        "NO_EDGE": float(uncertainty.get("no_edge_alpha", blue.get("no_edge_alpha", 0.08))),
    }
    return float(np.clip(alpha_by_state.get(str(edge_state), alpha_by_state["NO_EDGE"]), 0.0, 1.0))


def _normalize(probability: np.ndarray, size: int = 16) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.shape[0] != size:
        values = np.resize(values, size)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(size, 1.0 / size)
    return values / total


def shrink_blue_probability_to_uniform(
    probability: np.ndarray,
    *,
    alpha: float,
    reason: str = "v7_evidence_shrinkage",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply V7 evidence shrinkage: p=(1-alpha)U+alpha*p_model."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    model = _normalize(probability, 16)
    uniform = np.full(16, 1.0 / 16.0)
    final = _normalize((1.0 - alpha) * uniform + alpha * model, 16)
    entropy = -float(np.sum(final * np.log(np.clip(final, 1e-12, 1.0))) / np.log(16.0))
    return final, {
        "schema": "blue_probability_shrinkage_v7",
        "alpha": alpha,
        "uniform_share": float(1.0 - alpha),
        "entropy": entropy,
        "reason": reason,
    }
