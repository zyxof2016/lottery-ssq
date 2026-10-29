from __future__ import annotations

from typing import Any

import numpy as np

from src.blue.probability_shrinkage import alpha_for_blue_edge_state, shrink_blue_probability_to_uniform


def v7_blue_shrinkage_policy(probability: np.ndarray, edge_state: str, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = alpha_for_blue_edge_state(edge_state, config)
    shrunk, info = shrink_blue_probability_to_uniform(probability, alpha=alpha, reason=f"v7_{edge_state}")
    return shrunk, {**info, "schema": "blue_shrinkage_policy_v7", "edge_state": edge_state}
