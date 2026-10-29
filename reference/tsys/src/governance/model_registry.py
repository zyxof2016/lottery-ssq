from __future__ import annotations

from typing import Any


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic_regression": {"family": "ml", "colors": ["red", "blue"]},
    "xgboost": {"family": "ml", "colors": ["red", "blue"]},
    "lightgbm_ranker": {"family": "ranker", "colors": ["red", "blue"]},
    "catboost": {"family": "ml", "colors": ["red", "blue"]},
    "mlp": {"family": "neural", "colors": ["red", "blue"]},
    "frequency": {"family": "statistical", "colors": ["red", "blue"]},
    "time_decay": {"family": "statistical", "colors": ["red", "blue"]},
    "bayesian": {"family": "statistical", "colors": ["red", "blue"]},
    "pattern_memory": {"family": "sequence", "colors": ["blue"]},
    "blue_specialist": {"family": "specialist", "colors": ["blue"]},
    "uniform_baseline": {"family": "baseline", "colors": ["red", "blue"]},
}


def model_registry_report(model_scorecard: dict[str, Any]) -> dict[str, Any]:
    observed = sorted((model_scorecard.get("models") or {}).keys()) if isinstance(model_scorecard, dict) else []
    registry = {
        name: MODEL_REGISTRY.get(name, {"family": "unregistered", "colors": ["red", "blue"]})
        for name in observed
    }
    for name, payload in MODEL_REGISTRY.items():
        registry.setdefault(name, payload)
    return {
        "schema": "model_registry_v7",
        "flow": ["Model Registry", "Red/Blue", "Research", "Shadow", "Challenger", "Champion", "Retired"],
        "registered_models": registry,
        "observed_models": observed,
        "unregistered_observed_models": [name for name in observed if name not in MODEL_REGISTRY],
    }
