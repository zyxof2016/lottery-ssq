from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def validate_probability_vector(
    values: Iterable[float],
    *,
    expected_size: int,
    label: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate one probability vector and return an auditable health row."""
    array = np.asarray(list(values), dtype=float).reshape(-1)
    finite = bool(array.size == int(expected_size) and np.all(np.isfinite(array)))
    non_negative = bool(finite and np.all(array >= -float(tolerance)))
    total = float(array.sum()) if array.size else 0.0
    normalized = bool(finite and non_negative and abs(total - 1.0) <= float(tolerance))
    entropy = None
    if finite and non_negative and total > 0:
        probs = np.clip(array, 0.0, None)
        probs = probs / float(probs.sum())
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return {
        "label": str(label),
        "expected_size": int(expected_size),
        "actual_size": int(array.size),
        "finite": finite,
        "non_negative": non_negative,
        "sum": total,
        "normalized": normalized,
        "entropy": entropy,
        "passed": bool(finite and non_negative and normalized),
    }


def probability_health_report(
    predictions: dict[str, dict[str, Iterable[float]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, prediction in sorted(predictions.items()):
        if not isinstance(prediction, dict):
            continue
        rows.append(validate_probability_vector(
            prediction.get("red", []),
            expected_size=33,
            label=f"{name}.red",
        ))
        rows.append(validate_probability_vector(
            prediction.get("blue", []),
            expected_size=16,
            label=f"{name}.blue",
        ))
    return {
        "schema": "probability_health_v7",
        "rows": rows,
        "passed": all(row["passed"] for row in rows) if rows else False,
        "failed_labels": [row["label"] for row in rows if not row["passed"]],
    }
