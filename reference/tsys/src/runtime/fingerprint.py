from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

import numpy as np


def _stable_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _stable_native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _stable_native(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_stable_native(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        _stable_native(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prediction_fingerprint(
    *,
    config_hash: str,
    data_hash: str | None,
    source_hash: str,
    random_seed: int,
    probabilities: dict[str, Any] | None = None,
    candidate_pool: Any | None = None,
    portfolio_input: Any | None = None,
    final_tickets: Any | None = None,
) -> dict[str, Any]:
    """Build reproducibility hashes without changing prediction data."""

    parts: dict[str, Any] = {
        "config_hash": config_hash,
        "data_hash": data_hash,
        "source_hash": source_hash,
        "random_seed": int(random_seed),
    }
    if probabilities is not None:
        parts["prediction_hash"] = stable_hash(probabilities)
    if candidate_pool is not None:
        parts["candidate_hash"] = stable_hash(candidate_pool)
    if portfolio_input is not None:
        parts["portfolio_input_hash"] = stable_hash(portfolio_input)
    if final_tickets is not None:
        parts["final_prediction_hash"] = stable_hash(final_tickets)
    parts["fingerprint_hash"] = stable_hash(parts)
    parts["schema"] = "prediction_fingerprint_v1"
    parts["note"] = "用于新旧版本等价校验，不参与选号。"
    return parts

