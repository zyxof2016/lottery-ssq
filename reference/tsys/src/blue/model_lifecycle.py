from __future__ import annotations

from typing import Any


def blue_model_lifecycle(
    model_scorecard: dict[str, Any],
    *,
    min_shadow_periods: int = 300,
) -> dict[str, Any]:
    models = model_scorecard.get("models", {}) if isinstance(model_scorecard, dict) else {}
    result: dict[str, Any] = {}
    for name, payload in models.items():
        windows = payload.get("windows", {}) if isinstance(payload, dict) else {}
        latest = windows.get("300") or windows.get("100") or windows.get("30") or {}
        periods = int(payload.get("periods", 0) or latest.get("periods", 0) or 0)
        hit5 = latest.get("blue_hit_at_5")
        brier = latest.get("blue_brier")
        if periods < min_shadow_periods:
            stage = "Shadow"
        elif hit5 is not None and float(hit5) >= 5.0 / 16.0 and (brier is None or float(brier) <= 0.0589):
            stage = "Champion Candidate"
        elif hit5 is not None and float(hit5) >= 4.5 / 16.0:
            stage = "Challenger"
        else:
            stage = "Retired"
        result[str(name)] = {
            "stage": stage,
            "periods": periods,
            "blue_hit_at_5": hit5,
            "blue_brier": brier,
            "rule": "V7 Champion/Challenger/Shadow/Retired from OOS scorecard",
        }
    return {
        "schema": "blue_model_lifecycle_v7",
        "models": result,
        "min_shadow_periods": int(min_shadow_periods),
    }
