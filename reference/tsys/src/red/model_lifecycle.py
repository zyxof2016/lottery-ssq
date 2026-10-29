from __future__ import annotations

from typing import Any


def red_model_lifecycle(
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
        red_top6_hits = latest.get("red_top6_hits", latest.get("red_hit_at_6"))
        red_brier = latest.get("red_brier")
        if periods < min_shadow_periods:
            stage = "Shadow"
        elif red_top6_hits is not None and float(red_top6_hits) >= 1.25 and (red_brier is None or float(red_brier) <= 0.1500):
            stage = "Champion Candidate"
        elif red_top6_hits is not None and float(red_top6_hits) >= 1.10:
            stage = "Challenger"
        else:
            stage = "Retired"
        result[str(name)] = {
            "stage": stage,
            "periods": periods,
            "red_top6_hits": red_top6_hits,
            "red_brier": red_brier,
            "rule": "V7 Red lifecycle is decided from strict OOS scorecard only.",
        }
    return {
        "schema": "red_model_lifecycle_v7",
        "models": result,
        "min_shadow_periods": int(min_shadow_periods),
    }
