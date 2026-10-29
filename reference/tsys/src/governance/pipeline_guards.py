from __future__ import annotations

from typing import Any

import numpy as np


def temper_with_uniform(probabilities: np.ndarray, relief: float) -> np.ndarray:
    """Blend a probability vector with its uniform baseline.

    This helper is shared by the current-output path and the walk-forward
    validation path.  Keeping the guard math in one place avoids future drift
    where one path is patched but the other still uses the old behavior.
    """
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, 0.0, None)
    if values.size == 0:
        return values
    total = float(np.sum(values))
    if total <= 0:
        values = np.full(values.size, 1.0 / values.size)
    else:
        values = values / total
    uniform = np.full(values.size, 1.0 / values.size)
    relief = min(max(float(relief), 0.0), 1.0)
    mixed = (1.0 - relief) * values + relief * uniform
    mixed_total = float(np.sum(mixed))
    return mixed / mixed_total if mixed_total > 0 else uniform


def uniform_probabilities() -> dict[str, np.ndarray]:
    return {
        "red": np.full(33, 1.0 / 33.0),
        "blue": np.full(16, 1.0 / 16.0),
    }


def merge_candidate_pools(*pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}
    for pool in pools:
        for candidate in pool:
            key = (
                tuple(sorted(int(number) for number in candidate.get("red", []))),
                int(candidate.get("blue")),
            )
            if key not in merged:
                merged[key] = candidate
                continue
            current = merged[key]
            if float(candidate.get("quality_score", 0.0)) > float(current.get("quality_score", 0.0)):
                merged[key] = {**current, **candidate}
            else:
                roles = {
                    str(current.get("selection_role", "")).strip(),
                    str(candidate.get("selection_role", "")).strip(),
                }
                merged[key] = {
                    **current,
                    "selection_role": "+".join(sorted(role for role in roles if role)) or current.get("selection_role"),
                }
    return list(merged.values())


def red_guard_stoploss_decision(
    guard: dict[str, Any] | None,
    optimizer_config: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether recent red-side evidence should trigger safe fallback.

    This is a final-output safety layer.  It does not claim that a no-signal
    portfolio predicts better; it prevents a recently reversed model signal
    from being amplified into the archived recommendation.
    """
    if not bool(optimizer_config.get("red_guard_stoploss_to_no_signal", True)):
        return {"active": False, "reason": "disabled"}
    guard = guard or {}
    if not bool(guard.get("active")):
        return {"active": False, "reason": "guard_inactive", "guard": guard}

    require_severe = bool(optimizer_config.get("red_guard_stoploss_require_severe", True))
    severe_active = bool(guard.get("severe_active"))
    severe_methods = int(guard.get("severe_methods", 0) or 0)
    severe_required = int(guard.get("severe_required_methods", 1) or 1)
    emergency_collapse = str(guard.get("reason", "")).startswith("recent_consensus_red_collapse")

    methods = guard.get("methods", {}) or {}
    consensus = methods.get("consensus", {}) or {}
    consensus_shortfall = float(consensus.get("average_shortfall", 0.0) or 0.0)
    mean_shortfall = float(
        guard.get(
            "mean_shortfall",
            consensus_shortfall if consensus_shortfall else 0.0,
        )
        or 0.0
    )
    min_mean_shortfall = float(
        optimizer_config.get("red_guard_stoploss_min_mean_shortfall", 0.25)
    )
    min_consensus_shortfall = float(
        optimizer_config.get("red_guard_stoploss_min_consensus_shortfall", 0.20)
    )

    severe_enough = bool(
        severe_active
        or emergency_collapse
        or severe_methods >= severe_required
        or bool(consensus.get("severe_adverse"))
    )
    shortfall_enough = bool(
        mean_shortfall >= min_mean_shortfall
        or consensus_shortfall >= min_consensus_shortfall
    )
    active = bool(shortfall_enough and (severe_enough or not require_severe))
    return {
        "active": active,
        "reason": (
            "red_signal_reversed_stoploss_to_same_flow_no_signal"
            if active
            else "red_signal_not_reversed_enough_for_stoploss"
        ),
        "require_severe": require_severe,
        "severe_enough": severe_enough,
        "shortfall_enough": shortfall_enough,
        "mean_shortfall": mean_shortfall,
        "consensus_shortfall": consensus_shortfall,
        "min_mean_shortfall": min_mean_shortfall,
        "min_consensus_shortfall": min_consensus_shortfall,
        "guard_reason": guard.get("reason"),
    }
