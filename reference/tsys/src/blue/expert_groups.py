from __future__ import annotations

from typing import Any


DEFAULT_BLUE_EXPERT_GROUPS = {
    "frequency": "statistical",
    "bayesian": "statistical",
    "time_decay": "statistical",
    "pattern_memory": "statistical",
    "logistic_regression": "machine_learning",
    "xgboost": "machine_learning",
    "lightgbm_ranker": "machine_learning",
    "mlp": "machine_learning",
    "catboost": "machine_learning",
    "uniform_baseline": "baseline",
}


def cap_blue_expert_groups(
    weights: dict[str, float],
    *,
    group_map: dict[str, str] | None = None,
    group_caps: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    group_map = {**DEFAULT_BLUE_EXPERT_GROUPS, **(group_map or {})}
    group_caps = {
        "statistical": 0.60,
        "machine_learning": 0.45,
        "baseline": 1.0,
        **(group_caps or {}),
    }
    cleaned = {name: max(0.0, float(value)) for name, value in weights.items()}
    groups: dict[str, list[str]] = {}
    for name in cleaned:
        groups.setdefault(group_map.get(name, "other"), []).append(name)
    capped = dict(cleaned)
    events = []
    for group, names in groups.items():
        total = sum(capped[name] for name in names)
        cap = float(group_caps.get(group, 1.0))
        if total > cap > 0.0:
            scale = cap / total
            for name in names:
                capped[name] *= scale
            events.append({"group": group, "before": total, "cap": cap, "scale": scale})
    total = sum(capped.values())
    if total > 0.0:
        capped = {name: value / total for name, value in capped.items()}
    return capped, {
        "schema": "blue_expert_group_caps_v7",
        "events": events,
        "group_totals": {
            group: sum(capped.get(name, 0.0) for name in names)
            for group, names in groups.items()
        },
    }
