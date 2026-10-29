from __future__ import annotations

from typing import Any


def allocate_portfolio_budget(edge_state: str, count: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    count = max(1, int(count))
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    portfolio = (v7.get("portfolio", {}) if isinstance(v7, dict) else {}) or {}
    state = str(edge_state or "NO_EDGE").upper()
    presets = portfolio.get("budget_by_edge_state") or {
        "STRONG_EDGE": {"probability": 2, "coverage": 2, "exploration": 1},
        "WEAK_EDGE": {"probability": 1, "coverage": 3, "exploration": 1},
        "NO_EDGE": {"probability": 1, "coverage": 3, "exploration": 1},
        "RECALL_GUARD": {"probability": 1, "coverage": 3, "exploration": 1},
    }
    selected = dict(presets.get(state, presets["NO_EDGE"]))
    total = sum(max(0, int(value)) for value in selected.values())
    if total <= 0:
        selected = {"probability": 1, "coverage": max(0, count - 2), "exploration": 1}
        total = sum(selected.values())
    scale = count / total
    slots = {key: max(0, int(round(max(0, int(value)) * scale))) for key, value in selected.items()}
    while sum(slots.values()) < count:
        priority = ["coverage", "probability", "exploration"] if state == "NO_EDGE" else ["probability", "coverage", "exploration"]
        slots[priority[(count - sum(slots.values())) % len(priority)]] += 1
    while sum(slots.values()) > count:
        for key in ("exploration", "probability", "coverage"):
            if slots.get(key, 0) > 0 and sum(slots.values()) > count:
                slots[key] -= 1
    return {
        "schema": "portfolio_budget_v7",
        "edge_state": state,
        "count": count,
        "slots": {
            "probability": int(slots.get("probability", 0)),
            "coverage": int(slots.get("coverage", 0)),
            "exploration": int(slots.get("exploration", 0)),
        },
        "rule": "预算由红球独立 Inner/OOS Evidence 状态驱动；RECALL_GUARD 会优先保护候选召回，禁止由蓝球 Edge 控制红球预算。",
    }
