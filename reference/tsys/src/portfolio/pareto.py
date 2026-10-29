from __future__ import annotations

from typing import Any


def pareto_frontier(items: list[dict[str, Any]], objective_keys: list[str]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for item in items:
        dominated = False
        for other in items:
            if other is item:
                continue
            ge_all = all(float(other.get(key, 0.0) or 0.0) >= float(item.get(key, 0.0) or 0.0) for key in objective_keys)
            gt_any = any(float(other.get(key, 0.0) or 0.0) > float(item.get(key, 0.0) or 0.0) for key in objective_keys)
            if ge_all and gt_any:
                dominated = True
                break
        if not dominated:
            frontier.append(item)
    return frontier
