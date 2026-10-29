from __future__ import annotations

from typing import Any

from src.portfolio.budget_allocator import allocate_portfolio_budget
from src.red.portfolio import select_red_portfolio


def select_v7_adaptive_portfolio(
    pool: list[dict[str, Any]],
    selected_by_method: dict[str, list[dict[str, Any]]],
    count: int,
    config: dict[str, Any],
    *,
    edge_state: str = "WEAK_EDGE",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget = allocate_portfolio_budget(edge_state, count, config)
    shadow_config = {
        **config,
        "v7": {
            **(config.get("v7") or {}),
            "current_red_edge_state": edge_state,
            "portfolio": {
                **(((config.get("v7") or {}).get("portfolio") or {})),
                "dynamic_budget": True,
                "forced_slots": dict(budget["slots"]),
            },
        },
    }
    tickets, info = select_red_portfolio(pool, selected_by_method, count, shadow_config)
    return tickets, {
        "schema": "adaptive_portfolio_optimizer_v7",
        "active": True,
        "budget": budget,
        "selection": info,
    }
