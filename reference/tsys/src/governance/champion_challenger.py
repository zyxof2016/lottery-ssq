from __future__ import annotations

from typing import Any

from src.governance.model_lifecycle import model_lifecycle_report


def champion_challenger_report(backtest: dict[str, Any]) -> dict[str, Any]:
    lifecycle = model_lifecycle_report(backtest)
    return {
        "schema": "champion_challenger_report_v7",
        "unified_lifecycle_entry": "src.governance.model_lifecycle.model_lifecycle_report",
        "lifecycle": lifecycle,
        "red": lifecycle.get("red", {}),
        "blue": lifecycle.get("blue", {}),
        "policy": "Champion/Challenger state is now derived from the unified Red/Blue lifecycle entry.",
    }
