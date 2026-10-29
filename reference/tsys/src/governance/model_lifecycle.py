from __future__ import annotations

from typing import Any

from src.blue.model_lifecycle import blue_model_lifecycle
from src.governance.model_registry import model_registry_report
from src.red.model_lifecycle import red_model_lifecycle


def model_lifecycle_report(backtest: dict[str, Any]) -> dict[str, Any]:
    scorecard = backtest.get("model_scorecard", {}) if isinstance(backtest, dict) else {}
    return {
        "schema": "model_lifecycle_v7",
        "registry": model_registry_report(scorecard),
        "red": red_model_lifecycle(scorecard),
        "blue": blue_model_lifecycle(scorecard),
        "stages": ["Research", "Shadow", "Challenger", "Champion", "Retired"],
        "flow": {
            "model_registry": "Model Registry",
            "red": ["Research", "Shadow", "Challenger", "Champion", "Retired"],
            "blue": ["Research", "Shadow", "Challenger", "Champion", "Retired"],
        },
        "rule": "新模型必须先 Shadow 记录完整 OOS，再进入 Challenger；Release Gate 通过后才可 Champion。",
    }
