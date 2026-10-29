from __future__ import annotations

from typing import Any


def regime_control_report(backtest: dict[str, Any]) -> dict[str, Any]:
    periods = len(backtest.get("period_records", []) or []) if isinstance(backtest, dict) else 0
    return {
        "schema": "regime_controller_v7",
        "periods": periods,
        "mode": "risk_control_only",
        "controls": ["historical_weight", "shrinkage_alpha", "candidate_width", "expert_trust"],
        "rule": "状态模型不得直接执行“某状态→某号码大幅增益”，只作为可信度和风险控制信号。",
    }
