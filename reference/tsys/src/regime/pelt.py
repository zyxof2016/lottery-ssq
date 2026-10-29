from __future__ import annotations

from typing import Any


def pelt_changepoint_placeholder(history_length: int) -> dict[str, Any]:
    return {
        "schema": "pelt_changepoint_v7",
        "active": False,
        "history_length": int(history_length),
        "rule": "PELT/变点检测只用于历史降权、alpha 与候选宽度控制，不直接指定号码。",
    }
