from __future__ import annotations

from typing import Any


def hmm_regime_placeholder(history_length: int) -> dict[str, Any]:
    return {
        "schema": "hmm_regime_v7",
        "active": False,
        "history_length": int(history_length),
        "rule": "HMM 在 V7 中只作为风险/可信度控制的 Research/Shadow 输入，不直接放大号码概率。",
    }
