from __future__ import annotations

from typing import Any

from src.governance.scorecard import model_scorecard


def evidence_model_scorecard(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    report = model_scorecard(period_records)
    return {
        "schema": "model_scorecard_v7",
        "source": "strict_walk_forward_oos",
        "scorecard": report,
        "rule": "模型仅凭 OOS 证据进入 Shadow/Challenger/Champion；训练集表现不作为生产依据。",
    }
