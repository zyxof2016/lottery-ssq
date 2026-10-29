from __future__ import annotations

from typing import Any

import numpy as np


def model_scorecard(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize model/expert behavior across short, medium and long windows."""
    if not period_records:
        return {"schema": "model_scorecard_v7", "periods": 0, "models": {}}
    model_names = sorted({
        name
        for record in period_records
        for name in (record.get("model_validation") or {}).keys()
    })
    windows = [30, 100, 300]
    result: dict[str, Any] = {}
    for name in model_names:
        rows = [
            record
            for record in period_records
            if name in (record.get("model_validation") or {})
        ]
        model_row: dict[str, Any] = {"periods": len(rows), "windows": {}}
        for window in windows:
            window_rows = rows[-min(window, len(rows)):]
            if not window_rows:
                continue
            red_hits = [
                float(((row.get("model_validation") or {}).get(name) or {}).get("red_hits", 0.0))
                for row in window_rows
            ]
            blue_hits = [
                bool(((row.get("model_validation") or {}).get(name) or {}).get("blue_hit", False))
                for row in window_rows
            ]
            blue_hit_at_3 = [
                float(((row.get("model_blue_ranking_metrics") or {}).get(name) or {}).get("hit_at_3"))
                for row in window_rows
                if ((row.get("model_blue_ranking_metrics") or {}).get(name) or {}).get("hit_at_3") is not None
            ]
            blue_hit_at_5 = [
                float(((row.get("model_blue_ranking_metrics") or {}).get(name) or {}).get("hit_at_5"))
                for row in window_rows
                if ((row.get("model_blue_ranking_metrics") or {}).get(name) or {}).get("hit_at_5") is not None
            ]
            blue_ranks = [
                float(((row.get("model_blue_ranking_metrics") or {}).get(name) or {}).get("rank", np.nan))
                for row in window_rows
            ]
            blue_ranks = [rank for rank in blue_ranks if np.isfinite(rank)]
            red_brier = [
                float((((row.get("model_validation") or {}).get(name) or {}).get("brier") or {}).get("red"))
                for row in window_rows
                if (((row.get("model_validation") or {}).get(name) or {}).get("brier") or {}).get("red") is not None
            ]
            blue_brier = [
                float((((row.get("model_validation") or {}).get(name) or {}).get("brier") or {}).get("blue"))
                for row in window_rows
                if (((row.get("model_validation") or {}).get(name) or {}).get("brier") or {}).get("blue") is not None
            ]
            model_row["windows"][str(window)] = {
                "periods": len(window_rows),
                "average_red_hits": float(np.mean(red_hits)) if red_hits else None,
                "blue_hit_at_1": float(np.mean(blue_hits)) if blue_hits else None,
                "blue_hit_at_3": float(np.mean(blue_hit_at_3)) if blue_hit_at_3 else None,
                "blue_hit_at_5": float(np.mean(blue_hit_at_5)) if blue_hit_at_5 else None,
                "blue_mean_rank": float(np.mean(blue_ranks)) if blue_ranks else None,
                "red_brier": float(np.mean(red_brier)) if red_brier else None,
                "blue_brier": float(np.mean(blue_brier)) if blue_brier else None,
            }
        result[name] = model_row
    return {
        "schema": "model_scorecard_v7",
        "periods": len(period_records),
        "models": result,
    }
