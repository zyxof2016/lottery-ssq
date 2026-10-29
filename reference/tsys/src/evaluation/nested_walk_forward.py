from __future__ import annotations

from typing import Any


def nested_oos_report(backtest: dict[str, Any]) -> dict[str, Any]:
    calibration = backtest.get("calibration_summary", {}) if isinstance(backtest, dict) else {}
    release = backtest.get("release_evidence", {}) if isinstance(backtest, dict) else {}
    windows = release.get("windows", {}) if isinstance(release, dict) else {}
    confirmation = windows.get("confirmation", {}) if isinstance(windows, dict) else {}
    runtime = backtest.get("runtime_control", {}) if isinstance(backtest, dict) else {}
    inner = runtime.get("nested_inner_oos", {}) if isinstance(runtime, dict) else {}
    selected = inner.get("selected", {}) if isinstance(inner, dict) else {}
    records = backtest.get("period_records", []) if isinstance(backtest, dict) else []
    period_inner: list[dict[str, Any]] = []
    for record in records or []:
        control = record.get("runtime_control", {}) if isinstance(record, dict) else {}
        nested = control.get("nested_inner_oos", {}) if isinstance(control, dict) else {}
        chosen = nested.get("selected", {}) if isinstance(nested, dict) else {}
        comparison = nested.get("strategy_comparison", {}) if isinstance(nested, dict) else {}
        if chosen:
            period_inner.append(
                {
                    "period": record.get("period"),
                    "prediction_set_k": chosen.get("prediction_set_k"),
                    "blue_alpha": chosen.get("blue_alpha"),
                    "red_budget_state": chosen.get("red_budget_state"),
                    "jaccard_max_overlap": chosen.get("jaccard_max_overlap"),
                    "expert_cap": chosen.get("expert_cap"),
                    "portfolio_strategy": chosen.get("portfolio_strategy"),
                    "selected_strategy": comparison.get("selected"),
                    "strategy_scores": comparison.get("scores", []),
                }
            )
    return {
        "schema": "nested_oos_report_v7",
        "completed_holdout_periods": calibration.get("completed_holdout_periods"),
        "development_periods": (windows.get("development") or {}).get("periods") if isinstance(windows, dict) else None,
        "confirmatory_periods": confirmation.get("periods"),
        "confirmatory_hyperparameters_frozen": confirmation.get("hyperparameters_frozen"),
        "future_validation_rows_used_for_adjustment": calibration.get("future_validation_rows_used_for_adjustment"),
        "inner_search": {
            "executed": bool(inner.get("active") or period_inner),
            "parameters": [
                "Prediction Set K",
                "Blue alpha",
                "Red budget",
                "Jaccard penalty",
                "Expert cap",
                "Portfolio strategy",
            ],
            "current_selected": selected,
            "current_strategy_comparison": inner.get("strategy_comparison", {}) if isinstance(inner, dict) else {},
            "period_selected_tail": period_inner[-10:],
        },
        "outer_oos": {
            "evaluation_only": True,
            "release_windows": windows,
            "rule": "Outer OOS only evaluates frozen decisions; Inner OOS is the only dynamic parameter selector.",
        },
        "interpretation": "Inner OOS selects runtime controls; Outer validation rows are evaluation-only and are reported separately for release decisions.",
    }
