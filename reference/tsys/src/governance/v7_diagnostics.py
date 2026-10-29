from __future__ import annotations

from typing import Any

from src.candidate.adaptive_topk import adaptive_topk_report
from src.candidate.recall_guard import candidate_recall_guard_report
from src.candidate.survival_tracker import number_survival_report
from src.evidence.evidence_engine import build_evidence_report
from src.evidence.falsification import falsification_report
from src.evidence.runtime_control import build_evidence_runtime_control
from src.evidence.scorecard import evidence_model_scorecard
from src.evaluation.bootstrap import block_bootstrap_ci
from src.evaluation.matched_random import matched_random_report
from src.evaluation.nested_walk_forward import nested_oos_report
from src.governance.expert_decorrelation import expert_decorrelation_report
from src.governance.model_lifecycle import model_lifecycle_report
from src.portfolio.compression_audit import compression_audit_report
from src.portfolio.regret import portfolio_regret_report
from src.red.edge_detector import red_edge_report
from src.regime.regime_controller import regime_control_report
from src.uncertainty.calibration import uncertainty_calibration_report
from src.uncertainty.prediction_set import blue_prediction_set_report


def _method_series(backtest: dict[str, Any], method: str, metric: str) -> list[float]:
    rows: list[float] = []
    for record in backtest.get("period_records", []) or []:
        try:
            rows.append(float(record["methods"][method]["model"][metric]))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def build_v7_diagnostics(backtest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    records = backtest.get("period_records", []) if isinstance(backtest, dict) else []
    evidence = build_evidence_report(backtest, config)
    runtime_control = (
        backtest.get("runtime_control")
        if isinstance(backtest.get("runtime_control"), dict)
        else build_evidence_runtime_control(
            records,
            config,
            ticket_count=int((config.get("optimizer") or {}).get("consensus_count", (config.get("generation") or {}).get("ticket_count", 5))),
        )
    )
    last_portfolio = {}
    for record in reversed(records or []):
        candidate_exp = record.get("candidate_experiment") if isinstance(record, dict) else None
        if isinstance(candidate_exp, dict):
            last_portfolio = candidate_exp.get("v7_portfolio") or {}
        if last_portfolio:
            break
    bootstrap = {}
    for method in (backtest.get("summary") or {}):
        bootstrap[method] = {
            "average_red_hits": block_bootstrap_ci(
                _method_series(backtest, method, "average_red_hits"),
                random_seed=int((config.get("project") or {}).get("random_seed", 2026)),
            ),
            "any_blue_hit": block_bootstrap_ci(
                _method_series(backtest, method, "any_blue_hit"),
                random_seed=int((config.get("project") or {}).get("random_seed", 2026)) + 91,
            ),
        }
    return {
        "schema": "evidence_first_adaptive_portfolio_system_v7",
        "enabled": bool((config.get("v7") or {}).get("enabled", True)),
        "mode": ((config.get("v7") or {}).get("mode", "evidence_first")),
        "evidence_report": evidence,
        "runtime_control": runtime_control,
        "red_edge_state": red_edge_report(records, config),
        "blue_edge_state": runtime_control.get("blue_edge", {}),
        "blue_uncertainty_report": uncertainty_calibration_report(records),
        "blue_prediction_set": blue_prediction_set_report(records, config),
        "blue_topk_ladder": adaptive_topk_report(records, config),
        "red_recall_funnel": candidate_recall_guard_report(records),
        "number_survival": number_survival_report(records),
        "portfolio_pareto": last_portfolio.get("pareto", {
            "schema": "red_portfolio_pareto_v7",
            "active": False,
            "reason": "no_current_v7_portfolio_diagnostics_found",
        }),
        "compression_audit": compression_audit_report(records),
        "regret_report": portfolio_regret_report(records),
        "model_scorecard": evidence_model_scorecard(records),
        "random_baseline": matched_random_report(backtest),
        "nested_oos": nested_oos_report(backtest),
        "bootstrap": bootstrap,
        "falsification_report": falsification_report(backtest),
        "expert_decorrelation": expert_decorrelation_report(backtest, config),
        "model_lifecycle": model_lifecycle_report(backtest),
        "regime_control": regime_control_report(backtest),
        "release_evidence": backtest.get("release_evidence", {}),
        "principle": "NO-EDGE IS A VALID RESULT；无稳定证据时收缩概率、扩大候选集合并覆盖优先。",
    }
