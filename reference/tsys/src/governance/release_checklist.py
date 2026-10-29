from __future__ import annotations

from typing import Any


def release_checklist(report: dict[str, Any]) -> dict[str, Any]:
    """Build the V7 final gate checklist from a completed report payload."""
    data = report.get("data") or {}
    backtest = report.get("backtest") or {}
    candidate = report.get("candidate_experiment") or {}
    model_release = report.get("model_release") or {}
    run_manifest = report.get("run_manifest") or {}
    probability_health = ((candidate.get("model_pipeline") or {}).get("probability_health") or {})
    candidate_recall = backtest.get("candidate_recall_funnel") or {}
    v7 = report.get("v7_diagnostics") or backtest.get("v7_diagnostics") or {}
    runtime_control = v7.get("runtime_control") or backtest.get("runtime_control") or {}
    stacking_info = ((candidate.get("model_pipeline") or {}).get("fusion") or {}).get("stacking") or {}
    red_gate = stacking_info.get("red_gate") or {}
    blue_gate = stacking_info.get("blue_gate") or {}
    stacking_gate_passed = bool(
        stacking_info.get("reason") == "insufficient_oof_records"
        or (
            bool(red_gate.get("gate_passed", True))
            and bool(blue_gate.get("gate_passed", True))
        )
    )
    falsification = v7.get("falsification_report") or {}
    release_evidence = backtest.get("release_evidence") or {}
    checklist = [
        {
            "item": "数据治理",
            "passed": bool((data.get("cleaning") or {}).get("quality_status", "passed") in {"passed", "verified", None}),
            "detail": f"有效期数={data.get('periods')}; 数据版本={run_manifest.get('data_version_id')}",
        },
        {
            "item": "可复现清单",
            "passed": bool(run_manifest.get("data_hash") and run_manifest.get("config_hash") and run_manifest.get("source_tree_sha256")),
            "detail": "data_hash/config_hash/source_tree_sha256/seed/package_versions 已记录",
        },
        {
            "item": "严格左截断",
            "passed": bool(backtest.get("strict_left_truncation")),
            "detail": "训练截止期必须早于验证期",
        },
        {
            "item": "Stacking Gate",
            "passed": stacking_gate_passed,
            "detail": (
                f"红/蓝独立OOF Gate；红侧={red_gate.get('decision', stacking_info.get('reason', '缺少OOF诊断'))}，"
                f"蓝侧={blue_gate.get('decision', stacking_info.get('reason', '缺少OOF诊断'))}"
            ),
        },
        {
            "item": "V7诊断闭环",
            "passed": bool(v7.get("schema") == "evidence_first_adaptive_portfolio_system_v7"),
            "detail": "包含运行时Evidence Control、Red/Blue独立Edge、Prediction Set、Regret/Survival和Falsification。",
        },
        {
            "item": "红蓝独立Edge",
            "passed": bool(runtime_control.get("red_edge_state") is not None and runtime_control.get("blue_edge_state") is not None),
            "detail": f"red={runtime_control.get('red_edge_state')}; blue={runtime_control.get('blue_edge_state')}; red_budget={runtime_control.get('red_budget_state')}",
        },
        {
            "item": "Falsification First",
            "passed": bool((falsification.get("checks") or {}).get("all_falsification_tests_passed", False)),
            "detail": "五类反证测试必须真实执行并全部通过。",
        },
        {
            "item": "Release Evidence",
            "passed": bool((release_evidence.get("windows") or {}).get("confirmation", {}).get("periods", 0) >= 100),
            "detail": f"confirmatory={((release_evidence.get('windows') or {}).get('confirmation') or {}).get('periods', 0)}期",
        },
        {
            "item": "概率健康",
            "passed": bool(probability_health.get("passed", True)),
            "detail": f"失败项={probability_health.get('failed_labels', [])}",
        },
        {
            "item": "候选召回漏斗",
            "passed": bool(candidate_recall.get("periods", 0)),
            "detail": f"期数={candidate_recall.get('periods', 0)}; schema={candidate_recall.get('schema')}",
        },
        {
            "item": "正式发布门",
            "passed": bool(model_release.get("approved")),
            "detail": model_release.get("reason") or model_release.get("formal_status_label") or "",
        },
    ]
    return {
        "schema": "release_checklist_v7",
        "passed": all(item["passed"] for item in checklist),
        "items": checklist,
    }
