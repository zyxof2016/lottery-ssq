from __future__ import annotations

from typing import Any


def v7_release_gate(report: dict[str, Any]) -> dict[str, Any]:
    backtest = report.get("backtest", {}) if isinstance(report, dict) else {}
    v7 = report.get("v7_diagnostics") if isinstance(report, dict) else {}
    if not v7 and isinstance(backtest, dict):
        v7 = backtest.get("v7_diagnostics", {})
    model_release = report.get("model_release", {}) if isinstance(report, dict) else {}
    evidence = v7.get("evidence_report", {}) if isinstance(v7, dict) else {}
    best = evidence.get("best_score") or {}
    cards = evidence.get("evidence_cards") or []
    trusted_cards = [card for card in cards if isinstance(card, dict) and card.get("decision") == "TRUST"]
    release = backtest.get("release_evidence", {}) if isinstance(backtest, dict) else {}
    confirmation = (release.get("windows") or {}).get("confirmation") or {}
    runtime_control = v7.get("runtime_control", {}) if isinstance(v7, dict) else {}
    falsification = v7.get("falsification_report", {}) if isinstance(v7, dict) else {}
    checks = [
        {
            "item": "strict_left_truncation",
            "passed": bool(backtest.get("strict_left_truncation")),
            "detail": "验证期只允许使用目标期之前的历史数据。",
        },
        {
            "item": "nested_oos_or_confirmatory",
            "passed": bool(confirmation.get("periods", 0) >= 100 and confirmation.get("hyperparameters_frozen")),
            "detail": f"confirmatory_periods={confirmation.get('periods')}; frozen={confirmation.get('hyperparameters_frozen')}",
        },
        {
            "item": "matched_random_positive_edge",
            "passed": bool(trusted_cards or ((best.get("score") or 0.0) > 0 and best.get("decision") == "TRUST")),
            "detail": f"trusted_cards={len(trusted_cards)}; best={evidence.get('best_method')}; score={best.get('score')}; confidence={best.get('confidence')}; NO_EDGE only permits conservative/shadow output",
        },
        {
            "item": "red_blue_independent_edge",
            "passed": bool(runtime_control.get("red_edge_state") is not None and runtime_control.get("blue_edge_state") is not None),
            "detail": f"red={runtime_control.get('red_edge_state')}; blue={runtime_control.get('blue_edge_state')}; red_budget={runtime_control.get('red_budget_state')}",
        },
        {
            "item": "falsification_first",
            "passed": bool((falsification.get("checks") or {}).get("all_falsification_tests_passed", False)),
            "detail": "严格左截断、标签置换、随机特征、无信号合成、时间反转/随机种子稳定性需全部通过。",
        },
        {
            "item": "formal_model_release",
            "passed": bool(model_release.get("approved")),
            "detail": model_release.get("reason", ""),
        },
    ]
    approved = all(item["passed"] for item in checks)
    return {
        "schema": "release_gate_v7",
        "approved": approved,
        "decision": "production" if approved else "shadow_or_reference_only",
        "checks": checks,
        "rule": "无证据可输出 NO_EDGE/保守覆盖；只有确认窗、匹配随机和门禁全部通过才进入生产。",
    }
