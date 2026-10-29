from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.report.export import export_experiment_report
from src.report.notification import send_result_email
from src.report.trend_analysis import analyze_number_trends
from src.report.visual import plot_hit_distribution, plot_number_trend, plot_performance_comparison
from src.utils.config_utils import redact_sensitive_config


def _extract_first_dict_literal(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            quote = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_backtest_summary_from_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    last: dict[str, Any] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "BACKTEST_SUMMARY" not in line or "BACKTEST_SUMMARY_PARSE_FAILED" in line:
                continue
            marker = line.find("BACKTEST_SUMMARY")
            literal = _extract_first_dict_literal(line[marker:])
            if not literal:
                continue
            try:
                parsed = ast.literal_eval(literal)
            except (SyntaxError, ValueError):
                logging.warning("无法解析回测汇总行 | 事件=BACKTEST_SUMMARY_PARSE_FAILED")
                continue
            if isinstance(parsed, dict):
                last = parsed
    return last


def build_period_records_from_validation_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Rebuild lightweight per-period method metrics for plots from the JSONL trace."""
    records: list[dict[str, Any]] = []
    if not trace_path.exists():
        return records
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "validation_period_trace":
                continue
            methods: dict[str, Any] = {}
            for method, tickets in (payload.get("method_recommendations") or {}).items():
                ticket_list = list(tickets or [])
                if not ticket_list:
                    continue
                red_hits = [float(len(ticket.get("matched_red") or [])) for ticket in ticket_list]
                blue_hits = [1.0 if ticket.get("blue_hit") else 0.0 for ticket in ticket_list]
                average_red = float(sum(red_hits) / len(red_hits))
                model_metrics = {
                    "average_red_hits": average_red,
                    "best_red_hits": float(max(red_hits)),
                    "red_ge3_ratio": float(sum(value >= 3 for value in red_hits) / len(red_hits)),
                    "red_ge4_ratio": float(sum(value >= 4 for value in red_hits) / len(red_hits)),
                    "any_red_ge3": float(any(value >= 3 for value in red_hits)),
                    "any_red_ge4": float(any(value >= 4 for value in red_hits)),
                    "blue_hit_ratio": float(sum(blue_hits) / len(blue_hits)),
                    "any_blue_hit": float(any(blue_hits)),
                }
                methods[str(method)] = {
                    "model": model_metrics,
                    "matched_no_signal": {
                        **model_metrics,
                        "average_red_hits": 36.0 / 33.0,
                        "best_red_hits": model_metrics["best_red_hits"],
                    },
                    "random": {
                        "average_red_hits": 36.0 / 33.0,
                        "best_red_hits": 36.0 / 33.0,
                        "red_ge3_ratio": 0.0,
                        "red_ge4_ratio": 0.0,
                        "any_red_ge3": 0.0,
                        "any_red_ge4": 0.0,
                        "blue_hit_ratio": 1.0 / 16.0,
                        "any_blue_hit": 0.0,
                    },
                }
            dantuo = payload.get("dantuo_actual_summary") or {}
            if "consensus" not in methods and dantuo:
                red_hit = float(dantuo.get("red_hit_count", 0) or 0)
                blue_hit = 1.0 if dantuo.get("blue_hit") else 0.0
                methods["consensus"] = {
                    "model": {
                        "average_red_hits": red_hit,
                        "best_red_hits": red_hit,
                        "red_ge3_ratio": float(red_hit >= 3),
                        "red_ge4_ratio": float(red_hit >= 4),
                        "any_red_ge3": float(red_hit >= 3),
                        "any_red_ge4": float(red_hit >= 4),
                        "blue_hit_ratio": blue_hit,
                        "any_blue_hit": blue_hit,
                    },
                    "matched_no_signal": {
                        "average_red_hits": 36.0 / 33.0,
                        "best_red_hits": 36.0 / 33.0,
                        "red_ge3_ratio": 0.0,
                        "red_ge4_ratio": 0.0,
                        "any_red_ge3": 0.0,
                        "any_red_ge4": 0.0,
                        "blue_hit_ratio": 1.0 / 16.0,
                        "any_blue_hit": 0.0,
                    },
                    "random": {
                        "average_red_hits": 36.0 / 33.0,
                        "best_red_hits": 36.0 / 33.0,
                        "red_ge3_ratio": 0.0,
                        "red_ge4_ratio": 0.0,
                        "any_red_ge3": 0.0,
                        "any_red_ge4": 0.0,
                        "blue_hit_ratio": 1.0 / 16.0,
                        "any_blue_hit": 0.0,
                    },
                }
            records.append(
                {
                    "period": str(payload.get("validation_period")),
                    "validation_period": str(payload.get("validation_period")),
                    "training_through_period": str(payload.get("training_through_period")),
                    "actual": payload.get("actual") or {},
                    "methods": methods,
                    "dantuo_actual_summary": dantuo,
                }
            )
    return records


def build_report_from_artifacts(
    *,
    config: dict[str, Any],
    project_root: Path,
    project_path_fn,
    load_validation_trace_view_fn,
) -> dict[str, Any]:
    frame = pd.read_pickle(project_path_fn(config, "clean_data"))
    forecast_path = project_path_fn(config, "forecast_audit")
    forecast_payload = json.loads(forecast_path.read_text(encoding="utf-8")) if forecast_path.exists() else {"forecasts": []}
    forecasts = list(forecast_payload.get("forecasts") or [])
    if not forecasts:
        raise FileNotFoundError(f"缺少开奖前留痕，无法收尾: {forecast_path}")
    archived = dict(forecasts[-1])
    metadata = archived.get("metadata") or {}
    archived["dantuo_plan"] = metadata.get("dantuo_plan") or archived.get("dantuo_plan")
    archived["core_numbers"] = metadata.get("core_numbers") or archived.get("core_numbers")
    archived["forecast_publish_gate"] = metadata.get("forecast_publish_gate") or archived.get("forecast_publish_gate")

    summary = parse_backtest_summary_from_log(Path(config["paths"]["logs_dir"]) / "pipeline.log")
    if not summary:
        raise RuntimeError("pipeline.log 中找不到 BACKTEST_SUMMARY，无法沿用本轮回测结果")
    period_records = build_period_records_from_validation_trace(project_path_fn(config, "validation_trace"))
    if not period_records:
        raise RuntimeError("validation_trace.jsonl 中没有逐期记录，无法收尾")
    common_methods = set(summary)
    for record in period_records:
        common_methods &= set((record.get("methods") or {}).keys())
    if not common_methods:
        # Fall back to consensus-only plotting when method tickets are incomplete.
        common_methods = {"consensus"} if all("consensus" in (record.get("methods") or {}) for record in period_records) else set()
    summary = {key: value for key, value in summary.items() if key in common_methods} or summary

    tickets = list(archived.get("tickets") or [])
    manifest = {}
    try:
        from src.data_handler.versioning import load_data_manifest

        manifest = load_data_manifest(project_path_fn(config, "data_manifest")) or {}
    except Exception:
        manifest = {}
    backtest = {
        "strict_left_truncation": True,
        "random_repeats": int(config["backtest"].get("random_repeats", 0)),
        "period_records": period_records,
        "summary": summary,
        "final_weights_by_color": archived.get("weights_by_color") or {},
        "final_weight_reason_by_color": {"red": "finish_from_artifacts", "blue": "finish_from_artifacts"},
        "final_parameters": {},
        "research_benchmark": {"schema": "tsys_research_benchmark_v1", "available": False, "reason": "finish_from_artifacts"},
        "v7_diagnostics": {},
        "recovered_from_artifacts": True,
    }
    validation_trace = load_validation_trace_view_fn(project_path_fn(config, "validation_trace"), config)
    # Keep only metadata in the in-memory report; rows reload live in dashboard_status.
    validation_trace_meta = {
        "path": validation_trace.get("path"),
        "exists": validation_trace.get("exists"),
        "periods": validation_trace.get("periods"),
        "run_start": validation_trace.get("run_start"),
        "run_complete": validation_trace.get("run_complete"),
        "rows_deferred": True,
    }
    trend_analysis = analyze_number_trends(frame, config.get("trend", {}))
    weights = archived.get("weights_by_color") or {}
    candidate_section = {
        "consensus": {
            "selected": tickets,
            "display_selected": tickets,
            "experimental_selected": tickets,
            "dantuo_plan": archived.get("dantuo_plan"),
        },
        "dantuo_plan": archived.get("dantuo_plan"),
        "core_numbers": archived.get("core_numbers"),
        "weights": weights.get("red") or weights,
        "weights_by_color": weights,
        "model_pipeline": {"fusion": {"prediction_mode": archived.get("prediction_mode")}},
        "recovered_from_artifacts": True,
    }
    report = {
        "project": config["project"]["name"],
        "run_metadata": {
            "generated_at_utc": datetime_utcnow(),
            "recovery_mode": "finish_from_artifacts",
            "note": "回测已完成但报告导出内存溢出；本报告由 validation_trace + forecast_audit + BACKTEST_SUMMARY 补完。",
        },
        "run_manifest": {"recovered_from_artifacts": True},
        "config_snapshot": redact_sensitive_config(config),
        "data": {
            "periods": len(frame),
            "first_period": str(frame.iloc[0]["period"]),
            "latest_period": str(frame.iloc[-1]["period"]),
            "cleaning": (manifest.get("cleaning") if isinstance(manifest, dict) else {}) or {},
        },
        "randomness_audit": {"enabled": False, "skipped": True},
        "candidate_experiment": candidate_section,
        "backtest": backtest,
        "red_adverse_guard": {"active": False, "recovered_from_artifacts": True},
        "out_of_sample_forecast_audit": forecast_payload,
        "forecast_archive": archived,
        "trend_analysis": trend_analysis,
        "validation_trace": validation_trace_meta,
        "mathematical_baseline": {},
        "blue_evidence": {},
        "multi_period_strategy": {},
        "research_benchmark": backtest["research_benchmark"],
        "execution_summary": {},
        "performance_profile": {},
        "model_release": {
            "approved": bool((archived.get("forecast_publish_gate") or {}).get("formal_release_approved")),
            "reason": "finish_from_artifacts",
        },
        "v7_diagnostics": {},
        "random_baseline": {
            "candidate_coverage": {},
            "backtest_random_repeats": backtest["random_repeats"],
        },
        "compliance": {
            "enable_export_bet": bool(config["compliance"].get("enable_export_bet", False)),
            "display_candidates_only": True,
            "watermark": config["compliance"].get("watermark"),
        },
    }
    # Attach live rows only for return value used by email/dashboard session; export will slim again.
    report["validation_trace"] = validation_trace
    return report


def datetime_utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def finish_export_and_notify(
    *,
    config: dict[str, Any],
    project_root: Path,
    project_path_fn,
    load_validation_trace_view_fn,
    dashboard_url: str | None = None,
) -> dict[str, Any]:
    report = build_report_from_artifacts(
        config=config,
        project_root=project_root,
        project_path_fn=project_path_fn,
        load_validation_trace_view_fn=load_validation_trace_view_fn,
    )
    frame = pd.read_pickle(project_path_fn(config, "clean_data"))
    backtest = report["backtest"]
    export_experiment_report(report, project_path_fn(config, "report_json"))
    try:
        plot_performance_comparison(backtest, project_path_fn(config, "performance_plot"))
        plot_hit_distribution(backtest, project_path_fn(config, "distribution_plot"))
        plot_number_trend(
            frame,
            project_path_fn(config, "trend_plot"),
            display_periods=int((report.get("trend_analysis") or {}).get("display_periods", 60)),
        )
    except Exception:
        logging.exception("收尾绘图失败 | 事件=FINISH_EXPORT_PLOT_FAILED")
    try:
        send_result_email(
            report,
            config,
            project_root=project_root,
            dashboard_url=dashboard_url,
            delay_seconds=0.5,
        )
        logging.info("收尾邮件已发送 | 事件=FINISH_EXPORT_EMAIL_SENT")
    except Exception:
        logging.exception("收尾邮件发送失败 | 事件=FINISH_EXPORT_EMAIL_FAILED")
    logging.info(
        "收尾行完成 | 事件=FINISH_EXPORT_DONE | 报告=%s | 验证期数=%s | 预测=%s",
        project_path_fn(config, "report_json"),
        len(backtest.get("period_records") or []),
        (report.get("forecast_archive") or {}).get("forecast_id"),
    )
    return report
