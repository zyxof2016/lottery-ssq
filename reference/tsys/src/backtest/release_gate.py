from __future__ import annotations

from typing import Any


PIPELINE_NAME = "adaptive_fusion"


def _pipeline_color_result(window: dict[str, Any], color: str) -> dict[str, Any]:
    return (
        window.get("pipeline_diagnostics", {})
        .get("by_color", {})
        .get(color, {})
        .get(PIPELINE_NAME, {})
    )


def evaluate_model_release(
    backtest: dict[str, Any],
    release_config: dict[str, Any] | None = None,
    *,
    forecast_audit: dict[str, Any] | None = None,
    code_fingerprint: str = "",
) -> dict[str, Any]:
    config = release_config or {}
    minimum_periods = max(2, int(config.get("minimum_out_of_sample_periods", 200)))
    development_periods = max(2, int(config.get("development_periods", 100)))
    confirmatory_periods = max(2, int(config.get("confirmatory_periods", 100)))
    require_two_windows = bool(config.get("require_two_window_confirmation", True))
    require_frozen_confirmation = bool(
        config.get("require_frozen_confirmatory_parameters", True)
    )
    require_frozen_fusion = bool(
        config.get("freeze_confirmatory_fusion_weights", True)
    )
    require_blue_top1 = bool(config.get("require_blue_top1_evidence", True))
    minimum_blue_signal_coverage = min(
        1.0,
        max(0.0, float(config.get("minimum_blue_signal_coverage", 0.80))),
    )
    require_prospective = bool(
        config.get("require_prospective_forecast_validation", False)
    )
    minimum_prospective = max(
        1, int(config.get("minimum_prospective_validations", 30))
    )
    prospective_code_scope = str(
        config.get("prospective_code_scope", "current")
    ).strip().lower()
    if prospective_code_scope not in {"current", "any"}:
        prospective_code_scope = "current"
    allow_provisional_output = bool(
        config.get("allow_provisional_experimental_output", True)
    )
    provisional_min_periods = max(
        1, int(config.get("provisional_min_out_of_sample_periods", 60))
    )
    provisional_min_red_signal_strength = max(
        0.0, float(config.get("provisional_min_red_signal_strength", 0.03))
    )
    provisional_min_total_signal_strength = max(
        0.0, float(config.get("provisional_min_total_signal_strength", 0.12))
    )
    completed_periods = int(
        backtest.get("calibration_summary", {}).get("completed_holdout_periods", 0)
    )

    reasons: list[str] = []
    if completed_periods < minimum_periods:
        reasons.append(
            f"独立样本外验证仅 {completed_periods} 期，正式发布门槛需要 {minimum_periods} 期。"
        )

    evidence = backtest.get("release_evidence", {})
    confirmation: dict[str, Any] = {}
    development: dict[str, Any] = {}
    confirmatory_frozen: bool | None = None
    if require_two_windows:
        windows = evidence.get("windows", {})
        development = windows.get("development", {})
        confirmation = windows.get("confirmation", {})
        if not bool(evidence.get("available")):
            reasons.append("缺少相互隔离的开发窗口与最终确认窗口。")
        if int(development.get("periods", 0)) < development_periods:
            reasons.append(f"开发窗口不足 {development_periods} 期。")
        if int(confirmation.get("periods", 0)) < confirmatory_periods:
            reasons.append(f"最终确认窗口不足 {confirmatory_periods} 期。")
        confirmatory_frozen = bool(confirmation.get("hyperparameters_frozen"))
        fusion_frozen = bool(evidence.get("confirmatory_fusion_weights_frozen"))
        if require_frozen_confirmation and confirmatory_frozen is not True:
            reasons.append("最终确认窗口的超参数未全程冻结。")
        if require_frozen_fusion and fusion_frozen is not True:
            reasons.append("最终确认窗口的融合权重未全程冻结。")
    else:
        confirmation = {
            "periods": completed_periods,
            "pipeline_diagnostics": backtest.get("pipeline_diagnostics", {}),
        }

    red_confirmation = _pipeline_color_result(confirmation, "red")
    blue_confirmation = _pipeline_color_result(confirmation, "blue")
    if not red_confirmation:
        reasons.append("缺少冻结融合流程的红球确认依据。")
    elif not bool(red_confirmation.get("proven_better_than_uniform")):
        reasons.append("冻结融合流程的红球未在最终确认窗口证明优于均匀基线。")
    if not blue_confirmation:
        reasons.append("缺少冻结融合流程的蓝球确认依据。")
    elif not bool(blue_confirmation.get("proven_better_than_uniform")):
        reasons.append("冻结融合流程的蓝球未在最终确认窗口证明优于均匀基线。")
    if require_blue_top1 and blue_confirmation:
        signal_periods = int(blue_confirmation.get("signal_periods", 0))
        blue_periods = int(blue_confirmation.get("periods", 0))
        signal_coverage = signal_periods / blue_periods if blue_periods else 0.0
        if signal_coverage < minimum_blue_signal_coverage:
            reasons.append(
                f"冻结融合流程的蓝球有效信号覆盖率 {signal_coverage:.1%}，低于门槛 {minimum_blue_signal_coverage:.1%}。"
            )
        elif not bool(blue_confirmation.get("top1_proven_better_than_uniform")):
            reasons.append("冻结融合流程的蓝球首选未在有信号期证明优于均匀基线。")

    code_fp = str(code_fingerprint) if code_fingerprint is not None else ""
    prospective_records = [
        record
        for record in (forecast_audit or {}).get("records", [])
        if record.get("validation") is not None
        and record.get("prediction_mode") != "uniform_no_signal_baseline"
    ]
    prospective_targets_any_code = {
        str(record.get("target_period", ""))
        for record in prospective_records
    }
    prospective_targets_current_code = {
        str(record.get("target_period", ""))
        for record in prospective_records
        if str(record.get("code_fingerprint", "")) == code_fp
    }
    prospective_targets = (
        prospective_targets_any_code
        if prospective_code_scope == "any"
        else prospective_targets_current_code
    )
    prospective_count = len(prospective_targets)
    if require_prospective and prospective_count < minimum_prospective:
        scope_label = "全部代码版本" if prospective_code_scope == "any" else "当前代码版本"
        reasons.append(
            f"{scope_label}的开奖前独立留痕仅 {prospective_count} 个目标期，确认通过至少需要 {minimum_prospective} 个新目标期。"
        )

    final_signal_strength = backtest.get("final_signal_strength_by_color", {})
    red_signal_strength = float(final_signal_strength.get("red", 0.0))
    blue_signal_strength = float(final_signal_strength.get("blue", 0.0))
    total_signal_strength = red_signal_strength + blue_signal_strength
    current_nonbaseline = red_signal_strength > 1e-12 or blue_signal_strength > 1e-12

    provisional_reasons: list[str] = []
    if not allow_provisional_output:
        provisional_reasons.append("experimental_output_disabled")
    if completed_periods < provisional_min_periods:
        provisional_reasons.append("insufficient_provisional_holdout_periods")
    if not current_nonbaseline:
        provisional_reasons.append("uniform_baseline_only")
    elif (
        red_signal_strength < provisional_min_red_signal_strength
        and total_signal_strength < provisional_min_total_signal_strength
    ):
        provisional_reasons.append("red_signal_strength_below_provisional_threshold")

    provisional_approved = not provisional_reasons
    weak_structure_reference_ready = bool(
        allow_provisional_output
        and completed_periods >= provisional_min_periods
        and current_nonbaseline
    )
    experimental_display_ready = bool(
        current_nonbaseline and (provisional_approved or weak_structure_reference_ready)
    )

    approved = not reasons
    formal_status = "approved" if approved else "pending_confirmation"
    experimental_status = (
        "weak_structure_reference"
        if weak_structure_reference_ready and current_nonbaseline and not approved
        else (
            "display_ready"
            if provisional_approved and experimental_display_ready
            else ("baseline_only" if not current_nonbaseline else "withheld")
        )
    )

    if approved:
        display_reason = "当前正式发布门槛已通过，本期结果按正式发布输出展示。"
    elif not current_nonbaseline:
        display_reason = "当前仅保留基线实验留痕，不作为预测结果展示。"
    elif weak_structure_reference_ready:
        display_reason = "当前已有非基线弱结构，虽然正式发布门槛尚未通过，但本期最终实验输出会继续展示并单独标记。"
    elif experimental_display_ready:
        display_reason = "当前存在非基线实验输出，但仍需继续积累独立验证样本。"
    else:
        display_reason = "当前尚未形成可展示的非基线实验输出。"

    if approved:
        reason = "冻结融合流程已通过最终确认窗口，本期结果可以作为正式发布输出。"
    else:
        reason = "；".join(reasons)

    return {
        "approved": approved,
        "provisional_approved": provisional_approved,
        "weak_structure_reference_ready": weak_structure_reference_ready,
        "experimental_display_ready": experimental_display_ready,
        "formal_status": formal_status,
        "experimental_status": experimental_status,
        "minimum_out_of_sample_periods": minimum_periods,
        "completed_out_of_sample_periods": completed_periods,
        "development_periods": development_periods,
        "confirmatory_periods": confirmatory_periods,
        "allow_provisional_experimental_output": allow_provisional_output,
        "provisional_min_out_of_sample_periods": provisional_min_periods,
        "provisional_min_red_signal_strength": provisional_min_red_signal_strength,
        "provisional_min_total_signal_strength": provisional_min_total_signal_strength,
        "current_red_signal_strength": red_signal_strength,
        "current_blue_signal_strength": blue_signal_strength,
        "current_total_signal_strength": total_signal_strength,
        "two_window_confirmation_required": require_two_windows,
        "confirmatory_hyperparameters_frozen": confirmatory_frozen,
        "confirmatory_fusion_weights_frozen": bool(
            evidence.get("confirmatory_fusion_weights_frozen", False)
        ),
        "selection_basis": "开发窗口确定冻结的分色自适应融合规则；当前输出用全部已知历史数据重训生成。",
        "confirmation_basis": "最终确认窗口只校验冻结后的融合流程，不逐个要求基础模型都获胜。",
        "development_pipeline_available": bool(
            _pipeline_color_result(development, "red")
            and _pipeline_color_result(development, "blue")
        ),
        "confirmation_pipeline": {
            "name": PIPELINE_NAME,
            "red": red_confirmation,
            "blue": blue_confirmation,
        },
        "require_blue_top1_evidence": require_blue_top1,
        "prospective_validation_required": require_prospective,
        "minimum_prospective_validations": minimum_prospective,
        "completed_prospective_validations": prospective_count,
        "prospective_code_scope": prospective_code_scope,
        "completed_prospective_validations_current_code": len(prospective_targets_current_code),
        "completed_prospective_validations_any_code": len(prospective_targets_any_code),
        "minimum_blue_signal_coverage": minimum_blue_signal_coverage,
        "blue_signal_coverage": (
            int(blue_confirmation.get("signal_periods", 0))
            / int(blue_confirmation.get("periods", 0))
            if int(blue_confirmation.get("periods", 0)) > 0
            else 0.0
        ),
        "provisional_reason": (
            "current_best_experimental_output_ready"
            if provisional_approved
            else ";".join(provisional_reasons)
        ),
        "reference_reason": (
            "current_weak_structure_reference_ready"
            if weak_structure_reference_ready and not provisional_approved
            else (
                "current_best_experimental_output_ready"
                if provisional_approved
                else ";".join(provisional_reasons)
            )
        ),
        "formal_status_label": (
            "正式发布已通过" if approved else "正式发布待确认"
        ),
        "experimental_status_label": (
            "可参考弱结构"
            if experimental_status == "weak_structure_reference"
            else (
                "实验结果可展示"
                if experimental_display_ready
                else ("仅保留基线留痕" if not current_nonbaseline else "实验结果暂不展示")
            )
        ),
        "display_reason": display_reason,
        "reason": reason,
    }
