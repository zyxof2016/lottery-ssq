from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import queue
import re
from logging.handlers import RotatingFileHandler
import platform
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if sys.path[:1] != [str(ROOT)]:
    sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
    "scipy": "scipy",
    "openpyxl": "openpyxl",
    "pulp": "PuLP",
}
missing = [distribution for module, distribution in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "当前 Python 解释器缺少依赖: " + ", ".join(missing)
        + f"\n请在 PyCharm 中选择 {ROOT / '.venv' / 'Scripts' / 'python.exe'}，"
        + "或执行 python -m pip install -r requirements.txt"
    )

import numpy as np
import pandas as pd
import yaml

from compliance.risk_notice import append_audit_log, require_random_baseline, show_risk_notice
from src.backtest.adaptive_calibration import apply_parameter_snapshot, parameter_snapshot
from src.backtest.metrics import theoretical_ticket_baseline
from src.backtest.release_gate import evaluate_model_release
from src.backtest.walk_forward import run_walk_forward
from src.blue.probability_shrinkage import shrink_blue_probability_to_uniform
from src.blue.specialist_gateway import BLUE_SPECIALIST_VARIANTS, finalize_blue_portfolio
from src.combination_gen.monte_carlo import (
    generate_candidates,
    generate_core_candidates,
    generate_uniform_tickets,
)
from src.combination_gen.dantuo import build_dantuo_plan
from src.combination_gen.hit_coherence import apply_hit_coherence_final5, red_draws_from_frame
from src.combination_gen.optimizer import (
    _ensure_minimum_blue_diversity,
    adverse_guard_coverage_optimize,
    consensus_optimize,
    optimize_candidates,
    rotational_coverage_optimize,
)
from src.portfolio.coverage import coverage_summary
from src.red.core_numbers import compute_core_numbers
from src.data_handler.cleaner import assert_left_truncated, clean_history, save_clean_data
from src.data_handler.loader import load_csv
from src.data_handler.randomness_audit import run_randomness_audit
from src.data_handler.versioning import (
    create_data_manifest,
    load_data_manifest,
    source_tree_fingerprint,
    write_data_manifest,
)
from src.utils.config_utils import (
    apply_runtime_profile,
    config_fingerprint_payload,
    deep_update,
    redact_sensitive_config,
    runtime_profile_options,
)
from src.portfolio.final_rebalance import rebalance_final_portfolio
from src.candidate.exclusion_filter import apply_low_confidence_exclusion
from src.candidate.recall_protection import (
    apply_red_candidate_recall_protection,
    enforce_red_candidate_recall_slots,
)
from src.fusion.model_fusion import build_models, fuse_predictions, rank_prediction
from src.governance.pipeline_guards import (
    merge_candidate_pools as _merge_candidate_pools,
    red_guard_stoploss_decision as _red_guard_stoploss_decision,
    temper_with_uniform as _temper_with_uniform,
    uniform_probabilities as _uniform_probabilities,
)
from src.report.export import archive_forecast, export_experiment_report, reconcile_forecast_audit
from src.report.finish_export import finish_export_and_notify
from src.report.audit_key import load_or_create_audit_key
from src.report.log_summary import summarize_pipeline_log
from src.utils.retry_utils import retry_call
from src.portfolio.rolling_strategy import apply_rolling_portfolio_memory, summarize_multi_period_strategy
from src.utils.seed_offsets import seed_with_offset
from src.fusion.stacking_meta import blend_with_stacking, stacking_meta_predict
from src.report.notification import send_result_email
from src.evidence.runtime_control import apply_evidence_weight_control, build_evidence_runtime_control


from src.report.dashboard import serve_dashboard
from src.report.blue_evidence import summarize_blue_evidence
from src.report.trend_analysis import analyze_number_trends
from src.report.visual import plot_hit_distribution, plot_number_trend, plot_performance_comparison
from src.governance.manifest import build_run_manifest
from src.governance.probability import probability_health_report
from src.governance.release_checklist import release_checklist
from src.governance.release_gate import v7_release_gate
from src.governance.final_v7_guard import apply_final_v7_guard
from src.red.portfolio import select_red_portfolio
from src.runtime.fingerprint import prediction_fingerprint, stable_hash
from src.runtime.logging_compact import compact_sampling_log
from src.runtime.performance import RUNTIME_PROFILER


class WindowsSafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe log rotation for long dashboard/CLI runs.

    If another process or dashboard thread holds pipeline.log, Windows can
    reject the rename used by RotatingFileHandler.  Logging must not interrupt
    or spam stderr during a 300-period walk-forward run, so rollover degrades to
    plain append for the current process.
    """

    def doRollover(self) -> None:  # noqa: N802 - stdlib API name
        try:
            super().doRollover()
        except PermissionError:
            self.maxBytes = 0
            if self.stream is None:
                self.stream = self._open()
            try:
                self.stream.write(
                    "日志轮转被Windows文件占用阻止，本次运行已降级为追加写入，主流程继续。\n"
                )
                self.stream.flush()
            except Exception:
                pass


class NonBlockingStreamHandler(logging.StreamHandler):
    """Console logging that cannot freeze the experiment thread.

    Windows stops console writes while the screen is locked or while Quick Edit
    has a text selection. A normal StreamHandler holds the logging lock during
    that write, so the whole backtest waits. This handler formats on the caller
    and lets a daemon thread perform the write.
    """

    def __init__(self, stream=None) -> None:
        super().__init__(stream)
        self._queue: queue.Queue[str] = queue.Queue(maxsize=2000)
        self._worker = threading.Thread(
            target=self._drain,
            name="tsys-log-console",
            daemon=True,
        )
        self._worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            return

    def _drain(self) -> None:
        while True:
            message = self._queue.get()
            try:
                self.stream.write(message + self.terminator)
                self.flush()
            except Exception:
                pass


def _enable_console_scroll_and_selection() -> None:
    """让控制台自己处理鼠标选中和滚轮滚动。

    选中文字时，Windows 会暂停往这个窗口写日志。实验日志走
    NonBlockingStreamHandler，暂停只落在后台写窗口的线程上，回测继续跑。
    """

    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        input_handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(input_handle, ctypes.byref(mode)):
            quick_edit = 0x0040
            extended_flags = 0x0080
            mouse_input = 0x0010
            kernel32.SetConsoleMode(
                input_handle,
                (mode.value | extended_flags | quick_edit) & ~mouse_input,
            )

        class _Coord(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class _SmallRect(ctypes.Structure):
            _fields_ = [
                ("Left", ctypes.c_short),
                ("Top", ctypes.c_short),
                ("Right", ctypes.c_short),
                ("Bottom", ctypes.c_short),
            ]

        class _BufferInfo(ctypes.Structure):
            _fields_ = [
                ("dwSize", _Coord),
                ("dwCursorPosition", _Coord),
                ("wAttributes", ctypes.c_ushort),
                ("srWindow", _SmallRect),
                ("dwMaximumWindowSize", _Coord),
            ]

        output_handle = kernel32.GetStdHandle(-11)
        info = _BufferInfo()
        if not kernel32.GetConsoleScreenBufferInfo(output_handle, ctypes.byref(info)):
            return
        width = max(int(info.dwSize.X), 80)
        target_height = 8000
        if int(info.dwSize.Y) >= target_height:
            return
        kernel32.SetConsoleScreenBufferSize.argtypes = [ctypes.c_void_p, _Coord]
        kernel32.SetConsoleScreenBufferSize.restype = ctypes.c_int
        kernel32.SetConsoleScreenBufferSize(output_handle, _Coord(width, target_height))
    except Exception:
        logging.debug("恢复控制台选中和滚动失败", exc_info=True)


STEP_DESCRIPTIONS = {
    "data_load": "读取原始历史数据",
    "data_clean": "清洗、排序并执行号码合法性校验",
    "randomness_audit": "执行随机性与数据质量审计",
    "walk_forward_backtest": "执行严格左截断的逐期回测",
    "current_candidate_experiment": "使用全部已知历史数据生成下一期实验候选",
    "current_model_fit": "训练单个概率模型并生成号码评分",
    "monte_carlo_generation": "按融合分布生成蒙特卡洛候选池",
    "coverage_optimization": "筛选低重叠、高覆盖组合",
    "consensus_selection": "综合候选质量和两类优化器结果筛选3注",
    "forecast_reconciliation": "使用新到期开奖核验此前留存的开奖前预测",
    "forecast_archive": "固化下一期开奖前预测以供独立验证",
    "report_export": "导出实验报告和统计图",
}
MODEL_LABELS = {
    "frequency": "频率模型",
    "time_decay": "时间衰减模型",
    "bayesian": "贝叶斯模型",
    "logistic_regression": "逻辑回归模型",
    "xgboost": "XGBoost模型",
    "lightgbm_ranker": "LightGBM排序模型",
    "mlp": "浅层MLP模型",
    "catboost": "CatBoost模型",
    "pattern_memory": "形态记忆模型",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else ROOT / "config" / "config.yaml"
    if not source.is_absolute():
        source = ROOT / source
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fragments = config.get("config_fragments", {}) if isinstance(config, dict) else {}
    if bool(fragments.get("enabled", False)):
        for item in fragments.get("files", []):
            fragment_path = Path(str(item))
            if not fragment_path.is_absolute():
                fragment_path = source.parent / fragment_path
            if not fragment_path.exists():
                raise ValueError(f"配置片段不存在: {fragment_path}")
            with fragment_path.open("r", encoding="utf-8") as fragment_handle:
                fragment_config = yaml.safe_load(fragment_handle) or {}
            if not isinstance(fragment_config, dict):
                raise ValueError(f"配置片段必须是YAML字典: {fragment_path}")
            deep_update(config, fragment_config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    if logging_config and str(logging_config.get("level", "INFO")).strip().upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ValueError("logging.level 仅支持 DEBUG、INFO、WARNING、ERROR、CRITICAL")
    if int(logging_config.get("max_bytes", 10 * 1024 * 1024)) < 1024 * 1024:
        raise ValueError("logging.max_bytes 必须至少为 1048576")
    if int(logging_config.get("backup_count", 5)) < 1:
        raise ValueError("logging.backup_count 必须至少为1")
    required_sections = {
        "project", "paths", "data", "randomness_audit", "features", "models", "fusion",
        "generation", "scoring", "optimizer", "backtest", "calibration", "release",
        "compliance", "report",
    }
    missing = sorted(required_sections - set(config or {}))
    if missing:
        raise ValueError(f"配置缺少必要分区: {missing}")
    windows = [int(value) for value in config["features"]["windows"]]
    if not windows or windows != sorted(set(windows)) or min(windows) <= 0:
        raise ValueError("features.windows 必须是严格递增的正整数")
    decay = float(config["features"]["decay_factor"])
    if not 0.0 < decay <= 1.0:
        raise ValueError("features.decay_factor 必须在 (0, 1] 范围内")
    if int(config["features"].get("feature_cache_max_entries", 24)) < 0:
        raise ValueError("features.feature_cache_max_entries 不能为负数")
    if int(config["features"].get("feature_selection_max_features_by_color", {}).get("red", 1)) <= 0:
        raise ValueError("features.feature_selection_max_features_by_color.red 必须大于0")
    if int(config["features"].get("feature_selection_max_features_by_color", {}).get("blue", 1)) <= 0:
        raise ValueError("features.feature_selection_max_features_by_color.blue 必须大于0")
    if float(config["features"].get("feature_selection_min_variance", 1e-10)) < 0:
        raise ValueError("features.feature_selection_min_variance 不能为负数")
    exclusion = config.get("exclusion_filter", {})
    if int(exclusion.get("red_max_excluded", 0)) < 0 or int(exclusion.get("red_max_excluded", 0)) > 9:
        raise ValueError("exclusion_filter.red_max_excluded 必须在 0-9 之间")
    if int(exclusion.get("red_min_remaining", 24)) < 18:
        raise ValueError("exclusion_filter.red_min_remaining 至少为18")
    if int(exclusion.get("blue_max_excluded", 0)) < 0 or int(exclusion.get("blue_max_excluded", 0)) > 4:
        raise ValueError("exclusion_filter.blue_max_excluded 必须在 0-4 之间")
    if int(exclusion.get("blue_min_remaining", 13)) < 10:
        raise ValueError("exclusion_filter.blue_min_remaining 至少为10")
    for key in ("red_score_quantile", "blue_score_quantile"):
        if key in exclusion and not 0.0 <= float(exclusion.get(key)) <= 1.0:
            raise ValueError(f"exclusion_filter.{key} 必须在 [0, 1] 范围内")
    scoring = config["scoring"]
    if any(key in scoring for key in ("red_number_weight", "blue_number_weight")):
        component_weights = [
            float(scoring.get("red_number_weight", 0.0)),
            float(scoring.get("blue_number_weight", 0.0)),
            float(scoring.get("structure_weight", 0.0)),
        ]
        if any(value < 0 for value in component_weights) or not np.isclose(sum(component_weights), 1.0):
            raise ValueError("scoring.red_number_weight、blue_number_weight 与 structure_weight 必须非负且合计为1")
    else:
        number_weight = float(scoring["number_weight"])
        structure_weight = float(scoring["structure_weight"])
        if number_weight < 0 or structure_weight < 0 or not np.isclose(number_weight + structure_weight, 1.0):
            raise ValueError("scoring.number_weight 与 structure_weight 必须非负且合计为1")
    consensus_weights = [
        float(config["scoring"].get("consensus_quality_weight", 0.80)),
        float(config["scoring"].get("consensus_support_weight", 0.20)),
        float(config["scoring"].get("consensus_structure_weight", 0.0)),
    ]
    if any(value < 0 for value in consensus_weights) or not np.isclose(sum(consensus_weights), 1.0):
        raise ValueError("综合筛选的质量、算法支持和结构权重必须非负且合计为1")
    if float(config["scoring"].get("portfolio_subset_weight", 0.20)) < 0:
        raise ValueError("scoring.portfolio_subset_weight 不能为负数")
    warmup_periods = int(config["fusion"].get("warmup_periods", 0))
    minimum_observations = int(config["fusion"]["minimum_observations"])
    validation_periods = int(config["fusion"]["validation_periods"])
    if warmup_periods < minimum_observations or minimum_observations < 2:
        raise ValueError("fusion.warmup_periods 必须不少于 minimum_observations，且至少为2")
    if validation_periods < minimum_observations:
        raise ValueError("fusion.validation_periods 必须不少于 minimum_observations")
    skill_windows = [int(value) for value in config["fusion"].get("skill_windows", [validation_periods])]
    if not skill_windows or skill_windows != sorted(set(skill_windows)) or min(skill_windows) < minimum_observations:
        raise ValueError("fusion.skill_windows 必须严格递增，且每个窗口不少于 minimum_observations")
    weak_signal_min_windows = int(config["fusion"].get("weak_signal_min_windows", 2))
    if not 1 <= weak_signal_min_windows <= len(skill_windows):
        raise ValueError("fusion.weak_signal_min_windows 必须在 skill_windows 数量范围内")
    weak_signal_max_strength = float(config["fusion"].get("weak_signal_max_strength", 0.25))
    if not 0.0 <= weak_signal_max_strength <= 1.0:
        raise ValueError("fusion.weak_signal_max_strength 必须在 [0, 1] 范围内")
    weak_signal_min_windows_by_color = config["fusion"].get("weak_signal_min_windows_by_color", {})
    for color, value in weak_signal_min_windows_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.weak_signal_min_windows_by_color 仅支持 red 和 blue")
        color_value = int(value)
        if not 1 <= color_value <= len(skill_windows):
            raise ValueError("fusion.weak_signal_min_windows_by_color.* 必须在 skill_windows 数量范围内")
    weak_signal_max_strength_by_color = config["fusion"].get("weak_signal_max_strength_by_color", {})
    for color, value in weak_signal_max_strength_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.weak_signal_max_strength_by_color 仅支持 red 和 blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.weak_signal_max_strength_by_color.* 必须在 [0, 1] 范围内")
    weak_signal_min_strength = float(config["fusion"].get("weak_signal_min_strength", 0.0))
    if not 0.0 <= weak_signal_min_strength <= 1.0:
        raise ValueError("fusion.weak_signal_min_strength 必须在 [0, 1] 范围内")
    weak_signal_min_strength_by_color = config["fusion"].get("weak_signal_min_strength_by_color", {})
    for color, value in weak_signal_min_strength_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.weak_signal_min_strength_by_color 仅支持 red 和 blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.weak_signal_min_strength_by_color.* 必须在 [0, 1] 范围内")
    max_uniform_share_if_signal = float(config["fusion"].get("max_uniform_share_if_signal", 1.0))
    if not 0.0 <= max_uniform_share_if_signal <= 1.0:
        raise ValueError("fusion.max_uniform_share_if_signal must be in [0, 1]")
    max_uniform_share_if_signal_by_color = config["fusion"].get(
        "max_uniform_share_if_signal_by_color", {}
    )
    for color, value in max_uniform_share_if_signal_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.max_uniform_share_if_signal_by_color only supports red and blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.max_uniform_share_if_signal_by_color.* must be in [0, 1]")
    relative_tilt_max_strength = float(config["fusion"].get("relative_tilt_max_strength", 0.0))
    if not 0.0 <= relative_tilt_max_strength <= 1.0:
        raise ValueError("fusion.relative_tilt_max_strength 必须在 [0, 1] 范围内")
    relative_tilt_max_strength_by_color = config["fusion"].get("relative_tilt_max_strength_by_color", {})
    for color, value in relative_tilt_max_strength_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.relative_tilt_max_strength_by_color 仅支持 red 和 blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.relative_tilt_max_strength_by_color.* 必须在 [0, 1] 范围内")
    relative_tilt_min_strength = float(config["fusion"].get("relative_tilt_min_strength", 0.0))
    if not 0.0 <= relative_tilt_min_strength <= 1.0:
        raise ValueError("fusion.relative_tilt_min_strength 必须在 [0, 1] 范围内")
    relative_tilt_min_strength_by_color = config["fusion"].get("relative_tilt_min_strength_by_color", {})
    for color, value in relative_tilt_min_strength_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.relative_tilt_min_strength_by_color 仅支持 red 和 blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.relative_tilt_min_strength_by_color.* 必须在 [0, 1] 范围内")
    relative_tilt_consistency_boost = float(config["fusion"].get("relative_tilt_consistency_boost", 0.0))
    if not 0.0 <= relative_tilt_consistency_boost <= 1.0:
        raise ValueError("fusion.relative_tilt_consistency_boost must be in [0, 1]")
    relative_tilt_consistency_boost_by_color = config["fusion"].get(
        "relative_tilt_consistency_boost_by_color", {}
    )
    for color, value in relative_tilt_consistency_boost_by_color.items():
        if color not in {"red", "blue"}:
            raise ValueError("fusion.relative_tilt_consistency_boost_by_color only supports red and blue")
        color_value = float(value)
        if not 0.0 <= color_value <= 1.0:
            raise ValueError("fusion.relative_tilt_consistency_boost_by_color.* must be in [0, 1]")
    rank_temperature = float(config["models"]["xgboost"].get("rank_temperature", 2.0))
    if rank_temperature <= 0:
        raise ValueError("models.xgboost.rank_temperature 必须大于0")
    if int(config["models"].get("fit_max_retries", 2)) < 1:
        raise ValueError("models.fit_max_retries 必须至少为1")
    if float(config["models"]["logistic_regression"].get("blue_c", 0.05)) <= 0:
        raise ValueError("models.logistic_regression.blue_c 必须大于0")
    mlp_config = config["models"].get("mlp", {})
    if mlp_config:
        if "enabled" in mlp_config and not isinstance(mlp_config.get("enabled"), bool):
            raise ValueError("models.mlp.enabled 必须为布尔值")
        if int(mlp_config.get("max_iter", 80)) < 10:
            raise ValueError("models.mlp.max_iter 必须至少为10")
        hidden = mlp_config.get("hidden_layer_sizes", [64, 32])
        if not isinstance(hidden, list) or not hidden or any(int(value) <= 0 for value in hidden):
            raise ValueError("models.mlp.hidden_layer_sizes 必须是正整数列表")
    catboost_config = config["models"].get("catboost", {})
    if catboost_config:
        if "enabled" in catboost_config and not isinstance(catboost_config.get("enabled"), bool):
            raise ValueError("models.catboost.enabled 必须为布尔值")
        if int(catboost_config.get("iterations", 90)) < 10:
            raise ValueError("models.catboost.iterations 必须至少为10")
        if int(catboost_config.get("depth", 5)) < 1:
            raise ValueError("models.catboost.depth 必须至少为1")
    for model_name in ("xgboost", "lightgbm_ranker"):
        if int(config["models"].get(model_name, {}).get("n_jobs", 1)) < 1:
            raise ValueError(f"models.{model_name}.n_jobs 必须至少为1")
    samples = int(config["generation"]["monte_carlo_samples"])
    pool_size = int(config["generation"]["candidate_pool_size"])
    output_count = int(config["generation"]["output_count"])
    consensus_count = int(config["generation"]["consensus_output_count"])
    if not 1 <= consensus_count <= output_count <= pool_size <= samples:
        raise ValueError("生成规模必须满足 consensus_output_count <= output_count <= candidate_pool_size <= monte_carlo_samples")
    if config["optimizer"].get("no_signal_strategy", "rotational_coverage") != "rotational_coverage":
        raise ValueError("optimizer.no_signal_strategy 仅支持 rotational_coverage")
    known_models = set(MODEL_LABELS)
    grouped: set[str] = set()
    for group in config["fusion"].get("model_groups", []):
        unknown = set(group) - known_models
        overlap = set(group) & grouped
        if unknown or overlap:
            raise ValueError(f"fusion.model_groups 存在未知或重复模型: unknown={sorted(unknown)}, overlap={sorted(overlap)}")
        grouped.update(group)
    methods = list(config["optimizer"]["methods"])
    if not methods or set(methods) - {"greedy", "genetic", "pareto", "historical_hit", "red_ge3_objective", "global_coverage"}:
        raise ValueError("optimizer.methods 仅支持 greedy、genetic、pareto、historical_hit、red_ge3_objective 和 global_coverage")
    if int(config["optimizer"].get("global_coverage_candidate_cap", 300)) < int(output_count):
        raise ValueError("optimizer.global_coverage_candidate_cap 必须不小于 output_count")
    if int(config["optimizer"].get("global_coverage_time_limit_seconds", 20)) < 1:
        raise ValueError("optimizer.global_coverage_time_limit_seconds 必须至少为1")
    stacking = config.get("stacking", {})
    if "enabled" in stacking and not isinstance(stacking.get("enabled"), bool):
        raise ValueError("stacking.enabled 必须为布尔值")
    if int(stacking.get("min_oof_records", 60)) < 10:
        raise ValueError("stacking.min_oof_records 必须至少为10")
    for key in ("blend_alpha", "red_blend_alpha", "blue_blend_alpha"):
        if key in stacking and not 0.0 <= float(stacking.get(key)) <= 1.0:
            raise ValueError(f"stacking.{key} 必须在 [0, 1] 范围内")
    upgrade = config.get("upgrade", {})
    if str(upgrade.get("blue_mode", "simplified_guard")).strip().lower() not in {
        "research_full",
        "simplified_guard",
        "uniform_coverage",
    }:
        raise ValueError("upgrade.blue_mode 仅支持 research_full、simplified_guard、uniform_coverage")
    if str(upgrade.get("blue_coverage_policy", "probability_optimized")).strip().lower() not in {
        "probability_optimized",
        "rotating_buckets",
        "coverage_only",
        "no_edge_coverage_only",
    }:
        raise ValueError(
            "upgrade.blue_coverage_policy 仅支持 probability_optimized、rotating_buckets、coverage_only、no_edge_coverage_only"
        )
    for key in (
        "expert_pool_extend",
        "ada_hedge_enable",
        "hmm_regime_full_window",
        "nested_hedge_enable",
        "cross_red_blue_feature",
        "coverage_param_online_tune",
        "simplified_blue_experts",
        "blue_no_edge_strict_uniform_coverage",
    ):
        if key in upgrade and not isinstance(upgrade.get(key), bool):
            raise ValueError(f"upgrade.{key} 必须为布尔值")
    if "simplified_hedge_eta" in upgrade and not (0.001 <= float(upgrade.get("simplified_hedge_eta")) <= 1.0):
        raise ValueError("upgrade.simplified_hedge_eta 必须在 [0.001, 1.0] 范围内")
    if "simplified_blue_temperature" in upgrade and not (0.1 <= float(upgrade.get("simplified_blue_temperature")) <= 2.0):
        raise ValueError("upgrade.simplified_blue_temperature 必须在 [0.1, 2.0] 范围内")
    if int(upgrade.get("coverage_tune_window", 80)) < 20:
        raise ValueError("upgrade.coverage_tune_window 必须至少为20")
    if int(upgrade.get("coverage_tune_expensive_variant_cap", 40)) < 20:
        raise ValueError("upgrade.coverage_tune_expensive_variant_cap 必须至少为20")
    if int(upgrade.get("runtime_target_minutes", 30)) < 5:
        raise ValueError("upgrade.runtime_target_minutes 必须至少为5")
    if int(upgrade.get("hmm_half_life", 520)) < 30:
        raise ValueError("upgrade.hmm_half_life 必须至少为30")
    if not 0.0 < float(upgrade.get("hmm_pre_change_decay", 0.42)) <= 1.0:
        raise ValueError("upgrade.hmm_pre_change_decay 必须在 (0, 1] 范围内")
    core_numbers = config.get("core_numbers", {})
    if core_numbers:
        weights = core_numbers.get("weights", {})
        if not isinstance(weights, dict) or sum(max(0.0, float(value)) for value in weights.values()) <= 0:
            raise ValueError("core_numbers.weights 必须配置至少一个正权重")
        if int(core_numbers.get("red_output_count", 6)) < 6:
            raise ValueError("core_numbers.red_output_count 必须至少为6")
        if int(core_numbers.get("blue_output_count", 3)) < 1:
            raise ValueError("core_numbers.blue_output_count 必须至少为1")
    runtime = config.get("runtime", {})
    if runtime:
        profiles = runtime.get("profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("runtime.profiles 必须配置至少一个运行档位")
        default_profile = str(runtime.get("default_profile", "")).strip()
        if default_profile not in profiles:
            raise ValueError("runtime.default_profile 必须存在于 runtime.profiles")
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"runtime.profiles.{profile_name} 必须是配置字典")
            overrides = profile.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"runtime.profiles.{profile_name}.overrides 必须是配置字典")
    blue_variants = list(config["optimizer"].get("blue_specialist_research_variants", []))
    unknown_blue_variants = set(blue_variants) - set(BLUE_SPECIALIST_VARIANTS)
    if unknown_blue_variants:
        raise ValueError(f"optimizer.blue_specialist_research_variants 存在未知策略: {sorted(unknown_blue_variants)}")
    if int(config["backtest"]["random_repeats"]) < 1:
        raise ValueError("backtest.random_repeats 必须至少为1")
    calibration = config["calibration"]
    if int(calibration.get("holdout_periods", 30)) < 1:
        raise ValueError("calibration.holdout_periods 必须至少为1")
    if not 0.0 < float(calibration.get("maximum_brier_ratio", 1.0)):
        raise ValueError("calibration.maximum_brier_ratio 必须大于0")
    if int(calibration.get("rolling_objective_periods", 1)) < 1:
        raise ValueError("calibration.rolling_objective_periods 必须至少为1")
    if int(calibration.get("minimum_adjustment_observations", 1)) < 1:
        raise ValueError("calibration.minimum_adjustment_observations 必须至少为1")
    if int(calibration.get("minimum_failure_streak", 1)) < 1:
        raise ValueError("calibration.minimum_failure_streak 必须至少为1")
    if int(calibration.get("adjustment_cooldown_periods", 0)) < 0:
        raise ValueError("calibration.adjustment_cooldown_periods 不能为负数")
    release = config["release"]
    if int(release.get("minimum_out_of_sample_periods", 200)) < 2:
        raise ValueError("release.minimum_out_of_sample_periods 必须至少为2")
    development_periods = int(release.get("development_periods", 0))
    confirmatory_periods = int(release.get("confirmatory_periods", 0))
    if development_periods < 2 or confirmatory_periods < 2:
        raise ValueError("release.development_periods 与 confirmatory_periods 必须至少为2")
    if development_periods + confirmatory_periods > int(
        release.get("minimum_out_of_sample_periods", 200)
    ):
        raise ValueError("release 的开发窗口与确认窗口总期数不能超过样本外发布门槛")
    blue_signal_coverage = float(release.get("minimum_blue_signal_coverage", 0.80))
    if not 0.0 < blue_signal_coverage <= 1.0:
        raise ValueError("release.minimum_blue_signal_coverage 必须在 (0, 1] 范围内")
    if int(release.get("minimum_prospective_validations", 30)) < 1:
        raise ValueError("release.minimum_prospective_validations 必须至少为1")
    if str(release.get("prospective_code_scope", "current")).strip().lower() not in {"current", "any"}:
        raise ValueError("release.prospective_code_scope 仅支持 current 或 any")
    if int(release.get("provisional_min_out_of_sample_periods", 60)) < 1:
        raise ValueError("release.provisional_min_out_of_sample_periods 必须至少为1")
    provisional_red_signal = float(release.get("provisional_min_red_signal_strength", 0.03))
    if not 0.0 <= provisional_red_signal <= 1.0:
        raise ValueError("release.provisional_min_red_signal_strength 必须在 [0, 1] 范围内")
    provisional_total_signal = float(release.get("provisional_min_total_signal_strength", 0.12))
    if not 0.0 <= provisional_total_signal <= 2.0:
        raise ValueError("release.provisional_min_total_signal_strength must be in [0, 2]")
    audit_config = config["randomness_audit"]
    if not 0.0 < float(audit_config.get("significance_level", 0.05)) < 1.0:
        raise ValueError("randomness_audit.significance_level 必须在 (0, 1) 范围内")
    if int(audit_config.get("autocorrelation_lags", 10)) < 1:
        raise ValueError("randomness_audit.autocorrelation_lags 必须至少为1")
    if int(audit_config.get("spectral_permutations", 199)) < 19:
        raise ValueError("randomness_audit.spectral_permutations 必须至少为19")
    if int(audit_config.get("chi_square_simulations", 199)) < 99:
        raise ValueError("randomness_audit.chi_square_simulations 必须至少为99")
    if str(config["data"].get("invalid_row_policy", "reject")) not in {"reject", "quarantine"}:
        raise ValueError("data.invalid_row_policy 仅支持 reject 或 quarantine")
    if bool(config["compliance"].get("require_tamper_evident_audit", False)) and not str(
        config["compliance"].get("audit_hmac_env", "")
    ).strip():
        raise ValueError("启用防篡改预测审计时必须配置 compliance.audit_hmac_env")
    confidence = float(config["report"].get("confidence_level", 0.95))
    if not 0.5 < confidence < 1.0:
        raise ValueError("report.confidence_level 必须在 (0.5, 1.0) 范围内")
    if int(config["report"].get("permutation_repetitions", 20000)) < 1000:
        raise ValueError("report.permutation_repetitions 必须至少为1000")


def _assert_calendar_feature_dates_available(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    calendar_enabled = bool(config.get("features", {}).get("calendar_features_enabled", False)) or bool(
        config.get("upgrade", {}).get("calendar_features_enabled", False)
    )
    if not calendar_enabled:
        return
    if frame.empty or "date_provenance" not in frame.columns:
        raise ValueError("已启用日历特征，但清洗数据缺少日期来源标记 date_provenance")
    source_ratio = float(frame["date_provenance"].astype(str).str.lower().eq("source").mean())
    if source_ratio < 0.98:
        raise ValueError(
            f"已启用日历特征，但真实来源日期比例仅 {source_ratio:.2%}，低于 98%；"
            "请先导入含福彩官网开奖日期的数据，或关闭 calendar_features_enabled。"
        )


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config_fingerprint_payload(config),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def audit_signing_key(config: dict[str, Any]) -> str:
    compliance = config["compliance"]
    variable = str(compliance.get("audit_hmac_env", "TSYS_AUDIT_HMAC_KEY"))
    value, source = load_or_create_audit_key(variable)
    logging.info("预测审计密钥已就绪 | 来源=%s | 环境变量=%s", source, variable)
    return value


def runtime_metadata(config: dict[str, Any]) -> dict[str, Any]:
    distributions = [
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "matplotlib", "PyYAML", "scipy", "openpyxl", "optuna",
        "PuLP", "catboost", "pytest",
    ]
    versions: dict[str, str] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "random_seed": int(config["project"]["random_seed"]),
        "config_sha256": config_fingerprint(config),
        "source_tree_sha256": source_tree_fingerprint(ROOT),
    }


def project_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else ROOT / value


def load_validation_trace_view(path: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the JSONL validation trace into a compact dashboard-friendly view."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "rows": [],
            "periods": 0,
            "run_start": None,
            "run_complete": None,
        }
    rows: list[dict[str, Any]] = []
    run_start: dict[str, Any] | None = None
    run_complete: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("逐期追踪文件存在无法解析行 | 事件=VALIDATION_TRACE_PARSE_FAILED | 文件=%s", path)
                continue
            event = record.get("event")
            if event == "validation_trace_run_start":
                run_start = record
                continue
            if event == "validation_trace_run_complete":
                run_complete = record
                continue
            if event != "validation_period_trace":
                continue
            actual = record.get("actual", {}) or {}
            actual_trace = record.get("actual_number_trace", {}) or {}
            red_trace = actual_trace.get("red", []) or []
            blue_trace = actual_trace.get("blue", {}) or {}
            final_tickets = record.get("final_recommendation", []) or []
            final_ticket_rows = [
                {
                    "index": int(ticket.get("ticket_index", index + 1)),
                    "red": [int(number) for number in ticket.get("red", [])],
                    "blue": int(ticket.get("blue", 0) or 0),
                    "matched_red": [int(number) for number in ticket.get("matched_red", [])],
                    "blue_hit": bool(ticket.get("blue_hit", False)),
                    "role": ticket.get("ticket_selection_role"),
                    "method_sources": ticket.get("method_ticket_sources", []),
                    "candidate_scores": ticket.get("candidate_scores", {}),
                }
                for index, ticket in enumerate(final_tickets)
            ]
            recommendation = record.get("recommendation_context", {}) or {}
            method_recommendations = record.get("method_recommendations", {}) or {}
            attempted_methods = [
                str(method)
                for method in method_recommendations
                if str(method) != "consensus"
            ]

            def top_model_reasons(item: dict[str, Any]) -> list[dict[str, Any]]:
                values = item.get("model_reasons", []) or []
                return [
                    {
                        "model": str(reason.get("model", "")),
                        "rank": int(reason.get("rank", 0) or 0),
                        "score": float(reason.get("score", 0.0) or 0.0),
                        "color_weight": float(reason.get("color_weight", 0.0) or 0.0),
                    }
                    for reason in values[:3]
                ]

            def final_layer_reason(item: dict[str, Any], *, color: str) -> str:
                candidate_pool = item.get("candidate_pool") or {}
                best = candidate_pool.get("best_candidate") or {}
                final_sets = [set(ticket.get("red", []) or []) for ticket in final_ticket_rows]
                best_red = set(best.get("red", []) or [])
                max_overlap = max((len(best_red & red_set) for red_set in final_sets), default=0)
                best_rank = best.get("candidate_rank")
                support_count = int(candidate_pool.get("candidate_pool_support_count", 0) or 0)
                method_names = [
                    str(value.get("method"))
                    for value in (item.get("method_support") or [])
                    if value.get("method")
                ]
                strategy = str(recommendation.get("consensus_strategy") or "最终综合层")
                method_text = "、".join(method_names) if method_names else "候选优化方法"
                rank_text = f"候选池最佳第{best_rank}名，出现{support_count}次" if best_rank else f"候选池出现{support_count}次"
                if color == "blue":
                    final_blues = sorted({int(ticket.get("blue", 0) or 0) for ticket in final_ticket_rows if ticket.get("blue")})
                    return (
                        f"未进入预算胆拖：该蓝球虽被{method_text}选中过，且{rank_text}；"
                        f"但在{strategy}阶段被压缩掉。判断：系统优先保留蓝球唯一性、覆盖轮换和整票综合分更高的蓝球。"
                    )
                if max_overlap <= 2 and support_count >= 20:
                    detail = "含该号的高支持候选票与最终保留红球交集不足3个，属于“整票信号被拆散”；本轮逻辑已新增整票重叠保护，优先桥接这类候选票。"
                elif max_overlap >= 5:
                    detail = "主要原因是含该号候选与已保留组合高度重叠，最终多样性/去重再平衡优先替换。"
                elif best_rank and int(best_rank) > max(10, len(final_ticket_rows) * 3):
                    detail = "主要原因是含该号候选虽然进池，但综合排序靠后，未挤进最终保留集合。"
                elif support_count >= 20:
                    detail = "该号并非完全无信号，主要是在最终共识投票、重叠约束和覆盖目标之间被挤掉。"
                else:
                    detail = "主要原因是候选支持次数偏少，最终组合层优先保留跨方法支持更稳定的票。"
                return (
                    f"未进入预算胆拖：该号虽被{method_text}选中过，且{rank_text}；"
                    f"但在{strategy}阶段被压缩掉。证据：含该号候选与已保留候选最高重叠{max_overlap}个红球。判断：{detail}"
                )

            def eliminated_by(item: dict[str, Any], *, color: str) -> str:
                if item.get("selected_in_final"):
                    return "已通过候选筛选：进入最终筛选前的候选集合"
                stage = str(item.get("drop_stage_inference") or "")
                candidate_pool = item.get("candidate_pool") or {}
                support = [
                    str(value.get("method"))
                    for value in (item.get("method_support") or [])
                    if value.get("method")
                ]
                if stage == "low_confidence_exclusion_filter":
                    return "未进入预算胆拖：在低置信剔除器阶段被排除，未进入候选生成。判断：模型共识或历史支持不足。"
                channels = [str(channel) for channel in (item.get("candidate_channel") or [])]
                if "wide_candidate" in channels:
                    return (
                        "Final5组合优化失败：该号已进入Candidate，但最终5注没有保留。"
                        "这不是模型没有预测到，而是Final5把已经找到的号码拆散或丢掉了。"
                    )
                if not candidate_pool.get("entered_candidate_pool"):
                    return "未进入预算胆拖：在候选生成/候选池保留阶段被挡住。判断：模型概率、抽样结果或质量排序不足，未进入可选候选池。"
                if not support:
                    methods_text = "/".join(attempted_methods) if attempted_methods else "优化方法"
                    return f"未进入预算胆拖：该号进入候选池，但未被{methods_text}选中。判断：组合质量、结构约束或覆盖目标不占优。"
                return final_layer_reason(item, color=color)

            missed_red = [
                int(item.get("number"))
                for item in red_trace
                if item.get("number") is not None and not item.get("selected_in_final")
            ]
            red_drop_stages = [
                {
                    "number": int(item.get("number")),
                    "stage": item.get("drop_stage_inference"),
                    "selected_in_final": bool(item.get("selected_in_final")),
                    "entered_candidate_pool": bool(
                        (item.get("candidate_pool") or {}).get("entered_candidate_pool")
                    ),
                    "candidate_pool_support_count": int(
                        (item.get("candidate_pool") or {}).get("candidate_pool_support_count", 0)
                    ),
                    "best_candidate_rank": (
                        ((item.get("candidate_pool") or {}).get("best_candidate") or {}).get("candidate_rank")
                    ),
                    "method_support": [
                        support.get("method")
                        for support in (item.get("method_support") or [])
                        if support.get("method")
                    ],
                    "top_models": top_model_reasons(item),
                    "eliminated_by": eliminated_by(item, color="red"),
                }
                for item in red_trace
                if item.get("number") is not None
            ]
            dantuo_plan = record.get("dantuo_plan")
            if not dantuo_plan:
                dantuo_config = (config or {}).get("dantuo", {})
                dantuo_source_tickets = [
                    {
                        "red": ticket.get("red", []),
                        "blue": ticket.get("blue"),
                        "consensus_score": (ticket.get("candidate_scores") or {}).get("quality_score", 0.0),
                        "selection_role": ticket.get("ticket_selection_role")
                        or ",".join(str(item) for item in ticket.get("method_ticket_sources", []) or []),
                        "blue_selection_role": (ticket.get("blue_number_reason") or {}).get("final_reason", ""),
                    }
                    for ticket in final_tickets
                ]
                method_limit = max(1, int(dantuo_config.get("method_ticket_limit_per_method", 5)))
                for method, tickets in method_recommendations.items():
                    if str(method) == "consensus":
                        continue
                    for ticket in (tickets or [])[:method_limit]:
                        dantuo_source_tickets.append({
                            "red": ticket.get("red", []),
                            "blue": ticket.get("blue"),
                            "consensus_score": 0.0,
                            "selection_role": f"dantuo_method_source_{method}",
                            "blue_selection_role": str(ticket.get("selection_role", "") or ""),
                        })
                dantuo_plan = build_dantuo_plan(
                    dantuo_source_tickets,
                    final_recommendation=final_tickets,
                    fusion=recommendation.get("fused_prediction", {}),
                    actual=actual,
                    max_amount=int(dantuo_config.get("max_amount", 20)),
                    unit_price=int(dantuo_config.get("unit_price", 2)),
                    preferred_red_dan_count=int(dantuo_config.get("preferred_red_dan_count", 2)),
                    preferred_red_tuo_count=int(dantuo_config.get("preferred_red_tuo_count", 5)),
                    preferred_blue_count=int(dantuo_config.get("preferred_blue_count", 2)),
                    prefer_compound_when_coverage_like=bool(
                        dantuo_config.get("prefer_compound_when_coverage_like", True)
                    ),
                    preserve_final_portfolio_evidence=bool(
                        dantuo_config.get("preserve_final_portfolio_evidence", True)
                    ),
                    final_to_dantuo_audit_enabled=bool(
                        dantuo_config.get("final_to_dantuo_audit_enabled", True)
                    ),
                    compound_red_count=int(dantuo_config.get("compound_red_count", 7)),
                    compound_blue_count=int(dantuo_config.get("compound_blue_count", 1)),
                    blue_recent_gap_weight=float(dantuo_config.get("blue_recent_gap_weight", 0.20)),
                    blue_recent_frequency_weight=float(dantuo_config.get("blue_recent_frequency_weight", 0.15)),
                    blue_evidence_weight=float(dantuo_config.get("blue_evidence_weight", 0.75)),
                    blue_specialist_weight=float(dantuo_config.get("blue_specialist_weight", 0.25)),
                    blue_omission_balance_weight=float(dantuo_config.get("blue_omission_balance_weight", 0.80)),
                    blue_immediate_repeat_penalty=float(dantuo_config.get("blue_immediate_repeat_penalty", 0.35)),
                    blue_recent_windows=dantuo_config.get("blue_recent_windows", [20, 50, 100]),
                    auto_cap_blue_to_budget=bool(dantuo_config.get("auto_cap_blue_to_budget", True)),
                )
            dantuo_red_set = set()
            dantuo_blue_set = set()
            if dantuo_plan:
                dantuo_red_set = {
                    int(number)
                    for number in list(dantuo_plan.get("red_dan", []) or [])
                    + list(dantuo_plan.get("red_tuo", []) or [])
                }
                dantuo_blue_set = {int(number) for number in dantuo_plan.get("blue", []) or []}
            actual_red_set = {int(number) for number in actual.get("red", []) or []}
            actual_blue_value = int(actual.get("blue", 0)) if actual.get("blue") is not None else None
            dantuo_hit_red = sorted(actual_red_set & dantuo_red_set)
            dantuo_missed_red = sorted(actual_red_set - dantuo_red_set) if dantuo_plan else missed_red
            dantuo_blue_hit = bool(actual_blue_value in dantuo_blue_set) if actual_blue_value else False
            dantuo_missed_blue = None if dantuo_blue_hit else actual_blue_value
            blue_budget_eliminated_by = None
            if (
                dantuo_plan
                and actual_blue_value
                and bool(blue_trace.get("selected_in_final", False))
                and actual_blue_value not in dantuo_blue_set
            ):
                blue_budget_eliminated_by = (
                    "未进入预算胆拖：该蓝球已通过前置候选筛选，但在预算胆拖蓝球层被替换。"
                    "判断：系统优先选择融合排序、候选支持和蓝球覆盖结构综合更高的蓝球。"
                )

            def dantuo_layer_reason(item: dict[str, Any]) -> str:
                number = int(item.get("number", 0))
                method_names = [str(method) for method in item.get("method_support", []) if method]
                method_text = "、".join(method_names) if method_names else "候选优化方法"
                support_count = int(item.get("candidate_pool_support_count", 0) or 0)
                best_rank = item.get("best_candidate_rank")
                rank_text = f"候选池最佳第{best_rank}名，出现{support_count}次" if best_rank else f"候选池出现{support_count}次"
                strategy = str(recommendation.get("consensus_strategy") or "最终综合层")
                best_candidate = (
                    ((next(
                        (
                            trace_item
                            for trace_item in red_trace
                            if int(trace_item.get("number", 0) or 0) == number
                        ),
                        {},
                    ).get("candidate_pool") or {}).get("best_candidate") or {})
                )
                best_red = {int(value) for value in best_candidate.get("red", []) or []}
                retained_overlap = len(best_red & dantuo_red_set) if best_red else 0
                if retained_overlap <= 2 and support_count >= 20:
                    judgment = "含该号的高支持候选票与预算胆拖保留红球交集不足3个，属于整票信号被拆散；系统已新增整票重叠保护，优先桥接这类候选票。"
                elif retained_overlap >= 5:
                    judgment = "含该号候选与预算胆拖保留号码高度重叠，最终去重/分散约束优先保留了其它组合。"
                elif support_count >= 20:
                    judgment = "该号并非完全无信号，主要是在最终共识投票、重叠约束和覆盖目标之间被挤掉。"
                elif best_rank and int(best_rank) > 10:
                    judgment = "含该号候选进入过候选池，但综合排序未处在最优梯队。"
                else:
                    judgment = "该号候选支持偏弱，预算胆拖压缩层优先保留更稳定的共识号码。"
                return (
                    f"未进入预算胆拖：该号虽被{method_text}选中过，且{rank_text}；"
                    f"但在{strategy}阶段被压缩掉。证据：含该号候选与预算胆拖保留号码重叠{retained_overlap}个红球。判断：{judgment}"
                )

            for item in red_drop_stages:
                number = int(item.get("number", 0))
                item["selected_in_budget_plan"] = number in dantuo_red_set
                if dantuo_plan and item.get("selected_in_final") and number not in dantuo_red_set:
                    item["budget_eliminated_by"] = dantuo_layer_reason(item)
            rows.append({
                "validation_period": str(record.get("validation_period", "")),
                "training_through_period": str(record.get("training_through_period", "")),
                "actual_red": [int(number) for number in actual.get("red", [])],
                "actual_blue": int(actual.get("blue", 0)) if actual.get("blue") is not None else None,
                "final_tickets": final_ticket_rows,
                "red_hit_count": len(dantuo_hit_red) if dantuo_plan else int(sum(1 for item in red_trace if item.get("selected_in_final"))),
                "blue_hit": dantuo_blue_hit if dantuo_plan else bool(blue_trace.get("selected_in_final", False)),
                "missed_red": dantuo_missed_red,
                "missed_blue": dantuo_missed_blue if dantuo_plan else actual_trace.get("missed_blue"),
                "covered_red": dantuo_hit_red,
                "covered_blue": actual_blue_value if dantuo_blue_hit else None,
                "dantuo_plan": dantuo_plan,
                "blue_selected_count": len(dantuo_blue_set) if dantuo_plan else len({
                    int(ticket.get("blue", 0) or 0)
                    for ticket in final_ticket_rows
                    if 1 <= int(ticket.get("blue", 0) or 0) <= 16
                }),
                "blue_metric": (
                    f"dantuo_blue_hit_at_{len(dantuo_blue_set)}"
                    if dantuo_plan
                    else "final_portfolio_blue_hit_at_unique_count"
                ),
                "blue_theoretical_random_hit_probability": (
                    len(dantuo_blue_set) / 16.0
                    if dantuo_plan
                    else None
                ),
                "red_drop_stages": red_drop_stages,
                "final5_drop_regret": (record.get("final5_hit_coherence") or {}).get("production") or {},
                "blue_drop_stage": blue_trace.get("drop_stage_inference"),
                "blue_top_models": top_model_reasons(blue_trace),
                "blue_eliminated_by": eliminated_by(blue_trace, color="blue") if blue_trace else None,
                "blue_budget_eliminated_by": blue_budget_eliminated_by,
                "blue_method_support": [
                    support.get("method")
                    for support in (blue_trace.get("method_support") or [])
                    if support.get("method")
                ] if blue_trace else [],
                "blue_entered_candidate_pool": bool(
                    (blue_trace.get("candidate_pool") or {}).get("entered_candidate_pool")
                ),
                "blue_candidate_pool_support_count": int(
                    (blue_trace.get("candidate_pool") or {}).get("candidate_pool_support_count", 0)
                ),
                "blue_best_candidate_rank": (
                    ((blue_trace.get("candidate_pool") or {}).get("best_candidate") or {}).get("candidate_rank")
                ),
                "consensus_strategy": recommendation.get("consensus_strategy"),
                "candidate_pool_size": int(recommendation.get("candidate_pool_size", 0) or 0),
                "strict_left_truncation_passed": bool(record.get("strict_left_truncation_passed", False)),
            })
    return {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "periods": len(rows),
        "run_start": run_start,
        "run_complete": run_complete,
    }


def next_period_label(period: str) -> str:
    value = str(period).strip()
    if not value.isdigit():
        raise ValueError(f"期号必须是纯数字，收到: {period}")
    return str(int(value) + 1)


def configure_logging(config: dict[str, Any]) -> None:
    logs_dir = project_path(config, "logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    configured_level = str(config.get("logging", {}).get("level", "INFO")).strip().upper()
    level = getattr(logging, configured_level, logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    logging_config = config.get("logging", {})
    max_bytes = max(1024 * 1024, int(logging_config.get("max_bytes", 10 * 1024 * 1024)))
    backup_count = max(1, int(logging_config.get("backup_count", 5)))
    file_handler = WindowsSafeRotatingFileHandler(
        logs_dir / "pipeline.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    _enable_console_scroll_and_selection()
    console_handler = NonBlockingStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler],
        force=True,
    )
    logging.info(
        "日志系统就绪 | 事件=LOGGING_READY | 级别=%s | 控制台=标准输出 | 日志文件=%s",
        logging.getLevelName(level),
        logs_dir / "pipeline.log",
    )


@contextmanager
def logged_step(name: str, **fields: Any):
    started = time.perf_counter()
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    description = STEP_DESCRIPTIONS.get(name, name)
    logging.info("步骤开始 | 事件=STEP_START | 步骤=%s | 标识=%s | %s", description, name, detail)
    try:
        yield
    except Exception as exc:
        duration = time.perf_counter() - started
        RUNTIME_PROFILER.record(
            name,
            started_at=started,
            duration_seconds=duration,
            status="failed",
            fields=fields,
            error_type=exc.__class__.__name__,
        )
        logging.exception(
            "步骤失败 | 事件=STEP_FAILED | 步骤=%s | 标识=%s | 耗时=%.3f秒",
            description, name, duration,
        )
        raise
    duration = time.perf_counter() - started
    RUNTIME_PROFILER.record(
        name,
        started_at=started,
        duration_seconds=duration,
        status="done",
        fields=fields,
    )
    logging.info(
        "步骤完成 | 事件=STEP_DONE | 步骤=%s | 标识=%s | 耗时=%.3f秒",
        description, name, duration,
    )


def prepare_data(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    with logged_step("data_load", source=project_path(config, "raw_data")):
        raw = retry_call(
            lambda: load_csv(project_path(config, "raw_data")),
            attempts=int(config.get("data", {}).get("load_max_retries", 3)),
            delay_seconds=float(config.get("data", {}).get("load_retry_delay_seconds", 0.25)),
            label="load_raw_history_csv",
        )
        logging.info("数据读取完成 | 事件=DATA_LOADED | 原始行数=%s | 字段=%s", len(raw), list(raw.columns))
    with logged_step("data_clean", input_rows=len(raw)):
        result = clean_history(
            raw,
            fill_missing_dates=bool(config["data"].get("fill_missing_dates", True)),
            draw_weekdays=config["data"].get("draw_weekdays", [1, 3, 6]),
            draw_blackout_ranges=config["data"].get("draw_blackout_ranges", []),
            invalid_row_policy=str(config["data"].get("invalid_row_policy", "reject")),
            validate_source_dates=bool(config["data"].get("validate_source_dates", False)),
        )
        _assert_calendar_feature_dates_available(result.data, config)
        minimum = int(config["data"].get("min_rows", 60))
        if len(result.data) < minimum:
            raise ValueError(f"清洗后数据不足 {minimum} 期")
        save_clean_data(result.data, project_path(config, "clean_data"))
        manifest = create_data_manifest(
            raw_path=project_path(config, "raw_data"),
            clean_frame=result.data,
            cleaning_report=result.report,
        )
        write_data_manifest(project_path(config, "data_manifest"), manifest)
        result.report["data_manifest"] = manifest
        logging.info(
            "数据清洗完成 | 事件=DATA_CLEANED | 清洗统计=%s | 最早期=%s | 最新期=%s",
            result.report,
            result.data.iloc[0]["period"],
            result.data.iloc[-1]["period"],
        )
    return result.data, result.report


def _uniform_model_prediction() -> dict[str, np.ndarray]:
    return {
        "red": np.full(33, 1.0 / 33.0, dtype=float),
        "blue": np.full(16, 1.0 / 16.0, dtype=float),
    }


def _log_detail_enabled(config: dict[str, Any], key: str, *, default: str = "summary") -> bool:
    value = str(config.get("logging", {}).get(key, default)).strip().lower()
    return value in {"full", "verbose", "debug"}


def fit_current_probabilities(
    frame: Any,
    config: dict[str, Any],
    weights_by_color: dict[str, dict[str, float]],
    weight_reason_by_color: dict[str, str],
    stacking_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_idx = int(frame["idx"].max()) + 1
    assert_left_truncated(frame, target_idx)
    generation_config = config.get("generation", {})
    ranking_red_count = int(generation_config.get("model_ranking_red_count", 18))
    ranking_blue_count = int(generation_config.get("model_ranking_blue_count", 8))
    predictions = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    max_attempts = max(1, int(config.get("models", {}).get("fit_max_retries", 2)))
    model_detail_enabled = _log_detail_enabled(config, "model_detail")
    for name, model in build_models(config).items():
        with logged_step("current_model_fit", model=name, training_periods=len(frame)):
            fallback_error: str | None = None
            if bool(getattr(model, "disabled_", False)):
                fallback_error = str(getattr(model, "reason", "disabled_by_config"))
                predictions[name] = model.predict(frame, target_idx)
                logging.debug(
                    "当前模型已按配置禁用，使用均匀占位 | 事件=MODEL_CURRENT_SKIPPED_DISABLED | 模型=%s(%s) | 原因=%s",
                    MODEL_LABELS.get(name, name),
                    name,
                    fallback_error,
                )
            else:
                for attempt in range(1, max_attempts + 1):
                    try:
                        model.fit(frame)
                        predictions[name] = model.predict(frame, target_idx)
                        break
                    except Exception as exc:
                        fallback_error = str(exc)
                        if attempt >= max_attempts:
                            logging.exception(
                                "当前模型训练或预测失败，已降级为均匀分布 | 事件=MODEL_CURRENT_FALLBACK | 模型=%s(%s) | 尝试=%s/%s | 错误=%s",
                                MODEL_LABELS.get(name, name),
                                name,
                                attempt,
                                max_attempts,
                                exc,
                            )
                            predictions[name] = _uniform_model_prediction()
                        else:
                            logging.warning(
                                "当前模型训练或预测失败，准备重试 | 事件=MODEL_CURRENT_RETRY | 模型=%s(%s) | 尝试=%s/%s | 错误=%s",
                                MODEL_LABELS.get(name, name),
                                name,
                                attempt,
                                max_attempts,
                                exc,
                            )
            (logging.info if model_detail_enabled else logging.debug)(
                "当前模型评分完成 | 事件=MODEL_CURRENT_DONE | 模型=%s(%s) | 训练Brier=%s | 红球前6=%s | 蓝球首选=%s",
                MODEL_LABELS.get(name, name),
                name,
                getattr(model, "training_brier_", None),
                (np.argsort(predictions[name]["red"])[-6:] + 1).tolist(),
                int(np.argmax(predictions[name]["blue"]) + 1),
            )
            model_summaries[name] = {
                **rank_prediction(
                    predictions[name],
                    red_count=ranking_red_count,
                    blue_count=ranking_blue_count,
                ),
                "objective_by_color": getattr(model, "model_modes_", {}),
                "weight": float(np.mean([
                    weights_by_color["red"].get(name, 0.0),
                    weights_by_color["blue"].get(name, 0.0),
                ])),
                "red_weight": float(weights_by_color["red"].get(name, 0.0)),
                "blue_weight": float(weights_by_color["blue"].get(name, 0.0)),
                "training_brier": (
                    float(model.training_brier_)
                    if getattr(model, "training_brier_", None) is not None
                    else None
                ),
                "training_brier_by_color": {
                    color: float(value)
                    for color, value in getattr(model, "training_brier_by_color_", {}).items()
                },
                "feature_importance": getattr(model, "feature_importance_", {}),
                "fallback_error": fallback_error if getattr(model, "training_brier_", None) is None else None,
                "disabled": bool(getattr(model, "disabled_", False)),
            }
    runtime_control = build_evidence_runtime_control(
        stacking_records or [],
        config,
        ticket_count=int(config.get("optimizer", {}).get("consensus_count", config.get("generation", {}).get("ticket_count", 5))),
    )
    weights_by_color, evidence_weight_control = apply_evidence_weight_control(weights_by_color, runtime_control, config)
    weight_reason_by_color = {
        **weight_reason_by_color,
        "red": f"{weight_reason_by_color['red']};evidence_control={runtime_control.get('red_budget_state')}",
        "blue": f"{weight_reason_by_color['blue']};evidence_control={runtime_control.get('blue_edge_state')}",
    }
    fused_base = fuse_predictions(predictions, weights_by_color)
    stacking_prediction, stacking_info = stacking_meta_predict(
        stacking_records or [],
        predictions,
        config,
    )
    fused = blend_with_stacking(fused_base, stacking_prediction, config, stacking_info=stacking_info)
    if bool((config.get("v7") or {}).get("enabled", True)) and bool(
        ((config.get("v7") or {}).get("uncertainty") or {}).get("probability_shrinkage", True)
    ):
        alpha = float(runtime_control.get("blue_alpha", 0.5))
        shrunk_blue, shrinkage_info = shrink_blue_probability_to_uniform(
            fused["blue"],
            alpha=alpha,
            reason=f"v7_runtime_{runtime_control.get('blue_edge_state', 'NO_EDGE')}",
        )
        fused = {**fused, "blue": shrunk_blue}
        stacking_info["v7_blue_shrinkage"] = {
            **shrinkage_info,
            "edge_report": runtime_control.get("blue_edge", {}),
            "topk_policy": runtime_control.get("blue_prediction_set", {}).get("policy", {}),
            "production_candidate_width": runtime_control.get("blue_candidate_width"),
        }
    stacking_info["v7_runtime_control"] = runtime_control
    stacking_info["v7_evidence_weight_control"] = evidence_weight_control
    logging.info(
        "当前融合完成 | 事件=FUSION_CURRENT | 红球权重=%s | 蓝球权重=%s | 红球原因=%s | 蓝球原因=%s | Stacking=%s",
        weights_by_color["red"], weights_by_color["blue"],
        weight_reason_by_color["red"], weight_reason_by_color["blue"],
        stacking_info,
    )
    fused_ranking = rank_prediction(
        fused,
        red_count=ranking_red_count,
        blue_count=ranking_blue_count,
        weights_by_color=weights_by_color,
    )
    probability_health = probability_health_report({
        **predictions,
        "fused_final": fused,
    })
    return fused, {
        "training_periods": int(len(frame)),
        "training_through_period": str(frame.iloc[-1]["period"]),
        "history_max_idx": int(frame["idx"].max()),
        "target_idx": target_idx,
        "strict_left_truncation": int(frame["idx"].max()) + 1 == target_idx,
        "models": model_summaries,
        "fusion": {
            **fused_ranking,
            "weights": {
                name: float(np.mean([weights_by_color["red"].get(name, 0.0), weights_by_color["blue"].get(name, 0.0)]))
                for name in MODEL_LABELS
            },
            "weights_by_color": weights_by_color,
            "weight_reason": (
                f"red={weight_reason_by_color['red']};blue={weight_reason_by_color['blue']}"
            ),
            "weight_reason_by_color": weight_reason_by_color,
            "stacking": stacking_info,
            "v7_runtime_control": runtime_control,
            "evidence_weight_control": evidence_weight_control,
        },
        "probability_health": probability_health,
    }


def _core_aggregate_ticket(
    method_selected: dict[str, list[dict[str, Any]]],
    current_selected: list[dict[str, Any]],
    fusion: dict[str, Any],
    optimizer_config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(optimizer_config.get("final_core_aggregation_enabled", True)):
        return None
    source_tickets: list[dict[str, Any]] = []
    method_limit = max(1, int(optimizer_config.get("final_core_method_ticket_limit_per_method", 5)))
    for method, tickets in method_selected.items():
        for rank, ticket in enumerate(list(tickets or [])[:method_limit]):
            if len(ticket.get("red", []) or []) == 6:
                source_tickets.append({**ticket, "_source_method": method, "_source_rank": rank})
    for rank, ticket in enumerate(current_selected or []):
        if len(ticket.get("red", []) or []) == 6:
            source_tickets.append({**ticket, "_source_method": "current_final", "_source_rank": rank})
    if not source_tickets:
        return None

    fusion_order = [
        int(number)
        for number in fusion.get("top_red", [])
        if str(number).isdigit() and 1 <= int(number) <= 33
    ]
    fusion_position = {number: rank for rank, number in enumerate(fusion_order)}
    red_support: Counter[int] = Counter()
    pair_support: Counter[tuple[int, int]] = Counter()
    blue_support: Counter[int] = Counter()
    for ticket in source_tickets:
        rank = int(ticket.get("_source_rank", 0))
        source_weight = 1.15 if ticket.get("_source_method") != "current_final" else 0.85
        ticket_weight = source_weight / (1.0 + 0.18 * rank)
        ticket_score = max(
            float(ticket.get("consensus_score", 0.0) or 0.0),
            float(ticket.get("quality_score", 0.0) or 0.0),
            0.0,
        )
        red = sorted({int(number) for number in ticket.get("red", []) if 1 <= int(number) <= 33})
        if len(red) != 6:
            continue
        for number in red:
            red_support[number] += ticket_weight + 0.03 * ticket_score
        for left, right in combinations(red, 2):
            pair_support[(left, right)] += 0.5 * ticket_weight
        blue = ticket.get("blue")
        if str(blue).isdigit() and 1 <= int(blue) <= 16:
            blue_support[int(blue)] += ticket_weight + 0.02 * ticket_score

    if not red_support:
        return None
    current_final_sets: list[set[int]] = [
        set(sorted({int(number) for number in ticket.get("red", []) if 1 <= int(number) <= 33}))
        for ticket in current_selected or []
        if len(ticket.get("red", []) or []) == 6
    ]
    final_support: Counter[int] = Counter()
    for rank, red_set in enumerate(current_final_sets):
        for number in red_set:
            final_support[number] += 1.0 / (1.0 + 0.12 * rank)
    max_support = max(red_support.values(), default=1.0) or 1.0
    fusion_top_k = max(6, int(optimizer_config.get("final_core_fusion_top_k", 12)))

    def fusion_rank_score(number: int, top_k: int | None = None) -> float:
        limit = max(6, int(top_k or fusion_top_k))
        pos = fusion_position.get(number)
        return max(0.0, 1.0 - float(pos) / float(limit)) if pos is not None and pos < limit else 0.0

    def number_score(number: int) -> float:
        support_score = float(red_support.get(number, 0.0)) / max_support
        fusion_score = fusion_rank_score(number, fusion_top_k)
        return 2.2 * support_score + 1.65 * fusion_score

    candidate_pool_size = max(8, int(optimizer_config.get("final_core_candidate_pool_size", 14)))
    candidate_numbers = sorted(
        set(red_support) | set(fusion_order[:fusion_top_k]),
        key=lambda number: (number_score(number), -number),
        reverse=True,
    )[:candidate_pool_size]
    if len(candidate_numbers) < 6:
        return None

    max_final_support = max(final_support.values(), default=1.0) or 1.0
    max_pair_support = max(pair_support.values(), default=1.0) or 1.0

    def red_zone(number: int) -> int:
        if int(number) <= 11:
            return 1
        if int(number) <= 22:
            return 2
        return 3

    def is_stale_dominant(number: int) -> bool:
        return bool(
            red_support.get(number, 0.0) >= 0.75 * max_support
            and fusion_rank_score(number, max(18, fusion_top_k)) <= 0.12
        )

    def quota_number_score(number: int, selected_numbers: list[int] | None = None) -> float:
        selected_numbers = selected_numbers or []
        support_score = (float(red_support.get(number, 0.0)) / max_support) ** 0.5
        final_score = float(final_support.get(number, 0.0)) / max_final_support
        fusion_score = fusion_rank_score(number, max(18, fusion_top_k))
        pair_score = (
            sum(float(pair_support.get(tuple(sorted((number, other))), 0.0)) for other in selected_numbers)
            / max_pair_support
            if selected_numbers
            else 0.0
        )
        zone_load = sum(1 for other in selected_numbers if red_zone(other) == red_zone(number))
        boundary_penalty = 0.42 if int(number) in {1, 2, 32, 33} else 0.0
        stale_penalty = 0.72 if is_stale_dominant(number) and number not in selected_numbers else 0.0
        under_represented_bonus = (
            0.28
            if final_support.get(number, 0.0) > 0 and red_support.get(number, 0.0) < 0.68 * max_support
            else 0.0
        )
        return float(
            0.72 * support_score
            + 0.86 * final_score
            + 0.92 * fusion_score
            + 0.38 * pair_score
            + under_represented_bonus
            - 0.38 * max(0, zone_load - 1)
            - boundary_penalty
            - stale_penalty
        )

    def build_quota_collector_red() -> tuple[int, ...] | None:
        if not bool(optimizer_config.get("final_core_quota_collector_enabled", True)):
            return None
        if not current_final_sets:
            return None
        final_union = sorted(set().union(*current_final_sets))
        if len(final_union) < 6:
            return None
        selected_numbers: list[int] = []

        anchor_candidates = sorted(
            final_union,
            key=lambda number: (
                1.05 * (float(red_support.get(number, 0.0)) / max_support) ** 0.5
                + 0.88 * (float(final_support.get(number, 0.0)) / max_final_support)
                + 0.22 * fusion_rank_score(number, max(18, fusion_top_k))
                - (0.25 if int(number) in {1, 2, 32, 33} else 0.0),
                -number,
            ),
            reverse=True,
        )
        for number in anchor_candidates:
            if len(selected_numbers) >= 2:
                break
            if any(abs(int(number) - int(other)) <= 3 for other in selected_numbers):
                continue
            selected_numbers.append(int(number))
        for number in anchor_candidates:
            if len(selected_numbers) >= 2:
                break
            if int(number) not in selected_numbers:
                selected_numbers.append(int(number))

        def unrepresented_ticket_candidates(target_zone: int) -> list[int]:
            ordered: list[int] = []
            seen_numbers: set[int] = set()
            for red_set in current_final_sets:
                if red_set & set(selected_numbers):
                    continue
                zone_numbers = [
                    number
                    for number in sorted(red_set)
                    if red_zone(number) == target_zone
                    and number not in selected_numbers
                    and not is_stale_dominant(number)
                ]
                if not zone_numbers:
                    continue
                best_number = max(
                    zone_numbers,
                    key=lambda number: (
                        0.92 * quota_number_score(number, selected_numbers)
                        + 0.72
                        * (
                            sum(float(pair_support.get(tuple(sorted((number, other))), 0.0)) for other in selected_numbers)
                            / max_pair_support
                        )
                        + (
                            2.0
                            if final_support.get(number, 0.0) <= 1.05
                            and red_support.get(number, 0.0) >= 0.08 * max_support
                            else 0.0
                        )
                    ),
                )
                if best_number not in seen_numbers:
                    ordered.append(int(best_number))
                    seen_numbers.add(int(best_number))
            return ordered

        for target_zone in (3, 1):
            if len(selected_numbers) >= 4:
                break
            candidates = unrepresented_ticket_candidates(target_zone)
            if candidates:
                selected_numbers.append(int(candidates[0]))

        if len(selected_numbers) < 6:
            fusion_final_candidates = [
                number
                for number in fusion_order[: max(8, fusion_top_k)]
                if number in final_union
                and number not in selected_numbers
                and not is_stale_dominant(number)
                and number not in {1, 2, 32, 33}
            ]
            if fusion_final_candidates:
                selected_numbers.append(int(fusion_final_candidates[0]))

        high_zone_bridge_candidates = [
            number
            for number in final_union
            if number not in selected_numbers
            and not is_stale_dominant(number)
            and red_zone(number) == 3
        ]
        if high_zone_bridge_candidates:
            bridge = max(
                high_zone_bridge_candidates,
                key=lambda number: (
                    1.35
                    * (
                        sum(float(pair_support.get(tuple(sorted((number, other))), 0.0)) for other in selected_numbers)
                        / max_pair_support
                    )
                    + 0.72 * quota_number_score(number, selected_numbers)
                    + (
                        0.45
                        if final_support.get(number, 0.0) > 0
                        and red_support.get(number, 0.0) < 0.68 * max_support
                        else 0.0
                    )
                ),
            )
            selected_numbers.append(int(bridge))

        bridge_candidates = [
            number
            for number in final_union
            if number not in selected_numbers and not is_stale_dominant(number)
        ]
        if len(selected_numbers) < 4 and bridge_candidates:
            bridge = max(
                bridge_candidates,
                key=lambda number: (
                    0.95 * quota_number_score(number, selected_numbers)
                    + 0.65
                    * (
                        sum(float(pair_support.get(tuple(sorted((number, other))), 0.0)) for other in selected_numbers)
                        / max_pair_support
                    )
                ),
            )
            selected_numbers.append(int(bridge))

        if len(selected_numbers) < 6 and not any(red_zone(number) == 1 for number in selected_numbers):
            low_zone_candidates = [
                number
                for number in final_union
                if number not in selected_numbers
                and red_zone(number) == 1
                and not is_stale_dominant(number)
            ]
            if low_zone_candidates:
                selected_numbers.append(
                    int(
                        max(
                            low_zone_candidates,
                            key=lambda number: (
                                1.35 * fusion_rank_score(number, max(18, fusion_top_k))
                                + 0.42 * (float(final_support.get(number, 0.0)) / max_final_support)
                                + 0.25 * (float(red_support.get(number, 0.0)) / max_support) ** 0.5
                                - (0.65 if int(number) in {1, 2} else 0.0)
                            ),
                        )
                    )
                )

        while len(selected_numbers) < 6:
            missing_zones = [
                zone
                for zone in (1, 2, 3)
                if not any(red_zone(number) == zone for number in selected_numbers)
            ]
            zone_filtered = [
                number
                for number in final_union
                if number not in selected_numbers
                and not (is_stale_dominant(number) and len(selected_numbers) >= 3)
                and (not missing_zones or red_zone(number) in missing_zones)
            ]
            candidates = zone_filtered or [
                number
                for number in final_union
                if number not in selected_numbers
                and not (is_stale_dominant(number) and len(selected_numbers) >= 3)
            ]
            candidates = candidates or [number for number in final_union if number not in selected_numbers]
            if not candidates:
                break
            selected_numbers.append(
                int(max(candidates, key=lambda number: quota_number_score(number, selected_numbers)))
            )

        if len(set(selected_numbers)) < 6:
            return None
        return tuple(sorted(selected_numbers[:6]))

    def zone_balance(red: tuple[int, ...]) -> float:
        zones = [
            sum(1 for number in red if 1 <= number <= 11),
            sum(1 for number in red if 12 <= number <= 22),
            sum(1 for number in red if 23 <= number <= 33),
        ]
        return 1.0 - (max(zones) - min(zones)) / 6.0

    def combo_score(red: tuple[int, ...]) -> float:
        pair_score = sum(float(pair_support.get(tuple(sorted(pair)), 0.0)) for pair in combinations(red, 2))
        support_total = sum(number_score(number) for number in red)
        top_fusion_hits = sum(1 for number in red if fusion_position.get(number, 999) < fusion_top_k)
        span = max(red) - min(red)
        span_score = max(0.0, 1.0 - abs(span - 18.0) / 18.0)
        return support_total + 0.22 * pair_score + 0.55 * top_fusion_hits + 0.35 * zone_balance(red) + 0.18 * span_score

    best_red = max(combinations(candidate_numbers, 6), key=combo_score)
    if len(set(best_red)) != 6:
        return None
    blue_order = [
        number
        for number, _ in blue_support.most_common()
        if 1 <= int(number) <= 16
    ]
    for number in fusion.get("top_blue", []):
        if str(number).isdigit() and 1 <= int(number) <= 16 and int(number) not in blue_order:
            blue_order.append(int(number))
    blue = int(blue_order[0]) if blue_order else int((current_selected[0] or {}).get("blue", 1))
    quota_red = build_quota_collector_red()
    if quota_red is not None:
        return {
            "red": quota_red,
            "blue": blue,
            "quality_score": float(sum(quota_number_score(number, list(quota_red)) for number in quota_red)),
            "consensus_score": float(sum(quota_number_score(number, list(quota_red)) for number in quota_red)),
            "selection_role": "final_core_quota_collector",
            "selection_reason": {
                "reason": "cross_ticket_hit_collector_before_coverage_fill",
                "source_ticket_count": len(source_tickets),
                "candidate_pool_size": len(candidate_numbers),
                "fusion_top_k": fusion_top_k,
                "design": "2个稳定核心号 + 共现桥接号 + 跨注代表号，减少命中号分散在多注里的问题",
            },
        }
    return {
        "red": tuple(sorted(int(number) for number in best_red)),
        "blue": blue,
        "quality_score": float(combo_score(tuple(sorted(best_red)))),
        "consensus_score": float(combo_score(tuple(sorted(best_red)))),
        "selection_role": "final_core_aggregation",
        "selection_reason": {
            "reason": "method_consensus_core_aggregation_before_coverage_fill",
            "source_ticket_count": len(source_tickets),
            "candidate_pool_size": len(candidate_numbers),
            "fusion_top_k": fusion_top_k,
        },
    }


def _prepend_core_aggregate_ticket(
    selected: list[dict[str, Any]],
    core_ticket: dict[str, Any] | None,
    *,
    count: int,
) -> list[dict[str, Any]]:
    if not core_ticket or not selected:
        return selected
    core_red = tuple(sorted(int(number) for number in core_ticket.get("red", [])))
    result = [{**core_ticket, "red": core_red}]
    used_keys = {(core_red, int(core_ticket.get("blue", 0)))}
    for ticket in selected:
        key = (tuple(sorted(int(number) for number in ticket.get("red", []))), int(ticket.get("blue", 0)))
        if key in used_keys:
            continue
        result.append(ticket)
        used_keys.add(key)
        if len(result) >= int(count):
            break
    return result[: int(count)]


def candidate_experiment(
    frame: Any,
    config: dict[str, Any],
    weights_by_color: dict[str, dict[str, float]],
    weight_reason_by_color: dict[str, str],
    red_adverse_guard: dict[str, Any] | None = None,
    stacking_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    probabilities, model_pipeline = fit_current_probabilities(
        frame,
        config,
        weights_by_color,
        weight_reason_by_color,
        stacking_records=stacking_records,
    )
    exclusion_diagnostics: dict[str, Any] = {"enabled": False, "reason": "not_evaluated"}
    generation = config["generation"]
    fusion = model_pipeline["fusion"]
    runtime_control = (
        ((model_pipeline.get("fusion") or {}).get("stacking") or {}).get("v7_runtime_control")
        or build_evidence_runtime_control(stacking_records or [], config, ticket_count=int(config["optimizer"].get("consensus_count", 5)))
    )
    seed = int(config["project"]["random_seed"])
    red_uniform = bool(fusion.get("red_is_uniform"))
    weak_signal = fusion.get("prediction_mode") == "shrunk_weak_model_signal"
    red_guard_active = bool((red_adverse_guard or {}).get("active"))
    red_guard_stoploss = _red_guard_stoploss_decision(
        red_adverse_guard,
        config["optimizer"],
    )
    red_guard_stoploss_pool: list[dict[str, Any]] = []
    red_guard_stoploss_sampling: dict[str, Any] = {
        "enabled": False,
        "reason": red_guard_stoploss.get("reason", "not_evaluated"),
    }
    red_candidate_recall_protection: dict[str, Any] = {
        "schema": "candidate_recall_protection_v8",
        "enabled": False,
        "reason": "not_evaluated",
    }
    if red_guard_active:
        red_relief = float(
            config["optimizer"].get("red_adverse_guard_uniform_relief", 0.45)
        )
        model_pipeline["fusion"] = {
            **model_pipeline["fusion"],
            "red_adverse_guard": red_adverse_guard,
            "red_guarded_probability": f"tempered_red_with_original_blue_signal_relief={red_relief:.2f}",
            "prediction_mode": "red_adverse_guard_tempered_red_blue_signal",
        }
        fusion = model_pipeline["fusion"]
        weak_signal = False
    if bool(config["generation"].get("withhold_red_uniform_output", False)):
        pool = []
        sampling = {
            "mode": "red_signal_withheld_uniform",
            "reason": "red_fusion_uniform_no_publishable_core",
            "retained_candidates": 0,
            "unique_candidates": 0,
        }
        logging.info("候选生成已跳过 | 事件=MONTE_CARLO_SKIPPED | 统计=%s", sampling)
    else:
        effective_samples = int(generation["monte_carlo_samples"])
        if red_guard_active:
            effective_samples = int(generation.get(
                "red_adverse_guard_monte_carlo_samples",
                max(effective_samples, 8000),
            ))
        effective_pool_size = int(generation["candidate_pool_size"])
        if red_guard_active:
            effective_pool_size = int(generation.get(
                "red_adverse_guard_candidate_pool_size",
                max(effective_pool_size, 3000),
            ))
        with logged_step(
            "monte_carlo_generation",
            samples=effective_samples,
            pool_size=effective_pool_size,
        ):
            candidate_generation_probabilities = probabilities
            if red_guard_active:
                red_relief = float(
                    config["optimizer"].get("red_adverse_guard_uniform_relief", 0.45)
                )
                guard_probabilities = {
                    "red": _temper_with_uniform(probabilities["red"], red_relief),
                    "blue": probabilities["blue"],
                }
                guard_probabilities, exclusion_diagnostics = apply_low_confidence_exclusion(
                    guard_probabilities,
                    frame,
                    config,
                    model_rankings=model_pipeline.get("models", {}),
                )
                guard_probabilities, red_candidate_recall_protection = apply_red_candidate_recall_protection(
                    guard_probabilities,
                    model_rankings=model_pipeline.get("models", {}),
                    config=config,
                )
                candidate_generation_probabilities = guard_probabilities
                guarded_pool, guarded_sampling = generate_candidates(
                    guard_probabilities,
                    frame,
                    samples=effective_samples,
                    pool_size=effective_pool_size,
                    scoring_config=config["scoring"],
                    random_seed=seed_with_offset(seed, "red_guard_monte_carlo"),
                )
                include_uniform_coverage = bool(config["generation"].get(
                    "red_adverse_guard_include_uniform_coverage_pool",
                    False,
                ))
                coverage_pool: list[dict[str, Any]] = []
                coverage_sampling: dict[str, Any] = {
                    "enabled": False,
                    "reason": "uniform_coverage_pool_kept_as_baseline_only",
                }
                if red_guard_stoploss.get("active"):
                    red_guard_stoploss_pool, red_guard_stoploss_sampling = generate_candidates(
                        _uniform_probabilities(),
                        frame,
                        samples=int(config["generation"].get(
                            "red_adverse_guard_coverage_samples",
                            max(int(config["generation"]["monte_carlo_samples"]), effective_samples // 2),
                        )),
                        pool_size=int(config["generation"].get(
                            "red_adverse_guard_coverage_pool_size",
                            max(int(config["generation"]["candidate_pool_size"]), effective_pool_size // 2),
                        )),
                        scoring_config=config["scoring"],
                        random_seed=seed_with_offset(seed, "stoploss_pool"),
                    )
                    red_guard_stoploss_sampling = {
                        **red_guard_stoploss_sampling,
                        "enabled": True,
                        "decision": red_guard_stoploss,
                    }
                if include_uniform_coverage:
                    coverage_samples = int(config["generation"].get(
                        "red_adverse_guard_coverage_samples",
                        max(int(config["generation"]["monte_carlo_samples"]), effective_samples // 2),
                    ))
                    coverage_pool_size = int(config["generation"].get(
                        "red_adverse_guard_coverage_pool_size",
                        max(int(config["generation"]["candidate_pool_size"]), effective_pool_size // 2),
                    ))
                    coverage_pool, coverage_sampling = generate_candidates(
                        _uniform_probabilities(),
                        frame,
                        samples=coverage_samples,
                        pool_size=coverage_pool_size,
                        scoring_config=config["scoring"],
                        random_seed=seed_with_offset(seed, "uniform_coverage_pool"),
                    )
                pool = _merge_candidate_pools(guarded_pool, coverage_pool)
                sampling = {
                    **guarded_sampling,
                    "mode": (
                        "red_adverse_guard_hybrid_signal_plus_coverage"
                        if include_uniform_coverage
                        else "red_adverse_guard_tempered_signal_only"
                    ),
                    "reason": (
                        "red_models_underperformed_matched_baseline; "
                        f"red_signal_tempered_toward_uniform_relief={red_relief:.2f}; "
                        + (
                            "merged_independent_uniform_coverage_candidates; "
                            if include_uniform_coverage
                            else "uniform_coverage_pool_used_as_baseline_not_candidates; "
                        )
                        + "blue_signal_is_retained"
                    ),
                    "red_uniform_relief": red_relief,
                    "guarded_pool_candidates": len(guarded_pool),
                    "coverage_pool_candidates": len(coverage_pool),
                    "stoploss_pool_candidates": len(red_guard_stoploss_pool),
                    "merged_candidate_pool": len(pool),
                    "coverage_sampling": coverage_sampling,
                    "stoploss_sampling": red_guard_stoploss_sampling,
                    "red_adverse_guard": red_adverse_guard,
                    "red_guard_stoploss": red_guard_stoploss,
                    "red_candidate_recall_protection": red_candidate_recall_protection,
                }
            elif weak_signal and not red_uniform:
                probabilities, exclusion_diagnostics = apply_low_confidence_exclusion(
                    probabilities,
                    frame,
                    config,
                    model_rankings=model_pipeline.get("models", {}),
                )
                probabilities, red_candidate_recall_protection = apply_red_candidate_recall_protection(
                    probabilities,
                    model_rankings=model_pipeline.get("models", {}),
                    config=config,
                )
                candidate_generation_probabilities = probabilities
                pool, sampling = generate_core_candidates(
                    probabilities,
                    fusion,
                    frame,
                    pool_size=int(generation["candidate_pool_size"]),
                    scoring_config=config["scoring"],
                    model_rankings=model_pipeline.get("models", {}),
                    red_core_size=int(
                        generation.get("weak_signal_red_core_size", 9)
                    ),
                    red_anchor_size=int(
                        generation.get("weak_signal_red_anchor_size", 6)
                    ),
                    max_red_changes=int(
                        generation.get("weak_signal_max_red_changes", 2)
                    ),
                    min_anchor_kept=int(
                        generation.get("weak_signal_min_anchor_kept", 0)
                    ),
                    blue_core_size=int(
                        generation.get("weak_signal_blue_core_size", 3)
                    ),
                )
            else:
                probabilities, exclusion_diagnostics = apply_low_confidence_exclusion(
                    probabilities,
                    frame,
                    config,
                    model_rankings=model_pipeline.get("models", {}),
                )
                probabilities, red_candidate_recall_protection = apply_red_candidate_recall_protection(
                    probabilities,
                    model_rankings=model_pipeline.get("models", {}),
                    config=config,
                )
                candidate_generation_probabilities = probabilities
                pool, sampling = generate_candidates(
                    probabilities,
                    frame,
                    samples=int(generation["monte_carlo_samples"]),
                    pool_size=int(generation["candidate_pool_size"]),
                    scoring_config=config["scoring"],
                    random_seed=seed,
                )
            pool, slot_enforcement = enforce_red_candidate_recall_slots(
                pool,
                candidate_generation_probabilities,
                frame,
                red_candidate_recall_protection,
                pool_size=max(int(generation["candidate_pool_size"]), len(pool)),
                scoring_config=config["scoring"],
            )
            red_candidate_recall_protection = {
                **red_candidate_recall_protection,
                "candidate_slot_enforcement": slot_enforcement,
            }
            sampling = {
                **sampling,
                "exclusion_filter": exclusion_diagnostics,
                "red_candidate_recall_protection": red_candidate_recall_protection,
            }
            model_pipeline["exclusion_filter"] = exclusion_diagnostics
            model_pipeline["red_candidate_recall_protection"] = red_candidate_recall_protection
            logging.info("蒙特卡洛生成完成 | 事件=MONTE_CARLO_DONE | 统计=%s", compact_sampling_log(sampling))
    selected: dict[str, list[dict]] = {}
    coverage: dict[str, dict] = {}
    for method in config["optimizer"]["methods"]:
        with logged_step("coverage_optimization", method=method, output_count=generation["output_count"]):
            tickets = optimize_candidates(
                pool,
                int(generation["output_count"]),
                method,
                config["optimizer"],
                config["scoring"],
                seed,
                history_frame=frame,
            ) if pool else []
            selected[method] = tickets
            coverage[method] = coverage_summary(tickets)
            logging.info("覆盖优化完成 | 事件=OPTIMIZER_DONE | 方法=%s | 覆盖统计=%s", method, coverage[method])
    consensus_count = int(generation.get("consensus_output_count", generation["output_count"]))
    final5_hit_info: dict[str, Any] = {"schema": "final5_hit_coherence_v11_7", "active": False}
    with logged_step("consensus_selection", output_count=consensus_count):
        no_signal = bool(model_pipeline["fusion"].get("is_uniform"))
        consensus_scoring = {
            **config["scoring"],
            "signal_red_consensus_weight": float(
                config["optimizer"].get("signal_red_consensus_weight", 0.0)
            ),
            "signal_red_consensus_pool_size": int(
                config["optimizer"].get("signal_red_consensus_pool_size", 20)
            ),
            "signal_blue_consensus_pool_size": int(
                config["optimizer"].get("signal_blue_consensus_pool_size", 20)
            ),
            "allow_blue_unique_relaxation": bool(
                config["optimizer"].get("allow_blue_unique_relaxation", False)
            ),
            "blue_unique_relaxation_min_strength": float(
                config["optimizer"].get("blue_unique_relaxation_min_strength", 0.85)
            ),
            "blue_unique_relaxation_floor": int(
                config["optimizer"].get("blue_unique_relaxation_floor", 4)
            ),
        }
        if not pool:
            consensus_selected = []
            selection_strategy = "empty_candidate_pool"
        elif red_guard_active:
            consensus_selected = adverse_guard_coverage_optimize(
                pool,
                selected,
                consensus_count,
                max_red_overlap=int(
                    config["optimizer"].get("red_adverse_guard_max_red_overlap", 4)
                ),
                minimum_blue_unique_count=int(
                    config["optimizer"].get("red_adverse_guard_minimum_blue_unique_count", consensus_count)
                ),
                minimum_red_coverage=int(
                    config["optimizer"].get("red_adverse_guard_minimum_red_coverage", max(8, consensus_count + 3))
                ),
                red_signal_weight=float(
                    config["optimizer"].get("red_adverse_guard_red_signal_weight", 0.08)
                ),
                blue_signal_weight=float(
                    config["optimizer"].get("red_adverse_guard_blue_signal_weight", 0.55)
                ),
                method_anchor_count=int(
                    config["optimizer"].get("red_adverse_guard_method_anchor_count", 2)
                ),
                method_anchor_top_k_per_method=int(
                    config["optimizer"].get("red_adverse_guard_method_anchor_top_k_per_method", 2)
                ),
                random_seed=seed_with_offset(seed, "weak_signal_monte_carlo"),
            )
            selection_strategy = "red_adverse_guard_coverage_matrix"
        elif no_signal:
            consensus_selected = rotational_coverage_optimize(
                pool,
                consensus_count,
                max_red_overlap=int(config["optimizer"].get("max_red_overlap", 4)),
                minimum_blue_unique_count=int(
                    config["optimizer"].get(
                        "minimum_blue_unique_count",
                        min(consensus_count, 3),
                    )
                ),
                random_seed=seed_with_offset(seed, "weak_signal_core"),
            )
            selection_strategy = "no_signal_rotational_coverage"
        else:
            consensus_selected = consensus_optimize(
                pool,
                selected,
                consensus_count,
                scoring_config=consensus_scoring,
                max_red_overlap=int(
                    config["optimizer"].get("signal_max_red_overlap", config["optimizer"].get("max_red_overlap", 4))
                ),
                blue_diversity_weight=float(
                    config["optimizer"].get("signal_blue_diversity_weight", config["optimizer"].get("blue_diversity_weight", 0.30))
                ),
                minimum_blue_unique_count=int(
                    config["optimizer"].get(
                        "signal_minimum_blue_unique_count",
                        config["optimizer"].get("minimum_blue_unique_count", 3),
                    )
                ),
                concentration_weight=float(config["optimizer"].get("signal_concentration_weight", 0.0)),
                minimum_unique_red_sets=int(
                    config["optimizer"].get("signal_minimum_unique_red_sets", consensus_count)
                ),
                minimum_red_coverage=int(
                    config["optimizer"].get("signal_minimum_red_coverage", max(6, consensus_count + 5))
                ),
                preserve_top_k=int(
                    config["optimizer"].get("signal_preserve_top_ticket_count", 0)
                ),
                red_coverage_relief_max=float(
                    config["optimizer"].get("signal_red_coverage_relief_max", 0.0)
                ),
                red_set_relief_max=float(
                    config["optimizer"].get("signal_red_set_relief_max", 0.0)
                ),
                blue_focus_bonus_weight=float(
                    config["optimizer"].get("signal_blue_focus_bonus_weight", 0.0)
                ),
                blue_unique_relief_max=float(
                    config["optimizer"].get("signal_blue_unique_relief_max", 0.0)
                ),
                red_core_bonus_weight=float(
                    config["optimizer"].get("signal_red_core_bonus_weight", 0.0)
                ),
                red_core_top_k=int(
                    config["optimizer"].get("signal_red_core_top_k", 6)
                ),
                force_method_anchor_count=int(
                    config["optimizer"].get("signal_force_method_anchor_count", 0)
                ),
                force_method_anchor_top_k_per_method=int(
                    config["optimizer"].get("signal_force_method_anchor_top_k_per_method", 1)
                ),
                force_method_anchor_min_support=int(
                    config["optimizer"].get("signal_force_method_anchor_min_support", 1)
                ),
                method_agreement_bonus_weight=float(
                    config["scoring"].get("signal_method_agreement_bonus_weight", 0.0)
                ),
                top_consensus_margin_threshold=float(
                    config["optimizer"].get("signal_top_consensus_margin_threshold", 0.0)
                ),
                top_consensus_overlap_bonus_weight=float(
                    config["optimizer"].get("signal_top_consensus_overlap_bonus_weight", 0.0)
                ),
                allow_concentrated_portfolio=bool(
                    config["optimizer"].get("signal_allow_concentrated_portfolio", False)
                ),
                concentrated_ticket_count=int(
                    config["optimizer"].get("signal_concentrated_ticket_count", max(1, consensus_count // 2))
                ),
            )
            selection_strategy = "model_signal_consensus"
        recommended_method = str((red_adverse_guard or {}).get("recommended_method") or "")
        if (
            red_guard_stoploss.get("active")
            and red_guard_stoploss_pool
        ):
            consensus_selected = rotational_coverage_optimize(
                red_guard_stoploss_pool,
                consensus_count,
                max_red_overlap=int(config["optimizer"].get("max_red_overlap", 4)),
                minimum_blue_unique_count=int(
                    config["optimizer"].get(
                        "signal_minimum_blue_unique_count",
                        config["optimizer"].get("minimum_blue_unique_count", 3),
                    )
                ),
                random_seed=seed_with_offset(seed, "dantuo_random"),
            )
            consensus_selected = [
                {
                    **ticket,
                    "selection_role": "red_guard_no_signal_stoploss",
                    "selection_reason": red_guard_stoploss,
                }
                for ticket in consensus_selected
            ]
            selection_strategy = "red_guard_no_signal_stoploss"
            logging.info(
                "红球逆信号止损最终层接管 | 事件=RED_GUARD_STOPLOSS_SELECTED | 原因=%s | 注数=%s",
                red_guard_stoploss,
                len(consensus_selected),
            )
        elif (
            bool(config["optimizer"].get("rolling_best_method_enabled", True))
            and recommended_method
            and recommended_method in selected
            and selected.get(recommended_method)
        ):
            consensus_selected = [
                {
                    **ticket,
                    "selection_role": f"backtest_best_method_{recommended_method}",
                    "selection_reason": {
                        "reason": (red_adverse_guard or {}).get("recommendation_reason"),
                        "strategy_scores": (red_adverse_guard or {}).get("strategy_scores", {}),
                    },
                }
                for ticket in selected[recommended_method][:consensus_count]
            ]
            selection_strategy = f"backtest_best_method_{recommended_method}"
            logging.info(
                "回测择优最终层接管 | 事件=BACKTEST_BEST_METHOD_SELECTED | 方法=%s | 原因=%s",
                recommended_method,
                (red_adverse_guard or {}).get("recommendation_reason"),
            )
        elif (
            red_guard_active
            and bool(config["optimizer"].get("red_guard_prefer_method_portfolio", True))
        ):
            fallback_method = str(config["optimizer"].get("red_guard_default_method", "greedy"))
            if fallback_method in selected and selected.get(fallback_method):
                consensus_selected = [
                    {
                        **ticket,
                        "selection_role": f"red_guard_method_portfolio_{fallback_method}",
                        "selection_reason": {
                            "reason": "red_guard_prefers_existing_method_portfolio_over_new_coverage_matrix",
                            "guard": red_adverse_guard,
                        },
                    }
                    for ticket in selected[fallback_method][:consensus_count]
                ]
                selection_strategy = f"red_guard_method_portfolio_{fallback_method}"
                logging.info(
                    "红球保护最终层采用既有方法组合 | 事件=RED_GUARD_METHOD_PORTFOLIO_SELECTED | 方法=%s | 原因=%s",
                    fallback_method,
                    (red_adverse_guard or {}).get("reason"),
                )
        v7_portfolio_info = {"active": False, "reason": "v7_portfolio_disabled"}
        if bool(config.get("v7", {}).get("enabled", True)) and bool(
            config.get("v7", {}).get("red", {}).get("dynamic_portfolio", True)
        ) and pool:
            protected_numbers = sorted({
                *[
                    int(number)
                    for number in runtime_control.get("red_protected_numbers", []) or []
                    if 1 <= int(number) <= 33
                ],
                *[
                    int(number)
                    for number in red_candidate_recall_protection.get("protected_numbers", []) or []
                    if 1 <= int(number) <= 33
                ],
            })
            portfolio_config = {
                **config,
                "v7": {
                    **(config.get("v7") or {}),
                    "current_red_edge_state": runtime_control.get("red_budget_state", runtime_control.get("red_edge_state", "NO_EDGE")),
                    "red_budget": runtime_control.get("red_budget", {}),
                    "red_protected_numbers": protected_numbers,
                    "red_candidate_recall_protection": red_candidate_recall_protection,
                    "red_runtime_control": runtime_control,
                },
            }
            consensus_selected, v7_portfolio_info = select_red_portfolio(
                pool,
                selected,
                consensus_count,
                portfolio_config,
            )
            selection_strategy = f"{selection_strategy}_v7_portfolio"
            logging.info(
                "V7红球Portfolio已接管当前预测最终层 | 事件=V7_RED_PORTFOLIO | 诊断=%s",
                v7_portfolio_info,
            )
        consensus_coverage = coverage_summary(consensus_selected)
        if no_signal:
            coverage["rotation_fallback"] = consensus_coverage
        if v7_portfolio_info.get("active"):
            rebalance_info = {"active": False, "reason": "v7_portfolio_is_final_layer"}
        else:
            consensus_selected, rebalance_info = rebalance_final_portfolio(
                pool,
                consensus_selected,
                count=consensus_count,
                optimizer_config=config["optimizer"],
                random_seed=seed_with_offset(seed, "final_diversity_rebalance"),
                reason_prefix="final_diversity_rebalance",
            )
        if rebalance_info.get("active"):
            selection_strategy = "final_diversity_rebalance"
            consensus_coverage = coverage_summary(consensus_selected)
            logging.info(
                "最终组合覆盖再平衡接管 | 事件=FINAL_DIVERSITY_REBALANCE | 诊断=%s",
                rebalance_info,
            )
        core_ticket = None if v7_portfolio_info.get("active") else _core_aggregate_ticket(
            selected,
            consensus_selected,
            model_pipeline.get("fusion", {}),
            config["optimizer"],
        )
        updated_consensus = _prepend_core_aggregate_ticket(
            consensus_selected,
            core_ticket,
            count=consensus_count,
        )
        if updated_consensus != consensus_selected:
            consensus_selected = updated_consensus
            consensus_coverage = coverage_summary(consensus_selected)
            selection_strategy = f"{selection_strategy}_with_core_aggregation"
            logging.info(
                "最终组合核心聚合注已前置 | 事件=FINAL_CORE_AGGREGATION | 红球=%s | 蓝球=%s | 覆盖统计=%s",
                core_ticket.get("red") if core_ticket else None,
                core_ticket.get("blue") if core_ticket else None,
                consensus_coverage,
            )
        consensus_selected, rolling_memory_info = apply_rolling_portfolio_memory(
            consensus_selected,
            pool,
            stacking_records or [],
            config=config["optimizer"],
        )
        if rolling_memory_info.get("active"):
            consensus_coverage = coverage_summary(consensus_selected)
            selection_strategy = f"{selection_strategy}_with_rolling_memory"
            logging.info(
                "滚动组合记忆已参与最终组合 | 事件=ROLLING_PORTFOLIO_MEMORY_APPLIED | 诊断=%s | 覆盖统计=%s",
                rolling_memory_info,
                consensus_coverage,
            )
        hit_cfg = ((config.get("v11_7") or {}).get("hit_coherence") or {})
        if bool(hit_cfg.get("enabled", True)) and bool(hit_cfg.get("apply_to_final5", True)) and consensus_selected:
            red_width = int(
                ((config.get("v7") or {}).get("red") or {}).get(
                    "wide_candidate_limit",
                    ((config.get("v7") or {}).get("red") or {}).get("candidate_set_limit", 22),
                )
                or 22
            )
            consensus_selected, final5_hit_info = apply_hit_coherence_final5(
                consensus_selected,
                fusion_red=probabilities.get("red"),
                history_draws=red_draws_from_frame(frame),
                method_tickets=selected,
                pool=pool,
                model_rankings=model_pipeline.get("models"),
                rescue_numbers=red_candidate_recall_protection.get("protected_numbers", []),
                width=int(hit_cfg.get("default_width", red_width) or red_width),
                count=consensus_count,
            )
            final5_hit_info.pop("prepared", None)
            if final5_hit_info.get("active"):
                selection_strategy = f"{selection_strategy}_v11_7_{final5_hit_info.get('strategy')}"
                consensus_coverage = coverage_summary(consensus_selected)
                logging.info(
                    "当前预测Final5改为命中共现组票 | 事件=FINAL5_HIT_COHERENCE_SELECTED | 策略=%s | 宽度=%s | 共现组=%s",
                    final5_hit_info.get("strategy"),
                    final5_hit_info.get("width"),
                    final5_hit_info.get("groups"),
                )
        effective_blue_upgrade = config.get("upgrade", {})
        if bool((config.get("v7") or {}).get("enabled", True)) and bool(
            ((config.get("v7") or {}).get("blue") or {}).get("adaptive_top_k", True)
        ):
            blue_edge_state_for_policy = str(runtime_control.get("blue_edge_state", "NO_EDGE")).upper()
            blue_mode_by_state = (
                ((config.get("v7") or {}).get("blue") or {}).get("edge_coverage_mode_by_state")
                or {}
            )
            effective_blue_upgrade = {
                **effective_blue_upgrade,
                "blue_coverage_candidate_top_n": int(runtime_control.get("blue_candidate_width", effective_blue_upgrade.get("blue_coverage_candidate_top_n", 10))),
                "blue_coverage_policy": str(
                    blue_mode_by_state.get(
                        blue_edge_state_for_policy,
                        "no_edge_coverage_only"
                        if blue_edge_state_for_policy == "NO_EDGE"
                        else "weak_edge_mixed_coverage"
                        if blue_edge_state_for_policy == "WEAK_EDGE"
                        else "probability_optimized",
                    )
                ),
                "blue_no_edge_fast_runtime_path": True,
                "_v7_blue_prediction_set": runtime_control.get("blue_prediction_set", {}),
                "_v7_blue_topk_policy": (runtime_control.get("blue_prediction_set", {}) or {}).get("policy", {}),
                "_v7_blue_edge_report": runtime_control.get("blue_edge", {}),
                "_v7_runtime_control": runtime_control,
            }
        consensus_selected, blue_finalize_info = finalize_blue_portfolio(
            consensus_selected,
            frame,
            optimizer_config={
                **config["optimizer"],
                "_blue_research_cache_context": "production",
            },
            upgrade_config=effective_blue_upgrade,
            exclusion_diagnostics=exclusion_diagnostics,
        )
        if blue_finalize_info.get("active"):
            consensus_coverage = coverage_summary(consensus_selected)
            logging.info(
                "蓝球最终链路已执行 | 事件=BLUE_FINALIZE_PORTFOLIO_SELECTED | 注数=%s | 蓝球=%s | 诊断=%s",
                len(consensus_selected),
                [ticket.get("blue") for ticket in consensus_selected],
                blue_finalize_info,
            )
        final_blue_target = int(
            config["optimizer"].get(
                "signal_minimum_blue_unique_count",
                config["optimizer"].get("minimum_blue_unique_count", consensus_count),
            )
        )
        consensus_selected = _ensure_minimum_blue_diversity(
            consensus_selected,
            pool,
            minimum_unique_blues=min(consensus_count, final_blue_target, 16),
        )
        consensus_selected, final_v7_guard_info = apply_final_v7_guard(
            consensus_selected,
            pool,
            runtime_control=runtime_control,
            minimum_unique_blues=min(consensus_count, final_blue_target, 16),
            v7_portfolio_info=v7_portfolio_info,
        )
        if final_v7_guard_info.get("active"):
            logging.info(
                "当前预测Final V7 Guard已执行 | 事件=FINAL_V7_GUARD | 诊断=%s",
                final_v7_guard_info,
            )
        consensus_coverage = coverage_summary(consensus_selected)
        logging.info(
            "最终组合筛选完成 | 事件=CONSENSUS_DONE | 策略=%s | 注数=%s | 覆盖统计=%s | 号码=%s",
            selection_strategy,
            len(consensus_selected),
            consensus_coverage,
            [
                {
                    "red": ticket["red"],
                    "blue": ticket["blue"],
                    "score": ticket.get("consensus_score"),
                    "support": ticket.get("method_support", []),
                }
                for ticket in consensus_selected
            ],
        )
    random_tickets = generate_uniform_tickets(
        int(generation["output_count"]),
        np.random.default_rng(seed_with_offset(seed, "uniform_current")),
    )
    dantuo_config = config.get("dantuo", {})
    latest_row = frame.iloc[-1] if len(frame) else None
    recent_red = (
        [int(latest_row[f"r{index}"]) for index in range(1, 7)]
        if latest_row is not None
        else []
    )
    recent_blue = int(latest_row["blue"]) if latest_row is not None else None
    dantuo_source_tickets = list(consensus_selected)
    if bool(dantuo_config.get("include_method_tickets", True)):
        method_limit = max(1, int(dantuo_config.get("method_ticket_limit_per_method", 5)))
        for method, method_tickets in selected.items():
            for ticket in list(method_tickets or [])[:method_limit]:
                dantuo_source_tickets.append(
                    {
                        **ticket,
                        "selection_role": f"dantuo_method_source_{method}",
                        "source_method": method,
                    }
                )
    dantuo_plan = build_dantuo_plan(
        dantuo_source_tickets,
        final_recommendation=consensus_selected,
        candidate_pool=pool,
        candidate_pool_ticket_limit=int(dantuo_config.get("candidate_pool_ticket_limit", 80)),
        fusion=model_pipeline.get("fusion", {}),
        recent_red=recent_red,
        recent_blue=recent_blue,
        blue_history=frame["blue"].tolist() if "blue" in frame else None,
        red_priority_numbers=(model_pipeline.get("fusion") or {}).get("top_red") or [],
        max_amount=int(dantuo_config.get("max_amount", 20)),
        unit_price=int(dantuo_config.get("unit_price", 2)),
        preferred_red_dan_count=int(dantuo_config.get("preferred_red_dan_count", 2)),
        preferred_red_tuo_count=int(dantuo_config.get("preferred_red_tuo_count", 5)),
        preferred_blue_count=int(dantuo_config.get("preferred_blue_count", 2)),
        prefer_compound_when_coverage_like=bool(
            dantuo_config.get("prefer_compound_when_coverage_like", True)
        ),
        preserve_final_portfolio_evidence=bool(
            dantuo_config.get("preserve_final_portfolio_evidence", True)
        ),
        final_to_dantuo_audit_enabled=bool(
            dantuo_config.get("final_to_dantuo_audit_enabled", True)
        ),
        compound_red_count=int(dantuo_config.get("compound_red_count", 7)),
        compound_blue_count=int(dantuo_config.get("compound_blue_count", 1)),
        blue_recent_gap_weight=float(dantuo_config.get("blue_recent_gap_weight", 1.00)),
        blue_recent_frequency_weight=float(dantuo_config.get("blue_recent_frequency_weight", 0.00)),
        blue_evidence_weight=float(dantuo_config.get("blue_evidence_weight", 1.00)),
        blue_specialist_weight=float(dantuo_config.get("blue_specialist_weight", 0.20)),
        blue_omission_balance_weight=float(dantuo_config.get("blue_omission_balance_weight", 0.80)),
        blue_immediate_repeat_penalty=float(dantuo_config.get("blue_immediate_repeat_penalty", 0.35)),
        blue_recent_windows=dantuo_config.get("blue_recent_windows", [20, 50, 100]),
        auto_cap_blue_to_budget=bool(dantuo_config.get("auto_cap_blue_to_budget", True)),
    )
    core_numbers = compute_core_numbers(
        probabilities=probabilities,
        model_pipeline=model_pipeline,
        final_tickets=consensus_selected,
        method_tickets=selected,
        config=config,
    )
    model_pipeline["calibrated_parameters"] = parameter_snapshot(config)
    runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    candidate_pool_hash = None
    selected_hash = None
    consensus_hash = None
    if bool(runtime_config.get("prediction_fingerprint_enabled", True)):
        candidate_pool_hash = stable_hash(pool)
        selected_hash = stable_hash(selected)
        consensus_hash = stable_hash(consensus_selected)
    return {
        "sampling": sampling,
        "candidate_pool_hash": candidate_pool_hash,
        "selected_by_method_hash": selected_hash,
        "consensus_prediction_hash": consensus_hash,
        "weights": model_pipeline["fusion"]["weights"],
        "weights_by_color": weights_by_color,
        "coverage": coverage,
        "random_coverage": coverage_summary(random_tickets),
        "model_pipeline": model_pipeline,
        "runtime_control": runtime_control,
        "selected": selected,
        "consensus_selected": consensus_selected,
        "consensus_coverage": consensus_coverage,
        "selection_strategy": selection_strategy,
        "rolling_memory": rolling_memory_info,
        "final_v7_guard": final_v7_guard_info,
        "v7_portfolio": v7_portfolio_info,
        "final5_hit_coherence": final5_hit_info,
        "dantuo_plan": dantuo_plan,
        "core_numbers": core_numbers,
    }


def red_adverse_signal_guard(backtest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Detect when red-side model signals are empirically acting backwards.

    Lottery draws remain stochastic, so the guard does not claim a better
    forecast.  It prevents the final portfolio from concentrating around a
    red-number model family that just failed the same left-truncated test used
    for reporting.
    """
    guard_config = config.get("red_adverse_guard", {})
    if not bool(guard_config.get("enabled", True)):
        return {"active": False, "reason": "disabled"}

    summary = backtest.get("summary", {}) or {}
    minimum_periods = int(guard_config.get("minimum_periods", 20))
    margin = float(guard_config.get("red_average_margin", 0.03))
    period_margin = float(guard_config.get("period_any_red_ge3_margin", 0.03))
    severe_average_multiplier = float(guard_config.get("severe_average_margin_multiplier", 2.0))
    severe_period_multiplier = float(guard_config.get("severe_period_any_margin_multiplier", 2.0))
    require_consensus_adverse = bool(guard_config.get("require_consensus_adverse", True))
    activate_on_soft_adverse = bool(guard_config.get("activate_on_soft_adverse", True))

    method_diagnostics: dict[str, dict[str, float | bool]] = {}
    adverse_methods = 0
    severe_methods = 0
    strategy_scores: dict[str, dict[str, float]] = {}
    for method, values in summary.items():
        periods = int(values.get("periods", 0) or 0)
        if periods < minimum_periods:
            continue
        model_avg = float(values.get("model_average_red_hits", 0.0) or 0.0)
        matched_avg = float(values.get("matched_no_signal_average_red_hits", 0.0) or 0.0)
        random_avg = float(values.get("random_average_red_hits", 0.0) or 0.0)
        model_any = float(values.get("period_any_red_ge3_ratio", 0.0) or 0.0)
        matched_any = float(values.get("matched_no_signal_period_any_red_ge3_ratio", 0.0) or 0.0)
        random_any = float(values.get("random_period_any_red_ge3_ratio", 0.0) or 0.0)
        reference_avg = max(matched_avg, random_avg)
        reference_any = max(matched_any, random_any)
        average_failed = model_avg + margin < reference_avg
        period_failed = model_any + period_margin < reference_any
        average_shortfall = reference_avg - model_avg
        period_any_shortfall = reference_any - model_any
        severe_average_failed = average_shortfall >= margin * severe_average_multiplier
        severe_period_failed = (
            period_any_shortfall >= period_margin * severe_period_multiplier
            and average_shortfall > 0
        )
        adverse = bool(
            (average_failed and period_failed)
            or severe_period_failed
        )
        severe = bool(
            (severe_average_failed and period_failed)
            or severe_period_failed
        )
        if adverse:
            adverse_methods += 1
        if severe:
            severe_methods += 1
        strategy_scores[str(method)] = {
            "score": model_avg + 0.22 * model_any + 0.08 * float(values.get("period_any_blue_hit_ratio", 0.0) or 0.0) + 0.05 * float(values.get("average_best_red_hits", 0.0) or 0.0),
            "model_average_red_hits": model_avg,
            "period_any_red_ge3_ratio": model_any,
            "average_best_red_hits": float(values.get("average_best_red_hits", 0.0) or 0.0),
            "period_any_blue_hit_ratio": float(values.get("period_any_blue_hit_ratio", 0.0) or 0.0),
        }
        method_diagnostics[str(method)] = {
            "periods": float(periods),
            "model_average_red_hits": model_avg,
            "reference_average_red_hits": reference_avg,
            "average_shortfall": average_shortfall,
            "model_period_any_red_ge3_ratio": model_any,
            "reference_period_any_red_ge3_ratio": reference_any,
            "period_any_shortfall": period_any_shortfall,
            "average_failed": average_failed,
            "period_any_failed": period_failed,
            "severe_average_failed": severe_average_failed,
            "severe_period_any_failed": severe_period_failed,
            "severe_adverse": severe,
            "adverse": adverse,
        }

    required_methods = int(guard_config.get("minimum_adverse_methods", 2))
    consensus_adverse = bool(method_diagnostics.get("consensus", {}).get("adverse", False))
    consensus_severe = bool(method_diagnostics.get("consensus", {}).get("severe_adverse", False))
    severe_required_methods = int(guard_config.get("minimum_severe_methods", required_methods))
    soft_active = adverse_methods >= required_methods and (not require_consensus_adverse or consensus_adverse)
    severe_active = (
        soft_active
        and severe_methods >= severe_required_methods
        and (not require_consensus_adverse or consensus_severe)
    )
    active = bool(severe_active or (activate_on_soft_adverse and soft_active))
    recommended_method = None
    recommendation_reason = "no_alternative_outperformed_consensus"
    if "consensus" in strategy_scores:
        alternatives = [
            method for method in strategy_scores
            if method != "consensus"
        ]
        if alternatives:
            best_method = max(
                alternatives,
                key=lambda method: (
                    strategy_scores[method]["score"],
                    strategy_scores[method]["model_average_red_hits"],
                    strategy_scores[method]["average_best_red_hits"],
                ),
            )
            score_margin = strategy_scores[best_method]["score"] - strategy_scores["consensus"]["score"]
            average_margin = (
                strategy_scores[best_method]["model_average_red_hits"]
                - strategy_scores["consensus"]["model_average_red_hits"]
            )
            period_any_margin = (
                strategy_scores[best_method]["period_any_red_ge3_ratio"]
                - strategy_scores["consensus"]["period_any_red_ge3_ratio"]
            )
            reference_advantage = (
                method_diagnostics.get("consensus", {}).get("average_shortfall", 0.0)
                - method_diagnostics.get(best_method, {}).get("average_shortfall", 0.0)
            )
            optimizer_config = config.get("optimizer", {})
            if (
                (
                    score_margin >= float(optimizer_config.get("rolling_best_method_score_margin", 0.08))
                    and average_margin >= float(optimizer_config.get("rolling_best_method_average_margin", 0.12))
                    and period_any_margin >= float(optimizer_config.get("rolling_best_method_period_any_margin", 0.03))
                )
                and (
                    not bool(optimizer_config.get("rolling_best_method_require_reference_advantage", True))
                    or reference_advantage > 0.0
                )
            ):
                recommended_method = best_method
                recommendation_reason = "backtest_method_outperformed_consensus"
            else:
                recommendation_reason = "consensus_not_worse_than_backtest_alternatives"
    if active:
        if severe_active:
            reason = f"red_models_severely_underperformed_baselines_in_{severe_methods}_methods"
        else:
            reason = f"red_models_soft_adverse_activated_in_{adverse_methods}_methods"
    elif soft_active:
        reason = "red_models_soft_adverse_diagnostic_only"
    elif adverse_methods >= required_methods and require_consensus_adverse and not consensus_adverse:
        reason = "red_guard_blocked_consensus_not_adverse"
    else:
        reason = "red_models_not_consistently_adverse"
    return {
        "active": active,
        "reason": reason,
        "adverse_methods": adverse_methods,
        "severe_methods": severe_methods,
        "required_methods": required_methods,
        "severe_required_methods": severe_required_methods,
        "soft_active": soft_active,
        "severe_active": severe_active,
        "activate_on_soft_adverse": activate_on_soft_adverse,
        "minimum_periods": minimum_periods,
        "average_margin": margin,
        "period_any_margin": period_margin,
        "require_consensus_adverse": require_consensus_adverse,
        "recommended_method": recommended_method,
        "recommendation_reason": recommendation_reason,
        "strategy_scores": strategy_scores,
        "methods": method_diagnostics,
    }


def run_pipeline(config: dict[str, Any], periods: int | None = None) -> dict[str, Any]:
    from src.utils.idle_sleep import prevent_idle_sleep

    with prevent_idle_sleep():
        return _run_pipeline(config, periods=periods)


def _run_pipeline(config: dict[str, Any], periods: int | None = None) -> dict[str, Any]:
    RUNTIME_PROFILER.reset()
    pipeline_started = time.perf_counter()
    logging.info(
        "全流程启动 | 事件=PIPELINE_START | 回测期数=%s | 蒙特卡洛次数=%s | 每期输出注数=%s | 随机种子=%s",
        periods or config["backtest"]["periods"],
        config["generation"]["monte_carlo_samples"],
        config["generation"]["output_count"],
        config["project"]["random_seed"],
    )
    signing_key = audit_signing_key(config)
    code_fingerprint = source_tree_fingerprint(ROOT)
    runtime_meta = runtime_metadata(config)
    frame, clean_report = prepare_data(config)
    data_manifest = clean_report.get("data_manifest", {})
    run_manifest = build_run_manifest(
        config_hash=config_fingerprint(config),
        code_fingerprint=code_fingerprint,
        seed=int(config["project"]["random_seed"]),
        data_manifest=data_manifest,
        runtime_metadata=runtime_meta,
    )
    logging.info(
        "运行清单已生成 | 事件=RUN_MANIFEST | 数据版本=%s | 配置哈希=%s | 代码指纹=%s",
        run_manifest.get("data_version_id"),
        run_manifest.get("config_hash"),
        run_manifest.get("source_tree_sha256"),
    )
    audit_config = config["randomness_audit"]
    if bool(audit_config.get("enabled", True)):
        with logged_step(
            "randomness_audit",
            periods=len(frame),
            autocorrelation_lags=audit_config.get("autocorrelation_lags", 10),
            spectral_permutations=audit_config.get("spectral_permutations", 199),
        ):
            randomness_audit = run_randomness_audit(
                frame,
                audit_config,
                random_seed=int(config["project"]["random_seed"]),
            )
            logging.info(
                "随机性审计完成 | 事件=RANDOMNESS_AUDIT_DONE | "
                "卡方异常=%s | 游程异常=%s | 自相关异常=%s | 频谱异常=%s | "
                "校正后异常总数=%s | 结论=%s",
                randomness_audit["families"]["chi_square"]["significant_after_holm_count"],
                randomness_audit["families"]["runs"]["significant_after_holm_count"],
                randomness_audit["families"]["autocorrelation"]["significant_after_holm_count"],
                randomness_audit["families"]["spectral"]["significant_after_holm_count"],
                randomness_audit["significant_after_holm_total"],
                randomness_audit["conclusion"],
            )
    else:
        randomness_audit = {
            "enabled": False,
            "conclusion": "随机性与数据质量审计已在配置中关闭。",
            "usage_constraint": "审计关闭不影响模型运行，但报告缺少随机性诊断证据。",
        }
    with logged_step("forecast_reconciliation"):
        forecast_audit = reconcile_forecast_audit(
            frame,
            project_path(config, "forecast_audit"),
            data_manifest=data_manifest,
            signing_key=signing_key,
            require_integrity=bool(config["compliance"].get("require_tamper_evident_audit", False)),
        )
        latest_validation = next(
            (
                record for record in reversed(forecast_audit["records"])
                if record.get("validation") is not None
            ),
            None,
        )
        if latest_validation:
            logging.info(
                "开奖前预测核验完成 | 事件=FORECAST_VALIDATED | 目标期=%s | 实际号码=%s | 命中统计=%s",
                latest_validation["target_period"],
                latest_validation["validation"]["actual"],
                latest_validation["validation"]["metrics"],
            )
    with logged_step("walk_forward_backtest", periods=periods or config["backtest"]["periods"]):
        backtest = run_walk_forward(frame, config, periods=periods)
        require_random_baseline(backtest)
        checkpoint_path = Path(config["paths"]["logs_dir"]) / "backtest_checkpoint.pkl"
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(
                {
                    "backtest": backtest,
                    "periods": periods or config["backtest"]["periods"],
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                checkpoint_path,
            )
            logging.info(
                "回测检查点已保存 | 事件=BACKTEST_CHECKPOINT_SAVED | 路径=%s",
                checkpoint_path,
            )
        except Exception:
            logging.exception("回测检查点保存失败 | 事件=BACKTEST_CHECKPOINT_FAILED")
        logging.info(
            "严格回测完成 | 事件=BACKTEST_DONE | 左截断审计=%s | 红球最终权重=%s | 蓝球最终权重=%s | 汇总=%s",
            backtest["strict_left_truncation"],
            backtest["final_weights_by_color"]["red"],
            backtest["final_weights_by_color"]["blue"],
            backtest["summary"],
        )
    model_release = evaluate_model_release(
        backtest,
        config["release"],
        forecast_audit=forecast_audit,
        code_fingerprint=code_fingerprint,
    )
    logging.info(
        "模型发布门槛 | 事件=MODEL_RELEASE_GATE | 通过=%s | 原因=%s",
        model_release["approved"],
        model_release["reason"],
    )
    red_guard = red_adverse_signal_guard(backtest, config)
    if red_guard.get("active"):
        logging.info(
            "红球逆信号保护已启用 | 事件=RED_ADVERSE_GUARD | 原因=%s | 方法数=%s/%s",
            red_guard.get("reason"),
            red_guard.get("adverse_methods"),
            red_guard.get("required_methods"),
        )
    with logged_step("current_candidate_experiment"):
        calibrated_config = apply_parameter_snapshot(config, backtest["final_parameters"])
        candidates = candidate_experiment(
            frame,
            calibrated_config,
            backtest["final_weights_by_color"],
            backtest["final_weight_reason_by_color"],
            red_adverse_guard=red_guard,
            stacking_records=backtest.get("period_records", []),
        )
    prediction_mode = candidates["model_pipeline"]["fusion"]["prediction_mode"]
    with logged_step("forecast_archive"):
        source_period = str(frame.iloc[-1]["period"])
        target_period = next_period_label(source_period)
        forecast_config = config.get("forecast", {}) if isinstance(config.get("forecast", {}), dict) else {}
        effective_backtest_periods = int(periods or config["backtest"]["periods"])
        formal_release_approved = bool(model_release.get("approved"))
        minimum_official_periods = int(
            config.get("release", {}).get("minimum_out_of_sample_periods", 300)
        )
        forecast_scope = "official" if formal_release_approved else "shadow"
        forecast_publish_gate = {
            "schema": "forecast_publish_gate_v2",
            "scope": forecast_scope,
            "effective_backtest_periods": effective_backtest_periods,
            "minimum_official_backtest_periods": minimum_official_periods,
            "formal_release_approved": formal_release_approved,
            "official_eligible": formal_release_approved,
            "email_eligible": formal_release_approved,
            "lock_official_target": bool(forecast_config.get("lock_official_target", True)),
            "reason": (
                "正式 Release Gate 已通过，允许作为当前目标期 official 预测；若同目标期已存在 official，则锁定旧记录。"
                if forecast_scope == "official"
                else "正式 Release Gate 尚未通过，本期仅作为 shadow/前瞻独立验证留痕，不发送推荐邮件，不覆盖 official。"
            ),
        }
        archived_forecast = archive_forecast(
            project_path(config, "forecast_audit"),
            source_period=source_period,
            target_period=target_period,
            tickets=candidates["consensus_selected"],
            model_weights_by_color=backtest["final_weights_by_color"],
            prediction_mode=prediction_mode,
            method=(
                "same_pipeline_no_signal_baseline_with_rotational_coverage"
                if prediction_mode == "uniform_no_signal_baseline"
                else (
                    "multi_model_color_specific_fusion_plus_consensus_selection"
                    if model_release["approved"]
                    else "unreleased_model_experiment_for_independent_validation"
                )
            ),
            config_fingerprint=config_fingerprint(calibrated_config),
            data_manifest=data_manifest,
            code_fingerprint=code_fingerprint,
            metadata={
                "dantuo_plan": candidates.get("dantuo_plan"),
                "core_numbers": candidates.get("core_numbers"),
                "forecast_publish_gate": forecast_publish_gate,
                "email_eligible": forecast_publish_gate["email_eligible"],
            },
            signing_key=signing_key,
            require_integrity=bool(config["compliance"].get("require_tamper_evident_audit", False)),
            forecast_scope=forecast_scope,
            lock_official_target=bool(forecast_config.get("lock_official_target", True)),
        )
        archived_forecast = {
            **archived_forecast,
            "dantuo_plan": (archived_forecast.get("metadata") or {}).get("dantuo_plan") or candidates.get("dantuo_plan"),
            "core_numbers": (archived_forecast.get("metadata") or {}).get("core_numbers") or candidates.get("core_numbers"),
            "forecast_publish_gate": (archived_forecast.get("metadata") or {}).get("forecast_publish_gate") or forecast_publish_gate,
        }
        if forecast_scope != "official":
            archived_forecast = {
                **archived_forecast,
                "forecast_scope": "shadow",
                "forecast_publish_gate": forecast_publish_gate,
                "display_status": "shadow_debug_observation",
                "display_reason": "本次运行低于正式验证期数，仅作为调试/影子结果，不作为正式预测邮件输出。",
            }
        elif (archived_forecast.get("archive_decision") or {}).get("status") == "official_target_locked":
            archived_forecast = {
                **archived_forecast,
                "display_status": "official_forecast_locked",
                "display_reason": "同一目标期已有正式开奖前留痕，本次结果未覆盖旧正式记录。",
            }
        if bool(config["generation"].get("withhold_red_uniform_output", False)):
            archived_forecast = {
                **archived_forecast,
                "display_status": "withheld_red_uniform_no_publishable_core",
                "display_reason": "红球已退化为均匀基线，本期不输出模型实验组合，避免将近似随机候选误写为预测结果。",
            }
        if forecast_scope == "official" and prediction_mode == "uniform_no_signal_baseline":
            archived_forecast = {
                **archived_forecast,
                "display_status": "archived_experimental_baseline",
                "display_reason": "无模型信号，已固化均匀基线实验组合，等待独立核验（非预测）",
            }
        elif forecast_scope == "official" and not model_release["approved"]:
            archived_forecast = {
                **archived_forecast,
                "display_status": "archived_experimental_observation",
                "display_reason": "本期仍有弱结构或实验信号，但尚未通过正式发布门槛，仅作为开奖前留痕与独立核验用。",
            }
        forecast_audit = reconcile_forecast_audit(
            frame,
            project_path(config, "forecast_audit"),
            data_manifest=data_manifest,
            signing_key=signing_key,
            require_integrity=bool(config["compliance"].get("require_tamper_evident_audit", False)),
        )
        logging.info(
            "开奖前实验组合已留痕 | 事件=FORECAST_ARCHIVED | 训练截止期=%s | 目标期=%s | 注数=%s | 审计编号=%s | 模式=%s",
            source_period,
            target_period,
            len(archived_forecast["tickets"]),
            archived_forecast["forecast_id"],
            prediction_mode,
        )
    compliance = config["compliance"]
    include_combinations = bool(compliance.get("enable_export_bet", False))
    candidate_section = {
        "prediction_mode": candidates["model_pipeline"]["fusion"]["prediction_mode"],
        "sampling": candidates["sampling"],
        "candidate_pool_hash": candidates.get("candidate_pool_hash"),
        "selected_by_method_hash": candidates.get("selected_by_method_hash"),
        "consensus_prediction_hash": candidates.get("consensus_prediction_hash"),
        "weights": candidates["weights"],
        "weights_by_color": candidates["weights_by_color"],
        "coverage": candidates["coverage"],
        "random_coverage": candidates["random_coverage"],
        "model_pipeline": candidates["model_pipeline"],
        "release": model_release,
        "consensus": {
            "selection_count": len(candidates["consensus_selected"]),
            "coverage": candidates["consensus_coverage"],
            "logic": candidates["selection_strategy"],
            "dantuo_plan": candidates.get("dantuo_plan"),
            "rolling_memory": candidates.get("rolling_memory"),
            "final_v7_guard": candidates.get("final_v7_guard"),
        },
        "output_count_per_method": int(calibrated_config["generation"]["output_count"]),
        "dantuo_plan": candidates.get("dantuo_plan"),
        "core_numbers": candidates.get("core_numbers"),
        "rolling_memory": candidates.get("rolling_memory"),
        "final_v7_guard": candidates.get("final_v7_guard"),
        "v7_portfolio": candidates.get("v7_portfolio"),
    }
    candidate_section["prediction_fingerprint"] = prediction_fingerprint(
        config_hash=config_fingerprint(calibrated_config),
        data_hash=str(
            run_manifest.get("data_version_id")
            or data_manifest.get("clean_data_sha256")
            or data_manifest.get("raw_sha256")
            or ""
        ),
        source_hash=code_fingerprint,
        random_seed=int(config["project"]["random_seed"]),
        probabilities={
            "candidate_pool_hash": candidates.get("candidate_pool_hash"),
            "selected_by_method_hash": candidates.get("selected_by_method_hash"),
            "consensus_prediction_hash": candidates.get("consensus_prediction_hash"),
            "prediction_mode": prediction_mode,
        },
        candidate_pool=candidates.get("candidate_pool_hash"),
        portfolio_input=candidates.get("selected_by_method_hash"),
        final_tickets=candidates.get("consensus_prediction_hash"),
    )
    # Candidate generation is retained for a reproducible, pre-draw audit, but
    # a public model combination requires the independent release gate.
    publish_model_combination = bool(model_release["approved"]) and (
        prediction_mode != "uniform_no_signal_baseline"
    )
    candidate_section["selected"] = (
        candidates["selected"] if publish_model_combination else {}
    )
    candidate_section["consensus"]["selected"] = (
        candidates["consensus_selected"] if publish_model_combination else []
    )
    candidate_section["experimental_selected"] = candidates["selected"]
    candidate_section["consensus"]["experimental_selected"] = candidates["consensus_selected"]
    candidate_section["display_selected"] = (
        candidates["selected"]
        if publish_model_combination
        else candidates["selected"]
        if candidates["consensus_selected"]
        else {}
    )
    candidate_section["consensus"]["display_selected"] = (
        candidates["consensus_selected"]
        if publish_model_combination or candidates["consensus_selected"]
        else []
    )
    candidate_section["display_only"] = publish_model_combination
    candidate_section["experimental_display_ready"] = bool(
        model_release.get("experimental_display_ready")
    )
    candidate_section["weak_structure_reference_ready"] = bool(
        model_release.get("weak_structure_reference_ready")
    )
    candidate_section["has_displayable_candidates"] = bool(
        candidates["consensus_selected"]
        or any(candidates["selected"].values())
    )
    candidate_section["no_signal_action"] = (
        "baseline_experimental_combination_archived_for_independent_validation"
        if prediction_mode == "uniform_no_signal_baseline"
        else (
            "model_combination_released_after_independent_evidence"
            if publish_model_combination
            else "model_combination_withheld_pending_independent_evidence"
        )
    )
    trend_analysis = analyze_number_trends(frame, config.get("trend", {}))
    mathematical_baseline = theoretical_ticket_baseline(
        int(calibrated_config["generation"]["output_count"])
    )
    blue_evidence = summarize_blue_evidence(
        backtest,
        randomness_audit,
        ticket_count=int(calibrated_config["generation"]["consensus_output_count"]),
        report_config=config.get("report", {}),
        random_seed=int(config["project"]["random_seed"]),
        history=frame,
        trend_config=config.get("trend", {}),
    )
    multi_period_strategy = summarize_multi_period_strategy(
        backtest.get("period_records", []),
        window=int(config.get("strategy", {}).get("multi_period_window", 10)),
    )
    validation_trace = load_validation_trace_view(project_path(config, "validation_trace"), config)
    execution_summary = summarize_pipeline_log(Path(config["paths"]["logs_dir"]) / "pipeline.log")
    performance_profile = RUNTIME_PROFILER.report()
    report = {
        "project": config["project"]["name"],
        "run_metadata": runtime_meta,
        "run_manifest": run_manifest,
        "config_snapshot": redact_sensitive_config(config),
        "data": {
            "periods": len(frame),
            "first_period": str(frame.iloc[0]["period"]),
            "latest_period": str(frame.iloc[-1]["period"]),
            "cleaning": clean_report,
        },
        "randomness_audit": randomness_audit,
        "candidate_experiment": candidate_section,
        "backtest": backtest,
        "red_adverse_guard": red_guard,
        "out_of_sample_forecast_audit": forecast_audit,
        "forecast_archive": archived_forecast,
        "trend_analysis": trend_analysis,
        "validation_trace": validation_trace,
        "mathematical_baseline": mathematical_baseline,
        "blue_evidence": blue_evidence,
        "multi_period_strategy": multi_period_strategy,
        "research_benchmark": backtest.get("research_benchmark", {}),
        "execution_summary": execution_summary,
        "performance_profile": performance_profile,
        "model_release": model_release,
        "v7_diagnostics": backtest.get("v7_diagnostics", {}),
        "random_baseline": {
            "candidate_coverage": candidates["random_coverage"],
            "backtest_random_repeats": backtest["random_repeats"],
        },
        "compliance": {
            "enable_export_bet": include_combinations,
            "display_candidates_only": True,
            "watermark": compliance["watermark"],
        },
    }
    report["release_checklist"] = release_checklist(report)
    report["v7_release_gate"] = v7_release_gate(report)
    with logged_step("report_export"):
        export_experiment_report(report, project_path(config, "report_json"))
        plot_performance_comparison(backtest, project_path(config, "performance_plot"))
        plot_hit_distribution(backtest, project_path(config, "distribution_plot"))
        plot_number_trend(
            frame,
            project_path(config, "trend_plot"),
            display_periods=int(trend_analysis["display_periods"]),
        )
    logging.info(
        "全流程运行完成 | 事件=PIPELINE_DONE | 总耗时=%.3f秒 | 有效期数=%s | 最新期=%s | 报告=%s",
        time.perf_counter() - pipeline_started,
        len(frame),
        frame.iloc[-1]["period"],
        project_path(config, "report_json"),
    )
    return report


HISTORY_COLUMN_ALIASES = {
    "period": {"period", "issue", "期号", "期次", "开奖期号"},
    "date": {"date", "drawdate", "开奖日期", "日期"},
    "blue": {"blue", "blueball", "蓝球", "蓝球号码", "蓝"},
    **{
        f"r{index}": {
            f"r{index}", f"red{index}", f"redball{index}",
            f"红球{index}", f"红{index}", f"红球号码{index}",
        }
        for index in range(1, 7)
    },
}
COMBINED_RED_ALIASES = {
    "red", "reds", "redballs", "红球", "红球号码", "红球组合",
    "开奖号码", "开奖结果", "中奖号码", "号码", "drawnumbers", "balls",
}
SSQ_SHEET_ALIASES = {"双色球", "ssq", "redblue", "红蓝"}
SKIP_SHEET_ALIASES = {"3d", "福彩3d", "f3d", "pl3"}


def _column_key(value: Any) -> str:
    return "".join(character for character in str(value).strip().lower() if character not in " _-()（）")


def _extract_draw_number_lists(series: pd.Series) -> pd.Series:
    return series.astype("string").map(
        lambda value: [int(token) for token in re.findall(r"\d+", str(value))]
        if pd.notna(value) and str(value).strip() not in {"", "<NA>", "nan", "None"}
        else []
    )


def _ssq_reds_and_blue(numbers: list[int]) -> tuple[list[int], int | None] | None:
    if len(numbers) == 7:
        reds, blue = numbers[:6], numbers[6]
        if len(set(reds)) == 6 and all(1 <= item <= 33 for item in reds) and 1 <= int(blue) <= 16:
            return sorted(reds), int(blue)
        return None
    if len(numbers) == 6:
        if len(set(numbers)) == 6 and all(1 <= item <= 33 for item in numbers):
            return sorted(numbers), None
        return None
    return None


def _apply_combined_draw_column(normalized: pd.DataFrame, keyed: dict[str, Any]) -> pd.DataFrame:
    missing_reds = not all(f"r{index}" in normalized.columns for index in range(1, 7))
    missing_blue = "blue" not in normalized.columns
    if not missing_reds and not missing_blue:
        return normalized
    combined_source = next(
        (keyed[_column_key(alias)] for alias in COMBINED_RED_ALIASES if _column_key(alias) in keyed),
        None,
    )
    if combined_source is None:
        return normalized
    parsed = [_ssq_reds_and_blue(numbers) for numbers in _extract_draw_number_lists(normalized[combined_source])]
    usable = [item for item in parsed if item is not None]
    if len(usable) < max(1, int(0.8 * max(len(parsed), 1))):
        return normalized
    if missing_reds:
        for index in range(1, 7):
            normalized[f"r{index}"] = [
                (item[0][index - 1] if item is not None else pd.NA) for item in parsed
            ]
    if missing_blue and any(item[1] is not None for item in usable):
        normalized["blue"] = [(item[1] if item is not None else pd.NA) for item in parsed]
    return normalized


def _normalize_history_columns(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    keyed = {_column_key(column): column for column in frame.columns}
    renamed: dict[Any, str] = {}
    for standard, aliases in HISTORY_COLUMN_ALIASES.items():
        for alias in aliases:
            source = keyed.get(_column_key(alias))
            if source is not None:
                renamed[source] = standard
                break
    normalized = frame.rename(columns=renamed).copy()
    normalized = _apply_combined_draw_column(normalized, keyed)
    if "date" not in normalized.columns:
        normalized["date"] = pd.NA
    missing_columns = [column for column in required if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"历史文件缺少或无法识别字段: {missing_columns}；实际表头: {list(frame.columns)}")
    normalized = normalized[required].copy()
    normalized["period"] = normalized["period"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return normalized


def _official_ssq_frame_from_headerless_sheet(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Parse the official ssq_asc.xls sheet whose first 9 columns are fixed.

    The official Welfare Lottery export often has no semantic header row:
    period, draw date, six sorted red balls, and blue ball are stored in columns
    0..8. If the importer misses that date column, later cleaning has to derive
    dates from an approximate weekday schedule, which is wrong around suspension
    periods such as Spring Festival.
    """
    if frame.shape[1] < 9:
        return None
    candidate = frame.iloc[:, :9].copy()
    candidate.columns = ["period", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]
    candidate["period"] = (
        candidate["period"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )
    candidate["date"] = pd.to_datetime(candidate["date"], errors="coerce")
    for column in ["r1", "r2", "r3", "r4", "r5", "r6", "blue"]:
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
    red_columns = [f"r{index}" for index in range(1, 7)]
    valid = (
        candidate["period"].str.fullmatch(r"\d{7}", na=False)
        & candidate["date"].notna()
        & candidate[red_columns].apply(lambda column: column.between(1, 33)).all(axis=1)
        & candidate["blue"].between(1, 16)
    )
    parsed = candidate.loc[valid, ["period", "date", *red_columns, "blue"]].copy()
    if len(parsed) < 60:
        return None
    return parsed


def _read_excel_history(content: bytes) -> pd.DataFrame:
    alias_keys = {
        _column_key(alias)
        for aliases in HISTORY_COLUMN_ALIASES.values()
        for alias in aliases
    } | {_column_key(alias) for alias in COMBINED_RED_ALIASES}
    required = ["period", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"]
    excel = pd.ExcelFile(io.BytesIO(content))
    ranked: list[tuple[int, int, int, pd.DataFrame]] = []
    best_headerless: pd.DataFrame | None = None
    for sheet in excel.sheet_names:
        sheet_key = _column_key(sheet)
        skip_penalty = 1 if any(alias in sheet_key for alias in SKIP_SHEET_ALIASES) else 0
        ssq_bonus = 1 if any(alias in sheet_key for alias in {_column_key(item) for item in SSQ_SHEET_ALIASES}) else 0
        headerless = pd.read_excel(excel, sheet_name=sheet, header=None)
        header_row: int | None = None
        header_matches = 0
        for index, row in headerless.head(15).iterrows():
            matches = sum(_column_key(value) in alias_keys for value in row.dropna())
            if matches >= 2 and matches > header_matches:
                header_row = int(index)
                header_matches = matches
        if header_row is None:
            official = _official_ssq_frame_from_headerless_sheet(headerless)
            if official is not None and (best_headerless is None or len(official) > len(best_headerless)):
                best_headerless = official
                ranked.append((2 + ssq_bonus, len(official), -skip_penalty, official))
            continue
        headed = pd.read_excel(excel, sheet_name=sheet, header=header_row)
        try:
            normalized = _normalize_history_columns(headed, required)
        except ValueError:
            continue
        ranked.append((header_matches + 10 * ssq_bonus, len(normalized), -skip_penalty, headed))
    if ranked:
        ranked.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
        return ranked[0][3]
    if best_headerless is not None:
        return best_headerless
    return pd.read_excel(excel, sheet_name=excel.sheet_names[0])


def read_uploaded_history(config: dict[str, Any], filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(io.BytesIO(content), dtype={"period": "string", "date": "string"})
        elif suffix in {".xls", ".xlsx"}:
            frame = _read_excel_history(content)
        else:
            raise ValueError("仅支持上传 .csv、.xls 或 .xlsx 历史数据")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"{suffix or '文件'} 读取失败: {error}") from error
    return _normalize_history_columns(frame, list(config["data"]["required_columns"]))


def import_history_file(config: dict[str, Any], filename: str, content: bytes, mode: str) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode != "replace":
        raise ValueError("导入方式已统一为 replace（覆盖原始数据）")
    uploaded = read_uploaded_history(config, filename, content)
    required = list(config["data"]["required_columns"])
    if uploaded.empty:
        raise ValueError("上传文件没有数据行")
    raw_path = project_path(config, "raw_data")
    combined = uploaded
    result = clean_history(
        combined,
        fill_missing_dates=bool(config["data"].get("fill_missing_dates", True)),
        draw_weekdays=config["data"].get("draw_weekdays", [1, 3, 6]),
        draw_blackout_ranges=config["data"].get("draw_blackout_ranges", []),
        invalid_row_policy=str(config["data"].get("invalid_row_policy", "reject")),
        validate_source_dates=bool(config["data"].get("validate_source_dates", False)),
    )
    _assert_calendar_feature_dates_available(result.data, config)
    minimum = int(config["data"].get("min_rows", 60))
    if len(result.data) < minimum:
        raise ValueError(f"清洗后仅 {len(result.data)} 期，少于最低要求 {minimum} 期，原数据未修改")
    combined = combined.drop_duplicates(subset="period", keep="last")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(".upload.tmp.csv")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(raw_path)
    save_clean_data(result.data, project_path(config, "clean_data"))
    manifest = create_data_manifest(
        raw_path=raw_path,
        clean_frame=result.data,
        cleaning_report=result.report,
        source={"kind": "dashboard_upload", "filename": Path(filename).name, "mode": mode},
    )
    write_data_manifest(project_path(config, "data_manifest"), manifest)
    logging.info(
        "历史数据上传完成 | 事件=DATA_UPLOAD | 文件=%s | 格式=%s | 导入方式=%s | 上传行数=%s | 原始总行数=%s | 清洗有效行数=%s | 清洗统计=%s",
        filename,
        Path(filename).suffix.lower(),
        mode,
        len(uploaded),
        len(combined),
        len(result.data),
        result.report,
    )
    return {
        "filename": filename,
        "mode": mode,
        "upload_rows": int(len(uploaded)),
        "raw_rows": int(len(combined)),
        "clean_rows": int(len(result.data)),
        "cleaning": result.report,
        "data_manifest": manifest,
    }


def dashboard_status(config: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    raw_path = project_path(config, "raw_data")
    clean_path = project_path(config, "clean_data")
    manifest = load_data_manifest(project_path(config, "data_manifest"))
    raw_rows = len(load_csv(raw_path)) if raw_path.exists() else 0
    clean_frame = pd.read_pickle(clean_path) if clean_path.exists() else pd.DataFrame()
    runtime_report = job.get("last_result")
    job_view = {key: value for key, value in job.items() if key != "last_result"}
    # Prefer the current session result.  If the session crashed after backtest
    # (for example during report JSON export), fall back to the on-disk report so
    # finish-export / restart can still show the completed run.
    report = copy.deepcopy(runtime_report) if runtime_report else None
    report_source = "current_session" if report else "none"
    if report is None:
        report_path = project_path(config, "report_json")
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report_source = "disk"
            except Exception:
                logging.exception("磁盘报告读取失败 | 事件=REPORT_DISK_LOAD_FAILED | 路径=%s", report_path)
                report = None
    if report and isinstance(report.get("config_snapshot"), dict):
        report["config_snapshot"] = redact_sensitive_config(report["config_snapshot"])
    if report is not None:
        try:
            report["validation_trace"] = load_validation_trace_view(
                project_path(config, "validation_trace"),
                config,
            )
        except Exception:
            logging.exception("逐期追踪加载失败 | 事件=VALIDATION_TRACE_STATUS_LOAD_FAILED")
    report_cleaning = (report or {}).get("data", {}).get("cleaning", {})
    cleaning = report_cleaning or ((manifest or {}).get("cleaning") or {})
    latest_period = str(clean_frame.iloc[-1]["period"]) if not clean_frame.empty else None
    report_latest_period = str((report or {}).get("data", {}).get("latest_period", "")) or None
    report_data_version = (cleaning.get("data_manifest") or {}).get("data_version_id")
    current_data_version = (manifest or {}).get("data_version_id")
    return {
        "data": {
            "raw_rows": int(raw_rows),
            "clean_rows": int(len(clean_frame)),
            "latest_period": latest_period,
            "dates_filled": cleaning.get("dates_filled_from_schedule"),
            "report_latest_period": report_latest_period,
            "report_stale": bool(
                report
                and (
                    latest_period != report_latest_period
                    or (current_data_version is not None and current_data_version != report_data_version)
                )
            ),
            "data_version_id": current_data_version,
            "quality_status": ((manifest or {}).get("cleaning") or {}).get("quality_status"),
        },
        "job": job_view,
        "config": {
            "random_repeats": int(config["backtest"]["random_repeats"]),
            "min_train_size": int(config["backtest"]["min_train_size"]),
            "holdout_periods": int(config.get("calibration", {}).get("holdout_periods", 30)),
            "enable_export_bet": bool(config["compliance"]["enable_export_bet"]),
            "default_runtime_profile": str(config.get("runtime", {}).get("default_profile", "standard")),
            "runtime_profiles": runtime_profile_options(config),
        },
        "report": report,
        "report_meta": {
            "source": report_source,
            "generated_at_utc": (report or {}).get("run_metadata", {}).get("generated_at_utc"),
            "latest_period": report_latest_period,
        },
    }


def dashboard_run(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    runtime_profile = apply_runtime_profile(
        run_config,
        str(payload.get("runtime_profile") or run_config.get("runtime", {}).get("default_profile", "standard")),
    )
    periods = int(payload.get("periods", run_config["backtest"]["periods"]))
    samples = int(payload.get("monte_carlo_samples", run_config["generation"]["monte_carlo_samples"]))
    output_count = int(payload.get("output_count", run_config["generation"]["output_count"]))
    if periods < 1:
        raise ValueError("验证期数必须为正整数")
    if not 100 <= samples <= 100000:
        raise ValueError("蒙特卡洛次数必须在 100-100000 之间")
    if not 2 <= output_count <= 50:
        raise ValueError("输出注数必须在 2-50 之间")
    run_config["generation"]["monte_carlo_samples"] = samples
    run_config["generation"]["candidate_pool_size"] = min(
        samples,
        max(int(run_config["generation"]["candidate_pool_size"]), output_count * 20),
    )
    run_config["generation"]["output_count"] = output_count
    run_config["generation"]["consensus_output_count"] = output_count
    run_config["backtest"]["periods"] = periods
    run_config["calibration"]["holdout_periods"] = periods
    validate_config(run_config)
    logging.info(
        "网页实验请求 | 事件=DASHBOARD_RUN_REQUEST | 运行档位=%s | 验证期数=%s | 蒙特卡洛次数=%s | 每期输出注数=%s | 优化方法=%s | 随机对照=%s",
        runtime_profile,
        periods,
        samples,
        output_count,
        run_config["optimizer"]["methods"],
        run_config["backtest"]["random_repeats"],
    )
    append_audit_log(project_path(run_config, "audit_log"), "dashboard_run", run_config["compliance"]["watermark"])
    return run_pipeline(run_config, periods=periods)


def print_summary(report: dict[str, Any], config: dict[str, Any]) -> None:
    latest_validation = next(
        (
            item for item in reversed(report.get("out_of_sample_forecast_audit", {}).get("records", []))
            if item.get("validation") is not None
        ),
        None,
    )
    print("\n【运行结果说明】")
    print(f"历史数据：{report['data']['periods']}期，最新期号 {report['data']['latest_period']}")
    print("模型处理：红球与蓝球分开评估、分开赋权；同质模型证据合并计权。")
    holdout_periods = int(
        report.get("backtest", {}).get("calibration_summary", {}).get(
            "requested_holdout_periods", config["backtest"]["periods"]
        )
    )
    print(
        f"验证规则：保留期之前先做融合预热；最终{holdout_periods}期冻结超参数，"
        "只允许按既往损失更新下一期融合权重。"
    )
    if latest_validation:
        metrics = latest_validation["validation"]["metrics"]
        print(
            f"最新开奖前留痕验证：{latest_validation['target_period']}期，"
            f"每注平均红球命中 {metrics['average_red_hits']:.3f}，"
            f"蓝球{'至少一注命中' if metrics['any_blue_hit'] else '未命中'}。"
        )
    print("以下为机器可读汇总；分数不是中奖概率，是否优于随机以严格回测和校正后显著性为准。")
    candidate = report.get("candidate_experiment") or {}
    backtest = report.get("backtest") or {}
    print(json.dumps({
        "data": report.get("data"),
        "fusion_weights": candidate.get("weights"),
        "coverage": candidate.get("coverage"),
        "equal_size_random_coverage": candidate.get("random_coverage"),
        "backtest_summary": backtest.get("summary"),
        "calibration_summary": backtest.get("calibration_summary"),
        "strict_left_truncation": backtest.get("strict_left_truncation"),
        "forecast_id": (report.get("forecast_archive") or {}).get("forecast_id"),
        "report_file": str(project_path(config, "report_json")),
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="概率实验系统")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["serve", "run", "clean", "backtest", "predict", "setup-audit-key", "finish-export"],
        default="serve",
    )
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--periods", type=int, help="覆盖配置中的回测期数")
    parser.add_argument(
        "--profile",
        choices=["daily", "monthly", "yearly"],
        help="回测档位：daily(日常快速,50期)/monthly(月度标准,100期)/yearly(年度深度,200期)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="本地界面监听地址")
    parser.add_argument("--port", type=int, default=8001, help="本地界面端口")
    parser.add_argument("--no-browser", action="store_true", help="启动界面时不自动打开浏览器")
    return parser.parse_args()


def apply_backtest_profile(config, profile_name):
    """应用回测档位配置：daily/monthly/yearly。"""
    profiles = config.get("backtest_profiles", {})
    if profile_name not in profiles:
        logging.warning("回测档位 %s 未配置，使用默认配置", profile_name)
        return config

    profile = profiles[profile_name]
    logging.info(
        "应用回测档位 | 事件=BACKTEST_PROFILE_APPLIED | 档位=%s | 说明=%s | 期数=%s | 预计时间=%s",
        profile_name,
        profile.get("label", ""),
        profile.get("periods", config["backtest"].get("periods")),
        profile.get("estimated_minutes", "未知"),
    )

    if "periods" in profile:
        config["backtest"]["periods"] = int(profile["periods"])
    if "retrain_interval" in profile:
        config["backtest"]["retrain_interval"] = int(profile["retrain_interval"])
    if "optimizer_methods" in profile:
        if "optimizer" not in config:
            config["optimizer"] = {}
        config["optimizer"]["methods"] = list(profile["optimizer_methods"])
    if "monte_carlo_samples" in profile or "candidate_pool_size" in profile:
        if "generation" not in config:
            config["generation"] = {}
        if "monte_carlo_samples" in profile:
            config["generation"]["monte_carlo_samples"] = int(profile["monte_carlo_samples"])
        if "candidate_pool_size" in profile:
            config["generation"]["candidate_pool_size"] = int(profile["candidate_pool_size"])
    if "permutation_repetitions" in profile:
        if "report" not in config:
            config["report"] = {}
        config["report"]["permutation_repetitions"] = int(profile["permutation_repetitions"])

    return config


def _clean_temp_files(config: dict[str, Any], *, preserve_validation_artifacts: bool = False) -> None:
    """运行前清理上次运行的临时产物，避免 Web 读到旧数据。"""
    paths = config.get("paths", {})
    removed = []
    log_dir = ROOT / "logs"
    # 只删 Web 展示读取的报告/图/日志，不删 data_manifest.json（数据版本标识）
    # serve / finish-export 保留上一轮完整产物，避免 300 期跑完后收尾结果被启动命令清掉。
    if preserve_validation_artifacts:
        deletable_logs: list[str] = []
    else:
        deletable_logs = [
            "experiment_report.json",
            "forecast_audit.json",
            "pipeline.log",
            "hit_distribution.png",
            "number_trend.png",
            "performance_comparison.png",
            "optuna_tuning.json",
            "result_page_full.png",
            "backtest_checkpoint.pkl",
        ]
    for name in deletable_logs:
        fp = log_dir / name
        if fp.exists():
            try:
                fp.unlink()
                removed.append(name)
            except OSError:
                pass
    # 只删逐期 trace，不删 clean_data.pkl（清洗缓存）
    if not preserve_validation_artifacts:
        rel = paths.get("validation_trace")
        if rel:
            fp = ROOT / rel
            if fp.exists() and fp.is_file():
                try:
                    fp.unlink()
                    removed.append(rel)
                except OSError:
                    pass
        archive_dir = ROOT / "data" / "validation_trace_archive"
        if archive_dir.exists():
            for fp in archive_dir.glob("*.jsonl"):
                try:
                    fp.unlink()
                    removed.append(fp.name)
                except OSError:
                    pass
    if removed:
        logging.info("已清理 %d 个临时文件", len(removed))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    preserve_artifacts = args.command in {"serve", "finish-export", "setup-audit-key", "clean"}
    _clean_temp_files(config, preserve_validation_artifacts=preserve_artifacts)
    # 应用回测档位配置
    if args.profile:
        config = apply_backtest_profile(config, args.profile)
    configure_logging(config)
    show_risk_notice(
        bool(config["compliance"].get("show_risk_notice", True)),
        str(config["compliance"]["watermark"]),
    )
    if args.command == "setup-audit-key":
        audit_signing_key(config)
        print("预测审计密钥已就绪：优先使用环境变量；未配置时会自动使用本机受限密钥存储。")
        return
    logging.info(
        "系统就绪 | 事件=SYSTEM_READY | 命令=%s | 模型=%s | 流程=数据清洗→特征工程→红蓝分色模型融合→蒙特卡洛→覆盖筛选→严格逐期回测",
        args.command,
        list(MODEL_LABELS),
    )
    append_audit_log(project_path(config, "audit_log"), args.command, config["compliance"]["watermark"])
    if args.command == "serve":
        serve_dashboard(
            project_root=ROOT,
            host=args.host,
            port=args.port,
            status_callback=lambda job: dashboard_status(config, job),
            run_callback=lambda payload: dashboard_run(config, payload),
            upload_callback=lambda filename, content, mode: import_history_file(config, filename, content, mode),
            notification_config=config,
            open_browser=not args.no_browser,
        )
        return
    if args.command == "finish-export":
        report = finish_export_and_notify(
            config=config,
            project_root=ROOT,
            project_path_fn=project_path,
            load_validation_trace_view_fn=load_validation_trace_view,
            dashboard_url=f"http://{args.host}:{args.port}/Tsys/",
        )
        print_summary(report, config)
        return
    if args.command == "clean":
        frame, report = prepare_data(config)
        print(json.dumps({"periods": len(frame), "cleaning": report}, ensure_ascii=False, indent=2))
        return
    if args.command == "backtest":
        frame, _ = prepare_data(config)
        result = run_walk_forward(frame, config, periods=args.periods)
        require_random_baseline(result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return
    report = run_pipeline(config, periods=args.periods)
    send_result_email(report, config, project_root=ROOT)
    print_summary(report, config)


if __name__ == "__main__":
    main()
