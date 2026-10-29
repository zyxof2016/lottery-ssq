from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, chisquare, norm

from src.data_handler.cleaner import RED_COLUMNS, assert_clean_sequence


def _holm_adjust(p_values: list[float | None]) -> list[float | None]:
    available = sorted(
        (float(value), index)
        for index, value in enumerate(p_values)
        if value is not None and np.isfinite(value)
    )
    adjusted: list[float | None] = [None] * len(p_values)
    running_max = 0.0
    count = len(available)
    for rank, (p_value, index) in enumerate(available):
        running_max = max(running_max, min(1.0, (count - rank) * p_value))
        adjusted[index] = running_max
    return adjusted


def _apply_holm(details: list[dict[str, Any]], alpha: float) -> None:
    adjusted = _holm_adjust([detail.get("p_value") for detail in details])
    for detail, adjusted_p in zip(details, adjusted):
        detail["holm_adjusted_p_value"] = adjusted_p
        detail["significant_after_holm"] = bool(
            adjusted_p is not None and adjusted_p < alpha
        )


def _indicator_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    red = np.zeros((len(frame), 33), dtype=np.int8)
    red_values = frame[RED_COLUMNS].to_numpy(dtype=int) - 1
    rows = np.repeat(np.arange(len(frame)), len(RED_COLUMNS))
    red[rows, red_values.reshape(-1)] = 1

    blue = np.zeros((len(frame), 16), dtype=np.int8)
    blue[np.arange(len(frame)), frame["blue"].to_numpy(dtype=int) - 1] = 1
    return red, blue


def _series_label(color: str, number: int) -> str:
    return f"{'红球' if color == 'red' else '蓝球'}{number:02d}"


def _chi_square_detail(
    matrix: np.ndarray,
    *,
    color: str,
    alpha: float,
    simulations: int,
    random_seed: int,
) -> dict[str, Any]:
    counts = matrix.sum(axis=0).astype(float)
    total = float(counts.sum())
    expected = np.full(len(counts), total / len(counts), dtype=float)
    statistic, asymptotic_p_value = chisquare(counts, expected)
    if color == "red":
        # A draw contains six *distinct* red balls. A multinomial null treats
        # those six selections as independent and gives slightly wrong variance.
        rng = np.random.default_rng(int(random_seed))
        extreme = 0
        for _ in range(int(simulations)):
            random_values = rng.random((matrix.shape[0], matrix.shape[1]))
            selected = np.argpartition(random_values, kth=5, axis=1)[:, :6]
            simulated_counts = np.bincount(selected.reshape(-1), minlength=matrix.shape[1])
            simulated_statistic = float(np.sum((simulated_counts - expected) ** 2 / expected))
            extreme += int(simulated_statistic >= statistic)
        p_value = float((extreme + 1) / (int(simulations) + 1))
        p_value_method = "lottery_draw_parametric_bootstrap"
    else:
        p_value = float(asymptotic_p_value)
        p_value_method = "chi_square_asymptotic"
    result = {
        "color": color,
        "categories": int(len(counts)),
        "observations": int(total),
        "statistic": float(statistic),
        "degrees_of_freedom": int(len(counts) - 1),
        "p_value": p_value,
        "asymptotic_p_value": float(asymptotic_p_value),
        "p_value_method": p_value_method,
        "simulations": int(simulations) if color == "red" else None,
        "cohens_w": float(sqrt(float(statistic) / total)) if total > 0 else None,
        "maximum_relative_deviation": float(
            np.max(np.abs(counts - expected) / expected)
        ),
        "minimum_count": int(counts.min()),
        "maximum_count": int(counts.max()),
    }
    result["significant_at_alpha"] = bool(result["p_value"] < alpha)
    return result


def _runs_detail(values: np.ndarray, *, color: str, number: int) -> dict[str, Any]:
    binary = np.asarray(values, dtype=np.int8)
    n = int(len(binary))
    n1 = int(binary.sum())
    n0 = n - n1
    runs = int(1 + np.count_nonzero(binary[1:] != binary[:-1])) if n else 0
    if n < 2 or n0 == 0 or n1 == 0:
        return {
            "color": color,
            "number": number,
            "label": _series_label(color, number),
            "runs": runs,
            "expected_runs": None,
            "z_score": None,
            "effect_size": None,
            "p_value": None,
            "status": "insufficient_variation",
        }

    expected = 1.0 + 2.0 * n0 * n1 / n
    variance = (
        2.0 * n0 * n1 * (2.0 * n0 * n1 - n)
        / (n * n * (n - 1))
    )
    difference = runs - expected
    corrected = np.sign(difference) * max(abs(difference) - 0.5, 0.0)
    z_score = float(corrected / sqrt(variance)) if variance > 0 else 0.0
    return {
        "color": color,
        "number": number,
        "label": _series_label(color, number),
        "runs": runs,
        "expected_runs": float(expected),
        "z_score": z_score,
        "effect_size": float(abs(z_score) / sqrt(n)),
        "p_value": float(2.0 * norm.sf(abs(z_score))),
        "status": "ok",
    }


def _autocorrelation_detail(
    values: np.ndarray,
    *,
    color: str,
    number: int,
    max_lag: int,
) -> dict[str, Any]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    n = int(len(centered))
    lag_count = min(int(max_lag), max(0, n - 2))
    denominator = float(np.dot(centered, centered))
    if lag_count < 1 or denominator <= 0:
        return {
            "color": color,
            "number": number,
            "label": _series_label(color, number),
            "lags": lag_count,
            "ljung_box_q": None,
            "maximum_absolute_autocorrelation": None,
            "p_value": None,
            "status": "insufficient_variation",
        }

    correlations = [
        float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        for lag in range(1, lag_count + 1)
    ]
    q_statistic = n * (n + 2.0) * sum(
        value * value / (n - lag)
        for lag, value in enumerate(correlations, start=1)
    )
    return {
        "color": color,
        "number": number,
        "label": _series_label(color, number),
        "lags": lag_count,
        "ljung_box_q": float(q_statistic),
        "maximum_absolute_autocorrelation": float(
            max(abs(value) for value in correlations)
        ),
        "autocorrelations": correlations,
        "p_value": float(chi2.sf(q_statistic, lag_count)),
        "status": "ok",
    }


def _spectral_statistic(values: np.ndarray) -> tuple[float, float, int, float]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered))[1:] ** 2
    total = float(power.sum())
    if total <= 0 or len(power) == 0:
        return 0.0, 1.0, 0, 0.0
    normalized = power / total
    peak_index = int(np.argmax(normalized))
    entropy = float(
        -np.sum(normalized * np.log(np.clip(normalized, 1e-300, None)))
        / np.log(len(normalized))
    ) if len(normalized) > 1 else 0.0
    frequency_bin = peak_index + 1
    peak_period = float(len(centered) / frequency_bin)
    return float(normalized[peak_index]), entropy, frequency_bin, peak_period


def _spectral_detail(
    values: np.ndarray,
    *,
    color: str,
    number: int,
    permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    binary = np.asarray(values, dtype=np.int8)
    peak_ratio, entropy, frequency_bin, peak_period = _spectral_statistic(binary)
    if binary.min() == binary.max():
        return {
            "color": color,
            "number": number,
            "label": _series_label(color, number),
            "peak_power_ratio": None,
            "normalized_spectral_entropy": None,
            "peak_frequency_bin": None,
            "peak_period": None,
            "permutations": int(permutations),
            "p_value": None,
            "status": "insufficient_variation",
        }

    rng = np.random.default_rng(int(random_seed))
    extreme = 0
    for _ in range(int(permutations)):
        permuted_peak, _, _, _ = _spectral_statistic(rng.permutation(binary))
        extreme += int(permuted_peak >= peak_ratio)
    return {
        "color": color,
        "number": number,
        "label": _series_label(color, number),
        "peak_power_ratio": peak_ratio,
        "normalized_spectral_entropy": entropy,
        "peak_frequency_bin": frequency_bin,
        "peak_period": peak_period,
        "permutations": int(permutations),
        "p_value": float((extreme + 1) / (int(permutations) + 1)),
        "status": "ok",
    }


def _family_summary(
    name: str,
    details: list[dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    _apply_holm(details, alpha)
    valid = [detail for detail in details if detail.get("p_value") is not None]
    significant = [
        detail for detail in valid if detail.get("significant_after_holm")
    ]
    strongest = sorted(
        valid,
        key=lambda detail: (
            detail.get("holm_adjusted_p_value", 1.0),
            detail.get("p_value", 1.0),
        ),
    )[:5]
    return {
        "method": name,
        "test_count": int(len(details)),
        "valid_test_count": int(len(valid)),
        "significant_after_holm_count": int(len(significant)),
        "minimum_raw_p_value": (
            float(min(detail["p_value"] for detail in valid)) if valid else None
        ),
        "minimum_holm_adjusted_p_value": (
            float(min(detail["holm_adjusted_p_value"] for detail in valid))
            if valid else None
        ),
        "strongest_signals": [
            {
                "label": detail.get("label", detail.get("color")),
                "p_value": detail.get("p_value"),
                "holm_adjusted_p_value": detail.get("holm_adjusted_p_value"),
            }
            for detail in strongest
        ],
        "details": details,
    }


def _apply_global_holm(families: dict[str, dict[str, Any]], alpha: float) -> int:
    """Control the false-positive rate of the final cross-family conclusion."""
    indexed: list[tuple[str, int, float]] = []
    for family_name, family in families.items():
        for detail_index, detail in enumerate(family["details"]):
            value = detail.get("p_value")
            if value is not None and np.isfinite(value):
                indexed.append((family_name, detail_index, float(value)))
    ordered = sorted(indexed, key=lambda item: item[2])
    running_max = 0.0
    total = len(ordered)
    adjusted: dict[tuple[str, int], float] = {}
    for rank, (family_name, detail_index, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (total - rank) * p_value))
        adjusted[(family_name, detail_index)] = running_max
    total_significant = 0
    for family_name, family in families.items():
        global_count = 0
        for detail_index, detail in enumerate(family["details"]):
            global_p = adjusted.get((family_name, detail_index))
            detail["global_holm_adjusted_p_value"] = global_p
            detail["significant_after_global_holm"] = bool(global_p is not None and global_p < alpha)
            global_count += int(detail["significant_after_global_holm"])
        family["significant_after_global_holm_count"] = global_count
        total_significant += global_count
    return total_significant


def run_randomness_audit(
    frame: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    random_seed: int = 2026,
) -> dict[str, Any]:
    assert_clean_sequence(frame)
    settings = config or {}
    alpha = float(settings.get("significance_level", 0.05))
    max_lag = int(settings.get("autocorrelation_lags", 10))
    permutations = int(settings.get("spectral_permutations", 199))
    chi_square_simulations = int(settings.get("chi_square_simulations", 199))
    if not 0.0 < alpha < 1.0:
        raise ValueError("randomness_audit.significance_level 必须在 (0, 1) 范围内")
    if max_lag < 1:
        raise ValueError("randomness_audit.autocorrelation_lags 必须至少为1")
    if permutations < 19:
        raise ValueError("randomness_audit.spectral_permutations 必须至少为19")
    if chi_square_simulations < 99:
        raise ValueError("randomness_audit.chi_square_simulations 必须至少为99")

    red, blue = _indicator_matrices(frame)
    matrices = {"red": red, "blue": blue}

    chi_details = [
        _chi_square_detail(
            matrix,
            color=color,
            alpha=alpha,
            simulations=chi_square_simulations,
            random_seed=int(random_seed) + 90_001 + index,
        )
        for index, (color, matrix) in enumerate(matrices.items())
    ]
    for detail in chi_details:
        detail["label"] = "红球边际分布" if detail["color"] == "red" else "蓝球边际分布"

    runs_details: list[dict[str, Any]] = []
    autocorrelation_details: list[dict[str, Any]] = []
    spectral_details: list[dict[str, Any]] = []
    series_index = 0
    for color, matrix in matrices.items():
        for column in range(matrix.shape[1]):
            number = column + 1
            values = matrix[:, column]
            runs_details.append(
                _runs_detail(values, color=color, number=number)
            )
            autocorrelation_details.append(
                _autocorrelation_detail(
                    values,
                    color=color,
                    number=number,
                    max_lag=max_lag,
                )
            )
            spectral_details.append(
                _spectral_detail(
                    values,
                    color=color,
                    number=number,
                    permutations=permutations,
                    random_seed=int(random_seed) + series_index * 1009,
                )
            )
            series_index += 1

    families = {
        "chi_square": _family_summary(
            "chi_square_uniformity", chi_details, alpha=alpha
        ),
        "runs": _family_summary(
            "wald_wolfowitz_runs", runs_details, alpha=alpha
        ),
        "autocorrelation": _family_summary(
            "ljung_box_autocorrelation", autocorrelation_details, alpha=alpha
        ),
        "spectral": _family_summary(
            "permutation_periodogram_peak", spectral_details, alpha=alpha
        ),
    }
    anomaly_count = _apply_global_holm(families, alpha)
    return {
        "enabled": True,
        "periods": int(len(frame)),
        "first_period": str(frame.iloc[0]["period"]),
        "latest_period": str(frame.iloc[-1]["period"]),
        "significance_level": alpha,
        "multiple_testing_correction": "Holm within family plus global Holm for the final conclusion",
        "global_holm_test_count": sum(family["valid_test_count"] for family in families.values()),
        "series_definition": {
            "red": "33条逐期0/1出现序列，每期固定6个1",
            "blue": "16条逐期0/1出现序列，每期固定1个1",
        },
        "families": families,
        "significant_after_holm_total": int(anomaly_count),
        "significant_after_global_holm_total": int(anomaly_count),
        "requires_manual_review": bool(anomaly_count > 0),
        "conclusion": (
            "发现经校正后仍显著的历史序列异常，需要先复核数据与开奖机制；"
            "统计异常不等于下一期可预测。"
            if anomaly_count
            else "未发现经校正后显著偏离随机基线的历史序列证据。"
        ),
        "usage_constraint": (
            "本审计仅用于数据质量和随机性诊断，结果不得直接参与号码评分、"
            "模型融合权重或下一期候选生成。"
        ),
    }
