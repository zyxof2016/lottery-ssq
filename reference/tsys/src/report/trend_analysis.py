from __future__ import annotations

"""Descriptive number-trend statistics used by the local report.

The functions in this module deliberately report historical properties only.
They do not estimate an advantage for a future independent draw.
"""

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.data_handler.cleaner import RED_COLUMNS


def _as_int(value: Any) -> int:
    return int(value)


def _omission_lengths(flags: np.ndarray) -> np.ndarray:
    """Return the number of completed records since each number last appeared."""
    periods, width = flags.shape
    omissions = np.empty(width, dtype=int)
    for column in range(width):
        hits = np.flatnonzero(flags[:, column])
        omissions[column] = periods if len(hits) == 0 else periods - 1 - int(hits[-1])
    return omissions


def _slopes(flags: np.ndarray) -> np.ndarray:
    if len(flags) < 2:
        return np.zeros(flags.shape[1], dtype=float)
    x = np.arange(len(flags), dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0:
        return np.zeros(flags.shape[1], dtype=float)
    centered_y = flags - flags.mean(axis=0)
    return np.dot(centered_x, centered_y) / denominator


def _flags(frame: pd.DataFrame, columns: list[str], universe_size: int) -> np.ndarray:
    result = np.zeros((len(frame), universe_size), dtype=int)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        for column in columns:
            number = _as_int(row[column])
            result[row_index, number - 1] = 1
    return result


def _number_statistics(
    frame: pd.DataFrame,
    columns: list[str],
    universe_size: int,
    *,
    label: str,
    analysis_window: int,
    recent_window: int,
) -> dict[str, Any]:
    window = frame.tail(min(len(frame), analysis_window)).copy()
    flags = _flags(window, columns, universe_size)
    recent = flags[-min(len(flags), recent_window):]
    draws_per_period = len(columns)
    observations = len(flags)
    probability = draws_per_period / universe_size
    expected = observations * probability
    variance = observations * probability * (1.0 - probability)
    standard_deviation = sqrt(variance) if variance > 0 else 0.0
    counts = flags.sum(axis=0)
    recent_counts = recent.sum(axis=0)
    omissions = _omission_lengths(flags)
    slopes = _slopes(flags)
    rows = []
    for number in range(1, universe_size + 1):
        count = int(counts[number - 1])
        z_score = (count - expected) / standard_deviation if standard_deviation else 0.0
        rows.append(
            {
                "number": number,
                "count": count,
                "recent_count": int(recent_counts[number - 1]),
                "expected_count": round(float(expected), 3),
                "z_score": round(float(z_score), 4),
                "current_omission": int(omissions[number - 1]),
                "linear_slope_per_period": round(float(slopes[number - 1]), 6),
            }
        )
    descriptive_hot = sorted(rows, key=lambda item: (item["z_score"], item["recent_count"], -item["number"]), reverse=True)[:6 if label == "red" else 3]
    descriptive_cold = sorted(rows, key=lambda item: (item["z_score"], -item["recent_count"], item["number"]))[:6 if label == "red" else 3]
    longest_omission = sorted(rows, key=lambda item: (item["current_omission"], -item["number"]), reverse=True)[:6 if label == "red" else 3]
    return {
        "analysis_periods": observations,
        "recent_periods": len(recent),
        "draws_per_period": draws_per_period,
        "universe_size": universe_size,
        "per_number": rows,
        "descriptive_hot": descriptive_hot,
        "descriptive_cold": descriptive_cold,
        "longest_current_omission": longest_omission,
    }


def _structure_summary(frame: pd.DataFrame, display_periods: int) -> dict[str, Any]:
    recent = frame.tail(min(len(frame), display_periods))
    sums = recent[RED_COLUMNS].sum(axis=1).astype(int)
    spans = (recent[RED_COLUMNS].max(axis=1) - recent[RED_COLUMNS].min(axis=1)).astype(int)
    odd_counts = recent[RED_COLUMNS].apply(lambda row: sum(int(value) % 2 for value in row), axis=1)
    zone_counts = pd.DataFrame(
        {
            "zone_1_11": recent[RED_COLUMNS].apply(lambda row: sum(1 <= int(value) <= 11 for value in row), axis=1),
            "zone_12_22": recent[RED_COLUMNS].apply(lambda row: sum(12 <= int(value) <= 22 for value in row), axis=1),
            "zone_23_33": recent[RED_COLUMNS].apply(lambda row: sum(23 <= int(value) <= 33 for value in row), axis=1),
        }
    )
    return {
        "periods": int(len(recent)),
        "red_sum": {
            "average": round(float(sums.mean()), 3),
            "minimum": int(sums.min()),
            "maximum": int(sums.max()),
        },
        "red_span": {
            "average": round(float(spans.mean()), 3),
            "minimum": int(spans.min()),
            "maximum": int(spans.max()),
        },
        "odd_red_count": {str(number): int((odd_counts == number).sum()) for number in range(7)},
        "zone_average": {column: round(float(zone_counts[column].mean()), 3) for column in zone_counts},
    }


def _trend_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the compact source rows required to render the full web matrix."""
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        red = [_as_int(row[column]) for column in RED_COLUMNS]
        date_value = pd.Timestamp(row["date"])
        rows.append(
            {
                "period": str(row["period"]),
                "weekday": int(date_value.isoweekday()),
                "red": red,
                "blue": _as_int(row["blue"]),
                "red_sum": int(sum(red)),
                "red_span": int(max(red) - min(red)),
                "zone_ratio": [
                    int(sum(1 <= number <= 11 for number in red)),
                    int(sum(12 <= number <= 22 for number in red)),
                    int(sum(23 <= number <= 33 for number in red)),
                ],
                "odd_even_ratio": [
                    int(sum(number % 2 for number in red)),
                    int(sum(number % 2 == 0 for number in red)),
                ],
            }
        )
    return rows


def analyze_number_trends(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create JSON-safe historical trend and descriptive-statistics data."""
    settings = config or {}
    display_periods = max(10, min(int(settings.get("display_periods", 60)), len(frame)))
    analysis_window = max(10, min(int(settings.get("analysis_window", 100)), len(frame)))
    recent_window = max(10, min(int(settings.get("recent_window", 30)), analysis_window))
    display = frame.tail(display_periods)
    draws = _trend_rows(display)
    return {
        "display_periods": len(display),
        "analysis_window": analysis_window,
        "recent_window": recent_window,
        "first_display_period": str(display.iloc[0]["period"]),
        "latest_display_period": str(display.iloc[-1]["period"]),
        "draws": draws,
        "trend_rows": _trend_rows(frame),
        "red": _number_statistics(
            frame, RED_COLUMNS, 33, label="red", analysis_window=analysis_window, recent_window=recent_window
        ),
        "blue": _number_statistics(
            frame, ["blue"], 16, label="blue", analysis_window=analysis_window, recent_window=recent_window
        ),
        "structure": _structure_summary(frame, display_periods),
        "interpretation": (
            "频数、当前遗漏、二项分布标准化偏离和线性斜率均为历史描述统计。"
            "在独立随机前提下，它们不构成下一期概率优势；最终是否展示实验组合仍由严格样本外检验决定。"
        ),
    }
