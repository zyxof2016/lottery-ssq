from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineer.base_stats import RED_COLUMNS, occurrence_matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(values.sum())
    return values / total if total > 0 else np.full(len(values), 1.0 / max(len(values), 1))


def _tempered(values: np.ndarray, temperature: float) -> np.ndarray:
    values = _normalize(values)
    exponent = 1.0 / max(float(temperature), 1e-6)
    return _normalize(np.power(np.clip(values, 1e-12, None), exponent))


def _current_gap(matrix: np.ndarray) -> np.ndarray:
    periods, number_count = matrix.shape
    last_seen = np.full(number_count, -1, dtype=int)
    for row_index, row in enumerate(matrix):
        hits = np.flatnonzero(row)
        last_seen[hits] = row_index
    return np.where(last_seen >= 0, periods - 1 - last_seen, periods).astype(float)


def _gap_statistics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    periods, number_count = matrix.shape
    gaps: list[list[int]] = [[] for _ in range(number_count)]
    last_seen = np.full(number_count, -1, dtype=int)
    for row_index, row in enumerate(matrix):
        for index in np.flatnonzero(row):
            if last_seen[index] >= 0:
                gaps[index].append(row_index - int(last_seen[index]) - 1)
            last_seen[index] = row_index
    means = np.array([
        float(np.mean(items)) if items else float(periods)
        for items in gaps
    ])
    spreads = np.array([
        max(float(np.std(items)), 1.0) if len(items) > 1 else max(float(means[index]), 1.0)
        for index, items in enumerate(gaps)
    ])
    return means, spreads


def _omission_score(matrix: np.ndarray) -> np.ndarray:
    gap = _current_gap(matrix)
    mean_gap, spread = _gap_statistics(matrix)
    z = (gap - mean_gap) / np.maximum(spread, 1.0)
    due = 1.0 / (1.0 + np.exp(-z))
    too_far = np.exp(-np.square(np.maximum(z - 2.25, 0.0)) / 2.0)
    return _normalize(0.25 + due * too_far)


def _window_frequency(matrix: np.ndarray, window: int) -> np.ndarray:
    lookback = matrix[-min(int(window), len(matrix)):]
    return _normalize(lookback.sum(axis=0) + 1.0)


def _decayed_frequency(matrix: np.ndarray, decay: float) -> np.ndarray:
    ages = np.arange(len(matrix) - 1, -1, -1)
    weights = np.power(float(decay), ages)
    return _normalize(weights @ matrix + 1e-9)


def _red_transition_score(history: pd.DataFrame, matrix: np.ndarray, alpha: float) -> np.ndarray:
    if len(history) < 2:
        return np.full(33, 1.0 / 33.0)
    latest_red = set(int(number) for number in history.iloc[-1][RED_COLUMNS].to_numpy(dtype=int))
    scores = np.full(33, float(alpha), dtype=float)
    previous = history[RED_COLUMNS].to_numpy(dtype=int)[:-1]
    following = matrix[1:]
    for row_index, prior in enumerate(previous):
        overlap = len(latest_red & {int(number) for number in prior})
        if overlap <= 0:
            continue
        scores += float(overlap) * following[row_index]
    return _normalize(scores)


def _red_zone_score(history: pd.DataFrame) -> np.ndarray:
    reds = history[RED_COLUMNS].to_numpy(dtype=int)
    recent = reds[-min(len(reds), 120):]
    zones = np.array([
        (recent <= 11).sum(axis=1).mean(),
        ((recent >= 12) & (recent <= 22)).sum(axis=1).mean(),
        (recent >= 23).sum(axis=1).mean(),
    ])
    zone_sizes = np.array([11.0, 11.0, 11.0])
    zone_density = zones / np.maximum(zone_sizes, 1.0)
    values = np.zeros(33, dtype=float)
    values[:11] = zone_density[0]
    values[11:22] = zone_density[1]
    values[22:] = zone_density[2]
    return _normalize(values + 1e-9)


def _blue_transition_score(matrix: np.ndarray, alpha: float) -> np.ndarray:
    if len(matrix) < 2:
        return np.full(16, 1.0 / 16.0)
    sequence = np.argmax(matrix, axis=1)
    previous = int(sequence[-1])
    transitions = np.zeros((16, 16), dtype=float)
    steps = np.zeros(16, dtype=float)
    parity = np.zeros((2, 2), dtype=float)
    zones = np.zeros((4, 4), dtype=float)
    for left, right in zip(sequence[:-1], sequence[1:]):
        transitions[int(left), int(right)] += 1.0
        steps[abs(int(right) - int(left))] += 1.0
        parity[int(left) % 2, int(right) % 2] += 1.0
        zones[int(left) // 4, int(right) // 4] += 1.0
    candidates = np.arange(16, dtype=int)
    direct = (transitions[previous] + alpha) / (transitions[previous].sum() + alpha * 16.0)
    step = (steps[np.abs(candidates - previous)] + alpha) / (steps.sum() + alpha * 16.0)
    parity_part = (
        parity[previous % 2, candidates % 2] + alpha
    ) / (parity[previous % 2].sum() + alpha * 2.0)
    zone_part = (
        zones[previous // 4, candidates // 4] + alpha
    ) / (zones[previous // 4].sum() + alpha * 4.0)
    return _normalize(0.45 * direct + 0.25 * step + 0.15 * parity_part + 0.15 * zone_part)


class PatternMemoryModel:
    name = "pattern_memory"

    def __init__(self, feature_config: dict[str, Any], model_config: dict[str, Any] | None = None):
        self.feature_config = dict(feature_config)
        self.model_config = dict(model_config or {})

    def fit(self, history: pd.DataFrame) -> "PatternMemoryModel":
        red_matrix = occurrence_matrix(history, "red")
        blue_matrix = occurrence_matrix(history, "blue")
        decay = float(self.model_config.get("decay_factor", self.feature_config.get("decay_factor", 0.98)))
        alpha = float(self.model_config.get("transition_alpha", 1.0))
        temperature = float(self.model_config.get("temperature", 1.35))

        red_scores = (
            0.18 * _window_frequency(red_matrix, 30)
            + 0.12 * _window_frequency(red_matrix, 100)
            + 0.18 * _decayed_frequency(red_matrix, decay)
            + 0.22 * _omission_score(red_matrix)
            + 0.22 * _red_transition_score(history, red_matrix, alpha)
            + 0.08 * _red_zone_score(history)
        )
        blue_scores = (
            0.18 * _window_frequency(blue_matrix, 30)
            + 0.10 * _window_frequency(blue_matrix, 100)
            + 0.14 * _decayed_frequency(blue_matrix, decay)
            + 0.24 * _omission_score(blue_matrix)
            + 0.34 * _blue_transition_score(blue_matrix, alpha)
        )
        self.scores_ = {
            "red": _tempered(red_scores, temperature),
            "blue": _tempered(blue_scores, temperature),
        }
        self.model_modes_ = {
            "red": "omission_transition_zone_memory",
            "blue": "omission_step_transition_memory",
        }
        return self

    def predict(self, history: pd.DataFrame, target_idx: int) -> dict[str, np.ndarray]:
        del history, target_idx
        return {key: value.copy() for key, value in self.scores_.items()}
