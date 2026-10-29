from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_handler.cleaner import assert_clean_sequence, assert_left_truncated
from src.feature_engineer.base_stats import occurrence_matrix


BASE_COLUMNS = [
    "overall_frequency",
    "current_gap",
    "mean_gap",
    "max_gap",
    "time_decay_score",
]

RED_PAIRWISE_COLUMNS = [
    "pair_frequency",
    "pair_residual",
    "pair_conditional_probability",
    "pair_omission_pressure",
]

RED_STRUCTURE_COLUMNS = [
    "structure_completion_score",
    "structure_sum_similarity",
    "structure_span_similarity",
    "structure_odd_similarity",
    "structure_zone_similarity",
    "structure_extreme_prior",
]

RED_DYNAMIC_COLUMNS = [
    "acf_lag1",
    "acf_lag2",
    "acf_lag3",
    "rolling_volatility",
    "gap_z_score",
    "low_frequency_energy",
    "conditional_predictability",
]

CALENDAR_CONTEXT_COLUMNS = [
    "calendar_weekday_frequency",
    "calendar_month_frequency",
    "calendar_quarter_frequency",
    "calendar_fixed_holiday_frequency",
    "calendar_recent_drift",
]

BLUE_SEQUENCE_COLUMNS = [
    "transition_probability",
    "step_probability",
    "parity_transition_probability",
    "zone_transition_probability",
    "distance_from_previous",
    "repeat_from_previous",
    "adjacent_to_previous",
]

BLUE_HIGH_ORDER_COLUMNS = [
    "second_order_transition_probability",
    "variable_markov_probability",
    "gap_std_pressure",
    "gap_skew_pressure",
    "repeat_run_pressure",
]

RED_BLUE_CROSS_COLUMNS = [
    "cross_red_blue_conditional_frequency",
    "cross_recent_blue_red_shift",
]

BLUE_RED_CROSS_COLUMNS = [
    "cross_red_sum_blue_frequency",
    "cross_red_span_blue_frequency",
    "cross_red_odd_blue_frequency",
    "cross_red_zone_blue_frequency",
    "cross_red_structure_blue_shrinkage",
]

FEATURE_CACHE_VERSION = 3
_FEATURE_MATRIX_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, np.ndarray | None]] = OrderedDict()


def clear_feature_cache() -> None:
    """Clear the in-process feature matrix cache.

    The cache is deliberately process-local: it accelerates repeated feature
    construction during one backtest/web run without persisting data across
    program restarts.
    """
    _FEATURE_MATRIX_CACHE.clear()


def feature_cache_info() -> dict[str, int]:
    return {"entries": len(_FEATURE_MATRIX_CACHE), "version": FEATURE_CACHE_VERSION}


def _frame_signature(frame: pd.DataFrame) -> tuple:
    if frame.empty:
        return (0, None, None, None, None, 0, 0)
    idx_values = frame["idx"].to_numpy(dtype=np.int64) if "idx" in frame else np.arange(len(frame), dtype=np.int64)
    period_first = str(frame.iloc[0]["period"]) if "period" in frame else ""
    period_last = str(frame.iloc[-1]["period"]) if "period" in frame else ""
    red_columns = [f"r{index}" for index in range(1, 7)]
    available_red = [column for column in red_columns if column in frame]
    red_checksum = int(
        np.dot(
            frame[available_red].to_numpy(dtype=np.int64).reshape(-1),
            np.arange(1, len(frame) * max(len(available_red), 1) + 1, dtype=np.int64),
        )
    ) if available_red else 0
    blue_checksum = int(
        np.dot(
            frame["blue"].to_numpy(dtype=np.int64),
            np.arange(1, len(frame) + 1, dtype=np.int64),
        )
    ) if "blue" in frame else 0
    return (
        int(len(frame)),
        int(idx_values[0]),
        int(idx_values[-1]),
        period_first,
        period_last,
        red_checksum,
        blue_checksum,
    )


def _cache_get(key: tuple, enabled: bool) -> tuple[pd.DataFrame, np.ndarray | None] | None:
    if not enabled:
        return None
    item = _FEATURE_MATRIX_CACHE.get(key)
    if item is None:
        return None
    _FEATURE_MATRIX_CACHE.move_to_end(key)
    features, labels = item
    return features.copy(deep=False), None if labels is None else labels.copy()


def _cache_put(
    key: tuple,
    features: pd.DataFrame,
    labels: np.ndarray | None,
    *,
    enabled: bool,
    max_entries: int,
) -> None:
    if not enabled or int(max_entries) <= 0:
        return
    _FEATURE_MATRIX_CACHE[key] = (features.copy(deep=False), None if labels is None else labels.copy())
    _FEATURE_MATRIX_CACHE.move_to_end(key)
    while len(_FEATURE_MATRIX_CACHE) > int(max_entries):
        _FEATURE_MATRIX_CACHE.popitem(last=False)


def feature_columns(
    windows: Iterable[int],
    color: str | None = None,
    *,
    include_cross_feature: bool = False,
    include_calendar_feature: bool = True,
) -> list[str]:
    columns = [
        *[f"frequency_{int(window)}" for window in windows],
        *BASE_COLUMNS,
    ]
    if color == "red":
        for window in windows:
            columns.extend(f"{name}_{int(window)}" for name in RED_PAIRWISE_COLUMNS)
        columns.extend(RED_STRUCTURE_COLUMNS)
        columns.extend(RED_DYNAMIC_COLUMNS)
        if include_calendar_feature:
            columns.extend(CALENDAR_CONTEXT_COLUMNS)
        if include_cross_feature:
            columns.extend(RED_BLUE_CROSS_COLUMNS)
    if color == "blue":
        columns.extend(BLUE_SEQUENCE_COLUMNS)
        columns.extend(BLUE_HIGH_ORDER_COLUMNS)
        if include_calendar_feature:
            columns.extend(CALENDAR_CONTEXT_COLUMNS)
        if include_cross_feature:
            columns.extend(BLUE_RED_CROSS_COLUMNS)
    return columns


def _base_feature_columns(windows: Iterable[int]) -> list[str]:
    return [
        *[f"frequency_{int(window)}" for window in windows],
        *BASE_COLUMNS,
    ]


def _feature_states(matrix: np.ndarray, windows: list[int], decay: float) -> np.ndarray:
    periods, number_count = matrix.shape
    columns = _base_feature_columns(windows)
    states = np.zeros((periods + 1, number_count, len(columns)), dtype=float)
    prefix = np.vstack([np.zeros((1, number_count)), np.cumsum(matrix, axis=0)])
    last_seen = np.full(number_count, -1, dtype=int)
    gap_sum = np.zeros(number_count, dtype=float)
    gap_count = np.zeros(number_count, dtype=int)
    max_gap = np.zeros(number_count, dtype=float)
    decayed = np.zeros(number_count, dtype=float)
    decay_weight = 0.0

    for target in range(periods + 1):
        values: list[np.ndarray] = []
        for window in windows:
            start = max(0, target - window)
            denominator = max(target - start, 1)
            values.append((prefix[target] - prefix[start]) / denominator)
        values.append(prefix[target] / max(target, 1))
        current_gap = np.where(last_seen >= 0, target - 1 - last_seen, target)
        values.append(current_gap.astype(float))
        values.append(np.where(gap_count > 0, gap_sum / np.maximum(gap_count, 1), current_gap))
        values.append(np.maximum(max_gap, current_gap))
        values.append(decayed / max(decay_weight, 1e-12))
        states[target] = np.column_stack(values)

        if target == periods:
            break
        decayed = decay * decayed + matrix[target]
        decay_weight = decay * decay_weight + 1.0
        appeared = np.flatnonzero(matrix[target])
        for index in appeared:
            if last_seen[index] >= 0:
                gap = target - last_seen[index] - 1
                gap_sum[index] += gap
                gap_count[index] += 1
                max_gap[index] = max(max_gap[index], gap)
            last_seen[index] = target
    return states


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def _gaussian_similarity(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    z_score = (np.asarray(values, dtype=float) - float(mean)) / max(float(std), 0.5)
    return np.exp(-0.5 * z_score * z_score)


def _red_pairwise_states(matrix: np.ndarray, windows: list[int], top_k: int = 10) -> np.ndarray:
    """Strictly left-truncated red co-occurrence features.

    Each row describes how one candidate number connects to the current
    high-support context.  The context is rebuilt from prior data only, so the
    target period's result never leaks into its feature row.
    """
    periods, number_count = matrix.shape
    states = np.zeros(
        (periods + 1, number_count, len(windows) * len(RED_PAIRWISE_COLUMNS)),
        dtype=float,
    )
    pair_last_seen = np.full((number_count, number_count), -1, dtype=int)
    window_counts = {
        int(window): np.zeros(number_count, dtype=float)
        for window in windows
    }
    window_pair_counts = {
        int(window): np.zeros((number_count, number_count), dtype=float)
        for window in windows
    }

    for target in range(periods + 1):
        if target <= 0:
            states[target] = 0.0
        else:
            wide_window = int(max(windows))
            wide_denominator = max(min(wide_window, target), 1)
            context_score = window_counts[wide_window] / wide_denominator
            ranked_context = list(np.argsort(context_score)[::-1][: max(1, int(top_k))])
            feature_blocks: list[np.ndarray] = []
            current_omission = np.maximum(
                0.0,
                target - 1 - pair_last_seen.astype(float),
            )
            unseen_mask = pair_last_seen < 0
            current_omission[unseen_mask] = float(target)
            for window in windows:
                window = int(window)
                denominator = max(min(window, target), 1)
                freq = window_counts[window] / denominator
                co_count = window_pair_counts[window]
                observed = co_count / denominator
                expected = np.outer(freq, freq)
                residual = observed - expected
                count_i = np.maximum(co_count.diagonal(), 1e-12)
                conditional = co_count / count_i[:, None]
                context = np.asarray(ranked_context, dtype=int)
                context_size = max(1, int(context.size))
                in_context = np.zeros(number_count, dtype=bool)
                in_context[context] = True
                peer_denominator = np.where(in_context, max(context_size - 1, 1), context_size)
                observed_sum = observed[:, context].sum(axis=1)
                residual_sum = residual[:, context].sum(axis=1)
                conditional_sum = conditional[:, context].sum(axis=1)
                omission_values = np.clip(current_omission[:, context] / max(int(window), 1), 0.0, 3.0)
                omission_sum = omission_values.sum(axis=1)
                context_indices = np.flatnonzero(in_context)
                observed_sum[context_indices] -= observed[context_indices, context_indices]
                residual_sum[context_indices] -= residual[context_indices, context_indices]
                conditional_sum[context_indices] -= conditional[context_indices, context_indices]
                omission_sum[context_indices] -= np.clip(
                    current_omission[context_indices, context_indices] / max(int(window), 1),
                    0.0,
                    3.0,
                )
                block = np.column_stack([
                    observed_sum / peer_denominator,
                    residual_sum / peer_denominator,
                    conditional_sum / peer_denominator,
                    omission_sum / peer_denominator,
                ])
                feature_blocks.append(block)
            states[target] = np.concatenate(feature_blocks, axis=1)

        if target == periods:
            break
        row = matrix[target].astype(float)
        pair_update = np.outer(row, row)
        for window in windows:
            window = int(window)
            window_counts[window] += row
            window_pair_counts[window] += pair_update
            expired_index = target - window
            if expired_index >= 0:
                expired = matrix[expired_index].astype(float)
                window_counts[window] -= expired
                window_pair_counts[window] -= np.outer(expired, expired)
        appeared = np.flatnonzero(matrix[target])
        for left_pos, left in enumerate(appeared):
            for right in appeared[left_pos + 1:]:
                pair_last_seen[left, right] = target
                pair_last_seen[right, left] = target
    return states


def _red_structure_states(matrix: np.ndarray, profile_window: int = 300) -> np.ndarray:
    """Approximate pre-combination structure as a candidate-level feature.

    For each candidate, the system forms a strictly-left truncated hypothetical
    six-number set from that candidate plus the five strongest current context
    numbers.  This moves sum/span/odd/zone information from a post-filter into
    the model's probability features.
    """
    periods, number_count = matrix.shape
    states = np.zeros((periods + 1, number_count, len(RED_STRUCTURE_COLUMNS)), dtype=float)
    prefix = np.vstack([np.zeros((1, number_count)), np.cumsum(matrix, axis=0)])
    numbers = np.arange(1, number_count + 1, dtype=float)
    row_sums = matrix @ numbers
    row_indices = [np.flatnonzero(row) + 1 for row in matrix]
    row_spans = np.array([
        float(values.max() - values.min()) if len(values) else 27.0
        for values in row_indices
    ])
    row_odds = matrix @ (numbers % 2)
    zone_counts = np.column_stack([
        matrix[:, :11].sum(axis=1),
        matrix[:, 11:22].sum(axis=1),
        matrix[:, 22:].sum(axis=1),
    ]).astype(float)
    zone_center_fixed = np.array([2.0, 2.0, 2.0], dtype=float)
    row_zone_distance = np.linalg.norm(zone_counts - zone_center_fixed, axis=1)
    row_extreme = (
        (row_sums < 70)
        | (row_sums > 140)
        | (row_odds == 0)
        | (row_odds == 6)
        | (row_spans < 12)
        | (row_spans > 31)
    ).astype(float)

    profile_arrays = {
        "sum": row_sums,
        "span": row_spans,
        "odd_count": row_odds,
        "zone_distance": row_zone_distance,
        "extreme": row_extreme,
    }
    profile_prefix = {
        name: np.concatenate([[0.0], np.cumsum(values)])
        for name, values in profile_arrays.items()
    }
    profile_sq_prefix = {
        name: np.concatenate([[0.0], np.cumsum(values * values)])
        for name, values in profile_arrays.items()
    }

    def rolling_profile(name: str, target: int) -> tuple[float, float]:
        if target <= 0:
            defaults = {
                "sum": (102.0, 25.0),
                "span": (27.0, 5.0),
                "odd_count": (3.0, 1.25),
                "zone_distance": (0.0, 1.50),
                "extreme": (0.0, 1.0),
            }
            return defaults[name]
        start = max(0, target - int(profile_window))
        count = max(target - start, 1)
        total = float(profile_prefix[name][target] - profile_prefix[name][start])
        sq_total = float(profile_sq_prefix[name][target] - profile_sq_prefix[name][start])
        mean = total / count
        variance = max(0.0, sq_total / count - mean * mean)
        floor = 0.1 if name == "extreme" else 0.5
        return float(mean), max(float(np.sqrt(variance)), floor)

    for target in range(periods + 1):
        profile = {
            name: rolling_profile(name, target)
            for name in profile_arrays
        }
        if target <= 0:
            context_score = np.ones(number_count, dtype=float)
        else:
            start_context = max(0, target - min(100, target))
            context_score = (prefix[target] - prefix[start_context]) / max(target - start_context, 1)

        ranked_context = list(np.argsort(context_score)[::-1])
        for candidate in range(number_count):
            peers = [peer for peer in ranked_context if peer != candidate][:5]
            if len(peers) < 5:
                peers = [peer for peer in range(number_count) if peer != candidate][:5]
            selected = np.sort(np.array([candidate + 1, *[peer + 1 for peer in peers]], dtype=int))
            total = float(selected.sum())
            span = float(selected[-1] - selected[0])
            odd = float((selected % 2).sum())
            zones = np.array([
                np.sum(selected <= 11),
                np.sum((selected >= 12) & (selected <= 22)),
                np.sum(selected >= 23),
            ], dtype=float)
            zone_center = np.array([2.0, 2.0, 2.0], dtype=float)
            zone_distance = float(np.linalg.norm(zones - zone_center))
            extreme_value = float(
                total < 70 or total > 140 or odd in {0.0, 6.0} or span < 12 or span > 31
            )
            sum_sim = float(_gaussian_similarity(total, *profile["sum"]))
            span_sim = float(_gaussian_similarity(span, *profile["span"]))
            odd_sim = float(_gaussian_similarity(odd, *profile["odd_count"]))
            zone_sim = float(_gaussian_similarity(zone_distance, *profile["zone_distance"]))
            extreme_prior = float(_gaussian_similarity(extreme_value, *profile["extreme"]))
            states[target, candidate] = np.array([
                float(np.mean([sum_sim, span_sim, odd_sim, zone_sim, extreme_prior])),
                sum_sim,
                span_sim,
                odd_sim,
                zone_sim,
                extreme_prior,
            ])
    return states


def _binary_entropy(probability: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _red_dynamic_states(matrix: np.ndarray, window: int = 300) -> np.ndarray:
    """Strictly-left truncated per-number dynamic features.

    These features implement the document's high-value feature bucket:
    short-lag autocorrelation, rolling volatility, extreme omission z-score,
    low-frequency Fourier energy, and conditional predictability.  They are
    intentionally compact (7 columns) so the added signal does not explode the
    feature dimension like a full 33x33 lag tensor would.
    """
    periods, number_count = matrix.shape
    states = np.zeros((periods + 1, number_count, len(RED_DYNAMIC_COLUMNS)), dtype=float)
    max_window = max(30, int(window))
    last_seen = np.full(number_count, -1, dtype=int)
    gap_lists: list[list[int]] = [[] for _ in range(number_count)]
    two_pi = 2.0 * np.pi

    for target in range(periods + 1):
        start = max(0, target - max_window)
        prior = matrix[start:target].astype(float)
        length = int(len(prior))
        if length >= 8:
            frequency = prior.mean(axis=0)
            variance = np.maximum(frequency * (1.0 - frequency), 1e-6)
            acf_values: list[np.ndarray] = []
            for lag in (1, 2, 3):
                if length <= lag:
                    acf_values.append(np.zeros(number_count, dtype=float))
                    continue
                joint = (prior[lag:] * prior[:-lag]).mean(axis=0)
                acf = (joint - frequency * frequency) / variance
                acf_values.append(np.clip(acf, -1.0, 1.0))

            chunk = max(5, min(20, length // 6 or 5))
            rolling_means = []
            for offset in range(0, max(0, length - chunk + 1), chunk):
                rolling_means.append(prior[offset:offset + chunk].mean(axis=0))
            volatility = (
                np.std(np.stack(rolling_means, axis=0), axis=0)
                if len(rolling_means) >= 2
                else np.zeros(number_count, dtype=float)
            )

            positions = np.arange(length, dtype=float)
            centered = prior - frequency
            sin_component = np.abs(np.sin(two_pi * positions / max(length, 1)) @ centered) / max(length, 1)
            cos_component = np.abs(np.cos(two_pi * positions / max(length, 1)) @ centered) / max(length, 1)
            low_frequency = np.sqrt(sin_component * sin_component + cos_component * cos_component)

            if length >= 2:
                prev = prior[:-1]
                curr = prior[1:]
                prev_hit = prev.sum(axis=0)
                prev_miss = float(len(prev)) - prev_hit
                p_hit_after_hit = (np.sum(prev * curr, axis=0) + 1.0) / (prev_hit + 2.0)
                p_hit_after_miss = (np.sum((1.0 - prev) * curr, axis=0) + 1.0) / (prev_miss + 2.0)
                base_entropy = np.asarray([_binary_entropy(value) for value in frequency], dtype=float)
                conditional_entropy = (
                    (prev_hit / max(len(prev), 1)) * np.asarray([_binary_entropy(value) for value in p_hit_after_hit])
                    + (prev_miss / max(len(prev), 1)) * np.asarray([_binary_entropy(value) for value in p_hit_after_miss])
                )
                predictability = np.clip(base_entropy - conditional_entropy, -1.0, 1.0)
            else:
                predictability = np.zeros(number_count, dtype=float)
        else:
            acf_values = [np.zeros(number_count, dtype=float) for _ in range(3)]
            volatility = np.zeros(number_count, dtype=float)
            low_frequency = np.zeros(number_count, dtype=float)
            predictability = np.zeros(number_count, dtype=float)

        omission = np.where(last_seen >= 0, target - 1 - last_seen, target).astype(float)
        gap_z = np.zeros(number_count, dtype=float)
        for number in range(number_count):
            gaps = np.asarray(gap_lists[number], dtype=float)
            if len(gaps) >= 2:
                mean_gap = float(gaps.mean())
                std_gap = max(float(gaps.std(ddof=0)), 1.0)
            else:
                mean_gap = float(max_window / 6.0)
                std_gap = max(1.0, mean_gap * 0.75)
            gap_z[number] = float(np.clip((omission[number] - mean_gap) / std_gap, -5.0, 5.0))

        states[target] = np.column_stack([
            acf_values[0],
            acf_values[1],
            acf_values[2],
            volatility,
            gap_z,
            low_frequency,
            predictability,
        ])
        if target == periods:
            break
        appeared = np.flatnonzero(matrix[target])
        for index in appeared:
            if last_seen[index] >= 0:
                gap_lists[index].append(target - int(last_seen[index]) - 1)
            last_seen[index] = target
    return states


def _blue_sequence_states(matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Build strictly left-truncated transition and movement features.

    A blue draw contains one value, so its useful sequence descriptions differ
    from the six-number red set.  At target ``t`` every count below contains
    transitions ending no later than ``t-1``; the target result is never read.
    """
    periods, number_count = matrix.shape
    if number_count != 16:
        raise ValueError("蓝球序列特征要求 16 个候选号码")
    states = np.zeros((periods + 1, number_count, len(BLUE_SEQUENCE_COLUMNS)), dtype=float)
    transitions = np.zeros((number_count, number_count), dtype=float)
    steps = np.zeros(number_count, dtype=float)
    parity_transitions = np.zeros((2, 2), dtype=float)
    zone_transitions = np.zeros((4, 4), dtype=float)
    candidates = np.arange(number_count, dtype=int)
    previous: int | None = None

    for target in range(periods + 1):
        if previous is None:
            transition_probability = np.full(number_count, 1.0 / number_count)
            step_probability = np.full(number_count, 1.0 / number_count)
            parity_probability = np.full(number_count, 0.5)
            zone_probability = np.full(number_count, 0.25)
            distance = np.zeros(number_count, dtype=float)
            repeat = np.zeros(number_count, dtype=float)
            adjacent = np.zeros(number_count, dtype=float)
        else:
            transition_probability = (
                transitions[previous] + alpha
            ) / (transitions[previous].sum() + alpha * number_count)
            absolute_steps = np.abs(candidates - previous)
            step_probability = (
                steps[absolute_steps] + alpha
            ) / (steps.sum() + alpha * number_count)
            previous_parity = previous % 2
            parity_probability = (
                parity_transitions[previous_parity, candidates % 2] + alpha
            ) / (parity_transitions[previous_parity].sum() + alpha * 2.0)
            previous_zone = previous // 4
            zone_probability = (
                zone_transitions[previous_zone, candidates // 4] + alpha
            ) / (zone_transitions[previous_zone].sum() + alpha * 4.0)
            distance = absolute_steps.astype(float) / 15.0
            repeat = (candidates == previous).astype(float)
            adjacent = (absolute_steps == 1).astype(float)

        states[target] = np.column_stack([
            transition_probability,
            step_probability,
            parity_probability,
            zone_probability,
            distance,
            repeat,
            adjacent,
        ])
        if target == periods:
            break

        current = int(np.argmax(matrix[target]))
        if previous is not None:
            transitions[previous, current] += 1.0
            steps[abs(current - previous)] += 1.0
            parity_transitions[previous % 2, current % 2] += 1.0
            zone_transitions[previous // 4, current // 4] += 1.0
        previous = current
    return states


def _blue_high_order_states(matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Build second-order and variable-order blue sequence features.

    The variable-order score is a small context-tree style Bayesian average of
    marginal, first-order, second-order and third-order transitions.  Contexts
    with little evidence automatically shrink back toward lower orders.
    """
    periods, number_count = matrix.shape
    if number_count != 16:
        raise ValueError("蓝球高阶序列特征要求 16 个候选号码")
    states = np.zeros((periods + 1, number_count, len(BLUE_HIGH_ORDER_COLUMNS)), dtype=float)
    sequence: list[int] = []
    order0 = np.zeros(number_count, dtype=float)
    order1: dict[tuple[int], np.ndarray] = {}
    order2: dict[tuple[int, int], np.ndarray] = {}
    order3: dict[tuple[int, int, int], np.ndarray] = {}
    gap_lists: list[list[int]] = [[] for _ in range(number_count)]
    last_seen = np.full(number_count, -1, dtype=int)

    def probability_for_context(context: tuple[int, ...]) -> tuple[np.ndarray, float]:
        if len(context) == 0:
            counts = order0
        elif len(context) == 1:
            counts = order1.get(context, np.zeros(number_count, dtype=float))
        elif len(context) == 2:
            counts = order2.get(context, np.zeros(number_count, dtype=float))
        else:
            counts = order3.get(context, np.zeros(number_count, dtype=float))
        total = float(counts.sum())
        probability = (counts + alpha) / (total + alpha * number_count)
        reliability = total / (total + 8.0 + 4.0 * len(context))
        return probability, float(reliability)

    for target in range(periods + 1):
        if len(sequence) < 2:
            second_order = np.full(number_count, 1.0 / number_count)
        else:
            second_order, _ = probability_for_context(tuple(sequence[-2:]))

        contexts = [tuple()]
        if len(sequence) >= 1:
            contexts.append(tuple(sequence[-1:]))
        if len(sequence) >= 2:
            contexts.append(tuple(sequence[-2:]))
        if len(sequence) >= 3:
            contexts.append(tuple(sequence[-3:]))
        probabilities: list[np.ndarray] = []
        weights: list[float] = []
        for context in contexts:
            probability, reliability = probability_for_context(context)
            probabilities.append(probability)
            weights.append(max(0.08, reliability))
        weights_array = np.asarray(weights, dtype=float)
        weights_array /= max(float(weights_array.sum()), 1e-12)
        variable_markov = np.sum(
            np.stack(probabilities, axis=0) * weights_array[:, None],
            axis=0,
        )

        omission = np.where(last_seen >= 0, target - 1 - last_seen, target).astype(float)
        gap_std_pressure = np.zeros(number_count, dtype=float)
        gap_skew_pressure = np.zeros(number_count, dtype=float)
        for number in range(number_count):
            gaps = np.asarray(gap_lists[number], dtype=float)
            if len(gaps) < 2:
                mean_gap = 16.0
                std_gap = 8.0
                skew = 0.0
            else:
                mean_gap = float(gaps.mean())
                std_gap = max(float(gaps.std(ddof=0)), 1.0)
                centered = gaps - mean_gap
                skew = float(np.mean(centered ** 3) / max(std_gap ** 3, 1e-9))
            z_score = (omission[number] - mean_gap) / max(std_gap, 1.0)
            gap_std_pressure[number] = float(_sigmoid(z_score))
            gap_skew_pressure[number] = float(_sigmoid(z_score + 0.18 * skew))

        repeat_run_pressure = np.zeros(number_count, dtype=float)
        if sequence:
            last = int(sequence[-1])
            run_length = 1
            for previous in reversed(sequence[:-1]):
                if int(previous) != last:
                    break
                run_length += 1
            repeat_run_pressure[last] = min(1.0, run_length / 3.0)

        states[target] = np.column_stack([
            second_order,
            variable_markov,
            gap_std_pressure,
            gap_skew_pressure,
            repeat_run_pressure,
        ])

        if target == periods:
            break
        current = int(np.argmax(matrix[target]))
        order0[current] += 1.0
        if len(sequence) >= 1:
            order1.setdefault(tuple(sequence[-1:]), np.zeros(number_count, dtype=float))[current] += 1.0
        if len(sequence) >= 2:
            order2.setdefault(tuple(sequence[-2:]), np.zeros(number_count, dtype=float))[current] += 1.0
        if len(sequence) >= 3:
            order3.setdefault(tuple(sequence[-3:]), np.zeros(number_count, dtype=float))[current] += 1.0
        if last_seen[current] >= 0:
            gap_lists[current].append(target - int(last_seen[current]) - 1)
        last_seen[current] = target
        sequence.append(current)
    return states


def _next_draw_date(last_date: pd.Timestamp) -> pd.Timestamp:
    weekdays = {1, 3, 6}
    current = pd.Timestamp(last_date).normalize()
    for offset in range(1, 8):
        candidate = current + timedelta(days=offset)
        if int(candidate.weekday()) in weekdays:
            return candidate
    return current + timedelta(days=2)


def _fixed_holiday_near(value: pd.Timestamp) -> bool:
    value = pd.Timestamp(value).normalize()
    fixed = [
        pd.Timestamp(year=value.year, month=1, day=1),
        pd.Timestamp(year=value.year, month=5, day=1),
        pd.Timestamp(year=value.year, month=10, day=1),
    ]
    return any(abs((value - day).days) <= 3 for day in fixed)


def _calendar_number_states(
    frame: pd.DataFrame | None,
    matrix: np.ndarray,
    *,
    color: str,
) -> np.ndarray:
    periods, number_count = matrix.shape
    states = np.zeros((periods + 1, number_count, len(CALENDAR_CONTEXT_COLUMNS)), dtype=float)
    if frame is None or "date" not in frame.columns:
        return states
    if "date_provenance" in frame.columns:
        provenance = frame["date_provenance"].astype(str).str.lower()
        if not provenance.eq("source").any():
            return states
    dates = pd.to_datetime(frame["date"], errors="coerce").reset_index(drop=True)
    if dates.isna().any():
        return states
    weekday_counts = np.zeros((7, number_count), dtype=float)
    weekday_total = np.zeros(7, dtype=float)
    month_counts = np.zeros((12, number_count), dtype=float)
    month_total = np.zeros(12, dtype=float)
    quarter_counts = np.zeros((4, number_count), dtype=float)
    quarter_total = np.zeros(4, dtype=float)
    holiday_counts = np.zeros((2, number_count), dtype=float)
    holiday_total = np.zeros(2, dtype=float)
    prefix = np.vstack([np.zeros((1, number_count)), np.cumsum(matrix, axis=0)])

    for target in range(periods + 1):
        if target < periods:
            target_date = pd.Timestamp(dates.iloc[target]).normalize()
        else:
            target_date = _next_draw_date(pd.Timestamp(dates.iloc[-1]))
        weekday = int(target_date.weekday())
        month = int(target_date.month) - 1
        quarter = int((int(target_date.month) - 1) // 3)
        holiday = int(_fixed_holiday_near(target_date))
        weekday_frequency = (weekday_counts[weekday] + 1.0) / (
            weekday_total[weekday] + number_count
        )
        month_frequency = (month_counts[month] + 1.0) / (
            month_total[month] + number_count
        )
        quarter_frequency = (quarter_counts[quarter] + 1.0) / (
            quarter_total[quarter] + number_count
        )
        holiday_frequency = (holiday_counts[holiday] + 1.0) / (
            holiday_total[holiday] + number_count
        )
        recent_start = max(0, target - 300)
        older_start = max(0, target - 900)
        recent = (prefix[target] - prefix[recent_start]) / max(target - recent_start, 1)
        older = (prefix[recent_start] - prefix[older_start]) / max(recent_start - older_start, 1)
        drift = recent - older
        states[target] = np.column_stack([
            weekday_frequency,
            month_frequency,
            quarter_frequency,
            holiday_frequency,
            drift,
        ])

        if target == periods:
            break
        row = matrix[target].astype(float)
        weekday_counts[int(dates.iloc[target].weekday())] += row
        weekday_total[int(dates.iloc[target].weekday())] += float(row.sum())
        month_index = int(dates.iloc[target].month) - 1
        month_counts[month_index] += row
        month_total[month_index] += float(row.sum())
        quarter_index = int((int(dates.iloc[target].month) - 1) // 3)
        quarter_counts[quarter_index] += row
        quarter_total[quarter_index] += float(row.sum())
        holiday_index = int(_fixed_holiday_near(pd.Timestamp(dates.iloc[target])))
        holiday_counts[holiday_index] += row
        holiday_total[holiday_index] += float(row.sum())
    return states


def _red_blue_cross_states(frame: pd.DataFrame | None, matrix: np.ndarray, color: str) -> np.ndarray:
    periods, number_count = matrix.shape
    if frame is None or "blue" not in frame.columns:
        columns = len(RED_BLUE_CROSS_COLUMNS) if color == "red" else len(BLUE_RED_CROSS_COLUMNS)
        return np.zeros((periods + 1, number_count, columns), dtype=float)
    blue_values = frame["blue"].astype(int).to_numpy()
    red_columns = [f"r{index}" for index in range(1, 7)]
    has_red = all(column in frame.columns for column in red_columns)
    reds = frame[red_columns].astype(int).to_numpy() if has_red else None
    if color == "red":
        states = np.zeros((periods + 1, number_count, len(RED_BLUE_CROSS_COLUMNS)), dtype=float)
        for target in range(1, periods + 1):
            start = max(0, target - 600)
            recent_blue = int(blue_values[target - 1])
            prior_blues = blue_values[start:target]
            prior_red = matrix[start:target]
            global_rate = (prior_red.sum(axis=0) + 1.0) / (max(target - start, 1) + 33.0)
            same_blue = prior_blues == recent_blue
            conditional = (
                (prior_red[same_blue].sum(axis=0) + 1.0) / (float(np.sum(same_blue)) + 33.0)
                if np.any(same_blue)
                else global_rate
            )
            states[target, :, 0] = 0.45 * conditional + 0.55 * global_rate
            if target >= 2:
                offset = max(start, 1)
                previous_blue = blue_values[offset:target]
                current_red = matrix[offset:target]
                transition_mask = np.abs(previous_blue - recent_blue) <= 1
                shifted = (
                    (current_red[transition_mask].sum(axis=0) + 1.0)
                    / (float(np.sum(transition_mask)) + 33.0)
                    if np.any(transition_mask)
                    else global_rate
                )
                states[target, :, 1] = 0.55 * shifted + 0.45 * global_rate
            else:
                states[target, :, 1] = global_rate
        return states

    states = np.zeros((periods + 1, number_count, len(BLUE_RED_CROSS_COLUMNS)), dtype=float)
    if reds is None:
        return states
    red_sum = reds.sum(axis=1)
    red_span = reds.max(axis=1) - reds.min(axis=1)
    red_odd = (reds % 2).sum(axis=1)
    red_zone_hot = np.stack([
        (reds <= 11).sum(axis=1),
        ((reds >= 12) & (reds <= 22)).sum(axis=1),
        (reds >= 23).sum(axis=1),
    ], axis=1).argmax(axis=1)
    for target in range(1, periods + 1):
        start = max(0, target - 700)
        latest_index = target - 1
        values = blue_values[start:target]
        if len(values) == 0:
            continue
        global_counts = np.bincount(values - 1, minlength=number_count).astype(float) + 1.0
        global_probability = global_counts / max(float(global_counts.sum()), 1.0)
        masks = [
            np.abs(red_sum[start:target] - red_sum[latest_index]) <= 12,
            np.abs(red_span[start:target] - red_span[latest_index]) <= 4,
            np.abs(red_odd[start:target] - red_odd[latest_index]) <= 1,
            red_zone_hot[start:target] == red_zone_hot[latest_index],
        ]
        conditional_probabilities = []
        for mask in masks:
            matched = values[mask]
            if len(matched) < 16:
                probability = global_probability
            else:
                counts = np.bincount(matched - 1, minlength=number_count).astype(float) + 4.0 * global_probability
                probability = counts / max(float(counts.sum()), 1.0)
            conditional_probabilities.append(probability)
        for column, probability in enumerate(conditional_probabilities):
            states[target, :, column] = probability
        states[target, :, 4] = np.mean(np.stack(conditional_probabilities, axis=0), axis=0)
    return states


def _states_for_color(
    frame: pd.DataFrame | None,
    matrix: np.ndarray,
    windows: list[int],
    decay: float,
    color: str,
    include_blue_sequence: bool = True,
    include_cross_feature: bool = False,
    include_calendar_feature: bool = True,
) -> np.ndarray:
    base = _feature_states(matrix, windows, decay)
    blocks = [base]
    if color == "red":
        blocks.append(_red_pairwise_states(matrix, windows))
        blocks.append(_red_structure_states(matrix))
        blocks.append(_red_dynamic_states(matrix, window=max(windows)))
        if include_calendar_feature:
            blocks.append(_calendar_number_states(frame, matrix, color=color))
        if include_cross_feature:
            blocks.append(_red_blue_cross_states(frame, matrix, color))
        return np.concatenate(blocks, axis=2)
    if color == "blue" and include_blue_sequence:
        blocks.append(_blue_sequence_states(matrix))
        blocks.append(_blue_high_order_states(matrix))
        if include_calendar_feature:
            blocks.append(_calendar_number_states(frame, matrix, color=color))
        if include_cross_feature:
            blocks.append(_red_blue_cross_states(frame, matrix, color))
        return np.concatenate(blocks, axis=2)
    return np.concatenate(blocks, axis=2)


def _base_state_at_end(matrix: np.ndarray, windows: list[int], decay: float) -> np.ndarray:
    periods, number_count = matrix.shape
    values: list[np.ndarray] = []
    prefix = matrix.sum(axis=0).astype(float)
    for window in windows:
        window = int(window)
        start = max(0, periods - window)
        denominator = max(periods - start, 1)
        values.append(matrix[start:periods].sum(axis=0).astype(float) / denominator)
    values.append(prefix / max(periods, 1))

    last_seen = np.full(number_count, -1, dtype=int)
    gap_sum = np.zeros(number_count, dtype=float)
    gap_count = np.zeros(number_count, dtype=int)
    max_gap = np.zeros(number_count, dtype=float)
    for number in range(number_count):
        hits = np.flatnonzero(matrix[:, number])
        if len(hits):
            last_seen[number] = int(hits[-1])
        if len(hits) >= 2:
            gaps = np.diff(hits) - 1
            gap_sum[number] = float(gaps.sum())
            gap_count[number] = int(len(gaps))
            max_gap[number] = float(gaps.max())
    current_gap = np.where(last_seen >= 0, periods - 1 - last_seen, periods).astype(float)
    values.append(current_gap)
    values.append(np.where(gap_count > 0, gap_sum / np.maximum(gap_count, 1), current_gap))
    values.append(np.maximum(max_gap, current_gap))
    if periods <= 0:
        values.append(np.zeros(number_count, dtype=float))
    else:
        weights = np.power(float(decay), np.arange(periods - 1, -1, -1, dtype=float))
        decayed = weights @ matrix.astype(float)
        values.append(decayed / max(float(weights.sum()), 1e-12))
    return np.column_stack(values)


def _red_pairwise_state_at_end(matrix: np.ndarray, windows: list[int], top_k: int = 10) -> np.ndarray:
    periods, number_count = matrix.shape
    if periods <= 0:
        return np.zeros((number_count, len(windows) * len(RED_PAIRWISE_COLUMNS)), dtype=float)
    wide_window = int(max(windows))
    wide_start = max(0, periods - wide_window)
    wide_denominator = max(periods - wide_start, 1)
    context_score = matrix[wide_start:periods].sum(axis=0).astype(float) / wide_denominator
    ranked_context = list(np.argsort(context_score)[::-1][: max(1, int(top_k))])

    pair_last_seen = np.full((number_count, number_count), -1, dtype=int)
    for row_index, row in enumerate(matrix):
        appeared = np.flatnonzero(row)
        for left_pos, left in enumerate(appeared):
            for right in appeared[left_pos + 1:]:
                pair_last_seen[left, right] = row_index
                pair_last_seen[right, left] = row_index
    current_omission = np.maximum(0.0, periods - 1 - pair_last_seen.astype(float))
    current_omission[pair_last_seen < 0] = float(periods)

    context = np.asarray(ranked_context, dtype=int)
    context_size = max(1, int(context.size))
    in_context = np.zeros(number_count, dtype=bool)
    in_context[context] = True
    peer_denominator = np.where(in_context, max(context_size - 1, 1), context_size)
    context_indices = np.flatnonzero(in_context)

    feature_blocks: list[np.ndarray] = []
    for window in windows:
        window = int(window)
        start = max(0, periods - window)
        recent = matrix[start:periods].astype(float)
        denominator = max(periods - start, 1)
        counts = recent.sum(axis=0)
        freq = counts / denominator
        co_count = recent.T @ recent
        observed = co_count / denominator
        expected = np.outer(freq, freq)
        residual = observed - expected
        count_i = np.maximum(co_count.diagonal(), 1e-12)
        conditional = co_count / count_i[:, None]
        observed_sum = observed[:, context].sum(axis=1)
        residual_sum = residual[:, context].sum(axis=1)
        conditional_sum = conditional[:, context].sum(axis=1)
        omission_values = np.clip(current_omission[:, context] / max(window, 1), 0.0, 3.0)
        omission_sum = omission_values.sum(axis=1)
        observed_sum[context_indices] -= observed[context_indices, context_indices]
        residual_sum[context_indices] -= residual[context_indices, context_indices]
        conditional_sum[context_indices] -= conditional[context_indices, context_indices]
        omission_sum[context_indices] -= np.clip(
            current_omission[context_indices, context_indices] / max(window, 1),
            0.0,
            3.0,
        )
        feature_blocks.append(np.column_stack([
            observed_sum / peer_denominator,
            residual_sum / peer_denominator,
            conditional_sum / peer_denominator,
            omission_sum / peer_denominator,
        ]))
    return np.concatenate(feature_blocks, axis=1)


def _red_structure_state_at_end(matrix: np.ndarray, profile_window: int = 300) -> np.ndarray:
    periods, number_count = matrix.shape
    numbers = np.arange(1, number_count + 1, dtype=float)
    if periods <= 0:
        context_score = np.ones(number_count, dtype=float)
        profile = {
            "sum": (102.0, 25.0),
            "span": (27.0, 5.0),
            "odd_count": (3.0, 1.25),
            "zone_distance": (0.0, 1.50),
            "extreme": (0.0, 1.0),
        }
    else:
        row_indices = [np.flatnonzero(row) + 1 for row in matrix]
        row_sums = matrix @ numbers
        row_spans = np.array([
            float(values.max() - values.min()) if len(values) else 27.0
            for values in row_indices
        ])
        row_odds = matrix @ (numbers % 2)
        zone_counts = np.column_stack([
            matrix[:, :11].sum(axis=1),
            matrix[:, 11:22].sum(axis=1),
            matrix[:, 22:].sum(axis=1),
        ]).astype(float)
        zone_distance = np.linalg.norm(zone_counts - np.array([2.0, 2.0, 2.0]), axis=1)
        row_extreme = (
            (row_sums < 70)
            | (row_sums > 140)
            | (row_odds == 0)
            | (row_odds == 6)
            | (row_spans < 12)
            | (row_spans > 31)
        ).astype(float)
        profile_arrays = {
            "sum": row_sums,
            "span": row_spans,
            "odd_count": row_odds,
            "zone_distance": zone_distance,
            "extreme": row_extreme,
        }
        start = max(0, periods - int(profile_window))
        profile = {}
        for name, values in profile_arrays.items():
            segment = np.asarray(values[start:periods], dtype=float)
            if len(segment) == 0:
                profile[name] = (0.0, 1.0)
                continue
            floor = 0.1 if name == "extreme" else 0.5
            profile[name] = (float(segment.mean()), max(float(segment.std(ddof=0)), floor))
        context_start = max(0, periods - min(100, periods))
        context_score = matrix[context_start:periods].sum(axis=0) / max(periods - context_start, 1)

    ranked_context = list(np.argsort(context_score)[::-1])
    states = np.zeros((number_count, len(RED_STRUCTURE_COLUMNS)), dtype=float)
    for candidate in range(number_count):
        peers = [peer for peer in ranked_context if peer != candidate][:5]
        if len(peers) < 5:
            peers = [peer for peer in range(number_count) if peer != candidate][:5]
        selected = np.sort(np.array([candidate + 1, *[peer + 1 for peer in peers]], dtype=int))
        total = float(selected.sum())
        span = float(selected[-1] - selected[0])
        odd = float((selected % 2).sum())
        zones = np.array([
            np.sum(selected <= 11),
            np.sum((selected >= 12) & (selected <= 22)),
            np.sum(selected >= 23),
        ], dtype=float)
        zone_distance = float(np.linalg.norm(zones - np.array([2.0, 2.0, 2.0])))
        extreme_value = float(total < 70 or total > 140 or odd in {0.0, 6.0} or span < 12 or span > 31)
        sum_sim = float(_gaussian_similarity(total, *profile["sum"]))
        span_sim = float(_gaussian_similarity(span, *profile["span"]))
        odd_sim = float(_gaussian_similarity(odd, *profile["odd_count"]))
        zone_sim = float(_gaussian_similarity(zone_distance, *profile["zone_distance"]))
        extreme_prior = float(_gaussian_similarity(extreme_value, *profile["extreme"]))
        states[candidate] = np.array([
            float(np.mean([sum_sim, span_sim, odd_sim, zone_sim, extreme_prior])),
            sum_sim,
            span_sim,
            odd_sim,
            zone_sim,
            extreme_prior,
        ])
    return states


def _red_dynamic_state_at_end(matrix: np.ndarray, window: int = 300) -> np.ndarray:
    periods, number_count = matrix.shape
    max_window = max(30, int(window))
    start = max(0, periods - max_window)
    prior = matrix[start:periods].astype(float)
    length = int(len(prior))
    if length >= 8:
        frequency = prior.mean(axis=0)
        variance = np.maximum(frequency * (1.0 - frequency), 1e-6)
        acf_values: list[np.ndarray] = []
        for lag in (1, 2, 3):
            if length <= lag:
                acf_values.append(np.zeros(number_count, dtype=float))
                continue
            joint = (prior[lag:] * prior[:-lag]).mean(axis=0)
            acf_values.append(np.clip((joint - frequency * frequency) / variance, -1.0, 1.0))
        chunk = max(5, min(20, length // 6 or 5))
        rolling_means = [
            prior[offset:offset + chunk].mean(axis=0)
            for offset in range(0, max(0, length - chunk + 1), chunk)
        ]
        volatility = (
            np.std(np.stack(rolling_means, axis=0), axis=0)
            if len(rolling_means) >= 2
            else np.zeros(number_count, dtype=float)
        )
        positions = np.arange(length, dtype=float)
        centered = prior - frequency
        two_pi = 2.0 * np.pi
        sin_component = np.abs(np.sin(two_pi * positions / max(length, 1)) @ centered) / max(length, 1)
        cos_component = np.abs(np.cos(two_pi * positions / max(length, 1)) @ centered) / max(length, 1)
        low_frequency = np.sqrt(sin_component * sin_component + cos_component * cos_component)
        prev = prior[:-1]
        curr = prior[1:]
        prev_hit = prev.sum(axis=0)
        prev_miss = float(len(prev)) - prev_hit
        p_hit_after_hit = (np.sum(prev * curr, axis=0) + 1.0) / (prev_hit + 2.0)
        p_hit_after_miss = (np.sum((1.0 - prev) * curr, axis=0) + 1.0) / (prev_miss + 2.0)
        base_entropy = np.asarray([_binary_entropy(value) for value in frequency], dtype=float)
        conditional_entropy = (
            (prev_hit / max(len(prev), 1)) * np.asarray([_binary_entropy(value) for value in p_hit_after_hit])
            + (prev_miss / max(len(prev), 1)) * np.asarray([_binary_entropy(value) for value in p_hit_after_miss])
        )
        predictability = np.clip(base_entropy - conditional_entropy, -1.0, 1.0)
    else:
        acf_values = [np.zeros(number_count, dtype=float) for _ in range(3)]
        volatility = np.zeros(number_count, dtype=float)
        low_frequency = np.zeros(number_count, dtype=float)
        predictability = np.zeros(number_count, dtype=float)

    last_seen = np.full(number_count, -1, dtype=int)
    gap_z = np.zeros(number_count, dtype=float)
    for number in range(number_count):
        hits = np.flatnonzero(matrix[:, number])
        if len(hits):
            last_seen[number] = int(hits[-1])
        omission = float(periods - 1 - last_seen[number]) if last_seen[number] >= 0 else float(periods)
        if len(hits) >= 2:
            gaps = np.diff(hits) - 1
            mean_gap = float(gaps.mean())
            std_gap = max(float(gaps.std(ddof=0)), 1.0)
        else:
            mean_gap = float(max_window / 6.0)
            std_gap = max(1.0, mean_gap * 0.75)
        gap_z[number] = float(np.clip((omission - mean_gap) / std_gap, -5.0, 5.0))
    return np.column_stack([
        acf_values[0],
        acf_values[1],
        acf_values[2],
        volatility,
        gap_z,
        low_frequency,
        predictability,
    ])


def _blue_sequence_state_at_end(matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    periods, number_count = matrix.shape
    if number_count != 16:
        raise ValueError("蓝球序列特征要求 16 个候选号码")
    if periods <= 0:
        return np.column_stack([
            np.full(number_count, 1.0 / number_count),
            np.full(number_count, 1.0 / number_count),
            np.full(number_count, 0.5),
            np.full(number_count, 0.25),
            np.zeros(number_count, dtype=float),
            np.zeros(number_count, dtype=float),
            np.zeros(number_count, dtype=float),
        ])
    sequence = np.argmax(matrix, axis=1).astype(int)
    previous = int(sequence[-1])
    transitions = np.zeros((number_count, number_count), dtype=float)
    steps = np.zeros(number_count, dtype=float)
    parity_transitions = np.zeros((2, 2), dtype=float)
    zone_transitions = np.zeros((4, 4), dtype=float)
    for left, right in zip(sequence[:-1], sequence[1:]):
        transitions[int(left), int(right)] += 1.0
        steps[abs(int(right) - int(left))] += 1.0
        parity_transitions[int(left) % 2, int(right) % 2] += 1.0
        zone_transitions[int(left) // 4, int(right) // 4] += 1.0
    candidates = np.arange(number_count, dtype=int)
    absolute_steps = np.abs(candidates - previous)
    return np.column_stack([
        (transitions[previous] + alpha) / (transitions[previous].sum() + alpha * number_count),
        (steps[absolute_steps] + alpha) / (steps.sum() + alpha * number_count),
        (parity_transitions[previous % 2, candidates % 2] + alpha) / (
            parity_transitions[previous % 2].sum() + alpha * 2.0
        ),
        (zone_transitions[previous // 4, candidates // 4] + alpha) / (
            zone_transitions[previous // 4].sum() + alpha * 4.0
        ),
        absolute_steps.astype(float) / 15.0,
        (candidates == previous).astype(float),
        (absolute_steps == 1).astype(float),
    ])


def _blue_high_order_state_at_end(matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    periods, number_count = matrix.shape
    if number_count != 16:
        raise ValueError("蓝球高阶序列特征要求 16 个候选号码")
    sequence = list(np.argmax(matrix, axis=1).astype(int)) if periods else []
    order0 = np.zeros(number_count, dtype=float)
    order1: dict[tuple[int], np.ndarray] = {}
    order2: dict[tuple[int, int], np.ndarray] = {}
    order3: dict[tuple[int, int, int], np.ndarray] = {}
    for index, current in enumerate(sequence):
        current = int(current)
        order0[current] += 1.0
        if index >= 1:
            order1.setdefault(tuple(sequence[index - 1:index]), np.zeros(number_count, dtype=float))[current] += 1.0
        if index >= 2:
            order2.setdefault(tuple(sequence[index - 2:index]), np.zeros(number_count, dtype=float))[current] += 1.0
        if index >= 3:
            order3.setdefault(tuple(sequence[index - 3:index]), np.zeros(number_count, dtype=float))[current] += 1.0

    def probability_for_context(context: tuple[int, ...]) -> tuple[np.ndarray, float]:
        if len(context) == 0:
            counts = order0
        elif len(context) == 1:
            counts = order1.get(context, np.zeros(number_count, dtype=float))
        elif len(context) == 2:
            counts = order2.get(context, np.zeros(number_count, dtype=float))
        else:
            counts = order3.get(context, np.zeros(number_count, dtype=float))
        total = float(counts.sum())
        probability = (counts + alpha) / (total + alpha * number_count)
        reliability = total / (total + 8.0 + 4.0 * len(context))
        return probability, float(reliability)

    second_order = (
        probability_for_context(tuple(sequence[-2:]))[0]
        if len(sequence) >= 2
        else np.full(number_count, 1.0 / number_count)
    )
    contexts = [tuple()]
    if len(sequence) >= 1:
        contexts.append(tuple(sequence[-1:]))
    if len(sequence) >= 2:
        contexts.append(tuple(sequence[-2:]))
    if len(sequence) >= 3:
        contexts.append(tuple(sequence[-3:]))
    probabilities: list[np.ndarray] = []
    weights: list[float] = []
    for context in contexts:
        probability, reliability = probability_for_context(context)
        probabilities.append(probability)
        weights.append(max(0.08, reliability))
    weights_array = np.asarray(weights, dtype=float)
    weights_array /= max(float(weights_array.sum()), 1e-12)
    variable_markov = np.sum(np.stack(probabilities, axis=0) * weights_array[:, None], axis=0)

    gap_std_pressure = np.zeros(number_count, dtype=float)
    gap_skew_pressure = np.zeros(number_count, dtype=float)
    for number in range(number_count):
        hits = np.flatnonzero(matrix[:, number])
        omission = float(periods - 1 - hits[-1]) if len(hits) else float(periods)
        if len(hits) < 2:
            mean_gap = 16.0
            std_gap = 8.0
            skew = 0.0
        else:
            gaps = np.diff(hits) - 1
            mean_gap = float(gaps.mean())
            std_gap = max(float(gaps.std(ddof=0)), 1.0)
            centered = gaps - mean_gap
            skew = float(np.mean(centered ** 3) / max(std_gap ** 3, 1e-9))
        z_score = (omission - mean_gap) / max(std_gap, 1.0)
        gap_std_pressure[number] = float(_sigmoid(z_score))
        gap_skew_pressure[number] = float(_sigmoid(z_score + 0.18 * skew))
    repeat_run_pressure = np.zeros(number_count, dtype=float)
    if sequence:
        last = int(sequence[-1])
        run_length = 1
        for previous in reversed(sequence[:-1]):
            if int(previous) != last:
                break
            run_length += 1
        repeat_run_pressure[last] = min(1.0, run_length / 3.0)
    return np.column_stack([
        second_order,
        variable_markov,
        gap_std_pressure,
        gap_skew_pressure,
        repeat_run_pressure,
    ])


def _calendar_number_state_at_end(
    frame: pd.DataFrame | None,
    matrix: np.ndarray,
    *,
    color: str,
) -> np.ndarray:
    periods, number_count = matrix.shape
    if frame is None or "date" not in frame.columns:
        return np.zeros((number_count, len(CALENDAR_CONTEXT_COLUMNS)), dtype=float)
    if "date_provenance" in frame.columns:
        provenance = frame["date_provenance"].astype(str).str.lower()
        if not provenance.eq("source").any():
            return np.zeros((number_count, len(CALENDAR_CONTEXT_COLUMNS)), dtype=float)
    dates = pd.to_datetime(frame["date"], errors="coerce").reset_index(drop=True)
    if dates.isna().any() or periods <= 0:
        return np.zeros((number_count, len(CALENDAR_CONTEXT_COLUMNS)), dtype=float)
    target_date = _next_draw_date(pd.Timestamp(dates.iloc[-1]))
    weekday = int(target_date.weekday())
    month = int(target_date.month) - 1
    quarter = int((int(target_date.month) - 1) // 3)
    holiday = int(_fixed_holiday_near(target_date))
    weekday_counts = np.zeros((7, number_count), dtype=float)
    weekday_total = np.zeros(7, dtype=float)
    month_counts = np.zeros((12, number_count), dtype=float)
    month_total = np.zeros(12, dtype=float)
    quarter_counts = np.zeros((4, number_count), dtype=float)
    quarter_total = np.zeros(4, dtype=float)
    holiday_counts = np.zeros((2, number_count), dtype=float)
    holiday_total = np.zeros(2, dtype=float)
    for index in range(periods):
        row = matrix[index].astype(float)
        date_value = pd.Timestamp(dates.iloc[index])
        weekday_index = int(date_value.weekday())
        weekday_counts[weekday_index] += row
        weekday_total[weekday_index] += float(row.sum())
        month_index = int(date_value.month) - 1
        month_counts[month_index] += row
        month_total[month_index] += float(row.sum())
        quarter_index = int((int(date_value.month) - 1) // 3)
        quarter_counts[quarter_index] += row
        quarter_total[quarter_index] += float(row.sum())
        holiday_index = int(_fixed_holiday_near(date_value))
        holiday_counts[holiday_index] += row
        holiday_total[holiday_index] += float(row.sum())
    prefix = np.vstack([np.zeros((1, number_count)), np.cumsum(matrix, axis=0)])
    recent_start = max(0, periods - 300)
    older_start = max(0, periods - 900)
    recent = (prefix[periods] - prefix[recent_start]) / max(periods - recent_start, 1)
    older = (prefix[recent_start] - prefix[older_start]) / max(recent_start - older_start, 1)
    return np.column_stack([
        (weekday_counts[weekday] + 1.0) / (weekday_total[weekday] + number_count),
        (month_counts[month] + 1.0) / (month_total[month] + number_count),
        (quarter_counts[quarter] + 1.0) / (quarter_total[quarter] + number_count),
        (holiday_counts[holiday] + 1.0) / (holiday_total[holiday] + number_count),
        recent - older,
    ])


def _red_blue_cross_state_at_end(frame: pd.DataFrame | None, matrix: np.ndarray, color: str) -> np.ndarray:
    periods, number_count = matrix.shape
    if frame is None or "blue" not in frame.columns or periods <= 0:
        columns = len(RED_BLUE_CROSS_COLUMNS) if color == "red" else len(BLUE_RED_CROSS_COLUMNS)
        return np.zeros((number_count, columns), dtype=float)
    blue_values = frame["blue"].astype(int).to_numpy()
    red_columns = [f"r{index}" for index in range(1, 7)]
    has_red = all(column in frame.columns for column in red_columns)
    reds = frame[red_columns].astype(int).to_numpy() if has_red else None
    if color == "red":
        start = max(0, periods - 600)
        recent_blue = int(blue_values[periods - 1])
        prior_blues = blue_values[start:periods]
        prior_red = matrix[start:periods]
        global_rate = (prior_red.sum(axis=0) + 1.0) / (max(periods - start, 1) + 33.0)
        same_blue = prior_blues == recent_blue
        conditional = (
            (prior_red[same_blue].sum(axis=0) + 1.0) / (float(np.sum(same_blue)) + 33.0)
            if np.any(same_blue)
            else global_rate
        )
        shifted = global_rate
        if periods >= 2:
            offset = max(start, 1)
            previous_blue = blue_values[offset:periods]
            current_red = matrix[offset:periods]
            transition_mask = np.abs(previous_blue - recent_blue) <= 1
            shifted = (
                (current_red[transition_mask].sum(axis=0) + 1.0)
                / (float(np.sum(transition_mask)) + 33.0)
                if np.any(transition_mask)
                else global_rate
            )
        return np.column_stack([
            0.45 * conditional + 0.55 * global_rate,
            0.55 * shifted + 0.45 * global_rate,
        ])

    states = np.zeros((number_count, len(BLUE_RED_CROSS_COLUMNS)), dtype=float)
    if reds is None:
        return states
    red_sum = reds.sum(axis=1)
    red_span = reds.max(axis=1) - reds.min(axis=1)
    red_odd = (reds % 2).sum(axis=1)
    red_zone_hot = np.stack([
        (reds <= 11).sum(axis=1),
        ((reds >= 12) & (reds <= 22)).sum(axis=1),
        (reds >= 23).sum(axis=1),
    ], axis=1).argmax(axis=1)
    start = max(0, periods - 700)
    latest_index = periods - 1
    values = blue_values[start:periods]
    global_counts = np.bincount(values - 1, minlength=number_count).astype(float) + 1.0
    global_probability = global_counts / max(float(global_counts.sum()), 1.0)
    masks = [
        np.abs(red_sum[start:periods] - red_sum[latest_index]) <= 12,
        np.abs(red_span[start:periods] - red_span[latest_index]) <= 4,
        np.abs(red_odd[start:periods] - red_odd[latest_index]) <= 1,
        red_zone_hot[start:periods] == red_zone_hot[latest_index],
    ]
    conditional_probabilities = []
    for mask in masks:
        matched = values[mask]
        if len(matched) < 16:
            probability = global_probability
        else:
            counts = np.bincount(matched - 1, minlength=number_count).astype(float) + 4.0 * global_probability
            probability = counts / max(float(counts.sum()), 1.0)
        conditional_probabilities.append(probability)
    for column, probability in enumerate(conditional_probabilities):
        states[:, column] = probability
    states[:, 4] = np.mean(np.stack(conditional_probabilities, axis=0), axis=0)
    return states


def _state_for_color_at_end(
    frame: pd.DataFrame | None,
    matrix: np.ndarray,
    windows: list[int],
    decay: float,
    color: str,
    include_blue_sequence: bool = True,
    include_cross_feature: bool = False,
    include_calendar_feature: bool = True,
) -> np.ndarray:
    blocks = [_base_state_at_end(matrix, windows, decay)]
    if color == "red":
        blocks.append(_red_pairwise_state_at_end(matrix, windows))
        blocks.append(_red_structure_state_at_end(matrix))
        blocks.append(_red_dynamic_state_at_end(matrix, window=max(windows)))
        if include_calendar_feature:
            blocks.append(_calendar_number_state_at_end(frame, matrix, color=color))
        if include_cross_feature:
            blocks.append(_red_blue_cross_state_at_end(frame, matrix, color))
        return np.concatenate(blocks, axis=1)
    if color == "blue" and include_blue_sequence:
        blocks.append(_blue_sequence_state_at_end(matrix))
        blocks.append(_blue_high_order_state_at_end(matrix))
        if include_calendar_feature:
            blocks.append(_calendar_number_state_at_end(frame, matrix, color=color))
        if include_cross_feature:
            blocks.append(_red_blue_cross_state_at_end(frame, matrix, color))
        return np.concatenate(blocks, axis=1)
    return np.concatenate(blocks, axis=1)


def build_training_samples(
    frame: pd.DataFrame,
    *,
    color: str,
    windows: Iterable[int] = (5, 10, 30),
    decay_factor: float = 0.98,
    min_history: int = 60,
    lookback: int | None = None,
    include_blue_sequence: bool = True,
    include_cross_feature: bool = False,
    include_calendar_feature: bool = True,
    cache_enabled: bool = True,
    cache_max_entries: int = 24,
) -> tuple[pd.DataFrame, np.ndarray]:
    assert_clean_sequence(frame)
    if lookback:
        retained = max(int(min_history) + int(lookback) + 1, max(int(value) for value in windows) + 2)
        if len(frame) > retained:
            frame = frame.iloc[-retained:].copy().reset_index(drop=True)
            frame["idx"] = np.arange(len(frame), dtype=int)
    window_values = [int(value) for value in windows]
    cache_key = (
        FEATURE_CACHE_VERSION,
        "training",
        str(color),
        tuple(window_values),
        round(float(decay_factor), 12),
        int(min_history),
        None if lookback is None else int(lookback),
        bool(include_blue_sequence),
        bool(include_cross_feature),
        bool(include_calendar_feature),
        _frame_signature(frame),
    )
    cached = _cache_get(cache_key, bool(cache_enabled))
    if cached is not None:
        features, labels = cached
        if labels is None:
            raise RuntimeError("训练特征缓存缺少标签")
        return features, labels
    matrix = occurrence_matrix(frame, color)
    states = _states_for_color(
        frame,
        matrix,
        window_values,
        float(decay_factor),
        color,
        include_blue_sequence,
        include_cross_feature,
        include_calendar_feature,
    )
    start = max(int(min_history), 1)
    if lookback:
        start = max(start, len(frame) - int(lookback))
    if start >= len(frame):
        raise ValueError("历史数据不足以构造监督学习样本")
    selected = states[start:len(frame)]
    labels = matrix[start:len(frame)]
    # include_blue_sequence 只控制蓝球序列特征。红球侧始终会构造共现、
    # 结构、动态和日历特征，因此列名也必须始终按 red 生成；否则矩阵
    # reshape 会把特征维拆成更多行，造成模型训练时 X/y 样本数不一致。
    feature_color = str(color) if (str(color) == "red" or include_blue_sequence) else None
    columns = feature_columns(
        window_values,
        feature_color,
        include_cross_feature=bool(include_cross_feature),
        include_calendar_feature=bool(include_calendar_feature),
    )
    features = pd.DataFrame(selected.reshape(-1, len(columns)), columns=columns)
    flattened_labels = labels.reshape(-1)
    _cache_put(
        cache_key,
        features,
        flattened_labels,
        enabled=bool(cache_enabled),
        max_entries=int(cache_max_entries),
    )
    return features, flattened_labels


def build_prediction_features(
    history: pd.DataFrame,
    *,
    color: str,
    target_idx: int,
    windows: Iterable[int] = (5, 10, 30),
    decay_factor: float = 0.98,
    include_blue_sequence: bool = True,
    include_cross_feature: bool = False,
    include_calendar_feature: bool = True,
    lookback: int | None = None,
    cache_enabled: bool = True,
    cache_max_entries: int = 24,
) -> pd.DataFrame:
    assert_left_truncated(history, target_idx)
    original_history_max_idx = int(history["idx"].max())
    if lookback:
        retained = max(int(lookback), max(int(value) for value in windows) + 2, 760)
        if len(history) > retained:
            history = history.iloc[-retained:].copy().reset_index(drop=True)
            history["idx"] = np.arange(len(history), dtype=int)
    window_values = [int(value) for value in windows]
    cache_key = (
        FEATURE_CACHE_VERSION,
        "prediction",
        str(color),
        tuple(window_values),
        round(float(decay_factor), 12),
        int(target_idx),
        None if lookback is None else int(lookback),
        bool(include_blue_sequence),
        bool(include_cross_feature),
        bool(include_calendar_feature),
        _frame_signature(history),
    )
    cached = _cache_get(cache_key, bool(cache_enabled))
    if cached is not None:
        features, _ = cached
        features.attrs["history_max_idx"] = original_history_max_idx
        features.attrs["target_idx"] = int(target_idx)
        return features
    matrix = occurrence_matrix(history, color)
    state = _state_for_color_at_end(
        history,
        matrix,
        window_values,
        float(decay_factor),
        color,
        include_blue_sequence,
        include_cross_feature,
        include_calendar_feature,
    )
    result = pd.DataFrame(
        state,
        columns=feature_columns(
            window_values,
            str(color) if (str(color) == "red" or include_blue_sequence) else None,
            include_cross_feature=bool(include_cross_feature),
            include_calendar_feature=bool(include_calendar_feature),
        ),
    )
    result.attrs["history_max_idx"] = original_history_max_idx
    result.attrs["target_idx"] = int(target_idx)
    _cache_put(
        cache_key,
        result,
        None,
        enabled=bool(cache_enabled),
        max_entries=int(cache_max_entries),
    )
    return result


def sequence_feature_columns(color: str, windows: Iterable[int]) -> list[str]:
    number_count = 33 if color == "red" else 16
    return [
        f"n{number:02d}_{feature}"
        for number in range(1, number_count + 1)
        for feature in feature_columns(windows, color)
    ]


def build_sequence_training_samples(
    frame: pd.DataFrame,
    *,
    color: str,
    windows: Iterable[int] = (5, 10, 30),
    decay_factor: float = 0.98,
    min_history: int = 60,
    lookback: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build one whole-draw sample per period for constrained models.

    The flattened state preserves the identity of every number.  Blue labels
    are a single mutually exclusive class; red labels remain a six-hot set for
    ranking models.
    """
    assert_clean_sequence(frame)
    if color not in {"red", "blue"}:
        raise ValueError("color 必须为 red 或 blue")
    window_values = [int(value) for value in windows]
    matrix = occurrence_matrix(frame, color)
    states = _states_for_color(frame, matrix, window_values, float(decay_factor), color)
    start = max(int(min_history), 1)
    if lookback:
        start = max(start, len(frame) - int(lookback))
    if start >= len(frame):
        raise ValueError("历史数据不足以构造序列监督样本")
    selected = states[start:len(frame)]
    features = pd.DataFrame(
        selected.reshape(len(selected), -1),
        columns=sequence_feature_columns(color, window_values),
    )
    labels = matrix[start:len(frame)]
    if color == "blue":
        return features, np.argmax(labels, axis=1).astype(int)
    return features, labels.astype(int)


def build_prediction_sequence_features(
    history: pd.DataFrame,
    *,
    color: str,
    target_idx: int,
    windows: Iterable[int] = (5, 10, 30),
    decay_factor: float = 0.98,
) -> pd.DataFrame:
    assert_left_truncated(history, target_idx)
    if color not in {"red", "blue"}:
        raise ValueError("color 必须为 red 或 blue")
    window_values = [int(value) for value in windows]
    matrix = occurrence_matrix(history, color)
    state = _states_for_color(history, matrix, window_values, float(decay_factor), color)[-1]
    result = pd.DataFrame(
        state.reshape(1, -1),
        columns=sequence_feature_columns(color, window_values),
    )
    result.attrs["history_max_idx"] = int(history["idx"].max())
    result.attrs["target_idx"] = int(target_idx)
    return result
