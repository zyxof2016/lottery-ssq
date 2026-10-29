from __future__ import annotations

import hashlib
import copy
import json
from collections import OrderedDict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.combination_gen.score_func import base_quality, number_score, structure_score
from src.feature_engineer.base_stats import RED_COLUMNS, structure_profile


_CANDIDATE_BANK_CACHE_VERSION = 1
_CANDIDATE_BANK_CACHE_MAX_ENTRIES = 16
_CANDIDATE_BANK_CACHE: OrderedDict[str, tuple[list[dict], dict[str, int]]] = OrderedDict()


def _validated_probability(values: np.ndarray, *, name: str, expected_size: int) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.shape != (int(expected_size),):
        raise ValueError(f"{name} 概率长度必须为 {expected_size}")
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} 概率包含 NaN 或无穷值")
    if np.any(probability < 0):
        raise ValueError(f"{name} 概率不能为负数")
    total = float(probability.sum())
    if total <= 0:
        raise ValueError(f"{name} 概率和必须大于0")
    return probability / total


def _sample_ticket(
    rng: np.random.Generator,
    red_probability: np.ndarray,
    blue_probability: np.ndarray,
) -> tuple[tuple[int, ...], int]:
    red = tuple(sorted((rng.choice(33, size=6, replace=False, p=red_probability) + 1).tolist()))
    blue = int(rng.choice(16, p=blue_probability) + 1)
    return red, blue


def _tempered_probability(probability: np.ndarray, power: float) -> np.ndarray:
    values = np.power(np.clip(np.asarray(probability, dtype=float), 1e-12, None), float(power))
    values /= max(float(values.sum()), 1e-12)
    return values


def _candidate_bank_cache_key(
    probabilities: dict[str, np.ndarray],
    history: pd.DataFrame,
    *,
    samples: int,
    pool_size: int,
    scoring_config: dict[str, Any],
    random_seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"candidate_bank_v{_CANDIDATE_BANK_CACHE_VERSION}\0".encode("utf-8"))
    digest.update(np.asarray(probabilities["red"], dtype=float).round(12).tobytes())
    digest.update(np.asarray(probabilities["blue"], dtype=float).round(12).tobytes())
    digest.update(str(int(samples)).encode("ascii") + b"\0")
    digest.update(str(int(pool_size)).encode("ascii") + b"\0")
    digest.update(str(int(random_seed)).encode("ascii") + b"\0")
    digest.update(
        json.dumps(scoring_config, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        + b"\0"
    )
    digest.update(str(len(history)).encode("ascii") + b"\0")
    if "period" in history:
        digest.update(str(history.iloc[0]["period"]).encode("utf-8") + b"\0")
        digest.update(str(history.iloc[-1]["period"]).encode("utf-8") + b"\0")
    digest.update(str(list(history.columns)).encode("utf-8") + b"\0")
    digest.update(pd.util.hash_pandas_object(history, index=False).values.tobytes())
    return digest.hexdigest()


def candidate_bank_cache_info() -> dict[str, Any]:
    return {
        "version": _CANDIDATE_BANK_CACHE_VERSION,
        "entries": len(_CANDIDATE_BANK_CACHE),
        "max_entries": _CANDIDATE_BANK_CACHE_MAX_ENTRIES,
        "scope": "process_memory_only",
    }


def clear_candidate_bank_cache() -> None:
    _CANDIDATE_BANK_CACHE.clear()


def generate_candidates(
    probabilities: dict[str, np.ndarray],
    history: pd.DataFrame,
    *,
    samples: int,
    pool_size: int,
    scoring_config: dict[str, Any],
    random_seed: int,
) -> tuple[list[dict], dict[str, int]]:
    if history is None or history.empty:
        raise ValueError("history 不能为空")
    if int(samples) <= 0:
        raise ValueError("samples 必须大于0")
    if int(pool_size) <= 0:
        raise ValueError("pool_size 必须大于0")
    rng = np.random.default_rng(int(random_seed))
    red_probability = _validated_probability(probabilities["red"], name="red", expected_size=33)
    blue_probability = _validated_probability(probabilities["blue"], name="blue", expected_size=16)
    cache_key = _candidate_bank_cache_key(
        {"red": red_probability, "blue": blue_probability},
        history,
        samples=int(samples),
        pool_size=int(pool_size),
        scoring_config=scoring_config,
        random_seed=int(random_seed),
    )
    cached = _CANDIDATE_BANK_CACHE.get(cache_key)
    if cached is not None:
        _CANDIDATE_BANK_CACHE.move_to_end(cache_key)
        retained, sampling = cached
        return copy.deepcopy(retained), {
            **copy.deepcopy(sampling),
            "candidate_bank_cache_hit": True,
            "candidate_bank_cache_key": cache_key[:16],
        }
    red_variants = [
        ("model", red_probability),
        ("importance_focus", _tempered_probability(red_probability, 1.35)),
        ("importance_broad", _tempered_probability(red_probability, 0.72)),
    ]
    blue_variants = [
        ("model", blue_probability),
        ("importance_focus", _tempered_probability(blue_probability, 1.25)),
        ("importance_broad", _tempered_probability(blue_probability, 0.80)),
    ]
    variant_counts = {name: 0 for name, _ in red_variants}
    profile = structure_profile(history)
    unique: dict[tuple[tuple[int, ...], int], dict] = {}
    duplicate_draws = 0
    for sample_index in range(int(samples)):
        variant_index = int(sample_index % len(red_variants))
        variant_name, red_sampling = red_variants[variant_index]
        _, blue_sampling = blue_variants[variant_index]
        variant_counts[variant_name] += 1
        red, blue = _sample_ticket(rng, red_sampling, blue_sampling)
        key = (red, blue)
        if key in unique:
            duplicate_draws += 1
            continue
        probability_value = number_score(red, blue, probabilities)
        red_probability_value = float(np.exp(np.mean(np.log(np.clip(red_probability[np.array(red) - 1] * 33.0, 1e-12, None)))))
        blue_probability_value = float(np.clip(blue_probability[blue - 1] * 16.0, 1e-12, None))
        structure_value = structure_score(red, profile)
        unique[key] = {
            "red": red,
            "blue": blue,
            "number_score": probability_value,
            "red_number_score": red_probability_value,
            "blue_number_score": blue_probability_value,
            "structure_score": structure_value,
            "quality_score": base_quality(
                red_probability_value,
                blue_probability_value,
                structure_value,
                scoring_config,
            ),
        }
    candidates = sorted(unique.values(), key=lambda item: item["quality_score"], reverse=True)
    retained = candidates[: min(int(pool_size), len(candidates))]
    sampling = {
        "requested_samples": int(samples),
        "actual_samples": int(samples),
        "unique_candidates": len(unique),
        "duplicate_draws": duplicate_draws,
        "retained_candidates": len(retained),
        "mode": "importance_tempered_monte_carlo",
        "sampling_variants": variant_counts,
        "candidate_bank_cache_hit": False,
        "candidate_bank_cache_key": cache_key[:16],
    }
    _CANDIDATE_BANK_CACHE[cache_key] = (copy.deepcopy(retained), copy.deepcopy(sampling))
    while len(_CANDIDATE_BANK_CACHE) > _CANDIDATE_BANK_CACHE_MAX_ENTRIES:
        _CANDIDATE_BANK_CACHE.popitem(last=False)
    return retained, sampling


def generate_core_candidates(
    probabilities: dict[str, np.ndarray],
    ranking: dict[str, Any],
    history: pd.DataFrame,
    *,
    pool_size: int,
    scoring_config: dict[str, Any],
    model_rankings: dict[str, dict[str, Any]] | None = None,
    red_core_size: int = 9,
    red_anchor_size: int = 6,
    max_red_changes: int = 2,
    min_anchor_kept: int | None = None,
    blue_core_size: int = 3,
) -> tuple[list[dict], dict[str, Any]]:
    if history is None or history.empty:
        raise ValueError("history 不能为空")
    if int(pool_size) <= 0:
        raise ValueError("pool_size 必须大于0")
    red_probability = _validated_probability(probabilities["red"], name="red", expected_size=33)
    blue_probability = _validated_probability(probabilities["blue"], name="blue", expected_size=16)
    profile = structure_profile(history)

    desired_red = max(int(red_core_size), int(red_anchor_size))
    desired_blue = max(1, int(blue_core_size))
    ranked_red = _build_weak_signal_ranked_numbers(
        ranking,
        model_rankings,
        desired_count=max(int(red_core_size), int(red_anchor_size), 18),
    )
    model_diversity_red = _build_model_diversity_red_numbers(
        model_rankings,
        desired_count=max(int(red_core_size), int(red_anchor_size), 18),
    )
    ranked_blue, blue_signal_bonus = _build_weak_signal_ranked_blue_numbers(
        ranking,
        model_rankings,
        desired_count=max(int(blue_core_size), 8),
    )
    historical_red = _build_history_context_red_numbers(
        history,
        desired_count=max(int(red_core_size), int(red_anchor_size), 18),
    )
    fallback_red = (np.argsort(-red_probability, kind="stable")[:desired_red] + 1).tolist()
    fallback_blue = (np.argsort(-blue_probability, kind="stable")[:desired_blue] + 1).tolist()
    top_red = _merge_ranked_candidates(
        ranked_red,
        historical_red,
        model_diversity_red,
        fallback_red,
        desired_count=desired_red,
    )
    historical_support = _rank_support_map(historical_red)
    model_diversity_support = _rank_support_map(model_diversity_red)
    top_blue = _merge_ranked_candidates(ranked_blue, fallback_blue, desired_count=desired_blue)
    core_red = top_red[: max(int(red_core_size), 6, int(red_anchor_size))]
    anchor_red = tuple(sorted(core_red[: int(red_anchor_size)]))
    candidate_red_sets: set[tuple[int, ...]] = set()
    anchor_floor = (
        int(min_anchor_kept)
        if min_anchor_kept is not None
        else max(0, min(6, len(anchor_red)) - int(max_red_changes) - 1)
    )
    anchor_floor = max(0, min(6, anchor_floor))
    for red in combinations(core_red, 6):
        red_tuple = tuple(sorted(int(number) for number in red))
        if len(anchor_red) >= 1 and len(set(red_tuple) & set(anchor_red)) < anchor_floor:
            continue
        candidate_red_sets.add(red_tuple)

    blue_candidates = top_blue[: max(1, int(blue_core_size))]
    generated_red_set_count = len(candidate_red_sets)
    candidate_red_sets = set(_retain_importance_stratified_red_sets(
        candidate_red_sets,
        red_probability,
        profile,
        pool_size=int(pool_size),
        blue_count=max(1, len(blue_candidates)),
    ))
    unique: dict[tuple[tuple[int, ...], int], dict] = {}
    for red in sorted(candidate_red_sets):
        for blue in blue_candidates:
            key = (red, int(blue))
            probability_value = number_score(red, blue, probabilities)
            red_probability_value = float(np.exp(np.mean(np.log(np.clip(red_probability[np.array(red) - 1] * 33.0, 1e-12, None)))))
            blue_probability_value = float(np.clip(blue_probability[int(blue) - 1] * 16.0, 1e-12, None))
            structure_value = structure_score(red, profile)
            historical_context_value = float(np.mean([historical_support.get(int(number), 0.0) for number in red]))
            model_diversity_value = float(np.mean([model_diversity_support.get(int(number), 0.0) for number in red]))
            raw_quality = base_quality(
                red_probability_value,
                blue_probability_value,
                structure_value,
                scoring_config,
            )
            quality_value = (
                raw_quality
                + float(scoring_config.get("history_context_weight", 0.0)) * historical_context_value
                + float(scoring_config.get("model_diversity_context_weight", 0.0)) * model_diversity_value
            )
            unique[key] = {
                "red": red,
                "blue": int(blue),
                "number_score": probability_value,
                "red_number_score": red_probability_value,
                "blue_number_score": blue_probability_value,
                "blue_signal_bonus": float(blue_signal_bonus.get(int(blue), 0.0)),
                "history_context_score": historical_context_value,
                "model_diversity_context_score": model_diversity_value,
                "structure_score": structure_value,
                "quality_score": quality_value,
                "raw_quality_score": raw_quality,
            }
    candidates = sorted(unique.values(), key=lambda item: item["quality_score"], reverse=True)
    retained, retention = _retain_quality_and_diversity_candidates(
        candidates,
        pool_size=int(pool_size),
    )
    return retained, {
        "mode": "core_shrunk_candidates",
        "red_anchor": list(anchor_red),
        "red_core": core_red,
        "historical_red_core": historical_red[:desired_red],
        "model_diversity_red_core": model_diversity_red[:desired_red],
        "blue_core": blue_candidates,
        "max_red_changes": int(max_red_changes),
        "min_anchor_kept": int(anchor_floor),
        "red_sets_generated": int(generated_red_set_count),
        "red_sets_retained": int(len(candidate_red_sets)),
        "red_set_retention_mode": "importance_quality_head_plus_structure_strata",
        "retained_candidates": len(retained),
        "unique_candidates": len(unique),
        "retention": retention,
    }


def _retain_importance_stratified_red_sets(
    red_sets: set[tuple[int, ...]],
    red_probability: np.ndarray,
    profile: dict[str, tuple[float, float]],
    *,
    pool_size: int,
    blue_count: int,
) -> list[tuple[int, ...]]:
    if not red_sets:
        return []
    target = max(
        120,
        min(
            len(red_sets),
            int(max(pool_size, 1) * 3 / max(int(blue_count), 1)),
        ),
    )
    if len(red_sets) <= target:
        return sorted(red_sets)

    scored: list[tuple[float, tuple[int, ...], tuple[int, int, int]]] = []
    for red in red_sets:
        red_probability_value = float(
            np.exp(np.mean(np.log(np.clip(red_probability[np.array(red) - 1] * 33.0, 1e-12, None))))
        )
        structure_value = structure_score(red, profile)
        total = int(sum(red))
        span = int(max(red) - min(red))
        odd = int(sum(number % 2 for number in red))
        strata = (total // 15, span // 4, odd)
        score = 0.70 * red_probability_value + 0.30 * structure_value
        scored.append((float(score), red, strata))
    scored.sort(key=lambda item: (item[0], -sum(item[1])), reverse=True)

    head_size = min(target, max(60, int(target * 0.55)))
    retained = [red for _, red, _ in scored[:head_size]]
    retained_set = set(retained)
    buckets: dict[tuple[int, int, int], list[tuple[float, tuple[int, ...]]]] = {}
    for score, red, strata in scored[head_size:]:
        buckets.setdefault(strata, []).append((score, red))
    bucket_keys = sorted(
        buckets,
        key=lambda key: len(buckets[key]),
        reverse=True,
    )
    cursor = 0
    while len(retained) < target and bucket_keys:
        key = bucket_keys[cursor % len(bucket_keys)]
        bucket = buckets[key]
        while bucket:
            _, red = bucket.pop(0)
            if red not in retained_set:
                retained.append(red)
                retained_set.add(red)
                break
        bucket_keys = [item for item in bucket_keys if buckets.get(item)]
        cursor += 1
    if len(retained) < target:
        for _, red, _ in scored[head_size:]:
            if red in retained_set:
                continue
            retained.append(red)
            retained_set.add(red)
            if len(retained) >= target:
                break
    return retained[:target]


def _retain_quality_and_diversity_candidates(
    candidates: list[dict],
    *,
    pool_size: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Keep both high-score rows and low-overlap rows in the retained pool."""
    if not candidates or int(pool_size) <= 0:
        return [], {"mode": "empty"}
    limit = min(int(pool_size), len(candidates))
    if len(candidates) <= limit:
        return list(candidates), {
            "mode": "all_candidates",
            "quality_head": len(candidates),
            "diversity_reserve": 0,
            "fill": 0,
        }

    quality_head_size = min(limit, max(120, int(limit * 0.40)))
    diversity_target = max(0, limit - quality_head_size)
    retained = list(candidates[:quality_head_size])
    used_keys = {
        (tuple(int(number) for number in candidate["red"]), int(candidate["blue"]))
        for candidate in retained
    }
    diversity: list[dict] = []
    diversity_red_sets: list[set[int]] = []

    tail = candidates[quality_head_size:]
    max_scan = min(len(tail), max(20_000, diversity_target * 80))
    if max_scan >= len(tail):
        sampled_source = list(tail)
    else:
        # Evenly sample the whole quality tail instead of sorting hundreds of
        # thousands of rows.  The hash offset makes the sample stable while
        # avoiding a permanent bias toward the first row of each stride.
        offset_payload = f"{len(candidates)}|{pool_size}|{diversity_target}".encode("ascii")
        offset = int.from_bytes(hashlib.sha256(offset_payload).digest()[:8], "big")
        step = len(tail) / float(max_scan)
        seen_indices: set[int] = set()
        sampled_source = []
        for index in range(max_scan):
            sampled = int((offset + index * step) % len(tail))
            if sampled in seen_indices:
                continue
            seen_indices.add(sampled)
            sampled_source.append(tail[sampled])

    def diversity_order(candidate: dict) -> int:
        red = ",".join(str(int(number)) for number in candidate["red"])
        payload = f"{red}|{int(candidate['blue'])}".encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    diversity_source = sorted(sampled_source, key=diversity_order)
    for max_overlap in (1, 2, 3, 4, 5, 6):
        if len(diversity) >= diversity_target:
            break
        for candidate in diversity_source:
            key = (tuple(int(number) for number in candidate["red"]), int(candidate["blue"]))
            if key in used_keys:
                continue
            red_set = {int(number) for number in candidate["red"]}
            if all(len(red_set & prior) <= max_overlap for prior in diversity_red_sets):
                diversity.append(candidate)
                diversity_red_sets.append(red_set)
                used_keys.add(key)
                if len(diversity) >= diversity_target:
                    break

    retained.extend(diversity)
    fill_count = 0
    if len(retained) < limit:
        for candidate in candidates[quality_head_size:]:
            key = (tuple(int(number) for number in candidate["red"]), int(candidate["blue"]))
            if key in used_keys:
                continue
            retained.append(candidate)
            used_keys.add(key)
            fill_count += 1
            if len(retained) >= limit:
                break

    return retained[:limit], {
        "mode": "quality_head_plus_diversity_reserve",
        "quality_head": quality_head_size,
        "diversity_reserve": len(diversity),
        "fill": fill_count,
    }


def _build_weak_signal_ranked_numbers(
    fusion_ranking: dict[str, Any],
    model_rankings: dict[str, dict[str, Any]] | None,
    *,
    desired_count: int,
    top_k_per_model: int = 12,
    model_weight_floor: float = 0.04,
    dormant_model_weight: float = 0.18,
    fusion_bonus_weight: float = 0.80,
    agreement_bonus_weight: float = 0.30,
    dormant_top_rank_bonus_weight: float = 0.14,
) -> list[int]:
    desired = max(6, int(desired_count))
    scores = {number: 0.0 for number in range(1, 34)}
    fusion_top = [int(number) for number in fusion_ranking.get("top_red", [])]
    fusion_scores = [float(value) for value in fusion_ranking.get("top_red_scores", [])]
    fusion_max = max(fusion_scores, default=0.0)
    if fusion_top:
        for rank, number in enumerate(fusion_top[:desired]):
            base = (
                fusion_scores[rank] / fusion_max
                if rank < len(fusion_scores) and fusion_max > 0.0
                else 1.0 - rank / max(len(fusion_top) - 1, 1)
            )
            scores[int(number)] += float(fusion_bonus_weight) * float(base)

    model_support: dict[int, set[str]] = {number: set() for number in range(1, 34)}
    for model_name, summary in (model_rankings or {}).items():
        red_weight = float(summary.get("red_weight", summary.get("weight", 0.0)))
        top_numbers = [int(number) for number in summary.get("top_red", [])[: max(1, int(top_k_per_model))]]
        if not top_numbers:
            continue
        effective_weight = max(
            float(red_weight),
            float(model_weight_floor),
            float(dormant_model_weight if red_weight <= 0.0 else 0.0),
        )
        top_scores = [float(value) for value in summary.get("top_red_scores", [])[: len(top_numbers)]]
        model_max = max(top_scores, default=0.0)
        for rank, number in enumerate(top_numbers):
            base = (
                top_scores[rank] / model_max
                if rank < len(top_scores) and model_max > 0.0
                else 1.0 - rank / max(len(top_numbers) - 1, 1)
            )
            scores[number] += float(effective_weight) * float(base)
            if red_weight <= 0.0 and rank < 8:
                scores[number] += float(dormant_top_rank_bonus_weight) * (1.0 - rank / 8.0)
            model_support[number].add(str(model_name))

    if any(model_support.values()):
        for number, support in model_support.items():
            if not support:
                continue
            scores[number] += float(agreement_bonus_weight) * (len(support) / 3.0)

    ordered = sorted(
        scores.items(),
        key=lambda item: (
            float(item[1]),
            len(model_support[item[0]]),
            -(fusion_top.index(item[0]) if item[0] in fusion_top else 999),
            -item[0],
        ),
        reverse=True,
    )
    merged = [int(number) for number, value in ordered if value > 0.0]
    fallback = [int(number) for number in fusion_top if int(number) not in merged]
    return [*merged, *fallback][:desired]


def _normalize_score(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return values / total


def _rank_support_map(numbers: list[int]) -> dict[int, float]:
    total = max(len(numbers), 1)
    return {
        int(number): max(0.0, 1.0 - rank / total)
        for rank, number in enumerate(numbers)
    }


def _build_history_context_red_numbers(
    history: pd.DataFrame,
    *,
    desired_count: int,
    decay: float = 0.985,
    transition_alpha: float = 1.0,
) -> list[int]:
    if history is None or len(history) == 0:
        return []
    reds = history[RED_COLUMNS].to_numpy(dtype=int)
    matrix = np.zeros((len(reds), 33), dtype=float)
    rows = np.repeat(np.arange(len(reds)), 6)
    cols = reds.reshape(-1) - 1
    valid = (cols >= 0) & (cols < 33)
    matrix[rows[valid], cols[valid]] = 1.0

    window30 = _normalize_score(matrix[-min(len(matrix), 30):].sum(axis=0) + 1.0)
    window100 = _normalize_score(matrix[-min(len(matrix), 100):].sum(axis=0) + 1.0)
    ages = np.arange(len(matrix) - 1, -1, -1)
    decayed = _normalize_score(np.power(float(decay), ages) @ matrix + 1e-9)

    last_seen = np.full(33, -1, dtype=int)
    counts = matrix.sum(axis=0)
    for row_index, row in enumerate(matrix):
        last_seen[np.flatnonzero(row)] = row_index
    current_gap = np.where(last_seen >= 0, len(matrix) - 1 - last_seen, len(matrix)).astype(float)
    expected_gap = len(matrix) / np.maximum(counts, 1.0)
    omission = _normalize_score(1.0 / (1.0 + np.exp(-(current_gap - expected_gap) / np.maximum(expected_gap, 1.0))))

    latest_red = set(int(number) for number in reds[-1])
    transition = np.full(33, float(transition_alpha), dtype=float)
    if len(matrix) >= 2:
        for prior, following in zip(reds[:-1], matrix[1:]):
            overlap = len(latest_red & {int(number) for number in prior})
            if overlap:
                transition += (float(overlap) ** 1.25) * following
    transition = _normalize_score(transition)

    recent = reds[-min(len(reds), 120):]
    zone_density = np.array([
        (recent <= 11).sum(axis=1).mean(),
        ((recent >= 12) & (recent <= 22)).sum(axis=1).mean(),
        (recent >= 23).sum(axis=1).mean(),
    ]) / 11.0
    zone = np.zeros(33, dtype=float)
    zone[:11] = zone_density[0]
    zone[11:22] = zone_density[1]
    zone[22:] = zone_density[2]
    zone = _normalize_score(zone + 1e-9)

    combined = _normalize_score(
        0.20 * window30
        + 0.12 * window100
        + 0.18 * decayed
        + 0.24 * transition
        + 0.18 * omission
        + 0.08 * zone
    )
    return [int(index + 1) for index in np.argsort(-combined, kind="stable")[: max(6, int(desired_count))]]


def _build_model_diversity_red_numbers(
    model_rankings: dict[str, dict[str, Any]] | None,
    *,
    desired_count: int,
    top_k_per_model: int = 10,
) -> list[int]:
    if not model_rankings:
        return []
    ordered_models = sorted(
        model_rankings.items(),
        key=lambda item: (
            -float(item[1].get("red_weight", item[1].get("weight", 0.0)) or 0.0),
            str(item[0]),
        ),
    )
    per_model = [
        [
            int(number)
            for number in summary.get("top_red", [])[: max(1, int(top_k_per_model))]
            if 1 <= int(number) <= 33
        ]
        for _, summary in ordered_models
    ]
    result: list[int] = []
    seen: set[int] = set()
    max_len = max((len(numbers) for numbers in per_model), default=0)
    for rank in range(max_len):
        for numbers in per_model:
            if rank >= len(numbers):
                continue
            number = int(numbers[rank])
            if number in seen:
                continue
            seen.add(number)
            result.append(number)
            if len(result) >= int(desired_count):
                return result
    return result


def _build_weak_signal_ranked_blue_numbers(
    fusion_ranking: dict[str, Any],
    model_rankings: dict[str, dict[str, Any]] | None,
    *,
    desired_count: int,
    top_k_per_model: int = 4,
    model_weight_floor: float = 0.03,
    dormant_model_weight: float = 0.16,
    fusion_bonus_weight: float = 0.65,
    agreement_bonus_weight: float = 0.18,
    minority_top1_bonus_weight: float = 0.72,
) -> tuple[list[int], dict[int, float]]:
    desired = max(1, min(16, int(desired_count)))
    scores = {number: 0.0 for number in range(1, 17)}
    fusion_top = [int(number) for number in fusion_ranking.get("top_blue", [])]
    fusion_scores = [float(value) for value in fusion_ranking.get("top_blue_scores", [])]
    fusion_max = max(fusion_scores, default=0.0)
    if fusion_top:
        for rank, number in enumerate(fusion_top[:desired]):
            base = (
                fusion_scores[rank] / fusion_max
                if rank < len(fusion_scores) and fusion_max > 0.0
                else 1.0 - rank / max(len(fusion_top) - 1, 1)
            )
            scores[int(number)] += float(fusion_bonus_weight) * float(base)

    model_support: dict[int, set[str]] = {number: set() for number in range(1, 17)}
    minority_top1: dict[int, int] = {number: 0 for number in range(1, 17)}
    for model_name, summary in (model_rankings or {}).items():
        blue_weight = float(summary.get("blue_weight", summary.get("weight", 0.0)))
        top_numbers = [int(number) for number in summary.get("top_blue", [])[: max(1, int(top_k_per_model))]]
        if not top_numbers:
            continue
        effective_weight = max(
            float(blue_weight),
            float(model_weight_floor),
            float(dormant_model_weight if blue_weight <= 0.0 else 0.0),
        )
        top_scores = [float(value) for value in summary.get("top_blue_scores", [])[: len(top_numbers)]]
        model_max = max(top_scores, default=0.0)
        for rank, number in enumerate(top_numbers):
            base = (
                top_scores[rank] / model_max
                if rank < len(top_scores) and model_max > 0.0
                else 1.0 - rank / max(len(top_numbers) - 1, 1)
            )
            scores[number] += float(effective_weight) * float(base)
            if rank == 0 and number not in fusion_top[:2]:
                minority_top1[number] += 1
            model_support[number].add(str(model_name))

    for number, support in model_support.items():
        if support:
            scores[number] += float(agreement_bonus_weight) * (len(support) / 3.0)
        if minority_top1[number] > 0:
            scores[number] += float(minority_top1_bonus_weight) * float(minority_top1[number])

    ordered = sorted(
        scores.items(),
        key=lambda item: (
            float(item[1]),
            len(model_support[item[0]]),
            minority_top1[item[0]],
            -(fusion_top.index(item[0]) if item[0] in fusion_top else 999),
            -item[0],
        ),
        reverse=True,
    )
    merged = [int(number) for number, value in ordered if value > 0.0]
    fallback = [int(number) for number in fusion_top if int(number) not in merged]
    ranked_numbers = [*merged, *fallback][:desired]
    bonus_map: dict[int, float] = {}
    max_minority = max(minority_top1.values(), default=0)
    for rank, number in enumerate(ranked_numbers):
        minority_bonus = (
            float(minority_top1[number]) / float(max_minority)
            if max_minority > 0 and minority_top1[number] > 0
            else 0.0
        )
        tail_bonus = 1.0 - rank / max(len(ranked_numbers) - 1, 1) if rank >= 2 else 0.0
        bonus_map[int(number)] = float(0.75 * minority_bonus + 0.25 * tail_bonus)
    return ranked_numbers, bonus_map


def _merge_ranked_candidates(
    *sources: list[int],
    desired_count: int,
) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    max_len = max((len(source) for source in sources), default=0)
    for index in range(max_len):
        for source in sources:
            if index >= len(source):
                continue
            number = source[index]
            value = int(number)
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
            if len(merged) >= int(desired_count):
                return merged
    for source in sources:
        for number in source:
            value = int(number)
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
            if len(merged) >= int(desired_count):
                return merged
    return merged


def generate_uniform_tickets(count: int, rng: np.random.Generator) -> list[dict]:
    tickets: list[dict] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    while len(tickets) < int(count):
        red = tuple(sorted((rng.choice(33, size=6, replace=False) + 1).tolist()))
        blue = int(rng.integers(1, 17))
        if (red, blue) in seen:
            continue
        seen.add((red, blue))
        tickets.append({"red": red, "blue": blue})
    return tickets
