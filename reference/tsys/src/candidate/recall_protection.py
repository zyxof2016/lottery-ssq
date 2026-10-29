from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.combination_gen.score_func import base_quality, number_score, structure_score
from src.feature_engineer.base_stats import structure_profile


def _normalize(values: Iterable[float], expected_size: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size != int(expected_size) or not np.all(np.isfinite(array)):
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    return array / total


def _is_uniform(probability: np.ndarray) -> bool:
    return bool(np.allclose(probability, probability[0], rtol=1e-10, atol=1e-12))


def _top_numbers(probability: np.ndarray, top_k: int) -> list[int]:
    limit = max(1, min(int(top_k), int(probability.size)))
    return [int(index) + 1 for index in np.argsort(probability)[::-1][:limit]]


def _as_int_list(values: Iterable[Any], *, low: int, high: int) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if low <= number <= high and number not in seen:
            result.append(number)
            seen.add(number)
    return result


def _red_config(config: dict[str, Any] | None) -> dict[str, Any]:
    v7 = (config or {}).get("v7") or {}
    red = v7.get("red") or {}
    return red if isinstance(red, dict) else {}


def apply_red_fusion_recall_channel(
    fused_probability: dict[str, Any],
    predictions: dict[str, dict[str, Any]] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Blend a recall-first independent-model channel into the red fusion output.

    The ordinary fusion score is still the probability channel.  This pass adds
    a second channel before ranking/candidate generation so a number found by
    an independent model Top6/Top10 is not flattened away by probability
    averaging.  Blue is intentionally untouched; current evidence says blue is
    a no-edge coverage problem rather than a Top1 ranking problem.
    """

    red_cfg = _red_config(config)
    enabled = bool(red_cfg.get("fusion_recall_channel_enabled", True))
    red_probability = _normalize((fused_probability or {}).get("red", []), 33)
    blue_probability = _normalize((fused_probability or {}).get("blue", []), 16)
    if not enabled or not predictions:
        return {"red": red_probability, "blue": blue_probability}, {
            "schema": "red_fusion_recall_channel_v8",
            "enabled": False,
            "reason": "disabled_or_no_predictions",
        }

    alpha = min(0.60, max(0.0, float(red_cfg.get("fusion_recall_channel_alpha", 0.26) or 0.26)))
    top6_cutoff = max(1, min(int(red_cfg.get("independent_model_top6_cutoff", 6) or 6), 33))
    top10_cutoff = max(top6_cutoff, min(int(red_cfg.get("independent_model_top10_cutoff", 10) or 10), 33))
    top18_cutoff = max(top10_cutoff, min(int(red_cfg.get("fusion_recall_channel_top18_cutoff", 18) or 18), 33))
    top6_bonus = max(0.0, float(red_cfg.get("fusion_recall_channel_top6_bonus", 1.00) or 1.00))
    top10_bonus = max(0.0, float(red_cfg.get("fusion_recall_channel_top10_bonus", 0.62) or 0.62))
    top18_bonus = max(0.0, float(red_cfg.get("fusion_recall_channel_top18_bonus", 0.28) or 0.28))
    support_bonus = max(0.0, float(red_cfg.get("fusion_recall_channel_support_bonus", 0.10) or 0.10))
    floor_multiplier_top6 = max(1.0, float(red_cfg.get("fusion_recall_channel_top6_floor_multiplier", 1.22) or 1.22))
    floor_multiplier_top10 = max(1.0, float(red_cfg.get("fusion_recall_channel_top10_floor_multiplier", 1.10) or 1.10))
    include_uniform = bool(red_cfg.get("independent_model_rescue_include_uniform_models", False))

    recall_raw = np.zeros(33, dtype=float)
    support: dict[int, dict[str, Any]] = {
        number: {
            "number": number,
            "best_model_rank": 99,
            "model_support_count": 0,
            "top6_models": [],
            "top10_models": [],
            "top18_models": [],
        }
        for number in range(1, 34)
    }
    usable_models: list[str] = []
    for model_name, prediction in (predictions or {}).items():
        probability = _normalize((prediction or {}).get("red", []), 33)
        if _is_uniform(probability) and not include_uniform:
            continue
        usable_models.append(str(model_name))
        order = _top_numbers(probability, top18_cutoff)
        max_probability = max((float(probability[number - 1]) for number in order), default=1.0) or 1.0
        for rank, number in enumerate(order, start=1):
            if rank <= top6_cutoff:
                tier_bonus = top6_bonus * (1.0 + 0.18 * (top6_cutoff - rank) / max(1, top6_cutoff - 1))
                bucket = "top6_models"
            elif rank <= top10_cutoff:
                tier_bonus = top10_bonus
                bucket = "top10_models"
            else:
                tier_bonus = top18_bonus
                bucket = "top18_models"
            probability_tilt = float(probability[number - 1]) / max_probability
            recall_raw[number - 1] += tier_bonus * (0.70 + 0.30 * probability_tilt)
            item = support[int(number)]
            item[bucket].append({
                "model": str(model_name),
                "rank": int(rank),
                "probability": float(probability[number - 1]),
            })
            item["best_model_rank"] = min(int(item["best_model_rank"]), int(rank))

    supported_numbers = [number for number, item in support.items() if int(item["best_model_rank"]) < 99]
    if not supported_numbers or float(recall_raw.sum()) <= 0.0 or alpha <= 0.0:
        return {"red": red_probability, "blue": blue_probability}, {
            "schema": "red_fusion_recall_channel_v8",
            "enabled": False,
            "reason": "no_independent_recall_signal",
            "usable_models": usable_models,
        }

    for number in supported_numbers:
        item = support[number]
        models = {
            row["model"]
            for key in ("top6_models", "top10_models", "top18_models")
            for row in item[key]
        }
        item["model_support_count"] = len(models)
        recall_raw[number - 1] += support_bonus * max(0, len(models) - 1)

    recall_distribution = recall_raw / float(recall_raw.sum())
    adjusted = (1.0 - alpha) * red_probability + alpha * recall_distribution

    uniform_floor = 1.0 / 33.0
    floored_numbers: list[dict[str, Any]] = []
    for number in supported_numbers:
        rank = int(support[number]["best_model_rank"])
        multiplier = floor_multiplier_top6 if rank <= top6_cutoff else floor_multiplier_top10 if rank <= top10_cutoff else 1.0
        floor = uniform_floor * multiplier
        before = float(adjusted[number - 1])
        if multiplier > 1.0 and before < floor:
            adjusted[number - 1] = floor
            floored_numbers.append({
                "number": int(number),
                "best_model_rank": rank,
                "before": before,
                "after_floor": float(floor),
            })
    adjusted = _normalize(adjusted, 33)

    before_order = _top_numbers(red_probability, 18)
    after_order = _top_numbers(adjusted, 18)
    details = sorted(
        [support[number] for number in supported_numbers],
        key=lambda item: (
            -int(item.get("model_support_count", 0) or 0),
            -len(item.get("top6_models", []) or []),
            -len(item.get("top10_models", []) or []),
            -len(item.get("top18_models", []) or []),
            -int(item.get("best_model_rank", 99) or 99),
            -int(item.get("number", 0) or 0),
        ),
    )
    return {"red": adjusted, "blue": blue_probability}, {
        "schema": "red_fusion_recall_channel_v8",
        "enabled": True,
        "applied": True,
        "alpha": float(alpha),
        "policy": "Probability Channel + Independent Recall Channel before ranking and candidate generation.",
        "usable_models": usable_models,
        "top6_cutoff": int(top6_cutoff),
        "top10_cutoff": int(top10_cutoff),
        "top18_cutoff": int(top18_cutoff),
        "before_top18": before_order,
        "after_top18": after_order,
        "added_to_top18": sorted(set(after_order) - set(before_order)),
        "removed_from_top18": sorted(set(before_order) - set(after_order)),
        "floored_numbers": floored_numbers,
        "protected_details": details[:24],
    }


def build_independent_model_rescue(
    predictions: dict[str, dict[str, Any]] | None = None,
    *,
    model_rankings: dict[str, dict[str, Any]] | None = None,
    fused_red_probability: Iterable[float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find red numbers supported by independent model Top-K before candidate generation.

    The protection is intentionally conservative: it does not force numbers into
    the final portfolio.  It only prevents an independently discovered number
    from losing all mass before the Candidate Set is built.
    """
    red_cfg = _red_config(config)
    enabled = bool(red_cfg.get("candidate_recall_guard_before_candidate", True))
    if not enabled:
        return {
            "schema": "candidate_recall_protection_v8",
            "enabled": False,
            "reason": "candidate_recall_guard_before_candidate_disabled",
            "protected_numbers": [],
        }

    top6_cutoff = max(1, min(int(red_cfg.get("independent_model_top6_cutoff", 6)), 33))
    top10_cutoff = max(top6_cutoff, min(int(red_cfg.get("independent_model_top10_cutoff", 10)), 33))
    max_protected = max(0, min(int(red_cfg.get("independent_model_rescue_max_numbers", 18)), 33))
    max_total_protected = max(
        1,
        min(int(red_cfg.get("candidate_recall_guard_max_total_protected", max_protected or 24)), 33),
    )
    min_model_support = max(1, int(red_cfg.get("independent_model_rescue_min_model_support", 1)))
    include_uniform = bool(red_cfg.get("independent_model_rescue_include_uniform_models", False))
    floor = float(red_cfg.get("independent_model_evidence_floor", 0.0) or 0.0)
    if floor <= 0.0:
        floor = (1.0 / 33.0) * float(red_cfg.get("independent_model_evidence_floor_multiplier", 1.18))

    fused = _normalize([] if fused_red_probability is None else fused_red_probability, 33)
    fusion_ranks = {
        int(number): int(rank)
        for rank, number in enumerate(_top_numbers(fused, 33), start=1)
    }
    fusion_guard_enabled = bool(red_cfg.get("fusion_topn_recall_guard_enabled", True))
    fusion_guard_count = max(1, min(int(red_cfg.get("fusion_topn_recall_guard_count", 18)), 33))
    fusion_guard_max = max(0, min(int(red_cfg.get("fusion_topn_recall_guard_max_numbers", fusion_guard_count)), 33))
    fusion_guard_numbers = (
        _top_numbers(fused, fusion_guard_count)[:fusion_guard_max]
        if fusion_guard_enabled
        else []
    )

    support: dict[int, dict[str, Any]] = {}
    usable_models: list[str] = []

    if predictions:
        for model_name, prediction in predictions.items():
            probability = _normalize((prediction or {}).get("red", []), 33)
            if _is_uniform(probability) and not include_uniform:
                continue
            usable_models.append(str(model_name))
            for cutoff, field in ((top6_cutoff, "top6_models"), (top10_cutoff, "top10_models")):
                for rank, number in enumerate(_top_numbers(probability, cutoff), start=1):
                    item = support.setdefault(
                        int(number),
                        {
                            "number": int(number),
                            "top6_models": [],
                            "top10_models": [],
                            "best_model_rank": 99,
                            "best_model_probability": 0.0,
                        },
                    )
                    item[field].append(
                        {
                            "model": str(model_name),
                            "rank": int(rank),
                            "probability": float(probability[int(number) - 1]),
                        }
                    )
                    item["best_model_rank"] = min(int(item["best_model_rank"]), int(rank))
                    item["best_model_probability"] = max(
                        float(item["best_model_probability"]),
                        float(probability[int(number) - 1]),
                    )
    elif model_rankings:
        for model_name, ranking in model_rankings.items():
            if bool((ranking or {}).get("red_is_uniform")) and not include_uniform:
                continue
            usable_models.append(str(model_name))
            top_red = [
                int(value)
                for value in (ranking or {}).get("top_red", []) or []
                if str(value).isdigit() and 1 <= int(value) <= 33
            ]
            scores = [float(value) for value in (ranking or {}).get("top_red_scores", []) or []]
            for rank, number in enumerate(top_red[:top10_cutoff], start=1):
                item = support.setdefault(
                    int(number),
                    {
                        "number": int(number),
                        "top6_models": [],
                        "top10_models": [],
                        "best_model_rank": 99,
                        "best_model_probability": 0.0,
                    },
                )
                payload = {
                    "model": str(model_name),
                    "rank": int(rank),
                    "probability": float(scores[rank - 1]) if rank - 1 < len(scores) else None,
                }
                if rank <= top6_cutoff:
                    item["top6_models"].append(payload)
                item["top10_models"].append(payload)
                item["best_model_rank"] = min(int(item["best_model_rank"]), int(rank))
                if payload["probability"] is not None:
                    item["best_model_probability"] = max(
                        float(item["best_model_probability"]),
                        float(payload["probability"]),
                    )

    candidates = []
    for number, item in support.items():
        top6_models = item.get("top6_models", []) or []
        top10_models = item.get("top10_models", []) or []
        model_support_count = len({row["model"] for row in [*top6_models, *top10_models]})
        if model_support_count < min_model_support:
            continue
        candidates.append(
            {
                **item,
                "number": int(number),
                "model_support_count": int(model_support_count),
                "fusion_rank": int(fusion_ranks.get(int(number), 99)),
                "fusion_probability": float(fused[int(number) - 1]),
                "protection_sources": ["independent_model_top6_top10"],
            }
        )

    for rank, number in enumerate(fusion_guard_numbers, start=1):
        candidates.append(
            {
                "number": int(number),
                "top6_models": [],
                "top10_models": [],
                "best_model_rank": 99,
                "best_model_probability": 0.0,
                "model_support_count": 0,
                "fusion_rank": int(rank),
                "fusion_probability": float(fused[int(number) - 1]),
                "protection_sources": ["fusion_topn"],
            }
        )

    merged_candidates: dict[int, dict[str, Any]] = {}
    for item in candidates:
        number = int(item["number"])
        existing = merged_candidates.get(number)
        if not existing:
            merged_candidates[number] = item
            continue
        existing["top6_models"] = list(existing.get("top6_models", []) or []) + list(item.get("top6_models", []) or [])
        existing["top10_models"] = list(existing.get("top10_models", []) or []) + list(item.get("top10_models", []) or [])
        existing["best_model_rank"] = min(int(existing.get("best_model_rank", 99)), int(item.get("best_model_rank", 99)))
        existing["best_model_probability"] = max(
            float(existing.get("best_model_probability", 0.0) or 0.0),
            float(item.get("best_model_probability", 0.0) or 0.0),
        )
        existing["model_support_count"] = max(
            int(existing.get("model_support_count", 0) or 0),
            int(item.get("model_support_count", 0) or 0),
        )
        existing["fusion_rank"] = min(int(existing.get("fusion_rank", 99)), int(item.get("fusion_rank", 99)))
        existing["fusion_probability"] = max(
            float(existing.get("fusion_probability", 0.0) or 0.0),
            float(item.get("fusion_probability", 0.0) or 0.0),
        )
        existing["protection_sources"] = sorted(
            set(existing.get("protection_sources", []) or []) | set(item.get("protection_sources", []) or [])
        )

    ranked_candidates = sorted(
        merged_candidates.values(),
        key=lambda item: (
            "fusion_topn" in (item.get("protection_sources") or []),
            bool(item.get("top6_models")),
            int(item.get("model_support_count", 0)),
            -int(item.get("fusion_rank", 99)),
            float(item.get("best_model_probability", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for number in fusion_guard_numbers:
        item = merged_candidates.get(int(number))
        if item and item not in selected:
            selected.append(item)
    for item in ranked_candidates:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max_total_protected:
            break
    if max_protected > 0 and not fusion_guard_numbers:
        selected = selected[:max_protected]
    protected_numbers = [int(item["number"]) for item in selected]
    return {
        "schema": "candidate_recall_protection_v8",
        "enabled": True,
        "stage": "fusion_to_candidate_before_sampling",
        "policy": "Independent Model Evidence + Fusion TopN: independent Top6/Top10 and fused TopN receive minimum evidence floors before Candidate generation.",
        "top6_cutoff": int(top6_cutoff),
        "top10_cutoff": int(top10_cutoff),
        "fusion_topn_guard_enabled": bool(fusion_guard_enabled),
        "fusion_topn_guard_count": int(fusion_guard_count),
        "fusion_topn_guard_numbers": [int(number) for number in fusion_guard_numbers],
        "minimum_evidence_floor": float(floor),
        "min_model_support": int(min_model_support),
        "max_protected_numbers": int(max_protected),
        "max_total_protected_numbers": int(max_total_protected),
        "usable_models": usable_models,
        "protected_numbers": protected_numbers,
        "protected_details": selected,
    }


def apply_red_candidate_recall_protection(
    probabilities: dict[str, Any],
    predictions: dict[str, dict[str, Any]] | None = None,
    *,
    model_rankings: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the Candidate Recall Guard before the Monte Carlo candidate pool."""
    raw_probabilities = probabilities or {}
    red_probability = _normalize(raw_probabilities.get("red", []), 33)
    blue_probability = _normalize(raw_probabilities.get("blue", []), 16)
    info = build_independent_model_rescue(
        predictions,
        model_rankings=model_rankings,
        fused_red_probability=red_probability,
        config=config,
    )
    if not bool(info.get("enabled")) or not info.get("protected_numbers"):
        return {"red": red_probability, "blue": blue_probability}, {
            **info,
            "applied": False,
            "reason": info.get("reason", "no_independent_model_rescue_numbers"),
        }

    adjusted = red_probability.copy()
    floor = float(info.get("minimum_evidence_floor", 1.0 / 33.0))
    changed: list[dict[str, Any]] = []
    for number in info.get("protected_numbers", []) or []:
        index = int(number) - 1
        before = float(adjusted[index])
        if before < floor:
            adjusted[index] = floor
            changed.append({"number": int(number), "before": before, "after_floor": float(floor)})
    adjusted = _normalize(adjusted, 33)
    return {"red": adjusted, "blue": blue_probability}, {
        **info,
        "applied": True,
        "changed_numbers": changed,
        "red_probability_before": {
            f"{int(number):02d}": float(red_probability[int(number) - 1])
            for number in info.get("protected_numbers", []) or []
        },
        "red_probability_after": {
            f"{int(number):02d}": float(adjusted[int(number) - 1])
            for number in info.get("protected_numbers", []) or []
        },
    }


def enforce_red_candidate_recall_slots(
    pool: list[dict[str, Any]],
    probabilities: dict[str, Any],
    history: pd.DataFrame,
    recall_protection: dict[str, Any] | None,
    *,
    pool_size: int,
    scoring_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Force independently protected red numbers to survive into Candidate Set slots.

    Probability floors are useful but not strict enough: a protected number can
    still disappear during sampling/ranking.  This post-sampling slot pass adds a
    small synthetic candidate ticket for every protected red that is absent from
    the retained pool, then trims only non-protected overflow tickets.
    """

    info = recall_protection or {}
    protected_numbers = [
        int(number)
        for number in info.get("protected_numbers", []) or []
        if 1 <= int(number) <= 33
    ]
    if not bool(info.get("enabled")) or not protected_numbers:
        return list(pool or []), {
            "enabled": False,
            "reason": "no_protected_numbers",
            "added_numbers": [],
        }

    red_probability = _normalize((probabilities or {}).get("red", []), 33)
    blue_probability = _normalize((probabilities or {}).get("blue", []), 16)
    existing_red = {
        int(number)
        for ticket in list(pool or [])
        for number in _as_int_list(ticket.get("red", []), low=1, high=33)
    }
    missing = [number for number in protected_numbers if number not in existing_red]
    if not missing:
        return list(pool or []), {
            "enabled": True,
            "applied": False,
            "reason": "all_protected_numbers_already_in_candidate_pool",
            "added_numbers": [],
            "protected_numbers": protected_numbers,
        }

    profile = structure_profile(history)
    red_order = [int(index) + 1 for index in np.argsort(red_probability)[::-1]]
    blue_order = [int(index) + 1 for index in np.argsort(blue_probability)[::-1]]
    additions: list[dict[str, Any]] = []
    for offset, number in enumerate(missing):
        red: list[int] = [int(number)]
        cursor = offset % max(1, len(red_order))
        while len(red) < 6 and len(set(red)) < 33:
            candidate = red_order[cursor % len(red_order)]
            cursor += 1
            if candidate not in red:
                red.append(int(candidate))
        if len(red) != 6:
            continue
        red_tuple = tuple(sorted(red))
        blue = int(blue_order[offset % len(blue_order)]) if blue_order else 1
        probability_value = number_score(red_tuple, blue, {"red": red_probability, "blue": blue_probability})
        red_probability_value = float(
            np.exp(np.mean(np.log(np.clip(red_probability[np.array(red_tuple) - 1] * 33.0, 1e-12, None))))
        )
        blue_probability_value = float(np.clip(blue_probability[blue - 1] * 16.0, 1e-12, None))
        structure_value = structure_score(red_tuple, profile)
        additions.append({
            "red": red_tuple,
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
            "selection_role": "candidate_recall_guard_protected_slot",
            "protected_red_number": int(number),
        })

    if not additions:
        return list(pool or []), {
            "enabled": True,
            "applied": False,
            "reason": "protected_slot_candidate_build_failed",
            "added_numbers": [],
            "protected_numbers": protected_numbers,
        }

    existing_keys = {
        (tuple(_as_int_list(ticket.get("red", []), low=1, high=33)), int(ticket.get("blue", 0) or 0))
        for ticket in list(pool or [])
    }
    unique_additions = []
    for ticket in additions:
        key = (tuple(_as_int_list(ticket.get("red", []), low=1, high=33)), int(ticket.get("blue", 0) or 0))
        if key in existing_keys:
            continue
        existing_keys.add(key)
        unique_additions.append(ticket)

    protected_keys = {
        (tuple(_as_int_list(ticket.get("red", []), low=1, high=33)), int(ticket.get("blue", 0) or 0))
        for ticket in unique_additions
    }
    merged = sorted(
        list(pool or []) + unique_additions,
        key=lambda item: float(item.get("quality_score", item.get("consensus_score", 0.0)) or 0.0),
        reverse=True,
    )
    limit = max(1, int(pool_size))
    if len(merged) > limit:
        forced = [ticket for ticket in merged if (
            tuple(_as_int_list(ticket.get("red", []), low=1, high=33)),
            int(ticket.get("blue", 0) or 0),
        ) in protected_keys]
        forced = forced[:limit]
        forced_keys = {
            (tuple(_as_int_list(ticket.get("red", []), low=1, high=33)), int(ticket.get("blue", 0) or 0))
            for ticket in forced
        }
        fillers = [
            ticket for ticket in merged
            if (
                tuple(_as_int_list(ticket.get("red", []), low=1, high=33)),
                int(ticket.get("blue", 0) or 0),
            ) not in forced_keys
        ][: max(0, limit - len(forced))]
        merged = sorted(
            forced + fillers,
            key=lambda item: float(item.get("quality_score", item.get("consensus_score", 0.0)) or 0.0),
            reverse=True,
        )

    added_numbers = [int(ticket.get("protected_red_number")) for ticket in unique_additions]
    return merged, {
        "enabled": True,
        "applied": bool(unique_additions),
        "policy": "protected_evidence_to_candidate_set_forced_slots",
        "protected_numbers": protected_numbers,
        "added_numbers": added_numbers,
        "added_ticket_count": len(unique_additions),
        "candidate_pool_size_after": len(merged),
    }
