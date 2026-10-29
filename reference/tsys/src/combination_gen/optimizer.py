from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.combination_gen.score_func import selection_score
from src.portfolio.coverage import coverage_summary, marginal_subset_gain, red_jaccard


def _ticket_key(ticket: dict) -> tuple[tuple[int, ...], int]:
    return tuple(int(number) for number in ticket["red"]), int(ticket["blue"])


def _sync_selected_keys(
    result: list[dict],
    selected_keys: set[tuple[tuple[int, ...], int]],
) -> None:
    selected_keys.clear()
    selected_keys.update(_ticket_key(ticket) for ticket in result)


def _signal_anchor_ticket(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: float(candidate.get("quality_score", 0.0))
        + 0.20 * float(candidate.get("blue_number_score", 1.0)),
    )


def _flatten_method_candidates(
    method_results: dict[str, list[dict]],
    *,
    limit_per_method: int | None = None,
) -> list[dict]:
    flattened: list[dict] = []
    for tickets in method_results.values():
        values = list(tickets or [])
        if limit_per_method is not None:
            values = values[: max(0, int(limit_per_method))]
        flattened.extend(values)
    return flattened


def _build_method_ticket_support(
    method_results: dict[str, list[dict]],
    *,
    top_k_per_method: int,
) -> dict[tuple[tuple[int, ...], int], float]:
    support: dict[tuple[tuple[int, ...], int], float] = {}
    limit = max(1, int(top_k_per_method))
    for tickets in method_results.values():
        ranked = list(tickets or [])[:limit]
        if not ranked:
            continue
        weights = np.linspace(1.0, 0.45, len(ranked))
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            continue
        for rank, ticket in enumerate(ranked):
            key = _ticket_key(ticket)
            support[key] = support.get(key, 0.0) + float(weights[rank]) / total_weight
    maximum = max(support.values(), default=0.0)
    if maximum <= 0.0:
        return {}
    return {key: float(value / maximum) for key, value in support.items()}


def _build_shared_method_red_core(
    method_results: dict[str, list[dict]],
    *,
    top_k_per_method: int,
    minimum_method_support: int = 2,
) -> dict[int, float]:
    if not method_results:
        return {}
    limit = max(1, int(top_k_per_method))
    required_support = max(1, int(minimum_method_support))
    raw_support = {number: 0.0 for number in range(1, 34)}
    method_hits = {number: 0 for number in range(1, 34)}
    for tickets in method_results.values():
        ranked = list(tickets or [])[:limit]
        if not ranked:
            continue
        weights = np.linspace(1.0, 0.55, len(ranked))
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            continue
        method_numbers: set[int] = set()
        for rank, ticket in enumerate(ranked):
            weight = float(weights[rank]) / total_weight
            for number in ticket["red"]:
                raw_support[int(number)] += weight
                method_numbers.add(int(number))
        for number in method_numbers:
            method_hits[number] += 1
    filtered = {
        number: float(value)
        for number, value in raw_support.items()
        if method_hits[number] >= required_support and value > 0.0
    }
    maximum = max(filtered.values(), default=0.0)
    if maximum <= 0.0:
        return {}
    return {
        number: float(value / maximum)
        for number, value in filtered.items()
    }


def _seed_consensus_anchors(
    decorated_candidates: list[dict],
    method_results: dict[str, list[dict]],
    *,
    count: int,
    top_k_per_method: int = 1,
    minimum_method_support: int = 1,
) -> list[dict]:
    if not decorated_candidates or int(count) <= 0:
        return []
    candidate_map = {
        _ticket_key(candidate): candidate
        for candidate in decorated_candidates
    }
    anchors: list[dict] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    limit = max(1, int(top_k_per_method))
    required_support = max(1, int(minimum_method_support))
    for tickets in method_results.values():
        for ticket in list(tickets or [])[:limit]:
            key = _ticket_key(ticket)
            candidate = candidate_map.get(key)
            if candidate is None or key in seen:
                continue
            if int(candidate.get("method_support_count", 0)) < required_support:
                continue
            anchors.append(candidate)
            seen.add(key)
    anchors.sort(
        key=lambda candidate: (
            int(candidate.get("method_support_count", 0)),
            float(candidate.get("consensus_score", candidate.get("quality_score", 0.0))),
            float(candidate.get("quality_score", 0.0)),
        ),
        reverse=True,
    )
    return anchors[: max(0, min(int(count), len(anchors)))]


def _method_ticket_bonus(
    candidate: dict,
    support: dict[tuple[tuple[int, ...], int], float] | None,
) -> float:
    if not support:
        return 0.0
    return float(support.get(_ticket_key(candidate), 0.0))


def _anchor_overlap_bonus(candidate: dict, anchor: dict | None) -> float:
    if not anchor:
        return 0.0
    return red_jaccard(candidate["red"], anchor["red"])


def _build_red_consensus_support(
    candidates: list[dict],
    *,
    pool_size: int,
) -> dict[int, float]:
    if not candidates:
        return {}
    effective_pool = max(1, min(int(pool_size), len(candidates)))
    weights = np.linspace(1.0, 0.35, effective_pool)
    support = {number: 0.0 for number in range(1, 34)}
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return support
    for rank, candidate in enumerate(candidates[:effective_pool]):
        weight = float(weights[rank]) / total_weight
        for number in candidate["red"]:
            support[int(number)] += weight
    maximum = max(support.values(), default=0.0)
    if maximum <= 0:
        return support
    return {
        number: float(value / maximum)
        for number, value in support.items()
    }


def _red_consensus_bonus(candidate: dict, support: dict[int, float] | None) -> float:
    if not support:
        return 0.0
    values = [float(support.get(int(number), 0.0)) for number in candidate["red"]]
    if not values:
        return 0.0
    return float(np.mean(values))


def _red_core_alignment_bonus(
    candidate: dict,
    support: dict[int, float] | None,
    *,
    top_k: int = 6,
) -> float:
    if not support:
        return 0.0
    ranked = sorted(
        ((int(number), float(value)) for number, value in support.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    core_numbers = {number for number, value in ranked[: max(1, int(top_k))] if value > 0.0}
    if not core_numbers:
        return 0.0
    matched = sum(1 for number in candidate["red"] if int(number) in core_numbers)
    return float(matched / 6.0)


def _red_core_hit_count(
    candidate: dict,
    support: dict[int, float] | None,
    *,
    top_k: int = 10,
) -> int:
    if not support:
        return 0
    ranked = sorted(
        ((int(number), float(value)) for number, value in support.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    core_numbers = {number for number, value in ranked[: max(1, int(top_k))] if value > 0.0}
    if not core_numbers:
        return 0
    return sum(1 for number in candidate["red"] if int(number) in core_numbers)


def _shared_method_core_bonus(
    candidate: dict,
    support: dict[int, float] | None,
) -> float:
    if not support:
        return 0.0
    values = [float(support.get(int(number), 0.0)) for number in candidate["red"]]
    matched = [value for value in values if value > 0.0]
    if not matched:
        return 0.0
    hit_ratio = float(len(matched) / 6.0)
    support_strength = float(np.mean(matched))
    return float(0.65 * hit_ratio + 0.35 * support_strength)


def _signal_ticket_strength(candidate: dict, lead_ticket: dict | None = None) -> float:
    lead_overlap_bonus = (
        _anchor_overlap_bonus(candidate, lead_ticket)
        if lead_ticket is not None
        else 0.0
    )
    return float(
        0.95 * float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
        + 0.55 * float(candidate.get("blue_number_score", 1.0))
        + 0.75 * float(candidate.get("blue_signal_bonus", 0.0))
        + 0.65 * float(candidate.get("method_support_count", 0.0))
        + 0.80 * float(candidate.get("method_agreement_score", 0.0))
        + 0.70 * float(lead_overlap_bonus)
        + 0.50 * float(candidate.get("quality_score", 0.0))
    )


def _build_blue_consensus_support(
    candidates: list[dict],
    *,
    pool_size: int,
) -> tuple[dict[int, float], float]:
    if not candidates:
        return {}, 0.0
    effective_pool = max(1, min(int(pool_size), len(candidates)))
    weights = np.linspace(1.0, 0.35, effective_pool)
    raw_support = {number: 0.0 for number in range(1, 17)}
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return raw_support, 0.0
    for rank, candidate in enumerate(candidates[:effective_pool]):
        blue = int(candidate["blue"])
        raw_support[blue] += float(weights[rank]) / total_weight
    maximum = max(raw_support.values(), default=0.0)
    if maximum <= 0:
        return raw_support, 0.0
    ranked = sorted((float(value) for value in raw_support.values()), reverse=True)
    dominant_share = ranked[0]
    second_share = ranked[1] if len(ranked) > 1 else 0.0
    concentration = min(1.0, dominant_share + max(dominant_share - second_share, 0.0))
    normalized = {
        number: float(value / maximum)
        for number, value in raw_support.items()
    }
    return normalized, float(concentration)


def _blue_consensus_bonus(candidate: dict, support: dict[int, float] | None) -> float:
    if not support:
        return 0.0
    return float(support.get(int(candidate["blue"]), 0.0))


def _minimum_blue_unique_count(count: int, requested: int | None = None) -> int:
    if requested is None:
        requested = 3
    return max(1, min(int(count), int(requested), 16))


def _minimum_red_unique_set_count(count: int, requested: int | None = None) -> int:
    if requested is None:
        requested = count
    return max(1, min(int(count), int(requested)))


def _ensure_minimum_blue_diversity(
    selected: list[dict],
    candidates: list[dict],
    *,
    minimum_unique_blues: int,
) -> list[dict]:
    result = list(selected)
    if len({int(ticket["blue"]) for ticket in result}) >= int(minimum_unique_blues):
        return sorted(result, key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)

    seen_blues: set[int] = set()
    duplicate_positions: list[int] = []
    for position, ticket in enumerate(result):
        blue = int(ticket["blue"])
        if blue in seen_blues:
            duplicate_positions.append(position)
        else:
            seen_blues.add(blue)

    selected_keys = {_ticket_key(ticket) for ticket in result}
    for position in duplicate_positions:
        current_blues = {int(ticket["blue"]) for ticket in result}
        if len(current_blues) >= int(minimum_unique_blues):
            break
        alternatives = [
            candidate
            for candidate in candidates
            if int(candidate["blue"]) not in current_blues and _ticket_key(candidate) not in selected_keys
        ]
        if not alternatives:
            # 优先选择模型推荐的未使用蓝球，而不是任意未使用蓝球
            blue_scores = {number: 0.0 for number in range(1, 17)}
            for candidate in candidates[:100]:
                quality = float(candidate.get("quality_score", 1.0) or 1.0)
                blue = int(candidate.get("blue", 1))
                blue_scores[blue] += max(quality, 0.0) + 1e-6
            unused_blues = [
                number
                for number in range(1, 17)
                if number not in current_blues
            ]
            if not unused_blues:
                break
            # 按模型推荐分数排序，优先选择分数高的未使用蓝球
            unused_blues.sort(key=lambda n: (blue_scores.get(n, 0.0), -n), reverse=True)
            previous = result[position]
            replacement_blue = int(unused_blues[0])
            selected_keys.discard(_ticket_key(previous))
            result[position] = {
                **previous,
                "blue": replacement_blue,
                "selection_role": f"{previous.get('selection_role', 'selected')}_blue_diversity_fallback",
                "blue_diversity_fallback": True,
                "selection_reason": {
                    **(previous.get("selection_reason", {}) if isinstance(previous.get("selection_reason"), dict) else {}),
                    "blue_diversity_fallback": (
                        "候选池缺少未用蓝球候选，保留红球并强制替换蓝球以满足覆盖约束"
                    ),
                    "previous_blue": int(previous.get("blue", 0) or 0),
                    "replacement_blue": replacement_blue,
                },
            }
            selected_keys.add(_ticket_key(result[position]))
            logging.warning(
                "蓝球唯一性兜底触发 | 事件=BLUE_DIVERSITY_FALLBACK | 位置=%s | 原蓝球=%s | 新蓝球=%s | 目标唯一数=%s",
                position,
                previous.get("blue"),
                replacement_blue,
                minimum_unique_blues,
            )
            continue
        context = [ticket for index, ticket in enumerate(result) if index != position]
        replacement = max(
            alternatives,
            key=lambda candidate: float(candidate["quality_score"])
            + 0.30 * float(candidate.get("blue_number_score", 1.0))
            + 0.35 * float(candidate.get("blue_signal_bonus", 0.0))
            + 0.15 * marginal_subset_gain(context, candidate)
            - 0.20 * max((red_jaccard(candidate["red"], item["red"]) for item in context), default=0.0),
        )
        selected_keys.discard(_ticket_key(result[position]))
        result[position] = replacement
        selected_keys.add(_ticket_key(replacement))
        _sync_selected_keys(result, selected_keys)
    return sorted(result, key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)


def _replace_ticket(
    result: list[dict],
    position: int,
    replacement: dict,
    selected_keys: set[tuple[tuple[int, ...], int]],
) -> None:
    selected_keys.discard(_ticket_key(result[position]))
    result[position] = replacement
    selected_keys.add(_ticket_key(replacement))
    _sync_selected_keys(result, selected_keys)


def _locked_positions(
    result: list[dict],
    *,
    preserve_top_k: int = 0,
) -> set[int]:
    if int(preserve_top_k) <= 0 or not result:
        return set()
    ranked = sorted(
        range(len(result)),
        key=lambda index: (
            float(result[index].get("consensus_score", result[index].get("quality_score", 0.0))),
            float(result[index].get("quality_score", 0.0)),
        ),
        reverse=True,
    )
    return set(ranked[: max(0, min(int(preserve_top_k), len(result)))])


def _ensure_blue_signal_reserve(
    selected: list[dict],
    candidates: list[dict],
    *,
    reserve_count: int = 0,
    minimum_bonus: float = 0.0,
    preserve_top_k: int = 0,
    minimum_quality_ratio: float = 0.0,
) -> list[dict]:
    result = list(selected)
    if not result or int(reserve_count) <= 0:
        return result

    ranked_candidates = sorted(
        candidates,
        key=lambda ticket: (
            float(ticket.get("blue_signal_bonus", 0.0)),
            float(ticket.get("blue_number_score", 0.0)),
            float(ticket.get("consensus_score", ticket.get("quality_score", 0.0))),
        ),
        reverse=True,
    )
    reserved_candidates = [
        ticket
        for ticket in ranked_candidates
        if float(ticket.get("blue_signal_bonus", 0.0)) >= float(minimum_bonus)
    ][: max(0, int(reserve_count))]
    if not reserved_candidates:
        return result

    selected_keys = {_ticket_key(ticket) for ticket in result}
    selected_blue_keys = {
        int(ticket["blue"])
        for ticket in result
        if float(ticket.get("blue_signal_bonus", 0.0)) >= float(minimum_bonus)
    }
    missing_reserved = [
        ticket for ticket in reserved_candidates if int(ticket["blue"]) not in selected_blue_keys
    ]
    if not missing_reserved:
        return result

    locked_positions = _locked_positions(result, preserve_top_k=preserve_top_k)
    pool_quality_max = max(
        (
            float(ticket.get("consensus_score", ticket.get("quality_score", 0.0)))
            for ticket in candidates
        ),
        default=0.0,
    )

    def ticket_score(ticket: dict) -> float:
        return float(ticket.get("consensus_score", ticket.get("quality_score", 0.0)))

    for reserve_ticket in missing_reserved:
        quality_floor = float(pool_quality_max) * float(minimum_quality_ratio)
        if ticket_score(reserve_ticket) < quality_floor:
            continue
        replaceable_positions = [
            index for index in range(len(result)) if index not in locked_positions
        ]
        if not replaceable_positions:
            break
        position = min(
            replaceable_positions,
            key=lambda index: (
                float(result[index].get("blue_signal_bonus", 0.0)),
                float(result[index].get("blue_number_score", 0.0)),
                ticket_score(result[index]),
            ),
        )
        existing_red_sets = {
            tuple(int(number) for number in ticket["red"])
            for idx, ticket in enumerate(result)
            if idx != position
        }
        blue_taken_elsewhere = {
            int(ticket["blue"])
            for idx, ticket in enumerate(result)
            if idx != position
        }
        replacement = reserve_ticket
        if (
            _ticket_key(replacement) in selected_keys
            or tuple(int(number) for number in replacement["red"]) in existing_red_sets
            or int(replacement["blue"]) in blue_taken_elsewhere
        ):
            alternatives = [
                ticket
                for ticket in ranked_candidates
                if float(ticket.get("blue_signal_bonus", 0.0)) >= float(minimum_bonus)
                and _ticket_key(ticket) not in selected_keys
                and tuple(int(number) for number in ticket["red"]) not in existing_red_sets
                and int(ticket["blue"]) not in blue_taken_elsewhere
                and ticket_score(ticket) >= quality_floor
            ]
            if not alternatives:
                continue
            replacement = alternatives[0]
        selected_keys.discard(_ticket_key(result[position]))
        result[position] = replacement
        selected_keys.add(_ticket_key(replacement))
        _sync_selected_keys(result, selected_keys)
    return result


def _ensure_blue_focus_alignment(
    selected: list[dict],
    candidates: list[dict],
    *,
    dominant_blue: int | None = None,
    focus_strength: float = 0.0,
    minimum_repeat_count: int = 0,
    preserve_top_k: int = 0,
    minimum_quality_ratio: float = 0.0,
    lead_ticket: dict | None = None,
) -> list[dict]:
    result = list(selected)
    if (
        not result
        or dominant_blue is None
        or int(minimum_repeat_count) <= 0
        or float(focus_strength) <= 0.0
    ):
        return result

    dominant_blue = int(dominant_blue)
    current_repeat = sum(1 for ticket in result if int(ticket["blue"]) == dominant_blue)
    if current_repeat >= int(minimum_repeat_count):
        return result

    ranked_candidates = sorted(
        (
            ticket for ticket in candidates
            if int(ticket["blue"]) == dominant_blue
        ),
        key=lambda ticket: (
            float(ticket.get("blue_signal_bonus", 0.0)),
            float(ticket.get("blue_number_score", 0.0)),
            float(ticket.get("consensus_score", ticket.get("quality_score", 0.0))),
        ),
        reverse=True,
    )
    if not ranked_candidates:
        return result

    def ticket_score(ticket: dict) -> float:
        return float(ticket.get("consensus_score", ticket.get("quality_score", 0.0)))

    def lead_overlap(ticket: dict) -> float:
        if lead_ticket is None:
            return 0.0
        return _anchor_overlap_bonus(ticket, lead_ticket)

    def red_number_score(ticket: dict) -> float:
        return float(ticket.get("red_number_score", ticket.get("number_score", 0.0)))

    def method_support_score(ticket: dict) -> float:
        return float(ticket.get("method_support_count", 0.0))

    selected_keys = {_ticket_key(ticket) for ticket in result}
    locked_positions = _locked_positions(result, preserve_top_k=preserve_top_k)
    pool_quality_max = max((ticket_score(ticket) for ticket in candidates), default=0.0)
    quality_floor = float(pool_quality_max) * float(minimum_quality_ratio)

    for replacement in ranked_candidates:
        if current_repeat >= int(minimum_repeat_count):
            break
        if _ticket_key(replacement) in selected_keys or ticket_score(replacement) < quality_floor:
            continue
        replaceable_positions = [
            index
            for index, ticket in enumerate(result)
            if index not in locked_positions and int(ticket["blue"]) != dominant_blue
        ]
        if not replaceable_positions:
            break
        position = min(
            replaceable_positions,
            key=lambda index: (
                float(result[index].get("blue_signal_bonus", 0.0)),
                float(result[index].get("blue_number_score", 0.0)),
                method_support_score(result[index]),
                red_number_score(result[index]),
                lead_overlap(result[index]),
                ticket_score(result[index]),
            ),
        )
        existing_red_sets = {
            tuple(int(number) for number in ticket["red"])
            for idx, ticket in enumerate(result)
            if idx != position
        }
        if tuple(int(number) for number in replacement["red"]) in existing_red_sets:
            continue
        _replace_ticket(result, position, replacement, selected_keys)
        current_repeat += 1
    return result


def _ensure_signal_portfolio_diversity(
    selected: list[dict],
    candidates: list[dict],
    *,
    minimum_unique_red_sets: int,
    minimum_red_coverage: int,
    minimum_unique_blues: int,
    red_support: dict[int, float] | None = None,
    red_consensus_weight: float = 0.0,
    blue_support: dict[int, float] | None = None,
    blue_consensus_weight: float = 0.0,
    method_ticket_support: dict[tuple[tuple[int, ...], int], float] | None = None,
    method_ticket_bonus_weight: float = 0.0,
    method_core_red_support: dict[int, float] | None = None,
    method_core_bonus_weight: float = 0.0,
    shared_method_red_core: dict[int, float] | None = None,
    shared_method_core_bonus_weight: float = 0.0,
    preserve_top_k: int = 0,
    lead_ticket: dict | None = None,
    lead_overlap_floor: int = 0,
    maximum_duplicate_red_sets: int | None = None,
    reserved_blue_signal_count: int = 0,
    reserved_blue_signal_min_bonus: float = 0.0,
    reserved_blue_signal_min_quality_ratio: float = 0.0,
    dominant_blue: int | None = None,
    dominant_blue_focus_strength: float = 0.0,
    dominant_blue_min_repeat_count: int = 0,
    dominant_blue_min_quality_ratio: float = 0.0,
) -> list[dict]:
    result = list(selected)
    if not result:
        return result

    selected_keys = {_ticket_key(ticket) for ticket in result}
    locked_positions = _locked_positions(result, preserve_top_k=preserve_top_k)

    def portfolio_score(ticket: dict) -> float:
        return float(ticket.get("consensus_score", ticket.get("quality_score", 0.0)))

    pool_quality_max = max((portfolio_score(ticket) for ticket in candidates), default=0.0)

    def final_rank_score(ticket: dict) -> float:
        lead_overlap_bonus = (
            _anchor_overlap_bonus(ticket, lead_ticket)
            if lead_ticket is not None
            else 0.0
        )
        return (
            portfolio_score(ticket)
            + 0.50 * float(ticket.get("red_number_score", ticket.get("number_score", 1.0)))
            + 0.40 * float(ticket.get("blue_number_score", 1.0))
            + 0.55 * float(ticket.get("blue_signal_bonus", 0.0))
            + 0.70 * float(lead_overlap_bonus)
            + 0.45 * float(ticket.get("method_support_count", 0.0))
            + 0.60 * float(ticket.get("method_agreement_score", 0.0))
            + 0.75 * _signal_ticket_strength(ticket, lead_ticket)
        )

    def current_unique_red_sets() -> int:
        return len({tuple(int(number) for number in ticket["red"]) for ticket in result})

    def current_red_coverage() -> int:
        return len({int(number) for ticket in result for number in ticket["red"]})

    def replacement_score(candidate: dict, context: list[dict]) -> float:
        used_blues = {int(ticket["blue"]) for ticket in context}
        lead_overlap_bonus = (
            _anchor_overlap_bonus(candidate, lead_ticket)
            if lead_ticket is not None
            else 0.0
        )
        return (
            portfolio_score(candidate)
            + 0.35 * float(candidate.get("blue_number_score", 1.0))
            + 0.35 * float(candidate.get("blue_signal_bonus", 0.0))
            + 0.30 * float(candidate.get("blue_signal_bonus", 0.0)) * float(lead_overlap_bonus)
            + 0.30 * marginal_subset_gain(context, candidate)
            + 0.20 * float(int(candidate["blue"]) not in used_blues)
            - 0.30 * max((red_jaccard(candidate["red"], item["red"]) for item in context), default=0.0)
            + float(red_consensus_weight) * _red_consensus_bonus(candidate, red_support)
            + float(blue_consensus_weight) * _blue_consensus_bonus(candidate, blue_support)
            + float(method_ticket_bonus_weight) * _method_ticket_bonus(candidate, method_ticket_support)
            + float(method_core_bonus_weight) * _red_consensus_bonus(candidate, method_core_red_support)
            + float(shared_method_core_bonus_weight) * _shared_method_core_bonus(candidate, shared_method_red_core)
        )

    def replacement_quality_floor(current_ticket: dict) -> float:
        current_score = portfolio_score(current_ticket)
        current_red_score = float(current_ticket.get("red_number_score", current_ticket.get("number_score", 0.0)))
        support_count = float(current_ticket.get("method_support_count", 0.0))
        overlap_strength = (
            _anchor_overlap_bonus(current_ticket, lead_ticket)
            if lead_ticket is not None
            else 0.0
        )
        floor_ratio = 0.86
        if support_count >= 2.0 or overlap_strength >= 0.50:
            floor_ratio = 0.93
        if support_count >= 2.0 and overlap_strength >= 0.50:
            floor_ratio = 0.97
        return max(
            float(pool_quality_max) * 0.58,
            current_score * floor_ratio,
            current_red_score * 0.82,
        )

    def duplicate_relief_floor(current_ticket: dict) -> float:
        current_score = portfolio_score(current_ticket)
        current_red_score = float(current_ticket.get("red_number_score", current_ticket.get("number_score", 0.0)))
        return max(
            float(pool_quality_max) * 0.15,
            current_score * 0.45,
            current_red_score * 0.55,
        )

    def priority_red_core() -> list[int]:
        core_scores: dict[int, float] = {}
        for number in range(1, 34):
            score = (
                1.40 * float((shared_method_red_core or {}).get(number, 0.0))
                + 1.05 * float((method_core_red_support or {}).get(number, 0.0))
                + 0.70 * float((red_support or {}).get(number, 0.0))
            )
            if score > 0.0:
                core_scores[number] = score
        ordered = sorted(
            core_scores,
            key=lambda number: (-core_scores[number], number),
        )
        return ordered[: max(6, min(14, int(minimum_red_coverage)))]

    def ensure_priority_red_core_coverage() -> None:
        priority = priority_red_core()
        if not priority or not result:
            return
        target = max(6, min(len(priority), int(minimum_red_coverage)))
        locked = _locked_positions(result, preserve_top_k=preserve_top_k)
        adjustments = 0
        while adjustments < max(2, min(6, len(result))):
            covered = {int(number) for ticket in result for number in ticket["red"]}
            if len(set(priority[:target]) & covered) >= target:
                break
            missing = [
                number for number in priority[:target]
                if int(number) not in covered
            ]
            if not missing:
                break
            selected_keys_now = {_ticket_key(ticket) for ticket in result}
            best_move: tuple[float, int, dict] | None = None
            current_priority_coverage = len(set(priority[:target]) & covered)
            for position, current_ticket in enumerate(result):
                if position in locked:
                    continue
                context = [ticket for idx, ticket in enumerate(result) if idx != position]
                context_covered = {int(number) for ticket in context for number in ticket["red"]}
                current_ticket_core = len({int(number) for number in current_ticket["red"]} & set(priority[:target]))
                for candidate in candidates:
                    if _ticket_key(candidate) in selected_keys_now:
                        continue
                    candidate_red = {int(number) for number in candidate["red"]}
                    if not candidate_red & set(missing):
                        continue
                    candidate_priority_coverage = len((context_covered | candidate_red) & set(priority[:target]))
                    if candidate_priority_coverage <= current_priority_coverage:
                        continue
                    candidate_core = len(candidate_red & set(priority[:target]))
                    if candidate_core < current_ticket_core and candidate_priority_coverage < target:
                        continue
                    quality_floor = min(
                        replacement_quality_floor(current_ticket),
                        max(
                            duplicate_relief_floor(current_ticket),
                            portfolio_score(current_ticket) * 0.78,
                        ),
                    )
                    if portfolio_score(candidate) < quality_floor:
                        continue
                    max_overlap = max(
                        (len(candidate_red & {int(number) for number in item["red"]}) for item in context),
                        default=0,
                    )
                    gain = candidate_priority_coverage - current_priority_coverage
                    score = (
                        3.20 * gain
                        + 0.75 * candidate_core
                        + replacement_score(candidate, context)
                        + 0.35 * float(candidate.get("method_support_count", 0.0))
                        - 0.18 * max_overlap
                    )
                    if best_move is None or score > best_move[0]:
                        best_move = (float(score), position, candidate)
            if best_move is None:
                break
            _, position, replacement = best_move
            _replace_ticket(result, position, replacement, selected_keys)
            adjustments += 1

    while current_unique_red_sets() < int(minimum_unique_red_sets):
        duplicate_groups: dict[tuple[int, ...], list[int]] = {}
        for index, ticket in enumerate(result):
            duplicate_groups.setdefault(tuple(int(number) for number in ticket["red"]), []).append(index)
        duplicate_positions = [
            position
            for positions in duplicate_groups.values()
            if len(positions) > 1
            for position in positions[1:]
        ]
        duplicate_positions = [
            position for position in duplicate_positions if position not in locked_positions
        ] or duplicate_positions
        if not duplicate_positions:
            break

        changed = False
        for position in duplicate_positions:
            context = [ticket for idx, ticket in enumerate(result) if idx != position]
            current_red_sets = {tuple(int(number) for number in ticket["red"]) for ticket in context}
            alternatives = [
                candidate
                for candidate in candidates
                if _ticket_key(candidate) not in selected_keys
                and tuple(int(number) for number in candidate["red"]) not in current_red_sets
                and portfolio_score(candidate) >= duplicate_relief_floor(result[position])
            ]
            if not alternatives:
                continue
            replacement = max(alternatives, key=lambda candidate: replacement_score(candidate, context))
            _replace_ticket(result, position, replacement, selected_keys)
            changed = True
            break
        if not changed:
            break

    duplicate_cap = int(maximum_duplicate_red_sets) if maximum_duplicate_red_sets is not None else 0
    while duplicate_cap > 0:
        duplicate_groups: dict[tuple[int, ...], list[int]] = {}
        for index, ticket in enumerate(result):
            duplicate_groups.setdefault(tuple(int(number) for number in ticket["red"]), []).append(index)
        crowded_positions = [
            position
            for positions in duplicate_groups.values()
            if len(positions) > duplicate_cap
            for position in positions[duplicate_cap:]
        ]
        crowded_positions = [
            position for position in crowded_positions if position not in locked_positions
        ] or crowded_positions
        if not crowded_positions:
            break
        changed = False
        for position in crowded_positions:
            context = [ticket for idx, ticket in enumerate(result) if idx != position]
            current_red_sets = {tuple(int(number) for number in ticket["red"]) for ticket in context}
            alternatives = [
                candidate
                for candidate in candidates
                if _ticket_key(candidate) not in selected_keys
                and tuple(int(number) for number in candidate["red"]) not in current_red_sets
                and portfolio_score(candidate) >= duplicate_relief_floor(result[position])
            ]
            if not alternatives:
                continue
            replacement = max(
                alternatives,
                key=lambda candidate: (
                    replacement_score(candidate, context),
                    -max((red_jaccard(candidate["red"], item["red"]) for item in context), default=0.0),
                ),
            )
            _replace_ticket(result, position, replacement, selected_keys)
            changed = True
            break
        if not changed:
            break

    coverage_adjustments = 0
    max_coverage_adjustments = max(4, min(40, len(candidates)))
    while (
        current_red_coverage() < int(minimum_red_coverage)
        and coverage_adjustments < max_coverage_adjustments
    ):
        previous_coverage = current_red_coverage()
        previous_signature = tuple(sorted(_ticket_key(ticket) for ticket in result))
        coverage_numbers = {int(number) for ticket in result for number in ticket["red"]}
        contribution = [
            (
                sum(
                    1
                    for number in ticket["red"]
                    if sum(1 for item in result if int(number) in set(item["red"])) == 1
                ),
                portfolio_score(ticket),
                index,
            )
            for index, ticket in enumerate(result)
        ]
        contribution.sort(key=lambda item: (item[0], item[1]))
        unlocked = [item for item in contribution if item[2] not in locked_positions]
        if unlocked:
            contribution = unlocked
        changed = False
        for _, _, position in contribution:
            context = [ticket for idx, ticket in enumerate(result) if idx != position]
            context_coverage = {int(number) for ticket in context for number in ticket["red"]}
            alternatives = [
                candidate
                for candidate in candidates
                if _ticket_key(candidate) not in selected_keys
                and len(set(int(number) for number in candidate["red"]) - context_coverage) > 0
                and portfolio_score(candidate) >= duplicate_relief_floor(result[position])
            ]
            if not alternatives:
                continue
            replacement = max(
                alternatives,
                key=lambda candidate: (
                    len(set(int(number) for number in candidate["red"]) - context_coverage),
                    replacement_score(candidate, context),
                ),
            )
            _replace_ticket(result, position, replacement, selected_keys)
            changed = True
            coverage_adjustments += 1
            break
        if not changed:
            break
        if current_red_coverage() <= previous_coverage:
            current_signature = tuple(sorted(_ticket_key(ticket) for ticket in result))
            if current_signature == previous_signature or int(minimum_red_coverage) - current_red_coverage() > 0:
                break

    ensure_priority_red_core_coverage()
    result = _ensure_minimum_blue_diversity(
        result,
        candidates,
        minimum_unique_blues=minimum_unique_blues,
    )
    result = _ensure_blue_signal_reserve(
        result,
        candidates,
        reserve_count=reserved_blue_signal_count,
        minimum_bonus=reserved_blue_signal_min_bonus,
        preserve_top_k=preserve_top_k,
        minimum_quality_ratio=reserved_blue_signal_min_quality_ratio,
    )
    result = _ensure_blue_focus_alignment(
        result,
        candidates,
        dominant_blue=dominant_blue,
        focus_strength=dominant_blue_focus_strength,
        minimum_repeat_count=dominant_blue_min_repeat_count,
        preserve_top_k=preserve_top_k,
        minimum_quality_ratio=dominant_blue_min_quality_ratio,
        lead_ticket=lead_ticket,
    )
    result = _ensure_minimum_blue_diversity(
        result,
        candidates,
        minimum_unique_blues=minimum_unique_blues,
    )

    duplicate_cap = int(maximum_duplicate_red_sets) if maximum_duplicate_red_sets is not None else 0
    while duplicate_cap > 0:
        duplicate_groups: dict[tuple[int, ...], list[int]] = {}
        for index, ticket in enumerate(result):
            duplicate_groups.setdefault(tuple(int(number) for number in ticket["red"]), []).append(index)
        crowded_positions = [
            position
            for positions in duplicate_groups.values()
            if len(positions) > duplicate_cap
            for position in positions[duplicate_cap:]
        ]
        crowded_positions = [
            position for position in crowded_positions if position not in locked_positions
        ] or crowded_positions
        if not crowded_positions:
            break
        changed = False
        for position in crowded_positions:
            context = [ticket for idx, ticket in enumerate(result) if idx != position]
            current_red_sets = {tuple(int(number) for number in ticket["red"]) for ticket in context}
            alternatives = [
                candidate
                for candidate in candidates
                if _ticket_key(candidate) not in selected_keys
                and tuple(int(number) for number in candidate["red"]) not in current_red_sets
                and portfolio_score(candidate) >= duplicate_relief_floor(result[position])
            ]
            if not alternatives:
                continue
            replacement = max(
                alternatives,
                key=lambda candidate: (
                    replacement_score(candidate, context),
                    _signal_ticket_strength(candidate, lead_ticket),
                    -max((red_jaccard(candidate["red"], item["red"]) for item in context), default=0.0),
                ),
            )
            _replace_ticket(result, position, replacement, selected_keys)
            changed = True
            break
        if not changed:
            break

    if len(result) > 1 and current_unique_red_sets() >= int(minimum_unique_red_sets):
        selected_keys = {_ticket_key(ticket) for ticket in result}
        locked_positions = _locked_positions(
            result,
            preserve_top_k=max(1, int(preserve_top_k)),
        )
        existing_red_set_counts: dict[tuple[int, ...], int] = {}
        for ticket in result:
            key = tuple(int(number) for number in ticket["red"])
            existing_red_set_counts[key] = existing_red_set_counts.get(key, 0) + 1
        blue_hit_candidates = [
            candidate
            for candidate in candidates
            if float(candidate.get("blue_signal_bonus", 0.0)) >= max(0.18, float(reserved_blue_signal_min_bonus) * 0.75)
        ]
        for position, ticket in enumerate(result):
            if position in locked_positions:
                continue
            current_blue_bonus = float(ticket.get("blue_signal_bonus", 0.0))
            if current_blue_bonus >= max(0.28, float(reserved_blue_signal_min_bonus)):
                continue
            context = [item for idx, item in enumerate(result) if idx != position]
            current_strength = _signal_ticket_strength(ticket, lead_ticket)
            alternatives = [
                candidate
                for candidate in blue_hit_candidates
                if _ticket_key(candidate) not in selected_keys
                and (
                    tuple(int(number) for number in candidate["red"]) != tuple(int(number) for number in ticket["red"])
                    or existing_red_set_counts.get(tuple(int(number) for number in candidate["red"]), 0)
                    <= existing_red_set_counts.get(tuple(int(number) for number in ticket["red"]), 0)
                )
                and _signal_ticket_strength(candidate, lead_ticket) >= current_strength * 0.92
            ]
            if not alternatives:
                continue
            replacement = max(
                alternatives,
                key=lambda candidate: (
                    float(candidate.get("blue_signal_bonus", 0.0)),
                    _signal_ticket_strength(candidate, lead_ticket),
                    replacement_score(candidate, context),
                ),
            )
            _replace_ticket(result, position, replacement, selected_keys)

    ensure_priority_red_core_coverage()
    result = _ensure_minimum_blue_diversity(
        result,
        candidates,
        minimum_unique_blues=minimum_unique_blues,
    )

    if lead_ticket and int(lead_overlap_floor) > 0 and len(result) > 1:
        lead_key = _ticket_key(lead_ticket)
        selected_keys = {_ticket_key(ticket) for ticket in result}
        locked_positions = _locked_positions(
            result,
            preserve_top_k=max(1, int(preserve_top_k)),
        )
        lead_red = {int(number) for number in lead_ticket["red"]}

        def lead_overlap(ticket: dict) -> int:
            return len(lead_red & {int(number) for number in ticket["red"]})

        for position, ticket in enumerate(result):
            if position in locked_positions or _ticket_key(ticket) == lead_key:
                continue
            if lead_overlap(ticket) >= int(lead_overlap_floor):
                continue
            context = [item for idx, item in enumerate(result) if idx != position]
            existing_red_sets = {
                tuple(int(number) for number in item["red"])
                for item in context
            }
            alternatives = [
                candidate
                for candidate in candidates
                if _ticket_key(candidate) not in selected_keys
                and tuple(int(number) for number in candidate["red"]) not in existing_red_sets
                and lead_overlap(candidate) >= int(lead_overlap_floor)
                and portfolio_score(candidate) >= replacement_quality_floor(result[position])
            ]
            if not alternatives:
                continue
            replacement = max(
                alternatives,
                key=lambda candidate: (
                    replacement_score(candidate, context),
                    lead_overlap(candidate),
                    -max((red_jaccard(candidate["red"], item["red"]) for item in context), default=0.0),
                ),
            )
            _replace_ticket(result, position, replacement, selected_keys)

    ensure_priority_red_core_coverage()
    return sorted(result, key=final_rank_score, reverse=True)


def _stabilize_signal_method_portfolio(
    selected: list[dict],
    candidates: list[dict],
    *,
    count: int,
    minimum_blue_unique_count: int,
    minimum_unique_red_sets: int | None = None,
    minimum_red_coverage: int | None = None,
    red_consensus_weight: float = 0.0,
    red_consensus_pool_size: int = 20,
    red_core_bonus_weight: float = 0.0,
    red_core_top_k: int = 6,
    blue_focus_bonus_weight: float = 0.0,
) -> list[dict]:
    result = list(selected)
    if not result:
        return result

    effective_count = max(1, int(count))
    if effective_count < 5:
        return result
    red_set_target = max(
        2 if effective_count >= 5 else 1,
        _minimum_red_unique_set_count(
            effective_count,
            minimum_unique_red_sets if minimum_unique_red_sets is not None else (3 if effective_count >= 5 else 2),
        ),
    )
    red_coverage_target = max(
        6 if effective_count >= 5 else 6,
        int(minimum_red_coverage if minimum_red_coverage is not None else (effective_count + 4)),
    )
    blue_target = max(
        3 if effective_count >= 5 else 3 if effective_count >= 3 else 1,
        _minimum_blue_unique_count(effective_count, minimum_blue_unique_count),
    )
    red_support = _build_red_consensus_support(candidates, pool_size=red_consensus_pool_size)
    blue_support, blue_focus_strength = _build_blue_consensus_support(
        candidates,
        pool_size=max(12, min(24, len(candidates))),
    )
    dominant_blue = (
        max(blue_support, key=lambda number: (float(blue_support[number]), -int(number)))
        if blue_support
        else None
    )
    lead_ticket = max(
        result,
        key=lambda ticket: (
            float(ticket.get("quality_score", 0.0)),
            float(ticket.get("red_number_score", ticket.get("number_score", 0.0))),
            float(ticket.get("blue_number_score", 0.0)),
        ),
    )
    return _ensure_signal_portfolio_diversity(
        result,
        candidates,
        minimum_unique_red_sets=red_set_target,
        minimum_red_coverage=red_coverage_target,
        minimum_unique_blues=blue_target,
        red_support=red_support,
        red_consensus_weight=float(red_consensus_weight),
        blue_support=blue_support,
        blue_consensus_weight=float(blue_focus_bonus_weight),
        preserve_top_k=1 if effective_count >= 3 else 0,
        lead_ticket=lead_ticket,
        lead_overlap_floor=0,
        maximum_duplicate_red_sets=1 if effective_count >= 5 else None,
        reserved_blue_signal_count=2 if effective_count >= 5 else 1 if effective_count >= 3 else 0,
        reserved_blue_signal_min_bonus=max(0.16, 0.34 - 0.10 * float(blue_focus_strength)),
        reserved_blue_signal_min_quality_ratio=0.52,
        dominant_blue=dominant_blue,
        dominant_blue_focus_strength=blue_focus_strength,
        dominant_blue_min_repeat_count=2 if effective_count >= 5 and blue_focus_strength >= 0.60 else 0,
        dominant_blue_min_quality_ratio=0.62,
    )


def greedy_optimize(
    candidates: list[dict],
    count: int,
    *,
    scoring_config: dict[str, Any],
    max_red_overlap: int = 4,
    blue_diversity_weight: float = 0.30,
    minimum_blue_unique_count: int = 3,
    concentration_weight: float = 0.0,
    red_consensus_weight: float = 0.0,
    red_consensus_pool_size: int = 20,
    red_core_bonus_weight: float = 0.0,
    red_core_top_k: int = 6,
) -> list[dict]:
    selected: list[dict] = []
    remaining = list(candidates)
    anchor = _signal_anchor_ticket(candidates)
    red_support = _build_red_consensus_support(candidates, pool_size=red_consensus_pool_size)
    while remaining and len(selected) < int(count):
        feasible = [
            candidate for candidate in remaining
            if all(len(set(candidate["red"]) & set(item["red"])) <= max_red_overlap for item in selected)
        ]
        used_blues = {int(ticket["blue"]) for ticket in selected}
        choices = feasible or remaining
        best = max(
            choices,
            key=lambda candidate: (
                selection_score(candidate, selected, scoring_config)
                + 0.30 * float(candidate.get("blue_number_score", 1.0))
                + 0.35 * float(candidate.get("blue_signal_bonus", 0.0))
                + float(scoring_config.get("portfolio_subset_weight", 0.20))
                * marginal_subset_gain(selected, candidate)
                + float(blue_diversity_weight) * float(int(candidate["blue"]) not in used_blues)
                + float(concentration_weight) * _anchor_overlap_bonus(candidate, anchor)
                + float(red_consensus_weight) * _red_consensus_bonus(candidate, red_support)
                + float(red_core_bonus_weight) * _red_core_alignment_bonus(
                    candidate,
                    red_support,
                    top_k=red_core_top_k,
                )
            )
        )
        selected.append(best)
        remaining.remove(best)
    result = _ensure_minimum_blue_diversity(
        selected,
        candidates,
        minimum_unique_blues=_minimum_blue_unique_count(count, minimum_blue_unique_count),
    )
    return _stabilize_signal_method_portfolio(
        result,
        candidates,
        count=count,
        minimum_blue_unique_count=minimum_blue_unique_count,
        red_consensus_weight=red_consensus_weight,
        red_consensus_pool_size=red_consensus_pool_size,
        red_core_bonus_weight=red_core_bonus_weight,
        red_core_top_k=red_core_top_k,
        blue_focus_bonus_weight=max(0.15, float(blue_diversity_weight)),
    )


def _fitness(
    indices: np.ndarray,
    candidates: list[dict],
    diversity_penalty: float,
    blue_diversity_weight: float,
    portfolio_subset_weight: float,
    minimum_blue_unique_count: int,
    concentration_weight: float,
    red_consensus_weight: float,
    red_support: dict[int, float] | None,
    red_core_bonus_weight: float,
    red_core_top_k: int,
) -> float:
    tickets = [candidates[int(index)] for index in indices]
    anchor = _signal_anchor_ticket(candidates)
    quality = float(np.mean([ticket["quality_score"] for ticket in tickets]))
    blue_quality = float(np.mean([float(ticket.get("blue_number_score", 1.0)) for ticket in tickets]))
    anchor_overlap = float(np.mean([
        _anchor_overlap_bonus(ticket, anchor) for ticket in tickets
    ])) if tickets else 0.0
    pairs = [red_jaccard(left["red"], right["red"]) for i, left in enumerate(tickets) for right in tickets[i + 1:]]
    overlap = float(np.mean(pairs)) if pairs else 0.0
    coverage = coverage_summary(tickets)
    coverage_bonus = (float(coverage["red_coverage"]) / 33.0 + float(coverage["blue_coverage"]) / 16.0) / 2.0
    blue_target = max(min(int(minimum_blue_unique_count), len(tickets), 16), 1)
    blue_uniqueness = min(float(coverage["blue_coverage"]) / float(blue_target), 1.0)
    subset_uniqueness = float(np.mean([
        coverage["subsets"][key]["uniqueness_ratio"] for key in ("2", "3")
    ]))
    red_consensus = float(np.mean([
        _red_consensus_bonus(ticket, red_support) for ticket in tickets
    ])) if tickets else 0.0
    red_core_alignment = float(np.mean([
        _red_core_alignment_bonus(ticket, red_support, top_k=red_core_top_k)
        for ticket in tickets
    ])) if tickets else 0.0
    return (
        quality
        + 0.20 * blue_quality
        - diversity_penalty * overlap
        + 0.15 * coverage_bonus
        + blue_diversity_weight * blue_uniqueness
        + float(portfolio_subset_weight) * subset_uniqueness
        + float(concentration_weight) * anchor_overlap
        + float(red_consensus_weight) * red_consensus
        + float(red_core_bonus_weight) * red_core_alignment
    )


def genetic_optimize(
    candidates: list[dict],
    count: int,
    *,
    population_size: int = 24,
    generations: int = 30,
    mutation_rate: float = 0.15,
    diversity_penalty: float = 0.25,
    blue_diversity_weight: float = 0.30,
    portfolio_subset_weight: float = 0.20,
    minimum_blue_unique_count: int = 3,
    concentration_weight: float = 0.0,
    red_consensus_weight: float = 0.0,
    red_consensus_pool_size: int = 20,
    red_core_bonus_weight: float = 0.0,
    red_core_top_k: int = 6,
    random_seed: int = 2026,
) -> list[dict]:
    if int(count) >= len(candidates):
        return _ensure_minimum_blue_diversity(
            list(candidates),
            candidates,
            minimum_unique_blues=_minimum_blue_unique_count(count, minimum_blue_unique_count),
        )
    rng = np.random.default_rng(int(random_seed))
    candidate_count = len(candidates)
    red_support = _build_red_consensus_support(candidates, pool_size=red_consensus_pool_size)
    population: list[np.ndarray] = [np.arange(int(count), dtype=int)]
    while len(population) < int(population_size):
        population.append(np.sort(rng.choice(candidate_count, size=int(count), replace=False)))

    for _ in range(int(generations)):
        ranked = sorted(
            population,
            key=lambda item: _fitness(
                item,
                candidates,
                diversity_penalty,
                blue_diversity_weight,
                portfolio_subset_weight,
                _minimum_blue_unique_count(count, minimum_blue_unique_count),
                concentration_weight,
                red_consensus_weight,
                red_support,
                red_core_bonus_weight,
                red_core_top_k,
            ),
            reverse=True,
        )
        elite_count = max(2, len(ranked) // 4)
        next_population = [item.copy() for item in ranked[:elite_count]]
        while len(next_population) < int(population_size):
            left, right = rng.choice(elite_count, size=2, replace=False)
            pool = np.unique(np.concatenate([ranked[int(left)], ranked[int(right)]]))
            if len(pool) >= int(count):
                child = rng.choice(pool, size=int(count), replace=False)
            else:
                available = np.setdiff1d(np.arange(candidate_count), pool)
                child = np.concatenate([pool, rng.choice(available, size=int(count) - len(pool), replace=False)])
            if rng.random() < float(mutation_rate):
                replace_at = int(rng.integers(0, int(count)))
                available = np.setdiff1d(np.arange(candidate_count), child)
                child[replace_at] = int(rng.choice(available))
            next_population.append(np.sort(np.unique(child)))
        population = next_population

    best = max(
        population,
        key=lambda item: _fitness(
            item,
            candidates,
            diversity_penalty,
            blue_diversity_weight,
            portfolio_subset_weight,
            _minimum_blue_unique_count(count, minimum_blue_unique_count),
            concentration_weight,
            red_consensus_weight,
            red_support,
            red_core_bonus_weight,
            red_core_top_k,
        ),
    )
    result = _ensure_minimum_blue_diversity(
        [candidates[int(index)] for index in best],
        candidates,
        minimum_unique_blues=_minimum_blue_unique_count(count, minimum_blue_unique_count),
    )
    return _stabilize_signal_method_portfolio(
        result,
        candidates,
        count=count,
        minimum_blue_unique_count=minimum_blue_unique_count,
        red_consensus_weight=red_consensus_weight,
        red_consensus_pool_size=red_consensus_pool_size,
        red_core_bonus_weight=red_core_bonus_weight,
        red_core_top_k=red_core_top_k,
        blue_focus_bonus_weight=max(0.15, float(blue_diversity_weight)),
    )


def _candidate_objectives(candidate: dict, selected: list[dict]) -> tuple[float, float, float, float]:
    red_quality = float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
    blue_quality = float(candidate.get("blue_number_score", 1.0)) + 0.65 * float(candidate.get("blue_signal_bonus", 0.0))
    structure = float(candidate.get("structure_score", 0.0))
    coverage = float(marginal_subset_gain(selected, candidate))
    return red_quality, blue_quality, structure, coverage


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(a >= b - 1e-12 for a, b in zip(left, right)) and any(
        a > b + 1e-12 for a, b in zip(left, right)
    )


def _pareto_front(candidates: list[dict], selected: list[dict]) -> list[dict]:
    objectives = [_candidate_objectives(candidate, selected) for candidate in candidates]
    front: list[dict] = []
    for index, candidate in enumerate(candidates):
        if any(
            other_index != index and _dominates(objectives[other_index], objectives[index])
            for other_index in range(len(candidates))
        ):
            continue
        front.append(candidate)
    return front or list(candidates)


def pareto_optimize(
    candidates: list[dict],
    count: int = 3,
    *,
    scoring_config: dict[str, Any],
    max_red_overlap: int = 4,
    blue_diversity_weight: float = 0.30,
    minimum_blue_unique_count: int = 3,
    red_consensus_weight: float = 0.0,
    red_consensus_pool_size: int = 20,
    red_core_bonus_weight: float = 0.0,
    red_core_top_k: int = 6,
) -> list[dict]:
    """Pareto-style selector for Top-K oriented portfolios.

    This keeps the implementation dependency-free but changes the portfolio
    objective from a single blended score to a non-dominated front over red
    strength, blue strength, structure typicality and incremental coverage.
    """
    if not candidates or int(count) <= 0:
        return []
    selected: list[dict] = []
    remaining = list(candidates)
    red_support = _build_red_consensus_support(candidates, pool_size=red_consensus_pool_size)
    while remaining and len(selected) < int(count):
        feasible = [
            candidate for candidate in remaining
            if all(len(set(candidate["red"]) & set(item["red"])) <= max_red_overlap for item in selected)
        ] or remaining
        front = _pareto_front(feasible, selected)
        used_blues = {int(ticket["blue"]) for ticket in selected}
        best = max(
            front,
            key=lambda candidate: (
                0.50 * float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
                + 0.24 * float(candidate.get("blue_number_score", 1.0))
                + 0.22 * float(candidate.get("blue_signal_bonus", 0.0))
                + 0.20 * float(candidate.get("structure_score", 0.0))
                + 0.30 * marginal_subset_gain(selected, candidate)
                + float(blue_diversity_weight) * float(int(candidate["blue"]) not in used_blues)
                + float(red_consensus_weight) * _red_consensus_bonus(candidate, red_support)
                + float(red_core_bonus_weight) * _red_core_alignment_bonus(
                    candidate,
                    red_support,
                    top_k=red_core_top_k,
                )
                - float(scoring_config.get("diversity_penalty", 0.25))
                * max((red_jaccard(candidate["red"], item["red"]) for item in selected), default=0.0)
            ),
        )
        selected.append(best)
        remaining.remove(best)
    result = _ensure_minimum_blue_diversity(
        selected,
        candidates,
        minimum_unique_blues=_minimum_blue_unique_count(len(selected), minimum_blue_unique_count),
    )
    return _stabilize_signal_method_portfolio(
        result,
        candidates,
        count=count,
        minimum_blue_unique_count=minimum_blue_unique_count,
        red_consensus_weight=red_consensus_weight,
        red_consensus_pool_size=red_consensus_pool_size,
        red_core_bonus_weight=red_core_bonus_weight,
        red_core_top_k=red_core_top_k,
        blue_focus_bonus_weight=max(0.15, float(blue_diversity_weight)),
    )


def _history_draws(history_frame: Any, window: int) -> tuple[list[set[int]], np.ndarray]:
    """Return recent draws as red sets and blue array.

    The optimizer deliberately consumes only the already-known historical
    frame passed by the caller.  In walk-forward validation that frame is
    left-truncated before the target period, so this helper cannot leak the
    period being evaluated.
    """
    if history_frame is None or not hasattr(history_frame, "tail"):
        return [], np.asarray([], dtype=int)
    if len(history_frame) <= 0:
        return [], np.asarray([], dtype=int)
    recent = history_frame.tail(max(1, int(window)))
    red_draws: list[set[int]] = []
    blue_draws: list[int] = []
    for _index, row in recent.iterrows():
        try:
            red_draws.append({int(row[f"r{i}"]) for i in range(1, 7)})
            blue_draws.append(int(row["blue"]))
        except Exception:
            continue
    return red_draws, np.asarray(blue_draws, dtype=int)


def _historical_hit_matrices(
    candidates: list[dict],
    red_draws: list[set[int]],
    blue_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates or not red_draws:
        return (
            np.zeros((len(candidates), 0), dtype=np.int16),
            np.zeros((len(candidates), 0), dtype=np.int8),
        )
    red_hits = np.zeros((len(candidates), len(red_draws)), dtype=np.int16)
    blue_hits = np.zeros((len(candidates), len(red_draws)), dtype=np.int8)
    for row_index, candidate in enumerate(candidates):
        red = {int(number) for number in candidate["red"]}
        blue = int(candidate["blue"])
        red_hits[row_index, :] = [len(red & actual_red) for actual_red in red_draws]
        blue_hits[row_index, :] = (blue_draws == blue).astype(np.int8)
    return red_hits, blue_hits


def _portfolio_history_score(
    tickets: list[dict],
    red_draws: list[set[int]],
    blue_draws: np.ndarray,
) -> float:
    if not tickets or not red_draws:
        return 0.0
    red_max = []
    blue_cover = []
    for draw_index, actual_red in enumerate(red_draws):
        red_max.append(max(
            len(set(int(number) for number in ticket["red"]) & actual_red)
            for ticket in tickets
        ))
        blue_cover.append(max(
            int(int(ticket["blue"]) == int(blue_draws[draw_index]))
            for ticket in tickets
        ))
    red_values = np.asarray(red_max, dtype=float)
    blue_values = np.asarray(blue_cover, dtype=float)
    return float(
        0.40 * (float(np.mean(red_values)) / 6.0)
        + 0.32 * float(np.mean(red_values >= 3))
        + 0.12 * float(np.mean(red_values >= 4))
        + 0.16 * float(np.mean(blue_values))
    )


def _optuna_allowed(config: dict[str, Any]) -> bool:
    if not bool(config.get("optuna_enabled", False)):
        return False
    if bool(config.get("_walk_forward_context", False)) and not bool(config.get("optuna_apply_during_backtest", False)):
        return False
    return True


def _optuna_cache_key(kind: str, candidates: list[dict], history_frame: Any, config: dict[str, Any]) -> str:
    latest_period = ""
    history_size = 0
    if history_frame is not None and getattr(history_frame, "empty", True) is False:
        history_size = int(len(history_frame))
        latest_period = str(history_frame.iloc[-1].get("period", ""))
    head = [
        (
            tuple(int(number) for number in candidate.get("red", []) or []),
            int(candidate.get("blue", 0) or 0),
        )
        for candidate in list(candidates or [])[: int(config.get("optuna_candidate_cap", 400))]
    ]
    digest = hashlib.sha256(repr(head).encode("utf-8")).hexdigest()[:16]
    return ":".join([
        str(kind),
        str(history_size),
        latest_period,
        str(config.get("optuna_objective_window", "")),
        str(config.get("optuna_trials", "")),
        digest,
    ])


def _load_optuna_cache(config: dict[str, Any], key: str) -> dict[str, float] | None:
    path = Path(str(config.get("optuna_cache_path", "logs/optuna_tuning.json")))
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get(key)
        if not isinstance(value, dict):
            return None
        return {str(name): float(raw) for name, raw in value.items()}
    except Exception:
        return None


def _save_optuna_cache(config: dict[str, Any], key: str, value: dict[str, float]) -> None:
    path = Path(str(config.get("optuna_cache_path", "logs/optuna_tuning.json")))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        payload[str(key)] = {str(name): float(raw) for name, raw in value.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as error:
        logging.warning("Optuna调参缓存写入失败 | 事件=OPTUNA_CACHE_WRITE_FAILED | 错误=%s", error)


def _optuna_tune_historical_hit_parameters(
    candidates: list[dict],
    history_frame: Any,
    config: dict[str, Any],
    scoring_config: dict[str, Any],
    random_seed: int,
) -> dict[str, float]:
    if not _optuna_allowed(config):
        return {}
    red_draws, blue_draws = _history_draws(
        history_frame,
        int(config.get("optuna_objective_window", config.get("historical_hit_window", 200))),
    )
    if len(red_draws) < 30:
        return {}
    try:
        import optuna
    except Exception:
        return {}

    objective_count = min(
        int(config.get("optuna_candidate_cap", 400)),
        len(candidates),
    )
    objective_candidates = list(candidates[:objective_count])
    cache_key = _optuna_cache_key("historical_hit", objective_candidates, history_frame, config)
    cached = _load_optuna_cache(config, cache_key)
    if cached is not None:
        return cached
    trials = max(1, int(config.get("optuna_trials", 60)))
    timeout = config.get("optuna_timeout_seconds")
    timeout_value = int(timeout) if timeout is not None else None

    def objective(trial: Any) -> float:
        history_weight = trial.suggest_float("history_weight", 0.00, 0.45)
        current_signal_weight = trial.suggest_float("current_signal_weight", 0.25, 0.65)
        coverage_weight = trial.suggest_float("coverage_weight", 0.20, 0.60)
        max_overlap = trial.suggest_int(
            "max_red_overlap",
            2,
            max(2, int(config.get("signal_max_red_overlap", config.get("max_red_overlap", 4)))),
        )
        selected = historical_hit_optimize(
            objective_candidates,
            min(5, len(objective_candidates)),
            history_frame=history_frame,
            window=int(config.get("optuna_objective_window", 200)),
            scoring_config=scoring_config,
            max_red_overlap=max_overlap,
            minimum_blue_unique_count=int(config.get("signal_minimum_blue_unique_count", 5)),
            history_weight=history_weight,
            current_signal_weight=current_signal_weight,
            coverage_weight=coverage_weight,
        )
        return _portfolio_history_score(selected, red_draws, blue_draws)

    sampler = optuna.samplers.TPESampler(seed=int(random_seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout_value,
        show_progress_bar=False,
        catch=(Exception,),
    )
    result = {
        "history_weight": float(study.best_params.get("history_weight", config.get("historical_hit_weight", 0.25))),
        "current_signal_weight": float(study.best_params.get("current_signal_weight", config.get("historical_hit_current_signal_weight", 0.42))),
        "coverage_weight": float(study.best_params.get("coverage_weight", config.get("historical_hit_coverage_weight", 0.33))),
        "max_red_overlap": float(study.best_params.get(
            "max_red_overlap",
            config.get("signal_max_red_overlap", config.get("max_red_overlap", 4)),
        )),
    }
    _save_optuna_cache(config, cache_key, result)
    return result


def _optuna_tune_red_ge3_parameters(
    candidates: list[dict],
    history_frame: Any,
    config: dict[str, Any],
    scoring_config: dict[str, Any],
    random_seed: int,
) -> dict[str, float]:
    if not _optuna_allowed(config):
        return {}
    red_draws, blue_draws = _history_draws(
        history_frame,
        int(config.get("optuna_objective_window", config.get("red_ge3_window", 300))),
    )
    if len(red_draws) < 30:
        return {}
    try:
        import optuna
    except Exception:
        return {}

    objective_count = min(int(config.get("optuna_candidate_cap", 400)), len(candidates))
    objective_candidates = list(candidates[:objective_count])
    if len(objective_candidates) < 5:
        return {}
    cache_key = _optuna_cache_key("red_ge3", objective_candidates, history_frame, config)
    cached = _load_optuna_cache(config, cache_key)
    if cached is not None:
        return cached
    trials = max(1, int(config.get("optuna_trials", 60)))
    timeout = config.get("optuna_timeout_seconds")
    timeout_value = int(timeout) if timeout is not None else None

    def objective(trial: Any) -> float:
        any3_weight = trial.suggest_float("any3_weight", 0.65, 1.35)
        any4_weight = trial.suggest_float("any4_weight", 0.05, 0.40)
        best_hit_weight = trial.suggest_float("best_hit_weight", 0.08, 0.42)
        quality_weight = trial.suggest_float("quality_weight", 0.04, 0.30)
        coverage_weight = trial.suggest_float("coverage_weight", 0.04, 0.28)
        max_overlap = trial.suggest_int(
            "max_red_overlap",
            3,
            max(3, int(config.get("red_ge3_max_red_overlap", 5))),
        )
        selected = red_ge3_objective_optimize(
            objective_candidates,
            min(5, len(objective_candidates)),
            history_frame=history_frame,
            window=int(config.get("optuna_objective_window", config.get("red_ge3_window", 300))),
            scoring_config=scoring_config,
            max_red_overlap=max_overlap,
            minimum_blue_unique_count=int(config.get("signal_minimum_blue_unique_count", 5)),
            candidate_cap=objective_count,
            any3_weight=any3_weight,
            any4_weight=any4_weight,
            best_hit_weight=best_hit_weight,
            quality_weight=quality_weight,
            coverage_weight=coverage_weight,
        )
        return _portfolio_history_score(selected, red_draws, blue_draws)

    sampler = optuna.samplers.TPESampler(seed=int(random_seed) + 17)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=trials,
        timeout=timeout_value,
        show_progress_bar=False,
        catch=(Exception,),
    )
    result = {
        "any3_weight": float(study.best_params.get("any3_weight", config.get("red_ge3_any3_weight", 1.0))),
        "any4_weight": float(study.best_params.get("any4_weight", config.get("red_ge3_any4_weight", 0.18))),
        "best_hit_weight": float(study.best_params.get("best_hit_weight", config.get("red_ge3_best_hit_weight", 0.22))),
        "quality_weight": float(study.best_params.get("quality_weight", config.get("red_ge3_quality_weight", 0.14))),
        "coverage_weight": float(study.best_params.get("coverage_weight", config.get("red_ge3_coverage_weight", 0.10))),
        "max_red_overlap": float(study.best_params.get("max_red_overlap", config.get("red_ge3_max_red_overlap", 5))),
    }
    _save_optuna_cache(config, cache_key, result)
    return result


def historical_hit_optimize(
    candidates: list[dict],
    count: int,
    *,
    history_frame: Any = None,
    window: int = 300,
    scoring_config: dict[str, Any] | None = None,
    max_red_overlap: int = 4,
    minimum_blue_unique_count: int = 3,
    history_weight: float = 0.25,
    current_signal_weight: float = 0.42,
    coverage_weight: float = 0.33,
) -> list[dict]:
    """Select tickets by replaying candidate portfolios against recent history.

    This is a portfolio-level objective, not a per-ticket ranker.  Each slot is
    chosen by how much it improves the recent-history portfolio score: best
    red hit in any row, probability of at least one row reaching 3/4 red hits,
    blue coverage, plus a smaller current-model quality term.  It directly
    attacks the common failure mode where useful numbers are spread across
    several rows without forming a strong portfolio.
    """
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if not isinstance(candidates, list):
        raise ValueError("candidates 必须是列表")
    if not candidates:
        return []
    effective_count = min(int(count), len(candidates))
    scoring = scoring_config or {}
    red_draws, blue_draws = _history_draws(history_frame, window)
    if not red_draws:
        return pareto_optimize(
            candidates,
            effective_count,
            scoring_config=scoring,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=minimum_blue_unique_count,
        )

    red_hits, blue_hits = _historical_hit_matrices(candidates, red_draws, blue_draws)
    candidate_quality = np.asarray([
        float(candidate.get("quality_score", 0.0))
        + 0.24 * float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
        + 0.18 * float(candidate.get("blue_number_score", 1.0))
        + 0.16 * float(candidate.get("blue_signal_bonus", 0.0))
        + 0.10 * float(candidate.get("structure_score", 0.0))
        for candidate in candidates
    ], dtype=float)
    if candidate_quality.size:
        min_q = float(np.min(candidate_quality))
        span_q = max(float(np.max(candidate_quality) - min_q), 1e-9)
        candidate_quality = (candidate_quality - min_q) / span_q

    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))
    current_red_max = np.zeros(len(red_draws), dtype=np.int16)
    current_blue_cover = np.zeros(len(red_draws), dtype=np.int8)

    def portfolio_replay_score(projected_red: np.ndarray, projected_blue: np.ndarray) -> float:
        avg_best_red = float(np.mean(projected_red)) / 6.0
        any_ge3 = float(np.mean(projected_red >= 3))
        any_ge4 = float(np.mean(projected_red >= 4))
        blue_cover = float(np.mean(projected_blue))
        return float(
            0.40 * avg_best_red
            + 0.32 * any_ge3
            + 0.12 * any_ge4
            + 0.16 * blue_cover
        )

    while remaining and len(selected_indices) < effective_count:
        used_blues = {int(candidates[index]["blue"]) for index in selected_indices}
        selected_tickets = [candidates[index] for index in selected_indices]
        feasible = [
            index for index in remaining
            if all(
                len(set(candidates[index]["red"]) & set(ticket["red"])) <= int(max_red_overlap)
                for ticket in selected_tickets
            )
        ] or list(remaining)

        def slot_score(index: int) -> tuple[float, float, float]:
            candidate = candidates[index]
            projected_red = np.maximum(current_red_max, red_hits[index])
            projected_blue = np.maximum(current_blue_cover, blue_hits[index])
            replay = portfolio_replay_score(projected_red, projected_blue)
            novel_red = len(
                set(int(number) for number in candidate["red"])
                - {
                    int(number)
                    for chosen in selected_tickets
                    for number in chosen["red"]
                }
            ) / 6.0
            blue_gain = float(int(candidate["blue"]) not in used_blues)
            subset_gain = marginal_subset_gain(selected_tickets, candidate)
            overlap_penalty = max(
                (red_jaccard(candidate["red"], ticket["red"]) for ticket in selected_tickets),
                default=0.0,
            )
            coverage = 0.45 * novel_red + 0.35 * blue_gain + 0.20 * subset_gain
            score = (
                float(history_weight) * replay
                + float(current_signal_weight) * float(candidate_quality[index])
                + float(coverage_weight) * coverage
                - float(scoring.get("diversity_penalty", 0.18)) * 0.35 * overlap_penalty
            )
            return score, replay, float(candidate_quality[index])

        best_index = max(feasible, key=slot_score)
        selected_indices.append(best_index)
        remaining.remove(best_index)
        current_red_max = np.maximum(current_red_max, red_hits[best_index])
        current_blue_cover = np.maximum(current_blue_cover, blue_hits[best_index])

    result = [
        {
            **candidates[index],
            "selection_role": "historical_hit_replay",
            "historical_replay_window": int(min(len(red_draws), int(window))),
        }
        for index in selected_indices
    ]
    return _ensure_minimum_blue_diversity(
        result,
        candidates,
        minimum_unique_blues=_minimum_blue_unique_count(
            effective_count,
            minimum_blue_unique_count,
        ),
    )


def red_ge3_objective_optimize(
    candidates: list[dict],
    count: int,
    *,
    history_frame: Any = None,
    window: int = 300,
    scoring_config: dict[str, Any] | None = None,
    max_red_overlap: int = 5,
    minimum_blue_unique_count: int = 3,
    candidate_cap: int = 500,
    any3_weight: float = 1.00,
    any4_weight: float = 0.18,
    best_hit_weight: float = 0.22,
    quality_weight: float = 0.14,
    coverage_weight: float = 0.10,
) -> list[dict]:
    """Select a portfolio for the explicit red-ball objective: any row >= 3 hits.

    The older greedy/genetic/pareto selectors optimize row quality, coverage and
    diversity.  Those are useful proxies, but they are not the buying objective
    the dashboard evaluates most often.  This selector replays every candidate
    against recent completed draws and greedily maximizes the marginal gain of:
    at least one selected row hitting >=3 reds, then >=4 reds, then the best row
    red-hit average.  Current model quality is retained only as a tie-breaker so
    the optimizer does not spread the same useful red numbers across unrelated
    rows merely to satisfy diversity.
    """
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if not isinstance(candidates, list):
        raise ValueError("candidates 必须是列表")
    if not candidates:
        return []
    scoring = scoring_config or {}
    red_draws, blue_draws = _history_draws(history_frame, window)
    if not red_draws:
        return historical_hit_optimize(
            candidates,
            count,
            history_frame=history_frame,
            window=window,
            scoring_config=scoring,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=minimum_blue_unique_count,
        )

    cap = max(int(count), min(int(candidate_cap), len(candidates)))
    pool = list(candidates[:cap])
    effective_count = min(int(count), len(pool))
    red_hits, blue_hits = _historical_hit_matrices(pool, red_draws, blue_draws)
    quality_values = np.asarray([
        float(candidate.get("quality_score", 0.0))
        + 0.30 * float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
        + 0.10 * float(candidate.get("structure_score", 0.0))
        for candidate in pool
    ], dtype=float)
    if quality_values.size:
        min_q = float(np.min(quality_values))
        span_q = max(float(np.max(quality_values) - min_q), 1e-9)
        quality_values = (quality_values - min_q) / span_q

    selected_indices: list[int] = []
    remaining = set(range(len(pool)))
    current_red_max = np.zeros(len(red_draws), dtype=np.int16)
    current_blue_cover = np.zeros(len(red_draws), dtype=np.int8)

    def replay_metrics(red_max: np.ndarray, blue_cover: np.ndarray) -> tuple[float, float, float, float]:
        return (
            float(np.mean(red_max >= 3)),
            float(np.mean(red_max >= 4)),
            float(np.mean(red_max)) / 6.0,
            float(np.mean(blue_cover)),
        )

    while remaining and len(selected_indices) < effective_count:
        selected_tickets = [pool[index] for index in selected_indices]
        used_blues = {int(ticket["blue"]) for ticket in selected_tickets}
        feasible = [
            index for index in remaining
            if all(
                len(set(pool[index]["red"]) & set(ticket["red"])) <= int(max_red_overlap)
                for ticket in selected_tickets
            )
        ] or list(remaining)
        base_any3, base_any4, base_best, base_blue = replay_metrics(current_red_max, current_blue_cover)

        def slot_score(index: int) -> tuple[float, float, float, float]:
            candidate = pool[index]
            projected_red = np.maximum(current_red_max, red_hits[index])
            projected_blue = np.maximum(current_blue_cover, blue_hits[index])
            any3, any4, best_avg, blue_cover = replay_metrics(projected_red, projected_blue)
            marginal_replay = (
                float(any3_weight) * (any3 - base_any3)
                + float(any4_weight) * (any4 - base_any4)
                + float(best_hit_weight) * (best_avg - base_best)
                + 0.04 * (blue_cover - base_blue)
            )
            novel_red = len(
                set(int(number) for number in candidate["red"])
                - {
                    int(number)
                    for chosen in selected_tickets
                    for number in chosen["red"]
                }
            ) / 6.0
            subset_gain = marginal_subset_gain(selected_tickets, candidate)
            blue_gain = float(int(candidate["blue"]) not in used_blues)
            overlap_penalty = max(
                (red_jaccard(candidate["red"], ticket["red"]) for ticket in selected_tickets),
                default=0.0,
            )
            coverage = 0.45 * novel_red + 0.35 * subset_gain + 0.20 * blue_gain
            score = (
                marginal_replay
                + float(quality_weight) * float(quality_values[index])
                + float(coverage_weight) * coverage
                - float(scoring.get("diversity_penalty", 0.18)) * 0.22 * overlap_penalty
            )
            return score, any3, any4, float(quality_values[index])

        best_index = max(feasible, key=slot_score)
        selected_indices.append(best_index)
        remaining.remove(best_index)
        current_red_max = np.maximum(current_red_max, red_hits[best_index])
        current_blue_cover = np.maximum(current_blue_cover, blue_hits[best_index])

    result = [
        {
            **pool[index],
            "selection_role": "red_ge3_replay_objective",
            "red_ge3_replay_window": int(min(len(red_draws), int(window))),
            "red_ge3_objective": "maximize_any_ticket_red_hits_ge3",
        }
        for index in selected_indices
    ]
    return _ensure_minimum_blue_diversity(
        result,
        candidates,
        minimum_unique_blues=_minimum_blue_unique_count(
            effective_count,
            minimum_blue_unique_count,
        ),
    )


def global_coverage_optimize(
    candidates: list[dict],
    count: int,
    *,
    scoring_config: dict[str, Any] | None = None,
    max_red_overlap: int = 4,
    minimum_blue_unique_count: int = 5,
    candidate_cap: int = 300,
    time_limit_seconds: int = 20,
) -> list[dict]:
    """Solve the portfolio coverage problem as a binary integer program.

    The model is intentionally bounded to the top candidate_cap candidates so
    that the web flow remains interactive.  Within that bounded pool, CBC finds
    the global optimum for: pick count rows, cover high-support red numbers,
    rotate blue numbers, keep current candidate quality, and reject excessive
    pair overlap.
    """
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if not isinstance(candidates, list):
        raise ValueError("candidates 必须是列表")
    if not candidates:
        return []
    try:
        import pulp
    except Exception:
        return rotational_coverage_optimize(
            candidates,
            count,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=minimum_blue_unique_count,
        )

    scoring = scoring_config or {}
    pool = list(candidates[: max(1, min(int(candidate_cap), len(candidates)))])
    effective_count = min(int(count), len(pool))
    quality_values = np.asarray([
        float(candidate.get("quality_score", 0.0))
        + 0.25 * float(candidate.get("blue_signal_bonus", 0.0))
        for candidate in pool
    ], dtype=float)
    min_quality = float(np.min(quality_values))
    quality_span = max(float(np.max(quality_values) - min_quality), 1e-9)
    normalized_quality = (quality_values - min_quality) / quality_span

    red_support = {number: 0.0 for number in range(1, 34)}
    blue_support = {number: 0.0 for number in range(1, 17)}
    rank_weights = np.linspace(1.0, 0.35, len(pool))
    for idx, candidate in enumerate(pool):
        weight = float(rank_weights[idx])
        for number in candidate["red"]:
            red_support[int(number)] += weight
        blue_support[int(candidate["blue"])] += weight
    max_red_support = max(red_support.values(), default=1.0) or 1.0
    max_blue_support = max(blue_support.values(), default=1.0) or 1.0
    red_support = {number: value / max_red_support for number, value in red_support.items()}
    blue_support = {number: value / max_blue_support for number, value in blue_support.items()}

    # Unique name avoids colliding CBC temp artifacts across concurrent dashboard runs.
    problem = pulp.LpProblem(f"tsys_gc_{os.getpid()}_{uuid.uuid4().hex[:10]}", pulp.LpMaximize)
    x = {idx: pulp.LpVariable(f"x_{idx}", cat="Binary") for idx in range(len(pool))}
    y = {number: pulp.LpVariable(f"red_{number}", cat="Binary") for number in range(1, 34)}
    z = {number: pulp.LpVariable(f"blue_{number}", cat="Binary") for number in range(1, 17)}

    problem += pulp.lpSum(x.values()) == effective_count
    for number in range(1, 34):
        carriers = [x[idx] for idx, candidate in enumerate(pool) if number in set(candidate["red"])]
        problem += y[number] <= pulp.lpSum(carriers) if carriers else y[number] == 0
    for number in range(1, 17):
        carriers = [x[idx] for idx, candidate in enumerate(pool) if int(candidate["blue"]) == number]
        problem += z[number] <= pulp.lpSum(carriers) if carriers else z[number] == 0

    for left in range(len(pool)):
        left_red = set(pool[left]["red"])
        for right in range(left + 1, len(pool)):
            if len(left_red & set(pool[right]["red"])) > int(max_red_overlap):
                problem += x[left] + x[right] <= 1

    target_blue = _minimum_blue_unique_count(effective_count, minimum_blue_unique_count)
    if target_blue > 1:
        problem += pulp.lpSum(z.values()) >= min(target_blue, effective_count)

    quality_weight = float(scoring.get("global_coverage_quality_weight", 0.32))
    red_weight = float(scoring.get("global_coverage_red_weight", 1.0))
    blue_weight = float(scoring.get("global_coverage_blue_weight", 0.40))
    problem += (
        quality_weight * pulp.lpSum(float(normalized_quality[idx]) * x[idx] for idx in range(len(pool)))
        + red_weight * pulp.lpSum(float(red_support[number]) * y[number] for number in range(1, 34))
        + blue_weight * pulp.lpSum(float(blue_support[number]) * z[number] for number in range(1, 17))
    )

    def _fallback(reason: str) -> list[dict]:
        logging.warning(
            "全局覆盖整数规划不可用，回退轮换覆盖 | 事件=GLOBAL_COVERAGE_CBC_FALLBACK | 原因=%s | 候选=%s | 选取=%s",
            reason,
            len(pool),
            effective_count,
        )
        return rotational_coverage_optimize(
            candidates,
            count,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=minimum_blue_unique_count,
        )

    try:
        # PULP_CBC_CMD bundles its own cbc.exe; do not pass path= (raises on pulp>=3).
        solver = pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=max(1, int(time_limit_seconds)),
            mip=True,
        )
        status = problem.solve(solver)
    except Exception as error:  # noqa: BLE001 - CBC/OS failures must not abort walk-forward
        return _fallback(f"{type(error).__name__}: {error}")
    solver_status = str(pulp.LpStatus.get(status, status))
    if solver_status not in {"Optimal", "Not Solved", "Undefined"}:
        return _fallback(f"status={solver_status}")
    chosen = [
        idx for idx in range(len(pool))
        if float(x[idx].value() or 0.0) >= 0.5
    ]
    if len(chosen) != effective_count:
        return rotational_coverage_optimize(
            candidates,
            count,
            max_red_overlap=max_red_overlap,
            minimum_blue_unique_count=minimum_blue_unique_count,
        )
    return [
        {
            **pool[idx],
            "selection_role": "global_integer_coverage",
            "integer_program_status": solver_status,
            "integer_program_candidate_cap": int(len(pool)),
        }
        for idx in chosen
    ]


def rotational_coverage_optimize(
    candidates: list[dict],
    count: int,
    *,
    max_red_overlap: int = 4,
    minimum_blue_unique_count: int | None = None,
    random_seed: int = 2026,
) -> list[dict]:
    """Build a deterministic low-overlap covering matrix from candidate rows.

    This is a portfolio fallback for a uniform/no-signal probability surface.
    It maximizes red-number and 2/3-subset coverage while rotating distinct blue
    numbers.  It does not create a probability signal and must not be used to
    claim that one individual row is more likely than another.
    """
    if not candidates or int(count) <= 0:
        return []
    unique: dict[tuple[tuple[int, ...], int], dict] = {}
    for candidate in candidates:
        unique.setdefault(_ticket_key(candidate), candidate)
    remaining = list(unique.values())
    selected: list[dict] = []
    red_incidence = {number: 0 for number in range(1, 34)}
    while remaining and len(selected) < min(int(count), len(remaining)):
        used_blues = {int(ticket["blue"]) for ticket in selected}
        feasible = [
            candidate for candidate in remaining
            if all(
                len(set(candidate["red"]) & set(ticket["red"])) <= int(max_red_overlap)
                for ticket in selected
            )
        ]
        choices = feasible or remaining

        def rotation_score(candidate: dict) -> tuple[float, float, int]:
            red = tuple(int(number) for number in candidate["red"])
            novel_red = sum(red_incidence[number] == 0 for number in red) / 6.0
            incidence_balance = 1.0 / (1.0 + sum(red_incidence[number] for number in red))
            subset_gain = marginal_subset_gain(selected, candidate)
            blue_gain = float(int(candidate["blue"]) not in used_blues)
            max_overlap = max(
                (red_jaccard(red, ticket["red"]) for ticket in selected),
                default=0.0,
            )
            score = (
                0.35 * novel_red
                + 0.30 * subset_gain
                + 0.25 * blue_gain
                + 0.10 * incidence_balance
                - 0.15 * max_overlap
            )
            # A seeded content hash is reproducible but does not prefer smaller
            # numbers when every probability and coverage score is tied.
            payload = (
                f"{int(random_seed)}|{len(selected)}|"
                f"{','.join(str(number) for number in red)}|{int(candidate['blue'])}"
            ).encode("ascii")
            unbiased_tie = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            return score, novel_red, unbiased_tie

        best = max(choices, key=rotation_score)
        selected.append({
            **best,
            "rotation_matrix_row": len(selected) + 1,
            "selection_role": "no_signal_rotational_coverage",
        })
        for number in best["red"]:
            red_incidence[int(number)] += 1
        remaining.remove(best)
    if minimum_blue_unique_count is not None:
        selected = _ensure_minimum_blue_diversity(
            selected,
            candidates,
            minimum_unique_blues=_minimum_blue_unique_count(
                count,
                int(minimum_blue_unique_count),
            ),
        )
    return selected


def adverse_guard_coverage_optimize(
    candidates: list[dict],
    method_results: dict[str, list[dict]],
    count: int,
    *,
    max_red_overlap: int = 4,
    minimum_red_coverage: int | None = None,
    minimum_blue_unique_count: int | None = None,
    red_signal_weight: float = 0.0,
    blue_signal_weight: float = 0.55,
    method_anchor_count: int = 0,
    method_anchor_top_k_per_method: int = 2,
    random_seed: int = 2026,
) -> list[dict]:
    """Portfolio selector for periods where red-side model signal is adverse.

    In adverse windows the historical validation says that model concentration is
    worse than a same-pipeline no-signal baseline.  This selector therefore
    treats red model scores as a last-mile tie-breaker only, while explicitly
    maximizing total red coverage and reserving the strongest distinct blue
    signals available in the candidate pool.
    """
    if not candidates or int(count) <= 0:
        return []

    unique: dict[tuple[tuple[int, ...], int], dict] = {}
    for candidate in candidates:
        unique.setdefault(_ticket_key(candidate), candidate)
    remaining = list(unique.values())
    effective_count = min(int(count), len(remaining))

    method_pool = _flatten_method_candidates(
        method_results,
        limit_per_method=max(1, min(effective_count, max(3, int(method_anchor_top_k_per_method)))),
    )
    method_ticket_support = _build_method_ticket_support(
        method_results,
        top_k_per_method=max(1, int(method_anchor_top_k_per_method)),
    )
    method_red_support = _build_shared_method_red_core(
        method_results,
        top_k_per_method=max(1, int(method_anchor_top_k_per_method)),
        minimum_method_support=1,
    )
    # Do not build the guard's red core from the method winners.  The guard is
    # entered exactly when those winners have been validated as adverse against
    # a same-pipeline no-signal baseline; using them as the support set quietly
    # re-injects the wrong core into every final portfolio.
    red_support = _build_red_consensus_support(
        sorted(
            remaining,
            key=lambda candidate: hashlib.sha256(
                (
                    f"{int(random_seed)}|red_support|"
                    f"{','.join(str(number) for number in candidate['red'])}|"
                    f"{int(candidate['blue'])}"
                ).encode("ascii")
            ).digest(),
        ),
        pool_size=max(1, min(1200, len(remaining))),
    )
    blue_support, blue_focus_strength = _build_blue_consensus_support(
        remaining,
        pool_size=max(1, min(80, len(remaining))),
    )

    blue_scores: dict[int, float] = {}
    for candidate in remaining:
        blue = int(candidate["blue"])
        blue_scores[blue] = max(
            blue_scores.get(blue, float("-inf")),
            float(candidate.get("blue_number_score", 1.0))
            + float(blue_signal_weight) * float(candidate.get("blue_signal_bonus", 0.0))
            + 0.25 * float(blue_support.get(blue, 0.0)),
        )
    target_blue_count = _minimum_blue_unique_count(
        effective_count,
        minimum_blue_unique_count if minimum_blue_unique_count is not None else effective_count,
    )
    target_blues = [
        blue for blue, _score in sorted(
            blue_scores.items(),
            key=lambda item: (float(item[1]), -int(item[0])),
            reverse=True,
        )[:target_blue_count]
    ]
    target_blue_rank = {
        blue: 1.0 - rank / max(len(target_blues), 1)
        for rank, blue in enumerate(target_blues)
    }

    selected: list[dict] = []
    red_incidence = {number: 0 for number in range(1, 34)}
    minimum_red_target = (
        int(minimum_red_coverage)
        if minimum_red_coverage is not None
        else min(33, max(12, effective_count * 4))
    )

    candidate_by_key = {_ticket_key(candidate): candidate for candidate in remaining}
    anchor_limit = max(0, min(effective_count, int(method_anchor_count)))
    if anchor_limit > 0 and method_pool:
        method_anchor_pool: list[dict] = []
        seen_anchor_keys: set[tuple[tuple[int, ...], int]] = set()
        for ticket in method_pool:
            key = _ticket_key(ticket)
            if key in seen_anchor_keys:
                continue
            seen_anchor_keys.add(key)
            method_anchor_pool.append(candidate_by_key.get(key, ticket))

        def method_anchor_score(candidate: dict) -> tuple[float, float, int]:
            blue = int(candidate["blue"])
            red_alignment = _red_core_alignment_bonus(candidate, red_support, top_k=12)
            core_hits = _red_core_hit_count(candidate, red_support, top_k=10)
            score = (
                0.45 * float(candidate.get("quality_score", 0.0))
                + 0.35 * float(candidate.get("consensus_score", candidate.get("quality_score", 0.0)))
                + 0.28 * float(candidate.get("blue_number_score", 1.0))
                + 0.45 * float(candidate.get("blue_signal_bonus", 0.0))
                + 0.25 * float(blue_support.get(blue, 0.0))
                + 0.62 * red_alignment
                + 0.10 * float(core_hits)
            )
            payload = (
                f"{int(random_seed)}|adverse_anchor|"
                f"{','.join(str(number) for number in candidate['red'])}|{blue}"
            ).encode("ascii")
            unbiased_tie = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            return score, float(blue not in {int(item["blue"]) for item in selected}), unbiased_tie

        for anchor in sorted(method_anchor_pool, key=method_anchor_score, reverse=True):
            if len(selected) >= anchor_limit:
                break
            if any(
                len(set(anchor["red"]) & set(ticket["red"])) > int(max_red_overlap)
                for ticket in selected
            ):
                continue
            if (
                len({int(ticket["blue"]) for ticket in selected}) < len(selected)
                and int(anchor["blue"]) in {int(ticket["blue"]) for ticket in selected}
            ):
                continue
            selected.append({
                **anchor,
                "rotation_matrix_row": len(selected) + 1,
                "selection_role": "red_adverse_guard_method_anchor",
                "blue_focus_strength": float(blue_focus_strength),
            })
            for number in anchor["red"]:
                red_incidence[int(number)] += 1
            anchor_key = _ticket_key(anchor)
            remaining = [
                candidate for candidate in remaining
                if _ticket_key(candidate) != anchor_key
            ]

    while remaining and len(selected) < effective_count:
        desired_blue = (
            target_blues[len(selected)]
            if len(selected) < len(target_blues)
            else None
        )
        feasible = [
            candidate for candidate in remaining
            if all(
                len(set(candidate["red"]) & set(ticket["red"])) <= int(max_red_overlap)
                for ticket in selected
            )
        ]
        choices = feasible or remaining
        if desired_blue is not None:
            blue_choices = [
                candidate for candidate in choices
                if int(candidate["blue"]) == int(desired_blue)
            ]
            if blue_choices:
                choices = blue_choices

        used_blues = {int(ticket["blue"]) for ticket in selected}
        current_red_coverage = len({
            int(number)
            for ticket in selected
            for number in ticket["red"]
        })
        anchor_follow_mode = bool(anchor_limit > 0 and selected)
        selected_anchors = [
            ticket for ticket in selected
            if str(ticket.get("selection_role", "")).endswith("method_anchor")
        ] or list(selected[: max(1, min(len(selected), anchor_limit))])

        def coverage_score(candidate: dict) -> tuple[float, float, int]:
            red = tuple(int(number) for number in candidate["red"])
            red_set = set(red)
            novel_red_count = sum(red_incidence[number] == 0 for number in red)
            novel_red = float(novel_red_count / 6.0)
            red_deficit = max(0, minimum_red_target - current_red_coverage)
            deficit_bonus = float(min(novel_red_count, red_deficit) / 6.0)
            incidence_balance = 1.0 / (1.0 + sum(red_incidence[number] for number in red))
            subset_gain = marginal_subset_gain(selected, candidate)
            max_overlap = max(
                (red_jaccard(red, ticket["red"]) for ticket in selected),
                default=0.0,
            )
            blue = int(candidate["blue"])
            blue_rank_bonus = float(target_blue_rank.get(blue, 0.0))
            blue_unique_bonus = float(blue not in used_blues)
            low_band = sum(1 <= number <= 11 for number in red)
            mid_band = sum(12 <= number <= 22 for number in red)
            high_band = sum(23 <= number <= 33 for number in red)
            band_balance = 1.0 - (
                abs(low_band - 2) + abs(mid_band - 2) + abs(high_band - 2)
            ) / 8.0
            span = max(red) - min(red)
            span_balance = 1.0 - min(abs(span - 24), 24) / 24.0
            blue_signal_bonus = (
                0.45 * float(candidate.get("blue_number_score", 1.0))
                + float(blue_signal_weight) * float(candidate.get("blue_signal_bonus", 0.0))
                + 0.30 * float(blue_support.get(blue, 0.0))
            )
            red_alignment = _red_core_alignment_bonus(candidate, red_support, top_k=12)
            core_hits = _red_core_hit_count(candidate, red_support, top_k=10)
            method_core_alignment = _red_core_alignment_bonus(
                candidate,
                method_red_support,
                top_k=12,
            )
            method_ticket_bonus = _method_ticket_bonus(candidate, method_ticket_support)
            anchor_overlap = max(
                (_anchor_overlap_bonus(candidate, anchor) for anchor in selected_anchors),
                default=0.0,
            )
            # When the red-side adverse guard is active, the first priority is
            # to stop one wrong model core from contaminating the whole
            # portfolio.  Until the portfolio reaches its red coverage target,
            # treat model-core alignment as almost inert; once the coverage
            # floor is satisfied, allow it only as a very small tie-breaker.
            effective_red_signal_weight = float(red_signal_weight)
            if red_deficit > 0:
                effective_red_signal_weight *= 0.10
            red_signal_bonus = effective_red_signal_weight * (
                1.20 * float(red_alignment) + 0.12 * float(core_hits)
            )
            if anchor_follow_mode:
                score = (
                    0.18 * novel_red
                    + 0.10 * deficit_bonus
                    + 0.10 * subset_gain
                    + 0.08 * incidence_balance
                    + 0.12 * band_balance
                    + 0.04 * span_balance
                    + 0.18 * blue_unique_bonus
                    + 0.12 * blue_rank_bonus
                    + 0.12 * blue_signal_bonus
                    + red_signal_bonus
                    + 0.22 * float(candidate.get("quality_score", 0.0))
                    + 0.14 * float(candidate.get("red_number_score", candidate.get("number_score", 0.0)))
                    + 0.14 * float(candidate.get("consensus_score", candidate.get("quality_score", 0.0)))
                    + 0.42 * method_core_alignment
                    + 0.50 * method_ticket_bonus
                    + 0.62 * anchor_overlap
                    - 0.18 * max_overlap
                )
            else:
                score = (
                    0.40 * novel_red
                    + 0.28 * deficit_bonus
                    + 0.24 * subset_gain
                    + 0.16 * incidence_balance
                    + 0.14 * band_balance
                    + 0.06 * span_balance
                    + 0.20 * blue_unique_bonus
                    + 0.14 * blue_rank_bonus
                    + 0.12 * blue_signal_bonus
                    + red_signal_bonus
                    + 0.005 * float(candidate.get("quality_score", 0.0))
                    - 0.46 * max_overlap
                )
            payload = (
                f"{int(random_seed)}|adverse|{len(selected)}|"
                f"{','.join(str(number) for number in red)}|{blue}|"
                f"{','.join(str(number) for number in sorted(red_set))}"
            ).encode("ascii")
            unbiased_tie = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            return score, novel_red, unbiased_tie

        best = max(choices, key=coverage_score)
        selected.append({
            **best,
            "rotation_matrix_row": len(selected) + 1,
            "selection_role": (
                "red_adverse_guard_method_followup"
                if anchor_follow_mode
                else "red_adverse_guard_coverage_matrix"
            ),
            "blue_focus_strength": float(blue_focus_strength),
        })
        for number in best["red"]:
            red_incidence[int(number)] += 1
        remaining.remove(best)

    if minimum_blue_unique_count is not None:
        selected = _ensure_minimum_blue_diversity(
            selected,
            candidates,
            minimum_unique_blues=_minimum_blue_unique_count(
                effective_count,
                int(minimum_blue_unique_count),
            ),
        )
    normalized_selected: list[dict] = []
    for index, ticket in enumerate(selected, start=1):
        normalized_selected.append({
            **ticket,
            "rotation_matrix_row": int(ticket.get("rotation_matrix_row", index)),
            "selection_role": ticket.get("selection_role", "red_adverse_guard_coverage_matrix"),
            "blue_focus_strength": float(ticket.get("blue_focus_strength", blue_focus_strength)),
        })
    return sorted(
        normalized_selected,
        key=lambda ticket: (
            int(ticket.get("rotation_matrix_row", 9999)),
            -float(ticket.get("quality_score", 0.0)),
        ),
    )


def _consensus_slot_score(
    candidate: dict,
    selected: list[dict],
    *,
    scoring_config: dict[str, Any],
    used_blues: set[int],
    lead_ticket: dict | None,
    anchor: dict | None,
    dominant_focus_strength: float,
    strong_lead_mode: bool,
    red_support: dict[int, float],
    blue_support: dict[int, float],
    method_ticket_support: dict[tuple, float],
    method_core_red_support: dict[int, float],
    shared_method_red_core: dict[int, float],
    red_consensus_weight: float,
    blue_diversity_weight: float,
    concentration_weight: float,
    blue_focus_bonus_weight: float,
    red_core_bonus_weight: float,
    red_core_top_k: int,
    method_ticket_bonus_weight: float,
    method_core_bonus_weight: float,
    shared_method_core_bonus_weight: float,
    top_consensus_overlap_bonus_weight: float,
) -> float:
    """Score one slot in the final consensus portfolio.

    The terms are grouped by intent:
    - base quality: candidate consensus/number quality from upstream ranking
    - blue signal: keep blue information visible without letting one blue dominate
    - diversity/coverage: avoid producing several visually identical tickets
    - anchor following: when methods agree strongly, keep tickets near the lead family
    - method/core support: reward numbers repeatedly supported by independent methods
    """
    weights = scoring_config.get("consensus_slot_weights", {})
    blue_number_weight = float(weights.get("blue_number", 0.30))
    blue_signal_weight = float(weights.get("blue_signal", 0.70))
    blue_anchor_signal_weight = float(weights.get("blue_anchor_signal", 0.45))
    dominant_anchor_weight = float(weights.get("dominant_anchor", 0.95))
    dominant_method_count_weight = float(weights.get("dominant_method_count", 0.35))
    dominant_anchor_gap_penalty = float(weights.get("dominant_anchor_gap_penalty", 0.30))
    dominant_min_anchor_overlap = float(weights.get("dominant_min_anchor_overlap", 0.50))
    strong_anchor_weight = float(weights.get("strong_anchor", 0.42))
    strong_method_agreement_weight = float(weights.get("strong_method_agreement", 0.20))
    strong_method_count_weight = float(weights.get("strong_method_count", 0.16))
    strong_anchor_gap_penalty = float(weights.get("strong_anchor_gap_penalty", 0.22))
    strong_min_anchor_overlap = float(weights.get("strong_min_anchor_overlap", 0.62))

    lead_overlap = _anchor_overlap_bonus(candidate, lead_ticket)
    return (
        float(candidate["consensus_score"])
        + blue_number_weight * float(candidate.get("blue_number_score", 1.0))
        + blue_signal_weight * float(candidate.get("blue_signal_bonus", 0.0))
        + blue_anchor_signal_weight * float(candidate.get("blue_signal_bonus", 0.0)) * lead_overlap
        - float(scoring_config.get("diversity_penalty", 0.25))
        * max((red_jaccard(candidate["red"], item["red"]) for item in selected), default=0.0)
        + float(scoring_config.get("portfolio_subset_weight", 0.20))
        * marginal_subset_gain(selected, candidate)
        + float(blue_diversity_weight) * float(int(candidate["blue"]) not in used_blues)
        + float(concentration_weight) * _anchor_overlap_bonus(candidate, anchor)
        + float(top_consensus_overlap_bonus_weight) * float(dominant_focus_strength) * lead_overlap
        + dominant_anchor_weight * float(dominant_focus_strength) * lead_overlap
        + dominant_method_count_weight
        * float(dominant_focus_strength)
        * float(candidate.get("method_support_count", 0.0))
        - dominant_anchor_gap_penalty
        * float(dominant_focus_strength)
        * max(0.0, dominant_min_anchor_overlap - lead_overlap)
        + strong_anchor_weight * float(strong_lead_mode) * lead_overlap
        + strong_method_agreement_weight
        * float(strong_lead_mode)
        * float(candidate.get("method_agreement_score", 0.0))
        + strong_method_count_weight
        * float(strong_lead_mode)
        * float(candidate.get("method_support_count", 0.0))
        - strong_anchor_gap_penalty
        * float(strong_lead_mode)
        * max(0.0, strong_min_anchor_overlap - lead_overlap)
        + red_consensus_weight * _red_consensus_bonus(candidate, red_support)
        + float(blue_focus_bonus_weight) * _blue_consensus_bonus(candidate, blue_support)
        + float(red_core_bonus_weight) * _red_core_alignment_bonus(
            candidate,
            red_support,
            top_k=red_core_top_k,
        )
        + method_ticket_bonus_weight * _method_ticket_bonus(candidate, method_ticket_support)
        + method_core_bonus_weight * _red_consensus_bonus(candidate, method_core_red_support)
        + shared_method_core_bonus_weight * _shared_method_core_bonus(candidate, shared_method_red_core)
    )


def consensus_optimize(
    candidates: list[dict],
    method_results: dict[str, list[dict]],
    count: int = 3,
    *,
    scoring_config: dict[str, Any],
    max_red_overlap: int = 4,
    blue_diversity_weight: float = 0.30,
    minimum_blue_unique_count: int = 3,
    concentration_weight: float = 0.0,
    minimum_unique_red_sets: int | None = None,
    minimum_red_coverage: int | None = None,
    red_core_bonus_weight: float = 0.0,
    red_core_top_k: int = 6,
    preserve_top_k: int = 0,
    red_coverage_relief_max: float = 0.0,
    red_set_relief_max: float = 0.0,
    blue_focus_bonus_weight: float = 0.0,
    blue_unique_relief_max: float = 0.0,
    force_method_anchor_count: int = 0,
    force_method_anchor_top_k_per_method: int = 1,
    force_method_anchor_min_support: int = 1,
    method_agreement_bonus_weight: float = 0.0,
    top_consensus_margin_threshold: float = 0.0,
    top_consensus_overlap_bonus_weight: float = 0.0,
    allow_concentrated_portfolio: bool = False,
    concentrated_ticket_count: int | None = None,
) -> list[dict]:
    """Select a small multi-objective consensus set from the fused candidate pool."""
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if not candidates:
        return []
    count = min(int(count), len(candidates))
    pool_size = len(candidates)
    structure_values = np.asarray([float(candidate["structure_score"]) for candidate in candidates])
    structure_min = float(structure_values.min())
    structure_span = max(float(structure_values.max()) - structure_min, 1e-12)
    method_ranks = {
        method: {_ticket_key(ticket): rank for rank, ticket in enumerate(tickets)}
        for method, tickets in method_results.items()
    }
    method_count = max(len(method_ranks), 1)
    quality_weight = float(scoring_config.get("consensus_quality_weight", 0.80))
    support_weight = float(scoring_config.get("consensus_support_weight", 0.20))
    structure_weight = float(scoring_config.get("consensus_structure_weight", 0.0))
    method_ticket_bonus_weight = float(
        scoring_config.get("signal_method_ticket_bonus_weight", 0.0)
    )
    method_core_bonus_weight = float(
        scoring_config.get("signal_method_core_bonus_weight", 0.0)
    )
    shared_method_core_bonus_weight = float(
        scoring_config.get("signal_shared_method_core_bonus_weight", 0.0)
    )
    anchor = _signal_anchor_ticket(candidates)
    method_support_limit = max(1, min(int(count), 3))
    method_support_candidates = _flatten_method_candidates(
        method_results,
        limit_per_method=method_support_limit,
    )
    support_source = method_support_candidates or candidates
    method_ticket_support = _build_method_ticket_support(
        method_results,
        top_k_per_method=max(1, min(int(count), 2)),
    )
    shared_method_red_core = _build_shared_method_red_core(
        method_results,
        top_k_per_method=max(1, int(scoring_config.get("signal_shared_method_core_top_k_per_method", 2))),
        minimum_method_support=max(1, int(scoring_config.get("signal_shared_method_core_min_support", 2))),
    )
    red_support = _build_red_consensus_support(
        support_source,
        pool_size=int(scoring_config.get("signal_red_consensus_pool_size", 20)),
    )
    method_core_red_support = _build_red_consensus_support(
        method_support_candidates or support_source,
        pool_size=max(1, len(method_support_candidates or support_source)),
    )
    blue_support, blue_focus_strength = _build_blue_consensus_support(
        support_source,
        pool_size=int(scoring_config.get("signal_blue_consensus_pool_size", 20)),
    )
    dominant_blue = None
    if blue_support:
        dominant_blue = max(
            blue_support,
            key=lambda number: (float(blue_support[number]), -int(number)),
        )
    red_consensus_weight = float(scoring_config.get("signal_red_consensus_weight", 0.0))
    anchor_focus_strength = 0.0
    if anchor:
        anchor_focus_strength = max(
            _red_consensus_bonus(anchor, red_support),
            _red_core_alignment_bonus(anchor, red_support, top_k=red_core_top_k),
        )
    decorated: list[dict] = []
    for pool_rank, candidate in enumerate(candidates):
        key = _ticket_key(candidate)
        quality_rank_score = 1.0 - pool_rank / max(pool_size - 1, 1)
        support_methods = [method for method, ranks in method_ranks.items() if key in ranks]
        support_ratio = len(support_methods) / method_count
        support_score = sum(
            1.0 - method_ranks[method][key] / max(len(method_results[method]), 1)
            for method in support_methods
        ) / method_count
        structure_rank_score = (float(candidate["structure_score"]) - structure_min) / structure_span
        consensus_score = (
            quality_weight * quality_rank_score
            + support_weight * support_score
            + structure_weight * structure_rank_score
            + float(method_agreement_bonus_weight) * support_ratio
            + red_consensus_weight * _red_consensus_bonus(candidate, red_support)
            + float(red_core_bonus_weight) * _red_core_alignment_bonus(
                candidate,
                red_support,
                top_k=red_core_top_k,
            )
            + float(blue_focus_bonus_weight) * _blue_consensus_bonus(candidate, blue_support)
            + method_ticket_bonus_weight * _method_ticket_bonus(candidate, method_ticket_support)
            + method_core_bonus_weight * _red_consensus_bonus(candidate, method_core_red_support)
            + shared_method_core_bonus_weight * _shared_method_core_bonus(candidate, shared_method_red_core)
        )
        decorated.append({
            **candidate,
            "candidate_pool_rank": int(pool_rank + 1),
            "quality_rank_score": float(quality_rank_score),
            "structure_rank_score": float(structure_rank_score),
            "method_support": support_methods,
            "method_support_count": len(support_methods),
            "method_agreement_score": float(support_ratio),
            "consensus_score": float(consensus_score),
        })

    ranked_decorated = sorted(
        decorated,
        key=lambda candidate: (
            float(candidate.get("consensus_score", candidate.get("quality_score", 0.0))),
            float(candidate.get("quality_score", 0.0)),
        ),
        reverse=True,
    )
    lead_ticket = ranked_decorated[0] if ranked_decorated else None
    second_ticket = ranked_decorated[1] if len(ranked_decorated) > 1 else None
    lead_score = float(lead_ticket.get("consensus_score", 0.0)) if lead_ticket else 0.0
    second_score = float(second_ticket.get("consensus_score", 0.0)) if second_ticket else 0.0
    score_base = max(abs(lead_score), 1e-9)
    top_consensus_margin = max(0.0, (lead_score - second_score) / score_base)
    dominant_focus_strength = max(
        0.0,
        min(
            1.0,
            top_consensus_margin / max(float(top_consensus_margin_threshold), 1e-9),
        ),
    ) if float(top_consensus_margin_threshold) > 0 else 0.0

    anchor_count = max(0, min(int(count), int(force_method_anchor_count)))
    anchor_seed = _seed_consensus_anchors(
        decorated,
        method_results,
        count=anchor_count,
        top_k_per_method=max(1, int(force_method_anchor_top_k_per_method)),
        minimum_method_support=max(1, int(force_method_anchor_min_support)),
    )
    if lead_ticket is not None and dominant_focus_strength >= 0.60 and anchor_seed:
        tightened_seed: list[dict] = []
        for ticket in anchor_seed:
            if _ticket_key(ticket) == _ticket_key(lead_ticket):
                tightened_seed.append(ticket)
                continue
            overlap_to_lead = _anchor_overlap_bonus(ticket, lead_ticket)
            if (
                overlap_to_lead >= 0.50
                or int(ticket.get("method_support_count", 0)) >= 2
            ):
                tightened_seed.append(ticket)
        anchor_seed = tightened_seed
    if (
        lead_ticket is not None
        and dominant_focus_strength > 0.0
        and _ticket_key(lead_ticket) not in {_ticket_key(ticket) for ticket in anchor_seed}
    ):
        anchor_seed = [lead_ticket, *anchor_seed]
        anchor_seed = anchor_seed[:count]
    selected: list[dict] = list(anchor_seed)
    selected_keys = {_ticket_key(ticket) for ticket in selected}
    remaining = [
        candidate for candidate in decorated
        if _ticket_key(candidate) not in selected_keys
    ]
    strong_lead_mode = bool(
        lead_ticket is not None
        and (
            dominant_focus_strength >= 0.68
            or anchor_focus_strength >= 0.72
            or (
                int(lead_ticket.get("method_support_count", 0)) >= 2
                and dominant_focus_strength >= 0.45
            )
        )
    )
    concentrated_head_count = max(
        0,
        min(
            int(count),
            int(concentrated_ticket_count if concentrated_ticket_count is not None else max(1, int(round(count * 0.60)))),
        ),
    )
    while remaining and len(selected) < count:
        concentrate_this_slot = bool(
            allow_concentrated_portfolio and len(selected) < concentrated_head_count
        )
        feasible = [] if concentrate_this_slot else [
            candidate for candidate in remaining
            if all(len(set(candidate["red"]) & set(item["red"])) <= int(max_red_overlap) for item in selected)
        ]
        used_blues = {int(ticket["blue"]) for ticket in selected}
        choices = feasible or remaining
        best = max(
            choices,
            key=lambda candidate: _consensus_slot_score(
                candidate,
                selected,
                scoring_config=scoring_config,
                used_blues=used_blues,
                lead_ticket=lead_ticket,
                anchor=anchor,
                dominant_focus_strength=dominant_focus_strength,
                strong_lead_mode=strong_lead_mode,
                red_support=red_support,
                blue_support=blue_support,
                method_ticket_support=method_ticket_support,
                method_core_red_support=method_core_red_support,
                shared_method_red_core=shared_method_red_core,
                red_consensus_weight=red_consensus_weight,
                blue_diversity_weight=blue_diversity_weight,
                concentration_weight=concentration_weight,
                blue_focus_bonus_weight=blue_focus_bonus_weight,
                red_core_bonus_weight=red_core_bonus_weight,
                red_core_top_k=red_core_top_k,
                method_ticket_bonus_weight=method_ticket_bonus_weight,
                method_core_bonus_weight=method_core_bonus_weight,
                shared_method_core_bonus_weight=shared_method_core_bonus_weight,
                top_consensus_overlap_bonus_weight=top_consensus_overlap_bonus_weight,
            )
        )
        selected.append(best)
        remaining.remove(best)
    red_target = (
        int(minimum_red_coverage)
        if minimum_red_coverage is not None
        else min(33, max(6, int(count) + 5))
    )
    support_ratio = (
        max((float(ticket.get("method_support_count", 0)) for ticket in selected), default=0.0)
        / float(method_count)
    )
    focus_strength = max(
        0.0,
        min(
            1.0,
            0.55 * float(anchor_focus_strength)
            + 0.45 * float(support_ratio),
        ),
    )
    shared_core_strength = float(np.mean(list(shared_method_red_core.values()))) if shared_method_red_core else 0.0
    family_focus_strength = max(focus_strength, shared_core_strength, dominant_focus_strength)
    red_target = max(
        6,
        int(round(red_target * (1.0 - float(red_coverage_relief_max) * family_focus_strength))),
    )
    strong_red_family = max(family_focus_strength, focus_strength)
    if int(count) >= 5:
        if allow_concentrated_portfolio and strong_lead_mode:
            red_target = min(red_target, 8)
        if allow_concentrated_portfolio and (strong_red_family >= 0.70 or dominant_focus_strength >= 0.55):
            red_target = min(red_target, 8)
        elif dominant_focus_strength < 0.30:
            red_target = max(red_target, 7)
        if allow_concentrated_portfolio and strong_red_family >= 0.56:
            red_target = min(red_target, 8)
        if allow_concentrated_portfolio and strong_red_family >= 0.72:
            red_target = min(red_target, 7)
        if allow_concentrated_portfolio and (strong_red_family >= 0.82 or dominant_focus_strength >= 0.72):
            red_target = min(red_target, 6)
    if not allow_concentrated_portfolio and minimum_red_coverage is not None:
        red_target = max(red_target, int(minimum_red_coverage))
    explicit_red_set_target = _minimum_red_unique_set_count(count, minimum_unique_red_sets)
    red_set_target = int(explicit_red_set_target)
    red_set_target = max(
        1,
        int(round(red_set_target * (1.0 - float(red_set_relief_max) * family_focus_strength))),
    )
    if allow_concentrated_portfolio:
        red_target = max(
            max(6, int(count) + 3),
            min(red_target, max(6, int(count) + 6)),
        )
        red_set_target = min(int(explicit_red_set_target), int(red_set_target))
    else:
        red_set_target = max(int(explicit_red_set_target), int(red_set_target))
    if int(count) >= 5 and strong_red_family >= 0.55 and not allow_concentrated_portfolio:
        red_set_target = max(red_set_target, 2)
        if shared_core_strength >= 0.50 or dominant_focus_strength >= 0.60:
            red_set_target = max(red_set_target, 3)
    if dominant_focus_strength >= 0.60 and not allow_concentrated_portfolio:
        red_set_target = max(
            red_set_target,
            2 if int(count) >= 3 else 1,
        )
    if dominant_focus_strength >= 0.80 and not allow_concentrated_portfolio:
        red_set_target = max(
            red_set_target,
            3 if int(count) >= 5 else 2,
        )
    blue_target = _minimum_blue_unique_count(count, minimum_blue_unique_count)
    if bool(scoring_config.get("allow_blue_unique_relaxation", False)):
        relaxed_target = max(
            1,
            int(round(blue_target * (1.0 - float(blue_unique_relief_max) * float(blue_focus_strength)))),
        )
        if blue_focus_strength >= float(scoring_config.get("blue_unique_relaxation_min_strength", 0.85)):
            relaxed_floor = int(scoring_config.get("blue_unique_relaxation_floor", 4))
            blue_target = max(1, min(blue_target, max(relaxed_floor, relaxed_target)))
    if int(count) >= 5 and not allow_concentrated_portfolio:
        blue_target = max(blue_target, min(int(count), 5))
    reserved_blue_signal_count = 2 if int(count) >= 5 else 1 if int(count) >= 3 else 0
    if int(count) >= 5 and blue_focus_strength >= 0.60:
        reserved_blue_signal_count = max(reserved_blue_signal_count, 2)
    if blue_focus_strength < 0.35:
        reserved_blue_signal_count = max(reserved_blue_signal_count, 1)
    reserved_blue_signal_min_bonus = max(
        0.22,
        0.48 - 0.30 * float(blue_focus_strength),
    )
    dominant_blue_min_repeat_count = 0
    if int(count) < 5 and int(count) >= 3 and blue_focus_strength >= 0.62:
        dominant_blue_min_repeat_count = 2

    return _ensure_signal_portfolio_diversity(
        selected,
        candidates,
        minimum_unique_red_sets=red_set_target,
        minimum_red_coverage=red_target,
        minimum_unique_blues=blue_target,
        red_support=red_support,
        red_consensus_weight=red_consensus_weight,
        blue_support=blue_support,
        blue_consensus_weight=float(blue_focus_bonus_weight),
        method_ticket_support=method_ticket_support,
        method_ticket_bonus_weight=method_ticket_bonus_weight,
        method_core_red_support=method_core_red_support,
        method_core_bonus_weight=method_core_bonus_weight,
        shared_method_red_core=shared_method_red_core,
        shared_method_core_bonus_weight=shared_method_core_bonus_weight,
        preserve_top_k=(
            max(
                1 if int(count) >= 3 else 0,
                int(preserve_top_k),
            )
            if not allow_concentrated_portfolio
            else max(
            int(preserve_top_k),
            len(anchor_seed),
            min(
                int(count),
                len(anchor_seed) + (1 if focus_strength >= 0.45 else 0),
            ),
            min(
                int(count),
                len(anchor_seed) + (2 if shared_core_strength >= 0.50 else 0),
            ),
            min(
                int(count),
                len(anchor_seed) + (2 if dominant_focus_strength >= 0.60 else 0),
            ),
            min(
                int(count),
                len(anchor_seed) + (2 if strong_red_family >= 0.55 else 0),
            ),
            int(count) if dominant_focus_strength >= 0.80 else 0,
            )
        ),
        lead_ticket=lead_ticket,
        lead_overlap_floor=(
            0
            if not allow_concentrated_portfolio
            else
            5
            if dominant_focus_strength >= 0.80
            else 4 if dominant_focus_strength >= 0.60
            else 0
        ),
        maximum_duplicate_red_sets=(
            None
            if allow_concentrated_portfolio
            else 2
            if (
                int(count) >= 5
                and strong_red_family >= 0.62
                and dominant_focus_strength >= 0.66
            )
            else 1
            if int(count) >= 5
            else 1 if int(count) >= 3 and red_set_target >= 2
            else None
        ),
        reserved_blue_signal_count=reserved_blue_signal_count,
        reserved_blue_signal_min_bonus=reserved_blue_signal_min_bonus,
        reserved_blue_signal_min_quality_ratio=0.60,
        dominant_blue=dominant_blue,
        dominant_blue_focus_strength=blue_focus_strength,
        dominant_blue_min_repeat_count=dominant_blue_min_repeat_count,
        dominant_blue_min_quality_ratio=0.72,
    )


def optimize_candidates(
    candidates: list[dict],
    count: int,
    method: str,
    config: dict[str, Any],
    scoring_config: dict[str, Any],
    random_seed: int,
    *,
    history_frame: Any = None,
) -> list[dict]:
    if int(count) <= 0:
        raise ValueError("count 必须大于0")
    if not isinstance(candidates, list):
        raise ValueError("candidates 必须是列表")
    if not candidates:
        return []
    if method not in {"greedy", "genetic", "pareto", "historical_hit", "red_ge3_objective", "global_coverage"}:
        raise ValueError("优化方法仅支持 greedy、genetic、pareto、historical_hit、red_ge3_objective 或 global_coverage")
    signal_max_red_overlap = int(config.get("signal_max_red_overlap", config.get("max_red_overlap", 4)))
    signal_blue_diversity_weight = float(
        config.get("signal_blue_diversity_weight", config.get("blue_diversity_weight", 0.30))
    )
    if bool(config.get("coverage_param_online_tune", False)):
        blue_scores = np.asarray([
            float(candidate.get("blue_signal_bonus", candidate.get("blue_number_score", 1.0)))
            for candidate in candidates
        ], dtype=float)
        if blue_scores.size:
            concentration = float(np.std(blue_scores) / max(float(np.mean(blue_scores)), 1e-9))
            # Stronger blue signal: less forced diversity.  Weak/noisy signal:
            # more coverage diversity.  This mirrors the blue-only λ/μ grid
            # search while staying cheap enough for every optimizer call.
            signal_blue_diversity_weight = float(np.clip(
                signal_blue_diversity_weight * (1.15 - 0.55 * min(concentration, 1.0)),
                0.12,
                0.55,
            ))
    signal_minimum_blue_unique_count = int(
        config.get("signal_minimum_blue_unique_count", config.get("minimum_blue_unique_count", 3))
    )
    signal_diversity_penalty_scale = float(config.get("signal_diversity_penalty_scale", 1.0))
    signal_concentration_weight = float(config.get("signal_concentration_weight", 0.0))
    signal_red_consensus_weight = float(config.get("signal_red_consensus_weight", 0.0))
    signal_red_consensus_pool_size = int(config.get("signal_red_consensus_pool_size", 20))
    signal_red_core_bonus_weight = float(config.get("signal_red_core_bonus_weight", 0.0))
    signal_red_core_top_k = int(config.get("signal_red_core_top_k", 6))
    effective_scoring = {
        **scoring_config,
        "diversity_penalty": float(scoring_config.get("diversity_penalty", 0.25))
        * signal_diversity_penalty_scale,
        "signal_red_consensus_weight": signal_red_consensus_weight,
        "signal_red_consensus_pool_size": signal_red_consensus_pool_size,
    }
    if method == "greedy":
        return greedy_optimize(
            candidates,
            count,
            scoring_config=effective_scoring,
            max_red_overlap=signal_max_red_overlap,
            blue_diversity_weight=signal_blue_diversity_weight,
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            concentration_weight=signal_concentration_weight,
            red_consensus_weight=signal_red_consensus_weight,
            red_consensus_pool_size=signal_red_consensus_pool_size,
            red_core_bonus_weight=signal_red_core_bonus_weight,
            red_core_top_k=signal_red_core_top_k,
        )
    if method == "genetic":
        return genetic_optimize(
            candidates,
            count,
            population_size=int(config.get("genetic_population", 24)),
            generations=int(config.get("genetic_generations", 30)),
            mutation_rate=float(config.get("genetic_mutation_rate", 0.15)),
            diversity_penalty=float(effective_scoring.get("diversity_penalty", 0.25)),
            blue_diversity_weight=signal_blue_diversity_weight,
            portfolio_subset_weight=float(effective_scoring.get("portfolio_subset_weight", 0.20)),
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            concentration_weight=signal_concentration_weight,
            red_consensus_weight=signal_red_consensus_weight,
            red_consensus_pool_size=signal_red_consensus_pool_size,
            red_core_bonus_weight=signal_red_core_bonus_weight,
            red_core_top_k=signal_red_core_top_k,
            random_seed=random_seed,
        )
    if method == "pareto":
        return pareto_optimize(
            candidates,
            count,
            scoring_config=effective_scoring,
            max_red_overlap=signal_max_red_overlap,
            blue_diversity_weight=signal_blue_diversity_weight,
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            red_consensus_weight=signal_red_consensus_weight,
            red_consensus_pool_size=signal_red_consensus_pool_size,
            red_core_bonus_weight=signal_red_core_bonus_weight,
            red_core_top_k=signal_red_core_top_k,
        )
    if method == "historical_hit":
        tuned = _optuna_tune_historical_hit_parameters(
            candidates,
            history_frame,
            config,
            effective_scoring,
            random_seed,
        )
        return historical_hit_optimize(
            candidates,
            count,
            history_frame=history_frame,
            window=int(config.get("historical_hit_window", 300)),
            scoring_config=effective_scoring,
            max_red_overlap=int(tuned.get("max_red_overlap", signal_max_red_overlap)),
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            history_weight=float(tuned.get("history_weight", config.get("historical_hit_weight", 0.70))),
            current_signal_weight=float(
                tuned.get(
                    "current_signal_weight",
                    config.get("historical_hit_current_signal_weight", 0.22),
                )
            ),
            coverage_weight=float(tuned.get("coverage_weight", config.get("historical_hit_coverage_weight", 0.18))),
        )
    if method == "red_ge3_objective":
        tuned = _optuna_tune_red_ge3_parameters(
            candidates,
            history_frame,
            config,
            effective_scoring,
            random_seed,
        )
        return red_ge3_objective_optimize(
            candidates,
            count,
            history_frame=history_frame,
            window=int(config.get("red_ge3_window", config.get("historical_hit_window", 300))),
            scoring_config=effective_scoring,
            max_red_overlap=int(tuned.get("max_red_overlap", config.get("red_ge3_max_red_overlap", max(signal_max_red_overlap, 5)))),
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            candidate_cap=int(config.get("red_ge3_candidate_cap", 500)),
            any3_weight=float(tuned.get("any3_weight", config.get("red_ge3_any3_weight", 1.00))),
            any4_weight=float(tuned.get("any4_weight", config.get("red_ge3_any4_weight", 0.18))),
            best_hit_weight=float(tuned.get("best_hit_weight", config.get("red_ge3_best_hit_weight", 0.22))),
            quality_weight=float(tuned.get("quality_weight", config.get("red_ge3_quality_weight", 0.14))),
            coverage_weight=float(tuned.get("coverage_weight", config.get("red_ge3_coverage_weight", 0.10))),
        )
    if method == "global_coverage":
        in_walk_forward = bool(config.get("_walk_forward_context", False))
        candidate_cap_key = (
            "global_coverage_backtest_candidate_cap"
            if in_walk_forward and "global_coverage_backtest_candidate_cap" in config
            else "global_coverage_candidate_cap"
        )
        time_limit_key = (
            "global_coverage_backtest_time_limit_seconds"
            if in_walk_forward and "global_coverage_backtest_time_limit_seconds" in config
            else "global_coverage_time_limit_seconds"
        )
        return global_coverage_optimize(
            candidates,
            count,
            scoring_config=effective_scoring,
            max_red_overlap=signal_max_red_overlap,
            minimum_blue_unique_count=signal_minimum_blue_unique_count,
            candidate_cap=int(config.get(candidate_cap_key, config.get("global_coverage_candidate_cap", 300))),
            time_limit_seconds=int(
                config.get(
                    time_limit_key,
                    config.get("global_coverage_time_limit", 20),
                )
            ),
        )
    raise ValueError("优化方法仅支持 greedy、genetic、pareto、historical_hit、red_ge3_objective 或 global_coverage")
