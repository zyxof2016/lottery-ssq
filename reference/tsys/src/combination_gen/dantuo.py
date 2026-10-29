from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb, exp, log
from typing import Any, Iterable

import numpy as np

from src.blue.adaptive_policy import select_adaptive_blue_set


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


def _rank_support(
    tickets: list[dict[str, Any]],
    *,
    fusion_rank: list[int] | None,
    number_low: int,
    number_high: int,
    color: str,
) -> list[tuple[int, float, int, int]]:
    counter: Counter[int] = Counter()
    score_sum: Counter[int] = Counter()
    for index, ticket in enumerate(tickets):
        ticket_weight = 1.0 / (1.0 + index * 0.08)
        if color == "blue":
            role = str(ticket.get("selection_role", "") or "")
            blue_role = str(ticket.get("blue_selection_role", "") or "")
            if role.startswith("dantuo_method_source_"):
                ticket_weight *= 0.28
            if role.startswith("dantuo_candidate_pool_"):
                ticket_weight *= 0.16
            if blue_role == "blue_specialist_portfolio":
                ticket_weight *= 2.8
        ticket_score = float(ticket.get("consensus_score", 0.0) or 0.0)
        if color == "red":
            numbers = _as_int_list(ticket.get("red", []), low=number_low, high=number_high)
        else:
            numbers = _as_int_list([ticket.get("blue")], low=number_low, high=number_high)
        for number in numbers:
            counter[number] += 1
            score_sum[number] += ticket_weight + max(ticket_score, 0.0) * 0.02

    fusion_order = _as_int_list(fusion_rank or [], low=number_low, high=number_high)
    fusion_position = {number: rank for rank, number in enumerate(fusion_order)}
    all_numbers = set(counter) | set(fusion_order)
    for number in range(number_low, number_high + 1):
        all_numbers.add(number)

    ranked = []
    for number in all_numbers:
        support = int(counter.get(number, 0))
        fusion_bonus = 1.0 / (1.0 + fusion_position.get(number, number_high + 8))
        score = float(score_sum.get(number, 0.0)) + fusion_bonus
        ranked.append((number, score, support, fusion_position.get(number, 999)))
    return sorted(ranked, key=lambda item: (-item[2], -item[1], item[3], item[0]))


def _ranked_numbers(ranked: list[tuple[int, float, int, int]]) -> list[int]:
    return [number for number, _, _, _ in ranked]


def _fusion_score_map(
    fusion: dict[str, Any] | None,
    *,
    key_numbers: str,
    key_scores: str,
    low: int,
    high: int,
) -> dict[int, float]:
    numbers = _as_int_list((fusion or {}).get(key_numbers) or [], low=low, high=high)
    raw_scores = list((fusion or {}).get(key_scores) or [])
    result: dict[int, float] = {}
    for index, number in enumerate(numbers):
        try:
            result[int(number)] = float(raw_scores[index])
        except (IndexError, TypeError, ValueError):
            result[int(number)] = 0.0
    return result


def _source_method_support(
    tickets: list[dict[str, Any]],
    *,
    color: str,
    low: int,
    high: int,
) -> dict[int, set[str]]:
    support: dict[int, set[str]] = {number: set() for number in range(low, high + 1)}
    for ticket in tickets:
        role = str(ticket.get("selection_role", "") or "")
        if role.startswith("dantuo_method_source_"):
            method = role.replace("dantuo_method_source_", "", 1)
        elif role.startswith("dantuo_candidate_pool_"):
            method = "candidate_pool"
        else:
            method = "final_layer"
        numbers = (
            _as_int_list(ticket.get("red", []), low=low, high=high)
            if color == "red"
            else _as_int_list([ticket.get("blue")], low=low, high=high)
        )
        for number in numbers:
            support.setdefault(int(number), set()).add(method)
    return support


def _evidence_score_map(
    ranked: list[tuple[int, float, int, int]],
    method_support: dict[int, set[str]],
    *,
    fusion_scores: dict[int, float] | None = None,
) -> dict[int, float]:
    support_map = {number: support for number, _, support, _ in ranked}
    score_map = {number: score for number, score, _, _ in ranked}
    rank_map = {number: rank for number, _, _, rank in ranked}
    max_support = max(support_map.values(), default=1) or 1
    max_score = max(score_map.values(), default=1.0) or 1.0
    fusion_scores = fusion_scores or {}
    max_fusion = max(fusion_scores.values(), default=1.0) or 1.0
    result: dict[int, float] = {}
    for number in support_map:
        support_norm = float(support_map.get(number, 0)) / float(max_support)
        score_norm = float(score_map.get(number, 0.0)) / float(max_score)
        fusion_norm = float(fusion_scores.get(number, 0.0)) / float(max_fusion)
        rank_bonus = 1.0 / (1.0 + float(rank_map.get(number, 999)))
        method_diversity = min(len(method_support.get(number, set())), 6) / 6.0
        result[number] = (
            1.15 * support_norm
            + 0.85 * score_norm
            + 0.95 * fusion_norm
            + 0.55 * rank_bonus
            + 0.80 * method_diversity
        )
    return result


def _blue_history_sequence(blue_history: Iterable[Any] | None) -> list[int]:
    result: list[int] = []
    for value in blue_history or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 16:
            result.append(number)
    return result


def _blue_gap_since_last(blue_history: Iterable[Any] | None, number: int) -> int | None:
    history = _blue_history_sequence(blue_history)
    if not history:
        return None
    for gap, value in enumerate(reversed(history), start=1):
        if int(value) == int(number):
            return int(gap)
    return len(history) + 1


def _blue_omission_balance_score(gap: int | None) -> float:
    if gap is None:
        return 0.0
    ratio = max(float(gap), 1.0) / 16.0
    return float(exp(-abs(log(ratio))))


def _blue_immediate_repeat_score(gap: int | None) -> float:
    if gap is None:
        return 0.0
    return 1.0 / (float(gap) ** 0.65)


def _blue_prediction_set_width(
    fusion_scores: dict[int, float],
    *,
    requested_count: int,
) -> tuple[int, str]:
    """Choose production blue candidate width from uncertainty/edge strength."""

    if not fusion_scores:
        return max(10, min(16, int(requested_count) + 7)), "NO_EDGE"
    values = [max(0.0, float(fusion_scores.get(number, 0.0))) for number in range(1, 17)]
    total = sum(values)
    if total <= 0:
        return max(10, min(16, int(requested_count) + 7)), "NO_EDGE"
    probabilities = [value / total for value in values]
    entropy = -sum(value * log(max(value, 1e-12)) for value in probabilities) / log(16.0)
    top5_mass = sum(sorted(probabilities, reverse=True)[:5])
    if entropy <= 0.86 and top5_mass >= 0.44:
        return max(5, min(6, int(requested_count) + 2)), "STRONG_EDGE"
    if entropy <= 0.93 and top5_mass >= 0.36:
        return max(8, min(10, int(requested_count) + 5)), "WEAK_EDGE"
    return max(10, min(16, int(requested_count) + 9)), "NO_EDGE"


def _select_blue_numbers_for_prediction_set(
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    *,
    count: int,
    blue_history: Iterable[Any] | None = None,
    recent_gap_weight: float = 1.00,
    recent_frequency_weight: float = 0.00,
    evidence_weight: float = 1.00,
    specialist_weight: float = 0.20,
    omission_balance_weight: float = 0.00,
    immediate_repeat_penalty: float = 0.00,
    recent_windows: Iterable[int] | None = None,
) -> list[int]:
    """Select blue numbers with Prediction Set + Hit@5 coverage objective.

    Blue validation has repeatedly shown weak Top1 edge.  The budget layer now
    first builds a dynamic prediction set and then selects a small coverage set
    from probability tiers.  Top1 rank is only a tie-breaker, not the KPI.
    """

    method_support = _source_method_support(tickets, color="blue", low=1, high=16)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_blue",
        key_scores="top_blue_scores",
        low=1,
        high=16,
    )
    base_scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    prediction_width, edge_state = _blue_prediction_set_width(
        fusion_scores,
        requested_count=count,
    )
    prediction_set = [
        number for number, _, _, _ in sorted(
            ranked,
            key=lambda item: (
                -float(base_scores.get(item[0], 0.0)),
                item[3],
                item[0],
            ),
        )
    ][:prediction_width]
    specialist_bonus: Counter[int] = Counter()
    for ticket in tickets:
        blue_values = _as_int_list([ticket.get("blue")], low=1, high=16)
        if not blue_values:
            continue
        role = str(ticket.get("blue_selection_role", "") or "")
        if "blue_specialist" in role:
            specialist_bonus[blue_values[0]] += 1
    history = _blue_history_sequence(blue_history)
    window_source = [20, 50, 100] if recent_windows is None else recent_windows
    windows = [int(window) for window in window_source if int(window) > 0]
    max_base = max(base_scores.values(), default=1.0) or 1.0
    max_specialist = max(specialist_bonus.values(), default=1) or 1

    def current_gap(number: int) -> int | None:
        if not history:
            return None
        for gap, value in enumerate(reversed(history), start=1):
            if int(value) == int(number):
                return int(gap)
        return len(history) + 1

    def recent_gap_score(number: int) -> float:
        gap = current_gap(number)
        if gap is None:
            return 0.0
        return 1.0 / (float(gap) ** 0.50)

    def omission_balance_score(number: int) -> float:
        """Prefer blues whose omission is near the natural 16-ball cycle.

        The previous single-blue objective over-rewarded very recent repeats.
        In validation the blue candidate pool had high recall but top-1/top-5
        ranking was weak, so this term treats omission as a balance signal:
        neither immediate repeat nor very stale cold number should dominate.
        """

        gap = current_gap(number)
        if gap is None:
            return 0.0
        target_gap = 16.0
        ratio = max(float(gap), 1.0) / target_gap
        return float(exp(-abs(log(ratio))))

    def immediate_repeat_score(number: int) -> float:
        gap = current_gap(number)
        if gap is None:
            return 0.0
        return 1.0 / (float(gap) ** 0.65)

    def recent_frequency_score(number: int) -> float:
        if not history or not windows:
            return 0.0
        parts: list[float] = []
        for window in windows:
            subset = history[-window:]
            if not subset:
                continue
            observed = subset.count(int(number)) / float(len(subset))
            # 1/16 is the no-signal baseline.  Cap at 2x baseline so one
            # unusually hot window cannot swamp all model evidence.
            parts.append(min(observed / (1.0 / 16.0), 2.0) / 2.0)
        return float(sum(parts) / len(parts)) if parts else 0.0

    def objective(number: int) -> float:
        edge_probability_weight = 0.10 if edge_state == "NO_EDGE" else 0.35 if edge_state == "WEAK_EDGE" else 1.0
        evidence = edge_probability_weight * float(base_scores.get(number, 0.0)) / float(max_base)
        specialist = float(specialist_bonus.get(number, 0)) / float(max_specialist)
        in_prediction_set = 1.0 if number in set(prediction_set) else 0.0
        return (
            1.35 * in_prediction_set
            + float(recent_gap_weight) * recent_gap_score(number)
            + float(recent_frequency_weight) * recent_frequency_score(number)
            + float(evidence_weight) * evidence
            + float(specialist_weight) * specialist
            + float(omission_balance_weight) * omission_balance_score(number)
            - float(immediate_repeat_penalty) * immediate_repeat_score(number)
        )

    ordered_pool = prediction_set + [
        number for number in range(1, 17)
        if number not in set(prediction_set)
    ]
    probability_order = sorted(
        prediction_set,
        key=lambda number: -float(fusion_scores.get(number, base_scores.get(number, 0.0))),
    )
    tier_a = probability_order[: max(1, min(2, len(probability_order)))]
    tier_b = probability_order[max(1, min(2, len(probability_order))): max(1, min(2, len(probability_order))) + max(2, len(probability_order) // 3)]
    tier_c = [number for number in probability_order if number not in set(tier_a) | set(tier_b)]
    tier_quota = (
        {"A": 2, "B": 2, "C": 1}
        if edge_state in {"STRONG_EDGE", "WEAK_EDGE"}
        else {"A": 1, "B": 2, "C": 2}
    )
    selected: list[int] = []

    def append_from(source: list[int], limit: int) -> None:
        nonlocal selected
        for number in sorted(
            source,
            key=lambda item: (-objective(item), -float(base_scores.get(item, 0.0)), item),
        ):
            if number in selected:
                continue
            selected.append(number)
            if len(selected) >= int(limit):
                break

    append_from(tier_a, min(int(count), tier_quota["A"]))
    append_from(tier_b, min(int(count), len(selected) + tier_quota["B"]))
    append_from(tier_c, min(int(count), len(selected) + tier_quota["C"]))
    if len(selected) < int(count):
        append_from(ordered_pool, int(count))
    ordered = sorted(
        ordered_pool,
        key=lambda number: (
            -objective(number),
            -float(base_scores.get(number, 0.0)),
            number,
        ),
    )
    for number in ordered:
        if number not in selected:
            selected.append(number)
        if len(selected) >= int(count):
            break
    return sorted(selected[: max(1, int(count))])


def _select_blue_numbers_for_top1(*args: Any, **kwargs: Any) -> list[int]:
    """Backward-compatible alias; implementation is Prediction Set + Hit@5."""

    return _select_blue_numbers_for_prediction_set(*args, **kwargs)


def _apply_strong_red_evidence_rescue(
    plan: dict[str, Any],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep strong red evidence from being lost in the final budget compression.

    Recent validation traces showed many missed red numbers had excellent
    candidate-pool rank and support but were squeezed out by overlap/diversity.
    This rescue step is deliberately conservative: it only changes the budget red
    set when an outside number has materially stronger evidence than the
    weakest retained non-dan number.
    """

    if not plan:
        return plan
    red_dan = _as_int_list(plan.get("red_dan", []), low=1, high=33)
    red_tuo = _as_int_list(plan.get("red_tuo", []), low=1, high=33)
    red_count = len(red_dan) + len(red_tuo)
    if red_count < 6:
        return plan

    method_support = _source_method_support(tickets, color="red", low=1, high=33)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_red",
        key_scores="top_red_scores",
        low=1,
        high=33,
    )
    scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    support_map = {number: support for number, _, support, _ in ranked}
    rank_map = {number: rank for number, _, _, rank in ranked}
    retained = set(red_dan) | set(red_tuo)
    protected = set(red_dan)
    max_support = max(support_map.values(), default=1) or 1

    rescue_candidates = [
        number
        for number in range(1, 34)
        if number not in retained
        and (
            (
                int(rank_map.get(number, 999)) <= 10
                and len(method_support.get(number, set())) >= 3
            )
            or
            support_map.get(number, 0) >= 0.58 * max_support
            or (
                len(method_support.get(number, set())) >= 4
                and support_map.get(number, 0) >= 0.35 * max_support
            )
        )
    ]
    rescue_candidates = sorted(
        rescue_candidates,
        key=lambda number: (
            -scores.get(number, 0.0),
            int(rank_map.get(number, 999)),
            -support_map.get(number, 0),
            number,
        ),
    )
    if not rescue_candidates:
        return plan

    replacements: list[dict[str, Any]] = []
    mutable_tuo = list(red_tuo)
    for candidate in rescue_candidates[:7]:
        current = set(red_dan) | set(mutable_tuo)
        if candidate in current:
            continue
        replaceable = [number for number in mutable_tuo if number not in protected]
        if not replaceable:
            break
        weakest = min(
            replaceable,
            key=lambda number: (
                scores.get(number, 0.0),
                support_map.get(number, 0),
                -int(rank_map.get(number, 999)),
                -number,
            ),
        )
        gain = scores.get(candidate, 0.0) - scores.get(weakest, 0.0)
        candidate_methods = len(method_support.get(candidate, set()))
        weakest_methods = len(method_support.get(weakest, set()))
        candidate_support = int(support_map.get(candidate, 0))
        weakest_support = int(support_map.get(weakest, 0))
        candidate_rank = int(rank_map.get(candidate, 999))
        strongest_ranked_candidate = (
            candidate_rank <= 10
            and candidate_methods >= max(3, weakest_methods)
            and candidate_support >= max(1, int(0.78 * max(1, weakest_support)))
        )
        support_is_safe = (
            candidate_support >= weakest_support
            or strongest_ranked_candidate
            or (
                candidate_methods > weakest_methods
                and candidate_support >= 0.82 * max(1, weakest_support)
                and gain >= -0.08
            )
        )
        if not support_is_safe:
            continue
        if gain < -0.16 and not strongest_ranked_candidate:
            continue
        if gain < 0.10 and candidate_methods <= weakest_methods and not strongest_ranked_candidate:
            continue
        mutable_tuo[mutable_tuo.index(weakest)] = candidate
        replacements.append({
            "added": int(candidate),
            "removed": int(weakest),
            "added_support": int(support_map.get(candidate, 0)),
            "removed_support": int(support_map.get(weakest, 0)),
            "added_rank": int(rank_map.get(candidate, 999)),
            "removed_rank": int(rank_map.get(weakest, 999)),
            "added_methods": sorted(method_support.get(candidate, set())),
            "removed_methods": sorted(method_support.get(weakest, set())),
            "score_gain": float(gain),
        })

    if not replacements:
        return plan
    updated = dict(plan)
    updated["red_tuo"] = sorted(mutable_tuo)
    updated["red_evidence_rescue"] = {
        "active": True,
        "replacements": replacements,
        "reason": "预算胆拖筛选层启用强证据槽位保护：候选池Top10且多方法共同支持的红球，不再因为轻微分数差或覆盖分散被轻易挤掉。",
    }
    updated["selection_reason"] = (
        str(updated.get("selection_reason", ""))
        + " 已启用强证据红球救援：优先保留候选排名靠前、出现次数高、跨方法支持更强的红球。"
    ).strip()
    updated["hit_summary"] = _hit_summary(updated, actual)
    return updated


def _ticket_source_method(ticket: dict[str, Any]) -> str:
    role = str(ticket.get("selection_role", "") or "")
    if role.startswith("dantuo_method_source_"):
        return role.replace("dantuo_method_source_", "", 1)
    if role.startswith("dantuo_candidate_pool_"):
        return "candidate_pool"
    return role or "final_layer"


def _ticket_method_priority(method: str) -> float:
    priorities = {
        "red_ge3_objective": 1.45,
        "global_coverage": 1.36,
        "pareto": 1.30,
        "historical_hit": 1.18,
        "greedy": 1.08,
        "genetic": 1.05,
        "consensus": 1.00,
        "final_layer": 1.00,
        "candidate_pool": 1.12,
    }
    return float(priorities.get(str(method), 0.95))


def _apply_candidate_overlap_rescue(
    plan: dict[str, Any],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    actual: dict[str, Any] | None,
    min_ticket_overlap: int = 4,
    max_replacements: int = 5,
) -> dict[str, Any]:
    """Protect strong whole-ticket evidence before the final budget output.

    Single-number rescue is not enough when the trace shows "this number was in
    a good candidate ticket, but that whole ticket only overlapped the retained
    budget red set by one or two balls".  This pass nudges the final red set toward
    the strongest such candidate tickets while keeping the budget and the red
    count unchanged.
    """

    if not plan or not tickets:
        return plan
    red_dan = _as_int_list(plan.get("red_dan", []), low=1, high=33)
    red_tuo = _as_int_list(plan.get("red_tuo", []), low=1, high=33)
    red_count = len(red_dan) + len(red_tuo)
    if red_count < 7 or max_replacements <= 0:
        return plan

    method_support = _source_method_support(tickets, color="red", low=1, high=33)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_red",
        key_scores="top_red_scores",
        low=1,
        high=33,
    )
    scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    support_map = {number: support for number, _, support, _ in ranked}
    rank_map = {number: rank for number, _, _, rank in ranked}
    max_support = max(support_map.values(), default=1) or 1
    max_ticket_score = max(float(ticket.get("consensus_score", 0.0) or 0.0) for ticket in tickets) or 1.0

    def ticket_strength(ticket: dict[str, Any]) -> float:
        red = _as_int_list(ticket.get("red", []), low=1, high=33)
        if len(red) != 6:
            return -1.0
        method = _ticket_source_method(ticket)
        evidence = sum(scores.get(number, 0.0) for number in red) / 6.0
        support = sum(support_map.get(number, 0) for number in red) / 6.0
        model_diversity = sum(len(method_support.get(number, set())) for number in red) / 6.0
        row_score = float(ticket.get("consensus_score", 0.0) or 0.0) / max_ticket_score
        return (
            1.35 * _ticket_method_priority(method)
            + 1.10 * evidence
            + 0.42 * support
            + 0.24 * model_diversity
            + 0.30 * row_score
        )

    current = set(red_dan) | set(red_tuo)
    scored_tickets = sorted(
        [
            (ticket_strength(ticket), ticket)
            for ticket in tickets
            if len(_as_int_list(ticket.get("red", []), low=1, high=33)) == 6
        ],
        key=lambda item: (-item[0], _ticket_source_method(item[1])),
    )
    if not scored_tickets:
        return plan

    strongest_score = scored_tickets[0][0]
    mutable_tuo = list(red_tuo)
    replacements: list[dict[str, Any]] = []
    touched_tickets: list[dict[str, Any]] = []

    for strength, ticket in scored_tickets[:14]:
        if len(replacements) >= max_replacements:
            break
        if strength < 0.80 * strongest_score:
            break
        ticket_red = set(_as_int_list(ticket.get("red", []), low=1, high=33))
        current = set(red_dan) | set(mutable_tuo)
        overlap = len(ticket_red & current)
        if overlap >= min_ticket_overlap:
            continue

        additions = sorted(
            [number for number in ticket_red if number not in current],
            key=lambda number: (
                -scores.get(number, 0.0),
                -support_map.get(number, 0),
                int(rank_map.get(number, 999)),
                number,
            ),
        )
        for candidate in additions:
            if len(replacements) >= max_replacements:
                break
            current = set(red_dan) | set(mutable_tuo)
            if len(ticket_red & current) >= min_ticket_overlap:
                break
            replaceable = [number for number in mutable_tuo if number not in ticket_red]
            if not replaceable:
                break
            weakest = min(
                replaceable,
                key=lambda number: (
                    scores.get(number, 0.0),
                    support_map.get(number, 0),
                    len(method_support.get(number, set())),
                    -int(rank_map.get(number, 999)),
                    -number,
                ),
            )

            candidate_score = scores.get(candidate, 0.0)
            weakest_score = scores.get(weakest, 0.0)
            candidate_methods = len(method_support.get(candidate, set()))
            weakest_methods = len(method_support.get(weakest, set()))
            candidate_support = int(support_map.get(candidate, 0))
            weakest_support = int(support_map.get(weakest, 0))
            candidate_rank = int(rank_map.get(candidate, 999))
            strong_candidate_qualified = (
                candidate_rank <= 10
                or candidate_methods >= 2
                or candidate_support >= max(3, int(0.72 * max_support))
            )
            if not strong_candidate_qualified:
                continue
            support_is_credible = (
                candidate_support >= max(1, int(0.62 * max(1, weakest_support)))
                or candidate_methods > weakest_methods
                or candidate_rank <= 10
            )
            evidence_loss_is_controlled = (
                candidate_score >= 0.78 * max(weakest_score, 0.0001)
                or (
                    candidate_methods >= weakest_methods
                    and candidate_support >= max(1, int(0.74 * max(1, weakest_support)))
                )
            )
            if not (support_is_credible and evidence_loss_is_controlled):
                continue

            before_overlap = len(ticket_red & current)
            mutable_tuo[mutable_tuo.index(weakest)] = int(candidate)
            after_overlap = len(ticket_red & (set(red_dan) | set(mutable_tuo)))
            method = _ticket_source_method(ticket)
            replacements.append({
                "ticket_method": method,
                "ticket_strength": float(strength),
                "added": int(candidate),
                "removed": int(weakest),
                "overlap_before": int(before_overlap),
                "overlap_after": int(after_overlap),
                "added_support": int(candidate_support),
                "removed_support": int(weakest_support),
                "added_rank": int(candidate_rank),
                "removed_rank": int(rank_map.get(weakest, 999)),
                "added_methods": sorted(method_support.get(candidate, set())),
                "removed_methods": sorted(method_support.get(weakest, set())),
            })
            touched_tickets.append({
                "method": method,
                "red": sorted(ticket_red),
                "strength": float(strength),
                "overlap_before": int(before_overlap),
                "overlap_after": int(after_overlap),
            })

    if not replacements:
        return plan
    updated = dict(plan)
    updated["red_tuo"] = sorted(mutable_tuo)
    updated["candidate_overlap_rescue"] = {
        "active": True,
        "target_min_overlap": int(min_ticket_overlap),
        "max_replacements": int(max_replacements),
        "replacements": replacements,
        "touched_tickets": touched_tickets,
        "reason": "预算胆拖筛选层新增整票重叠保护：当高价值候选票与最终红球集合重叠不足4个时，用其强证据号码桥接替换证据较弱的拖码，减少命中号码被分散到不同候选票后全部落空的情况。",
    }
    updated["selection_reason"] = (
        str(updated.get("selection_reason", ""))
        + " 已启用整票重叠保护：优先让7个红球覆盖高价值候选票中的至少3个核心交集。"
    ).strip()
    updated["hit_summary"] = _hit_summary(updated, actual)
    return updated


def _finalize_dantuo_plan(
    plan: dict[str, Any],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> dict[str, Any]:
    rescued = _apply_strong_red_evidence_rescue(
        plan,
        ranked=ranked,
        tickets=tickets,
        fusion=fusion,
        actual=actual,
    )
    return _apply_candidate_overlap_rescue(
        rescued,
        ranked=ranked,
        tickets=tickets,
        fusion=fusion,
        actual=actual,
    )


def _red_zone(number: int) -> int:
    if int(number) <= 11:
        return 1
    if int(number) <= 22:
        return 2
    return 3


def _typical_span_bonus(span: int) -> float:
    return float(exp(-0.5 * ((float(span) - 16.0) / 7.0) ** 2))


def _boundary_penalty(number: int) -> float:
    return 0.75 if int(number) in {1, 2, 32, 33} else 0.0


def _select_red_dan(
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    dan_count: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    ranked_numbers = _ranked_numbers(ranked)
    if int(dan_count) != 2:
        return sorted(ranked_numbers[: int(dan_count)]), []

    support_map = {number: support for number, _, support, _ in ranked}
    score_map = {number: score for number, score, _, _ in ranked}
    max_support = max(support_map.values(), default=1) or 1
    candidate_numbers = ranked_numbers[: min(10, len(ranked_numbers))]
    if len(candidate_numbers) < 2:
        return sorted(ranked_numbers[:2]), []

    ticket_sets = [set(_as_int_list(ticket.get("red", []), low=1, high=33)) for ticket in tickets]
    max_score = max(score_map.values(), default=1.0) or 1.0
    best_pair: tuple[float, tuple[int, int]] | None = None
    pair_rows: list[dict[str, Any]] = []
    for left, right in combinations(candidate_numbers, 2):
        cooccurrence = sum(1 for numbers in ticket_sets if left in numbers and right in numbers)
        same_zone = _red_zone(left) == _red_zone(right)
        zone_bonus = 1.10 if not same_zone else -1.10
        span = abs(int(right) - int(left))
        near_neighbor_penalty = 1.20 if span <= 4 else 0.55 if span <= 7 else 0.0
        support_score = (support_map.get(left, 0) + support_map.get(right, 0)) / (2.0 * max_support)
        rank_score = (score_map.get(left, 0.0) + score_map.get(right, 0.0)) / (2.0 * max_score)
        weak_support_penalty = 0.65 if min(support_map.get(left, 0), support_map.get(right, 0)) < max(2, max_support - 1) else 0.0
        total = (
            1.55 * support_score
            + 0.75 * (cooccurrence / max(len(ticket_sets), 1))
            + 0.95 * rank_score
            + zone_bonus
            + 0.80 * _typical_span_bonus(span)
            - _boundary_penalty(left)
            - _boundary_penalty(right)
            - near_neighbor_penalty
            - weak_support_penalty
        )
        row = {
            "pair": [int(left), int(right)],
            "score": float(total),
            "support": [int(support_map.get(left, 0)), int(support_map.get(right, 0))],
            "cooccurrence": int(cooccurrence),
            "zone_diverse": not same_zone,
            "span": int(span),
        }
        pair_rows.append(row)
        if best_pair is None or total > best_pair[0]:
            best_pair = (total, (int(left), int(right)))

    pair_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return sorted(best_pair[1]) if best_pair else sorted(ranked_numbers[:2]), pair_rows[:5]


def _normalize_final_recommendation_tickets(
    final_recommendation: Iterable[dict[str, Any]] | None,
    normalized_tickets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the real final portfolio tickets used as dantuo compression source."""

    explicit: list[dict[str, Any]] = []
    for ticket in list(final_recommendation or []):
        red = sorted(_as_int_list(ticket.get("red", []), low=1, high=33))
        blue_values = _as_int_list([ticket.get("blue")], low=1, high=16)
        if len(red) == 6 and blue_values:
            explicit.append({
                "red": red,
                "blue": blue_values[0],
                "consensus_score": float(ticket.get("consensus_score", 0.0) or 0.0),
                "selection_role": str(ticket.get("selection_role", "") or "final_layer"),
                "blue_selection_role": str(ticket.get("blue_selection_role", "") or ""),
            })
    if explicit:
        return explicit

    inferred = []
    for ticket in normalized_tickets:
        role = str(ticket.get("selection_role", "") or "")
        if role.startswith("dantuo_method_source_") or role.startswith("dantuo_candidate_pool_"):
            continue
        inferred.append(ticket)
    return inferred


def _final_red_frequency(final_tickets: list[dict[str, Any]]) -> Counter[int]:
    frequency: Counter[int] = Counter()
    for ticket in final_tickets:
        frequency.update(_as_int_list(ticket.get("red", []), low=1, high=33))
    return frequency


def _number_rank_position(ranked: list[tuple[int, float, int, int]]) -> dict[int, int]:
    return {int(number): index + 1 for index, (number, _, _, _) in enumerate(ranked)}


def _is_final_high_value_number(
    number: int,
    *,
    final_frequency: Counter[int],
    method_support: dict[int, set[str]],
    ranked_position: dict[int, int],
) -> bool:
    frequency = int(final_frequency.get(int(number), 0))
    methods = {
        method
        for method in method_support.get(int(number), set())
        if method not in {"final_layer", "candidate_pool"}
    }
    if frequency >= 3:
        return True
    if frequency >= 2 and (len(methods) >= 2 or int(ranked_position.get(int(number), 999)) <= 10):
        return True
    if frequency >= 1 and len(methods) >= 3 and int(ranked_position.get(int(number), 999)) <= 8:
        return True
    return False


def _pack_final_pair_52(
    final_sets: list[set[int]],
    *,
    target_count: int,
    final_frequency: Counter[int],
    method_support: dict[int, set[str]],
    evidence_scores: dict[int, float],
    ranked_position: dict[int, int],
    priority_numbers: set[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Build the budget 7-red set as Method pair 5+2 from Final5 tickets.

    Consensus-frequency packing and fusion-seeded overlap bridges both flattened
    ticket structure and underperformed random-from-union.  This keeps 5 reds from
    the Final5 ticket with the most mid-frequency numbers, then adds 2 from a
    complementary ticket.  Independent-model priority numbers only break ties.
    """

    target_count = max(6, int(target_count))
    usable = [set(int(number) for number in ticket) for ticket in final_sets if len(ticket) >= 6]
    if not usable:
        return [], {"active": False, "reason": "no_final_tickets"}
    priority_set = {int(number) for number in (priority_numbers or set())}

    def seed_score(ticket: set[int]) -> tuple[int, int, int, int]:
        mid = sum(1 for number in ticket if 2 <= int(final_frequency.get(number, 0)) <= 3)
        priority_overlap = sum(1 for number in ticket if number in priority_set)
        vote = sum(
            len({method for method in method_support.get(number, set()) if method not in {"final_layer", "candidate_pool"}})
            for number in ticket
        )
        consensus = sum(1 for number in ticket if int(final_frequency.get(number, 0)) >= 4)
        return (int(mid), int(priority_overlap), int(vote), -int(consensus))

    ranked_tickets = sorted(usable, key=seed_score, reverse=True)
    seed = ranked_tickets[0]

    def keep_key(number: int) -> tuple[int, int, int, int]:
        in_priority = 1 if int(number) in priority_set else 0
        frequency = int(final_frequency.get(number, 0))
        mid = 1 if 2 <= frequency <= 3 else 0
        rare = 1 if frequency == 1 else 0
        return (in_priority, mid, rare, -int(ranked_position.get(number, 99)))

    keep5 = {int(number) for number in sorted(seed, key=keep_key, reverse=True)[:5]}
    selected = set(keep5)
    second: set[int] | None = None
    added: list[int] = []
    if len(ranked_tickets) >= 2:
        second = max(ranked_tickets[1:], key=lambda ticket: (len(ticket - selected), seed_score(ticket)))
        for number in sorted(second - selected, key=keep_key, reverse=True):
            selected.add(int(number))
            added.append(int(number))
            if len(selected) >= target_count:
                break
    union = set().union(*usable)
    filled: list[int] = []
    for number in sorted(union - selected, key=keep_key, reverse=True):
        if len(selected) >= target_count:
            break
        selected.add(int(number))
        filled.append(int(number))
    return sorted(selected)[:target_count], {
        "active": True,
        "seed_ticket": sorted(seed),
        "keep5": sorted(keep5),
        "second_ticket": sorted(second) if second else [],
        "added_from_second": added,
        "filled": filled,
        "rule": "Final5 Method pair 5+2：中频号最多的一注留5红，再从互补票补2红，禁止完整克隆6红。",
    }


def _optimize_final_portfolio_red_set(
    final_tickets: list[dict[str, Any]],
    *,
    target_count: int,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    red_priority_numbers: Iterable[Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Compress Final5 red union to dantuo red slots by preserving portfolio information."""

    target_count = max(6, int(target_count))
    final_frequency = _final_red_frequency(final_tickets)
    final_sets = [set(_as_int_list(ticket.get("red", []), low=1, high=33)) for ticket in final_tickets]
    final_union = sorted(final_frequency)
    ranked_numbers = _ranked_numbers(ranked)
    method_support = _source_method_support(tickets, color="red", low=1, high=33)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_red",
        key_scores="top_red_scores",
        low=1,
        high=33,
    )
    evidence_scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    ranked_position = _number_rank_position(ranked)

    candidate_numbers: list[int] = []
    seen: set[int] = set()
    for number in sorted(
        final_union,
        key=lambda value: (
            -int(final_frequency.get(value, 0)),
            -float(evidence_scores.get(value, 0.0)),
            int(ranked_position.get(value, 999)),
            value,
        ),
    ):
        candidate_numbers.append(int(number))
        seen.add(int(number))
    for number in ranked_numbers:
        if len(candidate_numbers) >= max(target_count, 18):
            break
        if number not in seen:
            candidate_numbers.append(int(number))
            seen.add(int(number))

    if len(candidate_numbers) <= target_count:
        selected = sorted(candidate_numbers[:target_count])
        return selected, {
            "strategy": "final_portfolio_direct_fill",
            "evaluated_sets": 1,
            "final_union": final_union,
            "selected_score": None,
        }

    max_frequency_total = sum(
        sorted((int(final_frequency.get(number, 0)) for number in candidate_numbers), reverse=True)[:target_count]
    ) or 1
    max_evidence_total = sum(
        sorted((float(evidence_scores.get(number, 0.0)) for number in candidate_numbers), reverse=True)[:target_count]
    ) or 1.0
    high_value_numbers = {
        number
        for number in final_union
        if _is_final_high_value_number(
            number,
            final_frequency=final_frequency,
            method_support=method_support,
            ranked_position=ranked_position,
        )
    }

    best: tuple[float, tuple[int, ...], dict[str, Any]] | None = None
    evaluated = 0
    for combo in combinations(candidate_numbers, target_count):
        selected_set = set(int(number) for number in combo)
        evaluated += 1
        ticket_hits = [len(ticket_set & selected_set) for ticket_set in final_sets] or [0]
        coverage_score = sum(ticket_hits) / max(1.0, 6.0 * len(final_sets))
        best_ticket_hit = max(ticket_hits) / 6.0
        min_ticket_coverage = min(ticket_hits) / 6.0
        frequency_score = sum(int(final_frequency.get(number, 0)) for number in selected_set) / max_frequency_total
        evidence_score = sum(float(evidence_scores.get(number, 0.0)) for number in selected_set) / max_evidence_total
        high_value_retention = (
            len(high_value_numbers & selected_set) / len(high_value_numbers)
            if high_value_numbers
            else 1.0
        )
        zones = len({_red_zone(number) for number in selected_set}) / 3.0
        final_union_share = len(set(final_union) & selected_set) / max(1, target_count)
        outside_penalty = len([number for number in selected_set if number not in set(final_union)]) / target_count
        total = (
            3.40 * coverage_score
            + 1.20 * best_ticket_hit
            + 0.95 * min_ticket_coverage
            + 1.35 * frequency_score
            + 0.70 * evidence_score
            + 1.85 * high_value_retention
            + 0.25 * zones
            + 0.80 * final_union_share
            - 1.20 * outside_penalty
        )
        detail = {
            "coverage_score": float(coverage_score),
            "best_ticket_hit_proxy": float(best_ticket_hit),
            "min_ticket_coverage": float(min_ticket_coverage),
            "frequency_score": float(frequency_score),
            "evidence_score": float(evidence_score),
            "high_value_retention": float(high_value_retention),
            "zone_diversity": float(zones),
            "final_union_share": float(final_union_share),
        }
        if best is None or total > best[0]:
            best = (float(total), tuple(sorted(selected_set)), detail)

    selected = sorted(best[1]) if best else sorted(candidate_numbers[:target_count])
    return selected, {
        "strategy": "final_portfolio_coverage_objective",
        "evaluated_sets": int(evaluated),
        "final_union": final_union,
        "high_value_numbers": sorted(high_value_numbers),
        "selected_score": float(best[0]) if best else None,
        "score_detail": best[2] if best else {},
    }


def _select_final_portfolio_dan(
    selected_red: list[int],
    final_tickets: list[dict[str, Any]],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    dan_count: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Pick red-dan as coverage anchors from the already selected Final5 compression set.

    A dan number is not interpreted as the single strongest prediction.  It is
    the fixed anchor that preserves the most Final5 information under the
    current budget layout.
    """

    dan_count = max(0, int(dan_count))
    if dan_count <= 0:
        return [], []
    selected_red = sorted(_as_int_list(selected_red, low=1, high=33))
    if len(selected_red) <= dan_count:
        return selected_red, []

    final_frequency = _final_red_frequency(final_tickets)
    final_sets = [set(_as_int_list(ticket.get("red", []), low=1, high=33)) for ticket in final_tickets]
    method_support = _source_method_support(tickets, color="red", low=1, high=33)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_red",
        key_scores="top_red_scores",
        low=1,
        high=33,
    )
    evidence_scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    ranked_position = _number_rank_position(ranked)
    max_frequency = max((int(final_frequency.get(number, 0)) for number in selected_red), default=1) or 1
    max_evidence = max((float(evidence_scores.get(number, 0.0)) for number in selected_red), default=1.0) or 1.0

    best: tuple[float, tuple[int, ...]] | None = None
    candidate_rows: list[dict[str, Any]] = []
    for combo in combinations(selected_red, dan_count):
        combo_set = set(combo)
        ticket_spread = sum(1 for ticket_set in final_sets if ticket_set & combo_set) / max(1, len(final_sets))
        cooccurrence = sum(1 for ticket_set in final_sets if combo_set <= ticket_set) / max(1, len(final_sets))
        frequency_score = sum(int(final_frequency.get(number, 0)) for number in combo) / (dan_count * max_frequency)
        evidence_score = sum(float(evidence_scores.get(number, 0.0)) for number in combo) / (dan_count * max_evidence)
        weak_fixed_count = sum(
            1
            for number in combo
            if int(final_frequency.get(number, 0)) < 2
            and int(ranked_position.get(number, 999)) > 8
            and len(method_support.get(number, set()) - {"final_layer"}) < 2
        )
        same_zone_penalty = 0.18 if len({_red_zone(number) for number in combo}) == 1 and dan_count >= 2 else 0.0
        near_penalty = 0.0
        for left, right in combinations(combo, 2):
            if abs(int(left) - int(right)) <= 3:
                near_penalty += 0.08
        total = (
            1.65 * frequency_score
            + 0.80 * ticket_spread
            + 0.35 * cooccurrence
            + 0.95 * evidence_score
            - 0.90 * weak_fixed_count
            - same_zone_penalty
            - near_penalty
        )
        row = {
            "pair" if dan_count == 2 else "numbers": [int(number) for number in combo],
            "score": float(total),
            "frequency": [int(final_frequency.get(number, 0)) for number in combo],
            "ticket_spread": float(ticket_spread),
            "cooccurrence": float(cooccurrence),
            "evidence": [float(evidence_scores.get(number, 0.0)) for number in combo],
            "fixed_risk_count": int(weak_fixed_count),
        }
        candidate_rows.append(row)
        if best is None or total > best[0]:
            best = (float(total), tuple(int(number) for number in combo))

    candidate_rows.sort(key=lambda item: float(item["score"]), reverse=True)
    if best is None:
        return sorted(selected_red[:dan_count]), candidate_rows[:5]
    return sorted(best[1]), candidate_rows[:5]


def _build_final_to_dantuo_audit(
    plan: dict[str, Any],
    final_tickets: list[dict[str, Any]],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
) -> dict[str, Any]:
    final_frequency = _final_red_frequency(final_tickets)
    final_union = sorted(final_frequency)
    dantuo_union = sorted(
        set(_as_int_list(plan.get("red_dan", []), low=1, high=33))
        | set(_as_int_list(plan.get("red_tuo", []), low=1, high=33))
    )
    dantuo_set = set(dantuo_union)
    method_support = _source_method_support(tickets, color="red", low=1, high=33)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_red",
        key_scores="top_red_scores",
        low=1,
        high=33,
    )
    evidence_scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    ranked_position = _number_rank_position(ranked)
    high_value_numbers = {
        number
        for number in final_union
        if _is_final_high_value_number(
            number,
            final_frequency=final_frequency,
            method_support=method_support,
            ranked_position=ranked_position,
        )
    }

    def support_methods(number: int) -> list[str]:
        return sorted(method for method in method_support.get(int(number), set()) if method != "final_layer")

    def max_overlap_with_kept(number: int) -> int:
        containing = [
            set(_as_int_list(ticket.get("red", []), low=1, high=33))
            for ticket in final_tickets
            if int(number) in set(_as_int_list(ticket.get("red", []), low=1, high=33))
        ]
        if not containing:
            return 0
        return max(len(ticket_set & dantuo_set) for ticket_set in containing)

    retained_rows = []
    for number in sorted(set(final_union) & dantuo_set):
        retained_rows.append({
            "number": int(number),
            "final_ticket_frequency": int(final_frequency.get(number, 0)),
            "rank": int(ranked_position.get(number, 999)),
            "model_support": support_methods(number),
            "reason": (
                f"保留：最终5注出现{int(final_frequency.get(number, 0))}次，"
                f"模型/方法支持={','.join(support_methods(number)) or '无'}，"
                f"进入2胆5拖后仍代表Final5覆盖信息。"
            ),
        })

    dropped_rows = []
    for number in sorted(set(final_union) - dantuo_set):
        methods = support_methods(number)
        overlap = max_overlap_with_kept(number)
        dropped_rows.append({
            "number": int(number),
            "final_ticket_frequency": int(final_frequency.get(number, 0)),
            "rank": int(ranked_position.get(number, 999)),
            "evidence_score": float(evidence_scores.get(number, 0.0)),
            "model_support": methods,
            "max_overlap_with_retained_final_ticket": int(overlap),
            "high_value": int(number) in high_value_numbers,
            "reason": (
                f"未进入预算胆拖红球集合：该号在最终5注出现{int(final_frequency.get(number, 0))}次；"
                f"在Final5覆盖压缩中，与已保留号码所在最终票最高重叠{overlap}个红球，"
                f"边际覆盖收益低于已保留号码；模型/方法支持={','.join(methods) or '无'}，"
                f"综合排序#{int(ranked_position.get(number, 999))}。"
            ),
        })

    high_value_retained = sorted(high_value_numbers & dantuo_set)
    return {
        "schema": "final_to_dantuo_audit_v1",
        "enabled": True,
        "source": "final_portfolio_to_dantuo_only",
        "final_union": final_union,
        "final_union_count": int(len(final_union)),
        "dantuo_union": dantuo_union,
        "dantuo_union_count": int(len(dantuo_union)),
        "retained": sorted(set(final_union) & dantuo_set),
        "dropped": sorted(set(final_union) - dantuo_set),
        "retention": len(set(final_union) & dantuo_set) / max(1, len(final_union)),
        "high_value_numbers": sorted(high_value_numbers),
        "high_value_retained": high_value_retained,
        "high_value_retention": len(high_value_retained) / max(1, len(high_value_numbers)),
        "retained_detail": retained_rows,
        "dropped_detail": dropped_rows,
    }


def _final_blue_frequency(final_tickets: list[dict[str, Any]]) -> Counter[int]:
    frequency: Counter[int] = Counter()
    for ticket in final_tickets:
        values = _as_int_list([ticket.get("blue")], low=1, high=16)
        if values:
            frequency[values[0]] += 1
    return frequency


def _final_blue_evidence(
    final_tickets: list[dict[str, Any]],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    blue_history: Iterable[Any] | None = None,
    omission_balance_weight: float = 0.80,
    immediate_repeat_penalty: float = 0.35,
) -> dict[int, dict[str, Any]]:
    final_frequency = _final_blue_frequency(final_tickets)
    method_support = _source_method_support(tickets, color="blue", low=1, high=16)
    fusion_scores = _fusion_score_map(
        fusion,
        key_numbers="top_blue",
        key_scores="top_blue_scores",
        low=1,
        high=16,
    )
    evidence_scores = _evidence_score_map(ranked, method_support, fusion_scores=fusion_scores)
    ranked_position = _number_rank_position(ranked)
    max_frequency = max(final_frequency.values(), default=1) or 1
    max_evidence = max((float(evidence_scores.get(number, 0.0)) for number in range(1, 17)), default=1.0) or 1.0
    rows: dict[int, dict[str, Any]] = {}
    for number in sorted(final_frequency):
        methods = sorted(method for method in method_support.get(int(number), set()) if method != "final_layer")
        final_positions = [
            int(index)
            for index, ticket in enumerate(final_tickets, start=1)
            if _as_int_list([ticket.get("blue")], low=1, high=16) == [int(number)]
        ]
        specialist_support = sum(
            1
            for ticket in tickets
            if _as_int_list([ticket.get("blue")], low=1, high=16) == [int(number)]
            and "blue_specialist" in str(ticket.get("blue_selection_role", "") or "")
        )
        frequency_score = float(final_frequency.get(number, 0)) / float(max_frequency)
        evidence_score = float(evidence_scores.get(number, 0.0)) / float(max_evidence)
        method_score = min(1.0, len(methods) / 3.0)
        rank_score = 1.0 / (1.0 + float(ranked_position.get(number, 16)))
        gap = _blue_gap_since_last(blue_history, int(number))
        omission_score = _blue_omission_balance_score(gap)
        repeat_score = _blue_immediate_repeat_score(gap)
        score = (
            3.20 * frequency_score
            + 0.90 * evidence_score
            + 0.55 * method_score
            + 0.35 * rank_score
            + 0.20 * float(specialist_support)
        )
        rows[int(number)] = {
            "number": int(number),
            "score": float(score),
            "final_frequency": int(final_frequency.get(number, 0)),
            "final_positions": final_positions,
            "rank": int(ranked_position.get(number, 999)),
            "fusion_score": float(fusion_scores.get(number, 0.0)),
            "evidence_score": float(evidence_scores.get(number, 0.0)),
            "omission_gap": gap,
            "omission_score": float(omission_score),
            "immediate_repeat_score": float(repeat_score),
            "model_support": methods,
            "specialist_support": int(specialist_support),
        }
    return rows


def _select_blue_numbers_from_final_evidence(
    final_tickets: list[dict[str, Any]],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    count: int,
    blue_history: Iterable[Any] | None = None,
    omission_balance_weight: float = 0.80,
    immediate_repeat_penalty: float = 0.35,
) -> tuple[list[int], dict[str, Any]]:
    """Compress Final5 blue evidence instead of generating a second blue forecast."""

    count = max(1, int(count))
    evidence = _final_blue_evidence(
        final_tickets,
        ranked=ranked,
        tickets=tickets,
        fusion=fusion,
        blue_history=blue_history,
        omission_balance_weight=omission_balance_weight,
        immediate_repeat_penalty=immediate_repeat_penalty,
    )
    if not evidence:
        return [], {
            "schema": "final_blue_evidence_v1",
            "active": False,
            "reason": "no_final_blue_evidence",
        }
    ordered = sorted(
        evidence.values(),
        key=lambda row: (
            float(row.get("score", 0.0)),
            int(row.get("final_frequency", 0) or 0),
            -int(row.get("rank", 999) or 999),
            -int(row.get("number", 0) or 0),
        ),
        reverse=True,
    )
    selected = [int(row["number"]) for row in ordered[:count]]
    if len(selected) < count:
        for number, _, _, _ in ranked:
            if int(number) not in selected and 1 <= int(number) <= 16:
                selected.append(int(number))
            if len(selected) >= count:
                break
    retained = [int(number) for number in selected if int(number) in evidence]
    dropped = sorted(set(evidence) - set(selected))
    return sorted(selected[:count]), {
        "schema": "final_blue_evidence_v1",
        "active": True,
        "source": "final5_blue_union_only",
        "final_blue_union": sorted(evidence),
        "selected_blue": sorted(selected[:count]),
        "retained": sorted(retained),
        "dropped": dropped,
        "retention": len(set(evidence) & set(selected)) / max(1, len(evidence)),
        "evidence_rows": [evidence[int(number)] for number in sorted(evidence)],
        "rule": "Final5是胆拖蓝球的唯一预测源；预算层只压缩Final5蓝球并集，按出现次数和票面证据排序。",
    }


def _build_blue_final_to_dantuo_audit(
    plan: dict[str, Any],
    final_tickets: list[dict[str, Any]],
    *,
    ranked: list[tuple[int, float, int, int]],
    tickets: list[dict[str, Any]],
    fusion: dict[str, Any] | None,
    blue_history: Iterable[Any] | None = None,
    omission_balance_weight: float = 0.80,
    immediate_repeat_penalty: float = 0.35,
) -> dict[str, Any]:
    evidence = _final_blue_evidence(
        final_tickets,
        ranked=ranked,
        tickets=tickets,
        fusion=fusion,
        blue_history=blue_history,
        omission_balance_weight=omission_balance_weight,
        immediate_repeat_penalty=immediate_repeat_penalty,
    )
    final_union = sorted(evidence)
    dantuo_blue = sorted(_as_int_list(plan.get("blue", []), low=1, high=16))
    dantuo_set = set(dantuo_blue)
    retained = sorted(set(final_union) & dantuo_set)
    dropped = sorted(set(final_union) - dantuo_set)
    retained_detail = []
    for number in retained:
        row = evidence.get(int(number), {})
        retained_detail.append({
            **row,
            "selected_in_dantuo": True,
            "reason": (
                f"保留：蓝球{int(number):02d}已在Final5出现{int(row.get('final_frequency', 0))}次，"
                f"位置={row.get('final_positions', [])}，排序证据#{int(row.get('rank', 999))}。"
            ),
        })
    dropped_detail = []
    best_selected = max(
        (evidence.get(number, {}) for number in retained),
        key=lambda row: float(row.get("score", 0.0)),
        default={},
    )
    for number in dropped:
        row = evidence.get(int(number), {})
        dropped_detail.append({
            **row,
            "selected_in_dantuo": False,
            "drop_stage": "final5_to_dantuo_blue_budget_compression",
            "drop_reason": "budget_competition_after_final5_evidence_freeze",
            "replacement_number": best_selected.get("number"),
            "replacement_advantage": {
                "replacement_score": best_selected.get("score"),
                "dropped_score": row.get("score"),
                "reason": "预算蓝球数量有限，保留Final5蓝球证据分数更高或出现次数更多的号码。",
            },
            "reason": (
                f"未进入预算蓝球：蓝球{int(number):02d}已在Final5出现{int(row.get('final_frequency', 0))}次，"
                f"位置={row.get('final_positions', [])}；但预算只允许{len(dantuo_blue)}个蓝球，"
                f"被Final5证据分数更高的号码替代。"
            ),
        })
    return {
        "schema": "blue_final_to_dantuo_audit_v1",
        "enabled": True,
        "source": "final5_blue_evidence_freeze",
        "final_blue_union": final_union,
        "dantuo_blue": dantuo_blue,
        "retained": retained,
        "dropped": dropped,
        "retention": len(retained) / max(1, len(final_union)),
        "retained_detail": retained_detail,
        "dropped_detail": dropped_detail,
        "rule": "蓝球胆拖不再重新生成Prediction Set；仅压缩Final5已经产生的蓝球证据。",
    }


def _hit_summary(plan: dict[str, Any], actual: dict[str, Any] | None) -> dict[str, Any] | None:
    if not actual:
        return None
    actual_red = set(_as_int_list(actual.get("red", []), low=1, high=33))
    actual_blue_values = _as_int_list([actual.get("blue")], low=1, high=16)
    if not actual_red or not actual_blue_values:
        return None
    actual_blue = actual_blue_values[0]
    dan_hits = sorted(set(plan["red_dan"]) & actual_red)
    tuo_hits = sorted(set(plan["red_tuo"]) & actual_red)
    return {
        "actual_red": sorted(actual_red),
        "actual_blue": actual_blue,
        "red_dan_hits": dan_hits,
        "red_tuo_hits": tuo_hits,
        "red_union_hits": sorted(set(dan_hits) | set(tuo_hits)),
        "blue_hit": actual_blue in set(plan["blue"]),
    }


def _is_coverage_like_portfolio(tickets: list[dict[str, Any]]) -> bool:
    roles = " ".join(str(ticket.get("selection_role", "")) for ticket in tickets)
    return any(
        marker in roles
        for marker in (
            "stoploss",
            "rotational",
            "coverage",
            "red_guard",
            "final_diversity",
        )
    )


def build_dantuo_plan(
    tickets: list[dict[str, Any]],
    *,
    final_recommendation: Iterable[dict[str, Any]] | None = None,
    candidate_pool: Iterable[dict[str, Any]] | None = None,
    candidate_pool_ticket_limit: int = 80,
    fusion: dict[str, Any] | None = None,
    actual: dict[str, Any] | None = None,
    recent_red: Iterable[Any] | None = None,
    recent_blue: Any | None = None,
    blue_history: Iterable[Any] | None = None,
    max_amount: int = 20,
    unit_price: int = 2,
    preferred_red_dan_count: int = 2,
    preferred_red_tuo_count: int = 5,
    preferred_blue_count: int = 2,
    prefer_compound_when_coverage_like: bool = False,
    preserve_final_portfolio_evidence: bool = True,
    final_to_dantuo_audit_enabled: bool = True,
    compound_red_count: int = 7,
    compound_blue_count: int = 1,
    blue_recent_gap_weight: float = 1.00,
    blue_recent_frequency_weight: float = 0.00,
    blue_evidence_weight: float = 1.00,
    blue_specialist_weight: float = 0.20,
    blue_omission_balance_weight: float = 0.80,
    blue_immediate_repeat_penalty: float = 0.35,
    blue_recent_windows: Iterable[int] | None = None,
    auto_cap_blue_to_budget: bool = True,
    blue_adaptive_budget_policy: bool = True,
    blue_adaptive_oos_window: int = 60,
    blue_adaptive_min_history: int = 120,
    blue_adaptive_fallback_policy: str = "frequency_100",
    red_priority_numbers: Iterable[Any] | None = None,
) -> dict[str, Any] | None:
    """Compress final tickets into a budget-controlled red-dan/red-tuo plan."""

    source_tickets: list[dict[str, Any]] = list(tickets or [])
    pool_limit = max(0, int(candidate_pool_ticket_limit))
    if pool_limit > 0:
        pool_tickets = [
            ticket
            for ticket in list(candidate_pool or [])
            if len(_as_int_list(ticket.get("red", []), low=1, high=33)) == 6
            and _as_int_list([ticket.get("blue")], low=1, high=16)
        ]
        pool_tickets = sorted(
            pool_tickets,
            key=lambda ticket: float(ticket.get("quality_score", ticket.get("consensus_score", 0.0)) or 0.0),
            reverse=True,
        )[:pool_limit]
        for index, ticket in enumerate(pool_tickets, start=1):
            source_tickets.append({
                "red": ticket.get("red", []),
                "blue": ticket.get("blue"),
                "consensus_score": float(ticket.get("quality_score", ticket.get("consensus_score", 0.0)) or 0.0),
                "selection_role": f"dantuo_candidate_pool_top_{index}",
                "blue_selection_role": "candidate_pool_blue_evidence",
            })

    normalized_tickets = [
        {
            "red": sorted(_as_int_list(ticket.get("red", []), low=1, high=33)),
            "blue": _as_int_list([ticket.get("blue")], low=1, high=16)[0],
            "consensus_score": float(ticket.get("consensus_score", 0.0) or 0.0),
            "selection_role": str(ticket.get("selection_role", "") or ""),
            "blue_selection_role": str(ticket.get("blue_selection_role", "") or ""),
        }
        for ticket in source_tickets
        if len(_as_int_list(ticket.get("red", []), low=1, high=33)) == 6
        and _as_int_list([ticket.get("blue")], low=1, high=16)
    ]
    if not normalized_tickets:
        return None

    max_bets = max(1, int(max_amount) // max(1, int(unit_price)))
    recent_red_numbers = set(_as_int_list(recent_red or [], low=1, high=33))
    recent_blue_numbers = _as_int_list([recent_blue], low=1, high=16)
    recent_blue_number = recent_blue_numbers[0] if recent_blue_numbers else None
    red_rank = _rank_support(
        normalized_tickets,
        fusion_rank=(fusion or {}).get("top_red"),
        number_low=1,
        number_high=33,
        color="red",
    )
    blue_rank = _rank_support(
        normalized_tickets,
        fusion_rank=(fusion or {}).get("top_blue"),
        number_low=1,
        number_high=16,
        color="blue",
    )
    red_numbers = _ranked_numbers(red_rank)
    blue_numbers = _ranked_numbers(blue_rank)
    support_map = {number: support for number, _, support, _ in red_rank}
    max_support = max(support_map.values(), default=1) or 1
    support_two_or_more = sum(1 for support in support_map.values() if support >= 2)
    coverage_like = _is_coverage_like_portfolio(normalized_tickets)
    red_union_size = len({number for ticket in normalized_tickets for number in ticket["red"]})
    final_tickets = _normalize_final_recommendation_tickets(final_recommendation, normalized_tickets)
    use_final_portfolio_compression = bool(preserve_final_portfolio_evidence and final_tickets)

    def make_round_robin_compound_plan(
        *,
        red_count: int = 7,
        blue_count: int = 1,
    ) -> dict[str, Any] | None:
        if red_count < 6 or len(blue_numbers) < blue_count:
            return None
        red_bets = comb(red_count, 6)
        bet_count = red_bets * blue_count
        if bet_count <= 0 or bet_count > max_bets:
            return None

        fusion_order = _as_int_list((fusion or {}).get("top_red") or [], low=1, high=33)
        fusion_position = {number: rank for rank, number in enumerate(fusion_order)}

        def recent_red_bonus(number: int) -> float:
            if not recent_red_numbers:
                return 0.0
            if number in recent_red_numbers:
                return 1.20
            nearest_gap = min(abs(int(number) - int(prev)) for prev in recent_red_numbers)
            if nearest_gap == 1:
                return 0.95
            if nearest_gap == 2:
                return 0.45
            return 0.0

        selected: list[int] = []
        seen: set[int] = set()
        ticket_rankings: list[list[int]] = []
        for ticket in normalized_tickets:
            numbers = list(ticket["red"])
            ticket_rankings.append(sorted(
                numbers,
                key=lambda number: (
                    -recent_red_bonus(number),
                    fusion_position.get(number, 999),
                    -float(ticket.get("consensus_score", 0.0) or 0.0),
                    number,
                ),
            ))

        # Coverage/stop-loss tickets are intentionally spread out.  In that
        # mode, round-robin selection keeps one or two high-ranked numbers from
        # each ticket instead of treating repeated matrix artifacts as red-dan
        # confidence.
        depth = 0
        while len(selected) < red_count and depth < 6:
            for ranking in ticket_rankings:
                if depth >= len(ranking):
                    continue
                number = ranking[depth]
                if number in seen:
                    continue
                selected.append(number)
                seen.add(number)
                if len(selected) >= red_count:
                    break
            depth += 1

        for number in red_numbers:
            if len(selected) >= red_count:
                break
            if number not in seen:
                selected.append(number)
                seen.add(number)

        if len(selected) < red_count:
            return None

        blue_final_evidence: dict[str, Any] = {}
        if use_final_portfolio_compression:
            blue, blue_final_evidence = _select_blue_numbers_from_final_evidence(
                final_tickets,
                ranked=blue_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                count=blue_count,
                blue_history=blue_history,
                omission_balance_weight=blue_omission_balance_weight,
                immediate_repeat_penalty=blue_immediate_repeat_penalty,
            )
            if not blue:
                blue = _select_blue_numbers_for_prediction_set(
                    blue_rank,
                    normalized_tickets,
                    fusion,
                    count=blue_count,
                    blue_history=blue_history,
                    recent_gap_weight=blue_recent_gap_weight,
                    recent_frequency_weight=blue_recent_frequency_weight,
                    evidence_weight=blue_evidence_weight,
                    specialist_weight=blue_specialist_weight,
                    omission_balance_weight=blue_omission_balance_weight,
                    immediate_repeat_penalty=blue_immediate_repeat_penalty,
                    recent_windows=blue_recent_windows,
                )
                blue_final_evidence = {
                    "schema": "final_blue_evidence_v1",
                    "active": False,
                    "reason": "fallback_to_prediction_set_no_final_blue_evidence",
                }
        else:
            blue = _select_blue_numbers_for_prediction_set(
                blue_rank,
                normalized_tickets,
                fusion,
                count=blue_count,
                blue_history=blue_history,
                recent_gap_weight=blue_recent_gap_weight,
                recent_frequency_weight=blue_recent_frequency_weight,
                evidence_weight=blue_evidence_weight,
                specialist_weight=blue_specialist_weight,
                omission_balance_weight=blue_omission_balance_weight,
                immediate_repeat_penalty=blue_immediate_repeat_penalty,
                recent_windows=blue_recent_windows,
            )
            blue_final_evidence = {
                "schema": "final_blue_evidence_v1",
                "active": False,
                "reason": "final_portfolio_preservation_disabled",
            }
        plan = {
            "mode": "red_compound_coverage_budget_20",
            "red_dan": [],
            "red_tuo": sorted(selected[:red_count]),
            "blue": blue,
            "red_bet_count": int(red_bets),
            "bet_count": int(bet_count),
            "unit_price": int(unit_price),
            "amount": int(bet_count * unit_price),
            "max_amount": int(max_amount),
            "source_ticket_count": len(normalized_tickets),
            "red_dan_candidates": [],
            "rule": f"覆盖型 0 胆 + {red_count} 红复式，红球展开 C({red_count},6)={red_bets} 注；蓝球 {blue_count} 个；总注数 {bet_count} 注。",
            "selection_reason": "当前最终组合属于覆盖/止损型且红球高度分散，重复次数不再视作可靠胆码；改用逐注轮转聚合，并优先保留与最新一期存在重复或邻近关系、同时仍在最终组合内出现的号码，优先解决命中号码分散在多注里的问题。",
            "blue_selection_reason": "蓝球按Prediction Set + Hit@5覆盖目标选择：先按Edge确定候选集合宽度，再按概率分层、跨方法支持、蓝球专家和合理遗漏周期选择有限蓝球。",
        }
        plan["hit_summary"] = _hit_summary(plan, actual)
        return plan

    def compound_plan_score(plan: dict[str, Any] | None) -> float:
        if not plan:
            return float("-inf")
        red_set = _as_int_list(plan.get("red_tuo", []), low=1, high=33)
        if len(red_set) < 6:
            return float("-inf")
        fusion_order = _as_int_list((fusion or {}).get("top_red") or [], low=1, high=33)
        fusion_position = {number: rank for rank, number in enumerate(fusion_order)}
        score_map = {number: score for number, score, _, _ in red_rank}
        support_part = sum(float(support_map.get(number, 0)) for number in red_set)
        score_part = sum(float(score_map.get(number, 0.0)) for number in red_set)
        fusion_part = sum(1.0 / (1.0 + float(fusion_position.get(number, 30))) for number in red_set)
        repeated_part = sum(1 for number in red_set if support_map.get(number, 0) >= 2)
        top_core_part = sum(1 for number in red_set if fusion_position.get(number, 999) < 12)
        return float(
            1.4 * support_part
            + 0.9 * score_part
            + 4.0 * fusion_part
            + 1.8 * repeated_part
            + 1.2 * top_core_part
            + 0.05 * int(plan.get("bet_count", 0))
        )

    def make_ticket_anchor_compound_plan(
        *,
        red_count: int = 7,
        blue_count: int = 1,
    ) -> dict[str, Any] | None:
        if red_count < 6 or len(blue_numbers) < blue_count:
            return None
        red_bets = comb(red_count, 6)
        bet_count = red_bets * blue_count
        if bet_count <= 0 or bet_count > max_bets:
            return None

        fusion_order = _as_int_list((fusion or {}).get("top_red") or [], low=1, high=33)
        fusion_position = {number: rank for rank, number in enumerate(fusion_order)}
        score_map = {number: score for number, score, _, _ in red_rank}
        max_score = max(score_map.values(), default=1.0) or 1.0

        def recent_red_bonus(number: int) -> float:
            if not recent_red_numbers:
                return 0.0
            if number in recent_red_numbers:
                return 1.0
            nearest_gap = min(abs(int(number) - int(prev)) for prev in recent_red_numbers)
            if nearest_gap == 1:
                return 0.70
            if nearest_gap == 2:
                return 0.35
            return 0.0

        final_union = []
        seen: set[int] = set()
        for ticket in normalized_tickets:
            for number in ticket["red"]:
                if number not in seen:
                    final_union.append(number)
                    seen.add(number)
        for number in red_numbers:
            if number not in seen:
                final_union.append(number)
                seen.add(number)

        best_plan: tuple[float, dict[str, Any]] | None = None
        for ticket_index, ticket in enumerate(normalized_tickets):
            base = sorted(set(ticket["red"]))
            if len(base) != 6:
                continue
            bridge_candidates = [number for number in final_union if number not in set(base)]
            if not bridge_candidates:
                bridge_candidates = [number for number in red_numbers if number not in set(base)]
            for bridge in bridge_candidates[:18]:
                red_set = sorted(set(base) | {int(bridge)})
                if len(red_set) != red_count:
                    continue
                repeated_support = sum(max(0, support_map.get(number, 0) - 1) for number in red_set)
                fusion_strength = sum(1.0 / (1.0 + fusion_position.get(number, 30)) for number in red_set)
                score_strength = sum(score_map.get(number, 0.0) for number in red_set) / max_score
                recent_strength = sum(recent_red_bonus(number) for number in red_set)
                top_core_hits = sum(1 for number in red_set if fusion_position.get(number, 999) < 8)
                row_quality = float(ticket.get("consensus_score", 0.0) or 0.0)
                bridge_bonus = (
                    0.65 * support_map.get(int(bridge), 0)
                    + 1.25 * (1.0 / (1.0 + fusion_position.get(int(bridge), 30)))
                    + recent_red_bonus(int(bridge))
                )
                utility = (
                    0.9 * repeated_support
                    + 4.0 * fusion_strength
                    + 1.1 * score_strength
                    + 1.7 * top_core_hits
                    + 1.0 * recent_strength
                    + 0.4 * row_quality
                    + bridge_bonus
                    - 0.10 * ticket_index
                )
                blue = _select_blue_numbers_for_prediction_set(
                    blue_rank,
                    normalized_tickets,
                    fusion,
                    count=blue_count,
                    blue_history=blue_history,
                    recent_gap_weight=blue_recent_gap_weight,
                    recent_frequency_weight=blue_recent_frequency_weight,
                    evidence_weight=blue_evidence_weight,
                    specialist_weight=blue_specialist_weight,
                    omission_balance_weight=blue_omission_balance_weight,
                    immediate_repeat_penalty=blue_immediate_repeat_penalty,
                    recent_windows=blue_recent_windows,
                )
                plan = {
                    "mode": "red_compound_ticket_anchor_budget_20",
                    "red_dan": [],
                    "red_tuo": red_set,
                    "blue": blue,
                    "red_bet_count": int(red_bets),
                    "bet_count": int(bet_count),
                    "unit_price": int(unit_price),
                    "amount": int(bet_count * unit_price),
                    "max_amount": int(max_amount),
                    "source_ticket_count": len(normalized_tickets),
                    "red_dan_candidates": [],
                    "anchor_ticket_index": int(ticket_index + 1),
                    "bridge_red": int(bridge),
                    "rule": f"核心票压缩 0 胆 + {red_count} 红复式，红球展开 C({red_count},6)={red_bets} 注；蓝球 {blue_count} 个；总注数 {bet_count} 注。",
                    "selection_reason": "覆盖型候选票中命中容易分散；该方案先保留一整注核心票，再补1个桥接红球，形成7红+1蓝观察口径，避免把同一注内的潜在集中命中拆散，同时仍控制在预算内。",
                    "blue_selection_reason": "蓝球按Prediction Set + Hit@5覆盖目标选择：先按Edge确定候选集合宽度，再按概率分层、跨方法支持、蓝球专家和合理遗漏周期选择有限蓝球。",
                }
                plan["hit_summary"] = _hit_summary(plan, actual)
                if best_plan is None or utility > best_plan[0]:
                    best_plan = (float(utility), plan)
        return best_plan[1] if best_plan else None

    dispersion_like = bool(
        coverage_like
        and (
            max_support <= 2
            or support_two_or_more >= 4
            or red_union_size >= max(12, len(normalized_tickets) * 3)
        )
    )
    if prefer_compound_when_coverage_like and dispersion_like and not use_final_portfolio_compression:
        plans = [
            make_ticket_anchor_compound_plan(red_count=int(compound_red_count), blue_count=int(compound_blue_count)),
            make_round_robin_compound_plan(red_count=int(compound_red_count), blue_count=int(compound_blue_count)),
        ]
        plans = [plan for plan in plans if plan is not None]
        if plans:
            best_compound = max(plans, key=compound_plan_score)
            best_compound["selection_reason"] = (
                str(best_compound.get("selection_reason", ""))
                + f" 触发原因：coverage_like={coverage_like}，红球并集={red_union_size}，重复支持号码数={support_two_or_more}，最高重复={max_support}；因此避免使用高风险胆码。"
            )
            return _finalize_dantuo_plan(
                best_compound,
                ranked=red_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                actual=actual,
            )

    def make_plan(dan_count: int, tuo_count: int, blue_count: int) -> dict[str, Any] | None:
        if not 1 <= dan_count <= 5:
            return None
        if dan_count + tuo_count < 6:
            return None
        requested_blue_count = max(1, int(blue_count))
        if len(red_numbers) < dan_count + tuo_count:
            return None
        red_bets = comb(tuo_count, 6 - dan_count)
        effective_blue_count = requested_blue_count
        budget_blue_capped = False
        if red_bets * effective_blue_count > max_bets and bool(auto_cap_blue_to_budget):
            effective_blue_count = max(1, min(requested_blue_count, max_bets // max(1, red_bets)))
            budget_blue_capped = effective_blue_count != requested_blue_count
        blue_count = effective_blue_count
        bet_count = red_bets * blue_count
        if len(blue_numbers) < blue_count:
            return None
        if bet_count <= 0 or bet_count > max_bets:
            return None

        compression_audit: dict[str, Any] | None = None
        if use_final_portfolio_compression:
            selected_red, compression_audit = _optimize_final_portfolio_red_set(
                final_tickets,
                target_count=dan_count + tuo_count,
                ranked=red_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                red_priority_numbers=red_priority_numbers,
            )
            if len(selected_red) < dan_count + tuo_count:
                return None
            red_dan, dan_candidates = _select_final_portfolio_dan(
                selected_red,
                final_tickets,
                ranked=red_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                dan_count=dan_count,
            )
            red_tuo = sorted([number for number in selected_red if number not in set(red_dan)])
            if len(red_tuo) > tuo_count:
                red_tuo = red_tuo[:tuo_count]
        else:
            red_dan, dan_candidates = _select_red_dan(red_rank, normalized_tickets, dan_count)
            red_tuo = sorted([number for number in red_numbers if number not in set(red_dan)][:tuo_count])
        blue_final_evidence: dict[str, Any]
        if use_final_portfolio_compression:
            blue, blue_final_evidence = _select_blue_numbers_from_final_evidence(
                final_tickets,
                ranked=blue_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                count=blue_count,
                blue_history=blue_history,
                omission_balance_weight=blue_omission_balance_weight,
                immediate_repeat_penalty=blue_immediate_repeat_penalty,
            )
            if not blue:
                blue = _select_blue_numbers_for_prediction_set(
                    blue_rank,
                    normalized_tickets,
                    fusion,
                    count=blue_count,
                    blue_history=blue_history,
                    recent_gap_weight=blue_recent_gap_weight,
                    recent_frequency_weight=blue_recent_frequency_weight,
                    evidence_weight=blue_evidence_weight,
                    specialist_weight=blue_specialist_weight,
                    omission_balance_weight=blue_omission_balance_weight,
                    immediate_repeat_penalty=blue_immediate_repeat_penalty,
                    recent_windows=blue_recent_windows,
                )
                blue_final_evidence = {
                    "schema": "final_blue_evidence_v1",
                    "active": False,
                    "reason": "fallback_to_prediction_set_no_final_blue_evidence",
                }
        else:
            blue = _select_blue_numbers_for_prediction_set(
                blue_rank,
                normalized_tickets,
                fusion,
                count=blue_count,
                blue_history=blue_history,
                recent_gap_weight=blue_recent_gap_weight,
                recent_frequency_weight=blue_recent_frequency_weight,
                evidence_weight=blue_evidence_weight,
                specialist_weight=blue_specialist_weight,
                omission_balance_weight=blue_omission_balance_weight,
                immediate_repeat_penalty=blue_immediate_repeat_penalty,
                recent_windows=blue_recent_windows,
            )
            blue_final_evidence = {
                "schema": "final_blue_evidence_v1",
                "active": False,
                "reason": "final_portfolio_preservation_disabled",
            }
        adaptive_blue_info: dict[str, Any] = {}
        if bool(blue_adaptive_budget_policy) and blue_history:
            fusion_scores = _fusion_score_map(
                fusion,
                key_numbers="top_blue",
                key_scores="top_blue_scores",
                low=1,
                high=16,
            )
            adaptive_probability = np.asarray(
                [float(fusion_scores.get(number, 0.0)) for number in range(1, 17)],
                dtype=float,
            )
            adaptive_blue, adaptive_blue_info = select_adaptive_blue_set(
                blue_history,
                count=int(blue_count),
                probability=adaptive_probability,
                evaluation_window=int(blue_adaptive_oos_window),
                minimum_history=int(blue_adaptive_min_history),
                fallback_policy=str(blue_adaptive_fallback_policy),
            )
            if len(adaptive_blue) >= int(blue_count):
                blue = adaptive_blue[: int(blue_count)]
                blue_final_evidence = {
                    **(blue_final_evidence or {}),
                    "adaptive_budget_policy": adaptive_blue_info,
                    "source": "adaptive_history_oos_budget",
                    "selected_blue": [int(value) for value in blue],
                }

        if len(red_tuo) != tuo_count:
            return None

        plan = {
            "mode": "red_dan_tuo_budget_20",
            "red_dan": red_dan,
            "red_tuo": red_tuo,
            "blue": blue,
            "red_bet_count": int(red_bets),
            "bet_count": int(bet_count),
            "unit_price": int(unit_price),
            "amount": int(bet_count * unit_price),
            "max_amount": int(max_amount),
            "source_ticket_count": len(normalized_tickets),
            "red_dan_candidates": dan_candidates,
            "requested_blue_count": int(requested_blue_count),
            "budget_blue_count_capped": bool(budget_blue_capped),
            "requested_layout": f"{dan_count}胆 + {tuo_count}拖 + {requested_blue_count}蓝",
            "preserve_final_portfolio_evidence": bool(use_final_portfolio_compression),
            "final_portfolio_compression": compression_audit or {},
            "final_blue_evidence": blue_final_evidence,
            "rule": f"{dan_count}胆 + {tuo_count}拖，红球展开 C({tuo_count},{6 - dan_count})={red_bets} 注；蓝球 {blue_count} 个；总注数 {bet_count} 注。"
            + (
                f" 原请求蓝球 {requested_blue_count} 个会超过 {max_amount} 元预算，已自动压缩为 {blue_count} 个。"
                if budget_blue_capped
                else ""
            ),
            "selection_reason": (
                "胆拖定位为Final5预算压缩器：先从最终5注并集中按票覆盖选出7红，再把最强一注的重叠补到4个（不完整克隆6红），"
                "避免跨票高频共识把整票信号拆散；候选池和方法票只作为证据，不再重新决定一套预测号码。"
                if use_final_portfolio_compression
                else "红胆从最终组合的高支持号码中择优，兼顾重复出现、同现次数、区间分散和合理跨度；红拖取剩余高支持号码。"
            ),
            "blue_selection_reason": (
                "蓝球预算层优先使用目标期之前完成数据的滚动OOS历史策略选择；当历史证据不足时才回退到原Prediction Set + Evidence路径。"
                if adaptive_blue_info else
                "蓝球按Prediction Set + Hit@5覆盖目标选择：融合候选支持、模型证据、蓝球专家、合理遗漏周期，并对近期刚开出的重复号降权。"
            ),
            "blue_adaptive_budget_policy": adaptive_blue_info,
        }
        if use_final_portfolio_compression and final_to_dantuo_audit_enabled:
            plan["final_to_dantuo_audit"] = _build_final_to_dantuo_audit(
                plan,
                final_tickets,
                ranked=red_rank,
                tickets=normalized_tickets,
                fusion=fusion,
            )
            plan["blue_final_to_dantuo_audit"] = _build_blue_final_to_dantuo_audit(
                plan,
                final_tickets,
                ranked=blue_rank,
                tickets=normalized_tickets,
                fusion=fusion,
                blue_history=blue_history,
                omission_balance_weight=blue_omission_balance_weight,
                immediate_repeat_penalty=blue_immediate_repeat_penalty,
            )
            plan["blue_selection_reason"] = "蓝球胆拖冻结Final5蓝球并集，按出现次数和票面证据压缩，不再用遗漏周期覆盖融合排序。"
        plan["hit_summary"] = _hit_summary(plan, actual)
        return plan

    preferred = (
        int(preferred_red_dan_count),
        int(preferred_red_tuo_count),
        int(preferred_blue_count),
    )
    preferred_plan = make_plan(*preferred)
    if preferred_plan is not None:
        if use_final_portfolio_compression:
            return preferred_plan
        return _finalize_dantuo_plan(
            preferred_plan,
            ranked=red_rank,
            tickets=normalized_tickets,
            fusion=fusion,
            actual=actual,
        )

    layouts: list[tuple[int, int, int]] = [preferred]
    for dan_count in range(1, 6):
        for tuo_count in range(max(1, 6 - dan_count), 13):
            for blue_count in range(1, 4):
                layouts.append((dan_count, tuo_count, blue_count))

    best: tuple[float, dict[str, Any]] | None = None
    for dan_count, tuo_count, blue_count in layouts:
        plan = make_plan(dan_count, tuo_count, blue_count)
        if plan is None:
            continue
        dan_support = sum(support_map.get(number, 0) for number in plan["red_dan"])
        tuo_support = sum(support_map.get(number, 0) for number in plan["red_tuo"])
        blue_bonus = 0.35 * blue_count
        budget_bonus = plan["bet_count"] / max_bets
        dan_risk_penalty = max(0, dan_count - 2) * 0.45
        utility = (
            dan_support * 1.2
            + tuo_support * 0.55
            + len(set(plan["red_dan"]) | set(plan["red_tuo"])) * 0.12
            + blue_bonus
            + budget_bonus
            - dan_risk_penalty
        )
        if best is None or utility > best[0]:
            best = (utility, plan)

    if not best:
        return None
    if use_final_portfolio_compression:
        return best[1]
    return _finalize_dantuo_plan(
        best[1],
        ranked=red_rank,
        tickets=normalized_tickets,
        fusion=fusion,
        actual=actual,
    )
