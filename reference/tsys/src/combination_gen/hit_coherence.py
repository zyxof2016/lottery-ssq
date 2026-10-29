"""Final5 hit-coherence constructor for V11.7.

Candidate selection stays a number-recall stage. This module only reorganizes
an already chosen candidate set into five tickets. Groups and tickets are built
from information available before the target draw. Actual draws are accepted
only by the evaluation helpers.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np

from src.candidate.recall_funnel import _ranking_candidate_set


SCHEMA = "final5_hit_coherence_v11_7"
CANDIDATE_WIDTHS = (18, 20, 22, 24, 26)
STRATEGIES = ("current", "coverage", "hit_coherence", "hybrid")
STRATEGY_LABELS = {
    "current": "当前Final5",
    "coverage": "Coverage",
    "hit_coherence": "Hit-Coherence",
    "hybrid": "Hybrid",
}
SCORE_WEIGHTS = {
    "probability": 0.25,
    "hit_coherence": 0.20,
    "evidence": 0.15,
    "coverage": 0.15,
    "model_survival": 0.10,
    "diversity": 0.10,
    "rescue": 0.05,
}
HISTORY_WINDOWS = (30, 60, 100, 300)
STRATEGY_TIE_RANK = {
    "hit_coherence": 3,
    "hybrid": 2,
    "current": 1,
    "coverage": 0,
}


def red_draws_from_frame(frame: Any) -> list[tuple[int, ...]]:
    if frame is None or len(frame) == 0:
        return []
    columns = [f"r{index}" for index in range(1, 7)]
    values = frame.loc[:, columns].to_numpy()
    return [tuple(sorted(int(number) for number in row)) for row in values]


def _probability_33(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.full(33, 1.0 / 33.0)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 33 or not np.all(np.isfinite(array)):
        return np.full(33, 1.0 / 33.0)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return np.full(33, 1.0 / 33.0)
    return array / total


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return float(sum(rows) / len(rows))


def _ticket_reds(ticket: dict[str, Any]) -> list[int]:
    reds = []
    seen: set[int] = set()
    for value in ticket.get("red", []) or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 33 and number not in seen:
            reds.append(number)
            seen.add(number)
    return sorted(reds)


def _historical_pair_strength(draws: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    strength = np.zeros((34, 34), dtype=float)
    frequency = np.zeros(34, dtype=float)
    used = 0
    if not draws:
        return strength, frequency
    recent = draws[-100:]
    for draw in recent:
        for number in draw:
            if 1 <= int(number) <= 33:
                frequency[int(number)] += 1.0
    frequency /= max(len(recent), 1)
    for window in HISTORY_WINDOWS:
        sample = draws[-int(window):]
        if len(sample) < 12:
            continue
        counts = np.zeros(34, dtype=float)
        joint = np.zeros((34, 34), dtype=float)
        for draw in sample:
            numbers = [int(number) for number in draw if 1 <= int(number) <= 33]
            for number in numbers:
                counts[number] += 1.0
            for left, right in combinations(numbers, 2):
                joint[left, right] += 1.0
                joint[right, left] += 1.0
        periods = float(len(sample))
        window_strength = np.zeros((34, 34), dtype=float)
        for left in range(1, 34):
            for right in range(left + 1, 34):
                expected = counts[left] * counts[right] / periods
                observed = joint[left, right]
                lift = observed / expected if expected > 1e-9 else 0.0
                value = min(max(lift, 0.0), 3.0) / 3.0
                window_strength[left, right] = value
                window_strength[right, left] = value
        strength += window_strength
        used += 1
    if used:
        strength /= float(used)
    strength[0, :] = 0.0
    strength[:, 0] = 0.0
    return strength, frequency


def _model_top10(
    model_probabilities: dict[str, dict[str, Any]] | None,
    model_rankings: dict[str, dict[str, Any]] | None,
) -> dict[str, list[int]]:
    tops: dict[str, list[int]] = {}
    for name, prediction in (model_probabilities or {}).items():
        red = np.asarray((prediction or {}).get("red", []), dtype=float).reshape(-1)
        if red.size != 33 or not np.all(np.isfinite(red)):
            continue
        if bool(np.allclose(red, red[0], rtol=1e-10, atol=1e-12)):
            continue
        order = [int(index) + 1 for index in np.argsort(-red, kind="stable")[:10]]
        tops[str(name)] = order
    if tops:
        return tops
    for name, ranking in (model_rankings or {}).items():
        numbers = []
        for value in (ranking or {}).get("top_red") or []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= 33 and number not in numbers:
                numbers.append(number)
        if len(numbers) >= 6:
            tops[str(name)] = numbers[:10]
    return tops


def _pair_value(matrix: np.ndarray, left: int, right: int) -> float:
    if left == right:
        return 0.0
    return float(matrix[int(left), int(right)])


def _ticket_coherence(reds: Iterable[int], pairs: np.ndarray) -> float:
    numbers = list(reds)
    if len(numbers) < 2:
        return 0.0
    return _mean(
        _pair_value(pairs, left, right)
        for left, right in combinations(numbers, 2)
    )


def build_coherence_context(
    *,
    fusion_red: Iterable[float] | None,
    history_draws: list[tuple[int, ...]] | None,
    model_probabilities: dict[str, dict[str, Any]] | None = None,
    model_rankings: dict[str, dict[str, Any]] | None = None,
    method_tickets: dict[str, list[dict[str, Any]]] | None = None,
    pool: list[dict[str, Any]] | None = None,
    rescue_numbers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Assemble pre-target number and pair evidence. No actual draw is read."""

    probability = _probability_33(fusion_red)
    history_pairs, frequency = _historical_pair_strength(list(history_draws or []))
    model_tops = _model_top10(model_probabilities, model_rankings)
    model_count = max(len(model_tops), 1)
    model_support = {number: 0 for number in range(1, 34)}
    model_pairs = np.zeros((34, 34), dtype=float)
    for numbers in model_tops.values():
        for number in numbers:
            model_support[int(number)] += 1
        for left, right in combinations(numbers, 2):
            model_pairs[left, right] += 1.0
            model_pairs[right, left] += 1.0
    if model_tops:
        model_pairs /= float(len(model_tops))

    method_support = {number: 0 for number in range(1, 34)}
    method_pairs = np.zeros((34, 34), dtype=float)
    method_rows = [
        _ticket_reds(ticket)
        for tickets in (method_tickets or {}).values()
        for ticket in tickets or []
    ]
    for reds in method_rows:
        for number in reds:
            method_support[number] += 1
        for left, right in combinations(reds, 2):
            method_pairs[left, right] += 1.0
            method_pairs[right, left] += 1.0
    if method_rows:
        method_pairs /= float(len(method_rows))

    pool_support = {number: 0 for number in range(1, 34)}
    for ticket in pool or []:
        for number in _ticket_reds(ticket):
            pool_support[number] += 1
    max_method = max(method_support.values(), default=0) or 1
    max_pool = max(pool_support.values(), default=0) or 1
    rescue = {
        int(number)
        for number in (rescue_numbers or [])
        if 1 <= int(number) <= 33
    }
    evidence = {}
    survival = {}
    number_score = {}
    for number in range(1, 34):
        evidence[number] = float(
            0.55 * (method_support[number] / max_method)
            + 0.25 * (pool_support[number] / max_pool)
            + 0.20 * frequency[number]
        )
        survival[number] = float(model_support[number] / model_count) if model_tops else 0.0
        number_score[number] = float(
            0.45 * probability[number - 1]
            + 0.30 * evidence[number]
            + 0.25 * survival[number]
            + (0.04 if number in rescue else 0.0)
        )
    pairs = 0.50 * history_pairs + 0.35 * model_pairs + 0.15 * method_pairs
    return {
        "probability": probability,
        "evidence": evidence,
        "survival": survival,
        "number_score": number_score,
        "pairs": pairs,
        "rescue": rescue,
        "model_count": int(len(model_tops)),
    }


def candidate_numbers_for_width(
    context_inputs: dict[str, Any],
    *,
    width: int,
) -> list[int]:
    selected, _meta = _ranking_candidate_set(
        context_inputs["fusion_red"],
        context_inputs.get("method_tickets") or {},
        limit=int(width),
        rescue_numbers=context_inputs.get("rescue_numbers") or [],
    )
    return sorted(int(number) for number in selected)


def _slot_plan(group_count: int, ticket_count: int) -> list[int]:
    if group_count <= 0 or ticket_count <= 0:
        return []
    if group_count == 1:
        return [int(ticket_count)]
    if group_count == 2:
        first = max(1, (int(ticket_count) + 1) // 2)
        return [first, int(ticket_count) - first]
    base = [1] * group_count
    extra = int(ticket_count) - group_count
    if extra > 0:
        base[0] += 1
        extra -= 1
    index = group_count - 1
    while extra > 0:
        base[index] += 1
        extra -= 1
        index = group_count - 1 if index == 0 else index - 1
    return base


def build_coherence_groups(
    numbers: Iterable[int],
    context: dict[str, Any],
    *,
    group_count: int = 3,
    group_size: int = 8,
) -> list[dict[str, Any]]:
    remaining = sorted(
        {int(number) for number in numbers if 1 <= int(number) <= 33},
        key=lambda number: (
            float(context["number_score"].get(number, 0.0)),
            -number,
        ),
        reverse=True,
    )
    pairs = context["pairs"]
    groups: list[dict[str, Any]] = []
    names = "ABCDEFGH"
    target = max(6, int(group_size))
    for index in range(max(1, int(group_count))):
        if not remaining:
            break
        seed = remaining.pop(0)
        members = [seed]
        while len(members) < target and remaining:
            def affinity(number: int, members: list[int] = members) -> tuple[float, float, int]:
                return (
                    _mean(_pair_value(pairs, number, other) for other in members),
                    float(context["number_score"].get(number, 0.0)),
                    -number,
                )
            best = max(remaining, key=affinity)
            remaining.remove(best)
            members.append(best)
        groups.append({
            "name": names[index] if index < len(names) else str(index + 1),
            "numbers": sorted(members),
            "mean_coherence": _ticket_coherence(members, pairs) if len(members) >= 2 else 0.0,
        })
    groups.sort(key=lambda group: float(group["mean_coherence"]), reverse=True)
    for index, group in enumerate(groups):
        group["name"] = names[index] if index < len(names) else str(index + 1)
    return groups


def _best_six(numbers: list[int], context: dict[str, Any]) -> list[int]:
    unique = sorted({int(number) for number in numbers if 1 <= int(number) <= 33})
    if len(unique) <= 6:
        return unique
    pairs = context["pairs"]
    scores = context["number_score"]
    best_red: tuple[int, ...] | None = None
    best_score = float("-inf")
    for combo in combinations(unique, 6):
        coherence = _ticket_coherence(combo, pairs)
        quality = _mean(float(scores.get(number, 0.0)) for number in combo)
        score = coherence + 0.15 * quality
        if score > best_score:
            best_score = score
            best_red = combo
    return list(best_red or unique[:6])


def _grow_ticket(seed: list[int], pool: list[int], context: dict[str, Any]) -> list[int]:
    core = [int(number) for number in seed if 1 <= int(number) <= 33]
    rest = [int(number) for number in pool if int(number) not in set(core) and 1 <= int(number) <= 33]
    pairs = context["pairs"]
    while len(core) < 8 and rest:
        def affinity(number: int) -> tuple[float, float, int]:
            partner = _mean(_pair_value(pairs, number, other) for other in core) if core else 0.0
            return (partner, float(context["number_score"].get(number, 0.0)), -number)

        best = max(rest, key=affinity)
        rest.remove(best)
        core.append(best)
    if len(core) < 6:
        return sorted(core)
    return _best_six(core, context)


def _variant_ticket(
    primary: list[int],
    pool: list[int],
    context: dict[str, Any],
    used: set[frozenset[int]],
) -> list[int] | None:
    if len(primary) != 6:
        return None
    pairs = context["pairs"]

    def leave_score(number: int) -> float:
        others = [other for other in primary if other != number]
        return _mean(_pair_value(pairs, number, other) for other in others)

    weakest = sorted(primary, key=lambda number: (leave_score(number), number))[:2]
    kept = [number for number in primary if number not in set(weakest)]
    outside = [number for number in pool if number not in set(primary)]
    best: tuple[float, list[int]] | None = None
    if len(outside) >= 2:
        for left, right in combinations(outside, 2):
            reds = sorted([*kept, int(left), int(right)])
            key = frozenset(reds)
            if key in used or len(reds) != 6:
                continue
            score = _ticket_coherence(reds, pairs)
            if best is None or score > best[0]:
                best = (score, reds)
    if best is None:
        return None
    return best[1]


def _make_ticket(
    reds: list[int],
    *,
    strategy: str,
    group_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    cleaned = sorted({int(number) for number in reds if 1 <= int(number) <= 33})
    return {
        "red": cleaned,
        "selection_role": f"final5_{strategy}",
        "final5_hit_coherence": {
            "schema": SCHEMA,
            "strategy": strategy,
            "group": group_name,
            "within_ticket_coherence": _ticket_coherence(cleaned, context["pairs"]),
        },
    }


def _normalize_current(tickets: list[dict[str, Any]] | None, count: int) -> list[dict[str, Any]]:
    rows = []
    for ticket in tickets or []:
        reds = _ticket_reds(ticket)
        if len(reds) != 6:
            continue
        rows.append({
            "red": reds,
            "selection_role": str(ticket.get("selection_role") or "final5_current"),
            "final5_hit_coherence": {
                "schema": SCHEMA,
                "strategy": "current",
                "group": "legacy",
            },
        })
        if len(rows) >= int(count):
            break
    return rows


def _coverage_tickets(
    numbers: list[int],
    context: dict[str, Any],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Spread high-score numbers across tickets. This is the coverage baseline."""

    ordered = sorted(
        numbers,
        key=lambda number: (float(context["number_score"].get(number, 0.0)), -int(number)),
        reverse=True,
    )
    buckets: list[list[int]] = [[] for _ in range(max(1, int(count)))]
    if len(ordered) < 6:
        return []
    for index, number in enumerate(ordered):
        buckets[index % len(buckets)].append(int(number))
        if all(len(bucket) >= 6 for bucket in buckets):
            break
    for bucket in buckets:
        if len(bucket) >= 6:
            continue
        for number in reversed(ordered):
            if int(number) not in bucket:
                bucket.append(int(number))
            if len(bucket) >= 6:
                break
    return [
        _make_ticket(bucket[:6], strategy="coverage", group_name=f"spread_{index + 1}", context=context)
        for index, bucket in enumerate(buckets)
        if len(bucket) >= 6
    ][: int(count)]


def _group_tickets(
    numbers: list[int],
    groups: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    count: int,
    per_group: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not groups:
        return []
    slots = per_group or _slot_plan(len(groups), int(count))
    used: set[frozenset[int]] = set()
    tickets: list[dict[str, Any]] = []
    for group, slot_count in zip(groups, slots):
        members = [int(number) for number in group["numbers"]]
        primary = _best_six(members, context) if len(members) >= 6 else _grow_ticket(members, numbers, context)
        if len(primary) != 6:
            continue
        key = frozenset(primary)
        if key not in used:
            used.add(key)
            tickets.append(_make_ticket(
                primary,
                strategy="hit_coherence",
                group_name=str(group["name"]),
                context=context,
            ))
        for _variant_index in range(max(0, int(slot_count) - 1)):
            variant = _variant_ticket(primary, numbers, context, used)
            if not variant:
                break
            used.add(frozenset(variant))
            tickets.append(_make_ticket(
                variant,
                strategy="hit_coherence",
                group_name=str(group["name"]),
                context=context,
            ))
            primary = variant
        if len(tickets) >= int(count):
            break
    return tickets[: int(count)]


def _hybrid_tickets(
    numbers: list[int],
    groups: list[dict[str, Any]],
    context: dict[str, Any],
    *,
    count: int,
) -> list[dict[str, Any]]:
    anchors = _group_tickets(
        numbers,
        groups,
        context,
        count=min(len(groups), int(count)),
        per_group=[1] * len(groups),
    )
    anchors = [
        {**ticket, "selection_role": "final5_hybrid", "final5_hit_coherence": {
            **(ticket.get("final5_hit_coherence") or {}),
            "strategy": "hybrid",
        }}
        for ticket in anchors
    ]
    used_numbers = {number for ticket in anchors for number in ticket["red"]}
    leftovers = [number for number in numbers if number not in used_numbers] or list(numbers)
    filler_groups = build_coherence_groups(leftovers, context, group_count=max(1, int(count) - len(anchors)), group_size=8)
    fillers = _group_tickets(
        numbers,
        filler_groups,
        context,
        count=max(0, int(count) - len(anchors)),
        per_group=[1] * max(1, int(count) - len(anchors)),
    )
    fillers = [
        {**ticket, "selection_role": "final5_hybrid", "final5_hit_coherence": {
            **(ticket.get("final5_hit_coherence") or {}),
            "strategy": "hybrid",
        }}
        for ticket in fillers
    ]
    return (anchors + fillers)[: int(count)]


def score_ticket_set(
    tickets: list[dict[str, Any]],
    candidate: Iterable[int],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Pre-target portfolio score. Actual draws are not an input."""

    reds = [_ticket_reds(ticket) for ticket in tickets if len(_ticket_reds(ticket)) == 6]
    candidate_numbers = [int(number) for number in candidate if 1 <= int(number) <= 33]
    if not reds or not candidate_numbers:
        return {"score": 0.0, "components": {}, "over_concentration": 0.0}
    probability = context["probability"]
    top_reference = sorted(candidate_numbers, key=lambda number: float(probability[number - 1]), reverse=True)[:6]
    reference = _mean(float(probability[number - 1]) for number in top_reference) or 1e-9
    selected_numbers = [number for ticket in reds for number in ticket]
    unique = sorted({number for ticket in reds for number in ticket})
    probability_component = min(1.0, _mean(float(probability[number - 1]) for number in selected_numbers) / reference)
    coherence_component = _mean(_ticket_coherence(ticket, context["pairs"]) for ticket in reds)
    evidence_component = _mean(float(context["evidence"].get(number, 0.0)) for number in selected_numbers)
    survival_component = _mean(float(context["survival"].get(number, 0.0)) for number in selected_numbers)
    coverage_denominator = max(1, min(len(candidate_numbers), 6 * len(reds)))
    coverage_component = min(1.0, len(unique) / float(coverage_denominator))
    overlaps = [
        len(set(left) & set(right))
        for left, right in combinations(reds, 2)
    ]
    jaccard = [
        len(set(left) & set(right)) / max(1, len(set(left) | set(right)))
        for left, right in combinations(reds, 2)
    ]
    diversity_component = 1.0 - _mean(jaccard) if jaccard else 1.0
    rescue = set(context.get("rescue") or [])
    rescue_component = (
        len(rescue & set(unique)) / float(len(rescue))
        if rescue
        else 1.0
    )
    over_concentration = max(0.0, (_mean(overlaps) - 4.0) / 2.0) if overlaps else 0.0
    components = {
        "probability": float(probability_component),
        "hit_coherence": float(coherence_component),
        "evidence": float(evidence_component),
        "coverage": float(coverage_component),
        "model_survival": float(survival_component),
        "diversity": float(diversity_component),
        "rescue": float(rescue_component),
    }
    score = sum(SCORE_WEIGHTS[name] * components[name] for name in SCORE_WEIGHTS) - 0.08 * over_concentration
    return {
        "score": float(score),
        "components": components,
        "over_concentration": float(over_concentration),
        "weights": dict(SCORE_WEIGHTS),
    }


def evaluate_red_outcome(
    tickets: list[dict[str, Any]],
    candidate: Iterable[int],
    actual_red: Iterable[int],
) -> dict[str, Any]:
    """Score frozen tickets after the draw. This must not be used to build them."""

    actual = {int(number) for number in actual_red if 1 <= int(number) <= 33}
    candidate_set = {int(number) for number in candidate if 1 <= int(number) <= 33}
    ticket_hits: list[int] = []
    final_union: set[int] = set()
    for ticket in tickets or []:
        reds = set(_ticket_reds(ticket))
        final_union |= reds
        ticket_hits.append(len(reds & actual))
    found = actual & candidate_set
    retained = found & final_union
    dropped = sorted(found - final_union)
    best = max(ticket_hits) if ticket_hits else 0
    if dropped:
        attribution = "final5_combination_failure"
    elif found:
        attribution = "retained"
    else:
        attribution = "model_or_candidate_miss"
    return {
        "candidate_hit": int(len(found)),
        "candidate_recall": float(len(found) / 6.0),
        "final5_hit": int(len(retained)),
        "final5_union_hit": int(len(actual & final_union)),
        "final5_drop_regret": int(len(dropped)),
        "dropped_numbers": dropped,
        "best_ticket_hit": int(best),
        "any3_plus": float(best >= 3),
        "any4_plus": float(best >= 4),
        "ticket_hits": ticket_hits,
        "split_regret": int(max(0, len(retained) - best)),
        "attribution": attribution,
    }


def _strategy_bundle(
    name: str,
    tickets: list[dict[str, Any]],
    candidate: list[int],
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy": name,
        "label": STRATEGY_LABELS[name],
        "tickets": tickets,
        "reds": [list(ticket["red"]) for ticket in tickets],
        "pre_target_score": score_ticket_set(tickets, candidate, context),
    }


def prepare_final5_counterfactual(
    *,
    current_tickets: list[dict[str, Any]] | None,
    fusion_red: Iterable[float] | None,
    history_draws: list[tuple[int, ...]] | None,
    method_tickets: dict[str, list[dict[str, Any]]] | None = None,
    pool: list[dict[str, Any]] | None = None,
    model_probabilities: dict[str, dict[str, Any]] | None = None,
    model_rankings: dict[str, dict[str, Any]] | None = None,
    rescue_numbers: Iterable[int] | None = None,
    widths: Iterable[int] | None = None,
    count: int = 5,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build every compared Final5 from the same pre-target candidate cuts."""

    context = build_coherence_context(
        fusion_red=fusion_red,
        history_draws=history_draws,
        model_probabilities=model_probabilities,
        model_rankings=model_rankings,
        method_tickets=method_tickets,
        pool=pool,
        rescue_numbers=rescue_numbers,
    )
    inputs = {
        "fusion_red": fusion_red,
        "method_tickets": method_tickets or {},
        "rescue_numbers": rescue_numbers or [],
    }
    chosen = dict(policy or {})
    width_values = []
    for value in list(widths or CANDIDATE_WIDTHS):
        width = int(value)
        if 6 <= width <= 33 and width not in width_values:
            width_values.append(width)
    if not width_values:
        width_values = list(CANDIDATE_WIDTHS)
    default_width = int(chosen.get("width") or width_values[0])
    if default_width not in width_values:
        width_values.append(default_width)
        width_values.sort()
    strategy_name = str(chosen.get("strategy") or "hit_coherence")
    if strategy_name not in STRATEGIES:
        strategy_name = "hit_coherence"
    bundles: dict[str, dict[str, Any]] = {}
    groups_by_width: dict[str, list[dict[str, Any]]] = {}
    for width in width_values:
        candidate = candidate_numbers_for_width(inputs, width=width)
        groups = build_coherence_groups(candidate, context)
        groups_by_width[str(width)] = [
            {
                "name": group["name"],
                "numbers": list(group["numbers"]),
                "mean_coherence": float(group["mean_coherence"]),
            }
            for group in groups
        ]
        bundles[str(width)] = {
            "candidate": candidate,
            "current": _strategy_bundle("current", _normalize_current(current_tickets, count), candidate, context),
            "coverage": _strategy_bundle("coverage", _coverage_tickets(candidate, context, count=count), candidate, context),
            "hit_coherence": _strategy_bundle(
                "hit_coherence",
                _group_tickets(candidate, groups, context, count=count),
                candidate,
                context,
            ),
            "hybrid": _strategy_bundle(
                "hybrid",
                _hybrid_tickets(candidate, groups, context, count=count),
                candidate,
                context,
            ),
        }
    width_key = str(default_width if default_width in width_values else width_values[0])
    production = bundles[width_key][strategy_name]
    return {
        "schema": SCHEMA,
        "count": int(count),
        "policy": {
            "strategy": strategy_name,
            "width": int(width_key),
            "reason": str(chosen.get("reason") or "configured_default"),
            "source": "pre_target_only",
        },
        "groups": groups_by_width,
        "bundles": bundles,
        "production_tickets": production["tickets"],
    }


def evaluate_final5_counterfactual(
    prepared: dict[str, Any],
    actual_red: Iterable[int],
    *,
    applied: bool,
) -> dict[str, Any]:
    policy = prepared.get("policy") or {}
    width_key = str(int(policy.get("width") or 0))
    bundles = prepared.get("bundles") or {}
    comparison = []
    width_sweep: dict[str, Any] = {}
    observation_widths: dict[str, Any] = {}
    for key, bundle in bundles.items():
        candidate = bundle.get("candidate") or []
        strategy_metrics: dict[str, Any] = {}
        for name in STRATEGIES:
            item = bundle.get(name) or {}
            outcome = evaluate_red_outcome(item.get("tickets") or [], candidate, actual_red)
            strategy_metrics[name] = {
                "any3_plus": outcome["any3_plus"],
                "any4_plus": outcome["any4_plus"],
                "best_ticket_hit": outcome["best_ticket_hit"],
                "final5_hit": outcome["final5_hit"],
                "final5_drop_regret": outcome["final5_drop_regret"],
                "candidate_hit": outcome["candidate_hit"],
                "candidate_recall": outcome["candidate_recall"],
            }
            if key == width_key:
                comparison.append({
                    "strategy": name,
                    "label": STRATEGY_LABELS[name],
                    "candidate_recall": outcome["candidate_recall"],
                    "candidate_hit": outcome["candidate_hit"],
                    "final5_hit": outcome["final5_hit"],
                    "final5_drop_regret": outcome["final5_drop_regret"],
                    "dropped_numbers": outcome["dropped_numbers"],
                    "any3_plus": outcome["any3_plus"],
                    "any4_plus": outcome["any4_plus"],
                    "best_ticket_hit": outcome["best_ticket_hit"],
                    "split_regret": outcome["split_regret"],
                    "attribution": outcome["attribution"],
                    "pre_target_score": (item.get("pre_target_score") or {}).get("score"),
                    "reds": item.get("reds") or [],
                })
        width_sweep[key] = {
            "candidate_recall": strategy_metrics.get("hit_coherence", {}).get("candidate_recall", 0.0),
            "candidate_hit": strategy_metrics.get("hit_coherence", {}).get("candidate_hit", 0),
            "strategies": strategy_metrics,
        }
        observation_widths[key] = {
            "candidate_recall": width_sweep[key]["candidate_recall"],
            "strategies": strategy_metrics,
        }
    production_name = str(policy.get("strategy") or "hit_coherence") if applied else "current"
    production_row = next((row for row in comparison if row["strategy"] == production_name), {})
    return {
        "schema": SCHEMA,
        "active": True,
        "applied": bool(applied),
        "policy": policy,
        "groups": (prepared.get("groups") or {}).get(width_key, []),
        "objective": {
            "primary": "any3_plus",
            "secondary": "any4_plus",
            "tertiary": "best_ticket_hit",
            "constraint": "candidate_recall_must_not_drop_materially",
            "weights": dict(SCORE_WEIGHTS),
        },
        "production": production_row,
        "comparison": comparison,
        "width_sweep": width_sweep,
        "policy_observation": {"widths": observation_widths},
    }


def choose_final5_policy(
    history: list[dict[str, Any]] | None,
    *,
    default_strategy: str = "hit_coherence",
    default_width: int = 22,
    min_periods: int = 8,
    recall_drop_limit: float = 0.05,
) -> dict[str, Any]:
    """Pick next period's strategy and width from completed periods only."""

    rows = [row for row in (history or []) if isinstance(row, dict)]
    fallback = {
        "strategy": default_strategy if default_strategy in STRATEGIES else "hit_coherence",
        "width": int(default_width),
        "reason": "insufficient_pre_target_history",
        "periods": len(rows),
    }
    if len(rows) < max(1, int(min_periods)):
        return fallback
    recalls: dict[int, list[float]] = {}
    metrics: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for row in rows:
        for key, payload in (row.get("widths") or {}).items():
            try:
                width = int(key)
            except (TypeError, ValueError):
                continue
            recalls.setdefault(width, []).append(float(payload.get("candidate_recall") or 0.0))
            for name, values in (payload.get("strategies") or {}).items():
                if name not in STRATEGIES or not isinstance(values, dict):
                    continue
                metrics.setdefault((str(name), width), []).append((
                    float(values.get("any3_plus") or 0.0),
                    float(values.get("any4_plus") or 0.0),
                    float(values.get("best_ticket_hit") or 0.0),
                ))
    if not recalls or not metrics:
        return fallback
    recall_mean = {width: _mean(values) for width, values in recalls.items()}
    best_recall = max(recall_mean.values())
    eligible = {
        width for width, value in recall_mean.items()
        if value >= best_recall - float(recall_drop_limit)
    }
    ranked = []
    for (name, width), values in metrics.items():
        if width not in eligible or len(values) < max(1, int(min_periods)):
            continue
        ranked.append((
            _mean(item[0] for item in values),
            _mean(item[1] for item in values),
            _mean(item[2] for item in values),
            STRATEGY_TIE_RANK.get(name, 0),
            -int(width),
            name,
            int(width),
        ))
    if not ranked:
        return fallback
    best = max(ranked)
    return {
        "strategy": best[5],
        "width": int(best[6]),
        "reason": "walk_forward_any3_any4_best_with_recall_constraint",
        "periods": len(rows),
        "eligible_widths": sorted(eligible),
        "candidate_recall_by_width": {str(width): float(value) for width, value in sorted(recall_mean.items())},
    }


def graft_ticket_reds(
    base_tickets: list[dict[str, Any]],
    planned_tickets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not base_tickets or not planned_tickets:
        return [dict(ticket) for ticket in base_tickets]
    grafted: list[dict[str, Any]] = []
    for index, planned in enumerate(planned_tickets):
        source = base_tickets[index] if index < len(base_tickets) else base_tickets[-1]
        ticket = dict(source)
        ticket["red"] = list(planned.get("red") or [])
        ticket["selection_role"] = planned.get("selection_role", ticket.get("selection_role"))
        ticket["final5_hit_coherence"] = planned.get("final5_hit_coherence", {})
        grafted.append(ticket)
    return grafted


def apply_hit_coherence_final5(
    tickets: list[dict[str, Any]],
    *,
    fusion_red: Iterable[float] | None,
    history_draws: list[tuple[int, ...]] | None,
    method_tickets: dict[str, list[dict[str, Any]]] | None = None,
    pool: list[dict[str, Any]] | None = None,
    model_probabilities: dict[str, dict[str, Any]] | None = None,
    model_rankings: dict[str, dict[str, Any]] | None = None,
    rescue_numbers: Iterable[int] | None = None,
    width: int = 22,
    count: int = 5,
    policy_history: list[dict[str, Any]] | None = None,
    min_periods: int = 8,
    recall_drop_limit: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = choose_final5_policy(
        policy_history,
        default_strategy="hit_coherence",
        default_width=int(width),
        min_periods=int(min_periods),
        recall_drop_limit=float(recall_drop_limit),
    )
    prepared = prepare_final5_counterfactual(
        current_tickets=tickets,
        fusion_red=fusion_red,
        history_draws=history_draws,
        method_tickets=method_tickets,
        pool=pool,
        model_probabilities=model_probabilities,
        model_rankings=model_rankings,
        rescue_numbers=rescue_numbers,
        widths=CANDIDATE_WIDTHS,
        count=count,
        policy=policy,
    )
    updated = graft_ticket_reds(tickets, prepared.get("production_tickets") or [])
    info = {
        "schema": SCHEMA,
        "active": bool(updated),
        "strategy": prepared["policy"]["strategy"],
        "width": prepared["policy"]["width"],
        "reason": prepared["policy"]["reason"],
        "groups": (prepared.get("groups") or {}).get(str(prepared["policy"]["width"]), []),
    }
    return updated or [dict(ticket) for ticket in tickets], {**info, "prepared": prepared}


def summarize_final5_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        record.get("final5_hit_coherence")
        for record in records
        if isinstance(record.get("final5_hit_coherence"), dict) and record["final5_hit_coherence"].get("active")
    ]
    if not reports:
        return {"schema": SCHEMA, "periods": 0}
    comparison: dict[str, dict[str, float]] = {}
    for name in STRATEGIES:
        rows = []
        for report in reports:
            match = next((item for item in report.get("comparison") or [] if item.get("strategy") == name), None)
            if match:
                rows.append(match)
        if not rows:
            continue
        comparison[name] = {
            "label": STRATEGY_LABELS[name],
            "candidate_recall": _mean(float(item.get("candidate_recall") or 0.0) for item in rows),
            "any3_plus": _mean(float(item.get("any3_plus") or 0.0) for item in rows),
            "any4_plus": _mean(float(item.get("any4_plus") or 0.0) for item in rows),
            "best_ticket_hit": _mean(float(item.get("best_ticket_hit") or 0.0) for item in rows),
            "final5_drop_regret": _mean(float(item.get("final5_drop_regret") or 0.0) for item in rows),
            "periods": float(len(rows)),
        }
    regrets = [float((report.get("production") or {}).get("final5_drop_regret") or 0.0) for report in reports]
    owners: dict[str, int] = {}
    for report in reports:
        owner = str((report.get("production") or {}).get("attribution") or "unknown")
        owners[owner] = owners.get(owner, 0) + 1
    return {
        "schema": SCHEMA,
        "periods": len(reports),
        "comparison": comparison,
        "avg_production_drop_regret": _mean(regrets),
        "production_attribution_counts": owners,
        "objective": {
            "primary": "any3_plus",
            "secondary": "any4_plus",
            "tertiary": "best_ticket_hit",
        },
    }
