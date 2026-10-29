from __future__ import annotations

from typing import Any


def _red_set(ticket: dict[str, Any]) -> set[int]:
    return {int(value) for value in ticket.get("red", []) or [] if 1 <= int(value) <= 33}


def _ticket_key(ticket: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(_red_set(ticket))), int(ticket.get("blue", 0) or 0)


def _quality(ticket: dict[str, Any]) -> float:
    scores = ticket.get("candidate_scores") if isinstance(ticket.get("candidate_scores"), dict) else {}
    return float(ticket.get("quality_score", ticket.get("score", scores.get("quality_score", 0.0))) or 0.0)


def _blue_score(ticket: dict[str, Any]) -> float:
    return float(ticket.get("blue_number_score", ticket.get("blue_signal_bonus", 0.0)) or 0.0)


def _candidate_red_evidence(pool: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    evidence = {number: {"number": number, "score": 0.0, "appearance": 0, "best_rank": None} for number in range(1, 34)}
    sorted_pool = sorted(
        [item for item in pool or [] if len(_red_set(item)) == 6],
        key=_quality,
        reverse=True,
    )
    max_quality = max((_quality(item) for item in sorted_pool), default=1.0) or 1.0
    for rank, ticket in enumerate(sorted_pool, start=1):
        quality = max(0.0, _quality(ticket)) / float(max_quality)
        weight = quality + 1.0 / (1.0 + rank)
        for number in _red_set(ticket):
            row = evidence[int(number)]
            row["score"] = float(row["score"]) + float(weight)
            row["appearance"] = int(row["appearance"]) + 1
            if row["best_rank"] is None:
                row["best_rank"] = int(rank)
    return {
        number: row
        for number, row in evidence.items()
        if int(row["appearance"]) > 0
    }


def _retained_red_evidence(
    tickets: list[dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
    protected: set[int],
) -> tuple[float, float, float, list[int]]:
    final_union = set().union(*[_red_set(ticket) for ticket in tickets]) if tickets else set()
    candidate_numbers = set(evidence) | set(protected)
    total = sum(float(evidence.get(number, {}).get("score", 0.0)) + (0.5 if number in protected else 0.0) for number in candidate_numbers)
    retained = sum(float(evidence.get(number, {}).get("score", 0.0)) + (0.5 if number in protected else 0.0) for number in final_union & candidate_numbers)
    rate = retained / total if total > 0 else 0.0
    return float(total), float(retained), float(rate), sorted(candidate_numbers - final_union)


def _final_union(tickets: list[dict[str, Any]]) -> set[int]:
    return set().union(*[_red_set(ticket) for ticket in tickets]) if tickets else set()


def _unique_ordered(values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = int(value)
        if number in seen or not (1 <= number <= 33):
            continue
        ordered.append(number)
        seen.add(number)
    return ordered


def _rank_numbers_by_evidence(
    numbers: set[int] | list[int],
    evidence: dict[int, dict[str, Any]],
) -> list[int]:
    return [
        int(number)
        for number in sorted(
            {int(value) for value in numbers if 1 <= int(value) <= 33},
            key=lambda value: (
                float(evidence.get(int(value), {}).get("score", 0.0)),
                int(evidence.get(int(value), {}).get("appearance", 0) or 0),
                -int(value),
            ),
            reverse=True,
        )
    ]


def _build_hard_and_soft_protection(
    *,
    top_candidate_numbers: list[int],
    v8_must_preserve: list[int],
    protected_numbers: set[int],
    candidate_red_evidence: dict[int, dict[str, Any]],
    guard_cfg: dict[str, Any],
) -> tuple[set[int], set[int], dict[str, Any]]:
    """Split broad evidence into a small hard set and a diagnostic soft set.

    Final Guard is a last safety pass, not a second portfolio optimizer.  The
    guard therefore receives a capped hard set that is feasible for a 5-ticket
    portfolio, while the remaining evidence stays visible as soft protection in
    diagnostics.
    """
    hard_limit = max(1, int(guard_cfg.get("hard_protect_limit", 12) or 12))
    must_limit = max(0, int(guard_cfg.get("hard_must_preserve_limit", 8) or 8))
    recovery_limit = max(0, int(guard_cfg.get("hard_recovery_protected_limit", 4) or 4))
    anchor_limit = max(0, int(guard_cfg.get("hard_anchor_limit", 3) or 3))

    must_ranked = _rank_numbers_by_evidence(v8_must_preserve, candidate_red_evidence)[:must_limit]
    recovery_ranked = _rank_numbers_by_evidence(protected_numbers, candidate_red_evidence)[:recovery_limit]
    anchor_ranked = _rank_numbers_by_evidence(top_candidate_numbers, candidate_red_evidence)[:anchor_limit]

    hard_ordered = _unique_ordered(must_ranked + recovery_ranked + anchor_ranked)[:hard_limit]
    broad_ordered = _unique_ordered(
        list(v8_must_preserve)
        + list(protected_numbers)
        + list(top_candidate_numbers)
    )
    hard = set(hard_ordered)
    soft = set(broad_ordered) - hard
    audit = {
        "hard_protect_limit": hard_limit,
        "hard_must_preserve_limit": must_limit,
        "hard_recovery_protected_limit": recovery_limit,
        "hard_anchor_limit": anchor_limit,
        "hard_protected_numbers": hard_ordered,
        "soft_protected_numbers": sorted(soft),
        "broad_protected_count": len(broad_ordered),
    }
    return hard, soft, audit


def _ticket_change_count(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    return sum(
        1
        for left, right in zip(before, after)
        if _ticket_key(left) != _ticket_key(right)
    ) + abs(len(before) - len(after))


def _red_ticket_change_count(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> int:
    return sum(
        1
        for left, right in zip(before, after)
        if _red_set(left) != _red_set(right)
    ) + abs(len(before) - len(after))


def _allowed_blues_from_pool(
    tickets: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    runtime_control: dict[str, Any],
) -> list[int]:
    prediction = runtime_control.get("blue_prediction_set") if isinstance(runtime_control, dict) else {}
    policy = prediction.get("policy") if isinstance(prediction, dict) else {}
    raw_allowed = prediction.get("allowed_blues") if isinstance(prediction, dict) else None
    if isinstance(raw_allowed, list) and raw_allowed:
        return [int(value) for value in raw_allowed if 1 <= int(value) <= 16]
    width = int((prediction or {}).get("selected_width", (policy or {}).get("selected_width", 16)) or 16)
    width = max(1, min(16, width))
    blue_scores = {number: 0.0 for number in range(1, 17)}
    for item in list(pool or []) + list(tickets or []):
        try:
            blue = int(item.get("blue", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 1 <= blue <= 16:
            blue_scores[blue] += max(1e-9, _quality(item)) + 0.35 * max(0.0, _blue_score(item))
    ranked = sorted(range(1, 17), key=lambda number: (blue_scores.get(number, 0.0), -number), reverse=True)
    return ranked[:width]


def _max_overlap(ticket: dict[str, Any], selected: list[dict[str, Any]]) -> int:
    red = _red_set(ticket)
    return max((len(red & _red_set(other)) for other in selected), default=0)


def _replace_blue_only(ticket: dict[str, Any], blue: int, reason: str) -> dict[str, Any]:
    return {
        **ticket,
        "blue": int(blue),
        "final_v7_guard": True,
        "blue_selection_reason": {
            **(ticket.get("blue_selection_reason") if isinstance(ticket.get("blue_selection_reason"), dict) else {}),
            "final_v7_guard": reason,
        },
    }


def apply_final_v7_guard(
    tickets: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    runtime_control: dict[str, Any],
    minimum_unique_blues: int,
    v7_portfolio_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the last non-bypassable V7 constraint pass before validation/reporting."""
    result = [dict(ticket) for ticket in tickets or []]
    if not result:
        return result, {"schema": "final_v7_guard_v8", "active": False, "reason": "empty_tickets"}
    count = len(result)
    max_overlap = int((runtime_control or {}).get("red_jaccard_max_overlap", 4) or 4)
    protected_numbers = {
        int(value)
        for value in (runtime_control or {}).get("red_protected_numbers", []) or []
        if 1 <= int(value) <= 33
    }
    blue_target = max(1, min(int(minimum_unique_blues), count, 16))
    allowed_blues = _allowed_blues_from_pool(result, pool, runtime_control or {})
    allowed_set = set(allowed_blues)
    guard_mode = str((runtime_control or {}).get("final_v7_guard_mode", "diagnostic")).strip().lower()
    guard_cfg = (
        (runtime_control or {}).get("final_v7_guard")
        if isinstance((runtime_control or {}).get("final_v7_guard"), dict)
        else {}
    )
    hard_constraints_only = bool(
        (runtime_control or {}).get(
            "final_v7_guard_hard_constraints_only",
            guard_cfg.get("hard_constraints_only", guard_mode in {"diagnostic", "hard_constraints_only", "check_only"}),
        )
    )
    max_red_ticket_changes = int(
        (runtime_control or {}).get(
            "final_v7_guard_max_ticket_changes",
            guard_cfg.get("max_ticket_changes", count if hard_constraints_only else 1),
        )
        or 0
    )
    if hard_constraints_only:
        max_red_ticket_changes = max(max_red_ticket_changes, count)
    else:
        max_red_ticket_changes = max(0, min(count, max_red_ticket_changes))
    max_union_loss = int(guard_cfg.get("max_union_loss", 1) or 1)
    max_retention_loss = float(guard_cfg.get("max_retention_loss", 0.03) or 0.03)
    rollback_on_loss = bool(guard_cfg.get("rollback_on_excessive_loss", True))
    events: list[dict[str, Any]] = []

    used_blues: set[int] = set()
    for index, ticket in enumerate(result):
        blue = int(ticket.get("blue", 0) or 0)
        if blue in allowed_set and blue not in used_blues:
            used_blues.add(blue)
            continue
        replacement = next((value for value in allowed_blues if value not in used_blues), None)
        if replacement is None and blue in allowed_set:
            replacement = blue
        if replacement is not None:
            events.append({"constraint": "Prediction Set / Blue Diversity", "position": index + 1, "from": blue, "to": int(replacement)})
            result[index] = _replace_blue_only(ticket, int(replacement), "final_v7_guard_prediction_set_blue_diversity")
            used_blues.add(int(replacement))

    if len({int(ticket.get("blue", 0) or 0) for ticket in result}) < blue_target:
        for index, ticket in enumerate(result):
            current = {int(item.get("blue", 0) or 0) for item in result}
            if len(current) >= blue_target:
                break
            replacement = next((value for value in allowed_blues if value not in current), None)
            if replacement is None:
                break
            old = int(ticket.get("blue", 0) or 0)
            result[index] = _replace_blue_only(ticket, int(replacement), "final_v7_guard_minimum_blue_unique")
            events.append({"constraint": "Blue Diversity", "position": index + 1, "from": old, "to": int(replacement)})

    result_after_blue = [dict(ticket) for ticket in result]
    portfolio_info = v7_portfolio_info if isinstance(v7_portfolio_info, dict) else {}
    portfolio_retention = (
        portfolio_info.get("information_retention")
        if isinstance(portfolio_info.get("information_retention"), dict)
        else {}
    )
    v7_candidate_numbers = {
        int(number)
        for number in (
            list((portfolio_retention or {}).get("retained_candidate_numbers") or [])
            + list((portfolio_retention or {}).get("dropped_candidate_numbers") or [])
        )
        if 1 <= int(number) <= 33
    }
    candidate_red_evidence = _candidate_red_evidence(pool)
    if v7_candidate_numbers:
        candidate_red_evidence = {
            number: row
            for number, row in candidate_red_evidence.items()
            if int(number) in v7_candidate_numbers
        }
    top_candidate_numbers = [
        int(number)
        for number, _ in sorted(
            candidate_red_evidence.items(),
            key=lambda item: (
                float(item[1].get("score", 0.0)),
                int(item[1].get("appearance", 0)),
                -int(item[0]),
            ),
            reverse=True,
        )[: max(8, min(14, count + 6))]
    ]
    red_pools = portfolio_info.get("red_pools") if isinstance(portfolio_info.get("red_pools"), dict) else {}
    v8_must_preserve = [
        int(value)
        for value in (red_pools or {}).get("must_preserve", []) or []
        if 1 <= int(value) <= 33
    ]
    if v8_must_preserve:
        top_candidate_numbers = list(dict.fromkeys(v8_must_preserve + top_candidate_numbers))
    must_keep_numbers, soft_protect_numbers, protection_audit = _build_hard_and_soft_protection(
        top_candidate_numbers=top_candidate_numbers,
        v8_must_preserve=v8_must_preserve,
        protected_numbers=protected_numbers,
        candidate_red_evidence=candidate_red_evidence,
        guard_cfg=guard_cfg,
    )
    _, _, baseline_retention_rate, baseline_dropped = _retained_red_evidence(
        result_after_blue,
        candidate_red_evidence,
        must_keep_numbers,
    )
    baseline_union_count = len(_final_union(result_after_blue) & (set(candidate_red_evidence) | set(must_keep_numbers)))
    _, _, initial_retention_rate, initially_dropped = _retained_red_evidence(
        result,
        candidate_red_evidence,
        must_keep_numbers,
    )
    used_keys = {_ticket_key(ticket) for ticket in result}
    for missing_number in ([] if hard_constraints_only else [
        number for number in top_candidate_numbers
        if number in set(initially_dropped)
    ]):
        current_union = set().union(*[_red_set(ticket) for ticket in result]) if result else set()
        if missing_number in current_union:
            continue
        if _red_ticket_change_count(result_after_blue, result) >= max_red_ticket_changes:
            events.append({
                "constraint": "Change Budget",
                "number": int(missing_number),
                "reason": "final_guard_red_change_budget_exhausted_before_restoring_soft_evidence",
                "max_red_ticket_changes": int(max_red_ticket_changes),
            })
            break
        _, _, current_retention_rate, _ = _retained_red_evidence(
            result,
            candidate_red_evidence,
            must_keep_numbers,
        )
        candidate_pool = [
            dict(item)
            for item in pool or []
            if missing_number in _red_set(item)
            and len(_red_set(item)) == 6
        ]
        if not candidate_pool:
            continue
        best_trial = None
        best_score = float("-inf")
        for position, current_ticket in enumerate(result):
            others = result[:position] + result[position + 1 :]
            for candidate in candidate_pool:
                candidate_blue = int(candidate.get("blue", 0) or 0)
                current_blue = int(current_ticket.get("blue", candidate_blue) or candidate_blue)
                if allowed_set and current_blue in allowed_set:
                    candidate = {**candidate, "blue": current_blue}
                elif allowed_set:
                    replacement_blue = next((blue for blue in allowed_blues if blue not in {int(item.get("blue", 0) or 0) for item in others}), allowed_blues[0] if allowed_blues else candidate_blue)
                    candidate = {**candidate, "blue": int(replacement_blue)}
                if _ticket_key(candidate) in used_keys and _ticket_key(candidate) != _ticket_key(current_ticket):
                    continue
                overlap = _max_overlap(candidate, others)
                total, retained, retention_rate, dropped = _retained_red_evidence(
                    others + [candidate],
                    candidate_red_evidence,
                    must_keep_numbers,
                )
                dropped_protected = len(set(dropped) & must_keep_numbers)
                score = (
                    8.0 * retention_rate
                    + 0.02 * retained
                    + 0.4 * len(_red_set(candidate) & must_keep_numbers)
                    - 0.35 * max(0, overlap - max_overlap)
                    - 0.45 * dropped_protected
                    + 0.01 * _quality(candidate)
                )
                if score > best_score:
                    best_score = float(score)
                    best_trial = (position, candidate, overlap, retention_rate, dropped)
        if best_trial is None:
            events.append({
                "constraint": "Information Retention",
                "number": int(missing_number),
                "reason": "high_evidence_number_found_but_no_feasible_final_ticket",
                "evidence": candidate_red_evidence.get(missing_number, {}),
            })
            continue
        position, replacement, overlap, retention_rate, dropped = best_trial
        if retention_rate + 1e-12 < current_retention_rate:
            events.append({
                "constraint": "Information Retention",
                "number": int(missing_number),
                "reason": "candidate_replacement_would_reduce_information_retention",
                "evidence": candidate_red_evidence.get(missing_number, {}),
            })
            continue
        old_ticket = result[position]
        used_keys.discard(_ticket_key(old_ticket))
        result[position] = {
            **replacement,
            "final_v7_guard": True,
            "selection_role": f"{replacement.get('selection_role', 'candidate')}_information_retention_guard",
            "selection_reason": {
                **(replacement.get("selection_reason") if isinstance(replacement.get("selection_reason"), dict) else {}),
                "final_v7_guard": "restore_high_evidence_candidate_number_before_finite_ticket_compression",
                "protected_number": int(missing_number),
                "retention_rate_after": float(retention_rate),
            },
        }
        used_keys.add(_ticket_key(result[position]))
        initial_retention_rate = max(initial_retention_rate, float(retention_rate))
        events.append({
            "constraint": "Information Retention",
            "position": int(position + 1),
            "number": int(missing_number),
            "from": sorted(_red_set(old_ticket)),
            "to": sorted(_red_set(result[position])),
            "max_overlap_after": int(overlap),
            "retention_rate_after": float(retention_rate),
            "reason": "high_evidence_number_was_found_in_candidate_pool_and_restored_to_final_portfolio",
            "evidence": candidate_red_evidence.get(missing_number, {}),
        })

    pool_candidates = sorted(
        [dict(item) for item in pool or [] if len(_red_set(item)) == 6],
        key=lambda item: (
            len(_red_set(item) & protected_numbers),
            -_max_overlap(item, result),
            _quality(item),
        ),
        reverse=True,
    )
    used_keys = {_ticket_key(ticket) for ticket in result}
    for index, ticket in enumerate(list(result)):
        if _red_ticket_change_count(result_after_blue, result) >= max_red_ticket_changes:
            events.append({
                "constraint": "Change Budget",
                "position": index + 1,
                "reason": "final_guard_red_change_budget_exhausted_before_jaccard_repair",
                "max_red_ticket_changes": int(max_red_ticket_changes),
            })
            break
        others = result[:index] + result[index + 1 :]
        violates_overlap = _max_overlap(ticket, others) > max_overlap
        # Protection is a portfolio-level union target.  Requiring every ticket
        # to touch protected numbers makes Final Guard a second optimizer and
        # repeatedly replaces otherwise valid tickets.
        misses_protected = False
        if not violates_overlap and not misses_protected:
            continue
        replacement = None
        for candidate in pool_candidates:
            key = _ticket_key(candidate)
            if key in used_keys:
                continue
            candidate_blue = int(candidate.get("blue", 0) or 0)
            current_blue = int(result[index].get("blue", candidate_blue) or candidate_blue)
            if allowed_set and current_blue in allowed_set:
                candidate = {**candidate, "blue": current_blue}
            elif allowed_set and candidate_blue not in allowed_set:
                candidate = {**candidate, "blue": int(next(iter(allowed_set)))}
            if _max_overlap(candidate, others) > max_overlap:
                continue
            if misses_protected and protected_numbers and not (_red_set(candidate) & protected_numbers):
                continue
            replacement = {
                **candidate,
                "final_v7_guard": True,
                "selection_role": f"{candidate.get('selection_role', 'candidate')}_final_v7_guard",
                "selection_reason": {
                    **(candidate.get("selection_reason") if isinstance(candidate.get("selection_reason"), dict) else {}),
                    "final_v7_guard": "restore_recall_jaccard_protected_constraints_after_rolling_memory",
                },
            }
            break
        if replacement is not None:
            used_keys.discard(_ticket_key(result[index]))
            result[index] = replacement
            used_keys.add(_ticket_key(replacement))
            events.append({
                "constraint": "Recall / Jaccard / Protected",
                "position": index + 1,
                "reason": "rolling_or_blue_step_changed_final_ticket_constraints",
            })

    candidate_evidence_total_after_red, retained_after_red, retention_after_red, _ = _retained_red_evidence(
        result,
        candidate_red_evidence,
        must_keep_numbers,
    )
    red_union_count_after = len(_final_union(result) & (set(candidate_red_evidence) | set(must_keep_numbers)))
    red_ticket_changes = _red_ticket_change_count(result_after_blue, result)
    union_loss = int(baseline_union_count - red_union_count_after)
    retention_loss = float(baseline_retention_rate - retention_after_red)
    if (
        rollback_on_loss
        and red_ticket_changes > 0
        and (
            red_ticket_changes > max_red_ticket_changes
            or union_loss > max_union_loss
            or retention_loss > max_retention_loss
        )
    ):
        events.append({
            "constraint": "Guard Stop Rule",
            "decision": "rollback_red_changes",
            "reason": "final_guard_red_repair_exceeded_change_or_information_loss_budget",
            "red_ticket_changes": int(red_ticket_changes),
            "max_red_ticket_changes": int(max_red_ticket_changes),
            "union_loss": int(union_loss),
            "max_union_loss": int(max_union_loss),
            "retention_loss": float(retention_loss),
            "max_retention_loss": float(max_retention_loss),
        })
        result = result_after_blue
        candidate_evidence_total_after_red, retained_after_red, retention_after_red, _ = _retained_red_evidence(
            result,
            candidate_red_evidence,
            must_keep_numbers,
        )

    final_overlaps = [
        len(_red_set(left) & _red_set(right))
        for left_index, left in enumerate(result)
        for right in result[left_index + 1 :]
    ]
    final_union_for_violations = _final_union(result)
    missing_must_preserve = sorted(set(must_keep_numbers) - final_union_for_violations)
    missing_soft_protect = sorted(set(soft_protect_numbers) - final_union_for_violations)
    violations = {
        "jaccard": any(value > max_overlap for value in final_overlaps),
        "blue_diversity": len({int(ticket.get("blue", 0) or 0) for ticket in result}) < blue_target,
        "prediction_set": bool(allowed_set) and any(int(ticket.get("blue", 0) or 0) not in allowed_set for ticket in result),
        "protected": False if hard_constraints_only else bool(missing_must_preserve),
    }
    candidate_evidence_total, retained_evidence, retention_rate, dropped_candidate_numbers = _retained_red_evidence(
        result,
        candidate_red_evidence,
        must_keep_numbers,
    )
    if v7_candidate_numbers:
        final_union = _final_union(result)
        dropped_candidate_numbers = sorted(v7_candidate_numbers - final_union)
        candidate_number_count = len(v7_candidate_numbers)
        final_union_count = len(final_union & v7_candidate_numbers)
        information_scope = "v7_portfolio_candidate_set"
    else:
        final_union = _final_union(result)
        candidate_number_count = len(set(candidate_red_evidence) | set(must_keep_numbers))
        final_union_count = len(final_union & (set(candidate_red_evidence) | set(must_keep_numbers)))
        information_scope = "guard_candidate_pool_fallback"
    ticket_changes_total = _ticket_change_count(result_after_blue, result)
    red_ticket_changes_total = _red_ticket_change_count(result_after_blue, result)
    return result[:count], {
        "schema": "final_v7_guard_v8",
        "active": True,
        "pipeline": ["V7 Portfolio", "Rolling Memory", "Blue Finalize", "Final V7 Guard"],
        "constraints": (
            ["Jaccard", "Blue Diversity", "Prediction Set", "Legal Number Range"]
            if hard_constraints_only
            else ["Recall", "Jaccard", "Protected", "Blue Diversity", "Prediction Set"]
        ),
        "mode": "hard_constraints_only" if hard_constraints_only else "information_restoration_enabled",
        "jaccard_max_red_overlap": max_overlap,
        "minimum_unique_blues": blue_target,
        "allowed_blues": allowed_blues,
        "protected_numbers": sorted(protected_numbers),
        "change_budget": {
            "max_red_ticket_changes": int(max_red_ticket_changes),
            "red_ticket_changes": int(red_ticket_changes_total),
            "ticket_changes_after_blue": int(ticket_changes_total),
            "max_union_loss": int(max_union_loss),
            "max_retention_loss": float(max_retention_loss),
            "rollback_on_excessive_loss": bool(rollback_on_loss),
        },
        "information_retention": {
            "scope": information_scope,
            "candidate_evidence_total": float(candidate_evidence_total),
            "retained_evidence": float(retained_evidence),
            "information_retention_rate": float(retention_rate),
            "compression_regret": float(max(0.0, 1.0 - retention_rate)),
            "candidate_number_count": int(candidate_number_count),
            "final_union_count": int(final_union_count),
            "protected_candidate_numbers": sorted(must_keep_numbers),
            "soft_protected_candidate_numbers": sorted(soft_protect_numbers),
            "broad_protected_candidate_count": int(len(must_keep_numbers | soft_protect_numbers)),
            "must_preserve_numbers": sorted(set(must_keep_numbers)),
            "missing_must_preserve_numbers": missing_must_preserve,
            "missing_soft_protect_numbers": missing_soft_protect,
            "dropped_candidate_numbers": dropped_candidate_numbers,
        },
        "protection_policy": protection_audit,
        "guard_delta": {
            "baseline_union_count": int(baseline_union_count),
            "final_union_count": int(final_union_count),
            "union_delta": int(final_union_count - baseline_union_count),
            "baseline_information_retention_rate": float(baseline_retention_rate),
            "final_information_retention_rate": float(retention_rate),
            "retention_delta": float(retention_rate - baseline_retention_rate),
        },
        "events": events,
        "violations": violations,
        "passed": not any(violations.values()),
    }
