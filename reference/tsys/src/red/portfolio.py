from __future__ import annotations

from itertools import combinations
from typing import Any

from src.portfolio.budget_allocator import allocate_portfolio_budget
from src.portfolio.pareto import pareto_frontier


def _ticket_key(ticket: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(int(value) for value in ticket.get("red", []))), int(ticket.get("blue", 0))


def _red_set(ticket: dict[str, Any]) -> set[int]:
    return {int(value) for value in ticket.get("red", [])}


def _quality(ticket: dict[str, Any]) -> float:
    scores = ticket.get("candidate_scores") if isinstance(ticket.get("candidate_scores"), dict) else {}
    return float(
        ticket.get(
            "score",
            ticket.get("quality_score", scores.get("quality_score", scores.get("base_score", 0.0))),
        )
        or 0.0
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _method_support_map(selected_by_method: dict[str, list[dict[str, Any]]]) -> dict[tuple[tuple[int, ...], int], list[str]]:
    support: dict[tuple[tuple[int, ...], int], list[str]] = {}
    for method, tickets in (selected_by_method or {}).items():
        for ticket in tickets or []:
            support.setdefault(_ticket_key(ticket), []).append(str(method))
    return support


def _method_red_support(selected_by_method: dict[str, list[dict[str, Any]]]) -> dict[int, int]:
    support: dict[int, int] = {number: 0 for number in range(1, 34)}
    seen: set[tuple[str, int]] = set()
    for method, tickets in (selected_by_method or {}).items():
        for ticket in tickets or []:
            for number in _red_set(ticket):
                key = (str(method), int(number))
                if key in seen:
                    continue
                support[int(number)] += 1
                seen.add(key)
    return support


def _pool_red_evidence(
    pool_items: list[dict[str, Any]],
    *,
    red_support: dict[int, int],
    protected_numbers: set[int],
) -> dict[int, dict[str, Any]]:
    """Aggregate candidate-level information before final compression.

    The V7 red layer treats the candidate pool as the information source and
    asks how much of that information survives into the finite portfolio.  This
    is intentionally number-level evidence, not a single-ticket score.
    """

    max_quality = max((_quality(ticket) for ticket in pool_items), default=1.0) or 1.0
    appearance: dict[int, int] = {number: 0 for number in range(1, 34)}
    quality_sum: dict[int, float] = {number: 0.0 for number in range(1, 34)}
    probability_sum: dict[int, float] = {number: 0.0 for number in range(1, 34)}
    structure_sum: dict[int, float] = {number: 0.0 for number in range(1, 34)}
    for ticket in pool_items:
        scores = ticket.get("candidate_scores") if isinstance(ticket.get("candidate_scores"), dict) else {}
        ticket_quality = _quality(ticket) / float(max_quality)
        ticket_probability = _safe_float(
            ticket.get("red_number_score", scores.get("red_number_score")),
            ticket_quality,
        )
        ticket_structure = _safe_float(
            ticket.get("structure_score", scores.get("structure_score")),
            1.0,
        )
        for number in _red_set(ticket):
            if not 1 <= int(number) <= 33:
                continue
            appearance[int(number)] += 1
            quality_sum[int(number)] += ticket_quality
            probability_sum[int(number)] += ticket_probability
            structure_sum[int(number)] += ticket_structure
    max_appearance = max(appearance.values(), default=1) or 1
    max_quality_sum = max(quality_sum.values(), default=1.0) or 1.0
    max_probability_sum = max(probability_sum.values(), default=1.0) or 1.0
    max_structure_sum = max(structure_sum.values(), default=1.0) or 1.0
    max_support = max(red_support.values(), default=1) or 1
    result: dict[int, dict[str, Any]] = {}
    for number in range(1, 34):
        if appearance[number] <= 0 and red_support.get(number, 0) <= 0 and number not in protected_numbers:
            continue
        evidence = (
            0.30 * float(appearance[number]) / float(max_appearance)
            + 0.28 * float(quality_sum[number]) / float(max_quality_sum)
            + 0.22 * float(probability_sum[number]) / float(max_probability_sum)
            + 0.12 * float(red_support.get(number, 0)) / float(max_support)
            + 0.06 * float(structure_sum[number]) / float(max_structure_sum)
            + (0.08 if number in protected_numbers else 0.0)
        )
        result[number] = {
            "number": int(number),
            "evidence": float(evidence),
            "appearance": int(appearance[number]),
            "method_support": int(red_support.get(number, 0)),
            "quality_mass": float(quality_sum[number]),
            "probability_mass": float(probability_sum[number]),
            "protected": bool(number in protected_numbers),
        }
    return result


def _build_red_layers(
    evidence: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    core_size = max(6, min(8, int(red_cfg.get("consensus_core_size", 7) or 7)))
    support_size = max(4, min(7, int(red_cfg.get("probability_support_size", 6) or 6)))
    rescue_size = max(2, min(4, int(red_cfg.get("model_rescue_pool_size", 4) or 4)))
    explore_size = max(2, min(4, int(red_cfg.get("exploration_pool_size", 4) or 4)))
    wide_limit = max(18, min(int(red_cfg.get("wide_candidate_limit", 22) or 22), 33))
    must_limit = max(6, min(int(red_cfg.get("max_must_preserve_union", 14) or 14), wide_limit))
    should_limit = max(4, min(int(red_cfg.get("should_preserve_limit", 8) or 8), wide_limit))
    must_top_evidence = max(0, min(int(red_cfg.get("must_preserve_top_evidence", 10) or 10), wide_limit))
    must_min_method_support = max(1, int(red_cfg.get("must_preserve_min_method_support", 2) or 2))
    mandatory_protected_limit = max(
        0,
        min(int(red_cfg.get("red_wide_candidate_mandatory_protected_limit", 18) or 18), wide_limit),
    )
    ranked = sorted(
        evidence.values(),
        key=lambda item: (
            _safe_float(item.get("evidence")),
            int(item.get("method_support", 0)),
            int(item.get("appearance", 0)),
            -int(item.get("number", 0)),
        ),
        reverse=True,
    )
    consensus_core = [int(item["number"]) for item in ranked[:core_size]]
    probability_support = [int(item["number"]) for item in ranked[core_size: core_size + support_size]]
    used = set(consensus_core) | set(probability_support)
    mandatory_protected = [
        int(item["number"])
        for item in sorted(
            [item for item in ranked if bool(item.get("protected"))],
            key=lambda item: (
                -int(item.get("fusion_rank", 99) or 99),
                _safe_float(item.get("evidence")),
                int(item.get("method_support", 0)),
                int(item.get("appearance", 0)),
            ),
            reverse=True,
        )[:mandatory_protected_limit]
    ]
    rescue_ranked = sorted(
        [item for item in ranked if int(item["number"]) not in used],
        key=lambda item: (
            bool(item.get("protected")),
            int(item.get("method_support", 0)),
            _safe_float(item.get("evidence")),
            int(item.get("appearance", 0)),
            -int(item.get("number", 0)),
        ),
        reverse=True,
    )
    model_rescue = [int(item["number"]) for item in rescue_ranked[:rescue_size]]
    used |= set(model_rescue)
    explore_source = [
        item for item in ranked
        if int(item["number"]) not in used
    ]
    # Explore is not simply the next few high-score numbers: preserve lower
    # correlation directions by preferring numbers farther from the core centre.
    core_center = sum(consensus_core) / float(len(consensus_core)) if consensus_core else 17.0
    explore_ranked = sorted(
        explore_source,
        key=lambda item: (
            abs(float(item["number"]) - core_center),
            _safe_float(item.get("evidence")),
            int(item.get("appearance", 0)),
        ),
        reverse=True,
    )
    exploration = [int(item["number"]) for item in explore_ranked[:explore_size]]

    must_source = []
    for index, item in enumerate(ranked, start=1):
        method_support = int(item.get("method_support", 0) or 0)
        independent_support = int(item.get("independent_model_support_count", 0) or 0)
        protected = bool(item.get("protected"))
        if (
            protected
            or index <= must_top_evidence
            or method_support >= must_min_method_support
            or independent_support >= 1
        ):
            must_source.append(item)
    must_preserve = [
        int(item["number"])
        for item in sorted(
            must_source,
            key=lambda item: (
                bool(item.get("protected")),
                int(item.get("independent_model_support_count", 0) or 0),
                int(item.get("method_support", 0) or 0),
                _safe_float(item.get("evidence")),
                int(item.get("appearance", 0) or 0),
                -int(item.get("number", 0)),
            ),
            reverse=True,
        )[:must_limit]
    ]
    should_preserve = [
        int(item["number"])
        for item in ranked
        if int(item["number"]) not in set(must_preserve)
    ][:should_limit]
    candidate_set = list(dict.fromkeys(
        must_preserve
        + should_preserve
        + mandatory_protected
        + consensus_core
        + probability_support
        + model_rescue
        + exploration
    ))[:wide_limit]
    tier_by_number: dict[int, str] = {}
    for number in candidate_set:
        if number in must_preserve:
            tier_by_number[int(number)] = "MUST_PRESERVE"
        elif number in should_preserve:
            tier_by_number[int(number)] = "SHOULD_PRESERVE"
        else:
            tier_by_number[int(number)] = "EXPLORE"
    evidence_rows: list[dict[str, Any]] = []
    for number in candidate_set:
        if number not in evidence:
            continue
        row = dict(evidence[number])
        row["evidence_tier"] = tier_by_number.get(int(number), "EXPLORE")
        row["must_preserve"] = bool(int(number) in must_preserve)
        row["should_preserve"] = bool(int(number) in should_preserve)
        evidence_rows.append(row)
    return {
        "schema": "red_wide_candidate_channels_v8_5",
        "must_preserve": must_preserve,
        "should_preserve": should_preserve,
        "explore": [number for number in candidate_set if tier_by_number.get(int(number)) == "EXPLORE"],
        "consensus_core": consensus_core,
        "probability_support": probability_support,
        "model_rescue": model_rescue,
        "exploration": exploration,
        "mandatory_protected": mandatory_protected,
        "candidate_set": candidate_set,
        "evidence_rows": evidence_rows,
        "number_evidence_ledger": evidence_rows,
        "rule": "V8.5：Wide Candidate先拆成Must Preserve / Should Preserve / Explore；Must Preserve是Final Union硬曝光资源，旧Consensus/Support/Rescue仅作为来源标签。",
    }


def _tier_counts(red: set[int], layers: dict[str, Any]) -> dict[str, int]:
    must_preserve = set(int(value) for value in layers.get("must_preserve", []) or [])
    should_preserve = set(int(value) for value in layers.get("should_preserve", []) or [])
    consensus_core = set(int(value) for value in layers.get("consensus_core", []) or [])
    probability_support = set(int(value) for value in layers.get("probability_support", []) or [])
    model_rescue = set(int(value) for value in layers.get("model_rescue", []) or [])
    exploration = set(int(value) for value in layers.get("exploration", []) or [])
    return {
        "must_preserve": len(red & must_preserve),
        "should_preserve": len(red & should_preserve),
        "consensus_core": len(red & consensus_core),
        "probability_support": len(red & probability_support),
        "model_rescue": len(red & model_rescue),
        "exploration": len(red & exploration),
    }


def _ticket_layer_fit(ticket: dict[str, Any], layers: dict[str, Any]) -> float:
    counts = _tier_counts(_red_set(ticket), layers)
    core_ok = 1.0 if 2 <= counts["consensus_core"] <= 4 else max(0.0, 1.0 - abs(counts["consensus_core"] - 3.0) / 3.0)
    support_ok = 1.0 if 1 <= counts["probability_support"] <= 3 else max(0.0, 1.0 - abs(counts["probability_support"] - 2.0) / 3.0)
    rescue_ok = 1.0 if counts["model_rescue"] >= 1 else 0.82
    explore_ok = 1.0 if 0 <= counts["exploration"] <= 1 else max(0.0, 1.0 - abs(counts["exploration"] - 1.0) / 3.0)
    return float(0.40 * core_ok + 0.30 * support_ok + 0.18 * rescue_ok + 0.12 * explore_ok)


def _portfolio_layer_budget_fit(portfolio: list[dict[str, Any]], layers: dict[str, Any]) -> dict[str, Any]:
    """Score the four red wide-candidate channels as portfolio-level budgets."""

    ticket_count = max(1, len(portfolio))
    totals = {"consensus_core": 0, "probability_support": 0, "model_rescue": 0, "exploration": 0}
    for ticket in portfolio:
        counts = _tier_counts(_red_set(ticket), layers)
        for key in totals:
            totals[key] += int(counts[key])
    targets = {
        "consensus_core": (max(0, 2 * ticket_count - 2), 2 * ticket_count + 2),
        "probability_support": (max(0, 2 * ticket_count - 2), 2 * ticket_count + 2),
        "model_rescue": (max(1, ticket_count - 3), ticket_count + 1),
        "exploration": (max(0, ticket_count - 3), ticket_count + 1),
    }
    penalties = {}
    for key, (low, high) in targets.items():
        value = totals[key]
        if low <= value <= high:
            penalties[key] = 0.0
        else:
            penalties[key] = float(min(abs(value - low), abs(value - high)) / max(1, high))
    fit = 1.0 - min(1.0, sum(penalties.values()) / 3.0)
    return {
        "schema": "red_portfolio_channel_budget_v8",
        "totals": totals,
        "targets": targets,
        "penalties": penalties,
        "fit": float(fit),
        "rule": "Consensus Core / Probability Support / Model Rescue / Exploration按整个5注Portfolio分配预算，Model Rescue独立保留，不再混入旧Explore。",
    }


def _average_jaccard(portfolio: list[dict[str, Any]]) -> float:
    if len(portfolio) < 2:
        return 0.0
    values: list[float] = []
    for left, right in combinations(portfolio, 2):
        left_red = _red_set(left)
        right_red = _red_set(right)
        union = len(left_red | right_red)
        values.append(len(left_red & right_red) / float(union) if union else 0.0)
    return float(sum(values) / len(values)) if values else 0.0


def _cooccurrence_scores(
    pool_items: list[dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
) -> dict[tuple[int, int], float]:
    """Build a lightweight candidate co-occurrence graph for red tickets.

    This is deliberately local to the current training window/candidate pool:
    if several high-evidence candidates repeatedly place two numbers together,
    final selection should treat that pair as a stronger 6-ball-building signal
    than two isolated numbers that only improve union coverage.
    """

    pair_mass: dict[tuple[int, int], float] = {}
    max_quality = max((_quality(ticket) for ticket in pool_items), default=1.0) or 1.0
    for ticket in pool_items:
        red = sorted(_red_set(ticket))
        if len(red) < 2:
            continue
        ticket_quality = _quality(ticket) / float(max_quality)
        support = float((ticket.get("_v7_objectives") or {}).get("support", 0.0) or 0.0)
        protected = float((ticket.get("_v7_objectives") or {}).get("protected_recall", 0.0) or 0.0)
        ticket_weight = 0.60 + 0.45 * ticket_quality + 0.10 * support + 0.08 * min(3.0, protected)
        for left, right in combinations(red, 2):
            left_evidence = _safe_float(evidence.get(left, {}).get("evidence"))
            right_evidence = _safe_float(evidence.get(right, {}).get("evidence"))
            evidence_weight = 0.70 + 0.30 * ((left_evidence + right_evidence) / 2.0)
            pair = (int(left), int(right))
            pair_mass[pair] = pair_mass.get(pair, 0.0) + ticket_weight * evidence_weight
    maximum = max(pair_mass.values(), default=0.0)
    if maximum <= 0.0:
        return {}
    return {pair: float(value / maximum) for pair, value in pair_mass.items()}


def _ticket_cooccurrence(ticket: dict[str, Any], pair_scores: dict[tuple[int, int], float]) -> float:
    red = sorted(_red_set(ticket))
    pairs = [(int(left), int(right)) for left, right in combinations(red, 2)]
    if not pairs:
        return 0.0
    return float(sum(pair_scores.get(pair, 0.0) for pair in pairs) / len(pairs))


def _attach_cooccurrence(
    ticket: dict[str, Any],
    pair_scores: dict[tuple[int, int], float],
) -> dict[str, Any]:
    cooccurrence = _ticket_cooccurrence(ticket, pair_scores)
    objectives = dict(ticket.get("_v7_objectives") or {})
    objectives["cooccurrence"] = float(cooccurrence)
    objectives["cooccurrence_concentration"] = float(
        0.65 * cooccurrence
        + 0.20 * float(objectives.get("protected_recall", 0.0) or 0.0) / 6.0
        + 0.15 * float(objectives.get("method_vote", 0.0) or 0.0) / 5.0
    )
    return {
        **ticket,
        "cooccurrence_score": float(cooccurrence),
        "_v7_objectives": objectives,
    }


def _max_red_overlap(portfolio: list[dict[str, Any]]) -> int:
    if len(portfolio) < 2:
        return 0
    return max(
        len(_red_set(left) & _red_set(right))
        for left, right in combinations(portfolio, 2)
    )


def _information_metrics(
    portfolio: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    candidate_numbers = [int(value) for value in layers.get("candidate_set", []) or []]
    candidate_set = set(candidate_numbers)
    final_union = set().union(*[_red_set(ticket) for ticket in portfolio]) if portfolio else set()
    exposure_counter = {number: 0 for number in candidate_set}
    for ticket in portfolio:
        for number in _red_set(ticket) & candidate_set:
            exposure_counter[number] = exposure_counter.get(number, 0) + 1
    total_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in candidate_set)
    retained_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in final_union & candidate_set)
    evidence_retention = retained_evidence / total_evidence if total_evidence > 0 else 0.0
    candidate_recall_proxy = len(final_union & candidate_set) / float(len(candidate_set) or 1)
    total_survival_value = sum(
        _safe_float(evidence.get(number, {}).get("evidence")) * (1.0 + 0.18 * min(3, int(evidence.get(number, {}).get("method_support", 0) or 0)))
        for number in candidate_set
    )
    retained_survival_value = sum(
        _safe_float(evidence.get(number, {}).get("evidence"))
        * (1.0 + 0.18 * min(3, int(evidence.get(number, {}).get("method_support", 0) or 0)))
        * min(1.0, exposure_counter.get(number, 0) / 2.0)
        for number in candidate_set
    )
    survival_retention = retained_survival_value / total_survival_value if total_survival_value > 0 else 0.0
    retention = 0.48 * evidence_retention + 0.34 * candidate_recall_proxy + 0.18 * survival_retention
    dropped = sorted(candidate_set - final_union)
    return {
        "schema": "information_retention_v7",
        "candidate_number_count": int(len(candidate_set)),
        "final_union_count": int(len(final_union)),
        "candidate_evidence_total": float(total_evidence),
        "retained_evidence": float(retained_evidence),
        "evidence_retention": float(evidence_retention),
        "candidate_recall_proxy": float(candidate_recall_proxy),
        "survival_value_retention": float(survival_retention),
        "information_retention_rate": float(retention),
        "compression_regret": float(max(0.0, 1.0 - retention)),
        "exposure_counts": exposure_counter,
        "retained_candidate_numbers": sorted(final_union & candidate_set),
        "dropped_candidate_numbers": dropped,
        "dropped_evidence": [
            evidence[number]
            for number in dropped
            if number in evidence
        ],
    }


def _portfolio_score(
    portfolio: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    strategy: str,
    max_red_overlap: int,
) -> tuple[float, dict[str, Any]]:
    if not portfolio:
        return float("-inf"), {}
    metrics = _information_metrics(portfolio, layers=layers, evidence=evidence)
    union = set(metrics.get("retained_candidate_numbers", []) or [])
    must_preserve = set(int(value) for value in layers.get("must_preserve", []) or [])
    should_preserve = set(int(value) for value in layers.get("should_preserve", []) or [])
    consensus_core = set(int(value) for value in layers.get("consensus_core", []) or [])
    probability_support = set(int(value) for value in layers.get("probability_support", []) or [])
    model_rescue = set(int(value) for value in layers.get("model_rescue", []) or [])
    exploration = set(int(value) for value in layers.get("exploration", []) or [])
    must_coverage = len(union & must_preserve) / float(len(must_preserve) or 1)
    should_coverage = len(union & should_preserve) / float(len(should_preserve) or 1)
    must_missing_ratio = 1.0 - must_coverage if must_preserve else 0.0
    core_coverage = len(union & consensus_core) / float(len(consensus_core) or 1)
    support_coverage = len(union & probability_support) / float(len(probability_support) or 1)
    rescue_coverage = len(union & model_rescue) / float(len(model_rescue) or 1)
    explore_coverage = len(union & exploration) / float(len(exploration) or 1)
    layer_budget = _portfolio_layer_budget_fit(portfolio, layers)
    layer_fit = _safe_float(layer_budget.get("fit"))
    avg_jaccard = _average_jaccard(portfolio)
    max_overlap = _max_red_overlap(portfolio)
    overlap_penalty = max(0.0, float(max_overlap - max_red_overlap) / 6.0)
    duplicate_pressure = max(0.0, avg_jaccard - 0.36)
    method_diversity = len({
        method
        for ticket in portfolio
        for method in ticket.get("_v7_method_support", []) or []
    }) / 6.0
    method_diversity = min(1.0, method_diversity)
    quality = sum(_quality(ticket) for ticket in portfolio) / float(len(portfolio))
    quality_scale = max(1.0, max((_quality(ticket) for ticket in portfolio), default=1.0))
    quality = quality / quality_scale
    cooccurrence_quality = float(
        sum(float((ticket.get("_v7_objectives") or {}).get("cooccurrence", ticket.get("cooccurrence_score", 0.0)) or 0.0) for ticket in portfolio)
        / float(len(portfolio))
    )
    union_count = int(metrics.get("final_union_count", 0) or 0)
    target_union_min = 18
    target_union_max = 21
    low_jaccard_floor = 0.12
    wide_union_penalty = max(0.0, float(union_count - target_union_max) / 12.0)
    narrow_union_penalty = max(0.0, float(target_union_min - union_count) / 12.0)
    underlap_penalty = max(0.0, low_jaccard_floor - avg_jaccard) / low_jaccard_floor

    weights_by_strategy = {
        "information_preservation": (6.8, 1.2, 1.0, 0.7, 0.4, 1.8),
        "high_coverage": (5.6, 1.8, 1.3, 0.6, 0.2, 1.2),
        "weak_jaccard": (5.4, 1.1, 1.0, 0.7, 0.4, 0.7),
        "pure_probability": (3.8, 0.6, 0.4, 0.3, 1.4, 0.4),
        "cooccurrence_concentration": (4.8, 1.4, 1.1, 0.4, 0.8, 0.8),
    }
    w_retention, w_core, w_support, w_explore, w_quality, w_overlap = weights_by_strategy.get(
        strategy,
        weights_by_strategy["information_preservation"],
    )
    score = (
        w_retention * _safe_float(metrics.get("information_retention_rate"))
        + 3.40 * must_coverage
        + 1.20 * should_coverage
        + w_core * core_coverage
        + w_support * support_coverage
        + 1.10 * rescue_coverage
        + w_explore * explore_coverage
        + 1.15 * layer_fit
        + 0.55 * method_diversity
        + (1.55 if strategy == "cooccurrence_concentration" else 0.55) * cooccurrence_quality
        + w_quality * quality
        - w_overlap * overlap_penalty
        - 1.25 * duplicate_pressure
        - (2.30 if strategy == "cooccurrence_concentration" else 0.70) * wide_union_penalty
        - 0.80 * narrow_union_penalty
        - (1.45 if strategy == "cooccurrence_concentration" else 0.35) * underlap_penalty
        - 4.0 * _safe_float(metrics.get("compression_regret"))
        - 6.00 * must_missing_ratio
    )
    details = {
        **metrics,
        "score": float(score),
        "strategy": strategy,
        "must_preserve_coverage": float(must_coverage),
        "should_preserve_coverage": float(should_coverage),
        "must_preserve_missing_ratio": float(must_missing_ratio),
        "must_preserve_missing_numbers": sorted(must_preserve - union),
        "should_preserve_missing_numbers": sorted(should_preserve - union),
        "core_coverage": float(core_coverage),
        "support_coverage": float(support_coverage),
        "model_rescue_coverage": float(rescue_coverage),
        "explore_coverage": float(explore_coverage),
        "ticket_layer_fit": float(layer_fit),
        "portfolio_layer_budget": layer_budget,
        "average_jaccard": float(avg_jaccard),
        "max_red_overlap": int(max_overlap),
        "overlap_penalty": float(overlap_penalty),
        "wide_union_penalty": float(wide_union_penalty),
        "narrow_union_penalty": float(narrow_union_penalty),
        "underlap_penalty": float(underlap_penalty),
        "cooccurrence_quality": float(cooccurrence_quality),
        "target_union_range": [int(target_union_min), int(target_union_max)],
        "target_jaccard_floor": float(low_jaccard_floor),
        "method_diversity": float(method_diversity),
    }
    return float(score), details


def _final_union(portfolio: list[dict[str, Any]]) -> set[int]:
    return set().union(*[_red_set(ticket) for ticket in portfolio]) if portfolio else set()


def _candidate_tier(number: int, layers: dict[str, Any]) -> str:
    if int(number) in {int(value) for value in layers.get("must_preserve", []) or []}:
        return "MUST_PRESERVE"
    if int(number) in {int(value) for value in layers.get("should_preserve", []) or []}:
        return "SHOULD_PRESERVE"
    return "EXPLORE"


def _number_exposure_matrix(
    portfolio: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_numbers = [int(value) for value in layers.get("candidate_set", []) or []]
    union = _final_union(portfolio)
    for number in candidate_numbers:
        tickets = [
            int(index)
            for index, ticket in enumerate(portfolio, start=1)
            if int(number) in _red_set(ticket)
        ]
        row = dict(evidence.get(int(number), {"number": int(number)}))
        row.update({
            "number": int(number),
            "evidence_tier": _candidate_tier(int(number), layers),
            "final_union": bool(int(number) in union),
            "final_ticket_count": int(len(tickets)),
            "final_ticket_positions": tickets,
        })
        rows.append(row)
    return rows


def _evidence_concentration_gap(
    portfolio: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    union = _final_union(portfolio)
    final_union_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in union)
    ticket_rows = []
    for index, ticket in enumerate(portfolio, start=1):
        ticket_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in _red_set(ticket))
        ticket_rows.append({
            "ticket": int(index),
            "red": sorted(_red_set(ticket)),
            "evidence": float(ticket_evidence),
            "must_preserve_count": int(len(_red_set(ticket) & {int(value) for value in layers.get("must_preserve", []) or []})),
        })
    best = max(ticket_rows, key=lambda row: float(row.get("evidence", 0.0)), default={})
    best_ticket_evidence = float(best.get("evidence", 0.0) or 0.0)
    concentration_ratio = best_ticket_evidence / final_union_evidence if final_union_evidence > 0 else 0.0
    return {
        "schema": "union_best_ticket_gap_v8_5",
        "final_union_count": int(len(union)),
        "final_union": sorted(union),
        "final_union_evidence": float(final_union_evidence),
        "best_ticket": best,
        "best_ticket_evidence": float(best_ticket_evidence),
        "evidence_concentration_ratio": float(concentration_ratio),
        "evidence_concentration_gap": float(max(0.0, 1.0 - concentration_ratio)),
        "ticket_rows": ticket_rows,
        "rule": "Union大不等于同票转化好；该指标定位已发现号码是否被过度分散。",
    }


def _build_evidence_recovery_ticket(
    portfolio: list[dict[str, Any]],
    *,
    missing: set[int],
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    fallback_blue: int,
) -> dict[str, Any]:
    must = [int(value) for value in layers.get("must_preserve", []) or []]
    should = [int(value) for value in layers.get("should_preserve", []) or []]
    candidate_numbers = [int(value) for value in layers.get("candidate_set", []) or []]
    ordered = list(dict.fromkeys(
        sorted(missing, key=lambda number: _safe_float(evidence.get(number, {}).get("evidence")), reverse=True)
        + sorted(must, key=lambda number: _safe_float(evidence.get(number, {}).get("evidence")), reverse=True)
        + sorted(should, key=lambda number: _safe_float(evidence.get(number, {}).get("evidence")), reverse=True)
        + sorted(candidate_numbers, key=lambda number: _safe_float(evidence.get(number, {}).get("evidence")), reverse=True)
    ))
    red = [number for number in ordered if 1 <= int(number) <= 33][:6]
    if len(red) < 6:
        for number in range(1, 34):
            if number not in red:
                red.append(number)
            if len(red) >= 6:
                break
    return {
        "red": sorted(int(value) for value in red[:6]),
        "blue": int(fallback_blue) if 1 <= int(fallback_blue) <= 16 else 1,
        "score": sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in red[:6]),
        "selection_role": "evidence_recovery_ticket",
        "ticket_selection_role": "evidence_recovery_ticket",
        "selection_reason": {
            "schema": "evidence_recovery_ticket_v8_5",
            "missing_must_preserve_numbers": sorted(missing),
            "reason": "Must Preserve号码已进入候选但未获得Final Union曝光，生成一张恢复票专门承接号码级证据。",
        },
    }


def _ensure_evidence_preservation(
    portfolio: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    config: dict[str, Any],
    max_red_overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    if not bool(red_cfg.get("evidence_preservation_enabled", True)):
        return portfolio, {"active": False, "passed": True, "reason": "evidence_preservation_disabled"}
    must = {int(value) for value in layers.get("must_preserve", []) or [] if 1 <= int(value) <= 33}
    if not must or not portfolio:
        return portfolio, {
            "active": False,
            "passed": True,
            "reason": "no_must_preserve_numbers",
            "must_preserve_numbers": sorted(must),
        }
    required_ratio = min(1.0, max(0.0, float(red_cfg.get("final_must_preserve_required_ratio", 1.0) or 1.0)))
    max_missing_allowed = int(max(0, len(must) - round(len(must) * required_ratio)))
    result = [dict(ticket) for ticket in portfolio]
    before_union = _final_union(result)
    before_missing = sorted(must - before_union)
    events: list[dict[str, Any]] = []
    if len(before_missing) <= max_missing_allowed:
        return result, {
            "schema": "evidence_preservation_guard_v8_5",
            "active": True,
            "corrective": False,
            "must_preserve_numbers": sorted(must),
            "missing_before": before_missing,
            "missing_after": before_missing,
            "required_ratio": float(required_ratio),
            "passed": True,
            "reason": "must_preserve_final_union_already_satisfied",
        }

    selected_keys = {_ticket_key(ticket) for ticket in result}
    attempts = 0
    while len(must - _final_union(result)) > max_missing_allowed and attempts < max(1, len(must)):
        attempts += 1
        missing = must - _final_union(result)
        best_trial: tuple[float, int, dict[str, Any], dict[str, Any]] | None = None
        for position, current in enumerate(result):
            others = result[:position] + result[position + 1 :]
            other_keys = {_ticket_key(ticket) for ticket in others}
            for candidate in candidates:
                red = _red_set(candidate)
                if not red & missing:
                    continue
                replacement = {**candidate, "blue": int(current.get("blue", candidate.get("blue", 1)) or 1)}
                if _ticket_key(replacement) in other_keys:
                    continue
                trial = others + [replacement]
                _, metrics = _portfolio_score(
                    trial,
                    layers=layers,
                    evidence=evidence,
                    strategy="high_coverage",
                    max_red_overlap=min(6, max_red_overlap + 1),
                )
                after_missing = set(metrics.get("must_preserve_missing_numbers", []) or [])
                restored = len(missing) - len(after_missing)
                if restored <= 0:
                    continue
                removed_must = len(_red_set(current) & must)
                added_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in red & must)
                overlap = _max_red_overlap(trial)
                score = (
                    100.0 * restored
                    + 8.0 * _safe_float(metrics.get("information_retention_rate"))
                    + 2.5 * added_evidence
                    - 1.2 * max(0, overlap - max_red_overlap)
                    - 0.8 * removed_must
                    + 0.01 * _quality(candidate)
                )
                if best_trial is None or score > best_trial[0]:
                    best_trial = (float(score), int(position), replacement, metrics)
        if best_trial is None:
            break
        _, position, replacement, metrics = best_trial
        old_ticket = result[position]
        selected_keys.discard(_ticket_key(old_ticket))
        result[position] = {
            **replacement,
            "selection_role": "evidence_preservation_restore",
            "ticket_selection_role": "evidence_preservation_restore",
            "selection_reason": {
                **(replacement.get("selection_reason") if isinstance(replacement.get("selection_reason"), dict) else {}),
                "schema": "evidence_preservation_restore_v8_5",
                "restored_numbers": sorted(_red_set(replacement) & set(before_missing)),
                "removed_red": sorted(_red_set(old_ticket)),
                "reason": "Must Preserve号码不能只停留在候选层；最终组合层执行号码级恢复。",
            },
        }
        selected_keys.add(_ticket_key(result[position]))
        events.append({
            "type": "candidate_replacement",
            "position": int(position + 1),
            "from": sorted(_red_set(old_ticket)),
            "to": sorted(_red_set(result[position])),
            "restored_must_preserve_numbers": sorted(_red_set(result[position]) & set(before_missing)),
            "information_retention_rate_after": metrics.get("information_retention_rate"),
            "must_preserve_missing_after": metrics.get("must_preserve_missing_numbers", []),
        })

    after_missing = sorted(must - _final_union(result))
    if len(after_missing) > max_missing_allowed:
        weakest_index = min(
            range(len(result)),
            key=lambda index: (
                len(_red_set(result[index]) & must),
                sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in _red_set(result[index])),
                _quality(result[index]),
            ),
        )
        recovery = _build_evidence_recovery_ticket(
            result,
            missing=set(after_missing),
            layers=layers,
            evidence=evidence,
            fallback_blue=int(result[weakest_index].get("blue", 1) or 1),
        )
        old_ticket = result[weakest_index]
        result[weakest_index] = recovery
        events.append({
            "type": "evidence_recovery_ticket",
            "position": int(weakest_index + 1),
            "from": sorted(_red_set(old_ticket)),
            "to": sorted(_red_set(recovery)),
            "restored_must_preserve_numbers": sorted(_red_set(recovery) & set(after_missing)),
            "reason": "候选票替换仍无法满足Must Preserve曝光，启用Evidence Recovery Ticket。",
        })

    final_missing = sorted(must - _final_union(result))
    return result, {
        "schema": "evidence_preservation_guard_v8_5",
        "active": True,
        "corrective": bool(events),
        "must_preserve_numbers": sorted(must),
        "should_preserve_numbers": [int(value) for value in layers.get("should_preserve", []) or []],
        "missing_before": before_missing,
        "missing_after": final_missing,
        "required_ratio": float(required_ratio),
        "max_missing_allowed": int(max_missing_allowed),
        "events": events,
        "passed": len(final_missing) <= max_missing_allowed,
        "reason": "Protected/Rescue不再只保护进入Candidate；Must Preserve贯穿到Final Union。",
    }


def _number_survival_ledger(
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    portfolio_before_guard: list[dict[str, Any]],
    final_portfolio: list[dict[str, Any]],
    preservation_info: dict[str, Any],
) -> list[dict[str, Any]]:
    before_union = _final_union(portfolio_before_guard)
    final_union = _final_union(final_portfolio)
    replacement_numbers = sorted(final_union - before_union)
    rows: list[dict[str, Any]] = []
    for row in _number_exposure_matrix(final_portfolio, layers=layers, evidence=evidence):
        number = int(row["number"])
        before_present = number in before_union
        final_present = number in final_union
        drop_stage = None
        drop_reason = None
        if not final_present:
            tier = str(row.get("evidence_tier", "EXPLORE"))
            drop_stage = "final_union_compression"
            if tier == "MUST_PRESERVE":
                drop_reason = "must_preserve_capacity_or_constraint_conflict"
            elif tier == "SHOULD_PRESERVE":
                drop_reason = "lower_marginal_exposure_than_retained_must_should_numbers"
            else:
                drop_reason = "exploration_budget_compressed_after_evidence_preservation"
        rows.append({
            **row,
            "portfolio_before_guard": bool(before_present),
            "survived_final": bool(final_present),
            "drop_stage": drop_stage,
            "drop_reason": drop_reason,
            "drop_score": None if final_present else float(row.get("evidence", 0.0) or 0.0),
            "replacement_number": None if final_present or not replacement_numbers else int(replacement_numbers[0]),
            "counterfactual_gain": None if final_present else "保留该号需替换现有Final Union号码；请结合counterfactual_compression与exposure_matrix比较。",
            "preservation_guard_passed": bool(preservation_info.get("passed", True)),
        })
    return rows


def _beam_select_portfolio(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    strategy: str,
    max_red_overlap: int,
    beam_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    beams: list[list[dict[str, Any]]] = [[]]
    for _ in range(count):
        next_beams: list[tuple[float, list[dict[str, Any]], dict[str, Any]]] = []
        for beam in beams:
            seen = {_ticket_key(ticket) for ticket in beam}
            for candidate in candidates:
                if _ticket_key(candidate) in seen:
                    continue
                trial = beam + [candidate]
                score, details = _portfolio_score(
                    trial,
                    layers=layers,
                    evidence=evidence,
                    strategy=strategy,
                    max_red_overlap=max_red_overlap,
                )
                next_beams.append((score, trial, details))
        if not next_beams:
            break
        dedup: dict[tuple[tuple[tuple[int, ...], int], ...], tuple[float, list[dict[str, Any]], dict[str, Any]]] = {}
        for score, trial, details in sorted(next_beams, key=lambda item: item[0], reverse=True):
            key = tuple(sorted(_ticket_key(ticket) for ticket in trial))
            if key not in dedup:
                dedup[key] = (score, trial, details)
            if len(dedup) >= int(beam_width) * 4:
                break
        beams = [item[1] for item in sorted(dedup.values(), key=lambda item: item[0], reverse=True)[:beam_width]]
    completed = beams or [[]]
    scored = [
        (*_portfolio_score(
            beam,
            layers=layers,
            evidence=evidence,
            strategy=strategy,
            max_red_overlap=max_red_overlap,
        ), beam)
        for beam in completed
    ]
    best_score, best_details, best = max(scored, key=lambda item: item[0])
    return best[:count], {"score": float(best_score), **best_details}


def _counterfactual_compression(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    count: int,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    max_red_overlap: int,
    beam_width: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    strategies = [
        ("actual_information_preservation", selected),
        ("Strategy A2 - cooccurrence_concentration", None),
        ("Strategy B - weak_jaccard", None),
        ("Strategy C - high_coverage", None),
        ("Strategy D - pure_probability", None),
        ("Strategy E - no_final_compression", sorted(candidates, key=_quality, reverse=True)[:count]),
    ]
    for name, fixed in strategies:
        if fixed is None:
            strategy_key = (
                "cooccurrence_concentration" if "cooccurrence" in name
                else "weak_jaccard" if "weak_jaccard" in name
                else "high_coverage" if "high_coverage" in name
                else "pure_probability"
            )
            portfolio, metrics = _beam_select_portfolio(
                candidates,
                count=count,
                layers=layers,
                evidence=evidence,
                strategy=strategy_key,
                max_red_overlap=max_red_overlap + (1 if strategy_key in {"weak_jaccard", "cooccurrence_concentration"} else 0),
                beam_width=max(8, int(beam_width / 2)),
            )
        else:
            portfolio = fixed[:count]
            _, metrics = _portfolio_score(
                portfolio,
                layers=layers,
                evidence=evidence,
                strategy="information_preservation",
                max_red_overlap=max_red_overlap,
            )
        rows.append({
            "strategy": name,
            "score": metrics.get("score"),
            "information_retention_rate": metrics.get("information_retention_rate"),
            "compression_regret": metrics.get("compression_regret"),
            "average_jaccard": metrics.get("average_jaccard"),
            "max_red_overlap": metrics.get("max_red_overlap"),
            "final_union_count": metrics.get("final_union_count"),
            "dropped_candidate_numbers": metrics.get("dropped_candidate_numbers", []),
            "tickets": [
                {"red": sorted(_red_set(ticket)), "blue": int(ticket.get("blue", 0) or 0)}
                for ticket in portfolio
            ],
        })
    best = max(
        rows,
        key=lambda item: (
            _safe_float(item.get("information_retention_rate")),
            -_safe_float(item.get("compression_regret")),
            _safe_float(item.get("score")),
        ),
    ) if rows else {}
    return {
        "schema": "counterfactual_compression_v7",
        "rows": rows,
        "best_by_information_retention": best.get("strategy"),
        "reason": "同一期后台影子比较不同压缩规则，用于定位Final Compression Regret来自哪类约束。",
    }


def _ticket_objectives(
    ticket: dict[str, Any],
    *,
    method_support: list[str],
    red_support: dict[int, int],
    protected_numbers: set[int],
) -> dict[str, Any]:
    red = _red_set(ticket)
    scores = ticket.get("candidate_scores") if isinstance(ticket.get("candidate_scores"), dict) else {}
    probability = float(scores.get("red_number_score", _quality(ticket)) or 0.0)
    quality = float(scores.get("quality_score", _quality(ticket)) or 0.0)
    structure = float(scores.get("structure_score", 1.0) or 0.0)
    method_vote = float(len(set(method_support)))
    support_score = float(sum(red_support.get(number, 0) for number in red) / max(len(red), 1))
    protected = float(len(red & protected_numbers))
    spread = float((max(red) - min(red)) / 32.0) if red else 0.0
    objectives = {
        "probability": probability,
        "quality": quality,
        "structure": structure,
        "method_vote": method_vote,
        "support": support_score,
        "protected_recall": protected,
        "spread": spread,
    }
    return {
        **ticket,
        **objectives,
        "_v7_objectives": objectives,
        "_v7_method_support": sorted(set(method_support)),
    }


def _coverage_gain(ticket: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    red = _red_set(ticket)
    if not selected:
        return 1.0
    max_overlap = max((len(red & _red_set(item)) for item in selected), default=0)
    return float(1.0 - max_overlap / 6.0)


def _select_from_frontier(
    frontier: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    role: str,
    limit: int,
    max_red_overlap: int,
    allow_relaxed_overlap: bool = False,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen = {_ticket_key(item) for item in selected}
    if role == "coverage":
        key = lambda ticket: (
            _coverage_gain(ticket, selected + chosen),
            float((ticket.get("_v7_objectives") or {}).get("support", 0.0)),
            _quality(ticket),
        )
    elif role == "exploration":
        key = lambda ticket: (
            _coverage_gain(ticket, selected + chosen),
            float((ticket.get("_v7_objectives") or {}).get("spread", 0.0)),
            -float((ticket.get("_v7_objectives") or {}).get("method_vote", 0.0)),
            _quality(ticket),
        )
    else:
        key = lambda ticket: (
            float((ticket.get("_v7_objectives") or {}).get("protected_recall", 0.0)),
            float((ticket.get("_v7_objectives") or {}).get("probability", 0.0)),
            float((ticket.get("_v7_objectives") or {}).get("method_vote", 0.0)),
            _quality(ticket),
        )
    for ticket in sorted(frontier, key=key, reverse=True):
        if len(chosen) >= limit:
            break
        item_key = _ticket_key(ticket)
        if item_key in seen:
            continue
        red = _red_set(ticket)
        overlap = max((len(red & _red_set(item)) for item in selected + chosen), default=0)
        if overlap > int(max_red_overlap) and not allow_relaxed_overlap:
            continue
        chosen.append(ticket)
        seen.add(item_key)
    return chosen


def _extend_unique(target: list[dict[str, Any]], source: list[dict[str, Any]], *, role: str, limit: int) -> None:
    seen = {_ticket_key(item) for item in target}
    for ticket in source:
        key = _ticket_key(ticket)
        if key in seen:
            continue
        enriched = {
            **ticket,
            "selection_role": f"v7_red_portfolio_{role}",
            "ticket_selection_role": f"v7_red_portfolio_{role}",
            "selection_reason": {
                "schema": "red_portfolio_slot_v7",
                "role": role,
                "reason": "Evidence state controls probability / coverage / exploration budget.",
            },
        }
        target.append(enriched)
        seen.add(key)
        if len([item for item in target if str(item.get("selection_role", "")).endswith(role)]) >= limit:
            break


def _matching_candidates(
    tickets: list[dict[str, Any]],
    candidate_by_key: dict[tuple[tuple[int, ...], int], dict[str, Any]],
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for ticket in tickets or []:
        matched.append(dict(candidate_by_key.get(_ticket_key(ticket), ticket)))
    return matched


def _model_rescue_numbers(
    evidence: dict[int, dict[str, Any]],
    *,
    protected_numbers: set[int],
    limit: int = 4,
) -> list[int]:
    protected = [
        int(item.get("number", 0))
        for item in sorted(
            [evidence[int(number)] for number in protected_numbers if 1 <= int(number) <= 33 and int(number) in evidence],
            key=lambda item: (
                -int(item.get("fusion_rank", 99) or 99),
                _safe_float(item.get("evidence")),
                int(item.get("method_support", 0)),
                int(item.get("appearance", 0)),
            ),
            reverse=True,
        )
    ]
    ranked = sorted(
        evidence.values(),
        key=lambda item: (
            int(item.get("number", 0)) in protected_numbers,
            -int(item.get("fusion_rank", 99) or 99),
            int(item.get("method_support", 0)),
            _safe_float(item.get("evidence")),
            int(item.get("appearance", 0)),
            -int(item.get("number", 0)),
        ),
        reverse=True,
    )
    result: list[int] = []
    for number in protected:
        if number not in result:
            result.append(number)
    for item in ranked:
        number = int(item.get("number", 0))
        if number in result or not 1 <= number <= 33:
            continue
        if int(item.get("method_support", 0) or 0) < 2 and number not in protected_numbers:
            continue
        result.append(number)
        if len(result) >= int(limit):
            break
    return result[: int(limit)]


def _ensure_model_rescue_channel(
    portfolio: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    rescue_numbers: list[int],
    evidence: dict[int, dict[str, Any]],
    max_red_overlap: int,
    min_rescue_numbers_in_final: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rescue_set = {int(number) for number in rescue_numbers if 1 <= int(number) <= 33}
    if not rescue_set or not portfolio:
        return portfolio, {
            "active": False,
            "reason": "no_model_rescue_numbers",
            "rescue_numbers": sorted(rescue_set),
        }
    target_count = max(1, min(int(min_rescue_numbers_in_final), len(rescue_set), 6))
    result = [dict(item) for item in portfolio]
    final_union = {number for ticket in result for number in _red_set(ticket)}
    retained = sorted(final_union & rescue_set)
    if len(retained) >= target_count:
        return portfolio, {
            "active": True,
            "entered_final": True,
            "replaced": False,
            "rescue_numbers": sorted(rescue_set),
            "retained_rescue_numbers": retained,
            "missing_rescue_numbers": sorted(rescue_set - set(retained)),
            "min_rescue_numbers_in_final": int(target_count),
            "reason": "model_rescue_target_covered_by_final5",
        }
    selected_keys = {_ticket_key(ticket) for ticket in result}
    ordered_missing = sorted(
        rescue_set - set(retained),
        key=lambda number: (
            -int(evidence.get(number, {}).get("fusion_rank", 99) or 99),
            _safe_float(evidence.get(number, {}).get("evidence")),
            int(evidence.get(number, {}).get("method_support", 0) or 0),
            int(evidence.get(number, {}).get("appearance", 0) or 0),
        ),
        reverse=True,
    )
    wanted_rescue = int(ordered_missing[0]) if ordered_missing else int(sorted(rescue_set)[0])
    rescue_candidates = [
        ticket for ticket in candidates
        if _ticket_key(ticket) not in selected_keys and wanted_rescue in _red_set(ticket)
    ]
    if not rescue_candidates:
        return portfolio, {
            "active": True,
            "entered_final": bool(retained),
            "replaced": False,
            "rescue_numbers": sorted(rescue_set),
            "retained_rescue_numbers": retained,
            "missing_rescue_numbers": sorted(rescue_set - set(retained)),
            "target_rescue_number": int(wanted_rescue),
            "min_rescue_numbers_in_final": int(target_count),
            "reason": "no_candidate_ticket_contains_missing_model_rescue_number",
        }
    def rescue_score(ticket: dict[str, Any]) -> tuple[float, float, float, float, float]:
        red = _red_set(ticket)
        rescue_count = len(red & rescue_set)
        rescue_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in red & rescue_set)
        overlap = max((len(red & _red_set(item)) for item in result), default=0)
        return (
            float(wanted_rescue in red),
            float(rescue_count),
            float(rescue_evidence),
            -float(max(0, overlap - max_red_overlap)),
            _quality(ticket),
        )
    replacement = max(rescue_candidates, key=rescue_score)
    def removable_score(ticket: dict[str, Any]) -> tuple[float, float, float]:
        red = _red_set(ticket)
        retained_evidence = sum(_safe_float(evidence.get(number, {}).get("evidence")) for number in red)
        return (
            float(len(red & rescue_set)),
            float(retained_evidence),
            _quality(ticket),
        )
    replace_index = min(range(len(result)), key=lambda idx: removable_score(result[idx]))
    removed = result[replace_index]
    enriched = {
        **replacement,
        "selection_role": "model_rescue_final",
        "ticket_selection_role": "model_rescue_final",
        "selection_reason": {
            "schema": "model_rescue_final_channel_v8",
            "target_rescue_number": int(wanted_rescue),
            "rescue_numbers": sorted(_red_set(replacement) & rescue_set),
            "all_rescue_numbers": sorted(rescue_set),
            "removed_ticket_red": sorted(_red_set(removed)),
            "reason": "Model Rescue必须至少有一条通道进入Final 5，避免已发现的高支持号码只停留在胆拖或归因层。",
        },
    }
    result[replace_index] = enriched
    final_union = {number for ticket in result for number in _red_set(ticket)}
    retained = sorted(final_union & rescue_set)
    replacements = [{
        "target_rescue_number": int(wanted_rescue),
        "replace_index": int(replace_index + 1),
        "replacement_red": sorted(_red_set(replacement)),
        "removed_red": sorted(_red_set(removed)),
    }]
    if len(retained) < target_count:
        recursive_result, recursive_info = _ensure_model_rescue_channel(
            result,
            candidates,
            rescue_numbers=rescue_numbers,
            evidence=evidence,
            max_red_overlap=max_red_overlap,
            min_rescue_numbers_in_final=target_count,
        )
        if recursive_info.get("replaced"):
            return recursive_result, {
                **recursive_info,
                "replacements": replacements + list(recursive_info.get("replacements", []) or []),
            }
    return result, {
        "active": True,
        "entered_final": len(retained) >= target_count,
        "replaced": True,
        "replace_index": int(replace_index + 1),
        "replacements": replacements,
        "rescue_numbers": sorted(rescue_set),
        "retained_rescue_numbers": retained,
        "missing_rescue_numbers": sorted(rescue_set - set(retained)),
        "min_rescue_numbers_in_final": int(target_count),
        "replacement_red": sorted(_red_set(replacement)),
        "removed_red": sorted(_red_set(removed)),
        "reason": "model_rescue_missing_number_replaced_lowest_contribution_final_ticket",
    }


def _number_survival_values(
    evidence: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Score candidate numbers by their chance to deserve a finite-ticket slot.

    This is the first V9-style conversion layer: it deliberately evaluates
    number-level evidence before a number is hidden inside a weak full ticket.
    """
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    weights = red_cfg.get("number_survival_value_weights") or {}
    w_evidence = float(weights.get("evidence", 0.25) or 0.25)
    w_probability = float(weights.get("probability_mass", 0.16) or 0.16)
    w_consensus = float(weights.get("method_consensus", 0.14) or 0.14)
    w_independent = float(weights.get("independent_model_support", 0.18) or 0.18)
    w_appearance = float(weights.get("candidate_appearance", 0.13) or 0.13)
    w_recovery = float(weights.get("uncertainty_recovery", 0.09) or 0.09)
    w_protected = float(weights.get("protected", 0.05) or 0.05)
    max_evidence = max((_safe_float(row.get("evidence")) for row in evidence.values()), default=1.0) or 1.0
    max_probability = max((_safe_float(row.get("probability_mass")) for row in evidence.values()), default=1.0) or 1.0
    max_method_support = max((int(row.get("method_support", 0) or 0) for row in evidence.values()), default=1) or 1
    max_independent = max((int(row.get("independent_model_support_count", 0) or 0) for row in evidence.values()), default=1) or 1
    max_appearance = max((int(row.get("appearance", 0) or 0) for row in evidence.values()), default=1) or 1
    values: dict[int, dict[str, Any]] = {}
    for number, row in evidence.items():
        fusion_rank = int(row.get("fusion_rank", 99) or 99)
        method_support = int(row.get("method_support", 0) or 0)
        appearance = int(row.get("appearance", 0) or 0)
        independent_support = int(row.get("independent_model_support_count", 0) or 0)
        recovery_gap = 1.0 if appearance > 0 and method_support <= int(red_cfg.get("ticket_conversion_max_method_support", 0) or 0) else 0.0
        rank_value = (34.0 - float(min(33, max(1, fusion_rank)))) / 33.0 if fusion_rank < 99 else 0.0
        score = (
            w_evidence * (_safe_float(row.get("evidence")) / max_evidence)
            + w_probability * (_safe_float(row.get("probability_mass")) / max_probability)
            + w_consensus * (float(method_support) / float(max_method_support))
            + w_independent * (float(independent_support) / float(max_independent))
            + w_appearance * (float(appearance) / float(max_appearance))
            + w_recovery * max(recovery_gap, rank_value * 0.5)
            + w_protected * (1.0 if bool(row.get("protected")) else 0.0)
        )
        values[int(number)] = {
            "number": int(number),
            "number_survival_value": float(score),
            "evidence": _safe_float(row.get("evidence")),
            "appearance": int(appearance),
            "method_support": int(method_support),
            "independent_model_support_count": int(independent_support),
            "fusion_rank": None if fusion_rank >= 99 else int(fusion_rank),
            "protected": bool(row.get("protected")),
            "recovery_gap": bool(recovery_gap),
        }
    return values


def _candidate_to_method_recovery_numbers(
    evidence: dict[int, dict[str, Any]],
    config: dict[str, Any],
    final_union: set[int],
) -> list[dict[str, Any]]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    if not bool(red_cfg.get("ticket_conversion_enabled", True)):
        return []
    values = _number_survival_values(evidence, config)
    limit = max(0, int(red_cfg.get("ticket_conversion_recovery_number_limit", 4) or 4))
    min_appearance = max(0, int(red_cfg.get("ticket_conversion_min_appearance", 3) or 3))
    max_method_support = max(0, int(red_cfg.get("ticket_conversion_max_method_support", 0) or 0))
    min_value = float(red_cfg.get("ticket_conversion_min_survival_value", 0.26) or 0.26)
    rows: list[dict[str, Any]] = []
    for number, row in values.items():
        source = evidence.get(number, {})
        if int(number) in final_union:
            continue
        if int(row.get("appearance", 0) or 0) < min_appearance:
            continue
        if int(row.get("method_support", 0) or 0) > max_method_support:
            continue
        independent_support = int(row.get("independent_model_support_count", 0) or 0)
        fusion_rank = row.get("fusion_rank")
        protected = bool(row.get("protected"))
        has_independent_or_fusion = bool(
            protected
            or independent_support > 0
            or (fusion_rank is not None and int(fusion_rank) <= int(red_cfg.get("ticket_conversion_fusion_rank_cutoff", 22) or 22))
        )
        if not has_independent_or_fusion:
            continue
        if _safe_float(row.get("number_survival_value")) < min_value and not protected:
            continue
        rows.append({
            **row,
            "source": {
                "quality_mass": source.get("quality_mass"),
                "probability_mass": source.get("probability_mass"),
                "protection_sources": source.get("protection_sources", []),
            },
            "reason": "candidate_number_has_evidence_but_no_optimizer_method_support",
        })
    return sorted(
        rows,
        key=lambda item: (
            _safe_float(item.get("number_survival_value")),
            int(item.get("independent_model_support_count", 0) or 0),
            int(item.get("appearance", 0) or 0),
            -int(item.get("fusion_rank") or 99),
        ),
        reverse=True,
    )[:limit]


def _ticket_number_survival_value(ticket: dict[str, Any], values: dict[int, dict[str, Any]]) -> float:
    return float(sum(_safe_float(values.get(number, {}).get("number_survival_value")) for number in _red_set(ticket)))


def _ensure_candidate_to_method_recovery_ticket(
    portfolio: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    config: dict[str, Any],
    max_red_overlap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_union = _final_union(portfolio)
    recovery_numbers = _candidate_to_method_recovery_numbers(evidence, config, final_union)
    if not recovery_numbers or not portfolio:
        return portfolio, {
            "schema": "candidate_to_method_ticket_conversion_v9",
            "active": False,
            "reason": "no_candidate_method_gap_numbers",
            "recovery_numbers": recovery_numbers,
        }
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    max_ticket_changes = max(0, int(red_cfg.get("ticket_conversion_max_ticket_changes", 1) or 1))
    if max_ticket_changes <= 0:
        return portfolio, {
            "schema": "candidate_to_method_ticket_conversion_v9",
            "active": True,
            "replaced": False,
            "reason": "ticket_conversion_change_budget_zero",
            "recovery_numbers": recovery_numbers,
        }
    max_retention_loss = float(red_cfg.get("ticket_conversion_max_retention_loss", 0.025) or 0.025)
    max_score_loss = float(red_cfg.get("ticket_conversion_max_score_loss", 0.20) or 0.20)
    allowed_overlap = min(6, int(max_red_overlap) + int(red_cfg.get("ticket_conversion_overlap_relaxation", 1) or 1))
    survival_values = _number_survival_values(evidence, config)
    target_numbers = {int(row["number"]) for row in recovery_numbers}
    before_score, before_metrics = _portfolio_score(
        portfolio,
        layers=layers,
        evidence=evidence,
        strategy="cooccurrence_concentration",
        max_red_overlap=max_red_overlap,
    )
    selected_keys = {_ticket_key(ticket) for ticket in portfolio}
    candidate_pool = [
        ticket for ticket in candidates
        if _ticket_key(ticket) not in selected_keys and bool(_red_set(ticket) & target_numbers)
    ]
    if not candidate_pool:
        return portfolio, {
            "schema": "candidate_to_method_ticket_conversion_v9",
            "active": True,
            "replaced": False,
            "reason": "no_ticket_contains_candidate_method_gap_number",
            "recovery_numbers": recovery_numbers,
        }
    best_trial: tuple[float, int, dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for position, current in enumerate(portfolio):
        others = portfolio[:position] + portfolio[position + 1 :]
        current_survival = _ticket_number_survival_value(current, survival_values)
        for candidate in candidate_pool:
            candidate_red = _red_set(candidate)
            overlap = max((len(candidate_red & _red_set(other)) for other in others), default=0)
            if overlap > allowed_overlap:
                continue
            trial = others + [candidate]
            trial_score, trial_metrics = _portfolio_score(
                trial,
                layers=layers,
                evidence=evidence,
                strategy="cooccurrence_concentration",
                max_red_overlap=max_red_overlap,
            )
            retention_loss = _safe_float(before_metrics.get("information_retention_rate")) - _safe_float(trial_metrics.get("information_retention_rate"))
            score_loss = float(before_score) - float(trial_score)
            if retention_loss > max_retention_loss:
                continue
            if score_loss > max_score_loss:
                continue
            added = sorted(_red_set(candidate) & target_numbers)
            if not added:
                continue
            survival_gain = _ticket_number_survival_value(candidate, survival_values) - current_survival
            concentration_gain = _safe_float(trial_metrics.get("cooccurrence_quality")) - _safe_float(before_metrics.get("cooccurrence_quality"))
            rank_score = (
                2.8 * sum(_safe_float(survival_values.get(number, {}).get("number_survival_value")) for number in added)
                + 0.9 * max(0.0, survival_gain)
                + 0.5 * max(0.0, concentration_gain)
                + 0.02 * _quality(candidate)
                - 0.35 * max(0, overlap - max_red_overlap)
                - 1.2 * max(0.0, retention_loss)
                - 0.35 * max(0.0, score_loss)
            )
            if best_trial is None or rank_score > best_trial[0]:
                best_trial = (float(rank_score), position, candidate, trial_metrics, {"overlap": overlap, "added": added, "retention_loss": retention_loss, "score_loss": score_loss})
    if best_trial is None:
        return portfolio, {
            "schema": "candidate_to_method_ticket_conversion_v9",
            "active": True,
            "replaced": False,
            "reason": "candidate_method_gap_numbers_failed_retention_or_overlap_gate",
            "recovery_numbers": recovery_numbers,
            "before_metrics": before_metrics,
        }
    _, position, replacement, trial_metrics, trial_audit = best_trial
    removed = portfolio[position]
    result = [dict(ticket) for ticket in portfolio]
    result[position] = {
        **replacement,
        "selection_role": "candidate_to_method_recovery_ticket",
        "ticket_selection_role": "candidate_to_method_recovery_ticket",
        "selection_reason": {
            "schema": "candidate_to_method_ticket_conversion_v9",
            "added_gap_numbers": trial_audit["added"],
            "removed_red": sorted(_red_set(removed)),
            "replacement_red": sorted(_red_set(replacement)),
            "reason": "Candidate中有号码证据但现有greedy/pareto/global_coverage等方法层未给席位；本票作为Recovery Ticket通过保留率/重叠/分数闸门后进入Final 5。",
        },
    }
    after_union = _final_union(result)
    return result, {
        "schema": "candidate_to_method_ticket_conversion_v9",
        "active": True,
        "replaced": True,
        "replace_index": int(position + 1),
        "max_ticket_changes": int(max_ticket_changes),
        "recovery_numbers": recovery_numbers,
        "added_gap_numbers": trial_audit["added"],
        "removed_red": sorted(_red_set(removed)),
        "replacement_red": sorted(_red_set(replacement)),
        "retained_recovery_numbers": sorted(after_union & target_numbers),
        "missing_recovery_numbers": sorted(target_numbers - after_union),
        "before_metrics": before_metrics,
        "after_metrics": trial_metrics,
        "audit": trial_audit,
        "reason": "evidence_recovery_optimizer_converted_candidate_number_without_method_support_into_one_final_ticket",
    }


def _coverage_backbone_portfolio(
    candidates: list[dict[str, Any]],
    selected_by_method: dict[str, list[dict[str, Any]]],
    candidate_by_key: dict[tuple[tuple[int, ...], int], dict[str, Any]],
    *,
    count: int,
    layers: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    max_red_overlap: int,
    beam_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backbone = _matching_candidates(
        list((selected_by_method or {}).get("global_coverage") or [])[:count],
        candidate_by_key,
    )
    if len(backbone) < count:
        fallback, _ = _beam_select_portfolio(
            candidates,
            count=count,
            layers=layers,
            evidence=evidence,
            strategy="high_coverage",
            max_red_overlap=max_red_overlap,
            beam_width=max(8, int(beam_width / 2)),
        )
        _extend_unique(backbone, fallback, role="coverage_backbone_fill", limit=count - len(backbone))
    _, metrics = _portfolio_score(
        backbone[:count],
        layers=layers,
        evidence=evidence,
        strategy="high_coverage",
        max_red_overlap=max_red_overlap,
    )
    return backbone[:count], {
        **metrics,
        "schema": "coverage_backbone_v7_1",
        "method": "global_coverage",
        "rule": "第一阶段以global_coverage形成Coverage Backbone，优先保护Candidate→Final信息保留。",
    }


def _controlled_concentration_portfolio(
    candidates: list[dict[str, Any]],
    backbone: list[dict[str, Any]],
    *,
    count: int,
    slots: dict[str, int],
    max_red_overlap: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    probability_slots = max(0, int(slots.get("probability", 2) or 0))
    coverage_slots = max(0, int(slots.get("coverage", 2) or 0))
    exploration_slots = max(0, int(slots.get("exploration", 1) or 0))
    probability_source = sorted(
        candidates,
        key=lambda ticket: (
            float((ticket.get("_v7_objectives") or {}).get("probability", 0.0)),
            float((ticket.get("_v7_objectives") or {}).get("method_vote", 0.0)),
            _quality(ticket),
        ),
        reverse=True,
    )
    _extend_unique(
        result,
        _select_from_frontier(
            probability_source,
            selected=result,
            role="probability",
            limit=probability_slots,
            max_red_overlap=max_red_overlap,
        ),
        role="controlled_probability",
        limit=probability_slots,
    )
    _extend_unique(
        result,
        _select_from_frontier(
            candidates,
            selected=result,
            role="coverage",
            limit=coverage_slots,
            max_red_overlap=max_red_overlap,
        ),
        role="controlled_coverage",
        limit=coverage_slots,
    )
    _extend_unique(
        result,
        _select_from_frontier(
            candidates,
            selected=result,
            role="exploration",
            limit=exploration_slots,
            max_red_overlap=min(6, max_red_overlap + 1),
            allow_relaxed_overlap=True,
        ),
        role="controlled_exploration",
        limit=exploration_slots,
    )
    _extend_unique(result, backbone, role="controlled_backbone_fill", limit=count - len(result))
    _extend_unique(
        result,
        _select_from_frontier(
            candidates,
            selected=result,
            role="coverage",
            limit=count - len(result),
            max_red_overlap=min(6, max_red_overlap + 1),
            allow_relaxed_overlap=True,
        ),
        role="controlled_final_fill",
        limit=count - len(result),
    )
    return result[:count]


def _retention_guard_decision(
    backbone_metrics: dict[str, Any],
    controlled_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    red_cfg = (v7.get("red") or {}) if isinstance(v7, dict) else {}
    floor = float(red_cfg.get("retention_guard_minimum", 0.80) or 0.80)
    max_backbone_loss = float(red_cfg.get("retention_guard_max_backbone_loss", 0.03) or 0.03)
    concentration_floor = float(red_cfg.get("concentration_retention_minimum", max(0.80, floor + 0.04)) or max(0.80, floor + 0.04))
    target_union_max = int(red_cfg.get("cooccurrence_target_union_max", 21) or 21)
    target_jaccard_floor = float(red_cfg.get("cooccurrence_target_jaccard_floor", 0.12) or 0.12)
    backbone_retention = _safe_float(backbone_metrics.get("information_retention_rate"))
    controlled_retention = _safe_float(controlled_metrics.get("information_retention_rate"))
    backbone_regret = _safe_float(backbone_metrics.get("compression_regret"))
    controlled_regret = _safe_float(controlled_metrics.get("compression_regret"))
    backbone_union = int(backbone_metrics.get("final_union_count", 0) or 0)
    controlled_union = int(controlled_metrics.get("final_union_count", 0) or 0)
    backbone_jaccard = _safe_float(backbone_metrics.get("average_jaccard"))
    controlled_jaccard = _safe_float(controlled_metrics.get("average_jaccard"))
    backbone_cooccurrence = _safe_float(backbone_metrics.get("cooccurrence_quality"))
    controlled_cooccurrence = _safe_float(controlled_metrics.get("cooccurrence_quality"))
    backbone_too_wide = bool(backbone_union > target_union_max or backbone_jaccard < target_jaccard_floor)
    concentration_improves_shape = bool(
        controlled_union <= max(target_union_max, backbone_union - 1)
        or controlled_jaccard >= backbone_jaccard + 0.015
        or controlled_cooccurrence >= backbone_cooccurrence + 0.025
    )
    controlled_passes = bool(
        controlled_retention >= floor
        and controlled_retention + max_backbone_loss >= backbone_retention
        and controlled_regret <= backbone_regret + max_backbone_loss
    )
    concentration_passes = bool(
        controlled_retention >= concentration_floor
        and backbone_too_wide
        and concentration_improves_shape
        and controlled_regret <= max(1.0, backbone_regret + max_backbone_loss * 3.0)
    )
    prefer_controlled = bool(
        (
            controlled_passes
            and _safe_float(controlled_metrics.get("score")) >= _safe_float(backbone_metrics.get("score"))
        )
        or concentration_passes
    )
    return {
        "schema": "red_retention_guard_v8",
        "floor": floor,
        "max_backbone_loss": max_backbone_loss,
        "concentration_retention_minimum": concentration_floor,
        "target_union_max": int(target_union_max),
        "target_jaccard_floor": float(target_jaccard_floor),
        "backbone_information_retention_rate": backbone_retention,
        "controlled_information_retention_rate": controlled_retention,
        "backbone_compression_regret": backbone_regret,
        "controlled_compression_regret": controlled_regret,
        "backbone_final_union_count": int(backbone_union),
        "controlled_final_union_count": int(controlled_union),
        "backbone_average_jaccard": float(backbone_jaccard),
        "controlled_average_jaccard": float(controlled_jaccard),
        "backbone_cooccurrence_quality": float(backbone_cooccurrence),
        "controlled_cooccurrence_quality": float(controlled_cooccurrence),
        "backbone_too_wide_or_underlapped": backbone_too_wide,
        "concentration_improves_shape": concentration_improves_shape,
        "controlled_passes": controlled_passes,
        "concentration_passes": concentration_passes,
        "selected": "controlled_concentration" if prefer_controlled else "coverage_backbone",
        "reason": (
            "受控集中/共现集中在保留率达标后改善了过宽并集或过低重叠，允许替换Backbone。"
            if prefer_controlled
            else "Retention Guard保留Coverage Backbone：当前集中方案未同时满足保留率和共现形态改善。"
        ),
    }


def select_red_portfolio(
    pool: list[dict[str, Any]],
    selected_by_method: dict[str, list[dict[str, Any]]],
    count: int,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build evidence-first red portfolio as an information-compression problem."""
    count = max(1, int(count))
    v7 = (config or {}).get("v7", {}) if isinstance(config, dict) else {}
    runtime = (v7.get("red_runtime_control") or {}) if isinstance(v7, dict) else {}
    forced_budget = (v7.get("red_budget") or {}) if isinstance(v7, dict) else {}
    runtime_strategy = str(runtime.get("portfolio_strategy", "information_preservation"))
    max_red_overlap = int(runtime.get("red_jaccard_max_overlap", 4) or 4)
    if bool(v7.get("enabled", False)) and bool((v7.get("portfolio") or {}).get("dynamic_budget", True)):
        edge_state = str(v7.get("current_red_edge_state") or "NO_EDGE")
        budget = forced_budget if isinstance(forced_budget, dict) and forced_budget.get("slots") else allocate_portfolio_budget(edge_state, count, config or {})
        probability_slots = min(count, int(budget["slots"].get("probability", 0)))
        coverage_slots = min(max(0, count - probability_slots), int(budget["slots"].get("coverage", 0)))
        exploration_slots = max(0, count - probability_slots - coverage_slots)
    else:
        budget = {
            "schema": "portfolio_budget_v7_static_fallback",
            "edge_state": "WEAK_EDGE",
            "slots": {"probability": 2, "coverage": 2, "exploration": 1},
        }
        probability_slots = min(count, 2)
        coverage_slots = min(max(0, count - probability_slots), 2)
        exploration_slots = max(0, count - probability_slots - coverage_slots)
    support_by_ticket = _method_support_map(selected_by_method)
    support_by_red = _method_red_support(selected_by_method)
    protected_numbers = {
        int(value)
        for value in (v7.get("red_protected_numbers") or [])
        if 1 <= int(value) <= 33
    }
    pool_items = [
        _ticket_objectives(
            dict(ticket),
            method_support=support_by_ticket.get(_ticket_key(ticket), []),
            red_support=support_by_red,
            protected_numbers=protected_numbers,
        )
        for ticket in pool
    ]
    red_evidence = _pool_red_evidence(
        pool_items,
        red_support=support_by_red,
        protected_numbers=protected_numbers,
    )
    recall_protection = v7.get("red_candidate_recall_protection") or {}
    protection_details = {
        int(item.get("number")): item
        for item in recall_protection.get("protected_details", []) or []
        if isinstance(item, dict)
        and str(item.get("number", "")).isdigit()
        and 1 <= int(item.get("number")) <= 33
    }
    for number in protected_numbers:
        if not 1 <= int(number) <= 33:
            continue
        detail = protection_details.get(int(number), {})
        row = red_evidence.setdefault(
            int(number),
            {
                "number": int(number),
                "evidence": 0.0,
                "appearance": 0,
                "method_support": int(support_by_red.get(int(number), 0) or 0),
                "quality_mass": 0.0,
                "probability_mass": 0.0,
                "protected": True,
            },
        )
        sources = list(detail.get("protection_sources", []) or [])
        row["protected"] = True
        row["protection_sources"] = sources or ["runtime_guard"]
        if detail.get("fusion_rank") is not None:
            row["fusion_rank"] = int(detail.get("fusion_rank") or 99)
        if detail.get("fusion_probability") is not None:
            row["fusion_probability"] = float(detail.get("fusion_probability") or 0.0)
        if detail.get("model_support_count") is not None:
            row["independent_model_support_count"] = int(detail.get("model_support_count") or 0)
        row["evidence"] = float(row.get("evidence", 0.0) or 0.0) + (
            0.12 if "fusion_topn" in sources else 0.0
        ) + (
            0.08 if "independent_model_top6_top10" in sources else 0.0
        )
    pair_scores = _cooccurrence_scores(pool_items, red_evidence)
    pool_items = [_attach_cooccurrence(ticket, pair_scores) for ticket in pool_items]
    red_layers = _build_red_layers(red_evidence, config or {})
    frontier = pareto_frontier(
        pool_items,
        ["probability", "quality", "structure", "method_vote", "support", "protected_recall", "spread", "cooccurrence"],
    )
    if not frontier:
        frontier = pool_items
    red_cfg = ((v7.get("red") or {}) if isinstance(v7, dict) else {})
    candidate_limit = int(red_cfg.get("portfolio_candidate_limit", 260) or 260)
    beam_width = int(red_cfg.get("portfolio_beam_width", 32) or 32)
    method_source = [
        _attach_cooccurrence(
            _ticket_objectives(
                dict(ticket),
                method_support=support_by_ticket.get(_ticket_key(ticket), [method]),
                red_support=support_by_red,
                protected_numbers=protected_numbers,
            ),
            pair_scores,
        )
        for method, tickets in selected_by_method.items()
        for ticket in tickets
    ]
    candidate_source = frontier + method_source + list(pool_items)
    dedup_candidates: dict[tuple[tuple[int, ...], int], dict[str, Any]] = {}
    for ticket in sorted(
        candidate_source,
        key=lambda item: (
            float((item.get("_v7_objectives") or {}).get("cooccurrence_concentration", 0.0)),
            sum(_safe_float(red_evidence.get(number, {}).get("evidence")) for number in _red_set(item)),
            _ticket_layer_fit(item, red_layers),
            _quality(item),
        ),
        reverse=True,
    ):
        dedup_candidates.setdefault(_ticket_key(ticket), ticket)
        if len(dedup_candidates) >= max(count, candidate_limit):
            break
    candidates = list(dedup_candidates.values())
    candidate_by_key = {key: value for key, value in dedup_candidates.items()}
    rescue_numbers = _model_rescue_numbers(
        red_evidence,
        protected_numbers=protected_numbers,
        limit=int(red_cfg.get("model_rescue_number_count", 4) or 4),
    )
    selected_strategy = "information_preservation"
    if "probability" in runtime_strategy:
        selected_strategy = "pure_probability"
    elif "coverage" in runtime_strategy or "recall_guard" in runtime_strategy:
        selected_strategy = "high_coverage"
    elif "jaccard" in runtime_strategy:
        selected_strategy = "weak_jaccard"
    production_strategy = str(
        red_cfg.get("production_portfolio_strategy", red_cfg.get("portfolio_objective", "coverage_backbone_controlled_concentration"))
    )
    controlled_slots = dict(red_cfg.get("controlled_concentration_slots") or {})
    if not controlled_slots:
        controlled_slots = {"probability": 2, "coverage": 2, "exploration": 1}
    backbone, backbone_metrics = _coverage_backbone_portfolio(
        candidates,
        selected_by_method,
        candidate_by_key,
        count=count,
        layers=red_layers,
        evidence=red_evidence,
        max_red_overlap=max_red_overlap,
        beam_width=max(6, beam_width),
    )
    controlled = _controlled_concentration_portfolio(
        candidates,
        backbone,
        count=count,
        slots=controlled_slots,
        max_red_overlap=max_red_overlap,
    )
    cooccurrence_controlled, cooccurrence_metrics = _beam_select_portfolio(
        candidates,
        count=count,
        layers=red_layers,
        evidence=red_evidence,
        strategy="cooccurrence_concentration",
        max_red_overlap=min(6, max_red_overlap + 1),
        beam_width=max(8, int(beam_width / 2)),
    )
    controlled_score, controlled_metrics = _portfolio_score(
        controlled,
        layers=red_layers,
        evidence=red_evidence,
        strategy=selected_strategy,
        max_red_overlap=max_red_overlap,
    )
    slot_controlled_metrics = dict(controlled_metrics)
    cooccurrence_score = _safe_float(cooccurrence_metrics.get("score"))
    controlled_rank_score = (
        _safe_float(slot_controlled_metrics.get("information_retention_rate"))
        + 0.55 * _safe_float(slot_controlled_metrics.get("cooccurrence_quality"))
        - 0.35 * _safe_float(slot_controlled_metrics.get("wide_union_penalty"))
        - 0.25 * _safe_float(slot_controlled_metrics.get("underlap_penalty"))
    )
    cooccurrence_rank_score = (
        _safe_float(cooccurrence_metrics.get("information_retention_rate"))
        + 0.55 * _safe_float(cooccurrence_metrics.get("cooccurrence_quality"))
        - 0.35 * _safe_float(cooccurrence_metrics.get("wide_union_penalty"))
        - 0.25 * _safe_float(cooccurrence_metrics.get("underlap_penalty"))
    )
    concentration_variant = "cooccurrence_concentration" if cooccurrence_rank_score > controlled_rank_score else "controlled_concentration"
    if concentration_variant == "cooccurrence_concentration":
        controlled = cooccurrence_controlled
        controlled_score = cooccurrence_score
        controlled_metrics = cooccurrence_metrics
    controlled_metrics = {
        **controlled_metrics,
        "score": controlled_score,
        "schema": "controlled_concentration_v8",
        "slots": controlled_slots,
        "selected_concentration_variant": concentration_variant,
        "candidate_variants": {
            "slot_controlled": {
                "rank_score": float(controlled_rank_score),
                "information_retention_rate": slot_controlled_metrics.get("information_retention_rate"),
                "cooccurrence_quality": slot_controlled_metrics.get("cooccurrence_quality"),
                "final_union_count": slot_controlled_metrics.get("final_union_count"),
                "average_jaccard": slot_controlled_metrics.get("average_jaccard"),
            },
            "cooccurrence_concentration": {
                "rank_score": float(cooccurrence_rank_score),
                "information_retention_rate": cooccurrence_metrics.get("information_retention_rate"),
                "cooccurrence_quality": cooccurrence_metrics.get("cooccurrence_quality"),
                "final_union_count": cooccurrence_metrics.get("final_union_count"),
                "average_jaccard": cooccurrence_metrics.get("average_jaccard"),
            },
        },
        "rule": "第二阶段在Backbone后加入共现集中：优先让已发现红球形成同票组合，同时受Retention Guard约束。",
    }
    retention_guard = _retention_guard_decision(backbone_metrics, controlled_metrics, config or {})
    if production_strategy == "coverage_backbone_controlled_concentration":
        result = controlled if retention_guard.get("selected") == "controlled_concentration" else backbone
        portfolio_metrics = controlled_metrics if result is controlled else backbone_metrics
    else:
        result, portfolio_metrics = _beam_select_portfolio(
            candidates,
            count=count,
            layers=red_layers,
            evidence=red_evidence,
            strategy=selected_strategy,
            max_red_overlap=max_red_overlap,
            beam_width=max(6, beam_width),
        )
        retention_guard = {
            **retention_guard,
            "selected": "legacy_information_preservation",
            "reason": "配置未启用Coverage Backbone + Controlled Concentration，保留旧Portfolio路径。",
        }
    result, model_rescue_info = _ensure_model_rescue_channel(
        result,
        candidates,
        rescue_numbers=rescue_numbers,
        evidence=red_evidence,
        max_red_overlap=max_red_overlap,
        min_rescue_numbers_in_final=int(red_cfg.get("min_model_rescue_numbers_in_final", 2) or 2),
    )
    result, ticket_conversion_info = _ensure_candidate_to_method_recovery_ticket(
        result,
        candidates,
        layers=red_layers,
        evidence=red_evidence,
        config=config or {},
        max_red_overlap=max_red_overlap,
    )
    portfolio_before_evidence_guard = [dict(ticket) for ticket in result]
    result, evidence_preservation_info = _ensure_evidence_preservation(
        result,
        candidates,
        layers=red_layers,
        evidence=red_evidence,
        config=config or {},
        max_red_overlap=max_red_overlap,
    )
    _, portfolio_metrics = _portfolio_score(
        result,
        layers=red_layers,
        evidence=red_evidence,
        strategy=selected_strategy,
        max_red_overlap=max_red_overlap,
    )
    if len(result) < count:
        fallback = _select_from_frontier(
            candidates,
            selected=result,
            role="coverage",
            limit=count - len(result),
            max_red_overlap=min(6, max_red_overlap + 1),
            allow_relaxed_overlap=True,
        )
        _extend_unique(result, fallback, role="fill", limit=count - len(result))
        _, portfolio_metrics = _portfolio_score(
            result,
            layers=red_layers,
            evidence=red_evidence,
            strategy=selected_strategy,
            max_red_overlap=max_red_overlap,
        )
    enriched_result: list[dict[str, Any]] = []
    for index, ticket in enumerate(result[:count], start=1):
        tier_counts = _tier_counts(_red_set(ticket), red_layers)
        enriched_result.append({
            **ticket,
            "selection_role": str(ticket.get("selection_role") or "v8_5_evidence_preserving_portfolio"),
            "ticket_selection_role": str(ticket.get("ticket_selection_role") or "v8_5_evidence_preserving_portfolio"),
            "selection_reason": {
                "schema": "red_evidence_preserving_portfolio_v8_5",
                "portfolio_index": int(index),
                "tier_counts": tier_counts,
                "retention_guard": retention_guard,
                "evidence_preservation_guard": evidence_preservation_info,
                "reason": "先用Coverage Backbone保护候选池信息，再用V8.5号码级Evidence Preservation确保Must Preserve进入Final Union。",
            },
        })
    result = enriched_result
    number_exposure_matrix = _number_exposure_matrix(result, layers=red_layers, evidence=red_evidence)
    union_best_ticket_gap = _evidence_concentration_gap(result, layers=red_layers, evidence=red_evidence)
    number_survival_ledger = _number_survival_ledger(
        layers=red_layers,
        evidence=red_evidence,
        portfolio_before_guard=portfolio_before_evidence_guard,
        final_portfolio=result,
        preservation_info=evidence_preservation_info,
    )
    counterfactual = _counterfactual_compression(
        candidates,
        result,
        count=count,
        layers=red_layers,
        evidence=red_evidence,
        max_red_overlap=max_red_overlap,
        beam_width=max(6, beam_width),
    )
    return result[:count], {
        "schema": "red_portfolio_v8_5",
        "active": True,
        "version": "v8_5_evidence_preserving_portfolio",
        "production_strategy": production_strategy,
        "red_ge3_lifecycle": str(red_cfg.get("red_ge3_lifecycle", "shadow_challenger")),
        "budget": budget,
        "slots": {
            "probability": probability_slots,
            "coverage": coverage_slots,
            "exploration": exploration_slots,
        },
        "controlled_concentration_slots": controlled_slots,
        "ticket_count": len(result[:count]),
        "red_pools": red_layers,
        "coverage_backbone": {
            "method": "global_coverage",
            "tickets": [{"red": sorted(_red_set(ticket)), "blue": int(ticket.get("blue", 0) or 0)} for ticket in backbone],
            "metrics": backbone_metrics,
        },
        "controlled_concentration": {
            "tickets": [{"red": sorted(_red_set(ticket)), "blue": int(ticket.get("blue", 0) or 0)} for ticket in controlled],
            "metrics": controlled_metrics,
        },
        "cooccurrence_concentration": {
            "tickets": [{"red": sorted(_red_set(ticket)), "blue": int(ticket.get("blue", 0) or 0)} for ticket in cooccurrence_controlled],
            "metrics": cooccurrence_metrics,
            "pair_count": len(pair_scores),
            "top_pairs": [
                {"pair": [left, right], "score": score}
                for (left, right), score in sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)[:15]
            ],
        },
        "retention_guard": retention_guard,
        "model_rescue": model_rescue_info,
        "ticket_conversion": ticket_conversion_info,
        "evidence_preservation": evidence_preservation_info,
        "number_exposure_matrix": number_exposure_matrix,
        "number_survival_ledger": number_survival_ledger,
        "union_best_ticket_gap": union_best_ticket_gap,
        "information_retention": {**portfolio_metrics, "schema": "information_retention_v8_5"},
        "compression_regret": portfolio_metrics.get("compression_regret"),
        "information_retention_rate": portfolio_metrics.get("information_retention_rate"),
        "counterfactual_compression": counterfactual,
        "pareto": {
            "candidate_count": len(pool_items),
            "frontier_count": len(frontier),
            "portfolio_candidate_count": len(candidates),
            "objective_keys": ["probability", "quality", "structure", "method_vote", "support", "protected_recall", "spread", "cooccurrence"],
            "protected_numbers": sorted(protected_numbers),
        },
        "guard_constraints": {
            "pipeline": ["Guard", "Budget", "Objective", "Candidate→Method Conversion", "Jaccard", "Protected Numbers", "Must Preserve", "Evidence Recovery", "Final Portfolio"],
            "guard_active": bool(runtime.get("red_guard_active", False)),
            "budget_state": str(runtime.get("red_budget_state", budget.get("edge_state", "NO_EDGE"))),
            "objective": runtime_strategy,
            "jaccard_max_red_overlap": max_red_overlap,
            "protected_numbers": sorted(protected_numbers),
            "must_preserve_numbers": red_layers.get("must_preserve", []),
            "should_preserve_numbers": red_layers.get("should_preserve", []),
        },
        "reason": "Candidate Pool → Number Survival Value → Candidate→Method Recovery Ticket → Global Coverage Backbone → Retention Guard → Evidence Preservation → Final 5；red_ge3_objective保留为Shadow/Challenger。",
    }
