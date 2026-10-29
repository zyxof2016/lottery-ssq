from __future__ import annotations

from typing import Any

from src.combination_gen.optimizer import rotational_coverage_optimize
from src.portfolio.coverage import coverage_summary


def _red_signature(ticket: dict) -> tuple[int, ...]:
    return tuple(sorted(int(number) for number in ticket.get("red", [])))


def _portfolio_score(summary: dict[str, Any], unique_red_sets: int) -> float:
    red_coverage = float(summary.get("red_coverage", 0))
    return (
        12.0 * red_coverage
        + 5.0 * float(summary.get("blue_coverage", 0))
        + 4.0 * float(unique_red_sets)
        - 8.0 * float(summary.get("average_red_overlap", 0.0))
    )


def _synthesize_coverage_portfolio(
    candidates: list[dict],
    *,
    count: int,
    minimum_red_coverage: int,
    target_red_coverage: int | None = None,
    minimum_blue_unique_count: int,
) -> list[dict]:
    red_scores = {number: 0.0 for number in range(1, 34)}
    blue_scores = {number: 0.0 for number in range(1, 17)}
    for candidate in candidates:
        quality = float(candidate.get("quality_score", candidate.get("score", 1.0)) or 1.0)
        for number in candidate.get("red", []):
            red_scores[int(number)] += max(quality, 0.0) + 1e-6
        blue_scores[int(candidate.get("blue", 1))] += max(quality, 0.0) + 1e-6

    red_order = sorted(red_scores, key=lambda number: (red_scores[number], -number), reverse=True)
    desired_red_count = min(
        33,
        max(
            int(minimum_red_coverage),
            int(target_red_coverage) if target_red_coverage is not None else int(count) * 6,
        ),
    )
    red_pool = red_order[:desired_red_count]
    if len(red_pool) < desired_red_count:
        for number in red_order[desired_red_count:]:
            if number not in red_pool:
                red_pool.append(number)
            if len(red_pool) >= desired_red_count:
                break

    blue_order = sorted(blue_scores, key=lambda number: (blue_scores[number], -number), reverse=True)
    blue_count = max(1, min(int(count), int(minimum_blue_unique_count), 16))
    blue_pool = blue_order[:blue_count]
    for number in blue_order[blue_count:]:
        if len(blue_pool) >= int(count):
            break
        if number not in blue_pool:
            blue_pool.append(number)
    while len(blue_pool) < int(count):
        for number in range(1, 17):
            if number not in blue_pool:
                blue_pool.append(number)
                break

    tickets: list[dict] = []
    used_red_sets: set[tuple[int, ...]] = set()
    red_stride = max(1, min(5, len(red_pool) // max(1, int(count))))
    for row in range(int(count)):
        start = row * red_stride
        red = [red_pool[(start + offset) % len(red_pool)] for offset in range(min(6, len(red_pool)))]
        if len(set(red)) < 6:
            for number in red_order:
                if number not in red:
                    red.append(number)
                if len(set(red)) >= 6:
                    break
        red_tuple = tuple(sorted(int(number) for number in red[:6]))
        if red_tuple in used_red_sets:
            for number in red_order:
                replacement = tuple(sorted(list(red_tuple[:5]) + [int(number)]))
                if len(set(replacement)) == 6 and replacement not in used_red_sets:
                    red_tuple = replacement
                    break
        used_red_sets.add(red_tuple)
        blue = int(blue_pool[row % len(blue_pool)])
        tickets.append(
            {
                "red": red_tuple,
                "blue": blue,
                "quality_score": float(sum(red_scores[number] for number in red_tuple) / 6.0),
                "selection_role": "final_diversity_synthesized_coverage",
            }
        )
    return tickets


def rebalance_final_portfolio(
    candidates: list[dict],
    selected: list[dict],
    *,
    count: int,
    optimizer_config: dict[str, Any],
    random_seed: int,
    reason_prefix: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Final safety pass that prevents the published portfolio from collapsing.

    Earlier optimizers can legitimately prefer a tight high-score cluster.  That
    is useful only when the signal is strong.  In weak or pending-release
    periods, a tight cluster quietly turns five tickets into near-duplicates, so
    this last pass switches to an explicit coverage matrix when the current
    portfolio violates the configured diversity floor.
    """
    if not bool(optimizer_config.get("final_diversity_rebalance_enabled", True)):
        return selected, {"active": False, "reason": "disabled"}
    if not candidates or not selected or int(count) <= 0:
        return selected, {"active": False, "reason": "empty_input"}

    effective_count = min(int(count), len(selected))
    current = selected[:effective_count]
    current_summary = coverage_summary(current)

    # 核心号码保护：从候选池中提取模型Top推荐的核心号码
    core_red_scores = {number: 0.0 for number in range(1, 34)}
    core_blue_scores = {number: 0.0 for number in range(1, 17)}
    for candidate in candidates[:200]:  # 只看Top200候选
        quality = float(candidate.get("quality_score", candidate.get("consensus_score", 1.0)) or 1.0)
        for number in candidate.get("red", []):
            core_red_scores[int(number)] += max(quality, 0.0) + 1e-6
        core_blue_scores[int(candidate.get("blue", 1))] += max(quality, 0.0) + 1e-6
    core_red = set(sorted(core_red_scores, key=lambda n: (core_red_scores[n], -n), reverse=True)[:12])
    core_blue = set(sorted(core_blue_scores, key=lambda n: (core_blue_scores[n], -n), reverse=True)[:5])

    def _core_number_score(ticket: dict) -> float:
        """计算注包含的核心号码得分，用于保护模型推荐的号码不被替换。"""
        red_core_count = len(set(int(n) for n in ticket.get("red", [])) & core_red)
        blue_core = 1.0 if int(ticket.get("blue", 0)) in core_blue else 0.0
        return float(red_core_count) + 2.0 * blue_core

    current_core_score = sum(_core_number_score(ticket) for ticket in current)

    current_unique_red_sets = len({_red_signature(ticket) for ticket in current})

    min_red_coverage = int(
        optimizer_config.get(
            "final_diversity_min_red_coverage",
            optimizer_config.get("signal_minimum_red_coverage", max(6, effective_count + 5)),
        )
    )
    min_unique_red_sets = int(
        optimizer_config.get(
            "final_diversity_min_unique_red_sets",
            optimizer_config.get("signal_minimum_unique_red_sets", effective_count),
        )
    )
    max_average_red_overlap = float(
        optimizer_config.get("final_diversity_max_average_red_overlap", 3.0)
    )
    target_red_coverage = int(
        optimizer_config.get(
            "final_diversity_target_red_coverage",
            max(min_red_coverage, effective_count * 4),
        )
    )
    max_red_coverage = int(
        optimizer_config.get(
            "final_diversity_max_red_coverage",
            max(target_red_coverage + 2, min_red_coverage),
        )
    )

    triggers: list[str] = []
    if int(current_summary.get("red_coverage", 0)) < min_red_coverage:
        triggers.append("red_coverage_below_floor")
    if current_unique_red_sets < min(effective_count, min_unique_red_sets):
        triggers.append("duplicate_red_sets")
    if float(current_summary.get("average_red_overlap", 0.0)) > max_average_red_overlap:
        triggers.append("average_red_overlap_too_high")

    if not triggers:
        return selected, {
            "active": False,
            "reason": "portfolio_already_diverse",
            "coverage": current_summary,
            "unique_red_sets": current_unique_red_sets,
        }

    rebalanced = rotational_coverage_optimize(
        candidates,
        effective_count,
        max_red_overlap=int(
            optimizer_config.get(
                "final_diversity_max_red_overlap",
                optimizer_config.get("signal_max_red_overlap", optimizer_config.get("max_red_overlap", 2)),
            )
        ),
        minimum_blue_unique_count=int(
            optimizer_config.get(
                "final_diversity_minimum_blue_unique_count",
                optimizer_config.get(
                    "signal_minimum_blue_unique_count",
                    optimizer_config.get("minimum_blue_unique_count", min(effective_count, 5)),
                ),
            )
        ),
        random_seed=int(random_seed),
    )
    if 0 < len(rebalanced) < effective_count:
        used_keys = {(_red_signature(ticket), int(ticket.get("blue", 0))) for ticket in rebalanced}
        for candidate in sorted(
            candidates,
            key=lambda item: float(item.get("quality_score", item.get("score", 0.0))),
            reverse=True,
        ):
            key = (_red_signature(candidate), int(candidate.get("blue", 0)))
            if key in used_keys:
                continue
            rebalanced.append(candidate)
            used_keys.add(key)
            if len(rebalanced) >= effective_count:
                break
    if not rebalanced:
        return selected, {
            "active": False,
            "reason": "rebalance_no_candidate",
            "triggers": triggers,
            "coverage": current_summary,
            "unique_red_sets": current_unique_red_sets,
        }

    rebalanced_summary = coverage_summary(rebalanced)
    rebalanced_unique_red_sets = len({_red_signature(ticket) for ticket in rebalanced})
    synthesized: list[dict] = []
    synthesized_summary: dict[str, Any] | None = None
    synthesized_unique_red_sets = 0
    rebalanced_too_wide = int(rebalanced_summary.get("red_coverage", 0)) > max_red_coverage
    if (
        bool(optimizer_config.get("final_diversity_synthesize_if_needed", True))
        and (
            int(rebalanced_summary.get("red_coverage", 0)) < min_red_coverage
            or rebalanced_too_wide
        )
    ):
        synthesized = _synthesize_coverage_portfolio(
            candidates,
            count=effective_count,
            minimum_red_coverage=min_red_coverage,
            target_red_coverage=target_red_coverage,
            minimum_blue_unique_count=int(
                optimizer_config.get(
                    "final_diversity_minimum_blue_unique_count",
                    optimizer_config.get(
                        "signal_minimum_blue_unique_count",
                        optimizer_config.get("minimum_blue_unique_count", min(effective_count, 5)),
                    ),
                )
            ),
        )
        synthesized_summary = coverage_summary(synthesized)
        synthesized_unique_red_sets = len({_red_signature(ticket) for ticket in synthesized})
        if (
            rebalanced_too_wide
            and int(synthesized_summary.get("red_coverage", 0)) >= min_red_coverage
        ) or _portfolio_score(synthesized_summary, synthesized_unique_red_sets) > _portfolio_score(
            rebalanced_summary,
            rebalanced_unique_red_sets,
        ):
            rebalanced = synthesized
            rebalanced_summary = synthesized_summary
            rebalanced_unique_red_sets = synthesized_unique_red_sets
    current_score = _portfolio_score(current_summary, current_unique_red_sets)
    rebalanced_score = _portfolio_score(rebalanced_summary, rebalanced_unique_red_sets)
    rebalanced_core_score = sum(_core_number_score(ticket) for ticket in rebalanced)
    force_apply = bool(optimizer_config.get("final_diversity_force_apply_when_triggered", True))
    # 核心号码保护：如果再平衡后核心号码得分下降超过20%，则不应用再平衡
    core_score_ratio = rebalanced_core_score / max(current_core_score, 1e-9)
    core_protection_active = bool(optimizer_config.get("final_diversity_core_protection_enabled", True))
    if core_protection_active and core_score_ratio < 0.80:
        return selected, {
            "active": False,
            "reason": "core_number_protection_blocked_rebalance",
            "triggers": triggers,
            "before": current_summary,
            "after": rebalanced_summary,
            "before_core_score": current_core_score,
            "after_core_score": rebalanced_core_score,
            "core_score_ratio": core_score_ratio,
            "core_red_protected": sorted(core_red),
            "core_blue_protected": sorted(core_blue),
        }
    if not force_apply and rebalanced_score <= current_score:
        return selected, {
            "active": False,
            "reason": "rebalance_not_better",
            "triggers": triggers,
            "before": current_summary,
            "after": rebalanced_summary,
            "before_unique_red_sets": current_unique_red_sets,
            "after_unique_red_sets": rebalanced_unique_red_sets,
            "before_core_score": current_core_score,
            "after_core_score": rebalanced_core_score,
        }

    info = {
        "active": True,
        "reason": reason_prefix,
        "triggers": triggers,
        "before": current_summary,
        "after": rebalanced_summary,
        "synthesized_after": synthesized_summary,
        "before_unique_red_sets": current_unique_red_sets,
        "after_unique_red_sets": rebalanced_unique_red_sets,
        "synthesized_unique_red_sets": synthesized_unique_red_sets,
        "before_score": current_score,
        "after_score": rebalanced_score,
    }
    tagged = [
        {
            **ticket,
            "selection_role": reason_prefix,
            "selection_reason": info,
        }
        for ticket in rebalanced
    ]
    return tagged, info
