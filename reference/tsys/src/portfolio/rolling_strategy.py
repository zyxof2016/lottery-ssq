from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def summarize_multi_period_strategy(
    period_records: list[dict[str, Any]] | None,
    *,
    window: int = 10,
    top_red_count: int = 10,
    top_blue_count: int = 5,
) -> dict[str, Any]:
    """Summarize recent rolling evidence for multi-period portfolio planning."""
    records = list(period_records or [])[-max(1, int(window)):]
    if not records:
        return {"available": False, "reason": "no_completed_backtest_records"}
    method_scores: dict[str, dict[str, float]] = {}
    red_hits_by_method: dict[str, list[float]] = defaultdict(list)
    any3_by_method: dict[str, list[float]] = defaultdict(list)
    blue_by_method: dict[str, list[float]] = defaultdict(list)
    red_counter: Counter[int] = Counter()
    blue_counter: Counter[int] = Counter()

    for record in records:
        actual_red = set(int(number) for number in (record.get("actual", {}) or {}).get("red", []))
        actual_blue = int((record.get("actual", {}) or {}).get("blue", 0) or 0)
        for method, payload in (record.get("methods", {}) or {}).items():
            tickets = list((payload or {}).get("tickets", []) or [])
            if not tickets:
                continue
            best_hit = 0
            blue_hit = 0
            for ticket in tickets:
                red = [int(number) for number in ticket.get("red", [])]
                hit = len(set(red) & actual_red)
                best_hit = max(best_hit, hit)
                blue_hit = max(blue_hit, int(int(ticket.get("blue", 0) or 0) == actual_blue))
                for number in red:
                    red_counter[number] += 1
                blue_counter[int(ticket.get("blue", 0) or 0)] += 1
            red_hits_by_method[method].append(float(best_hit))
            any3_by_method[method].append(float(best_hit >= 3))
            blue_by_method[method].append(float(blue_hit))

    for method in sorted(red_hits_by_method):
        red_values = red_hits_by_method[method]
        any3_values = any3_by_method[method]
        blue_values = blue_by_method[method]
        count = max(1, len(red_values))
        method_scores[method] = {
            "periods": float(count),
            "average_best_red_hits": float(sum(red_values) / count),
            "period_any_red_ge3_ratio": float(sum(any3_values) / count),
            "period_any_blue_hit_ratio": float(sum(blue_values) / count),
        }
    best_method = max(
        method_scores,
        key=lambda name: (
            method_scores[name]["period_any_red_ge3_ratio"],
            method_scores[name]["average_best_red_hits"],
            method_scores[name]["period_any_blue_hit_ratio"],
        ),
        default=None,
    )
    return {
        "available": True,
        "window": int(len(records)),
        "best_recent_method": best_method,
        "method_scores": method_scores,
        "stable_red_core": [number for number, _ in red_counter.most_common(int(top_red_count))],
        "stable_blue_core": [number for number, _ in blue_counter.most_common(int(top_blue_count)) if number],
        "usage": "rolling_memory_reference_for_next_portfolio_review",
    }


def _ticket_key(ticket: dict[str, Any]) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(int(number) for number in ticket.get("red", []) or [])), int(ticket.get("blue", 0) or 0)


def _ticket_hit_score(ticket: dict[str, Any]) -> float:
    return (
        float(ticket.get("red_hits", 0) or 0)
        + 1.35 * float(bool(ticket.get("blue_hit", False)))
        + 0.35 * float(bool((ticket.get("red_hits", 0) or 0) >= 3))
    )


def apply_rolling_portfolio_memory(
    current_tickets: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    period_records: list[dict[str, Any]] | None,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Let recent walk-forward evidence lightly influence the next portfolio.

    The summary report already shows which methods/tickets worked recently; this
    function turns that evidence into a small number of real portfolio anchors.
    It deliberately preserves only a minority of slots so one lucky recent round
    cannot hijack the whole output.
    """
    cfg = config or {}
    if not bool(cfg.get("rolling_portfolio_memory_enabled", False)):
        return list(current_tickets or []), {"active": False, "reason": "disabled"}
    count = len(current_tickets or [])
    if count <= 0:
        return list(current_tickets or []), {"active": False, "reason": "empty_current_tickets"}
    records = list(period_records or [])[-max(1, int(cfg.get("rolling_portfolio_memory_window", 20))):]
    min_periods = max(1, int(cfg.get("rolling_portfolio_memory_min_periods", 8)))
    if len(records) < min_periods:
        return list(current_tickets or []), {
            "active": False,
            "reason": "insufficient_rolling_periods",
            "periods": len(records),
            "minimum_periods": min_periods,
        }

    red_counter: Counter[int] = Counter()
    blue_counter: Counter[int] = Counter()
    method_scores: dict[str, float] = defaultdict(float)
    replayed_tickets: list[dict[str, Any]] = []
    min_hit = float(cfg.get("rolling_portfolio_memory_min_hit", 2.5))
    for record in records:
        for method, payload in (record.get("methods", {}) or {}).items():
            tickets = list((payload or {}).get("tickets", []) or [])
            if not tickets:
                continue
            best = max((_ticket_hit_score(ticket) for ticket in tickets), default=0.0)
            method_scores[str(method)] += best
            for ticket in tickets:
                score = _ticket_hit_score(ticket)
                if score < min_hit:
                    continue
                red = sorted({int(number) for number in ticket.get("red", []) or [] if 1 <= int(number) <= 33})
                blue = int(ticket.get("blue", 0) or 0)
                if len(red) != 6 or not (1 <= blue <= 16):
                    continue
                for number in red:
                    red_counter[number] += 1
                blue_counter[blue] += 1
                replayed_tickets.append({
                    "red": red,
                    "blue": blue,
                    "_rolling_score": score,
                    "_rolling_method": str(method),
                })

    if not replayed_tickets or not red_counter:
        return list(current_tickets or []), {
            "active": False,
            "reason": "no_recent_ticket_above_threshold",
            "periods": len(records),
        }

    stable_red = {number for number, _ in red_counter.most_common(int(cfg.get("rolling_portfolio_memory_red_core_size", 10)))}
    stable_blue = {number for number, _ in blue_counter.most_common(int(cfg.get("rolling_portfolio_memory_blue_core_size", 5)))}
    best_method = max(method_scores, key=method_scores.get, default="")
    candidate_map = {_ticket_key(ticket): ticket for ticket in candidate_pool or []}

    anchor_candidates: list[dict[str, Any]] = []
    for ticket in replayed_tickets:
        key = _ticket_key(ticket)
        pool_ticket = candidate_map.get(key)
        if pool_ticket is not None:
            anchor_candidates.append(pool_ticket)
    if not anchor_candidates:
        # If the exact replayed ticket is absent from the fresh candidate pool,
        # choose fresh candidates aligned with the stable recent red/blue cores.
        anchor_candidates = list(candidate_pool or [])

    preserve = max(0, min(count - 1, int(cfg.get("rolling_portfolio_memory_preserve_count", 2))))
    if preserve <= 0:
        return list(current_tickets or []), {"active": False, "reason": "preserve_count_zero"}

    def anchor_score(ticket: dict[str, Any]) -> float:
        red = {int(number) for number in ticket.get("red", []) or []}
        blue = int(ticket.get("blue", 0) or 0)
        return float(
            1.10 * len(red & stable_red)
            + 0.85 * float(blue in stable_blue)
            + 0.65 * float(ticket.get("quality_score", 0.0) or 0.0)
            + 0.35 * float(ticket.get("consensus_score", 0.0) or 0.0)
            + 0.20 * float(ticket.get("blue_number_score", 0.0) or 0.0)
        )

    anchors: list[dict[str, Any]] = []
    used: set[tuple[tuple[int, ...], int]] = set()
    for ticket in sorted(anchor_candidates, key=anchor_score, reverse=True):
        key = _ticket_key(ticket)
        if key in used:
            continue
        anchors.append({
            **ticket,
            "selection_role": f"{ticket.get('selection_role', 'selected')}_rolling_memory_anchor",
            "selection_reason": {
                **(ticket.get("selection_reason", {}) if isinstance(ticket.get("selection_reason"), dict) else {}),
                "rolling_memory": "recent_walk_forward_ticket_or_core_alignment",
                "best_recent_method": best_method,
            },
        })
        used.add(key)
        if len(anchors) >= preserve:
            break

    if not anchors:
        return list(current_tickets or []), {"active": False, "reason": "no_anchor_selected"}

    result = list(anchors)
    for ticket in list(current_tickets or []) + list(candidate_pool or []):
        if len(result) >= count:
            break
        key = _ticket_key(ticket)
        if key in used:
            continue
        result.append(ticket)
        used.add(key)
    return result[:count], {
        "active": True,
        "periods": len(records),
        "preserved": len(anchors),
        "best_recent_method": best_method,
        "stable_red_core": sorted(stable_red),
        "stable_blue_core": sorted(stable_blue),
    }
