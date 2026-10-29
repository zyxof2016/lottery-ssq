from __future__ import annotations

from typing import Any


def _limited_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return list(value)[:limit]


def compact_sampling_log(payload: dict[str, Any] | None, *, max_list: int = 12) -> dict[str, Any]:
    """Compact candidate-generation diagnostics for runtime logs.

    Full diagnostics remain in JSON reports/traces.  This prevents long
    walk-forward runs from spending excessive time formatting and writing huge
    nested structures into pipeline.log.
    """

    if not isinstance(payload, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "mode",
        "reason",
        "requested_samples",
        "samples",
        "pool_size",
        "retained_candidates",
        "unique_candidates",
        "duplicate_draws",
        "guarded_pool_candidates",
        "coverage_pool_candidates",
        "stoploss_pool_candidates",
        "merged_candidate_pool",
        "candidate_bank_cache_hit",
        "candidate_bank_cache_key",
    ):
        if key in payload:
            summary[key] = payload.get(key)

    protection = payload.get("red_candidate_recall_protection")
    if isinstance(protection, dict):
        compact_protection = {
            "schema": protection.get("schema"),
            "enabled": protection.get("enabled"),
            "reason": protection.get("reason"),
            "stage": protection.get("stage"),
            "protected_count": len(protection.get("protected_numbers", []) or []),
            "protected_numbers": _limited_list(protection.get("protected_numbers"), max_list),
            "fusion_topn_guard_count": len(protection.get("fusion_topn_guard_numbers", []) or []),
            "fusion_topn_guard_numbers": _limited_list(
                protection.get("fusion_topn_guard_numbers"),
                max_list,
            ),
        }
        enforcement = protection.get("candidate_slot_enforcement")
        if isinstance(enforcement, dict):
            compact_protection["candidate_slot_enforcement"] = {
                "enabled": enforcement.get("enabled"),
                "added_count": len(enforcement.get("added", []) or []),
                "added": _limited_list(enforcement.get("added"), max_list),
                "protected_present_count": len(enforcement.get("protected_present", []) or []),
                "protected_missing_count": len(enforcement.get("protected_missing", []) or []),
            }
        summary["red_candidate_recall_protection"] = compact_protection

    exclusion = payload.get("exclusion_filter")
    if isinstance(exclusion, dict):
        summary["exclusion_filter"] = {
            "enabled": exclusion.get("enabled"),
            "reason": exclusion.get("reason"),
            "red_excluded_count": len(exclusion.get("red_excluded", []) or []),
            "red_excluded": _limited_list(exclusion.get("red_excluded"), max_list),
            "blue_excluded_count": len(exclusion.get("blue_excluded", []) or []),
            "blue_excluded": _limited_list(exclusion.get("blue_excluded"), max_list),
        }

    stoploss = payload.get("red_guard_stoploss")
    if isinstance(stoploss, dict):
        summary["red_guard_stoploss"] = {
            "active": stoploss.get("active"),
            "reason": stoploss.get("reason"),
            "method": stoploss.get("method"),
        }
    return summary
