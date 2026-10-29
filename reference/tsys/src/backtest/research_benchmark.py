from __future__ import annotations

from typing import Any, Iterable

from src.backtest.metrics import paired_significance

DEFAULT_WINDOWS = (30, 50, 100, 200, 300, 500)


def _tail(records: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    return records[-window:] if len(records) >= window else []


def _method_metrics(records: list[dict[str, Any]], method: str, seed: int) -> dict[str, Any]:
    model = []
    random = []
    blue = []
    for record in records:
        detail = (record.get("methods") or {}).get(method) or {}
        m = detail.get("model") or {}
        r = detail.get("random") or {}
        model.append(float(m.get("average_red_hits", 0.0)))
        random.append(float(r.get("average_red_hits", 0.0)))
        blue.append(float(m.get("blue_hit_ratio", 0.0)))
    significance = paired_significance(model, random, random_seed=seed)
    return {
        "periods": len(records),
        "average_red_hits": sum(model) / len(model) if model else 0.0,
        "random_average_red_hits": sum(random) / len(random) if random else 0.0,
        "red_edge": (sum(model) - sum(random)) / len(model) if model else 0.0,
        "blue_hit_rate": sum(blue) / len(blue) if blue else 0.0,
        "significance": significance,
        "status": (
            "EDGE"
            if significance.get("significant_at_0_05") and significance.get("mean_difference", 0.0) > 0
            else "NO_EDGE"
        ),
    }


def _fused_metrics(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    model = []
    random = []
    blue = []
    blue_top5 = []
    for record in records:
        fused = record.get("fused_validation") or {}
        model.append(float(fused.get("red_hits", 0.0)))
        # matched random baseline is evaluated in each method; consensus is the
        # conservative reference for the same ticket-count objective.
        consensus = (record.get("methods") or {}).get("consensus") or {}
        random.append(float((consensus.get("random") or {}).get("average_red_hits", 0.0)))
        blue.append(float(bool(fused.get("blue_hit"))))
        ranking = fused.get("blue_ranking") or {}
        blue_top5.append(float(ranking.get("hit_at_5", 0.0)))
    sig = paired_significance(model, random, random_seed=seed)
    return {
        "periods": len(records),
        "average_red_hits": sum(model) / len(model) if model else 0.0,
        "random_average_red_hits": sum(random) / len(random) if random else 0.0,
        "red_edge": sig.get("mean_difference", 0.0),
        "blue_top1_hit_rate": sum(blue) / len(blue) if blue else 0.0,
        "blue_top5_hit_rate": sum(blue_top5) / len(blue_top5) if blue_top5 else 0.0,
        "significance": sig,
        "status": (
            "EDGE"
            if sig.get("significant_at_0_05") and sig.get("mean_difference", 0.0) > 0
            else "NO_EDGE"
        ),
    }


def build_research_benchmark(
    period_records: Iterable[dict[str, Any]],
    *,
    random_seed: int = 2026,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    """Research arena: identical chronological OOS windows for methods and fusion.

    Missing windows are explicitly marked unavailable; no shorter tail is silently
    substituted. This prevents window-shopping/selection bias when comparing 30/50/
    100/200/300/500-period stability.
    """
    records = list(period_records)
    available = []
    unavailable = []
    by_window: dict[str, Any] = {}
    methods = sorted(
        {
            str(name)
            for record in records
            for name in (record.get("methods") or {}).keys()
        }
    )
    for window in windows:
        size = int(window)
        if len(records) < size:
            unavailable.append(size)
            by_window[str(size)] = {"periods": len(records), "available": False}
            continue
        subset = _tail(records, size)
        method_results = {
            name: _method_metrics(subset, name, random_seed + size * 101 + index)
            for index, name in enumerate(methods)
        }
        by_window[str(size)] = {
            "available": True,
            "periods": size,
            "first_period": str(subset[0].get("validation_period", subset[0].get("period", ""))),
            "last_period": str(subset[-1].get("validation_period", subset[-1].get("period", ""))),
            "methods": method_results,
            "fusion": _fused_metrics(subset, random_seed + size * 1009),
        }
        available.append(size)

    # A strategy is stable only when it has positive, non-significant-loss
    # direction across every requested window that exists. Formal promotion still
    # requires the release gate; this is a research ranking, not a release rule.
    stability = {}
    for name in methods:
        rows = [by_window[str(w)]["methods"][name] for w in available]
        positive_windows = sum(float(row["red_edge"]) > 0 for row in rows)
        edge_windows = sum(row["status"] == "EDGE" for row in rows)
        stability[name] = {
            "available_windows": available,
            "positive_edge_windows": positive_windows,
            "significant_edge_windows": edge_windows,
            "stable_positive": bool(rows) and positive_windows == len(rows),
            "promotion_eligible": bool(rows) and positive_windows == len(rows) and edge_windows >= 1,
        }

    return {
        "schema": "tsys_research_benchmark_v1",
        "windows_requested": [int(w) for w in windows],
        "available_windows": available,
        "unavailable_windows": unavailable,
        "methods": methods,
        "by_window": by_window,
        "stability": stability,
        "random_control": {
            "paired_by_same_target_period": True,
            "significance": "paired_sign_flip_permutation",
            "multiple_testing": "method-level results remain diagnostic; production promotion is controlled by Release Gate/Holm evidence",
        },
        "promotion_rule": (
            "Research benchmark never promotes by itself. Candidate methods remain Shadow/Research "
            "until the independent release gate and prospective audit requirements pass."
        ),
    }
