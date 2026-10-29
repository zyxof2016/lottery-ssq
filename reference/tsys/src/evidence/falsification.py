from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def _normalize(values: Any, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != int(size) or not np.all(np.isfinite(array)):
        return np.full(int(size), 1.0 / float(size), dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return np.full(int(size), 1.0 / float(size), dtype=float)
    return array / total


def _record_probability(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    fused = record.get("fused_probability") or record.get("fused_probabilities")
    if isinstance(fused, dict) and "red" in fused and "blue" in fused:
        return _normalize(fused.get("red"), 33), _normalize(fused.get("blue"), 16)
    raw = record.get("model_oof_probabilities")
    if not isinstance(raw, dict) or not raw:
        return None
    red_arrays: list[np.ndarray] = []
    blue_arrays: list[np.ndarray] = []
    for payload in raw.values():
        if not isinstance(payload, dict):
            continue
        red_arrays.append(_normalize(payload.get("red"), 33))
        blue_arrays.append(_normalize(payload.get("blue"), 16))
    if not red_arrays or not blue_arrays:
        return None
    return _normalize(np.mean(np.stack(red_arrays, axis=0), axis=0), 33), _normalize(
        np.mean(np.stack(blue_arrays, axis=0), axis=0),
        16,
    )


def _actual(record: dict[str, Any]) -> tuple[set[int], int] | None:
    raw = record.get("actual") if isinstance(record, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        red = {int(value) for value in raw.get("red", [])}
        blue = int(raw.get("blue"))
    except (TypeError, ValueError):
        return None
    if len(red) != 6 or not all(1 <= value <= 33 for value in red) or not 1 <= blue <= 16:
        return None
    return red, blue


def _rows(records: list[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray, set[int], int]]:
    result: list[tuple[np.ndarray, np.ndarray, set[int], int]] = []
    for record in records or []:
        probability = _record_probability(record)
        actual = _actual(record)
        if probability is None or actual is None:
            continue
        result.append((probability[0], probability[1], actual[0], actual[1]))
    return result


def _score_rows(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]]) -> dict[str, float]:
    if not rows:
        return {"rows": 0.0, "red_top6_hits": 0.0, "blue_top1_hit": 0.0, "blue_top5_hit": 0.0}
    red_hits: list[float] = []
    blue1: list[float] = []
    blue5: list[float] = []
    for red_probability, blue_probability, actual_red, actual_blue in rows:
        red_top6 = {int(index + 1) for index in np.argsort(red_probability)[-6:]}
        blue_order = [int(index + 1) for index in np.argsort(blue_probability)[::-1]]
        red_hits.append(float(len(red_top6 & actual_red)))
        blue1.append(float(blue_order[0] == actual_blue))
        blue5.append(float(actual_blue in set(blue_order[:5])))
    return {
        "rows": float(len(rows)),
        "red_top6_hits": float(np.mean(red_hits)),
        "blue_top1_hit": float(np.mean(blue1)),
        "blue_top5_hit": float(np.mean(blue5)),
    }


def _row_quality_scores(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]]) -> np.ndarray:
    scores: list[float] = []
    for red_probability, blue_probability, actual_red, actual_blue in rows:
        red_top6 = {int(index + 1) for index in np.argsort(red_probability)[-6:]}
        blue_order = [int(index + 1) for index in np.argsort(blue_probability)[::-1]]
        red_component = float(len(red_top6 & actual_red)) / 6.0
        blue_component = 1.0 if blue_order and blue_order[0] == actual_blue else 0.0
        blue_cover_component = 0.35 if actual_blue in set(blue_order[:5]) else 0.0
        scores.append(0.70 * red_component + 0.20 * blue_component + 0.10 * blue_cover_component)
    return np.asarray(scores, dtype=float)


def _bootstrap_ci(differences: np.ndarray, *, seed: int, repeats: int = 1000) -> tuple[float | None, float | None]:
    if differences.size < 2:
        return None, None
    rng = np.random.default_rng(int(seed))
    means = [
        float(np.mean(rng.choice(differences, size=differences.size, replace=True)))
        for _ in range(int(repeats))
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _permutation_p_value(differences: np.ndarray, *, seed: int, repeats: int = 1000) -> float | None:
    if differences.size < 2:
        return None
    rng = np.random.default_rng(int(seed))
    observed = abs(float(np.mean(differences)))
    more_extreme = 0
    for _ in range(int(repeats)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=differences.size, replace=True)
        if abs(float(np.mean(differences * signs))) >= observed:
            more_extreme += 1
    return float((more_extreme + 1) / (int(repeats) + 1))


def _statistical_decision(
    real_rows: list[tuple[np.ndarray, np.ndarray, set[int], int]],
    random_rows: list[tuple[np.ndarray, np.ndarray, set[int], int]],
    *,
    seed: int,
) -> dict[str, Any]:
    paired_count = min(len(real_rows), len(random_rows))
    if paired_count < 8:
        return {
            "schema": "falsification_statistical_decision_v7",
            "status": "NO_EDGE",
            "reason": "insufficient_paired_oof_rows",
            "paired_rows": int(paired_count),
        }
    real_scores = _row_quality_scores(real_rows[-paired_count:])
    random_scores = _row_quality_scores(random_rows[-paired_count:])
    differences = real_scores - random_scores
    mean_difference = float(np.mean(differences))
    ci_low, ci_high = _bootstrap_ci(differences, seed=seed)
    p_value = _permutation_p_value(differences, seed=seed + 1)
    std = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    effect_size = None if std <= 1e-12 else float(mean_difference / std)
    if p_value is not None and ci_low is not None and ci_low > 0.0 and p_value <= 0.05:
        status = "PASS"
        reason = "real_oof_significantly_beats_random"
    elif p_value is not None and ci_high is not None and ci_high < 0.0 and p_value <= 0.05:
        status = "FAIL"
        reason = "random_oof_significantly_beats_real"
    else:
        status = "NO_EDGE"
        reason = "difference_not_statistically_significant"
    return {
        "schema": "falsification_statistical_decision_v7",
        "status": status,
        "reason": reason,
        "paired_rows": int(paired_count),
        "real_mean": float(np.mean(real_scores)),
        "random_mean": float(np.mean(random_scores)),
        "difference": mean_difference,
        "bootstrap_ci_95": [ci_low, ci_high],
        "permutation_p_value": p_value,
        "effect_size": effect_size,
        "rule": "PASS if Real-Random CI is positive with p<=0.05; FAIL if negative with p<=0.05; otherwise NO_EDGE.",
    }


def _feature_matrix(
    rows: list[tuple[np.ndarray, np.ndarray, set[int], int]],
    upto: int,
    *,
    color: str,
    mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 31 * upto + (0 if color == "red" else 10_000))
    samples: list[list[float]] = []
    labels: list[int] = []
    size = 33 if color == "red" else 16
    for row_index, (red_probability, blue_probability, actual_red, actual_blue) in enumerate(rows[:upto]):
        probability = red_probability if color == "red" else blue_probability
        order = np.argsort(probability)[::-1]
        ranks = np.empty(size, dtype=float)
        ranks[order] = np.arange(1, size + 1, dtype=float)
        entropy = float(-np.sum(np.clip(probability, 1e-12, 1.0) * np.log(np.clip(probability, 1e-12, 1.0))) / np.log(size))
        for number in range(1, size + 1):
            if mode == "random":
                feature = rng.normal(0.0, 1.0, size=6).tolist()
            else:
                p = float(probability[number - 1])
                rank_pct = float(ranks[number - 1] / size)
                feature = [
                    p,
                    rank_pct,
                    float(number / size),
                    float(row_index / max(1, len(rows))),
                    entropy,
                    float(p * (1.0 - rank_pct)),
                ]
                if mode == "seeded":
                    feature = [float(value + rng.normal(0.0, 0.002)) for value in feature]
            samples.append(feature)
            if color == "red":
                labels.append(int(number in actual_red))
            else:
                labels.append(int(number == actual_blue))
    return np.asarray(samples, dtype=float), np.asarray(labels, dtype=int)


def _target_features(
    row: tuple[np.ndarray, np.ndarray, set[int], int],
    row_index: int,
    total_rows: int,
    *,
    color: str,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 77 * row_index + (0 if color == "red" else 20_000))
    red_probability, blue_probability, _, _ = row
    probability = red_probability if color == "red" else blue_probability
    size = 33 if color == "red" else 16
    order = np.argsort(probability)[::-1]
    ranks = np.empty(size, dtype=float)
    ranks[order] = np.arange(1, size + 1, dtype=float)
    entropy = float(-np.sum(np.clip(probability, 1e-12, 1.0) * np.log(np.clip(probability, 1e-12, 1.0))) / np.log(size))
    samples: list[list[float]] = []
    for number in range(1, size + 1):
        if mode == "random":
            feature = rng.normal(0.0, 1.0, size=6).tolist()
        else:
            p = float(probability[number - 1])
            rank_pct = float(ranks[number - 1] / size)
            feature = [
                p,
                rank_pct,
                float(number / size),
                float(row_index / max(1, total_rows)),
                entropy,
                float(p * (1.0 - rank_pct)),
            ]
            if mode == "seeded":
                feature = [float(value + rng.normal(0.0, 0.002)) for value in feature]
        samples.append(feature)
    return np.asarray(samples, dtype=float)


def _train_binary_probability(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, seed: int) -> np.ndarray:
    if len(set(int(value) for value in y_train.tolist())) < 2:
        return np.full(x_test.shape[0], 1.0 / max(1, x_test.shape[0]))
    model = LogisticRegression(
        solver="liblinear",
        max_iter=120,
        random_state=int(seed),
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)


def _oof_retrained_rows(
    rows: list[tuple[np.ndarray, np.ndarray, set[int], int]],
    *,
    seed: int,
    mode: str,
    max_rows: int = 120,
) -> tuple[list[tuple[np.ndarray, np.ndarray, set[int], int]], int]:
    usable = list(rows or [])[-max_rows:]
    min_train = min(12, max(4, len(usable) // 4))
    if len(usable) <= min_train:
        return [], 0
    oof_rows: list[tuple[np.ndarray, np.ndarray, set[int], int]] = []
    runs = 0
    for index in range(min_train, len(usable)):
        red_x, red_y = _feature_matrix(usable, index, color="red", mode=mode, seed=seed)
        blue_x, blue_y = _feature_matrix(usable, index, color="blue", mode=mode, seed=seed)
        red_test = _target_features(usable[index], index, len(usable), color="red", mode=mode, seed=seed)
        blue_test = _target_features(usable[index], index, len(usable), color="blue", mode=mode, seed=seed)
        red_probability = _normalize(_train_binary_probability(red_x, red_y, red_test, seed=seed + index), 33)
        blue_probability = _normalize(_train_binary_probability(blue_x, blue_y, blue_test, seed=seed + 10_000 + index), 16)
        oof_rows.append((red_probability, blue_probability, usable[index][2], usable[index][3]))
        runs += 2
    return oof_rows, runs


def _label_permutation(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]]) -> dict[str, Any]:
    observed = _score_rows(rows)
    if len(rows) < 8:
        return {"executed": True, "passed": False, "reason": "insufficient_rows", "observed": observed}
    null_scores: list[dict[str, float]] = []
    null_rows_for_stats: list[tuple[np.ndarray, np.ndarray, set[int], int]] = []
    max_shift = min(len(rows) - 1, 64)
    for shift in range(1, max_shift + 1):
        shifted = [
            (rows[index][0], rows[index][1], rows[(index + shift) % len(rows)][2], rows[(index + shift) % len(rows)][3])
            for index in range(len(rows))
        ]
        null_scores.append(_score_rows(shifted))
        if shift == 1:
            null_rows_for_stats = shifted
    red_null = [item["red_top6_hits"] for item in null_scores]
    blue_null = [item["blue_top1_hit"] for item in null_scores]
    statistical = _statistical_decision(rows, null_rows_for_stats, seed=2026)
    return {
        "executed": True,
        "passed": bool(statistical["status"] != "FAIL"),
        "statistical_decision": statistical,
        "observed": observed,
        "null": {
            "permutations": len(null_scores),
            "red_top6_hits_mean": float(np.mean(red_null)),
            "blue_top1_hit_mean": float(np.mean(blue_null)),
            "red_empirical_p_value": float((1 + sum(value >= observed["red_top6_hits"] for value in red_null)) / (len(red_null) + 1)),
            "blue_empirical_p_value": float((1 + sum(value >= observed["blue_top1_hit"] for value in blue_null)) / (len(blue_null) + 1)),
        },
        "metric": "actual labels are circularly permuted across completed OOS rows",
    }


def _random_feature(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]], seed: int) -> dict[str, Any]:
    if len(rows) < 8:
        return {"executed": True, "passed": False, "reason": "insufficient_rows"}
    observed_rows, observed_runs = _oof_retrained_rows(rows, seed=seed, mode="model")
    random_rows, random_runs = _oof_retrained_rows(rows, seed=seed + 710_000, mode="random")
    observed = _score_rows(observed_rows or rows)
    random_score = _score_rows(random_rows)
    statistical = _statistical_decision(observed_rows or rows, random_rows, seed=seed + 720_000)
    return {
        "executed": True,
        "passed": bool(statistical["status"] != "FAIL"),
        "decision": statistical["status"],
        "statistical_decision": statistical,
        "observed": observed,
        "random_feature": random_score,
        "training_runs": {"observed_oof_refits": observed_runs, "random_feature_oof_refits": random_runs},
        "metric": "random features are retrained into OOF predictions and must not create a stronger apparent edge",
    }


def _synthetic_null(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]], seed: int) -> dict[str, Any]:
    if len(rows) < 8:
        return {"executed": True, "passed": False, "reason": "insufficient_rows"}
    rng = np.random.default_rng(int(seed) + 17)
    synthetic_rows = []
    for red_probability, blue_probability, _, _ in rows:
        synthetic_red = {int(value) for value in rng.choice(np.arange(1, 34), size=6, replace=False)}
        synthetic_blue = int(rng.integers(1, 17))
        synthetic_rows.append((red_probability, blue_probability, synthetic_red, synthetic_blue))
    score = _score_rows(synthetic_rows)
    statistical = _statistical_decision(rows, synthetic_rows, seed=seed + 19)
    return {
        "executed": True,
        "passed": bool(statistical["status"] != "FAIL"),
        "decision": statistical["status"],
        "statistical_decision": statistical,
        "synthetic_null": score,
        "theoretical": {"red_top6_hits": 36.0 / 33.0, "blue_top1_hit": 1.0 / 16.0},
        "metric": "real predictions evaluated against synthetic random draws",
    }


def _time_reversal(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]]) -> dict[str, Any]:
    if len(rows) < 20:
        return {"executed": True, "passed": False, "reason": "insufficient_rows"}
    reversed_rows = list(reversed(rows))
    first = _score_rows(reversed_rows[: len(reversed_rows) // 2])
    second = _score_rows(reversed_rows[len(reversed_rows) // 2 :])
    gap = abs(float(first["red_top6_hits"]) - float(second["red_top6_hits"]))
    return {
        "executed": True,
        "passed": bool(gap <= 1.25),
        "forward": _score_rows(rows),
        "reversed_first_half": first,
        "reversed_second_half": second,
        "red_half_gap": float(gap),
        "metric": "reverse chronological split must not expose extreme instability",
    }


def _seed_stability(rows: list[tuple[np.ndarray, np.ndarray, set[int], int]], seed: int) -> dict[str, Any]:
    if len(rows) < 8:
        return {"executed": True, "passed": False, "reason": "insufficient_rows"}
    red_scores: list[float] = []
    blue_scores: list[float] = []
    runs = 0
    for offset in range(8):
        retrained_rows, refits = _oof_retrained_rows(rows, seed=int(seed) + 810_000 + offset * 997, mode="seeded")
        runs += refits
        score = _score_rows(retrained_rows)
        red_scores.append(score["red_top6_hits"])
        blue_scores.append(score["blue_top1_hit"])
    return {
        "executed": True,
        "passed": bool(float(np.std(red_scores)) <= 0.45 and float(np.std(blue_scores)) <= 0.12),
        "seeds": 8,
        "oof_refits": runs,
        "red_top6_hits_mean": float(np.mean(red_scores)),
        "red_top6_hits_std": float(np.std(red_scores)),
        "blue_top1_hit_mean": float(np.mean(blue_scores)),
        "blue_top1_hit_std": float(np.std(blue_scores)),
        "metric": "seed stability retrains the OOF proxy model for each seed",
    }


def falsification_report(backtest: dict[str, Any]) -> dict[str, Any]:
    records = backtest.get("period_records", []) if isinstance(backtest, dict) else []
    rows = _rows(records)
    seed = int(((backtest.get("config_snapshot") or {}).get("project") or {}).get("random_seed", 2026))
    suspicious = []
    for method, metrics in (backtest.get("summary", {}) or {}).items():
        sig = metrics.get("significance") or {}
        if sig.get("significant_after_holm_0_05") and float(sig.get("mean_difference", 0.0) or 0.0) < 0:
            suspicious.append({"method": method, "reason": "negative_edge_marked_significant"})
    tests = {
        "label_permutation": _label_permutation(rows),
        "random_feature": _random_feature(rows, seed),
        "synthetic_null": _synthetic_null(rows, seed),
        "time_reversal": _time_reversal(rows),
        "seed_stability": _seed_stability(rows, seed),
    }
    leakage_ok = bool(backtest.get("strict_left_truncation"))
    all_passed = bool(leakage_ok and not suspicious and all(item.get("passed") for item in tests.values()))
    return {
        "schema": "falsification_report_v7",
        "periods": len(records),
        "usable_oof_rows": len(rows),
        "checks": {
            "strict_left_truncation": leakage_ok,
            "no_negative_significant_edge": not suspicious,
            "all_falsification_tests_passed": all_passed,
        },
        "tests": tests,
        "suspicious_findings": suspicious,
        "recommended_shadow_tests": [],
    }
