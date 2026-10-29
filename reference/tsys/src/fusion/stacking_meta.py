from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fusion.model_fusion import MODEL_NAMES


def _normalize(values: np.ndarray, expected_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != int(expected_size) or not np.all(np.isfinite(array)):
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0:
        return np.full(int(expected_size), 1.0 / float(expected_size), dtype=float)
    return array / total


def _record_probabilities(record: dict[str, Any]) -> dict[str, dict[str, np.ndarray]] | None:
    raw = record.get("model_oof_probabilities")
    if not isinstance(raw, dict):
        return None
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in MODEL_NAMES:
        item = raw.get(name)
        if not isinstance(item, dict):
            return None
        result[name] = {
            "red": _normalize(np.asarray(item.get("red", []), dtype=float), 33),
            "blue": _normalize(np.asarray(item.get("blue", []), dtype=float), 16),
        }
    return result


def serialize_model_probabilities(
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "red": _normalize(predictions[name]["red"], 33).tolist(),
            "blue": _normalize(predictions[name]["blue"], 16).tolist(),
        }
        for name in MODEL_NAMES
        if name in predictions
    }


def _red_training(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for record in records:
        probabilities = _record_probabilities(record)
        actual = record.get("actual", {})
        actual_red = {int(number) for number in actual.get("red", [])}
        if probabilities is None or len(actual_red) != 6:
            continue
        for number in range(1, 34):
            rows.append([
                float(probabilities[name]["red"][number - 1])
                for name in MODEL_NAMES
            ])
            labels.append(1 if number in actual_red else 0)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def _blue_training(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for record in records:
        probabilities = _record_probabilities(record)
        actual = record.get("actual", {})
        actual_blue = actual.get("blue")
        if probabilities is None or actual_blue is None:
            continue
        model_arrays = [
            _normalize(probabilities[name]["blue"], 16)
            for name in MODEL_NAMES
        ]
        stacked = np.stack(model_arrays, axis=0)
        mean_probability = np.mean(stacked, axis=0)
        std_probability = np.std(stacked, axis=0)
        consensus_rank = np.argsort(np.argsort(-mean_probability)) + 1
        entropy = float(-np.sum(mean_probability * np.log(np.clip(mean_probability, 1e-12, 1.0))))
        for number in range(1, 17):
            idx = number - 1
            rows.append([
                *[float(array[idx]) for array in model_arrays],
                float(mean_probability[idx]),
                float(std_probability[idx]),
                float(consensus_rank[idx] / 16.0),
                float(entropy),
            ])
            labels.append(1 if number == int(actual_blue) else 0)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def _current_red_features(predictions: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    return np.asarray([
        [
            float(_normalize(predictions[name]["red"], 33)[number - 1])
            for name in MODEL_NAMES
        ]
        for number in range(1, 34)
    ], dtype=float)


def _current_blue_features(predictions: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    model_arrays = [
        _normalize(predictions[name]["blue"], 16)
        for name in MODEL_NAMES
    ]
    stacked = np.stack(model_arrays, axis=0)
    mean_probability = np.mean(stacked, axis=0)
    std_probability = np.std(stacked, axis=0)
    consensus_rank = np.argsort(np.argsort(-mean_probability)) + 1
    entropy = float(-np.sum(mean_probability * np.log(np.clip(mean_probability, 1e-12, 1.0))))
    return np.asarray([
        [
            *[float(array[number - 1]) for array in model_arrays],
            float(mean_probability[number - 1]),
            float(std_probability[number - 1]),
            float(consensus_rank[number - 1] / 16.0),
            float(entropy),
        ]
        for number in range(1, 17)
    ], dtype=float)


def _record_red_features(record: dict[str, Any]) -> np.ndarray | None:
    probabilities = _record_probabilities(record)
    if probabilities is None:
        return None
    return np.asarray([
        [
            float(probabilities[name]["red"][number - 1])
            for name in MODEL_NAMES
        ]
        for number in range(1, 34)
    ], dtype=float)


def _record_blue_features(record: dict[str, Any]) -> np.ndarray | None:
    probabilities = _record_probabilities(record)
    if probabilities is None:
        return None
    model_arrays = [
        _normalize(probabilities[name]["blue"], 16)
        for name in MODEL_NAMES
    ]
    stacked = np.stack(model_arrays, axis=0)
    mean_probability = np.mean(stacked, axis=0)
    std_probability = np.std(stacked, axis=0)
    consensus_rank = np.argsort(np.argsort(-mean_probability)) + 1
    entropy = float(-np.sum(mean_probability * np.log(np.clip(mean_probability, 1e-12, 1.0))))
    return np.asarray([
        [
            *[float(array[number - 1]) for array in model_arrays],
            float(mean_probability[number - 1]),
            float(std_probability[number - 1]),
            float(consensus_rank[number - 1] / 16.0),
            float(entropy),
        ]
        for number in range(1, 17)
    ], dtype=float)


def _base_record_prediction(record: dict[str, Any], *, color: str, expected_size: int) -> np.ndarray | None:
    probabilities = _record_probabilities(record)
    if probabilities is None:
        return None
    weight_map = (record.get("weights_by_color") or {}).get(color)
    arrays: list[np.ndarray] = []
    weights: list[float] = []
    if isinstance(weight_map, dict):
        for name in MODEL_NAMES:
            if name not in probabilities:
                continue
            weight = float(weight_map.get(name, 0.0) or 0.0)
            if weight <= 0:
                continue
            arrays.append(_normalize(probabilities[name][color], expected_size))
            weights.append(weight)
    if arrays and sum(weights) > 0:
        weighted = np.average(np.stack(arrays, axis=0), axis=0, weights=np.asarray(weights, dtype=float))
        return _normalize(weighted, expected_size)
    return _normalize(
        np.mean(
            np.stack([
                _normalize(probabilities[name][color], expected_size)
                for name in MODEL_NAMES
            ], axis=0),
            axis=0,
        ),
        expected_size,
    )


def _red_meta_model(stacking: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=float(stacking.get("red_c", 0.45)),
            max_iter=int(stacking.get("max_iter", 220)),
            class_weight="balanced",
            random_state=int(stacking.get("random_seed", 2026)),
        )),
    ])


def _blue_meta_model(stacking: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0 / max(float(stacking.get("blue_ridge_alpha", 10.0)), 1e-6),
            max_iter=int(stacking.get("max_iter", 220)),
            class_weight="balanced",
            random_state=int(stacking.get("random_seed", 2026)),
        )),
    ])


def _runtime_oof_gate(
    usable_records: list[dict[str, Any]],
    stacking: dict[str, Any],
    *,
    color: str,
) -> dict[str, Any]:
    """Run a genuine temporal OOF gate before current-period stacking is used.

    The meta model is trained on earlier completed OOF rows and evaluated on a
    later completed slice.  The current target period is never part of this
    gate.  This makes the gate operational instead of a report-only summary.
    """

    expected_size = 33 if color == "red" else 16
    minimum_gate_records = max(20, int(stacking.get("min_gate_records", 60)))
    configured_gate_records = int(stacking.get("gate_validation_records", 100))
    gate_size = min(max(minimum_gate_records, configured_gate_records), max(0, len(usable_records) // 3))
    train_records = usable_records[:-gate_size] if gate_size else []
    gate_records = usable_records[-gate_size:] if gate_size else []
    if len(train_records) < minimum_gate_records or len(gate_records) < minimum_gate_records:
        return {
            "passed": False,
            "reason": "insufficient_runtime_oof_gate_records",
            "train_records": len(train_records),
            "gate_records": len(gate_records),
            "minimum_gate_records": minimum_gate_records,
        }

    try:
        if color == "red":
            train_x, train_y = _red_training(train_records)
            if train_x.shape[0] < len(train_records) * 20 or len(np.unique(train_y)) < 2:
                raise ValueError("红球运行时OOF门控样本不足")
            model = _red_meta_model(stacking)
            model.fit(train_x, train_y)
            base_hits: list[int] = []
            stack_hits: list[int] = []
            base_recall10: list[float] = []
            stack_recall10: list[float] = []
            for record in gate_records:
                actual_red = {int(number) for number in (record.get("actual") or {}).get("red", [])}
                if len(actual_red) != 6:
                    continue
                features = _record_red_features(record)
                base = _base_record_prediction(record, color="red", expected_size=expected_size)
                if features is None or base is None:
                    continue
                stacked = _normalize(model.predict_proba(features)[:, 1], expected_size)
                base_order = [int(index) + 1 for index in np.argsort(-base, kind="stable")]
                stack_order = [int(index) + 1 for index in np.argsort(-stacked, kind="stable")]
                base_hits.append(len(set(base_order[:6]) & actual_red))
                stack_hits.append(len(set(stack_order[:6]) & actual_red))
                base_recall10.append(len(set(base_order[:10]) & actual_red) / 6.0)
                stack_recall10.append(len(set(stack_order[:10]) & actual_red) / 6.0)
            base_avg = float(np.mean(base_hits)) if base_hits else None
            stack_avg = float(np.mean(stack_hits)) if stack_hits else None
            base_r10 = float(np.mean(base_recall10)) if base_recall10 else None
            stack_r10 = float(np.mean(stack_recall10)) if stack_recall10 else None
            min_delta = float(stacking.get("red_gate_min_top6_delta", 0.0))
            passed = bool(
                base_avg is not None
                and stack_avg is not None
                and stack_avg >= base_avg + min_delta
                and (base_r10 is None or stack_r10 is None or stack_r10 >= base_r10)
            )
            return {
                "passed": passed,
                "reason": "runtime_temporal_oof_base_vs_stacking_gate",
                "train_records": len(train_records),
                "gate_records": len(base_hits),
                "base_average_top6_hits": base_avg,
                "stacking_average_top6_hits": stack_avg,
                "base_recall_at_10": base_r10,
                "stacking_recall_at_10": stack_r10,
                "minimum_top6_delta": min_delta,
            }

        train_x, train_y = _blue_training(train_records)
        if train_x.shape[0] < len(train_records) * 16 or len(np.unique(train_y)) < 2:
            raise ValueError("蓝球运行时OOF门控样本不足")
        model = _blue_meta_model(stacking)
        model.fit(train_x, train_y)
        base_ranks: list[int] = []
        stack_ranks: list[int] = []
        base_logloss: list[float] = []
        stack_logloss: list[float] = []
        for record in gate_records:
            actual_blue = (record.get("actual") or {}).get("blue")
            if actual_blue is None:
                continue
            actual = int(actual_blue)
            features = _record_blue_features(record)
            base = _base_record_prediction(record, color="blue", expected_size=expected_size)
            if features is None or base is None:
                continue
            stacked = _normalize(model.predict_proba(features)[:, 1], expected_size)
            base_order = [int(index) + 1 for index in np.argsort(-base, kind="stable")]
            stack_order = [int(index) + 1 for index in np.argsort(-stacked, kind="stable")]
            base_ranks.append(int(base_order.index(actual) + 1))
            stack_ranks.append(int(stack_order.index(actual) + 1))
            base_logloss.append(float(-np.log(np.clip(base[actual - 1], 1e-12, 1.0))))
            stack_logloss.append(float(-np.log(np.clip(stacked[actual - 1], 1e-12, 1.0))))
        base_hit5 = float(np.mean([rank <= 5 for rank in base_ranks])) if base_ranks else None
        stack_hit5 = float(np.mean([rank <= 5 for rank in stack_ranks])) if stack_ranks else None
        base_mrr = float(np.mean([1.0 / rank for rank in base_ranks])) if base_ranks else None
        stack_mrr = float(np.mean([1.0 / rank for rank in stack_ranks])) if stack_ranks else None
        base_loss = float(np.mean(base_logloss)) if base_logloss else None
        stack_loss = float(np.mean(stack_logloss)) if stack_logloss else None
        min_hit5_delta = float(stacking.get("blue_gate_min_hit5_delta", 0.0))
        passed = bool(
            base_hit5 is not None
            and stack_hit5 is not None
            and stack_hit5 >= base_hit5 + min_hit5_delta
            and (base_mrr is None or stack_mrr is None or stack_mrr >= base_mrr)
            and (base_loss is None or stack_loss is None or stack_loss <= base_loss)
        )
        return {
            "passed": passed,
            "reason": "runtime_temporal_oof_base_vs_stacking_gate",
            "train_records": len(train_records),
            "gate_records": len(base_ranks),
            "base_hit_at_5": base_hit5,
            "stacking_hit_at_5": stack_hit5,
            "base_mrr": base_mrr,
            "stacking_mrr": stack_mrr,
            "base_logloss": base_loss,
            "stacking_logloss": stack_loss,
            "minimum_hit5_delta": min_hit5_delta,
        }
    except Exception as exc:
        return {
            "passed": False,
            "reason": "runtime_oof_gate_failed",
            "error": str(exc),
            "train_records": len(train_records),
            "gate_records": len(gate_records),
        }


def _blue_gate_metrics(usable_records: list[dict[str, Any]], blue_probability: np.ndarray) -> dict[str, Any]:
    """Lightweight sanity gate for whether blue stacking is informative enough."""
    actual_values = [
        int((record.get("actual") or {}).get("blue"))
        for record in usable_records
        if (record.get("actual") or {}).get("blue") is not None
    ]
    if not actual_values:
        return {"passed": False, "reason": "no_blue_actuals"}
    base_hit5 = []
    base_mrr = []
    for record in usable_records[-min(len(usable_records), 300):]:
        probabilities = _record_probabilities(record)
        actual = (record.get("actual") or {}).get("blue")
        if probabilities is None or actual is None:
            continue
        base = _normalize(
            np.mean(np.stack([probabilities[name]["blue"] for name in MODEL_NAMES], axis=0), axis=0),
            16,
        )
        rank = int((np.argsort(base)[::-1] + 1).tolist().index(int(actual)) + 1)
        base_hit5.append(float(rank <= 5))
        base_mrr.append(float(1.0 / rank))
    current_signal = float(np.sum(np.abs(_normalize(blue_probability, 16) - 1.0 / 16.0)) / 2.0)
    return {
        "passed": bool(current_signal >= 0.015 and len(base_hit5) >= 60),
        "reason": "candidate_level_blue_stacking_signal_gate",
        "recent_base_hit_at_5": float(np.mean(base_hit5)) if base_hit5 else None,
        "recent_base_mrr": float(np.mean(base_mrr)) if base_mrr else None,
        "current_signal_strength": current_signal,
    }


def _red_gate_metrics(usable_records: list[dict[str, Any]], red_probability: np.ndarray) -> dict[str, Any]:
    """Gate red stacking with OOF recall and current distribution strength."""
    recall10: list[float] = []
    recall18: list[float] = []
    for record in usable_records[-min(len(usable_records), 300):]:
        probabilities = _record_probabilities(record)
        actual_red = {int(number) for number in (record.get("actual") or {}).get("red", [])}
        if probabilities is None or len(actual_red) != 6:
            continue
        base = _normalize(
            np.mean(np.stack([probabilities[name]["red"] for name in MODEL_NAMES], axis=0), axis=0),
            33,
        )
        ordered = [int(index) + 1 for index in np.argsort(base)[::-1]]
        recall10.append(float(len(set(ordered[:10]) & actual_red) / 6.0))
        recall18.append(float(len(set(ordered[:18]) & actual_red) / 6.0))
    current_signal = float(np.sum(np.abs(_normalize(red_probability, 33) - 1.0 / 33.0)) / 2.0)
    return {
        "passed": bool(current_signal >= 0.010 and len(recall18) >= 60),
        "reason": "candidate_level_red_stacking_signal_gate",
        "recent_base_recall_at_10": float(np.mean(recall10)) if recall10 else None,
        "recent_base_recall_at_18": float(np.mean(recall18)) if recall18 else None,
        "current_signal_strength": current_signal,
    }


def _mean_current_prediction(
    current_predictions: dict[str, dict[str, np.ndarray]],
    *,
    color: str,
    expected_size: int,
) -> np.ndarray:
    return _normalize(
        np.mean(
            np.stack([
                _normalize(current_predictions[name][color], expected_size)
                for name in MODEL_NAMES
            ], axis=0),
            axis=0,
        ),
        expected_size,
    )


def stacking_meta_predict(
    completed_records: list[dict[str, Any]],
    current_predictions: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Train an OOF stacking meta learner and predict the next distribution.

    The caller must pass only completed walk-forward records.  Each record
    stores first-layer out-of-sample probabilities generated before its actual
    draw was known.  The current period's probabilities are then passed through
    the second layer.
    """
    stacking = config.get("stacking", {})
    if not bool(stacking.get("enabled", False)):
        return None, {"enabled": False, "reason": "stacking_disabled"}
    min_records = int(stacking.get("min_oof_records", 60))
    usable_records = [
        record for record in completed_records
        if _record_probabilities(record) is not None
    ]
    if len(usable_records) < min_records:
        return None, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient_oof_records",
            "usable_oof_records": len(usable_records),
            "min_oof_records": min_records,
        }
    try:
        red_x, red_y = _red_training(usable_records)
        blue_x, blue_y = _blue_training(usable_records)
        if red_x.shape[0] < min_records * 20 or len(np.unique(red_y)) < 2:
            raise ValueError("红球OOF样本不足")
        if blue_x.shape[0] < min_records * 16 or len(np.unique(blue_y)) < 2:
            raise ValueError("蓝球OOF样本不足")

        red_gate = _runtime_oof_gate(usable_records, stacking, color="red")
        blue_gate = _runtime_oof_gate(usable_records, stacking, color="blue")

        red_model = _red_meta_model(stacking)
        red_model.fit(red_x, red_y)
        red_probability = red_model.predict_proba(_current_red_features(current_predictions))[:, 1]
        red_probability = _normalize(red_probability, 33)
        red_applied = bool(
            (not bool(stacking.get("red_require_gate", True)))
            or bool(red_gate.get("passed"))
        )

        blue_model = _blue_meta_model(stacking)
        blue_model.fit(blue_x, blue_y)
        blue_probability = _normalize(
            blue_model.predict_proba(_current_blue_features(current_predictions))[:, 1],
            16,
        )
        blue_applied = bool(
            (not bool(stacking.get("blue_require_gate", True)))
            or bool(blue_gate.get("passed"))
        )
        return {
            "red": red_probability,
            "blue": blue_probability,
        }, {
            "enabled": True,
            "applied": bool(red_applied or blue_applied),
            "type": "out_of_fold_logistic_red_candidate_level_logistic_blue",
            "usable_oof_records": len(usable_records),
            "red_training_rows": int(red_x.shape[0]),
            "blue_training_rows": int(blue_x.shape[0]),
            "red_gate": red_gate,
            "blue_gate": blue_gate,
            "red_applied": red_applied,
            "blue_applied": blue_applied,
        }
    except Exception as exc:
        return None, {
            "enabled": True,
            "applied": False,
            "reason": "stacking_fit_failed",
            "error": str(exc),
            "usable_oof_records": len(usable_records),
        }


def blend_with_stacking(
    base_prediction: dict[str, np.ndarray],
    stacking_prediction: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    stacking_info: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if stacking_prediction is None:
        return base_prediction
    stacking = config.get("stacking", {})
    red_alpha = float(stacking.get("red_blend_alpha", stacking.get("blend_alpha", 0.35)))
    blue_alpha = float(stacking.get("blue_blend_alpha", stacking.get("blend_alpha", 0.35)))
    info = stacking_info or {}
    if info and not bool(info.get("red_applied", False)):
        red_alpha = 0.0
    if info and not bool(info.get("blue_applied", False)):
        blue_alpha = 0.0
    return {
        "red": _normalize(
            (1.0 - red_alpha) * _normalize(base_prediction["red"], 33)
            + red_alpha * _normalize(stacking_prediction["red"], 33),
            33,
        ),
        "blue": _normalize(
            (1.0 - blue_alpha) * _normalize(base_prediction["blue"], 16)
            + blue_alpha * _normalize(stacking_prediction["blue"], 16),
            16,
        ),
    }
