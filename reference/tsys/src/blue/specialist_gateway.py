from __future__ import annotations

import time
from collections import Counter
from datetime import timedelta
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import binomtest

from src.regime.time_regime_detector import regime_summary, weighted_recent_distribution
from src.blue.adaptive_policy import select_adaptive_blue_set

_BLUE_COMBINATION_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
BLUE_SPECIALIST_VARIANTS = {
    "online_hybrid_coverage",
    "online_hedge",
    "nested_hedge",
    "bayes_markov_only",
    "hmm_regime",
    "union_top3",
    "stable_frequency_transition",
    "recent_frequency_transition",
    "high_order_markov",
    "bayes_markov_blend",
    "change_point_periodic",
    "gap_shape_only",
    "periodic_only",
    "change_point_only",
    "frequency_300_only",
    "decay_only",
    "markov_only",
    "gap_only",
}

BLUE_CORE_EXPERTS = {"S1_freq_all", "S2_freq_300", "S3_decay", "S4_gap"}
BLUE_CHALLENGER_EXPERTS = {
    "E5_markov1",
    "E6_bayes_markov",
    "E8_weekday",
    "E9_month",
    "E10_markov2",
    "E11_markov3",
    "E12_change_point",
    "E14_periodic_phase",
}
BLUE_RESEARCH_EXPERTS = {
    "E13_gap_shape",
    "E15_red_structure_blue",
    "E16_blue_state_red_shift",
    "hmm_regime",
    "change_point_periodic",
}

DEFAULT_UPGRADE_CONFIG = {
    "blue_mode": "simplified_guard",
    "expert_pool_extend": True,
    "ada_hedge_enable": True,
    "hmm_regime_full_window": True,
    "nested_hedge_enable": True,
    "cross_red_blue_feature": True,
    "coverage_param_online_tune": True,
    "runtime_target_minutes": 30,
    # 方案2.5：蓝球专家系统彻底简化
    # 开启后仅保留4个核心专家（频率全历史/300期、时间衰减、遗漏风险），
    # 移除HMM、高阶马尔可夫、变点检测、周期相位、红蓝交叉特征，
    # Hedge使用固定学习率0.05（更稳定，不过度反应噪声）
    "simplified_blue_experts": False,
    "simplified_hedge_eta": 0.05,
    "simplified_blue_temperature": 1.0,
    "validated_blue_consensus": True,
    "blue_coverage_policy": "rotating_buckets",
    "production_specialists": "core_only",
    "calendar_features_enabled": True,
    "blue_expert_pruning_enabled": True,
    "blue_expert_pruning_loss_margin": 0.01,
    "blue_evidence_shrinkage_enabled": True,
    "blue_no_edge_model_alpha": 0.08,
    "blue_weak_edge_model_alpha": 0.22,
    "blue_medium_edge_model_alpha": 0.45,
    "blue_strong_edge_model_alpha": 0.70,
    "blue_coverage_candidate_top_n": 10,
    "blue_coverage_max_rank_penalty": 0.04,
    "blue_no_edge_strict_uniform_coverage": True,
}


def _upgrade_options(upgrade_config: dict[str, Any] | None = None) -> dict[str, Any]:
    options = {**DEFAULT_UPGRADE_CONFIG, **(upgrade_config or {})}
    mode = str(options.get("blue_mode", "simplified_guard")).strip().lower()
    if mode not in {"research_full", "simplified_guard", "uniform_coverage"}:
        raise ValueError("upgrade.blue_mode 仅支持 research_full、simplified_guard、uniform_coverage")
    options["blue_mode"] = mode
    if mode == "research_full":
        options["simplified_blue_experts"] = False
    elif mode in {"simplified_guard", "uniform_coverage"}:
        options["simplified_blue_experts"] = True
        options["expert_pool_extend"] = False
        options["hmm_regime_full_window"] = False
        options["nested_hedge_enable"] = False
    return options


def _normalize(values: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.sum(scores))
    if total <= 0:
        return np.full(16, 1.0 / 16.0)
    return scores / total


def _blue_array(history: pd.DataFrame | Iterable[Any]) -> np.ndarray:
    if isinstance(history, pd.DataFrame):
        values = history["blue"].astype(int).to_numpy()
    elif isinstance(history, np.ndarray):
        values = history.astype(int, copy=False)
    else:
        values = np.asarray([int(value) for value in history], dtype=int)
    return values[(1 <= values) & (values <= 16)]


def _date_array(history: pd.DataFrame | Iterable[Any]) -> pd.Series | None:
    if not isinstance(history, pd.DataFrame) or "date" not in history.columns:
        return None
    values = pd.to_datetime(history["date"], errors="coerce").reset_index(drop=True)
    if values.isna().any():
        return None
    return values


def _slice_history(history: pd.DataFrame | Iterable[Any], stop: int) -> pd.DataFrame | np.ndarray:
    if isinstance(history, pd.DataFrame):
        return history.iloc[: int(stop)]
    values = _blue_array(history)
    return values[: int(stop)]


def _next_draw_date(last_date: pd.Timestamp) -> pd.Timestamp:
    weekdays = {1, 3, 6}
    current = pd.Timestamp(last_date).normalize()
    for offset in range(1, 8):
        candidate = current + timedelta(days=offset)
        if int(candidate.weekday()) in weekdays:
            return candidate
    return current + timedelta(days=2)


def _order(probabilities: np.ndarray) -> list[int]:
    return [int(value) for value in (np.argsort(probabilities)[::-1] + 1)]


def _blue_ranking_detail(probability: np.ndarray, actual_blue: int, *, top_k: int = 5) -> dict[str, float | int]:
    probability = _normalize(probability)
    actual = int(actual_blue)
    if not 1 <= actual <= 16:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
            "mean_rank": 16.0,
            "rank": 16,
        }
    ordered = _order(probability)
    rank = int(ordered.index(actual) + 1)
    top_k = max(1, min(int(top_k), 16))
    return {
        "hit_at_1": float(rank <= 1),
        "hit_at_3": float(rank <= 3),
        "hit_at_5": float(rank <= 5),
        "mrr": float(1.0 / rank),
        "ndcg_at_5": float((1.0 / np.log2(rank + 1.0)) if rank <= top_k else 0.0),
        "mean_rank": float(rank),
        "rank": rank,
    }


def _frequency(history: np.ndarray, window: int | None = None) -> np.ndarray:
    values = history[-int(window):] if window else history
    return _normalize(np.bincount(values - 1, minlength=16) + 1.0)


def _time_decay(history: np.ndarray, decay: float = 0.985) -> np.ndarray:
    scores = np.ones(16, dtype=float)
    weights = float(decay) ** np.arange(len(history) - 1, -1, -1)
    for blue, weight in zip(history, weights):
        scores[int(blue) - 1] += float(weight)
    return _normalize(scores)


def _markov(history: np.ndarray) -> np.ndarray:
    if len(history) < 2:
        return np.full(16, 1.0 / 16.0)
    last_blue = int(history[-1])
    scores = np.ones(16, dtype=float)
    for previous, current in zip(history[:-1], history[1:]):
        if int(previous) == last_blue:
            scores[int(current) - 1] += 1.0
    return _normalize(scores)


def _transition_counts(history: np.ndarray) -> np.ndarray:
    counts = np.zeros((16, 16), dtype=float)
    if len(history) < 2:
        return counts
    previous = history[:-1].astype(int) - 1
    current = history[1:].astype(int) - 1
    np.add.at(counts, (previous, current), 1.0)
    return counts


def _dirichlet_multinomial_log_marginal(row: np.ndarray, alpha: np.ndarray) -> float:
    total = float(row.sum())
    alpha_total = float(alpha.sum())
    return float(
        gammaln(alpha_total)
        - gammaln(alpha_total + total)
        + np.sum(gammaln(alpha + row) - gammaln(alpha))
    )


def _estimate_shrinkage_kappa(counts: np.ndarray, pi: np.ndarray) -> float:
    row_totals = counts.sum(axis=1)
    if float(row_totals.sum()) <= 0:
        return 128.0
    candidates = np.unique(np.concatenate([
        np.geomspace(0.10, 600.0, 18),
        np.array([1.0, 4.0, 16.0, 64.0, 256.0]),
    ]))
    best_kappa = 128.0
    best_score = -np.inf
    safe_pi = _normalize(pi + 1e-9)
    for kappa in candidates:
        alpha = np.maximum(float(kappa) * safe_pi, 1e-9)
        score = 0.0
        for row in counts:
            if float(row.sum()) > 0:
                score += _dirichlet_multinomial_log_marginal(row, alpha)
        if score > best_score:
            best_score = score
            best_kappa = float(kappa)
    return best_kappa


def _bayesian_shrinkage_markov(history: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Empirical-Bayes hierarchical shrinkage Markov prediction.

    This is the fast online version of the hierarchical model:
    global pi is estimated from all blue observations; kappa is selected by
    marginal likelihood; each previous-state row shrinks toward pi.
    """
    if len(history) < 2:
        probability = _frequency(history, None) if len(history) else np.full(16, 1.0 / 16.0)
        return probability, {"kappa": None, "mode": "insufficient_history"}
    if len(history) > 900:
        history = history[-900:]
    counts = _transition_counts(history)
    pi = _frequency(history, None)
    kappa = _estimate_shrinkage_kappa(counts, pi)
    last_state = int(history[-1]) - 1
    row = counts[last_state]
    probability = _normalize(row + float(kappa) * pi)
    return probability, {
        "kappa": float(kappa),
        "last_state": int(last_state + 1),
        "row_count": int(row.sum()),
        "mode": "empirical_bayes_shrinkage_markov",
    }


def _markov_order(history: np.ndarray, order: int) -> np.ndarray:
    order = max(1, int(order))
    if len(history) <= order:
        return np.full(16, 1.0 / 16.0)
    if len(history) > 800:
        history = history[-800:]
    context = tuple(int(value) for value in history[-order:])
    scores = np.ones(16, dtype=float)
    if order == 1:
        mask = history[:-1] == context[0]
        followers = history[1:][mask]
    elif order == 2:
        mask = (history[:-2] == context[0]) & (history[1:-1] == context[1])
        followers = history[2:][mask]
    else:
        mask = (
            (history[:-3] == context[0])
            & (history[1:-2] == context[1])
            & (history[2:-1] == context[2])
        )
        followers = history[3:][mask]
    if len(followers):
        scores += np.bincount(followers - 1, minlength=16)
    return _normalize(scores)


def _variable_markov(history: np.ndarray) -> np.ndarray:
    if len(history) < 2:
        return np.full(16, 1.0 / 16.0)
    if len(history) > 800:
        history = history[-800:]
    parts = [_frequency(history, None)]
    weights = [0.20]
    for order in (1, 2, 3):
        if len(history) <= order:
            continue
        context = tuple(int(value) for value in history[-order:])
        if order == 1:
            context_count = int(np.sum(history[:-1] == context[0]))
        elif order == 2:
            context_count = int(np.sum((history[:-2] == context[0]) & (history[1:-1] == context[1])))
        else:
            context_count = int(np.sum(
                (history[:-3] == context[0])
                & (history[1:-2] == context[1])
                & (history[2:-1] == context[2])
            ))
        reliability = context_count / (context_count + 8.0 + 4.0 * order)
        parts.append(_markov_order(history, order))
        weights.append(max(0.08, reliability))
    weight_array = np.asarray(weights, dtype=float)
    weight_array /= max(float(weight_array.sum()), 1e-12)
    return _normalize(np.sum(np.stack(parts, axis=0) * weight_array[:, None], axis=0))


def _conditional_date_frequency(
    history: pd.DataFrame | Iterable[Any],
    *,
    mode: str,
) -> np.ndarray:
    blue_history = _blue_array(history)
    dates = _date_array(history)
    if len(blue_history) == 0 or dates is None or len(dates) != len(blue_history):
        return _frequency(blue_history, None) if len(blue_history) else np.full(16, 1.0 / 16.0)
    target_date = _next_draw_date(pd.Timestamp(dates.iloc[-1]))
    if mode == "weekday":
        mask = dates.dt.weekday.to_numpy(dtype=int) == int(target_date.weekday())
    elif mode == "month":
        mask = dates.dt.month.to_numpy(dtype=int) == int(target_date.month)
    else:
        raise ValueError("mode 必须为 weekday 或 month")
    matched = blue_history[mask]
    if len(matched) < 12:
        return _normalize(0.35 * _frequency(matched, None) + 0.65 * _frequency(blue_history, None)) if len(matched) else _frequency(blue_history, None)
    return _normalize(0.72 * _frequency(matched, None) + 0.28 * _frequency(blue_history, None))


def _third_order_markov(history: np.ndarray) -> np.ndarray:
    return _markov_order(history, 3)


def _change_point_adaptive_frequency(history: np.ndarray) -> np.ndarray:
    if len(history) < 180:
        return _frequency(history, min(len(history), 300))
    summary = regime_summary(history, size=16)
    return _normalize(summary["weighted_probability"])


def _red_structure_conditioned_blue(history: pd.DataFrame | Iterable[Any]) -> np.ndarray:
    blue_history = _blue_array(history)
    if not isinstance(history, pd.DataFrame) or len(history) < 120:
        return _frequency(blue_history, 300) if len(blue_history) else np.full(16, 1.0 / 16.0)
    red_columns = [f"r{index}" for index in range(1, 7)]
    if not all(column in history.columns for column in red_columns):
        return _frequency(blue_history, 300)
    reds = history[red_columns].astype(int)
    red_sum = reds.sum(axis=1).to_numpy(dtype=float)
    red_span = (reds.max(axis=1) - reds.min(axis=1)).to_numpy(dtype=float)
    odd_count = (reds % 2).sum(axis=1).to_numpy(dtype=int)
    latest = reds.iloc[-1].to_numpy(dtype=int)
    latest_sum = float(latest.sum())
    latest_span = float(latest.max() - latest.min())
    latest_odd = int(np.sum(latest % 2))
    mask = (
        (np.abs(red_sum - latest_sum) <= 12.0)
        & (np.abs(red_span - latest_span) <= 4.0)
        & (np.abs(odd_count - latest_odd) <= 1)
    )
    matched = blue_history[mask]
    global_probability = _frequency(blue_history, 500)
    if len(matched) < 18:
        return _normalize(0.25 * _frequency(matched, None) + 0.75 * global_probability) if len(matched) else global_probability
    return _normalize(0.58 * _frequency(matched, None) + 0.42 * global_probability)


def _blue_state_red_shift(history: pd.DataFrame | Iterable[Any]) -> np.ndarray:
    blue_history = _blue_array(history)
    if not isinstance(history, pd.DataFrame) or len(history) < 150:
        return _frequency(blue_history, 300) if len(blue_history) else np.full(16, 1.0 / 16.0)
    red_columns = [f"r{index}" for index in range(1, 7)]
    if not all(column in history.columns for column in red_columns):
        return _frequency(blue_history, 300)
    latest_blue = int(blue_history[-1])
    blues = blue_history
    reds = history[red_columns].astype(int).to_numpy(dtype=int)
    shifted_scores = np.ones(16, dtype=float)
    global_probability = _frequency(blue_history, 500)
    for index in range(1, len(blues)):
        prev_blue = int(blues[index - 1])
        if abs(prev_blue - latest_blue) <= 1 or prev_blue == latest_blue:
            red_overlap = len(set(reds[-1]) & set(reds[index - 1])) / 6.0
            shifted_scores[int(blues[index]) - 1] += 0.45 + 0.65 * red_overlap
    return _normalize(0.55 * _normalize(shifted_scores) + 0.45 * global_probability)


def _base_blue_experts(
    history: pd.DataFrame | Iterable[Any],
    upgrade_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    upgrade = _upgrade_options(upgrade_config)
    blue_history = _blue_array(history)
    if len(blue_history) == 0:
        uniform = np.full(16, 1.0 / 16.0)
        return {"E0_uniform": uniform}

    # V7 production gate: production uses only Core experts. Challenger and
    # Research experts may appear in shadow evidence, but cannot directly vote
    # in the production blue portfolio unless this file is explicitly run in a
    # research-only mode.
    production_specialists = str(upgrade.get("production_specialists", "core_only")).strip().lower()
    if bool(upgrade.get("simplified_blue_experts", False)) or production_specialists == "core_only":
        experts = {
            "S1_freq_all": _frequency(blue_history, None),
            "S2_freq_300": _frequency(blue_history, 300),
            "S3_decay": _time_decay(blue_history),
            "S4_gap": _gap_hazard(blue_history),
        }
        return experts

    bayes_probability, _ = _bayesian_shrinkage_markov(blue_history)
    experts = {
        "E0_uniform": np.full(16, 1.0 / 16.0),
        "E1_freq_all": _frequency(blue_history, None),
        "E2_freq_300": _frequency(blue_history, 300),
        "E3_freq_100": _frequency(blue_history, 100),
        "E4_decay": _time_decay(blue_history),
        "E5_markov1": _markov(blue_history),
        "E6_bayes_markov": bayes_probability,
        "E7_gap": _gap_hazard(blue_history),
    }
    if bool(upgrade.get("calendar_features_enabled", False)):
        experts["E8_weekday"] = _conditional_date_frequency(history, mode="weekday")
        experts["E9_month"] = _conditional_date_frequency(history, mode="month")
    if bool(upgrade.get("expert_pool_extend", True)):
        experts.update({
            "E10_markov2": _markov_order(blue_history, 2),
            "E11_markov3": _third_order_markov(blue_history),
            "E12_change_point": _change_point_adaptive_frequency(blue_history),
            "E13_gap_shape": _gap_shape(blue_history),
            "E14_periodic_phase": _periodic_phase_mixture(blue_history),
        })
        if bool(upgrade.get("cross_red_blue_feature", True)):
            experts["E15_red_structure_blue"] = _red_structure_conditioned_blue(history)
            experts["E16_blue_state_red_shift"] = _blue_state_red_shift(history)
    return experts


def _online_hedge_mixture(
    history: pd.DataFrame | Iterable[Any],
    *,
    max_replay: int = 180,
    warmup: int = 60,
    eta0: float = 0.16,
    upgrade_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    upgrade = _upgrade_options(upgrade_config)
    blue_history = _blue_array(history)
    if len(blue_history) <= max(5, int(warmup)):
        experts = _base_blue_experts(history, upgrade)
        names = list(experts)
        return _normalize(np.mean(np.stack(list(experts.values()), axis=0), axis=0)), {
            "mode": "hedge_bootstrap",
            "weights": {name: 1.0 / len(names) for name in names},
            "warmup": int(warmup),
        }

    expert_names = list(_base_blue_experts(
        blue_history[:max(2, min(len(blue_history), int(warmup)))],
        upgrade,
    ).keys())
    weights = np.full(len(expert_names), 1.0 / len(expert_names), dtype=float)
    start = max(int(warmup), len(blue_history) - int(max_replay))
    update_count = 0
    cumulative_losses = np.zeros(len(expert_names), dtype=float)
    cumulative_mix_loss = 0.0
    eta_trace: list[float] = []
    for target_index in range(start, len(blue_history)):
        prefix = blue_history[:target_index]
        experts = _base_blue_experts(prefix, upgrade)
        actual = int(blue_history[target_index]) - 1
        probabilities = np.stack([experts[name] for name in expert_names], axis=0)
        losses = -np.log(np.clip(probabilities[:, actual], 1e-12, 1.0))
        update_count += 1
        # 方案2.5：简化模式使用固定学习率，跳过AdaHedge自适应
        if bool(upgrade.get("simplified_blue_experts", False)):
            eta = float(upgrade.get("simplified_hedge_eta", 0.05))
            cumulative_losses += losses
        elif bool(upgrade.get("ada_hedge_enable", True)):
            mix_probability = float(np.clip(weights @ probabilities[:, actual], 1e-12, 1.0))
            cumulative_mix_loss += -np.log(mix_probability)
            cumulative_losses += losses
            best_loss = float(np.min(cumulative_losses))
            regret = max(0.0, cumulative_mix_loss - best_loss)
            variance = float(np.var(np.clip(losses, 0.0, 30.0)))
            eta = float(np.clip(np.sqrt(max(regret, 1e-9) / max(variance * update_count, 1e-9)), 0.02, 0.5))
        else:
            eta = float(eta0) / np.sqrt(max(update_count, 1))
            cumulative_losses += losses
        eta_trace.append(float(eta))
        weights *= np.exp(-eta * np.clip(losses, 0.0, 30.0))
        weights = _normalize(weights)

    current_experts = _base_blue_experts(history, upgrade)
    current_probabilities = np.stack([current_experts[name] for name in expert_names], axis=0)
    mean_loss_by_name = {
        name: float(cumulative_losses[index] / max(update_count, 1))
        for index, name in enumerate(expert_names)
    }
    inactive_pool: list[str] = []
    if bool(upgrade.get("blue_expert_pruning_enabled", True)) and update_count > 0 and len(expert_names) > 1:
        uniform_loss = float(np.log(16.0))
        margin = float(upgrade.get("blue_expert_pruning_loss_margin", 0.01))
        mean_losses = cumulative_losses / max(update_count, 1)
        inactive_indexes = [
            index for index, value in enumerate(mean_losses)
            if float(value) > uniform_loss + margin and float(weights[index]) < 1.0 / len(expert_names)
        ]
        if 0 < len(inactive_indexes) < len(expert_names):
            inactive_set = set(inactive_indexes)
            inactive_pool = [expert_names[index] for index in inactive_indexes]
            active_indexes = [index for index in range(len(expert_names)) if index not in inactive_set]
            weights = weights[active_indexes]
            weights = _normalize(weights)
            current_probabilities = current_probabilities[active_indexes]
            expert_names = [expert_names[index] for index in active_indexes]
    mixed = _normalize(weights @ current_probabilities)
    # 方案2.5优化：蓝球温度调节，使首选分布更尖锐，提升首选命中率
    if bool(upgrade.get("simplified_blue_experts", False)):
        temperature = float(upgrade.get("simplified_blue_temperature", 0.90))
        if abs(temperature - 1.0) > 1e-6:
            log_probs = np.log(np.clip(mixed, 1e-12, 1.0))
            mixed = _normalize(np.exp(log_probs / max(temperature, 0.1)))
        mode_name = "simplified_hedge_fixed_eta"
    elif bool(upgrade.get("ada_hedge_enable", True)):
        mode_name = "online_adahedge"
    else:
        mode_name = "online_hedge_ewa"
    return mixed, {
        "mode": mode_name,
        "updates": int(update_count),
        "eta0": float(eta0),
        "eta_last": float(eta_trace[-1]) if eta_trace else None,
        "eta_mean": float(np.mean(eta_trace)) if eta_trace else None,
        "simplified": bool(upgrade.get("simplified_blue_experts", False)),
        "weights": {
            name: float(weights[index])
            for index, name in enumerate(expert_names)
        },
        "inactive_pool": inactive_pool,
        "mean_log_losses": mean_loss_by_name,
    }


def _multinomial_nll(values: np.ndarray, alpha: float = 1.0) -> float:
    counts = np.bincount(values - 1, minlength=16).astype(float)
    probability = (counts + float(alpha)) / (float(counts.sum()) + float(alpha) * 16.0)
    return -float(np.sum(counts * np.log(np.clip(probability, 1e-12, 1.0))))


def _change_point_frequency(history: np.ndarray) -> np.ndarray:
    length = len(history)
    if length < 500:
        return _frequency(history, min(300, length))
    global_probability = _frequency(history, None)
    best_window = 500
    best_score = 0.0
    for window in (100, 200, 300, 500, 1000):
        if length <= window + 80:
            continue
        recent_probability = _frequency(history[-window:], None)
        divergence = float(np.sum(
            recent_probability
            * np.log(np.clip(recent_probability, 1e-12, 1.0) / np.clip(global_probability, 1e-12, 1.0))
        ))
        score = divergence * np.sqrt(window)
        if score > best_score:
            best_score = score
            best_window = window
    if best_score < 0.08:
        return _normalize(0.65 * _frequency(history, 500) + 0.35 * global_probability)
    return _normalize(0.78 * _frequency(history, best_window) + 0.22 * global_probability)


def _periodic_phase_mixture(history: np.ndarray) -> np.ndarray:
    if len(history) < 60:
        return _frequency(history, None)
    target_position = len(history)
    scores = np.ones(16, dtype=float) * 0.25
    positions = np.arange(len(history), dtype=int)
    for period, weight in ((3, 0.45), (7, 0.25), (16, 0.18), (33, 0.12)):
        same_phase = history[positions % period == target_position % period]
        if len(same_phase) == 0:
            continue
        scores += float(weight) * (_frequency(same_phase, None) * 16.0)
    scores += 0.35 * (_time_decay(history, 0.99) * 16.0)
    return _normalize(scores)


def _gap_shape(history: np.ndarray) -> np.ndarray:
    scores = np.ones(16, dtype=float) * 0.05
    length = len(history)
    for blue in range(1, 17):
        positions = np.where(history == blue)[0]
        if len(positions) == 0:
            continue
        omission = length - 1 - int(positions[-1])
        gaps = np.diff(positions).astype(float)
        if len(gaps) < 2:
            mean_gap = 16.0
            std_gap = 8.0
            skew = 0.0
        else:
            mean_gap = float(gaps.mean())
            std_gap = max(float(gaps.std(ddof=0)), 1.0)
            centered = gaps - mean_gap
            skew = float(np.mean(centered ** 3) / max(std_gap ** 3, 1e-9))
        z_score = (omission - mean_gap) / max(std_gap, 1.0)
        scores[blue - 1] += float(1.0 / (1.0 + np.exp(-np.clip(z_score + 0.18 * skew, -50.0, 50.0))))
    return _normalize(scores)


def _gap_hazard(history: np.ndarray) -> np.ndarray:
    scores = np.ones(16, dtype=float) * 0.05
    length = len(history)
    for blue in range(1, 17):
        positions = np.where(history == blue)[0]
        if len(positions) == 0:
            continue
        omission = length - 1 - int(positions[-1])
        gaps = np.diff(positions)
        mean_gap = float(np.mean(gaps)) if len(gaps) else 16.0
        scores[blue - 1] += min(3.0, (omission + 1.0) / max(mean_gap, 1e-9))
    return _normalize(scores)


def _neighbor_transition(history: np.ndarray) -> np.ndarray:
    if len(history) < 2:
        return np.full(16, 1.0 / 16.0)
    last_blue = int(history[-1])
    scores = np.ones(16, dtype=float) * 0.2
    for previous, current in zip(history[:-1], history[1:]):
        distance = abs(int(previous) - last_blue)
        if distance == 0:
            scores[int(current) - 1] += 1.0
        elif distance == 1:
            scores[int(current) - 1] += 0.45
        elif distance == 2:
            scores[int(current) - 1] += 0.20
    return _normalize(scores)


def _forward_backward(
    observations: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    periods = len(observations)
    states = len(initial)
    alpha = np.zeros((periods, states), dtype=float)
    scales = np.ones(periods, dtype=float)
    alpha[0] = initial * emission[:, int(observations[0]) - 1]
    scales[0] = max(float(alpha[0].sum()), 1e-300)
    alpha[0] /= scales[0]
    for index in range(1, periods):
        alpha[index] = (alpha[index - 1] @ transition) * emission[:, int(observations[index]) - 1]
        scales[index] = max(float(alpha[index].sum()), 1e-300)
        alpha[index] /= scales[index]

    beta = np.ones((periods, states), dtype=float)
    for index in range(periods - 2, -1, -1):
        beta[index] = transition @ (emission[:, int(observations[index + 1]) - 1] * beta[index + 1])
        beta[index] /= max(float(beta[index].sum()), 1e-300)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    log_likelihood = float(np.sum(np.log(scales)))
    return alpha, gamma, log_likelihood


def _fit_hmm_em(observations: np.ndarray, state_count: int, *, iterations: int = 8) -> dict[str, Any]:
    observations = observations.astype(int)
    periods = len(observations)
    states = int(state_count)
    if periods < states * 30:
        raise ValueError("HMM 样本不足")

    initial = np.full(states, 1.0 / states)
    transition = np.full((states, states), 0.08 / max(states - 1, 1))
    np.fill_diagonal(transition, 0.92)
    transition = transition / transition.sum(axis=1, keepdims=True)
    global_counts = np.bincount(observations - 1, minlength=16).astype(float) + 1.0
    global_probability = _normalize(global_counts)
    emission = np.zeros((states, 16), dtype=float)
    for state in range(states):
        shifted = np.roll(global_probability, state * max(1, 16 // states))
        emission[state] = _normalize(0.65 * global_probability + 0.35 * shifted)

    last_log_likelihood = -np.inf
    for _ in range(int(iterations)):
        alpha, gamma, log_likelihood = _forward_backward(observations, transition, emission, initial)
        xi_sum = np.zeros((states, states), dtype=float)
        for index in range(periods - 1):
            numerator = (
                alpha[index, :, None]
                * transition
                * emission[:, int(observations[index + 1]) - 1][None, :]
            )
            denominator = max(float(numerator.sum()), 1e-300)
            xi_sum += numerator / denominator
        initial = _normalize(gamma[0] + 0.05)
        transition = xi_sum + 0.35
        transition /= np.maximum(transition.sum(axis=1, keepdims=True), 1e-300)
        emission_counts = np.ones((states, 16), dtype=float) * 0.55
        for blue in range(1, 17):
            mask = observations == blue
            if np.any(mask):
                emission_counts[:, blue - 1] += gamma[mask].sum(axis=0)
        emission = emission_counts / np.maximum(emission_counts.sum(axis=1, keepdims=True), 1e-300)
        if abs(log_likelihood - last_log_likelihood) < 1e-7:
            break
        last_log_likelihood = log_likelihood
    _, gamma, log_likelihood = _forward_backward(observations, transition, emission, initial)
    parameters = (states - 1) + states * (states - 1) + states * 15
    bic = -2.0 * log_likelihood + parameters * np.log(max(periods, 2))
    return {
        "states": states,
        "initial": initial,
        "transition": transition,
        "emission": emission,
        "gamma": gamma,
        "log_likelihood": float(log_likelihood),
        "bic": float(bic),
    }


def _hmm_regime_distribution(
    history: np.ndarray,
    upgrade_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    upgrade = _upgrade_options(upgrade_config)
    if len(history) < 160:
        return _frequency(history, min(300, len(history))), {"mode": "hmm_insufficient_history"}
    if bool(upgrade.get("hmm_regime_full_window", True)):
        summary = regime_summary(history, size=16)
        change_points = list(summary.get("change_points", []))
        latest_segment_start = int(change_points[-1]) if change_points else max(0, len(history) - 1200)
        weighted_probability = weighted_recent_distribution(
            history,
            change_points=change_points,
            half_life=int(upgrade.get("hmm_half_life", 520)),
            pre_change_decay=float(upgrade.get("hmm_pre_change_decay", 0.42)),
            size=16,
        )
        observations = history[max(0, latest_segment_start - 260):]
        if len(observations) < 180:
            observations = history[-min(len(history), 900):]
        state_choices = (2, 3, 4)
    else:
        change_points = []
        weighted_probability = _frequency(history, min(len(history), 420))
        observations = history[-min(len(history), 420):]
        state_choices = (2, 3)
    fits: list[dict[str, Any]] = []
    for states in state_choices:
        try:
            fits.append(_fit_hmm_em(observations, states))
        except ValueError:
            continue
    if not fits:
        return _frequency(history, min(300, len(history))), {"mode": "hmm_fit_unavailable"}
    best = min(fits, key=lambda item: float(item["bic"]))
    filtered_state = best["gamma"][-1] @ best["transition"]
    regime_probability = _normalize(filtered_state @ best["emission"])
    probability = (
        _normalize(0.70 * regime_probability + 0.30 * weighted_probability)
        if bool(upgrade.get("hmm_regime_full_window", True))
        else regime_probability
    )
    return probability, {
        "mode": "hmm_regime_switching",
        "selected_states": int(best["states"]),
        "bic": float(best["bic"]),
        "state_posterior": [float(value) for value in best["gamma"][-1]],
        "change_points": [int(point) for point in change_points],
        "observations": int(len(observations)),
        "full_window": bool(upgrade.get("hmm_regime_full_window", True)),
    }


def _blue_set_diversity(values: tuple[int, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    ordered = np.asarray(sorted(values), dtype=float)
    spacing = np.diff(ordered)
    spacing_score = float(np.mean(spacing) / 15.0)
    zone_count = len({
        0 if value <= 4 else 1 if value <= 8 else 2 if value <= 12 else 3
        for value in ordered
    })
    parity_count = len({int(value) % 2 for value in ordered})
    return float(0.45 * spacing_score + 0.35 * zone_count / 4.0 + 0.20 * parity_count / 2.0)


def _blue_combination_cache(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = max(1, min(int(count), 16))
    cached = _BLUE_COMBINATION_CACHE.get(count)
    if cached is not None:
        return cached
    combos = np.asarray(list(combinations(range(1, 17), count)), dtype=int)
    indexes = combos - 1
    diversity = np.asarray([
        _blue_set_diversity(tuple(int(value) for value in row))
        for row in combos
    ], dtype=float)
    _BLUE_COMBINATION_CACHE[count] = (combos, indexes, diversity)
    return combos, indexes, diversity


def _uniform_no_edge_coverage_blues(
    history: pd.DataFrame | Iterable[Any],
    probability: np.ndarray,
    count: int,
    *,
    excluded_blues: set[int] | None = None,
    recent_blue_exposure: dict[int, int] | None = None,
    exposure_penalty_weight: float = 0.30,
) -> tuple[list[int], dict[str, Any]]:
    """Strict NO_EDGE blue allocation: coverage first, probability only tie-breaks."""

    probability = _normalize(probability)
    count = max(1, min(int(count), 16))
    excluded = {
        int(value)
        for value in (excluded_blues or set())
        if 1 <= int(value) <= 16
    }
    allowed = {blue for blue in range(1, 17) if blue not in excluded}
    if not allowed:
        return [], {
            "objective": "blue_no_edge_uniform_coverage",
            "reason": "all_blue_numbers_excluded",
            "excluded_blue_numbers": sorted(excluded),
        }
    count = min(count, len(allowed))
    blue_history = _blue_array(history)
    last_blue = int(blue_history[-1]) if len(blue_history) else 0
    ring = [((last_blue + offset) % 16) + 1 for offset in range(16)]
    bucket_by_blue = {
        int(blue): min(count - 1, int(offset * count / 16))
        for offset, blue in enumerate(ring)
    }
    combos, indexes, diversity = _blue_combination_cache(count)
    allowed_flags = np.zeros(17, dtype=bool)
    for blue in allowed:
        allowed_flags[int(blue)] = True
    allowed_mask = allowed_flags[combos].all(axis=1)
    if not bool(allowed_mask.any()):
        allowed_mask = np.ones(len(combos), dtype=bool)
    bucket_lookup = np.full(17, -1, dtype=int)
    for blue, bucket in bucket_by_blue.items():
        bucket_lookup[int(blue)] = int(bucket)
    combo_buckets = np.sort(bucket_lookup[combos], axis=1)
    unique_buckets = np.ones(len(combos), dtype=float)
    if count > 1:
        unique_buckets += (combo_buckets[:, 1:] != combo_buckets[:, :-1]).sum(axis=1)
    bucket_coverage = unique_buckets / float(count)
    exposure_counts = {
        int(key): int(value)
        for key, value in (recent_blue_exposure or {}).items()
        if 1 <= int(key) <= 16 and int(value) > 0
    }
    max_exposure = max(exposure_counts.values(), default=0)
    if max_exposure > 0:
        exposure_lookup = np.zeros(17, dtype=float)
        for key, value in exposure_counts.items():
            exposure_lookup[int(key)] = float(value) / float(max_exposure)
        exposure_penalty = exposure_lookup[combos].sum(axis=1) / float(count)
    else:
        exposure_penalty = np.zeros(len(combos), dtype=float)
    probability_tiebreak = probability[indexes].sum(axis=1)
    scores = (
        1.60 * bucket_coverage
        + 0.65 * diversity
        - float(exposure_penalty_weight) * exposure_penalty
        + 0.002 * probability_tiebreak
    )
    masked_scores = np.where(allowed_mask, scores, -np.inf)
    best_index = int(np.argmax(masked_scores))
    selected = sorted(int(value) for value in combos[best_index])
    return selected, {
        "objective": "blue_no_edge_uniform_coverage",
        "score": float(scores[best_index]),
        "edge_state": "NO_EDGE",
        "prediction_set_size": 16,
        "allowed_blues": sorted(allowed),
        "selected_blues": selected,
        "last_blue": int(last_blue) if last_blue else None,
        "bucket_coverage": float(bucket_coverage[best_index]),
        "diversity": float(diversity[best_index]),
        "cross_period_exposure_penalty": float(exposure_penalty[best_index]),
        "exposure_penalty_weight": float(exposure_penalty_weight),
        "probability_tiebreak_weight": 0.002,
        "diagnostic_probability_rank": {str(blue): int(rank) for rank, blue in enumerate(_order(probability), start=1)},
        "recent_blue_exposure": {str(key): int(value) for key, value in sorted(exposure_counts.items())},
        "excluded_blue_numbers": sorted(excluded),
        "primary_objective": "NO_EDGE：16号码完整Prediction Set → 覆盖桶/分散/跨期曝光；概率仅用于同分破平。",
    }


def _coverage_optimized_blues(
    probability: np.ndarray,
    count: int,
    *,
    lambda_value: float | None = None,
    mu_value: float | None = None,
    top1_weight: float | None = None,
    excluded_blues: set[int] | None = None,
    candidate_top_n: int | None = None,
    max_rank_penalty: float | None = None,
    recent_blue_exposure: dict[int, int] | None = None,
    exposure_penalty_weight: float | None = None,
    edge_state_override: str | None = None,
) -> tuple[list[int], dict[str, Any]]:
    probability = _normalize(probability)
    count = max(1, min(int(count), 16))
    excluded = {
        int(value)
        for value in (excluded_blues or set())
        if 1 <= int(value) <= 16
    }
    ranks = {blue: rank for rank, blue in enumerate(_order(probability), start=1)}
    candidate_top_n = max(count, min(int(candidate_top_n or 16), 16))
    allowed = {
        blue for blue in range(1, 17)
        if blue not in excluded and ranks.get(blue, 99) <= candidate_top_n
    }
    if len(allowed) < count:
        allowed.update(
            blue for blue in _order(probability)
            if blue not in excluded
        )
    if not allowed:
        return [], {
            "objective": "blue_coverage_exhaustive_search",
            "reason": "all_blue_numbers_excluded",
            "excluded_blue_numbers": sorted(excluded),
        }
    uniform = 1.0 / 16.0
    signal_strength = float(np.sum(np.abs(probability - uniform)) / 2.0)
    edge_state = (
        "STRONG_EDGE"
        if signal_strength >= 0.16
        else "WEAK_EDGE"
        if signal_strength >= 0.075
        else "NO_EDGE"
    )
    if edge_state_override:
        edge_state = str(edge_state_override).strip().upper()
    if edge_state == "NO_EDGE":
        candidate_top_n = max(candidate_top_n, min(16, max(count + 7, 14)))
    elif edge_state == "WEAK_EDGE":
        candidate_top_n = max(candidate_top_n, min(16, max(count + 5, 10)))
    if lambda_value is None:
        lambda_value = 0.18 + 0.70 * max(0.0, 0.18 - signal_strength) / 0.18
    if mu_value is None:
        mu_value = 0.12 + 0.55 * max(0.0, 0.18 - signal_strength) / 0.18
    if top1_weight is None:
        # Top1 is explicitly not the production objective when there is no
        # stable edge.  Only reintroduce a small head-probability bonus when the
        # probability surface has real separation.
        top1_weight = 0.0 if edge_state == "NO_EDGE" else (0.05 if edge_state == "WEAK_EDGE" else 0.28)
    combos, indexes, diversity = _blue_combination_cache(count)
    allowed_flags = np.zeros(17, dtype=bool)
    for blue in allowed:
        allowed_flags[int(blue)] = True
    allowed_mask = allowed_flags[combos].all(axis=1)
    if not bool(allowed_mask.any()):
        allowed_mask = np.ones(len(combos), dtype=bool)
    selected_probability = probability[indexes]
    ordered_by_probability = _order(probability)
    tier_a = set(ordered_by_probability[:5])
    tier_b = set(ordered_by_probability[5:10])
    tier_c = set(ordered_by_probability[10:])
    if edge_state == "STRONG_EDGE":
        target_tiers = {"A": min(2, count), "B": min(2, max(0, count - 2)), "C": max(0, count - 4)}
    elif edge_state == "WEAK_EDGE":
        target_tiers = {"A": min(2, count), "B": min(2, max(0, count - 2)), "C": max(0, count - 4)}
    else:
        target_tiers = {"A": min(1, count), "B": min(2, max(0, count - 1)), "C": max(0, count - 3)}
    tier_code = np.zeros(17, dtype=np.int8)
    for value in tier_a:
        tier_code[int(value)] = 1
    for value in tier_b:
        tier_code[int(value)] = 2
    for value in tier_c:
        tier_code[int(value)] = 3
    combo_tiers = tier_code[combos]
    distance = (
        np.abs(np.sum(combo_tiers == 1, axis=1) - target_tiers["A"])
        + np.abs(np.sum(combo_tiers == 2, axis=1) - target_tiers["B"])
        + np.abs(np.sum(combo_tiers == 3, axis=1) - target_tiers["C"])
    )
    tier_scores = 1.0 - distance / max(1.0, 2.0 * count)
    rank_lookup = np.full(17, 16.0, dtype=float)
    for blue, rank in ranks.items():
        rank_lookup[int(blue)] = float(rank)
    rank_penalty = np.clip(rank_lookup[combos] - float(candidate_top_n), 0.0, None).sum(axis=1)
    probability_weight = 0.06 if edge_state == "NO_EDGE" else 0.28 if edge_state == "WEAK_EDGE" else 1.00
    uniform_exposure = 1.0 / float(count)
    exposure_term = float(count) * abs(1.0 - uniform_exposure) + float(16 - count) * abs(uniform_exposure)
    exposure_balance = np.full(len(combos), 1.0 - min(1.0, exposure_term / 16.0), dtype=float)
    exposure_counts = {
        int(key): int(value)
        for key, value in (recent_blue_exposure or {}).items()
        if 1 <= int(key) <= 16 and int(value) > 0
    }
    max_exposure = max(exposure_counts.values(), default=0)
    if max_exposure > 0:
        exposure_lookup = np.zeros(17, dtype=float)
        for key, value in exposure_counts.items():
            exposure_lookup[int(key)] = float(value) / float(max_exposure)
        cross_period_exposure_penalty = exposure_lookup[combos].sum(axis=1) / float(count)
    else:
        cross_period_exposure_penalty = np.zeros(len(combos), dtype=float)
    if exposure_penalty_weight is None:
        exposure_penalty_weight = 0.30 if edge_state == "NO_EDGE" else 0.14 if edge_state == "WEAK_EDGE" else 0.04
    scores = (
        float(probability_weight) * selected_probability.sum(axis=1)
        + float(top1_weight) * selected_probability.max(axis=1)
        - float(lambda_value) * np.square(selected_probability).sum(axis=1)
        + float(mu_value) * diversity
        + (0.42 if edge_state == "NO_EDGE" else 0.26 if edge_state == "WEAK_EDGE" else 0.12) * tier_scores
        + (0.34 if edge_state == "NO_EDGE" else 0.16 if edge_state == "WEAK_EDGE" else 0.04) * exposure_balance
        - float(exposure_penalty_weight) * cross_period_exposure_penalty
        - float(max_rank_penalty if max_rank_penalty is not None else 0.0) * rank_penalty
    )
    masked_scores = np.where(allowed_mask, scores, -np.inf)
    best_index = int(np.argmax(masked_scores))
    best_score = float(scores[best_index])
    best_set = tuple(int(value) for value in combos[best_index])
    ordered = sorted(best_set, key=lambda value: float(probability[value - 1]), reverse=True)
    return ordered, {
        "objective": "blue_coverage_exhaustive_search",
        "score": float(best_score),
        "lambda": float(lambda_value),
        "mu": float(mu_value),
        "top1_weight": float(top1_weight),
        "probability_weight": float(probability_weight),
        "diagnostic_top1_blue": int(ordered[0]) if ordered else None,
        "signal_strength": signal_strength,
        "edge_state": edge_state,
        "primary_objective": "Prediction Set + Hit@5 set coverage",
        "tier_target": target_tiers,
        "tier_fit": float(tier_scores[best_index]),
        "exposure_balance": float(exposure_balance[best_index]),
        "cross_period_exposure_penalty": float(cross_period_exposure_penalty[best_index]),
        "exposure_penalty_weight": float(exposure_penalty_weight),
        "recent_blue_exposure": {str(key): int(value) for key, value in sorted(exposure_counts.items())},
        "candidate_top_n": int(candidate_top_n),
        "max_rank_penalty": float(max_rank_penalty if max_rank_penalty is not None else 0.0),
        "excluded_blue_numbers": sorted(excluded),
    }


def _rotating_bucket_blues(
    probability: np.ndarray,
    count: int,
    history: pd.DataFrame | Iterable[Any],
    *,
    excluded_blues: set[int] | None = None,
    candidate_top_n: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Choose distinct blue numbers from ring buckets; model only ranks inside buckets.

    This is the conservative strategy recommended by the整改文档: when the
    single-blue model is weak, the most defensible target is stable coverage
    rather than chasing small probability differences.  We still use the model
    probability, but only as a tie-breaker inside evenly rotated buckets.
    """
    probability = _normalize(probability)
    count = max(1, min(int(count), 16))
    excluded = {
        int(value)
        for value in (excluded_blues or set())
        if 1 <= int(value) <= 16
    }
    ranks = {blue: rank for rank, blue in enumerate(_order(probability), start=1)}
    candidate_top_n = max(count, min(int(candidate_top_n or 16), 16))
    usable = [
        blue for blue in range(1, 17)
        if blue not in excluded and ranks.get(blue, 99) <= candidate_top_n
    ]
    if len(usable) < count:
        usable = [blue for blue in range(1, 17) if blue not in excluded]
    if not usable:
        return [], {
            "objective": "blue_rotating_bucket_coverage",
            "reason": "all_blue_numbers_excluded",
            "excluded_blue_numbers": sorted(excluded),
        }
    count = min(count, len(usable))
    blue_history = _blue_array(history)
    last_blue = int(blue_history[-1]) if len(blue_history) else 0
    # Start after the latest blue so consecutive runs naturally rotate around
    # the 16-number ring.  This avoids permanently favoring the same buckets.
    ring = [((last_blue + offset) % 16) + 1 for offset in range(16)]
    buckets: list[list[int]] = [[] for _ in range(count)]
    for offset, blue in enumerate(ring):
        if blue in excluded:
            continue
        bucket_index = min(count - 1, int(offset * count / 16))
        buckets[bucket_index].append(int(blue))

    selected: list[int] = []
    for bucket in buckets:
        if not bucket:
            continue
        blue = max(bucket, key=lambda value: (float(probability[value - 1]), -value))
        if blue not in selected:
            selected.append(int(blue))

    if len(selected) < count:
        for blue in sorted(
            usable,
            key=lambda value: (
                value not in selected,
                float(probability[value - 1]),
                _blue_set_diversity(tuple(sorted([*selected, value]))),
                -value,
            ),
            reverse=True,
        ):
            if blue in selected:
                continue
            selected.append(int(blue))
            if len(selected) >= count:
                break

    ordered = sorted(selected[:count], key=lambda value: float(probability[value - 1]), reverse=True)
    return ordered, {
        "objective": "blue_rotating_bucket_coverage",
        "policy": "model_ranked_within_rotating_buckets",
        "last_blue": int(last_blue) if last_blue else None,
        "bucket_count": int(count),
        "excluded_blue_numbers": sorted(excluded),
        "candidate_top_n": int(candidate_top_n),
        "signal_strength": float(np.sum(np.abs(probability - 1.0 / 16.0)) / 2.0),
    }


def _policy_optimized_blues(
    history: pd.DataFrame | Iterable[Any],
    probability: np.ndarray,
    count: int,
    upgrade_config: dict[str, Any],
    *,
    excluded_blues: set[int] | None = None,
    lambda_value: float | None = None,
    mu_value: float | None = None,
) -> tuple[list[int], dict[str, Any]]:
    policy = str(upgrade_config.get("blue_coverage_policy", "probability_optimized")).strip().lower()
    recent_exposure = {
        int(key): int(value)
        for key, value in (upgrade_config.get("_recent_final_blue_exposure") or {}).items()
        if 1 <= int(key) <= 16 and int(value) > 0
    }
    if policy == "rotating_buckets":
        return _rotating_bucket_blues(
            probability,
            count,
            history,
            excluded_blues=excluded_blues,
            candidate_top_n=int(upgrade_config.get("blue_coverage_candidate_top_n", 10)),
        )
    if policy in {"coverage_only", "no_edge_coverage_only"}:
        if bool(upgrade_config.get("blue_no_edge_adaptive_history_policy", True)):
            adaptive_count = int(count)
            selected, adaptive_info = select_adaptive_blue_set(
                history["blue"].tolist() if hasattr(history, "columns") and "blue" in history.columns else history,
                count=adaptive_count,
                probability=probability,
                evaluation_window=int(upgrade_config.get("blue_adaptive_oos_window", 60)),
                minimum_history=int(upgrade_config.get("blue_adaptive_min_history", 120)),
                excluded_blues=excluded_blues,
                recency_decay=float(upgrade_config.get("blue_adaptive_recency_decay", 0.985)),
                fallback_policy=str(upgrade_config.get("blue_adaptive_fallback_policy", "frequency_100")),
            )
            return selected, {
                **adaptive_info,
                "policy": "adaptive_history_oos",
                "strict_no_edge": True,
                "model_probability_alpha": 0.0,
                "cross_period_coverage": {
                    "enabled": True,
                    "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
                    "rule": "NO_EDGE使用仅基于目标期之前完成数据的滚动OOS策略选择；模型概率只允许近似并列时参与。",
                },
            }
        if bool(upgrade_config.get("blue_no_edge_strict_uniform_coverage", True)):
            selected, info = _uniform_no_edge_coverage_blues(
                history,
                probability,
                count,
                excluded_blues=excluded_blues,
                recent_blue_exposure=recent_exposure,
                exposure_penalty_weight=float(upgrade_config.get("blue_no_edge_exposure_penalty_weight", 0.30)),
            )
            return selected, {
                **info,
                "policy": policy,
                "strict_no_edge": True,
                "model_probability_alpha": 0.0,
                "cross_period_coverage": {
                    "enabled": True,
                    "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
                    "rule": "NO_EDGE下不再让弱Top1/弱概率排序主导Final 5，概率仅作为同分破平。",
                },
            }
        uniform = np.full(16, 1.0 / 16.0, dtype=float)
        weak_probability = _normalize(0.02 * _normalize(probability) + 0.98 * uniform)
        selected, info = _coverage_optimized_blues(
            weak_probability,
            count,
            lambda_value=lambda_value,
            mu_value=mu_value,
            top1_weight=0.0,
            excluded_blues=excluded_blues,
            candidate_top_n=16,
            max_rank_penalty=0.0,
            recent_blue_exposure=recent_exposure,
            exposure_penalty_weight=float(upgrade_config.get("blue_no_edge_exposure_penalty_weight", 0.30)),
            edge_state_override="NO_EDGE",
        )
        return selected, {
            **info,
            "policy": policy,
            "primary_objective": "Coverage Only under NO_EDGE: unique 5-blue set coverage, Top1 is diagnostic only",
            "model_probability_alpha": 0.02,
            "cross_period_coverage": {
                "enabled": True,
                "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
                "rule": "NO_EDGE下惩罚近期Final 5中过度曝光的蓝球，避免弱Top1偏置重新集中。",
            },
        }
    if policy == "weak_edge_mixed_coverage":
        uniform = np.full(16, 1.0 / 16.0, dtype=float)
        alpha = float(upgrade_config.get("blue_weak_edge_model_alpha", 0.22))
        alpha = min(0.35, max(0.08, alpha))
        mixed_probability = _normalize(alpha * _normalize(probability) + (1.0 - alpha) * uniform)
        selected, info = _coverage_optimized_blues(
            mixed_probability,
            count,
            lambda_value=lambda_value,
            mu_value=mu_value,
            top1_weight=0.03,
            excluded_blues=excluded_blues,
            candidate_top_n=int(upgrade_config.get("blue_coverage_candidate_top_n", 12)),
            max_rank_penalty=float(upgrade_config.get("blue_coverage_max_rank_penalty", 0.02)),
            recent_blue_exposure=recent_exposure,
            exposure_penalty_weight=float(upgrade_config.get("blue_weak_edge_exposure_penalty_weight", 0.14)),
            edge_state_override="WEAK_EDGE",
        )
        return selected, {
            **info,
            "policy": policy,
            "primary_objective": "WEAK_EDGE mixed mode: weak probability ranking plus Prediction Set coverage; Top1 remains diagnostic.",
            "model_probability_alpha": float(alpha),
            "cross_period_coverage": {
                "enabled": True,
                "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
                "rule": "WEAK_EDGE下仅保留弱概率排序，覆盖效用和跨期曝光仍参与最终5蓝选择。",
            },
        }
    return _coverage_optimized_blues(
        probability,
        count,
        lambda_value=lambda_value,
        mu_value=mu_value,
        excluded_blues=excluded_blues,
        candidate_top_n=int(upgrade_config.get("blue_coverage_candidate_top_n", 10)),
        max_rank_penalty=float(upgrade_config.get("blue_coverage_max_rank_penalty", 0.04)),
        recent_blue_exposure=recent_exposure,
    )


def _research_evidence_alpha(research: dict[str, Any], upgrade: dict[str, Any]) -> float:
    if not bool(upgrade.get("blue_evidence_shrinkage_enabled", True)):
        return 1.0 if bool(research.get("approved")) else 0.0
    level = str(research.get("approval_level", "rejected"))
    if level == "strong":
        return float(upgrade.get("blue_strong_edge_model_alpha", 0.70))
    if level == "stable_weak":
        return float(upgrade.get("blue_weak_edge_model_alpha", 0.22))
    best_edge = -1.0
    for row in research.get("rows", []) or []:
        for window in row.get("windows", []) or []:
            best_edge = max(best_edge, float(window.get("edge", 0.0) or 0.0))
    if best_edge >= 0.03:
        return float(upgrade.get("blue_medium_edge_model_alpha", 0.45))
    if best_edge > 0.0:
        return float(upgrade.get("blue_weak_edge_model_alpha", 0.22))
    return float(upgrade.get("blue_no_edge_model_alpha", 0.08))


def _shrunk_blue_probability(
    history: pd.DataFrame | Iterable[Any],
    research: dict[str, Any],
    upgrade: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    variant = str(research.get("selected_variant") or "online_hedge")
    probability, info = _direct_variant_probability(history, variant, upgrade)
    if probability is None:
        portfolio = list(research.get("selected_portfolio") or [])
        scores = np.ones(16, dtype=float)
        for rank, item in enumerate(portfolio, start=1):
            try:
                blue = int(item.get("blue"))
            except (TypeError, ValueError):
                continue
            if 1 <= blue <= 16:
                scores[blue - 1] += max(0.0, float(item.get("score", 0.0) or 0.0)) + 1.0 / rank
        probability = _normalize(scores)
        info = {"mode": "portfolio_score_probability_proxy", "variant": variant}
    alpha = float(np.clip(_research_evidence_alpha(research, upgrade), 0.0, 1.0))
    uniform = np.full(16, 1.0 / 16.0, dtype=float)
    shrunk = _normalize(alpha * _normalize(probability) + (1.0 - alpha) * uniform)
    return shrunk, {
        "selected_variant": variant,
        "model_alpha": alpha,
        "model_probability_info": info,
        "shrinkage": "continuous_evidence_to_uniform",
    }


def _nested_meta_hedge_distribution(
    history: pd.DataFrame | Iterable[Any],
    *,
    upgrade_config: dict[str, Any] | None = None,
    replay: int = 120,
) -> tuple[np.ndarray, dict[str, Any]]:
    upgrade = _upgrade_options(upgrade_config)
    # 方案2.5：简化模式跳过HMM和嵌套结构，直接返回Hedge结果
    if bool(upgrade.get("simplified_blue_experts", False)):
        hedge_probability, hedge_info = _online_hedge_mixture(history, upgrade_config=upgrade)
        return hedge_probability, {
            "mode": "simplified_hedge_only",
            "simplified": True,
            "hedge": hedge_info,
        }
    blue_history = _blue_array(history)
    if len(blue_history) < 220:
        hedge_probability, hedge_info = _online_hedge_mixture(history, upgrade_config=upgrade)
        return hedge_probability, {"mode": "nested_hedge_bootstrap", "hedge": hedge_info}
    start = max(120, len(blue_history) - int(replay))
    warm_start = max(60, start - int(upgrade.get("nested_replay_warmup", 80)))
    replay_upgrade = {**upgrade, "expert_pool_extend": False, "cross_red_blue_feature": False}
    expert_names = list(_base_blue_experts(_slice_history(history, max(2, warm_start)), replay_upgrade).keys())
    inner_weights = np.full(len(expert_names), 1.0 / len(expert_names), dtype=float)
    meta_weights = np.asarray([0.5, 0.5], dtype=float)
    update_count = 0
    cached_hmm_probability: np.ndarray | None = None
    cached_hmm_info: dict[str, Any] = {}
    for target_index in range(warm_start, len(blue_history)):
        prefix = _slice_history(history, target_index)
        prefix_blue = _blue_array(prefix)
        experts = _base_blue_experts(prefix, replay_upgrade)
        probabilities = np.stack([experts[name] for name in expert_names], axis=0)
        hedge_probability = _normalize(inner_weights @ probabilities)
        if cached_hmm_probability is None or (target_index - warm_start) % 20 == 0:
            cached_hmm_probability, cached_hmm_info = _hmm_regime_distribution(prefix_blue, upgrade)
        hmm_probability = cached_hmm_probability
        actual = int(blue_history[target_index]) - 1
        expert_losses = -np.log(np.clip(probabilities[:, actual], 1e-12, 1.0))
        inner_eta = float(np.clip(0.16 / np.sqrt(max(update_count + 1, 1)), 0.02, 0.35))
        inner_weights *= np.exp(-inner_eta * np.clip(expert_losses, 0.0, 30.0))
        inner_weights = _normalize(inner_weights)
        if target_index >= start:
            meta_losses = -np.log(np.clip(np.asarray([
                hedge_probability[actual],
                hmm_probability[actual],
            ], dtype=float), 1e-12, 1.0))
            eta = float(np.clip(0.18 / np.sqrt(max(update_count + 1, 1)), 0.02, 0.35))
            meta_weights *= np.exp(-eta * np.clip(meta_losses, 0.0, 30.0))
            meta_weights = _normalize(meta_weights)
            update_count += 1
    current_experts = _base_blue_experts(history, replay_upgrade)
    current_probabilities = np.stack([current_experts[name] for name in expert_names], axis=0)
    hedge_probability = _normalize(inner_weights @ current_probabilities)
    hmm_probability, hmm_info = _hmm_regime_distribution(blue_history, upgrade)
    hedge_info = {
        "mode": "nested_inner_online_state",
        "weights": {
            name: float(inner_weights[index])
            for index, name in enumerate(expert_names)
        },
    }
    probability = _normalize(meta_weights[0] * hedge_probability + meta_weights[1] * hmm_probability)
    return probability, {
        "mode": "nested_hedge_meta_fusion",
        "updates": int(update_count),
        "meta_weights": {
            "inner_expert_hedge": float(meta_weights[0]),
            "hmm_regime": float(meta_weights[1]),
        },
        "hedge": hedge_info,
        "hmm": hmm_info,
    }


def _online_hybrid_distribution(
    history: pd.DataFrame | Iterable[Any],
    upgrade_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    upgrade = _upgrade_options(upgrade_config)
    # 方案2.5：简化模式跳过HMM，直接返回Hedge结果
    if bool(upgrade.get("simplified_blue_experts", False)):
        hedge_probability, hedge_info = _online_hedge_mixture(history, upgrade_config=upgrade)
        return hedge_probability, {
            "mode": "simplified_hedge_only",
            "simplified": True,
            "hedge": hedge_info,
        }
    if bool(upgrade.get("nested_hedge_enable", True)):
        return _nested_meta_hedge_distribution(history, upgrade_config=upgrade)
    blue_history = _blue_array(history)
    hedge_probability, hedge_info = _online_hedge_mixture(history, upgrade_config=upgrade)
    hmm_probability, hmm_info = _hmm_regime_distribution(blue_history, upgrade)
    # Treat Hedge and HMM as two meta experts.  Full nested online learning is
    # expensive for every web run, so alpha is a lightweight reliability proxy:
    # Hedge is the stable default; HMM receives more share only when a real
    # regime model was fitted and its state posterior is not completely diffuse.
    if str(hmm_info.get("mode")) != "hmm_regime_switching":
        alpha = 0.90
        regime_confidence = 0.0
    else:
        posterior = np.asarray(hmm_info.get("state_posterior", []), dtype=float)
        regime_confidence = float(max(posterior, default=0.0) - 1.0 / max(len(posterior), 1))
        alpha = float(np.clip(0.82 - 0.42 * max(0.0, regime_confidence), 0.45, 0.88))
    probability = _normalize(float(alpha) * hedge_probability + (1.0 - float(alpha)) * hmm_probability)
    return probability, {
        "mode": "online_hedge_plus_hmm",
        "alpha_hedge": float(alpha),
        "regime_confidence": float(regime_confidence),
        "hedge": hedge_info,
        "hmm": hmm_info,
    }


def blue_specialist_portfolio(
    history: pd.DataFrame | Iterable[Any],
    count: int,
    *,
    variant: str = "stable_frequency_transition",
    upgrade_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return distinct blue numbers from independent blue-only views.

    The portfolio intentionally optimizes group coverage rather than top-1
    confidence.  Each source is simple enough to audit: first-order transition,
    time decay, current omission pressure, 300-period frequency, all-history
    frequency, neighboring transition, and 100-period frequency.
    """
    blue_history = _blue_array(history)
    count = max(0, min(int(count), 16))
    if count == 0 or len(blue_history) == 0:
        return []
    if variant not in BLUE_SPECIALIST_VARIANTS:
        raise ValueError(f"未知蓝球专项策略: {variant}")

    upgrade = _upgrade_options(upgrade_config)
    excluded_blues = {
        int(value)
        for value in (upgrade.get("excluded_blue_numbers", []) or [])
        if 1 <= int(value) <= 16
    }
    direct_probability, direct_reason = _direct_variant_probability(history, variant, upgrade)
    if direct_probability is not None:
        tuned_parameters = (
            _tuned_blue_coverage_parameters(history, variant, count, upgrade)
            if bool(upgrade.get("coverage_param_online_tune", True))
            and bool(upgrade.get("coverage_tune_current_output", False))
            else {}
        )
        selected_blues, coverage_reason = _policy_optimized_blues(
            history,
            direct_probability,
            count,
            upgrade,
            excluded_blues=excluded_blues,
            lambda_value=tuned_parameters.get("lambda"),
            mu_value=tuned_parameters.get("mu"),
        )
        return [
            {
                "blue": int(blue),
                "source": str(variant),
                "score": float(direct_probability[int(blue) - 1]),
                "reason": _compact_reason({
                    **direct_reason,
                    "coverage": {
                        **coverage_reason,
                        "online_tuning": tuned_parameters,
                    },
                }),
            }
            for blue in selected_blues
        ]

    sources = _portfolio_sources(blue_history, variant)
    if str(upgrade.get("blue_coverage_policy", "probability_optimized")).strip().lower() == "rotating_buckets":
        blended = np.ones(16, dtype=float)
        for _, probabilities, quota in sources:
            blended += max(1, int(quota)) * _normalize(probabilities)
        selected_blues, coverage_reason = _policy_optimized_blues(
            history,
            blended,
            count,
            upgrade,
            excluded_blues=excluded_blues,
        )
        return [
            {
                "blue": int(blue),
                "source": f"{variant}_rotating_bucket_consensus",
                "score": float(_normalize(blended)[int(blue) - 1]),
                "reason": coverage_reason,
            }
            for blue in selected_blues
        ]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source, probabilities, quota in sources:
        for blue in _order(probabilities)[: int(quota)]:
            if int(blue) in excluded_blues:
                continue
            if blue in seen:
                continue
            selected.append({
                "blue": int(blue),
                "source": source,
                "score": float(probabilities[int(blue) - 1]),
            })
            seen.add(blue)
            if len(selected) >= count:
                return selected

    # Defensive fill; normally the source loop above returns enough numbers.
    fallback = _frequency(blue_history, 300)
    for blue in _order(fallback):
        if int(blue) in excluded_blues:
            continue
        if blue in seen:
            continue
        selected.append({
            "blue": int(blue),
            "source": "fallback_frequency_300",
            "score": float(fallback[int(blue) - 1]),
        })
        seen.add(blue)
        if len(selected) >= count:
            break
    return selected


def _direct_variant_probability(
    history: pd.DataFrame | Iterable[Any],
    variant: str,
    upgrade_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    upgrade = _upgrade_options(upgrade_config)
    blue_history = _blue_array(history)
    if len(blue_history) == 0:
        return None, {}
    if variant == "bayes_markov_only":
        probability, info = _bayesian_shrinkage_markov(blue_history)
        return probability, info
    if variant == "online_hedge":
        probability, info = _online_hedge_mixture(history, upgrade_config=upgrade)
        return probability, info
    if variant == "hmm_regime":
        probability, info = _hmm_regime_distribution(blue_history, upgrade)
        return probability, info
    if variant == "online_hybrid_coverage":
        probability, info = _online_hybrid_distribution(history, upgrade)
        return probability, info
    if variant == "nested_hedge":
        probability, info = _nested_meta_hedge_distribution(history, upgrade_config=upgrade)
        return probability, info
    return None, {}


def _tuned_blue_coverage_parameters(
    history: pd.DataFrame | Iterable[Any],
    variant: str,
    count: int,
    upgrade_config: dict[str, Any],
) -> dict[str, Any]:
    blue_history = _blue_array(history)
    tune_window = min(int(upgrade_config.get("coverage_tune_window", 80)), max(0, len(blue_history) - 160))
    tune_variant = "online_hedge" if variant in {"nested_hedge", "online_hybrid_coverage", "hmm_regime"} else variant
    if tune_variant != variant:
        tune_window = min(tune_window, int(upgrade_config.get("coverage_tune_expensive_variant_cap", 40)))
    if tune_window < 20:
        return {"enabled": False, "reason": "coverage_tune_insufficient_history"}
    lambda_grid = [0.05, 0.10, 0.15, 0.20]
    mu_grid = [0.02, 0.05, 0.10]
    scores = {(float(lam), float(mu)): 0 for lam in lambda_grid for mu in mu_grid}
    start = len(blue_history) - tune_window
    # Keep this search intentionally shallow; it runs once for current output and
    # preserves the project's roughly 30 minute target runtime.
    for target_index in range(start, len(blue_history)):
        prefix = _slice_history(history, target_index)
        probability, _ = _direct_variant_probability(prefix, tune_variant, {
            **upgrade_config,
            "coverage_param_online_tune": False,
        })
        if probability is None:
            continue
        actual = int(blue_history[target_index])
        for lam in lambda_grid:
            for mu in mu_grid:
                picks, _ = _coverage_optimized_blues(
                    probability,
                    count,
                    lambda_value=lam,
                    mu_value=mu,
                )
                scores[(float(lam), float(mu))] += int(actual in set(picks))
    best_pair, best_hits = max(scores.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
    return {
        "enabled": True,
        "window": int(tune_window),
        "lambda": float(best_pair[0]),
        "mu": float(best_pair[1]),
        "hits": int(best_hits),
        "hit_ratio": float(best_hits / tune_window),
        "proxy_variant": tune_variant,
        "grid": "lambda={0.05,0.10,0.15,0.20};mu={0.02,0.05,0.10}",
    }


def _compact_reason(reason: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        key: reason[key]
        for key in ("mode", "kappa", "last_state", "row_count", "alpha_hedge", "regime_confidence", "eta_last", "eta_mean")
        if key in reason
    }
    meta_weights = reason.get("meta_weights")
    if isinstance(meta_weights, dict):
        compact["meta_weights"] = {str(key): float(value) for key, value in meta_weights.items()}
    hedge = reason.get("hedge")
    if isinstance(hedge, dict):
        weights = hedge.get("weights", {})
        if isinstance(weights, dict):
            compact["hedge_top_weights"] = dict(sorted(
                ((str(name), float(value)) for name, value in weights.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:4])
        compact["hedge_updates"] = hedge.get("updates")
    hmm = reason.get("hmm")
    if isinstance(hmm, dict):
        compact["hmm"] = {
            key: hmm[key]
            for key in ("mode", "selected_states", "bic", "state_posterior", "change_points", "observations", "full_window")
            if key in hmm
        }
    coverage = reason.get("coverage")
    if isinstance(coverage, dict):
        compact["coverage"] = {
            key: coverage[key]
            for key in ("objective", "score", "lambda", "mu", "signal_strength")
            if key in coverage
        }
    return compact


def _portfolio_sources(
    blue_history: np.ndarray,
    variant: str,
) -> list[tuple[str, np.ndarray, int]]:
    source_factories = {
        "markov1": _markov,
        "bayes_markov": lambda values: _bayesian_shrinkage_markov(values)[0],
        "markov2": lambda values: _markov_order(values, 2),
        "markov3": lambda values: _markov_order(values, 3),
        "vmm_bma": _variable_markov,
        "time_decay": _time_decay,
        "gap_hazard": _gap_hazard,
        "gap_shape": _gap_shape,
        "change_point_frequency": _change_point_frequency,
        "periodic_phase": _periodic_phase_mixture,
        "frequency_300": lambda values: _frequency(values, 300),
        "frequency_all": lambda values: _frequency(values, None),
        "neighbor_transition": _neighbor_transition,
        "frequency_100": lambda values: _frequency(values, 100),
    }
    specs: dict[str, list[tuple[str, int]]] = {
        "union_top3": [
            ("markov1", 3),
            ("time_decay", 3),
            ("gap_hazard", 3),
            ("frequency_300", 3),
            ("frequency_all", 3),
            ("neighbor_transition", 3),
            ("frequency_100", 3),
        ],
        "stable_frequency_transition": [
            ("vmm_bma", 1),
            ("markov1", 1),
            ("frequency_300", 2),
            ("frequency_all", 2),
            ("time_decay", 1),
            ("gap_hazard", 1),
            ("neighbor_transition", 1),
            ("frequency_100", 1),
        ],
        "recent_frequency_transition": [
            ("frequency_300", 2),
            ("vmm_bma", 1),
            ("markov1", 1),
            ("time_decay", 1),
            ("frequency_all", 2),
            ("gap_hazard", 1),
            ("neighbor_transition", 1),
        ],
        "high_order_markov": [("vmm_bma", 3), ("markov2", 2), ("markov3", 1)],
        "bayes_markov_blend": [("bayes_markov", 3), ("frequency_300", 1), ("gap_hazard", 1)],
        "change_point_periodic": [
            ("change_point_frequency", 2),
            ("periodic_phase", 2),
            ("gap_shape", 1),
            ("time_decay", 1),
        ],
        "gap_shape_only": [("gap_shape", 5)],
        "periodic_only": [("periodic_phase", 5)],
        "change_point_only": [("change_point_frequency", 5)],
        "frequency_300_only": [("frequency_300", 5)],
        "decay_only": [("time_decay", 5)],
        "markov_only": [("markov1", 5)],
        "gap_only": [("gap_hazard", 5)],
    }
    selected_spec = specs.get(variant, specs["stable_frequency_transition"])
    return [
        (source, source_factories[source](blue_history), quota)
        for source, quota in selected_spec
    ]


def _stateless_blue_variant_window_results(
    history: pd.DataFrame | Iterable[Any],
    count: int,
    windows: list[int],
    *,
    variant: str,
    baseline: float,
    min_edge: float,
    max_p_value: float,
    upgrade: dict[str, Any],
) -> list[dict[str, Any]]:
    blue_history = _blue_array(history)
    if not windows or len(blue_history) < 2:
        return []
    max_window = max(int(window) for window in windows)
    start = max(1, len(blue_history) - max_window)
    hits = {int(window): 0 for window in windows}
    top1_hits = {int(window): 0 for window in windows}
    mrr = {int(window): 0.0 for window in windows}
    ndcg = {int(window): 0.0 for window in windows}
    rank = {int(window): 0.0 for window in windows}
    research_upgrade = {
        **upgrade,
        "coverage_param_online_tune": False,
        "coverage_tune_current_output": False,
        "blue_no_edge_adaptive_history_policy": False,
    }
    for target_index in range(start, len(blue_history)):
        prefix = blue_history[:target_index]
        portfolio = blue_specialist_portfolio(
            prefix,
            count,
            variant=variant,
            upgrade_config=research_upgrade,
        )
        picks = [
            int(item["blue"])
            for item in portfolio
            if 1 <= int(item.get("blue", 0) or 0) <= 16
        ]
        probability, _ = _direct_variant_probability(prefix, variant, research_upgrade)
        if probability is None:
            if picks:
                scores = np.ones(16, dtype=float)
                for offset, blue in enumerate(picks):
                    scores[int(blue) - 1] += float(len(picks) - offset)
                probability = _normalize(scores)
            else:
                probability = _frequency(prefix, 300)
        actual_blue = int(blue_history[target_index])
        selected_set = set(picks)
        ranking_detail = _blue_ranking_detail(probability, actual_blue)
        for window in windows:
            if target_index >= len(blue_history) - int(window):
                hits[int(window)] += int(actual_blue in selected_set)
                top1_hits[int(window)] += int(bool(picks) and actual_blue == int(picks[0]))
                mrr[int(window)] += float(ranking_detail["mrr"])
                ndcg[int(window)] += float(ranking_detail["ndcg_at_5"])
                rank[int(window)] += float(ranking_detail["mean_rank"])
    results: list[dict[str, Any]] = []
    for window in windows:
        window = int(window)
        hit_count = int(hits[window])
        hit_ratio = float(hit_count / window)
        top1_count = int(top1_hits[window])
        top1_hit_ratio = float(top1_count / window)
        p_value = float(binomtest(hit_count, window, baseline, alternative="greater").pvalue)
        top1_p_value = float(binomtest(top1_count, window, 1.0 / 16.0, alternative="greater").pvalue)
        results.append({
            "window": window,
            "hits": hit_count,
            "hit_ratio": hit_ratio,
            "edge": float(hit_ratio - baseline),
            "p_value": p_value,
            "passed": bool(hit_ratio >= baseline + min_edge and p_value <= max_p_value),
            "top1_hits": top1_count,
            "top1_hit_ratio": top1_hit_ratio,
            "top1_edge": float(top1_hit_ratio - 1.0 / 16.0),
            "top1_p_value": top1_p_value,
            "top1_passed": bool(
                top1_hit_ratio >= (1.0 / 16.0) + min_edge
                and top1_p_value <= max_p_value
            ),
            "mrr": float(mrr[window] / window),
            "ndcg_at_5": float(ndcg[window] / window),
            "mean_rank": float(rank[window] / window),
            "shared_prefix_evaluation": True,
        })
    return results


def blue_research_evaluation(
    history: pd.DataFrame | Iterable[Any],
    count: int,
    *,
    windows: Iterable[int] = (100, 300, 500, 1000),
    min_edge: float = 0.02,
    max_p_value: float = 0.05,
    allow_stable_weak_edge: bool = True,
    variants: Iterable[str] | None = None,
    upgrade_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    upgrade = _upgrade_options(upgrade_config)
    blue_history = _blue_array(history)
    count = max(1, min(int(count), 16))
    baseline = float(count / 16.0)
    default_variants = [
        "online_hybrid_coverage",
        "nested_hedge",
        "online_hedge",
        "high_order_markov",
        "bayes_markov_blend",
        "frequency_300_only",
        "stable_frequency_transition",
        "union_top3",
    ]
    variants = [str(value) for value in (variants or default_variants)]
    # 方案2.5：简化模式下强制只评估 online_hedge，避免研究门选择完整模式variant导致简化失效
    if bool(upgrade.get("simplified_blue_experts", False)):
        variants = ["online_hedge"]
    unknown_variants = [variant for variant in variants if variant not in BLUE_SPECIALIST_VARIANTS]
    if unknown_variants:
        raise ValueError(f"blue_specialist_research_variants 存在未知策略: {unknown_variants}")
    effective_windows = [
        int(window)
        for window in windows
        if 0 < int(window) < len(blue_history)
    ]
    rows: list[dict[str, Any]] = []
    stateful_variants = [
        variant
        for variant in variants
        if variant in {"online_hedge", "online_hybrid_coverage", "nested_hedge", "hmm_regime"}
    ]
    stateful_results_by_variant = (
        _stateful_blue_online_family_window_results(
            history,
            count,
            effective_windows,
            variants=stateful_variants,
            baseline=baseline,
            min_edge=min_edge,
            max_p_value=max_p_value,
            upgrade_config=upgrade,
        )
        if stateful_variants else {}
    )
    for variant in variants:
        if variant in {"online_hedge", "online_hybrid_coverage", "nested_hedge", "hmm_regime"}:
            window_results = stateful_results_by_variant.get(variant, [])
        else:
            window_results = _stateless_blue_variant_window_results(
                history,
                count,
                effective_windows,
                variant=variant,
                baseline=baseline,
                min_edge=min_edge,
                max_p_value=max_p_value,
                upgrade=upgrade,
            )
        passed_all = bool(window_results and all(item["passed"] for item in window_results))
        stable_non_negative = bool(
            window_results
            and all(float(item["hit_ratio"]) >= baseline for item in window_results)
        )
        score = float(
            0.40 * np.mean([item["edge"] for item in window_results])
            + 0.25 * np.mean([item.get("top1_edge", 0.0) for item in window_results])
            + 0.25 * (np.mean([item.get("mrr", 0.0) for item in window_results]) - (1.0 / 8.5))
            + 0.10 * (np.mean([item.get("ndcg_at_5", 0.0) for item in window_results]) - 0.25)
            - 0.02 * sum(1 for item in window_results if item["hit_ratio"] < baseline)
        ) if window_results else -1.0
        rows.append({
            "variant": variant,
            "passed": passed_all,
            "stable_non_negative": stable_non_negative,
            "score": score,
            "windows": window_results,
            "current_portfolio": blue_specialist_portfolio(
                history,
                count,
                variant=variant,
                upgrade_config=upgrade,
            ),
        })
    best = max(
        rows,
        key=lambda item: (
            bool(item["passed"]),
            bool(item["stable_non_negative"]),
            float(item["score"]),
        ),
        default=None,
    )
    weak_approved = bool(
        best
        and bool(best.get("stable_non_negative"))
        and float(best.get("score", -1.0)) >= max(0.0, float(min_edge) * 0.75)
        and any(
            float(item.get("p_value", 1.0)) <= float(max_p_value)
            and float(item.get("edge", 0.0)) > 0.0
            for item in best.get("windows", [])
        )
    )
    approved = bool(best and (best.get("passed") or (allow_stable_weak_edge and weak_approved)))
    return {
        "approved": approved,
        "reason": (
            "blue_strategy_passed_multi_window_research_gate"
            if bool(best and best.get("passed"))
            else "blue_strategy_stable_weak_edge_research_gate"
            if approved
            else "no_blue_strategy_passed_multi_window_research_gate"
        ),
        "approval_level": (
            "strong" if bool(best and best.get("passed")) else "stable_weak" if approved else "rejected"
        ),
        "baseline": baseline,
        "count": count,
        "windows": effective_windows,
        "min_edge": float(min_edge),
        "max_p_value": float(max_p_value),
        "allow_stable_weak_edge": bool(allow_stable_weak_edge),
        "variants": variants,
        "selected_variant": best.get("variant") if best else None,
        "selected_portfolio": best.get("current_portfolio") if best else [],
        "rows": rows,
    }


def _stateful_blue_online_family_window_results(
    history: pd.DataFrame | Iterable[Any],
    count: int,
    windows: list[int],
    *,
    variants: list[str],
    baseline: float,
    min_edge: float,
    max_p_value: float,
    upgrade_config: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    upgrade = _upgrade_options(upgrade_config)
    requested = set(variants)
    if not requested:
        return {}
    blue_history = _blue_array(history)
    if not windows:
        return {variant: [] for variant in requested}
    minimum_start = max(1, int(len(blue_history) - max(windows)))
    # 方案2.5：简化模式下，replay_upgrade需要传递simplified_blue_experts
    simplified = bool(upgrade.get("simplified_blue_experts", False))
    replay_upgrade = {
        **upgrade,
        "expert_pool_extend": False,
        "cross_red_blue_feature": False,
        "simplified_blue_experts": simplified,
    }
    expert_names = list(_base_blue_experts(blue_history[:max(2, minimum_start)], replay_upgrade).keys())
    weights = np.full(len(expert_names), 1.0 / len(expert_names), dtype=float)
    warm_start = max(60, minimum_start - 180)
    update_count = 0
    cached_hmm_probability: np.ndarray | None = None
    cached_hmm_info: dict[str, Any] = {}
    # 方案2.5：简化模式使用固定学习率
    simplified_eta = float(upgrade.get("simplified_hedge_eta", 0.05))
    excluded_blues = {
        int(value)
        for value in (upgrade.get("excluded_blue_numbers", []) or [])
        if 1 <= int(value) <= 16
    }

    for target_index in range(warm_start, minimum_start):
        prefix = blue_history[:target_index]
        experts = _base_blue_experts(prefix, replay_upgrade)
        actual = int(blue_history[target_index]) - 1
        probabilities = np.stack([experts[name] for name in expert_names], axis=0)
        losses = -np.log(np.clip(probabilities[:, actual], 1e-12, 1.0))
        update_count += 1
        eta = simplified_eta if simplified else 0.16 / np.sqrt(max(update_count, 1))
        weights *= np.exp(-eta * np.clip(losses, 0.0, 30.0))
        weights = _normalize(weights)

    hits_by_variant = {
        variant: {int(window): 0 for window in windows}
        for variant in requested
    }
    top1_hits_by_variant = {
        variant: {int(window): 0 for window in windows}
        for variant in requested
    }
    mrr_by_variant = {
        variant: {int(window): 0.0 for window in windows}
        for variant in requested
    }
    ndcg_by_variant = {
        variant: {int(window): 0.0 for window in windows}
        for variant in requested
    }
    rank_by_variant = {
        variant: {int(window): 0.0 for window in windows}
        for variant in requested
    }
    for target_index in range(minimum_start, len(blue_history)):
        prefix_blue = blue_history[:target_index]
        experts = _base_blue_experts(prefix_blue, replay_upgrade)
        probabilities = np.stack([experts[name] for name in expert_names], axis=0)
        hedge_probability = _normalize(weights @ probabilities)
        # 方案2.5：简化模式下不需要HMM，所有变体直接使用hedge_probability
        needs_hmm = (not simplified) and bool({"online_hybrid_coverage", "nested_hedge", "hmm_regime"} & requested)
        if needs_hmm and (cached_hmm_probability is None or (target_index - minimum_start) % 20 == 0):
            cached_hmm_probability, cached_hmm_info = _hmm_regime_distribution(prefix_blue, upgrade)

        variant_probability: dict[str, np.ndarray] = {}
        if "online_hedge" in requested:
            variant_probability["online_hedge"] = hedge_probability
        # 方案2.5：简化模式下，需要HMM的变体直接使用hedge_probability
        if simplified:
            for hmm_variant in ("hmm_regime", "online_hybrid_coverage", "nested_hedge"):
                if hmm_variant in requested:
                    variant_probability[hmm_variant] = hedge_probability
        else:
            if "hmm_regime" in requested and cached_hmm_probability is not None:
                variant_probability["hmm_regime"] = cached_hmm_probability
            if "online_hybrid_coverage" in requested and cached_hmm_probability is not None:
                posterior = np.asarray(cached_hmm_info.get("state_posterior", []), dtype=float)
                regime_confidence = float(max(posterior, default=0.0) - 1.0 / max(len(posterior), 1))
                alpha = float(np.clip(0.82 - 0.42 * max(0.0, regime_confidence), 0.45, 0.88))
                variant_probability["online_hybrid_coverage"] = _normalize(
                    alpha * hedge_probability + (1.0 - alpha) * cached_hmm_probability
                )
            if "nested_hedge" in requested and cached_hmm_probability is not None:
                # In rolling research we use a lightweight outer weight proxy so the
                # full 100/300/500/1000 validation stays within the runtime budget.
                posterior = np.asarray(cached_hmm_info.get("state_posterior", []), dtype=float)
                confidence = float(max(posterior, default=0.0) - 1.0 / max(len(posterior), 1))
                hmm_share = float(np.clip(0.25 + 0.45 * max(0.0, confidence), 0.20, 0.70))
                variant_probability["nested_hedge"] = _normalize(
                    (1.0 - hmm_share) * hedge_probability + hmm_share * cached_hmm_probability
                )

        actual = int(blue_history[target_index])
        for variant, probability in variant_probability.items():
            selected_blues = [
                blue for blue in _order(probability)
                if blue not in excluded_blues
            ][:count]
            selected_set = set(selected_blues)
            top1_blue = int(selected_blues[0]) if selected_blues else None
            ranking_detail = _blue_ranking_detail(probability, actual)
            for window in windows:
                if target_index >= len(blue_history) - int(window):
                    hits_by_variant[variant][int(window)] += int(actual in selected_set)
                    top1_hits_by_variant[variant][int(window)] += int(
                        top1_blue is not None and actual == top1_blue
                    )
                    mrr_by_variant[variant][int(window)] += float(ranking_detail["mrr"])
                    ndcg_by_variant[variant][int(window)] += float(ranking_detail["ndcg_at_5"])
                    rank_by_variant[variant][int(window)] += float(ranking_detail["mean_rank"])

        actual_index = actual - 1
        losses = -np.log(np.clip(probabilities[:, actual_index], 1e-12, 1.0))
        update_count += 1
        eta = simplified_eta if simplified else 0.16 / np.sqrt(max(update_count, 1))
        weights *= np.exp(-eta * np.clip(losses, 0.0, 30.0))
        weights = _normalize(weights)

    results_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in requested:
        results: list[dict[str, Any]] = []
        for window in windows:
            hits = int(hits_by_variant.get(variant, {}).get(int(window), 0))
            hit_ratio = float(hits / int(window))
            p_value = float(binomtest(hits, int(window), baseline, alternative="greater").pvalue)
            top1_hits = int(top1_hits_by_variant.get(variant, {}).get(int(window), 0))
            top1_hit_ratio = float(top1_hits / int(window))
            top1_p_value = float(
                binomtest(top1_hits, int(window), 1.0 / 16.0, alternative="greater").pvalue
            )
            results.append({
                "window": int(window),
                "hits": hits,
                "hit_ratio": hit_ratio,
                "edge": float(hit_ratio - baseline),
                "p_value": p_value,
                "passed": bool(hit_ratio >= baseline + min_edge and p_value <= max_p_value),
                "top1_hits": top1_hits,
                "top1_hit_ratio": top1_hit_ratio,
                "top1_edge": float(top1_hit_ratio - 1.0 / 16.0),
                "top1_p_value": top1_p_value,
                "top1_passed": bool(
                    top1_hit_ratio >= (1.0 / 16.0) + min_edge
                    and top1_p_value <= max_p_value
                ),
                "mrr": float(mrr_by_variant.get(variant, {}).get(int(window), 0.0) / int(window)),
                "ndcg_at_5": float(ndcg_by_variant.get(variant, {}).get(int(window), 0.0) / int(window)),
                "mean_rank": float(rank_by_variant.get(variant, {}).get(int(window), 0.0) / int(window)),
                "stateful_evaluation": True,
                "shared_online_state": True,
            })
        results_by_variant[variant] = results
    return results_by_variant


def _consensus_blue_portfolio_from_research(
    research: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    """Blend researched blue strategies without increasing the number of picks.

    The research gate already performs rolling validation for each strategy.
    This second-stage consensus converts those validated rows into a weighted
    vote over the current 5 blue picks.  It avoids the brittle behavior of
    selecting exactly one strategy while still keeping the output count fixed.
    """
    count = max(1, min(int(count), 16))
    rows = [row for row in research.get("rows", []) if row.get("current_portfolio")]
    if not rows:
        return []
    baseline = float(research.get("baseline", count / 16.0) or (count / 16.0))
    vote: dict[int, float] = {}
    top1_vote: dict[int, float] = {}
    reasons: dict[int, list[dict[str, Any]]] = {}
    top1_reasons: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        windows = list(row.get("windows", []) or [])
        if not windows:
            continue
        edges = [float(item.get("edge", 0.0) or 0.0) for item in windows]
        hit_ratios = [float(item.get("hit_ratio", 0.0) or 0.0) for item in windows]
        top1_edges = [float(item.get("top1_edge", 0.0) or 0.0) for item in windows]
        top1_hit_ratios = [float(item.get("top1_hit_ratio", 0.0) or 0.0) for item in windows]
        positive_windows = sum(1 for edge in edges if edge >= 0.0)
        top1_positive_windows = sum(1 for edge in top1_edges if edge >= 0.0)
        significant_windows = sum(
            1
            for item in windows
            if float(item.get("p_value", 1.0) or 1.0) <= float(research.get("max_p_value", 0.20) or 0.20)
            and float(item.get("hit_ratio", 0.0) or 0.0) >= baseline
        )
        top1_significant_windows = sum(
            1
            for item in windows
            if float(item.get("top1_p_value", 1.0) or 1.0) <= float(research.get("max_p_value", 0.20) or 0.20)
            and float(item.get("top1_hit_ratio", 0.0) or 0.0) >= (1.0 / 16.0)
        )
        mean_edge = float(np.mean(edges)) if edges else -0.05
        recent_edge = float(edges[0]) if edges else mean_edge
        top1_mean_edge = float(np.mean(top1_edges)) if top1_edges else -0.02
        top1_recent_edge = float(top1_edges[0]) if top1_edges else top1_mean_edge
        stability = positive_windows / max(len(windows), 1)
        top1_stability = top1_positive_windows / max(len(windows), 1)
        row_weight = max(
            0.05,
            1.0
            + 9.5 * mean_edge
            + 2.6 * max(0.0, recent_edge)
            + 0.75 * stability
            + 0.45 * significant_windows
            + (0.60 if bool(row.get("passed")) else 0.0)
            + (0.25 if bool(row.get("stable_non_negative")) else -0.35)
            - 0.25 * sum(1 for ratio in hit_ratios if ratio < baseline)
        )
        top1_row_weight = max(
            0.03,
            1.0
            + 14.0 * top1_mean_edge
            + 3.2 * max(0.0, top1_recent_edge)
            + 0.90 * top1_stability
            + 0.65 * top1_significant_windows
            + (0.35 if top1_stability >= 0.75 else 0.0)
            - 0.30 * sum(1 for ratio in top1_hit_ratios if ratio < (1.0 / 16.0))
        )
        variant = str(row.get("variant", "unknown"))
        current_portfolio = list(row.get("current_portfolio") or [])[:count]
        top1_has_usable_signal = bool(
            top1_significant_windows > 0
            or top1_recent_edge > 0.0
            or (
                top1_mean_edge >= -0.005
                and top1_stability >= 0.50
            )
        )
        if current_portfolio and top1_has_usable_signal:
            try:
                first_blue = int(current_portfolio[0].get("blue"))
            except (TypeError, ValueError):
                first_blue = 0
            if 1 <= first_blue <= 16:
                first_score = float(current_portfolio[0].get("score", 0.0) or 0.0)
                contribution = float(top1_row_weight * (1.0 + 0.35 * first_score))
                top1_vote[first_blue] = top1_vote.get(first_blue, 0.0) + contribution
                top1_reasons.setdefault(first_blue, []).append({
                    "variant": variant,
                    "rank": 1,
                    "row_weight": float(top1_row_weight),
                    "score": first_score,
                    "top1_mean_edge": top1_mean_edge,
                    "top1_recent_edge": top1_recent_edge,
                })
        for rank, blue_item in enumerate(current_portfolio, 1):
            try:
                blue = int(blue_item.get("blue"))
            except (TypeError, ValueError):
                continue
            if not 1 <= blue <= 16:
                continue
            item_score = float(blue_item.get("score", 0.0) or 0.0)
            contribution = float(row_weight * (1.0 / rank + 0.35 * item_score))
            vote[blue] = vote.get(blue, 0.0) + contribution
            reasons.setdefault(blue, []).append({
                "variant": variant,
                "rank": int(rank),
                "row_weight": float(row_weight),
                "score": item_score,
                "mean_edge": mean_edge,
                "recent_edge": recent_edge,
            })
    if not vote:
        return []
    diagnostic_top1_anchor = None
    if top1_vote:
        diagnostic_top1_anchor = max(top1_vote, key=lambda blue: (top1_vote[blue], vote.get(blue, 0.0), -blue))
    ordered = [
        blue
        for blue in sorted(vote, key=lambda blue: (vote[blue], -blue), reverse=True)
    ]
    ordered = ordered[:count]
    return [
        {
            "blue": int(blue),
            "source": "validated_research_consensus",
            "score": float(vote.get(blue, 0.0)),
            "reason": {
                "mode": "prediction_set_hit5_validated_strategy_vote",
                "selected_variant": research.get("selected_variant"),
                "primary_objective": "Prediction Set + Hit@5 coverage; Top1 anchor is diagnostic only.",
                "diagnostic_top1_anchor": int(diagnostic_top1_anchor) if diagnostic_top1_anchor is not None else None,
                "top1_contributors": sorted(
                    top1_reasons.get(blue, []),
                    key=lambda item: float(item.get("row_weight", 0.0)),
                    reverse=True,
                )[:3],
                "contributors": sorted(
                    reasons.get(blue, []),
                    key=lambda item: float(item.get("row_weight", 0.0)),
                    reverse=True,
                )[:3],
            },
        }
        for blue in ordered
    ]


def _replace_excluded_blue_portfolio(
    portfolio: list[dict[str, Any]],
    history: pd.DataFrame | Iterable[Any],
    *,
    count: int,
    variant: str,
    excluded_blues: set[int],
    upgrade_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not excluded_blues:
        return portfolio[:count]
    kept = [
        item for item in portfolio
        if int(item.get("blue", 0)) not in excluded_blues
    ]
    seen = {int(item.get("blue")) for item in kept}
    if len(kept) >= int(count):
        return kept[:count]
    fallback_upgrade = {
        **_upgrade_options(upgrade_config),
        "excluded_blue_numbers": [],
    }
    for item in blue_specialist_portfolio(
        history,
        16,
        variant=variant,
        upgrade_config=fallback_upgrade,
    ):
        blue = int(item.get("blue", 0))
        if blue in excluded_blues or blue in seen or not 1 <= blue <= 16:
            continue
        kept.append({
            **item,
            "source": f"{item.get('source', variant)}_exclusion_replacement",
            "reason": {
                **(item.get("reason") if isinstance(item.get("reason"), dict) else {}),
                "excluded_blue_numbers": sorted(excluded_blues),
                "replacement_reason": "avoid_low_confidence_excluded_blue",
            },
        })
        seen.add(blue)
        if len(kept) >= int(count):
            break
    return kept[:count]


def _prediction_set_allowed_blues(
    probability: np.ndarray,
    upgrade_config: dict[str, Any],
    count: int,
    *,
    excluded_blues: set[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Return the production blue candidate set controlled by V7 Prediction Set."""
    excluded = set(excluded_blues or set())
    prediction_set = upgrade_config.get("_v7_blue_prediction_set")
    policy = upgrade_config.get("_v7_blue_topk_policy")
    selected_width = None
    if isinstance(prediction_set, dict):
        selected_width = prediction_set.get("selected_width")
        if not isinstance(policy, dict):
            policy = prediction_set.get("policy")
    if selected_width is None and isinstance(policy, dict):
        selected_width = policy.get("selected_width")
        if selected_width is None:
            topk = policy.get("candidate_topk")
            if isinstance(topk, list) and topk:
                selected_width = max(int(value) for value in topk)
    configured_width = int(upgrade_config.get("blue_coverage_candidate_top_n", 10))
    width = int(selected_width or configured_width)
    width = max(1, min(16, width))
    # Production output still needs distinct blue coverage across the purchased
    # tickets, therefore the candidate set cannot be narrower than ticket count.
    width = max(width, min(int(count), 16))
    ranking = [
        int(index + 1)
        for index in np.argsort(_normalize(probability))[::-1]
        if int(index + 1) not in excluded
    ]
    allowed = ranking[:width]
    return allowed, {
        "schema": "blue_production_prediction_set_v7",
        "active": bool(isinstance(prediction_set, dict) or isinstance(policy, dict)),
        "selected_width": int(width),
        "raw_selected_width": int(selected_width or configured_width),
        "reason": "V7 Prediction Set controls the actual production blue candidate pool.",
        "allowed_blues": allowed,
        "excluded_blue_numbers": sorted(excluded),
        "policy": policy if isinstance(policy, dict) else {},
    }


def _apply_blue_prediction_set(
    selected_blues: list[int],
    probability: np.ndarray,
    count: int,
    upgrade_config: dict[str, Any],
    *,
    excluded_blues: set[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    allowed, info = _prediction_set_allowed_blues(
        probability,
        upgrade_config,
        count,
        excluded_blues=excluded_blues,
    )
    excluded = set(excluded_blues or set())
    result: list[int] = []
    allowed_set = set(allowed)
    for blue in selected_blues:
        blue = int(blue)
        if blue in excluded or blue not in allowed_set or blue in result:
            continue
        result.append(blue)
        if len(result) >= int(count):
            break
    for blue in allowed:
        if blue in result:
            continue
        result.append(int(blue))
        if len(result) >= int(count):
            break
    expanded = False
    if len(result) < int(count):
        expanded = True
        for blue in [
            int(index + 1)
            for index in np.argsort(_normalize(probability))[::-1]
            if int(index + 1) not in excluded
        ]:
            if blue in result:
                continue
            result.append(blue)
            if len(result) >= int(count):
                break
    return result[: int(count)], {
        **info,
        "requested_count": int(count),
        "final_blues": result[: int(count)],
        "expanded_outside_prediction_set": bool(expanded),
    }


def apply_blue_specialist_portfolio(
    tickets: list[dict[str, Any]],
    history: pd.DataFrame | Iterable[Any],
    *,
    require_research_edge: bool = True,
    windows: Iterable[int] = (100, 300, 500, 1000),
    min_edge: float = 0.02,
    max_p_value: float = 0.05,
    allow_stable_weak_edge: bool = True,
    variants: Iterable[str] | None = None,
    upgrade_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not tickets:
        return []
    _upgrade = _upgrade_options(upgrade_config)
    excluded_blues = {
        int(value)
        for value in (_upgrade.get("excluded_blue_numbers", []) or [])
        if 1 <= int(value) <= 16
    }
    if str(_upgrade.get("blue_mode")) == "uniform_coverage":
        uniform_probability = np.full(16, 1.0 / 16.0, dtype=float)
        selected_blues, coverage_info = _policy_optimized_blues(
            history,
            uniform_probability,
            len(tickets),
            _upgrade,
            excluded_blues=excluded_blues,
        )
        selected_blues, prediction_set_info = _apply_blue_prediction_set(
            selected_blues,
            uniform_probability,
            len(tickets),
            _upgrade,
            excluded_blues=excluded_blues,
        )
        selected_blues = [
            blue for blue in selected_blues
            if int(blue) not in excluded_blues
        ]
        if len(selected_blues) < len(tickets):
            selected_blues.extend(
                blue for blue in range(1, 17)
                if blue not in excluded_blues and blue not in set(selected_blues)
            )
        adjusted: list[dict[str, Any]] = []
        for ticket, blue in zip(tickets, selected_blues[:len(tickets)]):
            adjusted.append({
                **ticket,
                "blue": int(blue),
                "blue_selection_role": "blue_uniform_coverage_mode",
                "blue_selection_reason": {
                    **coverage_info,
                    "prediction_set": prediction_set_info,
                },
            })
        if len(tickets) > len(adjusted):
            adjusted.extend(tickets[len(adjusted):])
        return adjusted
    precomputed_research = _upgrade.get("_precomputed_blue_research")
    research = (
        precomputed_research
        if isinstance(precomputed_research, dict)
        else blue_research_evaluation(
            history,
            len(tickets),
            windows=windows,
            min_edge=min_edge,
            max_p_value=max_p_value,
            allow_stable_weak_edge=allow_stable_weak_edge,
            variants=variants,
            upgrade_config=upgrade_config,
        )
    )
    if require_research_edge and not bool(research.get("approved")):
        model_probability, shrinkage_info = _shrunk_blue_probability(history, research, _upgrade)
        selected_blues, coverage_info = _policy_optimized_blues(
            history,
            model_probability,
            len(tickets),
            _upgrade,
            excluded_blues=excluded_blues,
        )
        selected_blues, prediction_set_info = _apply_blue_prediction_set(
            selected_blues,
            model_probability,
            len(tickets),
            _upgrade,
            excluded_blues=excluded_blues,
        )
        selected_blues = [
            blue for blue in selected_blues
            if int(blue) not in excluded_blues
        ]
        if len(selected_blues) < len(tickets):
            selected_blues.extend(
                blue for blue in range(1, 17)
                if blue not in excluded_blues and blue not in set(selected_blues)
            )
        adjusted: list[dict[str, Any]] = []
        for ticket, blue in zip(tickets, selected_blues[:len(tickets)]):
            adjusted.append({
                **ticket,
                "blue": int(blue),
                "blue_selection_role": "blue_research_no_edge_shrunk_coverage",
                "blue_selection_reason": {
                    "research_gate": research,
                    "fallback": coverage_info,
                    "shrinkage": shrinkage_info,
                    "prediction_set": prediction_set_info,
                    "reason": "blue_research_no_edge_use_continuous_shrinkage_coverage",
                },
            })
        if len(tickets) > len(adjusted):
            adjusted.extend(tickets[len(adjusted):])
        return adjusted
    # 方案2.5：简化模式下默认使用 online_hedge，避免回退到完整模式variant
    _default_variant = "online_hedge" if bool(_upgrade.get("simplified_blue_experts", False)) else "stable_frequency_transition"
    variant = str(research.get("selected_variant") or _default_variant)
    portfolio = (
        _consensus_blue_portfolio_from_research(research, len(tickets))
        if bool(_upgrade.get("validated_blue_consensus", True))
        else []
    )
    if not portfolio:
        portfolio = blue_specialist_portfolio(
            history,
            len(tickets),
            variant=variant,
            upgrade_config=upgrade_config,
        )
    portfolio = _replace_excluded_blue_portfolio(
        list(portfolio),
        history,
        count=len(tickets),
        variant=variant,
        excluded_blues=excluded_blues,
        upgrade_config=upgrade_config,
    )
    if not portfolio:
        return tickets
    model_probability, probability_info = _shrunk_blue_probability(history, research, _upgrade)
    selected_blues, prediction_set_info = _apply_blue_prediction_set(
        [int(item["blue"]) for item in portfolio if 1 <= int(item.get("blue", 0)) <= 16],
        model_probability,
        len(tickets),
        _upgrade,
        excluded_blues=excluded_blues,
    )
    portfolio_by_blue = {int(item.get("blue", 0)): item for item in portfolio if 1 <= int(item.get("blue", 0)) <= 16}
    adjusted: list[dict[str, Any]] = []
    for ticket, blue in zip(tickets, selected_blues):
        blue_item = portfolio_by_blue.get(int(blue), {
            "blue": int(blue),
            "source": "v7_prediction_set_fill",
            "reason": {
                "probability_rank_fill": True,
                "model_probability": probability_info,
            },
        })
        adjusted.append({
            **ticket,
            "blue": int(blue_item["blue"]),
            "blue_selection_role": "blue_specialist_portfolio",
            "blue_selection_reason": {
                **blue_item,
                "research_gate": research,
                "prediction_set": prediction_set_info,
            },
        })
    if len(tickets) > len(adjusted):
        adjusted.extend(tickets[len(adjusted):])
    return adjusted


def finalize_blue_portfolio(
    tickets: list[dict[str, Any]],
    history: pd.DataFrame | Iterable[Any],
    *,
    optimizer_config: dict[str, Any],
    upgrade_config: dict[str, Any] | None = None,
    exclusion_diagnostics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single final blue path used by both target prediction and backtest.

    The old configuration kept separate switches for production and
    walk-forward.  This wrapper makes the portfolio step explicit and shared;
    callers differ only in diagnostics/logging, not in strategy logic.
    """
    if not bool(optimizer_config.get("blue_specialist_portfolio_enabled", True)):
        return tickets, {
            "active": False,
            "reason": "blue_specialist_portfolio_disabled",
            "blue_path": "shared_finalize_blue_portfolio",
        }
    started = time.perf_counter()
    exclusion_diagnostics = exclusion_diagnostics or {}
    windows = optimizer_config.get(
        "blue_specialist_research_windows", [100, 300, 500, 1000]
    )
    variants = optimizer_config.get("blue_specialist_research_variants")
    require_research_edge = bool(
        optimizer_config.get("blue_specialist_require_research_edge", True)
    )
    min_edge = float(optimizer_config.get("blue_specialist_min_edge", 0.0))
    max_p_value = float(optimizer_config.get("blue_specialist_max_p_value", 0.2))
    allow_stable_weak_edge = bool(
        optimizer_config.get("blue_specialist_allow_stable_weak_edge", True)
    )
    effective_upgrade = {
        **(upgrade_config or {}),
        "excluded_blue_numbers": exclusion_diagnostics.get("blue_excluded", []),
    }
    exposure_enabled = bool(effective_upgrade.get("blue_no_edge_cross_period_exposure_enabled", True))
    exposure_window = max(1, int(effective_upgrade.get("blue_no_edge_exposure_window", 30) or 30))
    exposure_history = optimizer_config.setdefault("_blue_final_exposure_history", []) if exposure_enabled else []
    recent_exposure_values = [
        int(value)
        for value in list(exposure_history)[-exposure_window:]
        if 1 <= int(value) <= 16
    ]
    recent_exposure = dict(Counter(recent_exposure_values))
    effective_upgrade["_recent_final_blue_exposure"] = recent_exposure
    effective_upgrade["_blue_final_exposure_window"] = exposure_window
    research_cache_hit = False
    cache_context = str(optimizer_config.get("_blue_research_cache_context", "production")).strip().lower()
    if cache_context in {"backtest", "research"}:
        cache_enabled = bool(
            optimizer_config.get(
                "blue_specialist_research_cache_enabled_for_backtest",
                optimizer_config.get("blue_specialist_research_cache_enabled", True),
            )
        )
    else:
        cache_enabled = bool(optimizer_config.get("blue_specialist_research_cache_enabled", False))
    cache_interval = max(1, int(optimizer_config.get("blue_specialist_research_cache_interval", 20)))
    blue_values = _blue_array(history)
    history_length = int(len(blue_values))
    latest_period = (
        str(history.iloc[-1]["period"])
        if isinstance(history, pd.DataFrame) and len(history) and "period" in history.columns
        else str(history_length)
    )
    v7_runtime = (
        effective_upgrade.get("_v7_runtime_control")
        if isinstance(effective_upgrade.get("_v7_runtime_control"), dict)
        else {}
    )
    v7_blue_edge = (
        effective_upgrade.get("_v7_blue_edge_report")
        if isinstance(effective_upgrade.get("_v7_blue_edge_report"), dict)
        else {}
    )
    blue_edge_state = str(
        v7_runtime.get("blue_edge_state")
        or v7_blue_edge.get("final_state")
        or v7_blue_edge.get("edge_state")
        or "NO_EDGE"
    ).strip().upper()
    blue_policy = str(effective_upgrade.get("blue_coverage_policy", "")).strip().lower()
    require_strong_edge_for_research = bool(
        effective_upgrade.get("blue_require_strong_edge_for_research", True)
    )
    coverage_first_runtime = (
        blue_policy in {"coverage_only", "no_edge_coverage_only", "weak_edge_mixed_coverage"}
        and bool(effective_upgrade.get("blue_no_edge_fast_runtime_path", True))
        and (
            blue_edge_state == "NO_EDGE"
            or (blue_policy == "weak_edge_mixed_coverage" and blue_edge_state == "WEAK_EDGE")
            or (require_strong_edge_for_research and blue_edge_state == "WEAK_EDGE")
        )
    )
    if coverage_first_runtime:
        runtime_variant = (
            "v7_no_edge_coverage_only"
            if blue_edge_state == "NO_EDGE"
            else "v7_weak_edge_mixed_coverage"
            if blue_edge_state == "WEAK_EDGE"
            else "v7_coverage_first_runtime"
        )
        research = {
            "schema": "blue_research_evaluation_v7_runtime_skip",
            "approved": False,
            "approval_level": "rejected",
            "selected_variant": runtime_variant,
            "reason": (
                "V7运行时未确认蓝球强Edge，生产路径执行Edge/Coverage双模式："
                "NO_EDGE=Coverage Only，WEAK_EDGE=弱概率+覆盖混合，STRONG_EDGE才允许概率研究路径。"
            ),
            "edge_state": blue_edge_state,
            "policy": blue_policy,
            "require_strong_edge_for_research": require_strong_edge_for_research,
            "rows": [],
            "selected_portfolio": [],
            "cache": {
                "hit": False,
                "skipped": True,
                "history_length": history_length,
                "latest_period": latest_period,
            },
        }
        adjusted = apply_blue_specialist_portfolio(
            tickets,
            history,
            require_research_edge=True,
            windows=windows,
            variants=variants,
            min_edge=min_edge,
            max_p_value=max_p_value,
            allow_stable_weak_edge=allow_stable_weak_edge,
            upgrade_config={
                **effective_upgrade,
                "_precomputed_blue_research": research,
            },
        )
        final_blues = [
            int(ticket.get("blue", 0) or 0)
            for ticket in adjusted
            if 1 <= int(ticket.get("blue", 0) or 0) <= 16
        ]
        if exposure_enabled:
            exposure_history.extend(final_blues)
            max_items = max(exposure_window * 6, 120)
            del exposure_history[:-max_items]
        return adjusted, {
            "active": True,
            "reason": "shared_finalize_blue_portfolio_applied",
            "blue_path": "shared_finalize_blue_portfolio",
            "runtime_fast_path": runtime_variant,
            "blue_edge_coverage_mode": (
                "NO_EDGE→Coverage Only"
                if blue_edge_state == "NO_EDGE"
                else "WEAK_EDGE→Mixed Coverage"
                if blue_edge_state == "WEAK_EDGE"
                else "STRONG_EDGE→Probability"
            ),
            "research_skipped": True,
            "research_skip_reason": research["reason"],
            "v7_blue_prediction_set": (upgrade_config or {}).get("_v7_blue_prediction_set"),
            "v7_blue_topk_policy": (upgrade_config or {}).get("_v7_blue_topk_policy"),
            "v7_blue_edge_report": (upgrade_config or {}).get("_v7_blue_edge_report"),
            "v7_runtime_control": (upgrade_config or {}).get("_v7_runtime_control"),
            "research_cache_hit": False,
            "research_cache_interval": int(cache_interval),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "blue_edge_state": blue_edge_state,
            "blues": final_blues,
            "unique_blues": sorted({int(ticket.get("blue", 0)) for ticket in adjusted if int(ticket.get("blue", 0))}),
            "cross_period_exposure": {
                "enabled": bool(exposure_enabled),
                "window": int(exposure_window),
                "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
                "rule": "NO_EDGE下蓝球Final 5加入跨期曝光惩罚，降低长期重复集中。",
            },
        }
    cache_key = (
        int(len(tickets)),
        int(history_length // cache_interval),
        tuple(int(value) for value in windows),
        tuple(str(value) for value in (variants or [])),
        bool(require_research_edge),
        float(min_edge),
        float(max_p_value),
        bool(allow_stable_weak_edge),
        bool((upgrade_config or {}).get("simplified_blue_experts", False)),
        str((upgrade_config or {}).get("blue_coverage_policy", "")),
        str(blue_edge_state),
        tuple(int(value) for value in exclusion_diagnostics.get("blue_excluded", []) or []),
    )
    cache = optimizer_config.setdefault("_blue_specialist_research_cache", {}) if cache_enabled else {}
    cached_research = cache.get(cache_key) if cache_enabled else None
    if isinstance(cached_research, dict):
        research = {
            **cached_research,
            "selected_portfolio": [],
            "rows": [
                {**row, "current_portfolio": []}
                for row in cached_research.get("rows", []) or []
                if isinstance(row, dict)
            ],
            "cache": {
                "hit": True,
                "interval": cache_interval,
                "history_length": history_length,
                "latest_period": latest_period,
            },
        }
        research_cache_hit = True
        adjusted = apply_blue_specialist_portfolio(
            tickets,
            history,
            require_research_edge=require_research_edge,
            windows=windows,
            variants=variants,
            min_edge=min_edge,
            max_p_value=max_p_value,
            allow_stable_weak_edge=allow_stable_weak_edge,
            upgrade_config={
                **effective_upgrade,
                "_precomputed_blue_research": research,
            },
        )
    else:
        adjusted = apply_blue_specialist_portfolio(
            tickets,
            history,
            require_research_edge=require_research_edge,
            windows=windows,
            variants=variants,
            min_edge=min_edge,
            max_p_value=max_p_value,
            allow_stable_weak_edge=allow_stable_weak_edge,
            upgrade_config=effective_upgrade,
        )
        if cache_enabled:
            stored_research = None
            for ticket in adjusted:
                reason = ticket.get("blue_selection_reason")
                if isinstance(reason, dict) and isinstance(reason.get("research_gate"), dict):
                    stored_research = reason["research_gate"]
                    break
            if isinstance(stored_research, dict):
                cache[cache_key] = {
                    **stored_research,
                    "selected_portfolio": [],
                    "rows": [
                        {**row, "current_portfolio": []}
                        for row in stored_research.get("rows", []) or []
                        if isinstance(row, dict)
                    ],
                    "cache": {
                        "hit": False,
                        "interval": cache_interval,
                        "history_length": history_length,
                        "latest_period": latest_period,
                    },
                }
    final_blues = [
        int(ticket.get("blue", 0) or 0)
        for ticket in adjusted
        if 1 <= int(ticket.get("blue", 0) or 0) <= 16
    ]
    if exposure_enabled:
        exposure_history.extend(final_blues)
        max_items = max(exposure_window * 6, 120)
        del exposure_history[:-max_items]
    return adjusted, {
        "active": True,
        "reason": "shared_finalize_blue_portfolio_applied",
        "blue_path": "shared_finalize_blue_portfolio",
        "v7_blue_prediction_set": (upgrade_config or {}).get("_v7_blue_prediction_set"),
        "v7_blue_topk_policy": (upgrade_config or {}).get("_v7_blue_topk_policy"),
        "v7_blue_edge_report": (upgrade_config or {}).get("_v7_blue_edge_report"),
        "v7_runtime_control": (upgrade_config or {}).get("_v7_runtime_control"),
        "research_cache_hit": bool(research_cache_hit),
        "research_cache_interval": int(cache_interval),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "blue_edge_state": blue_edge_state,
        "blues": final_blues,
        "unique_blues": sorted({int(ticket.get("blue", 0)) for ticket in adjusted if int(ticket.get("blue", 0))}),
        "cross_period_exposure": {
            "enabled": bool(exposure_enabled),
            "window": int(exposure_window),
            "recent_final_blue_exposure": {str(key): int(value) for key, value in sorted(recent_exposure.items())},
            "rule": "NO_EDGE下蓝球Final 5加入跨期曝光惩罚，降低长期重复集中。",
        },
    }
