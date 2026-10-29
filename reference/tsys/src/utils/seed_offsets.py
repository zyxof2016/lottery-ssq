from __future__ import annotations

SEED_OFFSETS = {
    # 候选池与保护池
    "monte_carlo": 0,
    "red_guard_monte_carlo": 80_000,
    "weak_signal_core": 70_000,
    "weak_signal_monte_carlo": 90_000,
    "uniform_coverage_pool": 180_000,
    "stoploss_pool": 280_000,
    "dantuo_random": 370_000,
    # 诊断 / 显著性
    "method_significance": 0,
    "pure_random_significance": 100_000,
    "best_red_significance": 200_000,
    "development_model_diagnostics": 410_000,
    "development_pipeline_diagnostics": 430_000,
    "confirmatory_model_diagnostics": 510_000,
    "confirmatory_pipeline_diagnostics": 530_000,
    "final_pipeline_diagnostics": 610_000,
    # 对照基线
    "random_baseline": 500_000,
    "consensus_random_baseline": 600_000,
    "matched_no_signal_pool": 2_000_000,
    "matched_consensus_pool": 2_070_000,
    "weak_release_pool": 2_170_000,
    "blue_diagnostic_color": 700_000,
    "blue_diagnostic_pipeline": 820_000,
    "matched_optimizer_pool": 3_000_000,
    "output_random_mean": 4_000_000,
    "method_optimizer": 4_100_000,
    "historical_hit_ablation": 4_180_000,
    "matched_historical_hit_ablation": 4_220_000,
    "final_diversity_rebalance": 4_300_000,
    "compression_experiment": 4_360_000,
    "research_benchmark": 4_400_000,
    # 当前期展示
    "uniform_current": 9_999,
}


def seed_with_offset(base_seed: int, name: str, *parts: int) -> int:
    if name not in SEED_OFFSETS:
        raise KeyError(f"未知随机种子偏移量: {name}")
    return int(base_seed) + int(SEED_OFFSETS[name]) + sum(int(part) for part in parts)
