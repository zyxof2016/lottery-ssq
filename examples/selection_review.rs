//! Read-only research: causal replay and equal-budget ticket construction comparisons.
//! This example never publishes, fits on the confirmation window, or updates archives.
use std::{collections::BTreeSet, path::Path};

use serde_json::{json, Value};
use ssq::{
    draw::{data_sha256, fmt_date, fmt_issue, load_csv, Draw},
    eval::prepare_forecast,
    model::roll_all,
    protocol::Protocol,
    review,
    select::{build_tickets, equal7, forecast, fuse7, rank_desc, top_mask, Forecast},
    stats::{expected_red_hits, mean_sd, wilson},
};

type Tickets = Vec<(u64, usize)>;
const METHODS: [&str; 4] = ["frozen", "balanced12", "pool15", "blue4"];

/// Four partitions and their complements give every candidate four exposures.
/// Both sides of a partition share a blue, so every red has two exposures per blue.
/// The fixed construction is independent of scores and outcomes once the pool is set.
fn balanced(fc: &Forecast) -> Tickets {
    assert_eq!(fc.pool.len(), 12);
    let partitions = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 6, 7, 8],
        [0, 3, 4, 6, 9, 10],
        [1, 3, 5, 7, 9, 11],
    ];
    let pool = fc.pool.iter().fold(0, |mask, &i| mask | (1 << i));
    partitions
        .iter()
        .enumerate()
        .flat_map(|(i, indices)| {
            let mask = indices.iter().fold(0, |mask, &j| mask | (1 << fc.pool[j]));
            let blue = fc.blue_order[i % 2];
            [(mask, blue), (pool ^ mask, blue)]
        })
        .collect()
}

fn ticket_sets(fc: &Forecast) -> [Tickets; 4] {
    [
        fc.tickets
            .iter()
            .copied()
            .zip(fc.ticket_blues.iter().copied())
            .collect(),
        balanced(fc),
        build_tickets(&fc.red_order[..15], &fc.fusion, 8, 4)
            .into_iter()
            .enumerate()
            .map(|(i, mask)| (mask, fc.blue_order[i % 2]))
            .collect(),
        fc.tickets
            .iter()
            .enumerate()
            .map(|(i, &mask)| (mask, fc.blue_order[i % 4]))
            .collect(),
    ]
}

fn numbers(mask: u64) -> Vec<u8> {
    (0..33)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| i + 1)
        .collect()
}

fn ticket_json(tickets: &Tickets) -> Value {
    json!(tickets
        .iter()
        .map(|&(red, blue)| json!({"red": numbers(red), "blue": blue + 1}))
        .collect::<Vec<_>>())
}

fn measure(tickets: &Tickets, draw: &Draw) -> Value {
    let union = tickets.iter().fold(0, |mask, (red, _)| mask | red);
    let blue_n = tickets
        .iter()
        .map(|(_, blue)| blue)
        .collect::<BTreeSet<_>>()
        .len();
    let hits: Vec<_> = tickets
        .iter()
        .map(|(red, _)| (red & draw.red_mask).count_ones())
        .collect();
    let max_exposure = (0..33)
        .map(|i| {
            tickets
                .iter()
                .filter(|(red, _)| red & (1 << i) != 0)
                .count()
        })
        .max()
        .unwrap();
    json!({
        "bets": tickets.len(), "cost": tickets.len() * 2,
        "red_per_bet": hits.iter().sum::<u32>() as f64 / tickets.len() as f64,
        "best_red": hits.iter().max(), "covered_red_n": union.count_ones(),
        "covered_red_hits": (union & draw.red_mask).count_ones(),
        "uniform_covered_red_hits": expected_red_hits(union.count_ones() as usize),
        "blue_covered": tickets.iter().any(|(_, blue)| *blue + 1 == draw.blue as usize),
        "blue_hit_bets": tickets.iter().filter(|(_, blue)| *blue + 1 == draw.blue as usize).count(),
        "uniform_blue_coverage": blue_n as f64 / 16.0,
        "max_red_exposure": max_exposure,
        "at_least_4_red": hits.iter().any(|h| *h >= 4),
    })
}

fn average(rows: &[&Value], method: &str, field: &str) -> f64 {
    rows.iter()
        .map(|r| r["methods"][method][field].as_f64().unwrap())
        .sum::<f64>()
        / rows.len() as f64
}

fn summarize(rows: &[&Value]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let mut methods = serde_json::Map::new();
    for name in METHODS {
        let blue_hits = rows
            .iter()
            .filter(|r| r["methods"][name]["blue_covered"] == true)
            .count();
        let blue_ci = wilson(blue_hits as f64, rows.len() as f64, 1.959963984540054);
        let diffs: Vec<_> = rows
            .iter()
            .map(|r| {
                r["methods"][name]["red_per_bet"].as_f64().unwrap()
                    - r["methods"]["frozen"]["red_per_bet"].as_f64().unwrap()
            })
            .collect();
        let (mean_diff, _, se) = mean_sd(&diffs);
        methods.insert(name.into(), json!({
            "mean_red_per_bet": average(rows, name, "red_per_bet"),
            "uniform_red_per_bet": expected_red_hits(6),
            "mean_best_red": average(rows, name, "best_red"),
            "mean_covered_red_n": average(rows, name, "covered_red_n"),
            "mean_covered_red_hits": average(rows, name, "covered_red_hits"),
            "uniform_covered_red_hits": average(rows, name, "uniform_covered_red_hits"),
            "mean_max_red_exposure": average(rows, name, "max_red_exposure"),
            "blue_covered_periods": blue_hits,
            "mean_blue_hit_bets": average(rows, name, "blue_hit_bets"),
            "uniform_blue_hit_bets": 8.0 / 16.0,
            "blue_coverage_wilson95": [blue_ci.0, blue_ci.1],
            "uniform_blue_covered_periods": average(rows, name, "uniform_blue_coverage") * rows.len() as f64,
            "at_least_4_red_periods": rows.iter().filter(|r| r["methods"][name]["at_least_4_red"] == true).count(),
            "paired_red_per_bet_minus_frozen": mean_diff,
            "paired_descriptive_iid_normal95": if rows.len() >= 30 {
                Some([mean_diff - 1.959963984540054 * se, mean_diff + 1.959963984540054 * se])
            } else { None },
        }));
    }
    json!({"n": rows.len(), "from": rows.first().unwrap()["issue"], "to": rows.last().unwrap()["issue"],
        "mean_fusion_top15_hits": rows.iter().map(|r| r["fusion_top15_hits"].as_f64().unwrap()).sum::<f64>() / rows.len() as f64,
        "mean_equal_top15_hits": rows.iter().map(|r| r["equal_top15_hits"].as_f64().unwrap()).sum::<f64>() / rows.len() as f64,
        "mean_final_set_hits": rows.iter().map(|r| r["final_set_hits"].as_f64().unwrap()).sum::<f64>() / rows.len() as f64,
        "mean_candidate_pool_hits": rows.iter().map(|r| r["candidate_pool_hits"].as_f64().unwrap()).sum::<f64>() / rows.len() as f64,
        "methods": methods})
}

fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let prepared = prepare_forecast(draws)?;
    let snapshot = roll_all(draws)?;
    let rows: Vec<_> = snapshot
        .rows
        .iter()
        .filter(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .collect();
    // These are exactly the fixed confirmation rows; later rows stay in the tracking window.
    if rows.len() < Protocol::CONFIRMATION_LEN || rows[0].issue != Protocol::CONFIRM_START_ISSUE {
        return Err("需要完整的冻结确认窗，不能将较早样本补入最近窗口".into());
    }
    let mut observations = Vec::new();
    for row in rows {
        let fc = forecast(
            &row.expert_p,
            &row.blue_p,
            &prepared.weights,
            &prepared.blue_weights,
        );
        let draw = draws
            .iter()
            .find(|d| d.issue == row.issue)
            .ok_or("缺少开奖")?;
        let mut methods = serde_json::Map::new();
        for (name, tickets) in METHODS.iter().zip(ticket_sets(&fc)) {
            if tickets.len() != 8 {
                return Err(format!("{} 的 {name} 不是 8 注，不能同预算比较", row.issue));
            }
            methods.insert((*name).into(), measure(&tickets, draw));
        }
        let pool_mask = fc.pool.iter().fold(0, |mask, &i| mask | (1 << i));
        let equal_order = rank_desc(&fuse7(&row.expert_p, &equal7()));
        observations.push(json!({
            "issue": fmt_issue(row.issue), "date": fmt_date(row.date), "methods": methods,
            "actual_red": draw.red, "actual_blue": draw.blue,
            "fusion_top15_hits": (top_mask(&fc.red_order, 15) & draw.red_mask).count_ones(),
            "equal_top15_hits": (top_mask(&equal_order, 15) & draw.red_mask).count_ones(),
            "candidate_pool_hits": (pool_mask & draw.red_mask).count_ones(),
            "final_set_hits": (fc.mild.mask & draw.red_mask).count_ones(),
            "frozen_diagnosis": review::explain(&fc, &draw.red, draw.blue),
        }));
    }
    let n = observations.len();
    let recent = |size: usize| {
        summarize(
            &observations[n.saturating_sub(size)..]
                .iter()
                .collect::<Vec<_>>(),
        )
    };
    let mut next = serde_json::Map::new();
    for (name, tickets) in METHODS.iter().zip(ticket_sets(&prepared.forecast)) {
        next.insert((*name).into(), ticket_json(&tickets));
    }
    Ok(json!({
        "kind": "exploratory_causal_replay", "data_sha256": data_sha256(draws),
        "protocol_sha256": Protocol::sha256(), "cutoff_issue": fmt_issue(prepared.cutoff_issue),
        "weight_cutoff_exclusive": fmt_issue(Protocol::CONFIRM_START_ISSUE),
        "red_weights": prepared.weights, "blue_weights": prepared.blue_weights,
        "notes": [
            "事后逐期回放，不是实际推荐或投注记录。每期专家状态只用该期之前的开奖。",
            "所有方案每期 8 注、16 元；均匀基准按实际红球并集和蓝球并集计算。",
            "balanced12 保留冻结候选池，每个红球出现 4 次，每个红蓝配对出现 2 次；其余实验单独改变池大小或蓝球数量。",
            "最近 30/60 期互相重叠，确认窗内的实验是探索性比较，不能当作新确认结果或提高中奖率的证明。",
            "描述性区间以期为单位计算，未校正多重比较或时间相关性；跟踪窗不足 30 期不报告此类区间。",
            "扩大并集增加覆盖基准；覆盖命中不能替代单注命中。未估算奖金、收益率或净利润。",
            "同为 8 注时，双蓝每蓝 4 注，四蓝每蓝 2 注；四蓝扩大命中期覆盖，但均匀开奖下蓝球命中注数期望仍为 0.5。",
            "next_research_tickets 仅供前瞻研究，未发布或封存，冻结生产规则保持原样。"
        ],
        "recent30": recent(30), "recent60": recent(60),
        "confirmation": summarize(&observations[..Protocol::CONFIRMATION_LEN].iter().collect::<Vec<_>>()),
        "tracking": summarize(&observations[Protocol::CONFIRMATION_LEN..].iter().collect::<Vec<_>>()),
        "periods": observations, "next_research_tickets": next,
    }))
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let result = analyze(&load_csv(Path::new(&path))?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_bets_preserve_pool_and_balance_each_blue_without_bankers() {
        let draws = ssq::draw::synth_draws(220, 8);
        let prepared = prepare_forecast(&draws).unwrap();
        let tickets = balanced(&prepared.forecast);
        assert_eq!(tickets.len(), 8);
        assert_eq!(tickets.iter().collect::<BTreeSet<_>>().len(), 8);
        for &(mask, _) in &tickets {
            assert_eq!(mask.count_ones(), 6);
        }
        for i in 0..33 {
            let count = tickets
                .iter()
                .filter(|(mask, _)| mask & (1 << i) != 0)
                .count();
            assert_eq!(
                count,
                if prepared.forecast.pool.contains(&i) {
                    4
                } else {
                    0
                }
            );
            for blue in &prepared.forecast.blue_order[..2] {
                assert_eq!(
                    tickets
                        .iter()
                        .filter(|(mask, b)| mask & (1 << i) != 0 && b == blue)
                        .count(),
                    count / 2
                );
            }
        }
        for i in 0..8 {
            for j in 0..i {
                assert!((tickets[i].0 & tickets[j].0).count_ones() <= 4);
            }
        }
    }

    #[test]
    fn later_draws_do_not_change_earlier_research_results() {
        let draws = load_csv(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/ssq.csv")
                .as_path(),
        )
        .unwrap();
        let before = analyze(&draws[..draws.len() - 1]).unwrap();
        let after = analyze(&draws).unwrap();
        assert_eq!(
            before["periods"],
            json!(
                &after["periods"].as_array().unwrap()
                    [..before["periods"].as_array().unwrap().len()]
            )
        );
        assert_eq!(before["confirmation"], after["confirmation"]);
        assert_eq!(before["red_weights"], after["red_weights"]);
        assert_eq!(before["blue_weights"], after["blue_weights"]);
    }

    #[test]
    fn frozen_metrics_match_the_existing_replay_and_ball_number_convention() {
        let draws = load_csv(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/ssq.csv")
                .as_path(),
        )
        .unwrap();
        let result = analyze(&draws).unwrap();
        let replay = ssq::history::replay(&draws).unwrap();
        for row in result["periods"].as_array().unwrap() {
            let old = replay
                .iter()
                .find(|r| r.issue == row["issue"].as_str().unwrap())
                .unwrap();
            let metrics = &row["methods"]["frozen"];
            assert_eq!(row["actual_blue"], old.actual_blue);
            assert_eq!(metrics["red_per_bet"], old.avg_red);
            assert_eq!(metrics["best_red"], old.best_red);
            assert_eq!(metrics["covered_red_n"], old.review.covered_red_n);
            assert_eq!(metrics["covered_red_hits"], old.review.red_hit_n);
            assert_eq!(metrics["blue_covered"], old.any_blue);
            assert_eq!(
                metrics["blue_hit_bets"],
                old.tickets.iter().filter(|t| t.blue_hit).count()
            );
            assert_eq!(
                row["frozen_diagnosis"],
                serde_json::to_value(&old.review).unwrap()
            );
        }
    }
}
