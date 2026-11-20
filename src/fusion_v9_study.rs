//! Read-only research v9: integrate the numbers of historically successful
//! small pools. For every model and every past draw, the minimum prefix that
//! covered all six winning reds is a "successful small pool" when its size is
//! <= K; each number inside it gains one support vote. The current pool is the
//! top-24 by cumulative support (ties broken by the weighted fusion score).
//! Strictly causal: only draws before the evaluated draw contribute. The
//! support count is evidence, not a probability; no archive, no promotion.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    fusion_model_v3::Distribution,
    fusion_model_v4::scores as v4_scores,
    fusion_model_v7::select_tickets,
    model::roll_all,
    protocol::Protocol,
    select::{equal7, fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};

pub const K: usize = 18;
pub const POOL: usize = 24;
pub const BLUES: usize = 4;
pub const REQUIRED_MEAN_GAIN: f64 = 1.0;
pub const UNIFORM_MEAN_COVER: f64 = 6. * 34. / 7.;
const RED_OUTCOMES: u64 = 1_107_568;

pub fn policy() -> Value {
    json!({"version":9,"research_only":true,"actual_recommendation":false,
        "objective":"integrate_numbers_of_historically_successful_small_pools",
        "success_definition":"model_minimum_full_cover_prefix <= K",
        "K":K,"models":"weighted_equal_and_seven_experts_equal_votes",
        "pool":"top24_by_cumulative_support_ties_by_weighted_fusion",
        "tickets":"v7_balanced_24_red_each_twice_plus_v8_four_blues_two_each",
        "causality":"only_draws_strictly_before_the_evaluated_draw_contribute",
        "required_mean_gain_vs_weighted_and_uniform":REQUIRED_MEAN_GAIN,
        "uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
        "score_kind":"cross_model_support_evidence_not_probability",
        "automatic_promotion":false,
        "note":"成功小池支持度是因果累计证据，不是覆盖概率；门槛未过不替换默认、不改写原档、不接入生成或封存。"})
}

fn orders(expert: &[[f64; 33]; 7], weights: &[f64; 7]) -> [[usize; 33]; 9] {
    let mut out = [[0usize; 33]; 9];
    out[0] = rank_desc(&fuse7(expert, weights));
    out[1] = rank_desc(&fuse7(expert, &equal7()));
    for e in 0..7 {
        out[e + 2] = rank_desc(&expert[e]);
    }
    out
}

fn prefix(order: &[usize; 33], mask: u64) -> usize {
    order
        .iter()
        .enumerate()
        .filter(|(_, i)| mask & (1 << **i) != 0)
        .map(|(r, _)| r + 1)
        .max()
        .unwrap_or(33)
}

pub fn replay(draws: &[Draw]) -> Result<Value, String> {
    let snap = roll_all(draws)?;
    let prepared = crate::eval::prepare_forecast(draws)?;
    let bf = {
        let samples = snap
            .rows
            .iter()
            .skip(30)
            .filter(|r| r.issue <= crate::blue_model_v2::TRAIN_END)
            .map(|r| {
                Ok(crate::blue_model_v2::Sample {
                    x: crate::blue_model_v2::features(&r.blue_p)?,
                    winner: r.blue as usize,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        crate::blue_model_v2::fit(&samples)?
    };
    let neutral = [1. / 33.; 33];
    let mut support = [0u64; 33];
    let mut successes = [0u64; 9];
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if row.issue >= 23101 && (frozen || nfit >= 30) {
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let os = orders(&row.expert_p, &w);
            let weighted = os[0];
            let fusion = fuse7(&row.expert_p, &w);
            let mut idx: Vec<usize> = (0..33).collect();
            idx.sort_by(|&a, &b| {
                support[b]
                    .cmp(&support[a])
                    .then(fusion[b].total_cmp(&fusion[a]))
                    .then(a.cmp(&b))
            });
            let mut order = [0usize; 33];
            order.copy_from_slice(&idx);
            let (s4, _) = v4_scores(&row.expert_p, &w, &neutral)?;
            let q = crate::blue_model_v2::distribution(
                &bf.beta,
                &crate::blue_model_v2::features(&row.blue_p)?,
            );
            let dist = Distribution::new(&s4, &q)?;
            let ranking: Vec<u8> = order.iter().map(|i| (i + 1) as u8).collect();
            let blues = dist.blue_order()[..BLUES].to_vec();
            // Ticket-level coverage enumeration is restricted to confirmation
            // and tracking rows; development rows report rank/full-six only.
            let coverage: Option<u64> = if frozen {
                let sel = select_tickets(&dist, &ranking)?;
                let tickets: Vec<Value> = sel
                    .sets
                    .iter()
                    .enumerate()
                    .map(|(i, r)| json!({"red":r,"blue":blues[i / 2]}))
                    .collect();
                let sets: std::collections::BTreeSet<[u8; 6]> = tickets
                    .iter()
                    .map(|t| {
                        let mut r: [u8; 6] = serde_json::from_value(t["red"].clone()).unwrap();
                        r.sort_unstable();
                        r
                    })
                    .collect();
                let r9 = crate::fusion_v3_comparison::red_prize_outcomes(&sets)?;
                Some(BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r9)
            } else {
                None
            };
            let in_pool: u64 = order[..POOL].iter().fold(0, |m, &i| m | (1 << i));
            rows.push(json!({
                "issue": fmt_issue(row.issue),
                "v9_cover_prefix": prefix(&order, row.red_mask),
                "weighted_cover_prefix": prefix(&weighted, row.red_mask),
                "v9_full_six_in_pool": (in_pool & row.red_mask).count_ones() == 6,
                "weighted_full_six_in_24": weighted[..24].iter().filter(|&&i| row.red_mask & (1 << i) != 0).count() == 6,
                "uniform_any_prize_rate": coverage.map(|c| c as f64 / (RED_OUTCOMES as f64 * 16.)),
                "blue_covered": blues.contains(&(row.blue + 1)),
                "pool": order[..POOL].iter().map(|i| i + 1).collect::<Vec<_>>(),
                "support_top": support[idx[0]],
            }));
        }
        if nfit >= 30 {
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let os = orders(&row.expert_p, &w);
            for (m, o) in os.iter().enumerate() {
                let p = prefix(o, row.red_mask);
                if p <= K {
                    successes[m] += 1;
                    for &i in &o[..p] {
                        support[i] += 1;
                    }
                }
            }
        }
        if !frozen {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            nfit += 1;
        }
    }
    let window = |lo: u32, hi: u32| {
        let rs: Vec<_> = rows
            .iter()
            .filter(|r| {
                let i: u32 = r["issue"].as_str().unwrap().parse().unwrap();
                i >= lo && i <= hi
            })
            .collect();
        let mean = |k: &str| {
            let vals: Vec<f64> = rs.iter().filter_map(|r| r[k].as_f64()).collect();
            (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
        };
        json!({"n":rs.len(),
            "v9_mean_cover_prefix":mean("v9_cover_prefix"),
            "weighted_mean_cover_prefix":mean("weighted_cover_prefix"),
            "v9_full_six_periods":rs.iter().filter(|r|r["v9_full_six_in_pool"]==true).count(),
            "weighted_full_six_periods":rs.iter().filter(|r|r["weighted_full_six_in_24"]==true).count(),
            "v9_mean_uniform_any_prize_rate":mean("uniform_any_prize_rate"),
            "v9_blue_covered_periods":rs.iter().filter(|r|r["blue_covered"]==true).count(),
            "scope":"事后因果探索回放，不是实际推荐样本"})
    };
    let confirm = window(25111, 26115);
    let gain_weighted = confirm["weighted_mean_cover_prefix"]
        .as_f64()
        .zip(confirm["v9_mean_cover_prefix"].as_f64())
        .map(|(w, v)| w - v);
    let gain_uniform = confirm["v9_mean_cover_prefix"]
        .as_f64()
        .map(|v| UNIFORM_MEAN_COVER - v);
    let clears = gain_weighted.is_some_and(|g| g >= REQUIRED_MEAN_GAIN)
        && gain_uniform.is_some_and(|g| g >= REQUIRED_MEAN_GAIN);
    let rs = &rows[rows.len().saturating_sub(60)..];
    let rmean = |k: &str| {
        (!rs.is_empty())
            .then(|| rs.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / rs.len() as f64)
    };
    let recent60 = json!({"n":rs.len(),"v9_mean_cover_prefix":rmean("v9_cover_prefix"),"weighted_mean_cover_prefix":rmean("weighted_cover_prefix"),"v9_full_six_periods":rs.iter().filter(|r|r["v9_full_six_in_pool"]==true).count(),"weighted_full_six_periods":rs.iter().filter(|r|r["weighted_full_six_in_24"]==true).count()});
    Ok(json!({
        "kind":"exploratory_cross_model_successful_small_pool_consensus_v9",
        "actual_recommendation":false,
        "data_sha256":data_sha256(draws),
        "cutoff_issue":fmt_issue(prepared.cutoff_issue),
        "policy":policy(),
        "success_counts_per_model":successes,
        "windows":{"development":window(23101,25110),"confirmation_exploratory":confirm,"tracking":window(26116,u32::MAX),"recent60":recent60},
        "gate":{"required_mean_gain":REQUIRED_MEAN_GAIN,"uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
            "confirmation_gain_vs_weighted":gain_weighted,"confirmation_gain_vs_uniform":gain_uniform,
            "clears_predeclared_mean_gate":clears,"replaces_weighted":false,"rewrites_archives":false},
        "latest_replay":rows.last(),
        "notes":[
            "成功小池＝某模型盖住全部6红的最小前缀且长度<=18；只累计严格早于当期的开奖，9模型等权。",
            "支持度是因果证据不是概率；同分用当期加权融合分数破，再按号码升序。",
            "票面沿用v7的24红每红2次均衡结构与v8的4蓝每蓝2注；均匀常规奖覆盖按精确枚举。",
            "门槛未过不替换默认、不改写原档；多候选已被查看，确认窗改善不能视作独立确认。",
        ],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn support_is_causal_and_prefix_definition_matches() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let a = replay(&draws).unwrap();
        let b = replay(&draws[..draws.len() - 1]).unwrap();
        assert_eq!(a["windows"]["development"], b["windows"]["development"]);
        assert_eq!(
            a["windows"]["confirmation_exploratory"],
            b["windows"]["confirmation_exploratory"]
        );
        let mut o = [0usize; 33];
        for i in 0..33 {
            o[i] = i;
        }
        assert_eq!(prefix(&o, (1 << 0) | (1 << 5) | (1 << 17)), 18);
    }
}