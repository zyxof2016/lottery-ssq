//! Read-only research v6: does shrunk pairwise co-occurrence structure select a
//! 16-red pool that covers all six winning reds sooner than marginal rankings?
//! The pool objective is a predeclared surrogate (sum of marginal and pair
//! log-lifts inside the pool), optimized by deterministic best-improvement
//! exchanges with a relaxation upper bound. No archive, no API, no promotion.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    model::roll_all,
    protocol::Protocol,
    red_model_v2::TRAIN_END,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};

pub const MARGINAL_KAPPA: f64 = 60.;
pub const PAIR_KAPPA: f64 = 200.;
pub const POOL: usize = 16;
pub const REQUIRED_MEAN_GAIN: f64 = 1.0;
pub const UNIFORM_MEAN_COVER: f64 = 6. * 34. / 7.;
const UNIFORM_PAIR: f64 = 15. / 528.;

#[derive(Clone)]
pub struct Study {
    pub marginal: [f64; 33],
    pub pair: [[f64; 33]; 33],
    pub training_sha: String,
    pub n: usize,
}

pub fn policy() -> Value {
    json!({"version":6,"research_only":true,"actual_recommendation":false,
        "objective":"predeclared_surrogate_sum_of_marginal_and_pair_log_lifts_inside_16_pool",
        "marginal_kappa":MARGINAL_KAPPA,"pair_kappa":PAIR_KAPPA,
        "training_last_issue":TRAIN_END,"expert_warmup_rows":30,
        "search":"deterministic_best_improvement_single_exchanges_from_top16_marginal",
        "certificate":"relaxation_upper_bound_top16_marginal_plus_top120_pair_lifts",
        "required_mean_gain_vs_weighted_and_uniform":REQUIRED_MEAN_GAIN,
        "uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
        "score_kind":"co_occurrence_lift_surrogate_not_probability",
        "automatic_promotion":false,
        "note":"共现lift只是代理目标，不是覆盖概率；搜索为局部最优加松弛上界证书。门槛未过不替换任何默认与已封存票面。"})
}

pub fn fit(draws: &[Draw]) -> Result<Study, String> {
    let training: Vec<_> = draws.iter().filter(|d| d.issue <= TRAIN_END).collect();
    if training.len() < 300 {
        return Err("共现研究需要至少300期训练历史".into());
    }
    let n = training.len() as f64;
    let mut count = [0f64; 33];
    let mut pair_count = [[0f64; 33]; 33];
    for d in &training {
        for &r in &d.red {
            count[(r - 1) as usize] += 1.;
        }
        for a in 0..6 {
            for b in (a + 1)..6 {
                let i = (d.red[a] - 1) as usize;
                let j = (d.red[b] - 1) as usize;
                pair_count[i][j] += 1.;
                pair_count[j][i] += 1.;
            }
        }
    }
    let marginal: [f64; 33] =
        std::array::from_fn(|i| (count[i] + MARGINAL_KAPPA * 6. / 33.) / (6. * n + MARGINAL_KAPPA));
    let mut pair = [[0f64; 33]; 33];
    for i in 0..33 {
        for j in 0..33 {
            if i == j {
                continue;
            }
            let expected = n * UNIFORM_PAIR * (marginal[i] / (6. / 33.)) * (marginal[j] / (6. / 33.));
            let q = (pair_count[i][j] + PAIR_KAPPA * expected) / (n + PAIR_KAPPA);
            pair[i][j] = (q / UNIFORM_PAIR).ln();
        }
    }
    Ok(Study {
        marginal,
        pair,
        training_sha: data_sha256(&training.iter().map(|d| (*d).clone()).collect::<Vec<_>>()),
        n: training.len(),
    })
}

pub struct PoolSearch {
    pub members: Vec<usize>,
    pub objective: f64,
    pub upper_bound: f64,
    pub evaluations: usize,
    pub passes: usize,
}

fn marginal_lift(s: &Study) -> [f64; 33] {
    std::array::from_fn(|i| (s.marginal[i] / (6. / 33.)).ln())
}

pub fn search(s: &Study) -> PoolSearch {
    let ml = marginal_lift(s);
    let mut order: Vec<usize> = (0..33).collect();
    order.sort_by(|&a, &b| s.marginal[b].total_cmp(&s.marginal[a]).then(a.cmp(&b)));
    let mut members = order[..POOL].to_vec();
    let mut in_pool = [false; 33];
    for &i in &members {
        in_pool[i] = true;
    }
    let objective = |m: &[usize]| -> f64 {
        m.iter().map(|&i| ml[i]).sum::<f64>()
            + m.iter()
                .enumerate()
                .map(|(a, &i)| m[a + 1..].iter().map(|&j| s.pair[i][j]).sum::<f64>())
                .sum::<f64>()
    };
    let mut row_sum: [f64; 33] = std::array::from_fn(|i| {
        members.iter().filter(|&&j| j != i).map(|&j| s.pair[i][j]).sum()
    });
    let mut evaluations = 0usize;
    let mut passes = 0usize;
    loop {
        passes += 1;
        let mut best: Option<(f64, usize, usize)> = None;
        for &out in &members.clone() {
            for in_ in 0..33 {
                if in_pool[in_] {
                    continue;
                }
                evaluations += 1;
                let delta = ml[in_] - ml[out] + row_sum[in_] - row_sum[out] - s.pair[in_][out];
                if delta > 1e-12 && best.is_none_or(|(d, ..)| delta > d) {
                    best = Some((delta, in_, out));
                }
            }
        }
        let Some((_, in_, out)) = best else { break };
        members.retain(|&x| x != out);
        members.push(in_);
        in_pool[out] = false;
        in_pool[in_] = true;
        for j in 0..33 {
            if j != in_ && j != out {
                row_sum[j] += s.pair[j][in_] - s.pair[j][out];
            }
        }
        row_sum[in_] = members.iter().filter(|&&j| j != in_).map(|&j| s.pair[in_][j]).sum();
        row_sum[out] = members.iter().map(|&j| s.pair[out][j]).sum();
        if passes >= 200 {
            break;
        }
    }
    let mut ml_sorted = ml;
    ml_sorted.sort_by(|a, b| b.total_cmp(a));
    let mut pairs: Vec<f64> = (0..33)
        .flat_map(|i| (i + 1..33).map(move |j| s.pair[i][j]))
        .collect();
    pairs.sort_by(|a, b| b.total_cmp(a));
    let upper_bound = ml_sorted[..POOL].iter().sum::<f64>()
        + pairs[..POOL * (POOL - 1) / 2].iter().sum::<f64>();
    let final_objective = objective(&members);
    PoolSearch {
        members,
        objective: final_objective,
        upper_bound,
        evaluations,
        passes,
    }
}

/// Full 33 order: searched pool first (by marginal), remaining numbers after.
pub fn order33(s: &Study, search: &PoolSearch) -> [usize; 33] {
    let mut inside = search.members.clone();
    inside.sort_by(|&a, &b| s.marginal[b].total_cmp(&s.marginal[a]).then(a.cmp(&b)));
    let mut outside: Vec<usize> = (0..33).filter(|i| !search.members.contains(i)).collect();
    outside.sort_by(|&a, &b| s.marginal[b].total_cmp(&s.marginal[a]).then(a.cmp(&b)));
    let mut all = inside;
    all.extend(outside);
    let mut out = [0usize; 33];
    out.copy_from_slice(&all);
    out
}

pub fn cover_prefix(order: &[usize; 33], mask: u64) -> usize {
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
    let study = fit(draws)?;
    let prepared = crate::eval::prepare_forecast(draws)?;
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if row.issue >= 23101 && (frozen || nfit >= 30) {
            let search = search(&study);
            let o6 = order33(&study, &search);
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let weighted = rank_desc(&fuse7(&row.expert_p, &w));
            let in_pool: u64 = search.members.iter().fold(0, |m, &i| m | (1 << i));
            rows.push(json!({
                "issue": fmt_issue(row.issue),
                "v6_cover_prefix": cover_prefix(&o6, row.red_mask),
                "weighted_cover_prefix": cover_prefix(&weighted, row.red_mask),
                "v6_full_six_in_pool": (in_pool & row.red_mask).count_ones() == 6,
                "weighted_full_six_in_16": weighted[..16].iter().filter(|&&i| row.red_mask & (1 << i) != 0).count() == 6,
                "pool": search.members.iter().map(|i| i + 1).collect::<Vec<_>>(),
                "objective": search.objective,
                "upper_bound": search.upper_bound,
                "evaluations": search.evaluations,
                "passes": search.passes,
            }));
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
            (!rs.is_empty())
                .then(|| rs.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / rs.len() as f64)
        };
        json!({"n":rs.len(),
            "v6_mean_cover_prefix":mean("v6_cover_prefix"),
            "weighted_mean_cover_prefix":mean("weighted_cover_prefix"),
            "v6_full_six_periods":rs.iter().filter(|r|r["v6_full_six_in_pool"]==true).count(),
            "weighted_full_six_periods":rs.iter().filter(|r|r["weighted_full_six_in_16"]==true).count(),
            "scope":"事后因果探索回放，不是实际推荐样本"})
    };
    let confirm = window(25111, 26115);
    let gain_weighted = confirm["weighted_mean_cover_prefix"]
        .as_f64()
        .zip(confirm["v6_mean_cover_prefix"].as_f64())
        .map(|(w, v)| w - v);
    let gain_uniform = confirm["v6_mean_cover_prefix"]
        .as_f64()
        .map(|v| UNIFORM_MEAN_COVER - v);
    let clears = gain_weighted.is_some_and(|g| g >= REQUIRED_MEAN_GAIN)
        && gain_uniform.is_some_and(|g| g >= REQUIRED_MEAN_GAIN);
    let rs = &rows[rows.len().saturating_sub(60)..];
    let rmean = |k: &str| {
        (!rs.is_empty())
            .then(|| rs.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / rs.len() as f64)
    };
    let recent60 = json!({"n":rs.len(),"v6_mean_cover_prefix":rmean("v6_cover_prefix"),"weighted_mean_cover_prefix":rmean("weighted_cover_prefix"),"v6_full_six_periods":rs.iter().filter(|r|r["v6_full_six_in_pool"]==true).count(),"weighted_full_six_periods":rs.iter().filter(|r|r["weighted_full_six_in_16"]==true).count()});
    let next = search(&study);
    Ok(json!({
        "kind":"exploratory_cooccurrence_pool_search_v6",
        "actual_recommendation":false,
        "data_sha256":data_sha256(draws),
        "cutoff_issue":fmt_issue(prepared.cutoff_issue),
        "policy":policy(),
        "training":{"last_issue":TRAIN_END,"history_sha256":study.training_sha,"n":study.n},
        "windows":{"development":window(23101,25110),"confirmation_exploratory":confirm,"tracking":window(26116,u32::MAX),"recent60":recent60},
        "gate":{"required_mean_gain":REQUIRED_MEAN_GAIN,"uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
            "confirmation_gain_vs_weighted":gain_weighted,"confirmation_gain_vs_uniform":gain_uniform,
            "clears_predeclared_mean_gate":clears,"replaces_weighted":false,"rewrites_archives":false},
        "next_preview":{"pool":next.members.iter().map(|i| i+1).collect::<Vec<_>>(),
            "objective":next.objective,"upper_bound":next.upper_bound,
            "absolute_gap":next.upper_bound-next.objective,"evaluations":next.evaluations,"passes":next.passes,
            "note":"研究预览，不生成票面、不留档。"},
        "latest_replay":rows.last(),
        "notes":[
            "边际与共现计数只用23100及以前开奖；收缩参数60/200事先写定，不在确认窗搜索。",
            "代理目标为池内边际log-lift与两两log-lift之和；交换搜索到局部最优，上界为前16边际lift加前120 pair lift的松弛。",
            "盖住名次与全6入池为探索指标；多候选已被查看，确认窗改善不能视作独立确认。",
            "共现结构未证明提高覆盖；门槛未过不替换默认、不改写原档、不接入生成或封存。",
        ],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn search_is_local_optimal_with_a_valid_relaxation_bound() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let study = fit(&draws).unwrap();
        let a = search(&study);
        let b = search(&study);
        assert_eq!(a.members, b.members);
        assert_eq!(a.members.len(), POOL);
        assert!(a.objective <= a.upper_bound + 1e-9);
        let ml = marginal_lift(&study);
        for &out in &a.members {
            for in_ in 0..33 {
                if a.members.contains(&in_) {
                    continue;
                }
                let delta = ml[in_] - ml[out]
                    + a.members
                        .iter()
                        .filter(|&&j| j != out && j != in_)
                        .map(|&j| study.pair[in_][j] - study.pair[out][j])
                        .sum::<f64>();
                assert!(delta <= 1e-9, "still improving swap {in_} {out} {delta}");
            }
        }
    }
    #[test]
    fn future_draws_do_not_change_the_frozen_cooccurrence_fit() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let a = fit(&draws).unwrap();
        let b = fit(&draws[..draws.len() - 1]).unwrap();
        assert_eq!(a.marginal, b.marginal);
        assert_eq!(a.pair, b.pair);
        assert_eq!(a.training_sha, b.training_sha);
    }
}