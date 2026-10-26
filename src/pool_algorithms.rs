//! Independently versioned observation algorithms. No guarantee of future coverage.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::{prepare_forecast, Prepared},
    pool,
    select::{fuse6, fuse7, rank_desc, rank_log_score, score_key, softmax_shrink},
};
use serde_json::{json, Value};

pub const MODELS: [&str; 4] = ["rank_consensus", "scenario_cover", "robust_cover", "dual_support"];
pub const COLLECTION_VERSION: u64 = 2;
pub const LEGACY_MODEL_COUNT: usize = 3;
pub const LABELS: [&str; 4] = [
    "名次共识 · 试验",
    "全6情景覆盖 · 试验",
    "三窗稳健覆盖 · 试验",
    "双专家支持 · 24红试验",
];
const WINDOW: usize = 312;
const WINDOWS: [usize; 3] = [60, 156, 312];
const PRIOR: f64 = 60.0;

pub fn is_model(model: &str) -> bool {
    MODELS.contains(&model)
}

pub fn policy(model: &str) -> Value {
    let i = MODELS
        .iter()
        .position(|m| *m == model)
        .expect("known pool algorithm");
    let methods = [
        "tie_aware_equal_borda_seven_experts",
        "greedy_backward_full_six_scenario_survival",
        "greedy_backward_minimum_three_window_survival",
        "two_best_tie_aware_expert_midranks",
    ];
    let notes = [
        "七个专家按并列平均名次等权投票，减轻不同评分尺度的影响；同分沿原加权排名。",
        "将此前最多312期的六红组合作为情景，逐次剔除使完整六红情景损失最少的号码；同分参考名次共识。",
        "在此前60/156/312期三个窗口中，优先保留平滑后最弱窗口的完整六红情景覆盖；同分参考窗口平均与名次共识。",
        "每号取七专家中支持最高的两位的并列平均名次，再按两者平均排序；不将这两位视为独立证据。24红作新增前瞻试验；初始研究中16红及长期2蓝联合全包均退步，不替换原默认。",
    ];
    if model == "dual_support" {
        return json!({"model":model,"label":LABELS[i],"version":1,"method":methods[i],
            "experts":7,"supporters":2,"rank_units":"twice_average_rank_for_ties",
            "ties":"original_weighted_rank","uses_scenario_masks":false,
            "primary_metric":"all_six_red_rate_at_24","auxiliary_metric":"all_six_red_and_blue_covered",
            "red_sizes":pool::TIER_RED_SIZES,"blue_sizes":pool::BLUE_SIZES,
            "blue_scope":"shared_weighted_blue_ranking","automatic_promotion":false,
            "score_kind":"ordinal_ranking_evidence_not_probability",
            "note":format!("{} 历史比较含多种已查看候选，不能视为独立确认或未来覆盖概率。",notes[i])});
    }
    json!({"model":model,"label":LABELS[i],"version":1,"method":methods[i],
        "training_draws_max":WINDOW,"windows":WINDOWS,"uniform_prior_periods":PRIOR,
        "primary_metric":"all_six_red_rate_at_same_pool_size","auxiliary_metric":"all_six_red_and_blue_covered",
        "red_sizes":pool::TIER_RED_SIZES,"blue_sizes":pool::BLUE_SIZES,
        "blue_scope":"shared_weighted_blue_ranking","ties":"consensus_then_original_weighted_rank",
        "score_kind":"ordinal_ranking_evidence_not_probability","automatic_promotion":false,
        "note":format!("{} 以少量红球包含全部6红为观察目标，历史情景不代表未来分布，不保证每期全覆盖。",notes[i])})
}

fn consensus_units(expert: &[[f64; 33]; 7]) -> [u32; 33] {
    let mut units = [0; 33];
    for p in expert {
        let order = rank_desc(p);
        let mut start = 0;
        while start < 33 {
            let mut end = start + 1;
            while end < 33 && score_key(p[order[end]]) == score_key(p[order[start]]) {
                end += 1;
            }
            // Twice the Borda score. Exact integer midranks keep tied experts neutral.
            for &n in &order[start..end] {
                units[n] += (67 - start - end) as u32;
            }
            start = end;
        }
    }
    units
}

fn dual_support_units(experts: &[[f64; 33]; 7]) -> [u32; 33] {
    let mut ranks = [[0u32; 33]; 7];
    for (e, p) in experts.iter().enumerate() {
        let order = rank_desc(p);let mut start = 0;
        while start < 33 {
            let mut end = start + 1;
            while end < 33 && score_key(p[order[end]]) == score_key(p[order[start]]) { end += 1; }
            for &n in &order[start..end] { ranks[e][n] = (start + 1 + end) as u32; }
            start = end;
        }
    }
    std::array::from_fn(|n| { let mut r = ranks.map(|e| e[n]);r.sort_unstable();134-r[0]-r[1] })
}

fn greedy_order(masks: &[u64], units: &[u32; 33], base: &[usize; 33], robust: bool) -> [usize; 33] {
    let mut order = [0; 33];
    let mut members = (1u64 << 33) - 1;
    let mut live = vec![true; masks.len()];
    let mut base_rank = [0; 33];
    for (i, &n) in base.iter().enumerate() {
        base_rank[n] = i;
    }
    let lengths = WINDOWS.map(|w| w.min(masks.len()));
    for position in (1..33).rev() {
        let mut totals = [0usize; 3];
        let mut losses = [[0usize; 3]; 33];
        for (i, &mask) in masks.iter().enumerate().filter(|(i, _)| live[*i]) {
            for w in 0..3 {
                if i < masks.len() - lengths[w] {
                    continue;
                }
                totals[w] += 1;
                let mut bits = mask;
                while bits != 0 {
                    losses[bits.trailing_zeros() as usize][w] += 1;
                    bits &= bits - 1;
                }
            }
        }
        let prior = PRIOR * pool::uniform_full_red(position);
        let objective = |n: usize| {
            let rates: [f64; 3] = std::array::from_fn(|w| {
                ((totals[w] - losses[n][w]) as f64 + prior) / (lengths[w] as f64 + PRIOR)
            });
            if robust {
                (
                    rates.into_iter().fold(f64::INFINITY, f64::min),
                    rates.iter().sum::<f64>() / 3.0,
                )
            } else {
                (rates[2], rates[2])
            }
        };
        let removed = (0..33)
            .filter(|&n| members & (1 << n) != 0)
            .max_by(|&a, &b| {
                let x = objective(a);
                let y = objective(b);
                x.0.total_cmp(&y.0)
                    .then(x.1.total_cmp(&y.1))
                    .then(units[b].cmp(&units[a]))
                    .then(base_rank[a].cmp(&base_rank[b]))
            })
            .unwrap();
        order[position] = removed;
        members &= !(1 << removed);
        for (i, mask) in masks.iter().enumerate() {
            live[i] &= mask & (1 << removed) == 0;
        }
    }
    order[0] = members.trailing_zeros() as usize;
    order
}

fn rankings(expert: &[[f64; 33]; 7], base: &[usize; 33], masks: &[u64]) -> [[usize; 33]; 4] {
    let units = consensus_units(expert);
    let mut consensus = *base;
    consensus.sort_by_key(|&n| std::cmp::Reverse(units[n]));
    let support=dual_support_units(expert);let mut dual=*base;dual.sort_by_key(|&n|std::cmp::Reverse(support[n]));
    [
        consensus,
        greedy_order(masks, &units, base, false),
        greedy_order(masks, &units, base, true),
        dual,
    ]
}

fn ordinal(order: &[usize; 33]) -> [f64; 33] {
    let mut score = [0.; 33];
    for (rank, &n) in order.iter().enumerate() {
        score[n] = (33 - rank) as f64 / 561.0;
    }
    score
}

pub fn saved_all(prepared: &Prepared, draws: &[Draw]) -> Vec<Value> {
    let recent = &draws[draws.len().saturating_sub(WINDOW)..];
    let masks: Vec<_> = recent.iter().map(|d| d.red_mask).collect();
    let orders = rankings(&prepared.expert_red, &prepared.forecast.red_order, &masks);
    let training = json!({"n":recent.len(),"from":recent.first().map(|d|fmt_issue(d.issue)),
        "to":recent.last().map(|d|fmt_issue(d.issue)),"scenario_masks":masks,
        "expert_red_scores":prepared.expert_red.iter().map(|r|r.to_vec()).collect::<Vec<_>>()});
    MODELS.iter().enumerate().map(|(i, &model)| {
        let order = &orders[i];
        let pools: Vec<_> = pool::TIER_RED_SIZES.into_iter().enumerate().flat_map(|(t, r)|
            pool::BLUE_SIZES.into_iter().map(move |b|json!({"id":format!("{model}_red{r}_blue{b}"),
                "model":model,"group":"observation","tier_id":((b'A'+t as u8)as char).to_string(),
                "red_count":r,"blue_count":b,"reds":order[..r].iter().map(|n|n+1).collect::<Vec<_>>(),
                "blues":prepared.forecast.blue_order[..b].iter().map(|n|n+1).collect::<Vec<_>>()}))).collect();
        json!({"model":model,"version":1,"policy":policy(model),"target_issue":fmt_issue(prepared.cutoff_issue+1),
            "input_history_sha256":prepared.data_sha,"training":training,
            "base_red_ranking":prepared.forecast.red_order.map(|n|n+1).to_vec(),
            "red_ranking":order.map(|n|n+1).to_vec(),"red_scores":ordinal(order).to_vec(),
            "red_weights":Value::Null,"score_kind":"ordinal_ranking_evidence_not_probability","pools":pools})
    }).collect()
}

pub fn prepare(draws: &[Draw], model: &str) -> Result<Prepared, String> {
    let index = MODELS
        .iter()
        .position(|m| *m == model)
        .ok_or("未知号码池算法")?;
    let prepared = prepare_forecast(draws)?;
    let metadata = saved_all(&prepared, draws).swap_remove(index);
    let score: Vec<f64> =
        serde_json::from_value(metadata["red_scores"].clone()).map_err(|e| e.to_string())?;
    Ok(crate::ranking::apply(
        prepared,
        score.try_into().map_err(|_| "排名评分长度错误")?,
        metadata,
    ))
}

pub fn validate_snapshot(saved: &Value) -> Result<(), String> {
    let model = saved["model"].as_str().ok_or("缺少号码池算法名称")?;
    let index = MODELS
        .iter()
        .position(|m| *m == model)
        .ok_or("未知号码池算法")?;
    if saved["version"] != 1 || saved["policy"] != policy(model) || !saved["red_weights"].is_null()
    {
        return Err("号码池算法版本或策略不一致".into());
    }
    let base: Vec<usize> = serde_json::from_value(saved["base_red_ranking"].clone())
        .map_err(|_| "算法缺少基础排名")?;
    if base.len() != 33
        || base
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            != (1..=33).collect()
    {
        return Err("算法基础排名无效".into());
    }
    let base: [usize; 33] = base
        .iter()
        .map(|n| n - 1)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let ps: Vec<Vec<f64>> = serde_json::from_value(saved["training"]["expert_red_scores"].clone())
        .map_err(|_| "缺少专家评分")?;
    if ps.len() != 7
        || ps.iter().any(|p| {
            p.len() != 33
                || p.iter().any(|v| !v.is_finite() || *v < 0.)
                || (p.iter().sum::<f64>() - 1.).abs() > 1e-10
        })
    {
        return Err("算法专家评分无效".into());
    }
    let ps: [[f64; 33]; 7] = ps
        .into_iter()
        .map(|p| p.try_into().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let masks: Vec<u64> = serde_json::from_value(saved["training"]["scenario_masks"].clone())
        .map_err(|_| "缺少六红情景")?;
    if masks.is_empty()
        || masks.len() > WINDOW
        || saved["training"]["n"] != masks.len()
        || masks.iter().any(|m| m.count_ones() != 6 || m >> 33 != 0)
    {
        return Err("六红情景无效".into());
    }
    let order = rankings(&ps, &base, &masks)[index];
    if saved["red_ranking"] != json!(order.map(|n| n + 1).to_vec())
        || saved["red_scores"] != json!(ordinal(&order).to_vec())
    {
        return Err("号码池算法排名与保存证据不一致".into());
    }
    Ok(())
}

/// Recompute against the bound prefix only, never the later evaluation draw.
pub fn validate_history(models: &[Value], draws: &[Draw]) -> Result<(), String> {
    let expected = saved_all(&prepare_forecast(draws)?, draws);
    if ![LEGACY_MODEL_COUNT, MODELS.len()].contains(&models.len()) || models != &expected[..models.len()] {
        return Err("号码池算法证据与输入历史不一致".into());
    }
    Ok(())
}

struct Period {
    issue: u32,
    hits: [[usize; 6]; 5],
    blue_rank: usize,
}
fn summarize(rows: &[Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells: Vec<_> = std::iter::once("weighted").chain(MODELS).enumerate().flat_map(|(i, model)|
        pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j, k)| {
            let full = rows.iter().filter(|r|r.hits[i][j]==6).count();
            let total: usize = rows.iter().map(|r|r.hits[i][j]).sum();
            json!({"model":model,"red_count":k,"n":rows.len(),"full_red_periods":full,
                "full_red_rate":full as f64/rows.len()as f64,"mean_red_hits":total as f64/rows.len()as f64,
                "mean_missed_reds":6.-total as f64/rows.len()as f64,"uniform_full_red_rate":pool::uniform_full_red(k),
                "full_draw_periods_by_blue":pool::BLUE_SIZES.map(|b|rows.iter().filter(|r|r.hits[i][j]==6&&r.blue_rank<=b).count()),
                "paired_full_gains":rows.iter().filter(|r|r.hits[i][j]==6&&r.hits[0][j]!=6).count(),
                "paired_full_losses":rows.iter().filter(|r|r.hits[i][j]!=6&&r.hits[0][j]==6).count()})
        })).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows.first().unwrap().issue),"to":fmt_issue(rows.last().unwrap().issue),"cells":cells})
}

pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    use crate::{model::roll_all, protocol::Protocol};
    let snapshot = roll_all(draws)?;
    let rowmap: std::collections::BTreeMap<_, _> =
        snapshot.rows.iter().map(|r| (r.issue, r)).collect();
    let (mut red_sum, mut blue_sum, mut nfit) = ([0.; 7], [0.; 6], 0);
    let mut rows = Vec::new();
    for (t, draw) in draws.iter().enumerate() {
        if let Some(row) = rowmap.get(&draw.issue) {
            if nfit >= 30 {
                let base = rank_desc(&fuse7(
                    &row.expert_p,
                    &softmax_shrink(&red_sum, nfit as f64),
                ));
                let masks: Vec<_> = draws[t.saturating_sub(WINDOW)..t]
                    .iter()
                    .map(|d| d.red_mask)
                    .collect();
                let candidates = rankings(&row.expert_p, &base, &masks);
                let blue = rank_desc(&fuse6(&row.blue_p, &softmax_shrink(&blue_sum, nfit as f64)));
                rows.push(Period {
                    issue: draw.issue,
                    hits: std::array::from_fn(|m| {
                        let order = if m == 0 { &base } else { &candidates[m - 1] };
                        pool::TIER_RED_SIZES.map(|k| {
                            order[..k]
                                .iter()
                                .filter(|&&n| draw.red_mask & (1 << n) != 0)
                                .count()
                        })
                    }),
                    blue_rank: blue.iter().position(|&n| n == row.blue as usize).unwrap() + 1,
                });
            }
            if row.issue < Protocol::CONFIRM_START_ISSUE {
                for e in 0..7 {
                    red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
                }
                for e in 0..6 {
                    blue_sum[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
                }
                nfit += 1;
            }
        }
    }
    let start = rows
        .iter()
        .position(|r| r.issue == Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let prepared = prepare_forecast(draws)?;
    Ok(
        json!({"kind":"exploratory_pool_algorithm_comparison","actual_recommendation":false,
        "data_sha256":data_sha256(draws),"cutoff_issue":fmt_issue(prepared.cutoff_issue),
        "source_sha256":crate::hash::sha256_hex(concat!(include_str!("pool_algorithms.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("eval.rs"),include_str!("protocol.rs"),include_str!("pool.rs"),include_str!("ranking.rs")).as_bytes()),
        "policies":MODELS.map(policy),"next_models":saved_all(&prepared,draws),
        "windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),
            "tracking":summarize(&rows[end..]),"recent30":summarize(&rows[rows.len().saturating_sub(30)..]),
            "recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},
        "latest_replay":rows.last().map(|r|json!({"issue":fmt_issue(r.issue),"hits":r.hits,"blue_rank":r.blue_rank})),
        "adoption":{"primary_protocol_replaced":false,"automatic_promotion":false},
        "notes":["目标为同数量的6红全入池；6红1蓝为辅助，入池全包不等于已组成中奖单注。",
            "逐期先预测再评价，仅用更早开奖形成情景和专家评分；规则属于探索性比较。",
            "同大小均匀全包基准仍很低，历史全6情景保留最多也不保证未来全覆盖。",
            "旧档不补造新算法，新算法实际样本从后续首次冻结开始；待开奖不计为漏号。"]}),
    )
}

pub fn cached_analyze(draws: &[Draw]) -> Result<Value, String> {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let key = data_sha256(draws);
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "算法报告缓存不可用")?;
    if let Some((sha, value)) = &*cache {
        if sha == &key {
            return Ok(value.clone());
        }
    }
    let report = analyze(draws)?;
    *cache = Some((key, report.clone()));
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tied_experts_are_neutral_and_scenarios_preserve_complete_sixes() {
        let expert = [[1. / 33.; 33]; 7];
        let base = std::array::from_fn(|n| n);
        let mask = (1u64 << 6) - 1;
        let orders = rankings(&expert, &base, &vec![mask; 100]);
        assert_eq!(orders[0], base);
        for order in &orders[1..] {
            assert_eq!(order[..6].iter().fold(0u64, |m, n| m | (1 << n)), mask);
            assert_eq!(
                order
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..33).collect()
            );
        }
    }
    #[test]
    fn dual_support_neutral_ties_and_minority_support_are_deterministic() {
        let mut experts=[[1.0/33.0;33];7];let base=std::array::from_fn(|n|32-n);
        let ranks=rankings(&experts,&base,&[(1u64<<6)-1]);assert_eq!(ranks[3],base);
        experts[0][0]=0.9;experts[1][0]=0.8;
        let units=dual_support_units(&experts);assert!(units[0]>units[1]);
    }

    #[test]
    fn snapshots_bind_training_and_reject_tampering() {
        let draws = crate::draw::synth_draws(220, 9);
        let p = prepare_forecast(&draws).unwrap();
        let models = saved_all(&p, &draws);
        for m in &models {
            validate_snapshot(m).unwrap();
        }
        validate_history(&models, &draws).unwrap();
        validate_history(&models[..LEGACY_MODEL_COUNT], &draws).unwrap();
        let mut bad = models[1].clone();
        bad["red_ranking"].as_array_mut().unwrap().swap(0, 32);
        assert!(validate_snapshot(&bad).is_err());
        let mut changed = draws.clone();
        changed[0] = crate::draw::synth_draws(220, 10)[0].clone();
        assert!(validate_history(&models, &changed).is_err());
        let mut bad = models[2].clone();
        bad["training"]["scenario_masks"][0] = json!(1);
        assert!(validate_snapshot(&bad).is_err());
    }
    #[test]
    fn replay_is_causal_and_matches_the_previous_next_preview() {
        let mut draws = crate::draw::synth_draws(220, 9);
        let old = analyze(&draws).unwrap();
        let mut future = draws.last().unwrap().clone();
        future.issue += 1;
        draws.push(future.clone());
        let later = analyze(&draws).unwrap();
        for (m, saved) in old["next_models"].as_array().unwrap().iter().enumerate() {
            for (j, k) in pool::TIER_RED_SIZES.into_iter().enumerate() {
                let hits = saved["red_ranking"].as_array().unwrap()[..k]
                    .iter()
                    .filter(|n| future.red.contains(&(n.as_u64().unwrap() as u8)))
                    .count();
                assert_eq!(later["latest_replay"]["hits"][m + 1][j], hits);
            }
        }
        let without_last = analyze(&draws[..draws.len() - 1]).unwrap();
        assert_eq!(old, without_last);
    }
}
