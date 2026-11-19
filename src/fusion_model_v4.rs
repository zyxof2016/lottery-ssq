//! Frozen fusion v4 trial: the red score counts EWMA exactly once (the extra
//! v2 recency term is removed), and the 8 complete tickets maximize model mass
//! under exact red exposure balance with 8 distinct blues. Eight distinct blues
//! cover 8*C(33,6) complete outcomes regardless of the reds, so uniform
//! regular-prize coverage is structurally above the v2 same-budget ticket set.
use crate::{
    blue_model_v2 as blue,
    draw::{data_sha256, Draw},
    fusion_model_v3::{mass, Distribution},
    model::roll_all,
    protocol::Protocol,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

pub const BLUE_TRAIN_END: u32 = 23100;
pub const POOL: usize = 16;
pub const BETS: usize = 8;
pub const BLUES: usize = 8;
const RED_OUTCOMES: u64 = 1_107_568;

pub fn policy() -> Value {
    json!({"version":4,"experimental":true,"prediction_advantage_established":false,
        "red_score":"0.9*six_expert_renormalized_fusion + 0.1*reference_union_support",
        "ewma_counted_once":true,"extra_ewma_term_removed":true,"explicit_repeat_weight":0,
        "reference_support_weight":0.1,"reference_support_kind":"uncalibrated_vote_retained_from_v2",
        "red_weight_boundary":"protocol_weights_frozen_before_25111",
        "blue_model":"deduplicated_blue_v2","blue_training_last_issue":BLUE_TRAIN_END,"ridge":40,
        "singles":{"bets":BETS,"cost":BETS*2,"red_pool":POOL,
            "red_selection":"balanced_feasible_start_then_deterministic_pairwise_swaps_maximizing_model_mass",
            "red_exposure_each":BETS*6/POOL,"blues":BLUES,
            "blue_selection":"top8_frozen_blue_probabilities_one_per_ticket",
            "pairing":"red_mass_desc_with_blue_probability_desc_rearrangement_optimum"},
        "coverage_guarantee":"8个不同蓝球无条件覆盖8*C(33,6)个完整结果，均匀常规奖覆盖不低于50%，结构性高于v2同预算票组；红球并集恰为16。",
        "dantuo":"本版本只试验8注单式；v2、v3胆拖保持各自原档。",
        "model_relation":"factorized_red_blue_no_learned_conditional_dependence",
        "automatic_promotion":false})
}

pub fn scores(
    experts: &[[f64; 33]; 7],
    weights: &[f64; 7],
    support: &[f64; 33],
) -> Result<([f64; 33], [f64; 7]), String> {
    if experts
        .iter()
        .flatten()
        .chain(support)
        .chain(weights)
        .any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("融合v4评分输入无效".into());
    }
    let total: f64 = weights[..6].iter().sum();
    if total <= 0. {
        return Err("融合v4没有可保留的专家权重".into());
    }
    let w = std::array::from_fn(|i| if i == 6 { 0. } else { weights[i] / total });
    let base = fuse7(experts, &w);
    let raw = std::array::from_fn(|i| 0.9 * base[i] + 0.1 * support[i]);
    Ok((raw, w))
}

#[derive(Clone)]
struct BlueFit {
    beta: [f64; 5],
    training_sha: String,
    n: usize,
}
fn blue_fit(draws: &[Draw], rows: &[crate::model::Row]) -> Result<BlueFit, String> {
    static CACHE: OnceLock<Mutex<Option<(String, BlueFit)>>> = OnceLock::new();
    let training: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= BLUE_TRAIN_END)
        .cloned()
        .collect();
    let sha = data_sha256(&training);
    let mut lock = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "融合v4蓝球拟合缓存异常")?;
    if let Some((key, f)) = &*lock {
        if key == &sha {
            return Ok(f.clone());
        }
    }
    let samples = rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= BLUE_TRAIN_END)
        .map(|r| {
            Ok(blue::Sample {
                x: blue::features(&r.blue_p)?,
                winner: r.blue as usize,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let b = blue::fit(&samples)?;
    let result = BlueFit {
        beta: b.beta,
        training_sha: sha.clone(),
        n: b.n,
    };
    *lock = Some((sha, result.clone()));
    Ok(result)
}

struct Cand {
    numbers: [u8; 6],
    mass: f64,
}
pub struct Selection {
    pub sets: Vec<[u8; 6]>,
    pub red_mass: f64,
    pub pool_upper_bound: f64,
    pub evaluated: usize,
}

fn combinations(pool: &[u8]) -> Vec<[usize; 6]> {
    let mut out = Vec::new();
    fn rec(pool: &[u8], start: usize, depth: usize, idx: &mut [usize; 6], out: &mut Vec<[usize; 6]>) {
        if depth == 6 {
            out.push(*idx);
            return;
        }
        for i in start..=pool.len() - (6 - depth) {
            idx[depth] = i;
            rec(pool, i + 1, depth + 1, idx, out);
        }
    }
    rec(pool, 0, 0, &mut [0; 6], &mut out);
    out
}

/// Start from a feasible exactly-balanced ticket template, then improve the
/// total model mass with deterministic pairwise red swaps. A swap moves one
/// red each way between two tickets, so every pool red stays at exactly three
/// exposures and all tickets stay distinct; only strictly improving swaps are
/// accepted, in a fixed scan order.
pub fn select_tickets(dist: &Distribution, ranking: &[u8]) -> Result<Selection, String> {
    if ranking.len() != 33 || BETS * 6 % POOL != 0 {
        return Err("融合v4组票参数无效".into());
    }
    let pool = &ranking[..POOL];
    let target = BETS * 6 / POOL;
    let mut cands: Vec<Cand> = combinations(pool)
        .into_iter()
        .map(|idx| {
            let mut numbers: [u8; 6] = std::array::from_fn(|k| pool[idx[k]]);
            numbers.sort_unstable();
            Ok(Cand {
                numbers,
                mass: dist.pool_mass(&numbers)?,
            })
        })
        .collect::<Result<_, String>>()?;
    let enumerated = cands.len();
    cands.sort_by(|a, b| b.mass.total_cmp(&a.mass).then(a.numbers.cmp(&b.numbers)));
    let pool_upper_bound: f64 = cands[..BETS].iter().map(|c| c.mass).sum();
    let pool_usize: Vec<usize> = pool.iter().map(|n| (*n - 1) as usize).collect();
    let mut sets: Vec<[u8; 6]> = crate::coverage::balanced_tickets(&pool_usize, BETS)?
        .iter()
        .map(|m| {
            let mut ns: [u8; 6] = std::array::from_fn(|k| {
                let bit = (0..33)
                    .filter(|b| m & (1 << b) != 0)
                    .nth(k)
                    .expect("均衡模板每注恰6红");
                bit as u8 + 1
            });
            ns.sort_unstable();
            ns
        })
        .collect();
    let mut masses: Vec<f64> = sets
        .iter()
        .map(|s| dist.pool_mass(s))
        .collect::<Result<_, String>>()?;
    let mut swap_evaluations = 0usize;
    for _pass in 0..64 {
        let mut best: Option<(f64, usize, usize, [u8; 6], [u8; 6])> = None;
        for i in 0..BETS {
            for j in (i + 1)..BETS {
                for a in sets[i] {
                    for b in sets[j] {
                        if a == b || sets[j].contains(&a) || sets[i].contains(&b) {
                            continue;
                        }
                        let mut ni = sets[i];
                        ni[ni.iter().position(|&n| n == a).unwrap()] = b;
                        ni.sort_unstable();
                        let mut nj = sets[j];
                        nj[nj.iter().position(|&n| n == b).unwrap()] = a;
                        nj.sort_unstable();
                        if (0..BETS).any(|k| k != i && k != j && (sets[k] == ni || sets[k] == nj))
                        {
                            continue;
                        }
                        let mi = dist.pool_mass(&ni)?;
                        let mj = dist.pool_mass(&nj)?;
                        swap_evaluations += 1;
                        let delta = mi + mj - masses[i] - masses[j];
                        if delta > 0. && best.as_ref().is_none_or(|(d, ..)| delta > *d) {
                            best = Some((delta, i, j, ni, nj));
                        }
                    }
                }
            }
        }
        let Some((_, i, j, ni, nj)) = best else { break };
        masses[i] = dist.pool_mass(&ni)?;
        masses[j] = dist.pool_mass(&nj)?;
        sets[i] = ni;
        sets[j] = nj;
    }
    let mut order: Vec<usize> = (0..BETS).collect();
    order.sort_by(|&a, &b| masses[b].total_cmp(&masses[a]).then(sets[a].cmp(&sets[b])));
    let sets: Vec<[u8; 6]> = order.iter().map(|&i| sets[i]).collect();
    let red_mass: f64 = order.iter().map(|&i| masses[i]).sum();
    for n in pool {
        if sets.iter().filter(|r| r.contains(n)).count() != target {
            return Err("融合v4曝光均衡未闭合".into());
        }
    }
    if sets.iter().collect::<BTreeSet<_>>().len() != BETS {
        return Err("融合v4票面重复".into());
    }
    Ok(Selection {
        sets,
        red_mass,
        pool_upper_bound,
        evaluated: enumerated + swap_evaluations,
    })
}

fn red_sets(tickets: &Value) -> Result<BTreeSet<[u8; 6]>, String> {
    let mut out = BTreeSet::new();
    for t in tickets.as_array().ok_or("缺少完整票面")? {
        let mut r: [u8; 6] =
            serde_json::from_value(t["red"].clone()).map_err(|_| "融合v4红票无效")?;
        r.sort_unstable();
        out.insert(r);
    }
    Ok(out)
}

pub fn build(draws: &[Draw], parent: &Value, legacy: &Value) -> Result<Value, String> {
    let prepared = crate::eval::prepare_forecast(draws)?;
    let snap = roll_all(draws)?;
    let fusion = &parent["payload"]["reference_pools"]["fusion"];
    let support: Vec<f64> =
        serde_json::from_value(fusion["reference_support"].clone()).map_err(|_| "参考支持无效")?;
    let support: [f64; 33] = support.try_into().map_err(|_| "参考支持长度无效")?;
    let (raw, weights) = scores(&prepared.expert_red, &prepared.weights, &support)?;
    let order = rank_desc(&raw);
    let ranking: Vec<u8> = order.iter().map(|i| (i + 1) as u8).collect();
    let bf = blue_fit(draws, &snap.rows)?;
    let blue_probs = blue::distribution(&bf.beta, &blue::features(&snap.next_blue)?);
    let dist = Distribution::new(&raw, &blue_probs)?;
    let sel = select_tickets(&dist, &ranking)?;
    let blues: Vec<u8> = dist.blue_order()[..BLUES].to_vec();
    let tickets: Vec<Value> = sel
        .sets
        .iter()
        .zip(&blues)
        .map(|(r, b)| json!({"red":r,"blue":b}))
        .collect();
    let ticket_json = json!(tickets);
    let joint_mass = mass(&dist, &ticket_json)?;
    let global = dist.top(BETS)?;
    let reds: BTreeSet<u8> = sel.sets.iter().flat_map(|r| r.iter().copied()).collect();
    let r4 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&ticket_json)?)?;
    let coverage4 = BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r4;
    let old = legacy["bundle"]["singles"]
        .as_array()
        .and_then(|a| a.iter().find(|p| p["red_count"] == 16))
        .ok_or("缺少v2同预算16红首次对照")?;
    let old_mass = mass(&dist, &old["tickets"])?;
    let old_blues = old["blues"].as_array().ok_or("v2对照缺少蓝球")?.len() as u64;
    let r2 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&old["tickets"])?)?;
    let coverage2 = old_blues * RED_OUTCOMES + (16 - old_blues) * r2;
    let plan = json!({"id":"coverage8","title":"融合覆盖试验v4 · 8注","mode":"joint",
        "bet_count":BETS,"cost":BETS*2,"red_count":reds.len(),"reds":reds,"blues":blues,
        "red_dan":[],"red_tuo":[],"tickets":tickets,
        "model_joint_mass":joint_mass,"ratio_to_uniform_budget":joint_mass/(BETS as f64/(RED_OUTCOMES as f64*16.)),
        "uniform_any_prize_outcomes":coverage4,"uniform_jackpot_outcomes":BETS,
        "red_prize_outcomes_without_blue":r4,
        "certificate":{"exposure_balance":"16红每红恰好3次","evaluations":sel.evaluated,
            "pool_unconstrained_upper_bound_mass":sel.pool_upper_bound,"achieved_red_mass":sel.red_mass,
            "absolute_red_mass_gap":sel.pool_upper_bound-sel.red_mass,
            "global_unconstrained_joint_mass":global.mass,"global_reference_evaluations":global.evaluated,
            "scope":"均衡曝光与8蓝分散是覆盖设计约束；可行均衡起点加确定性成对交换改进质量，池内无约束上界给出差距证书，不宣称全局最优或真实预测优势。"},
        "same_budget_v2_comparison":{"id":old["id"],"bets":old["bet_count"],"cost":old["cost"],
            "model_joint_mass":old_mass,"relative_model_gain":joint_mass/old_mass-1.,
            "uniform_any_prize_outcomes":coverage2,"used_reds":old["reds"],"used_blues":old["blues"]}});
    let previous = draws.last().ok_or("没有开奖数据")?;
    let mut records = Vec::new();
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    let neutral = [1. / 33.; 33];
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if frozen || nfit >= 30 {
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let (s, _) = scores(&row.expert_p, &w, &neutral)?;
            let o = rank_desc(&s);
            let q = blue::distribution(&bf.beta, &blue::features(&row.blue_p)?);
            let d = Distribution::new(&s, &q)?;
            let ns = crate::plan::mask_nums(row.red_mask);
            let bo = d.blue_order();
            records.push((
                row.issue,
                (d.probability(&ns, row.blue + 1)? * RED_OUTCOMES as f64 * 16.).ln(),
                o[..POOL]
                    .iter()
                    .filter(|n| row.red_mask & (1u64 << **n) != 0)
                    .count(),
                bo[..BLUES].contains(&(row.blue + 1)),
            ));
        }
        if !frozen {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            nfit += 1;
        }
    }
    let window = |lo, hi| {
        let rows: Vec<_> = records.iter().filter(|r| r.0 >= lo && r.0 <= hi).collect();
        json!({"n":rows.len(),
            "mean_joint_log_excess_vs_uniform":(!rows.is_empty()).then(||rows.iter().map(|r|r.1).sum::<f64>()/rows.len()as f64),
            "full_six_inside_16":rows.iter().filter(|r|r.2==6).count(),
            "blue_top8_hits":rows.iter().filter(|r|r.3).count(),
            "scope":"事后因果探索回放，参考支持取中性均匀；不是本版本实际首次票面成绩"})
    };
    Ok(
        json!({"version":4,"policy":policy(),"previous_reds":previous.red,
            "ranking":ranking,"scores":raw.to_vec(),
            "blue_ranking":dist.blue_order(),"blue_probabilities":blue_probs.to_vec(),
            "blue_training":{"last_issue":BLUE_TRAIN_END,"history_sha256":bf.training_sha,"n":bf.n,"coefficients":bf.beta},
            "red_weights":{"retained_expert_weights":weights,"protocol_weights":prepared.weights,"extra_ewma_weight_v2":0.3,"extra_ewma_weight_v4":0.},
            "candidate16_full_six_mass":dist.pool_mass(&ranking[..POOL])?,
            "evidence":{"previous_group_score_mass":previous.red.iter().map(|n|raw[(n-1)as usize]).sum::<f64>(),
                "top_six_previous_overlap":order[..6].iter().filter(|i|previous.red_mask&(1<<**i)!=0).count()},
            "plans":[plan],
            "windows":{"development":window(23101,25110),"confirmation_exploratory":window(25111,26115),"tracking":window(26116,u32::MAX)},
            "advantage_established":false,
            "note":"v4去掉v2额外的1/3 EWMA项（EWMA只在六专家融合内计一次），保留未校准的10%参考支持作为v2遗留投票；蓝球改用23100前冻结的去重v2分布。8注在16红池内按精确均衡曝光最大化模型质量，8个不同蓝球按概率与红球质量排序配对。均匀常规奖覆盖结构性高于v2，但模型本身仍未证明优于均匀，不替换v2默认。"}),
    )
}

pub fn reviewed(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = bundle.clone();
    for p in v["plans"].as_array_mut().ok_or("融合v4缺少方案")? {
        let r = crate::recommendations::review_strategy(
            &json!({"title":p["title"],"mode":p["mode"],"structural":true,"tickets":p["tickets"]}),
            draw,
        )?;
        p["summary"] = r["summary"].clone();
        p["ticket_results"] = r["tickets"].clone();
        p["single_jackpot"] = draw
            .map(|_| {
                r["tickets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["red_hits"] == 6 && t["blue_hit"] == true)
            })
            .into();
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::binom;
    #[test]
    fn ewma_is_counted_once_and_repeat_expert_cannot_move_the_score() {
        let mut experts = [[1. / 33.; 33]; 7];
        experts[1] = std::array::from_fn(|i| (i + 1) as f64 / 561.);
        let w = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1];
        let support = [1. / 33.; 33];
        let (before, retained) = scores(&experts, &w, &support).unwrap();
        experts[6] = std::array::from_fn(|i| if i < 6 { 1. / 6. } else { 0. });
        let (after, _) = scores(&experts, &w, &support).unwrap();
        assert_eq!(before, after);
        assert_eq!(retained[6], 0.);
        assert!((before.iter().sum::<f64>() - 1.).abs() < 1e-12);
        let base = fuse7(&experts, &retained);
        for i in 0..33 {
            assert!((before[i] - (0.9 * base[i] + 0.1 * support[i])).abs() < 1e-15);
        }
        let (v2, _) = crate::fusion_model_v2::scores(&experts, &w, &support).unwrap();
        assert!(before.iter().zip(&v2).any(|(a, b)| (a - b).abs() > 1e-9));
        assert!(scores(&experts, &[0., 0., 0., 0., 0., 0., 1.], &support).is_err());
    }
    #[test]
    fn balanced_selection_uses_full_pool_with_exact_exposure_and_certificate() {
        for reverse in [false, true] {
            let w: Vec<f64> = (1..=33)
                .map(|i| if reverse { 34 - i } else { i })
                .map(|v| v as f64)
                .collect();
            let dist = Distribution::new(&w, &[1.; 16]).unwrap();
            let ranking: Vec<u8> = dist.red_order();
            let sel = select_tickets(&dist, &ranking).unwrap();
            assert_eq!(sel.sets.len(), BETS);
            assert_eq!(sel.sets.iter().collect::<BTreeSet<_>>().len(), BETS);
            let union: BTreeSet<u8> = sel.sets.iter().flat_map(|r| r.iter().copied()).collect();
            assert_eq!(union.len(), POOL);
            assert_eq!(union, ranking[..POOL].iter().copied().collect::<BTreeSet<_>>());
            for n in &ranking[..POOL] {
                assert_eq!(sel.sets.iter().filter(|r| r.contains(n)).count(), 3);
            }
            assert!(sel.red_mass <= sel.pool_upper_bound + 1e-15);
            assert!(sel.pool_upper_bound - sel.red_mass >= -1e-15);
            assert!(sel.evaluated >= binom(16, 6) as usize);
            let again = select_tickets(&dist, &ranking).unwrap();
            assert_eq!(sel.sets, again.sets);
            let masses: Vec<f64> = sel
                .sets
                .iter()
                .map(|s| dist.pool_mass(s).unwrap())
                .collect();
            assert!(masses.windows(2).all(|p| p[0] >= p[1] - 1e-15));
        }
        let flat = Distribution::new(&[1.; 33], &[1.; 16]).unwrap();
        let ranking: Vec<u8> = (1..=33).collect();
        let sel = select_tickets(&flat, &ranking).unwrap();
        assert!((sel.red_mass - 8. / binom(33, 6) as f64).abs() < 1e-12);
        assert!((sel.pool_upper_bound - sel.red_mass).abs() < 1e-12);
    }
    #[test]
    fn eight_distinct_blues_structurally_beat_v2_uniform_coverage() {
        let w: Vec<f64> = (1..=33).map(|v| v as f64).collect();
        let dist = Distribution::new(&w, &(1..=16).map(|v| v as f64).collect::<Vec<_>>()).unwrap();
        let ranking = dist.red_order();
        let sel = select_tickets(&dist, &ranking).unwrap();
        let blues = &dist.blue_order()[..BLUES];
        assert_eq!(blues.iter().collect::<BTreeSet<_>>().len(), BLUES);
        let tickets: Vec<Value> = sel
            .sets
            .iter()
            .zip(blues)
            .map(|(r, b)| json!({"red":r,"blue":b}))
            .collect();
        let r4 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&json!(tickets)).unwrap())
            .unwrap();
        let coverage4 = BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r4;
        assert!(coverage4 >= 8 * RED_OUTCOMES);
        let v2_max = 2 * RED_OUTCOMES + 14 * 8 * (binom(6, 4) * binom(27, 2) + 6 * 27 + 1);
        assert!(coverage4 > v2_max);
        let paired = mass(&dist, &json!(tickets)).unwrap();
        let mut red_desc = sel.sets.clone();
        red_desc.sort_by(|a, b| dist.pool_mass(b).unwrap().total_cmp(&dist.pool_mass(a).unwrap()));
        let swapped = json!(red_desc
            .iter()
            .zip(blues.iter().rev())
            .map(|(r, b)| json!({"red":r,"blue":b}))
            .collect::<Vec<_>>());
        assert!(paired >= mass(&dist, &swapped).unwrap() - 1e-15);
    }
}