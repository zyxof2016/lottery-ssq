//! Frozen fusion v7: the v4 score with a 24-red ticket union (eight bets,
//! every pool red exactly twice) and eight distinct frozen blue-v2 blues.
//! Responds to the coverage objective "cover all six drawn reds with the
//! smallest pool": 24 is the compression tier whose uniform full-six baseline
//! is C(24,6)/C(33,6), not the 16-red compression tier. No predictive claim.
use crate::{
    blue_model_v2 as blue,
    draw::{data_sha256, Draw},
    fusion_model_v3::{mass, Distribution},
    fusion_model_v4::{scores, Selection},
    model::roll_all,
    protocol::Protocol,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

pub const POOL: usize = 24;
pub const BETS: usize = 8;
pub const BLUES: usize = 8;
const RED_OUTCOMES: u64 = 1_107_568;

pub fn policy() -> Value {
    json!({"version":7,"experimental":true,"prediction_advantage_established":false,
        "red_score":"v4_score_0.9_six_expert_renormalized_fusion_plus_0.1_reference_union_support",
        "ewma_counted_once":true,"explicit_repeat_weight":0,
        "red_pool":POOL,"pool_rule":"top24_of_v4_score",
        "objective":"cover_all_six_drawn_reds_with_compressed_pool_24_not_16",
        "uniform_full_six_baseline_24":crate::stats::binom(24,6) as f64/RED_OUTCOMES as f64,
        "uniform_full_six_baseline_16":crate::stats::binom(16,6) as f64/RED_OUTCOMES as f64,
        "singles":{"bets":BETS,"cost":BETS*2,"red_exposure_each":BETS*6/POOL,"blues":BLUES,
            "blue_model":"deduplicated_blue_v2","blue_training_last_issue":blue::TRAIN_END,
            "red_selection":"balanced_feasible_start_then_deterministic_pairwise_swaps_maximizing_model_mass",
            "pairing":"red_mass_desc_with_blue_probability_desc_rearrangement_optimum"},
        "coverage_guarantee":"8个不同蓝球无条件覆盖8*C(33,6)个完整结果；红球并集恰为24，每红恰2次。",
        "dantuo":"本版本只试验8注单式；v2至v5保持各自原档。",
        "automatic_promotion":false})
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
        .filter(|d| d.issue <= blue::TRAIN_END)
        .cloned()
        .collect();
    let sha = data_sha256(&training);
    let mut lock = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "融合v7蓝球拟合缓存异常")?;
    if let Some((key, f)) = &*lock {
        if key == &sha {
            return Ok(f.clone());
        }
    }
    let samples = rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= blue::TRAIN_END)
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

/// Same deterministic balanced-start plus pairwise-swap improvement as v4,
/// generalized to the 24-red pool where every red appears exactly twice.
pub fn select_tickets(dist: &Distribution, ranking: &[u8]) -> Result<Selection, String> {
    if ranking.len() != 33 || BETS * 6 % POOL != 0 {
        return Err("融合v7组票参数无效".into());
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
            return Err("融合v7曝光均衡未闭合".into());
        }
    }
    if sets.iter().collect::<BTreeSet<_>>().len() != BETS {
        return Err("融合v7票面重复".into());
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
            serde_json::from_value(t["red"].clone()).map_err(|_| "融合v7红票无效")?;
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
    let reds: BTreeSet<u8> = sel.sets.iter().flat_map(|r| r.iter().copied()).collect();
    let r7 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&ticket_json)?)?;
    let coverage7 = BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r7;
    let old = legacy["bundle"]["singles"]
        .as_array()
        .and_then(|a| a.iter().find(|p| p["red_count"] == 16))
        .ok_or("缺少v2同预算16红首次对照")?;
    let old_mass = mass(&dist, &old["tickets"])?;
    let old_blues = old["blues"].as_array().ok_or("v2对照缺少蓝球")?.len() as u64;
    let r2 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&old["tickets"])?)?;
    let coverage2 = old_blues * RED_OUTCOMES + (16 - old_blues) * r2;
    let plan = json!({"id":"cover24_8","title":"融合覆盖24试验v7 · 8注","mode":"joint",
        "bet_count":BETS,"cost":BETS*2,"red_count":reds.len(),"reds":reds,"blues":blues,
        "red_dan":[],"red_tuo":[],"tickets":tickets,
        "model_joint_mass":joint_mass,"ratio_to_uniform_budget":joint_mass/(BETS as f64/(RED_OUTCOMES as f64*16.)),
        "uniform_any_prize_outcomes":coverage7,"uniform_jackpot_outcomes":BETS,
        "red_prize_outcomes_without_blue":r7,
        "certificate":{"exposure_balance":"24红每红恰好2次","evaluations":sel.evaluated,
            "pool_unconstrained_upper_bound_mass":sel.pool_upper_bound,"achieved_red_mass":sel.red_mass,
            "absolute_red_mass_gap":sel.pool_upper_bound-sel.red_mass,
            "scope":"并集24红为覆盖目标档（均匀全6基准12.15%对16红0.72%）；票面为均衡可行起点加确定性交换，差距证书只针对本模型，不宣称预测优势。"},
        "same_budget_v2_comparison":{"id":old["id"],"bets":old["bet_count"],"cost":old["cost"],
            "model_joint_mass":old_mass,"relative_model_gain":joint_mass/old_mass-1.,
            "uniform_any_prize_outcomes":coverage2,"used_reds":old["reds"],"used_blues":old["blues"]}});
    let previous = draws.last().ok_or("没有开奖数据")?;
    let mut records = Vec::new();
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    let neutral = [1. / 33.; 33];
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if row.issue >= 23101 && (frozen || nfit >= 30) {
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let (s, _) = scores(&row.expert_p, &w, &neutral)?;
            let o = rank_desc(&s);
            let q = blue::distribution(&bf.beta, &blue::features(&row.blue_p)?);
            let bo = rank_desc(&q);
            records.push((
                row.issue,
                o[..POOL]
                    .iter()
                    .filter(|n| row.red_mask & (1u64 << **n) != 0)
                    .count(),
                o[..16]
                    .iter()
                    .filter(|n| row.red_mask & (1u64 << **n) != 0)
                    .count(),
                bo[..BLUES].contains(&(row.blue as usize)),
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
            "full_six_inside_24":rows.iter().filter(|r|r.1==6).count(),
            "full_six_inside_16":rows.iter().filter(|r|r.2==6).count(),
            "blue_top8_hits":rows.iter().filter(|r|r.3).count(),
            "scope":"事后因果探索回放，参考支持取中性均匀；不是本版本实际首次票面成绩"})
    };
    Ok(
        json!({"version":7,"policy":policy(),"previous_reds":previous.red,
            "ranking":ranking,"scores":raw.to_vec(),
            "blue_ranking":dist.blue_order(),"blue_probabilities":blue_probs.to_vec(),
            "blue_training":{"last_issue":blue::TRAIN_END,"history_sha256":bf.training_sha,"n":bf.n,"coefficients":bf.beta},
            "red_weights":{"retained_expert_weights":weights,"protocol_weights":prepared.weights},
            "candidate24_full_six_mass":dist.pool_mass(&ranking[..POOL])?,
            "evidence":{"previous_group_score_mass":previous.red.iter().map(|n|raw[(n-1)as usize]).sum::<f64>(),
                "top_six_previous_overlap":order[..6].iter().filter(|i|previous.red_mask&(1<<**i)!=0).count()},
            "plans":[plan],
            "windows":{"development":window(23101,25110),"confirmation_exploratory":window(25111,26115),"tracking":window(26116,u32::MAX)},
            "advantage_established":false,
            "note":"v7沿用v4评分（EWMA单计＋10%参考支持），但票面并集改为24红、每红恰2次，配8个不同冻结蓝v2球：把「覆盖全部开奖红球」放在压缩之上，24红均匀全6基准12.15%远高于16红0.72%。仍不保证每期全包、不宣称预测优势；自首次封存期起为融合推荐默认（2026-11-19用户决定），v4保留可切换。"}),
    )
}

pub fn reviewed(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = bundle.clone();
    for p in v["plans"].as_array_mut().ok_or("融合v7缺少方案")? {
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
        p["full_six_in_pool"] = draw.map(|d| {
            p["reds"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|n| d.red.iter().any(|r| json!(r) == **n))
                .count()
                == 6
        }).into();
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balanced_24_selection_uses_full_pool_twice_each_with_certificate() {
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
            for n in &ranking[..POOL] {
                assert_eq!(sel.sets.iter().filter(|r| r.contains(n)).count(), 2);
            }
            assert!(sel.red_mass <= sel.pool_upper_bound + 1e-15);
            assert!(sel.evaluated >= crate::stats::binom(24, 6) as usize);
            let again = select_tickets(&dist, &ranking).unwrap();
            assert_eq!(sel.sets, again.sets);
        }
    }
    #[test]
    fn eight_blues_and_24_union_beat_v2_uniform_coverage() {
        let w: Vec<f64> = (1..=33).map(|v| v as f64).collect();
        let dist = Distribution::new(&w, &(1..=16).map(|v| v as f64).collect::<Vec<_>>()).unwrap();
        let ranking = dist.red_order();
        let sel = select_tickets(&dist, &ranking).unwrap();
        let blues = &dist.blue_order()[..BLUES];
        let tickets: Vec<Value> = sel
            .sets
            .iter()
            .zip(blues)
            .map(|(r, b)| json!({"red":r,"blue":b}))
            .collect();
        let r7 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&json!(tickets)).unwrap())
            .unwrap();
        let coverage7 = BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r7;
        let v2_max = 2 * RED_OUTCOMES + 14 * 8 * (crate::stats::binom(6, 4) * crate::stats::binom(27, 2) + 6 * 27 + 1);
        assert!(coverage7 > v2_max);
        assert!(r7 > 8 * (crate::stats::binom(6, 4) * crate::stats::binom(27, 2) + 6 * 27 + 1) / 2);
    }
}