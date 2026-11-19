//! Frozen fusion v5: the red pool is limited to 16 numbers chosen by a signal
//! trained to pull the last winning red forward (minimum covering pool). The
//! ridge is predeclared weaker than cover-rank v1 and fitted only on <=23100;
//! tickets reuse the v4 balanced 16-red / 8-blue coverage structure. The score
//! is a covering signal, not a winning probability, and nothing is promoted.
use crate::{
    blue_model_v2 as blue,
    cover_rank,
    draw::{data_sha256, Draw},
    fusion_model_v3::{mass, Distribution},
    fusion_model_v4::select_tickets,
    model::roll_all,
    protocol::Protocol,
    red_model_v2 as red,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};

pub const RIDGE: f64 = 0.005;
pub const POOL: usize = 16;
pub const BETS: usize = 8;
pub const BLUES: usize = 8;
pub const REQUIRED_MEAN_GAIN: f64 = 1.0;
/// E[max of six distinct uniform ranks over 1..33] = 6*34/7.
pub const UNIFORM_MEAN_COVER: f64 = 6. * 34. / 7.;
const RED_OUTCOMES: u64 = 1_107_568;

pub fn policy() -> Value {
    json!({"version":5,"experimental":true,"prediction_advantage_established":false,
        "objective":"minimum_mean_last_winning_rank_with_pool_limited_to_16_reds",
        "red_signal":"cover_rank_loss_mean_logsumexp_of_per_winner_mean_softplus",
        "ridge":RIDGE,"ridge_predeclared":true,"cover_rank_v1_ridge":cover_rank::RIDGE,
        "training_last_issue":red::TRAIN_END,"expert_warmup_rows":cover_rank::WARMUP_ROWS,
        "score_kind":"cover_rank_signal_not_probability",
        "red_pool":POOL,"pool_rule":"top16_of_cover_signal",
        "ticket_weights":"exp(signal)_monotone_transform_for_mass_objective_only",
        "singles":{"bets":BETS,"cost":BETS*2,"red_exposure_each":BETS*6/POOL,"blues":BLUES,
            "blue_model":"deduplicated_blue_v2","blue_training_last_issue":blue::TRAIN_END,
            "structure":"v4_balanced_feasible_start_plus_deterministic_pairwise_swaps"},
        "required_mean_gain_vs_weighted_and_uniform":REQUIRED_MEAN_GAIN,
        "uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
        "dantuo":"本版本只试验8注单式；v2、v3、v4保持各自原档。",
        "automatic_promotion":false})
}

#[derive(Clone)]
struct CoverFit {
    beta: [f64; 6],
    training_sha: String,
    n: usize,
}
fn cover_fit(draws: &[Draw], rows: &[crate::model::Row]) -> Result<CoverFit, String> {
    static CACHE: OnceLock<Mutex<Option<(String, CoverFit)>>> = OnceLock::new();
    let training: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= red::TRAIN_END)
        .cloned()
        .collect();
    let sha = data_sha256(&training);
    let mut lock = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "融合v5覆盖拟合缓存异常")?;
    if let Some((key, f)) = &*lock {
        if key == &sha {
            return Ok(f.clone());
        }
    }
    let samples = rows
        .iter()
        .skip(cover_rank::WARMUP_ROWS)
        .filter(|r| r.issue <= red::TRAIN_END)
        .map(|r| {
            Ok(red::Sample {
                x: red::features(&r.expert_p)?,
                mask: r.red_mask,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let f = cover_rank::fit_with(&samples, RIDGE)?;
    let result = CoverFit {
        beta: f.beta,
        training_sha: sha.clone(),
        n: f.n,
    };
    *lock = Some((sha, result.clone()));
    Ok(result)
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
        .map_err(|_| "融合v5蓝球拟合缓存异常")?;
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

pub fn cover_prefix(order: &[usize; 33], mask: u64) -> usize {
    order
        .iter()
        .enumerate()
        .filter(|(_, i)| mask & (1 << **i) != 0)
        .map(|(r, _)| r + 1)
        .max()
        .unwrap_or(33)
}

fn red_sets(tickets: &Value) -> Result<std::collections::BTreeSet<[u8; 6]>, String> {
    let mut out = std::collections::BTreeSet::new();
    for t in tickets.as_array().ok_or("缺少完整票面")? {
        let mut r: [u8; 6] =
            serde_json::from_value(t["red"].clone()).map_err(|_| "融合v5红票无效")?;
        r.sort_unstable();
        out.insert(r);
    }
    Ok(out)
}

pub fn build(draws: &[Draw], parent: &Value, legacy: &Value) -> Result<Value, String> {
    let prepared = crate::eval::prepare_forecast(draws)?;
    let snap = roll_all(draws)?;
    let cf = cover_fit(draws, &snap.rows)?;
    let bf = blue_fit(draws, &snap.rows)?;
    let signal = cover_rank::signal(&cf.beta, &snap.next_red)?;
    let order = rank_desc(&signal);
    let ranking: Vec<u8> = order.iter().map(|i| (i + 1) as u8).collect();
    let weights: Vec<f64> = signal.iter().map(|v| v.exp()).collect();
    let blue_probs = blue::distribution(&bf.beta, &blue::features(&snap.next_blue)?);
    let dist = Distribution::new(&weights, &blue_probs)?;
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
    let reds: std::collections::BTreeSet<u8> =
        sel.sets.iter().flat_map(|r| r.iter().copied()).collect();
    let r5 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&ticket_json)?)?;
    let coverage5 = BLUES as u64 * RED_OUTCOMES + (16 - BLUES as u64) * r5;
    let old = legacy["bundle"]["singles"]
        .as_array()
        .and_then(|a| a.iter().find(|p| p["red_count"] == 16))
        .ok_or("缺少v2同预算16红首次对照")?;
    let old_mass = mass(&dist, &old["tickets"])?;
    let old_blues = old["blues"].as_array().ok_or("v2对照缺少蓝球")?.len() as u64;
    let r2 = crate::fusion_v3_comparison::red_prize_outcomes(&red_sets(&old["tickets"])?)?;
    let coverage2 = old_blues * RED_OUTCOMES + (16 - old_blues) * r2;
    let plan = json!({"id":"cover16_8","title":"融合最小覆盖池试验v5 · 8注","mode":"joint",
        "bet_count":BETS,"cost":BETS*2,"red_count":reds.len(),"reds":reds,"blues":blues,
        "red_dan":[],"red_tuo":[],"tickets":tickets,
        "model_joint_mass":joint_mass,"ratio_to_uniform_budget":joint_mass/(BETS as f64/(RED_OUTCOMES as f64*16.)),
        "uniform_any_prize_outcomes":coverage5,"uniform_jackpot_outcomes":BETS,
        "red_prize_outcomes_without_blue":r5,
        "certificate":{"exposure_balance":"16红每红恰好3次","evaluations":sel.evaluated,
            "pool_unconstrained_upper_bound_mass":sel.pool_upper_bound,"achieved_red_mass":sel.red_mass,
            "absolute_red_mass_gap":sel.pool_upper_bound-sel.red_mass,
            "scope":"池限定为覆盖信号前16；票面为v4均衡结构加确定性交换，差距证书只针对exp(signal)权重模型，不宣称预测优势。"},
        "same_budget_v2_comparison":{"id":old["id"],"bets":old["bet_count"],"cost":old["cost"],
            "model_joint_mass":old_mass,"relative_model_gain":joint_mass/old_mass-1.,
            "uniform_any_prize_outcomes":coverage2,"used_reds":old["reds"],"used_blues":old["blues"]}});
    let previous = draws.last().ok_or("没有开奖数据")?;
    let mut records = Vec::new();
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if row.issue >= 23101 && (frozen || nfit >= 30) {
            let s = cover_rank::signal(&cf.beta, &row.expert_p)?;
            let o = rank_desc(&s);
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let weighted = rank_desc(&fuse7(&row.expert_p, &w));
            let q = blue::distribution(&bf.beta, &blue::features(&row.blue_p)?);
            let bo = rank_desc(&q);
            records.push([
                row.issue as f64,
                cover_prefix(&o, row.red_mask) as f64,
                cover_prefix(&weighted, row.red_mask) as f64,
                o[..POOL]
                    .iter()
                    .filter(|n| row.red_mask & (1u64 << **n) != 0)
                    .count() as f64,
                f64::from(bo[..BLUES].contains(&(row.blue as usize))),
            ]);
        }
        if !frozen {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            nfit += 1;
        }
    }
    let window = |lo, hi| {
        let rows: Vec<_> = records
            .iter()
            .filter(|r| r[0] >= lo as f64 && r[0] <= hi as f64)
            .collect();
        let mean = |i: usize| {
            (!rows.is_empty()).then(|| rows.iter().map(|r| r[i]).sum::<f64>() / rows.len() as f64)
        };
        json!({"n":rows.len(),
            "mean_minimum_full_cover_prefix":mean(1),
            "mean_weighted_full_cover_prefix":mean(2),
            "full_six_inside_16":rows.iter().filter(|r|r[3]==6.).count(),
            "blue_top8_hits":rows.iter().filter(|r|r[4]==1.).count(),
            "scope":"事后因果探索回放，不是本版本实际首次票面成绩"})
    };
    let confirm = window(25111, 26115);
    let gain_weighted = confirm["mean_weighted_full_cover_prefix"]
        .as_f64()
        .zip(confirm["mean_minimum_full_cover_prefix"].as_f64())
        .map(|(w, c)| w - c);
    let gain_uniform = confirm["mean_minimum_full_cover_prefix"]
        .as_f64()
        .map(|c| UNIFORM_MEAN_COVER - c);
    let clears = gain_weighted.is_some_and(|g| g >= REQUIRED_MEAN_GAIN)
        && gain_uniform.is_some_and(|g| g >= REQUIRED_MEAN_GAIN);
    Ok(
        json!({"version":5,"policy":policy(),"previous_reds":previous.red,
            "ranking":ranking,"scores":signal.to_vec(),
            "blue_ranking":dist.blue_order(),"blue_probabilities":blue_probs.to_vec(),
            "cover_training":{"last_issue":red::TRAIN_END,"history_sha256":cf.training_sha,"n":cf.n,"ridge":RIDGE,"coefficients":cf.beta},
            "blue_training":{"last_issue":blue::TRAIN_END,"history_sha256":bf.training_sha,"n":bf.n,"coefficients":bf.beta},
            "plans":[plan],
            "windows":{"development":window(23101,25110),"confirmation_exploratory":confirm,"tracking":window(26116,u32::MAX)},
            "gate":{"required_mean_gain":REQUIRED_MEAN_GAIN,"uniform_mean_cover_rank":UNIFORM_MEAN_COVER,
                "confirmation_gain_vs_weighted":gain_weighted,"confirmation_gain_vs_uniform":gain_uniform,
                "clears_predeclared_mean_gate":clears,"replaces_weighted":false,"rewrites_archives":false},
            "advantage_established":false,
            "note":"v5把融合推荐限定在覆盖信号前16红，目标是最小化盖住全部6个开奖红球所需的最靠后名次；正则0.005事先写定、只在23100及以前拟合，不在确认窗搜索。票面沿用v4均衡16红8注与8个不同冻结蓝v2球。盖住名次是排序证据不是概率；门槛未过不替换任何默认与已封存票面。"}),
    )
}

pub fn reviewed(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = bundle.clone();
    for p in v["plans"].as_array_mut().ok_or("融合v5缺少方案")? {
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
    fn cover_prefix_is_the_last_winning_rank_and_signal_order_is_deterministic() {
        let mut s = [0.; 33];
        for i in 0..33 {
            s[i] = (33 - i) as f64;
        }
        let o = rank_desc(&s);
        let mask = (1 << 0) | (1 << 5) | (1 << 20);
        assert_eq!(cover_prefix(&o, mask), 21);
        assert_eq!(cover_prefix(&o, 0), 33);
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let snap = roll_all(&draws).unwrap();
        let a = cover_fit(&draws, &snap.rows).unwrap();
        let b = cover_fit(&draws, &snap.rows).unwrap();
        assert_eq!(a.beta, b.beta);
        let sig = cover_rank::signal(&a.beta, &snap.next_red).unwrap();
        let w: Vec<f64> = sig.iter().map(|v| v.exp()).collect();
        assert!(w.iter().all(|v| v.is_finite() && *v > 0.));
        assert_eq!(rank_desc(&sig), rank_desc(&w.try_into().unwrap()));
    }
    #[test]
    fn weaker_predeclared_ridge_separates_more_than_v1_on_training() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let snap = roll_all(&draws).unwrap();
        let samples = snap
            .rows
            .iter()
            .skip(cover_rank::WARMUP_ROWS)
            .filter(|r| r.issue <= red::TRAIN_END)
            .map(|r| {
                Ok(red::Sample {
                    x: red::features(&r.expert_p).unwrap(),
                    mask: r.red_mask,
                })
            })
            .collect::<Result<Vec<_>, String>>()
            .unwrap();
        let v1 = cover_rank::fit(&samples).unwrap();
        let v5 = cover_rank::fit_with(&samples, RIDGE).unwrap();
        let norm = |b: &[f64; 6]| b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm(&v5.beta) > norm(&v1.beta));
    }
}