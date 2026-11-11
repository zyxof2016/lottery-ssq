//! Read-only attribution and complete-ticket optimization for sealed fusion v2.
use crate::{
    calibration_model_v1 as model,
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    select::rank_desc,
    ticket_search::{Selection, Table, Ticket},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Mutex, OnceLock},
};
const SETS: f64 = 1_107_568.;
const UNIFORM_EIGHT: f64 = 8. / (SETS * 16.);
const RANK_TOLERANCE: f64 = 1e-12 / (SETS * 16.);
fn mask(ns: &Value) -> Result<u64, String> {
    let numbers: Vec<u8> = serde_json::from_value(ns.clone()).map_err(|_| "红球数组无效")?;
    if numbers.iter().any(|n| !(1..=33).contains(n))
        || numbers.iter().collect::<BTreeSet<_>>().len() != numbers.len()
    {
        return Err("红球越界或重复".into());
    }
    Ok(numbers.iter().fold(0, |m, n| m | (1u64 << (n - 1))))
}
fn table(
    pool: &[u8],
    red: &[[f64; 33]; 7],
    blue: &[[f64; 16]; 6],
    study: &model::Study,
    previous: u64,
) -> Result<Table, String> {
    let active: Vec<_> = red
        .iter()
        .zip(&study.red.weights)
        .filter(|(_, w)| **w > 0.)
        .map(|(p, w)| (model::SetDist::stratified(*p, previous), *w))
        .collect();
    let bp: [f64; 16] = std::array::from_fn(|b| {
        blue.iter()
            .zip(&study.blue.weights)
            .map(|(p, w)| p[b] * w)
            .sum::<f64>()
            + study.blue.weights[6] / 16.
    });
    // Cache a whole row so red ESP probability is evaluated once, not 16 times.
    let (mut last_mask, mut last_mass) = (0, 0.);
    Table::new(pool, |r, b| {
        if r != last_mask {
            last_mask = r;
            last_mass = active
                .iter()
                .map(|(m, w)| m.probability(r) * w)
                .sum::<f64>()
                + study.red.weights[7] / SETS;
        }
        last_mass * bp[(b - 1) as usize]
    })
}
fn tickets(v: &Value) -> Result<Vec<Ticket>, String> {
    v.as_array()
        .ok_or("票面数组无效")?
        .iter()
        .map(|t| {
            Ok(Ticket {
                red: mask(&t["red"])?,
                blue: serde_json::from_value(t["blue"].clone()).map_err(|_| "蓝球无效")?,
            })
        })
        .collect()
}
fn metrics(
    id: &str,
    s: &Selection,
    baseline: f64,
    pool_mass: f64,
    actual: Option<Ticket>,
) -> Value {
    let reds: BTreeSet<_> = s.tickets.iter().map(|t| t.red).collect();
    let blues: BTreeSet<_> = s.tickets.iter().map(|t| t.blue).collect();
    json!({"id":id,"bets":s.tickets.len(),"cost":s.tickets.len()*2,"model_joint_mass":s.mass,"ratio_to_uniform_eight":s.mass/UNIFORM_EIGHT,
        "model_gain_vs_original":s.mass/baseline-1.,"fraction_of_pool_all_blues_mass":s.mass/pool_mass,
        "distinct_red_sets":reds.len(),"red_union_count":reds.iter().fold(0u64,|a,m|a|m).count_ones(),"blue_count":blues.len(),
        "complete_6_plus_1_hit":actual.map(|t|s.tickets.contains(&t)),"actual_recommendation":id=="sealed_original"})
}
fn compare(
    table: &Table,
    original: &[Ticket],
    actual: Option<Ticket>,
    sealed: bool,
) -> Result<Value, String> {
    let saved = table.evaluate(original)?;
    if original.len() != 8 {
        return Err("本核验固定8注16元".into());
    }
    let mut quotas = [0usize; 16];
    for t in original {
        quotas[(t.blue - 1) as usize] += 1;
    }
    let red_only = table.optimize(8, Some(&quotas))?;
    let blue_only = table.optimize_blues_only(original)?;
    let joint = table.optimize(8, None)?;
    let selected_blues: Vec<_> = (1..=16).filter(|b| quotas[(b - 1) as usize] > 0).collect();
    let pool_mass = table.total_mass(&(1..=16).collect::<Vec<_>>())?;
    let selected_mass = table.total_mass(&selected_blues)?;
    let rank = actual
        .filter(|t| t.red & table.pool_mask == t.red)
        .map(|t| table.rank_interval(t, RANK_TOLERANCE))
        .transpose()?;
    let expected_tie_hit = rank.map(|(lo, hi)| {
        if lo > 8 {
            0.
        } else {
            ((9 - lo).min(hi - lo + 1)) as f64 / (hi - lo + 1) as f64
        }
    });
    Ok(
        json!({"rows":[metrics(if sealed {"sealed_original"} else {"causal_balanced_baseline"},&saved,saved.mass,pool_mass,actual),metrics("red_grouping_same_blue_counts",&red_only,saved.mass,pool_mass,actual),
        metrics("blue_only_same_red_sets",&blue_only,saved.mass,pool_mass,actual),metrics("joint_top_eight",&joint,saved.mass,pool_mass,actual)],
        "pool_all_six_mass":pool_mass,"pool_and_original_blues_mass":selected_mass,"original_ticket_retention_given_pool_and_original_blues":saved.mass/selected_mass,
        "original_blue_cartesian_outcomes":table.outcomes()/16*selected_blues.len(),"original_blue_counts":quotas,"certificate":{"method":"exhaustive_complete_outcomes_then_top_k_or_top_k_per_blue","complete_outcomes_evaluated":table.outcomes(),
        "same_original_red_pool":true,"blue_candidates":16,"bets":8,"cost":16,"unrestricted_optimal":true,"absolute_search_gap":0.,"conditional_blue_supported":true,
        "scope":"同原候选红池、8个不同6＋1结果、给定完整概率模型内的最优；不证明真实预测优势，不要求原16红全部出现在8注中"},
        "winning_result_rank_in_pool_all_blues":rank.map(|(a,b)|json!([a,b])),"rank_absolute_numeric_tolerance":RANK_TOLERANCE,
        "joint_top_eight_uniform_numeric_tie_expected_hit":expected_tie_hit,"ties_note":"数值同分区间只作诊断；期望为截断同分组内均匀选择的覆盖分数，不算实际中奖次数。"}),
    )
}
fn blue_explanation(
    prefix: &[Draw],
    actual: &Draw,
    original_blues: &[u8],
) -> Result<Value, String> {
    let p = crate::eval::prepare_forecast(prefix)?;
    let selected: Vec<_> = p.forecast.blue_order[..2]
        .iter()
        .map(|b| *b as u8 + 1)
        .collect();
    if selected != original_blues {
        return Err("原档蓝球与因果前缀原排序不一致".into());
    }
    let b = (actual.blue - 1) as usize;
    let names = [
        "frequency",
        "ewma",
        "overdue",
        "hot",
        "anti_repeat",
        "markov",
    ];
    let experts:Vec<_>=(0..6).map(|i|{
        let without: [f64;16]=std::array::from_fn(|j|(0..6).filter(|k|*k!=i).map(|k|p.blue_weights[k]*p.expert_blue[k][j]).sum());
        json!({"expert":names[i],"weight":p.blue_weights[i],"winning_blue_expert_rank":rank_desc(&p.expert_blue[i]).iter().position(|n|*n==b).unwrap()+1,
            "winning_blue_weighted_component":p.blue_weights[i]*p.expert_blue[i][b],"winning_blue_rank_without_this_expert":rank_desc(&without).iter().position(|n|*n==b).unwrap()+1})
    }).collect();
    Ok(
        json!({"winning_blue":actual.blue,"original_rank":p.forecast.blue_order.iter().position(|n|*n==b).unwrap()+1,"selected_blues":selected,"selected":selected.contains(&actual.blue),
        "winning_blue_original_score":p.forecast.blue[b],"experts":experts,"scope":"用首次原档输入前缀重算原蓝球公式，并核对原2蓝；删除单个专家的名次仅为归因，不是历史另发推荐或权重采用依据。"}),
    )
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let target = fmt_issue(last.issue + 1);
    let mut issues: Vec<_> = draws
        .iter()
        .rev()
        .filter(|d| {
            root.join(format!("fusion-v2/{}.json", fmt_issue(d.issue)))
                .exists()
        })
        .take(12)
        .map(|d| fmt_issue(d.issue))
        .collect();
    issues.reverse();
    if root.join(format!("fusion-v2/{target}.json")).exists() {
        issues.push(target.clone());
    }
    let mut archives = Vec::new();
    for issue in issues {
        let e: Value = serde_json::from_slice(&crate::fusion_v2::artifact(root, draws, &issue)?)
            .map_err(|_| "v2原档JSON无效")?;
        archives.push(e);
    }
    let key = format!(
        "{}:{:?}",
        data_sha256(draws),
        archives.iter().map(|e| &e["sha256"]).collect::<Vec<_>>()
    );
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "完整票面核验缓存异常")?;
    if let Some((k, v)) = cache.as_ref() {
        if *k == key {
            return Ok(v.clone());
        }
    }
    let study = model::study(draws)?;
    let mut cases = Vec::new();
    for e in archives {
        let issue = e["payload"]["issue"].as_str().ok_or("v2原档缺少期号")?;
        let parsed: u32 = issue.parse().map_err(|_| "v2期号无效")?;
        let plan = e["payload"]["bundle"]["singles"]
            .as_array()
            .ok_or("缺少v2单式")?
            .iter()
            .find(|p| p["red_count"] == 16)
            .ok_or("缺少v2的16红方案")?;
        let pool: Vec<u8> =
            serde_json::from_value(plan["reds"].clone()).map_err(|_| "v2红池无效")?;
        let original = tickets(&plan["tickets"])?;
        let actual = draws.iter().find(|d| d.issue == parsed);
        let (red, blue, previous) = if actual.is_some() {
            let s = study
                .samples
                .iter()
                .find(|s| s.row.issue == parsed)
                .ok_or("缺少当期因果专家特征")?;
            (&s.row.expert_p, &s.row.blue_p, s.previous)
        } else {
            (&study.next_red, &study.next_blue, study.previous)
        };
        let table = table(&pool, red, blue, &study, previous)?;
        let a = actual.map(|d| Ticket {
            red: d.red_mask,
            blue: d.blue,
        });
        let comparison = compare(&table, &original, a, true)?;
        let mut v = json!({"issue":issue,"pending":actual.is_none(),"archive_sha256":e["sha256"],"input_history_sha256":e["payload"]["input_history_sha256"],"data_cutoff":e["payload"]["data_cutoff"],
            "comparison":comparison,"raw_url":format!("api/reference-pools/fusion-v2/archives/{issue}"),"original_actual_recommendation":true});
        if let Some(d) = actual {
            let prefix: Vec<_> = draws
                .iter()
                .filter(|x| x.issue < d.issue)
                .cloned()
                .collect();
            if json!(data_sha256(&prefix)) != e["payload"]["input_history_sha256"] {
                return Err("票面核验前缀摘要不符".into());
            }
            let original_blues: Vec<u8> =
                serde_json::from_value(e["payload"]["bundle"]["blues"].clone())
                    .map_err(|_| "原蓝球无效")?;
            v["draw"] = json!({"red":d.red,"blue":d.blue});
            v["blue_attribution"] = blue_explanation(&prefix, d, &original_blues)?;
            v["red_pool_missed"] = json!(d
                .red
                .iter()
                .filter(|n| table.pool_mask & (1u64 << (**n - 1)) == 0)
                .collect::<Vec<_>>());
        }
        cases.push(v);
    }
    let values: Vec<_> = study
        .samples
        .iter()
        .filter(|s| s.row.issue >= 25111 && s.row.issue <= 26115)
        .map(|s| model::sample_score(s, &study).iter().sum::<f64>())
        .collect();
    let out = json!({"version":1,"kind":"same_budget_complete_ticket_audit","target_issue":target,"cutoff_issue":fmt_issue(last.issue),"data_sha256":data_sha256(draws),"cases":cases,
        "uniform_eight_ticket_mass":UNIFORM_EIGHT,"source_sha256":sha256_hex(include_str!("ticket_search.rs").as_bytes()),
        "model":"frozen_overlap_stratified_calibration_v1","training_last_issue":fmt_issue(model::TRAIN_END),"confirmation_n":values.len(),"confirmation_joint_log_excess":(!values.is_empty()).then(||values.iter().sum::<f64>()/values.len()as f64),
        "promoted":false,"tickets_published":false,"writes_archives":false,"reason":"搜索可提高此模型内同预算完整票面的分数；完整概率模型的确认窗仍未优于均匀，研究对照不作为新购票推荐。保留各期首次票面。",
        "scope":"最多最近12期实际v2及当前已封存v2；仅sealed_original行属于首次票面，其余均为相同输入的事后算法研究。不同方案独立比较，预算不合并。"});
    *cache = Some((key, out.clone()));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_budget_retention_and_replay_labels_do_not_claim_actual_recommendations() {
        let pool = [1, 2, 3, 4, 5, 6, 7, 8];
        let t = Table::new(&pool, |_, _| 1. / (SETS * 16.)).unwrap();
        let original: Vec<_> = crate::plan::combinations(&pool, 6)
            .unwrap()
            .iter()
            .take(8)
            .enumerate()
            .map(|(i, r)| Ticket {
                red: r.iter().fold(0, |m, n| m | (1u64 << (n - 1))),
                blue: (i % 2 + 1) as u8,
            })
            .collect();
        let actual = Some(original[0]);
        let research = compare(&t, &original, actual, false).unwrap();
        assert!(research["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["actual_recommendation"] == false));
        assert_eq!(research["original_blue_cartesian_outcomes"], 56);
        assert!(
            (research["original_ticket_retention_given_pool_and_original_blues"]
                .as_f64()
                .unwrap()
                - 8. / 56.)
                .abs()
                < 1e-12
        );
        assert_eq!(
            research["winning_result_rank_in_pool_all_blues"],
            json!([1, 448])
        );
        assert!(
            (research["joint_top_eight_uniform_numeric_tie_expected_hit"]
                .as_f64()
                .unwrap()
                - 8. / 448.)
                .abs()
                < 1e-12
        );
        let pending = compare(&t, &original, None, true).unwrap();
        assert_eq!(pending["rows"][0]["actual_recommendation"], true);
        assert!(pending["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["complete_6_plus_1_hit"].is_null()));
    }
}
pub fn replay(draws: &[Draw]) -> Result<Value, String> {
    let study = model::study(draws)?;
    let mut rows = Vec::new();
    for s in study
        .samples
        .iter()
        .filter(|s| s.row.issue > model::TRAIN_END)
    {
        let order = rank_desc(&model::red_marginals(
            &s.row.expert_p,
            &study.red.weights,
            s.previous,
        ));
        let pool: Vec<_> = order[..16].iter().map(|n| *n as u8 + 1).collect();
        let bp: [f64; 16] = std::array::from_fn(|b| {
            s.row
                .blue_p
                .iter()
                .zip(&study.blue.weights)
                .map(|(p, w)| p[b] * w)
                .sum::<f64>()
                + study.blue.weights[6] / 16.
        });
        let bo = rank_desc(&bp);
        let masks = crate::coverage::balanced_tickets(&order[..16], 8)?;
        let original: Vec<_> = masks
            .iter()
            .enumerate()
            .map(|(i, r)| Ticket {
                red: *r,
                blue: bo[i % 2] as u8 + 1,
            })
            .collect();
        let t = table(&pool, &s.row.expert_p, &s.row.blue_p, &study, s.previous)?;
        let result = compare(
            &t,
            &original,
            Some(Ticket {
                red: s.row.red_mask,
                blue: s.row.blue + 1,
            }),
            false,
        )?;
        rows.push(json!({"issue":fmt_issue(s.row.issue),"pool_all_six":s.row.red_mask&t.pool_mask==s.row.red_mask,"rows":result["rows"],"winning_result_rank":result["winning_result_rank_in_pool_all_blues"],"tie_expected_hit":result["joint_top_eight_uniform_numeric_tie_expected_hit"]}));
    }
    let summary = |lo: &str, hi: &str| {
        let rs: Vec<_> = rows
            .iter()
            .filter(|r| r["issue"].as_str().is_some_and(|i| i >= lo && i <= hi))
            .collect();
        let methods:Vec<_>=(0..4).map(|i|json!({"id":rs.first().map(|r|&r["rows"][i]["id"]),"representative_jackpots":rs.iter().filter(|r|r["rows"][i]["historical_exploratory_jackpot"]==true).count(),"mean_model_gain":rs.iter().map(|r|r["rows"][i]["model_gain_vs_original"].as_f64().unwrap()).sum::<f64>()/rs.len()as f64})).collect();
        json!({"n":rs.len(),"pool_all_six_periods":rs.iter().filter(|r|r["pool_all_six"]==true).count(),"methods":methods,"uniform_expected_jackpots":rs.len()as f64*UNIFORM_EIGHT})
    };
    Ok(
        json!({"training_last_issue":fmt_issue(model::TRAIN_END),"data_sha256":data_sha256(draws),"windows":{"validation":summary("23101","25110"),"confirmation_exploratory":summary("25111","26115"),"tracking":summary("26116","99999")},"periods":rows,"actual_recommendation":false,
        "note":"固定校准模型的边际16红和校准前2蓝作为两种组票的共同因果基线；不冒充历史融合ML、原v2号码或真正首次推荐。未利用当期开奖选号。零头奖在此预算下不能单独证明方法优劣。"}),
    )
}
