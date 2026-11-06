//! Read-only, same-pool comparison of ranking-prefix dantuo and the exact
//! complete-outcome optimizer. No promotion, tickets, archives or timer writes.
use crate::{
    calibration_model_v1 as model,
    dantuo_search::{Choice, Scores},
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    select::rank_desc,
};
use serde_json::{json, Value};
use std::{
    path::Path,
    sync::{Mutex, OnceLock},
};
const SETS: f64 = 1_107_568.;
const SPECS: [(usize, usize); 2] = [(2, 2), (3, 1)];

fn score_table(
    pool: &[u8],
    experts: &[[f64; 33]; 7],
    weights: &[f64],
    previous: u64,
) -> Result<Scores, String> {
    let active: Vec<_> = experts
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.)
        .map(|(p, w)| (model::SetDist::stratified(*p, previous), *w))
        .collect();
    Scores::new(pool, |mask| {
        active
            .iter()
            .map(|(m, w)| m.probability(mask) * w * SETS)
            .sum::<f64>()
            + weights[7]
    })
}
fn blue_prob(experts: &[[f64; 16]; 6], weights: &[f64]) -> [f64; 16] {
    std::array::from_fn(|b| {
        experts
            .iter()
            .zip(weights)
            .map(|(p, w)| p[b] * w)
            .sum::<f64>()
            + weights[6] / 16.
    })
}
fn shape(s: &Scores, c: &Choice, blue_mass: f64) -> Value {
    let union = s.union_mass(c);
    json!({"model_joint_mass":c.mass/SETS*blue_mass,"ratio_to_uniform_ten_tickets":c.mass/SETS*blue_mass/(10./(SETS*16.)),
        "model_red_union_mass":union/SETS,"model_legal_six_red_mass":c.mass/SETS,
        "banker_retention_given_union":c.mass/union,
        "red_union_count":(c.dan|c.tuo).count_ones(),"legal_complete_red_sets":if c.dan.count_ones()==2{5}else{10}})
}
fn comparisons(s: &Scores, blue: &[f64; 16], selected: &[u8]) -> Result<Value, String> {
    let mut rows = Vec::new();
    for (dan, blue_n) in SPECS {
        let blue_mass = selected[..blue_n]
            .iter()
            .map(|b| blue[(b - 1) as usize])
            .sum::<f64>();
        let original = s.prefix(dan)?;
        let (best, certificate) = s.optimize(dan)?;
        let mut singles: Vec<_> = crate::dantuo_search::subsets(s.pool.len(), 6)
            .into_iter()
            .flat_map(|m| {
                selected[..blue_n]
                    .iter()
                    .map(move |b| s.values[m as usize] / SETS * blue[(b - 1) as usize])
            })
            .collect();
        singles.sort_by(|a, b| b.total_cmp(a));
        let upper: f64 = singles[..10].iter().sum();
        let best_mass = best.mass / SETS * blue_mass;
        rows.push(json!({"id":format!("{dan}+5+{blue_n}"),"cost":20,"bet_count":10,
            "original_prefix":shape(s,&original,blue_mass),"exact_same_pool":shape(s,&best,blue_mass),
            "model_mass_relative_gain":best.mass/original.mass-1.,"same_blue_choices":true,
            "bankers_changed":original.dan!=best.dan,"union_changed":(original.dan|original.tuo)!=(best.dan|best.tuo),
            "top_ten_singles_same_pool_and_blues_mass":upper,"dantuo_mass_fraction_of_unrestricted_singles":best_mass/upper,
            "uniform":{"red_union_complete_sets":if dan==2{7}else{28},"legal_red_sets":if dan==2{5}else{10},"banker_retention_given_union":if dan==2{5./7.}else{10./28.},"joint_mass":10./(SETS*16.)},
            "certificate":certificate}));
    }
    Ok(json!(rows))
}
fn evidence(study: &model::Study) -> Value {
    let mean = |lo: u32, hi: u32| {
        let values: Vec<_> = study
            .samples
            .iter()
            .filter(|s| s.row.issue >= lo && s.row.issue <= hi)
            .map(|s| model::sample_score(s, study).iter().sum::<f64>())
            .collect();
        json!({"n":values.len(),"joint_log_excess_vs_uniform":(!values.is_empty()).then(||values.iter().sum::<f64>()/values.len()as f64)})
    };
    json!({"training_last_issue":fmt_issue(model::TRAIN_END),"red_uniform_weight":study.red.weights[7],"blue_uniform_weight":study.blue.weights[6],
        "validation":mean(model::TRAIN_END+1,25110),"confirmation_exploratory":mean(25111,26115),"tracking":mean(26116,u32::MAX),
        "promoted":false,"ticket_publication_enabled":false,"reason":"完整概率模型尚未证明优于均匀；搜索提高的是给定模型评分，不能当成真实头奖概率提升。保留首次票面，不另建购票观察池。"})
}
pub fn report(root: &Path, draws: &[Draw], requested: Option<&str>) -> Result<Value, String> {
    let issue = requested.map(str::to_owned).unwrap_or_else(|| {
        draws
            .last()
            .map(|d| fmt_issue(d.issue + 1))
            .unwrap_or_default()
    });
    if issue.len() != 5 || !issue.bytes().all(|b| b.is_ascii_digit()) {
        return Err("胆拖优化核验期号须为5位数字".into());
    }
    let Some(parent) = crate::reference::original(root, draws, &issue)? else {
        return Ok(
            json!({"issue":issue,"status":"waiting","reason":"缺少本期首次参考融合原档，不补造研究输入。"}),
        );
    };
    let cutoff = parent["payload"]["data_cutoff"]
        .as_str()
        .ok_or("参考原档缺少截止期")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    if data_sha256(&prefix) != parent["payload"]["input_history_sha256"] {
        return Err("胆拖搜索输入摘要不一致".into());
    }
    let sealed: Option<Value> = if root.join(format!("fusion-dantuo/{issue}.json")).exists() {
        Some(
            serde_json::from_slice(&crate::fusion_dantuo::artifact(root, draws, &issue)?)
                .map_err(|_| "融合胆拖原档JSON无效")?,
        )
    } else {
        None
    };
    let key = format!(
        "{}:{}:{}",
        data_sha256(&prefix),
        parent["sha256"],
        sealed
            .as_ref()
            .map(|v| &v["sha256"])
            .unwrap_or(&Value::Null)
    );
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "胆拖搜索缓存锁异常")?;
    if let Some((saved, v)) = cache.as_ref() {
        if saved == &key {
            return Ok(v.clone());
        }
    }
    let snapshot = &parent["payload"]["reference_pools"];
    let ranking: Vec<u8> =
        serde_json::from_value(snapshot["fusion"]["ranking"].clone()).map_err(|_| "融合榜无效")?;
    let blues: Vec<u8> =
        serde_json::from_value(snapshot["fusion"]["blues"].clone()).map_err(|_| "融合蓝球无效")?;
    if ranking.len() != 33
        || blues.len() != 2
        || blues.iter().any(|b| !(1..=16).contains(b))
        || blues[0] == blues[1]
    {
        return Err("融合榜或蓝球数量无效".into());
    }
    let study = model::study(&prefix)?;
    let table = score_table(
        &ranking[..16],
        &study.next_red,
        &study.red.weights,
        study.previous,
    )?;
    let blue = blue_prob(&study.next_blue, &study.blue.weights);
    let out = json!({"version":1,"status":"research_only","issue":issue,"data_cutoff":cutoff,"data_sha256":data_sha256(&prefix),"reference_archive_sha256":parent["sha256"],
        "score_model":"frozen_complete_probability_calibration_v1","solver_source_sha256":sha256_hex(include_str!("dantuo_search.rs").as_bytes()),
        "red_candidate_count":16,"same_fusion_parent":true,"blue_selection_changed":false,"uniform_outcomes":SETS as u64*16,
        "prefix_plan_sealed":sealed.is_some(),"prefix_archive_sha256":sealed.as_ref().map(|v|v["sha256"].clone()),
        "rows":comparisons(&table,&blue,&blues)?,"evidence":evidence(&study),"tickets_generated":false,"writes_archives":false,
        "note":"在同一融合16红和原蓝球选择内，对全部合法胆拖结构比较完整展开票面的概率质量。未限定为排序前7或8红；不随机抽号、不按开奖号挑胆。最优性只针对给定模型与候选，未通过预测验证，不输出新购票票面或覆盖原档。"});
    *cache = Some((key, out.clone()));
    Ok(out)
}

/// Exploratory causal replay: fixed trained probability model; each period's
/// 16-red pool comes from that model's marginals, not historical reference runs.
pub fn replay(draws: &[Draw]) -> Result<Value, String> {
    let study = model::study(draws)?;
    let mut rows = Vec::new();
    for sample in study
        .samples
        .iter()
        .filter(|s| s.row.issue > model::TRAIN_END)
    {
        let marg = model::red_marginals(&sample.row.expert_p, &study.red.weights, sample.previous);
        let order = rank_desc(&marg);
        let pool: Vec<u8> = order[..16].iter().map(|i| (i + 1) as u8).collect();
        let table = score_table(
            &pool,
            &sample.row.expert_p,
            &study.red.weights,
            sample.previous,
        )?;
        let blue = blue_prob(&sample.row.blue_p, &study.blue.weights);
        let selected = rank_desc(&blue);
        let pool_mask = pool.iter().fold(0u64, |m, n| m | (1 << (n - 1)));
        let actual: u16 = pool.iter().enumerate().fold(0, |m, (i, n)| {
            m | if sample.row.red_mask & (1 << (n - 1)) != 0 {
                1 << i
            } else {
                0
            }
        });
        let mut cells = Vec::new();
        for (dan, bn) in SPECS {
            let original = table.prefix(dan)?;
            let (best, _) = table.optimize(dan)?;
            let full = pool_mask & sample.row.red_mask == sample.row.red_mask;
            let hit =
                |c: &Choice| full && actual & c.dan == c.dan && actual & !(c.dan | c.tuo) == 0;
            let blue_in = selected[..bn].contains(&(sample.row.blue as usize));
            cells.push(json!({"id":format!("{dan}+5+{bn}"),"relative_model_gain":best.mass/original.mass-1.,"original_six":hit(&original),"exact_six":hit(&best),"original_jackpot":hit(&original)&&blue_in,"exact_jackpot":hit(&best)&&blue_in}));
        }
        rows.push(json!({"issue":fmt_issue(sample.row.issue),"pool_full_six":pool_mask&sample.row.red_mask==sample.row.red_mask,
            "pool_boundary_tied_at_original_precision":crate::select::score_key(marg[order[15]]/6.)==crate::select::score_key(marg[order[16]]/6.),"cells":cells}));
    }
    let summary = |lo: &str, hi: &str| {
        let rs: Vec<_> = rows
            .iter()
            .filter(|r| r["issue"].as_str().is_some_and(|i| i >= lo && i <= hi))
            .collect();
        json!({"n":rs.len(),"full_six_in_16":rs.iter().filter(|r|r["pool_full_six"]==true).count(),"pool_boundary_tied_periods":rs.iter().filter(|r|r["pool_boundary_tied_at_original_precision"]==true).count(),
            "cells":(0..2).map(|i|json!({"id":format!("{}+5+{}",SPECS[i].0,SPECS[i].1),"mean_model_gain":(!rs.is_empty()).then(||rs.iter().map(|r|r["cells"][i]["relative_model_gain"].as_f64().unwrap()).sum::<f64>()/rs.len()as f64),
            "original_six_periods":rs.iter().filter(|r|r["cells"][i]["original_six"]==true).count(),"exact_six_periods":rs.iter().filter(|r|r["cells"][i]["exact_six"]==true).count(),"original_jackpot_periods":rs.iter().filter(|r|r["cells"][i]["original_jackpot"]==true).count(),"exact_jackpot_periods":rs.iter().filter(|r|r["cells"][i]["exact_jackpot"]==true).count()})).collect::<Vec<_>>()})
    };
    Ok(
        json!({"evidence":evidence(&study),"data_sha256":data_sha256(draws),"pool_policy":"each_period_calibrated_marginal_top16; shared_by_both_selectors; not_reconstructed_reference_fusion",
        "windows":{"validation":summary("23101","25110"),"confirmation_exploratory":summary("25111","26115"),"tracking":summary("26116","99999")},"periods":rows,
        "actual_recommendation":false,"note":"历史探索回放，不是首次留档实绩；不按已查看窗口选择参数、出票或扩大预算。蓝球选择在两个组票器之间完全相同。"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_archive_is_read_only_and_invalid_issue_is_rejected() {
        let root =
            std::env::temp_dir().join(format!("ssq-dantuo-audit-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let d = crate::draw::synth_draws(220, 9);
        assert_eq!(report(&root, &d, None).unwrap()["status"], "waiting");
        assert!(!root.exists());
        assert!(report(&root, &d, Some("../bad")).is_err());
    }
    #[test]
    fn normalized_model_and_joint_mass_are_complete_outcomes() {
        let experts = [[1. / 33.; 33]; 7];
        let mut weights = vec![0.; 8];
        weights[7] = 1.;
        let table = score_table(&(1..=16).collect::<Vec<_>>(), &experts, &weights, 63).unwrap();
        let rows = comparisons(&table, &[1. / 16.; 16], &[1, 2]).unwrap();
        for r in rows.as_array().unwrap() {
            assert_eq!(r["model_mass_relative_gain"], 0.);
            assert!(
                (r["exact_same_pool"]["model_joint_mass"].as_f64().unwrap() - 10. / (SETS * 16.))
                    .abs()
                    < 1e-20
            );
            assert!(
                (r["dantuo_mass_fraction_of_unrestricted_singles"]
                    .as_f64()
                    .unwrap()
                    - 1.)
                    .abs()
                    < 1e-14
            );
        }
    }
}
