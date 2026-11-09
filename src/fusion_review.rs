//! Actual sealed-v2 loss attribution plus research-only full-six pool search.
//! Never changes a frozen model, reference weight, plan or archive.
use crate::{
    calibration_model_v1 as model,
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    pool_search::{Coverage, Search},
    recommendations as rec,
    select::{rank_desc, score_key},
};
use serde_json::{json, Value};
use std::{
    path::Path,
    sync::{Mutex, OnceLock},
};
pub const NODE_BUDGET: usize = 50_000;
const UNIVERSE: f64 = 1_107_568.;
fn mask(numbers: &Value) -> Result<u64, String> {
    let ns: Vec<u8> = serde_json::from_value(numbers.clone()).map_err(|_| "缺少有效号码数组")?;
    if ns.iter().any(|n| !(1..=33).contains(n))
        || ns.iter().collect::<std::collections::BTreeSet<_>>().len() != ns.len()
    {
        return Err("号码重复或越界".into());
    }
    Ok(ns.iter().fold(0, |m, n| m | (1 << (n - 1))))
}
fn nums(mask: u64) -> Vec<u8> {
    (1..=33).filter(|n| mask & (1 << (n - 1)) != 0).collect()
}
fn archive(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !root.join(format!("fusion-v2/{issue}.json")).exists() {
        return Ok(None);
    }
    serde_json::from_slice(&crate::fusion_v2::artifact(root, draws, issue)?)
        .map(Some)
        .map_err(|_| "融合v2原档无效".into())
}
fn coverage(experts: &[[f64; 33]; 7], weights: &[f64], previous: u64) -> Result<Coverage, String> {
    Coverage::new(
        33,
        6,
        16,
        previous,
        weights,
        &experts.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
    )
}
fn marginal(experts: &[[f64; 33]; 7], weights: &[f64], previous: u64) -> u64 {
    let p = model::red_marginals(experts, weights, previous);
    rank_desc(&p)[..16].iter().fold(0, |m, i| m | (1 << i))
}
fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.;
    }
    (0..k).fold(1., |v, i| v * (n - i) as f64 / (i + 1) as f64)
}
fn tie_outcome(pool: u64, actual: u64, groups: &Value) -> Value {
    let (mut expected, mut possible, mut guaranteed) = (1., true, true);
    for g in groups.as_array().unwrap() {
        let group = mask(&g["numbers"]).unwrap();
        let size = group.count_ones() as usize;
        let selected = (group & pool).count_ones() as usize;
        let needed = (group & actual).count_ones() as usize;
        if needed > selected {
            possible = false;
            expected = 0.;
        } else {
            expected *= choose(size - needed, selected - needed) / choose(size, selected);
        }
        if needed > 0 && selected < size {
            guaranteed = false;
        }
    }
    json!({"representative_full_six":pool&actual==actual,"possible_within_signature_ties":possible,"guaranteed_within_signature_ties":guaranteed,
        "expected_full_six_if_signature_members_are_chosen_uniformly":expected,"scope":"只对返回的同专家分布组计数模式内的成员同分进行组合计算，不当作已中次数或所有最优解的范围"})
}
fn search_result(
    m: &Coverage,
    s: &Search,
    marginal: u64,
    sealed: Option<u64>,
    actual: Option<u64>,
) -> Result<Value, String> {
    let uniform = choose(16, 6) / UNIVERSE;
    let marginal_mass = m.mass(marginal)?;
    let mut v = json!({"objective":"probability_all_six_reds_are_inside_16","candidate_count":16,"reds":nums(s.mask),"model_full_six_pool_mass":s.mass,"uniform_full_six_pool_mass":uniform,
        "marginal_pool":{"reds":nums(marginal),"model_full_six_pool_mass":marginal_mass},"gain_vs_marginal_pool":s.mass/marginal_mass-1.,
        "certificate":{"node_budget":NODE_BUDGET,"nodes":s.nodes,"signature_groups":s.groups,"signature_groups_detail":s.group_signatures,"numerically_optimal":s.complete,
            "upper_bound":s.upper,"absolute_gap":(s.upper-s.mass).max(0.),"relative_gap":(s.upper/s.mass-1.).max(0.),"numeric_tolerance_relative_to_uniform":1e-12,
            "method":"group_equal_expert_signatures_then_deterministic_branch_and_bound; componentwise_best_completion_upper_bound","number_ties_present":s.tied_membership,
            "scope":"最优性或差距仅针对冻结校准模型的16红入池目标，未证明真实预测优势；不是完整购票票面"},
        "tickets_generated":false,"actual_recommendation":false});
    if let Some(pool) = sealed {
        let p = m.mass(pool)?;
        v["sealed_v2_pool"] = json!({"reds":nums(pool),"model_full_six_pool_mass":p});
        v["gain_vs_sealed_v2_pool"] = json!(s.mass / p - 1.);
    }
    if let Some(a) = actual {
        v["historical_outcome_exploratory"] = tie_outcome(s.mask, a, &s.group_signatures);
        v["marginal_historical_outcome_exploratory"] =
            tie_outcome(marginal, a, &s.group_signatures);
    }
    Ok(v)
}
fn attribution(root: &Path, draws: &[Draw], e: &Value, draw: &Draw) -> Result<Value, String> {
    let issue = fmt_issue(draw.issue);
    let b = &e["payload"]["bundle"];
    let global = b["singles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["red_count"] == 16)
        .ok_or("v2缺少16红候选")?;
    let global_mask = mask(&global["reds"])?;
    let parent = crate::reference::original(root, draws, &issue)?.ok_or("缺少v2参考父档")?;
    let support: Vec<f64> = serde_json::from_value(
        parent["payload"]["reference_pools"]["fusion"]["reference_support"].clone(),
    )
    .map_err(|_| "参考支持无效")?;
    let score: Vec<f64> = serde_json::from_value(b["scores"].clone()).map_err(|_| "v2评分无效")?;
    let core: [f64; 33] = std::array::from_fn(|i| (score[i] - 0.1 * support[i]) / 0.9);
    let core_order = rank_desc(&core);
    let ranking: Vec<u8> =
        serde_json::from_value(b["ranking"].clone()).map_err(|_| "v2排序无效")?;
    let numbers:Vec<_>=draw.red.iter().map(|n|{let i=(n-1)as usize;json!({"number":n,"without_reference_rank":core_order.iter().position(|j|*j==i).unwrap()+1,"v2_rank":ranking.iter().position(|j|j==n).unwrap()+1,"reference_supported":support[i]>0.,"reference_score_component":0.1*support[i],"base_without_reference":core[i],"v2_score":score[i],"in_sixteen":global_mask&(1<<i)!=0})}).collect();
    let cutoff = e["payload"]["data_cutoff"]
        .as_str()
        .ok_or("v2缺少数据截止")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    let p = crate::eval::prepare_forecast(&prefix)?;
    let mut plans = Vec::new();
    for group in ["singles", "dantuo"] {
        for plan in b[group].as_array().unwrap() {
            let red_pool = mask(&plan["reds"])?;
            let banks = mask(&plan["red_dan"])?;
            let tickets = plan["tickets"].as_array().unwrap();
            let exact = tickets
                .iter()
                .any(|t| mask(&t["red"]).ok() == Some(draw.red_mask));
            let blue_in = tickets.iter().any(|t| t["blue"] == draw.blue);
            let head = tickets
                .iter()
                .any(|t| mask(&t["red"]).ok() == Some(draw.red_mask) && t["blue"] == draw.blue);
            let review = rec::review_strategy(
                &json!({"title":plan["title"],"mode":plan["mode"],"structural":true,"tickets":plan["tickets"]}),
                Some(draw),
            )?;
            plans.push(json!({"id":plan["id"],"title":plan["title"],"mode":plan["mode"],"bets":plan["bet_count"],"cost":plan["cost"],"red_pool_hits":(red_pool&draw.red_mask).count_ones(),"red_pool_missed":nums(draw.red_mask&!red_pool),
            "winning_reds_dropped_from_global16":nums(draw.red_mask&global_mask&!red_pool),"wrong_bankers":nums(banks&!draw.red_mask),"blue_selected":blue_in,"blue_choices":plan["blues"],"single_full_six":exact,"jackpot":head,"pairing_miss":exact&&blue_in&&!head,
            "primary_loss":if draw.red_mask&global_mask!=draw.red_mask{"global_red_pool_miss"}else if draw.red_mask&red_pool!=draw.red_mask{"plan_red_pool_miss"}else if banks&draw.red_mask!=banks{"banker_miss"}else if !exact{"complete_red_combination_miss"}else if !blue_in{"blue_selection_miss"}else if !head{"blue_pairing_miss"}else{"jackpot"},"summary":review["summary"]}));
        }
    }
    let review_url = if root
        .join(format!("reviews/{issue}/fusion-v2.json"))
        .exists()
    {
        crate::fusion_v2::review_artifact(root, draws, &issue)?;
        Some(format!("api/reference-pools/fusion-v2/reviews/{issue}"))
    } else {
        None
    };
    Ok(
        json!({"issue":issue,"draw":{"red":draw.red,"blue":draw.blue},"archive_sha256":e["sha256"],"published_at_unix":e["payload"]["published_at_unix"],"input_history_sha256":e["payload"]["input_history_sha256"],
        "global_sixteen_missed":nums(draw.red_mask&!global_mask),"winning_blue_rank":p.forecast.blue_order.iter().position(|b|*b+1==draw.blue as usize).map(|i|i+1),"winning_numbers":numbers,"plans":plans,
        "raw_url":format!("api/reference-pools/fusion-v2/archives/{issue}"),"review_url":review_url,"actual_recommendation":true,"note":"仅核对首次v2原档；去掉参考支持的名次是保存分数的公式分解，不是当时另存的推荐。"}),
    )
}
fn evidence(study: &model::Study, actual_n: usize) -> Value {
    let mean = |lo, hi| {
        let vals: Vec<_> = study
            .samples
            .iter()
            .filter(|s| s.row.issue >= lo && s.row.issue <= hi)
            .map(|s| model::sample_score(s, study).iter().sum::<f64>())
            .collect();
        json!({"n":vals.len(),"joint_log_excess_vs_uniform":(!vals.is_empty()).then(||vals.iter().sum::<f64>()/vals.len()as f64)})
    };
    json!({"model":"frozen_overlap_stratified_calibration_v1","training_last_issue":fmt_issue(model::TRAIN_END),"validation":mean(23101,25110),"confirmation_exploratory":mean(25111,26115),"tracking":mean(26116,u32::MAX),"v2_actual_drawn_periods":actual_n,
        "promoted":false,"reason":"新搜索只优化给定完整分布内的入池目标；6＋1确认窗的点估计仍低于均匀，v2真实样本仍少。研究结果不替换已封存号码，也不据单期结果重调参考权重。"})
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let target = fmt_issue(last.issue + 1);
    let current = archive(root, draws, &target)?;
    let latest = archive(root, draws, &fmt_issue(last.issue))?;
    let mut archive_fingerprints = Vec::new();
    if let Ok(files) = std::fs::read_dir(root.join("fusion-v2")) {
        for f in files {
            let p = f.map_err(|e| e.to_string())?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit()))
            {
                archive_fingerprints.push((
                    p.file_name().unwrap().to_string_lossy().to_string(),
                    sha256_hex(&std::fs::read(&p).map_err(|e| e.to_string())?),
                ));
                let review = root.join(format!(
                    "reviews/{}/fusion-v2.json",
                    p.file_stem().unwrap().to_string_lossy()
                ));
                if review.exists() {
                    archive_fingerprints.push((
                        format!("review:{}", p.file_stem().unwrap().to_string_lossy()),
                        sha256_hex(&std::fs::read(review).map_err(|e| e.to_string())?),
                    ));
                }
            }
        }
    }
    archive_fingerprints.sort();
    let key = format!(
        "{}:{}:{}:{archive_fingerprints:?}",
        data_sha256(draws),
        current
            .as_ref()
            .map(|v| &v["sha256"])
            .unwrap_or(&Value::Null),
        latest
            .as_ref()
            .map(|v| &v["sha256"])
            .unwrap_or(&Value::Null)
    );
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "完整入池优化缓存异常")?;
    if let Some((k, v)) = cache.as_ref() {
        if k == &key {
            return Ok(v.clone());
        }
    }
    let study = model::study(draws)?;
    let current_model = coverage(&study.next_red, &study.red.weights, study.previous)?;
    let current_marg = marginal(&study.next_red, &study.red.weights, study.previous);
    let pool = |e: &Value| -> Result<u64, String> {
        let p = e["payload"]["bundle"]["singles"]
            .as_array()
            .ok_or("缺少v2票面")?
            .iter()
            .find(|p| p["red_count"] == 16)
            .ok_or("缺少v2 16红")?;
        mask(&p["reds"])
    };
    let current_pool = current.as_ref().map(pool).transpose()?;
    let seeds: Vec<_> = std::iter::once(current_marg).chain(current_pool).collect();
    let found = current_model.optimize_with_seeds(NODE_BUDGET, &seeds)?;
    let mut search = search_result(&current_model, &found, current_marg, current_pool, None)?;
    search["issue"] = json!(target);
    search["sealed_v2_archive_sha256"] = current
        .as_ref()
        .map(|e| e["sha256"].clone())
        .unwrap_or(Value::Null);
    let mut historical = Value::Null;
    if let Some(sample) = study.samples.iter().find(|s| s.row.issue == last.issue) {
        let m = coverage(&sample.row.expert_p, &study.red.weights, sample.previous)?;
        let marg = marginal(&sample.row.expert_p, &study.red.weights, sample.previous);
        let old = latest.as_ref().map(pool).transpose()?;
        let seeds: Vec<_> = std::iter::once(marg).chain(old).collect();
        let found = m.optimize_with_seeds(NODE_BUDGET, &seeds)?;
        historical = search_result(&m, &found, marg, old, Some(last.red_mask))?;
        historical["issue"] = json!(fmt_issue(last.issue));
    }
    let mut actual_n = 0;
    for d in draws {
        let issue = fmt_issue(d.issue);
        if root.join(format!("fusion-v2/{issue}.json")).exists() {
            archive(root, draws, &issue)?.ok_or("v2实际样本原档缺失")?;
            actual_n += 1;
        }
    }
    let v = json!({"version":1,"kind":"complete_six_pool_coverage_review","data_sha256":data_sha256(draws),"target_issue":target,"cutoff_issue":fmt_issue(last.issue),"latest":latest.as_ref().map(|e|attribution(root,draws,e,last)).transpose()?,"current_search":search,"latest_search_exploratory":historical,"adoption":evidence(&study,actual_n),
        "solver_source_sha256":sha256_hex(include_str!("pool_search.rs").as_bytes()),"tickets_generated":false,"writes_archives":false,"note":"完整六红入池、实际单注6＋1和模型内概率质量分别核对；没有历史参考ML推荐的期数不补造v2融合回放。"});
    *cache = Some((key, v.clone()));
    Ok(v)
}
pub fn replay(draws: &[Draw]) -> Result<Value, String> {
    let study = model::study(draws)?;
    let mut rows = Vec::new();
    for sample in study
        .samples
        .iter()
        .filter(|s| s.row.issue > model::TRAIN_END)
    {
        let m = coverage(&sample.row.expert_p, &study.red.weights, sample.previous)?;
        let marg = marginal(&sample.row.expert_p, &study.red.weights, sample.previous);
        let found = m.optimize_with_seeds(NODE_BUDGET, &[marg])?;
        let scores =
            model::red_marginals(&sample.row.expert_p, &study.red.weights, sample.previous);
        let order = rank_desc(&scores);
        rows.push(json!({"issue":fmt_issue(sample.row.issue),"relative_model_gain":found.mass/m.mass(marg)?-1.,"numerically_optimal":found.complete,"nodes":found.nodes,"relative_gap":found.upper/found.mass-1.,"marginal":tie_outcome(marg,sample.row.red_mask,&found.group_signatures),"coverage_search":tie_outcome(found.mask,sample.row.red_mask,&found.group_signatures),"marginal_cutoff_tied_at_legacy_precision":score_key(scores[order[15]]/6.)==score_key(scores[order[16]]/6.)}));
    }
    let summary = |lo: &str, hi: &str| {
        let rs: Vec<_> = rows
            .iter()
            .filter(|r| r["issue"].as_str().is_some_and(|i| i >= lo && i <= hi))
            .collect();
        let method = |key: &str| json!({"representative_full_six_periods":rs.iter().filter(|r|r[key]["representative_full_six"]==true).count(),"possible_within_signature_ties_periods":rs.iter().filter(|r|r[key]["possible_within_signature_ties"]==true).count(),"guaranteed_within_signature_ties_periods":rs.iter().filter(|r|r[key]["guaranteed_within_signature_ties"]==true).count(),"tie_uniform_expected_full_six_periods":rs.iter().map(|r|r[key]["expected_full_six_if_signature_members_are_chosen_uniformly"].as_f64().unwrap()).sum::<f64>()});
        json!({"n":rs.len(),"mean_model_gain":(!rs.is_empty()).then(||rs.iter().map(|r|r["relative_model_gain"].as_f64().unwrap()).sum::<f64>()/rs.len()as f64),"numerically_optimal_periods":rs.iter().filter(|r|r["numerically_optimal"]==true).count(),"max_relative_gap":rs.iter().map(|r|r["relative_gap"].as_f64().unwrap()).fold(0.,f64::max),"uniform_expected_full_six_periods":rs.len()as f64*choose(16,6)/UNIVERSE,"marginal":method("marginal"),"coverage_search":method("coverage_search")})
    };
    Ok(
        json!({"data_sha256":data_sha256(draws),"node_budget_per_period":NODE_BUDGET,"training_last_issue":fmt_issue(model::TRAIN_END),"windows":{"validation":summary("23101","25110"),"confirmation_exploratory":summary("25111","26115"),"tracking":summary("26116","99999")},"periods":rows,"actual_recommendation":false,"note":"两种16红仅用目标期以前的专家特征和23100以前拟合的冻结权重；都是探索回放，不是历史参考融合ML或真实首次留档。成员同分的数字代表解次数不作为预测优势。"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tie_membership_is_combinatorial_and_does_not_count_an_arbitrary_number_order_as_evidence() {
        let g = json!([{"numbers":[1,2,3,4]},{"numbers":[5,6]}]);
        let result = tie_outcome(0b110011, 0b110001, &g);
        assert_eq!(result["representative_full_six"], true);
        assert_eq!(result["guaranteed_within_signature_ties"], false);
        assert!(
            (result["expected_full_six_if_signature_members_are_chosen_uniformly"]
                .as_f64()
                .unwrap()
                - 0.5)
                .abs()
                < 1e-12
        );
        let impossible = tie_outcome(0b110011, 0b1111, &g);
        assert_eq!(impossible["possible_within_signature_ties"], false);
    }
}
