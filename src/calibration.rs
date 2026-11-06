//! Proper-score calibration, latest sealed-ticket attribution and probability-
//! only prospective evidence. No new ticket pool or purchase recommendation.
use crate::{
    calibration_model_v1 as model,
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    recommendations as rec,
    select::rank_desc,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    sync::{Mutex, OnceLock},
};
const KIND: &str = "complete_probability_calibration_v1";
fn source_sha() -> String {
    sha256_hex(
        concat!(
            include_str!("calibration_model_v1.rs"),
            include_str!("model.rs"),
            include_str!("protocol.rs"),
            include_str!("select.rs")
        )
        .as_bytes(),
    )
}

fn summarize(rows: &[(&model::Sample, [f64; 2])]) -> Value {
    let cell = |new: bool| {
        let n = rows.len();
        let scores: Vec<_> = rows
            .iter()
            .map(|(s, p)| if new { *p } else { s.original })
            .collect();
        let mean = |i: usize| {
            (!scores.is_empty()).then(|| scores.iter().map(|p| p[i]).sum::<f64>() / n as f64)
        };
        json!({"n":n,"red_log_excess":mean(0),"blue_log_excess":mean(1),"joint_log_excess":mean(0).zip(mean(1)).map(|(a,b)|a+b)})
    };
    let diff: Vec<_> = rows
        .iter()
        .map(|(s, p)| p[0] + p[1] - s.original[0] - s.original[1])
        .collect();
    // Non-overlapping complete 20-draw batch means. Descriptive only.
    let blocks: Vec<_> = diff
        .chunks_exact(20)
        .map(|b| b.iter().sum::<f64>() / b.len() as f64)
        .collect();
    let uncertainty = if blocks.len() > 1 {
        let mean = blocks.iter().sum::<f64>() / blocks.len() as f64;
        Some(
            (blocks.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (blocks.len() * (blocks.len() - 1)) as f64)
                .sqrt(),
        )
    } else {
        None
    };
    json!({"original":cell(false),"calibrated":cell(true),"uniform":{"n":rows.len(),"joint_log_excess":0.},
        "paired_gain_mean":(!diff.is_empty()).then(||diff.iter().sum::<f64>()/diff.len()as f64),
        "paired_gain_block_standard_error":uncertainty,"block_size":20,"uncertainty_periods":blocks.len()*20,
        "first_issue":rows.first().map(|(s,_)|fmt_issue(s.row.issue)),"last_issue":rows.last().map(|(s,_)|fmt_issue(s.row.issue))})
}
fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let study = model::study(draws)?;
    let scores: Vec<_> = study
        .samples
        .iter()
        .map(|s| (s, model::sample_score(s, &study)))
        .collect();
    let confirm = scores
        .iter()
        .position(|(s, _)| s.row.issue >= crate::protocol::Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(scores.len());
    let end = (confirm + crate::protocol::Protocol::CONFIRMATION_LEN).min(scores.len());
    let train: Vec<_> = scores
        .iter()
        .filter(|(s, _)| s.row.issue <= model::TRAIN_END)
        .copied()
        .collect();
    let val: Vec<_> = scores
        .iter()
        .filter(|(s, _)| {
            s.row.issue > model::TRAIN_END
                && s.row.issue < crate::protocol::Protocol::CONFIRM_START_ISSUE
        })
        .copied()
        .collect();
    let mut windows = json!({"training":summarize(&train),"validation":summarize(&val),"confirmation_exploratory":summarize(&scores[confirm..end]),"tracking":summarize(&scores[end..])});
    let previous = crate::joint_score::report(draws)?;
    for window in [
        "training",
        "validation",
        "confirmation_exploratory",
        "tracking",
    ] {
        let p = &previous["windows"][window]["interaction"];
        windows[window]["interaction_v2"] = json!({"n":p["n"],"red_log_excess":p["mean_red_log_excess"],"blue_log_excess":p["mean_blue_log_excess"],"joint_log_excess":p["mean_joint_log_excess"]});
    }
    let mut ranks = Vec::new();
    for (sample, _) in scores
        .iter()
        .filter(|(s, _)| s.row.issue > model::TRAIN_END)
    {
        let marg = model::red_marginals(&sample.row.expert_p, &study.red.weights, sample.previous);
        let order = rank_desc(&marg);
        if crate::select::score_key(marg[order[15]] / 6.)
            == crate::select::score_key(marg[order[16]] / 6.)
        {
            continue;
        }
        let mut pool = order[..16].to_vec();
        pool.sort_unstable();
        let actual = sample.row.red_mask;
        if pool.iter().fold(0u64, |m, i| m | (1 << i)) & actual != actual {
            continue;
        }
        let models: Vec<_> = sample
            .row
            .expert_p
            .iter()
            .map(|p| model::SetDist::stratified(*p, sample.previous))
            .collect();
        let score = |mask: u64| {
            models
                .iter()
                .zip(&study.red.weights)
                .map(|(m, w)| m.probability(mask) * w)
                .sum::<f64>()
                + study.red.weights[7] / 1_107_568.
        };
        let key = |mask| (score(mask).ln() * 1e10).round() as i64;
        let target = key(actual);
        let (mut above, mut tied) = (0, 0);
        for a in 0..11 {
            for b in a + 1..12 {
                for c in b + 1..13 {
                    for d in c + 1..14 {
                        for e in d + 1..15 {
                            for f in e + 1..16 {
                                let k = key([a, b, c, d, e, f]
                                    .into_iter()
                                    .fold(0, |m, j| m | (1 << pool[j])));
                                above += usize::from(k > target);
                                tied += usize::from(k == target);
                            }
                        }
                    }
                }
            }
        }
        ranks.push(json!({"issue":fmt_issue(sample.row.issue),"true_six_rank_lower":above+1,"true_six_rank_upper":above+tied,"six_red_sets":8008}));
    }
    let latest = study.samples.last().ok_or("缺少校准复盘样本")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| d.issue < latest.row.issue)
        .cloned()
        .collect();
    let prepared = prepare_forecast(&prefix)?;
    let old_order = prepared.forecast.red_order;
    let marginal = model::red_marginals(&latest.row.expert_p, &study.red.weights, latest.previous);
    let new_keys = marginal.map(|p| crate::select::score_key(p / 6.));
    let winning:Vec<_>=(0..33).filter(|i|latest.row.red_mask&(1<<i)!=0).map(|i|{
        let contrib:Vec<_>=(0..7).map(|e|json!({"expert":crate::protocol::Protocol::EXPERTS[e],"weight":prepared.weights[e],"score":latest.row.expert_p[e][i],"contribution":prepared.weights[e]*latest.row.expert_p[e][i]})).collect();
        json!({"number":i+1,"old_rank":old_order.iter().position(|j|*j==i).unwrap()+1,
            "calibrated_rank_lower":new_keys.iter().filter(|k|**k>new_keys[i]).count()+1,"calibrated_rank_upper":new_keys.iter().filter(|k|**k>=new_keys[i]).count(),"old_score":prepared.forecast.fusion[i],"old_contributions":contrib})
    }).collect();
    let next_marginal = model::red_marginals(&study.next_red, &study.red.weights, study.previous);
    let previous_probability = study
        .next_red
        .iter()
        .zip(&study.red.weights)
        .map(|(p, w)| {
            model::SetDist::stratified(*p, study.previous).probability(study.previous) * w
        })
        .sum::<f64>()
        + study.red.weights[7] / 1_107_568.;
    let overlap = (0..33)
        .filter(|i| study.previous & (1 << i) != 0)
        .map(|i| next_marginal[i])
        .sum::<f64>();
    Ok(
        json!({"version":1,"kind":KIND,"data_sha256":data_sha256(draws),"cutoff_issue":draws.last().map(|d|fmt_issue(d.issue)),"target_issue":draws.last().map(|d|fmt_issue(d.issue+1)),
        "policy":model::policy(),"model_source_sha256":source_sha(),"prediction":model::prediction(&study),"windows":windows,
        "full_pool_cases":ranks,"concentration":{"exact_previous_six_ratio_to_uniform":previous_probability*1_107_568.,"expected_previous_overlap":overlap,"uniform_previous_overlap":36./33.,"note":"固定公平重合数量分布，弱重复专家不能再制造整组六红重复尖峰。"},"latest":{"issue":fmt_issue(latest.row.issue),"winning_reds":winning,"score":model::sample_score(latest,&study)},
        "adoption":{"promoted":false,"ticket_publication_enabled":false,"reason":"新校准需同时对照均匀与现有v2；历史评分不能证明头奖优势，只做概率验证，不替换购票推荐。"},
        "note":"截至23100期拟合完整六红与蓝球混合似然，之后权重冻结。历史窗口仅作探索；新前瞻只保存完整概率分布并评分，不生成随机票或新的号码观察池。"}),
    )
}
fn analysis(draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut c = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "校准缓存锁异常")?;
    let sha = data_sha256(draws);
    if let Some((key, value)) = c.as_ref() {
        if *key == sha {
            return Ok(value.clone());
        }
    }
    let v = analyze(draws)?;
    *c = Some((sha, v.clone()));
    Ok(v)
}
fn valid_issue(i: &str) -> bool {
    i.len() == 5 && i.bytes().all(|b| b.is_ascii_digit())
}
/// Archive verification needs the frozen fit and causal next-step features,
/// not every historical window and rank replay. Cache only the fixed fit.
fn compact_prediction(draws: &[Draw]) -> Result<Value, String> {
    static FIT: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let train: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= model::TRAIN_END)
        .cloned()
        .collect();
    let key = data_sha256(&train);
    let mut fit = FIT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "冻结权重缓存锁异常")?;
    if fit.as_ref().map(|(k, _)| k) != Some(&key) {
        *fit = Some((key, model::prediction(&model::study(&train)?)));
    }
    let mut p = fit.as_ref().unwrap().1.clone();
    drop(fit);
    let prepared = prepare_forecast(draws)?;
    let w: Vec<f64> =
        serde_json::from_value(p["blue_weights"].clone()).map_err(|_| "冻结蓝球权重无效")?;
    p["red_experts"] = json!(prepared
        .expert_red
        .iter()
        .map(|r| r.to_vec())
        .collect::<Vec<_>>());
    p["blue_probabilities"] = json!((0..16)
        .map(|i| (0..6)
            .map(|e| w[e] * prepared.expert_blue[e][i])
            .sum::<f64>()
            + w[6] / 16.)
        .collect::<Vec<_>>());
    p["baseline_red_scores"] = json!(prepared.forecast.fusion.to_vec());
    p["baseline_blue_probabilities"] = json!(prepared.forecast.blue);
    p["previous_red_mask"] = json!(draws.last().ok_or("没有开奖数据")?.red_mask);
    Ok(p)
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid_issue(issue) {
        return Err("概率观察期号无效".into());
    }
    let path = root.join(format!("calibration/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let cutoff = p["data_cutoff"].as_str().ok_or("概率观察缺少输入截止")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    let at = p["published_at_unix"].as_u64().ok_or("缺少封存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("缺少封存截止")?;
    if p["kind"] != KIND
        || p["model_source_sha256"] != source_sha()
        || p["known_outcome_at_generation"] != false
        || at >= deadline
        || issue != fmt_issue(prefix.last().ok_or("缺少概率观察历史")?.issue + 1)
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
    {
        return Err("概率观察来源、时间或期号无效".into());
    }
    if p["prediction"] != compact_prediction(&prefix)? {
        return Err("概率原档与冻结模型或输入历史不一致".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"archive_sha256":e["sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"actual_recommendation":false,"forecast_kind":"probability_score_only","raw_url":format!("api/jackpot/calibration/archives/{}",p["issue"].as_str().unwrap())})
}
pub fn freeze(root: &Path, draws: &[Draw], issue: &str, deadline: u64) -> Result<Value, String> {
    freeze_with_clock(root, draws, issue, deadline, rec::now_unix)
}
/// Preserve sealed v1 tickets. All-tied marginal rankings carry no learned
/// number preference at the legacy scorer's own precision.
pub fn legacy_ticket_identifiable(draws: &[Draw]) -> Result<bool, String> {
    let p = crate::joint_red::prediction(draws)?;
    let rows: Vec<Vec<f64>> =
        serde_json::from_value(p["red_experts"].clone()).map_err(|_| "旧研究专家无效")?;
    let rows: Vec<[f64; 33]> = rows
        .into_iter()
        .map(|r| r.try_into().map_err(|_| "旧研究专家长度无效"))
        .collect::<Result<_, _>>()?;
    let experts: [[f64; 33]; 7] = rows.try_into().map_err(|_| "旧研究专家数量无效")?;
    let weights: Vec<f64> =
        serde_json::from_value(p["red_weights"].clone()).map_err(|_| "旧研究权重无效")?;
    let marginal = model::unconditional_marginals(&experts, &weights);
    let keys = marginal.map(|p| crate::select::score_key(p / 6.));
    Ok(keys.iter().any(|k| *k != keys[0]))
}
pub fn freeze_legacy_if_identifiable(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
) -> Result<Value, String> {
    if !valid_issue(issue) {
        return Err("研究观察期号无效".into());
    }
    if !root.join(format!("joint-red/{issue}.json")).exists() && !legacy_ticket_identifiable(draws)?
    {
        return Ok(
            json!({"status":"skipped_unidentifiable","reason":"旧研究六红评分在排序精度下全体同分；不按号码大小生成新票面，继续完整概率评分。"}),
        );
    }
    crate::joint_observation::freeze(root, draws, issue, deadline)
}
fn freeze_with_clock(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
    clock: impl Fn() -> u64,
) -> Result<Value, String> {
    if let Some(e) = checked(root, draws, issue)? {
        let mut m = metadata(&e);
        m["status"] = json!("already_frozen");
        return Ok(m);
    }
    let started = clock();
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = crate::cycle::decide(started, last.issue, last.date, false);
    if !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || issue[..2] != fmt_issue(last.issue)[..2]
    {
        return Err("概率观察期号、数据或截止已变化".into());
    }
    let prediction = compact_prediction(draws)?;
    let prepared = prepare_forecast(draws)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("概率评分计算越过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["prediction"] = prediction;
    p["actual_recommendation"] = json!(false);
    p["forecast_kind"] = json!("probability_score_only");
    p["model_source_sha256"] = json!(source_sha());
    let (_, created) = rec::write_new_before(
        &root.join(format!("calibration/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("概率观察写入后未找到")?;
    let mut m = metadata(&e);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn evaluated(e: &Value, d: &Draw) -> Result<Value, String> {
    model::score_prediction(&e["payload"]["prediction"], d)
}
fn saved_review(root: &Path, e: &Value, d: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().unwrap();
    let path = root.join(format!("reviews/{issue}/calibration.json"));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = d.ok_or("已有概率复盘但缺少开奖")?;
    if r["payload"]["kind"] != "complete_probability_calibration_review_v1"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["result"] != evaluated(e, d)?
    {
        return Err("概率复盘与原档或开奖不一致".into());
    }
    Ok(Some(r))
}
fn issues(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(root.join("calibration")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                if let Some(i) = p
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .filter(|i| valid_issue(i))
                {
                    out.push(i.to_string());
                }
            }
        }
    }
    out.sort();
    out
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut out = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    for issue in issues(root) {
        let Some(d) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let result = (|| -> Result<usize, String> {
            let e = checked(root, draws, &issue)?.ok_or("概率原档缺失")?;
            if saved_review(root, &e, Some(d))?.is_some() {
                return Ok(0);
            }
            let p = json!({"kind":"complete_probability_calibration_review_v1","schema_version":1,"issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(d)),"reviewed_at_unix":rec::now_unix(),"result":evaluated(&e,d)?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/calibration.json")),
                p,
                None,
                rec::now_unix,
            )?;
            saved_review(root, &e, Some(d))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/calibration：{e}")),
        }
    }
    out
}
fn observations(root: &Path, draws: &[Draw]) -> Value {
    let (mut periods, mut errors) = (Vec::new(), Vec::new());
    let (mut n, mut pending, mut sum) = (0, 0, 0.);
    for issue in issues(root) {
        let result = (|| -> Result<Value, String> {
            let e = checked(root, draws, &issue)?.ok_or("概率原档缺失")?;
            let d = draws.iter().find(|d| fmt_issue(d.issue) == issue);
            let saved = saved_review(root, &e, d)?;
            let mut v = metadata(&e);
            v["pending"] = json!(d.is_none());
            v["result"] = match d {
                Some(d) => {
                    let r = if let Some(r) = saved {
                        r["payload"]["result"].clone()
                    } else {
                        evaluated(&e, d)?
                    };
                    r["joint_log_excess"].as_f64().ok_or("概率分数无效")?;
                    r
                }
                None => Value::Null,
            };
            Ok(v)
        })();
        match result {
            Ok(v) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    sum += v["result"]["joint_log_excess"].as_f64().unwrap();
                }
                periods.push(v);
            }
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"n":n,"pending":pending,"mean_joint_log_excess":(n>0).then(||sum/n as f64),"periods":periods,"errors":errors,"note":"只统计截止前首次封存的概率分布；不出票，不统计头奖命中，历史回放不补作前瞻。"})
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let mut v = analysis(draws)?;
    let old = crate::joint_observation::report(root, draws)?;
    v["latest_original"] = old["latest_saved_review"].clone();
    let last = draws.last().ok_or("没有开奖数据")?;
    v["latest_reference"] = if let Some(e) =
        crate::reference::original(root, draws, &fmt_issue(last.issue))?
    {
        let s = &e["payload"]["reference_pools"];
        let tickets = &s["reference"]["tickets"];
        let reviewed = rec::review_strategy(
            &json!({"title":"参考首次5注","mode":"structural","structural":true,"tickets":tickets}),
            Some(last),
        )?;
        json!({"issue":fmt_issue(last.issue),"archive_sha256":e["sha256"],"summary":reviewed["summary"],"missing_reds":last.red.iter().filter(|n|!s["reference"]["reds"].as_array().unwrap().iter().any(|v|v.as_u64()==Some(**n as u64))).collect::<Vec<_>>()})
    } else {
        Value::Null
    };
    v["legacy_ticket_identifiable"] = json!(legacy_ticket_identifiable(draws)?);
    v["prospective"] = observations(root, draws);
    v["current_observation"] = v["prospective"]["periods"]
        .as_array()
        .and_then(|p| p.iter().find(|p| p["issue"] == v["target_issue"]))
        .cloned()
        .unwrap_or(Value::Null);
    let g = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    v["publication"] = json!({"can_freeze":g.publish,"target":g.target,"deadline":g.deadline_text});
    Ok(v)
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("概率观察原档不存在")?;
    fs::read(root.join(format!("calibration/{issue}.json"))).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: u64 = 1793926800; // 2026-11-06 09:00 Shanghai
    fn fixture() -> Vec<Draw> {
        let mut d = crate::draw::synth_draws(220, 83);
        let x = d.last_mut().unwrap();
        x.issue = 26127;
        x.date = 20261105;
        d
    }
    #[test]
    fn score_archive_is_immutable_pending_and_does_not_generate_tickets() {
        let root = std::env::temp_dir().join(format!("ssq-calib-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut d = fixture();
        let gate = crate::cycle::decide(NOW, 26127, 20261105, false);
        assert_eq!(
            compact_prediction(&d).unwrap(),
            analysis(&d).unwrap()["prediction"]
        );
        assert!(analysis(&d).unwrap()["prediction"].get("tickets").is_none());
        assert!(!root.exists());
        assert!(freeze_with_clock(&root, &d, "26127", gate.deadline_unix, || NOW).is_err());
        freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW).unwrap();
        let bytes = artifact(&root, &d, "26128").unwrap();
        assert_eq!(observations(&root, &d)["pending"], 1);
        assert!(observations(&root, &d)["mean_joint_log_excess"].is_null());
        assert_eq!(
            freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW + 1).unwrap()
                ["status"],
            "already_frozen"
        );
        assert_eq!(artifact(&root, &d, "26128").unwrap(), bytes);
        let mut next = d.last().unwrap().clone();
        next.issue = 26128;
        next.date = 20261107;
        d.push(next);
        assert_eq!(settle(&root, &d).settled, 1);
        assert_eq!(settle(&root, &d).settled, 0);
        assert_eq!(observations(&root, &d)["n"], 1);
        assert!(observations(&root, &d)["errors"]
            .as_array()
            .unwrap()
            .is_empty());
        assert_eq!(artifact(&root, &d, "26128").unwrap(), bytes);
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn deadline_race_and_clock_reversal_cannot_leave_a_forecast() {
        let root = std::env::temp_dir().join(format!("ssq-calib-clock-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let d = fixture();
        let gate = crate::cycle::decide(NOW, 26127, 20261105, false);
        let calls = std::cell::Cell::new(0);
        assert!(
            freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || {
                let i = calls.get();
                calls.set(i + 1);
                if i < 2 {
                    NOW
                } else {
                    gate.deadline_unix
                }
            })
            .is_err()
        );
        assert!(!root.join("calibration/26128.json").exists());
        calls.set(0);
        assert!(
            freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || {
                let i = calls.get();
                calls.set(i + 1);
                if i == 0 {
                    NOW
                } else {
                    NOW - 1
                }
            })
            .is_err()
        );
        assert!(!root.join("calibration/26128.json").exists());
        assert!(artifact(&root, &d, "../26128").is_err());
        let _ = fs::remove_dir_all(root);
    }
}
