//! Blue-score v2 research, with separate immutable prospective score records.
use crate::{
    blue_model_v2 as model,
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    recommendations as rec,
    select::{rank_desc, score_key},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Mutex, OnceLock},
};
const KIND: &str = "blue_score_v2_prospective";
fn source() -> String {
    sha256_hex(
        concat!(
            include_str!("blue_model_v2.rs"),
            include_str!("model.rs"),
            include_str!("select.rs"),
            include_str!("protocol.rs")
        )
        .as_bytes(),
    )
}
fn predicted(s: &model::Study) -> Value {
    json!({"policy":model::policy(),"coefficients":s.fit.beta,"training_n":s.fit.n,"training_history_sha256":s.training_sha,
        "solver":{"iterations":s.fit.iterations,"normalized_gradient":s.fit.gradient,"objective_gap_upper_bound":s.fit.upper_gap,"penalized_log_excess_objective":s.fit.objective},
        "probabilities":s.next,"ranking":rank_desc(&s.next).map(|n|n+1),"original_probabilities":s.old_next,"original_ranking":rank_desc(&s.old_next).map(|n|n+1),
        "model_source_sha256":source(),"forecast_kind":"blue_score_only","tickets_generated":false})
}
fn mean_ci(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let se = if v.len() > 1 {
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ((v.len() - 1) * v.len()) as f64)
            .sqrt()
    } else {
        0.
    };
    json!({"mean":mean,"descriptive_95_low":mean-1.96*se,"descriptive_95_high":mean+1.96*se,"scope":"普通配对均值标准误，仅为描述区间；未校正多重尝试和时间相关，也不是新独立前瞻"})
}
fn tie_hit(p: &[f64; 16], winner: usize, k: usize) -> f64 {
    let order = rank_desc(p);
    let cutoff = score_key(p[order[k - 1]]);
    let key = score_key(p[winner]);
    if key > cutoff {
        return 1.;
    }
    if key < cutoff {
        return 0.;
    }
    let above = p.iter().filter(|p| score_key(**p) > cutoff).count();
    let tied = p.iter().filter(|p| score_key(**p) == cutoff).count();
    (k - above) as f64 / tied as f64
}
fn window(rows: &[&model::Record]) -> Value {
    let mut cells = Vec::new();
    for k in [1, 2, 4] {
        let mut hits = [0; 2];
        let (mut gains, mut losses) = (0, 0);
        let mut tie = [0.; 2];
        let mut deltas = Vec::new();
        for r in rows {
            let a = rank_desc(&r.old)[..k].contains(&r.winner);
            let b = rank_desc(&r.new)[..k].contains(&r.winner);
            hits[0] += usize::from(a);
            hits[1] += usize::from(b);
            gains += usize::from(b && !a);
            losses += usize::from(a && !b);
            tie[0] += tie_hit(&r.old, r.winner, k);
            tie[1] += tie_hit(&r.new, r.winner, k);
            deltas.push(f64::from(b) - f64::from(a));
        }
        cells.push(json!({"blue_count":k,"n":rows.len(),"original_hits":hits[0],"v2_hits":hits[1],"uniform_expected_hits":rows.len()as f64*k as f64/16.,"paired_gains":gains,"paired_losses":losses,"tie_adjusted_expected_hits":tie,"paired_hit_rate_difference":mean_ci(&deltas)}));
    }
    let old: Vec<_> = rows.iter().map(|r| (r.old[r.winner] * 16.).ln()).collect();
    let new: Vec<_> = rows.iter().map(|r| (r.new[r.winner] * 16.).ln()).collect();
    let diff: Vec<_> = old.iter().zip(&new).map(|(a, b)| b - a).collect();
    json!({"n":rows.len(),"cells":cells,"original_log_excess_vs_uniform":mean_ci(&old),"v2_log_excess_vs_uniform":mean_ci(&new),"paired_log_score_gain":mean_ci(&diff)})
}
fn analysis(draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let key = data_sha256(draws);
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "蓝球v2研究缓存异常")?;
    if let Some((k, v)) = cache.as_ref() {
        if *k == key {
            return Ok(v.clone());
        }
    }
    let s = model::study(draws)?;
    let subset = |lo, hi| {
        window(
            &s.records
                .iter()
                .filter(|r| r.issue >= lo && r.issue <= hi)
                .collect::<Vec<_>>(),
        )
    };
    let last = draws.last().ok_or("没有开奖数据")?;
    let latest=s.records.last().map(|r| {
        let contributions:Vec<_>=(0..5).map(|i|json!({"axis":model::AXES[i],"coefficient":s.fit.beta[i],"winning_feature":r.x[r.winner][i],"winning_logit_component":s.fit.beta[i]*r.x[r.winner][i]})).collect();
        json!({"issue":fmt_issue(r.issue),"winning_blue":r.winner+1,"original_rank":rank_desc(&r.old).iter().position(|n|*n==r.winner).unwrap()+1,"v2_rank":rank_desc(&r.new).iter().position(|n|*n==r.winner).unwrap()+1,"contributions":contributions,"actual_recommendation":false,"note":"同输入的事后模型研究，不补记历史成绩；该期结果没有参与拟合。"})
    });
    let out = json!({"version":2,"kind":"deduplicated_blue_log_score_model","target_issue":fmt_issue(last.issue+1),"cutoff_issue":fmt_issue(last.issue),"data_sha256":key,"prediction":predicted(&s),"latest_exploratory":latest,
        "windows":{"validation":subset(23101,25110),"confirmation_exploratory":subset(25111,26115),"tracking":subset(26116,u32::MAX),"all_held_out_exploratory":subset(23101,u32::MAX),"recent60":window(&s.records.iter().rev().take(60).rev().collect::<Vec<_>>())},
        "automatic_promotion":false,"reason":"同2蓝回放较原榜改善，但确认窗的完整蓝球对数评分点估计仍低于均匀；先独立封存前瞻分布，不替换现有购票票面。",
        "note":"固定23100训练边界、固定正则40；五个独立方向联合拟合，零系数为均匀，允许负系数，不给弱专家强制席位。历史已被查看，不能充当新的独立验证；蓝球命中不等于完整6＋1或收益。"});
    *cache = Some((key, out.clone()));
    Ok(out)
}
pub fn prediction(draws: &[Draw]) -> Result<Value, String> {
    Ok(analysis(draws)?["prediction"].clone())
}
fn valid(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|b| b.is_ascii_digit())
}
fn path(issue: &str) -> String {
    format!("blue-score-v2/{issue}.json")
}
fn review_path(issue: &str) -> String {
    format!("reviews/{issue}/blue-score-v2.json")
}
fn issues(root: &Path) -> Result<Vec<String>, String> {
    let mut set = BTreeSet::new();
    match std::fs::read_dir(root.join("blue-score-v2")) {
        Ok(entries) => {
            for entry in entries {
                let p = entry.map_err(|e| e.to_string())?.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Some(i) = p.file_stem().and_then(|x| x.to_str()).filter(|s| valid(s)) {
                        set.insert(i.to_owned());
                    }
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    Ok(set.into_iter().collect())
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid(issue) {
        return Err("蓝球v2期号须为5位数字".into());
    }
    let p = root.join(path(issue));
    if !p.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&p)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let cutoff = p["data_cutoff"]
        .as_str()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or("蓝球v2缺少输入截止")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= cutoff)
        .cloned()
        .collect();
    let last = prefix.last().ok_or("蓝球v2缺少输入历史")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("蓝球v2缺少保存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("蓝球v2缺少截止时间")?;
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if p["kind"] != KIND
        || p["schema_version"] != 1
        || p["actual_recommendation"] != false
        || p["known_outcome_at_generation"] != false
        || p["independent_timestamp_verified"] != false
        || issue != fmt_issue(cutoff + 1)
        || issue[..2] != fmt_issue(cutoff)[..2]
        || !rec::saved_after_draw(Some(at), Some(last))
        || at >= deadline
        || !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
        || p["prediction"] != prediction(&prefix)?
        || p["model_source_sha256"] != source()
    {
        return Err("蓝球v2的时间、分布、模型或首次输入不一致".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"archive_sha256":e["sha256"],"input_history_sha256":p["input_history_sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"model_source_sha256":p["model_source_sha256"],"actual_recommendation":false,"forecast_kind":"blue_score_only","raw_url":format!("api/blue/score-v2/archives/{}",p["issue"].as_str().unwrap())})
}
pub fn freeze(root: &Path, draws: &[Draw], issue: &str, deadline: u64) -> Result<Value, String> {
    freeze_clock(root, draws, issue, deadline, rec::now_unix)
}
fn freeze_clock(
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
    let last = draws.last().ok_or("没有开奖数据")?;
    let at = clock();
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || issue != fmt_issue(last.issue + 1)
        || issue[..2] != fmt_issue(last.issue)[..2]
        || !rec::saved_after_draw(Some(at), Some(last))
        || at >= deadline
    {
        return Err("蓝球v2只能在系统截止前封存当前紧邻下一期".into());
    }
    let prepared = crate::eval::prepare_forecast(draws)?;
    let forecast = prediction(draws)?;
    let completed = clock();
    if completed < at || completed >= deadline {
        return Err("蓝球v2计算跨过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["actual_recommendation"] = json!(false);
    p["prediction"] = forecast;
    p["model_source_sha256"] = json!(source());
    let (_, created) = rec::write_new_before(&root.join(path(issue)), p, Some(deadline), || {
        let t = clock();
        if t < completed {
            deadline
        } else {
            t
        }
    })?;
    let e = checked(root, draws, issue)?.ok_or("蓝球v2写入后未找到原档")?;
    let mut m = metadata(&e);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn evaluated(e: &Value, d: &Draw) -> Result<Value, String> {
    let p = &e["payload"]["prediction"];
    let new: [f64; 16] =
        serde_json::from_value(p["probabilities"].clone()).map_err(|_| "蓝球v2分布无效")?;
    let old: [f64; 16] = serde_json::from_value(p["original_probabilities"].clone())
        .map_err(|_| "原蓝球分布无效")?;
    let b = (d.blue - 1) as usize;
    Ok(
        json!({"winning_blue":d.blue,"v2_rank":rank_desc(&new).iter().position(|n|*n==b).unwrap()+1,"original_rank":rank_desc(&old).iter().position(|n|*n==b).unwrap()+1,"v2_top2_hit":rank_desc(&new)[..2].contains(&b),"original_top2_hit":rank_desc(&old)[..2].contains(&b),"v2_log_excess_vs_uniform":(new[b]*16.).ln(),"original_log_excess_vs_uniform":(old[b]*16.).ln(),"score_only":true}),
    )
}
fn review(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().ok_or("原档缺少期号")?;
    let p = root.join(review_path(issue));
    if !p.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&p)?;
    let d = draw.ok_or("蓝球v2复盘缺少开奖")?;
    let p = &r["payload"];
    if p["kind"] != "blue_score_v2_review"
        || p["issue"] != issue
        || p["archive_sha256"] != e["sha256"]
        || p["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || p["result"] != evaluated(e, d)?
    {
        return Err("蓝球v2复盘与首次原档或开奖不一致".into());
    }
    Ok(Some(r))
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("该期没有蓝球v2首次原档")?;
    std::fs::read(root.join(path(issue))).map_err(|e| e.to_string())
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut out = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    let all = match issues(root) {
        Ok(v) => v,
        Err(e) => {
            out.errors.push(e);
            return out;
        }
    };
    for i in all {
        let Some(d) = draws.iter().find(|d| fmt_issue(d.issue) == i) else {
            continue;
        };
        let result = (|| -> Result<usize, String> {
            let e = checked(root, draws, &i)?.ok_or("缺少蓝球v2原档")?;
            if review(root, &e, Some(d))?.is_some() {
                return Ok(0);
            }
            let p = json!({"kind":"blue_score_v2_review","schema_version":1,"issue":i,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(d)),"reviewed_at_unix":rec::now_unix(),"result":evaluated(&e,d)?});
            let (_, created) =
                rec::write_new_before(&root.join(review_path(&i)), p, None, rec::now_unix)?;
            review(root, &e, Some(d))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{i}/blue-score-v2：{e}")),
        }
    }
    out
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let mut v = analysis(draws)?;
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    let (mut n, mut pending, mut hits, mut original_hits) = (0, 0, 0, 0);
    for i in issues(root)? {
        let result = (|| -> Result<Value, String> {
            let e = checked(root, draws, &i)?.ok_or("蓝球v2原档缺失")?;
            let d = draws.iter().find(|d| fmt_issue(d.issue) == i);
            review(root, &e, d)?;
            let mut m = metadata(&e);
            m["pending"] = json!(d.is_none());
            m["result"] = d
                .map(|d| evaluated(&e, d))
                .transpose()?
                .unwrap_or(Value::Null);
            m["prediction"] = e["payload"]["prediction"].clone();
            Ok(m)
        })();
        match result {
            Ok(m) => {
                if m["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    hits += usize::from(m["result"]["v2_top2_hit"] == true);
                    original_hits += usize::from(m["result"]["original_top2_hit"] == true);
                }
                periods.push(m);
            }
            Err(e) => errors.push(format!("{i}：{e}")),
        }
    }
    v["current_observation"] = periods
        .iter()
        .find(|p| p["issue"] == v["target_issue"])
        .cloned()
        .unwrap_or(Value::Null);
    v["prospective"] = json!({"n":n,"pending":pending,"v2_top2_hits":hits,"original_top2_hits":original_hits,"periods":periods,"errors":errors,"note":"只计本模型首次封存后的真实开奖，缺档不算失败；分布观察不生成购票票面。"});
    let last = draws.last().ok_or("没有开奖数据")?;
    let g = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    v["publication"] =
        json!({"can_observe":g.publish,"target":g.target,"deadline":g.deadline_text});
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ssq-blue-v2-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> Vec<Draw> {
        let mut d = crate::draw::load_csv(Path::new("data/ssq.csv")).unwrap();
        let last = d.last_mut().unwrap();
        last.issue = 26129;
        last.date = 20261110;
        d
    }
    const NOW: u64 = 1794358800;
    #[test]
    fn prospective_is_immutable_pending_and_settles_only_saved_blue_distribution() {
        let root = Temp::new();
        let mut d = fixture();
        let g = crate::cycle::decide(NOW, 26129, 20261110, false);
        assert_eq!(report(&root.0, &d).unwrap()["prospective"]["n"], 0);
        assert!(!root.0.exists());
        freeze_clock(&root.0, &d, "26130", g.deadline_unix, || NOW).unwrap();
        let bytes = artifact(&root.0, &d, "26130").unwrap();
        assert_eq!(
            freeze_clock(&root.0, &d, "26130", g.deadline_unix, || NOW + 1).unwrap()["status"],
            "already_frozen"
        );
        let pending = report(&root.0, &d).unwrap();
        assert!(pending["current_observation"]["result"].is_null());
        assert_eq!(pending["prospective"]["pending"], 1);
        let e: Value = serde_json::from_slice(&bytes).unwrap();
        let b = e["payload"]["prediction"]["ranking"][0].as_u64().unwrap() as u8;
        d.push(Draw::from_parts(26130, 20261112, [1, 2, 3, 4, 5, 6], b).unwrap());
        assert_eq!(settle(&root.0, &d).settled, 1);
        let r = std::fs::read(root.0.join(review_path("26130"))).unwrap();
        assert_eq!(settle(&root.0, &d).settled, 0);
        assert_eq!(r, std::fs::read(root.0.join(review_path("26130"))).unwrap());
        assert_eq!(bytes, artifact(&root.0, &d, "26130").unwrap());
        assert_eq!(
            report(&root.0, &d).unwrap()["prospective"]["v2_top2_hits"],
            1
        );
    }
    #[test]
    fn deadline_clock_reversal_and_rehashed_probability_tampering_are_rejected() {
        let root = Temp::new();
        let d = fixture();
        let g = crate::cycle::decide(NOW, 26129, 20261110, false);
        assert!(freeze_clock(&root.0, &d, "26129", g.deadline_unix, || NOW).is_err());
        assert!(freeze_clock(&root.0, &d, "26130", g.deadline_unix, || g.deadline_unix).is_err());
        let tick = std::cell::Cell::new(0);
        assert!(freeze_clock(&root.0, &d, "26130", g.deadline_unix, || {
            let n = tick.get();
            tick.set(n + 1);
            if n == 0 {
                NOW
            } else {
                NOW - 1
            }
        })
        .is_err());
        assert!(!root.0.join(path("26130")).exists());
        freeze_clock(&root.0, &d, "26130", g.deadline_unix, || NOW).unwrap();
        let file = root.0.join(path("26130"));
        let mut e: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
        e["payload"]["prediction"]["probabilities"][0] = json!(0.9);
        e["sha256"] = json!(sha256_hex(&serde_json::to_vec(&e["payload"]).unwrap()));
        std::fs::write(file, serde_json::to_vec_pretty(&e).unwrap()).unwrap();
        assert!(artifact(&root.0, &d, "26130").is_err());
        assert!(artifact(&root.0, &d, "../26130").is_err());
        assert_eq!(report(&root.0, &d).unwrap()["prospective"]["n"], 0);
    }
}
