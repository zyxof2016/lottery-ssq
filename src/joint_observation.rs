//! Immutable prospective observations for the existing joint-red research model.
//! These records are research evidence, never automatic buy recommendations.
use crate::{
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    hash::sha256_hex,
    joint_red, recommendations as rec,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "joint_red_prospective_observation";
fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|b| b.is_ascii_digit())
}
fn archive_path(issue: &str) -> String {
    format!("joint-red/{issue}.json")
}
fn review_path(issue: &str) -> String {
    format!("reviews/{issue}/joint-red.json")
}
fn issues(root: &Path) -> Result<Vec<String>, String> {
    let mut out = BTreeSet::new();
    match fs::read_dir(root.join("joint-red")) {
        Ok(entries) => {
            for e in entries {
                let p = e.map_err(|e| e.to_string())?.path();
                if p.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(s) = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .filter(|s| valid_issue(s))
                    {
                        out.insert(s.to_string());
                    }
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    Ok(out.into_iter().collect())
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid_issue(issue) {
        return Err("研究观察期号需要5位数字".into());
    }
    let path = root.join(archive_path(issue));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let cutoff = p["data_cutoff"]
        .as_str()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or("研究观察缺少截止期")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= cutoff)
        .cloned()
        .collect();
    let last = prefix.last().ok_or("研究观察缺少输入历史")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("研究观察缺少保存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("研究观察缺少截止时间")?;
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if p["kind"] != KIND
        || p["schema_version"] != 1
        || p["known_outcome_at_generation"] != false
        || p["actual_recommendation"] != false
        || p["independent_timestamp_verified"] != false
        || issue != fmt_issue(cutoff + 1)
        || issue[..2] != fmt_issue(cutoff)[..2]
        || !rec::saved_after_draw(Some(at), Some(last))
        || at >= deadline
        || !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
    {
        return Err("研究观察类型、时间或期号边界无效".into());
    }
    if p["prediction"] != joint_red::prediction(&prefix)? {
        return Err("研究观察票面、权重或分布与开奖前输入不一致".into());
    }
    let source = p["model_source_sha256"]
        .as_str()
        .ok_or("研究观察缺少源码摘要")?;
    if source.len() != 64 || !source.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("研究源码摘要无效".into());
    }
    if let Some(parent) = p["linked_head_archive_sha256"].as_str() {
        let head = crate::jackpot::view(root, draws, issue)?.ok_or("缺少绑定的头奖原档")?;
        if head["archive_sha256"] != parent {
            return Err("研究观察绑定的头奖原档摘要已变化".into());
        }
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"archive_sha256":e["sha256"],"published_at_unix":p["published_at_unix"],"operational_cutoff_unix":p["operational_cutoff_unix"],"input_history_sha256":p["input_history_sha256"],"data_cutoff":p["data_cutoff"],"model_source_sha256":p["model_source_sha256"],"independent_timestamp_verified":false,"actual_recommendation":false,"raw_url":format!("api/jackpot/model-review/archives/{}",p["issue"].as_str().unwrap_or(""))})
}
pub fn freeze(root: &Path, draws: &[Draw], issue: &str, deadline: u64) -> Result<Value, String> {
    freeze_with_clock(root, draws, issue, deadline, rec::now_unix)
}
pub(crate) fn freeze_with_clock(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
    clock: impl Fn() -> u64,
) -> Result<Value, String> {
    if let Some(e) = checked(root, draws, issue)? {
        let mut out = metadata(&e);
        out["status"] = json!("already_frozen");
        return Ok(out);
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    let start = clock();
    let gate = crate::cycle::decide(start, last.issue, last.date, false);
    if !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || issue != fmt_issue(last.issue + 1)
        || issue[..2] != fmt_issue(last.issue)[..2]
        || !rec::saved_after_draw(Some(start), Some(last))
        || start >= deadline
    {
        return Err("研究观察只能在系统截止前封存当前紧邻下一期".into());
    }
    let prepared = crate::eval::prepare_forecast(draws)?;
    let prediction = joint_red::prediction(draws)?;
    let linked = crate::jackpot::view(root, draws, issue)?.map(|v| v["archive_sha256"].clone());
    let completed = clock();
    if completed < start || completed >= deadline {
        return Err("研究计算越过截止或时钟倒退，拒绝封存".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["actual_recommendation"] = json!(false);
    p["prediction"] = prediction;
    p["linked_head_archive_sha256"] = json!(linked);
    p["model_source_sha256"] = json!(sha256_hex(
        concat!(
            include_str!("joint_red.rs"),
            include_str!("joint_observation.rs"),
            include_str!("model.rs"),
            include_str!("select.rs"),
            include_str!("protocol.rs")
        )
        .as_bytes()
    ));
    let (_, created) =
        rec::write_new_before(&root.join(archive_path(issue)), p, Some(deadline), || {
            let at = clock();
            if at < completed {
                deadline
            } else {
                at
            }
        })?;
    let e = checked(root, draws, issue)?.ok_or("研究写入后未找到原档")?;
    let mut out = metadata(&e);
    out["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(out)
}
fn evaluated(e: &Value, draw: &Draw) -> Result<Value, String> {
    let prediction = &e["payload"]["prediction"];
    let saved = json!({"title":"联合评分研究观察·8注16元","mode":"research","structural":true,"tickets":prediction["tickets"]});
    let plan = rec::review_strategy(&saved, Some(draw))?;
    let scores = joint_red::score_prediction(prediction, draw)?;
    Ok(
        json!({"plan":plan,"scores":scores,"candidate_full_six_red":draw.red.iter().all(|n|prediction["candidate_reds"].as_array().is_some_and(|a|a.contains(&json!(n)))) ,"single_six_red":plan["tickets"].as_array().unwrap().iter().any(|t|t["red_hits"]==6),"jackpot":plan["tickets"].as_array().unwrap().iter().any(|t|t["red_hits"]==6&&t["blue_hit"]==true)}),
    )
}
fn persisted(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().ok_or("研究期号无效")?;
    let path = root.join(review_path(issue));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("研究已有复盘但缺少该期开奖")?;
    let p = &r["payload"];
    if p["kind"] != "joint_red_prospective_review"
        || p["issue"] != issue
        || p["archive_sha256"] != e["sha256"]
        || p["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || p["result"] != evaluated(e, d)?
    {
        return Err("研究复盘与首次原档或开奖摘要不一致".into());
    }
    Ok(Some(r))
}
fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let d = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let prior = persisted(root, &e, d)?;
    let mut v = metadata(&e);
    v["status"] = json!("frozen");
    v["pending"] = json!(d.is_none());
    v["candidate_reds"] = e["payload"]["prediction"]["candidate_reds"].clone();
    v["review_persisted"] = json!(prior.is_some());
    v["result"] = if let Some(r) = prior {
        r["payload"]["result"].clone()
    } else if let Some(d) = d {
        evaluated(&e, d)?
    } else {
        Value::Null
    };
    let saved = json!({"title":"联合评分研究观察·8注16元","mode":"research","structural":true,"tickets":e["payload"]["prediction"]["tickets"]});
    v["plan"] = rec::review_strategy(&saved, d)?;
    Ok(Some(v))
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("该期没有研究原档")?;
    fs::read(root.join(archive_path(issue))).map_err(|e| e.to_string())
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut out = rec::Settlement {
        settled: 0,
        errors: vec![],
    };
    let all = match issues(root) {
        Ok(v) => v,
        Err(e) => {
            out.errors.push(e);
            return out;
        }
    };
    for issue in all {
        let Some(d) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let result = (|| -> Result<usize, String> {
            let e = checked(root, draws, &issue)?.ok_or("研究原档缺失")?;
            if persisted(root, &e, Some(d))?.is_some() {
                return Ok(0);
            }
            let p = json!({"kind":"joint_red_prospective_review","issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(d)),"scored_at_unix":rec::now_unix(),"result":evaluated(&e,d)?});
            let (_, created) =
                rec::write_new_before(&root.join(review_path(&issue)), p, None, rec::now_unix)?;
            persisted(root, &e, Some(d))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/joint-red：{e}")),
        }
    }
    out
}
fn observations(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let (mut n, mut pending, mut heads, mut six, mut pools) = (0, 0, 0, 0, 0);
    let (mut red, mut joint) = (0., 0.);
    let (mut periods, mut errors) = (vec![], vec![]);
    for issue in issues(root)? {
        match view(root, draws, &issue) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    let r = &v["result"];
                    heads += usize::from(r["jackpot"] == true);
                    six += usize::from(r["single_six_red"] == true);
                    pools += usize::from(r["candidate_full_six_red"] == true);
                    red += r["scores"]["red_log_excess_vs_uniform"]
                        .as_f64()
                        .ok_or("缺少研究红球评分")?;
                    joint += r["scores"]["joint_log_excess_vs_uniform"]
                        .as_f64()
                        .ok_or("缺少研究联合评分")?;
                }
                periods.push(v);
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    Ok(
        json!({"n":n,"pending":pending,"jackpot_periods":heads,"single_six_red_periods":six,"candidate_full_six_periods":pools,"mean_red_log_excess_vs_uniform":(n>0).then(||red/n as f64),"mean_joint_log_excess_vs_uniform":(n>0).then(||joint/n as f64),"periods":periods,"errors":errors,"note":"仅统计本轮启用后在系统截止前首次封存的固定研究模型；历史预览和回放不补计实绩。待开奖不计失败，本地保存时间不等于独立时间证明。研究观察不会自动替换购票推荐。"}),
    )
}
fn latest_head_review(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    for d in draws.iter().rev().take(64) {
        if let Some(v) = crate::jackpot::view(root, draws, &fmt_issue(d.issue))? {
            let pool: Vec<u8> =
                serde_json::from_value(v["candidate_reds"].clone()).map_err(|_| "头奖候选无效")?;
            let ticketed: Vec<u8> =
                serde_json::from_value(v["plan"]["summary"]["unique_reds"].clone())
                    .map_err(|_| "头奖票面红球无效")?;
            let in_pool: Vec<_> = d.red.iter().filter(|n| pool.contains(n)).copied().collect();
            let missed: Vec<_> = d
                .red
                .iter()
                .filter(|n| !pool.contains(n))
                .copied()
                .collect();
            let dropped: Vec<_> = in_pool
                .iter()
                .filter(|n| !ticketed.contains(n))
                .copied()
                .collect();
            let blue_rank = v["plan"]["numbers"]
                .as_array()
                .and_then(|a| {
                    a.iter()
                        .find(|x| x["color"] == "blue" && x["number"] == d.blue)
                })
                .map(|x| x["fusion_rank"].clone())
                .unwrap_or(Value::Null);
            return Ok(
                json!({"issue":fmt_issue(d.issue),"draw":{"red":d.red,"blue":d.blue,"date":fmt_date(d.date)},"archive_sha256":v["archive_sha256"],"url":v["url"],"candidate_hit_reds":in_pool,"candidate_missed_reds":missed,"dropped_winning_reds":dropped,"ticket_hit_reds":v["plan"]["summary"]["covered_reds"],"selected_blues":v["plan"]["summary"]["unique_blues"],"winning_blue_rank":blue_rank,"best_single_red":v["diagnostic"]["best_single_red"],"single_exact_six_red_and_blue":v["diagnostic"]["single_exact_six_red_and_blue"],"note":"只读核对该期第一次真实保存的头奖票面；各层漏号分别展示，候选或并集命中不算头奖。"}),
            );
        }
    }
    Ok(Value::Null)
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let mut r = joint_red::report(root, draws)?;
    let actual = observations(root, draws)?;
    r["adoption"]["new_prospective_n"] = actual["n"].clone();
    r["prospective"] = actual;
    r["latest_saved_review"] = latest_head_review(root, draws)?;
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    r["publication"] =
        json!({"can_observe":gate.publish,"target":gate.target,"deadline":gate.deadline_text});
    r["current_observation"] = r["prospective"]["periods"]
        .as_array()
        .and_then(|a| a.iter().find(|v| v["issue"] == r["target_issue"]))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ssq-joint-observation-{}-{}",
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
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> Vec<Draw> {
        let mut d = crate::draw::synth_draws(220, 19);
        let last = d.last_mut().unwrap();
        last.issue = 26126;
        last.date = 20261103;
        d
    }
    const NOW: u64 = 1_793_754_000; // 2026-11-04 09:00 CST
    fn rehash(path: &Path, mutate: impl FnOnce(&mut Value)) {
        let mut e: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        mutate(&mut e["payload"]);
        e["sha256"] = json!(sha256_hex(&serde_json::to_vec(&e["payload"]).unwrap()));
        fs::write(path, serde_json::to_vec_pretty(&e).unwrap()).unwrap();
    }
    #[test]
    fn first_observation_is_immutable_pending_and_does_not_backfill_previews() {
        let root = Temp::new();
        let d = fixture();
        let gate = crate::cycle::decide(NOW, 26126, 20261103, false);
        assert!(gate.publish);
        let empty = report(&root.0, &d).unwrap();
        assert_eq!(empty["prospective"]["n"], 0);
        assert_eq!(empty["prospective"]["pending"], 0);
        assert!(!root.0.exists());
        assert!(freeze_with_clock(&root.0, &d, "26126", gate.deadline_unix, || NOW).is_err());
        assert!(artifact(&root.0, &d, "../26127").is_err());
        let v = freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW).unwrap();
        assert_eq!(v["status"], "frozen");
        assert_eq!(v["independent_timestamp_verified"], false);
        let before = artifact(&root.0, &d, "26127").unwrap();
        let again =
            freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW + 60).unwrap();
        assert_eq!(again["status"], "already_frozen");
        assert_eq!(before, artifact(&root.0, &d, "26127").unwrap());
        let observed = observations(&root.0, &d).unwrap();
        assert_eq!(observed["n"], 0);
        assert_eq!(observed["pending"], 1);
        assert!(observed["mean_joint_log_excess_vs_uniform"].is_null());
        assert!(observed["periods"][0]["result"].is_null());
        assert_eq!(observed["periods"][0]["plan"]["summary"]["cost"], 16);
        assert!(observed["periods"][0]["plan"]["summary"]["prize_tickets"].is_null());
        assert_eq!(issues(&root.0).unwrap(), vec!["26127"]);
        assert!(!root.0.join("daily").exists());
        assert!(!root.0.join("reviews").exists());
    }
    #[test]
    fn deadline_overrun_clock_reversal_and_rehashed_model_tampering_are_rejected() {
        let d = fixture();
        let gate = crate::cycle::decide(NOW, 26126, 20261103, false);
        for times in [
            [NOW, gate.deadline_unix, gate.deadline_unix],
            [NOW, NOW - 1, NOW],
            [NOW, NOW, gate.deadline_unix],
            [NOW, NOW, NOW - 1],
        ] {
            let root = Temp::new();
            let i = std::cell::Cell::new(0);
            assert!(
                freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || {
                    let n = i.get();
                    i.set(n + 1);
                    times[n.min(2)]
                })
                .is_err()
            );
            assert!(!root.0.join(archive_path("26127")).exists());
        }
        for field in ["ticket", "probability", "known", "deadline", "timestamp"] {
            let root = Temp::new();
            freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW).unwrap();
            let path = root.0.join(archive_path("26127"));
            rehash(&path, |p| match field {
                "ticket" => {
                    p["prediction"]["tickets"][0]["blue"] =
                        json!(p["prediction"]["tickets"][0]["blue"].as_u64().unwrap() % 16 + 1)
                }
                "probability" => p["prediction"]["red_weights"][0] = json!(0.9),
                "known" => p["known_outcome_at_generation"] = json!(true),
                "deadline" => p["operational_cutoff_unix"] = json!(gate.deadline_unix + 86400),
                _ => p["published_at_unix"] = json!(gate.deadline_unix),
            });
            assert!(view(&root.0, &d, "26127").is_err(), "{field}");
            let invalid = observations(&root.0, &d).unwrap();
            assert_eq!(invalid["n"], 0);
            assert_eq!(invalid["pending"], 0);
            assert_eq!(invalid["errors"].as_array().unwrap().len(), 1);
        }
    }
    #[test]
    fn actual_results_use_the_saved_ticket_and_distribution_and_review_is_never_replaced() {
        let root = Temp::new();
        let mut d = fixture();
        let gate = crate::cycle::decide(NOW, 26126, 20261103, false);
        freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW).unwrap();
        let before = artifact(&root.0, &d, "26127").unwrap();
        let e: Value = serde_json::from_slice(&before).unwrap();
        let t = &e["payload"]["prediction"]["tickets"][0];
        let red: [u8; 6] = serde_json::from_value(t["red"].clone()).unwrap();
        d.push(Draw::from_parts(26127, 20261105, red, t["blue"].as_u64().unwrap() as u8).unwrap());
        let v = observations(&root.0, &d).unwrap();
        assert_eq!(v["n"], 1);
        assert_eq!(v["pending"], 0);
        assert_eq!(v["jackpot_periods"], 1);
        assert_eq!(v["single_six_red_periods"], 1);
        assert!(v["periods"][0]["result"]["scores"]["joint_log_score"]
            .as_f64()
            .unwrap()
            .is_finite());
        assert_eq!(before, artifact(&root.0, &d, "26127").unwrap());
        assert_eq!(settle(&root.0, &d).settled, 1);
        let path = root.0.join(review_path("26127"));
        let review = fs::read(&path).unwrap();
        assert_eq!(settle(&root.0, &d).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), review);
        rehash(&path, |p| p["result"]["jackpot"] = json!(false));
        assert!(view(&root.0, &d, "26127").is_err());
        assert_eq!(settle(&root.0, &d).errors.len(), 1);
    }
}
