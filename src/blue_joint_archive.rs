//! Immutable prospective observations for the existing blue-joint research model.
//! These records are research evidence, never automatic buy recommendations.
use crate::{
    blue_deep,
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "blue_joint_prospective_observation";
fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|b| b.is_ascii_digit())
}
fn archive_path(issue: &str) -> String {
    format!("blue-joint/{issue}.json")
}
fn review_path(issue: &str) -> String {
    format!("reviews/{issue}/blue-joint.json")
}
fn issues(root: &Path) -> Result<Vec<String>, String> {
    let mut out = BTreeSet::new();
    match fs::read_dir(root.join("blue-joint")) {
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
    if p["prediction"] != blue_deep::prediction(&prefix)? {
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
    json!({"issue":p["issue"],"archive_sha256":e["sha256"],"published_at_unix":p["published_at_unix"],"operational_cutoff_unix":p["operational_cutoff_unix"],"input_history_sha256":p["input_history_sha256"],"data_cutoff":p["data_cutoff"],"model_source_sha256":p["model_source_sha256"],"independent_timestamp_verified":false,"actual_recommendation":false,"raw_url":format!("api/blue/deep/archives/{}",p["issue"].as_str().unwrap_or(""))})
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
    let prediction = blue_deep::prediction(draws)?;
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
            include_str!("blue_deep.rs"),
            include_str!("blue_joint_archive.rs"),
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
    let p = &e["payload"]["prediction"];
    let plan = rec::review_strategy(
        &json!({"title":"红蓝联合选票观察·8注16元","mode":"research_joint","structural":true,"tickets":p["tickets"]}),
        Some(draw),
    )?;
    let baseline = rec::review_strategy(
        &json!({"title":"原头奖4组×2蓝对照","mode":"research_joint","structural":true,"tickets":p["baseline_tickets"]}),
        Some(draw),
    )?;
    let six = |plan: &Value| {
        plan["tickets"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["red_hits"] == 6)
    };
    let head = |plan: &Value| {
        plan["tickets"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["red_hits"] == 6 && t["blue_hit"] == true)
    };
    Ok(
        json!({"blue_covered":plan["summary"]["blue_covered"],"baseline_blue_covered":baseline["summary"]["blue_covered"],"single_six_red":six(&plan),"jackpot":head(&plan),"baseline_jackpot":head(&baseline),"plan":plan,"baseline_plan":baseline}),
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
    if p["kind"] != "blue_joint_prospective_review"
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
    let saved = json!({"title":"红蓝联合选票观察·8注16元","mode":"research_joint","structural":true,"tickets":e["payload"]["prediction"]["tickets"]});
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
            let p = json!({"kind":"blue_joint_prospective_review","issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(d)),"scored_at_unix":rec::now_unix(),"result":evaluated(&e,d)?});
            let (_, created) =
                rec::write_new_before(&root.join(review_path(&issue)), p, None, rec::now_unix)?;
            persisted(root, &e, Some(d))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/blue-joint：{e}")),
        }
    }
    out
}

fn observations(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let (mut n, mut pending, mut heads, mut blue, mut base_blue) = (0, 0, 0, 0, 0);
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
                    blue += usize::from(r["blue_covered"] == true);
                    base_blue += usize::from(r["baseline_blue_covered"] == true);
                }
                periods.push(v);
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    Ok(
        json!({"n":n,"pending":pending,"jackpot_periods":heads,"blue_covered_periods":blue,"baseline_blue_covered_periods":base_blue,"periods":periods,"errors":errors,"note":"只统计截止前首次封存的联合选票观察；旧期结构回放不补计实绩，待开奖不计失败。本地封存不是独立时间证明，观察不会自动替换原头奖推荐。"}),
    )
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let mut r = blue_deep::report(root, draws)?;
    r["prospective"] = observations(root, draws)?;
    r["current_observation"] = r["prospective"]["periods"]
        .as_array()
        .and_then(|a| a.iter().find(|v| v["issue"] == r["target_issue"]))
        .cloned()
        .unwrap_or(Value::Null);
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    r["publication"] =
        json!({"can_observe":gate.publish,"target":gate.target,"deadline":gate.deadline_text});
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ssq-blue-joint-{}-{}",
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
        let mut d = crate::draw::synth_draws(220, 29);
        let last = d.last_mut().unwrap();
        last.issue = 26126;
        last.date = 20261103;
        d
    }
    const NOW: u64 = 1793754000;
    #[test]
    fn first_joint_observation_is_pending_preserves_original_and_scores_saved_tickets() {
        let root = Temp::new();
        let mut d = fixture();
        let gate = crate::cycle::decide(NOW, 26126, 20261103, false);
        let empty = report(&root.0, &d).unwrap();
        assert_eq!(empty["prospective"]["n"], 0);
        assert!(!root.0.exists());
        assert!(freeze_with_clock(&root.0, &d, "26126", gate.deadline_unix, || NOW).is_err());
        freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW).unwrap();
        let bytes = artifact(&root.0, &d, "26127").unwrap();
        assert_eq!(
            freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW + 1).unwrap()
                ["status"],
            "already_frozen"
        );
        assert_eq!(artifact(&root.0, &d, "26127").unwrap(), bytes);
        let v = view(&root.0, &d, "26127").unwrap().unwrap();
        assert_eq!(v["plan"]["summary"]["cost"], 16);
        assert!(v["result"].is_null());
        assert!(v["plan"]["summary"]["prize_tickets"].is_null());
        let e: Value = serde_json::from_slice(&bytes).unwrap();
        let t = &e["payload"]["prediction"]["tickets"][0];
        d.push(
            Draw::from_parts(
                26127,
                20261105,
                serde_json::from_value(t["red"].clone()).unwrap(),
                t["blue"].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        let v = view(&root.0, &d, "26127").unwrap().unwrap();
        assert_eq!(v["result"]["jackpot"], true);
        assert_eq!(settle(&root.0, &d).settled, 1);
        let review = fs::read(root.0.join(review_path("26127"))).unwrap();
        assert_eq!(settle(&root.0, &d).settled, 0);
        assert_eq!(review, fs::read(root.0.join(review_path("26127"))).unwrap());
        assert_eq!(bytes, artifact(&root.0, &d, "26127").unwrap());
    }
    #[test]
    fn rehashed_distribution_tampering_and_late_publication_are_rejected() {
        let root = Temp::new();
        let d = fixture();
        let gate = crate::cycle::decide(NOW, 26126, 20261103, false);
        assert!(
            freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || gate
                .deadline_unix)
            .is_err()
        );
        freeze_with_clock(&root.0, &d, "26127", gate.deadline_unix, || NOW).unwrap();
        let path = root.0.join(archive_path("26127"));
        let mut e: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        e["payload"]["prediction"]["blue_scores"][0] = json!(0.9);
        e["sha256"] = json!(sha256_hex(&serde_json::to_vec(&e["payload"]).unwrap()));
        fs::write(&path, serde_json::to_vec_pretty(&e).unwrap()).unwrap();
        assert!(view(&root.0, &d, "26127").is_err());
        assert!(artifact(&root.0, &d, "../26127").is_err());
        assert_eq!(observations(&root.0, &d).unwrap()["n"], 0);
    }
}
