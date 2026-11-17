//! Immutable, explicitly experimental fusion version. Existing v2 is never
//! rewritten. The existing cycle continues this trial after its first opt-in.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    fusion_model_v3 as model, recommendations as rec,
};
use serde_json::{json, Value};
use std::{fs, path::Path};
const KIND: &str = "fusion_complete_ticket_trial_v3";
fn valid(i: &str) -> bool {
    i.len() == 5 && i.bytes().all(|b| b.is_ascii_digit())
}
fn source() -> String {
    crate::hash::sha256_hex(
        concat!(
            include_str!("fusion_model_v3.rs"),
            include_str!("red_model_v2.rs"),
            include_str!("blue_model_v2.rs"),
            include_str!("model.rs"),
            include_str!("select.rs"),
            include_str!("protocol.rs"),
            include_str!("stats.rs"),
            include_str!("plan.rs")
        )
        .as_bytes(),
    )
}
fn parents(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<(Value, Value)>, String> {
    let Some(parent) = crate::reference::original(root, draws, issue)? else {
        return Ok(None);
    };
    if !root.join(format!("fusion-v2/{issue}.json")).is_file() {
        return Ok(None);
    }
    let legacy: Value = serde_json::from_slice(&crate::fusion_v2::artifact(root, draws, issue)?)
        .map_err(|_| "v2父档无效")?;
    Ok(Some((parent, legacy)))
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid(issue) {
        return Err("联合优化v3期号须为5位数字".into());
    }
    let path = root.join(format!("fusion-v3/{issue}.json"));
    if !path.is_file() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let (parent, legacy) = parents(root, draws, issue)?.ok_or("联合优化v3缺少首次参考或v2父档")?;
    let cutoff = p["data_cutoff"].as_str().ok_or("联合优化v3缺少截止期")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    let last = prefix.last().ok_or("联合优化v3输入为空")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("联合优化v3缺少保存时间")?;
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if p["kind"] != KIND
        || p["experimental"] != true
        || p["actual_recommendation"] != true
        || p["known_outcome_at_generation"] != false
        || p["model_source_sha256"] != source()
        || p["reference_archive_sha256"] != parent["sha256"]
        || p["v2_archive_sha256"] != legacy["sha256"]
        || p["input_history_sha256"] != parent["payload"]["input_history_sha256"]
        || p["data_cutoff"] != legacy["payload"]["data_cutoff"]
        || !gate.publish
        || gate.target != issue
        || p["operational_cutoff_unix"] != gate.deadline_unix
        || at
            < legacy["payload"]["published_at_unix"]
                .as_u64()
                .ok_or("v2父档缺少保存时间")?
        || p["bundle"] != model::build(&prefix, &parent, &legacy["payload"])?
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
    {
        return Err("联合优化v3与模型、父档、票面或开奖前边界不一致".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"version":3,"experimental":true,"actual_recommendation":true,"model_source_sha256":p["model_source_sha256"],"bundle_sha256":crate::hash::sha256_hex(&serde_json::to_vec(&p["bundle"]).unwrap()),"issue":p["issue"],"archive_sha256":e["sha256"],"reference_archive_sha256":p["reference_archive_sha256"],"v2_archive_sha256":p["v2_archive_sha256"],"input_history_sha256":p["input_history_sha256"],"data_cutoff":p["data_cutoff"],"published_at_unix":p["published_at_unix"],"independent_timestamp_verified":false,"raw_url":format!("api/reference-pools/fusion-v3/archives/{}",p["issue"].as_str().unwrap())})
}
pub fn freeze(root: &Path, draws: &[Draw], issue: &str, deadline: u64) -> Result<Value, String> {
    freeze_with_clock(root, draws, issue, deadline, rec::now_unix)
}
fn freeze_with_clock(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
    clock: impl Fn() -> u64,
) -> Result<Value, String> {
    if let Some(e) = checked(root, draws, issue)? {
        let mut v = metadata(&e);
        v["status"] = json!("already_frozen");
        return Ok(v);
    }
    let started = clock();
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = crate::cycle::decide(started, last.issue, last.date, false);
    if !valid(issue)
        || !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || issue != fmt_issue(last.issue + 1)
    {
        return Err("联合优化v3数据、期号或截止已变化".into());
    }
    let Some((parent, legacy)) = parents(root, draws, issue)? else {
        return Ok(json!({"status":"waiting","reason":"等待本期首次参考5注及融合v2对照封存"}));
    };
    if legacy["payload"]["published_at_unix"]
        .as_u64()
        .is_none_or(|at| at > started)
    {
        return Err("联合优化v3时钟早于父档".into());
    }
    let bundle = model::build(draws, &parent, &legacy["payload"])?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("联合优化v3计算越过截止或时钟倒退".into());
    }
    let p = json!({"schema_version":1,"kind":KIND,"issue":issue,"data_cutoff":fmt_issue(last.issue),"input_history_sha256":data_sha256(draws),"published_at_unix":completed,"operational_cutoff_unix":deadline,"known_outcome_at_generation":false,"actual_recommendation":true,"experimental":true,"model_source_sha256":source(),"reference_archive_sha256":parent["sha256"],"v2_archive_sha256":legacy["sha256"],"bundle":bundle});
    let previous_clock = std::cell::Cell::new(completed);
    let monotone_clock = || {
        let at = clock();
        if at < previous_clock.get() {
            deadline
        } else {
            previous_clock.set(at);
            at
        }
    };
    let (_, created) = rec::write_new_before(
        &root.join(format!("fusion-v3/{issue}.json")),
        p,
        Some(deadline),
        monotone_clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("联合优化v3写入后缺档")?;
    let mut v = metadata(&e);
    v["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(v)
}
fn issues(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(es) = fs::read_dir(root.join("fusion-v3")) {
        for e in es.flatten() {
            if let Some(i) = e
                .path()
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".json"))
                .filter(|i| valid(i))
            {
                out.push(i.to_string())
            }
        }
    }
    out.sort();
    out
}
pub fn enabled(root: &Path) -> bool {
    !issues(root).is_empty()
}
fn review_saved(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().ok_or("v3缺期号")?;
    let path = root.join(format!("reviews/{issue}/fusion-v3.json"));
    if !path.is_file() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("v3有复盘但缺少开奖")?;
    if r["payload"]["kind"] != "fusion_v3_review"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["bundle"] != model::reviewed(&e["payload"]["bundle"], Some(d))?
    {
        return Err("v3复盘与首次原档或开奖不一致".into());
    }
    Ok(Some(r))
}
fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let saved = review_saved(root, &e, draw)?;
    let mut v = metadata(&e);
    v["status"] = json!("frozen");
    v["pending"] = json!(draw.is_none());
    v["bundle"] = model::reviewed(&e["payload"]["bundle"], draw)?;
    v["review_persisted"] = json!(saved.is_some());
    if saved.is_some() {
        v["review_url"] = json!(format!("api/reference-pools/fusion-v3/reviews/{issue}"));
    }
    Ok(Some(v))
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut result = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    for issue in issues(root) {
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let saved = (|| -> Result<usize, String> {
            let e = checked(root, draws, &issue)?.ok_or("v3缺少原档")?;
            if review_saved(root, &e, Some(draw))?.is_some() {
                return Ok(0);
            }
            let p = json!({"schema_version":1,"kind":"fusion_v3_review","issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(draw)),"reviewed_at_unix":rec::now_unix(),"bundle":model::reviewed(&e["payload"]["bundle"],Some(draw))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/fusion-v3.json")),
                p,
                None,
                rec::now_unix,
            )?;
            review_saved(root, &e, Some(draw))?;
            Ok(usize::from(created))
        })();
        match saved {
            Ok(n) => result.settled += n,
            Err(e) => result.errors.push(format!("{issue}/fusion-v3：{e}")),
        }
    }
    result
}
pub fn report(root: &Path, draws: &[Draw], requested: Option<&str>) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let target = fmt_issue(last.issue + 1);
    let issue = requested.unwrap_or(&target);
    if !valid(issue) {
        return Err("联合优化v3期号无效".into());
    }
    let gate = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    let current = if let Some(v) = view(root, draws, issue)? {
        v
    } else if issue == target && gate.publish && gate.target == issue {
        if let Some((parent, legacy)) = parents(root, draws, issue)? {
            {
                let bundle = model::build(draws, &parent, &legacy["payload"])?;
                json!({"status":"preview","issue":issue,"pending":true,"actual_recommendation":false,"model_source_sha256":source(),"bundle_sha256":crate::hash::sha256_hex(&serde_json::to_vec(&bundle).unwrap()),"bundle":model::reviewed(&bundle,None)?})
            }
        } else {
            json!({"status":"waiting","reason":"等待本期首次参考5注及v2对照档案"})
        }
    } else {
        json!({"status":"missing","reason":"本期未封存联合优化v3，不补造历史或截止后的票面"})
    };
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    for i in issues(root) {
        match view(root, draws, &i) {
            Ok(Some(v)) => periods.push(v),
            Ok(None) => {}
            Err(e) => errors.push(format!("{i}：{e}")),
        }
    }
    let actual = json!({"n":periods.iter().filter(|p|p["pending"]==false).count(),"pending":periods.iter().filter(|p|p["pending"]==true).count(),"periods":periods,"errors":errors,"note":"仅统计本版本首次真实留档，历史研究回放不计入。各买法独立比较，预算不合并。"});
    Ok(
        json!({"version":3,"experimental":true,"issue":issue,"target":target,"data_sha256":data_sha256(draws),"current":current,"actual":actual,"policy":model::policy(),"publication":{"can_freeze":issue==target&&gate.publish&&gate.target==issue,"deadline":gate.deadline_text},"automatic_continuation":enabled(root),"automatic_promotion":false}),
    )
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("联合优化v3原档不存在")?;
    fs::read(root.join(format!("fusion-v3/{issue}.json"))).map_err(|e| e.to_string())
}
pub fn review_artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    let e = checked(root, draws, issue)?.ok_or("联合优化v3原档不存在")?;
    review_saved(root, &e, draws.iter().find(|d| fmt_issue(d.issue) == issue))?
        .ok_or("联合优化v3复盘不存在")?;
    fs::read(root.join(format!("reviews/{issue}/fusion-v3.json"))).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: u64 = 1794880800;
    fn fixture() -> (std::path::PathBuf, Vec<Draw>, u64) {
        let root = std::env::temp_dir().join(format!(
            "ssq-fusion-v3-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut draws = crate::draw::load_csv(Path::new("data/ssq.csv")).unwrap();
        let last = draws.last_mut().unwrap();
        last.issue = 26131;
        last.date = 20261115;
        let deadline = crate::cycle::decide(NOW, 26131, 20261115, false).deadline_unix;
        let cache = json!({"version":1,"data_sha256":data_sha256(&draws),"data_cutoff":"26131","target_issue":"26132","generated_at_unix":NOW-100,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
        fs::create_dir_all(root.join("reference-cache")).unwrap();
        fs::write(
            root.join(format!("reference-cache/{}.json", data_sha256(&draws))),
            serde_json::to_vec(&cache).unwrap(),
        )
        .unwrap();
        crate::reference::freeze_with_clock(&root, &draws, "26132", deadline, || NOW - 10).unwrap();
        let parent = crate::reference::original(&root, &draws, "26132")
            .unwrap()
            .unwrap();
        let prepared = crate::eval::prepare_forecast(&draws).unwrap();
        let mut payload = rec::base_payload(
            &prepared,
            "26132",
            "reference_fusion_repair_v2",
            NOW - 5,
            Some(deadline),
        );
        payload["reference_archive_sha256"] = parent["sha256"].clone();
        payload["model_source_sha256"] = json!(crate::hash::sha256_hex(
            concat!(
                include_str!("fusion_model_v2.rs"),
                include_str!("model.rs"),
                include_str!("select.rs"),
                include_str!("protocol.rs"),
                include_str!("eval.rs"),
                include_str!("coverage.rs"),
                include_str!("plan.rs")
            )
            .as_bytes()
        ));
        payload["actual_recommendation"] = json!(true);
        payload["bundle"] =
            crate::fusion_model_v2::build(&prepared, &parent, draws.last().unwrap()).unwrap();
        rec::write_new_before(
            &root.join("fusion-v2/26132.json"),
            payload,
            Some(deadline),
            || NOW - 5,
        )
        .unwrap();
        (root, draws, deadline)
    }
    #[test]
    fn first_record_is_immutable_causal_and_settles_original_complete_tickets() {
        let (root, mut draws, deadline) = fixture();
        let parent = fs::read(root.join("reference/26132.json")).unwrap();
        let legacy = fs::read(root.join("fusion-v2/26132.json")).unwrap();
        assert!(!enabled(&root));
        let original = model::build(
            &draws,
            &crate::reference::original(&root, &draws, "26132")
                .unwrap()
                .unwrap(),
            &serde_json::from_slice::<Value>(&legacy).unwrap()["payload"],
        )
        .unwrap();
        assert!(!root.join("fusion-v3").exists());
        assert_eq!(
            freeze_with_clock(&root, &draws, "26132", deadline, || NOW).unwrap()["status"],
            "frozen"
        );
        assert!(enabled(&root));
        let bytes = artifact(&root, &draws, "26132").unwrap();
        assert_eq!(
            freeze_with_clock(&root, &draws, "26132", deadline, || deadline + 1).unwrap()["status"],
            "already_frozen"
        );
        let envelope: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(original, envelope["payload"]["bundle"]);
        let v = view(&root, &draws, "26132").unwrap().unwrap();
        assert!(v["bundle"]["plans"][0]["single_jackpot"].is_null());
        assert!(v["bundle"]["plans"][0]["summary"]["max_ticket_red_hits"].is_null());
        assert_eq!(
            v["bundle"]["plans"][0]["tickets"].as_array().unwrap().len(),
            8
        );
        let t = &v["bundle"]["plans"][0]["tickets"][0];
        draws.push(
            Draw::from_parts(
                26132,
                20261117,
                serde_json::from_value(t["red"].clone()).unwrap(),
                t["blue"].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        assert_eq!(artifact(&root, &draws, "26132").unwrap(), bytes);
        assert_eq!(settle(&root, &draws).settled, 1);
        let review = review_artifact(&root, &draws, "26132").unwrap();
        assert_eq!(settle(&root, &draws).settled, 0);
        assert_eq!(review_artifact(&root, &draws, "26132").unwrap(), review);
        assert_eq!(
            view(&root, &draws, "26132").unwrap().unwrap()["bundle"]["plans"][0]["single_jackpot"],
            true
        );
        assert_eq!(fs::read(root.join("reference/26132.json")).unwrap(), parent);
        assert_eq!(fs::read(root.join("fusion-v2/26132.json")).unwrap(), legacy);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn deadline_clock_reversal_stale_issue_and_rehashed_tampering_are_rejected() {
        let (root, draws, deadline) = fixture();
        assert!(freeze_with_clock(&root, &draws, "26133", deadline, || NOW).is_err());
        let tick = std::cell::Cell::new(0);
        assert!(freeze_with_clock(&root, &draws, "26132", deadline, || {
            let n = tick.get();
            tick.set(n + 1);
            if n == 0 {
                NOW
            } else {
                NOW - 1
            }
        })
        .is_err());
        assert!(freeze_with_clock(&root, &draws, "26132", deadline, || deadline).is_err());
        assert!(!root.join("fusion-v3/26132.json").exists());
        let writes = std::cell::Cell::new(0);
        assert!(freeze_with_clock(&root, &draws, "26132", deadline, || {
            let n = writes.get();
            writes.set(n + 1);
            if n < 2 {
                NOW
            } else {
                NOW - 1
            }
        })
        .is_err());
        assert!(!root.join("fusion-v3/26132.json").exists());
        freeze_with_clock(&root, &draws, "26132", deadline, || NOW).unwrap();
        let path = root.join("fusion-v3/26132.json");
        let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for field in ["issue", "reference_archive_sha256", "model_source_sha256"] {
            let mut e = original.clone();
            e["payload"][field] = json!("wrong");
            e["sha256"] = json!(crate::hash::sha256_hex(
                &serde_json::to_vec(&e["payload"]).unwrap()
            ));
            fs::write(&path, serde_json::to_vec(&e).unwrap()).unwrap();
            assert!(artifact(&root, &draws, "26132").is_err(), "{field}");
        }
        let mut e = original;
        e["payload"]["bundle"]["plans"][0]["tickets"][0]["blue"] = json!(16);
        e["sha256"] = json!(crate::hash::sha256_hex(
            &serde_json::to_vec(&e["payload"]).unwrap()
        ));
        fs::write(&path, serde_json::to_vec(&e).unwrap()).unwrap();
        assert!(artifact(&root, &draws, "26132").is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
