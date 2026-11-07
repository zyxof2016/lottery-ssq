//! Versioned active repair. Legacy reference and fusion archives stay untouched.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    fusion_model_v2 as model,
    hash::sha256_hex,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::{fs, path::Path};
const KIND: &str = "reference_fusion_repair_v2";
fn valid(i: &str) -> bool {
    i.len() == 5 && i.bytes().all(|b| b.is_ascii_digit())
}
fn source() -> String {
    sha256_hex(
        concat!(
            include_str!("fusion_model_v2.rs"),
            include_str!("model.rs"),
            include_str!("select.rs"),
            include_str!("protocol.rs"),
            include_str!("eval.rs"),
            include_str!("coverage.rs"),
            include_str!("plan.rs")
        )
        .as_bytes(),
    )
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid(issue) {
        return Err("融合v2期号须为5位数字".into());
    }
    let path = root.join(format!("fusion-v2/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let parent = crate::reference::original(root, draws, issue)?.ok_or("融合v2缺少参考父档")?;
    let cutoff = parent["payload"]["data_cutoff"]
        .as_str()
        .ok_or("参考父档缺少截止期")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    let last = prefix.last().ok_or("融合v2输入为空")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("缺少融合v2保存时间")?;
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if p["kind"] != KIND
        || p["model_source_sha256"] != source()
        || p["reference_archive_sha256"] != parent["sha256"]
        || p["input_history_sha256"] != parent["payload"]["input_history_sha256"]
        || p["data_cutoff"] != cutoff
        || !gate.publish
        || gate.target != issue
        || p["operational_cutoff_unix"] != gate.deadline_unix
        || at
            < parent["payload"]["published_at_unix"]
                .as_u64()
                .ok_or("缺少参考保存时间")?
        || p["known_outcome_at_generation"] != false
        || p["bundle"] != model::build(&prepare_forecast(&prefix)?, &parent, last)?
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
    {
        return Err("融合v2原档与算法、父档、票面或封存边界不一致".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"version":2,"issue":p["issue"],"archive_sha256":e["sha256"],"reference_archive_sha256":p["reference_archive_sha256"],"input_history_sha256":p["input_history_sha256"],"data_cutoff":p["data_cutoff"],"published_at_unix":p["published_at_unix"],"independent_timestamp_verified":false,"raw_url":format!("api/reference-pools/fusion-v2/archives/{}",p["issue"].as_str().unwrap())})
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
    if !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || issue != fmt_issue(last.issue + 1)
    {
        return Err("融合v2输入、期号或截止已变化".into());
    }
    let Some(parent) = crate::reference::original(root, draws, issue)? else {
        return Ok(json!({"status":"waiting","reason":"等待本期参考原始5注封存"}));
    };
    if parent["payload"]["published_at_unix"]
        .as_u64()
        .is_none_or(|at| at > started)
    {
        return Err("融合v2时钟早于参考父档".into());
    }
    let prepared = prepare_forecast(draws)?;
    let bundle = model::build(&prepared, &parent, last)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("融合v2计算越过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["reference_archive_sha256"] = parent["sha256"].clone();
    p["model_source_sha256"] = json!(source());
    p["bundle"] = bundle;
    p["actual_recommendation"] = json!(true);
    let (_, created) = rec::write_new_before(
        &root.join(format!("fusion-v2/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("融合v2写入后未找到")?;
    let mut v = metadata(&e);
    v["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(v)
}
fn review_saved(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().ok_or("缺少融合v2期号")?;
    let path = root.join(format!("reviews/{issue}/fusion-v2.json"));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("融合v2已有复盘但缺少开奖")?;
    if r["payload"]["kind"] != "fusion_v2_review"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["bundle"] != model::reviewed(&e["payload"]["bundle"], Some(d))?
    {
        return Err("融合v2复盘与原档或开奖不一致".into());
    }
    Ok(Some(r))
}
fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let review = review_saved(root, &e, draw)?;
    let mut v = metadata(&e);
    v["status"] = json!("frozen");
    v["pending"] = json!(draw.is_none());
    v["bundle"] = model::reviewed(&e["payload"]["bundle"], draw)?;
    v["review_persisted"] = json!(review.is_some());
    if review.is_some() {
        v["review_url"] = json!(format!("api/reference-pools/fusion-v2/reviews/{issue}"));
    }
    Ok(Some(v))
}
fn issues(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(root.join("fusion-v2")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(i) = p.file_stem().and_then(|s| s.to_str()).filter(|i| valid(i)) {
                    out.push(i.to_string());
                }
            }
        }
    }
    out.sort();
    out
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut result = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    for i in issues(root) {
        let Some(d) = draws.iter().find(|d| fmt_issue(d.issue) == i) else {
            continue;
        };
        let saved = (|| -> Result<usize, String> {
            let e = checked(root, draws, &i)?.ok_or("缺少融合v2原档")?;
            if review_saved(root, &e, Some(d))?.is_some() {
                return Ok(0);
            }
            let p = json!({"schema_version":1,"kind":"fusion_v2_review","issue":i,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(d)),"reviewed_at_unix":rec::now_unix(),"bundle":model::reviewed(&e["payload"]["bundle"],Some(d))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{i}/fusion-v2.json")),
                p,
                None,
                rec::now_unix,
            )?;
            review_saved(root, &e, Some(d))?;
            Ok(usize::from(created))
        })();
        match saved {
            Ok(n) => result.settled += n,
            Err(e) => result.errors.push(format!("{i}/fusion-v2：{e}")),
        }
    }
    result
}
fn observations(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending, mut periods, mut errors) = (0, 0, Vec::new(), Vec::new());
    for issue in issues(root) {
        match view(root, draws, &issue) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                }
                periods.push(v);
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"version":2,"n":n,"pending":pending,"periods":periods,"errors":errors,"note":"仅统计v2首次封存，旧版v1与历史缺档不混入；待开奖指标为空。"})
}
pub fn report(root: &Path, draws: &[Draw], requested: Option<&str>) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let target = fmt_issue(last.issue + 1);
    let issue = requested.unwrap_or(&target);
    if !valid(issue) {
        return Err("融合v2期号无效".into());
    }
    let gate = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    let current = if let Some(v) = view(root, draws, issue)? {
        v
    } else if issue == target && gate.publish && gate.target == issue {
        if let Some(parent) = crate::reference::original(root, draws, issue)? {
            json!({"version":2,"status":"preview","issue":issue,"reference_archive_sha256":parent["sha256"],"bundle":model::reviewed(&model::build(&prepare_forecast(draws)?,&parent,last)?,None)?})
        } else {
            json!({"status":"waiting","reason":"等待本期参考原始5注封存"})
        }
    } else {
        json!({"status":"missing","reason":"本期未封存修正版v2，保留旧版原档，不补造历史或截止后的号码。"})
    };
    Ok(
        json!({"version":2,"kind":KIND,"issue":issue,"target":target,"data_sha256":data_sha256(draws),"current":current,"actual":observations(root,draws),"publication":{"can_freeze":issue==target&&gate.publish&&gate.target==issue,"deadline":gate.deadline_text},"policy":model::policy()}),
    )
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("融合v2原档不存在")?;
    fs::read(root.join(format!("fusion-v2/{issue}.json"))).map_err(|e| e.to_string())
}
pub fn review_artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    let e = checked(root, draws, issue)?.ok_or("融合v2原档不存在")?;
    review_saved(root, &e, draws.iter().find(|d| fmt_issue(d.issue) == issue))?
        .ok_or("融合v2复盘尚未保存")?;
    fs::read(root.join(format!("reviews/{issue}/fusion-v2.json"))).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: u64 = 1794010000;
    fn fixture() -> (std::path::PathBuf, Vec<Draw>) {
        let root = std::env::temp_dir().join(format!(
            "ssq-fusion-v2-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut d = crate::draw::synth_draws(220, 9);
        let last = d.last_mut().unwrap();
        last.issue = 26127;
        last.date = 20261105;
        let cache = json!({"version":1,"data_sha256":data_sha256(&d),"data_cutoff":"26127","target_issue":"26128","generated_at_unix":NOW-100,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
        fs::create_dir_all(root.join("reference-cache")).unwrap();
        fs::write(
            root.join(format!("reference-cache/{}.json", data_sha256(&d))),
            serde_json::to_vec(&cache).unwrap(),
        )
        .unwrap();
        let g = crate::cycle::decide(NOW, 26127, 20261105, false);
        crate::reference::freeze_with_clock(&root, &d, "26128", g.deadline_unix, || NOW - 10)
            .unwrap();
        (root, d)
    }
    #[test]
    fn seal_preserves_legacy_and_both_buyable_shapes_then_settles_original_tickets() {
        let (root, mut d) = fixture();
        let legacy = fs::read(root.join("reference/26128.json")).unwrap();
        let gate = crate::cycle::decide(NOW, 26127, 20261105, false);
        assert!(!root.join("fusion-v2").exists());
        freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW).unwrap();
        let raw = artifact(&root, &d, "26128").unwrap();
        assert_eq!(
            freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || gate
                .deadline_unix
                + 1)
            .unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(artifact(&root, &d, "26128").unwrap(), raw);
        let v = view(&root, &d, "26128").unwrap().unwrap();
        let b = &v["bundle"];
        assert_eq!(b["singles"].as_array().unwrap().len(), 4);
        assert_eq!(b["dantuo"].as_array().unwrap().len(), 2);
        for p in b["singles"]
            .as_array()
            .unwrap()
            .iter()
            .chain(b["dantuo"].as_array().unwrap())
        {
            let bets = if p["mode"] == "dantuo" { 10 } else { 8 };
            assert_eq!(p["summary"]["bets"], bets);
            assert_eq!(p["summary"]["duplicate_tickets"], 0);
            assert!(p["summary"]["max_ticket_red_hits"].is_null());
            for t in p["tickets"].as_array().unwrap() {
                assert_eq!(t["red"].as_array().unwrap().len(), 6);
                for dan in p["red_dan"].as_array().unwrap() {
                    assert!(t["red"].as_array().unwrap().contains(dan));
                }
            }
        }
        let t = &b["dantuo"][0]["tickets"][0];
        let red: Vec<u8> = serde_json::from_value(t["red"].clone()).unwrap();
        d.push(
            Draw::from_parts(
                26128,
                20261108,
                red.try_into().unwrap(),
                t["blue"].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        assert_eq!(settle(&root, &d).settled, 1);
        assert_eq!(settle(&root, &d).settled, 0);
        let v = view(&root, &d, "26128").unwrap().unwrap();
        assert_eq!(
            v["bundle"]["dantuo"][0]["summary"]["max_ticket_red_hits"],
            6
        );
        assert_eq!(v["review_persisted"], true);
        assert_eq!(fs::read(root.join("reference/26128.json")).unwrap(), legacy);
        assert_eq!(artifact(&root, &d, "26128").unwrap(), raw);
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn deadline_race_clock_reversal_and_stale_issue_do_not_write() {
        let (root, d) = fixture();
        let g = crate::cycle::decide(NOW, 26127, 20261105, false);
        assert!(freeze_with_clock(&root, &d, "26127", g.deadline_unix, || NOW).is_err());
        for times in [
            [NOW, NOW - 1, NOW - 1],
            [NOW, g.deadline_unix, g.deadline_unix],
            [NOW, NOW, g.deadline_unix],
        ] {
            let tick = std::cell::Cell::new(0);
            assert!(freeze_with_clock(&root, &d, "26128", g.deadline_unix, || {
                let i = tick.get();
                tick.set(i + 1);
                times[i.min(2)]
            })
            .is_err());
            assert!(!root.join("fusion-v2/26128.json").exists());
        }
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn rehashed_ticket_or_parent_tampering_is_rejected() {
        let (root, d) = fixture();
        let g = crate::cycle::decide(NOW, 26127, 20261105, false);
        freeze_with_clock(&root, &d, "26128", g.deadline_unix, || NOW).unwrap();
        let bytes = artifact(&root, &d, "26128").unwrap();
        let mut e: Value = serde_json::from_slice(&bytes).unwrap();
        e["payload"]["bundle"]["singles"][0]["tickets"][0]["blue"] = json!(16);
        e["sha256"] = json!(sha256_hex(
            serde_json::to_string(&e["payload"]).unwrap().as_bytes()
        ));
        fs::write(
            root.join("fusion-v2/26128.json"),
            serde_json::to_vec(&e).unwrap(),
        )
        .unwrap();
        assert!(artifact(&root, &d, "26128").is_err());
        let _ = fs::remove_dir_all(root);
    }
}
