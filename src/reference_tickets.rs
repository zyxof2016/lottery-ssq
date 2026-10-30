//! Buyable fusion tickets, sealed separately from the immutable red-pool record.
use crate::{
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    eval::prepare_forecast,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "reference_fusion_tickets";
const SIZES: [usize; 4] = [12, 14, 15, 16];
pub fn policy() -> Value {
    json!({"version":1,"mode":"balanced_single","tickets":8,"cost":16,"red_sizes":SIZES,"blue_count":2,"blue_assignment":"alternate_original_two_blues","red_assignment":"existing_balanced_coverage_v1","note":"每档8注单式，每注6个不同红球和1个蓝球，共16元；四档独立方案，购买时选择其中一档。红球和2蓝严格来自融合原档。"})
}
pub fn preview(snapshot: &Value) -> Result<Value, String> {
    let blues: Vec<u8> = serde_json::from_value(snapshot["fusion"]["blues"].clone())
        .map_err(|_| "融合原档缺少蓝球")?;
    if blues.len() != 2 || blues[0] == blues[1] || blues.iter().any(|n| !(1..=16).contains(n)) {
        return Err("融合原档须保存2个有效蓝球".into());
    }
    let pools = snapshot["fusion"]["pools"]
        .as_array()
        .ok_or("融合原档缺少红球池")?;
    let mut plans = Vec::new();
    for k in SIZES {
        let p = pools
            .iter()
            .find(|p| p["red_count"] == k)
            .ok_or("融合原档缺少规格")?;
        let reds: Vec<u8> =
            serde_json::from_value(p["reds"].clone()).map_err(|_| "融合原档红球无效")?;
        if reds.len() != k
            || reds.iter().any(|n| !(1..=33).contains(n))
            || reds.iter().collect::<BTreeSet<_>>().len() != k
        {
            return Err("融合红球规格或号码无效".into());
        }
        let order: Vec<_> = reds.iter().map(|n| *n as usize - 1).collect();
        let masks = crate::coverage::balanced_tickets(&order, 8)?;
        let tickets:Vec<_>=masks.into_iter().enumerate().map(|(i,mask)|json!({"red":(1..=33).filter(|n|mask&(1u64<<(n-1))!=0).collect::<Vec<_>>(),"blue":blues[i%2]})).collect();
        let saved = json!({"title":format!("参考融合{k}红 · 均衡8注"),"mode":"balanced","rationale":"按融合原档红球组票，完整覆盖候选池，2蓝交替；每档8注16元。","structural":true,"tickets":tickets});
        let check = rec::review_strategy(&saved, None)?;
        if check["summary"]["bets"] != 8
            || check["summary"]["cost"] != 16
            || check["summary"]["red_coverage"] != k
            || check["summary"]["blue_coverage"] != 2
            || check["summary"]["duplicate_tickets"] != 0
        {
            return Err("融合组票未达到完整覆盖及合法票组规格".into());
        }
        plans.push(json!({"id":format!("reference_fused_tickets_red{k}"),"red_count":k,"reds":reds,"blues":blues,"bet_count":8,"cost":16,"title":saved["title"],"tickets":saved["tickets"]}));
    }
    Ok(
        json!({"version":1,"policy":policy(),"issue":snapshot["target_issue"],"data_cutoff":snapshot["data_cutoff"],"input_history_sha256":snapshot["input_history_sha256"],"plans":plans}),
    )
}
fn score(snapshot: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = snapshot.clone();
    for p in v["plans"].as_array_mut().ok_or("缺少融合票组")? {
        let r = rec::review_strategy(
            &json!({"title":p["title"],"mode":"balanced","structural":true,"tickets":p["tickets"]}),
            draw,
        )?;
        p["ticket_results"] = r["tickets"].clone();
        p["summary"] = r["summary"].clone();
        p["prize_rules_url"] = r["prize_rules_url"].clone();
        p["prize_note"] = r["explanation_limit"].clone();
    }
    Ok(v)
}
fn issue_ok(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|c| c.is_ascii_digit())
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !issue_ok(issue) {
        return Err("融合票组期号须为5位".into());
    }
    let path = root.join(format!("reference-tickets/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    if p["kind"] != KIND {
        return Err("融合票组档案类型无效".into());
    }
    rec::validate_archive(p, issue, draws)?;
    let parent =
        crate::reference::original(root, draws, issue)?.ok_or("融合票组缺少绑定的参考原档")?;
    if p["reference_archive_sha256"] != parent["sha256"]
        || p["ticket_pools"] != preview(&parent["payload"]["reference_pools"])?
    {
        return Err("融合票组与原始红蓝号码或父档案摘要不一致".into());
    }
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    if rec::saved_after_draw(p["published_at_unix"].as_u64(), draw) {
        return Err("融合票组保存晚于开奖，不计入实际推荐".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"source":"reference-tickets","raw_file":format!("reference-tickets/{}.json",p["issue"].as_str().unwrap()),"archive_sha256":e["sha256"],"reference_archive_sha256":p["reference_archive_sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"independent_timestamp_verified":false})
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
        let mut m = metadata(&e);
        m["status"] = json!("already_frozen");
        return Ok(m);
    }
    let started = clock();
    if started >= deadline {
        return Err("已过融合票组封存截止".into());
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    if issue != fmt_issue(last.issue + 1) || issue[..2] != fmt_issue(last.issue)[..2] {
        return Err("融合票组只允许当前紧邻下一期，跨年需核对".into());
    }
    let parent = crate::reference::original(root, draws, issue)?
        .ok_or("参考原档尚未封存，不能封存融合票组")?;
    let tickets = preview(&parent["payload"]["reference_pools"])?;
    let prepared = prepare_forecast(draws)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("融合组票计算越过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["reference_archive_sha256"] = parent["sha256"].clone();
    p["ticket_pools"] = tickets;
    let (_, created) = rec::write_new_before(
        &root.join(format!("reference-tickets/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("融合票组写入后未找到原档")?;
    let mut m = metadata(&e);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn persisted(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().ok_or("缺少期号")?;
    let path = root.join(format!("reviews/{issue}/reference-tickets.json"));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("已有融合票组复盘但当前缺少开奖")?;
    if r["payload"]["kind"] != "reference_fusion_tickets_review"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["ticket_pools"] != score(&e["payload"]["ticket_pools"], Some(d))?
    {
        return Err("融合票组复盘与原档或开奖不一致".into());
    }
    Ok(Some(r))
}
pub fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let saved = persisted(root, &e, draw)?;
    let mut m = metadata(&e);
    let reviewed = score(&e["payload"]["ticket_pools"], draw)?;
    m["plans"] = reviewed["plans"].clone();
    m["policy"] = policy();
    m["status"] = json!("frozen");
    m["pending"] = json!(draw.is_none());
    m["review_persisted"] = json!(saved.is_some());
    if saved.is_some() {
        m["review_file"] = json!(format!("reviews/{issue}/reference-tickets.json"));
    }
    Ok(Some(m))
}
pub fn current(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    snapshot: &Value,
) -> Result<Value, String> {
    if let Some(v) = view(root, draws, issue)? {
        return Ok(v);
    }
    if snapshot.is_null() {
        return Ok(Value::Null);
    }
    let mut v = score(&preview(snapshot)?, None)?;
    v["status"] = json!("preview");
    v["actual_recommendation"] = json!(false);
    Ok(v)
}
fn issues(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for folder in ["daily", "reference", "reference-tickets"] {
        if let Ok(entries) = fs::read_dir(root.join(folder)) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|x| x.to_str()) == Some("json") {
                    if let Some(i) = p
                        .file_stem()
                        .and_then(|x| x.to_str())
                        .filter(|s| issue_ok(s))
                    {
                        out.insert(i.to_string());
                    }
                }
            }
        }
    }
    out
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut out = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    for issue in issues(root) {
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let result = (|| -> Result<usize, String> {
            let Some(e) = checked(root, draws, &issue)? else {
                return Ok(0);
            };
            if persisted(root, &e, Some(draw))?.is_some() {
                return Ok(0);
            }
            let p = json!({"kind":"reference_fusion_tickets_review","schema_version":1,"issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(draw)),"reviewed_at_unix":rec::now_unix(),"draw":{"issue":issue,"date":fmt_date(draw.date),"red":draw.red,"blue":draw.blue},"ticket_pools":score(&e["payload"]["ticket_pools"],Some(draw))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/reference-tickets.json")),
                p,
                None,
                rec::now_unix,
            )?;
            persisted(root, &e, Some(draw))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/reference-tickets：{e}")),
        }
    }
    out
}
pub fn observations(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending, mut missing) = (0usize, 0usize, 0usize);
    let (mut prizes, mut fixed, mut best, mut blue) =
        ([0usize; 4], [0u64; 4], [0u64; 4], [0usize; 4]);
    let mut errors = Vec::new();
    for issue in issues(root) {
        match view(root, draws, &issue) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    for (i, p) in v["plans"].as_array().unwrap().iter().enumerate() {
                        let s = &p["summary"];
                        prizes[i] += usize::from(s["prize_tickets"].as_u64().unwrap_or(0) > 0);
                        fixed[i] += s["fixed_prize"].as_u64().unwrap_or(0);
                        best[i] += s["max_ticket_red_hits"].as_u64().unwrap_or(0);
                        blue[i] += usize::from(s["blue_covered"] == true);
                    }
                }
            }
            Ok(None) => missing += 1,
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"cells":SIZES.iter().enumerate().map(|(i,k)|json!({"red_count":k,"bet_count":8,"cost":16,"n":n,"pending":pending,"excluded_missing":missing,"any_prize_periods":prizes[i],"any_prize_rate":(n>0).then(||prizes[i]as f64/n as f64),"fixed_prize_total":if n>0{json!(fixed[i])}else{Value::Null},"mean_best_red":(n>0).then(||best[i]as f64/n as f64),"blue_hit_periods":blue[i]})).collect::<Vec<_>>(),"errors":errors,"note":"只按实际封存的完整8注票组逐注核对；候选池全6红不等于单注中奖。"})
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(
                std::env::temp_dir().join(format!(
                    "ssq-fusion-tickets-{}-{}",
                    std::process::id(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )),
            )
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> (Temp, Vec<Draw>, String) {
        let t = Temp::new();
        let d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let c = json!({"version":1,"data_sha256":data_sha256(&d),"data_cutoff":fmt_issue(d.last().unwrap().issue),"target_issue":issue,"generated_at_unix":5,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
        fs::create_dir_all(t.0.join("reference-cache")).unwrap();
        fs::write(
            t.0.join(format!("reference-cache/{}.json", data_sha256(&d))),
            serde_json::to_vec(&c).unwrap(),
        )
        .unwrap();
        crate::reference::freeze_with_clock(&t.0, &d, &issue, 100, || 10).unwrap();
        (t, d, issue)
    }
    #[test]
    fn legal_balanced_tickets_and_read_only_preview_preserve_original_pool() {
        let (t, d, issue) = fixture();
        let raw = t.0.join(format!("reference/{issue}.json"));
        let original = fs::read(&raw).unwrap();
        let parent = crate::reference::original(&t.0, &d, &issue)
            .unwrap()
            .unwrap();
        let preview = current(&t.0, &d, &issue, &parent["payload"]["reference_pools"]).unwrap();
        assert_eq!(preview["status"], "preview");
        assert!(!t.0.join("reference-tickets").exists());
        assert!(!t.0.join("reviews").exists());
        for (i, k) in SIZES.iter().enumerate() {
            let p = &preview["plans"][i];
            assert_eq!(p["cost"], 16);
            assert_eq!(p["bet_count"], 8);
            let ts = p["tickets"].as_array().unwrap();
            assert_eq!(ts.len(), 8);
            let (mut red, mut blue) = (BTreeSet::new(), BTreeSet::new());
            let mut exposure = [0usize; 33];
            for ticket in ts {
                let r: Vec<u8> = serde_json::from_value(ticket["red"].clone()).unwrap();
                assert_eq!(r.len(), 6);
                assert_eq!(r.iter().collect::<BTreeSet<_>>().len(), 6);
                for n in r {
                    assert!(p["reds"].as_array().unwrap().contains(&json!(n)));
                    red.insert(n);
                    exposure[n as usize - 1] += 1;
                }
                let b = ticket["blue"].as_u64().unwrap();
                assert!(p["blues"].as_array().unwrap().contains(&json!(b)));
                blue.insert(b);
            }
            assert_eq!(red.len(), *k);
            assert_eq!(blue.len(), 2);
            assert_eq!(p["summary"]["duplicate_tickets"], 0);
            assert!(exposure
                .iter()
                .filter(|n| **n > 0)
                .all(|n| *n == 48 / k || *n == (48usize).div_ceil(*k)));
            assert!(p["ticket_results"][0]["red_hits"].is_null());
            assert!(p["summary"]["fixed_prize"].is_null());
        }
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        let sealed = view(&t.0, &d, &issue).unwrap().unwrap();
        assert_eq!(sealed["plans"], preview["plans"]);
        assert_eq!(fs::read(raw).unwrap(), original);
        assert_eq!(observations(&t.0, &d)["cells"][3]["pending"], 1);
        assert!(observations(&t.0, &d)["cells"][3]["any_prize_rate"].is_null());
    }
    #[test]
    fn first_archive_deadlines_parent_binding_and_tampering_are_enforced() {
        let (t, d, issue) = fixture();
        assert!(freeze_with_clock(&t.0, &d, &issue, 20, || 20).is_err());
        assert!(!t.0.join("reference-tickets").exists());
        let ticks = std::cell::Cell::new(0);
        assert!(freeze_with_clock(&t.0, &d, &issue, 20, || {
            let n = ticks.get();
            ticks.set(n + 1);
            if n == 0 {
                10
            } else {
                20
            }
        })
        .is_err());
        assert!(!t.0.join("reference-tickets").exists());
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        let path = t.0.join(format!("reference-tickets/{issue}.json"));
        let original = fs::read(&path).unwrap();
        assert_eq!(
            freeze_with_clock(&t.0, &d, &issue, 100, || 200).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        let mut changed = d.clone();
        changed[0].blue = changed[0].blue % 16 + 1;
        assert!(view(&t.0, &changed, &issue).is_err());
        let mut e = rec::read_checked(&path).unwrap();
        e["payload"]["ticket_pools"]["plans"][0]["tickets"][0]["blue"] = json!(16);
        e["sha256"] = json!(crate::hash::sha256_hex(
            &serde_json::to_vec(&e["payload"]).unwrap()
        ));
        fs::write(&path, serde_json::to_vec(&e).unwrap()).unwrap();
        assert!(view(&t.0, &d, &issue).is_err());
    }
    #[test]
    fn scoring_uses_original_singles_and_saved_review_is_immutable() {
        let (t, mut d, issue) = fixture();
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        let e = rec::read_checked(&t.0.join(format!("reference-tickets/{issue}.json"))).unwrap();
        let ticket = &e["payload"]["ticket_pools"]["plans"][3]["tickets"][0];
        let red: [u8; 6] = serde_json::from_value(ticket["red"].clone()).unwrap();
        let blue = ticket["blue"].as_u64().unwrap() as u8;
        d.push(Draw::from_parts(issue.parse().unwrap(), 20261029, red, blue).unwrap());
        let v = view(&t.0, &d, &issue).unwrap().unwrap();
        assert_eq!(v["plans"][3]["ticket_results"][0]["red_hits"], 6);
        assert_eq!(v["plans"][3]["ticket_results"][0]["blue_hit"], true);
        assert_eq!(v["plans"][3]["ticket_results"][0]["prize_level"], 1);
        assert!(
            v["plans"][3]["summary"]["floating_prize_tickets"]
                .as_u64()
                .unwrap()
                >= 1
        );
        assert!(!t.0.join("reviews").exists());
        assert_eq!(settle(&t.0, &d).settled, 1);
        let path = t.0.join(format!("reviews/{issue}/reference-tickets.json"));
        let old = fs::read(&path).unwrap();
        assert_eq!(settle(&t.0, &d).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), old);
        d.last_mut().unwrap().blue = blue % 16 + 1;
        assert!(!settle(&t.0, &d).errors.is_empty());
        assert_eq!(fs::read(path).unwrap(), old);
    }
    #[test]
    fn concurrent_publication_retains_one_first_record() {
        let (t, d, issue) = fixture();
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let (root, draws, target) = (t.0.clone(), d.clone(), issue.clone());
                std::thread::spawn(move || {
                    freeze_with_clock(&root, &draws, &target, 100, || 20).unwrap()
                })
            })
            .collect();
        let r: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(r[0]["archive_sha256"], r[1]["archive_sha256"]);
        assert_eq!(r.iter().filter(|r| r["status"] == "frozen").count(), 1);
    }
}
