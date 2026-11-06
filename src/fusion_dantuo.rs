//! Two buyable dantuo variants from the sealed fusion ranking. Independent
//! archives preserve the original reference, singles and budget plans.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "reference_fusion_dantuo_v1";
const SPECS: [(&str, usize, usize); 2] = [("2+5+2", 2, 2), ("3+5+1", 3, 1)];
fn issue_ok(i: &str) -> bool {
    i.len() == 5 && i.bytes().all(|b| b.is_ascii_digit())
}
fn policy() -> Value {
    json!({"version":1,"specifications":["2+5+2","3+5+1"],"selection":"sealed_fusion_ranking_prefix_bankers_then_five_drags; sealed_fusion_blue_prefix","unit_price":2,"bets_per_plan":10,"cost_per_plan":20,"note":"按融合原档排序选前2或3红作胆、随后5红作拖，蓝球取融合原档前2或1蓝；两种各10注20元，独立选择。"})
}
fn build(snapshot: &Value) -> Result<Value, String> {
    let ranking: Vec<u8> = serde_json::from_value(snapshot["fusion"]["ranking"].clone())
        .map_err(|_| "融合原档缺少排序")?;
    let blue: Vec<u8> = serde_json::from_value(snapshot["fusion"]["blues"].clone())
        .map_err(|_| "融合原档缺少蓝球")?;
    if ranking.len() != 33
        || ranking.iter().any(|r| !(1..=33).contains(r))
        || ranking.iter().collect::<BTreeSet<_>>().len() != 33
        || blue.len() != 2
        || blue[0] == blue[1]
        || blue.iter().any(|b| !(1..=16).contains(b))
    {
        return Err("融合排序或蓝球重复、越界或数量无效".into());
    }
    let mut plans = Vec::new();
    for (id, dan_n, blue_n) in SPECS {
        let dan = ranking[..dan_n].to_vec();
        let tuo = ranking[dan_n..dan_n + 5].to_vec();
        let blues = blue[..blue_n].to_vec();
        let mut tickets = Vec::new();
        for c in crate::plan::combinations(&tuo, 6 - dan_n)? {
            let mut red = dan.clone();
            red.extend(c);
            red.sort_unstable();
            for b in &blues {
                tickets.push(json!({"red":red,"blue":b}));
            }
        }
        let mut reds = dan.clone();
        reds.extend(&tuo);
        reds.sort_unstable();
        if tickets.len() != 10 {
            return Err("融合胆拖展开注数错误".into());
        }
        plans.push(json!({"id":id,"title":format!("融合胆拖 {id}"),"mode":"dantuo","red_dan":dan,"red_tuo":tuo,"reds":reds,"blues":blues,"bet_count":10,"cost":20,"unit_price":2,"tickets":tickets}));
    }
    Ok(json!({"version":1,"policy":policy(),"plans":plans}))
}
fn score(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = bundle.clone();
    for p in v["plans"].as_array_mut().ok_or("缺少融合胆拖方案")? {
        let r = rec::review_strategy(
            &json!({"title":p["title"],"mode":"dantuo","structural":true,"tickets":p["tickets"]}),
            draw,
        )?;
        p["ticket_results"] = r["tickets"].clone();
        p["summary"] = r["summary"].clone();
        p["prize_note"] = r["explanation_limit"].clone();
        p["dan_hits"] = json!(draw.map(|d| p["red_dan"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| d.red.iter().any(|r| json!(r) == **n))
            .count()));
        p["red_union_hits"] = json!(draw.map(|d| p["reds"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|n| d.red.iter().any(|r| json!(r) == **n))
            .count()));
    }
    Ok(v)
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !issue_ok(issue) {
        return Err("融合胆拖期号须为5位数字".into());
    }
    let path = root.join(format!("fusion-dantuo/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    rec::validate_archive(p, issue, draws)?;
    let parent =
        crate::reference::original(root, draws, issue)?.ok_or("融合胆拖缺少绑定的参考原档")?;
    if p["kind"] != KIND
        || p["reference_archive_sha256"] != parent["sha256"]
        || p["bundle"] != build(&parent["payload"]["reference_pools"])?
        || p["published_at_unix"]
            .as_u64()
            .zip(p["operational_cutoff_unix"].as_u64())
            .is_none_or(|(a, b)| a >= b)
        || p["known_outcome_at_generation"] != false
        || rec::saved_after_draw(
            p["published_at_unix"].as_u64(),
            draws.iter().find(|d| fmt_issue(d.issue) == issue),
        )
    {
        return Err("融合胆拖与原档、排序或封存边界不一致".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"archive_sha256":e["sha256"],"reference_archive_sha256":p["reference_archive_sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"independent_timestamp_verified":false,"raw_url":format!("api/reference-pools/fusion-dantuo/archives/{}",p["issue"].as_str().unwrap())})
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
        || issue != fmt_issue(last.issue + 1)
    {
        return Err("融合胆拖数据、目标期或封存截止已变化".into());
    }
    let Some(parent) = crate::reference::original(root, draws, issue)? else {
        return Ok(json!({"status":"waiting","reason":"等待本期参考融合原档封存"}));
    };
    let bundle = build(&parent["payload"]["reference_pools"])?;
    let prepared = prepare_forecast(draws)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("融合胆拖计算越过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["reference_archive_sha256"] = parent["sha256"].clone();
    p["bundle"] = bundle;
    let (_, created) = rec::write_new_before(
        &root.join(format!("fusion-dantuo/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("融合胆拖写入后未找到")?;
    let mut m = metadata(&e);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn saved_review(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().unwrap();
    let path = root.join(format!("reviews/{issue}/fusion-dantuo.json"));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("已有融合胆拖复盘但缺少开奖")?;
    if r["payload"]["kind"] != "reference_fusion_dantuo_review_v1"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["bundle"] != score(&e["payload"]["bundle"], Some(d))?
    {
        return Err("融合胆拖复盘与原档或开奖不一致".into());
    }
    Ok(Some(r))
}
fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let prior = saved_review(root, &e, draw)?;
    let scored = score(&e["payload"]["bundle"], draw)?;
    let mut v = metadata(&e);
    v["status"] = json!("frozen");
    v["pending"] = json!(draw.is_none());
    v["plans"] = scored["plans"].clone();
    v["policy"] = policy();
    v["review_persisted"] = json!(prior.is_some());
    Ok(Some(v))
}
fn issues(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(root.join("fusion-dantuo")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                if let Some(i) = p
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .filter(|i| issue_ok(i))
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
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let result = (|| -> Result<usize, String> {
            let e = checked(root, draws, &issue)?.ok_or("融合胆拖原档缺失")?;
            if saved_review(root, &e, Some(draw))?.is_some() {
                return Ok(0);
            }
            let p = json!({"kind":"reference_fusion_dantuo_review_v1","schema_version":1,"issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(draw)),"reviewed_at_unix":rec::now_unix(),"bundle":score(&e["payload"]["bundle"],Some(draw))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/fusion-dantuo.json")),
                p,
                None,
                rec::now_unix,
            )?;
            saved_review(root, &e, Some(draw))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/fusion-dantuo：{e}")),
        }
    }
    out
}
fn observations(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending) = (0, 0);
    let (mut prizes, mut best, mut fixed) = ([0; 2], [0.; 2], [0u64; 2]);
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
                        best[i] += s["max_ticket_red_hits"].as_f64().unwrap_or(0.);
                        fixed[i] += s["fixed_prize"].as_u64().unwrap_or(0);
                    }
                }
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"n":n,"pending":pending,"cells":SPECS.iter().enumerate().map(|(i,(id,_,_))|json!({"id":id,"n":n,"pending":pending,"bet_count":10,"cost":20,"any_prize_periods":prizes[i],"mean_best_red":(n>0).then(||best[i]/n as f64),"fixed_prize_total":(n>0).then_some(fixed[i])})).collect::<Vec<_>>(),"errors":errors,"note":"两种各10注20元，分别按首次封存的展开票面逐注核对。待开奖不计失败，旧期缺档不补造。"})
}
pub fn report(root: &Path, draws: &[Draw], requested: Option<&str>) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = crate::cycle::decide(rec::now_unix(), last.issue, last.date, false);
    let next = fmt_issue(last.issue + 1);
    let issue = requested.unwrap_or(&next);
    if !issue_ok(issue) {
        return Err("融合胆拖期号无效".into());
    }
    let current = if let Some(v) = view(root, draws, issue)? {
        v
    } else if issue == next && gate.publish && gate.target == issue {
        if let Some(parent) = crate::reference::original(root, draws, issue)? {
            let mut v = score(&build(&parent["payload"]["reference_pools"])?, None)?;
            v["status"] = json!("preview");
            v["issue"] = json!(issue);
            v["reference_archive_sha256"] = parent["sha256"].clone();
            v
        } else {
            json!({"status":"waiting","reason":"等待本期参考融合原档封存"})
        }
    } else {
        json!({"status":"missing","reason":"本期未封存融合胆拖，不补造历史或截止后的号码。"})
    };
    Ok(
        json!({"version":1,"kind":KIND,"issue":issue,"target":next,"data_sha256":data_sha256(draws),"current":current,"actual":observations(root,draws),"publication":{"can_freeze":issue==next && gate.publish && gate.target==issue,"deadline":gate.deadline_text}}),
    )
}
pub fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?.ok_or("融合胆拖原档不存在")?;
    fs::read(root.join(format!("fusion-dantuo/{issue}.json"))).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    const NOW: u64 = 1793926800;
    fn fixture() -> (std::path::PathBuf, Vec<Draw>) {
        let root = std::env::temp_dir().join(format!(
            "ssq-fusion-dantuo-{}-{}",
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
        let c = json!({"version":1,"data_sha256":data_sha256(&d),"data_cutoff":"26127","target_issue":"26128","generated_at_unix":NOW-10,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
        fs::create_dir_all(root.join("reference-cache")).unwrap();
        fs::write(
            root.join(format!("reference-cache/{}.json", data_sha256(&d))),
            serde_json::to_vec(&c).unwrap(),
        )
        .unwrap();
        let gate = crate::cycle::decide(NOW, 26127, 20261105, false);
        crate::reference::freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW - 5)
            .unwrap();
        (root, d)
    }
    #[test]
    fn exact_expansion_keeps_all_bankers_and_is_10_bets_yuan20() {
        let s = json!({"fusion":{"ranking":(1..=33).rev().collect::<Vec<_>>(),"blues":[12,4]}});
        let v = build(&s).unwrap();
        for (i, (_, dan_n, blue_n)) in SPECS.iter().enumerate() {
            let p = &v["plans"][i];
            let dan: Vec<u8> = serde_json::from_value(p["red_dan"].clone()).unwrap();
            let tuo: Vec<u8> = serde_json::from_value(p["red_tuo"].clone()).unwrap();
            assert_eq!(dan.len(), *dan_n);
            assert_eq!(tuo.len(), 5);
            assert_eq!(p["blues"].as_array().unwrap().len(), *blue_n);
            assert_eq!(p["red_dan"][0], 33);
            assert_eq!(p["blues"][0], 12);
            assert_eq!(p["bet_count"], 10);
            assert_eq!(p["cost"], 20);
            let ts = p["tickets"].as_array().unwrap();
            assert_eq!(ts.len(), 10);
            let mut unique = BTreeSet::new();
            let mut frequencies = std::collections::BTreeMap::new();
            for t in ts {
                let r: Vec<u8> = serde_json::from_value(t["red"].clone()).unwrap();
                assert_eq!(r.len(), 6);
                assert_eq!(r.iter().collect::<BTreeSet<_>>().len(), 6);
                assert!(dan.iter().all(|n| r.contains(n)));
                assert!(r.iter().all(|n| dan.contains(n) || tuo.contains(n)));
                assert!(unique.insert((r.clone(), t["blue"].as_u64().unwrap())));
                for n in r {
                    *frequencies.entry(n).or_insert(0) += 1;
                }
            }
            for n in &dan {
                assert_eq!(frequencies[n], 10);
            }
            for n in &tuo {
                assert_eq!(frequencies[n], if *dan_n == 2 { 8 } else { 6 });
            }
        }
        let mut bad = s;
        bad["fusion"]["ranking"][0] = bad["fusion"]["ranking"][1].clone();
        assert!(build(&bad).is_err());
    }
    #[test]
    fn seal_is_immutable_and_both_variants_settle_without_changing_parent() {
        let (root, mut d) = fixture();
        let path = root.join("reference/26128.json");
        let original = fs::read(&path).unwrap();
        let parent = crate::reference::original(&root, &d, "26128")
            .unwrap()
            .unwrap();
        let preview = score(&build(&parent["payload"]["reference_pools"]).unwrap(), None).unwrap();
        assert!(preview["plans"][0]["ticket_results"][0]["red_hits"].is_null());
        assert!(!root.join("fusion-dantuo").exists());
        let gate = crate::cycle::decide(NOW, 26127, 20261105, false);
        freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW).unwrap();
        let bytes = artifact(&root, &d, "26128").unwrap();
        assert_eq!(
            freeze_with_clock(&root, &d, "26128", gate.deadline_unix, || NOW + 1).unwrap()
                ["status"],
            "already_frozen"
        );
        assert_eq!(artifact(&root, &d, "26128").unwrap(), bytes);
        assert_eq!(observations(&root, &d)["pending"], 1);
        assert!(observations(&root, &d)["cells"][0]["fixed_prize_total"].is_null());
        let p = &preview["plans"][0];
        let red: Vec<u8> = serde_json::from_value(p["tickets"][0]["red"].clone()).unwrap();
        d.push(
            Draw::from_parts(
                26128,
                20261108,
                red.try_into().unwrap(),
                p["blues"][0].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        assert_eq!(settle(&root, &d).settled, 1);
        assert_eq!(settle(&root, &d).settled, 0);
        let v = view(&root, &d, "26128").unwrap().unwrap();
        assert!(v["review_persisted"].as_bool().unwrap());
        assert_eq!(v["plans"][0]["summary"]["max_ticket_red_hits"], 6);
        assert_eq!(v["plans"][0]["dan_hits"], 2);
        assert_eq!(observations(&root, &d)["n"], 1);
        assert_eq!(fs::read(path).unwrap(), original);
        assert_eq!(artifact(&root, &d, "26128").unwrap(), bytes);
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn deadline_race_stale_targets_and_clock_reversal_never_publish() {
        let (root, d) = fixture();
        let g = crate::cycle::decide(NOW, 26127, 20261105, false);
        assert!(freeze_with_clock(&root, &d, "26127", g.deadline_unix, || NOW).is_err());
        assert!(
            freeze_with_clock(&root, &d, "26128", g.deadline_unix, || g.deadline_unix).is_err()
        );
        let calls = std::cell::Cell::new(0);
        assert!(freeze_with_clock(&root, &d, "26128", g.deadline_unix, || {
            let i = calls.get();
            calls.set(i + 1);
            if i < 2 {
                NOW
            } else {
                g.deadline_unix
            }
        })
        .is_err());
        assert!(!root.join("fusion-dantuo/26128.json").exists());
        calls.set(0);
        assert!(freeze_with_clock(&root, &d, "26128", g.deadline_unix, || {
            let i = calls.get();
            calls.set(i + 1);
            if i == 0 {
                NOW
            } else {
                NOW - 1
            }
        })
        .is_err());
        assert!(artifact(&root, &d, "../26128").is_err());
        assert!(!root.join("fusion-dantuo/26128.json").exists());
        let _ = fs::remove_dir_all(root);
    }
}
