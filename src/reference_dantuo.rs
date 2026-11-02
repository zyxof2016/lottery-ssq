//! Preserve and expand the supplied engine's original budget plan; never infer it from Final5.
use crate::{
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "reference_dantuo";
fn issue_ok(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|c| c.is_ascii_digit())
}
fn nums(plan: &Value, key: &str, high: u8) -> Result<Vec<u8>, String> {
    let ns: Vec<u8> = serde_json::from_value(plan[key].clone())
        .map_err(|_| format!("参考胆拖{key}不是有效号码数组"))?;
    if ns.iter().collect::<BTreeSet<_>>().len() != ns.len()
        || ns.iter().any(|n| *n == 0 || *n > high)
    {
        return Err("参考胆拖号码重复或越界".into());
    }
    Ok(ns)
}
fn expand(plan: &Value) -> Result<Value, String> {
    let dan = nums(plan, "red_dan", 33)?;
    let tuo = nums(plan, "red_tuo", 33)?;
    let blue = nums(plan, "blue", 16)?;
    if dan.len() > 5
        || tuo.len() < 6 - dan.len()
        || dan.iter().any(|n| tuo.contains(n))
        || blue.is_empty()
    {
        return Err("参考胆拖胆拖结构无效".into());
    }
    let bets = crate::stats::binom(tuo.len() as u64, (6 - dan.len()) as u64) * blue.len() as u64;
    if bets == 0
        || bets > 10
        || plan["unit_price"].as_u64() != Some(2)
        || plan["bet_count"].as_u64() != Some(bets)
        || plan["amount"].as_u64() != Some(bets * 2)
        || !plan["max_amount"]
            .as_u64()
            .is_some_and(|n| n >= bets * 2 && n <= 20)
    {
        return Err("参考胆拖展开注数或预算与原报告不一致".into());
    }
    let mut tickets = Vec::new();
    for combo in crate::plan::combinations(&tuo, 6 - dan.len())? {
        let mut red = dan.clone();
        red.extend(combo);
        red.sort_unstable();
        for b in &blue {
            tickets.push(json!({"red":red,"blue":b}));
        }
    }
    let mut reds = dan.clone();
    reds.extend(&tuo);
    reds.sort_unstable();
    Ok(
        json!({"id":"reference_original_dantuo","title":if dan.is_empty(){"参考原始预算复式"}else{"参考原始胆拖"},"mode":if dan.is_empty(){"compound"}else{"dantuo"},"original_plan":plan,"red_dan":dan,"red_tuo":tuo,"reds":reds,"blues":blue,"bet_count":bets,"cost":bets*2,"tickets":tickets}),
    )
}
fn build(text: &str, reference_text: &str, published: u64) -> Result<Value, String> {
    let cache: Value = serde_json::from_str(text).map_err(|_| "参考胆拖缓存JSON无效")?;
    let parent: Value = serde_json::from_str(reference_text).map_err(|_| "原始5注缓存JSON无效")?;
    if cache["version"] != 1
        || cache["reference_cache_sha256"] != sha256_hex(reference_text.as_bytes())
        || [
            "data_sha256",
            "data_cutoff",
            "target_issue",
            "engine_sha256",
            "config_sha256",
            "generated_at_unix",
        ]
        .iter()
        .any(|k| cache[k] != parent[k])
    {
        return Err("参考胆拖与原始5注输入、期号或引擎不一致".into());
    }
    let generated = cache["generated_at_unix"]
        .as_u64()
        .ok_or("缺少原始生成时间")?;
    let exported = cache["exported_at_unix"]
        .as_u64()
        .ok_or("缺少胆拖导出时间")?;
    if generated == 0 || exported < generated || exported > published {
        return Err("参考胆拖生成或导出时间无效".into());
    }
    let source = cache["source_forecast_json"]
        .as_str()
        .ok_or("参考胆拖缺少原始报告")?;
    if cache["source_forecast_sha256"] != sha256_hex(source.as_bytes()) {
        return Err("参考胆拖原始报告摘要不一致".into());
    }
    let forecast: Value = serde_json::from_str(source).map_err(|_| "参考胆拖原始报告无效")?;
    let tickets: Vec<_> = forecast["tickets"]
        .as_array()
        .ok_or("原始报告缺少5注")?
        .iter()
        .map(|t| json!({"red":t["red"],"blue":t["blue"]}))
        .collect();
    if forecast["source_period"]
        != format!(
            "20{}",
            parent["data_cutoff"].as_str().ok_or("原始截止期无效")?
        )
        || forecast["target_period"]
            != format!(
                "20{}",
                parent["target_issue"].as_str().ok_or("原始目标期无效")?
            )
        || json!(tickets) != parent["tickets"]
        || cache["dantuo_plan"] != forecast["metadata"]["dantuo_plan"]
    {
        return Err("参考胆拖不属于已保存5注的同一次原始报告".into());
    }
    expand(&cache["dantuo_plan"])
}
fn score(plan: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let r = rec::review_strategy(
        &json!({"title":plan["title"],"mode":plan["mode"],"structural":true,"tickets":plan["tickets"]}),
        draw,
    )?;
    let mut out = plan.clone();
    out["ticket_results"] = r["tickets"].clone();
    out["summary"] = r["summary"].clone();
    out["prize_rules_url"] = r["prize_rules_url"].clone();
    out["prize_note"] = r["explanation_limit"].clone();
    out["dan_hits"] = json!(draw.map(|d| plan["red_dan"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|n| d.red.iter().any(|r| json!(r) == **n))
        .count()));
    out["red_union_hits"] = json!(draw.map(|d| plan["reds"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|n| d.red.iter().any(|r| json!(r) == **n))
        .count()));
    Ok(out)
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !issue_ok(issue) {
        return Err("参考胆拖期号须为5位".into());
    }
    let path = root.join(format!("reference-dantuo/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    if p["kind"] != KIND {
        return Err("参考胆拖原档类型无效".into());
    }
    rec::validate_archive(p, issue, draws)?;
    let parent =
        crate::reference::original(root, draws, issue)?.ok_or("参考胆拖缺少绑定的5注原档")?;
    let text = p["dantuo_cache_json"]
        .as_str()
        .ok_or("参考胆拖缺少原始缓存")?;
    if p["reference_archive_sha256"] != parent["sha256"]
        || p["dantuo_cache_sha256"] != sha256_hex(text.as_bytes())
        || p["plan"]
            != build(
                text,
                parent["payload"]["reference_cache_json"].as_str().unwrap(),
                p["published_at_unix"].as_u64().unwrap(),
            )?
    {
        return Err("参考胆拖原档与来源或展开票面不一致".into());
    }
    if rec::saved_after_draw(
        p["published_at_unix"].as_u64(),
        draws.iter().find(|d| fmt_issue(d.issue) == issue),
    ) {
        return Err("参考胆拖保存晚于开奖，拒绝计入实际推荐".into());
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"issue":p["issue"],"source":"reference-dantuo","raw_file":format!("reference-dantuo/{}.json",p["issue"].as_str().unwrap()),"archive_sha256":e["sha256"],"reference_archive_sha256":p["reference_archive_sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"independent_timestamp_verified":false})
}
fn cache_text(root: &Path, draws: &[Draw]) -> Result<Option<String>, String> {
    match fs::read_to_string(root.join(format!(
        "reference-dantuo-cache/{}.json",
        data_sha256(draws)
    ))) {
        Ok(s) if s.len() <= 2_000_000 => Ok(Some(s)),
        Ok(_) => Err("参考胆拖缓存过大".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
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
    let last = draws.last().ok_or("没有开奖历史")?;
    if issue != fmt_issue(last.issue + 1) || issue[..2] != fmt_issue(last.issue)[..2] {
        return Err("参考胆拖只允许当前紧邻下一期，跨年须核对".into());
    }
    let started = clock();
    if started >= deadline {
        return Err("已过参考胆拖封存截止".into());
    }
    let Some(text) = cache_text(root, draws)? else {
        return Ok(json!({"status":"waiting","reason":"原始报告中的胆拖尚未导出，不另造号码"}));
    };
    let parent = crate::reference::original(root, draws, issue)?.ok_or("参考原始5注尚未封存")?;
    let plan = build(
        &text,
        parent["payload"]["reference_cache_json"].as_str().unwrap(),
        started,
    )?;
    let prepared = prepare_forecast(draws)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("参考胆拖展开越过截止或时钟倒退".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["reference_archive_sha256"] = parent["sha256"].clone();
    p["dantuo_cache_sha256"] = json!(sha256_hex(text.as_bytes()));
    p["dantuo_cache_json"] = json!(text);
    p["plan"] = plan;
    let (_, created) = rec::write_new_before(
        &root.join(format!("reference-dantuo/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let e = checked(root, draws, issue)?.ok_or("参考胆拖写入后原档缺失")?;
    let mut m = metadata(&e);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn persisted(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().unwrap();
    let path = root.join(format!("reviews/{issue}/reference-dantuo.json"));
    if !path.exists() {
        return Ok(None);
    }
    let r = rec::read_checked(&path)?;
    let d = draw.ok_or("参考胆拖已有复盘但缺少开奖结果")?;
    if r["payload"]["kind"] != "reference_dantuo_review"
        || r["payload"]["archive_sha256"] != e["sha256"]
        || r["payload"]["outcome_sha256"] != data_sha256(std::slice::from_ref(d))
        || r["payload"]["plan"] != score(&e["payload"]["plan"], Some(d))?
    {
        return Err("参考胆拖复盘与原档或开奖结果不一致".into());
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
    m["plan"] = score(&e["payload"]["plan"], draw)?;
    m["status"] = json!("frozen");
    m["pending"] = json!(draw.is_none());
    m["review_persisted"] = json!(saved.is_some());
    if saved.is_some() {
        m["review_file"] = json!(format!("reviews/{issue}/reference-dantuo.json"));
    }
    Ok(Some(m))
}
pub fn current(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖历史")?;
    let issue = fmt_issue(last.issue + 1);
    if let Some(v) = view(root, draws, &issue)? {
        return Ok(v);
    }
    let Some(text) = cache_text(root, draws)? else {
        return Ok(
            json!({"status":"waiting","reason":"等待导出同次参考报告的原始胆拖；不根据5注另算号码。"}),
        );
    };
    let parent = crate::reference::original(root, draws, &issue)?;
    let reference = if let Some(p) = parent {
        p["payload"]["reference_cache_json"]
            .as_str()
            .unwrap()
            .to_string()
    } else {
        fs::read_to_string(root.join(format!("reference-cache/{}.json", data_sha256(draws))))
            .map_err(|_| "缺少绑定的5注缓存")?
    };
    let cache: Value = serde_json::from_str(&reference).map_err(|_| "原始5注缓存JSON无效")?;
    crate::reference::validate_cache(&cache, draws, rec::now_unix())?;
    let plan = build(&text, &reference, rec::now_unix())?;
    Ok(json!({"status":"preview","actual_recommendation":false,"plan":score(&plan,None)?}))
}
fn issues(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for dir in ["daily", "reference", "reference-dantuo"] {
        if let Ok(entries) = fs::read_dir(root.join(dir)) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(s) = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .filter(|s| issue_ok(s))
                    {
                        out.insert(s.to_string());
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
            let p = json!({"kind":"reference_dantuo_review","schema_version":1,"issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(draw)),"reviewed_at_unix":rec::now_unix(),"draw":{"issue":issue,"date":fmt_date(draw.date),"red":draw.red,"blue":draw.blue},"plan":score(&e["payload"]["plan"],Some(draw))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/reference-dantuo.json")),
                p,
                None,
                rec::now_unix,
            )?;
            persisted(root, &e, Some(draw))?;
            Ok(usize::from(created))
        })();
        match result {
            Ok(n) => out.settled += n,
            Err(e) => out.errors.push(format!("{issue}/reference-dantuo：{e}")),
        }
    }
    out
}
pub fn observations(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending, mut missing, mut prizes, mut fixed, mut best, mut cost) =
        (0, 0, 0, 0, 0u64, 0u64, 0u64);
    let mut errors = Vec::new();
    for issue in issues(root) {
        match view(root, draws, &issue) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    let s = &v["plan"]["summary"];
                    prizes += usize::from(s["prize_tickets"].as_u64().unwrap_or(0) > 0);
                    fixed += s["fixed_prize"].as_u64().unwrap_or(0);
                    best += s["max_ticket_red_hits"].as_u64().unwrap_or(0);
                    cost += v["plan"]["cost"].as_u64().unwrap_or(0);
                }
            }
            Ok(None) => missing += 1,
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"n":n,"pending":pending,"excluded_missing":missing,"any_prize_periods":prizes,"any_prize_rate":(n>0).then(||prizes as f64/n as f64),"mean_best_red":(n>0).then(||best as f64/n as f64),"fixed_prize_total":if n>0{json!(fixed)}else{Value::Null},"cost_total":if n>0{json!(cost)}else{Value::Null},"errors":errors,"note":"只统计截止前实际封存的原始预算方案，按完整展开单注对奖；红胆命中和红球并集命中分别显示。"})
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> (Temp, Vec<Draw>, String) {
        let root = Temp(
            std::env::temp_dir().join(format!(
                "ssq-dantuo-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )),
        );
        let d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let cache = json!({"version":1,"data_sha256":data_sha256(&d),"data_cutoff":fmt_issue(d.last().unwrap().issue),"target_issue":issue,"generated_at_unix":5,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
        let text = serde_json::to_string(&cache).unwrap();
        fs::create_dir_all(root.0.join("reference-cache")).unwrap();
        fs::write(
            root.0
                .join(format!("reference-cache/{}.json", data_sha256(&d))),
            &text,
        )
        .unwrap();
        crate::reference::freeze_with_clock(&root.0, &d, &issue, 100, || 10).unwrap();
        let plan = json!({"red_dan":[14,24],"red_tuo":[2,20,25,27,30],"blue":[4,5],"bet_count":10,"amount":20,"unit_price":2,"max_amount":20,"requested_layout":"2胆 + 5拖 + 2蓝","rule":"original"});
        let forecast=serde_json::to_string(&json!({"source_period":format!("20{}",cache["data_cutoff"].as_str().unwrap()),"target_period":format!("20{issue}"),"tickets":cache["tickets"],"metadata":{"dantuo_plan":plan}})).unwrap();
        let mut source = cache.clone();
        source["reference_cache_sha256"] = json!(sha256_hex(text.as_bytes()));
        source["source_forecast_sha256"] = json!(sha256_hex(forecast.as_bytes()));
        source["source_forecast_json"] = json!(forecast);
        source["dantuo_plan"] = plan;
        source["exported_at_unix"] = json!(15);
        fs::create_dir_all(root.0.join("reference-dantuo-cache")).unwrap();
        fs::write(
            root.0
                .join(format!("reference-dantuo-cache/{}.json", data_sha256(&d))),
            serde_json::to_vec(&source).unwrap(),
        )
        .unwrap();
        (root, d, issue)
    }
    #[test]
    fn original_dantuo_expands_exact_combinations_and_preview_is_read_only() {
        let (t, d, issue) = fixture();
        let old = fs::read(t.0.join(format!("reference/{issue}.json"))).unwrap();
        let v = current(&t.0, &d).unwrap();
        assert_eq!(v["status"], "preview");
        assert_eq!(v["plan"]["bet_count"], 10);
        assert_eq!(v["plan"]["cost"], 20);
        assert!(!t.0.join("reference-dantuo").exists());
        assert!(!t.0.join("reviews").exists());
        let tickets = v["plan"]["tickets"].as_array().unwrap();
        let mut identities = BTreeSet::new();
        for ticket in tickets {
            let red: Vec<u8> = serde_json::from_value(ticket["red"].clone()).unwrap();
            assert_eq!(red.len(), 6);
            assert!(red.contains(&14) && red.contains(&24));
            assert_eq!(
                red.iter()
                    .filter(|n| [2, 20, 25, 27, 30].contains(n))
                    .count(),
                4
            );
            assert!(!red.contains(&1));
            identities.insert(serde_json::to_string(ticket).unwrap());
        }
        assert_eq!(identities.len(), 10);
        assert_eq!(tickets.iter().filter(|t| t["blue"] == 4).count(), 5);
        assert!(v["plan"]["ticket_results"][0]["red_hits"].is_null());
        assert!(v["plan"]["summary"]["fixed_prize"].is_null());
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        assert_eq!(
            fs::read(t.0.join(format!("reference/{issue}.json"))).unwrap(),
            old
        );
        assert_eq!(view(&t.0, &d, &issue).unwrap().unwrap()["plan"], v["plan"]);
        assert_eq!(observations(&t.0, &d)["pending"], 1);
        assert!(observations(&t.0, &d)["any_prize_rate"].is_null());
    }
    #[test]
    fn compound_layout_and_bad_budgets_or_overlapping_numbers() {
        let plan = json!({"red_dan":[],"red_tuo":[1,2,3,4,5,6,7],"blue":[9],"bet_count":7,"amount":14,"unit_price":2,"max_amount":20});
        let p = expand(&plan).unwrap();
        assert_eq!(p["mode"], "compound");
        assert_eq!(p["tickets"].as_array().unwrap().len(), 7);
        for (key, value) in [
            ("red_dan", json!([1])),
            ("red_tuo", json!([1, 2, 3, 4, 5, 6, 6])),
            ("blue", json!([0])),
            ("blue", json!([9, 9])),
            ("amount", json!(12)),
            ("bet_count", json!(8)),
            ("max_amount", json!(12)),
            ("red_dan", json!([1, 2, 3, 4, 5, 6])),
        ] {
            let mut bad = plan.clone();
            bad[key] = value;
            assert!(expand(&bad).is_err(), "{key}");
        }
        let mut over = plan.clone();
        over["blue"] = json!([1, 2]);
        over["bet_count"] = json!(14);
        over["amount"] = json!(28);
        assert!(expand(&over).is_err());
    }
    #[test]
    fn deadlines_parent_binding_immutability_and_no_historical_backfill() {
        let (t, d, issue) = fixture();
        assert!(freeze_with_clock(&t.0, &d, &issue, 20, || 20).is_err());
        let tick = std::cell::Cell::new(0);
        assert!(freeze_with_clock(&t.0, &d, &issue, 20, || {
            let i = tick.get();
            tick.set(i + 1);
            if i == 0 {
                15
            } else {
                20
            }
        })
        .is_err());
        assert!(!t.0.join("reference-dantuo").exists());
        let path =
            t.0.join(format!("reference-dantuo-cache/{}.json", data_sha256(&d)));
        let bytes = fs::read(&path).unwrap();
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["target_issue"] = json!("99999");
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(current(&t.0, &d).is_err());
        assert!(freeze_with_clock(&t.0, &d, &issue, 100, || 20).is_err());
        fs::write(&path, bytes).unwrap();
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        let archive = t.0.join(format!("reference-dantuo/{issue}.json"));
        let original = fs::read(&archive).unwrap();
        assert_eq!(
            freeze_with_clock(&t.0, &d, &issue, 100, || 200).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(fs::read(&archive).unwrap(), original);
        let mut changed = d.clone();
        changed[0].blue = changed[0].blue % 16 + 1;
        assert!(view(&t.0, &changed, &issue).is_err());
        let old_issue = fmt_issue(d.last().unwrap().issue);
        assert!(view(&t.0, &d, &old_issue).unwrap().is_none());
        assert!(freeze_with_clock(&t.0, &d, &old_issue, 100, || 20).is_err());
        let mut bad: Value = serde_json::from_slice(&original).unwrap();
        bad["payload"]["plan"]["tickets"][0]["blue"] = json!(16);
        bad["sha256"] = json!(sha256_hex(&serde_json::to_vec(&bad["payload"]).unwrap()));
        fs::write(&archive, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(view(&t.0, &d, &issue).is_err());
    }
    #[test]
    fn actual_expanded_ticket_scoring_and_immutable_settlement() {
        let (t, mut d, issue) = fixture();
        freeze_with_clock(&t.0, &d, &issue, 100, || 20).unwrap();
        let v = view(&t.0, &d, &issue).unwrap().unwrap();
        let ticket = &v["plan"]["tickets"][0];
        d.push(
            Draw::from_parts(
                issue.parse().unwrap(),
                20261029,
                serde_json::from_value(ticket["red"].clone()).unwrap(),
                4,
            )
            .unwrap(),
        );
        let r = view(&t.0, &d, &issue).unwrap().unwrap();
        assert_eq!(r["plan"]["dan_hits"], 2);
        assert_eq!(r["plan"]["red_union_hits"], 6);
        assert_eq!(r["plan"]["ticket_results"][0]["prize_level"], 1);
        assert_eq!(r["plan"]["ticket_results"][1]["prize_level"], 2);
        assert!(!t.0.join("reviews").exists());
        assert_eq!(settle(&t.0, &d).settled, 1);
        let path = t.0.join(format!("reviews/{issue}/reference-dantuo.json"));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(settle(&t.0, &d).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(observations(&t.0, &d)["n"], 1);
        d.last_mut().unwrap().blue = 16;
        assert!(!settle(&t.0, &d).errors.is_empty());
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
    #[test]
    fn concurrent_publication_keeps_one_original() {
        let (t, d, issue) = fixture();
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let (root, draws, target) = (t.0.clone(), d.clone(), issue.clone());
                std::thread::spawn(move || {
                    freeze_with_clock(&root, &draws, &target, 100, || 20).unwrap()
                })
            })
            .collect();
        let values: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(values[0]["archive_sha256"], values[1]["archive_sha256"]);
        assert_eq!(values.iter().filter(|v| v["status"] == "frozen").count(), 1);
    }
}
