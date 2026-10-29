//! Faithful external reference tickets and a fixed, bounded fusion observation.
//! The Python forecast is retained verbatim; historical scoring never reruns it.
use crate::{
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    eval::{prepare_forecast, Prepared},
    hash::sha256_hex,
    recommendations as rec,
    select::rank_desc,
    small_red,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "reference_observation";
const SIZES: [usize; 4] = [12, 14, 15, 16];
fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|c| c.is_ascii_digit())
}
fn valid_sha(v: &Value) -> bool {
    v.as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
}
pub fn policy() -> Value {
    json!({"version":1,"reference_model":"reference_original5","fusion_model":"reference_fused16","method":"small_recent90_reference_union10","small_recent_weight":0.9,"reference_support_weight":0.1,"reference_support":"uniform_over_original_ticket_red_union","red_sizes":SIZES,"automatic_promotion":false,"score_kind":"ranking_evidence_not_probability","note":"原池保留参考程序5注及其实际并集；融合池固定90%现有小池评分、10%参考5注红球并集均匀支持，最多16红。并集全6红与单注命中分开统计，尚未证明稳定优势。"})
}
fn validate_cache(cache: &Value, draws: &[Draw], at: u64) -> Result<(), String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    if cache["version"] != 1
        || cache["data_sha256"] != data_sha256(draws)
        || cache["data_cutoff"] != fmt_issue(last.issue)
        || cache["target_issue"] != fmt_issue(last.issue + 1)
    {
        return Err("参考结果的版本、输入历史或目标期已变化".into());
    }
    let generated = cache["generated_at_unix"]
        .as_u64()
        .ok_or("参考结果缺少生成时间")?;
    if generated > at || !valid_sha(&cache["engine_sha256"]) || !valid_sha(&cache["config_sha256"])
    {
        return Err("参考结果生成时间或源码/配置摘要无效".into());
    }
    let tickets = cache["tickets"].as_array().ok_or("参考结果缺少原始5注")?;
    if tickets.len() != 5 {
        return Err("参考结果必须包含原始5注".into());
    }
    let mut unique = BTreeSet::new();
    for ticket in tickets {
        let reds: Vec<u8> =
            serde_json::from_value(ticket["red"].clone()).map_err(|_| "参考票面红球无效")?;
        let mut sorted = reds.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let blue = ticket["blue"].as_u64().ok_or("参考票面蓝球无效")?;
        if reds.len() != 6
            || sorted.len() != 6
            || sorted[0] < 1
            || sorted[5] > 33
            || !(1..=16).contains(&blue)
        {
            return Err("参考票面号码越界或红球重复".into());
        }
        if !unique.insert((sorted, blue)) {
            return Err("参考原始5注存在重复整注".into());
        }
    }
    Ok(())
}
fn cache_text(root: &Path, draws: &[Draw]) -> Result<Option<String>, String> {
    let path = root.join(format!("reference-cache/{}.json", data_sha256(draws)));
    match fs::read_to_string(path) {
        Ok(text) => {
            if text.len() > 2_000_000 {
                return Err("参考结果文件过大".into());
            }
            Ok(Some(text))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("参考结果读取失败：{e}")),
    }
}
fn snapshot(p: &Prepared, cache: &Value) -> Value {
    let tickets = cache["tickets"].as_array().unwrap();
    let mut counts = [0usize; 33];
    let mut blues = BTreeSet::new();
    for ticket in tickets {
        for n in ticket["red"].as_array().unwrap() {
            counts[n.as_u64().unwrap() as usize - 1] += 1;
        }
        blues.insert(ticket["blue"].as_u64().unwrap());
    }
    let reds: Vec<_> = (1..=33).filter(|&n| counts[n - 1] > 0).collect();
    let base = small_red::scores(&p.forecast.fusion, &p.expert_red);
    let exposure: [f64; 33] = counts.map(|n| n as f64 / 30.);
    let support: [f64; 33] = counts.map(|n| if n > 0 { 1. / reds.len() as f64 } else { 0. });
    let scores: [f64; 33] = std::array::from_fn(|i| base[i] * 0.9 + support[i] * 0.1);
    let order = rank_desc(&scores);
    let base_order = rank_desc(&base);
    let pools:Vec<_> = SIZES.iter().map(|&k|json!({"id":format!("reference_fused_red{k}"),"red_count":k,"reds":order[..k].iter().map(|n|n+1).collect::<Vec<_>>(),"baseline":{"model":"small_recent","red_count":k,"reds":base_order[..k].iter().map(|n|n+1).collect::<Vec<_>>()}})).collect();
    json!({"version":1,"policy":policy(),"target_issue":cache["target_issue"],"data_cutoff":cache["data_cutoff"],"input_history_sha256":p.data_sha,
      "reference":{"model":"reference_original5","label":"参考代码 · 原始5注观察池","tickets":tickets,"reds":reds,"red_count":reds.len(),"blues":blues.into_iter().collect::<Vec<_>>(),"blue_count":tickets.iter().map(|t|t["blue"].as_u64().unwrap()).collect::<BTreeSet<_>>().len(),"baseline":{"model":"weighted","red_count":reds.len(),"reds":p.forecast.red_order[..reds.len()].iter().map(|n|n+1).collect::<Vec<_>>() }},
      "fusion":{"model":"reference_fused16","label":"现有模型 × 参考5注 · 16红以内融合池","scores":scores.to_vec(),"base_scores":base.to_vec(),"reference_exposure":exposure.to_vec(),"reference_support":support.to_vec(),"ranking":order.map(|n|n+1).to_vec(),"blues":p.forecast.blue_order[..2].iter().map(|n|n+1).collect::<Vec<_>>(),"pools":pools}})
}
fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if !valid_issue(issue) {
        return Err("期号需要5位数字".into());
    }
    let path = root.join(format!("reference/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let e = rec::read_checked(&path)?;
    let p = &e["payload"];
    if p["kind"] != KIND {
        return Err("参考观察档案类型无效".into());
    }
    rec::validate_archive(p, issue, draws)?;
    let published = p["published_at_unix"]
        .as_u64()
        .ok_or("参考观察缺少保存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("参考观察缺少截止时间")?;
    if published >= deadline || p["known_outcome_at_generation"] != false {
        return Err("参考观察保存时间或数据边界无效".into());
    }
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= p["data_cutoff"].as_str().unwrap_or(""))
        .cloned()
        .collect();
    let last = prefix.last().ok_or("参考观察缺少输入历史")?;
    if issue != fmt_issue(last.issue + 1) || issue[..2] != fmt_issue(last.issue)[..2] {
        return Err("参考观察目标不是紧邻下一期".into());
    }
    let text = p["reference_cache_json"]
        .as_str()
        .ok_or("参考观察缺少原始结果")?;
    if p["reference_cache_sha256"] != sha256_hex(text.as_bytes()) {
        return Err("参考原始结果摘要不一致".into());
    }
    let cache: Value = serde_json::from_str(text).map_err(|_| "参考原始结果JSON无效")?;
    validate_cache(&cache, &prefix, published)?;
    if p["reference_pools"] != snapshot(&prepare_forecast(&prefix)?, &cache) {
        return Err("参考观察票面、融合评分或排序与保存证据不一致".into());
    }
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    if rec::saved_after_draw(Some(published), draw) {
        return Err("参考观察保存晚于开奖，拒绝计入".into());
    }
    if let Some(parent) = p["linked_daily_sha256"].as_str() {
        let daily = rec::read_checked(&root.join(format!("daily/{issue}.json")))?;
        if daily["sha256"] != parent {
            return Err("参考观察绑定的每日原档摘要变化".into());
        }
    }
    Ok(Some(e))
}
fn metadata(e: &Value) -> Value {
    let p = &e["payload"];
    let cache: Value = serde_json::from_str(p["reference_cache_json"].as_str().unwrap()).unwrap();
    json!({"issue":p["issue"],"source":"reference","raw_file":format!("reference/{}.json",p["issue"].as_str().unwrap()),"archive_sha256":e["sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"linked_daily_sha256":p["linked_daily_sha256"],"engine_sha256":cache["engine_sha256"],"config_sha256":cache["config_sha256"],"reference_cache_sha256":p["reference_cache_sha256"],"generated_at_unix":cache["generated_at_unix"],"independent_timestamp_verified":false})
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
    if let Some(old) = checked(root, draws, issue)? {
        let mut m = metadata(&old);
        m["status"] = json!("already_frozen");
        return Ok(m);
    }
    let started = clock();
    if started >= deadline {
        return Err("已到参考观察封存截止，禁止补发".into());
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    if issue != fmt_issue(last.issue + 1) || issue[..2] != fmt_issue(last.issue)[..2] {
        return Err("参考观察仅允许紧邻下一期，跨年需核对".into());
    }
    let text = cache_text(root, draws)?.ok_or("本期参考代码仍待运算，不能封存旧期结果")?;
    let cache: Value = serde_json::from_str(&text).map_err(|_| "参考结果JSON无效")?;
    validate_cache(&cache, draws, started)?;
    let prepared = prepare_forecast(draws)?;
    let pools = snapshot(&prepared, &cache);
    let parent = if root.join(format!("daily/{issue}.json")).exists() {
        let daily = rec::read_checked(&root.join(format!("daily/{issue}.json")))?;
        rec::validate_archive(&daily["payload"], issue, draws)?;
        Some(daily["sha256"].clone())
    } else {
        None
    };
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("参考观察计算越过截止或时钟倒退，拒绝封存".into());
    }
    let mut p = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    p["linked_daily_sha256"] = json!(parent);
    p["reference_cache_json"] = json!(text);
    p["reference_cache_sha256"] = json!(sha256_hex(text.as_bytes()));
    p["reference_pools"] = pools;
    let (_, created) = rec::write_new_before(
        &root.join(format!("reference/{issue}.json")),
        p,
        Some(deadline),
        &clock,
    )?;
    let saved = checked(root, draws, issue)?.ok_or("参考观察写入后未找到原档")?;
    let mut m = metadata(&saved);
    m["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(m)
}
fn score_pool(pool: &mut Value, draw: Option<&Draw>) {
    let reds: Vec<u8> = serde_json::from_value(pool["reds"].clone()).unwrap();
    let hits = draw.map(|d| d.red.iter().filter(|n| reds.contains(n)).count());
    pool["red_hits"] = json!(hits);
    pool["full_red"] = json!(hits.map(|h| h == 6));
    pool["missed_reds"] = json!(draw.map(|d| d
        .red
        .iter()
        .filter(|n| !reds.contains(n))
        .copied()
        .collect::<Vec<_>>()));
    if let Some(blues) = pool["blues"].as_array() {
        pool["blue_hit"] = json!(draw.map(|d| blues.contains(&json!(d.blue))));
    }
}
fn review_snapshot(snapshot: &Value, draw: Option<&Draw>) -> Value {
    let mut v = snapshot.clone();
    score_pool(&mut v["reference"], draw);
    score_pool(&mut v["reference"]["baseline"], draw);
    let results:Vec<_>=v["reference"]["tickets"].as_array().unwrap().iter().map(|t|{
        let hits=draw.map(|d|d.red.iter().filter(|n|t["red"].as_array().unwrap().contains(&json!(n))).count());
        json!({"red":t["red"],"blue":t["blue"],"red_hits":hits,"blue_hit":draw.map(|d|t["blue"]==d.blue)})
    }).collect();
    v["reference"]["best_ticket_red_hits"] = json!(draw.map(|_| results
        .iter()
        .filter_map(|t| t["red_hits"].as_u64())
        .max()
        .unwrap_or(0)));
    v["reference"]["ticket_results"] = json!(results);
    let blue_hit = draw.map(|d| {
        v["fusion"]["blues"]
            .as_array()
            .unwrap()
            .contains(&json!(d.blue))
    });
    v["fusion"]["blue_hit"] = json!(blue_hit);
    for pool in v["fusion"]["pools"].as_array_mut().unwrap() {
        score_pool(pool, draw);
        score_pool(&mut pool["baseline"], draw);
        pool["blue_hit"] = json!(blue_hit);
    }
    v
}
fn persisted(root: &Path, e: &Value, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = e["payload"]["issue"].as_str().unwrap();
    let path = root.join(format!("reviews/{issue}/reference.json"));
    if !path.exists() {
        return Ok(None);
    }
    let review = rec::read_checked(&path)?;
    let p = &review["payload"];
    let draw = draw.ok_or("参考观察已有复盘但当前缺少开奖")?;
    if p["kind"] != "reference_observation_review"
        || p["archive_sha256"] != e["sha256"]
        || p["outcome_sha256"] != data_sha256(std::slice::from_ref(draw))
        || p["reference_pools"] != review_snapshot(&e["payload"]["reference_pools"], Some(draw))
    {
        return Err("参考观察复盘与原档或开奖结果不一致".into());
    }
    Ok(Some(review))
}
pub fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(e) = checked(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let saved = persisted(root, &e, draw)?;
    let mut m = metadata(&e);
    m["reference_pools"] = review_snapshot(&e["payload"]["reference_pools"], draw);
    m["review_persisted"] = json!(saved.is_some());
    m["pending"] = json!(draw.is_none());
    if saved.is_some() {
        m["review_file"] = json!(format!("reviews/{issue}/reference.json"));
    }
    Ok(Some(m))
}
fn issue_set(root: &Path) -> Result<BTreeSet<String>, String> {
    let mut issues = BTreeSet::new();
    for folder in ["daily", "reference"] {
        match fs::read_dir(root.join(folder)) {
            Ok(entries) => {
                for entry in entries {
                    let p = entry.map_err(|e| e.to_string())?.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Some(issue) = p
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .filter(|s| valid_issue(s))
                        {
                            issues.insert(issue.to_string());
                        }
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(issues)
}
pub fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut result = rec::Settlement {
        settled: 0,
        errors: Vec::new(),
    };
    let issues = match issue_set(root) {
        Ok(v) => v,
        Err(e) => {
            result.errors.push(e);
            return result;
        }
    };
    for issue in issues {
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let outcome = (|| -> Result<usize, String> {
            let Some(e) = checked(root, draws, &issue)? else {
                return Ok(0);
            };
            if persisted(root, &e, Some(draw))?.is_some() {
                return Ok(0);
            }
            let payload = json!({"kind":"reference_observation_review","schema_version":1,"issue":issue,"archive_sha256":e["sha256"],"outcome_sha256":data_sha256(std::slice::from_ref(draw)),"reviewed_at_unix":rec::now_unix(),"draw":{"issue":issue,"date":fmt_date(draw.date),"red":draw.red,"blue":draw.blue},"reference_pools":review_snapshot(&e["payload"]["reference_pools"],Some(draw))});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/reference.json")),
                payload,
                None,
                rec::now_unix,
            )?;
            persisted(root, &e, Some(draw))?;
            Ok(usize::from(created))
        })();
        match outcome {
            Ok(n) => result.settled += n,
            Err(e) => result.errors.push(format!("{issue}/reference：{e}")),
        }
    }
    result
}
fn worker_job(root: &Path, data_sha: &str) -> Value {
    let Some(parent) = root.parent() else {
        return Value::Null;
    };
    let Ok(bytes) = fs::read(parent.join("reference-runtime/work/status.json")) else {
        return Value::Null;
    };
    if bytes.len() > 64_000 {
        return Value::Null;
    }
    let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
        return Value::Null;
    };
    if v["data_sha256"] != data_sha {
        return Value::Null;
    }
    let status = match v["status"].as_str() {
        Some("running") => "running",
        Some("error") => "failed",
        Some("ready" | "cached") => "complete",
        _ => return Value::Null,
    };
    json!({"status":status,"started_at_unix":v["started_at_unix"].as_u64(),"finished_at_unix":v["finished_at_unix"].as_u64(),"error":if status=="failed"{Some("参考运算失败，等待后台任务重试；未使用旧期结果") }else{None}})
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue + 1);
    let mut errors = Vec::new();
    let (mut n, mut pending, mut missing) = (0usize, 0usize, 0usize);
    let (mut full, mut hits, mut sizes, mut gains, mut losses) = (
        [0usize; 5],
        [0usize; 5],
        [0usize; 5],
        [0usize; 5],
        [0usize; 5],
    );
    let mut periods = Vec::new();
    for past in issue_set(root)? {
        match view(root, draws, &past) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    let s = &v["reference_pools"];
                    let pools: Vec<&Value> = std::iter::once(&s["reference"])
                        .chain(s["fusion"]["pools"].as_array().unwrap())
                        .collect();
                    for (i, p) in pools.iter().enumerate() {
                        let h = p["red_hits"].as_u64().unwrap() as usize;
                        let b = p["baseline"]["red_hits"].as_u64().unwrap() as usize;
                        full[i] += usize::from(h == 6);
                        hits[i] += h;
                        sizes[i] += p["red_count"].as_u64().unwrap() as usize;
                        gains[i] += usize::from(h == 6 && b < 6);
                        losses[i] += usize::from(h < 6 && b == 6);
                    }
                }
                periods.push(v);
            }
            Ok(None) => missing += 1,
            Err(e) => errors.push(format!("{past}：{e}")),
        }
    }
    let cells:Vec<_>=(0..5).map(|i|json!({"model":if i==0{"reference_original5"}else{"reference_fused16"},"label":if i==0{"参考原始5注并集".to_string()}else{format!("参考融合{}红",SIZES[i-1])},"red_count":if i==0{None}else{Some(SIZES[i-1])},"n":n,"pending":pending,"excluded_missing_pair":missing,"full_red_periods":full[i],"full_red_rate":(n>0).then(||full[i]as f64/n as f64),"mean_red_count":(n>0).then(||sizes[i]as f64/n as f64),"mean_missed_reds":(n>0).then(||6.-hits[i]as f64/n as f64),"paired_gains":gains[i],"paired_losses":losses[i],"baseline_model":if i==0{"weighted"}else{"small_recent"}})).collect();
    let (mut status, mut reason, mut next, mut archive) = (
        "waiting",
        "等待参考代码按当前历史完成原始5注运算".to_string(),
        Value::Null,
        Value::Null,
    );
    let preview = (|| -> Result<(), String> {
        if let Some(e) = checked(root, draws, &issue)? {
            next = e["payload"]["reference_pools"].clone();
            archive = metadata(&e);
            status = "frozen";
            reason = "两个参考观察池已按首次结果封存".into();
            return Ok(());
        }
        if let Some(text) = cache_text(root, draws)? {
            let cache: Value = serde_json::from_str(&text).map_err(|_| "参考结果JSON无效")?;
            validate_cache(&cache, draws, rec::now_unix())?;
            next = snapshot(&prepare_forecast(draws)?, &cache);
            status = "ready";
            reason = "参考原始5注及融合小池已就绪，等待截止前封存".into();
        }
        Ok(())
    })();
    if let Err(e) = preview {
        status = "error";
        reason = e.clone();
        errors.push(format!("{issue}：{e}"));
    }
    let job = worker_job(root, &data_sha256(draws));
    if status == "waiting" {
        if job["status"] == "running" {
            status = "running";
            reason = "参考代码正在按当前历史计算原始5注".into();
        } else if job["status"] == "failed" {
            status = "error";
            reason = "本期参考运算失败，等待后台任务重试；未使用旧期结果".into();
        }
    }
    Ok(
        json!({"version":1,"kind":"reference_pool_observations","data_sha256":data_sha256(draws),"data_cutoff":fmt_issue(last.issue),"target":issue,"status":status,"reason":reason,"job":job,"next":next,"next_archive":archive,"policy":policy(),"actual":{"actual_recommendation":true,"independent_timestamp_verified":false,"cells":cells,"periods":periods,"errors":errors,"note":"只统计真实封存原档，缺档与待开奖不计失败；原5注并集按相同红球数量与原榜比较，融合池与同规格small_recent比较。"}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ssq-reference-{}-{}",
                    std::process::id(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
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
    fn history() -> Vec<Draw> {
        crate::draw::synth_draws(220, 9)
    }
    fn cache(d: &[Draw]) -> Value {
        let tickets: Vec<_> = (0..5)
            .map(|i| json!({"red":(0..6).map(|j|i*5+j+1).collect::<Vec<_>>(),"blue":i+1}))
            .collect();
        json!({"version":1,"data_sha256":data_sha256(d),"data_cutoff":fmt_issue(d.last().unwrap().issue),"target_issue":fmt_issue(d.last().unwrap().issue+1),"generated_at_unix":10,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":tickets,"metadata":{"seed":2026,"backtest_periods":100}})
    }
    fn put_cache(root: &Path, d: &[Draw], v: &Value) {
        fs::create_dir_all(root.join("reference-cache")).unwrap();
        fs::write(
            root.join(format!("reference-cache/{}.json", data_sha256(d))),
            serde_json::to_vec(v).unwrap(),
        )
        .unwrap();
    }
    fn old_daily(root: &Path, d: &[Draw]) {
        let p = prepare_forecast(d).unwrap();
        let issue = fmt_issue(p.cutoff_issue + 1);
        let mut payload = rec::base_payload(&p, &issue, "daily", 5, Some(100));
        payload["strategies"] = json!({});
        rec::write_new_before(
            &root.join(format!("daily/{issue}.json")),
            payload,
            None,
            || 5,
        )
        .unwrap();
    }
    #[test]
    fn reference_preserves_exact_tickets_and_daily_with_bounded_nested_fusion() {
        let tmp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let c = cache(&d);
        put_cache(&tmp.0, &d, &c);
        old_daily(&tmp.0, &d);
        let daily = fs::read(tmp.0.join(format!("daily/{issue}.json"))).unwrap();
        let preview = report(&tmp.0, &d).unwrap();
        assert_eq!(preview["status"], "ready");
        assert_eq!(preview["actual"]["cells"][0]["excluded_missing_pair"], 1);
        let frozen = freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).unwrap();
        assert_eq!(frozen["status"], "frozen");
        let raw = fs::read(tmp.0.join(format!("reference/{issue}.json"))).unwrap();
        let v = view(&tmp.0, &d, &issue).unwrap().unwrap();
        let s = &v["reference_pools"];
        assert_eq!(s["reference"]["tickets"], c["tickets"]);
        assert_eq!(s["reference"]["red_count"], 26);
        assert_eq!(s["reference"]["baseline"]["red_count"], 26);
        // Repeated appearances remain evidence but do not overweight the union channel.
        assert_ne!(
            s["fusion"]["reference_exposure"][0],
            s["fusion"]["reference_exposure"][5]
        );
        assert_eq!(
            s["fusion"]["reference_support"][0],
            s["fusion"]["reference_support"][5]
        );
        for i in 0..33 {
            let support = if i < 26 { 1. / 26. } else { 0. };
            assert_eq!(s["fusion"]["reference_support"][i], json!(support));
            let base = s["fusion"]["base_scores"][i].as_f64().unwrap();
            assert_eq!(s["fusion"]["scores"][i], json!(base * 0.9 + support * 0.1));
        }
        assert!(s["reference"]["full_red"].is_null());
        assert!(s["reference"]["best_ticket_red_hits"].is_null());
        assert!(s["reference"]["ticket_results"][0]["red_hits"].is_null());
        let mut last = BTreeSet::new();
        for p in s["fusion"]["pools"].as_array().unwrap() {
            let set: BTreeSet<_> = p["reds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap())
                .collect();
            assert!(last.is_subset(&set));
            assert_eq!(set.len(), p["red_count"].as_u64().unwrap() as usize);
            assert!(set.len() <= 16);
            last = set;
            assert!(p["red_hits"].is_null());
        }
        assert!(!tmp.0.join("reviews").exists());
        let obs = report(&tmp.0, &d).unwrap();
        assert_eq!(obs["status"], "frozen");
        for c in obs["actual"]["cells"].as_array().unwrap() {
            assert_eq!(c["n"], 0);
            assert_eq!(c["pending"], 1);
            assert_eq!(c["excluded_missing_pair"], 0);
            assert!(c["full_red_rate"].is_null());
        }
        fs::remove_file(
            tmp.0
                .join(format!("reference-cache/{}.json", data_sha256(&d))),
        )
        .unwrap();
        assert_eq!(
            freeze_with_clock(&tmp.0, &d, &issue, 100, || 200).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(
            fs::read(tmp.0.join(format!("reference/{issue}.json"))).unwrap(),
            raw
        );
        assert_eq!(
            fs::read(tmp.0.join(format!("daily/{issue}.json"))).unwrap(),
            daily
        );
        let mut changed = d.clone();
        changed[0] = crate::draw::synth_draws(220, 10)[0].clone();
        assert!(view(&tmp.0, &changed, &issue).is_err());
        assert!(view(&tmp.0, &d, "../bad").is_err());
    }
    #[test]
    fn stale_cache_invalid_tickets_and_deadline_crossing_never_create_archive() {
        let tmp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).is_err());
        assert_eq!(report(&tmp.0, &d).unwrap()["status"], "waiting");
        let mut c = cache(&d);
        c["data_sha256"] = json!("stale");
        put_cache(&tmp.0, &d, &c);
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).is_err());
        assert_eq!(report(&tmp.0, &d).unwrap()["status"], "error");
        c = cache(&d);
        c["tickets"][0]["red"] = json!([1, 1, 2, 3, 4, 5]);
        put_cache(&tmp.0, &d, &c);
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).is_err());
        c = cache(&d);
        c["generated_at_unix"] = json!(21);
        put_cache(&tmp.0, &d, &c);
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).is_err());
        put_cache(&tmp.0, &d, &cache(&d));
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || 100).is_err());
        let calls = Cell::new(0);
        assert!(freeze_with_clock(&tmp.0, &d, &issue, 100, || {
            let n = calls.get();
            calls.set(n + 1);
            [20, 21, 100][n.min(2)]
        })
        .is_err());
        assert!(!tmp.0.join(format!("reference/{issue}.json")).exists());
        assert!(freeze_with_clock(
            &tmp.0,
            &d,
            &fmt_issue(d.last().unwrap().issue + 2),
            100,
            || 20
        )
        .is_err());
    }
    #[test]
    fn scoring_uses_saved_union_and_tickets_and_settlement_is_immutable() {
        let tmp = Temp::new();
        let mut d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let c = cache(&d);
        put_cache(&tmp.0, &d, &c);
        freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).unwrap();
        let original = fs::read(tmp.0.join(format!("reference/{issue}.json"))).unwrap();
        let draw = Draw::from_parts(
            d.last().unwrap().issue + 1,
            20031201,
            [1, 6, 11, 16, 21, 26],
            3,
        )
        .unwrap();
        d.push(draw);
        let v = view(&tmp.0, &d, &issue).unwrap().unwrap();
        let r = &v["reference_pools"]["reference"];
        assert_eq!(r["red_hits"], 6);
        assert_eq!(r["full_red"], true);
        assert_eq!(r["best_ticket_red_hits"], 2);
        assert_eq!(r["blue_hit"], true);
        assert!(!tmp.0.join("reviews").exists());
        let done = settle(&tmp.0, &d);
        assert_eq!(done.errors, Vec::<String>::new());
        assert_eq!(done.settled, 1);
        let path = tmp.0.join(format!("reviews/{issue}/reference.json"));
        let bytes = fs::read(&path).unwrap();
        assert_eq!(settle(&tmp.0, &d).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert_eq!(
            fs::read(tmp.0.join(format!("reference/{issue}.json"))).unwrap(),
            original
        );
        let actual = report(&tmp.0, &d).unwrap();
        let cell = &actual["actual"]["cells"][0];
        assert_eq!(cell["n"], 1);
        assert_eq!(cell["pending"], 0);
        assert_eq!(cell["full_red_periods"], 1);
        assert_eq!(cell["full_red_rate"], 1.0);
        d.last_mut().unwrap().blue = 4;
        assert!(view(&tmp.0, &d, &issue).is_err());
        let changed = settle(&tmp.0, &d);
        assert_eq!(changed.settled, 0);
        assert_eq!(changed.errors.len(), 1);
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
    #[test]
    fn rehashed_fusion_or_reference_evidence_tampering_is_rejected() {
        let tmp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        put_cache(&tmp.0, &d, &cache(&d));
        freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).unwrap();
        let path = tmp.0.join(format!("reference/{issue}.json"));
        let original = fs::read(&path).unwrap();
        for field in ["fusion", "cache"] {
            let mut e: Value = serde_json::from_slice(&original).unwrap();
            if field == "fusion" {
                e["payload"]["reference_pools"]["fusion"]["scores"][0] = json!(0.9);
            } else {
                e["payload"]["reference_cache_json"] = json!("{}");
            }
            e["sha256"] = json!(sha256_hex(&serde_json::to_vec(&e["payload"]).unwrap()));
            fs::write(&path, serde_json::to_vec(&e).unwrap()).unwrap();
            assert!(view(&tmp.0, &d, &issue).is_err());
        }
    }
    #[test]
    fn concurrent_freeze_keeps_one_complete_first_record() {
        let tmp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        put_cache(&tmp.0, &d, &cache(&d));
        let results = std::thread::scope(|s| {
            let a = s.spawn(|| freeze_with_clock(&tmp.0, &d, &issue, 100, || 20).unwrap());
            let b = s.spawn(|| freeze_with_clock(&tmp.0, &d, &issue, 100, || 21).unwrap());
            [a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(results[0]["archive_sha256"], results[1]["archive_sha256"]);
        assert!(view(&tmp.0, &d, &issue).unwrap().is_some());
    }
    #[test]
    fn worker_status_is_bound_to_current_history_and_redacts_diagnostics() {
        let tmp = Temp::new();
        let d = history();
        let root = tmp.0.join("recommendations");
        let path = tmp.0.join("reference-runtime/work/status.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            json!({"data_sha256":data_sha256(&d),"status":"running","started_at_unix":10})
                .to_string(),
        )
        .unwrap();
        let r = report(&root, &d).unwrap();
        assert_eq!(r["status"], "running");
        assert_eq!(r["job"]["started_at_unix"], 10);
        assert!(r["next"].is_null());
        fs::write(&path,json!({"data_sha256":data_sha256(&d),"status":"error","error":"private path and stack trace"}).to_string()).unwrap();
        let r = report(&root, &d).unwrap();
        assert_eq!(r["status"], "error");
        assert_eq!(r["job"]["status"], "failed");
        assert!(!r.to_string().contains("private path"));
        fs::write(
            &path,
            json!({"data_sha256":"stale","status":"running"}).to_string(),
        )
        .unwrap();
        let r = report(&root, &d).unwrap();
        assert_eq!(r["status"], "waiting");
        assert!(r["job"].is_null());
    }
}
