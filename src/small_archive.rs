//! Independent, immutable pre-draw small-pool observations for already sealed issues.
use crate::{
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    eval::prepare_forecast,
    recommendations as rec, small_red,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path};
const KIND: &str = "small_red_observation";
fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|b| b.is_ascii_digit())
}
#[derive(Clone)]
struct Record {
    envelope: Value,
    source: &'static str,
    raw_file: String,
}
fn checked(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    source: &'static str,
) -> Result<Option<Record>, String> {
    let directory = if source == "daily" {
        "daily"
    } else {
        "small-red"
    };
    let raw_file = format!("{directory}/{issue}.json");
    let path = root.join(&raw_file);
    if !path.exists() {
        return Ok(None);
    }
    let envelope = rec::read_checked(&path)?;
    let p = &envelope["payload"];
    if p["kind"] != if source == "daily" { "daily" } else { KIND } {
        return Err(format!("{raw_file}：档案类型不一致"));
    }
    rec::validate_archive(p, issue, draws)?;
    if source == "independent" {
        if p.get("small_red_pools").is_none() {
            return Err("独立观察档缺少原始小池".into());
        }
        if let Some(parent) = p["linked_daily_sha256"].as_str() {
            let daily = rec::read_checked(&root.join(format!("daily/{issue}.json")))?;
            if daily["sha256"] != parent {
                return Err("独立小池绑定的每日原档摘要已变化".into());
            }
        }
    }
    if p.get("small_red_pools").is_none() {
        return Ok(None);
    }
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    if rec::saved_after_draw(p["published_at_unix"].as_u64(), draw) {
        return Err("小池观察保存晚于常规开奖时间，拒绝计入".into());
    }
    Ok(Some(Record {
        envelope,
        source,
        raw_file,
    }))
}
fn record(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Record>, String> {
    if !valid_issue(issue) {
        return Err("期号需要5位数字".into());
    }
    let daily = checked(root, draws, issue, "daily")?;
    let independent = checked(root, draws, issue, "independent")?;
    match (daily, independent) {
        (Some(a), Some(b)) => {
            if a.envelope["payload"]["small_red_pools"] != b.envelope["payload"]["small_red_pools"]
            {
                return Err("同一期每日与独立小池原档不一致，拒绝混算".into());
            }
            let a_time = a.envelope["payload"]["published_at_unix"]
                .as_u64()
                .ok_or("缺少保存时间")?;
            let b_time = b.envelope["payload"]["published_at_unix"]
                .as_u64()
                .ok_or("缺少保存时间")?;
            Ok(Some(if a_time <= b_time { a } else { b }))
        }
        (a, b) => Ok(a.or(b)),
    }
}
fn metadata(record: &Record) -> Value {
    let p = &record.envelope["payload"];
    json!({"issue":p["issue"],"source":record.source,"raw_file":record.raw_file,"archive_sha256":record.envelope["sha256"],"published_at_unix":p["published_at_unix"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"linked_daily_sha256":p["linked_daily_sha256"],"independent_timestamp_verified":false})
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
    if !valid_issue(issue) {
        return Err("期号需要5位数字".into());
    }
    if let Some(old) = record(root, draws, issue)? {
        let mut result = metadata(&old);
        result["status"] = json!("already_frozen");
        return Ok(result);
    }
    let started = clock();
    if started >= deadline {
        return Err("已到小池封存截止，禁止补发".into());
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    if issue != fmt_issue(last.issue + 1) || issue[..2] != fmt_issue(last.issue)[..2] {
        return Err("独立小池仅允许当前开奖紧邻下一期，跨年需核对".into());
    }
    let prepared = prepare_forecast(draws)?;
    let daily_path = root.join(format!("daily/{issue}.json"));
    let parent = if daily_path.exists() {
        let e = rec::read_checked(&daily_path)?;
        rec::validate_archive(&e["payload"], issue, draws)?;
        Some(e["sha256"].clone())
    } else {
        None
    };
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("小池计算越过截止或时钟倒退，拒绝留档".into());
    }
    let mut payload = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    payload["linked_daily_sha256"] = json!(parent);
    payload["small_red_pools"] = small_red::saved(&prepared);
    let path = root.join(format!("small-red/{issue}.json"));
    let (_, created) = rec::write_new_before(&path, payload, Some(deadline), &clock)?;
    let saved = record(root, draws, issue)?.ok_or("小池写入后未找到原档")?;
    let mut result = metadata(&saved);
    result["status"] = json!(if created { "frozen" } else { "already_frozen" });
    Ok(result)
}
fn issue_set(root: &Path) -> Result<BTreeSet<String>, String> {
    let mut issues = BTreeSet::new();
    for folder in ["daily", "small-red"] {
        match fs::read_dir(root.join(folder)) {
            Ok(entries) => {
                for entry in entries {
                    let path = entry.map_err(|e| e.to_string())?.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Some(issue) = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .filter(|s| valid_issue(s))
                        {
                            issues.insert(issue.to_owned());
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
fn outcome_sha(draw: &Draw) -> String {
    data_sha256(std::slice::from_ref(draw))
}
fn persisted(root: &Path, record: &Record, draw: Option<&Draw>) -> Result<Option<Value>, String> {
    let issue = record.envelope["payload"]["issue"]
        .as_str()
        .ok_or("缺少期号")?;
    let path = root.join(format!("reviews/{issue}/small-red.json"));
    if !path.exists() {
        return Ok(None);
    }
    let review = rec::read_checked(&path)?;
    let p = &review["payload"];
    let draw = draw.ok_or("独立小池已有复盘但当前缺少该期开奖")?;
    if p["kind"] != "small_red_observation_review"
        || p["archive_sha256"] != record.envelope["sha256"]
        || p["outcome_sha256"] != outcome_sha(draw)
    {
        return Err("独立小池复盘与原档或开奖摘要不一致".into());
    }
    Ok(Some(review))
}
pub fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let path = root.join(format!("small-red/{issue}.json"));
    if !path.exists() {
        return Ok(None);
    }
    let Some(r) = record(root, draws, issue)? else {
        return Ok(None);
    };
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let review = persisted(root, &r, draw)?;
    let mut value = metadata(&r);
    value["small_red_pools"] = if let Some(saved) = &review {
        saved["payload"]["small_red_pools"].clone()
    } else {
        small_red::review(&r.envelope["payload"]["small_red_pools"], draw)?
    };
    value["review_persisted"] = json!(review.is_some());
    if review.is_some() {
        value["review_file"] = json!(format!("reviews/{issue}/small-red.json"));
    }
    Ok(Some(value))
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
        if !root.join(format!("small-red/{issue}.json")).exists() {
            continue;
        }
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let outcome = (|| -> Result<usize, String> {
            let r = record(root, draws, &issue)?.ok_or("没有小池原档")?;
            if persisted(root, &r, Some(draw))?.is_some() {
                return Ok(0);
            }
            let payload = json!({"kind":"small_red_observation_review","schema_version":1,"issue":issue,"archive_sha256":r.envelope["sha256"],"outcome_sha256":outcome_sha(draw),"reviewed_at_unix":rec::now_unix(),"draw":{"issue":issue,"date":fmt_date(draw.date),"red":draw.red,"blue":draw.blue},"small_red_pools":small_red::review(&r.envelope["payload"]["small_red_pools"],Some(draw))?});
            let (_, created) = rec::write_new_before(
                &root.join(format!("reviews/{issue}/small-red.json")),
                payload,
                None,
                rec::now_unix,
            )?;
            persisted(root, &r, Some(draw))?;
            Ok(usize::from(created))
        })();
        match outcome {
            Ok(n) => result.settled += n,
            Err(e) => result.errors.push(format!("{issue}/small-red：{e}")),
        }
    }
    result
}
pub fn observations(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let (mut n, mut pending, mut missing) = (0, 0, 0);
    let (mut full, mut sums, mut gains, mut losses) =
        ([0usize; 8], [0usize; 8], [0usize; 8], [0usize; 8]);
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    for issue in issue_set(root)? {
        let outcome = (|| -> Result<(), String> {
            let Some(r) = record(root, draws, &issue)? else {
                missing += 1;
                return Ok(());
            };
            let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
            persisted(root, &r, draw)?;
            let reviewed = small_red::review(&r.envelope["payload"]["small_red_pools"], draw)?;
            let mut entry = metadata(&r);
            entry["issue"] = json!(issue);
            entry["pending"] = json!(draw.is_none());
            if draw.is_none() {
                pending += 1;
                entry["hits"] = Value::Null;
                periods.push(entry);
                return Ok(());
            }
            let hits: [[usize; 4]; 2] = std::array::from_fn(|m| {
                std::array::from_fn(|j| {
                    reviewed["models"][m]["pools"][j]["red_hits"]
                        .as_u64()
                        .unwrap() as usize
                })
            });
            n += 1;
            for m in 0..2 {
                for j in 0..4 {
                    let i = m * 4 + j;
                    full[i] += usize::from(hits[m][j] == 6);
                    sums[i] += hits[m][j];
                    gains[i] += usize::from(hits[m][j] == 6 && hits[0][j] != 6);
                    losses[i] += usize::from(hits[m][j] != 6 && hits[0][j] == 6);
                }
            }
            entry["hits"] = json!(hits);
            periods.push(entry);
            Ok(())
        })();
        if let Err(e) = outcome {
            errors.push(format!("{issue}：{e}"));
        }
    }
    let cells:Vec<_>=small_red::MODELS.iter().enumerate().flat_map(|(m,name)|small_red::SIZES.iter().enumerate().map(move|(j,&k)|{let i=m*4+j;json!({"model":name,"red_count":k,"n":n,"pending":pending,"excluded_missing_pair":missing,"full_red_periods":full[i],"full_red_rate":(n>0).then(||full[i]as f64/n as f64),"mean_missed_reds":(n>0).then(||6.-sums[i]as f64/n as f64),"paired_gains":gains[i],"paired_losses":losses[i]})})).collect();
    let issue = draws
        .last()
        .map(|d| fmt_issue(d.issue + 1))
        .ok_or("没有开奖数据")?;
    let next = record(root, draws, &issue);
    let mut next_archive = json!({"issue":issue,"daily_exists":root.join(format!("daily/{issue}.json")).exists(),"has_small_red_pools":false});
    match next {
        Ok(Some(r)) => {
            next_archive["has_small_red_pools"] = json!(true);
            for (k, v) in metadata(&r).as_object().unwrap() {
                next_archive[k] = v.clone();
            }
        }
        Ok(None) => {}
        Err(e) => errors.push(format!("{issue}：{e}")),
    }
    Ok(
        json!({"kind":"actual_saved_small_red_observations","actual_recommendation":true,"independent_timestamp_verified":false,"cells":cells,"periods":periods,"errors":errors,"next_archive":next_archive,
        "note":"只核对同一期每日或独立原档保存的原榜与试验小池，每期只计一次；同红球数量配对，缺档及待开奖不计失败。独立小池不会改写每日档案。本地留档无独立时间戳证明。"}),
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
            Self(
                std::env::temp_dir().join(format!(
                    "ssq-small-archive-{}-{}",
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
    fn history() -> Vec<Draw> {
        crate::draw::synth_draws(220, 9)
    }
    fn old_daily(root: &Path, d: &[Draw], at: u64, with_small: bool) {
        let p = prepare_forecast(d).unwrap();
        let issue = fmt_issue(p.cutoff_issue + 1);
        let mut payload = rec::base_payload(&p, &issue, "daily", at, Some(100));
        payload["strategies"] = json!({});
        if with_small {
            payload["small_red_pools"] = small_red::saved(&p);
        }
        rec::write_new_before(
            &root.join(format!("daily/{issue}.json")),
            payload,
            None,
            || at,
        )
        .unwrap();
    }
    #[test]
    fn independent_freeze_preserves_daily_and_rejects_deadline_and_invalid_prefixes() {
        let temp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        old_daily(&temp.0, &d, 10, false);
        let original = fs::read(temp.0.join(format!("daily/{issue}.json"))).unwrap();
        assert!(freeze_with_clock(&temp.0, &d, &issue, 100, || 100).is_err());
        assert!(freeze_with_clock(&temp.0, &d, "../x", 100, || 20).is_err());
        assert!(freeze_with_clock(
            &temp.0,
            &d,
            &fmt_issue(d.last().unwrap().issue + 2),
            100,
            || 20
        )
        .is_err());
        let clock = Cell::new(0);
        assert!(freeze_with_clock(&temp.0, &d, &issue, 100, || {
            let n = clock.get();
            clock.set(n + 1);
            [20, 21, 100][n.min(2)]
        })
        .is_err());
        assert!(!temp.0.join(format!("small-red/{issue}.json")).exists());
        let saved = freeze_with_clock(&temp.0, &d, &issue, 100, || 20).unwrap();
        assert_eq!(saved["status"], "frozen");
        assert_eq!(saved["source"], "independent");
        let path = temp.0.join(format!("small-red/{issue}.json"));
        let first = fs::read(&path).unwrap();
        let repeated = freeze_with_clock(&temp.0, &d, &issue, 100, || 200).unwrap();
        assert_eq!(saved["archive_sha256"], repeated["archive_sha256"]);
        assert_eq!(repeated["status"], "already_frozen");
        let obs = observations(&temp.0, &d).unwrap();
        assert_eq!(obs["errors"], json!([]));
        for c in obs["cells"].as_array().unwrap() {
            assert_eq!(c["pending"], 1);
            assert_eq!(c["n"], 0);
            assert_eq!(c["excluded_missing_pair"], 0);
            assert!(c["full_red_rate"].is_null());
        }
        assert_eq!(
            fs::read(temp.0.join(format!("daily/{issue}.json"))).unwrap(),
            original
        );
        assert_eq!(fs::read(&path).unwrap(), first);
        assert_eq!(
            rec::artifact(&temp.0, &format!("small-red/{issue}.json")).unwrap(),
            first
        );
        assert!(rec::artifact(&temp.0, "small-red/../daily/3221.json").is_err());
        let mut changed = d.clone();
        changed[0] = crate::draw::synth_draws(220, 10)[0].clone();
        assert!(view(&temp.0, &changed, &issue).is_err());
    }
    #[test]
    fn daily_extension_is_reused_and_two_equal_sources_are_counted_once() {
        let temp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        old_daily(&temp.0, &d, 10, true);
        let saved = freeze_with_clock(&temp.0, &d, &issue, 100, || 20).unwrap();
        assert_eq!(saved["source"], "daily");
        assert!(!temp.0.join("small-red").exists());
        let other = Temp::new();
        let frozen = freeze_with_clock(&other.0, &d, &issue, 100, || 10).unwrap();
        old_daily(&other.0, &d, 20, true);
        let obs = observations(&other.0, &d).unwrap();
        assert_eq!(obs["periods"].as_array().unwrap().len(), 1);
        assert_eq!(
            obs["periods"][0]["archive_sha256"],
            frozen["archive_sha256"]
        );
        assert_eq!(obs["cells"][0]["pending"], 1);
        // A modified but freshly hashed parent is still rejected by its bound digest.
        let bound = Temp::new();
        old_daily(&bound.0, &d, 10, false);
        freeze_with_clock(&bound.0, &d, &issue, 100, || 20).unwrap();
        let path = bound.0.join(format!("daily/{issue}.json"));
        let mut e = rec::read_checked(&path).unwrap();
        e["payload"]["changed"] = json!(true);
        e["sha256"] = json!(crate::hash::sha256_hex(
            &serde_json::to_vec(&e["payload"]).unwrap()
        ));
        fs::write(&path, serde_json::to_vec(&e).unwrap()).unwrap();
        assert!(view(&bound.0, &d, &issue).is_err());
    }
    #[test]
    fn original_pool_scoring_is_read_only_and_first_review_cannot_be_overwritten() {
        let temp = Temp::new();
        let mut d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        freeze_with_clock(&temp.0, &d, &issue, 100, || 20).unwrap();
        let raw_path = temp.0.join(format!("small-red/{issue}.json"));
        let bytes = fs::read(&raw_path).unwrap();
        let saved = rec::read_checked(&raw_path).unwrap();
        let reds: Vec<u8> = serde_json::from_value(
            saved["payload"]["small_red_pools"]["models"][1]["pools"][0]["reds"].clone(),
        )
        .unwrap();
        let pending = view(&temp.0, &d, &issue).unwrap().unwrap();
        assert!(pending["small_red_pools"]["models"][1]["pools"][0]["red_hits"].is_null());
        assert!(!temp.0.join("reviews").exists());
        d.push(
            Draw::from_parts(
                issue.parse().unwrap(),
                20030901,
                reds[..6].try_into().unwrap(),
                1,
            )
            .unwrap(),
        );
        let preview = view(&temp.0, &d, &issue).unwrap().unwrap();
        assert_eq!(
            preview["small_red_pools"]["models"][1]["pools"][0]["full_red"],
            true
        );
        assert!(!temp.0.join("reviews").exists());
        let first = settle(&temp.0, &d);
        assert_eq!(first.settled, 1);
        assert!(first.errors.is_empty());
        let review_path = temp.0.join(format!("reviews/{issue}/small-red.json"));
        let review = fs::read(&review_path).unwrap();
        assert_eq!(settle(&temp.0, &d).settled, 0);
        assert_eq!(
            view(&temp.0, &d, &issue).unwrap().unwrap()["review_persisted"],
            true
        );
        let mut changed = d.clone();
        changed.last_mut().unwrap().blue = 2;
        assert!(view(&temp.0, &changed, &issue).is_err());
        assert!(!settle(&temp.0, &changed).errors.is_empty());
        assert_eq!(fs::read(review_path).unwrap(), review);
        assert_eq!(fs::read(raw_path).unwrap(), bytes);
    }
    #[test]
    fn concurrent_freezes_keep_one_first_archive() {
        let temp = Temp::new();
        let d = history();
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let results = std::thread::scope(|s| {
            let a = s.spawn(|| freeze_with_clock(&temp.0, &d, &issue, 100, || 20).unwrap());
            let b = s.spawn(|| freeze_with_clock(&temp.0, &d, &issue, 100, || 30).unwrap());
            [a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(results[0]["archive_sha256"], results[1]["archive_sha256"]);
        assert_eq!(
            results.iter().filter(|r| r["status"] == "frozen").count(),
            1
        );
    }
}
