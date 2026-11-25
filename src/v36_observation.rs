//! Pre-draw pool observations and causal replay for the V3.6 adaptation.
use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    audit::target_draw,
    cycle::decide,
    draw::{data_sha256, fmt_date, fmt_issue, Draw},
    eval::{prepare_forecast, Prepared},
    fusion_rank_v4,
    hash::sha256_hex,
    model::roll_all,
    protocol::Protocol,
    recommendations as rec,
    select::{fuse6, rank_desc, rank_log_score, softmax_shrink},
    stats::binom,
    v36_model::{self as model, BLUE_POOL, RED_POOL},
};

const KIND: &str = "v36_scoring_observation_v1";
const REVIEW: &str = "v36_scoring_review_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Pool {
    red: [u8; RED_POOL],
    blue: [u8; BLUE_POOL],
}

impl Pool {
    fn ranked(red: &[usize; 33], blue: &[usize; 16]) -> Self {
        Self {
            red: std::array::from_fn(|i| red[i] as u8 + 1),
            blue: std::array::from_fn(|i| blue[i] as u8 + 1),
        }
    }

    fn hits(self, draw: &Draw) -> Hits {
        Hits {
            red: self
                .red
                .into_iter()
                .filter(|n| draw.red.contains(n))
                .collect(),
            blue: self.blue.contains(&draw.blue),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Pools {
    v36: Pool,
    fusion_v4: Pool,
}

#[derive(Serialize, Deserialize)]
struct Prediction {
    pools: Pools,
    red_scores: Vec<f64>,
    blue_scores: Vec<f64>,
}

#[derive(Serialize)]
struct Hits {
    red: Vec<u8>,
    blue: bool,
}

#[derive(Default, Serialize)]
struct Totals {
    red_hits: usize,
    at_least_four_red: usize,
    all_six_red: usize,
    blue_hit: usize,
    all_six_red_and_blue: usize,
}

impl Totals {
    fn add(&mut self, hits: &Hits) {
        self.red_hits += hits.red.len();
        self.at_least_four_red += usize::from(hits.red.len() >= 4);
        self.all_six_red += usize::from(hits.red.len() == 6);
        self.blue_hit += usize::from(hits.blue);
        self.all_six_red_and_blue += usize::from(hits.red.len() == 6 && hits.blue);
    }
}

struct Period {
    issue: u32,
    v36: Hits,
    fusion_v4: Hits,
}

fn source() -> String {
    sha256_hex(include_bytes!("v36_model.rs"))
}

fn predict(draws: &[Draw]) -> Result<(Prepared, Prediction), String> {
    let prepared = prepare_forecast(draws)?;
    let scores = model::score(draws);
    let (red, blue) = scores.order();
    let points = |v: &f64| (v * 1_000_000.).round() / 1_000_000.;
    let pools = Pools {
        v36: Pool::ranked(&red, &blue),
        fusion_v4: Pool::ranked(
            &fusion_rank_v4::order(&prepared.expert_red, &prepared.weights)?,
            &prepared.forecast.blue_order,
        ),
    };
    let prediction = Prediction {
        pools,
        red_scores: scores.red.iter().map(points).collect(),
        blue_scores: scores.blue.iter().map(points).collect(),
    };
    Ok((prepared, prediction))
}

fn replay(draws: &[Draw]) -> Result<Vec<Period>, String> {
    let snap = roll_all(draws)?;
    let first = draws.len().saturating_sub(1000).max(Protocol::BURN_IN);
    let mut history = model::History::default();
    for (i, d) in draws[..Protocol::BURN_IN].iter().enumerate() {
        history.add(d, i + 1);
    }

    let mut rows = Vec::new();
    let mut red_sums = [0.; 7];
    let mut blue_sums = [0.; 6];
    let mut n_fit = 0;
    for (offset, row) in snap.rows.iter().enumerate() {
        let i = offset + Protocol::BURN_IN;
        let draw = &draws[i];
        if i >= first && n_fit >= 30 {
            let (red, blue) = history.score(&draws[..i]).order();
            let baseline_red =
                fusion_rank_v4::order(&row.expert_p, &softmax_shrink(&red_sums, n_fit as f64))?;
            let baseline_blue = rank_desc(&fuse6(
                &row.blue_p,
                &softmax_shrink(&blue_sums, n_fit as f64),
            ));
            rows.push(Period {
                issue: row.issue,
                v36: Pool::ranked(&red, &blue).hits(draw),
                fusion_v4: Pool::ranked(&baseline_red, &baseline_blue).hits(draw),
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for (sum, expert) in red_sums.iter_mut().zip(&row.expert_p) {
                *sum += rank_log_score(expert, row.red_mask);
            }
            for (sum, expert) in blue_sums.iter_mut().zip(&row.blue_p) {
                *sum += expert[usize::from(row.blue)].clamp(1e-12, 1.).ln();
            }
            n_fit += 1;
        }
        history.add(draw, i + 1);
    }
    Ok(rows)
}

fn replay_window(rows: &[Period]) -> Value {
    let mut v36 = Totals::default();
    let mut fusion_v4 = Totals::default();
    for row in rows {
        v36.add(&row.v36);
        fusion_v4.add(&row.fusion_v4);
    }
    json!({
        "n":rows.len(), "from":rows.first().map(|r|fmt_issue(r.issue)),
        "to":rows.last().map(|r|fmt_issue(r.issue)), "v36":v36, "fusion_v4":fusion_v4
    })
}

fn archive_path(root: &Path, issue: &str) -> PathBuf {
    root.join(format!("v36/{issue}.json"))
}

fn review_path(root: &Path, issue: &str) -> PathBuf {
    root.join(format!("reviews/{issue}/v36.json"))
}

fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|c| c.is_ascii_digit())
}

fn gate_valid(at: u64, last: &Draw, issue: &str, deadline: u64) -> bool {
    let gate = decide(at, last.issue, last.date, true);
    gate.publish
        && gate.target == issue
        && gate.deadline_unix == deadline
        && rec::saved_after_draw(Some(at), Some(last))
}

fn checked(root: &Path, draws: &[Draw], issue: &str) -> Result<Value, String> {
    if !valid_issue(issue) {
        return Err("V3.6 观察期号须为 5 位数字".into());
    }
    let envelope = rec::read_checked(&archive_path(root, issue))?;
    let p = &envelope["payload"];
    rec::validate_archive(p, issue, draws)?;
    let cutoff: u32 = p["data_cutoff"]
        .as_str()
        .and_then(|n| n.parse().ok())
        .ok_or("V3.6 观察缺少数据截止")?;
    let end = draws.partition_point(|d| d.issue <= cutoff);
    let prefix = &draws[..end];
    let last = prefix.last().ok_or("V3.6 观察缺少输入历史")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("V3.6 观察缺少封存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("V3.6 观察缺少封存截止")?;
    let draw = target_draw(draws, cutoff, issue)?;
    if p["kind"] != KIND
        || p["schema_version"] != 1
        || p["actual_recommendation"] != false
        || p["known_outcome_at_generation"] != false
        || p["model_source_sha256"] != source()
        || at >= deadline
        || !gate_valid(at, last, issue, deadline)
        || rec::saved_after_draw(Some(at), draw)
        || p["prediction"] != json!(predict(prefix)?.1)
    {
        return Err("V3.6 观察的输入、分数或封存时间不一致".into());
    }
    Ok(envelope)
}

pub(crate) fn freeze(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
) -> Result<bool, String> {
    freeze_with_clock(root, draws, issue, deadline, rec::now_unix)
}

fn freeze_with_clock(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
    clock: impl Fn() -> u64,
) -> Result<bool, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    if !valid_issue(issue) {
        return Err("V3.6 观察期号须为 5 位数字".into());
    }
    if target_draw(draws, last.issue, issue)?.is_some() {
        return Err("V3.6 观察的目标期已开奖".into());
    }
    let path = archive_path(root, issue);
    if path.exists() {
        checked(root, draws, issue)?;
        return Ok(false);
    }
    let started = clock();
    if started >= deadline || !gate_valid(started, last, issue, deadline) {
        return Err("V3.6 观察须在系统截止前封存紧邻下一期".into());
    }
    let (prepared, prediction) = predict(draws)?;
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("V3.6 观察计算跨过截止或时钟倒退".into());
    }
    let mut payload = rec::base_payload(&prepared, issue, KIND, completed, Some(deadline));
    payload["prediction"] = json!(prediction);
    payload["model_source_sha256"] = json!(source());
    payload["actual_recommendation"] = json!(false);
    let (_, created) = rec::write_new_before(&path, payload, Some(deadline), || {
        let at = clock();
        if at < completed {
            deadline
        } else {
            at
        }
    })?;
    checked(root, draws, issue)?;
    Ok(created)
}

fn issues(root: &Path) -> Result<Vec<String>, String> {
    let entries = match fs::read_dir(root.join("v36")) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("无法读取 V3.6 观察档案：{e}")),
    };
    let mut issues = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        if let Some(issue) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.strip_suffix(".json"))
        {
            if valid_issue(issue) {
                issues.push(issue.to_string());
            }
        }
    }
    issues.sort_unstable_by(|a, b| b.cmp(a));
    Ok(issues)
}

fn build_review(archive: &Value, draw: &Draw) -> Result<Value, String> {
    let pools: Pools = serde_json::from_value(archive["payload"]["prediction"]["pools"].clone())
        .map_err(|e| format!("V3.6 观察号码池无效：{e}"))?;
    Ok(json!({
        "kind":REVIEW, "schema_version":1, "issue":fmt_issue(draw.issue),
        "archive_sha256":archive["sha256"],
        "outcome_sha256":data_sha256(std::slice::from_ref(draw)),
        "draw":{"red":draw.red,"blue":draw.blue,"date":fmt_date(draw.date)},
        "v36":pools.v36.hits(draw), "fusion_v4":pools.fusion_v4.hits(draw)
    }))
}

fn checked_review(path: &Path, expected: &Value) -> Result<(), String> {
    let review = rec::read_checked(path)?;
    let mut payload = review["payload"].clone();
    let timestamp = payload
        .as_object_mut()
        .and_then(|p| p.remove("reviewed_at_unix"));
    if timestamp.and_then(|v| v.as_u64()).is_none() || &payload != expected {
        return Err("V3.6 复盘与首次原档或开奖不一致".into());
    }
    Ok(())
}

pub(crate) fn settle(root: &Path, draws: &[Draw]) -> rec::Settlement {
    let mut out = rec::Settlement::default();
    let issues = match issues(root) {
        Ok(issues) => issues,
        Err(e) => {
            out.errors.push(e);
            return out;
        }
    };
    for issue in issues {
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == issue) else {
            continue;
        };
        let path = review_path(root, &issue);
        if path.exists() {
            continue;
        }
        let result = (|| {
            let archive = checked(root, draws, &issue)?;
            let expected = build_review(&archive, draw)?;
            let mut payload = expected.clone();
            payload["reviewed_at_unix"] = json!(rec::now_unix());
            let (_, created) = rec::write_new_before(&path, payload, None, rec::now_unix)?;
            checked_review(&path, &expected)?;
            Ok::<_, String>(created)
        })();
        match result {
            Ok(created) => out.settled += usize::from(created),
            Err(e) => out.errors.push(format!("{issue} V3.6 复盘：{e}")),
        }
    }
    out
}

fn report_row(root: &Path, draws: &[Draw], issue: &str) -> Result<Value, String> {
    let archive = checked(root, draws, issue)?;
    let p = &archive["payload"];
    let result = draws
        .iter()
        .find(|d| fmt_issue(d.issue) == issue)
        .map(|d| build_review(&archive, d))
        .transpose()?;
    let path = review_path(root, issue);
    let review_saved = path.exists();
    if review_saved {
        checked_review(&path, result.as_ref().ok_or("已有 V3.6 复盘但缺少开奖")?)?;
    }
    Ok(json!({
        "issue":issue, "status":if result.is_some() { "scored" } else { "frozen" },
        "cutoff_issue":p["data_cutoff"], "published_at_unix":p["published_at_unix"],
        "archive_sha256":archive["sha256"], "prediction":p["prediction"],
        "result":result, "review_saved":review_saved
    }))
}

pub(crate) fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue + 1);
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    for archived in issues(root)? {
        match report_row(root, draws, &archived) {
            Ok(row) => rows.push(row),
            Err(e) => errors.push(format!("{archived}：{e}")),
        }
    }
    let current = if let Some(row) = rows.iter().find(|row| row["issue"] == issue) {
        row.clone()
    } else if archive_path(root, &issue).exists() {
        json!({"issue":issue,"status":"invalid"})
    } else {
        json!({"issue":issue,"status":"preview","cutoff_issue":fmt_issue(last.issue),"prediction":predict(draws)?.1})
    };
    let replay = replay(draws)?;
    let all_reds = binom(33, 6) as f64;
    let uniform = |hits| binom(12, hits) as f64 * binom(21, 6 - hits) as f64 / all_reds;
    Ok(json!({
        "version":1, "actual_recommendation":false, "data_sha256":data_sha256(draws),
        "current":current, "archives":rows, "errors":errors,
        "replay":{
            "recent1000":replay_window(&replay),
            "recent60":replay_window(&replay[replay.len().saturating_sub(60)..]),
            "uniform":{
                "mean_red_hits":72./33.,
                "at_least_four_red":(4..=6).map(uniform).sum::<f64>(),
                "all_six_red":uniform(6), "blue_hit":0.25, "all_six_red_and_blue":uniform(6)*0.25
            }
        }
    }))
}

pub(crate) fn artifact(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    checked(root, draws, issue)?;
    fs::read(archive_path(root, issue)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> Vec<Draw> {
        crate::draw::load_csv(Path::new("data/ssq.csv"))
            .unwrap()
            .into_iter()
            .take_while(|d| d.issue <= 26123)
            .collect()
    }

    #[test]
    fn replay_matches_the_forecast_made_before_the_outcome() {
        let draws = history();
        let rows = replay(&draws).unwrap();
        assert_eq!(rows.len(), 1000);
        let last = rows.last().unwrap();
        let draw = draws.last().unwrap();
        let (_, forecast) = predict(&draws[..draws.len() - 1]).unwrap();
        assert_eq!(last.issue, draw.issue);
        assert_eq!(json!(last.v36), json!(forecast.pools.v36.hits(draw)));
        assert_eq!(
            json!(last.fusion_v4),
            json!(forecast.pools.fusion_v4.hits(draw))
        );
    }

    #[test]
    fn first_archive_is_saved_only_before_the_deadline_and_review_binds_frozen_numbers() {
        let root = std::env::temp_dir().join(format!(
            "ssq-v36-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut draws = history();
        let issue = "26124";
        let now = 1_793_152_800; // 2026-10-28 10:00 CST
        let last = draws.last().unwrap();
        let gate = decide(now, last.issue, last.date, false);
        assert!(gate.publish);
        assert_eq!(gate.target, issue);
        let deadline = gate.deadline_unix;

        let preview = report(&root, &draws).unwrap();
        assert_eq!(preview["current"]["status"], "preview");
        assert!(!root.exists());
        assert!(freeze_with_clock(&root, &draws, issue, deadline, || deadline).is_err());
        assert!(!root.exists());
        assert!(freeze_with_clock(&root, &draws, issue, deadline, || now).unwrap());
        assert!(!freeze_with_clock(&root, &draws, issue, deadline, || now + 1).unwrap());
        let original = artifact(&root, &draws, issue).unwrap();
        let pending = report_row(&root, &draws, issue).unwrap();
        assert_eq!(pending["prediction"], preview["current"]["prediction"]);
        assert_eq!(pending["status"], "frozen");
        assert!(pending["result"].is_null());
        assert_eq!(settle(&root, &draws).settled, 0);

        let pools: Pools = serde_json::from_value(pending["prediction"]["pools"].clone()).unwrap();
        let winner = pools.v36;
        draws.push(
            Draw::from_parts(
                26124,
                20261029,
                winner.red[..6].try_into().unwrap(),
                winner.blue[0],
            )
            .unwrap(),
        );
        assert!(freeze_with_clock(&root, &draws, issue, deadline, || now).is_err());
        assert_eq!(settle(&root, &draws).settled, 1);
        let scored = report_row(&root, &draws, issue).unwrap();
        assert_eq!(scored["status"], "scored");
        assert_eq!(scored["review_saved"], true);
        assert_eq!(scored["result"]["v36"]["red"], json!(winner.red[..6]));
        assert_eq!(scored["result"]["v36"]["blue"], true);
        let first_review = fs::read(review_path(&root, issue)).unwrap();
        assert_eq!(settle(&root, &draws).settled, 0);
        assert_eq!(artifact(&root, &draws, issue).unwrap(), original);
        assert_eq!(fs::read(review_path(&root, issue)).unwrap(), first_review);

        draws.last_mut().unwrap().blue = (winner.blue[0] % 16) + 1;
        assert!(report_row(&root, &draws, issue).is_err());
        assert_eq!(fs::read(review_path(&root, issue)).unwrap(), first_review);
        fs::remove_dir_all(root).unwrap();
    }
}
