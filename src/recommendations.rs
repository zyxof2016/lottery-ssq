//! Actual recommendation archives. Reviews consume saved tickets and evidence;
//! this module never reconstructs missing historical recommendations.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::audit::target_draw;
use crate::draw::{data_sha256, fmt_date, fmt_issue, Draw};
use crate::efficiency;
use crate::eval::{prepare_forecast, Prepared};
use crate::hash::sha256_hex;
use crate::plan::{self, Plan, PlanTicket};
use crate::pool;
use crate::protocol::Protocol;
use crate::select::rank_desc;

pub const PRESETS: [&str; 11] = [
    "frozen",
    "disjoint5",
    "single8",
    "compound7",
    "compound8",
    "blue2",
    "dantuo",
    "balanced14",
    "calibrated14",
    "coverage24",
    "equal24",
];
pub const PRIZE_RULES_URL: &str = "https://www.scflcp.com.cn/yxgz/823145252.jhtml";

#[derive(Clone, Serialize, Deserialize)]
struct Ticket {
    red: [u8; 6],
    blue: u8,
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn digest(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|v| sha256_hex(&v))
        .map_err(|e| e.to_string())
}

fn source_sha() -> String {
    sha256_hex(
        concat!(
            env!("CARGO_PKG_VERSION"),
            include_str!("model.rs"),
            include_str!("select.rs"),
            include_str!("protocol.rs"),
            include_str!("plan.rs"),
            include_str!("eval.rs"),
            include_str!("recommendations.rs"),
            include_str!("efficiency.rs"),
            include_str!("stats.rs"),
            include_str!("pool.rs"),
            include_str!("coverage.rs"),
            include_str!("ranking.rs"),
            include_str!("conditional.rs"),
            include_str!("pool_algorithms.rs")
        )
        .as_bytes(),
    )
}

pub(crate) fn read_checked(path: &Path) -> Result<Value, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("无法读取档案 {}：{e}", path.display()))?;
    let envelope: Value =
        serde_json::from_str(&text).map_err(|e| format!("档案 JSON 无效：{e}"))?;
    let payload = envelope.get("payload").ok_or("档案缺少 payload")?;
    if envelope.get("sha256").and_then(Value::as_str) != Some(digest(payload)?.as_str()) {
        return Err(format!("档案摘要不一致，拒绝读取：{}", path.display()));
    }
    Ok(envelope)
}

/// Publish a complete file atomically without replacing an existing archive.
fn write_new(path: &Path, payload: Value) -> Result<(Value, bool), String> {
    write_new_before(path, payload, None, now_unix)
}

fn write_new_before(
    path: &Path,
    payload: Value,
    deadline: Option<u64>,
    clock: impl Fn() -> u64,
) -> Result<(Value, bool), String> {
    let envelope = json!({"sha256": digest(&payload)?, "payload": payload});
    let parent = path.parent().ok_or("档案目录无效")?;
    fs::create_dir_all(parent).map_err(|e| format!("无法创建档案目录：{e}"))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temporary = parent.join(format!(".archive-{}-{stamp}.tmp", std::process::id()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        file.write_all(&serde_json::to_vec_pretty(&envelope).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        if deadline.is_some_and(|limit| clock() >= limit) {
            return Err("写盘时已到推荐截止时间，拒绝留档".into());
        }
        match fs::hard_link(&temporary, path) {
            Ok(()) => Ok((envelope, true)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Ok((read_checked(path)?, false))
            }
            Err(e) => Err(format!("无法原子创建档案 {}：{e}", path.display())),
        }
    })();
    let _ = fs::remove_file(temporary);
    result
}

fn valid_issue(issue: &str) -> bool {
    issue.len() == 5 && issue.bytes().all(|c| c.is_ascii_digit())
}

fn valid_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn numbers(indices: &[usize]) -> Vec<u8> {
    indices.iter().map(|i| *i as u8 + 1).collect()
}

fn number_evidence(
    tickets: &[PlanTicket],
    reds: &[u8],
    blues: &[u8],
    prepared: &Prepared,
    frozen: bool,
) -> Vec<Value> {
    let fc = &prepared.forecast;
    let mut evidence = Vec::new();
    for (color, size) in [("red", 33), ("blue", 16)] {
        for idx in 0..size {
            let n = idx as u8 + 1;
            let (order, score, in_pool, rescued, displaced) = if color == "red" {
                (
                    fc.red_order.as_slice(),
                    fc.fusion[idx],
                    if frozen {
                        fc.pool.contains(&idx)
                    } else {
                        reds.contains(&n)
                    },
                    frozen && fc.mild.rescued.contains(&idx),
                    frozen && fc.mild.displaced.contains(&idx),
                )
            } else {
                (
                    fc.blue_order.as_slice(),
                    fc.blue[idx],
                    blues.contains(&n),
                    false,
                    false,
                )
            };
            let selected = tickets.iter().any(|t| {
                if color == "red" {
                    t.red.contains(&n)
                } else {
                    t.blue == n
                }
            });
            let reason = if selected {
                "ticketed"
            } else if displaced {
                "replaced_by_rescue"
            } else if in_pool {
                "not_selected_by_ticket_objective"
            } else {
                "below_candidate_cutoff"
            };
            let mut scores = Vec::new();
            let mut nominations = Vec::new();
            let experts = if color == "red" { 7 } else { 6 };
            for e in 0..experts {
                let (name, label, value, nominated) = if color == "red" {
                    (
                        Protocol::EXPERTS[e],
                        Protocol::EXPERTS_ZH[e],
                        prepared.expert_red[e][idx],
                        rank_desc(&prepared.expert_red[e])[..Protocol::RESCUE_M].contains(&idx),
                    )
                } else {
                    (
                        Protocol::BLUES[e],
                        Protocol::BLUES_ZH[e],
                        prepared.expert_blue[e][idx],
                        rank_desc(&prepared.expert_blue[e])[..Protocol::BLUE_SLOTS].contains(&idx),
                    )
                };
                scores.push(json!({"name": name, "label": label, "score": value}));
                if nominated {
                    nominations.push(label);
                }
            }
            evidence.push(json!({
                "color": color, "number": n, "fusion_score": score,
                "fusion_rank": order.iter().position(|i| *i == idx).map(|i| i + 1),
                "in_candidate_pool": in_pool, "rescued": rescued, "displaced": displaced,
                "reason": reason, "expert_scores": scores, "nominating_experts": nominations,
                "conditional_evidence": if color=="red" && prepared.ranking.as_ref().is_some_and(|m|m["model"]==crate::conditional::MODEL) {
                    let m=prepared.ranking.as_ref().unwrap();let b=m["training"]["next_gap_buckets"][idx].as_u64().unwrap()as usize;
                    json!({"gap_bucket":m["policy"]["gap_buckets"][b],"hits":m["training"]["bucket_hits"][b],
                        "opportunities":m["training"]["bucket_opportunities"][b],"smoothed_frequency":m["raw_conditional_scores"][idx]})
                } else {Value::Null},
            }));
        }
    }
    evidence
}

fn strategy(plan: &Plan, prepared: &Prepared) -> Value {
    let fc = &prepared.forecast;
    let frozen = plan.mode == "frozen";
    let evidence = number_evidence(&plan.tickets, &plan.reds, &plan.blues, prepared, frozen);
    let tickets: Vec<_> = plan
        .tickets
        .iter()
        .map(|t| json!({"red": t.red.to_vec(), "blue": t.blue}))
        .collect();
    let stages = if frozen {
        json!([
            {"stage": "fusion_top_k", "color": "red", "numbers": numbers(&fc.red_order[..Protocol::PRIMARY_K])},
            {"stage": "final_set", "color": "red", "numbers": numbers(&fc.mild.members)},
            {"stage": "rescued", "color": "red", "numbers": numbers(&fc.mild.rescued)},
            {"stage": "candidate_pool", "color": "red", "numbers": numbers(&fc.pool)}
        ])
    } else {
        json!([{ "stage": "candidate_pool", "color": "red", "numbers": plan.reds }])
    };
    json!({
        "title": plan.title, "mode": plan.mode, "rationale": plan.detail, "tickets": tickets,
        "coverage_optimization": if plan.mode == "balanced" { let mut p=crate::coverage::ticket_policy(plan.spec.pool,plan.spec.tickets,plan.spec.blue_count);if prepared.ranking.is_some() {p["ranking"]=prepared.ranking.as_ref().unwrap()["model"].clone();p["note"]=json!("完整使用所选模型排名号码池并均衡曝光；排名策略与组票分别留档。");}p } else { Value::Null },
        "parameters": {"mode": plan.spec.mode, "tickets": plan.spec.tickets, "pool": plan.spec.pool,
            "overlap": plan.spec.overlap, "red_count": plan.spec.red_count, "blue_count": plan.spec.blue_count,
            "bankers": plan.spec.bankers, "drags": plan.spec.drags},
        "evidence": evidence, "funnel": stages, "red_weights": prepared.ranking.as_ref().map(|r|r["red_weights"].clone()).unwrap_or_else(||json!(prepared.weights)),
        "ranking": prepared.ranking,
        "blue_weights": prepared.blue_weights, "bankers": plan.bankers, "drags": plan.drags,
    })
}

fn efficiency_strategy(profile: &efficiency::Profile, prepared: &Prepared) -> Value {
    let reds: Vec<_> = profile
        .tickets
        .iter()
        .flat_map(|t| t.red)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let blues: Vec<_> = profile
        .tickets
        .iter()
        .map(|t| t.blue)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let uniform = profile.basis == "uniform";
    json!({
        "title": profile.title, "mode": "efficiency", "rationale": profile.rationale,
        "tickets": profile.tickets.iter().map(|t| json!({"red": t.red, "blue": t.blue})).collect::<Vec<_>>(),
        "structural": uniform,
        "efficiency": {"version": efficiency::VERSION, "basis": profile.basis,
            "seed": format!("0x{:016x}", Protocol::SEED), "objective": "marginal_number_coverage",
            "parent_strategy": if profile.basis == "same_union" { Some("frozen") } else { None }},
        "evidence": if uniform { Value::Null } else { json!(number_evidence(&profile.tickets, &reds, &blues, prepared, false)) },
        "red_weights": if uniform { Value::Null } else { json!(prepared.weights) },
        "blue_weights": if uniform { Value::Null } else { json!(prepared.blue_weights) },
        "funnel": [{"stage": "candidate_pool", "color": "red", "numbers": reds}],
    })
}

fn base_payload(
    prepared: &Prepared,
    issue: &str,
    kind: &str,
    published: u64,
    deadline: Option<u64>,
) -> Value {
    json!({
        "schema_version": 1, "kind": kind, "issue": issue, "published_at_unix": published,
        "operational_cutoff_unix": deadline, "schedule_verified": false,
        "independent_timestamp_verified": false, "audit_scope": "operational_archive_not_confirmation",
        "data_cutoff": fmt_issue(prepared.cutoff_issue), "data_cutoff_date": fmt_date(prepared.cutoff_date),
        "input_history_sha256": prepared.data_sha, "history_count": prepared.n,
        "protocol_sha256": Protocol::sha256(), "source_sha256": source_sha(),
        "environment": {"package_version": env!("CARGO_PKG_VERSION"), "os": std::env::consts::OS, "arch": std::env::consts::ARCH},
        "known_outcome_at_generation": issue.parse::<u32>().map(|i| i <= prepared.cutoff_issue).unwrap_or(true),
    })
}

pub fn freeze_daily(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
) -> Result<Value, String> {
    freeze_daily_with_clock(root, draws, issue, deadline, now_unix)
}

fn freeze_daily_with_clock(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
    clock: impl Fn() -> u64,
) -> Result<Value, String> {
    if !valid_issue(issue) {
        return Err("期号需要 5 位数字".into());
    }
    let path = root.join("daily").join(format!("{issue}.json"));
    if path.exists() {
        let old = read_checked(&path)?;
        validate_archive(&old["payload"], issue, draws)?;
        return Ok(
            json!({"status": "already_frozen", "issue": issue, "sha256": old["sha256"],
            "strategies": old["payload"]["strategies"].as_object().map(|v| v.len()),
            "observation_tiers": old["payload"]["number_pools"]["observation_tiers"].as_array().map_or(0, Vec::len),
            "number_pools": old["payload"]["number_pools"]["pools"].as_array().map_or(0, Vec::len),
            "comparison_models": pool::reviewed_models(&old["payload"]["number_pools"]).len(),
            "saved_models": pool::reviewed_models(&old["payload"]["number_pools"]).iter().map(|m|&m["model"]).collect::<Vec<_>>(),
            "comparison_number_pools": comparison_pool_count(&old["payload"]["number_pools"])}),
        );
    }
    let started = clock();
    if started >= deadline {
        return Err("已过推荐截止时间，禁止补发历史推荐".into());
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    if issue.parse::<u32>().ok() != last.issue.checked_add(1)
        || issue[..2] != fmt_issue(last.issue)[..2]
    {
        return Err("每期推荐仅支持本地开奖紧邻下一期；跨年或缺期需核对".into());
    }
    let prepared = prepare_forecast(draws)?;
    let ranked = crate::ranking::prepare(draws)?;
    let conditional = crate::conditional::prepare(draws)?;
    let equal = crate::ranking::prepare_equal(draws)?;
    let mut variants = BTreeMap::new();
    for name in PRESETS {
        let prepared = if name == "calibrated14" {&ranked} else if name == "coverage24" {&conditional} else if name == "equal24" {&equal} else {&prepared};
        let spec = plan::resolve(Some(name), None, None, None, None, None, None, None, None)?;
        variants.insert(
            name.to_string(),
            strategy(&plan::build(&prepared.forecast, &spec)?, &prepared),
        );
    }
    for profile in efficiency::profiles(&prepared.forecast, issue)? {
        variants.insert(
            profile.name.clone(),
            efficiency_strategy(&profile, &prepared),
        );
    }
    let completed = clock();
    if completed < started || completed >= deadline {
        return Err("计算完成已到截止时间或时钟倒退，拒绝留档".into());
    }
    let mut payload = base_payload(&prepared, issue, "daily", completed, Some(deadline));
    payload["strategies"] = json!(variants);
    payload["presets"] = json!(PRESETS);
    payload["number_pools"] = pool::saved_current(&prepared, draws);
    payload["efficiency"] = json!({"version": efficiency::VERSION, "seed": format!("0x{:016x}", Protocol::SEED),
        "profiles": efficiency::PROFILE_COUNT, "objective": "marginal_number_coverage"});
    let (envelope, created) = write_new_before(&path, payload, Some(deadline), &clock)?;
    validate_archive(&envelope["payload"], issue, draws)?;
    Ok(
        json!({"status": if created { "frozen" } else { "already_frozen" }, "issue": issue, "sha256": envelope["sha256"],
            "strategies": envelope["payload"]["strategies"].as_object().map(|v| v.len()),
            "observation_tiers": envelope["payload"]["number_pools"]["observation_tiers"].as_array().map_or(0, Vec::len),
            "number_pools": envelope["payload"]["number_pools"]["pools"].as_array().map_or(0, Vec::len),
            "comparison_models": pool::reviewed_models(&envelope["payload"]["number_pools"]).len(),
            "saved_models": pool::reviewed_models(&envelope["payload"]["number_pools"]).iter().map(|m|&m["model"]).collect::<Vec<_>>(),
            "comparison_number_pools": comparison_pool_count(&envelope["payload"]["number_pools"])}),
    )
}

fn comparison_pool_count(saved: &Value) -> usize {
    if saved["version"] != 3 && saved["version"] != pool::RANKED_VERSION { return 0; }
    saved["comparison_models"].as_array().into_iter().flatten()
        .map(|m| m["pools"].as_array().map_or(0, Vec::len)).sum::<usize>() + saved["coverage_model"]["pools"].as_array().map_or(0,Vec::len)
        + saved["algorithm_models"].as_array().into_iter().flatten().map(|m|m["pools"].as_array().map_or(0,Vec::len)).sum::<usize>()
}

pub fn archive_plan(
    root: &Path,
    plan: &Plan,
    prepared: &Prepared,
    issue: &str,
    deadline: Option<u64>,
) -> Result<Value, String> {
    if !valid_issue(issue) {
        return Err("期号需要 5 位数字".into());
    }
    let saved = strategy(plan, prepared);
    let identity = strategy_identity(&saved, prepared, issue)?;
    let path = root
        .join("interactive")
        .join(issue)
        .join(format!("{identity}.json"));
    let envelope = if path.exists() {
        read_checked(&path)?
    } else {
        let mut payload = base_payload(prepared, issue, "interactive", now_unix(), deadline);
        payload["strategies"] = json!({"custom": saved});
        write_new(&path, payload)?.0
    };
    Ok(
        json!({"archive_id": format!("interactive:{identity}"), "sha256": envelope["sha256"],
        "known_outcome_at_generation": envelope["payload"]["known_outcome_at_generation"],
        "url": format!("reviews?issue={issue}&archive=interactive:{identity}") }),
    )
}

fn strategy_identity(saved: &Value, prepared: &Prepared, issue: &str) -> Result<String, String> {
    digest(&json!({
        "issue": issue, "strategy": saved, "source": source_sha(), "data": prepared.data_sha,
    }))
}

/// Reject a stale save rather than silently replacing the displayed tickets.
pub fn check_displayed_plan(
    root: &Path,
    plan: &Plan,
    prepared: &Prepared,
    issue: &str,
    expected_sha: &str,
) -> Result<(), String> {
    if !valid_issue(issue) {
        return Err("期号需要 5 位数字".into());
    }
    let identity = strategy_identity(&strategy(plan, prepared), prepared, issue)?;
    let path = root
        .join("interactive")
        .join(issue)
        .join(format!("{identity}.json"));
    let envelope = read_checked(&path)
        .map_err(|_| "显示的方案与当前数据、期号或参数不一致，请重新生成后保存")?;
    if envelope["sha256"].as_str() != Some(expected_sha) {
        return Err("显示的方案摘要不一致，请重新生成后保存".into());
    }
    Ok(())
}

fn validate_prefix(payload: &Value, draws: &[Draw]) -> Result<(), String> {
    let cutoff = payload["data_cutoff"]
        .as_str()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or("档案截止期无效")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= cutoff)
        .cloned()
        .collect();
    if prefix.last().map(|d| d.issue) != Some(cutoff)
        || payload["input_history_sha256"].as_str() != Some(data_sha256(&prefix).as_str())
    {
        return Err("推荐使用的历史前缀已变化，拒绝复盘".into());
    }
    Ok(())
}

fn validate_archive(payload: &Value, issue: &str, draws: &[Draw]) -> Result<(), String> {
    if payload["issue"] != issue {
        return Err("档案期号与路径不一致".into());
    }
    validate_prefix(payload, draws)?;
    if payload["number_pools"]["version"].as_u64().is_some_and(|v|v>=3) && payload["number_pools"]["target_issue"] != issue {
        return Err("三模型号码池目标期号与档案不一致".into());
    }
    if payload["number_pools"]["version"] == pool::RANKED_VERSION {
        let previous=draws.iter().find(|d|payload["data_cutoff"]==fmt_issue(d.issue)).ok_or("缺少校准输入开奖")?;
        let models=payload["number_pools"]["comparison_models"].as_array().ok_or("缺少排名模型")?;
        let ranked=models.iter().find(|m|m["model"]==crate::ranking::MODEL).ok_or("缺少校准排名")?;
        let mut numbers=previous.red.to_vec();numbers.sort_unstable();
        if ranked["previous_reds"]!=json!(numbers) {return Err("校准排名上期号码与输入开奖不一致".into());}
        pool::review(&payload["number_pools"],None)?;
        if !payload["number_pools"]["coverage_model"].is_null() {
            let prefix:Vec<_>=draws.iter().filter(|d|d.issue<=previous.issue).cloned().collect();
            crate::conditional::validate_history(&payload["number_pools"]["coverage_model"],&prefix)?;
        }
        if let Some(models)=payload["number_pools"]["algorithm_models"].as_array() {
            let prefix:Vec<_>=draws.iter().filter(|d|d.issue<=previous.issue).cloned().collect();
            crate::pool_algorithms::validate_history(models,&prefix)?;
        }
    }
    if payload["kind"] == "daily" {
        let cutoff = payload["data_cutoff"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .ok_or("截止期无效")?;
        target_draw(draws, cutoff, issue)?;
        let published = payload["published_at_unix"]
            .as_u64()
            .ok_or("留档时间无效")?;
        let deadline = payload["operational_cutoff_unix"]
            .as_u64()
            .ok_or("截止时间无效")?;
        if published >= deadline || payload["known_outcome_at_generation"] != false {
            return Err("每期推荐的发布时间或数据边界无效".into());
        }
    }
    Ok(())
}

fn archive_paths(root: &Path, issue: &str) -> Vec<(String, PathBuf)> {
    let mut paths = Vec::new();
    let daily = root.join("daily").join(format!("{issue}.json"));
    if daily.is_file() {
        paths.push(("daily".into(), daily));
    }
    if let Ok(entries) = fs::read_dir(root.join("interactive").join(issue)) {
        for entry in entries.flatten() {
            let path = entry.path();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if path.extension().and_then(|s| s.to_str()) == Some("json") && valid_id(stem) {
                paths.push((format!("interactive:{stem}"), path));
            }
        }
    }
    paths.sort_by(|a, b| a.0.cmp(&b.0));
    paths
}

fn review_path(root: &Path, issue: &str, id: &str) -> PathBuf {
    root.join("reviews").join(issue).join(format!(
        "{}.json",
        id.strip_prefix("interactive:").unwrap_or(id)
    ))
}

fn draw_json(draw: &Draw) -> Value {
    json!({"issue": fmt_issue(draw.issue), "date": fmt_date(draw.date), "red": draw.red.to_vec(), "blue": draw.blue})
}

fn saved_after_draw(published: Option<u64>, draw: Option<&Draw>) -> bool {
    published.zip(draw).is_some_and(|(time, d)| {
        crate::cycle::fmt_cst(time) >= format!("{} 21:10", fmt_date(d.date))
    })
}

/// Observe only pools saved in a daily archive; missing tiers are never rebuilt.
pub fn pool_observations(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let mut paths = Vec::new();
    match fs::read_dir(root.join("daily")) {
        Ok(entries) => {
            for entry in entries {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json")
                    && path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(valid_issue)
                {
                    paths.push(path);
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("无法读取号码池留档：{e}")),
    }
    paths.sort();
    let mut records = Vec::new();
    let mut errors = Vec::new();
    let mut missing_tier_archives = 0;
    for path in paths {
        let issue = path.file_stem().and_then(|s| s.to_str()).unwrap();
        let result: Result<Option<Value>, String> = (|| {
            let envelope = read_checked(&path)?;
            let payload = &envelope["payload"];
            if payload["kind"] != "daily" {
                return Err("每日目录中不是每日留档".into());
            }
            validate_archive(payload, issue, draws)?;
            let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
            if saved_after_draw(payload["published_at_unix"].as_u64(), draw) {
                return Err("保存晚于常规开奖时间，未纳入覆盖观察".into());
            }
            let reviewed = pool::review(&payload["number_pools"], draw)?;
            let has_tiers = reviewed["pools"]
                .as_array()
                .is_some_and(|pools| pools.iter().any(|p| p["group"] == "observation"));
            if !has_tiers {
                return Ok(None);
            }
            Ok(Some(
                json!({"issue": issue, "archive_sha256": envelope["sha256"],
                "published_at_unix": payload["published_at_unix"], "draw": draw.map(draw_json),
                "smallest_full_red_tier": reviewed["smallest_full_red_tier"], "pools": reviewed["pools"],
                "models": pool::reviewed_models(&reviewed)}),
            ))
        })();
        match result {
            Ok(Some(record)) => records.push(record),
            Ok(None) => missing_tier_archives += 1,
            Err(error) => errors.push(format!("{issue}：{error}")),
        }
    }
    let mut cells = Vec::new();
    let mut comparison_cells = Vec::new();
    let mut paired_cells = Vec::new();
    let mut ranking_paired_cells = Vec::new();
    let mut conditional_paired_cells = Vec::new();
    let mut algorithm_paired_cells = Vec::new();
    let mut first_algorithm_issues = serde_json::Map::new();
    let conditional_records:Vec<_>=records.iter().filter(|r|observation_pool(r,crate::conditional::MODEL,14,1).is_some()).cloned().collect();
    let ranking_records: Vec<_> = records.iter().filter(|r| observation_pool(r, crate::ranking::MODEL, 14, 1).is_some()).cloned().collect();
    for red_count in pool::TIER_RED_SIZES {
        for blue_count in pool::BLUE_SIZES {
            cells.push(observation_cell(&records, "weighted", red_count, blue_count, false));
            for model in pool::OBSERVATION_MODELS.into_iter().chain([crate::conditional::MODEL]).chain(crate::pool_algorithms::MODELS) {
                comparison_cells.push(observation_cell(&records, model, red_count, blue_count, false));
            }
            let paired: Vec<_> = pool::COMPARISON_MODELS.into_iter()
                .map(|model| observation_cell(&records, model, red_count, blue_count, true)).collect();
            paired_cells.push(json!({"red_count": red_count, "blue_count": blue_count,
                "n": paired[0]["n"], "pending": paired[0]["pending"],
                "excluded_missing_comparison": paired[0]["excluded_missing_comparison"], "models": paired}));
            let models:Vec<_>=["weighted",crate::ranking::MODEL].into_iter().map(|m| {
                let mut cell=observation_cell(&ranking_records,m,red_count,blue_count,false);
                cell["paired_periods_only"]=json!(true);
                cell["excluded_missing_comparison"]=json!(records.len()-ranking_records.len());
                cell
            }).collect();
            ranking_paired_cells.push(json!({"red_count":red_count,"blue_count":blue_count,"n":models[0]["n"],"pending":models[0]["pending"],
                "excluded_missing_comparison":records.len()-ranking_records.len(),"models":models}));
            let models:Vec<_>=["weighted",crate::conditional::MODEL].into_iter().map(|m| {
                let mut c=observation_cell(&conditional_records,m,red_count,blue_count,false);
                c["paired_periods_only"]=json!(true);c["excluded_missing_comparison"]=json!(records.len()-conditional_records.len());c
            }).collect();
            conditional_paired_cells.push(json!({"red_count":red_count,"blue_count":blue_count,"n":models[0]["n"],"pending":models[0]["pending"],"models":models}));
        }
    }
    for model in crate::pool_algorithms::MODELS {
        let shared:Vec<_>=records.iter().filter(|r|observation_pool(r,model,14,1).is_some()).cloned().collect();
        first_algorithm_issues.insert(model.into(),json!(shared.first().map(|r|&r["issue"])));
        for red_count in pool::TIER_RED_SIZES {for blue_count in pool::BLUE_SIZES {
            let models:Vec<_>=["weighted",model].into_iter().map(|name| {
                let mut cell=observation_cell(&shared,name,red_count,blue_count,false);
                cell["paired_periods_only"]=json!(true);cell["excluded_missing_comparison"]=json!(records.len()-shared.len());cell
            }).collect();
            algorithm_paired_cells.push(json!({"model":model,"red_count":red_count,"blue_count":blue_count,
                "n":models[0]["n"],"pending":models[0]["pending"],"excluded_missing_comparison":records.len()-shared.len(),"models":models}));
        }}
    }
    let periods: Vec<_> = records
        .iter()
        .map(|r| {
            json!({"issue": r["issue"],
        "pending": r["draw"].is_null(), "smallest_full_red_tier": r["smallest_full_red_tier"],
        "model_diagnostics": r["models"].as_array().unwrap().iter().map(|m| json!({"model": m["model"],
            "diagnostics": m["diagnostics"], "smallest_full_red_tier": m["smallest_full_red_tier"]})).collect::<Vec<_>>()})
        })
        .collect();
    Ok(
        json!({"kind": "saved_model_pool_observations", "actual_recommendation": true,
        "independent_timestamp_verified": false, "cells": cells, "periods": periods,
        "comparison_cells": comparison_cells, "paired_cells": paired_cells,
        "algorithm_paired_cells":algorithm_paired_cells,"first_algorithm_issues":first_algorithm_issues,
        "algorithm_policies":crate::pool_algorithms::MODELS.map(crate::pool_algorithms::policy),
        "conditional_paired_cells":conditional_paired_cells,"conditional_policy":crate::conditional::policy(),
        "first_conditional_issue":conditional_records.first().map(|r|&r["issue"]),
        "ranking_paired_cells":ranking_paired_cells,"ranking_policy":crate::ranking::policy(),
        "first_ranking_issue":ranking_records.first().map(|r|&r["issue"]),
        "comparison_policy": pool::comparison_policy(),
        "first_comparison_issue": records.iter().find(|r| r["models"].as_array().unwrap().len() >= 3).map(|r| &r["issue"]),
        "missing_tier_archives": missing_tier_archives, "errors": errors,
        "note": "只核对每日档案原始号码，每档只计实际保存且已开奖的期数；待开奖不计入比例，旧期缺档不补造。本地留档无独立时间戳证明。"}),
    )
}

fn observation_pool<'a>(record: &'a Value, model: &str, red: usize, blue: usize) -> Option<(&'a Value, &'a Value)> {
    let m = record["models"].as_array()?.iter().find(|m| m["model"] == model)?;
    let p = m["pools"].as_array()?.iter().find(|p| p["group"] == "observation"
        && p["red_count"] == red && p["blue_count"] == blue)?;
    Some((m, p))
}

fn observation_cell(records: &[Value], model: &str, red: usize, blue: usize, paired: bool) -> Value {
    let (mut n, mut pending, mut missing, mut excluded, mut full, mut five, mut blue_hits, mut joint) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut red_hits, mut prefix_sum, mut prefix_n) = (0u64, 0u64, 0u64);
    let mut issues = Vec::new();
    for record in records {
        if paired && !pool::COMPARISON_MODELS.iter().all(|m| observation_pool(record, m, red, blue).is_some()) {
            excluded += 1;
            continue;
        }
        let Some((m, p)) = observation_pool(record, model, red, blue) else { missing += 1; continue; };
        let Some(hits) = p["red_hits"].as_u64() else { pending += 1; continue; };
        n += 1;
        red_hits += hits;
        full += usize::from(p["full_red"] == true);
        five += usize::from(hits >= 5);
        blue_hits += usize::from(p["blue_hit"] == true);
        joint += usize::from(p["full_draw"] == true);
        issues.push(record["issue"].clone());
        if let Some(prefix) = m["diagnostics"]["minimum_full_red_prefix"].as_u64() {
            prefix_sum += prefix;
            prefix_n += 1;
        }
    }
    let rate = (n > 0).then(|| full as f64 / n as f64);
    json!({"model": model, "label": if model == crate::conditional::MODEL {crate::conditional::LABEL} else if model == crate::ranking::MODEL {crate::ranking::LABEL} else if let Some(i)=crate::pool_algorithms::MODELS.iter().position(|m|*m==model) {crate::pool_algorithms::LABELS[i]} else {pool::COMPARISON_LABELS[pool::COMPARISON_MODELS.iter().position(|m| *m == model).unwrap()]},
        "red_count": red, "blue_count": blue, "n": n, "pending": pending, "missing": missing,
        "excluded_missing_comparison": excluded, "paired_periods_only": paired, "scored_issues": issues,
        "full_red_periods": full, "full_red_rate": rate,
        "full_red_wilson95": (n > 0).then(|| {let (l,h)=crate::stats::wilson(full as f64,n as f64,1.959963984540054);[l,h]}),
        "full_red_rate_excess": rate.map(|r| r - pool::uniform_full_red(red)),
        "at_least_5_red_periods": five,
        "mean_red_hits": (n > 0).then(|| red_hits as f64 / n as f64),
        "mean_missed_reds": (n > 0).then(|| 6.0 - red_hits as f64 / n as f64),
        "mean_minimum_full_red_prefix": (prefix_n > 0).then(|| prefix_sum as f64 / prefix_n as f64),
        "diagnostic_periods": prefix_n, "blue_covered_periods": blue_hits, "full_draw_periods": joint,
        "full_draw_rate": (n > 0).then(|| joint as f64 / n as f64),
        "uniform_full_red_rate": pool::uniform_full_red(red)})
}

fn reason_text(reason: &str) -> &'static str {
    match reason {
        "ticketed" => "已进入完整票组",
        "replaced_by_rescue" => "融合前列号码被救援名额替换",
        "not_selected_by_ticket_objective" => "在候选池内，但构票与重合约束未选入",
        "below_candidate_cutoff" => "融合排名未进入本方案候选范围",
        "structural_unselected" => "确定性分散或组合构票未选入，无专家评分",
        _ => "旧记录未保存当时模型证据，无法恢复筛选原因",
    }
}

fn prize(red: usize, blue: bool) -> Option<u8> {
    match (red, blue) {
        (6, true) => Some(1),
        (6, false) => Some(2),
        (5, true) => Some(3),
        (5, false) | (4, true) => Some(4),
        (4, false) | (3, true) => Some(5),
        (_, true) => Some(6),
        _ => None,
    }
}

fn review_strategy(saved: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let tickets: Vec<Ticket> =
        serde_json::from_value(saved["tickets"].clone()).map_err(|e| format!("票组无效：{e}"))?;
    if tickets.is_empty() {
        return Err("票组不能为空".into());
    }
    let mut red_count = [0u32; 33];
    let mut blue_count = [0u32; 16];
    let mut unique_tickets = BTreeSet::new();
    let mut ticket_rows = Vec::new();
    let mut prize_tickets = 0;
    let mut fixed_prize = 0;
    let mut floating = 0;
    let mut conditional = 0;
    let mut best_red = 0;
    for (index, ticket) in tickets.iter().enumerate() {
        let checked = Draw::from_parts(0, 20000101, ticket.red, ticket.blue)?;
        unique_tickets.insert((checked.red, checked.blue));
        for n in ticket.red {
            red_count[n as usize - 1] += 1;
        }
        blue_count[ticket.blue as usize - 1] += 1;
        let hit_numbers = draw.map(|d| {
            ticket
                .red
                .iter()
                .filter(|n| d.red.contains(n))
                .copied()
                .collect::<Vec<_>>()
        });
        let hits = hit_numbers.as_ref().map(Vec::len);
        let blue_hit = draw.map(|d| d.blue == ticket.blue);
        let level = hits.and_then(|h| prize(h, blue_hit.unwrap_or(false)));
        let fixed = match level {
            Some(3) => 3000,
            Some(4) => 200,
            Some(5) => 10,
            Some(6) => 5,
            _ => 0,
        };
        let needs_special =
            draw.is_some_and(|d| d.issue >= 26014) && hits == Some(3) && blue_hit == Some(false);
        if let Some(h) = hits {
            best_red = best_red.max(h);
        }
        prize_tickets += usize::from(level.is_some());
        fixed_prize += fixed;
        floating += usize::from(matches!(level, Some(1 | 2)));
        conditional += usize::from(needs_special);
        ticket_rows.push(json!({"index": index + 1, "red": ticket.red.to_vec(), "blue": ticket.blue,
            "hit_reds": hit_numbers, "red_hits": hits, "blue_hit": blue_hit, "prize_level": level,
            "fixed_prize": if draw.is_some() { json!(fixed) } else { Value::Null },
            "special_prize_candidate": if draw.is_some() { json!(needs_special) } else { Value::Null },
            "result": if draw.is_none() { "pending" } else if level.is_some() { "prize" } else if needs_special { "special_prize_unverified" } else { "no_prize" }}));
    }
    let reds: Vec<u8> = (1..=33)
        .filter(|n| red_count[*n as usize - 1] > 0)
        .collect();
    let blues: Vec<u8> = (1..=16)
        .filter(|n| blue_count[*n as usize - 1] > 0)
        .collect();
    let covered = draw.map(|d| {
        d.red
            .iter()
            .filter(|n| reds.contains(n))
            .copied()
            .collect::<Vec<_>>()
    });
    let missed = draw.map(|d| {
        d.red
            .iter()
            .filter(|n| !reds.contains(n))
            .copied()
            .collect::<Vec<_>>()
    });
    let mut rows = Vec::new();
    for (color, size) in [("red", 33), ("blue", 16)] {
        for n in 1..=size {
            let evidence = saved["evidence"]
                .as_array()
                .and_then(|a| a.iter().find(|r| r["color"] == color && r["number"] == n));
            let exposure = if color == "red" {
                red_count[n - 1]
            } else {
                blue_count[n - 1]
            };
            let mut row = evidence.cloned().unwrap_or_else(|| {
                json!({"color": color, "number": n,
                "reason": if saved["structural"] == true {
                    if exposure > 0 { "ticketed" } else { "structural_unselected" }
                } else { "historical_evidence_unavailable" }})
            });
            let hit = draw.map(|d| {
                if color == "red" {
                    d.red.contains(&(n as u8))
                } else {
                    d.blue == n as u8
                }
            });
            row["ticket_exposure"] = json!(exposure);
            row["draw_hit"] = json!(hit);
            row["reason_text"] = json!(reason_text(row["reason"].as_str().unwrap_or("")));
            row["recommendation_result"] = json!(match (hit, exposure > 0) {
                (None, _) => "pending",
                (Some(true), true) => "covered_winner",
                (Some(true), false) => "uncovered_winner",
                (Some(false), true) => "recommended_nonwinner",
                _ => "not_selected_nonwinner",
            });
            rows.push(row);
        }
    }
    let mut funnel = saved["funnel"].as_array().cloned().unwrap_or_default();
    funnel.push(json!({"stage": "final_union", "color": "red", "numbers": reds}));
    for stage in &mut funnel {
        let nums = stage["numbers"].as_array().ok_or("号码漏斗无效")?;
        let hits = draw.map(|d| {
            nums.iter()
                .filter(|n| n.as_u64().is_some_and(|n| d.red.contains(&(n as u8))))
                .cloned()
                .collect::<Vec<_>>()
        });
        stage["size"] = json!(nums.len());
        stage["draw_hits"] = json!(hits.as_ref().map(Vec::len));
        stage["hit_numbers"] = json!(hits);
    }
    let mut result = json!({"title": saved["title"], "rationale": saved["rationale"],
        "mode": saved["mode"], "ranking": saved["ranking"], "coverage_optimization": saved["coverage_optimization"],
        "efficiency": saved["efficiency"], "structural": saved["structural"],
        "red_weights": saved["red_weights"], "blue_weights": saved["blue_weights"],
        "summary": {"bets": tickets.len(), "cost": tickets.len() * 2, "unique_reds": reds, "unique_blues": blues,
            "red_coverage": reds.len(), "blue_coverage": blues.len(),
            "total_red_positions": tickets.len() * 6, "repeated_red_positions": tickets.len() * 6 - reds.len(),
            "repeated_blue_positions": tickets.len() - blues.len(), "duplicate_tickets": tickets.len() - unique_tickets.len(),
            "max_red_exposure": red_count.iter().max(), "max_blue_exposure": blue_count.iter().max(),
            "uniform_expected_red_hits": reds.len() as f64 * 6.0 / 33.0,
            "covered_reds": covered, "missed_reds": missed, "covered_red_count": covered.as_ref().map(Vec::len),
            "blue_covered": draw.map(|d| blues.contains(&d.blue)), "max_ticket_red_hits": draw.map(|_| best_red),
            "prize_tickets": draw.map(|_| prize_tickets), "fixed_prize": draw.map(|_| fixed_prize),
            "floating_prize_tickets": draw.map(|_| floating), "special_prize_candidates": draw.map(|_| conditional)},
        "numbers": rows, "tickets": ticket_rows, "funnel": funnel,
        "prize_rules_url": PRIZE_RULES_URL,
        "explanation_limit": "专家评分是排序依据，不是中奖概率。合并覆盖不等于单注中奖。固定奖金不含一、二等奖、派奖及福运奖；3红0蓝是否中福运奖须核对当期是否执行特别规定。" });
    enrich_efficiency(&mut result);
    Ok(result)
}

/// Add comparison metrics from saved tickets and saved hits, without re-pricing
/// persisted reviews or rewriting their bytes.
fn enrich_efficiency(review: &mut Value) {
    let Some(tickets) = review["tickets"].as_array() else {
        return;
    };
    if tickets.is_empty() {
        return;
    }
    let reds: BTreeSet<_> = tickets
        .iter()
        .flat_map(|t| t["red"].as_array().into_iter().flatten())
        .filter_map(Value::as_u64)
        .collect();
    let blues: BTreeSet<_> = tickets.iter().filter_map(|t| t["blue"].as_u64()).collect();
    let all_ticket_reds: Vec<_> = reds
        .iter()
        .copied()
        .filter(|n| {
            tickets
                .iter()
                .all(|t| t["red"].as_array().is_some_and(|r| r.contains(&json!(n))))
        })
        .collect();
    let hits: Option<Vec<_>> = tickets.iter().map(|t| t["red_hits"].as_u64()).collect();
    let blue_hits: Option<Vec<_>> = tickets.iter().map(|t| t["blue_hit"].as_bool()).collect();
    let covered = review["summary"]["covered_red_count"].as_f64();
    let blue_covered = review["summary"]["blue_covered"].as_bool();
    let mut histogram = BTreeMap::<u64, usize>::new();
    if let Some(hits) = &hits {
        for &h in hits {
            *histogram.entry(h).or_default() += 1;
        }
    }
    let mut missed = BTreeMap::<String, usize>::new();
    if let Some(numbers) = review["numbers"].as_array() {
        for row in numbers
            .iter()
            .filter(|r| r["color"] == "red" && r["draw_hit"] == true && r["ticket_exposure"] == 0)
        {
            *missed
                .entry(
                    row["reason"]
                        .as_str()
                        .unwrap_or("historical_evidence_unavailable")
                        .to_string(),
                )
                .or_default() += 1;
        }
    }
    let bets = tickets.len();
    let cost = bets as f64 * 2.0;
    let summary = &mut review["summary"];
    summary["red_coverage_per_yuan"] = json!(reds.len() as f64 / cost);
    summary["blue_coverage_per_yuan"] = json!(blues.len() as f64 / cost);
    summary["minimum_union_bets"] = json!(reds.len().div_ceil(6).max(blues.len()));
    summary["all_ticket_reds"] = json!(all_ticket_reds);
    summary["uniform_expected_blue_coverage"] = json!(blues.len() as f64 / 16.0);
    summary["uniform_expected_ticket_red_hits"] = json!(36.0 / 33.0);
    summary["uniform_expected_blue_hit_bets"] = json!(bets as f64 / 16.0);
    summary["red_coverage_excess"] = json!(covered.map(|h| h - reds.len() as f64 * 6.0 / 33.0));
    summary["blue_coverage_excess"] =
        json!(blue_covered.map(|h| f64::from(h) - blues.len() as f64 / 16.0));
    summary["mean_ticket_red_hits"] = json!(hits
        .as_ref()
        .map(|h| h.iter().sum::<u64>() as f64 / bets as f64));
    summary["blue_hit_bets"] = json!(blue_hits.map(|h| h.iter().filter(|&&v| v).count()));
    summary["red_hit_histogram"] = if hits.is_some() {
        json!(histogram)
    } else {
        Value::Null
    };
    summary["missed_reds_by_reason"] = if hits.is_some() {
        json!(missed)
    } else {
        Value::Null
    };
}

fn compact_preview(review: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let original: Vec<Ticket> =
        serde_json::from_value(review["tickets"].clone()).map_err(|e| e.to_string())?;
    let reds: Vec<_> = original
        .iter()
        .flat_map(|t| t.red)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let blues: Vec<_> = original
        .iter()
        .map(|t| t.blue)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let tickets = efficiency::partition_cover(&reds, &blues)?;
    if tickets.len() >= original.len() {
        return Ok(Value::Null);
    }
    let saved = json!({"title": "同覆盖减注 · 结构试算", "structural": true,
        "rationale": "分别保留原红球并集和蓝球并集，达到最少单注数；红蓝配对及逐注命中可能变化。",
        "tickets": tickets.iter().map(|t| json!({"red": t.red, "blue": t.blue})).collect::<Vec<_>>()});
    let mut preview = review_strategy(&saved, draw)?;
    preview["kind"] = json!(if draw.is_some() {
        "post_draw_structural_replay"
    } else {
        "unpublished_structural_preview"
    });
    preview["actual_recommendation"] = json!(false);
    preview["saved_bets"] = json!(original.len() - tickets.len());
    preview["saved_cost"] = json!((original.len() - tickets.len()) * 2);
    preview["same_red_union"] = json!(true);
    preview["same_blue_union"] = json!(true);
    preview["note"] = json!("结构试算未在当时发布，不计入实际推荐成绩。仅分别保留红球与蓝球并集，不保留原红蓝配对、中奖注数或奖金。");
    Ok(preview)
}

fn archive_review(envelope: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut strategies = BTreeMap::new();
    for (name, saved) in envelope["payload"]["strategies"]
        .as_object()
        .ok_or("档案缺少策略")?
    {
        strategies.insert(name.clone(), review_strategy(saved, draw)?);
    }
    Ok(
        json!({"schema_version": 1, "archive_sha256": envelope["sha256"], "issue": envelope["payload"]["issue"],
        "draw": draw.map(draw_json), "outcome_sha256": draw.map(|d| data_sha256(std::slice::from_ref(d))),
        "strategies": strategies, "number_pools": pool::review(&envelope["payload"]["number_pools"], draw)?}),
    )
}

fn checked_review(
    root: &Path,
    issue: &str,
    id: &str,
    envelope: &Value,
    draw: Option<&Draw>,
) -> Result<(Value, bool), String> {
    let path = review_path(root, issue, id);
    if path.exists() {
        let review = read_checked(&path)?;
        if review["payload"]["archive_sha256"] != envelope["sha256"]
            || review["payload"]["outcome_sha256"]
                != json!(draw.map(|d| data_sha256(std::slice::from_ref(d))))
        {
            return Err("原始推荐或开奖结果与已保存复盘不一致，拒绝覆盖".into());
        }
        return Ok((review["payload"].clone(), true));
    }
    Ok((archive_review(envelope, draw)?, false))
}

#[derive(Default, Serialize)]
pub struct Settlement {
    pub settled: usize,
    pub errors: Vec<String>,
}

pub fn settle_archives(root: &Path, draws: &[Draw]) -> Settlement {
    let mut outcome = Settlement::default();
    for issue in all_issues(root) {
        let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
        for (id, path) in archive_paths(root, &issue) {
            let result = (|| -> Result<usize, String> {
                let envelope = read_checked(&path)?;
                validate_archive(&envelope["payload"], &issue, draws)?;
                let (mut review, persisted) = checked_review(root, &issue, &id, &envelope, draw)?;
                if draw.is_none() || persisted {
                    return Ok(0);
                }
                review["reviewed_at_unix"] = json!(now_unix());
                let (written, created) = write_new(&review_path(root, &issue, &id), review)?;
                if written["payload"]["archive_sha256"] != envelope["sha256"]
                    || written["payload"]["outcome_sha256"]
                        != json!(draw.map(|d| data_sha256(std::slice::from_ref(d))))
                {
                    return Err("已有复盘与推荐或开奖不一致，拒绝覆盖".into());
                }
                Ok(usize::from(created))
            })();
            match result {
                Ok(count) => outcome.settled += count,
                Err(error) => outcome.errors.push(format!("{issue}/{id}：{error}")),
            }
        }
    }
    outcome
}

fn all_issues(root: &Path) -> BTreeSet<String> {
    let mut issues = BTreeSet::new();
    for dir in [root.join("daily"), root.join("interactive")] {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = if path.is_dir() {
                    path.file_name()
                } else {
                    path.file_stem()
                };
                if let Some(issue) = name.and_then(|s| s.to_str()).filter(|s| valid_issue(s)) {
                    issues.insert(issue.to_string());
                }
            }
        }
    }
    issues
}

pub fn issue_index(root: &Path, draws: &[Draw], legacy: &[Value], limit: usize) -> Value {
    let mut issues = all_issues(root);
    issues.extend(draws.iter().rev().take(limit).map(|d| fmt_issue(d.issue)));
    issues.extend(
        legacy
            .iter()
            .filter_map(|v| v["target"].as_str())
            .filter(|s| valid_issue(s))
            .map(str::to_string),
    );
    let rows: Vec<_> = issues.iter().rev().take(limit).map(|issue| {
        let paths = archive_paths(root, issue);
        let draw = draws.iter().find(|d| fmt_issue(d.issue) == *issue);
        json!({"issue": issue, "draw_date": draw.map(|d| fmt_date(d.date)), "status": if draw.is_some() { "settled" } else { "pending" },
            "archives": paths.len(), "saved_plans": legacy.iter().filter(|v| v["target"] == *issue).count(),
            "missing_auto_archive": !paths.iter().any(|(id, _)| id == "daily")})
    }).collect();
    json!({"issues": rows, "presets": PRESETS, "independent_timestamp_verified": false})
}

pub fn issue_detail(
    root: &Path,
    draws: &[Draw],
    legacy: &[Value],
    issue: &str,
    archive: Option<&str>,
    strategy_name: Option<&str>,
) -> Result<Value, String> {
    if !valid_issue(issue) {
        return Err("期号需要 5 位数字".into());
    }
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let paths = archive_paths(root, issue);
    let mut sources = Vec::new();
    for (id, path) in &paths {
        let e = read_checked(path)?;
        sources.push(json!({"id": id, "kind": e["payload"]["kind"], "published_at_unix": e["payload"]["published_at_unix"]}));
    }
    for record in legacy.iter().filter(|r| r["target"] == issue) {
        sources.push(json!({"id": format!("saved:{}", record["file"].as_str().unwrap_or("")), "kind": "saved", "published_at_unix": record["published_at"]}));
    }
    let mut response = json!({"issue": issue, "draw": draw.map(draw_json), "sources": sources});
    if sources.is_empty() {
        response["missing"] = json!(true);
        response["message"] =
            json!("本期没有实际推荐档案，历史缺档不补造。事后回放请到回测页面查看。");
        return Ok(response);
    }
    let id = archive
        .or_else(|| sources[0]["id"].as_str())
        .ok_or("缺少档案来源")?;
    let (mut result, provenance) = if let Some((_, path)) = paths.iter().find(|(key, _)| key == id)
    {
        let envelope = read_checked(path)?;
        let p = &envelope["payload"];
        validate_archive(p, issue, draws)?;
        let (mut review, persisted) = checked_review(root, issue, id, &envelope, draw)?;
        // Derive new diagnostics from original saved rankings without rewriting
        // the original archive or an already persisted settlement.
        review["number_pools"] = pool::review(&p["number_pools"], draw)?;
        let raw = if id == "daily" {
            format!("daily/{issue}.json")
        } else {
            format!("interactive/{issue}/{}.json", &id[12..])
        };
        let filename = id.strip_prefix("interactive:").unwrap_or(id);
        (
            review,
            json!({"kind": p["kind"], "sha256": envelope["sha256"], "published_at_unix": p["published_at_unix"],
            "data_cutoff": p["data_cutoff"], "operational_cutoff_unix": p["operational_cutoff_unix"],
            "known_outcome_at_generation": p["known_outcome_at_generation"], "independent_timestamp_verified": false,
            "saved_after_scheduled_draw": saved_after_draw(p["published_at_unix"].as_u64(), draw),
            "schedule_verified": false, "evidence_available": true, "review_persisted": persisted,
            "raw_file": raw, "review_file": if persisted { Some(format!("reviews/{issue}/{filename}.json")) } else { None }}),
        )
    } else if let Some(record) = legacy.iter().find(|r| {
        r["target"] == issue && format!("saved:{}", r["file"].as_str().unwrap_or("")) == id
    }) {
        if let Some(error) = record["error"].as_str() {
            return Err(error.to_string());
        }
        let saved = json!({"title": record["title"], "tickets": record["tickets"], "rationale": record["detail"]});
        (
            json!({"strategies": {"saved": review_strategy(&saved, draw)?}}),
            json!({"kind": "saved", "evidence_available": false,
            "published_at_unix": record["published_at"], "data_cutoff": record["cutoff_issue"],
            "known_outcome_at_generation": record["cutoff_issue"].as_str().map(|c| c >= issue).unwrap_or(true),
            "saved_after_scheduled_draw": saved_after_draw(record["published_at"].as_u64(), draw),
            "independent_timestamp_verified": false, "review_persisted": false}),
        )
    } else {
        return Err("本期不存在该档案".into());
    };
    response["number_pools"] = result["number_pools"].clone();
    let variants = result["strategies"].as_object_mut().ok_or("档案缺少策略")?;
    for review in variants.values_mut() {
        enrich_efficiency(review);
    }
    let selected = strategy_name.unwrap_or(if variants.contains_key("frozen") {
        "frozen"
    } else {
        variants.keys().next().ok_or("档案策略为空")?
    });
    let detail = variants.get(selected).ok_or("策略不存在")?;
    response["archive_id"] = json!(id);
    response["provenance"] = provenance;
    response["strategy"] = json!(selected);
    response["strategies"] = json!(variants.iter().map(|(name, value)| json!({"name": name, "title": value["title"], "mode": value["mode"], "ranking": value["ranking"], "coverage_optimization": value["coverage_optimization"], "summary": value["summary"]})).collect::<Vec<_>>());
    response["detail"] = detail.clone();
    response["compact_preview"] = compact_preview(detail, draw)?;
    Ok(response)
}

pub fn artifact(root: &Path, raw: &str) -> Result<Vec<u8>, String> {
    let parts: Vec<_> = raw.split('/').collect();
    let valid = match parts.as_slice() {
        ["daily", file] => file.strip_suffix(".json").is_some_and(valid_issue),
        ["interactive", issue, file] => {
            valid_issue(issue) && file.strip_suffix(".json").is_some_and(valid_id)
        }
        ["reviews", issue, file] => {
            valid_issue(issue)
                && file
                    .strip_suffix(".json")
                    .is_some_and(|s| s == "daily" || valid_id(s))
        }
        _ => false,
    };
    if !valid {
        return Err("档案路径无效".into());
    }
    let base = fs::canonicalize(root).map_err(|e| e.to_string())?;
    let path = fs::canonicalize(root.join(raw)).map_err(|e| e.to_string())?;
    if !path.starts_with(base) {
        return Err("档案路径越界".into());
    }
    read_checked(&path)?;
    fs::read(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::synth_draws;
    use std::cell::Cell;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(std::env::temp_dir().join(format!(
                "ssq-recommendations-{}-{stamp}",
                std::process::id()
            )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn build(name: &str, prepared: &Prepared) -> Plan {
        plan::build(
            &prepared.forecast,
            &plan::resolve(Some(name), None, None, None, None, None, None, None, None).unwrap(),
        )
        .unwrap()
    }
    fn next(draws: &[Draw]) -> String {
        fmt_issue(draws.last().unwrap().issue + 1)
    }
    fn append_outcome(draws: &mut Vec<Draw>) {
        let last = draws.last().unwrap();
        draws.push(Draw::from_parts(last.issue + 1, 20030901, [1, 2, 3, 4, 5, 6], 7).unwrap());
    }

    #[test]
    fn algorithm_observations_exclude_missing_models_and_score_only_saved_pools() {
        let temp=Temp::new();let mut draws=synth_draws(220,9);
        let p=prepare_forecast(&draws).unwrap();let old_issue=next(&draws);
        let mut payload=base_payload(&p,&old_issue,"daily",10,Some(100));
        payload["number_pools"]=pool::saved_current(&p,&draws);
        payload["number_pools"].as_object_mut().unwrap().remove("algorithm_models");
        payload["strategies"]=json!({"frozen":strategy(&build("frozen",&p),&p)});
        let old_path=temp.0.join(format!("daily/{old_issue}.json"));write_new(&old_path,payload).unwrap();
        let old=fs::read(&old_path).unwrap();
        let existing=freeze_daily_with_clock(&temp.0,&draws,&old_issue,100,||10).unwrap();
        assert_eq!(existing["comparison_models"],5);assert_eq!(existing["saved_models"].as_array().unwrap().len(),5);
        append_outcome(&mut draws);let issue=next(&draws);
        let frozen=freeze_daily_with_clock(&temp.0,&draws,&issue,100,||10).unwrap();
        assert_eq!(frozen["comparison_models"],8);assert_eq!(frozen["comparison_number_pools"],126);
        let pending=pool_observations(&temp.0,&draws).unwrap();
        assert_eq!(pending["algorithm_paired_cells"].as_array().unwrap().len(),54);
        for c in pending["algorithm_paired_cells"].as_array().unwrap() {
            assert_eq!(c["n"],0);assert_eq!(c["pending"],1);assert_eq!(c["excluded_missing_comparison"],1);
            assert!(c["models"][1]["full_red_rate"].is_null());assert!(c["models"][1]["full_draw_rate"].is_null());
        }
        let path=temp.0.join(format!("daily/{issue}.json"));let original=fs::read(&path).unwrap();
        let raw=read_checked(&path).unwrap();
        let first=&raw["payload"]["number_pools"]["algorithm_models"][0];
        let reds:Vec<u8>=first["red_ranking"].as_array().unwrap()[..6].iter().map(|n|n.as_u64().unwrap()as u8).collect();
        let blue=raw["payload"]["number_pools"]["blue_ranking"][0].as_u64().unwrap()as u8;
        draws.push(Draw::from_parts(issue.parse().unwrap(),20030901,reds.try_into().unwrap(),blue).unwrap());
        let scored=pool_observations(&temp.0,&draws).unwrap();
        assert_eq!(scored["errors"],json!([]));
        for c in scored["algorithm_paired_cells"].as_array().unwrap() {
            assert_eq!(c["n"],1);assert_eq!(c["pending"],0);assert_eq!(c["models"][1]["scored_issues"],json!([issue]));
            if c["model"]==crate::pool_algorithms::MODELS[0] {assert_eq!(c["models"][1]["full_draw_periods"],1);}
        }
        assert_eq!(fs::read(&path).unwrap(),original);assert_eq!(fs::read(&old_path).unwrap(),old);
    }

    #[test]
    fn v3_daily_stays_sealed_and_new_daily_binds_calibrated_tickets_to_saved_pool() {
        let temp=Temp::new();
        let mut draws=synth_draws(220,9);
        let prepared=prepare_forecast(&draws).unwrap();
        let issue=next(&draws);
        let mut payload=base_payload(&prepared,&issue,"daily",10,Some(100));
        payload["number_pools"]=pool::saved(&prepared);
        payload["strategies"]=json!({"frozen":strategy(&build("frozen",&prepared),&prepared)});
        let path=temp.0.join(format!("daily/{issue}.json"));
        write_new(&path,payload).unwrap();
        let old=fs::read(&path).unwrap();
        assert_eq!(freeze_daily_with_clock(&temp.0,&draws,&issue,100,||10).unwrap()["comparison_models"],3);
        assert_eq!(fs::read(&path).unwrap(),old);
        append_outcome(&mut draws);
        let new_issue=next(&draws);
        let result=freeze_daily_with_clock(&temp.0,&draws,&new_issue,100,||10).unwrap();
        assert_eq!(result["strategies"],20);
        assert_eq!(result["comparison_models"],8);
        assert_eq!(result["comparison_number_pools"],126);
        let raw=read_checked(&temp.0.join(format!("daily/{new_issue}.json"))).unwrap();
        let p=&raw["payload"]["strategies"]["calibrated14"];
        let saved=&raw["payload"]["number_pools"]["comparison_models"][2];
        assert_eq!(p["ranking"],*saved);
        assert!(p["red_weights"].is_null());
        let reds:BTreeSet<_>=p["tickets"].as_array().unwrap().iter().flat_map(|t|t["red"].as_array().unwrap().iter().map(|n|n.as_u64().unwrap())).collect();
        assert_eq!(reds,saved["red_ranking"].as_array().unwrap()[..14].iter().map(|n|n.as_u64().unwrap()).collect());
        let equal = &raw["payload"]["strategies"]["equal24"];
        let equal_pool = &raw["payload"]["number_pools"]["comparison_models"][0];
        assert_eq!(equal["ranking"]["red_ranking"], equal_pool["red_ranking"]);
        assert_eq!(equal["ranking"]["red_scores"], equal_pool["red_scores"]);
        assert_eq!(equal["red_weights"], equal_pool["red_weights"]);
        assert_eq!(equal["coverage_optimization"]["ranking"], "equal");
        for n in equal_pool["red_ranking"].as_array().unwrap()[..24].iter() {
            assert_eq!(equal["tickets"].as_array().unwrap().iter().filter(|t|t["red"].as_array().unwrap().contains(n)).count(), 2);
        }
        let observations=pool_observations(&temp.0,&draws).unwrap();
        for cell in observations["ranking_paired_cells"].as_array().unwrap() {
            assert_eq!(cell["n"],0);assert_eq!(cell["pending"],1);
            assert_eq!(cell["excluded_missing_comparison"],1);
            assert_eq!(cell["models"][1]["paired_periods_only"],true);
            assert!(cell["models"][1]["full_red_rate"].is_null());
        }
        assert_eq!(fs::read(&path).unwrap(),old);
    }

    #[test]
    fn daily_archives_all_presets_and_pending_metrics_remain_null() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        let path = temp.0.join(format!("daily/{issue}.json"));
        let original = fs::read(&path).unwrap();
        let archive = read_checked(&path).unwrap();
        assert_eq!(
            archive["payload"]["efficiency"]["version"],
            efficiency::VERSION
        );
        let variants = archive["payload"]["strategies"].as_object().unwrap();
        assert_eq!(variants.len(), PRESETS.len() + efficiency::PROFILE_COUNT);
        for name in variants.keys() {
            let view = issue_detail(&temp.0, &draws, &[], &issue, None, Some(name)).unwrap();
            assert_eq!(view["detail"]["numbers"].as_array().unwrap().len(), 49);
            assert!(view["detail"]["summary"]["covered_red_count"].is_null());
            assert!(view["detail"]["summary"]["prize_tickets"].is_null());
            assert!(view["detail"]["summary"]["fixed_prize"].is_null());
            assert!(view["detail"]["tickets"][0]["red_hits"].is_null());
            assert!(view["detail"]["numbers"][0]["draw_hit"].is_null());
            assert!(view["detail"]["funnel"][0]["draw_hits"].is_null());
            for field in [
                "red_coverage_excess",
                "blue_coverage_excess",
                "mean_ticket_red_hits",
                "blue_hit_bets",
                "red_hit_histogram",
                "missed_reds_by_reason",
            ] {
                assert!(view["detail"]["summary"][field].is_null(), "{name}/{field}");
            }
            if name.starts_with("efficiency:uniform:") {
                assert_eq!(view["detail"]["structural"], true);
                assert!(view["detail"]["red_weights"].is_null());
                assert!(view["detail"]["numbers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|n| n["fusion_score"].is_null()));
            }
        }
        assert_eq!(settle_archives(&temp.0, &draws).settled, 0);
        assert!(!temp.0.join("reviews").exists());
        assert_eq!(
            freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 200).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn old_daily_archives_and_saved_reviews_are_not_augmented_or_rewritten() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let issue = next(&draws);
        let prepared = prepare_forecast(&draws).unwrap();
        let mut payload = base_payload(&prepared, &issue, "daily", 10, Some(100));
        payload["strategies"] = json!(PRESETS
            .into_iter()
            .filter(|name| !["balanced14","calibrated14","coverage24","equal24"].contains(name))
            .map(|name| (name, strategy(&build(name, &prepared), &prepared)))
            .collect::<BTreeMap<_, _>>());
        let path = temp.0.join(format!("daily/{issue}.json"));
        write_new(&path, payload).unwrap();
        let original = fs::read(&path).unwrap();
        assert_eq!(
            freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 200).unwrap()["strategies"],
            7
        );
        append_outcome(&mut draws);
        assert_eq!(settle_archives(&temp.0, &draws).settled, 1);
        // Simulate a persisted review created before comparison metrics existed.
        let review_path = review_path(&temp.0, &issue, "daily");
        let mut saved = read_checked(&review_path).unwrap()["payload"].clone();
        for value in saved["strategies"].as_object_mut().unwrap().values_mut() {
            value["summary"]
                .as_object_mut()
                .unwrap()
                .remove("mean_ticket_red_hits");
        }
        fs::write(
            &review_path,
            serde_json::to_vec(&json!({"sha256": digest(&saved).unwrap(), "payload": saved}))
                .unwrap(),
        )
        .unwrap();
        let old_review = fs::read(&review_path).unwrap();
        let view = issue_detail(&temp.0, &draws, &[], &issue, None, Some("frozen")).unwrap();
        assert_eq!(view["strategies"].as_array().unwrap().len(), 7);
        assert!(view["number_pools"].is_null());
        assert!(view["detail"]["summary"]["mean_ticket_red_hits"].is_number());
        assert_eq!(settle_archives(&temp.0, &draws).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(fs::read(&review_path).unwrap(), old_review);
    }

    #[test]
    fn small_pools_settle_from_the_original_numbers_and_do_not_refit_after_later_draws() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let issue = next(&draws);
        let frozen = freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        assert_eq!(frozen["number_pools"], 27);
        let pending = issue_detail(&temp.0, &draws, &[], &issue, None, None).unwrap();
        let pools = pending["number_pools"]["pools"].as_array().unwrap();
        assert_eq!(pools.len(), 27);
        assert!(pools
            .iter()
            .all(|p| p["full_draw"].is_null() && p["red_hits"].is_null()));
        let winning_reds: Vec<u8> = serde_json::from_value(pools[0]["reds"].clone()).unwrap();
        draws.push(
            Draw::from_parts(
                issue.parse().unwrap(),
                20030901,
                winning_reds[..6].try_into().unwrap(),
                pools[0]["blues"][0].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        assert_eq!(settle_archives(&temp.0, &draws).settled, 1);
        let scored =
            issue_detail(&temp.0, &draws, &[], &issue, None, None).unwrap()["number_pools"].clone();
        assert!(scored["pools"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["full_draw"] == true));
        for (a, b) in pools.iter().zip(scored["pools"].as_array().unwrap()) {
            assert_eq!(a["reds"], b["reds"]);
            assert_eq!(a["blues"], b["blues"]);
        }
        draws.push(
            Draw::from_parts(
                draws.last().unwrap().issue + 1,
                20030903,
                [7, 8, 9, 10, 11, 12],
                9,
            )
            .unwrap(),
        );
        assert_eq!(
            issue_detail(&temp.0, &draws, &[], &issue, None, None).unwrap()["number_pools"],
            scored
        );
        assert_eq!(settle_archives(&temp.0, &draws).settled, 0);
    }

    #[test]
    fn observation_rates_use_saved_settled_tiers_and_leave_pending_rates_null() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let empty = pool_observations(&temp.0, &draws).unwrap();
        assert!(empty["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["n"] == 0 && c["full_red_rate"].is_null()));
        assert!(!temp.0.exists());
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        let archive_path = temp.0.join(format!("daily/{issue}.json"));
        let original = fs::read(&archive_path).unwrap();
        let pending = pool_observations(&temp.0, &draws).unwrap();
        assert_eq!(pending["periods"].as_array().unwrap().len(), 1);
        assert!(pending["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["n"] == 0 && c["pending"] == 1 && c["full_red_rate"].is_null()));
        let archive = read_checked(&archive_path).unwrap();
        let ranks: Vec<u8> =
            serde_json::from_value(archive["payload"]["number_pools"]["red_ranking"].clone())
                .unwrap();
        let blue = archive["payload"]["number_pools"]["blue_ranking"][1]
            .as_u64()
            .unwrap() as u8;
        draws.push(
            Draw::from_parts(
                issue.parse().unwrap(),
                20030901,
                [ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[16]],
                blue,
            )
            .unwrap(),
        );
        let next_issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &next_issue, 100, || 10).unwrap();
        let actual = pool_observations(&temp.0, &draws).unwrap();
        for c in actual["cells"].as_array().unwrap() {
            assert_eq!(c["n"], 1);
            assert_eq!(c["pending"], 1);
            let full = c["red_count"].as_u64().unwrap() >= 18;
            assert_eq!(c["full_red_periods"], usize::from(full));
            assert_eq!(c["full_red_rate"], if full { 1.0 } else { 0.0 });
            assert_eq!(c["mean_missed_reds"], if full { 0.0 } else { 1.0 });
            assert_eq!(
                c["full_draw_periods"],
                usize::from(full && c["blue_count"] != 1)
            );
        }
        assert_eq!(actual["periods"][0]["smallest_full_red_tier"], 18);
        assert_eq!(fs::read(&archive_path).unwrap(), original);
        assert!(!temp.0.join("reviews").exists());
        fs::write(&archive_path, "{}").unwrap();
        let invalid = pool_observations(&temp.0, &draws).unwrap();
        assert_eq!(invalid["errors"].as_array().unwrap().len(), 1);
        assert!(invalid["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["n"] == 0 && c["pending"] == 1));
    }

    #[test]
    fn model_comparisons_use_the_same_saved_periods_and_never_backfill_v2() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let old_issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &old_issue, 100, || 10).unwrap();
        let old_path = temp.0.join(format!("daily/{old_issue}.json"));
        let mut legacy = read_checked(&old_path).unwrap()["payload"].clone();
        legacy["number_pools"]["version"] = json!(2);
        legacy["number_pools"].as_object_mut().unwrap().remove("comparison_models");
        legacy["number_pools"].as_object_mut().unwrap().remove("comparison_policy");
        fs::remove_file(&old_path).unwrap();
        write_new(&old_path, legacy).unwrap();
        let old_bytes = fs::read(&old_path).unwrap();
        let mut winner = draws.last().unwrap().clone();
        winner.issue = old_issue.parse().unwrap();
        draws.push(winner);
        let common_issue = next(&draws);
        let frozen = freeze_daily_with_clock(&temp.0, &draws, &common_issue, 100, || 10).unwrap();
        assert_eq!(frozen["comparison_models"], 8);
        assert_eq!(frozen["comparison_number_pools"], 126);
        let mut winner = draws.last().unwrap().clone();
        winner.issue = common_issue.parse().unwrap();
        draws.push(winner);
        freeze_daily_with_clock(&temp.0, &draws, &next(&draws), 100, || 10).unwrap();
        let result = pool_observations(&temp.0, &draws).unwrap();
        assert_eq!(result["errors"], json!([]));
        assert_eq!(result["first_comparison_issue"], common_issue);
        assert_eq!(result["comparison_cells"].as_array().unwrap().len(), 144);
        for cell in result["comparison_cells"].as_array().unwrap() {
            assert_eq!(cell["n"], if cell["model"] == "weighted" { 2 } else { 1 });
            assert_eq!(cell["pending"], 1);
            assert_eq!(cell["missing"], usize::from(cell["model"] != "weighted"));
        }
        for paired in result["paired_cells"].as_array().unwrap() {
            assert_eq!(paired["n"], 1);
            assert_eq!(paired["pending"], 1);
            assert_eq!(paired["excluded_missing_comparison"], 1);
            for model in paired["models"].as_array().unwrap() {
                assert_eq!(model["scored_issues"], json!([common_issue]));
                assert_eq!(model["diagnostic_periods"], 1);
            }
        }
        assert_eq!(fs::read(&old_path).unwrap(), old_bytes);
        assert!(!temp.0.join("reviews").exists());
        let again = freeze_daily_with_clock(&temp.0, &draws, &old_issue, 100, || 10).unwrap();
        assert_eq!(again["comparison_models"], 1);
        assert_eq!(again["comparison_number_pools"], 0);
        assert_eq!(fs::read(&old_path).unwrap(), old_bytes);
    }

    #[test]
    fn old_persisted_review_gets_read_only_rank_diagnostics_from_original_archive() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        let archive_path = temp.0.join(format!("daily/{issue}.json"));
        let mut payload = read_checked(&archive_path).unwrap()["payload"].clone();
        payload["number_pools"]["version"] = json!(2);
        payload["number_pools"].as_object_mut().unwrap().remove("comparison_models");
        payload["number_pools"].as_object_mut().unwrap().remove("comparison_policy");
        fs::remove_file(&archive_path).unwrap();
        write_new(&archive_path, payload).unwrap();
        append_outcome(&mut draws);
        assert_eq!(settle_archives(&temp.0, &draws).settled, 1);
        let review_path = review_path(&temp.0, &issue, "daily");
        let mut old_review = read_checked(&review_path).unwrap()["payload"].clone();
        old_review["number_pools"].as_object_mut().unwrap().remove("diagnostics");
        fs::remove_file(&review_path).unwrap();
        write_new(&review_path, old_review).unwrap();
        let archive_bytes = fs::read(&archive_path).unwrap();
        let review_bytes = fs::read(&review_path).unwrap();
        let detail = issue_detail(&temp.0, &draws, &[], &issue, Some("daily"), None).unwrap();
        assert!(detail["provenance"]["review_persisted"].as_bool().unwrap());
        assert!(detail["number_pools"]["diagnostics"]["minimum_full_red_prefix"].is_number());
        assert!(detail["number_pools"]["comparison_models"].is_null());
        assert_eq!(fs::read(&archive_path).unwrap(), archive_bytes);
        assert_eq!(fs::read(&review_path).unwrap(), review_bytes);
    }

    #[test]
    fn v1_daily_archives_do_not_gain_observation_tiers_or_change_bytes() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        let path = temp.0.join(format!("daily/{issue}.json"));
        let mut payload = read_checked(&path).unwrap()["payload"].clone();
        payload["number_pools"]["version"] = json!(1);
        let pools = payload["number_pools"]["pools"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["group"] == "legacy")
            .cloned()
            .collect::<Vec<_>>();
        payload["number_pools"]["pools"] = json!(pools);
        payload["number_pools"]
            .as_object_mut()
            .unwrap()
            .remove("observation_tiers");
        fs::remove_file(&path).unwrap();
        write_new(&path, payload).unwrap();
        let original = fs::read(&path).unwrap();
        assert_eq!(
            freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap()["number_pools"],
            9
        );
        let actual = pool_observations(&temp.0, &draws).unwrap();
        assert_eq!(actual["missing_tier_archives"], 1);
        assert_eq!(actual["periods"], json!([]));
        assert!(actual["cells"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["n"] == 0 && c["pending"] == 0));
        assert_eq!(
            issue_detail(&temp.0, &draws, &[], &issue, None, None).unwrap()["number_pools"]
                ["pools"]
                .as_array()
                .unwrap()
                .len(),
            9
        );
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn structural_preview_preserves_marginal_hits_but_not_ticket_prizes() {
        let prepared = prepare_forecast(&synth_draws(220, 9)).unwrap();
        let saved = strategy(&build("compound7", &prepared), &prepared);
        let tickets: Vec<Ticket> = serde_json::from_value(saved["tickets"].clone()).unwrap();
        let draw = Draw::from_parts(26001, 20260101, tickets[0].red, tickets[0].blue).unwrap();
        let reviewed = review_strategy(&saved, Some(&draw)).unwrap();
        let preview = compact_preview(&reviewed, Some(&draw)).unwrap();
        assert_eq!(preview["actual_recommendation"], false);
        assert_eq!(preview["kind"], "post_draw_structural_replay");
        assert_eq!(preview["summary"]["bets"], 2);
        assert_eq!(preview["saved_cost"], 10);
        assert_eq!(
            preview["summary"]["unique_reds"],
            reviewed["summary"]["unique_reds"]
        );
        assert_eq!(
            preview["summary"]["unique_blues"],
            reviewed["summary"]["unique_blues"]
        );
        assert_eq!(
            preview["summary"]["covered_reds"],
            reviewed["summary"]["covered_reds"]
        );
        assert_eq!(
            preview["summary"]["blue_covered"],
            reviewed["summary"]["blue_covered"]
        );
        assert_ne!(
            preview["summary"]["prize_tickets"],
            reviewed["summary"]["prize_tickets"]
        );
        assert!(preview["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .all(|n| n["fusion_score"].is_null()));
        let pending = compact_preview(&review_strategy(&saved, None).unwrap(), None).unwrap();
        assert!(pending["summary"]["fixed_prize"].is_null());
        assert!(pending["summary"]["mean_ticket_red_hits"].is_null());
        assert_eq!(pending["kind"], "unpublished_structural_preview");
        let minimal = strategy(&build("blue2", &prepared), &prepared);
        assert!(
            compact_preview(&review_strategy(&minimal, None).unwrap(), None)
                .unwrap()
                .is_null()
        );
    }

    #[test]
    fn daily_refuses_expiry_clock_reversal_and_nonadjacent_target() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let issue = next(&draws);
        assert!(freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 100).is_err());
        assert!(freeze_daily_with_clock(&temp.0, &draws, "03222", 100, || 10).is_err());
        for times in [[10, 100, 100], [10, 9, 9], [10, 99, 100]] {
            let index = Cell::new(0);
            assert!(freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || {
                let i = index.get();
                index.set(i + 1);
                times[i]
            })
            .is_err());
            assert!(!temp.0.join(format!("daily/{issue}.json")).exists());
        }
    }

    #[test]
    fn interactive_dedup_preserves_first_archive_and_marks_known_outcomes() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let prepared = prepare_forecast(&draws).unwrap();
        let plan = build("compound7", &prepared);
        let issue = next(&draws);
        let first = archive_plan(&temp.0, &plan, &prepared, &issue, None).unwrap();
        let path = archive_paths(&temp.0, &issue)[0].1.clone();
        let bytes = fs::read(path.clone()).unwrap();
        let repeated = archive_plan(&temp.0, &plan, &prepared, &issue, Some(123)).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(fs::read(path).unwrap(), bytes);
        assert_eq!(first["known_outcome_at_generation"], false);
        assert_eq!(
            archive_plan(
                &temp.0,
                &plan,
                &prepared,
                &fmt_issue(prepared.cutoff_issue),
                None
            )
            .unwrap()["known_outcome_at_generation"],
            true
        );
    }

    #[test]
    fn compound_and_dantuo_reviews_expand_every_bet_and_exposure() {
        let draws = synth_draws(220, 9);
        let prepared = prepare_forecast(&draws).unwrap();
        for (name, bets) in [
            ("compound7", 7),
            ("compound8", 28),
            ("blue2", 2),
            ("dantuo", 15),
        ] {
            let saved = strategy(&build(name, &prepared), &prepared);
            let view = review_strategy(&saved, None).unwrap();
            assert_eq!(view["summary"]["bets"], bets);
            assert_eq!(view["summary"]["cost"], bets * 2);
            let red: u64 = view["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v["color"] == "red")
                .map(|v| v["ticket_exposure"].as_u64().unwrap())
                .sum();
            let blue: u64 = view["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v["color"] == "blue")
                .map(|v| v["ticket_exposure"].as_u64().unwrap())
                .sum();
            assert_eq!(red, bets * 6);
            assert_eq!(blue, bets);
            assert!(view["numbers"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["rescued"] == false));
        }
    }

    #[test]
    fn prize_levels_and_duplicate_bets_are_counted_per_ticket() {
        let draw = Draw::from_parts(26014, 20260201, [1, 2, 3, 4, 5, 6], 7).unwrap();
        assert_eq!(
            (0..=6).map(|r| prize(r, true)).collect::<Vec<_>>(),
            vec![
                Some(6),
                Some(6),
                Some(6),
                Some(5),
                Some(4),
                Some(3),
                Some(1)
            ]
        );
        assert_eq!(
            (0..=6).map(|r| prize(r, false)).collect::<Vec<_>>(),
            vec![None, None, None, None, Some(5), Some(4), Some(2)]
        );
        let ticket = json!({"red": [1, 2, 3, 4, 10, 11], "blue": 8});
        let view = review_strategy(&json!({"tickets": [ticket, ticket]}), Some(&draw)).unwrap();
        assert_eq!(view["summary"]["duplicate_tickets"], 1);
        assert_eq!(view["summary"]["prize_tickets"], 2);
        assert_eq!(view["summary"]["fixed_prize"], 20);
        assert_eq!(view["summary"]["covered_red_count"], 4);
        assert_eq!(view["summary"]["repeated_red_positions"], 6);
        let special = review_strategy(
            &json!({"tickets": [{"red": [1, 2, 3, 10, 11, 12], "blue": 8}]}),
            Some(&draw),
        )
        .unwrap();
        assert_eq!(special["summary"]["special_prize_candidates"], 1);
        assert_eq!(special["summary"]["prize_tickets"], 0);
        assert_eq!(special["summary"]["fixed_prize"], 0);
        assert!(review_strategy(
            &json!({"tickets": [{"red": [1, 1, 3, 4, 5, 6], "blue": 7}]}),
            None
        )
        .is_err());
    }

    #[test]
    fn reviews_bind_original_tickets_and_never_overwrite_changed_outcomes() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        let archive = read_checked(&temp.0.join(format!("daily/{issue}.json"))).unwrap();
        append_outcome(&mut draws);
        let result = settle_archives(&temp.0, &draws);
        assert_eq!(result.settled, 1);
        assert!(result.errors.is_empty());
        let path = review_path(&temp.0, &issue, "daily");
        let original = fs::read(&path).unwrap();
        let detail = issue_detail(&temp.0, &draws, &[], &issue, None, Some("compound8")).unwrap();
        for (saved, reviewed) in archive["payload"]["strategies"]["compound8"]["tickets"]
            .as_array()
            .unwrap()
            .iter()
            .zip(detail["detail"]["tickets"].as_array().unwrap())
        {
            assert_eq!(saved["red"], reviewed["red"]);
            assert_eq!(saved["blue"], reviewed["blue"]);
        }
        assert_eq!(detail["provenance"]["review_persisted"], true);
        assert_eq!(settle_archives(&temp.0, &draws).settled, 0);
        assert_eq!(fs::read(&path).unwrap(), original);
        draws.last_mut().unwrap().blue = 8;
        assert!(!settle_archives(&temp.0, &draws).errors.is_empty());
        assert!(issue_detail(&temp.0, &draws, &[], &issue, None, None).is_err());
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn changed_history_or_tampered_archive_is_rejected_but_other_archives_settle() {
        let temp = Temp::new();
        let mut draws = synth_draws(220, 9);
        let issue = next(&draws);
        let prepared = prepare_forecast(&draws).unwrap();
        archive_plan(
            &temp.0,
            &build("compound7", &prepared),
            &prepared,
            &issue,
            None,
        )
        .unwrap();
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        draws[0].blue = draws[0].blue % 16 + 1;
        assert!(issue_detail(&temp.0, &draws, &[], &issue, None, None).is_err());
        draws = synth_draws(220, 9);
        append_outcome(&mut draws);
        let path = temp.0.join(format!("daily/{issue}.json"));
        let mut value = read_checked(&path).unwrap();
        value["payload"]["issue"] = json!("99999");
        fs::write(&path, value.to_string()).unwrap();
        let result = settle_archives(&temp.0, &draws);
        assert_eq!(result.settled, 1);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn missing_history_stays_missing_and_legacy_has_no_invented_evidence() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let issue = fmt_issue(draws.last().unwrap().issue);
        let missing = issue_detail(&temp.0, &draws, &[], &issue, None, None).unwrap();
        assert_eq!(missing["missing"], true);
        assert_eq!(
            issue_index(&temp.0, &draws, &[], 3)["issues"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let legacy = json!({"target": issue, "file": "old", "title": "原票组", "cutoff_issue": "03219", "tickets": [{"red": [1,2,3,4,5,6], "blue": 7}]});
        let view = issue_detail(&temp.0, &draws, &[legacy], &issue, None, None).unwrap();
        assert_eq!(view["provenance"]["evidence_available"], false);
        assert!(view["detail"]["numbers"][0]["fusion_score"].is_null());
        assert_eq!(
            view["detail"]["numbers"][0]["reason"],
            "historical_evidence_unavailable"
        );
        assert!(!temp.0.exists());
    }

    #[test]
    fn raw_exports_validate_hash_and_reject_traversal_and_symlink_escape() {
        let temp = Temp::new();
        let draws = synth_draws(220, 9);
        let issue = next(&draws);
        freeze_daily_with_clock(&temp.0, &draws, &issue, 100, || 10).unwrap();
        assert!(!artifact(&temp.0, &format!("daily/{issue}.json"))
            .unwrap()
            .is_empty());
        for path in [
            "../secret",
            "daily/../../secret",
            "/etc/passwd",
            "daily/%2e%2e.json",
        ] {
            assert!(artifact(&temp.0, path).is_err());
        }
        #[cfg(unix)]
        {
            let other = Temp::new();
            fs::create_dir_all(&other.0).unwrap();
            fs::write(other.0.join("outside.json"), "{}").unwrap();
            std::os::unix::fs::symlink(
                other.0.join("outside.json"),
                temp.0.join("daily/99999.json"),
            )
            .unwrap();
            assert!(artifact(&temp.0, "daily/99999.json")
                .unwrap_err()
                .contains("越界"));
        }
    }
}
