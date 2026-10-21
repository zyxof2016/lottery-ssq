use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::audit::{publish, render_score, target_draw, verify};
use crate::cycle::{self, decide};
use crate::draw::{data_sha256, fetch_500, fmt_date, fmt_issue, load_csv, Draw};
use crate::eval::{backtest, prepare_forecast};
use crate::hash::sha256_hex;
use crate::history::{replay, ReplayItem};
use crate::plan::{self, Plan};
use crate::pool;
use crate::protocol::Protocol;
use crate::recommendations;
use crate::report::{render, to_json};

const PAGE: &str = include_str!("web/index.html");
const REVIEWS_PAGE: &str = include_str!("web/reviews.html");
const POOLS_PAGE: &str = include_str!("web/pools.html");
const RESEARCH_PAGE: &str = include_str!("web/research.html");
const STATIC_ASSETS: [(&str, &str, &[u8]); 5] = [
    ("app.css", "text/css", include_bytes!("web/static/app.css")),
    ("shell.js", "application/javascript", include_bytes!("web/static/shell.js")),
    ("main.js", "application/javascript", include_bytes!("web/static/main.js")),
    ("pool-intersection.js", "application/javascript", include_bytes!("web/static/pool-intersection.js")),
    ("vendor/vue-3.5.24.js", "application/javascript", include_bytes!("web/static/vendor/vue-3.5.24.js")),
];

struct CachedReport {
    mtime: Option<SystemTime>,
    text: String,
    value: Value,
}

struct HistoryCache {
    data_sha: String,
    items: Arc<[ReplayItem]>,
}

struct App {
    data: PathBuf,
    audit: PathBuf,
    publish_dir: PathBuf,
    history_dir: PathBuf,
    report: Mutex<Option<CachedReport>>,
    history: Mutex<Option<HistoryCache>>,
    mutation: Mutex<()>,
}

pub fn serve(data: PathBuf, audit: PathBuf, publish_dir: PathBuf, bind: &str) -> Result<(), String> {
    let history_dir = publish_dir
        .parent()
        .map(|p| p.join("history"))
        .unwrap_or_else(|| publish_dir.join("history"));
    let app = Arc::new(App {
        data,
        audit,
        publish_dir,
        history_dir,
        report: Mutex::new(None),
        history: Mutex::new(None),
        mutation: Mutex::new(()),
    });
    let listener = TcpListener::bind(bind).map_err(|e| format!("无法监听 {bind}：{e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("无法读取监听地址：{e}"))?;
    eprintln!("界面 http://{addr}/");
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let app = Arc::clone(&app);
        thread::spawn(move || {
            let _ = handle(stream, app);
        });
    }
    Ok(())
}

fn handle(mut stream: std::net::TcpStream, app: Arc<App>) -> Result<(), String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(15)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(180)));
    let (method, path, query, body) = match read_request(&mut stream) {
        Ok(v) => v,
        Err(err) => {
            write_json(&mut stream, 400, &json!({"error": err}))?;
            return Ok(());
        }
    };
    let lookup = if method == "HEAD" { "GET" } else { method.as_str() };
    let (status, content_type, mut payload) = route(&app, lookup, &path, &query, &body);
    if method == "HEAD" {
        payload.clear();
    }
    write_response(&mut stream, status, content_type, &payload)
}

fn route(app: &App, method: &str, path: &str, query: &str, body: &str) -> (u16, &'static str, Vec<u8>) {
    // File mutations share a lock so sync cannot replace the input while a
    // recommendation is being generated, sealed or settled by this server.
    let _mutation = if method == "POST" { Some(app.mutation.lock().unwrap_or_else(|e| e.into_inner())) } else { None };
    if method == "GET" {
        let asset = ["/static/", "/reviews/static/", "/pools/static/", "/research/static/"].iter().find_map(|prefix| path.strip_prefix(*prefix));
        if let Some((_, mime, bytes)) = STATIC_ASSETS.iter().find(|(name, _, _)| Some(*name) == asset) {
            return (200, mime, bytes.to_vec());
        }
    }
    let result = match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            return (200, "text/html", PAGE.as_bytes().to_vec());
        }
        ("GET", "/reviews") | ("GET", "/reviews.html") | ("GET", "/reviews/") => {
            return (200, "text/html", REVIEWS_PAGE.as_bytes().to_vec());
        }
        ("GET", "/pools") | ("GET", "/pools/") | ("GET", "/pools.html") => {
            return (200, "text/html", POOLS_PAGE.as_bytes().to_vec());
        }
        ("GET", "/research") | ("GET", "/research/") | ("GET", "/research.html") => {
            return (200, "text/html", RESEARCH_PAGE.as_bytes().to_vec());
        }
        ("GET", "/api/pools") => load_csv(&app.data).and_then(|draws| pool::analyze(&draws)),
        ("GET", "/api/pool-algorithms") => load_csv(&app.data).and_then(|draws| crate::pool_algorithms::cached_analyze(&draws)),
        ("GET", "/api/full6") => load_csv(&app.data).and_then(|draws| crate::conditional::analyze(&draws)),
        ("GET", "/api/coverage") => load_csv(&app.data).and_then(|draws| crate::coverage::analyze(&draws)),
        ("GET", "/api/pools/observations") => load_csv(&app.data).and_then(|draws| recommendations::pool_observations(&recommendation_dir(app), &draws)),
        ("GET", "/api/recommendations/issues") => recommendation_index(app, query),
        ("GET", "/api/recommendations/files") => {
            let result = query_value(query, "path").ok_or("缺少档案路径".into())
                .and_then(|path| recommendations::artifact(&recommendation_dir(app), &path));
            match result {
                Ok(bytes) => return (200, "application/json", bytes),
                Err(error) => Err(error),
            }
        }
        ("POST", "/api/recommendations/freeze") => freeze_recommendations(app, body),
        ("POST", "/api/recommendations/settle") => {
            load_csv(&app.data).map(|draws| json!(recommendations::settle_archives(&recommendation_dir(app), &draws)))
        }
        ("GET", path) if path.starts_with("/api/recommendations/issues/") => {
            recommendation_detail(app, &path["/api/recommendations/issues/".len()..], query)
        }
        ("GET", "/health") => Ok(json!({"ok": true})),
        ("GET", "/api/overview") => overview_value(app),
        ("GET", "/api/draws") => draws_value(app, query),
        ("POST", "/api/sync") => sync_value(app),
        ("GET", "/api/records") => records_value(app),
        ("POST", "/api/settle") => settle_value(app),
        ("GET", "/api/forecast") => forecast_value(app),
        ("POST", "/api/generate") => generate_value(app, body),
        ("POST", "/api/plans") => save_plan_value(app, body),
        ("GET", "/api/report") | ("POST", "/api/backtest") => report_value(app, method == "POST"),
        ("GET", "/api/manifests") => Ok(json!({"manifests": list_manifests(&app.publish_dir)})),
        ("GET", "/api/history") => history_page(app, query),
        ("GET", "/api/cycle") => cycle_status(app),
        ("GET", "/api/reviews") => Ok(json!({"reviews": cycle::list_reviews(&review_dir(app))})),
        ("POST", "/api/publish") => publish_value(app, body),
        ("POST", "/api/verify") => verify_value(app, body),
        ("GET", path) if path.starts_with("/api/history/") => history_one(app, &path["/api/history/".len()..]),
        ("GET", path) if path.starts_with("/api/reviews/") => cycle::read_review(&review_dir(app), &path["/api/reviews/".len()..]),
        _ => Err("没有这个页面".into()),
    };
    match result {
        Ok(value) => (
            200,
            "application/json",
            serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec()),
        ),
        Err(err) => {
            let status = if path.starts_with("/api/") { 400 } else { 404 };
            (
                status,
                "application/json",
                serde_json::to_vec(&json!({"error": err})).unwrap_or_else(|_| b"{}".to_vec()),
            )
        }
    }
}

fn forecast_value(app: &App) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let prepared = prepare_forecast(&draws)?;
    let fc = &prepared.forecast;
    let rescued: Vec<u8> = fc.mild.rescued.iter().map(|i| *i as u8 + 1).collect();
    let tickets: Vec<Value> = fc
        .tickets
        .iter()
        .zip(fc.ticket_blues.iter())
        .map(|(mask, blue)| {
            json!({
                "red": mask_nums(*mask),
                "blue": *blue as u8 + 1,
            })
        })
        .collect();
    let weights: Vec<Value> = Protocol::EXPERTS_ZH
        .iter()
        .zip(prepared.weights.iter())
        .map(|(name, w)| json!({"name": name, "weight": w}))
        .collect();
    let blues: Vec<Value> = fc
        .blue_order
        .iter()
        .take(5)
        .map(|i| json!({"n": *i as u8 + 1, "p": fc.blue[*i]}))
        .collect();
    let last = draws.last().ok_or("没有开奖数据")?;
    Ok(json!({
        "n": prepared.n,
        "cutoff_issue": fmt_issue(prepared.cutoff_issue),
        "cutoff_date": fmt_date(prepared.cutoff_date),
        "input_history_sha256": prepared.data_sha,
        "suggested_target": format!("{:05}", prepared.cutoff_issue + 1),
        "last_draw": {
            "issue": fmt_issue(last.issue),
            "date": fmt_date(last.date),
            "red": last.red,
            "blue": last.blue,
        },
        "pool": fc.pool.iter().map(|i| *i as u8 + 1).collect::<Vec<_>>(),
        "rescued": rescued,
        "tickets": tickets,
        "blues": blues,
        "weights": weights,
        "protocol_sha256": Protocol::sha256(),
        "manifests": list_manifests(&app.publish_dir),
    }))
}

#[derive(Deserialize)]
struct GenReq {
    ranking_model: Option<String>,
    preset: Option<String>,
    mode: Option<String>,
    tickets: Option<usize>,
    pool: Option<usize>,
    overlap: Option<u32>,
    red_count: Option<usize>,
    blue_count: Option<usize>,
    bankers: Option<usize>,
    drags: Option<usize>,
    issue: Option<String>,
    not_after: Option<u64>,
    expected_archive_sha256: Option<String>,
}

fn generate_value(app: &App, body: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let (plan, prepared) = build_requested(body, &draws)?;
    let req = parse_gen(body)?;
    let issue = req.issue.unwrap_or_else(|| fmt_issue(prepared.cutoff_issue + 1));
    let archive = recommendations::archive_plan(&recommendation_dir(app), &plan, &prepared, &issue, req.not_after)?;
    let mut value = plan_json(&plan, &prepared, &draws);
    value["recommendation_archive"] = archive;
    value["issue"] = json!(issue);
    value["known_outcome_at_generation"] = json!(issue.as_str() <= fmt_issue(prepared.cutoff_issue).as_str());
    Ok(value)
}

fn save_plan_value(app: &App, body: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let req: GenReq = parse_gen(body)?;
    let issue = parse_issue(req.issue.as_deref().unwrap_or(""))?;
    let (plan, prepared) = build_requested(body, &draws)?;
    let canonical = plan_canonical(&issue, &plan);
    if let Some(expected) = req.expected_archive_sha256.as_deref() {
        recommendations::check_displayed_plan(&recommendation_dir(app), &plan, &prepared, &issue, expected)?;
    }
    let id = sha256_hex(canonical.as_bytes());
    let record = json!({
        "id": id,
        "issue": issue,
        "cutoff_issue": fmt_issue(prepared.cutoff_issue),
        "cutoff_date": fmt_date(prepared.cutoff_date),
        "mode": plan.mode,
        "ranking": prepared.ranking,
        "title": plan.title,
        "detail": plan.detail,
        "cost": plan.cost,
        "bet_count": plan.bet_count,
        "tickets": ticket_json(&plan),
        "created_unix": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    });
    let seal = if plan.mode == "frozen" {
        let outcome = publish(
            &app.data,
            &app.audit,
            &app.publish_dir,
            Some(issue.clone()),
            req.not_after,
        )?;
        Some(json!({
            "already": outcome.already,
            "content_sha256": outcome.content_sha256,
            "path": outcome.path.display().to_string(),
        }))
    } else {
        None
    };
    let archive = recommendations::archive_plan(&recommendation_dir(app), &plan, &prepared, &issue, req.not_after)?;
    let already = append_plan(&plans_file(app), &record, &id)?;
    let mut saved_plan = plan_json(&plan, &prepared, &draws);
    saved_plan["sealed"] = json!(seal.is_some());
    saved_plan["issue"] = json!(issue);
    saved_plan["known_outcome_at_generation"] = json!(issue.as_str() <= fmt_issue(prepared.cutoff_issue).as_str());
    saved_plan["recommendation_archive"] = archive;
    Ok(json!({
        "already": already,
        "id": id,
        "issue": issue,
        "title": plan.title,
        "bet_count": plan.bet_count,
        "cost": plan.cost,
        "plan": saved_plan,
        "seal": seal,
        "seal_error": null,
    }))
}

fn build_requested(body: &str, draws: &[Draw]) -> Result<(Plan, crate::eval::Prepared), String> {
    let req = parse_gen(body)?;
    let spec = plan::resolve(
        req.preset.as_deref(),
        req.mode.as_deref(),
        req.tickets,
        req.pool,
        req.overlap,
        req.red_count,
        req.blue_count,
        req.bankers,
        req.drags,
    )?;
    let model = req.ranking_model.as_deref().unwrap_or(if req.preset.as_deref()==Some("coverage24") {crate::conditional::MODEL} else if req.preset.as_deref()==Some("calibrated14") {crate::ranking::MODEL} else if req.preset.as_deref()==Some("equal24") {"equal"} else {"weighted"});
    if !["weighted","equal",crate::ranking::MODEL,crate::conditional::MODEL].contains(&model) && !crate::pool_algorithms::is_model(model) {return Err("未知红球排名模型".into());}
    if req.preset.as_deref()==Some("calibrated14") && model!="repeat_calibrated" {return Err("校准排名预设须使用校准模型".into());}
    if req.preset.as_deref()==Some("coverage24") && model!=crate::conditional::MODEL {return Err("24红条件频率预设须使用条件频率模型".into());}
    if req.preset.as_deref()==Some("weighted24") && model!="weighted" {return Err("原排名24红预设须使用原加权模型".into());}
    if req.preset.as_deref()==Some("equal24") && model!="equal" {return Err("等权24红预设须使用等权模型".into());}
    if spec.mode=="frozen" && model!="weighted" {return Err("协议冻结票组使用原加权排名，请用其他方案观察校准排名".into());}
    let prepared = if model==crate::conditional::MODEL {crate::conditional::prepare(draws)?} else if model==crate::ranking::MODEL {crate::ranking::prepare(draws)?} else if model=="equal" {crate::ranking::prepare_equal(draws)?} else if crate::pool_algorithms::is_model(model) {crate::pool_algorithms::prepare(draws,model)?} else {prepare_forecast(draws)?};
    let built = plan::build(&prepared.forecast, &spec)?;
    Ok((built, prepared))
}

fn parse_gen(body: &str) -> Result<GenReq, String> {
    if body.trim().is_empty() {
        return Ok(GenReq {
            ranking_model: None,
            preset: Some("frozen".into()),
            mode: None,
            tickets: None,
            pool: None,
            overlap: None,
            red_count: None,
            blue_count: None,
            bankers: None,
            drags: None,
            issue: None,
            not_after: None,
            expected_archive_sha256: None,
        });
    }
    serde_json::from_str(body).map_err(|_| "生成参数无法解析".to_string())
}

fn parse_issue(raw: &str) -> Result<String, String> {
    let issue = raw.trim();
    if issue.len() != 5 || !issue.chars().all(|c| c.is_ascii_digit()) {
        return Err("期号要是 5 位数字".into());
    }
    Ok(issue.to_string())
}

fn plans_file(app: &App) -> PathBuf {
    app.history_dir
        .parent()
        .map(|p| p.join("plans").join("plans.jsonl"))
        .unwrap_or_else(|| PathBuf::from("plans.jsonl"))
}

fn plan_canonical(issue: &str, plan: &Plan) -> String {
    let mut lines = vec![format!("issue={issue}"), format!("mode={}", plan.mode)];
    for (i, t) in plan.tickets.iter().enumerate() {
        let reds = t.red.iter().map(|n| format!("{n:02}")).collect::<Vec<_>>().join(",");
        lines.push(format!("ticket{i}={reds}+{:02}", t.blue));
    }
    lines.join("\n")
}

fn ticket_json(plan: &Plan) -> Vec<Value> {
    plan.tickets
        .iter()
        .map(|t| json!({"red": t.red.to_vec(), "blue": t.blue}))
        .collect()
}

fn plan_json(plan: &Plan, prepared: &crate::eval::Prepared, draws: &[Draw]) -> Value {
    let last = draws.last();
    json!({
        "mode": plan.mode,
        "ranking": prepared.ranking,
        "title": plan.title,
        "detail": plan.detail,
        "bet_count": plan.bet_count,
        "cost": plan.cost,
        "tickets": ticket_json(plan),
        "reds": plan.reds,
        "bankers": plan.bankers,
        "drags": plan.drags,
        "blues": plan.blues,
        "rescued": plan.rescued,
        "sealed": false,
        "requires_seal": plan.mode == "frozen",
        "cutoff_issue": fmt_issue(prepared.cutoff_issue),
        "cutoff_date": fmt_date(prepared.cutoff_date),
        "input_history_sha256": prepared.data_sha,
        "suggested_target": format!("{:05}", prepared.cutoff_issue + 1),
        "last_draw": last.map(draw_json),
    })
}

fn append_plan(path: &Path, value: &Value, id: &str) -> Result<bool, String> {
    if let Ok(text) = fs::read_to_string(path) {
        for line in text.lines() {
            if let Ok(old) = serde_json::from_str::<Value>(line) {
                if old.get("id").and_then(|v| v.as_str()) == Some(id) {
                    return Ok(true);
                }
            }
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("无法创建方案目录：{e}"))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("无法写入方案：{e}"))?;
    let line = serde_json::to_string(value).map_err(|e| format!("方案无法编码：{e}"))?;
    writeln!(file, "{line}").map_err(|e| format!("无法写入方案：{e}"))?;
    Ok(false)
}

fn overview_value(app: &App) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let last = draws.last().ok_or("没有开奖数据")?;
    let records = saved_records(app, &draws)?;
    let pending = records.iter().filter(|r| r["status"] == "pending").count();
    Ok(json!({
        "latest_draw": draw_json(last),
        "total_draws": draws.len(),
        "next_issue": format!("{:05}", last.issue + 1),
        "protocol_sha256": Protocol::sha256(),
        "data_sha256": data_sha256(&draws),
        "master_seed": format!("0x{:016x}", Protocol::SEED),
        "total_plans": records.len(),
        "pending": pending,
        "scored": records.iter().filter(|r| r["status"] == "scored").count(),
        "invalid": records.iter().filter(|r| r["status"] == "invalid").count(),
    }))
}

fn draws_value(app: &App, query: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let limit = query_usize(query, "limit").unwrap_or(30).clamp(1, 100);
    let before = query_value(query, "before");
    let mut rows: Vec<&Draw> = draws
        .iter()
        .filter(|d| before.as_ref().map(|b| fmt_issue(d.issue).as_str() < b.as_str()).unwrap_or(true))
        .collect();
    rows.reverse();
    let page: Vec<Value> = rows.into_iter().take(limit).map(draw_json).collect();
    Ok(json!({
        "total": draws.len(),
        "latest": draws.last().map(draw_json),
        "rows": page,
    }))
}

fn sync_value(app: &App) -> Result<Value, String> {
    let n = fetch_500(&app.data)?;
    if let Ok(mut slot) = app.report.lock() {
        *slot = None;
    }
    if let Ok(mut slot) = app.history.lock() {
        *slot = None;
    }
    let draws = load_csv(&app.data)?;
    let last = draws.last().ok_or("同步后没有开奖数据")?;
    let recommendations = recommendations::settle_archives(&recommendation_dir(app), &draws);
    Ok(json!({
        "n": n,
        "latest_draw": draw_json(last),
        "recommendations": recommendations,
    }))
}

fn records_value(app: &App) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    Ok(json!({ "records": saved_records(app, &draws)? }))
}

fn settle_value(app: &App) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let mut settled = 0usize;
    let mut skipped = 0usize;
    let mut errors = Vec::new();
    for path in manifest_files(&app.publish_dir) {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                errors.push(format!("{}: {e}", path.display()));
                continue;
            }
        };
        let canonical = value.get("canonical").and_then(|v| v.as_str()).unwrap_or("");
        let sha = value.get("content_sha256").and_then(|v| v.as_str()).unwrap_or("");
        let target = manifest_field(canonical, "target").unwrap_or_default();
        let cutoff = manifest_field(canonical, "cutoff_issue").unwrap_or_default();
        let cutoff_n: u32 = match cutoff.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        match target_draw(&draws, cutoff_n, &target) {
            Ok(Some(_)) => {}
            Ok(None) => {
                skipped += 1;
                continue;
            }
            Err(error) => {
                errors.push(format!("{target}: {error}"));
                continue;
            }
        }
        if audit_has_verify(&app.audit, &target, sha) {
            skipped += 1;
            continue;
        }
        match verify(&app.data, &path, Some(&app.audit)) {
            Ok(_) => settled += 1,
            Err(e) => errors.push(format!("{target}: {e}")),
        }
    }
    Ok(json!({
        "settled": settled,
        "skipped": skipped,
        "errors": errors,
        "records": saved_records(app, &draws)?,
        "recommendations": recommendations::settle_archives(&recommendation_dir(app), &draws),
    }))
}

fn saved_records(app: &App, draws: &[Draw]) -> Result<Vec<Value>, String> {
    let mut out = plan_records(app, draws);
    for path in manifest_files(&app.publish_dir) {
        let text = fs::read_to_string(&path).map_err(|e| format!("无法读取 {}: {e}", path.display()))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| format!("{} 不是清单：{e}", path.display()))?;
        let canonical = value.get("canonical").and_then(|v| v.as_str()).unwrap_or("");
        let tickets = manifest_tickets(canonical);
        let cutoff = manifest_field(canonical, "cutoff_issue").unwrap_or_default();
        let target = manifest_field(canonical, "target").unwrap_or_default();
        let matched = cutoff.parse::<u32>()
            .map_err(|_| "清单截止期无法解析".to_string())
            .and_then(|cutoff| target_draw(draws, cutoff, &target));
        let (draw, error) = match matched {
            Ok(draw) => (draw, None),
            Err(error) => (None, Some(error)),
        };
        let (ticket_hits, best_red, any_blue) = score_tickets(&tickets, draw);
        let coverage = coverage_json(&tickets, draw, Some(canonical));
        out.push(json!({
            "file": path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            "title": "开奖前哈希",
            "cost": tickets.len() as u32 * 2,
            "source": "seal",
            "target": target,
            "cutoff_issue": cutoff,
            "cutoff_date": manifest_field(canonical, "cutoff_date").unwrap_or_default(),
            "sha": value.get("content_sha256").and_then(|v| v.as_str()).unwrap_or(""),
            "published_at": value.get("published_at_unix").and_then(|v| v.as_u64()).unwrap_or(0),
            "status": if error.is_some() { "invalid" } else if draw.is_some() { "scored" } else { "pending" },
            "error": error,
            "tickets": tickets,
            "draw": draw.map(draw_json),
            "best_red": best_red,
            "any_blue": any_blue,
            "ticket_hits": ticket_hits,
            "coverage": coverage,
        }));
    }
    out.sort_by(|a, b| b["target"].as_str().unwrap_or("").cmp(a["target"].as_str().unwrap_or("")));
    Ok(out)
}

fn plan_records(app: &App, draws: &[Draw]) -> Vec<Value> {
    let Ok(text) = fs::read_to_string(plans_file(app)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else { continue };
        let tickets = value.get("tickets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let issue = value.get("issue").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
        let (ticket_hits, best_red, any_blue) = score_tickets(&tickets, draw);
        let coverage = coverage_json(&tickets, draw, None);
        out.push(json!({
            "file": value.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "title": value.get("title").and_then(|v| v.as_str()).unwrap_or("方案"),
            "detail": value.get("detail").and_then(|v| v.as_str()).unwrap_or(""),
            "cost": value.get("cost").and_then(|v| v.as_u64()).unwrap_or(0),
            "source": "plan",
            "target": issue,
            "cutoff_issue": value.get("cutoff_issue").and_then(|v| v.as_str()).unwrap_or(""),
            "cutoff_date": value.get("cutoff_date").and_then(|v| v.as_str()).unwrap_or(""),
            "sha": value.get("id").and_then(|v| v.as_str()).unwrap_or(""),
            "published_at": value.get("created_unix").and_then(|v| v.as_u64()).unwrap_or(0),
            "status": if draw.is_some() { "scored" } else { "pending" },
            "tickets": tickets,
            "draw": draw.map(draw_json),
            "best_red": best_red,
            "any_blue": any_blue,
            "ticket_hits": ticket_hits,
            "coverage": coverage,
        }));
    }
    out
}

fn score_tickets(tickets: &[Value], draw: Option<&Draw>) -> (Vec<Value>, u32, bool) {
    let Some(draw) = draw else {
        return (Vec::new(), 0, false);
    };
    let win: std::collections::HashSet<u8> = draw.red.into_iter().collect();
    let mut ticket_hits = Vec::new();
    let mut best_red = 0u32;
    let mut any_blue = false;
    for t in tickets {
        let reds = t.get("red").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let hits = reds
            .iter()
            .filter(|n| n.as_u64().is_some_and(|x| win.contains(&(x as u8))))
            .count() as u32;
        let blue = t.get("blue").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
        let blue_hit = blue == draw.blue;
        best_red = best_red.max(hits);
        any_blue |= blue_hit;
        ticket_hits.push(json!({"red_hits": hits, "blue_hit": blue_hit}));
    }
    (ticket_hits, best_red, any_blue)
}

fn coverage_json(tickets: &[Value], draw: Option<&Draw>, canonical: Option<&str>) -> Value {
    let Some(draw) = draw else {
        return Value::Null;
    };
    let mut red_count = [0u8; 33];
    let mut blue_count = [0u8; 16];
    for t in tickets {
        if let Some(reds) = t.get("red").and_then(|v| v.as_array()) {
            for n in reds {
                if let Some(x) = n.as_u64() {
                    if (1..=33).contains(&x) {
                        red_count[(x - 1) as usize] = red_count[(x - 1) as usize].saturating_add(1);
                    }
                }
            }
        }
        if let Some(b) = t.get("blue").and_then(|v| v.as_u64()) {
            if (1..=16).contains(&b) {
                blue_count[(b - 1) as usize] = blue_count[(b - 1) as usize].saturating_add(1);
            }
        }
    }
    let covered_reds: Vec<u8> = red_count
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, _)| i as u8 + 1)
        .collect();
    let missed_reds: Vec<u8> = draw.red.iter().copied().filter(|n| red_count[*n as usize - 1] == 0).collect();
    let red_hit_n = draw.red.len() - missed_reds.len();
    let mut red_repeat: Vec<Value> = red_count
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| json!({"n": i as u8 + 1, "count": c}))
        .collect();
    red_repeat.sort_by(|a, b| {
        b["count"].as_u64().unwrap_or(0).cmp(&a["count"].as_u64().unwrap_or(0)).then(a["n"].as_u64().unwrap_or(0).cmp(&b["n"].as_u64().unwrap_or(0)))
    });
    let blue_covered = blue_count[draw.blue as usize - 1] > 0;
    let covered_n = covered_reds.len() as u32;
    let expected_red_hits = 6.0 * f64::from(covered_n) / 33.0;
    let chance_at_least = chance_at_least(33, 6, covered_n, red_hit_n as u32);
    let gaps = coverage_gaps(
        canonical,
        &missed_reds,
        draw.blue,
        blue_covered,
        covered_reds.len(),
        tickets.len(),
    );
    json!({
        "red_hit_n": red_hit_n,
        "red_draw_n": draw.red.len(),
        "blue_covered": blue_covered,
        "expected_red_hits": expected_red_hits,
        "chance_at_least": chance_at_least,
        "missed_reds": missed_reds,
        "covered_reds": covered_reds,
        "covered_red_n": covered_reds.len(),
        "red_repeat": red_repeat,
        "gaps": gaps,
    })
}

fn coverage_gaps(
    canonical: Option<&str>,
    missed_reds: &[u8],
    blue: u8,
    blue_covered: bool,
    pool_fallback: usize,
    tickets_n: usize,
) -> Vec<Value> {
    let order = canonical.and_then(|c| manifest_field(c, "red_rank")).map(|s| parse_num_list(&s));
    let pool = canonical.and_then(|c| manifest_field(c, "pool")).map(|s| parse_num_list(&s));
    let displaced = canonical
        .and_then(|c| manifest_field(c, "displaced"))
        .map(|s| parse_num_list(&s))
        .unwrap_or_default();
    let pool_n = pool.as_ref().map(|p| p.len()).unwrap_or(pool_fallback);
    let mut gaps = Vec::new();
    for &n in missed_reds {
        let rank = order.as_ref().and_then(|o| o.iter().position(|x| *x == n).map(|i| i as u16 + 1));
        let in_pool = pool.as_ref().is_some_and(|p| p.contains(&n));
        let was_displaced = displaced.contains(&n);
        let reason = if let Some(rank) = rank {
            if was_displaced {
                format!("红球 {n:02} 封存时融合名次第 {rank} 名。救援换入别的号时把它移出了定稿，所以这 {tickets_n} 注里没有它。")
            } else if !in_pool && rank as usize <= pool_n.max(1) {
                format!("红球 {n:02} 封存时融合名次第 {rank} 名，本来靠前。出票池的 {pool_n} 个位置先留给救援号，它没有进入池子。")
            } else if !in_pool {
                format!("红球 {n:02} 封存时融合名次第 {rank} 名。出票池只有 {pool_n} 个红球，它在池外。")
            } else {
                format!("红球 {n:02} 封存时在出票池里，融合名次第 {rank} 名。{tickets_n} 注限制了红球重合，没有任何一注带上它。")
            }
        } else {
            format!("红球 {n:02} 不在这组封存注单里。")
        };
        gaps.push(json!({"n": n, "color": "red", "reason": reason}));
    }
    if !blue_covered {
        let blue_order = canonical.and_then(|c| manifest_field(c, "blue_rank")).map(|s| parse_num_list(&s));
        let rank = blue_order.as_ref().and_then(|o| o.iter().position(|x| *x == blue).map(|i| i + 1));
        let reason = if let Some(rank) = rank {
            format!("蓝球 {blue:02} 封存时名次第 {rank} 名，没有进入这组注单。")
        } else {
            format!("蓝球 {blue:02} 没有进入这组注单。")
        };
        gaps.push(json!({"n": blue, "color": "blue", "reason": reason}));
    }
    gaps
}

fn chance_at_least(population: u32, winning: u32, pool: u32, observed: u32) -> f64 {
    if pool > population || observed > winning {
        return 0.0;
    }
    let first = observed.max(pool.saturating_sub(population - winning));
    let last = pool.min(winning);
    if first > last {
        return 0.0;
    }
    let mut chance = 0.0;
    for hits in first..=last {
        chance += (log_choose(winning, hits)
            + log_choose(population - winning, pool - hits)
            - log_choose(population, pool))
        .exp();
    }
    chance.clamp(0.0, 1.0)
}

fn log_choose(total: u32, count: u32) -> f64 {
    if count > total {
        return f64::NEG_INFINITY;
    }
    let k = count.min(total - count);
    let mut result = 0.0;
    for i in 1..=k {
        result += f64::from(total - i + 1).ln() - f64::from(i).ln();
    }
    result
}

fn parse_num_list(raw: &str) -> Vec<u8> {
    if raw == "-" {
        return Vec::new();
    }
    raw.split(',').filter_map(|s| s.parse().ok()).collect()
}

fn manifest_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return paths;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

fn manifest_field(canonical: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    canonical.lines().find_map(|line| line.strip_prefix(&prefix).map(|v| v.to_string()))
}

fn manifest_tickets(canonical: &str) -> Vec<Value> {
    let mut tickets = Vec::new();
    for line in canonical.lines() {
        let Some(rest) = line.strip_prefix("ticket") else { continue };
        let Some((_, body)) = rest.split_once('=') else { continue };
        let Some((reds, blue)) = body.split_once('+') else { continue };
        let red: Vec<u8> = reds.split(',').filter_map(|s| s.parse().ok()).collect();
        let Ok(blue) = blue.parse::<u8>() else { continue };
        if red.len() == 6 {
            tickets.push(json!({"red": red, "blue": blue}));
        }
    }
    tickets
}

fn audit_has_verify(path: &Path, target: &str, sha: &str) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    text.lines().any(|line| {
        let Ok(v) = serde_json::from_str::<Value>(line) else { return false };
        v.get("kind").and_then(|x| x.as_str()) == Some("verify")
            && v.get("target").and_then(|x| x.as_str()) == Some(target)
            && v.get("content_sha256").and_then(|x| x.as_str()) == Some(sha)
    })
}

fn draw_json(draw: &Draw) -> Value {
    json!({
        "issue": fmt_issue(draw.issue),
        "date": fmt_date(draw.date),
        "red": draw.red,
        "blue": draw.blue,
    })
}

fn history_page(app: &App, query: &str) -> Result<Value, String> {
    let items = load_history(app)?;
    let limit = query_usize(query, "limit").unwrap_or(40).clamp(1, 200);
    let before = query_value(query, "before");
    let from = query_value(query, "from").map(|v| parse_issue(&v)).transpose()?;
    let to = query_value(query, "to").map(|v| parse_issue(&v)).transpose()?;
    if from.as_ref().zip(to.as_ref()).is_some_and(|(a, b)| a > b) {
        return Err("起始期号不能晚于结束期号".into());
    }
    let selected: Vec<&ReplayItem> = items.iter().filter(|it| {
        from.as_ref().is_none_or(|v| it.issue.as_str() >= v.as_str())
            && to.as_ref().is_none_or(|v| it.issue.as_str() <= v.as_str())
    }).collect();
    let mut rows: Vec<&ReplayItem> = selected
        .iter().copied()
        .filter(|it| before.as_ref().map(|b| it.issue.as_str() < b.as_str()).unwrap_or(true))
        .collect();
    rows.reverse();
    let page: Vec<&ReplayItem> = rows.into_iter().take(limit).collect();
    let n = selected.len() as f64;
    let mean = |f: fn(&ReplayItem) -> f64| {
        if selected.is_empty() {
            None
        } else {
            Some(selected.iter().map(|it| f(it)).sum::<f64>() / n)
        }
    };
    Ok(json!({
        "kind": "post_draw_causal_replay", "actual_recommendation": false,
        "total": selected.len(),
        "from": selected.first().map(|it| it.issue.clone()),
        "to": selected.last().map(|it| it.issue.clone()),
        "note": "回放只用该期之前的开奖。确认窗起点 25111 及之后使用冻结权重，与发布规则相同。这些记录是事后回放，不是开奖前封存的哈希。",
        "summary": {
            "mean_best": mean(|it| it.best_red as f64),
            "mean_avg": mean(|it| it.avg_red),
            "top1_blue": mean(|it| if it.top1_blue { 1.0 } else { 0.0 }),
            "any_blue": mean(|it| if it.any_blue { 1.0 } else { 0.0 }),
            "mean_red_hit": mean(|it| it.review.red_hit_n as f64),
            "mean_blue_covered": mean(|it| if it.review.blue_covered { 1.0 } else { 0.0 }),
            "mean_covered_red_n": mean(|it| it.review.covered_red_n as f64),
        },
        "rows": page.iter().map(|it| json!({
            "issue": it.issue,
            "date": it.date,
            "weight_mode": it.weight_mode,
            "best_red": it.best_red,
            "avg_red": it.avg_red,
            "top1_blue": it.top1_blue,
            "any_blue": it.any_blue,
            "red_hit_n": it.review.red_hit_n,
            "covered_red_n": it.review.covered_red_n,
            "blue_covered": it.review.blue_covered,
        })).collect::<Vec<_>>(),
    }))
}

fn review_dir(app: &App) -> PathBuf {
    app.history_dir
        .parent()
        .map(|p| p.join("reviews"))
        .unwrap_or_else(|| PathBuf::from("reviews"))
}

fn cycle_status(app: &App) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let last = draws.last().ok_or("没有开奖数据")?;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let gate = decide(now, last.issue, last.date, false);
    Ok(json!({
        "latest_issue": fmt_issue(last.issue),
        "latest_date": fmt_date(last.date),
        "publish": gate.publish,
        "target": gate.target,
        "deadline": gate.deadline_text,
        "reason": gate.reason,
        "log": cycle::log_tail(&review_dir(app), 12),
        "reviews": cycle::list_reviews(&review_dir(app)),
    }))
}

fn history_one(app: &App, issue: &str) -> Result<Value, String> {
    if issue.len() != 5 || !issue.chars().all(|c| c.is_ascii_digit()) {
        return Err("期号需要 5 位数字".into());
    }
    let items = load_history(app)?;
    let item = items
        .iter()
        .find(|it| it.issue == issue)
        .ok_or_else(|| format!("回放里没有 {issue}"))?;
    Ok(serde_json::to_value(item).map_err(|e| e.to_string())?)
}

fn load_history(app: &App) -> Result<Arc<[ReplayItem]>, String> {
    let draws = load_csv(&app.data)?;
    let sha = data_sha256(&draws);
    let mut slot = app.history.lock().unwrap_or_else(|e| e.into_inner());
    if slot.as_ref().map(|c| c.data_sha.as_str()) != Some(sha.as_str()) {
        let items = replay(&draws)?;
        fs::create_dir_all(&app.history_dir).map_err(|e| format!("无法创建历史目录：{e}"))?;
        write_history_file(&app.history_dir.join("replay.jsonl"), &sha, &items)?;
        *slot = Some(HistoryCache { data_sha: sha, items: items.into() });
    }
    Ok(Arc::clone(&slot.as_ref().ok_or("历史回放尚未生成")?.items))
}

fn write_history_file(path: &Path, sha: &str, items: &[ReplayItem]) -> Result<(), String> {
    let mut text = format!(
        "{}\n",
        serde_json::to_string(&json!({
            "kind": "replay-header",
            "protocol": Protocol::sha256(),
            "data": sha,
            "n": items.len(),
        }))
        .map_err(|e| e.to_string())?
    );
    for item in items {
        text.push_str(&serde_json::to_string(item).map_err(|e| e.to_string())?);
        text.push('\n');
    }
    fs::write(path, text).map_err(|e| format!("无法写入历史回放：{e}"))
}

fn query_value(query: &str, key: &str) -> Option<String> {
    for part in query.split('&') {
        if let Some((k, v)) = part.split_once('=') {
            if k == key && !v.is_empty() {
                return decode_query(v);
            }
        }
    }
    None
}

fn decode_query(raw: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut input = raw.bytes();
    while let Some(byte) = input.next() {
        match byte {
            b'+' => bytes.push(b' '),
            b'%' => {
                let high = (input.next()? as char).to_digit(16)?;
                let low = (input.next()? as char).to_digit(16)?;
                bytes.push((high * 16 + low) as u8);
            }
            _ => bytes.push(byte),
        }
    }
    String::from_utf8(bytes).ok()
}

fn recommendation_dir(app: &App) -> PathBuf {
    app.publish_dir.parent().unwrap_or_else(|| Path::new(".")).join("recommendations")
}

fn recommendation_index(app: &App, query: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let legacy = saved_records(app, &draws)?;
    let mut value = recommendations::issue_index(&recommendation_dir(app), &draws, &legacy, query_usize(query, "limit").unwrap_or(60).clamp(1, 500));
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = decide(recommendations::now_unix(), last.issue, last.date, false);
    value["next"] = json!({"issue": gate.target, "deadline": gate.deadline_text, "can_freeze": gate.publish, "reason": gate.reason});
    Ok(value)
}

fn recommendation_detail(app: &App, issue: &str, query: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    recommendations::issue_detail(&recommendation_dir(app), &draws, &saved_records(app, &draws)?, issue,
        query_value(query, "archive").as_deref(), query_value(query, "strategy").as_deref())
}

fn freeze_recommendations(app: &App, body: &str) -> Result<Value, String> {
    let draws = load_csv(&app.data)?;
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = decide(recommendations::now_unix(), last.issue, last.date, false);
    if !body.trim().is_empty() {
        let expected: Value = serde_json::from_str(body).map_err(|e| format!("请求无效：{e}"))?;
        if expected["data_sha256"] != data_sha256(&draws) || expected["target"] != gate.target {
            return Err("显示的选号池与当前数据或目标期不一致，请刷新后留档".into());
        }
    }
    if !gate.publish { return Err(gate.reason); }
    recommendations::freeze_daily(&recommendation_dir(app), &draws, &gate.target, gate.deadline_unix)
}

fn query_usize(query: &str, key: &str) -> Option<usize> {
    query_value(query, key)?.parse().ok()
}

fn report_value(app: &App, force: bool) -> Result<Value, String> {
    let mtime = fs::metadata(&app.data).and_then(|m| m.modified()).ok();
    let mut slot = app.report.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = slot
        .as_ref()
        .is_some_and(|c| !force && c.mtime == mtime);
    if !fresh {
        let draws = load_csv(&app.data)?;
        let bt = backtest(&draws)?;
        let text = render(&bt);
        let value = to_json(&bt);
        *slot = Some(CachedReport { mtime, text, value });
    }
    let cached = slot.as_ref().ok_or("验证结果还没有生成")?;
    Ok(json!({
        "text": cached.text,
        "summary": cached.value,
    }))
}

#[derive(Deserialize)]
struct PublishReq {
    target: String,
    not_after: Option<u64>,
}

fn publish_value(app: &App, body: &str) -> Result<Value, String> {
    let req: PublishReq = serde_json::from_str(body).map_err(|_| "发布请求无法解析".to_string())?;
    let target = req.target.trim();
    if target.is_empty() {
        return Err("目标期号不能为空".into());
    }
    fs::create_dir_all(&app.publish_dir)
        .map_err(|e| format!("无法创建发布目录：{e}"))?;
    if let Some(parent) = app.audit.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("无法创建审计目录：{e}"))?;
    }
    let outcome = publish(
        &app.data,
        &app.audit,
        &app.publish_dir,
        Some(target.to_string()),
        req.not_after,
    )?;
    Ok(json!({
        "already": outcome.already,
        "target": outcome.target,
        "content_sha256": outcome.content_sha256,
        "path": outcome.path.display().to_string(),
        "manifests": list_manifests(&app.publish_dir),
    }))
}

#[derive(Deserialize)]
struct VerifyReq {
    manifest: String,
}

fn verify_value(app: &App, body: &str) -> Result<Value, String> {
    let req: VerifyReq = serde_json::from_str(body).map_err(|_| "复算请求无法解析".to_string())?;
    let path = safe_manifest(&app.publish_dir, &req.manifest)?;
    let outcome = verify(&app.data, &path, Some(&app.audit))?;
    let mut text = format!(
        "复算通过。目标 {}，截止期 {}，哈希 {}",
        outcome.target,
        fmt_issue(outcome.cutoff_issue),
        outcome.content_sha256
    );
    if let Some(card) = outcome.score {
        text.push('\n');
        text.push_str(&render_score(&card));
    } else {
        text.push_str("\n截止期之后还没有开奖记录，本次只核对了留痕。");
    }
    Ok(json!({
        "text": text,
        "content_sha256": outcome.content_sha256,
    }))
}

fn safe_manifest(dir: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || !name.ends_with(".json")
    {
        return Err("清单文件名不合法".into());
    }
    let path = dir.join(name);
    if !path.is_file() {
        return Err(format!("找不到清单 {name}"));
    }
    Ok(path)
}

fn list_manifests(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(rd) = fs::read_dir(dir) else {
        return names;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".json") {
            names.push(name);
        }
    }
    names.sort();
    names
}

fn mask_nums(mask: u64) -> Vec<u8> {
    let mut nums = Vec::new();
    let mut bits = mask;
    while bits != 0 {
        let i = bits.trailing_zeros();
        nums.push((i + 1) as u8);
        bits &= bits - 1;
    }
    nums
}

fn read_request(stream: &mut std::net::TcpStream) -> Result<(String, String, String, String), String> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    let header_end = loop {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| format!("读取请求失败：{e}"))?;
        if n == 0 {
            return Err("连接已关闭".into());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_sep(&buf) {
            break pos;
        }
        if buf.len() > 65536 {
            return Err("请求头过长".into());
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.split("\r\n");
    let request = lines.next().unwrap_or("");
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/");
    let (path, query) = raw_path
        .split_once('?')
        .map(|(p, q)| (p.to_string(), q.to_string()))
        .unwrap_or_else(|| (raw_path.to_string(), String::new()));
    let mut length = 0usize;
    for line in lines {
        if let Some(v) = line.split_once(':') {
            if v.0.eq_ignore_ascii_case("content-length") {
                length = v.1.trim().parse().unwrap_or(0);
            }
        }
    }
    if length > 65536 {
        return Err("请求体过长".into());
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < length {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| format!("读取请求体失败：{e}"))?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(length);
    let body = String::from_utf8(body).map_err(|_| "请求体不是 UTF-8".to_string())?;
    Ok((method, path, query, body))
}

fn find_sep(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn write_json(stream: &mut std::net::TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    write_response(stream, status, "application/json", &body)
}

fn write_response(
    stream: &mut std::net::TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body))
        .map_err(|e| format!("写出响应失败：{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{synth_draws, write_csv};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture(name: &str) -> (PathBuf, App, Vec<Draw>) {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ssq-web-{name}-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("ssq.csv");
        let draws = synth_draws(220, 9);
        write_csv(&data, &draws).unwrap();
        let app = App {
            data,
            audit: dir.join("audit.jsonl"),
            publish_dir: dir.join("publish"),
            history_dir: dir.join("history"),
            report: Mutex::new(None),
            history: Mutex::new(None),
            mutation: Mutex::new(()),
        };
        (dir, app, draws)
    }

    #[test]
    fn research_window_summary_stays_fixed_across_pages_and_empty_windows_are_null() {
        let (dir, app, draws) = fixture("research-window");
        let from = fmt_issue(draws[draws.len()-5].issue);
        let to = fmt_issue(draws.last().unwrap().issue);
        let query = format!("from={from}&to={to}&limit=2");
        let first = history_page(&app, &query).unwrap();
        assert_eq!(first["total"], 5);
        assert_eq!(first["rows"].as_array().unwrap().len(), 2);
        assert_eq!(first["actual_recommendation"], false);
        let second = history_page(&app, &format!("{query}&before={}", first["rows"][1]["issue"].as_str().unwrap())).unwrap();
        assert_eq!(second["total"], 5);
        assert_eq!(second["summary"], first["summary"]);
        assert_ne!(second["rows"], first["rows"]);
        let all = load_history(&app).unwrap();
        let expected: f64 = all.iter().rev().take(5).map(|r| r.review.red_hit_n as f64).sum::<f64>() / 5.0;
        assert!((first["summary"]["mean_red_hit"].as_f64().unwrap()-expected).abs()<1e-12);
        let empty = history_page(&app, "from=99990&to=99999").unwrap();
        assert_eq!(empty["total"], 0);
        assert!(empty["summary"]["mean_red_hit"].is_null());
        assert!(history_page(&app, "from=99999&to=00001").is_err());
        assert!(history_page(&app, "from=bad").is_err());
        assert!(!recommendation_dir(&app).exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn pool_api_reports_red_curves_keeps_archived_grid_and_rejects_stale_freezing() {
        let (dir, app, _) = fixture("small-pools");
        let (status, _, bytes) = route(&app, "GET", "/api/pools", "", "");
        assert_eq!(status, 200);
        let report: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report["version"], 3);
        assert_eq!(report["archive_version"], 4);
        assert_eq!(report["red_sizes"], json!((6..=33).collect::<Vec<_>>()));
        assert_eq!(report["archived_red_sizes"], json!([8, 10, 12, 14, 16, 18, 20, 22, 24]));
        assert_eq!(report["observation_tiers"].as_array().unwrap().len(), 6);
        assert_eq!(report["focus_red_range"], json!([14, 24]));
        assert_eq!(report["windows"]["all_oos"]["red_curves"].as_array().unwrap().len(), 252);
        assert_eq!(report["blue_sizes"], json!([1, 2, 3]));
        assert_eq!(report["windows"]["all_oos"]["cells"].as_array().unwrap().len(), 81);
        assert_eq!(report["next"]["actual_recommendation"], false);
        let (status, _, bytes) = route(&app, "GET", "/api/pools/observations", "", "");
        assert_eq!(status, 200);
        let actual: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(actual["cells"].as_array().unwrap().len(), 18);
        assert!(actual["cells"].as_array().unwrap().iter().all(|c| c["n"] == 0 && c["full_red_rate"].is_null()));
        let body = json!({"data_sha256":"stale", "target":"00000"}).to_string();
        let (status, _, bytes) = route(&app, "POST", "/api/recommendations/freeze", "", &body);
        assert_eq!(status, 400);
        assert!(String::from_utf8(bytes).unwrap().contains("请刷新后留档"));
        assert!(!recommendation_dir(&app).exists());
        assert_eq!(route(&app, "GET", "/pools", "", "").1, "text/html");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn recommendation_api_archives_exports_and_settles_original_tickets() {
        let (dir, app, mut draws) = fixture("recommendations");
        let issue = fmt_issue(draws.last().unwrap().issue + 1);
        let request = json!({"preset": "compound8", "issue": issue});
        let (status, _, bytes) = route(&app, "POST", "/api/generate", "", &request.to_string());
        assert_eq!(status, 200);
        let generated: Value = serde_json::from_slice(&bytes).unwrap();
        let id = generated["recommendation_archive"]["archive_id"].as_str().unwrap().replace(':', "%3A");
        let query = format!("archive={id}&strategy=custom");
        let endpoint = format!("/api/recommendations/issues/{issue}");
        let (status, _, bytes) = route(&app, "GET", &endpoint, &query, ""); assert_eq!(status, 200);
        let pending: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(pending["detail"]["tickets"].as_array().unwrap().len(), 28);
        assert_eq!(pending["compact_preview"]["actual_recommendation"], false);
        assert_eq!(pending["compact_preview"]["summary"]["bets"], 2);
        assert!(pending["compact_preview"]["summary"]["covered_red_count"].is_null());
        assert!(pending["detail"]["summary"]["fixed_prize"].is_null());
        let raw = pending["provenance"]["raw_file"].as_str().unwrap().replace('/', "%2F");
        let (status, _, original) = route(&app, "GET", "/api/recommendations/files", &format!("path={raw}"), ""); assert_eq!(status, 200);
        let mut next = draws.last().unwrap().clone(); next.issue += 1; draws.push(next); write_csv(&app.data, &draws).unwrap();
        let (status, _, bytes) = route(&app, "POST", "/api/recommendations/settle", "", ""); assert_eq!(status, 200);
        assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap()["settled"], 1);
        let (status, _, bytes) = route(&app, "GET", &endpoint, &query, ""); assert_eq!(status, 200);
        let settled: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(settled["provenance"]["review_persisted"], true);
        assert!(!settled["detail"]["summary"]["fixed_prize"].is_null());
        for (saved, reviewed) in generated["tickets"].as_array().unwrap().iter().zip(settled["detail"]["tickets"].as_array().unwrap()) {
            assert_eq!(saved["red"], reviewed["red"]); assert_eq!(saved["blue"], reviewed["blue"]);
        }
        assert_eq!(route(&app, "GET", "/api/recommendations/files", &format!("path={raw}"), "").2, original);
        assert_eq!(route(&app, "GET", "/api/recommendations/files", "path=..%2F..%2Fetc%2Fpasswd", "").0, 400);
        assert_eq!(route(&app, "GET", "/reviews", "", "").1, "text/html");
        assert_eq!(query_value("archive=interactive%3Aa+b", "archive").as_deref(), Some("interactive:a b"));
        assert_eq!(query_value("path=%GG", "path"), None);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_rejects_a_stale_displayed_recommendation_before_writing() {
        let (dir, app, mut draws) = fixture("stale-plan");
        let mut body = json!({"preset": "compound7", "issue": fmt_issue(draws.last().unwrap().issue + 1)});
        let generated = generate_value(&app, &body.to_string()).unwrap();
        body["expected_archive_sha256"] = generated["recommendation_archive"]["sha256"].clone();
        draws[0].blue = draws[0].blue % 16 + 1; write_csv(&app.data, &draws).unwrap();
        let (status, _, bytes) = route(&app, "POST", "/api/plans", "", &body.to_string());
        assert_eq!(status, 400);
        assert!(String::from_utf8(bytes).unwrap().contains("重新生成"));
        assert!(!plans_file(&app).exists());
        assert!(!app.audit.exists());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn frozen_plan_is_saved_only_after_successful_sealing() {
        let (dir, app, mut draws) = fixture("save");
        let target = fmt_issue(draws.last().unwrap().issue + 1);
        let mut body = json!({"preset": "frozen", "issue": target});
        for deadline in [Value::Null, json!(0)] {
            body["not_after"] = deadline;
            assert_eq!(route(&app, "POST", "/api/plans", "", &body.to_string()).0, 400);
            assert!(!plans_file(&app).exists());
            assert!(!app.audit.exists());
            assert!(!app.publish_dir.exists());
        }
        body["not_after"] = json!(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 3600);
        let (status, _, payload) = route(&app, "POST", "/api/plans", "", &body.to_string());
        assert_eq!(status, 200);
        let saved: Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(saved["plan"]["sealed"], true);
        assert!(saved["seal"]["content_sha256"].is_string());
        assert_eq!(fs::read_to_string(plans_file(&app)).unwrap().lines().count(), 1);
        let (status, _, payload) = route(&app, "POST", "/api/plans", "", &body.to_string());
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&payload));
        let repeated: Value = serde_json::from_slice(&payload).unwrap();
        assert_eq!(repeated["already"], true);
        assert_eq!(repeated["seal"]["already"], true);
        draws[0].blue = draws[0].blue % 16 + 1;
        write_csv(&app.data, &draws).unwrap();
        assert_eq!(route(&app, "POST", "/api/plans", "", &body.to_string()).0, 400);
        assert_eq!(fs::read_to_string(plans_file(&app)).unwrap().lines().count(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mismatched_manifest_is_not_displayed_or_settled_as_scored() {
        let (dir, app, mut draws) = fixture("mismatch");
        let cutoff = draws.last().unwrap().issue;
        let deadline = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 3600;
        publish(&app.data, &app.audit, &app.publish_dir, Some(fmt_issue(cutoff + 2)), Some(deadline)).unwrap();
        let mut next = draws.last().unwrap().clone();
        next.issue = cutoff + 1;
        draws.push(next);
        write_csv(&app.data, &draws).unwrap();
        let records = records_value(&app).unwrap();
        assert_eq!(records["records"][0]["status"], "invalid");
        assert!(records["records"][0]["draw"].is_null());
        assert!(records["records"][0]["coverage"].is_null());
        assert!(records["records"][0]["error"].as_str().unwrap().contains("不一致"));
        let overview = overview_value(&app).unwrap();
        assert_eq!(overview["scored"], 0);
        assert_eq!(overview["invalid"], 1);
        let settled = settle_value(&app).unwrap();
        assert_eq!(settled["settled"], 0);
        assert_eq!(settled["errors"].as_array().unwrap().len(), 1);
        assert_eq!(fs::read_to_string(&app.audit).unwrap().lines().count(), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_cache_shares_items_and_refreshes_when_data_changes() {
        let (dir, app, mut draws) = fixture("cache");
        let first = load_history(&app).unwrap();
        let again = load_history(&app).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        let mut next = draws.last().unwrap().clone();
        next.issue += 1;
        draws.push(next);
        write_csv(&app.data, &draws).unwrap();
        let refreshed = load_history(&app).unwrap();
        assert!(!Arc::ptr_eq(&first, &refreshed));
        assert_eq!(refreshed.len(), first.len() + 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn all_new_pool_algorithms_build_their_own_balanced_pool_and_keep_protocol_separate() {
        let draws=synth_draws(220,9);let base=prepare_forecast(&draws).unwrap();
        let saved=crate::pool_algorithms::saved_all(&base,&draws);
        for (i,model) in crate::pool_algorithms::MODELS.into_iter().enumerate() {
            let body=json!({"mode":"balanced","ranking_model":model,"pool":16,"tickets":8,"blue_count":2}).to_string();
            let (plan,prepared)=build_requested(&body,&draws).unwrap();
            assert_eq!(prepared.ranking.as_ref().unwrap(),&saved[i]);
            assert_eq!(prepared.forecast.blue_order,base.forecast.blue_order);
            assert_eq!(plan.reds.iter().copied().collect::<std::collections::BTreeSet<_>>(),saved[i]["red_ranking"].as_array().unwrap()[..16].iter().map(|n|n.as_u64().unwrap()as u8).collect());
            for n in &plan.reds {assert_eq!(plan.tickets.iter().filter(|t|t.red.contains(n)).count(),3);}
            let invalid=json!({"preset":"frozen","ranking_model":model}).to_string();
            assert!(build_requested(&invalid,&draws).is_err());
        }
    }

    #[test]
    fn weighted24_uses_original_pool_and_rejects_trial_rank_substitution() {
        let draws=synth_draws(220,9);let base=prepare_forecast(&draws).unwrap();
        let (plan,prepared)=build_requested(r#"{"preset":"weighted24"}"#,&draws).unwrap();
        assert!(prepared.ranking.is_none());assert_eq!(prepared.forecast.red_order,base.forecast.red_order);
        assert_eq!(plan.reds.len(),24);assert_eq!(plan.tickets.len(),8);
        for n in &plan.reds {assert_eq!(plan.tickets.iter().filter(|t|t.red.contains(n)).count(),2);}
        assert!(build_requested(r#"{"preset":"weighted24","ranking_model":"conditional_gap"}"#,&draws).is_err());
    }

    #[test]
    fn equal_preset_uses_saved_equal_pool_and_rejects_model_substitution() {
        let draws = synth_draws(220, 9);
        let original = prepare_forecast(&draws).unwrap();
        let saved = crate::pool::saved(&original);
        let (plan, prepared) = build_requested(r#"{"preset":"equal24"}"#, &draws).unwrap();
        assert_eq!(prepared.ranking.as_ref().unwrap()["red_ranking"], saved["comparison_models"][0]["red_ranking"]);
        assert_eq!(prepared.forecast.blue_order, original.forecast.blue_order);
        assert_eq!(plan.tickets.len(), 8);
        assert_eq!(plan.reds.len(), 24);
        for n in &plan.reds {
            assert_eq!(plan.tickets.iter().filter(|t|t.red.contains(n)).count(), 2);
        }
        for body in [r#"{"preset":"equal24","ranking_model":"weighted"}"#,
            r#"{"preset":"frozen","ranking_model":"equal"}"#] {
            assert!(build_requested(body, &draws).is_err());
        }
        let (small, _) = build_requested(r#"{"mode":"balanced","ranking_model":"equal","pool":14,"tickets":8,"blue_count":2}"#, &draws).unwrap();
        assert_eq!(small.reds.len(), 14);
    }

    #[test]
    fn page_lists_tickets_for_synthetic_history() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ssq-web-{stamp}"));
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("ssq.csv");
        write_csv(&data, &synth_draws(220, 9)).unwrap();
        let app = App {
            data,
            audit: dir.join("audit.jsonl"),
            publish_dir: dir.join("publish"),
            history_dir: dir.join("history"),
            report: Mutex::new(None),
            history: Mutex::new(None),
            mutation: Mutex::new(()),
        };
        let (_status, kind, page) = route(&app, "GET", "/", "", "");
        assert_eq!(kind, "text/html");
        let html = String::from_utf8(page).unwrap();
        for panel in ["generator", "records", "reviews", "draws", "backtest", "research", "settings"] {
            assert!(html.contains(&format!("panel-{panel}")));
        }
        let (status, _, body) = route(&app, "POST", "/api/generate", "", r#"{"preset":"compound7"}"#);
        assert_eq!(status, 200);
        let generated: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(generated["bet_count"].as_u64().unwrap(), 7);
        assert_eq!(generated["tickets"].as_array().unwrap().len(), 7);
        assert_eq!(generated["cost"].as_u64().unwrap(), 14);
        assert_eq!(generated["issue"], fmt_issue(3221));
        assert_eq!(generated["input_history_sha256"], data_sha256(&load_csv(&app.data).unwrap()));
        let (status, _, _) = route(&app, "POST", "/api/generate", "", r#"{"mode":"dantuo","bankers":0,"drags":6}"#);
        assert_eq!(status, 400);
        let (status, _, body) = route(&app, "POST", "/api/plans", "", r#"{"preset":"compound7","issue":"99001"}"#);
        assert_eq!(status, 200);
        let saved: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(saved["already"].as_bool().unwrap(), false);
        assert_eq!(saved["bet_count"].as_u64().unwrap(), 7);
        let (status, _, body) = route(&app, "GET", "/api/records", "", "");
        assert_eq!(status, 200);
        let records: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(records["records"].as_array().unwrap().len(), 1);
        assert_eq!(records["records"][0]["status"].as_str().unwrap(), "pending");
        let (status, _, body) = route(&app, "GET", "/api/forecast", "", "");
        assert_eq!(status, 200);
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["tickets"].as_array().unwrap().len(), 8);
        let (status, _, body) = route(&app, "POST", "/api/generate", "", r#"{"preset":"frozen"}"#);
        assert_eq!(status, 200);
        let frozen: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(frozen["tickets"], value["tickets"]);
        assert_eq!(frozen["sealed"], false);
        assert_eq!(frozen["requires_seal"], true);
        let (status, _, body) = route(&app, "POST", "/api/generate", "", r#"{"mode":"single","tickets":5,"pool":30,"overlap":0,"blue_count":2}"#);
        assert_eq!(status, 200);
        let split: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(split["tickets"].as_array().unwrap().len(), 5);
        assert_eq!(value["last_draw"]["red"].as_array().unwrap().len(), 6);
        assert!(value["blues"].as_array().unwrap().len() >= 2);
        let (status, _, body) = route(&app, "GET", "/api/history", "limit=5", "");
        assert_eq!(status, 200);
        let hist: Value = serde_json::from_slice(&body).unwrap();
        assert!(hist["total"].as_u64().unwrap() > 5);
        assert_eq!(hist["rows"].as_array().unwrap().len(), 5);
        assert!(hist["rows"][0].get("red_hit_n").is_some());
        assert!(hist["summary"]["mean_red_hit"].as_f64().unwrap() >= 0.0);
        let issue = hist["rows"][0]["issue"].as_str().unwrap();
        let (status, _, body) = route(&app, "GET", &format!("/api/history/{issue}"), "", "");
        assert_eq!(status, 200);
        let one: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(one["review"]["red_draw_n"].as_u64().unwrap(), 6);
        assert!(one["review"]["covered_red_n"].as_u64().unwrap() >= 6);
        let (status, _, _) = route(&app, "GET", "/api/cycle", "", "");
        assert_eq!(status, 200);
        let (status, _, _) = route(&app, "GET", "/missing", "", "");
        assert_eq!(status, 404);
        let _ = fs::remove_dir_all(&dir);
    }
}
