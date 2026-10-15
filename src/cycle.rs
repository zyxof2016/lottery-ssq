use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::audit::{publish, verify};
use crate::draw::{fetch_500, fmt_date, fmt_issue, load_csv, Draw};
use crate::eval::prepare_forecast;
use crate::review;
use crate::recommendations;

#[derive(Clone, Debug)]
pub struct Gate {
    pub publish: bool,
    pub synced: bool,
    pub target: String,
    pub deadline_unix: u64,
    pub deadline_text: String,
    pub completed_date: i32,
    pub reason: String,
}

pub fn run_cycle(data: &Path, audit: &Path, publish_dir: &Path, review_dir: &Path) -> Result<String, String> {
    let mut notes = Vec::new();
    let mut synced = false;
    let mut draws = load_csv(data)?;
    let completed = last_completed_draw_date(now_unix());
    let stale = draws.last().map(|d| d.date < completed).unwrap_or(true);
    if stale {
        match fetch_500(data) {
            Ok(n) => {
                synced = true;
                notes.push(format!("已向500.com核对，本地 {} 期", n));
                draws = load_csv(data)?;
            }
            Err(e) => notes.push(format!("500.com核对失败，沿用本地数据：{e}")),
        }
    } else {
        notes.push("本地开奖已经覆盖上一档开奖日，本轮不重复下载".into());
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    let gate = decide(now_unix(), last.issue, last.date, synced || !stale);
    let recommendation_dir = publish_dir.parent().unwrap_or_else(|| Path::new(".")).join("recommendations");
    notes.push(gate.reason.clone());
    if gate.publish {
        fs::create_dir_all(publish_dir).map_err(|e| format!("无法创建发布目录：{e}"))?;
        if let Some(parent) = audit.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("无法创建审计目录：{e}"))?;
        }
        match publish(data, audit, publish_dir, Some(gate.target.clone()), Some(gate.deadline_unix)) {
            Ok(outcome) => {
                if outcome.already {
                    notes.push(format!("本期 {} 的开奖前哈希已在", outcome.target));
                } else {
                    notes.push(format!("已封存 {}，哈希 {}", outcome.target, outcome.content_sha256));
                }
                match recommendations::freeze_daily(&recommendation_dir, &draws, &gate.target, gate.deadline_unix) {
                    Ok(value) => notes.push(format!("每期推荐 {}：{}（{} 个主模型池、{} 个对照池、{} 个票组）", gate.target, value["status"].as_str().unwrap_or(""), value["number_pools"].as_u64().unwrap_or(0), value["comparison_number_pools"].as_u64().unwrap_or(0), value["strategies"].as_u64().unwrap_or(0))),
                    Err(error) => notes.push(format!("每期推荐未留档：{error}")),
                }
            }
            Err(e) => notes.push(format!("封存没有写入：{e}")),
        }
    }
    let settlement = recommendations::settle_archives(&recommendation_dir, &draws);
    notes.push(format!("实际推荐复盘新增 {} 份", settlement.settled));
    notes.extend(settlement.errors);
    notes.extend(write_reviews(data, audit, publish_dir, review_dir, &draws)?);
    let text = notes.join("\n");
    append_log(&log_path(review_dir), &text)?;
    Ok(text)
}

pub fn decide(now_unix: u64, latest_issue: u32, latest_date: i32, synced: bool) -> Gate {
    let (deadline_unix, deadline_text, target) = next_deadline(now_unix, latest_issue);
    let required = draw_day_before(deadline_unix);
    let waited = now_unix >= required_ready_unix(required);
    let (publish, reason) = if now_unix >= deadline_unix {
        (false, format!("已过 {deadline_text}，这一档不再封存"))
    } else if latest_date >= required {
        (
            true,
            format!("在 {deadline_text} 前封存 {target}。依据开奖到 {}", fmt_issue(latest_issue)),
        )
    } else if synced && waited {
        (
            true,
            format!(
                "500.com最新仍是 {}，{} 没有新号码。在 {deadline_text} 前封存 {target}",
                fmt_date(latest_date),
                fmt_date(required)
            ),
        )
    } else {
        (
            false,
            format!(
                "本地开奖停在 {}，封存 {target} 前要先有 {} 的号码。这期先不封存",
                fmt_date(latest_date),
                fmt_date(required)
            ),
        )
    };
    Gate {
        publish,
        synced,
        target,
        deadline_unix,
        deadline_text,
        completed_date: required,
        reason,
    }
}

fn write_reviews(
    data: &Path,
    audit: &Path,
    publish_dir: &Path,
    review_dir: &Path,
    draws: &[Draw],
) -> Result<Vec<String>, String> {
    let mut notes = Vec::new();
    let Ok(rd) = fs::read_dir(publish_dir) else {
        return Ok(notes);
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path).map_err(|e| format!("无法读取 {}: {e}", path.display()))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| format!("{} 不是清单：{e}", path.display()))?;
        let canonical = value.get("canonical").and_then(|v| v.as_str()).unwrap_or("");
        let target = manifest_field(canonical, "target").unwrap_or_default();
        let sha = value.get("content_sha256").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let cutoff = manifest_field(canonical, "cutoff_issue").unwrap_or_default();
        let Ok(cutoff_n) = cutoff.parse::<u32>() else { continue };
        let Some(draw) = draws.iter().find(|d| fmt_issue(d.issue) == target) else {
            continue;
        };
        let first = draws.iter().find(|d| d.issue > cutoff_n);
        if first.map(|d| d.issue) != Some(draw.issue) {
            notes.push(format!("{target} 不是截止期后的第一期，跳过自动复盘"));
            continue;
        }
        if !audit_has_verify(audit, &target, &sha) {
            match verify(data, &path, Some(audit)) {
                Ok(_) => notes.push(format!("已复算 {target}")),
                Err(e) => {
                    notes.push(format!("{target} 复算失败：{e}"));
                    continue;
                }
            }
        }
        match write_one_review(review_dir, draws, cutoff_n, &target, &sha) {
            Ok(p) => notes.push(format!("复盘 {} 已写入 {}", target, p.display())),
            Err(e) => notes.push(format!("{target} 复盘失败：{e}")),
        }
    }
    Ok(notes)
}

fn write_one_review(review_dir: &Path, draws: &[Draw], cutoff: u32, target: &str, sha: &str) -> Result<PathBuf, String> {
    let prefix: Vec<Draw> = draws.iter().filter(|d| d.issue <= cutoff).cloned().collect();
    let prepared = prepare_forecast(&prefix)?;
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == target).ok_or("复盘时找不到开奖")?;
    let review = review::explain(&prepared.forecast, &draw.red, draw.blue);
    let tickets: Vec<Value> = prepared
        .forecast
        .tickets
        .iter()
        .zip(prepared.forecast.ticket_blues.iter())
        .map(|(mask, blue)| json!({"red": mask_nums(*mask), "blue": *blue as u8 + 1}))
        .collect();
    let body = json!({
        "kind": "sealed-review",
        "issue": target,
        "date": fmt_date(draw.date),
        "content_sha256": sha,
        "cutoff_issue": fmt_issue(cutoff),
        "actual_red": draw.red,
        "actual_blue": draw.blue,
        "tickets": tickets,
        "review": review,
    });
    fs::create_dir_all(review_dir).map_err(|e| format!("无法创建复盘目录：{e}"))?;
    let path = review_dir.join(format!("{target}.json"));
    let text = serde_json::to_string_pretty(&body).map_err(|e| e.to_string())?;
    fs::write(&path, text + "\n").map_err(|e| format!("无法写入复盘：{e}"))?;
    Ok(path)
}

pub fn read_review(review_dir: &Path, issue: &str) -> Result<Value, String> {
    if issue.len() != 5 || !issue.chars().all(|c| c.is_ascii_digit()) {
        return Err("期号需要 5 位数字".into());
    }
    let path = review_dir.join(format!("{issue}.json"));
    let text = fs::read_to_string(&path).map_err(|_| format!("还没有 {issue} 的封存复盘"))?;
    serde_json::from_str(&text).map_err(|e| format!("复盘无法解析：{e}"))
}

pub fn list_reviews(review_dir: &Path) -> Vec<Value> {
    let mut rows = Vec::new();
    let Ok(rd) = fs::read_dir(review_dir) else {
        return rows;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else { continue };
        let Ok(value) = serde_json::from_str::<Value>(&text) else { continue };
        let review = value.get("review");
        rows.push(json!({
            "issue": value.get("issue").and_then(|v| v.as_str()).unwrap_or(""),
            "date": value.get("date").and_then(|v| v.as_str()).unwrap_or(""),
            "sha": value.get("content_sha256").and_then(|v| v.as_str()).unwrap_or(""),
            "red_hit_n": review.and_then(|v| v.get("red_hit_n")).and_then(|v| v.as_u64()).unwrap_or(0),
            "red_draw_n": review.and_then(|v| v.get("red_draw_n")).and_then(|v| v.as_u64()).unwrap_or(6),
            "blue_covered": review.and_then(|v| v.get("blue_covered")).and_then(|v| v.as_bool()).unwrap_or(false),
            "covered_red_n": review.and_then(|v| v.get("covered_red_n")).and_then(|v| v.as_u64()).unwrap_or(0),
        }));
    }
    rows.sort_by(|a, b| b["issue"].as_str().unwrap_or("").cmp(a["issue"].as_str().unwrap_or("")));
    rows
}

pub fn log_tail(review_dir: &Path, limit: usize) -> Vec<String> {
    let Ok(text) = fs::read_to_string(log_path(review_dir)) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    lines.iter().rev().take(limit).rev().map(|s| s.to_string()).collect()
}

fn log_path(review_dir: &Path) -> PathBuf {
    review_dir
        .parent()
        .map(|p| p.join("cycle.log"))
        .unwrap_or_else(|| PathBuf::from("cycle.log"))
}

fn append_log(path: &Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("无法创建日志目录：{e}"))?;
    }
    let stamp = fmt_cst(now_unix());
    let mut block = format!("## {stamp}\n{text}\n");
    if !block.ends_with('\n') {
        block.push('\n');
    }
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("无法写周期日志：{e}"))?;
    file.write_all(block.as_bytes()).map_err(|e| format!("无法写周期日志：{e}"))?;
    Ok(())
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

fn manifest_field(canonical: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    canonical.lines().find_map(|line| line.strip_prefix(&prefix).map(|v| v.to_string()))
}

fn mask_nums(mask: u64) -> Vec<u8> {
    let mut nums = Vec::new();
    let mut bits = mask;
    while bits != 0 {
        nums.push((bits.trailing_zeros() + 1) as u8);
        bits &= bits - 1;
    }
    nums
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn fmt_cst(unix: u64) -> String {
    let (y, m, d, h, min, _) = cst_parts(unix);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{min:02}")
}

fn next_deadline(now_unix: u64, latest_issue: u32) -> (u64, String, String) {
    for add in 0..8 {
        let unix = now_unix + add * 86400;
        let (y, m, d, _, _, wd) = cst_parts(unix);
        if wd != 0 && wd != 2 && wd != 4 {
            continue;
        }
        let deadline = unix_cst(y, m, d, 21, 10);
        if deadline > now_unix {
            let year_prefix = (y % 100) as u32;
            let target_n = if latest_issue / 1000 != year_prefix {
                year_prefix * 1000 + 1
            } else {
                latest_issue + 1
            };
            return (deadline, fmt_cst(deadline), format!("{target_n:05}"));
        }
    }
    (now_unix, fmt_cst(now_unix), format!("{:05}", latest_issue + 1))
}

fn draw_day_before(deadline_unix: u64) -> i32 {
    for back in 1..8 {
        let unix = deadline_unix.saturating_sub(back * 86400);
        let (y, m, d, _, _, wd) = cst_parts(unix);
        if wd == 0 || wd == 2 || wd == 4 {
            return y * 10000 + m as i32 * 100 + d as i32;
        }
    }
    0
}

fn required_ready_unix(yyyymmdd: i32) -> u64 {
    let y = yyyymmdd / 10000;
    let m = ((yyyymmdd / 100) % 100) as u32;
    let d = (yyyymmdd % 100) as u32;
    unix_cst(y, m, d, 21, 15) + 12 * 3600
}

fn last_completed_draw_date(now_unix: u64) -> i32 {
    for back in 0..8 {
        let unix = now_unix.saturating_sub(back * 86400);
        let (y, m, d, h, min, wd) = cst_parts(unix);
        if wd != 0 && wd != 2 && wd != 4 {
            continue;
        }
        let done = back > 0 || h > 21 || (h == 21 && min >= 15);
        if done {
            return y * 10000 + m as i32 * 100 + d as i32;
        }
    }
    0
}

fn cst_parts(unix: u64) -> (i32, u32, u32, u32, u32, u32) {
    let cst = unix + 8 * 3600;
    let days = (cst / 86400) as i64;
    let tod = (cst % 86400) as u32;
    let (y, m, d) = civil_from_days(days);
    let wd = ((days + 4) % 7) as u32;
    (y, m, d, tod / 3600, (tod % 3600) / 60, wd)
}

fn unix_cst(y: i32, m: u32, d: u32, h: u32, min: u32) -> u64 {
    let days = days_from_civil(y, m, d);
    (days * 86400 + h as i64 * 3600 + min as i64 * 60 - 8 * 3600) as u64
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_and_publish_gate() {
        let epoch = cst_parts(0);
        assert_eq!((epoch.0, epoch.1, epoch.2, epoch.3, epoch.5), (1970, 1, 1, 8, 4));
        let monday = unix_cst(2026, 10, 12, 9, 50);
        let parts = cst_parts(monday);
        assert_eq!((parts.0, parts.1, parts.2, parts.5), (2026, 10, 12, 1));
        let stale = decide(monday, 26115, 20261008, false);
        assert!(!stale.publish);
        assert!(stale.reason.contains("不封存"));
        let ready = decide(monday, 26116, 20261011, true);
        assert!(ready.publish);
        assert_eq!(ready.target, "26117");
        assert_eq!(ready.deadline_text, "2026-10-13 21:10");
        let late = decide(unix_cst(2026, 10, 13, 21, 11), 26116, 20261011, true);
        assert!(!late.publish);
        let before_sunday = decide(unix_cst(2026, 10, 11, 18, 0), 26115, 20261008, true);
        assert!(before_sunday.publish);
        assert_eq!(before_sunday.target, "26116");
    }
}
