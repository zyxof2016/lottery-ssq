use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::draw::{data_sha256, fmt_issue, load_csv, Draw};
use crate::eval::prepare_forecast;
use crate::hash::sha256_hex;
use crate::protocol::Protocol;
use crate::select::{max_red_hits, Forecast};

#[derive(Serialize, Deserialize)]
struct ManifestFile {
    published_at_unix: u64,
    content_sha256: String,
    canonical: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct AuditRec {
    ts_unix: u64,
    kind: String,
    target: String,
    content_sha256: String,
    data_sha256: String,
    cutoff_issue: String,
    ok: bool,
    scored_issue: String,
}

pub struct PublishOutcome {
    pub path: PathBuf,
    pub content_sha256: String,
    pub already: bool,
    pub target: String,
}

pub struct ScoreCard {
    pub issue: u32,
    pub date: i32,
    pub red: [u8; 6],
    pub blue: u8,
    pub best_red: u32,
    pub ticket_red: Vec<u32>,
    pub ticket_blue: Vec<bool>,
    pub top1_blue: bool,
    pub any_blue: bool,
}

pub struct VerifyOutcome {
    pub content_sha256: String,
    pub target: String,
    pub cutoff_issue: u32,
    pub score: Option<ScoreCard>,
}

pub fn publish(
    data: &Path,
    audit: &Path,
    out_dir: &Path,
    target: Option<String>,
    not_after: Option<u64>,
) -> Result<PublishOutcome, String> {
    let deadline = not_after.ok_or("开奖前发布必须提供截止时间 --not-after")?;
    check_deadline(deadline)?;
    let draws = load_csv(data)?;
    let prepared = prepare_forecast(&draws)?;
    let target = target
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| format!("after:{}", fmt_issue(prepared.cutoff_issue)));
    target_draw(&draws, prepared.cutoff_issue, &target)?;
    let canonical = canonical_manifest(
        &prepared.forecast,
        &prepared.weights,
        &prepared.blue_weights,
        prepared.n,
        prepared.cutoff_issue,
        prepared.cutoff_date,
        &prepared.data_sha,
        &target,
    );
    let digest = sha256_hex(canonical.as_bytes());
    let existing = read_audit(audit)?;
    if let Some(prev) = existing
        .iter()
        .find(|r| r.kind == "publish" && r.target == target)
    {
        if prev.content_sha256 != digest {
            return Err(format!(
                "目标 {target} 已有不同留痕 {}，拒绝覆盖",
                prev.content_sha256
            ));
        }
        let path = manifest_path(out_dir, &target, &digest);
        if !path.exists() {
            check_deadline(deadline)?;
            write_manifest(&path, &canonical, &digest)?;
        }
        return Ok(PublishOutcome {
            path,
            content_sha256: digest,
            already: true,
            target,
        });
    }
    check_deadline(deadline)?;
    fs::create_dir_all(out_dir).map_err(|e| format!("无法创建目录 {}: {e}", out_dir.display()))?;
    let path = manifest_path(out_dir, &target, &digest);
    write_manifest(&path, &canonical, &digest)?;
    append_audit(
        audit,
        &AuditRec {
            ts_unix: now_unix(),
            kind: "publish".into(),
            target: target.clone(),
            content_sha256: digest.clone(),
            data_sha256: prepared.data_sha,
            cutoff_issue: fmt_issue(prepared.cutoff_issue),
            ok: true,
            scored_issue: String::new(),
        },
    )?;
    Ok(PublishOutcome {
        path,
        content_sha256: digest,
        already: false,
        target,
    })
}

pub fn verify(data: &Path, manifest: &Path, audit: Option<&Path>) -> Result<VerifyOutcome, String> {
    let text = fs::read_to_string(manifest)
        .map_err(|e| format!("无法读取清单 {}: {e}", manifest.display()))?;
    let file: ManifestFile = serde_json::from_str(&text).map_err(|e| format!("清单不是合法 JSON：{e}"))?;
    let stored_hash = sha256_hex(file.canonical.as_bytes());
    if stored_hash != file.content_sha256 {
        return Err("清单内的规范文本和声明的哈希不一致，文件已被改动".into());
    }
    let target = field(&file.canonical, "target")?;
    let cutoff: u32 = field(&file.canonical, "cutoff_issue")?
        .parse()
        .map_err(|_| "清单截止期无法解析".to_string())?;
    let draws = load_csv(data)?;
    let mut prefix = Vec::new();
    for d in &draws {
        if d.issue <= cutoff {
            prefix.push(d.clone());
        }
    }
    if prefix.last().map(|d| d.issue) != Some(cutoff) {
        return Err(format!("数据中没有截止期 {}", fmt_issue(cutoff)));
    }
    if data_sha256(&prefix) != field(&file.canonical, "data")? {
        return Err("截止期以前的数据哈希和清单不一致".into());
    }
    let prepared = prepare_forecast(&prefix)?;
    let recomputed = canonical_manifest(
        &prepared.forecast,
        &prepared.weights,
        &prepared.blue_weights,
        prepared.n,
        prepared.cutoff_issue,
        prepared.cutoff_date,
        &prepared.data_sha,
        &target,
    );
    let recomputed_hash = sha256_hex(recomputed.as_bytes());
    if recomputed_hash != file.content_sha256 {
        return Err(format!(
            "复算哈希 {} 与清单 {} 不同。{}",
            recomputed_hash,
            file.content_sha256,
            first_diff(&file.canonical, &recomputed)
        ));
    }
    let score = target_draw(&draws, cutoff, &target)?
        .map(|draw| score_draw(&prepared.forecast, draw));
    if let Some(path) = audit {
        if let Some(card) = &score {
            append_audit(
                path,
                &AuditRec {
                    ts_unix: now_unix(),
                    kind: "verify".into(),
                    target: target.clone(),
                    content_sha256: file.content_sha256.clone(),
                    data_sha256: prepared.data_sha.clone(),
                    cutoff_issue: fmt_issue(cutoff),
                    ok: true,
                    scored_issue: fmt_issue(card.issue),
                },
            )?;
        }
    }
    Ok(VerifyOutcome {
        content_sha256: file.content_sha256,
        target,
        cutoff_issue: cutoff,
        score,
    })
}

/// 清单只预测截止期后的第一期；旧版 after:截止期 清单继续使用这个含义。
pub(crate) fn target_draw<'a>(
    draws: &'a [Draw],
    cutoff: u32,
    target: &str,
) -> Result<Option<&'a Draw>, String> {
    let next = draws.iter().find(|draw| draw.issue > cutoff);
    if target == format!("after:{}", fmt_issue(cutoff)) {
        return Ok(next);
    }
    if target.len() != 5 || !target.bytes().all(|c| c.is_ascii_digit()) {
        return Err("目标期号必须是 5 位数字，或 after:截止期".into());
    }
    let issue: u32 = target.parse().map_err(|_| "目标期号无法解析")?;
    if issue <= cutoff {
        return Err(format!("目标期号 {target} 必须晚于截止期 {}", fmt_issue(cutoff)));
    }
    if let Some(draw) = next {
        if draw.issue != issue {
            return Err(format!(
                "目标期号 {target} 与截止期后的第一期开奖 {} 不一致，拒绝计分",
                fmt_issue(draw.issue)
            ));
        }
    }
    Ok(next)
}

fn check_deadline(deadline: u64) -> Result<(), String> {
    let now = now_unix();
    if now >= deadline {
        return Err(format!("已到留痕截止时间 {deadline}，当前 {now}，拒绝发布"));
    }
    Ok(())
}

pub fn render_score(card: &ScoreCard) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "核对 {}（{}） 红球 {:02} {:02} {:02} {:02} {:02} {:02} 蓝球 {:02}",
        fmt_issue(card.issue),
        crate::draw::fmt_date(card.date),
        card.red[0],
        card.red[1],
        card.red[2],
        card.red[3],
        card.red[4],
        card.red[5],
        card.blue
    ));
    lines.push(format!("最佳票红球命中 {}", card.best_red));
    for (i, (red, blue)) in card.ticket_red.iter().zip(card.ticket_blue.iter()).enumerate() {
        lines.push(format!(
            "票 {i} 红球命中 {red} 蓝球 {}",
            if *blue { "命中" } else { "未中" }
        ));
    }
    lines.push(format!(
        "融合蓝球第一名 {}；票面蓝球 {}",
        if card.top1_blue { "命中" } else { "未中" },
        if card.any_blue { "有命中" } else { "未命中" }
    ));
    lines.join("\n")
}

fn score_draw(fc: &Forecast, draw: &Draw) -> ScoreCard {
    let best_red = max_red_hits(&fc.tickets, draw.red_mask);
    let ticket_red = fc
        .tickets
        .iter()
        .map(|t| (t & draw.red_mask).count_ones())
        .collect();
    let y = draw.blue as usize - 1;
    let ticket_blue = fc.ticket_blues.iter().map(|b| *b == y).collect();
    ScoreCard {
        issue: draw.issue,
        date: draw.date,
        red: draw.red,
        blue: draw.blue,
        best_red,
        ticket_red,
        ticket_blue,
        top1_blue: fc.blue_order[0] == y,
        any_blue: fc.ticket_blues.iter().any(|b| *b == y),
    }
}

fn canonical_manifest(
    fc: &Forecast,
    weights: &[f64; 7],
    blue_weights: &[f64; 6],
    n: usize,
    cutoff_issue: u32,
    cutoff_date: i32,
    data_sha: &str,
    target: &str,
) -> String {
    let mut s = String::new();
    s.push_str("ssq-manifest-v1\n");
    s.push_str(&format!("protocol={}\n", Protocol::sha256()));
    s.push_str(&format!("data={data_sha}\n"));
    s.push_str(&format!("n={n}\n"));
    s.push_str(&format!("cutoff_issue={}\n", fmt_issue(cutoff_issue)));
    s.push_str(&format!("cutoff_date={cutoff_date:08}\n"));
    s.push_str(&format!("target={target}\n"));
    s.push_str("red_weight=");
    s.push_str(&weight_list(Protocol::EXPERTS.as_slice(), weights));
    s.push('\n');
    s.push_str("blue_weight=");
    s.push_str(&weight_list(Protocol::BLUES.as_slice(), blue_weights));
    s.push('\n');
    s.push_str("red_rank=");
    s.push_str(&join_idx(&fc.red_order));
    s.push('\n');
    s.push_str("blue_rank=");
    s.push_str(&join_idx(&fc.blue_order));
    s.push('\n');
    s.push_str("pool=");
    s.push_str(&join_idx(&fc.pool));
    s.push('\n');
    s.push_str("rescued=");
    s.push_str(&join_idx_or_dash(&fc.mild.rescued));
    s.push('\n');
    s.push_str("displaced=");
    s.push_str(&join_idx_or_dash(&fc.mild.displaced));
    s.push('\n');
    for (i, (mask, blue)) in fc.tickets.iter().zip(fc.ticket_blues.iter()).enumerate() {
        let reds = mask_reds(*mask);
        s.push_str(&format!(
            "ticket{i}={:02},{:02},{:02},{:02},{:02},{:02}+{:02}\n",
            reds[0], reds[1], reds[2], reds[3], reds[4], reds[5], *blue as u8 + 1
        ));
    }
    s
}

fn weight_list(names: &[&str], weights: &[f64]) -> String {
    names
        .iter()
        .zip(weights.iter())
        .map(|(n, w)| format!("{n}:{w:.12}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn join_idx(idx: &[usize]) -> String {
    idx.iter()
        .map(|i| format!("{:02}", *i as u8 + 1))
        .collect::<Vec<_>>()
        .join(",")
}

fn join_idx_or_dash(idx: &[usize]) -> String {
    if idx.is_empty() {
        "-".into()
    } else {
        join_idx(idx)
    }
}

fn mask_reds(mask: u64) -> [u8; 6] {
    let mut reds = [0u8; 6];
    let mut n = 0;
    for i in 0..33 {
        if (mask >> i) & 1 == 1 {
            reds[n] = i as u8 + 1;
            n += 1;
        }
    }
    reds
}

fn field(canonical: &str, key: &str) -> Result<String, String> {
    let prefix = format!("{key}=");
    canonical
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).map(|v| v.to_string()))
        .ok_or_else(|| format!("清单缺少 {key}"))
}

fn first_diff(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("第 {} 行不同。发布：{x} 复算：{y}", i + 1);
        }
    }
    if a.lines().count() != b.lines().count() {
        "行数不同".into()
    } else {
        "文本相同但哈希不同".into()
    }
}

fn write_manifest(path: &Path, canonical: &str, digest: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("无法创建目录 {}: {e}", parent.display()))?;
        }
    }
    let file = ManifestFile {
        published_at_unix: now_unix(),
        content_sha256: digest.to_string(),
        canonical: canonical.to_string(),
    };
    let body = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    fs::write(path, body + "\n").map_err(|e| format!("无法写入 {}: {e}", path.display()))
}

fn manifest_path(dir: &Path, target: &str, digest: &str) -> PathBuf {
    let safe: String = target
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{safe}_{}.json", &digest[..12]))
}

fn read_audit(path: &Path) -> Result<Vec<AuditRec>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(path).map_err(|e| format!("无法读取审计日志 {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let rec: AuditRec = serde_json::from_str(line)
            .map_err(|e| format!("审计日志第 {} 行无法解析：{e}", i + 1))?;
        out.push(rec);
    }
    Ok(out)
}

fn append_audit(path: &Path, rec: &AuditRec) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("无法创建目录 {}: {e}", parent.display()))?;
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("无法打开审计日志 {}: {e}", path.display()))?;
    let line = serde_json::to_string(rec).map_err(|e| e.to_string())?;
    writeln!(file, "{line}").map_err(|e| format!("无法追加审计日志：{e}"))?;
    Ok(())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{synth_draws, write_csv};

    fn fixture(name: &str, draws: &[Draw]) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ssq-audit-{name}-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("ssq.csv");
        let audit = dir.join("log.jsonl");
        let out = dir.join("pub");
        write_csv(&data, draws).unwrap();
        (dir, data, audit, out)
    }

    #[test]
    fn publish_then_verify_and_refuse_rewrite() {
        let mut draws = synth_draws(180, 21);
        let (dir, data, audit, out) = fixture("roundtrip", &draws);
        let target = fmt_issue(draws.last().unwrap().issue + 1);
        let deadline = Some(now_unix() + 3600);
        let past = now_unix().saturating_sub(10);
        let err = publish(&data, &audit, &out, Some(target.clone()), Some(past));
        assert!(err.is_err());
        assert!(publish(&data, &audit, &out, Some(target.clone()), None).is_err());
        assert!(publish(&data, &audit, &out, Some(target.clone()), Some(now_unix())).is_err());
        assert!(!out.exists());
        assert!(!audit.exists());
        let first = publish(&data, &audit, &out, Some(target.clone()), deadline).unwrap();
        assert!(!first.already);
        let again = publish(&data, &audit, &out, Some(target.clone()), deadline).unwrap();
        assert!(again.already);
        assert_eq!(again.content_sha256, first.content_sha256);
        let checked = verify(&data, &first.path, None).unwrap();
        assert!(checked.score.is_none());
        let original_blue = draws[0].blue;
        draws[0].blue = original_blue % 16 + 1;
        write_csv(&data, &draws).unwrap();
        assert!(publish(&data, &audit, &out, Some(target), deadline).err().unwrap().contains("拒绝覆盖"));
        draws[0].blue = original_blue;
        let extra = synth_draws(1, 99);
        let mut one = extra[0].clone();
        one.issue = draws.last().unwrap().issue + 1;
        one.date = draws.last().unwrap().date + 1;
        draws.push(one);
        write_csv(&data, &draws).unwrap();
        let scored = verify(&data, &first.path, Some(&audit)).unwrap();
        assert_eq!(scored.score.unwrap().issue, draws.last().unwrap().issue);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_rejects_a_different_next_issue_without_writing_a_score() {
        let mut draws = synth_draws(180, 21);
        let (dir, data, audit, out) = fixture("target", &draws);
        let cutoff = draws.last().unwrap().issue;
        let target = fmt_issue(cutoff + 2);
        let published = publish(&data, &audit, &out, Some(target), Some(now_unix() + 3600)).unwrap();
        assert!(verify(&data, &published.path, None).unwrap().score.is_none());
        let mut next = draws.last().unwrap().clone();
        next.issue = cutoff + 1;
        draws.push(next.clone());
        write_csv(&data, &draws).unwrap();
        let error = verify(&data, &published.path, Some(&audit)).err().unwrap();
        assert!(error.contains("不一致"));
        assert_eq!(read_audit(&audit).unwrap().len(), 1);
        next.issue = cutoff + 2;
        draws.push(next);
        write_csv(&data, &draws).unwrap();
        assert!(verify(&data, &published.path, None).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_after_manifest_still_verifies_and_scores() {
        let mut draws = synth_draws(180, 21);
        let (dir, data, audit, out) = fixture("legacy", &draws);
        let published = publish(&data, &audit, &out, None, Some(now_unix() + 3600)).unwrap();
        let mut next = draws.last().unwrap().clone();
        next.issue += 1;
        draws.push(next);
        write_csv(&data, &draws).unwrap();
        let checked = verify(&data, &published.path, None).unwrap();
        assert_eq!(checked.score.unwrap().issue, draws.last().unwrap().issue);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn target_validation_handles_invalid_targets_and_year_boundaries() {
        for target in ["", "next", "123456", "after:26114", "26115", "25111"] {
            assert!(target_draw(&[], 26115, target).is_err(), "{target}");
        }
        assert!(target_draw(&[], 26115, "26116").unwrap().is_none());
        let mut next = synth_draws(1, 21).remove(0);
        next.issue = 27001;
        assert_eq!(target_draw(&[next], 26153, "27001").unwrap().unwrap().issue, 27001);
    }
}
