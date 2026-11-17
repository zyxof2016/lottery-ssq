//! Validate first daily archives and distinguish union coverage from real bets.
use crate::{
    draw::{fmt_issue, Draw},
    recommendations as rec,
};
use serde_json::{json, Value};
use std::path::Path;
pub fn daily(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    if issue.len() != 5 || !issue.bytes().all(|b| b.is_ascii_digit()) {
        return Err("每日原档期号须为5位数字".into());
    }
    let file = root.join("daily").join(format!("{issue}.json"));
    if !file.is_file() {
        return Ok(None);
    }
    let e = rec::read_checked(&file)?;
    let p = &e["payload"];
    if p["kind"] != "daily"
        || p["known_outcome_at_generation"] != false
        || p["actual_recommendation"] == false
    {
        return Err("排序页仅接受开奖前的每日首次档案".into());
    }
    rec::validate_archive(p, issue, draws)?;
    let cutoff = p["data_cutoff"].as_str().ok_or("每日原档缺少截止期")?;
    let last = draws
        .iter()
        .find(|d| fmt_issue(d.issue) == cutoff)
        .ok_or("每日原档缺少输入开奖")?;
    let at = p["published_at_unix"]
        .as_u64()
        .ok_or("每日原档缺少保存时间")?;
    let deadline = p["operational_cutoff_unix"]
        .as_u64()
        .ok_or("每日原档缺少系统截止")?;
    let gate = crate::cycle::decide(at, last.issue, last.date, false);
    if !gate.publish
        || gate.target != issue
        || gate.deadline_unix != deadline
        || rec::saved_after_draw(Some(at), draws.iter().find(|d| fmt_issue(d.issue) == issue))
    {
        return Err("每日原档保存时间或系统截止与该期不一致".into());
    }
    Ok(Some(e))
}
pub fn source(e: &Value) -> Value {
    let p = &e["payload"];
    json!({"kind":"validated_daily_first_archive","issue":p["issue"],"archive_sha256":e["sha256"],"data_cutoff":p["data_cutoff"],"input_history_sha256":p["input_history_sha256"],"published_at_unix":p["published_at_unix"],"independent_timestamp_verified":false,"raw_file":format!("daily/{}.json",p["issue"].as_str().unwrap())})
}
pub fn strategy(saved: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let reviewed = rec::review_strategy(saved, draw)?;
    let summary = &reviewed["summary"];
    let mut losses = Vec::new();
    let (six, head, full_union) = if let Some(d) = draw {
        let tickets = reviewed["tickets"].as_array().ok_or("方案缺少逐注结果")?;
        let six = tickets.iter().any(|t| t["red_hits"] == 6);
        let head = tickets
            .iter()
            .any(|t| t["red_hits"] == 6 && t["blue_hit"] == true);
        let red_union = summary["covered_red_count"] == 6;
        let blue = summary["blue_covered"] == true;
        if !red_union {
            losses.push("red_pool_miss");
        }
        if !blue {
            losses.push("blue_selection_miss");
        }
        if red_union && !six {
            losses.push("complete_red_set_miss");
        }
        if six && blue && !head {
            losses.push("blue_pairing_miss");
        }
        let _ = d;
        (Some(six), Some(head), Some(red_union && blue))
    } else {
        (None, None, None)
    };
    Ok(
        json!({"record_kind":"real_single_tickets","bets":summary["bets"],"cost":summary["cost"],"max_ticket_red_hits":summary["max_ticket_red_hits"],"duplicate_tickets":summary["duplicate_tickets"],
        "single_six_red":six,"single_jackpot":head,"full_red_and_blue_union":full_union,"losses":draw.map(|_|losses),"scope":"按首次保存的每张6红1蓝核对；号码并集全中不记作单注头奖，各方案不是独立开奖样本。"}),
    )
}
pub fn funnel(items: &[Value], settled: bool) -> Value {
    let rows: Vec<_> = items
        .iter()
        .filter_map(|p| p.get("ticket_audit"))
        .filter(|p| !p.is_null())
        .collect();
    json!({"strategy_count":rows.len(),"red_union_full_six_strategies":settled.then(||items.iter().filter(|r|r["kind"]=="strategy"&&r["covers_all_red"]==true).count()),"single_six_red_strategies":settled.then(||rows.iter().filter(|r|r["single_six_red"]==true).count()),"single_jackpot_strategies":settled.then(||rows.iter().filter(|r|r["single_jackpot"]==true).count()),"full_union_without_single_jackpot":settled.then(||rows.iter().filter(|r|r["full_red_and_blue_union"]==true&&r["single_jackpot"]==false).count()),"note":"这里只数同一期已保存的方案选项，不把26个方案当作26期，不合并购买预算。"})
}

#[cfg(test)]
pub(crate) fn fixture(root: &Path) -> Vec<Draw> {
    let mut d = crate::draw::load_csv(Path::new("data/ssq.csv")).unwrap();
    let last = d.last_mut().unwrap();
    last.issue = 26130;
    last.date = 20261112;
    let now = 1794574800;
    let gate = crate::cycle::decide(now, 26130, 20261112, false);
    assert!(gate.publish);
    assert_eq!(gate.target, "26131");
    let prepared = crate::eval::prepare_forecast(&d).unwrap();
    let mut p = rec::base_payload(&prepared, "26131", "daily", now, Some(gate.deadline_unix));
    p["number_pools"] = json!({"pools":[{"model":"weighted","group":"observation","red_count":16,"blue_count":2,"reds":(1..=16).collect::<Vec<_>>(),"blues":[7,9]}]});
    p["strategies"] =
        json!({"test":{"title":"测试票组","tickets":[{"red":[1,2,3,4,5,6],"blue":7}]}});
    rec::write_new_before(&root.join("daily/26131.json"), p, None, || now).unwrap();
    d.push(Draw::from_parts(26131, 20261115, [1, 2, 3, 4, 5, 6], 7).unwrap());
    d
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_union_is_not_complete_ticket_and_blue_pairing_is_separate() {
        let d = Draw::from_parts(26131, 20261115, [1, 2, 3, 4, 5, 6], 7).unwrap();
        let v = json!({"tickets":[{"red":[1,2,3,4,5,8],"blue":7},{"red":[1,2,3,4,6,9],"blue":8}]});
        let r = strategy(&v, Some(&d)).unwrap();
        assert_eq!(r["full_red_and_blue_union"], true);
        assert_eq!(r["single_six_red"], false);
        assert_eq!(r["single_jackpot"], false);
        assert_eq!(r["losses"], json!(["complete_red_set_miss"]));
        let v = json!({"tickets":[{"red":[1,2,3,4,5,6],"blue":8},{"red":[1,2,3,4,6,9],"blue":7}]});
        let r = strategy(&v, Some(&d)).unwrap();
        assert_eq!(r["single_six_red"], true);
        assert_eq!(r["single_jackpot"], false);
        assert_eq!(r["losses"], json!(["blue_pairing_miss"]));
        let p = strategy(&v, None).unwrap();
        assert!(p["single_jackpot"].is_null());
        assert!(p["max_ticket_red_hits"].is_null());
        assert!(p["losses"].is_null());
    }
    #[test]
    fn rehashed_wrong_issue_and_post_draw_archives_are_rejected_without_writes() {
        let dir = std::env::temp_dir().join(format!("ssq-rank-audit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let draws = fixture(&dir);
        let path = dir.join("daily/26131.json");
        let bytes = std::fs::read(&path).unwrap();
        assert!(daily(&dir, &draws, "26131").unwrap().is_some());
        assert_eq!(bytes, std::fs::read(&path).unwrap());
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        for (field, value) in [
            ("issue", json!("26130")),
            ("known_outcome_at_generation", json!(true)),
            ("input_history_sha256", json!("0".repeat(64))),
            (
                "operational_cutoff_unix",
                json!(
                    original["payload"]["operational_cutoff_unix"]
                        .as_u64()
                        .unwrap()
                        + 86400
                ),
            ),
        ] {
            let mut v = original.clone();
            v["payload"][field] = value;
            v["sha256"] = json!(crate::hash::sha256_hex(
                &serde_json::to_vec(&v["payload"]).unwrap()
            ));
            std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
            assert!(daily(&dir, &draws, "26131").is_err(), "{field}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
