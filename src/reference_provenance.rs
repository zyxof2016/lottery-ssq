//! Read-only provenance for the reference engine's SAME-run five tickets and
//! budget plan. Does not rerun the engine, change parameters or write archives.
use crate::{
    draw::{canonical_csv, data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    recommendations as rec,
};
use serde_json::{json, Value};
use std::path::Path;

fn issue_ok(i: &str) -> bool {
    i.len() == 5 && i.bytes().all(|b| b.is_ascii_digit())
}
fn ticket_numbers(value: &Value) -> Result<Value, String> {
    let rows = value.as_array().ok_or("原报告缺少5注")?;
    if rows.len() != 5 {
        return Err("原报告须包含5注".into());
    }
    Ok(json!(rows
        .iter()
        .map(|t| json!({"red":t["red"],"blue":t["blue"]}))
        .collect::<Vec<_>>()))
}
fn source_input(draws: &[Draw]) -> String {
    let csv = canonical_csv(draws);
    let mut out = String::from("period,date,r1,r2,r3,r4,r5,r6,blue\n");
    for line in csv.lines().skip(1) {
        out.push_str("20");
        out.push_str(line);
        out.push('\n');
    }
    out
}
struct Evidence {
    parent: Value,
    cache: Value,
    source: Option<(String, Value)>,
    prefix: Vec<Draw>,
    plan: Value,
}
fn evidence(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Evidence>, String> {
    if !issue_ok(issue) {
        return Err("参考来源期号须为5位数字".into());
    }
    let Some(parent) = crate::reference::original(root, draws, issue)? else {
        return Ok(None);
    };
    let cache: Value = serde_json::from_str(
        parent["payload"]["reference_cache_json"]
            .as_str()
            .ok_or("缺少原始5注缓存")?,
    )
    .map_err(|_| "原始5注JSON无效")?;
    let cutoff = parent["payload"]["data_cutoff"]
        .as_str()
        .ok_or("参考原档缺少数据截止")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    if data_sha256(&prefix) != parent["payload"]["input_history_sha256"] {
        return Err("参考来源的历史输入摘要不一致".into());
    }
    let budget = crate::reference_dantuo::view(root, draws, issue)?;
    let (source, plan) = if let Some(budget) = budget {
        let e = rec::read_checked(&root.join(format!("reference-dantuo/{issue}.json")))?;
        let c: Value = serde_json::from_str(
            e["payload"]["dantuo_cache_json"]
                .as_str()
                .ok_or("缺少原始胆拖缓存")?,
        )
        .map_err(|_| "原始胆拖缓存JSON无效")?;
        let text = c["source_forecast_json"]
            .as_str()
            .ok_or("缺少同次原预测记录")?
            .to_string();
        if sha256_hex(text.as_bytes()) != c["source_forecast_sha256"] {
            return Err("同次原预测记录摘要不一致".into());
        }
        let record: Value = serde_json::from_str(&text).map_err(|_| "同次原预测记录JSON无效")?;
        if record["source_period"] != format!("20{cutoff}")
            || record["target_period"] != format!("20{issue}")
            || ticket_numbers(&record["tickets"])? != cache["tickets"]
            || record["metadata"]["dantuo_plan"] != budget["plan"]["original_plan"]
        {
            return Err("5注及胆拖不是同次原始报告".into());
        }
        if record["data_version"]["raw_sha256"]
            .as_str()
            .is_some_and(|sha| sha != sha256_hex(source_input(&prefix).as_bytes()))
        {
            return Err("导出的源码输入与原报告输入摘要不一致".into());
        }
        (
            Some((text, record)),
            budget["plan"]["original_plan"].clone(),
        )
    } else {
        (None, Value::Null)
    };
    Ok(Some(Evidence {
        parent,
        cache,
        source,
        prefix,
        plan,
    }))
}
pub fn report(root: &Path, draws: &[Draw], requested: Option<&str>) -> Result<Value, String> {
    let next = fmt_issue(draws.last().ok_or("没有开奖数据")?.issue + 1);
    let issue = requested.unwrap_or(&next);
    let Some(e) = evidence(root, draws, issue)? else {
        return Ok(
            json!({"issue":issue,"status":"waiting","reason":"本期参考原始输出尚未封存，不能核验同次报告。"}),
        );
    };
    let meta = &e.cache["metadata"];
    let source = e.source.as_ref().map(|(_, s)| s);
    let original = meta
        .get("original_tickets")
        .map(ticket_numbers)
        .transpose()?;
    let five_matches_export = original.map(|v| v == e.cache["tickets"]);
    if five_matches_export == Some(false) {
        return Err("参考导出5注与引擎保存的原始5注不同".into());
    }
    let dantuo = if e.plan.is_null() {
        Value::Null
    } else {
        json!({"red_dan":e.plan["red_dan"],"red_tuo":e.plan["red_tuo"],"blue":e.plan["blue"],"bet_count":e.plan["bet_count"],"amount":e.plan["amount"]})
    };
    Ok(
        json!({"version":1,"issue":issue,"status":if source.is_some(){"verified"}else{"partial"},"data_cutoff":e.parent["payload"]["data_cutoff"],"history_count":e.prefix.len(),
        "profile":{"runtime_profile":meta.get("runtime_profile").or_else(||meta.get("profile")),"implementation":meta["implementation"],"monte_carlo_samples":meta["monte_carlo_samples"],"optimizer_methods":meta["optimizer_methods"],"backtest_periods":source.and_then(|s|s["metadata"]["forecast_publish_gate"]["effective_backtest_periods"].as_u64()).map(|n|json!(n)).unwrap_or_else(||meta["backtest_periods"].clone()),"declared_backtest_periods":meta["backtest_periods"],"random_seed":meta["random_seed"],"packages":meta["runtime_packages"],"python":meta["run_manifest"]["python"],"platform":meta["run_manifest"]["platform"],"prediction_mode":meta["prediction_mode"],"forecast_scope":meta["reference_scope"]},
        "verification":{"five_matches_engine_export":five_matches_export,"five_matches_same_run_report":source.map(|_|true),"dantuo_matches_same_run_report":source.map(|_|true)},
        "reference_archive_sha256":e.parent["sha256"],"canonical_history_sha256":e.parent["payload"]["input_history_sha256"],"source_input_sha256":sha256_hex(source_input(&e.prefix).as_bytes()),
        "clean_frame_sha256":source.map(|s|s["data_version"]["clean_frame_sha256"].clone()),"engine_sha256":e.cache["engine_sha256"],"config_sha256":e.cache["config_sha256"],
        "tickets":e.cache["tickets"],"dantuo":dantuo,
        "source_forecast_url":source.map(|_|format!("api/reference-pools/provenance/forecast/{issue}")),"source_input_url":format!("api/reference-pools/provenance/input/{issue}"),
        "date_policy":meta["date_validation_adapter"],"note":"核验的是本站参考引擎同次运行，不代表与不同期号、历史、回测期数或依赖版本的另一次运行一致。来源下载只读，不触发重算或覆盖原档。"}),
    )
}
pub fn forecast(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    let e = evidence(root, draws, issue)?.ok_or("参考原档不存在")?;
    e.source
        .map(|(text, _)| text.into_bytes())
        .ok_or_else(|| "尚未保存同次原始预测记录".into())
}
pub fn input(root: &Path, draws: &[Draw], issue: &str) -> Result<Vec<u8>, String> {
    let e = evidence(root, draws, issue)?.ok_or("参考原档不存在")?;
    Ok(source_input(&e.prefix).into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_period_conversion_preserves_dates_and_numbers() {
        let d = vec![Draw::from_parts(3001, 20030223, [1, 6, 7, 13, 25, 33], 12).unwrap()];
        assert_eq!(
            source_input(&d),
            "period,date,r1,r2,r3,r4,r5,r6,blue\n2003001,2003-02-23,01,06,07,13,25,33,12\n"
        );
    }
    #[test]
    fn ticket_normalization_only_removes_source_annotations() {
        let t = json!((0..5)
            .map(|i| json!({"red":[1,2,3,4,5,6],"blue":i+1,"consensus_score":0.7}))
            .collect::<Vec<_>>());
        let out = ticket_numbers(&t).unwrap();
        assert_eq!(out[0], json!({"red":[1,2,3,4,5,6],"blue":1}));
        assert!(ticket_numbers(&json!([])).is_err());
    }
    #[test]
    fn missing_or_invalid_source_never_creates_an_archive() {
        let root =
            std::env::temp_dir().join(format!("ssq-provenance-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let d = crate::draw::synth_draws(220, 1);
        assert_eq!(report(&root, &d, None).unwrap()["status"], "waiting");
        assert!(!root.exists());
        assert!(report(&root, &d, Some("../26128")).is_err());
    }
}
