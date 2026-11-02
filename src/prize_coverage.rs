//! Same-budget small-prize coverage, evaluated independently of prediction quality.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    model::roll_all,
    plan,
    protocol::Protocol,
    recommendations as rec,
    select::{fuse6, fuse7, rank_desc, softmax_shrink},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
pub const PRESET: &str = "bluefocus8";
pub const TITLE: &str = "小奖分散覆盖 · 8蓝8注";
pub fn policy() -> Value {
    json!({"version":1,"preset":PRESET,"objective":"any_regular_prize_period","ranking":"weighted","red_count":16,"blue_count":8,"bets":8,"cost":16,"blue_assignment":"eight_distinct_ranked_blues_one_per_ticket","uniform_blue_covered_rate":0.5,"uniform_expected_blue_hit_bets":0.5,"automatic_model_promotion":false,"note":"同8注预算分散蓝球覆盖；8个不同蓝球在均匀开奖假设下覆盖概率为8/16，不代表模型预测优势。只中普通六等奖为5元，低于16元成本；不保证中奖、回本或收益。"})
}
fn built(draws: &[Draw]) -> Result<(plan::Plan, crate::eval::Prepared), String> {
    let p = prepare_forecast(draws)?;
    let spec = plan::resolve(Some(PRESET), None, None, None, None, None, None, None, None)?;
    let plan = plan::build(&p.forecast, &spec)?;
    Ok((plan, p))
}
pub(crate) fn validate_saved(saved: &Value, payload: &Value, draws: &[Draw]) -> Result<(), String> {
    if saved["prize_coverage"] != policy() {
        return Err("小奖覆盖策略版本或参数变化".into());
    }
    let cutoff = payload["data_cutoff"]
        .as_str()
        .ok_or("小奖覆盖缺少截止期")?;
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| fmt_issue(d.issue).as_str() <= cutoff)
        .cloned()
        .collect();
    let (plan, _) = built(&prefix)?;
    let tickets: Vec<_> = plan
        .tickets
        .iter()
        .map(|t| json!({"red":t.red,"blue":t.blue}))
        .collect();
    if saved["tickets"] != json!(tickets)
        || saved["title"] != TITLE
        || saved["parameters"]["blue_count"] != 8
        || saved["parameters"]["pool"] != 16
        || saved["parameters"]["tickets"] != 8
        || !saved["ranking"].is_null()
        || !saved["blue_ranking"].is_null()
    {
        return Err("小奖覆盖票面与开奖前原榜及固定预算不一致".into());
    }
    let published = payload["published_at_unix"]
        .as_u64()
        .ok_or("缺少保存时间")?;
    let deadline = payload["operational_cutoff_unix"]
        .as_u64()
        .ok_or("缺少封存截止")?;
    if published >= deadline
        || payload["known_outcome_at_generation"] != false
        || payload["issue"] != fmt_issue(prefix.last().ok_or("缺少输入历史")?.issue + 1)
    {
        return Err("小奖覆盖保存时间或目标数据边界无效".into());
    }
    Ok(())
}
fn candidates(root: &Path, issue: &str) -> Vec<(String, PathBuf, String)> {
    let mut out = Vec::new();
    let daily = root.join(format!("daily/{issue}.json"));
    if daily.exists() {
        out.push(("daily".into(), daily, format!("daily/{issue}.json")));
    }
    if let Ok(entries) = fs::read_dir(root.join(format!("interactive/{issue}"))) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(id) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push((
                        format!("interactive:{id}"),
                        p.clone(),
                        format!("interactive/{issue}/{id}.json"),
                    ));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
fn record(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    record_for(root,draws,issue,"prize_coverage")
}
pub(crate) fn record_for(root: &Path, draws: &[Draw], issue: &str, tag: &str) -> Result<Option<Value>, String> {
    if issue.len() != 5 || !issue.bytes().all(|c| c.is_ascii_digit()) {
        return Err("期号须为5位数字".into());
    }
    let mut found = Vec::new();
    for (id, path, raw) in candidates(root, issue) {
        let e = rec::read_checked(&path)?;
        let p = &e["payload"];
        let Some(strategies) = p["strategies"].as_object() else {
            continue;
        };
        if !strategies
            .values()
            .any(|s| s.get(tag).is_some())
        {
            continue;
        }
        rec::validate_archive(p, issue, draws)?;
        let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
        if rec::saved_after_draw(p["published_at_unix"].as_u64(), draw) {
            return Err("小奖覆盖保存晚于开奖".into());
        }
        for (name, saved) in strategies {
            if saved.get(tag).is_none() {
                continue;
            }
            found.push(json!({"issue":issue,"archive_id":id,"strategy":name,"archive_sha256":e["sha256"],"published_at_unix":p["published_at_unix"],"input_history_sha256":p["input_history_sha256"],"raw_file":raw,"saved":saved}));
        }
    }
    found.sort_by(|a, b| {
        a["published_at_unix"]
            .as_u64()
            .cmp(&b["published_at_unix"].as_u64())
            .then_with(|| {
                a["archive_sha256"]
                    .as_str()
                    .cmp(&b["archive_sha256"].as_str())
            })
    });
    Ok(found.into_iter().next())
}
pub(crate) fn view_record(mut v: Value, draws: &[Draw]) -> Result<Value, String> {
    let issue = v["issue"].as_str().unwrap().to_string();
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    v["plan"] = rec::review_strategy(&v["saved"], draw)?;
    v.as_object_mut().unwrap().remove("saved");
    v["status"] = json!("frozen");
    v["pending"] = json!(draw.is_none());
    v["url"] = json!(format!(
        "reviews?issue={issue}&archive={}&strategy={}",
        v["archive_id"].as_str().unwrap(),
        v["strategy"].as_str().unwrap()
    ));
    Ok(v)
}
pub fn freeze_current(
    root: &Path,
    draws: &[Draw],
    issue: &str,
    deadline: u64,
) -> Result<Value, String> {
    if let Some(v) = record(root, draws, issue)? {
        return Ok(
            json!({"status":"already_frozen","issue":issue,"archive_id":v["archive_id"],"archive_sha256":v["archive_sha256"]}),
        );
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    if issue != fmt_issue(last.issue + 1)
        || issue[..2] != fmt_issue(last.issue)[..2]
        || rec::now_unix() >= deadline
    {
        return Err("小奖覆盖仅在截止前封存当前紧邻下一期".into());
    }
    let (plan, p) = built(draws)?;
    let archive = rec::archive_plan(root, &plan, &p, issue, Some(deadline))?;
    Ok(
        json!({"status":"frozen","issue":issue,"archive_id":archive["archive_id"],"archive_sha256":archive["sha256"]}),
    )
}
pub(crate) fn issues(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for folder in ["daily", "interactive"] {
        if let Ok(entries) = fs::read_dir(root.join(folder)) {
            for entry in entries.flatten() {
                let p = entry.path();
                let i = if folder == "daily" {
                    p.file_stem()
                } else {
                    p.file_name()
                };
                if let Some(s) = i
                    .and_then(|s| s.to_str())
                    .filter(|s| s.len() == 5 && s.bytes().all(|c| c.is_ascii_digit()))
                {
                    out.insert(s.to_string());
                }
            }
        }
    }
    out
}
fn actual(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending, mut missing, mut prizes, mut fixed, mut total_bets) =
        (0, 0, 0, 0, 0u64, 0u64);
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    for issue in issues(root) {
        let result =
            record(root, draws, &issue).and_then(|v| v.map(|v| view_record(v, draws)).transpose());
        match result {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    let s = &v["plan"]["summary"];
                    prizes += usize::from(s["prize_tickets"].as_u64().unwrap_or(0) > 0);
                    fixed += s["fixed_prize"].as_u64().unwrap_or(0);
                    total_bets += s["prize_tickets"].as_u64().unwrap_or(0);
                }
                periods.push(json!({"issue":issue,"archive_id":v["archive_id"],"archive_sha256":v["archive_sha256"],"pending":v["pending"],"summary":v["plan"]["summary"],"url":v["url"]}));
            }
            Ok(None) => missing += 1,
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"n":n,"pending":pending,"excluded_missing":missing,"prize_periods":prizes,"prize_rate":(n>0).then(||prizes as f64/n as f64),"prize_bets":if n>0{json!(total_bets)}else{Value::Null},"fixed_prize_total":if n>0{json!(fixed)}else{Value::Null},"periods":periods,"errors":errors,"note":"每期仅使用首次真实保存的小奖覆盖方案；待开奖和旧期缺档不计失败，日档与个人来源同一期不重复计数。"})
}
#[derive(Clone, Debug, Default, PartialEq)]
struct Outcome {
    blue: bool,
    prize_bets: usize,
    fixed: u64,
    floating: usize,
    special: usize,
}
#[derive(Clone, Debug, PartialEq)]
struct Period {
    issue: u32,
    outcomes: [Outcome; 3],
}
fn outcomes(masks: &[u64], order: &[usize; 16], red: u64, blue: usize, issue: u32) -> [Outcome; 3] {
    [2, 4, 8].map(|k| {
        let mut out = Outcome::default();
        for (i, m) in masks.iter().enumerate() {
            let h = (m & red).count_ones() as usize;
            let b = order[i % k] == blue;
            let level = rec::prize(h, b);
            out.blue |= b;
            out.prize_bets += usize::from(level.is_some());
            out.fixed += rec::fixed_prize_amount(level);
            out.floating += usize::from(matches!(level, Some(1 | 2)));
            out.special += usize::from(issue >= 26014 && h == 3 && !b);
        }
        out
    })
}
fn replay(draws: &[Draw]) -> Result<Vec<Period>, String> {
    let snapshot = roll_all(draws)?;
    let (mut red_sums, mut blue_sums, mut n) = ([0.; 7], [0.; 6], 0usize);
    let mut out = Vec::new();
    for r in snapshot.rows {
        if n >= 30 {
            let reds = rank_desc(&fuse7(&r.expert_p, &softmax_shrink(&red_sums, n as f64)));
            let blues = rank_desc(&fuse6(&r.blue_p, &softmax_shrink(&blue_sums, n as f64)));
            let masks = crate::coverage::balanced_tickets(&reds[..16], 8)?;
            out.push(Period {
                issue: r.issue,
                outcomes: outcomes(&masks, &blues, r.red_mask, r.blue as usize, r.issue),
            });
        }
        if r.issue < Protocol::CONFIRM_START_ISSUE {
            for (e, sum) in red_sums.iter_mut().enumerate() {
                *sum += crate::select::rank_log_score(&r.expert_p[e], r.red_mask);
            }
            for (e, sum) in blue_sums.iter_mut().enumerate() {
                *sum += r.blue_p[e][r.blue as usize].clamp(1e-12, 1.).ln();
            }
            n += 1;
        }
    }
    Ok(out)
}
fn summarize(rows: &[Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells:Vec<_>=[2,4,8].iter().enumerate().map(|(j,k)|{let prizes=rows.iter().filter(|r|r.outcomes[j].prize_bets>0).count();json!({"blue_count":k,"bets":8,"cost":16,"n":rows.len(),"uniform_blue_covered_rate":*k as f64/16.,"blue_covered_periods":rows.iter().filter(|r|r.outcomes[j].blue).count(),"prize_periods":prizes,"prize_rate":prizes as f64/rows.len()as f64,"prize_bets":rows.iter().map(|r|r.outcomes[j].prize_bets).sum::<usize>(),"fixed_prize_total":rows.iter().map(|r|r.outcomes[j].fixed).sum::<u64>(),"floating_prize_bets":rows.iter().map(|r|r.outcomes[j].floating).sum::<usize>(),"special_prize_unverified_bets":rows.iter().map(|r|r.outcomes[j].special).sum::<usize>(),"fixed_at_least_cost_periods":rows.iter().filter(|r|r.outcomes[j].fixed>=16).count(),"paired_prize_gains":rows.iter().filter(|r|r.outcomes[j].prize_bets>0&&r.outcomes[0].prize_bets==0).count(),"paired_prize_losses":rows.iter().filter(|r|r.outcomes[j].prize_bets==0&&r.outcomes[0].prize_bets>0).count()})}).collect();
    json!({"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"n":rows.len(),"cells":cells})
}
fn audit(root: &Path, draws: &[Draw]) -> Value {
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    for issue in issues(root)
        .into_iter()
        .rev()
        .filter(|i| draws.iter().any(|d| fmt_issue(d.issue) == *i))
        .take(8)
    {
        let result = (|| -> Result<Value, String> {
            let path = root.join(format!("daily/{issue}.json"));
            if !path.exists() {
                return Ok(Value::Null);
            }
            let e = rec::read_checked(&path)?;
            rec::validate_archive(&e["payload"], &issue, draws)?;
            let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue).unwrap();
            if rec::saved_after_draw(e["payload"]["published_at_unix"].as_u64(), Some(draw)) {
                return Err("每日原档保存晚于开奖".into());
            }
            let mut groups = Vec::new();
            let mut add = |name: String, review: Value| {
                let s = &review["summary"];
                groups.push(json!({"name":name,"bets":s["bets"],"cost":s["cost"],"blues":s["unique_blues"],"blue_hit":s["blue_covered"],"best_red":s["max_ticket_red_hits"],"prize_bets":s["prize_tickets"],"fixed_prize":s["fixed_prize"],"special_prize_unverified":s["special_prize_candidates"]}));
            };
            for (name, saved) in e["payload"]["strategies"]
                .as_object()
                .ok_or("缺少每日票组")?
            {
                if name.starts_with("efficiency:uniform") {
                    continue;
                }
                add(name.clone(), rec::review_strategy(saved, Some(draw))?);
            }
            if let Some(v) = crate::reference::view(root, draws, &issue)? {
                add(
                    "参考原始5注".into(),
                    rec::review_strategy(
                        &json!({"tickets":v["reference_pools"]["reference"]["tickets"]}),
                        Some(draw),
                    )?,
                );
                if let Some(plans) = v["fusion_tickets"]["plans"].as_array() {
                    for p in plans {
                        add(
                            format!("融合{}红8注", p["red_count"]),
                            json!({"summary":p["summary"]}),
                        );
                    }
                }
                if let Some(p) = v["dantuo"]["plan"].as_object() {
                    add("参考原始胆拖／复式".into(), json!({"summary":p["summary"]}));
                }
            }
            let blue_union: BTreeSet<_> = groups
                .iter()
                .flat_map(|g| {
                    g["blues"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|n| n.as_u64())
                })
                .collect();
            let winning = groups
                .iter()
                .filter(|g| g["prize_bets"].as_u64().unwrap_or(0) > 0)
                .count();
            Ok(
                json!({"issue":issue,"draw":{"red":draw.red,"blue":draw.blue},"archive_sha256":e["sha256"],"groups":groups,"group_count":groups.len(),"winning_groups":winning,"blue_union":blue_union,"any_blue_covered":blue_union.contains(&(draw.blue as u64)),"max_single_red":groups.iter().filter_map(|g|g["best_red"].as_u64()).max(),"url":format!("reviews?issue={issue}&archive=daily")}),
            )
        })();
        match result {
            Ok(v) if !v.is_null() => periods.push(v),
            Ok(_) => {}
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"periods":periods,"errors":errors,"note":"核对每日原档及参考独立原档，均匀机选研究对照不计入推荐组数；未实际购买不代表已发生投入或收益。3红0蓝特别奖资格单列待核，不计已确认奖金。"})
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let rows = replay(draws)?;
    let start = rows
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue + 1);
    let current = if let Some(v) = record(root, draws, &issue)? {
        view_record(v, draws)?
    } else {
        let (plan, _) = built(draws)?;
        let saved = json!({"title":TITLE,"rationale":plan.detail,"structural":true,"tickets":plan.tickets.iter().map(|t|json!({"red":t.red,"blue":t.blue})).collect::<Vec<_>>()});
        json!({"status":"preview","issue":issue,"actual_recommendation":false,"plan":rec::review_strategy(&saved,None)?})
    };
    Ok(
        json!({"version":1,"kind":"same_budget_prize_coverage","data_sha256":data_sha256(draws),"cutoff_issue":fmt_issue(last.issue),"target_issue":issue,"policy":policy(),"current":current,"actual":actual(root,draws),"audit":audit(root,draws),"windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),"tracking":summarize(&rows[end..]),"recent30":summarize(&rows[rows.len().saturating_sub(30)..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},"source_sha256":sha256_hex(concat!(include_str!("prize_coverage.rs"),include_str!("plan.rs"),include_str!("model.rs"),include_str!("select.rs")).as_bytes()),"note":"每期先预测再计分，2／4／8蓝使用完全相同8组红球；只改变蓝球重复曝光。已被查看的历史属于探索回放，不是新方案已发布实绩。普通奖金按实际单注规则计算，浮动奖与特别奖不纳入固定奖金；不按最近一期挑选参数。"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(
                std::env::temp_dir().join(format!(
                    "ssq-prize-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
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
    #[test]
    fn blue_diversification_increases_covered_outcomes_without_creating_expected_prize_money() {
        let masks = vec![(1u64 << 6) - 1; 8];
        let order = std::array::from_fn(|i| i);
        let red = ((1u64 << 6) - 1) << 10;
        let mut covered = [0; 3];
        let mut prizes = [0; 3];
        let mut money = [0; 3];
        for blue in 0..16 {
            for (j, out) in outcomes(&masks, &order, red, blue, 26125)
                .iter()
                .enumerate()
            {
                covered[j] += usize::from(out.prize_bets > 0);
                prizes[j] += out.prize_bets;
                money[j] += out.fixed;
            }
        }
        assert_eq!(covered, [2, 4, 8]);
        assert_eq!(prizes, [8, 8, 8]);
        assert_eq!(money, [40, 40, 40]);
        let d = crate::draw::synth_draws(220, 9);
        let (plan, p) = built(&d).unwrap();
        let spec = plan::resolve(
            Some("bluefocus2"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let base = plan::build(&p.forecast, &spec).unwrap();
        assert_eq!(plan.cost, 16);
        assert_eq!(plan.reds.len(), 16);
        assert_eq!(plan.blues.len(), 8);
        assert_eq!(
            plan.tickets
                .iter()
                .map(|t| t.blue)
                .collect::<BTreeSet<_>>()
                .len(),
            8
        );
        assert_eq!(
            plan.tickets.iter().map(|t| t.red).collect::<Vec<_>>(),
            base.tickets.iter().map(|t| t.red).collect::<Vec<_>>()
        );
    }
    #[test]
    fn replay_is_strictly_causal_and_rates_match_existing_blue_ranking() {
        let d = crate::draw::synth_draws(250, 9);
        let a = replay(&d[..240]).unwrap();
        let b = replay(&d).unwrap();
        assert_eq!(a, b[..a.len()]);
        let v = summarize(&b);
        for (j, k) in [2, 4, 8].iter().enumerate() {
            let hits = b.iter().filter(|r| r.outcomes[j].blue).count();
            assert!(v["cells"][j]["prize_periods"].as_u64().unwrap() >= hits as u64);
            assert_eq!(v["cells"][j]["blue_covered_periods"], hits);
            assert_eq!(v["cells"][j]["uniform_blue_covered_rate"], *k as f64 / 16.);
        }
    }
    #[test]
    fn current_archive_is_idempotent_deduplicated_and_old_daily_is_not_rewritten() {
        let t = Temp::new();
        let d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        let deadline = rec::now_unix() + 3600;
        let v = freeze_current(&t.0, &d, &issue, deadline).unwrap();
        assert_eq!(v["status"], "frozen");
        let r = record(&t.0, &d, &issue).unwrap().unwrap();
        let raw = t.0.join(r["raw_file"].as_str().unwrap());
        let before = fs::read(&raw).unwrap();
        assert_eq!(
            freeze_current(&t.0, &d, &issue, 1).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(fs::read(&raw).unwrap(), before);
        let pending = actual(&t.0, &d);
        assert_eq!(pending["pending"], 1);
        assert_eq!(pending["n"], 0);
        assert!(pending["prize_rate"].is_null());
        assert!(!t.0.join("reviews").exists());
        rec::freeze_daily(&t.0, &d, &issue, deadline).unwrap();
        let daily = t.0.join(format!("daily/{issue}.json"));
        let original = fs::read(&daily).unwrap();
        assert_eq!(actual(&t.0, &d)["pending"], 1);
        freeze_current(&t.0, &d, &issue, deadline).unwrap();
        assert_eq!(fs::read(&daily).unwrap(), original);
        let mut changed = d.clone();
        changed[0].blue = changed[0].blue % 16 + 1;
        assert!(record(&t.0, &changed, &issue).is_err());
        let mut bad: Value = serde_json::from_slice(&before).unwrap();
        bad["payload"]["strategies"]["custom"]["tickets"][0]["blue"] = json!(16);
        bad["sha256"] = json!(sha256_hex(&serde_json::to_vec(&bad["payload"]).unwrap()));
        fs::write(&raw, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(record(&t.0, &d, &issue).is_err());
    }
    #[test]
    fn missing_history_is_not_filled_and_settlement_uses_original_singles() {
        let t = Temp::new();
        let mut d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        assert!(record(&t.0, &d, &issue).unwrap().is_none());
        assert!(freeze_current(&t.0, &d, &issue, 1).is_err());
        assert!(!t.0.exists());
        let frozen = freeze_current(&t.0, &d, &issue, rec::now_unix() + 3600).unwrap();
        let r = record(&t.0, &d, &issue).unwrap().unwrap();
        let ticket = &r["saved"]["tickets"][0];
        d.push(
            Draw::from_parts(
                issue.parse().unwrap(),
                20990101,
                serde_json::from_value(ticket["red"].clone()).unwrap(),
                ticket["blue"].as_u64().unwrap() as u8,
            )
            .unwrap(),
        );
        let result = view_record(record(&t.0, &d, &issue).unwrap().unwrap(), &d).unwrap();
        assert_eq!(result["plan"]["tickets"][0]["prize_level"], 1);
        assert_eq!(actual(&t.0, &d)["n"], 1);
        assert_eq!(actual(&t.0, &d)["prize_periods"], 1);
        assert!(!t.0.join("reviews").exists());
        assert_eq!(rec::settle_archives(&t.0, &d).settled, 1);
        assert_eq!(rec::settle_archives(&t.0, &d).settled, 0);
        assert_eq!(
            freeze_current(&t.0, &d, &issue, 1).unwrap()["archive_sha256"],
            frozen["archive_sha256"]
        );
    }
}
