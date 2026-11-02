//! Experimental exact-ticket objective. Scores are ranking evidence, not winning probabilities.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    model::roll_all,
    plan::{self, Plan, PlanTicket, Spec},
    protocol::Protocol,
    recommendations as rec,
    select::{fuse6, fuse7, rank_desc, rank_log_score, score_key, softmax_shrink, Forecast},
};
use serde_json::{json, Value};
use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    path::Path,
};
pub const PRESET: &str = "jackpot8";
pub const TITLE: &str = "头奖目标 · 4组6红各配2蓝";
pub fn policy() -> Value {
    json!({"version":1,"objective":"exact_six_red_and_one_blue_same_ticket","preset":PRESET,"red_candidate_limit":16,"red_combinations":4,"blue_count":2,"bets":8,"cost":16,"selector":"exact_best_first_top4_product_of_quantized_ranking_scores","blue_assignment":"full_cartesian_product_each_red_set_with_both_ranked_blues","exposure_balance_required":false,"score_kind":"uncalibrated_ranking_evidence_not_joint_probability","ties":"sum_quantized_log_scores_descending_then_rank_indices_lexicographic","uniform_outcomes":crate::stats::binom(33,6)*16,"uniform_eight_distinct_tickets_probability":8.0/(crate::stats::binom(33,6)*16)as f64,"automatic_model_promotion":false,"note":"目标是单注6红1蓝。保留4个完整六红组合，每组配原榜2蓝；不以均衡曝光或小奖频率代替头奖。组合乘积分数未经概率校准，未证明提高头奖命中；公平开奖下，同8注不同票面的头奖基准概率相同。"})
}
pub fn spec() -> Spec {
    Spec{mode:"jackpot",title:TITLE.into(),detail:"最多16红候选，按固定组合分数保留4个完整六红组合，每组分别配原榜2蓝，共8注16元。允许红球高度重叠，不强求每个候选号入票。分数不是中奖概率，尚未证实头奖优势。".into(),tickets:8,pool:16,overlap:6,red_count:6,blue_count:2,bankers:0,drags:0}
}
fn top_sets(pool: &[usize], scores: &[f64; 33], count: usize) -> Result<Vec<u64>, String> {
    if !(6..=16).contains(&pool.len())
        || pool.iter().any(|n| *n >= 33)
        || pool.iter().collect::<BTreeSet<_>>().len() != pool.len()
        || count == 0
        || count > crate::stats::binom(pool.len() as u64, 6) as usize
    {
        return Err("完整六红组合候选或数量无效".into());
    }
    let weights: Vec<_> = pool.iter().map(|n| score_key(scores[*n]).max(1)).collect();
    if weights.windows(2).any(|w| w[0] < w[1]) {
        return Err("六红组合候选须按原评分递减排列".into());
    }
    let logs: Vec<i64> = weights
        .iter()
        .map(|w| ((*w as f64).ln() * 1e9).round() as i64)
        .collect();
    let key = |c: &[usize; 6]| c.iter().map(|i| logs[*i]).sum::<i64>();
    let first = [0, 1, 2, 3, 4, 5];
    let mut heap = BinaryHeap::from([(key(&first), Reverse(first))]);
    let mut seen = BTreeSet::from([first]);
    let mut out = Vec::new();
    while out.len() < count {
        let (_, Reverse(c)) = heap.pop().ok_or("缺少完整六红组合")?;
        out.push(c.iter().fold(0, |m, i| m | (1u64 << pool[*i])));
        for i in 0..6 {
            let mut next = c;
            next[i] += 1;
            if next[i] >= pool.len() || (i < 5 && next[i] >= next[i + 1]) {
                continue;
            }
            if seen.insert(next) {
                heap.push((key(&next), Reverse(next)));
            }
        }
    }
    Ok(out)
}
fn ticket_pairs(reds: &[u64], blues: &[usize]) -> Vec<PlanTicket> {
    reds.iter()
        .flat_map(|r| {
            blues.iter().map(move |b| PlanTicket {
                red: plan::mask_nums(*r),
                blue: (*b + 1) as u8,
            })
        })
        .collect()
}
pub fn build(fc: &Forecast, spec: &Spec) -> Result<Plan, String> {
    if spec.mode != "jackpot" || spec.pool != 16 || spec.tickets != 8 || spec.blue_count != 2 {
        return Err("头奖目标固定4组六红各配2蓝、8注16元".into());
    }
    let pool = &fc.red_order[..16];
    let sets = top_sets(pool, &fc.fusion, 4)?;
    let blues = &fc.blue_order[..2];
    let tickets = ticket_pairs(&sets, blues);
    Ok(plan::finish(
        spec,
        tickets,
        pool.iter().map(|n| (n + 1) as u8).collect(),
        vec![],
        vec![],
        blues.iter().map(|n| (n + 1) as u8).collect(),
        vec![],
    ))
}
fn built(draws: &[Draw]) -> Result<(Plan, crate::eval::Prepared), String> {
    let p = prepare_forecast(draws)?;
    let b = build(&p.forecast, &spec())?;
    Ok((b, p))
}
pub(crate) fn validate_saved(saved: &Value, payload: &Value, draws: &[Draw]) -> Result<(), String> {
    if saved["jackpot_goal"] != policy() {
        return Err("头奖目标原档策略版本或参数变化".into());
    }
    let cutoff = payload["data_cutoff"]
        .as_str()
        .ok_or("头奖原档缺少输入截止期")?;
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
        || saved["mode"] != "jackpot"
        || saved["parameters"]["pool"] != 16
        || saved["parameters"]["tickets"] != 8
        || saved["parameters"]["blue_count"] != 2
        || candidate(saved)? != plan.reds
        || !saved["ranking"].is_null()
        || !saved["blue_ranking"].is_null()
    {
        return Err("头奖票组与原榜、完整组合或蓝球交叉票面不一致".into());
    }
    let at = payload["published_at_unix"]
        .as_u64()
        .ok_or("缺少保存时间")?;
    let end = payload["operational_cutoff_unix"]
        .as_u64()
        .ok_or("缺少封存截止")?;
    let last = prefix.last().ok_or("缺少输入历史")?;
    if at >= end
        || payload["known_outcome_at_generation"] != false
        || payload["issue"] != fmt_issue(last.issue + 1)
    {
        return Err("头奖目标保存时间或期号边界无效".into());
    }
    Ok(())
}
fn candidate(saved: &Value) -> Result<Vec<u8>, String> {
    let stage = saved["funnel"]
        .as_array()
        .and_then(|a| a.iter().find(|v| v["stage"] == "candidate_pool"))
        .ok_or("头奖原档缺少16红候选")?;
    serde_json::from_value(stage["numbers"].clone()).map_err(|_| "头奖原档候选无效".into())
}
#[derive(Clone, Debug, Default, PartialEq)]
struct Measure {
    pool_full: bool,
    red_exact: bool,
    blue_in: bool,
    head: bool,
    pairing_miss: bool,
    best_red: usize,
    red5_blue: bool,
}
fn measure(pool: &[u8], tickets: &[PlanTicket], draw: &Draw) -> Measure {
    let red_exact = tickets
        .iter()
        .any(|t| t.red.iter().all(|n| draw.red.contains(n)));
    let blue_in = tickets.iter().any(|t| t.blue == draw.blue);
    let head = tickets
        .iter()
        .any(|t| t.blue == draw.blue && t.red.iter().all(|n| draw.red.contains(n)));
    Measure {
        pool_full: draw.red.iter().all(|n| pool.contains(n)),
        red_exact,
        blue_in,
        head,
        pairing_miss: red_exact && blue_in && !head,
        best_red: tickets
            .iter()
            .map(|t| t.red.iter().filter(|n| draw.red.contains(n)).count())
            .max()
            .unwrap_or(0),
        red5_blue: tickets.iter().any(|t| {
            t.blue == draw.blue && t.red.iter().filter(|n| draw.red.contains(n)).count() >= 5
        }),
    }
}
fn metric(m: &Measure) -> Value {
    json!({"candidate_full_six_red":m.pool_full,"single_exact_six_red":m.red_exact,"blue_in_selected_blues":m.blue_in,"single_exact_six_red_and_blue":m.head,"blue_pairing_miss":m.pairing_miss,"best_single_red":m.best_red,"five_or_more_red_and_blue":m.red5_blue,"funnel_result":if !m.pool_full{"red_pool_miss"}else if !m.red_exact{"red_combination_miss"}else if m.pairing_miss{"blue_pairing_miss"}else if !m.head{"blue_ranking_miss"}else{"jackpot"}})
}
fn record(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    crate::prize_coverage::record_for(root, draws, issue, "jackpot_goal")
}
pub fn view(root: &Path, draws: &[Draw], issue: &str) -> Result<Option<Value>, String> {
    let Some(v) = record(root, draws, issue)? else {
        return Ok(None);
    };
    let reds = candidate(&v["saved"])?;
    let tickets: Vec<PlanTicket> = v["saved"]["tickets"]
        .as_array()
        .ok_or("缺少头奖票面")?
        .iter()
        .map(|t| {
            Ok(PlanTicket {
                red: serde_json::from_value(t["red"].clone()).map_err(|_| "红球无效")?,
                blue: t["blue"].as_u64().ok_or("蓝球无效")? as u8,
            })
        })
        .collect::<Result<_, String>>()?;
    let draw = draws.iter().find(|d| fmt_issue(d.issue) == issue);
    let mut out = crate::prize_coverage::view_record(v, draws)?;
    out["candidate_reds"] = json!(reds);
    out["diagnostic"] = draw
        .map(|d| metric(&measure(&reds, &tickets, d)))
        .unwrap_or(Value::Null);
    out["red_combination_count"] = json!(4);
    Ok(Some(out))
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
        return Err("头奖目标仅在截止前封存当前紧邻下一期".into());
    }
    let (plan, p) = built(draws)?;
    let r = rec::archive_plan(root, &plan, &p, issue, Some(deadline))?;
    Ok(
        json!({"status":"frozen","issue":issue,"archive_id":r["archive_id"],"archive_sha256":r["sha256"]}),
    )
}
fn actual(root: &Path, draws: &[Draw]) -> Value {
    let (mut n, mut pending, mut missing, mut heads, mut six, mut pool) = (0, 0, 0, 0, 0, 0);
    let mut periods = Vec::new();
    let mut errors = Vec::new();
    for issue in crate::prize_coverage::issues(root) {
        match view(root, draws, &issue) {
            Ok(Some(v)) => {
                if v["pending"] == true {
                    pending += 1;
                } else {
                    n += 1;
                    heads += usize::from(v["diagnostic"]["single_exact_six_red_and_blue"] == true);
                    six += usize::from(v["diagnostic"]["single_exact_six_red"] == true);
                    pool += usize::from(v["diagnostic"]["candidate_full_six_red"] == true);
                }
                periods.push(json!({"issue":issue,"pending":v["pending"],"archive_sha256":v["archive_sha256"],"url":v["url"],"diagnostic":v["diagnostic"]}));
            }
            Ok(None) => missing += 1,
            Err(e) => errors.push(format!("{issue}：{e}")),
        }
    }
    json!({"n":n,"pending":pending,"excluded_missing":missing,"jackpot_periods":heads,"jackpot_rate":(n>0).then(||heads as f64/n as f64),"single_six_red_periods":six,"candidate_full_six_periods":pool,"periods":periods,"errors":errors,"note":"只统计本目标首次真实保存的完整票面；同一期多来源只计一次。头奖必须同一注6红1蓝，待开奖和旧期缺档不计失败。"})
}
#[derive(Clone, Debug, PartialEq)]
struct Period {
    issue: u32,
    m: [Measure; 3],
}
fn replay(draws: &[Draw]) -> Result<Vec<Period>, String> {
    let snap = roll_all(draws)?;
    let (mut rs, mut bs, mut n) = ([0.; 7], [0.; 6], 0usize);
    let mut out = Vec::new();
    for row in snap.rows {
        if n >= 30 {
            let scores = fuse7(&row.expert_p, &softmax_shrink(&rs, n as f64));
            let red_order = rank_desc(&scores);
            let blue_order = rank_desc(&fuse6(&row.blue_p, &softmax_shrink(&bs, n as f64)));
            let candidates: Vec<_> = red_order[..16].iter().map(|i| (i + 1) as u8).collect();
            let balanced = crate::coverage::balanced_tickets(&red_order[..16], 8)?;
            let old: Vec<_> = balanced
                .iter()
                .enumerate()
                .map(|(i, r)| PlanTicket {
                    red: plan::mask_nums(*r),
                    blue: (blue_order[i % 2] + 1) as u8,
                })
                .collect();
            let small: Vec<_> = balanced
                .iter()
                .enumerate()
                .map(|(i, r)| PlanTicket {
                    red: plan::mask_nums(*r),
                    blue: (blue_order[i % 8] + 1) as u8,
                })
                .collect();
            let head = ticket_pairs(&top_sets(&red_order[..16], &scores, 4)?, &blue_order[..2]);
            let draw = draws
                .iter()
                .find(|d| d.issue == row.issue)
                .ok_or("回放缺少开奖")?;
            out.push(Period {
                issue: row.issue,
                m: [
                    measure(&candidates, &old, draw),
                    measure(&candidates, &small, draw),
                    measure(&candidates, &head, draw),
                ],
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for (e, v) in rs.iter_mut().enumerate() {
                *v += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            for (e, v) in bs.iter_mut().enumerate() {
                *v += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
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
    let cells:Vec<_>=[("balanced16_2blue","原均衡16红2蓝8注"),("small_prize_8blue","小奖8蓝方案·对照"),(PRESET,"头奖目标4红组×2蓝")].iter().enumerate().map(|(j,(name,label))|{let count=|f:fn(&Measure)->bool|rows.iter().filter(|r|f(&r.m[j])).count();let heads=count(|m|m.head);json!({"model":name,"label":label,"bets":8,"cost":16,"n":rows.len(),"jackpot_periods":heads,"jackpot_rate":heads as f64/rows.len()as f64,"single_exact_six_red_periods":count(|m|m.red_exact),"candidate_full_six_red_periods":count(|m|m.pool_full),"blue_pairing_miss_periods":count(|m|m.pairing_miss),"five_or_more_red_and_blue_periods":count(|m|m.red5_blue),"red_pool_miss_periods":count(|m|!m.pool_full),"red_combination_miss_periods":count(|m|m.pool_full&&!m.red_exact),"blue_ranking_miss_periods":count(|m|m.red_exact&&!m.blue_in),"mean_best_single_red":rows.iter().map(|r|r.m[j].best_red).sum::<usize>()as f64/rows.len()as f64,"uniform_jackpot_probability":8.0/(crate::stats::binom(33,6)*16)as f64})}).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"cells":cells})
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue + 1);
    let current = if let Some(v) = view(root, draws, &issue)? {
        v
    } else {
        let (plan, _) = built(draws)?;
        let saved = json!({"title":TITLE,"mode":"jackpot","rationale":plan.detail,"structural":true,"tickets":plan.tickets.iter().map(|t|json!({"red":t.red,"blue":t.blue})).collect::<Vec<_>>()});
        json!({"status":"preview","issue":issue,"actual_recommendation":false,"candidate_reds":plan.reds,"red_combination_count":4,"diagnostic":Value::Null,"plan":rec::review_strategy(&saved,None)?})
    };
    let rows = replay(draws)?;
    let start = rows
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    Ok(
        json!({"version":1,"kind":"exact_jackpot_objective","data_sha256":data_sha256(draws),"cutoff_issue":fmt_issue(last.issue),"target_issue":issue,"policy":policy(),"current":current,"actual":actual(root,draws),"uniform":{"red_combinations":crate::stats::binom(33,6),"outcomes":crate::stats::binom(33,6)*16,"distinct_tickets":8,"one_in":(crate::stats::binom(33,6)*16)/8,"probability":8.0/(crate::stats::binom(33,6)*16)as f64,"six_red_sets_inside_16":crate::stats::binom(16,6)},"windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),"tracking":summarize(&rows[end..]),"recent30":summarize(&rows[rows.len().saturating_sub(30)..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},"adoption":{"objective_corrected":true,"prediction_advantage_established":false,"note":"此方案用于按单注6红1蓝目标前瞻观察；历史回放尚不能证明头奖优势，不以小奖、蓝球入池或红球并集覆盖代替头奖。"},"source_sha256":sha256_hex(concat!(include_str!("jackpot.rs"),include_str!("select.rs"),include_str!("model.rs"),include_str!("protocol.rs")).as_bytes()),"prize_rules_url":rec::PRIZE_RULES_URL,"note":"逐期只用开奖前历史；25111起沿用原冻结权重。已被查看的窗口为探索回放，与真实首次留档成绩分开，不按最近一期命中挑选参数。"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ssq-head-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                )))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn exact_best_first_matches_exhaustive_all_8008_combinations_including_ties() {
        for variant in 0..5 {
            let scores: [f64; 33] = std::array::from_fn(|i| match variant {
                0 => 1. / 33.,
                1 => (33 - i) as f64 / 561.,
                2 => ((i * 7) % 33 + 1) as f64 / 561.,
                3 => 1. + ((i * 11) % 4) as f64 * 1e-10,
                _ => ((i * 3) % 11 + 1) as f64 / 198.,
            });
            let order = rank_desc(&scores);
            let pool = &order[..16];
            let logs: Vec<i64> = pool
                .iter()
                .map(|n| ((score_key(scores[*n]).max(1) as f64).ln() * 1e9).round() as i64)
                .collect();
            let mut all: Vec<Vec<u8>> = (0..(1u64 << 16))
                .filter(|m| m.count_ones() == 6)
                .map(|m| (0..16u8).filter(|i| m & (1 << i) != 0).collect())
                .collect();
            assert_eq!(all.len(), 8008);
            all.sort_by(|a, b| {
                let sa = a.iter().map(|n| logs[*n as usize]).sum::<i64>();
                let sb = b.iter().map(|n| logs[*n as usize]).sum::<i64>();
                sb.cmp(&sa).then(a.cmp(b))
            });
            let expected: Vec<u64> = all[..4]
                .iter()
                .map(|c| c.iter().fold(0, |m, i| m | (1 << pool[*i as usize])))
                .collect();
            assert_eq!(top_sets(pool, &scores, 4).unwrap(), expected);
        }
    }
    #[test]
    fn eight_bets_are_four_complete_red_sets_crossed_with_two_blues() {
        let d = crate::draw::synth_draws(220, 9);
        let (b, p) = built(&d).unwrap();
        assert_eq!(b.bet_count, 8);
        assert_eq!(b.cost, 16);
        assert_eq!(b.reds.len(), 16);
        let mut sets = std::collections::BTreeMap::new();
        let mut pairs = BTreeSet::new();
        for t in &b.tickets {
            assert_eq!(t.red.iter().collect::<BTreeSet<_>>().len(), 6);
            assert!(t.red.iter().all(|n| b.reds.contains(n)));
            sets.entry(t.red)
                .or_insert_with(BTreeSet::new)
                .insert(t.blue);
            pairs.insert((t.red, t.blue));
        }
        assert_eq!(sets.len(), 4);
        assert_eq!(pairs.len(), 8);
        let expected: BTreeSet<u8> = p.forecast.blue_order[..2]
            .iter()
            .map(|n| (n + 1) as u8)
            .collect();
        assert!(sets.values().all(|v| *v == expected));
        assert_eq!(policy()["uniform_outcomes"], 17721088);
        assert_eq!(
            policy()["uniform_eight_distinct_tickets_probability"],
            8. / 17721088.
        );
    }
    #[test]
    fn union_hits_and_blue_union_do_not_count_as_a_jackpot() {
        let draw = Draw::from_parts(26125, 20261101, [1, 2, 3, 4, 5, 6], 9).unwrap();
        let pool = (1..=16).collect::<Vec<_>>();
        let split = vec![
            PlanTicket {
                red: [1, 2, 3, 7, 8, 9],
                blue: 9,
            },
            PlanTicket {
                red: [4, 5, 6, 10, 11, 12],
                blue: 9,
            },
        ];
        let m = measure(&pool, &split, &draw);
        assert!(m.pool_full && m.blue_in);
        assert!(!m.red_exact && !m.head);
        assert_eq!(metric(&m)["funnel_result"], "red_combination_miss");
        let wrong = vec![
            PlanTicket {
                red: draw.red,
                blue: 1,
            },
            PlanTicket {
                red: [7, 8, 9, 10, 11, 12],
                blue: 9,
            },
        ];
        let m = measure(&pool, &wrong, &draw);
        assert!(m.red_exact && m.blue_in && m.pairing_miss && !m.head);
        let mask = draw.red.iter().fold(0, |a, n| a | (1 << (*n - 1)));
        let crossed = ticket_pairs(&[mask], &[0, 8]);
        assert!(measure(&pool, &crossed, &draw).head);
        assert!(!measure(&pool, &crossed, &draw).pairing_miss);
    }
    #[test]
    fn replay_is_causal_and_miss_funnel_is_an_exact_partition() {
        let d = crate::draw::synth_draws(250, 9);
        let a = replay(&d[..240]).unwrap();
        let b = replay(&d).unwrap();
        assert_eq!(a, b[..a.len()]);
        let v = summarize(&b);
        for c in v["cells"].as_array().unwrap() {
            let sum = [
                "red_pool_miss_periods",
                "red_combination_miss_periods",
                "blue_ranking_miss_periods",
                "blue_pairing_miss_periods",
                "jackpot_periods",
            ]
            .iter()
            .map(|k| c[k].as_u64().unwrap())
            .sum::<u64>();
            assert_eq!(sum, b.len() as u64);
            assert_eq!(c["cost"], 16);
            assert_eq!(c["uniform_jackpot_probability"], 8. / 17721088.);
        }
        assert_eq!(v["cells"][2]["blue_pairing_miss_periods"], 0);
    }
    #[test]
    fn prospective_archives_are_first_only_and_never_backfill_a_daily_file() {
        let t = Temp::new();
        let d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        assert!(freeze_current(&t.0, &d, &issue, 1).is_err());
        assert!(!t.0.exists());
        let end = rec::now_unix() + 3600;
        let saved = freeze_current(&t.0, &d, &issue, end).unwrap();
        let r = record(&t.0, &d, &issue).unwrap().unwrap();
        let path = t.0.join(r["raw_file"].as_str().unwrap());
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            freeze_current(&t.0, &d, &issue, 1).unwrap()["status"],
            "already_frozen"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let v = view(&t.0, &d, &issue).unwrap().unwrap();
        assert!(v["diagnostic"].is_null());
        assert!(v["plan"]["summary"]["prize_tickets"].is_null());
        assert_eq!(actual(&t.0, &d)["pending"], 1);
        assert!(actual(&t.0, &d)["jackpot_rate"].is_null());
        assert!(!t.0.join("daily").exists());
        assert!(!t.0.join("reviews").exists());
        rec::freeze_daily(&t.0, &d, &issue, end).unwrap();
        let daily = t.0.join(format!("daily/{issue}.json"));
        let old = std::fs::read(&daily).unwrap();
        let daily_json: Value = serde_json::from_slice(&old).unwrap();
        assert!(daily_json["payload"]["strategies"].get(PRESET).is_some());
        assert!(daily_json["payload"]["strategies"]
            .get("bluefocus8")
            .is_none());
        assert_eq!(actual(&t.0, &d)["pending"], 1);
        freeze_current(&t.0, &d, &issue, end).unwrap();
        assert_eq!(std::fs::read(daily).unwrap(), old);
        let mut bad: Value = serde_json::from_slice(&bytes).unwrap();
        let b = bad["payload"]["strategies"]["custom"]["tickets"][0]["blue"]
            .as_u64()
            .unwrap();
        bad["payload"]["strategies"]["custom"]["tickets"][0]["blue"] = json!(b % 16 + 1);
        bad["sha256"] = json!(sha256_hex(&serde_json::to_vec(&bad["payload"]).unwrap()));
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(view(&t.0, &d, &issue).is_err());
        assert!(saved["archive_id"]
            .as_str()
            .unwrap()
            .starts_with("interactive:"));
    }
    #[test]
    fn scoring_a_real_complete_ticket_requires_the_matching_blue_and_preserves_review() {
        let t = Temp::new();
        let mut d = crate::draw::synth_draws(220, 9);
        let issue = fmt_issue(d.last().unwrap().issue + 1);
        freeze_current(&t.0, &d, &issue, rec::now_unix() + 3600).unwrap();
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
        let v = view(&t.0, &d, &issue).unwrap().unwrap();
        assert_eq!(v["diagnostic"]["single_exact_six_red_and_blue"], true);
        assert_eq!(v["plan"]["tickets"][0]["prize_level"], 1);
        assert_eq!(actual(&t.0, &d)["jackpot_periods"], 1);
        assert_eq!(rec::settle_archives(&t.0, &d).settled, 1);
        let id = r["archive_id"]
            .as_str()
            .unwrap()
            .strip_prefix("interactive:")
            .unwrap();
        let path = t.0.join(format!("reviews/{issue}/{id}.json"));
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(rec::settle_archives(&t.0, &d).settled, 0);
        d.last_mut().unwrap().blue = 16;
        assert!(!rec::settle_archives(&t.0, &d).errors.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), raw);
    }
}
