//! Versioned coverage experiment. Frozen publication rules are unchanged.
use crate::{
    draw::{fmt_issue, Draw},
    eval::Prepared,
    pool,
    select::{rank_desc, score_key},
};
use serde_json::{json, Value};

pub const VERSION: u32 = 1;
pub const MODEL: &str = "balanced_rank";

pub fn ticket_policy(pool: usize, tickets: usize, blue: usize) -> Value {
    json!({"version":VERSION,"ranking":"original_weighted_fusion","objective":"complete_selected_pool_balanced_exposure",
        "pool":pool,"tickets":tickets,"blue_count":blue,"minimum_red_exposure":tickets*6/pool,"maximum_red_exposure":(tickets*6).div_ceil(pool),
        "note":"完整使用所选原融合排名号码池并均衡单号曝光；只改组票，不更换融合权重。"})
}

pub fn policy() -> Value {
    json!({"version": VERSION, "model": MODEL, "label": "均衡排名试验",
        "rank_method": "seven_expert_midranks_trim_one_each_end",
        "repeat_cap": "ceil(6 * prefix_size / 33)", "blue_scope": "shared_weighted_blue_ranking",
        "ticket_objective": "full_pool_union_balanced_exposure_unique_red_combinations",
        "automatic_promotion": false,
        "note": "七专家先转并列中位名次，每号去掉最高和最低名次后平均；每个前缀限制上期号码集中度。固定规则试验，未证实优于原模型。"})
}

/// Midranks keep equal expert scores equal, without low-number tie bias.
pub fn scores(experts: &[[f64; 33]; 7]) -> [f64; 33] {
    let mut ranks = [[0.0; 33]; 7];
    for (e, p) in experts.iter().enumerate() {
        let order = rank_desc(p);
        let mut start = 0;
        while start < 33 {
            let mut end = start + 1;
            while end < 33 && score_key(p[order[start]]) == score_key(p[order[end]]) {
                end += 1;
            }
            let mid = (start + 1 + end) as f64 / 2.0;
            for i in start..end {
                ranks[e][order[i]] = mid;
            }
            start = end;
        }
    }
    std::array::from_fn(|i| {
        let mut r = std::array::from_fn::<_, 7, _>(|e| ranks[e][i]);
        r.sort_by(f64::total_cmp);
        (34.0 - r[1..6].iter().sum::<f64>() / 5.0) / 33.0
    })
}

pub fn order(experts: &[[f64; 33]; 7], previous: u64) -> [usize; 33] {
    let base = rank_desc(&scores(experts));
    let mut used = [false; 33];
    let mut repeats = 0;
    std::array::from_fn(|position| {
        let cap = (6 * (position + 1)).div_ceil(33);
        let number = *base
            .iter()
            .find(|&&n| !used[n] && (previous & (1 << n) == 0 || repeats < cap))
            .expect("six previous reds always fit the prefix quota");
        used[number] = true;
        repeats += usize::from(previous & (1 << number) != 0);
        number
    })
}

pub fn saved(prepared: &Prepared, previous: u64) -> Value {
    let order = order(&prepared.expert_red, previous);
    let pools: Vec<_> = pool::TIER_RED_SIZES.into_iter().enumerate().flat_map(|(i,r)| {
        let order = &order;
        pool::BLUE_SIZES.into_iter().map(move |b| json!({"id": format!("balanced_rank_red{r}_blue{b}"),
            "model": MODEL, "group": "observation", "tier_id": ((b'A'+i as u8) as char).to_string(),
            "red_count": r, "blue_count": b, "reds": order[..r].iter().map(|n| *n+1).collect::<Vec<_>>(),
            "blues": prepared.forecast.blue_order[..b].iter().map(|n| *n+1).collect::<Vec<_>>() }))
    }).collect();
    json!({"version": VERSION, "model": MODEL, "policy": policy(),
        "target_issue": fmt_issue(prepared.cutoff_issue+1), "previous_reds": (0..33).filter(|n| previous & (1 << n) != 0).map(|n| n+1).collect::<Vec<_>>(),
        "red_ranking": order.iter().map(|n| *n+1).collect::<Vec<_>>(),
        "rank_scores": scores(&prepared.expert_red).to_vec(), "score_kind": "trimmed_midrank_evidence_not_probability",
        "blue_ranking": prepared.forecast.blue_order.iter().map(|n| *n+1).collect::<Vec<_>>(), "pools": pools})
}

/// Enumerate legal six-red combinations, minimize exposure concentration, and
/// preserve feasibility of final floor/ceiling exposure bounds at every step.
pub fn balanced_tickets(pool: &[usize], count: usize) -> Result<Vec<u64>, String> {
    if !(12..=24).contains(&pool.len())
        || !(2..=20).contains(&count)
        || count * 6 < pool.len()
        || pool.iter().any(|&n| n >= 33)
        || pool.iter().collect::<std::collections::BTreeSet<_>>().len() != pool.len()
    {
        return Err("均衡覆盖须选12–24个不同红球、2–20注，且总红球位置不少于池大小".into());
    }
    type Templates = std::collections::BTreeMap<(usize, usize), Vec<u64>>;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Templates>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let cached = cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(pool.len(), count))
        .cloned();
    let template = if let Some(value) = cached {
        value
    } else {
        let value = balanced_template(pool.len(), count)?;
        cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((pool.len(), count), value.clone());
        value
    };
    Ok(template
        .into_iter()
        .map(|m| {
            pool.iter()
                .enumerate()
                .filter(|(i, _)| m & (1 << i) != 0)
                .fold(0, |a, (_, n)| a | (1 << n))
        })
        .collect())
}

fn balanced_template(size: usize, count: usize) -> Result<Vec<u64>, String> {
    let indices: Vec<_> = (0..size).collect();
    let pool = indices.as_slice();
    let mut combinations = Vec::new();
    fn enumerate(pool: &[usize], start: usize, left: usize, mask: u64, out: &mut Vec<u64>) {
        if left == 0 {
            out.push(mask);
            return;
        }
        for i in start..=pool.len() - left {
            enumerate(pool, i + 1, left - 1, mask | (1 << pool[i]), out);
        }
    }
    enumerate(pool, 0, 6, 0, &mut combinations);
    let low = count * 6 / pool.len();
    let high = (count * 6).div_ceil(pool.len());
    let mut exposure = [0usize; 33];
    let mut tickets = Vec::new();
    for t in 0..count {
        let remaining = count - t - 1;
        let chosen = combinations
            .iter()
            .copied()
            .filter(|mask| !tickets.contains(mask))
            .filter(|mask| {
                pool.iter().all(|&n| {
                    let e = exposure[n] + usize::from(mask & (1 << n) != 0);
                    e <= high && e + remaining >= low
                })
            })
            .min_by_key(|mask| {
                let concentration: usize = pool
                    .iter()
                    .map(|&n| {
                        let e = exposure[n] + usize::from(mask & (1 << n) != 0);
                        e * e
                    })
                    .sum();
                let overlap: u32 = tickets
                    .iter()
                    .map(|m: &u64| (m & mask).count_ones().pow(2))
                    .sum();
                let rank_sum: usize = pool
                    .iter()
                    .enumerate()
                    .filter(|(_, n)| mask & (1 << **n) != 0)
                    .map(|(i, _)| i)
                    .sum();
                (concentration, overlap, rank_sum, *mask)
            })
            .ok_or("当前参数未找到满足均衡曝光的完整票组，请减少注数或调整池大小")?;
        for &n in pool {
            exposure[n] += usize::from(chosen & (1 << n) != 0);
        }
        tickets.push(chosen);
    }
    Ok(tickets)
}

pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    use crate::{
        model::roll_all,
        protocol::Protocol,
        select::{build_tickets, equal7, forecast, fuse6, fuse7, rank_log_score, softmax_shrink},
    };
    let snapshot = roll_all(draws)?;
    let previous: std::collections::BTreeMap<_, _> = draws
        .windows(2)
        .map(|w| (w[1].issue, w[0].red_mask))
        .collect();
    let (mut red_sum, mut blue_sum, mut nfit) = ([0.0; 7], [0.0; 6], 0usize);
    let mut rows = Vec::new();
    fn ticket_measure(masks: &[u64], actual: u64) -> Value {
        let union = masks.iter().fold(0, |a, b| a | b);
        json!({"bets":masks.len(),"cost":masks.len()*2,"red_union":union.count_ones(),
            "red_hits":(union&actual).count_ones(),"best_red":masks.iter().map(|m|(m&actual).count_ones()).max(),
            "mean_ticket_red_hits":masks.iter().map(|m|(m&actual).count_ones() as f64).sum::<f64>()/masks.len() as f64,
            "max_exposure":(0..33).map(|n|masks.iter().filter(|m|*m&(1<<n)!=0).count()).max()})
    }
    for row in &snapshot.rows {
        if nfit >= 30 {
            let weights = softmax_shrink(&red_sum, nfit as f64);
            let bw = softmax_shrink(&blue_sum, nfit as f64);
            let weighted_score = fuse7(&row.expert_p, &weights);
            let weighted = rank_desc(&weighted_score);
            let calibrated = rank_desc(&crate::ranking::scores(&weighted_score, previous[&row.issue]));
            let equal = rank_desc(&fuse7(&row.expert_p, &equal7()));
            let balanced = order(&row.expert_p, previous[&row.issue]);
            let blue = rank_desc(&fuse6(&row.blue_p, &bw));
            let equal_blue = rank_desc(&fuse6(&row.blue_p, &[1.0 / 6.0; 6]));
            let models:Vec<_>=[("weighted",&weighted),("equal",&equal),(MODEL,&balanced),(crate::ranking::MODEL,&calibrated)].into_iter().map(|(name,order)| {
                let ranks:Vec<_>=order.iter().enumerate().filter(|(_,n)|row.red_mask&(1<<**n)!=0).map(|(i,_)|i+1).collect();
                json!({"model":name,"minimum_full_red_prefix":ranks.last(),"winning_red_ranks":ranks,
                    "hits":pool::TIER_RED_SIZES.map(|r|order[..r].iter().filter(|&&n|row.red_mask&(1<<n)!=0).count()),
                    "previous_selected":pool::TIER_RED_SIZES.map(|r|order[..r].iter().filter(|&&n|previous[&row.issue]&(1<<n)!=0).count())})
            }).collect();
            let tickets = if row.issue >= Protocol::CONFIRM_START_ISSUE {
                let fc = forecast(&row.expert_p, &row.blue_p, &weights, &bw);
                json!({"frozen":ticket_measure(&fc.tickets,row.red_mask),
                    "balanced_same12":ticket_measure(&balanced_tickets(&fc.pool,8)?,row.red_mask),
                    "weighted14_greedy":ticket_measure(&build_tickets(&weighted[..14],&fc.fusion,8,4),row.red_mask),
                    "weighted14_balanced":ticket_measure(&balanced_tickets(&weighted[..14],8)?,row.red_mask),
                    "balanced14_rank":ticket_measure(&balanced_tickets(&balanced[..14],8)?,row.red_mask)})
            } else {
                Value::Null
            };
            rows.push(
                json!({"issue":fmt_issue(row.issue),"models":models,"tickets":tickets,
                "weighted_blue_rank":blue.iter().position(|n|*n==row.blue as usize).unwrap()+1,
                "equal_blue_rank":equal_blue.iter().position(|n|*n==row.blue as usize).unwrap()+1}),
            );
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            for e in 0..6 {
                blue_sum[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.0).ln();
            }
            nfit += 1;
        }
    }
    fn summarize(rows: &[&Value]) -> Value {
        if rows.is_empty() {
            return Value::Null;
        }
        let n = rows.len();
        let mut cells = Vec::new();
        for (m, model) in ["weighted", "equal", MODEL, crate::ranking::MODEL].into_iter().enumerate() {
            for (i, r) in pool::TIER_RED_SIZES.into_iter().enumerate() {
                let total: u64 = rows
                    .iter()
                    .map(|x| x["models"][m]["hits"][i].as_u64().unwrap())
                    .sum();
                let full = rows
                    .iter()
                    .filter(|x| x["models"][m]["hits"][i] == 6)
                    .count();
                cells.push(json!({"model":model,"red_count":r,"n":n,"full_red_periods":full,"full_red_rate":full as f64/n as f64,
                    "mean_red_hits":total as f64/n as f64,"mean_missed_reds":6.0-total as f64/n as f64,
                    "mean_minimum_full_red_prefix":rows.iter().map(|x|x["models"][m]["minimum_full_red_prefix"].as_f64().unwrap()).sum::<f64>()/n as f64,
                    "mean_previous_selected":rows.iter().map(|x|x["models"][m]["previous_selected"][i].as_f64().unwrap()).sum::<f64>()/n as f64,
                    "uniform_full_red_rate":pool::uniform_full_red(r)}));
            }
        }
        let ticket_rows: Vec<_> = rows.iter().filter(|r| !r["tickets"].is_null()).collect();
        let mut tickets = serde_json::Map::new();
        if !ticket_rows.is_empty() {
            for name in [
                "frozen",
                "balanced_same12",
                "weighted14_greedy",
                "weighted14_balanced",
                "balanced14_rank",
            ] {
                let means: serde_json::Map<_, _> = [
                    "red_union",
                    "red_hits",
                    "best_red",
                    "mean_ticket_red_hits",
                    "max_exposure",
                ]
                .into_iter()
                .map(|k| {
                    (
                        k.into(),
                        json!(
                            ticket_rows
                                .iter()
                                .map(|x| x["tickets"][name][k].as_f64().unwrap())
                                .sum::<f64>()
                                / ticket_rows.len() as f64
                        ),
                    )
                })
                .collect();
                tickets.insert(
                    name.into(),
                    json!({"n":ticket_rows.len(),"bets":8,"cost":16,"means":means}),
                );
            }
        }
        let blue:Vec<_>=["weighted_blue_rank","equal_blue_rank"].into_iter().map(|model|json!({"model":model,
            "covered_periods":pool::BLUE_SIZES.map(|b|rows.iter().filter(|r|r[model].as_u64().unwrap()<=b as u64).count())})).collect();
        json!({"n":n,"from":rows[0]["issue"],"to":rows[n-1]["issue"],"cells":cells,"tickets":tickets,"blue":blue})
    }
    let references: Vec<_> = rows.iter().collect();
    let boundary = rows
        .iter()
        .position(|r| r["issue"] == fmt_issue(Protocol::CONFIRM_START_ISSUE))
        .ok_or("缺少固定确认窗起点")?;
    let end = (boundary + Protocol::CONFIRMATION_LEN).min(rows.len());
    let prepared = crate::eval::prepare_forecast(draws)?;
    Ok(
        json!({"kind":"exploratory_coverage_optimization_replay","actual_recommendation":false,"policy":policy(),
        "data_sha256":crate::draw::data_sha256(draws),"cutoff_issue":fmt_issue(prepared.cutoff_issue),
        "source_sha256":crate::hash::sha256_hex(concat!(include_str!("coverage.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("eval.rs"),include_str!("pool.rs"),include_str!("protocol.rs"),include_str!("stats.rs"),include_str!("ranking.rs")).as_bytes()),
        "windows":{"development":summarize(&references[..boundary]),"confirmation_exploratory":summarize(&references[boundary..end]),
            "tracking":summarize(&references[end..]),"recent30":summarize(&references[references.len().saturating_sub(30)..]),
            "recent60":summarize(&references[references.len().saturating_sub(60)..]),"all_oos":summarize(&references)},
        "latest_replay":rows.last(),"next_preview":saved(&prepared,draws.last().ok_or("没有开奖数据")?.red_mask),
        "ranking_policy":crate::ranking::policy(),"next_ranking_preview":crate::ranking::saved(&prepared,draws.last().unwrap().red_mask),
        "adoption":{"rank_candidate_promoted":false,"live_ticket_ranking":"weighted","decision_cutoff":"26118",
            "reason":"截至26118的近60期，均衡排名在14/18红的全6红和平均覆盖均未优于原排名，保留为只读研究；只上线可核对的均衡组票。"},
        "ranking_adoption":{"primary_protocol_replaced":false,"observation_model":crate::ranking::MODEL,"reason":"校准模型仅作为新期同时冻结的试验方案；历史全6红率并非各窗口一致改善。"},
        "notes":["校准排名按上期6红/其余27红分组消除原评分集中偏置，保留组内原排名，无开奖号输入、名额硬限制或参数搜索。",
        "历史逐期只用此前数据；权重在25111前停止更新。所有候选固定公式，不按26118结果搜索参数。",
        "此报告是事后探索性回放，不补造历史推荐；跟踪期已查看，只有未来新期才能提供新的验证证据。",
        "组票同为8注16元；同12码池和同14码池分别比较，扩大池的收益不算筛选能力提升。",
        "均衡票组完整使用所选号码，降低单号曝光，但不保证单注中奖或提高收益。"]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tied_expert_scores_stay_tied_and_previous_draw_cannot_dominate_prefixes() {
        let flat = [[1.0; 33]; 7];
        let s = scores(&flat);
        assert!(s.iter().all(|n| *n == s[0]));
        let mut experts = flat;
        for e in &mut experts {
            for (i, p) in e.iter_mut().enumerate() {
                *p = (33 - i) as f64;
            }
        }
        let previous = (1 << 6) - 1;
        let ranking = order(&experts, previous);
        assert_eq!(
            ranking
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            33
        );
        for size in 1..=33 {
            assert!(
                ranking[..size]
                    .iter()
                    .filter(|&&n| previous & (1 << n) != 0)
                    .count()
                    <= (6 * size).div_ceil(33)
            );
        }
    }
    #[test]
    fn future_outcomes_do_not_change_replayed_windows_or_old_pool_numbers() {
        let draws = crate::draw::load_csv(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("data/ssq.csv")
                .as_path(),
        )
        .unwrap();
        let short = analyze(&draws[..draws.len() - 1]).unwrap();
        let long = analyze(&draws).unwrap();
        assert_eq!(
            short["windows"]["development"],
            long["windows"]["development"]
        );
        assert_eq!(
            short["windows"]["confirmation_exploratory"],
            long["windows"]["confirmation_exploratory"]
        );
        assert_eq!(long["actual_recommendation"], false);
        assert_eq!(long["adoption"]["rank_candidate_promoted"], false);
        let prepared = crate::eval::prepare_forecast(&draws).unwrap();
        let spec = crate::plan::resolve(
            Some("balanced14"),
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
        let built = crate::plan::build(&prepared.forecast, &spec).unwrap();
        assert_eq!(
            built.reds,
            prepared.forecast.red_order[..14]
                .iter()
                .map(|n| *n as u8 + 1)
                .collect::<Vec<_>>()
        );
        assert_eq!(built.bet_count, 8);
        assert_eq!(built.cost, 16);
        assert!(built.rescued.is_empty());
    }

    #[test]
    fn balanced_bets_use_the_entire_selected_pool_without_hidden_bankers() {
        for size in [12, 14, 16, 18, 20, 22, 24] {
            let pool: Vec<_> = (0..size).collect();
            let bets = balanced_tickets(&pool, 8).unwrap();
            assert_eq!(bets.len(), 8);
            assert_eq!(
                bets.iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                8
            );
            assert_eq!(bets.iter().fold(0, |a, b| a | b), (1 << size) - 1);
            for &n in &pool {
                let e = bets.iter().filter(|m| *m & (1 << n) != 0).count();
                assert!((48 / size..=48usize.div_ceil(size)).contains(&e));
                assert!(e < 8);
            }
            assert!(bets.iter().all(|m| m.count_ones() == 6));
        }
        assert!(balanced_tickets(&(0..24).collect::<Vec<_>>(), 2).is_err());
    }
}
