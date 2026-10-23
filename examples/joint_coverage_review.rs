//! Causal blue-ranking and 22-red intersection research; never changes frozen recommendations.
use serde_json::{json, Value};
use ssq::{draw, model, protocol::Protocol, select};
fn borda<const N: usize, const E: usize>(experts: &[[f64; N]; E], base: &[usize; N]) -> [usize; N] {
    let mut units = [0u32; N];
    for p in experts {
        let o = select::rank_desc(p);
        let mut a = 0;
        while a < N {
            let mut b = a + 1;
            while b < N && select::score_key(p[o[b]]) == select::score_key(p[o[a]]) {
                b += 1;
            }
            for &n in &o[a..b] {
                units[n] += (2 * N + 1 - a - b) as u32;
            }
            a = b;
        }
    }
    let mut order = *base;
    order.sort_by_key(|&i| std::cmp::Reverse(units[i]));
    order
}
fn summarize(rows: &[Value]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells:Vec<_>=["weighted","equal","rank_consensus"].iter().enumerate().flat_map(|(m,name)|(1..=3).map(move |k|{
        let hits=rows.iter().filter(|r|r["blue_ranks"][m].as_u64().unwrap()<=k).count();
        json!({"model":name,"blue_count":k,"n":rows.len(),"hit_periods":hits,"hit_rate":hits as f64/rows.len()as f64,
            "paired_gains":rows.iter().filter(|r|r["blue_ranks"][m].as_u64().unwrap()<=k&&r["blue_ranks"][0].as_u64().unwrap()>k).count(),
            "paired_losses":rows.iter().filter(|r|r["blue_ranks"][m].as_u64().unwrap()>k&&r["blue_ranks"][0].as_u64().unwrap()<=k).count(),
            "joint_full6":(0..3).map(|red|rows.iter().filter(|r|r["red22_full"][red]==true&&r["blue_ranks"][m].as_u64().unwrap()<=k).count()).collect::<Vec<_>>()})
    })).collect();
    let full = rows
        .iter()
        .filter(|r| r["intersection22"]["red_hits"] == 6)
        .count();
    let matched = rows
        .iter()
        .filter(|r| r["intersection22"]["matched_weighted_hits"] == 6)
        .count();
    json!({"n":rows.len(),"from":rows.first().unwrap()["issue"],"to":rows.last().unwrap()["issue"],"cells":cells,
        "red_intersection22":{"source_models":["rank_consensus","repeat_calibrated"],"source_red_count":22,
        "mean_red_count":rows.iter().map(|r|r["intersection22"]["red_count"].as_f64().unwrap()).sum::<f64>()/rows.len() as f64,
        "minimum_red_count":rows.iter().map(|r|r["intersection22"]["red_count"].as_u64().unwrap()).min(),
        "maximum_red_count":rows.iter().map(|r|r["intersection22"]["red_count"].as_u64().unwrap()).max(),
        "full_red_periods":full,"full_red_rate":full as f64/rows.len()as f64,"matched_weighted_full_red_periods":matched,
        "source_full_red_periods":{"repeat_calibrated":rows.iter().filter(|r|r["red22_full"][1]==true).count(),"rank_consensus":rows.iter().filter(|r|r["red22_full"][2]==true).count()},
        "full_draw_periods_by_blue":([1,2,3].map(|b|rows.iter().filter(|r|r["intersection22"]["red_hits"]==6&&r["blue_ranks"][0].as_u64().unwrap()<=b).count()))}})
}
fn periods(draws: &[draw::Draw]) -> Result<Vec<Value>, String> {
    let snap = model::roll_all(draws)?;
    let map: std::collections::BTreeMap<_, _> = snap.rows.iter().map(|r| (r.issue, r)).collect();
    let (mut red_sum, mut blue_sum, mut n) = ([0.; 7], [0.; 6], 0usize);
    let mut rows = Vec::new();
    for (t, d) in draws.iter().enumerate() {
        if let Some(row) = map.get(&d.issue) {
            if n >= 30 {
                let blue = select::rank_desc(&select::fuse6(
                    &row.blue_p,
                    &select::softmax_shrink(&blue_sum, n as f64),
                ));
                let equal = select::rank_desc(&select::fuse6(&row.blue_p, &[1. / 6.; 6]));
                let consensus = borda(&row.blue_p, &blue);
                let score =
                    select::fuse7(&row.expert_p, &select::softmax_shrink(&red_sum, n as f64));
                let red = select::rank_desc(&score);
                let calibrated =
                    select::rank_desc(&ssq::ranking::scores(&score, draws[t - 1].red_mask));
                let red_consensus = borda(&row.expert_p, &red);
                let blue_ranks = [blue, equal, consensus]
                    .map(|o| o.iter().position(|&i| i == row.blue as usize).unwrap() + 1);
                let red22_full = [red, calibrated, red_consensus].map(|o| {
                    o[..22]
                        .iter()
                        .filter(|&&i| d.red_mask & (1 << i) != 0)
                        .count()
                        == 6
                });
                let reds: Vec<_> = red_consensus[..22]
                    .iter()
                    .copied()
                    .filter(|i| calibrated[..22].contains(i))
                    .collect();
                let hits = reds.iter().filter(|&&i| d.red_mask & (1 << i) != 0).count();
                let matched = red[..reds.len()]
                    .iter()
                    .filter(|&&i| d.red_mask & (1 << i) != 0)
                    .count();
                rows.push(json!({"issue":d.issue,"blue_ranks":blue_ranks,"red22_full":red22_full,
                    "intersection22":{"red_count":reds.len(),"red_hits":hits,"matched_weighted_hits":matched,"reds":reds.iter().map(|n|n+1).collect::<Vec<_>>()}}));
            }
            if row.issue < Protocol::CONFIRM_START_ISSUE {
                for e in 0..7 {
                    red_sum[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
                }
                for e in 0..6 {
                    blue_sum[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
                }
                n += 1;
            }
        }
    }
    Ok(rows)
}
fn main() -> Result<(), String> {
    let file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = draw::load_csv(std::path::Path::new(&file))?;
    let rows = periods(&draws)?;
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let prepared = ssq::eval::prepare_forecast(&draws)?;
    let base = prepared.forecast.blue_order;
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"exploratory_joint_coverage_review","actual_recommendation":false,"automatic_promotion":false,"data_sha256":draw::data_sha256(&draws),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("joint_coverage_review.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/protocol.rs"),include_str!("../src/eval.rs"),include_str!("../src/ranking.rs")).as_bytes()),"joint_red_models_22":["weighted","repeat_calibrated","rank_consensus"],"next_blue_rankings":{"weighted":base.map(|n|n+1),"equal":select::rank_desc(&select::fuse6(&prepared.expert_blue,&[1./6.;6])).map(|n|n+1),"rank_consensus":borda(&prepared.expert_blue,&base).map(|n|n+1)},"latest":rows.last(),"windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),"tracking":summarize(&rows[end..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)}})).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_experts_preserve_original_tie_order() {
        let p = [[1.0 / 16.0; 16]; 6];
        let base = std::array::from_fn(|n| 15 - n);
        assert_eq!(borda(&p, &base), base);
    }
    #[test]
    fn later_outcomes_never_change_earlier_blue_or_joint_red_results() {
        let draws = draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let before = periods(&draws[..230]).unwrap();
        let after = periods(&draws[..240]).unwrap();
        assert!(!before.is_empty());
        assert_eq!(before, after[..before.len()]);
    }
}
