use serde_json::{json, Value};
use ssq::{draw, model, protocol::Protocol, select};
use std::collections::VecDeque;
fn bucket(gap: usize) -> usize {
    match gap {
        0..=4 => gap,
        5..=7 => 5,
        8..=11 => 6,
        12..=17 => 7,
        _ => 8,
    }
}
fn logit(p: f64) -> f64 {
    (p / (1. - p)).ln()
}
fn main() -> Result<(), String> {
    let draws = draw::load_csv(std::path::Path::new(
        &std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/ssq.csv".into()),
    ))?;
    let snap = model::roll_all(&draws)?;
    let rowmap: std::collections::BTreeMap<_, _> = snap.rows.iter().map(|r| (r.issue, r)).collect();
    let (mut opportunities, mut wins, mut number, mut last, mut history, mut sums, mut n) = (
        [0usize; 9],
        [0usize; 9],
        [0usize; 33],
        [None; 33],
        VecDeque::new(),
        [0.; 7],
        0usize,
    );
    let mut rows = Vec::new();
    for (t, draw) in draws.iter().enumerate() {
        let buckets: [usize; 33] =
            std::array::from_fn(|i| bucket(last[i].map(|j| t - j - 1).unwrap_or(t)));
        if let Some(row) = rowmap.get(&draw.issue) {
            if n >= 30 {
                let base = select::rank_desc(&select::fuse7(
                    &row.expert_p,
                    &select::softmax_shrink(&sums, n as f64),
                ));
                let hazard: [f64; 33] = std::array::from_fn(|i| {
                    (wins[buckets[i]] as f64 + 156. * 6. / 33.)
                        / (opportunities[buckets[i]] as f64 + 156.)
                });
                let combo: [f64; 33] = std::array::from_fn(|i| {
                    logit(hazard[i])
                        + logit(
                            (number[i] as f64 + 156. * 6. / 33.) / (history.len() as f64 + 156.),
                        )
                        - logit(6. / 33.)
                });
                let mut tied = base;
                tied.sort_by(|&a, &b| {
                    select::score_key(hazard[b]).cmp(&select::score_key(hazard[a]))
                });
                let mut protected = base;
                protected[20..].sort_by(|&a, &b| {
                    select::score_key(hazard[b]).cmp(&select::score_key(hazard[a]))
                });
                let ranks = [
                    base,
                    select::rank_desc(&hazard),
                    select::rank_desc(&combo),
                    tied,
                    protected,
                ];
                rows.push(json!({"issue":draw.issue,"hits":ranks.iter().map(|o|ssq::pool::TIER_RED_SIZES.map(|k|o[..k].iter().filter(|&&i|draw.red_mask&(1<<i)!=0).count())).collect::<Vec<_>>()}));
            }
            if row.issue < Protocol::CONFIRM_START_ISSUE {
                for e in 0..7 {
                    sums[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
                }
                n += 1;
            }
        }
        for i in 0..33 {
            opportunities[buckets[i]] += 1;
            if draw.red_mask & (1 << i) != 0 {
                wins[buckets[i]] += 1;
                number[i] += 1;
                last[i] = Some(t);
            }
        }
        history.push_back((buckets, draw.red_mask));
        if history.len() > 312 {
            let (b, m) = history.pop_front().unwrap();
            for i in 0..33 {
                opportunities[b[i]] -= 1;
                if m & (1 << i) != 0 {
                    wins[b[i]] -= 1;
                    number[i] -= 1;
                }
            }
        }
    }
    fn summary(rows: &[Value]) -> Value {
        let cells:Vec<_>=["weighted","hazard_number_ties","hazard_frequency312","conditional_gap","hazard_tail20"].into_iter().enumerate().flat_map(|(m,model)|ssq::pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,k)|json!({"model":model,"red_count":k,"n":rows.len(),"full_red_periods":rows.iter().filter(|r|r["hits"][m][j]==6).count(),"mean_red_hits":rows.iter().map(|r|r["hits"][m][j].as_f64().unwrap()).sum::<f64>()/rows.len() as f64}))).collect();
        json!({"n":rows.len(),"cells":cells})
    }
    let b = rows.iter().position(|r| r["issue"] == 25111).unwrap();
    println!("{}",serde_json::to_string_pretty(&json!({"actual_recommendation":false,"data_sha256":draw::data_sha256(&draws),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("conditional_candidates.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/protocol.rs")).as_bytes()),"windows":{"all_oos":summary(&rows),"development":summary(&rows[..b]),"confirmation_exploratory":summary(&rows[b..b+156]),"recent60":summary(&rows[rows.len()-60..])},"latest":rows.last()})).unwrap());
    Ok(())
}
