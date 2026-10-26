//! Fixed, exploratory consensus variants; every ranking precedes the scored draw.
use serde_json::{json, Value};
use ssq::{draw, eval, model, pool, protocol::Protocol, select};
fn midranks(ps: &[[f64; 33]; 7]) -> [[f64; 33]; 7] {
    std::array::from_fn(|e| {
        let order = select::rank_desc(&ps[e]);
        let mut ranks = [0.; 33];
        let mut a = 0;
        while a < 33 {
            let mut b = a + 1;
            while b < 33 && select::score_key(ps[e][order[b]]) == select::score_key(ps[e][order[a]])
            {
                b += 1;
            }
            for &n in &order[a..b] {
                ranks[n] = (a + 1 + b) as f64 / 2.;
            }
            a = b;
        }
        ranks
    })
}
fn order(score: &[f64; 33], base: &[usize; 33]) -> [usize; 33] {
    let mut out = *base;
    out.sort_by(|&a, &b| select::score_key(score[b]).cmp(&select::score_key(score[a])));
    out
}
fn variants(ps: &[[f64; 33]; 7], base: &[usize; 33], previous: u64) -> [[usize; 33]; 5] {
    let ranks = midranks(ps);
    let mean = std::array::from_fn(|i| (34. - ranks.iter().map(|r| r[i]).sum::<f64>() / 7.) / 561.);
    let cal = ssq::ranking::scores(&mean, previous);
    let median = std::array::from_fn(|i| {
        let mut x = ranks.map(|r| r[i]);
        x.sort_by(f64::total_cmp);
        34. - x[3]
    });
    let top2 = std::array::from_fn(|i| {
        let mut x = ranks.map(|r| r[i]);
        x.sort_by(f64::total_cmp);
        34. - x[..2].iter().sum::<f64>() / 2.
    });
    let top3 = std::array::from_fn(|i| {
        let mut x = ranks.map(|r| r[i]);
        x.sort_by(f64::total_cmp);
        34. - x[..3].iter().sum::<f64>() / 3.
    });
    [
        order(&cal, base),
        order(&ssq::coverage::scores(ps), base),
        order(&median, base),
        order(&top2, base),
        order(&top3, base),
    ]
}
fn summary(rows: &[Value]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let names = [
        "weighted",
        "calibrated_consensus",
        "trimmed_consensus",
        "median_consensus",
        "dual_support",
        "triple_support",
    ];
    let cells:Vec<_>=names.iter().enumerate().flat_map(|(m,name)|pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,size)|{let full=rows.iter().filter(|r|r["hits"][m][j]==6).count();json!({"model":name,"red_count":size,"n":rows.len(),"full_red_periods":full,"mean_red_hits":rows.iter().map(|r|r["hits"][m][j].as_f64().unwrap()).sum::<f64>()/rows.len()as f64,"paired_gains":rows.iter().filter(|r|r["hits"][m][j]==6&&r["hits"][0][j]!=6).count(),"paired_losses":rows.iter().filter(|r|r["hits"][m][j]!=6&&r["hits"][0][j]==6).count(),"full_draw_by_blue":([1,2,3].map(|b|rows.iter().filter(|r|r["hits"][m][j]==6&&r["blue_rank"].as_u64().unwrap()<=b).count()))})})).collect();
    json!({"n":rows.len(),"from":rows.first().unwrap()["issue"],"to":rows.last().unwrap()["issue"],"cells":cells})
}
fn main() -> Result<(), String> {
    let file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = draw::load_csv(std::path::Path::new(&file))?;
    let snap = model::roll_all(&draws)?;
    let (mut sums, mut blue_sums, mut n) = ([0.; 7], [0.; 6], 0usize);
    let mut rows = Vec::new();
    let mut previous = 0;
    let rowmap: std::collections::BTreeMap<_, _> = snap.rows.iter().map(|r| (r.issue, r)).collect();
    for d in &draws {
        if let Some(row) = rowmap.get(&d.issue) {
            if n >= 30 {
                let base = select::rank_desc(&select::fuse7(
                    &row.expert_p,
                    &select::softmax_shrink(&sums, n as f64),
                ));
                let other = variants(&row.expert_p, &base, previous);
                let orders = [base, other[0], other[1], other[2], other[3], other[4]];
                let blue = select::rank_desc(&select::fuse6(
                    &row.blue_p,
                    &select::softmax_shrink(&blue_sums, n as f64),
                ));
                rows.push(json!({"issue":d.issue,"hits":orders.map(|o|pool::TIER_RED_SIZES.map(|k|o[..k].iter().filter(|&&i|d.red_mask&(1<<i)!=0).count())),"winning_ranks":orders.map(|o|d.red.map(|r|o.iter().position(|n|*n==r as usize-1).unwrap()+1)),"blue_rank":blue.iter().position(|n|*n==row.blue as usize).unwrap()+1}));
            }
            if row.issue < Protocol::CONFIRM_START_ISSUE {
                for e in 0..7 {
                    sums[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
                }
                for e in 0..6 {
                    blue_sums[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
                }
                n += 1;
            }
        }
        previous = d.red_mask;
    }
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let p = eval::prepare_forecast(&draws)?;
    let next = variants(
        &p.expert_red,
        &p.forecast.red_order,
        draws.last().unwrap().red_mask,
    );
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"exploratory_consensus_variants","actual_recommendation":false,"candidate_models":["calibrated_consensus","trimmed_consensus","median_consensus","dual_support","triple_support"],"multiple_candidates_examined":5,"independent_confirmation":false,"automatic_promotion":false,"blue_scope":"shared_weighted_blue_ranking","data_sha256":draw::data_sha256(&draws),"data_sha256_kind":"canonical_csv_lf","input_file_sha256":ssq::hash::sha256_hex(&std::fs::read(&file).map_err(|e|e.to_string())?),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("consensus_variants.rs"),include_str!("../src/model.rs"),include_str!("../src/eval.rs"),include_str!("../src/coverage.rs"),include_str!("../src/ranking.rs"),include_str!("../src/select.rs"),include_str!("../src/protocol.rs"),include_str!("../src/draw.rs")).as_bytes()),"next_red_rankings":next.map(|o|o.map(|n|n+1).to_vec()),"latest_replay":rows.last(),"windows":{"development":summary(&rows[..start]),"confirmation_exploratory":summary(&rows[start..end]),"tracking":summary(&rows[end..]),"recent60":summary(&rows[rows.len().saturating_sub(60)..]),"recent156":summary(&rows[rows.len().saturating_sub(156)..]),"all_oos":summary(&rows)}})).unwrap());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn research_dual_support_matches_frozen_algorithm_with_ties() {
        let draws = draw::synth_draws(220, 9);
        let p = eval::prepare_forecast(&draws).unwrap();
        let expected = variants(
            &p.expert_red,
            &p.forecast.red_order,
            draws.last().unwrap().red_mask,
        )[3];
        let models = ssq::pool_algorithms::saved_all(&p, &draws);
        let saved = models
            .iter()
            .find(|m| m["model"] == "dual_support")
            .unwrap();
        assert_eq!(
            saved["red_ranking"],
            json!(expected.map(|n| n + 1).to_vec())
        );
    }
}
