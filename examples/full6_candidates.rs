use serde_json::{json, Value};
use ssq::{draw, model, protocol::Protocol, select};
use std::collections::VecDeque;
fn main() -> Result<(), String> {
    let draws = draw::load_csv(std::path::Path::new("data/ssq.csv"))?;
    let snap = model::roll_all(&draws)?;
    let (mut sums, mut n) = ([0.; 7], 0usize);
    let mut history = VecDeque::new();
    let mut rows = Vec::new();
    for row in &snap.rows {
        if n >= 30 {
            let score = select::fuse7(&row.expert_p, &select::softmax_shrink(&sums, n as f64));
            let mut orders = vec![
                select::rank_desc(&score),
                select::rank_desc(&select::fuse7(&row.expert_p, &select::equal7())),
            ];
            for e in [0, 1, 2, 3, 4] {
                orders.push(select::rank_desc(&row.expert_p[e]));
            }
            let mut full = [[0usize; 6]; 7];
            for past in &history {
                let p: &[[u32; 6]; 7] = past;
                for m in 0..7 {
                    for k in 0..6 {
                        full[m][k] += usize::from(p[m][k] == 6);
                    }
                }
            }
            let best = (0..7)
                .max_by_key(|&m| (full[m][5], full[m][..5].iter().sum::<usize>(), 7 - m))
                .unwrap();
            let use_best = history.len() >= 156
                && full[best][5] >= full[0][5] + 3
                && full[best][..5].iter().sum::<usize>() >= full[0][..5].iter().sum::<usize>();
            let chosen = orders[if use_best { best } else { 0 }];
            let mut consensus = [0.; 33];
            let mut w = [0.; 7];
            let max = full.iter().map(|x| x[5]).max().unwrap();
            for m in 0..7 {
                w[m] = ((full[m][5] as f64 - max as f64) / 3.).exp();
                for (rank, &i) in orders[m].iter().enumerate() {
                    consensus[i] += w[m] * (33 - rank) as f64;
                }
            }
            let fused = if history.len() >= 156 {
                select::rank_desc(&consensus)
            } else {
                orders[0]
            };
            let hit: Vec<_> = [orders[0], chosen, fused]
                .iter()
                .map(|o| {
                    ssq::pool::TIER_RED_SIZES.map(|k| {
                        o[..k]
                            .iter()
                            .filter(|&&i| row.red_mask & (1 << i) != 0)
                            .count() as u32
                    })
                })
                .collect();
            rows.push(json!({"issue":row.issue,"hits":hit,"active":use_best,"selected":if use_best {best}else{0}}));
            let actual: [[_; 6]; 7] = std::array::from_fn(|m| {
                ssq::pool::TIER_RED_SIZES.map(|k| {
                    orders[m][..k]
                        .iter()
                        .filter(|&&i| row.red_mask & (1 << i) != 0)
                        .count() as u32
                })
            });
            history.push_back(actual);
            if history.len() > 156 {
                history.pop_front();
            }
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                sums[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
            }
            n += 1;
        }
    }
    fn summary(rows: &[Value]) -> Value {
        let cells:Vec<_>=["weighted","coverage_selector156","coverage_consensus156"].into_iter().enumerate().flat_map(|(m,model)|ssq::pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,k)|json!({"model":model,"red_count":k,"n":rows.len(),"full_red_periods":rows.iter().filter(|r|r["hits"][m][j]==6).count(),"mean_red_hits":rows.iter().map(|r|r["hits"][m][j].as_f64().unwrap()).sum::<f64>()/rows.len() as f64}))).collect();
        json!({"n":rows.len(),"active":rows.iter().filter(|r|r["active"]==true).count(),"cells":cells})
    }
    let boundary = rows.iter().position(|r| r["issue"] == 25111).unwrap();
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"exploratory_full6_candidates","actual_recommendation":false,"data_sha256":draw::data_sha256(&draws),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("full6_candidates.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/protocol.rs")).as_bytes()),"windows":{"all_oos":summary(&rows),"development":summary(&rows[..boundary]),"confirmation_exploratory":summary(&rows[boundary..boundary+156]),"recent60":summary(&rows[rows.len()-60..])},"latest":rows.last()})).unwrap());
    Ok(())
}
