use serde_json::json;
use ssq::{draw, model, protocol::Protocol, select};
use std::collections::VecDeque;
fn main() {
    let file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = draw::load_csv(std::path::Path::new(&file)).unwrap();
    let snap = model::roll_all(&draws).unwrap();
    let base = (1u64 << 16) - 1;
    let mut candidates = vec![base];
    let mut swaps = vec![0usize];
    for a in 0..16 {
        for b in 16..33 {
            candidates.push(base ^ (1 << a) ^ (1 << b));
            swaps.push(1);
        }
    }
    for a in 0..16 {
        for b in a + 1..16 {
            for c in 16..33 {
                for d in c + 1..33 {
                    candidates.push(base ^ (1 << a) ^ (1 << b) ^ (1 << c) ^ (1 << d));
                    swaps.push(2);
                }
            }
        }
    }
    let mut expand = vec![[0i32; 7]; candidates.len()];
    let mut roll = expand.clone();
    let mut history = VecDeque::new();
    let (mut sums, mut n) = ([0.; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        if n >= 30 {
            let order = select::rank_desc(&select::fuse7(
                &row.expert_p,
                &select::softmax_shrink(&sums, n as f64),
            ));
            let mask = ssq::tail::rank_mask(&order, row.red_mask);
            let mut chosen = Vec::new();
            for (counts, maxswap) in [(&expand, 1usize), (&expand, 2), (&roll, 1), (&roll, 2)] {
                let best = if history.len() < 156 {
                    0
                } else {
                    (0..candidates.len())
                        .filter(|&i| swaps[i] <= maxswap)
                        .max_by_key(|&i| {
                            (
                                counts[i][6],
                                counts[i][5],
                                counts[i]
                                    .iter()
                                    .enumerate()
                                    .map(|(k, c)| k as i32 * c)
                                    .sum::<i32>(),
                                std::cmp::Reverse(swaps[i]),
                                std::cmp::Reverse(i),
                            )
                        })
                        .unwrap()
                };
                chosen.push(best);
            }
            let mut values = vec![(mask & base).count_ones()];
            values.extend(chosen.iter().map(|&i| (mask & candidates[i]).count_ones()));
            rows.push(json!({"issue":row.issue,"hits":values,"selected":chosen.iter().map(|&i|candidates[i]).collect::<Vec<_>>(),"rank_mask":mask}));
            for (i, pool) in candidates.iter().enumerate() {
                let h = (mask & pool).count_ones() as usize;
                expand[i][h] += 1;
                roll[i][h] += 1;
            }
            history.push_back(mask);
            if history.len() > 312 {
                let old = history.pop_front().unwrap();
                for (i, pool) in candidates.iter().enumerate() {
                    roll[i][(old & pool).count_ones() as usize] -= 1;
                }
            }
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                sums[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
            }
            n += 1;
        }
    }
    let start = rows.iter().position(|r| r["issue"] == 25111).unwrap();
    let end = start + 156;
    let mut windows = serde_json::Map::new();
    for (name, a, b) in [
        ("development", 0, start),
        ("confirmation", start, end),
        ("tracking", end, rows.len()),
        ("recent60", rows.len() - 60, rows.len()),
        ("all", 0, rows.len()),
    ] {
        let full = (0..5)
            .map(|m| rows[a..b].iter().filter(|r| r["hits"][m] == 6).count())
            .collect::<Vec<_>>();
        windows.insert(name.into(), json!({"n":b-a,"full":full}));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"exploratory_small_red_swap_search","actual_recommendation":false,"automatic_promotion":false,"data_sha256":draw::data_sha256(&draws),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("small_red_swaps.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/eval.rs"),include_str!("../src/ranking.rs")).as_bytes()),"candidates":candidates.len(),"methods":["weighted","swap1_expand","swap2_expand","swap1_roll312","swap2_roll312"],"windows":windows,"rows":rows})).unwrap());
}
