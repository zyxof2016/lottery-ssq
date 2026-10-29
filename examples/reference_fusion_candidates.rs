//! Six fixed reference mixtures. Select on the first 197 valid rows only.
//! Imported reference traces are exploratory evidence, not prospective archives.
use serde_json::{json, Value};
use ssq::{draw, hash::sha256_hex, model, protocol::Protocol, select};
use std::{collections::BTreeMap, path::Path};

const SIZES: [usize; 4] = [12, 14, 15, 16];
const DEVELOPMENT: usize = 197;
const MODELS: [(&str, &str, f64); 8] = [
    ("weighted", "baseline", 0.),
    ("small_recent", "baseline", 0.),
    ("exposure10", "ticket_exposure", 0.1),
    ("exposure25", "ticket_exposure", 0.25),
    ("exposure50", "ticket_exposure", 0.5),
    ("union10", "uniform_union_support", 0.1),
    ("union25", "uniform_union_support", 0.25),
    ("union50", "uniform_union_support", 0.5),
];

#[derive(Clone)]
struct Row {
    issue: u32,
    hits: [[usize; 4]; 8],
}

fn summary(rows: &[Row], m: usize, s: usize) -> Value {
    let full = rows.iter().filter(|r| r.hits[m][s] == 6).count();
    let gains = rows
        .iter()
        .filter(|r| r.hits[m][s] == 6 && r.hits[1][s] != 6)
        .count();
    let losses = rows
        .iter()
        .filter(|r| r.hits[m][s] != 6 && r.hits[1][s] == 6)
        .count();
    json!({"model":MODELS[m].0,"red_count":SIZES[s],"n":rows.len(),
        "full_red_periods":full,"full_red_rate":full as f64/rows.len() as f64,
        "paired_gains_vs_small":gains,"paired_losses_vs_small":losses,
        "mean_red_hits":rows.iter().map(|r|r.hits[m][s]).sum::<usize>() as f64/rows.len() as f64})
}

fn selection_key(rows: &[Row], m: usize) -> (usize, usize, i32, bool) {
    let full16 = rows.iter().filter(|r| r.hits[m][3] == 6).count();
    let smaller = rows
        .iter()
        .map(|r| r.hits[m][..3].iter().filter(|&&n| n == 6).count())
        .sum();
    (
        full16,
        smaller,
        -(MODELS[m].2 * 100.) as i32,
        MODELS[m].1 == "ticket_exposure",
    )
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = draw::load_csv(Path::new(args.get(1).ok_or("CSV required")?))?;
    let imported_bytes =
        std::fs::read(args.get(2).ok_or("reference trace required")?).map_err(|e| e.to_string())?;
    let imported: Value = serde_json::from_slice(&imported_bytes).map_err(|e| e.to_string())?;
    let mut records = BTreeMap::new();
    for r in imported["records"].as_array().ok_or("records missing")? {
        let issue: u32 = r["issue"]
            .as_str()
            .ok_or("issue missing")?
            .parse()
            .map_err(|_| "invalid issue")?;
        if records.insert(issue, r).is_some() {
            return Err("duplicate reference issue".into());
        }
        let pos = draws
            .iter()
            .position(|d| d.issue == issue)
            .ok_or("reference target absent from CSV")?;
        if pos == 0 || r["data_cutoff"] != draw::fmt_issue(draws[pos - 1].issue) {
            return Err(format!(
                "{issue}: reference cutoff must be actual preceding issue"
            ));
        }
        let actual: Vec<u8> =
            serde_json::from_value(r["actual"]["red"].clone()).map_err(|e| e.to_string())?;
        if actual != draws[pos].red.to_vec() || r["actual"]["blue"] != draws[pos].blue {
            return Err(format!("{issue}: reference outcome differs from CSV"));
        }
        for t in r["tickets"].as_array().ok_or("tickets missing")? {
            let red: Vec<u8> =
                serde_json::from_value(t["red"].clone()).map_err(|e| e.to_string())?;
            let mut unique = red.clone();
            unique.sort();
            unique.dedup();
            if red.len() != 6
                || unique.len() != 6
                || red.iter().any(|n| !(1..=33).contains(n))
                || !t["blue"].as_u64().is_some_and(|b| (1..=16).contains(&b))
            {
                return Err(format!("{issue}: invalid ticket"));
            }
        }
    }
    let roll = model::roll_all(&draws)?;
    let (mut sums, mut n) = ([0.; 7], 0usize);
    let mut rows = Vec::new();
    let mut excluded = Vec::new();
    for r in &roll.rows {
        if let Some(reference) = records.get(&r.issue) {
            let tickets = reference["tickets"].as_array().unwrap();
            if tickets.len() != 5 {
                excluded.push(r.issue);
            } else {
                if n < 30 {
                    return Err("local model warmup unavailable".into());
                }
                let original = select::fuse7(&r.expert_p, &select::softmax_shrink(&sums, n as f64));
                let small = ssq::small_red::scores(&original, &r.expert_p);
                let mut exposure = [0.; 33];
                let mut union = 0u64;
                for ticket in tickets {
                    for number in ticket["red"].as_array().unwrap() {
                        let k = number.as_u64().unwrap() as usize - 1;
                        exposure[k] += 1. / 30.;
                        union |= 1 << k;
                    }
                }
                let uniform: [f64; 33] = std::array::from_fn(|i| {
                    if union & (1 << i) != 0 {
                        1. / union.count_ones() as f64
                    } else {
                        0.
                    }
                });
                let orders: [[usize; 33]; 8] = std::array::from_fn(|m| {
                    if m == 0 {
                        return select::rank_desc(&original);
                    }
                    if m == 1 {
                        return select::rank_desc(&small);
                    }
                    let reference = if MODELS[m].1 == "ticket_exposure" {
                        &exposure
                    } else {
                        &uniform
                    };
                    let alpha = MODELS[m].2;
                    select::rank_desc(&std::array::from_fn(|i| {
                        (1. - alpha) * small[i] + alpha * reference[i]
                    }))
                });
                rows.push(Row {
                    issue: r.issue,
                    hits: orders.map(|order| {
                        SIZES.map(|k| {
                            order[..k]
                                .iter()
                                .filter(|&&x| r.red_mask & (1 << x) != 0)
                                .count()
                        })
                    }),
                });
            }
        }
        // Learn only after this target's prediction, and freeze at original protocol boundary.
        if r.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                sums[e] += select::rank_log_score(&r.expert_p[e], r.red_mask);
            }
            n += 1;
        }
    }
    if rows.len() != DEVELOPMENT + 100 {
        return Err(format!(
            "expected exactly 297 valid comparisons, found {}",
            rows.len()
        ));
    }
    let development = &rows[..DEVELOPMENT];
    // Model choice sees only development; the following report may inspect later outcomes.
    let selected = (2..MODELS.len())
        .max_by_key(|&m| selection_key(development, m))
        .unwrap();
    let dev_better16 = selection_key(development, selected).0 > selection_key(development, 1).0;
    let mut windows = serde_json::Map::new();
    for (name, span) in [
        ("development_first197", development),
        ("later100_exploratory", &rows[DEVELOPMENT..]),
        ("all_exploratory", &rows[..]),
    ] {
        let cells: Vec<_> = (0..MODELS.len())
            .flat_map(|m| (0..4).map(move |s| summary(span, m, s)))
            .collect();
        windows.insert(name.into(), json!({"n":span.len(),"from":span.first().unwrap().issue,"to":span.last().unwrap().issue,"cells":cells}));
    }
    let report = json!({"kind":"fixed_reference_fusion_candidate_exploration","actual_recommendation":false,
        "reference_runtime_independently_reproduced":false,"independent_confirmation":false,"automatic_adoption":false,
        "input_data_sha256":draw::data_sha256(&draws),"imported_trace_file_sha256":sha256_hex(&imported_bytes),
        "reference_trace_source_sha256":imported["source_sha256"],"source_sha256":sha256_hex(include_bytes!("reference_fusion_candidates.rs")),
        "candidate_count":6,"sizes":SIZES,"models":MODELS.iter().map(|&(name,method,weight)|json!({"model":name,"method":method,"reference_weight":weight})).collect::<Vec<_>>(),
        "selection":{"used_rows":197,"unused_for_selection_rows":100,"rule":"maximize first197 full16; ties maximize full12+full14+full15; then smaller reference weight; then exposure before union", "selected_candidate":MODELS[selected].0,"selected_beats_small_recent_development_full16":dev_better16,
            "note":"The original 50% mixture and this imported span have already been inspected. Later100 is exploratory inspection, not independent confirmation. Imported reference predictions have not been independently rerun."},
        "excluded_not_five_tickets":excluded,"windows":windows,
        "rows":rows.iter().map(|r|json!({"issue":r.issue,"hits":r.hits})).collect::<Vec<_>>()});
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
