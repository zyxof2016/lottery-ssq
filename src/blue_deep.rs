//! Exact same-budget joint red/blue allocation, isolated from frozen jackpot v1.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    model::roll_all,
    plan::{mask_nums, PlanTicket},
    protocol::Protocol,
    recommendations as rec,
    select::{
        fuse6, fuse7, rank_desc, rank_log_score, score_key, softmax_shrink, Forecast, RescueResult,
    },
};
use serde_json::{json, Value};
use std::{
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    path::Path,
    sync::{Mutex, OnceLock},
};
pub fn policy() -> Value {
    json!({"version":1,"objective":"eight_highest_joint_red_blue_ranking_products","red_candidate_limit":16,"bets":8,"cost":16,"blue_limit":16,"forced_blue_count":false,"algorithm":"exact_monotone_best_first_over_six_red_indices_and_blue_rank","score":"sum_of_quantized_logs_of_quantized_original_scores","ties":"red_rank_indices_then_blue_rank","weights":"original_frozen_weights_unchanged","automatic_adoption":false,"score_meaning":"uncalibrated_model_assumption_not_verified_winning_probability"})
}
fn keys(fc: &Forecast) -> (Vec<i64>, Vec<i64>) {
    let key = |v: f64| ((score_key(v).max(1) as f64).ln() * 1e9).round() as i64;
    (
        fc.red_order[..16]
            .iter()
            .map(|i| key(fc.fusion[*i]))
            .collect(),
        fc.blue_order.iter().map(|i| key(fc.blue[*i])).collect(),
    )
}
pub(crate) fn build(fc: &Forecast) -> Vec<PlanTicket> {
    let (red, blue) = keys(fc);
    let value = |r: &[usize; 6], b: usize| r.iter().map(|i| red[*i]).sum::<i64>() + blue[b];
    let first = ([0, 1, 2, 3, 4, 5], 0usize);
    let mut heap = BinaryHeap::from([(value(&first.0, 0), Reverse(first))]);
    let mut seen = BTreeSet::from([first]);
    let mut out = vec![];
    while out.len() < 8 {
        let (_, Reverse((r, b))) = heap.pop().expect("sixteen-red joint frontier");
        out.push(PlanTicket {
            red: mask_nums(r.iter().fold(0, |m, i| m | (1 << fc.red_order[*i]))),
            blue: (fc.blue_order[b] + 1) as u8,
        });
        for i in 0..7 {
            let (mut next, mut nb) = (r, b);
            if i == 6 {
                nb += 1;
                if nb >= 16 {
                    continue;
                }
            } else {
                next[i] += 1;
                if next[i] >= 16 || (i < 5 && next[i] >= next[i + 1]) {
                    continue;
                }
            }
            if seen.insert((next, nb)) {
                heap.push((value(&next, nb), Reverse((next, nb))));
            }
        }
    }
    out
}
fn saved(tickets: &[PlanTicket]) -> Value {
    json!({"title":"红蓝联合选票观察·8注16元","mode":"research_joint","structural":true,"tickets":tickets.iter().map(|t|json!({"red":t.red,"blue":t.blue})).collect::<Vec<_>>()})
}
pub(crate) fn prediction(draws: &[Draw]) -> Result<Value, String> {
    let p = prepare_forecast(draws)?;
    let tickets = build(&p.forecast);
    let old = crate::jackpot::build(&p.forecast, &crate::jackpot::spec())?;
    Ok(
        json!({"policy":policy(),"candidate_reds":p.forecast.red_order[..16].iter().map(|i|i+1).collect::<Vec<_>>(),"blue_ranking":p.forecast.blue_order.map(|i|i+1),"red_scores":p.forecast.fusion.to_vec(),"blue_scores":p.forecast.blue,"tickets":saved(&tickets)["tickets"],"baseline_tickets":saved(&old.tickets)["tickets"]}),
    )
}
fn assess(tickets: &[PlanTicket], draw: &Draw) -> [bool; 3] {
    [
        tickets.iter().any(|t| t.blue == draw.blue),
        tickets.iter().any(|t| t.red == draw.red),
        tickets
            .iter()
            .any(|t| t.red == draw.red && t.blue == draw.blue),
    ]
}
#[derive(Clone)]
struct Row {
    issue: u32,
    metrics: [[bool; 3]; 2],
    blues: [usize; 2],
    third_in: bool,
}
fn replay(draws: &[Draw]) -> Result<Vec<Row>, String> {
    let snap = roll_all(draws)?;
    let (mut rs, mut bs, mut n) = ([0.; 7], [0.; 6], 0usize);
    let mut out = vec![];
    for r in snap.rows {
        if n >= 30 {
            let fusion = fuse7(&r.expert_p, &softmax_shrink(&rs, n as f64));
            let blue = fuse6(&r.blue_p, &softmax_shrink(&bs, n as f64));
            let fc = Forecast {
                fusion,
                red_order: rank_desc(&fusion),
                blue,
                blue_order: rank_desc(&blue),
                mild: RescueResult {
                    mask: 0,
                    members: vec![],
                    rescued: vec![],
                    displaced: vec![],
                },
                pool: vec![],
                tickets: vec![],
                ticket_blues: vec![],
            };
            let a = crate::jackpot::build(&fc, &crate::jackpot::spec())?.tickets;
            let b = build(&fc);
            let draw = draws
                .iter()
                .find(|d| d.issue == r.issue)
                .ok_or("联合回放缺少开奖")?;
            let count =
                |ts: &[PlanTicket]| ts.iter().map(|t| t.blue).collect::<BTreeSet<_>>().len();
            out.push(Row {
                issue: r.issue,
                metrics: [assess(&a, draw), assess(&b, draw)],
                blues: [count(&a), count(&b)],
                third_in: b.iter().any(|t| t.blue == (fc.blue_order[2] + 1) as u8),
            });
        }
        if r.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                rs[e] += rank_log_score(&r.expert_p[e], r.red_mask);
            }
            for e in 0..6 {
                bs[e] += r.blue_p[e][r.blue as usize].clamp(1e-12, 1.).ln();
            }
            n += 1;
        }
    }
    Ok(out)
}
fn summary(rows: &[Row]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells:Vec<_>=["fixed4x2","joint8"].iter().enumerate().map(|(j,name)|json!({"model":name,"n":rows.len(),"blue_covered_periods":rows.iter().filter(|r|r.metrics[j][0]).count(),"single_six_red_periods":rows.iter().filter(|r|r.metrics[j][1]).count(),"jackpot_periods":rows.iter().filter(|r|r.metrics[j][2]).count(),"mean_blue_count":rows.iter().map(|r|r.blues[j]).sum::<usize>()as f64/rows.len()as f64,"paired_blue_gains":rows.iter().filter(|r|r.metrics[j][0]&&!r.metrics[0][0]).count(),"paired_blue_losses":rows.iter().filter(|r|!r.metrics[j][0]&&r.metrics[0][0]).count(),"blue_pairing_miss_periods":rows.iter().filter(|r|r.metrics[j][0]&&r.metrics[j][1]&&!r.metrics[j][2]).count(),"blue_count_histogram":(1..=8).map(|k|json!({"count":k,"n":rows.iter().filter(|r|r.blues[j]==k).count()})).collect::<Vec<_>>()})).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"third_blue_retained_periods":rows.iter().filter(|r|r.third_in).count(),"cells":cells})
}
fn pure_report(draws: &[Draw]) -> Result<Value, String> {
    let rows = replay(draws)?;
    let start = rows
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let p = prediction(draws)?;
    let plan = rec::review_strategy(
        &json!({"title":"红蓝联合选票观察·8注16元","mode":"research_joint","structural":true,"tickets":p["tickets"]}),
        None,
    )?;
    Ok(
        json!({"version":1,"kind":"blue_joint_allocation_review","policy":policy(),"data_sha256":data_sha256(draws),"target_issue":draws.last().map(|d|fmt_issue(d.issue+1)),"cutoff_issue":draws.last().map(|d|fmt_issue(d.issue)),"next":{"prediction":p,"plan":plan},"windows":{"all_oos":summary(&rows),"confirmation_exploratory":summary(&rows[start..end]),"tracking":summary(&rows[end..]),"recent60":summary(&rows[rows.len().saturating_sub(60)..]),"recent156":summary(&rows[rows.len().saturating_sub(156)..])},"source_sha256":sha256_hex(concat!(include_str!("blue_deep.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("protocol.rs"),include_str!("jackpot.rs")).as_bytes()),"note":"联合选票取消固定2蓝限制，允许第3蓝进入部分注单；更多不同蓝球会改变红球组数与蓝球重复注数。同8注公平头奖基准相同，回放不计实绩，尚未证明头奖优势。"}),
    )
}
fn latest_why(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    for d in draws.iter().rev().take(64) {
        if let Some(head) = crate::jackpot::view(root, draws, &fmt_issue(d.issue))? {
            let nodes = head["plan"]["numbers"]
                .as_array()
                .ok_or("原档缺少逐号评分")?;
            let mut blues: Vec<_> = nodes
                .iter()
                .filter(|v| v["color"] == "blue")
                .cloned()
                .collect();
            blues.sort_by_key(|v| v["fusion_rank"].as_u64().unwrap_or(99));
            let weights: [f64; 6] = serde_json::from_value(head["plan"]["blue_weights"].clone())
                .map_err(|_| "原档缺少蓝球权重")?;
            let winner = blues
                .iter()
                .find(|x| x["number"] == d.blue)
                .ok_or("原档缺少中奖蓝球评分")?;
            let cutoff = &blues[1];
            let winner_score = winner["fusion_score"].as_f64().ok_or("蓝球评分无效")?;
            let cutoff_score = cutoff["fusion_score"].as_f64().ok_or("蓝球截线评分无效")?;
            let contributions:Vec<_>=(0..6).map(|e|{let w=winner["expert_scores"][e]["score"].as_f64().unwrap_or(0.);let c=cutoff["expert_scores"][e]["score"].as_f64().unwrap_or(0.);json!({"expert":Protocol::BLUES[e],"label":Protocol::BLUES_ZH[e],"weight":weights[e],"winner_score":w,"cutoff_score":c,"winner_contribution":weights[e]*w,"cutoff_contribution":weights[e]*c,"winner_minus_cutoff":weights[e]*(w-c)})}).collect();
            let prefix: Vec<_> = draws
                .iter()
                .filter(|x| x.issue < d.issue)
                .cloned()
                .collect();
            let counter = prediction(&prefix)?;
            let ts: Vec<PlanTicket> = counter["tickets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    Ok(PlanTicket {
                        red: serde_json::from_value(t["red"].clone())
                            .map_err(|_| "联合红球无效")?,
                        blue: t["blue"].as_u64().ok_or("联合蓝球无效")? as u8,
                    })
                })
                .collect::<Result<_, String>>()?;
            let result = assess(&ts, d);
            return Ok(
                json!({"issue":fmt_issue(d.issue),"winning_blue":d.blue,"winning_rank":winner["fusion_rank"],"winning_score":winner_score,"selected_blues":head["plan"]["summary"]["unique_blues"],"cutoff_blue":cutoff["number"],"cutoff_score":cutoff_score,"gap_to_second":cutoff_score-winner_score,"archive_sha256":head["archive_sha256"],"url":head["url"],"contributions":contributions,"all_blue_scores":blues.iter().map(|b|json!({"number":b["number"],"rank":b["fusion_rank"],"score":b["fusion_score"],"ticket_exposure":b["ticket_exposure"]})).collect::<Vec<_>>(),"counterfactual":{"actual_recommendation":false,"blue_in_joint_tickets":result[0],"single_six_red":result[1],"jackpot":result[2],"blue_count":ts.iter().map(|t|t.blue).collect::<BTreeSet<_>>().len(),"note":"同输入、同8注的事后结构回放；没有在当时发布，不能补算实际命中。"}}),
            );
        }
    }
    Ok(Value::Null)
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let key = data_sha256(draws);
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "蓝球联合缓存异常")?;
    if cache.as_ref().is_none_or(|(k, _)| *k != key) {
        *cache = Some((key, pure_report(draws)?));
    }
    let mut out = cache.as_ref().unwrap().1.clone();
    drop(cache);
    out["latest_why"] = latest_why(root, draws)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(kind: usize) -> Forecast {
        let red: [f64; 33] = std::array::from_fn(|i| match kind {
            0 => 1.,
            1 => (33 - i) as f64,
            2 => ((i * 7) % 33 + 1) as f64,
            _ => 1. + ((i * 11) % 4) as f64 * 1e-8,
        });
        let blue: [f64; 16] = std::array::from_fn(|i| match kind {
            0 => 1.,
            1 => (16 - i) as f64,
            2 => ((i * 5) % 16 + 1) as f64,
            _ => 1. + ((i * 13) % 3) as f64 * 1e-8,
        });
        Forecast {
            fusion: red,
            red_order: rank_desc(&red),
            blue,
            blue_order: rank_desc(&blue),
            mild: RescueResult {
                mask: 0,
                members: vec![],
                rescued: vec![],
                displaced: vec![],
            },
            pool: vec![],
            tickets: vec![],
            ticket_blues: vec![],
        }
    }
    #[test]
    fn joint_frontier_matches_all_128128_tickets_including_ties() {
        for k in 0..4 {
            let fc = fixture(k);
            let raw = |v: f64| {
                ((((v / Protocol::SCORE_QUANTUM).round().max(1.)).ln() * 1e9).round()) as i64
            };
            let mut all = vec![];
            for mask in (0..1u64 << 16).filter(|m| m.count_ones() == 6) {
                let positions: Vec<_> = (0..16).filter(|i| mask & (1 << i) != 0).collect();
                let rkey = positions
                    .iter()
                    .map(|i| raw(fc.fusion[fc.red_order[*i]]))
                    .sum::<i64>();
                for b in 0..16 {
                    all.push((rkey + raw(fc.blue[fc.blue_order[b]]), positions.clone(), b));
                }
            }
            assert_eq!(all.len(), 128128);
            all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            let expected: Vec<_> = all[..8]
                .iter()
                .map(|(_, r, b)| {
                    (
                        mask_nums(r.iter().fold(0, |m, i| m | (1 << fc.red_order[*i]))),
                        (fc.blue_order[*b] + 1) as u8,
                    )
                })
                .collect();
            assert_eq!(
                build(&fc)
                    .iter()
                    .map(|t| (t.red, t.blue))
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
    #[test]
    fn eight_unique_legal_tickets_can_include_third_blue_without_increasing_cost() {
        let mut fc = fixture(0);
        fc.fusion = std::array::from_fn(|i| if i < 6 { 1. } else { 0.9 - i as f64 * 0.001 });
        fc.red_order = rank_desc(&fc.fusion);
        fc.blue = [0.001; 16];
        fc.blue[..3].copy_from_slice(&[0.080, 0.074, 0.0734]);
        fc.blue_order = rank_desc(&fc.blue);
        let joint = build(&fc);
        assert_eq!(joint.len(), 8);
        assert_eq!(
            joint
                .iter()
                .map(|t| (t.red, t.blue))
                .collect::<BTreeSet<_>>()
                .len(),
            8
        );
        assert!(joint.iter().any(|t| t.blue == 3));
        assert!(joint
            .iter()
            .all(|t| t.red.iter().all(|n| *n <= 16) && (1..=16).contains(&t.blue)));
        assert_eq!(
            rec::review_strategy(&saved(&joint), None).unwrap()["summary"]["cost"],
            16
        );
    }
    #[test]
    fn appending_future_results_does_not_change_prior_joint_replay() {
        let d = crate::draw::synth_draws(240, 37);
        let a = replay(&d[..220]).unwrap();
        let b = replay(&d).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.issue, y.issue);
            assert_eq!(x.metrics, y.metrics);
            assert_eq!(x.blues, y.blues);
            assert_eq!(x.third_in, y.third_in);
        }
    }
}
