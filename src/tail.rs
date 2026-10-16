//! Learn which original rank positions to exclude, optimizing all-six coverage.
//! Every decision uses completed earlier draws; validation is separate from fit.
use crate::{
    draw::Draw,
    model::roll_all,
    protocol::Protocol,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};
use std::collections::VecDeque;

pub const MODEL: &str = "tail_coverage";
pub const LABEL: &str = "全6红尾部优化 · 试验";
pub const VERSION: u32 = 1;
const FIT: usize = 312;
const MIN_FIT: usize = 156;
const VALIDATE: usize = 52;
const ALL: u64 = (1u64 << 33) - 1;

pub fn policy() -> Value {
    json!({"model":MODEL,"label":LABEL,"version":VERSION,
        "objective":"full_six_red_coverage_at_24_then_nested_14_to_23",
        "method":"rank_position_joint_exclusion_greedy_plus_one_swap",
        "training_max":FIT,"training_min":MIN_FIT,"validation_periods":VALIDATE,
        "validation_blocks":[26,26],"validation_min_full24_gain":2,
        "validation_smaller_pool_full_count_must_not_decline":true,
        "fallback":"weighted","blue_scope":"shared_weighted_blue_ranking",
        "automatic_promotion":false,"score_kind":"ordinal_rank_evidence_not_probability",
        "note":"按过去开奖号在原排名中的位置，联合选择24红应排除的9个名次，再向下构造嵌套小池。训练与后续52期验证分开；两段验证均不退步、24红多全包至少2期且小池全包合计不降时使用试验顺序，否则回退原排名。无未来覆盖保证。"})
}

#[derive(Clone)]
pub struct Observation {
    pub issue: u32,
    pub mask: u64,
}

#[derive(Default)]
pub struct State {
    history: VecDeque<Observation>,
}

pub struct Decision {
    pub order: [usize; 33],
    pub candidate: [usize; 33],
    pub accepted: bool,
    pub evidence: Value,
}

fn prefix(order: &[usize; 33], size: usize) -> u64 {
    order[..size].iter().fold(0, |m, n| m | (1 << n))
}
fn counts(order: &[usize; 33], rows: &[Observation]) -> [usize; 6] {
    crate::pool::TIER_RED_SIZES.map(|k| {
        let pool = prefix(order, k);
        rows.iter().filter(|r| r.mask & !pool == 0).count()
    })
}

/// Small bitsets make the joint objective cheap without approximating counts.
fn optimize(rows: &[Observation]) -> [usize; 33] {
    let words = rows.len().div_ceil(64);
    let mut win: Vec<Vec<u64>> = (0..33).map(|_| vec![0; words]).collect();
    for (j, row) in rows.iter().enumerate() {
        for n in 0..33 {
            if row.mask & (1 << n) != 0 {
                win[n][j / 64] |= 1 << (j % 64);
            }
        }
    }
    let mut alive = vec![u64::MAX; words];
    if rows.len() % 64 != 0 {
        *alive.last_mut().unwrap() = (1u64 << (rows.len() % 64)) - 1;
    }
    let initial = alive.clone();
    let survive = |excluded: u64| -> Vec<u64> {
        let mut bits = initial.clone();
        for n in 0..33 {
            if excluded & (1 << n) != 0 {
                for w in 0..words {
                    bits[w] &= !win[n][w];
                }
            }
        }
        bits
    };
    let count = |bits: &[u64]| -> u32 { bits.iter().map(|v| v.count_ones()).sum() };
    let mut excluded = 0u64;
    for _ in 0..9 {
        let remove = (0..33)
            .filter(|n| excluded & (1 << n) == 0)
            .min_by_key(|&n| {
                let lost: u32 = (0..words)
                    .map(|w| (alive[w] & win[n][w]).count_ones())
                    .sum();
                (lost, 33 - n)
            })
            .unwrap();
        excluded |= 1 << remove;
        for w in 0..words {
            alive[w] &= !win[remove][w];
        }
    }
    // Improve the 24-red pool directly; stop at a one-swap local optimum.
    for _ in 0..16 {
        let current = count(&alive);
        let mut best = (current, excluded);
        for out in 0..33 {
            if excluded & (1 << out) == 0 {
                continue;
            }
            let restored = survive(excluded & !(1 << out));
            for inside in 0..33 {
                if excluded & (1 << inside) != 0 {
                    continue;
                }
                let score: u32 = (0..words)
                    .map(|w| (restored[w] & !win[inside][w]).count_ones())
                    .sum();
                let candidate = (excluded & !(1 << out)) | (1 << inside);
                if score > best.0 || (score == best.0 && score > current && candidate > best.1) {
                    best = (score, candidate);
                }
            }
        }
        if best.0 <= current {
            break;
        }
        excluded = best.1;
        alive = survive(excluded);
    }
    let outer = excluded;
    let mut removed = Vec::new();
    while excluded.count_ones() < 19 {
        let remove = (0..33)
            .filter(|n| excluded & (1 << n) == 0)
            .min_by_key(|&n| {
                let lost: u32 = (0..words)
                    .map(|w| (alive[w] & win[n][w]).count_ones())
                    .sum();
                (lost, 33 - n)
            })
            .unwrap();
        excluded |= 1 << remove;
        removed.push(remove);
        for w in 0..words {
            alive[w] &= !win[remove][w];
        }
    }
    let mut order: Vec<_> = (0..33).filter(|n| excluded & (1 << n) == 0).collect();
    order.extend(removed.into_iter().rev());
    order.extend((0..33).filter(|n| outer & (1 << n) != 0));
    order.try_into().expect("nested rank permutation")
}

impl State {
    pub fn observe(&mut self, issue: u32, mask: u64) {
        assert_eq!(mask.count_ones(), 6);
        assert_eq!(mask & !ALL, 0);
        assert!(self.history.back().is_none_or(|r| r.issue < issue));
        self.history.push_back(Observation { issue, mask });
        if self.history.len() > FIT + VALIDATE {
            self.history.pop_front();
        }
    }
    pub fn decision(&self) -> Decision {
        let original = std::array::from_fn(|n| n);
        let all: Vec<_> = self.history.iter().cloned().collect();
        if all.len() < MIN_FIT + VALIDATE {
            return Decision {
                order: original,
                candidate: original,
                accepted: false,
                evidence: json!({"reason":"insufficient_history","past_periods":all.len(),"required":MIN_FIT+VALIDATE}),
            };
        }
        let split = all.len() - VALIDATE;
        let fit = &all[..split];
        let validation = &all[split..];
        let candidate = optimize(fit);
        let old = counts(&original, validation);
        let new = counts(&candidate, validation);
        let old_blocks = [
            counts(&original, &validation[..26])[5],
            counts(&original, &validation[26..])[5],
        ];
        let new_blocks = [
            counts(&candidate, &validation[..26])[5],
            counts(&candidate, &validation[26..])[5],
        ];
        let accepted = new[5] >= old[5] + 2
            && new[..5].iter().sum::<usize>() >= old[..5].iter().sum::<usize>()
            && (0..2).all(|i| new_blocks[i] >= old_blocks[i]);
        Decision {
            order: if accepted { candidate } else { original },
            candidate,
            accepted,
            evidence: json!({"reason":if accepted {"past_validation_passed"} else {"past_validation_failed"},
            "training":{"n":fit.len(),"from":fit.first().unwrap().issue,"to":fit.last().unwrap().issue,"candidate_full_counts":counts(&candidate,fit)},
            "validation":{"n":validation.len(),"from":validation.first().unwrap().issue,"to":validation.last().unwrap().issue,
                "red_sizes":crate::pool::TIER_RED_SIZES,"baseline_full_counts":old,"candidate_full_counts":new,
                "baseline_full24_blocks":old_blocks,"candidate_full24_blocks":new_blocks},
            "fit_observations_sha256":crate::hash::sha256_hex(serde_json::to_vec(&all.iter().map(|r|(r.issue,r.mask)).collect::<Vec<_>>()).unwrap().as_slice())}),
        }
    }
}

/// Store actual winner positions only after obtaining the pre-draw decision.
pub fn rank_mask(order: &[usize; 33], winning: u64) -> u64 {
    order
        .iter()
        .enumerate()
        .filter(|(_, n)| winning & (1 << **n) != 0)
        .fold(0, |m, (i, _)| m | (1 << i))
}

pub fn replay(draws: &[Draw]) -> Result<(Value, Decision), String> {
    let snap = roll_all(draws)?;
    let (mut state, mut sums, mut n) = (State::default(), [0.0; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        if n >= 30 {
            let base = rank_desc(&fuse7(&row.expert_p, &softmax_shrink(&sums, n as f64)));
            let mask = rank_mask(&base, row.red_mask);
            let decision = state.decision();
            let original = std::array::from_fn(|i| i);
            let measures:Vec<_>=[original,decision.order,decision.candidate].iter().map(|o|{
                json!({"hits":crate::pool::TIER_RED_SIZES.map(|k|(prefix(o,k)&mask).count_ones()),
                    "minimum_full_red_prefix":o.iter().enumerate().filter(|(_,n)|mask&(1<<**n)!=0).map(|(i,_)|i+1).max().unwrap()})
            }).collect();
            rows.push(json!({"issue":row.issue,"accepted":decision.accepted,"models":measures}));
            state.observe(row.issue, mask);
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                sums[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            n += 1;
        }
    }
    fn summarize(rows: &[Value]) -> Value {
        if rows.is_empty() {
            return Value::Null;
        }
        let cells:Vec<_>=["weighted",MODEL,"tail_candidate_ungated"].into_iter().enumerate().flat_map(|(m,model)|{
            crate::pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,k)|{
                let total:u64=rows.iter().map(|r|r["models"][m]["hits"][j].as_u64().unwrap()).sum();
                let full=rows.iter().filter(|r|r["models"][m]["hits"][j]==6).count();
                json!({"model":model,"red_count":k,"n":rows.len(),"full_red_periods":full,"full_red_rate":full as f64/rows.len() as f64,
                    "mean_red_hits":total as f64/rows.len() as f64,"mean_missed_reds":6.0-total as f64/rows.len() as f64,
                    "uniform_full_red_rate":crate::pool::uniform_full_red(k)})
            })
        }).collect();
        json!({"n":rows.len(),"from":rows.first().unwrap()["issue"],"to":rows.last().unwrap()["issue"],"activated_periods":rows.iter().filter(|r|r["accepted"]==true).count(),"cells":cells})
    }
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .ok_or("缺少确认起点")?;
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    Ok((
        json!({"kind":"causal_full_six_tail_coverage_replay","actual_recommendation":false,"policy":policy(),
        "data_sha256":crate::draw::data_sha256(draws),"cutoff_issue":draws.last().unwrap().issue,
        "windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),
            "tracking":summarize(&rows[end..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),
            "recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},"latest_replay":rows.last()}),
        state.decision(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn joint_candidate_uses_separate_past_windows_and_a_complete_permutation() {
        let draws = crate::draw::synth_draws(400, 12);
        let mut state = State::default();
        for draw in &draws[..364] {
            state.observe(draw.issue, draw.red_mask);
        }
        let d = state.decision();
        assert_eq!(
            d.candidate
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            33
        );
        assert_eq!(d.evidence["training"]["n"], 312);
        assert_eq!(d.evidence["validation"]["n"], 52);
        assert!(
            d.evidence["training"]["to"].as_u64().unwrap()
                < d.evidence["validation"]["from"].as_u64().unwrap()
        );
        assert_eq!(d.evidence["validation"]["to"], draws[363].issue);
        if !d.accepted {
            assert_eq!(d.order, std::array::from_fn(|i| i));
        }
    }
}
