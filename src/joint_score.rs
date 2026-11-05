//! Versioned research scorer: normalized six-red chains and conditional blues.
//! No ordinal labels enter training, no exposure/random grouping, no publication.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    model::roll_all,
    protocol::Protocol,
    select::{fuse6, fuse7, rank_desc, softmax_shrink},
};
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};

const SETS: f64 = 1_107_568.;
const TRAIN_END: u32 = 23100;
const PRIOR: f64 = 200.;
const LEVELS: [f64; 4] = [0., 0.25, 0.5, 1.];

fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.;
    }
    (0..k).fold(1., |v, i| v * (n - i) as f64 / (i + 1) as f64)
}

/// Exact partition and marginals over strictly increasing length-k paths.
/// Edges connect adjacent SELECTED numbers, not necessarily consecutive labels.
struct Chain {
    node: Vec<f64>,
    edge: Vec<Vec<f64>>,
    k: usize,
    z: f64,
    marginal: Vec<f64>,
}
impl Chain {
    fn new(node: Vec<f64>, edge: Vec<Vec<f64>>, k: usize) -> Self {
        let n = node.len();
        let mut f = vec![vec![0.; n]; k + 1];
        f[1].clone_from(&node);
        for l in 2..=k {
            for j in 0..n {
                f[l][j] = node[j] * (0..j).map(|i| f[l - 1][i] * edge[i][j]).sum::<f64>();
            }
        }
        let z = f[k].iter().sum::<f64>();
        let mut tail = vec![vec![1.; n]; k];
        for l in 1..k {
            for i in 0..n {
                tail[l][i] = (i + 1..n)
                    .map(|j| edge[i][j] * node[j] * tail[l - 1][j])
                    .sum();
            }
        }
        let marginal = (0..n)
            .map(|j| (1..=k).map(|l| f[l][j] * tail[k - l][j]).sum::<f64>() / z)
            .collect();
        Self {
            node,
            edge,
            k,
            z,
            marginal,
        }
    }
    fn logp(&self, s: &[usize]) -> f64 {
        assert_eq!(s.len(), self.k);
        s.iter().map(|i| self.node[*i].ln()).sum::<f64>()
            + s.windows(2)
                .map(|w| self.edge[w[0]][w[1]].ln())
                .sum::<f64>()
            - self.z.ln()
    }
}

#[derive(Clone)]
struct Counts {
    n: usize,
    red: [usize; 33],
    adj: [[usize; 33]; 33],
    blue: [usize; 16],
    rb: [[usize; 16]; 33],
}
impl Default for Counts {
    fn default() -> Self {
        Self {
            n: 0,
            red: [0; 33],
            adj: [[0; 33]; 33],
            blue: [0; 16],
            rb: [[0; 16]; 33],
        }
    }
}
impl Counts {
    fn update(&mut self, d: &Draw) {
        self.n += 1;
        let s: Vec<_> = d.red.iter().map(|r| *r as usize - 1).collect();
        let b = d.blue as usize - 1;
        self.blue[b] += 1;
        for i in &s {
            self.red[*i] += 1;
            self.rb[*i][b] += 1;
        }
        for w in s.windows(2) {
            self.adj[w[0]][w[1]] += 1;
        }
    }
    fn features(&self) -> Features {
        let node = std::array::from_fn(|i| {
            ((self.red[i] as f64 + PRIOR * 6. / 33.) / ((self.n as f64 + PRIOR) * 6. / 33.)).ln()
        });
        let edge = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if j <= i {
                    return 0.;
                }
                // To be adjacent in sorted S, choose the other four outside [i,j].
                let p = choose(32 - (j - i), 4) / SETS;
                if p == 0. {
                    0.
                } else {
                    ((self.adj[i][j] as f64 + PRIOR * p) / ((self.n as f64 + PRIOR) * p)).ln()
                }
            })
        });
        let base: [f64; 16] =
            std::array::from_fn(|b| (self.blue[b] as f64 + PRIOR / 16.) / (self.n as f64 + PRIOR));
        let cond = std::array::from_fn(|r| {
            std::array::from_fn(|b| {
                // E[blue|red r], shrunk towards the current causal blue base.
                ((self.rb[r][b] as f64 + 64. * base[b]) / (self.red[r] as f64 + 64.) / base[b]).ln()
            })
        });
        Features {
            node,
            edge,
            base: base.map(|p| (16. * p).ln()),
            cond,
        }
    }
}
struct Features {
    node: [f64; 33],
    edge: [[f64; 33]; 33],
    base: [f64; 16],
    cond: [[f64; 16]; 33],
}
fn parameters(i: usize) -> (f64, f64) {
    (LEVELS[i / 4], LEVELS[i % 4])
}
impl Features {
    fn chain(&self, p: usize) -> Chain {
        let (a, b) = parameters(p);
        Chain::new(
            self.node.iter().map(|x| (a * x).exp()).collect(),
            self.edge
                .iter()
                .map(|r| r.iter().map(|x| (b * x).exp()).collect())
                .collect(),
            6,
        )
    }
    fn blue(&self, s: &[usize], p: usize) -> [f64; 16] {
        let (a, b) = parameters(p);
        let score: [f64; 16] = std::array::from_fn(|j| {
            a * self.base[j] + b * s.iter().map(|i| self.cond[*i][j]).sum::<f64>()
        });
        let max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut out = score.map(|x| (x - max).exp());
        let z = out.iter().sum::<f64>();
        for x in &mut out {
            *x /= z;
        }
        out
    }
}

/// Midranks remove the old numeric-label tie advantage. Used as an ablation,
/// not as the proper likelihood objective of the interaction model.
fn midrank_score(p: &[f64; 33], mask: u64) -> f64 {
    let keys = p.map(|x| (x * 1e9).round() as i64);
    (0..33)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| {
            let above = keys.iter().filter(|v| **v > keys[i]).count();
            let tied = keys.iter().filter(|v| **v == keys[i]).count();
            -(above as f64 + (tied as f64 + 1.) / 2.).ln()
        })
        .sum()
}
fn set_logp(p: &[f64; 33], s: &[usize]) -> f64 {
    let mut dp = [0.; 7];
    dp[0] = 1.;
    for v in p {
        for k in (1..=6).rev() {
            dp[k] += v * dp[k - 1];
        }
    }
    s.iter().map(|i| p[*i].ln()).sum::<f64>() - dp[6].ln()
}

#[derive(Debug, PartialEq)]
struct Row {
    issue: u32,
    red: [f64; 16],
    blue: [f64; 16],
    old: [f64; 2],
    mid: [f64; 2],
    pool_hit: [bool; 3],
    true_set_rank: Option<(usize, usize)>,
}
struct Replay {
    rows: Vec<Row>,
    counts: Counts,
    red_choice: usize,
    blue_choice: usize,
    train_n: usize,
    first: Option<u32>,
    last: Option<u32>,
    mid_weights: [f64; 7],
}
fn best(sum: &[f64; 16]) -> usize {
    (1..16).fold(0, |b, i| if sum[i] > sum[b] + 1e-12 { i } else { b })
}
fn replay(draws: &[Draw]) -> Result<Replay, String> {
    let snap = roll_all(draws)?;
    let mut counts = Counts::default();
    let (mut old_r, mut old_b, mut mid_r) = ([0.; 7], [0.; 6], [0.; 7]);
    let (mut original_n, mut mid_n, mut train_n) = (0, 0, 0);
    let (mut rs, mut bs) = ([0.; 16], [0.; 16]);
    let (mut first, mut last) = (None, None);
    let mut out = Vec::new();
    let mut cursor = 0;
    for row in snap.rows {
        while draws[cursor].issue < row.issue {
            counts.update(&draws[cursor]);
            cursor += 1;
        }
        let d = &draws[cursor];
        let s: Vec<_> = d.red.iter().map(|r| *r as usize - 1).collect();
        let b = d.blue as usize - 1;
        let feature = counts.features();
        if original_n >= 30 {
            let red = std::array::from_fn(|i| feature.chain(i).logp(&s) + SETS.ln());
            let blue = std::array::from_fn(|i| (16. * feature.blue(&s, i)[b]).ln());
            let oldp = fuse7(&row.expert_p, &softmax_shrink(&old_r, original_n as f64));
            let oldblue = fuse6(&row.blue_p, &softmax_shrink(&old_b, original_n as f64));
            let midp = fuse7(&row.expert_p, &softmax_shrink(&mid_r, mid_n as f64));
            let old = [set_logp(&oldp, &s) + SETS.ln(), (16. * oldblue[b]).ln()];
            let mid = [set_logp(&midp, &s) + SETS.ln(), old[1]];
            let pool_hit = |pool: &[usize]| s.iter().all(|i| pool.contains(i));
            let mut coverage = [
                pool_hit(&rank_desc(&oldp)[..16]),
                pool_hit(&rank_desc(&midp)[..16]),
                false,
            ];
            let mut true_set_rank = None;
            if row.issue > TRAIN_END {
                let chain = feature.chain(best(&rs));
                let order = rank_desc(&std::array::from_fn::<_, 33, _>(|i| chain.marginal[i]));
                coverage[2] = pool_hit(&order[..16]);
                if coverage[2] {
                    let mut pool = order[..16].to_vec();
                    pool.sort_unstable();
                    let key = |v: f64| (v * 1e12).round() as i64;
                    let target = key(chain.logp(&s));
                    let (mut greater, mut equal) = (0, 0);
                    combinations(&pool, 6, &mut Vec::new(), &mut |t| {
                        let score = key(chain.logp(t));
                        greater += usize::from(score > target);
                        equal += usize::from(score == target);
                    });
                    true_set_rank = Some((greater + 1, greater + equal));
                }
            }
            if row.issue <= TRAIN_END {
                for i in 0..16 {
                    rs[i] += red[i];
                    bs[i] += blue[i];
                }
                train_n += 1;
                first.get_or_insert(row.issue);
                last = Some(row.issue);
            }
            out.push(Row {
                issue: row.issue,
                red,
                blue,
                old,
                mid,
                pool_hit: coverage,
                true_set_rank,
            });
        }
        // The historical baseline must reproduce its original freeze boundary.
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for i in 0..7 {
                old_r[i] += crate::select::rank_log_score(&row.expert_p[i], row.red_mask);
            }
            for i in 0..6 {
                old_b[i] += row.blue_p[i][b].clamp(1e-12, 1.).ln();
            }
            original_n += 1;
        }
        // Isolate ties: this ablation keeps the original fitting boundary.
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for i in 0..7 {
                mid_r[i] += midrank_score(&row.expert_p[i], row.red_mask);
            }
            mid_n += 1;
        }
    }
    while cursor < draws.len() {
        counts.update(&draws[cursor]);
        cursor += 1;
    }
    if train_n == 0 {
        return Err("完整组合优化的训练历史不足".into());
    }
    Ok(Replay {
        rows: out,
        counts,
        red_choice: best(&rs),
        blue_choice: best(&bs),
        train_n,
        first,
        last,
        mid_weights: softmax_shrink(&mid_r, mid_n as f64),
    })
}
fn summarize(rows: &[&Row], r: usize, b: usize) -> Value {
    let metric = |which: usize| {
        let v: Vec<_> = rows
            .iter()
            .map(|row| match which {
                0 => row.old,
                1 => row.mid,
                _ => [row.red[r], row.blue[b]],
            })
            .collect();
        let n = v.len();
        let red = (n > 0).then(|| v.iter().map(|p| p[0]).sum::<f64>() / n as f64);
        let blue = (n > 0).then(|| v.iter().map(|p| p[1]).sum::<f64>() / n as f64);
        let joint = red.zip(blue).map(|(r, b)| r + b);
        let se = (n > 1).then(|| {
            let mean = joint.unwrap();
            (v.iter().map(|p| (p[0] + p[1] - mean).powi(2)).sum::<f64>() / (n * (n - 1)) as f64)
                .sqrt()
        });
        json!({"n":n,"mean_red_log_excess":red,"mean_blue_log_excess":blue,"mean_joint_log_excess":joint,
            "joint_standard_error":se,"joint_95_lower_descriptive":joint.zip(se).map(|(m,s)| m-1.96*s),
            "full_six_in_16":if which==2 && rows.iter().any(|r|r.issue<=TRAIN_END){None}else{Some(rows.iter().filter(|r|r.pool_hit[which]).count())},
            "true_six_unambiguously_in_top8":if which==2 && rows.iter().all(|r|r.issue>TRAIN_END){Some(rows.iter().filter(|r|r.true_set_rank.is_some_and(|(_,hi)|hi<=8)).count())}else{None}})
    };
    json!({"first_issue":rows.first().map(|r|fmt_issue(r.issue)),"last_issue":rows.last().map(|r|fmt_issue(r.issue)),
        "original":metric(0),"midrank_ablation":metric(1),"interaction":metric(2),
        "uniform":{"n":rows.len(),"mean_red_log_excess":0.,"mean_blue_log_excess":0.,"mean_joint_log_excess":0.}})
}

/// Exhaustive whole-ticket ordering within <=16 reds, including conditional
/// blue normalization for EACH six-red set. The independent-product frontier
/// is invalid for interactions; this enumerator deliberately does not use it.
fn top_tickets(
    feature: &Features,
    chain: &Chain,
    blue: usize,
    pool: &[usize],
    count: usize,
) -> Vec<Value> {
    let mut sorted = pool.to_vec();
    sorted.sort_unstable();
    let mut best: Vec<(f64, Vec<usize>, usize)> = Vec::new();
    combinations(&sorted, 6, &mut Vec::new(), &mut |s| {
        let lp = chain.logp(s);
        for (b, p) in feature.blue(s, blue).iter().enumerate() {
            let score = lp + p.ln();
            if best.len() < count || score > best.last().unwrap().0 {
                best.push((score, s.to_vec(), b));
                best.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                best.truncate(count);
            }
        }
    });
    best.iter().map(|(p,s,b)|json!({"red":s.iter().map(|i|i+1).collect::<Vec<_>>(),"blue":b+1,"log_probability":p})).collect()
}
fn combinations(pool: &[usize], k: usize, s: &mut Vec<usize>, f: &mut impl FnMut(&[usize])) {
    if k == 0 {
        f(s);
        return;
    }
    if pool.len() < k {
        return;
    }
    for i in 0..=pool.len() - k {
        s.push(pool[i]);
        combinations(&pool[i + 1..], k - 1, s, f);
        s.pop();
    }
}

fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let fit = replay(draws)?;
    let (r, b) = (fit.red_choice, fit.blue_choice);
    let parameter_audit: Vec<_> = (0..16)
        .map(|i| {
            let mean = |training: bool, blue: bool| {
                let rows: Vec<_> = fit
                    .rows
                    .iter()
                    .filter(|v| {
                        if training {
                            v.issue <= TRAIN_END
                        } else {
                            v.issue > TRAIN_END && v.issue < Protocol::CONFIRM_START_ISSUE
                        }
                    })
                    .collect();
                (!rows.is_empty()).then(|| {
                    rows.iter()
                        .map(|v| if blue { v.blue[i] } else { v.red[i] })
                        .sum::<f64>()
                        / rows.len() as f64
                })
            };
            let (a, b) = parameters(i);
            json!({"index":i,"node_or_base_coefficient":a,"pair_or_conditional_coefficient":b,
            "training_red_excess":mean(true,false),"training_blue_excess":mean(true,true),
            "validation_red_excess":mean(false,false),"validation_blue_excess":mean(false,true)})
        })
        .collect();
    let start = fit
        .rows
        .iter()
        .position(|v| v.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(fit.rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(fit.rows.len());
    let confirmation_end = fit
        .rows
        .get(end.wrapping_sub(1))
        .map(|v| v.issue)
        .unwrap_or(0);
    let windows: serde_json::Map<String, Value> = [
        (
            "training",
            fit.rows
                .iter()
                .filter(|v| v.issue <= TRAIN_END)
                .collect::<Vec<_>>(),
        ),
        (
            "validation",
            fit.rows
                .iter()
                .filter(|v| v.issue > TRAIN_END && v.issue < Protocol::CONFIRM_START_ISSUE)
                .collect(),
        ),
        (
            "confirmation_exploratory",
            fit.rows
                .iter()
                .filter(|v| v.issue >= Protocol::CONFIRM_START_ISSUE && v.issue <= confirmation_end)
                .collect(),
        ),
        (
            "tracking",
            fit.rows
                .iter()
                .filter(|v| v.issue > confirmation_end)
                .collect(),
        ),
    ]
    .into_iter()
    .map(|(name, rows)| (name.into(), summarize(&rows, r, b)))
    .collect();
    let validation = &windows["validation"]["interaction"];
    let passed = validation["n"].as_u64().is_some_and(|n| n >= 156)
        && validation["joint_95_lower_descriptive"]
            .as_f64()
            .is_some_and(|v| v > 0.);
    let features = fit.counts.features();
    let chain = features.chain(r);
    let order: [usize; 33] = rank_desc(&std::array::from_fn(|i| chain.marginal[i]));
    // A flat fitted red model has no favored pool. Never present numeric-label
    // tie breaks as learned recommendations, even if blue scores differ.
    let identifiable = chain
        .marginal
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - chain.marginal.iter().copied().fold(f64::INFINITY, f64::min)
        > 1e-12;
    let candidate = if identifiable && passed {
        json!({"candidate_reds":order[..16].iter().map(|i|i+1).collect::<Vec<_>>(),
            "tickets":top_tickets(&features,&chain,b,&order[..16],8),"cost_yuan":16,"status":"research_preview"})
    } else {
        Value::Null
    };
    let (rn, re) = parameters(r);
    let (bn, bc) = parameters(b);
    Ok(
        json!({"version":2,"kind":"normalized_interaction_score","data_sha256":data_sha256(draws),
        "cutoff_issue":draws.last().map(|d|fmt_issue(d.issue)),
        "recipe":{"red":"P(S) = product node(S) * product adjacent_selected_pair(S) / exact_partition",
            "blue":"P(b|S) = softmax(base(b) + sum conditional_red_blue(S,b))",
            "joint":"P(S,b)=P(S)*P(b|S)","red_parameter_grid":LEVELS,"blue_parameter_grid":LEVELS,
            "prior_draws":PRIOR,"conditional_prior_red_occurrences":64,"training_objective":"complete_outcome_log_likelihood",
            "red_universe":1_107_568,"joint_universe":17_721_088,"pool_limit":16,"ticket_budget":8,
            "selection":"enumerate all C(16,6)*16 complete tickets; no exposure penalty or random grouping"},
        "training":{"n":fit.train_n,"first_issue":fit.first.map(fmt_issue),"last_issue":fit.last.map(fmt_issue),
            "red_node_coefficient":rn,"red_pair_coefficient":re,"blue_base_coefficient":bn,"blue_red_condition_coefficient":bc,
            "midrank_ablation_weights":fit.mid_weights,"selection_uses_validation":false,"midrank_ablation_freeze_before":Protocol::CONFIRM_START_ISSUE},
        "windows":windows,"parameter_audit":parameter_audit,"candidate":candidate,
        "full_pool_cases":fit.rows.iter().filter_map(|v|v.true_set_rank.map(|(lo,hi)|json!({"issue":fmt_issue(v.issue),"true_six_rank_lower":lo,"true_six_rank_upper":hi,"six_red_sets":8008}))).collect::<Vec<_>>(),
        "adoption":{"validation_passed":passed,"red_identifiable":identifiable,"promoted":false,"new_prospective_n":0,
            "publication_enabled":false,"reason":if !identifiable {"训练选择均匀红球模型，没有可识别的优选六红；不以号码大小破同分制造推荐。"}
            else if !passed {"预留验证未证明完整6红1蓝评分优于均匀，阻止候选进入推荐；原档保持首次票面。"}
            else {"历史验证通过但属于探索复用，仍缺独立前瞻证据，不自动替换原档。"}},
        "note":"参数只由截至23100期训练窗决定；23101–25110仅为v2预留验证；旧规则及并列修正对照仍按原协议训练至25110期。特征每期仅用更早开奖更新；确认窗和后续跟踪均非新前瞻证明。95%下界为描述统计，非独立同分布检验或中奖概率。",
        "source_sha256":sha256_hex(concat!(include_str!("joint_score.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("protocol.rs")).as_bytes())}),
    )
}

pub fn report(draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "组合评分缓存锁异常")?;
    let sha = data_sha256(draws);
    if let Some((key, value)) = cache.as_ref() {
        if *key == sha {
            return Ok(value.clone());
        }
    }
    let value = analyze(draws)?;
    *cache = Some((sha, value.clone()));
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_partition_marginals_match_independent_enumeration() {
        let n = 8;
        let node: Vec<_> = (0..n).map(|i| 1. + i as f64 / 10.).collect();
        let edge: Vec<Vec<_>> = (0..n)
            .map(|i| (0..n).map(|j| 1. + ((i + j) % 3) as f64 / 5.).collect())
            .collect();
        let chain = Chain::new(node.clone(), edge.clone(), 3);
        let (mut z, mut marg) = (0., vec![0.; n]);
        for a in 0..n {
            for b in a + 1..n {
                for c in b + 1..n {
                    let p = node[a] * node[b] * node[c] * edge[a][b] * edge[b][c];
                    z += p;
                    for i in [a, b, c] {
                        marg[i] += p;
                    }
                }
            }
        }
        assert!((z - chain.z).abs() < 1e-10);
        for i in 0..n {
            assert!((marg[i] / z - chain.marginal[i]).abs() < 1e-12);
        }
        let mut total = 0.;
        combinations(&(0..n).collect::<Vec<_>>(), 3, &mut Vec::new(), &mut |s| {
            total += chain.logp(s).exp()
        });
        assert!((total - 1.).abs() < 1e-12);
    }
    #[test]
    fn uniform_chain_and_adjacency_null_are_exact() {
        let f = Counts::default().features();
        let c = f.chain(15);
        assert!((c.z - SETS).abs() < 1e-8);
        assert!(c.marginal.iter().all(|p| (*p - 6. / 33.).abs() < 1e-12));
        let expected = (0..33)
            .flat_map(|i| (i + 1..33).map(move |j| choose(32 - (j - i), 4) / SETS))
            .sum::<f64>();
        assert!((expected - 5.).abs() < 1e-12);
    }
    #[test]
    fn midrank_ties_do_not_depend_on_numeric_labels() {
        let mut p = [0.01; 33];
        p[0] = 0.1;
        p[8] = 0.1;
        assert_eq!(
            midrank_score(&p, 0b111111),
            midrank_score(
                &p,
                (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12) | (1 << 13)
            )
        );
    }
    #[test]
    fn blue_is_conditional_on_whole_red_set_and_normalized() {
        let mut c = Counts::default();
        for i in 0..100 {
            c.update(
                &Draw::from_parts(
                    i,
                    20200101,
                    if i % 2 == 0 {
                        [1, 2, 3, 4, 5, 6]
                    } else {
                        [7, 8, 9, 10, 11, 12]
                    },
                    if i % 2 == 0 { 1 } else { 2 },
                )
                .unwrap(),
            );
        }
        let f = c.features();
        let a = f.blue(&[0, 1, 2, 3, 4, 5], 15);
        let b = f.blue(&[6, 7, 8, 9, 10, 11], 15);
        assert!((a.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert!(a[0] > b[0]);
        assert!(b[1] > a[1]);
        // Pair interaction changes relative scores even with equal node sums.
        let mut f = Counts::default().features();
        f.edge[0][1] = 1.;
        let chain = f.chain(3);
        assert!(chain.logp(&[0, 1, 3, 4, 5, 6]) > chain.logp(&[0, 2, 3, 4, 5, 6]));
    }
    #[test]
    fn joint_enumerator_matches_full_oracle_and_has_no_split_blue_pairing() {
        let mut c = Counts::default();
        for i in 0..30 {
            c.update(
                &Draw::from_parts(i, 20200101, [1, 2, 4, 6, 7, 8], (i % 16 + 1) as u8).unwrap(),
            );
        }
        let f = c.features();
        let chain = f.chain(7);
        let pool: Vec<_> = (0..8).collect();
        let got = top_tickets(&f, &chain, 15, &pool, 8);
        let mut oracle = Vec::new();
        for a in 0..8 {
            for b in a + 1..8 {
                for cc in b + 1..8 {
                    for d in cc + 1..8 {
                        for e in d + 1..8 {
                            for ff in e + 1..8 {
                                let s = vec![a, b, cc, d, e, ff];
                                for (blue, p) in f.blue(&s, 15).iter().enumerate() {
                                    oracle.push((chain.logp(&s) + p.ln(), s.clone(), blue));
                                }
                            }
                        }
                    }
                }
            }
        }
        oracle.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for i in 0..8 {
            assert_eq!(
                got[i]["red"],
                json!(oracle[i].1.iter().map(|r| r + 1).collect::<Vec<_>>())
            );
            assert_eq!(got[i]["blue"], oracle[i].2 + 1);
        }
    }
    #[test]
    fn future_draws_cannot_retrain_parameters_or_change_prefix_scores() {
        let mut draws: Vec<_> = (0..330)
            .map(|i| {
                Draw::from_parts(
                    22000 + i,
                    20200101,
                    std::array::from_fn(|j| ((i as usize + j * 5) % 33 + 1) as u8),
                    (i % 16 + 1) as u8,
                )
                .unwrap()
            })
            .collect();
        let before = replay(&draws).unwrap();
        for i in 0..20 {
            draws.push(Draw::from_parts(23101 + i, 20210101, [1, 2, 3, 4, 5, 6], 1).unwrap());
        }
        let after = replay(&draws).unwrap();
        assert_eq!(before.red_choice, after.red_choice);
        assert_eq!(before.blue_choice, after.blue_choice);
        assert_eq!(before.train_n, after.train_n);
        assert_eq!(before.last, after.last);
        assert_eq!(&before.rows, &after.rows[..before.rows.len()]);
    }
}
