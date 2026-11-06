//! Frozen v1: fit convex mixtures of NORMALIZED complete outcomes.
//! The uniform prior is a likelihood penalty, not a minimum expert weight.
use crate::{
    draw::{fmt_issue, Draw},
    model::{roll_all, Row},
    protocol::Protocol,
    select::{fuse6, fuse7, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};

pub const TRAIN_END: u32 = 23100;
const SETS: f64 = 1_107_568.;
const PRIOR: f64 = 40.;
const TOL: f64 = 1e-8;

pub fn policy() -> Value {
    json!({"version":1,"training_last_issue":TRAIN_END,"warmup_rows":30,
    "objective":"sum log(sum weight * normalized_complete_outcome_probability) + 40*log(uniform_weight)",
    "red_normalization":"conditional products inside overlap strata; stratum weights fixed to fair hypergeometric(33,6,6)","overlap_probabilities":overlap_null(),
    "red_experts":["frequency","ewma","overdue","hot","cooccur","zone","repeat","uniform"],
    "blue_experts":["frequency","ewma","overdue","hot","anti_repeat","markov","uniform"],
    "uniform_prior_observations":PRIOR,"forced_expert_floor":false,"solver":"pairwise_concave_coordinate_ascent_exact_line_search",
    "normalized_dual_gap_tolerance":TOL,"forecast_kind":"probability_score_only","tickets_generated":false,
    "note":"先约束与上期六红重合数量的公平超几何分布，再混合层内完整六红分布；不生成额外票面。"})
}

pub struct SetDist {
    p: [f64; 33],
    z: f64,
    pub inclusion: [f64; 33],
    previous: Option<u64>,
    coefficients: [f64; 7],
}
impl SetDist {
    pub fn new(p: [f64; 33]) -> Self {
        assert!(p.iter().all(|v| v.is_finite() && *v > 0.));
        let mut pre = [[0.; 7]; 34];
        let mut post = pre;
        pre[0][0] = 1.;
        post[33][0] = 1.;
        for i in 0..33 {
            pre[i + 1][0] = 1.;
            for k in 1..=6 {
                pre[i + 1][k] = pre[i][k] + p[i] * pre[i][k - 1];
            }
        }
        for i in (0..33).rev() {
            post[i][0] = 1.;
            for k in 1..=6 {
                post[i][k] = post[i + 1][k] + p[i] * post[i + 1][k - 1];
            }
        }
        let z = pre[33][6];
        let inclusion = std::array::from_fn(|i| {
            p[i] * (0..6).map(|k| pre[i][k] * post[i + 1][5 - k]).sum::<f64>() / z
        });
        Self {
            p,
            z,
            inclusion,
            previous: None,
            coefficients: [0.; 7],
        }
    }
    /// Preserve the fair 6-of-33 overlap law, learning only the conditional
    /// composition within each overlap stratum. A tiny repeat expert can no
    /// longer create a large exact-previous-six spike in the complete mixture.
    pub fn stratified(p: [f64; 33], previous: u64) -> Self {
        assert_eq!(previous.count_ones(), 6);
        let mut out = Self::new(p);
        let group = |mask: u64| {
            let ids: Vec<_> = (0..33).filter(|i| mask & (1 << i) != 0).collect();
            let n = ids.len();
            let (mut pre, mut post) = (vec![[0.; 7]; n + 1], vec![[0.; 7]; n + 1]);
            pre[0][0] = 1.;
            post[n][0] = 1.;
            for i in 0..n {
                pre[i + 1][0] = 1.;
                for k in 1..=6 {
                    pre[i + 1][k] = pre[i][k] + p[ids[i]] * pre[i][k - 1];
                }
            }
            for i in (0..n).rev() {
                post[i][0] = 1.;
                for k in 1..=6 {
                    post[i][k] = post[i + 1][k] + p[ids[i]] * post[i + 1][k - 1];
                }
            }
            let mut inclusion = [[0.; 7]; 33];
            for (j, i) in ids.iter().enumerate() {
                for k in 1..=6 {
                    inclusion[*i][k] = p[*i]
                        * (0..k)
                            .map(|a| pre[j][a] * post[j + 1][k - 1 - a])
                            .sum::<f64>()
                        / pre[n][k];
                }
            }
            (pre[n], inclusion)
        };
        let (a, ma) = group(previous);
        let (b, mb) = group(((1u64 << 33) - 1) ^ previous);
        let q = overlap_null();
        out.previous = Some(previous);
        out.coefficients = std::array::from_fn(|r| q[r] / (a[r] * b[6 - r]));
        out.inclusion = std::array::from_fn(|i| {
            (0..=6)
                .map(|r| {
                    q[r] * if previous & (1 << i) != 0 {
                        ma[i][r]
                    } else {
                        mb[i][6 - r]
                    }
                })
                .sum()
        });
        out
    }
    pub fn probability(&self, mask: u64) -> f64 {
        if mask.count_ones() != 6 || mask >> 33 != 0 {
            return 0.;
        }
        let mut bits = mask;
        let mut p = self
            .previous
            .map(|previous| self.coefficients[(mask & previous).count_ones() as usize])
            .unwrap_or(1. / self.z);
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            p *= self.p[i];
            bits &= bits - 1;
        }
        p
    }
}
pub fn overlap_null() -> [f64; 7] {
    let choose = |n: usize, k: usize| (0..k).fold(1., |v, i| v * (n - i) as f64 / (i + 1) as f64);
    std::array::from_fn(|r| choose(6, r) * choose(27, 6 - r) / SETS)
}
fn dot(x: &[f64], w: &[f64]) -> f64 {
    x.iter().zip(w).map(|(x, w)| x * w).sum()
}
#[derive(Clone, Debug)]
pub struct Fit {
    pub weights: Vec<f64>,
    pub gap: f64,
    pub iterations: usize,
}
/// All x values are ratios to the uniform outcome probability.
pub fn fit(x: &[Vec<f64>]) -> Result<Fit, String> {
    let k = x.first().ok_or("校准训练数据为空")?.len();
    if k < 2
        || x.iter()
            .any(|r| r.len() != k || r.iter().any(|v| !v.is_finite() || *v <= 0.))
    {
        return Err("校准似然无效".into());
    }
    let u = k - 1;
    let n = x.len() as f64 + PRIOR;
    let mut w = vec![1. / k as f64; k];
    let mut gap = f64::INFINITY;
    for iteration in 0..20000 {
        let mix: Vec<_> = x.iter().map(|r| dot(r, &w)).collect();
        let mut g = vec![0.; k];
        for (r, p) in x.iter().zip(&mix) {
            for j in 0..k {
                g[j] += r[j] / p;
            }
        }
        g[u] += PRIOR / w[u];
        let recipient = (0..k)
            .max_by(|a, b| g[*a].total_cmp(&g[*b]).then(b.cmp(a)))
            .unwrap();
        let donor = (0..k)
            .filter(|j| w[*j] > 1e-14)
            .min_by(|a, b| g[*a].total_cmp(&g[*b]).then(a.cmp(b)))
            .unwrap();
        gap = (g[recipient] - dot(&g, &w)) / n;
        if gap <= TOL {
            return Ok(Fit {
                weights: w,
                gap,
                iterations: iteration,
            });
        }
        let sign = (if recipient == u { 1. } else { 0. }) - (if donor == u { 1. } else { 0. });
        let cap = if donor == u {
            w[donor] * (1. - 1e-12)
        } else {
            w[donor]
        };
        let delta: Vec<_> = x.iter().map(|r| r[recipient] - r[donor]).collect();
        let derivative = |a: f64| {
            delta
                .iter()
                .zip(&mix)
                .map(|(d, p)| d / (p + a * d))
                .sum::<f64>()
                + PRIOR * sign / (w[u] + a * sign)
        };
        let step = if derivative(cap) >= 0. {
            cap
        } else {
            let (mut lo, mut hi) = (0., cap);
            for _ in 0..48 {
                let m = (lo + hi) / 2.;
                if derivative(m) > 0. {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            (lo + hi) / 2.
        };
        w[recipient] += step;
        w[donor] -= step;
        if w[donor].abs() < 1e-14 {
            w[donor] = 0.;
        }
    }
    Err(format!("校准求解未收敛，标准化对偶间隙{gap}"))
}
pub struct Sample {
    pub row: Row,
    pub red: Vec<f64>,
    pub blue: Vec<f64>,
    pub original: [f64; 2],
    pub previous: u64,
}
pub struct Study {
    pub samples: Vec<Sample>,
    pub red: Fit,
    pub blue: Fit,
    pub next_red: [[f64; 33]; 7],
    pub next_blue: [[f64; 16]; 6],
    pub old_red: [f64; 33],
    pub old_blue: [f64; 16],
    pub previous: u64,
}
pub fn study(draws: &[Draw]) -> Result<Study, String> {
    let snap = roll_all(draws)?;
    let (mut rs, mut bs, mut n) = ([0.; 7], [0.; 6], 0);
    let mut samples = Vec::new();
    let previous: std::collections::BTreeMap<_, _> = draws
        .windows(2)
        .map(|w| (w[1].issue, w[0].red_mask))
        .collect();
    for row in snap.rows {
        if n >= 30 {
            let mut red: Vec<_> = row
                .expert_p
                .iter()
                .map(|p| {
                    SetDist::stratified(*p, previous[&row.issue]).probability(row.red_mask) * SETS
                })
                .collect();
            red.push(1.);
            let mut blue: Vec<_> = row
                .blue_p
                .iter()
                .map(|p| p[row.blue as usize] * 16.)
                .collect();
            blue.push(1.);
            let old_r = fuse7(&row.expert_p, &softmax_shrink(&rs, n as f64));
            let old_b = fuse6(&row.blue_p, &softmax_shrink(&bs, n as f64));
            let original = [
                (SetDist::new(old_r).probability(row.red_mask) * SETS).ln(),
                (old_b[row.blue as usize] * 16.).ln(),
            ];
            samples.push(Sample {
                row: row.clone(),
                red,
                blue,
                original,
                previous: previous[&row.issue],
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for i in 0..7 {
                rs[i] += rank_log_score(&row.expert_p[i], row.red_mask);
            }
            for i in 0..6 {
                bs[i] += row.blue_p[i][row.blue as usize].clamp(1e-12, 1.).ln();
            }
            n += 1;
        }
    }
    let xr: Vec<_> = samples
        .iter()
        .filter(|s| s.row.issue <= TRAIN_END)
        .map(|s| s.red.clone())
        .collect();
    let xb: Vec<_> = samples
        .iter()
        .filter(|s| s.row.issue <= TRAIN_END)
        .map(|s| s.blue.clone())
        .collect();
    let red = fit(&xr)?;
    let blue = fit(&xb)?;
    Ok(Study {
        samples,
        red,
        blue,
        old_red: fuse7(&snap.next_red, &softmax_shrink(&rs, n as f64)),
        old_blue: fuse6(&snap.next_blue, &softmax_shrink(&bs, n as f64)),
        next_red: snap.next_red,
        next_blue: snap.next_blue,
        previous: draws.last().ok_or("没有开奖数据")?.red_mask,
    })
}
pub fn unconditional_marginals(experts: &[[f64; 33]; 7], weights: &[f64]) -> [f64; 33] {
    let ms: Vec<_> = experts.iter().map(|p| SetDist::new(*p)).collect();
    std::array::from_fn(|i| {
        ms.iter()
            .zip(weights)
            .map(|(m, w)| m.inclusion[i] * w)
            .sum::<f64>()
            + weights[7] * 6. / 33.
    })
}
pub fn red_marginals(experts: &[[f64; 33]; 7], weights: &[f64], previous: u64) -> [f64; 33] {
    let ms: Vec<_> = experts
        .iter()
        .map(|p| SetDist::stratified(*p, previous))
        .collect();
    std::array::from_fn(|i| {
        ms.iter()
            .zip(weights)
            .map(|(m, w)| m.inclusion[i] * w)
            .sum::<f64>()
            + weights[7] * 6. / 33.
    })
}
pub fn sample_score(sample: &Sample, study: &Study) -> [f64; 2] {
    [
        dot(&sample.red, &study.red.weights).ln(),
        dot(&sample.blue, &study.blue.weights).ln(),
    ]
}
pub fn prediction(study: &Study) -> Value {
    let blue: Vec<_> = (0..16)
        .map(|b| {
            (0..6)
                .map(|e| study.blue.weights[e] * study.next_blue[e][b])
                .sum::<f64>()
                + study.blue.weights[6] / 16.
        })
        .collect();
    json!({"policy":policy(),"red_weights":study.red.weights,"blue_weights":study.blue.weights,
        "red_experts":study.next_red.iter().map(|p|p.to_vec()).collect::<Vec<_>>(),"blue_probabilities":blue,"previous_red_mask":study.previous,
        "baseline_red_scores":study.old_red.to_vec(),"baseline_blue_probabilities":study.old_blue,
        "training_n":study.samples.iter().filter(|s|s.row.issue<=TRAIN_END).count(),"training_last_issue":study.samples.iter().filter(|s|s.row.issue<=TRAIN_END).last().map(|s|fmt_issue(s.row.issue)),
        "red_solver":{"normalized_dual_gap":study.red.gap,"iterations":study.red.iterations},
        "blue_solver":{"normalized_dual_gap":study.blue.gap,"iterations":study.blue.iterations}})
}
pub fn score_prediction(p: &Value, d: &Draw) -> Result<Value, String> {
    let red: Vec<Vec<f64>> =
        serde_json::from_value(p["red_experts"].clone()).map_err(|_| "缺少红球分布")?;
    let red: Vec<[f64; 33]> = red
        .into_iter()
        .map(|r| r.try_into().map_err(|_| "红球分布长度无效"))
        .collect::<Result<_, _>>()?;
    let w: Vec<f64> =
        serde_json::from_value(p["red_weights"].clone()).map_err(|_| "红球权重无效")?;
    let blue: Vec<f64> =
        serde_json::from_value(p["blue_probabilities"].clone()).map_err(|_| "蓝球分布无效")?;
    let oldr: Vec<f64> =
        serde_json::from_value(p["baseline_red_scores"].clone()).map_err(|_| "原红球评分无效")?;
    let oldb: Vec<f64> = serde_json::from_value(p["baseline_blue_probabilities"].clone())
        .map_err(|_| "原蓝球评分无效")?;
    let simple = |a: &[f64], n: usize| {
        a.len() == n
            && a.iter().all(|v| v.is_finite() && *v >= 0.)
            && (a.iter().sum::<f64>() - 1.).abs() < 1e-9
    };
    if red.len() != 7
        || !simple(&w, 8)
        || !simple(&blue, 16)
        || !simple(&oldr, 33)
        || !simple(&oldb, 16)
        || red
            .iter()
            .any(|r| !simple(r, 33) || r.iter().any(|x| *x == 0.))
    {
        return Err("评分分布未归一化".into());
    }
    let previous = p["previous_red_mask"]
        .as_u64()
        .filter(|m| m.count_ones() == 6 && m >> 33 == 0)
        .ok_or("上期六红掩码无效")?;
    let prob = red
        .iter()
        .zip(&w)
        .map(|(p, w)| w * SetDist::stratified(*p, previous).probability(d.red_mask))
        .sum::<f64>()
        + w[7] / SETS;
    let oldr: [f64; 33] = oldr.try_into().unwrap();
    let red = (prob * SETS).ln();
    let blue = (blue[d.blue as usize - 1] * 16.).ln();
    let baseline = (SetDist::new(oldr).probability(d.red_mask) * SETS).ln()
        + (oldb[d.blue as usize - 1] * 16.).ln();
    Ok(
        json!({"red_log_excess":red,"blue_log_excess":blue,"joint_log_excess":red+blue,"baseline_joint_log_excess":baseline,"paired_joint_gain":red+blue-baseline}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solver_finds_boundary_and_reports_small_dual_gap() {
        let x = vec![vec![2., 0.5, 1.]; 200];
        let f = fit(&x).unwrap();
        // Analytic optimum: 200/(1+w0) - 40/(1-w0)=0 => w0=2/3.
        assert!((f.weights[0] - 2. / 3.).abs() < 1e-7);
        assert_eq!(f.weights[1], 0.);
        assert!(f.gap <= TOL);
        assert!((f.weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
    }
    #[test]
    fn uniformly_worse_experts_can_be_removed_entirely() {
        let f = fit(&vec![vec![0.8, 0.9, 1.]; 100]).unwrap();
        assert_eq!(f.weights, vec![0., 0., 1.]);
        assert!(fit(&[vec![0., 1.]]).is_err());
    }
    #[test]
    fn set_probabilities_and_marginals_match_uniform_and_oracle() {
        let p = [1. / 33.; 33];
        let m = SetDist::new(p);
        assert!((m.probability(0b111111) * SETS - 1.).abs() < 1e-12);
        assert!(m.inclusion.iter().all(|v| (*v - 6. / 33.).abs() < 1e-12));
        let p = std::array::from_fn(|i| (i + 1) as f64 / 561.);
        let m = SetDist::new(p);
        let (mut z, mut mass) = (0., [0.; 33]);
        for a in 0..28 {
            for b in a + 1..29 {
                for c in b + 1..30 {
                    for d in c + 1..31 {
                        for e in d + 1..32 {
                            for f in e + 1..33 {
                                let set = [a, b, c, d, e, f];
                                let v = set.iter().map(|i| p[*i]).product::<f64>();
                                z += v;
                                for i in set {
                                    mass[i] += v;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!((z - m.z).abs() < 1e-16);
        for i in 0..33 {
            assert!((mass[i] / z - m.inclusion[i]).abs() < 1e-11);
        }
    }
    #[test]
    fn future_append_cannot_retrain_and_saved_prediction_matches_causal_replay() {
        let mut d = crate::draw::synth_draws(220, 83);
        d.last_mut().unwrap().issue = 26127;
        let before = study(&d).unwrap();
        let prediction = prediction(&before);
        let next = Draw::from_parts(26128, 20261107, [2, 7, 13, 18, 24, 33], 16).unwrap();
        d.push(next.clone());
        let after = study(&d).unwrap();
        assert_eq!(before.red.weights, after.red.weights);
        assert_eq!(before.blue.weights, after.blue.weights);
        let saved = score_prediction(&prediction, &next).unwrap();
        let replay = sample_score(after.samples.last().unwrap(), &after);
        assert!((saved["red_log_excess"].as_f64().unwrap() - replay[0]).abs() < 1e-12);
        assert!((saved["blue_log_excess"].as_f64().unwrap() - replay[1]).abs() < 1e-12);
        let mut bad = prediction;
        bad["blue_probabilities"][0] = json!(0.);
        assert!(score_prediction(&bad, &next).is_err());
    }
    #[test]
    fn overlap_strata_are_normalized_and_cannot_create_a_repeat_spike() {
        let previous = 0b111111;
        let p = std::array::from_fn(|i| (i + 1) as f64 / 561.);
        let m = SetDist::stratified(p, previous);
        let q = overlap_null();
        let mut mass = [0.; 7];
        let mut marginal = [0.; 33];
        for a in 0..28 {
            for b in a + 1..29 {
                for c in b + 1..30 {
                    for d in c + 1..31 {
                        for e in d + 1..32 {
                            for f in e + 1..33 {
                                let s = [a, b, c, d, e, f];
                                let mask = s.iter().fold(0, |m, i| m | (1 << i));
                                let prob = m.probability(mask);
                                mass[(mask & previous).count_ones() as usize] += prob;
                                for i in s {
                                    marginal[i] += prob;
                                }
                            }
                        }
                    }
                }
            }
        }
        for r in 0..=6 {
            assert!((mass[r] - q[r]).abs() < 1e-11);
        }
        for i in 0..33 {
            assert!((marginal[i] - m.inclusion[i]).abs() < 1e-11);
        }
        assert!((m.probability(previous) * SETS - 1.).abs() < 1e-12);
        assert!((m.inclusion[..6].iter().sum::<f64>() - 36. / 33.).abs() < 1e-12);
        let repeat = std::array::from_fn(|i| if i < 6 { 20. / 147. } else { 1. / 147. });
        let original = SetDist::new(repeat);
        let calibrated = SetDist::stratified(repeat, previous);
        assert!(original.probability(previous) * SETS > 1000.);
        assert!((calibrated.probability(previous) * SETS - 1.).abs() < 1e-12);
    }
}
