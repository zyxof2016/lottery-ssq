//! Frozen blue v2: five independent log-score axes, uniform-centred ridge fit.
//! No expert floors, number-specific coefficients or post-23100 fitting.
use crate::{
    draw::{data_sha256, Draw},
    model::roll_all,
    select::{fuse6, softmax_shrink},
};
use serde_json::{json, Value};
pub const TRAIN_END: u32 = 23100;
pub const RIDGE: f64 = 40.;
pub const AXES: [&str; 5] = [
    "frequency",
    "ewma",
    "log_gap",
    "previous_blue_correction",
    "markov",
];
pub type Features = [[f64; 5]; 16];
// Equivalent to ridge on the six original log experts after eliminating the
// exact overdue/hot null direction. This is a reparameterization, not a second
// hyperparameter candidate chosen using the held-out results.
const Q: [[f64; 5]; 5] = [
    [1., 0., 0., -1., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0.5, 0., 0.],
    [-1., 0., 0., 2., 0.],
    [0., 0., 0., 0., 1.],
];
pub fn policy() -> Value {
    json!({"version":2,"training_last_issue":TRAIN_END,"expert_warmup_rows":30,"ridge":RIDGE,"axes":AXES,
        "objective":"sum_log_normalized_blue_probability_minus_20_times_structured_coefficient_norm_squared",
        "features":"centred_log_frequency;centred_log_ewma;centred_half_log_overdue_over_hot;centred_log_anti_repeat_over_frequency;centred_log_markov",
        "penalty":"(frequency-previous_correction)^2+previous_correction^2+ewma^2+log_gap^2/2+markov^2",
        "uniform_at_zero":true,"negative_coefficients_allowed":true,"forced_expert_floor":false,"per_number_coefficients":false,
        "ridge_or_feature_grid_searched":false,"solver":"strictly_concave_newton_cholesky_armijo","normalized_gradient_tolerance":1e-10,
        "score_kind":"normalized_probability_model_not_verified_winning_probability","automatic_promotion":false,"tickets_generated":false})
}
pub fn features(p: &[[f64; 16]; 6]) -> Result<Features, String> {
    if p.iter().any(|r| {
        r.iter().any(|v| !v.is_finite() || *v <= 0.) || (r.iter().sum::<f64>() - 1.).abs() > 1e-9
    }) {
        return Err("蓝球专家分布无效".into());
    }
    let logs: [[f64; 16]; 6] = std::array::from_fn(|e| p[e].map(f64::ln));
    let reciprocal = logs[2][0] + logs[3][0];
    if (0..16).any(|b| (logs[2][b] + logs[3][b] - reciprocal).abs() > 1e-10) {
        return Err("遗漏与热度不满足冻结版本的倒数关系".into());
    }
    let mut x: Features = std::array::from_fn(|b| {
        [
            logs[0][b],
            logs[1][b],
            (logs[2][b] - logs[3][b]) / 2.,
            logs[4][b] - logs[0][b],
            logs[5][b],
        ]
    });
    for e in 0..5 {
        let mean = x.iter().map(|r| r[e]).sum::<f64>() / 16.;
        for row in &mut x {
            row[e] -= mean;
        }
    }
    Ok(x)
}
pub fn distribution(beta: &[f64; 5], x: &Features) -> [f64; 16] {
    let z: [f64; 16] = std::array::from_fn(|b| (0..5).map(|e| beta[e] * x[b][e]).sum());
    let peak = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp = z.map(|v| (v - peak).exp());
    let sum = exp.iter().sum::<f64>();
    exp.map(|v| v / sum)
}
#[derive(Clone)]
pub struct Sample {
    pub x: Features,
    pub winner: usize,
}
#[derive(Clone)]
pub struct Fit {
    pub beta: [f64; 5],
    pub iterations: usize,
    pub gradient: f64,
    pub upper_gap: f64,
    pub objective: f64,
    pub n: usize,
}
fn quadratic(b: &[f64; 5]) -> f64 {
    (0..5)
        .map(|i| (0..5).map(|j| b[i] * Q[i][j] * b[j]).sum::<f64>())
        .sum()
}
fn objective(samples: &[Sample], b: &[f64; 5]) -> f64 {
    samples
        .iter()
        .map(|s| (distribution(b, &s.x)[s.winner] * 16.).ln())
        .sum::<f64>()
        - RIDGE * quadratic(b) / 2.
}
fn derivatives(samples: &[Sample], b: &[f64; 5]) -> ([f64; 5], [[f64; 5]; 5]) {
    let mut g = std::array::from_fn(|i| -RIDGE * (0..5).map(|j| Q[i][j] * b[j]).sum::<f64>());
    let mut h = Q.map(|r| r.map(|v| v * RIDGE));
    for s in samples {
        let p = distribution(b, &s.x);
        let mean: [f64; 5] = std::array::from_fn(|e| (0..16).map(|k| p[k] * s.x[k][e]).sum());
        for i in 0..5 {
            g[i] += s.x[s.winner][i] - mean[i];
            for j in 0..=i {
                let cov = (0..16)
                    .map(|k| p[k] * (s.x[k][i] - mean[i]) * (s.x[k][j] - mean[j]))
                    .sum::<f64>();
                h[i][j] += cov;
                if i != j {
                    h[j][i] += cov;
                }
            }
        }
    }
    (g, h)
}
fn solve(a: &[[f64; 5]; 5], g: &[f64; 5]) -> Result<[f64; 5], String> {
    let mut l = [[0.; 5]; 5];
    for i in 0..5 {
        for j in 0..=i {
            let v = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !v.is_finite() || v <= 0. {
                    return Err("蓝球v2牛顿矩阵无效".into());
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    let mut y = [0.; 5];
    for i in 0..5 {
        y[i] = (g[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.; 5];
    for i in (0..5).rev() {
        x[i] = (y[i] - (i + 1..5).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    Ok(x)
}
pub fn fit(samples: &[Sample]) -> Result<Fit, String> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|s| s.winner >= 16 || s.x.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err("蓝球v2训练样本无效".into());
    }
    let mut beta = [0.; 5];
    for iterations in 0..64 {
        let (g, h) = derivatives(samples, &beta);
        let gradient = g.iter().map(|v| v.abs()).fold(0., f64::max) / samples.len() as f64;
        if gradient <= 1e-10 {
            let gap =
                g.iter().map(|v| v * v).sum::<f64>() / (2. * RIDGE * ((3. - 5f64.sqrt()) / 2.));
            return Ok(Fit {
                beta,
                iterations,
                gradient,
                upper_gap: gap,
                objective: objective(samples, &beta),
                n: samples.len(),
            });
        }
        let step = solve(&h, &g)?;
        let ascent = (0..5).map(|i| g[i] * step[i]).sum::<f64>();
        let previous = objective(samples, &beta);
        let mut accepted = false;
        for k in 0..40 {
            let scale = 2f64.powi(-k);
            let next = std::array::from_fn(|i| beta[i] + scale * step[i]);
            let roundoff = 32. * f64::EPSILON * (1. + previous.abs());
            if objective(samples, &next) >= previous + 1e-4 * scale * ascent - roundoff {
                beta = next;
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err("蓝球v2线搜索未收敛，拒绝发布结果".into());
        }
    }
    Err("蓝球v2训练未达到梯度阈值".into())
}
pub struct Record {
    pub issue: u32,
    pub winner: usize,
    pub old: [f64; 16],
    pub new: [f64; 16],
    pub x: Features,
}
pub struct Study {
    pub fit: Fit,
    pub records: Vec<Record>,
    pub next: [f64; 16],
    pub old_next: [f64; 16],
    pub training_sha: String,
}
pub fn study(draws: &[Draw]) -> Result<Study, String> {
    if !draws.iter().any(|d| d.issue == TRAIN_END) {
        return Err("蓝球v2需要完整23100训练边界，不能用部分历史冒充冻结模型".into());
    }
    let snap = roll_all(draws)?;
    let training: Vec<_> = snap
        .rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= TRAIN_END)
        .map(|r| {
            Ok(Sample {
                x: features(&r.blue_p)?,
                winner: r.blue as usize,
            })
        })
        .collect::<Result<_, String>>()?;
    let fit = fit(&training)?;
    let (mut sums, mut n) = ([0.; 6], 0usize);
    let mut records = Vec::new();
    for row in &snap.rows {
        if row.issue > TRAIN_END {
            let x = features(&row.blue_p)?;
            records.push(Record {
                issue: row.issue,
                winner: row.blue as usize,
                old: fuse6(&row.blue_p, &softmax_shrink(&sums, n as f64)),
                new: distribution(&fit.beta, &x),
                x,
            });
        }
        if row.issue < crate::protocol::Protocol::CONFIRM_START_ISSUE {
            for e in 0..6 {
                sums[e] += row.blue_p[e][row.blue as usize].ln();
            }
            n += 1;
        }
    }
    let next = distribution(&fit.beta, &features(&snap.next_blue)?);
    let old_next = fuse6(&snap.next_blue, &softmax_shrink(&sums, n as f64));
    let prefix: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= TRAIN_END)
        .cloned()
        .collect();
    Ok(Study {
        fit,
        records,
        next,
        old_next,
        training_sha: data_sha256(&prefix),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn x() -> Features {
        std::array::from_fn(|b| std::array::from_fn(|e| ((b * (e + 2) + e) % 17) as f64 / 8. - 1.))
    }
    #[test]
    fn normalized_uniform_and_number_permutation_equivariance() {
        let a = x();
        assert_eq!(distribution(&[0.; 5], &a), [1. / 16.; 16]);
        let b = [0.1, -0.2, 0.3, 0.4, -0.1];
        let p = distribution(&b, &a);
        assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-14);
        let reversed = std::array::from_fn(|i| a[15 - i]);
        let q = distribution(&b, &reversed);
        for i in 0..16 {
            assert!((p[i] - q[15 - i]).abs() < 1e-15);
        }
    }
    #[test]
    fn analytic_derivatives_match_independent_finite_differences() {
        let rows = vec![Sample { x: x(), winner: 3 }, Sample { x: x(), winner: 8 }];
        let b = [0.1, -0.1, 0.05, 0.2, -0.3];
        let (g, h) = derivatives(&rows, &b);
        let eps = 1e-5;
        for i in 0..5 {
            let mut a = b;
            let mut c = b;
            a[i] += eps;
            c[i] -= eps;
            assert!(
                ((objective(&rows, &a) - objective(&rows, &c)) / (2. * eps) - g[i]).abs() < 1e-7
            );
            let (ga, _) = derivatives(&rows, &a);
            let (gc, _) = derivatives(&rows, &c);
            for j in 0..5 {
                assert!(((ga[j] - gc[j]) / (2. * eps) + h[j][i]).abs() < 1e-7);
            }
        }
    }
    #[test]
    fn negative_evidence_is_not_forced_positive_and_fit_reports_gap() {
        let mut rows = Vec::new();
        for _ in 0..40 {
            let mut a = [[0.; 5]; 16];
            a[0][0] = -1.;
            a[1][0] = 1.;
            rows.push(Sample { x: a, winner: 0 });
        }
        let f = fit(&rows).unwrap();
        assert!(f.beta[0] < 0.);
        assert!(f.gradient <= 1e-10);
        assert!(f.upper_gap < 1e-10);
        assert!(f.objective >= 0.);
        assert!(fit(&[]).is_err());
    }
    #[test]
    fn deduplication_matches_original_six_log_experts_with_identical_penalty() {
        let mut state = crate::model::State::new();
        let d = crate::draw::synth_draws(200, 7);
        for r in &d {
            state.update(r);
        }
        let p = state.blue_dist();
        let a = features(&p).unwrap();
        let b = [0.1, -0.2, 0.3, -0.1, 0.2];
        let six = [b[0] - b[3], b[1], b[2] / 2., -b[2] / 2., b[3], b[4]];
        let z: [f64; 16] = std::array::from_fn(|i| (0..6).map(|e| six[e] * p[e][i].ln()).sum());
        let peak = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp = z.map(|v| (v - peak).exp());
        let total = exp.iter().sum::<f64>();
        let predicted = distribution(&b, &a);
        for i in 0..16 {
            assert!((predicted[i] - exp[i] / total).abs() < 1e-14);
        }
        assert!((quadratic(&b) - six.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-14);
    }
    #[test]
    fn future_outcome_cannot_change_frozen_coefficients_or_previous_scores() {
        let mut d = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let before = study(&d).unwrap();
        let issue = d.last().unwrap().issue + 1;
        d.push(Draw::from_parts(issue, 20261112, [1, 2, 3, 4, 5, 6], 1).unwrap());
        let after = study(&d).unwrap();
        assert_eq!(before.fit.beta, after.fit.beta);
        assert_eq!(before.training_sha, after.training_sha);
        assert_eq!(before.next, after.records.last().unwrap().new);
        for (a, b) in before.records.iter().zip(&after.records) {
            assert_eq!(a.new, b.new);
        }
    }
}
