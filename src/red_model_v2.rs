//! Frozen red v2: one coefficient per identifiable log axis, scored by the
//! exact 6-subset likelihood. Overdue and hot stay one direction. Each axis,
//! including recent-weight and repeat, is counted once. No tickets.
use crate::{
    draw::{data_sha256, Draw},
    model::roll_all,
    select::rank_desc,
};
use serde_json::{json, Value};

pub const TRAIN_END: u32 = 23100;
pub const RIDGE: f64 = 40.;
pub const UNIVERSE: f64 = 1_107_568.;
pub const AXES: [&str; 6] = ["frequency", "ewma", "log_gap", "cooccur", "zone", "repeat"];
pub type Features = [[f64; 6]; 33];
// Equivalent to an isotropic ridge on the seven original log experts after
// removing the exact overdue/hot null direction. Not a second penalty chosen
// from held-out hits.
const Q: [[f64; 6]; 6] = [
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.],
    [0., 0., 0.5, 0., 0., 0.],
    [0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 1.],
];

pub fn policy() -> Value {
    json!({"version":2,"training_last_issue":TRAIN_END,"expert_warmup_rows":30,"ridge":RIDGE,"axes":AXES,
        "objective":"sum_log_complete_six_red_probability_minus_20_times_structured_coefficient_norm_squared",
        "features":"centred_log_frequency;centred_log_ewma;centred_half_log_overdue_over_hot;centred_log_cooccur;centred_log_zone;centred_log_repeat",
        "penalty":"frequency^2+ewma^2+log_gap^2/2+cooccur^2+zone^2+repeat^2",
        "likelihood":"product_of_positive_weights_over_elementary_symmetric_polynomial_degree_6",
        "uniform_at_zero":true,"negative_coefficients_allowed":true,"forced_expert_floor":false,"per_number_coefficients":false,
        "published_fusion_formula_changed":false,"ridge_or_feature_grid_searched":false,
        "solver":"strictly_concave_newton_cholesky_armijo","normalized_gradient_tolerance":1e-10,
        "score_kind":"complete_six_probability_not_verified_winning_probability","automatic_promotion":false,"tickets_generated":false,
        "note":"遗漏与热度在对数上方向相反，只保留一个间隔系数；近期和重号各只用一次。这是完整六红似然，不是把单号分数相加，也不替换已封存融合v2。"})
}

pub fn features(p: &[[f64; 33]; 7]) -> Result<Features, String> {
    if p.iter().any(|r| {
        r.iter().any(|v| !v.is_finite() || *v <= 0.) || (r.iter().sum::<f64>() - 1.).abs() > 1e-8
    }) {
        return Err("红球专家分布无效".into());
    }
    let logs: [[f64; 33]; 7] = std::array::from_fn(|e| p[e].map(f64::ln));
    let reciprocal = logs[2][0] + logs[3][0];
    if (0..33).any(|i| (logs[2][i] + logs[3][i] - reciprocal).abs() > 1e-8) {
        return Err("红球遗漏与热度不满足冻结版本的倒数关系".into());
    }
    let mut x: Features = std::array::from_fn(|i| {
        [
            logs[0][i],
            logs[1][i],
            (logs[2][i] - logs[3][i]) / 2.,
            logs[4][i],
            logs[5][i],
            logs[6][i],
        ]
    });
    for e in 0..6 {
        let mean = x.iter().map(|r| r[e]).sum::<f64>() / 33.;
        for row in &mut x {
            row[e] -= mean;
        }
    }
    Ok(x)
}

pub fn weights(beta: &[f64; 6], x: &Features) -> Result<[f64; 33], String> {
    let z: [f64; 33] = std::array::from_fn(|i| (0..6).map(|e| beta[e] * x[i][e]).sum());
    if z.iter().any(|v| !v.is_finite()) {
        return Err("红球v2线性得分无效".into());
    }
    let peak = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w = z.map(|v| (v - peak).exp());
    if w.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err("红球v2权重无效".into());
    }
    Ok(w)
}

fn elementary(w: &[f64; 33]) -> ([[f64; 7]; 34], [[f64; 7]; 34]) {
    let mut pre = [[0.; 7]; 34];
    let mut post = [[0.; 7]; 34];
    pre[0][0] = 1.;
    for i in 0..33 {
        pre[i + 1][0] = 1.;
        for k in 1..=6 {
            pre[i + 1][k] = pre[i][k] + w[i] * pre[i][k - 1];
        }
    }
    post[33][0] = 1.;
    for i in (0..33).rev() {
        post[i][0] = 1.;
        for k in 1..=6 {
            post[i][k] = post[i + 1][k] + w[i] * post[i + 1][k - 1];
        }
    }
    (pre, post)
}

pub fn inclusion(w: &[f64; 33]) -> Result<[f64; 33], String> {
    let (pre, post) = elementary(w);
    let z = pre[33][6];
    if !z.is_finite() || z <= 0. {
        return Err("红球v2六红归一化常数无效".into());
    }
    let pi = std::array::from_fn(|j| {
        let e5: f64 = (0..=5).map(|a| pre[j][a] * post[j + 1][5 - a]).sum();
        w[j] * e5 / z
    });
    if pi.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err("红球v2入选概率无效".into());
    }
    Ok(pi)
}

pub fn log_excess(w: &[f64; 33], mask: u64) -> Result<f64, String> {
    if mask.count_ones() != 6 {
        return Err("红球v2计分需要恰好6个红球".into());
    }
    let (pre, _) = elementary(w);
    let z = pre[33][6];
    if !z.is_finite() || z <= 0. {
        return Err("红球v2六红归一化常数无效".into());
    }
    let mut bits = mask;
    let mut log_w = 0.;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        log_w += w[i].ln();
        bits &= bits - 1;
    }
    let value = log_w - z.ln() + UNIVERSE.ln();
    if !value.is_finite() {
        return Err("红球v2对数评分无效".into());
    }
    Ok(value)
}

#[derive(Clone)]
pub struct Sample {
    pub x: Features,
    pub mask: u64,
}

#[derive(Clone)]
pub struct Fit {
    pub beta: [f64; 6],
    pub iterations: usize,
    pub gradient: f64,
    pub upper_gap: f64,
    pub objective: f64,
    pub n: usize,
}

fn quadratic(b: &[f64; 6]) -> f64 {
    (0..6)
        .map(|i| (0..6).map(|j| b[i] * Q[i][j] * b[j]).sum::<f64>())
        .sum()
}

fn objective(samples: &[Sample], b: &[f64; 6]) -> Result<f64, String> {
    let mut total = 0.;
    for s in samples {
        total += log_excess(&weights(b, &s.x)?, s.mask)?;
    }
    Ok(total - RIDGE * quadratic(b) / 2.)
}

fn moments(w: &[f64; 33]) -> Result<([f64; 33], [[f64; 33]; 33], f64), String> {
    let (pre, post) = elementary(w);
    let z = pre[33][6];
    if !z.is_finite() || z <= 0. {
        return Err("红球v2六红归一化常数无效".into());
    }
    let pi: [f64; 33] = std::array::from_fn(|j| {
        w[j] * (0..=5).map(|a| pre[j][a] * post[j + 1][5 - a]).sum::<f64>() / z
    });
    let mut both = [[0.; 33]; 33];
    for j in 0..33 {
        both[j][j] = pi[j];
        let mut mid = [0.; 7];
        mid[0] = 1.;
        for m in (j + 1)..33 {
            let mut e4 = 0.;
            for a in 0..=4 {
                for b in 0..=(4 - a) {
                    e4 += pre[j][a] * mid[b] * post[m + 1][4 - a - b];
                }
            }
            let p = w[j] * w[m] * e4 / z;
            both[j][m] = p;
            both[m][j] = p;
            for k in (1..=6).rev() {
                mid[k] += w[m] * mid[k - 1];
            }
        }
    }
    Ok((pi, both, z))
}

fn derivatives(samples: &[Sample], b: &[f64; 6]) -> Result<([f64; 6], [[f64; 6]; 6]), String> {
    let mut g = std::array::from_fn(|i| -RIDGE * Q[i][i] * b[i]);
    let mut h = Q.map(|r| r.map(|v| v * RIDGE));
    for s in samples {
        let w = weights(b, &s.x)?;
        let (pi, both, _) = moments(&w)?;
        let mut mean = [0.; 6];
        for i in 0..33 {
            for a in 0..6 {
                mean[a] += pi[i] * s.x[i][a];
            }
        }
        let mut second = [[0.; 6]; 6];
        for i in 0..33 {
            for j in 0..33 {
                for a in 0..6 {
                    for c in 0..6 {
                        second[a][c] += both[i][j] * s.x[i][a] * s.x[j][c];
                    }
                }
            }
        }
        let mut bits = s.mask;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            for a in 0..6 {
                g[a] += s.x[i][a];
            }
            bits &= bits - 1;
        }
        for a in 0..6 {
            g[a] -= mean[a];
            for c in 0..=a {
                let cov = second[a][c] - mean[a] * mean[c];
                h[a][c] += cov;
                if a != c {
                    h[c][a] += cov;
                }
            }
        }
    }
    Ok((g, h))
}

fn solve(a: &[[f64; 6]; 6], g: &[f64; 6]) -> Result<[f64; 6], String> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let v = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !v.is_finite() || v <= 0. {
                    return Err("红球v2牛顿矩阵无效".into());
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    let mut y = [0.; 6];
    for i in 0..6 {
        y[i] = (g[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.; 6];
    for i in (0..6).rev() {
        x[i] = (y[i] - (i + 1..6).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    Ok(x)
}

pub fn fit(samples: &[Sample]) -> Result<Fit, String> {
    if samples.is_empty()
        || samples.iter().any(|s| {
            s.mask.count_ones() != 6 || s.x.iter().flatten().any(|v| !v.is_finite())
        })
    {
        return Err("红球v2训练样本无效".into());
    }
    let mut beta = [0.; 6];
    for iterations in 0..64 {
        let (g, h) = derivatives(samples, &beta)?;
        let gradient = g.iter().map(|v| v.abs()).fold(0., f64::max) / samples.len() as f64;
        if gradient <= 1e-10 {
            let gap = g.iter().map(|v| v * v).sum::<f64>() / RIDGE;
            return Ok(Fit {
                beta,
                iterations,
                gradient,
                upper_gap: gap,
                objective: objective(samples, &beta)?,
                n: samples.len(),
            });
        }
        let step = solve(&h, &g)?;
        let ascent = (0..6).map(|i| g[i] * step[i]).sum::<f64>();
        if !ascent.is_finite() || ascent < 0. {
            return Err("红球v2牛顿方向无效".into());
        }
        let previous = objective(samples, &beta)?;
        let mut accepted = false;
        for k in 0..40 {
            let scale = 2f64.powi(-k);
            let next = std::array::from_fn(|i| beta[i] + scale * step[i]);
            let roundoff = 32. * f64::EPSILON * (1. + previous.abs());
            if objective(samples, &next)? >= previous + 1e-4 * scale * ascent - roundoff {
                beta = next;
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err("红球v2线搜索未收敛，拒绝发布结果".into());
        }
    }
    Err("红球v2训练未达到梯度阈值".into())
}

pub fn ranking(p: &[f64; 33]) -> Vec<u8> {
    rank_desc(p).iter().map(|i| (i + 1) as u8).collect()
}

pub struct Record {
    pub issue: u32,
    pub log_excess: f64,
    pub hits: u8,
    pub full: bool,
}

pub struct Study {
    pub fit: Fit,
    pub records: Vec<Record>,
    pub latest_inclusion: Option<[f64; 33]>,
    pub latest_issue: Option<u32>,
    pub next_inclusion: [f64; 33],
    pub training_sha: String,
}

fn score_pool(beta: &[f64; 6], x: &Features, mask: u64) -> Result<(f64, u8, bool, [f64; 33]), String> {
    let w = weights(beta, x)?;
    let pi = inclusion(&w)?;
    let order = rank_desc(&pi);
    let hits = order[..16].iter().filter(|i| mask & (1 << *i) != 0).count() as u8;
    Ok((log_excess(&w, mask)?, hits, hits == 6, pi))
}

pub fn study(draws: &[Draw]) -> Result<Study, String> {
    if !draws.iter().any(|d| d.issue == TRAIN_END) {
        return Err("红球v2需要完整23100训练边界，不能用部分历史冒充冻结模型".into());
    }
    let snap = roll_all(draws)?;
    let training = snap
        .rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= TRAIN_END)
        .map(|r| Ok(Sample { x: features(&r.expert_p)?, mask: r.red_mask }))
        .collect::<Result<Vec<_>, String>>()?;
    let fit = fit(&training)?;
    let mut records = Vec::new();
    let mut latest_inclusion = None;
    let mut latest_issue = None;
    for row in &snap.rows {
        if row.issue > TRAIN_END {
            let x = features(&row.expert_p)?;
            let (log_excess, hits, full, pi) = score_pool(&fit.beta, &x, row.red_mask)?;
            latest_inclusion = Some(pi);
            latest_issue = Some(row.issue);
            records.push(Record { issue: row.issue, log_excess, hits, full });
        }
    }
    let next_w = weights(&fit.beta, &features(&snap.next_red)?)?;
    let prefix: Vec<_> = draws.iter().filter(|d| d.issue <= TRAIN_END).cloned().collect();
    Ok(Study {
        fit,
        records,
        latest_inclusion,
        latest_issue,
        next_inclusion: inclusion(&next_w)?,
        training_sha: data_sha256(&prefix),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(mask: u64, scale: f64) -> Sample {
        Sample {
            x: std::array::from_fn(|i| std::array::from_fn(|e| ((i * 3 + e * 5) % 11) as f64 * scale - 1.)),
            mask,
        }
    }

    #[test]
    fn uniform_weights_match_exact_combination_count_and_inclusion() {
        let w = [1.; 33];
        let (pre, _) = elementary(&w);
        assert!((pre[33][6] - UNIVERSE).abs() < 1e-8);
        let pi = inclusion(&w).unwrap();
        assert!((pi.iter().sum::<f64>() - 6.).abs() < 1e-8);
        assert!(pi.iter().all(|v| (v - 6. / 33.).abs() < 1e-12));
        let mask = (0..6).fold(0u64, |m, i| m | (1 << i));
        assert!(log_excess(&w, mask).unwrap().abs() < 1e-10);
    }

    #[test]
    fn unequal_weights_match_full_enumeration() {
        let mut w = [1.; 33];
        w[0] = 2.5;
        w[4] = 0.4;
        w[9] = 1.7;
        let (pre, _) = elementary(&w);
        let mut total = 0.;
        let mut hit = 0.;
        let target = (0..6).fold(0u64, |m, i| m | (1 << i));
        let mut idx = [0usize; 6];
        for i in 0..6 {
            idx[i] = i;
        }
        loop {
            let mut p = 1.;
            let mut mask = 0u64;
            for &i in &idx {
                p *= w[i];
                mask |= 1 << i;
            }
            total += p;
            if mask == target {
                hit = p;
            }
            let mut t = 5isize;
            loop {
                if t < 0 {
                    assert!((pre[33][6] - total).abs() / total < 1e-10);
                    assert!((hit / total - (log_excess(&w, target).unwrap() - UNIVERSE.ln()).exp()).abs() < 1e-10);
                    let pi = inclusion(&w).unwrap();
                    assert!((pi.iter().sum::<f64>() - 6.).abs() < 1e-7);
                    return;
                }
                idx[t as usize] += 1;
                if idx[t as usize] <= 32 - (5 - t as usize) {
                    for j in (t as usize + 1)..6 {
                        idx[j] = idx[j - 1] + 1;
                    }
                    break;
                }
                t -= 1;
            }
        }
    }

    #[test]
    fn analytic_derivatives_match_independent_finite_differences() {
        let rows = vec![sample(0b111111, 0.02), sample(0b111111 << 3, 0.03)];
        let b = [0.05, -0.04, 0.03, 0.02, -0.01, 0.04];
        let (g, h) = derivatives(&rows, &b).unwrap();
        let eps = 1e-5;
        for i in 0..6 {
            let (mut up, mut down) = (b, b);
            up[i] += eps;
            down[i] -= eps;
            let slope = (objective(&rows, &up).unwrap() - objective(&rows, &down).unwrap()) / (2. * eps);
            assert!((slope - g[i]).abs() < 2e-6, "{i} slope {slope} g {}", g[i]);
            let (gu, _) = derivatives(&rows, &up).unwrap();
            let (gd, _) = derivatives(&rows, &down).unwrap();
            for j in 0..6 {
                assert!(((gu[j] - gd[j]) / (2. * eps) + h[j][i]).abs() < 2e-5, "{i},{j}");
            }
        }
    }

    #[test]
    fn negative_coefficient_is_retained_and_training_gap_is_bounded() {
        let mut rows = Vec::new();
        for _ in 0..24 {
            let mut x = [[0.; 6]; 33];
            x[0][0] = -1.;
            for i in 1..33 {
                x[i][0] = 0.2;
            }
            rows.push(Sample { x, mask: 0b111111 });
        }
        let f = fit(&rows).unwrap();
        assert!(f.beta[0] < 0., "{:?}", f.beta);
        assert!(f.gradient <= 1e-10);
        assert!(f.upper_gap < 1e-12);
        assert!(fit(&[]).is_err());
    }

    #[test]
    fn deduplicated_gap_matches_opposite_overdue_and_hot_logs() {
        let mut state = crate::model::State::new();
        for draw in &crate::draw::synth_draws(80, 11) {
            state.update(draw);
        }
        let p = state.expert_dist();
        let x = features(&p).unwrap();
        let b = [0.2, -0.1, 0.4, 0.05, -0.2, 0.3];
        let seven = [b[0], b[1], b[2] / 2., -b[2] / 2., b[3], b[4], b[5]];
        let mut raw = [0.; 33];
        for i in 0..33 {
            raw[i] = (0..7).map(|e| seven[e] * p[e][i].ln()).sum();
            let centred = (0..6).map(|e| b[e] * x[i][e]).sum::<f64>();
            raw[i] -= centred;
        }
        let gap = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - raw.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(gap < 1e-9, "{gap}");
        let squares: f64 = seven.iter().map(|v| v * v).sum();
        assert!((quadratic(&b) - squares).abs() < 1e-12);
    }

    #[test]
    fn future_draw_does_not_change_frozen_coefficients_or_past_scores() {
        let mut draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let before = study(&draws).unwrap();
        let issue = draws.last().unwrap().issue + 1;
        draws.push(Draw::from_parts(issue, 20261112, [1, 2, 3, 4, 5, 6], 7).unwrap());
        let after = study(&draws).unwrap();
        assert_eq!(before.fit.beta, after.fit.beta);
        assert_eq!(before.training_sha, after.training_sha);
        assert_eq!(before.records.len() + 1, after.records.len());
        for (a, b) in before.records.iter().zip(&after.records) {
            assert_eq!(a.issue, b.issue);
            assert_eq!(a.log_excess, b.log_excess);
            assert_eq!(a.hits, b.hits);
        }
    }
}
