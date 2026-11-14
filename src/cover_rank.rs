//! Causal red ranking trained to pull the last winning number forward.
//! The score is a linear signal, not a winning probability, and it does not
//! replace a sealed ticket.
use crate::model::Row;
use crate::red_model_v2::{features, Sample, AXES, TRAIN_END};

pub const RIDGE: f64 = 0.05;
pub const WARMUP_ROWS: usize = 30;
pub const REQUIRED_MEAN_GAIN: f64 = 1.0;
const LOSERS: f64 = 27.;

#[derive(Clone, Copy)]
pub struct Fit {
    pub beta: [f64; 6],
    pub iterations: usize,
    pub gradient: f64,
    pub objective: f64,
    pub n: usize,
}

pub fn policy() -> serde_json::Value {
    serde_json::json!({
        "version": 1,
        "name": "cover_rank_v1",
        "training_last_issue": TRAIN_END,
        "expert_warmup_rows": WARMUP_ROWS,
        "l2_penalty": RIDGE,
        "axes": AXES,
        "objective": "mean_logsumexp_of_per_winner_mean_softplus",
        "features": "red_score_v2_centred_log_axes",
        "parameter_search": false,
        "required_mean_gain_vs_weighted_and_uniform": REQUIRED_MEAN_GAIN,
        "score_kind": "cover_rank_signal_not_probability",
        "automatic_promotion": false,
        "tickets_generated": false,
        "solver": "convex_newton_cholesky_armijo",
        "note": "每个开奖红球对27个未开出红球取平均softplus，再对6个开奖红球做log-sum-exp。系数只在23100及以前拟合。这不是中奖概率，也不改已封存方案。"
    })
}

pub fn fit_rows(rows: &[Row]) -> Result<Fit, String> {
    if !rows.iter().any(|row| row.issue == TRAIN_END) {
        return Err("全覆盖排序需要完整23100训练边界".into());
    }
    let samples = rows
        .iter()
        .skip(WARMUP_ROWS)
        .filter(|row| row.issue <= TRAIN_END)
        .map(|row| Ok(Sample { x: features(&row.expert_p)?, mask: row.red_mask }))
        .collect::<Result<Vec<_>, String>>()?;
    fit(&samples)
}

pub fn signal(beta: &[f64; 6], expert: &[[f64; 33]; 7]) -> Result<[f64; 33], String> {
    let x = features(expert)?;
    let score: [f64; 33] = std::array::from_fn(|i| (0..6).map(|axis| beta[axis] * x[i][axis]).sum());
    if score.iter().any(|value| !value.is_finite()) {
        return Err("全覆盖得分无效".into());
    }
    Ok(score)
}

pub fn fit(samples: &[Sample]) -> Result<Fit, String> {
    if samples.is_empty()
        || samples.iter().any(|sample| {
            sample.mask.count_ones() != 6 || sample.x.iter().flatten().any(|value| !value.is_finite())
        })
    {
        return Err("全覆盖训练样本无效".into());
    }
    let mut beta = [0.; 6];
    for iterations in 0..60 {
        let (loss, grad, hess) = derivatives(samples, &beta)?;
        let gradient = grad.iter().copied().map(f64::abs).fold(0., f64::max);
        if gradient <= 1e-8 {
            return Ok(Fit { beta, iterations, gradient, objective: loss, n: samples.len() });
        }
        let step = solve(&hess, &grad)?;
        let descent = (0..6).map(|axis| grad[axis] * step[axis]).sum::<f64>();
        if !descent.is_finite() || descent <= 0. {
            return Err("全覆盖牛顿方向无效".into());
        }
        let mut accepted = false;
        for k in 0..40 {
            let scale = 2f64.powi(-k);
            let next = std::array::from_fn(|axis| beta[axis] - scale * step[axis]);
            let next_loss = objective(samples, &next)?;
            let roundoff = 32. * f64::EPSILON * (1. + loss.abs());
            if next_loss <= loss - 1e-4 * scale * descent + roundoff {
                beta = next;
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err("全覆盖牛顿步长未被接受".into());
        }
    }
    Err("全覆盖牛顿未收敛".into())
}

fn objective(samples: &[Sample], beta: &[f64; 6]) -> Result<f64, String> {
    let mut total = 0.;
    for sample in samples {
        total += period(beta, &sample.x, sample.mask)?.0;
    }
    let mean = total / samples.len() as f64 + 0.5 * RIDGE * beta.iter().map(|value| value * value).sum::<f64>();
    if !mean.is_finite() {
        return Err("全覆盖损失无效".into());
    }
    Ok(mean)
}

fn derivatives(samples: &[Sample], beta: &[f64; 6]) -> Result<(f64, [f64; 6], [[f64; 6]; 6]), String> {
    let mut loss = 0.;
    let mut grad = [0.; 6];
    let mut hess = [[0.; 6]; 6];
    for sample in samples {
        let (value, g, h) = period(beta, &sample.x, sample.mask)?;
        loss += value;
        for a in 0..6 {
            grad[a] += g[a];
            for c in 0..6 {
                hess[a][c] += h[a][c];
            }
        }
    }
    let n = samples.len() as f64;
    loss = loss / n + 0.5 * RIDGE * beta.iter().map(|value| value * value).sum::<f64>();
    for a in 0..6 {
        grad[a] = grad[a] / n + RIDGE * beta[a];
        for c in 0..6 {
            hess[a][c] /= n;
        }
        hess[a][a] += RIDGE;
    }
    if !loss.is_finite() || grad.iter().any(|value| !value.is_finite()) || hess.iter().flatten().any(|value| !value.is_finite()) {
        return Err("全覆盖导数无效".into());
    }
    Ok((loss, grad, hess))
}

fn period(beta: &[f64; 6], x: &[[f64; 6]; 33], mask: u64) -> Result<(f64, [f64; 6], [[f64; 6]; 6]), String> {
    let score: [f64; 33] = std::array::from_fn(|i| (0..6).map(|axis| beta[axis] * x[i][axis]).sum());
    let winners: Vec<usize> = (0..33).filter(|i| mask & (1 << i) != 0).collect();
    let losers: Vec<usize> = (0..33).filter(|i| mask & (1 << i) == 0).collect();
    if winners.len() != 6 || losers.len() != 27 {
        return Err("全覆盖训练需要恰好6个红球".into());
    }
    let mut per = [0.; 6];
    let mut winner_grad = [[0.; 6]; 6];
    let mut winner_hess = [[[0.; 6]; 6]; 6];
    for (index, &winner) in winners.iter().enumerate() {
        let mut value = 0.;
        let mut g = [0.; 6];
        let mut h = [[0.; 6]; 6];
        for &loser in &losers {
            let gap = score[loser] - score[winner];
            value += softplus(gap);
            let sig = sigmoid(gap);
            let curve = sig * (1. - sig);
            for a in 0..6 {
                let delta = x[loser][a] - x[winner][a];
                g[a] += sig * delta;
                for c in 0..6 {
                    h[a][c] += curve * delta * (x[loser][c] - x[winner][c]);
                }
            }
        }
        per[index] = value / LOSERS;
        for a in 0..6 {
            winner_grad[index][a] = g[a] / LOSERS;
            for c in 0..6 {
                winner_hess[index][a][c] = h[a][c] / LOSERS;
            }
        }
    }
    let peak = per.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let mut normalizer = 0.;
    let mut weight = [0.; 6];
    for index in 0..6 {
        weight[index] = (per[index] - peak).exp();
        normalizer += weight[index];
    }
    if !normalizer.is_finite() || normalizer <= 0. {
        return Err("全覆盖期损失无效".into());
    }
    for value in &mut weight {
        *value /= normalizer;
    }
    let loss = peak + normalizer.ln();
    let mut grad = [0.; 6];
    let mut hess = [[0.; 6]; 6];
    for index in 0..6 {
        for a in 0..6 {
            grad[a] += weight[index] * winner_grad[index][a];
        }
        for a in 0..6 {
            for c in 0..6 {
                hess[a][c] += weight[index] * winner_hess[index][a][c];
                hess[a][c] += weight[index] * winner_grad[index][a] * winner_grad[index][c];
            }
        }
    }
    for a in 0..6 {
        for c in 0..6 {
            hess[a][c] -= grad[a] * grad[c];
        }
    }
    Ok((loss, grad, hess))
}

fn sigmoid(gap: f64) -> f64 {
    if gap >= 0. {
        let e = (-gap).exp();
        1. / (1. + e)
    } else {
        let e = gap.exp();
        e / (1. + e)
    }
}

fn softplus(gap: f64) -> f64 {
    if gap >= 0. {
        gap + (1. + (-gap).exp()).ln()
    } else {
        (1. + gap.exp()).ln()
    }
}

fn solve(a: &[[f64; 6]; 6], g: &[f64; 6]) -> Result<[f64; 6], String> {
    let mut l = [[0.; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let value = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !value.is_finite() || value <= 0. {
                    return Err("全覆盖牛顿矩阵无效".into());
                }
                l[i][j] = value.sqrt();
            } else {
                l[i][j] = value / l[j][j];
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
    if x.iter().any(|value| !value.is_finite()) {
        return Err("全覆盖牛顿步无效".into());
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::rank_desc;

    fn separated() -> Sample {
        let mask = 0b11_1111;
        let mut x = [[0.; 6]; 33];
        for i in 0..33 {
            x[i][0] = if mask & (1 << i) != 0 { 1. } else { 0. };
        }
        let mean = x.iter().map(|row| row[0]).sum::<f64>() / 33.;
        for row in &mut x {
            row[0] -= mean;
        }
        Sample { x, mask }
    }

    #[test]
    fn zero_coefficients_match_the_closed_form() {
        let sample = separated();
        let loss = objective(&[sample], &[0.; 6]).unwrap();
        assert!((loss - (2f64.ln() + 6f64.ln())).abs() < 1e-12);
    }

    #[test]
    fn gradient_and_hessian_match_finite_differences() {
        let mut x = [[0.; 6]; 33];
        for i in 0..33 {
            for axis in 0..6 {
                x[i][axis] = ((i * 3 + axis * 5) % 17) as f64 / 17. - 0.5;
            }
        }
        for axis in 0..6 {
            let mean = x.iter().map(|row| row[axis]).sum::<f64>() / 33.;
            for row in &mut x {
                row[axis] -= mean;
            }
        }
        let sample = Sample { x, mask: (1 << 0) | (1 << 4) | (1 << 9) | (1 << 15) | (1 << 22) | (1 << 30) };
        let beta = [0.2, -0.1, 0.05, 0.3, -0.2, 0.15];
        let (_, grad, hess) = derivatives(std::slice::from_ref(&sample), &beta).unwrap();
        let eps = 1e-5;
        for axis in 0..6 {
            let mut up = beta;
            let mut down = beta;
            up[axis] += eps;
            down[axis] -= eps;
            let finite = (objective(std::slice::from_ref(&sample), &up).unwrap()
                - objective(std::slice::from_ref(&sample), &down).unwrap())
                / (2. * eps);
            assert!((finite - grad[axis]).abs() < 1e-6, "梯度 {axis}: {finite} {}", grad[axis]);
            let (_, up_grad, _) = derivatives(std::slice::from_ref(&sample), &up).unwrap();
            let (_, down_grad, _) = derivatives(std::slice::from_ref(&sample), &down).unwrap();
            for other in 0..6 {
                let finite_h = (up_grad[other] - down_grad[other]) / (2. * eps);
                assert!((finite_h - hess[other][axis]).abs() < 2e-5, "海森 {other},{axis}: {finite_h} {}", hess[other][axis]);
            }
        }
    }

    #[test]
    fn separable_axis_moves_the_six_winners_to_the_front() {
        let sample = separated();
        let fit = fit(&vec![sample.clone(); 40]).unwrap();
        assert!(fit.beta[0] > 0.);
        let score: [f64; 33] = std::array::from_fn(|i| fit.beta[0] * sample.x[i][0]);
        let order = rank_desc(&score);
        let top = order[..6].iter().map(|i| 1u64 << *i).sum::<u64>();
        assert_eq!(top, sample.mask);
    }
}
