/// 分布、区间、效应量、多重比较与可复现随机数。

pub fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c = 1u64;
    for i in 1..=k {
        c = c * (n - k + i) / i;
    }
    c
}

pub fn hg_mean_var(marked: usize) -> (f64, f64) {
    let n_pop = 33.0;
    let draw = 6.0;
    let k = marked as f64;
    let mean = draw * k / n_pop;
    let var = draw * (k / n_pop) * (1.0 - k / n_pop) * (n_pop - draw) / (n_pop - 1.0);
    (mean, var)
}

pub fn expected_red_hits(marked: usize) -> f64 {
    hg_mean_var(marked).0
}

pub fn norm_cdf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let d = 0.39894228040143267794 * (-0.5 * z * z).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = d * poly;
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

pub fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

pub fn two_sided_from_z(z: f64) -> f64 {
    (2.0 * (1.0 - norm_cdf(z.abs()))).clamp(0.0, 1.0)
}

pub fn z_test(stat: f64, se: f64) -> f64 {
    if se < 1e-15 {
        return if stat.abs() < 1e-12 { 1.0 } else { 0.0 };
    }
    two_sided_from_z(stat / se)
}

pub fn hg_total_p(sum_hits: f64, marked: usize, n_draws: usize) -> f64 {
    let (mean, var) = hg_mean_var(marked);
    let expected = mean * n_draws as f64;
    let sd = (var * n_draws as f64).sqrt();
    z_test(sum_hits - expected, sd)
}

pub fn binom_cc_p(hits: f64, n: f64, p0: f64) -> f64 {
    let se = (n * p0 * (1.0 - p0)).sqrt();
    let diff = (hits - n * p0).abs() - 0.5;
    if diff <= 0.0 || se < 1e-15 {
        return 1.0;
    }
    two_sided_from_z(diff / se)
}

pub fn mean_sd(xs: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n;
    if n < 2.0 {
        return (mean, 0.0, 0.0);
    }
    let mut v = 0.0;
    for x in xs {
        let d = x - mean;
        v += d * d;
    }
    let sd = (v / (n - 1.0)).sqrt();
    let se = sd / n.sqrt();
    (mean, sd, se)
}

pub fn normal_ci(mean: f64, se: f64, z: f64) -> (f64, f64) {
    (mean - z * se, mean + z * se)
}

pub fn wilson(hits: f64, n: f64, z: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let phat = (hits / n).clamp(0.0, 1.0);
    let z2 = z * z;
    let den = 1.0 + z2 / n;
    let center = (phat + z2 / (2.0 * n)) / den;
    let marg = z * ((phat * (1.0 - phat) / n) + z2 / (4.0 * n * n)).sqrt() / den;
    (
        (center - marg).clamp(0.0, 1.0),
        (center + marg).clamp(0.0, 1.0),
    )
}

pub fn cohen_h(p1: f64, p2: f64) -> f64 {
    2.0 * p1.clamp(0.0, 1.0).sqrt().asin() - 2.0 * p2.clamp(0.0, 1.0).sqrt().asin()
}

pub fn cohen_dz(mean: f64, sd: f64) -> f64 {
    if sd < 1e-15 {
        0.0
    } else {
        mean / sd
    }
}

pub fn effect_size_word(abs_effect: f64) -> &'static str {
    if abs_effect < 0.2 {
        "可忽略"
    } else if abs_effect < 0.5 {
        "小"
    } else if abs_effect < 0.8 {
        "中"
    } else {
        "大"
    }
}

pub fn adjust_holm(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut running = 0.0_f64;
    let mut adj_sorted = vec![0.0; m];
    for (rank, &idx) in order.iter().enumerate() {
        let val = (p[idx] * (m - rank) as f64).min(1.0);
        running = running.max(val);
        adj_sorted[rank] = running.min(1.0);
    }
    let mut out = vec![0.0; m];
    for (rank, &idx) in order.iter().enumerate() {
        out[idx] = adj_sorted[rank];
    }
    out
}

pub fn adjust_bh(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut raw = vec![0.0; m];
    for (rank, &idx) in order.iter().enumerate() {
        raw[rank] = (p[idx] * m as f64 / (rank as f64 + 1.0)).min(1.0);
    }
    for rank in (0..m - 1).rev() {
        raw[rank] = raw[rank].min(raw[rank + 1]);
    }
    let mut out = vec![0.0; m];
    for (rank, &idx) in order.iter().enumerate() {
        out[idx] = raw[rank];
    }
    out
}

pub fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n == 0 || n != b.len() {
        return None;
    }
    let nf = n as f64;
    let ma = a.iter().sum::<f64>() / nf;
    let mb = b.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da < 1e-18 || db < 1e-18 {
        return None;
    }
    Some(num / (da.sqrt() * db.sqrt()))
}

pub fn eigenvalues_sym(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..80 {
        let mut p = 0;
        let mut q = 1.min(n.saturating_sub(1));
        let mut max = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i][j].abs();
                if v > max {
                    max = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 || n < 2 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let aip = a[i][p];
            let aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[p][i] = a[i][p];
            a[i][q] = s * aip + c * aiq;
            a[q][i] = a[i][q];
        }
    }
    (0..n).map(|i| a[i][i]).collect()
}

pub fn participation_ratio(eigen: &[f64]) -> f64 {
    let vals: Vec<f64> = eigen.iter().map(|v| v.max(0.0)).collect();
    let sum = vals.iter().sum::<f64>();
    let sumsq = vals.iter().map(|v| v * v).sum::<f64>();
    if sumsq < 1e-18 {
        0.0
    } else {
        sum * sum / sumsq
    }
}

pub fn block_means(series: &[f64], block: usize) -> Vec<f64> {
    if block == 0 || series.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + block <= series.len() {
        let m = series[i..i + block].iter().sum::<f64>() / block as f64;
        out.push(m);
        i += block;
    }
    let rem = series.len() - i;
    if rem * 2 >= block && rem > 0 {
        out.push(series[i..].iter().sum::<f64>() / rem as f64);
    }
    out
}

pub fn ols_slope(y: &[f64]) -> Option<(f64, f64)> {
    let n = y.len();
    if n < 3 {
        return None;
    }
    let nf = n as f64;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let x = i as f64;
        sx += x;
        sy += v;
        sxx += x * x;
        sxy += x * v;
    }
    let den = nf * sxx - sx * sx;
    if den.abs() < 1e-18 {
        return None;
    }
    let slope = (nf * sxy - sx * sy) / den;
    let intercept = (sy - slope * sx) / nf;
    let mut sse = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let e = v - (intercept + slope * i as f64);
        sse += e * e;
    }
    let sigma2 = sse / (nf - 2.0);
    let slope_se = (sigma2 * nf / den).sqrt();
    Some((slope, slope_se))
}

pub fn acf1(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 {
        return 0.0;
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let d = y[i] - mean;
        den += d * d;
        if i + 1 < n {
            num += d * (y[i + 1] - mean);
        }
    }
    if den < 1e-18 {
        0.0
    } else {
        num / den
    }
}

pub fn block_signflip_p(diff: &[f64], block: usize, draws: usize, seed: u64) -> f64 {
    let n = diff.len();
    if n == 0 || draws == 0 {
        return 1.0;
    }
    let mean = diff.iter().sum::<f64>() / n as f64;
    let abs_obs = mean.abs();
    let mut rng = Rng::new(seed);
    let mut extreme = 0usize;
    let block = block.max(1);
    for _ in 0..draws {
        let mut s = 0.0;
        let mut i = 0;
        while i < n {
            let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
            let end = (i + block).min(n);
            for j in i..end {
                s += sign * diff[j];
            }
            i = end;
        }
        if (s / n as f64).abs() + 1e-15 >= abs_obs {
            extreme += 1;
        }
    }
    (extreme as f64 + 1.0) / (draws as f64 + 1.0)
}

pub fn red_ticket_pmf() -> [f64; 7] {
    let den = binom(33, 6) as f64;
    let mut p = [0.0; 7];
    for k in 0..=6 {
        let num = binom(6, k) * binom(27, 6 - k);
        p[k as usize] = num as f64 / den;
    }
    p
}

pub fn emax_iid(t: usize) -> f64 {
    if t == 0 {
        return 0.0;
    }
    let pmf = red_ticket_pmf();
    let mut e = 0.0;
    for k in 1..=6 {
        let f: f64 = pmf[..k].iter().sum();
        e += 1.0 - f.powi(t as i32);
    }
    e
}

pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let t = idx - lo as f64;
    sorted[lo] * (1.0 - t) + sorted[hi] * t
}

#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut x = self.0;
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    pub fn gen_range(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let n = n as u64;
        loop {
            let x = self.next_u64();
            let m = (x as u128) * (n as u128);
            let lo = m as u64;
            if lo < n {
                let t = n.wrapping_neg() % n;
                if lo < t {
                    continue;
                }
            }
            return (m >> 64) as usize;
        }
    }

    pub fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn gamma_shape(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.next_f64().max(1e-12);
            return self.gamma_shape(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.standard_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next_f64();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn dirichlet<const K: usize>(&mut self, alpha: &[f64; K]) -> [f64; K] {
        let mut g = [0.0; K];
        let mut s = 0.0;
        for i in 0..K {
            g[i] = self.gamma_shape(alpha[i].max(1e-6));
            s += g[i];
        }
        if s <= 0.0 {
            return [1.0 / K as f64; K];
        }
        for v in &mut g {
            *v /= s;
        }
        g
    }

    pub fn sample_red_mask(&mut self) -> u64 {
        let mut pop = [0u8; 33];
        for i in 0..33 {
            pop[i] = i as u8;
        }
        let mut mask = 0u64;
        for i in 0..6 {
            let j = i + self.gen_range(33 - i);
            pop.swap(i, j);
            mask |= 1u64 << pop[i];
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binom_known() {
        assert_eq!(binom(33, 6), 1_107_568);
        assert_eq!(binom(6, 3), 20);
        assert_eq!(binom(10, 3), 120);
        assert_eq!(binom(27, 0), 1);
    }

    #[test]
    fn pmf_and_emax() {
        let p = red_ticket_pmf();
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        let mean: f64 = p.iter().enumerate().map(|(k, pk)| k as f64 * pk).sum();
        assert!((mean - 36.0 / 33.0).abs() < 1e-9);
        assert!((emax_iid(1) - 36.0 / 33.0).abs() < 1e-9);
        assert!(emax_iid(8) > emax_iid(1));
    }

    #[test]
    fn normal_quantile() {
        let z = norm_ppf(0.975);
        assert!((z - 1.95996398454).abs() < 1e-6);
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn holm_and_bh() {
        let p = [0.01, 0.04, 0.03];
        let holm = adjust_holm(&p);
        assert!((holm[0] - 0.03).abs() < 1e-12);
        assert!((holm[1] - 0.06).abs() < 1e-12);
        assert!((holm[2] - 0.06).abs() < 1e-12);
        let bh = adjust_bh(&p);
        assert!((bh[0] - 0.03).abs() < 1e-12);
        assert!((bh[1] - 0.04).abs() < 1e-12);
        assert!((bh[2] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn eigen_participation() {
        let id = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let ev = eigenvalues_sym(id);
        assert!((participation_ratio(&ev) - 3.0).abs() < 1e-6);
        let c = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let mut ev = eigenvalues_sym(c);
        ev.sort_by(|a, b| b.total_cmp(a));
        assert!((ev[0] - 1.5).abs() < 1e-6);
        assert!((ev[1] - 0.5).abs() < 1e-6);
    }
}
