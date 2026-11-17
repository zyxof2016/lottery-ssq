//! Normalized complete-outcome model obtained by choosing a latent 16-red pool,
//! then choosing six uniformly inside it. Read-only research, no tickets.
use crate::{
    calibration_model_v1 as calibration,
    draw::Draw,
    model::{roll_all, Row},
};
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};
pub const SIZE: usize = 16;
pub const BUDGET: usize = 512;
pub const TRAIN_END: u32 = 23100;
fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.;
    }
    (0..k).fold(1., |p, i| p * (n - i) as f64 / (i + 1) as f64)
}
fn esp(weights: impl Iterator<Item = f64>, k: usize) -> Vec<f64> {
    let mut p = vec![0.; k + 1];
    p[0] = 1.;
    for w in weights {
        for j in (1..=k).rev() {
            p[j] += w * p[j - 1];
        }
    }
    p
}
pub struct Latent {
    pub weights: Vec<f64>,
    size: usize,
    k: usize,
    z: f64,
}
impl Latent {
    pub fn new(weights: &[f64], size: usize, k: usize) -> Result<Self, String> {
        if weights.len() > 63
            || k == 0
            || k > size
            || size > weights.len()
            || weights.iter().any(|p| !p.is_finite() || *p <= 0.)
        {
            return Err("潜在号码池参数无效".into());
        }
        let scale = weights.iter().copied().fold(0., f64::max);
        let weights: Vec<_> = weights.iter().map(|w| w / scale).collect();
        let z = esp(weights.iter().copied(), size)[size];
        if !z.is_finite() || z <= 0. {
            return Err("潜在号码池归一化失败".into());
        }
        Ok(Self {
            weights,
            size,
            k,
            z,
        })
    }
    pub fn probability(&self, mask: u64) -> Result<f64, String> {
        if mask >> self.weights.len() != 0 || mask.count_ones() as usize != self.k {
            return Err("完整开奖号无效".into());
        }
        let product = self
            .weights
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, w)| w)
            .product::<f64>();
        let remaining = esp(
            self.weights
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) == 0)
                .map(|(_, w)| *w),
            self.size - self.k,
        );
        Ok(product * remaining[self.size - self.k] / self.z / choose(self.size, self.k))
    }
    pub fn pool_mass(&self, mask: u64) -> Result<f64, String> {
        if mask >> self.weights.len() != 0 {
            return Err("候选号码池越界".into());
        }
        let a = esp(
            self.weights
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, w)| *w),
            self.size,
        );
        let b = esp(
            self.weights
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) == 0)
                .map(|(_, w)| *w),
            self.size,
        );
        Ok((self.k..=self.size)
            .map(|r| choose(r, self.k) * a[r] * b[self.size - r])
            .sum::<f64>()
            / self.z
            / choose(self.size, self.k))
    }
    fn inclusion(&self) -> Vec<f64> {
        (0..self.weights.len())
            .map(|i| {
                self.k as f64 / self.size as f64
                    * self.weights[i]
                    * esp(
                        self.weights
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, w)| *w),
                        self.size - 1,
                    )[self.size - 1]
                    / self.z
            })
            .collect()
    }
    fn best_pool(&self, size: usize) -> u64 {
        let mut order: Vec<_> = (0..self.weights.len()).collect();
        order.sort_by(|a, b| self.weights[*b].total_cmp(&self.weights[*a]).then(a.cmp(b)));
        order[..size].iter().fold(0, |m, i| m | (1 << i))
    }
}
pub struct Mixture {
    models: Vec<(Latent, f64)>,
    uniform: f64,
    n: usize,
    size: usize,
    k: usize,
}
pub struct Search {
    pub mask: u64,
    pub marginal: u64,
    pub mass: f64,
    pub upper: f64,
    pub evaluations: usize,
    pub swap_optimal: bool,
    pub identified: bool,
}
impl Mixture {
    pub fn new(
        experts: &[Vec<f64>],
        weights: &[f64],
        n: usize,
        size: usize,
        k: usize,
    ) -> Result<Self, String> {
        if weights.len() != experts.len() + 1
            || weights.iter().any(|v| !v.is_finite() || *v < 0.)
            || (weights.iter().sum::<f64>() - 1.).abs() > 1e-9
            || n > 63
            || size > n
            || k == 0
            || k > size
            || experts.iter().any(|e| e.len() != n)
        {
            return Err("覆盖混合模型参数无效".into());
        }
        let models = experts
            .iter()
            .zip(weights)
            .filter(|(_, w)| **w > 0.)
            .map(|(p, w)| Ok((Latent::new(p, size, k)?, *w)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            models,
            uniform: *weights.last().unwrap(),
            n,
            size,
            k,
        })
    }
    pub fn mass(&self, mask: u64) -> Result<f64, String> {
        if mask >> self.n != 0 || mask.count_ones() as usize != self.size {
            return Err("候选池数量或号码无效".into());
        }
        Ok(
            self.uniform * choose(self.size, self.k) / choose(self.n, self.k)
                + self
                    .models
                    .iter()
                    .map(|(m, w)| m.pool_mass(mask).map(|v| v * w))
                    .collect::<Result<Vec<_>, _>>()?
                    .iter()
                    .sum::<f64>(),
        )
    }
    fn order(&self) -> Vec<usize> {
        let mut p = vec![self.uniform * self.k as f64 / self.n as f64; self.n];
        for (m, w) in &self.models {
            for (i, v) in m.inclusion().into_iter().enumerate() {
                p[i] += w * v;
            }
        }
        let mut order: Vec<_> = (0..self.n).collect();
        order.sort_by(|a, b| {
            ((p[*b] * 1e12).round() as i64)
                .cmp(&((p[*a] * 1e12).round() as i64))
                .then(a.cmp(b))
        });
        order
    }
    pub fn search(&self, budget: usize, seeds: &[u64]) -> Result<Search, String> {
        if budget < self.models.len() + 1 + seeds.len() {
            return Err("覆盖搜索预算不足".into());
        }
        let order = self.order();
        let marginal = order[..self.size].iter().fold(0, |m, i| m | (1 << i));
        let mut best = marginal;
        let mut mass = self.mass(best)?;
        let mut evaluations = 1;
        let tolerance = choose(self.size, self.k) / choose(self.n, self.k) * 1e-12;
        let mut upper = self.uniform * choose(self.size, self.k) / choose(self.n, self.k);
        let mut candidates = seeds.to_vec();
        for (m, w) in &self.models {
            let pool = m.best_pool(self.size);
            upper += w * m.pool_mass(pool)?;
            candidates.push(pool);
        }
        for candidate in candidates {
            let v = self.mass(candidate)?;
            evaluations += 1;
            if v > mass + tolerance {
                best = candidate;
                mass = v;
            }
        }
        let mut swap_optimal = false;
        while evaluations < budget {
            let old = best;
            let mut exhausted = false;
            for a in 0..self.n {
                if old & (1 << a) == 0 {
                    continue;
                }
                for b in 0..self.n {
                    if old & (1 << b) != 0 {
                        continue;
                    }
                    if evaluations == budget {
                        exhausted = true;
                        break;
                    }
                    let candidate = old ^ (1 << a) ^ (1 << b);
                    let v = self.mass(candidate)?;
                    evaluations += 1;
                    if v > mass + tolerance {
                        best = candidate;
                        mass = v;
                    }
                }
                if exhausted {
                    break;
                }
            }
            if exhausted {
                break;
            }
            if best == old {
                swap_optimal = true;
                break;
            }
        }
        let identified = self
            .models
            .iter()
            .any(|(m, _)| m.weights.windows(2).any(|w| w[0] != w[1]));
        Ok(Search {
            mask: best,
            marginal,
            mass,
            upper: upper.max(mass),
            evaluations,
            swap_optimal,
            identified,
        })
    }
    pub fn ranking(&self, s: &Search) -> Vec<u8> {
        let order = self.order();
        order
            .iter()
            .filter(|i| s.mask & (1 << **i) != 0)
            .chain(order.iter().filter(|i| s.mask & (1 << **i) == 0))
            .map(|i| *i as u8 + 1)
            .collect()
    }
    pub fn tie_expected(&self, pool: u64, actual: u64) -> f64 {
        let mut groups = std::collections::BTreeMap::<Vec<u64>, (usize, usize, usize)>::new();
        for i in 0..self.n {
            let signature = self
                .models
                .iter()
                .map(|(m, _)| m.weights[i].to_bits())
                .collect();
            let entry = groups.entry(signature).or_default();
            entry.0 += 1;
            entry.1 += usize::from(pool & (1 << i) != 0);
            entry.2 += usize::from(actual & (1 << i) != 0);
        }
        groups
            .values()
            .map(|(n, s, w)| {
                if w > s {
                    0.
                } else {
                    choose(n - w, s - w) / choose(*n, *s)
                }
            })
            .product()
    }
}
pub fn mixture(expert: &[[f64; 33]; 7], weights: &[f64]) -> Result<Mixture, String> {
    Mixture::new(
        &expert.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        weights,
        33,
        SIZE,
        6,
    )
}
pub fn fit_rows(rows: &[Row]) -> Result<calibration::Fit, String> {
    if !rows.iter().any(|r| r.issue == TRAIN_END) {
        return Err("覆盖模型缺少23100冻结训练边界".into());
    }
    let mut x = Vec::new();
    for row in rows.iter().skip(30).filter(|r| r.issue <= TRAIN_END) {
        let mut p = Vec::new();
        for expert in &row.expert_p {
            p.push(Latent::new(expert, SIZE, 6)?.probability(row.red_mask)? * 1_107_568.);
        }
        p.push(1.);
        x.push(p);
    }
    calibration::fit(&x)
}
pub fn policy() -> Value {
    json!({"kind":"latent_16_pool_complete_outcome_mixture","training_last_issue":TRAIN_END,"latent_pool_size":SIZE,"outcome_red_count":6,"fit":"convex_complete_six_log_likelihood_plus_40_log_uniform_weight","selection_objective":"all_six_simultaneously_inside_16","candidate_budget":BUDGET,"tickets_generated":false,"automatic_promotion":false})
}
pub fn review(draws: &[Draw]) -> Result<Value, String> {
    let snap = roll_all(draws)?;
    let fit = fit_rows(&snap.rows)?;
    let mut records = Vec::new();
    for row in snap.rows.iter().filter(|r| r.issue > TRAIN_END) {
        let m = mixture(&row.expert_p, &fit.weights)?;
        let s = m.search(BUDGET, &[])?;
        let q = fit.weights[7] / 1_107_568.
            + row
                .expert_p
                .iter()
                .zip(&fit.weights)
                .filter(|(_, w)| **w > 0.)
                .map(|(p, w)| {
                    Latent::new(p, SIZE, 6)
                        .and_then(|m| m.probability(row.red_mask))
                        .map(|v| v * w)
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .sum::<f64>();
        records.push(json!({"issue":row.issue,"log_excess":(q*1_107_568.).ln(),"marginal_full":s.marginal&row.red_mask==row.red_mask,"search_full":s.mask&row.red_mask==row.red_mask,"marginal_tie_expected":m.tie_expected(s.marginal,row.red_mask),"search_tie_expected":m.tie_expected(s.mask,row.red_mask),"mass_gain":s.mass/m.mass(s.marginal)?-1.,"upper_gap":s.upper/s.mass-1.,"evaluations":s.evaluations}));
    }
    let window = |lo, hi| {
        let r: Vec<_> = records
            .iter()
            .filter(|r| r["issue"].as_u64().is_some_and(|i| i >= lo && i <= hi))
            .collect();
        let n = r.len();
        json!({"n":n,"mean_log_excess":(n>0).then(||r.iter().map(|r|r["log_excess"].as_f64().unwrap()).sum::<f64>()/n as f64),"marginal_full_six":r.iter().filter(|r|r["marginal_full"]==true).count(),"search_full_six":r.iter().filter(|r|r["search_full"]==true).count(),"marginal_tie_expected":r.iter().map(|r|r["marginal_tie_expected"].as_f64().unwrap()).sum::<f64>(),"search_tie_expected":r.iter().map(|r|r["search_tie_expected"].as_f64().unwrap()).sum::<f64>()})
    };
    let m = mixture(&snap.next_red, &fit.weights)?;
    let s = m.search(BUDGET, &[])?;
    let last = draws.last().ok_or("没有开奖数据")?;
    Ok(
        json!({"kind":"complete_six_in_sixteen_research","actual_recommendation":false,"source_sha256":crate::hash::sha256_hex(include_str!("cover16.rs").as_bytes()),"policy":policy(),"data_sha256":crate::draw::data_sha256(draws),"cutoff_issue":crate::draw::fmt_issue(last.issue),"target_issue":crate::draw::fmt_issue(last.issue+1),"weights":fit.weights,"fit_gap":fit.gap,"fit_iterations":fit.iterations,"windows":{"validation":window(23101,25110),"confirmation_exploratory":window(25111,26115),"tracking":window(26116,u64::MAX)},"current":{"ranking":m.ranking(&s),"mass":s.mass,"upper":s.upper,"evaluations":s.evaluations,"one_swap_optimal":s.swap_optimal,"identified":s.identified,"globally_optimal_within_tolerance":s.upper-s.mass<=choose(16,6)/choose(33,6)*1e-12,"scope":"分量最优值加权和是上界；单交换搜索只证明已查邻域，差距未闭合时不冒充全局最优。模型内最优不证明真实覆盖优势。"},"periods":records,"tickets_generated":false,"writes_archives":false,"automatic_promotion":false,"reason":"直接覆盖目标也需要完整六红评分和独立前瞻检验；保留现有推荐，研究结果不生成新票面。历史探索不充当新的前瞻成绩。"}),
    )
}
pub fn report(draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let key = crate::draw::data_sha256(draws);
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "全六覆盖缓存异常")?;
    if let Some((k, v)) = cache.as_ref() {
        if *k == key {
            return Ok(v.clone());
        }
    }
    let mut v = review(draws)?;
    v.as_object_mut().unwrap().remove("periods");
    *cache = Some((key, v.clone()));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn subsets(n: usize, k: usize) -> Vec<u64> {
        (0u64..1 << n)
            .filter(|m| m.count_ones() as usize == k)
            .collect()
    }
    #[test]
    fn complete_outcomes_and_pool_coverage_match_independent_latent_enumeration() {
        let w = [1., 2., 3., 4., 5., 6., 7., 8.];
        let m = Latent::new(&w, 5, 3).unwrap();
        let pools = subsets(8, 5);
        let z = pools
            .iter()
            .map(|a| {
                (0..8)
                    .filter(|i| a & (1 << i) != 0)
                    .map(|i| w[i])
                    .product::<f64>()
            })
            .sum::<f64>();
        let mut total = 0.;
        for s in subsets(8, 3) {
            let oracle = pools
                .iter()
                .filter(|a| **a & s == s)
                .map(|a| {
                    (0..8)
                        .filter(|i| a & (1 << i) != 0)
                        .map(|i| w[i])
                        .product::<f64>()
                        / z
                        / 10.
                })
                .sum::<f64>();
            let q = m.probability(s).unwrap();
            assert!((q - oracle).abs() < 1e-14);
            total += q;
        }
        assert!((total - 1.).abs() < 1e-14);
        for (i, pi) in m.inclusion().iter().enumerate() {
            let expected = subsets(8, 3)
                .iter()
                .filter(|s| **s & (1 << i) != 0)
                .map(|s| m.probability(*s).unwrap())
                .sum::<f64>();
            assert!((pi - expected).abs() < 1e-14);
        }
        for b in subsets(8, 4) {
            let oracle = subsets(8, 3)
                .iter()
                .filter(|s| **s & b == **s)
                .map(|s| m.probability(*s).unwrap())
                .sum::<f64>();
            assert!((m.pool_mass(b).unwrap() - oracle).abs() < 1e-14);
        }
    }
    #[test]
    fn component_optimum_bounds_every_mixture_pool_and_budget_never_fakes_global_optimum() {
        let e = vec![
            vec![8., 7., 6., 5., 4., 3., 2., 1.],
            vec![1., 2., 3., 4., 5., 6., 7., 8.],
        ];
        let m = Mixture::new(&e, &[0.3, 0.4, 0.3], 8, 4, 3).unwrap();
        let s = m.search(512, &[]).unwrap();
        let oracle = subsets(8, 4)
            .into_iter()
            .map(|b| m.mass(b).unwrap())
            .fold(0., f64::max);
        assert!(s.mass <= oracle + 1e-14);
        assert!(s.upper + 1e-14 >= oracle);
        assert!(s.mass + 1e-14 >= m.mass(s.marginal).unwrap());
        let partial = m.search(3, &[]).unwrap();
        assert!(!partial.swap_optimal);
        assert!(partial.upper + 1e-14 >= oracle);
        let seeded = m.search(4, &[s.mask]).unwrap();
        assert!(seeded.mass + 1e-14 >= s.mass);
    }
    #[test]
    fn uniform_predictions_have_no_preferred_numbers_and_symmetric_ties() {
        let m = Latent::new(&[1.; 8], 4, 3).unwrap();
        for s in subsets(8, 3) {
            assert!((m.probability(s).unwrap() - 1. / 56.).abs() < 1e-14);
        }
        let mix = Mixture::new(&[], &[1.], 8, 4, 3).unwrap();
        let s = mix.search(512, &[]).unwrap();
        assert!(!s.identified);
        assert!(
            !Mixture::new(&[vec![1.; 8]], &[0.5, 0.5], 8, 4, 3)
                .unwrap()
                .search(512, &[])
                .unwrap()
                .identified
        );
        assert!((s.mass - 4. / 56.).abs() < 1e-14);
        assert!((mix.tie_expected(s.mask, 0b111) - 4. / 56.).abs() < 1e-14);
        assert!(Latent::new(&[0.; 8], 4, 3).is_err());
        assert!(Latent::new(&[1.; 8], 2, 3).is_err());
    }
    #[test]
    fn appended_outcome_cannot_change_frozen_weights_or_previous_pool_search() {
        let mut d = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let before = roll_all(&d).unwrap();
        let weights = fit_rows(&before.rows).unwrap().weights;
        let a = mixture(&before.next_red, &weights)
            .unwrap()
            .search(BUDGET, &[])
            .unwrap();
        d.push(
            Draw::from_parts(d.last().unwrap().issue + 1, 20261117, [1, 2, 3, 4, 5, 6], 1).unwrap(),
        );
        let after = roll_all(&d).unwrap();
        let new = fit_rows(&after.rows).unwrap().weights;
        assert_eq!(weights, new);
        let b = mixture(&after.rows.last().unwrap().expert_p, &new)
            .unwrap()
            .search(BUDGET, &[])
            .unwrap();
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.upper, b.upper);
        assert_eq!(a.evaluations, b.evaluations);
    }
}
