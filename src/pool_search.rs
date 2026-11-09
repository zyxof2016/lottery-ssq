//! Deterministic search for full-set coverage, not sum of single-number scores.
//! Equal expert signatures are grouped. Component-wise completion bounds give
//! an auditable optimality gap when the deterministic node budget is exhausted.
use serde_json::{json, Value};
const D: usize = 7;
type Poly = [f64; D];
fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.;
    }
    (0..k).fold(1., |v, i| v * (n - i) as f64 / (i + 1) as f64)
}
fn product(a: &Poly, b: &Poly, k: usize) -> Poly {
    let mut out = [0.; D];
    for i in 0..=k {
        for j in 0..=k - i {
            out[i + j] += a[i] * b[j];
        }
    }
    out
}
fn polynomial(values: impl IntoIterator<Item = f64>, k: usize) -> Poly {
    let mut p = [0.; D];
    p[0] = 1.;
    for w in values {
        for j in (1..=k).rev() {
            p[j] += w * p[j - 1];
        }
    }
    p
}
#[derive(Clone)]
struct Expert {
    p: Vec<f64>,
    weight: f64,
    coefficient: Poly,
}
pub struct Coverage {
    n: usize,
    k: usize,
    size: usize,
    previous: u64,
    experts: Vec<Expert>,
    uniform: f64,
}
#[derive(Clone)]
struct Group {
    ids: Vec<usize>,
    previous: bool,
}
#[derive(Clone)]
struct Node {
    depth: usize,
    selected: usize,
    mask: u64,
    poly: Vec<[Poly; 2]>,
    upper: f64,
}
pub struct Search {
    pub mask: u64,
    pub mass: f64,
    pub upper: f64,
    pub nodes: usize,
    pub groups: usize,
    pub group_signatures: Value,
    pub complete: bool,
    pub tied_membership: bool,
}
impl Coverage {
    pub fn new(
        n: usize,
        k: usize,
        size: usize,
        previous: u64,
        weights: &[f64],
        probabilities: &[Vec<f64>],
    ) -> Result<Self, String> {
        if n > 63
            || k == 0
            || k >= D
            || size < k
            || size > n
            || previous >> n != 0
            || weights.len() != probabilities.len() + 1
            || weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || (weights.iter().sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err("完整入池模型参数无效".into());
        }
        let previous_n = previous.count_ones() as usize;
        let total = choose(n, k);
        let mut experts = Vec::new();
        for (p, w) in probabilities.iter().zip(weights) {
            if p.len() != n || p.iter().any(|x| !x.is_finite() || *x <= 0.) {
                return Err("完整入池专家分布无效".into());
            }
            if *w == 0. {
                continue;
            }
            let a = polynomial((0..n).filter(|i| previous & (1 << i) != 0).map(|i| p[i]), k);
            let b = polynomial((0..n).filter(|i| previous & (1 << i) == 0).map(|i| p[i]), k);
            let coefficient = std::array::from_fn(|r| {
                if r <= k {
                    let q = choose(previous_n, r) * choose(n - previous_n, k - r) / total;
                    if q > 0. {
                        q / (a[r] * b[k - r])
                    } else {
                        0.
                    }
                } else {
                    0.
                }
            });
            if coefficient.iter().any(|v| !v.is_finite())
                || (0..=k).any(|r| {
                    choose(previous_n, r) * choose(n - previous_n, k - r) > 0.
                        && (!(a[r] * b[k - r]).is_finite() || a[r] * b[k - r] <= 0.)
                })
            {
                return Err("完整入池专家归一化溢出".into());
            }
            experts.push(Expert {
                p: p.clone(),
                weight: *w,
                coefficient,
            });
        }
        Ok(Self {
            n,
            k,
            size,
            previous,
            experts,
            uniform: *weights.last().unwrap() * choose(size, k) / total,
        })
    }
    fn empty(&self) -> Vec<[Poly; 2]> {
        let mut one = [0.; D];
        one[0] = 1.;
        vec![[one; 2]; self.experts.len()]
    }
    fn value(&self, polys: &[[Poly; 2]]) -> f64 {
        self.uniform
            + self
                .experts
                .iter()
                .zip(polys)
                .map(|(e, p)| {
                    e.weight
                        * (0..=self.k)
                            .map(|r| e.coefficient[r] * p[1][r] * p[0][self.k - r])
                            .sum::<f64>()
                })
                .sum::<f64>()
    }
    pub fn mass(&self, mask: u64) -> Result<f64, String> {
        if mask >> self.n != 0 || mask.count_ones() as usize != self.size {
            return Err("候选池数量或号码无效".into());
        }
        let polys: Vec<_> = self
            .experts
            .iter()
            .map(|e| {
                std::array::from_fn(|g| {
                    polynomial(
                        (0..self.n)
                            .filter(|i| {
                                mask & (1 << i) != 0
                                    && usize::from(self.previous & (1 << i) != 0) == g
                            })
                            .map(|i| e.p[i]),
                        self.k,
                    )
                })
            })
            .collect();
        Ok(self.value(&polys))
    }
    pub fn marginal_seed(&self) -> u64 {
        // Marginals are differences between full-universe and leave-one-out
        // partitions under the SAME fixed-overlap complete distribution.
        let mut scored = Vec::new();
        for i in 0..self.n {
            let mut inclusion = 0.;
            for e in &self.experts {
                let g = usize::from(self.previous & (1 << i) != 0);
                let a = polynomial(
                    (0..self.n)
                        .filter(|j| *j != i && self.previous & (1 << j) != 0)
                        .map(|j| e.p[j]),
                    self.k,
                );
                let b = polynomial(
                    (0..self.n)
                        .filter(|j| *j != i && self.previous & (1 << j) == 0)
                        .map(|j| e.p[j]),
                    self.k,
                );
                inclusion += e.weight
                    * e.p[i]
                    * (0..=self.k)
                        .filter(|r| if g == 1 { *r > 0 } else { *r < self.k })
                        .map(|r| e.coefficient[r] * a[r - g] * b[self.k - r - (1 - g)])
                        .sum::<f64>();
            }
            scored.push((inclusion, i));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored[..self.size].iter().fold(0, |m, (_, i)| m | (1 << i))
    }
    pub fn optimize(&self, node_budget: usize) -> Result<Search, String> {
        self.optimize_with_seeds(node_budget, &[])
    }
    pub fn optimize_with_seeds(&self, node_budget: usize, seeds: &[u64]) -> Result<Search, String> {
        if node_budget == 0 {
            return Err("搜索节点预算须为正数".into());
        }
        let mut map = std::collections::BTreeMap::<Vec<u64>, Vec<usize>>::new();
        for i in 0..self.n {
            let mut key = vec![u64::from(self.previous & (1 << i) != 0)];
            key.extend(self.experts.iter().map(|e| e.p[i].to_bits()));
            map.entry(key).or_default().push(i);
        }
        let mut groups: Vec<_> = map
            .into_values()
            .map(|ids| Group {
                previous: self.previous & (1 << ids[0]) != 0,
                ids,
            })
            .collect();
        groups.sort_by(|a, b| b.ids.len().cmp(&a.ids.len()).then(a.ids[0].cmp(&b.ids[0])));
        // suffix[depth][expert][membership][take] is the ESP of that expert's
        // best `take` remaining numbers in that membership group.
        let mut suffix = Vec::new();
        let mut counts = Vec::new();
        for d in 0..=groups.len() {
            let ids: Vec<_> = groups[d..]
                .iter()
                .flat_map(|g| g.ids.iter().copied())
                .collect();
            counts.push([
                ids.iter()
                    .filter(|i| self.previous & (1 << **i) == 0)
                    .count(),
                ids.iter()
                    .filter(|i| self.previous & (1 << **i) != 0)
                    .count(),
            ]);
            let tables: Vec<[Vec<Poly>; 2]> = self
                .experts
                .iter()
                .map(|e| {
                    std::array::from_fn(|g| {
                        let mut w: Vec<_> = ids
                            .iter()
                            .filter(|i| usize::from(self.previous & (1 << **i) != 0) == g)
                            .map(|i| e.p[*i])
                            .collect();
                        w.sort_by(|a, b| b.total_cmp(a));
                        (0..=w.len().min(self.size))
                            .map(|n| polynomial(w[..n].iter().copied(), self.k))
                            .collect()
                    })
                })
                .collect();
            suffix.push(tables);
        }
        let baseline = choose(self.size, self.k) / choose(self.n, self.k);
        let tolerance = baseline * 1e-12;
        let bound = |node: &Node| -> f64 {
            let need = self.size - node.selected;
            let remaining = counts[node.depth];
            let lo = need.saturating_sub(remaining[0]);
            let hi = need.min(remaining[1]);
            if lo > hi {
                return f64::NEG_INFINITY;
            }
            let mut out = self.uniform;
            for (eidx, e) in self.experts.iter().enumerate() {
                let mut best: f64 = 0.;
                for previous in lo..=hi {
                    let a = product(
                        &node.poly[eidx][1],
                        &suffix[node.depth][eidx][1][previous],
                        self.k,
                    );
                    let b = product(
                        &node.poly[eidx][0],
                        &suffix[node.depth][eidx][0][need - previous],
                        self.k,
                    );
                    best = best.max(
                        (0..=self.k)
                            .map(|r| e.coefficient[r] * a[r] * b[self.k - r])
                            .sum(),
                    );
                }
                out += e.weight * best;
            }
            out + baseline * 1e-13
        };
        let mut mask = self.marginal_seed();
        let mut best = self.mass(mask)?;
        for seed in seeds {
            let mass = self.mass(*seed)?;
            if mass > best {
                mask = *seed;
                best = mass;
            }
        }
        // Expert-wise seeds help the lower bound; they do not use any outcomes.
        for e in &self.experts {
            for prev_n in 0..=self.size.min(self.previous.count_ones() as usize) {
                if self.size - prev_n > self.n - self.previous.count_ones() as usize {
                    continue;
                }
                let mut selected = 0;
                for (g, take) in [(true, prev_n), (false, self.size - prev_n)] {
                    let mut ids: Vec<_> = (0..self.n)
                        .filter(|i| (self.previous & (1 << i) != 0) == g)
                        .collect();
                    ids.sort_by(|a, b| e.p[*b].total_cmp(&e.p[*a]).then(a.cmp(b)));
                    selected |= ids[..take].iter().fold(0, |m, i| m | (1 << i));
                }
                let mass = self.mass(selected)?;
                if mass > best {
                    mask = selected;
                    best = mass;
                }
            }
        }
        let mut root = Node {
            depth: 0,
            selected: 0,
            mask: 0,
            poly: self.empty(),
            upper: 0.,
        };
        root.upper = bound(&root);
        let mut stack = vec![root];
        let mut nodes = 0;
        let mut pruned_upper = best;
        while nodes < node_budget {
            let Some(node) = stack.pop() else {
                break;
            };
            nodes += 1;
            if node.upper <= best + tolerance {
                pruned_upper = pruned_upper.max(node.upper);
                continue;
            }
            if node.selected == self.size {
                let mass = self.value(&node.poly);
                if mass > best {
                    best = mass;
                    mask = node.mask;
                }
                continue;
            }
            if node.depth == groups.len() {
                continue;
            }
            let group = &groups[node.depth];
            let need = self.size - node.selected;
            let after = counts[node.depth + 1].iter().sum::<usize>();
            let lo = need.saturating_sub(after);
            let hi = need.min(group.ids.len());
            let mut children = Vec::new();
            for take in lo..=hi {
                let mut child = node.clone();
                child.depth += 1;
                child.selected += take;
                child.mask |= group.ids[..take].iter().fold(0, |m, i| m | (1 << i));
                for (eidx, e) in self.experts.iter().enumerate() {
                    let q = polynomial(std::iter::repeat_n(e.p[group.ids[0]], take), self.k);
                    let g = usize::from(group.previous);
                    child.poly[eidx][g] = product(&child.poly[eidx][g], &q, self.k);
                }
                if child.selected == self.size {
                    let mass = self.value(&child.poly);
                    if mass > best {
                        best = mass;
                        mask = child.mask;
                    }
                } else {
                    child.upper = bound(&child);
                    if child.upper > best + tolerance {
                        children.push(child);
                    } else {
                        pruned_upper = pruned_upper.max(child.upper);
                    }
                }
            }
            children.sort_by(|a, b| a.upper.total_cmp(&b.upper).then(b.mask.cmp(&a.mask)));
            stack.extend(children);
        }
        let best_before_canonicalization = best;
        mask = groups.iter().fold(0, |m, g| {
            let take = g.ids.iter().filter(|i| mask & (1 << **i) != 0).count();
            m | g.ids[..take].iter().fold(0, |b, i| b | (1 << i))
        });
        best = self.mass(mask)?;
        let upper = stack
            .iter()
            .map(|s| s.upper)
            .fold(best_before_canonicalization, f64::max)
            .max(best)
            .max(pruned_upper);
        let complete = upper <= best + tolerance;
        let tied = groups.iter().any(|g| {
            let used = g.ids.iter().filter(|i| mask & (1 << **i) != 0).count();
            used > 0 && used < g.ids.len()
        });
        Ok(Search{mask,mass:best,upper,nodes,groups:groups.len(),group_signatures:json!(groups.iter().map(|g|json!({"numbers":g.ids.iter().map(|i|i+1).collect::<Vec<_>>(),"previous":g.previous,"selected":g.ids.iter().filter(|i|mask&(1<<**i)!=0).count()})).collect::<Vec<_>>()),complete,tied_membership:tied})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn masks(n: usize, k: usize) -> Vec<u64> {
        (0..1u64 << n)
            .filter(|m| m.count_ones() as usize == k)
            .collect()
    }
    #[test]
    fn coverage_objective_and_search_match_independent_complete_outcome_enumeration() {
        let probs = vec![
            vec![1., 4., 7., 2., 5., 8., 3., 6.],
            vec![8., 3., 1., 7., 2., 4., 6., 5.],
        ];
        let weights = [0.4, 0.5, 0.1];
        assert!(
            (Coverage::new(8, 3, 8, 7, &weights, &probs)
                .unwrap()
                .mass(255)
                .unwrap()
                - 1.)
                .abs()
                < 1e-12
        );
        let model = Coverage::new(8, 3, 5, 7, &weights, &probs).unwrap();
        let mut best: f64 = 0.;
        for pool in masks(8, 5) {
            let mut mass = 0.;
            for outcome in masks(8, 3).into_iter().filter(|s| s & pool == *s) {
                let overlap = (outcome & 7).count_ones() as usize;
                let q = choose(3, overlap) * choose(5, 3 - overlap) / choose(8, 3);
                let mut p = weights[2] / choose(8, 3);
                for (e, w) in probs.iter().zip(weights) {
                    let allowed: Vec<_> = masks(8, 3)
                        .into_iter()
                        .filter(|s| (s & 7).count_ones() as usize == overlap)
                        .collect();
                    let raw = |s: u64| {
                        (0..8)
                            .filter(|i| s & (1 << i) != 0)
                            .map(|i| e[i])
                            .product::<f64>()
                    };
                    let z: f64 = allowed.iter().map(|s| raw(*s)).sum();
                    p += w * q * raw(outcome) / z;
                }
                mass += p;
            }
            assert!((model.mass(pool).unwrap() - mass).abs() < 1e-12);
            best = best.max(mass);
        }
        let r = model.optimize(10000).unwrap();
        assert!(r.complete);
        assert!((r.mass - best).abs() < 1e-12);
        assert!(r.upper + 1e-12 >= best);
    }
    #[test]
    fn bounded_run_reports_a_valid_upper_bound_instead_of_claiming_optimality() {
        let model = Coverage::new(
            9,
            3,
            5,
            7,
            &[0.4, 0.5, 0.1],
            &[
                vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                vec![9., 3., 7., 1., 6., 2., 5., 8., 4.],
            ],
        )
        .unwrap();
        let best = masks(9, 5)
            .iter()
            .map(|m| model.mass(*m).unwrap())
            .fold(0., f64::max);
        let r = model.optimize(1).unwrap();
        assert!(r.mass <= best + 1e-12);
        assert!(r.upper + 1e-12 >= best);
        assert!(r.nodes <= 1);
        let seeded = model.optimize_with_seeds(1, &[31]).unwrap();
        assert!(seeded.mass + 1e-12 >= model.mass(31).unwrap());
        assert!(seeded.upper + 1e-12 >= best);
    }
    #[test]
    fn nonuniform_identical_signatures_preserve_the_full_optimum() {
        let model = Coverage::new(
            9,
            3,
            5,
            7,
            &[0.4, 0.5, 0.1],
            &[
                vec![2., 2., 2., 1., 1., 4., 4., 7., 7.],
                vec![5., 5., 5., 8., 8., 2., 2., 1., 1.],
            ],
        )
        .unwrap();
        let best = masks(9, 5)
            .iter()
            .map(|m| model.mass(*m).unwrap())
            .fold(0., f64::max);
        let result = model.optimize(10000).unwrap();
        assert_eq!(result.groups, 4);
        assert!(result.complete);
        assert!((result.mass - best).abs() < 1e-12);
        assert!(result.upper + 1e-12 >= best);
    }
    #[test]
    fn uniform_distribution_has_no_number_preference_and_zero_gap() {
        let model = Coverage::new(9, 3, 5, 7, &[1.], &[]).unwrap();
        let r = model.optimize(1).unwrap();
        assert!(r.complete);
        assert!((r.mass - choose(5, 3) / choose(9, 3)).abs() < 1e-14);
        assert!(r.tied_membership);
    }
    #[test]
    fn best_marginals_need_not_maximize_complete_set_coverage() {
        let mut found = false;
        for seed in 1..30 {
            let a = (0..8)
                .map(|i| ((i * seed + 3) % 17 + 1) as f64)
                .collect::<Vec<_>>();
            let b = (0..8)
                .map(|i| ((i * i + seed * 3) % 19 + 1) as f64)
                .collect::<Vec<_>>();
            let m = Coverage::new(8, 3, 5, 7, &[0.45, 0.45, 0.1], &[a, b]).unwrap();
            let seed_mass = m.mass(m.marginal_seed()).unwrap();
            let best = m.optimize(10000).unwrap();
            if best.mass > seed_mass + 1e-8 {
                found = true;
                break;
            }
        }
        assert!(found);
    }
    #[test]
    fn invalid_models_and_pool_sizes_are_rejected() {
        assert!(Coverage::new(8, 3, 2, 7, &[1.], &[]).is_err());
        assert!(Coverage::new(8, 3, 5, 7, &[0.4], &[]).is_err());
        let m = Coverage::new(8, 3, 5, 7, &[1.], &[]).unwrap();
        assert!(m.optimize(0).is_err());
        assert!(m.mass(7).is_err());
    }
}
