//! Deterministic maximization of complete six-red mass under a fixed dantuo
//! shape. Scores are supplied by the caller; this is not a prediction model.
use serde_json::{json, Value};

pub fn subsets(n: usize, k: usize) -> Vec<u16> {
    if k > n || n > 16 {
        return vec![];
    }
    if k == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut m = (1u32 << k) - 1;
    while m < 1u32 << n {
        out.push(m as u16);
        let low = m & m.wrapping_neg();
        let next = m + low;
        m = next | (((m ^ next) >> 2) / low);
    }
    out
}
fn positions(mask: u16) -> Vec<usize> {
    (0..16).filter(|i| mask & (1 << i) != 0).collect()
}
fn scatter(pattern: u16, ids: &[usize]) -> u16 {
    ids.iter().enumerate().fold(0, |m, (i, p)| {
        m | if pattern & (1 << i) != 0 { 1 << p } else { 0 }
    })
}
#[derive(Clone, Debug)]
pub struct Choice {
    pub dan: u16,
    pub tuo: u16,
    /// Sum of complete six-set scores. No marginal, exposure or union score.
    pub mass: f64,
}
pub struct Scores {
    pub pool: Vec<u8>,
    pub values: Vec<f64>,
}
impl Scores {
    pub fn new(pool: &[u8], mut score: impl FnMut(u64) -> f64) -> Result<Self, String> {
        if !(8..=16).contains(&pool.len())
            || pool.iter().any(|n| !(1..=33).contains(n))
            || pool.iter().collect::<std::collections::BTreeSet<_>>().len() != pool.len()
        {
            return Err("胆拖搜索需要8至16个互异有效红球".into());
        }
        let mut values = vec![0.; 1 << pool.len()];
        for m in subsets(pool.len(), 6) {
            let global = pool.iter().enumerate().fold(0, |v, (i, n)| {
                v | if m & (1 << i) != 0 {
                    1u64 << (n - 1)
                } else {
                    0
                }
            });
            let p = score(global);
            if !p.is_finite() || p <= 0. {
                return Err("完整六红评分必须为有限正数".into());
            }
            values[m as usize] = p;
        }
        Ok(Self {
            pool: pool.to_vec(),
            values,
        })
    }
    pub fn choice(&self, dan: u16, tuo: u16) -> Result<Choice, String> {
        let d = dan.count_ones() as usize;
        if ![2, 3].contains(&d)
            || tuo.count_ones() != 5
            || dan & tuo != 0
            || (dan | tuo) as usize >= 1 << self.pool.len()
        {
            return Err("只支持互不重复的2胆5拖或3胆5拖".into());
        }
        let ids = positions(tuo);
        let mass = subsets(5, 6 - d)
            .into_iter()
            .map(|m| self.values[(dan | scatter(m, &ids)) as usize])
            .sum();
        Ok(Choice { dan, tuo, mass })
    }
    pub fn prefix(&self, dan: usize) -> Result<Choice, String> {
        if ![2, 3].contains(&dan) {
            return Err("胆码数无效".into());
        }
        self.choice((1 << dan) - 1, ((1 << 5) - 1) << dan)
    }
    pub fn numbers(&self, mask: u16) -> Vec<u8> {
        let mut out: Vec<_> = positions(mask).into_iter().map(|i| self.pool[i]).collect();
        out.sort_unstable();
        out
    }
    pub fn expanded(&self, choice: &Choice) -> Vec<Vec<u8>> {
        let ids = positions(choice.tuo);
        subsets(5, 6 - choice.dan.count_ones() as usize)
            .into_iter()
            .map(|m| self.numbers(choice.dan | scatter(m, &ids)))
            .collect()
    }
    pub fn union_mass(&self, choice: &Choice) -> f64 {
        let ids = positions(choice.dan | choice.tuo);
        subsets(ids.len(), 6)
            .into_iter()
            .map(|m| self.values[scatter(m, &ids) as usize])
            .sum()
    }
    pub fn optimize(&self, dan: usize) -> Result<(Choice, Value), String> {
        if ![2, 3].contains(&dan) {
            return Err("胆码数无效".into());
        }
        let mut best = self.prefix(dan)?;
        let unions = subsets(self.pool.len(), dan + 5);
        let mut examined = 0usize;
        let mut consider = |choice: Choice| {
            examined += 1;
            if choice.mass > best.mass
                || (choice.mass == best.mass && (choice.dan, choice.tuo) < (best.dan, best.tuo))
            {
                best = choice;
            }
        };
        if dan == 2 {
            // In a seven-red union every six-set omits one red. Bankers are the
            // two omissions excluded from the five legal sets. Exclude the two
            // smallest weights: exact for this union, without testing 21 pairs.
            for union in &unions {
                let ids = positions(*union);
                let mut omissions: Vec<_> = ids
                    .iter()
                    .map(|i| (self.values[(union ^ (1 << i)) as usize], *i))
                    .collect();
                omissions.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let banks = (1 << omissions[0].1) | (1 << omissions[1].1);
                consider(self.choice(banks, union ^ banks)?);
            }
        } else {
            let patterns = subsets(8, 5);
            for union in &unions {
                let ids = positions(*union);
                let mut edges = [[0.; 8]; 8];
                for i in 0..8 {
                    for j in i + 1..8 {
                        edges[i][j] = self.values[(union ^ (1 << ids[i]) ^ (1 << ids[j])) as usize];
                    }
                }
                // A six-set omits two reds. Its omission pair must both be
                // drags, so the legal ten sets form a five-vertex weighted clique.
                for pattern in &patterns {
                    let drag_ids: Vec<_> = (0..8).filter(|i| pattern & (1 << i) != 0).collect();
                    let mut mass = 0.;
                    for (i, a) in drag_ids.iter().enumerate() {
                        for b in &drag_ids[i + 1..] {
                            mass += edges[*a][*b];
                        }
                    }
                    let tuo = scatter(*pattern, &ids);
                    consider(Choice {
                        dan: union ^ tuo,
                        tuo,
                        mass,
                    });
                }
            }
        }
        let variants_per_union = crate::stats::binom((dan + 5) as u64, dan as u64);
        let certificate = json!({"red_pool_size":self.pool.len(),"complete_six_scores":crate::stats::binom(self.pool.len() as u64,6),
            "unions_examined":unions.len(),"all_legal_red_plans":unions.len() as u64*variants_per_union,"objectives_evaluated":examined,
            "within_pool_global_optimum":true,"floating_point":"f64; deterministic; independent brute force tests",
            "algorithm":if dan==2{"enumerate_all_7_red_unions_then_exclude_two_minimum_omission_weights"}else{"enumerate_all_8_red_unions_and_all_56_five_drag_cliques"},
            "scope":"同一候选池和给定完整六红评分下最优，不是全33红最优或真实中奖概率保证"});
        Ok((best, certificate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn oracle(s: &Scores, dan: usize) -> f64 {
        let mut best: f64 = 0.;
        for d in subsets(s.pool.len(), dan) {
            for t in subsets(s.pool.len(), 5).into_iter().filter(|t| t & d == 0) {
                // Independent direct membership oracle: enumerate every global
                // six-set instead of reusing choice(), scatter() or clique sums.
                let mass = subsets(s.pool.len(), 6)
                    .into_iter()
                    .filter(|m| m & d == d && m & !(d | t) == 0)
                    .map(|m| s.values[m as usize])
                    .sum::<f64>();
                best = best.max(mass);
            }
        }
        best
    }
    #[test]
    fn reduced_search_matches_independent_all_banker_drag_enumeration() {
        for seed in [3u64, 91, 2026] {
            let s = Scores::new(&(1..=9).collect::<Vec<_>>(), |m| {
                // Nonseparable arbitrary complete-set weights, not node products.
                ((m.wrapping_mul(seed + 12345) ^ (m >> 2)) % 10007 + 1) as f64
            })
            .unwrap();
            for d in [2, 3] {
                let (b, _) = s.optimize(d).unwrap();
                assert_eq!(b.mass, oracle(&s, d));
                assert_eq!(b.mass, s.choice(b.dan, b.tuo).unwrap().mass);
                assert!(b.mass >= s.prefix(d).unwrap().mass);
            }
        }
    }
    #[test]
    fn independent_product_already_has_prefix_optimum() {
        let s = Scores::new(&(1..=9).collect::<Vec<_>>(), |m| {
            (0..9)
                .filter(|i| m & (1 << i) != 0)
                .map(|i| (10 - i) as f64)
                .product()
        })
        .unwrap();
        for d in [2, 3] {
            assert_eq!(s.optimize(d).unwrap().0.mass, s.prefix(d).unwrap().mass);
        }
    }
    #[test]
    fn uniform_mass_and_bankers_explain_union_vs_full_tickets() {
        let s = Scores::new(&(1..=16).collect::<Vec<_>>(), |_| 1.).unwrap();
        for (d, legal, unions, plans) in [(2, 5, 11440, 240240), (3, 10, 12870, 720720)] {
            let (b, c) = s.optimize(d).unwrap();
            assert_eq!(b.mass, legal as f64);
            assert_eq!(c["unions_examined"], unions);
            assert_eq!(c["all_legal_red_plans"], plans);
            let expanded = s.expanded(&b);
            assert_eq!(expanded.len(), legal);
            for t in expanded {
                assert_eq!(t.len(), 6);
                assert!(s.numbers(b.dan).iter().all(|n| t.contains(n)));
            }
            assert_eq!(s.union_mass(&b), if d == 2 { 7. } else { 28. });
        }
    }
    #[test]
    fn invalid_scores_or_pool_or_shape_are_rejected() {
        assert!(Scores::new(&[1; 8], |_| 1.).is_err());
        assert!(Scores::new(&(1..=9).collect::<Vec<_>>(), |_| f64::NAN).is_err());
        let s = Scores::new(&(1..=9).collect::<Vec<_>>(), |_| 1.).unwrap();
        assert!(s.optimize(4).is_err());
        assert!(s.choice(3, 31).is_err());
        assert!(s.choice(3, 31 << 5).is_err());
    }
    #[test]
    fn full_union_with_a_wrong_banker_has_no_exact_six_ticket() {
        let s = Scores::new(&(1..=16).collect::<Vec<_>>(), |_| 1.).unwrap();
        for (dan, actual) in [(2, vec![2, 3, 4, 5, 6, 7]), (3, vec![2, 3, 4, 5, 6, 7])] {
            let c = s.prefix(dan).unwrap();
            let union = s.numbers(c.dan | c.tuo);
            assert!(actual.iter().all(|n| union.contains(n)));
            assert!(!s.expanded(&c).contains(&actual));
        }
    }
}
