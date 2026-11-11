//! Exhaustive optimization of distinct, complete 6+1 outcomes at a fixed budget.
//! The callback is a joint mass, so conditional-blue models are also supported.
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ticket {
    pub red: u64,
    pub blue: u8,
}
#[derive(Clone, Copy, Debug)]
struct Cell {
    ticket: Ticket,
    mass: f64,
}
impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.mass
            .total_cmp(&other.mass)
            .then_with(|| other.ticket.cmp(&self.ticket))
    }
}
pub struct Table {
    rows: BTreeMap<u64, [f64; 16]>,
    pub pool_mask: u64,
}
#[derive(Debug)]
pub struct Selection {
    pub tickets: Vec<Ticket>,
    pub mass: f64,
}
impl Table {
    pub fn new(pool: &[u8], mut joint: impl FnMut(u64, u8) -> f64) -> Result<Self, String> {
        if !(6..=16).contains(&pool.len())
            || pool.iter().any(|n| !(1..=33).contains(n))
            || pool.iter().collect::<BTreeSet<_>>().len() != pool.len()
        {
            return Err("完整票面搜索须有6–16个不同合法红球".into());
        }
        let mut rows = BTreeMap::new();
        // plan::combinations is a purchase expansion helper capped at 300.
        // Research must visit all C(16,6)=8008 red sets without that limit.
        let mut indices = [0, 1, 2, 3, 4, 5];
        loop {
            let mask = indices.iter().fold(0, |m, i| m | (1u64 << (pool[*i] - 1)));
            let values = std::array::from_fn(|b| joint(mask, b as u8 + 1));
            if values.iter().any(|p| !p.is_finite() || *p < 0. || *p > 1.) {
                return Err("完整票面概率无效".into());
            }
            rows.insert(mask, values);
            let Some(next) = (0..6).rev().find(|i| indices[*i] < pool.len() - 6 + *i) else {
                break;
            };
            indices[next] += 1;
            for i in next + 1..6 {
                indices[i] = indices[i - 1] + 1;
            }
        }
        if rows.values().flatten().sum::<f64>() > 1. + 1e-10 {
            return Err("候选内完整票面概率总和超过1".into());
        }
        Ok(Self {
            rows,
            pool_mask: pool.iter().fold(0, |m, n| m | (1u64 << (n - 1))),
        })
    }
    pub fn outcomes(&self) -> usize {
        self.rows.len() * 16
    }
    pub fn mass(&self, t: Ticket) -> Result<f64, String> {
        if !(1..=16).contains(&t.blue) {
            return Err("票面蓝球无效".into());
        }
        self.rows
            .get(&t.red)
            .map(|r| r[(t.blue - 1) as usize])
            .ok_or("完整六红不在候选池".into())
    }
    pub fn evaluate(&self, tickets: &[Ticket]) -> Result<Selection, String> {
        if tickets.is_empty() || tickets.iter().collect::<BTreeSet<_>>().len() != tickets.len() {
            return Err("完整票面为空或存在重复6＋1，不能重复计入覆盖质量".into());
        }
        let mass = tickets
            .iter()
            .map(|t| self.mass(*t))
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum();
        Ok(Selection {
            tickets: tickets.to_vec(),
            mass,
        })
    }
    /// Every cell denotes a mutually exclusive complete outcome. With equal
    /// ticket prices and no exposure constraint, the largest N cells are optimal.
    pub fn optimize(
        &self,
        count: usize,
        quotas: Option<&[usize; 16]>,
    ) -> Result<Selection, String> {
        if count == 0
            || count > self.outcomes()
            || quotas.is_some_and(|q| {
                q.iter().sum::<usize>() != count || q.iter().any(|n| *n > self.rows.len())
            })
        {
            return Err("完整票面搜索预算或每蓝注数无效".into());
        }
        let mut heaps: Vec<BinaryHeap<std::cmp::Reverse<Cell>>> =
            (0..if quotas.is_some() { 16 } else { 1 })
                .map(|_| BinaryHeap::new())
                .collect();
        for (&red, row) in &self.rows {
            for (b, mass) in row.iter().enumerate() {
                let capacity = quotas.map(|q| q[b]).unwrap_or(count);
                if capacity == 0 {
                    continue;
                }
                let h = &mut heaps[if quotas.is_some() { b } else { 0 }];
                let candidate = Cell {
                    ticket: Ticket {
                        red,
                        blue: b as u8 + 1,
                    },
                    mass: *mass,
                };
                if h.len() < capacity {
                    h.push(std::cmp::Reverse(candidate));
                } else if candidate > h.peek().unwrap().0 {
                    h.pop();
                    h.push(std::cmp::Reverse(candidate));
                }
            }
        }
        let mut cells: Vec<_> = heaps.into_iter().flatten().map(|c| c.0).collect();
        cells.sort_by(|a, b| b.cmp(a));
        self.evaluate(&cells.iter().map(|c| c.ticket).collect::<Vec<_>>())
    }
    pub fn optimize_blues_only(&self, tickets: &[Ticket]) -> Result<Selection, String> {
        self.evaluate(tickets)?;
        if tickets.iter().map(|t| t.red).collect::<BTreeSet<_>>().len() != tickets.len() {
            return Err("仅换蓝对照要求原六红组合各不相同".into());
        }
        let selected = tickets
            .iter()
            .map(|t| {
                (1..=16)
                    .map(|blue| Cell {
                        ticket: Ticket { red: t.red, blue },
                        mass: self.rows[&t.red][(blue - 1) as usize],
                    })
                    .max()
                    .unwrap()
                    .ticket
            })
            .collect::<Vec<_>>();
        self.evaluate(&selected)
    }
    pub fn total_mass(&self, blues: &[u8]) -> Result<f64, String> {
        if blues.is_empty()
            || blues.iter().any(|b| !(1..=16).contains(b))
            || blues.iter().collect::<BTreeSet<_>>().len() != blues.len()
        {
            return Err("蓝球选择为空、重复或越界".into());
        }
        Ok(self
            .rows
            .values()
            .map(|r| blues.iter().map(|b| r[(b - 1) as usize]).sum::<f64>())
            .sum())
    }
    /// Numerical rank interval, rather than a lucky numeric tie breaker. The
    /// tolerance is explicit and diagnostics do not change the actual selection.
    pub fn rank_interval(&self, t: Ticket, tolerance: f64) -> Result<(usize, usize), String> {
        if !tolerance.is_finite() || tolerance < 0. {
            return Err("同分容差无效".into());
        }
        let p = self.mass(t)?;
        let better = self
            .rows
            .values()
            .flatten()
            .filter(|x| **x > p + tolerance)
            .count();
        let tied = self
            .rows
            .values()
            .flatten()
            .filter(|x| (**x - p).abs() <= tolerance)
            .count();
        Ok((better + 1, better + tied))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn oracle(pool: &[u8], score: impl Fn(u64, u8) -> f64) -> Vec<(Ticket, f64)> {
        // Independent subset enumeration; does not use plan::combinations or heaps.
        let mut rows = Vec::new();
        for pattern in 0u64..1 << pool.len() {
            if pattern.count_ones() != 6 {
                continue;
            }
            let red = pool
                .iter()
                .enumerate()
                .filter(|(i, _)| pattern & (1 << i) != 0)
                .fold(0, |m, (_, n)| m | (1u64 << (n - 1)));
            for blue in 1..=16 {
                rows.push((Ticket { red, blue }, score(red, blue)));
            }
        }
        rows.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        rows
    }
    #[test]
    fn joint_heap_and_blue_quotas_match_independent_conditional_oracle() {
        let pool = [1, 3, 5, 7, 9, 11, 13, 15];
        let score = |r: u64, b: u8| (1 + ((r * 31 + b as u64 * 13) % 71)) as f64 / 100_000.;
        let table = Table::new(&pool, score).unwrap();
        let oracle = oracle(&pool, score);
        let best = table.optimize(8, None).unwrap();
        assert_eq!(
            best.tickets,
            oracle[..8].iter().map(|r| r.0).collect::<Vec<_>>()
        );
        assert!((best.mass - oracle[..8].iter().map(|r| r.1).sum::<f64>()).abs() < 1e-15);
        let mut quotas = [0; 16];
        quotas[2] = 3;
        quotas[10] = 5;
        let constrained = table.optimize(8, Some(&quotas)).unwrap();
        let mut expected = Vec::new();
        for b in [3, 11] {
            expected.extend(
                oracle
                    .iter()
                    .filter(|r| r.0.blue == b)
                    .take(quotas[(b - 1) as usize])
                    .copied(),
            );
        }
        assert!((constrained.mass - expected.iter().map(|r| r.1).sum::<f64>()).abs() < 1e-15);
        assert!(best.mass >= constrained.mass);
    }
    #[test]
    fn blue_only_preserves_red_sets_and_duplicate_outcomes_are_rejected() {
        let pool = [1, 2, 3, 4, 5, 6, 7];
        let table = Table::new(&pool, |r, b| ((r % 9 + b as u64) as f64) / 100_000.).unwrap();
        let original: Vec<_> = oracle(&pool, |_, _| 0.)
            .into_iter()
            .filter(|r| r.0.blue == 1)
            .take(5)
            .map(|r| r.0)
            .collect();
        let best = table.optimize_blues_only(&original).unwrap();
        assert_eq!(
            original.iter().map(|t| t.red).collect::<Vec<_>>(),
            best.tickets.iter().map(|t| t.red).collect::<Vec<_>>()
        );
        assert!(best.tickets.iter().all(|t| t.blue == 16));
        assert!(best.mass >= table.evaluate(&original).unwrap().mass);
        assert!(table.evaluate(&[original[0], original[0]]).is_err());
        assert!(table
            .optimize_blues_only(&[
                original[0],
                Ticket {
                    red: original[0].red,
                    blue: 2
                }
            ])
            .is_err());
    }
    #[test]
    fn uniform_ties_and_invalid_parameters_do_not_create_evidence() {
        let table = Table::new(&[1, 2, 3, 4, 5, 6, 7], |_, _| 1. / 112.).unwrap();
        assert!((table.optimize(8, None).unwrap().mass - 8. / 112.).abs() < 1e-15);
        assert_eq!(
            table
                .rank_interval(Ticket { red: 63, blue: 1 }, 1e-15)
                .unwrap(),
            (1, 112)
        );
        assert!(table.optimize(0, None).is_err());
        assert!(table.optimize(113, None).is_err());
        assert!(table.optimize(8, Some(&[0; 16])).is_err());
        assert!(Table::new(&[1, 2, 3, 4, 5, 5], |_, _| 0.).is_err());
        assert!(Table::new(&[1, 2, 3, 4, 5, 6], |_, _| f64::NAN).is_err());
        assert!(Table::new(&[1, 2, 3, 4, 5, 6], |_, _| 1.).is_err());
    }
    #[test]
    fn full_sixteen_pool_visits_every_red_set_and_all_blues_without_purchase_cap() {
        let pool: Vec<_> = (1..=16).collect();
        let table = Table::new(&pool, |_, _| 1. / (1_107_568. * 16.)).unwrap();
        assert_eq!(table.rows.len(), 8008);
        assert_eq!(table.outcomes(), 128128);
        assert!(
            (table.total_mass(&(1..=16).collect::<Vec<_>>()).unwrap() - 8008. / 1_107_568.).abs()
                < 1e-13
        );
        assert!((table.optimize(8, None).unwrap().mass - 8. / (1_107_568. * 16.)).abs() < 1e-18);
    }
}
