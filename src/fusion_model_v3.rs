//! Experimental complete 6+1 selection. Frozen component fits are reused;
//! searches optimize complete, distinct outcomes rather than marginal exposure.
use crate::{
    blue_model_v2 as blue,
    draw::{data_sha256, Draw},
    model::roll_all,
    red_model_v2 as red,
};
use serde_json::{json, Value};
use std::{
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap},
    sync::{Mutex, OnceLock},
};

pub const TRAIN_END: u32 = 23100;
#[derive(Clone, Debug)]
pub struct Distribution {
    pub red: Vec<f64>,
    pub blue: Vec<f64>,
    z: f64,
}
fn elementary(w: &[f64], degree: usize) -> f64 {
    let mut a = vec![0.; degree + 1];
    a[0] = 1.;
    for v in w {
        for j in (1..=degree).rev() {
            a[j] += v * a[j - 1];
        }
    }
    a[degree]
}
#[derive(Clone, Debug)]
struct Cell {
    r: [usize; 6],
    b: usize,
    mass: f64,
    numbers: [u8; 6],
    number_blue: u8,
}
impl PartialEq for Cell {
    fn eq(&self, b: &Self) -> bool {
        self.cmp(b) == Ordering::Equal
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Cell {
    fn cmp(&self, b: &Self) -> Ordering {
        self.mass
            .total_cmp(&b.mass)
            .then_with(|| b.numbers.cmp(&self.numbers))
            .then_with(|| b.number_blue.cmp(&self.number_blue))
    }
}
#[derive(Clone, Debug)]
pub struct Selection {
    pub tickets: Vec<([u8; 6], u8)>,
    pub mass: f64,
    pub evaluated: usize,
}
impl Distribution {
    pub fn new(red: &[f64], blue: &[f64]) -> Result<Self, String> {
        if !(6..=33).contains(&red.len())
            || blue.is_empty()
            || blue.len() > 16
            || red.iter().chain(blue).any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("联合分布红蓝权重无效".into());
        }
        let max = red.iter().copied().fold(0., f64::max);
        let red: Vec<_> = red.iter().map(|v| v / max).collect();
        let sum = blue.iter().sum::<f64>();
        let blue: Vec<_> = blue.iter().map(|v| v / sum).collect();
        let z = elementary(&red, 6);
        if !z.is_finite() || z <= 0. || !sum.is_finite() {
            return Err("联合分布归一化失败".into());
        }
        Ok(Self { red, blue, z })
    }
    pub fn probability(&self, ns: &[u8], b: u8) -> Result<f64, String> {
        if ns.len() != 6
            || ns.iter().any(|n| *n == 0 || *n as usize > self.red.len())
            || ns.iter().collect::<BTreeSet<_>>().len() != 6
            || b == 0
            || b as usize > self.blue.len()
        {
            return Err("联合评分需要合法的完整6红1蓝".into());
        }
        Ok(ns
            .iter()
            .map(|n| self.red[(*n - 1) as usize])
            .product::<f64>()
            / self.z
            * self.blue[(b - 1) as usize])
    }
    pub fn red_order(&self) -> Vec<u8> {
        let mut ids: Vec<_> = (0..self.red.len()).collect();
        ids.sort_by(|a, b| self.red[*b].total_cmp(&self.red[*a]).then(a.cmp(b)));
        ids.iter().map(|i| *i as u8 + 1).collect()
    }
    pub fn blue_order(&self) -> Vec<u8> {
        let mut ids: Vec<_> = (0..self.blue.len()).collect();
        ids.sort_by(|a, b| self.blue[*b].total_cmp(&self.blue[*a]).then(a.cmp(b)));
        ids.iter().map(|i| *i as u8 + 1).collect()
    }
    pub fn pool_mass(&self, ns: &[u8]) -> Result<f64, String> {
        if ns.len() < 6
            || ns.iter().any(|n| *n == 0 || *n as usize > self.red.len())
            || ns.iter().collect::<BTreeSet<_>>().len() != ns.len()
        {
            return Err("完整入池评分号码无效".into());
        }
        Ok(elementary(
            &ns.iter()
                .map(|n| self.red[(*n - 1) as usize])
                .collect::<Vec<_>>(),
            6,
        ) / self.z)
    }
    pub fn top(&self, count: usize) -> Result<Selection, String> {
        if count == 0
            || count > 100
            || count as f64
                > crate::stats::binom(self.red.len() as u64, 6) as f64 * self.blue.len() as f64
        {
            return Err("联合票面预算无效".into());
        }
        let ro = self.red_order();
        let bo = self.blue_order();
        let cell = |r: [usize; 6], b: usize| -> Result<Cell, String> {
            let mut ns = r.map(|i| ro[i]);
            ns.sort_unstable();
            Ok(Cell {
                r,
                b,
                mass: self.probability(&ns, bo[b])?,
                numbers: ns,
                number_blue: bo[b],
            })
        };
        let start = [0, 1, 2, 3, 4, 5];
        let mut seen = BTreeSet::from([(start, 0usize)]);
        let mut heap = BinaryHeap::from([cell(start, 0)?]);
        let mut tickets = Vec::new();
        let mut mass = 0.;
        while tickets.len() < count {
            let c = heap.pop().ok_or("联合搜索前沿为空")?;
            tickets.push((c.numbers, c.number_blue));
            mass += c.mass;
            for axis in 0..7 {
                let mut r = c.r;
                let mut b = c.b;
                if axis == 6 {
                    b += 1;
                    if b >= bo.len() {
                        continue;
                    }
                } else {
                    r[axis] += 1;
                    if r[axis] >= ro.len() || (axis < 5 && r[axis] >= r[axis + 1]) {
                        continue;
                    }
                }
                if seen.insert((r, b)) {
                    heap.push(cell(r, b)?);
                }
            }
        }
        Ok(Selection {
            tickets,
            mass,
            evaluated: seen.len(),
        })
    }
    pub fn dantuo(
        &self,
        dan: usize,
        blues: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, Selection), String> {
        if ![(2, 2), (3, 1)].contains(&(dan, blues))
            || self.red.len() < dan + 5
            || self.blue.len() < blues
        {
            return Err("只支持2+5+2和3+5+1".into());
        }
        let ro = self.red_order();
        let bo = self.blue_order();
        let mut tickets = Vec::new();
        let mut mass = 0.;
        for c in crate::plan::combinations(&ro[dan..dan + 5], 6 - dan)? {
            let mut ns = ro[..dan].to_vec();
            ns.extend(c);
            ns.sort_unstable();
            let ns: [u8; 6] = ns.try_into().map_err(|_| "胆拖展开无效")?;
            for b in &bo[..blues] {
                mass += self.probability(&ns, *b)?;
                tickets.push((ns, *b));
            }
        }
        Ok((
            ro[..dan].to_vec(),
            ro[dan..dan + 5].to_vec(),
            Selection {
                evaluated: tickets.len(),
                tickets,
                mass,
            },
        ))
    }
}
#[derive(Clone)]
struct Fit {
    red: [f64; 6],
    blue: [f64; 5],
    training_sha: String,
    n: usize,
}
fn fit(draws: &[Draw], rows: &[crate::model::Row]) -> Result<Fit, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Fit)>>> = OnceLock::new();
    let training: Vec<_> = draws
        .iter()
        .filter(|d| d.issue <= TRAIN_END)
        .cloned()
        .collect();
    let sha = data_sha256(&training);
    let mut lock = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "联合拟合缓存异常")?;
    if let Some((key, f)) = &*lock {
        if key == &sha {
            return Ok(f.clone());
        }
    }
    let r = red::fit_causal(rows)?;
    let samples = rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= TRAIN_END)
        .map(|r| {
            Ok(blue::Sample {
                x: blue::features(&r.blue_p)?,
                winner: r.blue as usize,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let b = blue::fit(&samples)?;
    let result = Fit {
        red: r.beta,
        blue: b.beta,
        training_sha: sha.clone(),
        n: r.n,
    };
    *lock = Some((sha, result.clone()));
    Ok(result)
}
fn distribution(r: &[[f64; 33]; 7], b: &[[f64; 16]; 6], f: &Fit) -> Result<Distribution, String> {
    Distribution::new(
        &red::weights(&f.red, &red::features(r)?)?,
        &blue::distribution(&f.blue, &blue::features(b)?),
    )
}
pub fn policy() -> Value {
    json!({"version":3,"experimental":true,"prediction_advantage_established":false,"training_last_issue":TRAIN_END,"ridge":40,"objective":"sum_of_distinct_complete_6_plus_1_model_masses","red_model":"deduplicated_complete_six_red_v2","blue_model":"deduplicated_blue_v2","reference_use":"immutable_comparator_not_uncalibrated_vote_bonus","singles":{"bets":8,"cost":16,"search":"monotone_best_first_over_all_33_choose_6_and_16_blues"},"dantuo":["2+5+2","3+5+1"],"dantuo_search":"global_product_weight_exchange_optimum","model_relation":"factorized_red_blue_no_learned_conditional_dependence","automatic_promotion":false})
}
fn plan(id: &str, title: &str, s: Selection, dan: Vec<u8>, tuo: Vec<u8>) -> Value {
    let reds: BTreeSet<_> = s
        .tickets
        .iter()
        .flat_map(|(r, _)| r.iter().copied())
        .collect();
    let blues: BTreeSet<_> = s.tickets.iter().map(|(_, b)| *b).collect();
    json!({"id":id,"title":title,"mode":if dan.is_empty(){"joint"}else{"dantuo"},"bet_count":s.tickets.len(),"cost":s.tickets.len()*2,"red_count":reds.len(),"reds":reds,"blues":blues,"red_dan":dan,"red_tuo":tuo,"tickets":s.tickets.iter().map(|(r,b)|json!({"red":r,"blue":b})).collect::<Vec<_>>(),"model_joint_mass":s.mass,"ratio_to_uniform_budget":s.mass/(s.tickets.len()as f64/(1_107_568.*16.)),"certificate":{"global_model_optimum":true,"evaluations":s.evaluated,"absolute_search_gap":0.,"scope":"只针对本版本固定的可分解完整分布及所选买法；模型内最优不代表真实预测优势"}})
}
pub fn mass(model: &Distribution, tickets: &Value) -> Result<f64, String> {
    let mut seen = BTreeSet::new();
    let mut total = 0.;
    for t in tickets.as_array().ok_or("缺少完整票面")? {
        let r: Vec<u8> = serde_json::from_value(t["red"].clone()).map_err(|_| "红票无效")?;
        let b: u8 = serde_json::from_value(t["blue"].clone()).map_err(|_| "蓝票无效")?;
        let mut key = r.clone();
        key.sort_unstable();
        if !seen.insert((key, b)) {
            return Err("重复完整票面不能重复计入覆盖".into());
        }
        total += model.probability(&r, b)?;
    }
    Ok(total)
}
pub fn build(draws: &[Draw], parent: &Value, legacy: &Value) -> Result<Value, String> {
    let snap = roll_all(draws)?;
    let f = fit(draws, &snap.rows)?;
    let m = distribution(&snap.next_red, &snap.next_blue, &f)?;
    let mut plans = vec![plan(
        "joint8",
        "联合优化试验v3 · 8注",
        m.top(8)?,
        vec![],
        vec![],
    )];
    for (d, b) in [(2, 2), (3, 1)] {
        let (dan, tuo, s) = m.dantuo(d, b)?;
        plans.push(plan(
            &format!("{d}+5+{b}"),
            &format!("联合优化试验v3 · {d}+5+{b}"),
            s,
            dan,
            tuo,
        ));
    }
    for p in &mut plans {
        let old = if p["id"] == "joint8" {
            legacy["bundle"]["singles"]
                .as_array()
                .and_then(|a| a.iter().find(|p| p["red_count"] == 16))
        } else {
            legacy["bundle"]["dantuo"]
                .as_array()
                .and_then(|a| a.iter().find(|o| o["id"] == p["id"]))
        }
        .ok_or("缺少v2同预算首次对照")?;
        let original = mass(&m, &old["tickets"])?;
        p["same_budget_v2_comparison"] = json!({"id":old["id"],"bets":old["bet_count"],"cost":old["cost"],"model_joint_mass":original,"relative_model_gain":p["model_joint_mass"].as_f64().unwrap()/original-1.});
    }
    let mut records = Vec::new();
    for r in snap.rows.iter().filter(|r| r.issue > TRAIN_END) {
        let d = distribution(&r.expert_p, &r.blue_p, &f)?;
        let ns = crate::plan::mask_nums(r.red_mask);
        records.push((
            r.issue,
            (d.probability(&ns, r.blue + 1)? * 1_107_568. * 16.).ln(),
            d.red_order()[..16]
                .iter()
                .filter(|n| r.red_mask & (1u64 << (**n - 1)) != 0)
                .count(),
            d.blue_order()[..2].contains(&(r.blue + 1)),
        ));
    }
    let window = |lo, hi| {
        let rows: Vec<_> = records.iter().filter(|r| r.0 >= lo && r.0 <= hi).collect();
        json!({"n":rows.len(),"mean_joint_log_excess_vs_uniform":(!rows.is_empty()).then(||rows.iter().map(|r|r.1).sum::<f64>()/rows.len()as f64),"full_six_inside_16":rows.iter().filter(|r|r.2==6).count(),"blue_top2_hits":rows.iter().filter(|r|r.3).count(),"scope":"事后因果探索回放，不是本版本实际首次票面成绩"})
    };
    let reference = &parent["payload"]["reference_pools"]["reference"]["tickets"];
    let reference_mass = mass(&m, reference)?;
    let ro = m.red_order();
    Ok(
        json!({"version":3,"policy":policy(),"previous_reds":draws.last().ok_or("无开奖")?.red,"training":{"last_issue":TRAIN_END,"history_sha256":f.training_sha,"n":f.n,"red_coefficients":f.red,"blue_coefficients":f.blue},"ranking":ro,"blue_ranking":m.blue_order(),"blue_probabilities":m.blue,"candidate16_full_six_mass":m.pool_mass(&ro[..16])?,"plans":plans,"reference_five_comparator":{"bets":5,"cost":10,"model_joint_mass":reference_mass,"ratio_to_uniform_five":reference_mass/(5./(1_107_568.*16.)),"note":"原始5注保持原样；与8注或10注的预算不同，不能直接比较总质量"},"windows":{"validation":window(23101,25110),"confirmation_exploratory":window(25111,26115),"tracking":window(26116,u32::MAX)},"advantage_established":false,"note":"优化的是固定模型下完整票面的质量，历史完整评分未建立优于均匀的证据。本版本单独前瞻留档，不替换v2首次票面。参考ML没有足够同版本历史，保留原5注作为对照，不把并集支持直接当成概率。"}),
    )
}
pub fn reviewed(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut v = bundle.clone();
    for p in v["plans"].as_array_mut().ok_or("联合版本缺少方案")? {
        let r = crate::recommendations::review_strategy(
            &json!({"title":p["title"],"mode":p["mode"],"structural":true,"tickets":p["tickets"]}),
            draw,
        )?;
        p["summary"] = r["summary"].clone();
        p["ticket_results"] = r["tickets"].clone();
        p["single_jackpot"] = draw
            .map(|_| {
                r["tickets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["red_hits"] == 6 && t["blue_hit"] == true)
            })
            .into();
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn all_sets(n: usize, k: usize) -> Vec<Vec<u8>> {
        (0u64..1 << n)
            .filter(|m| m.count_ones() as usize == k)
            .map(|m| {
                (0..n)
                    .filter(|i| m & (1 << i) != 0)
                    .map(|i| i as u8 + 1)
                    .collect()
            })
            .collect()
    }
    #[test]
    fn monotone_joint_frontier_matches_independent_exhaustive_outcomes() {
        for reverse in [false, true] {
            let w: Vec<_> = (1..=10)
                .map(|i| if reverse { 11 - i } else { i })
                .map(|v| v as f64)
                .collect();
            let m = Distribution::new(&w, &[2., 1., 3., 4.]).unwrap();
            let mut oracle = Vec::new();
            for r in all_sets(10, 6) {
                for b in 1..=4 {
                    let ns: [u8; 6] = r.clone().try_into().unwrap();
                    oracle.push((ns, b, m.probability(&r, b).unwrap()));
                }
            }
            oracle.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
            let actual = m.top(8).unwrap();
            assert_eq!(
                actual.tickets,
                oracle[..8].iter().map(|r| (r.0, r.1)).collect::<Vec<_>>()
            );
            assert!((actual.mass - oracle[..8].iter().map(|r| r.2).sum::<f64>()).abs() < 1e-14);
            assert!(actual.evaluated < oracle.len());
            assert!((oracle.iter().map(|r| r.2).sum::<f64>() - 1.).abs() < 1e-13);
        }
        let m = Distribution::new(&[1.; 10], &[1.; 4]).unwrap();
        let a = m.top(8).unwrap();
        assert_eq!(a.tickets.len(), 8);
        assert_eq!(a.tickets.iter().collect::<BTreeSet<_>>().len(), 8);
        assert!((a.mass - 8. / (210. * 4.)).abs() < 1e-14);
    }
    #[test]
    fn dantuo_exchange_solution_matches_all_banker_drag_choices() {
        let m = Distribution::new(&[5., 1., 8., 3., 9., 2., 6., 4., 7.], &[1., 4., 2.]).unwrap();
        for (d, bn) in [(2, 2), (3, 1)] {
            let (_, _, actual) = m.dantuo(d, bn).unwrap();
            let mut best = 0f64;
            for dan in all_sets(9, d) {
                for tuo in all_sets(9, 5)
                    .into_iter()
                    .filter(|t| t.iter().all(|n| !dan.contains(n)))
                {
                    for bs in all_sets(3, bn) {
                        let mut total = 0.;
                        for r in all_sets(9, 6).into_iter().filter(|r| {
                            dan.iter().all(|n| r.contains(n))
                                && r.iter().all(|n| dan.contains(n) || tuo.contains(n))
                        }) {
                            for b in &bs {
                                total += m.probability(&r, *b).unwrap();
                            }
                        }
                        best = best.max(total);
                    }
                }
            }
            assert!((best - actual.mass).abs() < 1e-14);
            assert_eq!(actual.tickets.len(), 10);
        }
    }
    #[test]
    fn full_pool_mass_and_invalid_inputs_are_checked() {
        let m = Distribution::new(&[1.; 9], &[1.; 3]).unwrap();
        assert!((m.pool_mass(&[1, 2, 3, 4, 5, 6, 7]).unwrap() - 7. / 84.).abs() < 1e-14);
        assert!(m.probability(&[1, 2, 3, 4, 5, 5], 1).is_err());
        assert!(m.top(0).is_err());
        assert!(m.dantuo(2, 1).is_err());
        assert!(Distribution::new(&[0.; 9], &[1.; 3]).is_err());
    }
}
