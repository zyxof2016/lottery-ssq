//! Research-only normalized six-red set models and proper-score audit.
//! This module never publishes, changes a frozen forecast, or promotes a model.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::prepare_forecast,
    hash::sha256_hex,
    model::roll_all,
    plan::{mask_nums, PlanTicket},
    protocol::Protocol,
    select::{fuse6, fuse7, rank_desc, rank_log_score, softmax_shrink, Forecast, RescueResult},
};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Mutex, OnceLock},
};
const UNIVERSE: u64 = 1_107_568;
const MODELS: [&str; 3] = ["weighted_product_proxy", "joint_mixture", "uniform"];
#[derive(Clone, Debug)]
struct SetModel {
    w: Vec<f64>,
    k: usize,
    z: f64,
    inclusion: Vec<f64>,
}
impl SetModel {
    fn new(raw: &[f64], k: usize) -> Result<Self, String> {
        if raw.len() > 63
            || k == 0
            || k > raw.len()
            || raw.iter().any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err("组合模型权重须为有限正数且数量有效".into());
        }
        let scale = raw.iter().copied().fold(0., f64::max);
        let w: Vec<_> = raw.iter().map(|x| x / scale).collect();
        let n = w.len();
        let stride = k + 1;
        let mut pre = vec![0.; (n + 1) * stride];
        let mut post = pre.clone();
        pre[0] = 1.;
        post[n * stride] = 1.;
        for i in 0..n {
            pre[(i + 1) * stride] = 1.;
            for j in 1..=k {
                pre[(i + 1) * stride + j] = pre[i * stride + j] + w[i] * pre[i * stride + j - 1];
            }
        }
        for i in (0..n).rev() {
            post[i * stride] = 1.;
            for j in 1..=k {
                post[i * stride + j] =
                    post[(i + 1) * stride + j] + w[i] * post[(i + 1) * stride + j - 1];
            }
        }
        let z = pre[n * stride + k];
        if !z.is_finite() || z <= 0. {
            return Err("组合归一化失败".into());
        }
        let inclusion = (0..n)
            .map(|i| {
                w[i] * (0..k)
                    .map(|j| pre[i * stride + j] * post[(i + 1) * stride + k - 1 - j])
                    .sum::<f64>()
                    / z
            })
            .collect();
        Ok(Self { w, k, z, inclusion })
    }
    fn logp(&self, mask: u64) -> f64 {
        if mask.count_ones() as usize != self.k || mask >> self.w.len() != 0 {
            return f64::NEG_INFINITY;
        }
        let mut bits = mask;
        let mut p = -self.z.ln();
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            p += self.w[i].ln();
            bits &= bits - 1;
        }
        p
    }
}
#[derive(Clone, Debug, Default, PartialEq)]
struct Fit {
    sum: [f64; 8],
    n: usize,
    last: Option<u32>,
}
impl Fit {
    fn weights(&self) -> [f64; 8] {
        let best = self.sum.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e = self.sum.map(|v| (v - best).exp());
        let total = e.iter().sum::<f64>();
        e.map(|v| v / total)
    }
    fn observe(&mut self, issue: u32, lp: &[f64; 8]) {
        if issue >= Protocol::CONFIRM_START_ISSUE {
            return;
        }
        for (e, v) in self.sum.iter_mut().zip(lp) {
            *e += v;
        }
        self.n += 1;
        self.last = Some(issue);
    }
}
fn models(experts: &[[f64; 33]; 7]) -> Result<Vec<SetModel>, String> {
    experts
        .iter()
        .map(|p| SetModel::new(p, 6))
        .chain(std::iter::once(SetModel::new(&[1.; 33], 6)))
        .collect()
}
fn mixture_logp(models: &[SetModel], weights: &[f64; 8], mask: u64) -> f64 {
    let terms: Vec<_> = models
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.)
        .map(|(m, w)| w.ln() + m.logp(mask))
        .collect();
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + terms.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}
fn mixture_rank(models: &[SetModel], weights: &[f64; 8]) -> [usize; 33] {
    let scores = std::array::from_fn(|i| {
        models
            .iter()
            .zip(weights)
            .map(|(m, w)| w * m.inclusion[i] / 6.)
            .sum()
    });
    rank_desc(&scores)
}
fn templates() -> &'static Vec<[usize; 6]> {
    static T: OnceLock<Vec<[usize; 6]>> = OnceLock::new();
    T.get_or_init(|| {
        let mut out = Vec::new();
        for a in 0..11 {
            for b in a + 1..12 {
                for c in b + 1..13 {
                    for d in c + 1..14 {
                        for e in d + 1..15 {
                            for f in e + 1..16 {
                                out.push([a, b, c, d, e, f]);
                            }
                        }
                    }
                }
            }
        }
        out
    })
}
fn best_sets(models: &[SetModel], weights: &[f64; 8], pool: &[usize]) -> Vec<u64> {
    let active: Vec<_> = models
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.)
        .map(|(m, w)| (m, *w / m.z))
        .collect();
    let mut best: Vec<(i64, [usize; 6])> = Vec::new();
    for t in templates() {
        let p = active
            .iter()
            .map(|(m, c)| c * t.iter().map(|i| m.w[pool[*i]]).product::<f64>())
            .sum::<f64>();
        let key = (p.max(1e-300).ln() * 1e12).round() as i64;
        if best.len() < 4 || key > best[3].0 {
            best.push((key, *t));
            best.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            best.truncate(4);
        }
    }
    best.iter()
        .map(|(_, c)| c.iter().fold(0, |m, i| m | (1 << pool[*i])))
        .collect()
}
fn pairs(sets: &[u64], blues: &[usize]) -> Vec<PlanTicket> {
    sets.iter()
        .flat_map(|s| {
            blues.iter().map(move |b| PlanTicket {
                red: mask_nums(*s),
                blue: (b + 1) as u8,
            })
        })
        .collect()
}
#[derive(Clone, Debug, PartialEq)]
struct Period {
    issue: u32,
    red_excess: [f64; 3],
    joint_excess: [f64; 3],
    pool_hits: [bool; 2],
    exact: [bool; 2],
    head: [bool; 2],
    copy_previous: [bool; 2],
    red_union: [usize; 2],
}
fn full(mask: u64, tickets: &[PlanTicket], blue: u8) -> (bool, bool) {
    let exact = tickets
        .iter()
        .any(|t| t.red.iter().fold(0, |m, n| m | (1 << (*n - 1))) == mask);
    let head = tickets
        .iter()
        .any(|t| t.blue == blue && t.red.iter().fold(0, |m, n| m | (1 << (*n - 1))) == mask);
    (exact, head)
}
fn rows(draws: &[Draw]) -> Result<(Vec<Period>, Fit, [f64; 7], [f64; 6], usize), String> {
    let snap = roll_all(draws)?;
    let (mut fit, mut rs, mut bs, mut n) = (Fit::default(), [0.; 7], [0.; 6], 0usize);
    let mut result = Vec::new();
    let previous: std::collections::BTreeMap<_, _> = draws
        .windows(2)
        .map(|w| (w[1].issue, w[0].red_mask))
        .collect();
    let uniform_log = -(UNIVERSE as f64).ln();
    for row in snap.rows {
        let ms = models(&row.expert_p)?;
        let expert_lp: [f64; 8] = std::array::from_fn(|i| ms[i].logp(row.red_mask));
        if n >= 30 {
            let old_scores = fuse7(&row.expert_p, &softmax_shrink(&rs, n as f64));
            let old_order = rank_desc(&old_scores);
            let old_dist = SetModel::new(&old_scores, 6)?;
            let blue = fuse6(&row.blue_p, &softmax_shrink(&bs, n as f64));
            let bo = rank_desc(&blue);
            let weights = fit.weights();
            let order = mixture_rank(&ms, &weights);
            let fc = Forecast {
                fusion: old_scores,
                red_order: old_order,
                blue,
                blue_order: bo,
                mild: RescueResult {
                    mask: 0,
                    members: vec![],
                    rescued: vec![],
                    displaced: vec![],
                },
                pool: vec![],
                tickets: vec![],
                ticket_blues: vec![],
            };
            let old = crate::jackpot::build(&fc, &crate::jackpot::spec())?;
            let candidate = pairs(&best_sets(&ms, &weights, &order[..16]), &bo[..2]);
            let old_result = full(row.red_mask, &old.tickets, row.blue + 1);
            let new_result = full(row.red_mask, &candidate, row.blue + 1);
            let mut red_excess = [0.; 3];
            red_excess[0] = old_dist.logp(row.red_mask) - uniform_log;
            red_excess[1] = mixture_logp(&ms, &weights, row.red_mask) - uniform_log;
            let blue_excess = blue[row.blue as usize].ln() + 16f64.ln();
            let joint_excess = [red_excess[0] + blue_excess, red_excess[1] + blue_excess, 0.];
            let pm = previous[&row.issue];
            let mask = |order: &[usize]| order.iter().fold(0, |m, n| m | (1 << n));
            let red_union =
                |ts: &[PlanTicket]| ts.iter().flat_map(|t| t.red).collect::<BTreeSet<_>>().len();
            let copy = |ts: &[PlanTicket]| {
                ts.iter()
                    .any(|t| t.red.iter().fold(0, |m, n| m | (1 << (*n - 1))) == pm)
            };
            result.push(Period {
                issue: row.issue,
                red_excess,
                joint_excess,
                pool_hits: [
                    row.red_mask & mask(&old_order[..16]) == row.red_mask,
                    row.red_mask & mask(&order[..16]) == row.red_mask,
                ],
                exact: [old_result.0, new_result.0],
                head: [old_result.1, new_result.1],
                copy_previous: [copy(&old.tickets), copy(&candidate)],
                red_union: [red_union(&old.tickets), red_union(&candidate)],
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for (e, v) in rs.iter_mut().enumerate() {
                *v += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            for (e, v) in bs.iter_mut().enumerate() {
                *v += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
            }
            n += 1;
        }
        fit.observe(row.issue, &expert_lp);
    }
    Ok((result, fit, rs, bs, n))
}
fn summarize(rows: &[Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells: Vec<_> = MODELS.iter().enumerate().map(|(j, name)| {
        let sum = |joint: bool| rows.iter().map(|r| if joint {r.joint_excess[j]} else {r.red_excess[j]}).sum::<f64>() / rows.len() as f64;
        json!({
            "model":name,
            "label":(["原乘积规则·归一化审计","完整组合混合·研究","均匀基准"][j]),
            "n":rows.len(),
            "mean_red_log_excess_vs_uniform":sum(false),
            "mean_joint_log_excess_vs_uniform":sum(true),
            "candidate_full_six_periods":if j<2{json!(rows.iter().filter(|r|r.pool_hits[j]).count())}else{Value::Null},
            "single_six_red_periods":if j<2{json!(rows.iter().filter(|r|r.exact[j]).count())}else{Value::Null},
            "jackpot_periods":if j<2{json!(rows.iter().filter(|r|r.head[j]).count())}else{Value::Null},
            "copied_previous_six_periods":if j<2{json!(rows.iter().filter(|r|r.copy_previous[j]).count())}else{Value::Null},
            "mean_actual_red_union":if j<2{json!(rows.iter().map(|r|r.red_union[j]).sum::<usize>()as f64/rows.len()as f64)}else{Value::Null}
        })
    }).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"cells":cells})
}
fn policy() -> Value {
    json!({"version":1,"objective":"proper_log_score_of_normalized_complete_six_red_sets","set_distribution":"product_of_positive_weights_divided_by_elementary_symmetric_polynomial_order6","experts":"seven_original_causal_experts_plus_uniform","mixture":"equal_prior_bayesian_model_weights_from_pre_boundary_log_likelihood","training_freeze_before_issue":Protocol::CONFIRM_START_ISSUE,"minimum_evaluation_fit_periods":30,"forced_expert_weight_floor":false,"red_limit":16,"red_sets":4,"blue_count":2,"bets":8,"cost":16,"auto_promote":false,"publication_allowed":false,"required_prospective_periods":156,"score_meaning":"model_assumption_probability_not_verified_winning_probability","note":"完整组合按全部C(33,6)种结果归一化，并用对数评分与均匀基准比较；新混合不强制给弱专家最低权重。仅研究，不替换当前原档，不宣称提高头奖。"})
}
fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let (result, fit, _, bs, n) = rows(draws)?;
    let start = result
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(result.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(result.len());
    let snap = roll_all(draws)?;
    let next_models = models(&snap.next_red)?;
    let weights = fit.weights();
    let order = mixture_rank(&next_models, &weights);
    let blues = fuse6(&snap.next_blue, &softmax_shrink(&bs, n as f64));
    let bo = rank_desc(&blues);
    let tickets = pairs(&best_sets(&next_models, &weights, &order[..16]), &bo[..2]);
    let preview = json!({"title":"联合评分研究候选·未发布","mode":"research","structural":true,"tickets":tickets.iter().map(|t|json!({"red":t.red,"blue":t.blue})).collect::<Vec<_>>()});
    let train = json!({"n":fit.n,"last_issue":fit.last.map(fmt_issue),"weights":weights,"expert_names":["frequency","ewma","overdue","hot","cooccur","zone","repeat","uniform"],"expert_cumulative_red_log_scores":fit.sum});
    let tracking = summarize(&result[end..]);
    let means = tracking["cells"]
        .as_array()
        .and_then(|v| v.get(1))
        .map(|c| c["mean_joint_log_excess_vs_uniform"].as_f64().unwrap_or(0.));
    Ok(
        json!({"version":1,"kind":"joint_red_scoring_research","data_sha256":data_sha256(draws),"cutoff_issue":draws.last().map(|d|fmt_issue(d.issue)),"target_issue":draws.last().map(|d|fmt_issue(d.issue+1)),"policy":policy(),"training":train,"windows":{"development":summarize(&result[..start]),"confirmation_exploratory":summarize(&result[start..end]),"tracking":tracking,"recent30":summarize(&result[result.len().saturating_sub(30)..]),"recent60":summarize(&result[result.len().saturating_sub(60)..]),"recent156":summarize(&result[result.len().saturating_sub(156)..]),"all_oos":summarize(&result)},"adoption":{"replace_current_recommendation":false,"prediction_advantage_established":false,"historical_tracking_n":result.len()-end,"historical_tracking_joint_score_positive":means.is_some_and(|v|v>0.),"new_prospective_n":0,"minimum_new_prospective_periods":156,"reason":"新模型缺少独立前瞻验证，且须先超过均匀完整组合基准；不替换当前已封存号码。历史窗口已被查看，仅作探索。"},"candidate":{"status":"research_preview","actual_recommendation":false,"publication_allowed":false,"candidate_reds":order[..16].iter().map(|i|i+1).collect::<Vec<_>>(),"plan":crate::recommendations::review_strategy(&preview,None)?},"source_sha256":sha256_hex(concat!(include_str!("joint_red.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("protocol.rs")).as_bytes()),"method_sources":["https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf","https://arxiv.org/abs/2012.03103"],"note":"原规则的乘积仅为审计而归一化，旧发布评分并非已校准概率。所有比较逐期只使用此前数据，边界后权重冻结；数学归一化和评分改善不等于头奖能力提升。"}),
    )
}
fn health(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue + 1);
    let current = crate::jackpot::view(root, draws, &issue)?;
    let (tickets, source) = if let Some(v) = current {
        (
            v["plan"]["tickets"].clone(),
            json!({"status":"frozen","archive_sha256":v["archive_sha256"],"raw_file":v["raw_file"]}),
        )
    } else {
        let p = prepare_forecast(draws)?;
        let b = crate::jackpot::build(&p.forecast, &crate::jackpot::spec())?;
        (
            json!(b
                .tickets
                .iter()
                .map(|t| json!({"red":t.red,"blue":t.blue}))
                .collect::<Vec<_>>()),
            json!({"status":"preview"}),
        )
    };
    let sets: BTreeSet<Vec<u8>> = tickets
        .as_array()
        .ok_or("缺少原始票面")?
        .iter()
        .map(|t| serde_json::from_value(t["red"].clone()).map_err(|_| "原始红球无效".to_string()))
        .collect::<Result<_, _>>()?;
    let union: BTreeSet<_> = sets.iter().flat_map(|s| s.iter().copied()).collect();
    let overlap: Vec<_> = sets
        .iter()
        .map(|s| s.iter().filter(|n| last.red.contains(n)).count())
        .collect();
    let copied = overlap.iter().filter(|n| **n == 6).count();
    Ok(
        json!({"issue":issue,"latest_draw":{"issue":fmt_issue(last.issue),"red":last.red,"blue":last.blue},"source":source,"outcome_known":false,"status":"pending","complete_red_sets":sets.len(),"actual_red_union":union.len(),"copied_previous_six_sets":copied,"previous_red_overlap_per_set":overlap,"common_reds_in_all_sets":sets.iter().next().map(|s|s.iter().filter(|n|sets.iter().all(|t|t.contains(n))).copied().collect::<Vec<_>>()),"note":"检查原始封存票面，不改写号码。尚无本期入库结果，不计算成败。复制上期号码说明评分集中，不等于下一期必然不会重复。"}),
    )
}
pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let key = data_sha256(draws);
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "联合评分缓存锁异常")?;
    if cache.as_ref().is_none_or(|(k, _)| *k != key) {
        *cache = Some((key, analyze(draws)?));
    }
    let mut value = cache.as_ref().unwrap().1.clone();
    drop(cache);
    value["current_health"] = health(root, draws)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-11, "{a} != {b}");
    }
    #[test]
    fn normalization_and_marginals_match_independent_exhaustive_oracle() {
        for k in [1, 3, 6, 8] {
            let raw = [0.1, 0.8, 0.2, 1.7, 0.5, 2.1, 0.01, 0.3];
            let model = SetModel::new(&raw, k).unwrap();
            let all: Vec<_> = (0..256u64)
                .filter(|m| m.count_ones() as usize == k)
                .collect();
            let product = |m: u64| {
                (0..8)
                    .filter(|i| m & (1 << i) != 0)
                    .map(|i| raw[i])
                    .product::<f64>()
            };
            let total = all.iter().map(|m| product(*m)).sum::<f64>();
            let mut sum = 0.;
            let mut inc = [0.; 8];
            for m in all {
                let p = model.logp(m).exp();
                close(p, product(m) / total);
                sum += p;
                for (i, v) in inc.iter_mut().enumerate() {
                    if m & (1 << i) != 0 {
                        *v += p;
                    }
                }
            }
            close(sum, 1.);
            close(model.inclusion.iter().sum(), k as f64);
            for (a, b) in inc.iter().zip(&model.inclusion) {
                close(*a, *b);
            }
            let scaled = SetModel::new(&raw.map(|v| v * 123.4), k).unwrap();
            for m in (0..256u64).filter(|m| m.count_ones() as usize == k) {
                close(model.logp(m), scaled.logp(m));
            }
        }
        let uniform = SetModel::new(&[1.; 33], 6).unwrap();
        close(uniform.logp(63), -(UNIVERSE as f64).ln());
        for p in uniform.inclusion {
            close(p, 6. / 33.);
        }
        assert!(SetModel::new(&[0., 1.], 1).is_err());
        assert!(SetModel::new(&[f64::NAN, 1.], 1).is_err());
        assert!(SetModel::new(&[1.; 64], 6).is_err());
        assert_eq!(
            SetModel::new(&[1.; 8], 6).unwrap().logp(1 << 9 | 31),
            f64::NEG_INFINITY
        );
    }
    #[test]
    fn full_set_likelihood_suppresses_bad_experts_and_freezes_before_boundary() {
        let mut fit = Fit::default();
        assert_eq!(fit.weights(), [0.125; 8]);
        fit.observe(25109, &[-10., -11., -12., -13., -14., -15., -100., -9.]);
        let w = fit.weights();
        close(w.iter().sum(), 1.);
        assert!(w[6] < 1e-30);
        assert!(w[7] > w[0]);
        fit.observe(25110, &[-10.; 8]);
        let frozen = fit.clone();
        fit.observe(25111, &[1000.; 8]);
        fit.observe(26125, &[-1000.; 8]);
        assert_eq!(fit, frozen);
    }
    #[test]
    fn complete_mixture_tickets_match_exhaustive_mass_ranking_with_ties() {
        for uniform in [true, false] {
            let mut experts = [[1.; 33]; 7];
            if !uniform {
                for (e, p) in experts.iter_mut().enumerate() {
                    for (i, v) in p.iter_mut().enumerate() {
                        *v = (((i + 1) * (e + 3)) % 37 + 1) as f64;
                    }
                }
            }
            let ms = models(&experts).unwrap();
            let weights = [0.125; 8];
            let order = mixture_rank(&ms, &weights);
            let pool = &order[..16];
            let sets = best_sets(&ms, &weights, pool);
            let mut oracle: Vec<_> = (0..1u64 << 16)
                .filter(|m| m.count_ones() == 6)
                .map(|m| {
                    let indices: Vec<_> = (0..16).filter(|i| m & (1 << i) != 0).collect();
                    let actual = indices.iter().fold(0, |mask, i| mask | (1 << pool[*i]));
                    (
                        (mixture_logp(&ms, &weights, actual) * 1e12).round() as i64,
                        indices,
                        actual,
                    )
                })
                .collect();
            oracle.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            assert_eq!(sets, oracle[..4].iter().map(|v| v.2).collect::<Vec<_>>());
            let tickets = pairs(&sets, &[2, 9]);
            assert_eq!(tickets.len(), 8);
            assert_eq!(
                tickets
                    .iter()
                    .map(|t| (t.red, t.blue))
                    .collect::<BTreeSet<_>>()
                    .len(),
                8
            );
            assert!(tickets.iter().all(
                |t| t.red.len() == 6 && t.red.iter().all(|n| pool.contains(&(*n as usize - 1)))
            ));
        }
    }
    #[test]
    fn scores_and_tickets_are_causal_when_future_draws_are_appended() {
        let draws = crate::draw::synth_draws(240, 91);
        let a = rows(&draws[..220]).unwrap();
        let b = rows(&draws).unwrap();
        assert!(!a.0.is_empty());
        assert_eq!(a.0, b.0[..a.0.len()]);
        let report = analyze(&draws).unwrap();
        assert_eq!(report["adoption"]["new_prospective_n"], 0);
        assert_eq!(report["adoption"]["replace_current_recommendation"], false);
        assert_eq!(report["candidate"]["publication_allowed"], false);
        assert_eq!(report["candidate"]["plan"]["summary"]["cost"], 16);
        assert!(report["candidate"]["plan"]["summary"]["prize_tickets"].is_null());
        assert!(report["candidate"]["plan"]["tickets"]
            .as_array()
            .unwrap()
            .iter()
            .all(|t| t["red_hits"].is_null() && t["blue_hit"].is_null()));
    }
    #[test]
    fn current_inspection_is_pending_and_does_not_create_or_replace_files() {
        let root = std::env::temp_dir().join(format!("ssq-joint-no-write-{}", std::process::id()));
        assert!(!root.exists());
        let draws = crate::draw::synth_draws(200, 17);
        let v = report(&root, &draws).unwrap();
        assert!(!root.exists());
        assert_eq!(v["current_health"]["status"], "pending");
        assert_eq!(v["current_health"]["outcome_known"], false);
        assert_eq!(v["current_health"]["source"]["status"], "preview");
        let cached = report(&root, &draws).unwrap();
        assert_eq!(v, cached);
        assert!(!root.exists());
    }
}
