//! Blue-ball hit objective, isolated from the original frozen protocol.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::Prepared,
    model::roll_all,
    protocol::Protocol,
    select::{fuse6, quantize_simplex, rank_desc, softmax_shrink},
};
use serde_json::{json, Value};

pub const MODEL: &str = "blue_hit_frozen";
pub const VERSION: u32 = 1;
pub const PRESETS: [&str; 4] = ["bluefocus2", "bluehit2", "bluefocus4", "bluehit4"];
pub const MODELS: [&str; 3] = ["weighted", "equal", MODEL];
pub const LABELS: [&str; 3] = ["原加权蓝榜", "等权蓝榜 · 研究", "命中目标融合 · 试验"];

pub fn policy() -> Value {
    json!({"version":VERSION,"model":MODEL,"objective":"top_2_blue_hit_rate",
        "fit_before_issue":Protocol::CONFIRM_START_ISSUE,"minimum_fit_periods":30,
        "method":"six_expert_top2_hit_softmax","temperature":0.0625,
        "ties":"original_score_key_then_number","automatic_promotion":false,
        "note":"用25111之前的更早样本外记录累计各专家前2蓝命中，按固定温度融合其分布；边界后权重冻结，专家状态继续只用此前开奖。评分不是校准概率，尚未证明稳定优于原榜。"})
}

#[derive(Default)]
struct Fit {
    hits: [usize; 6],
    n: usize,
    last_issue: Option<u32>,
}
impl Fit {
    fn observe(&mut self, issue: u32, experts: &[[f64; 16]; 6], winner: usize) {
        if issue >= Protocol::CONFIRM_START_ISSUE {
            return;
        }
        for (e, p) in experts.iter().enumerate() {
            self.hits[e] += usize::from(rank_desc(p)[..2].contains(&winner));
        }
        self.n += 1;
        self.last_issue = Some(issue);
    }
    fn weights(&self) -> [f64; 6] {
        let peak = *self.hits.iter().max().unwrap() as f64;
        let exp = self
            .hits
            .map(|h| ((h as f64 - peak) * 16.0 / self.n.max(1) as f64).exp());
        let total: f64 = exp.iter().sum();
        quantize_simplex(exp.map(|v| v / total))
    }
}

pub fn apply(mut prepared: Prepared, draws: &[Draw]) -> Result<Prepared, String> {
    let snapshot = roll_all(draws)?;
    let mut fit = Fit::default();
    for row in &snapshot.rows {
        // Train on the same pre-boundary OOS domain as the original blend.
        fit.observe(row.issue, &row.blue_p, row.blue as usize);
    }
    if fit.n < 30 {
        return Err("蓝球命中目标试验至少需要30条边界前样本外记录".into());
    }
    prepared.blue_weights = fit.weights();
    prepared.forecast.blue = fuse6(&prepared.expert_blue, &prepared.blue_weights);
    prepared.forecast.blue_order = rank_desc(&prepared.forecast.blue);
    prepared.forecast.ticket_blues = (0..prepared.forecast.tickets.len())
        .map(|i| prepared.forecast.blue_order[i % Protocol::BLUE_SLOTS])
        .collect();
    prepared.blue_ranking = Some(
        json!({"model":MODEL,"policy":policy(),"input_history_sha256":prepared.data_sha,
        "target_issue":fmt_issue(prepared.cutoff_issue+1),"training_periods":fit.n,"training_last_issue":fit.last_issue,
        "expert_top2_hits":fit.hits,"weights":prepared.blue_weights,
        "blue_ranking":prepared.forecast.blue_order.map(|n|n+1),"scores":prepared.forecast.blue,
        "score_kind":"ranking_evidence_not_calibrated_probability"}),
    );
    Ok(prepared)
}

#[derive(Clone, Debug, PartialEq)]
struct Period {
    issue: u32,
    ranks: [usize; 3],
    blue_hit_bets: [[usize; 3]; 3],
    blue_red4_bets: [[usize; 3]; 3],
}
fn summarize(rows: &[Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells:Vec<_>=MODELS.iter().enumerate().flat_map(|(m,name)|(1..=8).map(move |k|{
        let hits=rows.iter().filter(|r|r.ranks[m]<=k).count();
        let gains=rows.iter().filter(|r|r.ranks[m]<=k&&r.ranks[0]>k).count();
        let losses=rows.iter().filter(|r|r.ranks[m]>k&&r.ranks[0]<=k).count();
        json!({"model":name,"blue_count":k,"n":rows.len(),"hit_periods":hits,"hit_rate":hits as f64/rows.len()as f64,
            "uniform_hit_rate":k as f64/16.0,"paired_gains":gains,"paired_losses":losses})
    })).collect();
    let tickets:Vec<_>=MODELS.iter().enumerate().flat_map(|(m,name)|[2,3,4].into_iter().enumerate().map(move |(j,k)|json!({
        "model":name,"blue_count":k,"bets":8,"cost":16,"red_count":16,"n":rows.len(),
        "blue_covered_periods":rows.iter().filter(|r|r.ranks[m]<=k).count(),
        "blue_hit_bets":rows.iter().map(|r|r.blue_hit_bets[m][j]).sum::<usize>(),
        "blue_and_at_least_four_red_bets":rows.iter().map(|r|r.blue_red4_bets[m][j]).sum::<usize>()}))).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"cells":cells,"same_budget_tickets":tickets})
}

fn replay(draws: &[Draw]) -> Result<(Vec<Period>, Fit), String> {
    let snap = roll_all(draws)?;
    let (mut fit, mut sums, mut red_sums, mut n) = (Fit::default(), [0.; 6], [0.; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        if n >= 30 {
            let scores = [
                fuse6(&row.blue_p, &softmax_shrink(&sums, n as f64)),
                fuse6(&row.blue_p, &[1. / 6.; 6]),
                fuse6(&row.blue_p, &fit.weights()),
            ];
            let orders = scores.map(|p| rank_desc(&p));
            let ranks = orders.map(|o| o.iter().position(|&i| i == row.blue as usize).unwrap() + 1);
            // Same original weighted red tickets for every blue model and size.
            let red_weights = crate::select::softmax_shrink(&red_sums, n as f64);
            let red_order = rank_desc(&crate::select::fuse7(&row.expert_p, &red_weights));
            let masks = crate::coverage::balanced_tickets(&red_order[..16], 8)?;
            let bets = orders.map(|o| {
                [2, 3, 4].map(|k| (0..8).filter(|i| o[i % k] == row.blue as usize).count())
            });
            let red4 = orders.map(|o| {
                [2, 3, 4].map(|k| {
                    (0..8)
                        .filter(|&i| {
                            o[i % k] == row.blue as usize
                                && (masks[i] & row.red_mask).count_ones() >= 4
                        })
                        .count()
                })
            });
            rows.push(Period {
                issue: row.issue,
                ranks,
                blue_hit_bets: bets,
                blue_red4_bets: red4,
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..6 {
                sums[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.).ln();
            }
            for e in 0..7 {
                red_sums[e] += crate::select::rank_log_score(&row.expert_p[e], row.red_mask);
            }
            n += 1;
        }
        fit.observe(row.issue, &row.blue_p, row.blue as usize);
    }
    Ok((rows, fit))
}
pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let (rows, fit) = replay(draws)?;
    let start = rows
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let prepared = crate::eval::prepare_forecast(draws)?;
    let score = [
        prepared.forecast.blue,
        fuse6(&prepared.expert_blue, &[1. / 6.; 6]),
        fuse6(&prepared.expert_blue, &fit.weights()),
    ];
    let next:Vec<_>=MODELS.iter().enumerate().map(|(m,name)|json!({"model":name,"label":LABELS[m],"ranking":rank_desc(&score[m]).map(|i|i+1),"score_kind":"ranking_evidence_not_probability"})).collect();
    Ok(
        json!({"kind":"exploratory_blue_hit_review","actual_recommendation":false,"automatic_promotion":false,"data_sha256":data_sha256(draws),
        "cutoff_issue":draws.last().map(|d|fmt_issue(d.issue)),"target_issue":draws.last().map(|d|fmt_issue(d.issue+1)),"policy":policy(),
        "source_sha256":crate::hash::sha256_hex(concat!(include_str!("blue.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("protocol.rs"),include_str!("coverage.rs")).as_bytes()),
        "next":next,"windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),"tracking":summarize(&rows[end..]),
        "recent30":summarize(&rows[rows.len().saturating_sub(30)..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},
        "note":"同数量比较排序，增加蓝球数量的覆盖收益另列；每期先预测再计分，回放不算实际发布。确认窗已被查看，新增分析属于探索，不加入原预注册检验。更多不同蓝球在相同8注预算下会减少每个蓝球重复注数，不能推断奖金或收益提高。"}),
    )
}
pub fn validate_history(saved: &Value, draws: &[Draw]) -> Result<(), String> {
    let expected = apply(crate::eval::prepare_forecast(draws)?, draws)?
        .blue_ranking
        .unwrap();
    if *saved != expected {
        return Err("蓝球试验排名或训练证据与原输入不一致".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn append_outcomes_never_changes_earlier_predictions_or_frozen_weights() {
        let d = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let before = apply(
            crate::eval::prepare_forecast(&d[..d.len() - 1]).unwrap(),
            &d[..d.len() - 1],
        )
        .unwrap();
        let after = apply(crate::eval::prepare_forecast(&d).unwrap(), &d).unwrap();
        assert_eq!(before.blue_weights, after.blue_weights);
        let (a, _) = replay(&d[..240]).unwrap();
        let (b, _) = replay(&d[..250]).unwrap();
        assert_eq!(a.len(), 60);
        assert_eq!(a, b[..a.len()]);
        let mut fit = Fit::default();
        fit.observe(25111, &[[1. / 16.; 16]; 6], 0);
        assert_eq!(fit.n, 0);
    }
    #[test]
    fn hit_weights_and_scores_roundtrip_and_red_tickets_are_preserved() {
        let d = crate::draw::synth_draws(220, 9);
        let p = crate::eval::prepare_forecast(&d).unwrap();
        let reds = p.forecast.red_order;
        let masks = p.forecast.tickets.clone();
        let p = apply(p, &d).unwrap();
        assert_eq!(p.forecast.red_order, reds);
        assert_eq!(p.forecast.tickets, masks);
        let v = p.blue_ranking.unwrap();
        assert_eq!(serde_json::from_str::<Value>(&v.to_string()).unwrap(), v);
        assert_eq!(v["blue_ranking"].as_array().unwrap().len(), 16);
        assert!((p.blue_weights.iter().sum::<f64>() - 1.).abs() < 1e-10);
    }
}
