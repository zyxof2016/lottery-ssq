//! Empirical gap buckets, learned from earlier draws only; shared original tie order.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::Prepared,
    pool,
    select::{rank_desc, score_key},
};
use serde_json::{json, Value};
use std::collections::VecDeque;

pub const MODEL: &str = "conditional_gap";
pub const LABEL: &str = "条件频率 · 24红试验";
pub const VERSION: u32 = 1;
const WINDOW: usize = 312;
const PRIOR: f64 = 156.0;

pub fn policy() -> Value {
    json!({"model":MODEL,"label":LABEL,"version":VERSION,"training_draws_max":WINDOW,
        "gap_buckets":["0","1","2","3","4","5–7","8–11","12–17","18+"],
        "prior_opportunities_per_bucket":PRIOR,"prior_red_inclusion":6.0/33.0,
        "score_formula":"(bucket_hits + 156 * 6 / 33) / (bucket_opportunities + 156)",
        "ties":"original_weighted_ranking","blue_scope":"shared_weighted_blue_ranking",
        "primary_metric":"full_six_red_rate_at_24","automatic_promotion":false,
        "note":"用此前最多312期开奖统计各遗漏区间的实际命中频率，并向均匀基准收缩；同分沿原排名。重点观察24红全包表现，14–22红一并报告，历史结果不代表未来覆盖概率。"})
}

fn bucket(gap: usize) -> usize {
    match gap {
        0..=4 => gap,
        5..=7 => 5,
        8..=11 => 6,
        12..=17 => 7,
        _ => 8,
    }
}

struct State {
    t: usize,
    last: [Option<usize>; 33],
    opportunities: [usize; 9],
    wins: [usize; 9],
    history: VecDeque<(u32, [usize; 33], u64)>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            t: 0,
            last: [None; 33],
            opportunities: [0; 9],
            wins: [0; 9],
            history: VecDeque::new(),
        }
    }
}
impl State {
    fn buckets(&self) -> [usize; 33] {
        std::array::from_fn(|i| bucket(self.last[i].map(|j| self.t - j - 1).unwrap_or(self.t)))
    }
    fn scores(&self) -> [f64; 33] {
        let bins = self.buckets();
        std::array::from_fn(|i| {
            (self.wins[bins[i]] as f64 + PRIOR * 6.0 / 33.0)
                / (self.opportunities[bins[i]] as f64 + PRIOR)
        })
    }
    fn update(&mut self, draw: &Draw) {
        let bins = self.buckets();
        for i in 0..33 {
            self.opportunities[bins[i]] += 1;
            if draw.red_mask & (1 << i) != 0 {
                self.wins[bins[i]] += 1;
                self.last[i] = Some(self.t);
            }
        }
        self.history.push_back((draw.issue, bins, draw.red_mask));
        self.t += 1;
        if self.history.len() > WINDOW {
            let (_, bins, mask) = self.history.pop_front().unwrap();
            for i in 0..33 {
                self.opportunities[bins[i]] -= 1;
                if mask & (1 << i) != 0 {
                    self.wins[bins[i]] -= 1;
                }
            }
        }
    }
    fn evidence(&self) -> Value {
        json!({"n":self.history.len(),"from":self.history.front().map(|r|fmt_issue(r.0)),"to":self.history.back().map(|r|fmt_issue(r.0)),
            "bucket_opportunities":self.opportunities,"bucket_hits":self.wins,"next_gap_buckets":self.buckets().to_vec()})
    }
}

pub fn order(raw: &[f64; 33], base: &[usize; 33]) -> [usize; 33] {
    let mut order = *base;
    // Stable sort retains the frozen ranking within exactly tied buckets.
    order.sort_by(|&a, &b| score_key(raw[b]).cmp(&score_key(raw[a])));
    order
}
fn ordinal(order: &[usize; 33]) -> [f64; 33] {
    let mut s = [0.0; 33];
    for (r, &n) in order.iter().enumerate() {
        s[n] = (33 - r) as f64 / 561.0;
    }
    s
}

pub fn saved(prepared: &Prepared, draws: &[Draw]) -> Value {
    let mut state = State::default();
    for draw in draws {
        state.update(draw);
    }
    let raw = state.scores();
    let order = order(&raw, &prepared.forecast.red_order);
    let pools:Vec<_>=pool::TIER_RED_SIZES.into_iter().enumerate().flat_map(|(i,r)|{
        let order=&order;pool::BLUE_SIZES.into_iter().map(move |b|json!({"id":format!("{MODEL}_red{r}_blue{b}"),
            "model":MODEL,"group":"observation","tier_id":((b'A'+i as u8)as char).to_string(),"red_count":r,"blue_count":b,
            "reds":order[..r].iter().map(|n|n+1).collect::<Vec<_>>(),"blues":prepared.forecast.blue_order[..b].iter().map(|n|n+1).collect::<Vec<_>>()}))
    }).collect();
    json!({"model":MODEL,"version":VERSION,"policy":policy(),"target_issue":fmt_issue(prepared.cutoff_issue+1),
        "input_history_sha256":prepared.data_sha,"training":state.evidence(),"raw_conditional_scores":raw.to_vec(),
        "base_red_weights":Value::Null,"base_red_ranking":prepared.forecast.red_order.iter().map(|n|n+1).collect::<Vec<_>>(),
        "red_ranking":order.iter().map(|n|n+1).collect::<Vec<_>>(),"red_scores":ordinal(&order).to_vec(),
        "score_kind":"ordinal_rank_evidence_not_probability","pools":pools})
}

pub fn prepare(draws: &[Draw]) -> Result<Prepared, String> {
    let prepared = crate::eval::prepare_forecast(draws)?;
    let metadata = saved(&prepared, draws);
    let score: Vec<f64> =
        serde_json::from_value(metadata["red_scores"].clone()).map_err(|e| e.to_string())?;
    Ok(crate::ranking::apply(
        prepared,
        score.try_into().map_err(|_| "条件频率排名缺少分数")?,
        metadata,
    ))
}

pub fn validate(saved: &Value, base: &[u8], target: &Value) -> Result<(), String> {
    if saved["model"] != MODEL
        || saved["version"] != VERSION
        || saved["policy"] != policy()
        || saved["target_issue"] != *target
        || saved["base_red_ranking"] != json!(base)
    {
        return Err("条件频率模型规则或基础排名不一致".into());
    }
    let opportunities: Vec<usize> =
        serde_json::from_value(saved["training"]["bucket_opportunities"].clone())
            .map_err(|_| "条件频率缺少样本数")?;
    let hits: Vec<usize> = serde_json::from_value(saved["training"]["bucket_hits"].clone())
        .map_err(|_| "条件频率缺少命中数")?;
    let bins: Vec<usize> = serde_json::from_value(saved["training"]["next_gap_buckets"].clone())
        .map_err(|_| "条件频率缺少遗漏区间")?;
    let raw: Vec<f64> = serde_json::from_value(saved["raw_conditional_scores"].clone())
        .map_err(|_| "条件频率缺少分数")?;
    let n = saved["training"]["n"]
        .as_u64()
        .ok_or("条件频率训练期数无效")? as usize;
    if n == 0
        || n > WINDOW
        || opportunities.len() != 9
        || hits.len() != 9
        || bins.len() != 33
        || raw.len() != 33
        || opportunities.iter().sum::<usize>() != n * 33
        || hits.iter().sum::<usize>() != n * 6
        || (0..9).any(|i| hits[i] > opportunities[i])
        || bins.iter().any(|&b| b >= 9)
    {
        return Err("条件频率训练证据无效".into());
    }
    let expected = std::array::from_fn(|i| {
        (hits[bins[i]] as f64 + PRIOR * 6.0 / 33.0) / (opportunities[bins[i]] as f64 + PRIOR)
    });
    let base = std::array::from_fn(|i| base[i] as usize - 1);
    let order = order(&expected, &base);
    if raw
        .iter()
        .zip(expected)
        .any(|(a, b)| !a.is_finite() || (*a - b).abs() > 1e-12)
        || saved["red_ranking"] != json!(order.map(|n| n + 1).to_vec())
        || saved["red_scores"] != json!(ordinal(&order).to_vec())
    {
        return Err("条件频率排名与保存的训练证据不一致".into());
    }
    Ok(())
}

pub fn validate_history(saved: &Value, draws: &[Draw]) -> Result<(), String> {
    let mut state = State::default();
    for draw in draws {
        state.update(draw);
    }
    if saved["input_history_sha256"] != data_sha256(draws) || saved["training"] != state.evidence()
    {
        return Err("条件频率训练证据与开奖输入历史不一致".into());
    }
    Ok(())
}

pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    use crate::{
        model::roll_all,
        protocol::Protocol,
        select::{fuse7, rank_log_score, softmax_shrink},
    };
    let snap = roll_all(draws)?;
    let rowmap: std::collections::BTreeMap<_, _> = snap.rows.iter().map(|r| (r.issue, r)).collect();
    let (mut state, mut sums, mut n) = (State::default(), [0.0; 7], 0usize);
    let mut rows = Vec::new();
    for draw in draws {
        if let Some(row) = rowmap.get(&draw.issue) {
            if n >= 30 {
                let base = rank_desc(&fuse7(&row.expert_p, &softmax_shrink(&sums, n as f64)));
                let candidate = order(&state.scores(), &base);
                let models:Vec<_>=[base,candidate].iter().map(|o|{
                    let ranks:Vec<_>=o.iter().enumerate().filter(|(_,n)|draw.red_mask&(1<<**n)!=0).map(|(i,_)|i+1).collect();
                    json!({"hits":pool::TIER_RED_SIZES.map(|k|o[..k].iter().filter(|&&n|draw.red_mask&(1<<n)!=0).count()),"winning_red_ranks":ranks})
                }).collect();
                rows.push(json!({"issue":draw.issue,"models":models}));
            }
            if row.issue < Protocol::CONFIRM_START_ISSUE {
                for e in 0..7 {
                    sums[e] += rank_log_score(&row.expert_p[e], row.red_mask);
                }
                n += 1;
            }
        }
        state.update(draw);
    }
    fn summary(rows: &[Value]) -> Value {
        if rows.is_empty() {
            return Value::Null;
        }
        let cells:Vec<_>=["weighted",MODEL].into_iter().enumerate().flat_map(|(m,model)|{
            pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,k)|{
                let total:u64=rows.iter().map(|r|r["models"][m]["hits"][j].as_u64().unwrap()).sum();
                let full=rows.iter().filter(|r|r["models"][m]["hits"][j]==6).count();
                json!({"model":model,"red_count":k,"n":rows.len(),"full_red_periods":full,"full_red_rate":full as f64/rows.len()as f64,
                    "mean_red_hits":total as f64/rows.len()as f64,"mean_missed_reds":6.0-total as f64/rows.len()as f64,
                    "uniform_full_red_rate":pool::uniform_full_red(k),
                    "paired_full_gains":rows.iter().filter(|r|r["models"][m]["hits"][j]==6 && r["models"][1-m]["hits"][j]!=6).count()})
            })
        }).collect();
        json!({"n":rows.len(),"from":rows.first().unwrap()["issue"],"to":rows.last().unwrap()["issue"],"cells":cells})
    }
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let prepared = crate::eval::prepare_forecast(draws)?;
    Ok(
        json!({"kind":"exploratory_full_six_coverage_review","actual_recommendation":false,"policy":policy(),
        "data_sha256":data_sha256(draws),"cutoff_issue":draws.last().unwrap().issue,
        "source_sha256":crate::hash::sha256_hex(concat!(include_str!("conditional.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("eval.rs"),include_str!("protocol.rs"),include_str!("pool.rs"),include_str!("ranking.rs")).as_bytes()),
        "windows":{"development":summary(&rows[..start]),"confirmation_exploratory":summary(&rows[start..end]),"tracking":summary(&rows[end..]),
            "recent30":summary(&rows[rows.len().saturating_sub(30)..]),"recent60":summary(&rows[rows.len().saturating_sub(60)..]),
            "recent156":summary(&rows[rows.len().saturating_sub(156)..]),"all_oos":summary(&rows)},"latest_replay":rows.last(),
        "next_preview":saved(&prepared,draws),"adoption":{"primary_protocol_replaced":false,"scope":"additional_prospective_24_red_observation"},
        "notes":["原权重在25111前冻结；条件统计逐期先预测再纳入该期开奖。所有同档比较使用相同期次。",
            "本轮研究查看了多种候选，属于事后探索，不能把近期改善当作独立确认或未来覆盖承诺。",
            "条件频率重点观察24红；14–22红结果完整报告。全6红入池不等于某一注已经含全6红。",
            "已有每日档案不追加新模型。只有未来新期首次冻结后，才按原号码累计实际覆盖。"]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_binds_history_and_rejects_changed_training_or_rank() {
        let draws = crate::draw::synth_draws(600, 9);
        let p = crate::eval::prepare_forecast(&draws).unwrap();
        let saved = saved(&p, &draws);
        let base = p.forecast.red_order.map(|n| n as u8 + 1);
        validate(&saved, &base, &json!(fmt_issue(p.cutoff_issue + 1))).unwrap();
        validate_history(&saved, &draws).unwrap();
        assert_eq!(saved["training"]["n"], 312);
        let mut bad = saved.clone();
        bad["training"]["bucket_hits"][0] = json!(999999);
        assert!(validate(&bad, &base, &saved["target_issue"]).is_err());
        let mut bad = saved.clone();
        bad["red_ranking"].as_array_mut().unwrap().swap(0, 32);
        assert!(validate(&bad, &base, &saved["target_issue"]).is_err());
        assert!(validate_history(&saved, &draws[..599]).is_err());
    }
    #[test]
    fn later_outcomes_cannot_change_past_replay() {
        let draws = crate::draw::synth_draws(600, 9);
        let a = analyze(&draws[..599]).unwrap();
        let b = analyze(&draws).unwrap();
        let ranking = a["next_preview"]["red_ranking"].as_array().unwrap();
        let mut expected: Vec<_> = draws
            .last()
            .unwrap()
            .red
            .iter()
            .map(|n| ranking.iter().position(|v| v == n).unwrap() + 1)
            .collect();
        expected.sort_unstable();
        assert_eq!(
            b["latest_replay"]["models"][1]["winning_red_ranks"],
            json!(expected)
        );
        let flat = [1.0; 33];
        let base = std::array::from_fn(|i| 32 - i);
        assert_eq!(order(&flat, &base), base);
    }
}
