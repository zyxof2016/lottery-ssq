//! Small red pools: a fixed shrinkage experiment, isolated from existing pools.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    eval::{prepare_forecast, Prepared},
    model::roll_all,
    protocol::Protocol,
    select::{fuse7, rank_desc, rank_log_score, softmax_shrink},
};
use serde_json::{json, Value};
pub const MODEL: &str = "small_recent";
pub const MODELS: [&str; 2] = ["weighted", MODEL];
pub const SIZES: [usize; 4] = [12, 14, 15, 16];
pub fn policy() -> Value {
    json!({"version":1,"model":MODEL,"label":"小池近况融合 · 试验",
    "method":"half_original_quarter_ewma_quarter_repeat","original_weight":0.5,"ewma_weight":0.25,"repeat_weight":0.25,
    "primary_metric":"all_six_red_covered_at_same_size_at_most_16","red_sizes":SIZES,"automatic_promotion":false,
    "blue_scope":"unchanged_original_blue_ranking","score_kind":"ranking_evidence_not_probability",
    "note":"固定使用原融合50%、近期指数25%、上期重号25%；原权重仍仅拟合25111前数据。长期小池回放有增量，但部分近窗退步，尚未证明稳定提高全6红率，不保证未来全覆盖。"})
}
pub fn scores(original: &[f64; 33], experts: &[[f64; 33]; 7]) -> [f64; 33] {
    std::array::from_fn(|n| 0.5 * original[n] + 0.25 * experts[1][n] + 0.25 * experts[6][n])
}
pub fn saved(p: &Prepared) -> Value {
    let raw = scores(&p.forecast.fusion, &p.expert_red);
    let orders = [p.forecast.red_order, rank_desc(&raw)];
    let models:Vec<_>=MODELS.iter().enumerate().map(|(m,name)|{
        let pools:Vec<_>=SIZES.iter().map(|&k|json!({"id":format!("{name}_red{k}"),"model":name,"red_count":k,"reds":orders[m][..k].iter().map(|n|n+1).collect::<Vec<_>>()})).collect();
        json!({"model":name,"label":if m==0{"原加权融合 · 小池对照"}else{"小池近况融合 · 试验"},"red_ranking":orders[m].map(|n|n+1).to_vec(),"pools":pools})
    }).collect();
    json!({"version":1,"kind":"small_red_pool_observation","policy":policy(),"target_issue":fmt_issue(p.cutoff_issue+1),
        "data_cutoff":fmt_issue(p.cutoff_issue),"input_history_sha256":p.data_sha,"base_weights":p.weights,
        "base_scores":p.forecast.fusion.to_vec(),"ewma_scores":p.expert_red[1].to_vec(),"repeat_scores":p.expert_red[6].to_vec(),
        "trial_scores":raw.to_vec(),"models":models})
}
pub fn validate_snapshot(v: &Value) -> Result<(), String> {
    if v["version"] != 1 || v["policy"] != policy() {
        return Err("小红球池版本或规则无效".into());
    }
    let mut arrays = Vec::new();
    for key in [
        "base_scores",
        "ewma_scores",
        "repeat_scores",
        "trial_scores",
    ] {
        let a: Vec<f64> = serde_json::from_value(v[key].clone()).map_err(|_| "小红球池缺少评分")?;
        if a.len() != 33
            || a.iter().any(|x| !x.is_finite() || *x < 0.)
            || (a.iter().sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err("小红球池评分无效".into());
        }
        arrays.push(a.try_into().unwrap());
    }
    let a: &Vec<[f64; 33]> = &arrays;
    let expected: [f64; 33] =
        std::array::from_fn(|n| a[0][n] * 0.5 + a[1][n] * 0.25 + a[2][n] * 0.25);
    if a[3] != expected {
        return Err("小红球池试验分数与保存证据不一致".into());
    }
    let models = v["models"].as_array().ok_or("小红球池缺少模型")?;
    if models.len() != 2 {
        return Err("小红球池缺少成对模型".into());
    }
    for (m, name) in MODELS.iter().enumerate() {
        let order = rank_desc(&a[if m == 0 { 0 } else { 3 }]);
        let pools = models[m]["pools"].as_array().ok_or("小红球池缺少号码")?;
        if models[m]["model"] != *name
            || models[m]["red_ranking"] != json!(order.map(|n| n + 1).to_vec())
            || pools.len() != SIZES.len()
        {
            return Err("小红球池排名无效".into());
        }
        for (i, &k) in SIZES.iter().enumerate() {
            if pools[i]["model"] != *name
                || pools[i]["red_count"] != k
                || pools[i]["reds"] != json!(order[..k].iter().map(|n| n + 1).collect::<Vec<_>>())
            {
                return Err("小红球池与保存排名不一致".into());
            }
        }
    }
    Ok(())
}
pub fn validate_history(v: &Value, draws: &[Draw]) -> Result<(), String> {
    validate_snapshot(v)?;
    if *v != saved(&prepare_forecast(draws)?) {
        return Err("小红球池训练证据与原输入历史不一致".into());
    }
    Ok(())
}
pub fn review(v: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    validate_snapshot(v)?;
    let mut v = v.clone();
    for m in v["models"].as_array_mut().unwrap() {
        for p in m["pools"].as_array_mut().unwrap() {
            let reds: Vec<u8> =
                serde_json::from_value(p["reds"].clone()).map_err(|e| e.to_string())?;
            let hits = draw.map(|d| d.red.iter().filter(|n| reds.contains(n)).count());
            p["red_hits"] = json!(hits);
            p["full_red"] = json!(hits.map(|h| h == 6));
            p["missed_reds"] = json!(draw.map(|d| d
                .red
                .iter()
                .filter(|n| !reds.contains(n))
                .copied()
                .collect::<Vec<_>>()));
        }
    }
    Ok(v)
}
pub fn prepare(draws: &[Draw]) -> Result<Prepared, String> {
    let p = prepare_forecast(draws)?;
    let raw = scores(&p.forecast.fusion, &p.expert_red);
    let mut metadata = saved(&p);
    metadata["model"] = json!(MODEL);
    metadata["red_weights"] = json!(std::array::from_fn::<_, 7, _>(
        |e| p.weights[e] * 0.5 + if e == 1 || e == 6 { 0.25 } else { 0.0 }
    ));
    metadata["base_red_weights"] = json!(p.weights);
    metadata["red_ranking"] = metadata["models"][1]["red_ranking"].clone();
    metadata["red_scores"] = json!(raw.to_vec());
    Ok(crate::ranking::apply(p, raw, metadata))
}
#[derive(Clone, Debug, PartialEq)]
struct Period {
    issue: u32,
    hits: [[usize; 4]; 2],
}
fn periods(draws: &[Draw]) -> Result<Vec<Period>, String> {
    let snap = roll_all(draws)?;
    let (mut sums, mut n) = ([0.; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        if n >= 30 {
            let raw = fuse7(&row.expert_p, &softmax_shrink(&sums, n as f64));
            let orders = [rank_desc(&raw), rank_desc(&scores(&raw, &row.expert_p))];
            rows.push(Period {
                issue: row.issue,
                hits: orders.map(|o| {
                    SIZES.map(|k| {
                        o[..k]
                            .iter()
                            .filter(|&&i| row.red_mask & (1 << i) != 0)
                            .count()
                    })
                }),
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                sums[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            n += 1;
        }
    }
    Ok(rows)
}
fn summarize(rows: &[Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let cells:Vec<_>=MODELS.iter().enumerate().flat_map(|(m,name)|SIZES.iter().enumerate().map(move |(j,&k)|{
    let full=rows.iter().filter(|r|r.hits[m][j]==6).count();let total:usize=rows.iter().map(|r|r.hits[m][j]).sum();
    json!({"model":name,"red_count":k,"n":rows.len(),"full_red_periods":full,"full_red_rate":full as f64/rows.len()as f64,
        "uniform_full_red_rate":crate::pool::uniform_full_red(k),"at_least_5_red_periods":rows.iter().filter(|r|r.hits[m][j]>=5).count(),"mean_missed_reds":6.-total as f64/rows.len()as f64,
        "paired_gains":rows.iter().filter(|r|r.hits[m][j]==6&&r.hits[0][j]!=6).count(),"paired_losses":rows.iter().filter(|r|r.hits[m][j]!=6&&r.hits[0][j]==6).count()})
})).collect();
    json!({"n":rows.len(),"from":fmt_issue(rows[0].issue),"to":fmt_issue(rows.last().unwrap().issue),"cells":cells})
}
pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let rows = periods(draws)?;
    let start = rows
        .iter()
        .position(|r| r.issue >= Protocol::CONFIRM_START_ISSUE)
        .unwrap_or(rows.len());
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let p = prepare_forecast(draws)?;
    Ok(
        json!({"kind":"exploratory_small_red_pool_review","actual_recommendation":false,"automatic_promotion":false,"data_sha256":data_sha256(draws),"cutoff_issue":fmt_issue(p.cutoff_issue),"target_issue":fmt_issue(p.cutoff_issue+1),"red_sizes":SIZES,
        "source_sha256":crate::hash::sha256_hex(concat!(include_str!("small_red.rs"),include_str!("model.rs"),include_str!("select.rs"),include_str!("eval.rs"),include_str!("protocol.rs")).as_bytes()),"next":saved(&p),
        "windows":{"development":summarize(&rows[..start]),"confirmation_exploratory":summarize(&rows[start..end]),"tracking":summarize(&rows[end..]),"recent30":summarize(&rows[rows.len().saturating_sub(30)..]),"recent60":summarize(&rows[rows.len().saturating_sub(60)..]),"recent156":summarize(&rows[rows.len().saturating_sub(156)..]),"all_oos":summarize(&rows)},
        "note":"主指标为相同数量下全部6红入池，平均中红只作辅助。历史回放不是实际发布表现；已查看多个候选和重叠窗口，不能视为独立确认。16红均匀全包基准约0.723%，缩小号码池不能保证全覆盖。"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pools_are_nested_at_most16_and_history_bound() {
        let d = crate::draw::synth_draws(220, 9);
        let p = prepare_forecast(&d).unwrap();
        let v = saved(&p);
        validate_history(&v, &d).unwrap();
        for m in v["models"].as_array().unwrap() {
            let mut previous = Vec::<Value>::new();
            for p in m["pools"].as_array().unwrap() {
                let reds = p["reds"].as_array().unwrap();
                assert!(reds.len() <= 16 && previous.iter().all(|n| reds.contains(n)));
                previous = reds.clone();
            }
        }
        let pending = review(&v, None).unwrap();
        assert!(pending["models"][1]["pools"][0]["full_red"].is_null());
        let reds: Vec<u8> =
            serde_json::from_value(v["models"][1]["pools"][0]["reds"].clone()).unwrap();
        let draw = Draw::from_parts(3221, 20260101, reds[..6].try_into().unwrap(), 1).unwrap();
        assert_eq!(
            review(&v, Some(&draw)).unwrap()["models"][1]["pools"][0]["full_red"],
            true
        );
        let mut bad = v.clone();
        bad["trial_scores"][0] = json!(0.1);
        assert!(validate_snapshot(&bad).is_err());
        let mut bad = v;
        bad["models"][1]["pools"][0]["reds"][0] = json!(99);
        assert!(validate_snapshot(&bad).is_err());
    }
    #[test]
    fn later_outcomes_never_change_earlier_small_pool_predictions() {
        let d = crate::draw::synth_draws(250, 9);
        let a = periods(&d[..230]).unwrap();
        let b = periods(&d).unwrap();
        assert_eq!(a, b[..a.len()]);
        let old = analyze(&d[..249]).unwrap();
        let last = periods(&d).unwrap().pop().unwrap();
        for (m, model) in old["next"]["models"].as_array().unwrap().iter().enumerate() {
            for (j, &k) in SIZES.iter().enumerate() {
                let n = model["red_ranking"].as_array().unwrap()[..k]
                    .iter()
                    .filter(|n| d[249].red.contains(&(n.as_u64().unwrap() as u8)))
                    .count();
                assert_eq!(last.hits[m][j], n);
            }
        }
    }
}
