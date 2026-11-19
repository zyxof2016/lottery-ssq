//! v4-caliber red ranking shared by the observation pools, the rank board and
//! the generation page: the six retained experts renormalized so EWMA is
//! counted exactly once, with neutral reference support (a constant shift that
//! cannot change the order). The fusion recommendation v4 additionally mixes
//! the real 10% reference vote; this module is the support-free caliber.
//! Frozen publication rules and sealed archives are untouched.
use crate::{
    draw::{fmt_issue, Draw},
    eval::{prepare_forecast, Prepared},
    pool,
    select::{fuse7, rank_desc},
};
use serde_json::{json, Value};

pub const MODEL: &str = "fusion_v4";
pub const LABEL: &str = "融合v4口径 · EWMA单计";
pub const VERSION: u32 = 1;

pub fn policy() -> Value {
    json!({"version":VERSION,"model":MODEL,"label":LABEL,
        "method":"six_retained_experts_renormalized_ewma_counted_once",
        "explicit_repeat_weight":0,"extra_ewma_weight":0,
        "reference_support":"neutral_uniform_constant_shift_order_identical_to_six_expert_fusion",
        "weight_boundary":"protocol_weights_frozen_before_25111",
        "blue_scope":"shared_weighted_blue_ranking",
        "score_kind":"ranking_evidence_not_probability",
        "relation":"融合推荐v4另加10%参考并集支持；本口径为无支持版本，供观察池、排序与生成页使用。",
        "automatic_promotion":false})
}

pub fn weights(w: &[f64; 7]) -> Result<[f64; 7], String> {
    if w.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err("融合v4口径权重无效".into());
    }
    let total: f64 = w[..6].iter().sum();
    if total <= 0. {
        return Err("融合v4口径没有可保留的专家权重".into());
    }
    Ok(std::array::from_fn(|i| if i == 6 { 0. } else { w[i] / total }))
}

pub fn scores(experts: &[[f64; 33]; 7], w: &[f64; 7]) -> Result<[f64; 33], String> {
    Ok(fuse7(experts, &weights(w)?))
}

pub fn order(experts: &[[f64; 33]; 7], w: &[f64; 7]) -> Result<[usize; 33], String> {
    Ok(rank_desc(&scores(experts, w)?))
}

pub fn saved(prepared: &Prepared) -> Value {
    let score = scores(&prepared.expert_red, &prepared.weights).expect("协议权重非负且前六项和为正");
    let order = rank_desc(&score);
    let pools: Vec<_> = pool::TIER_RED_SIZES
        .into_iter()
        .enumerate()
        .flat_map(|(i, r)| {
            let order = &order;
            pool::BLUE_SIZES.into_iter().map(move |b| {
                json!({"id":format!("{MODEL}_red{r}_blue{b}"),
                    "model":MODEL,"group":"observation","tier_id":((b'A'+i as u8) as char).to_string(),
                    "red_count":r,"blue_count":b,
                    "reds":order[..r].iter().map(|n|n+1).collect::<Vec<_>>(),
                    "blues":prepared.forecast.blue_order[..b].iter().map(|n|n+1).collect::<Vec<_>>()})
            })
        })
        .collect();
    json!({"model":MODEL,"version":VERSION,"policy":policy(),
        "target_issue":fmt_issue(prepared.cutoff_issue+1),
        "input_history_sha256":prepared.data_sha,
        "base_red_weights":prepared.weights,
        "retained_expert_weights":weights(&prepared.weights).expect("协议权重有效"),
        "red_ranking":order.iter().map(|n|n+1).collect::<Vec<_>>(),
        "red_scores":score.to_vec(),
        "score_kind":"ranking_evidence_not_probability",
        "pools":pools})
}

pub fn validate_history(value: &Value, prefix: &[Draw]) -> Result<(), String> {
    if value["model"] != MODEL || value["version"] != VERSION {
        return Err("融合v4口径模型版本不符".into());
    }
    let prepared = prepare_forecast(prefix)?;
    if value != &saved(&prepared) {
        return Err("融合v4口径排名与输入历史重算不一致".into());
    }
    Ok(())
}

/// Generation-page prepared forecast whose red order and tickets use the
/// v4 caliber; blue ranking and protocol weights stay intact.
pub fn prepare(draws: &[Draw]) -> Result<Prepared, String> {
    let mut prepared = prepare_forecast(draws)?;
    let metadata = saved(&prepared);
    let w = weights(&prepared.weights)?;
    prepared.forecast = crate::select::forecast(
        &prepared.expert_red,
        &prepared.expert_blue,
        &w,
        &prepared.blue_weights,
    );
    prepared.ranking = Some(metadata);
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_support_shift_cannot_change_the_order_and_repeat_expert_is_drop() {
        let mut experts = [[1. / 33.; 33]; 7];
        experts[1] = std::array::from_fn(|i| (i + 1) as f64 / 561.);
        let w = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1];
        let before = order(&experts, &w).unwrap();
        experts[6] = std::array::from_fn(|i| if i < 6 { 1. / 6. } else { 0. });
        assert_eq!(before, order(&experts, &w).unwrap());
        let base = scores(&experts, &w).unwrap();
        let shifted: [f64; 33] = std::array::from_fn(|i| 0.9 * base[i] + 0.1 / 33.);
        assert_eq!(rank_desc(&base), rank_desc(&shifted));
        assert!(weights(&[0., 0., 0., 0., 0., 0., 1.]).is_err());
    }
    #[test]
    fn saved_roundtrips_against_the_same_history_and_rejects_other_inputs() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let prepared = prepare_forecast(&draws).unwrap();
        let value = saved(&prepared);
        validate_history(&value, &draws).unwrap();
        assert!(validate_history(&value, &draws[..draws.len() - 1]).is_err());
        let mut tampered = value.clone();
        tampered["red_ranking"][0] = json!(33);
        assert!(validate_history(&tampered, &draws).is_err());
        let gen = prepare(&draws).unwrap();
        assert_eq!(
            gen.forecast.red_order,
            order(&prepared.expert_red, &prepared.weights).unwrap()
        );
    }
}