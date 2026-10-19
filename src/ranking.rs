//! A separate, versioned red ranking. Never changes the frozen protocol.
use crate::{
    draw::{fmt_issue, Draw},
    eval::Prepared,
    pool,
    select::{rank_desc, Forecast, RescueResult},
};
use serde_json::{json, Value};

pub const MODEL: &str = "repeat_calibrated";
pub const LABEL: &str = "重号偏置校准 · 试验";
pub const VERSION: u32 = 1;

pub fn policy() -> Value {
    json!({"model":MODEL,"version":VERSION,"label":LABEL,
        "method":"preserve_within_group_order_normalize_previous_group_mass",
        "previous_red_mass":6.0/33.0,"other_red_mass":27.0/33.0,
        "base_ranking":"frozen_weighted_fusion","blue_scope":"shared_weighted_blue_ranking",
        "score_kind":"ranking_evidence_not_probability","automatic_promotion":false,
        "note":"上期6红与其余27红分别归一化，按6:27分配总评分；保留两组内部的原融合顺序，不强制重号名额。修正集中偏置，尚未证实提高未来全6红率。"})
}

pub fn scores(original: &[f64; 33], previous: u64) -> [f64; 33] {
    assert_eq!(previous.count_ones(), 6);
    assert_eq!(previous >> 33, 0);
    let sums: [f64; 2] = std::array::from_fn(|g| {
        (0..33)
            .filter(|&i| usize::from(previous & (1 << i) != 0) == g)
            .map(|i| original[i])
            .sum()
    });
    std::array::from_fn(|i| {
        let g = usize::from(previous & (1 << i) != 0);
        let size = if g == 1 { 6.0 } else { 27.0 };
        if sums[g] > 0.0 {
            original[i] / sums[g] * size / 33.0
        } else {
            1.0 / 33.0
        }
    })
}

pub fn saved(prepared: &Prepared, previous: u64) -> Value {
    let score = scores(&prepared.forecast.fusion, previous);
    let order = rank_desc(&score);
    let pools: Vec<_> = pool::TIER_RED_SIZES
        .into_iter()
        .enumerate()
        .flat_map(|(i, r)| {
            let order = &order;
            pool::BLUE_SIZES.into_iter().map(move |b| {
                json!({"id":format!("{MODEL}_red{r}_blue{b}"),
            "model":MODEL,"group":"observation","tier_id":((b'A'+i as u8) as char).to_string(),
            "red_count":r,"blue_count":b,"reds":order[..r].iter().map(|n|n+1).collect::<Vec<_>>(),
            "blues":prepared.forecast.blue_order[..b].iter().map(|n|n+1).collect::<Vec<_>>()})
            })
        })
        .collect();
    json!({"model":MODEL,"version":VERSION,"policy":policy(),"target_issue":fmt_issue(prepared.cutoff_issue+1),
        "base_red_weights":prepared.weights,"base_red_scores":prepared.forecast.fusion.to_vec(),
        "base_red_ranking":prepared.forecast.red_order.iter().map(|n|n+1).collect::<Vec<_>>(),
        "original_previous_mass":(0..33).filter(|n|previous&(1<<n)!=0).map(|n|prepared.forecast.fusion[n]).sum::<f64>(),
        "previous_reds":(0..33).filter(|n|previous&(1<<n)!=0).map(|n|n+1).collect::<Vec<_>>(),
        "red_ranking":order.iter().map(|n|n+1).collect::<Vec<_>>(),"red_scores":score.to_vec(),
        "score_kind":"ranking_evidence_not_probability","red_weights":Value::Null,"pools":pools})
}

/// Used only by explicitly selected new presets; the protocol forecast stays intact.
pub fn prepare(draws: &[Draw]) -> Result<Prepared, String> {
    let prepared = crate::eval::prepare_forecast(draws)?;
    let previous = draws.last().ok_or("没有开奖数据")?.red_mask;
    let metadata = saved(&prepared, previous);
    let score = scores(&prepared.forecast.fusion, previous);
    Ok(apply(prepared, score, metadata))
}

/// Reuse the equal-weight ranking already frozen in comparison pools.
pub fn prepare_equal(draws: &[Draw]) -> Result<Prepared, String> {
    let prepared = crate::eval::prepare_forecast(draws)?;
    let score = crate::select::fuse7(&prepared.expert_red, &crate::select::equal7());
    let mut metadata = pool::saved(&prepared)["comparison_models"][0].clone();
    metadata["version"] = json!(1);
    metadata["target_issue"] = json!(fmt_issue(prepared.cutoff_issue + 1));
    metadata["base_red_ranking"] = json!(prepared.forecast.red_order.iter().map(|n| n + 1).collect::<Vec<_>>());
    metadata["policy"] = json!({"model":"equal","version":1,"label":"等权融合 · 对照",
        "method":"equal_weight_seven_red_experts","blue_scope":"shared_weighted_blue_ranking",
        "automatic_promotion":false,
        "note":"七个红球专家各占1/7，直接使用原等权对照排名；蓝球沿用协议排名。同数量持续观察，不按单期命中自动替换模型。"});
    Ok(apply(prepared, score, metadata))
}

pub(crate) fn apply(mut prepared: Prepared, score: [f64; 33], metadata: Value) -> Prepared {
    let order = rank_desc(&score);
    let members = order[..crate::protocol::Protocol::PRIMARY_K].to_vec();
    let mask = members.iter().fold(0, |a, n| a | (1 << n));
    let fc = prepared.forecast;
    let pool = order[..crate::protocol::Protocol::POOL].to_vec();
    let tickets = crate::select::build_tickets(
        &pool,
        &score,
        crate::protocol::Protocol::TICKETS,
        crate::protocol::Protocol::MAX_OVERLAP,
    );
    let ticket_blues = (0..tickets.len())
        .map(|i| fc.blue_order[i % crate::protocol::Protocol::BLUE_SLOTS])
        .collect();
    prepared.forecast = Forecast {
        fusion: score,
        red_order: order,
        blue: fc.blue,
        blue_order: fc.blue_order,
        mild: RescueResult {
            mask,
            members,
            rescued: vec![],
            displaced: vec![],
        },
        pool,
        tickets,
        ticket_blues,
    };
    prepared.ranking = Some(metadata);
    prepared
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removes_group_mass_bias_without_reordering_members() {
        let previous = (1 << 6) - 1;
        let old = std::array::from_fn(|i| {
            if i < 6 {
                1.0 + i as f64 * 0.01
            } else {
                0.05 + i as f64 * 0.001
            }
        });
        let new = scores(&old, previous);
        assert!((new[..6].iter().sum::<f64>() - 6.0 / 33.0).abs() < 1e-14);
        assert!((new.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        for i in 0..33 {
            for j in 0..33 {
                if (i < 6) == (j < 6) {
                    assert_eq!(old[i].total_cmp(&old[j]), new[i].total_cmp(&new[j]));
                }
            }
        }
        assert!(rank_desc(&new)[..14].iter().filter(|&&n| n < 6).count() < 6);
    }
    #[test]
    fn zero_mass_group_still_has_finite_equal_scores() {
        let new = scores(&[0.0; 33], (1 << 6) - 1);
        assert!(new
            .iter()
            .all(|v| v.is_finite() && (*v - 1.0 / 33.0).abs() < 1e-14));
    }
}
