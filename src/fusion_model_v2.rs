//! Frozen fusion v2: remove both explicit repeat-expert additions. Keep the
//! remaining expert ratios, EWMA/reference support and existing blue ranking.
use crate::{
    draw::Draw,
    eval::Prepared,
    select::{fuse7, rank_desc},
};
use serde_json::{json, Value};
pub const SIZES: [usize; 4] = [12, 14, 15, 16];
pub fn policy() -> Value {
    json!({"version":2,"method":"remove_repeat_expert_at_both_layers_then_renormalize_retained_weights",
        "explicit_repeat_weight":0,"retained_base_weight":2./3.,"ewma_weight":1./3.,"base_weight":0.9,"reference_support_weight":0.1,
        "blue_selection":"unchanged_original_top2","singles":"existing_balanced_coverage_8_tickets","dantuo":["2+5+2","3+5+1"],
        "red_sizes":SIZES,"score_kind":"ranking_evidence_not_probability","prediction_advantage_established":false,
        "note":"去除小池额外重号分及原融合里的显式重号专家，剩余权重按原比例归一化；保留参考支持、正常重号候选与蓝球排序。修复重复加分，不保证提高中奖率。"})
}
pub fn scores(
    experts: &[[f64; 33]; 7],
    weights: &[f64; 7],
    support: &[f64; 33],
) -> Result<([f64; 33], [f64; 7]), String> {
    if experts
        .iter()
        .flatten()
        .chain(support)
        .chain(weights)
        .any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("融合v2评分输入无效".into());
    }
    let total: f64 = weights[..6].iter().sum();
    if total <= 0. {
        return Err("融合v2没有可保留的专家权重".into());
    }
    let w = std::array::from_fn(|i| if i == 6 { 0. } else { weights[i] / total });
    let base = fuse7(experts, &w);
    let raw = std::array::from_fn(|i| {
        0.9 * ((2. / 3.) * base[i] + (1. / 3.) * experts[1][i]) + 0.1 * support[i]
    });
    Ok((raw, w))
}
fn plan(
    id: String,
    title: String,
    reds: Vec<u8>,
    dan: Vec<u8>,
    tuo: Vec<u8>,
    blues: Vec<u8>,
    tickets: Vec<Value>,
) -> Value {
    json!({"id":id,"title":title,"mode":if dan.is_empty(){"balanced"}else{"dantuo"},"reds":reds,"red_dan":dan,"red_tuo":tuo,"blues":blues,
        "red_count":if dan.is_empty(){tickets.iter().flat_map(|t|t["red"].as_array().unwrap().iter().filter_map(Value::as_u64)).collect::<std::collections::BTreeSet<_>>().len()}else{dan.len()+tuo.len()},
        "bet_count":tickets.len(),"cost":tickets.len()*2,"unit_price":2,"tickets":tickets})
}
pub fn build(prepared: &Prepared, parent: &Value, previous: &Draw) -> Result<Value, String> {
    let fusion = &parent["payload"]["reference_pools"]["fusion"];
    let support: Vec<f64> =
        serde_json::from_value(fusion["reference_support"].clone()).map_err(|_| "参考支持无效")?;
    let support: [f64; 33] = support.try_into().map_err(|_| "参考支持长度无效")?;
    let blues: Vec<u8> =
        serde_json::from_value(fusion["blues"].clone()).map_err(|_| "原蓝球无效")?;
    if blues.len() != 2 || blues[0] == blues[1] || blues.iter().any(|b| !(1..=16).contains(b)) {
        return Err("融合v2须有原排序2蓝".into());
    }
    let (raw, weights) = scores(&prepared.expert_red, &prepared.weights, &support)?;
    let order = rank_desc(&raw);
    let ranking: Vec<u8> = order.iter().map(|i| (i + 1) as u8).collect();
    let mut singles = Vec::new();
    for k in SIZES {
        let masks = crate::coverage::balanced_tickets(&order[..k], 8)?;
        let tickets: Vec<_> = masks
            .iter()
            .enumerate()
            .map(|(i, m)| json!({"red":crate::plan::mask_nums(*m),"blue":blues[i%2]}))
            .collect();
        singles.push(plan(
            format!("fusion_v2_red{k}"),
            format!("融合修正版v2 · {k}红8注"),
            ranking[..k].to_vec(),
            vec![],
            vec![],
            blues.clone(),
            tickets,
        ));
    }
    let mut dantuo = Vec::new();
    for (dan_n, blue_n) in [(2, 2), (3, 1)] {
        let dan = ranking[..dan_n].to_vec();
        let tuo = ranking[dan_n..dan_n + 5].to_vec();
        let selected = blues[..blue_n].to_vec();
        let mut tickets = Vec::new();
        for c in crate::plan::combinations(&tuo, 6 - dan_n)? {
            let mut red = dan.clone();
            red.extend(c);
            red.sort_unstable();
            for b in &selected {
                tickets.push(json!({"red":red,"blue":b}));
            }
        }
        let mut reds = dan.clone();
        reds.extend(&tuo);
        reds.sort_unstable();
        dantuo.push(plan(
            format!("{dan_n}+5+{blue_n}"),
            format!("融合修正版v2 · {dan_n}+5+{blue_n}"),
            reds,
            dan,
            tuo,
            selected,
            tickets,
        ));
    }
    let legacy_mass = previous
        .red
        .iter()
        .map(|n| {
            fusion["scores"][(n - 1) as usize]
                .as_f64()
                .ok_or("旧融合评分无效")
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum::<f64>();
    Ok(
        json!({"version":2,"policy":policy(),"ranking":ranking,"scores":raw.to_vec(),"blues":blues,"previous_reds":previous.red,
        "evidence":{"retained_expert_weights":weights,"explicit_repeat_weight":0.,"legacy_explicit_repeat_weight":0.225+0.45*prepared.weights[6],
            "previous_group_score_mass":previous.red.iter().map(|n|raw[(n-1)as usize]).sum::<f64>(),"legacy_previous_group_score_mass":legacy_mass,
            "top_six_previous_overlap":order[..6].iter().filter(|i|previous.red_mask&(1<<**i)!=0).count()},"singles":singles,"dantuo":dantuo}),
    )
}
pub fn reviewed(bundle: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut out = bundle.clone();
    let previous: Vec<u8> =
        serde_json::from_value(bundle["previous_reds"].clone()).map_err(|_| "缺少上期红球")?;
    for group in ["singles", "dantuo"] {
        for p in out[group].as_array_mut().ok_or("融合v2方案缺失")? {
            let r = crate::recommendations::review_strategy(
                &json!({"title":p["title"],"mode":p["mode"],"structural":true,"tickets":p["tickets"]}),
                draw,
            )?;
            p["ticket_results"] = r["tickets"].clone();
            p["summary"] = r["summary"].clone();
            p["previous_overlap"] = json!(p["tickets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["red"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|n| previous.iter().any(|p| json!(p) == **n))
                    .count())
                .collect::<Vec<_>>());
            p["dan_hits"] = json!(draw.map(|d| p["red_dan"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|n| d.red.iter().any(|r| json!(r) == **n))
                .count()));
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_repeat_distribution_cannot_change_corrected_score() {
        let mut experts = [[1. / 33.; 33]; 7];
        let w = [1. / 7.; 7];
        let support = [1. / 33.; 33];
        let (before, weights) = scores(&experts, &w, &support).unwrap();
        experts[6] = std::array::from_fn(|i| if i < 6 { 1. / 6. } else { 0. });
        let (after, _) = scores(&experts, &w, &support).unwrap();
        assert_eq!(before, after);
        assert_eq!(weights[6], 0.);
        assert!((before.iter().sum::<f64>() - 1.).abs() < 1e-12);
        for i in 0..6 {
            assert!((weights[i] / weights[0] - w[i] / w[0]).abs() < 1e-12);
        }
    }
    #[test]
    fn ordinary_previous_numbers_are_not_excluded() {
        let mut experts = [[1. / 33.; 33]; 7];
        experts[0][0] = 1.;
        let (s, _) = scores(&experts, &[1. / 7.; 7], &[1. / 33.; 33]).unwrap();
        assert!(s[0] > s[20]);
    }
    #[test]
    fn invalid_scores_reject_instead_of_silently_falling_back() {
        assert!(scores(
            &[[1. / 33.; 33]; 7],
            &[0., 0., 0., 0., 0., 0., 1.],
            &[1. / 33.; 33]
        )
        .is_err());
    }
}
