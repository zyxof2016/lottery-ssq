//! Red pool cardinality and all-six recall, independently of bet construction.
use serde_json::{json, Value};

use crate::draw::{data_sha256, fmt_date, fmt_issue, Draw};
use crate::eval::{prepare_forecast, Prepared};
use crate::model::roll_all;
use crate::protocol::Protocol;
use crate::select::{equal7, fuse6, fuse7, rank_desc, rank_log_score, softmax_shrink};
use crate::stats::{binom, wilson, Rng};

pub const VERSION: u32 = 3;
pub const RANKED_VERSION: u32 = 4;
pub const OBSERVATION_MODELS: [&str; 4] = ["weighted", "equal", "uniform", crate::ranking::MODEL];
pub const COMPARISON_MODELS: [&str; 3] = ["weighted", "equal", "uniform"];
pub const COMPARISON_LABELS: [&str; 3] = ["协议加权融合", "等权融合", "均匀红球对照"];
pub const REPORT_VERSION: u32 = 3;
// Keep v1 sizes for legacy replay cells and archive validation.
pub const RED_SIZES: [usize; 3] = [8, 10, 12];
pub const TIER_RED_SIZES: [usize; 6] = [14, 16, 18, 20, 22, 24];
pub const ARCHIVED_RED_SIZES: [usize; 9] = [8, 10, 12, 14, 16, 18, 20, 22, 24];
pub const FOCUS_RED_MIN: usize = 14;
pub const FOCUS_RED_MAX: usize = 24;
pub const BLUE_SIZES: [usize; 3] = [1, 2, 3];
pub const MODELS: [&str; 9] = [
    "weighted",
    "equal",
    "frequency",
    "ewma",
    "overdue",
    "hot",
    "cooccur",
    "zone",
    "repeat",
];
pub const LABELS: [&str; 9] = [
    "协议加权融合",
    "等权融合",
    "长期频率",
    "指数近期",
    "遗漏回补",
    "近期热号",
    "上期共现",
    "区间均衡",
    "上期重号",
];

pub fn tier_definitions() -> Vec<Value> {
    TIER_RED_SIZES
        .iter()
        .enumerate()
        .map(|(i, &red_count)| {
            let id = ((b'A' + i as u8) as char).to_string();
            json!({"id": id, "label": format!("{id}档"), "red_count": red_count})
        })
        .collect()
}

#[derive(Clone)]
struct Period {
    issue: u32,
    red_ranks: [[usize; 6]; 9],
    blue_rank: usize,
}

fn red_orders(expert: &[[f64; 33]; 7], weights: &[f64; 7]) -> [[usize; 33]; 9] {
    let mut orders = [[0; 33]; 9];
    orders[0] = rank_desc(&fuse7(expert, weights));
    orders[1] = rank_desc(&fuse7(expert, &equal7()));
    for e in 0..7 {
        orders[e + 2] = rank_desc(&expert[e]);
    }
    orders
}

fn periods(draws: &[Draw]) -> Result<Vec<Period>, String> {
    let snapshot = roll_all(draws)?;
    let mut red_sum = [0.0; 7];
    let mut blue_sum = [0.0; 6];
    let mut n_fit = 0;
    let mut out = Vec::new();
    for row in &snapshot.rows {
        // Fit on earlier OOS rows only; stop updating at the frozen boundary.
        if n_fit >= 30 {
            let orders = red_orders(&row.expert_p, &softmax_shrink(&red_sum, n_fit as f64));
            let blue_order = rank_desc(&fuse6(
                &row.blue_p,
                &softmax_shrink(&blue_sum, n_fit as f64),
            ));
            let mut red_ranks = [[0; 6]; 9];
            for (model, order) in orders.iter().enumerate() {
                let ranks: Vec<_> = order
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| row.red_mask & (1 << **i) != 0)
                    .map(|(i, _)| i + 1)
                    .collect();
                red_ranks[model].copy_from_slice(&ranks);
            }
            out.push(Period {
                issue: row.issue,
                red_ranks,
                blue_rank: blue_order
                    .iter()
                    .position(|i| *i == row.blue as usize)
                    .expect("blue rank")
                    + 1,
            });
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            for e in 0..6 {
                blue_sum[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.0).ln();
            }
            n_fit += 1;
        }
    }
    Ok(out)
}

pub fn uniform_full_red(red_count: usize) -> f64 {
    binom(red_count as u64, 6) as f64 / binom(33, 6) as f64
}

fn rate_ci(hits: usize, n: usize) -> Value {
    let (low, high) = wilson(hits as f64, n as f64, 1.959963984540054);
    json!([low, high])
}

fn summarize(rows: &[&Period]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let n = rows.len();
    let mut cells = Vec::new();
    let mut red_curves = Vec::new();
    let mut red_frontiers = Vec::new();
    for model in 0..MODELS.len() {
        let mut previous_full = 0;
        let mut frontier_full = None;
        let mut frontier = Vec::new();
        for red_count in 6..=33 {
            let mut histogram = [0usize; 7];
            for row in rows {
                histogram[row.red_ranks[model]
                    .iter()
                    .filter(|&&r| r <= red_count)
                    .count()] += 1;
            }
            let sum_hits: usize = histogram.iter().enumerate().map(|(h, n)| h * n).sum();
            let mean = sum_hits as f64 / n as f64;
            let joint_counts: Vec<_> = BLUE_SIZES
                .iter()
                .map(|&b| {
                    rows.iter()
                        .filter(|r| {
                            r.blue_rank <= b
                                && r.red_ranks[model].iter().all(|&rank| rank <= red_count)
                        })
                        .count()
                })
                .collect();
            let blue_counts: Vec<_> = BLUE_SIZES
                .iter()
                .map(|&b| rows.iter().filter(|r| r.blue_rank <= b).count())
                .collect();
            let null_rate = uniform_full_red(red_count);
            red_curves.push(json!({
                "model": MODELS[model], "label": LABELS[model], "red_count": red_count,
                "mean_red_hits": mean, "mean_missed_reds": 6.0 - mean, "red_draw_recall": mean / 6.0,
                "at_least_5_red_periods": histogram[5] + histogram[6],
                "full_red_periods": histogram[6], "full_red_rate": histogram[6] as f64 / n as f64,
                "full_red_wilson95": rate_ci(histogram[6], n),
                "uniform_full_red_rate": null_rate,
                "full_red_rate_excess": histogram[6] as f64 / n as f64 - null_rate,
                "additional_full_red_periods": if red_count > 6 { Some(histogram[6] - previous_full) } else { None },
                "uniform_mean_red_hits": 6.0 * red_count as f64 / 33.0,
                "full_draw_periods_by_blue": joint_counts, "blue_covered_periods_by_count": blue_counts,
            }));
            previous_full = histogram[6];
            if (FOCUS_RED_MIN..=FOCUS_RED_MAX).contains(&red_count)
                && frontier_full.is_none_or(|h| histogram[6] > h)
            {
                frontier.push(
                    json!({"red_count": red_count, "full_red_periods": histogram[6],
                    "full_red_rate": histogram[6] as f64 / n as f64}),
                );
                frontier_full = Some(histogram[6]);
            }
            // Retain the original nine-specification API cells for compatibility.
            if !RED_SIZES.contains(&red_count) {
                continue;
            }
            for blue_count in BLUE_SIZES {
                let blue_hits = rows.iter().filter(|r| r.blue_rank <= blue_count).count();
                let joint = rows
                    .iter()
                    .filter(|r| {
                        r.blue_rank <= blue_count
                            && r.red_ranks[model].iter().all(|&rank| rank <= red_count)
                    })
                    .count();
                cells.push(json!({
                    "model": MODELS[model], "label": LABELS[model], "red_count": red_count, "blue_count": blue_count,
                    "mean_red_hits": mean, "mean_missed_reds": 6.0 - mean,
                    "red_draw_recall": mean / 6.0, "red_selection_precision": mean / red_count as f64,
                    "red_hit_histogram": histogram, "at_least_5_red_periods": histogram[5] + histogram[6],
                    "full_red_periods": histogram[6], "full_red_rate": histogram[6] as f64 / n as f64,
                    "full_red_wilson95": rate_ci(histogram[6], n),
                    "blue_covered_periods": blue_hits, "blue_coverage_rate": blue_hits as f64 / n as f64,
                    "full_draw_periods": joint, "full_draw_rate": joint as f64 / n as f64,
                    "full_draw_wilson95": rate_ci(joint, n),
                    "uniform_mean_red_hits": 6.0 * red_count as f64 / 33.0,
                    "red_hit_excess": mean - 6.0 * red_count as f64 / 33.0,
                    "uniform_full_red_rate": uniform_full_red(red_count),
                    "uniform_full_draw_rate": uniform_full_red(red_count) * blue_count as f64 / 16.0,
                }));
            }
        }
        red_frontiers
            .push(json!({"model": MODELS[model], "label": LABELS[model], "points": frontier}));
    }
    json!({"n": n, "from": fmt_issue(rows[0].issue), "to": fmt_issue(rows[n - 1].issue),
        "cells": cells, "red_curves": red_curves, "red_frontiers": red_frontiers})
}

/// Archive pre-draw rankings, six observation tiers, and the legacy small pools.
/// Scores express rank evidence, never calibrated probabilities of inclusion.
pub fn saved(prepared: &Prepared) -> Value {
    let fc = &prepared.forecast;
    let mut pools = Vec::new();
    for red_count in ARCHIVED_RED_SIZES {
        let tier = TIER_RED_SIZES.iter().position(|&r| r == red_count);
        for blue_count in BLUE_SIZES {
            pools.push(
                json!({"id": format!("red{red_count}_blue{blue_count}"), "model": "weighted",
            "group": if tier.is_some() { "observation" } else { "legacy" },
            "tier_id": tier.map(|i| ((b'A' + i as u8) as char).to_string()),
            "red_count": red_count, "blue_count": blue_count,
            "reds": fc.red_order[..red_count].iter().map(|i| *i + 1).collect::<Vec<_>>(),
            "blues": fc.blue_order[..blue_count].iter().map(|i| *i + 1).collect::<Vec<_>>() }),
            );
        }
    }
    let equal_scores = fuse7(&prepared.expert_red, &equal7());
    let equal_ranking: Vec<_> = rank_desc(&equal_scores).iter().map(|n| *n as u8 + 1).collect();
    let issue = fmt_issue(prepared.cutoff_issue + 1);
    let uniform_ranking = uniform_ranking(&issue);
    let comparisons: Vec<_> = [("equal", &equal_ranking), ("uniform", &uniform_ranking)]
        .into_iter().map(|(model, ranking)| {
            let pools: Vec<_> = TIER_RED_SIZES.into_iter().enumerate().flat_map(|(i, r)| {
                BLUE_SIZES.into_iter().map(move |b| json!({"id": format!("{model}_red{r}_blue{b}"),
                    "model": model, "group": "observation", "tier_id": ((b'A' + i as u8) as char).to_string(),
                    "red_count": r, "blue_count": b, "reds": ranking[..r],
                    "blues": fc.blue_order[..b].iter().map(|n| *n + 1).collect::<Vec<_>>() }))
            }).collect();
            json!({"model": model, "red_ranking": ranking, "pools": pools,
                "red_scores": if model == "equal" { Some(equal_scores.to_vec()) } else { None },
                "red_weights": if model == "equal" { Some(equal7().to_vec()) } else { None },
                "score_kind": if model == "equal" { "ranking_evidence_not_probability" } else { "fixed_seed_uniform_control" }})
        }).collect();
    json!({"version": VERSION, "objective": "all_six_reds_with_fewer_selected_reds", "pools": pools,
        "target_issue": issue, "comparison_models": comparisons, "comparison_policy": comparison_policy(),
        "observation_tiers": tier_definitions(), "default_blue_count": 2,
        "red_ranking": fc.red_order.iter().map(|i| *i + 1).collect::<Vec<_>>(),
        "blue_ranking": fc.blue_order.iter().map(|i| *i + 1).collect::<Vec<_>>(),
        "red_scores": fc.fusion.to_vec(), "blue_scores": fc.blue.to_vec(),
        "red_weights": prepared.weights, "blue_weights": prepared.blue_weights,
        "score_kind": "ranking_evidence_not_probability"})
}

pub fn comparison_policy() -> Value {
    json!({"version": 1, "models": COMPARISON_MODELS, "labels": COMPARISON_LABELS,
        "red_sizes": TIER_RED_SIZES, "blue_sizes": BLUE_SIZES,
        "primary_metric": "full_red_rate_at_same_red_count", "secondary_metric": "mean_missed_reds",
        "blue_scope": "shared_weighted_blue_ranking", "paired_periods_only": true,
        "seed": format!("0x{:016x}", Protocol::SEED), "uniform_stream": "ssq-pool-comparison-v1:seed:issue:red",
        "automatic_promotion": false,
        "note": "规则从首次保存三模型的期次起前瞻观察；只比较相同期次、相同红蓝数量。旧期不补造对照，不按近期胜率自动换模型。"})
}

pub fn ranked_comparison_policy() -> Value {
    let mut value = comparison_policy();
    value["version"] = json!(2);
    value["models"] = json!(OBSERVATION_MODELS);
    value["labels"] = json!([COMPARISON_LABELS[0],COMPARISON_LABELS[1],COMPARISON_LABELS[2],crate::ranking::LABEL]);
    value["ranking_policy"] = crate::ranking::policy();
    value["note"] = json!("新期同时冻结原三模型和重号偏置校准试验；旧档案不追加。新旧排名只比较共同保存的期次及相同红蓝数量，未验证稳定优势，不自动替换协议模型。");
    value
}

pub fn saved_ranked(prepared: &Prepared, previous: u64) -> Value {
    let mut value = saved(prepared);
    value["version"] = json!(RANKED_VERSION);
    value["comparison_policy"] = ranked_comparison_policy();
    value["comparison_models"].as_array_mut().unwrap().push(crate::ranking::saved(prepared,previous));
    value
}

/// Optional independently versioned extension; v3/v4 archives stay unchanged.
pub fn saved_current(prepared:&Prepared,draws:&[Draw])->Value {
    let mut value=saved_ranked(prepared,draws.last().unwrap().red_mask);
    value["coverage_model"]=crate::conditional::saved(prepared,draws);
    value
}

fn uniform_ranking(issue: &str) -> Vec<u8> {
    let hash = crate::hash::sha256_hex(format!("ssq-pool-comparison-v1:{}:{issue}:red", Protocol::SEED).as_bytes());
    let mut rng = Rng::new(u64::from_str_radix(&hash[..16], 16).expect("sha256 prefix"));
    let mut numbers: Vec<_> = (1..=33).collect();
    for i in (1..numbers.len()).rev() { numbers.swap(i, rng.gen_range(i + 1)); }
    numbers
}

fn ranking(saved: &Value, key: &str, size: usize) -> Result<Option<Vec<u8>>, String> {
    if saved[key].is_null() { return Ok(None); }
    let values: Vec<u8> = serde_json::from_value(saved[key].clone()).map_err(|_| "原始排名格式无效")?;
    if values.len() != size || values.iter().any(|&n| n == 0 || n as usize > size)
        || values.iter().collect::<std::collections::BTreeSet<_>>().len() != size {
        return Err("原始排名须完整且号码不重复".into());
    }
    Ok(Some(values))
}

fn diagnostics(order: Option<&[u8]>, blue: Option<&[u8]>, draw: Option<&Draw>) -> Value {
    let (Some(order), Some(draw)) = (order, draw) else { return Value::Null; };
    let winners: Vec<_> = draw.red.iter().map(|n| json!({"number": n,
        "rank": order.iter().position(|r| r == n).unwrap() + 1})).collect();
    let minimum = winners.iter().map(|v| v["rank"].as_u64().unwrap()).max().unwrap();
    json!({"kind": "post_draw_saved_ranking_diagnostic", "is_recommended_pool": false,
        "winning_red_ranks": winners, "minimum_full_red_prefix": minimum,
        "winning_blue_rank": blue.map(|b| b.iter().position(|n| *n == draw.blue).unwrap() + 1),
        "note": "按开奖前保存的完整排名计算，仅用于漏号诊断；这个最小前缀不是当时推荐的号码池。"})
}

/// A normalized view of saved models; never constructs absent comparisons.
pub fn reviewed_models(reviewed: &Value) -> Vec<Value> {
    if reviewed.is_null() { return Vec::new(); }
    let mut models = vec![json!({"model": "weighted", "pools": reviewed["pools"],
        "diagnostics": reviewed["diagnostics"], "smallest_full_red_tier": reviewed["smallest_full_red_tier"]})];
    if reviewed["version"] == 3 || reviewed["version"] == RANKED_VERSION {
        models.extend(reviewed["comparison_models"].as_array().into_iter().flatten().cloned());
    }
    if reviewed["version"]==RANKED_VERSION && !reviewed["coverage_model"].is_null() {models.push(reviewed["coverage_model"].clone());}
    models
}

/// Score the originally saved pools, without running a model or building bets.
pub fn review(saved: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    if saved.is_null() {
        return Ok(Value::Null);
    }
    let version = saved["version"].as_u64().ok_or("缺少选号池版本")?;
    let allowed_red_sizes = match version {
        1 => RED_SIZES.as_slice(),
        2 | 3 | 4 => ARCHIVED_RED_SIZES.as_slice(),
        _ => return Err("不支持的选号池档案版本".into()),
    };
    let red_order = ranking(saved, "red_ranking", 33)?;
    let blue_order = ranking(saved, "blue_ranking", 16)?;
    let strict = version >= 3;
    let expected_policy = if version == 4 {ranked_comparison_policy()} else {comparison_policy()};
    if strict && (red_order.is_none() || blue_order.is_none()
        || saved["comparison_policy"] != expected_policy) {
        return Err("模型档案缺少完整排名或固定对照规则".into());
    }
    let mut result = review_set(saved, draw, allowed_red_sizes, red_order.as_deref(), blue_order.as_deref(), "weighted", strict)?;
    if strict {
        let comparisons = saved["comparison_models"].as_array().ok_or("缺少对照模型")?;
        let allowed = if version == 4 {&OBSERVATION_MODELS[1..]} else {&COMPARISON_MODELS[1..]};
        if comparisons.len() != allowed.len() { return Err("档案缺少完整对照模型".into()); }
        let mut reviewed = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for model in comparisons {
            let name = model["model"].as_str().ok_or("缺少对照名称")?;
            if !allowed.contains(&name) || !names.insert(name) {
                return Err("对照模型重复或无效".into());
            }
            let order = ranking(model, "red_ranking", 33)?.ok_or("对照缺少完整红球排名")?;
            if name == crate::ranking::MODEL {
                let previous:Vec<u8>=serde_json::from_value(model["previous_reds"].clone()).map_err(|_|"校准排名缺少上期号码")?;
                if previous.len()!=6 || previous.iter().any(|&n|!(1..=33).contains(&n))
                    || previous.iter().collect::<std::collections::BTreeSet<_>>().len()!=6
                    || model["version"]!=crate::ranking::VERSION || model["policy"]!=crate::ranking::policy()
                    || model["target_issue"]!=saved["target_issue"] {return Err("校准排名规则或上期号码无效".into());}
                let mask=previous.iter().fold(0,|m,n|m|(1u64<<(n-1)));
                let original:[f64;33]=serde_json::from_value::<Vec<f64>>(saved["red_scores"].clone()).map_err(|_|"缺少原融合分数")?.try_into().map_err(|_|"原分数须有33项")?;
                if original.iter().any(|v|!v.is_finite()||*v<0.0) {return Err("原融合分数无效".into());}
                let actual:[f64;33]=serde_json::from_value::<Vec<f64>>(model["red_scores"].clone()).map_err(|_|"缺少校准分数")?.try_into().map_err(|_|"校准分数须有33项")?;
                let expected=crate::ranking::scores(&original,mask);
                if actual.iter().zip(expected).any(|(a,b)|!a.is_finite() || (*a-b).abs()>1e-12)
                    || order!=rank_desc(&expected).iter().map(|n|*n as u8+1).collect::<Vec<_>>() {return Err("校准分数与保存的排名不一致".into());}
            }
            reviewed.push(review_set(model, draw, &TIER_RED_SIZES, Some(&order), blue_order.as_deref(), name, true)?);
        }
        result["comparison_models"] = json!(reviewed);
        if version==4 && !saved["coverage_model"].is_null() {
            let model=&saved["coverage_model"];
            crate::conditional::validate(model,red_order.as_ref().unwrap(),&saved["target_issue"])?;
            let order=ranking(model,"red_ranking",33)?.ok_or("条件频率缺少完整排名")?;
            result["coverage_model"]=review_set(model,draw,&TIER_RED_SIZES,Some(&order),blue_order.as_deref(),crate::conditional::MODEL,true)?;
        } else {result.as_object_mut().unwrap().remove("coverage_model");}
    } else {
        // A v1/v2 archive never gains a model it did not save.
        result.as_object_mut().unwrap().remove("comparison_models");
        result.as_object_mut().unwrap().remove("comparison_policy");
        result.as_object_mut().unwrap().remove("coverage_model");
    }
    Ok(result)
}

fn review_set(saved: &Value, draw: Option<&Draw>, allowed_red_sizes: &[usize], red_order: Option<&[u8]>,
    blue_order: Option<&[u8]>, model: &str, strict: bool) -> Result<Value, String> {
    let mut result = saved.clone();
    let mut specs = std::collections::BTreeSet::new();
    for pool in result["pools"]
        .as_array_mut()
        .ok_or("选号池档案缺少 pools")?
    {
        let reds: Vec<u8> =
            serde_json::from_value(pool["reds"].clone()).map_err(|e| e.to_string())?;
        let blues: Vec<u8> =
            serde_json::from_value(pool["blues"].clone()).map_err(|e| e.to_string())?;
        if !allowed_red_sizes.contains(&reds.len())
            || !BLUE_SIZES.contains(&blues.len())
            || reds.iter().any(|n| !(1..=33).contains(n))
            || blues.iter().any(|n| !(1..=16).contains(n))
            || reds.iter().collect::<std::collections::BTreeSet<_>>().len() != reds.len()
            || blues
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != blues.len()
            || pool["red_count"] != reds.len()
            || pool["blue_count"] != blues.len()
            || !specs.insert((reds.len(), blues.len()))
        {
            return Err("选号池档案号码或数量无效".into());
        }
        if strict {
            let reds_set: std::collections::BTreeSet<_> = reds.iter().copied().collect();
            let blues_set: std::collections::BTreeSet<_> = blues.iter().copied().collect();
            if pool["model"] != model
                || reds_set != red_order.unwrap()[..reds.len()].iter().copied().collect()
                || blues_set != blue_order.unwrap()[..blues.len()].iter().copied().collect()
                || pool["group"] != if TIER_RED_SIZES.contains(&reds.len()) { "observation" } else { "legacy" } {
                return Err("号码池与保存的模型排名或共享蓝球排名不一致".into());
            }
        }
        let hits = draw.map(|d| {
            d.red
                .iter()
                .copied()
                .filter(|n| reds.contains(n))
                .collect::<Vec<_>>()
        });
        pool["hit_reds"] = json!(hits);
        pool["missed_reds"] = json!(draw.map(|d| d
            .red
            .iter()
            .copied()
            .filter(|n| !reds.contains(n))
            .collect::<Vec<_>>()));
        pool["red_hits"] = json!(hits.as_ref().map(Vec::len));
        pool["missed_red_ranks"] = json!(draw.zip(red_order).map(|(d, order)| d.red.iter()
            .filter(|n| !reds.contains(n)).map(|n| json!({"number": n,
                "rank": order.iter().position(|r| r == n).unwrap() + 1})).collect::<Vec<_>>()));
        pool["full_red"] = json!(hits.as_ref().map(|r| r.len() == 6));
        pool["blue_hit"] = json!(draw.map(|d| blues.contains(&d.blue)));
        pool["full_draw"] =
            json!(draw
                .map(|d| hits.as_ref().is_some_and(|r| r.len() == 6) && blues.contains(&d.blue)));
    }
    if strict && specs.len() != allowed_red_sizes.len() * BLUE_SIZES.len() {
        return Err("三模型号码池规格不完整".into());
    }
    result["diagnostics"] = diagnostics(red_order, blue_order, draw);
    result["smallest_full_red_tier"] = json!(result["pools"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["full_red"] == true && p["group"] == "observation")
        .filter_map(|p| p["red_count"].as_u64())
        .min());
    Ok(result)
}

pub fn analyze(draws: &[Draw]) -> Result<Value, String> {
    let prepared = prepare_forecast(draws)?;
    let rows = periods(draws)?;
    let recent = |n: usize| {
        summarize(
            &rows[rows.len().saturating_sub(n)..]
                .iter()
                .collect::<Vec<_>>(),
        )
    };
    let confirm_start = rows
        .iter()
        .position(|r| r.issue == Protocol::CONFIRM_START_ISSUE);
    let confirm_end = confirm_start.and_then(|s| {
        (s + Protocol::CONFIRMATION_LEN <= rows.len()).then_some(s + Protocol::CONFIRMATION_LEN)
    });
    let orders = red_orders(&prepared.expert_red, &prepared.weights);
    let next_models: Vec<_> = orders.iter().enumerate().map(|(i, order)| json!({
        "model": MODELS[i], "label": LABELS[i], "red_ranking": order.iter().map(|n| *n + 1).collect::<Vec<_>>()
    })).collect();
    Ok(json!({
        "version": REPORT_VERSION, "archive_version": RANKED_VERSION,
        "kind": "exploratory_small_pool_replay", "objective": "all_six_reds_with_fewer_selected_reds",
        "data_sha256": data_sha256(draws), "protocol_sha256": Protocol::sha256(),
        "source_sha256": crate::hash::sha256_hex(concat!(include_str!("pool.rs"), include_str!("model.rs"), include_str!("select.rs"), include_str!("protocol.rs"), include_str!("eval.rs"), include_str!("stats.rs"), include_str!("ranking.rs"), include_str!("conditional.rs")).as_bytes()),
        "cutoff_issue": fmt_issue(prepared.cutoff_issue), "cutoff_date": fmt_date(prepared.cutoff_date), "suggested_target": fmt_issue(prepared.cutoff_issue + 1),
        "red_sizes": (6..=33).collect::<Vec<_>>(), "archived_red_sizes": ARCHIVED_RED_SIZES, "blue_sizes": BLUE_SIZES,
        "observation_tiers": tier_definitions(), "default_observation_blue_count": 2,
        "focus_red_range": [FOCUS_RED_MIN, FOCUS_RED_MAX],
        "windows": {"recent30": recent(30), "recent60": recent(60),
            "confirmation": confirm_end.map(|e| summarize(&rows[confirm_start.unwrap()..e].iter().collect::<Vec<_>>())),
            "tracking": confirm_end.map(|e| summarize(&rows[e..].iter().collect::<Vec<_>>())),
            "all_oos": summarize(&rows.iter().collect::<Vec<_>>())},
        "next": {"actual_recommendation": false, "known_outcome_at_generation": false,
            "candidate_models": next_models, "weighted_pools": review(&saved_current(&prepared, draws), None)?},
        "notes": [
            "主要目标是用更少红球包含全部6个开奖红球；先比较14–24红，全6红率为主指标，蓝球1–3个及6红1蓝全包作辅助。",
            "全部历史为事后因果回放，不冒充实际发布。权重先用更早样本外记录展开拟合，25111起冻结；每期专家状态只用该期之前的数据。",
            "9个红球排序方法均使用相同的冻结蓝球排序，以同一池大小比较红球筛选能力；只报告全部结果，不事后挑冠军或调种子。",
            "确认窗内新增的这些指标仍属探索性复盘，不加入原25项检验；最近30/60期互相重叠，Wilson区间未校正多重比较或时间相关。",
            "全包6红1蓝指开奖号码在所选号码池中，不表示已经组成中奖单注。小池预览不自动展开复式或购票。",
            "模型分数不是覆盖概率。缩小池会提高漏号风险，当前没有验证可用小池稳定包住全部开奖号。",
            "6–33红曲线报告全部大小；14–24红前沿只描述当前窗口中全包期数上升所需的最小数量，不能当作未来最优数量。",
            "扩大池本身会提高均匀全6红基准；全包率上升只有在相同大小下持续超过基准，才可能支持筛选能力提升。",
            "新期留档使用14/16/18/20/22/24红六档，各配1–3蓝，原加权另保留8/10/12红；加权、等权、均匀及校准试验各用自身红球排名，共用原蓝球排序，大档包含小档。",
            "校准排名的历史同档位比较见覆盖复盘；它不加入原九模型检验，不替换协议模型，已有档案不追加。",
            "实际留档统计仅核对当期原池；没有保存观察档的旧期不补造。研究回放与实际留档分开计数，样本为0时实际覆盖率为空。"
        ]
    }))
}

pub fn render(report: &Value) -> String {
    let mut text = format!(
        "红球数量与全6红覆盖复盘（截止 {}）\n观察档：14/16/18/20/22/24红，各配1–3蓝；继续比较14–24红的数量与全6红率。\n下表为协议加权融合；JSON保留全部9种排序的6–33红曲线。所有结果为历史因果回放，并非实际推荐发布记录。\n\n",
        report["cutoff_issue"].as_str().unwrap_or("")
    );
    for window in [
        "recent30",
        "recent60",
        "confirmation",
        "tracking",
        "all_oos",
    ] {
        let w = &report["windows"][window];
        if w.is_null() {
            continue;
        }
        text.push_str(&format!("{window}：{}—{}，{}期\n红球数 | 平均中红 | 全6红期数 | 全6红率 | 同大小均匀基准 | 增选1红新增全包期数\n", w["from"].as_str().unwrap_or(""), w["to"].as_str().unwrap_or(""), w["n"]));
        for cell in w["red_curves"].as_array().unwrap().iter().filter(|c| {
            c["model"] == "weighted"
                && (FOCUS_RED_MIN..=FOCUS_RED_MAX)
                    .contains(&(c["red_count"].as_u64().unwrap() as usize))
        }) {
            text.push_str(&format!(
                "{} | {:.3} | {} | {:.2}% | {:.2}% | {}\n",
                cell["red_count"],
                cell["mean_red_hits"].as_f64().unwrap(),
                cell["full_red_periods"],
                cell["full_red_rate"].as_f64().unwrap() * 100.0,
                cell["uniform_full_red_rate"].as_f64().unwrap() * 100.0,
                cell["additional_full_red_periods"]
            ));
        }
        text.push('\n');
    }
    for note in report["notes"].as_array().unwrap() {
        text.push_str(note.as_str().unwrap());
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_conditional_pool_is_bound_to_its_saved_ranking_and_does_not_augment_v4() {
        let draws=crate::draw::synth_draws(600,9);let p=prepare_forecast(&draws).unwrap();
        let old=saved_ranked(&p,draws.last().unwrap().red_mask);
        assert_eq!(reviewed_models(&review(&old,None).unwrap()).len(),4);
        let saved=saved_current(&p,&draws);
        assert_eq!(reviewed_models(&review(&saved,None).unwrap()).len(),5);
        assert_eq!(old["comparison_models"],saved["comparison_models"]);
        let mut bad=saved.clone();bad["coverage_model"]["pools"][0]["blues"]=json!([saved["blue_ranking"][15]]);
        assert!(review(&bad,None).is_err());
        let mut bad=saved;bad["coverage_model"]["pools"][0]["reds"][0]=bad["coverage_model"]["red_ranking"][32].clone();
        assert!(review(&bad,None).is_err());
    }

    #[test]
    fn calibrated_archives_validate_scores_and_keep_v3_models_unchanged() {
        let draws=crate::draw::synth_draws(220,9);
        let prepared=prepare_forecast(&draws).unwrap();
        let original=saved(&prepared);
        let saved=saved_ranked(&prepared,draws.last().unwrap().red_mask);
        assert_eq!(reviewed_models(&review(&original,None).unwrap()).len(),3);
        assert_eq!(reviewed_models(&review(&saved,None).unwrap()).len(),4);
        assert_eq!(original["pools"],saved["pools"]);
        assert_eq!(original["comparison_models"],json!(saved["comparison_models"].as_array().unwrap()[..2]));
        let mut bad=saved.clone();
        bad["comparison_models"][2]["red_scores"][0]=json!(0.9);
        assert!(review(&bad,None).is_err());
        let mut bad=saved.clone();
        bad["comparison_models"][2]["red_ranking"].as_array_mut().unwrap().swap(0,32);
        assert!(review(&bad,None).is_err());
        let mut bad=saved.clone();
        bad["comparison_models"][2]["previous_reds"]=json!([1,1,2,3,4,5]);
        assert!(review(&bad,None).is_err());
        let mut bad=saved.clone();
        bad["comparison_models"][2]["pools"][0]["blues"]=json!([saved["blue_ranking"][15]]);
        assert!(review(&bad,None).is_err());
        let mut bad=saved;
        bad["comparison_models"].as_array_mut().unwrap().pop();
        assert!(review(&bad,None).is_err());
    }

    #[test]
    fn theoretical_baselines_and_pool_scoring_use_draw_recall() {
        assert!((uniform_full_red(8) - 28.0 / 1_107_568.0).abs() < 1e-14);
        assert_eq!(uniform_full_red(33), 1.0);
        let prepared = prepare_forecast(&crate::draw::synth_draws(220, 9)).unwrap();
        let saved = saved(&prepared);
        let p = &saved["pools"][0];
        let reds: Vec<u8> = serde_json::from_value(p["reds"].clone()).unwrap();
        let draw = Draw::from_parts(
            1,
            20260101,
            reds[..6].try_into().unwrap(),
            p["blues"][0].as_u64().unwrap() as u8,
        )
        .unwrap();
        assert_eq!(
            review(&saved, Some(&draw)).unwrap()["pools"][0]["full_draw"],
            true
        );
        assert!(review(&saved, None).unwrap()["pools"][0]["full_draw"].is_null());
        let mut invalid = saved.clone();
        invalid["pools"][0]["reds"][1] = invalid["pools"][0]["reds"][0].clone();
        assert!(review(&invalid, None).is_err());
    }

    #[test]
    fn observation_tiers_are_nested_and_v1_pools_remain_reviewable() {
        let prepared = prepare_forecast(&crate::draw::synth_draws(220, 9)).unwrap();
        let saved = saved(&prepared);
        assert_eq!(saved["version"], 3);
        let pools = saved["pools"].as_array().unwrap();
        assert_eq!(pools.len(), 27);
        for blue_count in BLUE_SIZES {
            let mut previous: Vec<u8> = Vec::new();
            for red_count in TIER_RED_SIZES {
                let p = pools
                    .iter()
                    .find(|p| p["red_count"] == red_count && p["blue_count"] == blue_count)
                    .unwrap();
                let reds: Vec<u8> = serde_json::from_value(p["reds"].clone()).unwrap();
                assert_eq!(p["group"], "observation");
                assert_eq!(&reds[..previous.len()], previous.as_slice());
                previous = reds;
            }
        }
        let pending = review(&saved, None).unwrap();
        assert!(pending["smallest_full_red_tier"].is_null());
        let ranking: Vec<u8> = serde_json::from_value(saved["red_ranking"].clone()).unwrap();
        let draw = Draw::from_parts(
            26001,
            20260101,
            [
                ranking[0],
                ranking[1],
                ranking[2],
                ranking[3],
                ranking[4],
                ranking[16],
            ],
            saved["blue_ranking"][0].as_u64().unwrap() as u8,
        )
        .unwrap();
        let scored = review(&saved, Some(&draw)).unwrap();
        assert_eq!(scored["smallest_full_red_tier"], 18);
        for p in scored["pools"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["group"] == "observation")
        {
            let size = p["red_count"].as_u64().unwrap();
            assert_eq!(p["red_hits"], if size < 18 { 5 } else { 6 });
            assert_eq!(p["full_red"], size >= 18);
        }
        let mut legacy = saved.clone();
        legacy["version"] = json!(1);
        legacy["pools"] = json!(pools
            .iter()
            .filter(|p| p["group"] == "legacy")
            .cloned()
            .collect::<Vec<_>>());
        legacy.as_object_mut().unwrap().remove("observation_tiers");
        let reviewed = review(&legacy, Some(&draw)).unwrap();
        assert_eq!(reviewed["pools"].as_array().unwrap().len(), 9);
        assert!(reviewed["smallest_full_red_tier"].is_null());
        let mut invalid = saved.clone();
        invalid["pools"]
            .as_array_mut()
            .unwrap()
            .push(pools[0].clone());
        assert!(review(&invalid, None).is_err());
    }

    #[test]
    fn comparison_controls_are_fixed_share_blues_and_validate_saved_prefixes() {
        let mut prepared = prepare_forecast(&crate::draw::synth_draws(220, 9)).unwrap();
        let original = saved(&prepared);
        let reviewed = review(&original, None).unwrap();
        let models = reviewed_models(&reviewed);
        assert_eq!(models.len(), 3);
        assert_eq!(original["comparison_policy"]["automatic_promotion"], false);
        for model in &models {
            let pools = model["pools"].as_array().unwrap();
            for b in BLUE_SIZES {
                let mut previous = Vec::<Value>::new();
                for r in TIER_RED_SIZES {
                    let p = pools.iter().find(|p| p["red_count"] == r && p["blue_count"] == b).unwrap();
                    let primary = original["pools"].as_array().unwrap().iter()
                        .find(|p| p["red_count"] == r && p["blue_count"] == b).unwrap();
                    assert_eq!(p["blues"], primary["blues"]);
                    let reds = p["reds"].as_array().unwrap();
                    assert_eq!(&reds[..previous.len()], previous.as_slice());
                    previous = reds.clone();
                    assert!(p["red_hits"].is_null());
                }
            }
            assert!(model["diagnostics"].is_null());
        }
        for expert in &mut prepared.expert_red {
            for (i, score) in expert.iter_mut().enumerate() { *score = (i + 1) as f64; }
        }
        let changed = saved(&prepared);
        assert_eq!(original["comparison_models"][1]["red_ranking"], changed["comparison_models"][1]["red_ranking"]);
        assert_ne!(original["comparison_models"][0]["red_ranking"], changed["comparison_models"][0]["red_ranking"]);
        assert_ne!(uniform_ranking("26119"), uniform_ranking("26120"));
        let mut corrupt = original.clone();
        corrupt["comparison_models"][0]["pools"][0]["blues"] = json!([17]);
        assert!(review(&corrupt, None).is_err());
        corrupt = original.clone();
        let omitted = corrupt["comparison_models"][0]["red_ranking"][32].clone();
        corrupt["comparison_models"][0]["pools"][0]["reds"][0] = omitted;
        assert!(review(&corrupt, None).is_err());
        corrupt = original.clone();
        corrupt["comparison_models"][0]["pools"].as_array_mut().unwrap().pop();
        assert!(review(&corrupt, None).is_err());
        corrupt = original;
        corrupt["comparison_models"][1]["model"] = json!("equal");
        assert!(review(&corrupt, None).is_err());
    }

    #[test]
    fn exact_prefix_diagnostic_is_not_the_recommended_tier_or_a_backfilled_model() {
        let original = saved(&prepare_forecast(&crate::draw::synth_draws(220, 9)).unwrap());
        let order: Vec<u8> = serde_json::from_value(original["red_ranking"].clone()).unwrap();
        let draw = Draw::from_parts(26001, 20260101,
            [order[0], order[1], order[2], order[3], order[4], order[16]],
            original["blue_ranking"][1].as_u64().unwrap() as u8).unwrap();
        let scored = review(&original, Some(&draw)).unwrap();
        assert_eq!(scored["smallest_full_red_tier"], 18);
        assert_eq!(scored["diagnostics"]["minimum_full_red_prefix"], 17);
        assert_eq!(scored["diagnostics"]["winning_blue_rank"], 2);
        assert_eq!(scored["diagnostics"]["is_recommended_pool"], false);
        let p = scored["pools"].as_array().unwrap().iter()
            .find(|p| p["red_count"] == 14 && p["blue_count"] == 2).unwrap();
        assert_eq!(p["missed_red_ranks"], json!([{"number": order[16], "rank": 17}]));
        let mut old = original;
        old["version"] = json!(2);
        old.as_object_mut().unwrap().remove("comparison_models");
        old.as_object_mut().unwrap().remove("comparison_policy");
        let old_bytes = serde_json::to_vec(&old).unwrap();
        let reviewed = review(&old, Some(&draw)).unwrap();
        assert_eq!(reviewed_models(&reviewed).len(), 1);
        assert!(reviewed["comparison_models"].is_null());
        assert_eq!(reviewed["diagnostics"]["minimum_full_red_prefix"], 17);
        assert_eq!(serde_json::to_vec(&old).unwrap(), old_bytes);
    }

    #[test]
    fn full_red_curves_count_cardinality_gains_and_retain_first_plateau_size() {
        let rows: Vec<_> = [14, 16, 18, 24, 33]
            .into_iter()
            .enumerate()
            .map(|(i, last_rank)| Period {
                issue: 26001 + i as u32,
                red_ranks: [[1, 2, 3, 4, 5, last_rank]; 9],
                blue_rank: i + 1,
            })
            .collect();
        let report = summarize(&rows.iter().collect::<Vec<_>>());
        let curves = report["red_curves"].as_array().unwrap();
        assert_eq!(curves.len(), MODELS.len() * 28);
        assert_eq!(report["cells"].as_array().unwrap().len(), 81);
        for model in MODELS {
            let mut previous = 0;
            for (i, c) in curves.iter().filter(|c| c["model"] == model).enumerate() {
                assert_eq!(c["red_count"], 6 + i);
                let full = c["full_red_periods"].as_u64().unwrap();
                assert!(full >= previous);
                if i == 0 {
                    assert!(c["additional_full_red_periods"].is_null());
                } else {
                    assert_eq!(c["additional_full_red_periods"], full - previous);
                }
                for b in 0..3 {
                    assert!(c["full_draw_periods_by_blue"][b].as_u64().unwrap() <= full);
                }
                previous = full;
            }
            let end = curves
                .iter()
                .find(|c| c["model"] == model && c["red_count"] == 33)
                .unwrap();
            assert_eq!(end["full_red_periods"], rows.len());
            assert_eq!(end["full_red_rate"], 1.0);
            assert_eq!(end["mean_missed_reds"], 0.0);
            assert_eq!(end["uniform_full_red_rate"], 1.0);
            assert_eq!(end["full_draw_periods_by_blue"], json!([1, 2, 3]));
            let frontier = report["red_frontiers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["model"] == model)
                .unwrap();
            let sizes: Vec<_> = frontier["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c["red_count"].as_u64().unwrap())
                .collect();
            assert_eq!(sizes, [14, 16, 18, 24]);
        }
        assert!(summarize(&[]).is_null());
    }

    #[test]
    fn adding_future_draws_does_not_change_previous_pool_results() {
        let draws = crate::draw::load_csv(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ssq.csv"),
        )
        .unwrap();
        let early = periods(&draws[..draws.len() - 1]).unwrap();
        let later = periods(&draws).unwrap();
        for (a, b) in early.iter().zip(later.iter()) {
            assert_eq!(a.issue, b.issue);
            assert_eq!(a.red_ranks, b.red_ranks);
            assert_eq!(a.blue_rank, b.blue_rank);
        }
        let short = analyze(&draws[..draws.len() - 1]).unwrap();
        let long = analyze(&draws).unwrap();
        assert_eq!(
            short["windows"]["confirmation"],
            long["windows"]["confirmation"]
        );
        assert_eq!(
            long["next"]["weighted_pools"]["pools"]
                .as_array()
                .unwrap()
                .len(),
            ARCHIVED_RED_SIZES.len() * BLUE_SIZES.len()
        );
        let existing = crate::eval::backtest(&draws).unwrap();
        for k in RED_SIZES {
            let expected = existing
                .topk
                .iter()
                .find(|p| p.window == "确认窗" && p.model == "加权融合" && p.k == k)
                .unwrap();
            let actual = long["windows"]["confirmation"]["cells"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["model"] == "weighted" && c["red_count"] == k && c["blue_count"] == 2)
                .unwrap();
            assert!((expected.mean_hits - actual["mean_red_hits"].as_f64().unwrap()).abs() < 1e-12);
        }
    }
}
