//! Read-only red/blue ranking board and frozen-pool coverage. Does not seal tickets.
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};

use crate::draw::{data_sha256, fmt_date, fmt_issue, Draw};
use crate::eval::prepare_forecast;
use crate::model::roll_all;
use crate::{fusion_model_v3, pool_five};
use crate::protocol::Protocol;
use crate::select::{equal7, fuse6, fuse7, quantize_simplex, rank_desc, score_key};

const CONFIRM_END: u32 = 26115;
const EXPECTED_COVER: f64 = 6.0 * 34.0 / 7.0;
const EXPECTED_TOP16: f64 = 16.0 * 6.0 / 33.0;
const EXPECTED_BLUE: f64 = 8.5;

struct Algo {
    name: &'static str,
    label: &'static str,
    role: &'static str,
    red: Vec<u8>,
    blue: Vec<u8>,
    blue_source: &'static str,
}

struct BlueAlgo {
    name: &'static str,
    label: &'static str,
    role: &'static str,
    order: Vec<u8>,
}

struct Settled {
    issue: u32,
    date: i32,
    red: [u8; 6],
    previous_red: [u8; 6],
    blue: u8,
    algos: Vec<Algo>,
    blues: Vec<BlueAlgo>,
    pool_five: pool_five::Selection,
    pool_five_comparison: pool_five::Comparison,
}

struct Board {
    sha: String,
    settled: Vec<Settled>,
    next_issue: u32,
    next_algos: Vec<Algo>,
    next_blues: Vec<BlueAlgo>,
    next_pool_five: pool_five::Selection,
    cover: crate::cover_rank::Fit,
}

pub fn overview(draws: &[Draw]) -> Result<Value, String> {
    let guard = cached(draws)?;
    Ok(overview_value(&guard.as_ref().ok_or("排序缓存不可用")?.1))
}

pub fn issue(root: &Path, draws: &[Draw], issue: &str) -> Result<Value, String> {
    if issue.len() != 5 || !issue.bytes().all(|c| c.is_ascii_digit()) {
        return Err("期号需要 5 位数字".into());
    }
    let guard = cached(draws)?;
    let board = &guard.as_ref().ok_or("排序缓存不可用")?.1;
    let parsed: u32 = issue.parse().map_err(|_| "期号需要 5 位数字")?;
    if parsed == board.next_issue {
        let mut value = algorithms_value(issue, false, None, &board.next_algos);
        value["pools"] = frozen_pools(root, draws, issue, None)?;
        value["blue_rankings"] = blue_value(&board.next_blues, None);
        value["pool_five"] = board.next_pool_five.view(None);
        value["data_sha256"] = json!(board.sha);
        return Ok(value);
    }
    let row = board
        .settled
        .iter()
        .find(|row| row.issue == parsed)
        .ok_or_else(|| {
            format!("排序页只回放确认窗 {} 及之后已开奖期，以及当前下一期", fmt_issue(Protocol::CONFIRM_START_ISSUE))
        })?;
    let draw = draws.iter().find(|d| d.issue == parsed);
    let mut value = algorithms_value(issue, true, Some((row.date, &row.red, row.blue)), &row.algos);
    value["pools"] = frozen_pools(root, draws, issue, draw)?;
    value["blue_rankings"] = blue_value(&row.blues, Some(row.blue));
    value["pool_five"] = row.pool_five.view(draw);
    value["data_sha256"] = json!(board.sha);
    Ok(value)
}

fn cached(draws: &[Draw]) -> Result<std::sync::MutexGuard<'static, Option<(String, Board)>>, String> {
    static CACHE: OnceLock<Mutex<Option<(String, Board)>>> = OnceLock::new();
    let key = data_sha256(draws);
    let mut guard = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "排序缓存不可用")?;
    if guard.as_ref().is_none_or(|(sha, _)| sha != &key) {
        *guard = Some((key, build(draws)?));
    }
    Ok(guard)
}

fn build(draws: &[Draw]) -> Result<Board, String> {
    let prepared = prepare_forecast(draws)?;
    let snap = roll_all(draws)?;
    let cover = crate::cover_rank::fit_rows(&snap.rows)?;
    let fit = fusion_model_v3::fit(draws, &snap.rows)?;
    let hit_weights = blue_hit_weights(&snap.rows)?;
    let weights = prepared.weights;
    let blue_weights = prepared.blue_weights;
    let index: HashMap<u32, usize> = draws.iter().enumerate().map(|(i, d)| (d.issue, i)).collect();
    let mut settled = Vec::new();
    for row in &snap.rows {
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            continue;
        }
        let pos = *index.get(&row.issue).ok_or("排序期号缺少对应开奖")?;
        if pos == 0 {
            return Err("排序缺少上一期开奖".into());
        }
        let draw = &draws[pos];
        let algos = rank_all(
            &row.expert_p,
            &row.blue_p,
            &weights,
            &blue_weights,
            draws[pos - 1].red_mask,
            &fmt_issue(row.issue),
            &cover.beta,
            &fit.red,
            &fit.blue,
        )?;
        let weighted_red = fuse7(&row.expert_p, &weights);
        let pool_five = pool_five::Selection::new(
            &fusion_model_v3::distribution(&row.expert_p, &row.blue_p, &fit)?,
            &rank_desc(&weighted_red),
        );
        let pool_five_comparison = pool_five.compare(&weighted_red, &fuse6(&row.blue_p, &blue_weights), draw);
        settled.push(Settled {
            issue: row.issue,
            date: draw.date,
            red: draw.red,
            previous_red: draws[pos - 1].red,
            blue: draw.blue,
            algos,
            blues: blue_rankings(&row.blue_p, &blue_weights, &hit_weights, &fit.blue)?,
            pool_five,
            pool_five_comparison,
        });
    }
    let last = draws.last().ok_or("没有开奖数据")?;
    let next_algos = rank_all(
        &snap.next_red,
        &snap.next_blue,
        &weights,
        &blue_weights,
        last.red_mask,
        &fmt_issue(last.issue + 1),
        &cover.beta,
        &fit.red,
        &fit.blue,
    )?;
    Ok(Board {
        sha: data_sha256(draws),
        settled,
        next_issue: last.issue + 1,
        next_algos,
        next_blues: blue_rankings(&snap.next_blue, &blue_weights, &hit_weights, &fit.blue)?,
        next_pool_five: pool_five::Selection::new(
            &fusion_model_v3::distribution(&snap.next_red, &snap.next_blue, &fit)?,
            &prepared.forecast.red_order,
        ),
        cover,
    })
}

fn rank_all(
    expert: &[[f64; 33]; 7],
    blue_p: &[[f64; 16]; 6],
    weights: &[f64; 7],
    blue_weights: &[f64; 6],
    previous: u64,
    issue: &str,
    cover_beta: &[f64; 6],
    red_beta: &[f64; 6],
    blue_beta: &[f64; 5],
) -> Result<Vec<Algo>, String> {
    let weighted = fuse7(expert, weights);
    let equal = fuse7(expert, &equal7());
    let equal_rank_red = equal_midrank(expert);
    let calibrated = crate::ranking::scores(&weighted, previous);
    let shared_blue = order16(&fuse6(blue_p, blue_weights));
    let equal_blue = order16(&fuse6(blue_p, &[1.0 / 6.0; 6]));
    let cover_score = crate::cover_rank::signal(cover_beta, expert)?;
    let fitted_red = crate::red_model_v2::observation_order(red_beta, expert)?;
    let fitted_blue = blue_order(blue_beta, blue_p)?;
    let mut algos = vec![
        algo(crate::fusion_rank_v4::MODEL, crate::fusion_rank_v4::LABEL, "main", &crate::fusion_rank_v4::order(expert, weights)?, &shared_blue, "协议加权蓝球"),
        algo("weighted", "协议加权融合", "main", &rank_desc(&weighted), &shared_blue, "协议加权蓝球"),
        algo("equal", "等权融合", "main", &rank_desc(&equal), &equal_blue, "等权蓝球"),
        algo("equal_rank", "名次等权", "trial", &rank_desc(&equal_rank_red), &equal_blue, "等权蓝球"),
        Algo {
            name: "historical_fit",
            label: "历史拟合",
            role: "trial",
            red: fitted_red,
            blue: fitted_blue,
            blue_source: "历史拟合蓝球",
        },
    ];
    for (i, (name, label)) in Protocol::EXPERTS.iter().zip(Protocol::EXPERTS_ZH).enumerate() {
        algos.push(algo(name, label, "expert", &rank_desc(&expert[i]), &shared_blue, "协议加权蓝球"));
    }
    algos.push(algo(
        crate::ranking::MODEL,
        crate::ranking::LABEL,
        "trial",
        &rank_desc(&calibrated),
        &shared_blue,
        "协议加权蓝球",
    ));
    algos.push(algo(
        "cover_rank",
        "全覆盖排序 · 试验",
        "trial",
        &rank_desc(&cover_score),
        &shared_blue,
        "协议加权蓝球",
    ));
    algos.push(Algo {
        name: "uniform",
        label: "均匀红球对照",
        role: "baseline",
        red: uniform_red(issue),
        blue: shared_blue,
        blue_source: "协议加权蓝球",
    });
    Ok(algos)
}

fn blue_order(beta: &[f64; 5], expert: &[[f64; 16]; 6]) -> Result<Vec<u8>, String> {
    let score = crate::blue_model_v2::distribution(beta, &crate::blue_model_v2::features(expert)?);
    if score.iter().any(|value| !value.is_finite() || *value < 0.) {
        return Err("蓝球v2分布无效".into());
    }
    Ok(rank_desc(&score).iter().map(|index| *index as u8 + 1).collect())
}

fn blue_hit_weights(rows: &[crate::model::Row]) -> Result<[f64; 6], String> {
    let mut hits = [0usize; 6];
    let mut n = 0usize;
    for row in rows {
        if row.issue >= Protocol::CONFIRM_START_ISSUE {
            continue;
        }
        for (expert, scores) in row.blue_p.iter().enumerate() {
            hits[expert] += usize::from(rank_desc(scores)[..2].contains(&(row.blue as usize)));
        }
        n += 1;
    }
    if n < 30 {
        return Err("命中目标蓝球至少需要30条确认窗前记录".into());
    }
    let peak = hits.iter().copied().max().unwrap_or(0) as f64;
    let exp = hits.map(|hit| ((hit as f64 - peak) * 16.0 / n as f64).exp());
    let total = exp.iter().sum::<f64>();
    Ok(quantize_simplex(exp.map(|value| value / total)))
}

fn blue_rankings(
    blue_p: &[[f64; 16]; 6],
    weighted: &[f64; 6],
    hit_weights: &[f64; 6],
    v2_beta: &[f64; 5],
) -> Result<Vec<BlueAlgo>, String> {
    let mut rows = vec![
        BlueAlgo { name: "blue_weighted", label: "协议加权蓝球", role: "main", order: order16(&fuse6(blue_p, weighted)) },
        BlueAlgo { name: "blue_equal", label: "等权蓝球", role: "main", order: order16(&fuse6(blue_p, &[1.0 / 6.0; 6])) },
        BlueAlgo { name: "blue_v2", label: "历史拟合蓝球", role: "trial", order: blue_order(v2_beta, blue_p)? },
        BlueAlgo { name: "blue_hit", label: "命中目标蓝球", role: "trial", order: order16(&fuse6(blue_p, hit_weights)) },
    ];
    for (index, label) in Protocol::BLUES_ZH.iter().enumerate() {
        rows.push(BlueAlgo {
            name: Protocol::BLUES[index],
            label,
            role: "expert",
            order: order16(&blue_p[index]),
        });
    }
    Ok(rows)
}

fn blue_value(rows: &[BlueAlgo], winner: Option<u8>) -> Value {
    let algorithms: Vec<_> = rows
        .iter()
        .map(|row| {
            let (rank, top2, top4) = if let Some(winner) = winner {
                let rank = blue_rank(&row.order, winner);
                (json!(rank), json!(rank <= 2), json!(rank <= 4))
            } else {
                (Value::Null, Value::Null, Value::Null)
            };
            json!({"name":row.name,"label":row.label,"role":row.role,"blue_ranking":row.order,
                "blue_rank":rank,"top2_hit":top2,"top4_hit":top4})
        })
        .collect();
    json!(algorithms)
}

fn algo(
    name: &'static str,
    label: &'static str,
    role: &'static str,
    order: &[usize; 33],
    blue: &[u8],
    blue_source: &'static str,
) -> Algo {
    Algo {
        name,
        label,
        role,
        red: order.iter().map(|n| *n as u8 + 1).collect(),
        blue: blue.to_vec(),
        blue_source,
    }
}

fn uniform_red(issue: &str) -> Vec<u8> {
    let hash = crate::hash::sha256_hex(
        format!("ssq-pool-comparison-v1:{}:{issue}:red", Protocol::SEED).as_bytes(),
    );
    let mut rng = crate::stats::Rng::new(u64::from_str_radix(&hash[..16], 16).expect("sha256 prefix"));
    let mut numbers: Vec<u8> = (1..=33).collect();
    for i in (1..numbers.len()).rev() {
        numbers.swap(i, rng.gen_range(i + 1));
    }
    numbers
}

fn order16(score: &[f64; 16]) -> Vec<u8> {
    rank_desc(score).iter().map(|n| *n as u8 + 1).collect()
}

/// 并列中位名次转成高分优先。分数差距不再影响名次，只保留专家内部的先后。
fn midrank_better<const N: usize>(scores: &[f64; N]) -> [f64; N] {
    let order = rank_desc(scores);
    let mut out = [0.0; N];
    let mut start = 0;
    while start < N {
        let mut end = start + 1;
        while end < N && score_key(scores[order[end]]) == score_key(scores[order[start]]) {
            end += 1;
        }
        let better = (N + 1) as f64 - (start + 1 + end) as f64 / 2.0;
        for &index in &order[start..end] {
            out[index] = better;
        }
        start = end;
    }
    out
}

fn equal_midrank<const N: usize, const K: usize>(rows: &[[f64; N]; K]) -> [f64; N] {
    let mut score = [0.0; N];
    for row in rows {
        let mid = midrank_better(row);
        for i in 0..N {
            score[i] += mid[i];
        }
    }
    score
}

fn previous_in_top6(order: &[u8], previous: &[u8; 6]) -> usize {
    order.iter().take(6).filter(|n| previous.contains(*n)).count()
}

fn cover(order: &[u8], reds: &[u8; 6]) -> (usize, usize) {
    let mut worst = 0usize;
    let mut top16 = 0usize;
    for n in reds {
        let rank = order.iter().position(|v| v == n).expect("排名包含全部红球") + 1;
        worst = worst.max(rank);
        if rank <= 16 {
            top16 += 1;
        }
    }
    (worst, top16)
}

fn blue_rank(order: &[u8], blue: u8) -> usize {
    order.iter().position(|v| *v == blue).expect("排名包含全部蓝球") + 1
}

fn overview_value(board: &Board) -> Value {
    let window = |start: u32, end: u32| {
        let rows: Vec<_> = board
            .settled
            .iter()
            .filter(|row| row.issue >= start && row.issue <= end)
            .collect();
        if rows.is_empty() {
            return json!({"n":0,"algorithms":[]});
        }
        let algorithms: Vec<_> = (0..rows[0].algos.len())
            .map(|i| {
                let sample = &rows[0].algos[i];
                let mut cover_sum = 0.0;
                let mut top_sum = 0.0;
                let mut blue_sum = 0.0;
                let mut full16 = 0usize;
                for row in &rows {
                    let (need, hits) = cover(&row.algos[i].red, &row.red);
                    cover_sum += need as f64;
                    top_sum += hits as f64;
                    blue_sum += blue_rank(&row.algos[i].blue, row.blue) as f64;
                    full16 += usize::from(need <= 16);
                }
                let n = rows.len() as f64;
                json!({"name":sample.name,"label":sample.label,"role":sample.role,"n":rows.len(),
                    "mean_cover_rank":cover_sum/n,"mean_top16_hits":top_sum/n,"mean_blue_rank":blue_sum/n,
                    "top16_covers_all_six":full16})
            })
            .collect();
        json!({"n":rows.len(),"algorithms":algorithms})
    };
    let confirm: Vec<_> = board
        .settled
        .iter()
        .filter(|row| row.issue >= Protocol::CONFIRM_START_ISSUE && row.issue <= CONFIRM_END)
        .collect();
    let mid = confirm.len() / 2;
    let cover_mean = mean_cover(&confirm, "cover_rank");
    let weighted_mean = mean_cover(&confirm, "weighted");
    let gain_weighted = cover_mean.zip(weighted_mean).map(|(cover, weighted)| weighted - cover);
    let gain_uniform = cover_mean.map(|cover| EXPECTED_COVER - cover);
    let clears = gain_weighted.is_some_and(|gain| gain >= crate::cover_rank::REQUIRED_MEAN_GAIN)
        && gain_uniform.is_some_and(|gain| gain >= crate::cover_rank::REQUIRED_MEAN_GAIN);
    let cover_rank = json!({
        "name": "cover_rank",
        "label": "全覆盖排序 · 试验",
        "policy": crate::cover_rank::policy(),
        "training_rows": board.cover.n,
        "iterations": board.cover.iterations,
        "gradient": board.cover.gradient,
        "objective": board.cover.objective,
        "coefficients": {
            "frequency": board.cover.beta[0],
            "ewma": board.cover.beta[1],
            "log_gap": board.cover.beta[2],
            "cooccur": board.cover.beta[3],
            "zone": board.cover.beta[4],
            "repeat": board.cover.beta[5]
        },
        "confirmation_n": confirm.len(),
        "confirmation_mean_cover_rank": num(cover_mean),
        "weighted_confirmation_mean_cover_rank": num(weighted_mean),
        "gain_vs_weighted": num(gain_weighted),
        "gain_vs_uniform": num(gain_uniform),
        "first_half_gain_vs_weighted": num(mean_cover(&confirm[..mid], "weighted").zip(mean_cover(&confirm[..mid], "cover_rank")).map(|(weighted, cover)| weighted - cover)),
        "second_half_gain_vs_weighted": num(mean_cover(&confirm[mid..], "weighted").zip(mean_cover(&confirm[mid..], "cover_rank")).map(|(weighted, cover)| weighted - cover)),
        "required_mean_gain": crate::cover_rank::REQUIRED_MEAN_GAIN,
        "clears_predeclared_mean_gate": clears,
        "replaces_weighted": false,
        "rewrites_archives": false,
        "automatic_promotion": false,
        "note": cover_note(confirm.len(), cover_mean, weighted_mean, gain_weighted, gain_uniform, clears)
    });
    let fitted = fitted_means(&confirm);
    let historical_fit = json!({
        "name": "historical_fit",
        "label": "历史拟合",
        "red_model": "red_score_v2",
        "blue_model": "blue_score_v2",
        "training_last_issue": fmt_issue(crate::red_model_v2::TRAIN_END),
        "writes_archives": false,
        "actual_recommendation": false,
        "automatic_promotion": false,
        "confirmation_mean_cover_rank": num(fitted.map(|row| row.0)),
        "confirmation_mean_top16_hits": num(fitted.map(|row| row.1)),
        "confirmation_mean_blue_rank": num(fitted.map(|row| row.2)),
        "note": historical_note(confirm.len(), fitted)
    });
    let equal_rank = equal_rank_report(&confirm, board.settled.last());
    let blue_confirmation = blue_study(&confirm);
    let blue_tracking = blue_study(&board.settled.iter().filter(|row| row.issue > CONFIRM_END).collect::<Vec<_>>());
    let blue_all = blue_study(&board.settled.iter().collect::<Vec<_>>());
    let blue_study = json!({
        "writes_archives": false,
        "actual_recommendation": false,
        "automatic_promotion": false,
        "uniform_expected_rank": EXPECTED_BLUE,
        "uniform_top2_rate": 2.0 / 16.0,
        "uniform_top4_rate": 4.0 / 16.0,
        "confirmation": blue_confirmation,
        "tracking": blue_tracking,
        "all_from_25111": blue_all,
        "note": blue_note(&blue_confirmation)
    });
    let mut issues: Vec<_> = board
        .settled
        .iter()
        .rev()
        .map(|row| json!({"issue":fmt_issue(row.issue),"date":fmt_date(row.date),"settled":true}))
        .collect();
    issues.insert(0, json!({"issue":fmt_issue(board.next_issue),"settled":false}));
    json!({
        "kind":"causal_rank_and_frozen_pool_board",
        "writes_archives":false,
        "actual_recommendation":false,
        "data_sha256":board.sha,
        "weight_freeze_before_issue":fmt_issue(Protocol::CONFIRM_START_ISSUE),
        "uniform_expected_cover_rank":EXPECTED_COVER,
        "uniform_expected_top16_hits":EXPECTED_TOP16,
        "uniform_expected_blue_rank":EXPECTED_BLUE,
        "windows":{
            "confirmation_25111_26115":window(Protocol::CONFIRM_START_ISSUE, CONFIRM_END),
            "tracking_after_26115":window(CONFIRM_END+1, u32::MAX),
            "all_from_25111":window(Protocol::CONFIRM_START_ISSUE, u32::MAX)
        },
        "cover_rank":cover_rank,
        "historical_fit":historical_fit,
        "equal_rank":equal_rank,
        "blue_study":blue_study,
        "pool_five": {
            "confirmation_25111_26115": pool_five::summary(confirm.iter().map(|row| row.pool_five_comparison)),
            "tracking_after_26115": pool_five::summary(board.settled.iter().filter(|row| row.issue > CONFIRM_END).map(|row| row.pool_five_comparison)),
            "all_from_25111": pool_five::summary(board.settled.iter().map(|row| row.pool_five_comparison)),
        },
        "issues":issues,
        "next_issue":fmt_issue(board.next_issue),
        "note":"名次只用目标期以前的专家特征和25111前冻结的融合权重。盖住6个红球的名次是这6个号码里最靠后的名次；均匀随机顺序的期望是6×34/7。前16名期望命中16×6/33个，蓝球期望名次是8.5。这些对照不是中奖概率，也不替换已封存号码池。"
    })
}

fn blue_study(rows: &[&Settled]) -> Value {
    if rows.is_empty() {
        return json!({"n": 0, "algorithms": []});
    }
    let algorithms: Vec<_> = (0..rows[0].blues.len())
        .map(|index| {
            let sample = &rows[0].blues[index];
            let mut rank_sum = 0.0;
            let mut top2 = 0usize;
            let mut top4 = 0usize;
            for row in rows {
                let rank = blue_rank(&row.blues[index].order, row.blue);
                rank_sum += rank as f64;
                top2 += usize::from(rank <= 2);
                top4 += usize::from(rank <= 4);
            }
            let n = rows.len();
            json!({"name":sample.name,"label":sample.label,"role":sample.role,"n":n,
                "mean_rank":rank_sum / n as f64,"top2":top2,"top4":top4})
        })
        .collect();
    json!({"n": rows.len(), "algorithms": algorithms})
}

fn blue_field(summary: &Value, name: &str, field: &str) -> Option<f64> {
    summary["algorithms"]
        .as_array()?
        .iter()
        .find(|row| row["name"] == name)?
        .get(field)?
        .as_f64()
}

fn blue_note(summary: &Value) -> String {
    let Some(n) = summary["n"].as_u64() else {
        return "蓝球排序缺少确认窗样本。".into();
    };
    match (
        blue_field(summary, "blue_weighted", "mean_rank"),
        blue_field(summary, "blue_weighted", "top2"),
        blue_field(summary, "blue_v2", "mean_rank"),
        blue_field(summary, "blue_v2", "top2"),
        blue_field(summary, "blue_hit", "mean_rank"),
        blue_field(summary, "blue_hit", "top2"),
    ) {
        (Some(weighted), Some(weighted_top2), Some(fitted), Some(fitted_top2), Some(hit), Some(hit_top2)) => format!(
            "蓝球按各自冻结算法分开排列。确认窗{n}期，协议加权平均第{weighted:.2}名、前2命中{weighted_top2:.0}期，历史拟合平均第{fitted:.2}名、前2命中{fitted_top2:.0}期，命中目标平均第{hit:.2}名、前2命中{hit_top2:.0}期。均匀期望是第8.50名、前2命中{:.1}期。平均名次仍靠近16个蓝球的中位，这些对照用来看选号位置，不替换已封存蓝球。",
            n as f64 * 2.0 / 16.0
        ),
        _ => "蓝球排序缺少确认窗样本。".into(),
    }
}

fn fitted_means(rows: &[&Settled]) -> Option<(f64, f64, f64)> {
    if rows.is_empty() {
        return None;
    }
    let mut cover_sum = 0.0;
    let mut top_sum = 0.0;
    let mut blue_sum = 0.0;
    for row in rows {
        let algo = row.algos.iter().find(|algo| algo.name == "historical_fit")?;
        let (need, hits) = cover(&algo.red, &row.red);
        cover_sum += need as f64;
        top_sum += hits as f64;
        blue_sum += blue_rank(&algo.blue, row.blue) as f64;
    }
    let n = rows.len() as f64;
    Some((cover_sum / n, top_sum / n, blue_sum / n))
}

fn equal_rank_report(confirm: &[&Settled], latest: Option<&Settled>) -> Value {
    let equal = rank_means(confirm, "equal");
    let ordinal = rank_means(confirm, "equal_rank");
    json!({
        "name": "equal_rank",
        "label": "名次等权",
        "method": "tie_aware_midrank_then_equal_mean",
        "writes_archives": false,
        "actual_recommendation": false,
        "automatic_promotion": false,
        "replaces_equal": false,
        "confirmation_n": confirm.len(),
        "confirmation_mean_cover_rank": num(ordinal.map(|row| row.0)),
        "equal_confirmation_mean_cover_rank": num(equal.map(|row| row.0)),
        "confirmation_mean_previous_in_top6": num(ordinal.map(|row| row.1)),
        "equal_confirmation_mean_previous_in_top6": num(equal.map(|row| row.1)),
        "blue_scope": "equal_blue",
        "note": equal_rank_note(confirm.len(), equal, ordinal, latest)
    })
}

fn rank_means(rows: &[&Settled], name: &str) -> Option<(f64, f64, f64)> {
    if rows.is_empty() {
        return None;
    }
    let mut cover_sum = 0.0;
    let mut head_sum = 0.0;
    let mut blue_sum = 0.0;
    for row in rows {
        let algo = row.algos.iter().find(|algo| algo.name == name)?;
        cover_sum += cover(&algo.red, &row.red).0 as f64;
        head_sum += previous_in_top6(&algo.red, &row.previous_red) as f64;
        blue_sum += blue_rank(&algo.blue, row.blue) as f64;
    }
    let n = rows.len() as f64;
    Some((cover_sum / n, head_sum / n, blue_sum / n))
}

fn equal_rank_note(
    n: usize,
    equal: Option<(f64, f64, f64)>,
    ordinal: Option<(f64, f64, f64)>,
    latest: Option<&Settled>,
) -> String {
    let (Some((equal_cover, equal_head, _)), Some((rank_cover, rank_head, _))) = (equal, ordinal) else {
        return "名次等权缺少确认窗样本。".into();
    };
    let mut note = format!(
        "等权融合把七个专家的概率各乘1/7。上期重号的分数跨度大，前6名经常被上期红球占住。名次等权先把每个专家收成并列中位名次再平均，七个专家对名次的发言权相同。确认窗{n}期，等权融合前6名平均有{equal_head:.2}个上期红球，名次等权是{rank_head:.2}个；盖住6个开奖红球的平均名次分别是{equal_cover:.2}和{rank_cover:.2}，均匀期望是{EXPECTED_COVER:.2}。蓝球仍用等权蓝球。"
    );
    if let Some(row) = latest {
        if let (Some(equal_algo), Some(rank_algo)) = (
            row.algos.iter().find(|algo| algo.name == "equal"),
            row.algos.iter().find(|algo| algo.name == "equal_rank"),
        ) {
            note.push_str(&format!(
                "最近一期{}，等权前6名含{}个上期红球、全覆盖第{}名，名次等权前6名含{}个上期红球、全覆盖第{}名。",
                fmt_issue(row.issue),
                previous_in_top6(&equal_algo.red, &row.previous_red),
                cover(&equal_algo.red, &row.red).0,
                previous_in_top6(&rank_algo.red, &row.previous_red),
                cover(&rank_algo.red, &row.red).0
            ));
        }
    }
    note.push_str("名次等权不替换等权融合，也不改已封存号码池。");
    note
}

fn historical_note(n: usize, means: Option<(f64, f64, f64)>) -> String {
    match means {
        Some((cover, top, blue)) => format!(
            "历史拟合在本期排序里给出全部33个红球和16个蓝球。红球按23100前的完整六红似然，蓝球按同期冻结的蓝球似然，从高到低排列。确认窗{n}期平均到第{cover:.2}名盖住6个红球，前16名平均命中{top:.2}个，蓝球平均第{blue:.2}名。均匀期望是29.14、2.91和8.50。推荐排序仍是协议加权融合，冻结号码池保持原样。"
        ),
        None => "历史拟合缺少确认窗样本。".into(),
    }
}

fn mean_cover(rows: &[&Settled], name: &str) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    for row in rows {
        let algo = row.algos.iter().find(|algo| algo.name == name)?;
        sum += cover(&algo.red, &row.red).0 as f64;
    }
    Some(sum / rows.len() as f64)
}

fn num(value: Option<f64>) -> Value {
    value.map_or(Value::Null, |value| json!(value))
}

fn cover_note(
    n: usize,
    cover_mean: Option<f64>,
    weighted_mean: Option<f64>,
    gain_weighted: Option<f64>,
    gain_uniform: Option<f64>,
    clears: bool,
) -> String {
    match (cover_mean, weighted_mean, gain_weighted, gain_uniform) {
        (Some(cover), Some(weighted), Some(_gain_weighted), Some(_gain_uniform)) if clears => format!(
            "确认窗{n}期里，全覆盖排序平均第{cover:.2}名盖住6个红球，协议加权融合是第{weighted:.2}名，均匀期望是{EXPECTED_COVER:.2}。相对两者都至少少1个名次。这条排序只供对照，已封存方案保持原样。"
        ),
        (Some(cover), Some(weighted), Some(gain_weighted), Some(gain_uniform)) => format!(
            "确认窗{n}期里，全覆盖排序平均第{cover:.2}名盖住6个红球，协议加权融合是第{weighted:.2}名，均匀期望是{EXPECTED_COVER:.2}。正数表示全覆盖更靠前：相对融合{gain_weighted:.2}，相对均匀期望{gain_uniform:.2}。预先要求这两项都至少为1。推荐排序仍是协议加权融合，冻结号码池保持原样。"
        ),
        _ => "全覆盖排序缺少确认窗样本。".into(),
    }
}

fn algorithms_value(issue: &str, settled: bool, outcome: Option<(i32, &[u8; 6], u8)>, algos: &[Algo]) -> Value {
    let algorithms: Vec<_> = algos
        .iter()
        .map(|algo| {
            let (cover_rank, top16, blue) = if let Some((_, reds, blue)) = outcome {
                let (need, hits) = cover(&algo.red, reds);
                (json!(need), json!(hits), json!(blue_rank(&algo.blue, blue)))
            } else {
                (Value::Null, Value::Null, Value::Null)
            };
            json!({"name":algo.name,"label":algo.label,"role":algo.role,
                "red_ranking":algo.red,"blue_ranking":algo.blue,"blue_source":algo.blue_source,
                "cover_all_six_rank":cover_rank,"top16_hits":top16,"blue_rank":blue})
        })
        .collect();
    let mut value = json!({
        "kind":"causal_rank_and_frozen_pool_issue",
        "writes_archives":false,
        "actual_recommendation":false,
        "issue":issue,
        "settled":settled,
        "uniform_expected_cover_rank":EXPECTED_COVER,
        "uniform_expected_top16_hits":EXPECTED_TOP16,
        "uniform_expected_blue_rank":EXPECTED_BLUE,
        "algorithms":algorithms,
        "note":"排序在开奖前确定。号码池只读取当时已经冻结的每日档案，缺档不补造，研究排序不写进原档。"
    });
    if let Some((date, reds, blue)) = outcome {
        value["date"] = json!(fmt_date(date));
        value["draw"] = json!({"red":reds,"blue":blue});
    }
    value
}

fn frozen_pools(root: &Path, draws: &[Draw], issue: &str, draw: Option<&Draw>) -> Result<Value, String> {
    let path = root.join("daily").join(format!("{issue}.json"));
    if !path.is_file() {
        return Ok(json!({"available":false,"items":[],"note":"本期没有每日冻结档案，页面不补造号码池。"}));
    }
    let doc = crate::rank_pool_audit::daily(root, draws, issue)?.ok_or("每日原档缺失")?;
    let payload = &doc["payload"];
    let mut items = Vec::new();
    if let Some(pools) = payload.get("number_pools") {
        collect_models(pools, "tier", &mut items, draw)?;
    }
    if let Some(models) = payload["small_red_pools"]["models"].as_array() {
        for model in models {
            collect_model_pools(model, "small", &mut items, draw)?;
        }
    }
    if let Some(strategies) = payload["strategies"].as_object() {
        for (key, strategy) in strategies {
            let Some(tickets) = strategy["tickets"].as_array() else { continue };
            let mut reds = BTreeSet::new();
            let mut blues = BTreeSet::new();
            for ticket in tickets {
                for n in ticket["red"].as_array().ok_or("方案红球格式无效")? {
                    reds.insert(ball(n, 33, "方案红球")?);
                }
                blues.insert(ball(&ticket["blue"], 16, "方案蓝球")?);
            }
            if reds.is_empty() {
                continue;
            }
            items.push(pool_item(
                "strategy",
                key,
                strategy["title"].as_str().unwrap_or(key),
                reds.into_iter().collect(),
                blues.into_iter().collect(),
                draw,
            ));
            items.last_mut().unwrap()["ticket_audit"] = crate::rank_pool_audit::strategy(strategy, draw)?;
        }
    }
    let source = crate::rank_pool_audit::source(&doc);
    let funnel = crate::rank_pool_audit::funnel(&items, draw.is_some());
    Ok(json!({"available":true,"source":source,"strategy_funnel":funnel,"items":items,"note":"号码池来自该期首次冻结的每日档案。分档池保留当时的2个蓝球；方案行是该方案全部单注的号码并集。"}))
}

fn collect_models(root: &Value, kind: &str, items: &mut Vec<Value>, draw: Option<&Draw>) -> Result<(), String> {
    collect_model_pools(root, kind, items, draw)?;
    for key in ["comparison_models", "algorithm_models"] {
        if let Some(rows) = root[key].as_array() {
            for model in rows {
                collect_model_pools(model, kind, items, draw)?;
            }
        }
    }
    if root["coverage_model"]["model"].is_string() {
        collect_model_pools(&root["coverage_model"], kind, items, draw)?;
    }
    Ok(())
}

fn collect_model_pools(model: &Value, kind: &str, items: &mut Vec<Value>, draw: Option<&Draw>) -> Result<(), String> {
    let Some(pools) = model["pools"].as_array() else { return Ok(()) };
    for pool in pools {
        let name = pool["model"].as_str().or_else(|| model["model"].as_str()).unwrap_or(kind);
        if kind == "tier" {
            let red_count = pool["red_count"].as_u64().unwrap_or(0) as usize;
            let blue_count = pool["blue_count"].as_u64().unwrap_or(0) as usize;
            if !(14..=24).contains(&red_count) || blue_count != 2 || pool["group"] != "observation" {
                continue;
            }
        }
        let reds = number_list(&pool["reds"], 33, "号码池红球")?;
        let blues = if pool["blues"].is_array() {
            number_list(&pool["blues"], 16, "号码池蓝球")?
        } else {
            Vec::new()
        };
        let label = match model["label"].as_str() {
            Some(text) => format!("{text} · {}红", reds.len()),
            None => format!("{} · {}红", model_label(name), reds.len()),
        };
        items.push(pool_item(kind, name, &label, reds, blues, draw));
    }
    Ok(())
}

fn number_list(value: &Value, limit: u8, what: &str) -> Result<Vec<u8>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what}格式无效"))?
        .iter()
        .map(|n| ball(n, limit, what))
        .collect()
}

fn ball(value: &Value, limit: u8, what: &str) -> Result<u8, String> {
    let n = value.as_u64().ok_or_else(|| format!("{what}格式无效"))?;
    if (1..=u64::from(limit)).contains(&n) {
        Ok(n as u8)
    } else {
        Err(format!("{what}超出范围"))
    }
}

fn model_label(name: &str) -> String {
    match name {
        "weighted" => "协议加权融合",
        "equal" => "等权融合",
        "uniform" => "均匀红球对照",
        "repeat_calibrated" => crate::ranking::LABEL,
        "conditional_gap" => "条件频率",
        "small_recent" => "小池近况融合",
        "rank_consensus" => "名次共识",
        "scenario_cover" => "全6情景覆盖",
        "robust_cover" => "三窗稳健覆盖",
        "dual_support" => "双专家支持",
        other => other,
    }
    .to_string()
}

fn pool_item(kind: &str, model: &str, label: &str, mut reds: Vec<u8>, mut blues: Vec<u8>, draw: Option<&Draw>) -> Value {
    reds.sort_unstable();
    reds.dedup();
    blues.sort_unstable();
    blues.dedup();
    let (red_hits, blue_hit, covers) = if let Some(draw) = draw {
        let hits = draw.red.iter().filter(|n| reds.contains(n)).count();
        (
            json!(hits),
            if blues.is_empty() { Value::Null } else { json!(blues.contains(&draw.blue)) },
            json!(hits == 6),
        )
    } else {
        (Value::Null, Value::Null, Value::Null)
    };
    json!({"kind":kind,"model":model,"label":label,"red_count":reds.len(),"blue_count":blues.len(),
        "reds":reds,"blues":blues,"red_hits":red_hits,"blue_hit":blue_hit,"covers_all_red":covers})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn equal_probability_follows_a_sharp_expert_and_midrank_follows_the_majority() {
        let mut expert = [[1.0; 33]; 7];
        for ball in 0..6 {
            expert[0][ball] = 1000.0;
        }
        for row in expert.iter_mut().skip(1) {
            row[10] = 1.01;
        }
        let probability = rank_desc(&fuse7(&expert, &equal7()));
        let ordinal = rank_desc(&equal_midrank(&expert));
        assert!(probability[0] < 6);
        assert_eq!(ordinal[0], 10);
    }

    #[test]
    fn cover_uses_the_worst_winning_rank() {
        let order: Vec<u8> = (1..=33).collect();
        assert_eq!(cover(&order, &[1, 2, 3, 4, 5, 33]), (33, 5));
        assert_eq!(blue_rank(&(1..=16).collect::<Vec<_>>(), 16), 16);
        assert!((EXPECTED_COVER - 29.142857).abs() < 1e-6);
    }

    #[test]
    fn board_is_read_only_and_future_draw_does_not_change_past_ranks() {
        let draws = crate::draw::load_csv(Path::new("data/ssq.csv")).unwrap();
        let dir = std::env::temp_dir().join(format!("ssq-rank-board-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let before = fs::read_dir(&dir).unwrap().count();
        let report = overview(&draws).unwrap();
        let latest = draws.last().unwrap();
        let prior = fmt_issue(draws[draws.len() - 2].issue);
        let view = issue(&dir, &draws, &fmt_issue(latest.issue)).unwrap();
        let pending = issue(&dir, &draws, &fmt_issue(latest.issue + 1)).unwrap();
        assert_eq!(before, fs::read_dir(&dir).unwrap().count());
        assert_eq!(report["writes_archives"], false);
        assert_eq!(view["algorithms"].as_array().unwrap().len(), 15);
        let trial = view["algorithms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["name"] == "cover_rank")
            .unwrap();
        assert_eq!(trial["role"], "trial");
        assert_eq!(trial["red_ranking"].as_array().unwrap().len(), 33);
        let adoption = &report["cover_rank"];
        assert_eq!(adoption["rewrites_archives"], false);
        assert_eq!(adoption["replaces_weighted"], false);
        assert_eq!(adoption["automatic_promotion"], false);
        let gain_w = adoption["gain_vs_weighted"].as_f64().unwrap();
        let gain_u = adoption["gain_vs_uniform"].as_f64().unwrap();
        assert_eq!(
            adoption["clears_predeclared_mean_gate"],
            json!(gain_w >= crate::cover_rank::REQUIRED_MEAN_GAIN && gain_u >= crate::cover_rank::REQUIRED_MEAN_GAIN)
        );
        assert_eq!(view["algorithms"][0]["red_ranking"].as_array().unwrap().len(), 33);
        assert!(view["algorithms"][0]["cover_all_six_rank"].as_u64().unwrap() >= 6);
        assert_eq!(pending["settled"], false);
        assert!(pending["algorithms"][0]["cover_all_six_rank"].is_null());
        assert_eq!(pending["pools"]["available"], false);
        let fitted = view["algorithms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["name"] == "historical_fit")
            .unwrap();
        assert_eq!(fitted["role"], "trial");
        assert_eq!(fitted["red_ranking"].as_array().unwrap().len(), 33);
        assert_eq!(fitted["blue_ranking"].as_array().unwrap().len(), 16);
        assert_eq!(fitted["blue_source"], "历史拟合蓝球");
        assert_eq!(report["historical_fit"]["writes_archives"], false);
        assert_eq!(report["historical_fit"]["automatic_promotion"], false);
        assert!(report["historical_fit"]["confirmation_mean_cover_rank"].as_f64().unwrap() >= 6.0);
        let ordinal = view["algorithms"].as_array().unwrap().iter().find(|row| row["name"] == "equal_rank").unwrap();
        assert_eq!(ordinal["role"], "trial");
        assert_eq!(ordinal["red_ranking"].as_array().unwrap().len(), 33);
        assert_eq!(ordinal["blue_ranking"].as_array().unwrap().len(), 16);
        assert_eq!(ordinal["blue_source"], "等权蓝球");
        assert_eq!(ordinal["blue_ranking"], view["algorithms"].as_array().unwrap().iter().find(|row| row["name"] == "equal").unwrap()["blue_ranking"]);
        assert_eq!(report["equal_rank"]["replaces_equal"], false);
        assert_eq!(report["equal_rank"]["automatic_promotion"], false);
        assert_eq!(report["equal_rank"]["confirmation_n"], 156);
        assert!(report["equal_rank"]["note"].as_str().unwrap().contains("名次等权"));
        let blues = view["blue_rankings"].as_array().unwrap();
        assert_eq!(blues.len(), 10);
        assert_eq!(blues[0]["name"], "blue_weighted");
        assert_eq!(blues[2]["name"], "blue_v2");
        let mut numbers: Vec<_> = blues[0]["blue_ranking"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        numbers.sort_unstable();
        assert_eq!(numbers, (1..=16).collect::<Vec<_>>());
        assert!(blues[0]["blue_rank"].as_u64().unwrap() <= 16);
        assert!(pending["blue_rankings"][0]["blue_rank"].is_null());
        assert_eq!(report["blue_study"]["writes_archives"], false);
        assert_eq!(report["blue_study"]["automatic_promotion"], false);
        assert_eq!(report["blue_study"]["confirmation"]["n"], 156);
        let mut earlier = draws.clone();
        let newest = earlier.pop().unwrap();
        let replay = issue(&dir, &earlier, &fmt_issue(newest.issue)).unwrap();
        let kept = issue(&dir, &earlier, &prior).unwrap();
        let full_prior = issue(&dir, &draws, &prior).unwrap();
        let (plan, prepared) = pool_five::build(&earlier, prepare_forecast(&earlier).unwrap()).unwrap();
        let expected: Vec<_> = plan.tickets.iter().map(|t| json!({"red":t.red,"blue":t.blue})).collect();
        assert_eq!(plan.cost, 10);
        assert_eq!(plan.reds.iter().collect::<BTreeSet<_>>().len(), 30);
        assert_eq!(plan.blues.iter().collect::<BTreeSet<_>>().len(), 5);
        assert_eq!(replay["pool_five"]["tickets"], json!(expected));
        assert_eq!(replay["pool_five"]["tickets"], view["pool_five"]["tickets"]);
        assert_eq!(replay["pool_five"]["red_ranking"], prepared.ranking.unwrap()["red_ranking"]);
        assert_eq!(kept["pool_five"], full_prior["pool_five"]);
        for (left, right) in replay["algorithms"].as_array().unwrap().iter().zip(view["algorithms"].as_array().unwrap()) {
            assert_eq!(left["red_ranking"], right["red_ranking"]);
            assert_eq!(left["blue_ranking"], right["blue_ranking"]);
            assert!(left["cover_all_six_rank"].is_null());
        }
        assert_eq!(kept["algorithms"], full_prior["algorithms"]);
        assert_eq!(kept["blue_rankings"], full_prior["blue_rankings"]);
        for (left, right) in replay["blue_rankings"].as_array().unwrap().iter().zip(view["blue_rankings"].as_array().unwrap()) {
            assert_eq!(left["blue_ranking"], right["blue_ranking"]);
            assert!(left["blue_rank"].is_null());
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn frozen_daily_pools_are_counted_without_rewriting_the_archive() {
        let dir = std::env::temp_dir().join(format!("ssq-rank-pools-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let draws = crate::rank_pool_audit::fixture(&dir);
        let path = dir.join("daily/26131.json");
        let bytes = fs::read(&path).unwrap();
        let view = issue(&dir, &draws, "26131").unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        let items = view["pools"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["red_hits"], 6);
        assert_eq!(items[0]["blue_hit"], true);
        assert_eq!(items[1]["kind"], "strategy");
        assert_eq!(items[1]["ticket_audit"]["single_jackpot"], true);
        assert_eq!(view["pools"]["source"]["issue"], "26131");
        let small = json!({"model":"small_recent","label":"小池近况融合 · 试验","pools":[{"model":"small_recent","reds":[1,2,3,4,5,6]}]});
        let mut mapped = Vec::new();
        collect_model_pools(&small, "small", &mut mapped, draws.last()).unwrap();
        assert_eq!(mapped[0]["label"], "小池近况融合 · 试验 · 6红");
        assert_eq!(mapped[0]["covers_all_red"], true);
        let _ = fs::remove_dir_all(&dir);
    }
}
