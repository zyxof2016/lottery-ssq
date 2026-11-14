//! Read-only red/blue ranking board and frozen-pool coverage. Does not seal tickets.
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::{json, Value};

use crate::draw::{data_sha256, fmt_date, fmt_issue, Draw};
use crate::eval::prepare_forecast;
use crate::model::roll_all;
use crate::protocol::Protocol;
use crate::select::{equal7, fuse6, fuse7, rank_desc};

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

struct Settled {
    issue: u32,
    date: i32,
    red: [u8; 6],
    blue: u8,
    algos: Vec<Algo>,
}

struct Board {
    sha: String,
    settled: Vec<Settled>,
    next_issue: u32,
    next_algos: Vec<Algo>,
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
        value["pools"] = frozen_pools(root, issue, None)?;
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
    value["pools"] = frozen_pools(root, issue, draw)?;
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
        )?;
        settled.push(Settled {
            issue: row.issue,
            date: draw.date,
            red: draw.red,
            blue: draw.blue,
            algos,
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
    )?;
    Ok(Board {
        sha: data_sha256(draws),
        settled,
        next_issue: last.issue + 1,
        next_algos,
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
) -> Result<Vec<Algo>, String> {
    let weighted = fuse7(expert, weights);
    let equal = fuse7(expert, &equal7());
    let calibrated = crate::ranking::scores(&weighted, previous);
    let shared_blue = order16(&fuse6(blue_p, blue_weights));
    let equal_blue = order16(&fuse6(blue_p, &[1.0 / 6.0; 6]));
    let cover_score = crate::cover_rank::signal(cover_beta, expert)?;
    let mut algos = vec![
        algo("weighted", "协议加权融合", "main", &rank_desc(&weighted), &shared_blue, "协议加权蓝球"),
        algo("equal", "等权融合", "main", &rank_desc(&equal), &equal_blue, "等权蓝球"),
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
        "issues":issues,
        "next_issue":fmt_issue(board.next_issue),
        "note":"名次只用目标期以前的专家特征和25111前冻结的融合权重。盖住6个红球的名次是这6个号码里最靠后的名次；均匀随机顺序的期望是6×34/7。前16名期望命中16×6/33个，蓝球期望名次是8.5。这些对照不是中奖概率，也不替换已封存号码池。"
    })
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

fn frozen_pools(root: &Path, issue: &str, draw: Option<&Draw>) -> Result<Value, String> {
    let path = root.join("daily").join(format!("{issue}.json"));
    if !path.is_file() {
        return Ok(json!({"available":false,"items":[],"note":"本期没有每日冻结档案，页面不补造号码池。"}));
    }
    let doc = crate::recommendations::read_checked(&path)?;
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
        }
    }
    Ok(json!({"available":true,"items":items,"note":"号码池来自该期首次冻结的每日档案。分档池保留当时的2个蓝球；方案行是该方案全部单注的号码并集。"}))
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
        assert_eq!(view["algorithms"].as_array().unwrap().len(), 12);
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
        let mut earlier = draws.clone();
        let newest = earlier.pop().unwrap();
        let replay = issue(&dir, &earlier, &fmt_issue(newest.issue)).unwrap();
        let kept = issue(&dir, &earlier, &prior).unwrap();
        let full_prior = issue(&dir, &draws, &prior).unwrap();
        for (left, right) in replay["algorithms"].as_array().unwrap().iter().zip(view["algorithms"].as_array().unwrap()) {
            assert_eq!(left["red_ranking"], right["red_ranking"]);
            assert_eq!(left["blue_ranking"], right["blue_ranking"]);
            assert!(left["cover_all_six_rank"].is_null());
        }
        assert_eq!(kept["algorithms"], full_prior["algorithms"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn frozen_daily_pools_are_counted_without_rewriting_the_archive() {
        let draws = crate::draw::load_csv(Path::new("data/ssq.csv")).unwrap();
        let draw = draws.last().unwrap();
        let mut reds: Vec<u8> = draw.red.to_vec();
        reds.extend((1..=33).filter(|n| !draw.red.contains(n)).take(10));
        let payload = json!({"number_pools":{"pools":[{"model":"weighted","group":"observation","red_count":16,"blue_count":2,"reds":reds,"blues":[draw.blue, if draw.blue==1 {2} else {1}]}],"comparison_models":[],"coverage_model":null},"small_red_pools":{"models":[{"model":"small_recent","label":"小池近况融合 · 试验","pools":[{"model":"small_recent","red_count":16,"reds":draw.red.to_vec()}]}]},"strategies":{"frozen":{"title":"测试方案","tickets":[{"red":draw.red.to_vec(),"blue":draw.blue}]}}});
        let raw = serde_json::to_vec(&payload).unwrap();
        let dir = std::env::temp_dir().join(format!("ssq-rank-pools-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("daily").join(format!("{}.json", fmt_issue(draw.issue)));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let envelope = json!({"sha256": crate::hash::sha256_hex(&raw), "payload": payload});
        fs::write(&path, serde_json::to_vec(&envelope).unwrap()).unwrap();
        let bytes = fs::read(&path).unwrap();
        let view = issue(&dir, &draws, &fmt_issue(draw.issue)).unwrap();
        assert_eq!(fs::read(&path).unwrap(), bytes);
        let items = view["pools"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["red_hits"], 6);
        assert_eq!(items[0]["blue_hit"], true);
        assert_eq!(items[1]["kind"], "small");
        assert_eq!(items[1]["label"], "小池近况融合 · 试验 · 6红");
        assert_eq!(items[2]["kind"], "strategy");
        assert_eq!(items[1]["covers_all_red"], true);
        let _ = fs::remove_dir_all(&dir);
    }
}
