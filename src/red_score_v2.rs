//! Read-only 26130-style review: deduplicated red-set likelihood, plus the
//! sealed blue v2 decision compared with the two pure 8-ticket objectives.
//! Does not create archives, tickets or a new published ranking.
use crate::{
    draw::{data_sha256, fmt_issue, Draw},
    hash::sha256_hex,
    red_model_v2::{self as model, Study},
    select::rank_desc,
};
use serde_json::{json, Value};
use std::{
    path::Path,
    sync::{Mutex, OnceLock},
};

fn window(rows: &[&model::Record]) -> Value {
    let n = rows.len();
    let mean = |f: fn(&model::Record) -> f64| {
        if n == 0 {
            None
        } else {
            Some(rows.iter().map(|r| f(r)).sum::<f64>() / n as f64)
        }
    };
    json!({"n":n,"mean_log_excess_vs_uniform":mean(|r| r.log_excess),"mean_reds_inside_top16":mean(|r| r.hits as f64),
        "uniform_expected_reds_inside_top16":16. * 6. / 33.,"full_six_inside_top16":rows.iter().filter(|r| r.full).count(),
        "top16_meaning":"按该期模型的单号入选概率取前16。这是期望中红目标，不等于全部六红同时入池。"})
}

fn slice<'a>(study: &'a Study, lo: u32, hi: u32) -> Vec<&'a model::Record> {
    study.records.iter().filter(|r| r.issue >= lo && r.issue <= hi).collect()
}

fn number_rows(draw: &Draw, inclusion: &[f64; 33], fusion: Option<&Value>) -> Result<Vec<Value>, String> {
    let order = rank_desc(inclusion);
    let cutoff = inclusion[order[15]];
    let fusion_ranking = fusion.and_then(|v| v["payload"]["bundle"]["ranking"].as_array());
    let fusion_scores = fusion.and_then(|v| v["payload"]["bundle"]["scores"].as_array());
    let fusion_cutoff = fusion_ranking.and_then(|ranking| {
        let number = ranking.get(15)?.as_u64()? as usize;
        fusion_scores?.get(number - 1)?.as_f64()
    });
    let previous = fusion
        .and_then(|v| v["payload"]["bundle"]["previous_reds"].as_array())
        .map(|a| a.iter().filter_map(|n| n.as_u64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let ratio = |value: f64, cutoff: f64| {
        (cutoff > 0.).then(|| value / cutoff).filter(|v| v.is_finite())
    };
    let mut rows = Vec::new();
    for n in draw.red {
        let i = (n - 1) as usize;
        let model_rank = order.iter().position(|j| *j == i).unwrap() + 1;
        let fusion_rank = fusion_ranking.and_then(|a| a.iter().position(|v| v.as_u64() == Some(n as u64)).map(|p| p + 1));
        let fusion_score = fusion_scores.and_then(|a| a.get(i)).and_then(Value::as_f64);
        rows.push(json!({"number":n,"model_rank":model_rank,"model_inclusion":inclusion[i],"in_model_top16":model_rank<=16,
            "model_inclusion_ratio_to_rank16":ratio(inclusion[i], cutoff),"fusion_v2_rank":fusion_rank,"fusion_v2_score":fusion_score,
            "in_fusion_v2_top16":fusion_rank.map(|r| r<=16),"fusion_v2_score_ratio_to_rank16":fusion_score.zip(fusion_cutoff).and_then(|(s,c)| ratio(s,c)),
            "previous_red":previous.contains(&(n as u64))}));
    }
    Ok(rows)
}

fn assignment(name: &str, blues: &[u8], p: &[f64], winner: u8) -> Value {
    let sum: f64 = blues.iter().map(|b| p[(b - 1) as usize]).sum();
    let mut unique = blues.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let coverage: f64 = unique.iter().map(|b| p[(b - 1) as usize]).sum();
    json!({"id":name,"tickets":blues,"distinct_blues":unique,"sum_of_ticket_probabilities":sum,
        "probability_winning_blue_is_used":coverage,"includes_winning_blue":unique.contains(&winner)})
}

fn blue_decision(root: &Path, draws: &[Draw], issue: &str, winner: u8) -> Result<Value, String> {
    let path = root.join(format!("blue-score-v2/{issue}.json"));
    if !path.exists() {
        return Ok(Value::Null);
    }
    let envelope: Value = serde_json::from_slice(&crate::blue_v2::artifact(root, draws, issue)?)
        .map_err(|_| "蓝球v2原档无法读取")?;
    let prediction = &envelope["payload"]["prediction"];
    let probabilities: Vec<f64> = serde_json::from_value(prediction["probabilities"].clone()).map_err(|_| "蓝球v2分布无效")?;
    let ranking: Vec<u8> = serde_json::from_value(prediction["ranking"].clone()).map_err(|_| "蓝球v2排序无效")?;
    if probabilities.len() != 16 || ranking.len() != 16 || probabilities.iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err("蓝球v2分布长度无效".into());
    }
    let p: [f64; 16] = probabilities.try_into().unwrap();
    let rank = ranking.iter().position(|n| *n == winner).map(|i| i + 1);
    let fusion_path = root.join(format!("fusion-v2/{issue}.json"));
    let sealed = if fusion_path.exists() {
        let fusion: Value = serde_json::from_slice(&crate::fusion_v2::artifact(root, draws, issue)?)
            .map_err(|_| "融合v2原档无法读取")?;
        let plan = fusion["payload"]["bundle"]["singles"]
            .as_array()
            .and_then(|a| a.iter().find(|p| p["red_count"] == 16))
            .ok_or("融合v2缺少16红8注")?;
        let blues: Vec<u8> = plan["tickets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["blue"].as_u64().ok_or("融合票蓝球无效").map(|n| n as u8))
            .collect::<Result<_, _>>()?;
        json!(assignment("sealed_v2_two_blues_four_each", &blues, &p, winner))
    } else {
        Value::Null
    };
    let best = ranking[0];
    let distinct: Vec<u8> = ranking.iter().copied().take(8).collect();
    Ok(json!({"source":"sealed_blue_score_v2_distribution","archive_sha256":envelope["sha256"],"winning_blue":winner,
        "v2_rank":rank,"v2_probability":p[(winner-1)as usize],"uniform_probability":1./16.,
        "log_excess_vs_uniform":(p[(winner-1)as usize]*16.).ln(),"top2":ranking[..2],"top2_probability_mass":p[(ranking[0]-1)as usize]+p[(ranking[1]-1)as usize],
        "uniform_top2_probability_mass":2./16.,"rank2_probability":p[(ranking[1]-1)as usize],
        "assignments":{"sealed_v2_16_red_8_tickets":sealed,"all_eight_tickets_on_highest_probability":assignment("all_eight_on_highest",&[best;8],&p,winner),
            "eight_distinct_highest":assignment("eight_distinct",&distinct,&p,winner)},
        "objective_note":"8注各配一个蓝球时，单注概率之和的模型最优是8注都用概率最高的蓝球；至少覆盖一次开奖蓝的模型最优是按概率取8个不同蓝球。当前4注加4注不是这两个目标中的任何一个。8个不同蓝球覆盖一半蓝球空间，覆盖到开奖蓝不等于模型把它识别出来。",
        "adopted":false}))
}

pub fn report(root: &Path, draws: &[Draw]) -> Result<Value, String> {
    let last = draws.last().ok_or("没有开奖数据")?;
    let issue = fmt_issue(last.issue);
    let fusion_file = root.join(format!("fusion-v2/{issue}.json"));
    let blue_file = root.join(format!("blue-score-v2/{issue}.json"));
    let fingerprint = |path: &Path| -> Result<String, String> {
        if path.exists() { Ok(sha256_hex(&std::fs::read(path).map_err(|e| e.to_string())?)) } else { Ok(String::new()) }
    };
    let key = format!("{}:{}:{}", data_sha256(draws), fingerprint(&fusion_file)?, fingerprint(&blue_file)?);
    static CACHE: OnceLock<Mutex<Option<(String, Value)>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(|| Mutex::new(None)).lock().map_err(|_| "红球v2复盘缓存异常")?;
    if let Some((saved, value)) = cache.as_ref() {
        if saved == &key {
            return Ok(value.clone());
        }
    }
    let study = model::study(draws)?;
    let fusion = if fusion_file.exists() {
        Some(serde_json::from_slice::<Value>(&crate::fusion_v2::artifact(root, draws, &issue)?).map_err(|_| "融合v2原档无法读取")?)
    } else {
        None
    };
    let inclusion = study.latest_inclusion.ok_or("红球v2没有训练边界后的开奖")?;
    if study.latest_issue != Some(last.issue) {
        return Err("红球v2最新评分与最新开奖不一致".into());
    }
    let numbers = number_rows(last, &inclusion, fusion.as_ref())?;
    let confirm = window(&slice(&study, 25111, 26115));
    let mean = confirm["mean_log_excess_vs_uniform"].as_f64();
    let reason = if mean.is_some_and(|v| v > 0.) {
        "确认窗完整六红对数评分的点估计高于均匀，但这些历史窗口已被查看，且没有本模型的独立前瞻封存。结果不替换已封存融合v2，不生成新的16红或8注。"
    } else {
        "确认窗完整六红对数评分没有高于均匀。去掉重复方向后的红球模型不替换已封存融合v2，也不生成新票面。"
    };
    let validation = slice(&study, 23101, 25110);
    let tracking = slice(&study, 26116, u32::MAX);
    let held: Vec<_> = study.records.iter().collect();
    let value = json!({"version":2,"kind":"deduplicated_red_set_log_score_review","target_issue":fmt_issue(last.issue+1),"cutoff_issue":issue,
        "data_sha256":data_sha256(draws),"policy":model::policy(),"training":{"n":study.fit.n,"coefficients":study.fit.beta,"iterations":study.fit.iterations,
            "normalized_gradient":study.fit.gradient,"objective_gap_upper_bound":study.fit.upper_gap,"training_history_sha256":study.training_sha},
        "latest":{"issue":issue,"draw":{"red":last.red,"blue":last.blue},"log_excess_vs_uniform":study.records.last().unwrap().log_excess,
            "reds_inside_model_top16":study.records.last().unwrap().hits,"full_six_inside_model_top16":study.records.last().unwrap().full,
            "numbers":numbers,"fusion_v2_archive_sha256":fusion.as_ref().map(|v| v["sha256"].clone()).unwrap_or(Value::Null)},
        "blue_decision":blue_decision(root, draws, &issue, last.blue)?,
        "next":{"issue":fmt_issue(last.issue+1),"ranking":model::ranking(&study.next_inclusion),"inclusion":study.next_inclusion.to_vec(),
            "top16":model::ranking(&study.next_inclusion).into_iter().take(16).collect::<Vec<_>>(),"tickets_generated":false,"actual_recommendation":false},
        "windows":{"validation_23101_25110":window(&validation),"confirmation_exploratory_25111_26115":confirm,
            "tracking_after_26115":window(&tracking),"all_after_training":window(&held)},
        "adoption":{"promoted":false,"tickets_generated":false,"published_fusion_changed":false,"reason":reason},
        "writes_archives":false,"note":"已封存融合排序仍是专家混合分，遗漏和热度分开加权，近期项还会在融合之外再加一次。本模型把它们收成每个方向一个系数，并用完整六红概率计分。名次只用于和16红截线对照。"});
    *cache = Some((key, value.clone()));
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_read_only_and_does_not_promote() {
        let draws = crate::draw::load_csv(std::path::Path::new("data/ssq.csv")).unwrap();
        let dir = std::env::temp_dir().join(format!("ssq-red-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let before = std::fs::read_dir(&dir).unwrap().count();
        let v = report(&dir, &draws).unwrap();
        assert_eq!(v["adoption"]["promoted"], false);
        assert_eq!(v["writes_archives"], false);
        assert_eq!(v["next"]["tickets_generated"], false);
        assert!(v["latest"]["numbers"].as_array().unwrap().len() == 6);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), before);
        let again = report(&dir, &draws).unwrap();
        assert_eq!(v["training"]["coefficients"], again["training"]["coefficients"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
