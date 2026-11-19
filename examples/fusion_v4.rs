//! Read-only causal replay of the fusion coverage trial v4 against the v2 and
//! v3 same-budget ticket sets, plus an isolated local acceptance fixture.
//!
//! ```sh
//! cargo run --release --locked --example fusion_v4 -- data/ssq.csv > fusion-v4-replay.json
//! cargo run --release --locked --example fusion_v4 -- data/ssq.csv --fixture /tmp/ssq-v4-accept
//! ```
//!
//! The fixture relabels the last local draw so the gate opens, seals reference,
//! v2 and v4 archives under the given directory and prints a local serve
//! command. It is isolated acceptance only: never upload its archives or CSV
//! to production.
use serde_json::{json, Value};
use ssq::{
    blue_model_v2 as blue,
    cover_rank,
    coverage::balanced_tickets,
    draw::{data_sha256, fmt_issue, load_csv, write_csv},
    eval::prepare_forecast,
    fusion_model_v2, fusion_model_v3, fusion_model_v4, fusion_model_v5, fusion_model_v7,
    fusion_v3_comparison::red_prize_outcomes,
    model::roll_all,
    protocol::Protocol,
    red_model_v2 as red,
    select::{fuse6, rank_desc, rank_log_score, softmax_shrink},
};
use std::{collections::BTreeSet, path::Path};

const RED_OUTCOMES: u64 = 1_107_568;
const UNIFORM: u64 = RED_OUTCOMES * 16;

fn sets_blues(tickets: &[([u8; 6], u8)]) -> (BTreeSet<[u8; 6]>, BTreeSet<u8>) {
    (
        tickets.iter().map(|(r, _)| *r).collect(),
        tickets.iter().map(|(_, b)| *b).collect(),
    )
}
fn measure(tickets: &[([u8; 6], u8)], row: &ssq::model::Row, prefix: usize) -> Result<Value, String> {
    let (sets, blues) = sets_blues(tickets);
    let r = red_prize_outcomes(&sets)?;
    let b = blues.len() as u64;
    let union = sets.iter().fold(0u64, |m, s| {
        s.iter().fold(m, |m, n| m | (1 << (n - 1)))
    });
    Ok(json!({
        "uniform_any_prize_rate": (b * RED_OUTCOMES + (16 - b) * r) as f64 / UNIFORM as f64,
        "full_six_in_union": (union & row.red_mask).count_ones() == 6,
        "blue_covered": blues.contains(&(row.blue + 1)),
        "used_reds": union.count_ones(),
        "used_blues": b,
        "cover_prefix": prefix,
    }))
}
fn summarize(rows: &[Value]) -> Value {
    if rows.is_empty() {
        return Value::Null;
    }
    let n = rows.len();
    let variant = |name: &str| {
        let mean = |k: &str| {
            rows.iter()
                .map(|r| r[name][k].as_f64().unwrap())
                .sum::<f64>()
                / n as f64
        };
        json!({
            "mean_uniform_any_prize_rate": mean("uniform_any_prize_rate"),
            "mean_cover_prefix": mean("cover_prefix"),
            "full_six_periods": rows.iter().filter(|r| r[name]["full_six_in_union"] == true).count(),
            "blue_covered_periods": rows.iter().filter(|r| r[name]["blue_covered"] == true).count(),
            "mean_used_reds": mean("used_reds"),
            "mean_used_blues": mean("used_blues"),
        })
    };
    json!({"n":n,"from":rows[0]["issue"],"to":rows[n-1]["issue"],"v8":variant("v8"),"v7":variant("v7"),"v5":variant("v5"),"v4":variant("v4"),"v2":variant("v2"),"v3":variant("v3")})
}

fn replay(csv: &Path) -> Result<Value, String> {
    let draws = load_csv(csv)?;
    let snap = roll_all(&draws)?;
    let prepared = prepare_forecast(&draws)?;
    let red_fit = red::fit_causal(&snap.rows)?;
    let blue_samples = snap
        .rows
        .iter()
        .skip(30)
        .filter(|r| r.issue <= blue::TRAIN_END)
        .map(|r| {
            Ok(blue::Sample {
                x: blue::features(&r.blue_p)?,
                winner: r.blue as usize,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let blue_fit = blue::fit(&blue_samples)?;
    let cover_samples = snap
        .rows
        .iter()
        .skip(cover_rank::WARMUP_ROWS)
        .filter(|r| r.issue <= red::TRAIN_END)
        .map(|r| {
            Ok(red::Sample {
                x: red::features(&r.expert_p)?,
                mask: r.red_mask,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let cover_fit = cover_rank::fit_with(&cover_samples, fusion_model_v5::RIDGE)?;
    let neutral = [1. / 33.; 33];
    let (mut red_sum, mut blue_sum, mut nfit) = ([0.0; 7], [0.0; 6], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
        if row.issue >= 23101 && (frozen || nfit >= 30) {
            let w = if frozen {
                prepared.weights
            } else {
                softmax_shrink(&red_sum, nfit as f64)
            };
            let bw = if frozen {
                prepared.blue_weights
            } else {
                softmax_shrink(&blue_sum, nfit as f64)
            };
            let q = blue::distribution(&blue_fit.beta, &blue::features(&row.blue_p)?);
            let (s2, _) = fusion_model_v2::scores(&row.expert_p, &w, &neutral)?;
            let o2 = rank_desc(&s2);
            let masks2 = balanced_tickets(&o2[..16], 8)?;
            let bo2 = rank_desc(&fuse6(&row.blue_p, &bw));
            let t2: Vec<([u8; 6], u8)> = masks2
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let mut ns: [u8; 6] = std::array::from_fn(|k| {
                        (0..33).filter(|b| m & (1 << b) != 0).nth(k).unwrap() as u8 + 1
                    });
                    ns.sort_unstable();
                    (ns, bo2[i % 2] as u8 + 1)
                })
                .collect();
            let (s4, _) = fusion_model_v4::scores(&row.expert_p, &w, &neutral)?;
            let o4 = rank_desc(&s4);
            let ranking4: Vec<u8> = o4.iter().map(|i| (i + 1) as u8).collect();
            let d4 = fusion_model_v3::Distribution::new(&s4, &q)?;
            let sel = fusion_model_v4::select_tickets(&d4, &ranking4)?;
            let blues4 = d4.blue_order()[..8].to_vec();
            let t4: Vec<([u8; 6], u8)> =
                sel.sets.iter().zip(&blues4).map(|(r, b)| (*r, *b)).collect();
            let rw = red::weights(&red_fit.beta, &red::features(&row.expert_p)?)?;
            let d3 = fusion_model_v3::Distribution::new(&rw, &q)?;
            let t3 = d3.top(8)?.tickets;
            let s5 = cover_rank::signal(&cover_fit.beta, &row.expert_p)?;
            let o5 = rank_desc(&s5);
            let ranking5: Vec<u8> = o5.iter().map(|i| (i + 1) as u8).collect();
            let w5: Vec<f64> = s5.iter().map(|v| v.exp()).collect();
            let d5 = fusion_model_v3::Distribution::new(&w5, &q)?;
            let sel5 = fusion_model_v4::select_tickets(&d5, &ranking5)?;
            let blues5 = d5.blue_order()[..8].to_vec();
            let t5: Vec<([u8; 6], u8)> =
                sel5.sets.iter().zip(&blues5).map(|(r, b)| (*r, *b)).collect();
            let sel7 = fusion_model_v7::select_tickets(&d4, &ranking4)?;
            let blues7 = d4.blue_order()[..8].to_vec();
            let t7: Vec<([u8; 6], u8)> =
                sel7.sets.iter().zip(&blues7).map(|(r, b)| (*r, *b)).collect();
            let blues8 = d4.blue_order()[..4].to_vec();
            let t8: Vec<([u8; 6], u8)> = sel7
                .sets
                .iter()
                .enumerate()
                .map(|(i, r)| (*r, blues8[i / 2]))
                .collect();
            rows.push(json!({
                "issue": fmt_issue(row.issue),
                "v8": measure(&t8, row, fusion_model_v5::cover_prefix(&o4, row.red_mask))?,
                "v7": measure(&t7, row, fusion_model_v5::cover_prefix(&o4, row.red_mask))?,
                "v2": measure(&t2, row, fusion_model_v5::cover_prefix(&o2, row.red_mask))?,
                "v4": measure(&t4, row, fusion_model_v5::cover_prefix(&o4, row.red_mask))?,
                "v3": measure(&t3, row, fusion_model_v5::cover_prefix(&rank_desc(&rw), row.red_mask))?,
                "v5": measure(&t5, row, fusion_model_v5::cover_prefix(&o5, row.red_mask))?,
            }));
        }
        if !frozen {
            for e in 0..7 {
                red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
            }
            for e in 0..6 {
                blue_sum[e] += row.blue_p[e][row.blue as usize].clamp(1e-12, 1.0).ln();
            }
            nfit += 1;
        }
    }
    let window = |lo: u32, hi: u32| {
        let rs: Vec<_> = rows
            .iter()
            .filter(|r| {
                let i: u32 = r["issue"].as_str().unwrap().parse().unwrap();
                i >= lo && i <= hi
            })
            .cloned()
            .collect();
        summarize(&rs)
    };
    Ok(json!({
        "kind": "exploratory_fusion_v4_coverage_replay",
        "actual_recommendation": false,
        "data_sha256": data_sha256(&draws),
        "cutoff_issue": fmt_issue(prepared.cutoff_issue),
        "policy": fusion_model_v4::policy(),
        "uniform_outcomes": UNIFORM,
        "windows": {
            "development": window(23101, 25110),
            "confirmation_exploratory": window(25111, 26115),
            "tracking": window(26116, u32::MAX),
            "recent60": summarize(&rows[rows.len().saturating_sub(60)..]),
            "all_oos": summarize(&rows),
        },
        "latest_replay": rows.last(),
        "notes": [
            "逐期只用此前数据；25111前权重按协议展开拟合，之后冻结。参考支持在回放中取中性均匀，实际封存票面使用同期参考父档支持。",
            "v2、v4、v5红球各取自身评分前16并均衡8注，v7取v4评分前24并均衡8注（每红2次）；v2配协议蓝榜前2，v4、v5、v7配冻结蓝v2前8，v8配前4每蓝2注；v3为全33红16蓝无约束质量最优8注。v5排序为预声明正则0.005的最小盖住名次信号。",
            "cover_prefix为盖住全部6个开奖红球所需的最靠后名次；均匀随机顺序期望为6*34/7≈29.14。",
            "均匀常规奖覆盖按全部C(33,6)×16个等可能结果精确枚举去重；这是覆盖结构差异，不是已验证的预测优势。",
            "本回放是事后探索材料，不计入v4、v5实际首次留档样本；多候选已被查看，近期改善不能视作独立确认。",
        ],
    }))
}

fn fixture(csv: &Path, dir: &Path) -> Result<(), String> {
    let mut draws = load_csv(csv)?;
    let last = draws.last_mut().ok_or("没有开奖数据")?;
    last.issue = 26133;
    last.date = 20261117;
    let now = ssq::recommendations::now_unix();
    let gate = ssq::cycle::decide(now, last.issue, last.date, false);
    if !gate.publish {
        return Err(format!("隔离验收当前不可封存：{}", gate.reason));
    }
    let target = gate.target.clone();
    std::fs::create_dir_all(dir.join("data")).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(dir.join("publish")).map_err(|e| e.to_string())?;
    let root = dir.join("recommendations");
    std::fs::create_dir_all(root.join("reference-cache")).map_err(|e| e.to_string())?;
    write_csv(&dir.join("data/ssq.csv"), &draws)?;
    let cache = json!({"version":1,"data_sha256":data_sha256(&draws),"data_cutoff":"26133","target_issue":target,"generated_at_unix":now-100,"engine_sha256":"a".repeat(64),"config_sha256":"b".repeat(64),"tickets":(0..5).map(|i|json!({"red":(1..=6).map(|n|n+i*5).collect::<Vec<_>>(),"blue":i+1})).collect::<Vec<_>>()});
    std::fs::write(
        root.join(format!("reference-cache/{}.json", data_sha256(&draws))),
        serde_json::to_vec(&cache).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let reference = ssq::reference::freeze(&root, &draws, &target, gate.deadline_unix)?;
    let v2 = ssq::fusion_v2::freeze(&root, &draws, &target, gate.deadline_unix)?;
    let v4 = ssq::fusion_v4::freeze(&root, &draws, &target, gate.deadline_unix)?;
    let v5 = ssq::fusion_v5::freeze(&root, &draws, &target, gate.deadline_unix)?;
    let v7 = ssq::fusion_v7::freeze(&root, &draws, &target, gate.deadline_unix)?;
    let v8 = ssq::fusion_v8::freeze(&root, &draws, &target, gate.deadline_unix)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "warning": "隔离本地验收环境：末期的期号/日期已重标注，档案与CSV禁止上传生产。",
            "target": target,
            "reference_status": reference["status"],
            "v2_status": v2["status"],
            "v4_status": v4["status"],
            "v5_status": v5["status"],
            "v7_status": v7["status"],
            "v8_status": v8["status"],
            "serve": format!("cargo run --release -- serve --data {} --audit {}/audit/log.jsonl --out {} --bind 127.0.0.1:6791", dir.join("data/ssq.csv").display(), dir.display(), dir.join("publish").display()),
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let csv = Path::new(args.get(1).ok_or("需要CSV路径")?);
    if let Some(pos) = args.iter().position(|a| a == "--fixture") {
        let dir = Path::new(args.get(pos + 1).ok_or("--fixture 需要目录")?);
        return fixture(csv, dir);
    }
    let report = replay(csv)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}