//! Reproduce the three rejected causal ridge candidates; never publishes numbers.
use serde_json::{json, Value};
use ssq::{draw, model, protocol::Protocol, select};
use std::collections::VecDeque;

fn features(ps: &[[f64; 33]; 7]) -> [[f64; 7]; 33] {
    let mut x = [[0.0; 7]; 33];
    for e in 0..7 {
        let order = select::rank_desc(&ps[e]);
        let mut start = 0;
        while start < 33 {
            let mut end = start + 1;
            while end < 33
                && select::score_key(ps[e][order[end]]) == select::score_key(ps[e][order[start]])
            {
                end += 1;
            }
            let v = (17.0 - (start + 1 + end) as f64 / 2.0) / 16.0;
            for j in start..end {
                x[order[j]][e] = v;
            }
            start = end;
        }
    }
    x
}

#[derive(Default)]
struct Ridge {
    a: [[f64; 7]; 7],
    b: [f64; 7],
    history: VecDeque<([[f64; 7]; 33], u64)>,
    window: usize,
}
impl Ridge {
    fn coefficients(&self) -> [f64; 7] {
        let mut m = [[0.0; 8]; 7];
        for i in 0..7 {
            m[i][..7].copy_from_slice(&self.a[i]);
            m[i][i] += 1100.0; // 100 neutral periods * 33 balls * rank-feature variance 1/3.
            m[i][7] = self.b[i];
        }
        for i in 0..7 {
            let p = (i..7)
                .max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))
                .unwrap();
            m.swap(i, p);
            let d = m[i][i];
            for j in i..8 {
                m[i][j] /= d;
            }
            for r in 0..7 {
                if r != i {
                    let f = m[r][i];
                    for j in i..8 {
                        m[r][j] -= f * m[i][j];
                    }
                }
            }
        }
        std::array::from_fn(|i| m[i][7])
    }
    fn accumulate(&mut self, x: &[[f64; 7]; 33], mask: u64, sign: f64) {
        for i in 0..33 {
            let y = if mask & (1 << i) != 0 { 1.0 } else { 0.0 };
            for e in 0..7 {
                self.b[e] += sign * x[i][e] * (y - 6.0 / 33.0);
                for f in 0..7 {
                    self.a[e][f] += sign * x[i][e] * x[i][f];
                }
            }
        }
    }
    fn update(&mut self, x: [[f64; 7]; 33], mask: u64) {
        self.accumulate(&x, mask, 1.0);
        self.history.push_back((x, mask));
        if self.window > 0 && self.history.len() > self.window {
            let (x, mask) = self.history.pop_front().unwrap();
            self.accumulate(&x, mask, -1.0);
        }
    }
}
fn summary(rows: &[Value]) -> Value {
    let cells:Vec<_>=["weighted","ridge_expanding","ridge156","ridge600"].into_iter().enumerate().flat_map(|(m,model)| {
        ssq::pool::TIER_RED_SIZES.into_iter().enumerate().map(move |(j,k)| {
            let hits:u64=rows.iter().map(|r|r["hits"][m][j].as_u64().unwrap()).sum();
            json!({"model":model,"red_count":k,"n":rows.len(),"mean_red_hits":hits as f64/rows.len() as f64,
                "full_red_periods":rows.iter().filter(|r|r["hits"][m][j]==6).count()})
        })
    }).collect();
    json!({"n":rows.len(),"cells":cells})
}
fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = draw::load_csv(std::path::Path::new(&path))?;
    let snap = model::roll_all(&draws)?;
    let mut candidates: [Ridge; 3] = [0, 156, 600].map(|window| Ridge {
        window,
        ..Default::default()
    });
    let (mut red_sum, mut nfit) = ([0.0; 7], 0usize);
    let mut rows = Vec::new();
    for row in &snap.rows {
        let x = features(&row.expert_p);
        if nfit >= 30 {
            let mut orders = vec![select::rank_desc(&select::fuse7(
                &row.expert_p,
                &select::softmax_shrink(&red_sum, nfit as f64),
            ))];
            for candidate in &candidates {
                let c = candidate.coefficients();
                let scores: [f64; 33] =
                    std::array::from_fn(|i| x[i].iter().zip(c).map(|(x, c)| x * c).sum());
                orders.push(select::rank_desc(&scores));
            }
            rows.push(json!({"issue":row.issue,"hits":orders.iter().map(|o|ssq::pool::TIER_RED_SIZES.map(|k|o[..k].iter().filter(|&&n|row.red_mask&(1<<n)!=0).count())).collect::<Vec<_>>()}));
        }
        for candidate in &mut candidates {
            candidate.update(x, row.red_mask);
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            for e in 0..7 {
                red_sum[e] += select::rank_log_score(&row.expert_p[e], row.red_mask);
            }
            nfit += 1;
        }
    }
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .ok_or("缺少确认起点")?;
    let end = (start + Protocol::CONFIRMATION_LEN).min(rows.len());
    let report = json!({"kind":"rejected_ranking_candidates_exploratory_replay","actual_recommendation":false,
        "promoted":false,"reason":"三种学习型排名在近期全6红率未优于原排名，不进入自动推荐。",
        "method":"seven_tie_aware_centered_midranks_joint_ridge_inclusion_regression","ridge_penalty":1100,
        "windows_considered":[0,156,600],"data_sha256":draw::data_sha256(&draws),
        "source_sha256":ssq::hash::sha256_hex(concat!(include_str!("ranking_candidates.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/protocol.rs")).as_bytes()),
        "windows":{"development":summary(&rows[..start]),"confirmation_exploratory":summary(&rows[start..end]),
            "recent60":summary(&rows[rows.len().saturating_sub(60)..]),"all_oos":summary(&rows)},
        "note":"每期评分先于该期开奖加入拟合；三种窗口均完整报告。窗口之间和与校准模型的事后比较属于探索，不能算独立验证。"});
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
