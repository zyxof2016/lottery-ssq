use serde::Serialize;

use crate::draw::{fmt_date, fmt_issue, Draw};
use crate::model::{roll_all, Row};
use crate::protocol::Protocol;
use crate::review::{self, Review};
use crate::select::{forecast, rank_log_score, softmax_shrink, Forecast};

#[derive(Clone, Serialize)]
pub struct ReplayTicket {
    pub red: [u8; 6],
    pub blue: u8,
    pub red_hits: u8,
    pub blue_hit: bool,
}

#[derive(Clone, Serialize)]
pub struct ReplayItem {
    pub issue: String,
    pub date: String,
    /// 确认窗起点之前按更早数据重估权重；之后用停在起点前的冻结权重。
    pub weight_mode: &'static str,
    pub actual_red: [u8; 6],
    pub actual_blue: u8,
    pub best_red: u8,
    pub avg_red: f64,
    pub top1_blue: bool,
    pub any_blue: bool,
    pub tickets: Vec<ReplayTicket>,
    pub review: Review,
}

/// 用每一期开奖之前能看见的数据回放票面，并和当期开奖对照。
pub fn replay(draws: &[Draw]) -> Result<Vec<ReplayItem>, String> {
    let snap = roll_all(draws)?;
    let rows = &snap.rows;
    let mut red_sum = [0.0; 7];
    let mut blue_sum = [0.0; 6];
    let mut n_w = 0usize;
    let mut frozen_red: Option<[f64; 7]> = None;
    let mut frozen_blue: Option<[f64; 6]> = None;
    let mut out = Vec::new();
    for row in rows {
        if n_w >= 30 {
            let frozen = row.issue >= Protocol::CONFIRM_START_ISSUE;
            let (weights, blue_weights, mode) = if frozen {
                if frozen_red.is_none() {
                    frozen_red = Some(softmax_shrink(&red_sum, n_w as f64));
                    frozen_blue = Some(softmax_shrink(&blue_sum, n_w as f64));
                }
                (
                    frozen_red.unwrap(),
                    frozen_blue.unwrap(),
                    "冻结",
                )
            } else {
                (
                    softmax_shrink(&red_sum, n_w as f64),
                    softmax_shrink(&blue_sum, n_w as f64),
                    "展开",
                )
            };
            out.push(score_row(row, &forecast(&row.expert_p, &row.blue_p, &weights, &blue_weights), mode));
        }
        if row.issue < Protocol::CONFIRM_START_ISSUE {
            accumulate(row, &mut red_sum, &mut blue_sum);
            n_w += 1;
        }
    }
    Ok(out)
}

fn accumulate(row: &Row, red_sum: &mut [f64; 7], blue_sum: &mut [f64; 6]) {
    for e in 0..7 {
        red_sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
    }
    let y = row.blue as usize;
    for e in 0..6 {
        blue_sum[e] += row.blue_p[e][y].clamp(1e-12, 1.0).ln();
    }
}

fn score_row(row: &Row, fc: &Forecast, mode: &'static str) -> ReplayItem {
    let mut actual_red = [0u8; 6];
    let mut n = 0;
    let mut bits = row.red_mask;
    while bits != 0 {
        let i = bits.trailing_zeros();
        if n < 6 {
            actual_red[n] = (i + 1) as u8;
            n += 1;
        }
        bits &= bits - 1;
    }
    let y = row.blue;
    let tickets: Vec<ReplayTicket> = fc
        .tickets
        .iter()
        .zip(fc.ticket_blues.iter())
        .map(|(mask, blue)| {
            let mut red = [0u8; 6];
            let mut k = 0;
            let mut bits = *mask;
            while bits != 0 && k < 6 {
                let i = bits.trailing_zeros();
                red[k] = (i + 1) as u8;
                k += 1;
                bits &= bits - 1;
            }
            ReplayTicket {
                red,
                blue: *blue as u8 + 1,
                red_hits: (mask & row.red_mask).count_ones() as u8,
                blue_hit: *blue == y as usize,
            }
        })
        .collect();
    let best_red = tickets.iter().map(|t| t.red_hits).max().unwrap_or(0);
    let avg_red = if tickets.is_empty() {
        0.0
    } else {
        tickets.iter().map(|t| t.red_hits as f64).sum::<f64>() / tickets.len() as f64
    };
    let actual_blue = y + 1;
    let review = review::explain(fc, &actual_red, actual_blue);
    ReplayItem {
        issue: fmt_issue(row.issue),
        date: fmt_date(row.date),
        weight_mode: mode,
        actual_red,
        actual_blue,
        best_red,
        avg_red,
        top1_blue: fc.blue_order.first().copied() == Some(y as usize),
        any_blue: tickets.iter().any(|t| t.blue_hit),
        review,
        tickets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::synth_draws;

    #[test]
    fn replay_is_causal_and_scored() {
        let items = replay(&synth_draws(220, 4)).unwrap();
        assert!(items.len() > 10);
        assert!(items.iter().all(|it| it.weight_mode == "展开"));
        assert!(items.iter().all(|it| it.best_red <= 6 && it.tickets.len() == 8));
        assert!(items.iter().all(|it| it.review.red_draw_n == 6 && it.review.covered_red_n >= 6));
        let mut prev = String::new();
        for it in &items {
            assert!(it.issue > prev);
            prev = it.issue.clone();
        }
    }
}
