use serde::Serialize;

use crate::select::Forecast;

#[derive(Clone, Debug, Serialize)]
pub struct Repeat {
    pub n: u8,
    pub count: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct Gap {
    pub n: u8,
    pub color: String,
    pub rank: u16,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pick {
    pub n: u8,
    pub count: u8,
    pub rank: u16,
    pub hit: bool,
    pub note: String,
}

/// 一张已经生成的票面相对一期开奖的覆盖说明。
#[derive(Clone, Debug, Serialize)]
pub struct Review {
    pub covered_reds: Vec<u8>,
    pub covered_red_n: usize,
    pub red_repeat: Vec<Repeat>,
    pub covered_blues: Vec<u8>,
    pub blue_repeat: Vec<Repeat>,
    pub win_red_covered: Vec<u8>,
    pub win_red_missed: Vec<u8>,
    pub red_hit_n: usize,
    pub red_draw_n: usize,
    pub blue_covered: bool,
    pub blue_rank: u16,
    pub gaps: Vec<Gap>,
    pub picks: Vec<Pick>,
    pub blue_picks: Vec<Pick>,
}

pub fn explain(fc: &Forecast, win_red: &[u8], win_blue: u8) -> Review {
    let mut red_count = [0u8; 33];
    let mut blue_count = [0u8; 16];
    for (mask, blue) in fc.tickets.iter().zip(fc.ticket_blues.iter()) {
        let mut bits = *mask;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            if i < 33 {
                red_count[i] = red_count[i].saturating_add(1);
            }
            bits &= bits - 1;
        }
        if *blue < 16 {
            blue_count[*blue] = blue_count[*blue].saturating_add(1);
        }
    }
    let covered_reds = counts_to_nums(&red_count);
    let covered_blues: Vec<u8> = blue_count
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, _)| i as u8 + 1)
        .collect();
    let win_red_covered: Vec<u8> = win_red.iter().copied().filter(|n| red_count[*n as usize - 1] > 0).collect();
    let win_red_missed: Vec<u8> = win_red.iter().copied().filter(|n| red_count[*n as usize - 1] == 0).collect();
    let blue_idx = win_blue.saturating_sub(1) as usize;
    let blue_covered = blue_idx < 16 && blue_count[blue_idx] > 0;
    let blue_rank = rank_of(&fc.blue_order, blue_idx.min(15));
    let mut gaps = Vec::new();
    for n in &win_red_missed {
        gaps.push(red_gap(fc, *n));
    }
    if !blue_covered {
        gaps.push(Gap {
            n: win_blue,
            color: "blue".into(),
            rank: blue_rank,
            reason: format!(
                "蓝球融合名次第 {blue_rank} 名。票面只轮换前 {} 个蓝球，它不在这些号码里。",
                covered_blues.len().max(1)
            ),
        });
    }
    let picks = covered_reds
        .iter()
        .copied()
        .map(|n| {
            let count = red_count[n as usize - 1];
            let rank = rank_of(&fc.red_order, n as usize - 1);
            let hit = win_red.contains(&n);
            let note = if hit {
                format!("开奖红球有 {n:02}，它出现在 {count} 注里，融合名次第 {rank} 名。")
            } else {
                format!("开奖红球没有 {n:02}。它出现在 {count} 注里，是因为融合名次第 {rank} 名。")
            };
            Pick { n, count, rank, hit, note }
        })
        .collect();
    let red_hit_n = win_red_covered.len();
    let red_draw_n = win_red.len();
    let blue_picks = covered_blues
        .iter()
        .copied()
        .map(|n| {
            let count = blue_count[n as usize - 1];
            let rank = rank_of(&fc.blue_order, n as usize - 1);
            let hit = n == win_blue;
            let note = if hit {
                format!("开奖蓝球是 {n:02}，它出现在 {count} 注里，融合名次第 {rank} 名。")
            } else {
                format!("开奖蓝球不是 {n:02}。它出现在 {count} 注里，是因为融合名次第 {rank} 名。")
            };
            Pick { n, count, rank, hit, note }
        })
        .collect();
    Review {
        covered_red_n: covered_reds.len(),
        red_repeat: repeats(&red_count),
        covered_reds,
        blue_repeat: repeats(&blue_count),
        covered_blues,
        win_red_covered,
        win_red_missed,
        red_hit_n,
        red_draw_n,
        blue_covered,
        blue_rank,
        gaps,
        picks,
        blue_picks,
    }
}

fn red_gap(fc: &Forecast, n: u8) -> Gap {
    let idx = n as usize - 1;
    let rank = rank_of(&fc.red_order, idx);
    let in_pool = fc.pool.iter().any(|i| *i == idx);
    let displaced = fc.mild.displaced.iter().any(|i| *i == idx);
    let tickets_n = fc.tickets.len();
    let pool_n = fc.pool.len();
    let reason = if displaced {
        format!("红球 {n:02} 融合名次第 {rank} 名。救援换入别的号时把它移出了定稿，所以这 {tickets_n} 注里没有它。")
    } else if !in_pool && rank as usize <= pool_n {
        format!("红球 {n:02} 融合名次第 {rank} 名，本来靠前。出票池的 {pool_n} 个位置先留给救援号，它没有进入池子，所以票面没有它。")
    } else if !in_pool {
        format!("红球 {n:02} 融合名次第 {rank} 名。出票池只有 {pool_n} 个红球，它在池外，票面不会出现。")
    } else {
        format!(
            "红球 {n:02} 融合名次第 {rank} 名，当时在出票池里。{} 注要限制红球重合，选票时没有任何一注带上它。",
            fc.tickets.len()
        )
    };
    Gap {
        n,
        color: "red".into(),
        rank,
        reason,
    }
}

fn rank_of(order: &[usize], idx: usize) -> u16 {
    order.iter().position(|i| *i == idx).map(|p| p as u16 + 1).unwrap_or(0)
}

fn counts_to_nums(count: &[u8]) -> Vec<u8> {
    count
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, _)| i as u8 + 1)
        .collect()
}

fn repeats(count: &[u8]) -> Vec<Repeat> {
    let mut rows: Vec<Repeat> = count
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .map(|(i, c)| Repeat { n: i as u8 + 1, count: *c })
        .collect();
    rows.sort_by(|a, b| b.count.cmp(&a.count).then(a.n.cmp(&b.n)));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::{rank_desc, RescueResult};

    #[test]
    fn coverage_names_the_gap_and_counts_repeats() {
        let mut fusion = [0.0; 33];
        for i in 0..33 {
            fusion[i] = (33 - i) as f64;
        }
        let mut blue = [0.0; 16];
        blue[0] = 2.0;
        blue[1] = 1.0;
        let mut tickets = Vec::new();
        for start in 0..2 {
            let mut mask = 0u64;
            for j in 0..6 {
                mask |= 1u64 << (start * 6 + j);
            }
            tickets.push(mask);
        }
        let fc = Forecast {
            fusion,
            red_order: rank_desc(&fusion),
            blue,
            blue_order: rank_desc(&blue),
            mild: RescueResult {
                mask: 0,
                members: vec![],
                rescued: vec![],
                displaced: vec![12],
            },
            pool: (0..12).collect(),
            tickets,
            ticket_blues: vec![0, 1],
        };
        let review = explain(&fc, &[1, 2, 3, 7, 13, 33], 16);
        assert_eq!(review.covered_red_n, 12);
        assert_eq!(review.red_repeat.iter().map(|r| r.count as usize).sum::<usize>(), 12);
        assert_eq!(review.red_hit_n, 4);
        assert_eq!(review.win_red_missed, vec![13, 33]);
        assert!(review.gaps.iter().any(|g| g.n == 13 && g.reason.contains("移出")));
        assert!(review.gaps.iter().any(|g| g.n == 33 && g.reason.contains("池外")));
        assert!(!review.blue_covered);
        assert!(review.gaps.iter().any(|g| g.color == "blue"));
        assert!(review.picks.iter().any(|p| p.n == 1 && p.hit && p.count == 1));
        assert!(review.picks.iter().any(|p| p.n == 12 && !p.hit));
    }
}
