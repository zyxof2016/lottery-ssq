use crate::draw::Draw;
use crate::protocol::Protocol;

/// 一期开奖之前，仅由更早历史得到的专家分布。
#[derive(Clone)]
pub struct Row {
    pub issue: u32,
    pub date: i32,
    pub red_mask: u64,
    pub blue: u8,
    pub expert_p: [[f64; 33]; 7],
    pub blue_p: [[f64; 16]; 6],
}

pub struct Snapshot {
    pub rows: Vec<Row>,
    pub next_red: [[f64; 33]; 7],
    pub next_blue: [[f64; 16]; 6],
}

#[derive(Clone)]
pub struct State {
    n: u32,
    red_count: [u32; 33],
    blue_count: [u32; 16],
    red_last: [i32; 33],
    blue_last: [i32; 16],
    red_ewma: [f64; 33],
    blue_ewma: [f64; 16],
    cooccur: [[u32; 33]; 33],
    blue_trans: [[u32; 16]; 16],
    prev_red: u64,
    prev_blue: Option<usize>,
    has_prev: bool,
    recent: [u64; 30],
    recent_i: usize,
    recent_filled: usize,
}

impl State {
    pub fn new() -> Self {
        debug_assert_eq!(Protocol::ZONE_WINDOW, 30);
        Self {
            n: 0,
            red_count: [0; 33],
            blue_count: [0; 16],
            red_last: [-1; 33],
            blue_last: [-1; 16],
            red_ewma: [1.0 / 33.0; 33],
            blue_ewma: [1.0 / 16.0; 16],
            cooccur: [[0; 33]; 33],
            blue_trans: [[0; 16]; 16],
            prev_red: 0,
            prev_blue: None,
            has_prev: false,
            recent: [0; 30],
            recent_i: 0,
            recent_filled: 0,
        }
    }

    pub fn expert_dist(&self) -> [[f64; 33]; 7] {
        [
            normalize(&self.score_frequency()),
            normalize(&self.score_ewma()),
            normalize(&self.score_overdue()),
            normalize(&self.score_hot()),
            normalize(&self.score_cooccur()),
            normalize(&self.score_zone()),
            normalize(&self.score_repeat()),
        ]
    }

    pub fn blue_dist(&self) -> [[f64; 16]; 6] {
        [
            normalize(&self.blue_frequency()),
            normalize(&self.blue_ewma),
            normalize(&self.blue_overdue()),
            normalize(&self.blue_hot()),
            normalize(&self.blue_anti_repeat()),
            normalize(&self.blue_markov()),
        ]
    }

    pub fn update(&mut self, d: &Draw) {
        let idx = self.n as i32;
        for &r in &d.red {
            let i = r as usize - 1;
            self.red_count[i] += 1;
            self.red_last[i] = idx;
        }
        let b = d.blue as usize - 1;
        self.blue_count[b] += 1;
        self.blue_last[b] = idx;

        let alpha = Protocol::EWMA_ALPHA;
        let decay = 1.0 - alpha;
        for x in &mut self.red_ewma {
            *x *= decay;
        }
        for &r in &d.red {
            self.red_ewma[r as usize - 1] += alpha / 6.0;
        }
        for x in &mut self.blue_ewma {
            *x *= decay;
        }
        self.blue_ewma[b] += alpha;

        for i in 0..6 {
            for j in (i + 1)..6 {
                let a = d.red[i] as usize - 1;
                let c = d.red[j] as usize - 1;
                self.cooccur[a][c] += 1;
                self.cooccur[c][a] += 1;
            }
        }
        if let Some(prev) = self.prev_blue {
            self.blue_trans[prev][b] += 1;
        }

        self.recent[self.recent_i] = d.red_mask;
        self.recent_i = (self.recent_i + 1) % Protocol::ZONE_WINDOW;
        if self.recent_filled < Protocol::ZONE_WINDOW {
            self.recent_filled += 1;
        }
        self.prev_red = d.red_mask;
        self.prev_blue = Some(b);
        self.has_prev = true;
        self.n += 1;
    }

    fn gap_red(&self, i: usize) -> f64 {
        if self.red_last[i] < 0 {
            self.n as f64
        } else {
            (self.n as i32 - self.red_last[i] - 1) as f64
        }
    }

    fn gap_blue(&self, i: usize) -> f64 {
        if self.blue_last[i] < 0 {
            self.n as f64
        } else {
            (self.n as i32 - self.blue_last[i] - 1) as f64
        }
    }

    fn score_frequency(&self) -> [f64; 33] {
        let mut s = [0.0; 33];
        for i in 0..33 {
            s[i] = self.red_count[i] as f64 + 1.0;
        }
        s
    }

    fn score_ewma(&self) -> [f64; 33] {
        self.red_ewma
    }

    fn score_overdue(&self) -> [f64; 33] {
        let mut s = [0.0; 33];
        for i in 0..33 {
            s[i] = self.gap_red(i) + 1.0;
        }
        s
    }

    fn score_hot(&self) -> [f64; 33] {
        let mut s = [0.0; 33];
        for i in 0..33 {
            s[i] = 1.0 / (self.gap_red(i) + 1.0);
        }
        s
    }

    fn score_cooccur(&self) -> [f64; 33] {
        let mut s = [1.0; 33];
        if !self.has_prev {
            return s;
        }
        let mut bits = self.prev_red;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            for j in 0..33 {
                if i != j {
                    s[j] += self.cooccur[i][j] as f64;
                }
            }
            bits &= bits - 1;
        }
        s
    }

    fn score_zone(&self) -> [f64; 33] {
        if self.recent_filled == 0 {
            return [1.0; 33];
        }
        let mut zc = [0u32; 3];
        for t in 0..self.recent_filled {
            let mut bits = self.recent[t];
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                zc[i / 11] += 1;
                bits &= bits - 1;
            }
        }
        let expected = self.recent_filled as f64 * 2.0;
        let mut s = [0.0; 33];
        for i in 0..33 {
            let deficit = expected - zc[i / 11] as f64;
            s[i] = deficit + expected + 1.0;
        }
        s
    }

    fn score_repeat(&self) -> [f64; 33] {
        let mut s = [0.05; 33];
        if self.has_prev {
            let mut bits = self.prev_red;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                s[i] = 1.0;
                bits &= bits - 1;
            }
        }
        s
    }

    fn blue_frequency(&self) -> [f64; 16] {
        let mut s = [0.0; 16];
        for i in 0..16 {
            s[i] = self.blue_count[i] as f64 + 1.0;
        }
        s
    }

    fn blue_overdue(&self) -> [f64; 16] {
        let mut s = [0.0; 16];
        for i in 0..16 {
            s[i] = self.gap_blue(i) + 1.0;
        }
        s
    }

    fn blue_hot(&self) -> [f64; 16] {
        let mut s = [0.0; 16];
        for i in 0..16 {
            s[i] = 1.0 / (self.gap_blue(i) + 1.0);
        }
        s
    }

    fn blue_anti_repeat(&self) -> [f64; 16] {
        let mut s = self.blue_frequency();
        if let Some(b) = self.prev_blue {
            s[b] *= 0.25;
        }
        s
    }

    fn blue_markov(&self) -> [f64; 16] {
        if let Some(b) = self.prev_blue {
            let mut s = [0.0; 16];
            for j in 0..16 {
                s[j] = self.blue_trans[b][j] as f64 + 1.0;
            }
            s
        } else {
            self.blue_frequency()
        }
    }
}

pub fn normalize<const N: usize>(s: &[f64; N]) -> [f64; N] {
    let mut p = [0.0; N];
    let mut sum = 0.0;
    for i in 0..N {
        let v = if s[i].is_finite() { s[i].max(0.0) } else { 0.0 };
        p[i] = v;
        sum += v;
    }
    if sum <= 1e-15 {
        return [1.0 / N as f64; N];
    }
    for v in &mut p {
        *v /= sum;
    }
    p
}

pub fn roll_all(draws: &[Draw]) -> Result<Snapshot, String> {
    if draws.len() <= Protocol::BURN_IN {
        return Err(format!(
            "至少需要 {} 期历史，当前 {}",
            Protocol::BURN_IN + 1,
            draws.len()
        ));
    }
    let mut state = State::new();
    let mut rows = Vec::with_capacity(draws.len() - Protocol::BURN_IN);
    for (t, d) in draws.iter().enumerate() {
        if t >= Protocol::BURN_IN {
            rows.push(Row {
                issue: d.issue,
                date: d.date,
                red_mask: d.red_mask,
                blue: d.blue - 1,
                expert_p: state.expert_dist(),
                blue_p: state.blue_dist(),
            });
        }
        state.update(d);
    }
    Ok(Snapshot {
        rows,
        next_red: state.expert_dist(),
        next_blue: state.blue_dist(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::synth_draws;

    #[test]
    fn no_leakage_and_probability() {
        let draws = synth_draws(120, 7);
        let mut a = State::new();
        for d in &draws[..80] {
            a.update(d);
        }
        let before = a.expert_dist();
        let mut b = a.clone();
        b.update(&draws[80]);
        let again = a.expert_dist();
        for e in 0..7 {
            let sum: f64 = before[e].iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
            for i in 0..33 {
                assert!((before[e][i] - again[e][i]).abs() < 1e-12);
            }
        }
        let after = b.expert_dist();
        let mut changed = false;
        for e in 0..7 {
            for i in 0..33 {
                if (after[e][i] - before[e][i]).abs() > 1e-12 {
                    changed = true;
                }
            }
        }
        assert!(changed);
    }
}
