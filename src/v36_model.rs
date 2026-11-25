//! Deterministic adaptation of the lottery-predictor-v3.6 scoring script.
use crate::{draw::Draw, select::rank_desc};

const REDS: usize = 33;
const BLUES: usize = 16;
pub(crate) const RED_POOL: usize = 12;
pub(crate) const BLUE_POOL: usize = 4;

pub(crate) struct History {
    red_count: [u32; REDS],
    blue_count: [u32; BLUES],
    red_seen: [usize; REDS],
    blue_seen: [usize; BLUES],
    pairs: [[u32; REDS]; REDS],
}

impl Default for History {
    fn default() -> Self {
        Self {
            red_count: [0; REDS],
            blue_count: [0; BLUES],
            red_seen: [0; REDS],
            blue_seen: [0; BLUES],
            pairs: [[0; REDS]; REDS],
        }
    }
}

pub(crate) struct Scores {
    pub(crate) red: [f64; REDS],
    pub(crate) blue: [f64; BLUES],
}

impl History {
    pub(crate) fn add(&mut self, draw: &Draw, position: usize) {
        for (index, &red) in draw.red.iter().enumerate() {
            let r = usize::from(red - 1);
            self.red_count[r] += 1;
            self.red_seen[r] = position;
            for &other in &draw.red[index + 1..] {
                self.pairs[r][usize::from(other - 1)] += 1;
            }
        }
        let blue = usize::from(draw.blue - 1);
        self.blue_count[blue] += 1;
        self.blue_seen[blue] = position;
    }

    pub(crate) fn score(&self, prefix: &[Draw]) -> Scores {
        let last = prefix
            .last()
            .expect("prediction requires a nonempty prefix");
        let n = prefix.len();
        let red_gaps = self.red_seen.map(|seen| n - seen);
        let blue_gaps = self.blue_seen.map(|seen| n - seen);
        let mean_red_gap = mean(&red_gaps);
        let mean_blue_gap = mean(&blue_gaps);
        let freq_mean = self.red_count.iter().sum::<u32>() as f64 / REDS as f64;
        let freq_sd = (self
            .red_count
            .iter()
            .map(|&v| (f64::from(v) - freq_mean).powi(2))
            .sum::<f64>()
            / (REDS - 1) as f64)
            .sqrt();

        let mut tails = [0; 10];
        for &r in &last.red {
            tails[usize::from(r % 10)] += 1;
        }
        let pair_support = self.pair_support();
        let recent_red_count = |i: usize, span| {
            prefix
                .iter()
                .rev()
                .take(span)
                .filter(|d| d.red_mask & (1u64 << i) != 0)
                .count() as f64
        };
        let was_last = |number| last.red.contains(&number);
        let red = std::array::from_fn(|i| {
            let number = i as u8 + 1;
            let frequency = if freq_sd > 0. {
                18. / (1. + (-(f64::from(self.red_count[i]) - freq_mean) / freq_sd).exp())
            } else {
                9.
            };
            let gap = if mean_red_gap > 0. {
                (11. * red_gaps[i] as f64 / mean_red_gap).min(22.)
            } else {
                11.
            };
            let repeat = if was_last(number) { 15. } else { 0. };
            let recent = recent_red_count(i, 10) * 1.5 + recent_red_count(i, 30) / 3.;
            let neighbors = u8::from(was_last(number - 1)) + u8::from(was_last(number + 1));
            let adjacent = f64::from((neighbors * 5).min(8));
            let tail_count = tails[usize::from(number % 10)];
            let same_tail = if tail_count == 0 {
                2
            } else {
                (5 + tail_count).min(7)
            };
            let pair = if pair_support[i] { 5. } else { 2. };
            frequency + gap + repeat + recent + adjacent + f64::from(same_tail) + pair
        });

        let mut recent_blues = [0u32; BLUES];
        for d in prefix.iter().rev().take(100) {
            recent_blues[usize::from(d.blue - 1)] += 1;
        }
        let max_all = f64::from(self.blue_count.into_iter().max().unwrap());
        let max_recent = f64::from(recent_blues.into_iter().max().unwrap());
        let blue = std::array::from_fn(|i| {
            let gap = blue_gaps[i] as f64;
            let gap_points = if gap > 1.5 * mean_blue_gap {
                20.
            } else if gap > mean_blue_gap {
                15.
            } else {
                10.
            };
            let distance = last.blue.abs_diff(i as u8 + 1);
            let adjacent = if distance == 1 { 5. } else { 0. };
            let same_tail = if distance == 0 || distance == 10 {
                5.
            } else {
                0.
            };
            40. * f64::from(self.blue_count[i]) / max_all
                + 30. * f64::from(recent_blues[i]) / max_recent
                + gap_points
                + adjacent
                + same_tail
        });
        Scores { red, blue }
    }

    fn pair_support(&self) -> [bool; REDS] {
        let mut pairs = Vec::new();
        for (first, counts) in self.pairs.iter().enumerate() {
            for (second, &count) in counts.iter().enumerate().skip(first + 1) {
                if count > 0 {
                    pairs.push((std::cmp::Reverse(count), first, second));
                }
            }
        }
        pairs.sort_unstable();
        let mut supported = [false; REDS];
        for (_, a, b) in pairs.into_iter().take(20) {
            supported[a] = true;
            supported[b] = true;
        }
        supported
    }
}

fn mean<const N: usize>(values: &[usize; N]) -> f64 {
    values.iter().sum::<usize>() as f64 / N as f64
}

pub(crate) fn score(prefix: &[Draw]) -> Scores {
    let mut history = History::default();
    for (i, draw) in prefix.iter().enumerate() {
        history.add(draw, i + 1);
    }
    history.score(prefix)
}

impl Scores {
    pub(crate) fn order(&self) -> ([usize; REDS], [usize; BLUES]) {
        (rank_desc(&self.red), rank_desc(&self.blue))
    }
}
