use crate::protocol::Protocol;

pub fn rank_desc<const N: usize>(s: &[f64; N]) -> [usize; N] {
    let mut idx = [0usize; N];
    for (i, slot) in idx.iter_mut().enumerate() {
        *slot = i;
    }
    idx.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
    idx
}

pub fn hits_prefix(order: &[usize; 33], mask: u64) -> [u32; 8] {
    let ks = Protocol::KS;
    let mut out = [0u32; 8];
    let mut h = 0u32;
    let mut next = 0;
    for r in 0..33 {
        if (mask >> order[r]) & 1 == 1 {
            h += 1;
        }
        while next < ks.len() && ks[next] == r + 1 {
            out[next] = h;
            next += 1;
        }
    }
    out
}

pub fn top_mask(order: &[usize; 33], k: usize) -> u64 {
    let mut m = 0u64;
    for i in 0..k {
        m |= 1u64 << order[i];
    }
    m
}

pub fn fuse7(ps: &[[f64; 33]; 7], w: &[f64; 7]) -> [f64; 33] {
    let mut o = [0.0; 33];
    for e in 0..7 {
        for i in 0..33 {
            o[i] += w[e] * ps[e][i];
        }
    }
    let s: f64 = o.iter().sum();
    if s > 0.0 {
        for v in &mut o {
            *v /= s;
        }
    }
    o
}

pub fn fuse6(ps: &[[f64; 16]; 6], w: &[f64; 6]) -> [f64; 16] {
    let mut o = [0.0; 16];
    for e in 0..6 {
        for i in 0..16 {
            o[i] += w[e] * ps[e][i];
        }
    }
    let s: f64 = o.iter().sum();
    if s > 0.0 {
        for v in &mut o {
            *v /= s;
        }
    }
    o
}

pub fn equal7() -> [f64; 7] {
    [1.0 / 7.0; 7]
}

pub fn votes(orders: &[[usize; 33]; 7], m: usize) -> [u8; 33] {
    let mut v = [0u8; 33];
    for order in orders {
        for j in 0..m {
            v[order[j]] = v[order[j]].saturating_add(1);
        }
    }
    v
}

pub fn mask_min_votes(votes: &[u8; 33], min: u8) -> u64 {
    let mut m = 0u64;
    for (i, &c) in votes.iter().enumerate() {
        if c >= min {
            m |= 1u64 << i;
        }
    }
    m
}

#[derive(Clone, Debug)]
pub struct RescueSpec {
    pub q: usize,
    pub s: usize,
    pub k: usize,
    pub wide: usize,
}

#[derive(Clone, Debug)]
pub struct RescueResult {
    pub mask: u64,
    pub members: Vec<usize>,
    pub rescued: Vec<usize>,
    pub displaced: Vec<usize>,
}

fn worse(score: &[f64; 33], a: usize, b: usize) -> bool {
    match score[a].total_cmp(&score[b]) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => a > b,
    }
}

/// 定稿集大小始终为 k。救援号一旦换入就不再被后续换出挤掉。
pub fn rescue(
    fusion_order: &[usize; 33],
    fusion_score: &[f64; 33],
    vote: &[u8; 33],
    spec: &RescueSpec,
) -> RescueResult {
    let k = spec.k.min(33);
    let wide = spec.wide.clamp(k, 33);
    let mut set: Vec<usize> = fusion_order[..k].to_vec();
    let mut protected = [false; 33];
    let mut rescued = Vec::new();
    let mut displaced = Vec::new();
    if spec.s > 0 && spec.q > 0 {
        let mut cands = Vec::new();
        for idx in k..wide {
            let i = fusion_order[idx];
            if vote[i] as usize >= spec.q {
                cands.push(i);
            }
        }
        cands.sort_by(|&a, &b| {
            vote[b].cmp(&vote[a]).then_with(|| {
                fusion_score[b]
                    .total_cmp(&fusion_score[a])
                    .then(a.cmp(&b))
            })
        });
        for c in cands.into_iter().take(spec.s) {
            let mut worst_pos = None;
            for (p, &idx) in set.iter().enumerate() {
                if protected[idx] {
                    continue;
                }
                match worst_pos {
                    None => worst_pos = Some(p),
                    Some(w) if worse(fusion_score, idx, set[w]) => worst_pos = Some(p),
                    _ => {}
                }
            }
            let Some(pos) = worst_pos else { break };
            let out = set[pos];
            set[pos] = c;
            protected[c] = true;
            displaced.push(out);
            rescued.push(c);
        }
    }
    let mut mask = 0u64;
    for &i in &set {
        mask |= 1u64 << i;
    }
    set.sort_by(|&a, &b| {
        fusion_score[b]
            .total_cmp(&fusion_score[a])
            .then(a.cmp(&b))
    });
    RescueResult {
        mask,
        members: set,
        rescued,
        displaced,
    }
}

pub fn mild_spec() -> RescueSpec {
    RescueSpec {
        q: Protocol::RESCUE_Q,
        s: Protocol::RESCUE_S,
        k: Protocol::PRIMARY_K,
        wide: Protocol::FUSION_WIDE_K,
    }
}

pub fn aggressive_spec() -> RescueSpec {
    RescueSpec {
        q: Protocol::RESCUE_Q_AGG,
        s: Protocol::RESCUE_S_AGG,
        k: Protocol::PRIMARY_K,
        wide: Protocol::FUSION_WIDE_K,
    }
}

/// 预留席保证救援号进入出票池；其余席位按融合分从定稿集里补齐。
pub fn pool_with_seats(
    final_mask: u64,
    fusion_order: &[usize; 33],
    rescued: &[usize],
    pool: usize,
    seats: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut used = 0u64;
    let seat_n = seats.min(pool).min(rescued.len());
    for &r in rescued.iter().take(seat_n) {
        if (final_mask >> r) & 1 == 1 && (used >> r) & 1 == 0 {
            out.push(r);
            used |= 1u64 << r;
        }
    }
    for &i in fusion_order {
        if out.len() >= pool {
            break;
        }
        if (final_mask >> i) & 1 == 1 && (used >> i) & 1 == 0 {
            out.push(i);
            used |= 1u64 << i;
        }
    }
    out
}

#[derive(Clone)]
struct Combo {
    mask: u64,
    score: i64,
}

pub fn build_tickets(pool: &[usize], scores: &[f64; 33], n_tickets: usize, max_overlap: u32) -> Vec<u64> {
    if pool.len() < 6 || n_tickets == 0 {
        return Vec::new();
    }
    let mut fp = [0i64; 33];
    for &i in pool {
        fp[i] = (scores[i] * 1.0e9).round() as i64;
    }
    let mut combos = Vec::new();
    enumerate(pool, &fp, 0, 6, 0, 0, &mut combos);
    combos.sort_by(|a, b| b.score.cmp(&a.score).then(a.mask.cmp(&b.mask)));
    let mut chosen = Vec::new();
    for c in combos {
        if chosen
            .iter()
            .any(|m: &u64| (m & c.mask).count_ones() > max_overlap)
        {
            continue;
        }
        chosen.push(c.mask);
        if chosen.len() == n_tickets {
            break;
        }
    }
    chosen
}

fn enumerate(
    pool: &[usize],
    fp: &[i64; 33],
    start: usize,
    left: usize,
    mask: u64,
    score: i64,
    out: &mut Vec<Combo>,
) {
    if left == 0 {
        out.push(Combo { mask, score });
        return;
    }
    let max_start = pool.len() - left;
    for i in start..=max_start {
        let idx = pool[i];
        enumerate(
            pool,
            fp,
            i + 1,
            left - 1,
            mask | (1u64 << idx),
            score + fp[idx],
            out,
        );
    }
}

pub fn max_red_hits(tickets: &[u64], win: u64) -> u32 {
    tickets
        .iter()
        .map(|t| (t & win).count_ones())
        .max()
        .unwrap_or(0)
}

pub fn mean_red_hits(tickets: &[u64], win: u64) -> f64 {
    if tickets.is_empty() {
        return 0.0;
    }
    let s: u32 = tickets.iter().map(|t| (t & win).count_ones()).sum();
    s as f64 / tickets.len() as f64
}

pub fn rank_log_score(p: &[f64; 33], mask: u64) -> f64 {
    let order = rank_desc(p);
    let mut rank_of = [0usize; 33];
    for (r, &i) in order.iter().enumerate() {
        rank_of[i] = r;
    }
    let mut s = 0.0;
    let mut bits = mask;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        s -= ((rank_of[i] + 1) as f64).ln();
        bits &= bits - 1;
    }
    s
}

pub fn softmax_shrink<const K: usize>(sum_score: &[f64; K], n: f64) -> [f64; K] {
    let mut mean = [0.0; K];
    let mut best = f64::NEG_INFINITY;
    for i in 0..K {
        mean[i] = sum_score[i] / n;
        best = best.max(mean[i]);
    }
    let mut e = [0.0; K];
    let mut s = 0.0;
    for i in 0..K {
        e[i] = ((mean[i] - best) / Protocol::WEIGHT_TAU).exp();
        s += e[i];
    }
    let mut w = [0.0; K];
    let shrink = Protocol::WEIGHT_SHRINKAGE;
    for i in 0..K {
        w[i] = (1.0 - shrink) * (e[i] / s) + shrink / K as f64;
    }
    w
}

pub struct Forecast {
    pub fusion: [f64; 33],
    pub red_order: [usize; 33],
    pub blue: [f64; 16],
    pub blue_order: [usize; 16],
    pub mild: RescueResult,
    pub pool: Vec<usize>,
    pub tickets: Vec<u64>,
    pub ticket_blues: Vec<usize>,
}

pub fn forecast(
    expert_p: &[[f64; 33]; 7],
    blue_p: &[[f64; 16]; 6],
    weights: &[f64; 7],
    blue_weights: &[f64; 6],
) -> Forecast {
    let fusion = fuse7(expert_p, weights);
    let red_order = rank_desc(&fusion);
    let mut orders = [[0usize; 33]; 7];
    for e in 0..7 {
        orders[e] = rank_desc(&expert_p[e]);
    }
    let vote = votes(&orders, Protocol::RESCUE_M);
    let mild = rescue(&red_order, &fusion, &vote, &mild_spec());
    let pool = pool_with_seats(
        mild.mask,
        &red_order,
        &mild.rescued,
        Protocol::POOL,
        Protocol::RESCUE_S,
    );
    let tickets = build_tickets(&pool, &fusion, Protocol::TICKETS, Protocol::MAX_OVERLAP);
    let blue = fuse6(blue_p, blue_weights);
    let blue_order = rank_desc(&blue);
    let slots = Protocol::BLUE_SLOTS.max(1);
    let ticket_blues = (0..tickets.len())
        .map(|i| blue_order[i % slots.min(16)])
        .collect();
    Forecast {
        fusion,
        red_order,
        blue,
        blue_order,
        mild,
        pool,
        tickets,
        ticket_blues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescue_keeps_protected_and_enters_pool() {
        let mut score = [0.0; 33];
        for i in 0..33 {
            score[i] = (33 - i) as f64;
        }
        let order = rank_desc(&score);
        assert_eq!(order[0], 0);
        let mut vote = [0u8; 33];
        vote[15] = 3;
        vote[16] = 3;
        let spec = RescueSpec {
            q: 3,
            s: 2,
            k: 15,
            wide: 20,
        };
        let got = rescue(&order, &score, &vote, &spec);
        assert_eq!(got.mask.count_ones(), 15);
        assert!(got.rescued.contains(&15) && got.rescued.contains(&16));
        assert!(got.displaced.contains(&13) && got.displaced.contains(&14));
        assert!(!got.displaced.iter().any(|d| got.rescued.contains(d)));
        let pool = pool_with_seats(got.mask, &order, &got.rescued, 12, 2);
        assert!(pool.contains(&15) && pool.contains(&16));
        assert_eq!(pool.len(), 12);
    }

    #[test]
    fn tickets_respect_overlap() {
        let mut score = [0.0; 33];
        for i in 0..33 {
            score[i] = (33 - i) as f64;
        }
        let pool: Vec<usize> = (0..12).collect();
        let tickets = build_tickets(&pool, &score, 8, 4);
        assert_eq!(tickets.len(), 8);
        for i in 0..tickets.len() {
            assert_eq!(tickets[i].count_ones(), 6);
            for j in (i + 1)..tickets.len() {
                assert!((tickets[i] & tickets[j]).count_ones() <= 4);
            }
        }
    }

    #[test]
    fn shrink_uniform_when_tied() {
        let sum = [3.0; 7];
        let w = softmax_shrink(&sum, 10.0);
        for v in w {
            assert!((v - 1.0 / 7.0).abs() < 1e-12);
        }
    }
}
