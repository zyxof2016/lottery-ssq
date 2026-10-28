use serde_json::{json, Value};
use ssq::{draw, model, protocol::Protocol, select};
fn ranked(s: &[f64; 33], base: &[usize; 33]) -> [usize; 33] {
    let mut o = *base;
    o.sort_by(|&a, &b| select::score_key(s[b]).cmp(&select::score_key(s[a])));
    o
}
fn quotas(base: &[usize; 33], previous: u64, ceil: bool) -> [usize; 33] {
    let (mut used, mut repeats) = (0u64, 0usize);
    std::array::from_fn(|p| {
        let cap = if ceil {
            (6 * (p + 1)).div_ceil(33)
        } else {
            6 * (p + 1) / 33
        };
        let n = *base
            .iter()
            .find(|&&n| used & (1 << n) == 0 && (previous & (1 << n) == 0 || repeats < cap))
            .unwrap();
        used |= 1 << n;
        repeats += usize::from(previous & (1 << n) != 0);
        n
    })
}
fn variants(
    ps: &[[f64; 33]; 7],
    base: &[usize; 33],
    previous: u64,
    raw: &[f64; 33],
) -> Vec<[usize; 33]> {
    let score = select::fuse7(ps, &[1. / 7.; 7]);
    let mut v = vec![*base, select::rank_desc(&score)];
    for p in ps {
        v.push(select::rank_desc(p));
    }
    let wscore = std::array::from_fn(|n| {
        let rank = base.iter().position(|&i| i == n).unwrap();
        (33 - rank) as f64
    });
    v.push(quotas(base, previous, true));
    v.push(quotas(base, previous, false));
    v.push(ranked(&ssq::ranking::scores(&wscore, previous), base));
    for a in 0..7 {
        for b in a + 1..7 {
            v.push(ranked(
                &std::array::from_fn(|i| (ps[a][i] + ps[b][i]) / 2.),
                base,
            ));
        }
    }
    for t in [0., 0.5, 2., -1.] {
        let s = std::array::from_fn(|i| {
            if t == 0. {
                (ps.iter().map(|p| p[i].max(1e-12).ln()).sum::<f64>() / 7.).exp()
            } else {
                (ps.iter().map(|p| p[i].max(1e-12).powf(t)).sum::<f64>() / 7.).powf(1. / t)
            }
        });
        v.push(ranked(&s, base));
    }
    for q in 1..=5 {
        let s = std::array::from_fn(|i| {
            ps.iter()
                .map(|p| {
                    let o = select::rank_desc(p);
                    (34 - o.iter().position(|&n| n == i).unwrap()) as f64
                })
                .sum::<f64>()
                / 7.
        });
        let o = ranked(&s, base);
        let mut group = Vec::new();
        group.extend(
            base.iter()
                .copied()
                .filter(|n| previous & (1 << n) != 0)
                .take(q),
        );
        group.extend(
            o.iter()
                .copied()
                .filter(|n| previous & (1 << n) == 0)
                .take(16 - q),
        );
        group.sort_by_key(|n| base.iter().position(|i| i == n).unwrap());
        group.extend(
            base.iter()
                .copied()
                .filter(|n| !group_contains(previous, *n, base, &o, q)),
        );
        v.push(group.try_into().unwrap());
    }
    for alpha in [0.25, 0.5, 0.75] {
        v.push(select::rank_desc(&std::array::from_fn(|i| {
            (1. - alpha) * raw[i] + (alpha / 2.) * ps[1][i] + (alpha / 2.) * ps[6][i]
        })));
    }
    v
}
fn group_contains(
    prev: u64,
    n: usize,
    base: &[usize; 33],
    outside: &[usize; 33],
    q: usize,
) -> bool {
    base.iter()
        .copied()
        .filter(|i| prev & (1 << i) != 0)
        .take(q)
        .chain(
            outside
                .iter()
                .copied()
                .filter(|i| prev & (1 << i) == 0)
                .take(16 - q),
        )
        .any(|i| i == n)
}
fn main() {
    let file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let d = draw::load_csv(std::path::Path::new(&file)).unwrap();
    let snap = model::roll_all(&d).unwrap();
    let (mut sums, mut n) = ([0.; 7], 0usize);
    let mut names = vec!["weighted".to_owned(), "equal".to_owned()];
    names.extend((0..7).map(|e| format!("expert{e}")));
    names.extend(["quota_ceil", "quota_floor", "rank_mass_calibrated"].map(str::to_owned));
    for a in 0..7 {
        for b in a + 1..7 {
            names.push(format!("pair_{a}_{b}"));
        }
    }
    names.extend(["geomean", "rootmean", "squaremean", "harmonic"].map(str::to_owned));
    names.extend((1..=5).map(|q| format!("quota{q}_consensus_outside")));
    names.extend(["shrink_recent25", "shrink_recent50", "shrink_recent75"].map(str::to_owned));
    let mut rows = Vec::new();
    for r in &snap.rows {
        if n >= 30 {
            let base = select::rank_desc(&select::fuse7(
                &r.expert_p,
                &select::softmax_shrink(&sums, n as f64),
            ));
            let t = d.iter().position(|d| d.issue == r.issue).unwrap();
            let o = variants(
                &r.expert_p,
                &base,
                d[t - 1].red_mask,
                &select::fuse7(&r.expert_p, &select::softmax_shrink(&sums, n as f64)),
            );
            assert_eq!(o.len(), names.len());
            rows.push(json!({"issue":r.issue,"hits":o.iter().map(|o|[12,14,15,16].map(|k|o[..k].iter().filter(|&&n|r.red_mask&(1<<n)!=0).count())).collect::<Vec<_>>()}));
        }
        if r.issue < 25111 {
            for e in 0..7 {
                sums[e] += select::rank_log_score(&r.expert_p[e], r.red_mask);
            }
            n += 1;
        }
    }
    let start = rows
        .iter()
        .position(|r| r["issue"] == Protocol::CONFIRM_START_ISSUE)
        .unwrap();
    let end = start + 156;
    let mut windows = serde_json::Map::new();
    for (name, a, b) in [
        ("development", 0, start),
        ("confirmation", start, end),
        ("tracking", end, rows.len()),
        ("recent60", rows.len() - 60, rows.len()),
        ("all", 0, rows.len()),
    ] {
        let counts: Vec<_> = (0..names.len())
            .map(|m| {
                [0, 1, 2, 3].map(|j| rows[a..b].iter().filter(|r| r["hits"][m][j] == 6).count())
            })
            .collect();
        windows.insert(name.into(), json!({"n":b-a,"full":counts}));
    }
    let mut summary:Vec<Value>=names.iter().enumerate().map(|(i,m)|json!({"model":m,"all":windows["all"]["full"][i],"confirm":windows["confirmation"]["full"][i],"recent60":windows["recent60"]["full"][i]})).collect();
    summary.sort_by_key(|r| std::cmp::Reverse(r["all"][3].as_u64().unwrap()));
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"exploratory_small_red_candidates","actual_recommendation":false,"automatic_promotion":false,"data_sha256":draw::data_sha256(&d),"source_sha256":ssq::hash::sha256_hex(concat!(include_str!("small_red_candidates.rs"),include_str!("../src/model.rs"),include_str!("../src/select.rs"),include_str!("../src/eval.rs"),include_str!("../src/ranking.rs")).as_bytes()),"red_sizes":[12,14,15,16],"candidate_count":names.len(),"models":names,"windows":windows,"summary":summary,"rows":rows})).unwrap());
}
