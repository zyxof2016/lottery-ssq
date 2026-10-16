use std::time::Instant;

use crate::draw::{data_sha256, Draw};
use crate::model::{roll_all, Row};
use crate::protocol::Protocol;
use crate::select::{
    aggressive_spec, build_tickets, equal7, forecast, fuse6, fuse7, hits_prefix, mask_min_votes,
    max_red_hits, mean_red_hits, mild_spec, rank_desc, rank_log_score, rescue, score_key,
    softmax_shrink, top_mask, votes, Forecast,
};
use crate::stats::{
    acf1, adjust_bh, adjust_holm, binom_cc_p, block_means, block_signflip_p, cohen_dz, cohen_h,
    eigenvalues_sym, emax_iid, expected_red_hits, hg_mean_var, hg_total_p, mean_sd, normal_ci,
    norm_ppf, ols_slope, participation_ratio, pearson, percentile_sorted, wilson, z_test, Rng,
};

pub struct TopkPoint {
    pub model: String,
    pub window: &'static str,
    pub k: usize,
    pub mean_hits: f64,
    pub expected: f64,
    pub excess: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub band_lo: usize,
    pub band_hi: usize,
    pub iiv: f64,
    pub iiv_low: f64,
    pub iiv_high: f64,
    pub iiv_pos_frac: f64,
    pub iiv_slope: Option<f64>,
    pub n: usize,
    pub p_raw: f64,
    /// 确认窗 Top-K 曲线内部的 Holm 校正。开发窗不进入这族。
    pub p_curve: Option<f64>,
}

pub struct Stability {
    pub model: String,
    pub window: &'static str,
    pub mean_excess: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub pos_frac: f64,
    pub block_sd: f64,
    pub slope: Option<f64>,
    pub slope_se: Option<f64>,
    pub acf1: f64,
    pub half_delta: f64,
    pub block_min: f64,
    pub block_max: f64,
    pub n_blocks: usize,
}

pub struct AblationRow {
    pub name: String,
    pub confirm_excess: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub confirm_hits: f64,
    pub dev_excess: f64,
}

pub struct Funnel {
    pub stage_names: Vec<&'static str>,
    pub mean_alive: Vec<f64>,
    pub pooled_conditional: Vec<f64>,
    pub death_hist: Vec<f64>,
    pub expert_union_recall: f64,
    pub majority_recall: f64,
    pub fusion15_recall: f64,
    pub rescued_true: f64,
    pub displaced_true: f64,
    pub pool_recall: f64,
    pub ticket_recall: f64,
    pub evidence_recall: f64,
    pub evidence_keep: f64,
    pub mean_tickets: f64,
    pub mean_best_hits: f64,
    pub mean_cond_null: f64,
    pub mean_iid_null: f64,
    pub blue_top1: f64,
    pub blue_top3: f64,
    pub blue_ticket_hit: f64,
    pub blue_ticket_null: f64,
}

pub struct TradeCell {
    pub mass: f64,
    pub pool: usize,
    pub tickets: usize,
    pub overlap: u32,
    pub confirm_evidence: f64,
    pub confirm_coverage: f64,
    pub confirm_pairs: f64,
    pub confirm_best: f64,
    pub confirm_avg: f64,
    pub confirm_lift: f64,
    pub confirm_fill: f64,
    pub repl_evidence: f64,
    pub repl_coverage: f64,
    pub repl_lift: f64,
    pub pareto: bool,
}

pub struct BlueRow {
    pub name: String,
    pub hit1: f64,
    pub wilson_low: f64,
    pub wilson_high: f64,
    pub hit3: f64,
    pub logloss: f64,
    pub matched_logloss: f64,
    pub uniform_logloss: f64,
    pub brier: f64,
    pub matched_brier: f64,
    pub uniform_brier: f64,
    pub cohen_h: f64,
}

pub struct LooRow {
    pub dropped: String,
    pub excess: f64,
}

pub struct Robust {
    pub weights: [f64; 7],
    pub blue_weights: [f64; 6],
    pub best_expert: usize,
    pub loo: Vec<LooRow>,
    pub dirichlet_p05: f64,
    pub dirichlet_p50: f64,
    pub dirichlet_p95: f64,
    pub frozen_excess: f64,
    pub l1_mean: f64,
    pub l1_max: f64,
    pub corr_score: [[f64; 7]; 7],
    pub corr_hit: [[f64; 7]; 7],
    pub eigenvalues: [f64; 7],
    pub n_eff: f64,
    pub n_eff_eq: Option<f64>,
    pub mean_corr: f64,
    pub mean_abs_corr: f64,
}

pub struct TestResult {
    pub name: String,
    pub estimate: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub ci_bonf_low: f64,
    pub ci_bonf_high: f64,
    pub effect: f64,
    pub effect_kind: &'static str,
    pub p: f64,
    pub p_holm: f64,
    pub p_bh: f64,
    pub p_block: Option<f64>,
    pub n: usize,
    pub better_if_positive: bool,
    pub note: String,
}

pub struct Backtest {
    pub protocol_sha: String,
    pub data_sha: String,
    pub n: usize,
    pub first_issue: u32,
    pub last_issue: u32,
    pub first_date: i32,
    pub last_date: i32,
    pub confirm_issue_start: u32,
    pub confirm_issue_end: u32,
    pub weight_issue_end: u32,
    pub n_weight: usize,
    pub n_confirm: usize,
    pub n_repl: usize,
    pub n_track: usize,
    pub track_issue_start: Option<u32>,
    pub track_issue_end: Option<u32>,
    pub track_fusion_excess: Option<f64>,
    pub track_mild_excess: Option<f64>,
    pub elapsed_ms: u128,
    pub mde_red: f64,
    pub mde_red_bonf: f64,
    pub mde_blue: f64,
    pub mde_blue_bonf: f64,
    pub hg_sd: f64,
    pub topk: Vec<TopkPoint>,
    pub stability: Vec<Stability>,
    pub ablation: Vec<AblationRow>,
    pub funnel: Funnel,
    pub tradeoff: Vec<TradeCell>,
    pub blue: Vec<BlueRow>,
    pub robust: Robust,
    pub tests: Vec<TestResult>,
}

pub struct Prepared {
    pub ranking: Option<serde_json::Value>,
    pub weights: [f64; 7],
    pub blue_weights: [f64; 6],
    pub expert_red: [[f64; 33]; 7],
    pub expert_blue: [[f64; 16]; 6],
    pub forecast: Forecast,
    pub n: usize,
    pub cutoff_issue: u32,
    pub cutoff_date: i32,
    pub data_sha: String,
}

struct HitPack {
    expert: [Vec<f64>; 7],
    fusion: Vec<f64>,
    equal: Vec<f64>,
    mild: Vec<f64>,
    agg: Vec<f64>,
}

struct CellAcc {
    n: f64,
    evidence: f64,
    coverage: f64,
    pairs: f64,
    best: f64,
    avg: f64,
    lift: f64,
    fill: f64,
}

impl CellAcc {
    fn new() -> Self {
        Self {
            n: 0.0,
            evidence: 0.0,
            coverage: 0.0,
            pairs: 0.0,
            best: 0.0,
            avg: 0.0,
            lift: 0.0,
            fill: 0.0,
        }
    }
}

struct Spec {
    mass: f64,
    pool: usize,
    tickets: usize,
    overlap: u32,
}

enum CiKind {
    Normal { se: f64 },
    Wilson { hits: f64, n: f64, p0: f64 },
}

struct Draft {
    name: String,
    estimate: f64,
    effect: f64,
    effect_kind: &'static str,
    p: f64,
    n: usize,
    better_if_positive: bool,
    p_block: Option<f64>,
    note: String,
    ci: CiKind,
}

pub fn prepare_forecast(draws: &[Draw]) -> Result<Prepared, String> {
    let snap = roll_all(draws)?;
    let weight_rows = weight_rows_for_publish(&snap.rows)?;
    let weights = fit_red(weight_rows);
    let blue_weights = fit_blue(weight_rows);
    let forecast = forecast(
        &snap.next_red,
        &snap.next_blue,
        &weights,
        &blue_weights,
    );
    let last = draws.last().ok_or("没有开奖数据")?;
    Ok(Prepared {
        ranking: None,
        weights,
        blue_weights,
        expert_red: snap.next_red,
        expert_blue: snap.next_blue,
        forecast,
        n: draws.len(),
        cutoff_issue: last.issue,
        cutoff_date: last.date,
        data_sha: data_sha256(draws),
    })
}

pub fn backtest(draws: &[Draw]) -> Result<Backtest, String> {
    backtest_span(
        draws,
        Protocol::CONFIRM_START_ISSUE,
        Protocol::CONFIRMATION_LEN,
    )
}

pub fn backtest_span(
    draws: &[Draw],
    confirm_start_issue: u32,
    confirm_len: usize,
) -> Result<Backtest, String> {
    let started = Instant::now();
    let snap = roll_all(draws)?;
    let (weight_end, confirm_end) = locate_span(&snap.rows, confirm_start_issue, confirm_len)?;
    let n_weight = weight_end;
    let n_confirm = confirm_end - weight_end;
    let weight_rows = &snap.rows[..weight_end];
    let confirm_rows = &snap.rows[weight_end..confirm_end];
    let track_rows = &snap.rows[confirm_end..];
    let n_repl = confirm_len.min(n_weight);
    let repl_rows = &snap.rows[n_weight - n_repl..n_weight];

    let weights = fit_red(weight_rows);
    let blue_weights = fit_blue(weight_rows);
    let best_expert = best_expert_index(weight_rows);

    let z95 = norm_ppf(0.975);
    let mut topk = Vec::new();
    let mut stability = Vec::new();
    for (e, name) in Protocol::EXPERTS_ZH.iter().enumerate() {
        push_model_curves(
            &mut topk,
            &mut stability,
            name,
            &expert_prefixes(weight_rows, e),
            &expert_prefixes(confirm_rows, e),
            z95,
        );
    }
    push_model_curves(
        &mut topk,
        &mut stability,
        "加权融合",
        &fusion_prefixes(weight_rows, &weights),
        &fusion_prefixes(confirm_rows, &weights),
        z95,
    );
    adjust_curve(&mut topk);

    let dev_hits = collect_hits(weight_rows, &weights);
    let confirm_hits = collect_hits(confirm_rows, &weights);
    let ablation = ablation_table(&dev_hits, &confirm_hits, z95);

    let (funnel, ticket_excess, ticket_p, cond_null, iid_null) =
        funnel_and_tickets(confirm_rows, &weights, &blue_weights);

    let specs = grid_specs();
    let confirm_grid = eval_grid(confirm_rows, &weights, &specs);
    let repl_grid = eval_grid(repl_rows, &weights, &specs);
    let tradeoff = trade_cells(&specs, &confirm_grid, &repl_grid);

    let blue = blue_table(confirm_rows, &blue_weights, z95);
    let robust = robustness(
        &snap.rows,
        confirm_rows,
        n_weight,
        &weights,
        &blue_weights,
        best_expert,
        &confirm_hits,
    );

    let tests = assemble_tests(
        &confirm_hits,
        confirm_rows,
        &blue_weights,
        best_expert,
        &ticket_excess,
        ticket_p,
        cond_null,
        iid_null,
    );

    let m = tests.len() as f64;
    let z_b = norm_ppf(1.0 - 0.025 / m);
    let z80 = norm_ppf(0.8);
    let hg_sd = hg_mean_var(Protocol::PRIMARY_K).1.sqrt();
    let n_c = n_confirm as f64;
    let mde_red = (z95 + z80) * hg_sd / n_c.sqrt();
    let mde_red_bonf = (z_b + z80) * hg_sd / n_c.sqrt();
    let p0: f64 = 1.0 / 16.0;
    let blue_sd = (p0 * (1.0 - p0)).sqrt();
    let mde_blue = (z95 + z80) * blue_sd / n_c.sqrt();
    let mde_blue_bonf = (z_b + z80) * blue_sd / n_c.sqrt();
    let (n_track, track_issue_start, track_issue_end, track_fusion_excess, track_mild_excess) =
        if track_rows.is_empty() {
            (0, None, None, None, None)
        } else {
            let pack = collect_hits(track_rows, &weights);
            (
                track_rows.len(),
                Some(track_rows[0].issue),
                Some(track_rows.last().unwrap().issue),
                Some(mean_of(&excess_from_hits(&pack.fusion))),
                Some(mean_of(&excess_from_hits(&pack.mild))),
            )
        };

    Ok(Backtest {
        protocol_sha: Protocol::sha256(),
        data_sha: data_sha256(draws),
        n: draws.len(),
        first_issue: draws[0].issue,
        last_issue: draws.last().unwrap().issue,
        first_date: draws[0].date,
        last_date: draws.last().unwrap().date,
        confirm_issue_start: confirm_rows[0].issue,
        confirm_issue_end: confirm_rows.last().unwrap().issue,
        weight_issue_end: weight_rows.last().unwrap().issue,
        n_weight,
        n_confirm,
        n_repl,
        n_track,
        track_issue_start,
        track_issue_end,
        track_fusion_excess,
        track_mild_excess,
        elapsed_ms: started.elapsed().as_millis(),
        mde_red,
        mde_red_bonf,
        mde_blue,
        mde_blue_bonf,
        hg_sd,
        topk,
        stability,
        ablation,
        funnel,
        tradeoff,
        blue,
        robust,
        tests,
    })
}

fn fit_red(rows: &[Row]) -> [f64; 7] {
    let mut sum = [0.0; 7];
    for row in rows {
        for e in 0..7 {
            sum[e] += rank_log_score(&row.expert_p[e], row.red_mask);
        }
    }
    softmax_shrink(&sum, rows.len() as f64)
}

fn fit_blue(rows: &[Row]) -> [f64; 6] {
    let mut sum = [0.0; 6];
    for row in rows {
        let y = row.blue as usize;
        for e in 0..6 {
            sum[e] += row.blue_p[e][y].clamp(1e-12, 1.0).ln();
        }
    }
    softmax_shrink(&sum, rows.len() as f64)
}

fn best_expert_index(rows: &[Row]) -> usize {
    let mut best = 0usize;
    let mut best_ex = f64::NEG_INFINITY;
    for e in 0..7 {
        let ex = mean_of(&excess_from_hits(&expert_hits(rows, e, Protocol::PRIMARY_K)));
        if ex > best_ex {
            best_ex = ex;
            best = e;
        }
    }
    best
}

fn expert_hits(rows: &[Row], expert: usize, k: usize) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            let order = rank_desc(&row.expert_p[expert]);
            let mut h = 0.0;
            for i in 0..k {
                if (row.red_mask >> order[i]) & 1 == 1 {
                    h += 1.0;
                }
            }
            h
        })
        .collect()
}

fn expert_prefixes(rows: &[Row], expert: usize) -> Vec<[u32; 8]> {
    rows.iter()
        .map(|row| {
            let order = rank_desc(&row.expert_p[expert]);
            hits_prefix(&order, row.red_mask)
        })
        .collect()
}

fn fusion_prefixes(rows: &[Row], w: &[f64; 7]) -> Vec<[u32; 8]> {
    rows.iter()
        .map(|row| {
            let fusion = fuse7(&row.expert_p, w);
            let order = rank_desc(&fusion);
            hits_prefix(&order, row.red_mask)
        })
        .collect()
}

fn push_model_curves(
    topk: &mut Vec<TopkPoint>,
    stability_out: &mut Vec<Stability>,
    name: &str,
    dev: &[[u32; 8]],
    confirm: &[[u32; 8]],
    z: f64,
) {
    topk.extend(curves(name, "开发窗", dev, z));
    topk.extend(curves(name, "确认窗", confirm, z));
    let pk = Protocol::KS.iter().position(|&k| k == Protocol::PRIMARY_K).unwrap();
    stability_out.push(stability_of(name, "开发窗", &excess_at(dev, pk), z));
    stability_out.push(stability_of(name, "确认窗", &excess_at(confirm, pk), z));
}

fn excess_at(prefixes: &[[u32; 8]], ki: usize) -> Vec<f64> {
    let expected = expected_red_hits(Protocol::KS[ki]);
    prefixes
        .iter()
        .map(|p| p[ki] as f64 - expected)
        .collect()
}

fn curves(model: &str, window: &'static str, prefixes: &[[u32; 8]], z: f64) -> Vec<TopkPoint> {
    let ks = Protocol::KS;
    let mut prev: Option<Vec<f64>> = None;
    let mut out = Vec::new();
    let n_draws = prefixes.len();
    let n = n_draws as f64;
    for (ki, &k) in ks.iter().enumerate() {
        let hits: Vec<f64> = prefixes.iter().map(|p| p[ki] as f64).collect();
        let expected = expected_red_hits(k);
        let excess: Vec<f64> = hits.iter().map(|h| h - expected).collect();
        let (mean_ex, _, se) = mean_sd(&excess);
        let (lo, hi) = normal_ci(mean_ex, se, z);
        let mean_hits = hits.iter().sum::<f64>() / n;
        let band_lo = if ki == 0 { 0 } else { ks[ki - 1] };
        let width = k - band_lo;
        let band_expected = 6.0 * width as f64 / 33.0;
        let iiv: Vec<f64> = if let Some(prev_hits) = &prev {
            hits.iter()
                .zip(prev_hits.iter())
                .map(|(h, p)| (h - p) - band_expected)
                .collect()
        } else {
            excess.clone()
        };
        let (iiv_mean, _, iiv_se) = mean_sd(&iiv);
        let (iiv_lo, iiv_hi) = normal_ci(iiv_mean, iiv_se, z);
        let blocks = block_means(&iiv, Protocol::STABILITY_BLOCK);
        let pos = positive_frac(&blocks);
        let slope = ols_slope(&blocks).map(|(s, _)| s);
        out.push(TopkPoint {
            model: model.to_string(),
            window,
            k,
            mean_hits,
            expected,
            excess: mean_ex,
            ci_low: lo,
            ci_high: hi,
            band_lo,
            band_hi: k,
            iiv: iiv_mean,
            iiv_low: iiv_lo,
            iiv_high: iiv_hi,
            iiv_pos_frac: pos,
            iiv_slope: slope,
            n: n_draws,
            p_raw: if n_draws == 0 {
                1.0
            } else {
                hg_total_p(hits.iter().sum(), k, n_draws)
            },
            p_curve: None,
        });
        prev = Some(hits);
    }
    out
}

fn stability_of(model: &str, window: &'static str, excess: &[f64], z: f64) -> Stability {
    let (mean, _, se) = mean_sd(excess);
    let (lo, hi) = normal_ci(mean, se, z);
    let blocks = block_means(excess, Protocol::STABILITY_BLOCK);
    let (slope, slope_se) = match ols_slope(&blocks) {
        Some((s, se_s)) => (Some(s), Some(se_s)),
        None => (None, None),
    };
    let (block_min, block_max) = if blocks.is_empty() {
        (0.0, 0.0)
    } else {
        (
            blocks.iter().copied().fold(f64::INFINITY, f64::min),
            blocks.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let (_, block_sd, _) = mean_sd(&blocks);
    let half_delta = half_diff(&blocks);
    Stability {
        model: model.to_string(),
        window,
        mean_excess: mean,
        ci_low: lo,
        ci_high: hi,
        pos_frac: positive_frac(&blocks),
        block_sd,
        slope,
        slope_se,
        acf1: acf1(excess),
        half_delta,
        block_min,
        block_max,
        n_blocks: blocks.len(),
    }
}

fn positive_frac(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().filter(|v| **v > 0.0).count() as f64 / xs.len() as f64
    }
}

fn half_diff(blocks: &[f64]) -> f64 {
    if blocks.len() < 2 {
        return 0.0;
    }
    let mid = blocks.len() / 2;
    mean_of(&blocks[mid..]) - mean_of(&blocks[..mid])
}

fn mean_of(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn collect_hits(rows: &[Row], w: &[f64; 7]) -> HitPack {
    let mut pack = HitPack {
        expert: std::array::from_fn(|_| Vec::with_capacity(rows.len())),
        fusion: Vec::with_capacity(rows.len()),
        equal: Vec::with_capacity(rows.len()),
        mild: Vec::with_capacity(rows.len()),
        agg: Vec::with_capacity(rows.len()),
    };
    let eq = equal7();
    let mild = mild_spec();
    let agg = aggressive_spec();
    for row in rows {
        let mut orders = [[0usize; 33]; 7];
        for e in 0..7 {
            orders[e] = rank_desc(&row.expert_p[e]);
            let mut h = 0.0;
            for i in 0..Protocol::PRIMARY_K {
                if (row.red_mask >> orders[e][i]) & 1 == 1 {
                    h += 1.0;
                }
            }
            pack.expert[e].push(h);
        }
        let fusion = fuse7(&row.expert_p, w);
        let order = rank_desc(&fusion);
        pack.fusion.push(count_top(&order, row.red_mask, Protocol::PRIMARY_K));
        let fused_eq = fuse7(&row.expert_p, &eq);
        let order_eq = rank_desc(&fused_eq);
        pack.equal.push(count_top(&order_eq, row.red_mask, Protocol::PRIMARY_K));
        let vote_m = votes(&orders, Protocol::RESCUE_M);
        let vote_a = votes(&orders, Protocol::RESCUE_M_AGG);
        let mild_set = rescue(&order, &fusion, &vote_m, &mild);
        let agg_set = rescue(&order, &fusion, &vote_a, &agg);
        pack.mild.push((mild_set.mask & row.red_mask).count_ones() as f64);
        pack.agg.push((agg_set.mask & row.red_mask).count_ones() as f64);
    }
    pack
}

fn count_top(order: &[usize; 33], mask: u64, k: usize) -> f64 {
    let mut h = 0.0;
    for i in 0..k {
        if (mask >> order[i]) & 1 == 1 {
            h += 1.0;
        }
    }
    h
}

fn ablation_table(dev: &HitPack, confirm: &HitPack, z: f64) -> Vec<AblationRow> {
    let mut rows = Vec::new();
    for e in 0..7 {
        rows.push(ablation_one(
            Protocol::EXPERTS_ZH[e],
            &dev.expert[e],
            &confirm.expert[e],
            z,
        ));
    }
    rows.push(ablation_one("等权融合", &dev.equal, &confirm.equal, z));
    rows.push(ablation_one("加权融合", &dev.fusion, &confirm.fusion, z));
    rows.push(ablation_one("温和救援", &dev.mild, &confirm.mild, z));
    rows.push(ablation_one("强救援", &dev.agg, &confirm.agg, z));
    rows
}

fn ablation_one(name: &str, dev: &[f64], confirm: &[f64], z: f64) -> AblationRow {
    let confirm_ex = excess_from_hits(confirm);
    let (mean, _, se) = mean_sd(&confirm_ex);
    let (lo, hi) = normal_ci(mean, se, z);
    AblationRow {
        name: name.to_string(),
        confirm_excess: mean,
        ci_low: lo,
        ci_high: hi,
        confirm_hits: mean_of(confirm),
        dev_excess: mean_of(&excess_from_hits(dev)),
    }
}

fn excess_from_hits(hits: &[f64]) -> Vec<f64> {
    let expected = expected_red_hits(Protocol::PRIMARY_K);
    hits.iter().map(|h| h - expected).collect()
}

fn funnel_and_tickets(
    rows: &[Row],
    weights: &[f64; 7],
    blue_weights: &[f64; 6],
) -> (Funnel, Vec<f64>, f64, f64, f64) {
    let n = rows.len();
    let mut alive = [0u64; 5];
    let mut death = [0u64; 6];
    let mut union_true = 0u64;
    let mut majority_true = 0u64;
    let mut fusion15_true = 0u64;
    let mut rescued_true = 0u64;
    let mut displaced_true = 0u64;
    let mut pool_true = 0u64;
    let mut ticket_true = 0u64;
    let mut evidence_true = 0u64;
    let mut evidence_keep = 0u64;
    let mut ticket_n = 0u64;
    let mut best_sum = 0u64;
    let mut blue_top1 = 0u64;
    let mut blue_top3 = 0u64;
    let mut blue_any = 0u64;
    let mut blue_null_sum = 0.0;
    let mut tickets_per: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut obs_best: Vec<u32> = Vec::with_capacity(n);
    let mut iid_sum = 0.0;

    for row in rows {
        let fc = forecast(&row.expert_p, &row.blue_p, weights, blue_weights);
        let mut orders = [[0usize; 33]; 7];
        for e in 0..7 {
            orders[e] = rank_desc(&row.expert_p[e]);
        }
        let vote_nom = votes(&orders, Protocol::NOMINATION_M);
        let expert_union = mask_min_votes(&vote_nom, 1);
        let majority = mask_min_votes(&vote_nom, Protocol::MAJORITY_VOTES as u8);
        let fusion20 = top_mask(&fc.red_order, Protocol::FUSION_WIDE_K);
        let fusion15 = top_mask(&fc.red_order, Protocol::PRIMARY_K);
        let mut pool_mask = 0u64;
        for &i in &fc.pool {
            pool_mask |= 1u64 << i;
        }
        let mut ticket_mask = 0u64;
        for t in &fc.tickets {
            ticket_mask |= *t;
        }
        let stages = [expert_union | fusion20, fusion20, fc.mild.mask, pool_mask, ticket_mask];
        debug_assert!(subset(stages[4], stages[3]));
        debug_assert!(subset(stages[3], stages[2]));
        debug_assert!(subset(stages[2], stages[1]));
        debug_assert!(subset(stages[1], stages[0]));

        let win = row.red_mask;
        for (i, stage) in stages.iter().enumerate() {
            alive[i] += (stage & win).count_ones() as u64;
        }
        let mut bits = win;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            let bit = 1u64 << i;
            death[deepest(bit, &stages)] += 1;
            bits &= bits - 1;
        }
        union_true += (expert_union & win).count_ones() as u64;
        majority_true += (majority & win).count_ones() as u64;
        fusion15_true += (fusion15 & win).count_ones() as u64;
        for &i in &fc.mild.rescued {
            if (win >> i) & 1 == 1 {
                rescued_true += 1;
            }
        }
        for &i in &fc.mild.displaced {
            if (win >> i) & 1 == 1 {
                displaced_true += 1;
            }
        }
        pool_true += (pool_mask & win).count_ones() as u64;
        ticket_true += (ticket_mask & win).count_ones() as u64;

        for &i in &take_mass(&fc.mild.members, &fc.fusion, Protocol::EVIDENCE_MASS) {
            evidence_keep += 1;
            if (win >> i) & 1 == 1 {
                evidence_true += 1;
            }
        }
        ticket_n += fc.tickets.len() as u64;
        let best = max_red_hits(&fc.tickets, win);
        best_sum += best as u64;
        obs_best.push(best);
        tickets_per.push(fc.tickets.clone());
        iid_sum += emax_iid(fc.tickets.len());

        if fc.blue_order[0] == row.blue as usize {
            blue_top1 += 1;
        }
        let mut better = 0usize;
        let y = row.blue as usize;
        for i in 0..16 {
            if i != y && (fc.blue[i] > fc.blue[y] || (fc.blue[i] == fc.blue[y] && i < y)) {
                better += 1;
            }
        }
        if better < 3 {
            blue_top3 += 1;
        }
        let mut distinct = 0u64;
        let mut seen = [false; 16];
        let mut hit_any = false;
        for &b in &fc.ticket_blues {
            if !seen[b] {
                seen[b] = true;
                distinct += 1;
            }
            if b == y {
                hit_any = true;
            }
        }
        if hit_any {
            blue_any += 1;
        }
        blue_null_sum += if fc.ticket_blues.is_empty() {
            0.0
        } else {
            distinct as f64 / 16.0
        };
    }

    let nf = n as f64;
    let (excess, p_mc, cond_null) = ticket_null(&tickets_per, &obs_best);
    let mean_alive: Vec<f64> = alive.iter().map(|c| *c as f64 / nf).collect();
    let mut pooled = Vec::with_capacity(5);
    pooled.push(mean_alive[0] / 6.0);
    for i in 1..5 {
        let prev = alive[i - 1] as f64;
        pooled.push(if prev > 0.0 {
            alive[i] as f64 / prev
        } else {
            f64::NAN
        });
    }
    let death_hist = death.iter().map(|c| *c as f64 / nf).collect();
    let funnel = Funnel {
        stage_names: vec!["系统可见", "融合宽入围", "救援定稿", "出票池", "票组覆盖"],
        mean_alive,
        pooled_conditional: pooled,
        death_hist,
        expert_union_recall: union_true as f64 / nf,
        majority_recall: majority_true as f64 / nf,
        fusion15_recall: fusion15_true as f64 / nf,
        rescued_true: rescued_true as f64 / nf,
        displaced_true: displaced_true as f64 / nf,
        pool_recall: pool_true as f64 / nf,
        ticket_recall: ticket_true as f64 / nf,
        evidence_recall: evidence_true as f64 / nf,
        evidence_keep: evidence_keep as f64 / nf,
        mean_tickets: ticket_n as f64 / nf,
        mean_best_hits: best_sum as f64 / nf,
        mean_cond_null: cond_null,
        mean_iid_null: iid_sum / nf,
        blue_top1: blue_top1 as f64 / nf,
        blue_top3: blue_top3 as f64 / nf,
        blue_ticket_hit: blue_any as f64 / nf,
        blue_ticket_null: blue_null_sum / nf,
    };
    (funnel, excess, p_mc, cond_null, iid_sum / nf)
}

fn subset(inner: u64, outer: u64) -> bool {
    inner & outer == inner
}

fn deepest(bit: u64, stages: &[u64; 5]) -> usize {
    let mut d = 0;
    for (i, stage) in stages.iter().enumerate() {
        if stage & bit != 0 {
            d = i + 1;
        } else {
            break;
        }
    }
    d
}

fn ticket_null(tickets: &[Vec<u64>], obs: &[u32]) -> (Vec<f64>, f64, f64) {
    let b = Protocol::TICKET_MC_B;
    let mut rng = Rng::new(Protocol::SEED ^ 0x71C4_E7);
    let mut sim_totals = vec![0u32; b];
    let mut draw_sum = vec![0u64; tickets.len()];
    for sim in &mut sim_totals {
        let mut total = 0u32;
        for (d, set) in tickets.iter().enumerate() {
            let win = rng.sample_red_mask();
            let h = max_red_hits(set, win);
            total += h;
            draw_sum[d] += h as u64;
        }
        *sim = total;
    }
    let mean_sim = sim_totals.iter().map(|t| *t as f64).sum::<f64>() / b as f64;
    let obs_total: u32 = obs.iter().copied().sum();
    let dev = (obs_total as f64 - mean_sim).abs();
    let extreme = sim_totals
        .iter()
        .filter(|t| (**t as f64 - mean_sim).abs() + 1e-9 >= dev)
        .count();
    let p = (extreme as f64 + 1.0) / (b as f64 + 1.0);
    let excess: Vec<f64> = obs
        .iter()
        .enumerate()
        .map(|(d, h)| *h as f64 - draw_sum[d] as f64 / b as f64)
        .collect();
    let cond = mean_of(
        &draw_sum
            .iter()
            .map(|s| *s as f64 / b as f64)
            .collect::<Vec<_>>(),
    );
    (excess, p, cond)
}

fn grid_specs() -> Vec<Spec> {
    let mut v = Vec::new();
    for &mass in &[1.0_f64, 0.70, 0.50] {
        for &pool in &[10usize, 12, 15] {
            for &tickets in &[1usize, 8, 20] {
                v.push(Spec {
                    mass,
                    pool,
                    tickets,
                    overlap: 4,
                });
            }
        }
    }
    v.push(Spec {
        mass: 1.0,
        pool: 12,
        tickets: 8,
        overlap: 3,
    });
    v.push(Spec {
        mass: 1.0,
        pool: 12,
        tickets: 8,
        overlap: 5,
    });
    v
}

fn eval_grid(rows: &[Row], w: &[f64; 7], specs: &[Spec]) -> Vec<CellAcc> {
    let mut acc = Vec::new();
    for _ in specs {
        acc.push(CellAcc::new());
    }
    for row in rows {
        let fusion = fuse7(&row.expert_p, w);
        let order = rank_desc(&fusion);
        for (s, spec) in specs.iter().enumerate() {
            let cand = candidates(&order, &fusion, spec.mass, spec.pool);
            let tickets = build_tickets(&cand, &fusion, spec.tickets, spec.overlap);
            let metrics = coverage_of(&order, &fusion, &tickets, row.red_mask);
            let lift = max_red_hits(&tickets, row.red_mask) as f64 - emax_iid(spec.tickets);
            let cell = &mut acc[s];
            cell.n += 1.0;
            cell.evidence += metrics.0;
            cell.coverage += metrics.1;
            cell.pairs += metrics.2;
            cell.best += max_red_hits(&tickets, row.red_mask) as f64;
            cell.avg += mean_red_hits(&tickets, row.red_mask);
            cell.lift += lift;
            cell.fill += if spec.tickets == 0 {
                0.0
            } else {
                tickets.len() as f64 / spec.tickets as f64
            };
        }
    }
    acc
}

fn candidates(order: &[usize; 33], scores: &[f64; 33], mass: f64, pool: usize) -> Vec<usize> {
    let mut kept = take_mass(&order[..Protocol::PRIMARY_K], scores, mass);
    if kept.len() > pool {
        kept.truncate(pool);
    }
    kept
}

fn take_mass(members: &[usize], scores: &[f64; 33], mass: f64) -> Vec<usize> {
    let mut order = members.to_vec();
    order.sort_by(|&a, &b| score_key(scores[b]).cmp(&score_key(scores[a])).then(a.cmp(&b)));
    if order.is_empty() || mass >= 1.0 - 1e-12 {
        return order;
    }
    let total: f64 = order.iter().map(|&i| scores[i]).sum();
    let mut kept = Vec::new();
    let mut acc = 0.0;
    for &i in &order {
        kept.push(i);
        acc += scores[i];
        if total <= 0.0 || acc + 1e-15 >= mass * total {
            break;
        }
    }
    kept
}

fn coverage_of(order: &[usize; 33], scores: &[f64; 33], tickets: &[u64], _win: u64) -> (f64, f64, f64) {
    let u = &order[..Protocol::PRIMARY_K];
    let mass_u: f64 = u.iter().map(|&i| scores[i]).sum();
    let mut covered = 0u64;
    for t in tickets {
        covered |= *t;
    }
    let mut mass_c = 0.0;
    let mut n_cov = 0.0;
    for &i in u {
        if (covered >> i) & 1 == 1 {
            mass_c += scores[i];
            n_cov += 1.0;
        }
    }
    let evidence = if mass_u > 0.0 { mass_c / mass_u } else { 0.0 };
    let coverage = n_cov / u.len() as f64;
    let mut hit_pairs = 0.0;
    let mut total_pairs = 0.0;
    for a in 0..u.len() {
        for b in (a + 1)..u.len() {
            total_pairs += 1.0;
            if (covered >> u[a]) & 1 == 1 && (covered >> u[b]) & 1 == 1 {
                hit_pairs += 1.0;
            }
        }
    }
    let pairs = if total_pairs > 0.0 {
        hit_pairs / total_pairs
    } else {
        0.0
    };
    (evidence, coverage, pairs)
}

fn trade_cells(specs: &[Spec], confirm: &[CellAcc], repl: &[CellAcc]) -> Vec<TradeCell> {
    let mut cells = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let c = &confirm[i];
        let r = &repl[i];
        let cn = c.n.max(1.0);
        let rn = r.n.max(1.0);
        cells.push(TradeCell {
            mass: spec.mass,
            pool: spec.pool,
            tickets: spec.tickets,
            overlap: spec.overlap,
            confirm_evidence: c.evidence / cn,
            confirm_coverage: c.coverage / cn,
            confirm_pairs: c.pairs / cn,
            confirm_best: c.best / cn,
            confirm_avg: c.avg / cn,
            confirm_lift: c.lift / cn,
            confirm_fill: c.fill / cn,
            repl_evidence: r.evidence / rn,
            repl_coverage: r.coverage / rn,
            repl_lift: r.lift / rn,
            pareto: false,
        });
    }
    for i in 0..cells.len() {
        let dominated = cells.iter().enumerate().any(|(j, other)| {
            if i == j {
                return false;
            }
            other.confirm_evidence + 1e-12 >= cells[i].confirm_evidence
                && other.confirm_coverage + 1e-12 >= cells[i].confirm_coverage
                && other.confirm_avg + 1e-12 >= cells[i].confirm_avg
                && (other.confirm_evidence > cells[i].confirm_evidence + 1e-12
                    || other.confirm_coverage > cells[i].confirm_coverage + 1e-12
                    || other.confirm_avg > cells[i].confirm_avg + 1e-12)
        });
        cells[i].pareto = !dominated;
    }
    cells
}

fn blue_table(rows: &[Row], blue_weights: &[f64; 6], z: f64) -> Vec<BlueRow> {
    let mut rows_out = Vec::new();
    for e in 0..6 {
        let series: Vec<[f64; 16]> = rows.iter().map(|r| r.blue_p[e]).collect();
        rows_out.push(blue_one(Protocol::BLUES_ZH[e], rows, &series, z));
    }
    let fused: Vec<[f64; 16]> = rows
        .iter()
        .map(|r| fuse6(&r.blue_p, blue_weights))
        .collect();
    rows_out.push(blue_one("加权融合", rows, &fused, z));
    rows_out
}

fn blue_one(name: &str, rows: &[Row], probs: &[[f64; 16]], z: f64) -> BlueRow {
    let n = rows.len() as f64;
    let mut hit1 = 0.0;
    let mut hit3 = 0.0;
    let mut ll = 0.0;
    let mut matched_ll = 0.0;
    let mut brier = 0.0;
    let mut matched_brier = 0.0;
    for (row, p) in rows.iter().zip(probs.iter()) {
        let y = row.blue as usize;
        let mut better = 0usize;
        for i in 0..16 {
            if i != y && (p[i] > p[y] || (p[i] == p[y] && i < y)) {
                better += 1;
            }
        }
        if better == 0 {
            hit1 += 1.0;
        }
        if better < 3 {
            hit3 += 1.0;
        }
        let py = p[y].clamp(1e-12, 1.0);
        ll += -py.ln();
        let mut mll = 0.0;
        let mut ss = 0.0;
        for i in 0..16 {
            let pi = p[i].clamp(1e-12, 1.0);
            mll += -pi.ln();
            ss += p[i] * p[i];
            let t = if i == y { 1.0 } else { 0.0 };
            let d = p[i] - t;
            brier += d * d;
        }
        matched_ll += mll / 16.0;
        matched_brier += ss + 1.0 - 2.0 / 16.0;
    }
    let (wlo, whi) = wilson(hit1, n, z);
    let rate = hit1 / n;
    BlueRow {
        name: name.to_string(),
        hit1: rate,
        wilson_low: wlo,
        wilson_high: whi,
        hit3: hit3 / n,
        logloss: ll / n,
        matched_logloss: matched_ll / n,
        uniform_logloss: 16.0_f64.ln(),
        brier: brier / n,
        matched_brier: matched_brier / n,
        uniform_brier: 240.0 / 256.0,
        cohen_h: cohen_h(rate, 1.0 / 16.0),
    }
}

fn robustness(
    all_rows: &[Row],
    confirm: &[Row],
    n_weight: usize,
    weights: &[f64; 7],
    blue_weights: &[f64; 6],
    best_expert: usize,
    confirm_hits: &HitPack,
) -> Robust {
    let frozen_excess = mean_of(&excess_from_hits(&confirm_hits.fusion));
    let mut loo = Vec::new();
    for drop in 0..7 {
        let mut w = *weights;
        w[drop] = 0.0;
        let s: f64 = w.iter().sum();
        for x in &mut w {
            *x /= s;
        }
        loo.push(LooRow {
            dropped: Protocol::EXPERTS_ZH[drop].to_string(),
            excess: fusion_excess(confirm, &w),
        });
    }

    let floor = Protocol::DIRICHLET_FLOOR;
    let keep = 1.0 - floor * 7.0;
    let mut floored = *weights;
    for w in &mut floored {
        *w = keep * *w + floor;
    }
    let alpha = floored.map(|w| Protocol::DIRICHLET_KAPPA * w);
    let mut rng = Rng::new(Protocol::SEED ^ 0xD1A1);
    let mut samples = Vec::with_capacity(Protocol::DIRICHLET_DRAWS);
    for _ in 0..Protocol::DIRICHLET_DRAWS {
        let w = rng.dirichlet(&alpha);
        samples.push(fusion_excess(confirm, &w));
    }
    samples.sort_by(|a, b| a.total_cmp(b));

    let mut l1s = Vec::new();
    let mut end = n_weight;
    while end < all_rows.len() {
        end = (end + Protocol::STABILITY_BLOCK).min(all_rows.len());
        let w = fit_red(&all_rows[..end]);
        let l1: f64 = w
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        l1s.push(l1);
    }

    let corr_score = mean_score_corr(confirm);
    let corr_hit = series_corr(&confirm_hits.expert);
    let mut mat = vec![vec![0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            mat[i][j] = corr_score[i][j];
        }
    }
    let ev = eigenvalues_sym(mat);
    let n_eff = participation_ratio(&ev);
    let mut ev_arr = [0.0; 7];
    for (i, v) in ev.iter().enumerate() {
        ev_arr[i] = *v;
    }
    ev_arr.sort_by(|a, b| b.total_cmp(a));
    let (mean_corr, mean_abs_corr, corr_span) = offdiag_span(&corr_score);
    let denom = 1.0 + 6.0 * mean_corr;
    let n_eff_eq = if denom > 0.05 && corr_span < 0.25 {
        let v = 7.0 / denom;
        if v <= 7.05 { Some(v) } else { None }
    } else {
        None
    };
    Robust {
        weights: *weights,
        blue_weights: *blue_weights,
        best_expert,
        loo,
        dirichlet_p05: percentile_sorted(&samples, 0.05),
        dirichlet_p50: percentile_sorted(&samples, 0.50),
        dirichlet_p95: percentile_sorted(&samples, 0.95),
        frozen_excess,
        l1_mean: mean_of(&l1s),
        l1_max: l1s.iter().copied().fold(0.0, f64::max),
        corr_score,
        corr_hit,
        eigenvalues: ev_arr,
        n_eff,
        n_eff_eq,
        mean_corr,
        mean_abs_corr,
    }
}

fn fusion_excess(rows: &[Row], w: &[f64; 7]) -> f64 {
    let expected = expected_red_hits(Protocol::PRIMARY_K);
    let mut s = 0.0;
    for row in rows {
        let fusion = fuse7(&row.expert_p, w);
        let order = rank_desc(&fusion);
        s += count_top(&order, row.red_mask, Protocol::PRIMARY_K) - expected;
    }
    s / rows.len() as f64
}

fn mean_score_corr(rows: &[Row]) -> [[f64; 7]; 7] {
    let mut sum = [[0.0; 7]; 7];
    let mut cnt = [[0u32; 7]; 7];
    for row in rows {
        for i in 0..7 {
            for j in (i + 1)..7 {
                if let Some(c) = pearson(&row.expert_p[i], &row.expert_p[j]) {
                    sum[i][j] += c;
                    cnt[i][j] += 1;
                }
            }
        }
    }
    let mut m = [[0.0; 7]; 7];
    for i in 0..7 {
        m[i][i] = 1.0;
        for j in (i + 1)..7 {
            let c = if cnt[i][j] == 0 {
                0.0
            } else {
                sum[i][j] / cnt[i][j] as f64
            };
            m[i][j] = c;
            m[j][i] = c;
        }
    }
    m
}

fn series_corr(series: &[Vec<f64>; 7]) -> [[f64; 7]; 7] {
    let mut m = [[0.0; 7]; 7];
    for i in 0..7 {
        m[i][i] = 1.0;
        for j in (i + 1)..7 {
            let c = pearson(&series[i], &series[j]).unwrap_or(0.0);
            m[i][j] = c;
            m[j][i] = c;
        }
    }
    m
}

fn offdiag_span(corr: &[[f64; 7]; 7]) -> (f64, f64, f64) {
    let mut sum = 0.0;
    let mut abs_sum = 0.0;
    let mut n = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..7 {
        for j in (i + 1)..7 {
            sum += corr[i][j];
            abs_sum += corr[i][j].abs();
            lo = lo.min(corr[i][j]);
            hi = hi.max(corr[i][j]);
            n += 1.0;
        }
    }
    (sum / n, abs_sum / n, hi - lo)
}

fn adjust_curve(topk: &mut [TopkPoint]) {
    let idx: Vec<usize> = topk
        .iter()
        .enumerate()
        .filter(|(_, p)| p.window == "确认窗")
        .map(|(i, _)| i)
        .collect();
    let raw: Vec<f64> = idx.iter().map(|&i| topk[i].p_raw).collect();
    let holm = adjust_holm(&raw);
    for (j, &i) in idx.iter().enumerate() {
        topk[i].p_curve = Some(holm[j]);
    }
}

fn locate_span(rows: &[Row], start_issue: u32, confirm_len: usize) -> Result<(usize, usize), String> {
    let start = rows
        .iter()
        .position(|r| r.issue == start_issue)
        .ok_or_else(|| format!("数据中没有确认窗起点 {start_issue}"))?;
    if start < 30 {
        return Err("确认窗之前的样本外期数不足".into());
    }
    let end = start + confirm_len;
    if end > rows.len() {
        return Err(format!(
            "确认窗需要 {confirm_len} 期，起点 {start_issue} 之后只剩 {} 期",
            rows.len() - start
        ));
    }
    Ok((start, end))
}

fn weight_rows_for_publish(rows: &[Row]) -> Result<&[Row], String> {
    let end = rows
        .iter()
        .take_while(|r| r.issue < Protocol::CONFIRM_START_ISSUE)
        .count();
    if end < 30 {
        return Err("确认窗之前的样本外期数不足，无法拟合融合权重".into());
    }
    Ok(&rows[..end])
}

fn assemble_tests(
    hits: &HitPack,
    rows: &[Row],
    blue_weights: &[f64; 6],
    best_expert: usize,
    ticket_excess: &[f64],
    ticket_p: f64,
    cond_null: f64,
    iid_null: f64,
) -> Vec<TestResult> {
    let mut drafts = Vec::new();
    for e in 0..7 {
        drafts.push(excess_test(
            format!("红球超额|{}", Protocol::EXPERTS_ZH[e]),
            &hits.expert[e],
            "零假设为超几何分布，p 用其方差；区间用样本标准误",
        ));
    }
    drafts.push(excess_test(
        "红球超额|加权融合".into(),
        &hits.fusion,
        "冻结权重，无救援",
    ));
    drafts.push(excess_test(
        "红球超额|等权融合".into(),
        &hits.equal,
        "等权，无救援",
    ));
    drafts.push(excess_test(
        "红球超额|温和救援".into(),
        &hits.mild,
        "生产路径使用的救援规格，定稿集大小仍为 15",
    ));
    drafts.push(excess_test(
        "红球超额|强救援".into(),
        &hits.agg,
        "消融规格，不进入发布路径",
    ));
    drafts.push(paired_test(
        "配对|加权融合-等权融合".into(),
        &hits.fusion,
        &hits.equal,
        Protocol::SEED ^ 12,
        true,
        "正值表示加权融合命中更多",
    ));
    drafts.push(paired_test(
        format!(
            "配对|加权融合-开发窗最优({})",
            Protocol::EXPERTS_ZH[best_expert]
        ),
        &hits.fusion,
        &hits.expert[best_expert],
        Protocol::SEED ^ 13,
        true,
        "最优专家只在开发窗按 K=15 超额选出",
    ));
    drafts.push(paired_test(
        "配对|温和救援-加权融合".into(),
        &hits.mild,
        &hits.fusion,
        Protocol::SEED ^ 14,
        true,
        "正值表示救援通道提高了定稿集命中",
    ));
    drafts.push(paired_test(
        "配对|强救援-加权融合".into(),
        &hits.agg,
        &hits.fusion,
        Protocol::SEED ^ 15,
        true,
        "正值表示强救援提高了定稿集命中",
    ));

    for e in 0..6 {
        drafts.push(blue_hit_test(
            format!("蓝球命中差|{}", Protocol::BLUES_ZH[e]),
            rows,
            &rows.iter().map(|r| r.blue_p[e]).collect::<Vec<_>>(),
        ));
    }
    let fused = rows
        .iter()
        .map(|r| fuse6(&r.blue_p, blue_weights))
        .collect::<Vec<_>>();
    drafts.push(blue_hit_test("蓝球命中差|加权融合".into(), rows, &fused));
    drafts.push(blue_ll_test(
        "蓝球损失|融合-匹配随机".into(),
        rows,
        &fused,
        true,
        Protocol::SEED ^ 23,
    ));
    drafts.push(blue_ll_test(
        "蓝球损失|融合-均匀".into(),
        rows,
        &fused,
        false,
        Protocol::SEED ^ 24,
    ));

    let (mean, sd, se) = mean_sd(ticket_excess);
    drafts.push(Draft {
        name: "票组红球|最佳票-条件零假设".into(),
        estimate: mean,
        effect: cohen_dz(mean, sd),
        effect_kind: "dz",
        p: ticket_p,
        n: ticket_excess.len(),
        better_if_positive: true,
        p_block: None,
        note: format!(
            "条件于当期已生成的票组做蒙特卡洛。条件期望 {cond_null:.4}，同等张数的独立均匀票期望 {iid_null:.4}"
        ),
        ci: CiKind::Normal { se },
    });

    let m = drafts.len();
    let ps: Vec<f64> = drafts.iter().map(|d| d.p).collect();
    let holm = adjust_holm(&ps);
    let bh = adjust_bh(&ps);
    let z95 = norm_ppf(0.975);
    let z_b = norm_ppf(1.0 - 0.025 / m as f64);
    drafts
        .into_iter()
        .enumerate()
        .map(|(i, d)| {
            let (lo, hi) = ci_of(&d, z95);
            let (blo, bhi) = ci_of(&d, z_b);
            let mut note = d.note;
            if let Some(pb) = d.p_block {
                let disagree = (d.p < 0.05 && pb > 0.10) || (pb < 0.05 && d.p > 0.10);
                note.push_str(&format!(
                    "。块符号翻转参照 p={pb:.4}{}",
                    if disagree { "，与主 p 不一致" } else { "" }
                ));
            }
            TestResult {
                name: d.name,
                estimate: d.estimate,
                ci_low: lo,
                ci_high: hi,
                ci_bonf_low: blo,
                ci_bonf_high: bhi,
                effect: d.effect,
                effect_kind: d.effect_kind,
                p: d.p,
                p_holm: holm[i],
                p_bh: bh[i],
                p_block: d.p_block,
                n: d.n,
                better_if_positive: d.better_if_positive,
                note,
            }
        })
        .collect()
}

fn excess_test(name: String, hits: &[f64], note: &str) -> Draft {
    let excess = excess_from_hits(hits);
    let (mean, sd, se) = mean_sd(&excess);
    let sum_hits = hits.iter().sum::<f64>();
    Draft {
        name,
        estimate: mean,
        effect: cohen_dz(mean, sd),
        effect_kind: "dz",
        p: hg_total_p(sum_hits, Protocol::PRIMARY_K, hits.len()),
        n: hits.len(),
        better_if_positive: true,
        p_block: None,
        note: note.into(),
        ci: CiKind::Normal { se },
    }
}

fn paired_test(
    name: String,
    a: &[f64],
    b: &[f64],
    seed: u64,
    better_if_positive: bool,
    note: &str,
) -> Draft {
    let diff: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    let (mean, sd, se) = mean_sd(&diff);
    let p_block = block_signflip_p(
        &diff,
        Protocol::SIGNFLIP_BLOCK,
        Protocol::SIGNFLIP_B,
        seed,
    );
    Draft {
        name,
        estimate: mean,
        effect: cohen_dz(mean, sd),
        effect_kind: "dz",
        p: z_test(mean, se),
        n: diff.len(),
        better_if_positive,
        p_block: Some(p_block),
        note: note.into(),
        ci: CiKind::Normal { se },
    }
}

fn blue_hit_test(name: String, rows: &[Row], probs: &[[f64; 16]]) -> Draft {
    let mut hits = 0.0;
    for (row, p) in rows.iter().zip(probs.iter()) {
        let y = row.blue as usize;
        let mut better = 0usize;
        for i in 0..16 {
            if i != y && (p[i] > p[y] || (p[i] == p[y] && i < y)) {
                better += 1;
            }
        }
        if better == 0 {
            hits += 1.0;
        }
    }
    let n = rows.len() as f64;
    let p0 = 1.0 / 16.0;
    let rate = hits / n;
    Draft {
        name,
        estimate: rate - p0,
        effect: cohen_h(rate, p0),
        effect_kind: "h",
        p: binom_cc_p(hits, n, p0),
        n: rows.len(),
        better_if_positive: true,
        p_block: None,
        note: "相对 1/16。p 为连续性校正后的二项 z；区间为 Wilson 再减去 1/16".into(),
        ci: CiKind::Wilson { hits, n, p0 },
    }
}

fn blue_ll_test(
    name: String,
    rows: &[Row],
    probs: &[[f64; 16]],
    matched: bool,
    seed: u64,
) -> Draft {
    let mut diff = Vec::with_capacity(rows.len());
    for (row, p) in rows.iter().zip(probs.iter()) {
        let y = row.blue as usize;
        let model = -p[y].clamp(1e-12, 1.0).ln();
        let base = if matched {
            let mut s = 0.0;
            for i in 0..16 {
                s += -p[i].clamp(1e-12, 1.0).ln();
            }
            s / 16.0
        } else {
            16.0_f64.ln()
        };
        diff.push(model - base);
    }
    let (mean, sd, se) = mean_sd(&diff);
    let p_block = block_signflip_p(
        &diff,
        Protocol::SIGNFLIP_BLOCK,
        Protocol::SIGNFLIP_B,
        seed,
    );
    Draft {
        name,
        estimate: mean,
        effect: cohen_dz(mean, sd),
        effect_kind: "dz",
        p: z_test(mean, se),
        n: diff.len(),
        better_if_positive: false,
        p_block: Some(p_block),
        note: "负值表示对数损失低于基线".into(),
        ci: CiKind::Normal { se },
    }
}

fn ci_of(d: &Draft, z: f64) -> (f64, f64) {
    match d.ci {
        CiKind::Normal { se } => normal_ci(d.estimate, se, z),
        CiKind::Wilson { hits, n, p0 } => {
            let (lo, hi) = wilson(hits, n, z);
            (lo - p0, hi - p0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::synth_draws;

    #[test]
    fn synthetic_protocol_runs() {
        let draws = synth_draws(400, 11);
        let start = draws[400 - 156].issue;
        let report = backtest_span(&draws, start, 156).unwrap();
        assert_eq!(report.tests.len(), 25);
        let wsum: f64 = report.robust.weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-9);
        assert!(report.robust.n_eff > 0.5 && report.robust.n_eff <= 7.01);
        assert_eq!(report.funnel.mean_alive.len(), 5);
        assert!(report.tradeoff.iter().any(|c| c.pareto));
        assert!(report.elapsed_ms > 0);
        let curve: Vec<_> = report
            .topk
            .iter()
            .filter(|p| p.window == "确认窗")
            .collect();
        assert_eq!(curve.len(), 64);
        assert!(curve.iter().all(|p| p.p_curve.is_some()));
    }

    #[test]
    fn publish_weights_stay_before_confirm_start() {
        let mut draws = synth_draws(400, 11);
        for (i, d) in draws.iter_mut().enumerate() {
            d.issue = 24881 + i as u32;
        }
        let early = &draws[..386];
        let a = prepare_forecast(early).unwrap();
        let b = prepare_forecast(&draws).unwrap();
        for i in 0..7 {
            assert!((a.weights[i] - b.weights[i]).abs() < 1e-15);
        }
        for i in 0..6 {
            assert!((a.blue_weights[i] - b.blue_weights[i]).abs() < 1e-15);
        }
        let snap = crate::model::roll_all(&draws).unwrap();
        let start = snap
            .rows
            .iter()
            .position(|r| r.issue == Protocol::CONFIRM_START_ISSUE)
            .unwrap();
        assert!(start >= 30);
        let w = fit_red(&snap.rows[..start]);
        for i in 0..7 {
            assert!((b.weights[i] - w[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn missing_confirmation_start_does_not_change_frozen_weights() {
        let mut draws = synth_draws(400, 11);
        for (i, draw) in draws.iter_mut().enumerate() {
            draw.issue = 24881 + i as u32;
        }
        let baseline = prepare_forecast(&draws).unwrap();
        draws.retain(|draw| draw.issue != Protocol::CONFIRM_START_ISSUE);
        let early = prepare_forecast(&draws[..draws.len() - 1]).unwrap();
        let late = prepare_forecast(&draws).unwrap();
        assert_eq!(baseline.weights, early.weights);
        assert_eq!(baseline.blue_weights, early.blue_weights);
        assert_eq!(early.weights, late.weights);
        assert_eq!(early.blue_weights, late.blue_weights);

        let mut only_confirmation = synth_draws(400, 11);
        for (i, draw) in only_confirmation.iter_mut().enumerate() {
            draw.issue = Protocol::CONFIRM_START_ISSUE + i as u32;
        }
        assert!(prepare_forecast(&only_confirmation).is_err());
    }

    #[test]
    fn real_confirmation_window_is_pinned() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ssq.csv");
        let draws = crate::draw::load_csv(&path).expect("data/ssq.csv");
        let bt = backtest(&draws).unwrap();
        assert_eq!(bt.confirm_issue_start, 25111);
        assert_eq!(bt.confirm_issue_end, 26115);
        assert_eq!(bt.n_confirm, 156);
        assert_eq!(bt.tests.len(), 25);
        assert!(bt.tests.iter().all(|t| t.p_holm + 1e-15 >= 0.05));
    }
}
