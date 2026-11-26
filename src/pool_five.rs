//! Five disjoint singles from one ranked 30-red / 5-blue pool.
use crate::{
    blue_model_v2 as blue,
    draw::{data_sha256, fmt_issue, Draw},
    eval::Prepared,
    fusion_model_v3::Distribution,
    model::Row,
    plan::{self, Plan, PlanTicket, Spec},
    red_model_v2 as red,
    select::rank_desc,
    stats::binom,
};
use serde_json::{json, Value};
use std::sync::{Mutex, OnceLock};

pub(crate) const MODE: &str = "pool_five";
const TICKETS: usize = 5;
const POOL: usize = TICKETS * 6;
const TITLE: &str = "号码池分散推荐 · 5注";
const SWAP_PASSES: usize = 12;

fn spec() -> Spec {
    Spec {
        mode: MODE,
        title: TITLE.into(),
        detail: "原加权榜圈定30红，历史拟合重排池内红球，并取前5蓝。五注红球互不重复，每注用不同蓝球；在固定号码池内搜索常规奖覆盖。".into(),
        tickets: TICKETS,
        pool: POOL,
        overlap: 0,
        red_count: 6,
        blue_count: TICKETS,
        bankers: 0,
        drags: 0,
    }
}

pub(crate) fn available(draws: &[Draw]) -> bool {
    draws
        .binary_search_by_key(&red::TRAIN_END, |draw| draw.issue)
        .is_ok()
}

#[derive(Clone)]
pub(crate) struct Fit {
    pub(crate) red: [f64; 6],
    pub(crate) blue: [f64; 5],
    training_sha: String,
}

// Use the frozen model APIs here so existing fusion archive source hashes stay valid.
pub(crate) fn fit(draws: &[Draw], rows: &[Row]) -> Result<Fit, String> {
    static CACHE: OnceLock<Mutex<Option<Fit>>> = OnceLock::new();
    let train_end = draws.partition_point(|d| d.issue <= red::TRAIN_END);
    let training_sha = data_sha256(&draws[..train_end]);
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| "历史拟合缓存不可用")?;
    if let Some(fit) = cache
        .as_ref()
        .filter(|fit| fit.training_sha == training_sha)
    {
        return Ok(fit.clone());
    }
    let red = red::fit_causal(rows)?.beta;
    let training = rows
        .iter()
        .skip(30)
        .take_while(|row| row.issue <= blue::TRAIN_END)
        .map(|row| {
            Ok(blue::Sample {
                x: blue::features(&row.blue_p)?,
                winner: row.blue as usize,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let result = Fit {
        red,
        blue: blue::fit(&training)?.beta,
        training_sha,
    };
    *cache = Some(result.clone());
    Ok(result)
}

impl Fit {
    pub(crate) fn distribution(
        &self,
        red: &[[f64; 33]; 7],
        blue: &[[f64; 16]; 6],
    ) -> Result<Distribution, String> {
        Distribution::new(
            &red::weights(&self.red, &red::features(red)?)?,
            &blue::distribution(&self.blue, &blue::features(blue)?),
        )
    }
}

fn policy() -> Value {
    json!({
        "version": 1,
        "model": MODE,
        "label": "30红池覆盖排序",
        "training_last_issue": fmt_issue(red::TRAIN_END),
        "red_pool_selection": "protocol_weighted_top30",
        "red_pool_order": "historical_fit_within_selected_and_excluded_groups",
        "red_model": "complete_six_red_v2",
        "blue_model": "blue_score_v2",
        "objective": "disjoint_regular_prize_search_within_fixed_red_pool",
        "starts": ["rank_blocks", "snake_ranks"],
        "max_swap_passes_per_start": SWAP_PASSES,
        "blue_pairing": "strongest_red_set_with_highest_ranked_blue",
    })
}

fn uniform(blue_count: usize) -> Value {
    let red_total = binom(33, 6);
    let one_four_plus: u64 = (4..=6).map(|k| binom(6, k) * binom(27, 6 - k)).sum();
    // With only six drawn reds, two disjoint tickets cannot both have four reds.
    let any_prize =
        blue_count as u64 * red_total + (16 - blue_count as u64) * TICKETS as u64 * one_four_plus;
    let full_red = crate::pool::uniform_full_red(POOL);
    json!({
        "full_red_union": full_red,
        "blue_union": blue_count as f64 / 16.0,
        "full_draw_union": full_red * blue_count as f64 / 16.0,
        "any_regular_prize": any_prize as f64 / (16 * red_total) as f64,
        "jackpot": TICKETS as f64 / (16 * red_total) as f64,
    })
}

type Group = [u8; 6];
type Groups = [Group; TICKETS];

pub(crate) struct Selection {
    reds: Vec<u8>,
    blues: Vec<u8>,
    tickets: [PlanTicket; TICKETS],
    fitted_red_mask: u64,
}

impl Selection {
    pub(crate) fn new(dist: &Distribution, weighted_order: &[usize; 33]) -> Self {
        let pool = crate::select::top_mask(weighted_order, POOL);
        let mut reds = dist.red_order();
        let fitted_red_mask = reds[..POOL]
            .iter()
            .fold(0, |mask, &n| mask | (1u64 << (n - 1)));
        // A stable sort preserves the fitted order on each side of the pool cut.
        reds.sort_by_key(|&n| pool & (1u64 << (n - 1)) == 0);
        let blues = dist.blue_order();
        let blocks = std::array::from_fn(|g| std::array::from_fn(|i| reds[6 * g + i]));
        let snake = std::array::from_fn(|g| {
            std::array::from_fn(|i| {
                reds[i * TICKETS + if i % 2 == 0 { g } else { TICKETS - 1 - g }]
            })
        });
        let objective = RedPrize::new(&dist.red);
        let (a, a_score) = objective.improve(blocks);
        let (b, b_score) = objective.improve(snake);
        let mut groups = if b_score > a_score { b } else { a };
        groups.iter_mut().for_each(|group| group.sort_unstable());
        let mass =
            |group: &Group| -> f64 { group.iter().map(|&n| dist.red[n as usize - 1]).product() };
        // This ordering maximizes model 6+1 mass for the five already chosen red sets.
        groups.sort_by(|a, b| mass(b).total_cmp(&mass(a)).then(a.cmp(b)));
        let tickets = std::array::from_fn(|i| PlanTicket {
            red: groups[i],
            blue: blues[i],
        });
        Self {
            reds,
            blues,
            tickets,
            fitted_red_mask,
        }
    }

    pub(crate) fn view(&self, draw: Option<&Draw>) -> Value {
        json!({
            "policy": policy(),
            "red_ranking": self.reds,
            "blue_ranking": self.blues,
            "excluded_reds": self.reds[POOL..],
            "ticket_count": TICKETS,
            "cost": TICKETS * 2,
            "tickets": ticket_values(&self.tickets),
            "outcome": draw.map(|d| Counts::of(&self.tickets, d)),
            "uniform": uniform(TICKETS),
        })
    }

    pub(crate) fn compare(&self, red: &[f64; 33], blue: &[f64; 16], draw: &Draw) -> Comparison {
        let mut pool: Vec<_> = rank_desc(red)[..POOL].to_vec();
        // Same greedy, disjoint choices as disjoint5's exhaustive builder:
        // highest remaining six by its integer scores, then lowest mask on ties.
        let key = |n: usize| (red[n] * 1.0e9).round() as i64;
        pool.sort_by(|&a, &b| key(b).cmp(&key(a)).then(a.cmp(&b)));
        let blue_order = rank_desc(blue);
        let legacy: [PlanTicket; TICKETS] = std::array::from_fn(|i| {
            let mut group = std::array::from_fn(|j| pool[i * 6 + j] as u8 + 1);
            group.sort_unstable();
            PlanTicket {
                red: group,
                blue: blue_order[i % 2] as u8 + 1,
            }
        });
        Comparison {
            ranked: Counts::of(&self.tickets, draw),
            legacy: Counts::of(&legacy, draw),
            fitted_full_red: draw.red_mask & self.fitted_red_mask == draw.red_mask,
        }
    }
}

struct RedPrize<'a> {
    weights: &'a [f64],
    sum: f64,
    pairs: f64,
}

impl<'a> RedPrize<'a> {
    fn new(weights: &'a [f64]) -> Self {
        let (mut sum, mut pairs) = (0.0, 0.0);
        for &weight in weights {
            pairs += sum * weight;
            sum += weight;
        }
        Self {
            weights,
            sum,
            pairs,
        }
    }

    fn mass(&self, group: &Group) -> f64 {
        let mut inside = [0.0; 7];
        inside[0] = 1.0;
        for (i, &n) in group.iter().enumerate() {
            for k in (1..=i + 1).rev() {
                inside[k] += self.weights[n as usize - 1] * inside[k - 1];
            }
        }
        let outside_single = self.sum - inside[1];
        let outside_pair = self.pairs - inside[1] * outside_single - inside[2];
        // Unnormalized probability of 4, 5 or 6 red matches in this ticket.
        inside[4] * outside_pair + inside[5] * outside_single + inside[6]
    }

    fn improve(&self, mut groups: Groups) -> (Groups, f64) {
        let mut scores = groups.map(|g| self.mass(&g));
        for _ in 0..SWAP_PASSES {
            let mut best_gain = 1e-8;
            let mut best = None;
            for a in 0..TICKETS {
                for b in a + 1..TICKETS {
                    let (mut left, mut right) = (groups[a], groups[b]);
                    for i in 0..6 {
                        for j in 0..6 {
                            std::mem::swap(&mut left[i], &mut right[j]);
                            let (l, r) = (self.mass(&left), self.mass(&right));
                            let gain = (l + r) - (scores[a] + scores[b]);
                            if gain > best_gain {
                                best_gain = gain;
                                best = Some((a, b, left, right, l, r));
                            }
                            std::mem::swap(&mut left[i], &mut right[j]);
                        }
                    }
                }
            }
            let Some((a, b, left, right, l, r)) = best else {
                break;
            };
            (groups[a], groups[b]) = (left, right);
            (scores[a], scores[b]) = (l, r);
        }
        (groups, scores.iter().sum())
    }
}

fn ticket_values(tickets: &[PlanTicket]) -> Vec<Value> {
    tickets
        .iter()
        .map(|t| json!({"red": t.red, "blue": t.blue}))
        .collect()
}

pub(crate) fn build(draws: &[Draw], mut prepared: Prepared) -> Result<(Plan, Prepared), String> {
    let fit = fit(draws, &crate::model::roll_all(draws)?.rows)?;
    let dist = fit.distribution(&prepared.expert_red, &prepared.expert_blue)?;
    let selection = Selection::new(&dist, &prepared.forecast.red_order);
    prepared.ranking = Some(json!({
        "model": MODE,
        "policy": policy(),
        "target_issue": fmt_issue(prepared.cutoff_issue + 1),
        "training_history_sha256": fit.training_sha,
        "pool_red_weights": prepared.weights,
        "red_fit": {"axes": crate::red_model_v2::AXES, "coefficients": fit.red},
        "blue_fit": {"axes": crate::blue_model_v2::AXES, "coefficients": fit.blue},
        "input_history_sha256": prepared.data_sha,
        "red_ranking": selection.reds,
        "blue_ranking": selection.blues,
        "excluded_reds": selection.reds[POOL..],
        "uniform": uniform(TICKETS),
    }));
    let fc = &mut prepared.forecast;
    // The generic archive evidence uses these same orders and scores.
    fc.red_order = std::array::from_fn(|i| selection.reds[i] as usize - 1);
    fc.blue_order = std::array::from_fn(|i| selection.blues[i] as usize - 1);
    fc.fusion.copy_from_slice(&dist.red);
    fc.blue.copy_from_slice(&dist.blue);
    let plan = plan::finish(
        &spec(),
        selection.tickets.to_vec(),
        selection.reds[..POOL].to_vec(),
        vec![],
        vec![],
        selection.blues[..TICKETS].to_vec(),
        vec![],
    );
    Ok((plan, prepared))
}

pub(crate) fn validate_saved(saved: &Value, payload: &Value, draws: &[Draw]) -> Result<(), String> {
    let cutoff: u32 = payload["data_cutoff"]
        .as_str()
        .ok_or("分散5注缺少输入截止")?
        .parse()
        .map_err(|_| "分散5注输入截止无效")?;
    let target = payload["issue"].as_str().ok_or("分散5注缺少目标期")?;
    let draw = crate::audit::target_draw(draws, cutoff, target)?;
    let published = payload["published_at_unix"]
        .as_u64()
        .ok_or("分散5注缺少保存时间")?;
    let deadline = payload["operational_cutoff_unix"]
        .as_u64()
        .ok_or("分散5注缺少截止时间")?;
    if published >= deadline
        || payload["known_outcome_at_generation"] != false
        || crate::recommendations::saved_after_draw(Some(published), draw)
        || target != fmt_issue(cutoff + 1)
    {
        return Err("分散5注保存时间或目标期无效".into());
    }
    let prefix = &draws[..draws.partition_point(|d| d.issue <= cutoff)];
    let (plan, prepared) = build(prefix, crate::eval::prepare_forecast(prefix)?)?;
    if saved["ranking"] != json!(prepared.ranking)
        || saved["tickets"] != json!(ticket_values(&plan.tickets))
        || saved["mode"] != MODE
        || saved["parameters"]["mode"] != MODE
    {
        return Err("分散5注票面与开奖前排序不一致".into());
    }
    Ok(())
}

#[derive(Clone, Copy, serde::Serialize)]
struct Counts {
    red_hits: usize,
    blue_covered: bool,
    max_ticket_red_hits: usize,
    any_regular_prize: bool,
    jackpot: bool,
}

impl Counts {
    fn of(tickets: &[PlanTicket], draw: &Draw) -> Self {
        let mut result = Self {
            red_hits: 0,
            blue_covered: false,
            max_ticket_red_hits: 0,
            any_regular_prize: false,
            jackpot: false,
        };
        for ticket in tickets {
            let red = ticket.red.iter().filter(|r| draw.red.contains(r)).count();
            let blue = ticket.blue == draw.blue;
            result.red_hits += red; // Valid here because the five red groups are disjoint.
            result.blue_covered |= blue;
            result.max_ticket_red_hits = result.max_ticket_red_hits.max(red);
            result.any_regular_prize |= crate::recommendations::prize(red, blue).is_some();
            result.jackpot |= red == 6 && blue;
        }
        result
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Comparison {
    ranked: Counts,
    legacy: Counts,
    fitted_full_red: bool,
}

pub(crate) fn summary(rows: impl IntoIterator<Item = Comparison>) -> Value {
    let mut sums = [[0; 7]; 2];
    let (mut n, mut fitted, mut gains, mut losses) = (0, 0, 0, 0);
    for Comparison {
        ranked,
        legacy,
        fitted_full_red,
    } in rows
    {
        n += 1;
        fitted += usize::from(fitted_full_red);
        gains += usize::from(fitted_full_red && legacy.red_hits != 6);
        losses += usize::from(!fitted_full_red && legacy.red_hits == 6);
        for (sum, row) in sums.iter_mut().zip([ranked, legacy]) {
            sum[0] += row.red_hits;
            sum[1] += usize::from(row.red_hits == 6);
            sum[2] += usize::from(row.blue_covered);
            sum[3] += usize::from(row.red_hits == 6 && row.blue_covered);
            sum[4] += usize::from(row.max_ticket_red_hits >= 4);
            sum[5] += usize::from(row.any_regular_prize);
            sum[6] += usize::from(row.jackpot);
        }
    }
    let cells = sums
        .into_iter()
        .zip([TICKETS, 2])
        .map(|(s, blue_count)| {
            json!({
                "blue_count": blue_count, "mean_red_hits": (n > 0).then(|| s[0] as f64 / n as f64),
                "full_red_union": s[1], "blue_union": s[2], "full_draw_union": s[3],
                "any_ticket_four_reds": s[4], "any_regular_prize": s[5], "jackpot": s[6],
                "uniform": uniform(blue_count),
            })
        })
        .collect::<Vec<_>>();
    json!({"n": n, "ranked": cells[0], "legacy": cells[1],
        "fit_only_red_union": {"full_red_union": fitted, "paired_gains": gains, "paired_losses": losses}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn fixed_budget_uses_only_top_ranks_and_improves_the_search_start() {
        let red = std::array::from_fn::<_, 33, _>(|i| 1. + (i * 17 % 33) as f64 / 41.);
        let blue = std::array::from_fn::<_, 16, _>(|i| 1. + (i * 5 % 16) as f64 / 17.);
        let dist = Distribution::new(&red, &blue).unwrap();
        let order = std::array::from_fn(|i| i);
        let selected = Selection::new(&dist, &order);
        let expected_red: BTreeSet<_> = (1..=30).collect();
        let actual_red: BTreeSet<_> = selected.tickets.iter().flat_map(|t| t.red).collect();
        assert_eq!(expected_red, actual_red);
        assert_eq!(
            selected.tickets.each_ref().map(|t| t.blue).as_slice(),
            &dist.blue_order()[..TICKETS]
        );
        let objective = RedPrize::new(&dist.red);
        let actual: f64 = selected
            .tickets
            .iter()
            .map(|t| objective.mass(&t.red))
            .sum();
        let initial: f64 = selected.reds[..POOL]
            .chunks_exact(6)
            .map(|g| objective.mass(g.try_into().unwrap()))
            .sum();
        assert!(actual >= initial - 1e-8);

        let uniform_red = [1.; 33];
        assert_eq!(RedPrize::new(&uniform_red).mass(&[1, 2, 3, 4, 5, 6]), 5428.);
        assert_eq!(uniform(5)["full_red_union"], uniform(2)["full_red_union"]);
        assert_eq!(uniform(5)["jackpot"], uniform(2)["jackpot"]);
        assert!(
            uniform(5)["any_regular_prize"].as_f64() > uniform(2)["any_regular_prize"].as_f64()
        );
    }
}
