//! Exact same-budget comparison of the complete tickets in the v3 and v2 bundles.
use crate::{draw::Draw, plan::combinations, recommendations::review_strategy, stats::binom};
use serde_json::{json, Value};
use std::collections::HashSet;

const RED_OUTCOMES: u64 = 1_107_568;
const BLUE_OUTCOMES: u64 = 16;

fn red_mask(numbers: &[u8]) -> u64 {
    numbers.iter().fold(0, |mask, n| mask | (1 << (n - 1)))
}

fn red_prize_outcomes(reds: &HashSet<[u8; 6]>) -> Result<u64, String> {
    let mut outcomes = HashSet::new();
    // With no blue match, a regular prize requires at least four matching reds.
    // Deduplicate the qualifying outcomes across tickets and across blue variants.
    for red in reds {
        let mask = red_mask(red);
        outcomes.insert(mask);
        let outside: Vec<_> = (1..=33).filter(|n| mask & (1 << (n - 1)) == 0).collect();
        for five in combinations(red, 5)? {
            let five = red_mask(&five);
            for n in &outside {
                outcomes.insert(five | (1 << (n - 1)));
            }
        }
        for four in combinations(red, 4)? {
            let four = red_mask(&four);
            for (i, a) in outside.iter().enumerate() {
                for b in &outside[i + 1..] {
                    outcomes.insert(four | (1 << (a - 1)) | (1 << (b - 1)));
                }
            }
        }
    }
    Ok(outcomes.len() as u64)
}

fn entry(plan: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let reviewed = review_strategy(
        &json!({"title":plan["title"],"mode":plan["mode"],"structural":true,"tickets":plan["tickets"]}),
        draw,
    )?;
    let summary = &reviewed["summary"];
    let tickets = reviewed["tickets"].as_array().ok_or("缺少完整对照票面")?;
    let mut red_sets = HashSet::new();
    for ticket in tickets {
        let mut red: [u8; 6] =
            serde_json::from_value(ticket["red"].clone()).map_err(|_| "对照红票无效")?;
        red.sort_unstable();
        red_sets.insert(red);
    }
    let blues = summary["blue_coverage"].as_u64().ok_or("缺少实用蓝球数")?;
    let jackpot = tickets.len() as u64
        - summary["duplicate_tickets"]
            .as_u64()
            .ok_or("缺少重复注数")?;
    let red_only = red_prize_outcomes(&red_sets)?;
    Ok(json!({
        "id": plan["id"], "used_reds": summary["unique_reds"], "used_blues": summary["unique_blues"],
        "uniform_any_prize_outcomes": blues * RED_OUTCOMES + (BLUE_OUTCOMES - blues) * red_only,
        "uniform_jackpot_outcomes": jackpot,
        "actual": draw.map(|_| json!({
            "covered_reds": summary["covered_reds"], "missed_reds": summary["missed_reds"],
            "best_ticket_red_hits": summary["max_ticket_red_hits"], "blue_covered": summary["blue_covered"],
            "prize_tickets": summary["prize_tickets"], "fixed_prize": summary["fixed_prize"],
            "special_prize_candidates": summary["special_prize_candidates"],
            "single_jackpot": tickets.iter().any(|t| t["prize_level"] == 1),
        })),
    }))
}

pub(crate) fn compare(v3: &Value, legacy: &Value, draw: Option<&Draw>) -> Result<Value, String> {
    let mut pairs = Vec::new();
    for plan in v3["plans"].as_array().ok_or("v3缺少对照方案")? {
        let old_id = &plan["same_budget_v2_comparison"]["id"];
        let group = if plan["id"] == "joint8" {
            "singles"
        } else {
            "dantuo"
        };
        let old = legacy["payload"]["bundle"][group]
            .as_array()
            .and_then(|plans| plans.iter().find(|p| p["id"] == *old_id))
            .ok_or("v3缺少已绑定的v2同预算方案")?;
        if old["cost"] != plan["cost"] || old["bet_count"] != plan["bet_count"] {
            return Err("v3与v2对照预算不一致".into());
        }
        pairs.push(json!({
            "id": plan["id"], "bets": plan["bet_count"], "cost": plan["cost"],
            "v3": entry(plan, draw)?, "v2": entry(old, draw)?,
        }));
    }
    Ok(json!({
        "uniform_outcomes": binom(33, 6) * BLUE_OUTCOMES,
        "v2_archive_sha256": legacy["sha256"], "plans": pairs,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_tickets_match_independent_uniform_enumeration() {
        let reds = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 8]];
        let masks = reds.map(|red| red_mask(&red));
        let mut expected = 0;
        let mut red = (1u64 << 6) - 1;
        while red < (1u64 << 33) {
            if masks.iter().any(|mask| {
                crate::recommendations::prize((red & mask).count_ones() as usize, false).is_some()
            }) {
                expected += 1;
            }
            let low_bit = red & red.wrapping_neg();
            let advanced = red + low_bit;
            red = advanced | ((advanced ^ red) >> (low_bit.trailing_zeros() + 2));
        }
        let one = binom(6, 4) * binom(27, 2) + 6 * 27 + 1;
        assert!(expected > one && expected < 2 * one);
        let plan = json!({"tickets":[
            {"red":reds[0],"blue":1}, {"red":reds[1],"blue":1},
            {"red":reds[0],"blue":2}, {"red":reds[0],"blue":2},
        ]});
        let pending = entry(&plan, None).unwrap();
        assert_eq!(
            pending["uniform_any_prize_outcomes"],
            2 * RED_OUTCOMES + 14 * expected
        );
        assert_eq!(pending["uniform_jackpot_outcomes"], 3);
        assert!(pending["actual"].is_null());
        let draw = Draw::from_parts(26132, 20261117, reds[1], 1).unwrap();
        let drawn = entry(&plan, Some(&draw)).unwrap();
        assert_eq!(drawn["actual"]["best_ticket_red_hits"], 6);
        assert_eq!(drawn["actual"]["single_jackpot"], true);
        assert_eq!(drawn["actual"]["prize_tickets"], 4);
    }
}
