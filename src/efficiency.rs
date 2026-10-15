//! Fixed structural objectives, independent of the draw being reviewed.
use std::collections::BTreeSet;

use crate::hash::sha256_hex;
use crate::plan::PlanTicket;
use crate::protocol::Protocol;
use crate::select::Forecast;
use crate::stats::Rng;

pub const VERSION: u32 = 1;
pub const PROFILE_COUNT: usize = 9;
pub const BUDGETS: [usize; 4] = [2, 3, 4, 5];

pub struct Profile {
    pub name: String,
    pub title: String,
    pub rationale: String,
    pub basis: &'static str,
    pub tickets: Vec<PlanTicket>,
}

/// Preserve both marginal unions with the fewest valid SSQ bets.
/// Each bet covers at most six reds and one blue, so max(ceil(R/6), B)
/// is a lower bound. Snake allocation and deterministic padding attain it.
/// Red/blue pair exposures and prize outcomes are not preserved.
pub fn partition_cover(reds: &[u8], blues: &[u8]) -> Result<Vec<PlanTicket>, String> {
    if reds.len() < 6
        || reds.iter().any(|n| !(1..=33).contains(n))
        || reds.iter().collect::<BTreeSet<_>>().len() != reds.len()
        || blues.is_empty()
        || blues.iter().any(|n| !(1..=16).contains(n))
        || blues.iter().collect::<BTreeSet<_>>().len() != blues.len()
    {
        return Err("覆盖池须包含至少6个不同红球及至少1个不同蓝球，且号码在合法范围内".into());
    }
    let count = reds.len().div_ceil(6).max(blues.len());
    let mut groups = vec![Vec::new(); count];
    for (offset, &number) in reds.iter().enumerate() {
        let row = offset / count;
        let col = offset % count;
        groups[if row % 2 == 0 { col } else { count - 1 - col }].push(number);
    }
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(i, mut group)| {
            for &n in reds {
                if group.len() == 6 {
                    break;
                }
                if !group.contains(&n) {
                    group.push(n);
                }
            }
            group.sort_unstable();
            PlanTicket {
                red: group.try_into().expect("six distinct reds"),
                blue: blues[i % blues.len()],
            }
        })
        .collect())
}

fn shuffled(size: u8, issue: &str, color: &str) -> Vec<u8> {
    // Separate deterministic streams for each issue and color; no historical
    // outcome or score is used to choose the seed or a winning permutation.
    let hash = sha256_hex(
        format!(
            "ssq-efficiency-v{VERSION}:{}:{issue}:{color}",
            Protocol::SEED
        )
        .as_bytes(),
    );
    let mut rng = Rng::new(u64::from_str_radix(&hash[..16], 16).expect("sha256 prefix"));
    let mut numbers: Vec<_> = (1..=size).collect();
    for i in (1..numbers.len()).rev() {
        numbers.swap(i, rng.gen_range(i + 1));
    }
    numbers
}

pub fn profiles(fc: &Forecast, issue: &str) -> Result<Vec<Profile>, String> {
    let original_union = fc.tickets.iter().fold(0u64, |mask, red| mask | red);
    let original_reds: Vec<_> = fc
        .red_order
        .iter()
        .filter(|&&i| original_union & (1 << i) != 0)
        .map(|i| *i as u8 + 1)
        .collect();
    let original_blues: Vec<_> = fc
        .ticket_blues
        .iter()
        .map(|i| *i as u8 + 1)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut out = vec![Profile {
        name: "efficiency:compact".into(), title: "同覆盖最少注重组".into(), basis: "same_union",
        rationale: "分别保留原冻结票组的红球并集与蓝球并集，以最少单注数重组；红蓝配对、单注命中和奖金可能改变。".into(),
        tickets: partition_cover(&original_reds, &original_blues)?,
    }];
    let uniform_red = shuffled(33, issue, "red");
    let uniform_blue = shuffled(16, issue, "blue");
    let model_red: Vec<_> = fc.red_order.iter().map(|i| *i as u8 + 1).collect();
    let model_blue: Vec<_> = fc.blue_order[..2].iter().map(|i| *i as u8 + 1).collect();
    for budget in BUDGETS {
        for (basis, reds, blues, label) in [
            ("model", &model_red, &model_blue, "融合排名"),
            ("uniform", &uniform_red, &uniform_blue, "均匀对照"),
        ] {
            out.push(Profile {
                name: format!("efficiency:{basis}:{budget}"),
                title: format!("{budget}注{label}分散"),
                basis,
                rationale: format!(
                    "{budget}注、{}元，红球不重复覆盖{}个号码，配2个蓝球。{}",
                    budget * 2,
                    budget * 6,
                    if basis == "model" {
                        "按开奖前融合排名取号，评分不是中奖概率。"
                    } else {
                        "按期固定种子打乱红蓝球，作为同预算结构对照，无模型评分。"
                    }
                ),
                tickets: partition_cover(&reds[..budget * 6], &blues[..2])?,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::Draw;

    #[test]
    fn minimum_bets_preserve_both_unions_for_every_valid_union_size() {
        for red_n in 6usize..=33 {
            for blue_n in 1usize..=16 {
                let reds: Vec<_> = (1..=red_n as u8).collect();
                let blues: Vec<_> = (1..=blue_n as u8).collect();
                let tickets = partition_cover(&reds, &blues).unwrap();
                assert_eq!(tickets.len(), red_n.div_ceil(6).max(blue_n));
                assert_eq!(
                    tickets.iter().flat_map(|t| t.red).collect::<BTreeSet<_>>(),
                    reds.into_iter().collect()
                );
                assert_eq!(
                    tickets.iter().map(|t| t.blue).collect::<BTreeSet<_>>(),
                    blues.into_iter().collect()
                );
                for t in &tickets {
                    Draw::from_parts(0, 20000101, t.red, t.blue).unwrap();
                }
                assert_eq!(
                    tickets
                        .iter()
                        .map(|t| (t.red, t.blue))
                        .collect::<BTreeSet<_>>()
                        .len(),
                    tickets.len()
                );
            }
        }
        assert!(partition_cover(&[1, 1, 2, 3, 4, 5], &[1]).is_err());
        assert!(partition_cover(&[1, 2, 3, 4, 5, 6], &[17]).is_err());
    }

    #[test]
    fn fixed_profiles_have_matched_budgets_and_uniform_controls_ignore_model_scores() {
        let draws = crate::draw::synth_draws(220, 9);
        let mut prepared = crate::eval::prepare_forecast(&draws).unwrap();
        let first = profiles(&prepared.forecast, "03221").unwrap();
        prepared.forecast.red_order.reverse();
        prepared.forecast.blue_order.reverse();
        let changed = profiles(&prepared.forecast, "03221").unwrap();
        assert_eq!(first.len(), PROFILE_COUNT);
        for budget in BUDGETS {
            for basis in ["model", "uniform"] {
                let name = format!("efficiency:{basis}:{budget}");
                let p = first.iter().find(|p| p.name == name).unwrap();
                assert_eq!(p.tickets.len(), budget);
                assert_eq!(
                    p.tickets
                        .iter()
                        .flat_map(|t| t.red)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    budget * 6
                );
                assert_eq!(
                    p.tickets
                        .iter()
                        .map(|t| t.blue)
                        .collect::<BTreeSet<_>>()
                        .len(),
                    2
                );
                if basis == "uniform" {
                    let other = changed.iter().find(|p| p.name == name).unwrap();
                    assert_eq!(
                        p.tickets
                            .iter()
                            .map(|t| (t.red, t.blue))
                            .collect::<Vec<_>>(),
                        other
                            .tickets
                            .iter()
                            .map(|t| (t.red, t.blue))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        assert_ne!(shuffled(33, "03221", "red"), shuffled(33, "03222", "red"));
    }
}
