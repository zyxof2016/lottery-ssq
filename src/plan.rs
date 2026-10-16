use crate::protocol::Protocol;
use crate::select::{build_tickets, Forecast};

#[derive(Clone, Debug)]
pub struct Spec {
    pub mode: &'static str,
    pub title: String,
    pub detail: String,
    pub tickets: usize,
    pub pool: usize,
    pub overlap: u32,
    pub red_count: usize,
    pub blue_count: usize,
    pub bankers: usize,
    pub drags: usize,
}

#[derive(Clone, Debug)]
pub struct PlanTicket {
    pub red: [u8; 6],
    pub blue: u8,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub spec: Spec,
    pub mode: String,
    pub title: String,
    pub detail: String,
    pub bet_count: usize,
    pub cost: u32,
    pub tickets: Vec<PlanTicket>,
    pub reds: Vec<u8>,
    pub bankers: Vec<u8>,
    pub drags: Vec<u8>,
    pub blues: Vec<u8>,
    pub rescued: Vec<u8>,
}

pub fn resolve(
    preset: Option<&str>,
    mode: Option<&str>,
    tickets: Option<usize>,
    pool: Option<usize>,
    overlap: Option<u32>,
    red_count: Option<usize>,
    blue_count: Option<usize>,
    bankers: Option<usize>,
    drags: Option<usize>,
) -> Result<Spec, String> {
    if let Some(name) = preset {
        return preset_spec(name);
    }
    let mode = mode.unwrap_or("frozen");
    match mode {
        "frozen" => Ok(frozen_spec()),
        "balanced" => Ok(Spec {
            mode: "balanced", title: "均衡覆盖单式 · 试验".into(), detail: String::new(),
            tickets: tickets.unwrap_or(8), pool: pool.unwrap_or(14), overlap: 5,
            red_count: 6, blue_count: blue_count.unwrap_or(2), bankers: 0, drags: 0,
        }),
        "single" => Ok(Spec {
            mode: "single",
            title: "单式多注".into(),
            detail: String::new(),
            tickets: tickets.unwrap_or(8),
            pool: pool.unwrap_or(Protocol::POOL),
            overlap: overlap.unwrap_or(Protocol::MAX_OVERLAP),
            red_count: 6,
            blue_count: blue_count.unwrap_or(Protocol::BLUE_SLOTS),
            bankers: 0,
            drags: 0,
        }),
        "compound_red" => Ok(Spec {
            mode: "compound_red",
            title: "红球复式".into(),
            detail: String::new(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: red_count.unwrap_or(7),
            blue_count: blue_count.unwrap_or(1),
            bankers: 0,
            drags: 0,
        }),
        "compound_blue" => Ok(Spec {
            mode: "compound_blue",
            title: "蓝球复式".into(),
            detail: String::new(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 6,
            blue_count: blue_count.unwrap_or(2),
            bankers: 0,
            drags: 0,
        }),
        "dantuo" => Ok(Spec {
            mode: "dantuo",
            title: "胆拖".into(),
            detail: String::new(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 0,
            blue_count: blue_count.unwrap_or(1),
            bankers: bankers.unwrap_or(2),
            drags: drags.unwrap_or(6),
        }),
        _ => Err("未知生成方式".into()),
    }
}

pub fn build(fc: &Forecast, spec: &Spec) -> Result<Plan, String> {
    let rescued = fc.mild.rescued.iter().map(|i| *i as u8 + 1).collect();
    let plan = match spec.mode {
        "frozen" => frozen_plan(fc, spec, rescued),
        "single" => single_plan(fc, spec, rescued)?,
        "balanced" => balanced_plan(fc, spec)?,
        "compound_red" => compound_red(fc, spec, rescued)?,
        "compound_blue" => compound_blue(fc, spec, rescued)?,
        "dantuo" => dantuo(fc, spec, rescued)?,
        _ => return Err("未知生成方式".into()),
    };
    if plan.tickets.is_empty() {
        return Err("这组参数没有排出注单".into());
    }
    Ok(plan)
}

fn preset_spec(name: &str) -> Result<Spec, String> {
    Ok(match name {
        "frozen" => frozen_spec(),
        "balanced14" => Spec {
            mode: "balanced", title: "融合14红均衡覆盖 · 8注".into(),
            detail: "原融合排名前14红、2蓝、8注16元；完整使用14红，每红出现3–4次，无隐含全胆。单注组合不保证包含全部开奖号。".into(),
            tickets: 8, pool: 14, overlap: 5, red_count: 6, blue_count: 2, bankers: 0, drags: 0,
        },
        "disjoint5" => Spec {
            mode: "single",
            title: "5 注红球不重叠".into(),
            detail: "从融合名次前 30 个红球里取，注与注的红球互不重复。33 个号最多放下 5 注。".into(),
            tickets: 5,
            pool: 30,
            overlap: 0,
            red_count: 6,
            blue_count: 2,
            bankers: 0,
            drags: 0,
        },
        "single8" => Spec {
            mode: "single",
            title: "8 注低重合单式".into(),
            detail: "融合名次前 15 个红球，任意两注红球重合不超过 4。".into(),
            tickets: 8,
            pool: 15,
            overlap: 4,
            red_count: 6,
            blue_count: 2,
            bankers: 0,
            drags: 0,
        },
        "compound7" => Spec {
            mode: "compound_red",
            title: "红球复式 7 码".into(),
            detail: "展开融合名次前 7 个红球，配蓝球第一名。".into(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 7,
            blue_count: 1,
            bankers: 0,
            drags: 0,
        },
        "compound8" => Spec {
            mode: "compound_red",
            title: "红球复式 8 码".into(),
            detail: "展开融合名次前 8 个红球，配蓝球第一名。".into(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 8,
            blue_count: 1,
            bankers: 0,
            drags: 0,
        },
        "blue2" => Spec {
            mode: "compound_blue",
            title: "蓝球复式 2 码".into(),
            detail: "融合前 6 个红球不变，分别配蓝球前两名。".into(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 6,
            blue_count: 2,
            bankers: 0,
            drags: 0,
        },
        "dantuo" => Spec {
            mode: "dantuo",
            title: "2 胆 6 拖".into(),
            detail: "融合前 2 个红球做胆，其后 6 个做拖，每注再补 4 个拖码。".into(),
            tickets: 0,
            pool: 0,
            overlap: 0,
            red_count: 0,
            blue_count: 1,
            bankers: 2,
            drags: 6,
        },
        _ => return Err(format!("没有这个快捷方案：{name}")),
    })
}

fn frozen_spec() -> Spec {
    Spec {
        mode: "frozen",
        title: "冻结发布 8 注".into(),
        detail: "发布路径：温和救援、出票池 12、8 注、红球重合不超过 4。保存时同时写入开奖前哈希。".into(),
        tickets: Protocol::TICKETS,
        pool: Protocol::POOL,
        overlap: Protocol::MAX_OVERLAP,
        red_count: 6,
        blue_count: Protocol::BLUE_SLOTS,
        bankers: 0,
        drags: 0,
    }
}

fn frozen_plan(fc: &Forecast, spec: &Spec, rescued: Vec<u8>) -> Plan {
    let tickets = fc
        .tickets
        .iter()
        .zip(fc.ticket_blues.iter())
        .map(|(mask, blue)| PlanTicket {
            red: mask_nums(*mask),
            blue: *blue as u8 + 1,
        })
        .collect::<Vec<_>>();
    finish(spec, tickets, nums(&fc.pool), vec![], vec![], blues_of(fc, spec.blue_count.max(1)), rescued)
}

fn single_plan(fc: &Forecast, spec: &Spec, rescued: Vec<u8>) -> Result<Plan, String> {
    if !(1..=20).contains(&spec.tickets) {
        return Err("单式注数要在 1 到 20".into());
    }
    if !(6..=33).contains(&spec.pool) {
        return Err("出票池要在 6 到 33".into());
    }
    if spec.overlap > 5 {
        return Err("红球重合上限不能超过 5".into());
    }
    if !(1..=16).contains(&spec.blue_count) {
        return Err("蓝球个数要在 1 到 16".into());
    }
    let pool: Vec<usize> = fc.red_order.iter().take(spec.pool).copied().collect();
    let masks = build_tickets(&pool, &fc.fusion, spec.tickets, spec.overlap);
    let blues = blues_of(fc, spec.blue_count);
    let tickets = masks
        .iter()
        .enumerate()
        .map(|(i, mask)| PlanTicket {
            red: mask_nums(*mask),
            blue: blues[i % blues.len()],
        })
        .collect::<Vec<_>>();
    let mut spec = spec.clone();
    if spec.title == "单式多注" {
        spec.title = format!("单式 {} 注", tickets.len());
    }
    if tickets.len() < spec.tickets {
        spec.detail = format!(
            "按融合分和重合上限，实际排出 {} 注，少于要求的 {} 注。",
            tickets.len(),
            spec.tickets
        );
    } else if spec.detail.is_empty() {
        spec.detail = format!(
            "融合前 {} 个红球，{} 注，任意两注红球重合不超过 {}。",
            spec.pool, tickets.len(), spec.overlap
        );
    }
    Ok(finish(&spec, tickets, nums(&pool), vec![], vec![], blues, rescued))
}

fn balanced_plan(fc: &Forecast, spec: &Spec) -> Result<Plan,String> {
    if !(1..=3).contains(&spec.blue_count) || spec.blue_count>spec.tickets {return Err("均衡覆盖蓝球须为1–3个且不多于注数".into());}
    if !(12..=24).contains(&spec.pool) {return Err("均衡覆盖号码池须为12–24红".into());}
    let pool=&fc.red_order[..spec.pool];
    let masks=crate::coverage::balanced_tickets(pool,spec.tickets)?;
    let blues=blues_of(fc,spec.blue_count);
    let tickets=masks.into_iter().enumerate().map(|(i,m)|PlanTicket{red:mask_nums(m),blue:blues[i%blues.len()]}).collect();
    let mut spec=spec.clone();
    if spec.detail.is_empty(){spec.detail=format!("原融合排名前{}红、{}蓝、{}注，完整覆盖所选红球池并均衡单号曝光；固定试验规则，尚未证实预测优势。",spec.pool,spec.blue_count,spec.tickets);}
    Ok(finish(&spec,tickets,nums(pool),vec![],vec![],blues,vec![]))
}

fn compound_red(fc: &Forecast, spec: &Spec, rescued: Vec<u8>) -> Result<Plan, String> {
    if !(7..=10).contains(&spec.red_count) {
        return Err("红球复式这里展开 7 到 10 个红球。再多注数会超过 300。".into());
    }
    if !(1..=5).contains(&spec.blue_count) {
        return Err("红球复式搭配的蓝球个数要在 1 到 5".into());
    }
    let reds = nums(&fc.red_order[..spec.red_count]);
    let blues = blues_of(fc, spec.blue_count);
    let bets = binom(spec.red_count, 6) * blues.len();
    ensure_bets(bets)?;
    let mut tickets = Vec::new();
    for combo in combinations(&reds, 6)? {
        for &blue in &blues {
            tickets.push(PlanTicket { red: arr6(&combo), blue });
        }
    }
    let mut spec = spec.clone();
    if spec.title == "红球复式" {
        spec.title = format!("红球复式 {} 码", spec.red_count);
    }
    if spec.detail.is_empty() {
        spec.detail = format!("融合前 {} 个红球展开，蓝球 {} 个。", spec.red_count, blues.len());
    }
    Ok(finish(&spec, tickets, reds, vec![], vec![], blues, rescued))
}

fn compound_blue(fc: &Forecast, spec: &Spec, rescued: Vec<u8>) -> Result<Plan, String> {
    if !(2..=16).contains(&spec.blue_count) {
        return Err("蓝球复式要选 2 到 16 个蓝球".into());
    }
    let reds = nums(&fc.red_order[..6]);
    let blues = blues_of(fc, spec.blue_count);
    let tickets = blues
        .iter()
        .map(|&blue| PlanTicket { red: arr6(&reds), blue })
        .collect();
    let mut spec = spec.clone();
    if spec.title == "蓝球复式" {
        spec.title = format!("蓝球复式 {} 码", blues.len());
    }
    if spec.detail.is_empty() {
        spec.detail = format!("融合前 6 个红球固定，配蓝球前 {} 名。", blues.len());
    }
    Ok(finish(&spec, tickets, reds, vec![], vec![], blues, rescued))
}

fn dantuo(fc: &Forecast, spec: &Spec, rescued: Vec<u8>) -> Result<Plan, String> {
    if !(1..=5).contains(&spec.bankers) {
        return Err("胆码要在 1 到 5 个".into());
    }
    let need = 6 - spec.bankers;
    if spec.drags < need.max(2) || spec.bankers + spec.drags > 16 {
        return Err(format!("拖码至少 {need} 个，胆码加拖码不超过 16"));
    }
    if !(1..=5).contains(&spec.blue_count) {
        return Err("胆拖搭配的蓝球个数要在 1 到 5".into());
    }
    let bankers = nums(&fc.red_order[..spec.bankers]);
    let drags = nums(&fc.red_order[spec.bankers..spec.bankers + spec.drags]);
    let blues = blues_of(fc, spec.blue_count);
    let bets = binom(spec.drags, need) * blues.len();
    ensure_bets(bets)?;
    let mut tickets = Vec::new();
    for combo in combinations(&drags, need)? {
        let mut red = bankers.clone();
        red.extend_from_slice(&combo);
        red.sort_unstable();
        for &blue in &blues {
            tickets.push(PlanTicket { red: arr6(&red), blue });
        }
    }
    let mut spec = spec.clone();
    if spec.title == "胆拖" {
        spec.title = format!("{} 胆 {} 拖", spec.bankers, spec.drags);
    }
    if spec.detail.is_empty() {
        spec.detail = format!("融合前 {} 个红球为胆，其后 {} 个为拖。", spec.bankers, spec.drags);
    }
    let mut reds = bankers.clone();
    reds.extend_from_slice(&drags);
    Ok(finish(&spec, tickets, reds, bankers, drags, blues, rescued))
}

fn finish(
    spec: &Spec,
    tickets: Vec<PlanTicket>,
    reds: Vec<u8>,
    bankers: Vec<u8>,
    drags: Vec<u8>,
    blues: Vec<u8>,
    rescued: Vec<u8>,
) -> Plan {
    let bet_count = tickets.len();
    Plan {
        spec: spec.clone(),
        mode: spec.mode.to_string(),
        title: spec.title.clone(),
        detail: spec.detail.clone(),
        bet_count,
        cost: bet_count as u32 * 2,
        tickets,
        reds,
        bankers,
        drags,
        blues,
        rescued,
    }
}

fn blues_of(fc: &Forecast, n: usize) -> Vec<u8> {
    fc.blue_order.iter().take(n.max(1).min(16)).map(|i| *i as u8 + 1).collect()
}

fn nums(idx: &[usize]) -> Vec<u8> {
    idx.iter().map(|i| *i as u8 + 1).collect()
}

fn mask_nums(mask: u64) -> [u8; 6] {
    let mut red = [0u8; 6];
    let mut k = 0;
    let mut bits = mask;
    while bits != 0 && k < 6 {
        red[k] = (bits.trailing_zeros() + 1) as u8;
        k += 1;
        bits &= bits - 1;
    }
    red
}

fn arr6(v: &[u8]) -> [u8; 6] {
    let mut red = [0u8; 6];
    for (i, n) in v.iter().take(6).enumerate() {
        red[i] = *n;
    }
    red
}

fn ensure_bets(n: usize) -> Result<(), String> {
    if n == 0 || n > 300 {
        return Err(format!("展开后有 {n} 注，请把号码收到 300 注以内"));
    }
    Ok(())
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c = 1usize;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

fn combinations(nums: &[u8], k: usize) -> Result<Vec<Vec<u8>>, String> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    comb_rec(nums, k, 0, &mut cur, &mut out)?;
    Ok(out)
}

fn comb_rec(nums: &[u8], k: usize, start: usize, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) -> Result<(), String> {
    if cur.len() == k {
        out.push(cur.clone());
        if out.len() > 300 {
            return Err("展开注数超过 300".into());
        }
        return Ok(());
    }
    let need = k - cur.len();
    if start + need > nums.len() {
        return Ok(());
    }
    let last = nums.len() - need;
    for i in start..=last {
        cur.push(nums[i]);
        comb_rec(nums, k, i + 1, cur, out)?;
        cur.pop();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::select::{rank_desc, RescueResult};

    fn sample() -> Forecast {
        let mut fusion = [0.0; 33];
        for i in 0..33 {
            fusion[i] = (33 - i) as f64;
        }
        let mut blue = [0.0; 16];
        for i in 0..16 {
            blue[i] = (16 - i) as f64;
        }
        let mut tickets = Vec::new();
        for start in 0..8 {
            let mut mask = 0u64;
            for j in 0..6 {
                mask |= 1u64 << ((start + j) % 33);
            }
            tickets.push(mask);
        }
        Forecast {
            fusion,
            red_order: rank_desc(&fusion),
            blue,
            blue_order: rank_desc(&blue),
            mild: RescueResult { mask: 0, members: vec![], rescued: vec![0, 1], displaced: vec![] },
            pool: (0..12).collect(),
            tickets,
            ticket_blues: vec![0, 1, 0, 1, 0, 1, 0, 1],
        }
    }

    #[test]
    fn presets_expand_real_tickets() {
        let fc = sample();
        let frozen = build(&fc, &resolve(Some("frozen"), None, None, None, None, None, None, None, None).unwrap()).unwrap();
        assert_eq!(frozen.tickets.len(), 8);
        assert!(frozen.tickets.iter().all(|t| t.red.len() == 6));
        let seven = build(&fc, &resolve(Some("compound7"), None, None, None, None, None, None, None, None).unwrap()).unwrap();
        assert_eq!(seven.bet_count, 7);
        assert_eq!(seven.cost, 14);
        assert_eq!(seven.reds, vec![1, 2, 3, 4, 5, 6, 7]);
        let split = build(&fc, &resolve(Some("disjoint5"), None, None, None, None, None, None, None, None).unwrap()).unwrap();
        assert_eq!(split.tickets.len(), 5);
        for i in 0..split.tickets.len() {
            for j in (i + 1)..split.tickets.len() {
                let a: u64 = split.tickets[i].red.iter().map(|n| 1u64 << (n - 1)).sum();
                let b: u64 = split.tickets[j].red.iter().map(|n| 1u64 << (n - 1)).sum();
                assert_eq!(a & b, 0);
            }
        }
        let drag = build(&fc, &resolve(Some("dantuo"), None, None, None, None, None, None, None, None).unwrap()).unwrap();
        assert_eq!(drag.bet_count, 15);
        assert!(drag.tickets.iter().all(|t| t.red[0] == 1 && t.red[1] == 2));
    }
}
