use crate::draw::{fmt_date, fmt_issue};
use crate::eval::{Backtest, TestResult};
use crate::protocol::Protocol;
use crate::stats::effect_size_word;
use serde_json::{json, Value};

pub fn render(bt: &Backtest) -> String {
    let mut o = String::new();
    line(&mut o, "双色球滚动验证报告");
    line(
        &mut o,
        "开奖机制是从固定号码集合里均匀抽取。下面所有数字都是相对随机基线的离差，不是中奖承诺，也不是扣除奖池结构之后的期望收益。",
    );
    line(&mut o, "");
    render_conclusion(&mut o, bt);
    render_protocol(&mut o, bt);
    render_topk(&mut o, bt);
    render_stability(&mut o, bt);
    render_ablation(&mut o, bt);
    render_funnel(&mut o, bt);
    render_tradeoff(&mut o, bt);
    render_blue(&mut o, bt);
    render_experts(&mut o, bt);
    render_tests(&mut o, bt);
    render_audit_guide(&mut o, bt);
    line(
        &mut o,
        &format!("计算耗时 {:.2} 秒", bt.elapsed_ms as f64 / 1000.0),
    );
    o
}

pub fn to_json(bt: &Backtest) -> Value {
    json!({
        "protocol_sha256": bt.protocol_sha,
        "data_sha256": bt.data_sha,
        "n": bt.n,
        "confirmation": {
            "start": fmt_issue(bt.confirm_issue_start),
            "end": fmt_issue(bt.confirm_issue_end),
            "n": bt.n_confirm,
            "weight_end": fmt_issue(bt.weight_issue_end),
            "track_n": bt.n_track,
            "track_start": bt.track_issue_start.map(fmt_issue),
            "track_end": bt.track_issue_end.map(fmt_issue),
        },
        "mde": {
            "red_hits": bt.mde_red,
            "red_hits_bonferroni": bt.mde_red_bonf,
            "blue_hit_rate": bt.mde_blue,
            "blue_hit_rate_bonferroni": bt.mde_blue_bonf,
        },
        "weights": bt.robust.weights,
        "blue_weights": bt.robust.blue_weights,
        "best_expert": Protocol::EXPERTS[bt.robust.best_expert],
        "n_eff": bt.robust.n_eff,
        "n_eff_eq": bt.robust.n_eff_eq,
        "mean_score_corr": bt.robust.mean_corr,
        "eigenvalues": bt.robust.eigenvalues,
        "dirichlet_excess": [bt.robust.dirichlet_p05, bt.robust.dirichlet_p50, bt.robust.dirichlet_p95],
        "funnel_mean_alive": bt.funnel.mean_alive,
        "funnel_death_hist": bt.funnel.death_hist,
        "tests": bt.tests.iter().map(test_json).collect::<Vec<_>>(),
        "tradeoff": bt.tradeoff.iter().map(|c| json!({
            "mass": c.mass,
            "avg_hits": c.confirm_avg,
            "pool": c.pool,
            "tickets": c.tickets,
            "overlap": c.overlap,
            "evidence": c.confirm_evidence,
            "coverage": c.confirm_coverage,
            "lift": c.confirm_lift,
            "pareto": c.pareto,
        })).collect::<Vec<_>>(),
    })
}

fn test_json(t: &TestResult) -> Value {
    json!({
        "name": t.name,
        "estimate": t.estimate,
        "ci95": [t.ci_low, t.ci_high],
        "ci_bonferroni": [t.ci_bonf_low, t.ci_bonf_high],
        "effect": t.effect,
        "effect_kind": t.effect_kind,
        "p": t.p,
        "p_holm": t.p_holm,
        "p_bh": t.p_bh,
        "p_block": t.p_block,
        "better_if_positive": t.better_if_positive,
    })
}

fn render_conclusion(o: &mut String, bt: &Backtest) {
    h(o, "结论");
    let sig_good: Vec<_> = bt
        .tests
        .iter()
        .filter(|t| t.p_holm < 0.05 && beneficial(t))
        .collect();
    let sig_bad: Vec<_> = bt
        .tests
        .iter()
        .filter(|t| t.p_holm < 0.05 && harmful(t))
        .collect();
    if sig_good.is_empty() {
        line(
            o,
            "确认窗里，预注册的 25 项检验在 Holm 校正后没有任何一项指向稳定的预测优势。",
        );
    } else {
        line(
            o,
            &format!(
                "确认窗里有 {} 项检验在 Holm 校正后指向有利方向，需要连同效应量和区间一起看：",
                sig_good.len()
            ),
        );
        for t in sig_good {
            line(
                o,
                &format!(
                    "- {} 估计 {}，效应 {}（{}）",
                    t.name,
                    f4(t.estimate),
                    f4(t.effect),
                    effect_size_word(t.effect.abs())
                ),
            );
        }
    }
    if !sig_bad.is_empty() {
        line(o, "另有检验在校正后显著差于基线：");
        for t in sig_bad {
            line(o, &format!("- {}", t.name));
        }
    }
    let curve_sig = bt
        .topk
        .iter()
        .filter(|p| p.p_curve.is_some_and(|q| q < 0.05))
        .count();
    if curve_sig == 0 {
        line(
            o,
            "确认窗 8 个模型、8 个 K 共 64 项累计超额是另一族描述性检验，Holm 校正后也没有一项低于 0.05。未校正区间盖不住 0，不能单独当成发现。",
        );
    } else {
        line(
            o,
            &format!(
                "确认窗 Top-K 曲线的 64 项累计超额里，Holm 校正后有 {curve_sig} 项低于 0.05。这族不在预注册的 25 项里。"
            ),
        );
    }
    line(
        o,
        &format!(
            "确认窗 {} 期、红球 K=15 的 80% 功效可检出超额约为 {:.3} 个号（Bonferroni 后约为 {:.3}）。蓝球命中率差值的对应门槛约为 {:.3}（校正后 {:.3}）。小于这个量级的波动应看成噪声。",
            bt.n_confirm, bt.mde_red, bt.mde_red_bonf, bt.mde_blue, bt.mde_blue_bonf
        ),
    );
    let dev_out = bt
        .stability
        .iter()
        .filter(|s| s.window == "开发窗" && (s.ci_low > 0.0 || s.ci_high < 0.0))
        .map(|s| format!("{}（超额 {}）", s.model, f4(s.mean_excess)))
        .collect::<Vec<_>>();
    if dev_out.is_empty() {
        line(o, "开发窗 K=15 的区间也都盖住 0。");
    } else {
        line(
            o,
            &format!(
                "开发窗里这些 K=15 区间不盖住 0：{}。开发窗不进预注册族；确认窗没有复现同方向的优势。",
                dev_out.join("、")
            ),
        );
    }
}

fn render_protocol(o: &mut String, bt: &Backtest) {
    h(o, "冻结协议与窗口");
    line(o, &format!("协议哈希 {}", bt.protocol_sha));
    line(o, &format!("数据哈希 {}", bt.data_sha));
    line(
        o,
        &format!(
            "样本 {} 期，{}（{}）至 {}（{}）。预热 {} 期不评分。",
            bt.n,
            fmt_issue(bt.first_issue),
            fmt_date(bt.first_date),
            fmt_issue(bt.last_issue),
            fmt_date(bt.last_date),
            Protocol::BURN_IN
        ),
    );
    line(
        o,
        &format!(
            "确认窗起点冻结为 {}，长度 {}。开发窗到 {} 为止，共 {} 个样本外期，只用于拟合融合权重和指定对照专家。确认窗 {} 至 {}，共 {} 期，不参与规则选择，也不回流进权重。",
            fmt_issue(Protocol::CONFIRM_START_ISSUE),
            Protocol::CONFIRMATION_LEN,
            fmt_issue(bt.weight_issue_end),
            bt.n_weight,
            fmt_issue(bt.confirm_issue_start),
            fmt_issue(bt.confirm_issue_end),
            bt.n_confirm
        ),
    );
    if bt.n_track == 0 {
        line(
            o,
            "确认窗终点之后没有跟踪期。以后追加的开奖只记入跟踪窗，不移动确认窗，也不进入 Holm 族。",
        );
    } else {
        line(
            o,
            &format!(
                "跟踪窗 {} 至 {}，共 {} 期。融合超额 {}，温和救援超额 {}。跟踪窗使用冻结权重，不计进预注册族。",
                fmt_issue(bt.track_issue_start.unwrap_or(0)),
                fmt_issue(bt.track_issue_end.unwrap_or(0)),
                bt.n_track,
                opt(bt.track_fusion_excess),
                opt(bt.track_mild_excess)
            ),
        );
    }
    line(
        o,
        &format!(
            "发布路径写死为：加权融合，温和救援（专家 Top-{} 中至少 {} 票、最多换入 {} 个、只从融合第 {}-{} 名里救），出票池 {} 个并预留救援席，{} 张票，任意两张红球重合不超过 {}。漏斗支路按融合分数保留定稿集累计质量的 {:.0}%，只用于观察，不进入出票。排序使用 {:.0e} 的分数步长，权重使用 {:.0e}。",
            Protocol::RESCUE_M,
            Protocol::RESCUE_Q,
            Protocol::RESCUE_S,
            Protocol::PRIMARY_K + 1,
            Protocol::FUSION_WIDE_K,
            Protocol::POOL,
            Protocol::TICKETS,
            Protocol::MAX_OVERLAP,
            Protocol::EVIDENCE_MASS * 100.0,
            Protocol::SCORE_QUANTUM,
            Protocol::WEIGHT_QUANTUM
        ),
    );
}

fn render_topk(o: &mut String, bt: &Backtest) {
    h(o, "单模型 Top-K 的独立增量");
    line(
        o,
        "累计超额 = 命中数 − 6K/33。带宽独立增量（IIV）只看新进入这一带的号码，并减去这一带自己的随机期望。块为正是每 26 期一块的 IIV 均值大于 0 的比例。开发窗的加权融合用了同窗权重，只能对照。",
    );
    line(
        o,
        "模型 | 窗 | K | 命中 | 期望 | 超额 | 95%区间 | 带 | IIV | IIV区间 | 块为正 | 斜率 | 曲线Holm",
    );
    for p in &bt.topk {
        if p.window == "开发窗" && p.k != Protocol::PRIMARY_K && p.model == "加权融合" {
            continue;
        }
        if p.window == "开发窗" && p.k != Protocol::PRIMARY_K {
            continue;
        }
        line(
            o,
            &format!(
                "{} | {} | {} | {} | {} | {} | [{}, {}] | {}-{} | {} | [{}, {}] | {} | {} | {}",
                p.model,
                p.window,
                p.k,
                f4(p.mean_hits),
                f4(p.expected),
                f4(p.excess),
                f4(p.ci_low),
                f4(p.ci_high),
                p.band_lo,
                p.band_hi,
                f4(p.iiv),
                f4(p.iiv_low),
                f4(p.iiv_high),
                f4(p.iiv_pos_frac),
                opt(p.iiv_slope),
                p.p_curve.map(fp).unwrap_or_else(|| "—".into()),
            ),
        );
    }
    line(
        o,
        "开发窗只保留主终点 K=15，避免拿整条曲线做事后挑选。确认窗给出全部预注册带宽。曲线 Holm 只在确认窗的 64 项累计超额里校正，不并入主检验的 25 项。",
    );
}

fn render_stability(o: &mut String, bt: &Backtest) {
    h(o, "K=15 超额的时间稳定性");
    line(
        o,
        "后半减前半 > 0 表示后半段超额更高。斜率是块均值对块序号的普通最小二乘斜率。自相关是逐期超额的滞后 1 相关。",
    );
    line(
        o,
        "模型 | 窗 | 超额 | 区间 | 块为正 | 块标准差 | 斜率 | 斜率标准误 | 自相关 | 后半-前半 | 块极差 | 块数",
    );
    for s in &bt.stability {
        line(
            o,
            &format!(
                "{} | {} | {} | [{}, {}] | {} | {} | {} | {} | {} | {} | [{}, {}] | {}",
                s.model,
                s.window,
                f4(s.mean_excess),
                f4(s.ci_low),
                f4(s.ci_high),
                f4(s.pos_frac),
                f4(s.block_sd),
                opt(s.slope),
                opt(s.slope_se),
                f4(s.acf1),
                f4(s.half_delta),
                f4(s.block_min),
                f4(s.block_max),
                s.n_blocks
            ),
        );
    }
}

fn render_ablation(o: &mut String, bt: &Backtest) {
    h(o, "融合与救援通道消融");
    line(
        o,
        "全部系统都收成 15 个红球，所以可以直接比超额。温和救援是发布规格；强救援只做对照。开发窗超额里，融合和救援含有同窗权重，单模型没有这个问题。",
    );
    line(o, "系统 | 确认窗命中 | 期望 | 确认窗超额 | 95%区间 | 开发窗超额");
    let expected = 6.0 * Protocol::PRIMARY_K as f64 / 33.0;
    for row in &bt.ablation {
        line(
            o,
            &format!(
                "{} | {} | {} | {} | [{}, {}] | {}",
                row.name,
                f4(row.confirm_hits),
                f4(expected),
                f4(row.confirm_excess),
                f4(row.ci_low),
                f4(row.ci_high),
                f4(row.dev_excess)
            ),
        );
    }
}

fn render_funnel(o: &mut String, bt: &Backtest) {
    h(o, "真实红球从模型到票组的生存漏斗");
    line(o, "数字是确认窗每期 6 个真实红球里，平均还有几个留在该阶段。条件存活是后一阶段人数除以前一阶段人数。");
    line(o, "阶段 | 平均存活 | 条件存活");
    for (i, name) in bt.funnel.stage_names.iter().enumerate() {
        let cond = if i == 0 {
            format!("占 6 个的 {}", f4(bt.funnel.pooled_conditional[i]))
        } else {
            f4(bt.funnel.pooled_conditional[i])
        };
        line(
            o,
            &format!(
                "{} | {} | {}",
                name,
                f4(bt.funnel.mean_alive[i]),
                cond
            ),
        );
    }
    line(o, "真实红球停在各深度的平均个数（0 表示系统可见之前就离开，5 表示出现在票上）：");
    line(
        o,
        &bt.funnel
            .death_hist
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{i}:{}", f4(*c)))
            .collect::<Vec<_>>()
            .join(" | "),
    );
    let f = &bt.funnel;
    line(
        o,
        &format!(
            "独立关卡召回：专家并集 {:.3}，多数专家 {:.3}，融合 Top-15 {:.3}，出票池 {:.3}，票组 {:.3}。",
            f.expert_union_recall, f.majority_recall, f.fusion15_recall, f.pool_recall, f.ticket_recall
        ),
    );
    line(
        o,
        &format!(
            "救援换入的真实红球平均 {:.3} 个，被挤出的真实红球平均 {:.3} 个。若在定稿集上按融合分数保留累计质量的 {:.0}%，平均保留 {:.2} 个号，其中真实红球 {:.3} 个。",
            f.rescued_true,
            f.displaced_true,
            Protocol::EVIDENCE_MASS * 100.0,
            f.evidence_keep,
            f.evidence_recall
        ),
    );
    line(
        o,
        &format!(
            "生产票组平均 {:.2} 张。最佳票红球命中 {:.3}，条件零假设 {:.3}，购买同样张数的独立均匀票 {:.3}。",
            f.mean_tickets, f.mean_best_hits, f.mean_cond_null, f.mean_iid_null
        ),
    );
    line(
        o,
        &format!(
            "蓝球：融合 Top-1 命中 {:.3}，Top-3 命中 {:.3}，票面出现过的蓝球命中 {:.3}（这些蓝球在均匀下的期望 {:.3}）。",
            f.blue_top1, f.blue_top3, f.blue_ticket_hit, f.blue_ticket_null
        ),
    );
}

fn render_tradeoff(o: &mut String, bt: &Backtest) {
    h(o, "号码证据与票组覆盖");
    line(
        o,
        "网格只使用加权融合名次，不含救援，以免和上一节混在一起。质量比例是融合 Top-15 里按分数从高到低累计到该比例后留下的前缀，再按池大小截断。证据保留是票面覆盖号码的融合概率质量除以融合 Top-15 的质量。覆盖率的分母也是这 15 个号。最佳票超额对比的是从全部 33 个号里买同样张数的独立均匀票。多张票的超额明显为负，是因为票被关在前 15 个号里，盖不住全号段随机票的最大值；票均命中仍贴着 6×6/33≈1.09，那一列才表示单张票有没有排序优势。帕累托按证据、号码覆盖和票均命中三者比较，只描述确认窗，不回写发布规则。对照窗是确认窗之前的同样长度。",
    );
    line(
        o,
        "质量比例 | 池 | 张数 | 重合上限 | 证据 | 号码覆盖 | 号码对覆盖 | 票均命中 | 最佳票超额 | 填满率 | 对照证据 | 对照超额 | 帕累托",
    );
    for c in &bt.tradeoff {
        line(
            o,
            &format!(
                "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                f4(c.mass),
                c.pool,
                c.tickets,
                c.overlap,
                f4(c.confirm_evidence),
                f4(c.confirm_coverage),
                f4(c.confirm_pairs),
                f4(c.confirm_avg),
                f4(c.confirm_lift),
                f4(c.confirm_fill),
                f4(c.repl_evidence),
                f4(c.repl_lift),
                if c.pareto { "是" } else { "" }
            ),
        );
    }
}

fn render_blue(o: &mut String, bt: &Backtest) {
    h(o, "蓝球策略与两个基线");
    line(
        o,
        "命中率的均匀基线是 1/16，Top-3 是 3/16。匹配随机保持这期概率向量的形状，只打乱它和开奖号的对应关系；它的命中率期望仍是 k/16，所以匹配基线体现在对数损失和 Brier 上。损失越低越好。",
    );
    line(
        o,
        "策略 | Top1 | Wilson区间 | Top3 | 对数损失 | 均匀损失 | 匹配损失 | Brier | 均匀Brier | 匹配Brier | Cohen h",
    );
    for b in &bt.blue {
        line(
            o,
            &format!(
                "{} | {} | [{}, {}] | {} | {} | {} | {} | {} | {} | {} | {}",
                b.name,
                f4(b.hit1),
                f4(b.wilson_low),
                f4(b.wilson_high),
                f4(b.hit3),
                f4(b.logloss),
                f4(b.uniform_logloss),
                f4(b.matched_logloss),
                f4(b.brier),
                f4(b.uniform_brier),
                f4(b.matched_brier),
                f4(b.cohen_h)
            ),
        );
    }
}

fn render_experts(o: &mut String, bt: &Backtest) {
    h(o, "专家相关、有效信息源与权重稳健性");
    let r = &bt.robust;
    line(o, "红球冻结权重：");
    for (i, w) in r.weights.iter().enumerate() {
        line(o, &format!("- {} {:.4}", Protocol::EXPERTS_ZH[i], w));
    }
    line(o, "蓝球冻结权重：");
    for (i, w) in r.blue_weights.iter().enumerate() {
        line(o, &format!("- {} {:.4}", Protocol::BLUES_ZH[i], w));
    }
    line(
        o,
        &format!(
            "开发窗 K=15 超额最高的单模型是 {}，只用于配对对照。",
            Protocol::EXPERTS_ZH[r.best_expert]
        ),
    );
    let eq_note = match r.n_eff_eq {
        Some(v) if v <= 7.05 => format!("等相关近似 {}", f4(v)),
        Some(v) => format!("等相关近似 {v:.4} 已经越出专家个数，这个近似作废"),
        None => "各对相关差得太大，或平均相关使公式越界，等相关近似作废，不以它当作信息源个数".into(),
    };
    line(
        o,
        &format!(
            "分数向量平均相关 {:.3}，平均绝对相关 {:.3}。有效独立源以特征值参与比 {:.3} 为准。{eq_note}。",
            r.mean_corr, r.mean_abs_corr, r.n_eff
        ),
    );
    line(
        o,
        &format!(
            "相关矩阵特征值（大到小）：{}",
            r.eigenvalues.iter().map(|v| f4(*v)).collect::<Vec<_>>().join(", ")
        ),
    );
    line(o, "分数相关（行和列顺序与上面的红球专家一致）：");
    for row in &r.corr_score {
        line(o, &row.iter().map(|v| f4(*v)).collect::<Vec<_>>().join(" "));
    }
    line(o, "确认窗 K=15 超额命中的相关。它含有同一期开奖带来的共同波动，不能单独当成信息源个数。");
    for row in &r.corr_hit {
        line(o, &row.iter().map(|v| f4(*v)).collect::<Vec<_>>().join(" "));
    }
    line(
        o,
        &format!(
            "冻结权重的确认窗超额 {:.4}。Dirichlet 扰动（κ={:.0}，{} 次）的 5/50/95 分位是 {} / {} / {}。",
            r.frozen_excess,
            Protocol::DIRICHLET_KAPPA,
            Protocol::DIRICHLET_DRAWS,
            f4(r.dirichlet_p05),
            f4(r.dirichlet_p50),
            f4(r.dirichlet_p95)
        ),
    );
    line(o, "留一专家并重新归一后的确认窗超额：");
    for row in &r.loo {
        line(o, &format!("- 去掉 {} ： {}", row.dropped, f4(row.excess)));
    }
    line(
        o,
        &format!(
            "若把确认窗结果回流进权重，相对冻结权重的 L1 漂移平均 {:.4}，最大 {:.4}。这列只诊断权重会不会被新数据甩开，不进入主评分。",
            r.l1_mean, r.l1_max
        ),
    );
}

fn render_tests(o: &mut String, bt: &Backtest) {
    h(o, "预注册推断");
    line(
        o,
        &format!(
            "族大小 {}。主 p 做 Holm 和 Benjamini-Hochberg。Bonferroni 区间用 α/{}。块符号翻转 p 只作参照，不进这族。效应 dz 是均值除以逐期标准差，h 是比率的 Cohen h。",
            bt.tests.len(),
            bt.tests.len()
        ),
    );
    line(
        o,
        "检验 | 估计 | 95%区间 | Bonferroni区间 | 效应 | 量级 | 主p | Holm | BH | 有利方向",
    );
    for t in &bt.tests {
        line(
            o,
            &format!(
                "{} | {} | [{}, {}] | [{}, {}] | {} {} | {} | {} | {} | {} | {}",
                t.name,
                f4(t.estimate),
                f4(t.ci_low),
                f4(t.ci_high),
                f4(t.ci_bonf_low),
                f4(t.ci_bonf_high),
                t.effect_kind,
                f4(t.effect),
                effect_size_word(t.effect.abs()),
                fp(t.p),
                fp(t.p_holm),
                fp(t.p_bh),
                if t.better_if_positive { "正" } else { "负" }
            ),
        );
        line(o, &format!("  {}", t.note));
    }
}

fn render_audit_guide(o: &mut String, bt: &Backtest) {
    h(o, "开奖前留痕与复算");
    line(
        o,
        "专家状态用到截止期为止的全部开奖。融合权重停在确认窗起点之前，与回测同一套；截止期若还没到 25111，才用当时已有的样本外结果。分数在排序前收到 1e-9，权重收到 1e-12。票数、救援和重合上限仍用冻结常数，不会改成网格里看起来最好的那一格。",
    );
    line(o, "开奖前执行：");
    line(
        o,
        "ssq publish --data data/ssq.csv --audit audit/log.jsonl --out publish --target 下一期期号 --not-after 开奖时刻的Unix秒",
    );
    line(
        o,
        "命令把规范文本的 SHA-256 写入清单和只追加的审计日志。同一目标若已有不同哈希，程序拒绝覆盖。截止时间一过就拒绝再发布。",
    );
    line(o, "开奖后把新的一期追加进 CSV，再执行：");
    line(o, "ssq verify --data data/ssq.csv --manifest publish/对应清单.json --audit audit/log.jsonl");
    line(
        o,
        "复算使用清单里的截止期以前的数据。哈希一致才对之后的第一期计分。请在开奖前把 content_sha256 抄到彩票系统之外的介质上。",
    );
    line(
        o,
        &format!(
            "本次回测数据截止于 {}，协议哈希 {}。",
            fmt_issue(bt.last_issue),
            bt.protocol_sha
        ),
    );
}

fn beneficial(t: &TestResult) -> bool {
    if t.estimate.abs() < 1e-12 {
        return false;
    }
    if t.better_if_positive {
        t.estimate > 0.0
    } else {
        t.estimate < 0.0
    }
}

fn harmful(t: &TestResult) -> bool {
    if t.estimate.abs() < 1e-12 {
        return false;
    }
    !beneficial(t)
}

fn h(o: &mut String, title: &str) {
    o.push('\n');
    o.push_str(title);
    o.push('\n');
}

fn line(o: &mut String, text: &str) {
    o.push_str(text);
    o.push('\n');
}

fn f4(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.4}")
    } else {
        "—".into()
    }
}

fn opt(x: Option<f64>) -> String {
    x.map(f4).unwrap_or_else(|| "—".into())
}

fn fp(p: f64) -> String {
    if p < 1e-4 {
        format!("{p:.2e}")
    } else {
        format!("{p:.4}")
    }
}
