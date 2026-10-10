use crate::hash::sha256_hex;

/// 预先冻结的协议。改任何一个常数都会改变协议哈希，从而使旧清单无法复算通过。
pub struct Protocol;

impl Protocol {
    pub const VERSION: u32 = 1;
    pub const BURN_IN: usize = 150;
    pub const CONFIRMATION_LEN: usize = 156;
    pub const EWMA_ALPHA: f64 = 0.08;
    pub const WEIGHT_SHRINKAGE: f64 = 0.5;
    pub const WEIGHT_TAU: f64 = 0.05;
    pub const PRIMARY_K: usize = 15;
    pub const KS: [usize; 8] = [6, 8, 10, 12, 15, 18, 22, 27];
    pub const NOMINATION_M: usize = 12;
    pub const MAJORITY_VOTES: usize = 4;
    pub const FUSION_WIDE_K: usize = 20;
    pub const RESCUE_Q: usize = 3;
    pub const RESCUE_S: usize = 2;
    pub const RESCUE_M: usize = 8;
    pub const RESCUE_Q_AGG: usize = 2;
    pub const RESCUE_S_AGG: usize = 4;
    pub const RESCUE_M_AGG: usize = 10;
    pub const POOL: usize = 12;
    pub const TICKETS: usize = 8;
    pub const MAX_OVERLAP: u32 = 4;
    pub const BLUE_SLOTS: usize = 2;
    pub const EVIDENCE_RHO: f64 = 0.55;
    pub const ZONE_WINDOW: usize = 30;
    pub const STABILITY_BLOCK: usize = 26;
    pub const SIGNFLIP_BLOCK: usize = 20;
    pub const SIGNFLIP_B: usize = 2000;
    pub const SEED: u64 = 0x5A17_C0DE_F00D;
    pub const DIRICHLET_KAPPA: f64 = 40.0;
    pub const DIRICHLET_DRAWS: usize = 400;
    pub const DIRICHLET_FLOOR: f64 = 0.02;
    pub const TICKET_MC_B: usize = 4000;

    pub const EXPERTS: [&'static str; 7] = [
        "frequency",
        "ewma",
        "overdue",
        "hot",
        "cooccur",
        "zone",
        "repeat",
    ];
    pub const EXPERTS_ZH: [&'static str; 7] = [
        "长期频率",
        "指数近期",
        "遗漏回补",
        "近期热号",
        "上期共现",
        "区间均衡",
        "上期重号",
    ];
    pub const BLUES: [&'static str; 6] = [
        "frequency",
        "ewma",
        "overdue",
        "hot",
        "anti_repeat",
        "markov",
    ];
    pub const BLUES_ZH: [&'static str; 6] = [
        "长期频率",
        "指数近期",
        "遗漏回补",
        "近期热号",
        "回避上期",
        "转移矩阵",
    ];

    pub fn canonical() -> String {
        let ks = Self::KS
            .iter()
            .map(|k| k.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "\
ssq-protocol-v{version}
burn_in={burn}
confirmation_len={confirm}
ewma_alpha={ewma:.8}
weight_shrinkage={shrink:.8}
weight_tau={tau:.8}
primary_k={pk}
ks={ks}
nomination_m={nom_m}
majority_votes={maj}
fusion_wide_k={wide}
rescue_q={rq}
rescue_s={rs}
rescue_m={rm}
rescue_q_agg={rqa}
rescue_s_agg={rsa}
rescue_m_agg={rma}
pool={pool}
tickets={tickets}
max_overlap={overlap}
blue_slots={blue_slots}
evidence_rho={rho:.8}
zone_window={zone}
stability_block={block}
signflip_block={sb}
signflip_b={sbb}
seed={seed}
dirichlet_kappa={kappa:.8}
dirichlet_draws={dd}
dirichlet_floor={floor:.8}
ticket_mc_b={tmc}
experts={experts}
blues={blues}
",
            version = Self::VERSION,
            burn = Self::BURN_IN,
            confirm = Self::CONFIRMATION_LEN,
            ewma = Self::EWMA_ALPHA,
            shrink = Self::WEIGHT_SHRINKAGE,
            tau = Self::WEIGHT_TAU,
            pk = Self::PRIMARY_K,
            ks = ks,
            nom_m = Self::NOMINATION_M,
            maj = Self::MAJORITY_VOTES,
            wide = Self::FUSION_WIDE_K,
            rq = Self::RESCUE_Q,
            rs = Self::RESCUE_S,
            rm = Self::RESCUE_M,
            rqa = Self::RESCUE_Q_AGG,
            rsa = Self::RESCUE_S_AGG,
            rma = Self::RESCUE_M_AGG,
            pool = Self::POOL,
            tickets = Self::TICKETS,
            overlap = Self::MAX_OVERLAP,
            blue_slots = Self::BLUE_SLOTS,
            rho = Self::EVIDENCE_RHO,
            zone = Self::ZONE_WINDOW,
            block = Self::STABILITY_BLOCK,
            sb = Self::SIGNFLIP_BLOCK,
            sbb = Self::SIGNFLIP_B,
            seed = Self::SEED,
            kappa = Self::DIRICHLET_KAPPA,
            dd = Self::DIRICHLET_DRAWS,
            floor = Self::DIRICHLET_FLOOR,
            tmc = Self::TICKET_MC_B,
            experts = Self::EXPERTS.join(","),
            blues = Self::BLUES.join(","),
        )
    }

    pub fn sha256() -> String {
        sha256_hex(Self::canonical().as_bytes())
    }
}
