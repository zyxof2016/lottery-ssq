use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use ssq::audit::{publish, render_score, verify};
use ssq::draw::{fetch_500, load_csv};
use ssq::eval::backtest;
use ssq::protocol::Protocol;
use ssq::report::{render, to_json};

#[derive(Parser)]
#[command(
    name = "ssq",
    about = "双色球严格滚动样本外验证与可审计预测。确认窗、救援规格和出票规则预先冻结，开奖前用哈希留痕。"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// 打印冻结协议原文和哈希
    Protocol,
    /// 下载历史开奖并写成规范 CSV
    Fetch {
        #[arg(long)]
        out: PathBuf,
    },
    /// 在冻结确认窗上评估增量价值、消融、漏斗、蓝球和显著性
    Backtest {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// 用截止期以前的数据发布下一期预测，并追加审计日志
    Publish {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        audit: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        target: Option<String>,
        /// 开奖时刻的 Unix 秒。当前时间晚于它时拒绝发布
        #[arg(long)]
        not_after: Option<u64>,
    },
    /// 复算清单哈希；若 CSV 已包含截止期之后的开奖则计分
    Verify {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        audit: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    if let Err(err) = run() {
        eprintln!("{err}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), String> {
    match Cli::parse().cmd {
        Cmd::Protocol => {
            println!("{}", Protocol::canonical());
            println!("sha256={}", Protocol::sha256());
        }
        Cmd::Fetch { out } => {
            let n = fetch_500(&out)?;
            println!("已写入 {} 期到 {}", n, out.display());
        }
        Cmd::Backtest { data, out, json } => {
            let draws = load_csv(&data)?;
            let report = backtest(&draws)?;
            let text = render(&report);
            if let Some(path) = &out {
                write_text(path, &text)?;
            }
            if let Some(path) = &json {
                write_text(path, &serde_json::to_string_pretty(&to_json(&report)).map_err(|e| e.to_string())?)?;
            }
            print!("{text}");
        }
        Cmd::Publish {
            data,
            audit,
            out,
            target,
            not_after,
        } => {
            let outcome = publish(&data, &audit, &out, target, not_after)?;
            if outcome.already {
                println!(
                    "目标 {} 的留痕已存在，哈希 {}，清单 {}",
                    outcome.target,
                    outcome.content_sha256,
                    outcome.path.display()
                );
            } else {
                println!(
                    "已发布 {}。把这个哈希抄到开奖系统之外：{}",
                    outcome.path.display(),
                    outcome.content_sha256
                );
            }
        }
        Cmd::Verify {
            data,
            manifest,
            audit,
        } => {
            let outcome = verify(&data, &manifest, audit.as_deref())?;
            println!(
                "复算通过。目标 {}，截止期 {}，哈希 {}",
                outcome.target,
                ssq::draw::fmt_issue(outcome.cutoff_issue),
                outcome.content_sha256
            );
            if let Some(card) = outcome.score {
                println!("{}", render_score(&card));
            } else {
                println!("截止期之后还没有开奖记录，本次只核对了留痕。");
            }
        }
    }
    Ok(())
}

fn write_text(path: &std::path::Path, text: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("无法创建目录 {}: {e}", parent.display()))?;
        }
    }
    fs::write(path, text).map_err(|e| format!("无法写入 {}: {e}", path.display()))
}
