from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def show_risk_notice(
    enabled: bool = True,
    watermark: str = "仅供历史数据概率实验，不构成投注建议",
) -> None:
    if enabled:
        print(f"【风险提示】{watermark}")


def append_audit_log(path: str | Path, action: str, watermark: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    action_labels = {
        "serve": "启动网页服务",
        "run": "运行完整实验",
        "clean": "清洗历史数据",
        "backtest": "运行严格回测",
        "predict": "生成实验候选",
        "dashboard_run": "网页端运行完整实验",
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\t{action_labels.get(action, action)}\t事件={action}\t{watermark}\n")


def require_random_baseline(result: dict) -> None:
    if "summary" not in result:
        raise RuntimeError("实验结果缺少随机对照汇总，禁止输出")
    for method, metrics in result["summary"].items():
        required = {
            "model_average_red_hits",
            "matched_no_signal_average_red_hits",
            "random_average_red_hits",
        }
        if not required.issubset(metrics):
            raise RuntimeError(f"{method} 缺少同流程无信号或纯随机基线，禁止输出")
