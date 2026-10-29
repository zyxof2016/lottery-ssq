from __future__ import annotations

from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
METHOD_LABELS = {"greedy": "贪心覆盖", "genetic": "遗传算法", "pareto": "Pareto多目标", "consensus": "综合筛选"}


def plot_performance_comparison(backtest: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    methods = list(backtest["summary"])
    model_values = [backtest["summary"][method]["model_average_red_hits"] for method in methods]
    matched_values = [backtest["summary"][method]["matched_no_signal_average_red_hits"] for method in methods]
    random_values = [backtest["summary"][method]["random_average_red_hits"] for method in methods]
    positions = np.arange(len(methods))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - width, model_values, width, label="模型组合", color="#c92f45")
    axis.bar(positions, matched_values, width, label="同流程无信号", color="#1f7a61")
    axis.bar(positions + width, random_values, width, label="纯随机票", color="#5c677d")
    axis.set_xticks(positions, [METHOD_LABELS.get(method, method) for method in methods])
    axis.set_ylabel("每注平均红球命中数")
    axis.set_title("严格逐期回测与双基线")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    for position, method in zip(positions, methods):
        significance = backtest["summary"][method].get("significance", {})
        p_value = significance.get("holm_adjusted_p_value")
        label = "Holm p=-" if p_value is None else f"Holm p={float(p_value):.3f}"
        height = max(
            backtest["summary"][method]["model_average_red_hits"],
            backtest["summary"][method]["matched_no_signal_average_red_hits"],
            backtest["summary"][method]["random_average_red_hits"],
        )
        axis.annotate(label, (position, height), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)
    return target


def plot_hit_distribution(backtest: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for method in backtest["summary"]:
        records = backtest["period_records"]
        model = [item["methods"][method]["model"]["average_red_hits"] for item in records]
        matched = [item["methods"][method]["matched_no_signal"]["average_red_hits"] for item in records]
        label = METHOD_LABELS.get(method, method)
        axis.plot(model, marker="o", linewidth=1.5, label=f"{label} 模型")
        axis.plot(matched, linestyle="--", linewidth=1.2, label=f"{label} 无信号")
    axis.set_xlabel("逐期验证轮次")
    axis.set_ylabel("每注平均红球命中数")
    axis.set_title("逐期模型与同流程无信号基线")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)
    return target


def _current_omissions(frame: pd.DataFrame, columns: list[str], universe_size: int) -> np.ndarray:
    omissions = np.zeros(universe_size, dtype=int)
    for number in range(1, universe_size + 1):
        hit_rows = frame.index[frame[columns].eq(number).any(axis=1)].tolist()
        omissions[number - 1] = len(frame) if not hit_rows else len(frame) - 1 - frame.index.get_loc(hit_rows[-1])
    return omissions


def plot_number_trend(frame: pd.DataFrame, path: str | Path, *, display_periods: int = 60) -> Path:
    """Render a compact, history-only trend chart for the report page."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    recent = frame.tail(max(10, min(int(display_periods), len(frame)))).reset_index(drop=True)
    y = np.arange(len(recent))
    fig, (red_axis, blue_axis, omission_axis) = plt.subplots(
        3,
        1,
        figsize=(15, 13),
        gridspec_kw={"height_ratios": [3.8, 2.1, 2.2]},
        constrained_layout=True,
    )
    for row_index, (_, row) in enumerate(recent.iterrows()):
        red_axis.scatter(
            [int(row[f"r{index}"]) for index in range(1, 7)],
            [row_index] * 6,
            color="#c62d43",
            s=24,
            zorder=3,
        )
    red_axis.set_title("历史红球号码走势（每行一期开出位置）")
    red_axis.set_xlim(0.5, 33.5)
    red_axis.set_xticks(range(1, 34))
    red_axis.set_ylabel("历史期次")
    red_axis.set_yticks(y[::max(1, len(recent) // 12)])
    red_axis.set_yticklabels([str(recent.iloc[index]["period"])[-4:] for index in y[::max(1, len(recent) // 12)]])
    red_axis.invert_yaxis()
    red_axis.grid(axis="x", alpha=0.22)
    for boundary in (11.5, 22.5):
        red_axis.axvline(boundary, color="#a66a00", linewidth=1, linestyle="--", alpha=0.8)
    for row_index, (_, row) in enumerate(recent.iterrows()):
        blue_axis.scatter(int(row["blue"]), row_index, color="#1468a8", s=30, zorder=3)
    blue_axis.set_title("历史蓝球号码走势")
    blue_axis.set_xlim(0.5, 16.5)
    blue_axis.set_xticks(range(1, 17))
    blue_axis.set_ylabel("历史期次")
    blue_axis.set_yticks(y[::max(1, len(recent) // 12)])
    blue_axis.set_yticklabels([str(recent.iloc[index]["period"])[-4:] for index in y[::max(1, len(recent) // 12)]])
    blue_axis.invert_yaxis()
    blue_axis.grid(axis="x", alpha=0.22)
    red_omissions = _current_omissions(recent, [f"r{index}" for index in range(1, 7)], 33)
    blue_omissions = _current_omissions(recent, ["blue"], 16)
    red_positions = np.arange(1, 34)
    omission_axis.bar(red_positions, red_omissions, color="#c62d43", alpha=0.82, label="红球当前遗漏")
    omission_axis.set_xlim(0.5, 33.5)
    omission_axis.set_xticks(range(1, 34))
    omission_axis.set_ylabel("距上次出现的期数")
    omission_axis.set_title("当前遗漏（仅描述历史，不表示即将出现）")
    omission_axis.grid(axis="y", alpha=0.22)
    blue_twin = omission_axis.twinx()
    blue_twin.plot(range(1, 17), blue_omissions, color="#1468a8", marker="o", linewidth=1.7, label="蓝球当前遗漏")
    blue_twin.set_ylim(bottom=0)
    blue_twin.set_ylabel("蓝球当前遗漏", color="#1468a8")
    handles, labels = omission_axis.get_legend_handles_labels()
    blue_handles, blue_labels = blue_twin.get_legend_handles_labels()
    omission_axis.legend(handles + blue_handles, labels + blue_labels, loc="upper right")
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(target)
    return target
