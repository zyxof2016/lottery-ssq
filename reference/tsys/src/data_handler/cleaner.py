from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


RED_COLUMNS = [f"r{i}" for i in range(1, 7)]
REQUIRED_COLUMNS = ["period", "date", *RED_COLUMNS, "blue"]


@dataclass(frozen=True)
class CleanResult:
    data: pd.DataFrame
    report: dict[str, Any]
    quarantine: list[dict[str, Any]]


class DataQualityError(ValueError):
    """Raised when input data cannot be used for a formal experiment."""


def _parse_date(value: Any) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DataQualityError(f"无法解析开奖日历日期: {value!r}")
    return parsed.date()


def _blackout_ranges(values: Iterable[Any] | None) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    for item in values or []:
        if isinstance(item, dict):
            start_value = item.get("start")
            end_value = item.get("end", start_value)
        elif isinstance(item, (list, tuple)) and item:
            start_value = item[0]
            end_value = item[1] if len(item) > 1 else item[0]
        else:
            start_value = item
            end_value = item
        start = _parse_date(start_value)
        end = _parse_date(end_value)
        if end < start:
            raise DataQualityError(f"开奖休市区间结束日期早于开始日期: {item!r}")
        ranges.append((start, end))
    return ranges


def _in_blackout_ranges(value: date, ranges: Iterable[tuple[date, date]]) -> bool:
    return any(start <= value <= end for start, end in ranges)


def _scheduled_dates(
    year: int,
    count: int,
    weekdays: set[int],
    blackout_ranges: Iterable[tuple[date, date]] = (),
) -> list[date]:
    current = date(year, 1, 1)
    values: list[date] = []
    while len(values) < count:
        if current.weekday() in weekdays and not _in_blackout_ranges(current, blackout_ranges):
            values.append(current)
        current += timedelta(days=1)
    return values


def _fill_dates(
    frame: pd.DataFrame,
    weekdays: Iterable[int],
    blackout_ranges: Iterable[Any] | None = None,
) -> int:
    missing = frame["date"].isna()
    if not missing.any():
        return 0
    weekday_set = {int(value) for value in weekdays}
    blackout = _blackout_ranges(blackout_ranges)
    for year_text, group in frame.loc[missing].groupby(frame.loc[missing, "period"].str[:4]):
        year = int(year_text)
        sequences = frame.loc[group.index, "period"].str[4:].astype(int)
        calendar = _scheduled_dates(year, int(sequences.max()), weekday_set, blackout)
        frame.loc[group.index, "date"] = pd.to_datetime([calendar[value - 1] for value in sequences])
    return int(missing.sum())


def assert_clean_sequence(frame: pd.DataFrame) -> None:
    missing = [column for column in [*REQUIRED_COLUMNS, "idx"] if column not in frame.columns]
    if missing:
        raise ValueError(f"清洗数据缺少字段: {missing}")
    if frame["idx"].tolist() != list(range(len(frame))):
        raise ValueError("idx 必须从 0 开始连续递增")
    if not frame["period"].astype(int).is_monotonic_increasing:
        raise ValueError("期号必须严格按时间升序排列")
    if frame["period"].duplicated().any():
        raise ValueError("清洗数据仍包含重复期号")


def assert_left_truncated(history: pd.DataFrame, target_idx: int) -> None:
    assert_clean_sequence(history)
    if history.empty:
        raise ValueError("历史窗口不能为空")
    latest = int(history["idx"].max())
    if latest >= int(target_idx):
        raise RuntimeError(
            f"检测到未来数据泄漏: 特征历史最大 idx={latest}，目标 idx={target_idx}"
        )
    if latest + 1 != int(target_idx):
        raise RuntimeError(
            f"目标期必须是训练截止期的下一期: 历史最大 idx={latest}，目标 idx={target_idx}"
        )


def clean_history(
    raw: pd.DataFrame,
    *,
    fill_missing_dates: bool = True,
    draw_weekdays: Iterable[int] = (1, 3, 6),
    draw_blackout_ranges: Iterable[Any] | None = None,
    invalid_row_policy: str = "reject",
    validate_source_dates: bool = False,
) -> CleanResult:
    """Normalize history while preserving every rejected row in a quarantine report."""
    if invalid_row_policy not in {"reject", "quarantine"}:
        raise ValueError("invalid_row_policy 仅支持 reject 或 quarantine")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise DataQualityError(f"原始数据缺少标准字段: {missing_columns}")

    frame = raw[REQUIRED_COLUMNS].copy()
    input_rows = len(frame)
    frame["_source_row"] = range(2, len(frame) + 2)
    frame["period"] = frame["period"].astype("string").str.strip()
    supplied_date = frame["date"].notna() & frame["date"].astype("string").str.strip().ne("")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date_provenance"] = "source"
    frame.loc[~supplied_date, "date_provenance"] = "missing"
    for column in [*RED_COLUMNS, "blue"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    issues: dict[int, list[str]] = {}

    def add_issue(mask: pd.Series, reason: str) -> None:
        for source_row in frame.loc[mask, "_source_row"].astype(int):
            issues.setdefault(int(source_row), []).append(reason)

    essential = ["period", *RED_COLUMNS, "blue"]
    add_issue(frame[essential].isna().any(axis=1), "missing_required_value")
    add_issue(~frame["period"].str.fullmatch(r"\d{7}", na=False), "invalid_period_format")
    add_issue(supplied_date & frame["date"].isna(), "invalid_source_date")
    sequence = pd.to_numeric(frame["period"].str[-3:], errors="coerce")
    add_issue(~sequence.between(1, 999), "invalid_period_sequence")
    valid_red_range = frame[RED_COLUMNS].apply(lambda column: column.between(1, 33)).all(axis=1)
    valid_blue = frame["blue"].between(1, 16)
    unique_reds = frame[RED_COLUMNS].nunique(axis=1).eq(6)
    add_issue(~valid_red_range, "red_out_of_range")
    add_issue(~valid_blue, "blue_out_of_range")
    add_issue(~unique_reds, "duplicate_red_numbers")

    quarantine = [
        {
            "source_row": source_row,
            "period": str(raw.iloc[source_row - 2].get("period", "")),
            "reasons": sorted(set(reasons)),
        }
        for source_row, reasons in sorted(issues.items())
    ]
    if quarantine and invalid_row_policy == "reject":
        preview = "; ".join(
            f"row {item['source_row']} ({item['period']}): {','.join(item['reasons'])}"
            for item in quarantine[:5]
        )
        suffix = " ..." if len(quarantine) > 5 else ""
        raise DataQualityError(
            "检测到无效历史记录；正式实验已停止，需修正后重新导入。" + preview + suffix
        )
    if issues:
        frame = frame.loc[~frame["_source_row"].isin(issues)].copy()
    for column in [*RED_COLUMNS, "blue"]:
        frame[column] = frame[column].astype("int64")

    sorted_reds = frame[RED_COLUMNS].apply(lambda row: sorted(row.tolist()), axis=1, result_type="expand")
    sorted_reds.columns = RED_COLUMNS
    frame[RED_COLUMNS] = sorted_reds.to_numpy(dtype="int64")
    before_duplicates = len(frame)
    duplicate_rows = frame[frame["period"].duplicated(keep=False)]
    conflicting_periods = [
        str(period)
        for period, group in duplicate_rows.groupby("period", sort=False)
        if len(group[[*RED_COLUMNS, "blue"]].drop_duplicates()) > 1
    ]
    if conflicting_periods:
        preview = ", ".join(conflicting_periods[:10])
        suffix = "..." if len(conflicting_periods) > 10 else ""
        raise DataQualityError(f"检测到同一期号对应不同开奖号码，拒绝静默覆盖: {preview}{suffix}")
    frame = frame.drop_duplicates(subset="period", keep="last")
    frame = frame.sort_values("period", key=lambda values: values.astype(int)).reset_index(drop=True)

    filled_dates = 0
    if fill_missing_dates:
        filled_dates = _fill_dates(frame, draw_weekdays, draw_blackout_ranges)
        frame.loc[frame["date_provenance"].eq("missing"), "date_provenance"] = "derived_schedule"
    if frame["date"].isna().any():
        raise DataQualityError("存在缺失日期且未启用日期补全，无法生成可审计时间序列")
    if validate_source_dates:
        weekdays = {int(value) for value in draw_weekdays}
        blackout = _blackout_ranges(draw_blackout_ranges)
        supplied = frame["date_provenance"].eq("source")
        invalid_dates = supplied & (
            ~frame["date"].dt.weekday.isin(weekdays)
            | frame["date"].dt.year.ne(frame["period"].str[:4].astype(int))
            | frame["date"].dt.date.map(lambda value: _in_blackout_ranges(value, blackout))
        )
        if invalid_dates.any() or frame["date"].duplicated().any():
            raise DataQualityError("来源日期不符合开奖日历、年份或唯一性约束")
    frame = frame.dropna(subset=["date"]).copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.drop(columns="_source_row")
    frame.insert(0, "idx", range(len(frame)))
    assert_clean_sequence(frame)

    report = {
        "input_rows": int(input_rows),
        "clean_rows": int(len(frame)),
        "dropped_rows": int(input_rows - len(frame)),
        "invalid_rows_removed": int(len(quarantine)),
        "duplicate_periods_removed": int(before_duplicates - len(frame)),
        "conflicting_duplicate_periods": 0,
        "dates_filled_from_schedule": int(filled_dates),
        "source_dates": int(frame["date_provenance"].eq("source").sum()),
        "derived_schedule_dates": int(frame["date_provenance"].eq("derived_schedule").sum()),
        "quarantined_rows": quarantine,
        "quality_status": "requires_review" if quarantine else "verified",
    }
    return CleanResult(frame, report, quarantine)


def save_clean_data(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    frame.to_pickle(temporary)
    temporary.replace(target)
    return target
