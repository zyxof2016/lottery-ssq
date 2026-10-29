from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from src.data_handler.cleaner import RED_COLUMNS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(root: str | Path) -> str:
    """Fingerprint first-party Python and configuration files, excluding generated output."""
    base = Path(root)
    digest = hashlib.sha256()
    paths = [base / "main.py"]
    paths.extend((base / "config").glob("*.yaml"))
    paths.extend((base / "src").rglob("*.py"))
    paths.extend((base / "compliance").rglob("*.py"))
    for path in sorted({item.resolve() for item in paths if item.exists()}):
        relative = path.relative_to(base.resolve()).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash a cleaned frame in a stable, cross-platform representation."""
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["datetime", "datetimetz"]).columns:
        normalized[column] = normalized[column].astype("string")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(np.asarray(expected, dtype=float), 1e-9, None)
    actual = np.clip(np.asarray(actual, dtype=float), 1e-9, None)
    expected = expected / float(np.sum(expected))
    actual = actual / float(np.sum(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def data_drift_summary(frame: pd.DataFrame, *, window: int = 300) -> dict[str, Any]:
    """Summarize recent-vs-previous drift for the cleaned historical frame."""
    if frame.empty or len(frame) < max(40, int(window)):
        return {
            "enabled": True,
            "status": "insufficient_history",
            "window": int(window),
            "rows": int(len(frame)),
        }
    effective_window = min(int(window), max(20, len(frame) // 2))
    previous = frame.iloc[-2 * effective_window:-effective_window]
    recent = frame.iloc[-effective_window:]
    red_previous = np.bincount(
        previous[list(RED_COLUMNS)].to_numpy(dtype=int).reshape(-1) - 1,
        minlength=33,
    )
    red_recent = np.bincount(
        recent[list(RED_COLUMNS)].to_numpy(dtype=int).reshape(-1) - 1,
        minlength=33,
    )
    blue_previous = np.bincount(previous["blue"].to_numpy(dtype=int) - 1, minlength=16)
    blue_recent = np.bincount(recent["blue"].to_numpy(dtype=int) - 1, minlength=16)
    previous_red_sum = previous[list(RED_COLUMNS)].sum(axis=1).to_numpy(dtype=float)
    recent_red_sum = recent[list(RED_COLUMNS)].sum(axis=1).to_numpy(dtype=float)
    previous_span = (
        previous[list(RED_COLUMNS)].max(axis=1) - previous[list(RED_COLUMNS)].min(axis=1)
    ).to_numpy(dtype=float)
    recent_span = (
        recent[list(RED_COLUMNS)].max(axis=1) - recent[list(RED_COLUMNS)].min(axis=1)
    ).to_numpy(dtype=float)
    red_psi = _psi(red_previous, red_recent)
    blue_psi = _psi(blue_previous, blue_recent)
    sum_ks = ks_2samp(previous_red_sum, recent_red_sum, method="asymp")
    span_ks = ks_2samp(previous_span, recent_span, method="asymp")
    max_psi = max(red_psi, blue_psi)
    status = "high_drift" if max_psi >= 0.25 else "moderate_drift" if max_psi >= 0.10 else "stable"
    return {
        "enabled": True,
        "status": status,
        "window": int(effective_window),
        "red_margin_psi": red_psi,
        "blue_margin_psi": blue_psi,
        "red_sum_ks_p_value": float(sum_ks.pvalue),
        "red_span_ks_p_value": float(span_ks.pvalue),
        "latest_period": str(frame.iloc[-1]["period"]),
    }


def json_default(value: Any) -> Any:
    """JSON fallback shared by reports, manifests and web API responses."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def atomic_write_json(
    path: str | Path,
    payload: dict[str, Any],
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            default=json_default,
        )
        handle.write("\n")
    temporary.replace(target)
    return target


def create_data_manifest(
    *,
    raw_path: str | Path,
    clean_frame: pd.DataFrame,
    cleaning_report: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_source = Path(raw_path)
    raw_sha256 = sha256_file(raw_source) if raw_source.exists() else None
    clean_sha256 = frame_fingerprint(clean_frame)
    version_material = json.dumps(
        {
            "raw_sha256": raw_sha256,
            "clean_sha256": clean_sha256,
            "periods": len(clean_frame),
            "first_period": str(clean_frame.iloc[0]["period"]) if not clean_frame.empty else None,
            "latest_period": str(clean_frame.iloc[-1]["period"]) if not clean_frame.empty else None,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    # The caller can attach this manifest back to its live cleaning report for
    # dashboard use. Keep an independent snapshot here so that attachment can
    # never create a report <-> manifest reference cycle during JSON export.
    cleaning_snapshot = deepcopy(dict(cleaning_report))
    cleaning_snapshot.pop("data_manifest", None)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(raw_source),
        "raw_sha256": raw_sha256,
        "clean_frame_sha256": clean_sha256,
        "data_version_id": sha256_bytes(version_material)[:24],
        "periods": int(len(clean_frame)),
        "first_period": str(clean_frame.iloc[0]["period"]) if not clean_frame.empty else None,
        "latest_period": str(clean_frame.iloc[-1]["period"]) if not clean_frame.empty else None,
        "drift": data_drift_summary(clean_frame),
        "cleaning": cleaning_snapshot,
        "source": source or {"kind": "local_file"},
    }


def write_data_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(path, manifest)


def load_data_manifest(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"数据版本清单格式无效: {source}")
    return payload
