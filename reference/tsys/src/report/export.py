from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.metrics import evaluate_tickets, validate_tickets
from src.data_handler.cleaner import RED_COLUMNS
from src.data_handler.versioning import atomic_write_json
from src.fusion.model_fusion import (
    ANCHOR_HEAD_SHARE_KEY,
    ANCHOR_MIN_SIGNAL_KEY,
    ANCHOR_MODE_KEY,
    ANCHOR_TOP_K_KEY,
)


MANDATORY_INTERPRETATION = (
    "所有打分和组合仅为历史数据模拟，不代表下期开奖趋势，"
    "不存在提升中奖概率效果。显著性检验未通过时，不得表述为优于随机。"
)


class AuditIntegrityError(RuntimeError):
    """Raised when a forecast ledger is missing or fails its integrity proof."""


def _native(value: Any) -> Any:
    return _native_value(value, active_containers=set())


def _native_value(value: Any, active_containers: set[int]) -> Any:
    if isinstance(value, dict):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("报告包含循环引用，无法导出为可审计 JSON")
        active_containers.add(identity)
        try:
            return {
                str(key): _native_value(item, active_containers)
                for key, item in value.items()
            }
        finally:
            active_containers.remove(identity)
    if isinstance(value, (list, tuple, set)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("报告包含循环引用，无法导出为可审计 JSON")
        active_containers.add(identity)
        try:
            return [_native_value(item, active_containers) for item in value]
        finally:
            active_containers.remove(identity)
    if isinstance(value, np.ndarray):
        return _native_value(value.tolist(), active_containers)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        _native(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _key_bytes(signing_key: str | bytes | None) -> bytes | None:
    if signing_key is None:
        return None
    value = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    return value or None


def _hmac_digest(value: str, signing_key: str | bytes | None) -> str | None:
    key = _key_bytes(signing_key)
    return hmac.new(key, value.encode("ascii"), hashlib.sha256).hexdigest() if key else None


_PERIOD_RECORD_DROP_KEYS = {
    "model_oof_probabilities",
    "model_predictions",
    "model_blue_ranking_metrics",
    "parameters_used",
    "sampling",
    "matched_no_signal_sampling",
    "most_similar_prior_context",
}


def slim_report_for_export(report: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy diagnostic blobs so 300-period research reports fit in memory."""
    payload = dict(report)
    backtest = payload.get("backtest")
    if isinstance(backtest, dict):
        slim_backtest = dict(backtest)
        records = slim_backtest.get("period_records")
        if isinstance(records, list):
            slim_backtest["period_records"] = [
                {
                    key: value
                    for key, value in record.items()
                    if key not in _PERIOD_RECORD_DROP_KEYS
                }
                if isinstance(record, dict)
                else record
                for record in records
            ]
        payload["backtest"] = slim_backtest
    validation_trace = payload.get("validation_trace")
    if isinstance(validation_trace, dict):
        payload["validation_trace"] = {
            "path": validation_trace.get("path"),
            "exists": validation_trace.get("exists"),
            "periods": validation_trace.get("periods"),
            "run_start": validation_trace.get("run_start"),
            "run_complete": validation_trace.get("run_complete"),
            "rows_deferred": True,
            "note": "逐期明细保留在 validation_trace.jsonl，仪表盘按需加载，避免报告 JSON 内存溢出。",
        }
    return payload


def export_experiment_report(report: dict, path: str | Path) -> Path:
    if "random_baseline" not in report:
        raise RuntimeError("报告缺少同等规模随机基线，禁止导出")
    payload = slim_report_for_export(report)
    payload["mandatory_interpretation"] = MANDATORY_INTERPRETATION
    # Compact JSON avoids a second multi-hundred-MB indented string copy.
    return atomic_write_json(path, _native(payload), indent=None, sort_keys=False)


def export_candidate_csv(candidates: list[dict], path: str | Path, *, enable_export_bet: bool) -> Path:
    if not enable_export_bet:
        raise PermissionError("enable_export_bet=False，已禁止导出批量组合")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"red": " ".join(f"{number:02d}" for number in item["red"]), "blue": f"{item['blue']:02d}"}
        for item in candidates
    ]
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(target)
    return target


def load_forecast_audit(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"version": 2, "forecasts": [], "events": [], "integrity": {"schema": "tsys-audit-v2"}}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("forecasts"), list):
        raise ValueError(f"预测审计文件格式无效: {target}")
    payload.setdefault("version", 2)
    payload.setdefault("events", [])
    payload.setdefault("integrity", {"schema": "tsys-audit-v2"})
    return payload


def _write_forecast_audit(path: str | Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, _native(payload))


def _forecast_identity(record: dict[str, Any]) -> dict[str, Any]:
    """Fields that must never change after a pre-draw forecast is archived."""
    return {
        "forecast_id": str(record.get("forecast_id", "")),
        "forecast_revision": int(record.get("forecast_revision", 0)),
        "source_period": str(record.get("source_period", "")),
        "target_period": str(record.get("target_period", "")),
        "created_at": str(record.get("created_at", "")),
        "method": str(record.get("method", "")),
        "prediction_mode": str(record.get("prediction_mode", "")),
        "config_fingerprint": str(record.get("config_fingerprint", "")),
        "data_version": _native(record.get("data_version", {})),
        "code_fingerprint": str(record.get("code_fingerprint", "")),
        "weights_by_color": _native(record.get("weights_by_color", {})),
        "tickets": _native(record.get("tickets", [])),
    }


def _forecast_signature(
    tickets: list[dict[str, Any]],
    model_weights_by_color: dict[str, dict[str, float]],
    *,
    prediction_mode: str = "",
    config_fingerprint: str = "",
    data_version_id: str = "",
    code_fingerprint: str = "",
) -> str:
    anchor_keys = {
        ANCHOR_MODE_KEY,
        ANCHOR_TOP_K_KEY,
        ANCHOR_HEAD_SHARE_KEY,
        ANCHOR_MIN_SIGNAL_KEY,
    }

    def _split_weights_and_anchor(weights: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        numeric_weights: dict[str, float] = {}
        anchor_config: dict[str, Any] = {}
        for name, weight in sorted(weights.items()):
            if name in anchor_keys:
                anchor_config[name] = _native(weight)
                continue
            numeric_weights[name] = round(float(weight), 12)
        return numeric_weights, anchor_config

    identity = {
        "tickets": [
            {"red": [int(number) for number in ticket["red"]], "blue": int(ticket["blue"])}
            for ticket in tickets
        ],
        "weights_by_color": {},
        "anchor_config_by_color": {},
        "prediction_mode": str(prediction_mode),
        "config_fingerprint": str(config_fingerprint),
        "data_version_id": str(data_version_id),
        "code_fingerprint": str(code_fingerprint),
    }
    for color, weights in sorted(model_weights_by_color.items()):
        numeric_weights, anchor_config = _split_weights_and_anchor(weights)
        identity["weights_by_color"][color] = numeric_weights
        if anchor_config:
            identity["anchor_config_by_color"][color] = anchor_config
    return _sha256(identity)


def _append_event(
    payload: dict[str, Any],
    event: dict[str, Any],
    signing_key: str | bytes | None,
) -> dict[str, Any]:
    events = payload.setdefault("events", [])
    previous_hash = str(events[-1].get("event_hash", "")) if events else ""
    unsigned = {
        **event,
        "event_sequence": len(events) + 1,
        "previous_event_hash": previous_hash,
    }
    event_hash = _sha256(unsigned)
    signed = {
        **unsigned,
        "event_hash": event_hash,
        "event_hmac_sha256": _hmac_digest(event_hash, signing_key),
    }
    events.append(signed)
    integrity = payload.setdefault("integrity", {"schema": "tsys-audit-v2"})
    integrity.update(
        {
            "schema": "tsys-audit-v2",
            "event_chain_head": event_hash,
            "hmac_enabled": _key_bytes(signing_key) is not None,
            "key_fingerprint": (
                hashlib.sha256(_key_bytes(signing_key)).hexdigest()[:16]
                if _key_bytes(signing_key)
                else None
            ),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return signed


def _migrate_legacy_payload(payload: dict[str, Any], signing_key: str | bytes | None) -> None:
    if payload.get("events"):
        return
    for record in payload.get("forecasts", []):
        _append_event(
            payload,
            {
                "event_type": "legacy_forecast_anchor",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "forecast_id": str(record.get("forecast_id", "")),
                "forecast_identity_sha256": _sha256(_forecast_identity(record)),
                "migration_note": "legacy record anchored without external historical attestation",
            },
            signing_key,
        )
    payload.setdefault("integrity", {})["legacy_records_migrated"] = bool(payload.get("forecasts"))


def verify_forecast_audit(
    payload: dict[str, Any],
    *,
    signing_key: str | bytes | None = None,
    require_integrity: bool = False,
) -> dict[str, Any]:
    events = payload.get("events", [])
    if not events:
        if require_integrity:
            raise AuditIntegrityError("预测审计账本尚未建立完整性链")
        return {"verified": False, "reason": "legacy_or_empty_ledger", "event_count": 0}
    previous_hash = ""
    archives: dict[str, str] = {}
    for expected_sequence, event in enumerate(events, start=1):
        unsigned = {
            key: value
            for key, value in event.items()
            if key not in {"event_hash", "event_hmac_sha256"}
        }
        if int(event.get("event_sequence", 0)) != expected_sequence:
            raise AuditIntegrityError("预测审计事件序号不连续")
        if str(event.get("previous_event_hash", "")) != previous_hash:
            raise AuditIntegrityError("预测审计事件链已断裂")
        event_hash = _sha256(unsigned)
        if not hmac.compare_digest(str(event.get("event_hash", "")), event_hash):
            raise AuditIntegrityError("预测审计事件哈希校验失败")
        expected_hmac = _hmac_digest(event_hash, signing_key)
        if require_integrity and expected_hmac is None:
            raise AuditIntegrityError("未配置预测审计 HMAC 密钥")
        stored_hmac = event.get("event_hmac_sha256")
        if expected_hmac is not None and not hmac.compare_digest(str(stored_hmac or ""), expected_hmac):
            raise AuditIntegrityError("预测审计事件 HMAC 校验失败")
        if event.get("event_type") in {"forecast_archived", "shadow_forecast_archived", "legacy_forecast_anchor"}:
            archives[str(event.get("forecast_id", ""))] = str(event.get("forecast_identity_sha256", ""))
        previous_hash = event_hash
    for record in payload.get("forecasts", []):
        forecast_id = str(record.get("forecast_id", ""))
        expected_identity = archives.get(forecast_id)
        if expected_identity and not hmac.compare_digest(expected_identity, _sha256(_forecast_identity(record))):
            raise AuditIntegrityError(f"预测记录 {forecast_id} 的开奖前内容已被改写")
    return {
        "verified": True,
        "event_count": len(events),
        "hmac_enabled": _key_bytes(signing_key) is not None,
        "event_chain_head": previous_hash,
    }


def _ensure_integrity(
    payload: dict[str, Any],
    signing_key: str | bytes | None,
    require_integrity: bool,
) -> None:
    if require_integrity and _key_bytes(signing_key) is None:
        raise AuditIntegrityError("必须配置预测审计 HMAC 密钥后才能运行正式实验")
    _migrate_legacy_payload(payload, signing_key)
    if not payload.get("forecasts") and not payload.get("events"):
        return
    verify_forecast_audit(payload, signing_key=signing_key, require_integrity=require_integrity)


def archive_forecast(
    path: str | Path,
    *,
    source_period: str,
    target_period: str,
    tickets: list[dict],
    model_weights_by_color: dict[str, dict[str, float]],
    prediction_mode: str = "model_weighted_signal",
    method: str = "multi_model_color_specific_fusion_plus_consensus_selection",
    config_fingerprint: str = "",
    data_manifest: dict[str, Any] | None = None,
    code_fingerprint: str = "",
    metadata: dict[str, Any] | None = None,
    signing_key: str | bytes | None = None,
    require_integrity: bool = False,
    forecast_scope: str = "official",
    lock_official_target: bool = True,
) -> dict[str, Any]:
    payload = load_forecast_audit(path)
    _ensure_integrity(payload, signing_key, require_integrity)
    scope = "official" if str(forecast_scope or "official").lower() == "official" else "shadow"
    normalized_tickets = [
        {
            "red": [int(number) for number in ticket["red"]],
            "blue": int(ticket["blue"]),
            "consensus_score": float(ticket.get("consensus_score", 0.0)),
        }
        for ticket in tickets
    ]
    data_version = {
        key: value
        for key, value in (data_manifest or {}).items()
        if key in {"data_version_id", "raw_sha256", "clean_frame_sha256", "periods", "latest_period"}
    }
    signature = _forecast_signature(
        normalized_tickets,
        model_weights_by_color,
        prediction_mode=prediction_mode,
        config_fingerprint=config_fingerprint,
        data_version_id=str(data_version.get("data_version_id", "")),
        code_fingerprint=code_fingerprint,
    )
    same_target = [
        item for item in payload["forecasts"]
        if str(item.get("source_period")) == str(source_period)
        and str(item.get("target_period")) == str(target_period)
    ]
    official_same_target = [
        item for item in same_target
        if str(item.get("forecast_scope", "official")).lower() == "official"
    ]
    for existing in same_target:
        existing_signature = existing.get("forecast_signature") or _forecast_signature(
            existing.get("tickets", []),
            existing.get("weights_by_color", {}),
            prediction_mode=str(existing.get("prediction_mode", "")),
            config_fingerprint=str(existing.get("config_fingerprint", "")),
            data_version_id=str(existing.get("data_version", {}).get("data_version_id", "")),
            code_fingerprint=str(existing.get("code_fingerprint", "")),
        )
        if existing_signature == signature:
            if metadata:
                existing_metadata = existing.setdefault("metadata", {})
                before = _sha256(existing_metadata)
                existing_metadata.update(_native(metadata))
                if _sha256(existing_metadata) != before:
                    _append_event(
                        payload,
                        {
                            "event_type": "forecast_metadata_updated",
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "forecast_id": existing.get("forecast_id"),
                            "data_version_id": data_version.get("data_version_id"),
                        },
                        signing_key,
                    )
                    _write_forecast_audit(path, payload)
            return existing
    if scope == "official" and lock_official_target and official_same_target:
        locked = dict(sorted(official_same_target, key=lambda item: int(item.get("forecast_revision", 1) or 1))[0])
        locked["archive_decision"] = {
            "status": "official_target_locked",
            "reason": "同一训练截止期和目标期已有正式开奖前留痕；后续运行不得覆盖正式预测，只能作为影子实验另行记录。",
            "suppressed_signature": signature,
            "requested_scope": scope,
        }
        return locked
    revision = len(same_target) + 1
    forecast_id = f"{source_period}_to_{target_period}"
    if scope == "shadow":
        shadow_index = 1 + sum(
            1
            for item in same_target
            if str(item.get("forecast_scope", "official")).lower() != "official"
        )
        forecast_id = f"{forecast_id}_shadow{shadow_index}"
    elif revision > 1:
        forecast_id = f"{forecast_id}_r{revision}"
    record = {
        "forecast_id": forecast_id,
        "forecast_revision": revision,
        "forecast_scope": scope,
        "forecast_signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_period": str(source_period),
        "target_period": str(target_period),
        "status": "pending" if scope == "official" else "shadow_pending",
        "method": str(method),
        "prediction_mode": str(prediction_mode),
        "config_fingerprint": str(config_fingerprint),
        "data_version": data_version,
        "code_fingerprint": str(code_fingerprint),
        "weights_by_color": _native(model_weights_by_color),
        "tickets": normalized_tickets,
        "metadata": _native(metadata or {}),
        "validation": None,
    }
    payload["forecasts"].append(record)
    _append_event(
        payload,
        {
            "event_type": "forecast_archived" if scope == "official" else "shadow_forecast_archived",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "forecast_id": forecast_id,
            "forecast_scope": scope,
            "forecast_identity_sha256": _sha256(_forecast_identity(record)),
            "data_version_id": data_version.get("data_version_id"),
        },
        signing_key,
    )
    _write_forecast_audit(path, payload)
    return record


def reconcile_forecast_audit(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    data_manifest: dict[str, Any] | None = None,
    signing_key: str | bytes | None = None,
    require_integrity: bool = False,
) -> dict[str, Any]:
    payload = load_forecast_audit(path)
    _ensure_integrity(payload, signing_key, require_integrity)
    by_period = {str(row["period"]): row for _, row in frame.iterrows()}
    updated = False
    for record in payload["forecasts"]:
        target_period = str(record.get("target_period", ""))
        target = by_period.get(target_period)
        if target is None or record.get("validation") is not None:
            continue
        actual_red = {int(target[column]) for column in RED_COLUMNS}
        actual_blue = int(target["blue"])
        tickets = record.get("tickets", [])
        if not tickets:
            continue
        metrics = evaluate_tickets(tickets, actual_red, actual_blue)
        validated = validate_tickets(tickets, actual_red, actual_blue)
        unique_blues = len({int(ticket["blue"]) for ticket in tickets})
        random_expected_red_hits = 36.0 / 33.0
        validation = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "actual": {"red": sorted(actual_red), "blue": actual_blue},
            "tickets": validated,
            "metrics": metrics,
            "reconciliation_data_version": {
                key: value
                for key, value in (data_manifest or {}).items()
                if key in {"data_version_id", "raw_sha256", "clean_frame_sha256"}
            },
            "theoretical_random_baseline": {
                "average_red_hits_per_ticket": random_expected_red_hits,
                "any_blue_hit_probability_for_unique_blues": unique_blues / 16.0,
            },
            "red_hit_delta_vs_random_expectation": metrics["average_red_hits"] - random_expected_red_hits,
            "interpretation": "单期独立验证只能记录结果，不能据此证明或否定预测优势。",
        }
        record["status"] = "validated"
        record["validation"] = validation
        _append_event(
            payload,
            {
                "event_type": "forecast_validated",
                "created_at": validation["validated_at"],
                "forecast_id": str(record["forecast_id"]),
                "forecast_identity_sha256": _sha256(_forecast_identity(record)),
                "validation_sha256": _sha256(validation),
                "reconciliation_data_version_id": validation["reconciliation_data_version"].get("data_version_id"),
            },
            signing_key,
        )
        updated = True
    if updated or payload.get("events"):
        _write_forecast_audit(path, payload)
    validated_records = [item for item in payload["forecasts"] if item.get("validation") is not None]
    integrity = (
        {
            "verified": False,
            "reason": "no_forecasts_archived",
            "event_count": 0,
            "hmac_enabled": _key_bytes(signing_key) is not None,
        }
        if not payload["forecasts"] and not payload.get("events")
        else verify_forecast_audit(
            payload, signing_key=signing_key, require_integrity=require_integrity
        )
    )
    return {
        "forecast_count": len(payload["forecasts"]),
        "validated_count": len(validated_records),
        "pending_count": len(payload["forecasts"]) - len(validated_records),
        "records": payload["forecasts"],
        "integrity": integrity,
    }
