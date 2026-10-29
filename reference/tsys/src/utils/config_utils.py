from __future__ import annotations

import copy
from datetime import date, datetime
from pathlib import Path
from typing import Any


SENSITIVE_CONFIG_KEYS = {"password", "token", "secret", "authorization", "auth_code", "api_key"}


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def redact_sensitive_config(config: dict[str, Any]) -> dict[str, Any]:
    def redact(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {item_key: redact(item_value, item_key) for item_key, item_value in value.items()}
        if key.lower() in SENSITIVE_CONFIG_KEYS:
            return "***REDACTED***" if value not in (None, "") else value
        return copy.deepcopy(value)

    return redact(config)


def config_fingerprint_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = redact_sensitive_config(config)
    notification = payload.get("notification", {})
    if isinstance(notification, dict):
        email = notification.get("email", {})
        if isinstance(email, dict):
            for key in ("password", "from", "to", "username"):
                email.pop(key, None)

    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(item_key): json_safe(item_value) for item_key, item_value in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [json_safe(item) for item in value]
        if isinstance(value, set):
            return sorted(json_safe(item) for item in value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        return copy.deepcopy(value)

    return json_safe(payload)


def runtime_profile_options(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    profiles = runtime.get("profiles", {})
    return {
        key: {
            "label": value.get("label", key),
            "estimated_minutes": value.get("estimated_minutes", "-"),
            "description": value.get("description", ""),
            "periods": value.get("overrides", {}).get("backtest", {}).get("periods"),
            "monte_carlo_samples": value.get("overrides", {}).get("generation", {}).get("monte_carlo_samples"),
            "random_repeats": value.get("overrides", {}).get("backtest", {}).get("random_repeats"),
            "methods": value.get("overrides", {}).get("optimizer", {}).get("methods"),
        }
        for key, value in profiles.items()
        if isinstance(value, dict)
    }


def apply_runtime_profile(config: dict[str, Any], requested_profile: str | None = None) -> str:
    runtime = config.setdefault("runtime", {})
    profiles = runtime.get("profiles", {})
    profile = str(requested_profile or runtime.get("default_profile", "standard")).strip() or "standard"
    if profile not in profiles:
        raise ValueError(f"未知运行档位: {profile}")
    overrides = profiles[profile].get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"运行档位 {profile} 的 overrides 必须是配置字典")
    deep_update(config, overrides)
    config.setdefault("runtime", {})["active_profile"] = profile
    return profile
