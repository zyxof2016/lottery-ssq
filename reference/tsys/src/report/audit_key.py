from __future__ import annotations

import os
import secrets
from pathlib import Path


class AuditKeyError(RuntimeError):
    """Raised when the local audit signing key cannot be prepared safely."""


def default_audit_key_path() -> Path:
    """Return a user-local path that intentionally sits outside the project tree."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
    return base / "Tsys" / "secrets" / "forecast_audit_hmac.key"


def load_or_create_audit_key(
    environment_variable: str,
    storage_path: Path | None = None,
) -> tuple[str, str]:
    """Use a managed environment key when available, otherwise keep one locally.

    The returned source is deliberately non-sensitive and can be logged for support.
    """
    external_key = os.environ.get(environment_variable, "").strip()
    if external_key:
        return external_key, "environment"

    path = storage_path or default_audit_key_path()
    try:
        if path.exists():
            local_key = path.read_text(encoding="utf-8").strip()
            if len(local_key) < 32:
                raise AuditKeyError(f"本机审计密钥无效，请删除后重试：{path}")
            return local_key, "local_file"

        path.parent.mkdir(parents=True, exist_ok=True)
        local_key = secrets.token_urlsafe(48)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(local_key + "\n", encoding="utf-8")
        temporary.replace(path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return local_key, "generated_local_file"
    except AuditKeyError:
        raise
    except OSError as exc:
        raise AuditKeyError(f"无法准备本机审计密钥：{path}") from exc
