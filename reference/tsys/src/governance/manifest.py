from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_run_manifest(
    *,
    config_hash: str,
    code_fingerprint: str,
    seed: int,
    data_manifest: dict[str, Any] | None,
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create the V7 reproducibility manifest for one pipeline run."""
    data_manifest = data_manifest or {}
    runtime_metadata = runtime_metadata or {}
    return {
        "schema": "run_manifest_v7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_manifest.get("clean_frame_sha256"),
        "raw_data_hash": data_manifest.get("raw_sha256"),
        "data_version_id": data_manifest.get("data_version_id"),
        "first_period": data_manifest.get("first_period"),
        "latest_period": data_manifest.get("latest_period"),
        "config_hash": str(config_hash),
        "git_commit": str(code_fingerprint),
        "source_tree_sha256": str(code_fingerprint),
        "seed": int(seed),
        "package_versions": dict(runtime_metadata.get("packages", {})),
        "python": runtime_metadata.get("python"),
        "platform": runtime_metadata.get("platform"),
    }
