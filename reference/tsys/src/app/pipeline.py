from __future__ import annotations

from typing import Any


def run_pipeline(config: dict[str, Any], periods: int | None = None) -> dict[str, Any]:
    """V7 application pipeline entrypoint."""
    from main import run_pipeline as _run_pipeline

    return _run_pipeline(config, periods=periods)


def prepare_data(config: dict[str, Any]):
    from main import prepare_data as _prepare_data

    return _prepare_data(config)


def load_config(path: str | None = None) -> dict[str, Any]:
    from main import load_config as _load_config

    return _load_config(path)
