from __future__ import annotations

from typing import Any


def dashboard_run(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Stable web-callback boundary for the dashboard."""
    from main import dashboard_run as _dashboard_run

    return _dashboard_run(config, payload)


def run_dashboard_app() -> None:
    """Start the local dashboard through the V7 application boundary."""
    from main import main

    main()
