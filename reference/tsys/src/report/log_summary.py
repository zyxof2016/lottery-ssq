from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


EVENT_PATTERN = re.compile(r"事件=([A-Z0-9_]+)")


def summarize_pipeline_log(path: str | Path, *, tail_lines: int = 4000) -> dict[str, Any]:
    """Aggregate the recent pipeline log into compact report diagnostics."""
    source = Path(path)
    if not source.exists():
        return {"available": False, "reason": "log_file_missing", "path": str(source)}
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    recent = lines[-max(1, int(tail_lines)):]
    levels = Counter()
    events = Counter()
    for line in recent:
        parts = line.split()
        if len(parts) >= 3:
            levels[parts[2]] += 1
        match = EVENT_PATTERN.search(line)
        if match:
            events[match.group(1)] += 1
    return {
        "available": True,
        "path": str(source),
        "scanned_lines": int(len(recent)),
        "total_lines": int(len(lines)),
        "levels": dict(levels),
        "top_events": dict(events.most_common(25)),
        "error_lines": [line for line in recent if " ERROR " in line][-10:],
        "warning_count": int(levels.get("WARNING", 0)),
        "error_count": int(levels.get("ERROR", 0)),
    }
