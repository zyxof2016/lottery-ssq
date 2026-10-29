from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class RuntimeStep:
    name: str
    started_at: float
    duration_seconds: float
    status: str
    fields: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None


class RuntimeProfiler:
    """Small in-memory profiler for pipeline steps.

    The profiler records timings only. It does not wrap or memoize decision
    functions, so enabling it cannot change prediction output.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started = time.perf_counter()
        self._steps: list[RuntimeStep] = []

    def record(
        self,
        name: str,
        *,
        started_at: float,
        duration_seconds: float,
        status: str,
        fields: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> None:
        self._steps.append(
            RuntimeStep(
                name=str(name),
                started_at=float(started_at),
                duration_seconds=float(duration_seconds),
                status=str(status),
                fields=dict(fields or {}),
                error_type=error_type,
            )
        )

    def report(self) -> dict[str, Any]:
        steps = [
            {
                "name": step.name,
                "duration_seconds": round(step.duration_seconds, 6),
                "status": step.status,
                "fields": step.fields,
                **({"error_type": step.error_type} if step.error_type else {}),
            }
            for step in self._steps
        ]
        by_name: dict[str, float] = {}
        for step in self._steps:
            by_name[step.name] = by_name.get(step.name, 0.0) + step.duration_seconds
        slowest = sorted(
            (
                {
                    "name": name,
                    "duration_seconds": round(duration, 6),
                }
                for name, duration in by_name.items()
            ),
            key=lambda item: item["duration_seconds"],
            reverse=True,
        )
        total = time.perf_counter() - self._started
        return {
            "schema": "runtime_performance_profile_v1",
            "total_observed_seconds": round(total, 6),
            "step_count": len(self._steps),
            "slowest_steps": slowest[:10],
            "steps": steps,
            "note": "仅记录耗时，不参与模型、候选池或最终组合决策。",
        }


RUNTIME_PROFILER = RuntimeProfiler()


@contextmanager
def profiled_step(name: str, **fields: Any) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        RUNTIME_PROFILER.record(
            name,
            started_at=started,
            duration_seconds=time.perf_counter() - started,
            status="failed",
            fields=fields,
            error_type=exc.__class__.__name__,
        )
        raise
    else:
        RUNTIME_PROFILER.record(
            name,
            started_at=started,
            duration_seconds=time.perf_counter() - started,
            status="done",
            fields=fields,
        )

