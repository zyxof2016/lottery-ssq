from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.25,
    label: str = "operation",
) -> T:
    """Run a small recoverable operation with bounded retry."""
    max_attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except (OSError, TimeoutError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            logging.warning(
                "可恢复操作失败，准备重试 | 事件=RETRYABLE_OPERATION_RETRY | 操作=%s | 尝试=%s/%s | 错误=%s",
                label,
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(max(0.0, float(delay_seconds)))
    assert last_error is not None
    raise last_error
