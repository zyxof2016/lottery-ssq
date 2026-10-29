"""Keep a long experiment from being paused by Windows idle sleep."""

from __future__ import annotations

import ctypes
import logging
import os
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def pulse_idle_sleep_block() -> None:
    """Refresh the request that the system stay awake. Safe to call every period."""

    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    except Exception:
        logging.debug("刷新防休眠状态失败", exc_info=True)


def _clear_idle_sleep_block() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        logging.debug("恢复系统休眠设置失败", exc_info=True)


@contextmanager
def prevent_idle_sleep():
    """Block idle sleep for the duration of a pipeline run.

    The screen can still lock. Manual sleep and closing the lid still suspend
    the machine. This only stops Windows from sleeping because the session
    looks idle.
    """

    pulse_idle_sleep_block()
    if os.name == "nt":
        logging.info(
            "实验运行期间阻止空闲休眠 | 事件=IDLE_SLEEP_BLOCKED | 说明=屏幕可以锁定，系统不会因为空闲自动睡眠；手动睡眠或合上盖子仍会暂停",
        )
    try:
        yield
    finally:
        _clear_idle_sleep_block()
