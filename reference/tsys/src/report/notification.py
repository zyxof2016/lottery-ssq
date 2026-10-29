# -*- coding: utf-8 -*-
"""QQ邮箱通知模块：项目运行结束后发送结果邮件。"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import subprocess
import tempfile
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def _get_env_value(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                registry_value, _ = winreg.QueryValueEx(key, name)
                return str(registry_value or default)
        except OSError:
            return default
    return default


def _format_ticket(ticket: dict[str, Any], index: int) -> str:
    red = " ".join(f"{int(number):02d}" for number in ticket.get("red", []))
    blue = ticket.get("blue", "-")
    blue_text = f"{int(blue):02d}" if isinstance(blue, int) or str(blue).isdigit() else str(blue)
    return f"{index}. {red} + {blue_text}"


def _format_dantuo(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return ["胆拖方案：无"]
    red_dan = " ".join(f"{int(number):02d}" for number in plan.get("red_dan", [])) or "-"
    red_tuo = " ".join(f"{int(number):02d}" for number in plan.get("red_tuo", [])) or "-"
    blue = " ".join(f"{int(number):02d}" for number in plan.get("blue", [])) or "-"
    amount = plan.get("amount", "-")
    bet_count = plan.get("bet_count", "-")
    rule = plan.get("rule", "")
    return [
        "胆拖方案：",
        f"红胆：{red_dan}",
        f"红拖：{red_tuo}",
        f"蓝球：{blue}",
        f"注数/金额：{bet_count} 注 / {amount} 元",
        f"规则：{rule}" if rule else "",
    ]


def _format_core_numbers(core: dict[str, Any] | None) -> list[str]:
    if not core or not bool(core.get("enabled", False)):
        return ["核心号码提前告知：无"]
    red = " ".join(f"{int(number):02d}" for number in core.get("core_red", [])[:6]) or "-"
    blue = " ".join(f"{int(number):02d}" for number in core.get("core_blue", [])[:3]) or "-"
    concentrated = core.get("concentrated_ticket", {}) or {}
    concentrated_red = " ".join(f"{int(number):02d}" for number in concentrated.get("red", [])[:6]) or "-"
    concentrated_blue = concentrated.get("blue", "-")
    concentrated_blue_text = (
        f"{int(concentrated_blue):02d}"
        if isinstance(concentrated_blue, int) or str(concentrated_blue).isdigit()
        else str(concentrated_blue)
    )
    return [
        "核心号码提前告知：",
        f"核心红球Top6：{red}",
        f"核心蓝球Top3：{blue}",
        f"集中型参考单注：{concentrated_red} + {concentrated_blue_text}",
        "说明：核心号码用于观察“系统共同看好但可能分散在多注中”的号码，不保证命中。",
    ]


def _project_label(project_root: str | Path | None) -> str:
    if not project_root:
        return "Tsys"
    root = Path(project_root).resolve()
    if root.parent.name.lower() == "tsystem":
        return "Tsystem"
    return root.name or "Tsys"


def build_email_result_body(report: dict[str, Any], *, project_root: str | Path | None = None) -> tuple[str, str]:
    forecast = report.get("forecast_archive", {})
    candidate = report.get("candidate_experiment", {})
    consensus = candidate.get("consensus", {})
    tickets = (
        consensus.get("display_selected")
        or consensus.get("experimental_selected")
        or consensus.get("selected")
        or forecast.get("tickets")
        or []
    )
    dantuo_plan = (
        candidate.get("dantuo_plan")
        or consensus.get("dantuo_plan")
        or forecast.get("dantuo_plan")
    )
    core_numbers = (
        candidate.get("core_numbers")
        or forecast.get("core_numbers")
        or {}
    )
    source_period = forecast.get("source_period") or report.get("data", {}).get("latest_period", "-")
    target_period = forecast.get("target_period", "-")
    project_name = _project_label(project_root)
    project_path = str(Path(project_root).resolve()) if project_root else "-"
    title = f"Tsys运行结果[{project_name}]：{source_period} → {target_period}"
    lines = [
        f"项目：{project_name}（{project_path}）",
        f"训练截止期：{source_period}",
        f"目标期：{target_period}",
        "",
        "生成的5注：",
    ]
    for index, ticket in enumerate(list(tickets)[:5], 1):
        lines.append(_format_ticket(ticket, index))
    if not tickets:
        lines.append("无可展示号码")
    lines.extend(["", *_format_dantuo(dantuo_plan), "", *_format_core_numbers(core_numbers), ""])
    lines.append("说明：仅发送本次生成号码和预算内胆拖方案；不附带截图或其他附件。")
    lines.append("风险提示：仅供历史数据概率实验，不构成投注建议。")
    return title, "\n".join(line for line in lines if line is not None)


def _configured_chrome_path(config: dict[str, Any]) -> str | None:
    configured = (
        os.environ.get("TSYS_CHROME_PATH")
        or config.get("chrome_path")
        or config.get("browser_path")
    )
    candidates = [
        configured,
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def capture_result_page_screenshot(
    *,
    url: str,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path | None:
    """Capture the local result page with Chrome/Edge headless.

    This intentionally avoids browser automation dependencies.  The screenshot
    is viewport-sized; the configured window height is large enough to include
    the normal result page top-to-bottom summary area.
    """
    screenshot_config = config or {}
    chrome = _configured_chrome_path(screenshot_config)
    if not chrome:
        logging.warning("结果页截图跳过 | 事件=EMAIL_SCREENSHOT_SKIPPED | 原因=未找到Chrome或Edge")
        return None
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass
    width = int(screenshot_config.get("screenshot_width", 1920))
    height = int(screenshot_config.get("screenshot_height", 2600))
    wait_ms = int(screenshot_config.get("screenshot_virtual_time_budget_ms", 6000))
    with tempfile.TemporaryDirectory(prefix="tsys_chrome_profile_") as profile:
        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            f"--virtual-time-budget={wait_ms}",
            f"--screenshot={str(target)}",
            url,
        ]
        try:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(10, wait_ms // 1000 + 10),
            )
        except Exception as exc:
            logging.warning("结果页截图失败 | 事件=EMAIL_SCREENSHOT_FAILED | 错误=%s", exc)
            return None
    if target.exists() and target.stat().st_size > 0:
        logging.info("结果页截图完成 | 事件=EMAIL_SCREENSHOT_DONE | 文件=%s", target)
        return target
    logging.warning("结果页截图失败 | 事件=EMAIL_SCREENSHOT_EMPTY | 文件=%s", target)
    return None


class EmailNotifier:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.to_addr = str(
            self.config.get("to")
            or _get_env_value("TSYS_EMAIL_TO")
            or "1498465975@qq.com"
        )
        self.smtp_host = str(self.config.get("smtp_host") or _get_env_value("TSYS_SMTP_HOST") or "smtp.qq.com")
        self.smtp_port = int(self.config.get("smtp_port") or _get_env_value("TSYS_SMTP_PORT") or 465)
        self.use_ssl = bool(self.config.get("smtp_ssl", True))
        self.timeout = int(self.config.get("timeout", 20))
        self.username = str(
            self.config.get("username")
            or _get_env_value(str(self.config.get("username_env", "TSYS_EMAIL_USER")))
            or ""
        )
        self.password = str(
            self.config.get("password")
            or _get_env_value(str(self.config.get("password_env", "TSYS_EMAIL_PASSWORD")))
            or ""
        )
        self.from_addr = str(
            self.config.get("from")
            or _get_env_value(str(self.config.get("from_env", "TSYS_EMAIL_FROM")))
            or self.username
        )

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.to_addr and self.smtp_host and self.username and self.password and self.from_addr)

    def send(
        self,
        *,
        subject: str,
        body: str,
        attachments: list[str | Path] | None = None,
    ) -> bool:
        if not self.ready:
            logging.warning(
                "邮件通知跳过 | 事件=EMAIL_NOTIFICATION_SKIPPED | 原因=未配置发件邮箱或SMTP授权码 | 收件人=%s",
                self.to_addr,
            )
            return False
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_addr
        message["To"] = self.to_addr
        message.set_content(body)
        for attachment in attachments or []:
            path = Path(attachment)
            if not path.exists() or path.stat().st_size <= 0:
                continue
            maintype, subtype = ("image", "png") if path.suffix.lower() == ".png" else ("application", "octet-stream")
            message.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
        attempts = self._smtp_attempts()
        last_error: Exception | None = None
        for host, port, use_ssl in attempts:
            try:
                self._send_message_once(message, host=host, port=port, use_ssl=use_ssl)
                logging.info(
                    "邮件通知已发送 | 事件=EMAIL_NOTIFICATION_SENT | 收件人=%s | SMTP=%s:%s | SSL=%s",
                    self.to_addr,
                    host,
                    port,
                    use_ssl,
                )
                return True
            except Exception as exc:
                last_error = exc
                logging.warning(
                    "邮件SMTP尝试失败 | 事件=EMAIL_SMTP_ATTEMPT_FAILED | SMTP=%s:%s | SSL=%s | 错误类型=%s | 错误=%s",
                    host,
                    port,
                    use_ssl,
                    type(exc).__name__,
                    exc,
                )
        logging.error(
            "邮件通知发送失败 | 事件=EMAIL_NOTIFICATION_FAILED | 收件人=%s | 原因=所有SMTP连接方式均失败；请检查QQ邮箱SMTP服务、授权码、网络或安全拦截 | 最后错误类型=%s | 最后错误=%s",
            self.to_addr,
            type(last_error).__name__ if last_error else "-",
            last_error or "-",
        )
        return False

    def _smtp_attempts(self) -> list[tuple[str, int, bool]]:
        attempts = [(self.smtp_host, self.smtp_port, self.use_ssl)]
        if self.smtp_host.lower() == "smtp.qq.com":
            attempts.extend([
                ("smtp.qq.com", 465, True),
                ("smtp.qq.com", 587, False),
            ])
        unique: list[tuple[str, int, bool]] = []
        seen: set[tuple[str, int, bool]] = set()
        for item in attempts:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    def _send_message_once(self, message: EmailMessage, *, host: str, port: int, use_ssl: bool) -> None:
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(host, port, timeout=self.timeout) as smtp:
                    smtp.ehlo_or_helo_if_needed()
                    smtp.login(self.username, self.password)
                    smtp.send_message(message)
            else:
                with smtplib.SMTP(host, port, timeout=self.timeout) as smtp:
                    smtp.ehlo_or_helo_if_needed()
                    smtp.starttls()
                    smtp.ehlo()
                    smtp.login(self.username, self.password)
                    smtp.send_message(message)
        except smtplib.SMTPAuthenticationError as exc:
            raise RuntimeError("SMTP认证失败，请确认QQ邮箱SMTP服务已开启且使用的是授权码，不是QQ登录密码") from exc


def send_result_email(
    report: dict[str, Any],
    config: dict[str, Any],
    *,
    project_root: str | Path,
    dashboard_url: str | None = None,
    delay_seconds: float = 0.0,
) -> bool:
    notification = config.get("notification", {})
    email_config = notification.get("email", {})
    notifier = EmailNotifier(email_config)
    if not notifier.enabled:
        return False
    forecast = report.get("forecast_archive", {}) if isinstance(report.get("forecast_archive", {}), dict) else {}
    archive_decision = forecast.get("archive_decision") or {}
    if archive_decision.get("status") == "official_target_locked":
        logging.info(
            "同目标期正式留痕已锁定，仍发送本次运行结果邮件 | 事件=EMAIL_NOTIFICATION_CONTINUE | 审计编号=%s | 项目=%s",
            forecast.get("forecast_id"),
            project_root,
        )
    subject, body = build_email_result_body(report, project_root=project_root)
    if not notifier.ready:
        return notifier.send(subject=subject, body=body, attachments=[])
    if delay_seconds > 0:
        time.sleep(float(delay_seconds))
    screenshot_path: Path | None = None
    if bool(email_config.get("attach_result_screenshot", False)) and dashboard_url:
        screenshot_path = capture_result_page_screenshot(
            url=str(dashboard_url or email_config.get("dashboard_url") or ""),
            output_path=Path(project_root) / str(email_config.get("screenshot_path", "logs/result_page_full.png")),
            config=email_config,
        )
    attachments = [screenshot_path] if screenshot_path else []
    return notifier.send(subject=subject, body=body, attachments=attachments)


if __name__ == "__main__":
    notifier = EmailNotifier(
        {
            "enabled": True,
            "to": _get_env_value("TSYS_EMAIL_TO", "1498465975@qq.com"),
            "smtp_host": "smtp.qq.com",
            "smtp_port": 465,
            "smtp_ssl": True,
        }
    )
    if notifier.ready:
        ok = notifier.send(subject="Tsys邮件通知测试", body="QQ邮箱通知通道可用。")
        print("邮件发送成功" if ok else "邮件发送失败")
    else:
        print("未配置发件邮箱或SMTP授权码")
