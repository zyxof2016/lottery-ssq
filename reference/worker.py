#!/usr/bin/env python3
"""Run the supplied Tsys pipeline on a bound CSV snapshot, outside the web process."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SOURCE_DATE_EXCEPTIONS = {
    ("04113", "2004-12-10"), ("04114", "2004-12-13"),
    ("05017", "2005-02-16"), ("06084", "2006-07-21"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_snapshot(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if len(rows) < 180:
        raise ValueError("参考引擎至少需要180期历史")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"])
    last = 0
    last_date = None
    for row in rows:
        issue = int(row["issue"])
        if issue <= last or len(row["issue"]) != 5:
            raise ValueError("历史期号必须为递增且互异的5位期号")
        last = issue
        date = dt.date.fromisoformat(row["date"])
        if date.year != 2000 + issue // 1000 or (last_date is not None and date <= last_date):
            raise ValueError("历史日期必须与期号年份一致、递增且唯一")
        if date.weekday() not in (1, 3, 6) and (row["issue"], row["date"]) not in SOURCE_DATE_EXCEPTIONS:
            raise ValueError("历史日期不符合常规开奖星期或已列明的原数据例外")
        last_date = date
        reds = [int(row[f"r{i}"]) for i in range(1, 7)]
        if len(set(reds)) != 6 or reds != sorted(reds) or not all(1 <= n <= 33 for n in reds):
            raise ValueError("输入红球无效")
        blue = int(row["blue"])
        if not 1 <= blue <= 16:
            raise ValueError("输入蓝球无效")
        writer.writerow([f"{issue:05}", row["date"], *[f"{n:02}" for n in reds], f"{blue:02}"])
    if (last + 1) // 1000 != last // 1000:
        raise ValueError("跨年目标需人工核验")
    return output.getvalue().encode(), rows


def engine_digest() -> str:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    digest = hashlib.sha256()
    for name, expected in sorted(manifest["files"].items()):
        data = (ROOT / "tsys" / name).read_bytes()
        if sha(data) != expected:
            raise ValueError(f"参考代码校验失败：{name}")
        digest.update(name.encode() + b"\0" + data)
    return digest.hexdigest()


def validate_cache(value: dict, data_sha: str, engine_sha: str) -> None:
    if value["version"] != 1 or value["data_sha256"] != data_sha or value["engine_sha256"] != engine_sha:
        raise ValueError("已有参考结果与输入或引擎不一致，保留首次结果并拒绝覆盖")
    cutoff, target = value.get("data_cutoff", ""), value.get("target_issue", "")
    if (not isinstance(cutoff, str) or not isinstance(target, str) or len(cutoff) != 5
            or len(target) != 5 or not cutoff.isascii() or not cutoff.isdigit()
            or not target.isascii() or not target.isdigit() or int(target) != int(cutoff) + 1
            or target[:2] != cutoff[:2]):
        raise ValueError("参考结果期号无效")
    generated = value.get("generated_at_unix")
    if type(generated) is not int or not 0 < generated <= time.time():
        raise ValueError("参考结果生成时间无效")
    config_sha = value.get("config_sha256", "")
    if not isinstance(config_sha, str) or len(config_sha) != 64 or any(c not in "0123456789abcdef" for c in config_sha):
        raise ValueError("参考结果缺少配置摘要")
    tickets = value["tickets"]
    if len(tickets) != 5:
        raise ValueError(f"source_count: 参考引擎返回{len(tickets)}注原始推荐，要求5注；未补造")
    seen = set()
    for ticket in tickets:
        red, blue = ticket["red"], ticket["blue"]
        if len(red) != 6 or len(set(red)) != 6 or any(type(n) is not int or not 1 <= n <= 33 for n in red):
            raise ValueError("参考推荐红球无效")
        if type(blue) is not int or not 1 <= blue <= 16:
            raise ValueError("参考推荐蓝球无效")
        identity = (tuple(sorted(red)), blue)
        if identity in seen:
            raise ValueError("参考引擎返回重复整注，未替换原票面")
        seen.add(identity)


def resolve_runtime_profile(requested: str) -> str:
    import yaml
    config = yaml.safe_load((ROOT / "tsys/config/config.yaml").read_text())
    profile = config.get("runtime", {}).get("default_profile", "standard") if requested == "source_default" else requested
    if profile != "supplied_default" and profile not in config.get("runtime", {}).get("profiles", {}):
        raise ValueError(f"参考源码未定义运行档位：{profile}")
    return profile


def cache_profile(value: dict, desired: str, *, allow_legacy: bool) -> dict:
    metadata = value.get("metadata", {})
    actual = metadata.get("runtime_profile", metadata.get("profile", "unknown_legacy"))
    if actual == desired:
        return {"requested_runtime_profile": desired, "cached_runtime_profile": actual,
                "runtime_profile_applied": True}
    if allow_legacy and actual in {"supplied_default", "unknown_legacy"}:
        return {"requested_runtime_profile": desired, "cached_runtime_profile": actual,
                "runtime_profile_applied": False,
                "profile_note": "保留旧默认配置的首次缓存；新源码网页档位从新输入开始应用，未把旧缓存标成新计算。"}
    raise ValueError(f"已有参考缓存档位为{actual}，请求为{desired}；拒绝冒用或覆盖。对照运行请使用独立cache-dir/work-dir且不传--ssq。")


def run_source_pipeline(main, config: dict, profile: str) -> dict:
    if profile == "supplied_default":
        return main.run_pipeline(config)
    # Use the actual source UI entrypoint: it applies all profile overrides,
    # validates the effective config and synchronizes holdout/generation counts.
    return main.dashboard_run(config, {"runtime_profile": profile})


def atomic_json(path: Path, value: dict, *, first: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".reference-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        if first:
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
        else:
            os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def validate_dantuo_plan(plan: dict) -> None:
    from math import comb
    numbers = []
    for key, high in [("red_dan", 33), ("red_tuo", 33), ("blue", 16)]:
        values = plan.get(key)
        if (not isinstance(values, list) or len(values) != len(set(values))
                or any(type(n) is not int or not 1 <= n <= high for n in values)):
            raise ValueError("参考胆拖号码无效")
        numbers.append(values)
    dan, tuo, blue = numbers
    if len(dan) > 5 or len(tuo) < 6 - len(dan) or set(dan) & set(tuo) or not blue:
        raise ValueError("参考胆拖胆拖结构无效")
    bets = comb(len(tuo), 6 - len(dan)) * len(blue)
    if (any(type(plan.get(k)) is not int for k in ["unit_price", "bet_count", "amount", "max_amount"])
            or plan.get("unit_price") != 2 or plan.get("bet_count") != bets
            or plan.get("amount") != bets * 2 or not 0 < bets <= 10
            or not bets * 2 <= plan.get("max_amount", 0) <= 20):
        raise ValueError("参考胆拖注数或预算不一致")


def verify_source_audit(payload: dict, work: Path) -> None:
    # Reuse the supplied verifier. The signing key stays in the private work dir.
    sys.path.insert(0, str(ROOT / "tsys"))
    try:
        from src.report.export import verify_forecast_audit
        result = verify_forecast_audit(payload, signing_key=(work / "audit.key").read_text().strip(),
                                       require_integrity=True)
        if not result["verified"]:
            raise ValueError("参考原始审计未通过核验")
    finally:
        sys.path.pop(0)


def validate_dantuo_cache(saved: dict, reference: dict, parent_sha: str) -> None:
    if (saved.get("version") != 1 or saved["reference_cache_sha256"] != parent_sha
            or any(saved.get(k) != reference[k] for k in ["data_sha256", "data_cutoff", "target_issue",
                                                        "engine_sha256", "config_sha256", "generated_at_unix"])):
        raise ValueError("参考胆拖绑定的首次5注缓存已变化")
    source = saved["source_forecast_json"]
    forecast = json.loads(source)
    if (sha(source.encode()) != saved["source_forecast_sha256"]
            or saved["dantuo_plan"] != forecast["metadata"]["dantuo_plan"]
            or forecast["source_period"] != "20" + reference["data_cutoff"]
            or forecast["target_period"] != "20" + reference["target_issue"]
            or [{"red": t["red"], "blue": t["blue"]} for t in forecast["tickets"]] != reference["tickets"]):
        raise ValueError("参考胆拖与同次原始报告不一致")
    exported = saved["exported_at_unix"]
    if type(exported) is not int or not reference["generated_at_unix"] <= exported <= time.time():
        raise ValueError("参考胆拖导出时间无效")
    validate_dantuo_plan(saved["dantuo_plan"])


def export_dantuo(args: argparse.Namespace, reference: dict, cache: Path) -> dict:
    """Export the original run's plan, including old completed runs; never rerun it."""
    destination = args.cache_dir.parent / "reference-dantuo-cache" / cache.name
    parent_sha = sha(cache.read_bytes())
    if destination.exists():
        saved = json.loads(destination.read_text())
        validate_dantuo_cache(saved, reference, parent_sha)
        return {"dantuo_status": "cached", "dantuo_cache_file": str(destination)}
    for work in sorted(args.work_dir.glob(reference["data_sha256"][:12] + "-*")):
        result_path, request_path = work / "result.json", work / "request.json"
        if not result_path.exists() or not request_path.exists():
            continue
        if json.loads(result_path.read_text()) != reference:
            continue
        request = json.loads(request_path.read_text())
        if (any(request.get(k) != reference[k] for k in ["data_sha256", "engine_sha256", "data_cutoff", "target_issue"])
                or sha((work / "input.csv").read_bytes()) != reference["data_sha256"]):
            raise ValueError("参考原始工作记录与缓存或输入不一致")
        source = work / "engine/logs/forecast_audit.json"
        raw = source.read_bytes()
        audit = json.loads(raw)
        verify_source_audit(audit, work)
        matches = [forecast for forecast in audit["forecasts"]
                   if forecast.get("source_period") == "20" + reference["data_cutoff"]
                   and forecast.get("target_period") == "20" + reference["target_issue"]
                   and [{"red": t["red"], "blue": t["blue"]} for t in forecast["tickets"]] == reference["tickets"]]
        if len(matches) != 1:
            raise ValueError("参考胆拖缺少唯一且与原始5注一致的来源记录")
        forecast = matches[0]
        plan = forecast["metadata"]["dantuo_plan"]
        validate_dantuo_plan(plan)
        source_json = json.dumps(forecast, ensure_ascii=False, sort_keys=True, allow_nan=False)
        value = {"version": 1, **{k: reference[k] for k in ["data_sha256", "data_cutoff", "target_issue",
                 "engine_sha256", "config_sha256", "generated_at_unix"]},
                 "exported_at_unix": int(time.time()), "reference_cache_sha256": parent_sha,
                 "source_audit_file_sha256": sha(raw), "source_forecast_json": source_json,
                 "source_forecast_sha256": sha(source_json.encode()), "dantuo_plan": plan}
        if not atomic_json(destination, value, first=True):
            if json.loads(destination.read_text())["source_forecast_sha256"] != value["source_forecast_sha256"]:
                raise ValueError("已有参考胆拖与原始报告不同，拒绝覆盖")
        validate_dantuo_cache(json.loads(destination.read_text()), reference, parent_sha)
        return {"dantuo_status": "ready", "dantuo_cache_file": str(destination)}
    return {"dantuo_status": "unavailable", "dantuo_reason": "原始计算报告尚未就绪，未按5注另算胆拖"}


def _run(args: argparse.Namespace) -> dict:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        data, rows = input_snapshot(args.data)
        data_sha, engine_sha = sha(data), engine_digest()
        requested_profile = getattr(args, "runtime_profile", "source_default")
        runtime_profile = resolve_runtime_profile(requested_profile)
        cutoff = rows[-1]["issue"]
        target = f"{int(cutoff) + 1:05}"
        cache = args.cache_dir / f"{data_sha}.json"
        status_path = args.work_dir / "status.json"
        state = {"data_sha256": data_sha, "data_cutoff": cutoff, "target_issue": target,
                 "engine_sha256": engine_sha, "started_at_unix": int(time.time())}
        if cache.exists():
            value = json.loads(cache.read_text())
            validate_cache(value, data_sha, engine_sha)
            profile_state = cache_profile(value, runtime_profile, allow_legacy=requested_profile == "source_default")
            if value["data_cutoff"] != cutoff or value["target_issue"] != target:
                raise ValueError("已有参考结果期号与当前历史不一致")
            result = {**state, **profile_state, "status": "cached", "cache_file": str(cache)}
        else:
            atomic_json(status_path, {**state, "status": "running"})
            try:
                work = Path(tempfile.mkdtemp(prefix=f"{data_sha[:12]}-", dir=args.work_dir))
                engine = work / "engine"
                shutil.copytree(ROOT / "tsys", engine, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                (work / "input.csv").write_bytes(data)
                raw = engine / "data/raw/history.csv"
                raw.parent.mkdir(parents=True)
                with raw.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file, lineterminator="\n")
                    writer.writerow(["period", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue"])
                    for row in rows:
                        writer.writerow(["20" + row["issue"], row["date"], *[row[f"r{i}"] for i in range(1, 7)], row["blue"]])
                request = {**state, "work": str(work), "engine": str(engine), "runtime_profile": runtime_profile}
                (work / "request.json").write_text(json.dumps(request))
                adapter_path = work / "adapter-worker.py"
                shutil.copy2(ROOT / "worker.py", adapter_path)
                key = secrets.token_urlsafe(48)
                key_path = work / "audit.key"
                key_path.write_text(key + "\n")
                key_path.chmod(0o600)
                environment = dict(os.environ, PYTHONHASHSEED="2026", OMP_NUM_THREADS="1",
                                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
                                   TSYS_AUDIT_HMAC_KEY=key)
                with (work / "worker.log").open("w") as log:
                    subprocess.run([sys.executable, str(adapter_path), "--child", str(work / "request.json")],
                                   cwd=engine, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=args.timeout)
                value = json.loads((work / "result.json").read_text())
                validate_cache(value, data_sha, engine_sha)
                profile_state = cache_profile(value, runtime_profile, allow_legacy=False)
                # A changed input cannot be silently frozen as the current forecast.
                current_data, _ = input_snapshot(args.data)
                if sha(current_data) != data_sha:
                    raise ValueError("计算期间开奖历史已更新，旧结果未发布；下一周期重新计算")
                if not atomic_json(cache, value, first=True):
                    value = json.loads(cache.read_text())
                    validate_cache(value, data_sha, engine_sha)
                    profile_state = cache_profile(value, runtime_profile, allow_legacy=False)
                result = {**state, **profile_state, "status": "ready", "cache_file": str(cache), "work": str(work)}
            except Exception as error:
                atomic_json(status_path, {**state, "status": "error", "error": str(error), "finished_at_unix": int(time.time())})
                raise
        result.update(export_dantuo(args, value, cache))
        atomic_json(status_path, {**result, "finished_at_unix": int(time.time())})
        if args.ssq:
            completed = subprocess.run([str(args.ssq), "reference-freeze", "--data", str(args.data),
                                        "--out", str(args.publish_dir)], capture_output=True, text=True, timeout=120)
            result["freeze_status"] = "accepted" if completed.returncode == 0 else "not_frozen"
            result["freeze_result"] = (completed.stdout if completed.returncode == 0 else completed.stderr)[-2000:]
            atomic_json(status_path, {**result, "finished_at_unix": int(time.time())})
        return result


def run(args: argparse.Namespace) -> dict:
    try:
        return _run(args)
    except Exception as error:
        try:
            data, rows = input_snapshot(args.data)
            state = {"data_sha256": sha(data), "data_cutoff": rows[-1]["issue"]}
        except Exception:
            state = {"data_sha256": None}
        atomic_json(args.work_dir / "status.json", {**state, "status": "error", "error": str(error),
                                                    "finished_at_unix": int(time.time())})
        raise


def child(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    engine, work = Path(request["engine"]), Path(request["work"])
    sys.path.insert(0, str(engine))
    import main
    config = main.load_config(engine / "config/config.yaml")
    if config.get("notification") != {"email": {"enabled": False}}:
        raise ValueError("参考引擎通知必须完全禁用")
    # The project's supplied dates include four explicit early-history exceptions.
    # Validate them in input_snapshot; retain their actual bytes/features instead
    # of replacing dates with a derived schedule or dropping historical rows.
    config["data"]["validate_source_dates"] = False
    main.configure_logging(config)
    profile = request.get("runtime_profile", "supplied_default")
    report = run_source_pipeline(main, config, profile)
    actual_profile = report["config_snapshot"].get("runtime", {}).get("active_profile") if profile != "supplied_default" else "supplied_default"
    if actual_profile != profile:
        raise ValueError("源码报告未确认请求的运行档位，拒绝保存误标结果")
    forecast = report["forecast_archive"]
    if forecast["source_period"] != "20" + request["data_cutoff"] or forecast["target_period"] != "20" + request["target_issue"]:
        raise ValueError("参考报告期号与输入不一致")
    original = forecast["tickets"]
    tickets = [{"red": [int(n) for n in t["red"]], "blue": int(t["blue"])} for t in original]
    value = {"version": 1, **{key: request[key] for key in ["data_sha256", "data_cutoff", "target_issue", "engine_sha256"]},
             "generated_at_unix": int(time.time()), "config_sha256": report["run_manifest"]["config_hash"],
             "tickets": tickets,
             "metadata": {"implementation": "Tsys original dashboard_run" if profile != "supplied_default" else "Tsys original run_pipeline",
                          "profile": profile, "runtime_profile": profile,
                          "adapter_sha256": ADAPTER_SHA256,
                          "runtime_packages": report["run_metadata"]["packages"],
                          "backtest_periods": forecast["metadata"]["forecast_publish_gate"]["effective_backtest_periods"],
                          "random_seed": report["run_manifest"]["seed"],
                          "monte_carlo_samples": report["config_snapshot"]["generation"]["monte_carlo_samples"],
                          "optimizer_methods": report["config_snapshot"]["optimizer"]["methods"],
                          "xgboost_distribution": "xgboost==3.0.2; official manylinux2014 CPU wheel",
                          "date_validation_adapter": {"reference_fixed_weekday_check": False,
                                                      "project_dates_preserved": True,
                                                      "validation": "ISO date, matching issue year, chronological unique dates, weekdays with four explicit source exceptions",
                                                      "source_exceptions": sorted([list(x) for x in SOURCE_DATE_EXCEPTIONS])},
                          "started_at_unix": request["started_at_unix"], "notification_enabled": False,
                          "reference_scope": forecast.get("forecast_scope"), "original_tickets": original,
                          "prediction_mode": forecast.get("prediction_mode"), "run_manifest": report.get("run_manifest"),
                          "model_release": report.get("model_release"), "source_tree_sha256": main.source_tree_fingerprint(engine)}}
    atomic_json(work / "result.json", value, first=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--runtime-profile", choices=["source_default", "supplied_default", "fast", "standard", "research"], default="source_default",
                        help="source_default采用源码网页默认档；supplied_default仅复现旧适配器的直接run_pipeline入口")
    parser.add_argument("--ssq", type=Path)
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    if not all([args.data, args.cache_dir, args.work_dir]) or (args.ssq and not args.publish_dir):
        parser.error("--data --cache-dir --work-dir are required; --ssq requires --publish-dir")
    for key in ["data", "cache_dir", "work_dir", "ssq", "publish_dir"]:
        if getattr(args, key):
            setattr(args, key, getattr(args, key).resolve())
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    cli()
