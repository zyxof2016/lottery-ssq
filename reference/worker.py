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


def _run(args: argparse.Namespace) -> dict:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        data, rows = input_snapshot(args.data)
        data_sha, engine_sha = sha(data), engine_digest()
        cutoff = rows[-1]["issue"]
        target = f"{int(cutoff) + 1:05}"
        cache = args.cache_dir / f"{data_sha}.json"
        status_path = args.work_dir / "status.json"
        state = {"data_sha256": data_sha, "data_cutoff": cutoff, "target_issue": target,
                 "engine_sha256": engine_sha, "started_at_unix": int(time.time())}
        if cache.exists():
            value = json.loads(cache.read_text())
            validate_cache(value, data_sha, engine_sha)
            if value["data_cutoff"] != cutoff or value["target_issue"] != target:
                raise ValueError("已有参考结果期号与当前历史不一致")
            result = {**state, "status": "cached", "cache_file": str(cache)}
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
                request = {**state, "work": str(work), "engine": str(engine)}
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
                # A changed input cannot be silently frozen as the current forecast.
                current_data, _ = input_snapshot(args.data)
                if sha(current_data) != data_sha:
                    raise ValueError("计算期间开奖历史已更新，旧结果未发布；下一周期重新计算")
                if not atomic_json(cache, value, first=True):
                    validate_cache(json.loads(cache.read_text()), data_sha, engine_sha)
                result = {**state, "status": "ready", "cache_file": str(cache), "work": str(work)}
            except Exception as error:
                atomic_json(status_path, {**state, "status": "error", "error": str(error), "finished_at_unix": int(time.time())})
                raise
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
    report = main.run_pipeline(config)
    forecast = report["forecast_archive"]
    if forecast["source_period"] != "20" + request["data_cutoff"] or forecast["target_period"] != "20" + request["target_issue"]:
        raise ValueError("参考报告期号与输入不一致")
    original = forecast["tickets"]
    tickets = [{"red": [int(n) for n in t["red"]], "blue": int(t["blue"])} for t in original]
    value = {"version": 1, **{key: request[key] for key in ["data_sha256", "data_cutoff", "target_issue", "engine_sha256"]},
             "generated_at_unix": int(time.time()), "config_sha256": main.config_fingerprint(config),
             "tickets": tickets,
             "metadata": {"implementation": "Tsys original run_pipeline", "profile": "supplied_default",
                          "adapter_sha256": ADAPTER_SHA256,
                          "runtime_packages": report["run_metadata"]["packages"],
                          "backtest_periods": config["backtest"]["periods"], "random_seed": config["project"]["random_seed"],
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
