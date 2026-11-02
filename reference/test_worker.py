import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("reference_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class WorkerTest(unittest.TestCase):
    def test_source_manifest_and_sanitized_config(self):
        import yaml
        self.assertEqual(len(worker.engine_digest()), 64)
        config = yaml.safe_load((worker.ROOT / "tsys/config/config.yaml").read_text())
        self.assertEqual(config["notification"], {"email": {"enabled": False}})

    def test_canonical_hash_and_current_period(self):
        path = worker.ROOT.parent / "data/ssq.csv"
        data, rows = worker.input_snapshot(path)
        self.assertEqual(data, path.read_bytes().replace(b"\r\n", b"\n"))
        self.assertEqual(len(rows[-1]["issue"]), 5)

    def test_duplicate_history_and_invalid_tickets_rejected(self):
        data, _ = worker.input_snapshot(worker.ROOT.parent / "data/ssq.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_bytes(data + data.splitlines(keepends=True)[-1])
            with self.assertRaises(ValueError):
                worker.input_snapshot(path)
        value = {"version": 1, "data_sha256": "data", "engine_sha256": "engine",
                 "tickets": [{"red": [1, 1, 2, 3, 4, 5], "blue": 1}] * 5}
        with self.assertRaises(ValueError):
            worker.validate_cache(value, "data", "engine")

    def test_new_nonstandard_weekday_is_not_silently_accepted(self):
        data, rows = worker.input_snapshot(worker.ROOT.parent / "data/ssq.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            previous = rows[-1]["date"].encode()
            last_line = data.splitlines(keepends=True)[-1]
            date = worker.dt.date.fromisoformat(rows[-1]["date"])
            while date.weekday() in (1, 3, 6):
                date += worker.dt.timedelta(days=1)
            path.write_bytes(data[:-len(last_line)] + last_line.replace(previous, str(date).encode()))
            with self.assertRaisesRegex(ValueError, "常规开奖星期"):
                worker.input_snapshot(path)

    def test_atomic_first_result_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            self.assertTrue(worker.atomic_json(path, {"a": 1}, first=True))
            original = path.read_bytes()
            self.assertFalse(worker.atomic_json(path, {"a": 2}, first=True))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(".reference-*")), [])

    def test_matching_cache_skips_engine_and_changed_engine_rejected(self):
        source = worker.ROOT.parent / "data/ssq.csv"
        data, rows = worker.input_snapshot(source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache" / f"{worker.sha(data)}.json"
            value = {"version": 1, "data_sha256": worker.sha(data), "engine_sha256": worker.engine_digest(),
                     "data_cutoff": rows[-1]["issue"], "target_issue": f"{int(rows[-1]['issue']) + 1:05}",
                     "generated_at_unix": 1, "config_sha256": "a" * 64,
                     "tickets": [{"red": [1, 2, 3, 4, 5, 6], "blue": i} for i in range(1, 6)]}
            worker.atomic_json(cache, value, first=True)
            before = cache.read_bytes()
            args = argparse.Namespace(data=source, cache_dir=cache.parent, work_dir=root / "work", ssq=None)
            self.assertEqual(worker.run(args)["status"], "cached")
            self.assertEqual(cache.read_bytes(), before)
            args.ssq, args.publish_dir = root / "ssq", root / "publish"
            with mock.patch.object(worker.subprocess, "run", return_value=argparse.Namespace(returncode=1, stdout="", stderr="已过截止")):
                result = worker.run(args)
            self.assertEqual(result["status"], "cached")
            self.assertEqual(result["freeze_status"], "not_frozen")
            self.assertEqual(json.loads((args.work_dir / "status.json").read_text())["status"], "cached")
            self.assertEqual(cache.read_bytes(), before)
            value["engine_sha256"] = "different"
            worker.atomic_json(cache, value)
            with self.assertRaises(ValueError):
                worker.run(args)
            self.assertEqual(json.loads((args.work_dir / "status.json").read_text())["status"], "error")

    def test_original_dantuo_export_uses_same_run_and_preserves_five_ticket_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(cache_dir=root / "reference-cache", work_dir=root / "work")
            data = b"original bound history"
            reference = {"data_sha256": worker.sha(data), "data_cutoff": "26125", "target_issue": "26126",
                         "engine_sha256": "a" * 64, "config_sha256": "b" * 64, "generated_at_unix": 5,
                         "tickets": [{"red": [1, 2, 3, 4, 5, 6], "blue": i} for i in range(1, 6)]}
            cache = args.cache_dir / (reference["data_sha256"] + ".json")
            worker.atomic_json(cache, reference, first=True)
            original = cache.read_bytes()
            self.assertEqual(worker.export_dantuo(args, reference, cache)["dantuo_status"], "unavailable")
            work = args.work_dir / (reference["data_sha256"][:12] + "-original")
            work.mkdir(parents=True)
            worker.atomic_json(work / "result.json", reference)
            worker.atomic_json(work / "request.json", reference)
            (work / "input.csv").write_bytes(data)
            plan = {"red_dan": [14, 24], "red_tuo": [2, 20, 25, 27, 30], "blue": [4, 5],
                    "unit_price": 2, "bet_count": 10, "amount": 20, "max_amount": 20}
            forecast = {"source_period": "2026125", "target_period": "2026126", "tickets": reference["tickets"],
                        "metadata": {"dantuo_plan": plan}}
            source = work / "engine/logs/forecast_audit.json"
            worker.atomic_json(source, {"forecasts": [forecast]})
            with mock.patch.object(worker, "verify_source_audit") as verify:
                result = worker.export_dantuo(args, reference, cache)
                self.assertEqual(result["dantuo_status"], "ready")
                verify.assert_called_once()
                saved_path = Path(result["dantuo_cache_file"])
                saved_bytes = saved_path.read_bytes()
                saved = json.loads(saved_bytes)
                self.assertEqual(saved["dantuo_plan"], plan)
                self.assertEqual(saved["reference_cache_sha256"], worker.sha(original))
                self.assertEqual(worker.export_dantuo(args, reference, cache)["dantuo_status"], "cached")
                verify.assert_called_once()
                self.assertEqual(saved_path.read_bytes(), saved_bytes)
                self.assertEqual(cache.read_bytes(), original)
                saved["dantuo_plan"]["blue"] = [8, 9]
                worker.atomic_json(saved_path, saved)
                with self.assertRaisesRegex(ValueError, "同次原始报告"):
                    worker.export_dantuo(args, reference, cache)

    def test_dantuo_compound_budget_and_invalid_numbers(self):
        plan = {"red_dan": [], "red_tuo": [1, 2, 3, 4, 5, 6, 7], "blue": [9],
                "unit_price": 2, "bet_count": 7, "amount": 14, "max_amount": 20}
        worker.validate_dantuo_plan(plan)
        for key, value in [("red_dan", [1]), ("blue", [9, 9]), ("amount", 20), ("bet_count", 7.0),
                           ("max_amount", 12), ("red_tuo", [1, 2, 3, 4, 5, 6, 34])]:
            with self.assertRaises(ValueError):
                worker.validate_dantuo_plan({**plan, key: value})


if __name__ == "__main__":
    unittest.main()
