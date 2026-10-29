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


if __name__ == "__main__":
    unittest.main()
