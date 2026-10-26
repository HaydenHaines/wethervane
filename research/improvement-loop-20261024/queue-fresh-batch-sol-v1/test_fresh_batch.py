"""Bounded temporary-store admission checks. No executor dispatch or models."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import fresh_batch as batch
import experiment_queue as queue
import execution as executor
import test_experiment_queue as fixtures


class FreshBatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.QueueTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.base = self.fixture.root, self.fixture.base

    def manifest(self, count=2, jobs=None, batch_id="batch"):
        plans = []
        self.plans = []
        for i in range(count):
            job = jobs[i] if jobs else self.fixture.job("job-" + str(i))
            plan = dict(schema="LocalExperimentQueuePlanV1", queue_id="q-" + str(i), jobs=[job])
            path = self.base / ("plan-" + str(i) + ".json")
            path.write_text(json.dumps(plan))
            self.plans.append(plan)
            plans.append(executor.pin(path))
        value = dict(schema="FreshQueueBatchPlanV1", batch_id=batch_id,
                     store=executor.pin(self.root / "store.json"), sources=batch.sources(), plans=plans)
        path = self.base / "manifest.json"
        path.write_text(json.dumps(value))
        return path, executor.pin(path)["sha256"]

    def test_fixed_batch_one_scan_no_launch_and_ordinary_idempotence(self):
        path, sha = self.manifest()
        with patch.object(queue, "entries", wraps=queue.entries) as scan:
            receipt = batch.admit(self.root, path, sha)
        self.assertEqual(scan.call_count, 1)
        result = queue.document(receipt["path"])
        self.assertEqual(len(result["admissions"]), 2)
        self.assertEqual(result["qualification"], "unassessed")
        for plan, pin in zip(self.plans, result["admissions"]):
            self.assertEqual(queue.enqueue(self.root, plan, queue.digest(plan)), pin)
        self.assertFalse((self.root / "executions").exists())

    def test_cross_batch_and_history_conflicts(self):
        for field in ("attempt_id", "output_directory"):
            with self.subTest(field=field):
                a, b = self.fixture.job("a"), self.fixture.job("b")
                b["execution_spec"][field] = a["execution_spec"][field]
                path, sha = self.manifest(jobs=[a, b])
                with self.assertRaisesRegex(ValueError, "already belongs"):
                    batch.admit(self.root, path, sha)
                self.assertFalse(any((self.root / "queues").iterdir()))
        old = self.fixture.job("a")
        self.fixture.admit([old], "old")
        path, sha = self.manifest(1, jobs=[old])
        with self.assertRaisesRegex(ValueError, "already belongs"):
            batch.admit(self.root, path, sha)

    def test_untrusted_and_changed_manifest_plan_sources(self):
        path, sha = self.manifest()
        with self.assertRaisesRegex(ValueError, "untrusted manifest"):
            batch.admit(self.root, path, "0" * 64)
        (self.base / "plan-0.json").write_text("{}")
        with self.assertRaises(ValueError):
            batch.admit(self.root, path, sha)
        path, sha = self.manifest()
        value = queue.document(path)
        value["sources"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "source roster"):
            batch.admit(self.root, path, executor.pin(path)["sha256"])
        self.assertFalse(any((self.root / "queues").iterdir()))

    def test_freshness_partial_corrupt_and_symlink_rejected(self):
        path, sha = self.manifest()
        partial = self.root / "queues" / "partial"
        partial.mkdir()
        with self.assertRaisesRegex(ValueError, "partial admission"):
            batch.admit(self.root, path, sha)
        partial.rmdir()
        output = Path(self.plans[0]["jobs"][0]["execution_spec"]["output_directory"])
        output.symlink_to(self.base / "missing")
        with self.assertRaisesRegex(ValueError, "symlink|canonical path"):
            batch.admit(self.root, path, sha)
        output.unlink()
        old = self.fixture.admit([self.fixture.job("old")], "old")
        (self.root / "queues/old/receipt.json").write_text("{}")
        with self.assertRaises((ValueError, KeyError)):
            batch.admit(self.root, path, sha)

    def test_existing_queue_retry_and_resume_require_ordinary_enqueue(self):
        path, sha = self.manifest(1)
        job = copy.deepcopy(self.plans[0]["jobs"][0])
        job["execution_spec"]["previous_attempt"] = dict(attempt_id="prior", receipt={})
        job["execution_spec"]["retry_reason"] = "explicit retry"
        path, sha = self.manifest(1, jobs=[job])
        with self.assertRaisesRegex(ValueError, "resume/retry"):
            batch.admit(self.root, path, sha)
        job = self.fixture.job("fresh")
        job["resume_prelaunch"] = dict(queue_id="old", job_id="old", result={}, release={}, binding={}, reason="resume")
        path, sha = self.manifest(1, jobs=[job])
        with self.assertRaisesRegex(ValueError, "resume/retry"):
            batch.admit(self.root, path, sha)
        path, sha = self.manifest(1)
        plan = self.plans[0]
        queue.enqueue(self.root, plan, queue.digest(plan))
        with self.assertRaisesRegex(ValueError, "already exists"):
            batch.admit(self.root, path, sha)

    def test_crash_prefix_and_partial_plan_recover_with_ordinary_enqueue(self):
        path, sha = self.manifest(3)
        original = executor.publish
        def crash(target, value):
            if Path(target) == self.root / "queues/q-1/receipt.json":
                raise OSError("simulated crash")
            return original(target, value)
        with patch.object(executor, "publish", side_effect=crash):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                batch.admit(self.root, path, sha)
        self.assertTrue((self.root / "batch-admissions/batch/intent.json").exists())
        self.assertFalse((self.root / "batch-admissions/batch/receipt.json").exists())
        first = executor.pin(self.root / "queues/q-0/receipt.json")
        self.assertTrue((self.root / "queues/q-1/plan.json").exists())
        self.assertFalse((self.root / "queues/q-2").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            batch.admit(self.root, path, sha)
        for plan in self.plans:
            queue.enqueue(self.root, plan, queue.digest(plan))
        self.assertEqual(first, executor.pin(first["path"]))
        self.assertEqual(len(list(queue.entries(self.root))), 3)
        self.assertFalse((self.root / "executions").exists())

    def test_bounded_scale_one_history_scan(self):
        path, sha = self.manifest(100)
        start = time.monotonic()
        with patch.object(queue, "entries", wraps=queue.entries) as scan:
            receipt = batch.admit(self.root, path, sha)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(len(queue.document(receipt["path"])["admissions"]), 100)
        self.assertLess(time.monotonic() - start, 30)
        self.assertFalse((self.root / "executions").exists())

    def test_mid_batch_plan_tamper_preserves_prefix_and_durable_intent(self):
        path, sha = self.manifest(2)
        original = executor.publish
        def tamper(target, value):
            result = original(target, value)
            if Path(target) == self.root / "queues/q-0/receipt.json":
                (self.base / "plan-1.json").write_text("{}")
            return result
        with patch.object(executor, "publish", side_effect=tamper):
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                batch.admit(self.root, path, sha)
        self.assertTrue((self.root / "queues/q-0/receipt.json").exists())
        self.assertFalse((self.root / "queues/q-1").exists())
        self.assertTrue((self.root / "batch-admissions/batch/intent.json").exists())
        self.assertFalse((self.root / "batch-admissions/batch/receipt.json").exists())

    def test_invalid_later_plan_writes_no_prefix_and_external_attempt_rejected(self):
        path, sha = self.manifest()
        plan = queue.document(self.base / "plan-1.json")
        plan["jobs"][0]["dependencies"] = [dict(job_id="missing", role="receipt", path_arguments=[1], sha256_arguments=[2])]
        (self.base / "plan-1.json").write_text(json.dumps(plan))
        manifest = queue.document(path)
        manifest["plans"][1] = executor.pin(self.base / "plan-1.json")
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            batch.admit(self.root, path, executor.pin(path)["sha256"])
        self.assertFalse(any((self.root / "queues").iterdir()))
        self.assertFalse((self.root / "batch-admissions").exists())
        path, sha = self.manifest()
        (self.root / "executions/attempts/job-0").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            batch.admit(self.root, path, sha)

    def test_cli_waits_for_existing_dispatcher_lock(self):
        path, sha = self.manifest(1)
        _, descriptor = queue.dispatcher(self.root)
        child = subprocess.Popen([sys.executable, "-B", str(Path(batch.__file__).resolve()),
                                 "--root", str(self.root), "--manifest", str(path), "--sha256", sha],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                child.communicate(timeout=.2)
            self.assertFalse(any((self.root / "queues").iterdir()))
        finally:
            os.close(descriptor)
        try:
            stdout, stderr = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 0, stderr)
            executor.verify_pin(json.loads(stdout))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    unittest.main()
