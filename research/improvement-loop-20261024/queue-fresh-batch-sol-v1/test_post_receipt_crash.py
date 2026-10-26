"""Regression at the immutable admission receipt / job-directory boundary."""
import unittest
from pathlib import Path
from unittest.mock import patch

import fresh_batch as batch
import experiment_queue as queue
import execution as executor
import test_fresh_batch as fixtures


class PostReceiptCrashTests(unittest.TestCase):
    def test_receipt_survives_scaffolding_crash_and_ordinary_recovery(self):
        fixture = fixtures.FreshBatchTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path, sha = fixture.manifest(1)
        job_directory = fixture.root / "queues/q-0/jobs/job-0"
        original = executor.ensure_directory
        def crash(target):
            if Path(target) == job_directory:
                raise OSError("injected after admission receipt")
            return original(target)
        with patch.object(executor, "ensure_directory", side_effect=crash):
            with self.assertRaisesRegex(OSError, "after admission receipt"):
                batch.admit(fixture.root, path, sha)
        pin = executor.pin(fixture.root / "queues/q-0/receipt.json")
        self.assertFalse(job_directory.exists())
        plan = fixture.plans[0]
        self.assertEqual(queue.enqueue(fixture.root, plan, queue.digest(plan)), pin)
        self.assertFalse(job_directory.exists())  # Ordinary idempotence is unchanged.
        # The unchanged dispatcher restores its job scaffold before considering
        # dispatch. Stop at bind, before any reservation, preparation, or launch.
        with patch.object(queue, "bind", side_effect=AssertionError("stop before bind")), \
             patch.object(executor, "prepare") as prepare, patch.object(executor, "launch") as launch:
            with self.assertRaisesRegex(AssertionError, "stop before bind"):
                queue.tick(fixture.root)
            prepare.assert_not_called()
            launch.assert_not_called()
        self.assertTrue(job_directory.is_dir())
        self.assertEqual(executor.pin(pin["path"]), pin)
        self.assertFalse((job_directory / "reservation.json").exists())
        self.assertFalse((fixture.root / "executions").exists())
        self.assertTrue((fixture.root / "batch-admissions/batch/intent.json").exists())
        self.assertFalse((fixture.root / "batch-admissions/batch/receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
