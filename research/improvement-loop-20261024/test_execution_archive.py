"""Synthetic subprocess tests for portable completed-attempt evidence only."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import execution
import execution_archive as archive
from ledger import canonical


OWNER = r'''
import json, pathlib, sys, time
output, plan_sha, delay = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
output.mkdir()
if delay: time.sleep(delay)
(output / "receipt.json").write_text(json.dumps({
    "schema": "SyntheticNativeV1",
    "plan_sha256": plan_sha,
    "status": "scientific_failure_unassessed",
}))
'''


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = self.base / "store"
        self.owner = self.base / "owner.py"
        self.owner.write_text(OWNER)
        self.plan = self.base / "plan.json"
        self.plan.write_text(canonical({"schema": "InventedPlanV1", "purpose": "archive software test"}))

    def spec(self, attempt_id, previous=None, delay=0):
        output = self.base / ("output-" + attempt_id)
        plan_sha = execution.pin(self.plan)["sha256"]
        return {"schema": "LocalExecutionSpecV1", "cycle_id": "archive-test-cycle", "job_key": "same-job",
                "attempt_id": attempt_id,
                "argv": [sys.executable, str(self.owner), str(output), plan_sha, str(delay)],
                "executable": execution.executable_pin(sys.executable), "cwd": str(self.base),
                "environment": {}, "input_pins": [execution.pin(self.owner), execution.pin(self.plan)],
                "output_directory": str(output),
                "native_receipts": [{"path": "receipt.json", "required_fields":
                                     {"schema": "SyntheticNativeV1", "plan_sha256": plan_sha}}],
                "timeout_seconds": 5, "grace_seconds": 0.05,
                "previous_attempt": previous, "retry_reason": "explicit test retry" if previous else ""}

    def completed(self, attempt_id, previous=None):
        spec = self.spec(attempt_id, previous)
        execution.prepare(self.store, spec, hashlib.sha256(canonical(spec).encode()).hexdigest())
        execution.launch(self.store, attempt_id)
        until = time.monotonic() + 8
        while time.monotonic() < until:
            state = execution.inspect(self.store, attempt_id)
            if state["attempt_terminal"] and not state["job_lock_held"]:
                return state
            time.sleep(0.02)
        self.fail("synthetic owner did not complete")

    def export(self, attempt_id="a0"):
        destination = self.base / "archive"
        result = archive.export_completed_attempt(self.store, attempt_id, destination)
        self.assertEqual(result["qualification"], "unassessed")
        return destination, result

    def test_relocated_archive_survives_source_store_and_inputs_removal(self):
        first = self.completed("a0")
        second = self.completed("a1", {"attempt_id": "a0", "receipt": first["terminal_receipt"]})
        source_pins = [execution.pin(self.owner), execution.pin(self.plan), first["terminal_receipt"],
                       second["terminal_receipt"]]
        destination, result = self.export("a1")
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual([execution.pin(item["path"]) for item in source_pins], source_pins)
        moved = self.base / "moved-archive"
        shutil.move(destination, moved)
        shutil.rmtree(self.store)
        self.owner.unlink()
        self.plan.unlink()
        for attempt_id in ("a0", "a1"):
            shutil.rmtree(self.base / ("output-" + attempt_id))
        with patch.object(execution, "inspect", side_effect=AssertionError("offline archive consulted live execution")):
            inspected = archive.inspect_archive(moved, result["manifest"]["sha256"], result["manifest"]["size"])
        self.assertEqual(inspected["attempt_count"], 2)
        self.assertEqual(inspected["qualification"], "unassessed")
        self.assertTrue(inspected["portable_evidence_verified"])
        manifest = json.loads((moved / "manifest.json").read_text())
        self.assertEqual(manifest["attempts"][0]["terminal"], second["terminal_receipt"])
        self.assertEqual(manifest["attempts"][1]["terminal"], first["terminal_receipt"])

    def test_cli_external_manifest_pin_and_no_overwrite_or_source_mutation(self):
        state = self.completed("a0")
        before = [execution.pin(self.owner), execution.pin(self.plan), state["terminal_receipt"]]
        with self.assertRaisesRegex(archive.ArchiveError, "outside source store"):
            archive.export_completed_attempt(self.store, "a0", self.store / "archive")
        with self.assertRaisesRegex(archive.ArchiveError, "outside native output"):
            archive.export_completed_attempt(self.store, "a0", self.base / "output-a0" / "archive")
        cli = [sys.executable, "-B", str(Path(archive.__file__)), "export", "--root", str(self.store),
               "--attempt", "a0", "--destination", str(self.base / "archive")]
        result = subprocess.run(cli, capture_output=True, text=True, timeout=15, check=True)
        exported = json.loads(result.stdout)
        self.assertEqual([execution.pin(item["path"]) for item in before], before)
        self.assertEqual(json.loads(subprocess.run(
            [sys.executable, "-B", str(Path(archive.__file__)), "inspect", "--archive", exported["archive"],
             "--manifest-sha256", exported["manifest"]["sha256"]],
            capture_output=True, text=True, timeout=15, check=True).stdout)["qualification"], "unassessed")
        with self.assertRaisesRegex(archive.ArchiveError, "fresh archive"):
            archive.export_completed_attempt(self.store, "a0", exported["archive"])
        with self.assertRaisesRegex(archive.ArchiveError, "external manifest pin"):
            archive.inspect_archive(exported["archive"], "0" * 64)

    def test_corrupt_missing_and_conflicting_archived_object_fail_closed(self):
        self.completed("a0")
        destination, result = self.export()
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        object_path = destination / "objects" / "sha256" / manifest["objects"][0]["sha256"]
        original = object_path.read_bytes()
        object_path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(archive.ArchiveError, "fingerprint"):
            archive.inspect_archive(destination, result["manifest"]["sha256"])
        object_path.write_bytes(original)
        object_path.unlink()
        with self.assertRaisesRegex(archive.ArchiveError, "inventory"):
            archive.inspect_archive(destination, result["manifest"]["sha256"])
        object_path.write_bytes(original)
        altered = copy.deepcopy(manifest)
        duplicate = dict(altered["objects"][0], sha256=altered["objects"][1]["sha256"],
                         size=altered["objects"][1]["size"])
        altered["objects"].insert(1, duplicate)
        altered["objects"].sort(key=lambda item: item["path"])
        data = canonical(altered).encode()
        manifest_path.write_bytes(data)
        with self.assertRaisesRegex(archive.ArchiveError, "conflicting pin"):
            archive.inspect_archive(destination, hashlib.sha256(data).hexdigest())

    def test_corrupt_copied_object_prevents_manifest_publication(self):
        state = self.completed("a0")
        source_pin = state["terminal_receipt"]
        original_copy = archive._copy_pinned
        corrupted = False

        def corrupt_after_copy(item, destination):
            nonlocal corrupted
            original_copy(item, destination)
            if not corrupted:
                object_path = destination / "objects" / "sha256" / item["sha256"]
                with object_path.open("ab") as stream:
                    stream.write(b"injected corruption")
                corrupted = True

        destination = self.base / "archive"
        with patch.object(archive, "_copy_pinned", side_effect=corrupt_after_copy):
            with self.assertRaisesRegex(archive.ArchiveError, "fingerprint"):
                archive.export_completed_attempt(self.store, "a0", destination)
        self.assertTrue(corrupted)
        self.assertTrue(destination.is_dir())
        self.assertFalse((destination / "manifest.json").exists())
        self.assertEqual(execution.pin(source_pin["path"]), source_pin)
        with self.assertRaisesRegex(archive.ArchiveError, "manifest missing"):
            archive.inspect_archive(destination, "0" * 64)

    def test_live_attempt_export_rejected_then_completion_exportable(self):
        spec = self.spec("a0", delay=0.8)
        execution.prepare(self.store, spec, hashlib.sha256(canonical(spec).encode()).hexdigest())
        execution.launch(self.store, "a0")
        with self.assertRaisesRegex(archive.ArchiveError, "not completed and quiescent"):
            archive.export_completed_attempt(self.store, "a0", self.base / "archive")
        self.assertFalse((self.base / "archive").exists())
        until = time.monotonic() + 8
        while time.monotonic() < until:
            state = execution.inspect(self.store, "a0")
            if state["attempt_terminal"] and not state["job_lock_held"]:
                break
            time.sleep(0.02)
        else:
            self.fail("synthetic owner did not complete")
        self.export()

    def test_copied_lock_inode_is_original_metadata_not_a_portable_live_identity(self):
        self.completed("a0")
        destination, result = self.export()
        manifest = json.loads((destination / "manifest.json").read_text())
        prepared = manifest["attempts"][0]["prepared"]
        copied = (destination / "objects" / "sha256" / prepared["sha256"]).read_bytes()
        self.assertIn(b'"job_lock"', copied)
        self.assertEqual(archive.inspect_archive(destination, result["manifest"]["sha256"])["qualification"], "unassessed")


if __name__ == "__main__":
    unittest.main()
