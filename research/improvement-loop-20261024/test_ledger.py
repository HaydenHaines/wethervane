"""Bounded recorder tests; no scientific experiments or registry activation."""

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ledger import Ledger, LedgerError


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / "journal")
        self.binding = dict(
            incumbent_config={"joint_model": "pinned-config", "layers": {"population": "v2", "polls": "v1"}},
            scope={"cycles": [2018, 2022], "target": "final winner"},
            target_layer="measurement",
            hypothesis="integrating scale uncertainty improves calibration",
        )
        self.ledger.create_cycle("create", "cycle-1", **self.binding)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def artifact(self, name="plan", content=b"prespecified plan"):
        source = self.root / name
        source.write_bytes(content)
        return self.ledger.put_artifact("artifact-" + name, source)["sha256"]

    def test_reopen_binding_and_copied_artifact(self):
        digest = self.artifact()
        (self.root / "plan").write_bytes(b"changed source")
        self.ledger.checkpoint("plan", "cycle-1", 0, "planned", "Plan retained", {"plan": digest})
        self.ledger.close()
        self.ledger = Ledger(self.root / "journal")
        exported = self.ledger.export("cycle-1")
        self.assertEqual(exported["binding"], self.binding)
        self.assertEqual(self.ledger._artifact(digest), b"prespecified plan")
        self.assertEqual(exported["checkpoints"][1]["references"]["plan"], digest)

    def test_idempotence_payload_conflict_and_stale_write(self):
        result = self.ledger.checkpoint("step", "cycle-1", 0, "planned", "Plan")
        self.assertEqual(result, self.ledger.checkpoint("step", "cycle-1", 0, "planned", "Plan"))
        with self.assertRaisesRegex(LedgerError, "different payload"):
            self.ledger.checkpoint("step", "cycle-1", 0, "planned", "Different")
        other = Ledger(self.root / "journal")
        try:
            with self.assertRaisesRegex(LedgerError, "stale"):
                other.checkpoint("stale", "cycle-1", 0, "developing", "Old reader")
            # A failed command does not reserve its id.
            other.checkpoint("stale", "cycle-1", 1, "developing", "Retry after refresh")
        finally:
            other.close()
        self.assertEqual(len(self.ledger.export("cycle-1")["checkpoints"]), 3)

    def test_transaction_rolls_back_checkpoint_and_command_together(self):
        # Fail after checkpoint insertion, at the command insert, using a temporary
        # SQLite trigger rather than a production-only failure injection API.
        self.ledger.db.execute(
            "CREATE TEMP TRIGGER reject_command BEFORE INSERT ON commands "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected failure"):
            self.ledger.checkpoint("retry", "cycle-1", 0, "planned", "Plan")
        self.assertEqual(self.ledger.status("cycle-1")["revision"], 0)
        self.assertIsNone(self.ledger.db.execute("SELECT id FROM commands WHERE id='retry'").fetchone())
        self.ledger.db.execute("DROP TRIGGER reject_command")
        self.ledger.checkpoint("retry", "cycle-1", 0, "planned", "Plan")

    def test_corrupt_artifact_rejected_on_reference_export_and_retry(self):
        digest = self.artifact()
        self.ledger.checkpoint("plan", "cycle-1", 0, "planned", "Plan", {"plan": digest})
        path = self.ledger.objects / digest
        path.chmod(0o644)
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(LedgerError, "corruption"):
            self.ledger.export("cycle-1")
        with self.assertRaisesRegex(LedgerError, "corruption"):
            self.ledger.checkpoint("eval", "cycle-1", 1, "evaluating", "Run", {"plan": digest})
        with self.assertRaisesRegex(LedgerError, "corruption"):
            self.ledger.put_artifact("artifact-plan", self.root / "plan")
        self.assertEqual(self.ledger.status("cycle-1")["revision"], 1)

    def test_failure_and_negative_outcome_retained_without_fake_progress(self):
        plan = self.artifact()
        rejection = self.artifact("review", b"material regression; reject challenger")
        with self.assertRaisesRegex(LedgerError, "failure cannot advance"):
            self.ledger.checkpoint("bad-failure", "cycle-1", 0, "evaluating", "Run failed", outcome="failure")
        self.ledger.checkpoint("failure", "cycle-1", 0, "research", "Input unavailable", outcome="failure")
        self.ledger.checkpoint("plan", "cycle-1", 1, "planned", "Revised plan", {"plan": plan})
        self.ledger.checkpoint(
            "decision", "cycle-1", 2, "decided", "Keep incumbent", {"review": rejection}, recommendation="reject"
        )
        self.ledger.create_cycle("next", "cycle-2", **self.binding)
        events = self.ledger.export("cycle-1")["checkpoints"]
        self.assertEqual(events[1]["outcome"], "failure")
        self.assertEqual(events[1]["status"], "research")
        self.assertEqual(events[-1]["recommendation"], "reject")
        self.assertEqual(events[2]["references"]["plan"], plan)
        self.assertEqual(self.ledger._artifact(rejection), b"material regression; reject challenger")
        for table in ("cycles", "artifacts", "checkpoints", "commands"):
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                self.ledger.db.execute("DELETE FROM " + table)
        self.assertFalse(hasattr(self.ledger, "delete"))
        self.assertNotIn("active", self.ledger.export("cycle-1"))

    def test_invalid_binding_references_and_nonfinite_configuration(self):
        with self.assertRaises(LedgerError):
            self.ledger.create_cycle("bad", "bad", {}, {}, "layer", "hypothesis")
        with self.assertRaises(ValueError):
            self.ledger.create_cycle("nan", "nan", {"x": float("nan")}, {"x": 1}, "layer", "hypothesis")
        with self.assertRaises(LedgerError):
            self.ledger.checkpoint("missing", "cycle-1", 0, "planned", "Missing", {"plan": "0" * 64})
        with self.assertRaises(LedgerError):
            self.ledger.checkpoint("promotion", "cycle-1", 0, "decided", "Activate", recommendation="activate")
        self.assertEqual(self.ledger.status("cycle-1")["revision"], 0)

    def test_promotion_is_only_timestamped_history_and_retry_preserves_time(self):
        self.ledger.checkpoint("win", "cycle-1", 0, "decided", "Recommend challenger", recommendation="promote")
        before = self.ledger.export("cycle-1")
        self.ledger.checkpoint("win", "cycle-1", 0, "decided", "Recommend challenger", recommendation="promote")
        self.assertEqual(before, self.ledger.export("cycle-1"))
        for event in before["checkpoints"]:
            self.assertEqual(datetime.fromisoformat(event["recorded_at"]).utcoffset().total_seconds(), 0)
        self.assertEqual(before["checkpoints"][-1]["recommendation"], "promote")
        self.assertNotIn("active", before)
        self.assertNotIn(
            "promote", {row[0] for row in self.ledger.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        )

    def test_failed_object_flush_cleans_temporary_file_without_registration(self):
        source = self.root / "failed-flush"
        source.write_bytes(b"flush failure")
        with patch("ledger.os.fsync", side_effect=[OSError("disk failure"), None]):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.ledger.put_artifact("failed-flush", source)
        self.assertEqual(list(self.ledger.objects.iterdir()), [])
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 0)
        self.assertIsNone(self.ledger.db.execute("SELECT id FROM commands WHERE id='failed-flush'").fetchone())
        self.ledger.put_artifact("failed-flush", source)

    def test_replacement_rejected_on_independent_raw_connection(self):
        self.artifact()
        before = self.ledger.export("cycle-1")
        raw = sqlite3.connect(self.root / "journal" / "ledger.sqlite3")
        try:
            self.assertEqual(raw.execute("PRAGMA recursive_triggers").fetchone()[0], 0)
            for table in ("cycles", "artifacts", "checkpoints", "commands"):
                row = raw.execute("SELECT * FROM " + table + " LIMIT 1").fetchone()
                placeholders = ",".join("?" for _ in row)
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only duplicate"):
                    raw.execute(f"INSERT OR REPLACE INTO {table} VALUES({placeholders})", row)
                raw.rollback()
        finally:
            raw.close()
        self.assertEqual(before, self.ledger.export("cycle-1"))

    def test_existing_artifact_flush_required_before_registration_and_retry(self):
        digest = self.artifact()
        source = self.root / "plan"
        with patch("ledger.os.fsync", side_effect=OSError("directory failure")) as sync:
            with self.assertRaisesRegex(OSError, "directory failure"):
                self.ledger.put_artifact("second-artifact", source)
            self.assertEqual(sync.call_count, 1)
        self.assertIsNone(self.ledger.db.execute("SELECT id FROM commands WHERE id='second-artifact'").fetchone())
        with patch("ledger.os.fsync") as sync:
            self.assertEqual(self.ledger.put_artifact("second-artifact", source)["sha256"], digest)
            self.assertEqual(sync.call_count, 1)
        with patch("ledger.os.fsync") as sync:
            self.ledger.put_artifact("second-artifact", source)
            self.assertEqual(sync.call_count, 1)

    def test_thousand_cycles_retained_after_reopen(self):
        digest = self.artifact()
        for number in range(1000):
            cycle = f"retention-{number}"
            self.ledger.create_cycle(f"create-{cycle}", cycle, **self.binding)
            self.ledger.checkpoint(
                f"reject-{cycle}",
                cycle,
                0,
                "decided",
                "Retain negative outcome",
                {"review": digest},
                recommendation="reject",
            )
        self.ledger.close()
        self.ledger = Ledger(self.root / "journal")
        self.assertEqual(self.ledger.db.execute("SELECT COUNT(*) FROM cycles").fetchone()[0], 1001)
        for number in range(1000):
            exported = self.ledger.export(f"retention-{number}")
            self.assertEqual(len(exported["checkpoints"]), 2)
            self.assertEqual(exported["checkpoints"][-1]["recommendation"], "reject")
            self.assertEqual(exported["checkpoints"][-1]["references"]["review"], digest)

    def test_cli_status_and_export(self):
        script = Path(__file__).with_name("ledger.py")
        for operation in ("status", "export"):
            result = subprocess.run(
                [sys.executable, "-B", str(script), "--root", str(self.root / "journal"), operation, "cycle-1"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(result.stdout)["cycle"], "cycle-1")


if __name__ == "__main__":
    unittest.main()
