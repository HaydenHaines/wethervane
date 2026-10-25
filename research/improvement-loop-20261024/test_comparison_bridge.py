"""Metadata-backed bridge tests; journal writes use temporary software fixtures only."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import comparison_bridge as bridge
import execution
from ledger import Ledger, canonical

HERE = Path(__file__).resolve().parent
ACTUAL_REQUEST = HERE / "comparison-bridge-sol-v1/actual-request-v1.json"
ACTUAL_JOURNAL = HERE.parents[1] / ".improvement-ledger"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        if not ACTUAL_REQUEST.is_file():
            self.skipTest("pinned actual M02 metadata request unavailable")
        self.temporary = tempfile.TemporaryDirectory(prefix=".comparison-test-", dir=HERE)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.request = json.loads(ACTUAL_REQUEST.read_text())
        self.counter = 0

    def write(self, path, value):
        execution.publish(path, value)
        return execution.pin(path)

    def pin_request(self):
        self.counter += 1
        return self.write(self.base / ("request-" + str(self.counter) + ".json"), self.request)

    def validated(self):
        request = self.pin_request()
        return request, bridge.validate(request, self.base / ("validation-" + str(self.counter)))

    def fixture_ledger(self, phase="evaluating", binding=None):
        root = self.base / "journal"
        ledger = Ledger(root)
        supplied = binding or self.request["cycle_binding"]
        ledger.create_cycle("create", self.request["cycle_id"], **supplied)
        ledger.checkpoint("start", self.request["cycle_id"], 0, phase, "Invented temporary journal fixture")
        ledger.close()
        return root

    def reconcile(self, receipt, request, root, revision=1, command="retain", name="reconciliation"):
        return bridge.reconcile(receipt, request, root, self.request["cycle_id"], revision, command, self.base / name)

    def dossier(self, scores_change=None, decision_change=None, review_change=None, acceptance_change=None):
        """Rebind compact top-level fixtures without opening predictions or labels."""
        docs = {role: json.loads(Path(pin["path"]).read_text()) for role, pin in self.request["sources"].items()}
        if scores_change:
            scores_change(docs["scores"])
        if decision_change:
            decision_change(docs["root_decision"])
        if review_change:
            review_change(docs["independent_review"])
        if acceptance_change:
            acceptance_change(docs["root_acceptance"])
        pins = self.request["sources"]
        pins["scores"] = self.write(self.base / "scores.json", docs["scores"])
        docs["scoring_receipt"]["artifacts"]["scores.json"] = pins["scores"]
        pins["scoring_receipt"] = self.write(self.base / "receipt.json", docs["scoring_receipt"])
        docs["scoring_terminal"]["receipt"] = docs["scoring_receipt"]
        pins["scoring_terminal"] = self.write(self.base / "terminal.json", docs["scoring_terminal"])
        review = docs["independent_review"]
        review["actual_terminal"], review["actual_output_receipt"] = pins["scoring_terminal"], pins["scoring_receipt"]
        review["actual_artifacts"]["scores.json"] = pins["scores"]
        pins["independent_review"] = self.write(self.base / "review.json", review)
        acceptance = docs["root_acceptance"]
        acceptance["actual_terminal"], acceptance["actual_receipt"] = pins["scoring_terminal"], pins["scoring_receipt"]
        acceptance["independent_review"] = pins["independent_review"]
        pins["root_acceptance"] = self.write(self.base / "acceptance.json", acceptance)
        docs["root_decision"]["acceptance"] = pins["root_acceptance"]
        pins["root_decision"] = self.write(self.base / "decision.json", docs["root_decision"])

    def test_actual_metadata_only_preserves_all_precision_and_display_limits(self):
        original = Path.read_bytes
        def metadata_only(path):
            if path.suffix.lower() in (".npz", ".npy", ".csv", ".nc", ".netcdf"):
                raise AssertionError("raw tensors/outcomes must not be opened")
            return original(path)
        with patch.object(Path, "read_bytes", metadata_only):
            report, _ = bridge.inspect(self.pin_request())
        self.assertEqual(len(report["metric_cells"]), 360)
        self.assertEqual(report["prior_independent_review"]["positive_point_regressions"], 19)
        self.assertEqual(report["prior_independent_review"]["numerically_resolved_positive_regressions"], 18)
        display = [r for r in report["forecast_prior_numerical_statuses"] if not r["interval_display_qualified"]]
        self.assertEqual([(r["cycle"], r["arm"]) for r in display], [(2020, "LP5")])
        self.assertFalse(report["raw_bank_revalidated"])
        self.assertTrue(any(c["value"] is None for c in report["metric_cells"]))
        self.assertTrue(any(c["status"] == "source_chain_dispersion_only" for c in report["metric_cells"]))
        primary = next(c for c in report["metric_cells"] if c.get("tier") == "A" and c["mask"] == "common" and c["arm"] == "FS5" and c["metric"] == "crps")
        self.assertGreater(primary["native_cell"]["mcse"], 0)
        self.assertEqual(primary["native_cell"]["qualified"], primary["conditional_numerical_precision"])

    def test_actual_decided_noop_is_read_only_and_exact(self):
        request, receipt = self.validated()
        before = bridge.journal_snapshot(ACTUAL_JOURNAL, self.request["cycle_id"])
        database_hash = execution.pin(ACTUAL_JOURNAL / "ledger.sqlite3")
        with patch.object(bridge, "Ledger", side_effect=AssertionError("actual journal writer forbidden")):
            result = self.reconcile(receipt, request, ACTUAL_JOURNAL, revision=7)
        self.assertEqual(json.loads(Path(result["path"]).read_text())["status"], "already_recorded")
        self.assertEqual(before, bridge.journal_snapshot(ACTUAL_JOURNAL, self.request["cycle_id"]))
        self.assertEqual(database_hash, execution.pin(ACTUAL_JOURNAL / "ledger.sqlite3"))

    def test_temporary_evaluating_cycle_records_once_then_read_only_retry(self):
        request, receipt = self.validated()
        root = self.fixture_ledger()
        result = self.reconcile(receipt, request, root)
        self.assertEqual(json.loads(Path(result["path"]).read_text())["status"], "recorded")
        current = bridge.journal_snapshot(root, self.request["cycle_id"], "retain")
        self.assertEqual((current["history"][-1]["revision"], current["history"][-1]["status"]), (2, "decided"))
        database = execution.pin(root / "ledger.sqlite3")
        with patch.object(bridge, "Ledger", side_effect=AssertionError("idempotent retry is read only")):
            retry = self.reconcile(receipt, request, root, name="retry")
        self.assertEqual(json.loads(Path(retry["path"]).read_text())["status"], "recorded_idempotent")
        self.assertEqual(database, execution.pin(root / "ledger.sqlite3"))
        with self.assertRaisesRegex(ValueError, "command ID conflicts"):
            self.reconcile(receipt, request, root, revision=2, name="conflicting-command")
        self.assertEqual(database, execution.pin(root / "ledger.sqlite3"))

    def test_commit_before_output_crash_has_stable_command_retry(self):
        request, receipt = self.validated()
        root = self.fixture_ledger("reviewing")
        original = execution.publish
        def fail_receipt(path, value):
            if Path(path) == self.base / "reconciliation/receipt.json":
                raise OSError("invented output failure after journal commit")
            return original(path, value)
        with patch.object(execution, "publish", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "after journal commit"):
                self.reconcile(receipt, request, root)
        snapshot = bridge.journal_snapshot(root, self.request["cycle_id"], "retain")
        self.assertEqual(snapshot["history"][-1]["revision"], 2)
        self.assertTrue((self.base / "reconciliation").is_dir())
        result = self.reconcile(receipt, request, root, name="after-crash")
        self.assertEqual(json.loads(Path(result["path"]).read_text())["status"], "recorded_idempotent")
        self.assertEqual(snapshot, bridge.journal_snapshot(root, self.request["cycle_id"], "retain"))

    def test_normalized_receipt_and_independent_request_anchor_cannot_be_forged(self):
        request, receipt = self.validated()
        root = self.fixture_ledger()
        altered = json.loads(Path(receipt["path"]).read_text())
        altered["recommendation"] = "promote"
        forged = self.write(self.base / "forged.json", altered)
        with self.assertRaisesRegex(ValueError, "differs from reconstructed"):
            self.reconcile(forged, request, root)
        wrong_request = self.write(self.base / "wrong-request.json", dict(self.request, cycle_id="foreign-cycle"))
        with self.assertRaisesRegex(ValueError, "independently expected request"):
            self.reconcile(receipt, wrong_request, root)
        self.assertEqual(bridge.journal_snapshot(root, self.request["cycle_id"])["history"][-1]["revision"], 1)

    def test_stale_cas_binding_conflict_missing_db_and_existing_output_do_not_write(self):
        request, receipt = self.validated()
        root = self.fixture_ledger()
        before = execution.pin(root / "ledger.sqlite3")
        with self.assertRaisesRegex(ValueError, "stale revision"):
            self.reconcile(receipt, request, root, revision=0)
        self.assertEqual(before, execution.pin(root / "ledger.sqlite3"))
        (self.base / "existing").mkdir()
        with self.assertRaisesRegex(ValueError, "output must be fresh"):
            self.reconcile(receipt, request, root, name="existing")
        self.assertEqual(before, execution.pin(root / "ledger.sqlite3"))
        missing = self.base / "missing"
        with self.assertRaisesRegex(ValueError, "database does not exist"):
            self.reconcile(receipt, request, missing, name="missing-output")
        self.assertFalse(missing.exists())
        empty = self.base / "empty-journal"
        Ledger(empty).close()
        with self.assertRaisesRegex(ValueError, "cycle does not exist"):
            self.reconcile(receipt, request, empty, name="empty-journal-output")
        self.request["cycle_binding"]["incumbent_config"]["law"] = "different full configuration"
        other_request, other_receipt = self.validated()
        with self.assertRaisesRegex(ValueError, "full cycle binding mismatch"):
            self.reconcile(other_receipt, other_request, root, name="wrong-binding")

    def test_idempotent_retry_rejects_corrupted_registered_evidence_objects(self):
        request, receipt = self.validated()
        root = self.fixture_ledger()
        self.reconcile(receipt, request, root)
        digest = self.request["sources"]["root_decision"]["sha256"]
        path = root / "objects" / digest
        path.chmod(0o644)
        path.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.reconcile(receipt, request, root, name="corrupt-object-retry")

    def test_unsupported_decision_schema_and_scope_are_rejected(self):
        for field, value in (("decision", "promote"), ("decision", "defer"), ("schema", "unknown")):
            with self.subTest(field=field, value=value):
                saved = copy.deepcopy(self.request)
                directory = self.base / (field + "-" + value)
                directory.mkdir()
                original_base = self.base
                self.base = directory
                self.dossier(decision_change=lambda d: d.update({field:value}))
                with self.assertRaisesRegex(ValueError, "unsupported decision"):
                    bridge.inspect(self.pin_request())
                self.request, self.base = saved, original_base
        self.request["cycle_binding"]["scope"]["horizon_days"] = 7
        with self.assertRaisesRegex(ValueError, "scientific scope"):
            bridge.inspect(self.pin_request())

    def test_score_arm_status_and_primary_summary_cross_binding_rejected(self):
        self.dossier(scores_change=lambda s: s["tiers"]["A"]["competitive"]["arms"].pop("PO5"))
        with self.assertRaisesRegex(ValueError, "score arm roster"):
            bridge.inspect(self.pin_request())

    def test_cell_precision_status_cannot_claim_scientific_qualification(self):
        def corrupt(scores):
            scores["cycles"]["2018"]["masks"]["competitive"]["arms"]["FS5"]["aggregates"]["crps"]["status"] = "scientifically_qualified"
        self.dossier(scores_change=corrupt)
        with self.assertRaisesRegex(ValueError, "numerical cell status"):
            bridge.inspect(self.pin_request())

    def test_prior_independent_review_cannot_point_to_foreign_score_receipt(self):
        self.dossier(review_change=lambda d: d.update(disposition="unreviewed"))
        with self.assertRaisesRegex(ValueError, "independent comparison review"):
            bridge.inspect(self.pin_request())

    def test_native_metadata_fingerprint_corruption_is_detected(self):
        item = self.request["sources"]["root_acceptance"]
        copied = self.base / "acceptance.json"
        copied.write_bytes(Path(item["path"]).read_bytes())
        self.request["sources"]["root_acceptance"] = execution.pin(copied)
        copied.write_text("{}")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            bridge.inspect(self.pin_request())

    def test_already_decided_failure_outcome_and_extra_reference_are_rejected(self):
        request, receipt = self.validated()
        for kind in ("failure", "extra-reference"):
            root = self.base / kind
            binding = self.request["cycle_binding"]
            ledger = Ledger(root)
            ledger.create_cycle("create", self.request["cycle_id"], **binding)
            refs = {}
            for field, role in (("evaluation", "root_decision"), ("review", "independent_review")):
                refs[field] = ledger.put_artifact("copy-" + field, self.request["sources"][role]["path"])["sha256"]
            if kind == "extra-reference":
                refs["plan"] = refs["evaluation"]
            ledger.checkpoint("decide", self.request["cycle_id"], 0, "decided", "Temporary decided fixture", refs, recommendation="retain")
            revision = 1
            if kind == "failure":
                ledger.checkpoint("failed", self.request["cycle_id"], 1, "decided", "Temporary failed decision fixture", refs, outcome="failure", recommendation="retain")
                revision = 2
            ledger.close()
            before = execution.pin(root / "ledger.sqlite3")
            with self.assertRaisesRegex(ValueError, "decided journal differs"):
                self.reconcile(receipt, request, root, revision=revision, name=kind + "-output")
            self.assertEqual(before, execution.pin(root / "ledger.sqlite3"))

    def test_empty_cycle_cell_cannot_be_replaced_with_numeric_zero(self):
        def corrupt(scores):
            scores["cycles"]["2022"]["masks"]["near_even"]["arms"]["FS5"]["aggregates"]["brier"]["value"] = 0
        self.dossier(scores_change=corrupt)
        with self.assertRaisesRegex(ValueError, "empty score cell"):
            bridge.inspect(self.pin_request())

    def test_missing_cycle_value_must_propagate_to_unavailable_tier(self):
        def corrupt(scores):
            scores["tiers"]["A"]["near_even"]["arms"]["FS5"]["crps"]["value"] = 0
        self.dossier(scores_change=corrupt)
        with self.assertRaisesRegex(ValueError, "unavailable cycle must propagate"):
            bridge.inspect(self.pin_request())

    def test_review_auxiliary_artifact_cannot_be_cross_bound(self):
        self.dossier(review_change=lambda d: d["actual_artifacts"]["launch.json"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "review artifact cross-binding"):
            bridge.inspect(self.pin_request())


if __name__ == "__main__":
    unittest.main()
