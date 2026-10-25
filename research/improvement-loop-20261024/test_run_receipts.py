"""Read-only real native receipt checks plus bounded corruption fixtures."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ledger import Ledger, canonical
from run_receipts import SCOPE_KEYS, Bundle, ReceiptError, attach_evidence, decode, inspect_house, inspect_receipts

SENATE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
HOUSE = Path("/home/hayden/projects/ballotmancy-house-benchmark")


def pin(root, name):
    return dict(path=name, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())


class RealCheckoutReceiptTests(unittest.TestCase):
    """Optional owner-checkout integrations; missing checkouts are explicitly skipped."""

    def senate_inputs(self):
        name = "research/senate-stack-integration-20261020/checkpoint-root-v16.json"
        if not (SENATE / name).exists():
            self.skipTest("real Senate checkout absent")
        checkpoint = json.loads((SENATE / name).read_text())
        attempt = checkpoint["terminal_attempt"]
        failure = json.loads((SENATE / attempt["failure"]["path"]).read_text())
        plan = json.loads((SENATE / failure["plan"]["path"]).read_text())
        return pin(SENATE, name), failure["plan"]["sha256"], {key: plan[key] for key in SCOPE_KEYS}, attempt

    def test_real_failed_attempt_and_optional_checkpoint(self):
        checkpoint, plan, scope, attempt = self.senate_inputs()
        report, _ = inspect_receipts(SENATE, checkpoint, plan, scope)
        native, _ = inspect_receipts(SENATE, None, plan, scope, attempt)
        self.assertEqual(report["dimensions"], native["dimensions"])
        self.assertEqual(report["dimensions"]["numerical_fit"], "failed")
        self.assertFalse(report["qualification"])
        self.assertFalse(report["owner_observation"]["is_liveness"])
        self.assertIsNone(native["owner_observation"])
        with self.assertRaisesRegex(ReceiptError, "scope mismatch"):
            inspect_receipts(SENATE, None, plan, dict(scope, purpose="historical"), attempt)
        with self.assertRaisesRegex(ReceiptError, "plan binding"):
            inspect_receipts(SENATE, None, "0" * 64, scope, attempt)

    def house_inputs(self):
        base = "research/house-benchmark-20261020/"
        name = base + "frontend-results-v2/review-receipt.json"
        if not (HOUSE / name).exists():
            self.skipTest("real House checkout absent")
        review = json.loads((HOUSE / name).read_text())
        cell = next(
            cell
            for cell in review["cells"]
            if cell["cycle"] == 2024 and cell["configuration"] == "house_history_refreshed_v1"
        )
        fit = json.loads((HOUSE / cell["fit_receipt"]["path"]).read_text())
        return pin(HOUSE, name), json.loads((HOUSE / fit["plan"]["path"]).read_text())

    def test_actual_reviewed_house_success_keeps_failed_subtotal(self):
        review, plan = self.house_inputs()
        drift = [entry["path"] for entry in plan["source_pins"] if pin(HOUSE, entry["path"]) != entry]
        if drift:
            with self.assertRaisesRegex(ReceiptError, "hash mismatch"):
                inspect_house(HOUSE, review, 2024, "house_history_refreshed_v1", plan)
            self.skipTest("real House historical positive unavailable: frozen source pins drifted: " + ", ".join(drift))
        report, bundle = inspect_house(HOUSE, review, 2024, "house_history_refreshed_v1", plan)
        self.assertEqual(report["dimensions"]["prediction_precision"], "race_pass_subtotal_fail")
        self.assertEqual(report["scope"]["target_id"], "house_aggregate_fundamentals_retrospective_v1")
        self.assertFalse(report["qualification"])
        self.assertIsNone(report["recommendation"])
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(directory)
            try:
                ledger.create_cycle(
                    "new", "test", {"law": "incumbent"}, {"cycles": [2024]}, "fundamentals", "test only"
                )
                before = ledger.export("test")
                evidence = attach_evidence(ledger, "receipt", report, bundle)
                self.assertEqual(before, ledger.export("test"))
                self.assertEqual(json.loads(ledger._artifact(evidence["sha256"]))["dimensions"], report["dimensions"])
                self.assertEqual(evidence, attach_evidence(ledger, "receipt", report, bundle))
            finally:
                ledger.close()
        with self.assertRaisesRegex(ReceiptError, "untrusted"):
            inspect_house(HOUSE, dict(review, sha256="0" * 64), 2024, "house_history_refreshed_v1", plan)
        with self.assertRaisesRegex(ReceiptError, "declared plan mismatch"):
            inspect_house(HOUSE, review, 2024, "house_history_refreshed_v1", dict(plan, target_cycle=2022))

    def test_native_pending_positive_and_parent_mismatch_fail_closed(self):
        _, plan, scope, attempt = self.senate_inputs()
        original = Bundle.document
        for change in ("pending", "positive", "parent"):

            def altered(bundle, pin):
                value = original(bundle, pin)
                if pin == attempt["fit_failure"]:
                    if change == "pending":
                        value["status"] = "pending"
                    elif change == "positive":
                        value["complete_cut"] = True
                    else:
                        value["required_parent_ids"] = ["different"]
                return value

            with patch.object(Bundle, "document", altered):
                with self.assertRaises(ReceiptError):
                    inspect_receipts(SENATE, None, plan, scope, attempt)

    def test_review_cannot_override_failed_race_precision(self):
        review, plan = self.house_inputs()
        original = Bundle.document

        def altered(bundle, pin):
            value = original(bundle, pin)
            if value.get("schema") == "house-fit-receipt-v1":
                value["race_prediction_precision_passed"] = False
            return value

        with patch.object(Bundle, "document", altered):
            with self.assertRaisesRegex(ReceiptError, "race fit not qualified"):
                inspect_house(HOUSE, review, 2024, "house_history_refreshed_v1", plan)

    def test_integrity_paths_duplicate_keys_and_unknown_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "receipt.json").write_text('{"status":"passed"}')
            bundle = Bundle(root)
            good = pin(root, "receipt.json")
            bundle.read(good)
            (root / "receipt.json").write_text('{"status":"pending"}')
            with self.assertRaisesRegex(ReceiptError, "hash mismatch"):
                bundle.read(good)
            with self.assertRaisesRegex(ReceiptError, "outside"):
                bundle.read(dict(path="../outside", sha256="0" * 64))
            with self.assertRaisesRegex(ReceiptError, "unsupported checkpoint"):
                inspect_receipts(root, pin(root, "receipt.json"), "0" * 64, {})
        with self.assertRaisesRegex(ReceiptError, "duplicate"):
            decode('{"status":"failed","status":"passed"}')
        with self.assertRaisesRegex(ReceiptError, "nonfinite"):
            decode('{"x":NaN}')


class PortableReceiptTests(unittest.TestCase):
    """Synthetic schema fixtures test adapter integrity, never scientific qualification."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(value))
        return pin(self.root, name)

    def senate(self, attempt="attempt1"):
        scope = dict.fromkeys(SCOPE_KEYS, "synthetic integrity fixture")
        scope["parent_ids"] = ["fixture:0"]
        policy = {"mode": "diagnostic_first_parent"}
        plan = self.write(
            "plan.json",
            dict(
                scope,
                schema="actual_TX_role_execution_plan_v2",
                forecast_qualified=False,
                publication_authorized=False,
                execution_adapter_version="share_coordinate_bounded_execution_v1",
                execution_policy=policy,
                provider={},
                source_inputs={},
                source_pins={},
                runtime={},
            ),
        )
        heartbeat = self.write(f"{attempt}/fit/parent-0-observer-heartbeat.jsonl", {"fixture": True})
        observer = self.write(
            f"{attempt}/fit/parent-0-observer-receipt.json",
            dict(
                schema="sampling_observer_receipt_v1",
                version="returned_draw_and_observer_heartbeat_v1",
                returned_draw_records=0,
                sampling_returned_to_coordinator=False,
                parent_id="fixture:0",
                observer_liveness_is_not_sampler_progress=True,
                sampling_error="fixture error",
                files={"parent-0-observer-heartbeat.jsonl": heartbeat["sha256"]},
            ),
        )
        execution = self.write(
            f"{attempt}/fit/parent-0-execution.json",
            dict(schema="conditional_process_observation_v1", error="fixture error"),
        )
        fit = self.write(
            f"{attempt}/fit/failure.json",
            dict(
                status="incomplete_preserved_attempt",
                complete_cut=False,
                completed_parents=[],
                required_parent_ids=scope["parent_ids"],
                no_automatic_retry=True,
                error="fixture error",
                execution_policy=dict(version="share_coordinate_bounded_first_parent_execution_v1", policy=policy),
                observer_inventory=dict(
                    complete=True,
                    errors=[],
                    files={
                        "parent-0-observer-receipt.json": observer["sha256"],
                        "parent-0-observer-heartbeat.jsonl": heartbeat["sha256"],
                    },
                ),
                progress_files={},
            ),
        )
        failure = self.write(
            f"{attempt}/failure.json",
            dict(
                status="incomplete_preserved_attempt",
                forecast_qualified=False,
                publication_authorized=False,
                no_automatic_retry=True,
                plan=plan,
                error="fixture error",
            ),
        )
        terminal = dict(
            path=attempt,
            failure=failure,
            fit_failure=fit,
            execution=execution,
            observer=observer,
            log=self.write(attempt + ".log", "fixture failed"),
            error="fixture error",
            returned_draw_records=0,
            complete_cut=False,
            qualified_counts=False,
            full_fit_followup_allowed=False,
        )
        checkpoint = self.write(
            "checkpoint.json",
            dict(
                schema="SenateStackCheckpointV16",
                status="partial_affine_worker_startup_failed_repair_underway",
                issued_utc="fixture",
                terminal_attempt=terminal,
            ),
        )
        return checkpoint, plan["sha256"], scope, terminal

    def house(self, source_count=12):
        sources = [self.write(f"sources/{index}.py", "synthetic fixture") for index in range(source_count)]
        panel = self.write("panel.json", {"synthetic": True})
        plan = dict(
            schema="house-fit-plan-v1",
            target_id="house_aggregate_fundamentals_retrospective_v1",
            target_cycle=2024,
            house_configuration_id="fixture",
            source_pins=sources,
            panel=panel,
            bank_reused_from=self.write("bank-receipt.json", {"fixture": "opaque ancestor"}),
        )
        plan_pin = self.write("house-plan.json", plan)
        fit = self.write(
            "house-fit.json",
            dict(
                schema="house-fit-receipt-v1",
                plan=plan_pin,
                diagnostics=dict(passed=True),
                race_prediction_precision_passed=True,
                subset_seat_precision_passed=False,
                prediction_diagnostics=dict(checks=dict(subtotal_mcse=False)),
            ),
        )
        result = self.write(
            "results/fixture-2024-h14.json",
            dict(
                schemaVersion=1,
                reviewStatus="reviewed",
                run=dict(
                    id="fixture-2024-h14", configurationSha256=plan_pin["sha256"], historical=True, experimental=True
                ),
                metadata=dict(full_chamber=False),
                limitations=["Synthetic test fixture; no scientific evidence."],
            ),
        )
        review = self.write(
            "review.json",
            dict(
                schema="house-actual-output-review-v1",
                cells=[
                    dict(
                        cycle=2024,
                        configuration="fixture",
                        numerical_status="passed",
                        structural_and_score_audit_passed=True,
                        subset_seat_precision_passed=False,
                        fit_receipt=fit,
                    )
                ],
                outputs=[result],
                panel=panel,
                scores=self.write("scores.json", {}),
                review_source=self.write("review-source.py", "fixture"),
            ),
        )
        return review, plan

    def inspect_fixture_house(self, review, plan):
        # Trust replacement is limited to this synthetic fixture; production retains
        # the single canonical independent review SHA and cannot accept this review.
        with patch("run_receipts.HOUSE_REVIEW_SHA", review["sha256"]):
            return inspect_house(self.root, review, 2024, "fixture", plan)

    def test_frozen_report_rejects_mutation_and_wrong_inspection(self):
        checkpoint, digest, scope, _ = self.senate()
        report, inspected = inspect_receipts(self.root, checkpoint, digest, scope)
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(directory)
            self.addCleanup(ledger.close)
            saved = attach_evidence(ledger, "valid", report, inspected)
            self.assertFalse(json.loads(ledger._artifact(saved["sha256"]))["qualification"])
            for mutate in (
                lambda value: value.update(qualification=True),
                lambda value: value["dimensions"].update(numerical_fit="passed"),
                lambda value: value.update(verified_artifacts={}),
                lambda value: value["scope"].update(purpose="different"),
            ):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaisesRegex(ReceiptError, "report changed"):
                    attach_evidence(ledger, "tampered", changed, inspected)
            other_checkpoint, other_digest, other_scope, _ = self.senate("attempt2")
            _, other = inspect_receipts(self.root, other_checkpoint, other_digest, other_scope)
            with self.assertRaisesRegex(ReceiptError, "report changed"):
                attach_evidence(ledger, "wrong", report, other)

    def test_attachment_rejects_changed_source_before_writing(self):
        checkpoint, digest, scope, _ = self.senate()
        report, inspected = inspect_receipts(self.root, checkpoint, digest, scope)
        (self.root / "plan.json").write_text("changed")
        ledger = unittest.mock.Mock()
        with self.assertRaisesRegex(ReceiptError, "hash mismatch"):
            attach_evidence(ledger, "changed", report, inspected)
        ledger.put_artifact.assert_not_called()

    def test_each_cross_attempt_pin_is_rejected_despite_equal_errors(self):
        _, digest, scope, first = self.senate()
        _, _, _, second = self.senate("attempt2")
        for key in ("fit_failure", "execution", "observer", "log", "failure"):
            mixed = dict(first, **{key: second[key]})
            with self.subTest(key=key), self.assertRaisesRegex(ReceiptError, "attempt path mismatch"):
                inspect_receipts(self.root, None, digest, scope, mixed)

    def test_checkpoint_terminal_path_and_error_bind_native_attempt(self):
        _, digest, scope, terminal = self.senate()
        for key, value in (("path", "attempt2"), ("error", "different")):
            changed = dict(terminal, **{key: value})
            checkpoint = self.write(
                "checkpoint.json",
                dict(
                    schema="SenateStackCheckpointV16",
                    status="partial_affine_worker_startup_failed_repair_underway",
                    issued_utc="fixture",
                    terminal_attempt=changed,
                ),
            )
            with self.subTest(key=key), self.assertRaisesRegex(ReceiptError, f"terminal attempt {key} mismatch"):
                inspect_receipts(self.root, checkpoint, digest, scope)

    def test_fit_inventory_binds_observer_and_progress_bytes(self):
        _, digest, scope, terminal = self.senate()
        original = json.loads((self.root / terminal["fit_failure"]["path"]).read_text())
        for change in ("observer", "progress"):
            fit = copy.deepcopy(original)
            if change == "observer":
                fit["observer_inventory"]["files"]["parent-0-observer-receipt.json"] = "0" * 64
            else:
                progress = self.write("attempt1/fit/parent-0-progress.jsonl", "fixture")
                fit["progress_files"] = {"parent-0-progress.jsonl": "0" * 64}
                self.assertNotEqual(progress["sha256"], "0" * 64)
            changed = dict(terminal, fit_failure=self.write("attempt1/fit/failure.json", fit))
            with self.subTest(change=change), self.assertRaises(ReceiptError):
                inspect_receipts(self.root, None, digest, scope, changed)

    def test_same_path_and_alias_cannot_replace_first_pin(self):
        first = self.write("receipt.json", {"first": True})
        (self.root / "alias.json").symlink_to("receipt.json")
        bundle = Bundle(self.root)
        bundle.read(first)
        bundle.read(dict(first, path="./alias.json"))
        self.assertEqual(list(bundle.artifacts), ["receipt.json"])
        second = self.write("receipt.json", {"second": True})
        for name in ("receipt.json", "./receipt.json", "alias.json"):
            with self.subTest(name=name), self.assertRaisesRegex(ReceiptError, "conflicting pin"):
                bundle.read(dict(second, path=name))
        self.assertEqual(bundle.artifacts["receipt.json"]["sha256"], first["sha256"])
        exposed = bundle.artifacts
        exposed["receipt.json"]["sha256"] = second["sha256"]
        self.assertEqual(bundle.artifacts["receipt.json"]["sha256"], first["sha256"])

    def test_house_source_and_plan_pointers_are_actually_verified(self):
        review, plan = self.house()
        report, inspected = self.inspect_fixture_house(review, plan)
        self.assertEqual(report["dimensions"]["prediction_precision"], "race_pass_subtotal_fail")
        self.assertFalse(report["qualification"])
        for entry in plan["source_pins"] + [plan["panel"], plan["bank_reused_from"]]:
            path = self.root / entry["path"]
            original = path.read_bytes()
            path.write_text("changed")
            with self.subTest(path=entry["path"]), self.assertRaisesRegex(ReceiptError, "hash mismatch"):
                self.inspect_fixture_house(review, plan)
            path.write_bytes(original)
            self.assertIn(entry["path"], inspected.artifacts)

    def test_house_uses_loaded_source_inventory_and_reports_verification_scope(self):
        review, plan = self.house(source_count=8)
        report, _ = self.inspect_fixture_house(review, plan)
        self.assertEqual(len(report["scope"]["source_pins"]), 8)
        self.assertEqual(
            report["verification_scope"],
            "selected review/fit/result and direct loaded-plan pins; transitive ancestors not replayed",
        )

    def test_synthetic_house_review_is_not_trusted_by_production(self):
        review, plan = self.house()
        with self.assertRaisesRegex(ReceiptError, "untrusted House review"):
            inspect_house(self.root, review, 2024, "fixture", plan)


if __name__ == "__main__":
    unittest.main()
