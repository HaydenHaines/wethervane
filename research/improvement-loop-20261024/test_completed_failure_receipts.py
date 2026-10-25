"""Portable failure-contract checks and one bounded real-owner integration."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ledger import Ledger, canonical
from run_receipts import SCOPE_KEYS, ReceiptError, attach_evidence, inspect_completed_failure

BASE = Path(__file__).parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
NATIVE = "research/senate-stack-integration-20261020/partial-affine-completion-root-v1/"
TERMINAL_SHA = "32b7c1de3904fefed9be105cb11fe68f4fd1e97dc183c710fec623868cca1fb3"
PLAN_SHA = "adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1"


class CompletedFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = {}
        self.scope = dict.fromkeys(SCOPE_KEYS, "synthetic contract fixture")
        self.settings = dict(chains=4, draws=1000, tune=1500, max_rhat=1.01, min_ess=400)
        self.scope.update(
            parent_ids=["p0", "p1"],
            parent_masses=[0.25, 0.75],
            scientific_target_ids=["target0", "target1"],
            settings=self.settings,
        )
        policy = dict(mode="full_all_parents", sampling_wall_seconds=86400)
        version = "share_coordinate_completion_execution_v2"
        plan = dict(
            self.scope,
            provider={"coordinate_version": "fixture_coordinates"},
            schema="actual_TX_role_execution_plan_v2",
            execution_adapter_version=version,
            execution_policy=policy,
            complete_cut_expected=True,
            forecast_qualified=False,
            publication_authorized=False,
            automatic_retry=False,
        )
        self.plan_pin = self.write("plan.json", plan)
        admission = self.write("admission.json", {"fixture": True})
        archive_pin = self.write("attempt1/fit/parent-0-returned-inference.pkl", b"opaque fixture, never unpickled")
        archive = dict(
            complete=True,
            returned_object_available=True,
            numerical_qualification=False,
            error=None,
            format="python_pickle_complete_InferenceData_same_pinned_runtime_v1",
            groups=["posterior", "sample_stats"],
            files={"parent-0-returned-inference.pkl": archive_pin["sha256"]},
        )
        execution = self.write(
            "attempt1/fit/parent-0-execution.json", dict(schema="conditional_process_observation_v1", error=None)
        )
        budget = dict(
            version=version,
            policy=policy,
            sampler_returned=True,
            budget_exceeded=False,
            error=None,
            automatic_retry=False,
            publication_authorized=False,
            returned_transitions=10000,
        )
        diagnostic = dict(
            passed=False,
            finite_diagnostics=True,
            gates=self.settings,
            max_rhat=1.2,
            min_bulk_ess=10,
            min_tail_ess=20,
            divergent=0,
            depth_saturation=0,
        )
        result = dict(
            parent_id="p0",
            coordinate_version="fixture_coordinates",
            scientific_target_id="target0",
            conditional_bank_only=True,
            sampling_budget=budget,
            returned_inference_archive=archive,
            diagnostics=diagnostic,
            execution_observation=dict(file="parent-0-execution.json", sha256=execution["sha256"]),
        )
        failure = dict(
            status="incomplete_preserved_attempt",
            no_automatic_retry=True,
            error="numerical failure",
            forecast_qualified=False,
            publication_authorized=False,
            plan=self.plan_pin,
            admission=admission,
        )
        fit = dict(
            status="incomplete_preserved_attempt",
            no_automatic_retry=True,
            error="numerical failure",
            complete_cut=False,
            completed_parents=[result],
            required_parent_ids=["p0", "p1"],
            execution_policy=dict(version=version, policy=policy),
            complete_returned_inference_archive=archive,
        )
        for name, value in [
            ("failure.json", failure),
            ("fit/failure.json", fit),
            ("fit/parent-0-result.json", result),
            ("fit/parent-0-sampling-budget.json", budget),
            ("fit/parent-0-returned-inference.json", archive),
        ]:
            self.write("attempt1/" + name, value)
        self.write(
            "attempt1/fit/parent-0-diagnostics.csv",
            b",r_hat,ess_bulk,ess_tail\nquantity1,1.2,10,20\nquantity2,1.01,500,500\n",
        )
        self.terminal = dict(
            schema="FullActualFitTerminalRootV1",
            issued_utc="2026-10-25T00:00:00Z",
            classification="COMPLETED_FIRST_CONDITIONAL_SAMPLING_FAILED_NUMERICAL_CONVERGENCE",
            authoritative_exit_code=1,
            terminal_confirmed_by_tool_handle=True,
            complete_cut=False,
            no_automatic_retry=True,
            no_numerical_gates_relaxed=True,
            resource_budget_exceeded=False,
            completed_population_conditionals=1,
            numerically_qualified_population_conditionals=0,
            required_population_conditionals=2,
            chains=4,
            retained_per_chain=1000,
            warmup_per_chain=1500,
            diagnostics=diagnostic,
        )

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if isinstance(value, bytes) else canonical(value).encode())
        pin = dict(path=name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        self.files[name] = pin
        return pin

    def terminal_pin(self):
        self.terminal["preserved_files"] = [
            pin for name, pin in sorted(self.files.items()) if name.startswith("attempt1/")
        ]
        return self.write("terminal.json", self.terminal)

    def inspect(self):
        return inspect_completed_failure(self.root, self.terminal_pin(), self.plan_pin["sha256"], self.scope)

    def change(self, name, edit):
        value = json.loads((self.root / name).read_text())
        edit(value)
        self.write(name, value)

    def test_completed_sampling_is_failed_incomplete_and_unqualified(self):
        report, inspection = self.inspect()
        self.assertEqual(report["dimensions"]["execution"], "first_conditional_sampling_returned")
        self.assertEqual(report["dimensions"]["numerical_fit"], "failed")
        self.assertEqual(report["dimensions"]["complete_population_mixture"], "incomplete")
        self.assertEqual(report["original_parent_masses"], [0.25, 0.75])
        self.assertFalse(report["qualification"])
        self.assertFalse(report["automatic_retry"])
        self.assertFalse(report["owner_observation"]["is_liveness"])
        self.assertIsNone(report["recommendation"])
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(temp)
            self.addCleanup(ledger.close)
            ledger.create_cycle("create", "fixture", {"role": "original"}, {"cycle": 2026}, "counts", "fixture")
            before = ledger.export("fixture")
            captured = attach_evidence(ledger, "failure", report, inspection)
            self.assertEqual(ledger.export("fixture"), before)
            self.assertEqual(captured, attach_evidence(ledger, "failure", report, inspection))
            self.assertEqual(json.loads(ledger._artifact(captured["sha256"]))["dimensions"], report["dimensions"])

    def test_cli_uses_same_inspection_contract_for_attachment(self):
        request = dict(
            schema="run_receipt_request_v1",
            kind="senate_completed_failure",
            terminal_pin=self.terminal_pin(),
            expected_plan_sha256=self.plan_pin["sha256"],
            declared_scope=self.scope,
        )
        self.write("request.json", request)
        result = subprocess.run(
            [
                sys.executable,
                str(BASE / "run_receipts.py"),
                "--source-root",
                str(self.root),
                "--request",
                str(self.root / "request.json"),
                "--journal-root",
                str(self.root / "journal"),
                "--command-id",
                "cli-fixture",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["report"]["qualification"])
        self.assertIn("sha256", value["journal_evidence"])

    def test_positive_pending_and_timeout_claims_rejected(self):
        original = copy.deepcopy(self.terminal)
        for key, value in [
            ("authoritative_exit_code", 0),
            ("complete_cut", True),
            ("numerically_qualified_population_conditionals", 1),
            ("resource_budget_exceeded", True),
            ("terminal_confirmed_by_tool_handle", False),
            ("no_automatic_retry", False),
            ("completed_population_conditionals", True),
        ]:
            with self.subTest(key=key):
                self.terminal = dict(original, **{key: value})
                with self.assertRaises(ReceiptError):
                    self.inspect()

    def test_scope_and_plan_cannot_silently_change(self):
        pin = self.terminal_pin()
        with self.assertRaisesRegex(ReceiptError, "plan binding"):
            inspect_completed_failure(self.root, pin, "0" * 64, self.scope)
        with self.assertRaisesRegex(ReceiptError, "scope mismatch"):
            inspect_completed_failure(self.root, pin, self.plan_pin["sha256"], dict(self.scope, purpose="different"))

    def test_original_parents_cannot_be_renormalized_to_completed_subset(self):
        self.change("attempt1/fit/failure.json", lambda d: d.update(required_parent_ids=["p0"]))
        with self.assertRaisesRegex(ReceiptError, "parent support"):
            self.inspect()

    def test_empty_terminal_diagnostics_rejected(self):
        self.terminal["diagnostics"] = {}
        with self.assertRaisesRegex(ReceiptError, "diagnostics keys"):
            self.inspect()

    def test_preservation_error_cannot_hide_behind_complete_archive(self):
        self.change("attempt1/fit/failure.json", lambda d: d.update(returned_inference_preservation_error="failed"))
        with self.assertRaisesRegex(ReceiptError, "preservation failed"):
            self.inspect()

    def test_native_terminal_directory_is_bound(self):
        pin = self.terminal_pin()
        moved = self.write("unrelated/terminal.json", (self.root / pin["path"]).read_bytes())
        with self.assertRaisesRegex(ReceiptError, "terminal attempt directory"):
            inspect_completed_failure(self.root, moved, self.plan_pin["sha256"], self.scope)

    def test_observer_inventory_cannot_escape_fit_or_hide_drift(self):
        self.change(
            "attempt1/fit/failure.json",
            lambda d: d.update(observer_inventory={"complete": True, "errors": [], "files": {"../other": "0" * 64}}),
        )
        with self.assertRaisesRegex(ReceiptError, "observation file name"):
            self.inspect()

    def test_result_cannot_be_replaced_with_other_parent(self):
        self.change("attempt1/fit/parent-0-result.json", lambda d: d.update(parent_id="p1"))
        with self.assertRaisesRegex(ReceiptError, "result identity"):
            self.inspect()

    def test_returned_archive_hash_drift_rejected(self):
        pin = self.terminal_pin()
        (self.root / "attempt1/fit/parent-0-returned-inference.pkl").write_bytes(b"changed")
        with self.assertRaisesRegex(ReceiptError, "hash mismatch"):
            inspect_completed_failure(self.root, pin, self.plan_pin["sha256"], self.scope)

    def test_archive_rebinding_cannot_skip_native_links(self):
        self.write("attempt1/fit/parent-0-returned-inference.pkl", b"different but newly pinned")
        with self.assertRaisesRegex(ReceiptError, "archive file mismatch"):
            self.inspect()

    def test_cross_attempt_inventory_rejected(self):
        old = self.files.pop("attempt1/fit/parent-0-result.json")
        self.write("attempt2/fit/parent-0-result.json", (self.root / old["path"]).read_bytes())
        pin = self.terminal_pin()
        self.terminal["preserved_files"].append(self.files["attempt2/fit/parent-0-result.json"])
        pin = self.write("terminal.json", self.terminal)
        with self.assertRaisesRegex(ReceiptError, "native attempt path mismatch"):
            inspect_completed_failure(self.root, pin, self.plan_pin["sha256"], self.scope)

    def test_csv_extrema_and_nonfinite_values_rejected(self):
        for value in ["1.3", "nan", "inf"]:
            with self.subTest(value=value):
                self.write(
                    "attempt1/fit/parent-0-diagnostics.csv",
                    f",r_hat,ess_bulk,ess_tail\nquantity,{value},10,20\n".encode(),
                )
                with self.assertRaises(ReceiptError):
                    self.inspect()

    def test_report_or_source_cannot_change_between_inspect_and_attach(self):
        report, inspection = self.inspect()
        with tempfile.TemporaryDirectory() as temp:
            ledger = Ledger(temp)
            self.addCleanup(ledger.close)
            with self.assertRaisesRegex(ReceiptError, "report changed"):
                attach_evidence(ledger, "bad", dict(report, qualification=True), inspection)
            (self.root / "attempt1/fit/parent-0-returned-inference.pkl").write_bytes(b"drift")
            with self.assertRaisesRegex(ReceiptError, "hash mismatch"):
                attach_evidence(ledger, "bad", report, inspection)
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0], 0)


class RealCompletedFailureTests(unittest.TestCase):
    def test_exact_owner_failure_preserves_all_parent_masses(self):
        if not (OWNER / (NATIVE + "terminal-root-v1.json")).exists():
            self.skipTest("real owner checkout absent")
        plan = json.loads((OWNER / (NATIVE + "plan.json")).read_text())
        report, _ = inspect_completed_failure(
            OWNER,
            dict(path=NATIVE + "terminal-root-v1.json", sha256=TERMINAL_SHA),
            PLAN_SHA,
            {key: plan[key] for key in SCOPE_KEYS},
        )
        self.assertEqual(report["csv_rows"], 1376)
        self.assertEqual(report["completed_population_conditionals"], 1)
        self.assertEqual(report["required_population_conditionals"], 8)
        self.assertEqual(report["original_parent_masses"], [0.125] * 8)
        self.assertAlmostEqual(report["numerical_diagnostics"]["max_rhat"], 1.3208674745084075)
        self.assertFalse(report["qualification"])


if __name__ == "__main__":
    unittest.main()
