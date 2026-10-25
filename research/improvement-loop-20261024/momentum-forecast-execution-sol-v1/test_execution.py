"""SYNTHETIC ONLY: no historical adapter loading, archived values, labels or scores."""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import execution_sources as sources
import runner

MODULES = sources.load_dependencies()
np = MODULES["sampler"].np
import numerical  # noqa: E402
import references  # noqa: E402


def fixture():
    rng = np.random.default_rng(2026102502)
    k = 20
    prior = MODULES["sampler"].EmpiricalBasisFundamentals(
        offset=np.array([0.02, -0.04, 0.1]),
        design=np.array([[1.0, 0.3, 1, 0, 0, 0], [1.0, -0.2, 0, 1, 0, 0], [1.0, 0.1, 0, 0, 1, 0.2]]),
        beta=rng.normal(0, 0.12, (k, 6)),
        tau_cycle=rng.uniform(0.06, 0.13, k),
        tau_race=rng.uniform(0.12, 0.25, k),
        race_ids=("synthetic:A", "synthetic:B", "synthetic:C"),
        source_identity="synthetic-complete-joint-bank",
        basis_names=tuple("abcdef"),
        coefficient_names=tuple("abcdef"),
        posterior_chain=np.repeat(np.arange(4), 5),
        posterior_draw=np.tile(np.arange(5), 4),
        metadata=dict(synthetic=True),
    )
    likelihood = MODULES["sampler"].GaussianPollLikelihood(
        np.array([0, 1, 0]),
        np.array([0.1, -0.1, 0.13]),
        np.array([[0.04, 0.005, 0.008], [0.005, 0.05, 0.002], [0.008, 0.002, 0.03]]),
        ("syn:p1", "syn:p2", "syn:p3"),
    )
    return prior, likelihood


def provenance():
    return dict(run_id="synthetic-harness", trace_source_pins={"synthetic_fixture": "a" * 64}, synthetic=True)


def small_job(draws=1600):
    return dict(
        cycle=2018, arm="FS5", attempt=0, chains=4, warmup=20, draws=draws, seed=10202026, synthetic_budget=True
    )


class ExecutionTests(unittest.TestCase):
    def temp(self):
        return tempfile.TemporaryDirectory(dir=sources.HERE)

    def test_selected_seeds_and_budgets(self):
        seeds = []
        for cycle in (2018, 2020, 2022, 2024):
            for arm, code in [("FS5", 1), ("FS6", 2), ("LP5", 3)]:
                for attempt in (0, 1):
                    expected = int(
                        np.random.SeedSequence([2026102501, cycle, code, attempt]).generate_state(1, dtype=np.uint64)[0]
                    )
                    self.assertEqual(runner.job_seed(cycle, arm, attempt), expected)
                    self.assertEqual(runner.job_settings(cycle, arm, attempt)["draws"], 4000 if attempt == 0 else 8000)
                    seeds.append(expected)
        self.assertEqual(len(set(seeds)), 24)
        with self.assertRaises(ValueError):
            runner.job_seed(2026, "FS5", 0)
        with self.assertRaises(ValueError):
            runner.job_seed(2018, "PO5", 0)
        with self.assertRaises(ValueError):
            runner.job_seed(2018, "FS5", 2)

    def test_full_synthetic_persistence_and_separate_statuses(self):
        prior, likelihood = fixture()
        with self.temp() as tmp:
            output = Path(tmp) / "synthetic-only"
            with mock.patch.object(
                MODULES["adapter"], "load_adapter", side_effect=AssertionError("actual target forbidden")
            ):
                terminal = runner.run_prepared_job(
                    prior, likelihood, MODULES, small_job(), output, provenance=provenance(), synthetic=True
                )
            self.assertEqual(terminal["status"], "complete_prelabel")
            self.assertTrue(terminal["synthetic"])
            self.assertFalse(terminal["numeric_target_labels"])
            self.assertFalse(terminal["scores"])
            self.assertEqual(sources.json_pin(terminal["artifacts"]["launch"])["pid"], os.getpid())
            seal = json.loads((output / "prediction-seal.json").read_bytes())
            self.assertEqual(seal["artifacts"]["trace"]["sha256"], sources.sha(output / "trace/receipt.json"))
            trace = MODULES["trace"].load_trace(
                output / "trace",
                expected_source_pins={
                    **dict(MODULES["trace"].REQUIRED_SOURCE_PINS),
                    **provenance()["trace_source_pins"],
                },
                expected_receipt_sha256=terminal["artifacts"]["trace"]["sha256"],
            )
            self.assertEqual(len(trace["metadata"]["final_rng_states"]), 4)
            with np.load(output / "original-predictions.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["shares"].shape, (4, 1600, 3))
                np.testing.assert_array_equal(raw["posterior_chain"], prior.posterior_chain[raw["posterior_index"]])
            with np.load(output / "prelabel-components.npz", allow_pickle=False) as component:
                self.assertEqual(component["component_expected_shares"].shape, (4, 1600, 2))
                self.assertEqual(component["quantiles"].shape, (2, 7))
            summary = sources.json_pin(terminal["prelabel_summary"])
            self.assertEqual(
                set(summary["gates"]),
                {"core_state", "bulk_mean_share", "bulk_absolute_win_probability", "bank_chain_consistency"},
            )
            self.assertIn("inherited_bank_warning", summary)
            self.assertIn("interval_display_qualified", summary)
            self.assertIn("tail_qualification", summary)
            self.assertTrue(terminal["partial_observer_receipts"])
            with self.assertRaises(FileExistsError):
                runner.run_prepared_job(
                    prior, likelihood, MODULES, small_job(), output, provenance=provenance(), synthetic=True
                )

    def test_failure_each_phase_keeps_initialized_prefix_and_exception(self):
        prior, likelihood = fixture()
        original = MODULES["sampler"].sample_poll_posterior
        for phase in ("gaussian_conditional", "row_weights", "completed_sweep", "chain_final_rng"):

            def broken(prior, polls, *, observer, **kwargs):
                def callback(record):
                    observer(record)
                    if record["phase"] == phase:
                        raise RuntimeError("synthetic injected " + phase)

                return original(prior, polls, observer=callback, **kwargs)

            modules = {**MODULES, "sampler": SimpleNamespace(sample_poll_posterior=broken)}
            with self.subTest(phase=phase), self.temp() as tmp:
                output = Path(tmp) / "failure"
                with self.assertRaisesRegex(RuntimeError, "synthetic injected"):
                    runner.run_prepared_job(
                        prior, likelihood, modules, small_job(10), output, provenance=provenance(), synthetic=True
                    )
                terminal = json.loads((output / "terminal.json").read_bytes())
                self.assertEqual(terminal["status"], "failed_preserved_no_restart")
                self.assertIn(phase, terminal["exception"])
                self.assertTrue(terminal["partial_observer_receipts"])
                part = sources.json_pin(terminal["partial_observer"])
                metadata = sources.json_pin(part["metadata"])
                self.assertFalse(metadata["restartable"])
                self.assertFalse(metadata["raw_predictive_draw_completeness_claimed"])
                self.assertTrue(metadata["completed_sweep_callback_precedes_missing_race_rng_and_output_assignment"])
                self.assertEqual(len(metadata["final_rng_states_emitted"]), 1 if phase == "chain_final_rng" else 0)
                with np.load(part["arrays"]["file"]["path"], allow_pickle=False) as arrays:
                    self.assertTrue(all(np.isfinite(arrays[name]).all() for name in arrays.files))
                    self.assertEqual(
                        arrays["chain0__observed_theta"].shape[1],
                        metadata["delta_stop_completed_records_per_chain"][0]
                        - metadata["delta_start_completed_records_per_chain"][0],
                    )
                self.assertFalse((output / "prediction-seal.json").exists())

    def test_post_prediction_numerical_failure_keeps_original_and_trace(self):
        prior, likelihood = fixture()
        with self.temp() as tmp:
            output = Path(tmp) / "numerical-failure"
            with mock.patch.object(
                numerical, "conditional_summary", side_effect=ArithmeticError("synthetic quadrature failure")
            ):
                with self.assertRaises(ArithmeticError):
                    runner.run_prepared_job(
                        prior, likelihood, MODULES, small_job(10), output, provenance=provenance(), synthetic=True
                    )
            terminal = json.loads((output / "terminal.json").read_bytes())
            self.assertEqual(terminal["status"], "failed_preserved_no_restart")
            self.assertTrue((output / "prediction-seal.json").exists())
            self.assertEqual(
                sources.json_pin(terminal["artifacts"]["original"])["arrays"]["arrays"]["shares"]["shape"], [4, 10, 3]
            )
            self.assertTrue((output / "trace/provenance.json").exists())

    def test_source_drift_before_scientific_module_import(self):
        with self.temp() as tmp:
            root = Path(tmp)
            shutil.copy(sources.HERE / "source-bindings.json", root / "source-bindings.json")
            shutil.copy(sources.HERE / "runtime.json", root / "runtime.json")
            shutil.copy(sources.HERE / "plan.json", root / "plan.json")
            shutil.copytree(sources.HERE / "snapshots", root / "snapshots")
            f = next((root / "snapshots").iterdir())
            f.write_bytes(f.read_bytes() + b"changed")
            with (
                mock.patch.object(sources, "HERE", root),
                mock.patch.object(sources, "module", side_effect=AssertionError("premature import")),
            ):
                with self.assertRaisesRegex(ValueError, "source byte drift"):
                    sources.load_dependencies()
        with mock.patch.object(sys, "executable", "/usr/bin/python3"):
            with self.assertRaisesRegex(ValueError, "interpreter/prefix"):
                sources.verify_runtime()

    def test_receipt_must_be_external_and_exact(self):
        with self.temp() as tmp:
            path = Path(tmp) / "receipt.json"
            runner.write_json(path, {"schema": "untrusted"})
            with self.assertRaisesRegex(ValueError, "source byte drift"):
                runner.validate_receipt(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "unique source-bound"):
                runner.validate_receipt(path, sources.sha(path))

    def retry_fixture(self, root, trigger=True):
        expected = runner.expected_adapter_report(2018)
        binding = dict(
            execution_group_id="synthetic-retry-fixture",
            execution_manifest_sha256="e" * 64,
            plan_sha256=sources.PLAN_SHA,
            source_bindings_sha256=sources.BINDINGS_SHA,
            runtime_sha256=sources.RUNTIME_SHA,
            independent_execution_review={"path": "synthetic-review", "sha256": "d" * 64},
        )
        records = []
        for arm in ("FS5", "FS6", "LP5"):
            folder = root / arm
            folder.mkdir()
            receipt = dict(
                schema="MomentumForecastRootExecutionReceiptV1",
                run_id="fixture-" + arm,
                action="job",
                output=str(folder),
                job=runner.job_settings(2018, arm, 0),
                **binding,
            )
            runner.write_json(folder / "receipt.json", receipt)
            gates = dict(
                core_state=not trigger,
                bulk_mean_share=True,
                bulk_absolute_win_probability=True,
                bank_chain_consistency=True,
            )
            retry = {key: not value for key, value in gates.items()}
            summary = dict(
                schema="MomentumPollingPrelabelNumericalSummaryV1",
                source_identity=expected["source_identity"],
                synthetic=False,
                numeric_target_labels=False,
                scores=False,
                gates=gates,
                permitted_retry_triggers=retry,
                any_permitted_retry_trigger=trigger,
            )
            runner.write_json(folder / "summary.json", summary)
            provenance = dict(
                execution_receipt=sources.pin(folder / "receipt.json"),
                run_id=receipt["run_id"],
                source_identity=expected["source_identity"],
                forecast_design_sha256=sources.DESIGN_SHA,
                data_source_pins=expected["source_pins"],
                measurement_source_pins=expected["variants"][arm]["source_pins"],
            )
            summary_pin = sources.pin(folder / "summary.json")
            terminal = dict(
                schema="MomentumForecastJobTerminalV1",
                job=receipt["job"],
                status="complete_prelabel",
                synthetic=False,
                output=str(folder),
                source_identity=expected["source_identity"],
                provenance=provenance,
                prelabel_summary=summary_pin,
                artifacts={"prelabel_summary": summary_pin},
            )
            runner.write_json(folder / "terminal.json", terminal)
            records.append(sources.pin(folder / "terminal.json"))
        return binding, records

    def test_matched_retry_receipts_and_source_revision_rejection(self):
        with self.temp() as tmp:
            root = Path(tmp)
            binding, records = self.retry_fixture(root)
            self.assertEqual(len(runner.validate_retry(records, 2018, root / "new", binding)), 3)
            for field in (
                "execution_group_id",
                "plan_sha256",
                "source_bindings_sha256",
                "runtime_sha256",
                "execution_manifest_sha256",
            ):
                other = copy.deepcopy(binding)
                other[field] = "other"
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                    runner.validate_retry(records, 2018, root / "new", other)
            with self.assertRaisesRegex(ValueError, "all three"):
                runner.validate_retry(records[:2], 2018, root / "new", binding)
            terminal = json.loads(Path(records[0]["path"]).read_bytes())
            terminal["job"]["seed"] += 1
            path = root / "changed-terminal.json"
            runner.write_json(path, terminal)
            with self.assertRaisesRegex(ValueError, "exact completed"):
                runner.validate_retry([sources.pin(path), *records[1:]], 2018, root / "new", binding)
        with self.temp() as tmp:
            binding, records = self.retry_fixture(Path(tmp), False)
            with self.assertRaisesRegex(ValueError, "no selected numerical trigger"):
                runner.validate_retry(records, 2018, Path(tmp) / "new", binding)

    def test_input_preparation_failure_keeps_authentic_launch(self):
        with self.temp() as tmp:
            output = Path(tmp) / "source-bound-input-failure"
            receipt = dict(
                action="job",
                job=runner.job_settings(2018, "FS5", 0),
                output=str(output),
                run_id="synthetic-validation-only",
                execution_group_id="synthetic-fixture",
            )
            receipt_pin = dict(path="synthetic-root-receipt", sha256="f" * 64)
            with (
                mock.patch.object(runner, "validate_receipt", return_value=(receipt, receipt_pin, None, None)),
                mock.patch.object(
                    runner, "_actual_job_inputs", side_effect=ValueError("synthetic source/axis failure")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "synthetic source/axis failure"):
                    runner.actual_job(receipt, receipt_pin, MODULES)
            launch = json.loads((output / "launch.json").read_bytes())
            terminal = json.loads((output / "terminal.json").read_bytes())
            self.assertEqual(launch["pid"], os.getpid())
            self.assertEqual(launch["stage"], "preparing_authenticated_inputs")
            self.assertEqual(terminal["stage"], "input_preparation")
            self.assertEqual(terminal["status"], "failed_preserved_no_restart")
            self.assertFalse((output / "prediction-seal.json").exists())

    def test_gaussian_vector_integration_and_no_row_reweighting(self):
        math = MODULES["math"]
        means = np.array([[[0.0, 0.2], [0.3, -0.4]], [[-0.2, 0.1], [0.1, -0.1]]])
        sd = np.full_like(means, 0.5)
        got, receipts = numerical.component_means(means, sd, math, chunk=3)
        expected = np.array(
            [
                math.GaussianLogitMixture(np.array([m]), np.array([s])).mean_share()[0]
                for m, s in zip(means.flat, sd.flat)
            ]
        ).reshape(means.shape)
        np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0)
        self.assertEqual(len(receipts), 3)
        self.assertTrue(
            numerical.all_measured_gates(
                {
                    "coordinates": [
                        {"status": "structural", "structural_constant": True, "gates": {"rhat": None, "mean_ess": None}}
                    ]
                }
            )
        )
        self.assertFalse(
            numerical.all_measured_gates(
                {"coordinates": [{"status": "unresolved_constant", "gates": {"mean_mcse": None}}]}
            )
        )

    def test_missing_requested_gate_never_passes(self):
        self.assertFalse(
            numerical.all_measured_gates({"coordinates": [{"status": "measured", "gates": {}}]}, {"absolute_mcse"})
        )
        self.assertFalse(
            numerical.all_measured_gates(
                {"coordinates": [{"status": "measured", "gates": {"other": True}}]}, {"absolute_mcse"}
            )
        )
        self.assertFalse(
            numerical.all_measured_gates({"coordinates": [{"status": "structural", "gates": {"rhat": None}}]})
        )

    def test_paired_helper_near_one_boundary_and_contradiction_guard(self):
        from scipy.special import log_ndtr

        z = np.linspace(9.0, 10.0, 6400).reshape(4, 1600, 1)
        win, lose = log_ndtr(z), log_ndtr(-z)
        pair = MODULES["paired"].paired_tail_diagnostics(win, lose, ["synthetic:rare"])
        self.assertEqual(pair["win"]["coordinates"][0]["original_one_sided_status"], "unresolved_constant")
        self.assertEqual(pair["win"]["coordinates"][0]["status"], "measured")
        self.assertTrue(numerical.all_measured_gates(pair["win"], {"absolute_mcse"}))
        self.assertTrue(numerical.all_measured_gates(pair["lose"], {"absolute_mcse"}))
        with self.assertRaisesRegex(ValueError, "consistent|consistency|recoverable"):
            MODULES["paired"].paired_tail_diagnostics(win, lose + 0.1, ["synthetic:contradiction"])

    def test_analytic_po5_against_independent_scalar_conjugacy(self):
        like = MODULES["sampler"].GaussianPollLikelihood(
            np.array([0]), np.array([0.8]), np.array([[0.25]]), ("synthetic:p",)
        )
        mean, cov, receipt = references.analytic_po5(like, 2)
        variance = 1 / (1 / 6.25 + 1 / 0.25)
        np.testing.assert_allclose(mean, [variance * 0.8 / 0.25, 0], atol=1e-14)
        np.testing.assert_allclose(cov, np.diag([variance, 6.25]), atol=1e-14)
        self.assertLessEqual(max(receipt["residuals"].values()), 1e-10)
        summary, _ = references.po5_reference(like, ("synthetic:a", "synthetic:b"), ("synthetic:a",), MODULES)
        self.assertAlmostEqual(summary["rows"][1]["expected_share"], 0.5, places=12)
        self.assertFalse(summary["numeric_target_labels"])
        self.assertFalse(summary["scores"])

    def test_archived_reference_synthetic_order_masks_quantiles_and_constant_tails(self):
        # Exactly archived shapes, but all supplied values are generated here.
        rng = np.random.default_rng(32)
        shares = rng.uniform(0, 1, (4, 20000, 2))
        shares[..., 1] = 0.9
        names = ("synthetic:even", "synthetic:constant")
        summary, arrays = references.archived_f0_reference(
            shares, names, np.repeat(np.arange(4000), 5), np.tile(np.arange(5), 4000), names, MODULES
        )
        np.testing.assert_array_equal(arrays["joint_empirical_shares"], shares)
        self.assertEqual(summary["near_even_race_ids"], ["synthetic:even"])
        self.assertEqual(summary["competitive_race_ids"], ["synthetic:even"])
        self.assertEqual(summary["rows"][1]["quantiles"]["0.5"], 0.9)
        self.assertTrue(summary["win_side"]["coordinates"][1]["status"].startswith("unresolved"))
        with self.assertRaisesRegex(ValueError, "row/replicate"):
            references.archived_f0_reference(
                shares, names, np.arange(20000), np.tile(np.arange(5), 4000), names, MODULES
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
