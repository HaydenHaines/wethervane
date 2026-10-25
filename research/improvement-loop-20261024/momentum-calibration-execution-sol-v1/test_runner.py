"""Synthetic solver/gate tests and supplied actual points; no historical solve."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import runner as r
from scipy.optimize import OptimizeResult

HERE = Path(__file__).resolve().parent
THREADS = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def toy(width=6, location=None):
    medians = np.array([0.10, 0.06, 0.06, 0.03, 0.05, 0.05])[:width]
    location = np.log(medians) if location is None else np.asarray(location, float)
    hessian = np.diag(np.arange(1, width + 1, dtype=float))

    def objective(x):
        delta = np.asarray(x) - location
        return float(0.5 * delta @ hessian @ delta), hessian @ delta

    return objective, medians, location


def returned(point, objective, success=True):
    value, gradient = objective(point)
    return OptimizeResult(
        x=np.asarray(point),
        fun=value,
        jac=gradient,
        success=success,
        status=0 if success else 1,
        message="explicit synthetic solver fixture",
        nit=1,
        nfev=1,
        njev=1,
    )


def geometry(scales):
    return [
        dict(
            cycle="synthetic",
            observations=3,
            covariance_condition=2.0,
            covariance_min_eigenvalue=0.01,
            brownian_slope_frobenius_cosine=0.9,
            synthetic=True,
            scales=np.asarray(scales).tolist(),
        )
    ]


class ExecutionTests(unittest.TestCase):
    def fit(self, objective, medians, folder):
        with contextlib.redirect_stdout(io.StringIO()):
            return r.fit_job(objective, medians, folder, dict(synthetic=True, label="toy quadratic"), geometry)

    def test_actual_prepared_points_and_alias_identity_no_optimizer(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "supplied.json"
            with patch.object(r, "minimize", side_effect=AssertionError("no historical optimization")):
                r.supplied_checks(path)
            values = json.loads(path.read_bytes())
            self.assertEqual(len(values["points"]), 12)
            self.assertTrue(
                all(p["value_absolute_difference"] == 0 and p["gradient_max_difference"] == 0 for p in values["points"])
            )
            prep = r.preparation()
            plan, _, _ = prep.load_prepared(
                r.DESIGN["references"][1]["path"], r.DESIGN["references"][1]["sha256"], 2022
            )
            r.verify_alias(plan)
            plan["folds"][-1]["input_identity_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "alias identity"):
                r.verify_alias(plan)

    def test_real_synthetic_five_six_solver_and_declared_budget(self):
        for width in (5, 6):
            objective, medians, location = toy(width)
            with self.subTest(width=width), tempfile.TemporaryDirectory(dir=HERE) as directory:
                folder = Path(directory) / "job"
                with patch.object(r, "minimize", wraps=r.minimize) as solver:
                    report = self.fit(objective, medians, folder)
                self.assertEqual(solver.call_count, 3)
                self.assertTrue(report["gate"]["passed"])
                np.testing.assert_allclose(report["fitted_scales"], np.exp(location), atol=1e-7)
                for call in solver.call_args_list:
                    self.assertEqual(call.kwargs["method"], "L-BFGS-B")
                    self.assertEqual(call.kwargs["options"], r.DESIGN["optimizer"]["options"])
                    self.assertEqual(len(call.kwargs["bounds"]), width)
                for index in range(3):
                    self.assertTrue((folder / f"start-{index}/evaluations.jsonl").stat().st_size)
                    self.assertTrue((folder / f"start-{index}/terminal.json").exists())
                before = (folder / "result.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    self.fit(objective, medians, folder)
                self.assertEqual(before, (folder / "result.json").read_bytes())

    def test_false_success_and_disagreeing_starts_fail(self):
        objective, medians, location = toy()
        fixtures = [
            ([returned(location + 0.1, objective)] * 3, "gradient"),
            (
                [
                    returned(location, objective),
                    returned(location + 0.05, objective),
                    returned(location - 0.05, objective),
                ],
                "agreeing_starts",
            ),
        ]
        for answers, gate in fixtures:
            with self.subTest(gate=gate), tempfile.TemporaryDirectory(dir=HERE) as directory:
                with patch.object(r, "minimize", side_effect=answers):
                    report = self.fit(objective, medians, Path(directory) / "job")
                self.assertFalse(report["gate"]["checks"][gate])
                self.assertFalse(report["gate"]["passed"])
                self.assertEqual(len(report["attempts"]), 3)

    def test_boundary_and_indefinite_hessian_fail(self):
        location = np.full(6, np.log(1e-5))
        objective, medians, _ = toy(location=location)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            with patch.object(r, "minimize", side_effect=[returned(location, objective)] * 3):
                report = self.fit(objective, medians, Path(directory) / "boundary")
            self.assertFalse(report["gate"]["checks"]["bounds"])
        objective, medians, location = toy()
        diagonal = np.array([-1, 2, 3, 4, 5, 6])

        def saddle(x):
            delta = x - location
            return float(0.5 * np.dot(diagonal * delta, delta)), diagonal * delta

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            with patch.object(r, "minimize", side_effect=[returned(location, saddle)] * 3):
                report = self.fit(saddle, medians, Path(directory) / "saddle")
            self.assertFalse(report["gate"]["checks"]["hessian_positive"])
            self.assertFalse(report["gate"]["passed"])

    def test_hessian_step_instability_fails_without_changed_objective(self):
        _, medians, location = toy()
        epsilon = 7e-5
        amplitude = 0.5

        def rough(x):
            d = x - location
            gradient = d.copy()
            gradient[0] = d[0] * (1 + amplitude * np.cos(d[0] / epsilon))
            value = 0.5 * np.dot(d, d) + amplitude * (
                epsilon * d[0] * np.sin(d[0] / epsilon) + epsilon**2 * (np.cos(d[0] / epsilon) - 1)
            )
            return float(value), gradient

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            with patch.object(r, "minimize", side_effect=[returned(location, rough)] * 3):
                report = self.fit(rough, medians, Path(directory) / "rough")
            self.assertFalse(report["gate"]["checks"]["hessian_stable"])
            self.assertTrue(report["gate"]["checks"]["hessian_positive"])

    def test_failed_start_preserved_with_two_agreeing_successes_no_retry(self):
        objective, medians, location = toy()
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            folder = Path(directory) / "job"
            with patch.object(
                r,
                "minimize",
                side_effect=[
                    RuntimeError("synthetic start failure"),
                    returned(location, objective),
                    returned(location, objective),
                ],
            ) as solver:
                report = self.fit(objective, medians, folder)
            self.assertEqual(solver.call_count, 3)
            self.assertTrue((folder / "start-0/failure.json").exists())
            self.assertTrue(report["gate"]["passed"])
            self.assertFalse(report["attempts"][0]["optimizer_success"])
            with patch.object(r, "minimize", side_effect=[returned(location, objective, False)] * 3):
                failed = self.fit(objective, medians, Path(directory) / "all-unsuccessful")
            self.assertFalse(failed["gate"]["passed"])
            self.assertIsNone(failed["gate"]["best_attempt"])

    def test_nonfinite_evaluation_and_hessian_exception_are_preserved_failures(self):
        objective, medians, location = toy()

        def invalid(point):
            return np.nan, np.full(len(point), np.nan)

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            folder = Path(directory) / "nonfinite"
            report = self.fit(invalid, medians, folder)
            self.assertFalse(report["gate"]["passed"])
            for index in range(3):
                self.assertTrue((folder / f"start-{index}/failure.json").exists())
                records = [
                    json.loads(line) for line in (folder / f"start-{index}/evaluations.jsonl").read_text().splitlines()
                ]
                self.assertEqual(records[0]["phase"], "evaluation_started")
                self.assertFalse(records[-1]["finite"])

        def bad_hessian(point):
            if np.max(abs(point - location)) > 1e-8:
                raise RuntimeError("synthetic Hessian evaluation failure")
            return objective(point)

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            folder = Path(directory) / "bad-hessian"
            with patch.object(r, "minimize", side_effect=[returned(location, objective)] * 3):
                report = self.fit(bad_hessian, medians, folder)
            self.assertFalse(report["gate"]["checks"]["hessian_finite"])
            self.assertIn("synthetic Hessian", (folder / "hessian.json").read_text())

    def test_six_job_routing_and_eight_fold_receipts_with_fit_stubs_only(self):
        def synthetic_stub(objective, medians, folder, metadata, geometry_callback):
            folder.mkdir(parents=True)
            r.write(folder / "terminal.json", dict(passed=True, synthetic_fit_stub=True, metadata=metadata))
            return dict(gate=dict(passed=True))

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory) / "stubbed-orchestration"
            with (
                patch.object(
                    r.sources,
                    "verify_execution",
                    return_value=({"run_id": "explicit-synthetic-orchestration"}, r.DESIGN),
                ),
                patch.object(r, "fit_job", side_effect=synthetic_stub) as fit,
                patch.object(r, "minimize", side_effect=AssertionError("no historical solve")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                r.run(dict(path="explicit-synthetic-receipt", sha256="synthetic"), output)
            self.assertEqual(fit.call_count, 6)
            self.assertEqual(len(list((output / "folds").rglob("*.json"))), 8)
            for path in (output / "folds/2024").iterdir():
                alias = json.loads(path.read_bytes())
                self.assertTrue(alias["aliased"])
                self.assertEqual(alias["calibration_target"], 2022)
                r.sources.checked(alias["fit_terminal"])

    def test_external_receipt_pin_runtime_and_output_binding(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            output = root / "attempt"
            receipt = dict(
                schema="MomentumCalibrationExecutionReceiptV1",
                run_id="synthetic-receipt-test",
                design=r.BINDINGS["files"][0],
                preparation_acceptance=r.BINDINGS["files"][-2],
                runtime=dict(path=str(HERE / "runtime.json"), sha256=r.sources.RUNTIME_SHA),
                runner_manifest=dict(path=str(HERE / "artifact-manifest.json"), sha256="synthetic-manifest"),
                output_directory=str(output),
            )
            path = root / "receipt.json"
            r.write(path, receipt)
            receipt_pin = r.sources.pin(path)
            manifest = dict(execution_design_sha256=r.sources.DESIGN_SHA, source_bindings_sha256=r.sources.BINDINGS_SHA)
            with patch.dict(os.environ, THREADS), patch.object(r.sources, "verify_manifest", return_value=manifest):
                r.sources.verify_execution(receipt_pin, output)
                with self.assertRaisesRegex(ValueError, "bound exclusive output"):
                    r.sources.verify_execution(receipt_pin, root / "different")
                with patch.object(r.sources, "runtime_identity", return_value={"wrong": "environment"}):
                    with self.assertRaisesRegex(ValueError, "runtime"):
                        r.sources.verify_execution(receipt_pin, output)
            path.write_bytes(path.read_bytes() + b"changed")
            with patch.object(r, "minimize", side_effect=AssertionError("no solve")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    r.run(receipt_pin, output)
            self.assertFalse(output.exists())

    def test_source_corruption_before_scientific_imports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            for name in ("runner.py", "execution_sources.py", "source-bindings.json"):
                shutil.copyfile(HERE / name, root / name)
            path = root / "source-bindings.json"
            path.write_bytes(path.read_bytes() + b"changed")
            code = (
                "try:\n import runner\nexcept ValueError:\n import sys\n assert 'numpy' not in sys.modules\n"
                " assert 'scipy' not in sys.modules\n print('blocked')\nelse:\n"
                " raise AssertionError('accepted changed source')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("blocked", result.stdout)

    def test_missing_runtime_rejected_before_scientific_imports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            for name in ("runner.py", "execution_sources.py", "source-bindings.json"):
                shutil.copyfile(HERE / name, root / name)
            shutil.copytree(HERE / "snapshots", root / "snapshots")
            code = (
                "try:\n import runner\nexcept FileNotFoundError:\n import sys\n"
                " assert 'numpy' not in sys.modules\n assert 'scipy' not in sys.modules\n print('missing rejected')\n"
                "else:\n raise AssertionError('accepted missing runtime')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("missing rejected", result.stdout)


if __name__ == "__main__":
    unittest.main()
