"""Actual saved-bank consumer tests; no MCMC and no owner-output mutations."""

import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
from consumer import FORECAST_CUTOFF, HERE, REVIEW_SHA, TARGET, actual_inputs, load_bank, make_mixture
from numpy.testing import assert_allclose


def load(root=HERE, **changes):
    arguments = dict(
        expected_review_sha256=REVIEW_SHA, target_cycle=2026, forecast_cutoff_utc=FORECAST_CUTOFF, target_label=TARGET
    )
    arguments.update(changes)
    return load_bank(root, **arguments)


class ConsumerTests(unittest.TestCase):
    def test_output_receipt_binds_code_inputs_and_report(self):
        receipt = json.loads((HERE / "receipt.json").read_text())
        self.assertFalse(receipt["mcmc_launched"])
        self.assertEqual(receipt["expected_review_sha256"], REVIEW_SHA)
        for pin in receipt["artifacts"]:
            self.assertEqual(hashlib.sha256((HERE / pin["file"]).read_bytes()).hexdigest(), pin["sha256"])

    def test_actual_joint_axes_exact_rows_equal_weights_immutable(self):
        bank = load()
        with np.load(HERE / "inputs/posterior-bank.npz", allow_pickle=False) as source:
            assert_allclose(bank.standard_deviations, source["standard_deviations"], rtol=0, atol=0)
            assert_allclose(bank.joint_rows, source["standard_deviations"].reshape(4000, 5), rtol=0, atol=0)
        assert_allclose(bank.weights, np.full(4000, 1 / 4000), rtol=0, atol=0)
        self.assertEqual(bank.standard_deviations.shape, (4, 1000, 5))
        for array in [bank.standard_deviations, bank.log_scales, bank.joint_rows, bank.weights]:
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            bank.target_label = "cut"
        # Joint dependence is materially present, not five independent marginals.
        correlation = np.corrcoef(bank.log_scales.reshape(4000, 5), rowvar=False)
        self.assertLess(correlation[2, 4], -0.35)
        self.assertLess(correlation[0, 3], -0.3)

    def test_wrong_review_digest_and_failed_review_rejected(self):
        with self.assertRaisesRegex(ValueError, "digest"):
            load(expected_review_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(HERE / "inputs", root / "inputs")
            path = root / "inputs/independent-review.json"
            review = json.loads(path.read_text())
            review["diagnostic_gates_passed"] = False
            data = json.dumps(review).encode()
            path.write_bytes(data)
            # Even a caller explicitly pinning these new bytes cannot pass false gates.
            with self.assertRaisesRegex(ValueError, "review did not pass"):
                load(root, expected_review_sha256=hashlib.sha256(data).hexdigest())

    def test_terminal_npz_and_plan_pins_reject_changed_bytes(self):
        for filename in ("plan.json", "terminal.json", "posterior-bank.npz"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copytree(HERE / "inputs", root / "inputs")
                path = root / "inputs" / filename
                path.write_bytes(path.read_bytes() + b"corruption")
                with self.assertRaisesRegex(ValueError, "digest"):
                    load(root)

    def test_failed_terminal_gate_cannot_be_promoted_by_rebinding_its_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(HERE / "inputs", root / "inputs")
            terminal_path = root / "inputs/terminal.json"
            terminal = json.loads(terminal_path.read_text())
            terminal["diagnostics"]["passed"] = False
            data = json.dumps(terminal).encode()
            terminal_path.write_bytes(data)
            review_path = root / "inputs/independent-review.json"
            review = json.loads(review_path.read_text())
            review["terminal_sha256"] = hashlib.sha256(data).hexdigest()
            review_bytes = json.dumps(review).encode()
            review_path.write_bytes(review_bytes)
            with self.assertRaisesRegex(ValueError, "terminal numerical gates"):
                load(root, expected_review_sha256=hashlib.sha256(review_bytes).hexdigest())

    def test_changed_snapshot_and_rebound_manifest_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "consumer"
            shutil.copytree(HERE, root)
            provider = root / "snapshot-source/src/forecasting/poll_covariance_v1.py"
            provider.write_bytes(provider.read_bytes() + b"\nraise AssertionError('untrusted provider executed')\n")
            manifest_path = root / "input-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            pin = next(
                pin
                for pin in manifest["files"]
                if pin["file"] == "snapshot-source/src/forecasting/poll_covariance_v1.py"
            )
            pin["sha256"] = hashlib.sha256(provider.read_bytes()).hexdigest()
            pin["bytes"] = provider.stat().st_size
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "reviewed snapshot manifest digest mismatch"):
                actual_inputs(root)

    def test_historical_future_source_and_target_reuse_rejected(self):
        for cycle in [2018, 2020, 2022, 2024, 2028, True]:
            with self.subTest(cycle=cycle), self.assertRaisesRegex(ValueError, "historical replay"):
                load(target_cycle=cycle)
        for cutoff in ["2026-10-19T23:59:59Z", "2026-10-20T16:21:43Z", "2026-10-20", "2025-10-25T00:00:00Z"]:
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                load(forecast_cutoff_utc=cutoff)
        for target in ["population_cut_v1", "calibration_cut", ""]:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "explicit marginal"):
                load(target_label=target)

    def test_actual_inputs_no_second_sampling_term_and_report_oracles(self):
        coordinate, y, covariance, provider = actual_inputs(HERE)
        component = json.loads((HERE / "inputs/poll-component.json").read_text())
        assert_allclose(covariance, component["covariance"], rtol=0, atol=1e-15)
        bank = load()
        model = make_mixture(bank, coordinate, provider)
        self.assertEqual(model.covariances.shape, (4000, 6, 6))
        for index in [0, 999, 1000, 3999]:
            expected = provider(
                coordinate, dict(zip(("independent", "cycle", "race", "house", "movement"), bank.joint_rows[index]))
            )
            assert_allclose(model.covariances[index], expected, rtol=0, atol=1e-15)
        report = json.loads((HERE / "report.json").read_text())
        self.assertFalse(report["calibration_cut"])
        self.assertFalse(report["downstream_posterior_fitted"])
        for evaluation in report["evaluations"]:
            self.assertLess(abs(evaluation["log_density_error"]), 1e-12)
            self.assertLess(evaluation["gradient_max_error"], 1e-11)
            self.assertLess(abs(evaluation["fixed_oracle_error"]), 1e-12)
            self.assertLess(evaluation["fixed_gradient_max_error"], 1e-11)
            self.assertEqual(len(evaluation["chain_sensitivity"]), 4)
            self.assertAlmostEqual(model.logpdf(y, evaluation["mean"]), evaluation["log_density"], places=12)


if __name__ == "__main__":
    unittest.main()
