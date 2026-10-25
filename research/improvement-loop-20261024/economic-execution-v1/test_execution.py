"""Bounded execution tests: synthetic fixtures, no sampling or scientific fits."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("execution_test_runner", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
scoring = runner.load_module("execution_test_scoring", HERE / "scoring.py")


class ExecutionTests(unittest.TestCase):
    def test_frozen_preparation_and_all_16_jobs_without_sampling(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "prepared"
            with patch("pymc.sample", side_effect=AssertionError("no sampler")):
                runner.prepare_execution(path)
                plan, frozen, producer, _ = runner.validate_execution(
                    path / "plan.json", runner.pin(path / "plan.json")["sha256"]
                )
                self.assertEqual(len(plan["jobs"]), 16)
                self.assertEqual([j["arm"] for j in plan["jobs"][:4]], list(runner.ARMS))
                economic = next(p for p in plan["jobs"][3]["recipe"]["providers"] if p["instance_id"] == "economy")
                self.assertEqual(economic["parameters"][0]["sd"], 0.025)
                module, training, target = runner.compile_job(plan["jobs"][3], frozen["folds"][0], producer)
                self.assertEqual(len(target), 30)
                self.assertEqual(len(training.race_ids), 299)
                self.assertIn("economy.headline_cpi", training.parameter_names)
                self.assertEqual(training.recipe, module.Recipe.model_validate(plan["jobs"][3]["recipe"]))
                changed = json.loads((path / "plan.json").read_bytes())
                changed["source_pins"] = changed["source_pins"][:-1]
                altered = path / "altered.json"
                runner.write(altered, changed)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    runner.validate_execution(altered, runner.pin(altered)["sha256"])
                with self.assertRaises(FileExistsError):
                    runner.prepare_execution(path)

    def test_runtime_acceptance_and_nested_pin_corruption(self):
        source = HERE / "runtime/acceptance.json"
        review = runner.validate_runtime_acceptance(runner.pin(source))
        self.assertTrue(review["execution_plan_review_still_required"])
        with tempfile.TemporaryDirectory() as directory:
            malformed = {**review, "result": {**review["result"], "sha256": "0" * 64}}
            path = Path(directory) / "acceptance.json"
            runner.write(path, malformed)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                runner.validate_runtime_acceptance(runner.pin(path))
            with patch.object(runner, "validate_execution", side_effect=AssertionError("before imports")):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    runner.run(None, None, None, runner.pin(path))

    def test_genuine_joint_rows_and_shared_new_cycle(self):
        beta = np.arange(24, dtype=float).reshape(4, 3, 2) / 20
        tc = np.full((4, 3), 0.2)
        tr = np.full((4, 3), 0.3)
        bank = az.from_dict(posterior={"beta": beta, "tau_cycle": tc, "tau_race": tr})
        design = np.array([[1.0, 2.0], [1.0, -1.0]])
        module = SimpleNamespace(_design=lambda *_: (np.array([0.1, -0.1]), design))
        training = SimpleNamespace(recipe=None, admission=None)
        target = pd.DataFrame({"race_id": ["synthetic-a", "synthetic-b"]})
        actual = runner.predictive_arrays(training, target, bank, module, 137, 5)
        repeated = runner.predictive_arrays(training, target, bank, module, 137, 5)
        location = np.repeat(np.array([0.1, -0.1]) + beta @ design.T, 5, axis=1)
        expected = location + 0.2 * actual["new_cycle_z"][:, :, None] + 0.3 * actual["new_race_z"]
        np.testing.assert_allclose(actual["logits"], expected, atol=1e-15)
        np.testing.assert_array_equal(actual["posterior_draw"], np.repeat(np.arange(3), 5))
        np.testing.assert_array_equal(actual["shares"], repeated["shares"])
        self.assertFalse(np.array_equal(actual["new_race_z"][:, :, 0], actual["new_race_z"][:, :, 1]))

    def test_scoring_matches_pairwise_crps_and_frozen_masks(self):
        def brute_crps(x, w, y):
            self.assertTrue((w == 1).all())
            return float(abs(x - y).mean() - 0.5 * abs(x[:, None] - x[None, :]).mean())

        values = np.array([0.2, 0.5, 0.8, 0.9]).reshape(1, 4, 1)
        row = scoring.scores(values, [0.7], ["synthetic"], brute_crps)[0]
        self.assertAlmostEqual(row["share_crps"], 0.1)
        self.assertEqual(row["win_probability"], 0.5)
        self.assertAlmostEqual(row["win_brier"], 0.25)
        self.assertAlmostEqual(row["win_log_score"], np.log(2))
        lo95, lo80, median, hi80, hi95 = np.quantile(values.ravel(), [0.025, 0.1, 0.5, 0.9, 0.975])
        expected = (0.5 * abs(median - 0.7) + 0.1 * (hi80 - lo80) + 0.025 * (hi95 - lo95)) / 2.5
        self.assertAlmostEqual(row["share_wis"], expected)
        self.assertEqual(scoring.masks(["synthetic"], values)["near_even"], ["synthetic"])
        with self.assertRaises(ValueError):
            scoring.scores(values, [0.5], ["synthetic"], brute_crps)
        with self.assertRaises(ValueError):
            scoring.scores(values * 2, [0.7], ["synthetic"], brute_crps)

    def test_crps_influence_matches_weight_perturbation_not_raw_mean(self):
        values = np.array([0.1, 0.3, 0.7, 0.95])
        truth = 0.6
        influence = scoring.crps_influence(values.reshape(1, 4, 1), np.array([truth])).ravel()
        centered = influence - influence.mean()

        def weighted(w):
            return w @ abs(values - truth) - 0.5 * w @ abs(values[:, None] - values[None, :]) @ w

        initial = np.full(4, 0.25)
        for i in range(4):
            direction = np.eye(4)[i] - initial
            numerical = (weighted(initial + 1e-6 * direction) - weighted(initial - 1e-6 * direction)) / 2e-6
            self.assertAlmostEqual(centered[i], numerical, places=9)

    def test_constant_win_score_precision_is_explicitly_unresolved(self):
        def mcse(values):
            return {"mcse": np.std(values, axis=(0, 1))}

        result, unresolved = scoring.score_mcse(np.full((4, 100, 1), 0.8), np.array([0.7]), mcse)
        self.assertEqual(unresolved, [0])
        self.assertEqual(result["win_log_score"], 0)

    def test_all_seals_and_masks_checked_before_label_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            producer = SimpleNamespace(PANEL=folder)
            with patch("pandas.read_csv", side_effect=AssertionError("labels opened")) as reader:
                with self.assertRaises(FileNotFoundError):
                    runner.load_labels_after_seals(folder, "plan", 2018, ["synthetic"], producer)
                self.assertFalse(reader.called)
                seals = []
                for arm in runner.ARMS:
                    (folder / arm).mkdir()
                    prediction = folder / arm / "prediction.npz"
                    prediction.write_bytes(b"synthetic sealed prediction")
                    runner.write(
                        folder / arm / "prediction-seal.json",
                        dict(plan_sha256="plan", target=2018, arm=arm, predictions=runner.pin(prediction)),
                    )
                    seals.append(runner.pin(folder / arm / "prediction-seal.json"))
                runner.write(
                    folder / "masks.json",
                    dict(
                        policy=scoring.POLICY,
                        race_ids={"all": ["synthetic"], "competitive": [], "near_even": []},
                        source_prediction_seals=seals,
                    ),
                )
                (folder / "F0/prediction.npz").write_bytes(b"changed after sealing")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    runner.load_labels_after_seals(folder, "plan", 2018, ["synthetic"], producer)
                self.assertFalse(reader.called)

    def test_target_label_parser_filters_metadata_before_numeric_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "labels").mkdir()
            (folder / "labels/outcomes-SEPARATE.csv").write_text(
                "race_id,cycle,actual_dem_share\nsynthetic,2018,0.6\nfuture,2020,DO_NOT_PARSE\n"
            )
            seals = []
            for arm in runner.ARMS:
                (folder / arm).mkdir()
                predictions = folder / arm / "prediction.npz"
                np.savez(predictions, shares=np.full((4, 8, 1), 0.6))
                runner.write(
                    folder / arm / "prediction-seal.json",
                    dict(plan_sha256="plan", target=2018, arm=arm, predictions=runner.pin(predictions)),
                )
                seals.append(runner.pin(folder / arm / "prediction-seal.json"))
            runner.write(
                folder / "masks.json",
                dict(
                    policy=scoring.POLICY,
                    race_ids={"all": ["synthetic"], "competitive": [], "near_even": []},
                    source_prediction_seals=seals,
                ),
            )
            producer = SimpleNamespace(PANEL=folder, EXPECTED={"outcomes": "fixture"}, authenticated=lambda *_: None)
            values = runner.load_labels_after_seals(folder, "plan", 2018, ["synthetic"], producer)
            np.testing.assert_array_equal(values, [0.6])

    @staticmethod
    def fake_bank(draws=8, tune=4):
        rng = np.random.default_rng(19)
        return az.from_dict(
            posterior={
                "beta": rng.normal(size=(1, draws, 2)),
                "tau_cycle": np.ones((1, draws)),
                "tau_race": np.ones((1, draws)),
                "cycle_z": rng.normal(size=(1, draws, 2)),
            },
            warmup_posterior={"beta": rng.normal(size=(1, tune, 2))},
            dims={"beta": ["coefficient"], "cycle_z": ["source_cycle"]},
            coords={"coefficient": ["synthetic-x", "synthetic-y"], "source_cycle": [2010, 2012]},
            save_warmup=True,
        )

    def test_failure_preserves_chain_and_resume_is_new_and_verified(self):
        job = dict(
            target=2018,
            arm="H0",
            chain_seeds=[1, 2, 3, 4],
            source_binding=dict(parameter_names=["synthetic-x", "synthetic-y"], cycles=[2010, 2012]),
            sampler=dict(draws=8, tune=4, target_accept=0.97, max_treedepth=10),
        )
        module = SimpleNamespace(build_pymc_model=lambda _: contextlib.nullcontext())
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "first"
            with patch("pymc.sample", side_effect=[self.fake_bank(), RuntimeError("synthetic stop")]):
                with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                    runner.fit_job(job, module, None, old, "plan")
            original = runner.pin(old / "chain-0/raw-bank.nc")
            receipt_pin = runner.pin(old / "chain-0-complete.json")
            resume_receipts = {receipt_pin["path"]: receipt_pin}
            self.assertTrue((old / "failure.json").exists())
            with patch("pymc.sample", side_effect=[self.fake_bank() for _ in range(3)]) as sampler:
                combined, refs = runner.fit_job(
                    job, module, None, Path(directory) / "second", "plan", old, resume_receipts
                )
                self.assertEqual(sampler.call_count, 3)
                self.assertEqual(combined.posterior.sizes["chain"], 4)
                self.assertEqual(refs[0], original)
                self.assertEqual(runner.pin(old / "chain-0/raw-bank.nc"), original)
            (old / "chain-0/raw-bank.nc").write_bytes(b"changed")
            rebound = json.loads((old / "chain-0-complete.json").read_bytes())
            rebound["raw_bank"] = runner.pin(old / "chain-0/raw-bank.nc")
            (old / "chain-0-complete.json").write_bytes(runner.canonical(rebound))
            with patch("pymc.sample", side_effect=AssertionError("no sampler")) as sampler:
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    runner.fit_job(job, module, None, Path(directory) / "third", "plan", old, resume_receipts)
                self.assertFalse(sampler.called)

    def test_nonfinite_coordinate_cannot_disappear_from_diagnostic_reduction(self):
        bank = az.from_dict(
            posterior={
                "beta": np.ones((4, 100, 2)),
                "tau_cycle": np.ones((4, 100)),
                "tau_race": np.ones((4, 100)),
                "cycle_z": np.ones((4, 100, 2)),
            },
            sample_stats={"diverging": np.zeros((4, 100), bool), "tree_depth": np.ones((4, 100), int)},
        )
        report = runner.fit_diagnostics(bank, dict(max_rank_rhat=1.01, min_bulk_ess=400, min_tail_ess=400))
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["finite"])

    def test_all_fold_execution_roundtrip_with_explicit_synthetic_banks(self):
        """Exercise real compilation/sealing/scoring; replace every fit with a toy bank."""

        def synthetic_fit(job, module, training, directory, *_):
            directory.mkdir()
            rng = np.random.default_rng(job["sampler"]["seed"])
            bank = az.from_dict(
                posterior={
                    "beta": rng.normal(0, 0.01, (4, 8, len(training.parameter_names))),
                    "tau_cycle": np.full((4, 8), 0.15),
                    "tau_race": np.full((4, 8), 0.2),
                }
            )
            return bank, []

        def toy_mcse(values, **_):
            return {"mcse": np.std(values, axis=(0, 1)) / np.sqrt(values.shape[0] * values.shape[1])}

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            directory = Path(directory)
            runner.prepare_execution(directory / "prepared")
            plan = directory / "prepared/plan.json"
            runner.validate_execution(plan, runner.pin(plan)["sha256"])[2].factory()
            certificate = runner.pin(HERE / "runtime/acceptance.json")
            with (
                patch("pymc.sample", side_effect=AssertionError("no sampler")) as sampler,
                patch.object(runner, "fit_job", side_effect=synthetic_fit) as fitting,
                patch.object(runner, "fit_diagnostics", return_value={"passed": True}),
                patch.object(runner, "prediction_diagnostics", return_value={"passed": True}),
                patch("src.forecasting.response_v5_reuse.correlated_mcse", side_effect=toy_mcse),
            ):
                runner.run(plan, runner.pin(plan)["sha256"], directory / "synthetic-attempt", certificate)
                self.assertEqual(fitting.call_count, 16)
                self.assertFalse(sampler.called)
            summary = json.loads((directory / "synthetic-attempt/summary.json").read_bytes())
            self.assertEqual(len(summary["cycles"]), 4)
            self.assertEqual(summary["tiers"]["A"]["cycles"], [2018, 2022])
            self.assertEqual(summary["tiers"]["B"]["cycles"], [2020, 2024])
            self.assertFalse(summary["count_qualification"])
            terminal_path = directory / "synthetic-attempt/2018/H0/terminal.json"
            terminal = json.loads(terminal_path.read_bytes())
            terminal.update(status="numerical_failed_preserved", numerical_passed=False)
            terminal_path.write_bytes(runner.canonical(terminal))
            manifest = directory / "continuation.json"
            runner.seal_continuation(directory / "synthetic-attempt", manifest)
            frozen_manifest = runner.pin(manifest)
            negative_bytes = terminal_path.read_bytes()
            with patch("pymc.sample", side_effect=AssertionError("no sampler")):
                terminal_path.unlink()
                with self.assertRaisesRegex(ValueError, "presence changed"):
                    runner.run(
                        plan, runner.pin(plan)["sha256"], directory / "rejected-absent", certificate, frozen_manifest
                    )
                self.assertFalse((directory / "rejected-absent").exists())
                terminal_path.write_bytes(negative_bytes)
                changed = {**terminal, "status": "completed", "numerical_passed": True}
                terminal_path.write_bytes(runner.canonical(changed))
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    runner.run(
                        plan, runner.pin(plan)["sha256"], directory / "rejected-change", certificate, frozen_manifest
                    )
                self.assertFalse((directory / "rejected-change").exists())
                terminal_path.write_bytes(negative_bytes)
                with self.assertRaisesRegex(ValueError, "not an automatic retry"):
                    runner.run(
                        plan, runner.pin(plan)["sha256"], directory / "rejected-negative", certificate, frozen_manifest
                    )
                self.assertFalse((directory / "rejected-negative").exists())

    def test_continuation_absence_cannot_be_replaced_after_sealing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            binding = dict(path=str(path), pin=None)
            self.assertIsNone(runner.checked_continuation_terminal(path, binding))
            runner.write(path, dict(status="completed", numerical_passed=True))
            with self.assertRaisesRegex(ValueError, "presence changed"):
                runner.checked_continuation_terminal(path, binding)

    def test_regression_flags_use_independent_arm_numerical_error(self):
        base = dict(
            n=2,
            share_crps=0.1,
            win_brier=0.2,
            win_log_score=0.3,
            mcse=dict(share_crps=0.001, win_brier=0.001, win_log_score=0.001),
        )
        worse = {**base, "share_crps": 0.11}
        cycle = dict(
            target=2022,
            tier="A",
            arms={arm: {"strata": {"all": base if arm == "F0" else worse}} for arm in runner.ARMS},
        )
        report = runner.regression_report([cycle])["per_cycle"][0]
        self.assertTrue(report["loss_beyond_two_mcse"])
        self.assertAlmostEqual(report["numerical_mcse"], np.sqrt(2) * 0.001)


if __name__ == "__main__":
    unittest.main()
