"""Supplied-point verification only: these tests never launch an MCMC sampler."""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from calibration_bank import (
    COMPONENTS,
    HERE,
    LOG_SD,
    MEDIANS,
    CalibrationTarget,
    build_model,
    diagnostics,
    likelihood_op,
    load_prepared,
    prepare_target,
    run,
    validate_rows,
    verify_compiled_density,
)
from scipy.stats import multivariate_normal, norm


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = Path(os.environ.get("CALIBRATION_PREPARED", HERE / "prepared-v1"))
        cls.plan, cls.full, cls.original = prepare_target(cls.prepared)
        _, cls.rows, _ = load_prepared(cls.prepared)
        provider = cls.full.provider
        coordinates = provider.PollCoordinates(
            ("synthetic:0", "synthetic:1", "synthetic:2"),
            ("fixture:a", "fixture:a", "fixture:b"),
            ("2020",) * 3,
            ("one", "two", "one"),
            np.array([1.0, 12.0, 30.0]),
        )
        cls.tiny = CalibrationTarget(provider, [(np.array([0.12, -0.03, 0.08]), coordinates.kernels())])

    def test_normalized_density_prior_once_and_no_optimizer_bounds(self):
        for point in (np.log(MEDIANS), np.log([0.09, 0.08, 0.07, 0.04, 0.06]), np.log([3.0] * 5)):
            value, _ = self.tiny.value_gradient(point)
            expected = norm.logpdf(point, loc=np.log(MEDIANS), scale=LOG_SD).sum()
            for residual, kernels in self.tiny.blocks:
                covariance = np.einsum("k,kij->ij", np.exp(2 * point), kernels)
                expected += multivariate_normal.logpdf(residual, mean=np.zeros(len(residual)), cov=covariance)
            self.assertAlmostEqual(value, expected, places=11)

    def test_supplied_point_analytic_gradients_against_finite_differences(self):
        points = (np.log(MEDIANS), np.log([0.09, 0.08, 0.07, 0.04, 0.06]))
        for target in (self.tiny, self.full):
            for point in points:
                _, gradient = target.value_gradient(point)
                step = 1e-5
                finite = np.array(
                    [
                        (target.value_gradient(point + step * axis)[0] - target.value_gradient(point - step * axis)[0])
                        / (2 * step)
                        for axis in np.eye(5)
                    ]
                )
                np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=2e-5)

    def test_original_owner_mode_objective_is_exactly_reproduced(self):
        logs = np.log([self.original["standard_deviations"][key] for key in COMPONENTS])
        value, gradient = self.full.value_gradient(logs)
        constant = 5 * np.log(LOG_SD * np.sqrt(2 * np.pi))
        self.assertAlmostEqual(-value - constant, self.original["recomputed_objective"], places=8)
        np.testing.assert_allclose(-gradient, self.original["analytic_gradient"], atol=1e-7, rtol=0)
        self.assertEqual([len(residual) for residual, _ in self.full.blocks], [467, 463, 288, 265])

    def test_pytensor_likelihood_gradient_and_pymc_full_density(self):
        import pytensor
        import pytensor.tensor as pt

        point = np.log([0.09, 0.08, 0.07, 0.04, 0.06])
        variable = pt.dvector("supplied_point")
        expression = likelihood_op(self.tiny)(variable)
        compiled = pytensor.function([variable], [expression, pt.grad(expression, variable)])
        expected_value, expected_gradient = self.tiny.likelihood_value_gradient(point)
        actual_value, actual_gradient = compiled(point)
        self.assertAlmostEqual(actual_value, expected_value, places=12)
        np.testing.assert_allclose(actual_gradient, expected_gradient, atol=1e-12)
        model = build_model(self.tiny)
        self.assertAlmostEqual(
            model.compile_logp()({"log_scales": point}), self.tiny.value_gradient(point)[0], places=10
        )
        np.testing.assert_allclose(
            model.compile_dlogp()({"log_scales": point}), self.tiny.value_gradient(point)[1], atol=1e-10
        )

    def test_time_cutoff_and_event_identity_guards(self):
        with self.assertRaisesRegex(ValueError, "target 2026"):
            validate_rows(self.rows, target_cycle=2020)
        with self.assertRaisesRegex(ValueError, "source cutoff"):
            validate_rows(self.rows, source_cutoff="2019-01-01T00:00:00Z")
        for field, value in (
            ("available_date", "2027-01-01"),
            ("cycle", "2024"),
            ("strict_asof_qualified", "True"),
            ("state", "TX"),
        ):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rows(rows)
        rows = copy.deepcopy(self.rows)
        row = next(row for row in rows if row["cycle"] == "2020")
        row["state"], row["race_id"] = "TX", "2020:TX:senate:regular"
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_actual_compiled_model_matches_density_and_gradient_at_supplied_points(self):
        result = verify_compiled_density(build_model(self.full), self.full, self.original)
        self.assertFalse(result["mcmc_launched"])
        self.assertEqual(len(result["points"]), 2)

    def test_source_snapshot_hash_and_trusted_binding_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "prepared"
            shutil.copytree(self.prepared, copied)
            plan = json.loads((copied / "plan.json").read_text())
            item = plan["snapshots"][0]
            (copied / item["snapshot"]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "source pin mismatch"):
                load_prepared(copied)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "prepared"
            shutil.copytree(self.prepared, copied)
            plan = json.loads((copied / "plan.json").read_text())
            plan["snapshots"][0]["owner"]["sha256"] = "0" * 64
            (copied / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_prepared(copied)
        for field in ("snapshots", "code_pins"):
            with tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "prepared"
                shutil.copytree(self.prepared, copied)
                plan = json.loads((copied / "plan.json").read_text())
                plan[field] = []
                (copied / "plan.json").write_text(json.dumps(plan))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "required .* pin"):
                    load_prepared(copied)

    def test_failure_preserves_completed_chain_and_transition_journal_without_sampling(self):
        import arviz as az

        def synthetic_sample(**kwargs):
            if sampler.call_count == 2:
                raise RuntimeError("injected second-chain failure; no actual sampling")
            kwargs["callback"](
                None,
                SimpleNamespace(
                    draw_idx=0, tuning=True, point={"log_scales": np.log(MEDIANS)}, stats=[{"synthetic_fixture": True}]
                ),
            )
            rng = np.random.default_rng(71)
            logs = rng.normal(size=(1, 1000, 5))
            return az.from_dict(
                posterior=dict(log_scales=logs, standard_deviations=np.exp(logs)),
                sample_stats=dict(
                    diverging=np.zeros((1, 1000), dtype=bool),
                    tree_depth=np.full((1, 1000), 3),
                    energy=rng.normal(size=(1, 1000)),
                ),
            )

        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory) / "synthetic-failure-test"
            with patch("pymc.sample", side_effect=synthetic_sample) as sampler:
                with self.assertRaisesRegex(RuntimeError, "injected second-chain"):
                    run(self.prepared, output)
            self.assertEqual(sampler.call_count, 2)
            self.assertTrue((output / "chain-0-raw-bank.nc").exists())
            self.assertEqual(len((output / "chain-0-transitions.jsonl").read_text().splitlines()), 1)
            self.assertTrue((output / "failure.json").exists())
            self.assertFalse((output / "terminal.json").exists())
            self.assertFalse(json.loads((output / "failure.json").read_text())["forecast_qualified"])

    def test_diagnostics_fail_closed_and_preserve_separate_flags(self):
        import arviz as az

        # Synthetic iid draws exercise diagnostic gates, not the calibration sampler.
        rng = np.random.default_rng(17)
        idata = az.from_dict(
            posterior=dict(log_scales=rng.normal(size=(4, 1000, 5))),
            sample_stats=dict(
                diverging=np.zeros((4, 1000), dtype=bool),
                tree_depth=np.full((4, 1000), 3),
                energy=rng.normal(size=(4, 1000)),
            ),
        )
        self.assertTrue(diagnostics(idata)["passed"])
        idata.sample_stats.diverging.values[0, 0] = True
        idata.sample_stats.tree_depth.values[1, 2] = 10
        result = diagnostics(idata)
        self.assertFalse(result["passed"])
        self.assertEqual(result["divergences"], 1)
        self.assertEqual(result["depth_saturations"], 1)


if __name__ == "__main__":
    unittest.main()
