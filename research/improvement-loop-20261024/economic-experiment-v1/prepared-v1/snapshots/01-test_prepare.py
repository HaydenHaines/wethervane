"""Actual-source compilation/supplied-point tests. No inference is run."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from prepare import (
    FEATURE,
    SAMPLER,
    SEEDS,
    TARGETS,
    VINTAGES,
    compile_fold,
    earlier_outcomes,
    load_features,
    load_macro,
    prior_diagnostics,
    transformed_features,
)
from scipy.stats import norm


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.macro = load_macro()
        cls.folds = {target: compile_fold(target, cls.macro) for target in TARGETS}

    def test_exact_actual_support_and_earlier_only_outcomes(self):
        for target, (module, compiled, frame, scaler, evidence) in self.folds.items():
            self.assertEqual(len(compiled["F0"].race_ids), {2018: 299, 2020: 329, 2022: 357, 2024: 387}[target])
            for arm in compiled.values():
                self.assertTrue((arm.cycles < target).all())
                self.assertEqual(arm.race_ids, compiled["F0"].race_ids)
                self.assertEqual(arm.outcome_rows_sha256, compiled["F0"].outcome_rows_sha256)
            self.assertTrue((earlier_outcomes(target).cycle < target).all())
            self.assertNotIn("actual_dem_share", frame)
            self.assertEqual(scaler["independent_macro_realizations"], (target - 1998) // 2)
            self.assertEqual(evidence["status"], "not_execution_or_count_admission")
            self.assertEqual(module.SamplerPlan(**SAMPLER, seed=SEEDS[target]).chains, 4)

    def test_H0_F0_exact_existing_normalized_densities_at_supplied_points(self):
        from src.forecasting.fundamentals_v1 import FundamentalsPriorsV1, log_density_v1, prepare_training_v1

        for target, (module, compiled, frame, _, _) in self.folds.items():
            for arm, implementation in (("H0", "senate_history_refreshed_v1"), ("F0", "senate_relative_context_v1")):
                original = prepare_training_v1(frame, earlier_outcomes(target), target, implementation)
                mean, _ = FundamentalsPriorsV1().coefficient_parameters(implementation)
                beta = mean + np.linspace(0.01, 0.06, len(mean))
                named = {module.LEGACY_NAMES[name]: value for name, value in zip(original["coefficient_names"], beta)}
                actual = np.array([named[name] for name in compiled[arm].parameter_names])
                z = np.linspace(-0.1, 0.1, len(original["cycles"]))
                self.assertAlmostEqual(
                    module.log_density(compiled[arm], actual, z, 0.2, 0.3),
                    log_density_v1(original, beta, z, 0.2, 0.3),
                    places=9,
                )

    def test_E01_zero_restriction_and_analytic_gradient(self):
        for _, (module, compiled, _, _, _) in self.folds.items():
            baseline, economic = compiled["F0"], compiled["E01"]
            beta0 = np.array([p.mean for p in baseline.recipe.parameters])
            named = dict(zip(baseline.parameter_names, beta0))
            beta1 = np.array([named.get(name, 0) for name in economic.parameter_names])
            z = np.zeros(len(baseline.cycles))
            self.assertAlmostEqual(
                module.log_density(economic, beta1, z, 0.2, 0.3),
                module.log_density(baseline, beta0, z, 0.2, 0.3) + norm.logpdf(0, scale=0.05),
                places=9,
            )
            index = economic.parameter_names.index("economy.headline_cpi")
            axis = np.eye(len(beta1))[index]
            eps = 1e-6
            finite = (
                module.log_density(economic, beta1 + eps * axis, z, 0.2, 0.3)
                - module.log_density(economic, beta1 - eps * axis, z, 0.2, 0.3)
            ) / (2 * eps)
            self.assertAlmostEqual(module.gradient(economic, beta1, z, 0.2, 0.3)[index], finite, places=5)

    def test_unique_cycle_scaling_party_orientation_and_target_isolation(self):
        frame = load_features()
        full, scale = transformed_features(frame, self.macro, 2018)
        one_per_cycle = frame.drop_duplicates("cycle")
        _, small = transformed_features(one_per_cycle, self.macro, 2018)
        self.assertEqual(scale, small, "race counts never weight a national macro realization")
        flipped = frame.copy()
        flipped["president_party_asof"] *= -1
        other, _ = transformed_features(flipped, self.macro, 2018)
        np.testing.assert_allclose(other[FEATURE], -full[FEATURE], atol=1e-14)
        changed = copy.deepcopy(self.macro)
        changed[2018]["value"] += 100
        target_changed, same = transformed_features(frame, changed, 2018)
        self.assertEqual(scale, same)
        np.testing.assert_array_equal(
            full.loc[full.cycle.lt(2018), FEATURE], target_changed.loc[target_changed.cycle.lt(2018), FEATURE]
        )

    def test_midnight_sentinels_and_guards(self):
        self.assertEqual(self.macro[2004]["value"], 2.7)
        self.assertEqual(self.macro[2004]["source"]["reference_month"], "2004-08")
        self.assertEqual(self.macro[2014]["value"], 1.7)
        for target in TARGETS:
            self.assertEqual(self.macro[target]["value"], {2018: 2.3, 2020: 1.4, 2022: 8.2, 2024: 2.4}[target])
        with self.assertRaisesRegex(ValueError, "2026"):
            transformed_features(load_features(), self.macro, 2026)
        bad = load_features()
        bad.loc[bad.cycle.eq(2004), "forecast_cutoff"] = "2004-10-20"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            transformed_features(bad, self.macro, 2018)

    def test_prior_review_declares_extrapolation_and_coordinate_prior_difference(self):
        for target, (module, compiled, frame, scaler, _) in self.folds.items():
            result = prior_diagnostics(target, module, compiled, frame, scaler, self.macro)
            self.assertEqual(result["role"], "prior_predictive_diagnostic_not_fit_or_score")
            primary = result["mean_shift_checks"]["primary"]
            tight = result["mean_shift_checks"]["tighter_sensitivity"]
            self.assertEqual(primary["mean_logit_shift"], 0)
            np.testing.assert_allclose(tight["central95_logit_shift"], np.array(primary["central95_logit_shift"]) / 2)
            if target == 2022:
                self.assertTrue(result["target_outside_earlier_raw_range"])
                self.assertGreater(abs(primary["coordinate"]), 3)
            self.assertIn("F0", result["prior_predictive"])
            self.assertIn("E01", result["prior_predictive"])

    def test_source_bytes_are_required_before_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "economic-cutoff-adapter-v2"
            shutil.copytree(VINTAGES, target)
            (target / "selections.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "source pin mismatch"):
                load_macro(target)


if __name__ == "__main__":
    unittest.main()
