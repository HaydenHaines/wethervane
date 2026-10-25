import unittest
from unittest.mock import patch

import numpy as np
from marginal import GaussianLogitMixture, integral, weighted_interval_score, winner_scores
from scipy.integrate import quad
from scipy.special import expit, log_ndtr, logsumexp, ndtri
from scipy.stats import norm


class MarginalTests(unittest.TestCase):
    def test_zero_width_and_bounded_failure_validation(self):
        for value in (np.inf, np.array([0.0, np.nan])):
            with self.assertRaises(ArithmeticError):
                integral(lambda x: value, 0.5, 0.5)
        zero, receipt = integral(lambda x: np.array([1.0, 2.0]), 0.5, 0.5, vector=True)
        np.testing.assert_array_equal(zero, [0.0, 0.0])
        self.assertEqual(receipt["estimated_error"], 0.0)
        with patch("marginal.quad", side_effect=[(1.0, 0.1), (1.1, 0.1)]) as mocked:
            with self.assertRaises(ArithmeticError):
                integral(lambda x: x, 0, 1)
            self.assertEqual(mocked.call_count, 2)

    def test_tiny_mixture_weight_can_dominate_rare_tail(self):
        model = GaussianLogitMixture([-100.0, 100.0], [1.0, 1.0], [1e-300, 1e300])
        raw = np.log([1e-300, 1e300])
        expected = logsumexp(raw + log_ndtr([100.0, -100.0])) - logsumexp(raw)
        win, lose = model.log_win_lose()
        self.assertEqual(model.weights[0], 0.0)
        self.assertAlmostEqual(lose, expected, delta=1e-10)
        self.assertAlmostEqual(winner_scores(win, lose, False)["negative_log_score"], -expected, delta=1e-10)
        self.assertTrue(np.isfinite(model.logweights).all())

    def test_complementary_tail_roundoff_does_not_reject_valid_score(self):
        model = GaussianLogitMixture([50.0] * 4, [1.0] * 4, [1, 2, 3, 4])
        win, lose = model.log_win_lose()
        self.assertLessEqual(win, 0)
        self.assertLessEqual(lose, 0)
        self.assertAlmostEqual(lose, float(log_ndtr(-50)), delta=1e-10)
        self.assertEqual(winner_scores(win, lose, True)["negative_log_score"], -win)

    def test_normal_transform_quantiles_density_and_extreme_tails(self):
        model = GaussianLogitMixture([1.4], [0.7])
        for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975):
            value, receipt = model.quantile(q)
            self.assertAlmostEqual(value, expit(1.4 + 0.7 * ndtri(q)), delta=5.1e-11)
            self.assertLessEqual(receipt["cdf_lower"], q)
            self.assertGreaterEqual(receipt["cdf_upper"], q)
        x = 0.7
        numerical = (model.cdf(x + 1e-6) - model.cdf(x - 1e-6)) / 2e-6
        self.assertAlmostEqual(model.density(x), numerical, delta=1e-8)
        win, lose = GaussianLogitMixture([50.0], [1.0]).log_win_lose()
        self.assertTrue(np.isfinite(lose))
        self.assertLess(lose, -1200)
        self.assertEqual(winner_scores(win, lose, False)["negative_log_score"], -lose)

    def test_symmetric_means_and_normal_expectation_oracle(self):
        symmetric = GaussianLogitMixture([-2, 2], [1.3, 1.3])
        self.assertAlmostEqual(symmetric.mean_share()[0], 0.5, places=10)
        mixture = GaussianLogitMixture([-1.0, 1.7], [0.5, 1.2], [1, 3])
        oracle = np.array(
            [
                quad(lambda z: expit(mu + sd * z) * norm.pdf(z), -np.inf, np.inf, epsabs=1e-11)[0]
                for mu, sd in zip(mixture.mean, mixture.sd)
            ]
        )
        means, _ = mixture.component_mean_share()
        np.testing.assert_allclose(means, oracle, atol=1e-9, rtol=0)
        self.assertAlmostEqual(mixture.mean_share()[0], mixture.weights @ oracle, delta=1e-9)
        self.assertGreater(abs(means[1] - expit(mixture.mean[1])), 0.01)

    def test_crps_integral_and_mixture_weight_derivative(self):
        y = 0.37
        # Uniform-share CDF oracle has CRPS y^2-y+1/3.
        uniform = integral(lambda x: x * x, 0, y)[0] + integral(lambda x: (1 - x) ** 2, y, 1)[0]
        self.assertAlmostEqual(uniform, y * y - y + 1 / 3, places=13)
        means, scales, weights = [-1.2, 0.8], [0.5, 0.7], np.array([0.3, 0.7])
        mixture = GaussianLogitMixture(means, scales, weights)
        influence, _ = mixture.crps_influence(y)
        self.assertAlmostEqual(weights @ influence, 0.0, places=11)
        epsilon = 1e-5
        up = GaussianLogitMixture(means, scales, weights + [epsilon, -epsilon]).crps(y)[0]
        down = GaussianLogitMixture(means, scales, weights + [-epsilon, epsilon]).crps(y)[0]
        self.assertAlmostEqual((up - down) / (2 * epsilon), influence[0] - influence[1], delta=1e-7)
        self.assertAlmostEqual(mixture.crps(0)[0], integral(lambda x: mixture.sf(x) ** 2, 0, 1)[0], places=12)

    def test_wis_matches_seven_pinball_losses(self):
        quantiles = {q: q for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)}
        for truth in (0, 0.1, 0.5, 0.9, 1):
            pinball = sum((truth - value) * (q - int(truth < value)) for q, value in quantiles.items()) / 3.5
            self.assertAlmostEqual(weighted_interval_score(truth, quantiles), pinball, places=14)

    def test_inputs_are_owned_immutable_and_invalid_values_rejected(self):
        values = np.array([1.0, 2.0])
        model = GaussianLogitMixture(values, [1, 1])
        values[:] = 20
        np.testing.assert_array_equal(model.mean, [1, 2])
        with self.assertRaises(ValueError):
            model.mean.setflags(write=True)
        for means, scales, weights in (([], [], None), ([0], [0], None), ([np.nan], [1], None), ([0], [1], [-1])):
            with self.assertRaises(ValueError):
                GaussianLogitMixture(means, scales, weights)
        with self.assertRaises(ValueError):
            model.cdf(np.nan)
        with self.assertRaises(ValueError):
            winner_scores(-1, -1, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
