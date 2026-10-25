"""Small independent numerical checks; no fits or actual posterior bank."""

import unittest
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose
from poll_scale_mixture import GaussianScaleMixture
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm


class ScaleMixtureTests(unittest.TestCase):
    def setUp(self):
        self.covariances = np.array([[[1.0, 0.3], [0.3, 2.0]], [[4.0, -0.4], [-0.4, 3.0]]])
        self.weights = np.array([0.3, 0.7])
        self.model = GaussianScaleMixture(self.covariances, self.weights)
        self.y = np.array([0.8, -1.1])
        self.mean = np.array([-0.2, 0.3])

    def test_independent_scipy_joint_density_responsibilities_and_gradient(self):
        densities = np.array(
            [multivariate_normal.pdf(self.y, self.mean, covariance) for covariance in self.covariances]
        )
        total = self.weights @ densities
        self.assertAlmostEqual(self.model.logpdf(self.y, self.mean), np.log(total), places=13)
        assert_allclose(self.model.responsibilities(self.y, self.mean), self.weights * densities / total, atol=1e-14)
        step = 1e-5
        gradient = []
        for direction in np.eye(2):
            plus = np.log(
                sum(
                    weight * multivariate_normal.pdf(self.y, self.mean + step * direction, covariance)
                    for weight, covariance in zip(self.weights, self.covariances)
                )
            )
            minus = np.log(
                sum(
                    weight * multivariate_normal.pdf(self.y, self.mean - step * direction, covariance)
                    for weight, covariance in zip(self.weights, self.covariances)
                )
            )
            gradient.append((plus - minus) / (2 * step))
        assert_allclose(self.model.gradient_mean(self.y, self.mean), gradient, rtol=1e-9, atol=1e-10)

    def test_point_mass_permutation_and_splitting(self):
        fixed = GaussianScaleMixture(self.covariances, [1, 0])
        self.assertAlmostEqual(
            fixed.logpdf(self.y, self.mean), multivariate_normal.logpdf(self.y, self.mean, self.covariances[0])
        )
        assert_allclose(fixed.responsibilities(self.y, self.mean), [1, 0])
        reversed_model = GaussianScaleMixture(self.covariances[::-1], self.weights[::-1])
        split = GaussianScaleMixture(self.covariances[[0, 1, 1]], [0.3, 0.2, 0.5])
        for model in (reversed_model, split):
            self.assertAlmostEqual(model.logpdf(self.y, self.mean), self.model.logpdf(self.y, self.mean))
            assert_allclose(model.gradient_mean(self.y, self.mean), self.model.gradient_mean(self.y, self.mean))

    def test_normalized_one_dimensional_quadrature_and_tails(self):
        model = GaussianScaleMixture([[[0.25]], [[9.0]]], [0.4, 0.6])
        integral, error = quad(lambda y: np.exp(model.logpdf([y], [0.7])), -np.inf, np.inf, epsabs=1e-10)
        self.assertAlmostEqual(integral, 1.0, places=10)
        self.assertLess(error, 1e-8)
        self.assertTrue(np.isfinite(model.logpdf([1000.0], [0.7])))
        self.assertAlmostEqual(model.responsibilities([1000.0], [0.7]).sum(), 1.0)

    def test_joint_index_is_not_independent_per_poll_mixing(self):
        model = GaussianScaleMixture([np.eye(2), 9 * np.eye(2)], [0.5, 0.5])
        y = [0.0, 5.0]
        product_of_marginals = np.prod([0.5 * norm.pdf(value, scale=1) + 0.5 * norm.pdf(value, scale=3) for value in y])
        self.assertGreater(abs(np.exp(model.logpdf(y, [0, 0])) - product_of_marginals), 1e-4)

    def test_two_level_exact_counterexample_marginal_is_not_calibration_cut(self):
        # Scale s has calibration weights 1/2; theta has prior mass 1/2 at 0,2.
        # One global Bayesian normalizer updates scale mass. A cut would instead
        # normalize theta separately for each s, preserving original scale mass.
        likelihood = np.array([[norm.pdf(0, loc=theta, scale=scale) for theta in (0, 2)] for scale in (1, 3)])
        joint = 0.25 * likelihood
        joint /= joint.sum()
        cut_joint = 0.5 * likelihood / likelihood.sum(axis=1, keepdims=True)
        assert_allclose(cut_joint.sum(axis=1), [0.5, 0.5])
        self.assertGreater(abs(joint.sum(axis=1)[0] - 0.5), 0.1)
        model = GaussianScaleMixture([[[1]], [[9]]], [0.5, 0.5])
        for column, theta in enumerate((0, 2)):
            assert_allclose(model.responsibilities([0], [theta]), joint[:, column] / joint[:, column].sum())
        marginal_theta = np.array([np.exp(model.logpdf([0], [theta])) for theta in (0, 2)])
        assert_allclose(marginal_theta / marginal_theta.sum(), joint.sum(axis=0))
        self.assertGreater(abs(joint.sum(axis=0)[0] - cut_joint.sum(axis=0)[0]), 0.01)

    def test_invalid_inputs_and_copied_immutable_storage(self):
        self.covariances[:] = 77
        self.weights[:] = 0
        self.assertEqual(self.model.covariances[0, 0, 0], 1)
        assert_allclose(self.model.weights, [0.3, 0.7])
        for array in (self.model.weights, self.model.covariances):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            self.model.weights = np.array([1, 0])
        for covariances, weights in [
            ([], []),
            ([[1]], [1]),
            ([[[1, 0]]], [1]),
            ([[[np.nan]]], [1]),
            ([[[0]]], [1]),
            ([[[-1]]], [1]),
            ([[[1, 0.5], [0, 1]]], [1]),
            ([[[1]]], [0.9]),
            ([[[1]], [[2]]], [-0.1, 1.1]),
            ([[[1]]], [np.nan]),
            ([[[1]]], [[1]]),
        ]:
            with self.subTest(covariances=covariances, weights=weights):
                with self.assertRaises(ValueError):
                    GaussianScaleMixture(covariances, weights)
        for y, mean in [([1], [0, 0]), ([np.inf, 0], [0, 0]), ([0, 0], [np.nan, 0]), ([[0, 0]], [0, 0])]:
            with self.assertRaises(ValueError):
                self.model.logpdf(y, mean)


if __name__ == "__main__":
    unittest.main()
