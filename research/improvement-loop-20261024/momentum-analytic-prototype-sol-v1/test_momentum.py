"""Deterministic synthetic mathematical oracles; no optimizer or sampler."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import momentum as m
import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

HERE = Path(__file__).resolve().parent
SCALES = dict(zip(m.COMPONENTS, [0.12, 0.04, 0.06, 0.03, 0.05, 0.07], strict=True))
MEDIANS = np.array([0.1, 0.04, 0.04, 0.03, 0.05, 0.05])


def fixture():
    return m.PollCoordinatesV1(
        tuple(f"synthetic-{i}" for i in range(6)),
        ("a", "a", "a", "b", "b", "a"),
        ("2020",) * 5 + ("2022",),
        ("x", "x", "y", "x", "z", "x"),
        [60.0, 30.0, 0.0, 30.0, 7.0, 10.0],
    )


def independent_latent_covariance(coordinates, scales):
    """Independent latent intercepts/increments/slopes; no producer kernels."""
    n = len(coordinates.observation_ids)
    columns = [np.eye(n)[:, i] * scales["independent"] for i in range(n)]
    for kind, key in (
        ("cycle", list(coordinates.cycles)),
        ("race", list(zip(coordinates.cycles, coordinates.race_ids, strict=True))),
        ("house", list(zip(coordinates.cycles, coordinates.pollsters, strict=True))),
    ):
        for group in sorted(set(key)):
            columns.append(np.array([float(v == group) for v in key]) * scales[kind])
    groups = list(zip(coordinates.cycles, coordinates.race_ids, strict=True))
    for group in sorted(set(groups)):
        selected = [i for i, g in enumerate(groups) if g == group]
        times = sorted(set(coordinates.days_to_election[selected]))
        previous = 0.0
        for age in times:
            column = np.zeros(n)
            for i in selected:
                if coordinates.days_to_election[i] >= age:
                    column[i] = math.sqrt((age - previous) / 30) * scales["movement"]
            columns.append(column)
            previous = age
        slope = np.zeros(n)
        for i in selected:
            slope[i] = coordinates.days_to_election[i] / 30 * scales["persistent_linear"]
        columns.append(slope)
    matrix = np.column_stack(columns)
    return matrix @ matrix.T


def independent_nll(residual, covariance):
    sign, determinant = np.linalg.slogdet(covariance)
    if sign != 1:
        raise ValueError("oracle requires SPD")
    return 0.5 * (len(residual) * np.log(2 * np.pi) + determinant + residual @ np.linalg.solve(covariance, residual))


class MathematicalTests(unittest.TestCase):
    def test_sources_verified_before_scientific_imports(self):
        m.source_guard.verify()
        for corrupt in ("source-bindings.json", "snapshots/01-poll_covariance_v1.py"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory(dir=HERE) as directory:
                target = Path(directory)
                for name in ("source_guard.py", "momentum.py", "source-bindings.json"):
                    shutil.copyfile(HERE / name, target / name)
                shutil.copytree(HERE / "snapshots", target / "snapshots")
                path = target / corrupt
                path.write_bytes(path.read_bytes() + b"changed")
                code = (
                    "try:\n import momentum\nexcept ValueError:\n import sys\n"
                    " assert 'numpy' not in sys.modules\n assert 'scipy' not in sys.modules\n"
                    " print('blocked before scientific imports')\nelse:\n"
                    " raise AssertionError('accepted changed source')\n"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=target,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("blocked before scientific imports", result.stdout)

    def test_all_five_exact_and_zero_slope_incumbent_identity(self):
        coordinates = fixture()
        old = coordinates.incumbent_coordinates()
        np.testing.assert_array_equal(coordinates.kernels()[:5], old.kernels())
        scales = {**SCALES, "persistent_linear": 0.0}
        augmented = m.error_covariance(coordinates, scales)
        incumbent = m.INCUMBENT.error_covariance(old, {k: scales[k] for k in m.INCUMBENT.COMPONENTS})
        np.testing.assert_array_equal(augmented, incumbent)
        shares = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.35])
        ids = ("a", "b")
        new_factor = m.build_poll_likelihood(ids, coordinates, shares, scales)
        old_factor = m.INCUMBENT.build_poll_likelihood(ids, old, shares, scales)
        for name in ("values", "race_indices", "covariance"):
            np.testing.assert_array_equal(getattr(new_factor, name), getattr(old_factor, name))
        residual = np.array([0.1, -0.2, 0.05, 0.04, -0.03, 0.08])
        new_nll = independent_nll(residual, new_factor.covariance)
        old_value, _ = m.INCUMBENT.gaussian_error_objective(
            np.log(MEDIANS[:5]), [(residual, old.kernels())], prior_medians=MEDIANS[:5], prior_log_sd=0.7
        )
        # Identity with the incumbent normalized likelihood, excluding zero penalty.
        reference = m.INCUMBENT.error_covariance(old, dict(zip(m.INCUMBENT.COMPONENTS, MEDIANS[:5], strict=True)))
        self.assertAlmostEqual(old_value, independent_nll(residual, reference), places=12)
        self.assertAlmostEqual(new_nll, independent_nll(residual, old_factor.covariance), places=14)

    def test_latent_gaussian_covariance_oracle_and_structure(self):
        coordinates = fixture()
        expected = independent_latent_covariance(coordinates, SCALES)
        np.testing.assert_allclose(m.error_covariance(coordinates, SCALES), expected, atol=1e-17, rtol=2e-15)
        slope = coordinates.kernels()[5]
        self.assertEqual(slope[0, 1], 2.0)
        self.assertEqual(slope[0, 2], 0.0)
        self.assertEqual(slope[0, 3], 0.0)
        self.assertEqual(slope[0, 5], 0.0)
        for kernel in coordinates.kernels():
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel).min(), -1e-13)
        self.assertGreater(np.linalg.eigvalsh(expected).min(), 0)
        np.linalg.cholesky(expected)

    def test_independent_latent_slope_quadrature_integration(self):
        coordinates = m.PollCoordinatesV1(
            ("q0", "q1", "q2", "q3"), ("a", "a", "b", "b"), ("2020",) * 4, ("x", "y", "x", "y"), [40.0, 5.0, 20.0, 0.0]
        )
        y = np.array([0.1, -0.07, 0.04, 0.06])
        base = independent_latent_covariance(coordinates, {**SCALES, "persistent_linear": 0.0})
        nodes, weights = roots_hermitenorm(60)
        first, second = np.meshgrid(nodes, nodes, indexing="ij")
        slope_design = np.array([[40 / 30, 0], [5 / 30, 0], [0, 20 / 30], [0, 0]])
        slopes = np.stack((first.ravel(), second.ravel()), axis=1) * SCALES["persistent_linear"]
        residual = y - slopes @ slope_design.T
        sign, det = np.linalg.slogdet(base)
        self.assertEqual(sign, 1)
        loglik = -0.5 * (
            4 * np.log(2 * np.pi) + det + np.einsum("bi,ij,bj->b", residual, np.linalg.inv(base), residual)
        )
        logweight = (np.log(weights[:, None]) + np.log(weights[None, :])).ravel() - np.log(2 * np.pi)
        integrated = -float(logsumexp(logweight + loglik))
        actual, _ = m.gaussian_error_objective(
            np.log(list(SCALES.values())),
            [m.ResidualBlockV1(coordinates, y)],
            prior_medians=list(SCALES.values()),
            prior_log_sd=0.7,
        )
        self.assertAlmostEqual(actual, integrated, places=11)

    def test_all_six_analytic_gradients_and_objective_constants(self):
        coordinates = fixture()
        y = np.array([0.14, -0.07, 0.03, 0.11, -0.1, 0.05])
        block = m.ResidualBlockV1(coordinates, y)
        for offset in (np.zeros(6), np.array([0.2, -0.1, 0.15, -0.25, 0.3, -0.2])):
            point = np.log(MEDIANS) + offset
            value, gradient = m.gaussian_error_objective(point, [block], prior_medians=MEDIANS, prior_log_sd=0.7)
            reference = independent_nll(
                y, independent_latent_covariance(coordinates, dict(zip(m.COMPONENTS, np.exp(point), strict=True)))
            )
            reference += 0.5 * np.sum((offset / 0.7) ** 2)
            self.assertAlmostEqual(value, reference, places=12)
            for k in range(6):
                step = np.eye(6)[k] * 1e-6
                plus = m.gaussian_error_objective(point + step, [block], prior_medians=MEDIANS, prior_log_sd=0.7)[0]
                minus = m.gaussian_error_objective(point - step, [block], prior_medians=MEDIANS, prior_log_sd=0.7)[0]
                self.assertAlmostEqual(gradient[k], (plus - minus) / 2e-6, places=8)

    def test_permutation_and_independent_cycle_partition_identity(self):
        coordinates = fixture()
        y = np.arange(6) * 0.03 - 0.1
        order = np.array([5, 2, 0, 4, 1, 3])

        def selected(indices):
            return m.PollCoordinatesV1(
                *(
                    tuple(getattr(coordinates, name)[i] for i in indices)
                    for name in ("observation_ids", "race_ids", "cycles", "pollsters")
                ),
                coordinates.days_to_election[indices],
            )

        permuted = selected(order)
        np.testing.assert_allclose(
            m.error_covariance(permuted, SCALES),
            m.error_covariance(coordinates, SCALES)[np.ix_(order, order)],
            atol=0,
            rtol=0,
        )
        point = np.log(MEDIANS)
        before = m.gaussian_error_objective(
            point, [m.ResidualBlockV1(coordinates, y)], prior_medians=MEDIANS, prior_log_sd=0.7
        )
        after = m.gaussian_error_objective(
            point, [m.ResidualBlockV1(permuted, y[order])], prior_medians=MEDIANS, prior_log_sd=0.7
        )
        split = [m.ResidualBlockV1(selected(indices), y[indices]) for indices in (np.arange(5), np.array([5]))]
        partition = m.gaussian_error_objective(point, split, prior_medians=MEDIANS, prior_log_sd=0.7)
        self.assertAlmostEqual(before[0], after[0], places=12)
        self.assertAlmostEqual(before[0], partition[0], places=12)
        np.testing.assert_allclose(before[1], after[1], atol=1e-13)
        np.testing.assert_allclose(before[1], partition[1], atol=1e-13)

    def test_synthetic_conditional_symmetry_and_shrinkage(self):
        loose = m.synthetic_conditional_example(0.05)
        tight = m.synthetic_conditional_example(0.005)
        self.assertTrue(loose["synthetic"])
        self.assertFalse(loose["source_data_used"])
        self.assertLess(loose["posterior_mean_slope"], 0)
        self.assertLess(abs(tight["posterior_mean_slope"]), abs(loose["posterior_mean_slope"]))
        np.testing.assert_allclose(
            loose["mirrored_posterior_mean"],
            -np.array([loose["posterior_mean_theta"], loose["posterior_mean_slope"]]),
            atol=0,
            rtol=0,
        )
        np.testing.assert_array_equal(loose["flat_data_posterior_mean"], [0.0, 0.0])
        self.assertTrue((np.linalg.eigvalsh(loose["posterior_covariance"]) > 0).all())

    def test_invalid_inputs_and_owned_immutable_arrays(self):
        coordinates = fixture()
        days = np.array([2.0, 1.0])
        copy = m.PollCoordinatesV1(("i", "j"), ("r", "r"), ("c", "c"), ("h", "h"), days)
        days[0] = 100
        self.assertEqual(copy.days_to_election[0], 2)
        with self.assertRaises(ValueError):
            copy.days_to_election.setflags(write=True)
        for ages in ([-1.0, 2.0], [np.nan, 1.0], [np.inf, 2.0], [1.0], np.array([1 + 2j, 2])):
            with self.subTest(ages=str(ages)), self.assertRaises(ValueError):
                m.PollCoordinatesV1(("i", "j"), ("r", "r"), ("c", "c"), ("h", "h"), ages)
        for ids in (("i", "i"), ("", "j")):
            with self.assertRaises(ValueError):
                m.PollCoordinatesV1(ids, ("r", "r"), ("c", "c"), ("h", "h"), [1.0, 2.0])
        for scales in (
            {**SCALES, "independent": 0.0},
            {**SCALES, "movement": -0.1},
            {**SCALES, "extra": 1},
            {**SCALES, "persistent_linear": np.nan},
            {k: v for k, v in SCALES.items() if k != "race"},
        ):
            with self.assertRaises(ValueError):
                m.error_covariance(coordinates, scales)
        block = m.ResidualBlockV1(coordinates, np.zeros(6))
        for point, medians, sd in (
            (np.zeros(5), MEDIANS, 0.7),
            (np.full(6, 1000.0), MEDIANS, 0.7),
            (np.full(6, -1000.0), MEDIANS, 0.7),
            (np.zeros(6), np.zeros(6), 0.7),
            (np.zeros(6), MEDIANS, 0),
            (np.zeros(6), MEDIANS, np.nan),
        ):
            with self.assertRaises(ValueError):
                m.gaussian_error_objective(point, [block], prior_medians=medians, prior_log_sd=sd)
        with self.assertRaises(ValueError):
            m.gaussian_error_objective(np.log(MEDIANS), [], prior_medians=MEDIANS, prior_log_sd=0.7)
        with self.assertRaises(ValueError):
            m.gaussian_error_objective(np.log(MEDIANS), [block, block], prior_medians=MEDIANS, prior_log_sd=0.7)
        with self.assertRaises(ValueError):
            m.ResidualBlockV1(coordinates, np.zeros(5))
        with self.assertRaises(ValueError):
            m.build_poll_likelihood(("a",), coordinates, np.full(6, 0.5), SCALES)
        with self.assertRaises(ValueError):
            m.build_poll_likelihood(("a", "b"), coordinates, np.zeros(6), SCALES)

    def test_no_optimization_or_sampling_is_called(self):
        with (
            patch("scipy.optimize.minimize", side_effect=AssertionError("no optimization")),
            patch("numpy.random.default_rng", side_effect=AssertionError("no sampling")),
        ):
            m.gaussian_error_objective(
                np.log(MEDIANS), [m.ResidualBlockV1(fixture(), np.zeros(6))], prior_medians=MEDIANS, prior_log_sd=0.7
            )
            m.synthetic_conditional_example(0.05)


if __name__ == "__main__":
    unittest.main()
