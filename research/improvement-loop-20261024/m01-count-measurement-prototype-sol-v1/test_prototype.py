"""Deterministic supplied-state checks. No fits or historical label loads."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
from numpy.testing import assert_allclose
import scipy
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal, norm
import pytensor
import pytensor.tensor as pt

from prototype import (CountPairPort, SharedVectorMixture, load_frozen_current_case,
                       protected_parent_joint, stable_logsumexp)

HERE = Path(__file__).resolve().parent
CONSUMER = HERE.parent / "calibration-consumer-v1"
V2 = HERE.parent / "m01-count-measurement-prototype-spec-v2/specification.json"
DETAILS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_mean(model, y):
    mean = pt.dvector("mean")
    value = model.symbolic_logpdf(y, mean)
    return pytensor.function([mean], [value, pt.grad(value, mean)], mode="FAST_COMPILE")


def finite_difference(function, point, step):
    point = np.asarray(point, dtype=float)
    result = []
    for i in range(len(point)):
        offset = np.zeros(len(point)); offset[i] = step
        result.append((function(point + offset) - function(point - offset)) / (2 * step))
    return np.asarray(result)


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_frozen_current_case(CONSUMER)
        cls.model, cls.y = cls.case["mixture"], cls.case["y"]
        cls.autodiff = compile_mean(cls.model, cls.y)
        cls.before = {str(CONSUMER / r["file"]): digest(CONSUMER / r["file"])
                      for r in cls.case["manifest"]["files"]}
        cls.before[str(CONSUMER / "consumer.py")] = digest(CONSUMER / "consumer.py")

    def test_01_single_component_exact_incumbent(self):
        point = SharedVectorMixture([self.case["fixed"]], [1.0])
        fn = compile_mean(point, self.y)
        maximum = 0.0
        for mean in [self.y, np.zeros(6), np.full(6, .1)]:
            expected = multivariate_normal.logpdf(self.y, mean, self.case["fixed"])
            gradient = np.linalg.solve(self.case["fixed"], self.y - mean)
            numeric = point.evaluate(self.y, mean)
            actual, automatic = fn(mean)
            assert_allclose([actual, numeric[0]], expected, rtol=1e-12, atol=1e-10)
            assert_allclose(automatic, gradient, rtol=1e-9, atol=1e-9)
            assert_allclose(numeric[1], gradient, rtol=1e-9, atol=1e-9)
            maximum = max(maximum, abs(float(actual) - expected))
        saved = np.array(self.case["component"]["covariance"])
        assert_allclose(self.case["fixed"], saved, atol=1e-15, rtol=0)
        DETAILS["singlePointMaximumLogDensityError"] = maximum

    def test_02_all_joint_bank_rows_independent_reference_and_autodiff(self):
        means = [self.y, np.zeros(6), np.full(6, .1),
                 self.y + .25 * np.array([1, -1, 1, -1, 1, -1]), np.full(6, -.5), np.full(6, .5)]
        maximum_log, maximum_gradient = 0.0, 0.0
        rows = []
        for mean in means:
            individual = np.array([multivariate_normal.logpdf(self.y, mean, c)
                                   for c in self.model.covariances])
            expected = logsumexp(individual + np.log(self.model.weights))
            responsibility = np.exp(individual + np.log(self.model.weights) - expected)
            gradient = sum(p * np.linalg.solve(c, self.y - mean)
                           for p, c in zip(responsibility, self.model.covariances))
            numeric, analytic, responsibilities = self.model.evaluate(self.y, mean)
            actual, automatic = self.autodiff(mean)
            assert_allclose([actual, numeric, self.case["saved_reference"].logpdf(self.y, mean)], expected,
                            rtol=1e-12, atol=1e-10)
            assert_allclose(automatic, gradient, rtol=1e-9, atol=1e-9)
            assert_allclose(analytic, gradient, rtol=1e-9, atol=1e-9)
            assert_allclose(responsibilities.sum(), 1, atol=1e-12, rtol=0)
            maximum_log = max(maximum_log, abs(float(actual) - expected))
            maximum_gradient = max(maximum_gradient, float(np.max(np.abs(automatic - gradient))))
            rows.append(dict(mean=mean.tolist(), logDensity=float(actual), gradient=automatic.tolist(),
                             largestConditionalResponsibility=float(responsibilities.max())))
        self.assertEqual(self.case["bank"].joint_rows.shape, (4000, 5))
        assert_allclose(self.model.weights, np.full(4000, 1/4000), rtol=0, atol=0)
        DETAILS.update(fullBankMaximumLogDensityError=maximum_log,
                       fullBankMaximumGradientError=maximum_gradient, suppliedMeans=rows)

    def test_03_native_same_pair_broadcast_chain_rule_and_finite_differences(self):
        port = CountPairPort(("first", "second", "minor"), tuple(f"p{i}" for i in range(6)),
                             (("first", "second"),) * 6)
        logs = pt.dvector("candidate_log_rates")
        means = port.symbolic_mean_from_log_rates(logs)
        value = self.model.symbolic_logpdf(self.y, means)
        fn = pytensor.function([logs], [value, pt.grad(value, logs), means], mode="FAST_COMPILE")
        point = np.array([.2, -.1, -1.0])
        logp, automatic, observed_means = fn(point)
        assert_allclose(observed_means, np.full(6, .3), atol=1e-14)
        _, coordinate_gradient, _ = self.model.evaluate(self.y, observed_means)
        expected = np.array([coordinate_gradient.sum(), -coordinate_gradient.sum(), 0])
        assert_allclose(automatic, expected, rtol=1e-9, atol=1e-9)
        errors = []
        for step in [1e-4, 3e-5, 1e-5]:
            finite = finite_difference(lambda x: float(fn(x)[0]), point, step)
            assert_allclose(finite, automatic, rtol=2e-6, atol=2e-6)
            errors.append(float(np.max(np.abs(finite - automatic))))
        shifted = fn(point + 2.0)
        assert_allclose(shifted[0], logp, rtol=1e-12, atol=1e-10)
        assert_allclose(automatic.sum(), 0, atol=1e-12)
        assert_allclose(port.mean_from_rates(np.exp(point)), observed_means, atol=1e-14)
        with self.assertRaises(ValueError):
            port.mean_from_rates([1, 0, 1])
        with self.assertRaises(AssertionError):
            fn([.2, -.1, -1.0, .5])
        DETAILS["countPortFiniteDifferenceMaximumErrors"] = errors

    def test_04_calibration_input_conditional_and_integrated_weights_differ(self):
        likelihood = np.array([[norm.pdf(0, loc=m, scale=s) for m in [0, 2]] for s in [1, 3]])
        joint = .25 * likelihood; joint /= joint.sum()
        cut = .5 * likelihood / likelihood.sum(axis=1, keepdims=True)
        assert_allclose(cut.sum(axis=1), [.5, .5])
        self.assertGreater(abs(joint.sum(axis=1)[0] - .5), .1)
        model = SharedVectorMixture([[[1]], [[9]]], [.5, .5])
        marginal = []
        for i, mean in enumerate([0, 2]):
            value, _, responsibility = model.evaluate([0], [mean])
            assert_allclose(responsibility, joint[:, i] / joint[:, i].sum(), atol=1e-12)
            marginal.append(np.exp(value))
        assert_allclose(np.array(marginal) / sum(marginal), joint.sum(axis=0))
        self.assertGreater(abs(joint.sum(axis=0)[0] - cut.sum(axis=0)[0]), .01)
        DETAILS["weightSemantics"] = dict(inputWeights=[.5, .5], integratedPosteriorWeights=joint.sum(axis=1).tolist(),
                                          protectedCalibrationCutWeights=cut.sum(axis=1).tolist())

    def test_05_protected_population_mass_is_fixed_despite_unequal_normalizers(self):
        kernel = np.array([[1, 9], [2, 1]], dtype=float)
        result = protected_parent_joint(np.log(kernel), [.3, .7])
        assert_allclose(result, [[.03, .27], [7/15, 7/30]], rtol=0, atol=1e-14)
        assert_allclose(result.sum(axis=1), [.3, .7], rtol=0, atol=1e-14)
        wrong = np.array([.3, .7])[:, None] * kernel; wrong /= wrong.sum()
        self.assertGreater(np.max(np.abs(wrong.sum(axis=1) - [.3, .7])), .1)
        eight = protected_parent_joint(np.log(np.arange(1, 25).reshape(8, 3)), np.full(8, .125))
        assert_allclose(eight.sum(axis=1), np.full(8, .125), rtol=0, atol=1e-14)
        with self.assertRaises(ValueError):
            protected_parent_joint([[float("-inf")]], [1])

    def test_06_shared_vector_density_is_not_independent_poll_mixtures(self):
        model = SharedVectorMixture([np.eye(2), 9*np.eye(2)], [.5, .5])
        y = np.array([.2, 2.0])
        correct = .5*multivariate_normal.pdf(y, cov=np.eye(2)) + .5*multivariate_normal.pdf(y, cov=9*np.eye(2))
        wrong = np.prod([.5*norm.pdf(v) + .5*norm.pdf(v, scale=3) for v in y])
        assert_allclose(np.exp(model.evaluate(y, [0, 0])[0]), correct, rtol=1e-12)
        self.assertGreater(abs(correct - wrong), 1e-4)

    def test_07_row_permutation_weight_split_zero_and_immutability(self):
        cov = np.array([np.eye(2), [[2, .3], [.3, 3]]])
        model = SharedVectorMixture(cov, [.3, .7]); expected = model.evaluate([.1, -.3], [.5, 0])
        variants = [SharedVectorMixture(cov[::-1], [.7, .3]),
                    SharedVectorMixture([cov[0], cov[0], cov[1]], [.1, .2, .7]),
                    SharedVectorMixture([cov[0], cov[1], np.eye(2)], [.3, .7, 0])]
        for variant in variants:
            observed = variant.evaluate([.1, -.3], [.5, 0])
            assert_allclose(observed[0], expected[0], atol=1e-10, rtol=1e-12)
            assert_allclose(observed[1], expected[1], atol=1e-9, rtol=1e-9)
            fn = compile_mean(variant, [.1, -.3]); value, gradient = fn([.5, 0])
            assert_allclose(value, expected[0], atol=1e-10, rtol=1e-12)
            assert_allclose(gradient, expected[1], atol=1e-9, rtol=1e-9)
        cov[:] = 20
        assert_allclose(model.evaluate([.1, -.3], [.5, 0])[0], expected[0])
        for value in [model.covariances, model.precisions, model.weights, model.log_weights]:
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_08_stable_extreme_logsumexp_without_clipping(self):
        for terms in [[-10000, -10001, -11000], [1000, 999, -1000], [0, float("-inf")]]:
            with localcontext() as ctx:
                ctx.prec = 60
                maximum = max(t for t in terms if np.isfinite(t))
                total = sum((Decimal(str(t-maximum)).exp() for t in terms if np.isfinite(t)), Decimal(0))
                expected = float(Decimal(maximum) + total.ln())
            self.assertAlmostEqual(stable_logsumexp(terms), expected, delta=1e-10)
            for shift in [-10000, 10000]:
                shifted = np.asarray(terms) + shift
                assert_allclose(stable_logsumexp(shifted) - shift, expected, atol=1e-10, rtol=1e-12)
        for invalid in [[float("nan")], [float("inf")], [float("-inf")], []]:
            with self.assertRaises(ValueError):
                stable_logsumexp(invalid)
        model = SharedVectorMixture([[[1]], [[4]]], [.5, .5])
        numeric = model.evaluate([1000], [0]); actual, gradient = compile_mean(model, [1000])([0])
        self.assertTrue(np.isfinite(actual))
        assert_allclose(actual, numeric[0], rtol=1e-12, atol=1e-10)
        assert_allclose(gradient, numeric[1], rtol=1e-9, atol=1e-9)
        extreme = compile_mean(SharedVectorMixture([[[1]]], [1]), [0])
        for invalid in [[1e200], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError): extreme(invalid)
        # A zero-weight term must not hide an overflowing quadratic/derivative.
        dead = SharedVectorMixture([[[1e-300]], [[1]]], [0, 1])
        with self.assertRaises(ValueError): dead.evaluate([1e100], [0])
        with self.assertRaises(ValueError): compile_mean(dead, [1e100])([0])

    def test_09_normalization_covariance_support_and_no_jitter(self):
        model = SharedVectorMixture([[[1]], [[9]]], [.3, .7])
        integral, error = quad(lambda y: np.exp(model.evaluate([y], [.2])[0]), -np.inf, np.inf, epsabs=1e-11)
        assert_allclose(integral, 1, atol=1e-9, rtol=0)
        DETAILS["oneDimensionalNormalization"] = dict(integral=integral, quadratureError=error)
        for bad in [[[[1, 1], [0, 1]]], [[[1, 2], [2, 1]]], [[[float("nan")]]]]:
            with self.assertRaises(ValueError): SharedVectorMixture(bad, [1])
        for weights in [[-.1, 1.1], [.4, .4], [float("nan"), 1]]:
            with self.assertRaises(ValueError): SharedVectorMixture([[[1]], [[2]]], weights)
        with self.assertRaises(ValueError): self.model.evaluate([1, 2], np.zeros(6))
        with self.assertRaises(ValueError): self.model.evaluate(self.y, np.full(6, np.inf))
        with self.assertRaises(AssertionError): self.autodiff([0.0])

    def test_10_pinned_source_scope_and_alias_guards(self):
        for kwargs in [dict(target_cycle=2018), dict(target_cycle=2020), dict(target_cycle=2022),
                       dict(target_cycle=2024), dict(cutoff="2026-10-19T00:00:00Z"), dict(target="calibration_cut")]:
            with self.assertRaises(ValueError): load_frozen_current_case(CONSUMER, **kwargs)
        for mutation in ["consumer.py", "input-manifest.json", "inputs/posterior-bank.npz", "inputs/independent-review.json",
                         "inputs/poll-component.json", "inputs/target-coordinates.json", "snapshot-source/src/forecasting/poll_covariance_v1.py"]:
            with tempfile.TemporaryDirectory(prefix="m01-metadata-", dir=os.environ["M01_OUTPUT"]) as directory:
                root = Path(directory)
                for entry in ["input-manifest.json", "consumer.py"] + [r["file"] for r in self.case["manifest"]["files"]]:
                    destination = root / entry; destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(CONSUMER / entry, destination)
                (root / mutation).write_bytes((root / mutation).read_bytes() + b" ")
                with self.assertRaises(ValueError): load_frozen_current_case(root)
        # Rebinding a changed provider in its manifest must still fail the external manifest pin.
        with tempfile.TemporaryDirectory(prefix="m01-rebound-", dir=os.environ["M01_OUTPUT"]) as directory:
            root = Path(directory)
            shutil.copyfile(CONSUMER / "consumer.py", root / "consumer.py")
            manifest = json.loads((CONSUMER / "input-manifest.json").read_text())
            manifest["files"][-1]["sha256"] = "0" * 64
            (root / "input-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): load_frozen_current_case(root)
        for args in [(("a", "b"), ("p", "p"), (("a", "b"),)*2),
                     (("a", "b"), ("p",), (("b", "b"),)),
                     (("a", "b"), ("p",), (("a", "missing"),))]:
            with self.assertRaises(ValueError): CountPairPort(*args)

    def test_11_tiny_complete_count_factor_limiting_case_and_ownership(self):
        port = CountPairPort(("a", "b", "minor"), ("p1", "p2"), (("a", "b"),)*2)
        with self.assertRaises(ValueError): port.claim_observations({"p2"})
        self.assertEqual(port.claim_observations({"official"}), {"official", "p1", "p2"})
        x = pt.dvector("six_category_log_rates")
        rates = pt.exp(x)
        observed = np.array([3, 2, 1, 1, 0, 0])  # invented candidate/nonresponse/blank/invalid counts
        nonpoll = pt.sum(observed*x - rates - gammaln(observed+1)) - .5*pt.sum(x*x) - len(observed)*.5*np.log(2*np.pi)
        mean = port.symbolic_mean_from_log_rates(x[:3])
        covariance = np.array([[.2, .05], [.05, .3]])
        y = np.array([.1, -.1])
        fixed = -.5*(2*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]
                      + pt.dot(y-mean, pt.dot(np.linalg.inv(covariance), y-mean)))
        one = SharedVectorMixture([covariance], [1]).symbolic_logpdf(y, mean)
        full = SharedVectorMixture([covariance, covariance*2], [.4, .6]).symbolic_logpdf(y, mean)
        fn = pytensor.function([x], [nonpoll+fixed, nonpoll+one, nonpoll+full,
                                    pt.grad(nonpoll+fixed, x), pt.grad(nonpoll+one, x), rates, pt.sum(rates)], mode="FAST_COMPILE")
        a, b, alternative, ga, gb, category_rates, total = fn(np.array([.1, -.2, -.8, -1, -2, -2.5]))
        assert_allclose(a, b, rtol=1e-12, atol=1e-10)
        assert_allclose(ga, gb, rtol=1e-9, atol=1e-9)
        self.assertNotAlmostEqual(float(alternative), float(a), places=6)
        assert_allclose(category_rates.sum(), total, atol=1e-14)
        self.assertEqual(int(observed.sum()), 7)
        DETAILS["tinyCountGraph"] = dict(categoryCount=6, observedParticipants=7, source="invented", nativeProviderInstalled=False)

    def test_12_reproducibility_source_continuity_and_scope(self):
        for path, expected in self.before.items(): self.assertEqual(digest(path), expected)
        self.assertNotIn("pymc", sys.modules, "No PyMC model or sampler imported")
        self.assertTrue(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1")
        self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")
        self.assertTrue(str(Path(os.environ["M01_OUTPUT"]).resolve()).startswith("/mnt/research-storage/"))
        specification = json.loads(V2.read_text())
        self.assertEqual(specification["version"], 2)
        self.assertEqual(specification["authorization"]["basis"].split(".")[0],
                         "Existing user standing goal and parent task authorize an isolated lab-only mathematical prototype and deterministic tests")
        self.assertEqual(len(specification["governingInstructions"]["skillFiles"]), 2)


if __name__ == "__main__":
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrototypeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = Path(os.environ["M01_OUTPUT"])
    report = dict(schema="M01SuppliedStatePrototypeTestsV1", status="passed" if result.wasSuccessful() else "failed",
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-start,
                  runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pytensor=pytensor.__version__,
                               mode="FAST_COMPILE CPU float64", threads=1), details=DETAILS,
                  sourceHashes={name: digest(HERE/name) for name in ["prototype.py", "test_prototype.py"]},
                  scope=dict(samplerCalls=0, optimizerFitCalls=0, nativeProviderInstalled=False,
                             current2026BankSuppliedPointsOnly=True, realHistoricalLabelLoads=0,
                             empiricalQualification=False, publication=False))
    (output/"test-report.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
