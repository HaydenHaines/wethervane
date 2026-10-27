import hashlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check import (
    CENTERED_PAIRS,
    ORIGINAL_PAIRS,
    WEIGHT_CASES,
    build_result,
    ordinary_poll_logit,
    read_pinned,
    softmax3,
)


class StableInteractionTests(unittest.TestCase):
    def test_pinned_sources_and_preserved_legacy_fixture(self):
        result = build_result()
        self.assertEqual(result["legacy_fixed_fixture_output_preserved"]["schema"],
                         "A2018PinnedAggregateAndInventedInteractionCheckV1")
        self.assertEqual(result["aggregate_only_a2018"]["source_closure_checks_passed"], [
            "open_person_absent", "no_official_counts_or_reports", "no_ranked_or_runoff_observations",
            "survey_loop_eventG_free", "ordinary_poll_formula_verified", "contest_response_is_scalar",
        ])
        self.assertFalse(result["aggregate_only_a2018"]["forecast_qualified"])

    def test_centered_contrast_fixture_matches_independent_review(self):
        result = build_result()["reviewed_centered_contrast_sensitivity"]
        self.assertEqual(result["pair_utilities_by_profile"], [list(row) for row in CENTERED_PAIRS])
        self.assertAlmostEqual(result["difference_in_differences"], -0.0068878407953504705, places=13)
        self.assertGreater(abs(result["difference_in_differences"]), 0.001)

    def test_extreme_finite_utilities_use_stable_softmax_and_log_rates(self):
        probabilities = softmax3(1000.0, 999.0, 998.0)
        self.assertTrue(all(math.isfinite(value) for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=15)
        value = ordinary_poll_logit([1.0], [[-1000.0, 1000.0]], 0.0, 0.5, 2.0)
        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, -2000.0, places=10)

    def test_mismatched_profile_shapes_reject(self):
        for weights, profiles in [([1.0], [[1.0, 0.0], [2.0, 1.0]]),
                                  ([1.0, 1.0], [[1.0, 0.0]]),
                                  ([1.0], [[1.0, 0.0, -1.0]])]:
            with self.subTest(weights=weights, profiles=profiles), self.assertRaises(ValueError):
                ordinary_poll_logit(weights, profiles, 0.0, 0.5, 1.0)

    def test_zero_nan_and_nonfinite_weights_utilities_scales_reject(self):
        bad_cases = [
            ([0.0], [[1.0, 0.0]], 0.0, 0.5, 1.0),
            ([float("nan")], [[1.0, 0.0]], 0.0, 0.5, 1.0),
            ([1.0], [[float("nan"), 0.0]], 0.0, 0.5, 1.0),
            ([1.0], [[1.0, 0.0]], 0.0, 0.0, 1.0),
            ([1.0], [[1.0, 0.0]], 0.0, float("nan"), 1.0),
            ([1.0], [[1.0, 0.0]], 0.0, 0.5, 0.0),
            ([1.0], [[1.0, 0.0]], 0.0, 0.5, float("nan")),
            ([1.0], [[1.0, 0.0]], 0.0, float("inf"), 1.0),
        ]
        for arguments in bad_cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ordinary_poll_logit(*arguments)
        with self.assertRaises(ValueError):
            softmax3(1.0, float("nan"), 0.0)

    def test_scalar_cancellation_and_scale_invariances(self):
        for weights in WEIGHT_CASES:
            for other in (-1.0, 1.0):
                base = ordinary_poll_logit(weights, ORIGINAL_PAIRS, other, 0.72, 1.5)
                changed_scales = ordinary_poll_logit(weights, ORIGINAL_PAIRS, other, 0.19, 6.4)
                changed_weights = ordinary_poll_logit([w * 17.0 for w in weights], ORIGINAL_PAIRS,
                                                       other, 0.72, 1.5)
                shifted = ordinary_poll_logit(weights,
                    [(d + 100.0, r + 100.0) for d, r in ORIGINAL_PAIRS], other + 100.0, 0.72, 1.5)
                self.assertAlmostEqual(base, changed_scales, delta=1e-13)
                self.assertAlmostEqual(base, changed_weights, delta=1e-13)
                self.assertAlmostEqual(base, shifted, delta=1e-13)

    def test_exact_representable_large_gauge_offset_regression(self):
        weights = (1.0, 9.0)
        pairs = ((4.0, -4.0), (-4.0, 4.0))
        baseline = ordinary_poll_logit(weights, pairs, 0.0, 0.72, 1.5)
        shifted_pairs = tuple((d + 1e16, r + 1e16) for d, r in pairs)
        shifted = ordinary_poll_logit(weights, shifted_pairs, 1e16, 0.72, 1.5)
        self.assertEqual(tuple((d - 1e16, r - 1e16) for d, r in shifted_pairs), pairs)
        self.assertAlmostEqual(baseline, -2.1942472351374973, delta=1e-14)
        self.assertAlmostEqual(baseline, shifted, delta=1e-14)
        pairs20 = ((65536.0, -65536.0), (-65536.0, 65536.0))
        offset = 1e20
        shifted20 = tuple((d + offset, r + offset) for d, r in pairs20)
        self.assertEqual(tuple((d - offset, r - offset) for d, r in shifted20), pairs20)
        base20 = ordinary_poll_logit(weights, pairs20, 0.0, 0.72, 1.5)
        value20 = ordinary_poll_logit(weights, shifted20, offset, 0.72, 1.5)
        self.assertAlmostEqual(base20, value20, delta=1e-10)

    def test_source_tamper_is_rejected_when_assertions_are_optimized(self):
        # The explicit exception gate is tested in a real -O child interpreter.
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "input.json"
            original = b'{"fixture": 1}'
            candidate.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            candidate.write_bytes(b'{"fixture": 2}')
            script = (
                "from pathlib import Path; import sys; "
                "from check import read_pinned; "
                "read_pinned(Path(sys.argv[1]), '" + expected + "')"
            )
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, "-O", "-B", "-c", script, str(candidate)],
                                 cwd=Path(__file__).parent, env=env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("ValueError", proc.stderr)
            self.assertIn("Pinned source hash differs", proc.stderr)

    def test_centered_fixture_is_not_mistaken_for_native_or_fitted_state(self):
        result = build_result()
        self.assertIn("not a solved full native parameter state",
                      " ".join(result["limits"]))
        self.assertFalse(result["aggregate_only_a2018"]["compiled_graph_authenticated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
