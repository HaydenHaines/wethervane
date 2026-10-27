import json
import tempfile
import unittest
from pathlib import Path

from check import frozen_check, inspect


class CovarianceCheckTest(unittest.TestCase):
    def test_positive_and_ill_conditioned(self):
        value = inspect([[2, 0.5], [0.5, 1]])
        self.assertEqual(value["dimension"], 2)
        self.assertLess(value["cholesky_relative_frobenius_residual"], 1e-14)
        with self.assertRaisesRegex(ValueError, "margin"):
            inspect([[1, 0], [0, 1e-20]])

    def test_invalid_and_indefinite_inputs(self):
        for matrix in ([], [[1, 2]], [[1, 0.2], [0.1, 1]], [[1, float("nan")], [float("nan"), 1]], [[0, 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                inspect(matrix)
        with self.assertRaisesRegex(ValueError, "margin"):
            inspect([[1, 2], [2, 1]])

    def test_byte_pin_and_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}")
            with self.assertRaisesRegex(ValueError, "byte pin"):
                frozen_check(bad)
        value = frozen_check()
        self.assertEqual(value["statistics"]["dimension"], 274)
        self.assertFalse(value["forecast_qualified"])
        self.assertFalse(value["compiled_graph_authenticated"])
        self.assertNotIn("observation_id", json.dumps(value))
        self.assertNotIn("covariance\": [", json.dumps(value))


if __name__ == "__main__":
    unittest.main()
