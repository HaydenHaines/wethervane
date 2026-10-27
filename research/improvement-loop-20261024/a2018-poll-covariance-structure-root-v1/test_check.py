import json
import tempfile
import unittest
from pathlib import Path

from check import frozen_check, inspect


class StructureTest(unittest.TestCase):
    def test_invented_support_and_signs(self):
        value = inspect([[1, .2, -.3], [.2, 1, 0], [-.3, 0, 1]], ["A", "A", "B"])
        self.assertEqual(value["same_contest"]["positive"], 1)
        self.assertEqual(value["different_contest"]["negative"], 1)
        self.assertEqual(value["different_contest"]["exact_zero"], 1)

    def test_invalid_inputs_rejected(self):
        for matrix, contests in [([[1]], []), ([[1, 2], [2, 1]], ["A", "B"]),
                                 ([[1, 0], [.1, 1]], ["A", "B"]), ([[0]], ["A"]), ([[1]], [""])]:
            with self.subTest(matrix=matrix, contests=contests), self.assertRaises(ValueError):
                inspect(matrix, contests)

    def test_frozen_source_pin_and_aggregate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{}")
            with self.assertRaisesRegex(ValueError, "byte pin"):
                frozen_check(bad)
        result = frozen_check()
        self.assertEqual(result["structure"]["rows"], 274)
        self.assertEqual(result["structure"]["distinct_contests"], 29)
        self.assertFalse(result["raw_sample_overlap_verified"])
        self.assertNotIn("observation_id", json.dumps(result))
        self.assertNotIn("contest_id", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
