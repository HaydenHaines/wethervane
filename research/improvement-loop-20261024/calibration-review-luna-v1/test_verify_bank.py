import unittest

import numpy as np

from verify_bank import COMPONENTS, EXPECTED_CHAINS, EXPECTED_POSTERIOR, verify_bank_arrays


def fixture_bank():
    rng = np.random.default_rng(84)
    common = rng.normal(size=(EXPECTED_CHAINS, EXPECTED_POSTERIOR, 1))
    independent = rng.normal(size=(EXPECTED_CHAINS, EXPECTED_POSTERIOR, len(COMPONENTS)))
    logs = -2.5 + 0.08 * common + 0.03 * independent
    scales = np.exp(logs)
    return logs, scales, np.asarray(COMPONENTS)


class BankGuardTests(unittest.TestCase):
    def test_accepts_one_row_aligned_joint_five_scale_bank(self):
        logs, scales, names = fixture_bank()
        report = verify_bank_arrays(logs, scales, names, logs.copy())
        self.assertEqual(report["shape"], [4, 1000, 5])
        self.assertEqual(report["components"], COMPONENTS)
        self.assertGreater(abs(report["log_scale_correlation"][0][1]), 0.1)

    def test_rejects_marginal_recombination_or_row_reordering(self):
        logs, scales, names = fixture_bank()
        wrong = logs.copy()
        wrong[:, :, [0, 1]] = wrong[:, :, [1, 0]]
        with self.assertRaisesRegex(ValueError, "joint draws"):
            verify_bank_arrays(wrong, np.exp(wrong), names, logs)

    def test_rejects_wrong_shape_component_order_or_bad_transform(self):
        logs, scales, names = fixture_bank()
        with self.assertRaisesRegex(ValueError, "wrong posterior bank shape"):
            verify_bank_arrays(logs[:, :-1], scales[:, :-1], names)
        with self.assertRaisesRegex(ValueError, "component ordering"):
            verify_bank_arrays(logs, scales, names[::-1])
        bad = scales.copy()
        bad[0, 0, 0] *= 1.01
        with self.assertRaisesRegex(ValueError, "not exp"):
            verify_bank_arrays(logs, bad, names)


if __name__ == "__main__":
    unittest.main()
