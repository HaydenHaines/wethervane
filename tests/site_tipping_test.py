"""Focused unit tests for tie-safe saved-draw tipping ranks."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.export_tipping_points import (
    TippingSnapshotError,
    _fractional_rank_frequencies,
    _member_configuration,
)


class FractionalRankTest(unittest.TestCase):
    def test_kth_tie_is_split_without_race_order_bias(self) -> None:
        scores = np.array([[0.9, 0.5, 0.5, 0.1], [0.8, 0.7, 0.3, 0.2]])
        kth_scores, assignment = _fractional_rank_frequencies(scores, 2)

        np.testing.assert_allclose(kth_scores, [0.5, 0.7])
        np.testing.assert_allclose(
            assignment,
            [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(assignment.sum(axis=1), [1.0, 1.0])
        # Expected frequencies retain each simulation at equal weight.
        np.testing.assert_allclose(assignment.mean(axis=0), [0.0, 0.75, 0.25, 0.0])

    def test_tie_spanning_multiple_ranks_still_assigns_one_tipping_point(self) -> None:
        scores = np.array([[0.8, 0.7, 0.7, 0.7, 0.1]])
        _, assignment = _fractional_rank_frequencies(scores, 3)
        np.testing.assert_allclose(assignment, [[0.0, 1 / 3, 1 / 3, 1 / 3, 0.0]])
        self.assertAlmostEqual(float(assignment.sum()), 1.0)

    def test_all_equal_scores_split_across_every_eligible_race(self) -> None:
        scores = np.full((3, 5), 0.61)
        kth_scores, assignment = _fractional_rank_frequencies(scores, 3)

        np.testing.assert_allclose(kth_scores, [0.61, 0.61, 0.61])
        np.testing.assert_allclose(assignment, np.full((3, 5), 0.2))
        np.testing.assert_allclose(assignment.mean(axis=0).sum(), 1.0)

    def test_synthetic_membership_maps_candidate_and_retained_seats(self) -> None:
        races = []
        democratic = [f"contender:state-{index}:D" for index in range(35)]
        republican = [f"contender:state-{index}:R" for index in range(35)]
        for index in range(35):
            races.append(
                {
                    "race_id": f"2026:state-{index}:senate:regular",
                    "first_contender": {"id": democratic[index]},
                    "second_contender": {"id": republican[index]},
                }
            )
        d_holders = [f"holder:D-{index}" for index in range(34)]
        r_holders = [f"holder:R-{index}" for index in range(31)]
        other_holders = [f"holder:I-{index}" for index in range(65 - 34 - 31)]
        input_data = {
            "non_up_seats": [
                {"holder_id": holder}
                for holder in d_holders + r_holders + other_holders
            ]
        }
        scenario = {
            "id": "campaign_pledge_unaligned",
            "democratic_caucus_members": democratic + d_holders,
            "republican_caucus_members": republican + r_holders,
        }

        d = _member_configuration(races, input_data, scenario, "D")
        r = _member_configuration(races, input_data, scenario, "R")
        self.assertEqual(d["retained"], 34)
        self.assertEqual(r["retained"], 31)
        self.assertEqual(d["eligible_indices"], list(range(35)))
        self.assertEqual(r["eligible_indices"], list(range(35)))
        self.assertEqual(d["member_candidate_by_index"][12], democratic[12])
        self.assertEqual(r["member_candidate_by_index"][12], republican[12])

    def test_invalid_rank_is_rejected(self) -> None:
        scores = np.array([[0.8, 0.2]])
        for rank in (0, 3):
            with self.subTest(rank=rank), self.assertRaises(TippingSnapshotError):
                _fractional_rank_frequencies(scores, rank)


if __name__ == "__main__":
    unittest.main()
