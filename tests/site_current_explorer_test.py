"""Synthetic contract checks for current explorer candidate and chamber mapping."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.export_current_explorer import (
    CurrentExplorerError,
    _race_outcomes,
    _summary_from_counts,
    _scenario_payload,
    _check_current_model_alignment,
    _validate_polling_scales,
)


class CurrentExplorerMappingTest(unittest.TestCase):
    def test_principal_pair_maps_complementary_share_quantiles(self) -> None:
        race = {
            "id": "2026:AA:senate:regular",
            "state": "AA",
            "seatId": "US-SEN-AA-CLASS-1",
            "candidates": [
                {"id": "d-1", "name": "Dana D", "party": "D", "incumbent": True, "sourceUrls": []},
                {"id": "r-1", "name": "Riley R", "party": "R", "incumbent": False, "sourceUrls": []},
            ],
            "firstWinProbability": 0.6,
            "secondWinProbability": 0.4,
            "firstShareMean": 0.55,
            "shareQuantiles": {"p05": 0.42, "p50": 0.55, "p95": 0.68},
            "notes": ["Two-principal proxy."],
        }
        source_race = {"share_interpretation": "democratic_two_party"}
        adapted = _race_outcomes(race, source_race)

        self.assertEqual([row["id"] for row in adapted["outcomes"]], ["d-1", "r-1"])
        first, second = adapted["outcomes"]
        self.assertAlmostEqual(first["winProbability"] + second["winProbability"], 1.0)
        self.assertEqual(first["share"]["mean"], 0.55)
        self.assertAlmostEqual(second["share"]["mean"], 0.45)
        for key, expected in {"p05": 0.32, "p50": 0.45, "p95": 0.58}.items():
            self.assertAlmostEqual(second["share"]["quantiles"][key], expected)
        self.assertEqual(
            first["share"]["denominator"],
            "Share of votes for the two named principal contenders",
        )
        self.assertEqual(first["members"][0]["id"], "d-1")

    def test_unassigned_seat_pmf_is_jointly_summarized(self) -> None:
        summary = _summary_from_counts(np.array([0, 1, 1, 3]))
        self.assertEqual(len(summary["pmf"]), 101)
        self.assertAlmostEqual(sum(summary["pmf"]), 1.0)
        self.assertEqual(summary["pmf"][0], 0.25)
        self.assertEqual(summary["pmf"][1], 0.5)
        self.assertEqual(summary["pmf"][3], 0.25)
        self.assertEqual(summary["expectedSeats"], 1.25)

    def test_scenario_uses_joint_unassigned_seat_draws(self) -> None:
        d_draws = np.array([60, 52, 49, 55])
        r_draws = np.array([40, 43, 46, 42])
        model = {
            "controls": {
                "campaign_pledge_unaligned": {
                    "description": "Synthetic organizing case.",
                    "democraticMajorityProbability": 0.75,
                    "republicanMajorityProbability": 0.25,
                    "unresolvedOrganizingProbability": 0.0,
                }
            },
            "chamberDistributions": {
                "caucus": {
                    "campaign_pledge_unaligned": {
                        "D": _summary_from_counts(d_draws),
                        "R": _summary_from_counts(r_draws),
                    }
                }
            },
        }
        tipping = {
            "scenarios": {
                "campaign_pledge_unaligned": {
                    "D": {"retainedSeats": 34},
                    "R": {"retainedSeats": 31},
                }
            }
        }
        scenario = {"id": "campaign_pledge_unaligned"}
        payload = _scenario_payload(
            model,
            tipping,
            scenario,
            {
                "scenario:campaign_pledge_unaligned:D": d_draws,
                "scenario:campaign_pledge_unaligned:R": r_draws,
            },
        )
        self.assertEqual(payload["retainedSeats"], {"D": 34, "R": 31, "U": 0})
        self.assertEqual(payload["seats"]["U"]["pmf"][0], 0.25)
        self.assertEqual(payload["seats"]["U"]["pmf"][3], 0.25)
        self.assertEqual(payload["seats"]["U"]["pmf"][5], 0.5)
        self.assertAlmostEqual(sum(payload["seats"]["U"]["pmf"]), 1.0)

    def test_model_alignment_rejects_candidate_identity_change(self) -> None:
        race = {
            "id": "2026:AA:senate:regular", "state": "AA", "seatId": "US-SEN-AA-CLASS-1",
            "outcomes": [{"id": "candidate:D"}, {"id": "candidate:R"}],
        }
        base = {"races": [race]}
        changed = {"races": [{**race, "outcomes": [{"id": "candidate:other"}, {"id": "candidate:R"}]}]}
        with self.assertRaises(CurrentExplorerError):
            _check_current_model_alignment({"current-fundamentals": base, "current-polling": changed})

    def test_invalid_seat_counts_are_rejected(self) -> None:
        with self.assertRaises(CurrentExplorerError):
            _summary_from_counts(np.array([100.2]))
        with self.assertRaises(CurrentExplorerError):
            _summary_from_counts(np.array([101]))

    def test_polling_sensitivity_label_must_match_saved_multipliers(self) -> None:
        spec = {"id": "current-error-low", "uncertainty": "error-low"}
        _validate_polling_scales(
            spec,
            {"all_scale_multiplier": 2 / 3, "cycle_scale_multiplier": 1.0},
        )
        with self.assertRaises(CurrentExplorerError):
            _validate_polling_scales(
                spec,
                {"all_scale_multiplier": 1.5, "cycle_scale_multiplier": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
