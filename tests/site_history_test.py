"""Historical explorer checks independent of the saved fitting implementation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.export_history_explorer import (
    HistoryExportError,
    checked_file,
    control_probabilities,
    distribution,
    export_outcomes,
    reconstruct_seats,
)


OUTPUT = Path(__file__).resolve().parents[1] / "public/data/explorer"


class HistoryAccountingTest(unittest.TestCase):
    def test_tied_chamber_uses_the_declared_vice_president(self) -> None:
        draws = np.array([[50, 50, 0], [49, 50, 1], [51, 49, 0]])
        republican_vp = control_probabilities(draws, "R")
        self.assertAlmostEqual(republican_vp["democraticMajorityProbability"], 1 / 3)
        self.assertAlmostEqual(republican_vp["republicanMajorityProbability"], 2 / 3)
        self.assertEqual(republican_vp["unresolvedOrganizingProbability"], 0)
        democratic_vp = control_probabilities(draws, "D")
        self.assertAlmostEqual(democratic_vp["democraticMajorityProbability"], 2 / 3)
        self.assertEqual(democratic_vp["republicanMajorityProbability"], 0)
        self.assertAlmostEqual(democratic_vp["unresolvedOrganizingProbability"], 1 / 3)

    def test_shared_state_races_are_counted_as_distinct_seats(self) -> None:
        contests = [{"race_id": f"race:{i}", "group_ids": ["D", "R"]} for i in range(35)]
        contests[0]["race_id"] = "2018:MN:senate:regular"
        contests[1]["race_id"] = "2018:MN:senate:special"
        scenario = {
            "fixed_seats": [(f"fixed:{i}", "D" if i < 23 else "R") for i in range(65)],
            "race_memberships": [(race["race_id"], ["D", "R"]) for race in contests],
        }
        winners = np.ones((2, 35), dtype=int)
        winners[0, :2] = 0
        self.assertEqual(reconstruct_seats(winners, contests, scenario).tolist(), [[25, 75, 0], [23, 77, 0]])

    def test_group_probability_is_not_distributed_among_members(self) -> None:
        candidates = {
            ("CA", "a"): {"group_id": "D", "candidate_name": "Candidate A", "ballot_party": "D", "incumbent": "True"},
            ("CA", "b"): {"group_id": "D", "candidate_name": "Candidate B", "ballot_party": "D", "incumbent": "False"},
            ("CA", "c"): {"group_id": "O:c", "candidate_name": "Independent C", "ballot_party": "IND", "incumbent": "False"},
        }
        groups = {
            ("CA", "D"): {"candidate_ids": "a|b", "bloc": "D"},
            ("CA", "O:c"): {"candidate_ids": "c", "bloc": "O"},
        }
        outcomes = export_outcomes({"race_id": "CA", "group_ids": ["D", "O:c"]}, np.array([0, 0, 0, 1]), candidates, groups)
        self.assertEqual(outcomes[0]["kind"], "group")
        self.assertEqual(outcomes[0]["winProbability"], 0.75)
        self.assertEqual(len(outcomes[0]["members"]), 2)
        self.assertTrue(all("winProbability" not in member for member in outcomes[0]["members"]))
        self.assertEqual(outcomes[1]["party"], "IND")
        self.assertEqual(outcomes[1]["winProbability"], 0.25)
        with self.assertRaises(HistoryExportError):
            export_outcomes({"race_id": "CA", "group_ids": ["D", "O:c"]}, np.array([2]), candidates, groups)

    def test_distribution_rejects_fractional_seats_and_preserves_mass(self) -> None:
        result = distribution(np.array([40, 50, 50, 60]))
        self.assertEqual(result["expectedSeats"], 50)
        self.assertEqual(result["pmf"][50], 0.5)
        self.assertEqual(sum(result["pmf"]), 1)
        with self.assertRaises(HistoryExportError):
            distribution(np.array([50.5]))

    def test_changed_source_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.json").write_text("changed", encoding="utf-8")
            with self.assertRaises(HistoryExportError):
                checked_file(root, "source.json", hashlib.sha256(b"sealed").hexdigest())


class PublishedHistoricalArchiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = json.loads((OUTPUT / "history-index.json").read_text())
        cls.bundles = {}
        for entry in cls.entries:
            path = OUTPUT / (entry["id"] + ".json")
            cls.bundles[entry["id"]] = json.loads(path.read_text())

    def test_complete_registry_has_immutable_bundle_hashes(self) -> None:
        self.assertEqual(len(self.entries), 32)
        self.assertEqual(len({entry["id"] for entry in self.entries}), 32)
        for entry in self.entries:
            path = OUTPUT / (entry["id"] + ".json")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
            self.assertEqual(entry["href"], f"/data/explorer/{path.name}")
            self.assertEqual(self.bundles[entry["id"]]["run"], {k: v for k, v in entry.items() if k not in {"href", "sha256"}})

    def test_all_runs_keep_complete_group_support_and_fixed_observed_truth(self) -> None:
        for bundle in self.bundles.values():
            run = bundle["run"]
            year = run["year"]
            observed = {"D": 47, "R": 53, "U": 0} if year == 2018 else {"D": 51, "R": 49, "U": 0}
            self.assertEqual(len(bundle["races"]), 35)
            self.assertEqual(len({race["id"] for race in bundle["races"]}), 35)
            self.assertEqual(len({race["seatId"] for race in bundle["races"]}), 35)
            self.assertIsNone(bundle["tipping"])
            self.assertFalse(run["strictAsOf"])
            self.assertEqual(run["simulationCount"], 80000)
            self.assertEqual(run["horizonDays"], 14)
            self.assertTrue(run["issuedAt"].startswith("2026-"))
            self.assertTrue(run["forecastAt"].startswith(str(year)))
            for race in bundle["races"]:
                self.assertAlmostEqual(sum(outcome["winProbability"] for outcome in race["outcomes"]), 1, places=12)
                self.assertIn(race["observed"]["outcomeId"], [outcome["id"] for outcome in race["outcomes"]])
                for outcome in race["outcomes"]:
                    self.assertNotIn("share", outcome)
                    self.assertTrue(0 <= outcome["winProbability"] <= 1)
                    self.assertTrue(all("winProbability" not in member for member in outcome["members"]))
            for scenario in bundle["scenarios"]:
                self.assertEqual(sum(scenario["retainedSeats"].values()), 65)
                self.assertEqual(scenario["performance"]["observedSeats"], observed)
                self.assertEqual(scenario["performance"]["observedControl"], "R" if year == 2018 else "D")
                self.assertEqual(scenario["thresholds"], {"D": 51, "R": 50} if year == 2018 else {"D": 50, "R": 51})
                seats = scenario["seats"]
                self.assertAlmostEqual(sum(row["expectedSeats"] for row in seats.values()), 100, places=10)
                for party, scores_key in (("D", "dSeatCrps"), ("R", "rSeatCrps")):
                    pmf = np.array(seats[party]["pmf"])
                    self.assertAlmostEqual(float(pmf.sum()), 1, places=12)
                    self.assertAlmostEqual(float(np.dot(np.arange(101), pmf)), seats[party]["expectedSeats"], places=10)
                    # Independent integer-CDF CRPS, not the exporter's copied sealed score.
                    empirical_crps = float(np.sum((pmf.cumsum() - (np.arange(101) >= observed[party])) ** 2))
                    self.assertAlmostEqual(empirical_crps, scenario["performance"][scores_key], places=10)
                control = scenario["control"]
                self.assertAlmostEqual(sum(seats["D"]["pmf"][scenario["thresholds"]["D"]:]), control["democraticMajorityProbability"], places=12)
                self.assertAlmostEqual(sum(seats["R"]["pmf"][scenario["thresholds"]["R"]:]), control["republicanMajorityProbability"], places=12)
                observed_probability = control["republicanMajorityProbability"] if year == 2018 else control["democraticMajorityProbability"]
                self.assertAlmostEqual(-math.log(observed_probability), scenario["performance"]["controlLogScore"], places=10)
                observed_control = "R" if year == 2018 else "D"
                brier = sum(
                    (control[key] - (party == observed_control)) ** 2
                    for party, key in (
                        ("D", "democraticMajorityProbability"),
                        ("R", "republicanMajorityProbability"),
                        ("U", "unresolvedOrganizingProbability"),
                    )
                )
                self.assertAlmostEqual(brier, scenario["performance"]["controlBrier"], places=12)
            public_text = json.dumps(bundle)
            self.assertNotIn("/home/", public_text)
            self.assertNotIn(".redesign-runtime", public_text)

    def test_known_primary_scores_and_grouped_candidates_are_preserved(self) -> None:
        first = self.bundles["history-2018-FP-1"]
        second = self.bundles["history-2022-FP-1"]
        self.assertEqual(first["scenarios"][0]["performance"]["dSeatCrps"], 0.829265416875)
        self.assertEqual(second["scenarios"][0]["performance"]["dSeatCrps"], 0.5572843862500002)
        ca = next(race for race in first["races"] if race["state"] == "CA")
        self.assertEqual(ca["outcomes"][0]["kind"], "group")
        self.assertEqual(ca["outcomes"][0]["winProbability"], 1)
        self.assertEqual({member["name"] for member in ca["outcomes"][0]["members"]}, {"Dianne Feinstein", "Kevin De Leon"})
        self.assertEqual(ca["observed"]["winnerName"], "Dianne Feinstein")
        alaska = next(race for race in second["races"] if race["state"] == "AK")
        republican_group = next(outcome for outcome in alaska["outcomes"] if outcome["id"] == "R")
        self.assertEqual(republican_group["kind"], "group")
        self.assertEqual(len(republican_group["members"]), 3)
        self.assertTrue(any("Murkowski" in member["name"] for member in republican_group["members"]))
        for state in ("ME", "VT"):
            race = next(race for race in first["races"] if race["state"] == state)
            self.assertEqual(race["observed"]["winnerParty"], "IND")
        self.assertEqual(len([race for race in first["races"] if race["state"] == "MN"]), 2)
        self.assertEqual(len([race for race in second["races"] if race["state"] == "OK"]), 2)


if __name__ == "__main__":
    unittest.main()
