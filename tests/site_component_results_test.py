"""Exact race-only result projection and source admission; no forecast execution."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.export_component_results as adapter
from scripts.export_component_catalog import CatalogError

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/component-results"
SOURCE = Path(
    os.environ.get("BALLOTMANCY_RESEARCH_ROOT", "/home/hayden/projects/wethervane-redesign-20261006/integration")
)
read = lambda path: json.loads(path.read_text())


def fixture():
    ids = [f"2018:S{i}:senate:regular" for i in range(30)]
    summary = {
        "race_ids": ids,
        "quantile_levels": [0.025, 0.1, 0.5, 0.9, 0.975],
        "dem_win_probability": [0.7] * 30,
        "mean_dem_share": [0.55] * 30,
        "share_quantiles": [[value] * 30 for value in [0.35, 0.4, 0.55, 0.7, 0.75]],
    }
    features = [
        dict(
            race_id=race,
            cycle="2018",
            year="2018",
            state=f"S{i}",
            state_abbrev=f"S{i}",
            eligible="True",
            special="False",
            dem_candidate_name=f"Democrat {i}",
            rep_candidate_name=f"Republican {i}",
            dem_incumbent="True",
            rep_incumbent="False",
            seat_id=f"seat-{i}",
            roster_vintage="final_result_roster_proxy;not_dated_ballot_snapshot",
        )
        for i, race in enumerate(ids)
    ]
    outcomes = [
        dict(race_id=race, cycle="2018", dem_votes="60", rep_votes="40", two_party_votes="100", actual_dem_share="0.6")
        for race in ids
    ]
    return summary, features, outcomes


class ComponentResultsTests(unittest.TestCase):
    def test_result_index_hashes_scope_and_no_chamber_outputs(self):
        index = read(DATA / "index.json")
        self.assertEqual({e["id"] for e in index["results"]}, {"ordinary-2018-f", "ordinary-2018-fp"})
        for entry in index["results"]:
            raw = (DATA / entry["file"]).read_bytes()
            result = json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
            self.assertEqual(result["configurationSha256"], entry["configurationSha256"])
            self.assertEqual(result["scope"], "ordinary_dr_race_subset")
            self.assertEqual(result["year"], 2018)
            self.assertFalse(result["strictAsOf"])
            self.assertEqual(
                result["status"],
                {"lifecycle": "ready", "numerical": "passed", "evidence": "unassessed", "experimental": True},
            )
            self.assertEqual(len(result["races"]), 30)
            for field in ("scenarios", "control", "tipping", "retainedSeats"):
                self.assertNotIn(field, result)
            for capability in ("chamberControl", "tipping", "counts"):
                self.assertFalse(result["capabilities"][capability])
            self.assertTrue(result["sourceAttribution"])
            for fingerprint in result["sourceFingerprints"].values():
                self.assertRegex(fingerprint, r"^[a-f0-9]{64}$")
            serialized = json.dumps(result)
            for private in ("/home/", "research/", ".redesign-runtime/", "src/"):
                self.assertNotIn(private, serialized)

    def test_probability_share_and_quantile_complements_keep_exact_levels(self):
        for filename in ("f.json", "fp.json"):
            result = read(DATA / filename)
            self.assertEqual(result["quantileLevels"], [0.025, 0.1, 0.5, 0.9, 0.975])
            for race in result["races"]:
                d, r = race["candidates"]
                self.assertEqual([d["party"], r["party"]], ["D", "R"])
                self.assertEqual(r["winProbability"], 1 - d["winProbability"])
                self.assertEqual(r["share"]["mean"], 1 - d["share"]["mean"])
                self.assertEqual([q["level"] for q in d["share"]["quantiles"]], result["quantileLevels"])
                self.assertEqual(
                    [q["value"] for q in r["share"]["quantiles"]],
                    [1 - q["value"] for q in reversed(d["share"]["quantiles"])],
                )
                actual = race["observed"]
                self.assertEqual(actual["twoPartyVotes"], actual["democraticVotes"] + actual["republicanVotes"])
                self.assertAlmostEqual(
                    actual["democraticTwoPartyShare"], actual["democraticVotes"] / actual["twoPartyVotes"]
                )
                self.assertNotIn("winnerName", actual)
                self.assertNotIn("outcomeId", actual)

    def test_join_uses_race_identity_not_csv_row_order(self):
        summary, features, outcomes = fixture()
        races = adapter.project_races(summary, list(reversed(features)), list(reversed(outcomes)))
        self.assertEqual([r["id"] for r in races], summary["race_ids"])
        self.assertEqual(races[0]["candidates"][0]["name"], "Democrat 0")

    def test_missing_duplicate_or_inconsistent_joins_fail_closed(self):
        for kind in (
            "missing-feature",
            "duplicate-feature",
            "missing-outcome",
            "duplicate-outcome",
            "wrong-cycle",
            "wrong-state",
            "special",
            "roster",
            "count",
            "share",
        ):
            summary, features, outcomes = fixture()
            if kind == "missing-feature":
                features.pop()
            elif kind == "duplicate-feature":
                features.append(copy.deepcopy(features[0]))
            elif kind == "missing-outcome":
                outcomes.pop()
            elif kind == "duplicate-outcome":
                outcomes.append(copy.deepcopy(outcomes[0]))
            elif kind == "wrong-cycle":
                outcomes[0]["cycle"] = "2022"
            elif kind == "wrong-state":
                features[0]["state"] = "XX"
            elif kind == "special":
                features[0]["special"] = "True"
            elif kind == "roster":
                features[0]["roster_vintage"] = "verified_asof"
            elif kind == "count":
                outcomes[0]["two_party_votes"] = "101"
            elif kind == "share":
                outcomes[0]["actual_dem_share"] = "0.5"
            with self.subTest(kind=kind), self.assertRaises(CatalogError):
                adapter.project_races(summary, features, outcomes)

    def test_invalid_probabilities_arrays_and_interval_levels_fail_closed(self):
        for kind in ("probability", "bool", "mean", "nan", "quantile-order", "quantile-level", "shape"):
            summary, features, outcomes = fixture()
            if kind == "probability":
                summary["dem_win_probability"][0] = 1.1
            elif kind == "bool":
                summary["dem_win_probability"][0] = True
            elif kind == "mean":
                summary["mean_dem_share"][0] = -0.1
            elif kind == "nan":
                summary["share_quantiles"][0][0] = float("nan")
            elif kind == "quantile-order":
                summary["share_quantiles"][0][0] = 0.99
            elif kind == "quantile-level":
                summary["quantile_levels"][0] = 0.05
            elif kind == "shape":
                summary["share_quantiles"][0].pop()
            with self.subTest(kind=kind), self.assertRaises(CatalogError):
                adapter.project_races(summary, features, outcomes)

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned scientific checkout unavailable")
    def test_source_replay_reproduces_all_result_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter.export(SOURCE, Path(directory))
            for generated in Path(directory).glob("*.json"):
                self.assertEqual(generated.read_bytes(), (DATA / generated.name).read_bytes())

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned scientific checkout unavailable")
    def test_changed_feature_or_outcome_source_pin_rejects_publication(self):
        original = adapter.pinned
        for suffix in ("target-2018.csv", "outcomes-SEPARATE.csv"):

            def changed(root, reference):
                if reference["path"].endswith(suffix):
                    reference = {**reference, "sha256": "0" * 64}
                return original(root, reference)

            with patch.object(adapter, "pinned", side_effect=changed):
                with self.assertRaisesRegex(CatalogError, "fingerprint mismatch"):
                    adapter.build_results(SOURCE)


if __name__ == "__main__":
    unittest.main()
