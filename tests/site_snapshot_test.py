"""Focused checks for the public static forecast snapshot export."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_site_snapshot import (
    EXPECTED_TARGETS,
    SOURCE_FILES,
    SnapshotError,
    _probability,
    build_snapshot,
    export_snapshot,
)


def synthetic_source(model_key: str) -> dict:
    scenario_id = "campaign_pledge_unaligned"
    races = []
    for index in range(35):
        state = f"S{index:02d}"
        race = {
            "race_id": f"2026:{state}:senate:regular",
            "state": state,
            "seat_id": f"US-SEN-{state}-CLASS-1",
            "first_contender": {
                "id": f"contender:{state}:candidate-d",
                "name": f"Candidate D {index}",
                "party": "D",
                "incumbent": index % 2 == 0,
                "source_support": {"source_urls": ["https://example.org/candidate-d"]},
                "source_path": ".redesign-runtime/private/source.json",
            },
            "second_contender": {
                "id": f"contender:{state}:candidate-r",
                "name": f"Candidate R {index}",
                "party": "R",
                "incumbent": False,
                "source_support": {"source_urls": ["https://example.org/candidate-r"]},
            },
            "first_win_probability": 0.4,
            "second_win_probability": 0.6,
            "first_share_mean": 0.48,
            "first_share_quantiles": {"p05": 0.3, "p50": 0.48, "p95": 0.66},
            "rule_proxy": "Synthetic two-candidate proxy.",
            "excluded_options_assumption": "Synthetic omitted-option assumption.",
        }
        if model_key == "polling":
            race["admitted_poll_observations"] = index % 4
        races.append(race)

    distribution = {
        "expected_seats": 50.5,
        "seat_quantiles": {"p05": 48, "p50": 51, "p95": 53},
        "pmf": [0.5, 0.5],
    }
    chamber = {f"party_{party}": distribution for party in ("D", "R", "I")}
    chamber[f"scenario:{scenario_id}:D"] = distribution
    chamber[f"scenario:{scenario_id}:R"] = distribution
    source = {
        "target_id": EXPECTED_TARGETS[model_key],
        "input_cutoff_utc": "2026-10-20T16:21:44Z",
        "races": races,
        "chamber": chamber,
        "control_scenarios": {
            scenario_id: {
                "democratic_majority_probability": 0.3,
                "republican_majority_probability": 0.6,
                "unresolved_organizing_probability": 0.1,
            }
        },
        "caucus_scenarios": [
            {"id": scenario_id, "description": "Synthetic unaligned scenario."}
        ],
        "limitations": (
            {"current_scope": "Synthetic test only."}
            if model_key == "polling"
            else ["Synthetic test only."]
        ),
    }
    if model_key == "polling":
        source["execution_completed_utc"] = "2026-10-20T16:27:31Z"
    else:
        source["issuance_utc"] = "2026-10-20T16:26:00Z"
    return source


def write_synthetic_sources(root: Path) -> None:
    for model_key, relative_path in SOURCE_FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(synthetic_source(model_key)), encoding="utf-8")


class SiteSnapshotTest(unittest.TestCase):
    def test_snapshot_preserves_aligned_races_and_scenario_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp)
            write_synthetic_sources(source_root)
            snapshot, provenance = build_snapshot(source_root)
        polling = snapshot["models"]["polling"]
        fundamentals = snapshot["models"]["fundamentals"]

        self.assertEqual(snapshot["schemaVersion"], 1)
        self.assertEqual(len(polling["races"]), 35)
        self.assertEqual(len(fundamentals["races"]), 35)
        self.assertEqual(
            {race["id"] for race in polling["races"]},
            {race["id"] for race in fundamentals["races"]},
        )
        self.assertTrue(all("pollCount" in race for race in polling["races"]))
        self.assertTrue(all("pollCount" not in race for race in fundamentals["races"]))
        self.assertEqual(
            set(polling["chamberDistributions"]["party"]), {"D", "R", "I"}
        )
        self.assertEqual(
            set(polling["chamberDistributions"]["caucus"]),
            {"campaign_pledge_unaligned"},
        )
        self.assertEqual(set(provenance["sources"]), {"polling", "fundamentals"})

        for race in polling["races"] + fundamentals["races"]:
            self.assertAlmostEqual(
                race["firstWinProbability"] + race["secondWinProbability"], 1.0
            )
            self.assertTrue(all(candidate["sourceUrls"] for candidate in race["candidates"]))

        public_json = json.dumps(snapshot)
        self.assertNotIn("source_support", public_json)
        self.assertNotIn("source_path", public_json)
        self.assertNotIn(".redesign-runtime", public_json)

    def test_candidate_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp)
            write_synthetic_sources(source_root)
            polling_path = source_root / SOURCE_FILES["polling"]
            polling = json.loads(polling_path.read_text(encoding="utf-8"))
            polling["races"][0]["first_contender"]["name"] = "Mismatched name"
            polling_path.write_text(json.dumps(polling), encoding="utf-8")
            with self.assertRaises(SnapshotError):
                build_snapshot(source_root)

    def test_export_records_hashes_for_sources_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            write_synthetic_sources(source_root)
            forecast_path, provenance_path = export_snapshot(source_root, root / "out")
            forecast_bytes = forecast_path.read_bytes()
            forecast = json.loads(forecast_bytes)
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(
                provenance["snapshotSha256"], hashlib.sha256(forecast_bytes).hexdigest()
            )
            self.assertEqual(
                forecast["sources"]["pollingSha256"],
                provenance["sources"]["polling"]["sha256"],
            )
            self.assertEqual(
                forecast["sources"]["fundamentalsSha256"],
                provenance["sources"]["fundamentals"]["sha256"],
            )

    def test_probability_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(SnapshotError):
            _probability(1.01, "test probability")
        with self.assertRaises(SnapshotError):
            _probability(float("nan"), "test probability")


if __name__ == "__main__":
    unittest.main()
