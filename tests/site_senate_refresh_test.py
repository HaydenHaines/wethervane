"""Pinned October 26 Senate refresh importer checks; no fitting or publication."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import scripts.export_senate_refresh as adapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("BALLOTMANCY_RESEARCH_ROOT", "/home/hayden/projects/wethervane-redesign-20261006/integration"))
PUBLIC = ROOT / "public/data/explorer"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RefreshContractTests(unittest.TestCase):
    def test_pin_rejects_changed_native_bytes_and_parent_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "forecast.json").write_text('{"target_id":"wrong"}')
            with self.assertRaisesRegex(adapter.RefreshImportError, "hash mismatch"):
                adapter.pinned(root, {"path": "forecast.json", "sha256": "0" * 64})
            with self.assertRaisesRegex(adapter.RefreshImportError, "Unsafe source"):
                adapter.pinned(root, {"path": "../forecast.json", "sha256": "0" * 64})
            with tempfile.TemporaryDirectory() as outside:
                target = Path(outside) / "forecast.json"
                target.write_text("{}")
                (root / "linked-forecast.json").symlink_to(target)
                with self.assertRaisesRegex(adapter.RefreshImportError, "Unsafe source"):
                    adapter.pinned(root, {"path": "linked-forecast.json", "sha256": adapter.sha(target)})

    @unittest.skipUnless((SOURCE / adapter.HANDOFF).is_file(), "Pinned native handoff unavailable")
    def test_wrong_target_rejected_before_projection(self):
        handoff = read(SOURCE / adapter.HANDOFF)
        for arm in (handoff["arms"][0], handoff["arms"][5]):
            files = arm["files"]
            forecast = read(SOURCE / files["forecast.json"]["path"])
            spec = read(SOURCE / files["run-specification.json"]["path"])
            receipt = read(SOURCE / files["receipt.json"]["path"])
            reference = read(SOURCE / arm["unchanged_fundamentals_only"]["path"])
            forecast["target_id"] = "wrong-target"
            with self.assertRaisesRegex(adapter.RefreshImportError, "target mismatch"):
                adapter._validate_identity(arm, forecast, spec, receipt, reference)

    @unittest.skipUnless((SOURCE / adapter.HANDOFF).is_file(), "Pinned native handoff unavailable")
    def test_exact_ten_arms_joint_coherence_and_archive_preservation(self):
        # Source and site archive hashes bracket the staged export. It writes no
        # public data, and all numerical diagnostics derive from native joint rows.
        old_entries = (read(PUBLIC / "current-index.json")
                       + read(PUBLIC / "current-comparisons-index.json")
                       + read(PUBLIC / "history-index.json"))
        preserved = {entry["id"]: adapter.sha(PUBLIC / Path(entry["href"]).name)
                     for entry in old_entries if entry["id"] == "current-fundamentals"
                     or entry["id"].endswith("-f") or entry["historical"]}
        self.assertEqual(len(preserved), 38)  # six fundamentals-only + 32 historical editions
        with tempfile.TemporaryDirectory() as folder:
            paths = adapter.export(SOURCE, Path(folder))
            self.assertEqual(len(paths), 11)
            index = read(Path(folder) / "senate-refresh-index.json")
            self.assertEqual(len(index), 10)
            self.assertEqual({entry["id"] for entry in index},
                             {spec[0] for spec in adapter.EXPECTED_ARMS.values()})
            self.assertEqual({entry["dateId"] for entry in index}, {"2026-10-26"})
            self.assertEqual({entry["strictAsOf"] for entry in index}, {False})
            handoff = read(SOURCE / adapter.HANDOFF)
            by_arm = {arm["arm"]: arm for arm in handoff["arms"]}
            for entry in index:
                output_path = Path(folder) / Path(entry["href"]).name
                self.assertEqual(adapter.sha(output_path), entry["sha256"])
                bundle = read(output_path)
                run = bundle["run"]
                arm = next(a for a, spec in adapter.EXPECTED_ARMS.items() if spec[0] == run["id"])
                source = by_arm[arm]
                self.assertEqual(run["targetId"], source["target_id"])
                self.assertEqual(run["components"]["base"], adapter.EXPECTED_ARMS[arm][1])
                self.assertEqual(run["simulationCount"], source["retained_joint_draws"])
                self.assertEqual(run["inputCutoffs"][0]["at"], adapter.STRUCTURAL_CUTOFF)
                self.assertEqual(run["inputCutoffs"][1]["at"], adapter.POLL_CUTOFF)
                self.assertEqual(len(bundle["races"]), 35)
                self.assertEqual({s["id"] for s in bundle["scenarios"]}, adapter.SCENARIOS)
                self.assertTrue(any("covariance" in x.lower() for x in bundle["limitations"]))
                self.assertTrue(any("Superseded polling edition:" in x for x in bundle["limitations"]))
                for race in bundle["races"]:
                    self.assertEqual(len(race["notes"]), 3)
                    self.assertTrue(any("principal" in note.lower() for note in race["notes"]))
                    share_note = race["notes"][2]
                    if race["outcomes"][0]["party"] == "I":
                        self.assertIn("independent, not a Democrat", share_note)
                        self.assertIn("opposition_proxy", share_note)
                    else:
                        self.assertIn("Democratic candidate's share", share_note)
                        self.assertIn("democratic_two_party", share_note)
                for scenario in bundle["scenarios"]:
                    for party in ("D", "R", "U"):
                        self.assertAlmostEqual(sum(scenario["seats"][party]["pmf"]), 1.0)
                for scenario in bundle["tipping"]["scenarios"].values():
                    for party in ("D", "R"):
                        self.assertAlmostEqual(sum(scenario[party]["frequencyByRace"].values()), 1.0)
            # Independently compare two different row layouts against staged race
            # and party seat summaries, in addition to the importer's all-arm checks.
            for arm in ("standard", "recent_calendar_history"):
                source = by_arm[arm]
                bundle = read(Path(folder) / f"{adapter.EXPECTED_ARMS[arm][0]}.json")
                with np.load(SOURCE / source["files"]["joint-draws.npz"]["path"], allow_pickle=False) as joint:
                    shares = joint["shares"].reshape(-1, 35)
                    wins = joint["first_wins"].reshape(-1, 35)
                self.assertAlmostEqual(bundle["races"][0]["outcomes"][0]["share"]["mean"],
                                       float(shares[:, 0].mean()))
                self.assertAlmostEqual(bundle["races"][0]["outcomes"][0]["winProbability"],
                                       float(wins[:, 0].mean()))
                with np.load(SOURCE / source["files"]["chamber-draws.npz"]["path"], allow_pickle=False) as chamber:
                    draw_d = chamber["scenario:campaign_pledge_unaligned:D"]
                public_d = next(s for s in bundle["scenarios"] if s["id"] == "campaign_pledge_unaligned")["seats"]["D"]
                self.assertAlmostEqual(public_d["expectedSeats"], float(draw_d.mean()))
        self.assertEqual(preserved, {entry["id"]: adapter.sha(PUBLIC / Path(entry["href"]).name)
                                     for entry in old_entries if entry["id"] in preserved})

    def test_output_guard_rejects_live_explorer_directory(self):
        with patch.object(adapter, "build", side_effect=AssertionError("should not build")):
            with self.assertRaisesRegex(adapter.RefreshImportError, "Stage to an isolated"):
                adapter.export(SOURCE, PUBLIC)


if __name__ == "__main__":
    unittest.main()
