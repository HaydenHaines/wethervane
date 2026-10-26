"""Saved comparison integration tests; never execute inference."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import scripts.export_current_comparisons as adapter

from scripts.export_current_comparisons import ComparisonError, LABELS, LAW, TARGET, export, pinned
from scripts.export_composition_capabilities import _cutoffs, _evidence

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public/data/explorer"
SOURCE = Path(
    os.environ.get("BALLOTMANCY_RESEARCH_ROOT", "/home/hayden/projects/wethervane-redesign-20261006/integration")
)
read = lambda path: json.loads(path.read_text())


class SavedComparisonTests(unittest.TestCase):
    def test_complete_ten_run_coverage_and_exact_component_bindings(self):
        entries = read(DATA / "current-comparisons-index.json")
        self.assertEqual(
            {(e["components"]["base"], e["components"]["polling"]) for e in entries},
            {(base, poll) for base in LABELS for poll in ("none", "standard")},
        )
        self.assertEqual(len(entries), 10)
        for entry in entries:
            raw = (DATA / Path(entry["href"]).name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
            bundle = json.loads(raw)
            self.assertEqual(entry["targetId"], TARGET)
            self.assertEqual(entry["lawId"], LAW)
            self.assertEqual(entry["components"]["uncertainty"], "standard")
            self.assertEqual(entry["researchEvidence"]["forecastSkill"], "unobserved_2026")
            self.assertEqual(_evidence(entry)["status"], "experimental")
            self.assertEqual(_cutoffs(entry), entry["inputCutoffs"])
            self.assertEqual(len(entry["inputCutoffs"]), 2 if entry["components"]["polling"] == "standard" else 1)
            self.assertEqual(entry["inputCutoffs"][0]["at"], "2026-10-20T15:21:23Z")
            if len(entry["inputCutoffs"]) == 2:
                self.assertEqual(entry["inputCutoffs"][1]["at"], "2026-10-26T15:03:57.212405+00:00")
                self.assertEqual(entry["dateId"], "2026-10-26")
            self.assertEqual(
                [r["state"] for r in bundle["races"] if r["outcomes"][0]["party"] == "I"], ["ID", "MT", "NE", "SD"]
            )
            for scenario in bundle["tipping"]["scenarios"].values():
                for party in ("D", "R"):
                    self.assertAlmostEqual(sum(scenario[party]["frequencyByRace"].values()), 1)
            self.assertTrue(any("borrowed" in note for note in bundle["limitations"]))

    def test_original_archive_manifest_and_default_are_preserved(self):
        registry = read(DATA / "registry.json")
        self.assertEqual(len(registry["runs"]), 48)
        self.assertEqual(registry["defaults"]["2026"], "current-polling")
        for name in ("current-index.json", "history-index.json"):
            for entry in read(DATA / name):
                self.assertIn(entry, registry["runs"])
                self.assertEqual(
                    hashlib.sha256((DATA / Path(entry["href"]).name).read_bytes()).hexdigest(), entry["sha256"]
                )

    def test_source_path_and_hash_admission(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "source.json").write_text("{}")
            with self.assertRaisesRegex(ComparisonError, "hash mismatch"):
                pinned(root, {"path": "source.json", "sha256": "0" * 64})
            with self.assertRaisesRegex(ComparisonError, "Unsafe source"):
                pinned(root, {"path": "../source.json", "sha256": "0" * 64})

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned research checkout not available")
    def test_changed_cutoff_and_failed_diagnostics_are_rejected_before_projection(self):
        entry = read(SOURCE / adapter.PACKET / "current-comparisons.json")[0]
        original_read = adapter.read
        for filename, field, value, message in (
            ("forecast.json", "input_cutoff_utc", "2026-10-21T00:00:00Z", "Cutoff mismatch"),
            ("numerical-gates.json", "status", "failed", "Numerical gates"),
        ):
            changed_path = SOURCE / entry["files"][filename]["path"]

            def substituted(path):
                result = original_read(path)
                if path == changed_path:
                    result[field] = value
                return result

            with patch.object(adapter, "read", side_effect=substituted):
                with self.assertRaisesRegex(ComparisonError, message):
                    adapter.build_one(SOURCE, entry, {})

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned research checkout not available")
    def test_replaying_original_source_projection_reproduces_preserved_bytes(self):
        # Re-derives all tipping frequencies and checks all source PMFs/quantiles
        # directly from existing joint arrays, without random draws or fitting.
        with tempfile.TemporaryDirectory() as d:
            export(SOURCE, Path(d))
            for generated in Path(d).glob("*.json"):
                archived = DATA / "editions/2026-10-20" / generated.name
                original = archived if archived.exists() else DATA / generated.name
                self.assertEqual(generated.read_bytes(), original.read_bytes(), generated.name)


if __name__ == "__main__":
    unittest.main()
