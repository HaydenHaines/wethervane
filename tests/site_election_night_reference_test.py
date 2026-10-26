"""Reviewed model-implied scorecard reference import; no fitting or outcome grading."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.export_election_night_reference as adapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
BUNDLE = ROOT / "public/data/explorer/current-polling.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class ElectionNightReferenceTests(unittest.TestCase):
    def test_pin_rejects_changed_bytes_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            source = root / "reference.json"
            source.write_text("{}")
            with self.assertRaisesRegex(adapter.ReferenceExportError, "hash mismatch"):
                adapter.pinned(root, {"path": "reference.json", "sha256": "0" * 64})
            target = Path(outside) / "reference.json"
            target.write_text("{}")
            (root / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(adapter.ReferenceExportError, "Unsafe source path"):
                adapter.pinned(root, {"path": "linked.json", "sha256": adapter.sha(target)})
            with self.assertRaisesRegex(adapter.ReferenceExportError, "Unsafe external review source"):
                adapter.pinned(root, {"path": str(target), "sha256": adapter.sha(target)})

    @unittest.skipUnless(SOURCE.is_dir() and BUNDLE.is_file(), "Pinned Senate research and bundle unavailable")
    def test_actual_reviewed_reference_and_publication_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "references.json"
            manifest = Path(folder) / "publication.json"
            result = adapter.export(SOURCE, BUNDLE, output, manifest)
            self.assertEqual(len(result["entries"]), 1)
            entry = result["entries"][0]
            self.assertEqual(entry["chamber"], "senate")
            self.assertEqual(entry["runId"], "current-polling")
            self.assertEqual(entry["resultSha256"], adapter.PUBLIC_BUNDLE_SHA256)
            self.assertEqual(entry["forecastCutoffAt"], adapter.POLL_CUTOFF)
            self.assertEqual(entry["structuralCutoffAt"], adapter.STRUCTURAL_CUTOFF)
            self.assertEqual(entry["sourcePins"]["referenceSha256"], adapter.REFERENCE_SHA256)
            self.assertEqual(entry["sourcePins"]["reviewSha256"], adapter.REVIEW_SHA256)
            self.assertFalse(entry["evaluationLock"])
            self.assertFalse(entry["observedResultsConsumed"])
            self.assertEqual(entry["reference"]["drawIdentity"]["shape"], [4, 10000])
            self.assertEqual(entry["reference"]["drawIdentity"]["weight_per_row"], 1 / 40000)
            self.assertEqual(len(entry["reference"]["cohort"]), 35)
            self.assertEqual(set(entry["reference"]["distributions"]), adapter.METRICS)
            for row in entry["reference"]["cohort"]:
                self.assertTrue(row["included_in_native_diagnostic_cohort"])
                self.assertEqual(row["margin_sign"], "first_candidate_minus_second_candidate")
                self.assertGreaterEqual(row["expected_absolute_margin_error_pp"], 0)
                self.assertIn(row["modal_pair_winner_id"], [row["first_candidate_id"], row["second_candidate_id"]])
            self.assertEqual(read(output), result)
            self.assertEqual(read(manifest), {"schemaVersion": 1, "href": "/data/election-night/references.json",
                                              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                              "editions": [{"chamber": "senate", "runId": "current-polling",
                                                            "resultSha256": adapter.PUBLIC_BUNDLE_SHA256}]})

    @unittest.skipUnless(SOURCE.is_dir() and BUNDLE.is_file(), "Pinned Senate research and bundle unavailable")
    def test_changed_bundle_hash_and_wrong_model_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            wrong = Path(folder) / "current-polling.json"
            data = read(BUNDLE)
            data["run"]["id"] = "current-error-low"
            wrong.write_text(json.dumps(data))
            with self.assertRaisesRegex(adapter.ReferenceExportError, "Public forecast bundle hash changed"):
                adapter.build(SOURCE, wrong)
            with patch.object(adapter, "PUBLIC_BUNDLE_SHA256", adapter.sha(wrong)):
                with self.assertRaisesRegex(adapter.ReferenceExportError, "Public forecast edition/provenance mismatch"):
                    adapter.build(SOURCE, wrong)

    @unittest.skipUnless(SOURCE.is_dir() and BUNDLE.is_file(), "Pinned Senate research and bundle unavailable")
    def test_reviewed_cohort_row_and_draw_identity_mismatch_rejected(self):
        original_read = adapter.read
        reference_path = SOURCE / adapter.REFERENCE_REL
        for mutate, message in (
            (lambda r: r["cohort"][0].__setitem__("first_candidate_id", "wrong-candidate"), "pair rule/denominator/support mismatch"),
            (lambda r: r["draw_identity"].__setitem__("shape", [1, 40000]), "Original shared-row identity"),
        ):
            def changed(path):
                result = original_read(path)
                if path == reference_path:
                    result = copy.deepcopy(result)
                    mutate(result)
                    result["cohort_sha256"] = hashlib.sha256(json.dumps(result["cohort"], sort_keys=True,
                                                                   separators=(",", ":")).encode()).hexdigest()
                return result
            with patch.object(adapter, "read", side_effect=changed):
                with self.assertRaisesRegex(adapter.ReferenceExportError, message):
                    adapter.build(SOURCE, BUNDLE)


if __name__ == "__main__":
    unittest.main()
