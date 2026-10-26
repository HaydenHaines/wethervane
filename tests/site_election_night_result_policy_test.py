from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.export_election_night_result_policy as adapter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
OUTPUT = ROOT / "data/election-night-senate-result-policy.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class SenateResultPolicyExportTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE.is_dir(), "Pinned Senate research source is unavailable")
    def test_actual_policy_freeze_and_both_forecasts_bind_exactly(self):
        value = adapter.build(SOURCE, ROOT)
        self.assertEqual(value["schemaVersion"], 1)
        self.assertEqual(value["status"], "preparation_not_evaluation_lock")
        self.assertFalse(value["evaluationLock"])
        self.assertFalse(value["observedResultsUsed"])
        self.assertFalse(value["actualGradesAssigned"])
        self.assertEqual(value["sourceHashes"]["policySha256"], adapter.POLICY_SHA256)
        self.assertEqual(value["sourceHashes"]["mappingSha256"], adapter.MAPPING_SHA256)
        self.assertEqual(value["sourceHashes"]["operatorSha256"], adapter.OPERATOR_SHA256)
        self.assertEqual(value["sourceHashes"]["freezeManifestSha256"], adapter.FREEZE_SHA256)
        self.assertEqual(value["sourceHashes"]["FPublicSha256"], adapter.F_PUBLIC_SHA256)
        self.assertEqual(value["sourceHashes"]["FPPublicSha256"], adapter.FP_PUBLIC_SHA256)
        self.assertEqual(len(value["sourceFiles"]), 20)
        self.assertEqual(len(value["fixedCohorts"]["full35"]), 35)
        self.assertEqual(len(value["fixedCohorts"]["pluralityCore32"]), 32)
        self.assertEqual(len(value["fixedCohorts"]["conditional3"]), 3)
        self.assertEqual([row["raceId"] for row in value["rows"]], value["fixedCohorts"]["full35"])
        self.assertEqual([row["raceId"] for row in value["rows"] if row["coreMember"]], value["fixedCohorts"]["pluralityCore32"])
        self.assertEqual([row["raceId"] for row in value["rows"] if not row["coreMember"]], value["fixedCohorts"]["conditional3"])
        self.assertEqual({row["operatorFamily"] for row in value["rows"] if not row["coreMember"]},
                         {"rcv_locked_final_two", "ga_certified_decisive_event"})
        for row in value["rows"]:
            self.assertIsNone(row["officialFeedCandidateIds"])
            self.assertIsNone(row["officialFeedContestId"])
            self.assertEqual(row["crosswalkStatus"], "requires_separate_review; no name/party fuzzy or replacement mapping")
            self.assertTrue(row["officialAuthorityUrl"] is None or row["officialAuthorityUrl"].startswith("https://"))
            self.assertTrue(row["officialResultsUrl"] is None or row["officialResultsUrl"].startswith("https://"))
            self.assertTrue(all(url.startswith("https://") for url in row["firstCandidateRosterSourceUrls"]))
            self.assertTrue(all(url.startswith("https://") for url in row["secondCandidateRosterSourceUrls"]))
            self.assertIn("first+second", row["denominator"])
        self.assertTrue(any("Alabama" in gap or "Georgia" in gap for gap in value["sourceGaps"]["resultRouteRegistry"]))
        self.assertEqual(value["forecastBindings"]["F"]["nativeForecastSha256"], adapter.F_NATIVE_SHA256)
        self.assertEqual(value["forecastBindings"]["FP"]["nativeForecastSha256"], adapter.FP_NATIVE_SHA256)
        self.assertEqual(value["rows"][0]["operatorFamily"], "rcv_locked_final_two")
        self.assertEqual(next(row for row in value["rows"] if row["state"] == "GA")["runoffDate"], "2026-12-01")
        self.assertEqual(read(OUTPUT), value)

    def test_freeze_manifest_verifies_all_declared_bytes_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payloads = {f"evidence/{i:02d}.json": f"payload {i}\n".encode() for i in range(16)}
            files = []
            for name, data in payloads.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            manifest = {"schema": "SenatePreResultPolicyFreezeManifestV1", "files": files}
            verified = adapter._verify_manifest_files(root, manifest)
            self.assertEqual(len(verified), 16)
            (root / "evidence/00.json").write_bytes(b"changed\n")
            with self.assertRaisesRegex(adapter.ResultPolicyExportError, "Source hash mismatch"):
                adapter._verify_manifest_files(root, manifest)
            with self.assertRaisesRegex(adapter.ResultPolicyExportError, "Unsafe source path"):
                adapter._safe_source(root, "../escape.json")

    @unittest.skipUnless(SOURCE.is_dir(), "Pinned Senate research source is unavailable")
    def test_pinned_mapping_mutations_and_foreign_feed_ids_fail_closed(self):
        original_read = adapter.read
        mapping_path = SOURCE / adapter.MAPPING_REL
        cases = [
            (lambda mapping: mapping["races"][0].__setitem__("first_candidate_id", "wrong-person"), "locked pair malformed|candidate ID/name/party differs"),
            (lambda mapping: mapping["races"][0].__setitem__("official_future_feed_candidate_ids", ["invented-feed-id"]), "future-feed or pre-result status changed"),
            (lambda mapping: mapping["fixed_cohorts"].__setitem__("plurality_core_32", mapping["fixed_cohorts"]["plurality_core_32"][1:]), "Fixed 35/32/3 cohorts"),
        ]
        for mutate, message in cases:
            def changed(path):
                value = original_read(path)
                if path == mapping_path:
                    value = copy.deepcopy(value)
                    mutate(value)
                return value
            with self.subTest(message=message), patch.object(adapter, "read", side_effect=changed):
                with self.assertRaisesRegex(adapter.ResultPolicyExportError, message):
                    adapter.build(SOURCE, ROOT)

    @unittest.skipUnless(SOURCE.is_dir(), "Pinned Senate research source is unavailable")
    def test_unknown_policy_packet_and_changed_public_bundle_are_rejected(self):
        with self.assertRaisesRegex(adapter.ResultPolicyExportError, "Unexpected integration source root"):
            adapter.build(SOURCE.parent, ROOT)
        with tempfile.TemporaryDirectory() as folder:
            wrong_repo = Path(folder)
            data_dir = wrong_repo / "public/data/explorer"
            data_dir.mkdir(parents=True)
            for name in ["registry.json", "current-fundamentals.json", "current-polling.json"]:
                (data_dir / name).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(adapter.ResultPolicyExportError, "Current F/FP public forecast or registry bytes changed"):
                adapter.build(SOURCE, wrong_repo)


if __name__ == "__main__":
    unittest.main()
