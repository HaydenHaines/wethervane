"""Metadata-only provider contract; no inference or live result activation."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.export_component_catalog import (
    CatalogError,
    INTERFACE_SHA256,
    READY_SHA256,
    ROLE_COPY,
    configuration_digest,
    export_catalog,
    pinned,
    request_digest,
    validate_option_schema,
    validate_ready,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.environ.get("BALLOTMANCY_RESEARCH_ROOT", "/home/hayden/projects/wethervane-redesign-20261006/integration")
)
CATALOG = json.loads((ROOT / "data/model-components-v2.json").read_text())


class ComponentCatalogTests(unittest.TestCase):
    def test_all_roles_preserve_provider_identity_options_and_no_executor(self):
        self.assertEqual([r["role"] for r in CATALOG["roles"]], list(ROLE_COPY))
        self.assertEqual(CATALOG["activation"], "metadata_only")
        self.assertEqual(CATALOG["execution"]["available"], False)
        self.assertEqual(CATALOG["execution"]["schedulerAvailable"], False)
        self.assertEqual(CATALOG["sourceFingerprints"]["interfaceReceipt"], INTERFACE_SHA256)
        self.assertEqual(CATALOG["sourceFingerprints"]["readyReceipt"], READY_SHA256)
        self.assertEqual(CATALOG["defaultPresetId"], "best_guess_polling_20261020_v2")
        roles = {r["role"]: r for r in CATALOG["roles"]}
        self.assertEqual(len(roles["environment"]["providers"]), 3)
        self.assertEqual(len(roles["fundamentals"]["providers"]), 3)
        self.assertEqual(len(roles["measurement"]["providers"]), 1)
        self.assertEqual(sum(not r["providers"] for r in CATALOG["roles"]), 5)
        for provider in [p for role in CATALOG["roles"] for p in role["providers"]]:
            self.assertEqual(provider["version"], "v1")
            self.assertEqual(provider["siteAvailability"], "metadata_only")
            validate_option_schema(provider["options_schema"])
        candidate = next(p for p in roles["fundamentals"]["providers"] if p["default_instance_id"] == "candidate")
        self.assertEqual(candidate["provider_id"], "aggregate_candidate_incumbency_v1")
        self.assertEqual(
            candidate["options_schema"]["properties"]["candidate"],
            {"type": "string", "enum": ["signed_incumbency", "off"], "default": "signed_incumbency"},
        )

    def test_ordinary_2018_support_and_evidence_are_not_complete_chamber_forecasts(self):
        self.assertEqual(len(CATALOG["savedExamples"]), 2)
        for context in CATALOG["supportedContexts"]:
            self.assertEqual(context["year"], 2018)
            self.assertEqual(context["informationCutoff"], "2018-10-23T00:00:00Z")
            self.assertEqual(len(set(context["raceIds"])), 30)
            self.assertEqual(context["outputs"]["chamberControl"], False)
            self.assertEqual(context["outputs"]["tipping"], False)
            self.assertFalse(context["strictAsOf"])
        for example in CATALOG["savedExamples"]:
            self.assertEqual(example["activation"], "not_activated")
            self.assertEqual(example["sourceLifecycle"], "ready")
            self.assertEqual(example["evidenceStatus"], "unassessed")
            self.assertEqual(example["numericalStatus"], "passed")
            self.assertTrue(example["experimental"])
            self.assertEqual(example["quantileLevels"], [0.025, 0.1, 0.5, 0.9, 0.975])
            self.assertEqual(example["shareQuantileShape"], [5, 30])
            self.assertNotIn("summary", example)
            self.assertNotIn("data_bundle", example["requestTemplate"])
        for example in CATALOG["lifecycleExamples"]:
            self.assertEqual(example["sourceLifecycle"], "requires_computation")
            self.assertIsNone(example["displayedResultId"])
            self.assertFalse(example["scheduled"])
            candidate = next(c for c in example["requestTemplate"]["selections"] if c["instance_id"] == "candidate")
            self.assertEqual(candidate["options"], {"candidate": "off"})

    def test_sanitized_metadata_contains_no_source_paths_or_executable_code(self):
        serialized = json.dumps(CATALOG)
        for value in ("/home/", ".redesign-runtime/", "research/", "src/", "predictions.npz", "function(", "<script"):
            self.assertNotIn(value, serialized)
        self.assertEqual(len(CATALOG["sourceFingerprints"]["schemas"]), 6)

    def test_unreviewed_option_editor_shapes_fail_closed(self):
        valid = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"choice": {"type": "string", "enum": ["on", "off"], "default": "on"}},
        }
        validate_option_schema(valid)
        for change in (
            {"additionalProperties": True},
            {"$ref": "https://example.invalid/code"},
            {"properties": {"choice": {"type": "string", "enum": ["on"], "default": "off"}}},
        ):
            with self.assertRaises(CatalogError):
                validate_option_schema({**valid, **change})

    def test_unsafe_or_changed_source_references_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").write_text("changed")
            for reference in ({"path": "../source", "sha256": "0" * 64}, {"path": "source", "sha256": "0" * 64}):
                with self.assertRaises(CatalogError):
                    pinned(root, reference)

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned research checkout is unavailable")
    def test_export_is_exact_replay_of_pinned_source_metadata(self):
        self.assertEqual(export_catalog(SOURCE), CATALOG)

    @unittest.skipUnless((SOURCE / "research").is_dir(), "Pinned research checkout is unavailable")
    def test_ready_identity_and_evidence_changes_are_rejected(self):
        base = SOURCE / "research/fundamentals-iteration-20261020"
        ready = json.loads((base / "frontend-contract-root-v2/ready-v1/F-ready.json").read_text())
        request = ready["configuration"]["request"]
        arm = json.loads((base / "aggregate-driver-astra/real-execution-v1/result-index.json").read_text())["arms"][0]
        validate_ready(ready, request, arm)
        self.assertEqual(request_digest(request), ready["request_sha256"])
        self.assertEqual(configuration_digest(ready["configuration"]), arm["configuration_sha256"])
        for field, value in [
            ("configuration_sha256", "0" * 64),
            ("evidence_status", "validated"),
            ("numerical_status", "failed"),
        ]:
            changed = copy.deepcopy(ready)
            changed["record"][field] = value
            with self.assertRaises(CatalogError):
                validate_ready(changed, request, arm)


if __name__ == "__main__":
    unittest.main()
