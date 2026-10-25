"""Boundary checks for the read-only owner count inspection adapter.

Invented worker receipts here test transport validation only. Native ready-result
validation remains the owner's CompositionApplication/CountDriver responsibility.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import native_selection as n


def invented_spec():
    sources = {name: {"path": path, "sha256": digest} for name, (path, digest) in n.NATIVE_MODULE_PINS.items()}
    sources.update({name: {"path": name + ".json", "sha256": "a" * 64}
                    for name in ("request", "bundle", "registration")})
    return {"schema": n.SCHEMA, "ownerRoot": "/invented/owner", "pythonExecutable": "/invented/python",
            "pythonSha256": "a" * 64, "sources": sources,
            "expectedRequestSha256": "b" * 64, "expectedConfigurationSha256": "c" * 64,
            "artifactStoreRoot": "/invented/store", "artifactRef": None}


def invented_receipt(spec):
    task = {"targetId": n.TARGET, "outputContractId": n.OUTPUT,
            "predictionEvents": [{"id": "invented-event", "state": "48", "cycle": 2026, "date": "2026-11-03"}],
            "predictionContests": [{"id": "invented-contest", "eventId": "invented-event", "kind": "regular",
                                    "rule": "single_mark_plurality", "candidateIds": ["invented-D", "invented-R"],
                                    "noncandidateCategories": [{"id": "invented-blank", "kind": "blank"}]}]}
    observations = {"trainingEventIds": ["invented-training"],
                    "collections": [{"kind": "poll", "collectionId": "invented-polls", "rowCount": 1,
                                     "observationIdsSha256": "d" * 64}]}
    return {"schema": n.RECEIPT_SCHEMA, "adapterSha256": n._sha(Path(n.__file__).resolve().read_bytes()),
            "specSha256": n._sha(n._canonical(spec)), "status": "requires_computation",
            "request": {"sha256": spec["expectedRequestSha256"], "targetId": n.TARGET,
                        "outputContractId": n.OUTPUT, "informationCutoff": "2026-10-20T00:00:00Z",
                        "dataBundle": spec["sources"]["bundle"],
                        "selections": [{"instance_id": role, "provider_id": "invented_" + role,
                                        "version": "1", "options": {}} for role in n.ROLES]},
            "configuration": {"sha256": spec["expectedConfigurationSha256"],
                              "driverId": "scientific_count_application", "driverVersion": "1",
                              "lawSha256": "e" * 64, "graphSha256": "f" * 64,
                              "supportId": "1" * 64, "dependenciesSha256": "2" * 64,
                              "scientificPlan": {"path": "invented-plan.json", "sha256": "3" * 64},
                              "outputs": [{"contract_id": n.OUTPUT, "unit": "integer_counts",
                                           "denominator": "declared_event_participants_and_contest_category_universes"}]},
            "scope": {"kind": "actual_research", "problemIdentity": "1" * 64,
                      "upstreamIdentity": "4" * 64, "compositionIdentity": "5" * 64,
                      "parentProjectionDigest": "6" * 64, "parentIds": ["invented-parent"],
                      "parentMasses": [1.0], "scientificTargetIds": ["invented-target"]},
            "taskScope": task, "taskScopeSha256": n._sha(n._canonical(task)),
            "observationSupport": observations, "observationSupportSha256": n._sha(n._canonical(observations)),
            "sources": spec["sources"], "result": None,
            "qualification": {"independentReviewVerified": False, "empiricalForecastQualified": False,
                              "selectedResearchIncumbent": False, "publiclyAvailable": False}}


class NativeSelectionTests(unittest.TestCase):
    def test_source_pin_rejects_escape_and_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixed.json").write_bytes(b"{}")
            pin = {"path": "fixed.json", "sha256": n._sha(b"{}")}
            self.assertEqual(n._read_pin(root, pin), b"{}")
            (root / "fixed.json").write_bytes(b"{ } ")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                n._read_pin(root, pin)
            with self.assertRaisesRegex(ValueError, "Unsafe source path"):
                n._read_pin(root, {"path": "../elsewhere", "sha256": "0" * 64})

    def test_spec_rejects_unreviewed_module_and_extra_field_before_import(self):
        spec = invented_spec()
        with self.assertRaisesRegex(ValueError, "Unreviewed native module"):
            changed = copy.deepcopy(spec)
            changed["sources"]["driver"]["sha256"] = "0" * 64
            n._parse_spec(changed)
        with self.assertRaisesRegex(ValueError, "fields"):
            n._parse_spec({**spec, "selected": True})

    def test_invented_pending_receipt_preserves_no_result_or_qualification(self):
        spec = invented_spec()
        receipt = invented_receipt(spec)
        n._validate_worker_receipt(receipt, spec, receipt["adapterSha256"])
        for change, pattern in [
            (lambda row: row["qualification"].update(selectedResearchIncumbent=True), "qualification"),
            (lambda row: row["request"].update(targetId="other"), "request"),
            (lambda row: row["taskScope"]["predictionEvents"].append({"id": "forged"}), "physical task"),
            (lambda row: row["observationSupport"]["collections"].clear(), "observation support"),
            (lambda row: row.update(status="ready"), "gained a result"),
            (lambda row: row["sources"]["request"].update(sha256="0" * 64), "source identity"),
        ]:
            changed = copy.deepcopy(receipt)
            change(changed)
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                n._validate_worker_receipt(changed, spec, receipt["adapterSha256"])

    def test_invented_ready_transport_requires_full_native_result_shape(self):
        spec = invented_spec()
        spec["artifactRef"] = {"kind": n.RESULT_KIND, "artifact_id": "7" * 64}
        receipt = invented_receipt(spec)
        receipt["status"] = "ready"
        receipt["result"] = {"artifactRef": spec["artifactRef"],
                             "record": {"configuration_sha256": spec["expectedConfigurationSha256"],
                                        "numerical_status": "passed", "evidence_status": "unassessed"},
                             "summarySha256": "8" * 64, "terminalReceipt": {"path": "receipt.json", "sha256": "9" * 64},
                             "predictionRef": {"kind": "compact-senate-prediction-set-v1", "artifact_id": "a" * 64},
                             "commonDrawIdentity": "original_parent_chain_draw_shared_across_all_event_contest_counts",
                             "forecastQualifiedByNativeResult": False, "publicationAuthorizedByNativeResult": False}
        n._validate_worker_receipt(receipt, spec, receipt["adapterSha256"])
        changed = copy.deepcopy(receipt)
        changed["result"]["record"]["evidence_status"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "ready result evidence"):
            n._validate_worker_receipt(changed, spec, receipt["adapterSha256"])
        changed = copy.deepcopy(receipt)
        changed["result"]["record"]["configuration_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ready result evidence"):
            n._validate_worker_receipt(changed, spec, receipt["adapterSha256"])

    def test_worker_timeout_is_explicit_and_does_not_use_cached_metadata(self):
        spec = invented_spec()
        with patch.object(n, "_parse_spec", return_value=spec), patch.object(n.subprocess, "run",
              side_effect=subprocess.TimeoutExpired(["invented-worker"], 1)):
            with self.assertRaisesRegex(TimeoutError, "worker was terminated"):
                n.inspect_count_candidate(spec, timeout_seconds=1)

    @unittest.skipUnless(os.environ.get("BALLOTMANCY_NATIVE_COUNT_SPEC"), "Opt in to native owner source validation")
    def test_actual_owner_packet_is_pending_and_unselected(self):
        spec = json.loads(Path(os.environ["BALLOTMANCY_NATIVE_COUNT_SPEC"]).read_text())
        candidate = n.inspect_count_candidate(spec)
        receipt = candidate.receipt()
        self.assertEqual(candidate.status, "requires_computation")
        self.assertIsNone(candidate.artifact_id)
        self.assertIsNone(receipt["result"])
        self.assertFalse(any(receipt["qualification"].values()))


if __name__ == "__main__":
    unittest.main()
