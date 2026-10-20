import copy
import hashlib
import json
import unittest

from scripts.export_composition_capabilities import (
    EXPECTED_EXPERIMENTAL_BASES,
    ExportError,
    _verify_bundle_bytes,
    build_manifest,
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_inputs():
    runs = []
    bundles = {}
    for index in range(38):
        run_id = ["current-fundamentals", "current-polling"][index] if index < 2 else f"saved-{index}"
        year = 2026 if index < 6 else (2018 if index < 22 else 2022)
        components = (
            {"base": "fundamentals", "polling": "none", "uncertainty": "standard"}
            if index == 0 else
            {"base": "fundamentals", "polling": "standard", "uncertainty": "standard"}
            if index == 1 else
            {"base": "fundamentals", "polling": "none", "uncertainty": "loading-1"}
        )
        raw_bundle = f"bundle:{run_id}"
        run = {
            "id": run_id,
            "year": year,
            "label": f"label {index}",
            "targetId": "national_senate_aggregate_baseline_v1" if index == 0 else "national_senate_aggregate_poll_v1",
            "lawId": "senate_principal_pair_aggregate_v1" if year == 2026 else "historical_categorical_winner_gate_v1",
            "dateId": "2026-10-20" if year == 2026 else "h14",
            "dateLabel": "October 20, 2026" if year == 2026 else "14 days before election",
            "cutoffAt": "2026-10-20T15:21:23Z" if index == 0 else "2026-10-20T16:21:44Z" if year == 2026 else f"{year}-10-23",
            "components": components,
            "href": f"/data/explorer/{run_id}.json",
            "sha256": sha(raw_bundle),
        }
        runs.append(run)
        bundle = {"run": {"id": run_id}, "provenance": {"sources": {}}}
        if index == 0:
            bundle["provenance"]["sources"]["forecastSha256"] = "f" * 64
        elif index == 1:
            bundle["provenance"]["sources"]["forecastSha256"] = "e" * 64
        bundles[run_id] = bundle

    registry = {"runs": runs}
    catalog = {
        "catalog_id": "frozen-catalog",
        "default_preset_id": "best_guess_polling_20261020_v2",
        "implementations": [
            {"implementation_id": "senate_history_v5", "role": "fundamentals", "status": "preserved",
             "target_ids": ["senate_aggregate_logit_empirical_mixture_v1"]},
            *[{"implementation_id": item, "role": "fundamentals", "status": "experimental",
               "target_ids": ["senate_aggregate_logit_empirical_mixture_v1"]}
              for item in sorted(EXPECTED_EXPERIMENTAL_BASES)],
            {"implementation_id": "senate_polling_basis_v1", "role": "measurement", "status": "experimental"},
        ],
        "preserved_forecasts": [
            {"preset_id": "basic_electoral_history_v5_20261020", "forecast": {"sha256": "f" * 64}},
            {"preset_id": "best_guess_polling_20261020_v2", "forecast": {"sha256": "e" * 64}},
        ],
    }
    catalog_sha = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    handoff = {
        "catalog_sha256": catalog_sha,
        "current_best_guess_preset_id": "best_guess_polling_20261020_v2",
        "saved_forecast_choices": [
            {"preset_id": "basic_electoral_history_v5_20261020", "label": "Basics",
             "limitations": ["Saved aggregate forecast"], "forecast": {"sha256": "f" * 64}},
            {"preset_id": "best_guess_polling_20261020_v2", "label": "Polling",
             "limitations": ["Saved polling forecast"], "forecast": {"sha256": "e" * 64}},
        ],
    }
    source_hashes = {"frozenCatalog": catalog_sha, "catalogUiHandoff": "b" * 64,
                     "currentForecastPreparation": "c" * 64, "historicalFitBatch": "d" * 64}
    return registry, bundles, catalog, handoff, source_hashes


class CompositionManifestTests(unittest.TestCase):
    def test_emits_all_saved_runs_and_only_supported_computation_family(self):
        inputs = fixture_inputs()
        result = build_manifest(*inputs)
        self.assertEqual(len(result["saved"]), 38)
        self.assertEqual(len(result["computation"]), 4)
        self.assertTrue(all(row["years"] == [2026] for row in result["computation"]))
        self.assertTrue(all(row["outputs"] == ["race-probabilities"] for row in result["computation"]))
        self.assertFalse(result["execution"]["available"])
        self.assertEqual(result["defaultPresetId"], "best_guess_polling_20261020_v2")

    def test_rejects_preserved_forecast_hash_mismatch(self):
        inputs = fixture_inputs()
        inputs[3]["saved_forecast_choices"][0]["forecast"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ExportError, "forecast hash mismatch"):
            build_manifest(*inputs)

    def test_authenticates_canonical_catalog_hash_against_handoff(self):
        inputs = fixture_inputs()
        inputs[2]["catalog_id"] = "changed-catalog"
        with self.assertRaisesRegex(ExportError, "catalog hash"):
            build_manifest(*inputs)

    def test_rejects_duplicate_configuration_ids(self):
        inputs = fixture_inputs()
        inputs[0]["runs"][2]["id"] = inputs[0]["runs"][3]["id"]
        with self.assertRaisesRegex(ExportError, "Duplicate or missing saved run ID"):
            build_manifest(*inputs)

    def test_rejects_missing_saved_bundle(self):
        inputs = fixture_inputs()
        del inputs[1]["saved-2"]
        with self.assertRaisesRegex(ExportError, "Missing saved bundle"):
            build_manifest(*inputs)

    def test_rejects_saved_bundle_hash_mismatch(self):
        with self.assertRaisesRegex(ExportError, "Saved bundle hash mismatch"):
            _verify_bundle_bytes("run-a", b"changed bundle", sha("expected bundle"))

    def test_configuration_identity_uses_full_run_metadata_and_result_hash(self):
        inputs = fixture_inputs()
        first = build_manifest(*inputs)["saved"][2]["id"]
        modified = copy.deepcopy(inputs)
        modified[0]["runs"][2]["label"] = "changed metadata"
        second = build_manifest(*modified)["saved"][2]["id"]
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
