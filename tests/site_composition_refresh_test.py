"""Mixed-edition composition preset contract; no research catalog edits."""
from __future__ import annotations

import copy
import hashlib
import unittest

from scripts.export_composition_capabilities import (
    ExportError, REFRESH_ARMS, REFRESH_HANDOFF_SHA256, REFRESH_POLL_CUTOFF,
    REFRESH_PRESET_ID, REFRESH_STRUCTURAL_CUTOFF, build_manifest,
)
from tests.site_composition_test import fixture_inputs


def refreshed_inputs():
    registry, bundles, catalog, handoff, hashes = fixture_inputs()
    hashes["refreshHandoff"] = REFRESH_HANDOFF_SHA256
    template = registry["runs"][1]
    arms = []
    for name, run_id in REFRESH_ARMS.items():
        comparison = name not in {"standard", "standard_all_low", "standard_all_high",
                                  "standard_cycle_low", "standard_cycle_high"}
        forecast_sha = hashlib.sha256(name.encode()).hexdigest()
        target = ("senate2026_principal_pair_fundamentals_comparison_v1" if comparison
                  else "national_senate_aggregate_poll_v1")
        arm = {"arm": name, "target_id": target,
               "files": {"forecast.json": {"sha256": forecast_sha}}}
        arms.append(arm)
        run = next((run for run in registry["runs"] if run["id"] == run_id), None)
        if run is None:
            run = copy.deepcopy(template)
            run["id"] = run_id
            registry["runs"].append(run)
        run["targetId"] = target
        run["lawId"] = ("principal_pair_empirical_logit_shared_normal_student_t5_v1" if comparison
                        else "senate_principal_pair_aggregate_v1")
        run["dateId"] = "2026-10-26"
        run["cutoffAt"] = REFRESH_POLL_CUTOFF
        run["components"] = {"base": name if comparison else "fundamentals", "polling": "standard",
                             "uncertainty": {"standard_all_low": "error-low", "standard_all_high": "error-high",
                                             "standard_cycle_low": "cycle-low",
                                             "standard_cycle_high": "cycle-high"}.get(name, "standard")}
        run["inputCutoffs"] = [{"label": "Structural inputs", "at": REFRESH_STRUCTURAL_CUTOFF},
                               {"label": "Polling inputs", "at": REFRESH_POLL_CUTOFF}]
        if run_id not in bundles:
            bundles[run_id] = {"run": {"id": run_id}, "provenance": {"sources": {}}}
        bundles[run_id]["provenance"]["sources"]["forecastSha256"] = forecast_sha
    refresh = {"status": "numerically_qualified_and_independently_reviewed_not_published",
               "polling_input_cutoff_utc": REFRESH_POLL_CUTOFF,
               "structural_input_cutoff_utc": REFRESH_STRUCTURAL_CUTOFF,
               "arms": arms}
    return registry, bundles, catalog, handoff, hashes, refresh


class RefreshedCompositionTests(unittest.TestCase):
    def test_new_preset_binds_new_standard_and_keeps_basic(self):
        inputs = refreshed_inputs()
        result = build_manifest(*inputs)
        self.assertEqual(result["id"], "composition-capabilities-2026-10-26-v1")
        self.assertEqual(result["defaultPresetId"], REFRESH_PRESET_ID)
        self.assertEqual({p["id"] for p in result["presets"]},
                         {"basic_electoral_history_v5_20261020", REFRESH_PRESET_ID})
        self.assertEqual(result["presets"][0]["forecastSha256"], "f" * 64)
        self.assertEqual(result["presets"][1]["forecastSha256"],
                         next(a["files"]["forecast.json"]["sha256"] for a in inputs[5]["arms"]
                              if a["arm"] == "standard"))
        self.assertEqual(len(result["saved"]), 47)

    def test_old_preset_hash_is_not_silently_redefined(self):
        inputs = refreshed_inputs()
        with self.assertRaisesRegex(ExportError, "Preserved forecast hash mismatch"):
            build_manifest(*inputs[:5])

    def test_tampered_new_forecast_and_misbased_recent_calendar_rejected(self):
        inputs = refreshed_inputs()
        inputs[1]["current-polling"]["provenance"]["sources"]["forecastSha256"] = "0" * 64
        with self.assertRaisesRegex(ExportError, "refresh forecast hash mismatch"):
            build_manifest(*inputs)
        inputs = refreshed_inputs()
        run = next(r for r in inputs[0]["runs"] if r["id"] == "current-comparison-recent_calendar_history-fp")
        run["components"]["base"] = "senate_history_refreshed_v1"
        with self.assertRaisesRegex(ExportError, "run contract mismatch"):
            build_manifest(*inputs)

    def test_basic_and_handoff_pins_remain_mandatory(self):
        inputs = refreshed_inputs()
        inputs[1]["current-fundamentals"]["provenance"]["sources"]["forecastSha256"] = "0" * 64
        with self.assertRaisesRegex(ExportError, "Preserved forecast hash mismatch"):
            build_manifest(*inputs)
        inputs = refreshed_inputs()
        inputs[4]["refreshHandoff"] = "0" * 64
        with self.assertRaisesRegex(ExportError, "refresh handoff hash"):
            build_manifest(*inputs)


if __name__ == "__main__":
    unittest.main()
