#!/usr/bin/env python3
"""Export a static, hash-pinned model-composition capability manifest.

This describes saved artifacts and metadata-only computation requirements. It
does not run forecasts or authorize a SiteGround computation job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESEARCH = Path("research/fundamentals-iteration-20261020")
CATALOG_REL = RESEARCH / "registry-sol/catalog-v1/catalog.json"
HANDOFF_REL = RESEARCH / "registry-sol/catalog-v1/ui-handoff.json"
PREPARATION_REL = RESEARCH / "current-forecast-sol/preparation-verification-v1.json"
FITS_REL = RESEARCH / "fits-v1/batch-results.json"
BENCHMARK_JOURNAL_REL = RESEARCH / "benchmark-root-v1/execution-v1/journal.jsonl"
EXPECTED_EXPERIMENTAL_BASES = {
    "senate_history_refreshed_v1",
    "senate_presidential_slope_v1",
    "senate_relative_lean_v1",
    "senate_relative_context_v1",
}
REFRESH_HANDOFF_REL = Path("research/senate-polling-refresh-20261026/frontend-handoff-v2.json")
REFRESH_HANDOFF_SHA256 = "53ecb8ea7c322932a64784453a00bd4181c7e95cea474c2f843e5b0e0bcd356b"
REFRESH_PRESET_ID = "best_guess_polling_20261026_v1"
REFRESH_ARMS = {
    "standard": "current-polling",
    "standard_all_low": "current-error-low",
    "standard_all_high": "current-error-high",
    "standard_cycle_low": "current-cycle-low",
    "standard_cycle_high": "current-cycle-high",
    **{base: f"current-comparison-{base}-fp" for base in (
        "recent_calendar_history", "senate_history_refreshed_v1", "senate_presidential_slope_v1",
        "senate_relative_lean_v1", "senate_relative_context_v1",
    )},
}
REFRESH_POLL_CUTOFF = "2026-10-26T15:03:57.212405+00:00"
REFRESH_STRUCTURAL_CUTOFF = "2026-10-20T15:21:23Z"


class ExportError(ValueError):
    """Input artifacts do not satisfy the frozen export contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_bundle_bytes(run_id: str, raw: bytes, expected_sha256: str) -> None:
    if _sha256(raw) != expected_sha256:
        raise ExportError(f"Saved bundle hash mismatch: {run_id}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_bundle_path(href: str) -> Path:
    prefix = "/data/explorer/"
    if not href.startswith(prefix):
        raise ExportError(f"Unexpected saved-run href: {href}")
    relative = Path(href[len(prefix):])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ExportError(f"Unsafe saved-run href: {href}")
    return relative


def _read_pinned_research(source_root: Path, reference: dict[str, str]) -> dict[str, Any]:
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ExportError("Unsafe refresh research path")
    path = source_root / relative
    if not path.resolve().is_relative_to(source_root.resolve()):
        raise ExportError("Unsafe refresh research path")
    if not path.is_file() or _sha256(path.read_bytes()) != reference["sha256"]:
        raise ExportError(f"Refresh research hash mismatch: {relative}")
    result = _load_json(path)
    if not isinstance(result, dict):
        raise ExportError(f"Refresh research document is not an object: {relative}")
    return result


def _verify_refresh_runs(registry: dict[str, Any], bundles: dict[str, dict[str, Any]],
                         refresh: dict[str, Any], source_hashes: dict[str, str]) -> str:
    if source_hashes.get("refreshHandoff") != REFRESH_HANDOFF_SHA256:
        raise ExportError("Reviewed refresh handoff hash is missing or changed")
    if refresh.get("status") != "numerically_qualified_and_independently_reviewed_not_published" or \
            refresh.get("polling_input_cutoff_utc") != REFRESH_POLL_CUTOFF or \
            refresh.get("structural_input_cutoff_utc") != REFRESH_STRUCTURAL_CUTOFF:
        raise ExportError("Reviewed refresh status or cutoffs changed")
    arms = refresh.get("arms")
    if not isinstance(arms, list) or len(arms) != 10 or {arm.get("arm") for arm in arms} != set(REFRESH_ARMS):
        raise ExportError("Reviewed refresh arm set changed")
    by_run = {run["id"]: run for run in registry["runs"]}
    for arm in arms:
        name = arm["arm"]
        run_id = REFRESH_ARMS[name]
        run = by_run.get(run_id)
        bundle = bundles.get(run_id)
        forecast_pin = arm.get("files", {}).get("forecast.json", {}).get("sha256")
        if not run or not bundle or not isinstance(forecast_pin, str) or \
                bundle.get("provenance", {}).get("sources", {}).get("forecastSha256") != forecast_pin:
            raise ExportError(f"Reviewed refresh forecast hash mismatch: {name}")
        comparison = name not in {"standard", "standard_all_low", "standard_all_high",
                                  "standard_cycle_low", "standard_cycle_high"}
        target = ("senate2026_principal_pair_fundamentals_comparison_v1" if comparison
                  else "national_senate_aggregate_poll_v1")
        law = ("principal_pair_empirical_logit_shared_normal_student_t5_v1" if comparison
               else "senate_principal_pair_aggregate_v1")
        expected_components = {"base": name if comparison else "fundamentals", "polling": "standard",
                               "uncertainty": {"standard": "standard", "standard_all_low": "error-low",
                                               "standard_all_high": "error-high", "standard_cycle_low": "cycle-low",
                                               "standard_cycle_high": "cycle-high"}.get(name, "standard")}
        if run.get("dateId") != "2026-10-26" or run.get("cutoffAt") != REFRESH_POLL_CUTOFF or \
                run.get("targetId") != arm.get("target_id") or run.get("targetId") != target or \
                run.get("lawId") != law or run.get("components") != expected_components or \
                _cutoffs(run) != [{"label": "Structural inputs", "at": REFRESH_STRUCTURAL_CUTOFF},
                                  {"label": "Polling inputs", "at": REFRESH_POLL_CUTOFF}]:
            raise ExportError(f"Reviewed refresh run contract mismatch: {name}")
    return next(arm["files"]["forecast.json"]["sha256"] for arm in arms if arm["arm"] == "standard")


def _cutoffs(run: dict[str, Any]) -> list[dict[str, str]]:
    if run.get("inputCutoffs"):
        return run["inputCutoffs"]
    year = run["year"]
    if year != 2026:
        return [{"label": f"Historical reconstruction cutoff ({run['dateLabel']})", "at": run["cutoffAt"]}]
    if run["components"]["polling"] == "none":
        return [{"label": "Structural inputs", "at": run["cutoffAt"]}]
    return [
        {"label": "Structural inputs", "at": "2026-10-20T15:21:23Z"},
        {"label": "Polling data", "at": "2026-10-20T16:21:44Z"},
    ]


def _evidence(run: dict[str, Any]) -> dict[str, str]:
    if run.get("researchEvidence"):
        return {"status": "experimental", "implementation": "Saved forecasts for 35 races, each modeling its two principal contenders, with simulations of chamber outcomes.", "numerical": "Calculation checks passed. Race and chamber summaries match the saved simulations of races moving together.", "predictive": "No 2026 outcomes are available. Reviewed historical comparisons did not justify replacing the preserved default.", "scope": "These are saved research comparisons. The site does not yet calculate arbitrary new combinations for the full chamber."}
    if run["year"] < 2026:
        return {
            "status": "historical_reconstruction",
            "implementation": "Saved historical aggregate/chamber reconstruction; its winner fallback and model components are preserved as-run.",
            "numerical": "Saved simulation output is available for this exact reconstruction.",
            "predictive": "Retrospective comparison is descriptive; this reconstruction is not strict-as-of validation.",
            "scope": "Historical Senate race and chamber comparison; do not interpret as a live forecast.",
        }
    if run["id"] == "current-polling":
        predictive = (
            "The saved fundamentals-plus-standard-polling composition has positive retrospective CRPS evidence "
            "in tested cycle groups; this is not 2026 validation."
        )
    else:
        predictive = "This saved 2026 output has not been validated against its future election outcome."
    return {
        "status": "preserved",
        "implementation": "Preserved completed research forecast; no component has been recomputed for this site manifest.",
        "numerical": "Exact saved simulation bundle and source forecast fingerprint are available.",
        "predictive": predictive,
        "scope": "Current aggregate principal-contender race forecast with its saved joint chamber scenarios; not a full vote-count model.",
    }


def _configuration(run: dict[str, Any], bundle: dict[str, Any], catalog_sha: str) -> dict[str, Any]:
    result_sha = run["sha256"]
    identity = _sha256(_canonical({"catalogSha256": catalog_sha, "run": run, "resultSha256": result_sha}))
    if run["year"] == 2026:
        outputs = ["race-probabilities", "joint-chamber-distributions", "tipping-points"]
    else:
        outputs = ["race-probabilities", "historical-chamber-distributions", "retrospective-scorecard"]
    return {
        "id": f"cfg-{identity}",
        "runId": run["id"],
        "resultSha256": result_sha,
        "year": run["year"],
        "dateId": run["dateId"],
        "components": dict(run["components"]),
        "targetId": run["targetId"],
        "lawId": run["lawId"],
        "outputs": outputs,
        "cutoffs": _cutoffs(run),
        "evidence": _evidence(run),
    }


def _options(registry: dict[str, Any], implementations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = registry["runs"]
    by_slot: dict[str, dict[str, set[int]]] = {"base": {}, "polling": {}, "uncertainty": {}}
    for run in runs:
        for slot in by_slot:
            value = run["components"][slot]
            by_slot[slot].setdefault(value, set()).add(run["year"])

    def option(slot: str, id_: str, label: str, description: str, *, experimental: bool = False,
               availability: str | None = None, implementation_id: str | None = None,
               years: list[int] | None = None) -> dict[str, Any]:
        present_years = sorted(by_slot[slot].get(id_, set()))
        return {
            "id": id_, "label": label, "description": description,
            "years": years if years is not None else present_years,
            "availability": availability or ("saved" if present_years else "unavailable"),
            "experimental": experimental,
            **({"implementationId": implementation_id} if implementation_id else {}),
        }

    base_options = [
        option("base", "fundamentals", "Electoral-history aggregate", "Preserved aggregate fundamentals base used in current and historical saved runs."),
        option("base", "winner", "Historical winner-rule baseline", "Retrospective categorical winner fallback used only in saved historical chamber comparisons."),
    ]
    labels = {
        "senate_history_refreshed_v1": ("Refreshed history", "Saved experimental comparison using refreshed electoral history. Exact with-polls and without-polls results are available; new combinations may require calculation."),
        "senate_presidential_slope_v1": ("Learned presidential slope", "Saved experimental comparison learning the relationship to earlier presidential voting. Exact with-polls and without-polls results are available."),
        "senate_relative_lean_v1": ("Relative state and national lean", "Saved experimental comparison using relative state voting and national carryover. Exact with-polls and without-polls results are available."),
        "senate_relative_context_v1": ("Relative lean with presidential context", "Saved experimental comparison adding the president’s party and midterm context. Exact with-polls and without-polls results are available."),
    }
    if by_slot["base"].get("recent_calendar_history"):
        base_options.append(option("base", "recent_calendar_history", "Recent calendar history", "Saved experimental comparison using recent calendar election cycles; available with or without polls.", experimental=True))
    for implementation_id in sorted(EXPECTED_EXPERIMENTAL_BASES):
        label, description = labels[implementation_id]
        base_options.append(option("base", implementation_id, label, description, experimental=True,
                                   availability="saved" if by_slot["base"].get(implementation_id) else "requires_computation", implementation_id=implementation_id,
                                   years=[2026]))

    poll_options = [
        option("polling", "none", "No polling", "No polling likelihood factor; saved configurations only."),
        option("polling", "standard", "Standard polling", "Fixed Gaussian polling measurement; saved configurations exist, while custom combinations need new authenticated bindings."),
        option("polling", "persistent", "Persistent polling errors", "Historical polling-error sensitivity; no current 2026 saved run."),
    ]
    uncertainty_options = [
        option("uncertainty", "standard", "Standard uncertainty", "Preserved default uncertainty law; exact available combinations are listed under saved runs."),
        *[
            option("uncertainty", key, label, desc)
            for key, label, desc in [
                ("error-low", "Polling error scales × 2/3", "Saved 2026 sensitivity scaling all polling-error scales."),
                ("error-high", "Polling error scales × 1.5", "Saved 2026 sensitivity scaling all polling-error scales."),
                ("cycle-low", "Common-cycle error SD × 0.5", "Saved 2026 sensitivity scaling the shared cycle-error SD."),
                ("cycle-high", "Common-cycle error SD × 2", "Saved 2026 sensitivity scaling the shared cycle-error SD."),
                ("loading-0", "Historical national loading × 0", "Saved historical categorical-winner sensitivity; not a current forecast option."),
                ("loading-0.5", "Historical national loading × 0.5", "Saved historical categorical-winner sensitivity; not a current forecast option."),
                ("loading-1", "Historical national loading × 1", "Saved historical categorical-winner setting; historical-only in this catalog."),
                ("loading-2", "Historical national loading × 2", "Saved historical categorical-winner sensitivity; not a current forecast option."),
            ]
        ],
    ]
    return [
        {"id": "base", "label": "Base model", "description": "Select a saved base or inspect a documented experimental aggregate base.", "options": base_options},
        {"id": "polling", "label": "Polling component", "description": "Polling choices are year-scoped; saved configurations below define exact valid combinations.", "options": poll_options},
        {"id": "uncertainty", "label": "Uncertainty", "description": "Preserved scenarios; values do not imply arbitrary component compatibility.", "options": uncertainty_options},
    ]


def _computation(implementations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["implementation_id"]: item for item in implementations}
    if not EXPECTED_EXPERIMENTAL_BASES.issubset(by_id):
        raise ExportError("Frozen catalog is missing an expected experimental fundamentals implementation")
    evidence = {
        "status": "experimental",
        "implementation": "Experimental ordinary D/R aggregate implementation is cataloged; the component factory is not a frozen site resolver.",
        "numerical": "Historical-parameter fitting passed numerical diagnostics for the four experimental arms; this is not a current forecast.",
        "predictive": "Saved current comparisons are available separately. Custom requests still need their exact inputs, evidence and output qualification; a successful calculation does not establish forecast accuracy.",
        "scope": "Ordinary D/R aggregate race probabilities only. There is no custom full-chamber adapter or SiteGround execution service.",
    }
    reason = (
        "Requires authenticated data, cutoff, support, candidate and posterior/poll bindings plus a new joint forecast. "
        "Only ordinary D/R race probabilities are supported; no custom chamber adapter or SiteGround execution service exists."
    )
    requirements: list[dict[str, Any]] = []
    for base_id in sorted(EXPECTED_EXPERIMENTAL_BASES):
        requirements.append({
            "baseId": base_id,
            "years": [2026],
            "polling": ["none", "standard"],
            "uncertainty": ["standard"],
            "targetId": "senate_aggregate_logit_empirical_mixture_v1",
            "reason": reason,
            "requiredSteps": [
                "Bind and verify the exact current source data, cutoff, support, candidates and posterior/poll evidence.",
                "Fit or import one compatible joint aggregate law and generate a current forecast.",
                "Validate and save race probabilities; chamber distributions require a separate unsupported adapter.",
            ],
            "outputs": ["race-probabilities"],
            "evidence": dict(evidence),
        })
    return requirements


def build_manifest(
    registry: dict[str, Any],
    bundles: dict[str, dict[str, Any]],
    catalog: dict[str, Any],
    handoff: dict[str, Any],
    source_hashes: dict[str, str],
    refresh_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs = registry.get("runs")
    if not isinstance(runs, list) or len(runs) < 38:
        raise ExportError("Expected the complete frozen current and historical archive")
    ids = [run.get("id") for run in runs]
    if None in ids or len(set(ids)) != len(ids):
        raise ExportError("Duplicate or missing saved run ID")
    catalog_sha = source_hashes.get("frozenCatalog")
    computed_catalog_sha = _sha256(_canonical(catalog))
    if not catalog_sha or catalog_sha != computed_catalog_sha or handoff.get("catalog_sha256") != computed_catalog_sha:
        raise ExportError("Frozen catalog hash and UI handoff do not agree")
    if catalog.get("catalog_id") != handoff.get("catalog_id", catalog.get("catalog_id")) or \
            catalog.get("default_preset_id") != handoff.get("current_best_guess_preset_id"):
        raise ExportError("Frozen catalog identity or default preset mismatch")
    choices = {entry["preset_id"]: entry for entry in handoff.get("saved_forecast_choices", [])}
    catalog_presets = {entry["preset_id"]: entry for entry in catalog.get("preserved_forecasts", [])}
    required_presets = {
        "basic_electoral_history_v5_20261020": ("current-fundamentals", "national_senate_aggregate_baseline_v1"),
    }
    if refresh_handoff is None:
        required_presets["best_guess_polling_20261020_v2"] = ("current-polling", "national_senate_aggregate_poll_v1")
    presets = []
    run_by_id = {run["id"]: run for run in runs}
    for preset_id, (run_id, target_id) in required_presets.items():
        if preset_id not in choices:
            raise ExportError(f"Missing preserved preset {preset_id}")
        choice = choices[preset_id]
        expected_hash = choice["forecast"]["sha256"]
        catalog_choice = catalog_presets.get(preset_id)
        if not catalog_choice or catalog_choice.get("forecast", {}).get("sha256") != expected_hash:
            raise ExportError(f"Frozen catalog forecast hash mismatch for {preset_id}")
        run = run_by_id.get(run_id)
        bundle = bundles.get(run_id)
        if not run or run["targetId"] != target_id or not bundle:
            raise ExportError(f"Missing or mismatched bundle for preserved preset {preset_id}")
        if bundle.get("provenance", {}).get("sources", {}).get("forecastSha256") != expected_hash:
            raise ExportError(f"Preserved forecast hash mismatch for {preset_id}")
        if run["components"] != ({"base": "fundamentals", "polling": "none", "uncertainty": "standard"}
                                 if preset_id == "basic_electoral_history_v5_20261020" else
                                 {"base": "fundamentals", "polling": "standard", "uncertainty": "standard"}):
            raise ExportError(f"Preserved preset components changed for {preset_id}")
        presets.append({"id": preset_id, "label": choice["label"], "description": choice["limitations"][0],
                        "runId": run_id, "forecastSha256": expected_hash})

    if refresh_handoff is not None:
        refreshed_hash = _verify_refresh_runs(registry, bundles, refresh_handoff, source_hashes)
        presets.append({
            "id": REFRESH_PRESET_ID,
            "label": "Best guess with October 26 polling",
            "description": "Reviewed saved Senate aggregate forecast using qualified polling through October 26; historical calibration is reused and no 2026 outcome has been scored.",
            "runId": "current-polling",
            "forecastSha256": refreshed_hash,
        })

    configurations = []
    for run in runs:
        bundle = bundles.get(run["id"])
        if not bundle:
            raise ExportError(f"Missing saved bundle for {run['id']}")
        if bundle.get("run", {}).get("id") != run["id"]:
            raise ExportError(f"Bundle/run identity mismatch for {run['id']}")
        configurations.append(_configuration(run, bundle, catalog_sha))

    implementations = catalog.get("implementations", [])
    experimental = {item["implementation_id"] for item in implementations
                    if item.get("role") == "fundamentals" and item.get("status") == "experimental"}
    if experimental != EXPECTED_EXPERIMENTAL_BASES:
        raise ExportError("Experimental base implementation set differs from the reviewed catalog")
    experimental_records = {item["implementation_id"]: item for item in implementations
                            if item.get("implementation_id") in EXPECTED_EXPERIMENTAL_BASES}
    if any("senate_aggregate_logit_empirical_mixture_v1" not in record.get("target_ids", [])
           for record in experimental_records.values()):
        raise ExportError("Experimental implementation target differs from the ordinary D/R aggregate target")
    if not any(item.get("implementation_id") == "senate_polling_basis_v1" and item.get("role") == "measurement"
               for item in implementations):
        raise ExportError("Frozen catalog is missing its standard polling measurement adapter")

    unavailable = [
        {"id": "fine-component-toggles", "label": "Fine-grained component toggles",
         "reason": "The component factory is outside the frozen catalog/site resolver; changing one coefficient generally requires a new joint fit."},
        {"id": "population-community-participation-counts", "label": "Population, community, participation and vote counts",
         "reason": "These components are not connected end to end to the aggregate target or a deployable current forecast."},
        {"id": "custom-full-chamber", "label": "Custom full-chamber forecast",
         "reason": "No custom chamber adapter exists for experimental component combinations."},
        {"id": "forecast-ensemble", "label": "Weighted forecast blending",
         "reason": "No ensemble service preserves and combines complete joint forecasts for arbitrary model choices."},
    ]
    return {
        "schemaVersion": 1,
        "id": "composition-capabilities-2026-10-26-v1" if refresh_handoff is not None
              else "composition-capabilities-2026-10-20-v2",
        "sourceCatalog": {"id": catalog["catalog_id"], "sha256": catalog_sha},
        "sources": source_hashes,
        "defaultPresetId": REFRESH_PRESET_ID if refresh_handoff is not None
                           else handoff["current_best_guess_preset_id"],
        "execution": {"available": False,
                      "reason": "This static site can load saved bundles only; it has no authenticated fit or forecast job service."},
        "presets": presets,
        "slots": _options(registry, implementations),
        "saved": configurations,
        "computation": _computation(implementations),
        "unavailable": unavailable,
    }


def export(source_root: Path, output: Path, registry_path: Path, bundle_root: Path,
           refresh_handoff_path: Path | None = None) -> dict[str, Any]:
    catalog_path = source_root / CATALOG_REL
    handoff_path = source_root / HANDOFF_REL
    preparation_path = source_root / PREPARATION_REL
    fits_path = source_root / FITS_REL
    benchmark_journal_path = source_root / BENCHMARK_JOURNAL_REL
    for path in (catalog_path, handoff_path, preparation_path, fits_path, benchmark_journal_path):
        if not path.is_file():
            raise ExportError(f"Required frozen evidence missing: {path.name}")
    registry = _load_json(registry_path)
    catalog = _load_json(catalog_path)
    handoff = _load_json(handoff_path)
    preparation = _load_json(preparation_path)
    fits = _load_json(fits_path)
    if preparation.get("empirical_predictions_sampled") is not False:
        raise ExportError("Pinned historical preparation receipt changed")
    fit_results = fits.get("results", [])
    if (len(fit_results) != 21 or any(result.get("status") != "passed" for result in fit_results)
            or not EXPECTED_EXPERIMENTAL_BASES.issubset({r.get("implementation_id") for r in fit_results})):
        raise ExportError("Historical fit batch no longer has all passed status")
    for choice in handoff.get("saved_forecast_choices", []):
        source = Path(choice["forecast"]["path"])
        if source.is_absolute() or ".." in source.parts:
            raise ExportError("Frozen forecast source path must be repository-relative")
        source_path = source_root / source
        if not source_path.is_file() or _sha256(source_path.read_bytes()) != choice["forecast"]["sha256"]:
            raise ExportError(f"Frozen source forecast hash mismatch: {choice['preset_id']}")
    bundle_entries = {entry["id"]: entry for entry in registry["runs"]}
    bundles: dict[str, dict[str, Any]] = {}
    for run_id, entry in bundle_entries.items():
        path = bundle_root / _relative_bundle_path(entry["href"]).name
        if not path.is_file():
            raise ExportError(f"Missing saved bundle: {run_id}")
        raw = path.read_bytes()
        _verify_bundle_bytes(run_id, raw, entry["sha256"])
        bundles[run_id] = json.loads(raw)
    source_hashes = {
        # The catalog identity is its canonical content hash, as pinned by the
        # frozen UI handoff; retain the byte hash too for artifact provenance.
        "frozenCatalog": _sha256(_canonical(catalog)),
        "catalogFile": _sha256(catalog_path.read_bytes()),
        "catalogUiHandoff": _sha256(handoff_path.read_bytes()),
        "currentForecastPreparation": _sha256(preparation_path.read_bytes()),
        "historicalFitBatch": _sha256(fits_path.read_bytes()),
        "historicalPredictionJournal": _sha256(benchmark_journal_path.read_bytes()),
    }
    comparison_index = source_root / RESEARCH / "frontend-contract-root-v2/interface-v1/current-comparisons.json"
    if any(run.get("researchEvidence") for run in registry["runs"]):
        source_hashes["currentComparisonIndex"] = _sha256(comparison_index.read_bytes())
        source_hashes["compositionInterfaceReceipt"] = _sha256((comparison_index.parent / "receipt.json").read_bytes())
    refresh = None
    if refresh_handoff_path is not None:
        if refresh_handoff_path != REFRESH_HANDOFF_REL:
            raise ExportError("Only the exact reviewed October 26 refresh handoff is supported")
        path = source_root / refresh_handoff_path
        if not path.resolve().is_relative_to(source_root.resolve()) or not path.is_file() or \
                _sha256(path.read_bytes()) != REFRESH_HANDOFF_SHA256:
            raise ExportError("Reviewed refresh handoff hash changed")
        refresh = _load_json(path)
        source_review = _read_pinned_research(source_root, refresh["root_source_review"])
        output_review = _read_pinned_research(source_root, refresh["independent_output_review"])
        if source_review.get("status") != "admitted_for_existing_aggregate_target" or \
                source_review.get("candidate_selection_approved") is not True or \
                output_review.get("status") != "passed" or \
                set(output_review.get("arms", {})) != set(REFRESH_ARMS) or \
                any(item.get("status") != "passed" for item in output_review["arms"].values()):
            raise ExportError("Reviewed refresh source or all-arm output review failed")
        source_hashes["refreshHandoff"] = REFRESH_HANDOFF_SHA256
        source_hashes["refreshSourceAdmissionReview"] = refresh["root_source_review"]["sha256"]
        source_hashes["refreshAllArmOutputReview"] = refresh["independent_output_review"]["sha256"]
    manifest = build_manifest(registry, bundles, catalog, handoff, source_hashes, refresh)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Read-only research integration checkout root")
    parser.add_argument("--output", type=Path, default=Path("public/data/explorer/capabilities.json"))
    parser.add_argument("--registry", type=Path, default=Path("public/data/explorer/registry.json"))
    parser.add_argument("--bundle-root", type=Path, default=Path("public/data/explorer"))
    parser.add_argument("--refresh-handoff", type=Path,
                        help="Explicit reviewed October 26 handoff; omitted for archived October 20 projection")
    args = parser.parse_args()
    manifest = export(args.source_root, args.output, args.registry, args.bundle_root, args.refresh_handoff)
    print(f"wrote {args.output} with {len(manifest['saved'])} saved configs and {len(manifest['computation'])} metadata-only options")


if __name__ == "__main__":
    main()
