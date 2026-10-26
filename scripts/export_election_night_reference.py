#!/usr/bin/env python3
"""Bind a reviewed model-implied Senate scorecard reference to one dated public bundle.

The owner sidecar already derives expectations from unchanged native joint rows and
has an independent numerical review. This adapter authenticates those bytes and
projects the reviewed values; it never reads outcomes, resamples, or grades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REFERENCE_REL = Path("research/senate-polling-refresh-20261026/standard-scorecard-reference-v1/reference.json")
REVIEW_REL = Path("research/senate-polling-refresh-20261026/scorecard-review-sol/review-v1.json")
REFERENCE_SHA256 = "48d36fdce404a74569553c859a8b8d66bb9790c0eef614950bfa98ca5da058db"
REVIEW_SHA256 = "4ba519f76c03852659c69eec96e84913c0fbcd00c8e8e3d82de6e9e7d845fe05"
PUBLIC_BUNDLE_SHA256 = "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0"
RUN_ID = "current-polling"
TARGET_ID = "national_senate_aggregate_poll_v1"
LAW_ID = "senate_principal_pair_aggregate_v1"
POLL_CUTOFF = "2026-10-26T15:03:57.212405+00:00"
STRUCTURAL_CUTOFF = "2026-10-20T15:21:23Z"
METRICS = {"mean_signed_margin_error_pp", "mean_absolute_margin_error_pp",
           "interval_90_coverage_count", "modal_pair_winner_miss_count"}
EXTERNAL_CONTRACT_REL = Path("ballotmancy-improvement-lab/research/improvement-loop-20261024/election-night-contract-audit-sol-v1/adapter-contract.json")


class ReferenceExportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceExportError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def pinned(root: Path, reference: dict[str, str]) -> Path:
    relative = Path(reference["path"])
    if relative.is_absolute():
        # The independent review pins this one preexisting contract in the
        # sibling lab checkout. Never permit arbitrary absolute source paths.
        expected = (root.resolve().parents[1] / EXTERNAL_CONTRACT_REL).resolve()
        require(relative == expected and relative.is_file(), "Unsafe external review source")
        path = relative
    else:
        require(".." not in relative.parts, "Unsafe source path")
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "Unsafe source path")
    require(path.is_file() and sha(path) == reference["sha256"], f"Source hash mismatch: {reference['path']}")
    return path


def near(actual: Any, expected: Any, label: str, tolerance: float = 1e-9) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f"{label} differs from reviewed forecast")


def _validate_distribution(reference: dict[str, Any], review: dict[str, Any]) -> None:
    definitions = reference.get("metric_definitions")
    distributions = reference.get("distributions")
    reviewed = review.get("independently_rebuilt_distributions")
    require(isinstance(definitions, dict) and set(definitions) == METRICS
            and all(isinstance(v, str) and v for v in definitions.values())
            and isinstance(distributions, dict) and set(distributions) == METRICS
            and isinstance(reviewed, dict) and set(reviewed) == METRICS,
            "Diagnostic metric set changed")
    for metric in METRICS:
        row, checked = distributions[metric], reviewed[metric]
        require(set(row) == {"mean", "standard_deviation", "quantiles"}
                and set(row["quantiles"]) == {"p01", "p05", "p50", "p95", "p99"},
                f"Malformed diagnostic distribution: {metric}")
        near(row["mean"], checked["mean"], metric + ".mean")
        near(row["standard_deviation"], checked["standard_deviation"], metric + ".sd")
        require(row["standard_deviation"] >= 0, f"Negative diagnostic spread: {metric}")
        quantiles = [row["quantiles"][key] for key in ("p01", "p05", "p50", "p95", "p99")]
        require(quantiles == sorted(quantiles), f"Disordered diagnostic quantiles: {metric}")
        for key in row["quantiles"]:
            near(row["quantiles"][key], checked["quantiles"][key], metric + "." + key)
        if metric.endswith("_count"):
            require(all(isinstance(value, int) and 0 <= value <= 35 for value in quantiles),
                    f"Invalid count diagnostic quantile: {metric}")


def _validate_cohort(reference: dict[str, Any], review: dict[str, Any],
                     forecast: dict[str, Any], bundle: dict[str, Any]) -> None:
    cohort = reference.get("cohort")
    native = forecast.get("races")
    public = bundle.get("races")
    require(isinstance(cohort, list) and isinstance(native, list) and isinstance(public, list)
            and len(cohort) == len(native) == len(public) == review.get("cohort_size") == 35,
            "Reviewed cohort count changed")
    digest = hashlib.sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == reference.get("cohort_sha256"), "Reviewed cohort hash changed")
    require(len({row.get("race_id") for row in cohort}) == 35, "Duplicate diagnostic race")
    expected_fields = {"race_id", "state", "first_candidate_id", "second_candidate_id", "target_id",
                       "rule_proxy", "share_interpretation", "excluded_options_assumption", "margin_sign",
                       "denominator", "point_margin_mean_pp", "interval_90_lower_margin_pp",
                       "interval_90_upper_margin_pp", "expected_absolute_margin_error_pp",
                       "first_pair_win_probability", "modal_pair_winner_id", "forecast_probability_exact_half",
                       "included_in_native_diagnostic_cohort"}
    for index, (row, source, site) in enumerate(zip(cohort, native, public, strict=True)):
        label = f"cohort[{index}]"
        require(set(row) == expected_fields and row["race_id"] == source["race_id"] == site["id"]
                and row["state"] == source["state"] == site["state"]
                and row["target_id"] == TARGET_ID and row["included_in_native_diagnostic_cohort"] is True,
                f"{label} identity/fields changed")
        first, second = site["outcomes"]
        require(len(site["outcomes"]) == 2
                and row["first_candidate_id"] == source["first_contender"]["id"] == first["id"]
                and row["second_candidate_id"] == source["second_contender"]["id"] == second["id"]
                and row["rule_proxy"] == source["rule_proxy"]
                and row["share_interpretation"] == source["share_interpretation"]
                and row["excluded_options_assumption"] == source["excluded_options_assumption"]
                and row["margin_sign"] == "first_candidate_minus_second_candidate"
                and row["denominator"] == "sum_of_named_principal_candidate_votes_under_declared_rule_proxy"
                and first["share"]["denominator"] == "Share of votes for the two named principal contenders",
                f"{label} pair rule/denominator/support mismatch")
        near(row["point_margin_mean_pp"], source["first_minus_second_margin_mean_pp"], label + ".native point")
        near(row["point_margin_mean_pp"], 200 * first["share"]["mean"] - 100, label + ".public point")
        near(row["interval_90_lower_margin_pp"], 200 * first["share"]["quantiles"]["p05"] - 100,
             label + ".lower")
        near(row["interval_90_upper_margin_pp"], 200 * first["share"]["quantiles"]["p95"] - 100,
             label + ".upper")
        near(row["first_pair_win_probability"], source["first_win_probability"], label + ".native win")
        near(row["first_pair_win_probability"], first["winProbability"], label + ".public win")
        require(row["interval_90_lower_margin_pp"] <= row["interval_90_upper_margin_pp"],
                f"{label} malformed margin interval")
        require(isinstance(row["expected_absolute_margin_error_pp"], (int, float))
                and math.isfinite(row["expected_absolute_margin_error_pp"])
                and 0 <= row["expected_absolute_margin_error_pp"] <= 200,
                f"{label} invalid model-implied error")
        p = row["first_pair_win_probability"]
        require(row["modal_pair_winner_id"] == (first["id"] if p >= .5 else second["id"])
                and row["forecast_probability_exact_half"] == (p == .5), f"{label} modal tie rule mismatch")


def build(source_root: Path, public_bundle: Path) -> dict[str, Any]:
    root = source_root.resolve()
    reference = read(pinned(root, {"path": str(REFERENCE_REL), "sha256": REFERENCE_SHA256}))
    review = read(pinned(root, {"path": str(REVIEW_REL), "sha256": REVIEW_SHA256}))
    require(reference.get("schema") == "SenateJointScorecardReferenceV1"
            and reference.get("status") == "derived_from_qualified_joint_draws"
            and reference.get("evaluation_lock") is False
            and reference.get("observed_results_consumed") is False,
            "Reference status/meaning changed")
    require(review.get("status") == "passed_scoped_reference_not_full_frontend_adapter"
            and review.get("checks_passed") == len(review.get("checks", [])) == 128
            and review.get("no_actual_outcomes_grades_fits_or_new_draws") is True,
            "Independent reference review changed")
    for pin in reference["sources"]:
        pinned(root, pin)
    for pin in review["sources"]:
        pinned(root, pin)
    diagnostic_path = pinned(root, reference["joint_rows"])
    require(reference["joint_rows"] in review["sources"], "Reviewed diagnostic rows differ")
    require({pin["path"]: pin["sha256"] for pin in review["sources"]}.get(str(REFERENCE_REL)) == REFERENCE_SHA256,
            "Independent review does not bind exact reference")
    source_by_name = {Path(pin["path"]).name: pin for pin in reference["sources"]}
    native_base = REFERENCE_REL.parents[1] / "runner-sol/predictions-v1/standard"
    native_path = root / native_base / "forecast.json"
    native_pin = {"path": str(native_base / "forecast.json"),
                  "sha256": source_by_name["forecast.json"]["sha256"]}
    pinned(root, native_pin)
    forecast = read(native_path)
    receipt = read(root / native_base / "receipt.json")
    gates = read(root / native_base / "numerical-gates.json")
    owner_review = read(root / "research/senate-polling-refresh-20261026/root-standard-output-review-v1.json")
    require(receipt.get("status") == gates.get("status") == owner_review.get("status") == "passed"
            and gates.get("checks") and all(value is True for value in gates["checks"].values())
            and receipt.get("artifacts", {}).get("forecast.json") == source_by_name["forecast.json"]["sha256"]
            and receipt.get("artifacts", {}).get("joint-draws.npz") == source_by_name["joint-draws.npz"]["sha256"]
            and receipt.get("artifacts", {}).get("numerical-gates.json") == source_by_name["numerical-gates.json"]["sha256"]
            and owner_review.get("arms", {}).get("standard", {}).get("forecast") == native_pin,
            "Native numerical gates or owner output review changed")
    require(forecast.get("schema") == "SenatePollingRefreshForecastV1"
            and forecast.get("mode") == "FP" and forecast.get("comparison_arm") == "standard"
            and forecast.get("target_id") == TARGET_ID
            and forecast.get("input_cutoff_utc") == reference.get("forecast_edition_cutoff_utc") == POLL_CUTOFF
            and forecast.get("nonpoll_input_cutoff_utc") == reference.get("structural_cutoff_utc") == STRUCTURAL_CUTOFF,
            "Native forecast identity/cutoffs changed")
    require(sha(public_bundle) == PUBLIC_BUNDLE_SHA256, "Public forecast bundle hash changed")
    bundle = read(public_bundle)
    run = bundle.get("run", {})
    provenance = bundle.get("provenance", {}).get("sources", {})
    require(run.get("id") == RUN_ID and run.get("dateId") == "2026-10-26"
            and run.get("targetId") == TARGET_ID and run.get("lawId") == LAW_ID
            and run.get("cutoffAt") == POLL_CUTOFF and run.get("issuedAt") == forecast["executed_utc"]
            and run.get("simulationCount") == 40000 and run.get("historical") is False
            and run.get("components") == {"base": "fundamentals", "polling": "standard", "uncertainty": "standard"}
            and provenance.get("forecastSha256") == native_pin["sha256"]
            and provenance.get("jointDrawsSha256") == source_by_name["joint-draws.npz"]["sha256"]
            and provenance.get("receiptSha256") == source_by_name["receipt.json"]["sha256"]
            and provenance.get("numericalGatesSha256") == source_by_name["numerical-gates.json"]["sha256"],
            "Public forecast edition/provenance mismatch")
    identity = reference.get("draw_identity", {})
    require(identity == {"shape": [4, 10000], "order": "unchanged native chain, within-chain draw; C-order flattening",
                         "weight_per_row": 2.5e-05, "shared_across_all_races": True,
                         "resampling_or_independence_reconstruction": False}
            and review.get("joint_row_shape") == [4, 10000]
            and review.get("total_native_rows") == 40000
            and review.get("weight_per_native_row") == 2.5e-05
            and review.get("exacthalf_modal_ties") == 0,
            "Original shared-row identity or weights changed")
    _validate_cohort(reference, review, forecast, bundle)
    _validate_distribution(reference, review)
    near(reference["distributions"]["modal_pair_winner_miss_count"]["mean"],
         review["expected_modal_pair_winner_misses"], "expected modal misses")
    return {
        "schemaVersion": 1,
        "entries": [{
            "chamber": "senate", "runId": RUN_ID, "resultSha256": PUBLIC_BUNDLE_SHA256,
            "targetId": TARGET_ID, "lawId": LAW_ID,
            "forecastCutoffAt": POLL_CUTOFF, "structuralCutoffAt": STRUCTURAL_CUTOFF,
            "forecastIssuedAt": run["issuedAt"], "referenceIssuedAt": reference["issued_utc"],
            "evaluationLock": False, "observedResultsConsumed": False,
            "sourcePins": {"referenceSha256": REFERENCE_SHA256, "reviewSha256": REVIEW_SHA256,
                           "forecastSha256": native_pin["sha256"],
                           "jointDrawsSha256": source_by_name["joint-draws.npz"]["sha256"],
                           "jointRowsSha256": sha(diagnostic_path),
                           "receiptSha256": source_by_name["receipt.json"]["sha256"],
                           "numericalGatesSha256": source_by_name["numerical-gates.json"]["sha256"],
                           "ownerOutputReviewSha256": source_by_name["root-standard-output-review-v1.json"]["sha256"]},
            "reference": {"drawIdentity": identity, "cohortSha256": reference["cohort_sha256"],
                          "cohort": reference["cohort"],
                          "metricDefinitions": reference["metric_definitions"],
                          "distributions": reference["distributions"],
                          "qualifications": reference["qualifications"]},
        }],
    }


def export(source_root: Path, public_bundle: Path, output: Path,
           publication: Path) -> dict[str, Any]:
    catalog = build(source_root, public_bundle)
    data = (json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    editions = [{"chamber": row["chamber"], "runId": row["runId"],
                 "resultSha256": row["resultSha256"]} for row in catalog["entries"]]
    require(len(editions) == len({(row["chamber"], row["runId"], row["resultSha256"]) for row in editions}),
            "Duplicate reference edition declaration")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    publication.parent.mkdir(parents=True, exist_ok=True)
    publication.write_text(json.dumps({"schemaVersion": 1,
                                       "href": "/data/election-night/references.json",
                                       "sha256": hashlib.sha256(data).hexdigest(),
                                       "editions": editions},
                                      indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return catalog


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--public-bundle", type=Path, default=repo / "public/data/explorer/current-polling.json")
    parser.add_argument("--output", type=Path, default=repo / "public/data/election-night/references.json")
    parser.add_argument("--publication", type=Path, default=repo / "data/election-night-reference-publication.json")
    args = parser.parse_args()
    result = export(args.source_root, args.public_bundle, args.output, args.publication)
    print(f"Exported {len(result['entries'])} reviewed model-implied reference entry")


if __name__ == "__main__":
    main()
