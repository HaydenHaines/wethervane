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

REFERENCE_REL = Path("research/senate-polling-refresh-20261026/standard-scorecard-reference-v2/reference.json")
REVIEW_REL = Path("research/senate-polling-refresh-20261026/scorecard-tail-review-sol/review-v1.json")
REVIEW_MANIFEST_REL = Path("research/senate-polling-refresh-20261026/scorecard-tail-review-sol/manifest-v1.json")
PREVIOUS_REFERENCE_REL = Path("research/senate-polling-refresh-20261026/standard-scorecard-reference-v1/reference.json")
PREVIOUS_REVIEW_REL = Path("research/senate-polling-refresh-20261026/scorecard-review-sol/review-v1.json")
REFERENCE_SHA256 = "64df0b35a79e46d7d838b5aba70dbd678d110b19392ce30fa720c0ae3840a983"
REVIEW_SHA256 = "b5f793d959c484fc977e46f92afc63bc0375b8817708cd771fe9ae637e5f14e0"
REVIEW_MANIFEST_SHA256 = "b4cae1844dce11bc9bee0ed287cc0eb8e461142bce91359408207d3e6e0c6ab9"
PREVIOUS_REFERENCE_SHA256 = "48d36fdce404a74569553c859a8b8d66bb9790c0eef614950bfa98ca5da058db"
PREVIOUS_REVIEW_SHA256 = "4ba519f76c03852659c69eec96e84913c0fbcd00c8e8e3d82de6e9e7d845fe05"
PUBLIC_BUNDLE_SHA256 = "905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0"
RUN_ID = "current-polling"
TARGET_ID = "national_senate_aggregate_poll_v1"
LAW_ID = "senate_principal_pair_aggregate_v1"
POLL_CUTOFF = "2026-10-26T15:03:57.212405+00:00"
STRUCTURAL_CUTOFF = "2026-10-20T15:21:23Z"
METRICS = {"mean_signed_margin_error_pp", "mean_absolute_margin_error_pp",
           "interval_90_coverage_count", "modal_pair_winner_miss_count"}
MARGIN_EVENT_KEYS = ("close_margin_probability_2pp", "first_decisive_margin_probability_10pp",
                     "second_decisive_margin_probability_10pp")
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
                     previous: dict[str, Any], forecast: dict[str, Any], bundle: dict[str, Any]) -> None:
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
                       "included_in_native_diagnostic_cohort", "margin_event_draw_counts",
                       "margin_event_draw_denominator", "close_margin_probability_2pp",
                       "first_decisive_margin_probability_10pp", "second_decisive_margin_probability_10pp"}
    previous_cohort = previous.get("cohort")
    independent = review.get("independent_counts")
    require(isinstance(previous_cohort, list) and len(previous_cohort) == 35
            and isinstance(independent, list) and len(independent) == 35,
            "Prior cohort or independent tail review count changed")
    event_keys = MARGIN_EVENT_KEYS
    definitions = reference.get("race_margin_probability_definitions")
    require(definitions == {
        "margin": "M = 100*(V_first-V_second)/(V_first+V_second) = 200*first_share-100, under each row's declared principal-pair rule proxy.",
        "units": "percentage_points_of_named_principal_pair_vote_margin",
        "close_margin_probability_2pp": "Uniform fraction of retained joint rows with abs(M) <= 2; both boundaries inclusive.",
        "first_decisive_margin_probability_10pp": "Uniform fraction of retained joint rows with M >= 10; boundary inclusive.",
        "second_decisive_margin_probability_10pp": "Uniform fraction of retained joint rows with M <= -10; boundary inclusive.",
        "candidate_identity": "First/second refer to the named candidate IDs in each cohort row. An independent first candidate is not recoded Democratic.",
        "partition": "The three events are mutually exclusive but not exhaustive. Margins between 2 and 10 points in either direction belong to none of them."
    }, "Tail-event definitions changed")
    for index, (row, source, site) in enumerate(zip(cohort, native, public, strict=True)):
        label = f"cohort[{index}]"
        old = previous_cohort[index]
        independent_row = independent[index]
        require(set(row) == expected_fields and row["race_id"] == source["race_id"] == site["id"] == old["race_id"]
                and row["state"] == source["state"] == site["state"]
                and row["target_id"] == TARGET_ID and row["included_in_native_diagnostic_cohort"] is True,
                f"{label} identity/fields changed")
        require(all(row.get(key) == old.get(key) for key in old)
                and independent_row.get("race_id") == row["race_id"]
                and independent_row.get("first_candidate_id") == row["first_candidate_id"]
                and independent_row.get("second_candidate_id") == row["second_candidate_id"],
                f"{label} prior V1 values or independent review identity changed")
        require(independent_row.get("first_party") == source["first_contender"].get("party"),
                f"{label} first candidate party identity changed")
        counts = row["margin_event_draw_counts"]
        independent_counts = independent_row.get("counts")
        probabilities = independent_row.get("probabilities")
        require(set(counts) == set(event_keys) and counts == independent_counts
                and isinstance(probabilities, dict) and set(probabilities) == set(event_keys)
                and row["margin_event_draw_denominator"] == review["native_rows"] == 40000
                and all(isinstance(counts[key], int) and not isinstance(counts[key], bool)
                        and 0 <= counts[key] <= 40000 for key in event_keys),
                f"{label} event counts or denominator changed")
        require(sum(counts[key] for key in event_keys) <= 40000,
                f"{label} disjoint event counts exceed the shared row count")
        for key in event_keys:
            near(row[key], counts[key] / 40000, f"{label}.{key} count fraction")
            near(row[key], probabilities[key], f"{label}.{key} independent fraction")
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
        require(row["first_decisive_margin_probability_10pp"] <= row["first_pair_win_probability"] + 1e-12
                and row["second_decisive_margin_probability_10pp"] <= 1 - row["first_pair_win_probability"] + 1e-12,
                f"{label} decisive probability exceeds its named candidate win probability")
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
    review_manifest = read(pinned(root, {"path": str(REVIEW_MANIFEST_REL), "sha256": REVIEW_MANIFEST_SHA256}))
    previous = read(pinned(root, {"path": str(PREVIOUS_REFERENCE_REL), "sha256": PREVIOUS_REFERENCE_SHA256}))
    previous_review = read(pinned(root, {"path": str(PREVIOUS_REVIEW_REL), "sha256": PREVIOUS_REVIEW_SHA256}))
    require(reference.get("schema") == "SenateJointScorecardReferenceV2"
            and reference.get("status") == "derived_from_qualified_joint_draws"
            and reference.get("evaluation_lock") is False
            and reference.get("observed_results_consumed") is False,
            "Reference status/meaning changed")
    require(review.get("schema") == "IndependentSenateScorecardTailReviewV1"
            and review.get("status") == "passed_scoped_v2_tail_extension"
            and review.get("checks_passed") == len(review.get("checks", [])) == 429
            and review.get("cohort_size") == 35 and review.get("probabilities_independently_reconstructed") == 105
            and review.get("maximum_probability_discrepancy") == 0
            and review.get("native_rows") == 40000 and review.get("native_shape") == [4, 10000, 35]
            and review.get("weight_per_row") == 1 / 40000
            and review.get("prior_v1_bytes_preserved") is True
            and review.get("original_joint_diagnostic_bytes_preserved") is True
            and review.get("no_actual_outcomes_historical_labels_fits_or_new_draws") is True
            and review.get("frontend_adapter_conformance") == "Not assessed; this review qualifies the v2 native reference extension only",
            "Independent reference review changed")
    require(review_manifest.get("schema") == "IndependentTailReviewManifestV1"
            and {row.get("path"): row.get("sha256") for row in review_manifest.get("files", [])}.get(str(REVIEW_REL)) == REVIEW_SHA256
            and previous_review.get("status") == "passed_scoped_reference_not_full_frontend_adapter"
            and previous_review.get("no_actual_outcomes_grades_fits_or_new_draws") is True,
            "Tail review manifest or archived V1 review changed")
    for pin in reference["sources"]:
        pinned(root, pin)
    for pin in review["sources"]:
        pinned(root, pin)
    require(reference.get("previous_reference") == {"path": str(PREVIOUS_REFERENCE_REL), "sha256": PREVIOUS_REFERENCE_SHA256}
            and reference.get("previous_cohort_sha256") == previous.get("cohort_sha256"),
            "V2 does not bind the preserved V1 reference")
    diagnostic_path = pinned(root, reference["joint_rows"])
    require(reference["joint_rows"] in review["sources"], "Reviewed diagnostic rows differ")
    review_sources = {pin["path"]: pin["sha256"] for pin in review["sources"]}
    require(review_sources.get(str(REFERENCE_REL)) == REFERENCE_SHA256
            and review_sources.get(str(PREVIOUS_REFERENCE_REL)) == PREVIOUS_REFERENCE_SHA256
            and review_sources.get(str(PREVIOUS_REVIEW_REL)) == PREVIOUS_REVIEW_SHA256,
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
            and review.get("native_shape") == [4, 10000, 35]
            and review.get("native_rows") == 40000
            and review.get("weight_per_row") == 2.5e-05
            and previous_review.get("joint_row_shape") == [4, 10000]
            and previous_review.get("total_native_rows") == 40000
            and previous_review.get("weight_per_native_row") == 2.5e-05
            and previous_review.get("exacthalf_modal_ties") == 0,
            "Original shared-row identity or weights changed")
    _validate_cohort(reference, review, previous, forecast, bundle)
    require(reference.get("metric_definitions") == previous.get("metric_definitions")
            and reference.get("distributions") == previous.get("distributions")
            and reference.get("cohort_sha256") != previous.get("cohort_sha256"),
            "V2 altered original diagnostics or failed to bind its additive cohort")
    _validate_distribution(reference, previous_review)
    near(reference["distributions"]["modal_pair_winner_miss_count"]["mean"],
         previous_review["expected_modal_pair_winner_misses"], "expected modal misses")
    return {
        "schemaVersion": 1,
        "entries": [{
            "chamber": "senate", "runId": RUN_ID, "resultSha256": PUBLIC_BUNDLE_SHA256,
            "targetId": TARGET_ID, "lawId": LAW_ID,
            "forecastCutoffAt": POLL_CUTOFF, "structuralCutoffAt": STRUCTURAL_CUTOFF,
            "forecastIssuedAt": run["issuedAt"], "referenceIssuedAt": reference["issued_utc"],
            "evaluationLock": False, "observedResultsConsumed": False,
            "sourcePins": {"referenceSha256": REFERENCE_SHA256, "reviewSha256": REVIEW_SHA256,
                           "reviewManifestSha256": REVIEW_MANIFEST_SHA256,
                           "previousReferenceSha256": PREVIOUS_REFERENCE_SHA256,
                           "previousReviewSha256": PREVIOUS_REVIEW_SHA256,
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
                          "raceMarginProbabilityDefinitions": reference["race_margin_probability_definitions"],
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
