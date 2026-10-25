# ruff: noqa: E402
# Private imports intentionally follow source snapshot path binding.
"""Receipt-first score orchestration. Author verification is synthetic only."""

import argparse
import datetime
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "snapshots"))
sys.path.insert(0, str(HERE))
import numpy as np
from marginal import GaussianLogitMixture
from score_math import (
    METRICS,
    ScaledInfluence,
    equal_cycle_summary,
    paired_difference,
    score_empirical_forecast,
    score_gaussian_forecast,
    winner_reference_scores,
)
from score_reader import ARMS, read_labels_after_admission, require, verify_admission


def _plain(value):
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    require(not isinstance(value, ScaledInfluence), "Influence traces belong in numeric NPZ, not JSON objects")
    return value


def calibration_and_width(rows):
    """Descriptive fixed-mask summaries; no race-iid inference claim."""
    eligible = [row for row in rows if not row["tie"]]
    bins = []
    for index in range(10):
        lo, hi = index / 10, (index + 1) / 10
        selected = [
            r
            for r in eligible
            if lo <= r["win_probability"] and (r["win_probability"] < hi if index < 9 else r["win_probability"] <= hi)
        ]
        bins.append(
            dict(
                lower=lo,
                upper=hi,
                count=len(selected),
                mean_probability=None if not selected else float(np.mean([r["win_probability"] for r in selected])),
                observed_fraction=None
                if not selected
                else float(np.mean([r["actual_dem_share"] > 0.5 for r in selected])),
            )
        )
    intervals = {}
    for alpha in [0.5, 0.2, 0.05]:
        selected = [r for r in rows if r.get("quantiles")]
        intervals[str(1 - alpha)] = dict(
            count=len(selected),
            coverage=None
            if not selected
            else float(
                np.mean(
                    [
                        r["quantiles"][alpha / 2] <= r["actual_dem_share"] <= r["quantiles"][1 - alpha / 2]
                        for r in selected
                    ]
                )
            ),
            mean_width=None
            if not selected
            else float(np.mean([r["quantiles"][1 - alpha / 2] - r["quantiles"][alpha / 2] for r in selected])),
        )
    return dict(
        win_calibration_bins=bins, interval_coverage_and_width=intervals, independent_election_count_not_race_count=True
    )


def _band_probability(model, lower, upper):
    lo_cdf, lo_sf = model.logtails(lower)
    hi_cdf, hi_sf = model.logtails(upper)
    # Prefer the smaller cumulative side, retaining tiny interval mass.
    if hi_cdf <= lo_sf:
        return float(np.exp(hi_cdf) * (-np.expm1(lo_cdf - hi_cdf)))
    return float(np.exp(lo_sf) * (-np.expm1(hi_sf - lo_sf)))


def gaussian_descriptives(report, means, sds, truths, expected):
    require(expected.shape == means.shape, "Saved expected-share component axes required")
    for j, row in enumerate(report["rows"]):
        model = GaussianLogitMixture(means[..., j].reshape(-1), sds[..., j].reshape(-1))
        row["expected_margin"] = float(2 * expected[..., j].mean() - 1)
        row["median_margin"] = 2 * row["quantiles"][0.5] - 1
        row["p_absolute_margin_le_02"] = _band_probability(model, 0.49, 0.51)
        row["p_absolute_margin_ge_10"] = float(model.cdf(0.45) + model.sf(0.55))
        logcdf, logsf = model.logtails(truths[j])
        row["PIT"] = dict(kind="continuous_CDF", value=float(np.exp(logcdf)), log_cdf=logcdf, log_survival=logsf)
        row["margin_intervals"] = {
            str(1 - alpha): [2 * row["quantiles"][alpha / 2] - 1, 2 * row["quantiles"][1 - alpha / 2] - 1]
            for alpha in [0.5, 0.2, 0.05]
        }


def empirical_descriptives(report, values, truths):
    for j, row in enumerate(report["rows"]):
        pooled = values[..., j].reshape(-1)
        left = float(np.mean(pooled < truths[j]))
        right = float(np.mean(pooled <= truths[j]))
        row["PIT"] = dict(
            kind="discrete_CDF_bracket_no_randomization", lower=left, upper=right, midpoint=0.5 * (left + right)
        )
        row["expected_margin"] = float(2 * pooled.mean() - 1)
        row["median_margin"] = 2 * row["quantiles"][0.5] - 1
        row["p_absolute_margin_le_02"] = float(np.mean(abs(2 * pooled - 1) <= 0.02))
        row["p_absolute_margin_ge_10"] = float(np.mean(abs(2 * pooled - 1) >= 0.10))
        row["margin_intervals"] = {
            str(1 - alpha): [2 * row["quantiles"][alpha / 2] - 1, 2 * row["quantiles"][1 - alpha / 2] - 1]
            for alpha in [0.5, 0.2, 0.05]
        }


def apply_prelabel_status(report, *, core_bulk_qualified, frozen_failure_disposition=None):
    """Math precision cannot override a failed prelabel forecast computation."""
    failed = not core_bulk_qualified or frozen_failure_disposition is not None
    report["prelabel_core_bulk_qualified"] = bool(core_bulk_qualified)
    report["frozen_failure_disposition"] = frozen_failure_disposition
    if failed:
        for metric in METRICS:
            row = report["aggregates"][metric]
            row["score_math_status"] = row["status"]
            row["qualified"] = False
            row["status"] = "prelabel_core_bulk_or_frozen_experiment_unresolved"
    report["calibration_and_width"] = calibration_and_width(report["rows"])
    return report


def score_verified(bundle, label_result):
    """Exact frozen common/subset masks; no outcome-dependent successful subset."""
    output = dict(
        schema="MomentumMatchedForecastScoresV1",
        cycles={},
        tiers={},
        primary_differences={},
        target="conditional_aggregate_DR_share_no_full_chamber_claim",
        post_label_sampling=False,
        deterministic_quad_error_separate=True,
        inherited_bank_and_calibration_scale_error_separate=True,
    )
    all_influences = {}
    for cycle, fold in bundle["folds"].items():
        full = tuple(fold["mask"]["full_bank_race_ids"])
        common = tuple(fold["mask"]["common_polled_race_ids"])
        subset_specs = {
            "common": common,
            "competitive": tuple(fold["mask"]["competitive_race_ids"]),
            "near_even": tuple(fold["mask"]["near_even_race_ids"]),
        }
        cycle_report = dict(
            selected_attempt=fold["selected_attempt"], full_bank_race_count=len(full), masks={}, references={}
        )
        for subset, ids in subset_specs.items():
            y = np.array([label_result["labels"][rid]["actual_dem_share"] for rid in ids])
            common_index = np.array([common.index(rid) for rid in ids], dtype=int)
            full_index = np.array([full.index(rid) for rid in ids], dtype=int)
            reports = {}
            for arm in ARMS:
                job = fold["jobs"][arm]
                report = score_gaussian_forecast(
                    job["means"][..., common_index],
                    job["sds"][..., common_index],
                    y,
                    ids,
                    [job["win_tail_qualified"][j] for j in common_index],
                    [job["lose_tail_qualified"][j] for j in common_index],
                )
                gaussian_descriptives(
                    report,
                    job["means"][..., common_index],
                    job["sds"][..., common_index],
                    y,
                    job["components"]["component_expected_shares"][..., common_index],
                )
                apply_prelabel_status(
                    report,
                    core_bulk_qualified=job["core_bulk_qualified"],
                    frozen_failure_disposition=fold["frozen_failure_disposition"],
                )
                report["prelabel_interval_display_qualified"] = job["summary"]["interval_display_qualified"]
                report["inherited_bank_warning"] = job["summary"]["inherited_bank_warning"]
                report["prelabel_numerical_status_pin"] = job["numerical_status_pin"]
                reports[arm] = report
            joint = fold["f0"]["joint_empirical_shares"]
            require(joint.shape[1] % 5 == 0, "Contiguous archived five-replicate grouping required")
            archived = joint.reshape(4, joint.shape[1] // 5, 5, len(full))[..., full_index]
            f0_summary = fold["f0summary"]
            win_rows = {r["coordinate"]: r for r in f0_summary["win_side"]["coordinates"]}
            lose_rows = {r["coordinate"]: r for r in f0_summary["lose_side"]["coordinates"]}

            def f0_flag(row):
                return row["status"] in ("measured", "structural") and all(
                    row["gates"].get(k) is True for k in ["relative_mcse", "chain_log_difference"]
                )

            reports["F0_archived_predictive"] = score_empirical_forecast(
                archived, y, ids, [f0_flag(win_rows[r]) for r in ids], [f0_flag(lose_rows[r]) for r in ids]
            )
            empirical_descriptives(reports["F0_archived_predictive"], archived, y)
            reports["F0_archived_predictive"]["calibration_and_width"] = calibration_and_width(
                reports["F0_archived_predictive"]["rows"]
            )
            po = fold["po5"]
            means = po["mean_logit"][full_index]
            sd = np.sqrt(np.diag(po["covariance"])[full_index])
            reports["PO5"] = score_gaussian_forecast(
                means[None, None, :],
                sd[None, None, :],
                y,
                ids,
                [True] * len(ids),
                [True] * len(ids),
                deterministic=True,
            )
            saved_po5 = {row["race_id"]: row["expected_share"] for row in fold["po5summary"]["rows"]}
            expected_po5 = np.array([saved_po5[r] for r in ids])[None, None, :]
            gaussian_descriptives(reports["PO5"], means[None, None, :], sd[None, None, :], y, expected_po5)
            reports["PO5"]["calibration_and_width"] = calibration_and_width(reports["PO5"]["rows"])
            differences = {
                metric: paired_difference(reports["FS6"], reports["FS5"], metric, independent_streams=True)
                for metric in METRICS
            }
            cycle_report["masks"][subset] = dict(
                race_ids=list(ids),
                binary_ties=int(np.sum(y == 0.5)),
                arms=reports,
                primary_FS6_minus_FS5=differences,
                empty_mask_unavailable=not bool(ids),
                winner_half=winner_reference_scores(0.5, y),
                winner_frequency=winner_reference_scores(label_result["frequency_references"][cycle]["probability"], y),
            )
            for arm, report in reports.items():
                for metric, influence in report.pop("cycle_influences").items():
                    if influence is not None:
                        key = f"{cycle}__{subset}__{arm}__{metric}"
                        all_influences[key] = influence
        cycle_report["frequency_reference"] = label_result["frequency_references"][cycle]
        output["cycles"][cycle] = cycle_report
    for tier, cycles in [("A", (2018, 2022)), ("B", (2020, 2024))]:
        tier_rows = {}
        for subset in ["common", "competitive", "near_even"]:
            arm_rows = {}
            for arm in [*ARMS, "F0_archived_predictive", "PO5"]:
                selected = {
                    cycle: output["cycles"][cycle]["masks"][subset]["arms"][arm]
                    for cycle in cycles
                    if cycle in output["cycles"]
                }
                arm_rows[arm] = {metric: equal_cycle_summary(selected, metric, tier=tier) for metric in METRICS}
            combined_ids = [
                f"{cycle}:{rid}"
                for cycle in cycles
                if cycle in output["cycles"]
                for rid in output["cycles"][cycle]["masks"][subset]["race_ids"]
            ]
            primary = {
                metric: paired_difference(
                    dict(mask_race_ids=combined_ids, aggregates=arm_rows["FS6"]),
                    dict(mask_race_ids=combined_ids, aggregates=arm_rows["FS5"]),
                    metric,
                    independent_streams=True,
                )
                for metric in METRICS
            }
            winner_refs = {}
            for reference in ["winner_half", "winner_frequency"]:
                winner_refs[reference] = {
                    metric: None
                    if any(
                        cycle not in output["cycles"]
                        or output["cycles"][cycle]["masks"][subset][reference][metric] is None
                        for cycle in cycles
                    )
                    else float(
                        np.mean([output["cycles"][cycle]["masks"][subset][reference][metric] for cycle in cycles])
                    )
                    for metric in ["brier", "negative_log"]
                }
            tier_rows[subset] = dict(
                arms=arm_rows,
                cycles=list(cycles),
                no_empty_cycle_drop=True,
                primary_FS6_minus_FS5=primary,
                winner_references=winner_refs,
            )
            output["primary_differences"].setdefault(tier, {})[subset] = primary
        output["tiers"][tier] = tier_rows
    output["label_audit"] = label_result["audit"]
    return output, all_influences


def run_scoring(admission_pin, output, *, allow_actual=False):
    """Exclusive authentic score action; only Root may later allow actual labels."""
    destination = Path(output)
    require(not destination.exists(), "Scoring output must be new before label access")
    bundle = verify_admission(admission_pin)
    require(
        bundle["admission"]["synthetic"] or allow_actual, "Actual labels require an explicit later Root scoring action"
    )
    destination.mkdir(parents=True, exist_ok=False)

    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    launch = dict(
        schema="MomentumScoringLaunchV1",
        utc=now(),
        pid=os.getpid(),
        admission=admission_pin,
        scoring_manifest=bundle["admission"]["scoring_manifest"],
        independent_scoring_review=bundle["admission"]["independent_scoring_review"],
        numeric_labels_opened=False,
        post_label_sampling=False,
    )
    (destination / "launch.json").write_text(json.dumps(_plain(launch), sort_keys=True, indent=2) + "\n")
    opened = False
    invoked = False
    try:
        invoked = True
        labels = read_labels_after_admission(bundle)
        opened = True
        (destination / "label-access.json").write_text(
            json.dumps(
                dict(
                    schema="MomentumNumericLabelAccessV1",
                    utc=now(),
                    label_source=_plain(bundle["admission"]["label_source"]),
                    all_required_receipts_verified=True,
                ),
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        report, influences = score_verified(bundle, labels)
        arrays = {key: value.values for key, value in influences.items()}
        scales = {key: value.log_scale for key, value in influences.items()}
        np.savez_compressed(destination / "cycle-influences.npz", **arrays)
        (destination / "cycle-influence-scales.json").write_text(json.dumps(scales, sort_keys=True, indent=2) + "\n")
        (destination / "scores.json").write_text(
            json.dumps(_plain(report), sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        import hashlib

        receipt = dict(
            schema="MomentumScoringOutputReceiptV1",
            admission=admission_pin,
            artifacts={
                name: dict(
                    path=str(destination / name), sha256=hashlib.sha256((destination / name).read_bytes()).hexdigest()
                )
                for name in [
                    "scores.json",
                    "cycle-influences.npz",
                    "cycle-influence-scales.json",
                    "launch.json",
                    "label-access.json",
                ]
            },
            source_and_numeric_status_failures_distinct=True,
            post_label_sampling=False,
        )
        (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        (destination / "terminal.json").write_text(
            json.dumps(
                dict(
                    schema="MomentumScoringTerminalV1",
                    status="complete_scores",
                    utc=now(),
                    admission=admission_pin,
                    numeric_labels_opened=True,
                    receipt=receipt,
                ),
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        return receipt
    except BaseException as error:
        terminal = dict(
            schema="MomentumScoringTerminalV1",
            status="failed_preserved_no_repair_or_sampling",
            utc=now(),
            admission=admission_pin,
            numeric_label_reader_returned=opened,
            numeric_label_reader_invoked=invoked,
            labels_may_have_been_opened=invoked,
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
            post_label_sampling=False,
        )
        (destination / "terminal.json").write_text(json.dumps(terminal, sort_keys=True, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-actual", action="store_true")
    args = parser.parse_args()
    run_scoring(dict(path=args.admission, sha256=args.expected_sha256), args.output, allow_actual=args.allow_actual)


if __name__ == "__main__":
    main()
