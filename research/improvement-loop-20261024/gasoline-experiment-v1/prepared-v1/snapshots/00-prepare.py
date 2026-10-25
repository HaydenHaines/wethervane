"""Source-bound preparation of matched H0/F0/G01 aggregate comparisons; no fitting API.

Actual Senate ordinary D/R labels and pre-midnight archived TWIP gasoline are joined
under their existing qualifications. This does not qualify a count model.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PANEL = OWNER / "research/fundamentals-iteration-20261020/panel-sol"
TWIP = LAB / "gasoline-twip-cutoff-v1"
FEATURE = "president_oriented_twip_gas_log_change_13_weeks_z"
TARGETS = (2018, 2020, 2022, 2024)
TRAINING_COUNTS = {2018: 236, 2020: 266, 2022: 294, 2024: 324}
TARGET_COUNTS = {2018: 30, 2020: 28, 2022: 30, 2024: 30}
EXPECTED = {
    "panel_receipt": "3beff840bcf37aa348cd9bb8ba5b28535ba752d717d45d6a9a9f97503e9ca7f3",
    "features": "0a2a1b9c15a83f33406828e7b7e2c1b24230570ed0639c995c410b2bbe9ec786",
    "outcomes": "5f5cafa1afc53387331d4e75b3c65f92ed4cab3a91578dcb7cb0815357b029ad",
    "twip_receipt": "329ddc83d6f17b8399363f77dbaacc15c0b7f9fded29b1f2d9f859be88c4b397",
    "twip_panel": "9f4627170f846bfa6768e2bda43659a6218c4a577a2daf92989b14bd78ed3902",
    "twip_repair": "6f6f5081828537ec6f44e2293f6ad0b70fea1c03d2bacc979355b3d049b8c7a6",
    "factory": "583f4e204360fc71f980bd70a86aa3f27afaefaee13a27c5f1833b8b54ae5366",
}
SAMPLER = dict(
    chains=4,
    draws=4000,
    tune=1500,
    cores=1,
    target_accept=0.97,
    max_treedepth=10,
    max_rank_rhat=1.01,
    min_bulk_ess=400,
    min_tail_ess=400,
)
SEEDS = {2018: 2026102518, 2020: 2026102520, 2022: 2026102522, 2024: 2026102524}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def authenticated(path, digest):
    data = Path(path).read_bytes()
    require(sha(data) == digest, "source pin mismatch: " + str(path))
    return data


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path.read_bytes()))


def verify_inventory(directory, rows):
    require(isinstance(rows, list) and rows, "nonempty source inventory required")
    seen = set()
    for item in rows:
        path = (directory / item["file"]).resolve()
        require(path.is_relative_to(directory.resolve()) and path not in seen, "invalid source inventory path")
        data = authenticated(path, item["sha256"])
        require(len(data) == item["bytes"], "source size mismatch")
        seen.add(path)


def load_macro(directory=TWIP):
    """Verify the complete dated TWIP source inventory before using any row."""
    directory = Path(directory).resolve()
    receipt = json.loads(authenticated(directory / "receipt.json", EXPECTED["twip_receipt"]))
    require(receipt["schema"] == "twip_build_receipt_v1" and receipt["fit"] is False,
            "wrong TWIP source receipt")
    require(receipt["latest_GDFU_values_claimed"] is False, "wrong gasoline source arm")
    require(len(receipt["files"]) == 32, "complete 32-file TWIP inventory required")
    verify_inventory(directory, receipt["files"])
    authenticated(directory / "repair-receipt.json", EXPECTED["twip_repair"])
    panel = json.loads(authenticated(directory / "panel.json", EXPECTED["twip_panel"]))
    require(panel["schema"] == "twip_pre_cutoff_panel_v1" and panel["fit"] is False,
            "wrong TWIP panel")
    result = {}
    for row in panel["panel"]:
        cycle = row["cycle"]
        require(cycle not in result and row["status"] == "available" and
                row["schema"] == "twip_pre_cutoff_gasoline_contrast_v1" and
                row["arm"] == "older_verified_dated_TWIP_issue_not_latest_GDFU" and
                row["strict_original_posting_authenticated"] is False,
                "unsupported TWIP row or duplicate cycle")
        require(row["crosses_2018_method_break"] is False, "gasoline method break crossed")
        current, lag = row["current"], row["lag13"]
        for item in (current, lag):
            require(item["series_id"] == "US_regular_retail_gasoline_TWIP" and
                    item["geography"] == "US" and item["grade"] == "regular" and
                    item["formulations"] == "national_regular_summary_not_separately_reconstructed" and
                    item["units"] == "USD_per_gallon" and
                    item["original_posting_log_authenticated"] is False,
                    "wrong regular-retail gasoline series or units")
            require(item["method_regime"] == ("pre_2018_05_14" if cycle < 2018 else "post_2018_05_14"),
                    "gasoline method regime mismatch")
            require(isinstance(item["value"], (int, float)) and np.isfinite(item["value"]) and
                    item["value"] > 0, "invalid gasoline price")
            source = item["source"]
            raw = (directory / source["file"]).resolve()
            require(raw.is_relative_to(directory) and source["url"].startswith("https://www.eia.gov/petroleum/weekly/archive/"),
                    "wrong TWIP raw source")
            require(sha(raw.read_bytes()) == source["sha256"], "TWIP source row/raw mismatch")
            require(item["issue_date"] == source["issue_date"], "TWIP issue/source mismatch")
        current_date = datetime.fromisoformat(current["observation_date"])
        lag_date = datetime.fromisoformat(lag["observation_date"])
        require(current_date.weekday() == lag_date.weekday() == 0 and
                (current_date - lag_date).days == 91,
                "non-Monday or nonexact 91-day gasoline contrast")
        require(current["method_regime"] == lag["method_regime"], "method regimes differ")
        require(np.isclose(np.log(current["value"] / lag["value"]), row["log_change_13_weeks"],
                           rtol=0, atol=1e-12), "gasoline log-change arithmetic mismatch")
        require(np.isclose(current["value"] - lag["value"], row["change_USD_per_gallon"],
                           rtol=0, atol=1e-12), "gasoline dollar arithmetic mismatch")
        date_only_cutoff = datetime.fromisoformat(row["cutoff_utc"])
        midnight = date_only_cutoff.replace(hour=0, minute=0, second=0)
        require(date_only_cutoff.time().isoformat() == "23:59:59" and date_only_cutoff.tzinfo == timezone.utc,
                "TWIP panel calendar cutoff mismatch")
        for item in (current, lag):
            require(datetime.fromisoformat(item["available_upper_bound_utc"]) < midnight,
                    "TWIP release not before exact midnight H14")
            require(datetime.fromisoformat(item["issue_date"]).replace(tzinfo=timezone.utc) < midnight,
                    "TWIP issue after cutoff")
        if cycle == 2004:
            exception = current["issue_date_exception"]
            require(exception["id"] == "twip_2004_10_14_stale_header_year_v1" and
                    exception["source_sha256"] == current["source"]["sha256"] and
                    exception["admitted_issue_date"] == "2004-10-14" and
                    exception["source_header_issue_date"] == "2003-10-14" and
                    exception["observation_date"] == "2004-10-11" and
                    lag["issue_date_exception"] is None,
                    "2004 exact reviewed header exception changed")
        else:
            require(current["issue_date_exception"] is None and lag["issue_date_exception"] is None and
                    current["source_header_issue_date"] == current["issue_date"] and
                    lag["source_header_issue_date"] == lag["issue_date"],
                    "unreviewed TWIP header exception")
        result[cycle] = dict(value=float(row["log_change_13_weeks"]),
                             cutoff_utc=midnight.isoformat(), source=row)
    require(set(result) == set(range(2002, 2026, 2)), "complete 2002-2024 gasoline support required")
    return result


def load_features():
    authenticated(PANEL / "receipt.json", EXPECTED["panel_receipt"])
    path = PANEL / "features/all-cycles.csv"
    authenticated(path, EXPECTED["features"])
    frame = pd.read_csv(path)
    require(
        not {"actual_dem_share", "dem_votes", "rep_votes", "winner"}.intersection(frame.columns), "labels in features"
    )
    return frame.loc[frame.cycle.le(2024)].copy()


def earlier_outcomes(target):
    require(target in TARGETS, "unsupported replay target")
    path = PANEL / "labels/outcomes-SEPARATE.csv"
    authenticated(path, EXPECTED["outcomes"])
    metadata = pd.read_csv(path, usecols=["cycle"])
    keep = set(np.flatnonzero(metadata.cycle.ge(2002) & metadata.cycle.lt(target)) + 1)
    return pd.read_csv(
        path, usecols=["race_id", "cycle", "actual_dem_share"], skiprows=lambda line: line > 0 and line not in keep
    )


def transformed_features(frame, macro, target):
    require(target in TARGETS, "unsupported target; current 2026 inputs are absent")
    frame = frame.loc[frame.cycle.ge(2002) & frame.cycle.le(target)].copy()
    require(
        frame.eligible.eq(True).all() and not frame.race_id.duplicated().any(),
        "actual unique admitted feature rows required",
    )
    values = {}
    for cycle, group in frame.groupby("cycle"):
        require(
            group.president_party_asof.nunique() == 1 and group.president_party_asof.isin([-1, 1]).all(),
            "one dated D+1/R-1 presidential party per cycle required",
        )
        require(
            group.forecast_cutoff.nunique() == 1
            and pd.Timestamp(group.forecast_cutoff.iloc[0], tz="UTC") == pd.Timestamp(macro[cycle]["cutoff_utc"]),
            "row economic cutoff differs from original Senate cutoff",
        )
        values[int(cycle)] = float(group.president_party_asof.iloc[0]) * macro[cycle]["value"]
    earlier = sorted(cycle for cycle in values if cycle < target)
    require(earlier == list(range(2002, target, 2)), "complete strictly earlier unique-cycle support required")
    oriented = np.array([values[cycle] for cycle in earlier])
    mean, sd = float(oriented.mean()), float(oriented.std(ddof=0))
    require(np.isfinite([mean, sd]).all() and sd > 0, "nondegenerate unique-cycle scale required")
    frame[FEATURE] = frame.cycle.map({cycle: (value - mean) / sd for cycle, value in values.items()})
    scaler = dict(
        raw_units="natural_log_US_regular_retail_price_ratio_exact91days",
        orientation="president_party_asof_D_plus1_R_minus1",
        operation="orient_then_standardize",
        training_cycles=earlier,
        mean=mean,
        sd=sd,
        ddof=0,
        independent_macro_realizations=len(earlier),
        weighting="one_equal_weight_per_unique_earlier_cycle",
    )
    return frame, scaler


def factory():
    path = OWNER / "src/forecasting/aggregate_component_factory_v1.py"
    authenticated(path, EXPECTED["factory"])
    if str(OWNER) not in sys.path:
        sys.path.insert(0, str(OWNER))
    module = importlib.import_module("src.forecasting.aggregate_component_factory_v1")
    require(Path(module.__file__).resolve() == path, "wrong factory source loaded")
    return module


def recipes(module):
    baseline = module.Recipe(providers=module.standard_providers())
    economic = module.ProviderSpec(
        instance_id="economy",
        provider_id="twip_gasoline_oriented_mean_v1",
        source_sha256=sha(Path(__file__).read_bytes()),
        role="environment",
        kind="mean",
        kernel="linear",
        parameters=(module.ParameterSpec(parameter_id="economy.twip_gas_change", owner="economy", mean=0, sd=0.05),),
        terms=(
            module.MeanTerm(
                term_id="economy.twip_gas_change.term",
                owner="economy",
                feature_id=FEATURE,
                coefficient=module.ParameterRef(parameter_id="economy.twip_gas_change"),
            ),
        ),
        covariate_evidence_ids=("eia:TWIP:US_regular_retail_gasoline:exact_midnight_H14",),
    )
    return dict(
        H0=module.legacy_recipe("senate_history_refreshed_v1"),
        F0=baseline,
        G01=module.Recipe(providers=(*baseline.providers, economic)),
    )


def compile_fold(target, macro=None):
    module = factory()
    frame, scaler = transformed_features(load_features(), macro or load_macro(), target)
    outcomes = earlier_outcomes(target)
    feature_hash = module.frame_digest(frame)
    evidence = dict(
        schema="G01TWIPSourceJoinPreparationEvidenceV1",
        status="not_execution_or_count_admission",
        panel_receipt_sha256=EXPECTED["panel_receipt"],
        twip_receipt_sha256=EXPECTED["twip_receipt"],
        twip_panel_sha256=EXPECTED["twip_panel"],
        twip_repair_sha256=EXPECTED["twip_repair"],
        feature_rows_sha256=feature_hash,
        target_cycle=target,
        scaler=scaler,
        adapter_sha256=sha(Path(__file__).read_bytes()),
    )
    admission = module.Admission(
        cutoff_cycle=target,
        cutoff_utc=str(pd.Timestamp(frame.loc[frame.cycle.eq(target), "forecast_cutoff"].iloc[0], tz="UTC")),
        support_id="actual_senate_ordinary_DR_panel_2002plus_matched_TWIP",
        feature_source_sha256=feature_hash,
        outcome_source_sha256=EXPECTED["outcomes"],
        receipt_sha256=sha(canonical(evidence)),
        roster_qualification="qualified_retrospective_proxy_not_count_admission",
        extra_feature_ids=(FEATURE,),
        start_cycle=2002,
    )
    compiled = {
        name: module.compile_training(recipe, frame, outcomes, admission) for name, recipe in recipes(module).items()
    }
    ids = compiled["F0"].race_ids
    require(
        len(ids) == TRAINING_COUNTS[target] and all(value.race_ids == ids for value in compiled.values()),
        "matched actual training support differs",
    )
    require(len(frame.loc[frame.cycle.eq(target)]) == TARGET_COUNTS[target], "actual target support differs")
    return module, compiled, frame, scaler, evidence


def prior_diagnostics(target, module, compiled, frame, scaler, macro):
    """Outcome-blind prior draws and exact mean-shift checks; no fitting."""
    training = frame.loc[frame.cycle.lt(target)].drop_duplicates("cycle").sort_values("cycle")
    target_frame = frame.loc[frame.cycle.eq(target)].copy()
    cycle_ids = training.cycle.astype(int).tolist()
    gas = np.array([macro[cycle]["value"] for cycle in cycle_ids])
    party = training.president_party_asof.to_numpy(float)
    chosen = (party * gas - scaler["mean"]) / scaler["sd"]
    raw_mean, raw_sd = float(gas.mean()), float(gas.std(ddof=0))
    target_party = float(target_frame.president_party_asof.iloc[0])
    target_value = macro[target]["value"]
    target_z = float(target_frame[FEATURE].iloc[0])
    alternate_z = target_party * (target_value - raw_mean) / raw_sd
    shifts = {}
    for label, coordinate in (("primary", target_z), ("raw_first_diagnostic_only", alternate_z)):
        radius = float(1.959963984540054 * 0.05 * abs(coordinate))
        shifts[label] = dict(coordinate=float(coordinate), coefficient_sd=0.05,
                             mean_logit_shift=0,
                             central95_logit_shift=[-radius, radius],
                             central95_shifted_share_at_baseline_half=expit(np.array([-radius, radius])).tolist())
    other = dict(party=party, party_midterm=party * training.midterm.to_numpy(float),
                 national=training.prior_national_presidential_logit.to_numpy(float))
    correlations = {name: float(np.corrcoef(chosen, value)[0, 1]) for name, value in other.items()}
    require(np.isfinite(list(correlations.values())).all(), "nonfinite gas/context prior correlation")
    rng = np.random.default_rng(SEEDS[target])
    draws = 4096
    parameters = {p.parameter_id: p for item in compiled.values() for p in item.recipe.parameters}
    standard = {name: rng.normal(size=draws) for name in sorted(parameters)}
    shared = rng.normal(size=draws) * abs(rng.normal(0, 0.2, size=draws))
    residual = rng.standard_t(5, size=(draws, len(target_frame))) * abs(rng.normal(0, 0.5, size=draws))[:, None]
    summaries, locations = {}, {}

    def summary(logits):
        shares = expit(logits)
        return dict(share_p01_p10_p50_p90_p99=np.quantile(shares, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
                    fraction_share_below01_or_above99=float(((shares < 0.01) | (shares > 0.99)).mean()),
                    minimum_race_prior_mean=float(shares.mean(axis=0).min()),
                    maximum_race_prior_mean=float(shares.mean(axis=0).max()))

    for name, item in compiled.items():
        offset, design = module._design(item.recipe, target_frame, item.admission)
        beta = np.column_stack([p.mean + p.sd * standard[p.parameter_id] for p in item.recipe.parameters])
        location = offset + beta @ design.T
        locations[name] = location
        summaries[name] = summary(location + shared[:, None] + residual)
    summaries["raw_first_diagnostic_only"] = summary(
        locations["F0"] + 0.05 * standard["economy.twip_gas_change"][:, None] * alternate_z
        + shared[:, None] + residual
    )
    return dict(role="prior_predictive_diagnostic_not_fit_or_score", draws=draws, seed=SEEDS[target],
                mean_shift_checks=shifts, raw_target_log_change=target_value,
                earlier_raw_range=[float(gas.min()), float(gas.max())],
                target_outside_earlier_raw_range=not float(gas.min()) <= target_value <= float(gas.max()),
                raw_gas_mean=raw_mean, raw_gas_sd=raw_sd,
                unique_cycle_correlations_with_chosen_coordinate=correlations,
                prior_predictive=summaries,
                comparison="Raw-first independent same-SD prior has different office/party offsets and slope prior")


def prepare(output):
    # Validate and compile every real fold before creating any output directory.
    macro = load_macro()
    folds = {target: compile_fold(target, macro) for target in TARGETS}
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "preparation must stay in gasoline-experiment-v1")
    output.mkdir(parents=True, exist_ok=False)
    twip_receipt = json.loads(authenticated(TWIP / "receipt.json", EXPECTED["twip_receipt"]))
    pins = [
        record(path)
        for path in [
            Path(__file__),
            HERE / "test_prepare.py",
            PANEL / "receipt.json",
            PANEL / "features/all-cycles.csv",
            PANEL / "labels/outcomes-SEPARATE.csv",
            TWIP / "receipt.json",
            TWIP / "panel.json",
            TWIP / "repair-receipt.json",
            OWNER / "src/forecasting/aggregate_component_factory_v1.py",
            OWNER / "src/forecasting/fundamentals_v1.py",
            OWNER / "src/forecasting/baseline_v5.py",
            OWNER / "src/forecasting/polling_basis_v1.py",
            OWNER / "src/model_stack/contracts.py",
        ]
    ] + [record(TWIP / item["file"]) for item in twip_receipt["files"]]
    snapshots = output / "snapshots"
    snapshots.mkdir()
    for index, item in enumerate(pins):
        path = Path(item["path"])
        destination = snapshots / f"{index:02d}-{path.name}"
        with destination.open("xb") as stream:
            stream.write(authenticated(path, item["sha256"]))
    plan = dict(
        schema="G01AggregatePreparatoryExperimentV1",
        status="prepared_not_fitted",
        fit_authorized=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        target_id="senate_aggregate_logit_empirical_mixture_v1",
        scope="ordinary_D_R_aggregate_preparation_only_not_population_cut_or_count_qualification",
        arms={
            "H0": "exact_existing_refreshed_history",
            "F0": "exact_existing_relative_context",
            "G01": "F0_plus_one_president_oriented_13week_TWIP_gas_mean",
        },
        feature_id=FEATURE,
        economic_prior=dict(family="Normal", mean=0, sd=0.05, sign_restriction=None),
        sampler_proposal=SAMPLER,
        fit_diagnostic_gates=dict(
            max_rank_rhat=1.01, min_bulk_ess=400, min_tail_ess=400, divergences=0, max_depth_hits=0
        ),
        initial_execution_stage="F_only; FP_poll_factor_bindings_require_a_separate_reviewed_prediction_plan",
        conditioning=(
            "Each arm jointly infers coefficients, earlier-cycle effects and shared cycle/race scales "
            "given only earlier labels and frozen covariates."
        ),
        label_sealing=(
            "For each fold, seal all H0/F0/G01 predictive draws before target scoring. "
            "Earlier evaluation cycles can train later folds. "
            "This preparation is not a globally unopened-label holdout."
        ),
        runtime_preparation=dict(
            python=sys.version,
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "pydantic")},
            fitter_runtime="Existing E01 runtime may be reused only with explicit exact package and new-plan review",
        ),
        fitting_budget="same complete joint fit budget for all 12 arm/fold combinations",
        prediction_sequence="F first; same fold-specific reviewed poll factor for later matched FP comparisons",
        primary_metric="share_CRPS",
        tiers={"A": [2018, 2022], "B": [2020, 2024]},
        score_controls=[
            "WIS",
            "coverage_and_width",
            "winner_Brier_and_log_score",
            "always_0.5",
            "earlier_only_frequency",
            "fixed_common_forecast_time_competitive_mask",
            "near_even_margins",
        ],
        source_pins=pins,
        folds=[],
        limitations=[
            "Final roster and archive provenance remain retrospective qualifications.",
            "National feature has election-cycle replication, not independent race-row replication.",
            "No CPI/core/income/state interaction, extra national shock, bank splicing, "
            "post-poll shift or current2026 feature.",
            "TWIP archives are dated retrospective evidence, not authenticated original posting logs.",
            "The 2004 exact stale-header exception and 2018 gasoline method regime remain explicit.",
            "Factory Admission object is compiled contract metadata, not a count execution/source certificate.",
            "An aggregate win cannot qualify or promote full count fundamentals.",
        ],
        public_activation=False,
        count_fit_launched=False,
        mcmc_launched=False,
    )
    for target, (module, compiled, frame, scaler, evidence) in folds.items():
        filename = f"features-through-{target}-LABEL-FREE.csv"
        with (output / filename).open("x") as stream:
            frame.to_csv(stream, index=False)
        evidence_path = output / f"feature-join-{target}-evidence.json"
        with evidence_path.open("xb") as stream:
            stream.write(canonical(evidence))
        item = dict(
            target_cycle=target,
            tier="A" if target in (2018, 2022) else "B",
            scaler=scaler,
            sampler=module.SamplerPlan(**SAMPLER, seed=SEEDS[target]).model_dump(mode="json"),
            prior_review=prior_diagnostics(target, module, compiled, frame, scaler, macro),
            features=record(output / filename),
            join_evidence=record(evidence_path),
            target_race_ids=frame.loc[frame.cycle.eq(target), "race_id"].tolist(),
            arms={},
        )
        for name, training in compiled.items():
            item["arms"][name] = dict(
                recipe=training.recipe.payload(),
                law_hash=training.law_hash,
                parameter_names=list(training.parameter_names),
                training_race_ids=list(training.race_ids),
                cycles=training.cycles.tolist(),
                feature_rows_sha256=training.feature_rows_sha256,
                outcome_rows_sha256=training.outcome_rows_sha256,
                design_sha256=sha(training.design.tobytes()),
                response_sha256=sha(training.response.tobytes()),
                implementation_sha256=training.implementation_sha256,
            )
        plan["folds"].append(item)
    with (output / "plan.json").open("xb") as stream:
        stream.write(canonical(plan))
    print(json.dumps(dict(status="prepared_not_fitted", plan=record(output / "plan.json"), mcmc_launched=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared-v1")
    prepare(parser.parse_args().output)
