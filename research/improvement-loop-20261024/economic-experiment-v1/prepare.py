"""Source-bound preparation of H0/F0/E01 aggregate comparisons; no fitting API.

Actual Senate ordinary D/R labels and exact-midnight archived CPI are joined
under their existing qualifications. This does not qualify a count model.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PANEL = OWNER / "research/fundamentals-iteration-20261020/panel-sol"
VINTAGES = LAB / "economic-cutoff-adapter-v2"
FEATURE = "president_oriented_headline_cpi_yoy_z"
TARGETS = (2018, 2020, 2022, 2024)
TRAINING_COUNTS = {2018: 299, 2020: 329, 2022: 357, 2024: 387}
TARGET_COUNTS = {2018: 30, 2020: 28, 2022: 30, 2024: 30}
EXPECTED = {
    "panel_receipt": "3beff840bcf37aa348cd9bb8ba5b28535ba752d717d45d6a9a9f97503e9ca7f3",
    "features": "0a2a1b9c15a83f33406828e7b7e2c1b24230570ed0639c995c410b2bbe9ec786",
    "outcomes": "5f5cafa1afc53387331d4e75b3c65f92ed4cab3a91578dcb7cb0815357b029ad",
    "vintage_receipt": "4646bc2ea152f441a1b34388ce4bd5693306a065947c3f0d17a09c1f781ac733",
    "selections": "208c02593a8f39771ae23248ec2c5608aa493948cda2d90d64a53fdc3fbf0a0b",
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


def load_macro(directory=VINTAGES):
    directory = Path(directory).resolve()
    receipt = json.loads(authenticated(directory / "receipt.json", EXPECTED["vintage_receipt"]))
    require(
        receipt["schema"] == "economic_cutoff_adapter_v2_receipt_v1" and receipt["fit"] is False,
        "wrong economic source receipt",
    )
    verify_inventory(directory, receipt["files"])
    # Verify the exact inherited v1 receipt and its original source-byte inventory.
    for key in ("v1_parser", "v1_rows", "v1_selections", "v1_receipt"):
        item = receipt["input_manifest"][key]
        path = (directory / item["file"]).resolve()
        require(path.is_relative_to(directory.parent), "inherited input outside economic package root")
        authenticated(path, item["sha256"])
        if key == "v1_receipt":
            original = json.loads(path.read_bytes())
            verify_inventory(path.parent, original["source_files"])
    selections = json.loads(authenticated(directory / "selections.json", EXPECTED["selections"]))
    require(selections["schema"] == "bls_cpi_h14_midnight_selection_v2", "wrong cutoff adapter schema")
    result = {}
    for item in selections["selections"]:
        if item["kind"] != "headline":
            continue
        cycle, row = item["cycle"], item["row"]
        require(cycle not in result and item["status"] == "available", "missing or duplicate headline cycle")
        expected_cutoff = datetime.fromisoformat(item["election_date"]).replace(tzinfo=timezone.utc) - timedelta(
            days=14
        )
        cutoff = datetime.fromisoformat(item["cutoff_utc"].replace("Z", "+00:00"))
        require(cutoff == expected_cutoff and item["horizon_days"] == 14, "exact midnight H14 cutoff required")
        require(
            row["series_id"] == "CUUR0000SA0"
            and row["kind"] == "headline"
            and row["units"] == "percent_change_12_months"
            and row["seasonality"] == "NSA"
            and row["population"] == "CPI_U"
            and row["geography"] == "US_city_average",
            "CPI estimand differs",
        )
        for field in ("release_utc", "vintage_utc"):
            require(datetime.fromisoformat(row[field].replace("Z", "+00:00")) <= cutoff, "future economic release")
        require(np.isfinite(row["value"]), "finite economic observation required")
        result[cycle] = dict(value=float(row["value"]), cutoff_utc=cutoff.isoformat(), source=row)
    require(set(result) == set(range(1998, 2026, 2)), "complete 14-cycle headline support required")
    require(result[2004]["value"] == 2.7 and result[2014]["value"] == 1.7, "cutoff sentinel changed")
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
    keep = set(np.flatnonzero(metadata.cycle.ge(1998) & metadata.cycle.lt(target)) + 1)
    return pd.read_csv(
        path, usecols=["race_id", "cycle", "actual_dem_share"], skiprows=lambda line: line > 0 and line not in keep
    )


def transformed_features(frame, macro, target):
    require(target in TARGETS, "unsupported target; current 2026 inputs are absent")
    frame = frame.loc[frame.cycle.le(target)].copy()
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
    require(earlier == list(range(1998, target, 2)), "complete strictly earlier unique-cycle support required")
    oriented = np.array([values[cycle] for cycle in earlier])
    mean, sd = float(oriented.mean()), float(oriented.std(ddof=0))
    require(np.isfinite([mean, sd]).all() and sd > 0, "nondegenerate unique-cycle scale required")
    frame[FEATURE] = frame.cycle.map({cycle: (value - mean) / sd for cycle, value in values.items()})
    scaler = dict(
        raw_units="headline_CPI_yoy_percentage_points",
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
        provider_id="headline_cpi_oriented_mean_v1",
        source_sha256=sha(Path(__file__).read_bytes()),
        role="environment",
        kind="mean",
        kernel="linear",
        parameters=(module.ParameterSpec(parameter_id="economy.headline_cpi", owner="economy", mean=0, sd=0.05),),
        terms=(
            module.MeanTerm(
                term_id="economy.headline_cpi.term",
                owner="economy",
                feature_id=FEATURE,
                coefficient=module.ParameterRef(parameter_id="economy.headline_cpi"),
            ),
        ),
        covariate_evidence_ids=("cpi:CUUR0000SA0:midnight-v2",),
    )
    return dict(
        H0=module.legacy_recipe("senate_history_refreshed_v1"),
        F0=baseline,
        E01=module.Recipe(providers=(*baseline.providers, economic)),
    )


def compile_fold(target, macro=None):
    module = factory()
    frame, scaler = transformed_features(load_features(), macro or load_macro(), target)
    outcomes = earlier_outcomes(target)
    feature_hash = module.frame_digest(frame)
    evidence = dict(
        schema="E01SourceJoinPreparationEvidenceV1",
        status="not_execution_or_count_admission",
        panel_receipt_sha256=EXPECTED["panel_receipt"],
        vintage_receipt_sha256=EXPECTED["vintage_receipt"],
        vintage_selections_sha256=EXPECTED["selections"],
        feature_rows_sha256=feature_hash,
        target_cycle=target,
        scaler=scaler,
        adapter_sha256=sha(Path(__file__).read_bytes()),
    )
    admission = module.Admission(
        cutoff_cycle=target,
        cutoff_utc=str(pd.Timestamp(frame.loc[frame.cycle.eq(target), "forecast_cutoff"].iloc[0], tz="UTC")),
        support_id="actual_senate_ordinary_DR_panel_1998plus",
        feature_source_sha256=feature_hash,
        outcome_source_sha256=EXPECTED["outcomes"],
        receipt_sha256=sha(canonical(evidence)),
        roster_qualification="qualified_retrospective_proxy_not_count_admission",
        extra_feature_ids=(FEATURE,),
        start_cycle=1998,
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
    """Prior draws and exact mean-shift checks only; no likelihood fitting."""
    training = frame.loc[frame.cycle.lt(target)].drop_duplicates("cycle").sort_values("cycle")
    target_frame = frame.loc[frame.cycle.eq(target)].copy()
    cycle_ids = training.cycle.astype(int).tolist()
    pi = np.array([macro[cycle]["value"] for cycle in cycle_ids])
    party = training.president_party_asof.to_numpy(float)
    chosen = (party * pi - scaler["mean"]) / scaler["sd"]
    unsigned_mean, unsigned_sd = float(pi.mean()), float(pi.std(ddof=0))
    target_party = float(target_frame.president_party_asof.iloc[0])
    target_value = macro[target]["value"]
    target_z = float(target_frame[FEATURE].iloc[0])
    alternate_z = target_party * (target_value - unsigned_mean) / unsigned_sd
    shifts = {}
    for label, sd, coordinate in (
        ("primary", 0.05, target_z),
        ("tighter_sensitivity", 0.025, target_z),
        ("alternative_centering_diagnostic_only", 0.05, alternate_z),
    ):
        radius = float(1.959963984540054 * sd * abs(coordinate))
        shifts[label] = dict(
            coordinate=float(coordinate),
            coefficient_sd=sd,
            mean_logit_shift=0,
            central95_logit_shift=[-radius, radius],
            central95_shifted_share_at_baseline_half=expit(np.array([-radius, radius])).tolist(),
        )
    correlations = {}
    other = dict(
        party=party,
        party_midterm=party * training.midterm.to_numpy(float),
        national=training.prior_national_presidential_logit.to_numpy(float),
    )
    for name, values in other.items():
        correlations[name] = float(np.corrcoef(chosen, values)[0, 1])
    rng = np.random.default_rng(SEEDS[target])
    draws = 4096
    parameters = {p.parameter_id: p for item in compiled.values() for p in item.recipe.parameters}
    standard = {name: rng.normal(size=draws) for name in sorted(parameters)}
    shared = rng.normal(size=draws) * abs(rng.normal(0, 0.2, size=draws))
    residual = rng.standard_t(5, size=(draws, len(target_frame))) * abs(rng.normal(0, 0.5, size=draws))[:, None]
    summaries, locations = {}, {}

    def summary(logits):
        shares = expit(logits)
        return dict(
            share_p01_p10_p50_p90_p99=np.quantile(shares, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist(),
            fraction_share_below01_or_above99=float(((shares < 0.01) | (shares > 0.99)).mean()),
            minimum_race_prior_mean=float(shares.mean(axis=0).min()),
            maximum_race_prior_mean=float(shares.mean(axis=0).max()),
        )

    for name, item in compiled.items():
        offset, design = module._design(item.recipe, target_frame, item.admission)
        beta = np.column_stack([p.mean + p.sd * standard[p.parameter_id] for p in item.recipe.parameters])
        location = offset + beta @ design.T
        locations[name] = location
        summaries[name] = summary(location + shared[:, None] + residual)
    economic_standard = standard["economy.headline_cpi"]
    summaries["E01_sd025"] = summary(
        locations["F0"] + 0.025 * economic_standard[:, None] * target_z + shared[:, None] + residual
    )
    summaries["alternative_centering_diagnostic_only"] = summary(
        locations["F0"] + 0.05 * economic_standard[:, None] * alternate_z + shared[:, None] + residual
    )
    return dict(
        role="prior_predictive_diagnostic_not_fit_or_score",
        draws=draws,
        seed=SEEDS[target],
        mean_shift_checks=shifts,
        raw_target_headline_percent=target_value,
        earlier_raw_range=[float(pi.min()), float(pi.max())],
        target_outside_earlier_raw_range=not float(pi.min()) <= target_value <= float(pi.max()),
        unsigned_cpi_mean=unsigned_mean,
        unsigned_cpi_sd=unsigned_sd,
        unique_cycle_correlations_with_chosen_coordinate=correlations,
        prior_predictive=summaries,
        comparison=(
            "Alternative coordinate is diagnostic only; its independent same-SD prior is a different joint prior"
        ),
    )


def prepare(output):
    # Validate and compile every real fold before creating any output directory.
    macro = load_macro()
    folds = {target: compile_fold(target, macro) for target in TARGETS}
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "preparation must stay in economic-experiment-v1")
    output.mkdir(parents=True, exist_ok=False)
    pins = [
        record(path)
        for path in [
            Path(__file__),
            HERE / "test_prepare.py",
            PANEL / "receipt.json",
            PANEL / "features/all-cycles.csv",
            PANEL / "labels/outcomes-SEPARATE.csv",
            VINTAGES / "receipt.json",
            VINTAGES / "selections.json",
            OWNER / "src/forecasting/aggregate_component_factory_v1.py",
            OWNER / "src/forecasting/fundamentals_v1.py",
            OWNER / "src/forecasting/baseline_v5.py",
            OWNER / "src/forecasting/polling_basis_v1.py",
            OWNER / "src/model_stack/contracts.py",
        ]
    ]
    snapshots = output / "snapshots"
    snapshots.mkdir()
    for index, item in enumerate(pins):
        path = Path(item["path"])
        destination = snapshots / f"{index:02d}-{path.name}"
        with destination.open("xb") as stream:
            stream.write(authenticated(path, item["sha256"]))
    plan = dict(
        schema="E01AggregatePreparatoryExperimentV1",
        status="prepared_not_fitted",
        fit_authorized=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        target_id="senate_aggregate_logit_empirical_mixture_v1",
        scope="ordinary_D_R_aggregate_preparation_only_not_population_cut_or_count_qualification",
        arms={
            "H0": "exact_existing_refreshed_history",
            "F0": "exact_existing_relative_context",
            "E01": "F0_plus_one_president_oriented_headline_CPI_mean",
        },
        feature_id=FEATURE,
        economic_prior=dict(family="Normal", mean=0, sd=0.05, sign_restriction=None),
        prior_sensitivity=dict(
            tighter_coefficient_sd=0.025,
            primary_coefficient_sd=0.05,
            rule="one predeclared tighter prior; no target-score selection or additional covariates",
        ),
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
            "For each fold, seal all H0/F0/E01 predictive draws before target scoring. "
            "Earlier evaluation cycles can train later folds. "
            "This preparation is not a globally unopened-label holdout."
        ),
        runtime_preparation=dict(
            python=sys.version,
            packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "pydantic")},
            fitter_runtime="Not certified here; working compatible PyMC/ArviZ runtime required before fitting",
        ),
        fitting_budget="same complete joint fit budget for all12 arm/fold combinations",
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
            "No gas/core/income/state interaction, bank splicing, post-poll shift or current2026 feature.",
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
