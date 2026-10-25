"""H14 target polls/features only. No numeric outcomes, bank export or forecasts."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import source_guard as guard

SELECTION, COMPATIBILITY, BINDINGS = guard.verify()
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
sys.path.insert(0, str(OWNER))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.forecasting import aggregate_component_factory_v1 as factory  # noqa: E402
from src.forecasting.senate_data import senate_polls_as_of  # noqa: E402

BASE = OWNER / "research/polling-layer-implementation-20261020"
PANEL = OWNER / "research/fundamentals-iteration-20261020/panel-sol"
COVERAGE = HERE.parent / "momentum-preflight-coverage-luna-v1"
ARCHIVE = OWNER / ".redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv"
F0_PLAN = HERE.parent / "economic-experiment-v1/prepared-v1/plan.json"
OPERATORS = {"native_full_series": "538_native_full_series", "latest_per_race_pollster": "538_latest_per_race_pollster"}
require = guard.require


def coverage_helpers():
    spec = importlib.util.spec_from_file_location("reviewed_target_coverage", COVERAGE / "profile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COVERAGE_HELPERS = coverage_helpers()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def rosters():
    columns = [
        "cycle",
        "race_id",
        "source_race_id",
        "eligible",
        "special",
        "election_date",
        "dem_candidate_id",
        "rep_candidate_id",
        "dem_candidate_name",
        "rep_candidate_name",
        "state",
    ]
    frame = pd.concat(
        [
            pd.read_csv(BASE / "full-chamber-intake-sol/event-roster-2018-2022.csv", usecols=columns),
            pd.read_csv(BASE / "cross-context-intake-sol/roster-2020-2024.csv", usecols=columns),
        ],
        ignore_index=True,
    )
    frame = frame[
        frame.cycle.isin(SELECTION["target_cycles"]) & frame.eligible.eq(True) & frame.special.eq(False)
    ].copy()
    for column in ("source_race_id", "dem_candidate_id", "rep_candidate_id"):
        frame[column] = frame[column].map(COVERAGE_HELPERS.ident)
    frame["election_date"] = pd.to_datetime(frame.election_date)
    require(not frame.race_id.duplicated().any(), "unique admitted event roster required")
    return frame


def raw_polls():
    frame = pd.read_csv(ARCHIVE, low_memory=False)
    frame["cycle"] = pd.to_numeric(frame.cycle, errors="coerce").astype("Int64")
    for name in ("created_at", "end_date", "start_date"):
        frame[name] = pd.to_datetime(frame[name], format="mixed", errors="coerce")
    return frame


def native_at(raw, roster, cycle, cutoff):
    source = raw[raw.cycle.eq(cycle)].copy()
    proxy = source.created_at.dt.normalize() + pd.Timedelta(days=1)
    allowed = source.end_date.le(cutoff) & source.created_at.notna() & proxy.le(cutoff)
    part, audit = senate_polls_as_of(source.loc[allowed], roster, cycle=cycle, cutoff=cutoff)
    require(not part.empty, "selected H14 source must have native observations")
    part = part.copy()
    for name in ("start_date", "end_date", "available_date"):
        part[name] = pd.to_datetime(part[name], errors="coerce")
    return part, audit, int(allowed.sum())


def select_cycle(raw, roster, cycle):
    rr = roster[roster.cycle.eq(cycle)].copy()
    election = pd.Timestamp(rr.election_date.iloc[0]).normalize()
    cutoff = election - pd.Timedelta(days=14)
    helper, audit, input_rows = native_at(raw, rr, cycle, cutoff)
    helper["end_lead"] = (election - helper.end_date).dt.days
    full = helper[helper.end_lead.between(14, 100, inclusive="both")].copy()
    full["pollster_key"] = [
        COVERAGE_HELPERS.pollster_key(p, s) for p, s in zip(full.pollster, full.sample_id, strict=True)
    ]
    latest = full.sort_values(["end_date", "available_date", "sample_id", "observation_id"]).drop_duplicates(
        ["race_id", "pollster_key"], keep="last"
    )
    latest_ids = set(latest.observation_id.astype(str))
    latest = full[full.observation_id.astype(str).isin(latest_ids)].copy()
    base, base_audit, _ = native_at(raw, rr, cycle, election - pd.Timedelta(days=1))
    base["end_lead"] = (election - base.end_date).dt.days
    base = base[base.end_lead.between(0, 100, inclusive="both")]
    expected = SELECTION["expected_observations"][str(cycle)]
    require(
        len(full) == expected["native_full_series"] and len(latest) == expected["latest_per_race_pollster"],
        "native observation count drift",
    )
    require(full.race_id.nunique() == latest.race_id.nunique() == expected["polled_races"], "polled-race mask drift")
    audit = {
        **audit,
        "cycle": cycle,
        "horizon_days": 14,
        "extra_lag_days": 0,
        "cutoff_utc": cutoff.strftime("%Y-%m-%dT00:00:00Z"),
        "input_rows_after_proxy_filter": input_rows,
        "availability_proxy": "created_at_date_plus_1_day_then_extra_lag",
        "strict_asof_qualified": False,
    }
    return rr, cutoff, helper, full, latest, base, audit, base_audit


def selection_ledger(helper, full, latest, base, cutoff, election):
    full_ids = set(full.observation_id)
    latest_ids = set(latest.observation_id)
    rows = []
    for item in helper.itertuples():
        rows.append(
            dict(
                observation_id=item.observation_id,
                sample_id=item.sample_id,
                poll_id=item.poll_id,
                question_id=item.question_id,
                race_id=item.race_id,
                end_lead_days=int(item.end_lead),
                selected_native_full=item.observation_id in full_ids,
                selected_latest=item.observation_id in latest_ids,
                disposition="outside_100_day_maximum"
                if item.observation_id not in full_ids
                else "selected_latest_for_race_pollster"
                if item.observation_id in latest_ids
                else "selected_full_older_same_race_pollster",
            )
        )
    exclusions = []
    for item in full.itertuples():
        if item.observation_id not in latest_ids:
            exclusions.append(
                dict(
                    operator="538_latest_per_race_pollster",
                    observation_id=item.observation_id,
                    sample_id=item.sample_id,
                    poll_id=item.poll_id,
                    question_id=item.question_id,
                    race_id=item.race_id,
                    exclusion_reason="older_observation_for_same_race_pollster",
                )
            )
    for item in base.itertuples():
        if item.observation_id in full_ids:
            continue
        if (election - item.end_date).days > 100:
            why = "outside_100_day_maximum"
        elif item.end_date > cutoff:
            why = "field_end_after_requested_horizon"
        elif item.available_date > cutoff:
            why = "native_availability_proxy_plus_extra_lag_after_horizon"
        elif item.sample_id in set(full.sample_id):
            why = "alternate_question_selected_after_cutoff_refilter"
        else:
            why = "not_selected_by_horizon_helper_or_question_gate"
        exclusions.append(
            dict(
                operator="538_native_full_series",
                observation_id=item.observation_id,
                sample_id=item.sample_id,
                poll_id=item.poll_id,
                question_id=item.question_id,
                race_id=item.race_id,
                exclusion_reason=why,
            )
        )
    return rows, exclusions


def reconcile(cycle, full, latest, exclusions):
    old = pd.read_csv(COVERAGE / "operator-selection-rows.csv", low_memory=False)
    old = old[
        old.source_panel.eq("538_target_archive")
        & old.cycle.eq(cycle)
        & old.horizon_days.eq(14)
        & old.extra_lag_days_beyond_native_proxy.eq(0)
    ]
    for name, part in (("native_full_series", full), ("latest_per_race_pollster", latest)):
        selected = old[old.operator.eq(OPERATORS[name]) & old.selected.astype(str).str.casefold().eq("true")]
        require(selected.observation_id.tolist() == part.observation_id.tolist(), "ordered preflight ID drift: " + name)
        require(
            selected.question_id.map(COVERAGE_HELPERS.ident).tolist()
            == part.question_id.map(COVERAGE_HELPERS.ident).tolist(),
            "preflight question identity drift",
        )
    old = pd.read_csv(COVERAGE / "selection-exclusions.csv", low_memory=False)
    old = old[
        old.source_panel.eq("538_target_archive")
        & old.cycle.eq(cycle)
        & old.horizon_days.eq(14)
        & old.extra_lag_days_beyond_native_proxy.eq(0)
    ]

    def key(row):
        return row["operator"], row["observation_id"], row["exclusion_reason"]

    require(
        sorted(key(r) for r in exclusions) == sorted(key(r) for r in old.to_dict("records")),
        "preflight exclusion ledger drift",
    )
    return dict(
        exact_ordered_selected_IDs_and_questions=True,
        exact_helper_output_exclusions=True,
        preflight_selected_rows=guard.pin(COVERAGE / "operator-selection-rows.csv"),
        preflight_exclusions=guard.pin(COVERAGE / "selection-exclusions.csv"),
    )


def target_features(cycle):
    plan = json.loads(F0_PLAN.read_bytes())
    fold = next(f for f in plan["folds"] if f["target_cycle"] == cycle)
    path = PANEL / "features/all-cycles.csv"
    metadata = pd.read_csv(path, usecols=["cycle"])
    keep = set(metadata.index[metadata.cycle.eq(cycle)] + 1)
    target = pd.read_csv(path, skiprows=lambda n: n > 0 and n not in keep)
    frozen_meta = pd.read_csv(fold["features"]["path"], usecols=["cycle"])
    frozen_keep = set(frozen_meta.index[frozen_meta.cycle.eq(cycle)] + 1)
    frozen = pd.read_csv(fold["features"]["path"], skiprows=lambda n: n > 0 and n not in frozen_keep)
    require(
        target.race_id.tolist() == fold["target_race_ids"] == frozen.race_id.tolist(), "F0 target event order drift"
    )
    # CPI is not a F0 term. Retain its original admission metadata solely to
    # reproduce the exact reviewed target-design validation contract.
    feature = "president_oriented_headline_cpi_yoy_z"
    target[feature] = frozen[feature].to_numpy()
    evidence = json.loads(guard.checked(fold["join_evidence"]))
    admission = factory.Admission(
        cutoff_cycle=cycle,
        cutoff_utc=str(pd.Timestamp(target.forecast_cutoff.iloc[0], tz="UTC")),
        support_id="actual_senate_ordinary_DR_panel_1998plus",
        feature_source_sha256=fold["arms"]["F0"]["feature_rows_sha256"],
        outcome_source_sha256="5f5cafa1afc53387331d4e75b3c65f92ed4cab3a91578dcb7cb0815357b029ad",
        receipt_sha256=digest(canonical(evidence)),
        roster_qualification="qualified_retrospective_proxy_not_count_admission",
        extra_feature_ids=(feature,),
        start_cycle=1998,
    )
    recipe = factory.Recipe.model_validate(fold["arms"]["F0"]["recipe"])
    offset, design = factory._design(recipe, target, admission)
    bank = next(b for b in COMPATIBILITY["banks"] if b["year"] == cycle and b["arm"] == "F0")
    require(
        digest(offset.tobytes()) == bank["target_offset_sha256"]
        and digest(design.tobytes()) == bank["target_design_sha256"],
        "exact accepted F0 design/offset drift",
    )
    require(
        target.race_id.tolist() == bank["target_race_ids"]
        and bank["coefficient_names"] == [p.parameter_id for p in recipe.parameters],
        "F0 bank/event/coefficient identity drift",
    )
    require(
        bank["joint_shape"] == [4, 4000, 6] and len(bank["raw_banks"]) == 4,
        "complete 16000-row bank references required",
    )
    return target, offset, design, admission, recipe, bank


def outcome_identity_metadata(cycle):
    # Explicit projection: no outcome share, vote count, winner or score is parsed.
    path = PANEL / "labels/outcomes-SEPARATE.csv"
    metadata = pd.read_csv(path, usecols=["cycle", "race_id"])
    return metadata[metadata.cycle.eq(cycle)].copy(), guard.pin(path)


def raw_provenance(cycle, full, cutoff):
    body = ARCHIVE.read_bytes()
    lines = body.splitlines(keepends=True)
    reader = csv.DictReader(io.StringIO(body.decode(), newline=""))
    _ = reader.fieldnames
    previous = reader.line_num
    rows = []
    selected = {(str(r.poll_id), str(r.question_id)) for r in full.itertuples()}
    eligible_sources = set(rosters().loc[lambda f: f.cycle.eq(cycle), "source_race_id"])
    for number, row in enumerate(reader, 2):
        end = reader.line_num
        start = previous + 1
        previous = end
        if int(row["cycle"]) != cycle:
            continue
        key = (row["poll_id"], row["question_id"])
        end_date = pd.to_datetime(row["end_date"], format="mixed", errors="coerce")
        created = pd.to_datetime(row["created_at"], format="mixed", errors="coerce")
        proxy = pd.NaT if pd.isna(created) else created.normalize() + pd.Timedelta(days=1)
        rows.append(
            dict(
                csv_record_number=number,
                physical_start_line=start,
                physical_end_line=end,
                raw_record_sha256=digest(b"".join(lines[start - 1 : end])),
                poll_id=row["poll_id"],
                question_id=row["question_id"],
                source_race_id=row["race_id"],
                candidate_id=row["candidate_id"],
                candidate_name=row["candidate_name"],
                party=row["party"],
                pct_raw=row["pct"],
                population_raw=row["population"],
                start_date_raw=row["start_date"],
                end_date_raw=row["end_date"],
                created_at_raw=row["created_at"],
                source_url=row["url"],
                stage=row["stage"],
                field_end_before_cutoff=bool(pd.notna(end_date) and end_date <= cutoff),
                native_availability_before_cutoff=bool(pd.notna(proxy) and proxy <= cutoff),
                in_current_admitted_event_roster=row["race_id"] in eligible_sources,
                selected_question_in_native_full=key in selected,
                qualification=(
                    "raw candidate answer; not an independent poll; "
                    "unresolved pre-helper reasons remain aggregate audit"
                ),
            )
        )
    return rows


def json_poll_input(part, bank_ids):
    require(
        not part.observation_id.duplicated().any() and set(part.race_id) <= set(bank_ids),
        "unique known observation events required",
    )
    q = part.dem_share.to_numpy(float)
    days = part.end_lead.to_numpy(float)
    require(
        np.isfinite(q).all()
        and ((q > 0) & (q < 1)).all()
        and np.isfinite(days).all()
        and ((days >= 14) & (days <= 100)).all(),
        "finite interior H14 poll input required",
    )
    require(
        np.array_equal(q, part.dem_pct.to_numpy() / (part.dem_pct.to_numpy() + part.rep_pct.to_numpy())),
        "native conditional share drift",
    )
    return dict(
        schema="MomentumH14JointPollInputV1",
        observation_ids=part.observation_id.tolist(),
        sample_ids=part.sample_id.tolist(),
        race_ids=part.race_id.tolist(),
        cycles=part.cycle.astype(str).tolist(),
        pollster_keys=part.pollster_key.tolist(),
        days_to_election=days.tolist(),
        shares=q.tolist(),
        logits=(np.log(q) - np.log1p(-q)).tolist(),
        race_indices=[bank_ids.index(r) for r in part.race_id],
        bank_race_ids=bank_ids,
        interval_averaging=False,
        nominal_variance_not_added=True,
        strict_asof_qualified=False,
    )


@dataclass(frozen=True)
class FrozenPollInput:
    observation_ids: tuple
    sample_ids: tuple
    race_ids: tuple
    cycles: tuple
    pollster_keys: tuple
    days_to_election: np.ndarray
    shares: np.ndarray
    logits: np.ndarray
    race_indices: np.ndarray


def load_prepared(plan_path, expected_sha, cycle, operator):
    guard.verify()
    plan = json.loads(guard.checked(dict(path=str(plan_path), sha256=expected_sha)))
    require(
        plan["schema"] == "MomentumH14TargetPreparationPlanV1"
        and plan["selection_sha256"] == guard.SELECTION_SHA
        and operator in OPERATORS,
        "selected target contract required",
    )
    fold = next(f for f in plan["folds"] if f["cycle"] == cycle)
    row = fold["operators"][operator]
    data = json.loads(guard.checked(row["joint_input"]))
    require(data["schema"] == "MomentumH14JointPollInputV1", "selected joint input schema required")
    require(len(set(data["observation_ids"])) == len(data["observation_ids"]), "duplicate target observation IDs")
    n = len(data["observation_ids"])
    require(
        all(
            len(data[key]) == n and all(isinstance(v, str) and v.strip() for v in data[key])
            for key in ("observation_ids", "sample_ids", "race_ids", "cycles", "pollster_keys")
        ),
        "complete ordered named target coordinates required",
    )
    require(
        n == SELECTION["expected_observations"][str(cycle)][operator]
        and set(data["race_ids"]) <= set(fold["bank_race_ids"]),
        "target observation/event support mismatch",
    )
    require(set(data["cycles"]) == {str(cycle)}, "target cycle mismatch")

    def frozen(values):
        values = np.asarray(values, float)
        require(values.shape == (n,) and np.isfinite(values).all(), "finite complete target arrays required")
        return np.frombuffer(values.tobytes(), dtype=float)

    days, shares = frozen(data["days_to_election"]), frozen(data["shares"])
    logits = frozen(data["logits"])
    indices = frozen(data["race_indices"])
    require(
        ((days >= 14) & (days <= 100)).all() and ((shares > 0) & (shares < 1)).all(),
        "H14 ages/interior shares required",
    )
    require(
        data["bank_race_ids"] == fold["bank_race_ids"]
        and np.array_equal(logits, np.log(shares) - np.log1p(-shares))
        and np.array_equal(indices, [fold["bank_race_ids"].index(rid) for rid in data["race_ids"]]),
        "target logits/race-axis identity mismatch",
    )
    for key in ("target_design", "target_features", "outcome_identity_metadata", "bank_reference"):
        guard.checked(fold[key])
    design = json.loads(guard.checked(fold["target_design"]))
    bank = json.loads(guard.checked(fold["bank_reference"]))
    require(
        design["race_ids"] == fold["bank_race_ids"] == bank["target_race_ids"]
        and bank["year"] == cycle
        and bank["arm"] == "F0",
        "target bank/event binding mismatch",
    )
    require(
        digest(np.asarray(design["offset"], float).tobytes()) == bank["target_offset_sha256"]
        and digest(np.asarray(design["design"], float).tobytes()) == bank["target_design_sha256"],
        "accepted target-design hash mismatch",
    )
    return (
        plan,
        fold,
        FrozenPollInput(
            tuple(data["observation_ids"]),
            tuple(data["sample_ids"]),
            tuple(data["race_ids"]),
            tuple(data["cycles"]),
            tuple(data["pollster_keys"]),
            days,
            shares,
            logits,
            indices,
        ),
    )


def prepare(output):
    guard.verify()
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "exclusive target package output required")
    output.mkdir(parents=True, exist_ok=False)
    folds = []
    try:
        raw, roster = raw_polls(), rosters()
        for cycle in SELECTION["target_cycles"]:
            folder = output / str(cycle)
            folder.mkdir()
            rr, cutoff, helper, full, latest, base, audit, base_audit = select_cycle(raw, roster, cycle)
            ledgers, exclusions = selection_ledger(helper, full, latest, base, cutoff, rr.election_date.iloc[0])
            reconciliation = reconcile(cycle, full, latest, exclusions)
            target, offset, design, admission, recipe, bank = target_features(cycle)
            bank_ids = bank["target_race_ids"]
            require(pd.Timestamp(bank["target_cutoff_utc"]) == cutoff, "poll/F0 H14 cutoff mismatch")
            labels, label_pin = outcome_identity_metadata(cycle)
            require(
                not labels.race_id.duplicated().any() and set(labels.race_id) == set(bank_ids),
                "full-bank outcome identity metadata mismatch",
            )
            labels.to_csv(folder / "outcome-identity-METADATA-ONLY.csv", index=False)
            target.to_csv(folder / "target-features-LABEL-FREE.csv", index=False)
            write(
                folder / "target-design.json",
                dict(
                    schema="F0H14TargetDesignV1",
                    race_ids=bank_ids,
                    event_ids=target.event_id.tolist(),
                    coefficient_names=bank["coefficient_names"],
                    offset=offset.tolist(),
                    design=design.tolist(),
                    offset_array_sha256=digest(offset.tobytes()),
                    design_array_sha256=digest(design.tobytes()),
                    recipe=recipe.model_dump(mode="json"),
                    admission=admission.model_dump(mode="json"),
                    law_hash=bank["law_hash"],
                ),
            )
            write(
                folder / "bank-reference.json",
                dict(
                    schema="CompleteReviewedF0BankReferenceV1",
                    **bank,
                    row_count=16000,
                    original_chain_draw_rows_preserved=True,
                    bank_exported=False,
                    historical_cycle_z_not_target_effect=True,
                    future_F_only_replicates_not_reused=True,
                ),
            )
            pd.DataFrame(ledgers).to_csv(folder / "helper-output-selection-ledger.csv", index=False)
            pd.DataFrame(exclusions).to_csv(folder / "helper-output-exclusions.csv", index=False)
            pd.DataFrame(raw_provenance(cycle, full, cutoff)).to_csv(folder / "raw-answer-provenance.csv", index=False)
            write(folder / "helper-audits.json", dict(H14=audit, final_pre_election=base_audit))
            polled = [rid for rid in bank_ids if rid in set(full.race_id)]
            require(set(polled) == set(latest.race_id), "operator common-polled mask differs")
            mask = [
                dict(
                    race_id=rid,
                    event_id=target.set_index("race_id").loc[rid, "event_id"],
                    in_full_F0_bank=True,
                    in_current_poll_roster=rid in set(rr.race_id),
                    in_common_polled_mask=rid in set(polled),
                    outcome_identity_metadata_present=rid in set(labels.race_id),
                    disposition="common_polled"
                    if rid in set(polled)
                    else "unpolled_admitted_roster"
                    if rid in set(rr.race_id)
                    else "bank_only_no_new_event_admission",
                )
                for rid in bank_ids
            ]
            pd.DataFrame(mask).to_csv(folder / "full-bank-and-common-polled-masks.csv", index=False)
            operators = {}
            for name, part in (("native_full_series", full), ("latest_per_race_pollster", latest)):
                joint = folder / (name + "-joint-input.json")
                write(joint, json_poll_input(part, bank_ids))
                part.to_csv(folder / (name + "-observations.csv"), index=False)
                operators[name] = dict(
                    rows=len(part),
                    polled_races=part.race_id.nunique(),
                    population_counts={str(k): int(v) for k, v in part.population.value_counts().items()},
                    joint_input=guard.pin(joint),
                    observations=guard.pin(folder / (name + "-observations.csv")),
                )
            folds.append(
                dict(
                    cycle=cycle,
                    cutoff_utc=cutoff.strftime("%Y-%m-%dT00:00:00Z"),
                    horizon_days=14,
                    additional_lag_days=0,
                    bank_race_ids=bank_ids,
                    poll_roster_race_ids=rr.race_id.tolist(),
                    common_polled_race_ids=polled,
                    operators=operators,
                    target_design=guard.pin(folder / "target-design.json"),
                    target_features=guard.pin(folder / "target-features-LABEL-FREE.csv"),
                    bank_reference=guard.pin(folder / "bank-reference.json"),
                    outcome_identity_metadata=guard.pin(folder / "outcome-identity-METADATA-ONLY.csv"),
                    label_source_metadata_projection=dict(
                        source=label_pin, columns=["cycle", "race_id"], numeric_label_columns_read=False
                    ),
                    full_and_polled_mask=guard.pin(folder / "full-bank-and-common-polled-masks.csv"),
                    coverage_reconciliation=reconciliation,
                    raw_provenance=guard.pin(folder / "raw-answer-provenance.csv"),
                    selection_ledger=guard.pin(folder / "helper-output-selection-ledger.csv"),
                    exclusion_ledger=guard.pin(folder / "helper-output-exclusions.csv"),
                    helper_audits=guard.pin(folder / "helper-audits.json"),
                )
            )
        write(
            output / "plan.json",
            dict(
                schema="MomentumH14TargetPreparationPlanV1",
                selection_sha256=guard.SELECTION_SHA,
                source_bindings=guard.pin(HERE / "source-bindings.json"),
                folds=folds,
                numeric_target_labels_parsed=False,
                original_bank_rows_exported=False,
                proposed_event_expansions_admitted=False,
                CA2022_new_poll_event_unresolved=True,
                strict_asof_qualified=False,
                optimized=False,
                forecast_sampled=False,
                forecast_scored=False,
                promotion_authorized=False,
            ),
        )
        guard.verify()
        print(json.dumps(guard.pin(output / "plan.json")))
    except BaseException as error:
        write(
            output / "failure.json",
            dict(
                error=repr(error),
                traceback=traceback.format_exc(),
                silent_rebinding=False,
                owner_changed=False,
                numeric_target_labels_parsed=False,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared-v1")
    prepare(parser.parse_args().output)
