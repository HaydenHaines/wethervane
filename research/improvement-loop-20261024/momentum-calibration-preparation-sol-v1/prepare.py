"""Earlier-only residuals and supplied five/six-scale objectives; never fit."""

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import guard

SELECTION, BINDINGS = guard.verify()
sys.dont_write_bytecode = True
PROTOTYPE = Path(SELECTION["references"][1]["path"]).parent
sys.path.insert(0, str(PROTOTYPE))

import momentum as math_model  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logit  # noqa: E402

HERE = Path(__file__).resolve().parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
POLL = OWNER / "research/polling-layer-implementation-20261020"
HIST = POLL / "historical-intake-sol"
COVERAGE = HERE.parent / "momentum-preflight-coverage-luna-v1"
ALLOWED = (2012, 2016, 2018, 2020)
require = guard.require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=guard.sha(path), bytes=Path(path).stat().st_size)


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def read_earlier_csv(path, target):
    """Read cycle metadata first, then exclude future numeric rows at CSV parsing."""
    require(target in (2018, 2020, 2022, 2024), "selected target cycle required")
    metadata = pd.read_csv(path, usecols=["cycle"])
    require(metadata.cycle.notna().all() and set(metadata.cycle) <= set(ALLOWED), "unsupported source cycle")
    indices = set(metadata.index[metadata.cycle.lt(target)] + 1)
    result = pd.read_csv(path, skiprows=lambda number: number > 0 and number not in indices, low_memory=False)
    require(not result.empty and result.cycle.lt(target).all(), "earlier numeric data only")
    return result


def selector_ast(path):
    module = ast.parse(Path(path).read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "calibrate")
    start = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "selected" for t in n.targets)
    )
    statements = function.body[start : start + 5]
    require(
        len(statements) == 5 and all(isinstance(n, ast.Assign) for n in statements),
        "exact native selector shape changed",
    )
    return statements


def select_native(polls, outcomes):
    live = selector_ast(POLL / "standard_runner.py")
    frozen = selector_ast(POLL / "forecast-v2/source-snapshots/00-standard_runner.py")
    require(
        [ast.dump(n, include_attributes=False) for n in live]
        == [ast.dump(n, include_attributes=False) for n in frozen],
        "historical selector differs from selected snapshot",
    )
    plan = json.loads((POLL / "standard-plan-root-v1.json").read_bytes())
    require(plan["selection"]["maximum_days_to_election"] == 100, "exact original age bound required")
    expected = {k: SELECTION["regularizer"]["medians"][k] for k in math_model.INCUMBENT.COMPONENTS}
    require(
        plan["covariance_model"]["regularizer_medians"] == expected
        and plan["covariance_model"]["regularizer_log_sd"] == 0.7,
        "selected prior changed",
    )
    scope = {"polls": polls, "outcomes": outcomes, "plan": plan}
    # Execute only the five byte-authenticated selector assignments, never the
    # runner's calibrate function, optimizer, sampler or current forecast path.
    exec(compile(ast.Module(body=live, type_ignores=[]), "<pinned-native-historical-selector>", "exec"), scope)
    selected = scope["selected"]
    require(not selected.observation_id.duplicated().any(), "unique selected observations required")
    require(selected.actual_dem_share.notna().all(), "complete earlier truth required")
    require(
        (
            (selected.dem_share > 0)
            & (selected.dem_share < 1)
            & (selected.actual_dem_share > 0)
            & (selected.actual_dem_share < 1)
        ).all(),
        "interior D/R shares required",
    )
    return selected


def fold_inputs(target):
    polls = read_earlier_csv(HIST / "derived/historical_poll_panel.csv", target)
    outcomes = read_earlier_csv(HIST / "derived/outcomes_allowed_cycles_SEPARATE.csv", target)
    expected = next(f for f in SELECTION["folds"] if f["target"] == target)
    require(
        sorted(set(polls.cycle)) == expected["training_cycles"]
        and sorted(set(outcomes.cycle)) == expected["training_cycles"],
        "complete earlier cycles required",
    )
    require(not outcomes.race_id.duplicated().any(), "unique earlier outcome identities required")
    selected = select_native(polls, outcomes)
    require(
        len(selected) == expected["expected_rows"] and selected.cycle.lt(target).all(),
        "selected earlier rows/count mismatch",
    )
    q = selected.dem_share.to_numpy()
    truth = selected.actual_dem_share.to_numpy()
    residual = logit(q) - logit(truth)
    require(np.isfinite(residual).all(), "finite normalized logit residuals required")
    # Check source normalization; retain the exact CSV/pandas/scipy arithmetic
    # of the incumbent instead of silently re-rounding its admitted shares.
    require(
        np.allclose(
            q,
            selected.source_democratic_value / (selected.source_democratic_value + selected.source_republican_value),
            atol=1e-15,
            rtol=0,
        ),
        "source D/R normalization mismatch",
    )
    require(
        np.allclose(
            outcomes.actual_dem_share,
            outcomes.dem_votes / (outcomes.dem_votes + outcomes.rep_votes),
            atol=1e-15,
            rtol=0,
        ),
        "admitted outcome D/R normalization mismatch",
    )
    coordinates = math_model.PollCoordinatesV1(
        tuple(selected.observation_id),
        tuple(selected.race_id),
        tuple(selected.cycle.astype(str)),
        tuple(selected.pollster_key),
        selected.days_to_election.to_numpy(),
    )
    blocks = []
    for cycle in expected["training_cycles"]:
        indices = np.flatnonzero(selected.cycle.to_numpy() == cycle)
        frame = selected.iloc[indices]
        coords = math_model.PollCoordinatesV1(
            tuple(frame.observation_id),
            tuple(frame.race_id),
            tuple(frame.cycle.astype(str)),
            tuple(frame.pollster_key),
            frame.days_to_election.to_numpy(),
        )
        blocks.append(math_model.ResidualBlockV1(coords, residual[indices]))
    return polls, outcomes, selected, coordinates, tuple(blocks)


def source_masks(target, polls, outcomes, selected):
    source = []
    selected_ids = set(selected.observation_id)
    winners = {(r.cycle, r.race_id, r.pollster_key, int(r.lead_bin)): r.observation_id for r in selected.itertuples()}
    earlier = polls.set_index("observation_id")
    with (HIST / "derived/historical_poll_panel.csv").open() as stream:
        for record, row in enumerate(csv.DictReader(stream), 2):
            cycle = int(row["cycle"])
            is_earlier = cycle < target
            oid = row["observation_id"]
            winner = None
            in_age = None
            if is_earlier:
                value = earlier.loc[oid]
                age = float(value.days_to_election)
                in_age = 0 <= age <= 100
                if in_age:
                    winner = winners[(cycle, row["race_id"], str(value.pollster).strip().casefold(), int(age // 14))]
            disposition = (
                "target_or_future_cycle_excluded_before_numeric_parse"
                if not is_earlier
                else "outside_inclusive_0_100_days"
                if not in_age
                else "selected_latest_in_fixed_14day_bin"
                if oid in selected_ids
                else "superseded_in_fixed_14day_bin"
            )
            source.append(
                dict(
                    target=target,
                    source_csv_record=record,
                    observation_id=oid,
                    source_row_number=row["source_row_number"],
                    source_cycle=cycle,
                    race_id=row["race_id"],
                    days_to_election_raw=row["days_to_election"],
                    end_date_raw=row["end_date"],
                    pollster_raw=row["pollster"],
                    availability_basis=row["availability_basis"],
                    publication_date_verified=row["publication_date_verified"],
                    earlier_cycle=is_earlier,
                    within_0_100=in_age,
                    selected=oid in selected_ids,
                    winner_observation_id=winner,
                    disposition=disposition,
                )
            )
    labels = []
    with (HIST / "derived/outcomes_allowed_cycles_SEPARATE.csv").open() as stream:
        for record, row in enumerate(csv.DictReader(stream), 2):
            earlier_label = int(row["cycle"]) < target
            labels.append(
                dict(
                    target=target,
                    source_csv_record=record,
                    race_id=row["race_id"],
                    source_cycle=int(row["cycle"]),
                    earlier_label=earlier_label,
                    used_by_selected_poll=row["race_id"] in set(selected.race_id),
                    disposition="admitted_earlier_outcome"
                    if earlier_label
                    else "target_or_future_label_not_parsed_numeric",
                )
            )
    return source, labels


def reconcile_coverage(target, polls, selected):
    ledger = pd.read_csv(COVERAGE / "historical-calibration-baseline-id-ledger-v2.csv")
    ledger = ledger[ledger.target_cycle.eq(target)]
    require(
        set(ledger.observation_id) == set(polls.observation_id) and not ledger.observation_id.duplicated().any(),
        "corrected coverage earlier-source ledger mismatch",
    )
    retained = ledger[ledger.selected_by_exact_fixed_bin_operator.eq(True)]
    require(set(retained.observation_id) == set(selected.observation_id), "corrected coverage selected ID mismatch")
    return dict(
        corrected_earlier_source_rows=len(ledger),
        selected_rows=len(retained),
        all_ids_agree=True,
        authoritative_ledger=pin(COVERAGE / "historical-calibration-baseline-id-ledger-v2.csv"),
    )


def block_payload(block):
    c = block.coordinates
    return dict(
        observation_ids=list(c.observation_ids),
        race_ids=list(c.race_ids),
        cycles=list(c.cycles),
        pollsters=list(c.pollsters),
        days_to_election=c.days_to_election.tolist(),
        residual=block.residual.tolist(),
    )


def input_identity(selected, blocks):
    value = dict(
        schema="MomentumActualEarlierCalibrationInputV1",
        training_cycles=sorted(set(selected.cycle)),
        blocks=[block_payload(b) for b in blocks],
        regularizer=SELECTION["regularizer"],
        source_bindings_sha256=guard.BINDINGS_SHA,
        mathematics_manifest_sha256=SELECTION["references"][1]["sha256"],
        selector_ast_sha256=digest(
            canonical([ast.dump(n, include_attributes=False) for n in selector_ast(POLL / "standard_runner.py")])
        ),
    )
    return digest(canonical(value)), value


def evaluate_points(blocks):
    old = math_model.INCUMBENT
    medians = np.array([SELECTION["regularizer"]["medians"][k] for k in math_model.COMPONENTS])
    sd = 0.7
    points = []
    for offset in (np.zeros(6), np.array([0.08, -0.06, 0.04, -0.05, 0.07, -0.03])):
        log_scales = np.log(medians) + offset
        five_blocks = [(b.residual, b.coordinates.kernels()[:5]) for b in blocks]
        five = old.gaussian_error_objective(log_scales[:5], five_blocks, prior_medians=medians[:5], prior_log_sd=sd)
        six = math_model.gaussian_error_objective(log_scales, blocks, prior_medians=medians, prior_log_sd=sd)
        points.append(
            dict(
                log_scales=log_scales.tolist(),
                five=dict(value=five[0], gradient=five[1].tolist()),
                six=dict(value=six[0], gradient=six[1].tolist()),
                supplied_only=True,
            )
        )
    return points


def prepare(output):
    guard.verify()
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "preparation must remain in new package")
    output.mkdir(parents=True, exist_ok=False)
    (output / "blocks").mkdir()
    folds = []
    identities = {}
    cycle_pins = {}
    for expected in SELECTION["folds"]:
        target = expected["target"]
        polls, outcomes, selected, coordinates, blocks = fold_inputs(target)
        identity, payload = input_identity(selected, blocks)
        folder = output / str(target)
        folder.mkdir()
        selected.to_csv(folder / "selected-training-polls.csv", index=False)
        outcomes.to_csv(folder / "earlier-outcomes.csv", index=False)
        masks, labels = source_masks(target, polls, outcomes, selected)
        pd.DataFrame(masks).to_csv(folder / "poll-selection-mask.csv", index=False)
        pd.DataFrame(labels).to_csv(folder / "label-mask.csv", index=False)
        coverage = reconcile_coverage(target, polls, selected)
        pins = []
        for block in blocks:
            cycle = block.coordinates.cycles[0]
            path = output / "blocks" / (cycle + ".json")
            if cycle not in cycle_pins:
                write(path, dict(schema="MomentumCalibrationCycleBlockV1", **block_payload(block)))
                cycle_pins[cycle] = pin(path)
            else:
                require(
                    json.loads(guard.checked(cycle_pins[cycle]))
                    == dict(schema="MomentumCalibrationCycleBlockV1", **block_payload(block)),
                    "shared earlier cycle block changed across target folds",
                )
            pins.append(cycle_pins[cycle])
        if identity not in identities:
            points = evaluate_points(blocks)
            write(folder / "supplied-objective-points.json", points)
            identities[identity] = dict(first_target=target, points=pin(folder / "supplied-objective-points.json"))
        old = read_earlier_csv(POLL / "calibration-v1/selected-training-polls.csv", target)
        require(
            old.observation_id.tolist() == selected.observation_id.tolist(),
            "original selected-MAP ordered IDs mismatch",
        )
        require(
            np.allclose(old.dem_share, selected.dem_share, atol=1e-15, rtol=0)
            and np.allclose(old.actual_dem_share, selected.actual_dem_share, atol=1e-15, rtol=0),
            "original admitted shares mismatch",
        )
        tx = int((selected.race_id.eq("2020:TX:senate:regular")).sum())
        require(tx == (31 if target > 2020 else 0), "selected-MAP 2020 Texas rows must remain")
        write(folder / "calibration-input.json", payload)
        folds.append(
            dict(
                target=target,
                training_cycles=expected["training_cycles"],
                rows=len(selected),
                per_cycle_rows={str(k): int(v) for k, v in selected.groupby("cycle").size().items()},
                input_identity_sha256=identity,
                input_payload=pin(folder / "calibration-input.json"),
                blocks=pins,
                selected=pin(folder / "selected-training-polls.csv"),
                earlier_outcomes=pin(folder / "earlier-outcomes.csv"),
                poll_mask=pin(folder / "poll-selection-mask.csv"),
                label_mask=pin(folder / "label-mask.csv"),
                coverage_reconciliation=coverage,
                original_MAP_ordered_ID_match=True,
                texas_2020_rows=tx,
                objective_points=identities[identity]["points"],
            )
        )
    require(
        folds[2]["input_identity_sha256"] == folds[3]["input_identity_sha256"],
        "2022/2024 numerical input identity must be proven",
    )
    write(
        output / "plan.json",
        dict(
            schema="MomentumActualCalibrationPreparationPlanV1",
            selection_sha256=guard.ROOT_SHA,
            source_bindings=pin(HERE / "source-bindings.json"),
            folds=folds,
            unique_numerical_calibrations=len(identities),
            shared_2022_2024_input_identity=folds[2]["input_identity_sha256"],
            identical_training_allows_later_fit_reuse=True,
            arms=SELECTION["arms"],
            regularizer=SELECTION["regularizer"],
            shared_outcome_dependence=(
                "Conditional empirical Bayes: calibration errors and fundamentals share "
                "historical labels; dependence not integrated"
            ),
            source_vintage=(
                "Retrospective later-corrected author/archive/official-label exports; field-end availability proxy only"
            ),
            strict_asof_qualified=False,
            optimized=False,
            sampled=False,
            forecast_scored=False,
            promotion_authorized=False,
        ),
    )
    guard.verify()
    print(json.dumps(pin(output / "plan.json")))


def load_prepared(plan_path, expected_sha, target):
    """Authenticate numerical inputs and rebuild immutable versioned blocks."""
    guard.verify()
    plan = json.loads(guard.checked(dict(path=str(plan_path), sha256=expected_sha)))
    require(
        plan["schema"] == "MomentumActualCalibrationPreparationPlanV1" and plan["selection_sha256"] == guard.ROOT_SHA,
        "wrong preparation plan",
    )
    guard.checked(plan["source_bindings"])
    require(plan["regularizer"] == SELECTION["regularizer"], "wrong selected regularizer")
    fold = next((f for f in plan["folds"] if f["target"] == target), None)
    require(fold is not None, "selected fold required")
    selected = next(f for f in SELECTION["folds"] if f["target"] == target)
    require(
        fold["training_cycles"] == selected["training_cycles"] and fold["rows"] == selected["expected_rows"],
        "wrong earlier-cycle support",
    )
    for key in ("input_payload", "selected", "earlier_outcomes", "poll_mask", "label_mask", "objective_points"):
        guard.checked(fold[key])
    payload = json.loads(guard.checked(fold["input_payload"]))
    require(
        digest(canonical(payload)) == fold["input_identity_sha256"]
        and payload["training_cycles"] == selected["training_cycles"],
        "prepared numerical identity changed",
    )
    require(
        payload["regularizer"] == SELECTION["regularizer"]
        and payload["source_bindings_sha256"] == guard.BINDINGS_SHA
        and payload["mathematics_manifest_sha256"] == SELECTION["references"][1]["sha256"],
        "wrong objective/source bindings",
    )
    blocks = []
    for item in fold["blocks"]:
        data = json.loads(guard.checked(item))
        require(
            set(map(int, data["cycles"])) <= set(selected["training_cycles"])
            and all(int(c) < target for c in data["cycles"]),
            "target/future numerical block rejected",
        )
        coordinates = math_model.PollCoordinatesV1(
            tuple(data["observation_ids"]),
            tuple(data["race_ids"]),
            tuple(data["cycles"]),
            tuple(data["pollsters"]),
            data["days_to_election"],
        )
        blocks.append(math_model.ResidualBlockV1(coordinates, data["residual"]))
    require([block_payload(b) for b in blocks] == payload["blocks"], "cycle block and numerical payload disagree")
    require(sum(len(b.residual) for b in blocks) == fold["rows"], "prepared numerical rows incomplete")
    require(
        [sorted(set(map(int, b.coordinates.cycles))) for b in blocks]
        == [[cycle] for cycle in selected["training_cycles"]],
        "all ordered earlier cycle blocks required",
    )
    metadata = pd.read_csv(fold["selected"]["path"], usecols=["cycle", "observation_id"])
    require(
        metadata.cycle.lt(target).all()
        and metadata.observation_id.tolist() == [oid for b in blocks for oid in b.coordinates.observation_ids],
        "selected row/block identity mismatch",
    )
    math_model.validated_blocks(blocks)
    return plan, fold, tuple(blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared-v1")
    prepare(parser.parse_args().output)
