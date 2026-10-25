"""Frozen-input standard polling calibration and national aggregate rerun."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.forecasting.poll_covariance_v1 import (
    PollCoordinates, build_poll_likelihood, fit_error_scales,
)
from src.forecasting.polling_v1 import (
    EmpiricalFundamentals, posterior_diagnostics, require, sample_poll_posterior,
)
from src.forecasting.response_v5_reuse import correlated_mcse

HERE = Path(__file__).resolve().parent
PRIOR_DIR = ROOT / "research/national-senate-forecast-20261020"
PLAN = HERE / "standard-plan-root-v1.json"
ADDENDUM = HERE / "standard-plan-addendum-root-v1.json"
HISTORICAL = HERE / "historical-intake-sol/derived"
CURRENT = HERE / "current-intake-sol/current-poll-catalog-v1.json"
CURRENT_ADAPTER = HERE / "current-intake-sol/aggregate-observations-v1.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def old_runner():
    spec = importlib.util.spec_from_file_location("prior_forecast_runner", PRIOR_DIR / "forecast_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(folder, inputs):
    folder.mkdir(parents=True, exist_ok=False)
    snapshots = folder / "source-snapshots"
    snapshots.mkdir()
    files = [
        Path(__file__), PLAN, ADDENDUM, ROOT / "src/forecasting/polling_v1.py",
        ROOT / "src/forecasting/poll_covariance_v1.py",
        ROOT / "src/forecasting/response_v5_reuse.py",
        PRIOR_DIR / "forecast_runner.py", *inputs,
    ]
    optional = ROOT / "src/forecasting/poll_diagnostics_v1.py"
    if optional.exists():
        files.append(optional)
    inventory = []
    for i, path in enumerate(files):
        path = path.resolve()
        target = snapshots / f"{i:02d}-{path.name}"
        shutil.copyfile(path, target)
        inventory.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                          "snapshot": str(target.relative_to(folder))})
    write(folder / "input-receipt.json", {"started_utc": now(), "files": inventory})


def verify_input_snapshot(folder):
    inventory = json.loads((folder / "input-receipt.json").read_text())["files"]
    for row in inventory:
        require(sha(ROOT / row["path"]) == row["sha256"]
                and sha(folder / row["snapshot"]) == row["sha256"], "Input/source changed during execution")


def calibrate(folder):
    plan = json.loads(PLAN.read_text())
    polls_path = HISTORICAL / "historical_poll_panel.csv"
    outcome_path = HISTORICAL / "outcomes_allowed_cycles_SEPARATE.csv"
    prepare(folder, [polls_path, outcome_path])
    polls, outcomes = pd.read_csv(polls_path), pd.read_csv(outcome_path)
    allowed = set(plan["allowed_error_training_cycles"])
    require(set(polls.cycle) <= allowed and set(outcomes.cycle) <= allowed, "Disallowed historical cycle")
    selected = polls.loc[polls.days_to_election.between(0, plan["selection"]["maximum_days_to_election"])].copy()
    selected["pollster_key"] = selected.pollster.str.strip().str.casefold()
    selected["lead_bin"] = selected.days_to_election // 14
    selected = selected.sort_values(["end_date", "observation_id"]).drop_duplicates(
        ["cycle", "race_id", "pollster_key", "lead_bin"], keep="last"
    )
    selected = selected.merge(outcomes[["race_id", "actual_dem_share"]], on="race_id",
                              how="left", validate="many_to_one").sort_values(["cycle", "race_id", "observation_id"])
    require(selected.actual_dem_share.notna().all(), "Missing admitted historical outcome")
    q, truth = selected.dem_share.to_numpy(), selected.actual_dem_share.to_numpy()
    require(((q > 0) & (q < 1) & (truth > 0) & (truth < 1)).all(), "Endpoint historical share")
    residual = logit(q) - logit(truth)
    coordinates = PollCoordinates(
        tuple(selected.observation_id), tuple(selected.race_id),
        tuple(selected.cycle.astype(str)), tuple(selected.pollster_key),
        selected.days_to_election.to_numpy(),
    )
    selected.to_csv(folder / "selected-training-polls.csv", index=False)
    config = plan["covariance_model"]
    fit = fit_error_scales(
        coordinates, residual, prior_medians=config["regularizer_medians"],
        prior_log_sd=config["regularizer_log_sd"],
        starts=tuple(config["optimizer_start_multipliers"]),
    )
    fit.update({
        "plan_sha256": sha(PLAN), "addendum_sha256": sha(ADDENDUM), "completed_utc": now(),
        "selected_training_sha256": sha(folder / "selected-training-polls.csv"),
        "per_cycle_rows": {str(k): int(v) for k, v in selected.groupby("cycle").size().items()},
        "training_races": int(selected.race_id.nunique()),
        "training_pollsters": int(selected.pollster_key.nunique()),
        "historical_availability": plan["selection"]["historical_availability"],
        "historical_source_limitations": [
            "Original nominal/effective N and respondent universe unknown",
            "Unknown repeated sample overlap; one row per house/race/14-day bin retained",
            "Historical author candidate identities not independently verified",
            "Four national cycles weakly identify common-error variance",
        ],
    })
    write(folder / "calibration.json", fit)
    verify_input_snapshot(folder)
    print(json.dumps({"calibration": str(folder), "fit": fit}), flush=True)


def current_observations(data, plan):
    addendum = json.loads(ADDENDUM.read_text())
    require(sha(CURRENT) == addendum["current_catalog_sha256"], "Current catalog changed")
    require(sha(CURRENT_ADAPTER) == addendum["current_adapter_sha256"], "Current metadata adapter changed")
    catalog = json.loads(CURRENT.read_text())
    provider = pd.read_csv(CURRENT_ADAPTER).set_index("sample_id").fieldwork_provider_id.to_dict()
    cutoff_instant = datetime.fromisoformat(plan["current_poll_cutoff_utc"].replace("Z", "+00:00"))
    acquisition_instant = datetime.strptime(catalog["acquisition_cutoff_utc"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
    require(acquisition_instant <= cutoff_instant, "Source acquisition after information cutoff")
    require(catalog["roster_sha256"] == sha(PRIOR_DIR / "inputs-frozen-root-v1.json"),
            "Poll roster does not match frozen fundamentals roster")
    cutoff = date.fromisoformat(plan["current_poll_cutoff_utc"][:10])
    lookup = {r["race_id"]: r for r in data["races"]}
    retained, excluded = [], []
    for source in catalog["records"]:
        if not source["selected_for_aggregate"]:
            continue
        row = dict(source)
        require(row["race_id"] in lookup, "Unknown race")
        race = lookup[row["race_id"]]
        require(row["first_candidate_id"] == race["first_contender"]["id"]
                and row["second_candidate_id"] == race["second_contender"]["id"], "Candidate mismatch")
        require(row["publication_date"] and date.fromisoformat(row["publication_date"]) <= cutoff,
                "Future or undated publication")
        if row["publication_time_utc"]:
            publication_instant = datetime.fromisoformat(row["publication_time_utc"].replace("Z", "+00:00"))
            require(publication_instant.tzinfo is not None and publication_instant <= cutoff_instant,
                    "Release after information instant")
        require(bool(row["source_files"]), "Missing acquired source evidence")
        for source_file in row["source_files"]:
            require(sha(ROOT / source_file["path"]) == source_file["sha256"], "Acquired source bytes changed")
        row["information_time_basis"] = (
            "Publisher date plus collector's pinned batch-acquisition cutoff; exact publisher time unavailable"
            if not row["publication_time_utc"] else "Publisher timestamp and pinned acquisition cutoff"
        )
        require(date.fromisoformat(row["field_end"]) <= cutoff, "Fieldwork after cutoff")
        require(row["admission_status"] in {
            "primary_documented_qualified_aggregate_only", "publisher_reported_qualified_aggregate_only"
        }, "Unadmitted current poll")
        age = (date.fromisoformat(race["event_date"]) - date.fromisoformat(row["field_end"])).days
        if age > plan["selection"]["maximum_days_to_election"]:
            excluded.append({"sample_id": row["sample_id"], "reason": "outside 100-day election window"})
            continue
        require(age >= 0, "Poll after election")
        require(0 < row["first_pct"] < 100 and 0 < row["second_pct"] < 100, "Endpoint projection")
        require(abs(row["first_conditional_share"] - row["first_pct"] / (row["first_pct"] + row["second_pct"])) < 1e-12,
                "Invalid principal-pair projection")
        row["days_to_election"] = age
        row["pollster_key"] = str(provider.get(row["sample_id"]) or row["pollster"]).strip().casefold()
        retained.append(row)
    # One latest sample per house/race. Source alternatives never become extra observations.
    retained.sort(key=lambda r: (r["field_end"], r["publication_date"], r["sample_id"]))
    latest = {}
    for row in retained:
        key = (row["race_id"], row["pollster_key"])
        if key in latest:
            excluded.append({"sample_id": latest[key]["sample_id"], "reason": "earlier same-provider race sample"})
        latest[key] = row
    retained = sorted(latest.values(), key=lambda r: (r["race_id"], r["pollster_key"]))
    require(len({r["sample_id"] for r in retained}) == len(retained), "Shared sample needs covariance adapter")
    return retained, excluded


def summarize(data, prediction, baseline, observation_rows):
    helper = old_runner()
    shape = prediction["shares"].shape[:2]
    wins, shares = prediction["first_wins"].reshape(-1, len(data["races"])), prediction["shares"].reshape(-1, len(data["races"]))
    race_mcse = correlated_mcse(prediction["first_wins"].astype(float))["mcse"]
    prior = {r["race_id"]: r for r in baseline["races"]}
    records = []
    for j, race in enumerate(data["races"]):
        before = prior[race["race_id"]]
        polls = [r for r in observation_rows if r["race_id"] == race["race_id"]]
        win, share = float(wins[:, j].mean()), float(shares[:, j].mean())
        records.append({
            "race_id": race["race_id"], "state": race["state"], "seat_id": race["seat_id"],
            "first_contender": race["first_contender"], "second_contender": race["second_contender"],
            "first_win_probability": win, "second_win_probability": 1 - win,
            "first_share_mean": share,
            "first_share_quantiles": dict(zip(("p05", "p50", "p95"), np.quantile(shares[:, j], [.05, .5, .95]).tolist())),
            "first_minus_second_margin_mean_pp": 100 * (2 * share - 1),
            "win_probability_mcse": float(race_mcse[j]), "admitted_poll_observations": len(polls),
            "latest_field_end": max((p["field_end"] for p in polls), default=None),
            "fundamentals_first_win_probability": before["first_win_probability"],
            "polling_win_probability_change_pp": 100 * (win - before["first_win_probability"]),
            "polling_margin_change_pp": 200 * (share - before["first_share_mean"]),
            "share_interpretation": race["share_interpretation"], "rule_proxy": race["rule_proxy"],
            "excluded_options_assumption": race["excluded_options_assumption"],
        })
    groups, chamber, controls, monitored, monitor_names = {}, {}, {}, [], []
    for name, members in helper.membership_groups(data).items():
        seats, fixed, first, second = helper.group_draws(data, wins, members)
        require(abs(seats.mean() - fixed - (wins.mean(axis=0) * first + (1 - wins.mean(axis=0)) * second).sum()) < 1e-10,
                "Seat expectation conservation failure")
        groups[name] = seats
        monitored.append(seats.reshape(shape))
        monitor_names.append("seats:" + name)
        chamber[name] = {
            "expected_seats": float(seats.mean()),
            "seat_quantiles": dict(zip(("p05", "p50", "p95"), np.quantile(seats, [.05, .5, .95], method="inverted_cdf").tolist())),
            "pmf": (np.bincount(seats, minlength=101) / len(seats)).tolist(),
        }
    require(np.all(groups["party_D"] + groups["party_R"] + groups["party_I"] == 100), "Chamber conservation failure")
    for scenario in data["caucus_scenarios"]:
        prefix = "scenario:" + scenario["id"] + ":"
        partition = helper.control_partition(groups[prefix + "D"], groups[prefix + "R"], data["tie_break_party"])
        labels = ("democratic_majority_probability", "republican_majority_probability", "unresolved_organizing_probability")
        controls[scenario["id"]] = {}
        for label, indicators in zip(labels, partition):
            controls[scenario["id"]][label] = float(indicators.mean())
            monitored.append(indicators.reshape(shape).astype(float))
            monitor_names.append("control:" + scenario["id"] + ":" + label)
        require(abs(sum(controls[scenario["id"]].values()) - 1) < 1e-12, "Control partition failure")
    monitor = np.stack(monitored, axis=2)
    diag = correlated_mcse(monitor)
    chamber_diag = {
        "coordinates": monitor_names, "rhat": diag["rank_rhat"].tolist(),
        "mcse": diag["mcse"].tolist(),
        "max_control_mcse": float(max(s for name, s in zip(monitor_names, diag["mcse"]) if name.startswith("control:"))),
    }
    try:
        from src.forecasting.poll_diagnostics_v1 import effective_sample_sizes
        effective = effective_sample_sizes(monitor)
        chamber_diag.update({k + "_ess": np.asarray(v).tolist() for k, v in effective.items()})
    except ImportError:
        chamber_diag["ess_status"] = "not yet implemented; numerical qualification incomplete"
    return {"races": records, "chamber": chamber, "control_scenarios": controls}, chamber_diag, groups


def forecast(folder, calibration_path, multiplier=1.0, cycle_multiplier=1.0):
    plan = json.loads(PLAN.read_text())
    helper = old_runner()
    data_path = PRIOR_DIR / "inputs-frozen-root-v1.json"
    baseline_path = PRIOR_DIR / "forecast-v1/forecast.json"
    baseline_receipt_path = PRIOR_DIR / "forecast-v1/receipt.json"
    baseline_receipt = json.loads(baseline_receipt_path.read_text())
    require(sha(data_path) == baseline_receipt["input_sha256"], "Nonpoll inputs changed since prior forecast")
    require(sha(baseline_path) == baseline_receipt["artifacts"]["forecast.json"], "Prior forecast changed")
    require(sha(PRIOR_DIR / "forecast_runner.py") == baseline_receipt["runner_sha256"], "Prior helper changed")
    data = helper.validate_inputs(json.loads(data_path.read_text()))
    calibration = json.loads(calibration_path.read_text())
    require(calibration["plan_sha256"] == sha(PLAN), "Calibration uses a different frozen plan")
    require(calibration["addendum_sha256"] == sha(ADDENDUM), "Calibration uses a different frozen addendum")
    require(sha(helper.POSTERIOR) == plan["historical_prior_sha256"], "Historical posterior mismatch")
    prepare(folder, [calibration_path, CURRENT, CURRENT_ADAPTER, data_path, baseline_path, baseline_receipt_path])
    rows, excluded = current_observations(data, plan)
    write(folder / "admitted-observations.json", rows)
    write(folder / "excluded-observations.json", excluded)
    samples = helper.flatten_posterior(np.load(helper.POSTERIOR))
    prior = EmpiricalFundamentals(
        np.array([np.log(r["presidential_dem_votes"] / r["presidential_rep_votes"]) for r in data["races"]]),
        np.array([[1, int(r["first_contender"]["incumbent"]) - int(r["second_contender"]["incumbent"])] for r in data["races"]]),
        samples["beta"], samples["tau_cycle"], samples["tau_race"],
        tuple(r["race_id"] for r in data["races"]), plan["historical_prior_sha256"],
    )
    coords = PollCoordinates(
        tuple(r["sample_id"] for r in rows), tuple(r["race_id"] for r in rows),
        tuple("2026" for r in rows), tuple(r["pollster_key"] for r in rows),
        np.array([r["days_to_election"] for r in rows]),
    )
    scales = {key: value * multiplier for key, value in calibration["standard_deviations"].items()}
    scales["cycle"] *= cycle_multiplier
    polls = build_poll_likelihood(prior.race_ids, coords, [r["first_conditional_share"] for r in rows], scales)
    np.savez_compressed(folder / "poll-likelihood.npz", values=polls.values, covariance=polls.covariance,
                        race_indices=polls.race_indices, observation_ids=np.array(polls.observation_ids))
    write(folder / "run-specification.json", {
        "schema": "StandardPollingRunV1", "target_id": plan["target_id"],
        "covariance_scales": scales, "all_scale_multiplier": multiplier, "cycle_scale_multiplier": cycle_multiplier,
        "admitted_observations": len(rows), "admitted_races": len(set(coords.race_ids)),
        "sampler": plan["sampler"], "started_utc": now(),
    })
    config = plan["sampler"]
    prediction = sample_poll_posterior(
        prior, polls, chains=config["chains"], warmup=config["warmup"],
        draws=config["draws_per_chain"], seed=config["seed"],
    )
    np.savez_compressed(folder / "joint-draws.npz", **{
        k: v for k, v in prediction.items() if isinstance(v, np.ndarray)
    })
    diag = posterior_diagnostics(prediction)
    try:
        from src.forecasting.poll_diagnostics_v1 import effective_sample_sizes
        parameters = np.concatenate([
            prediction["shares"], prediction["beta"], prediction["tau_cycle"][:, :, None],
            prediction["tau_race"][:, :, None], prediction["shared_cycle"][:, :, None],
        ], axis=2)
        diag.update({k + "_ess": np.asarray(v).tolist() for k, v in effective_sample_sizes(parameters).items()})
    except ImportError:
        diag["ess_status"] = "not yet implemented; numerical qualification incomplete"
    baseline = json.loads(baseline_path.read_text())
    summary, chamber_diag, groups = summarize(data, prediction, baseline, rows)
    write(folder / "diagnostics.json", {"posterior": diag, "chamber": chamber_diag})
    np.savez_compressed(folder / "chamber-draws.npz", **groups)
    gates = plan["numerical_reporting_gates"]
    checks = {
        "posterior_rhat": diag["max_rhat"] <= gates["max_rank_split_rhat"],
        "chamber_rhat": max(chamber_diag["rhat"]) <= gates["max_rank_split_rhat"],
        "race_win_mcse": diag["max_win_probability_mcse"] <= gates["max_race_win_mcse"],
        "control_mcse": chamber_diag["max_control_mcse"] <= gates["max_control_mcse"],
        "posterior_bulk_ess": min(diag.get("bulk_ess", [0])) >= gates["min_bulk_ess_nonconstant"],
        "posterior_tail_ess": min(diag.get("tail_ess", [0])) >= gates["min_tail_ess_nonconstant"],
        "chamber_bulk_ess": min(chamber_diag.get("bulk_ess", [0])) >= gates["min_bulk_ess_nonconstant"],
        "chamber_tail_ess": min(chamber_diag.get("tail_ess", [0])) >= gates["min_tail_ess_nonconstant"],
    }
    write(folder / "numerical-gates.json", {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "thresholds": gates,
        "constant_event_limitation": "Zero empirical variation or MCSE does not prove a true zero probability.",
    })
    summary.update({
        "schema": "NationalSenateAggregatePollingForecastV1", "target_id": plan["target_id"],
        "input_cutoff_utc": plan["current_poll_cutoff_utc"], "execution_completed_utc": now(),
        "prior_forecast_sha256": sha(PRIOR_DIR / "forecast-v1/forecast.json"),
        "nonpoll_inputs": "Identical to preserved fundamentals forecast; current source intake extended through declared cutoff",
        "limitations": plan["claims"], "covariance_assumptions": plan["covariance_model"],
        "caucus_scenarios": data["caucus_scenarios"], "tie_break_party": data["tie_break_party"],
        "admitted_poll_observations": len(rows), "admitted_races": len(set(coords.race_ids)),
        "numerical_gate_status": "passed" if all(checks.values()) else "failed",
    })
    write(folder / "forecast.json", summary)
    pd.DataFrame(summary["races"]).drop(columns=["first_contender", "second_contender", "first_share_quantiles"]).to_csv(
        folder / "race-comparison.csv", index=False
    )
    verify_input_snapshot(folder)
    print(json.dumps({"forecast": str(folder), "control": summary["control_scenarios"],
                      "max_rhat": diag["max_rhat"], "max_control_mcse": chamber_diag["max_control_mcse"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["calibrate", "forecast"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--cycle-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.output)
    else:
        require(args.calibration is not None, "Calibration file required")
        forecast(args.output, args.calibration, args.scale_multiplier, args.cycle_multiplier)
