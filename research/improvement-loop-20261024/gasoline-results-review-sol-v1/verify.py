"""Independent, read-only G01 chain/predictive/lineage verifier.

Only complete three-arm folds with a frozen F0 mask are inspected. Target
numeric outcomes are never opened by this program.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import expit, logit

BASE = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
EXEC = BASE / "gasoline-execution-v1"
RUN = EXEC / "attempt-v1"
PLAN_SHA = "63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab"
PREP_SHA = "ee3cdefc9b3a5f9bc98598acd47ef7db81d0393f32c874396843e587566849ef"
ARMS = ("H0", "F0", "G01")
YEARS = (2018, 2020, 2022, 2024)
PINS = {}


def digest(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    PINS[str(path)] = value
    return value


def pinned(item):
    path = Path(item["path"])
    assert digest(path) == item["sha256"], f"changed pin: {path}"
    return path


def read(path):
    digest(path)
    return json.loads(Path(path).read_bytes())


def rows(path):
    digest(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def features(row):
    state = float(row["prior_presidential_dem_share"])
    national = float(row["prior_national_presidential_dem_share"])
    assert 0 < state < 1 and 0 < national < 1
    party = float(row["president_party_asof"])
    midterm = float(row["midterm"])
    assert party in (-1, 1) and midterm in (0, 1)
    assert row["dem_incumbent"] in ("True", "False")
    assert row["rep_incumbent"] in ("True", "False")
    assert int(row["prior_presidential_cycle"]) < int(row["cycle"])
    s, n = logit(state), logit(national)
    return {
        "one": 1.0,
        "relative_lean": float(s - n),
        "national": float(n),
        "incumbency": float((row["dem_incumbent"] == "True") - (row["rep_incumbent"] == "True")),
        "party": party,
        "party_midterm": party * midterm,
        "president_oriented_twip_gas_log_change_13_weeks_z": float(
            row["president_oriented_twip_gas_log_change_13_weeks_z"]
        ),
    }


def target_linear(recipe, target_rows, parameter_names):
    lookup = {name: i for i, name in enumerate(parameter_names)}
    offset = np.zeros(len(target_rows))
    design = np.zeros((len(target_rows), len(parameter_names)))
    for j, row in enumerate(target_rows):
        values = features(row)
        for provider in recipe["providers"]:
            for term in provider["terms"]:
                amount = values[term["feature_id"]]
                coef = term["coefficient"]
                if isinstance(coef, dict):
                    design[j, lookup[coef["parameter_id"]]] += amount
                else:
                    offset[j] += float(coef) * amount
    return offset, design


def check_scaler(fold, target_rows, all_rows, gas):
    year = fold["target_cycle"]
    scaler = fold["scaler"]
    earlier = list(range(2002, year, 2))
    assert scaler["training_cycles"] == earlier
    assert scaler["independent_macro_realizations"] == len(earlier)
    assert scaler["ddof"] == 0
    by_cycle = {}
    for row in all_rows:
        cycle = int(row["cycle"])
        if cycle <= year:
            by_cycle.setdefault(cycle, []).append(row)
    assert list(sorted(by_cycle)) == list(range(2002, year + 2, 2))
    oriented = {}
    for cycle, group in by_cycle.items():
        party = {int(row["president_party_asof"]) for row in group}
        assert len(party) == 1 and party <= {-1, 1}
        oriented[cycle] = party.pop() * gas[cycle]
    previous = np.array([oriented[c] for c in earlier])
    mean, sd = float(previous.mean()), float(previous.std(ddof=0))
    np.testing.assert_allclose([mean, sd], [scaler["mean"], scaler["sd"]], rtol=0, atol=1e-12)
    for cycle, group in by_cycle.items():
        expected = (oriented[cycle] - mean) / sd
        z = np.array([features(row)["president_oriented_twip_gas_log_change_13_weeks_z"] for row in group])
        np.testing.assert_allclose(z, expected, rtol=0, atol=1e-12)
    assert [row["race_id"] for row in target_rows] == fold["target_race_ids"]
    return {"earlier_unique_cycles": len(earlier), "mean": mean, "sd": sd}


def chain_diagnostics(banks, expected, gates):
    joint = az.concat(*banks, dim="chain")
    names = ["beta", "tau_cycle", "tau_race", "cycle_z"]
    values = {}
    for key, method in (("max_rank_rhat", "rank"), ("min_bulk_ess", "bulk"), ("min_tail_ess", "tail")):
        ds = az.rhat(joint, var_names=names, method=method) if key == "max_rank_rhat" else az.ess(
            joint, var_names=names, method=method
        )
        vector = np.concatenate([np.asarray(ds[name]).ravel() for name in names])
        assert np.isfinite(vector).all()
        values[key] = float(vector.max() if key == "max_rank_rhat" else vector.min())
    values["divergences"] = int(joint.sample_stats.diverging.sum())
    values["max_depth_hits"] = int((joint.sample_stats.tree_depth >= 10).sum())
    for key, value in values.items():
        np.testing.assert_allclose(value, expected["values"][key], rtol=1e-12, atol=1e-12)
    checks = {
        "rhat": values["max_rank_rhat"] <= gates["max_rank_rhat"],
        "bulk": values["min_bulk_ess"] >= gates["min_bulk_ess"],
        "tail": values["min_tail_ess"] >= gates["min_tail_ess"],
        "divergences": values["divergences"] == gates["divergences"],
        "depth": values["max_depth_hits"] == gates["max_depth_hits"],
    }
    assert expected["passed"] == all(checks.values())
    return values, checks, joint


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--year", type=int, choices=YEARS, help="Review one complete sealed fold")
    args = cli.parse_args()
    digest(Path(__file__))
    plan_path = EXEC / "prepared-v2/plan.json"
    assert digest(plan_path) == PLAN_SHA
    plan = read(plan_path)
    prep_path = pinned(plan["preparation"])
    assert digest(prep_path) == PREP_SHA
    prep = read(prep_path)
    launch = read(RUN / "launch.json")
    assert launch["plan_sha256"] == PLAN_SHA
    assert pinned(launch["plan"]) == plan_path
    assert launch["runtime_certificate"] == plan["inherited_runtime_acceptance"]
    pinned(launch["runtime_certificate"])
    # A source-byte check of the outcome CSV would open every target's numbers.
    # Defer that one pin until the entire attempt has sealed, even for a single
    # fold review. All other source files can be checked now.
    complete_attempt = (RUN / "terminal.json").is_file()
    deferred_pins = []
    for item in plan["source_pins"] + prep["source_pins"]:
        if item["path"].endswith("outcomes-SEPARATE.csv") and not complete_attempt:
            deferred_pins.append(item)
        else:
            pinned(item)
    pinned(plan["requirements"])
    assert plan["execution_order"] == "chronological_fold; all_three_prediction_seals_and_F0_masks_before_target_score_labels"
    assert not plan["promotion_authorized"]
    assert plan["fit_count"] == 12 and len(plan["jobs"]) == 12
    assert plan["prediction"]["replicates_per_joint_row"] == 5
    gas_pin = next(x for x in prep["source_pins"] if x["path"].endswith("gasoline-twip-cutoff-v1/panel.json"))
    gas_panel = read(pinned(gas_pin))
    gas = {int(r["cycle"]): float(r["log_change_13_weeks"]) for r in gas_panel["panel"]}
    assert set(gas) == set(range(2002, 2026, 2))
    for row in gas_panel["panel"]:
        current, earlier = row["current"], row["lag13"]
        np.testing.assert_allclose(
            np.log(current["value"] / earlier["value"]), row["log_change_13_weeks"], rtol=0, atol=1e-12
        )
        assert not row["crosses_2018_method_break"]
        cutoff = datetime.fromisoformat(row["cutoff_utc"]).replace(hour=0, minute=0, second=0)
        assert all(datetime.fromisoformat(x["available_upper_bound_utc"]) < cutoff for x in (current, earlier))
    reviewed = []
    for fold in prep["folds"]:
        year = fold["target_cycle"]
        if args.year and year != args.year:
            continue
        directory = RUN / str(year)
        seals = [directory / arm / "prediction-seal.json" for arm in ARMS]
        mask_path = directory / "masks.json"
        if not all(path.is_file() for path in seals + [mask_path]):
            if args.year:
                raise RuntimeError(f"{year} has not sealed all three predictions and F0 mask")
            continue
        mask = read(mask_path)
        assert mask["race_ids"]["all"] == fold["target_race_ids"]
        assert mask["source_prediction_seals"] == [
            {"path": str(path), "sha256": digest(path)} for path in seals
        ]
        seal_objects = [read(path) for path in seals]
        for arm, seal in zip(ARMS, seal_objects, strict=True):
            assert (seal["plan_sha256"], seal["target"], seal["arm"]) == (PLAN_SHA, year, arm)
            pinned(seal["predictions"])
        assert max(path.stat().st_mtime_ns for path in seals) <= mask_path.stat().st_mtime_ns
        feature_path = pinned(fold["features"])
        feature_rows = rows(feature_path)
        forbidden = {"actual_dem_share", "dem_votes", "rep_votes", "winner"}
        assert not forbidden.intersection(feature_rows[0])
        target_rows = [r for r in feature_rows if int(r["cycle"]) == year]
        assert len(target_rows) == len(fold["target_race_ids"])
        scaler_check = check_scaler(fold, target_rows, feature_rows, gas)
        jobs = [j for j in plan["jobs"] if j["target"] == year]
        assert [j["arm"] for j in jobs] == list(ARMS)
        support = [j["source_binding"]["training_race_ids"] for j in jobs]
        assert all(ids == support[0] for ids in support)
        for key in ("feature_rows_sha256", "outcome_rows_sha256", "response_sha256"):
            assert len({j["source_binding"][key] for j in jobs}) == 1
        assert len(support[0]) == {2018: 236, 2020: 266, 2022: 294, 2024: 324}[year]
        assert all(int(r.split(":")[0]) < year and int(r.split(":")[0]) >= 2002 for r in support[0])
        assert [j["source_binding"]["cycles"] for j in jobs] == [list(range(2002, year, 2))] * 3
        f_providers = {p["instance_id"]: p for p in jobs[1]["recipe"]["providers"]}
        g_providers = {p["instance_id"]: p for p in jobs[2]["recipe"]["providers"]}
        assert set(g_providers) == set(f_providers) | {"economy"}
        assert all(g_providers[name] == provider for name, provider in f_providers.items())
        gas_spec = g_providers["economy"]
        assert gas_spec["parameters"] == [
            {"mean": 0.0, "owner": "economy", "parameter_id": "economy.twip_gas_change", "sd": 0.05}
        ]
        assert gas_spec["terms"][0]["feature_id"] == "president_oriented_twip_gas_log_change_13_weeks_z"
        arm_records = []
        for job, seal in zip(jobs, seal_objects, strict=True):
            arm = job["arm"]
            home = directory / arm
            terminal = read(home / "terminal.json")
            fit = read(home / "fit-diagnostics.json")
            predictive = read(home / "prediction-diagnostics.json")
            assert terminal["plan_sha256"] == PLAN_SHA
            assert terminal["forecast_qualified"] is False and terminal["promotion_authorized"] is False
            assert len(terminal["raw_chains"]) == 4
            banks = []
            for chain, item in enumerate(terminal["raw_chains"]):
                path = pinned(item)
                receipt = read(home / f"chain-{chain}-complete.json")
                assert receipt["raw_bank"] == item
                assert (receipt["chain"], receipt["seed"], receipt["job"], receipt["plan_sha256"]) == (
                    chain, job["chain_seeds"][chain], [year, arm], PLAN_SHA
                )
                bank = az.from_netcdf(path)
                assert bank.posterior.sizes["draw"] == 4000 and bank.warmup_posterior.sizes["draw"] == 1500
                assert bank.posterior.chain.values.tolist() == [chain]
                assert bank.posterior.coefficient.values.tolist() == job["source_binding"]["parameter_names"]
                assert bank.posterior.source_cycle.values.tolist() == list(range(2002, year, 2))
                banks.append(bank)
            values, checks, joint = chain_diagnostics(banks, fit, plan["fit_gates"])
            pv = predictive["values"]
            pg = plan["predictive_gates"]
            predictive_checks = {
                "finite": all(np.isfinite(list(pv.values()))),
                "rhat": pv["max_rank_rhat"] <= pg["max_rank_rhat"],
                "bulk": pv["min_bulk_ess"] >= pg["min_bulk_ess"],
                "tail": pv["min_tail_ess"] >= pg["min_tail_ess"],
                "wins": pv["max_win_mcse"] <= pg["max_win_mcse"],
                "subtotal": pv["subset_seat_mean_mcse"] <= pg["max_subset_seat_mean_mcse"],
            }
            assert predictive["checks"] == predictive_checks
            assert predictive["passed"] == all(predictive_checks.values())
            with np.load(seal["predictions"]["path"], allow_pickle=False) as bank:
                assert bank["race_ids"].tolist() == fold["target_race_ids"]
                shape = (4, 20000, len(target_rows))
                assert bank["logits"].shape == bank["shares"].shape == bank["new_race_z"].shape == shape
                assert bank["new_cycle_z"].shape == (4, 20000)
                assert np.array_equal(bank["posterior_draw"], np.repeat(np.arange(4000), 5))
                assert np.array_equal(bank["replicate"], np.tile(np.arange(5), 4000))
                assert np.isfinite(bank["logits"]).all()
                assert np.isfinite(bank["new_cycle_z"]).all() and np.isfinite(bank["new_race_z"]).all()
                offset, design = target_linear(job["recipe"], target_rows, job["source_binding"]["parameter_names"])
                beta = np.asarray(joint.posterior.beta)
                tc = np.asarray(joint.posterior.tau_cycle)
                tr = np.asarray(joint.posterior.tau_race)
                assert beta.shape == (4, 4000, len(job["source_binding"]["parameter_names"]))
                assert tc.shape == tr.shape == (4, 4000)
                assert (tc > 0).all() and (tr > 0).all()
                location = offset + beta @ design.T
                reconstructed = location[:, :, None, :] + tc[:, :, None, None] * bank["new_cycle_z"].reshape(
                    4, 4000, 5, 1
                ) + tr[:, :, None, None] * bank["new_race_z"].reshape(4, 4000, 5, len(target_rows))
                logit_error = float(np.max(np.abs(reconstructed.reshape(shape) - bank["logits"])))
                share_error = float(np.max(np.abs(expit(bank["logits"]) - bank["shares"])))
                assert logit_error < 1e-12 and share_error < 1e-15
                if arm == "F0":
                    p = (bank["shares"] > 0.5).mean(axis=(0, 1))
                    ids = fold["target_race_ids"]
                    assert mask["race_ids"]["competitive"] == [r for r, x in zip(ids, p) if 0.1 <= x <= 0.9]
                    assert mask["race_ids"]["near_even"] == [r for r, x in zip(ids, p) if 0.4 <= x <= 0.6]
            assert terminal["numerical_passed"] == fit["passed"] and predictive["passed"]
            assert seal["numerical_passed"] == terminal["numerical_passed"]
            assert terminal["status"] == ("completed" if terminal["numerical_passed"] else "numerical_failed_preserved")
            arm_records.append({
                "arm": arm, "fit_values": values, "fit_checks": checks, "predictive_stored": predictive,
                "predictive_threshold_checks": predictive_checks,
                "numerical_passed": terminal["numerical_passed"], "max_logit_error": logit_error,
                "max_share_error": share_error,
            })
        score_times = [directory / arm / "scores.json" for arm in ARMS]
        if any(x.exists() for x in score_times):
            assert all(x.exists() for x in score_times)
            assert mask_path.stat().st_mtime_ns <= min(x.stat().st_mtime_ns for x in score_times)
        reviewed.append({"year": year, "tier": fold["tier"], "scaler": scaler_check, "arms": arm_records,
                         "seal_before_mask_before_score_mtime_consistency": True})
    top = None
    if complete_attempt:
        top = read(RUN / "terminal.json")
        assert top["plan_sha256"] == PLAN_SHA
        assert top["status"] == "completed_F_only_comparison"
        assert top["count_qualification"] is False and top["promotion_authorized"] is False
        pinned(top["summary"])
        if args.year is None:
            assert len(reviewed) == 4 and sum(len(r["arms"]) for r in reviewed) == 12
            assert top["all_numerical_passed"] == all(
                arm["numerical_passed"] for fold_record in reviewed for arm in fold_record["arms"]
            )
    report = {
        "schema": "G01IndependentRawVerifierSolV1", "utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_folds": reviewed, "inspected_sha256": PINS,
        "deferred_outcome_source_pins": deferred_pins,
        "terminal": top,
        "numeric_target_labels_parsed": False, "sampler_or_predictive_simulation_invoked": False,
        "limitations": ["Filesystem chronology is consistency evidence, not authenticated first label access.",
                        "Stored predictive ArviZ gates are checked for status but not independently recomputed."],
    }
    (HERE / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"reviewed_years": [r["year"] for r in reviewed],
                      "failed_gates": [(r["year"], a["arm"]) for r in reviewed for a in r["arms"]
                                       if not a["numerical_passed"]]}))


if __name__ == "__main__":
    main()
