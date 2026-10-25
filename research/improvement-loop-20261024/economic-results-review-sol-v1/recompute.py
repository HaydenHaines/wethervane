"""Read-only independent arithmetic review; never imports runner or fits."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "economic-execution-v1"
RUN = SOURCE / "attempt-v1"
ARMS = ("H0", "F0", "E01", "E01_sd025")
PINS = {}


def read(path):
    data = path.read_bytes()
    PINS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def checked(item):
    data = Path(item["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"]
    PINS[item["path"]] = item["sha256"]


def main():
    plan_path = SOURCE / "prepared-v2/plan.json"
    assert (
        hashlib.sha256(plan_path.read_bytes()).hexdigest()
        == "e3faefeb89a5bca837bc7f8b01e666112d563b95fe16e909060edf5aa1cbfc50"
    )
    plan = read(plan_path)
    launch = read(RUN / "launch.json")
    checked(launch["plan"])
    checked(launch["runtime_certificate"])
    for item in plan["source_pins"]:
        checked(item)
    findings = []
    errors = []
    chronology = []
    preparation = read(Path(plan["preparation"]["path"]))
    outcome_pin = next(x for x in preparation["source_pins"] if x["path"].endswith("outcomes-SEPARATE.csv"))
    checked(outcome_pin)
    with Path(outcome_pin["path"]).open() as stream:
        actual = {r["race_id"]: float(r["actual_dem_share"]) for r in csv.DictReader(stream)}
    for year in (2018, 2020, 2022, 2024):
        folder = RUN / str(year)
        if not (folder / "masks.json").exists():
            continue
        masks = read(folder / "masks.json")
        for item in masks["source_prediction_seals"]:
            checked(item)
        seal_times = [(folder / arm / "prediction-seal.json").stat().st_mtime_ns for arm in ARMS]
        mask_time = (folder / "masks.json").stat().st_mtime_ns
        score_times = [(folder / arm / "scores.json").stat().st_mtime_ns for arm in ARMS]
        assert max(seal_times) <= mask_time <= min(score_times)
        chronology.append(
            dict(
                year=year,
                last_arm_seal_mtime_ns=max(seal_times),
                mask_mtime_ns=mask_time,
                first_score_mtime_ns=min(score_times),
                interpretation="filesystem consistency only, not authenticated label-access timing",
            )
        )
        ids = masks["race_ids"]["all"]
        for arm in ARMS:
            directory = folder / arm
            seal = read(directory / "prediction-seal.json")
            assert (seal["target"], seal["arm"], seal["plan_sha256"]) == (year, arm, launch["plan_sha256"])
            checked(seal["predictions"])
            with np.load(seal["predictions"]["path"], allow_pickle=False) as bank:
                shares = bank["shares"]
                assert bank["race_ids"].tolist() == ids
                assert shares.shape == (4, 20000, len(ids))
                assert np.isfinite(shares).all() and ((shares >= 0) & (shares <= 1)).all()
                assert np.array_equal(bank["posterior_draw"], np.repeat(np.arange(4000), 5))
                assert np.array_equal(bank["replicate"], np.tile(np.arange(5), 4000))
            terminal = read(directory / "terminal.json")
            fit = read(directory / "fit-diagnostics.json")
            predictive = read(directory / "prediction-diagnostics.json")
            assert terminal["numerical_passed"] == (fit["passed"] and predictive["passed"]) == seal["numerical_passed"]
            for chain in terminal["raw_chains"]:
                checked(chain)
            score = read(directory / "scores.json")
            rows = score["rows"]
            assert [r["race_id"] for r in rows] == ids
            truth = np.array([r["actual_dem_share"] for r in rows])
            # stdlib and pandas CSV float parsing can differ by a final ULP.
            np.testing.assert_allclose(truth, [actual[r] for r in ids], rtol=0, atol=2e-16)
            flat = shares.reshape(-1, len(ids))
            probability = (flat > 0.5).mean(axis=0)
            if arm == "F0":
                assert masks["race_ids"]["competitive"] == [r for r, p in zip(ids, probability) if 0.1 <= p <= 0.9]
                assert masks["race_ids"]["near_even"] == [r for r, p in zip(ids, probability) if 0.4 <= p <= 0.6]
            q = np.quantile(flat, [0.025, 0.1, 0.5, 0.9, 0.975], axis=0)
            sorted_values = np.sort(flat, axis=0)
            count = len(flat)
            # E|X-y| - .5 E|X-X'|; order-statistic identity, independent of owner helper.
            crps = (
                abs(flat - truth).mean(axis=0)
                - ((2 * np.arange(1, count + 1) - count - 1)[:, None] * sorted_values).sum(axis=0) / count**2
            )
            brier = (probability - (truth > 0.5)) ** 2
            logscore = -np.log(np.clip(np.where(truth > 0.5, probability, 1 - probability), 1e-6, 1))
            wis = 0.5 * abs(q[2] - truth)
            independent = {"share_crps": crps, "win_brier": brier, "win_log_score": logscore}
            for level, alpha, lo, hi in [(80, 0.2, q[1], q[3]), (95, 0.05, q[0], q[4])]:
                interval = hi - lo + 2 / alpha * (np.maximum(lo - truth, 0) + np.maximum(truth - hi, 0))
                wis += alpha / 2 * interval
                independent[f"coverage{level}"] = ((lo <= truth) & (truth <= hi)).astype(float)
                independent[f"width{level}"] = hi - lo
            independent["share_wis"] = wis / 2.5
            # Independent finite-chain/batch variance calculation in original order.
            order = np.argsort(flat, axis=0)
            distance = (
                (2 * np.arange(count) - count)[:, None] * sorted_values
                + sorted_values.sum(axis=0)
                - 2 * (sorted_values.cumsum(axis=0) - sorted_values)
            ) / count
            pair = np.empty_like(flat)
            np.put_along_axis(pair, order, distance, axis=0)
            crps_influence = (abs(flat - truth) - pair).reshape(shares.shape)
            indicator = (shares > 0.5).astype(float)
            centered = indicator - probability
            win = (truth > 0.5).astype(float)
            clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            derivative = ((1 - win) / (1 - clipped) - win / clipped) * ((probability > 1e-6) & (probability < 1 - 1e-6))
            influences = [crps_influence, 2 * (probability - win) * centered, derivative * centered]
            for key, values in independent.items():
                errors.append(float(np.max(abs(values - [r[key] for r in rows]))))
                for stratum, members in masks["race_ids"].items():
                    index = [ids.index(r) for r in members]
                    if index:
                        errors.append(abs(float(values[index].mean()) - score["strata"][stratum][key]))
            for stratum, members in masks["race_ids"].items():
                index = [ids.index(r) for r in members]
                if not index:
                    continue
                for key, influence in zip(("share_crps", "win_brier", "win_log_score"), influences):
                    values = influence[:, :, index].mean(axis=2)
                    variances = [values.mean(axis=1).var(ddof=1) / 4]
                    for batch in (250, 500):
                        means = values.reshape(4, 20000 // batch, batch).mean(axis=2).ravel()
                        variances.append(means.var(ddof=1) / len(means))
                    errors.append(abs(float(np.sqrt(max(variances))) - score["strata"][stratum]["mcse"][key]))
            findings.append(
                dict(
                    year=year,
                    arm=arm,
                    numerical_passed=terminal["numerical_passed"],
                    all=score["strata"]["all"],
                    competitive=score["strata"]["competitive"],
                    near_even=score["strata"]["near_even"],
                    fit=fit,
                    predictive=predictive,
                )
            )
    assert max(errors) < 1e-10
    top = read(RUN / "terminal.json") if (RUN / "terminal.json").exists() else None
    if top:
        checked(top["summary"])
        summary = read(Path(top["summary"]["path"]))
        assert top["all_numerical_passed"] == all(f["numerical_passed"] for f in findings)
        assert not top["count_qualification"] and not top["promotion_authorized"]
        by_key = {(f["year"], f["arm"]): f for f in findings}
        for cycle in summary["cycles"]:
            for arm in ARMS:
                for stratum in ("all", "competitive", "near_even"):
                    assert cycle["arms"][arm]["strata"][stratum] == by_key[cycle["target"], arm][stratum]
        for tier, years in (("A", (2018, 2022)), ("B", (2020, 2024))):
            means = summary["tiers"][tier]["equal_cycle_means"]
            for arm in ARMS:
                for key, value in means[arm].items():
                    assert abs(value - np.mean([by_key[y, arm]["all"][key] for y in years])) < 1e-14
        for row in summary["regressions"]["per_cycle"]:
            base = by_key[row["target"], "F0"][row["stratum"]]
            challenger = by_key[row["target"], row["arm"]][row["stratum"]]
            key = row["metric"]
            delta = challenger[key] - base[key]
            error = np.hypot(challenger["mcse"][key], base["mcse"][key])
            assert abs(delta - row["challenger_minus_F0"]) < 1e-14
            assert abs(error - row["numerical_mcse"]) < 1e-14
        for row in summary["regressions"]["equal_cycle_tiers"]:
            if row["status"] != "available":
                continue
            parts = [
                r
                for r in summary["regressions"]["per_cycle"]
                if all(r[k] == row[k] for k in ("tier", "arm", "stratum", "metric"))
            ]
            assert len(parts) == 2
            assert abs(np.mean([p["challenger_minus_F0"] for p in parts]) - row["equal_cycle_delta"]) < 1e-14
            assert abs(np.sqrt(sum(p["numerical_mcse"] ** 2 for p in parts)) / 2 - row["numerical_mcse"]) < 1e-14
    report = dict(
        inspected_sha256=PINS,
        max_independent_score_error=max(errors),
        completed_scored_jobs=len(findings),
        findings=findings,
        chronology=chronology,
        terminal=top,
        fit_or_sampler_invoked=False,
    )
    (HERE / "recomputation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(jobs=len(findings), max_error=max(errors), terminal=top is not None)))


if __name__ == "__main__":
    main()
