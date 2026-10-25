"""Calibration-only posterior on five log error SDs; no downstream feedback target.

The unchanged owner's Gaussian objective contains a quadratic regularizer. Here
we explicitly define a NEW proper untruncated Normal prior on log SDs; its
normalizing constant is included. Optimizer bounds are never used as a prior.
"""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import sysconfig
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import logit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
ATTEMPT = "research/senate-stack-integration-20261020/tx-disjoint-calibration-sol-v1/attempt-v1"
COMPONENTS = ("independent", "cycle", "race", "house", "movement")
MEDIANS = np.array([0.1, 0.06, 0.06, 0.03, 0.05])
LOG_SD = 0.7
COUNTS = {"2012": 467, "2016": 463, "2018": 288, "2020": 265}
EXCLUDED_EVENTS = {"2020:TX:senate:regular", "2024:TX:senate:regular"}
SEED = 2026102401
CHAIN_SEEDS = [2678211351, 1767967282, 2782219441, 2794758142]
CODE_FILES = {"calibration_bank.py", "prepare.py", "test_calibration_bank.py"}
SOURCE_CUTOFF = "2026-10-20T16:21:44Z"
ROUTE = "research/senate-stack-integration-20261020/actual-source-review-astra-v1/disjoint-calibration-route.json"
TRUSTED = {
    ATTEMPT + "/selected-disjoint-training.csv": "151a14846e850ff0b4537897a29faa772a5a5e0a81f8cd92f41fcdec202dcb61",
    ATTEMPT + "/calibration.json": "953b9c4989eb565e2776f48ff7a53c34b8b135afed4ee118c6d4860bd88e3374",
    "src/forecasting/poll_covariance_v1.py": "9ae130b3f314d75a5255c7c92fab502ec596a7cca2960090706fa203b133e05d",
    "src/forecasting/polling_v1.py": "5876ab790f43c8bd6324590f2a3d007f2da853bce49d0e60cd19142046ab0bcb",
    "src/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/forecasting/response_v5_reuse.py": "d4330647dba36d08d437237098cd9ac198657e1c9980d243b0e82b1ee65fd850",
    ATTEMPT + "/receipt.json": "07c53f2727a86680cf5f5e7acdc657e915728ab7310f39bd42d8b6f6bb2e2efe",
    ATTEMPT + "/excluded-original-rows.csv": "df7e7e85b32ec7f42751e5154b32b010c1255d408d1845ece2800b6efdd386e4",
    ROUTE: "f745bebf7f268a7b67c4fad010346c2efd42dd4258ad611ae59adde1d9f26361",
}
SAMPLER = dict(
    chains=4, cores=1, tune=1000, draws=1000, target_accept=0.9, max_treedepth=10, seed=SEED, chain_seeds=CHAIN_SEEDS
)
GATES = dict(max_rank_rhat=1.01, min_bulk_ess=400, min_tail_ess=400, divergences=0, depth_saturations=0, min_ebfmi=0.3)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def environment_packages():
    # Importing PyMC can add setuptools/_vendor to sys.path. Freeze installed
    # distributions rather than changing the inventory when that import occurs.
    paths = set(sysconfig.get_path(key) for key in ("purelib", "platlib"))
    return sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions(path=paths)})


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_pin(root, item):
    path = (Path(root) / item["path"]).resolve()
    require(path.is_relative_to(Path(root).resolve()), "pin outside snapshot root")
    data = path.read_bytes()
    require(sha(data) == item["sha256"], "source pin mismatch: " + item["path"])
    return data


def pin(root, path):
    return dict(path=str(Path(path).relative_to(root)), sha256=sha(Path(path).read_bytes()))


def validate_rows(rows, *, target_cycle=2026, source_cutoff=SOURCE_CUTOFF):
    """Validate frozen identities and retrospective scope, never upgrade proxies."""
    require(target_cycle == 2026, "this bank supports target 2026 only; no earlier time-safe claim")
    require(source_cutoff == SOURCE_CUTOFF, "source cutoff differs from frozen 2026 intake")
    require(Counter(row["cycle"] for row in rows) == COUNTS, "training cycles/counts differ")
    ids = [row["observation_id"] for row in rows]
    require(len(set(ids)) == len(ids), "duplicate observation ID")
    require(len({row["race_id"] for row in rows}) == 105, "training race support differs")
    for row in rows:
        cycle, state, office, contest = row["race_id"].split(":")
        require(cycle == row["cycle"] and state == row["state"], "canonical event identity mismatch")
        require(office == "senate" and contest == "regular", "unsupported event")
        require(row["race_id"] not in EXCLUDED_EVENTS, "excluded downstream event leaked")
        require(not (state == "TX" and cycle in {"2020", "2024"}), "downstream state/cycle leaked")
        require(int(cycle) < target_cycle, "training outcome is not earlier than target")
        election = datetime.fromisoformat(row["election_date"]).date()
        end = datetime.fromisoformat(row["end_date"]).date()
        available = datetime.fromisoformat(row["available_date"]).date()
        cutoff = datetime.fromisoformat(source_cutoff.replace("Z", "+00:00")).date()
        require(election.year == int(cycle) and available <= cutoff and election <= cutoff, "time cutoff failure")
        require(available == end <= election, "field-end proxy date mismatch")
        require(float(row["days_to_election"]) == (election - end).days, "poll age mismatch")
        require(
            row["availability_basis"] == "field_end_date_proxy_only_NOT_verified_publication", "proxy basis changed"
        )
        for name in ("publication_date_verified", "historical_version_verified", "strict_asof_qualified"):
            require(row[name] == "False", "source qualification changed; cannot upgrade date proxies")
        require(row["publication_date"] == "", "unexpected verified publication date")
        require(all(0 < float(row[name]) < 1 for name in ("dem_share", "actual_dem_share")), "interior shares required")
        require(bool(row["pollster_key"]), "missing pollster identity")


def load_prepared(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    require(plan["schema"] == "M01CalibrationPosteriorPlanV1", "unsupported plan")
    require(plan["target_cycle"] == 2026 and plan["source_cutoff_utc"] == SOURCE_CUTOFF, "time cutoff plan mismatch")
    require(plan["sampler"] == SAMPLER and plan["diagnostic_gates"] == GATES, "sampler/gates changed")
    require(
        plan["prior"]
        == dict(
            coordinates="log_standard_deviations",
            family="independent_untruncated_normal",
            medians=MEDIANS.tolist(),
            log_sd=LOG_SD,
            optimizer_bounds_used=False,
        ),
        "prior changed",
    )
    require(set(TRUSTED) <= {item["owner"]["path"] for item in plan["snapshots"]}, "required source pin missing")
    require(
        len({item["owner"]["path"] for item in plan["snapshots"]}) == len(plan["snapshots"]), "duplicate source pin"
    )
    for item in plan["snapshots"]:
        owner_path = item["owner"]["path"]
        expected_snapshot = (
            "snapshot-source/" + owner_path if owner_path.startswith("src/") else "inputs/" + Path(owner_path).name
        )
        require(item["snapshot"]["path"] == expected_snapshot, "snapshot source path mismatch")
        read_pin(directory, item["snapshot"])
        require(item["snapshot"]["sha256"] == item["owner"]["sha256"], "snapshot/owner identity mismatch")
        if item["owner"]["path"] in TRUSTED:
            require(item["owner"]["sha256"] == TRUSTED[item["owner"]["path"]], "trusted source identity mismatch")
    require(
        {item["path"] for item in plan["code_pins"]} == CODE_FILES and len(plan["code_pins"]) == len(CODE_FILES),
        "required code pin inventory differs",
    )
    for item in plan["code_pins"]:
        read_pin(HERE, item)
        read_pin(directory, dict(path="code/" + item["path"], sha256=item["sha256"]))
    read_pin(directory, plan["environment"]["freeze"])
    selected = next(
        item for item in plan["snapshots"] if item["owner"]["path"].endswith("selected-disjoint-training.csv")
    )
    calibration = next(item for item in plan["snapshots"] if item["owner"]["path"].endswith("/calibration.json"))
    rows = list(csv.DictReader(io.StringIO(read_pin(directory, selected["snapshot"]).decode())))
    validate_rows(rows, target_cycle=plan["target_cycle"], source_cutoff=plan["source_cutoff_utc"])
    original = json.loads(read_pin(directory, calibration["snapshot"]))
    require(
        original["selected_training"]["sha256"] == selected["owner"]["sha256"], "calibration/training binding differs"
    )
    require(
        set(original["excluded_event_keys"]) == EXCLUDED_EVENTS and original["downstream_event_overlap"] == [],
        "original exclusion binding differs",
    )
    return plan, rows, original


def load_provider(directory):
    """Import only the exact copied owner modules, without replacing any function."""
    directory = Path(directory).resolve()
    snapshot_root = directory / "snapshot-source"
    sys.path.insert(0, str(snapshot_root))
    module = importlib.import_module("src.forecasting.poll_covariance_v1")
    for name in ("poll_covariance_v1", "polling_v1", "response_v5_reuse"):
        loaded = sys.modules["src.forecasting." + name]
        expected = snapshot_root / "src/forecasting" / (name + ".py")
        require(Path(loaded.__file__).resolve() == expected, "provider resolved outside source snapshot")
    require(tuple(module.COMPONENTS) == COMPONENTS, "component order changed")
    return module


class CalibrationTarget:
    def __init__(self, provider, blocks):
        self.provider = provider
        self.blocks = tuple(blocks)
        self._last = None
        self._result = None

    def value_gradient(self, logs):
        logs = np.asarray(logs, dtype=np.float64)
        require(logs.shape == (5,) and np.isfinite(logs).all(), "five finite log scales required")
        if self._last is None or not np.array_equal(logs, self._last):
            with threadpool_limits(limits=1):
                value, gradient = self.provider.gaussian_error_objective(
                    logs, self.blocks, prior_medians=MEDIANS, prior_log_sd=LOG_SD
                )
            constant = len(COMPONENTS) * np.log(LOG_SD * np.sqrt(2 * np.pi))
            require(np.isfinite(value) and np.isfinite(gradient).all(), "nonfinite owner density/gradient")
            self._last = logs.copy()
            self._result = (-value - constant, -np.asarray(gradient))
        return self._result[0], self._result[1].copy()

    def likelihood_value_gradient(self, logs):
        value, gradient = self.value_gradient(logs)
        standardized = (np.asarray(logs) - np.log(MEDIANS)) / LOG_SD
        log_prior = -0.5 * np.dot(standardized, standardized) - 5 * np.log(LOG_SD * np.sqrt(2 * np.pi))
        prior_gradient = -standardized / LOG_SD
        return value - log_prior, gradient - prior_gradient


def prepare_target(directory):
    plan, rows, original = load_prepared(directory)
    provider = load_provider(directory)
    blocks = []
    for cycle in sorted(COUNTS):
        selected = [row for row in rows if row["cycle"] == cycle]
        coords = provider.PollCoordinates(
            tuple(row["observation_id"] for row in selected),
            tuple(row["race_id"] for row in selected),
            tuple(row["cycle"] for row in selected),
            tuple(row["pollster_key"] for row in selected),
            np.array([float(row["days_to_election"]) for row in selected]),
        )
        residuals = logit([float(row["dem_share"]) for row in selected]) - logit(
            [float(row["actual_dem_share"]) for row in selected]
        )
        blocks.append((residuals, coords.kernels()))
    return plan, CalibrationTarget(provider, blocks), original


def likelihood_op(target):
    import pytensor.tensor as pt
    from pytensor.graph.op import Op

    class Gradient(Op):
        itypes = [pt.dvector]
        otypes = [pt.dvector]

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(target.likelihood_value_gradient(inputs[0])[1], dtype=np.float64)

    class LogLikelihood(Op):
        itypes = [pt.dvector]
        otypes = [pt.dscalar]

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(target.likelihood_value_gradient(inputs[0])[0], dtype=np.float64)

        def grad(self, inputs, output_grads):
            return [output_grads[0] * Gradient()(inputs[0])]

    return LogLikelihood()


def build_model(target):
    import pymc as pm

    with pm.Model(coords={"component": list(COMPONENTS)}) as model:
        logs = pm.Normal("log_scales", mu=np.log(MEDIANS), sigma=LOG_SD, dims="component")
        pm.Deterministic("standard_deviations", pm.math.exp(logs), dims="component")
        pm.Potential("historical_calibration_likelihood", likelihood_op(target)(logs))
    return model


def verify_compiled_density(model, target, original):
    """Evaluate actual data at supplied points only; never run a sampler."""
    logp, dlogp = model.compile_logp(), model.compile_dlogp()
    mode = np.log([original["standard_deviations"][key] for key in COMPONENTS])
    records = []
    for offset in (np.zeros(5), np.array([0.11, -0.14, 0.07, -0.08, 0.13])):
        point = mode + offset
        expected, gradient = target.value_gradient(point)
        compiled = float(logp({"log_scales": point}))
        compiled_gradient = dlogp({"log_scales": point})
        require(np.isclose(compiled, expected, rtol=0, atol=1e-8), "compiled target density differs")
        require(np.allclose(compiled_gradient, gradient, rtol=1e-9, atol=1e-8), "compiled target gradient differs")
        records.append(
            dict(
                log_scales=point.tolist(),
                log_density=expected,
                density_difference=compiled - expected,
                gradient_difference_max=float(np.max(np.abs(compiled_gradient - gradient))),
            )
        )
    return dict(status="supplied_point_density_gradient_passed", mcmc_launched=False, points=records)


def diagnostics(idata):
    import arviz as az

    result = {}
    for method, label in (("rank", "rank_rhat"),):
        result[label] = np.asarray(az.rhat(idata, var_names=["log_scales"], method=method)["log_scales"]).tolist()
    for method in ("bulk", "tail"):
        result[method + "_ess"] = np.asarray(
            az.ess(idata, var_names=["log_scales"], method=method)["log_scales"]
        ).tolist()
    result["ebfmi"] = np.asarray(az.bfmi(idata)).tolist()
    result["divergences"] = int(idata.sample_stats.diverging.sum())
    result["depth_saturations"] = int((idata.sample_stats.tree_depth >= SAMPLER["max_treedepth"]).sum())
    finite = all(np.isfinite(result[key]).all() for key in ("rank_rhat", "bulk_ess", "tail_ess", "ebfmi"))
    result["checks"] = dict(
        finite=bool(finite),
        rank_rhat=bool(np.max(result["rank_rhat"]) <= GATES["max_rank_rhat"]),
        bulk_ess=bool(np.min(result["bulk_ess"]) >= GATES["min_bulk_ess"]),
        tail_ess=bool(np.min(result["tail_ess"]) >= GATES["min_tail_ess"]),
        ebfmi=bool(np.min(result["ebfmi"]) >= GATES["min_ebfmi"]),
        divergences=result["divergences"] == 0,
        depth_saturations=result["depth_saturations"] == 0,
    )
    result["passed"] = all(result["checks"].values())
    return result


def run(prepared, output):
    """Called only after external plan/code review authorizes this explicit CLI."""
    import arviz as az
    import pymc as pm

    plan, target, original = prepare_target(prepared)
    installed = environment_packages()
    require(installed == plan["environment"]["packages"], "runtime environment differs from frozen plan")
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "attempt output must stay inside calibration-bank-v1")
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = (Path(prepared) / "plan.json").read_bytes()
    (output / "plan.json").write_bytes(plan_bytes)
    launch = dict(
        schema="M01CalibrationPosteriorLaunchV1",
        started_utc=now(),
        pid=os.getpid(),
        plan_sha256=sha(plan_bytes),
        sampler=SAMPLER,
        diagnostic_gates=GATES,
        python=sys.version,
        platform=platform.platform(),
        executable=sys.executable,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        actual_population_or_count_fit=False,
        promotion_authorized=False,
    )
    write_json(output / "launch.json", launch)
    try:
        model = build_model(target)
        write_json(output / "compiled-target-check.json", verify_compiled_density(model, target, original))
        # Dispersed fixed starts are not covariance draws and do not define a prior.
        offsets = np.random.default_rng(SEED).normal(0, 0.2, size=(4, 5))
        initial = np.log([original["standard_deviations"][key] for key in COMPONENTS])
        chain_banks = []
        for chain, (offset, seed) in enumerate(zip(offsets, CHAIN_SEEDS)):
            with (output / f"chain-{chain}-transitions.jsonl").open("x") as transitions:

                def preserve_transition(trace, draw):
                    # Persist the actual warmup/posterior state immediately so a
                    # later numerical failure cannot erase earlier transitions.
                    record = dict(
                        chain=chain,
                        draw_index=draw.draw_idx,
                        tuning=bool(draw.tuning),
                        log_scales=np.asarray(draw.point["log_scales"]).tolist(),
                        sampler_stats_repr=repr(draw.stats),
                    )
                    transitions.write(json.dumps(record, allow_nan=False) + "\n")
                    transitions.flush()
                    if draw.draw_idx % 100 == 0:
                        print(
                            json.dumps(
                                dict(
                                    status="sampling_calibration",
                                    chain=chain,
                                    draw_index=draw.draw_idx,
                                    tuning=bool(draw.tuning),
                                    utc=now(),
                                )
                            ),
                            flush=True,
                        )

                with model, threadpool_limits(limits=1):
                    bank = pm.sample(
                        draws=1000,
                        tune=1000,
                        chains=1,
                        cores=1,
                        random_seed=seed,
                        step=pm.NUTS(target_accept=0.9, max_treedepth=10),
                        initvals=dict(log_scales=initial + offset),
                        callback=preserve_transition,
                        return_inferencedata=True,
                        discard_tuned_samples=False,
                        idata_kwargs={"save_warmup": True},
                        progressbar=True,
                    )
            bank = bank.assign_coords(chain=[chain])
            bank.to_netcdf(output / f"chain-{chain}-raw-bank.nc")
            chain_banks.append(bank)
        idata = az.concat(*chain_banks, dim="chain")
        # Keep every calibration draw, chain axis, warmup, and sample statistic.
        idata.to_netcdf(output / "raw-bank.nc")
        np.savez_compressed(
            output / "posterior-bank.npz",
            log_scales=idata.posterior.log_scales.values,
            standard_deviations=idata.posterior.standard_deviations.values,
            components=np.asarray(COMPONENTS),
        )
        assessed = diagnostics(idata)
        write_json(output / "diagnostics.json", assessed)
        write_json(
            output / "terminal.json",
            dict(
                schema="M01CalibrationPosteriorTerminalV1",
                completed_utc=now(),
                status="calibration_diagnostics_passed" if assessed["passed"] else "calibration_diagnostics_failed",
                diagnostics=assessed,
                artifacts=[pin(output, p) for p in sorted(output.iterdir()) if p.is_file()],
                downstream_feedback_target="not_chosen",
                forecast_qualified=False,
                publication_authorized=False,
                no_automatic_retry=True,
            ),
        )
    except BaseException as error:
        write_json(
            output / "failure.json",
            dict(
                schema="M01CalibrationPosteriorFailureV1",
                failed_utc=now(),
                error=repr(error),
                traceback=traceback.format_exc(),
                raw_partial_files_preserved=True,
                no_automatic_retry=True,
                forecast_qualified=False,
                publication_authorized=False,
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared-v1")
    parser.add_argument("--run", type=Path, help="explicit reviewed attempt directory; absent means verification only")
    args = parser.parse_args()
    if args.run:
        run(args.prepared, args.run)
    else:
        plan, target, original = prepare_target(args.prepared)
        logs = np.log([original["standard_deviations"][key] for key in COMPONENTS])
        value, gradient = target.value_gradient(logs)
        print(
            json.dumps(
                dict(
                    status="verified_not_sampled",
                    plan=plan["schema"],
                    log_density=value,
                    gradient=gradient.tolist(),
                    mcmc_launched=False,
                )
            )
        )


if __name__ == "__main__":
    main()
