"""Exclusive, resumable F-only G01 execution. Import/preparation never fits.

Complete joint laws come from the unchanged owner factory. A separately supplied
runtime certificate and exact execution-plan SHA are mandatory for execution.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "gasoline-experiment-v1"
FROZEN = PREP / "prepared-v2/plan.json"
FROZEN_SHA = "ee3cdefc9b3a5f9bc98598acd47ef7db81d0393f32c874396843e587566849ef"
RUNTIME_PARENT = HERE.parent / "economic-execution-v1/runtime"
RUNTIME_CERT = RUNTIME_PARENT / "acceptance.json"
RUNTIME_REQUIREMENTS = RUNTIME_PARENT / "requirements.txt"
ORIGINAL_E01_PREPARATION_SHA = "cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
ARMS = ("H0", "F0", "G01")
EXECUTION_ORDER = "chronological_fold; all_three_prediction_seals_and_F0_masks_before_target_score_labels"
PREDICTION = dict(
    replicates_per_joint_row=5,
    new_cycle="Normal(0,tau_cycle)_shared_across_target_races",
    new_race="independent_StudentT5_scale_tau_race",
    posterior_rows="all_chains_and_draws_in_original_order",
)
PREDICTIVE_GATES = dict(
    max_rank_rhat=1.01, min_bulk_ess=1000, min_tail_ess=1000, max_win_mcse=0.005, max_subset_seat_mean_mcse=0.03
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(Path(path).read_bytes()))


def checked(item):
    path = Path(item["path"]).resolve()
    data = path.read_bytes()
    require(digest(data) == item["sha256"], "source/artifact fingerprint mismatch: " + str(path))
    return data


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_preparation():
    raw = FROZEN.read_bytes()
    require(digest(raw) == FROZEN_SHA, "frozen G01 preparation changed")
    plan = json.loads(raw)
    for index, item in enumerate(plan["source_pins"]):
        checked(item)
        snapshot = PREP / "prepared-v2/snapshots" / f"{index:02d}-{Path(item['path']).name}"
        require(digest(snapshot.read_bytes()) == item["sha256"], "preparation snapshot changed")
    for fold in plan["folds"]:
        checked(fold["features"])
        checked(fold["join_evidence"])
    return plan


def runtime_inventory(requirements):
    versions = {}
    for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", requirements.decode(), re.MULTILINE):
        require(importlib.metadata.version(name) == version, "runtime package differs: " + name)
        versions[name] = version
    require(versions.get("pymc") == "5.28.0" and versions.get("arviz") == "0.22.0", "reviewed PyMC/ArviZ pair required")
    for name in ("h5py", "h5netcdf"):
        require(name in versions, "raw NetCDF runtime dependency missing: " + name)
    return dict(python=sys.version, packages=versions)


def scientific_sources():
    names = [
        "src/forecasting/aggregate_component_factory_v1.py",
        "src/forecasting/fundamentals_v1.py",
        "src/forecasting/baseline_v5.py",
        "src/forecasting/polling_basis_v1.py",
        "src/forecasting/polling_v1.py",
        "src/forecasting/response_v5_reuse.py",
        "src/forecasting/predictive_metrics.py",
        "src/model_stack/artifacts.py",
        "src/model_stack/contracts.py",
    ]
    return [pin(OWNER / name) for name in names]


def prepare_execution(output):
    frozen = frozen_preparation()
    requirements = pin(RUNTIME_REQUIREMENTS)
    runtime = runtime_inventory(checked(requirements))
    scoring = load_module("g01_scoring_preparation", HERE / "scoring.py")
    jobs = []
    for fold in frozen["folds"]:
        for index, arm in enumerate(ARMS):
            source = fold["arms"][arm]
            recipe = json.loads(canonical(source["recipe"]))
            seed = fold["sampler"]["seed"] + index * 1000
            # Explicit disjoint chain/predictive streams; same draw budget for all arms.
            chain_seeds = [seed + chain * 100 for chain in range(4)]
            jobs.append(
                dict(
                    target=fold["target_cycle"],
                    arm=arm,
                    recipe=recipe,
                    source_binding=source,
                    sampler={**fold["sampler"], "seed": seed},
                    chain_seeds=chain_seeds,
                    prediction_seed=seed + 50000,
                    recipe_sha256=digest(canonical(recipe)),
                )
            )
    plan = dict(
        schema="G01AggregateExecutionPlanV1",
        created_utc=now(),
        preparation=pin(FROZEN),
        stage="F_only",
        scope=frozen["scope"],
        jobs=jobs,
        fit_count=12,
        inherited_runtime_acceptance=pin(RUNTIME_CERT),
        runtime_reuse_scope=(
            "Existing E01 certificate covers locked package/runtime behavior only; "
            "this new code and plan require independent review"
        ),
        source_pins=[pin(HERE / name) for name in ("runner.py", "scoring.py", "test_execution.py")]
        + scientific_sources(),
        requirements=requirements,
        runtime=runtime,
        fit_gates=frozen["fit_diagnostic_gates"],
        prediction=PREDICTION,
        predictive_gates=PREDICTIVE_GATES,
        score_policy=scoring.POLICY,
        execution_order=EXECUTION_ORDER,
        target_labels="filter cycle metadata before numeric parse; no target outcomes in fitted features or scalers",
        continuation=(
            "new exclusive directory; exact same plan/runtime only; reuse verified "
            "complete chains; never overwrite or silently retry numerical failure"
        ),
        inherited_scope="retrospective ordinary Senate D/R aggregate; no count/population cut/FP/public activation",
        promotion_authorized=False,
        fit_launched=False,
    )
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "execution preparation must stay in package")
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    for index, item in enumerate(plan["source_pins"] + [requirements]):
        with (snapshots / f"{index:02d}-{Path(item['path']).name}").open("xb") as stream:
            stream.write(checked(item))
    write(output / "plan.json", plan)
    print(json.dumps(pin(output / "plan.json")))


def validate_execution(path, expected):
    plan = json.loads(checked(dict(path=str(path), sha256=expected)))
    require(
        plan["schema"] == "G01AggregateExecutionPlanV1" and plan["fit_count"] == 12 and plan["stage"] == "F_only",
        "wrong execution scope",
    )
    require(plan["preparation"] == pin(FROZEN), "wrong frozen preparation")
    require(plan["inherited_runtime_acceptance"] == pin(RUNTIME_CERT), "wrong inherited runtime certificate")
    require(plan["requirements"] == pin(RUNTIME_REQUIREMENTS), "wrong inherited runtime requirements")
    checked(plan["inherited_runtime_acceptance"])
    frozen = frozen_preparation()
    expected_sources = [
        pin(HERE / name) for name in ("runner.py", "scoring.py", "test_execution.py")
    ] + scientific_sources()
    require(plan["source_pins"] == expected_sources, "complete exact executable source inventory required")
    for item in plan["source_pins"]:
        checked(item)
    for index, item in enumerate(plan["source_pins"] + [plan["requirements"]]):
        snapshot = Path(path).parent / "snapshots" / f"{index:02d}-{Path(item['path']).name}"
        require(digest(snapshot.read_bytes()) == item["sha256"], "execution source snapshot changed")
    require(runtime_inventory(checked(plan["requirements"])) == plan["runtime"], "execution runtime changed")
    require(plan["fit_gates"] == frozen["fit_diagnostic_gates"], "fit gates changed")
    require(
        plan["prediction"] == PREDICTION and plan["predictive_gates"] == PREDICTIVE_GATES, "prediction contract changed"
    )
    require(
        [(j["target"], j["arm"]) for j in plan["jobs"]] == [(t, a) for t in (2018, 2020, 2022, 2024) for a in ARMS],
        "all12 ordered fits required",
    )
    require(
        plan["execution_order"] == EXECUTION_ORDER,
        "prediction sealing order changed",
    )
    for fold in frozen["folds"]:
        for index, arm in enumerate(ARMS):
            job = next(j for j in plan["jobs"] if j["target"] == fold["target_cycle"] and j["arm"] == arm)
            seed = fold["sampler"]["seed"] + index * 1000
            require(
                job["sampler"] == {**fold["sampler"], "seed": seed}
                and job["chain_seeds"] == [seed + chain * 100 for chain in range(4)]
                and job["prediction_seed"] == seed + 50000
                and job["source_binding"] == fold["arms"][arm]
                and job["recipe"] == fold["arms"][arm]["recipe"],
                "job source, budget or seed changed",
            )
    producer = load_module("frozen_g01_preparation", PREP / "prepare.py")
    scoring = load_module("frozen_g01_scoring", HERE / "scoring.py")
    require(plan["score_policy"] == scoring.POLICY, "score policy changed")
    return plan, frozen, producer, scoring


def validate_runtime_acceptance(certificate):
    review = json.loads(checked(certificate))
    require(
        review["schema"] == "E01AggregateRuntimeResearchAcceptanceV1"
        and review["status"] == "accepted_for_reviewed_execution",
        "reviewed runtime acceptance required",
    )
    require(
        review["preparation_plan_sha256"] == ORIGINAL_E01_PREPARATION_SHA
        and review["execution_plan_review_still_required"] is True,
        "runtime acceptance is not execution approval",
    )
    require(
        Path(os.path.abspath(review["python"])) == Path(os.path.abspath(sys.executable))
        and Path(sys.prefix) == Path(review["python"]).parent.parent,
        "reviewed runtime executable and prefix required",
    )
    for key in ("result", "requirements", "checker", "independent_review", "initial_failure", "factor_investigation"):
        checked(review[key])
    require(review["requirements"]["path"] == str(RUNTIME_REQUIREMENTS), "wrong runtime lock")
    inventory = runtime_inventory(checked(review["requirements"]))
    require(all(inventory["packages"].get(k) == v for k, v in review["versions"].items()), "accepted versions changed")
    return review


def validate_chain(bank, job, chain):
    import numpy as np

    require(
        bank.posterior.sizes["chain"] == 1 and bank.posterior.sizes["draw"] == job["sampler"]["draws"],
        "complete retained chain required",
    )
    require(bank.warmup_posterior.sizes["draw"] == job["sampler"]["tune"], "complete warmup must be preserved")
    require(list(bank.posterior.chain.values) == [chain], "original chain identity required")
    require(
        list(bank.posterior.coefficient.values) == job["source_binding"]["parameter_names"],
        "complete ordered coefficient identity required",
    )
    require(
        list(bank.posterior.source_cycle.values) == job["source_binding"]["cycles"],
        "complete earlier-cycle identity required",
    )
    for name in ("beta", "tau_cycle", "tau_race", "cycle_z"):
        require(np.isfinite(np.asarray(bank.posterior[name])).all(), "finite raw posterior required")


def compile_job(job, fold, producer):
    module, compiled, frame, scaler, _ = producer.compile_fold(job["target"])
    for name in ARMS:
        item = compiled[name]
        expected = fold["arms"][name]
        require(
            item.law_hash == expected["law_hash"]
            and digest(item.design.tobytes()) == expected["design_sha256"]
            and digest(item.response.tobytes()) == expected["response_sha256"],
            "compiled law differs from frozen preparation",
        )
    require(scaler == fold["scaler"], "historical-only scaler changed")
    source = compiled[job["arm"]]
    require(digest(canonical(job["recipe"])) == job["recipe_sha256"], "recipe changed")
    recipe = module.Recipe.model_validate(job["recipe"])
    training = module.compile_training(recipe, frame, producer.earlier_outcomes(job["target"]), source.admission)
    require(
        training.race_ids == source.race_ids
        and training.outcome_rows_sha256 == source.outcome_rows_sha256
        and digest(training.design.tobytes()) == job["source_binding"]["design_sha256"],
        "joint training rows/design changed",
    )
    target = frame.loc[frame.cycle.eq(job["target"])].copy()
    require(target.race_id.tolist() == fold["target_race_ids"], "target support changed")
    return module, training, target


def serial(value):
    import numpy as np

    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fit_diagnostics(idata, gates):
    import arviz as az
    import numpy as np

    names = ["beta", "tau_cycle", "tau_race", "cycle_z"]

    def coordinates(dataset):
        return np.concatenate([np.asarray(dataset[name]).ravel() for name in names])

    rhat = coordinates(az.rhat(idata, var_names=names, method="rank"))
    bulk = coordinates(az.ess(idata, var_names=names, method="bulk"))
    tail = coordinates(az.ess(idata, var_names=names, method="tail"))
    # Reduce each variable separately so NaN coordinates cannot disappear in a skip-NaN broadcast reduction.
    values = dict(
        max_rank_rhat=float(np.max(rhat)),
        min_bulk_ess=float(np.min(bulk)),
        min_tail_ess=float(np.min(tail)),
        divergences=int(idata.sample_stats.diverging.sum()),
        max_depth_hits=int((idata.sample_stats.tree_depth >= 10).sum()),
    )
    finite = all(np.isfinite(v) for v in values.values())
    checks = dict(
        finite=bool(finite),
        rhat=values["max_rank_rhat"] <= gates["max_rank_rhat"],
        bulk=values["min_bulk_ess"] >= gates["min_bulk_ess"],
        tail=values["min_tail_ess"] >= gates["min_tail_ess"],
        divergences=values["divergences"] == 0,
        depth=values["max_depth_hits"] == 0,
    )
    return serial(dict(values=values, checks=checks, passed=all(checks.values())))


def fit_job(job, module, training, directory, plan_sha, previous=None, resume_receipts=None):
    import arviz as az
    import pymc as pm

    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "launch.json", dict(plan_sha256=plan_sha, job=job, started_utc=now()))
    banks = []
    references = []
    try:
        for chain, seed in enumerate(job["chain_seeds"]):
            old = previous / f"chain-{chain}-complete.json" if previous else None
            if old and old.exists():
                require(
                    resume_receipts and str(old.resolve()) in resume_receipts,
                    "externally pinned continuation receipt required",
                )
                receipt = json.loads(checked(resume_receipts[str(old.resolve())]))
                require(
                    receipt["plan_sha256"] == plan_sha
                    and receipt["seed"] == seed
                    and receipt["chain"] == chain
                    and receipt["job"] == [job["target"], job["arm"]]
                    and receipt["complete"] is True,
                    "continuation chain binding differs",
                )
                data = checked(receipt["raw_bank"])
                require(data, "empty resumed chain")
                bank = az.from_netcdf(receipt["raw_bank"]["path"])
                validate_chain(bank, job, chain)
                banks.append(bank)
                references.append(receipt["raw_bank"])
                write(directory / f"chain-{chain}-complete.json", receipt)
                continue
            chain_dir = directory / f"chain-{chain}"
            chain_dir.mkdir()
            write(chain_dir / "launch.json", dict(seed=seed, chain=chain, plan_sha256=plan_sha))
            with (chain_dir / "transitions.jsonl").open("x") as stream:

                def callback(trace, draw):
                    stream.write(
                        json.dumps(
                            serial(
                                dict(
                                    chain=chain,
                                    draw=draw.draw_idx,
                                    tuning=draw.tuning,
                                    point=draw.point,
                                    sampler_stats_repr=repr(draw.stats),
                                )
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    if draw.draw_idx % 100 == 0:
                        print(
                            json.dumps(
                                dict(
                                    target=job["target"],
                                    arm=job["arm"],
                                    chain=chain,
                                    draw=draw.draw_idx,
                                    tuning=draw.tuning,
                                    utc=now(),
                                )
                            ),
                            flush=True,
                        )

                with module.build_pymc_model(training):
                    bank = pm.sample(
                        draws=job["sampler"]["draws"],
                        tune=job["sampler"]["tune"],
                        chains=1,
                        cores=1,
                        random_seed=seed,
                        target_accept=job["sampler"]["target_accept"],
                        nuts={"max_treedepth": job["sampler"]["max_treedepth"]},
                        return_inferencedata=True,
                        discard_tuned_samples=False,
                        idata_kwargs={"save_warmup": True},
                        compute_convergence_checks=False,
                        callback=callback,
                        progressbar=False,
                        init="jitter+adapt_diag",
                        jitter_max_retries=0,
                    )
            bank = bank.assign_coords(chain=[chain])
            validate_chain(bank, job, chain)
            path = chain_dir / "raw-bank.nc"
            bank.to_netcdf(path)
            receipt = dict(
                plan_sha256=plan_sha,
                seed=seed,
                chain=chain,
                job=[job["target"], job["arm"]],
                raw_bank=pin(path),
                complete=True,
            )
            write(directory / f"chain-{chain}-complete.json", receipt)
            banks.append(bank)
            references.append(pin(path))
        combined = az.concat(*banks, dim="chain")
        return combined, references
    except BaseException as error:
        write(
            directory / "failure.json",
            dict(
                status="incomplete_preserved_attempt",
                error=repr(error),
                traceback=traceback.format_exc(),
                completed_chains=references,
                automatic_retry=False,
                forecast_qualified=False,
            ),
        )
        raise


def predictive_arrays(training, target, idata, module, seed, replicates=5):
    import numpy as np
    from scipy.special import expit

    posterior = idata.posterior
    beta = np.asarray(posterior.beta)
    tc = np.asarray(posterior.tau_cycle)
    tr = np.asarray(posterior.tau_race)
    require(beta.ndim == 3 and beta.shape[:2] == tc.shape == tr.shape, "complete aligned joint fitted rows required")
    offset, design = module._design(training.recipe, target, training.admission)
    rng = np.random.default_rng(seed)
    chains, draws = tc.shape
    n = len(target)
    cycle_z = rng.normal(size=(chains, draws, replicates))
    race_z = rng.standard_t(5, size=(chains, draws, replicates, n))
    location = offset + beta @ design.T
    logits = location[:, :, None, :] + tc[:, :, None, None] * cycle_z[:, :, :, None] + tr[:, :, None, None] * race_z
    require(np.isfinite(logits).all(), "nonfinite genuine predictive draw")
    return dict(
        shares=expit(logits).reshape(chains, draws * replicates, n),
        logits=logits.reshape(chains, draws * replicates, n),
        new_cycle_z=cycle_z.reshape(chains, draws * replicates),
        new_race_z=race_z.reshape(chains, draws * replicates, n),
        posterior_draw=np.repeat(np.arange(draws), replicates),
        replicate=np.tile(np.arange(replicates), draws),
        race_ids=target.race_id.to_numpy(dtype=str),
    )


def prediction_diagnostics(arrays, gates, correlated_mcse):
    import arviz as az
    import numpy as np

    shares = arrays["shares"]
    wins = (shares > 0.5).astype(float)
    n = shares.shape[-1]
    values = np.concatenate((shares, wins, wins.sum(axis=2, keepdims=True)), axis=2)
    nonconstant = np.ptp(values, axis=(0, 1)) > 0
    diagnostic = correlated_mcse(values)
    require(np.isfinite(values).all() and nonconstant.any(), "finite nonconstant predictive coordinates required")
    selected = values[:, :, nonconstant]
    data = az.from_dict(posterior={"prediction": selected})
    rhat = np.asarray(az.rhat(data, method="rank")["prediction"])
    bulk = np.asarray(az.ess(data, method="bulk")["prediction"])
    tail = np.asarray(az.ess(data, method="tail")["prediction"])
    result = dict(
        max_rank_rhat=float(rhat.max()),
        min_bulk_ess=float(bulk.min()),
        min_tail_ess=float(tail.min()),
        max_win_mcse=float(diagnostic["mcse"][n : 2 * n].max()),
        subset_seat_mean_mcse=float(diagnostic["mcse"][2 * n]),
    )
    checks = dict(
        finite=all(np.isfinite(v) for v in result.values()),
        rhat=result["max_rank_rhat"] <= gates["max_rank_rhat"],
        bulk=result["min_bulk_ess"] >= gates["min_bulk_ess"],
        tail=result["min_tail_ess"] >= gates["min_tail_ess"],
        wins=result["max_win_mcse"] <= gates["max_win_mcse"],
        subtotal=result["subset_seat_mean_mcse"] <= gates["max_subset_seat_mean_mcse"],
    )
    return serial(
        dict(
            values=result,
            checks=checks,
            passed=all(checks.values()),
            unresolved_constant_win_race_ids=arrays["race_ids"][np.ptp(wins, axis=(0, 1)) == 0].tolist(),
            rare_tail_precision_resolved=bool((np.ptp(wins, axis=(0, 1)) > 0).all()),
            scope=(
                "joint-chain Monte Carlo integration and future simulation; constant "
                "empirical winner tails remain unresolved; not election-to-election "
                "uncertainty"
            ),
        )
    )


def load_labels_after_seals(fold_directory, plan_sha, target, ids, producer):
    import numpy as np
    import pandas as pd

    mask = json.loads((fold_directory / "masks.json").read_bytes())
    scorer = load_module("seal_policy", HERE / "scoring.py")
    require(mask["policy"] == scorer.POLICY, "mask policy changed")
    require(mask["race_ids"]["all"] == ids, "mask support changed")
    require(
        mask["source_prediction_seals"] == [pin(fold_directory / arm / "prediction-seal.json") for arm in ARMS],
        "mask/seal binding changed",
    )
    for arm in ARMS:
        seal = json.loads((fold_directory / arm / "prediction-seal.json").read_bytes())
        require(
            seal["plan_sha256"] == plan_sha and seal["target"] == target and seal["arm"] == arm, "wrong prediction seal"
        )
        checked(seal["predictions"])
        if arm == "F0":
            with np.load(seal["predictions"]["path"], allow_pickle=False) as bank:
                require(mask["race_ids"] == scorer.masks(ids, bank["shares"]), "common F0 masks changed")
    require((fold_directory / "masks.json").exists(), "common forecast-time masks must be frozen before labels")
    path = producer.PANEL / "labels/outcomes-SEPARATE.csv"
    producer.authenticated(path, producer.EXPECTED["outcomes"])
    metadata = pd.read_csv(path, usecols=["cycle"])
    keep = set(metadata.index[metadata.cycle.eq(target)] + 1)
    rows = pd.read_csv(
        path, usecols=["race_id", "cycle", "actual_dem_share"], skiprows=lambda line: line > 0 and line not in keep
    )
    require(
        not rows.race_id.duplicated().any() and set(ids) == set(rows.race_id), "missing or duplicate score outcomes"
    )
    result = rows.set_index("race_id").loc[ids, "actual_dem_share"].to_numpy(float)
    require(((result > 0) & (result < 1)).all(), "interior ordinary share truth required")
    return result


def seal_continuation(source, output):
    """Create a new manifest for independent review; never authorize continuation."""
    source = Path(source).resolve()
    launch = json.loads((source / "launch.json").read_bytes())
    plan, _, _, _ = validate_execution(launch["plan"]["path"], launch["plan_sha256"])
    require(launch["plan"] == pin(launch["plan"]["path"]), "continuation launch plan changed")
    complete = []
    terminals = []
    for job in plan["jobs"]:
        terminal = source / str(job["target"]) / job["arm"] / "terminal.json"
        terminals.append(dict(path=str(terminal), pin=pin(terminal) if terminal.exists() else None))
        for chain in range(4):
            receipt = source / str(job["target"]) / job["arm"] / f"chain-{chain}-complete.json"
            if receipt.exists():
                value = json.loads(receipt.read_bytes())
                require(
                    value["plan_sha256"] == launch["plan_sha256"]
                    and value["chain"] == chain
                    and value["job"] == [job["target"], job["arm"]]
                    and value["seed"] == job["chain_seeds"][chain]
                    and value["complete"] is True,
                    "continuation chain identity changed",
                )
                checked(value["raw_bank"])
                complete.append(pin(receipt))
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "continuation manifest must stay in execution package")
    write(
        output,
        dict(
            schema="G01AggregateContinuationManifestV1",
            source=str(source),
            launch=pin(source / "launch.json"),
            plan_sha256=launch["plan_sha256"],
            complete_chain_receipts=complete,
            job_terminals=terminals,
            review_required=True,
        ),
    )
    print(json.dumps(pin(output)))


def checked_continuation_terminal(path, binding):
    """Pin both presence and bytes, including absence at the review instant."""
    path = Path(path).resolve()
    require(binding["path"] == str(path), "continuation terminal identity differs")
    require(path.exists() == (binding["pin"] is not None), "continuation terminal presence changed")
    return json.loads(checked(binding["pin"])) if binding["pin"] is not None else None


def run(path, expected, output, certificate, continuation=None):
    require(certificate == pin(RUNTIME_CERT), "wrong inherited runtime certificate")
    runtime_review = validate_runtime_acceptance(certificate)
    plan, frozen, producer, scoring = validate_execution(path, expected)
    require(certificate == plan["inherited_runtime_acceptance"], "runtime certificate/plan mismatch")
    require(runtime_review["requirements"] == plan["requirements"], "runtime acceptance/plan lock differs")
    require(
        all(
            os.environ.get(name) == "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        ),
        "one BLAS thread required",
    )
    import numpy as np

    if str(OWNER) not in sys.path:
        sys.path.insert(0, str(OWNER))
    from src.forecasting.predictive_metrics import weighted_crps
    from src.forecasting.response_v5_reuse import correlated_mcse

    correlated_mcse = partial(correlated_mcse, batch_length=50 * PREDICTION["replicates_per_joint_row"])
    resume_from = None
    resume_receipts = {}
    resume_terminals = {}
    if continuation:
        manifest = json.loads(checked(continuation))
        require(
            manifest["schema"] == "G01AggregateContinuationManifestV1" and manifest["plan_sha256"] == expected,
            "continuation manifest plan differs",
        )
        resume_from = Path(manifest["source"])
        launch = json.loads(checked(manifest["launch"]))
        require(launch["plan_sha256"] == expected, "continuation plan differs")
        require(launch["runtime_certificate"] == certificate, "continuation runtime acceptance differs")
        require(
            [t["path"] for t in manifest["job_terminals"]]
            == [str(resume_from / str(j["target"]) / j["arm"] / "terminal.json") for j in plan["jobs"]],
            "complete ordered continuation terminal inventory required",
        )
        for item in manifest["job_terminals"]:
            terminal = checked_continuation_terminal(item["path"], item)
            if terminal is not None:
                require(
                    terminal["plan_sha256"] == expected
                    and terminal["numerical_passed"] is True
                    and terminal["status"] == "completed",
                    "numerical failure is not an automatic retry",
                )
            resume_terminals[item["path"]] = item
        for item in manifest["complete_chain_receipts"]:
            checked(item)
            resume_receipts[item["path"]] = item
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "attempt output must stay inside execution package")
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "launch.json",
        dict(
            plan_sha256=expected,
            plan=pin(path),
            runtime_certificate=certificate,
            continuation_manifest=continuation,
            resume_from=str(resume_from) if resume_from else None,
            started_utc=now(),
            pid=os.getpid(),
        ),
    )
    summaries = []
    try:
        for fold in frozen["folds"]:
            target_cycle = fold["target_cycle"]
            folder = output / str(target_cycle)
            folder.mkdir()
            predictions = {}
            qualified = {}
            # Only compile this fold. Later numeric labels remain unparsed until this fold is sealed/scored.
            jobs = [job for job in plan["jobs"] if job["target"] == target_cycle]
            for job in jobs:
                module, training, target = compile_job(job, fold, producer)
                job_dir = folder / job["arm"]
                prior = Path(resume_from) / str(target_cycle) / job["arm"] if resume_from else None
                if prior:
                    terminal = checked_continuation_terminal(
                        prior / "terminal.json", resume_terminals[str(prior / "terminal.json")]
                    )
                    if terminal is not None:
                        require(
                            terminal["plan_sha256"] == expected
                            and terminal["numerical_passed"] is True
                            and terminal["status"] == "completed",
                            "numerical failure is not an automatic retry",
                        )
                idata, references = fit_job(job, module, training, job_dir, expected, prior, resume_receipts)
                fit = fit_diagnostics(idata, plan["fit_gates"])
                write(job_dir / "fit-diagnostics.json", fit)
                arrays = predictive_arrays(
                    training, target, idata, module, job["prediction_seed"], PREDICTION["replicates_per_joint_row"]
                )
                predictive = prediction_diagnostics(arrays, PREDICTIVE_GATES, correlated_mcse)
                np.savez_compressed(job_dir / "predictions.npz", **arrays)
                write(job_dir / "prediction-diagnostics.json", predictive)
                passed = fit["passed"] and predictive["passed"]
                write(
                    job_dir / "prediction-seal.json",
                    dict(
                        plan_sha256=expected,
                        target=target_cycle,
                        arm=job["arm"],
                        predictions=pin(job_dir / "predictions.npz"),
                        numerical_passed=passed,
                        source_law_hash=training.law_hash,
                    ),
                )
                write(
                    job_dir / "terminal.json",
                    dict(
                        status="completed" if passed else "numerical_failed_preserved",
                        numerical_passed=passed,
                        plan_sha256=expected,
                        raw_chains=references,
                        forecast_qualified=False,
                        promotion_authorized=False,
                    ),
                )
                predictions[job["arm"]] = arrays
                qualified[job["arm"]] = passed
            ids = fold["target_race_ids"]
            mask = scoring.masks(ids, predictions["F0"]["shares"])
            write(
                folder / "masks.json",
                dict(
                    policy=plan["score_policy"],
                    race_ids=mask,
                    source_prediction_seals=[pin(folder / arm / "prediction-seal.json") for arm in ARMS],
                ),
            )
            truth = load_labels_after_seals(folder, expected, target_cycle, ids, producer)
            frequencies = (
                producer.earlier_outcomes(target_cycle)
                .groupby("cycle")
                .actual_dem_share.apply(lambda y: (y > 0.5).mean())
            )
            frequency = float(frequencies.mean())
            cycle_summary = {}
            for arm, arrays in predictions.items():
                rows = scoring.scores(arrays["shares"], truth, ids, weighted_crps)
                strata = {}
                for name, races in mask.items():
                    index = [ids.index(race) for race in races]
                    strata[name] = scoring.summarize([row for row in rows if row["race_id"] in races])
                    if index:
                        strata[name]["mcse"], unresolved = scoring.score_mcse(
                            arrays["shares"][:, :, index], truth[index], correlated_mcse
                        )
                        strata[name]["unresolved_win_tail_race_ids"] = [races[j] for j in unresolved]
                write(
                    folder / arm / "scores.json",
                    dict(
                        rows=rows,
                        strata=strata,
                        numerical_passed=qualified[arm],
                        references=dict(
                            always_half_winner_brier=0.25,
                            always_half_winner_log_score=float(np.log(2)),
                            earlier_equal_cycle_D_win_frequency=frequency,
                            earlier_frequency_brier=float(np.mean((frequency - (truth > 0.5)) ** 2)),
                            earlier_frequency_log_score=float(
                                np.mean(-np.log(np.where(truth > 0.5, frequency, 1 - frequency)))
                            ),
                        ),
                        forecast_qualified=False,
                    ),
                )
                cycle_summary[arm] = dict(strata=strata, numerical_passed=qualified[arm])
            summaries.append(dict(target=target_cycle, tier=fold["tier"], arms=cycle_summary))
        write(
            output / "summary.json",
            dict(
                cycles=summaries,
                tiers=tier_summary(summaries),
                regressions=regression_report(summaries),
                scoring_policy=plan["score_policy"],
                prospective_evidence=False,
                count_qualification=False,
                promotion_authorized=False,
            ),
        )
        write(
            output / "terminal.json",
            dict(
                status="completed_F_only_comparison",
                plan_sha256=expected,
                summary=pin(output / "summary.json"),
                all_numerical_passed=all(a["numerical_passed"] for c in summaries for a in c["arms"].values()),
                count_qualification=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write(
            output / "failure.json",
            dict(
                status="incomplete_preserved_attempt",
                error=repr(error),
                traceback=traceback.format_exc(),
                automatic_retry=False,
                count_qualification=False,
            ),
        )
        raise


def tier_summary(cycles):
    import numpy as np

    result = {}
    for tier in ("A", "B"):
        selected = [c for c in cycles if c["tier"] == tier]
        arms = {}
        for arm in ARMS:
            metrics = {
                name: float(np.mean([c["arms"][arm]["strata"]["all"][name] for c in selected]))
                for name in (
                    "share_crps",
                    "share_wis",
                    "win_brier",
                    "win_log_score",
                    "coverage80",
                    "width80",
                    "coverage95",
                    "width95",
                )
            }
            arms[arm] = metrics
        result[tier] = dict(
            equal_cycle_means=arms,
            cycles=[c["target"] for c in selected],
            G01_minus_F0={key: arms["G01"][key] - arms["F0"][key] for key in arms["F0"]},
            decision=(
                "descriptive retrospective comparison; review every "
                "per-cycle/competitive/B-tier regression; no automatic adoption"
            ),
        )
    return result


def regression_report(cycles):
    """Independent-arm numerical errors; no races-as-independent-election claim."""
    import numpy as np

    reports = []
    for cycle in cycles:
        for arm in ("G01",):
            for stratum, baseline in cycle["arms"]["F0"]["strata"].items():
                challenger = cycle["arms"][arm]["strata"][stratum]
                if baseline["n"] == 0:
                    continue
                for metric in ("share_crps", "win_brier", "win_log_score"):
                    delta = challenger[metric] - baseline[metric]
                    error = float(np.hypot(challenger["mcse"][metric], baseline["mcse"][metric]))
                    resolved = metric == "share_crps" or not (
                        challenger.get("unresolved_win_tail_race_ids") or baseline.get("unresolved_win_tail_race_ids")
                    )
                    reports.append(
                        dict(
                            target=cycle["target"],
                            tier=cycle["tier"],
                            arm=arm,
                            stratum=stratum,
                            metric=metric,
                            challenger_minus_F0=delta,
                            numerical_mcse=error,
                            loss=delta > 0,
                            tail_precision_resolved=resolved,
                            loss_beyond_two_mcse=delta > 2 * error if resolved else None,
                            practical_CRPS_aid=metric == "share_crps" and delta > max(0.0005, 0.02 * baseline[metric]),
                        )
                    )
    tiers = []
    for tier in ("A", "B"):
        for arm in ("G01",):
            for stratum in ("all", "competitive", "near_even"):
                for metric in ("share_crps", "win_brier", "win_log_score"):
                    selected = [
                        r
                        for r in reports
                        if r["tier"] == tier and r["arm"] == arm and r["stratum"] == stratum and r["metric"] == metric
                    ]
                    if len(selected) != 2:
                        tiers.append(
                            dict(
                                tier=tier,
                                arm=arm,
                                stratum=stratum,
                                metric=metric,
                                status="unavailable_missing_or_empty_cycle_mask",
                            )
                        )
                        continue
                    delta = float(np.mean([r["challenger_minus_F0"] for r in selected]))
                    error = float(np.sqrt(sum(r["numerical_mcse"] ** 2 for r in selected)) / 2)
                    resolved = all(r["tail_precision_resolved"] for r in selected)
                    tiers.append(
                        dict(
                            tier=tier,
                            arm=arm,
                            stratum=stratum,
                            metric=metric,
                            status="available",
                            equal_cycle_delta=delta,
                            numerical_mcse=error,
                            loss=delta > 0,
                            tail_precision_resolved=resolved,
                            loss_beyond_two_mcse=delta > 2 * error if resolved else None,
                        )
                    )
    return dict(
        per_cycle=reports,
        equal_cycle_tiers=tiers,
        scope=(
            "descriptive losses; independent-chain numerical error only; A/B cycles are the independent national units"
        ),
        decision=(
            "No automatic adoption. Review all B-tier, per-cycle and competitive-mask "
            "losses; G01 is one fixed-prior gasoline hypothesis, never automatically adopted."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--seal-continuation", type=Path)
    parser.add_argument("--continuation-output", type=Path)
    parser.add_argument("--continuation-manifest", type=Path)
    parser.add_argument("--continuation-manifest-sha256")
    parser.add_argument("--runtime-certificate", type=Path)
    parser.add_argument("--runtime-certificate-sha256")
    args = parser.parse_args()
    if args.seal_continuation:
        require(args.continuation_output, "exclusive continuation manifest output required")
        seal_continuation(args.seal_continuation, args.continuation_output)
        return
    require(
        bool(args.continuation_manifest) == bool(args.continuation_manifest_sha256),
        "continuation path and external SHA required together",
    )
    if args.prepare:
        prepare_execution(args.prepare)
        return
    require(args.plan and args.plan_sha256, "exact plan path and SHA required")
    if args.run:
        require(
            args.runtime_certificate and args.runtime_certificate_sha256,
            "independently pinned runtime certificate required",
        )
        run(
            args.plan,
            args.plan_sha256,
            args.run,
            dict(path=str(args.runtime_certificate), sha256=args.runtime_certificate_sha256),
            dict(path=str(args.continuation_manifest), sha256=args.continuation_manifest_sha256)
            if args.continuation_manifest
            else None,
        )
    else:
        validate_execution(args.plan, args.plan_sha256)
        print("Verified execution plan; no sampler called.")


if __name__ == "__main__":
    main()
