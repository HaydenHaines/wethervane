"""Check the actual aggregate PyMC density/gradient without sampling or scoring."""

import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import halfnorm, norm, t

HERE = Path(__file__).resolve().parent
PREPARATION = HERE.parents[1] / "economic-experiment-v1"
PLAN_SHA = "cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9"
EXPECTED_VERSIONS = {
    "numpy": "2.4.4", "pandas": "2.3.3", "scipy": "1.17.1",
    "pydantic": "2.12.5", "pymc": "5.28.0", "pytensor": "2.38.2", "arviz": "0.22.0",
    "h5py": "3.16.0", "h5netcdf": "1.8.1",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    plan_path = PREPARATION / "prepared-v1/plan.json"
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("Frozen preparation plan changed")
    plan = json.loads(plan_path.read_bytes())
    for pin in plan["source_pins"]:
        if digest(pin["path"]) != pin["sha256"]:
            raise ValueError("Preparation source changed: " + pin["path"])
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise ValueError("Dedicated numerical/runtime versions differ")
    # Import the authenticated adapter only after its complete recorded sources verify.
    sys.path.insert(0, str(PREPARATION))
    import prepare
    import pymc
    import arviz

    module, compiled, _, _, _ = prepare.compile_fold(2018)
    checks = []
    for arm, training in compiled.items():
        model = module.build_pymc_model(training)
        names = [v.name for v in model.value_vars]
        if names != ["beta", "tau_cycle_log__", "tau_race_log__", "cycle_z"]:
            raise ValueError("Unexpected transformed-coordinate ordering")
        natural_logp = model.compile_logp(jacobian=False)
        transformed_logp = model.compile_logp(jacobian=True)
        terms = model.compile_logp(jacobian=False, sum=False)
        transformed_gradient = model.compile_dlogp(jacobian=True)
        offsets, points = [], []
        for index, (direction, tc, tr) in enumerate(((1, 0.2, 0.3), (-1, 0.17, 0.42))):
            params = training.recipe.parameters
            beta = np.array([p.mean for p in params]) + direction * np.linspace(
                0.01, 0.06, len(training.parameter_names)
            )
            z = direction * np.linspace(-0.1, 0.1, len(training.cycles))
            point = dict(beta=beta, tau_cycle_log__=np.log(tc), tau_race_log__=np.log(tr), cycle_z=z)
            reference = module.log_density(training, beta, z, tc, tr)
            location = training.offset + training.design @ beta + tc * z[training.cycle_index]
            references = [
                norm.logpdf(beta, [p.mean for p in params], [p.sd for p in params]),
                halfnorm.logpdf(tc, scale=training.recipe.provider("cycle").scale_prior_sd),
                halfnorm.logpdf(tr, scale=training.recipe.provider("race").scale_prior_sd),
                norm.logpdf(z),
                t.logpdf(training.response, 5, loc=location, scale=tr),
            ]
            actual_terms = terms(point)
            if len(actual_terms) != len(references):
                raise ValueError("Unexpected PyMC density factor grouping")
            # PyMC's scalar normalizers differ from SciPy by about 2.5e-9 per row.
            # Check each factor, unchanged derivatives, and offset invariance at a
            # second state; do not hide that discrepancy with a loose joint bound.
            max_term_error = max(float(np.max(np.abs(a - b))) for a, b in zip(actual_terms, references))
            natural_gradient = module.gradient(training, beta, z, tc, tr)
            transform = np.ones_like(natural_gradient)
            n = len(beta)
            transform[n:n + 2] = [tc, tr]
            expected_gradient = natural_gradient * transform
            expected_gradient[n:n + 2] += 1
            offset = float(natural_logp(point)) - reference
            offsets.append(offset)
            errors = dict(
                natural_log_density=abs(offset),
                factor_max_absolute_error=max_term_error,
                jacobian_identity_error=abs(float(transformed_logp(point)) - float(natural_logp(point)) - np.log(tc) - np.log(tr)),
                transformed_gradient=float(np.max(np.abs(transformed_gradient(point) - expected_gradient))),
            )
            if (not all(np.isfinite(value) for value in errors.values())
                    or max_term_error > 5e-9 or errors["jacobian_identity_error"] > 1e-10
                    or errors["transformed_gradient"] > 1e-7):
                raise ValueError(f"Runtime density/gradient disagreement: {arm}: {errors}")
            points.append(dict(point_index=index, errors=errors))
        if abs(offsets[0] - offsets[1]) > 1e-10:
            raise ValueError("PyMC/SciPy normalization offset varies across supplied states")
        checks.append(dict(arm=arm, law_hash=training.law_hash, points=points,
                           offset_difference=abs(offsets[0] - offsets[1])))
    # Exercise the diagnostic API expected by the existing factory without MCMC.
    fixture = arviz.from_dict(posterior={"x": np.random.default_rng(281).normal(size=(4, 1000))})
    diagnostic_probe = {
        "rank_rhat": float(arviz.rhat(fixture, method="rank")["x"]),
        "bulk_ess": float(arviz.ess(fixture, method="bulk")["x"]),
        "tail_ess": float(arviz.ess(fixture, method="tail")["x"]),
    }
    if not all(np.isfinite(value) for value in diagnostic_probe.values()):
        raise ValueError("Diagnostic API returned nonfinite fixture values")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.nc"
        fixture.to_netcdf(path)
        restored = arviz.from_netcdf(path)
        if not np.array_equal(restored.posterior["x"].values, fixture.posterior["x"].values):
            raise ValueError("NetCDF fixture round trip changed draws")
        restored.posterior.close()
    return dict(
        schema="E01AggregateRuntimeSuppliedPointCheckV1",
        status="passed",
        preparation_plan_sha256=PLAN_SHA,
        checker_sha256=digest(__file__),
        requirements_sha256=digest(HERE / "requirements.txt"),
        versions=versions,
        pymc_imported_version=pymc.__version__,
        supplied_point_checks=checks,
        diagnostic_api_fixture=diagnostic_probe,
        netcdf_fixture_round_trip=True,
        normalization_roundoff_scope="per-factor <=5e-9 and state-invariant aggregate offset at two supplied points; owner law unchanged",
        mcmc_launched=False,
        target_labels_scored=False,
        forecast_qualified=False,
    )


if __name__ == "__main__":
    result = check()
    with (HERE / "check-result.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, allow_nan=False))
