"""Create an exclusive frozen review bundle; never run MCMC."""

import argparse
import json
import sys
from pathlib import Path

from calibration_bank import (
    COMPONENTS,
    COUNTS,
    EXCLUDED_EVENTS,
    GATES,
    HERE,
    LOG_SD,
    MEDIANS,
    OWNER,
    SAMPLER,
    SOURCE_CUTOFF,
    TRUSTED,
    environment_packages,
    now,
    pin,
    require,
    sha,
    write_json,
)


def prepare(output):
    output = Path(output).resolve()
    require(output.is_relative_to(HERE), "prepared bundle must stay inside calibration-bank-v1")
    output.mkdir(parents=True, exist_ok=False)
    names = list(TRUSTED)
    snapshots = []
    for name in names:
        data = (OWNER / name).read_bytes()
        if name in TRUSTED:
            require(sha(data) == TRUSTED[name], "trusted owner source changed: " + name)
        relative = "snapshot-source/" + name if name.startswith("src/") else "inputs/" + Path(name).name
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        snapshots.append(dict(owner=dict(path=name, sha256=sha(data)), snapshot=pin(output, target)))
    packages = environment_packages()
    with (output / "environment-freeze.txt").open("x") as stream:
        stream.write("\n".join(packages) + "\n")
    code = [HERE / name for name in ("calibration_bank.py", "prepare.py", "test_calibration_bank.py")]
    code_pins = [pin(HERE, path) for path in code]
    for item, path in zip(code_pins, code):
        target = output / "code" / path.name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        require(sha(target.read_bytes()) == item["sha256"], "code snapshot mismatch")
    plan = dict(
        schema="M01CalibrationPosteriorPlanV1",
        created_utc=now(),
        status="prepared_not_launched",
        target_id="M01_disjoint_poll_covariance_calibration_posterior_v1",
        target_cycle=2026,
        source_cutoff_utc=SOURCE_CUTOFF,
        components=list(COMPONENTS),
        training_cycles=sorted(COUNTS),
        training_observations=1483,
        training_races=105,
        per_cycle=COUNTS,
        excluded_event_keys=sorted(EXCLUDED_EVENTS),
        residual="logit(dem_share)-logit(actual_dem_share)",
        covariance="unchanged PollCoordinates.kernels and gaussian_error_objective; one block per election cycle",
        prior=dict(
            coordinates="log_standard_deviations",
            family="independent_untruncated_normal",
            medians=MEDIANS.tolist(),
            log_sd=LOG_SD,
            optimizer_bounds_used=False,
        ),
        density="product of normalized Gaussian residual blocks times independent normalized Normal log-SD priors",
        density_normalizer="owner objective plus 5*log(0.7*sqrt(2*pi)); no SD-coordinate Jacobian in log coordinates",
        sampler=SAMPLER,
        diagnostic_gates=GATES,
        initial_states=(
            "original optimizer mode + fixed seeded Normal(0,0.2) offsets in log coordinates; not prior bounds"
        ),
        source_qualifications=dict(
            availability_basis="field_end_date_proxy_only_NOT_verified_publication",
            row_count=1483,
            publication_date_verified=False,
            historical_version_verified=False,
            strict_asof_qualified=False,
            earliest_historical_target_claim=None,
            qualification=(
                "Retrospective 2026 calibration using outcomes through 2020; not a time-safe historical score"
            ),
        ),
        limitations=[
            "New explicit prior target; original optimizer regularizer alone did not define posterior uncertainty.",
            "Event-disjoint exclusions do not remove same-cycle cross-state dependence on downstream data.",
            (
                "No population/count likelihood, downstream feedback target, mixture choice, score, forecast, "
                "or activation."
            ),
            "The empirical bank by itself chooses neither marginal Bayes feedback nor calibration-cut feedback.",
            "Untruncated real log scales: no optimizer bound reused; numerical failure preserves failed attempt.",
            "Finite calibration bank has Monte Carlo error and is not a conjugate approximation or a Hessian bank.",
        ],
        snapshots=snapshots,
        code_pins=code_pins,
        environment=dict(
            executable=sys.executable,
            python=sys.version,
            packages=packages,
            freeze=pin(output, output / "environment-freeze.txt"),
        ),
        output_policy="exclusive attempt directories; raw chain banks/warmup/stats retained even when diagnostics fail",
        automatic_retry=False,
        actual_population_or_count_fit=False,
        downstream_feedback_target="not_chosen",
        forecast_qualified=False,
        publication_authorized=False,
    )
    write_json(output / "plan.json", plan)
    print(json.dumps(dict(status="prepared_not_launched", plan=pin(output, output / "plan.json"), mcmc_launched=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared-v1")
    prepare(parser.parse_args().output)
