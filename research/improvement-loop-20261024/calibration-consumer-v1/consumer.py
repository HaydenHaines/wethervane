"""Actual M01 bank consumption at supplied means, explicitly MARGINAL BAYES.

Not a calibration cut, downstream posterior, forecast, active provider or sampler.
Caller owns calibration/source reuse and complete within-population normalization.
One joint five-scale row controls the COMPLETE six-poll vector; no added MOE term.
"""

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
TARGET = "M01_marginal_bayes_complete_poll_vector_likelihood_v1"
COMPONENTS = ("independent", "cycle", "race", "house", "movement")
REVIEW_SHA = "3a07125bed8944a7c9582a9c50d484cec154d1a3f76fa9c716eea70bf9d43b54"
SNAPSHOT_MANIFEST_SHA = "0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8"
SOURCE_CUTOFF = "2026-10-20T16:21:44Z"
FORECAST_CUTOFF = "2026-10-20T22:13:33Z"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def immutable(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "explicit UTC cutoff required")
    return parsed


def read_pinned(root, name, digest):
    require(Path(name).name == name, "unsafe input basename")
    data = (Path(root) / "inputs" / name).read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "input digest mismatch: " + name)
    return data


@dataclass(frozen=True)
class JointBank:
    log_scales: np.ndarray
    standard_deviations: np.ndarray
    weights: np.ndarray
    target_label: str
    source_cutoff_utc: str
    forecast_cutoff_utc: str

    @property
    def joint_rows(self):
        return self.standard_deviations.reshape(4000, 5)


def load_bank(root, *, expected_review_sha256, target_cycle, forecast_cutoff_utc, target_label):
    require(target_label == TARGET, "explicit marginal-Bayes likelihood target required; not calibration cut")
    require(type(target_cycle) is int and target_cycle == 2026, "bank forbidden for historical replay or other cycles")
    require(
        utc(forecast_cutoff_utc) >= utc(SOURCE_CUTOFF) and utc(forecast_cutoff_utc).year == 2026,
        "forecast predates source cutoff or target year mismatch",
    )
    review = json.loads(read_pinned(root, "independent-review.json", expected_review_sha256))
    require(
        review.get("status") == "independent_terminal_bank_review_passed"
        and review.get("diagnostic_gates_passed") is True
        and review.get("all_terminal_artifact_hashes_match") is True,
        "independent review did not pass",
    )
    require(
        review["scientific_scope"]["training_cutoff"] == SOURCE_CUTOFF
        and review["scientific_scope"]["not_forecast_qualified"] is True,
        "review scope mismatch",
    )
    require(
        review["warmup_discarded_for_posterior_bank"] is True
        and review["joint_bank"]["shape"] == [4, 1000, 5]
        and tuple(review["joint_bank"]["components"]) == COMPONENTS,
        "reviewed joint axes mismatch",
    )
    plan = json.loads(read_pinned(root, "plan.json", review["plan_sha256"]))
    terminal = json.loads(read_pinned(root, "terminal.json", review["terminal_sha256"]))
    require(
        plan["schema"] == "M01CalibrationPosteriorPlanV1"
        and plan["target_id"] == "M01_disjoint_poll_covariance_calibration_posterior_v1"
        and plan["target_cycle"] == 2026
        and plan["source_cutoff_utc"] == SOURCE_CUTOFF,
        "plan target or source cutoff mismatch",
    )
    require(
        tuple(plan["components"]) == COMPONENTS and plan["sampler"]["chains"] == 4 and plan["sampler"]["draws"] == 1000,
        "joint axes mismatch",
    )
    require(
        plan["forecast_qualified"] is False
        and plan["publication_authorized"] is False
        and plan["downstream_feedback_target"] == "not_chosen",
        "bank itself cannot select downstream target",
    )
    require(
        terminal["schema"] == "M01CalibrationPosteriorTerminalV1"
        and terminal["status"] == "calibration_diagnostics_passed"
        and terminal["diagnostics"]["passed"] is True
        and all(value is True for value in terminal["diagnostics"]["checks"].values()),
        "terminal numerical gates failed",
    )
    require(
        terminal["forecast_qualified"] is False and terminal["publication_authorized"] is False,
        "unexpected qualification",
    )
    pins = {item["path"]: item["sha256"] for item in terminal["artifacts"]}
    require(
        pins["plan.json"] == review["plan_sha256"] and pins["posterior-bank.npz"] == review["posterior_bank_sha256"],
        "terminal/plan/bank binding mismatch",
    )
    payload = read_pinned(root, "posterior-bank.npz", review["posterior_bank_sha256"])
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(set(archive.files) == {"log_scales", "standard_deviations", "components"}, "unknown bank arrays")
        logs = archive["log_scales"]
        scales = archive["standard_deviations"]
        require(tuple(archive["components"].tolist()) == COMPONENTS, "component order mismatch")
        require(
            logs.shape == scales.shape == (4, 1000, 5)
            and np.isfinite(logs).all()
            and np.isfinite(scales).all()
            and (scales > 0).all(),
            "invalid complete joint bank",
        )
        require(np.array_equal(np.exp(logs), scales), "log/SD coordinate mismatch")
    return JointBank(
        immutable(logs),
        immutable(scales),
        immutable(np.full(4000, 1 / 4000)),
        target_label,
        SOURCE_CUTOFF,
        forecast_cutoff_utc,
    )


def verify_snapshots(root):
    root = Path(root)
    manifest_bytes = (root / "input-manifest.json").read_bytes()
    require(
        hashlib.sha256(manifest_bytes).hexdigest() == SNAPSHOT_MANIFEST_SHA,
        "reviewed snapshot manifest digest mismatch",
    )
    manifest = json.loads(manifest_bytes)
    require(manifest["schema"] == "M01ConsumerSnapshotV1", "unknown snapshot manifest")
    for pin in manifest["files"]:
        path = (root / pin["file"]).resolve()
        require(path.is_relative_to(root.resolve()), "snapshot path escape")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], "snapshot digest mismatch")
    return manifest


def actual_inputs(root):
    root = Path(root)
    verify_snapshots(root)
    sys.path.insert(0, str(root / "snapshot-source"))
    # Exact copied owner implementation; no locally rewritten covariance kernels.
    import src.forecasting.poll_covariance_v1 as loaded_provider
    from src.forecasting.poll_covariance_v1 import COMPONENTS as owner_components
    from src.forecasting.poll_covariance_v1 import PollCoordinates, error_covariance

    require(
        Path(loaded_provider.__file__).resolve()
        == (root / "snapshot-source/src/forecasting/poll_covariance_v1.py").resolve(),
        "provider import must resolve to pinned local snapshot",
    )
    require(tuple(owner_components) == COMPONENTS, "owner component contract mismatch")
    owner_receipt = json.loads((root / "inputs/owner-receipt.json").read_text())
    require(
        owner_receipt["schema"] == "DisjointTXPollCalibrationSuccessorV1"
        and owner_receipt["status"] == "completed_event_disjoint_calibration",
        "owner component receipt mismatch",
    )
    for key, name in [
        ("component", "poll-component.json"),
        ("calibration", "calibration.json"),
        ("target_coordinates", "target-coordinates.json"),
    ]:
        require(
            hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() == owner_receipt[key]["sha256"],
            "owner input binding mismatch",
        )
    coordinates = json.loads((root / "inputs/target-coordinates.json").read_text())
    component = json.loads((root / "inputs/poll-component.json").read_text())
    calibration = json.loads((root / "inputs/calibration.json").read_text())
    require(
        len(coordinates["observation_ids"]) == 6
        and set(coordinates["cycles"]) == {"2026"}
        and set(coordinates["race_ids"]) == {"2026:TX:senate:regular"},
        "not actual six-poll TX2026 vector",
    )
    require(
        [row["observation_id"] for row in component["rows"]]
        == [row["model_observation_id"] for row in coordinates["source_map"]],
        "poll row ordering mismatch",
    )
    require(
        [row["original_observation_id"] for row in coordinates["source_map"]] == coordinates["observation_ids"],
        "original source ordering mismatch",
    )
    coordinate = PollCoordinates(
        *(tuple(coordinates[key]) for key in ("observation_ids", "race_ids", "cycles", "pollsters")),
        np.array(coordinates["days_to_election"]),
    )
    fixed = error_covariance(coordinate, calibration["standard_deviations"])
    require(np.allclose(fixed, component["covariance"], rtol=0, atol=1e-15), "fixed covariance not reproduced")
    return coordinate, np.array([row["logit_value"] for row in component["rows"]]), fixed, error_covariance


def make_mixture(bank, coordinates, error_covariance):
    from poll_scale_mixture import GaussianScaleMixture

    require(bank.target_label == TARGET and bank.joint_rows.shape == (4000, 5), "complete marginal bank required")
    require(
        set(coordinates.cycles) == {"2026"}
        and set(coordinates.race_ids) == {"2026:TX:senate:regular"}
        and len(coordinates.observation_ids) == 6,
        "bank restricted to supplied TX2026 poll coordinates",
    )
    covariances = np.array([error_covariance(coordinates, dict(zip(COMPONENTS, row))) for row in bank.joint_rows])
    return GaussianScaleMixture(covariances, bank.weights)


def independent_check(covariances, y, mean, weights):
    terms = np.array([multivariate_normal.logpdf(y, mean=mean, cov=cov) for cov in covariances]) + np.log(weights)
    log_density = logsumexp(terms)
    responsibilities = np.exp(terms - log_density)
    gradients = np.array([np.linalg.solve(cov, y - mean) for cov in covariances])
    return float(log_density), np.einsum("s,si->i", responsibilities, gradients)


def main():
    from poll_scale_mixture import GaussianScaleMixture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--target-label", required=True, choices=[TARGET])
    parser.add_argument("--target-cycle", required=True, type=int, choices=[2026])
    parser.add_argument("--forecast-cutoff-utc", required=True)
    args = parser.parse_args()
    bank = load_bank(
        HERE,
        expected_review_sha256=args.expected_review_sha256,
        target_cycle=args.target_cycle,
        forecast_cutoff_utc=args.forecast_cutoff_utc,
        target_label=args.target_label,
    )
    coordinates, y, fixed, error_covariance = actual_inputs(HERE)
    model = make_mixture(bank, coordinates, error_covariance)
    point = GaussianScaleMixture([fixed], [1])
    evaluations = []
    for name, mean in [("observed_logits", y.copy()), ("zeros", np.zeros(6)), ("common_offset", np.full(6, 0.1))]:
        oracle, gradient = independent_check(model.covariances, y, mean, bank.weights)
        fixed_oracle, fixed_gradient = independent_check(np.array([fixed]), y, mean, np.ones(1))
        chains = []
        for chain in range(4):
            selection = slice(chain * 1000, (chain + 1) * 1000)
            # Diagnostic ONLY: every full chain, all 1000 joint rows, no ranking or selection.
            d, g = independent_check(model.covariances[selection], y, mean, np.full(1000, 1 / 1000))
            chains.append(dict(chain=chain, log_density=d, gradient_mean=g.tolist()))
        evaluations.append(
            dict(
                mean_name=name,
                mean=mean.tolist(),
                log_density=model.logpdf(y, mean),
                gradient_mean=model.gradient_mean(y, mean).tolist(),
                scipy_log_density=oracle,
                log_density_error=model.logpdf(y, mean) - oracle,
                gradient_max_error=float(np.max(np.abs(model.gradient_mean(y, mean) - gradient))),
                fixed_log_density=point.logpdf(y, mean),
                fixed_oracle_error=point.logpdf(y, mean) - fixed_oracle,
                fixed_gradient_max_error=float(np.max(np.abs(point.gradient_mean(y, mean) - fixed_gradient))),
                chain_sensitivity=chains,
                chain_log_density_range=float(np.ptp([row["log_density"] for row in chains])),
            )
        )
    flat = bank.joint_rows
    report = dict(
        schema="M01ActualBankSuppliedMeanCheckV1",
        target_label=TARGET,
        source_cutoff_utc=SOURCE_CUTOFF,
        forecast_cutoff_utc=bank.forecast_cutoff_utc,
        target_cycle=2026,
        bank_shape=list(bank.standard_deviations.shape),
        joint_rows=4000,
        weights="equal_1_over_4000",
        components=list(COMPONENTS),
        scale_quantiles={
            name: dict(zip(["p05", "p50", "p95"], np.quantile(flat[:, i], [0.05, 0.5, 0.95]).tolist()))
            for i, name in enumerate(COMPONENTS)
        },
        scale_correlation=np.corrcoef(flat, rowvar=False).tolist(),
        log_scale_correlation=np.corrcoef(bank.log_scales.reshape(4000, 5), rowvar=False).tolist(),
        evaluations=evaluations,
        extra_sampling_or_MOE_term=False,
        calibration_cut=False,
        downstream_posterior_fitted=False,
        forecast_qualified=False,
        public_activation=False,
    )
    (HERE / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import scipy

    pins = [HERE / "input-manifest.json", HERE / "consumer.py", HERE / "test_consumer.py", HERE / "report.json"]
    receipt = dict(
        schema="M01ConsumerCheckReceiptV1",
        expected_review_sha256=args.expected_review_sha256,
        source_cutoff_utc=SOURCE_CUTOFF,
        target_label=TARGET,
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        artifacts=[
            dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            for path in pins
        ],
        mcmc_launched=False,
        calibration_cut=False,
        forecast_qualified=False,
    )
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                joint_rows=4000,
                max_log_density_error=max(abs(row["log_density_error"]) for row in evaluations),
                max_gradient_error=max(row["gradient_max_error"] for row in evaluations),
                chain_log_density_ranges=[row["chain_log_density_range"] for row in evaluations],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
