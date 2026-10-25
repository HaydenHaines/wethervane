"""Independent, read-only replay checks for the M01 posterior bank.

Uses ArviZ for established rank-normalized R-hat/ESS calculations. Does not
import or execute the owner's sampler, and never modifies the run directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np


HERE = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab")
ROOT = LAB / "research/improvement-loop-20261024/calibration-bank-v1"
PREPARED = ROOT / "prepared-v1"
RUN = ROOT / "attempt-v1"
COMPONENTS = ["independent", "cycle", "race", "house", "movement"]
EXPECTED_CHAINS = 4
EXPECTED_WARMUP = 1000
EXPECTED_POSTERIOR = 1000
MAX_DEPTH = 10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_pin(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing pinned file: {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch: {path}")


def verify_bank_arrays(
    log_scales: np.ndarray,
    standard_deviations: np.ndarray,
    components: np.ndarray,
    expected_log_scales: np.ndarray | None = None,
) -> dict:
    """Validate a joint (chain, draw, component) bank and its exp transform."""
    logs = np.asarray(log_scales, dtype=np.float64)
    scales = np.asarray(standard_deviations, dtype=np.float64)
    names = [str(item) for item in np.asarray(components).tolist()]
    require(logs.ndim == 3, "log_scales must have chain/draw/component axes")
    require(logs.shape == (EXPECTED_CHAINS, EXPECTED_POSTERIOR, len(COMPONENTS)), "wrong posterior bank shape")
    require(scales.shape == logs.shape, "standard_deviations shape differs from log_scales")
    require(names == COMPONENTS, "component ordering differs from frozen plan")
    require(np.isfinite(logs).all(), "nonfinite log scales")
    require(np.isfinite(scales).all() and (scales > 0).all(), "invalid standard deviations")
    require(np.allclose(scales, np.exp(logs), rtol=2e-14, atol=0.0), "standard deviations are not exp(log_scales)")
    if expected_log_scales is not None:
        ref = np.asarray(expected_log_scales, dtype=np.float64)
        require(ref.shape == logs.shape, "posterior-bank shape differs from raw NetCDF draws")
        require(np.array_equal(logs, ref), "posterior-bank rows/components differ from raw NetCDF joint draws")
    corr = np.corrcoef(logs.reshape(-1, logs.shape[-1]), rowvar=False)
    return {"shape": list(logs.shape), "components": names, "log_scale_correlation": corr.tolist()}


def _group_variable(idata, group: str, name: str) -> np.ndarray:
    dataset = getattr(idata, group)
    require(name in dataset, f"{group} missing {name}")
    return np.asarray(dataset[name].values)


def _verify_group_dimensions(idata, *, group: str, chain_count: int, draw_count: int) -> None:
    require(group in idata.groups(), f"NetCDF missing {group} group")
    dataset = getattr(idata, group)
    require("chain" in dataset.dims and "draw" in dataset.dims, f"{group} lacks chain/draw axes")
    require(dataset.sizes["chain"] == chain_count, f"{group} chain count mismatch")
    require(dataset.sizes["draw"] == draw_count, f"{group} draw count mismatch")


def recompute_diagnostics(idata) -> dict:
    """Recompute sampler diagnostics from retained post-warmup chain draws."""
    _verify_group_dimensions(idata, group="posterior", chain_count=EXPECTED_CHAINS, draw_count=EXPECTED_POSTERIOR)
    _verify_group_dimensions(idata, group="sample_stats", chain_count=EXPECTED_CHAINS, draw_count=EXPECTED_POSTERIOR)
    require("log_scales" in idata.posterior, "posterior lacks log_scales")
    require(list(idata.posterior.coords["component"].values.astype(str)) == COMPONENTS, "NetCDF component order mismatch")

    rank_rhat = np.asarray(az.rhat(idata, var_names=["log_scales"], method="rank")["log_scales"].values)
    bulk_ess = np.asarray(az.ess(idata, var_names=["log_scales"], method="bulk")["log_scales"].values)
    tail_ess = np.asarray(az.ess(idata, var_names=["log_scales"], method="tail")["log_scales"].values)
    sample_stats = idata.sample_stats
    require("diverging" in sample_stats and "tree_depth" in sample_stats and "energy" in sample_stats,
            "sample_stats missing divergence/tree depth/energy")
    divergences = int(np.asarray(sample_stats["diverging"].values).sum())
    depths = np.asarray(sample_stats["tree_depth"].values)
    depth_saturations = int(np.count_nonzero(depths >= MAX_DEPTH))
    energy = np.asarray(sample_stats["energy"].values, dtype=np.float64)
    ebfmi = np.asarray(az.bfmi(idata), dtype=np.float64)

    finite = bool(np.isfinite(rank_rhat).all() and np.isfinite(bulk_ess).all() and np.isfinite(tail_ess).all()
                  and np.isfinite(energy).all() and np.isfinite(ebfmi).all())
    return {
        "rank_rhat": rank_rhat.tolist(),
        "bulk_ess": bulk_ess.tolist(),
        "tail_ess": tail_ess.tolist(),
        "divergences": divergences,
        "depth_saturations": depth_saturations,
        "max_observed_tree_depth": int(depths.max()),
        "ebfmi": ebfmi.tolist(),
        "finite": finite,
    }


def verify_transition_journals(run: Path, seed: int, native_idata: list) -> dict:
    """Check callback journals against native warmup and posterior NetCDF rows."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, 0.2, size=(EXPECTED_CHAINS, 5))
    original = json.loads((PREPARED / "inputs/calibration.json").read_text())
    initial = np.log([original["standard_deviations"][key] for key in COMPONENTS])
    init_matches = []
    for chain in range(EXPECTED_CHAINS):
        path = run / f"chain-{chain}-transitions.jsonl"
        require(path.is_file(), f"missing transition journal: chain {chain}")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(records) == EXPECTED_WARMUP + EXPECTED_POSTERIOR, f"chain {chain} journal length mismatch")
        require([record.get("draw_index") for record in records] == list(range(2000)),
                f"chain {chain} draw indices incomplete or reordered")
        require(all(record.get("chain") == chain for record in records), f"chain {chain} identity mismatch")
        require(all(bool(record.get("tuning")) == (i < EXPECTED_WARMUP) for i, record in enumerate(records)),
                f"chain {chain} warmup boundary mismatch")
        first = np.asarray(records[0].get("log_scales"), dtype=np.float64)
        require(first.shape == (5,) and np.isfinite(first).all(), f"chain {chain} first transition state invalid")
        init_matches.append(bool(np.allclose(first, initial + offsets[chain], rtol=0, atol=1e-13)))
        all_states = np.asarray([record["log_scales"] for record in records], dtype=np.float64)
        warmup = np.asarray(native_idata[chain].warmup_posterior.log_scales.values[0])
        posterior = np.asarray(native_idata[chain].posterior.log_scales.values[0])
        require(np.array_equal(all_states[:EXPECTED_WARMUP], warmup),
                f"chain {chain} warmup journal differs from native NetCDF")
        require(np.array_equal(all_states[EXPECTED_WARMUP:], posterior),
                f"chain {chain} posterior journal differs from native NetCDF")
    return {"journals": 4, "records_per_chain": 2000, "warmup_records_per_chain": EXPECTED_WARMUP,
            "posterior_records_per_chain": EXPECTED_POSTERIOR,
            "all_transition_states_match_native_warmup_and_posterior_draws": True,
            "first_transition_matches_declared_seeded_init_after_transition": init_matches,
            "note": "Seeded initialization and chain seeds are source/plan-bound; first callback is post-transition and is not required to equal the init point."}


def verify_run(run: Path = RUN) -> dict:
    run = Path(run).resolve()
    frozen_plan_path = PREPARED / "plan.json"
    frozen_plan = json.loads(frozen_plan_path.read_text())
    plan_copy = run / "plan.json"
    require(plan_copy.is_file() and plan_copy.read_bytes() == frozen_plan_path.read_bytes(), "run plan differs from frozen plan")
    launch = json.loads((run / "launch.json").read_text())
    require(launch["sampler"] == frozen_plan["sampler"], "launch sampler differs from frozen plan")
    require(launch["diagnostic_gates"] == frozen_plan["diagnostic_gates"], "launch gates differ from frozen plan")
    require(launch["plan_sha256"] == sha256(plan_copy), "launch plan hash mismatch")
    require(frozen_plan["sampler"]["chain_seeds"] == [2678211351, 1767967282, 2782219441, 2794758142],
            "declared per-chain seeds differ")
    for item in frozen_plan["code_pins"]:
        for code_path in (ROOT / item["path"], PREPARED / "code" / item["path"]):
            verify_pin(code_path, item["sha256"])

    terminal_path = run / "terminal.json"
    require(terminal_path.is_file(), "terminal receipt absent; partial chains are not completion")
    terminal = json.loads(terminal_path.read_text())
    require(terminal.get("schema") == "M01CalibrationPosteriorTerminalV1", "unexpected terminal schema")
    require(terminal.get("status") in ("calibration_diagnostics_passed", "calibration_diagnostics_failed"),
            "terminal does not declare a completed diagnostic status")
    require(terminal.get("forecast_qualified") is False and terminal.get("publication_authorized") is False,
            "terminal overstates scientific/product scope")
    require(terminal.get("downstream_feedback_target") == "not_chosen", "terminal changes unchosen feedback target")
    artifact_entries = terminal.get("artifacts", [])
    artifact_paths = [entry["path"] for entry in artifact_entries]
    require(len(artifact_paths) == len(set(artifact_paths)), "duplicate terminal artifact paths")
    terminal_files = {entry["path"]: entry["sha256"] for entry in artifact_entries}
    require("terminal.json" not in terminal_files, "terminal self-pin must not be assumed")
    required_artifacts = {
        "plan.json", "launch.json", "compiled-target-check.json", "diagnostics.json",
        "posterior-bank.npz", "raw-bank.nc",
        *(f"chain-{chain}-raw-bank.nc" for chain in range(EXPECTED_CHAINS)),
        *(f"chain-{chain}-transitions.jsonl" for chain in range(EXPECTED_CHAINS)),
    }
    require(required_artifacts <= set(terminal_files),
            "terminal artifact pins omit a required bank, receipt, diagnostic, or journal")
    for relative, digest in terminal_files.items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run), "terminal pin escapes attempt directory")
        verify_pin(path, digest)

    # Independent files must exist for all four chains. A combined file alone
    # cannot conceal a missing native chain receipt.
    for chain in range(EXPECTED_CHAINS):
        require((run / f"chain-{chain}-raw-bank.nc").is_file(), f"missing native chain {chain} NetCDF")
    combined_path = run / "raw-bank.nc"
    require(combined_path.is_file(), "combined raw NetCDF absent")
    idata = az.from_netcdf(combined_path)
    for group, count in (("posterior", EXPECTED_POSTERIOR), ("sample_stats", EXPECTED_POSTERIOR),
                         ("warmup_posterior", EXPECTED_WARMUP), ("warmup_sample_stats", EXPECTED_WARMUP)):
        _verify_group_dimensions(idata, group=group, chain_count=EXPECTED_CHAINS, draw_count=count)

    # Verify that the separately preserved native files assemble into exactly
    # the retained combined chain/draw arrays, including warmup and statistics.
    native = [az.from_netcdf(run / f"chain-{chain}-raw-bank.nc") for chain in range(EXPECTED_CHAINS)]
    for chain, bank in enumerate(native):
        _verify_group_dimensions(bank, group="posterior", chain_count=1, draw_count=EXPECTED_POSTERIOR)
        _verify_group_dimensions(bank, group="sample_stats", chain_count=1, draw_count=EXPECTED_POSTERIOR)
        _verify_group_dimensions(bank, group="warmup_posterior", chain_count=1, draw_count=EXPECTED_WARMUP)
        _verify_group_dimensions(bank, group="warmup_sample_stats", chain_count=1, draw_count=EXPECTED_WARMUP)
        require(int(bank.posterior.coords["chain"].values[0]) == chain, f"native chain file has wrong chain label: {chain}")
    for group in ("posterior", "sample_stats", "warmup_posterior", "warmup_sample_stats"):
        for name in getattr(idata, group).data_vars:
            combined = np.asarray(getattr(idata, group)[name].values)
            pieces = [np.asarray(getattr(bank, group)[name].values) for bank in native]
            require(np.array_equal(combined, np.concatenate(pieces, axis=0), equal_nan=True),
                    f"combined {group}.{name} differs from native chain files")

    logs = _group_variable(idata, "posterior", "log_scales")
    scales = _group_variable(idata, "posterior", "standard_deviations")
    bank_path = run / "posterior-bank.npz"
    require(bank_path.is_file(), "posterior bank absent")
    with np.load(bank_path, allow_pickle=False) as bank:
        bank_report = verify_bank_arrays(bank["log_scales"], bank["standard_deviations"], bank["components"], logs)
    require(np.allclose(scales, np.exp(logs), rtol=2e-14, atol=0.0),
            "NetCDF standard deviations not exp of joint log-scale draws")

    diagnostics = recompute_diagnostics(idata)
    stored = json.loads((run / "diagnostics.json").read_text())
    # Compare recomputed metrics against run's values; report both and make the
    # independent recomputation authoritative for this review.
    for key in ("rank_rhat", "bulk_ess", "tail_ess", "ebfmi"):
        require(np.allclose(diagnostics[key], stored[key], rtol=1e-12, atol=1e-12), f"stored {key} mismatch")
    for key in ("divergences", "depth_saturations"):
        require(diagnostics[key] == stored[key], f"stored {key} mismatch")

    journals = verify_transition_journals(run, int(frozen_plan["sampler"]["seed"]), native)
    gates = frozen_plan["diagnostic_gates"]
    gates_pass = bool(
        diagnostics["finite"]
        and np.max(diagnostics["rank_rhat"]) <= gates["max_rank_rhat"]
        and np.min(diagnostics["bulk_ess"]) >= gates["min_bulk_ess"]
        and np.min(diagnostics["tail_ess"]) >= gates["min_tail_ess"]
        and diagnostics["divergences"] == gates["divergences"]
        and diagnostics["depth_saturations"] == gates["depth_saturations"]
        and np.min(diagnostics["ebfmi"]) >= gates["min_ebfmi"]
    )
    require(gates_pass == (terminal["status"] == "calibration_diagnostics_passed"),
            "terminal pass/fail does not match recomputed diagnostic gates")

    return {
        "status": "independent_terminal_bank_review_passed",
        "plan_sha256": sha256(plan_copy),
        "terminal_sha256": sha256(terminal_path),
        "posterior_bank_sha256": sha256(bank_path),
        "raw_bank_sha256": sha256(combined_path),
        "reviewer_verifier_sha256": sha256(Path(__file__).resolve()),
        "terminal_status": terminal["status"],
        "netcdf_groups_checked": ["posterior", "sample_stats", "warmup_posterior", "warmup_sample_stats"],
        "warmup_discarded_for_posterior_bank": True,
        "joint_bank": bank_report,
        "recomputed_diagnostics": diagnostics,
        "diagnostic_gates_passed": gates_pass,
        "transition_journals": journals,
        "all_terminal_artifact_hashes_match": True,
        "scientific_scope": {
            "prior_and_target": "M01 explicit untruncated log-SD prior and normalized Gaussian residual likelihood",
            "training_cutoff": frozen_plan["source_cutoff_utc"],
            "availability_limit": "field-end proxy; publication dates/history unverified",
            "retrospective_only": True,
            "downstream_feedback_target": "not_chosen",
            "not_forecast_qualified": True,
        },
    }


if __name__ == "__main__":
    result = verify_run()
    out = HERE / "independent-review.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
