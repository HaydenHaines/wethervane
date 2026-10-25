"""No-fit bank compatibility: reads pinned fits, constructs laws, no scoring."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
sys.path.insert(0, str(OWNER))
from src.forecasting.polling_basis_v1 import EmpiricalBasisFundamentals  # noqa: E402
from src.forecasting.polling_basis_v1 import conditional_row_log_weights as basis_weights  # noqa: E402
from src.forecasting.polling_v1 import EmpiricalFundamentals  # noqa: E402
from src.forecasting.polling_v1 import conditional_row_log_weights as legacy_weights  # noqa: E402

PINS = {}


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    item = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    PINS[str(path)] = item
    return item


def read(path, expected=None):
    item = pin(path)
    if expected is not None:
        assert item["sha256"] == expected
    return json.loads(Path(path).read_bytes())


def checked(item):
    assert pin(item["path"])["sha256"] == item["sha256"]


def main():
    selection = read(
        LAB / "checkpoints/momentum-preflight-root-selection-v1.json",
        "5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9",
    )
    for key in ("proposal", "independent_review", "source_audit"):
        checked(selection[key])
    frozen = read(
        LAB / "economic-experiment-v1/prepared-v1/plan.json",
        "cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9",
    )
    for item in frozen["source_pins"]:
        checked(item)
    spec = importlib.util.spec_from_file_location("bank_preparation", LAB / "economic-experiment-v1/prepare.py")
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    old_dir = OWNER / ".redesign-runtime/response-v5-20261007/baseline-2022-prospective-v2/attempt"
    old_receipt = read(old_dir / "receipt.json")
    for name in ("posterior.npz", "prepared-inputs.npz"):
        assert pin(old_dir / name)["sha256"] == old_receipt["ledger"][name]["sha256"]
    with np.load(old_dir / "posterior.npz", allow_pickle=False) as old:
        old_shapes = {k: list(old[k].shape) for k in old.files}
    with np.load(old_dir / "prepared-inputs.npz", allow_pickle=False) as old:
        old_arrays = {k: old[k].copy() for k in old.files}
        old_training = dict(
            cutoff=int(old["cutoff_cycle"]),
            cycles=old["cycles"].tolist(),
            race_ids=old["race_ids"].tolist(),
            target_ids=old["target_race_ids"].tolist(),
        )
    records = []
    common_source_differences = {}
    for fold in frozen["folds"]:
        year = fold["target_cycle"]
        module, compiled, frame, scaler, evidence = producer.compile_fold(year)
        target = frame.loc[frame.cycle.eq(year)].copy()
        checked(fold["features"])
        for arm in ("H0", "F0"):
            source = compiled[arm]
            assert source.law_hash == fold["arms"][arm]["law_hash"]
            assert (source.cycles < year).all()
            if year == 2022 and arm == "H0":
                index = [list(source.race_ids).index(r) for r in old_training["race_ids"]]
                for label, new, old in (
                    ("offset", source.offset[index], old_arrays["offset"]),
                    ("design", source.design[index], old_arrays["design"]),
                    ("response_logit", source.response[index], old_arrays["y"] + old_arrays["offset"]),
                ):
                    difference = np.abs(new - old)
                    common_source_differences[label] = dict(
                        max_absolute_difference=float(difference.max()),
                        rows_different_beyond_1e_minus12=int(
                            (difference.reshape(len(index), -1).max(axis=1) > 1e-12).sum()
                        ),
                    )
            directory = LAB / f"economic-execution-v1/attempt-v1/{year}/{arm}"
            terminal = read(directory / "terminal.json")
            assert terminal["numerical_passed"] is True
            beta_rows, cycle_rows, race_rows = [], [], []
            for chain, bank_pin in enumerate(terminal["raw_chains"]):
                checked(bank_pin)
                receipt = read(directory / f"chain-{chain}-complete.json")
                assert receipt["raw_bank"] == bank_pin and receipt["chain"] == chain
                bank = az.from_netcdf(bank_pin["path"])
                post = bank.posterior
                assert post.chain.values.tolist() == [chain]
                assert post.sizes["draw"] == 4000 and bank.warmup_posterior.sizes["draw"] == 1500
                assert post.coefficient.values.tolist() == list(source.parameter_names)
                assert post.source_cycle.values.tolist() == source.cycles.tolist()
                beta_rows.append(np.asarray(post.beta)[0])
                cycle_rows.append(np.asarray(post.tau_cycle)[0])
                race_rows.append(np.asarray(post.tau_race)[0])
                bank.close()
            beta, tc, tr = np.concatenate(beta_rows), np.concatenate(cycle_rows), np.concatenate(race_rows)
            offset, design = module._design(source.recipe, target, source.admission)
            ids = tuple(target.race_id)
            n, q = design.shape
            generic = EmpiricalBasisFundamentals(
                offset, design, beta, tc, tr, ids, source.law_hash, tuple(source.parameter_names)
            )
            obs = np.arange(min(3, n))
            theta = np.linspace(-0.2, 0.3, len(obs))
            precision = np.linspace(0.8, 1.2, len(obs))
            cycle = 0.03
            direct = (
                -np.log(tc)
                - len(obs) * np.log(tr)
                - 0.5 * (cycle / tc) ** 2
                - 0.5 * ((theta - offset[obs] - cycle - beta @ design[obs].T) ** 2 * precision).sum(axis=1) / tr**2
            )
            np.testing.assert_allclose(
                basis_weights(generic, obs, theta, cycle, precision), direct, rtol=1e-11, atol=1e-9
            )
            legacy_error = None
            if arm == "H0":
                order = [
                    source.parameter_names.index("structure.office"),
                    source.parameter_names.index("candidate.incumbency"),
                ]
                legacy = EmpiricalFundamentals(offset, design[:, order], beta[:, order], tc, tr, ids, source.law_hash)
                result = legacy_weights(legacy, obs, theta, cycle, precision)
                legacy_error = float(np.max(abs(result - direct)))
                np.testing.assert_allclose(result, direct, rtol=1e-11, atol=1e-9)
            assert np.isfinite(beta).all() and (tc > 0).all() and (tr > 0).all()
            records.append(
                dict(
                    year=year,
                    arm=arm,
                    law_hash=source.law_hash,
                    coefficient_names=list(source.parameter_names),
                    training_cycles=source.cycles.tolist(),
                    training_race_ids=list(source.race_ids),
                    target_race_ids=list(ids),
                    target_cutoff_utc=target.forecast_cutoff.iloc[0],
                    source_cutoff_values=sorted(set(str(v) for v in target.source_cutoff_utc.dropna()))
                    if "source_cutoff_utc" in target
                    else [],
                    joint_shape=[4, 4000, q],
                    equal_prior_weight="1/16000",
                    flatten="flat_index=chain*4000+draw; no sorting/shuffling/thinning",
                    raw_banks=terminal["raw_chains"],
                    target_offset_sha256=hashlib.sha256(offset.tobytes()).hexdigest(),
                    target_design_sha256=hashlib.sha256(design.tobytes()).hexdigest(),
                    row_weight_max_error=float(
                        np.max(abs(basis_weights(generic, obs, theta, cycle, precision) - direct))
                    ),
                    legacy_row_weight_max_error=legacy_error,
                    old_2022_training_added=sorted(set(source.race_ids) - set(old_training["race_ids"]))
                    if year == 2022
                    else None,
                    old_2022_target_added=sorted(set(ids) - set(old_training["target_ids"])) if year == 2022 else None,
                    old_2022_target_removed=sorted(set(old_training["target_ids"]) - set(ids))
                    if year == 2022
                    else None,
                )
            )
    for relative in [
        "src/forecasting/polling_v1.py",
        "src/forecasting/polling_basis_v1.py",
        "src/forecasting/poll_covariance_v1.py",
        "src/forecasting/baseline_v5.py",
        "research/polling-layer-implementation-20261020/standard_runner.py",
        "research/polling-layer-implementation-20261020/forecast-v2/source-snapshots/01-standard-plan-root-v1.json",
        "research/polling-layer-implementation-20261020/forecast-v2/source-snapshots/06-forecast_runner.py",
    ]:
        pin(OWNER / relative)
    report = dict(
        schema="MomentumBankCompatibilityNoFitSolV1",
        selection_sha256="5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9",
        old_selected_posterior_shapes=old_shapes,
        old_selected_training=old_training,
        shared_118_training_rows_differences_2022_H0=common_source_differences,
        banks=records,
        inspected_pins=list(PINS.values()),
        sampler_invoked=False,
        optimizer_invoked=False,
        scores_computed=False,
    )
    (HERE / "compatibility.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("PASS:", len(records), "complete banks; constructor/conditional-row-law checks only; no fit/optimizer/scores")


if __name__ == "__main__":
    main()
