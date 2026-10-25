"""Read-only root verification of the fixed actual attempt0 execution group."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "momentum-forecast-root-receipts-v1/inventory.json"
INVENTORY_SHA = "7bd18c7c5fe2d7145b1f98ecb931dfc3b686ddff4010589ff1360569d4d8735f"


def pin(path):
    path = Path(path)
    return dict(
        path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
    )


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(pin(INVENTORY)["sha256"] == INVENTORY_SHA, "Root inventory changed")
    inventory = json.loads(INVENTORY.read_text())
    reference = next(r for r in inventory["items"] if r["action"] == "references")
    checked_pins = {}

    def verify(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                actual = pin(value["path"])
                require(actual["sha256"] == value["sha256"], "Changed artifact: " + value["path"])
                checked_pins[actual["path"]] = actual
            for child in value.values():
                verify(child)
        elif isinstance(value, list):
            for child in value:
                verify(child)

    jobs = []
    for entry in inventory["items"]:
        if entry["action"] != "job":
            continue
        verify(entry["receipt"])
        receipt = json.loads(Path(entry["receipt"]["path"]).read_text())
        output = Path(entry["output"])
        terminal_path = output / "terminal.json"
        if not terminal_path.exists():
            jobs.append(dict(name=entry["name"], status="terminal_not_yet_present", output=str(output)))
            continue
        terminal = json.loads(terminal_path.read_text())
        require(terminal["job"] == receipt["job"], "Job settings mismatch")
        require(terminal["synthetic"] is False, "Actual execution required")
        require(terminal["numeric_target_labels"] is False and terminal["scores"] is False, "Prelabel scope required")
        verify(terminal)
        record = dict(name=entry["name"], terminal=pin(terminal_path), job=terminal["job"], status=terminal["status"])
        jobs.append(record)
        if terminal["status"] != "complete_prelabel":
            record["exception"] = terminal.get("exception")
            continue
        seal = json.loads((output / "prediction-seal.json").read_text())
        verify(seal)
        summary = json.loads((output / "prelabel-summary.json").read_text())
        require(
            set(summary["gates"])
            == {
                "core_state",
                "bulk_mean_share",
                "bulk_absolute_win_probability",
                "bank_chain_consistency",
            },
            "Exact selected gate set required",
        )
        require(all(type(x) is bool for x in summary["gates"].values()), "No unresolved core gate may pass")
        triggers = {key: not value for key, value in summary["gates"].items()}
        require(summary["permitted_retry_triggers"] == triggers, "Numerical retry trigger mismatch")
        require(summary["any_permitted_retry_trigger"] == any(triggers.values()), "Aggregate retry mismatch")
        original_receipt = json.loads((output / "original-receipt.json").read_text())
        verify(original_receipt)
        trace_receipt = json.loads((output / "trace/receipt.json").read_text())
        verify(trace_receipt)
        provenance = json.loads((output / "trace/provenance.json").read_text())
        require(len(provenance["final_rng_states"]) == 4, "Four actual final RNG states required")
        cycle = receipt["job"]["cycle"]
        with (
            np.load(output / "original-predictions.npz", allow_pickle=False) as raw,
            np.load(output / "trace/trace.npz", allow_pickle=False) as trace,
            np.load(output / "prelabel-components.npz", allow_pickle=False) as components,
            np.load(Path(reference["output"]) / str(cycle) / "archived-F0-cdf.npz", allow_pickle=False) as ref,
        ):
            n = len(ref["race_ids"])
            common = np.flatnonzero(ref["common_polled_mask"])
            require(raw["shares"].shape == (4, 4000, n), "Complete original prediction axes required")
            require(raw["coefficients"].shape == (4, 4000, 6), "Complete coefficient axes required")
            require(np.array_equal(raw["observed_race_indices"], common), "Common observed mask mismatch")
            require(np.array_equal(components["common_polled_race_indices"], common), "Component mask mismatch")
            require(
                np.array_equal(components["common_polled_race_ids"], ref["race_ids"][common]), "Race order mismatch"
            )
            require(
                summary["common_polled_race_ids"] == ref["race_ids"][common].tolist(), "Summary race order mismatch"
            )
            require(np.array_equal(raw["logits"][:, :, common], trace["observed_theta"]), "Observed raw/trace mismatch")
            require(np.array_equal(raw["shared_cycle"], trace["cycle"]), "Shared-cycle raw/trace mismatch")
            require(np.array_equal(raw["posterior_index"], trace["sampled_row_index"]), "Sampled-row mismatch")
            for raw_key, trace_key in (
                ("posterior_chain", "posterior_source_chain"),
                ("posterior_draw", "posterior_source_draw"),
            ):
                require(
                    np.array_equal(raw[raw_key], trace[trace_key][raw["posterior_index"]]),
                    "Source-row mapping mismatch",
                )
            require(np.array_equal(raw["first_wins"], raw["shares"] > 0.5), "Original winner indicator mismatch")
            require(np.isfinite(raw["shares"]).all(), "Nonfinite raw shares")
            require(np.all(trace["bank_batch_counts"].sum(axis=1) == 4000), "Complete row-weight batches required")
            require(np.all(trace["bank_sampled_occupancy"].sum(axis=1) == 4000), "Complete occupancy required")
            require(
                np.max(abs(trace["bank_batch_weight_sums"].sum(axis=2) - trace["bank_batch_counts"])) < 1e-8,
                "Row-weight mass mismatch",
            )
            require(components["component_expected_shares"].shape == (4, 4000, len(common)), "Component axes mismatch")
        record.update(
            gates=summary["gates"],
            tail_qualification=summary["tail_qualification"],
            interval_display_qualified=summary["interval_display_qualified"],
            inherited_bank_warning=summary["inherited_bank_warning"],
            any_permitted_retry_trigger=summary["any_permitted_retry_trigger"],
            original_trace_axes_and_observed_values_exact=True,
            numeric_target_labels=False,
            scores=False,
        )
    report = dict(
        schema="MomentumActualPrelabelRootVerificationV1",
        utc=datetime.now(timezone.utc).isoformat(),
        inventory=pin(INVENTORY),
        audit_source=pin(__file__),
        jobs=jobs,
        verified_pins=len(checked_pins),
        complete_jobs=sum(r["status"] == "complete_prelabel" for r in jobs),
        scope="Artifact identity, raw/observer alignment, complete axes and declared diagnostics; no labels or scores.",
        numeric_target_labels=False,
        scores=False,
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(report=pin(args.output), complete_jobs=report["complete_jobs"], verified_pins=len(checked_pins))
        )
    )


if __name__ == "__main__":
    main()
