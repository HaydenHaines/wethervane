"""Versioned descriptive deltas; never change either audit's selection rules."""

import json
from pathlib import Path

import identity
import pandas as pd

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "momentum-audit-v1"


def main():
    identity.authenticate()
    outputs = ("race-deltas.csv", "comparison-deltas.csv", "summary-deltas.csv", "delta-receipt.json")
    for name in outputs:
        identity.require(not (HERE / name).exists(), "exclusive new delta output required")
    before, after = (pd.read_csv(root / "selected-polls.csv") for root in (V1, HERE))
    old_counts, new_counts = before.groupby("race_id").size(), after.groupby("race_id").size()
    old_cov, new_cov = (pd.read_csv(root / "coverage.csv") for root in (V1, HERE))
    races = []
    for race_id in sorted(set(old_cov.race_id)):
        old = old_cov[old_cov.race_id.eq(race_id) & old_cov.variant.eq("field_end")].iloc[0]
        new = new_cov[new_cov.race_id.eq(race_id) & new_cov.variant.eq("field_end")].iloc[0]
        races.append(
            dict(
                race_id=race_id,
                cycle=int(new.cycle),
                v1_has_truth=bool(old.has_truth),
                v2_has_truth=bool(new.has_truth),
                v1_selected_polls=int(old_counts.get(race_id, 0)),
                v2_selected_polls=int(new_counts.get(race_id, 0)),
                restored_poll_rows=int(new_counts.get(race_id, 0) - old_counts.get(race_id, 0)),
                v1_field_end_sufficient=bool(old.sufficient),
                v2_field_end_sufficient=bool(new.sufficient),
            )
        )
    pd.DataFrame(races).to_csv(HERE / outputs[0], index=False)
    keys = ["variant", "comparison", "race_id", "cycle"]
    old, new = (pd.read_csv(root / "race-comparisons.csv") for root in (V1, HERE))
    original_comparison_rows = len(old)
    merged = old.merge(new, on=keys, how="outer", suffixes=("_v1", "_v2"), indicator=True, validate="one_to_one")
    merged.to_csv(HERE / outputs[1], index=False)
    keys = ["variant", "comparison", "threshold_pp", "cycle"]
    old, new = (pd.read_csv(root / "summary.csv", dtype={"cycle": str}) for root in (V1, HERE))
    summary = old.merge(new, on=keys, suffixes=("_v1", "_v2"), validate="one_to_one")
    for name in ("n", "below", "within", "above", "mean_performance_pp"):
        summary[name + "_delta"] = summary[name + "_v2"] - summary[name + "_v1"]
    summary.to_csv(HERE / outputs[2], index=False)
    receipt = dict(
        schema="momentum_identity_only_delta_v2",
        restored_rows=len(after) - len(before),
        restored_by_race={r["race_id"]: r["restored_poll_rows"] for r in races if r["restored_poll_rows"]},
        original_comparison_rows=original_comparison_rows,
        v1_untouched=True,
        fit=False,
        strict_asof_qualified=False,
        output_pins=[dict(path=name, sha256=identity.sha(HERE / name)) for name in outputs[:3]],
    )
    (HERE / outputs[3]).write_text(json.dumps(receipt, indent=2) + "\n")
    identity.authenticate()


if __name__ == "__main__":
    main()
