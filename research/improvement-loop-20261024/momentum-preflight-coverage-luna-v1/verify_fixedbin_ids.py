"""Compare fixed-bin selection IDs across first-pass and exact-runner versions."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
OLD = OUT / "operator-selection-rows.csv"
NEW = OUT / "historical-fixed-bin-horizon-selection.csv"
DEST = OUT / "historical-fixed-bin-selection-id-delta.csv"
RECEIPT = OUT / "historical-fixed-bin-id-check-receipt.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if DEST.exists() or RECEIPT.exists():
        raise RuntimeError("refusing to overwrite fixed-bin ID comparison outputs")
    old = pd.read_csv(OLD, low_memory=False)
    old = old[(old.source_panel == "tierney_historical_calibration") &
              (old.operator == "tierney_historical_latest_per_14day_bin") &
              old.selected.astype(str).str.lower().eq("true")]
    new = pd.read_csv(NEW, low_memory=False)
    new = new[new.selected_by_corrected_fixed_bin]
    rows = []
    for target in (2018, 2020, 2022, 2024):
        for h in (30, 14, 7):
            for lag in (0, 1, 3, 7):
                a = old[(old.target_cycle == target) & (old.horizon_days == h) &
                        (old.extra_lag_days_beyond_native_proxy == lag)]
                b = new[(new.target_cycle == target) & (new.horizon_days == h) &
                        (new.extra_lag_days_beyond_native_proxy == lag)]
                aa, bb = set(a.observation_id.astype(str)), set(b.observation_id.astype(str))
                rows.append(dict(target_cycle=target, horizon_days=h, extra_lag_days=lag,
                                 first_pass_selected_ids=len(aa), corrected_selected_ids=len(bb),
                                 common_ids=len(aa & bb), first_pass_only_ids=len(aa - bb),
                                 corrected_only_ids=len(bb - aa), exact_selected_id_match=aa == bb))
    frame = pd.DataFrame(rows)
    if not frame.exact_selected_id_match.all():
        raise RuntimeError("fixed-bin selected IDs differ; preserve as an explicit correction failure")
    frame.to_csv(DEST, index=False)
    receipt = dict(schema="MomentumFixedBinSelectionIdDeltaV1",
                   first_pass_manifest_sha256=sha(OUT / "artifact-manifest.json"),
                   first_pass_operator_rows_sha256=sha(OLD),
                   corrected_selection_rows_sha256=sha(NEW),
                   all_48_selected_id_sets_equal=True, all_48_row_count_deltas_zero=True,
                   outputs=[DEST.name])
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
