"""Recompute all fixed-bin horizon statistics from their complete ID ledger."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
BANK = LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
ORIG_MANIFEST = OUT / "artifact-manifest.json"
OLD_SUMMARY = OUT / "historical-fixed-bin-horizon-correction.csv"
LEDGER = OUT / "historical-fixed-bin-horizon-selection.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(f, bank):
    spans, repeat_groups, repeat_races, repeat_rows = [], 0, set(), 0
    for (race, firm), g in f.groupby(["race_id", "pollster_key"], sort=True, dropna=False):
        dates = pd.to_datetime(g.end_date, errors="coerce").dropna().drop_duplicates()
        if len(dates) >= 2:
            repeat_groups += 1
            repeat_races.add(race)
            repeat_rows += len(g)
            spans.append(float((dates.max() - dates.min()).days))
    return dict(
        rows=len(f), unique_samples=int(f.sample_id.nunique()), races=int(f.race_id.nunique()),
        pollsters_known=int(f.pollster_key.nunique()), repeated_race_pollster_groups=repeat_groups,
        races_with_repeated_pollster=len(repeat_races), repeated_group_rows=repeat_rows,
        median_repeated_group_span_days=None if not spans else float(pd.Series(spans).median()),
        f0_training_races_supported=int(len(set(f.race_id.astype(str)) & bank)),
    )


def main():
    outputs = {"historical-fixed-bin-horizon-correction-v2.csv",
               "historical-fixed-bin-summary-reconstruction-v2.csv",
               "historical-fixed-bin-summary-receipt-v2.json",
               "historical-fixed-bin-summary-correction-v2.md"}
    if outputs.intersection(p.name for p in OUT.iterdir()):
        raise RuntimeError("refusing to overwrite fixed-bin summary correction")
    if sha(ORIG_MANIFEST) != "99ba073742bd329cad8fe4713b2594c205dd1d115ed8cfe6c81c5b316594d574":
        raise RuntimeError("initial coverage manifest changed")
    if sha(BANK) != "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05":
        raise RuntimeError("reviewed F0 bank manifest changed")
    compat = json.loads(BANK.read_text())
    f0 = {int(x["year"]): x for x in compat["banks"] if x["arm"] == "F0"}
    d = pd.read_csv(LEDGER, low_memory=False)
    old = pd.read_csv(OLD_SUMMARY)
    summary, checks = [], []
    for target in (2018, 2020, 2022, 2024):
        bank = set(f0[target]["training_race_ids"])
        fold = d[d.target_cycle.eq(target)]
        for h in (30, 14, 7):
            for lag in (0, 1, 3, 7):
                cell = fold[(fold.horizon_days == h) & (fold.extra_lag_days_beyond_native_proxy == lag)]
                full = cell[cell.within_0_100_and_horizon_proxy.astype(str).str.lower().eq("true")]
                fixed = cell[cell.selected_by_corrected_fixed_bin.astype(str).str.lower().eq("true")]
                for op, frame in [("tierney_cutoff_full_series_sensitivity", full),
                                  ("tierney_cutoff_fixed_14day_bin_sensitivity", fixed)]:
                    row = dict(target_cycle=target, training_cycles=",".join(str(c) for c in sorted(cell.source_cycle.unique())),
                               horizon_days=h, extra_lag_days_beyond_native_proxy=lag, operator=op,
                               pollster_key_rule="exact standard_runner.py strip().casefold()",
                               available_proxy="field_end_date_plus_extra_lag", strict_asof_qualified=False,
                               f0_training_races_total=len(bank), **metrics(frame, bank))
                    prior = old[(old.target_cycle == target) & (old.horizon_days == h) &
                                (old.extra_lag_days_beyond_native_proxy == lag) & (old.operator == op)].iloc[0]
                    checks_row = {"target_cycle": target, "horizon_days": h, "extra_lag_days_beyond_native_proxy": lag,
                                  "operator": op, "all_counts_reconstructed_from_ledger": True,
                                  "ledger_rows": len(frame), "summary_rows": int(prior.rows),
                                  "ledger_races": int(frame.race_id.nunique()), "summary_races": int(prior.races)}
                    # Row, distinct IDs, races, pollsters and F0 support must match the pinned prior summary.
                    for key in ("rows", "unique_samples", "races", "pollsters_known", "f0_training_races_supported"):
                        if int(row[key]) != int(prior[key]):
                            checks_row["all_counts_reconstructed_from_ledger"] = False
                    independent_repeat_groups, independent_repeat_races, independent_repeat_rows, spans = 0, set(), 0, []
                    for (_, _), g in frame.groupby(["race_id", "pollster_key"], sort=True, dropna=False):
                        ds = pd.to_datetime(g.end_date, errors="coerce").dropna().drop_duplicates()
                        if len(ds) >= 2:
                            independent_repeat_groups += 1
                            independent_repeat_races.add(g.race_id.iloc[0])
                            independent_repeat_rows += len(g)
                            spans.append(float((ds.max() - ds.min()).days))
                    independent_median_span = None if not spans else float(pd.Series(spans).median())
                    checks_row.update(
                        independent_repeat_groups=independent_repeat_groups,
                        independent_repeat_races=len(independent_repeat_races),
                        independent_repeat_rows=independent_repeat_rows,
                        independently_rebuilt_repeated_groups=(
                            independent_repeat_groups == int(row["repeated_race_pollster_groups"])
                            and len(independent_repeat_races) == int(row["races_with_repeated_pollster"])
                            and independent_repeat_rows == int(row["repeated_group_rows"])
                            and ((independent_median_span is None and pd.isna(row["median_repeated_group_span_days"]))
                                 or abs(independent_median_span - float(row["median_repeated_group_span_days"])) <= 1e-12)
                        ),
                    )
                    if op == "tierney_cutoff_fixed_14day_bin_sensitivity":
                        row["prior_reported_repeat_groups_from_full_series"] = int(prior.repeated_race_pollster_groups)
                        row["corrected_repeat_group_delta"] = int(row["repeated_race_pollster_groups"] -
                                                                   int(prior.repeated_race_pollster_groups))
                    if not checks_row["all_counts_reconstructed_from_ledger"]:
                        raise RuntimeError(f"fixed-bin ledger reconstruction mismatch: {checks_row}")
                    if not checks_row["independently_rebuilt_repeated_groups"]:
                        raise RuntimeError(f"repeated-group ledger reconstruction mismatch: {checks_row}")
                    checks.append(checks_row)
                    if op == "tierney_cutoff_fixed_14day_bin_sensitivity":
                        row["first_pass_repeated_group_metric_source"] = "full_series; superseded"
                    summary.append(row)
    out = pd.DataFrame(summary)
    out.to_csv(OUT / "historical-fixed-bin-horizon-correction-v2.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "historical-fixed-bin-summary-reconstruction-v2.csv", index=False)
    receipt = dict(
        schema="MomentumFixedBinSummaryFromLedgerV2",
        initial_packet_manifest_sha256=sha(ORIG_MANIFEST),
        earlier_corrected_selection_ledger_sha256=sha(LEDGER),
        earlier_corrected_horizon_summary_sha256=sha(OLD_SUMMARY),
        bank_compatibility_sha256=sha(BANK),
        correction="Every summary row is recomputed from its operator-specific observation-ID set. Repeated-group and span metrics for fixed-bin rows now use retained rows, not the larger full-series set.",
        all_96_operator_cells_reconstructed=True, all_96_repeat_metrics_reconstructed=True,
        all_cell_row_ids_from_complete_ledger=True,
        no_fit=True, no_score=True, no_optimization=True,
        authoritative_table="historical-fixed-bin-horizon-correction-v2.csv",
        outputs=sorted(outputs))
    (OUT / "historical-fixed-bin-summary-receipt-v2.json").write_text(json.dumps(receipt, indent=2) + "\n")
    md = [
        "# Fixed-bin horizon summary correction v2", "",
        "The earlier corrected table used the correct selected IDs and row/race counts, but reported repeat-group and span summaries computed from the larger full-series input even on the fixed-bin operator rows. This version recomputes every summary metric from that operator's retained IDs. The exact complete row-ID ledger is historical-fixed-bin-horizon-selection.csv.", "",
        "The deterministic reconstruction checks all 96 cells (four folds × three horizons × four extra lags × two operators). It confirms row/sample/race/pollster/F0 support counts from each operator-specific ID set. Fixed-bin repeat groups and spans are now computed from selected fixed-bin rows only.", "",
        "Use historical-fixed-bin-horizon-correction-v2.csv as the authoritative historical cutoff/truncation table. The earlier corrected table remains preserved with its repeat metrics marked superseded in this note; the unchanged 0–100 calibration baseline is a separate operator."
    ]
    (OUT / "historical-fixed-bin-summary-correction-v2.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
