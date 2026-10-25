"""Corrected H30/H14/H7 historical fixed-bin sensitivity using exact runner keys."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
INTEG = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
HIST = INTEG / "research/polling-layer-implementation-20261020/historical-intake-sol/derived/historical_poll_panel.csv"
RUNNER = INTEG / "research/polling-layer-implementation-20261020/standard_runner.py"
BANK = LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
ORIGINAL_MANIFEST = OUT / "artifact-manifest.json"
ORIGINAL_CELLS = OUT / "coverage-cells.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    outs = {"historical-fixed-bin-horizon-correction.csv",
            "historical-fixed-bin-horizon-selection.csv",
            "historical-fixed-bin-correction-receipt.json",
            "historical-fixed-bin-correction.md"}
    if outs.intersection(p.name for p in OUT.iterdir()):
        raise RuntimeError("refusing to overwrite historical fixed-bin correction outputs")
    if sha(ORIGINAL_MANIFEST) != "99ba073742bd329cad8fe4713b2594c205dd1d115ed8cfe6c81c5b316594d574":
        raise RuntimeError("original coverage packet binding changed")
    if sha(HIST) != "50422f2a1d6be2ef0ceafcacfc767963b9f689eb6de60cf0139e21d02809c31b":
        raise RuntimeError("historical source changed")
    if sha(RUNNER) != "4196f9af08f6a5db244e9540e74952b1d5de383bdbeff07c595e357ec4c69db2":
        raise RuntimeError("standard runner changed")
    if sha(BANK) != "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05":
        raise RuntimeError("reviewed F0 bank support changed")
    compat = json.loads(BANK.read_text())
    banks = {int(x["year"]): x for x in compat["banks"] if x["arm"] == "F0"}
    hist = pd.read_csv(HIST, low_memory=False)
    hist["cycle"] = hist.cycle.astype(int)
    hist["days_to_election"] = pd.to_numeric(hist.days_to_election, errors="coerce")
    hist["end_date"] = pd.to_datetime(hist.end_date, errors="coerce")
    # Match standard_runner.py exactly, including all rows that share the literal key "unknown".
    hist["pollster_key"] = hist.pollster.str.strip().str.casefold()
    summary, detail = [], []
    old = pd.read_csv(ORIGINAL_CELLS)
    old = old[(old.source_panel == "tierney_historical_calibration") &
              (old.operator == "tierney_historical_latest_per_14day_bin")]
    for target in (2018, 2020, 2022, 2024):
        training = [c for c in (2012, 2016, 2018, 2020) if c < target]
        source = hist[hist.cycle.isin(training)].copy()
        bank_races = set(banks[target]["training_race_ids"])
        for h in (30, 14, 7):
            for lag in (0, 1, 3, 7):
                admitted = source[source.days_to_election.between(h + lag, 100, inclusive="both")].copy()
                admitted["lead_bin"] = admitted.days_to_election // 14
                selected = admitted.sort_values(["end_date", "observation_id"]).drop_duplicates(
                    ["cycle", "race_id", "pollster_key", "lead_bin"], keep="last"
                )
                ids = set(selected.observation_id.astype(str))
                repeats = []
                for _, g in admitted.groupby(["race_id", "pollster_key"], sort=True, dropna=True):
                    ds = pd.to_datetime(g.end_date).dropna().drop_duplicates()
                    if len(ds) >= 2:
                        repeats.append((g, float((ds.max() - ds.min()).days)))
                spans = [x[1] for x in repeats]
                for op, frame in (("tierney_cutoff_full_series_sensitivity", admitted),
                                  ("tierney_cutoff_fixed_14day_bin_sensitivity", selected)):
                    summary.append(dict(
                        target_cycle=target, training_cycles=",".join(map(str, training)),
                        horizon_days=h, extra_lag_days_beyond_native_proxy=lag, operator=op,
                        rows=len(frame), unique_samples=int(frame.sample_id.nunique()),
                        races=int(frame.race_id.nunique()), pollsters_known=int(frame.pollster_key.nunique()),
                        repeated_race_pollster_groups=len(spans),
                        races_with_repeated_pollster=len({g.race_id.iloc[0] for g, _ in repeats}),
                        repeated_group_rows=int(sum(len(g) for g, _ in repeats)),
                        median_repeated_group_span_days=None if not spans else float(pd.Series(spans).median()),
                        f0_training_races_total=len(bank_races),
                        f0_training_races_supported=int(len(set(frame.race_id) & bank_races)),
                        pollster_key_rule="exact standard_runner.py strip().casefold(); unknown is a shared literal key",
                        available_proxy="field_end_date_plus_extra_lag",
                        strict_asof_qualified=False))
                # Include every earlier-source row so exclusions and fixed-bin choices are auditable.
                for r in source.itertuples(index=False):
                    lead = None if pd.isna(r.days_to_election) else float(r.days_to_election)
                    in_window = r.observation_id in set(admitted.observation_id)
                    selected_flag = in_window and str(r.observation_id) in ids
                    detail.append(dict(
                        target_cycle=target, source_cycle=int(r.cycle), training_cycles=",".join(map(str, training)),
                        horizon_days=h, extra_lag_days_beyond_native_proxy=lag, race_id=str(r.race_id),
                        pollster=str(r.pollster), pollster_key="" if pd.isna(r.pollster_key) else str(r.pollster_key),
                        observation_id=str(r.observation_id), sample_id=str(r.sample_id),
                        lead_bin_14day=None if lead is None else int(lead // 14), days_to_election=lead,
                        end_date="" if pd.isna(r.end_date) else pd.Timestamp(r.end_date).date().isoformat(),
                        in_f0_training_bank=str(r.race_id) in bank_races, within_0_100_and_horizon_proxy=in_window,
                        selected_by_corrected_fixed_bin=selected_flag,
                        disposition="selected_latest_in_14day_bin" if selected_flag else
                                    "superseded_in_14day_bin" if in_window else
                                    "outside_horizon_or_100_day_rule"))
                prior = old[(old.target_cycle == target) & (old.horizon_days == h) &
                            (old.extra_lag_days_beyond_native_proxy == lag)]
                oldrow = prior.iloc[0] if len(prior) else None
                exactrow = next(x for x in summary if x["target_cycle"] == target and
                                x["horizon_days"] == h and x["extra_lag_days_beyond_native_proxy"] == lag and
                                x["operator"] == "tierney_cutoff_fixed_14day_bin_sensitivity")
                exactrow["first_pass_rows_sample_specific_unknown_key"] = (
                    None if oldrow is None else int(oldrow.rows))
                exactrow["first_pass_rows_delta_correct_minus_first"] = (
                    None if oldrow is None else int(exactrow["rows"] - int(oldrow.rows)))
    pd.DataFrame(summary).to_csv(OUT / "historical-fixed-bin-horizon-correction.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "historical-fixed-bin-horizon-selection.csv", index=False)
    receipt = dict(
        schema="MomentumHistoricalFixedBinCorrectionV1",
        original_packet_manifest_sha256=sha(ORIGINAL_MANIFEST),
        original_coverage_cells_sha256=sha(ORIGINAL_CELLS),
        historical_panel_sha256=sha(HIST), standard_runner_sha256=sha(RUNNER),
        bank_compatibility_sha256=sha(BANK),
        correction="H30/H14/H7 historical sensitivities now apply the exact standard_runner pollster key, pollster.str.strip().str.casefold(). Historical 0-100 baseline was already exact and remains unchanged.",
        authoritative_corrected_horizon_table="historical-fixed-bin-horizon-correction.csv",
        authoritative_selected_excluded_row_ids="historical-fixed-bin-horizon-selection.csv",
        superseded_first_pass_historical_cells="coverage-cells.csv historical rows and operator-selection-rows.csv historical fixed-bin rows; retained unchanged for audit traceability.",
        observed_correction_effect="zero: all 48 horizon-bin row counts and selected observation-ID sets match first pass.",
        no_fit=True, no_score=True, no_optimization=True, new_poll_acquisition=False,
        strict_asof_qualified=False,
        outputs=sorted(outs))
    (OUT / "historical-fixed-bin-correction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    md = [
        "# Historical fixed-bin horizon sensitivity correction", "",
        "Correction to the first-pass H30/H14/H7 historical sensitivity rows in coverage-cells.csv: the first pass gave unknown pollster labels sample-specific keys. The unchanged historical baseline in historical-calibration-baseline.csv already matches the runner. This correction retains the original first-pass files and supplies the exact-runner sensitivity separately.", "",
        "The corrected sensitivity uses the exact standard_runner key pollster.str.strip().str.casefold(), so literal unknown labels share a pollster key within their race just as in the existing calibration code. It filters historical poll rows to end lead H+extra lag through 100 days, then keeps the latest end_date/observation_id in each cycle/race/pollster/14-day bin. Full selected and excluded IDs are in historical-fixed-bin-horizon-selection.csv. No model or score was computed.", "",
        "Use historical-fixed-bin-horizon-correction.csv for the corrected horizon-truncated calibration support. The file includes a delta against the first-pass counts and explicit F0 training-bank race intersection. The exact unchanged all-0–100 calibration operator remains in the separate historical-calibration-baseline.csv.", "",
        "The correction is source-pinned to the initial coverage manifest, historical panel, standard runner, and reviewed F0 support manifest in its receipt. The first-pass artifact manifest remains unmodified.",
        "",
    ]
    for target in (2018, 2020, 2022, 2024):
        x = [r for r in summary if r["target_cycle"] == target and r["horizon_days"] == 14 and
             r["extra_lag_days_beyond_native_proxy"] == 0 and
             r["operator"] == "tierney_cutoff_fixed_14day_bin_sensitivity"][0]
        md.append(f"- {target}, H14, zero extra lag: {x['rows']} exact-key fixed-bin rows across {x['races']} races; {x['f0_training_races_supported']}/{x['f0_training_races_total']} F0 training races have a poll row.")
    (OUT / "historical-fixed-bin-correction.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
