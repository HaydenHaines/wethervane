"""Complete earlier-cycle source-row ledger and reconstruct baseline summaries."""
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
ORIG_MANIFEST = OUT / "artifact-manifest.json"
ORIG_SUMMARY = OUT / "historical-calibration-baseline.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    names = {"historical-calibration-baseline-id-ledger-v2.csv",
             "historical-calibration-baseline-summary-recheck-v2.csv",
             "historical-calibration-baseline-correction-receipt-v2.json",
             "historical-calibration-baseline-correction-v2.md"}
    if names.intersection(p.name for p in OUT.iterdir()):
        raise RuntimeError("refusing to overwrite baseline correction outputs")
    expected = {
        HIST: "50422f2a1d6be2ef0ceafcacfc767963b9f689eb6de60cf0139e21d02809c31b",
        RUNNER: "4196f9af08f6a5db244e9540e74952b1d5de383bdbeff07c595e357ec4c69db2",
        BANK: "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05",
        ORIG_MANIFEST: "99ba073742bd329cad8fe4713b2594c205dd1d115ed8cfe6c81c5b316594d574",
    }
    for p, h in expected.items():
        if sha(p) != h:
            raise RuntimeError("source pin mismatch: " + str(p))
    compat = json.loads(BANK.read_text())
    banks = {int(x["year"]): x for x in compat["banks"] if x["arm"] == "F0"}
    src = pd.read_csv(HIST, low_memory=False)
    src["cycle"] = src.cycle.astype(int)
    src["days_to_election"] = pd.to_numeric(src.days_to_election, errors="coerce")
    src["end_date"] = pd.to_datetime(src.end_date, errors="coerce")
    src["pollster_key"] = src.pollster.str.strip().str.casefold()
    old = pd.read_csv(ORIG_SUMMARY)
    detail, summaries = [], []
    for target in (2018, 2020, 2022, 2024):
        train_cycles = [c for c in (2012, 2016, 2018, 2020) if c < target]
        bank_ids = set(banks[target]["training_race_ids"])
        training = src[src.cycle.isin(train_cycles)].copy()
        training["lead_bin"] = training.days_to_election // 14
        all_in_range = training[training.days_to_election.between(0, 100, inclusive="both")].copy()
        all_selected = all_in_range.sort_values(["end_date", "observation_id"]).drop_duplicates(
            ["cycle", "race_id", "pollster_key", "lead_bin"], keep="last"
        )
        selected_ids = set(all_selected.observation_id.astype(str))

        for label, source_part in [("all_earlier_cycles", training)] + [
                (str(c), training[training.cycle.eq(c)]) for c in train_cycles]:
            if label == "all_earlier_cycles":
                admitted = all_in_range
                selected = all_selected
            else:
                cycle = int(label)
                admitted = all_in_range[all_in_range.cycle.eq(cycle)]
                selected = all_selected[all_selected.cycle.eq(cycle)]
            repeat_groups, repeat_spans = [], []
            for _, g in admitted.groupby(["race_id", "pollster_key"], sort=True, dropna=False):
                dates = pd.to_datetime(g.end_date).dropna().drop_duplicates()
                if len(dates) >= 2:
                    repeat_groups.append(g)
                    repeat_spans.append(float((dates.max() - dates.min()).days))
            expected_row = old[(old.target_cycle == target) & (old.training_cycle_group.astype(str) == label)].iloc[0]
            row = dict(
                target_cycle=target, training_cycle_group=label, training_cycles=",".join(map(str, train_cycles)),
                operator="tierney_historical_latest_per_14day_bin",
                raw_eligible_rows_0_100=len(admitted), fixed_bin_selected_rows=len(selected),
                raw_unique_races=int(admitted.race_id.nunique()),
                fixed_bin_unique_races=int(selected.race_id.nunique()),
                raw_unique_observations=int(admitted.observation_id.nunique()),
                fixed_bin_unique_observations=int(selected.observation_id.nunique()),
                raw_named_pollsters=int(admitted.pollster_key.nunique()),
                fixed_bin_named_pollsters=int(selected.pollster_key.nunique()),
                raw_repeated_race_pollster_groups=len(repeat_groups),
                median_raw_repeat_span_days=None if not repeat_spans else float(pd.Series(repeat_spans).median()),
                f0_training_races_total=len(bank_ids),
                f0_training_races_with_any_raw_panel_poll=int(len(set(admitted.race_id.astype(str)) & bank_ids)),
                f0_training_races_with_selected_bin_poll=int(len(set(selected.race_id.astype(str)) & bank_ids)),
                max_age_days=100, lower_lead_inclusive=0, availability_proxy="field_end_only",
                strict_asof_qualified=False,
            )
            checks = {}
            for k in ("raw_eligible_rows_0_100", "fixed_bin_selected_rows", "raw_unique_races",
                      "fixed_bin_unique_races", "raw_unique_observations", "fixed_bin_unique_observations",
                      "raw_named_pollsters", "fixed_bin_named_pollsters", "raw_repeated_race_pollster_groups",
                      "f0_training_races_total", "f0_training_races_with_any_raw_panel_poll",
                      "f0_training_races_with_selected_bin_poll", "max_age_days", "lower_lead_inclusive"):
                checks[k] = int(row[k]) == int(expected_row[k])
            if expected_row.median_raw_repeat_span_days != expected_row.median_raw_repeat_span_days:
                checks["median_raw_repeat_span_days"] = row["median_raw_repeat_span_days"] is None
            else:
                checks["median_raw_repeat_span_days"] = abs(float(row["median_raw_repeat_span_days"]) -
                                                            float(expected_row.median_raw_repeat_span_days)) <= 1e-12
            if not all(checks.values()):
                raise RuntimeError(f"baseline summary reconstruction mismatch for {target}/{label}: {checks}")
            summaries.append(row | {"all_summary_fields_reconstructed": True})

        # Ledger intentionally covers every row from every earlier source cycle, not only last source year.
        for r in training.itertuples(index=False):
            lead = None if pd.isna(r.days_to_election) else float(r.days_to_election)
            in_range = lead is not None and 0 <= lead <= 100
            chosen = in_range and str(r.observation_id) in selected_ids
            detail.append(dict(
                target_cycle=target, training_cycles=",".join(map(str, train_cycles)),
                source_cycle=int(r.cycle), race_id=str(r.race_id), pollster=str(r.pollster),
                pollster_key="" if pd.isna(r.pollster_key) else str(r.pollster_key),
                observation_id=str(r.observation_id), sample_id=str(r.sample_id),
                lead_bin_14day=None if lead is None else int(lead // 14),
                days_to_election=lead,
                end_date="" if pd.isna(r.end_date) else pd.Timestamp(r.end_date).date().isoformat(),
                in_f0_training_bank=str(r.race_id) in bank_ids,
                within_existing_0_100_age_rule=bool(in_range),
                selected_by_exact_fixed_bin_operator=bool(chosen),
                disposition="selected_latest_in_14day_bin" if chosen else
                            "superseded_in_14day_bin" if in_range else "outside_existing_0_100_age_rule",
            ))
    detail_df = pd.DataFrame(detail)
    summary_df = pd.DataFrame(summaries)
    # Reconstruct every summary field from the complete ID ledger, including all source years.
    for target in (2018, 2020, 2022, 2024):
        fold = detail_df[detail_df.target_cycle.eq(target)]
        training_cycles = [int(c) for c in fold.source_cycle.unique()]
        expected_cycles = [c for c in (2012, 2016, 2018, 2020) if c < target]
        if training_cycles != expected_cycles:
            raise RuntimeError(f"training cycle ledger incomplete for {target}: {training_cycles}")
        bank_ids = set(banks[target]["training_race_ids"])
        for label, sub in [("all_earlier_cycles", fold)] + [
                (str(c), fold[fold.source_cycle.eq(c)]) for c in expected_cycles]:
            eligible = sub[sub.within_existing_0_100_age_rule]
            selected = sub[sub.selected_by_exact_fixed_bin_operator]
            s = summary_df[(summary_df.target_cycle == target) & (summary_df.training_cycle_group.astype(str) == label)].iloc[0]
            repeat_spans = []
            repeat_groups = 0
            repeat_rows = 0
            repeat_races = set()
            for (race, key), g in eligible.groupby(["race_id", "pollster_key"], sort=True, dropna=False):
                dates = pd.to_datetime(g.end_date, errors="coerce").dropna().drop_duplicates()
                if len(dates) >= 2:
                    repeat_groups += 1
                    repeat_rows += len(g)
                    repeat_races.add(race)
                    repeat_spans.append(float((dates.max() - dates.min()).days))
            values = {
                "raw_eligible_rows_0_100": len(eligible),
                "fixed_bin_selected_rows": len(selected),
                "raw_unique_races": int(eligible.race_id.nunique()),
                "fixed_bin_unique_races": int(selected.race_id.nunique()),
                "raw_unique_observations": int(eligible.observation_id.nunique()),
                "fixed_bin_unique_observations": int(selected.observation_id.nunique()),
                "raw_named_pollsters": int(eligible.pollster_key.nunique()),
                "fixed_bin_named_pollsters": int(selected.pollster_key.nunique()),
                "raw_repeated_race_pollster_groups": repeat_groups,
                "f0_training_races_total": len(bank_ids),
                "f0_training_races_with_any_raw_panel_poll": int(len(set(eligible.race_id.astype(str)) & bank_ids)),
                "f0_training_races_with_selected_bin_poll": int(len(set(selected.race_id.astype(str)) & bank_ids)),
                "max_age_days": 100, "lower_lead_inclusive": 0,
            }
            ledger_median = None if not repeat_spans else float(pd.Series(repeat_spans).median())
            matches = all(int(values[k]) == int(s[k]) for k in values)
            median_matches = ((ledger_median is None and pd.isna(s.median_raw_repeat_span_days)) or
                              (ledger_median is not None and
                               abs(ledger_median - float(s.median_raw_repeat_span_days)) <= 1e-12))
            original = old[(old.target_cycle == target) & (old.training_cycle_group.astype(str) == label)].iloc[0]
            original_matches = all(int(values[k]) == int(original[k]) for k in values)
            original_median_matches = ((ledger_median is None and pd.isna(original.median_raw_repeat_span_days)) or
                                       (ledger_median is not None and
                                        abs(ledger_median - float(original.median_raw_repeat_span_days)) <= 1e-12))
            if not (matches and median_matches and original_matches and original_median_matches):
                raise RuntimeError(f"complete-ledger summary reconstruction mismatch for {target}/{label}")
            summary_df.loc[(summary_df.target_cycle == target) &
                           (summary_df.training_cycle_group.astype(str) == label), "ledger_full_summary_reconstructed"] = True
    if len(detail_df) != 6395 or detail_df.duplicated(["target_cycle", "observation_id"]).any():
        raise RuntimeError("complete training-fold source-row ledger count/key check failed")
    if not summary_df.ledger_full_summary_reconstructed.all():
        raise RuntimeError("not every baseline summary row was rebuilt from the complete ID ledger")
    detail_df.to_csv(OUT / "historical-calibration-baseline-id-ledger-v2.csv", index=False)
    summary_df.to_csv(OUT / "historical-calibration-baseline-summary-recheck-v2.csv", index=False)
    receipt = dict(
        schema="MomentumHistoricalBaselineCompleteLedgerV2",
        original_packet_manifest_sha256=sha(ORIG_MANIFEST),
        original_summary_sha256=sha(ORIG_SUMMARY), historical_panel_sha256=sha(HIST),
        standard_runner_sha256=sha(RUNNER), bank_compatibility_sha256=sha(BANK),
        correction="The first-pass detailed ID ledger iterated only the final training-cycle frame. This version includes every historical row from every strictly earlier cycle in each fold.",
        source_cycles_by_target={str(y): [c for c in (2012, 2016, 2018, 2020) if c < y] for y in (2018, 2020, 2022, 2024)},
        ledger_rows=len(detail_df), reconstructed_summary_rows=len(summary_df),
        every_existing_baseline_summary_field_rebuilt_from_complete_id_ledger=True,
        all_fold_cycle_sets_complete=True, duplicate_observation_ids_within_fold=0,
        no_fit=True, no_score=True, no_optimization=True,
        outputs=["historical-calibration-baseline-id-ledger-v2.csv",
                 "historical-calibration-baseline-summary-recheck-v2.csv",
                 "historical-calibration-baseline-correction-v2.md"])
    (OUT / "historical-calibration-baseline-correction-receipt-v2.json").write_text(json.dumps(receipt, indent=2) + "\n")
    md = [
        "# Historical baseline ledger correction v2", "",
        "The first-pass baseline summary used all earlier training cycles correctly, but its detailed selection ledger iterated only the last individual training-cycle frame. This version supersedes that ledger with every source row from every strictly earlier cycle. The historical calibration operator and its summary counts are unchanged.", "",
        "The new ledger has 6,395 target-fold/source-row records (2018: 2012/2016; 2020: 2012/2016/2018; 2022 and 2024: 2012/2016/2018/2020). Every selected row is retained as a separate original observation ID; no source rows are combined.", "",
        "The deterministic check reconstructs every field of all 16 existing fold/cycle summary rows from the complete source rows and selected IDs. It also confirms every earlier training cycle is represented and that raw 0–100 and fixed-bin selected row counts are recoverable from the ledger. No outcomes, model fit, or score are used.", "",
        "Use historical-calibration-baseline-id-ledger-v2.csv as the complete detailed ledger and historical-calibration-baseline-summary-recheck-v2.csv as the reconstructed summary check. The first-pass files remain preserved."
    ]
    (OUT / "historical-calibration-baseline-correction-v2.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
