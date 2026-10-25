"""Exact no-fit 0-100-day historical fixed-bin coverage, for each F0 fold."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
INTEG = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
HIST = INTEG / "research/polling-layer-implementation-20261020/historical-intake-sol/derived/historical_poll_panel.csv"
RUNNER = INTEG / "research/polling-layer-implementation-20261020/standard_runner.py"
F0_PLAN = LAB / "economic-experiment-v1/prepared-v1/plan.json"
BANK_COMPAT = LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
SELECTION = LAB / "checkpoints/momentum-preflight-root-selection-v1.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def span_median(groups, dates):
    spans = []
    for _, idx in groups:
        d = pd.to_datetime(dates.loc[idx], errors="coerce").dropna()
        if d.nunique() >= 2:
            spans.append(float((d.max() - d.min()).days))
    return None if not spans else float(pd.Series(spans).median())


def main():
    outputs = {"historical-calibration-baseline.csv",
               "historical-calibration-baseline-selection-rows.csv",
               "historical-calibration-baseline-receipt.json",
               "historical-calibration-baseline-addendum.md"}
    if outputs.intersection(p.name for p in OUT.iterdir()):
        raise RuntimeError("refusing to overwrite historical calibration baseline outputs")
    if sha(SELECTION) != "5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9":
        raise RuntimeError("root selection hash mismatch")
    if sha(BANK_COMPAT) != "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05":
        raise RuntimeError("reviewed F0 bank support hash mismatch")
    if sha(HIST) != "50422f2a1d6be2ef0ceafcacfc767963b9f689eb6de60cf0139e21d02809c31b":
        raise RuntimeError("historical panel hash mismatch")
    if sha(RUNNER) != "4196f9af08f6a5db244e9540e74952b1d5de383bdbeff07c595e357ec4c69db2":
        raise RuntimeError("standard runner hash mismatch")

    plan = json.loads(F0_PLAN.read_text())
    manifest = json.loads(BANK_COMPAT.read_text())
    f0 = {int(x["year"]): x for x in manifest["banks"] if x["arm"] == "F0"}
    hist = pd.read_csv(HIST, low_memory=False)
    hist["days_to_election"] = pd.to_numeric(hist.days_to_election, errors="coerce")
    hist["cycle"] = hist.cycle.astype(int)
    hist["end_date"] = pd.to_datetime(hist.end_date, errors="coerce")
    hist["pollster_key"] = hist.pollster.str.strip().str.casefold()
    summaries, detail = [], []
    for target in [2018, 2020, 2022, 2024]:
        training_cycles = [c for c in (2012, 2016, 2018, 2020) if c < target]
        source = hist[hist.cycle.isin(training_cycles)].copy()
        source["lead_bin"] = source.days_to_election // 14
        # Exact standard_runner historical-calibration observation operator.
        source["operator_selected"] = source.days_to_election.between(0, 100, inclusive="both")
        in_range = source[source.operator_selected].copy()
        chosen = in_range.sort_values(["end_date", "observation_id"]).drop_duplicates(
            ["cycle", "race_id", "pollster_key", "lead_bin"], keep="last"
        )
        chosen_ids = set(chosen.observation_id.astype(str))
        bank_ids = set(f0[target]["training_race_ids"])
        for cycle_group, frame in [("all_earlier_cycles", source)] + [
            (str(c), source[source.cycle.eq(c)]) for c in training_cycles
        ]:
            if cycle_group == "all_earlier_cycles":
                rs = in_range
                selected = chosen
            else:
                rs = in_range[in_range.cycle.eq(int(cycle_group))]
                selected = chosen[chosen.cycle.eq(int(cycle_group))]
            pollster_groups = list(rs.groupby(["race_id", "pollster_key"], sort=True))
            repeat_groups = [(k, g) for k, g in pollster_groups
                             if pd.to_datetime(g.end_date).nunique() >= 2]
            spans = [(pd.to_datetime(g.end_date).max() - pd.to_datetime(g.end_date).min()).days
                     for _, g in repeat_groups]
            summaries.append(dict(
                target_cycle=target, training_cycle_group=cycle_group,
                training_cycles=",".join(map(str, training_cycles)),
                operator="tierney_historical_latest_per_14day_bin",
                raw_eligible_rows_0_100=len(rs), fixed_bin_selected_rows=len(selected),
                raw_unique_races=int(rs.race_id.nunique()),
                fixed_bin_unique_races=int(selected.race_id.nunique()),
                raw_unique_observations=int(rs.observation_id.nunique()),
                fixed_bin_unique_observations=int(selected.observation_id.nunique()),
                raw_named_pollsters=int(rs.pollster_key.nunique()),
                fixed_bin_named_pollsters=int(selected.pollster_key.nunique()),
                raw_repeated_race_pollster_groups=len(repeat_groups),
                median_raw_repeat_span_days=None if not spans else float(pd.Series(spans).median()),
                f0_training_races_total=len(bank_ids),
                f0_training_races_with_any_raw_panel_poll=len(set(rs.race_id) & bank_ids),
                f0_training_races_with_selected_bin_poll=len(set(selected.race_id) & bank_ids),
                max_age_days=100, lower_lead_inclusive=0, availability_proxy="field_end_only",
                strict_asof_qualified=False,
            ))
        for r in frame.itertuples(index=False):
            lead = None if pd.isna(r.days_to_election) else float(r.days_to_election)
            in_range_flag = bool(0 <= lead <= 100) if lead is not None else False
            chosen_flag = in_range_flag and str(r.observation_id) in chosen_ids
            detail.append(dict(
                target_cycle=target, training_cycles=",".join(map(str, training_cycles)),
                source_cycle=int(r.cycle), race_id=str(r.race_id), pollster=str(r.pollster),
                pollster_key=str(r.pollster_key), observation_id=str(r.observation_id),
                sample_id=str(r.sample_id), lead_bin_14day=None if lead is None else int(lead // 14),
                days_to_election=lead, end_date="" if pd.isna(r.end_date) else pd.Timestamp(r.end_date).date().isoformat(),
                in_f0_training_bank=str(r.race_id) in bank_ids, within_existing_0_100_age_rule=in_range_flag,
                selected_by_exact_fixed_bin_operator=chosen_flag,
                disposition="selected_latest_in_14day_bin" if chosen_flag else
                            "superseded_in_14day_bin" if in_range_flag else "outside_existing_0_100_age_rule",
            ))
    pd.DataFrame(summaries).to_csv(OUT / "historical-calibration-baseline.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "historical-calibration-baseline-selection-rows.csv", index=False)
    receipt = dict(
        schema="MomentumHistoricalCalibrationCoverageLunaV1",
        root_selection_sha256=sha(SELECTION), f0_support_manifest_sha256=sha(BANK_COMPAT),
        f0_prepared_plan_sha256=sha(F0_PLAN), historical_panel_sha256=sha(HIST),
        standard_runner_sha256=sha(RUNNER), cycles_by_f0_fold="strictly earlier among 2012/2016/2018/2020",
        exact_existing_selection="days_to_election in [0,100]; lead_bin=floor(days/14); sort end_date, observation_id; last per cycle/race/pollster/bin.",
        availability_proxy="field-end date only; original publication time not present; strict_asof_qualified=false",
        no_fit=True, no_score=True, no_optimization=True,
        outputs=["historical-calibration-baseline.csv", "historical-calibration-baseline-selection-rows.csv",
                 "historical-calibration-baseline-addendum.md"])
    (OUT / "historical-calibration-baseline-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lines = [
        "# Exact historical calibration baseline supplement", "",
        "This is the unchanged historical polling-calibration selection, shown separately from the H30/H14/H7 cutoff-truncated coverage sensitivities in REPORT.md. The original runner uses every earlier-cycle panel observation with lead in [0,100] days, then retains the latest by end date and observation ID in each cycle/race/pollster/14-day lead bin. It does not apply the target-cycle H30/H14/H7 cutoff to historical training cycles.", "",
        "The fold sets are strictly earlier: 2018 uses 2012/2016; 2020 uses 2012/2016/2018; 2022 and 2024 use 2012/2016/2018/2020. The 2022 and 2024 F0 folds have the same historical panel years because this source ends at 2020. This is coverage only, not a covariance fit or forecast.", "",
        "Coverage counts and within-pollster time-span support are in historical-calibration-baseline.csv. Every row ID, fixed bin, selected/excluded disposition, and F0 training-race membership is in historical-calibration-baseline-selection-rows.csv. Source and runner hashes are pinned in the receipt. Historical availability is only field-end proxy; original publication time and exact vintage are unavailable.", "",
    ]
    for r in summaries:
        if r["training_cycle_group"] == "all_earlier_cycles":
            lines.append(f"- {r['target_cycle']} fold ({r['training_cycles']}): {r['raw_eligible_rows_0_100']} raw rows to {r['fixed_bin_selected_rows']} fixed-bin rows; {r['raw_unique_races']} races with raw polls, {r['fixed_bin_unique_races']} with retained bins; F0 training race intersection {r['f0_training_races_with_selected_bin_poll']}/{r['f0_training_races_total']}.")
    (OUT / "historical-calibration-baseline-addendum.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
