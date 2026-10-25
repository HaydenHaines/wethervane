"""Exact advertised-field-interval overlap and ActiVote sensitivity receipt."""
import hashlib
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024")
INTEG = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
V2 = LAB / "momentum-audit-v2"
SELECTION = LAB / "checkpoints/momentum-preflight-root-selection-v1.json"
BANK = LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json"
ROWS = OUT / "operator-selection-rows.csv"
ARCHIVE_SELECTED = V2 / "selected-polls.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs = {"field-interval-overlap-groups.csv", "field-interval-overlap-pairs.csv",
               "activote-rolling-intervals.csv", "field-interval-support-addendum.md",
               "field-interval-support-receipt.json"}
    if outputs.intersection(p.name for p in OUT.iterdir()):
        raise RuntimeError("refusing to overwrite field interval outputs")
    if sha(SELECTION) != "5de976e5facbc84aa9db48e64fcc43864d1b8741fc5d1ddc134cc2e0fa05e5e9":
        raise RuntimeError("root selection changed")
    d = pd.read_csv(ROWS, low_memory=False)
    d = d[(d.source_panel == "538_target_archive") &
          (d.operator == "538_native_full_series")].copy()
    d["start_date"] = pd.to_datetime(d.start_date, errors="coerce")
    d["end_date"] = pd.to_datetime(d.end_date, errors="coerce")
    group_rows, pair_rows = [], []
    for keys, g in d.groupby(["cycle", "horizon_days", "extra_lag_days_beyond_native_proxy",
                              "race_id", "pollster_key"], sort=True):
        cycle, horizon, lag, race, firm = keys
        g = g.dropna(subset=["start_date", "end_date"]).sort_values(
            ["start_date", "end_date", "observation_id"])
        recs = list(g.itertuples(index=False))
        overlaps = []
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if a.start_date <= b.end_date and b.start_date <= a.end_date:
                    overlaps.append((a, b))
        if len(recs) >= 2:
            group_rows.append(dict(cycle=int(cycle), horizon_days=int(horizon), extra_lag_days=int(lag),
                                   race_id=race, pollster_key=firm, repeated_field_interval_count=len(recs),
                                   distinct_end_dates=int(g.end_date.nunique()),
                                   first_field_start=g.start_date.min().date().isoformat(),
                                   last_field_end=g.end_date.max().date().isoformat(),
                                   any_advertised_interval_overlap=bool(overlaps),
                                   overlapping_interval_pairs=len(overlaps)))
        for a, b in overlaps:
            pair_rows.append(dict(cycle=int(cycle), horizon_days=int(horizon), extra_lag_days=int(lag),
                                  race_id=race, pollster_key=firm,
                                  observation_a=a.observation_id, sample_a=a.sample_id,
                                  start_a=a.start_date.date().isoformat(), end_a=a.end_date.date().isoformat(),
                                  observation_b=b.observation_id, sample_b=b.sample_id,
                                  start_b=b.start_date.date().isoformat(), end_b=b.end_date.date().isoformat(),
                                  interpretation="reported date ranges overlap; respondent/sample overlap unknown"))

    selected = pd.read_csv(ARCHIVE_SELECTED, usecols=[
        "cycle", "pollster", "observation_id", "sample_id", "question_id", "race_id",
        "start_date", "end_date", "available_date", "field_duration_days"
    ], low_memory=False)
    selected["start_date"] = pd.to_datetime(selected.start_date, errors="coerce")
    selected["end_date"] = pd.to_datetime(selected.end_date, errors="coerce")
    selected["available_date"] = pd.to_datetime(selected.available_date, errors="coerce")
    activote = selected[(selected.cycle.eq(2022)) & selected.pollster.str.contains("ActiVote", case=False, na=False)].copy()
    election = pd.Timestamp("2022-11-08")
    act_rows = []
    for r in activote.itertuples(index=False):
        for h in (30, 14, 7):
            cutoff = election - pd.Timedelta(days=h)
            for lag in (0, 1, 3, 7):
                field_ok = r.end_date <= cutoff
                proxy_ok = r.available_date + pd.Timedelta(days=lag) <= cutoff
                act_rows.append(dict(
                    cycle=2022, pollster=str(r.pollster), race_id=str(r.race_id),
                    observation_id=str(r.observation_id), sample_id=str(r.sample_id),
                    question_id=str(r.question_id),
                    field_start=r.start_date.date().isoformat(), field_end=r.end_date.date().isoformat(),
                    native_available_date_proxy=r.available_date.date().isoformat(),
                    reported_field_duration_days=int(r.field_duration_days), horizon_days=h,
                    extra_lag_days_beyond_native_proxy=lag, cutoff_utc=cutoff.strftime("%Y-%m-%dT00:00:00Z"),
                    field_end_eligible=bool(field_ok), proxy_eligible=bool(proxy_ok),
                    eligible_for_horizon=bool(field_ok and proxy_ok),
                    reason="included_by_reported_dates_and_proxy" if field_ok and proxy_ok else
                           "field_end_after_horizon" if not field_ok else "availability_proxy_plus_lag_after_horizon",
                    interview_date_distribution_verified=False, respondent_independence_verified=False))

    groups = pd.DataFrame(group_rows)
    pairs = pd.DataFrame(pair_rows)
    acts = pd.DataFrame(act_rows)
    groups.to_csv(OUT / "field-interval-overlap-groups.csv", index=False)
    pairs.to_csv(OUT / "field-interval-overlap-pairs.csv", index=False)
    acts.to_csv(OUT / "activote-rolling-intervals.csv", index=False)

    lines = [
        "# Advertised field-interval overlap supplement", "",
        "The native full-series target view contains repeated samples from the same named pollster in the same race. This supplement checks whether their reported field-start/end intervals intersect. It does not determine respondent-level timing, repeated respondents, shared samples, or covariance.", "",
        "At zero additional lag, the repeated race/pollster groups and groups with at least one interval intersection by cycle/horizon are:",
    ]
    for c in (2018, 2020, 2022, 2024):
        for h in (30, 14, 7):
            x = groups[(groups.cycle == c) & (groups.horizon_days == h) &
                       (groups.extra_lag_days == 0)]
            lines.append(f"- {c} H{h}: {len(x)} repeated groups; {int(x.any_advertised_interval_overlap.sum())} group(s) with overlapping advertised intervals; {int(x.overlapping_interval_pairs.sum())} intersecting pair(s).")
    lines += [
        "",
        "Thus overlap is sparse in this archive: zero intersecting pairs at 2018 H30/H14/H7, 2020 H30/H14 and H7 has 1; zero at 2022 H30/H14/H7; 2024 H30/H14 zero and H7 has 1. The exact group/pair IDs and dates are preserved in the two CSVs.",
        "",
        "The 2022 selected 538 archive includes two 91-day ActiVote records. The Florida series (observation 538:80838:2022:FL:senate:regular, field 2022-06-27 through 2022-09-25, availability proxy 2022-09-29) passes the field-end and proxy filters at H30/H14/H7 under all four extra-lag values. The North Carolina series (538:81483:2022:NC:senate:regular, field 2022-08-05 through 2022-11-03, proxy 2022-11-04) fails the requested H30/H14/H7 cutoffs because its reported field end is after them. These are archive metadata classifications only; the long Florida interval is not expanded to daily polls or treated as known respondent dates.",
        "",
        "Distinct sample and observation IDs do not prove independent respondents. The counts are not a correction for overlap or an authenticated measurement covariance. A later likelihood decision must keep those limits visible.",
        "",
    ]
    (OUT / "field-interval-support-addendum.md").write_text("\n".join(lines))
    receipt = dict(
        schema="MomentumFieldIntervalSupportLunaV1",
        root_selection_sha256=sha(SELECTION), bank_compatibility_sha256=sha(BANK),
        operator_selection_rows_sha256=sha(ROWS), v2_selected_archive_rows_sha256=sha(ARCHIVE_SELECTED),
        strict_asof_qualified=False, interview_date_distribution_verified=False,
        respondent_overlap_or_independence_verified=False,
        outputs=[p for p in sorted(outputs)],
        primary_rule="Within each target cycle/horizon/extra-lag/race/pollster group, count pairs whose reported closed field-date ranges intersect.",
        note="Raw selected-polls export is read only for pollster, IDs, dates and duration; candidate outcome/share columns are not imported.")
    (OUT / "field-interval-support-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
