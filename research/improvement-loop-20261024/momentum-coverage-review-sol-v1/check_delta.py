"""Independent repair reconstruction from raw historical rows; no fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "momentum-preflight-coverage-luna-v1"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert (
    digest(SOURCE / "artifact-correction-manifest-v2.json")
    == "2ed7cefbb8078bbba1a59a717a8e98ef56e7627dabb7a0574a1a6aa45081b715"
)
assert (
    digest(SOURCE / "authoritative-index-v2.json") == "1fb3dd6905da7bfab2bd351dc486a3f525cc9f49c2da0bce9029d0f4d0c0c7da"
)
manifest = json.loads((SOURCE / "artifact-correction-manifest-v2.json").read_bytes())
for item in manifest["files"]:
    p = SOURCE / item["path"]
    assert digest(p) == item["sha256"] and p.stat().st_size == item["bytes"]
hist = pd.read_csv(
    OWNER / "research/polling-layer-implementation-20261020/historical-intake-sol/derived/historical_poll_panel.csv"
)
hist["end_date"] = pd.to_datetime(hist.end_date)
hist["pollster_key"] = hist.pollster.str.strip().str.casefold()
compat = json.loads((HERE.parent / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json").read_bytes())
banks = {b["year"]: set(b["training_race_ids"]) for b in compat["banks"] if b["arm"] == "F0"}
ledger = pd.read_csv(SOURCE / "historical-calibration-baseline-id-ledger-v2.csv")
summary = pd.read_csv(SOURCE / "historical-calibration-baseline.csv")
fixed = pd.read_csv(SOURCE / "historical-fixed-bin-horizon-correction-v2.csv")
first_pass = pd.read_csv(SOURCE / "operator-selection-rows.csv", low_memory=False)
first_pass = first_pass[
    first_pass.operator.eq("tierney_historical_latest_per_14day_bin") & first_pass.selected.eq(True)
]
baseline_checks = horizon_checks = 0


def repeats(frame):
    groups = [g for _, g in frame.groupby(["race_id", "pollster_key"]) if g.end_date.nunique() >= 2]
    spans = [(g.end_date.max() - g.end_date.min()).days for g in groups]
    return dict(
        repeated_race_pollster_groups=len(groups),
        races_with_repeated_pollster=len({g.race_id.iloc[0] for g in groups}),
        repeated_group_rows=sum(map(len, groups)),
        median_repeated_group_span_days=float(np.median(spans)) if spans else None,
    )


def same(actual, expected):
    for field, value in expected.items():
        if value is None:
            assert pd.isna(actual[field]), field
        else:
            assert float(actual[field]) == float(value), (field, actual[field], value)


for year in (2018, 2020, 2022, 2024):
    source = hist[hist.cycle.lt(year)].copy()
    a = source[source.days_to_election.between(0, 100)].copy()
    a["bin"] = a.days_to_election // 14
    b = a.sort_values(["end_date", "observation_id"]).drop_duplicates(
        ["cycle", "race_id", "pollster_key", "bin"], keep="last"
    )
    d = ledger[ledger.target_cycle.eq(year)]
    assert len(d) == len(source) and set(d.observation_id) == set(source.observation_id)
    assert not d.observation_id.duplicated().any()
    assert set(d.loc[d.selected_by_exact_fixed_bin_operator, "observation_id"]) == set(b.observation_id)
    assert set(d.loc[d.within_existing_0_100_age_rule, "observation_id"]) == set(a.observation_id)
    assert set(d.loc[d.in_f0_training_bank, "race_id"]) == set(source.race_id) & banks[year]
    for group, aa, bb in [("all_earlier_cycles", a, b)] + [
        (str(c), a[a.cycle.eq(c)], b[b.cycle.eq(c)]) for c in sorted(source.cycle.unique())
    ]:
        row = summary[summary.target_cycle.eq(year) & summary.training_cycle_group.eq(group)].iloc[0]
        rr = repeats(aa)
        expected = dict(
            raw_eligible_rows_0_100=len(aa),
            fixed_bin_selected_rows=len(bb),
            raw_unique_races=aa.race_id.nunique(),
            fixed_bin_unique_races=bb.race_id.nunique(),
            raw_unique_observations=aa.observation_id.nunique(),
            fixed_bin_unique_observations=bb.observation_id.nunique(),
            raw_named_pollsters=aa.pollster_key.nunique(),
            fixed_bin_named_pollsters=bb.pollster_key.nunique(),
            raw_repeated_race_pollster_groups=rr["repeated_race_pollster_groups"],
            median_raw_repeat_span_days=rr["median_repeated_group_span_days"],
            f0_training_races_total=len(banks[year]),
            f0_training_races_with_any_raw_panel_poll=len(set(aa.race_id) & banks[year]),
            f0_training_races_with_selected_bin_poll=len(set(bb.race_id) & banks[year]),
        )
        same(row, expected)
        baseline_checks += 1
    for horizon in (30, 14, 7):
        for lag in (0, 1, 3, 7):
            a = source[source.days_to_election.between(horizon + lag, 100)].copy()
            a["bin"] = a.days_to_election // 14
            b = a.sort_values(["end_date", "observation_id"]).drop_duplicates(
                ["cycle", "race_id", "pollster_key", "bin"], keep="last"
            )
            old = first_pass[
                first_pass.target_cycle.eq(year)
                & first_pass.horizon_days.eq(horizon)
                & first_pass.extra_lag_days_beyond_native_proxy.eq(lag)
            ]
            assert set(old.observation_id) == set(b.observation_id)
            for operator, part in [
                ("tierney_cutoff_full_series_sensitivity", a),
                ("tierney_cutoff_fixed_14day_bin_sensitivity", b),
            ]:
                row = fixed[
                    fixed.target_cycle.eq(year)
                    & fixed.horizon_days.eq(horizon)
                    & fixed.extra_lag_days_beyond_native_proxy.eq(lag)
                    & fixed.operator.eq(operator)
                ].iloc[0]
                same(
                    row,
                    dict(
                        rows=len(part),
                        unique_samples=part.sample_id.nunique(),
                        races=part.race_id.nunique(),
                        pollsters_known=part.pollster_key.nunique(),
                        f0_training_races_supported=len(set(part.race_id) & banks[year]),
                        **repeats(part),
                    ),
                )
                horizon_checks += 1
result = dict(
    manifest_sha256=digest(SOURCE / "artifact-correction-manifest-v2.json"),
    authority_index_sha256=digest(SOURCE / "authoritative-index-v2.json"),
    manifest_artifacts_verified=len(manifest["files"]),
    baseline_ledger_records=len(ledger),
    baseline_summaries_recomputed=baseline_checks,
    historical_operator_cells_recomputed=horizon_checks,
    prior_findings_preserved=True,
    remaining_defects=[],
    sampler=False,
    optimizer=False,
    forecast_scores=False,
)
(HERE / "delta-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
