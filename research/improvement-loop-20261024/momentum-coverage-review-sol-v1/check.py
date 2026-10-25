"""Read-only coverage reconstruction. No optimizer, sampler or outcome scores."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "momentum-preflight-coverage-luna-v1"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PINS = {}


def checked(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected:
        assert digest == expected
    PINS[str(path)] = digest
    return data


for name, digest in [
    ("artifact-manifest.json", "99ba073742bd329cad8fe4713b2594c205dd1d115ed8cfe6c81c5b316594d574"),
    ("artifact-correction-manifest.json", "b39a7f14b96ccdf96c501c952a85a6ae67e388d289aa7ac0da91257bd631e746"),
]:
    manifest = json.loads(checked(SOURCE / name, digest))
    for item in manifest["files"]:
        body = checked(SOURCE / item["path"], item["sha256"])
        assert len(body) == item["bytes"]
receipt = json.loads((SOURCE / "receipt.json").read_bytes())
for item in receipt["source_pins"]:
    checked(item["path"], item["sha256"])
sys.path.insert(0, str(OWNER))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.forecasting.senate_data import senate_polls_as_of  # noqa: E402

BASE = OWNER / "research/polling-layer-implementation-20261020"
raw = pd.read_csv(OWNER / ".redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv", low_memory=False)
for field in ["start_date", "end_date", "created_at"]:
    raw[field] = pd.to_datetime(raw[field], format="mixed", errors="coerce")
roster = pd.concat(
    [
        pd.read_csv(BASE / "full-chamber-intake-sol/event-roster-2018-2022.csv"),
        pd.read_csv(BASE / "cross-context-intake-sol/roster-2020-2024.csv"),
    ],
    ignore_index=True,
)
roster = roster[roster.eligible.eq(True) & roster.special.eq(False)].copy()
for field in ["source_race_id", "dem_candidate_id", "rep_candidate_id"]:
    roster[field] = roster[field].map(lambda x: str(int(float(x))) if pd.notna(x) else "")
rows = pd.read_csv(SOURCE / "operator-selection-rows.csv", low_memory=False)
cells = pd.read_csv(SOURCE / "coverage-cells.csv")
native = rows[rows.source_panel.eq("538_target_archive")].copy()
for field in ["start_date", "end_date"]:
    native[field] = pd.to_datetime(native[field], errors="coerce")
compat = json.loads(
    checked(
        HERE.parent / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json",
        "53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05",
    )
)
banks = {b["year"]: b for b in compat["banks"] if b["arm"] == "F0"}
target_checks = 0
pair_keys = []
for year in (2018, 2020, 2022, 2024):
    rr = roster[roster.cycle.eq(year)]
    election = pd.Timestamp(rr.election_date.iloc[0])
    for horizon in (30, 14, 7):
        cutoff = election - pd.Timedelta(days=horizon)
        for lag in (0, 1, 3, 7):
            proxy = raw.created_at.dt.normalize() + pd.Timedelta(days=1 + lag)
            filtered = raw[raw.cycle.eq(year) & raw.end_date.le(cutoff) & raw.created_at.notna() & proxy.le(cutoff)]
            f, _ = senate_polls_as_of(filtered, rr, cycle=year, cutoff=cutoff)
            f["end_date"] = pd.to_datetime(f.end_date)
            f["available_date"] = pd.to_datetime(f.available_date)
            lead = (election - f.end_date).dt.days
            f = f[lead.between(horizon, 100)].copy()

            def key(r):
                value = "" if pd.isna(r.pollster) else str(r.pollster).strip().casefold()
                return "__unknown_sample__:" + r.sample_id if value in ("", "nan", "none", "unknown", "<na>") else value

            f["pollster_key"] = [key(r) for r in f.itertuples()]
            latest = f.sort_values(["end_date", "available_date", "sample_id", "observation_id"]).drop_duplicates(
                ["race_id", "pollster_key"], keep="last"
            )
            for operator, part in [("538_native_full_series", f), ("538_latest_per_race_pollster", latest)]:
                ledger = native[
                    native.cycle.eq(year)
                    & native.horizon_days.eq(horizon)
                    & native.extra_lag_days_beyond_native_proxy.eq(lag)
                    & native.operator.eq(operator)
                    & native.selected.eq(True)
                ]
                assert set(ledger.observation_id) == set(part.observation_id)
                s = cells[
                    cells.cycle.eq(year)
                    & cells.horizon_days.eq(horizon)
                    & cells.extra_lag_days_beyond_native_proxy.eq(lag)
                    & cells.operator.eq(operator)
                ].iloc[0]
                groups = [g for _, g in ledger.groupby(["race_id", "pollster_key"]) if g.end_date.nunique() >= 2]
                expected = dict(
                    rows=len(part),
                    unique_samples=part.sample_id.nunique(),
                    races=part.race_id.nunique(),
                    repeated_race_pollster_groups=len(groups),
                    races_with_repeated_pollster=len({g.race_id.iloc[0] for g in groups}),
                    repeated_group_rows=sum(map(len, groups)),
                    bank_races_supported=len(set(part.race_id) & set(banks[year]["target_race_ids"])),
                )
                spans = [(g.end_date.max() - g.end_date.min()).days for g in groups]
                race_spans = [
                    (g.end_date.max() - g.end_date.min()).days for _, g in ledger.groupby("race_id") if len(g) >= 2
                ]
                expected["pollsters_known"] = ledger.loc[
                    ~ledger.pollster_key.str.startswith("__unknown_sample__:"), "pollster_key"
                ].nunique()
                expected["interval_rows_known"] = int((ledger.start_date.notna() & ledger.end_date.notna()).sum())
                for field, values in [
                    ("median_repeated_group_span_days", spans),
                    ("median_race_time_span_days", race_spans),
                ]:
                    if values:
                        assert float(s[field]) == float(np.median(values))
                    else:
                        assert pd.isna(s[field])
                overlapping = comparable = 0
                for _, g in ledger.groupby(["race_id", "pollster_key"]):
                    dates = list(g.dropna(subset=["start_date", "end_date"]).itertuples())
                    for i, a in enumerate(dates):
                        for b in dates[i + 1 :]:
                            comparable += 1
                            overlapping += a.start_date <= b.end_date and b.start_date <= a.end_date
                expected.update(interval_overlap_pairs=overlapping, interval_comparable_pairs=comparable)
                for field, value in expected.items():
                    assert int(s[field]) == value, (year, horizon, lag, operator, field)
                target_checks += 1
            for (race, firm), g in f.groupby(["race_id", "pollster_key"]):
                records = list(g.sort_values(["start_date", "end_date", "observation_id"]).itertuples())
                for i, a in enumerate(records):
                    for b in records[i + 1 :]:
                        if (
                            pd.notna(a.start_date)
                            and pd.notna(b.start_date)
                            and a.start_date <= b.end_date
                            and b.start_date <= a.end_date
                        ):
                            pair_keys.append((year, horizon, lag, race, firm, a.observation_id, b.observation_id))
print("target helper/coverage cells verified", target_checks, flush=True)
support = pd.read_csv(SOURCE / "f0-race-support.csv")
for year in (2018, 2020, 2022, 2024):
    r = set(roster.loc[roster.cycle.eq(year), "race_id"])
    b = set(banks[year]["target_race_ids"])
    exported = support[support.cycle.eq(year)]
    assert set(exported.race_id) == r | b
    for row in exported.itertuples():
        assert row.exact_v2_poll_roster == (row.race_id in r)
        assert row.exact_f0_bank_target == (row.race_id in b)
        assert row.in_intersection == (row.race_id in r & b)
pairs = pd.read_csv(SOURCE / "field-interval-overlap-pairs.csv")
assert set(pair_keys) == set(
    zip(
        pairs.cycle,
        pairs.horizon_days,
        pairs.extra_lag_days,
        pairs.race_id,
        pairs.pollster_key,
        pairs.observation_a,
        pairs.observation_b,
    )
)
hist = pd.read_csv(BASE / "historical-intake-sol/derived/historical_poll_panel.csv")
hist["end_date"] = pd.to_datetime(hist.end_date)
hist["pollster_key"] = hist.pollster.str.strip().str.casefold()
summary = pd.read_csv(SOURCE / "historical-calibration-baseline.csv")
detail = pd.read_csv(SOURCE / "historical-calibration-baseline-selection-rows.csv")
correction = pd.read_csv(SOURCE / "historical-fixed-bin-horizon-correction.csv")
correct_ids = pd.read_csv(SOURCE / "historical-fixed-bin-horizon-selection.csv")
historical_checks = 0
baseline_gaps = []
repeat_mismatches = []
for year in (2018, 2020, 2022, 2024):
    source = hist[hist.cycle.lt(year)]
    admitted = source[source.days_to_election.between(0, 100)].copy()
    admitted["bin"] = admitted.days_to_election // 14
    selected = admitted.sort_values(["end_date", "observation_id"]).drop_duplicates(
        ["cycle", "race_id", "pollster_key", "bin"], keep="last"
    )
    s = summary[summary.target_cycle.eq(year) & summary.training_cycle_group.eq("all_earlier_cycles")].iloc[0]
    assert s.raw_eligible_rows_0_100 == len(admitted) and s.fixed_bin_selected_rows == len(selected)
    d = detail[detail.target_cycle.eq(year)]
    baseline_gaps.append(
        dict(
            year=year,
            expected_source_rows=len(source),
            recorded_source_rows=len(d),
            missing_ids=sorted(set(source.observation_id) - set(d.observation_id)),
        )
    )
    for horizon in (30, 14, 7):
        for lag in (0, 1, 3, 7):
            a = source[source.days_to_election.between(horizon + lag, 100)].copy()
            a["bin"] = a.days_to_election // 14
            b = a.sort_values(["end_date", "observation_id"]).drop_duplicates(
                ["cycle", "race_id", "pollster_key", "bin"], keep="last"
            )
            exported = correct_ids[
                correct_ids.target_cycle.eq(year)
                & correct_ids.horizon_days.eq(horizon)
                & correct_ids.extra_lag_days_beyond_native_proxy.eq(lag)
            ]
            assert set(exported.observation_id) == set(source.observation_id)
            assert set(exported.loc[exported.selected_by_corrected_fixed_bin, "observation_id"]) == set(
                b.observation_id
            )
            for operator, part in [
                ("tierney_cutoff_full_series_sensitivity", a),
                ("tierney_cutoff_fixed_14day_bin_sensitivity", b),
            ]:
                s = correction[
                    correction.target_cycle.eq(year)
                    & correction.horizon_days.eq(horizon)
                    & correction.extra_lag_days_beyond_native_proxy.eq(lag)
                    & correction.operator.eq(operator)
                ].iloc[0]
                assert s.rows == len(part) and s.races == part.race_id.nunique()
                groups = [g for _, g in part.groupby(["race_id", "pollster_key"]) if g.end_date.nunique() >= 2]
                if len(groups) != s.repeated_race_pollster_groups:
                    repeat_mismatches.append(
                        dict(
                            year=year,
                            horizon=horizon,
                            lag=lag,
                            operator=operator,
                            reported=int(s.repeated_race_pollster_groups),
                            actual=len(groups),
                        )
                    )
                historical_checks += 1
all_admitted = hist[hist.days_to_election.between(0, 100)].copy()
all_admitted["bin"] = all_admitted.days_to_election // 14
all_selected = all_admitted.sort_values(["end_date", "observation_id"]).drop_duplicates(
    ["cycle", "race_id", "pollster_key", "bin"], keep="last"
)
disjoint_path = (
    OWNER
    / "research/senate-stack-integration-20261020/tx-disjoint-calibration-sol-v1"
    / "attempt-v1/selected-disjoint-training.csv"
)
checked(disjoint_path)
disjoint = pd.read_csv(disjoint_path)
removed = all_selected[~all_selected.observation_id.isin(disjoint.observation_id)]
assert len(all_selected) == 1514 and len(disjoint) == 1483 and len(removed) == 31
assert set(removed.race_id) == {"2020:TX:senate:regular"}
assert set(disjoint.observation_id) <= set(all_selected.observation_id)
result = dict(
    inspected_pins=PINS,
    target_cells_verified=target_checks,
    historical_sensitivity_cells_verified=historical_checks,
    overlap_pair_rows_verified=len(pair_keys),
    baseline_ledger_gaps=baseline_gaps,
    selected_repeat_metric_mismatches=repeat_mismatches,
    selected_MAP_rows=len(all_selected),
    actual_disjoint_rows=len(disjoint),
    removed_from_selected_MAP=removed[["observation_id", "race_id"]].to_dict("records"),
    sampler=False,
    optimizer=False,
    forecast_scoring=False,
)
(HERE / "recomputation.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    "PASS counts/IDs/overlaps; recorded defects:",
    len(repeat_mismatches),
    "metric cells and",
    len(baseline_gaps),
    "incomplete ledgers",
)
