"""Independent H14 selection/feature audit; no numeric truth or posterior draws."""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
SOURCE = LAB / "momentum-target-preparation-sol-v1"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PINS = {}


def checked(path, expected=None):
    data = Path(path).read_bytes()
    value = hashlib.sha256(data).hexdigest()
    if expected:
        assert value == expected, str(path)
    PINS[str(path)] = value
    return data


def main(expected):
    manifest = json.loads(checked(SOURCE / "artifact-manifest.json", expected))
    for item in manifest["files"]:
        data = checked(SOURCE / item["path"], item["sha256"])
        assert len(data) == item["bytes"]
    bindings = json.loads((SOURCE / "source-bindings.json").read_bytes())
    for i, item in enumerate(bindings["files"]):
        checked(item["path"], item["sha256"])
        checked(SOURCE / "snapshots" / f"{i:02d}-{Path(item['path']).name}", item["sha256"])
    sys.path.insert(0, str(OWNER))
    import numpy as np
    import pandas as pd
    from scipy.special import logit
    from src.forecasting.senate_data import senate_polls_as_of

    base = OWNER / "research/polling-layer-implementation-20261020"
    archive = OWNER / ".redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv"
    raw = pd.read_csv(archive, low_memory=False)
    archive_bytes = archive.read_bytes()
    physical_lines = archive_bytes.splitlines(keepends=True)
    csv_source = csv.DictReader(io.StringIO(archive_bytes.decode(), newline=""))
    _ = csv_source.fieldnames
    previous_line = csv_source.line_num
    raw_record_map = {}
    for number, record in enumerate(csv_source, 2):
        end_line = csv_source.line_num
        raw_record_map[number] = (record, previous_line + 1, end_line)
        previous_line = end_line
    for key in ("start_date", "end_date", "created_at"):
        raw[key] = pd.to_datetime(raw[key], format="mixed", errors="coerce")
    roster = pd.concat(
        [
            pd.read_csv(
                base / "full-chamber-intake-sol/event-roster-2018-2022.csv",
                usecols=[
                    "cycle",
                    "race_id",
                    "source_race_id",
                    "eligible",
                    "special",
                    "election_date",
                    "dem_candidate_id",
                    "rep_candidate_id",
                ],
            ),
            pd.read_csv(
                base / "cross-context-intake-sol/roster-2020-2024.csv",
                usecols=[
                    "cycle",
                    "race_id",
                    "source_race_id",
                    "eligible",
                    "special",
                    "election_date",
                    "dem_candidate_id",
                    "rep_candidate_id",
                ],
            ),
        ],
        ignore_index=True,
    )
    roster = roster[roster.eligible.eq(True) & roster.special.eq(False)].copy()
    for key in ("source_race_id", "dem_candidate_id", "rep_candidate_id"):
        roster[key] = roster[key].map(lambda x: str(int(float(x))))
    features = pd.read_csv(OWNER / "research/fundamentals-iteration-20261020/panel-sol/features/all-cycles.csv")
    # No numeric outcome column is requested or read into a scientific dataframe.
    identity = pd.read_csv(
        OWNER / "research/fundamentals-iteration-20261020/panel-sol/labels/outcomes-SEPARATE.csv",
        usecols=["cycle", "race_id"],
    )
    compat = json.loads((LAB / "momentum-preflight-bank-compatibility-sol-v1/compatibility.json").read_bytes())
    banks = {b["year"]: b for b in compat["banks"] if b["arm"] == "F0"}
    plan = json.loads((SOURCE / "prepared-v1/plan.json").read_bytes())
    coverage = pd.read_csv(LAB / "momentum-preflight-coverage-luna-v1/operator-selection-rows.csv", low_memory=False)
    preflight_exclusions = pd.read_csv(
        LAB / "momentum-preflight-coverage-luna-v1/selection-exclusions.csv", low_memory=False
    )
    records = []
    expected_counts = {2018: (202, 170, 25), 2020: (190, 134, 22), 2022: (220, 171, 23), 2024: (351, 226, 27)}
    for fold in plan["folds"]:
        year = fold["cycle"]
        rr = roster[roster.cycle.eq(year)]
        election = pd.Timestamp(rr.election_date.iloc[0])
        cutoff = election - pd.Timedelta(days=14)
        assert fold["cutoff_utc"] == cutoff.strftime("%Y-%m-%dT00:00:00Z")
        proxy = raw.created_at.dt.normalize() + pd.Timedelta(days=1)
        permitted = raw[raw.cycle.eq(year) & raw.end_date.le(cutoff) & raw.created_at.notna() & proxy.le(cutoff)]
        f, _ = senate_polls_as_of(permitted, rr, cycle=year, cutoff=cutoff)
        f["end_date"] = pd.to_datetime(f.end_date)
        f["available_date"] = pd.to_datetime(f.available_date)
        f["age"] = (election - f.end_date).dt.days
        all_helper = f.copy()
        f = f[f.age.between(14, 100)].copy()

        def firm(r):
            label = "" if pd.isna(r.pollster) else str(r.pollster).strip().casefold()
            return "__unknown_sample__:" + r.sample_id if label in ("", "none", "nan", "unknown", "<na>") else label

        f["key"] = [firm(r) for r in f.itertuples()]
        latest = f.sort_values(["end_date", "available_date", "sample_id", "observation_id"]).drop_duplicates(
            ["race_id", "key"], keep="last"
        )
        latest = f[f.observation_id.isin(latest.observation_id)]
        assert (len(f), len(latest), f.race_id.nunique()) == expected_counts[year]
        for name, operator, part in [
            ("native_full_series", "538_native_full_series", f),
            ("latest_per_race_pollster", "538_latest_per_race_pollster", latest),
        ]:
            item = fold["operators"][name]
            checked(item["joint_input"]["path"], item["joint_input"]["sha256"])
            data = json.loads(Path(item["joint_input"]["path"]).read_bytes())
            observations = pd.read_csv(item["observations"]["path"])
            assert data["observation_ids"] == part.observation_id.tolist()
            assert data["sample_ids"] == part.sample_id.tolist() and data["race_ids"] == part.race_id.tolist()
            assert data["pollster_keys"] == part.key.tolist() and set(data["cycles"]) == {str(year)}
            assert data["days_to_election"] == part.age.tolist()
            np.testing.assert_array_equal(data["shares"], part.dem_share)
            np.testing.assert_array_equal(
                data["logits"], np.log(part.dem_share.to_numpy()) - np.log1p(-part.dem_share.to_numpy())
            )
            assert observations.question_id.astype(str).tolist() == part.question_id.astype(str).tolist()
            assert observations.population.tolist() == part.population.tolist()
            previous = coverage[
                coverage.source_panel.eq("538_target_archive")
                & coverage.cycle.eq(year)
                & coverage.horizon_days.eq(14)
                & coverage.extra_lag_days_beyond_native_proxy.eq(0)
                & coverage.operator.eq(operator)
                & coverage.selected.eq(True)
            ]
            assert previous.observation_id.tolist() == part.observation_id.tolist()
            assert [str(int(float(x))) for x in previous.question_id] == [str(int(float(x))) for x in part.question_id]
            assert data["race_indices"] == [fold["bank_race_ids"].index(r) for r in part.race_id]
        target = features[features.cycle.eq(year)]
        bank = banks[year]
        assert target.race_id.tolist() == bank["target_race_ids"] == fold["bank_race_ids"]
        n = len(target)
        offset = np.zeros(n)
        design = np.zeros((n, 6))
        state = logit(target.prior_presidential_dem_share.to_numpy(float))
        national = logit(target.prior_national_presidential_dem_share.to_numpy(float))
        party = target.president_party_asof.to_numpy(float)
        values = [
            np.ones(n),
            target.dem_incumbent.to_numpy(float) - target.rep_incumbent.to_numpy(float),
            state - national,
            national,
            party,
            party * target.midterm.to_numpy(float),
        ]
        for j, value in enumerate(values):
            design[:, j] += value
        matrix = json.loads(Path(fold["target_design"]["path"]).read_bytes())
        np.testing.assert_array_equal(matrix["offset"], offset)
        np.testing.assert_array_equal(matrix["design"], design)
        assert (
            hashlib.sha256(design.tobytes()).hexdigest()
            == bank["target_design_sha256"]
            == matrix["design_array_sha256"]
        )
        assert (
            hashlib.sha256(offset.tobytes()).hexdigest()
            == bank["target_offset_sha256"]
            == matrix["offset_array_sha256"]
        )
        assert (
            matrix["event_ids"] == target.event_id.tolist() and matrix["coefficient_names"] == bank["coefficient_names"]
        )
        metadata = pd.read_csv(fold["outcome_identity_metadata"]["path"])
        assert set(metadata.columns) == {"cycle", "race_id"} and set(metadata.race_id) == set(
            identity.loc[identity.cycle.eq(year), "race_id"]
        ) == set(fold["bank_race_ids"])
        mask = pd.read_csv(fold["full_and_polled_mask"]["path"])
        assert mask.race_id.tolist() == fold["bank_race_ids"]
        assert set(mask.loc[mask.in_common_polled_mask, "race_id"]) == set(f.race_id) == set(latest.race_id)
        assert set(f.race_id) <= set(fold["bank_race_ids"])
        bankref = json.loads(Path(fold["bank_reference"]["path"]).read_bytes())
        assert (
            bankref["raw_banks"] == bank["raw_banks"]
            and bankref["row_count"] == 16000
            and bankref["bank_exported"] is False
        )
        selection = pd.read_csv(fold["selection_ledger"]["path"])
        assert set(selection.observation_id) == set(all_helper.observation_id)
        assert set(selection.loc[selection.selected_native_full, "observation_id"]) == set(f.observation_id)
        assert set(selection.loc[selection.selected_latest, "observation_id"]) == set(latest.observation_id)
        exclusions = pd.read_csv(fold["exclusion_ledger"]["path"])
        old = preflight_exclusions[
            preflight_exclusions.source_panel.eq("538_target_archive")
            & preflight_exclusions.cycle.eq(year)
            & preflight_exclusions.horizon_days.eq(14)
            & preflight_exclusions.extra_lag_days_beyond_native_proxy.eq(0)
        ]
        keys = ["operator", "observation_id", "exclusion_reason"]
        assert sorted(map(tuple, exclusions[keys].to_numpy())) == sorted(map(tuple, old[keys].to_numpy()))
        with Path(fold["raw_provenance"]["path"]).open() as stream:
            provenance = list(csv.DictReader(stream))
        expected_raw = {number for number, (rawrow, _, _) in raw_record_map.items() if int(rawrow["cycle"]) == year}
        assert {int(row["csv_record_number"]) for row in provenance} == expected_raw
        selected_questions = set(zip(f.poll_id.astype(str), f.question_id.astype(str)))
        for row in provenance:
            original, start, end = raw_record_map[int(row["csv_record_number"])]
            assert (int(row["physical_start_line"]), int(row["physical_end_line"])) == (start, end)
            assert row["raw_record_sha256"] == hashlib.sha256(b"".join(physical_lines[start - 1 : end])).hexdigest()
            for outname, inname in [
                ("poll_id", "poll_id"),
                ("question_id", "question_id"),
                ("candidate_id", "candidate_id"),
                ("candidate_name", "candidate_name"),
                ("pct_raw", "pct"),
                ("population_raw", "population"),
            ]:
                assert row[outname] == original[inname]
            assert (row["selected_question_in_native_full"] == "True") == (
                (original["poll_id"], original["question_id"]) in selected_questions
            )
        records.append(
            dict(
                cycle=year,
                full=len(f),
                latest=len(latest),
                polled_races=f.race_id.nunique(),
                complete_F0_races=n,
                population_counts={str(k): int(v) for k, v in f.population.value_counts().items()},
                exact_questions_shares_ages_and_event_axes=True,
                F0_array_hashes_match=True,
                raw_candidate_answer_records_verified=len(provenance),
                helper_output_exclusions_verified=len(exclusions),
                outcome_projection_columns=["cycle", "race_id"],
            )
        )
    result = dict(
        manifest_sha256=expected,
        artifact_count=len(manifest["files"]),
        source_snapshot_pairs=len(bindings["files"]),
        inspected_pins=PINS,
        folds=records,
        numeric_target_outcome_columns_read=False,
        original_bank_arrays_loaded=False,
        optimizer=False,
        sampler=False,
        forecast_scores=False,
    )
    (HERE / "reconstruction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["artifact_count", "source_snapshot_pairs", "folds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256", required=True)
    main(parser.parse_args().manifest_sha256)
