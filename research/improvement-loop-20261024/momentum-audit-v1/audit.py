"""Bounded descriptive Senate poll momentum audit. Writes only beside this script."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.forecasting.senate_data import normalize_candidate_name, senate_polls_as_of

BASE = ROOT / "research/polling-layer-implementation-20261020"
paths = []


def read(relative):
    path = ROOT / relative
    paths.append(path)
    return pd.read_csv(path, low_memory=False)


def ident(value):
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return str(value)


def counts(frame):
    return dict(
        n=len(frame),
        below=int((frame.performance_pp < -1).sum()),
        within=int(frame.performance_pp.between(-1, 1).sum()),
        above=int((frame.performance_pp > 1).sum()),
        mean_performance_pp=None if frame.empty else float(frame.performance_pp.mean()),
    )


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    paths.append(OUT / "plan.json")
    roster_a = read(str((BASE / "full-chamber-intake-sol/event-roster-2018-2022.csv").relative_to(ROOT)))
    roster_b = read(str((BASE / "cross-context-intake-sol/roster-2020-2024.csv").relative_to(ROOT)))
    roster = pd.concat([roster_a, roster_b], ignore_index=True)
    roster = roster[roster.cycle.isin(plan["cycles"]) & roster.eligible.eq(True) & roster.special.eq(False)].copy()
    assert not roster.race_id.duplicated().any()
    for col in ["source_race_id", "dem_candidate_id", "rep_candidate_id"]:
        roster[col] = roster[col].map(ident)
    outcomes = read(
        str((BASE / "historical-intake-sol/derived/outcomes_allowed_cycles_SEPARATE.csv").relative_to(ROOT))
    )
    outcomes = outcomes[outcomes.cycle.eq(2018)]
    for year in [2020, 2024]:
        outcomes = pd.concat(
            [outcomes, read(str((BASE / f"cross-context-intake-sol/outcomes-{year}-SEPARATE.csv").relative_to(ROOT)))],
            ignore_index=True,
        )
    candidates = read(str((BASE / "full-chamber-intake-sol/candidates.csv").relative_to(ROOT)))
    votes = read(
        str((BASE / "full-chamber-intake-sol/official-outcomes/all-events-candidate-votes.csv").relative_to(ROOT))
    )
    truth_2022 = []
    truth_joins = []
    for race in roster[roster.cycle.eq(2022)].itertuples():
        pair = []
        valid = True
        for party in ["dem", "rep"]:
            candidate_id = getattr(race, party + "_candidate_id")
            matched = candidates[
                candidates.race_id.eq(race.race_id) & candidates.group_id.eq("D" if party == "dem" else "R")
            ]
            if len(matched) != 1:
                valid = False
                truth_joins.append(
                    dict(race_id=race.race_id, party=party, accepted=False, reason="nonunique_official_group")
                )
                continue
            fec_id = matched.iloc[0].fec_candidate_id
            source = votes[votes.race_id.eq(race.race_id) & votes.fec_candidate_id.eq(fec_id)]
            if len(source) < 1 or not source.general_votes.notna().all():
                valid = False
                truth_joins.append(
                    dict(
                        race_id=race.race_id,
                        party=party,
                        accepted=False,
                        reason="official_FEC_id_has_no_complete_vote_rows",
                    )
                )
                continue
            name_equal = normalize_candidate_name(matched.iloc[0].candidate_name) == normalize_candidate_name(
                getattr(race, party + "_candidate_name")
            )
            valid = valid and name_equal
            pair.append(float(source.general_votes.sum()))
            truth_joins.append(
                dict(
                    race_id=race.race_id,
                    party=party,
                    roster_candidate_id=candidate_id,
                    roster_name=getattr(race, party + "_candidate_name"),
                    official_candidate_id=matched.iloc[0].candidate_id,
                    official_name=matched.iloc[0].candidate_name,
                    votes=pair[-1],
                    accepted=name_equal,
                    reason="" if name_equal else "full_name_mismatch_no_reviewed_alias",
                    join_basis="unique_official_event_group_plus_exact_lexical_fullname",
                )
            )
        if not valid or len(pair) != 2:
            continue
        assert min(pair) > 0
        truth_2022.append(
            dict(
                race_id=race.race_id, cycle=2022, actual_dem_share=pair[0] / sum(pair), election_date=race.election_date
            )
        )
    pd.DataFrame(truth_joins).to_csv(OUT / "truth-2022-event-party-joins.csv", index=False)
    outcomes = pd.concat([outcomes, pd.DataFrame(truth_2022)], ignore_index=True)
    assert not outcomes.race_id.duplicated().any()
    raw_path = ROOT / ".redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv"
    paths.append(raw_path)
    columns = [
        "poll_id",
        "question_id",
        "cycle",
        "state",
        "party",
        "pct",
        "sample_size",
        "population",
        "start_date",
        "end_date",
        "created_at",
        "race_id",
        "stage",
        "candidate_name",
        "pollster",
        "candidate_id",
        "election_date",
        "tracking",
        "hypothetical",
        "ranked_choice_reallocated",
        "methodology",
        "url",
        "partisan",
    ]
    raw = pd.read_csv(raw_path, usecols=columns, low_memory=False)
    selected, helper_audits = [], []
    for cycle in plan["cycles"]:
        election = pd.Timestamp(roster.loc[roster.cycle.eq(cycle), "election_date"].iloc[0])
        frame, audit = senate_polls_as_of(raw, roster, cycle=cycle, cutoff=election - pd.Timedelta(days=1))
        helper_audits.append(audit)
        selected.append(frame)
    selected = pd.concat(selected, ignore_index=True)
    # Native helper has already deduplicated samples with LV > RV > adults.
    selected = selected[selected.population.isin(["LV", "RV"])].copy()
    polls_before_truth_join = len(selected)
    selected = selected.merge(
        outcomes[["race_id", "actual_dem_share"]], on="race_id", validate="many_to_one", how="inner"
    )
    assert selected.actual_dem_share.between(0, 1).all()
    assert not selected[["race_id", "sample_id"]].duplicated().any()
    selected["end_lead"] = (selected.election_date - selected.end_date).dt.days
    selected["start_date"] = pd.to_datetime(selected.start_date)
    selected["mid_lead"] = (
        selected.election_date - (selected.start_date + (selected.end_date - selected.start_date) / 2)
    ).dt.total_seconds() / 86400
    selected["field_duration_days"] = (selected.end_date - selected.start_date).dt.days + 1
    selected["proxy_release_lag_days"] = (selected.available_date - selected.end_date).dt.days
    selected["explicit_partisan"] = ~selected.partisan.str.lower().isin(["nan", "", "none", "false"])
    selected.to_csv(OUT / "selected-polls.csv", index=False)
    race_rows, coverage_rows, fixed_rows = [], [], []
    for variant, lead, nonpartisan in [
        ("field_end", "end_lead", False),
        ("midpoint", "mid_lead", False),
        ("no_explicit_partisan", "end_lead", True),
        ("independent_earlier_trend", "end_lead", False),
    ]:
        for race in roster.itertuples():
            frame = selected[selected.race_id.eq(race.race_id)]
            if nonpartisan:
                frame = frame[~frame.explicit_partisan]
            late = frame[frame[lead].ge(1) & frame[lead].lt(8)]
            prior = frame[frame[lead].ge(8) & frame[lead].lt(15)]
            benchmark = late
            independent = variant == "independent_earlier_trend"
            if independent:
                late = prior
                prior = frame[frame[lead].ge(15) & frame[lead].lt(22)]
            sufficient = (
                len(benchmark) >= 2
                and benchmark.pollster.nunique() >= 2
                and len(late) >= 2
                and len(prior) >= 2
                and late.pollster.nunique() >= 2
                and prior.pollster.nunique() >= 2
            )
            coverage_rows.append(
                dict(
                    variant=variant,
                    cycle=race.cycle,
                    race_id=race.race_id,
                    late_samples=len(late),
                    prior_samples=len(prior),
                    late_pollsters=late.pollster.nunique(),
                    prior_pollsters=prior.pollster.nunique(),
                    sufficient=sufficient,
                    has_truth=race.race_id in set(outcomes.race_id),
                    exclusion_reason=""
                    if sufficient
                    else (
                        "missing_candidate_truth_join"
                        if race.race_id not in set(outcomes.race_id)
                        else "insufficient_window_sample_or_pollster_coverage"
                    ),
                )
            )
            if not sufficient:
                continue
            actual = float(benchmark.actual_dem_share.iloc[0])
            fixed_rows.append(
                dict(
                    variant=variant,
                    race_id=race.race_id,
                    cycle=race.cycle,
                    trend_dem_pp=100 * (late.dem_share.mean() - prior.dem_share.mean()),
                    performance_dem_pp=100 * (actual - benchmark.dem_share.mean()),
                )
            )
            modes = [
                (
                    "all_pollsters",
                    float(prior.dem_share.mean()),
                    float(late.dem_share.mean()),
                    float(benchmark.dem_share.mean()),
                    0,
                )
            ]
            shared = sorted(set(late.pollster) & set(prior.pollster) & set(benchmark.pollster))
            if len(shared) >= 2:
                modes.append(
                    (
                        "matched_pollsters",
                        float(prior[prior.pollster.isin(shared)].groupby("pollster").dem_share.mean().mean()),
                        float(late[late.pollster.isin(shared)].groupby("pollster").dem_share.mean().mean()),
                        float(benchmark[benchmark.pollster.isin(shared)].groupby("pollster").dem_share.mean().mean()),
                        len(shared),
                    )
                )
            for mode, first, last, benchmark_mean, shared_count in modes:
                trend = 100 * (last - first)
                if trend == 0:
                    continue
                sign = 1 if trend > 0 else -1
                race_rows.append(
                    dict(
                        variant=variant,
                        comparison=mode,
                        race_id=race.race_id,
                        cycle=race.cycle,
                        rising_party="D" if sign > 0 else "R",
                        trend_pp=abs(trend),
                        prior_dem_share=first,
                        late_dem_share=benchmark_mean,
                        trend_last_dem_share=last,
                        actual_dem_share=actual,
                        performance_pp=sign * 100 * (actual - benchmark_mean),
                        shared_pollsters=shared_count,
                    )
                )
    pd.DataFrame(fixed_rows).to_csv(OUT / "fixed-D-comparisons.csv", index=False)
    races = pd.DataFrame(race_rows)
    races.to_csv(OUT / "race-comparisons.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "coverage.csv", index=False)
    summaries = []
    for variant in sorted(set(row["variant"] for row in coverage_rows)):
        for comparison in ["all_pollsters", "matched_pollsters"]:
            group = races[races.variant.eq(variant) & races.comparison.eq(comparison)]
            for threshold in [0, 1, 2]:
                subset = group[group.trend_pp.ge(threshold)]
                for cycle in ["all"] + plan["cycles"]:
                    cell = subset if cycle == "all" else subset[subset.cycle.eq(cycle)]
                    summaries.append(
                        dict(
                            variant=variant, comparison=comparison, threshold_pp=threshold, cycle=cycle, **counts(cell)
                        )
                    )
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    metadata = []
    for cycle, frame in selected[selected.end_lead.between(1, 7)].groupby("cycle"):
        for field in ["end_lead", "mid_lead", "field_duration_days", "proxy_release_lag_days"]:
            series = frame[field].dropna()
            metadata.append(
                dict(
                    cycle=int(cycle),
                    field=field,
                    n=len(series),
                    minimum=float(series.min()),
                    median=float(series.median()),
                    p90=float(series.quantile(0.9)),
                    maximum=float(series.max()),
                )
            )
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paths.extend([ROOT / "src/forecasting/senate_data.py", ROOT / "src/forecasting/poll_data.py", Path(__file__)])
    receipt = dict(
        schema="senate_momentum_descriptive_receipt_v1",
        source_scope=plan["scope"],
        helper_audits=helper_audits,
        inputs=[
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            for path in paths
        ],
        selected_polls=len(selected),
        polls_before_truth_join=polls_before_truth_join,
        excluded_poll_rows_missing_truth=polls_before_truth_join - len(selected),
        eligible_races=roster.groupby("cycle").size().to_dict(),
        fit=False,
        strict_asof_qualified=False,
        model_qualification=False,
    )
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(pd.DataFrame(summaries).query("variant=='field_end' and threshold_pp==1").to_string(index=False))


if __name__ == "__main__":
    main()
