"""Auditable adapters for *published aggregate* election polls, never pseudo counts.

Archive snapshots are not historical database vintages. ``created_at`` is an
aggregator ingestion proxy, not a verified poll publication timestamp. Selection
uses known metadata, never topline values, outcomes, or retrospective ratings.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

STATE_NAMES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|District of Columbia|Florida|"
    "Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|"
    "Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
    "North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|"
    "Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming"
).split("|")
STATE_CODES = (
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY "
    "NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY"
).split()
STATES = dict(zip(STATE_NAMES, STATE_CODES))
STATES.update(dict(zip(STATE_CODES, STATE_CODES)))
STATES.update({"U.S.": "US", "US": "US"})
ELECTION_DATES = {2016: "2016-11-08", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _date(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _true(value):
    return str(value).lower() in {"true", "1", "1.0"}


def _id(value):
    return str(int(float(value)))


def _region(value):
    return "US" if pd.isna(value) else STATES.get(str(value))


def normalize_538(
    frame: pd.DataFrame, *, source_id: str, cycles=(2020, 2024), office="president", exclude_tracking=True
) -> tuple[pd.DataFrame, dict]:
    """Collapse candidate rows, then choose one question per underlying sample.

    Prefer LV, RV, adults; then the largest candidate lineup; finally smallest
    question ID. No preference uses observed percentages or sample outcomes.
    Distinct poll IDs can still hide shared panels; that limitation is audited.
    """
    counts = Counter()
    rows = []
    required = {
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
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing required source columns: {sorted(missing)}")
    for (poll_id, question_id), group in frame.groupby(["poll_id", "question_id"], sort=True):
        # Diagnostic/rating columns cannot even alter semantic duplicate removal.
        semantic = sorted(
            required
            | {"candidate_id", "election_date", "tracking", "hypothetical", "ranked_choice_reallocated", "methodology"}
        )
        g = group.drop_duplicates(subset=[c for c in semantic if c in group])
        first = g.iloc[0]
        cycle = int(first.cycle)
        reason = None
        if cycle not in cycles:
            reason = "cycle_out_of_scope"
        elif first.stage != "general":
            reason = "not_general"
        elif any(_true(v) for v in g.get("hypothetical", [])):
            reason = "hypothetical"
        elif any(_true(v) for v in g.get("ranked_choice_reallocated", [])):
            reason = "ranked_choice_reallocated"
        elif exclude_tracking and any(_true(v) for v in g.get("tracking", [])):
            reason = "tracking_overlap"
        region = _region(first.state)
        if reason is None and region is None:
            reason = "unsupported_geography"
        population = str(first.population).lower()
        if reason is None and population not in {"lv", "rv", "a"}:
            reason = "unknown_population"
        for col in [
            "state",
            "cycle",
            "race_id",
            "start_date",
            "end_date",
            "sample_size",
            "population",
            "created_at",
            "pollster",
        ]:
            if reason is None and g[col].nunique(dropna=False) != 1:
                reason = "conflicting_question_metadata"
        d, r = g[g.party.eq("DEM")], g[g.party.eq("REP")]
        if reason is None and (len(d) != 1 or len(r) != 1):
            reason = "not_unique_dem_rep_pair"
        if reason:
            counts[reason] += 1
            continue
        start, end = _date(first.start_date), _date(first.end_date)
        created = _date(first.created_at)
        election = _date(first.get("election_date", ELECTION_DATES.get(cycle)))
        n, dp, rp = float(first.sample_size), float(d.iloc[0].pct), float(r.iloc[0].pct)
        if not (np.isfinite(n) and n > 0 and 0 < dp < 100 and 0 < rp < 100 and dp + rp <= 100.01):
            counts["invalid_n_or_topline"] += 1
            continue
        if pd.isna(start) or pd.isna(end) or pd.isna(election) or start > end or end >= election:
            counts["invalid_field_dates"] += 1
            continue
        if pd.isna(created):
            counts["missing_ingestion_timestamp"] += 1
            continue
        # General-election matchups switch only after the publicly announced withdrawal.
        # Earlier Harris hypotheticals and later Biden alternatives are not fungible.
        if office == "president":
            name = d.iloc[0].candidate_name.lower()
            expected = (
                "harris"
                if cycle == 2024 and start >= pd.Timestamp("2024-07-22")
                else ("biden" if cycle in (2020, 2024) else "clinton")
            )
            if expected not in name or (cycle == 2024 and start < pd.Timestamp("2024-07-22") <= end):
                counts["candidate_regime_mismatch"] += 1
                continue
        available = max(end.normalize(), created.normalize() + pd.Timedelta(days=1))
        # Availability is next day after ingestion: time zone is unspecified in source.
        q = dp / (dp + rp)
        rows.append(
            dict(
                source_id=source_id,
                poll_id=_id(poll_id),
                question_id=_id(question_id),
                sample_id=f"538:{_id(poll_id)}:{region}",
                observation_id=f"538:{_id(poll_id)}:{_id(first.race_id)}",
                race_id=_id(first.race_id),
                cycle=cycle,
                office=office,
                state=region,
                region=region,
                start_date=start.normalize(),
                end_date=end.normalize(),
                election_date=election.normalize(),
                created_at=created,
                publication_date=pd.NaT,
                available_date=available,
                availability_basis="aggregator_created_at_plus_one_day",
                publication_date_verified=False,
                historical_version_verified=False,
                population=population.upper(),
                sample_size=n,
                dem_pct=dp,
                rep_pct=rp,
                dem_share=q,
                share=q,
                variance=q * (1 - q) / (n * (dp + rp) / 100),
                sample_size_basis="reported_question_sample",
                pollster=str(first.pollster),
                methodology=str(first.get("methodology", "unknown")),
                dem_candidate=str(d.iloc[0].candidate_name),
                rep_candidate=str(r.iloc[0].candidate_name),
                lineup_size=int(g.candidate_name.nunique()),
                tracking=any(_true(v) for v in g.get("tracking", [])),
                source_url=str(first.get("url", "")),
                partisan=str(first.get("partisan", "")),
                _population_rank={"lv": 0, "rv": 1, "a": 2}[population],
                _question_order=int(question_id),
            )
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["_population_rank", "lineup_size", "_question_order"], ascending=[True, False, True]
        )
        before = len(result)
        result = result.drop_duplicates(["cycle", "office", "sample_id", "race_id"])
        counts["alternate_question_same_sample"] += before - len(result)
        result = (
            result.drop(columns=["_population_rank", "_question_order"])
            .sort_values(["cycle", "available_date", "sample_id"])
            .reset_index(drop=True)
        )
    return result, {
        "source_id": source_id,
        "input_candidate_rows": len(frame),
        "input_questions": int(frame.groupby(["poll_id", "question_id"]).ngroups),
        "retained_observations": len(result),
        "question_exclusions": dict(counts),
    }


def normalize_2016(frame: pd.DataFrame, *, source_id: str):
    """2016 model-export adapter: raw toplines only; repeated model variants collapse.

    Tracking labels are absent. In availability order, retain only disjoint field
    windows per pollster/state. Future arrivals cannot remove earlier polls.
    """
    keep = [
        "poll_id",
        "question_id",
        "cycle",
        "state",
        "startdate",
        "enddate",
        "pollster",
        "samplesize",
        "population",
        "rawpoll_clinton",
        "rawpoll_trump",
        "rawpoll_johnson",
        "rawpoll_mcmullin",
        "createddate",
        "url",
    ]
    f = frame[keep].drop_duplicates()
    candidates = []
    for _, row in f.iterrows():
        base = dict(
            poll_id=row.poll_id,
            question_id=row.question_id,
            cycle=2016,
            state=row.state,
            start_date=row.startdate,
            end_date=row.enddate,
            pollster=row.pollster,
            sample_size=row.samplesize,
            population=row.population,
            created_at=row.createddate,
            race_id=100000 + (STATE_CODES.index(_region(row.state)) + 1 if _region(row.state) in STATE_CODES else 0),
            stage="general",
            url=row.url,
            election_date="2016-11-08",
        )
        candidates.extend(
            [
                dict(base, party="DEM", pct=row.rawpoll_clinton, candidate_name="Hillary Clinton"),
                dict(base, party="REP", pct=row.rawpoll_trump, candidate_name="Donald Trump"),
            ]
        )
        for col, name, party in [
            ("rawpoll_johnson", "Gary Johnson", "LIB"),
            ("rawpoll_mcmullin", "Evan McMullin", "IND"),
        ]:
            if pd.notna(row[col]):
                candidates.append(dict(base, party=party, pct=row[col], candidate_name=name))
    result, audit = normalize_538(pd.DataFrame(candidates), source_id=source_id, cycles=(2016,))
    # Online greedy thinning. Strictly later arrivals cannot alter earlier retention.
    # Inclusive field windows touching on a day count as overlapping respondents.
    exclude = set()
    for _, g in result.groupby(["region", "pollster"]):
        g = g.sort_values(["available_date", "sample_id", "question_id"])
        latest_selected_end = pd.Timestamp.min
        for index, row in g.iterrows():
            if row.start_date <= latest_selected_end:
                exclude.add(index)
            else:
                latest_selected_end = row.end_date
    audit["question_exclusions"]["unknown_tracking_online_overlap_thinning"] = len(exclude)
    result = result.drop(index=list(exclude)).reset_index(drop=True)
    result["tracking_metadata_available"] = False
    result["source_completeness"] = "censored_2016_11_01_snapshot"
    audit["retained_observations"] = len(result)
    audit["input_model_variant_rows"] = len(frame)
    return result, audit


def normalize_outcomes(frame: pd.DataFrame, *, office: str, cycles) -> pd.DataFrame:
    """Separate evaluation targets, summing fusion-party ballot lines per candidate.

    Use first-choice rounds to match poll estimands. Races with multiple major-party
    candidates or no Democratic/Republican nominee cannot define our D/R target.
    """
    f = frame[frame.cycle.isin(cycles) & frame.stage.eq("general")].copy()
    f = f[f.ranked_choice_round.isna() | f.ranked_choice_round.eq(1)]
    rows = []
    for (cycle, race_id), group in f.groupby(["cycle", "race_id"]):
        state = group.state_abbrev.iloc[0]
        region = _region(state)
        if region is None:
            continue
        candidates = []
        for cid, candidate in group.dropna(subset=["candidate_id"]).groupby("candidate_id"):
            labels = set(candidate.ballot_party.dropna())
            party = "DEM" if "DEM" in labels else ("REP" if "REP" in labels else None)
            if party:
                candidates.append((party, candidate.votes.sum(min_count=1), cid))
        ds, rs = [x for x in candidates if x[0] == "DEM"], [x for x in candidates if x[0] == "REP"]
        if len(ds) != 1 or len(rs) != 1:
            continue
        dv, rv = ds[0][1], rs[0][1]
        if not np.isfinite(dv + rv) or dv <= 0 or rv <= 0:
            continue
        rows.append(
            dict(
                cycle=int(cycle),
                office=office,
                race_id=_id(race_id),
                region=region,
                state=region,
                dem_votes=float(dv),
                rep_votes=float(rv),
                actual_dem_share=float(dv / (dv + rv)),
                special=any(_true(v) for v in group.get("special", [])),
                election_date=ELECTION_DATES[int(cycle)],
                target_role="evaluation_only",
            )
        )
    return pd.DataFrame(rows)


def polls_as_of(
    polls: pd.DataFrame, cutoff, *, cycle=None, office=None, lookback_days=None, final_nominee_only=True, race_id=None
) -> pd.DataFrame:
    """Replay using explicit ingestion proxy; never imply vintage-exact publication.

    Final-nominee 2024 evaluations omit Biden observations, because their estimand
    is a different candidate. That restriction is prospective after July 21.
    """
    cutoff = pd.Timestamp(cutoff)
    keep = polls.available_date.le(cutoff) & polls.end_date.le(cutoff)
    if cycle is not None:
        keep &= polls.cycle.eq(cycle)
    if office is not None:
        keep &= polls.office.eq(office)
    if race_id is not None:
        keep &= polls.race_id.eq(str(race_id))
    if lookback_days is not None:
        keep &= polls.end_date.ge(cutoff - pd.Timedelta(days=lookback_days))
    if final_nominee_only:
        keep &= ~(
            polls.cycle.eq(2024)
            & polls.office.eq("president")
            & ~polls.dem_candidate.str.contains("Harris", case=False)
        )
    return polls.loc[keep].copy()


def acquire_source(spec: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / spec["filename"]
    if not path.exists():
        with urlopen(spec["url"], timeout=60) as response:
            content = response.read()
        if spec.get("zip_member"):
            import io

            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                content = archive.read(spec["zip_member"])
        if hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError(f"Source hash mismatch for {spec['id']}; refusing changed archive")
        path.write_bytes(content)
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Source hash mismatch for {spec['id']}")
    return path


def load_poll_dataset(config: dict | str | Path) -> tuple[pd.DataFrame, dict]:
    """Return normalized public aggregate observations and provenance audit."""
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    frames, audits = [], []
    for spec in config["sources"]:
        path = acquire_source(spec, config["runtime_directory"])
        raw = pd.read_csv(path, low_memory=False)
        if spec["format"] == "538_2016_model":
            result, audit = normalize_2016(raw, source_id=spec["id"])
        else:
            result, audit = normalize_538(raw, source_id=spec["id"], cycles=spec["cycles"], office=spec["office"])
        frames.append(result)
        audits.append(dict(audit, sha256=sha256(path), url=spec["url"], completeness=spec["completeness"]))
    polls = pd.concat(frames, ignore_index=True)
    report = {"sources": audits, "observations": len(polls), "policy": config["policy"], "coverage": []}
    for (cycle, office), g in polls.groupby(["cycle", "office"]):
        report["coverage"].append(
            dict(
                cycle=int(cycle),
                office=office,
                observations=len(g),
                regions=int(g.region.nunique()),
                first_field_date=str(g.start_date.min().date()),
                last_field_date=str(g.end_date.max().date()),
                population_counts=g.population.value_counts().to_dict(),
                national_observations=int(g.region.eq("US").sum()),
                unavailable_by_election=int(g.available_date.ge(g.election_date).sum()),
                unique_samples=int(g.sample_id.nunique()),
            )
        )
    return polls, report
