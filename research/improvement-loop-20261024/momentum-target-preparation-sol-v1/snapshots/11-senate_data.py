"""Audited Senate roster, candidate aliases, lagged returns and as-of poll adapter.

Target outcome tables are separate. Roster eligibility never uses vote shares,
winners, poll availability, or whether a contingent runoff actually occurred.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.forecasting.poll_data import normalize_538

PARTIES = {"DEM": "Democratic", "REP": "Republican"}


def normalize_candidate_name(value):
    """Exact lexical key: remove terminal party tag, accents and punctuation only."""
    s = re.sub(r"\s*\([^()]*\)\s*$", "", str(value)).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _id(x):
    return str(int(float(x)))


def _true(x):
    return str(x).lower() == "true"


def _incumbent_marker(value):
    if pd.isna(value) or not str(value).strip():
        return False
    if str(value).strip() != "(I)":
        raise ValueError(f"Unknown official incumbent marker: {value!r}")
    return True


def read_fec_senate(path, year):
    """Return candidate metadata and independently retained official vote totals."""
    book = pd.ExcelFile(path)
    sheet = next(s for s in book.sheet_names if "US" in s and "Senate" in s)
    f = pd.read_excel(book, sheet_name=sheet)
    f.columns = [str(c).strip() for c in f.columns]
    district = "DISTRICT" if "DISTRICT" in f else "D"
    f = f[f[district].astype(str).str.startswith("S")].copy()
    incumbent = next(c for c in f if c.startswith("(I)"))
    combined = next(c for c in f if c.startswith("COMBINED GE PARTY TOTALS"))
    f["general_votes_numeric"] = pd.to_numeric(f["GENERAL VOTES"].astype(str).str.replace(",", ""), errors="coerce")
    f = f[f.general_votes_numeric.notna() & f["CANDIDATE NAME"].notna()]
    out = []
    for (state, name), g in f.groupby(["STATE ABBREVIATION", "CANDIDATE NAME"]):
        labels = set(g.PARTY.astype(str).str.strip())
        tokens = {token.strip().rstrip("*") for label in labels for token in label.split("/")}
        party = "Democratic" if tokens & {"D", "DFL", "DNL", "DEM"} else ("Republican" if "R" in tokens else "Other")
        if any(label.startswith("N(D)") for label in labels):
            party = "Other"  # Alaska nonpartisan primary affiliation is not a Democratic nominee.
        if all(label == "W" or label.endswith("/W") for label in labels):
            party = "Other"  # Named write-ins are not another major-party ballot nominee.
        first, last = str(g["CANDIDATE NAME (First)"].iloc[0]), str(g["CANDIDATE NAME (Last)"].iloc[0])
        forward = f"{first} {last}" if first != "nan" and last != "nan" else str(name)
        if (first == "nan" or last == "nan") and "," in forward:
            surname, given = forward.split(",", 1)
            forward = given.strip() + " " + surname.strip()
        totals = pd.to_numeric(g[combined], errors="coerce").dropna().unique()
        if len(totals) > 1:
            raise ValueError(f"Conflicting official fusion totals {year} {state} {name}")
        votes = float(totals[0]) if len(totals) else float(g.general_votes_numeric.sum())
        out.append(
            dict(
                year=year,
                state_abbrev=state,
                candidate_name=forward,
                normalized_candidate=normalize_candidate_name(forward),
                fec_name=str(name),
                last_name=last,
                party=party,
                ballot_parties="|".join(sorted(labels)),
                incumbent=any(g[incumbent].map(_incumbent_marker)),
                official_votes=votes,
                source_sheet=sheet,
            )
        )
    return pd.DataFrame(out)


def eligibility_reasons(year, state, special, stage, simultaneous_special, candidates):
    """Declared structural restrictions, independent of actual election outcomes."""
    reasons = []
    if special:
        reasons.append("special_election_outside_regular_scope")
    if simultaneous_special:
        reasons.append("simultaneous_special_state_year_ambiguous_for_CES")
    if state in ("GA", "LA"):
        reasons.append("majority_runoff_or_jungle_system")
    if (state == "ME" and year >= 2018) or (state == "AK" and year >= 2022):
        reasons.append("ranked_choice_election_system")
    if stage != "general":
        reasons.append("not_general_stage")
    if (year, state) in {(2012, "ME"), (2020, "AK")}:
        reasons.append("major_independent_contest_outside_D_R_scope")
    for party in ("Democratic", "Republican"):
        count = sum(c["party"] == party for c in candidates)
        if count != 1:
            reasons.append(f"{party.lower()}_candidate_count_{count}")
    return reasons


def build_roster(results, races, *, years, fec_candidates=None):
    """Return metadata-only roster/crosswalk plus strictly separate outcome rows."""
    f = results[results.cycle.isin(years) & results.stage.isin(["general", "jungle primary"])].copy()
    metadata = races[races.type.eq("SenateRace")].set_index("id")
    specials = set(zip(f.loc[f.special.map(_true), "cycle"], f.loc[f.special.map(_true), "state_abbrev"]))
    roster, crosswalk, outcomes = [], [], []
    fec = fec_candidates if fec_candidates is not None else pd.DataFrame()
    for (year, rid), g in f.groupby(["cycle", "race_id"]):
        year = int(year)
        state = str(g.state_abbrev.iloc[0])
        special = any(g.special.map(_true))
        stage = str(g.stage.iloc[0])
        canonical = f"{year}:{state}:senate:" + ("special" if special else "regular")
        candidates = []
        # Metadata are deduplicated over ranked-choice rounds and fusion ballot lines.
        for cid, c in g.dropna(subset=["candidate_id", "candidate_name"]).groupby("candidate_id"):
            labels = set(c.ballot_party.dropna().astype(str))
            party = "Democratic" if "DEM" in labels else ("Republican" if "REP" in labels else "Other")
            name = str(c.candidate_name.iloc[0])
            aliases = {name}
            incumbent = None
            official_name = None
            official_votes = None
            match_kind = "unverified"
            if not fec.empty:
                pool = fec[(fec.year == year) & (fec.state_abbrev == state) & (fec.party == party)]
                exact = pool[pool.normalized_candidate.eq(normalize_candidate_name(name))]
                if len(exact) == 1:
                    matched = exact
                    match_kind = "exact_normalized_name"
                elif party in ("Democratic", "Republican") and len(pool) == 1 and (year, state) not in specials:
                    # Cross-source alias corroboration, not a respondent surname fallback.
                    matched = pool
                    match_kind = "unique_official_general_nominee_same_state_party"
                else:
                    matched = pool.iloc[0:0]
                if len(matched) == 1:
                    rec = matched.iloc[0]
                    aliases.add(rec.candidate_name)
                    incumbent = bool(rec.incumbent)
                    official_name = rec.candidate_name
                    official_votes = float(rec.official_votes)
            candidates.append(
                dict(
                    candidate_id=_id(cid),
                    candidate_name=name,
                    party=party,
                    ballot_party="|".join(sorted(labels)),
                    aliases=sorted(aliases),
                    incumbent=incumbent,
                    official_candidate_name=official_name,
                    official_match=match_kind,
                    official_votes=official_votes,
                )
            )
        reason = eligibility_reasons(year, state, special, stage, (year, state) in specials, candidates)
        # Unknown major-party incumbency in the requested FEC years is unresolved, never zero-filled.
        if (
            not fec.empty
            and year in set(fec.year)
            and any(c["party"] in ("Democratic", "Republican") and c["incumbent"] is None for c in candidates)
        ):
            reason.append("official_major_nominee_or_incumbency_unresolved")
        m = metadata.loc[int(rid)] if int(rid) in metadata.index else None
        election_date = pd.to_datetime(m["date"]).date().isoformat() if m is not None and pd.notna(m["date"]) else None
        row = dict(
            year=year,
            cycle=year,
            state=state,
            state_abbrev=state,
            state_name=g.state.iloc[0],
            race_id=canonical,
            source_race_id=_id(rid),
            office="senate",
            office_seat_name=g.office_seat_name.iloc[0],
            special=special,
            stage=stage,
            election_date=election_date,
            eligible=not reason,
            exclusion_reason=";".join(reason),
            incumbent_party=str(m.incumbent_party) if m is not None and pd.notna(m.incumbent_party) else None,
            source_urls="|".join(sorted(set(g.source.dropna().astype(str)))),
        )
        for party, prefix in [("Democratic", "dem"), ("Republican", "rep")]:
            cs = [c for c in candidates if c["party"] == party]
            row[prefix + "_candidate_id"] = cs[0]["candidate_id"] if len(cs) == 1 else None
            row[prefix + "_candidate_name"] = cs[0]["candidate_name"] if len(cs) == 1 else None
            row[prefix + "_incumbent"] = cs[0]["incumbent"] if len(cs) == 1 else None
        if not reason and row["dem_incumbent"] is True and row["rep_incumbent"] is True:
            raise ValueError(f"Two incumbent nominees in one regular Senate seat: {canonical}")
        roster.append(row)
        for c in candidates:
            party_short = {"Democratic": "D", "Republican": "R"}.get(c["party"], c["ballot_party"].split("|")[0])
            for alias in c["aliases"]:
                crosswalk.append(
                    dict(
                        year=year,
                        state=state,
                        state_abbrev=state,
                        state_name=g.state.iloc[0],
                        race_id=canonical,
                        source_race_id=_id(rid),
                        candidate_id=c["candidate_id"],
                        candidate_name=c["candidate_name"],
                        source_candidate=f"{alias} ({party_short})",
                        normalized_candidate=normalize_candidate_name(alias),
                        party=c["party"],
                        ballot_party=c["ballot_party"],
                        eligible=not reason,
                        exclusion_reason=";".join(reason),
                        official_candidate_name=c["official_candidate_name"],
                        official_match=c["official_match"],
                    )
                )
        if not reason:
            totals = {}
            official = {}
            for party, prefix in [("Democratic", "dem"), ("Republican", "rep")]:
                c = next(c for c in candidates if c["party"] == party)
                rows = g[g.candidate_id.map(lambda x: _id(x) if pd.notna(x) else "").eq(c["candidate_id"])]
                # Eligible races have no RCV rounds. Fusion lines sum once per source result ID.
                totals[prefix] = float(pd.to_numeric(rows.drop_duplicates("id").votes, errors="raise").sum())
                official[prefix] = c["official_votes"]
            if min(totals.values()) <= 0:
                raise ValueError(f"Invalid outcome counts: {canonical}")
            chosen = official if all(v is not None and v > 0 for v in official.values()) else totals
            outcomes.append(
                dict(
                    year=year,
                    cycle=year,
                    state=state,
                    race_id=canonical,
                    source_race_id=_id(rid),
                    election_date=election_date,
                    dem_votes=chosen["dem"],
                    rep_votes=chosen["rep"],
                    actual_dem_share=chosen["dem"] / (chosen["dem"] + chosen["rep"]),
                    archive_538_dem_votes=totals["dem"],
                    archive_538_rep_votes=totals["rep"],
                    outcome_source="FEC official" if chosen is official else "538 archive",
                    official_dem_votes=official["dem"],
                    official_rep_votes=official["rep"],
                    role="outcome_only",
                )
            )
    roster = pd.DataFrame(roster)
    crosswalk = pd.DataFrame(crosswalk).drop_duplicates()
    if roster.race_id.duplicated().any():
        raise ValueError("Ambiguous canonical race identity")
    return roster, crosswalk, pd.DataFrame(outcomes)


def senate_polls_as_of(raw, roster, *, cycle, cutoff):
    """Apply cutoff BEFORE selecting alternative questions from one sample/race."""
    cutoff = pd.Timestamp(cutoff)
    eligible = roster[(roster.cycle == cycle) & roster.eligible].copy()
    selected = raw[raw.cycle.eq(cycle) & raw.stage.eq("general")].copy()
    end = pd.to_datetime(selected.end_date, format="mixed", errors="coerce")
    created = pd.to_datetime(selected.created_at, format="mixed", errors="coerce")
    available = created.dt.normalize() + pd.Timedelta(days=1)
    selected = selected[end.le(cutoff) & created.notna() & available.le(cutoff)]
    valid = {str(r.source_race_id): r for r in eligible.itertuples()}
    kept = []
    exclusions = {}
    for (poll_id, question_id), g in selected.groupby(["poll_id", "question_id"]):
        rid = _id(g.race_id.iloc[0])
        race = valid.get(rid)
        reason = None
        if race is None:
            reason = "race_outside_eligible_roster"
        else:
            for party, cid in [("DEM", race.dem_candidate_id), ("REP", race.rep_candidate_id)]:
                observed = g.loc[g.party.eq(party), "candidate_id"].dropna().map(_id).unique()
                if len(observed) != 1 or observed[0] != str(cid):
                    reason = "question_not_actual_nominee_pair"
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            kept.append(g)
    if not kept:
        return pd.DataFrame(), dict(cycle=cycle, cutoff=str(cutoff.date()), question_exclusions=exclusions)
    result, audit = normalize_538(
        pd.concat(kept, ignore_index=True), source_id="538_senate_archive_v4", office="senate", cycles=(cycle,)
    )
    result["source_race_id"] = result.race_id
    result["race_id"] = result.source_race_id.map({k: r.race_id for k, r in valid.items()})
    result["sample_subsample_id"] = result.sample_id
    result["sample_id"] = "538:" + result.poll_id
    result["observation_id"] = result.sample_id + ":" + result.race_id
    if not result.available_date.le(cutoff).all():
        raise ValueError("Observation availability exceeds requested cutoff")
    result["source_vintage"] = "Current historical archive, not an exact historical revision ledger"
    audit.update(
        cutoff=str(cutoff.date()),
        roster_question_exclusions=exclusions,
        cross_race_shared_sample_ids=int(result.groupby("sample_id").race_id.nunique().gt(1).sum()),
    )
    return result, audit


def lagged_presidential_inputs(roster, presidential_results):
    """Join only strictly earlier presidential cycles, never same-cycle outcomes."""
    p = presidential_results[presidential_results.stage.eq("general")].copy()
    rows = []
    for row in roster.itertuples(index=False):
        prior = p[p.state_abbrev.eq(row.state) & p.cycle.lt(row.year)]
        if prior.empty:
            continue
        year = int(prior.cycle.max())
        g = prior[prior.cycle.eq(year)]
        g = g[g.ranked_choice_round.isna() | g.ranked_choice_round.eq(1)]
        cs = []
        for cid, c in g.dropna(subset=["candidate_id"]).groupby("candidate_id"):
            labels = set(c.ballot_party.dropna())
            party = "DEM" if "DEM" in labels else ("REP" if "REP" in labels else None)
            if party:
                cs.append((party, float(pd.to_numeric(c.votes, errors="raise").sum())))
        d = [v for party, v in cs if party == "DEM"]
        r = [v for party, v in cs if party == "REP"]
        if len(d) != 1 or len(r) != 1:
            raise ValueError("Ambiguous lagged presidential result")
        item = row._asdict()
        item.update(
            prior_presidential_cycle=year,
            prior_presidential_dem_share=d[0] / (d[0] + r[0]),
            prior_pres_cycle=year,
            prior_pres_dem_share=d[0] / (d[0] + r[0]),
        )
        rows.append(item)
    return pd.DataFrame(rows)


def training_returns_before(outcomes, target_cycle):
    """Explicit training boundary; target/future votes cannot cross this function."""
    return outcomes[outcomes.cycle.lt(target_cycle)].copy()


def add_reviewed_aliases(crosswalk, aliases):
    """Apply explicit candidate-ID aliases; never infer respondent identity fuzzily."""
    additions = []
    for alias in aliases:
        match = crosswalk[
            (crosswalk.year == alias["year"])
            & crosswalk.state.eq(alias["state"])
            & crosswalk.candidate_id.eq(str(alias["candidate_id"]))
            & crosswalk.eligible
        ]
        if match.empty or match.race_id.nunique() != 1:
            raise ValueError(f"Alias has no unique eligible candidate: {alias}")
        row = match.iloc[0].to_dict()
        if row["party"] != alias["party"]:
            raise ValueError("Alias party conflicts with verified roster")
        row.update(
            source_candidate=alias["source_candidate"],
            normalized_candidate=normalize_candidate_name(alias["source_candidate"]),
            alias_review_basis=alias["review_basis"],
        )
        additions.append(row)
    result = pd.concat([crosswalk, pd.DataFrame(additions)], ignore_index=True).drop_duplicates()
    valid = result[result.eligible]
    if valid.groupby(["year", "state", "normalized_candidate"]).candidate_id.nunique().gt(1).any():
        raise ValueError("Alias maps to multiple eligible candidates")
    return result


def load_config(path):
    config = json.loads(Path(path).read_text())
    for spec in config["sources"]:
        p = Path(spec["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Source bytes changed: {p}")
    return config
