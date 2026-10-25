"""Only four reviewed 2022 joins; surrounding-whitespace FEC-ID keys only."""

import hashlib
import json
from pathlib import Path

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
INTAKE = HERE.parent / "momentum-identity-intake-luna-v2"
RECEIPT_SHA = "6aa19aa95aadd85a3a04a5802754c87ca1be835e265ddbf8088b26d0123be77f"
INVENTORY_SHA = "0157621a50a2c01cdafc3d34784b496001093f50ce6f4a5f69ad7965cac7281f"
REVIEWED = {
    "2022:IN:senate:regular": ("dem", "29603", "Thomas McDermott", "S2IN00208", 704480, 222),
    "2022:OR:senate:regular": ("rep", "30317", "Jo Rae Perkins", "S4OR00156", 788991, 526),
    "2022:PA:senate:regular": ("dem", "20700", "John Fetterman", "S6PA00274", 2751012, 540),
    "2022:NC:senate:regular": ("rep", "28723", "Ted Budd", "S2NC00505", 1905786, 423),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected, size=None):
    path = Path(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "source fingerprint mismatch: " + str(path))
    if size is not None:
        require(len(data) == size, "source byte count mismatch")
    return data


def authenticate():
    inventory = json.loads(checked(HERE / "v1-inventory.json", INVENTORY_SHA))
    require(
        {Path(p["path"]).name for p in inventory["files"]}
        == {p.name for p in (HERE.parent / "momentum-audit-v1").iterdir() if p.is_file()},
        "v1 inventory changed",
    )
    for item in inventory["files"]:
        checked(item["path"], item["sha256"], item["bytes"])
    receipt = json.loads(checked(INTAKE / "receipt.json", RECEIPT_SHA))
    for item in receipt["artifacts"]:
        path = (INTAKE / item["path"]).resolve()
        require(path.is_relative_to(INTAKE), "source packet escaped intake")
        checked(path, item["sha256"], item["bytes"])
    for item in json.loads((INTAKE / "source-inputs.json").read_bytes())["pins"]:
        checked(item["path"], item["sha256"], item["bytes"])
    v1 = json.loads((HERE.parent / "momentum-audit-v1/receipt.json").read_bytes())
    for item in v1["inputs"]:
        checked(item["path"], item["sha256"], item["bytes"])
    require(
        (HERE / "plan.json").read_bytes() == (HERE.parent / "momentum-audit-v1/plan.json").read_bytes(),
        "v1 thresholds/windows/universe plan must remain exact",
    )
    require(
        (HERE / "reviewed-crosswalk.csv").read_bytes() == (INTAKE / "crosswalk-proposal.csv").read_bytes(),
        "reviewed crosswalk changed",
    )
    crosswalk = pd.read_csv(HERE / "reviewed-crosswalk.csv", dtype=str)
    require(
        len(crosswalk) == 4 and not crosswalk.race_id.duplicated().any(), "four explicit reviewed identities required"
    )
    for row in crosswalk.itertuples():
        expected = REVIEWED[row.race_id]
        require(
            (
                row.poll_archive_candidate_id,
                row.poll_archive_candidate_name,
                row.official_fec_candidate_id,
                int(row.official_result_votes),
            )
            == expected[1:5],
            "crosswalk identity differs",
        )
    return inventory, receipt, crosswalk


def normalized_votes(votes):
    result = votes.copy()
    result["fec_id_key"] = result.fec_candidate_id.fillna("").map(str).str.strip()
    # Fusion rows can share an identical raw ID, but distinct raw IDs cannot collapse.
    collisions = result[result.fec_id_key.ne("")].groupby(["race_id", "fec_id_key"]).fec_candidate_id.nunique()
    require(not (collisions > 1).any(), "conflicting raw FEC IDs collapse under whitespace trim")
    return result


def verify_workbook_and_poll_ids(votes, roster, raw_path):
    """Independently compare every official general category in the four races."""
    workbook = openpyxl.load_workbook(INTAKE / "raw/federalelections2022.xlsx", read_only=True, data_only=True)
    sheet = workbook["7. US Senate Results by State"]
    raw = list(sheet.iter_rows(values_only=True))
    poll = pd.read_csv(raw_path, usecols=["cycle", "state", "race_id", "candidate_id", "candidate_name", "party"])
    evidence = []
    for race_id, (party, archive_id, name, fec_id, count, excel_row) in REVIEWED.items():
        state = race_id.split(":")[1]
        rows = [
            (i + 1, r)
            for i, r in enumerate(raw)
            if r[1] == state and r[3] == "S" and r[8] is not None and r[15] is not None
        ]
        exported = votes[votes.race_id.eq(race_id)]
        require(
            set(exported.source_excel_row) == {i for i, _ in rows}, "official general vote/category rows incomplete"
        )
        for number, row in rows:
            record = exported[exported.source_excel_row.eq(number)]
            require(len(record) == 1, "duplicate official category")
            record = record.iloc[0]
            raw_id = "" if row[4] in (None, "n/a") else row[4]
            actual_id = "" if pd.isna(record.fec_candidate_id) else record.fec_candidate_id
            require(
                (actual_id, record.fec_candidate_name, record.fec_party, record.general_votes)
                == (raw_id, row[8], row[10], row[15]),
                "raw official workbook/export mismatch",
            )
            require(float(row[15]).is_integer() and row[15] >= 0, "incomplete/noninteger official votes")
        total = [r[15] for r in raw if r[1] == state and r[3] == "S" and r[9] == "Total State Votes:"]
        require(len(total) == 1 and sum(r[15] for _, r in rows) == total[0], "official all-category total mismatch")
        reviewed = raw[excel_row - 1]
        require(reviewed[4].strip() == fec_id and reviewed[15] == count, "reviewed raw identity/votes differ")
        if state == "NC":
            require(reviewed[4] == "S2NC00505 ", "NC raw trailing-space evidence changed")
        race = roster[roster.race_id.eq(race_id)]
        require(len(race) == 1, "reviewed eligible race missing or duplicate")
        race = race.iloc[0]
        require(
            str(race[party + "_candidate_id"]) == archive_id and race[party + "_candidate_name"] == name,
            "reviewed poll roster identity differs",
        )
        candidate_rows = poll[poll.cycle.eq(2022) & poll.candidate_id.eq(int(archive_id))]
        require(
            len(candidate_rows) > 0
            and candidate_rows.candidate_name.eq(name).all()
            and candidate_rows.race_id.eq(int(race.source_race_id)).all()
            and candidate_rows.party.eq("DEM" if party == "dem" else "REP").all(),
            "poll candidate ID/name/party/race conflict",
        )
        evidence.append(
            dict(
                race_id=race_id,
                raw_fec_id=reviewed[4],
                normalized_fec_id=fec_id,
                source_excel_row=excel_row,
                raw_votes=count,
                complete_category_count=len(rows),
                complete_all_category_votes=total[0],
                poll_raw_candidate_rows=len(candidate_rows),
                archive_candidate_id=archive_id,
                archive_candidate_name=name,
            )
        )
    workbook.close()
    return evidence


def build_truth(roster, candidates, votes, normalize_name, ident):
    votes = normalized_votes(votes)
    truth, joins = [], []
    for race in roster[roster.cycle.eq(2022)].itertuples():
        pair = []
        valid = True
        for party in ("dem", "rep"):
            archive_id = getattr(race, party + "_candidate_id")
            name = getattr(race, party + "_candidate_name")
            members = candidates[
                candidates.race_id.eq(race.race_id) & candidates.group_id.eq("D" if party == "dem" else "R")
            ]
            if len(members) != 1:
                valid = False
                joins.append(dict(race_id=race.race_id, party=party, accepted=False, reason="nonunique_official_group"))
                continue
            member = members.iloc[0]
            fec_id = str(member.fec_candidate_id).strip()
            source = votes[votes.race_id.eq(race.race_id) & votes.fec_id_key.eq(fec_id)]
            if source.empty or not source.general_votes.notna().all():
                valid = False
                joins.append(
                    dict(
                        race_id=race.race_id,
                        party=party,
                        accepted=False,
                        reason="official_FEC_id_has_no_complete_vote_rows",
                    )
                )
                continue
            exact = normalize_name(member.candidate_name) == normalize_name(name)
            reviewed = REVIEWED.get(race.race_id)
            explicit = reviewed is not None and (party, ident(archive_id), name, fec_id) == reviewed[:4]
            if explicit:
                require(float(source.general_votes.sum()) == reviewed[4], "reviewed official count changed")
            accepted = exact or explicit
            valid = valid and accepted
            pair.append(float(source.general_votes.sum()))
            joins.append(
                dict(
                    race_id=race.race_id,
                    party=party,
                    roster_candidate_id=archive_id,
                    roster_name=name,
                    official_candidate_id=member.candidate_id,
                    official_name=member.candidate_name,
                    fec_id_key=fec_id,
                    raw_vote_fec_ids=json.dumps(source.fec_candidate_id.tolist()),
                    source_excel_rows=json.dumps(source.source_excel_row.tolist()),
                    votes=pair[-1],
                    accepted=accepted,
                    reason="" if accepted else "full_name_mismatch_no_reviewed_alias",
                    join_basis="reviewed_explicit_2022_crosswalk"
                    if explicit and not exact
                    else "unique_official_event_group_plus_exact_lexical_fullname",
                    whitespace_trim_applied=any(str(v) != str(v).strip() for v in source.fec_candidate_id),
                )
            )
        if valid and len(pair) == 2:
            require(min(pair) > 0, "positive ordinary D/R vote totals required")
            truth.append(
                dict(
                    race_id=race.race_id,
                    cycle=2022,
                    actual_dem_share=pair[0] / sum(pair),
                    election_date=race.election_date,
                )
            )
    return truth, joins
