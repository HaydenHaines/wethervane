#!/usr/bin/env python3
"""Reproduce a conservative candidate-link inventory; no row is admitted."""
import collections
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[2]
ARCHIVE = HERE / "raw-polls-20180530.csv"
TRAINING = LAB / "research/improvement-loop-20261024/momentum-calibration-preparation-sol-v1/prepared-v1/2018/selected-training-polls.csv"
EXPECTED = {
    ARCHIVE: "79489a9d7f796d8f50841e91d8dcb858679ada75690812d4757f095a107efb91",
    TRAINING: "df682ca44d043cdc9ff6372448f205769bf763197ddfaa17c109aaaa21974e21",
}


def load(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED[path]:
        raise ValueError(f"source digest mismatch: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalized_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    archive = [row for row in load(ARCHIVE) if row["type_simple"] == "Sen-G" and row["year"] in {"2012", "2016"}]
    training = load(TRAINING)
    index = collections.defaultdict(list)
    days = []
    for row in archive:
        poll_date = dt.datetime.strptime(row["polldate"], "%m/%d/%y").date()
        election_date = dt.datetime.strptime(row["electiondate"], "%m/%d/%y").date()
        days.append((election_date - poll_date).days)
        index[(row["year"], row["location"], poll_date.isoformat())].append(row)
    totals = collections.Counter()
    candidates = []
    for row in training:
        same_day = index[(row["cycle"], row["state"], row["end_date"])]
        if same_day:
            totals["state_cycle_date"] += 1
        name_matches = [v for v in same_day if normalized_name(v["pollster"]) == normalized_name(row["pollster"])]
        if name_matches:
            totals["state_cycle_date_normalized_name"] += 1
        if len(name_matches) != 1:
            continue
        totals["unique_archive_candidate_on_key"] += 1
        other = name_matches[0]
        try:
            denominator = float(other["cand1_pct"]) + float(other["cand2_pct"])
            if other["cand1_name"] != "Democrat" or other["cand2_name"] != "Republican" or denominator <= 0:
                continue
            share = float(other["cand1_pct"]) / denominator
            delta = abs(share - float(row["dem_share"]))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if delta < .005:
            totals["unique_key_with_dem_two_party_share_within_half_percentage_point"] += 1
            candidates.append({"tierneyObservationId": row["observation_id"], "archivePollNumber": other["pollno"], "absoluteShareDifference": delta})
    result = {
        "schema": "M01Earlier538ArchiveCandidateInventoryV1",
        "archiveSha256": EXPECTED[ARCHIVE],
        "trainingSha256": EXPECTED[TRAINING],
        "archiveSenateGeneral2012And2016Rows": len(archive),
        "existing2018TrainingRows": len(training),
        "archiveDaysBeforeElection": {"min": min(days), "max": max(days)},
        "candidateMatchingCounts": dict(sorted(totals.items())),
        "candidates": candidates,
        "meaning": "Candidate links only, not verified sample identities, byte lineage, publication dates or admissible paired observations. A date/name/share match can be a coincidence or duplicate; differing source transformations and populations require primary-source review.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
