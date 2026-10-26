#!/usr/bin/env python3
"""Count available archive support without selecting any model training data."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "research/improvement-loop-20261024/m01-538-archive-root-v1/raw-polls-20180530.csv"
TRAINING = ROOT / "research/improvement-loop-20261024/momentum-calibration-preparation-sol-v1/prepared-v1/2018/selected-training-polls.csv"
PINS = {
    ARCHIVE: "79489a9d7f796d8f50841e91d8dcb858679ada75690812d4757f095a107efb91",
    TRAINING: "df682ca44d043cdc9ff6372448f205769bf763197ddfaa17c109aaaa21974e21",
}


def read(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[path]:
        raise ValueError(f"source digest changed: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    archive = [r for r in read(ARCHIVE) if r["type_simple"] == "Sen-G" and r["year"] in {"2012", "2016"}]
    training = read(TRAINING)
    standard = {"2012": "11/6/12", "2016": "11/8/16"}
    strata = collections.Counter((r["year"], r["cand1_name"] == "Democrat" and r["cand2_name"] == "Republican", r["electiondate"] == standard[r["year"]]) for r in archive)
    comparable = [r for r in archive if r["cand1_name"] == "Democrat" and r["cand2_name"] == "Republican" and r["electiondate"] == standard[r["year"]]]
    archive_keys = {(r["year"], r["location"]) for r in comparable}
    training_keys = {(r["cycle"], r["state"]) for r in training}
    result = {
        "schema": "Earlier538TrainingSupportInventoryV1",
        "pins": {"archive": PINS[ARCHIVE], "prepared2018": PINS[TRAINING]},
        "total_archive_rows": len(archive),
        "strata": [{"year": year, "democrat_republican_labels": dr, "standard_november_date": regular, "rows": count} for (year, dr, regular), count in sorted(strata.items())],
        "standard_date_democrat_republican_rows": len(comparable),
        "state_year_keys": {
            "archive": len(archive_keys), "prepared": len(training_keys),
            "shared": len(archive_keys & training_keys),
            "archive_only": sorted([list(k) for k in archive_keys - training_keys]),
            "prepared_only": sorted([list(k) for k in training_keys - archive_keys]),
        },
        "scope": "State-year is a coarse support screen, not canonical event identity or a verified individual poll match. This inventory admits no observation and does not use 2018 or later election results in its counts. The complete archive is parsed before filtering and contains other-office 2018 outcomes. Candidate labels and earlier outcomes have retrospective semantics; no target outcome is a feature.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
