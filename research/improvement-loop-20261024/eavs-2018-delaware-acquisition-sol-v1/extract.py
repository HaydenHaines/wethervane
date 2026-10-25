"""Extract 2018 Delaware F1a from official EAC release CSVs and compare briefs."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
BRIEFS = Path(
    "/home/hayden/projects/wethervane-redesign-20261006/integration/"
    "research/senate-stack-integration-20261020/eavs-2018-source-sol-v1/"
    "state-f1a-source-rows.json"
)


def extract(filename):
    by_state = defaultdict(list)
    excluded = defaultdict(Counter)
    delaware = []
    with (RAW / filename).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert "F1a" in reader.fieldnames and "State_Abbr" in reader.fieldnames
        for row in reader:
            state = row["State_Abbr"]
            literal = row["F1a"]
            if state == "DE":
                delaware.append(
                    {
                        "FIPSCode": row["FIPSCode"],
                        "Jurisdiction_Name": row["Jurisdiction_Name"],
                        "State_Abbr": state,
                        "F1a_literal": literal,
                    }
                )
            try:
                value = int(literal)
            except ValueError:
                excluded[state][literal] += 1
                continue
            if value < 0:
                excluded[state][literal] += 1
                continue
            by_state[state].append(value)
    return {
        "delaware_rows": delaware,
        "delaware_f1a_sum": sum(by_state["DE"]),
        "delaware_nonnegative_numeric_count": len(by_state["DE"]),
        "delaware_excluded_literals": dict(excluded["DE"]),
        "state_f1a_sums": {state: sum(values) for state, values in sorted(by_state.items())},
        "excluded_f1a_literals": {
            state: dict(counts) for state, counts in sorted(excluded.items()) if counts
        },
    }


brief_rows = json.loads(BRIEFS.read_text())["rows"]
brief_values = {row["state"]: row["value"] for row in brief_rows if row["value"] is not None}
assert len(brief_values) == 50 and "DE" not in brief_values

versions = {
    "EAC_version_1_0_released_2019_06_27": extract("eac-2018-v1-0.csv"),
    "EAC_version_1_3_released_2020_07_15": extract("eac-2018-v1-3.csv"),
}
for info in versions.values():
    info["brief_comparison"] = {
        "brief_values_checked": len(brief_values),
        "exact_matches": sum(info["state_f1a_sums"].get(s) == v for s, v in brief_values.items()),
        "differences": [
            {"state": s, "brief": v, "dataset": info["state_f1a_sums"].get(s)}
            for s, v in brief_values.items()
            if info["state_f1a_sums"].get(s) != v
        ],
    }
    assert info["brief_comparison"]["exact_matches"] == 50
    assert info["delaware_nonnegative_numeric_count"] == 3
    assert not info["delaware_excluded_literals"]
    assert info["delaware_f1a_sum"] == 366550
assert versions[next(iter(versions))]["delaware_rows"] == versions[list(versions)[1]]["delaware_rows"]

result = {
    "schema": "EAVS2018DelawareOfficialDatasetExtractionV1",
    "source_question": "F1a",
    "source_question_codebook_location": "2018_EAVS_Codebook.xlsx xl/worksheets/sheet1.xml row 291: F1a, Numeric, Participation: Total",
    "reference_election_date": "2018-11-06",
    "calculation": "Sum the three Delaware jurisdiction F1a nonnegative integer literals; exclude missing or negative literals from comparison sums without treating them as observed zero.",
    "versions": versions,
    "selected_2018_brief_panel_unchanged": True,
    "admission_performed": False,
}
(BASE / "extraction.json").write_text(json.dumps(result, indent=2) + "\n")
print("DE F1a", versions[next(iter(versions))]["delaware_f1a_sum"])
print("2018 brief matches", versions[next(iter(versions))]["brief_comparison"]["exact_matches"])
