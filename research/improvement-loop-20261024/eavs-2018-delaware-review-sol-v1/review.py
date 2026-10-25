"""Independent raw-byte and F1a review of the EAC Delaware acquisition."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
AUTHOR = PROJECT / "research/improvement-loop-20261024/eavs-2018-delaware-acquisition-sol-v1"
OWNER = Path("/home/hayden/projects/wethervane-redesign-20261006/integration")
PANEL = OWNER / "research/senate-stack-integration-20261020/eavs-2018-source-sol-v1/state-f1a-source-rows.json"
OUT = Path(__file__).resolve().parent
REQUIRED = ("FIPSCode", "Jurisdiction_Name", "State_Full", "State_Abbr", "F1a", "F2")
DE_EXPECTED = {
    ("1000100000", "KENT COUNTY"): "60230",
    ("1000300000", "NEW CASTLE COUNTY"): "212978",
    ("1000500000", "SUSSEX COUNTY"): "93342",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_csv(path, panel):
    totals = defaultdict(int)
    rows_by_state = Counter()
    invalid_by_state = defaultdict(Counter)
    state_full = defaultdict(set)
    seen_fips = set()
    duplicate_fips = []
    delaware = {}
    delaware_f2 = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert all(field in reader.fieldnames for field in REQUIRED)
        fieldnames = reader.fieldnames
        for number, row in enumerate(reader, start=2):
            assert None not in row
            state, fips = row["State_Abbr"], row["FIPSCode"]
            rows_by_state[state] += 1
            state_full[state].add(row["State_Full"])
            if fips in seen_fips:
                duplicate_fips.append({"line": number, "fips": fips, "state": state})
            seen_fips.add(fips)
            literal = row["F1a"]
            if re.fullmatch(r"[0-9]+", literal):
                totals[state] += int(literal)
            else:
                invalid_by_state[state][literal] += 1
            if state == "DE":
                key = (fips, row["Jurisdiction_Name"])
                assert key not in delaware, (number, key)
                delaware[key] = literal
                delaware_f2[key] = row["F2"]
                assert row["State_Full"].casefold() == "delaware"
    matches = {}
    for state, owner_row in panel.items():
        if state == "DE":
            continue
        assert state in rows_by_state
        assert len(state_full[state]) == 1
        assert owner_row["state_name"].casefold() in {name.casefold() for name in state_full[state]}, state
        matches[state] = totals[state] == owner_row["value"]
    assert len(matches) == 50
    return {
        "sha256": digest(path),
        "file_bytes": path.stat().st_size,
        "header_field_count": len(fieldnames),
        "header_sha256": hashlib.sha256((",".join(fieldnames)).encode()).hexdigest(),
        "required_field_positions_zero_based": {name: fieldnames.index(name) for name in REQUIRED},
        "data_rows": sum(rows_by_state.values()),
        "state_abbreviations": sorted(rows_by_state),
        "state_row_counts": dict(sorted(rows_by_state.items())),
        "duplicate_fips": duplicate_fips,
        "delaware_rows": [
            {"fips": fips, "jurisdiction": name, "f1a_literal": literal}
            for (fips, name), literal in sorted(delaware.items())
        ],
        "delaware_exact_three_expected": delaware == DE_EXPECTED,
        "delaware_f2_literals": [
            {"fips": fips, "jurisdiction": name, "f2_literal": literal}
            for (fips, name), literal in sorted(delaware_f2.items())
        ],
        "delaware_f1a_total": totals["DE"],
        "delaware_invalid_f1a": dict(invalid_by_state["DE"]),
        "non_numeric_or_negative_f1a_by_state": {
            state: dict(sorted(c.items()))
            for state, c in sorted(invalid_by_state.items()) if c
        },
        "owner_brief_matches": sum(matches.values()),
        "owner_brief_differences": [
            {"state": state, "owner": panel[state]["value"], "raw_aggregate": totals[state]}
            for state, yes in matches.items() if not yes
        ],
    }


def main():
    receipt_path = AUTHOR / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    panel_obj = json.loads(PANEL.read_text())
    assert digest(PANEL) == receipt["extraction"]["selected_50_brief_panel_sha256"]
    panel = {r["state"]: r for r in panel_obj["rows"]}
    assert len(panel) == 51 and panel["DE"]["value"] is None
    result = {
        "schema": "EAVS2018DelawareIndependentReviewCalculationV1",
        "author_receipt_sha256": digest(receipt_path),
        "owner_panel_sha256": digest(PANEL),
        "files_checked": [],
        "csv": {},
    }
    for item in receipt["files"]:
        path = PROJECT / item["path"]
        header_path = PROJECT / item["response_headers_path"]
        assert path.is_file() and header_path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
        assert header_path.stat().st_size == item["response_headers_bytes"]
        assert digest(header_path) == item["response_headers_sha256"]
        header = header_path.read_text().lower()
        assert f"{item['http_status']}" in header.splitlines()[0]
        media_type = item["content_type"].split(";")[0].split()[0].lower()
        assert "content-type: " + media_type in header
        if item["http_status"] == 200 and "content-length:" in header:
            assert int(re.search(r"content-length: ([0-9]+)", header).group(1)) == item["bytes"]
        result["files_checked"].append({
            "path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
            "header_sha256": item["response_headers_sha256"], "http_status": item["http_status"],
        })
    assert digest(AUTHOR / "extraction.json") == receipt["extraction"]["sha256"]
    assert digest(AUTHOR / "extract.py") == receipt["extraction"]["script_sha256"]
    for version in ("1-0", "1-3"):
        path = AUTHOR / "raw" / f"eac-2018-v{version}.csv"
        result["csv"][version] = check_csv(path, panel)
        assert result["csv"][version]["delaware_exact_three_expected"]
        assert result["csv"][version]["owner_brief_matches"] == 50
        assert not result["csv"][version]["owner_brief_differences"]
        assert not result["csv"][version]["delaware_invalid_f1a"]
        assert len({r["fips"] for r in result["csv"][version]["delaware_rows"]}) == 3
        expected_f2 = "4" if version == "1-0" else "Votes cast"
        assert {r["f2_literal"] for r in result["csv"][version]["delaware_f2_literals"]} == {expected_f2}
    assert result["csv"]["1-0"]["delaware_rows"] == result["csv"]["1-3"]["delaware_rows"]
    assert result["csv"]["1-0"]["header_sha256"] == result["csv"]["1-3"]["header_sha256"]
    assert result["csv"]["1-0"]["delaware_f1a_total"] == 366550
    OUT.mkdir(exist_ok=True)
    (OUT / "calculation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified author file digests, three DE rows, 366550, and 50/50 owner brief totals in each version")


if __name__ == "__main__":
    main()
