"""Extract public office/status fields from the preserved official NM response.

This describes the retrieved list. It does not admit model support or establish
historical publication timing. Contact and address values are excluded.
"""

import collections
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup


def extract(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.find("table", id="ctl00_MainContent_rgCandidate_ctl00")
    assert table is not None
    headers = [x.get_text(" ", strip=True) for x in table.find("thead").find_all("th")]
    assert len(headers) == 24 and headers[0:2] == ["Contest", "Contest"]
    fields = ("District", "Filing County", "Name", "Party", "Filing Date/Time", "Ballot Order", "Status", "DistrictType")
    assert all(headers.count(field) == 1 for field in fields)
    selected = {}
    for field in ("ddlCounty", "ddlContest", "ddlParty"):
        select = soup.find("select", attrs={"name": "ctl00$MainContent$" + field})
        assert select is not None
        options = select.find_all("option", selected=True)
        assert len(options) == 1
        selected[field] = {"value": options[0].get("value"), "label": options[0].get_text(" ", strip=True)}
    assert selected["ddlCounty"] == {"value": "99", "label": "State"}
    assert selected["ddlContest"]["value"] == selected["ddlParty"]["value"] == "0"
    match = re.search(r'"_gridTableViewsData":("(?:\\.|[^"\\])*")', text)
    assert match is not None
    views = json.loads(json.loads(match.group(1)))
    assert len(views) == 1
    view = views[0]
    assert view["ClientID"] == "ctl00_MainContent_rgCandidate_ctl00"
    assert view["AllowPaging"] is False and view["PageCount"] == 1 and view["CurrentPageIndex"] == 0
    senate = []
    marked_writeins = []
    status_counts = collections.Counter()
    rows = table.select("tr.rgRow,tr.rgAltRow")
    for row in rows:
        cells = [x.get_text(" ", strip=True) for x in row.find_all("td", recursive=False)]
        assert len(cells) == len(headers)
        candidate = {field: cells[headers.index(field)] for field in fields}
        candidate["Contest display"] = cells[0]
        candidate["Contest"] = cells[1]
        status_counts[candidate["Status"]] += 1
        if candidate["Contest"] == "United States Senator":
            assert candidate["Contest display"] == "United States Senator"
            senate.append(candidate)
        if re.search(r"\(write-in\)", candidate["Name"], re.IGNORECASE):
            marked_writeins.append(candidate)
    return {
        "schema": "NMOfficialRetrievedCandidateListExtractionV1",
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "title": soup.title.get_text(" ", strip=True),
        "selected_filters": selected,
        "pagination": {key: view[key] for key in ("AllowPaging", "PageCount", "CurrentPageIndex", "PageSize", "VirtualItemCount")},
        "data_rows": len(rows),
        "status_counts": dict(status_counts),
        "senate_rows": senate,
        "explicit_writein_name_rows_other_contests": marked_writeins,
        "limitations": [
            "PageSize is a dormant setting when AllowPaging is false; VirtualItemCount zero is not a zero-candidate observation.",
            "Observed selected State/all-contest/all-party list, not a declaration about unobserved records or later changes.",
            "Current retrieval does not establish exact list contents at the October 20 forecast cutoff.",
            "Qualified status and explicit name suffixes are preserved as displayed; no source/model admission performed.",
        ],
        "contact_fields_excluded": True,
        "source_admitted": False,
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    result = extract(base / "raw/nm-candidate-list.html")
    with (base / "extraction.json").open("x") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps({"data_rows": result["data_rows"], "senate_rows": len(result["senate_rows"]), "explicit_writein_rows": len(result["explicit_writein_name_rows_other_contests"])}))
