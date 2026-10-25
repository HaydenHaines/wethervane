"""Extract only Illinois 2026 U.S. Senate names/statuses and write-in scope."""

from html.parser import HTMLParser
from pathlib import Path
import json


BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"


class CandidateTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table" and attrs.get("id") == "ContentPlaceHolder1_gvCandidates":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.cell = []
        elif self.in_cell and tag == "br":
            self.cell.append("\n")

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell:
            self.row.append("".join(self.cell).strip())
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            self.rows.append(self.row)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False


def rows_from(path):
    parser = CandidateTable()
    parser.feed(path.read_text())
    headers, *rows = parser.rows
    assert headers == [
        "Name", "Write-In", "Objection Pending", "Office", "Party", "File Date", "Status"
    ]
    assert all(len(row) == 7 for row in rows)
    return rows


post_rows = rows_from(RAW / "il-senate-all.html")
link_rows = rows_from(RAW / "il-senate-office-link.html")
assert post_rows == link_rows and len(post_rows) == 4

senate = []
for row in post_rows:
    assert row[3] == "UNITED STATES SENATOR"
    status = row[6].split("\n")
    senate.append(
        {
            "name": row[0].split("\n")[0],
            "office": row[3],
            "party": row[4],
            "file_date": row[5].split("\n")[0],
            "status": status[0],
            "status_date": status[1] if len(status) > 1 else None,
            "write_in_cell_literal": row[1],
            "objection_pending_cell_literal": row[2],
        }
    )
assert [r["status"] for r in senate] == ["Active", "Active", "Active", "Removed"]
assert all(r["write_in_cell_literal"] == "" for r in senate)
assert {r["name"] for r in senate} == {
    "Stratton, Juliana", "Tracy, Don", "Harrington Jr, Whitfield", "Muhammad, Tyrone F"
}

calendar = (RAW / "il-writein-calendar.html").read_text()
assert "WRITE-IN CANDIDATES" in calendar and "9/3/2026" in calendar
assert "NOT filed with the State Board of Elections" in calendar.replace("<br />", " ")

out = {
    "schema": "Illinois2026GeneralUSSenateCandidateAcquisitionExtractionV1",
    "event": "GENERAL ELECTION - 11/3/2026",
    "office": "UNITED STATES SENATOR",
    "all_statuses_selected": True,
    "all_statuses_request_radio_value": "radAll",
    "office_request_value": "13387",
    "visible_office_rows": len(senate),
    "office_filtered_table_pagination_control_present": any(
        term in (RAW / "il-senate-all.html").read_text() for term in ("Page$", "Next", "Pager")
    ),
    "rows": senate,
    "calendar_writein_event_date": "2026-09-03",
    "calendar_writein_declarations_filed_with_state_board": False,
    "calendar_scope_note": "Declarations go to proper election authority or authorities; official calendar explicitly says they are not filed with State Board.",
    "blank_writein_cells_imply_zero_declarations": False,
    "contact_or_address_fields_extracted": False,
    "support_admission_performed": False,
}
(BASE / "extraction.json").write_text(json.dumps(out, indent=2) + "\n")
print("Senate rows", len(senate), "removed", sum(r["status"] == "Removed" for r in senate))
