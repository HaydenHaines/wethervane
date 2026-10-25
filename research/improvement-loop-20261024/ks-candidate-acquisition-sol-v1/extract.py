"""Extract only election, office, name, party, and filing date from the KS public table."""

from html.parser import HTMLParser
from pathlib import Path
import json


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "raw" / "ks-post.html"


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
        if tag == "table" and attrs.get("id") == "gvCandidateList":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.cell = []

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


html = SOURCE.read_text()
parser = CandidateTable()
parser.feed(html)
headers, *rows = parser.rows
assert len(headers) == 25
assert headers[0] == "Candidate" and headers[1] == "Office"
assert headers[5] == "Party" and headers[22] == "Date Filed"
assert all(len(row) == len(headers) for row in rows)
senate = [
    {"candidate": row[0], "office": row[1], "party": row[5], "date_filed": row[22]}
    for row in rows
    if row[1] == "United States Senate"
]
assert len(senate) == 3
assert {row["candidate"] for row in senate} == {
    "Adam Hamilton", "Roger Marshall", "David C Graham"
}
out = {
    "schema": "Kansas2026GeneralPublicCandidateListExtractionV1",
    "election_selector": {"label": "2026 General", "value": "36"},
    "visible_candidate_table_rows": len(rows),
    "table_columns": headers,
    "united_states_senate_rows": senate,
    "write_in_or_affidavit_term_present_in_returned_html": any(
        term in html.lower() for term in ("write-in", "write in", "affidavit")
    ),
    "table_pagination_control_present": any(
        term in html for term in ("Page$", "gvCandidateList$", "Pager")
    ),
    "limitation": "This returned table documents candidate list rows, not the completeness of qualified write-in affidavits.",
    "contact_or_address_fields_extracted": False,
}
(BASE / "extraction.json").write_text(json.dumps(out, indent=2) + "\n")
print("visible rows", len(rows), "Senate rows", len(senate))
