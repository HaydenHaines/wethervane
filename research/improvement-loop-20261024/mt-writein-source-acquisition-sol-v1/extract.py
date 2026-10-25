"""Extract only public Montana Senate filing/status fields from official HTML."""

from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import re


BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
BODY = RAW / "mt-candidate-list.html"
HEADERS = RAW / "mt-candidate-list.headers"
SOURCE_URL = "https://candidatefiling.mt.gov/candidatefiling/CandidateList.aspx?e=450002987"
TABLE_ID = "ctl00_ContentPlaceHolder1_grdCandidates_ctl00"
SAFE_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13)


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def find(self, tag=None, **attrs):
        matches = []
        if (tag is None or tag == self.tag) and all(self.attrs.get(k) == v for k, v in attrs.items()):
            matches.append(self)
        for child in self.children:
            if isinstance(child, Node):
                matches.extend(child.find(tag, **attrs))
        return matches

    def text(self):
        return "".join(child.text() if isinstance(child, Node) else child for child in self.children)


class Document(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def clean(value):
    return re.sub(r"\s+", " ", value).strip()


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    doc = Document()
    doc.feed(BODY.read_text(encoding="utf-8", errors="replace"))
    root = doc.root
    assert "FEDERAL GENERAL 2026" in clean(root.text())
    tables = root.find("table", id=TABLE_ID)
    assert len(tables) == 1
    table = tables[0]
    header_rows = []
    senate_rows = []
    for tr in table.find("tr"):
        cells = [child for child in tr.children if isinstance(child, Node) and child.tag in {"td", "th"}]
        if len(cells) != 14:
            continue
        if all(cell.tag == "th" for cell in cells):
            header_rows.append({i: clean(cells[i].text()) for i in SAFE_COLUMNS})
            continue
        if clean(cells[3].text()) != "UNITED STATES SENATOR":
            continue
        row = {
            "status_left": clean(cells[0].text()),
            "district_type": clean(cells[1].text()),
            "district": clean(cells[2].text()),
            "race": clean(cells[3].text()),
            "term_type": clean(cells[4].text()),
            "term_length": clean(cells[5].text()),
            "name": clean(cells[6].text()),
            "filing_date_ui_literal": clean(cells[10].text()),
            "party_preference_literal": clean(cells[11].text()),
            "ballot_order_literal": clean(cells[12].text()),
            "status_right": clean(cells[13].text()),
            "linked_detail_count": len(tr.find("a")),
        }
        senate_rows.append(row)
    assert len(header_rows) == 1
    assert header_rows[0][10] == "Filing Date"
    assert len(senate_rows) == 5
    assert all(row["district_type"] == "Statewide" and row["district"] == "STATE" for row in senate_rows)
    assert all(row["status_left"] == row["status_right"] == "FILED" for row in senate_rows)
    woodman = [row for row in senate_rows if row["name"] == "JAMI DEE WOODMAN"]
    assert len(woodman) == 1
    assert woodman[0]["filing_date_ui_literal"] == "8/7/2026 2:30:07 PM"
    assert woodman[0]["party_preference_literal"] == "NON"
    assert woodman[0]["linked_detail_count"] == 0
    header_text = HEADERS.read_text(errors="replace")
    assert "HTTP/1.1 200 OK" in header_text
    assert "Content-Type: text/html; charset=utf-8" in header_text
    assert f"Content-Length: {BODY.stat().st_size}" in header_text
    server_date = re.search(r"(?im)^Date: (.+)$", header_text)
    assert server_date
    output = {
        "schema": "Montana2026SenateOfficialPortalSafeExtractionV1",
        "request": {"method": "GET", "url": SOURCE_URL},
        "response": {
            "status": 200,
            "content_type": "text/html; charset=utf-8",
            "body_bytes": BODY.stat().st_size,
            "body_sha256": digest(BODY),
            "headers_bytes": HEADERS.stat().st_size,
            "headers_sha256": digest(HEADERS),
            "server_date_utc_literal": server_date.group(1).strip(),
            "local_body_write_utc": datetime.fromtimestamp(BODY.stat().st_mtime, timezone.utc).isoformat(),
        },
        "event_heading": "FEDERAL GENERAL 2026",
        "field_label": "Filing Date",
        "us_senate_rows_visible": len(senate_rows),
        "us_senate_rows_safe_fields": senate_rows,
        "woodman_status": "FILED",
        "woodman_filing_date_ui_literal": woodman[0]["filing_date_ui_literal"],
        "woodman_linked_candidate_detail": False,
        "ui_date_proves_original_receipt_or_qualification": False,
        "contact_or_address_fields_copied": False,
    }
    (BASE / "extraction.json").write_text(json.dumps(output, indent=2) + "\n")
    print("MT Senate rows", len(senate_rows), "Woodman UI status", output["woodman_status"])


if __name__ == "__main__":
    main()
