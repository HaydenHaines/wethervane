"""Independent Illinois Senate filing source review; emits no address/contact data."""

from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import re


PROJECT = Path(__file__).resolve().parents[3]
AUTHOR = PROJECT / "research/improvement-loop-20261024/il-candidate-acquisition-sol-v1"
OUT = Path(__file__).resolve().parent


class Node:
    def __init__(self, tag="root", attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or [])
        self.children = []

    def find(self, tag=None, **attrs):
        hits = []
        if (tag is None or self.tag == tag) and all(self.attrs.get(k) == v for k, v in attrs.items()):
            hits.append(self)
        for child in self.children:
            if isinstance(child, Node):
                hits.extend(child.find(tag, **attrs))
        return hits

    def text(self):
        if self.tag == "br":
            return "\n"
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)


class DOM(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].children.append(n)
        if tag not in self.VOID:
            self.stack.append(n)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def load_dom(path):
    parser = DOM()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.root


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flat(text):
    return re.sub(r"\s+", " ", text).strip()


def table_rows(dom):
    matches = dom.find("table", id="ContentPlaceHolder1_gvCandidates")
    assert len(matches) == 1
    table = matches[0]
    result = []
    for tr in table.find("tr"):
        cells = [c for c in tr.children if isinstance(c, Node) and c.tag in {"td", "th"}]
        if cells:
            result.append([re.sub(r"[ \t\r\f\v]+", " ", cell.text()).strip() for cell in cells])
    assert [flat(value) for value in result[0]] == ["Name", "Write-In", "Objection Pending", "Office", "Party", "File Date", "Status"]
    assert all(len(row) == 7 for row in result)
    return result[1:]


def checked_candidates(path):
    rows = table_rows(load_dom(path))
    assert len(rows) == 4
    safe = []
    for row in rows:
        assert flat(row[3]) == "UNITED STATES SENATOR"
        assert flat(row[1]) == "" and flat(row[2]) == ""
        # Name and status cell contain a second date line. Keep only public
        # office/status fields; never print or serialize addresses or contacts.
        name = flat(row[0].split("\n", 1)[0])
        status_match = re.fullmatch(r"(Active|Removed)\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})", flat(row[6]))
        assert status_match
        safe.append({
            "name": name,
            "write_in_cell": flat(row[1]),
            "objection_pending_cell": flat(row[2]),
            "office": flat(row[3]),
            "party": flat(row[4]),
            "file_date": flat(row[5]).split(" ", 1)[0],
            "status": status_match.group(1),
            "status_date": status_match.group(2),
        })
    return safe


def main():
    receipt_path = AUTHOR / "receipt.json"
    author = json.loads(receipt_path.read_text())
    result = {
        "schema": "Illinois2026SenateIndependentSourceReviewCalculationV1",
        "author_receipt_sha256": digest(receipt_path),
        "raw_files_verified": [],
    }
    for item in author["source_files"]:
        path = PROJECT / item["path"]
        headers = PROJECT / item["response_headers_path"]
        assert path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"]
        assert headers.stat().st_size == item["response_headers_bytes"]
        assert digest(headers) == item["response_headers_sha256"]
        header_text = headers.read_text(errors="replace").lower()
        assert "content-type: text/html" in header_text
        for status in item["http_status_chain"]:
            assert f" {status} " in header_text
        assert urlparse(item["request_url"]).netloc == "www.elections.il.gov"
        result["raw_files_verified"].append({
            "path": item["path"], "sha256": item["sha256"],
            "bytes": item["bytes"], "http_status_chain": item["http_status_chain"],
        })
    assert digest(AUTHOR / "extract.py") == author["extraction"]["script_sha256"]
    assert digest(AUTHOR / "extraction.json") == author["extraction"]["sha256"]
    body_path = PROJECT / author["request"]["body_path"]
    assert digest(body_path) == author["request"]["body_sha256"]
    assert body_path.stat().st_size == author["request"]["body_bytes"]
    form = parse_qs(body_path.read_text(), keep_blank_values=True, strict_parsing=True)
    assert sorted(form) == sorted(author["request"]["field_names"])
    expected = {
        "ctl00$ContentPlaceHolder1$ddlOfficeGroup": "US SENATOR",
        "ctl00$ContentPlaceHolder1$ddlOffices": "13387",
        "ctl00$ContentPlaceHolder1$Status": "radAll",
        "ctl00$ContentPlaceHolder1$txtCanLastName": "",
        "ctl00$ContentPlaceHolder1$txtBegDate": "",
        "ctl00$ContentPlaceHolder1$txtEndDate": "",
    }
    assert all(form[key] == [value] for key, value in expected.items())
    assert all(form[key][0] for key in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"))
    result["request_filter_values"] = {key.rsplit("$", 1)[-1]: value for key, value in expected.items()}

    office_dom = load_dom(AUTHOR / "raw/il-office-search.html")
    election_links = []
    for row in office_dom.find("tr"):
        cells = [c for c in row.children if isinstance(c, Node) and c.tag == "td"]
        if len(cells) >= 2 and flat(cells[0].text()) == "GENERAL ELECTION" and flat(cells[1].text()) == "11/3/2026":
            election_links.extend(a.attrs.get("href", "") for a in cells[0].find("a"))
    assert any("CandidateFilingSearch.aspx?ID=sejIrI%2bQmww%3d" in link for link in election_links)
    filing_dom = load_dom(AUTHOR / "raw/il-filing-form.html")
    office_options = [n for n in filing_dom.find("option") if flat(n.text()) == "UNITED STATES SENATOR"]
    assert len(office_options) == 1 and office_options[0].attrs.get("value") == "13387"
    assert any(n.attrs.get("value") == "US SENATOR" for n in filing_dom.find("option"))
    assert len(filing_dom.find("input", value="radAll")) == 1
    for token in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"):
        nodes = filing_dom.find("input", name=token)
        assert len(nodes) == 1 and nodes[0].attrs.get("value") == form[token][0]
    result["route"] = {
        "official_election_link_found": True,
        "official_office_option_value": "13387",
        "form_hidden_tokens_matched_without_recording_values": True,
        "post_event_id_in_url": "sejIrI%2bQmww%3d",
    }
    post_headers = (AUTHOR / "raw/il-senate-all.headers").read_text(errors="replace")
    assert author["request"]["response_redirect_location"] in post_headers
    post_dom = load_dom(AUTHOR / "raw/il-senate-all.html")
    office_link = "CandidateList.aspx?ElectionID=sejIrI%2bQmww%3d&OfficeID=b3l9OOoBLkjHeNv3Pz40NQ%3d%3d"
    assert any(a.attrs.get("href") == office_link for a in post_dom.find("a"))
    assert author["source_files"][5]["request_url"].endswith(office_link)
    candidate_table = post_dom.find("table", id="ContentPlaceHolder1_gvCandidates")[0]
    pager_markers = [a.attrs.get("href", "") for a in candidate_table.find("a") if "Page$" in a.attrs.get("href", "")]
    assert not pager_markers
    post = checked_candidates(AUTHOR / "raw/il-senate-all.html")
    direct = checked_candidates(AUTHOR / "raw/il-senate-office-link.html")
    assert post == direct
    assert Counter(row["status"] for row in post) == {"Active": 3, "Removed": 1}
    result["candidates"] = post
    result["post_and_direct_office_link_rows_identical"] = True
    result["client_derived_office_link_verified"] = True
    result["visible_candidate_table_pager_links"] = len(pager_markers)

    calendar = load_dom(AUTHOR / "raw/il-writein-calendar.html")
    calendar_text = flat(calendar.text())
    assert "WRITE-IN CANDIDATES" in calendar_text.upper()
    assert "9/3/2026" in calendar_text
    assert "NOT filed with the State Board of Elections" in calendar_text
    assert "proper election authority or authorities" in calendar_text
    result["calendar"] = {
        "event_date_literal": "9/3/2026",
        "writein_declarations_filed_with_state_board": False,
        "proper_election_authority_phrase_present": True,
        "publication_date_proven": False,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "calculation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified raw hashes, client-derived form route, four Senate rows, and write-in calendar scope")


if __name__ == "__main__":
    main()
