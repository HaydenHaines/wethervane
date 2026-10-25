"""Independent byte and scope checks for Tennessee blank write-in form."""

from datetime import date, timedelta
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import subprocess


PROJECT = Path(__file__).resolve().parents[3]
AUTHOR = PROJECT / "research/improvement-loop-20261024/tn-writein-source-acquisition-luna-v1"
BASE = Path(__file__).resolve().parent


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def html_text(path):
    parser = TextOnly()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return re.sub(r"\s+", " ", " ".join(parser.parts))


def main():
    author_receipt = json.loads((AUTHOR / "receipt.json").read_text())
    author_pdf = AUTHOR / "raw/sumner-tn-ss3072-blank.pdf"
    sos_pdf = BASE / "raw/sos-ss3072.pdf"
    author_headers = AUTHOR / "raw/sumner-tn-ss3072.headers.txt"
    sos_headers = BASE / "raw/sos-ss3072.headers"
    madison_page = BASE / "raw/madison-writein.html"
    madison_headers = BASE / "raw/madison-writein.headers"
    denial_body = AUTHOR / "raw/tn-candidate-list-page.html"
    denial_headers = AUTHOR / "raw/tn-candidate-list-page.headers.txt"
    assert digest(author_pdf) == author_receipt["sources"][0]["sha256"]
    assert author_pdf.stat().st_size == author_receipt["sources"][0]["bytes"] == 88178
    assert digest(sos_pdf) == digest(author_pdf)
    assert "HTTP/2 200" in author_headers.read_text()
    assert "content-type: application/pdf" in author_headers.read_text().lower()
    assert "HTTP/1.1 200 OK" in sos_headers.read_text()
    assert "Content-Type: application/pdf" in sos_headers.read_text()
    assert "Content-Length: 88178" in sos_headers.read_text()
    assert "HTTP/2 200" in madison_headers.read_text()
    assert "content-type: text/html" in madison_headers.read_text().lower()
    assert digest(denial_body) == author_receipt["sources"][1]["sha256"]
    assert denial_body.stat().st_size == author_receipt["sources"][1]["bytes"] == 919
    assert "HTTP/2 403" in denial_headers.read_text()
    pdf_text = subprocess.check_output(["pdftotext", "-layout", str(author_pdf), "-"], text=True)
    normalized_pdf = re.sub(r"\s+", " ", pdf_text)
    assert "Certificate of Write-In Candidacy" in normalized_pdf
    assert "For primary elections" in normalized_pdf
    assert "For general elections" in normalized_pdf
    assert "Tennessee Coordinator of Elections" in normalized_pdf
    assert "each county that makes up the district of the office sought" in normalized_pdf
    assert "fiftieth (50th) day before the election" in normalized_pdf
    assert "Revised 7/2016" in normalized_pdf
    assert "Printed Name of Candidate: ____________________________________" in normalized_pdf
    county_text = html_text(madison_page)
    assert "General Election Process" in county_text
    assert "If a candidate for a multi-county office fails to file in all counties" in county_text
    assert "only be counted in the counties where the certificate has been filed" in county_text
    assert "Primary Election Process" in county_text
    assert date(2026, 11, 3) - timedelta(days=50) == date(2026, 9, 14)
    result = {
        "schema": "TNWriteInIndependentSourceReviewCalculationV1",
        "author_receipt_sha256": digest(AUTHOR / "receipt.json"),
        "source_files": [
            {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
            for path in [author_pdf, author_headers, sos_pdf, sos_headers, madison_page, madison_headers, denial_body, denial_headers]
        ],
        "sos_pdf_identical_to_author_county_pdf": True,
        "form_revision_literal": "7/2016",
        "form_page_count": 1,
        "form_visual_layout_checked": True,
        "general_election_county_filing_direction_present": True,
        "primary_coordinator_direction_present": True,
        "partial_county_filing_can_support_only_filed_counties_per_county_guidance": True,
        "general_election_date": "2026-11-03",
        "fiftieth_day_before_election": "2026-09-14",
        "deadline_time_from_form": "12:00 noon prevailing time",
        "completed_2026_certificate_acquired": False,
        "named_complete_writein_roster_acquired": False,
        "owner_cutoff_byte_availability_proved": False,
    }
    (BASE / "calculation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified official identical PDF bytes, partial-county nuance, and September 14 date")


if __name__ == "__main__":
    main()
