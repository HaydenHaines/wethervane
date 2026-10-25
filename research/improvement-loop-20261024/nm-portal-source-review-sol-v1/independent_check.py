"""Independent, contact-free review of the NM SOS candidate portal source bytes."""

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import json


LAB = Path("/home/hayden/projects/ballotmancy-improvement-lab")
SOURCE = LAB / "research/improvement-loop-20261024/nm-portal-acquisition-root-v1/raw"
OUT = Path(__file__).resolve().parent
TARGET_TABLE = "ctl00_MainContent_rgCandidate_ctl00"


class PortalReview(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title_on = False
        self.title = []
        self.select = None
        self.option = None
        self.option_text = []
        self.selected = {}
        self.in_table = False
        self.row = None
        self.cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.title_on = True
        elif tag == "select":
            self.select = attrs.get("name")
        elif tag == "option" and self.select:
            self.option = attrs
            self.option_text = []
        elif tag == "table" and attrs.get("id") == TARGET_TABLE:
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.row = {"class": attrs.get("class"), "cells": []}
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.title_on:
            self.title.append(data)
        if self.option is not None:
            self.option_text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "title":
            self.title_on = False
        elif tag == "option" and self.option is not None:
            if "selected" in self.option:
                self.selected[self.select] = {
                    "value": self.option.get("value"),
                    "label": " ".join("".join(self.option_text).split()),
                }
            self.option = None
        elif tag == "select":
            self.select = None
        elif tag in ("td", "th") and self.cell is not None:
            self.row["cells"].append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.in_table:
            self.in_table = False


class HomepageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.text = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.text = []

    def handle_data(self, data):
        if self.href:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((" ".join("".join(self.text).split()), self.href))
            self.href = None


home = HomepageLinks()
home.feed((SOURCE / "nm-sos-home.html").read_text())
links = [(text, href) for text, href in home.links if "2026 General Election" in text and "Candidate Filing Day" in text]
assert len(links) == 1
expected_url = "https://candidateportal.servis.sos.state.nm.us/CandidateList.aspx?eid=2917&cty=99"
assert links[0][1] == expected_url

html = (SOURCE / "nm-candidate-list.html").read_text()
portal = PortalReview()
portal.feed(html)
assert " ".join("".join(portal.title).split()) == "2026 General Election Contest/Candidate List"
selected_controls = {name.rsplit("$", 1)[-1]: value for name, value in portal.selected.items()}
filters = {name: selected_controls[name] for name in ("ddlCounty", "ddlContest", "ddlParty")}
assert filters["ddlCounty"] == {"value": "99", "label": "State"}
assert filters["ddlContest"] == {"value": "0", "label": ""}
assert filters["ddlParty"] == {"value": "0", "label": ""}

assert len(portal.rows) == 525
headers = portal.rows[0]["cells"]
assert len(headers) == 24
assert headers[:5] == ["Contest", "Contest", "District", "Filing County", "Name"]
assert headers[20:23] == ["Ballot Order", "Status", "DistrictType"]
records = [row["cells"] for row in portal.rows[1:]]
assert all(row["class"] in ("rgRow", "rgAltRow") for row in portal.rows[1:])
assert all(len(row) == len(headers) for row in records)
status_counts = Counter(row[21] for row in records)
assert status_counts == Counter({"Qualified": 503, "Disqualified": 11, "Withdrawn": 10})

def safe_record(row):
    return {
        "Contest display": row[0],
        "Contest": row[1],
        "District": row[2],
        "Filing County": row[3],
        "Name": row[4],
        "Party": row[8],
        "Filing Date/Time": row[19],
        "Ballot Order": row[20],
        "Status": row[21],
        "DistrictType": row[22],
    }


senate = [safe_record(row) for row in records if row[0] == "United States Senator" and row[1] == "United States Senator"]
assert len(senate) == 4
assert [(row["Name"], row["Status"]) for row in senate] == [
    ("BEN R LUJAN", "Qualified"),
    ("LARRY E MARKER", "Qualified"),
    ("MIRA OCONNELL", "Disqualified"),
    ("BOB PERLS", "Disqualified"),
]
writeins = [safe_record(row) for row in records if "(write-in)" in row[4].lower()]
assert len(writeins) == 6
assert all(row["Contest"] != "United States Senator" for row in writeins)

assert '\\"AllowPaging\\":false' in html
assert '\\"PageCount\\":1' in html
assert '\\"CurrentPageIndex\\":0' in html
assert '\\"PageSize\\":10' in html
assert '\\"VirtualItemCount\\":0' in html
assert len(records) == 524

out = {
    "schema": "NMPortalIndependentSourceChecksSolV1",
    "homepage_link": {"text": links[0][0], "target": links[0][1]},
    "portal_title": " ".join("".join(portal.title).split()),
    "selected_filters": filters,
    "other_selected_controls": {name: value for name, value in selected_controls.items() if name not in filters},
    "grid": {"AllowPaging": False, "PageCount": 1, "CurrentPageIndex": 0, "PageSize": 10, "VirtualItemCount": 0, "visible_data_rows": len(records)},
    "status_counts": dict(status_counts),
    "senate_rows": senate,
    "explicit_other_contest_writein_rows": writeins,
    "other_rows_with_different_contest_display_and_base_office": sum(row[0] != row[1] for row in records),
    "safe_fields_only": True,
    "contact_address_fields_extracted": False,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "independent-checks.json").write_text(json.dumps(out, indent=2) + "\n")
print("rows", len(records), "Senate", len(senate), "other-write-ins", len(writeins))
