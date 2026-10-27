"""Independently check archived page bodies against CDX and prior evidence pins.

This checks two public pages, not survey respondents or prepared model rows.
"""
import base64
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

EVIDENCE = Path(__file__).resolve().parent.parent / "m01-primary-vintage-authentication-luna-v1/evidence.json"
EVIDENCE_SHA = "bc6214e6dd7680e7d181e8a1f641a209e7bddd60d17ab55bad28a9f9e56d6715"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "Ballotmancy-vintage-review/1.0"}), timeout=20) as response:
        need(response.status == 200, "unexpected HTTP status")
        return response.read()


def checked_page(record):
    cdx = json.loads(fetch(record["cdx_query_url"]))
    capture = record["archive_capture_utc"].replace("-", "").replace(":", "").replace("T", "").replace("Z", "")
    matches = [dict(zip(cdx[0], row)) for row in cdx[1:] if row[1] == capture]
    need(len(matches) == 1 and matches[0]["statuscode"] == "200", "archive capture missing or ambiguous")
    body = fetch(record["archive_capture_url"])
    sha = hashlib.sha256(body).hexdigest()
    digest = base64.b32encode(hashlib.sha1(body).digest()).decode()
    need(sha == record["retrieved_replay_body_sha256"] and digest == matches[0]["digest"] == record["archive_cdx_record"]["digest"],
         "archive page or CDX digest differs")
    return body.decode("utf-8", errors="replace"), {"capture_utc": record["archive_capture_utc"],
            "sha256": sha, "cdx_sha1_base32": digest, "body_bytes": len(body)}


def main():
    raw = EVIDENCE.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == EVIDENCE_SHA, "reviewed evidence pin changed")
    evidence = json.loads(raw)
    question_html, question_pin = checked_page(evidence["rasmussen_single_release"]["archived_question_page"])
    question_text = " ".join(html.unescape(re.sub(r"<[^>]*>", " ", question_html)).split()).lower()
    need(all(word in question_text for word in ("500 likely voters", "october 21, 2012", "flake", "carmona", "split the survey to rotate")),
         "archived question-page content differs")
    article, article_pin = checked_page(evidence["fivethirtyeight"]["official_article"])
    need("https://github.com/fivethirtyeight/data/tree/master/pollster-ratings" in article and
         "2018-05-30T15:00:00+00:00" in article, "archive article link/time differs")
    nonfinding = evidence["fivethirtyeight"]["raw_csv_exact_object_archive_check"]
    rows = json.loads(fetch(nonfinding["cdx_query_url"]))
    need(rows == [], "exact commit CSV archive query now has a capture; re-evaluate")
    print(json.dumps({"schema": "A2018PrimaryVintageIndependentReviewV1", "source_evidence_sha256": EVIDENCE_SHA,
        "question_archive": question_pin, "article_archive": article_pin,
        "question_sample_and_rotation_text_found": True, "article_links_mutable_master": True,
        "queried_exact_csv_archive_matches": len(rows),
        "exact_csv_public_availability_proved": False, "original_sample_identity_proved": False,
        "full_a2018_source_vintage_proved": False, "forecast_qualified": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
