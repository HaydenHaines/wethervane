"""Read dated TWIP archives, never current gasoline history or later reports.

Older-source arm: reported issue dates, not authenticated original posting logs.
Date-only releases get a conservative next-day UTC availability bound.
"""

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

HERE = Path(__file__).resolve().parent
INDEX = "https://www.eia.gov/petroleum/weekly/"
SERIES = "US_regular_retail_gasoline_TWIP"
METHOD_BREAK = date(2018, 5, 14)


def text(html):
    return " ".join(unescape(re.sub("<[^>]+>", " ", html)).split())


class Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def availability_bound(issue):
    # Date-only archive headers do not establish an intraday posting time.
    return datetime.combine(issue + timedelta(days=1), time(23, 59, 59), timezone.utc)


DATE_EXCEPTION_2004 = dict(
    id="twip_2004_10_14_stale_header_year_v1",
    source_sha256="58543d9c56d405343e4752a47d678a2d5dc522bb124f87d2731ba1af466d2528",
    index_sha256="58f8c79895d1e7f9a8834d5bdef40d9032b1ba3a9e145f5b131947cf128d5db8",
    source_url="https://www.eia.gov/petroleum/weekly/archive/2004/041014/twipprint.html",
    header_literal="Released on October 14, 2003 (Next Release on October 20, 2003)",
    source_header_issue_date="2003-10-14",
    admitted_issue_date="2004-10-14",
    observation_date="2004-10-11",
    basis="Exact official index link/date, 2004 retail and supply table dates, weekday alignment, independent review",
    scope="intake-only date exception; not original posting authentication or scientific qualification",
)


def issue_date_exception(index_bytes, source):
    """A single explicit byte-bound exception; never general year correction."""
    if source.get("issue_date") != "2004-10-14":
        return None
    if hashlib.sha256(index_bytes).hexdigest() != DATE_EXCEPTION_2004["index_sha256"]:
        raise ValueError("date exception index digest mismatch")
    if parse_index(index_bytes).get(date(2004, 10, 14)) != DATE_EXCEPTION_2004["source_url"]:
        raise ValueError("date exception index identity mismatch")
    return dict(DATE_EXCEPTION_2004)


def parse_html(data, source, date_exception=None):
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("source digest mismatch")
    html = data.decode("utf-8", errors="replace")
    plain = text(html)
    if not re.search(r"This Week [Ii]n Petroleum", plain):
        raise ValueError("not TWIP")
    match = re.search(r"(?:Release Date:|Released on|Released:)\s*([A-Za-z]+\.? \d{1,2}, \d{4})", plain, re.I)
    if not match:
        match = re.search(r"([A-Za-z]+\.? \d{1,2}, \d{4})\s*\(Next Release on", plain, re.I)
    if not match:
        raise ValueError("source issue date absent")
    literal = match[1].replace(".", "")
    try:
        issue = datetime.strptime(literal, "%B %d, %Y").date()
    except ValueError:
        issue = datetime.strptime(literal, "%b %d, %Y").date()
    source_header_issue_date = issue.isoformat()
    if issue.isoformat() != source["issue_date"]:
        if (
            date_exception != DATE_EXCEPTION_2004
            or source.get("sha256") != DATE_EXCEPTION_2004["source_sha256"]
            or source.get("url") != DATE_EXCEPTION_2004["source_url"]
            or source.get("issue_date") != DATE_EXCEPTION_2004["admitted_issue_date"]
            or issue.isoformat() != DATE_EXCEPTION_2004["source_header_issue_date"]
            or DATE_EXCEPTION_2004["header_literal"] not in plain
        ):
            raise ValueError("issue date disagrees with archive index")
        issue = date.fromisoformat(DATE_EXCEPTION_2004["admitted_issue_date"])
    elif date_exception is not None:
        raise ValueError("unexpected issue-date exception")
    # Scope the national retail summary, never spot/futures or a regional row.
    row_match = re.search(r'<tr\b[^>]*id=["\']rp_gasoline["\'][^>]*>(.*?)</tr>', html, re.S | re.I)
    if row_match:
        start = row_match.start()
        context = html[max(0, start - 2500) : start]
        dates = re.findall(r">\s*(\d{1,2}/\d{1,2}/\d{2,4})\s*</(?:th|td)>", context, re.I)
        if not dates:
            raise ValueError("retail table observation date absent")
        observation = datetime.strptime(
            dates[-1], "%m/%d/%y" if len(dates[-1].split("/")[-1]) == 2 else "%m/%d/%Y"
        ).date()
        parser = Rows()
        parser.feed(row_match[0])
        cells = parser.rows[0]
        if cells[0] not in ("Gasoline", "U.S."):
            raise ValueError("not national gasoline row")
        raw_value = cells[1]
        retail_heading = plain.lower()
        unit = "USD_per_gallon" if re.search(r"retail prices\s*\(dollars per gallon\)", retail_heading) else None
    else:
        marker = re.search(r"<!--\s*RETAIL TABLE\s*-->", html, re.I)
        if not marker:
            raise ValueError("unsupported legacy retail table")
        section = html[marker.end() :]
        parser = Rows()
        parser.feed(section)
        dates = []
        cells = None
        for row in parser.rows:
            for cell in row:
                if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", cell):
                    dates.append(cell)
            if row and row[0] == "Gasoline":
                cells = row
                break
        if not cells or not dates:
            raise ValueError("legacy gasoline/date missing")
        observation = datetime.strptime(dates[0], "%m/%d/%y").date()
        raw_value = cells[1]
        unit = (
            "cents_per_gallon"
            if re.search(r"Retail Prices \(Cents Per Gallon\)", text(section), re.I)
            else ("USD_per_gallon" if re.search(r"Retail Prices \(Dollars Per Gallon\)", text(section), re.I) else None)
        )
    if unit is None:
        raise ValueError("unverified retail units")
    if not re.fullmatch(r"\d+(?:\.\d+)?", raw_value):
        raise ValueError("invalid numeric price")
    value = float(Decimal(raw_value) / (100 if unit == "cents_per_gallon" else 1))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("nonpositive or nonfinite price")
    if observation.weekday() != 0 or observation > issue:
        raise ValueError("invalid Monday observation date")
    if date_exception is not None and observation.isoformat() != DATE_EXCEPTION_2004["observation_date"]:
        raise ValueError("date exception observation mismatch")
    # National regular grade must be described in source prose or retail chart.
    if (
        not re.search(
            r"(?:U\.S\..{0,120}regular gasoline|regular gasoline.{0,100}U\.S\.|Retail Average Regular Gasoline)",
            plain,
            re.I,
        )
        and "Regular Gasoline" not in html
    ):
        raise ValueError("regular-grade measure not supported")
    return dict(
        schema="twip_regular_retail_row_v1",
        series_id=SERIES,
        geography="US",
        grade="regular",
        formulations="national_regular_summary_not_separately_reconstructed",
        units="USD_per_gallon",
        value=value,
        source_value=raw_value,
        source_units=unit,
        observation_date=observation.isoformat(),
        issue_date=issue.isoformat(),
        source_header_issue_date=source_header_issue_date,
        issue_date_exception=date_exception,
        available_upper_bound_utc=availability_bound(issue).isoformat(),
        release_time_basis="date_only_source_header_end_following_UTC_day",
        source=source,
        method_regime="post_2018_05_14" if observation >= METHOD_BREAK else "pre_2018_05_14",
        original_posting_log_authenticated=False,
    )


def parse_index(data):
    html = data.decode("utf-8")
    links = re.findall(r'href=["\']([^"\']+)["\']', html)
    issues = {}
    for link in links:
        match = re.search(r"/archive/(\d{4})/(\d{6})/", link)
        if not match:
            short = re.search(r"/archive/(\d{6})/", link)
            if not short:
                continue
            match = re.match(r"(\d{4})/(\d{6})", "20" + short[1][:2] + "/" + short[1])
        stamp = datetime.strptime(match[2], "%y%m%d").date()
        if stamp.year != int(match[1]):
            raise ValueError("archive date mismatch")
        if not any(name in link for name in ["twipprint.html", "analysis_print.php", "hometables_print.php"]):
            continue
        issues[stamp] = urljoin(INDEX, link)
    if not issues:
        raise ValueError("empty archive index")
    return issues


def election_day(year):
    first = date(year, 11, 1)
    return first + timedelta(days=(-first.weekday()) % 7 + 1)


def combine(current, lagged, cutoff_utc):
    cutoff = datetime.fromisoformat(cutoff_utc)
    if cutoff.tzinfo is None or cutoff.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC cutoff required")
    for row in (current, lagged):
        if (
            row.get("schema") != "twip_regular_retail_row_v1"
            or row["series_id"] != SERIES
            or row["units"] != "USD_per_gallon"
            or row["grade"] != "regular"
            or row["geography"] != "US"
        ):
            raise ValueError("series or unit mismatch")
        if datetime.fromisoformat(row["available_upper_bound_utc"]) > cutoff:
            raise ValueError("post-cutoff issue")
        if isinstance(row["value"], bool) or not math.isfinite(row["value"]) or row["value"] <= 0:
            raise ValueError("invalid price")
        issue = date.fromisoformat(row["issue_date"])
        if datetime.fromisoformat(row["available_upper_bound_utc"]) != availability_bound(issue):
            raise ValueError("invalid date-only release bound")
        if date.fromisoformat(row["observation_date"]) > issue:
            raise ValueError("observation follows issue")
    observation = date.fromisoformat(current["observation_date"])
    earlier = date.fromisoformat(lagged["observation_date"])
    if observation - earlier != timedelta(weeks=13):
        raise ValueError("not exactly thirteen observation weeks")
    if observation.weekday() != 0 or earlier.weekday() != 0:
        raise ValueError("not Monday observations")
    return dict(
        schema="twip_pre_cutoff_gasoline_contrast_v1",
        arm="older_verified_dated_TWIP_issue_not_latest_GDFU",
        cutoff_utc=cutoff_utc,
        current=current,
        lag13=lagged,
        log_change_13_weeks=math.log(current["value"] / lagged["value"]),
        change_USD_per_gallon=current["value"] - lagged["value"],
        crosses_2018_method_break=earlier < METHOD_BREAK <= observation,
        observation_age_days=(cutoff.date() - observation).days,
        strict_original_posting_authenticated=False,
    )


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    previous = (
        json.loads((HERE / "acquisition.json").read_text())
        if (HERE / "acquisition.json").exists()
        else {"attempts": []}
    )
    pinned = {item["file"]: item["sha256"] for item in previous["attempts"] if "file" in item}
    attempts = []

    def fetch(url, name, issue=None):
        path = raw / name
        if path.exists():
            data = path.read_bytes()
            status = "existing_bytes"
            if "raw/" + name in pinned and hashlib.sha256(data).hexdigest() != pinned["raw/" + name]:
                raise ValueError("existing artifact digest mismatch")
        else:
            try:
                with urllib.request.urlopen(url, timeout=25) as response:
                    data = response.read()
                path.write_bytes(data)
                status = "downloaded"
            except urllib.error.URLError as error:
                attempts.append(
                    dict(
                        url=url,
                        status="fetch_failed",
                        error=str(error),
                        observed_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
                return None
        source = dict(
            url=url,
            file="raw/" + name,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            status=status,
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
        )
        if issue:
            source["issue_date"] = issue.isoformat()
        attempts.append(source)
        return data, source

    index = fetch(INDEX, "index.html")
    issues = parse_index(index[0])
    rows = {}
    failures = []
    panel = []

    def load(issue):
        if issue in rows:
            return rows[issue]
        fetched = fetch(issues[issue], issue.isoformat() + ".html", issue)
        if not fetched:
            return None
        try:
            row = parse_html(*fetched, date_exception=issue_date_exception(index[0], fetched[1]))
        except ValueError as error:
            failures.append(dict(issue_date=issue.isoformat(), reason=str(error)))
            return None
        rows[issue] = row
        return row

    for year in range(2002, 2025, 2):
        cutoff = datetime.combine(election_day(year) - timedelta(days=14), time(23, 59, 59), timezone.utc)
        eligible = sorted(issue for issue in issues if issue < cutoff.date() and (cutoff.date() - issue).days <= 14)
        if not eligible:
            panel.append(dict(cycle=year, status="unavailable", reason="no_pre_cutoff_indexed_issue"))
            continue
        latest = eligible[-1]
        current = load(latest)
        if current is None:
            panel.append(
                dict(
                    cycle=year,
                    status="unavailable",
                    reason="latest_indexed_issue_not_parsed",
                    issue_date=latest.isoformat(),
                )
            )
            continue
        target = date.fromisoformat(current["observation_date"]) - timedelta(weeks=13)
        lagged = None
        for issue in sorted(
            day for day in issues if target <= day <= target + timedelta(days=7) and day < cutoff.date()
        ):
            row = load(issue)
            if row and row["observation_date"] == target.isoformat():
                lagged = row
                break
        if lagged is None:
            panel.append(
                dict(
                    cycle=year,
                    status="unavailable",
                    reason="exact_thirteen_week_observation_not_found",
                    target_date=target.isoformat(),
                    current=current,
                )
            )
            continue
        panel.append(dict(cycle=year, status="available", **combine(current, lagged, cutoff.isoformat())))
        print(
            year,
            current["issue_date"],
            current["observation_date"],
            current["value"],
            lagged["observation_date"],
            lagged["value"],
            flush=True,
        )
    result = dict(
        schema="twip_pre_cutoff_panel_v1",
        cutoff_convention="end_H14_calendar_day_UTC",
        panel=panel,
        parse_failures=failures,
        fit=False,
    )
    (HERE / "panel.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "acquisition.json").write_text(
        json.dumps(dict(schema="twip_acquisition_v1", attempts=attempts), indent=2) + "\n"
    )
    files = [HERE / "panel.json", HERE / "acquisition.json", Path(__file__)] + [
        HERE / item["file"] for item in attempts if "file" in item
    ]
    files.extend(path for path in (HERE / "raw").iterdir() if path.is_file())
    files.extend(
        HERE / name
        for name in ("supplementary-acquisition.json", "2004-review-evidence.json")
        if (HERE / name).exists()
    )
    files = sorted(set(files))
    receipt = dict(
        schema="twip_build_receipt_v1",
        files=[
            dict(
                file=str(path.relative_to(HERE)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
            )
            for path in files
        ],
        fit=False,
        latest_GDFU_values_claimed=False,
    )
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
