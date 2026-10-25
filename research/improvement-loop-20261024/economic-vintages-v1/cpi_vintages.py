"""Offline BLS CPI-U release-vintage parser and explicit UTC cutoff selector.

Only NSA published twelve-month percentage changes are supported. Original
release contents can themselves revise older observations. No current series,
seasonal-factor substitution, interpolation, forecasting or fitting occurs here.
"""

import hashlib
import json
import math
import re
import subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
SERIES = {"headline": "CUUR0000SA0", "core": "CUUR0000SA0L1E"}
MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC timestamp required")
    return parsed


def election_day(year):
    first = date(year, 11, 1)
    return first + timedelta(days=(0 - first.weekday()) % 7 + 1)


def cutoff_for(year, horizon_days=14):
    if type(horizon_days) is not int or horizon_days < 0:
        raise ValueError("horizon must be nonnegative integer calendar days")
    # Explicit convention: end of H14 calendar day in UTC, not local election close.
    cutoff = datetime.combine(election_day(year) - timedelta(days=horizon_days), time(23, 59, 59), timezone.utc)
    return cutoff.isoformat()


def parse_table_cells(text):
    """Read complete whitespace-delimited numeric cells, including R markers."""
    values = []
    for token in text.split():
        match = re.fullmatch(r"[Rr]?([-+]?(?:\d+\.\d+|\.\d+))[Rr]?", token)
        if match is None:
            raise ValueError("unrecognized Table1 numeric cell")
        values.append(float(match.group(1)))
    return values


def parse_release(pdf, record, source):
    data = Path(pdf).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("source digest mismatch")
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"]).decode("utf-8")
    header = text[:2200]
    if "Bureau of Labor Statistics" not in text or "CPI-U" not in text:
        raise ValueError("not BLS CPI-U release")
    date_match = re.search(r"(" + "|".join(MONTHS) + r")\s+(\d{1,2}),\s*(\d{4})", header, re.I)
    clock = re.search(r"8:30\s*[Aa]\.?[Mm]\.?\s*\((EDT|EST|ET)\)", header)
    if not date_match or not clock:
        raise ValueError("unrecognized source embargo timestamp")
    month, day, year = date_match.groups()
    released = datetime(int(year), MONTHS[month.title()], int(day), 8, 30, tzinfo=ZoneInfo("America/New_York"))
    if released.date().isoformat() != record["release_date"]:
        raise ValueError("release date disagrees with source header")
    reference = record["reference_month"]
    reference_label = date.fromisoformat(reference + "-01").strftime("%B %Y")
    if not re.search(r"CONSUMER\s+PRICE\s+INDEX\s*[:–—-]\s*" + reference_label, header, re.I):
        raise ValueError("reference month disagrees with release title")
    pieces = re.split(r"Table\s+1\.", text, maxsplit=1, flags=re.I)
    if len(pieces) != 2:
        raise ValueError("Table1 absent")
    table = re.split(r"Table\s+2\.", pieces[1], maxsplit=1, flags=re.I)[0]
    if not re.search(r"Unadjusted[\s\S]{0,100}?percent", table, re.I):
        raise ValueError("Table1 NSA percent header missing")
    rows = []
    for kind, label in [("headline", r"All\s+items"), ("core", r"All\s+items\s+less\s+food\s+and\s+energy")]:
        pattern = r"(?im)^\s*" + label + r"[\d, ]*[. ]+\s+([Rr]?\d{1,3}\.\d{3}\s+[^\n]+)"
        matches = re.findall(pattern, table)
        if len(matches) != 1:
            raise ValueError(f"nonunique {kind} Table1 row: {len(matches)}")
        values = parse_table_cells(matches[0])
        # Older Table1 has two index columns; newer has previous-year + two
        # current-year indexes. Their next column is NSA12month, not monthly SA.
        if len(values) not in (8, 9):
            raise ValueError("unsupported Table1 column layout")
        index = 3 if len(values) == 8 else 4
        if kind == "headline" and values[0] != 100:
            raise ValueError("headline relative importance mismatch")
        value = values[index]
        rows.append(
            dict(
                schema="bls_cpi_release_row_v1",
                series_id=SERIES[kind],
                kind=kind,
                reference_month=reference,
                release_utc=released.astimezone(timezone.utc).isoformat(),
                vintage_utc=released.astimezone(timezone.utc).isoformat(),
                value=value,
                units="percent_change_12_months",
                seasonality="NSA",
                geography="US_city_average",
                population="CPI_U",
                revision_status="as_published_in_dated_release",
                release_timestamp_basis="BLS_PDF_embargo_header",
                original_BLS_server_bytes_authenticated=False,
                document_revision_notice_present=bool(
                    re.search(r"Corrections to Prescription Drug|revised Consumer Price|corrected", text, re.I)
                ),
                revision_notice_scope="document_only_not_selected_NSA_series_revision_evidence",
                source=source,
                original_url=record["original_url"],
                extraction=dict(
                    table="1",
                    column="unadjusted_percent_change_from_same_month_previous_year",
                    row_text=matches[0],
                    numeric_columns=len(values),
                ),
            )
        )
    return rows


def select(
    rows,
    *,
    cutoff_utc,
    election_date,
    horizon_days,
    series_id,
    units="percent_change_12_months",
    seasonality="NSA",
    reference_month=None,
):
    cutoff = timestamp(cutoff_utc)
    election = date.fromisoformat(election_date)
    if type(horizon_days) is not int or horizon_days < 0 or election - cutoff.date() != timedelta(days=horizon_days):
        raise ValueError("cutoff/election/horizon mismatch")
    if units != "percent_change_12_months" or seasonality != "NSA" or series_id not in SERIES.values():
        raise ValueError("unsupported series/unit/seasonality request")
    candidates = []
    seen = {}
    for row in rows:
        if row.get("schema") != "bls_cpi_release_row_v1":
            raise ValueError("unknown row schema")
        if row["series_id"] != series_id:
            continue
        if (
            row["units"] != units
            or row["seasonality"] != seasonality
            or row["geography"] != "US_city_average"
            or row["population"] != "CPI_U"
            or row["revision_status"] != "as_published_in_dated_release"
        ):
            raise ValueError("row estimand or revision policy mismatch")
        if (
            not isinstance(row["value"], (int, float))
            or isinstance(row["value"], bool)
            or not math.isfinite(row["value"])
        ):
            raise ValueError("nonfinite or invalid value")
        reference = date.fromisoformat(row["reference_month"] + "-01")
        if reference > date.fromisoformat(row["release_utc"][:7] + "-01"):
            raise ValueError("reference month follows release")
        released, vintage = timestamp(row["release_utc"]), timestamp(row["vintage_utc"])
        if vintage < released:
            raise ValueError("vintage precedes release")
        key = (row["reference_month"], row["vintage_utc"])
        payload = json.dumps(row, sort_keys=True)
        if key in seen and seen[key] != payload:
            raise ValueError("conflicting same-vintage observations")
        seen[key] = payload
        if released > cutoff or vintage > cutoff or cutoff - released > timedelta(days=90):
            continue
        if reference_month is None or row["reference_month"] == reference_month:
            candidates.append(row)
    if not candidates:
        return dict(status="unavailable", reason="no_eligible_dated_release", value=None)
    # Latest reference period first; then latest eligible revision within period.
    chosen = max(candidates, key=lambda row: (row["reference_month"], timestamp(row["vintage_utc"])))
    return dict(status="available", row=chosen)


def build():
    acquisition = json.loads((HERE / "acquisition.json").read_text())
    rows, failures = [], []
    for record in acquisition["records"]:
        sources = [
            attempt
            for attempt in record["attempts"]
            if attempt["representation"] == "fraser_archived_bls_release_pdf" and attempt["status"] != "fetch_failed"
        ]
        if not sources:
            failures.append(dict(reference_month=record["reference_month"], reason="no_downloaded_release_pdf"))
            continue
        source = sources[0]
        try:
            rows.extend(parse_release(HERE / source["file"], record, source))
        except ValueError as error:
            failures.append(dict(reference_month=record["reference_month"], reason=str(error)))
    selections = []
    for year in range(1998, 2025, 2):
        for kind, series in SERIES.items():
            cutoff = cutoff_for(year)
            selected = select(
                rows, cutoff_utc=cutoff, election_date=election_day(year).isoformat(), horizon_days=14, series_id=series
            )
            selections.append(
                dict(
                    cycle=year,
                    kind=kind,
                    cutoff_utc=cutoff,
                    election_date=election_day(year).isoformat(),
                    horizon_days=14,
                    **selected,
                )
            )
    (HERE / "rows.json").write_text(
        json.dumps(dict(schema="bls_cpi_release_rows_v1", rows=rows, failures=failures), indent=2) + "\n"
    )
    (HERE / "selections.json").write_text(
        json.dumps(
            dict(
                schema="bls_cpi_h14_selection_v1",
                cutoff_convention="end_of_H14_calendar_day_UTC",
                selections=selections,
            ),
            indent=2,
        )
        + "\n"
    )
    files = [
        HERE / "acquisition.json",
        HERE / "rows.json",
        HERE / "selections.json",
        Path(__file__),
        HERE / "acquire.py",
    ]
    files.extend(
        HERE / source["file"]
        for record in acquisition["records"]
        for source in record["attempts"]
        if source.get("file")
    )
    receipt = dict(
        schema="bls_cpi_vintage_build_receipt_v1",
        parser="pdftotext_layout_table1_v1",
        pdftotext_version=subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr.splitlines()[0],
        original_bls_server_bytes_authenticated=False,
        archive_copy_provenance=(
            "Federal Reserve Bank of St Louis FRASER archived BLS release PDF; source header and Table1 checked"
        ),
        source_files=[
            dict(
                file=str(path.relative_to(HERE)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
            )
            for path in files
        ],
        fit=False,
    )
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                parsed_rows=len(rows),
                failures=failures,
                selected=[
                    dict(
                        cycle=r["cycle"],
                        kind=r["kind"],
                        status=r["status"],
                        reference=r.get("row", {}).get("reference_month"),
                        value=r.get("row", {}).get("value"),
                    )
                    for r in selections
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
