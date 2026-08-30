"""
Parse Cygnal polling report PDFs/text into WetherVane poll crosstab columns.

Cygnal public decks typically expose two useful structures:

1. A demographic profile page with sample composition by race, education,
   age, and community type. These become xt_* columns.
2. A ballot crosstab block with per-group Republican/Democratic shares.
   These become xt_vote_* two-party Democratic vote-share columns.

The parser accepts either a PDF path or text extracted from a PDF. Keeping the
text parser separate makes tests deterministic while still supporting live
report ingestion when a Cygnal PDF is available.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLLS_CSV = PROJECT_ROOT / "data" / "polls" / "polls_2026.csv"

POLL_FIELDNAMES = {
    "race",
    "geography",
    "geo_level",
    "dem_share",
    "n_sample",
    "date",
    "pollster",
    "notes",
    "methodology",
}

COMPOSITION_LABEL_MAP = {
    "White or Caucasian": "xt_race_white",
    "Black or African American": "xt_race_black",
    "Hispanic or Latino": "xt_race_hispanic",
    "Asian or Pacific Islander": "xt_race_asian",
    "At least College": "xt_education_college",
    "College graduate": "xt_education_college",
    "No degree": "xt_education_noncollege",
    "65+": "xt_age_senior",
    "Urban": "xt_urbanicity_urban",
    "Rural": "xt_urbanicity_rural",
}

VOTE_LABEL_MAP = {
    "White or Caucasian": "xt_vote_race_white",
    "Black or African American": "xt_vote_race_black",
    "Hispanic or Latino": "xt_vote_race_hispanic",
    "Asian or Pacific Islander": "xt_vote_race_asian",
    "At least College": "xt_vote_education_college",
    "College graduate": "xt_vote_education_college",
    "No degree": "xt_vote_education_noncollege",
    "65+": "xt_vote_age_senior",
    "Urban": "xt_vote_urbanicity_urban",
    "Rural": "xt_vote_urbanicity_rural",
}

KNOWN_LABELS = sorted(
    set(COMPOSITION_LABEL_MAP) | set(VOTE_LABEL_MAP),
    key=len,
    reverse=True,
)

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def normalize_text(text: str) -> str:
    """Normalize common PDF extraction artifacts in Cygnal decks."""
    replacements = {
        "\x00": "",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00b1": "+/-",
        "\u00a0": " ",
        "\u2019": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[ \t]+", " ", text)


def two_party_dem_share(dem_pct: float, rep_pct: float) -> Optional[float]:
    """Convert Democratic/Republican percentages to two-party Democratic share."""
    if dem_pct + rep_pct == 0:
        return None
    return dem_pct / (dem_pct + rep_pct)


def _percentages(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", line)]


def _is_percentage_only_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    without_pcts = re.sub(r"\d+(?:\.\d+)?\s*%", "", stripped)
    return without_pcts.strip() == ""


def _find_known_labels(line: str, *, allowed: set[str]) -> list[str]:
    """Return canonical labels appearing in the given line, in order."""
    line = normalize_text(line)
    matches: list[tuple[int, str]] = []
    for label in KNOWN_LABELS:
        if label not in allowed:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])")
        for match in pattern.finditer(line):
            matches.append((match.start(), label))
    matches.sort(key=lambda item: item[0])

    labels: list[str] = []
    last_start = -1
    for start, label in matches:
        if start == last_start:
            continue
        labels.append(label)
        last_start = start
    return labels


def parse_header(text: str) -> dict[str, str | int]:
    """Extract field dates and sample size from Cygnal report header text."""
    normalized = normalize_text(text)
    result: dict[str, str | int] = {}

    header_match = re.search(
        r"([A-Za-z]+)\s+(\d{1,2})\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2}),\s*(\d{4})\s*\|\s*n\s*=\s*([\d,]+)",
        normalized,
        re.IGNORECASE,
    )
    if not header_match:
        raise ValueError("Could not parse date/sample from Cygnal report header")

    month1_name = header_match.group(1).lower()
    day1 = int(header_match.group(2))
    month2_name = (header_match.group(3) or header_match.group(1)).lower()
    day2 = int(header_match.group(4))
    year = int(header_match.group(5))
    n_sample = int(header_match.group(6).replace(",", ""))

    month1 = MONTH_MAP.get(month1_name)
    month2 = MONTH_MAP.get(month2_name)
    if month1 is None or month2 is None:
        raise ValueError("Could not parse month names from Cygnal report header")

    result["date_start"] = f"{year}-{month1:02d}-{day1:02d}"
    result["date_end"] = f"{year}-{month2:02d}-{day2:02d}"
    result["n_sample"] = n_sample
    return result


def parse_sample_composition(text: str) -> dict[str, float]:
    """Parse Cygnal demographic-profile text into xt_* composition columns."""
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    result: dict[str, float] = {}
    allowed = set(COMPOSITION_LABEL_MAP)

    for i, line in enumerate(lines):
        labels = _find_known_labels(line, allowed=allowed)
        if not labels:
            continue

        inline_values = _percentages(line)
        if len(labels) == 1 and len(inline_values) == 1:
            col = COMPOSITION_LABEL_MAP.get(labels[0])
            if col:
                result[col] = inline_values[0] / 100.0
            continue
        if len(labels) == 1 and not inline_values:
            for lookahead in range(i + 1, min(i + 3, len(lines))):
                values = _percentages(lines[lookahead])
                if len(values) == 1 and _is_percentage_only_line(lines[lookahead]):
                    col = COMPOSITION_LABEL_MAP.get(labels[0])
                    if col:
                        result[col] = values[0] / 100.0
                    break
            continue

        if len(labels) < 2:
            continue

        value_line = ""
        for lookahead in range(i + 1, min(i + 4, len(lines))):
            candidate = lines[lookahead]
            if _is_percentage_only_line(candidate):
                value_line = candidate
                break
        if not value_line:
            continue

        values = _percentages(value_line)
        if len(values) != len(labels):
            continue
        for label, pct in zip(labels, values):
            col = COMPOSITION_LABEL_MAP.get(label)
            if col:
                result[col] = pct / 100.0

    return result


def _classify_response_line(line: str) -> Optional[str]:
    prefix = re.split(r"\d+(?:\.\d+)?\s*%", line.strip(), maxsplit=1)[0].lower()
    prefix = re.sub(r"[^a-z]+", "", prefix)
    if not prefix:
        return None
    if "democratic" in prefix or prefix.startswith("democrat") or prefix == "dem":
        return "dem"
    if "republican" in prefix or prefix.startswith("republican") or prefix == "rep":
        return "rep"
    return None


def parse_demographic_vote_shares(text: str) -> dict[str, float]:
    """Parse Cygnal ballot crosstab blocks into xt_vote_* columns."""
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    result: dict[str, float] = {}
    allowed = set(VOTE_LABEL_MAP)

    for i, line in enumerate(lines):
        labels = _find_known_labels(line, allowed=allowed)
        if not labels:
            continue

        if len(labels) == 1:
            values = _percentages(line)
            if len(values) >= 2:
                col = VOTE_LABEL_MAP.get(labels[0])
                share = two_party_dem_share(values[1], values[0])
                if col and share is not None:
                    result[col] = share
                continue
            continue

        rep_values: list[float] | None = None
        dem_values: list[float] | None = None

        for lookahead in range(i + 1, min(i + 6, len(lines))):
            row_type = _classify_response_line(lines[lookahead])
            values = _percentages(lines[lookahead])
            if not values:
                continue
            if row_type == "rep":
                rep_values = values
            elif row_type == "dem":
                dem_values = values
            if rep_values is not None and dem_values is not None:
                break

        if rep_values is None or dem_values is None:
            pct_rows = [
                _percentages(candidate)
                for candidate in lines[i + 1 : min(i + 4, len(lines))]
                if _is_percentage_only_line(candidate)
            ]
            if len(pct_rows) >= 2:
                rep_values, dem_values = pct_rows[0], pct_rows[1]

        if rep_values is None or dem_values is None:
            continue
        if len(rep_values) != len(labels) or len(dem_values) != len(labels):
            continue

        for label, rep_pct, dem_pct in zip(labels, rep_values, dem_values):
            col = VOTE_LABEL_MAP.get(label)
            share = two_party_dem_share(dem_pct, rep_pct)
            if col and share is not None:
                result[col] = share

    return result


def parse_cygnal_text(text: str) -> dict[str, object]:
    """Parse all supported Cygnal fields from extracted report text."""
    normalized = normalize_text(text)
    parsed: dict[str, object] = {
        "pollster": "Cygnal",
    }
    parsed.update(parse_header(normalized))
    parsed.update(parse_sample_composition(normalized))
    parsed.update(parse_demographic_vote_shares(normalized))
    return parsed


def extract_text(path: str | Path) -> str:
    """Extract text from a Cygnal PDF or read an already-extracted text file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text()


def parse_cygnal_report(path: str | Path) -> dict[str, object]:
    """Parse a Cygnal report PDF/text file into polls_2026.csv-compatible fields."""
    return parse_cygnal_text(extract_text(path))


def update_polls_csv(
    extracted: dict[str, object],
    race_filter: Optional[str] = None,
    pollster_filter: str = "Cygnal",
) -> bool:
    """Update matching Cygnal rows in polls_2026.csv with parsed xt_* fields."""
    if not POLLS_CSV.exists():
        logger.error("Polls CSV not found: %s", POLLS_CSV)
        return False

    with POLLS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        logger.error("Empty CSV")
        return False

    update_columns = [
        key
        for key in extracted
        if key in fieldnames and (key.startswith("xt_") or key in POLL_FIELDNAMES)
    ]
    if not update_columns:
        logger.warning("No polls_2026.csv-compatible fields parsed")
        return False

    updated = False
    for row in rows:
        pollster_match = pollster_filter.lower() in row.get("pollster", "").lower()
        race_match = race_filter is None or race_filter in row.get("race", "")
        if not pollster_match or not race_match:
            continue
        for col in update_columns:
            value = extracted[col]
            if value is None:
                continue
            row[col] = f"{value:.6f}" if isinstance(value, float) else str(value)
        updated = True
        logger.info("Updated poll: %s / %s", row.get("race"), row.get("date"))

    if updated:
        with POLLS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote updated CSV to %s", POLLS_CSV)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Cygnal reports for xt_* poll crosstab fields."
    )
    parser.add_argument("path", help="Path to a Cygnal report PDF or extracted text")
    parser.add_argument("--race", help="Optional race substring for CSV updates")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write parsed values back into data/polls/polls_2026.csv",
    )
    args = parser.parse_args()

    parsed = parse_cygnal_report(args.path)
    for key in sorted(parsed):
        print(f"{key}: {parsed[key]}")

    if args.update:
        success = update_polls_csv(parsed, race_filter=args.race)
        raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
