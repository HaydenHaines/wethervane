"""
Parse Quantus Insights crosstab reports to extract per-group vote shares and
sample-composition demographics.

Quantus publishes crosstabs in several channels: Substack posts with inline
HTML tables, PDF reports linked from their site, and a JS-rendered polls
portal at polls.quantusinsights.org.  The portal is behind a Vercel bot
challenge so cannot be fetched without a browser.  Substack and PDF reports
collapse cleanly to plain text when extracted with pdfplumber or BeautifulSoup
— and in both cases the crosstab table renders as one row per demographic
group with two or three trailing percentages (Dem, Rep, and optionally
Undecided/Other).

This parser is therefore **text-in, dict-out**: callers are responsible for
turning the source artifact into text (``pdfplumber.Page.extract_text()`` for
PDFs, ``BeautifulSoup.get_text()`` for Substack HTML, or just ``open().read()``
for saved fixtures).  Keeping the pattern-matching layer pure means the tests
do not depend on any specific extraction backend.

Two kinds of output:

1. **Crosstab vote shares** (``xt_vote_*``): per-demographic two-party
   Democratic share for the headline ballot question.  Drives Tier 1 W-vector
   construction in ``src/prediction/forecast_engine.py``.
2. **Sample composition** (``xt_*``): weighted fraction of the poll sample in
   each demographic group.  Required for Tier 2 observations.

Column names match the canonical set used by the Emerson scraper and the
Marist PDF parser so the downstream pipeline does not need pollster-specific
branches.

Usage:
    uv run python scripts/parse_quantus_report.py path/to/quantus_report.txt
    uv run python scripts/parse_quantus_report.py path/to/quantus_report.pdf
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Canonical label → xt_vote_* column name.  Keys are normalized (lowercase,
# whitespace collapsed) so we can match labels that appear with varying case
# and spacing in the source text.  When multiple raw labels map to the same
# canonical group (e.g., "65+" and "65 or older"), list each one.
VOTE_LABEL_MAP: dict[str, str] = {
    "white": "xt_vote_race_white",
    "black": "xt_vote_race_black",
    "african american": "xt_vote_race_black",
    "hispanic": "xt_vote_race_hispanic",
    "latino": "xt_vote_race_hispanic",
    "asian": "xt_vote_race_asian",
    "college graduate": "xt_vote_education_college",
    "college": "xt_vote_education_college",
    "college grad": "xt_vote_education_college",
    "4-year college": "xt_vote_education_college",
    "no college": "xt_vote_education_noncollege",
    "non-college": "xt_vote_education_noncollege",
    "not college graduate": "xt_vote_education_noncollege",
    "65+": "xt_vote_age_senior",
    "65 or older": "xt_vote_age_senior",
    "65 and older": "xt_vote_age_senior",
    "urban": "xt_vote_urbanicity_urban",
    "rural": "xt_vote_urbanicity_rural",
}

# Canonical label → xt_* sample composition column name.  Same label keys as
# VOTE_LABEL_MAP but the output column names omit the ``_vote_`` infix.
COMPOSITION_LABEL_MAP: dict[str, str] = {
    "white": "xt_race_white",
    "black": "xt_race_black",
    "african american": "xt_race_black",
    "hispanic": "xt_race_hispanic",
    "latino": "xt_race_hispanic",
    "asian": "xt_race_asian",
    "college graduate": "xt_education_college",
    "college": "xt_education_college",
    "college grad": "xt_education_college",
    "4-year college": "xt_education_college",
    "no college": "xt_education_noncollege",
    "non-college": "xt_education_noncollege",
    "not college graduate": "xt_education_noncollege",
    "65+": "xt_age_senior",
    "65 or older": "xt_age_senior",
    "65 and older": "xt_age_senior",
    "urban": "xt_urbanicity_urban",
    "rural": "xt_urbanicity_rural",
}

# Regex for a crosstab vote row — a label followed by at least two trailing
# percentages (Dem, Rep) and optionally a third (Undecided/Other).  The
# single-column composition row uses a separate regex below.
#
# We capture up to four percentages and let the caller decide which to use;
# this lets the same function cope with Quantus's common "D/R/Und" triple and
# with occasional four-column publications that add Other.
_VOTE_ROW = re.compile(
    r"^(?P<label>.+?)\s+"
    r"(?P<d>\d+)%\s+"
    r"(?P<r>\d+)%"
    r"(?:\s+\d+%){0,2}"
    r"\s*$"
)

# Regex for a sample-composition row — a label followed by exactly one
# trailing percentage.
_SINGLE_PCT_ROW = re.compile(r"^(?P<label>.+?)\s+(?P<pct>\d+)%\s*$")

# Heading lines that introduce the sample-composition table.  We only parse
# single-percentage rows after one of these headers has been seen, so that an
# incidental "Total 100%" line elsewhere in the document is not misread.
_COMPOSITION_HEADERS = (
    "SAMPLE COMPOSITION",
    "NATURE OF THE SAMPLE",
    "WEIGHTED SAMPLE",
)


def two_party_dem_share(dem_pct: float, rep_pct: float) -> Optional[float]:
    """Convert raw percentages to two-party Democratic share.

    Args:
        dem_pct: Democratic candidate percentage (0-100 scale).
        rep_pct: Republican candidate percentage (0-100 scale).

    Returns:
        Two-party share as a float in [0, 1], or None if both are zero.
    """
    if dem_pct + rep_pct == 0:
        return None
    return dem_pct / (dem_pct + rep_pct)


def _normalize_label(label: str) -> str:
    """Lowercase, strip, and collapse whitespace for robust label matching."""
    return re.sub(r"\s+", " ", label.strip().lower())


def parse_quantus_crosstab_text(text: str) -> dict[str, Optional[float]]:
    """Parse crosstab vote-share text into an ``xt_vote_*`` dict.

    Expects the Quantus demographic crosstab laid out one group per line, with
    the Democratic candidate in the first percentage column and the Republican
    candidate in the second.  Additional trailing columns (Undecided, Other)
    are ignored.

    An "Overall" / "All voters" / "Total" row produces a ``dem_share_topline``
    entry so callers can sanity-check against the published headline number.

    Args:
        text: Raw crosstab text.

    Returns:
        Dict mapping ``xt_vote_*`` column names to two-party Dem share values
        in [0, 1], plus optionally ``dem_share_topline``.
    """
    result: dict[str, Optional[float]] = {}

    for raw_line in text.split("\n"):
        match = _VOTE_ROW.match(raw_line.strip())
        if not match:
            continue

        label = _normalize_label(match.group("label"))
        dem_pct = int(match.group("d"))
        rep_pct = int(match.group("r"))
        dem_share = two_party_dem_share(dem_pct, rep_pct)

        if label in {"overall", "all voters", "total", "likely voters"}:
            result["dem_share_topline"] = dem_share
            continue

        column = VOTE_LABEL_MAP.get(label)
        if column is not None:
            result[column] = dem_share

    return result


def parse_quantus_composition_text(text: str) -> dict[str, float]:
    """Parse the sample-composition block into an ``xt_*`` dict.

    Only rows that appear *after* one of ``_COMPOSITION_HEADERS`` are
    considered.  This mirrors the Marist NOS pattern: the composition table is
    a distinct region of the document, and we must not interpret random
    single-percent rows from the narrative text as composition data.

    Args:
        text: Raw report text containing both crosstab and composition
            sections.  The order must be crosstab first, then composition, or
            an empty dict is returned.

    Returns:
        Dict mapping ``xt_*`` column names to fractions in [0, 1].  Empty if
        no composition section is found.
    """
    result: dict[str, float] = {}

    in_composition = False
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if any(header in stripped.upper() for header in _COMPOSITION_HEADERS):
            in_composition = True
            continue
        if not in_composition:
            continue

        match = _SINGLE_PCT_ROW.match(stripped)
        if not match:
            continue

        label = _normalize_label(match.group("label"))
        pct = int(match.group("pct"))
        column = COMPOSITION_LABEL_MAP.get(label)
        if column is not None:
            result[column] = pct / 100.0

    return result


def parse_quantus_report_text(text: str) -> dict[str, float]:
    """Parse a full Quantus report text, merging vote and composition dicts.

    Convenience wrapper that calls both extractors and returns a single dict.
    Sample-composition keys (``xt_*``) and vote-share keys (``xt_vote_*``) do
    not collide, so the merge is unambiguous.

    Args:
        text: Raw report text (from PDF extraction, HTML scrape, or saved
            fixture).

    Returns:
        Merged dict with ``xt_vote_*``, ``xt_*``, and (optionally)
        ``dem_share_topline`` keys.
    """
    merged: dict[str, float] = {}
    vote = parse_quantus_crosstab_text(text)
    for key, value in vote.items():
        if value is not None:
            merged[key] = value
    composition = parse_quantus_composition_text(text)
    merged.update(composition)
    return merged


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full text from a PDF using pdfplumber.

    Imported lazily so callers that only parse fixture text files do not
    require pdfplumber to be installed.
    """
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def parse_quantus_report(path: str | Path) -> dict[str, float]:
    """Parse a Quantus report from a ``.txt`` or ``.pdf`` file on disk.

    Args:
        path: Path to the Quantus report.  ``.pdf`` files are extracted via
            pdfplumber; any other extension is read as UTF-8 text.

    Returns:
        Same dict structure as ``parse_quantus_report_text()``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Quantus report not found: {path}")

    if path.suffix.lower() == ".pdf":
        text = _extract_text_from_pdf(path)
    else:
        text = path.read_text(encoding="utf-8")

    return parse_quantus_report_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Quantus Insights crosstab reports for demographic vote shares and sample composition."
    )
    parser.add_argument("report_path", help="Path to Quantus report (.pdf or .txt)")
    args = parser.parse_args()

    extracted = parse_quantus_report(args.report_path)

    if not extracted:
        logger.warning("No crosstab or composition data extracted from %s", args.report_path)
        sys.exit(1)

    print(f"\nExtracted Quantus crosstab data from {Path(args.report_path).name}:")
    print("-" * 60)
    for key, value in sorted(extracted.items()):
        print(f"  {key}: {value:.4f}")
    print()


if __name__ == "__main__":
    main()
