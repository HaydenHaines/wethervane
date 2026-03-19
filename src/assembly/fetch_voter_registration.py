"""
Stage 1 data assembly: fetch Florida voter registration by party and county.

Source: Florida Division of Elections — Book Closing Reports
URL: https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/
Data: Active registered voters by party, county-level snapshots at each election's
      book closing date (29 days before the election).

Florida is the primary focus state. The model covers FL+GA+AL but voter registration
data in this format is only available from Florida's Division of Elections. Georgia
and Alabama do not publish equivalent county-by-party registration tables.

**Why book closing dates?**
Book closing is the registration deadline 29 days before each election. It provides
a clean, comparable snapshot of the electorate as it stood immediately before voting.
Using book-closing dates rather than mid-year snapshots avoids registration churn
and makes cross-cycle comparisons meaningful.

**Election cycles covered (book closing dates)**:
  2016 General: October 18, 2016 (Election: November 8, 2016)
  2018 General: October 9, 2018  (Election: November 6, 2018)
  2020 General: October 6, 2020  (Election: November 3, 2020)
  2022 General: October 11, 2022 (Election: November 8, 2022)
  2024 General: October 7, 2024  (Election: November 5, 2024)

**Key party columns**:
  Republican Party of Florida  (REP)
  Florida Democratic Party     (DEM)
  No Party Affiliation         (NPA)
  Minor parties (all others)   (OTHER)
  Total registered             (TOTAL)

**Excel format variation**:
  2016 files have header in row 7 (0-indexed) with columns including ElectionDate,
  BookClosing, JurisType, CountyName then parties. Later files (2018-2024) have
  header in row 8 with just CountyName then parties. Both are handled by
  _parse_registration_excel().

**Florida county FIPS codes**:
  Florida has 67 counties. FIPS codes run from 12001 (Alachua) to 12133 (Washington)
  in alphabetical order in odd increments of 2. We apply a hardcoded mapping because
  the DOS files report by county name, not FIPS code.

Output: data/raw/fl_voter_registration.parquet
  Columns: county_fips, county_name, election_year, book_closing_date,
           rep, dem, npa, other, total
  Rows: one per county × election cycle (67 counties × 5 cycles = 335 rows max)
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "fl_voter_registration.parquet"

# Florida DOS base URL (files redirect to files.floridados.gov automatically)
FL_DOS_BASE = "https://files.floridados.gov"

# Book-closing "By Party" Excel files from the FL DOS book closing reports page.
# Format: (election_year, book_closing_date_str, media_id)
# Retrieved from: https://dos.fl.gov/elections/data-statistics/voter-registration-statistics/
#   bookclosing/bookclosing-reports-regular/
ELECTION_FILES: list[tuple[int, str, str]] = [
    (2016, "2016-10-18", "697211"),  # 2016general_party.xlsx
    (2018, "2018-10-09", "700198"),  # 2018gen_party.xlsx
    (2020, "2020-10-06", "703608"),  # 1-party-by-county.xlsx
    (2022, "2022-10-11", "706002"),  # 1-2022-gen_bycountybyparty.xlsx
    (2024, "2024-10-07", "708493"),  # 1-party-by-county.xlsx
]

# Polite delay between downloads (seconds)
REQUEST_DELAY = 1.0

# Florida county name → 5-digit FIPS code mapping.
# Florida's 67 counties run alphabetically with FIPS 12001–12133 (odd numbers only).
FL_COUNTY_FIPS: dict[str, str] = {
    "Alachua": "12001",
    "Baker": "12003",
    "Bay": "12005",
    "Bradford": "12007",
    "Brevard": "12009",
    "Broward": "12011",
    "Calhoun": "12013",
    "Charlotte": "12015",
    "Citrus": "12017",
    "Clay": "12019",
    "Collier": "12021",
    "Columbia": "12023",
    "DeSoto": "12027",
    "Dixie": "12029",
    "Duval": "12031",
    "Escambia": "12033",
    "Flagler": "12035",
    "Franklin": "12037",
    "Gadsden": "12039",
    "Gilchrist": "12041",
    "Glades": "12043",
    "Gulf": "12045",
    "Hamilton": "12047",
    "Hardee": "12049",
    "Hendry": "12051",
    "Hernando": "12053",
    "Highlands": "12055",
    "Hillsborough": "12057",
    "Holmes": "12059",
    "Indian River": "12061",
    "Jackson": "12063",
    "Jefferson": "12065",
    "Lafayette": "12067",
    "Lake": "12069",
    "Lee": "12071",
    "Leon": "12073",
    "Levy": "12075",
    "Liberty": "12077",
    "Madison": "12079",
    "Manatee": "12081",
    "Marion": "12083",
    "Martin": "12085",
    "Miami-Dade": "12086",
    "Monroe": "12087",
    "Nassau": "12089",
    "Okaloosa": "12091",
    "Okeechobee": "12093",
    "Orange": "12095",
    "Osceola": "12097",
    "Palm Beach": "12099",
    "Pasco": "12101",
    "Pinellas": "12103",
    "Polk": "12105",
    "Putnam": "12107",
    "St. Johns": "12109",
    "St. Lucie": "12111",
    "Santa Rosa": "12113",
    "Sarasota": "12115",
    "Seminole": "12117",
    "Sumter": "12119",
    "Suwannee": "12121",
    "Taylor": "12123",
    "Union": "12125",
    "Volusia": "12127",
    "Wakulla": "12129",
    "Walton": "12131",
    "Washington": "12133",
}

# Canonical party column names in the output
OUTPUT_COLUMNS = [
    "county_fips",
    "county_name",
    "election_year",
    "book_closing_date",
    "rep",
    "dem",
    "npa",
    "other",
    "total",
]


def build_url(media_id: str) -> str:
    """Construct the FL DOS file download URL for a given media ID.

    Args:
        media_id: Numeric media ID string from the FL DOS URL path, e.g. "708493".

    Returns:
        Full URL to the Excel file on files.floridados.gov.
    """
    return f"{FL_DOS_BASE}/media/{media_id}/"


def download_excel(media_id: str) -> bytes | None:
    """Download a FL DOS Excel file by its media ID.

    Args:
        media_id: Numeric media ID string.

    Returns:
        Raw bytes of the Excel file, or None on download failure.
    """
    url = build_url(media_id)
    log.info("  Downloading media ID %s (%s)...", media_id, url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        log.warning("  HTTP error for media ID %s: %s", media_id, exc)
        return None


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Find the row index containing the county data column headers.

    The FL DOS files have varying numbers of metadata rows before the actual
    data table. We detect the header row by finding a row where one of the
    cells contains "County" (case-insensitive).

    Args:
        df_raw: Raw DataFrame read with header=None.

    Returns:
        0-based row index of the header row.

    Raises:
        ValueError: If no header row containing "County" is found.
    """
    for i, row in df_raw.iterrows():
        row_str = " ".join(str(v) for v in row if pd.notna(v)).lower()
        if "county" in row_str:
            return int(i)
    raise ValueError("No header row containing 'County' found in Excel file")


def _parse_registration_excel(
    content: bytes,
    election_year: int,
    book_closing_date: str,
) -> pd.DataFrame:
    """Parse a FL DOS voter registration Excel file into a normalized DataFrame.

    Handles the format variation between 2016 (header row 7, ElectionDate/BookClosing
    prefix columns) and 2018–2024 (header row 8, just CountyName then parties).

    Logic:
    1. Read the Excel with no header to get raw layout
    2. Auto-detect the header row (contains "County")
    3. Re-read with that row as the header
    4. Locate canonical party columns (REP, DEM, NPA) by fuzzy name matching
    5. Sum remaining party columns into "other"
    6. Compute county FIPS codes from county name
    7. Drop the statewide total row (county name = "Total" or NaN)

    Args:
        content: Raw bytes of the Excel file.
        election_year: Election year (e.g. 2024), added as a column.
        book_closing_date: Book closing date string (ISO 8601), added as a column.

    Returns:
        Normalized DataFrame with OUTPUT_COLUMNS, one row per county.
        Empty DataFrame on parse failure.
    """
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df_raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    except Exception as exc:
        log.warning("  Excel parse error: %s", exc)
        return empty

    # Find header row
    try:
        header_row = _find_header_row(df_raw)
    except ValueError as exc:
        log.warning("  %s", exc)
        return empty

    # Re-read with proper header
    try:
        df = pd.read_excel(
            io.BytesIO(content),
            sheet_name=0,
            header=header_row,
            dtype=str,
        )
    except Exception as exc:
        log.warning("  Re-read with header row %d failed: %s", header_row, exc)
        return empty

    log.info("  Parsed %d rows × %d cols (header at row %d)", len(df), len(df.columns), header_row)

    # Normalize column names: strip whitespace, lower-case for searching
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Find the county name column (first column containing "county" case-insensitive,
    # or the first string column after any date/type prefix columns)
    county_col = None
    for col in df.columns:
        if "county" in str(col).lower():
            county_col = col
            break

    if county_col is None:
        log.warning("  No county column found; columns: %s", list(df.columns)[:10])
        return empty

    # Find party columns by fuzzy name matching
    cols_lower = {c: str(c).lower().strip() for c in df.columns}

    def _find_col(keywords: list[str]) -> str | None:
        """Return the first column whose name contains all keywords (case-insensitive)."""
        for col, lower_name in cols_lower.items():
            if all(kw in lower_name for kw in keywords):
                return col
        return None

    rep_col = _find_col(["republican"])
    dem_col = _find_col(["democrat"])
    # NPA differs by year: "No Party Affiliation" (2018+) vs "No Party Affiliation" (2016 has same)
    npa_col = _find_col(["no party"])
    total_col = _find_col(["total"])

    if not rep_col:
        log.warning("  Could not find Republican column; columns: %s", list(df.columns))
        return empty
    if not dem_col:
        log.warning("  Could not find Democrat column; columns: %s", list(df.columns))
        return empty
    if not npa_col:
        log.warning("  Could not find NPA column; columns: %s", list(df.columns))
        return empty
    if not total_col:
        log.warning("  Could not find Total column; columns: %s", list(df.columns))
        return empty

    log.info(
        "  Columns: county=%r, rep=%r, dem=%r, npa=%r, total=%r",
        county_col, rep_col, dem_col, npa_col, total_col,
    )

    # Extract county name and filter to real county rows (drop total/empty rows)
    df["_county_name"] = df[county_col].astype(str).str.strip()
    # Drop rows where county name is blank, NaN placeholder "nan", or "Total"
    county_mask = (
        df["_county_name"].notna()
        & (df["_county_name"] != "nan")
        & (~df["_county_name"].str.lower().isin(["total", ""]))
        & (df["_county_name"].str.len() > 1)
    )
    df = df[county_mask].copy()

    log.info("  After county filter: %d county rows", len(df))

    # Coerce party columns to numeric
    for col in [rep_col, dem_col, npa_col, total_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute "other" = total - rep - dem - npa (handles varying minor-party columns)
    df["_rep"] = df[rep_col]
    df["_dem"] = df[dem_col]
    df["_npa"] = df[npa_col]
    df["_total"] = df[total_col]
    df["_other"] = (df["_total"] - df["_rep"] - df["_dem"] - df["_npa"]).clip(lower=0)

    # Look up county FIPS
    def _lookup_fips(name: str) -> str | None:
        """Return FIPS for a county name, trying exact then stripped match."""
        if name in FL_COUNTY_FIPS:
            return FL_COUNTY_FIPS[name]
        # Try stripping trailing/leading whitespace (already done above)
        # Try case-insensitive match as fallback
        name_lower = name.lower()
        for k, v in FL_COUNTY_FIPS.items():
            if k.lower() == name_lower:
                return v
        return None

    df["county_fips"] = df["_county_name"].apply(_lookup_fips)

    n_no_fips = df["county_fips"].isna().sum()
    if n_no_fips > 0:
        missing_names = df[df["county_fips"].isna()]["_county_name"].tolist()
        log.warning(
            "  %d county names without FIPS mapping: %s",
            n_no_fips, missing_names[:10],
        )
        df = df[df["county_fips"].notna()]

    # Build output DataFrame
    result = pd.DataFrame({
        "county_fips": df["county_fips"],
        "county_name": df["_county_name"],
        "election_year": election_year,
        "book_closing_date": pd.to_datetime(book_closing_date),
        "rep": df["_rep"].astype(float),
        "dem": df["_dem"].astype(float),
        "npa": df["_npa"].astype(float),
        "other": df["_other"].astype(float),
        "total": df["_total"].astype(float),
    })

    return result[OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_election_cycle(
    election_year: int,
    book_closing_date: str,
    media_id: str,
) -> pd.DataFrame:
    """Fetch and parse one election cycle's voter registration data.

    Args:
        election_year: 4-digit election year.
        book_closing_date: Book closing date string (ISO 8601, e.g. "2024-10-07").
        media_id: FL DOS media ID for the "By Party" Excel file.

    Returns:
        Parsed DataFrame with OUTPUT_COLUMNS for that election cycle,
        or empty DataFrame on failure.
    """
    log.info("Fetching %d general election voter registration (media ID %s)...",
             election_year, media_id)
    content = download_excel(media_id)
    if content is None:
        log.warning("  Download failed for %d; skipping.", election_year)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = _parse_registration_excel(content, election_year, book_closing_date)
    log.info("  %d: %d county rows parsed", election_year, len(df))
    return df


def main(election_files: list[tuple[int, str, str]] | None = None) -> None:
    """Fetch FL voter registration data for all configured election cycles.

    Downloads the "By Party" book-closing Excel files from the FL DOS website,
    parses each into a normalized county-level DataFrame, combines them, and
    saves to data/raw/fl_voter_registration.parquet.

    Args:
        election_files: List of (election_year, book_closing_date, media_id) tuples.
            Defaults to ELECTION_FILES (all 5 configured cycles).
    """
    if election_files is None:
        election_files = ELECTION_FILES

    log.info(
        "Fetching FL voter registration data for %d election cycles",
        len(election_files),
    )
    log.info("Cycles: %s", [year for year, _, _ in election_files])
    log.info("Source: FL Division of Elections — Book Closing Reports")

    frames: list[pd.DataFrame] = []
    for i, (year, book_date, media_id) in enumerate(election_files):
        df = fetch_election_cycle(year, book_date, media_id)
        if not df.empty:
            frames.append(df)

        # Polite delay between downloads (skip after last)
        if i < len(election_files) - 1:
            time.sleep(REQUEST_DELAY)

    if not frames:
        log.error("No data retrieved for any election cycle. Aborting.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Summary
    n_rows = len(combined)
    n_years = combined["election_year"].nunique()
    n_counties = combined["county_fips"].nunique()
    log.info(
        "\nSummary: %d rows | %d election cycles | %d unique FL counties",
        n_rows, n_years, n_counties,
    )

    # Per-cycle summary
    for year, grp in combined.groupby("election_year"):
        total_reg = grp["total"].sum()
        log.info(
            "  %d: %d counties | %,.0f total registered voters",
            year, len(grp), total_reg,
        )

    # Validate FIPS format
    fips_ok = combined["county_fips"].str.match(r"^12\d{3}$")
    if not fips_ok.all():
        bad = combined[~fips_ok]["county_fips"].unique()
        log.warning("Non-FL FIPS codes detected (dropping): %s", bad[:10])
        combined = combined[fips_ok]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH,
        len(combined),
        len(combined.columns),
    )


if __name__ == "__main__":
    main()
