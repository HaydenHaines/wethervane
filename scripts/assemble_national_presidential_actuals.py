"""Assemble national county-level presidential actuals for the backtest harness.

Reads the cached MEDSL tab file (data/raw/medsl/county_presidential_2000_2024.tab)
and produces one parquet per election year covering all 51 states/DC.

This supersedes the FL+GA+AL-only outputs produced by
src/assembly/fetch_medsl_county_presidential.py for backtest purposes.

Output (one parquet per election year, data/assembled/):
    medsl_county_presidential_{year}.parquet
    Columns:
        county_fips         -- zero-padded 5-digit string (e.g. "01001")
        state_abbr          -- 2-letter USPS abbreviation (from state_po)
        pres_dem_{year}     -- Democratic candidate votes (sum across candidates)
        pres_rep_{year}     -- Republican candidate votes (sum across candidates)
        pres_total_{year}   -- All-candidate total votes for the county
        pres_dem_share_{year} -- Two-party dem share: dem / (dem + rep)

Vote share is TWO-PARTY (not total-vote) to match the backtest harness convention.
The existing FL/GA/AL-only files are overwritten with national data.

Usage:
    python scripts/assemble_national_presidential_actuals.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "medsl" / "county_presidential_2000_2024.tab"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"

# All presidential election years present in the MEDSL dataset.
PRES_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]


def _resolve_voting_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse split-mode rows to one row per (year, state_po, county_fips, party).

    MEDSL reports some states broken out by voting mode (ABSENTEE, ELECTION DAY,
    etc.) and others with a single TOTAL row. A minority of states have NaN for
    mode and report one row per candidate — those are treated as already-collapsed.

    Strategy:
    - Where a (year, county_fips) has at least one TOTAL row, keep only TOTAL rows.
    - Where no TOTAL row exists, sum candidatevotes across all mode rows per
      (year, state_po, county_fips, party); keep totalvotes from the first row
      because MEDSL repeats the county total in every mode row.
    """
    # Normalise mode for comparison (strip whitespace, upper-case, handle NaN)
    mode_upper = df["mode"].fillna("").str.strip().str.upper()

    # Identify which (year, county_fips) pairs have at least one TOTAL row
    is_total_row = mode_upper == "TOTAL"
    has_total_index = (
        df[is_total_row]
        .groupby(["year", "county_fips"])
        .size()
        .reset_index()[["year", "county_fips"]]
        .assign(_has_total=True)
    )
    df = df.merge(has_total_index, on=["year", "county_fips"], how="left")
    df["_has_total"] = df["_has_total"].fillna(False)

    total_rows = df[df["_has_total"] & is_total_row].copy()

    # For counties without a TOTAL row, sum across all mode rows.
    # This handles both explicit modes (ABSENTEE, ELECTION DAY…) and NaN-mode
    # states that already have one row per candidate.
    non_total = df[~df["_has_total"]]
    # Include "office" in the groupby so the presidential-race filter downstream
    # can still match on office name. All rows for a given county-party-year have
    # the same office value, so grouping by it is safe and lossless.
    summed_rows = (
        non_total
        .groupby(
            ["year", "state_po", "county_fips", "office", "party"],
            dropna=False,
        )
        .agg(
            candidatevotes=("candidatevotes", "sum"),
            totalvotes=("totalvotes", "first"),
        )
        .reset_index()
    )

    combined = pd.concat([total_rows, summed_rows], ignore_index=True)
    log.info(
        "After mode resolution: %d rows (raw: %d)", len(combined), len(df)
    )
    return combined


def _filter_presidential(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only DEMOCRAT and REPUBLICAN rows for presidential office."""
    # The MEDSL file uses "US PRESIDENT" for the office column.
    # We match on contains("PRESIDENT") to be robust to minor variations.
    is_pres = (
        df["office"].str.upper().str.contains("PRESIDENT", na=False)
        & ~df["office"].str.upper().str.contains("VICE", na=False)
    )
    is_major_party = df["party"].isin({"DEMOCRAT", "REPUBLICAN"})
    return df[is_pres & is_major_party].copy()


def _aggregate_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Produce one row per county for *year* with D/R vote totals and shares.

    Two-party dem share = dem_votes / (dem_votes + rep_votes).
    county_fips is zero-padded to 5 digits.
    state_abbr is taken directly from state_po.
    """
    yr = df[df["year"] == year].copy()
    if yr.empty:
        log.warning("No rows for year %d — skipping", year)
        return pd.DataFrame()

    # Normalise county_fips: MEDSL stores as float (e.g. 12086.0); convert to
    # zero-padded 5-digit string.
    yr["county_fips"] = (
        pd.to_numeric(yr["county_fips"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    dem = (
        yr[yr["party"] == "DEMOCRAT"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_dem_{year}")
    )
    rep = (
        yr[yr["party"] == "REPUBLICAN"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_rep_{year}")
    )
    # totalvotes is the all-candidate county total repeated in every row;
    # grab it from DEMOCRAT rows so we have one value per county.
    total_all = (
        yr[yr["party"] == "DEMOCRAT"]
        .groupby("county_fips")["totalvotes"]
        .first()
        .rename(f"pres_total_{year}")
    )
    # state_abbr: stable per county, pull from DEMOCRAT rows
    state_abbr = (
        yr[yr["party"] == "DEMOCRAT"]
        .groupby("county_fips")["state_po"]
        .first()
        .rename("state_abbr")
    )

    result = pd.concat([dem, rep, total_all, state_abbr], axis=1).reset_index()

    # Two-party dem share. Counties where either party is missing get NaN.
    dem_col = f"pres_dem_{year}"
    rep_col = f"pres_rep_{year}"
    two_party_total = result[dem_col] + result[rep_col]
    result[f"pres_dem_share_{year}"] = result[dem_col] / two_party_total

    return result[[
        "county_fips",
        "state_abbr",
        dem_col,
        rep_col,
        f"pres_total_{year}",
        f"pres_dem_share_{year}",
    ]]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw MEDSL tab file not found: {RAW_PATH}\n"
            "Run src/assembly/fetch_medsl_county_presidential.py first to cache it."
        )

    log.info("Loading %s", RAW_PATH)
    df = pd.read_csv(RAW_PATH, sep="\t", low_memory=False)
    log.info("Raw rows: %d", len(df))

    df = _resolve_voting_modes(df)
    df = _filter_presidential(df)
    log.info("Rows after presidential D/R filter: %d", len(df))

    years_present = sorted(df["year"].unique())
    log.info("Years in data: %s", years_present)

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)

    for year in PRES_YEARS:
        if year not in years_present:
            log.warning("Year %d not present in data — skipping", year)
            continue

        agg = _aggregate_year(df, year)
        if agg.empty:
            continue

        out_path = ASSEMBLED_DIR / f"medsl_county_presidential_{year}.parquet"
        agg.to_parquet(out_path, index=False)
        log.info(
            "  %d → %s  (%d counties, %d states)",
            year,
            out_path.name,
            len(agg),
            agg["state_abbr"].nunique(),
        )


if __name__ == "__main__":
    main()
