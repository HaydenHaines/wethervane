"""Fetch MEDSL county-level presidential returns 2000–2024.

Source: MIT Election Data + Science Lab, Harvard Dataverse
  doi:10.7910/DVN/VOQCHQ

Downloads a single unified CSV covering all US presidential elections
2000–2024 at the county level. Filters to FL, GA, AL. Computes
two-party dem share per county per year.

Output (one parquet per election year, data/assembled/):
  medsl_county_presidential_{year}.parquet
  Columns: county_fips, state_abbr, pres_dem_{year}, pres_rep_{year},
           pres_total_{year}, pres_dem_share_{year}

Cache: data/raw/medsl/county_presidential_2000_2024.tab
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
CACHE_PATH = RAW_DIR / "county_presidential_2000_2024.tab"

DATAVERSE_DOI = "doi:10.7910/DVN/VOQCHQ"
DATAVERSE_API = "https://dataverse.harvard.edu/api"

# Maps fips_prefix → state abbreviation (STATES.values() = {"FL", "GA", "AL"})
# state_po in MEDSL data is the abbreviation (FL, GA, AL)
STATES: dict[str, str] = {"12": "FL", "13": "GA", "01": "AL"}

PRES_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]


def _dataverse_download(doi: str, cache_path: Path) -> Path:
    """Download primary data file from a Harvard Dataverse DOI."""
    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        return cache_path

    # List files in dataset
    list_url = f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(list_url, params={"persistentId": doi}, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]

    # Find the main tabular data file (CSV or tab-delimited, skip README/codebook/sources)
    data_files = [
        f for f in files
        if f["dataFile"]["filename"].lower().endswith((".csv", ".tab"))
        and "readme" not in f["dataFile"]["filename"].lower()
        and "codebook" not in f["dataFile"]["filename"].lower()
        and "sources" not in f["dataFile"]["filename"].lower()
        and ".md" not in f["dataFile"]["filename"].lower()
    ]
    if not data_files:
        raise FileNotFoundError(f"No tabular data file found in {doi}")

    # Pick the largest file (most likely to be the main data file)
    main_file = max(data_files, key=lambda f: f["dataFile"].get("filesize", 0))
    file_id = main_file["dataFile"]["id"]
    filename = main_file["dataFile"]["filename"]
    log.info("Downloading %s (id=%s)...", filename, file_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dl_url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    with requests.get(dl_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

    log.info("Saved → %s", cache_path)
    return cache_path


def filter_presidential_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only D/R presidential rows for FL, GA, AL.

    Expects state_po column to contain state abbreviations (FL, GA, AL).
    STATES.values() provides the set of target abbreviations.
    """
    mask = (
        (df["office"].str.upper().str.contains("PRESIDENT"))
        & (~df["office"].str.upper().str.contains("VICE"))
        & (df["party_simplified"].isin({"DEMOCRAT", "REPUBLICAN"}))
        & (df["state_po"].isin(set(STATES.values())))
    )
    return df[mask].copy()


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate filtered rows to one row per county for a given year.

    Returns county_fips, state_abbr, pres_dem_{year}, pres_rep_{year},
    pres_total_{year}, pres_dem_share_{year}.
    """
    yr = df[df["year"] == year].copy()
    # Convert county_fips: handle float representation (e.g. 12086.0 → "12086")
    yr["county_fips"] = (
        pd.to_numeric(yr["county_fips"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )

    dem = (
        yr[yr["party_simplified"] == "DEMOCRAT"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_dem_{year}")
    )
    rep = (
        yr[yr["party_simplified"] == "REPUBLICAN"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"pres_rep_{year}")
    )
    result = pd.concat([dem, rep], axis=1).reset_index()
    result[f"pres_total_{year}"] = result[f"pres_dem_{year}"] + result[f"pres_rep_{year}"]
    result[f"pres_dem_share_{year}"] = (
        result[f"pres_dem_{year}"] / result[f"pres_total_{year}"]
    )
    # STATES maps fips_prefix → abbreviation
    result["state_abbr"] = result["county_fips"].str[:2].map(STATES)
    return result[["county_fips", "state_abbr",
                   f"pres_dem_{year}", f"pres_rep_{year}",
                   f"pres_total_{year}", f"pres_dem_share_{year}"]]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MEDSL column names and data to the canonical schema.

    The actual downloaded file uses 'party' instead of 'party_simplified',
    has county_fips as float, and includes rows for each voting mode.
    This function:
      - Renames 'party' → 'party_simplified' (uppercase already)
      - Resolves voting mode: use TOTAL rows where they exist; for counties
        that have no TOTAL row (e.g. GA 2020 splits by absentee/election day),
        sum across all non-TOTAL modes to produce equivalent totals.

    Result has one row per (year, state_po, county_fips, party_simplified).
    """
    if "party" in df.columns and "party_simplified" not in df.columns:
        df = df.rename(columns={"party": "party_simplified"})

    if "mode" not in df.columns:
        return df

    # Identify which (year, county_fips) combos have a TOTAL row
    has_total = (
        df[df["mode"].str.upper().str.strip() == "TOTAL"]
        .groupby(["year", "county_fips"])
        .size()
        .reset_index(drop=False)[["year", "county_fips"]]
        .assign(_has_total=True)
    )
    df = df.merge(has_total, on=["year", "county_fips"], how="left")
    df["_has_total"] = df["_has_total"].fillna(False)

    # For counties with TOTAL: keep only TOTAL rows
    # For counties without TOTAL: sum across all mode rows
    total_rows = df[df["_has_total"] & (df["mode"].str.upper().str.strip() == "TOTAL")]
    summed_rows = (
        df[~df["_has_total"]]
        .groupby(["year", "state_po", "county_fips", "office", "party_simplified"],
                 dropna=False)
        .agg(candidatevotes=("candidatevotes", "sum"),
             totalvotes=("totalvotes", "first"))
        .reset_index()
    )

    combined = pd.concat([total_rows, summed_rows], ignore_index=True)
    log.info("After mode resolution: %d rows (from %d raw)", len(combined), len(df))
    return combined


def main() -> None:
    csv_path = _dataverse_download(DATAVERSE_DOI, CACHE_PATH)
    # Determine separator from file extension
    sep = "\t" if str(csv_path).endswith(".tab") else ","
    log.info("Loading data file (sep=%r)...", sep)
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    log.info("Raw rows: %d", len(df))

    df = _normalize_columns(df)
    df_filtered = filter_presidential_rows(df)
    log.info("After filter (FL+GA+AL D/R pres only): %d rows", len(df_filtered))

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    years_in_data = sorted(df_filtered["year"].unique())
    log.info("Years in data: %s", years_in_data)

    for year in PRES_YEARS:
        if year not in years_in_data:
            log.warning("Year %d not in dataset — skipping", year)
            continue
        agg = aggregate_county_year(df_filtered, year)
        out = ASSEMBLED_DIR / f"medsl_county_presidential_{year}.parquet"
        agg.to_parquet(out, index=False)
        log.info("  %d → %s (%d counties)", year, out.name, len(agg))


if __name__ == "__main__":
    main()
