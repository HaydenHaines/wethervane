"""Fetch MEDSL county-level U.S. Senate returns for FL, GA, AL.

Source: MIT Election Data + Science Lab, Harvard Dataverse
  doi:10.7910/DVN/PEJ5QU  (U.S. Senate Returns, county-level)

Downloads a single unified tab-delimited file covering U.S. Senate elections
at the county level. Filters to FL, GA, AL for the target Senate years.
Computes total-vote dem share (dem / all_candidates) per county per year.

Senate seats per state in this model:
  FL  Class I  (2000, 2006, 2012, 2018, 2024)  Nelson/Scott seat
  FL  Class III (2004, 2010, 2016, 2022)        Martinez/Rubio seat
  GA  Class II  (2002, 2008, 2014, 2020)        Cleland/Warnock seat
  GA  Class III (2004, 2010, 2016, 2022)        Miller/Ossoff seat
  AL  Class II  (2002, 2008, 2014, 2020)        Sessions/Jones seat
  AL  Class III (2004, 2010, 2016, 2022)        Shelby seat

NOTE: Multiple Senate races may exist in the same state-year (e.g. GA 2020
special + regular). When that happens this module keeps all rows and lets the
caller aggregate; downstream uncontested_policy handles structural zeros.

Output (one parquet per election year, data/assembled/):
  medsl_county_senate_{year}.parquet
  Columns: county_fips, state_abbr, senate_dem_{year}, senate_rep_{year},
           senate_total_{year}, senate_dem_share_{year}

Cache: data/raw/medsl/county_senate.tab
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.core import config as _cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "medsl"
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
CACHE_PATH = RAW_DIR / "county_senate.tab"

DATAVERSE_DOI = "doi:10.7910/DVN/PEJ5QU"
DATAVERSE_API = "https://dataverse.harvard.edu/api"

STATES: dict[str, str] = _cfg.STATE_ABBR   # fips_prefix → abbr
SENATE_YEARS: list[int] = _cfg.SENATE_YEARS


def _dataverse_download(doi: str, cache_path: Path) -> Path:
    """Download primary data file from a Harvard Dataverse DOI."""
    if cache_path.exists():
        log.info("Using cached file: %s", cache_path)
        return cache_path

    list_url = f"{DATAVERSE_API}/datasets/:persistentId/versions/:latest/files"
    resp = requests.get(list_url, params={"persistentId": doi}, timeout=30)
    resp.raise_for_status()
    files = resp.json()["data"]

    # Find the main tabular data file; skip README / codebook / sources
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

    # Pick the largest file (most likely the main data file)
    main_file = max(data_files, key=lambda f: f["dataFile"].get("filesize", 0))
    file_id = main_file["dataFile"]["id"]
    filename = main_file["dataFile"]["filename"]
    log.info("Downloading %s (id=%s)...", filename, file_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dl_url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    with requests.get(dl_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)

    log.info("Saved → %s", cache_path)
    return cache_path


def filter_senate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only D/R U.S. Senate rows for FL, GA, AL.

    Expects state_po column to contain state abbreviations (FL, GA, AL).
    The MEDSL office field is 'US SENATE' or similar.
    """
    mask = (
        (df["office"].str.upper().str.contains("SENATE"))
        & (~df["office"].str.upper().str.contains("STATE"))
        & (df["party_simplified"].isin({"DEMOCRAT", "REPUBLICAN"}))
        & (df["state_po"].isin(set(STATES.values())))
    )
    return df[mask].copy()


def aggregate_county_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate filtered rows to one row per county for a given year.

    When a state has two Senate races in the same year (e.g. GA 2020 regular +
    special), the candidatevotes and totalvotes are summed across all races for
    that county. This produces a blended dem_share that represents the average
    D performance weighted by turnout.

    Returns county_fips, state_abbr, senate_dem_{year}, senate_rep_{year},
    senate_total_{year}, senate_dem_share_{year}.

    Applies uncontested_policy=drop: counties where dem or rep votes are zero
    are dropped (NaN dem_share would distort log-odds shifts).
    """
    yr = df[df["year"] == year].copy()
    if yr.empty:
        return pd.DataFrame(columns=[
            "county_fips", "state_abbr",
            f"senate_dem_{year}", f"senate_rep_{year}",
            f"senate_total_{year}", f"senate_dem_share_{year}",
        ])

    # Normalise county_fips: handle float representation (e.g. 12086.0 → "12086")
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
        .rename(f"senate_dem_{year}")
    )
    rep = (
        yr[yr["party_simplified"] == "REPUBLICAN"]
        .groupby("county_fips")["candidatevotes"]
        .sum()
        .rename(f"senate_rep_{year}")
    )
    # totalvotes is reported per row in MEDSL; for counties with multiple races
    # (GA/AL specials) we sum totalvotes across races so the denominator matches
    # the summed candidatevotes.
    total_all = (
        yr[yr["party_simplified"] == "DEMOCRAT"]
        .groupby("county_fips")["totalvotes"]
        .sum()
        .rename(f"senate_total_{year}")
    )

    result = pd.concat([dem, rep, total_all], axis=1).reset_index()
    result[f"senate_dem_share_{year}"] = (
        result[f"senate_dem_{year}"] / result[f"senate_total_{year}"]
    )

    # Uncontested policy: drop counties where either party has zero votes
    dem_col = f"senate_dem_{year}"
    rep_col = f"senate_rep_{year}"
    contested = (result[dem_col] > 0) & (result[rep_col] > 0)
    n_dropped = (~contested).sum()
    if n_dropped:
        log.warning(
            "Year %d: dropping %d uncontested counties (uncontested_policy=drop)",
            year, n_dropped,
        )
    result = result[contested].copy()

    result["state_abbr"] = result["county_fips"].str[:2].map(STATES)
    return result[["county_fips", "state_abbr",
                   f"senate_dem_{year}", f"senate_rep_{year}",
                   f"senate_total_{year}", f"senate_dem_share_{year}"]]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MEDSL column names and voting-mode rows to canonical schema.

    Mirrors the same logic used in fetch_medsl_county_presidential:
      - Renames 'party' → 'party_simplified' if needed.
      - For counties with a TOTAL mode row, keep only those.
      - For counties without a TOTAL row, sum across all non-TOTAL mode rows.

    Result has one row per (year, state_po, county_fips, party_simplified).
    """
    if "party" in df.columns and "party_simplified" not in df.columns:
        df = df.rename(columns={"party": "party_simplified"})

    if "mode" not in df.columns:
        return df

    has_total = (
        df[df["mode"].str.upper().str.strip() == "TOTAL"]
        .groupby(["year", "county_fips"])
        .size()
        .reset_index(drop=False)[["year", "county_fips"]]
        .assign(_has_total=True)
    )
    df = df.merge(has_total, on=["year", "county_fips"], how="left")
    df["_has_total"] = df["_has_total"].fillna(False)

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
    sep = "\t" if str(csv_path).endswith(".tab") else ","
    log.info("Loading data file (sep=%r)...", sep)
    df = pd.read_csv(csv_path, sep=sep, low_memory=False)
    log.info("Raw rows: %d", len(df))

    df = _normalize_columns(df)
    df_filtered = filter_senate_rows(df)
    log.info("After filter (FL+GA+AL D/R Senate only): %d rows", len(df_filtered))

    ASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)
    years_in_data = sorted(df_filtered["year"].unique())
    log.info("Years in data: %s", years_in_data)

    for year in SENATE_YEARS:
        if year not in years_in_data:
            log.warning("Year %d not in dataset — skipping", year)
            continue
        agg = aggregate_county_year(df_filtered, year)
        if agg.empty:
            log.warning("Year %d produced no rows after aggregation — skipping", year)
            continue
        out = ASSEMBLED_DIR / f"medsl_county_senate_{year}.parquet"
        agg.to_parquet(out, index=False)
        log.info("  %d → %s (%d counties)", year, out.name, len(agg))


if __name__ == "__main__":
    main()
