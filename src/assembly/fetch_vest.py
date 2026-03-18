"""
Stage 1 data assembly: fetch VEST precinct-level election returns (2020)
and crosswalk to census tracts via spatial join.

Key discovery from AL 2020 inspection:
  VEST 2020 uses VTD-level GEOIDs (11-digit: state+county+VTD code), not
  block-level GEOIDs (15-digit). The GEOID20[:11] string-slice from the
  reference file produces a VTD FIPS, not a tract FIPS — they are the same
  length but different geographies. A spatial join is required.

  Strategy: compute each precinct's centroid, find which Census tract polygon
  contains it, assign all votes from that precinct to that tract.

Downloads:
  VEST 2020 shapefiles   → data/raw/vest/2020/[state]_2020.zip
  Census TIGER/Line tracts → data/raw/tiger/tl_2022_[fips]_tract.zip

Output:
  data/assembled/vest_tracts_2020.parquet

Reference: docs/references/data-sources/vest-election-crosswalk.md
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
RAW_VEST_DIR = PROJECT_ROOT / "data" / "raw" / "vest"
RAW_TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"

STATES = {"AL": "01", "FL": "12", "GA": "13"}

# Harvard Dataverse file IDs for doi:10.7910/DVN/K7760H
# "2020 Precinct-Level Election Results" — Voting and Election Science Team
VEST_2020_FILE_IDS = {
    "AL": 4751074,
    "FL": 12070362,
    "GA": 11070054,
}

DATAVERSE_DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# 2022 TIGER/Line tract boundaries — matches the ACS 2022 5-year vintage
TIGER_TRACT_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_{fips}_tract.zip"
)


# ── Download ──────────────────────────────────────────────────────────────────


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with a progress bar. No-ops if the file already exists."""
    if dest.exists():
        log.info("  Cached: %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", desc)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))


# ── Load ──────────────────────────────────────────────────────────────────────


def load_vest_gdf(state: str, year: int) -> gpd.GeoDataFrame:
    """Unzip and read VEST shapefile; return GeoDataFrame with geometry + vote cols."""
    zip_path = RAW_VEST_DIR / str(year) / f"{state.lower()}_{year}.zip"
    extract_dir = zip_path.parent / f"{state.lower()}_{year}"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file in {extract_dir}")

    gdf = gpd.read_file(shp_files[0])
    log.info("  %s %d VEST: %d precincts, CRS=%s", state, year, len(gdf), gdf.crs)
    return gdf


def load_tract_gdf(state: str) -> gpd.GeoDataFrame:
    """Unzip and read Census TIGER tract shapefile; return GeoDataFrame with GEOID + geometry."""
    fips = STATES[state]
    zip_path = RAW_TIGER_DIR / f"tl_2022_{fips}_tract.zip"
    extract_dir = zip_path.parent / f"tl_2022_{fips}_tract"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file in {extract_dir}")

    gdf = gpd.read_file(shp_files[0])[["GEOID", "geometry"]].rename(
        columns={"GEOID": "tract_geoid"}
    )
    log.info("  %s TIGER tracts: %d loaded", state, len(gdf))
    return gdf


# ── GEOID detection ───────────────────────────────────────────────────────────


def detect_geoid_type(gdf: gpd.GeoDataFrame) -> str:
    """Return 'block' (15-digit), 'vtd' (11-digit), or 'unknown_N'."""
    if "GEOID20" not in gdf.columns:
        return "none"
    sample = str(gdf["GEOID20"].dropna().iloc[0])
    n = len(sample)
    if n == 15:
        return "block"
    if n == 11:
        return "vtd"
    return f"unknown_{n}"


# ── Vote column detection ─────────────────────────────────────────────────────


def detect_presidential_cols(gdf: gpd.GeoDataFrame, year: int) -> tuple[list[str], list[str], list[str]]:
    """
    Find presidential vote columns and split by party.
    Returns (all_cols, dem_cols, rep_cols).

    VEST naming: G[YY][OFFICE][PARTY][CAND]
      G20PREDBID = General 2020 PRE(sidential) D(emocrat) BID(en)
      G20PRERTRU = General 2020 PRE(sidential) R(epublican) TRU(mp)
    """
    yy = str(year)[2:]
    pattern = re.compile(rf"^G{yy}PRE", re.IGNORECASE)
    all_cols = [c for c in gdf.columns if pattern.match(c)]
    if not all_cols:
        raise ValueError(
            f"No presidential columns found for year {year}. "
            f"Columns: {list(gdf.columns)}"
        )
    d_cols = [c for c in all_cols if len(c) > 6 and c[6].upper() == "D"]
    r_cols = [c for c in all_cols if len(c) > 6 and c[6].upper() == "R"]
    log.info("  Presidential columns: %s", all_cols)
    return all_cols, d_cols, r_cols


# ── Vote allocation ───────────────────────────────────────────────────────────


def allocate_votes_string(
    vest_gdf: gpd.GeoDataFrame, vote_cols: list[str], year: int
) -> pd.DataFrame:
    """
    Block-level GEOID: exact 1-to-1 match via first 11 chars of GEOID20.
    Each precinct IS a census block, so it's fully contained within one tract.
    """
    df = vest_gdf[vote_cols].copy().fillna(0)
    df["tract_geoid"] = vest_gdf["GEOID20"].str[:11]
    return df.groupby("tract_geoid")[vote_cols].sum().reset_index()


def allocate_votes_area_weighted(
    vest_gdf: gpd.GeoDataFrame,
    tract_gdf: gpd.GeoDataFrame,
    vote_cols: list[str],
    year: int,
) -> pd.DataFrame:
    """
    VTD-level GEOID: area-weighted vote allocation from precincts to census tracts.

    For each precinct polygon, finds every tract it intersects. Allocates the
    precinct's votes to each tract proportionally to the intersection area.

    This correctly handles the common case of large rural precincts that span
    multiple census tracts — votes are split rather than assigned to the tract
    containing just the centroid.

    Uses EPSG:5070 (NAD83 Conus Albers equal-area) for area computation so
    that weights are proportional to real-world area, not angular degrees.
    """
    aea = "EPSG:5070"
    vest_proj = vest_gdf[vote_cols + ["geometry"]].fillna(0).to_crs(aea).reset_index()
    vest_proj = vest_proj.rename(columns={"index": "_prec_idx"})
    vest_proj["_precinct_area"] = vest_proj.geometry.area
    tract_proj = tract_gdf.to_crs(aea)

    log.info("  Computing precinct-tract intersections (area-weighted)...")
    intersections = gpd.overlay(
        vest_proj[["_prec_idx", "_precinct_area"] + vote_cols + ["geometry"]],
        tract_proj[["tract_geoid", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    intersections["_weight"] = intersections.geometry.area / intersections["_precinct_area"]

    for col in vote_cols:
        intersections[col] = intersections[col] * intersections["_weight"]

    result = intersections.groupby("tract_geoid")[vote_cols].sum().reset_index()

    # Report coverage
    n_covered = len(result)
    n_total = len(tract_gdf)
    log.info("  Tract coverage: %d / %d (%.0f%%)", n_covered, n_total, 100 * n_covered / n_total)
    return result


# ── Per-state orchestration ───────────────────────────────────────────────────


def process_state(state: str, year: int) -> pd.DataFrame:
    """Download, allocate, and aggregate VEST data for one state → tract-level DataFrame."""
    fips = STATES[state]

    download_file(
        DATAVERSE_DOWNLOAD_URL.format(file_id=VEST_2020_FILE_IDS[state]),
        RAW_VEST_DIR / str(year) / f"{state.lower()}_{year}.zip",
        f"VEST {state} {year}",
    )
    download_file(
        TIGER_TRACT_URL.format(fips=fips),
        RAW_TIGER_DIR / f"tl_2022_{fips}_tract.zip",
        f"TIGER tracts {state}",
    )

    vest_gdf = load_vest_gdf(state, year)
    tract_gdf = load_tract_gdf(state)

    geoid_type = detect_geoid_type(vest_gdf)
    log.info("  GEOID type: %s", geoid_type)

    all_cols, d_cols, r_cols = detect_presidential_cols(vest_gdf, year)

    if geoid_type == "block":
        tract_votes = allocate_votes_string(vest_gdf, all_cols, year)
    else:
        log.info("  Using area-weighted allocation (VTD-level GEOID)")
        tract_votes = allocate_votes_area_weighted(vest_gdf, tract_gdf, all_cols, year)

    # Rename raw vote cols to semantic names
    d_sum = tract_votes[d_cols].sum(axis=1) if d_cols else 0
    r_sum = tract_votes[r_cols].sum(axis=1) if r_cols else 0
    total = tract_votes[all_cols].sum(axis=1)

    result = pd.DataFrame({"tract_geoid": tract_votes["tract_geoid"]})
    result[f"pres_total_{year}"] = total
    result[f"pres_dem_{year}"] = d_sum
    result[f"pres_rep_{year}"] = r_sum

    denom = result[f"pres_dem_{year}"] + result[f"pres_rep_{year}"]
    result[f"pres_dem_share_{year}"] = result[f"pres_dem_{year}"] / denom.replace(0, float("nan"))

    total_votes = int(result[f"pres_total_{year}"].sum())
    log.info(
        "  %s %d: %d precincts → %d tracts, total votes: %s",
        state, year, len(vest_gdf), len(result), f"{total_votes:,}",
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    year = 2020

    frames: list[pd.DataFrame] = []
    for state in STATES:
        log.info("=== %s %d ===", state, year)
        df = process_state(state, year)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"vest_tracts_{year}.parquet"
    combined.to_parquet(out_path, index=False)

    total = int(combined[f"pres_total_{year}"].sum())
    log.info(
        "Saved → %s | %d tracts | %s total presidential votes",
        out_path, len(combined), f"{total:,}",
    )


if __name__ == "__main__":
    main()
