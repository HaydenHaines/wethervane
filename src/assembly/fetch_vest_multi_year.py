"""
Stage 1 data assembly: fetch VEST precinct-level election returns for
2016 and 2018, and crosswalk to census tracts via spatial join.

  2016: presidential returns (G16PRE*)
  2018: gubernatorial returns (G18GOV*) — no presidential race this year

Note: 2020 is handled separately by fetch_vest.py.
      2022 is absent from VEST — series ends at 2021. MIT Election Data Lab
      has 2022 tabular data but no geometry, incompatible with area-weighted
      allocation. Working historical set: 2016 + 2018 + 2020.

Temporal consistency (see ASSUMPTIONS_LOG A011):
  All elections crosswalk to 2022 TIGER tract boundaries + 2022 ACS
  community type assignments. Tract boundaries from TIGER 2022 match 2020
  Census definitions and the 2022 ACS vintage, so all three election years
  map to consistent tract geoids. The ACS vintage mismatch (2022 demographics
  characterizing 2016/2018 tracts) is an accepted MVP approximation documented
  in A011.

Downloads:
  VEST 2016 shapefiles   → data/raw/vest/2016/[state]_2016.zip
  VEST 2018 shapefiles   → data/raw/vest/2018/[state]_2018.zip
  Census TIGER/Line tracts → data/raw/tiger/tl_2022_[fips]_tract.zip (cached)

Outputs:
  data/assembled/vest_tracts_2016.parquet  (pres_total_2016, pres_dem_share_2016, ...)
  data/assembled/vest_tracts_2018.parquet  (gov_total_2018, gov_dem_share_2018, ...)

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

# Harvard Dataverse file IDs
# 2016: doi:10.7910/DVN/NH5S2I — "2016 Precinct-Level Election Results"
VEST_2016_FILE_IDS = {
    "AL": 4751068,
    "FL": 12070343,
    "GA": 11070010,
}

# 2018: doi:10.7910/DVN/UBKYRU — "2018 Precinct-Level Election Results"
VEST_2018_FILE_IDS = {
    "AL": 4751072,
    "FL": 12070358,
    "GA": 11070036,
}

DATAVERSE_DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# 2022 TIGER/Line tract boundaries — consistent with 2022 ACS and 2020 Census tract definitions
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
        # 2016 may use GEOID16
        if "GEOID16" in gdf.columns:
            sample = str(gdf["GEOID16"].dropna().iloc[0])
            n = len(sample)
            if n == 15:
                return "block16"
            if n == 11:
                return "vtd16"
            return f"unknown16_{n}"
        return "none"
    sample = str(gdf["GEOID20"].dropna().iloc[0])
    n = len(sample)
    if n == 15:
        return "block"
    if n == 11:
        return "vtd"
    return f"unknown_{n}"


# ── Vote column detection ─────────────────────────────────────────────────────


def detect_vote_cols(
    gdf: gpd.GeoDataFrame, year: int
) -> tuple[str, list[str], list[str], list[str]]:
    """
    Find the relevant office's vote columns for the given year.

    2016: presidential (G16PRE*) — use dem+rep totals
    2018: gubernatorial (G18GOV*) — no presidential race, governor is the
          top-of-ticket race and the best proxy for partisan lean

    Returns (office_prefix, all_cols, dem_cols, rep_cols).
    VEST naming: G[YY][OFFICE][PARTY][CAND]
    """
    yy = str(year)[2:]

    if year == 2016:
        office = "PRE"
        prefix = "pres"
    elif year == 2018:
        office = "GOV"
        prefix = "gov"
    else:
        raise ValueError(f"Unsupported year: {year}. Use 2016 or 2018.")

    pattern = re.compile(rf"^G{yy}{office}", re.IGNORECASE)
    all_cols = [c for c in gdf.columns if pattern.match(c)]

    if not all_cols:
        raise ValueError(
            f"No {office} columns found for year {year}. "
            f"Columns: {list(gdf.columns)[:20]}"
        )

    d_cols = [c for c in all_cols if len(c) > len(f"G{yy}{office}") and c[len(f"G{yy}{office}")].upper() == "D"]
    r_cols = [c for c in all_cols if len(c) > len(f"G{yy}{office}") and c[len(f"G{yy}{office}")].upper() == "R"]
    log.info("  %s columns (%s): %s", office, prefix, all_cols)
    return prefix, all_cols, d_cols, r_cols


# ── Vote allocation ───────────────────────────────────────────────────────────


def allocate_votes_string(
    vest_gdf: gpd.GeoDataFrame, vote_cols: list[str], geoid_col: str
) -> pd.DataFrame:
    """Block-level GEOID: exact match via first 11 chars."""
    df = vest_gdf[vote_cols].copy().fillna(0)
    df["tract_geoid"] = vest_gdf[geoid_col].str[:11]
    return df.groupby("tract_geoid")[vote_cols].sum().reset_index()


def allocate_votes_area_weighted(
    vest_gdf: gpd.GeoDataFrame,
    tract_gdf: gpd.GeoDataFrame,
    vote_cols: list[str],
) -> pd.DataFrame:
    """
    VTD-level GEOID: area-weighted vote allocation from precincts to census tracts.
    Uses EPSG:5070 (NAD83 Conus Albers equal-area) for proportional area weights.
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

    n_covered = len(result)
    n_total = len(tract_gdf)
    log.info("  Tract coverage: %d / %d (%.0f%%)", n_covered, n_total, 100 * n_covered / n_total)
    return result


# ── Per-state orchestration ───────────────────────────────────────────────────


def get_file_ids(year: int) -> dict[str, int]:
    if year == 2016:
        return VEST_2016_FILE_IDS
    if year == 2018:
        return VEST_2018_FILE_IDS
    raise ValueError(f"No file IDs configured for year {year}")


def process_state(state: str, year: int) -> pd.DataFrame:
    """Download, allocate, and aggregate VEST data for one state → tract-level DataFrame."""
    fips = STATES[state]
    file_ids = get_file_ids(year)

    download_file(
        DATAVERSE_DOWNLOAD_URL.format(file_id=file_ids[state]),
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

    office_prefix, all_cols, d_cols, r_cols = detect_vote_cols(vest_gdf, year)

    if geoid_type in ("block", "block16"):
        geoid_col = "GEOID20" if geoid_type == "block" else "GEOID16"
        tract_votes = allocate_votes_string(vest_gdf, all_cols, geoid_col)
    else:
        log.info("  Using area-weighted allocation (VTD-level GEOID)")
        tract_votes = allocate_votes_area_weighted(vest_gdf, tract_gdf, all_cols)

    d_sum = tract_votes[d_cols].sum(axis=1) if d_cols else 0
    r_sum = tract_votes[r_cols].sum(axis=1) if r_cols else 0
    total = tract_votes[all_cols].sum(axis=1)

    result = pd.DataFrame({"tract_geoid": tract_votes["tract_geoid"]})
    result[f"{office_prefix}_total_{year}"] = total
    result[f"{office_prefix}_dem_{year}"] = d_sum
    result[f"{office_prefix}_rep_{year}"] = r_sum

    denom = result[f"{office_prefix}_dem_{year}"] + result[f"{office_prefix}_rep_{year}"]
    result[f"{office_prefix}_dem_share_{year}"] = (
        result[f"{office_prefix}_dem_{year}"] / denom.replace(0, float("nan"))
    )

    total_votes = int(result[f"{office_prefix}_total_{year}"].sum())
    log.info(
        "  %s %d: %d precincts → %d tracts, total votes: %s",
        state, year, len(vest_gdf), len(result), f"{total_votes:,}",
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in (2016, 2018):
        frames: list[pd.DataFrame] = []
        for state in STATES:
            log.info("=== %s %d ===", state, year)
            df = process_state(state, year)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        office_prefix = "pres" if year == 2016 else "gov"
        out_path = OUTPUT_DIR / f"vest_tracts_{year}.parquet"
        combined.to_parquet(out_path, index=False)

        total = int(combined[f"{office_prefix}_total_{year}"].sum())
        log.info(
            "Saved → %s | %d tracts | %s total %s votes",
            out_path, len(combined), f"{total:,}",
            "presidential" if year == 2016 else "gubernatorial",
        )


if __name__ == "__main__":
    main()
