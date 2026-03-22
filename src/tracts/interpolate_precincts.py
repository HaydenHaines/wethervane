"""Allocate precinct-level votes to census tracts using geometric area overlap.

Algorithm (areal interpolation):
1. Reproject both layers to EPSG:5070 (NAD83 Conus Albers) for accurate area.
2. Fix invalid geometries with buffer(0).
3. Filter out zero-area precincts (point geometries).
4. Compute geometric intersection via gpd.overlay.
5. For each intersection piece: area_fraction = piece_area / precinct_area.
6. Allocate votes: votes_allocated = votes * area_fraction.
7. Aggregate to tract: groupby GEOID, sum allocated votes.
8. Compute dem_share = votes_dem / votes_total.

CLI: python -m src.tracts.interpolate_precincts
Output: data/tracts/tract_votes_{state}_{year}_{race}.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Equal-area projection for accurate area calculations
_TARGET_CRS = "EPSG:5070"

# State abbreviation -> FIPS prefix (for filtering NYT data by GEOID)
_STATE_FIPS = {
    "AL": "01",
    "FL": "12",
    "GA": "13",
}

# Which races are available per source/year
_VEST_CONFIGS: list[tuple[str, int, str]] = [
    ("AL", 2016, "president"),
    ("FL", 2016, "president"),
    ("GA", 2016, "president"),
    ("AL", 2018, "governor"),
    ("FL", 2018, "governor"),
    ("GA", 2018, "governor"),
    ("AL", 2020, "president"),
    ("FL", 2020, "president"),
    ("GA", 2020, "president"),
]


def interpolate_precincts_to_tracts(
    precinct_gdf: gpd.GeoDataFrame,
    tract_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Allocate precinct votes to tracts proportional to area overlap.

    Parameters
    ----------
    precinct_gdf : GeoDataFrame
        Must have columns: geometry, votes_dem, votes_rep, votes_total.
    tract_gdf : GeoDataFrame
        Must have columns: geometry, GEOID (tract FIPS code).

    Returns
    -------
    DataFrame with columns: GEOID, votes_dem, votes_rep, votes_total, dem_share.
    One row per tract GEOID (including tracts with zero overlap).
    """
    # -- Reproject to equal-area CRS ----------------------------------------
    precincts = precinct_gdf.to_crs(_TARGET_CRS).copy()
    tracts = tract_gdf.to_crs(_TARGET_CRS).copy()

    # -- Fix invalid geometries ---------------------------------------------
    precincts["geometry"] = precincts.geometry.buffer(0)
    tracts["geometry"] = tracts.geometry.buffer(0)

    # -- Filter out zero-area precincts (points, degenerate polygons) --------
    precincts["_precinct_area"] = precincts.geometry.area
    precincts = precincts[precincts["_precinct_area"] > 0].copy()

    if precincts.empty:
        # All precincts were zero-area; return tracts with zero votes
        return _empty_result(tracts)

    # -- Assign a precinct index for tracking after overlay ------------------
    precincts = precincts.reset_index(drop=True)
    precincts["_precinct_idx"] = precincts.index

    # -- Geometric intersection (overlay) ------------------------------------
    # Keep only necessary columns for the overlay to reduce memory
    p_cols = ["_precinct_idx", "_precinct_area", "votes_dem", "votes_rep", "votes_total", "geometry"]
    t_cols = ["GEOID", "geometry"]

    pieces = gpd.overlay(
        precincts[p_cols],
        tracts[t_cols],
        how="intersection",
        keep_geom_type=False,
    )

    if pieces.empty:
        return _empty_result(tracts)

    # -- Compute area fractions and allocate votes ---------------------------
    pieces["_piece_area"] = pieces.geometry.area
    pieces["_area_frac"] = pieces["_piece_area"] / pieces["_precinct_area"]

    for col in ("votes_dem", "votes_rep", "votes_total"):
        pieces[col] = pieces[col] * pieces["_area_frac"]

    # -- Aggregate to tract level --------------------------------------------
    agg = (
        pieces.groupby("GEOID", as_index=False)[["votes_dem", "votes_rep", "votes_total"]]
        .sum()
    )

    # -- Ensure all tracts appear (even with zero votes) ---------------------
    all_geoids = tracts[["GEOID"]].drop_duplicates()
    result = all_geoids.merge(agg, on="GEOID", how="left")
    for col in ("votes_dem", "votes_rep", "votes_total"):
        result[col] = result[col].fillna(0.0)

    # -- Compute dem_share ---------------------------------------------------
    result["dem_share"] = np.where(
        result["votes_total"] > 0,
        result["votes_dem"] / result["votes_total"],
        np.nan,
    )

    return result


def _empty_result(tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return a DataFrame with all tract GEOIDs and zero votes."""
    return pd.DataFrame({
        "GEOID": tracts["GEOID"].unique(),
        "votes_dem": 0.0,
        "votes_rep": 0.0,
        "votes_total": 0.0,
        "dem_share": np.nan,
    })


# ── High-level loader ─────────────────────────────────────────────────────────


def _load_tiger_tracts(tiger_dir: Path, state: str) -> gpd.GeoDataFrame:
    """Load TIGER tract shapefile for a state."""
    fips = _STATE_FIPS[state.upper()]
    # Try 2020 tracts first (preferred), fall back to 2022
    for vintage in ("2020", "2022"):
        shp_dir = tiger_dir / f"tl_{vintage}_{fips}_tract"
        if shp_dir.exists():
            gdf = gpd.read_file(shp_dir)
            gdf = gdf.rename(columns={"GEOID": "GEOID"})  # already named GEOID in TIGER
            return gdf[["GEOID", "geometry"]]
    raise FileNotFoundError(
        f"No TIGER tract shapefile found for {state} (FIPS {fips}) in {tiger_dir}"
    )


def _load_vest_precincts(vest_dir: Path, state: str, year: int, race: str) -> gpd.GeoDataFrame:
    """Load VEST precinct data and extract standardized votes."""
    from src.tracts.vest_columns import extract_vest_votes

    fname = f"{state.lower()}_{year}.zip"
    fpath = vest_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"VEST file not found: {fpath}")

    gdf = gpd.read_file(fpath)
    return extract_vest_votes(gdf, state, year, race)


def _load_nyt_precincts(nyt_dir: Path, state: str) -> gpd.GeoDataFrame:
    """Load NYTimes 2024 precinct data for a single state.

    Joins the TopoJSON (geometry) with the CSV (votes).
    The TopoJSON may already contain vote columns; if so, use those directly.
    """
    topo_path = nyt_dir / "precincts_2024_national.topojson.gz"
    if not topo_path.exists():
        raise FileNotFoundError(f"NYT TopoJSON not found: {topo_path}")

    fips = _STATE_FIPS[state.upper()]
    gdf = gpd.read_file(topo_path)

    # Identify GEOID-like column
    geoid_col = None
    for col in ("GEOID", "geoid", "GEOID20", "id"):
        if col in gdf.columns:
            geoid_col = col
            break

    if geoid_col is None:
        raise ValueError(f"No GEOID column found in NYT TopoJSON. Columns: {list(gdf.columns)}")

    gdf[geoid_col] = gdf[geoid_col].astype(str)

    # Filter to target state by FIPS prefix
    mask = gdf[geoid_col].str.startswith(fips)
    gdf = gdf[mask].copy()

    if gdf.empty:
        raise ValueError(f"No precincts found for state {state} (FIPS {fips}) in NYT data")

    # Check if vote columns already exist in TopoJSON
    vote_cols_present = all(c in gdf.columns for c in ("votes_dem", "votes_rep", "votes_total"))

    if vote_cols_present:
        return gdf[["geometry", "votes_dem", "votes_rep", "votes_total"]].copy()

    # Otherwise, join with CSV
    csv_path = nyt_dir / "precincts_2024_national.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(f"NYT CSV not found: {csv_path}")

    csv_df = pd.read_csv(csv_path, dtype={"GEOID": str})
    csv_df = csv_df[csv_df["GEOID"].str.startswith(fips)]

    # Merge geometry + votes
    merged = gdf[[geoid_col, "geometry"]].merge(
        csv_df[["GEOID", "votes_dem", "votes_rep", "votes_total"]],
        left_on=geoid_col,
        right_on="GEOID",
        how="inner",
    )

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf.crs)


def interpolate_election(
    state: str,
    year: int,
    race: str,
    vest_dir: Path,
    nyt_dir: Path,
    tiger_dir: Path,
) -> pd.DataFrame:
    """Load precinct + tract data for one state/year/race, run interpolation.

    Parameters
    ----------
    state : str
        State abbreviation ("AL", "FL", "GA").
    year : int
        Election year (2016, 2018, 2020, 2024).
    race : str
        Race name ("president", "governor", etc.).
    vest_dir : Path
        Directory containing VEST zip files.
    nyt_dir : Path
        Directory containing NYT precinct files.
    tiger_dir : Path
        Directory containing TIGER tract shapefiles.

    Returns
    -------
    DataFrame with tract-level interpolated votes.
    """
    log.info("Interpolating %s %d %s", state, year, race)

    # Load tracts
    tract_gdf = _load_tiger_tracts(tiger_dir, state)

    # Load precincts
    if year == 2024:
        precinct_gdf = _load_nyt_precincts(nyt_dir, state)
    else:
        precinct_gdf = _load_vest_precincts(vest_dir, state, year, race)

    # Run interpolation
    result = interpolate_precincts_to_tracts(precinct_gdf, tract_gdf)
    result["state"] = state.upper()
    result["year"] = year
    result["race"] = race

    log.info(
        "%s %d %s: %d tracts, D=%.0f R=%.0f total=%.0f",
        state, year, race,
        len(result),
        result["votes_dem"].sum(),
        result["votes_rep"].sum(),
        result["votes_total"].sum(),
    )

    return result


def main() -> None:
    """Process all available state/year/race combinations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    project_root = Path(__file__).parents[2]
    vest_dir = project_root / "data" / "raw" / "vest"
    nyt_dir = project_root / "data" / "raw" / "nyt_precinct"
    tiger_dir = project_root / "data" / "raw" / "tiger"
    out_dir = project_root / "data" / "tracts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # VEST elections (2016, 2018, 2020)
    configs: list[tuple[str, int, str]] = list(_VEST_CONFIGS)

    # NYT 2024 presidential
    for st in ("AL", "FL", "GA"):
        configs.append((st, 2024, "president"))

    for state, year, race in configs:
        try:
            result = interpolate_election(
                state=state,
                year=year,
                race=race,
                vest_dir=vest_dir,
                nyt_dir=nyt_dir,
                tiger_dir=tiger_dir,
            )
            out_path = out_dir / f"tract_votes_{state.lower()}_{year}_{race}.parquet"
            result.to_parquet(out_path, index=False)
            log.info("Wrote %s (%d rows)", out_path, len(result))

        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping %s %d %s: %s", state, year, race, exc)
        except Exception:
            log.exception("Failed %s %d %s", state, year, race)


if __name__ == "__main__":
    main()
