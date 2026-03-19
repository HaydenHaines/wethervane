"""Build Queen contiguity adjacency matrix for counties (FL+GA+AL).

Dissolves TIGER tract shapefiles to county level, then builds a Queen
contiguity adjacency matrix. Island counties (zero neighbors) are connected
to their nearest county by centroid distance.

Outputs:
    data/communities/county_adjacency.npz        — scipy sparse CSR (N x N)
    data/communities/county_adjacency.fips.txt   — ordered FIPS list

Usage:
    uv run python src/discovery/build_county_adjacency.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIGER_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"
COMMUNITIES_DIR = PROJECT_ROOT / "data" / "communities"

STATE_FIPS = ["01", "12", "13"]  # AL, FL, GA


def load_tiger_tracts() -> gpd.GeoDataFrame:
    """Load TIGER 2022 tract shapefiles for AL, FL, GA."""
    gdfs = []
    for fips in STATE_FIPS:
        extract_dir = TIGER_DIR / f"tl_2022_{fips}_tract"
        shp_files = list(extract_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"TIGER shapefile not found at {extract_dir}")
        gdf = gpd.read_file(shp_files[0])
        gdf = gdf.rename(columns={"GEOID": "tract_geoid"})
        gdfs.append(gdf[["tract_geoid", "geometry"]])
        log.info("  Loaded %d tracts for FIPS %s", len(gdf), fips)
    combined = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=gdfs[0].crs
    )
    log.info("Total tracts loaded: %d", len(combined))
    return combined


def dissolve_to_county(tract_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve tract geometries to county polygons."""
    tract_gdf = tract_gdf.copy()
    tract_gdf["county_fips"] = tract_gdf["tract_geoid"].str[:5]
    county_gdf = tract_gdf.dissolve(by="county_fips", as_index=False)[["county_fips", "geometry"]]
    # Reproject to a projected CRS for accurate adjacency detection
    county_gdf = county_gdf.to_crs("EPSG:3857")
    log.info("Dissolved to %d counties", len(county_gdf))
    return county_gdf


def build_queen_adjacency(gdf: gpd.GeoDataFrame) -> tuple[csr_matrix, list[str]]:
    """Build Queen contiguity sparse adjacency matrix from county geometries."""
    from libpysal.weights import Queen

    n = len(gdf)
    log.info("Building Queen adjacency for %d counties...", n)

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)

    sparse = w.sparse
    sparse = (sparse > 0).astype(np.int8)
    sparse = (sparse + sparse.T)
    sparse = (sparse > 0).astype(np.int8)
    W = csr_matrix(sparse)

    fips_list = list(gdf["county_fips"].astype(str))

    degrees = np.array(W.sum(axis=1)).flatten()
    island_count = int((degrees == 0).sum())
    log.info(
        "Queen adjacency built: mean neighbors=%.2f, min=%d, max=%d, islands=%d",
        degrees.mean(), int(degrees.min()), int(degrees.max()), island_count,
    )

    return W, fips_list


def handle_islands(W: csr_matrix, gdf: gpd.GeoDataFrame) -> csr_matrix:
    """Connect island counties (zero neighbors) to their nearest by centroid."""
    degrees = np.array(W.sum(axis=1)).flatten()
    island_indices = np.where(degrees == 0)[0]

    if len(island_indices) == 0:
        log.info("No island counties — nothing to fix.")
        return W

    log.info("Connecting %d island county(ies) to nearest neighbor...", len(island_indices))

    W_lil = W.tolil()
    centroids = np.column_stack(
        [gdf.geometry.centroid.x, gdf.geometry.centroid.y]
    )

    for idx in island_indices:
        centroid = centroids[idx]
        dists = np.linalg.norm(centroids - centroid, axis=1)
        dists[idx] = np.inf
        nearest = int(np.argmin(dists))
        W_lil[idx, nearest] = 1
        W_lil[nearest, idx] = 1
        log.info("  Island county idx=%d connected to idx=%d", idx, nearest)

    W_fixed = csr_matrix(W_lil)
    W_fixed = (W_fixed + W_fixed.T)
    W_fixed = (W_fixed > 0).astype(np.int8)
    W_fixed = csr_matrix(W_fixed)

    remaining = int((np.array(W_fixed.sum(axis=1)).flatten() == 0).sum())
    log.info("Island handling complete. Remaining islands: %d", remaining)
    return W_fixed


def main() -> None:
    log.info("Loading TIGER tract geometries...")
    tracts = load_tiger_tracts()

    log.info("Dissolving tracts to county boundaries...")
    county_gdf = dissolve_to_county(tracts)

    log.info("Building Queen contiguity adjacency...")
    W, fips_list = build_queen_adjacency(county_gdf)

    log.info("Handling island counties...")
    W = handle_islands(W, county_gdf)

    COMMUNITIES_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = COMMUNITIES_DIR / "county_adjacency.npz"
    save_npz(str(npz_path), W)
    log.info("Adjacency matrix saved to %s", npz_path)

    fips_path = COMMUNITIES_DIR / "county_adjacency.fips.txt"
    fips_path.write_text("\n".join(fips_list))
    log.info("FIPS ordering saved to %s (%d counties)", fips_path, len(fips_list))


if __name__ == "__main__":
    main()
