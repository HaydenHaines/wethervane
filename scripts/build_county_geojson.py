# scripts/build_county_geojson.py
"""Download Census TIGER/Line county shapefiles and generate national US GeoJSON.

Output: web/public/counties-us.geojson
Properties per feature: county_fips, county_name (if crosswalk present)

Covers all 50 states + DC (3,143+ counties/county-equivalents). Simplification
tolerance of 0.005 degrees keeps the output ~8–12 MB, suitable for web delivery.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
OUT_PATH = PROJECT_ROOT / "web" / "public" / "counties-us.geojson"
CROSSWALK_PATH = PROJECT_ROOT / "data" / "raw" / "fips_county_crosswalk.csv"


def main() -> None:
    print(f"Downloading TIGER/Line county shapefile...")
    gdf = gpd.read_file(TIGER_URL)
    print(f"Downloaded {len(gdf)} counties (all US)")

    # Build county_fips
    gdf["county_fips"] = gdf["STATEFP"] + gdf["COUNTYFP"]

    # Simplify geometry (tolerance 0.005 degrees ≈ ~500m; keeps output ~8–12 MB for national coverage)
    gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)

    # Join county names from crosswalk if available
    keep_cols = ["county_fips", "geometry"]
    if CROSSWALK_PATH.exists():
        xwalk = pd.read_csv(CROSSWALK_PATH, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        gdf = gdf.merge(xwalk, on="county_fips", how="left")
        missing = gdf["county_name"].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} counties missing county_name after crosswalk merge (territories/outlying areas)")
        keep_cols = ["county_fips", "county_name", "geometry"]

    gdf = gdf[keep_cols].set_geometry("geometry")

    # Reproject to WGS84 (EPSG:4326) — required for Deck.gl
    gdf = gdf.to_crs(epsg=4326)

    # Write GeoJSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"Saved {len(gdf)} counties to {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
