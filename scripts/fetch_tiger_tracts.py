"""Download Census TIGER/Line 2020 tract shapefiles for FL, GA, and AL.

Downloads zipped shapefiles from the Census Bureau, extracts them, and verifies
tract counts by reading with geopandas.

Output:
    data/raw/tiger/tl_2020_{fips}_tract.zip  (zipped shapefiles)
    data/raw/tiger/tl_2020_{fips}_tract/     (extracted directories)

States:
    AL (01), FL (12), GA (13)
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "tiger"

STATES = {"AL": "01", "FL": "12", "GA": "13"}

BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT"


# ── Download ──────────────────────────────────────────────────────────────────


def download_state_tracts(state_abbr: str, state_fips: str) -> Path:
    """Download and extract TIGER/Line 2020 tract shapefile for one state.

    Returns the path to the extracted directory.
    """
    filename = f"tl_2020_{state_fips}_tract.zip"
    url = f"{BASE_URL}/{filename}"
    zip_path = OUTPUT_DIR / filename
    extract_dir = OUTPUT_DIR / f"tl_2020_{state_fips}_tract"

    if extract_dir.exists() and any(extract_dir.glob("*.shp")):
        log.info("  %s (FIPS %s): already downloaded, skipping", state_abbr, state_fips)
        return extract_dir

    log.info("  %s (FIPS %s): downloading %s", state_abbr, state_fips, url)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    log.info("  %s: extracting to %s", state_abbr, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


# ── Verify ────────────────────────────────────────────────────────────────────


def verify_tract_counts(state_abbr: str, extract_dir: Path) -> int:
    """Read the extracted shapefile and return the tract count."""
    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {extract_dir}")

    gdf = gpd.read_file(shp_files[0])
    n_tracts = len(gdf)
    log.info("  %s: %d tracts", state_abbr, n_tracts)
    return n_tracts


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("Downloading TIGER/Line 2020 tract shapefiles for %d states", len(STATES))

    total = 0
    for state_abbr, state_fips in sorted(STATES.items()):
        extract_dir = download_state_tracts(state_abbr, state_fips)
        n = verify_tract_counts(state_abbr, extract_dir)
        total += n

    log.info("Total: %d tracts across %d states", total, len(STATES))


if __name__ == "__main__":
    main()
