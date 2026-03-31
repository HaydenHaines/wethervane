"""SCI data loading and preprocessing for electoral type validation.

Handles all I/O for the Facebook Social Connectedness Index validation:
loading the raw SCI CSV, loading type assignments, loading county centroids,
and fetching centroids from the Census Gazetteer if not on disk.

Memory strategy: the full SCI file has 10.3M rows (symmetric pairs). We load
only the upper triangle (~5.1M pairs) and filter to counties present in our
type assignments (~3,154 counties).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]

# Default paths (overridable for testing)
SCI_PATH = PROJECT_ROOT / "data" / "raw" / "facebook_sci" / "us_counties.csv"
TYPE_ASSIGNMENTS_PATH = (
    PROJECT_ROOT / "data" / "communities" / "type_assignments.parquet"
)
COUNTY_CENTROIDS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "county_centroids_2020.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "validation"


def load_sci_upper_triangle(
    path: Path,
    valid_fips: set[str] | None = None,
) -> pd.DataFrame:
    """Load SCI data, keeping only upper triangle (user < friend) for unique pairs.

    Parameters
    ----------
    path:
        Path to the SCI CSV file.
    valid_fips:
        If provided, filter to pairs where both counties are in this set.

    Returns
    -------
    DataFrame with columns: user_fips, friend_fips, scaled_sci.
    """
    log.info("Loading SCI data from %s", path)
    df = pd.read_csv(
        path,
        usecols=["user_region", "friend_region", "scaled_sci"],
        dtype={"user_region": str, "friend_region": str, "scaled_sci": np.int64},
    )
    log.info("  Raw rows: %d", len(df))

    # Zero-pad FIPS to 5 chars
    df["user_region"] = df["user_region"].str.zfill(5)
    df["friend_region"] = df["friend_region"].str.zfill(5)

    # Drop self-connections
    df = df[df["user_region"] != df["friend_region"]].copy()

    # Keep only upper triangle (user < friend) to deduplicate symmetric pairs
    df = df[df["user_region"] < df["friend_region"]].copy()
    log.info("  Upper-triangle rows: %d", len(df))

    # Filter to valid FIPS if provided
    if valid_fips is not None:
        mask = df["user_region"].isin(valid_fips) & df["friend_region"].isin(valid_fips)
        df = df[mask].copy()
        log.info("  After FIPS filter: %d", len(df))

    df = df.rename(columns={"user_region": "user_fips", "friend_region": "friend_fips"})
    return df[["user_fips", "friend_fips", "scaled_sci"]].reset_index(drop=True)


def load_type_assignments(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load type assignments and extract score matrix + dominant types.

    Returns
    -------
    (assignments_df, score_matrix, dominant_types)
        score_matrix: (N, J) array of soft membership scores
        dominant_types: (N,) array of primary type indices
    """
    log.info("Loading type assignments from %s", path)
    df = pd.read_parquet(path)

    score_cols = sorted(
        [c for c in df.columns if c.endswith("_score")],
        key=lambda c: int(c.split("_")[1]),
    )
    score_matrix = df[score_cols].values  # (N, J)
    dominant_types = df["dominant_type"].values
    log.info(
        "  %d counties, %d types, dominant types range [%d, %d]",
        len(df),
        len(score_cols),
        dominant_types.min(),
        dominant_types.max(),
    )
    return df, score_matrix, dominant_types


def load_county_centroids(path: Path) -> pd.DataFrame:
    """Load county centroids (lat/lon) from Census Gazetteer file.

    Expected columns: county_fips, latitude, longitude.
    """
    log.info("Loading county centroids from %s", path)
    df = pd.read_csv(path, dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].str.zfill(5)
    log.info("  %d counties with centroids", len(df))
    return df


def fetch_county_centroids(output_path: Path) -> pd.DataFrame:
    """Download 2020 Census Gazetteer county centroids if not on disk.

    Source: Census Bureau county gazetteer file (public domain, ~100KB).
    """
    if output_path.exists():
        log.info("County centroids already on disk: %s", output_path)
        return load_county_centroids(output_path)

    import io
    import zipfile
    from urllib.request import urlopen

    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2020_Gazetteer/2020_Gaz_counties_national.zip"
    )
    log.info("Downloading county centroids from %s", url)
    with urlopen(url) as resp:
        zip_data = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        # The zip contains a single tab-delimited file
        names = zf.namelist()
        txt_name = [n for n in names if n.endswith(".txt")][0]
        with zf.open(txt_name) as f:
            raw = pd.read_csv(f, sep="\t", dtype={"GEOID": str})

    # Census gazetteer has trailing whitespace in column names
    raw.columns = raw.columns.str.strip()

    # Standardize columns
    centroids = pd.DataFrame({
        "county_fips": raw["GEOID"].str.zfill(5),
        "latitude": raw["INTPTLAT"].astype(float),
        "longitude": raw["INTPTLONG"].astype(float),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_csv(output_path, index=False)
    log.info("Saved %d county centroids to %s", len(centroids), output_path)
    return centroids
