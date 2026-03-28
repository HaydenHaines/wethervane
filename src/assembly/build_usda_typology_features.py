"""Build county-level USDA economic typology features from ERS typology codes.

USDA Economic Research Service classifies counties into economic specializations
and demographic stress categories. These binary flags capture structural economic
identity -- farming-dependent vs manufacturing-dependent vs government-dependent --
which strongly predicts partisan realignment patterns.

Features computed (13 binary flags):
  High_Farming_2025, High_Mining_2025, High_Manufacturing_2025,
  High_Government_2025, High_Recreation_2025, Nonspecialized_2025,
  Industry_Dependence_2025, Low_PostSecondary_Ed_2025, Low_Employment_2025,
  Population_Loss_2025, Housing_Stress_2025, Retirement_Destination_2025,
  Persistent_Poverty_1721

Input format is long (one row per county per attribute). Pivoted to wide format
with one row per county and binary flags as columns.

Input:  data/raw/usda_county_typology_2025.csv
Output: data/assembled/usda_typology_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "usda_county_typology_2025.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "usda_typology_features.parquet"

# Binary flag features (all are 0/1 except Industry_Dependence_2025 which is 0-6)
USDA_FEATURE_COLS = [
    "High_Farming_2025",
    "High_Mining_2025",
    "High_Manufacturing_2025",
    "High_Government_2025",
    "High_Recreation_2025",
    "Nonspecialized_2025",
    "Low_PostSecondary_Ed_2025",
    "Low_Employment_2025",
    "Population_Loss_2025",
    "Housing_Stress_2025",
    "Retirement_Destination_2025",
    "Persistent_Poverty_1721",
]

# Industry_Dependence_2025 is an ordinal count (0-6), not a binary flag.
# We include it as a separate numeric feature.
USDA_NUMERIC_COLS = ["Industry_Dependence_2025"]

ALL_USDA_COLS = USDA_FEATURE_COLS + USDA_NUMERIC_COLS


def build_usda_typology_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot USDA long-format typology data to wide county-level features.

    Parameters
    ----------
    raw:
        Long-format CSV with columns: FIPStxt, Attribute, Value.
        One row per county per typology attribute.

    Returns
    -------
    DataFrame with county_fips + USDA typology feature columns.
    """
    df = raw.copy()
    df = df.rename(columns={"FIPStxt": "county_fips"})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    # Filter to valid 5-digit county FIPS
    df = df[df["county_fips"].str.match(r"^\d{5}$", na=False)].copy()
    df = df[df["county_fips"].str[2:] != "000"].copy()

    # Keep only recognized attributes
    all_attrs = set(USDA_FEATURE_COLS + USDA_NUMERIC_COLS)
    df = df[df["Attribute"].isin(all_attrs)].copy()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Pivot: county_fips × Attribute → Value
    wide = df.pivot_table(
        index="county_fips", columns="Attribute", values="Value", aggfunc="first"
    ).reset_index()

    # Ensure all expected columns exist
    for col in ALL_USDA_COLS:
        if col not in wide.columns:
            log.warning("Missing attribute '%s' — filling with 0", col)
            wide[col] = 0

    # Binary flags: coerce to int (0/1)
    for col in USDA_FEATURE_COLS:
        wide[col] = wide[col].fillna(0).clip(0, 1).astype(int)

    # Industry_Dependence_2025: keep as numeric (0-6 count)
    for col in USDA_NUMERIC_COLS:
        wide[col] = wide[col].fillna(0).astype(int)

    result = wide[["county_fips"] + ALL_USDA_COLS].reset_index(drop=True)
    log.info("USDA typology: %d counties × %d features", len(result), len(ALL_USDA_COLS))
    return result


def main() -> None:
    if not INPUT_PATH.exists():
        log.error("USDA typology data not found at %s", INPUT_PATH)
        return

    log.info("Loading USDA county typology from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH)
    log.info("  %d rows × %d cols", len(raw), len(raw.columns))

    features = build_usda_typology_features(raw)

    # Summary: count of flagged counties per attribute
    log.info("\nAttribute prevalence:")
    for col in ALL_USDA_COLS:
        n_flagged = (features[col] > 0).sum()
        pct = 100 * n_flagged / len(features)
        log.info("  %-30s  %d counties (%.1f%%)", col, n_flagged, pct)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
