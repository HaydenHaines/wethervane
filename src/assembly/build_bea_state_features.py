"""Build state-level BEA economic features mapped to counties via FIPS prefix.

Reads state-level GDP and personal income per capita from the pre-fetched BEA
sample CSVs and maps them to counties by matching the first two digits of the
county FIPS code against the state FIPS prefix.

These are macro-economic signals that complement the existing county-level BEA
income decomposition (county_bea_features.parquet). While the county-level data
captures income composition (earnings/transfers/investment shares), these
state-level features capture the overall economic context:

  bea_state_gdp_millions     — state real GDP in millions of chained dollars
  bea_state_income_per_capita — state personal income per capita in dollars

Both features are mapped uniformly to every county in the state. Missing states
(e.g., territories not in the sample) are filled with the national median so
counties are never dropped from the feature matrix.

Data sources:
  data/raw/bea_state_gdp/state_gdp_2024_sample.csv
  data/raw/bea_state_gdp/state_income_2024_sample.csv

Outputs:
  data/assembled/county_bea_state_features.parquet
      county_fips + bea_state_gdp_millions + bea_state_income_per_capita

FIPS format: all county_fips values are 5-char zero-padded strings throughout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bea_state_gdp"

# Paths to the pre-fetched BEA state-level sample data
GDP_PATH = _DATA_DIR / "state_gdp_2024_sample.csv"
INCOME_PATH = _DATA_DIR / "state_income_2024_sample.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_state_features.parquet"

# Maps state name to 2-digit FIPS prefix (matching STATE_TO_FIPS in fetch script,
# but keyed on 2-digit prefix for easy county matching).
_STATE_NAME_TO_FIPS_PREFIX: dict[str, str] = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}

# Output column names — kept as module-level constants so callers can reference
# them without importing the full build pipeline.
COL_GDP = "bea_state_gdp_millions"
COL_INCOME = "bea_state_income_per_capita"
FEATURE_COLS = [COL_GDP, COL_INCOME]


def load_state_gdp(gdp_path: Path = GDP_PATH) -> pd.Series:
    """Load state GDP (millions) keyed on 2-digit FIPS prefix.

    Returns a Series indexed by 2-digit state FIPS prefix (e.g., "12" for FL).
    States not in the CSV are absent from the Series; callers should fill with
    national median via :func:`build_county_bea_state_features`.
    """
    if not gdp_path.exists():
        raise FileNotFoundError(
            f"BEA state GDP file not found: {gdp_path}. "
            "Run scripts/fetch_bea_state_data.py to populate it."
        )

    df = pd.read_csv(gdp_path)

    # Validate expected columns
    if "state" not in df.columns or "gdp_millions_2024" not in df.columns:
        raise ValueError(
            f"BEA GDP CSV must have 'state' and 'gdp_millions_2024' columns. "
            f"Found: {list(df.columns)}"
        )

    df["fips_prefix"] = df["state"].map(_STATE_NAME_TO_FIPS_PREFIX)
    missing = df[df["fips_prefix"].isna()]["state"].tolist()
    if missing:
        log.warning("State names not mapped to FIPS: %s", missing)

    df = df.dropna(subset=["fips_prefix"])
    return df.set_index("fips_prefix")["gdp_millions_2024"]


def load_state_income(income_path: Path = INCOME_PATH) -> pd.Series:
    """Load state personal income per capita keyed on 2-digit FIPS prefix.

    Returns a Series indexed by 2-digit state FIPS prefix (e.g., "12" for FL).
    States not in the CSV are absent from the Series.
    """
    if not income_path.exists():
        raise FileNotFoundError(
            f"BEA state income file not found: {income_path}. "
            "Run scripts/fetch_bea_state_data.py to populate it."
        )

    df = pd.read_csv(income_path)

    if "state" not in df.columns or "income_per_capita_2024" not in df.columns:
        raise ValueError(
            f"BEA income CSV must have 'state' and 'income_per_capita_2024' columns. "
            f"Found: {list(df.columns)}"
        )

    df["fips_prefix"] = df["state"].map(_STATE_NAME_TO_FIPS_PREFIX)
    missing = df[df["fips_prefix"].isna()]["state"].tolist()
    if missing:
        log.warning("State names not mapped to FIPS: %s", missing)

    df = df.dropna(subset=["fips_prefix"])
    return df.set_index("fips_prefix")["income_per_capita_2024"]


def build_county_bea_state_features(
    county_fips: list[str],
    gdp_path: Path = GDP_PATH,
    income_path: Path = INCOME_PATH,
) -> pd.DataFrame:
    """Map state-level BEA GDP and income data to counties via FIPS prefix.

    Each county inherits its state's economic values. Counties in states with
    no BEA data are imputed with the national median so the output covers all
    input counties without any gaps.

    Parameters
    ----------
    county_fips:
        List of 5-char zero-padded county FIPS strings (e.g., "12001").
    gdp_path:
        Path to the state GDP CSV. Defaults to the standard sample file.
    income_path:
        Path to the state income CSV. Defaults to the standard sample file.

    Returns
    -------
    DataFrame with columns: county_fips, bea_state_gdp_millions,
    bea_state_income_per_capita. One row per county in county_fips.

    Notes
    -----
    - All county_fips values must be exactly 5 characters.
    - Duplicate FIPS in the input are preserved (not deduplicated).
    - National medians are used for any state not in the BEA CSV.
    """
    gdp_by_state = load_state_gdp(gdp_path)
    income_by_state = load_state_income(income_path)

    df = pd.DataFrame({"county_fips": county_fips})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    if not df["county_fips"].str.len().eq(5).all():
        bad = df[df["county_fips"].str.len() != 5]["county_fips"].tolist()[:5]
        raise ValueError(f"county_fips must be 5-char strings. Bad examples: {bad}")

    state_prefix = df["county_fips"].str[:2]

    # Map state-level values onto counties
    df[COL_GDP] = state_prefix.map(gdp_by_state)
    df[COL_INCOME] = state_prefix.map(income_by_state)

    # Fill missing states with national median
    # (e.g., territories not covered by the BEA sample CSV)
    n_missing_gdp = df[COL_GDP].isna().sum()
    n_missing_income = df[COL_INCOME].isna().sum()

    if n_missing_gdp > 0:
        median_gdp = float(df[COL_GDP].median())
        df[COL_GDP] = df[COL_GDP].fillna(median_gdp)
        log.info(
            "%d counties lack BEA state GDP data — filled with national median %.1f",
            n_missing_gdp,
            median_gdp,
        )

    if n_missing_income > 0:
        median_income = float(df[COL_INCOME].median())
        df[COL_INCOME] = df[COL_INCOME].fillna(median_income)
        log.info(
            "%d counties lack BEA state income data — filled with national median %.1f",
            n_missing_income,
            median_income,
        )

    log.info(
        "Built BEA state features: %d counties, GDP range [%.0f, %.0f] M$, "
        "income range [%.0f, %.0f] $/person",
        len(df),
        df[COL_GDP].min(),
        df[COL_GDP].max(),
        df[COL_INCOME].min(),
        df[COL_INCOME].max(),
    )

    return df[["county_fips", COL_GDP, COL_INCOME]].reset_index(drop=True)


def main() -> None:
    """Build and save the county BEA state features parquet.

    Loads all county FIPS from the ACS spine (required to be present) and maps
    state-level GDP and income to each county. Output is written to
    data/assembled/county_bea_state_features.parquet.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    acs_path = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS spine not found: {acs_path}. "
            "Run src/assembly/build_county_acs_features.py first."
        )

    log.info("Loading ACS county spine from %s", acs_path)
    acs = pd.read_parquet(acs_path)
    county_fips = acs["county_fips"].astype(str).str.zfill(5).tolist()
    log.info("Found %d counties in ACS spine", len(county_fips))

    features = build_county_bea_state_features(county_fips)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d county BEA state features → %s", len(features), OUTPUT_PATH)

    log.info("BEA state GDP — Q1=%.0f, median=%.0f, Q3=%.0f",
             *features[COL_GDP].quantile([0.25, 0.5, 0.75]))
    log.info("BEA state income/cap — Q1=%.0f, median=%.0f, Q3=%.0f",
             *features[COL_INCOME].quantile([0.25, 0.5, 0.75]))


if __name__ == "__main__":
    main()
