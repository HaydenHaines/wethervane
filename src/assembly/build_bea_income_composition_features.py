"""Build county-level BEA income composition features from state-level shares.

Reads state-level income composition data (fetched by fetch_bea_income_composition.py)
and maps each state's shares uniformly to every county in that state by matching the
first two digits of the county FIPS code against the state FIPS prefix.

Why map state shares to counties?
  The state-level signal captures the *structural context* that every county in a state
  shares: a state dominated by transfer income (e.g., high-retirement, high-disability)
  creates different baseline incentives than a high-wage state.

Features produced (one per county, mapped from its state):
  bea_wages_share      -- wages & salaries as share of state personal income
  bea_transfer_share   -- transfer receipts (Social Security, disability, UI, etc.)
  bea_investment_share -- dividends, interest, and rent as share of personal income

Missing states are filled with the national median so no county is dropped.

Input:
  data/assembled/bea_income_composition.parquet  (state-level shares from SAINC4)

Output:
  data/assembled/county_bea_income_composition.parquet
    county_fips + bea_wages_share + bea_transfer_share + bea_investment_share
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_SHARES_PATH = PROJECT_ROOT / "data" / "assembled" / "bea_income_composition.parquet"
ACS_PATH = PROJECT_ROOT / "data" / "assembled" / "county_acs_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_bea_income_composition.parquet"

COL_WAGES = "bea_wages_share"
COL_TRANSFER = "bea_transfer_share"
COL_INVESTMENT = "bea_investment_share"
FEATURE_COLS = [COL_WAGES, COL_TRANSFER, COL_INVESTMENT]

_SRC_WAGES = "wages_share"
_SRC_TRANSFER = "transfer_share"
_SRC_INVESTMENT = "investment_share"
_SRC_FIPS = "state_fips_prefix"


def load_state_shares(path: Path = STATE_SHARES_PATH) -> pd.DataFrame:
    """Load state-level income composition shares from assembled parquet.

    Returns a DataFrame indexed by 2-digit state FIPS prefix with columns:
    wages_share, transfer_share, investment_share.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"State income composition file not found: {path}\n"
            "Run src/assembly/fetch_bea_income_composition.py first:\n"
            "  uv run python -m src.assembly.fetch_bea_income_composition"
        )

    df = pd.read_parquet(path)

    required = [_SRC_FIPS, _SRC_WAGES, _SRC_TRANSFER, _SRC_INVESTMENT]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"State shares file missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.set_index(_SRC_FIPS)
    return df[[_SRC_WAGES, _SRC_TRANSFER, _SRC_INVESTMENT]]


def build_county_bea_income_composition(
    county_fips: list[str],
    state_shares_path: Path = STATE_SHARES_PATH,
) -> pd.DataFrame:
    """Map state-level income composition shares to counties via FIPS prefix.

    Each county inherits its state's income composition values. Counties in
    states with no BEA data are imputed with the national median.

    Parameters
    ----------
    county_fips:
        List of 5-char zero-padded county FIPS strings (e.g., "12001").
    state_shares_path:
        Path to the state-level income composition parquet.

    Returns
    -------
    DataFrame with columns: county_fips, bea_wages_share, bea_transfer_share,
    bea_investment_share.
    """
    shares = load_state_shares(state_shares_path)

    df = pd.DataFrame({"county_fips": county_fips})
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

    bad = df[df["county_fips"].str.len() != 5]["county_fips"].tolist()
    if bad:
        raise ValueError(f"county_fips must be 5-char zero-padded strings. Bad examples: {bad[:5]}")

    state_prefix = df["county_fips"].str[:2]

    df[COL_WAGES] = state_prefix.map(shares[_SRC_WAGES])
    df[COL_TRANSFER] = state_prefix.map(shares[_SRC_TRANSFER])
    df[COL_INVESTMENT] = state_prefix.map(shares[_SRC_INVESTMENT])

    # Fill missing states with the national median so no county is dropped
    for col, src_col in [(COL_WAGES, _SRC_WAGES), (COL_TRANSFER, _SRC_TRANSFER), (COL_INVESTMENT, _SRC_INVESTMENT)]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            log.info("%d counties lack BEA state %s -- filled with national median %.4f", n_missing, col, median_val)

    log.info(
        "Built county BEA income composition features: %d counties  wages [%.3f, %.3f]  transfer [%.3f, %.3f]  investment [%.3f, %.3f]",
        len(df), df[COL_WAGES].min(), df[COL_WAGES].max(), df[COL_TRANSFER].min(), df[COL_TRANSFER].max(), df[COL_INVESTMENT].min(), df[COL_INVESTMENT].max(),
    )

    return df[["county_fips"] + FEATURE_COLS].reset_index(drop=True)


def main() -> None:
    """Build and save county BEA income composition features."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not ACS_PATH.exists():
        raise FileNotFoundError(f"ACS spine not found: {ACS_PATH}\nRun src/assembly/build_county_acs_features.py first.")

    log.info("Loading ACS county spine from %s", ACS_PATH)
    acs = pd.read_parquet(ACS_PATH)
    county_fips = acs["county_fips"].astype(str).str.zfill(5).tolist()
    log.info("Found %d counties in ACS spine", len(county_fips))

    features = build_county_bea_income_composition(county_fips)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved %d county BEA income composition features -> %s", len(features), OUTPUT_PATH)

    for col in FEATURE_COLS:
        q1, med, q3 = features[col].quantile([0.25, 0.5, 0.75])
        log.info("  %-25s  Q1=%.4f  median=%.4f  Q3=%.4f", col, q1, med, q3)


if __name__ == "__main__":
    main()
