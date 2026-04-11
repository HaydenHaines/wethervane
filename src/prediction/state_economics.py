"""State-level economic signals from QCEW county employment data.

Builds state-level employment growth and wage growth features from the BLS
Quarterly Census of Employment and Wages (QCEW) county data.  These features
capture regional economic conditions that vary across states — a fundamentals
signal that the national-level model misses.

The QCEW data on disk (data/raw/qcew_county.parquet) contains annual averages
for ~3,192 counties across 2020-2023, with breakdowns by industry sector:
  10     = Total (all industries)
  23     = Construction
  31-33  = Manufacturing
  44-45  = Retail trade
  48-49  = Transportation/warehousing
  52     = Finance/insurance
  62     = Healthcare/social assistance
  72     = Accommodation/food services
  92     = Public administration

This module produces per-state features:
  - Employment growth (total, 2-year change relative to national)
  - Wage growth (average wage, 2-year change relative to national)
  - Manufacturing employment share (structural indicator)

These are designed to be added to the Ridge county priors model as additional
features: each county inherits its state's economic signal.

Why state-level rather than county-level:
  - County QCEW data is noisy for small counties (suppressed cells, small N)
  - State aggregation provides stable signals with real variation (std ~1.4pp
    for employment growth, ~2.2pp for wage growth)
  - State-level econ is what voters feel: local labor markets are regional

Usage:
    from src.prediction.state_economics import build_state_econ_features

    econ_df = build_state_econ_features()
    # Returns DataFrame with state_fips + feature columns
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QCEW_PATH = PROJECT_ROOT / "data" / "raw" / "qcew_county.parquet"

# Industry code for total employment (all industries combined).
_TOTAL_INDUSTRY_CODE = "10"

# Manufacturing NAICS supersector code.
_MANUFACTURING_CODE = "31-33"

# Minimum county employment to include in state aggregation.
# Avoids division-by-zero and noise from tiny counties.
_MIN_COUNTY_EMPLOYMENT = 10

# Feature column names produced by this module.
ECON_FEATURE_COLS = [
    "qcew_emp_growth_rel",
    "qcew_wage_growth_rel",
    "qcew_mfg_emp_share",
]


def _load_qcew(qcew_path: Path | None = None) -> pd.DataFrame:
    """Load QCEW county parquet and validate expected columns.

    Returns the raw DataFrame with columns:
    county_fips, own_code, industry_code, year,
    annual_avg_estabs, annual_avg_emplvl, total_annual_wages.
    """
    if qcew_path is None:
        qcew_path = _DEFAULT_QCEW_PATH

    if not qcew_path.exists():
        raise FileNotFoundError(
            f"QCEW county data not found at {qcew_path}. "
            "Expected: data/raw/qcew_county.parquet"
        )

    df = pd.read_parquet(qcew_path)
    required = {"county_fips", "industry_code", "year", "annual_avg_emplvl", "total_annual_wages"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QCEW parquet missing columns: {missing}")

    return df


def _aggregate_to_state(
    df: pd.DataFrame,
    industry_code: str,
) -> pd.DataFrame:
    """Aggregate county-level QCEW data to state level for a given industry.

    Parameters
    ----------
    df : DataFrame
        Raw QCEW data with county_fips, industry_code, year, etc.
    industry_code : str
        NAICS industry code to filter (e.g. "10" for total, "31-33" for manufacturing).

    Returns
    -------
    DataFrame with columns: state_fips, year, employment, wages, avg_wage.
    """
    sector = df[df["industry_code"] == industry_code].copy()
    sector["state_fips"] = sector["county_fips"].astype(str).str[:2]

    # Filter out tiny counties that add noise
    sector = sector[sector["annual_avg_emplvl"] >= _MIN_COUNTY_EMPLOYMENT]

    state = sector.groupby(["state_fips", "year"]).agg(
        employment=("annual_avg_emplvl", "sum"),
        wages=("total_annual_wages", "sum"),
    ).reset_index()

    state["avg_wage"] = state["wages"] / state["employment"].clip(lower=1)
    return state


def _compute_growth(
    state_df: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """Compute employment and wage growth between two years, relative to national.

    Returns DataFrame with state_fips, emp_growth_rel, wage_growth_rel.
    Both are deviations from the national (employment-weighted) mean.
    """
    start = state_df[state_df["year"] == year_start].set_index("state_fips")
    end = state_df[state_df["year"] == year_end].set_index("state_fips")

    # Only include states present in both years
    common = start.index.intersection(end.index)
    start = start.loc[common]
    end = end.loc[common]

    # State-level growth rates
    emp_growth = (end["employment"] - start["employment"]) / start["employment"]
    avg_wage_start = start["wages"] / start["employment"].clip(lower=1)
    avg_wage_end = end["wages"] / end["employment"].clip(lower=1)
    wage_growth = (avg_wage_end - avg_wage_start) / avg_wage_start

    # National average (employment-weighted for emp growth, simple mean for wage)
    nat_emp_growth = float(
        (end["employment"].sum() - start["employment"].sum()) / start["employment"].sum()
    )
    nat_wage_growth = float(wage_growth.mean())

    result = pd.DataFrame({
        "state_fips": common,
        "emp_growth_rel": (emp_growth - nat_emp_growth).values,
        "wage_growth_rel": (wage_growth - nat_wage_growth).values,
    })

    log.info(
        "QCEW growth %d->%d: national emp=%.3f, wage=%.3f | "
        "state emp_rel std=%.4f, wage_rel std=%.4f",
        year_start, year_end, nat_emp_growth, nat_wage_growth,
        result["emp_growth_rel"].std(), result["wage_growth_rel"].std(),
    )

    return result


def _compute_mfg_share(
    df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Compute manufacturing employment share per state for a given year.

    Returns DataFrame with state_fips, mfg_emp_share.
    """
    total_state = _aggregate_to_state(df, _TOTAL_INDUSTRY_CODE)
    mfg_state = _aggregate_to_state(df, _MANUFACTURING_CODE)

    total_yr = total_state[total_state["year"] == year].set_index("state_fips")
    mfg_yr = mfg_state[mfg_state["year"] == year].set_index("state_fips")

    common = total_yr.index.intersection(mfg_yr.index)
    share = mfg_yr.loc[common, "employment"] / total_yr.loc[common, "employment"]

    return pd.DataFrame({
        "state_fips": common,
        "mfg_emp_share": share.values,
    })


def build_state_econ_features(
    qcew_path: Path | None = None,
    year_start: int = 2021,
    year_end: int = 2023,
) -> pd.DataFrame:
    """Build state-level economic features from QCEW data.

    Produces a DataFrame with one row per state and columns:
      - state_fips: 2-digit state FIPS code
      - qcew_emp_growth_rel: employment growth relative to national (fraction)
      - qcew_wage_growth_rel: wage growth relative to national (fraction)
      - qcew_mfg_emp_share: manufacturing share of total employment (fraction)

    All growth rates are computed over the year_start to year_end window
    (default 2021-2023, a 2-year window for stability).

    Parameters
    ----------
    qcew_path : Path or None
        Path to QCEW county parquet.  Defaults to data/raw/qcew_county.parquet.
    year_start : int
        Start year for growth calculation.
    year_end : int
        End year for growth calculation.

    Returns
    -------
    pd.DataFrame
        State-level economic features (51 rows: 50 states + DC).
    """
    df = _load_qcew(qcew_path)

    # Validate year range
    available_years = sorted(df["year"].unique())
    if year_start not in available_years:
        raise ValueError(
            f"year_start={year_start} not in QCEW data (available: {available_years})"
        )
    if year_end not in available_years:
        raise ValueError(
            f"year_end={year_end} not in QCEW data (available: {available_years})"
        )

    # Total employment growth relative to national
    total_state = _aggregate_to_state(df, _TOTAL_INDUSTRY_CODE)
    growth = _compute_growth(total_state, year_start, year_end)

    # Manufacturing share
    mfg_share = _compute_mfg_share(df, year_end)

    # Merge
    result = growth.merge(mfg_share, on="state_fips", how="left")

    # Rename to final feature names
    result = result.rename(columns={
        "emp_growth_rel": "qcew_emp_growth_rel",
        "wage_growth_rel": "qcew_wage_growth_rel",
        "mfg_emp_share": "qcew_mfg_emp_share",
    })

    # Fill NaN manufacturing share with 0 (DC has very little manufacturing)
    result["qcew_mfg_emp_share"] = result["qcew_mfg_emp_share"].fillna(0.0)

    log.info(
        "Built state econ features: %d states, columns=%s",
        len(result), ECON_FEATURE_COLS,
    )

    return result[["state_fips"] + ECON_FEATURE_COLS]


def map_county_econ_features(
    county_fips: list[str] | np.ndarray,
    state_econ: pd.DataFrame | None = None,
    qcew_path: Path | None = None,
) -> pd.DataFrame:
    """Map state-level economic features to county FIPS codes.

    Each county inherits its state's economic signal.  This is the function
    to call from the Ridge model training pipeline.

    Parameters
    ----------
    county_fips : list[str] or ndarray
        County FIPS codes (5-digit, zero-padded).
    state_econ : DataFrame or None
        Pre-computed state features from build_state_econ_features().
        If None, computes from QCEW data on disk.
    qcew_path : Path or None
        Path to QCEW parquet (only used if state_econ is None).

    Returns
    -------
    pd.DataFrame
        DataFrame with county_fips + ECON_FEATURE_COLS, same order as input.
        Counties in states without QCEW data get 0.0 for all features.
    """
    if state_econ is None:
        state_econ = build_state_econ_features(qcew_path)

    county_df = pd.DataFrame({"county_fips": np.asarray(county_fips).astype(str)})
    county_df["state_fips"] = county_df["county_fips"].str[:2]

    merged = county_df.merge(state_econ, on="state_fips", how="left")

    # Fill missing states with 0 (neutral = no deviation from national)
    for col in ECON_FEATURE_COLS:
        merged[col] = merged[col].fillna(0.0)

    return merged[["county_fips"] + ECON_FEATURE_COLS]


# ---------------------------------------------------------------------------
# State-varying fundamentals adjustment
# ---------------------------------------------------------------------------

# Sensitivity of the fundamentals shift to state economic conditions.
# At scale=1.0, a state with +3% relative employment growth gets +3pp
# shift adjustment.  The default 0.5 dampens this to +1.5pp.
_DEFAULT_ECON_SENSITIVITY: float = 0.5


def compute_state_econ_adjustment(
    county_fips: list[str] | np.ndarray,
    states: list[str],
    national_shift: float,
    econ_sensitivity: float = _DEFAULT_ECON_SENSITIVITY,
    state_econ: pd.DataFrame | None = None,
    qcew_path: Path | None = None,
) -> np.ndarray:
    """Compute per-county state-varying fundamentals adjustment.

    Modulates the national fundamentals shift by state-level economic
    conditions from QCEW.  States performing better than the national
    average get a smaller in-party penalty (or larger bonus); states
    performing worse get a larger penalty.

    The adjustment for each county is:
        shift[c] = national_shift + econ_sensitivity * econ_signal[state(c)]

    where econ_signal is the state's relative employment growth (deviation
    from national average).  Employment growth is used rather than wage
    growth because: (1) it captures job creation/loss which voters feel
    directly, and (2) it has tighter cross-state variance, making it
    less noisy as a signal.

    Parameters
    ----------
    county_fips : list[str] or ndarray
        County FIPS codes (5-digit, zero-padded).  Length N.
    states : list[str]
        State abbreviation per county (used for logging only; state is
        derived from county_fips[:2]).
    national_shift : float
        The national fundamentals shift (scalar, from compute_fundamentals_shift
        or the combined GB + fundamentals blend).
    econ_sensitivity : float
        How strongly state economic conditions modulate the national shift.
        Higher = more state variation.  Default 0.5.
    state_econ : DataFrame or None
        Pre-computed state features.  If None, computes from QCEW data.
    qcew_path : Path or None
        Path to QCEW parquet (only used if state_econ is None).

    Returns
    -------
    ndarray of shape (N,)
        Per-county fundamentals shift (national + state adjustment).
    """
    if state_econ is None:
        try:
            state_econ = build_state_econ_features(qcew_path)
        except FileNotFoundError:
            log.warning(
                "QCEW data not found; falling back to uniform national shift"
            )
            return np.full(len(county_fips), national_shift)

    econ_map = map_county_econ_features(county_fips, state_econ=state_econ)

    # Use employment growth relative to national as the primary signal.
    # This is a fraction (e.g., +0.03 means 3pp faster than national).
    econ_signal = econ_map["qcew_emp_growth_rel"].values

    # Per-county shift = national + sensitivity * state_econ_deviation
    adjustment = national_shift + econ_sensitivity * econ_signal

    # Log summary
    adj_range = adjustment.max() - adjustment.min()
    log.info(
        "State econ adjustment: national=%.4f, sensitivity=%.2f, "
        "range=%.4f (min=%.4f, max=%.4f)",
        national_shift, econ_sensitivity,
        adj_range, adjustment.min(), adjustment.max(),
    )

    return adjustment
