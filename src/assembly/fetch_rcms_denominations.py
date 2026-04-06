"""Parse ARDA RCMSCY20 county-level denomination data into a clean feature parquet.

Source: Association of Religion Data Archives (ARDA)
File: data/raw/rcmscy20.csv (838 columns, ~3,100 county rows)

The RCMS 2020 dataset contains denomination-level adherent counts that go beyond
the major-group aggregates (evangelical/mainline/Catholic/Black Protestant) already
used in the pipeline. The "Other" category in the aggregate data lumps together LDS,
Muslim, Jewish, Hindu, and Sikh communities — groups with meaningfully distinct
electoral behavior. This module extracts those denomination-specific signals.

Denomination columns in the raw CSV:
  LDS (Latter-day Saints):
    LDSADH_2020   — adherent count
    LDSRATE_2020  — adherents per 1,000 residents (RCMS-computed)

  Muslim:
    MSLMADH_2020  — adherent count
    MSLMRATE_2020 — adherents per 1,000 residents

  Jewish:
    JWADH_2020    — adherent count
    JWRATE_2020   — adherents per 1,000 residents

  Hindu (two sub-bodies; combined into one feature):
    HINTADH_2020  — Hindu Temple adherent count
    HINYMADH_2020 — Hindu Yoga/Meditation adherent count

  Sikh:
    SIKHCNG_2020  — congregation count only (no adherent data in RCMS 2020)

Rate computation:
  LDS, Muslim, Jewish: use the pre-computed RCMS rate columns (LDSRATE, MSLMRATE, JWRATE)
  where available; fall back to ADH / POP2020 * 1000 for counties missing the rate column.

  Hindu: sum HINTADH + HINYMADH, then divide by POP2020 * 1000 to get rate.

  Sikh (hindu_sikh_rate): Sikh has no adherent data, only congregation count. We scale
  congregations by the national average Hindu adherents-per-congregation (~2,085, computed
  from the HINTADH/HINTCNG ratio across non-null rows) to estimate Sikh adherents, then
  add to Hindu adherents and compute a combined hindu_sikh_rate per 1,000 residents.
  This is a rough proxy — Sikh congregations are small enough (~165 counties) that the
  combined feature is dominated by Hindu adherents where both exist.

NaN policy:
  All four output columns are filled with 0.0 for counties with no data. Absence of
  denomination data in RCMS almost always means near-zero adherents in rural/small counties,
  not missing data. The few metropolitan counties with genuinely suppressed data will
  be imputed by the state-median fallback in build_county_features_national.py if needed
  (but since we fill with 0.0 here, no imputation is required).

Output columns:
  county_fips     — 5-digit zero-padded string
  lds_rate        — LDS adherents per 1,000 residents
  muslim_rate     — Muslim adherents per 1,000 residents
  jewish_rate     — Jewish adherents per 1,000 residents
  hindu_sikh_rate — (Hindu + Sikh-estimated) adherents per 1,000 residents

Output: data/raw/rcms_denominations.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rcmscy20.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rcms_denominations.parquet"

# Columns needed from the raw CSV
RAW_COLS = [
    "FIPS",
    "POP2020",
    "LDSADH_2020",
    "LDSRATE_2020",
    "MSLMADH_2020",
    "MSLMRATE_2020",
    "JWADH_2020",
    "JWRATE_2020",
    "HINTADH_2020",
    "HINYMADH_2020",
    "HINTCNG_2020",   # Hindu Temple congregations — used to estimate adh/cng ratio
    "SIKHCNG_2020",
]

# Fallback: if RCMS pre-computed rate is missing, we can derive it from ADH/POP.
# This average is computed from the non-null HINT rows.
# Recomputed dynamically in _compute_hindu_sikh_rate() but this constant is the
# expected value used for tests and documentation.
EXPECTED_HINDU_ADH_PER_CNG = 2085.0


def _pad_fips(fips_series: pd.Series) -> pd.Series:
    """Zero-pad FIPS codes to 5 digits.

    The raw CSV stores FIPS as integers (e.g. 1001 for Alabama's Autauga County)
    which drops the leading zero for Southern states. Convert to string and zero-pad.
    """
    return fips_series.astype(str).str.zfill(5)


def _compute_rate_from_adh_pop(adh: pd.Series, pop: pd.Series) -> pd.Series:
    """Compute denomination rate as adherents per 1,000 residents.

    Uses population as denominator. Counties with zero or NaN population
    return NaN (the downstream fillna(0.0) handles them).
    """
    safe_pop = pop.replace(0, np.nan)
    return (adh / safe_pop) * 1000.0


def _compute_lds_rate(df: pd.DataFrame) -> pd.Series:
    """Compute LDS adherents per 1,000 residents.

    Prefers the RCMS pre-computed LDSRATE_2020 column. Falls back to
    LDSADH_2020 / POP2020 * 1000 for any county with a missing rate but
    non-missing adherent count.
    """
    rate = df["LDSRATE_2020"].copy()
    missing_rate = rate.isna() & df["LDSADH_2020"].notna()
    if missing_rate.any():
        fallback = _compute_rate_from_adh_pop(df["LDSADH_2020"], df["POP2020"])
        rate[missing_rate] = fallback[missing_rate]
    return rate


def _compute_muslim_rate(df: pd.DataFrame) -> pd.Series:
    """Compute Muslim adherents per 1,000 residents."""
    rate = df["MSLMRATE_2020"].copy()
    missing_rate = rate.isna() & df["MSLMADH_2020"].notna()
    if missing_rate.any():
        fallback = _compute_rate_from_adh_pop(df["MSLMADH_2020"], df["POP2020"])
        rate[missing_rate] = fallback[missing_rate]
    return rate


def _compute_jewish_rate(df: pd.DataFrame) -> pd.Series:
    """Compute Jewish adherents per 1,000 residents."""
    rate = df["JWRATE_2020"].copy()
    missing_rate = rate.isna() & df["JWADH_2020"].notna()
    if missing_rate.any():
        fallback = _compute_rate_from_adh_pop(df["JWADH_2020"], df["POP2020"])
        rate[missing_rate] = fallback[missing_rate]
    return rate


def _compute_hindu_sikh_rate(df: pd.DataFrame) -> pd.Series:
    """Compute combined Hindu + Sikh adherents per 1,000 residents.

    Hindu: sum HINTADH_2020 (Hindu Temple) + HINYMADH_2020 (Hindu Yoga/Meditation),
    treating NaN as 0 when at least one sub-body is present.

    Sikh: RCMS 2020 does not provide Sikh adherent counts — only congregation counts
    (SIKHCNG_2020). We scale by the national average Hindu adherents per congregation,
    computed from HINTADH_2020 / HINTCNG_2020 across rows where both are non-null.
    This yields an approximate Sikh adherent estimate. The Sikh signal is present in
    only ~165 counties; the combined feature is dominated by Hindu where overlap occurs.
    """
    # Compute average Hindu adherents per congregation from non-null rows
    valid_hint = df["HINTADH_2020"].notna() & df["HINTCNG_2020"].notna() & (df["HINTCNG_2020"] > 0)
    if valid_hint.sum() > 0:
        avg_adh_per_cng = (df.loc[valid_hint, "HINTADH_2020"] / df.loc[valid_hint, "HINTCNG_2020"]).mean()
        log.info("Average Hindu adherents per congregation: %.1f (from %d counties)", avg_adh_per_cng, valid_hint.sum())
    else:
        avg_adh_per_cng = EXPECTED_HINDU_ADH_PER_CNG
        log.warning("No valid HINTADH/HINTCNG rows; using fallback %.1f", avg_adh_per_cng)

    # Hindu adherents: sum two sub-bodies, treating NaN as 0 if the other sub-body is present
    hindu_adh = df[["HINTADH_2020", "HINYMADH_2020"]].sum(axis=1, min_count=1)
    # Where both are NaN, sum returns NaN — that's correct (means no Hindu data)

    # Sikh estimated adherents: scale congregations by avg adh/cng ratio
    sikh_adh = df["SIKHCNG_2020"] * avg_adh_per_cng

    # Combined: add Hindu + Sikh adherents; treat NaN as 0 when at least one is present
    combined_adh = hindu_adh.fillna(0.0) + sikh_adh.fillna(0.0)
    # Mark rows where neither Hindu nor Sikh data exists as NaN (to fill with 0 downstream)
    no_data = hindu_adh.isna() & sikh_adh.isna()
    combined_adh[no_data] = np.nan

    return _compute_rate_from_adh_pop(combined_adh, df["POP2020"])


def parse_denominations(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract denomination features from raw RCMSCY20 CSV data.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw RCMSCY20 CSV loaded with at least the columns in RAW_COLS.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, lds_rate, muslim_rate, jewish_rate, hindu_sikh_rate.
        All rate columns filled with 0.0 for counties with no denomination data.
        county_fips is a 5-digit zero-padded string.
    """
    df = raw[RAW_COLS].copy()
    df["county_fips"] = _pad_fips(df["FIPS"])

    result = pd.DataFrame({"county_fips": df["county_fips"]})
    result["lds_rate"] = _compute_lds_rate(df).fillna(0.0)
    result["muslim_rate"] = _compute_muslim_rate(df).fillna(0.0)
    result["jewish_rate"] = _compute_jewish_rate(df).fillna(0.0)
    result["hindu_sikh_rate"] = _compute_hindu_sikh_rate(df).fillna(0.0)

    # Validate FIPS format before returning
    bad_fips = result[~result["county_fips"].str.match(r"^\d{5}$")]
    if len(bad_fips) > 0:
        log.warning("Dropping %d rows with malformed FIPS: %s", len(bad_fips), bad_fips["county_fips"].tolist()[:5])
        result = result[result["county_fips"].str.match(r"^\d{5}$")]

    return result.reset_index(drop=True)


def main() -> None:
    """Parse RCMSCY20 CSV and write denomination features parquet."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"RCMSCY20 CSV not found at {INPUT_PATH}")

    log.info("Loading RCMSCY20 CSV from %s", INPUT_PATH)
    raw = pd.read_csv(INPUT_PATH, usecols=lambda c: c in RAW_COLS)
    log.info("  %d rows × %d cols loaded", len(raw), len(raw.columns))

    result = parse_denominations(raw)

    # Summary statistics
    log.info("\nDenomination coverage (non-zero counties):")
    for col in ["lds_rate", "muslim_rate", "jewish_rate", "hindu_sikh_rate"]:
        n_nonzero = (result[col] > 0).sum()
        log.info("  %-20s  %d counties with non-zero values", col, n_nonzero)

    log.info("\nRate distributions (non-zero counties only):")
    for col in ["lds_rate", "muslim_rate", "jewish_rate", "hindu_sikh_rate"]:
        nonzero = result[result[col] > 0][col]
        if len(nonzero) > 0:
            log.info(
                "  %-20s  median=%.1f  p75=%.1f  p95=%.1f  max=%.1f",
                col, nonzero.median(), nonzero.quantile(0.75), nonzero.quantile(0.95), nonzero.max()
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    log.info("\nSaved %d counties × %d columns → %s", len(result), len(result.columns), OUTPUT_PATH)


if __name__ == "__main__":
    main()
