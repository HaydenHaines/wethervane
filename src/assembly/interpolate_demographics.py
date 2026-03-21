"""Interpolate decennial census demographics for election years.

Linear interpolation between census years (2000, 2010, 2020):
  weight_later = (election_year - earlier_census) / (later_census - earlier_census)
  value = (1 - weight_later) * earlier + weight_later * later

- Pre-2000: use Census 2000 flat (no extrapolation)
- Post-2020: use Census 2020 flat (no extrapolation)
- Income is CPI-adjusted to 2020 dollars BEFORE interpolation

Output:
    data/assembled/demographics_interpolated.parquet
    Long format keyed by (county_fips, year)

Usage:
    python -m src.assembly.interpolate_demographics
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"
CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"

CENSUS_YEARS = [2000, 2010, 2020]

# CPI adjustment factors (CPI-U annual average).
# Income year → CPI value.  Normalize to 2020 dollars.
# Loaded from config/model.yaml; these are defaults.
DEFAULT_CPI = {
    1999: 166.6,  # Census 2000 reports 1999 income
    2010: 218.1,
    2020: 258.8,
}

# Map census year → income reference year for CPI adjustment
INCOME_REF_YEAR = {2000: 1999, 2010: 2010, 2020: 2020}

# Columns to interpolate (all numeric measures from fetch_census_decennial)
INTERPOLATION_COLS = [
    "pop_total", "pop_white_nh", "pop_black", "pop_asian", "pop_hispanic",
    "median_age", "median_hh_income",
    "housing_total", "housing_owner",
    "educ_total", "educ_bachelors_plus",
    "commute_total", "commute_car", "commute_transit", "commute_wfh",
]

# Derived ratio definitions: (name, numerator_col, denominator_col)
DERIVED_RATIOS = [
    ("pct_white_nh", "pop_white_nh", "pop_total"),
    ("pct_black", "pop_black", "pop_total"),
    ("pct_asian", "pop_asian", "pop_total"),
    ("pct_hispanic", "pop_hispanic", "pop_total"),
    ("pct_bachelors_plus", "educ_bachelors_plus", "educ_total"),
    ("pct_owner_occupied", "housing_owner", "housing_total"),
    ("pct_wfh", "commute_wfh", "commute_total"),
    ("pct_transit", "commute_transit", "commute_total"),
    ("pct_car", "commute_car", "commute_total"),
]


# ── CPI adjustment ───────────────────────────────────────────────────────────


def _load_cpi() -> dict[int, float]:
    """Load CPI adjustment factors from config, falling back to defaults."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get("census", {}).get("cpi_adjustments", {})
        return {int(k): float(v) for k, v in raw.items()}
    except (FileNotFoundError, KeyError):
        return DEFAULT_CPI


def _adjust_income_to_2020(
    frames: dict[int, pd.DataFrame],
    cpi: dict[int, float] | None = None,
) -> dict[int, pd.DataFrame]:
    """CPI-adjust median_hh_income in each census frame to 2020 dollars.

    Modifies frames in place and returns them.
    """
    if cpi is None:
        cpi = _load_cpi()
    target_cpi = cpi[2020]

    for year, df in frames.items():
        ref_year = INCOME_REF_YEAR[year]
        factor = target_cpi / cpi[ref_year]
        df["median_hh_income"] = df["median_hh_income"] * factor

    return frames


# ── Interpolation ─────────────────────────────────────────────────────────────


def interpolate_for_year(
    census_frames: dict[int, pd.DataFrame],
    election_year: int,
    cpi: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Interpolate demographics for a single election year.

    Parameters
    ----------
    census_frames : dict
        Mapping of census year (2000, 2010, 2020) to DataFrame.
        Each DataFrame must have 'county_fips' and all INTERPOLATION_COLS.
    election_year : int
        The year to interpolate for.
    cpi : dict, optional
        CPI factors.  Defaults to config/model.yaml values.

    Returns
    -------
    pd.DataFrame
        Interpolated demographics with derived ratio columns and 'year'.
    """
    if cpi is None:
        cpi = _load_cpi()

    # CPI-adjust income (work on copies to avoid mutating input)
    adjusted = {}
    for yr, df in census_frames.items():
        adjusted[yr] = df.copy()
    adjusted = _adjust_income_to_2020(adjusted, cpi)

    # Determine bracketing census years
    if election_year <= CENSUS_YEARS[0]:
        # Pre-2000: use 2000 flat
        base = adjusted[CENSUS_YEARS[0]].copy()
    elif election_year >= CENSUS_YEARS[-1]:
        # Post-2020: use 2020 flat
        base = adjusted[CENSUS_YEARS[-1]].copy()
    else:
        # Find bracketing years
        earlier = max(y for y in CENSUS_YEARS if y <= election_year)
        later = min(y for y in CENSUS_YEARS if y > election_year)

        if earlier == election_year:
            base = adjusted[earlier].copy()
        else:
            weight_later = (election_year - earlier) / (later - earlier)
            weight_earlier = 1.0 - weight_later

            df_early = adjusted[earlier].set_index("county_fips")
            df_late = adjusted[later].set_index("county_fips")

            # Align on county_fips
            common_fips = df_early.index.intersection(df_late.index)
            df_early = df_early.loc[common_fips]
            df_late = df_late.loc[common_fips]

            result = pd.DataFrame({"county_fips": common_fips})
            for col in INTERPOLATION_COLS:
                result[col] = (
                    weight_earlier * df_early[col].values
                    + weight_later * df_late[col].values
                )
            base = result

    # Set year
    base["year"] = election_year

    # Drop old year column if present from census frame
    if "year" in base.columns and base.columns.tolist().count("year") > 1:
        base = base.loc[:, ~base.columns.duplicated(keep="last")]

    # Compute derived ratios
    for name, num, den in DERIVED_RATIOS:
        base[name] = base[num] / base[den]

    # Keep only standardized columns
    keep = ["county_fips", "year"] + INTERPOLATION_COLS + [r[0] for r in DERIVED_RATIOS]
    base = base[[c for c in keep if c in base.columns]]

    return base.reset_index(drop=True)


# ── Collect all election years ────────────────────────────────────────────────


def _get_election_years() -> list[int]:
    """Read all election years from config/model.yaml, deduplicated and sorted."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        election = cfg.get("election", {})
        years = set()
        for key in ("presidential_years", "governor_years", "senate_years"):
            years.update(election.get(key, []))
        return sorted(years)
    except FileNotFoundError:
        log.warning("config/model.yaml not found; using default years")
        return sorted({2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024})


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Interpolate demographics for all election years and save."""
    election_years = _get_election_years()
    log.info("Election years: %s", election_years)

    # Load census parquet files
    census_frames: dict[int, pd.DataFrame] = {}
    for yr in CENSUS_YEARS:
        path = OUTPUT_DIR / f"census_{yr}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run fetch_census_decennial first."
            )
        census_frames[yr] = pd.read_parquet(path)
        log.info("Loaded census_%d: %d counties", yr, len(census_frames[yr]))

    # Interpolate for each election year
    frames = []
    for yr in election_years:
        df = interpolate_for_year(census_frames, yr)
        frames.append(df)
        log.info("  %d: %d counties interpolated", yr, len(df))

    combined = pd.concat(frames, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "demographics_interpolated.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(
        "Saved -> %s (%d rows, %d years, %d counties/year)",
        out_path,
        len(combined),
        len(election_years),
        len(combined) // max(len(election_years), 1),
    )


if __name__ == "__main__":
    main()
