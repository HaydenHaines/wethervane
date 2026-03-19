"""Build county-level electoral shift vectors (9-dimensional).

Aggregates tract-level VEST data to county level by population-weighted
averaging, then computes the same 9-dimensional shift vectors used in the
tract-level pipeline.

  pres_d_shift_16_20, pres_r_shift_16_20, pres_turnout_shift_16_20
  pres_d_shift_20_24, pres_r_shift_20_24, pres_turnout_shift_20_24
  mid_d_shift_18_22,  mid_r_shift_18_22,  mid_turnout_shift_18_22

Special cases:
  - Alabama (FIPS "01") 2018 gubernatorial was uncontested; midterm shift
    dimensions are set to 0.0 (structural zero).

Inputs (data/assembled/):
  vest_tracts_2016.parquet
  vest_tracts_2018.parquet
  vest_tracts_2020.parquet
  medsl_county_2022_governor.parquet
  medsl_county_2024_president.parquet

Output:
  data/shifts/county_shifts.parquet  — county_fips (str) + 9 shift columns
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

AL_FIPS_PREFIX = "01"

SHIFT_COLS: list[str] = [
    "pres_d_shift_16_20",
    "pres_r_shift_16_20",
    "pres_turnout_shift_16_20",
    "pres_d_shift_20_24",
    "pres_r_shift_20_24",
    "pres_turnout_shift_20_24",
    "mid_d_shift_18_22",
    "mid_r_shift_18_22",
    "mid_turnout_shift_18_22",
]


def aggregate_tracts_to_county(
    df: pd.DataFrame,
    dem_share_col: str,
    total_col: str,
) -> pd.DataFrame:
    """Population-weight aggregate tract dem_share to county level.

    county_dem_share = sum(tract_dem_share * tract_total) / sum(tract_total)

    Returns DataFrame with columns: county_fips, <dem_share_col>, <total_col>
    """
    df = df.copy()
    df["county_fips"] = df["tract_geoid"].str[:5]
    df["dem_votes"] = df[dem_share_col] * df[total_col]

    agg = (
        df.groupby("county_fips")
        .agg(
            dem_votes=("dem_votes", "sum"),
            total_votes=(total_col, "sum"),
        )
        .reset_index()
    )
    agg[dem_share_col] = agg["dem_votes"] / agg["total_votes"].replace(0, np.nan)
    agg = agg.rename(columns={"total_votes": total_col})
    return agg[["county_fips", dem_share_col, total_col]]


def main() -> None:
    log.info("Loading assembled election data...")
    vest_2016 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2016.parquet")
    vest_2018 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2018.parquet")
    vest_2020 = pd.read_parquet(ASSEMBLED_DIR / "vest_tracts_2020.parquet")
    medsl_2022 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2022_governor.parquet")
    medsl_2024 = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")

    # Ensure county_fips is string-padded
    medsl_2022["county_fips"] = medsl_2022["county_fips"].astype(str).str.zfill(5)
    medsl_2024["county_fips"] = medsl_2024["county_fips"].astype(str).str.zfill(5)

    # ── Aggregate VEST tract data to county level ─────────────────────────────
    county_2016 = aggregate_tracts_to_county(
        vest_2016, "pres_dem_share_2016", "pres_total_2016"
    )
    county_2018 = aggregate_tracts_to_county(
        vest_2018, "gov_dem_share_2018", "gov_total_2018"
    )
    county_2020 = aggregate_tracts_to_county(
        vest_2020, "pres_dem_share_2020", "pres_total_2020"
    )

    log.info(
        "Aggregated: 2016=%d counties, 2018=%d, 2020=%d, 2022=%d, 2024=%d",
        len(county_2016), len(county_2018), len(county_2020),
        len(medsl_2022), len(medsl_2024),
    )

    # ── Build spine from 2020 county list ─────────────────────────────────────
    spine = county_2020[["county_fips"]].copy()

    # ── Presidential 2016 → 2020 ──────────────────────────────────────────────
    m = spine.merge(county_2016, on="county_fips", how="left").merge(
        county_2020, on="county_fips", how="left", suffixes=("_16", "_20")
    )
    # After merge, columns are pres_dem_share_2016, pres_total_2016, pres_dem_share_2020, pres_total_2020
    spine["pres_d_shift_16_20"] = m["pres_dem_share_2020"] - m["pres_dem_share_2016"]
    spine["pres_r_shift_16_20"] = -spine["pres_d_shift_16_20"]
    early_total = m["pres_total_2016"].replace(0, np.nan)
    spine["pres_turnout_shift_16_20"] = (m["pres_total_2020"] - m["pres_total_2016"]) / early_total

    # ── Presidential 2020 → 2024 ──────────────────────────────────────────────
    m24 = spine[["county_fips"]].merge(county_2020, on="county_fips", how="left").merge(
        medsl_2024[["county_fips", "pres_dem_share_2024", "pres_total_2024"]],
        on="county_fips", how="left"
    )
    spine["pres_d_shift_20_24"] = m24["pres_dem_share_2024"] - m24["pres_dem_share_2020"]
    spine["pres_r_shift_20_24"] = -spine["pres_d_shift_20_24"]
    early_total_20 = m24["pres_total_2020"].replace(0, np.nan)
    spine["pres_turnout_shift_20_24"] = (
        m24["pres_total_2024"] - m24["pres_total_2020"]
    ) / early_total_20

    # ── Midterm 2018 → 2022 ───────────────────────────────────────────────────
    m22 = spine[["county_fips"]].merge(county_2018, on="county_fips", how="left").merge(
        medsl_2022[["county_fips", "gov_dem_share_2022", "gov_total_2022"]],
        on="county_fips", how="left"
    )
    spine["mid_d_shift_18_22"] = m22["gov_dem_share_2022"] - m22["gov_dem_share_2018"]
    spine["mid_r_shift_18_22"] = -spine["mid_d_shift_18_22"]
    early_total_18 = m22["gov_total_2018"].replace(0, np.nan)
    spine["mid_turnout_shift_18_22"] = (
        m22["gov_total_2022"] - m22["gov_total_2018"]
    ) / early_total_18

    # ── Structural zero for Alabama midterm ───────────────────────────────────
    al_mask = spine["county_fips"].str.startswith(AL_FIPS_PREFIX)
    spine.loc[al_mask, ["mid_d_shift_18_22", "mid_r_shift_18_22", "mid_turnout_shift_18_22"]] = 0.0
    log.info("Set %d Alabama counties' midterm shifts to 0.0 (structural zero)", al_mask.sum())

    # ── Validate ──────────────────────────────────────────────────────────────
    n_missing = spine[SHIFT_COLS].isna().any(axis=1).sum()
    if n_missing:
        log.warning("%d counties have NaN shifts; filling with column means", n_missing)
        spine[SHIFT_COLS] = spine[SHIFT_COLS].fillna(spine[SHIFT_COLS].mean())

    assert list(spine.columns) == ["county_fips"] + SHIFT_COLS, (
        f"Unexpected columns: {list(spine.columns)}"
    )

    log.info(
        "Shift summary: d_shift_16_20 mean=%.4f, d_shift_20_24 mean=%.4f, mid_d mean=%.4f",
        spine["pres_d_shift_16_20"].mean(),
        spine["pres_d_shift_20_24"].mean(),
        spine["mid_d_shift_18_22"].mean(),
    )

    SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SHIFTS_DIR / "county_shifts.parquet"
    spine.to_parquet(output_path, index=False)
    log.info("Saved → %s | %d counties | %d shift dimensions", output_path, len(spine), len(SHIFT_COLS))


if __name__ == "__main__":
    main()
