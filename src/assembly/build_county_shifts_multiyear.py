"""Build multi-year county shift vectors (33 dimensions).

Training (30 dims):
  5 consecutive presidential pairs: 2000→2004, 2004→2008, 2008→2012,
                                    2012→2016, 2016→2020
  5 consecutive governor pairs:     2002→2006, 2006→2010, 2010→2014,
                                    2014→2018, 2018→2022

Holdout (3 dims — never used during clustering):
  Presidential 2020→2024

Note: All AL governor cycles are fully contested. No structural zeros needed
(AL 2018 governor: Kay Ivey vs Walter Maddox, Ivey won ~60/40 — real data).

Output:
  data/shifts/county_shifts_multiyear.parquet
  Columns: county_fips + 30 training + 3 holdout shift dims
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED_DIR = PROJECT_ROOT / "data" / "assembled"
SHIFTS_DIR = PROJECT_ROOT / "data" / "shifts"

AL_FIPS_PREFIX = "01"

# ── Column name constants ─────────────────────────────────────────────────────

PRES_PAIRS = [("00", "04"), ("04", "08"), ("08", "12"), ("12", "16"), ("16", "20")]
GOV_PAIRS  = [("02", "06"), ("06", "10"), ("10", "14"), ("14", "18"), ("18", "22")]
HOLDOUT_PAIRS = [("20", "24")]

TRAINING_SHIFT_COLS: list[str] = []
for a, b in PRES_PAIRS:
    TRAINING_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
for a, b in GOV_PAIRS:
    TRAINING_SHIFT_COLS += [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]

HOLDOUT_SHIFT_COLS: list[str] = []
for a, b in HOLDOUT_PAIRS:
    HOLDOUT_SHIFT_COLS += [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]


# ── Core computation ──────────────────────────────────────────────────────────

def _dem_share_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "dem_share" in c)

def _total_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if "_total_" in c)


def compute_pres_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two presidential election DataFrames."""
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = merged[late_dem] - merged[early_dem]
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    return pd.DataFrame({
        "county_fips": merged["county_fips"].values,
        f"pres_d_shift_{a}_{b}": d_shift.values,
        f"pres_r_shift_{a}_{b}": r_shift.values,
        f"pres_turnout_shift_{a}_{b}": t_shift.values,
    })


def compute_gov_shift(
    early: pd.DataFrame, late: pd.DataFrame, a: str, b: str
) -> pd.DataFrame:
    """Compute D/R/turnout shifts between two governor election DataFrames."""
    early_dem = _dem_share_col(early)
    early_tot = _total_col(early)
    late_dem = _dem_share_col(late)
    late_tot = _total_col(late)

    merged = early[["county_fips", early_dem, early_tot]].merge(
        late[["county_fips", late_dem, late_tot]], on="county_fips", how="inner"
    )
    d_shift = merged[late_dem] - merged[early_dem]
    r_shift = -(d_shift)
    early_total = merged[early_tot].replace(0, float("nan"))
    t_shift = (merged[late_tot] - merged[early_tot]) / early_total

    return pd.DataFrame({
        "county_fips": merged["county_fips"].values,
        f"gov_d_shift_{a}_{b}": d_shift.values,
        f"gov_r_shift_{a}_{b}": r_shift.values,
        f"gov_turnout_shift_{a}_{b}": t_shift.values,
    })


def build_multiyear_shifts(
    spine: pd.DataFrame,
    pres_pairs: list,
    gov_pairs: list,
) -> pd.DataFrame:
    """Join all shift pairs onto the county spine.

    pres_pairs: list of (a_str, b_str, early_df, late_df)
    gov_pairs: list of (a_str, b_str, early_df, late_df)

    Missing pairs produce zero-filled columns (logged as warning).
    """
    result = spine[["county_fips"]].copy()

    all_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in all_cols:
        result[col] = 0.0

    for a, b, early, late in pres_pairs:
        shifts = compute_pres_shift(early, late, a, b)
        cols = [f"pres_d_shift_{a}_{b}", f"pres_r_shift_{a}_{b}", f"pres_turnout_shift_{a}_{b}"]
        merged = result[["county_fips"]].merge(shifts[["county_fips"] + cols], on="county_fips", how="left")
        for col in cols:
            result[col] = merged[col].fillna(0.0).values

    for a, b, early, late in gov_pairs:
        shifts = compute_gov_shift(early, late, a, b)
        cols = [f"gov_d_shift_{a}_{b}", f"gov_r_shift_{a}_{b}", f"gov_turnout_shift_{a}_{b}"]
        merged = result[["county_fips"]].merge(shifts[["county_fips"] + cols], on="county_fips", how="left")
        for col in cols:
            result[col] = merged[col].fillna(0.0).values

    return result[["county_fips"] + TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS]


def _load(filename: str) -> pd.DataFrame | None:
    path = ASSEMBLED_DIR / filename
    if not path.exists():
        log.warning("Missing: %s — will zero-fill this pair", path.name)
        return None
    return pd.read_parquet(path)


def main() -> None:
    """Load all election parquets and build multi-year shifts."""
    # Spine: use medsl_county_presidential_2024 (new format file)
    spine_df = _load("medsl_county_presidential_2024.parquet")
    if spine_df is None:
        # Fallback to old-format file
        spine_df = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")
    spine = spine_df[["county_fips"]].copy()
    log.info("County spine: %d counties", len(spine))

    # ── Presidential pairs ────────────────────────────────────────────────────
    pres_file = {
        "00": "medsl_county_presidential_2000.parquet",
        "04": "medsl_county_presidential_2004.parquet",
        "08": "medsl_county_presidential_2008.parquet",
        "12": "medsl_county_presidential_2012.parquet",
        "16": "medsl_county_presidential_2016.parquet",
        "20": "medsl_county_presidential_2020.parquet",
        "24": "medsl_county_presidential_2024.parquet",
    }
    pres_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in pres_file.items()}
    # Fallback for 24 if new file not present
    if pres_dfs.get("24") is None:
        pres_dfs["24"] = pd.read_parquet(ASSEMBLED_DIR / "medsl_county_2024_president.parquet")

    pres_pairs = []
    for a, b in PRES_PAIRS + HOLDOUT_PAIRS:
        early, late = pres_dfs.get(a), pres_dfs.get(b)
        if early is not None and late is not None:
            pres_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping pres pair %s→%s (data missing)", a, b)

    # ── Governor pairs ────────────────────────────────────────────────────────
    gov_file = {
        "02": "algara_county_governor_2002.parquet",
        "06": "algara_county_governor_2006.parquet",
        "10": "algara_county_governor_2010.parquet",
        "14": "algara_county_governor_2014.parquet",
        "18": "algara_county_governor_2018.parquet",
        "22": "medsl_county_2022_governor.parquet",
    }
    gov_dfs: dict[str, pd.DataFrame | None] = {k: _load(v) for k, v in gov_file.items()}

    gov_pairs = []
    for a, b in GOV_PAIRS:
        early, late = gov_dfs.get(a), gov_dfs.get(b)
        if early is not None and late is not None:
            gov_pairs.append((a, b, early, late))
        else:
            log.warning("Skipping gov pair %s→%s (data missing)", a, b)

    # ── Build ─────────────────────────────────────────────────────────────────
    log.info("Building multi-year shifts: %d pres pairs, %d gov pairs",
             len(pres_pairs), len(gov_pairs))
    shifts = build_multiyear_shifts(spine, pres_pairs, gov_pairs)

    SHIFTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SHIFTS_DIR / "county_shifts_multiyear.parquet"
    shifts.to_parquet(out, index=False)
    log.info("Saved → %s | %d counties | %d training dims + %d holdout dims",
             out, len(shifts), len(TRAINING_SHIFT_COLS), len(HOLDOUT_SHIFT_COLS))

    zero_cols = [c for c in TRAINING_SHIFT_COLS if shifts[c].abs().max() < 1e-10]
    if zero_cols:
        log.warning("These training columns are all-zero (missing data): %s", zero_cols)


if __name__ == "__main__":
    main()
