"""
Stage 1 feature engineering: compute county-level voter registration features.

Takes the raw FL voter registration snapshots from fetch_voter_registration.py and
produces a clean feature DataFrame capturing both point-in-time party composition
and inter-cycle registration change — two distinct types of political signal.

**Point-in-time features** (most recent available snapshot):
  dem_share        : registered Democrats / total registered (fraction)
  rep_share        : registered Republicans / total registered (fraction)
  npa_share        : registered NPA / total registered (fraction)
  other_share      : minor-party registered / total registered (fraction)
  rep_minus_dem    : rep_share - dem_share (GOP registration advantage)

**Change features** (2016→2024 where available, else largest available window):
  dem_share_change    : dem_share(latest) - dem_share(earliest)
  rep_share_change    : rep_share(latest) - rep_share(earliest)
  npa_share_change    : npa_share(latest) - npa_share(earliest)
  registration_growth : (total_latest - total_earliest) / total_earliest

Positive dem_share_change means Democrats gained registration share over the window.
Positive rep_share_change means Republicans gained registration share.
NPA growth typically signals disengagement from both major parties.

**Why registration matters**:
Registration composition is a leading indicator of electoral outcomes. NPA growth
in particular tracks with political volatility: counties where NPA surged (like
The Villages suburbs) swung Republican; historically Democratic counties with
NPA growth often shifted toward Republicans. Registration change is distinct from
vote-share change — it reflects underlying partisan identity trends, not just
turnout or candidate effects.

**Scope**:
Florida only. GA and AL do not publish equivalent county-by-party registration
data in machine-readable format.

Input:  data/raw/fl_voter_registration.parquet
Output: data/assembled/county_voter_registration_features.parquet
  One row per county_fips (67 FL counties).
  Columns: county_fips, county_name, + all feature columns listed above.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "fl_voter_registration.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assembled" / "county_voter_registration_features.parquet"

# Feature column names for point-in-time (share) features
SHARE_FEATURE_COLS = [
    "dem_share",
    "rep_share",
    "npa_share",
    "other_share",
    "rep_minus_dem",
]

# Feature column names for inter-cycle change features
CHANGE_FEATURE_COLS = [
    "dem_share_change",
    "rep_share_change",
    "npa_share_change",
    "registration_growth",
]

# All voter registration feature columns
VOTER_REG_FEATURE_COLS = SHARE_FEATURE_COLS + CHANGE_FEATURE_COLS


def compute_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute party share features for each county-year observation.

    Adds dem_share, rep_share, npa_share, other_share, rep_minus_dem as new
    columns. Counties with total == 0 or NaN produce NaN shares.

    Args:
        df: DataFrame from data/raw/fl_voter_registration.parquet with columns:
            county_fips, county_name, election_year, book_closing_date,
            rep, dem, npa, other, total

    Returns:
        Input DataFrame with 5 share columns appended.
    """
    df = df.copy()

    total = pd.to_numeric(df["total"], errors="coerce")
    # Guard: replace 0 total with NaN to avoid division by zero
    total_safe = total.where(total > 0, other=float("nan"))

    df["dem_share"] = pd.to_numeric(df["dem"], errors="coerce") / total_safe
    df["rep_share"] = pd.to_numeric(df["rep"], errors="coerce") / total_safe
    df["npa_share"] = pd.to_numeric(df["npa"], errors="coerce") / total_safe
    df["other_share"] = pd.to_numeric(df["other"], errors="coerce") / total_safe
    df["rep_minus_dem"] = df["rep_share"] - df["dem_share"]

    return df


def compute_change_features(df_shares: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-cycle registration change features per county.

    For each county, computes the change in party shares and total registration
    between the earliest and latest available election cycles. This produces
    one row per county with the change over the full available window.

    Change is defined as latest_value - earliest_value. Counties with only one
    observation get NaN change features (cannot compute change from a single point).

    Args:
        df_shares: DataFrame from compute_share_features(), with columns including
            county_fips, county_name, election_year, dem_share, rep_share,
            npa_share, total.

    Returns:
        DataFrame with one row per county containing:
            county_fips, county_name, dem_share_change, rep_share_change,
            npa_share_change, registration_growth
        Counties without enough observations to compute change have NaN values.
    """
    result_rows: list[dict] = []

    for fips, grp in df_shares.groupby("county_fips"):
        grp_sorted = grp.sort_values("election_year")
        county_name = grp_sorted["county_name"].iloc[-1]

        row: dict = {"county_fips": fips, "county_name": county_name}

        if len(grp_sorted) < 2:
            # Cannot compute change with a single snapshot
            row["dem_share_change"] = float("nan")
            row["rep_share_change"] = float("nan")
            row["npa_share_change"] = float("nan")
            row["registration_growth"] = float("nan")
        else:
            earliest = grp_sorted.iloc[0]
            latest = grp_sorted.iloc[-1]

            row["dem_share_change"] = _safe_diff(latest["dem_share"], earliest["dem_share"])
            row["rep_share_change"] = _safe_diff(latest["rep_share"], earliest["rep_share"])
            row["npa_share_change"] = _safe_diff(latest["npa_share"], earliest["npa_share"])

            # Registration growth = (total_latest - total_earliest) / total_earliest
            total_early = earliest["total"]
            total_late = latest["total"]
            if pd.notna(total_early) and pd.notna(total_late) and total_early > 0:
                row["registration_growth"] = (total_late - total_early) / total_early
            else:
                row["registration_growth"] = float("nan")

        result_rows.append(row)

    if not result_rows:
        return pd.DataFrame(
            columns=["county_fips", "county_name"] + CHANGE_FEATURE_COLS
        )

    return pd.DataFrame(result_rows).reset_index(drop=True)


def _safe_diff(a: float, b: float) -> float:
    """Return a - b, or NaN if either operand is NaN."""
    if pd.isna(a) or pd.isna(b):
        return float("nan")
    return float(a) - float(b)


def build_county_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the final county-level feature table from raw registration snapshots.

    Combines point-in-time features (from the latest available snapshot) with
    inter-cycle change features. Output has one row per county.

    Pipeline:
    1. Compute share features for every county × year row
    2. Extract latest snapshot per county for point-in-time features
    3. Compute change features across all snapshots
    4. Join on county_fips

    Args:
        df: Raw DataFrame from data/raw/fl_voter_registration.parquet.

    Returns:
        Feature DataFrame with columns:
            county_fips, county_name, [SHARE_FEATURE_COLS], [CHANGE_FEATURE_COLS]
        One row per county.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["county_fips", "county_name"] + VOTER_REG_FEATURE_COLS
        )

    # Step 1: compute share features for all year-county rows
    df_shares = compute_share_features(df)

    # Step 2: latest snapshot per county (for point-in-time share features)
    latest_idx = df_shares.groupby("county_fips")["election_year"].idxmax()
    df_latest = df_shares.loc[latest_idx].reset_index(drop=True)

    # Step 3: change features across all snapshots
    df_change = compute_change_features(df_shares)

    # Step 4: join latest-snapshot features with change features
    share_cols = ["county_fips", "county_name"] + SHARE_FEATURE_COLS
    result = df_latest[share_cols].merge(
        df_change[["county_fips"] + CHANGE_FEATURE_COLS],
        on="county_fips",
        how="left",
    )

    log.info(
        "Built features for %d counties | share features: %d | change features: %d",
        len(result),
        sum(result[SHARE_FEATURE_COLS].notna().all(axis=0)),
        sum(result[CHANGE_FEATURE_COLS].notna().any(axis=0)),
    )

    return result.reset_index(drop=True)


def main() -> None:
    """Compute voter registration features from raw FL DOS data and save to parquet.

    Reads data/raw/fl_voter_registration.parquet (produced by
    fetch_voter_registration.py), computes 9 features per county, and saves to
    data/assembled/county_voter_registration_features.parquet.
    """
    if not INPUT_PATH.exists():
        log.error(
            "FL voter registration data not found at %s.\n"
            "Run: uv run python src/assembly/fetch_voter_registration.py",
            INPUT_PATH,
        )
        return

    log.info("Loading FL voter registration data from %s...", INPUT_PATH)
    raw = pd.read_parquet(INPUT_PATH)
    log.info("Loaded: %d rows × %d cols", len(raw), len(raw.columns))

    # Year coverage summary
    years = sorted(raw["election_year"].unique())
    log.info("Election years in data: %s", years)

    # Build features
    features = build_county_features(raw)

    # NaN audit
    log.info("\nNaN counts per feature column:")
    for col in VOTER_REG_FEATURE_COLS:
        if col in features.columns:
            n_nan = features[col].isna().sum()
            if n_nan > 0:
                pct = 100 * n_nan / max(len(features), 1)
                log.info("  %-30s  %d NaN (%.1f%%)", col, n_nan, pct)

    # Summary
    n_counties = len(features)
    log.info("\nSummary: %d FL counties with voter registration features", n_counties)

    # Feature distribution summary
    log.info("\nShare features (latest snapshot):")
    for col in SHARE_FEATURE_COLS:
        if col in features.columns:
            valid = features[col].dropna()
            if len(valid) > 0:
                log.info(
                    "  %-28s  min=%.3f  median=%.3f  max=%.3f",
                    col, valid.min(), valid.median(), valid.max(),
                )

    log.info("\nChange features (earliest→latest cycle):")
    for col in CHANGE_FEATURE_COLS:
        if col in features.columns:
            valid = features[col].dropna()
            if len(valid) > 0:
                log.info(
                    "  %-28s  min=%.3f  median=%.3f  max=%.3f",
                    col, valid.min(), valid.median(), valid.max(),
                )

    # Final column order
    out_cols = ["county_fips", "county_name"] + VOTER_REG_FEATURE_COLS
    output = features[[c for c in out_cols if c in features.columns]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "\nSaved → %s  (%d rows × %d cols)",
        OUTPUT_PATH, len(output), len(output.columns),
    )


if __name__ == "__main__":
    main()
