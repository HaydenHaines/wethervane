"""Integration test: full county pipeline on a synthetic mini-dataset.

Exercises the shift computation → DuckDB ingestion path end-to-end using
synthetic data for a small set of fictional counties. Validates:
  - Output shape: 293-ish counties × correct shift dimensions
  - No NaN in shift columns after assembly
  - Log-odds shifts are finite (epsilon clipping works)
  - DuckDB is queryable and returns the expected row counts
  - community_assignments FK integrity against counties table

This test does NOT re-run the data fetchers (no network calls). It creates
minimal synthetic parquets, calls the core assembly + DB builder functions
directly, and checks invariants.

Run with:
    pytest tests/integration/ -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _make_synthetic_election_df(
    fips_list: list[str],
    dem_share_mean: float = 0.45,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a synthetic assembled election parquet for one election year.

    Returns a DataFrame with columns:
        county_fips, state_abbr, *_dem_share_YYYY, *_total_YYYY
    """
    if rng is None:
        rng = np.random.default_rng(42)
    shares = rng.uniform(0.2, 0.8, size=len(fips_list))
    totals = rng.integers(1_000, 100_000, size=len(fips_list)).astype(float)
    return pd.DataFrame({
        "county_fips": fips_list,
        "dem_share": shares,
        "total_votes": totals,
    })


def _build_synthetic_shifts(fips_list: list[str]) -> pd.DataFrame:
    """Build a synthetic shift DataFrame mimicking county_shifts_multiyear.parquet."""
    rng = np.random.default_rng(0)
    n = len(fips_list)

    # Use _logodds_shift logic directly to generate plausible shifts
    from src.assembly.build_county_shifts_multiyear import _logodds_shift, TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS

    rows = {"county_fips": fips_list}

    # Training cols: 30 dims (10 election pairs × 3)
    for col in TRAINING_SHIFT_COLS:
        if "turnout" in col:
            rows[col] = rng.normal(0.02, 0.05, n)
        else:
            # Simulate a log-odds shift by applying _logodds_shift to random shares
            earlier = pd.Series(rng.uniform(0.2, 0.8, n))
            later = pd.Series((earlier + rng.normal(0.0, 0.05, n)).clip(0.01, 0.99))
            rows[col] = _logodds_shift(later, earlier).values

    # Holdout cols: 3 dims
    for col in HOLDOUT_SHIFT_COLS:
        if "turnout" in col:
            rows[col] = rng.normal(0.01, 0.03, n)
        else:
            earlier = pd.Series(rng.uniform(0.2, 0.8, n))
            later = pd.Series((earlier + rng.normal(-0.02, 0.05, n)).clip(0.01, 0.99))
            rows[col] = _logodds_shift(later, earlier).values

    return pd.DataFrame(rows)


def _build_synthetic_assignments(fips_list: list[str], k: int = 5) -> pd.DataFrame:
    """Assign counties to k communities in round-robin fashion."""
    rng = np.random.default_rng(1)
    community_ids = (rng.integers(0, k, len(fips_list))).astype(int)
    return pd.DataFrame({
        "county_fips": fips_list,
        "community_id": community_ids,
    })


def _build_synthetic_predictions(fips_list: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = len(fips_list)
    shares = rng.uniform(0.35, 0.65, n)
    return pd.DataFrame({
        "county_fips": fips_list,
        "state_abbr": ["FL"] * n,
        "race": ["FL_Senate"] * n,
        "pred_dem_share": shares,
        "pred_std": rng.uniform(0.02, 0.06, n),
        "pred_lo90": shares - 0.08,
        "pred_hi90": shares + 0.08,
        "state_pred": [0.44] * n,
        "poll_avg": [0.46] * n,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_fips() -> list[str]:
    """100 synthetic county FIPS codes across FL/GA/AL."""
    fl = [f"12{str(i).zfill(3)}" for i in range(1, 67, 2)][:34]  # ~34 FL counties
    ga = [f"13{str(i).zfill(3)}" for i in range(1, 67, 2)][:34]  # ~34 GA counties
    al = [f"01{str(i).zfill(3)}" for i in range(1, 69, 2)][:32]  # ~32 AL counties
    return fl + ga + al


@pytest.fixture(scope="module")
def synthetic_db(tmp_path_factory, synthetic_fips):
    """Build a full synthetic DuckDB from scratch."""
    tmp = tmp_path_factory.mktemp("integration")
    import src.db.build_database as mod

    # Build synthetic parquets
    shifts = _build_synthetic_shifts(synthetic_fips)
    shifts_path = tmp / "shifts.parquet"
    shifts.to_parquet(shifts_path, index=False)

    assignments = _build_synthetic_assignments(synthetic_fips)
    assignments_path = tmp / "assignments.parquet"
    assignments.to_parquet(assignments_path, index=False)

    preds = _build_synthetic_predictions(synthetic_fips)
    preds_path = tmp / "predictions.parquet"
    preds.to_parquet(preds_path, index=False)

    # Build version meta
    ver_dir = tmp / "versions" / "synthetic_v1"
    ver_dir.mkdir(parents=True)
    with open(ver_dir / "meta.yaml", "w") as f:
        yaml.dump({
            "version": "synthetic_v1",
            "role": "current",
            "k": 5,
            "shift_type": "logodds",
            "vote_share_type": "total",
            "training_dims": 30,
            "holdout_dims": 3,
            "geography": "synthetic",
            "description": "Synthetic integration test",
            "date_created": "2026-01-01",
        }, f)

    # Patch paths and build
    original = {
        "SHIFTS_MULTIYEAR": mod.SHIFTS_MULTIYEAR,
        "COUNTY_ASSIGNMENTS": mod.COUNTY_ASSIGNMENTS,
        "PREDICTIONS_2026": mod.PREDICTIONS_2026,
        "TYPE_ASSIGNMENTS_STUB": mod.TYPE_ASSIGNMENTS_STUB,
        "VERSIONS_DIR": mod.VERSIONS_DIR,
    }
    mod.SHIFTS_MULTIYEAR = shifts_path
    mod.COUNTY_ASSIGNMENTS = assignments_path
    mod.PREDICTIONS_2026 = preds_path
    mod.TYPE_ASSIGNMENTS_STUB = tmp / "nonexistent.parquet"
    mod.VERSIONS_DIR = tmp / "versions"

    db_path = tmp / "synthetic_bedrock.duckdb"
    try:
        mod.build(db_path=db_path, reset=True)
    finally:
        # Restore originals
        for k, v in original.items():
            setattr(mod, k, v)

    return db_path, shifts, assignments


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shift_output_shape(synthetic_fips):
    """Synthetic shift DataFrame has correct column count."""
    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS
    shifts = _build_synthetic_shifts(synthetic_fips)
    expected_cols = 1 + len(TRAINING_SHIFT_COLS) + len(HOLDOUT_SHIFT_COLS)  # +1 for county_fips
    assert shifts.shape == (len(synthetic_fips), expected_cols)


def test_shifts_no_nan(synthetic_fips):
    """No NaN values in the synthetic shift DataFrame."""
    shifts = _build_synthetic_shifts(synthetic_fips)
    assert not shifts.isnull().any().any(), "NaN values found in synthetic shifts"


def test_shifts_finite(synthetic_fips):
    """All shift values are finite (no ±inf from logodds without epsilon clipping)."""
    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS
    shifts = _build_synthetic_shifts(synthetic_fips)
    shift_cols = TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS
    for col in shift_cols:
        if "turnout" not in col:
            assert np.isfinite(shifts[col].values).all(), f"Non-finite values in {col}"


def test_logodds_epsilon_clipping():
    """_logodds_shift clips extreme values to avoid ±inf."""
    from src.assembly.build_county_shifts_multiyear import _logodds_shift
    # Unclipped 0.0 or 1.0 would produce -inf/+inf
    extreme_shares = pd.Series([0.0, 0.001, 0.5, 0.999, 1.0])
    base_shares = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
    result = _logodds_shift(extreme_shares, base_shares)
    assert np.isfinite(result.values).all(), "Epsilon clipping failed: got ±inf"


def test_db_has_all_tables(synthetic_db):
    """All required tables exist and are non-empty."""
    db_path, _, _ = synthetic_db
    con = duckdb.connect(str(db_path))
    for table in ["counties", "model_versions", "community_assignments", "type_assignments", "county_shifts", "predictions"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert count > 0, f"Table '{table}' is empty"
    con.close()


def test_db_counties_match_fips(synthetic_db, synthetic_fips):
    """DuckDB counties table has exactly the expected FIPS codes."""
    db_path, _, _ = synthetic_db
    con = duckdb.connect(str(db_path))
    stored = set(row[0] for row in con.execute("SELECT county_fips FROM counties").fetchall())
    expected = set(synthetic_fips)
    assert stored == expected
    con.close()


def test_db_community_assignments_fk(synthetic_db, synthetic_fips):
    """Every county in community_assignments exists in the counties table."""
    db_path, _, _ = synthetic_db
    con = duckdb.connect(str(db_path))
    orphans = con.execute("""
        SELECT ca.county_fips
        FROM community_assignments ca
        LEFT JOIN counties c ON ca.county_fips = c.county_fips
        WHERE c.county_fips IS NULL
    """).fetchall()
    assert len(orphans) == 0, f"Orphan assignments found: {orphans}"
    con.close()


def test_db_shift_columns_present(synthetic_db):
    """county_shifts table contains all expected training and holdout columns."""
    from src.assembly.build_county_shifts_multiyear import TRAINING_SHIFT_COLS, HOLDOUT_SHIFT_COLS
    db_path, _, _ = synthetic_db
    con = duckdb.connect(str(db_path))
    cols = [row[0] for row in con.execute("DESCRIBE county_shifts").fetchall()]
    for expected_col in TRAINING_SHIFT_COLS + HOLDOUT_SHIFT_COLS:
        assert expected_col in cols, f"Missing shift column: {expected_col}"
    con.close()


def test_db_queryable_by_state(synthetic_db):
    """Can filter counties by state using SQL."""
    db_path, _, _ = synthetic_db
    con = duckdb.connect(str(db_path))
    fl_count = con.execute(
        "SELECT COUNT(*) FROM counties WHERE state_abbr = 'FL'"
    ).fetchone()[0]
    assert fl_count > 0
    con.close()
