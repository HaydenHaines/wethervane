"""Validation and reporting for wethervane.duckdb.

Contains the API-frontend contract validator, referential integrity
checks, and the row-count summary reporter.
"""
from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)


def validate_contract(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Validate DuckDB matches the API-frontend contract.

    Returns a list of violation strings. Empty list = pass.
    See docs/superpowers/specs/2026-03-21-api-frontend-contract-design.md
    """
    errors: list[str] = []

    required = {
        "super_types": ["super_type_id", "display_name"],
        "types": ["type_id", "super_type_id", "display_name"],
        "county_type_assignments": ["county_fips", "dominant_type", "super_type"],
        "tract_type_assignments": ["tract_geoid", "dominant_type", "super_type"],
        "counties": ["county_fips", "state_abbr", "county_name", "total_votes_2024"],
        "type_scores": ["county_fips", "type_id", "score"],
        "type_priors": ["type_id", "mean_dem_share"],
        "polls": ["poll_id", "race", "geography", "dem_share"],
        # poll_crosstabs is required so the crosstab-adjusted W pipeline can
        # always query it — it will simply be empty when no crosstab data exists.
        "poll_crosstabs": ["poll_id", "demographic_group", "group_value", "pct_of_sample"],
    }

    optional = {
        "predictions": ["county_fips", "race", "pred_dem_share"],
        "tract_predictions": ["tract_geoid", "race", "forecast_mode", "pred_dem_share"],
    }

    def _check_table(table: str, columns: list[str], is_required: bool) -> None:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            if is_required:
                errors.append(f"MISSING TABLE: {table}")
            return
        actual_cols = set(con.execute(f'SELECT * FROM "{table}" LIMIT 0').fetchdf().columns)
        for col in columns:
            if col not in actual_cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    for table, columns in required.items():
        _check_table(table, columns, is_required=True)
    for table, columns in optional.items():
        _check_table(table, columns, is_required=False)

    # Referential integrity (only if required tables exist)
    if not any("MISSING TABLE" in e for e in errors):
        orphans = con.execute("""
            SELECT DISTINCT cta.super_type
            FROM county_type_assignments cta
            LEFT JOIN super_types st ON cta.super_type = st.super_type_id
            WHERE st.super_type_id IS NULL AND cta.super_type IS NOT NULL
        """).fetchdf()
        if not orphans.empty:
            ids = orphans["super_type"].tolist()
            errors.append(f"ORPHAN super_type values in county_type_assignments: {ids}")

        orphan_types = con.execute("""
            SELECT DISTINCT cta.dominant_type
            FROM county_type_assignments cta
            LEFT JOIN types t ON cta.dominant_type = t.type_id
            WHERE t.type_id IS NULL AND cta.dominant_type IS NOT NULL
        """).fetchdf()
        if not orphan_types.empty:
            ids = orphan_types["dominant_type"].tolist()
            errors.append(f"ORPHAN dominant_type values in county_type_assignments: {ids}")

    return errors


def report_summary(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Print a row-count summary for every table and the model versions registry."""
    log.info("Database build complete: %s", db_path)
    print("\n=== wethervane.duckdb summary ===")
    for table in [
        "counties", "model_versions", "community_assignments", "type_assignments",
        "county_shifts", "predictions", "community_sigma", "community_profiles",
        "county_demographics", "types", "county_type_assignments", "tract_type_assignments",
        "tract_predictions",
        "super_types", "type_covariance", "demographics_interpolated",
        "type_scores", "type_priors", "ridge_county_priors", "polls", "poll_notes",
        "races",
    ]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR -- {e}")

    print("\n=== Model Versions ===")
    rows = con.execute(
        "SELECT version_id, role, k, shift_type, vote_share_type, holdout_r FROM model_versions ORDER BY version_id"
    ).fetchall()
    print(f"  {'version_id':<45}  {'role':<20}  {'k':>4}  {'shift_type':<10}  {'holdout_r'}")
    for row in rows:
        vid, role, k, st, vst, hr = row
        print(f"  {str(vid):<45}  {str(role):<20}  {str(k):>4}  {str(st):<10}  {str(hr)}")


def validate_integrity(con: duckdb.DuckDBPyConnection) -> None:
    """Run contract validation and exit with status 1 on any violation."""
    errors = validate_contract(con)
    if errors:
        for e in errors:
            log.error("CONTRACT VIOLATION: %s", e)
        del con
        gc.collect()
        sys.exit(1)
    log.info("Contract validation passed")
