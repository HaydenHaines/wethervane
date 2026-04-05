#!/usr/bin/env python3
"""Regenerate type narratives with political lean + trend and update wethervane.duckdb.

Reads prediction and shift data already in the DB, generates new narratives
via the template engine, then updates the narrative column in-place.

Usage:
    uv run python scripts/regenerate_narratives.py [--db PATH]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from src.description.generate_narratives import generate_all_narratives

log = logging.getLogger(__name__)


def _load_predictions(con: duckdb.DuckDBPyConnection) -> dict[int, float]:
    """Return mean predicted Democratic share per type from the predictions table."""
    rows = con.execute(
        """
        SELECT cta.dominant_type AS type_id,
               AVG(p.pred_dem_share) AS mean_pred
        FROM county_type_assignments cta
        JOIN predictions p ON cta.county_fips = p.county_fips
        GROUP BY cta.dominant_type
        """
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}


def _load_shifts(con: duckdb.DuckDBPyConnection) -> dict[int, dict[str, float]]:
    """Return per-type mean shift profiles (2012-16, 2016-20, 2020-24)."""
    # Check which shift columns are available before querying
    available = {
        c for c in con.execute("SELECT * FROM county_shifts LIMIT 0").fetchdf().columns
    }

    def _col(c: str) -> str:
        return f"AVG(cs.{c})" if c in available else "NULL"

    rows = con.execute(
        f"""
        SELECT
            cta.dominant_type                  AS type_id,
            {_col('pres_d_shift_12_16')}       AS shift_12_16,
            {_col('pres_d_shift_16_20')}       AS shift_16_20,
            {_col('pres_d_shift_20_24')}       AS shift_20_24
        FROM county_type_assignments cta
        JOIN county_shifts cs ON cta.county_fips = cs.county_fips
        GROUP BY cta.dominant_type
        """
    ).fetchall()

    result: dict[int, dict[str, float]] = {}
    for r in rows:
        type_id = int(r[0])
        result[type_id] = {
            "shift_12_16": float(r[1]) if r[1] is not None else None,
            "shift_16_20": float(r[2]) if r[2] is not None else None,
            "shift_20_24": float(r[3]) if r[3] is not None else None,
        }
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--db",
        default=str(root / "data" / "wethervane.duckdb"),
        help="Path to wethervane.duckdb",
    )
    parser.add_argument(
        "--profiles",
        default=str(root / "data" / "communities" / "type_profiles.parquet"),
        help="Path to type_profiles.parquet",
    )
    args = parser.parse_args()

    log.info("Opening DB: %s", args.db)
    con = duckdb.connect(args.db)

    log.info("Loading prediction data from DB")
    predictions = _load_predictions(con)
    log.info("Loaded predictions for %d types", len(predictions))

    log.info("Loading shift data from DB")
    shifts = _load_shifts(con)
    log.info("Loaded shifts for %d types", len(shifts))

    log.info("Generating narratives from %s", args.profiles)
    narratives = generate_all_narratives(
        profiles_path=args.profiles,
        predictions=predictions,
        shifts=shifts,
    )
    log.info("Generated %d narratives", len(narratives))

    log.info("Updating narrative column in types table")
    updated = 0
    for type_id, narrative in narratives.items():
        con.execute(
            "UPDATE types SET narrative = ? WHERE type_id = ?",
            [narrative, type_id],
        )
        updated += 1

    log.info("Updated %d type narratives in DB", updated)
    con.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
