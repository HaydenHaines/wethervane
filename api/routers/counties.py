from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.db import get_db
from api.models import CountyDetail, CountyRow, ElectionHistoryPoint, SiblingCounty

# Parquet files live under $WETHERVANE_DATA_DIR/assembled/ (default: data/assembled/)
_DATA_DIR = Path(os.environ.get("WETHERVANE_DATA_DIR", "data"))
_ASSEMBLED = _DATA_DIR / "assembled"

# Presidential election years available as parquet files
_PRES_YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
# Senate election years available
_SEN_YEARS = [2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022]
# Governor election years available
_GOV_YEARS_ALGARA = [1994, 1998, 2002, 2006, 2010, 2014, 2018]
_GOV_YEARS_MEDSL = [2022]

router = APIRouter(tags=["counties"])


def _has_table(db: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result is not None and result[0] > 0


@router.get("/counties", response_model=list[CountyRow])
def list_counties(request: Request, db: duckdb.DuckDBPyConnection = Depends(get_db)):
    version_id = request.app.state.version_id

    has_types = _has_table(db, "county_type_assignments")
    has_communities = _has_table(db, "community_assignments")

    has_predictions = _has_table(db, "predictions")

    if has_types:
        if has_predictions:
            rows = db.execute(
                """
                SELECT c.county_fips, c.state_abbr,
                       ca.community_id,
                       cta.dominant_type, cta.super_type,
                       AVG(p.pred_dem_share) AS pred_dem_share
                FROM counties c
                JOIN county_type_assignments cta
                    ON c.county_fips = cta.county_fips
                LEFT JOIN community_assignments ca
                    ON c.county_fips = ca.county_fips
                    AND ca.version_id = ?
                LEFT JOIN predictions p
                    ON c.county_fips = p.county_fips
                GROUP BY c.county_fips, c.state_abbr, ca.community_id,
                         cta.dominant_type, cta.super_type
                ORDER BY c.county_fips
                """,
                [version_id],
            ).fetchdf()
        else:
            rows = db.execute(
                """
                SELECT c.county_fips, c.state_abbr,
                       ca.community_id,
                       cta.dominant_type, cta.super_type
                FROM counties c
                JOIN county_type_assignments cta
                    ON c.county_fips = cta.county_fips
                LEFT JOIN community_assignments ca
                    ON c.county_fips = ca.county_fips
                    AND ca.version_id = ?
                ORDER BY c.county_fips
                """,
                [version_id],
            ).fetchdf()
    elif has_communities:
        rows = db.execute(
            """
            SELECT c.county_fips, c.state_abbr, ca.community_id
            FROM counties c
            JOIN community_assignments ca
                ON c.county_fips = ca.county_fips
                AND ca.version_id = ?
            ORDER BY c.county_fips
            """,
            [version_id],
        ).fetchdf()
    else:
        rows = db.execute(
            """
            SELECT county_fips, state_abbr
            FROM counties
            ORDER BY county_fips
            """,
        ).fetchdf()

    results = []
    for _, row in rows.iterrows():
        community_id = None
        dominant_type = None
        super_type = None
        pred_dem_share = None
        if "community_id" in row.index and not pd.isna(row["community_id"]):
            community_id = int(row["community_id"])
        if "dominant_type" in row.index and not pd.isna(row["dominant_type"]):
            dominant_type = int(row["dominant_type"])
        if "super_type" in row.index and not pd.isna(row["super_type"]):
            super_type = int(row["super_type"])
        if "pred_dem_share" in row.index and not pd.isna(row["pred_dem_share"]):
            pred_dem_share = float(row["pred_dem_share"])
        results.append(
            CountyRow(
                county_fips=row["county_fips"],
                state_abbr=row["state_abbr"],
                community_id=community_id,
                dominant_type=dominant_type,
                super_type=super_type,
                pred_dem_share=pred_dem_share,
            )
        )
    return results


@router.get("/counties/{fips}", response_model=CountyDetail)
def get_county_detail(
    fips: str,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
):
    """Return detailed profile for a single county (SEO county page)."""
    # ── Core county + type info ───────────────────────────────────────────
    row = db.execute(
        """
        SELECT c.county_fips, c.county_name, c.state_abbr,
               cta.dominant_type, cta.super_type,
               t.display_name  AS type_display_name,
               st.display_name AS super_type_display_name,
               t.narrative
        FROM counties c
        JOIN county_type_assignments cta ON c.county_fips = cta.county_fips
        JOIN types t ON cta.dominant_type = t.type_id
        JOIN super_types st ON t.super_type_id = st.super_type_id
        WHERE c.county_fips = ?
        LIMIT 1
        """,
        [fips],
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"County {fips} not found")

    (
        county_fips, county_name, state_abbr,
        dominant_type, super_type,
        type_display_name, super_type_display_name,
        narrative,
    ) = row

    # ── Baseline prediction (AVG across races) ────────────────────────────
    pred_row = db.execute(
        "SELECT AVG(pred_dem_share) FROM predictions WHERE county_fips = ?",
        [fips],
    ).fetchone()
    pred_dem_share = float(pred_row[0]) if pred_row and pred_row[0] is not None else None

    # ── Demographics ──────────────────────────────────────────────────────
    demo_row = db.execute(
        "SELECT * FROM county_demographics WHERE county_fips = ?",
        [fips],
    ).fetchone()
    demographics: dict[str, float] = {}
    if demo_row is not None:
        demo_cols = [
            desc[0]
            for desc in db.execute("DESCRIBE county_demographics").fetchall()
        ]
        for col, val in zip(demo_cols, demo_row):
            if col == "county_fips":
                continue
            if val is not None:
                demographics[col] = float(val)

    # ── Sibling counties (same dominant_type, limit 20) ───────────────────
    siblings = db.execute(
        """
        SELECT c.county_fips, c.county_name, c.state_abbr
        FROM county_type_assignments cta
        JOIN counties c ON cta.county_fips = c.county_fips
        WHERE cta.dominant_type = ? AND cta.county_fips != ?
        ORDER BY c.state_abbr, c.county_name
        LIMIT 20
        """,
        [dominant_type, fips],
    ).fetchall()
    sibling_counties = [
        SiblingCounty(county_fips=s[0], county_name=s[1], state_abbr=s[2])
        for s in siblings
    ]

    return CountyDetail(
        county_fips=county_fips,
        county_name=county_name,
        state_abbr=state_abbr,
        dominant_type=int(dominant_type),
        super_type=int(super_type),
        type_display_name=type_display_name,
        super_type_display_name=super_type_display_name,
        narrative=narrative,
        pred_dem_share=pred_dem_share,
        demographics=demographics,
        sibling_counties=sibling_counties,
    )


def _read_share(path: Path, fips: str, col: str, total_col: str) -> tuple[float, int | None] | None:
    """Read a single county's dem_share and total_votes from a parquet file.

    Returns (dem_share, total_votes) or None if county not found or value is NaN.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["county_fips", col, total_col])
    except Exception:
        # Column may not exist (e.g. old schema)
        try:
            df = pd.read_parquet(path, columns=["county_fips", col])
            df[total_col] = None
        except Exception:
            return None
    row = df[df["county_fips"] == fips]
    if row.empty:
        return None
    val = row[col].values[0]
    if pd.isna(val):
        return None
    total = row[total_col].values[0] if total_col in row.columns else None
    total_int: int | None = None if (total is None or pd.isna(total)) else int(total)
    return float(val), total_int


@router.get("/counties/{fips}/history", response_model=list[ElectionHistoryPoint])
def get_county_history(fips: str):
    """Return raw election results (Dem two-party share) for a county across all available cycles.

    Covers presidential (2000-2024), Senate (2002-2022), and gubernatorial (1994-2022) elections
    where county-level parquet data is available.  Years with no data for this county are omitted.
    Results are sorted by year then election_type.
    """
    points: list[ElectionHistoryPoint] = []

    # Presidential
    for year in _PRES_YEARS:
        path = _ASSEMBLED / f"medsl_county_presidential_{year}.parquet"
        result = _read_share(path, fips, f"pres_dem_share_{year}", f"pres_total_{year}")
        if result is not None:
            dem_share, total_votes = result
            points.append(ElectionHistoryPoint(
                year=year,
                election_type="president",
                dem_share=dem_share,
                total_votes=total_votes,
            ))

    # Senate
    for year in _SEN_YEARS:
        path = _ASSEMBLED / f"medsl_county_senate_{year}.parquet"
        result = _read_share(path, fips, f"senate_dem_share_{year}", f"senate_total_{year}")
        if result is not None:
            dem_share, total_votes = result
            points.append(ElectionHistoryPoint(
                year=year,
                election_type="senate",
                dem_share=dem_share,
                total_votes=total_votes,
            ))

    # Governor — Algara & Amlani dataset
    for year in _GOV_YEARS_ALGARA:
        path = _ASSEMBLED / f"algara_county_governor_{year}.parquet"
        result = _read_share(path, fips, f"gov_dem_share_{year}", f"gov_total_{year}")
        if result is not None:
            dem_share, total_votes = result
            points.append(ElectionHistoryPoint(
                year=year,
                election_type="governor",
                dem_share=dem_share,
                total_votes=total_votes,
            ))

    # Governor — MEDSL 2022
    for year in _GOV_YEARS_MEDSL:
        path = _ASSEMBLED / f"medsl_county_2022_governor.parquet"
        result = _read_share(path, fips, f"gov_dem_share_{year}", f"gov_total_{year}")
        if result is not None:
            dem_share, total_votes = result
            points.append(ElectionHistoryPoint(
                year=year,
                election_type="governor",
                dem_share=dem_share,
                total_votes=total_votes,
            ))

    # Sort by year, then election_type for consistent ordering
    points.sort(key=lambda p: (p.year, p.election_type))
    return points
