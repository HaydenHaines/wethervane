"""Build data/wethervane.duckdb — the central query layer for the WetherVane pipeline.

Reads parquet artifacts from data/shifts/, data/communities/, data/predictions/,
and model version metadata from data/models/versions/*/meta.yaml, then ingests
everything into a single DuckDB file.

Schema
------
counties               one row per county (FIPS, state)
model_versions         version registry with metadata, K, shift_type, holdout_r
community_assignments  Layer 1: county → community_id, keyed by (county_fips, k, version_id)
type_assignments       Layer 2: community → dominant_type_id [stub], keyed by (community_id, k, version_id)
county_shifts          one row per county per version, all shift dims as columns
predictions            2026 forward predictions per (county_fips, race, version_id)

Usage
-----
    python src/db/build_database.py                  # builds from defaults
    python src/db/build_database.py --reset          # drop and rebuild
    python src/db/build_database.py --db path/to/other.duckdb
"""
from __future__ import annotations

import argparse
import gc
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB = PROJECT_ROOT / "data" / "wethervane.duckdb"

# Election cycle to ingest polling data for.
# Update this when targeting a new cycle year.
POLL_INGEST_CYCLE = "2026"

# ── Data source paths ──────────────────────────────────────────────────────────
# All paths are relative to PROJECT_ROOT. These change when the data layout changes.
# If you rename/move a file, update the corresponding constant here.
# Note: build() re-derives these from a caller-supplied project_root to support
# tests that pass a custom root (e.g. tmp_path). The module-level constants
# exist for reference and any external callers.

DATA_DIR = PROJECT_ROOT / "data"

# ── Shift data ─────────────────────────────────────────────────────────────────
SHIFTS_MULTIYEAR = DATA_DIR / "shifts" / "county_shifts_multiyear.parquet"

# ── Community / type assignment data ──────────────────────────────────────────
COMMUNITIES_DIR = DATA_DIR / "communities"
COUNTY_ASSIGNMENTS = COMMUNITIES_DIR / "county_community_assignments.parquet"
TYPE_ASSIGNMENTS_STUB = COMMUNITIES_DIR / "county_type_assignments_stub.parquet"
COMMUNITY_PROFILES_PATH = COMMUNITIES_DIR / "community_profiles.parquet"
TYPE_PROFILES_PATH = COMMUNITIES_DIR / "type_profiles.parquet"
COUNTY_TYPE_ASSIGNMENTS_PATH = COMMUNITIES_DIR / "county_type_assignments_full.parquet"
SUPER_TYPES_PATH = COMMUNITIES_DIR / "super_types.parquet"

# ── Predictions ────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_2026 = PREDICTIONS_DIR / "county_predictions_2026.parquet"
PREDICTIONS_2026_HAC = PREDICTIONS_DIR / "county_predictions_2026_hac.parquet"
PREDICTIONS_2026_TYPES = PREDICTIONS_DIR / "county_predictions_2026_types.parquet"

# ── Covariance ─────────────────────────────────────────────────────────────────
COVARIANCE_DIR = DATA_DIR / "covariance"
SIGMA_PATH = COVARIANCE_DIR / "county_community_sigma.parquet"

# ── Tract assignments ──────────────────────────────────────────────────────────
TRACTS_DIR = DATA_DIR / "tracts"
TRACT_TYPE_ASSIGNMENTS_PATH = TRACTS_DIR / "national_tract_assignments.parquet"

# ── Model version metadata ─────────────────────────────────────────────────────
VERSIONS_DIR = DATA_DIR / "models" / "versions"

# ── Assembled / raw reference data ────────────────────────────────────────────
CROSSWALK_PATH = DATA_DIR / "raw" / "fips_county_crosswalk.csv"
PRES_2024_PATH = DATA_DIR / "assembled" / "medsl_county_presidential_2024.parquet"
COUNTY_ACS_FEATURES_PATH = DATA_DIR / "assembled" / "county_acs_features.parquet"
DEMOGRAPHICS_INTERPOLATED_PATH = DATA_DIR / "assembled" / "demographics_interpolated.parquet"

# ── Domain module imports ──────────────────────────────────────────────────────
# Narrative generator (template-based, no LLM)
from src.description.generate_narratives import generate_all_narratives  # noqa: E402
from src.db.domains.model import ingest as ingest_model, create_tables as model_ddl  # noqa: E402
from src.db.domains.polling import ingest as ingest_polling, create_tables as polling_ddl  # noqa: E402


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS counties (
    county_fips      VARCHAR PRIMARY KEY,
    state_abbr       VARCHAR NOT NULL,
    state_fips       VARCHAR NOT NULL,
    county_name      VARCHAR,
    total_votes_2024 INTEGER  -- 2024 presidential total votes (for population-weighted aggregation)
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id        VARCHAR PRIMARY KEY,
    role              VARCHAR,          -- 'current', 'previous', 'county_baseline', etc.
    k                 INTEGER,          -- community count K (NULL = not yet determined)
    j                 INTEGER,          -- type count J (NULL = not yet determined)
    shift_type        VARCHAR,          -- 'logodds' or 'raw'
    vote_share_type   VARCHAR,          -- 'total' or 'twoparty'
    n_training_dims   INTEGER,
    n_holdout_dims    INTEGER,
    holdout_r         VARCHAR,          -- holdout Pearson r or range (NULL if not yet validated)
    geography         VARCHAR,          -- e.g. 'FL+GA+AL (293 counties)'
    description       VARCHAR,
    created_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS community_assignments (
    county_fips   VARCHAR  NOT NULL,
    community_id  INTEGER  NOT NULL,
    k             INTEGER  NOT NULL,    -- total communities in this model run
    version_id    VARCHAR  NOT NULL,
    PRIMARY KEY (county_fips, k, version_id)
);

CREATE TABLE IF NOT EXISTS type_assignments (
    community_id      INTEGER  NOT NULL,
    k                 INTEGER  NOT NULL,
    dominant_type_id  INTEGER,          -- NULL if stub
    j                 INTEGER,          -- total types
    version_id        VARCHAR  NOT NULL,
    PRIMARY KEY (community_id, k, version_id)
);

CREATE TABLE IF NOT EXISTS county_shifts (
    county_fips  VARCHAR  NOT NULL,
    version_id   VARCHAR  NOT NULL,
    PRIMARY KEY (county_fips, version_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    county_fips    VARCHAR  NOT NULL,
    race           VARCHAR  NOT NULL,
    version_id     VARCHAR  NOT NULL,
    forecast_mode  VARCHAR  NOT NULL DEFAULT 'local',
    pred_dem_share DOUBLE,
    pred_std       DOUBLE,
    pred_lo90      DOUBLE,
    pred_hi90      DOUBLE,
    state_pred     DOUBLE,
    poll_avg       DOUBLE,
    PRIMARY KEY (county_fips, race, version_id, forecast_mode)
);

CREATE TABLE IF NOT EXISTS community_sigma (
    community_id_row  INTEGER NOT NULL,
    community_id_col  INTEGER NOT NULL,
    sigma_value       DOUBLE,
    version_id        VARCHAR NOT NULL,
    PRIMARY KEY (community_id_row, community_id_col, version_id)
);

CREATE TABLE IF NOT EXISTS community_profiles (
    community_id          INTEGER PRIMARY KEY,
    n_counties            INTEGER,
    pop_total             DOUBLE,
    pct_white_nh          DOUBLE,
    pct_black             DOUBLE,
    pct_asian             DOUBLE,
    pct_hispanic          DOUBLE,
    median_age            DOUBLE,
    median_hh_income      DOUBLE,
    pct_bachelors_plus    DOUBLE,
    pct_owner_occupied    DOUBLE,
    pct_wfh               DOUBLE,
    pct_management        DOUBLE,
    evangelical_share     DOUBLE,
    mainline_share        DOUBLE,
    catholic_share        DOUBLE,
    black_protestant_share DOUBLE,
    congregations_per_1000 DOUBLE,
    religious_adherence_rate DOUBLE
);

CREATE TABLE IF NOT EXISTS county_demographics (
    county_fips           VARCHAR PRIMARY KEY,
    pop_total             DOUBLE,
    pct_white_nh          DOUBLE,
    pct_black             DOUBLE,
    pct_asian             DOUBLE,
    pct_hispanic          DOUBLE,
    median_age            DOUBLE,
    median_hh_income      DOUBLE,
    pct_bachelors_plus    DOUBLE,
    pct_owner_occupied    DOUBLE,
    pct_wfh               DOUBLE,
    pct_management        DOUBLE
);

CREATE TABLE IF NOT EXISTS super_types (
    super_type_id  INTEGER PRIMARY KEY,
    display_name   VARCHAR
);

CREATE TABLE IF NOT EXISTS types (
    type_id        INTEGER PRIMARY KEY,
    super_type_id  INTEGER,
    display_name   VARCHAR
);

CREATE TABLE IF NOT EXISTS county_type_assignments (
    county_fips    VARCHAR NOT NULL,
    dominant_type  INTEGER,
    super_type     INTEGER
);

CREATE TABLE IF NOT EXISTS tract_type_assignments (
    tract_geoid    VARCHAR PRIMARY KEY,
    dominant_type  INTEGER,
    super_type     INTEGER
);

CREATE TABLE IF NOT EXISTS races (
    race_id    VARCHAR PRIMARY KEY,
    race_type  VARCHAR NOT NULL,
    state      VARCHAR NOT NULL,
    year       INTEGER NOT NULL,
    district   INTEGER
);
"""

# State FIPS → abbreviation mapping: sourced from config/model.yaml (all 50+DC).
# Crosswalk CSV is the primary source for county names; this dict handles state_abbr
# for any county whose state is in scope.
from src.core import config as _cfg  # noqa: E402
_STATE_FIPS_TO_ABBR: dict[str, str] = _cfg.STATE_ABBR  # fips_prefix → abbr


def _load_version_meta(versions_dir: Path) -> list[dict]:
    """Load all meta.yaml files from versioned model directories."""
    meta_list = []
    if not versions_dir.exists():
        log.warning("Versions dir not found: %s", versions_dir)
        return meta_list
    for version_dir in sorted(versions_dir.iterdir()):
        meta_path = version_dir / "meta.yaml"
        if meta_path.exists():
            with open(meta_path) as f:
                m = yaml.safe_load(f)
            meta_list.append(m)
            vid = m.get("version_id") or m.get("version") or version_dir.name
            log.info("Loaded version meta: %s (%s)", vid, m.get("role"))
    return meta_list


_DEFAULT_CROSSWALK = object()  # sentinel: use module-level CROSSWALK_PATH


def _build_counties(
    shifts: pd.DataFrame,
    crosswalk_path: Path | None = _DEFAULT_CROSSWALK,  # type: ignore[assignment]
    pres_2024_path: Path | None = None,
) -> pd.DataFrame:
    """Derive the counties table from shift FIPS column, optionally joining county names
    and 2024 presidential vote totals (used for population-weighted state aggregation).

    Args:
        shifts: DataFrame with a county_fips column.
        crosswalk_path: Path to fips_county_crosswalk.csv.  Pass ``None`` to
            skip name lookup (county_name will be all-NULL).  Omit (or pass the
            sentinel ``_DEFAULT_CROSSWALK``) to use the module-level
            ``CROSSWALK_PATH`` constant.
        pres_2024_path: Path to medsl_county_presidential_2024.parquet.  When
            provided (and the file exists), ``total_votes_2024`` is populated
            from ``pres_total_2024``.  Falls back to NULL when not available.
    """
    if crosswalk_path is _DEFAULT_CROSSWALK:
        crosswalk_path = CROSSWALK_PATH

    fips = shifts["county_fips"].unique()
    df = pd.DataFrame({"county_fips": sorted(fips)})
    df["state_fips"] = df["county_fips"].str[:2]
    df["state_abbr"] = df["state_fips"].map(_STATE_FIPS_TO_ABBR).fillna("??")

    if crosswalk_path is not None and Path(crosswalk_path).exists():
        xwalk = pd.read_csv(crosswalk_path, dtype=str)
        xwalk["county_fips"] = xwalk["county_fips"].str.zfill(5)
        df = df.merge(xwalk[["county_fips", "county_name"]], on="county_fips", how="left")
    else:
        df["county_name"] = None

    # Join 2024 presidential vote totals for population-weighted state aggregation.
    # total_votes_2024 is NULL when data is unavailable; the API falls back to
    # uniform weighting when the column is missing or all-NULL.
    _pres_path = pres_2024_path if pres_2024_path is not None else PRES_2024_PATH
    if _pres_path is not None and Path(_pres_path).exists():
        pres_df = pd.read_parquet(_pres_path)
        pres_df["county_fips"] = pres_df["county_fips"].astype(str).str.zfill(5)
        df = df.merge(
            pres_df[["county_fips", "pres_total_2024"]].rename(
                columns={"pres_total_2024": "total_votes_2024"}
            ),
            on="county_fips",
            how="left",
        )
        # Cast to nullable int (some counties may be missing from the parquet)
        df["total_votes_2024"] = pd.to_numeric(df["total_votes_2024"], errors="coerce").astype(
            "Int64"
        )
        n_matched = df["total_votes_2024"].notna().sum()
        log.info("Joined total_votes_2024 for %d / %d counties", n_matched, len(df))
    else:
        df["total_votes_2024"] = None
        log.warning(
            "2024 presidential parquet not found at %s; total_votes_2024 will be NULL",
            _pres_path,
        )

    return df[["county_fips", "state_abbr", "state_fips", "county_name", "total_votes_2024"]]


def _build_county_shifts(shifts: pd.DataFrame, version_id: str) -> pd.DataFrame:
    """Add version_id column to the wide shifts DataFrame."""
    df = shifts.copy()
    df["version_id"] = version_id
    return df


def _build_community_assignments(
    assignments: pd.DataFrame, version_id: str
) -> pd.DataFrame:
    """Normalize community assignments for DuckDB ingestion."""
    df = assignments[["county_fips", "community_id"]].copy()
    k = int(df["community_id"].nunique())
    df["k"] = k
    df["version_id"] = version_id
    return df[["county_fips", "community_id", "k", "version_id"]]


def _build_type_assignments(
    type_df: pd.DataFrame | None, assignments: pd.DataFrame, version_id: str
) -> pd.DataFrame:
    """Build type_assignments rows from stub or empty DataFrame."""
    k = int(assignments["community_id"].nunique())
    unique_communities = sorted(assignments["community_id"].unique())

    if type_df is not None and "dominant_type_id" in type_df.columns:
        df = type_df[["community_id", "dominant_type_id"]].copy()
        j = int(type_df.get("j", [None])[0]) if "j" in type_df.columns else None
    else:
        # Stub: one row per community with NULL dominant_type_id
        df = pd.DataFrame({"community_id": unique_communities, "dominant_type_id": None})
        j = None

    df["k"] = k
    df["j"] = j
    df["version_id"] = version_id
    return df[["community_id", "k", "dominant_type_id", "j", "version_id"]]


def _build_races_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create and populate the races table from the race registry."""
    from src.assembly.define_races import load_races

    con.execute("DELETE FROM races")
    races = load_races(2026)
    for r in races:
        con.execute(
            "INSERT INTO races VALUES (?, ?, ?, ?, ?)",
            [r.race_id, r.race_type, r.state, r.year, r.district],
        )
    log.info("Ingested races: %d rows", len(races))


def _build_predictions(preds: pd.DataFrame, version_id: str) -> pd.DataFrame:
    df = preds.copy()
    df["version_id"] = version_id
    cols = [
        "county_fips", "race", "version_id", "forecast_mode",
        "pred_dem_share", "pred_std", "pred_lo90", "pred_hi90",
        "state_pred", "poll_avg",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = "local" if c == "forecast_mode" else None
    return df[cols]


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


def _insert_via_parquet(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    *,
    mode: str = "insert",
) -> None:
    """Insert a DataFrame into a DuckDB table using register/unregister.

    Using ``con.register()`` + ``con.unregister()`` instead of the implicit
    Python-DataFrame bridge (``INSERT INTO t SELECT * FROM df``) avoids the
    heap corruption that accumulates when many large DataFrames are transferred
    through the bridge in a single process run.

    Args:
        con: Active DuckDB connection.
        table: Target table name (must already exist for ``mode='insert'``).
        df: DataFrame to insert.
        mode: ``'insert'`` (INSERT INTO ... SELECT) or ``'create'``
            (CREATE TABLE AS SELECT — drops and recreates the table).
    """
    _VIEW = "_tmp_insert_view"
    con.register(_VIEW, df)
    try:
        if mode == "create":
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM {_VIEW}")
        else:
            con.execute(f"INSERT INTO {table} SELECT * FROM {_VIEW}")
    finally:
        con.unregister(_VIEW)


def _resolve_paths(project_root: Path) -> dict[str, Path]:
    """Derive all data source paths from a project root.

    Centralizes path construction so callers can pass a custom root
    (e.g. tmp_path in tests) without monkeypatching module-level constants.
    Returns a flat dict keyed by a short logical name.
    """
    data = project_root / "data"
    communities = data / "communities"
    predictions = data / "predictions"
    return {
        "shifts": data / "shifts" / "county_shifts_multiyear.parquet",
        "assignments": communities / "county_community_assignments.parquet",
        "stub": communities / "county_type_assignments_stub.parquet",
        "predictions": predictions / "county_predictions_2026.parquet",
        "predictions_hac": predictions / "county_predictions_2026_hac.parquet",
        "predictions_types": predictions / "county_predictions_2026_types.parquet",
        "versions_dir": data / "models" / "versions",
        "crosswalk": data / "raw" / "fips_county_crosswalk.csv",
        "sigma": data / "covariance" / "county_community_sigma.parquet",
        "community_profiles": communities / "community_profiles.parquet",
        "county_acs": data / "assembled" / "county_acs_features.parquet",
        "type_profiles": communities / "type_profiles.parquet",
        "county_type_assignments": communities / "county_type_assignments_full.parquet",
        "tract_type_assignments": data / "tracts" / "national_tract_assignments.parquet",
        "super_types": communities / "super_types.parquet",
        "demographics_interpolated": data / "assembled" / "demographics_interpolated.parquet",
    }


def _reset_database(db_path: Path) -> None:
    """Drop the existing database file so the next connect starts fresh."""
    if db_path.exists():
        db_path.unlink()
        log.info("Dropped existing database: %s", db_path)


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Execute all CREATE TABLE IF NOT EXISTS statements from _SCHEMA_SQL."""
    con.executemany("", [])
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    log.info("Schema created/verified")


def _ingest_data(
    con: duckdb.DuckDBPyConnection,
    db_path: Path,
    paths: dict[str, Path],
    project_root: Path,
) -> None:
    """Load all parquet artifacts and insert them into the open DuckDB connection.

    DuckDB 1.5.x corrupts the glibc malloc heap after many large DataFrame
    inserts. To stay below the threshold, the connection is cycled (del + reconnect)
    at several checkpoints. Each cycle is marked with a log line explaining which
    stage triggered it.
    """
    # ── Load source data ───────────────────────────────────────────────────────
    shifts = None
    if paths["shifts"].exists():
        log.info("Loading shifts from %s", paths["shifts"])
        shifts = pd.read_parquet(paths["shifts"])
        shifts["county_fips"] = shifts["county_fips"].astype(str).str.zfill(5)
    else:
        log.warning("Shifts file not found: %s; skipping core ingestion", paths["shifts"])

    assignments = None
    if paths["assignments"].exists():
        log.info("Loading community assignments from %s", paths["assignments"])
        assignments = pd.read_parquet(paths["assignments"])
        assignments["county_fips"] = assignments["county_fips"].astype(str).str.zfill(5)
        # Prefer community_id column; fall back to 'community' if needed
        if "community_id" not in assignments.columns and "community" in assignments.columns:
            assignments = assignments.rename(columns={"community": "community_id"})
    else:
        log.warning(
            "Assignments file not found: %s; skipping community ingestion",
            paths["assignments"],
        )

    type_df = None
    if paths["stub"].exists():
        log.info("Loading type assignments stub from %s", paths["stub"])
        type_df = pd.read_parquet(paths["stub"])

    predictions = None
    if paths["predictions"].exists():
        log.info("Loading predictions from %s", paths["predictions"])
        predictions = pd.read_parquet(paths["predictions"])
        predictions["county_fips"] = predictions["county_fips"].astype(str).str.zfill(5)

    # ── Load model version metadata ────────────────────────────────────────────
    version_metas = _load_version_meta(paths["versions_dir"])

    # ── Ingest counties ────────────────────────────────────────────────────────
    if shifts is not None:
        counties_df = _build_counties(shifts, crosswalk_path=paths["crosswalk"])
        con.execute("DELETE FROM counties")
        _insert_via_parquet(con, "counties", counties_df)
        log.info("Ingested %d counties", len(counties_df))

    # ── Ingest race registry ──────────────────────────────────────────────────
    _build_races_table(con)

    # ── Ingest model versions ──────────────────────────────────────────────────
    con.execute("DELETE FROM model_versions")
    for m in version_metas:
        # meta.yaml uses "version" key; "date_created" for creation date
        version_id = m.get("version_id") or m.get("version")
        created_raw = m.get("created_at") or m.get("date_created") or datetime.now(timezone.utc).isoformat()
        con.execute(
            """
            INSERT INTO model_versions
                (version_id, role, k, j, shift_type, vote_share_type,
                 n_training_dims, n_holdout_dims, holdout_r, geography, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                version_id,
                m.get("role"),
                m.get("k"),
                m.get("j"),
                m.get("shift_type"),
                m.get("vote_share_type"),
                m.get("n_training_dims") or m.get("training_dims"),
                m.get("n_holdout_dims") or m.get("holdout_dims"),
                str(m.get("holdout_r")) if m.get("holdout_r") is not None else None,
                m.get("geography"),
                m.get("description"),
                created_raw,
            ],
        )
    log.info("Ingested %d model versions", len(version_metas))

    # ── Ingest county shifts (current version) ─────────────────────────────────
    # Identify current version: prefer role='current', fallback to latest alphabetically
    current_meta = next(
        (m for m in version_metas if m.get("role") == "current"),
        version_metas[-1] if version_metas else {"version": "unknown"},
    )
    current_version_id = current_meta.get("version_id") or current_meta.get("version", "unknown")
    log.info("Using version_id '%s' for shift/assignment ingestion", current_version_id)

    if shifts is not None:
        shift_rows = _build_county_shifts(shifts, current_version_id)

        # Drop existing shifts for this version and rebuild
        con.execute(f"DELETE FROM county_shifts WHERE version_id = '{current_version_id}'")

        # county_shifts has dynamic columns (one per election shift dim) so the
        # fixed CREATE TABLE schema doesn't match. Recreate the table from the
        # DataFrame via register/unregister to handle arbitrary column sets.
        _insert_via_parquet(con, "county_shifts", shift_rows, mode="create")
        log.info("Ingested county_shifts: %d rows × %d cols", len(shift_rows), len(shift_rows.columns))

    # ── Ingest community assignments ───────────────────────────────────────────
    if assignments is not None:
        ca_rows = _build_community_assignments(assignments, current_version_id)
        con.execute(f"DELETE FROM community_assignments WHERE version_id = '{current_version_id}'")
        _insert_via_parquet(con, "community_assignments", ca_rows)
        log.info(
            "Ingested community_assignments: %d rows, k=%d", len(ca_rows), ca_rows["k"].iloc[0]
        )

        # ── Ingest type assignments ────────────────────────────────────────────
        ta_rows = _build_type_assignments(type_df, assignments, current_version_id)
        con.execute(f"DELETE FROM type_assignments WHERE version_id = '{current_version_id}'")
        _insert_via_parquet(con, "type_assignments", ta_rows)
        log.info("Ingested type_assignments: %d rows", len(ta_rows))

    # ── Connection cycle (post-shifts) ─────────────────────────────────────────
    # DuckDB develops heap corruption after several DataFrame INSERTs. Use
    # `del con` (not close()) — close() itself crashes on a corrupted heap.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (post-shifts)")

    # ── Ingest predictions ─────────────────────────────────────────────────────
    if predictions is not None:
        pred_rows = _build_predictions(predictions, current_version_id)
        con.execute(f"DELETE FROM predictions WHERE version_id = '{current_version_id}'")
        _insert_via_parquet(con, "predictions", pred_rows)
        log.info("Ingested predictions: %d rows", len(pred_rows))
    else:
        log.warning("No predictions file found; skipping predictions table")

    # ── Ingest HAC predictions ──────────────────────────────────────────────────
    if paths["predictions_hac"].exists():
        pred_hac = pd.read_parquet(paths["predictions_hac"])
        pred_hac["county_fips"] = pred_hac["county_fips"].astype(str).str.zfill(5)
        pred_hac_rows = _build_predictions(pred_hac, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        new_rows = pred_hac_rows[~pred_hac_rows["race"].isin(existing_races)]
        if len(new_rows):
            _insert_via_parquet(con, "predictions", new_rows)
            log.info("Ingested HAC predictions: %d rows", len(new_rows))
    else:
        log.info("No HAC predictions file found; skipping")

    # ── Ingest types predictions ───────────────────────────────────────────────
    if paths["predictions_types"].exists():
        pred_types = pd.read_parquet(paths["predictions_types"])
        pred_types["county_fips"] = pred_types["county_fips"].astype(str).str.zfill(5)
        pred_types_rows = _build_predictions(pred_types, current_version_id)
        existing_races = set(con.execute("SELECT DISTINCT race FROM predictions").fetchdf()["race"])
        # Types predictions take precedence: remove any overlapping (county, race) pairs first
        overlap_races = set(pred_types_rows["race"].unique()) & existing_races
        if overlap_races:
            for r in overlap_races:
                overlap_fips = pred_types_rows.loc[pred_types_rows["race"] == r, "county_fips"].tolist()
                placeholders = ", ".join(["?"] * len(overlap_fips))
                con.execute(
                    f"DELETE FROM predictions WHERE race = ? AND county_fips IN ({placeholders}) AND version_id = ?",
                    [r] + overlap_fips + [current_version_id],
                )
        _insert_via_parquet(con, "predictions", pred_types_rows)
        log.info("Ingested types predictions: %d rows", len(pred_types_rows))
    else:
        log.info("No types predictions file found; skipping")

    # ── Connection cycle (post-predictions) ────────────────────────────────────
    # The types-predictions insert (~19K rows) frequently triggers heap
    # corruption. Cycle immediately after to reset allocator state.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (post-predictions)")

    # ── Ingest community sigma ─────────────────────────────────────────────────
    if paths["sigma"].exists():
        sigma_df = pd.read_parquet(paths["sigma"])
        sigma_long_rows = []
        for row_id in sigma_df.index:
            for col_id in sigma_df.columns:
                sigma_long_rows.append({
                    "community_id_row": int(row_id),
                    "community_id_col": int(col_id),
                    "sigma_value": float(sigma_df.loc[row_id, col_id]),
                    "version_id": current_version_id,
                })
        sigma_long = pd.DataFrame(sigma_long_rows)
        con.execute(f"DELETE FROM community_sigma WHERE version_id = '{current_version_id}'")
        _insert_via_parquet(con, "community_sigma", sigma_long)
        log.info("Ingested community_sigma: %d cells", len(sigma_long))
    else:
        log.info("No county_community_sigma.parquet found; skipping sigma table")

    # ── Connection cycle (mid-build) ───────────────────────────────────────────
    # Use `del con` (not close()) — close() itself crashes on a corrupted heap.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (mid-build)")

    # ── Ingest community profiles (demographics overlay) ───────────────────────
    if paths["community_profiles"].exists():
        cp_df = pd.read_parquet(paths["community_profiles"])
        # Only load the fixed demographic columns into the structured table
        profile_cols = [
            "community_id", "n_counties", "pop_total",
            "pct_white_nh", "pct_black", "pct_asian", "pct_hispanic",
            "median_age", "median_hh_income", "pct_bachelors_plus",
            "pct_owner_occupied", "pct_wfh", "pct_management",
            "evangelical_share", "mainline_share", "catholic_share",
            "black_protestant_share", "congregations_per_1000",
            "religious_adherence_rate",
        ]
        cp_insert = cp_df[[c for c in profile_cols if c in cp_df.columns]].copy()
        con.execute("DELETE FROM community_profiles")
        _insert_via_parquet(con, "community_profiles", cp_insert)
        log.info("Ingested community_profiles: %d rows", len(cp_insert))
    else:
        log.info("No community_profiles.parquet found; skipping")

    # ── Ingest county demographics ──────────────────────────────────────────────
    if paths["county_acs"].exists():
        cd_df = pd.read_parquet(paths["county_acs"])
        cd_df["county_fips"] = cd_df["county_fips"].astype(str).str.zfill(5)
        _insert_via_parquet(con, "county_demographics", cd_df, mode="create")
        log.info("Ingested county_demographics: %d rows (%d columns)", len(cd_df), len(cd_df.columns))
    else:
        log.info("No county_acs_features.parquet found; skipping")

    # ── Connection cycle (post-demographics) ───────────────────────────────────
    # county_demographics (3K rows × 15 cols) triggers heap corruption.
    # Cycle before types/county_type_assignments to stay below threshold.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (post-demographics)")

    # ── Ingest type profiles (types table) ────────────────────────────────────
    if paths["type_profiles"].exists():
        tp_df = pd.read_parquet(paths["type_profiles"])
        # Add super_type_id from county_type_assignments if available
        if paths["county_type_assignments"].exists() and "super_type_id" not in tp_df.columns:
            cta_tmp = pd.read_parquet(paths["county_type_assignments"])
            if "super_type" in cta_tmp.columns and "dominant_type" in cta_tmp.columns:
                type_to_super = cta_tmp.groupby("dominant_type")["super_type"].first()
                tp_df["super_type_id"] = tp_df["type_id"].map(type_to_super)
                log.info("Added super_type_id to types table from county_type_assignments")
        # Add display_name if missing (generic names until proper naming pipeline)
        if "display_name" not in tp_df.columns:
            tp_df["display_name"] = tp_df["type_id"].apply(lambda x: f"Type {x}")
        # ── Generate and attach narratives ────────────────────────────────────
        log.info("Generating type narratives from demographic z-scores")
        try:
            narratives = generate_all_narratives(str(paths["type_profiles"]))
            tp_df["narrative"] = tp_df["type_id"].map(narratives)
            log.info("Attached narratives to %d types", tp_df["narrative"].notna().sum())
        except Exception as exc:
            log.warning("Narrative generation failed (%s); types table will lack narrative column", exc)
        _insert_via_parquet(con, "types", tp_df, mode="create")
        log.info("Ingested types: %d rows", len(tp_df))
    else:
        log.info("No type_profiles.parquet found; skipping types table")

    # ── Ingest county type assignments ─────────────────────────────────────────
    # Use DuckDB's native parquet reader directly — avoids the Python bridge
    # entirely, no DataFrame → DuckDB transfer needed since the file exists.
    if paths["county_type_assignments"].exists():
        p = str(paths["county_type_assignments"])
        con.execute("DROP TABLE IF EXISTS county_type_assignments")
        con.execute(f"CREATE TABLE county_type_assignments AS SELECT * FROM read_parquet('{p}')")
        n_cta = con.execute("SELECT COUNT(*) FROM county_type_assignments").fetchone()[0]
        log.info("Ingested county_type_assignments: %d rows", n_cta)
    else:
        log.info("No county_type_assignments_full.parquet found; skipping")

    # ── Ingest tract type assignments ──────────────────────────────────────────
    # The source parquet has 112,257 rows but only 81,129 unique GEOIDs (some
    # tracts appear in multiple state runs). Dedup on GEOID before loading.
    if paths["tract_type_assignments"].exists():
        tta_df = pd.read_parquet(
            paths["tract_type_assignments"],
            columns=["GEOID", "dominant_type", "super_type"],
        )
        tta_df = tta_df.drop_duplicates(subset="GEOID")
        tta_df = tta_df.rename(columns={"GEOID": "tract_geoid"})
        tta_df["dominant_type"] = tta_df["dominant_type"].astype("int32")
        tta_df["super_type"] = tta_df["super_type"].astype("int32")
        con.execute("DROP TABLE IF EXISTS tract_type_assignments")
        con.execute(
            "CREATE TABLE tract_type_assignments "
            "(tract_geoid VARCHAR PRIMARY KEY, dominant_type INTEGER, super_type INTEGER)"
        )
        con.register("_tta_view", tta_df)
        con.execute("INSERT INTO tract_type_assignments SELECT * FROM _tta_view")
        con.unregister("_tta_view")
        n_tta = con.execute("SELECT COUNT(*) FROM tract_type_assignments").fetchone()[0]
        log.info("Ingested tract_type_assignments: %d rows (from %d source rows)", n_tta, len(tta_df) + (112257 - 81129))
    else:
        log.info("No national_tract_assignments.parquet found; skipping tract_type_assignments")

    # ── Ingest super-types ─────────────────────────────────────────────────────
    if paths["super_types"].exists():
        st_df = pd.read_parquet(paths["super_types"])
        con.execute("DELETE FROM super_types")
        _insert_via_parquet(con, "super_types", st_df)
        log.info("Ingested super_types: %d rows", len(st_df))
    else:
        log.info("No super_types.parquet found; skipping")

    # ── Connection cycle (pre-domain) ─────────────────────────────────────────
    # Use `del con` (not close()) — close() itself crashes on a corrupted heap.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (pre-domain)")

    # ── Domain ingest ────────────────────────────────────────────────────────
    model_ddl(con)
    polling_ddl(con)

    ingest_model(con, current_version_id, project_root)
    ingest_polling(con, POLL_INGEST_CYCLE, project_root)

    # ── Connection cycle (post-domain) ────────────────────────────────────────
    # Domain ingest adds ~315K type_scores rows. Cycle before demographics to
    # avoid heap corruption on the final large table.
    del con
    gc.collect()
    con = duckdb.connect(str(db_path))
    log.info("Connection cycled (post-domain)")

    # ── Ingest demographics interpolated ───────────────────────────────────────
    if paths["demographics_interpolated"].exists():
        di_df = pd.read_parquet(paths["demographics_interpolated"])
        di_df["county_fips"] = di_df["county_fips"].astype(str).str.zfill(5)
        _insert_via_parquet(con, "demographics_interpolated", di_df, mode="create")
        log.info("Ingested demographics_interpolated: %d rows", len(di_df))
    else:
        log.info("No demographics_interpolated.parquet found; skipping")

    # Use `del con` (not close()) — close() crashes on corrupted glibc heap.
    # See S204/S246 for DuckDB 1.5.x malloc bug context.
    del con
    gc.collect()


def _report_summary(con: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    """Print a row-count summary for every table and the model versions registry."""
    log.info("Database build complete: %s", db_path)
    print("\n=== wethervane.duckdb summary ===")
    for table in [
        "counties", "model_versions", "community_assignments", "type_assignments",
        "county_shifts", "predictions", "community_sigma", "community_profiles",
        "county_demographics", "types", "county_type_assignments", "tract_type_assignments",
        "super_types", "type_covariance", "demographics_interpolated",
        "type_scores", "type_priors", "ridge_county_priors", "polls", "poll_notes",
        "races",
    ]:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n:,} rows")
        except Exception as e:
            print(f"  {table}: ERROR — {e}")

    print("\n=== Model Versions ===")
    rows = con.execute(
        "SELECT version_id, role, k, shift_type, vote_share_type, holdout_r FROM model_versions ORDER BY version_id"
    ).fetchall()
    print(f"  {'version_id':<45}  {'role':<20}  {'k':>4}  {'shift_type':<10}  {'holdout_r'}")
    for row in rows:
        vid, role, k, st, vst, hr = row
        print(f"  {str(vid):<45}  {str(role):<20}  {str(k):>4}  {str(st):<10}  {str(hr)}")


def _validate_integrity(con: duckdb.DuckDBPyConnection) -> None:
    """Run contract validation and exit with status 1 on any violation."""
    errors = validate_contract(con)
    if errors:
        for e in errors:
            log.error("CONTRACT VIOLATION: %s", e)
        del con
        gc.collect()
        sys.exit(1)
    log.info("Contract validation passed")


def build(db_path: Path, reset: bool = False, project_root: Path | None = None) -> None:
    """Build or update wethervane.duckdb from current pipeline artifacts.

    Orchestrates five stages in sequence:
    1. Optional reset (drop existing DB file)
    2. Schema creation (CREATE TABLE IF NOT EXISTS for all tables)
    3. Data ingestion (load parquets, insert rows, cycle connection to avoid DuckDB heap bug)
    4. Summary report (row counts + model versions)
    5. Contract validation (API-frontend contract check; exits 1 on violation)
    """
    _project_root = project_root if project_root is not None else PROJECT_ROOT
    paths = _resolve_paths(_project_root)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if reset:
        _reset_database(db_path)

    con = duckdb.connect(str(db_path))
    _create_schema(con)
    # _ingest_data manages its own connection cycling internally and closes the
    # connection before returning (del con + gc.collect at each checkpoint).
    del con
    gc.collect()

    _ingest_data(duckdb.connect(str(db_path)), db_path, paths, _project_root)

    con = duckdb.connect(str(db_path))
    _report_summary(con, db_path)
    _validate_integrity(con)
    # Use `del con` (not close()) — close() crashes on corrupted glibc heap.
    # See S204/S246 for DuckDB 1.5.x malloc bug context.
    del con
    gc.collect()


def main() -> None:
    """Entry point for build_database.py.

    DuckDB 1.5.x corrupts glibc's malloc heap after many large DataFrame
    inserts, causing a SIGABRT crash mid-build. The fix is to use jemalloc
    as the memory allocator (LD_PRELOAD). This function auto-detects jemalloc
    and re-execs itself with LD_PRELOAD set if it's not already active.
    """
    import os
    import subprocess
    import shutil

    parser = argparse.ArgumentParser(description="Build wethervane.duckdb from pipeline artifacts")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output DuckDB path")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild the database")
    args = parser.parse_args()

    # ── jemalloc auto-detection ────────────────────────────────────────────────
    # DuckDB 1.5.x corrupts glibc's malloc heap after many large DataFrame
    # inserts. We must verify jemalloc is *actually loaded* (not just that the
    # env var is set) because `uv run` may not propagate LD_PRELOAD to the
    # Python subprocess. A sentinel var prevents infinite re-exec loops.
    SENTINEL = "_BUILD_DB_JEMALLOC_REEXEC"
    jemalloc_paths = [
        "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
        "/usr/lib/libjemalloc.so.2",
        "/usr/local/lib/libjemalloc.so.2",
    ]

    def _jemalloc_actually_loaded() -> bool:
        """Check /proc/self/maps to verify jemalloc is in the process address space."""
        try:
            maps = Path("/proc/self/maps").read_text()
            return "libjemalloc" in maps
        except OSError:
            return False

    already_reexeced = os.environ.get(SENTINEL) == "1"

    if not _jemalloc_actually_loaded() and not already_reexeced:
        jemalloc_lib = next((p for p in jemalloc_paths if os.path.exists(p)), None)
        if jemalloc_lib:
            log.info(
                "DuckDB 1.5.x malloc bug: re-executing with LD_PRELOAD=%s to avoid SIGABRT",
                jemalloc_lib,
            )
            env = os.environ.copy()
            env["LD_PRELOAD"] = jemalloc_lib
            env[SENTINEL] = "1"
            result = subprocess.run([sys.executable] + sys.argv, env=env)
            sys.exit(result.returncode)
        else:
            log.warning(
                "jemalloc not found — DuckDB 1.5.x may crash with SIGABRT on large builds. "
                "Install libjemalloc2 (apt install libjemalloc2) to fix this."
            )
    elif _jemalloc_actually_loaded():
        log.info("jemalloc verified loaded in process address space")
    elif already_reexeced:
        log.warning(
            "jemalloc re-exec attempted but jemalloc still not loaded — "
            "proceeding without jemalloc (crash risk)"
        )

    build(db_path=args.db, reset=args.reset)

    # DuckDB 1.5.x corrupts the glibc heap during large DataFrame inserts.
    # Even with `del con; gc.collect()`, Python's finalization hits the
    # corrupted heap and crashes with `free(): corrupted unsorted chunks`.
    # Since all data is written and contract-validated, skip finalization.
    os._exit(0)


if __name__ == "__main__":
    main()
