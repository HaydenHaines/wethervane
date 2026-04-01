"""DDL schema definitions for wethervane.duckdb.

All CREATE TABLE IF NOT EXISTS statements live here. The orchestrator
(build_database.py) calls create_schema() once at the start of a build.
"""
from __future__ import annotations

import logging

import duckdb

log = logging.getLogger(__name__)

SCHEMA_SQL = """
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


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Execute all CREATE TABLE IF NOT EXISTS statements."""
    con.executemany("", [])
    for stmt in SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
    log.info("Schema created/verified")
