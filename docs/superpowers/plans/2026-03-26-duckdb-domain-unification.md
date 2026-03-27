# DuckDB Domain Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all parquet reads in `api/main.py` with DuckDB queries, and move poll CSV parsing from request-time to build-time, by introducing a typed domain contract layer with Pydantic validation.

**Architecture:** A `DomainSpec` dataclass per data domain declares which DuckDB tables it owns. Each active domain exports an `ingest()` function that validates source data and writes to DuckDB. `build_database.py` calls domain ingest in sequence. `api/main.py` reads only from DuckDB. Poll endpoints query the `polls` table instead of parsing CSVs at request time.

**Tech Stack:** Python, DuckDB, Pydantic v2, pandas, numpy. Tests use pytest + in-memory DuckDB.

**Spec:** `docs/superpowers/specs/2026-03-26-duckdb-domain-unification-design.md`

---

## File Map

**Create:**
- `src/db/domains/__init__.py` — `DomainSpec` dataclass, `DomainIngestionError`, `REGISTRY`
- `src/db/domains/model.py` — model domain DDL, Pydantic schemas, `ingest()`
- `src/db/domains/polling.py` — polling domain DDL, Pydantic schema, `ingest()`
- `src/db/domains/candidate.py` — reserved stub, `active=False`
- `tests/test_domain_model.py` — unit tests for model domain ingest
- `tests/test_domain_polling.py` — unit tests for polling domain ingest

**Modify:**
- `src/db/build_database.py` — wrap stages in named functions; add domain ingest calls; remove `TYPE_COVARIANCE_LONG_PATH` (replaced by model domain); remove HAC parquet path constants (moved into `model.py`)
- `api/main.py` — replace 6 `pd.read_parquet()` calls with DuckDB queries
- `api/routers/forecast.py` — update `/polls` and `/forecast/polls` to query DuckDB
- `api/tests/conftest.py` — add `polls` table to test DB
- `api/tests/test_forecast.py` — update poll endpoint tests
- `tests/test_db_builder.py` — extend for domain table presence after build
- `tests/test_api_contract.py` — extend for new domain tables

---

## Background: Key File Behaviors

Before touching any code, understand these:

- **`api/main.py` lines 98–169**: six parquet reads that bypass DuckDB. Each populates an `app.state` field that the prediction pipeline reads as numpy arrays or DataFrames.
- **`type_assignments.parquet`** format: wide — `county_fips` + `type_0_score … type_{J-1}_score` columns. The `api/main.py` reads score columns sorted by name.
- **`type_covariance.parquet`** format: J×J square matrix written as `pd.DataFrame(covariance_matrix)` — integer column indices 0…J-1. `api/main.py` takes `.values[:J,:J]`.
- **`type_profiles.parquet`**: includes `mean_dem_share` column (used as type prior) plus demographic columns (used by `types` table — not touched here).
- **`ridge_county_priors.parquet`**: columns `county_fips`, `ridge_pred_dem_share`.
- **`api/tests/conftest.py`**: sets `app.state` fields directly using `_noop_lifespan` — do not change how it bypasses startup.
- **`load_polls_with_notes()`** in `src/propagation/poll_weighting.py`: returns `(list[PollObservation], list[str])`. The notes strings (e.g. `"grade=A"`) are used by `apply_all_weights`. After this change, loading polls from DuckDB must reconstruct the same two lists.

---

## Task 1: DomainSpec and registry

**Files:**
- Create: `src/db/domains/__init__.py`
- Create: `src/db/domains/candidate.py`

- [ ] **Write the failing test**

Create `tests/test_domain_registry.py`:

```python
"""Tests for DomainSpec registry and error types."""
from src.db.domains import DomainSpec, DomainIngestionError, REGISTRY


def test_domain_spec_fields():
    spec = DomainSpec(
        name="test",
        tables=["foo", "bar"],
        description="A test domain",
    )
    assert spec.name == "test"
    assert spec.tables == ["foo", "bar"]
    assert spec.active is True
    assert spec.version_key == "version_id"


def test_registry_has_four_domains():
    names = [d.name for d in REGISTRY]
    assert set(names) == {"model", "polling", "candidate", "runtime"}


def test_candidate_is_inactive():
    candidate = next(d for d in REGISTRY if d.name == "candidate")
    assert candidate.active is False


def test_domain_ingestion_error_message():
    err = DomainIngestionError("model", "/path/to/file.parquet", "bad row 3")
    assert "model" in str(err)
    assert "bad row 3" in str(err)
```

- [ ] **Run to verify it fails**

```bash
python -m pytest tests/test_domain_registry.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.db.domains'`

- [ ] **Implement `src/db/domains/__init__.py`**

```python
"""Data domain registry for the WetherVane pipeline.

A domain is a named set of DuckDB tables with a defined source, schema,
and version discriminator. Two domains are active (model, polling); two
are reserved for future integration (candidate, runtime).
"""
from __future__ import annotations

from dataclasses import dataclass, field


class DomainIngestionError(Exception):
    """Raised when a domain's source data fails validation or is missing."""

    def __init__(self, domain: str, path: str, reason: str) -> None:
        self.domain = domain
        self.path = path
        self.reason = reason
        super().__init__(f"[{domain}] {path}: {reason}")


@dataclass
class DomainSpec:
    name: str               # "model" | "polling" | "candidate" | "runtime"
    tables: list[str]       # DuckDB tables this domain owns
    description: str
    active: bool = True     # False = reserved, skip on build
    version_key: str = "version_id"  # discriminator column name (docs only)


# Populated by each domain module at import time
REGISTRY: list[DomainSpec] = []


# ── Import domain modules to self-register ───────────────────────────────
from src.db.domains import model as _model  # noqa: E402, F401
from src.db.domains import polling as _polling  # noqa: E402, F401
from src.db.domains import candidate as _candidate  # noqa: E402, F401

REGISTRY.extend([
    _model.DOMAIN_SPEC,
    _polling.DOMAIN_SPEC,
    _candidate.DOMAIN_SPEC,
    DomainSpec(
        name="runtime",
        tables=[],
        description="User what-ifs and recalculate inputs — always API request bodies, never persisted",
        active=False,
        version_key="n/a",
    ),
])
```

- [ ] **Implement `src/db/domains/candidate.py`**

```python
"""Candidate domain — reserved for political sabermetrics silo.

active=False: no ingest() function; DomainSpec is registered to
establish the domain boundary for future implementation.
"""
from src.db.domains import DomainSpec

DOMAIN_SPEC = DomainSpec(
    name="candidate",
    tables=[],  # TBD when sabermetrics pipeline is implemented
    description="Politician stats: CTOV, district fit scores, career composites",
    active=False,
    version_key="version_id",
)
```

- [ ] **Run to verify tests pass**

```bash
python -m pytest tests/test_domain_registry.py -v
```
Expected: 4 tests PASS

- [ ] **Commit**

```bash
git add src/db/domains/__init__.py src/db/domains/candidate.py tests/test_domain_registry.py
git commit -m "feat: add DomainSpec registry with candidate stub"
```

---

## Task 2: Model domain — DDL and ingest

**Files:**
- Create: `src/db/domains/model.py`
- Create: `tests/test_domain_model.py`

- [ ] **Write the failing tests**

Create `tests/test_domain_model.py`:

```python
"""Tests for model domain ingest: type_scores, type_covariance, type_priors,
ridge_county_priors, hac_state_weights, hac_county_weights."""
from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.db.domains import DomainIngestionError
from src.db.domains.model import ingest, _cross_compliance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_FIPS = ["12001", "12003", "13001"]
TEST_J = 3  # number of types


def _base_db() -> duckdb.DuckDBPyConnection:
    """In-memory DB with counties + model_versions tables."""
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE counties (
            county_fips VARCHAR PRIMARY KEY,
            state_abbr VARCHAR, state_fips VARCHAR, county_name VARCHAR
        )
    """)
    for fips in TEST_FIPS:
        state = {"12": "FL", "13": "GA"}[fips[:2]]
        con.execute("INSERT INTO counties VALUES (?, ?, ?, ?)", [fips, state, fips[:2], f"County {fips}"])
    con.execute("""
        CREATE TABLE model_versions (
            version_id VARCHAR PRIMARY KEY, role VARCHAR, k INTEGER, j INTEGER,
            shift_type VARCHAR, vote_share_type VARCHAR, n_training_dims INTEGER,
            n_holdout_dims INTEGER, holdout_r VARCHAR, geography VARCHAR,
            description VARCHAR, created_at TIMESTAMP
        )
    """)
    con.execute("INSERT INTO model_versions VALUES ('test_v1','current',3,3,'logodds','total',30,3,'0.90','test','test','2026-01-01')")
    return con


def _write_type_assignments(tmp: Path) -> None:
    """Write a valid wide-format type_assignments.parquet."""
    df = pd.DataFrame({
        "county_fips": TEST_FIPS,
        "type_0_score": [0.5, 0.3, 0.2],
        "type_1_score": [0.3, 0.5, 0.4],
        "type_2_score": [0.2, 0.2, 0.4],
    })
    (tmp / "communities").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "communities" / "type_assignments.parquet", index=False)


def _write_type_covariance(tmp: Path) -> None:
    """Write a valid J×J square covariance matrix."""
    cov = np.eye(TEST_J) * 0.01 + np.ones((TEST_J, TEST_J)) * 0.002
    (tmp / "covariance").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(tmp / "covariance" / "type_covariance.parquet")


def _write_type_profiles(tmp: Path) -> None:
    """Write a valid type_profiles.parquet with mean_dem_share."""
    df = pd.DataFrame({"type_id": range(TEST_J), "mean_dem_share": [0.40, 0.50, 0.60]})
    df.to_parquet(tmp / "communities" / "type_profiles.parquet", index=False)


def _write_ridge_priors(tmp: Path) -> None:
    """Write a valid ridge_county_priors.parquet."""
    df = pd.DataFrame({"county_fips": TEST_FIPS, "ridge_pred_dem_share": [0.42, 0.38, 0.55]})
    (tmp / "models" / "ridge_model").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "models" / "ridge_model" / "ridge_county_priors.parquet", index=False)


@pytest.fixture
def tmp_data(tmp_path):
    """Write all four parquets to a temp directory tree."""
    _write_type_assignments(tmp_path)
    _write_type_covariance(tmp_path)
    _write_type_profiles(tmp_path)
    _write_ridge_priors(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

def test_ingest_creates_type_scores(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM type_scores WHERE version_id = 'test_v1'").fetchone()[0]
    assert n == len(TEST_FIPS) * TEST_J  # N counties × J types


def test_type_scores_values_sum_to_one_per_county(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    df = con.execute("SELECT county_fips, SUM(score) as total FROM type_scores WHERE version_id='test_v1' GROUP BY county_fips").fetchdf()
    assert (df["total"] - 1.0).abs().max() < 0.01


def test_ingest_creates_type_covariance_symmetric(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM type_covariance WHERE version_id='test_v1'").fetchone()[0]
    assert n == TEST_J ** 2
    # Symmetry: value at (i,j) == value at (j,i)
    asym = con.execute("""
        SELECT COUNT(*) FROM type_covariance a
        JOIN type_covariance b ON a.type_i=b.type_j AND a.type_j=b.type_i AND a.version_id=b.version_id
        WHERE ABS(a.value - b.value) > 1e-6 AND a.version_id='test_v1'
    """).fetchone()[0]
    assert asym == 0


def test_ingest_creates_type_priors(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    df = con.execute("SELECT * FROM type_priors WHERE version_id='test_v1' ORDER BY type_id").fetchdf()
    assert len(df) == TEST_J
    assert list(df["mean_dem_share"]) == pytest.approx([0.40, 0.50, 0.60])


def test_ingest_creates_ridge_priors(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    n = con.execute("SELECT COUNT(*) FROM ridge_county_priors WHERE version_id='test_v1'").fetchone()[0]
    assert n == len(TEST_FIPS)


def test_type_ids_zero_indexed_contiguous(tmp_data):
    con = _base_db()
    ingest(con, "test_v1", tmp_data)
    ids = sorted(con.execute("SELECT DISTINCT type_id FROM type_scores WHERE version_id='test_v1'").df()["type_id"].tolist())
    assert ids == list(range(TEST_J))


# ---------------------------------------------------------------------------
# Tests: validation failures abort ingest
# ---------------------------------------------------------------------------

def test_asymmetric_covariance_raises(tmp_data):
    # Overwrite with asymmetric matrix
    cov = np.eye(TEST_J) * 0.01
    cov[0, 1] = 0.5  # asymmetric
    pd.DataFrame(cov).to_parquet(tmp_data / "covariance" / "type_covariance.parquet")
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="not symmetric"):
        ingest(con, "test_v1", tmp_data)


def test_unknown_county_fips_raises(tmp_data):
    # Add a row with unknown FIPS to type_assignments
    df = pd.read_parquet(tmp_data / "communities" / "type_assignments.parquet")
    df.loc[len(df)] = {"county_fips": "99999", "type_0_score": 0.5, "type_1_score": 0.3, "type_2_score": 0.2}
    df.to_parquet(tmp_data / "communities" / "type_assignments.parquet", index=False)
    con = _base_db()
    with pytest.raises(DomainIngestionError, match="county_fips"):
        ingest(con, "test_v1", tmp_data)


def test_score_out_of_range_raises(tmp_data):
    df = pd.read_parquet(tmp_data / "communities" / "type_assignments.parquet")
    df["type_0_score"] = 1.5  # out of [0,1]
    df.to_parquet(tmp_data / "communities" / "type_assignments.parquet", index=False)
    con = _base_db()
    with pytest.raises(DomainIngestionError):
        ingest(con, "test_v1", tmp_data)


# ---------------------------------------------------------------------------
# Tests: HAC weight ingestion
# ---------------------------------------------------------------------------

def _write_hac_state_weights(tmp: Path) -> None:
    """Write a valid wide-format community_weights_state_hac.parquet."""
    df = pd.DataFrame({
        "state_abbr": ["FL", "GA"],
        "community_0": [0.6, 0.4],
        "community_1": [0.3, 0.4],
        "community_2": [0.1, 0.2],
    })
    (tmp / "propagation").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp / "propagation" / "community_weights_state_hac.parquet", index=False)


def _write_hac_county_weights(tmp: Path) -> None:
    """Write a valid wide-format community_weights_county_hac.parquet."""
    df = pd.DataFrame({
        "county_fips": TEST_FIPS,
        "community_0": [0.5, 0.2, 0.3],
        "community_1": [0.3, 0.5, 0.4],
        "community_2": [0.2, 0.3, 0.3],
    })
    df.to_parquet(tmp / "propagation" / "community_weights_county_hac.parquet", index=False)


@pytest.fixture
def tmp_data_with_hac(tmp_data):
    """tmp_data plus HAC weight parquets."""
    _write_hac_state_weights(tmp_data)
    _write_hac_county_weights(tmp_data)
    return tmp_data


def test_ingest_creates_hac_state_weights(tmp_data_with_hac):
    con = _base_db()
    ingest(con, "test_v1", tmp_data_with_hac)
    n = con.execute("SELECT COUNT(*) FROM hac_state_weights WHERE version_id='test_v1'").fetchone()[0]
    # 2 states × 3 communities = 6
    assert n == 2 * 3


def test_ingest_creates_hac_county_weights(tmp_data_with_hac):
    con = _base_db()
    ingest(con, "test_v1", tmp_data_with_hac)
    n = con.execute("SELECT COUNT(*) FROM hac_county_weights WHERE version_id='test_v1'").fetchone()[0]
    # 3 counties × 3 communities = 9
    assert n == len(TEST_FIPS) * 3


def test_hac_weights_missing_gracefully_skipped(tmp_data):
    """HAC parquets absent → ingest succeeds (tables are empty, not error)."""
    con = _base_db()
    ingest(con, "test_v1", tmp_data)  # no HAC files in tmp_data
    n_sw = con.execute("SELECT COUNT(*) FROM hac_state_weights").fetchone()[0]
    n_cw = con.execute("SELECT COUNT(*) FROM hac_county_weights").fetchone()[0]
    assert n_sw == 0
    assert n_cw == 0
```

- [ ] **Run to verify it fails**

```bash
python -m pytest tests/test_domain_model.py -v
```
Expected: `ImportError: cannot import name 'ingest' from 'src.db.domains.model'`

- [ ] **Implement `src/db/domains/model.py`**

```python
"""Model domain: type scores, covariance, priors, ridge priors, HAC weights.

Replaces the six pd.read_parquet() calls in api/main.py at startup.
All tables are version-linked (version_id FK → model_versions).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.db.domains import DomainIngestionError, DomainSpec

log = logging.getLogger(__name__)

DOMAIN_SPEC = DomainSpec(
    name="model",
    tables=[
        "type_scores", "type_covariance", "type_priors",
        "ridge_county_priors", "hac_state_weights", "hac_county_weights",
    ],
    description="KMeans type scores, covariance matrix, priors, ridge predictions, HAC fallback weights",
    version_key="version_id",
)

# ---------------------------------------------------------------------------
# Pydantic schemas — validate before writing to DuckDB
# ---------------------------------------------------------------------------

class TypeScoreRow(BaseModel):
    county_fips: str
    type_id: int = Field(ge=0)
    score: float = Field(ge=0.0, le=1.0)


class TypeCovarianceRow(BaseModel):
    type_i: int = Field(ge=0)
    type_j: int = Field(ge=0)
    value: float


class TypePriorRow(BaseModel):
    type_id: int = Field(ge=0)
    mean_dem_share: float = Field(ge=0.0, le=1.0)


class RidgeCountyPriorRow(BaseModel):
    county_fips: str
    ridge_pred_dem_share: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS type_scores (
    county_fips VARCHAR NOT NULL,
    type_id     INTEGER NOT NULL,
    score       FLOAT   NOT NULL,
    version_id  VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, type_id, version_id)
);
CREATE TABLE IF NOT EXISTS type_covariance (
    type_i      INTEGER NOT NULL,
    type_j      INTEGER NOT NULL,
    value       FLOAT   NOT NULL,
    version_id  VARCHAR NOT NULL,
    PRIMARY KEY (type_i, type_j, version_id)
);
CREATE TABLE IF NOT EXISTS type_priors (
    type_id        INTEGER NOT NULL,
    mean_dem_share FLOAT   NOT NULL,
    version_id     VARCHAR NOT NULL,
    PRIMARY KEY (type_id, version_id)
);
CREATE TABLE IF NOT EXISTS ridge_county_priors (
    county_fips          VARCHAR NOT NULL,
    ridge_pred_dem_share FLOAT   NOT NULL,
    version_id           VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, version_id)
);
CREATE TABLE IF NOT EXISTS hac_state_weights (
    state_abbr   VARCHAR NOT NULL,
    community_id INTEGER NOT NULL,
    weight       FLOAT   NOT NULL,
    version_id   VARCHAR NOT NULL,
    PRIMARY KEY (state_abbr, community_id, version_id)
);
CREATE TABLE IF NOT EXISTS hac_county_weights (
    county_fips  VARCHAR NOT NULL,
    community_id INTEGER NOT NULL,
    weight       FLOAT   NOT NULL,
    version_id   VARCHAR NOT NULL,
    PRIMARY KEY (county_fips, community_id, version_id)
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_rows(schema_class, rows: list[dict], source: str) -> None:
    for i, row in enumerate(rows):
        try:
            schema_class(**row)
        except Exception as exc:
            raise DomainIngestionError("model", source, f"row {i}: {exc}") from exc


def _cross_compliance(con: duckdb.DuckDBPyConnection, version_id: str) -> None:
    # type_scores.county_fips must exist in counties
    orphans = con.execute("""
        SELECT DISTINCT ts.county_fips
        FROM type_scores ts
        LEFT JOIN counties c ON ts.county_fips = c.county_fips
        WHERE c.county_fips IS NULL AND ts.version_id = ?
    """, [version_id]).fetchdf()
    if not orphans.empty:
        raise DomainIngestionError(
            "model", "type_scores",
            f"unknown county_fips (first 5): {orphans['county_fips'].tolist()[:5]}"
        )

    # type_ids consistent across tables
    max_ts = con.execute("SELECT MAX(type_id) FROM type_scores WHERE version_id=?", [version_id]).fetchone()[0]
    max_tc = con.execute("SELECT MAX(type_i) FROM type_covariance WHERE version_id=?", [version_id]).fetchone()[0]
    max_tp = con.execute("SELECT MAX(type_id) FROM type_priors WHERE version_id=?", [version_id]).fetchone()[0]
    if not (max_ts == max_tc == max_tp):
        raise DomainIngestionError(
            "model", "type tables",
            f"type_id max inconsistent: scores={max_ts}, covariance={max_tc}, priors={max_tp}"
        )

    # type_ids must be zero-indexed contiguous
    actual = sorted(
        con.execute("SELECT DISTINCT type_id FROM type_scores WHERE version_id=?", [version_id])
        .df()["type_id"].tolist()
    )
    expected = list(range(max_ts + 1))
    if actual != expected:
        raise DomainIngestionError(
            "model", "type_scores",
            f"type_ids not zero-indexed contiguous (first 5 actual: {actual[:5]})"
        )

    # ridge_county_priors.county_fips must exist in counties
    orphan_r = con.execute("""
        SELECT DISTINCT r.county_fips
        FROM ridge_county_priors r
        LEFT JOIN counties c ON r.county_fips = c.county_fips
        WHERE c.county_fips IS NULL AND r.version_id = ?
    """, [version_id]).fetchdf()
    if not orphan_r.empty:
        raise DomainIngestionError(
            "model", "ridge_county_priors",
            f"unknown county_fips (first 5): {orphan_r['county_fips'].tolist()[:5]}"
        )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all model domain tables (idempotent)."""
    con.execute(_DDL)


def ingest(
    con: duckdb.DuckDBPyConnection,
    version_id: str,
    project_root: Path,
) -> None:
    """Validate and ingest all model domain parquets into DuckDB.

    Gracefully skips individual files that are missing (type pipeline
    artifacts are generated during model runs — they may not exist yet
    during schema-only builds). Cross-compliance runs only when type_scores
    was populated.
    """
    create_tables(con)

    # Clear existing rows for this version
    for table in DOMAIN_SPEC.tables:
        con.execute(f"DELETE FROM {table} WHERE version_id = ?", [version_id])

    data = project_root / "data"
    _ingest_type_scores(con, version_id, data / "communities" / "type_assignments.parquet")
    _ingest_type_covariance(con, version_id, data / "covariance" / "type_covariance.parquet")
    _ingest_type_priors(con, version_id, data / "communities" / "type_profiles.parquet")
    _ingest_ridge_priors(con, version_id, data / "models" / "ridge_model" / "ridge_county_priors.parquet")
    _ingest_hac_state_weights(con, version_id, data / "propagation" / "community_weights_state_hac.parquet")
    _ingest_hac_county_weights(con, version_id, data / "propagation" / "community_weights_county_hac.parquet")

    # Cross-compliance only if type_scores were ingested
    has_scores = con.execute("SELECT COUNT(*) FROM type_scores WHERE version_id=?", [version_id]).fetchone()[0]
    if has_scores > 0:
        _cross_compliance(con, version_id)
        log.info("Model domain cross-compliance passed")


def _ingest_type_scores(con, version_id, path):
    if not path.exists():
        log.warning("type_assignments.parquet not found; skipping type_scores")
        return
    ta = pd.read_parquet(path)
    ta["county_fips"] = ta["county_fips"].astype(str).str.zfill(5)
    score_cols = sorted([c for c in ta.columns if c.endswith("_score")])
    if not score_cols:
        raise DomainIngestionError("model", str(path), "no *_score columns found")

    rows = []
    for col in score_cols:
        type_id = int(col.split("_")[1])  # "type_7_score" → 7
        for fips, score in zip(ta["county_fips"], ta[col]):
            rows.append({"county_fips": fips, "type_id": type_id, "score": float(score)})

    _validate_rows(TypeScoreRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_scores SELECT * FROM df")
    log.info("type_scores: %d rows (%d counties × %d types)", len(rows), len(ta), len(score_cols))


def _ingest_type_covariance(con, version_id, path):
    if not path.exists():
        log.warning("type_covariance.parquet not found; skipping type_covariance")
        return
    cov_df = pd.read_parquet(path)
    J = cov_df.shape[0]
    mat = cov_df.values[:J, :J].astype(float)
    if not np.allclose(mat, mat.T, atol=1e-6):
        raise DomainIngestionError("model", str(path), "covariance matrix is not symmetric")

    rows = [{"type_i": i, "type_j": j, "value": float(mat[i, j])} for i in range(J) for j in range(J)]
    _validate_rows(TypeCovarianceRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_covariance SELECT * FROM df")
    log.info("type_covariance: %d×%d (%d rows)", J, J, len(rows))


def _ingest_type_priors(con, version_id, path):
    """Ingest type-level dem_share priors.

    `type_profiles.parquet` is written by describe_types.py with demographic
    columns only — it does NOT include mean_dem_share yet. When the column is
    absent, fall back to 0.45 for all types (matching api/main.py behavior).
    The fallback reads J from the already-ingested type_scores table.
    """
    if not path.exists():
        log.warning("type_profiles.parquet not found; skipping type_priors")
        return
    profiles = pd.read_parquet(path)
    if "mean_dem_share" in profiles.columns:
        # Use explicit type_id column, not positional index
        rows = [
            {"type_id": int(row["type_id"]), "mean_dem_share": float(row["mean_dem_share"])}
            for _, row in profiles.iterrows()
        ]
    else:
        # Graceful fallback: use 0.45 for every type discovered in type_scores
        log.warning(
            "type_profiles.parquet has no mean_dem_share column — using 0.45 default for all types"
        )
        type_ids = (
            con.execute(
                "SELECT DISTINCT type_id FROM type_scores WHERE version_id=? ORDER BY type_id",
                [version_id],
            )
            .df()["type_id"]
            .tolist()
        )
        if not type_ids:
            log.warning("type_scores also empty; skipping type_priors ingest")
            return
        rows = [{"type_id": int(tid), "mean_dem_share": 0.45} for tid in type_ids]
    _validate_rows(TypePriorRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO type_priors SELECT * FROM df")
    log.info("type_priors: %d rows", len(rows))


def _ingest_ridge_priors(con, version_id, path):
    if not path.exists():
        log.warning("ridge_county_priors.parquet not found; skipping")
        return
    rf = pd.read_parquet(path)
    rf["county_fips"] = rf["county_fips"].astype(str).str.zfill(5)
    rows = [
        {"county_fips": row["county_fips"], "ridge_pred_dem_share": float(row["ridge_pred_dem_share"])}
        for _, row in rf.iterrows()
    ]
    _validate_rows(RidgeCountyPriorRow, rows, str(path))
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO ridge_county_priors SELECT * FROM df")
    log.info("ridge_county_priors: %d rows", len(rows))


def _ingest_hac_state_weights(con, version_id, path):
    if not path.exists():
        log.warning("community_weights_state_hac.parquet not found; skipping hac_state_weights")
        return
    sw = pd.read_parquet(path)
    comm_cols = sorted([c for c in sw.columns if c.startswith("community_")])
    rows = [
        {"state_abbr": row["state_abbr"], "community_id": int(col.split("_")[1]), "weight": float(row[col])}
        for _, row in sw.iterrows()
        for col in comm_cols
    ]
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO hac_state_weights SELECT * FROM df")
    log.info("hac_state_weights: %d rows", len(rows))


def _ingest_hac_county_weights(con, version_id, path):
    if not path.exists():
        log.warning("community_weights_county_hac.parquet not found; skipping hac_county_weights")
        return
    cw = pd.read_parquet(path)
    comm_cols = sorted([c for c in cw.columns if c.startswith("community_")])
    rows = [
        {"county_fips": str(row["county_fips"]).zfill(5), "community_id": int(col.split("_")[1]), "weight": float(row[col])}
        for _, row in cw.iterrows()
        for col in comm_cols
    ]
    df = pd.DataFrame(rows)
    df["version_id"] = version_id
    con.execute("INSERT INTO hac_county_weights SELECT * FROM df")
    log.info("hac_county_weights: %d rows", len(rows))
```

- [ ] **Run to verify tests pass**

```bash
python -m pytest tests/test_domain_model.py -v
```
Expected: all tests PASS

- [ ] **Commit**

```bash
git add src/db/domains/model.py tests/test_domain_model.py
git commit -m "feat: add model domain ingest with Pydantic validation"
```

---

## Task 3: Polling domain — DDL and ingest

**Files:**
- Create: `src/db/domains/polling.py`
- Create: `tests/test_domain_polling.py`

The polling domain ingests from the poll CSV (`data/polls/polls_{cycle}.csv`) into three DuckDB tables. The `polls` table includes a `notes` VARCHAR column (raw notes string) so the endpoint can reconstruct `PollObservation` lists without re-parsing.

- [ ] **Write the failing tests**

Create `tests/test_domain_polling.py`:

```python
"""Tests for polling domain ingest."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import duckdb
import pytest

from src.db.domains.polling import ingest, _make_poll_id


def _base_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    return con


def _write_poll_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["race", "geography", "geo_level", "dem_share", "n_sample", "date", "pollster", "notes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_ROWS = [
    {"race": "FL Senate", "geography": "FL", "geo_level": "state",
     "dem_share": "0.45", "n_sample": "600", "date": "2026-01-15",
     "pollster": "Siena", "notes": "grade=A"},
    {"race": "FL Senate", "geography": "FL", "geo_level": "state",
     "dem_share": "0.47", "n_sample": "800", "date": "2026-02-01",
     "pollster": "Emerson", "notes": "grade=B+"},
]


def test_ingest_creates_polls_table(tmp_path):
    _write_poll_csv(tmp_path / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls WHERE cycle='2026'").fetchone()[0]
    assert n == 2


def test_poll_id_is_stable():
    id1 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    id2 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    assert id1 == id2
    assert len(id1) == 16  # first 16 chars of SHA-256 hex


def test_poll_id_differs_on_different_pollster():
    id1 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Siena", "2026")
    id2 = _make_poll_id("FL Senate", "FL", "2026-01-15", "Emerson", "2026")
    assert id1 != id2


def test_notes_preserved(tmp_path):
    _write_poll_csv(tmp_path / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    notes = con.execute("SELECT notes FROM polls WHERE pollster='Siena' AND cycle='2026'").fetchone()[0]
    assert notes == "grade=A"


def test_poll_notes_table_populated(tmp_path):
    _write_poll_csv(tmp_path / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM poll_notes WHERE note_type='grade'").fetchone()[0]
    assert n == 2


def test_poll_crosstabs_table_exists_empty(tmp_path):
    _write_poll_csv(tmp_path / "polls" / "polls_2026.csv", SAMPLE_ROWS)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM poll_crosstabs").fetchone()[0]
    assert n == 0


def test_missing_csv_returns_empty(tmp_path):
    """Missing CSV creates empty tables (no error).

    Deliberate deviation from the spec's error-table which lists
    DomainIngestionError for missing sources: that rule applies to the
    model domain parquets (required for prediction). Polls are optional
    — a missing poll CSV is valid for historical cycles or future races
    not yet polled. This matches the existing /polls endpoint behavior
    (returns [] on FileNotFoundError).
    """
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
    assert n == 0


def test_invalid_dem_share_row_is_skipped(tmp_path):
    """Rows with out-of-range dem_share are filtered at CSV parse time."""
    rows = SAMPLE_ROWS + [
        {"race": "FL Senate", "geography": "FL", "geo_level": "state",
         "dem_share": "1.5", "n_sample": "600", "date": "2026-03-01",
         "pollster": "Bad Poll", "notes": ""},
    ]
    _write_poll_csv(tmp_path / "polls" / "polls_2026.csv", rows)
    con = _base_db()
    ingest(con, "2026", tmp_path)
    n = con.execute("SELECT COUNT(*) FROM polls WHERE cycle='2026'").fetchone()[0]
    assert n == 2  # bad row skipped
```

- [ ] **Run to verify it fails**

```bash
python -m pytest tests/test_domain_polling.py -v
```
Expected: `ImportError: cannot import name 'ingest' from 'src.db.domains.polling'`

- [ ] **Implement `src/db/domains/polling.py`**

```python
"""Polling domain: polls, poll_crosstabs, poll_notes.

Ingests from polls_{cycle}.csv into DuckDB. Replaces CSV-at-request-time
parsing in /polls and /forecast/polls endpoints.

The `polls` table stores a `notes` VARCHAR column (raw notes string)
alongside the structured `poll_notes` table so endpoints can reconstruct
PollObservation objects via a simple SELECT.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

from src.db.domains import DomainIngestionError, DomainSpec

log = logging.getLogger(__name__)

DOMAIN_SPEC = DomainSpec(
    name="polling",
    tables=["polls", "poll_crosstabs", "poll_notes"],
    description="Poll rows ingested from CSV; crosstabs and quality notes queryable via SQL",
    version_key="cycle",
)


class PollIngestRow(BaseModel):
    race: str
    geography: str
    geo_level: Literal["state", "county", "district"]
    dem_share: float = Field(ge=0.0, le=1.0)
    n_sample: int = Field(gt=0)
    date: str | None
    cycle: str


_DDL = """
CREATE TABLE IF NOT EXISTS polls (
    poll_id    VARCHAR NOT NULL,
    race       VARCHAR NOT NULL,
    geography  VARCHAR NOT NULL,
    geo_level  VARCHAR NOT NULL,
    dem_share  FLOAT   NOT NULL,
    n_sample   INTEGER NOT NULL,
    date       VARCHAR,
    pollster   VARCHAR,
    notes      VARCHAR,
    cycle      VARCHAR NOT NULL,
    PRIMARY KEY (poll_id)
);
CREATE TABLE IF NOT EXISTS poll_crosstabs (
    poll_id           VARCHAR NOT NULL,
    demographic_group VARCHAR NOT NULL,
    group_value       VARCHAR NOT NULL,
    dem_share         FLOAT,
    n_sample          INTEGER
);
CREATE TABLE IF NOT EXISTS poll_notes (
    poll_id    VARCHAR NOT NULL,
    note_type  VARCHAR NOT NULL,
    note_value VARCHAR NOT NULL
);
"""


def _make_poll_id(race: str, geography: str, date: str | None, pollster: str | None, cycle: str) -> str:
    """SHA-256 hex digest of pipe-delimited fields, truncated to 16 chars."""
    key = "|".join([race, geography, date or "", pollster or "", cycle])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _parse_note_kvs(notes_str: str) -> list[tuple[str, str]]:
    """Parse 'grade=A foo=bar' into [('grade','A'), ('foo','bar')]."""
    pairs = []
    for part in notes_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            pairs.append((k.strip(), v.strip()))
    return pairs


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all polling domain tables (idempotent)."""
    con.execute(_DDL)


def ingest(con: duckdb.DuckDBPyConnection, cycle: str, project_root: Path) -> None:
    """Ingest polls_{cycle}.csv into DuckDB polling tables.

    Missing CSV is treated as an empty poll set — tables are created
    but left empty. Invalid rows (bad dem_share, zero n_sample) are
    skipped with a warning, matching the existing CSV parse behavior.
    """
    create_tables(con)

    # Clear existing data for this cycle
    for table in DOMAIN_SPEC.tables:
        con.execute(f"DELETE FROM {table} WHERE poll_id IN (SELECT poll_id FROM polls WHERE cycle=?)", [cycle])
    con.execute("DELETE FROM polls WHERE cycle=?", [cycle])

    path = project_root / "data" / "polls" / f"polls_{cycle}.csv"
    if not path.exists():
        log.warning("polls_%s.csv not found at %s; polling tables will be empty", cycle, path)
        return

    poll_rows = []
    note_rows = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dem = row.get("dem_share", "").strip()
            raw_n = row.get("n_sample", "").strip()
            if not raw_dem or not raw_n:
                continue
            try:
                dem_share = float(raw_dem)
                n_sample = int(float(raw_n))
            except ValueError:
                continue
            if not (0.0 < dem_share < 1.0) or n_sample <= 0:
                continue

            race = row.get("race", "").strip()
            geography = row.get("geography", "").strip()
            geo_level = row.get("geo_level", "state").strip() or "state"
            date = row.get("date", "").strip() or None
            pollster = row.get("pollster", "").strip() or None
            notes = row.get("notes", "").strip()

            # Validate via Pydantic (skips rows that fail)
            try:
                PollIngestRow(
                    race=race, geography=geography, geo_level=geo_level,
                    dem_share=dem_share, n_sample=n_sample, date=date, cycle=cycle,
                )
            except Exception as exc:
                log.warning("Skipping invalid poll row (%s): %s", row, exc)
                continue

            poll_id = _make_poll_id(race, geography, date, pollster, cycle)
            poll_rows.append({
                "poll_id": poll_id, "race": race, "geography": geography,
                "geo_level": geo_level, "dem_share": dem_share, "n_sample": n_sample,
                "date": date, "pollster": pollster, "notes": notes, "cycle": cycle,
            })
            for note_type, note_value in _parse_note_kvs(notes):
                note_rows.append({"poll_id": poll_id, "note_type": note_type, "note_value": note_value})

    if poll_rows:
        df = pd.DataFrame(poll_rows)
        con.execute("INSERT INTO polls SELECT * FROM df")
        log.info("polls: ingested %d rows for cycle=%s", len(poll_rows), cycle)

    if note_rows:
        ndf = pd.DataFrame(note_rows)
        con.execute("INSERT INTO poll_notes SELECT * FROM ndf")
        log.info("poll_notes: ingested %d rows", len(note_rows))
```

- [ ] **Run to verify tests pass**

```bash
python -m pytest tests/test_domain_polling.py -v
```
Expected: all tests PASS

- [ ] **Commit**

```bash
git add src/db/domains/polling.py tests/test_domain_polling.py
git commit -m "feat: add polling domain ingest from CSV to DuckDB"
```

---

## Task 4: Wire domains into build_database.py

**Files:**
- Modify: `src/db/build_database.py`

This task reorganizes existing `build()` logic into named stage functions and calls domain ingest in sequence. **Behavioral change to existing logic is minimal** — only the type_covariance and HAC weight ingest paths are replaced; everything else stays.

- [ ] **Write the failing test**

Add to `tests/test_db_builder.py`:

```python
def test_domain_tables_created_after_build(tmp_path):
    """build() creates all domain tables even when source parquets are missing."""
    import yaml
    from src.db.build_database import build

    # Minimal version dir
    version_id = "test-build-v1"
    ver_dir = tmp_path / "models" / "versions" / version_id
    ver_dir.mkdir(parents=True)
    meta = {
        "version_id": version_id, "role": "current", "k": 3, "j": 3,
        "shift_type": "logodds", "vote_share_type": "total",
        "n_training_dims": 30, "n_holdout_dims": 3,
        "holdout_r": "0.70", "geography": "test", "description": "test",
    }
    (ver_dir / "meta.yaml").write_text(yaml.dump(meta))

    db_path = tmp_path / "test.duckdb"
    build(db_path=db_path, reset=True, project_root=tmp_path)

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    for table in ["type_scores", "type_covariance", "type_priors",
                  "ridge_county_priors", "hac_state_weights", "hac_county_weights",
                  "polls", "poll_crosstabs", "poll_notes"]:
        n = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?", [table]
        ).fetchone()[0]
        assert n == 1, f"Table {table!r} not created by build()"
    con.close()
```

- [ ] **Run to verify it fails**

```bash
python -m pytest tests/test_db_builder.py::test_domain_tables_created_after_build -v
```
Expected: FAIL — either `TypeError` (unexpected `project_root` arg) or missing tables

- [ ] **Modify `build_database.py`**

The changes are additive. At the top of `build()`:

1. Add `project_root: Path | None = None` parameter to `build()`. Default to `PROJECT_ROOT`.

2. After `build_core_tables()` (the existing county/version ingest block), call:

```python
# ── Domain ingest ───────────────────────────────────────────────────────
from src.db.domains.model import ingest as ingest_model, create_tables as model_ddl
from src.db.domains.polling import ingest as ingest_polling, create_tables as polling_ddl

model_ddl(con)
polling_ddl(con)

ingest_model(con, current_version_id, _project_root)
ingest_polling(con, "2026", _project_root)
```

Where `_project_root = project_root or PROJECT_ROOT`.

3. Remove the existing `TYPE_COVARIANCE_LONG_PATH` ingest block (lines ~620–627 — this is now handled by `ingest_model`).

4. Remove `state_w_path` and `county_w_path` constants from `build_database.py` top-level (they are now in `model.py`).

5. Extend `validate_contract()` to check the new tables:

```python
# In validate_contract(), add to required dict:
"type_scores": ["county_fips", "type_id", "score"],
"type_priors": ["type_id", "mean_dem_share"],
"polls": ["poll_id", "race", "geography", "dem_share"],
```

- [ ] **Run to verify new test passes, existing tests still pass**

```bash
python -m pytest tests/test_db_builder.py -v
```
Expected: all tests PASS

- [ ] **Commit**

```bash
git add src/db/build_database.py tests/test_db_builder.py
git commit -m "feat: wire domain ingest into build_database.py pipeline"
```

---

## Task 5: Replace parquet reads in api/main.py

**Files:**
- Modify: `api/main.py`

Replace all six `pd.read_parquet()` calls in the lifespan function with DuckDB queries. The `app.state` field names and types stay the same — only the source changes.

- [ ] **Write the failing test**

Add to `tests/test_api_contract.py` (or create it if it doesn't exist):

```python
def test_app_state_reconstruction_shapes(tmp_path):
    """Verify app.state arrays have correct shapes after loading from DuckDB."""
    import duckdb
    import numpy as np
    from api.main import _load_type_data_from_db  # function extracted in this task

    J = 4
    N = 3

    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE type_scores (county_fips VARCHAR, type_id INTEGER, score FLOAT, version_id VARCHAR)
    """)
    for fips_i, fips in enumerate(["12001", "12003", "13001"]):
        for j in range(J):
            con.execute("INSERT INTO type_scores VALUES (?,?,?,?)", [fips, j, 1.0/J, "v1"])

    con.execute("""
        CREATE TABLE type_covariance (type_i INTEGER, type_j INTEGER, value FLOAT, version_id VARCHAR)
    """)
    for i in range(J):
        for j in range(J):
            con.execute("INSERT INTO type_covariance VALUES (?,?,?,?)", [i, j, float(i==j)*0.01, "v1"])

    con.execute("""
        CREATE TABLE type_priors (type_id INTEGER, mean_dem_share FLOAT, version_id VARCHAR)
    """)
    for j in range(J):
        con.execute("INSERT INTO type_priors VALUES (?,?,?)", [j, 0.45, "v1"])

    scores, fips_list, covariance, priors = _load_type_data_from_db(con, "v1")
    assert scores.shape == (N, J)
    assert covariance.shape == (J, J)
    assert priors.shape == (J,)
    assert len(fips_list) == N
```

- [ ] **Run to verify it fails**

```bash
python -m pytest tests/test_api_contract.py::test_app_state_reconstruction_shapes -v
```
Expected: `ImportError: cannot import name '_load_type_data_from_db'`

- [ ] **Refactor `api/main.py`**

Extract a helper function `_load_type_data_from_db` and a `_load_hac_weights_from_db` helper, then call them from the lifespan:

```python
def _load_type_data_from_db(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray] | tuple[None, None, None, None]:
    """Load type scores, fips list, covariance, and priors from DuckDB.

    Returns (type_scores, type_county_fips, type_covariance, type_priors)
    or (None, None, None, None) if the tables are empty for this version.
    """
    scores_df = db.execute(
        "SELECT county_fips, type_id, score FROM type_scores WHERE version_id=? ORDER BY county_fips, type_id",
        [version_id],
    ).fetchdf()
    if scores_df.empty:
        return None, None, None, None

    # Pivot: rows=county_fips, cols=type_id (0-indexed contiguous)
    pivot = scores_df.pivot(index="county_fips", columns="type_id", values="score").sort_index()
    pivot = pivot[sorted(pivot.columns)]  # ensure column order 0..J-1
    type_scores = pivot.values.astype(float)
    type_county_fips = pivot.index.tolist()
    J = type_scores.shape[1]

    # Covariance J×J
    cov_df = db.execute(
        "SELECT type_i, type_j, value FROM type_covariance WHERE version_id=? ORDER BY type_i, type_j",
        [version_id],
    ).fetchdf()
    cov_pivot = cov_df.pivot(index="type_i", columns="type_j", values="value").sort_index()
    cov_pivot = cov_pivot[sorted(cov_pivot.columns)]
    type_covariance = cov_pivot.values[:J, :J].astype(float)

    # Priors J-vector
    priors_df = db.execute(
        "SELECT type_id, mean_dem_share FROM type_priors WHERE version_id=? ORDER BY type_id",
        [version_id],
    ).fetchdf()
    type_priors = priors_df["mean_dem_share"].values[:J].astype(float)

    return type_scores, type_county_fips, type_covariance, type_priors


def _load_ridge_priors_from_db(db, version_id: str) -> dict[str, float]:
    df = db.execute(
        "SELECT county_fips, ridge_pred_dem_share FROM ridge_county_priors WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if df.empty:
        return {}
    return dict(zip(df["county_fips"], df["ridge_pred_dem_share"]))


def _load_hac_weights_from_db(
    db: duckdb.DuckDBPyConnection, version_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct state_weights and county_weights DataFrames from DuckDB.

    Returns DataFrames matching the shape that _forecast_poll_hac expects:
    state_weights has columns [state_abbr, community_0, community_1, ...]
    county_weights has columns [county_fips, community_id]
    """
    sw_df = db.execute(
        "SELECT state_abbr, community_id, weight FROM hac_state_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    if sw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    state_weights = sw_df.pivot(index="state_abbr", columns="community_id", values="weight").reset_index()
    state_weights.columns = ["state_abbr"] + [f"community_{i}" for i in state_weights.columns[1:]]

    # `_forecast_poll_hac` only reads county_fips → community_id mapping.
    # Other columns that may exist in the source parquet (state_fips, etc.)
    # are intentionally excluded here.
    cw_df = db.execute(
        "SELECT county_fips, community_id FROM hac_county_weights WHERE version_id=?",
        [version_id],
    ).fetchdf()
    return state_weights, cw_df
```

Replace the parquet reads in `lifespan()` with calls to these helpers. Remove the `pd.read_parquet` calls and path constants for the six files. Keep the graceful fallback logic (log warnings, set to None/empty).

- [ ] **Run to verify test passes**

```bash
python -m pytest tests/test_api_contract.py -v
```
Expected: all tests PASS

- [ ] **Verify the API still starts with a real DB (smoke test)**

```bash
python -m uvicorn api.main:app --port 8001 &
sleep 2
curl -s http://localhost:8001/api/v1/health | python -m json.tool
kill %1
```
Expected: `{"status": "ok", ...}`

- [ ] **Commit**

```bash
git add api/main.py tests/test_api_contract.py
git commit -m "feat: replace parquet reads in api/main.py with DuckDB queries"
```

---

## Task 6: Update /polls and /forecast/polls to query DuckDB

**Files:**
- Modify: `api/routers/forecast.py`
- Modify: `api/tests/conftest.py`
- Modify: `api/tests/test_forecast.py`

Both endpoints currently call `load_polls_with_notes()` which parses a CSV. After this task they query the `polls` table. The weighted Bayesian update logic in `/forecast/polls` is unchanged.

- [ ] **Add `polls` table to the test DB in `conftest.py`**

In `_build_test_db()`, after the existing tables, add:

```python
con.execute("""
    CREATE TABLE polls (
        poll_id   VARCHAR NOT NULL,
        race      VARCHAR NOT NULL,
        geography VARCHAR NOT NULL,
        geo_level VARCHAR NOT NULL,
        dem_share FLOAT   NOT NULL,
        n_sample  INTEGER NOT NULL,
        date      VARCHAR,
        pollster  VARCHAR,
        notes     VARCHAR,
        cycle     VARCHAR NOT NULL,
        PRIMARY KEY (poll_id)
    )
""")
con.execute("""
    INSERT INTO polls VALUES
    ('abc123', 'FL_Senate', 'FL', 'state', 0.45, 600, '2026-01-15', 'Siena', 'grade=A', '2026')
""")
con.execute("CREATE TABLE poll_crosstabs (poll_id VARCHAR, demographic_group VARCHAR, group_value VARCHAR, dem_share FLOAT, n_sample INTEGER)")
con.execute("CREATE TABLE poll_notes (poll_id VARCHAR, note_type VARCHAR, note_value VARCHAR)")
con.execute("INSERT INTO poll_notes VALUES ('abc123', 'grade', 'A')")
```

- [ ] **Write the failing test for `/polls`**

Add to `api/tests/test_forecast.py`:

```python
def test_get_polls_queries_duckdb(client):
    """GET /polls returns rows from DuckDB, not CSV."""
    resp = client.get("/api/v1/polls?cycle=2026&state=FL")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["geography"] == "FL"
    assert data[0]["dem_share"] == pytest.approx(0.45)
```

- [ ] **Run to verify it fails**

```bash
python -m pytest api/tests/test_forecast.py::test_get_polls_queries_duckdb -v
```
Expected: FAIL (either 500 because CSV missing, or empty list if CSV exists but doesn't match)

- [ ] **Update `GET /polls` in `api/routers/forecast.py`**

Replace the `load_polls_with_notes()` call with a DuckDB query:

```python
@router.get("/polls", response_model=list[PollRow])
def get_polls(
    request: Request,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
    race: str | None = Query(None),
    state: str | None = Query(None),
    cycle: str = Query("2026"),
):
    """Return available polls from DuckDB polls table."""
    conditions = ["cycle = ?"]
    params: list = [cycle]
    if race:
        conditions.append("LOWER(race) LIKE ?")
        params.append(f"%{race.lower()}%")
    if state:
        conditions.append("geography = ?")
        params.append(state)
    where = " AND ".join(conditions)
    rows = db.execute(f"SELECT * FROM polls WHERE {where} ORDER BY date", params).fetchdf()
    return [
        PollRow(
            race=row["race"],
            geography=row["geography"],
            geo_level=row["geo_level"],
            dem_share=float(row["dem_share"]),
            n_sample=int(row["n_sample"]),
            date=row["date"] if row["date"] else None,
            pollster=row["pollster"] if row["pollster"] else None,
        )
        for _, row in rows.iterrows()
    ]
```

Note: add `request: Request` parameter and the `db` dependency so the endpoint has access to DuckDB. Remove `from src.propagation.poll_weighting import load_polls_with_notes` from this endpoint (keep it in the `/forecast/polls` handler — addressed next).

- [ ] **Run to verify test passes**

```bash
python -m pytest api/tests/test_forecast.py::test_get_polls_queries_duckdb -v
```
Expected: PASS

- [ ] **Update `POST /forecast/polls`**

The `/forecast/polls` handler currently calls `load_polls_with_notes()` at the top. Replace that block with a DuckDB query that reconstructs `(polls, notes)` tuples:

```python
# Replace the load_polls_with_notes() call with:
rows_df = db.execute(
    "SELECT * FROM polls WHERE cycle=? ORDER BY date",
    [body.cycle],
).fetchdf()

if body.state:
    rows_df = rows_df[rows_df["geography"] == body.state]
if body.race:
    rows_df = rows_df[rows_df["race"].str.contains(body.race, case=False, na=False)]

if rows_df.empty:
    raise HTTPException(
        status_code=404,
        detail=f"No matching polls for cycle={body.cycle}, state={body.state}, race={body.race}",
    )

from src.propagation.propagate_polls import PollObservation
polls = [
    PollObservation(
        geography=row["geography"],
        dem_share=float(row["dem_share"]),
        n_sample=int(row["n_sample"]),
        race=row["race"],
        date=row["date"] or "",
        pollster=row["pollster"] or "",
        geo_level=row["geo_level"],
    )
    for _, row in rows_df.iterrows()
]
notes = list(rows_df["notes"].fillna(""))
```

The rest of the handler (weighting, predict_race) is unchanged.

- [ ] **Run all API tests**

```bash
python -m pytest api/tests/ -v
```
Expected: all tests PASS (or pre-existing failures unrelated to this change)

- [ ] **Commit**

```bash
git add api/routers/forecast.py api/tests/conftest.py api/tests/test_forecast.py
git commit -m "feat: /polls and /forecast/polls query DuckDB instead of CSV"
```

---

## Task 7: Full integration smoke test and cleanup

**Files:**
- Run existing test suites

- [ ] **Run the full test suite**

```bash
python -m pytest -v
```
Expected: all tests PASS. Note any pre-existing failures to confirm they are not regressions.

- [ ] **Verify no remaining `pd.read_parquet` calls in `api/`**

```bash
grep -rn "read_parquet" api/
```
Expected: no output (all parquet reads removed from API layer)

- [ ] **Verify no remaining `load_polls_with_notes` calls in `api/`**

```bash
grep -rn "load_polls_with_notes" api/
```
Expected: no output

- [ ] **Commit (if any final cleanup was needed)**

```bash
git add -A
git commit -m "chore: remove legacy parquet reads from API layer"
```

---

## Reference: app.state fields before and after

| Field | Before (parquet) | After (DuckDB) |
|---|---|---|
| `type_scores` | `ta_df[score_cols].values` | `pivot(county_fips × type_id).values` |
| `type_county_fips` | `ta_df["county_fips"].tolist()` | `pivot.index.tolist()` |
| `type_covariance` | `cov_df.values[:J,:J]` | `cov_pivot.values[:J,:J]` |
| `type_priors` | `profiles["mean_dem_share"].values[:J]` | `priors_df["mean_dem_share"].values[:J]` |
| `ridge_priors` | `dict(zip(fips, ridge_pred_dem_share))` | same, from DuckDB query |
| `state_weights` | `pd.read_parquet(state_w_path)` | `pivot(state_abbr × community_id)` |
| `county_weights` | `pd.read_parquet(county_w_path)` | `SELECT county_fips, community_id` |
