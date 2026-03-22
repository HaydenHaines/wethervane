# API–Frontend Contract Design

**Date:** 2026-03-21
**Status:** Approved
**Problem:** Model pipeline changes have broken the frontend 4 times. Each incident burned tokens diagnosing silent failures — empty tables, wrong labels, null demographics — because the frontend hardcodes assumptions about model output shape.

**Principle:** The API is the contract boundary. The frontend reads everything from the API and hardcodes nothing about the model. Only new features or schema additions can break the frontend — not changes to J, super-type count, type names, demographics columns, or race strings.

---

## 1. API Contract

The API guarantees these response shapes. **Required** fields are always present and non-null. **Dynamic** fields vary in count/content across model runs.

### `GET /api/v1/super-types`

The single source of truth for legend rendering. The frontend must not hardcode names, counts, or color assignments for super-types.

```typescript
interface SuperTypeSummary {
  super_type_id: number;     // required — stable integer key
  display_name: string;      // required — human-readable label
  member_type_ids: number[]; // required — dynamic length
  n_counties: number;        // required
}
```

### `GET /api/v1/counties`

```typescript
interface CountyRow {
  county_fips: string;       // required
  state_abbr: string;        // required
  community_id: number;      // required
  dominant_type: number | null;  // nullable — valid key into types if present
  super_type: number | null;     // nullable — valid key into super-types if present
}
```

**Referential integrity guarantee:** Every non-null `super_type` value returned here exists as a `super_type_id` in the `/super-types` response. Every non-null `dominant_type` exists as a `type_id` in the `/types` response.

### `GET /api/v1/types`

```typescript
interface TypeSummary {
  type_id: number;           // required
  super_type_id: number;     // required — valid key into super-types
  display_name: string;      // required
  n_counties: number;        // required
  mean_pred_dem_share: number | null;
}
```

### `GET /api/v1/types/{id}`

```typescript
interface TypeDetail extends TypeSummary {
  counties: string[];                    // FIPS codes, dynamic length
  demographics: Record<string, number>;  // dynamic keys — render all of them
  shift_profile: Record<string, number> | null;  // dynamic keys — render all of them
}
```

**Dynamic dict contract:** `demographics` and `shift_profile` are generic key-value maps with **numeric values only**. The API must filter out non-numeric metadata columns (e.g., `version_id`, `county_fips`) before returning these dicts. The frontend renders whatever keys are present. New demographic features (e.g., FEC donors, urbanicity) appear automatically without frontend changes.

**Implementation note:** The shift profile query in `communities.py` must exclude `version_id` from the column list (in addition to `county_fips`). Both the `get_community` and `get_type` endpoints must apply this filter.

### `GET /api/v1/forecast`

```typescript
interface ForecastRow {
  county_fips: string;       // required
  county_name: string | null;
  state_abbr: string;        // required — frontend groups by this, never parses race
  race: string;              // required — human-readable label, displayed as-is
  pred_dem_share: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
  state_pred: number | null;
  poll_avg: number | null;
}
```

**Race string contract:** `race` is an opaque display string (e.g., `"2026 FL Governor"`). The frontend never parses, splits, or extracts substrings from it. State filtering uses `state_abbr`.

### Empty-state guarantee

If a table is missing or empty, the API returns `[]` (empty list), not a 500. The frontend renders a "no data" state. No silent nulls that look like working-but-empty.

---

## 2. Data-Driven Frontend

### What gets deleted from MapShell.tsx

| Constant | Lines | Why |
|----------|-------|-----|
| `COUNTY_SUPER_TYPE_NAMES` | hardcoded 5 names | Names come from `/super-types` API |
| `TRACT_SUPER_TYPE_NAMES` | hardcoded 10 names | Names come from `/super-types` API |
| `SUPER_TYPE_COLORS` | hardcoded 10 RGB values | Colors assigned dynamically from palette |
| `COMMUNITY_COLORS` | hardcoded 10 legacy colors | Dead code path, pre-type era |
| `export let SUPER_TYPE_NAMES` | mutable global | Replaced by React state |

### What replaces them

**Color palette:** A single const array of 15 perceptually-distinct RGB colors, indexed by `super_type_id`. This is purely a visual concern — the palette defines "color 0 is blue, color 1 is orange" etc. It lives in the frontend because it's a rendering decision, not a model decision.

```typescript
// 15-color palette covering super_type_id 0..14
// Covers any realistic super-type count (2-15)
// If > 15 super-types, cycles with modulo
const PALETTE: [number, number, number][] = [
  [31, 119, 180],   // blue
  [255, 127, 14],   // orange
  [44, 160, 44],    // green
  [214, 39, 40],    // red
  [148, 103, 189],  // purple
  [140, 86, 75],    // brown
  [227, 119, 194],  // pink
  [127, 127, 127],  // gray
  [188, 189, 34],   // olive
  [23, 190, 207],   // teal
  [174, 199, 232],  // light blue
  [255, 187, 120],  // light orange
  [152, 223, 138],  // light green
  [255, 152, 150],  // light red
  [197, 176, 213],  // light purple
];
```

**Color stability:** Colors are assigned by `super_type_id`, not by array position in the response. `super_type_id=0` always gets `PALETTE[0]`, regardless of how many super-types exist or their response order. If a super-type is removed, other super-types keep their colors. Users could later configure custom mappings if needed.

**Data flow:**

```
useEffect on mount:
  1. fetch /super-types → build superTypeMap: Map<id, {name, color}>
  2. fetch /counties → build countyMap: Map<fips, CountyRow>
  3. Enrich GeoJSON features with countyMap data
  4. Legend entries = Array.from(superTypeMap.values())
```

All downstream rendering (getColor, getLineWidth, tooltip, legend) reads from `superTypeMap`. If the map is empty, counties render in neutral gray and the legend is hidden.

### ForecastView.tsx fix

Current bug:
```typescript
const selectedState = selectedRace.split("_")[0]; // BROKEN: "2026 FL Governor" → "2026 FL Governor"
```

Fix: derive available states from the data, don't parse race strings.
```typescript
// Group forecast rows by state_abbr (from API, not parsed from race)
const states = [...new Set(displayRows.map(r => r.state_abbr))].sort();
// selectedState is picked from this list, or defaults to first
```

### TypePanel.tsx — generic demographics rendering

Current: hardcoded list of 7 demographic rows with manual labels.

New: render all keys from `demographics` dict with a formatter lookup.

```typescript
// Label and format lookup — keys not in this map render as-is with default format
const DEMO_DISPLAY: Record<string, { label: string; fmt: "pct" | "dollar" | "num" }> = {
  median_hh_income: { label: "Median income", fmt: "dollar" },
  median_age: { label: "Median age", fmt: "num" },
  pct_white_nh: { label: "White (non-Hispanic)", fmt: "pct" },
  pct_black: { label: "Black", fmt: "pct" },
  pct_hispanic: { label: "Hispanic", fmt: "pct" },
  pct_asian: { label: "Asian", fmt: "pct" },
  pct_bachelors_plus: { label: "Bachelor's+", fmt: "pct" },
  pct_owner_occupied: { label: "Owner-occupied", fmt: "pct" },
  pct_wfh: { label: "Work from home", fmt: "pct" },
  evangelical_share: { label: "Evangelical", fmt: "pct" },
  // ... etc — new keys render automatically with raw key as label
};
```

Unknown keys (e.g., a new `fec_donors_per_1000` feature) render with the raw key prettified (`fec_donors_per_1000` → `"Fec donors per 1000"`) and default number format. This means adding demographic features to the pipeline never requires a frontend change — though you can optionally add a display entry for a nicer label.

---

## 3. Pipeline Validation

### build_database.py exit validation

After all tables are built, before `con.close()`, run a contract check:

```python
def validate_contract(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Return list of contract violations. Empty = pass."""
    errors = []

    # Required tables with required columns
    required = {
        "super_types": ["super_type_id", "display_name"],
        "types": ["type_id", "super_type_id", "display_name"],
        "county_type_assignments": ["county_fips", "dominant_type", "super_type"],
        "counties": ["county_fips", "state_abbr", "county_name"],
    }

    # Optional tables — validated if present, not required to exist
    optional = {
        "predictions": ["county_fips", "race", "pred_dem_share"],
    }

    for table, columns in required.items():
        # Table exists?
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            errors.append(f"MISSING TABLE: {table}")
            continue
        # Required columns?
        actual_cols = set(
            con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns
        )
        for col in columns:
            if col not in actual_cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    # Optional tables — validate columns if table exists
    for table, columns in optional.items():
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if exists:
            actual_cols = set(
                con.execute(f"SELECT * FROM {table} LIMIT 0").fetchdf().columns
            )
            for col in columns:
                if col not in actual_cols:
                    errors.append(f"MISSING COLUMN: {table}.{col}")

    # Referential integrity: super_type values exist in super_types
    if not errors:  # only check if tables exist
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
```

Called at the end of `build()`:
```python
errors = validate_contract(con)
if errors:
    for e in errors:
        log.error("CONTRACT VIOLATION: %s", e)
    con.close()
    sys.exit(1)
log.info("Contract validation passed")
```

### API startup validation

In `main.py` lifespan, after opening DuckDB:

```python
contract_ok = True
for table in ["super_types", "types", "county_type_assignments"]:
    if not _has_table(app.state.db, table):
        log.warning("CONTRACT: missing table %s — frontend will show degraded state", table)
        contract_ok = False
app.state.contract_ok = contract_ok
```

The `/health` endpoint reports `"degraded"` if `contract_ok` is False. The API still starts and serves what it can.

---

## 4. Integration Test

File: `tests/test_api_contract.py`

Uses the real DuckDB (not in-memory fixture) to validate the full pipeline → API chain.

### Test 1: DuckDB schema

```python
def test_duckdb_contract():
    """Required tables and columns exist in bedrock.duckdb."""
    from src.db.build_database import validate_contract
    con = duckdb.connect("data/bedrock.duckdb", read_only=True)
    errors = validate_contract(con)
    con.close()
    assert errors == [], f"Contract violations: {errors}"
```

### Test 2: API response shapes

```python
def test_super_types_response(client):
    """Super-types endpoint returns non-empty list with required fields."""
    resp = client.get("/api/v1/super-types")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) > 0, "super-types must not be empty"
    for st in data:
        assert "super_type_id" in st
        assert "display_name" in st
        assert isinstance(st["display_name"], str)
        assert len(st["display_name"]) > 0

def test_counties_reference_valid_super_types(client):
    """Every county super_type exists in the super-types response."""
    super_types = {st["super_type_id"] for st in client.get("/api/v1/super-types").json()}
    counties = client.get("/api/v1/counties").json()
    for c in counties:
        if c["super_type"] is not None:
            assert c["super_type"] in super_types, \
                f"County {c['county_fips']} has super_type={c['super_type']} not in super-types"

def test_forecast_has_state_abbr(client):
    """Every forecast row has state_abbr so frontend can group without parsing race."""
    rows = client.get("/api/v1/forecast").json()
    for r in rows:
        assert "state_abbr" in r
        assert isinstance(r["state_abbr"], str)
        assert len(r["state_abbr"]) == 2

def test_type_detail_has_dynamic_dicts(client):
    """Type detail returns demographics and shift_profile as dicts."""
    types = client.get("/api/v1/types").json()
    if not types:
        pytest.skip("No types in database")
    detail = client.get(f"/api/v1/types/{types[0]['type_id']}").json()
    assert isinstance(detail["demographics"], dict)
    # shift_profile can be null but not a non-dict type
    if detail["shift_profile"] is not None:
        assert isinstance(detail["shift_profile"], dict)
```

### Test 3: Cross-layer consistency

```python
def test_super_type_coverage(client):
    """Every county super_type exists in the super-types list."""
    super_type_ids = {st["super_type_id"] for st in client.get("/api/v1/super-types").json()}
    county_super_types = {c["super_type"] for c in client.get("/api/v1/counties").json() if c["super_type"] is not None}
    # Hard check: counties must not reference nonexistent super-types
    assert county_super_types <= super_type_ids, \
        f"Counties reference super-types not in API: {county_super_types - super_type_ids}"
    # Soft check: warn if super-types exist with no counties (degenerate model run)
    unused = super_type_ids - county_super_types
    if unused:
        import warnings
        warnings.warn(f"Super-types with no counties: {unused}")
```

---

## Scope Boundaries

### Community endpoints (out of scope)

The community endpoints (`/communities`, `/communities/{id}`) and `CommunityPanel.tsx` are legacy from the HAC pipeline. They are not addressed by this spec because:
- The type-primary architecture (ADR-006) superseded communities as the primary model unit
- `CommunityDemographics` in `lib/api.ts` is a hardcoded struct, but it only serves the legacy path
- When communities are removed or refactored, their coupling should be cleaned up then — not as part of this contract work

### Tract toggle

The tract toggle in MapShell.tsx loads data from a static GeoJSON file (`tract-communities.geojson`), not from the API. Tract super-type names and IDs are embedded as GeoJSON feature properties (`super_type`, `type_id`).

For this spec: the tract view reads names from GeoJSON `properties.super_type_name` (a string property that the tract GeoJSON generator must include). The frontend palette assigns colors by `super_type` ID, same as county view. If the tract GeoJSON lacks `super_type_name`, the frontend falls back to `"Type {id}"`. No hardcoded tract names in the frontend.

The tract GeoJSON generator (`src/viz/bubble_dissolve.py`) must include `super_type_name` as a property when building the GeoJSON. This is a one-line addition.

---

## Summary of Changes

| Layer | File(s) | Change |
|-------|---------|--------|
| Frontend | `MapShell.tsx` | Delete all hardcoded names/colors. Fetch super-types from API, build color/name maps dynamically. Palette const (15 colors) stays as visual concern. |
| Frontend | `ForecastView.tsx` | Group by `state_abbr` from data, never parse `race` string. |
| Frontend | `TypePanel.tsx` | Render demographics generically from dict with formatter lookup. Unknown keys auto-render. |
| Frontend | `lib/api.ts` | No changes — types already correct. |
| API | `api/routers/communities.py` | Already fixed (S163). No further changes. |
| API | `api/main.py` | Add startup contract check, report degraded health if tables missing. |
| Pipeline | `src/db/build_database.py` | Add `validate_contract()` at end of build. Exit non-zero on violations. |
| Tests | `tests/test_api_contract.py` | New file: schema check, response shape check, referential integrity check. |

## What This Prevents

- Changing J (number of types) → frontend auto-adjusts legend and colors
- Renaming super-types → frontend reads names from API
- Adding/removing demographic features → TypePanel renders all keys
- Changing race string format → frontend uses `state_abbr`, ignores race format
- Partial DuckDB build → build fails loudly, API reports degraded health
- Missing tables → API returns `[]`, frontend shows empty state gracefully
