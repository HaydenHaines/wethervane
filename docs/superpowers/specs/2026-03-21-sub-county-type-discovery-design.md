# Design Spec: Sub-County Type Discovery with Experiment Framework

**Date:** 2026-03-21
**Status:** Draft
**Depends on:** Type-primary architecture (ADR-006), VEST precinct data, NYTimes precinct data

---

## Problem Statement

The county-level model (J=20 types, 293 counties) produces meaningful electoral types, but counties are coarse units. A single county can contain suburban professionals, rural farmland, a Black Belt town, and a college campus — all assigned to one type. Sub-county resolution reveals the actual communities within counties.

The goal: discover electoral types at census tract level (9,393 tracts in FL+GA+AL), then merge adjacent same-type tracts into community polygons — a "bubble dissolve" that produces emergent geographic communities.

---

## Design Goals

1. **Independent tract-level type discovery** — types discovered from tract data, not inherited from counties
2. **Configurable experiment framework** — swap features, weights, algorithms, and J via YAML config without code changes
3. **Two foundational experiments:** political-only features vs nonpolitical-only features, compared side-by-side
4. **Bubble dissolve visualization** — adjacent same-type tracts merge into community polygons that approximate real-life communities
5. **J=30-40 types** — more granular than the county model (J=20), enabled by 9,393 spatial units
6. **Reproducible and versioned** — every experiment run produces a timestamped output directory with config, assignments, validation, and comparison reports

---

## Architecture

### Pipeline

```
1. Areal Interpolation (precinct → tract)
   ├── Load precinct shapefiles (VEST 2016/2018/2020, NYTimes 2024)
   ├── Load census tract shapefiles (TIGER/Line 2020)
   ├── Geometric intersection: allocate votes proportionally by area overlap
   └── Output: tract-level Dem/Rep/Total per election
       → data/tracts/tract_votes_{year}.parquet

2. Tract Feature Engineering
   ├── Electoral features: shifts, lean, turnout, density, house/senate shifts
   ├── Nonpolitical features: demographics, housing, religion proxy, etc.
   ├── Feature registry: each feature has a name, source, and category tag
   └── Output: data/tracts/tract_features.parquet (9,393 × ~60 columns)

3. Experiment Runner (configurable via YAML)
   ├── Read config → select features by category → apply weights
   ├── Scale features (min-max to [0,1])
   ├── Run clustering (KMeans / HAC / GMM / HDBSCAN)
   ├── Holdout validation (leave-one-shift-pair-out)
   ├── Bubble dissolve (merge adjacent same-type tracts)
   └── Output: data/experiments/{name}/
       ├── config.yaml (frozen copy)
       ├── assignments.parquet (tract_fips, type, super_type)
       ├── dissolved_communities.geojson (merged polygons)
       ├── validation.json (holdout r, coherence, stability)
       └── type_profiles.parquet (demographic profile per type)

4. Comparison Framework
   ├── Load two experiment runs
   ├── Compute overlap metrics (Adjusted Rand Index, Jaccard, NMI)
   ├── Visual side-by-side map
   └── Output: data/experiments/comparisons/{run_a}_vs_{run_b}.json
```

### Experiment Config Schema

```yaml
experiment:
  name: string                    # unique identifier
  description: string             # human-readable description

geography:
  level: tract                    # tract or county
  states: [FL, GA, AL]
  tract_shapefile: data/raw/tiger/tl_2020_tract.shp  # TIGER/Line source

features:
  # Each category can be enabled/disabled independently
  # Weight multiplier applies to all features in the category
  electoral:
    enabled: true
    weight: 1.0
    include:
      presidential_shifts: true       # 2016→2020, 2020→2024 (D, R, turnout) = 6 dims
      presidential_lean: true         # Dem share 2016, 2020, 2024 = 3 dims
      turnout_level: true             # Turnout rate 2016, 2020, 2024 = 3 dims
      turnout_shift: true             # Turnout shift 16→20, 20→24 = 2 dims
      vote_density: true              # Votes/sq mi 2020, 2024 = 2 dims
      house_shifts: true              # State-centered, 16→18, 18→20 = 4 dims
      senate_shifts: true             # State-centered, available cycles = 3 dims
      split_ticket: true              # Pres vs house divergence 2016, 2020 = 2 dims
      governor_shift: false           # Needs 2022 precinct data (not yet available)
      donor_density: true             # FEC county proxy = 1 dim
      state_center_nonpresidential: true  # Remove state mean from non-pres shifts

  demographic:
    enabled: true
    weight: 1.0
    include:
      race_ethnicity: true            # white NH, Black, Hispanic, Asian = 4 dims
      white_working_class: true       # white × no-BA interaction = 1 dim
      foreign_born: true              # % foreign-born = 1 dim
      income: true                    # median HH income, poverty rate, Gini = 3 dims
      education: true                 # % BA+, % graduate, % no HS diploma = 3 dims
      housing: true                   # owner-occ, median value, % multi-unit, % pre-1960 = 4 dims
      rent_burden: true               # median rent / median income = 1 dim
      age_household: true             # median age, % <18, % >65, % single-person = 4 dims
      commute: true                   # % WFH, mean commute, % no vehicle = 3 dims
      military: true                  # % veteran = 1 dim

  religion:
    enabled: true
    weight: 1.0
    include:
      rcms_proxy: true                # County-level RCMS allocated to tracts = 4 dims
      # evangelical_share, catholic_share, black_protestant_share, adherence_rate

clustering:
  algorithm: kmeans                   # kmeans, hac, gmm, hdbscan
  j_candidates: [25, 30, 35, 40]
  j_selection: holdout_cv             # holdout_cv, silhouette, bic, manual
  n_init: 10                          # KMeans restarts
  random_state: 42
  presidential_weight: 1.0            # multiplier on presidential shift dims (county model uses 2.5)
  min_tracts_per_type: 50             # minimum type population (0.5% of 9,393)

nesting:
  enabled: true
  s_candidates: [8, 10, 12, 15]
  method: ward_hac                    # on type centroids, no spatial constraint

visualization:
  bubble_dissolve: true
  min_polygon_area_sqkm: 0.1         # dissolve slivers smaller than this
  simplify_tolerance: 0.001          # Douglas-Peucker tolerance for polygon simplification

holdout:
  pairs:
    - [2020, 2024]                    # hold out presidential 20→24 shift
  metric: pearson_r
  min_threshold: 0.5                  # minimum acceptable holdout r
  # IMPORTANT: holdout exclusion rule — when holding out a shift pair (e.g., 2020→2024),
  # ALL features derived from the later year (2024) must be excluded from clustering.
  # This means presidential_lean_2024, turnout_level_2024, vote_density_2024, etc.
  # are removed from the feature matrix during holdout validation. The experiment
  # runner enforces this automatically by checking each feature's source year against
  # the holdout pair's end year.
```

---

## Areal Interpolation: Technical Details

The core challenge: precinct boundaries ≠ tract boundaries. A precinct may span parts of 3 tracts, and a tract may contain parts of 5 precincts.

### Algorithm

```python
def interpolate_precincts_to_tracts(
    precinct_gdf: gpd.GeoDataFrame,  # geometry + votes_dem, votes_rep, votes_total
    tract_gdf: gpd.GeoDataFrame,     # geometry + GEOID (tract FIPS)
) -> pd.DataFrame:
    """Allocate precinct votes to tracts proportional to area overlap."""

    # 1. Ensure same CRS (reproject to equal-area for accurate overlap)
    precinct_gdf = precinct_gdf.to_crs("EPSG:5070")  # NAD83 Conus Albers
    tract_gdf = tract_gdf.to_crs("EPSG:5070")

    # 2. Spatial overlay (intersection)
    overlay = gpd.overlay(precinct_gdf, tract_gdf, how="intersection")

    # 3. Compute area fractions
    overlay["overlap_area"] = overlay.geometry.area
    precinct_areas = precinct_gdf.set_index("precinct_id").geometry.area
    overlay["precinct_area"] = overlay["precinct_id"].map(precinct_areas)
    overlay["area_fraction"] = overlay["overlap_area"] / overlay["precinct_area"]

    # 4. Allocate votes
    for col in ["votes_dem", "votes_rep", "votes_total"]:
        overlay[f"{col}_allocated"] = overlay[col] * overlay["area_fraction"]

    # 5. Aggregate to tract level
    tract_votes = overlay.groupby("GEOID").agg(
        votes_dem=("votes_dem_allocated", "sum"),
        votes_rep=("votes_rep_allocated", "sum"),
        votes_total=("votes_total_allocated", "sum"),
    ).reset_index()

    return tract_votes
```

### Data Sources for Each Election

| Election | Source | Format | Coverage |
|----------|--------|--------|----------|
| 2016 Presidential | VEST | Shapefile | FL, GA, AL |
| 2016 House | VEST | Shapefile | FL, GA, AL |
| 2018 Governor | VEST | Shapefile | FL, GA, AL |
| 2018 House/Senate | VEST | Shapefile | FL, GA, AL |
| 2020 Presidential | VEST + NYTimes | Shapefile + GeoJSON | FL, GA (AL: VEST only) |
| 2020 House | VEST | Shapefile | FL, GA, AL |
| 2024 Presidential | NYTimes | TopoJSON + CSV | FL, GA, AL |

### TIGER/Line Tract Shapefiles

Census tract boundaries from TIGER/Line 2020. Download per state:
```
https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{fips}_tract.zip
```
States: 01 (AL), 12 (FL), 13 (GA)

### Edge Cases

- **Water-only tracts**: exclude (zero population, no votes)
- **Zero-overlap tracts**: tracts with no precinct overlap get zero votes (rare, usually water/park)
- **Precinct-tract misalignment**: some precincts extend beyond state boundaries or have topology errors. Use `buffer(0)` to fix invalid geometries before overlay.
- **AL 2020 from NYTimes is unusable** (absentee reported countywide). Use VEST 2020 for AL instead.
- **Uncontested races**: drop from shift computation (same as county model). Retain turnout.

---

## Tract Feature Engineering

### Feature Registry

All features are registered with metadata so the experiment config can select them by category:

```python
FEATURE_REGISTRY = {
    # Electoral features
    "pres_d_shift_16_20": {"category": "electoral", "subcategory": "presidential_shifts", "source": "vest+nyt"},
    "pres_r_shift_16_20": {"category": "electoral", "subcategory": "presidential_shifts", "source": "vest+nyt"},
    "pres_turnout_shift_16_20": {"category": "electoral", "subcategory": "turnout_shift", "source": "vest+nyt"},
    "pres_dem_share_2024": {"category": "electoral", "subcategory": "presidential_lean", "source": "nyt"},
    "turnout_2024": {"category": "electoral", "subcategory": "turnout_level", "source": "nyt"},
    "vote_density_2024": {"category": "electoral", "subcategory": "vote_density", "source": "nyt+tiger"},
    "house_d_shift_16_18_sc": {"category": "electoral", "subcategory": "house_shifts", "source": "vest"},
    # ... etc

    # Demographic features
    "pct_white_nh": {"category": "demographic", "subcategory": "race_ethnicity", "source": "acs_tract"},
    "pct_black": {"category": "demographic", "subcategory": "race_ethnicity", "source": "acs_tract"},
    "pct_wwc": {"category": "demographic", "subcategory": "white_working_class", "source": "acs_tract"},
    "median_hh_income": {"category": "demographic", "subcategory": "income", "source": "acs_tract"},
    "gini": {"category": "demographic", "subcategory": "income", "source": "acs_tract"},
    "pct_veteran": {"category": "demographic", "subcategory": "military", "source": "acs_tract"},
    # ... etc

    # Religion features
    "evangelical_share": {"category": "religion", "subcategory": "rcms_proxy", "source": "rcms_county"},
    # ... etc
}
```

### State-Centering for Non-Presidential Features

Same approach as county model: subtract state mean from governor, house, and senate shifts before clustering. Presidential shifts are left raw (same race everywhere).

```python
for col in nonpresidential_shift_cols:
    for state_fips in ["01", "12", "13"]:
        mask = tracts["state_fips"] == state_fips
        tracts.loc[mask, col] -= tracts.loc[mask, col].mean()
```

### Missing Data Handling

Some tracts may have missing values (zero population, suppressed ACS estimates):
- Tracts with zero population: exclude entirely
- Tracts with population < 50: exclude (insufficient for reliable estimates)
- ACS estimates with high MOE: keep but flag. KMeans handles mild noise.
- **RCMS county proxy**: all tracts in a county get the same 4 religion values. This creates artificial within-county homogeneity that biases KMeans toward county-shaped clusters for those dimensions. Known limitation — same issue that caused tract-level HAC to fail (r=-0.14) when most shift dims were county-level MEDSL data. Mitigated by being only 4 of ~29 nonpolitical dims. Consider weighting religion features at 0.5× in nonpolitical-only runs.
- **Senate/governor shift availability varies by state**: FL had Senate races in 2016, 2018; GA in 2016, 2020, 2022; AL in 2016, 2020. When a shift pair is unavailable for a state, those columns are set to 0 for that state's tracts (neutral — no shift). This is equivalent to state-centering when only one state has data.
- **Areal interpolation assumes uniform population density within precincts.** A precinct that is 80% farmland and 20% subdivision will over-allocate votes to farmland tracts. This is a known limitation of area-weighted interpolation. Future improvement: dasymetric refinement using Census block population as a weight surface. For now, the uniform assumption is standard practice and acceptable for initial experiments.

---

## Bubble Dissolve Visualization

### Algorithm

After type assignment, merge adjacent same-type tracts into community polygons:

```python
def bubble_dissolve(
    tract_gdf: gpd.GeoDataFrame,  # geometry + dominant_type
    min_area_sqkm: float = 0.1,
) -> gpd.GeoDataFrame:
    """Merge adjacent same-type tracts into community polygons."""

    # 1. Dissolve by dominant_type, keeping only contiguous groups
    #    (geopandas dissolve merges ALL same-type, even non-adjacent)
    #    Need to first find connected components of same-type tracts

    import networkx as nx
    from shapely.ops import unary_union

    # Build adjacency graph (Queen contiguity)
    from libpysal.weights import Queen
    w = Queen.from_dataframe(tract_gdf)

    # For each type, find connected components of adjacent same-type tracts
    communities = []
    for type_id in tract_gdf["dominant_type"].unique():
        type_mask = tract_gdf["dominant_type"] == type_id
        type_indices = tract_gdf[type_mask].index.tolist()

        # Subgraph of same-type adjacency
        G = nx.Graph()
        G.add_nodes_from(type_indices)
        for i in type_indices:
            for j in w.neighbors[i]:
                if j in type_indices:
                    G.add_edge(i, j)

        # Each connected component = one community polygon
        for component in nx.connected_components(G):
            component_tracts = tract_gdf.loc[list(component)]
            merged_geom = unary_union(component_tracts.geometry)
            area_sqkm = merged_geom.area / 1e6  # if CRS is meters

            if area_sqkm >= min_area_sqkm:
                communities.append({
                    "geometry": merged_geom,
                    "type_id": type_id,
                    "super_type": component_tracts["super_type"].mode().iloc[0],
                    "n_tracts": len(component),
                    "area_sqkm": area_sqkm,
                })

    return gpd.GeoDataFrame(communities, crs=tract_gdf.crs)
```

### Expected Output

With J=35 types and 9,393 tracts, the bubble dissolve will produce roughly 500-2,000 community polygons (depending on how fragmented the types are). Rural types will form large polygons spanning many tracts. Urban types will form small, interleaved polygons.

The stained glass map at tract level will look qualitatively different from county level:
- County borders disappear — communities flow across county lines
- Urban areas show internal structure (downtown core vs inner suburbs vs outer ring)
- Rural areas may look similar to county level (less tract-level variation)
- The Black Belt will emerge as a continuous ribbon, not county-by-county chunks

---

## Comparison Framework

### Metrics

For comparing two experiment runs (e.g., political-only vs nonpolitical-only):

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compare_runs(
    assignments_a: np.ndarray,  # tract dominant type from run A
    assignments_b: np.ndarray,  # tract dominant type from run B
) -> dict:
    return {
        "adjusted_rand_index": adjusted_rand_score(assignments_a, assignments_b),
        "normalized_mutual_info": normalized_mutual_info_score(assignments_a, assignments_b),
        "n_types_a": len(np.unique(assignments_a)),
        "n_types_b": len(np.unique(assignments_b)),
    }
```

- **Adjusted Rand Index > 0.5**: strong agreement — political and nonpolitical features find similar communities
- **ARI 0.2-0.5**: moderate agreement — some overlap, meaningful divergence
- **ARI < 0.2**: weak agreement — features find fundamentally different structures

### Visual Comparison

Side-by-side dissolved community maps:
- Left: political-only types (colored by super-type)
- Right: nonpolitical-only types (colored by super-type)
- Overlay: highlight tracts where the two runs disagree

---

## Foundational Experiments

### Run 1: Political Features Only (~27 dims)

```yaml
experiment:
  name: tract_political_only
  description: "Types from electoral behavior — shifts, lean, turnout, density"

features:
  electoral:
    enabled: true
    weight: 1.0
    include:
      presidential_shifts: true       # 6 dims
      presidential_lean: true         # 3 dims
      turnout_level: true             # 3 dims
      turnout_shift: true             # 2 dims
      vote_density: true              # 2 dims
      house_shifts: true              # 4 dims (state-centered)
      senate_shifts: true             # 3 dims (state-centered)
      split_ticket: true              # 2 dims
      donor_density: true             # 1 dim (county proxy)
      governor_shift: false
      state_center_nonpresidential: true
  demographic:
    enabled: false
  religion:
    enabled: false

clustering:
  algorithm: kmeans
  j_candidates: [25, 30, 35, 40]
  presidential_weight: 2.5
```

### Run 2: Nonpolitical Features Only (~29 dims)

```yaml
experiment:
  name: tract_nonpolitical_only
  description: "Types from social structure — demographics, housing, religion"

features:
  electoral:
    enabled: false
  demographic:
    enabled: true
    weight: 1.0
    include:
      race_ethnicity: true            # 4 dims
      white_working_class: true       # 1 dim
      foreign_born: true              # 1 dim
      income: true                    # 3 dims
      education: true                 # 3 dims
      housing: true                   # 4 dims
      rent_burden: true               # 1 dim
      age_household: true             # 4 dims
      commute: true                   # 3 dims
      military: true                  # 1 dim
  religion:
    enabled: true
    weight: 1.0
    include:
      rcms_proxy: true                # 4 dims

clustering:
  algorithm: kmeans
  j_candidates: [25, 30, 35, 40]
```

### Success Criteria

1. **Both runs produce 30+ viable types** from 9,393 tracts
2. **Bubble dissolve produces recognizable communities** — visual inspection confirms emergent geographic structure
3. **Comparison ARI > 0.3** — political and nonpolitical features find overlapping but not identical communities
4. **Holdout r > 0.5** for political-only run (predicting 2020→2024 presidential shift)
5. **Nonpolitical-only run predicts political shifts at r > 0.3** — the Bedrock hypothesis holds at tract level

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/tracts/interpolate_precincts.py` | Areal interpolation: precinct → tract vote allocation |
| `src/tracts/build_tract_features.py` | Compute all tract-level features (electoral + demographic) |
| `src/tracts/feature_registry.py` | Feature metadata registry for config-driven selection |
| `src/experiments/run_experiment.py` | Config-driven experiment runner |
| `src/experiments/compare_runs.py` | Side-by-side comparison of two experiment runs |
| `src/viz/bubble_dissolve.py` | Merge adjacent same-type tracts into community polygons |
| `scripts/fetch_tiger_tracts.py` | Download TIGER/Line tract shapefiles |
| `tests/test_interpolation.py` | Areal interpolation tests |
| `tests/test_experiment_runner.py` | Experiment framework tests |
| `tests/test_bubble_dissolve.py` | Dissolve algorithm tests |

### Experiment Output Structure

```
data/experiments/
├── tract_political_only_20260321_143022/    # name + timestamp
│   ├── config.yaml                           # frozen copy of experiment config
│   ├── assignments.parquet                   # tract_fips, type, super_type, scores
│   ├── dissolved_communities.geojson         # merged same-type polygons
│   ├── type_profiles.parquet                 # demographic profile per type
│   ├── validation.json                       # holdout r, coherence, stability
│   └── meta.yaml                             # J, holdout_r, n_types, timestamp, git_commit
├── tract_nonpolitical_only_20260321_144512/
│   └── (same structure)
└── comparisons/
    └── political_vs_nonpolitical_20260321.json
```

**Re-run policy:** Each run creates a new timestamped directory (`{name}_{YYYYMMDD_HHMMSS}`). No overwriting. A symlink `data/experiments/{name}_latest → {name}_{timestamp}` points to the most recent run. Old runs are retained for comparison.

---

## Dependencies

- **geopandas** — spatial operations, overlay, dissolve
- **libpysal** — Queen contiguity weights
- **networkx** — connected components for bubble dissolve
- **shapely** — geometry operations
- **TIGER/Line tract shapefiles** — need to download for FL, GA, AL

---

## Open Questions

| ID | Question | When to Answer |
|----|----------|----------------|
| OQ-T1 | Is VEST 2022 available? Would add governor shift pair (2018→2022) at precinct level. | During data exploration |
| OQ-T2 | Should religion features use county proxy or attempt tract estimation from county + demographic correlation? Weight at 0.5× in nonpolitical-only to reduce county-boundary bias? | During feature engineering |
| OQ-T3 | RESOLVED: minimum tract population = 50 (specified in Missing Data Handling) |  |
| OQ-T4 | For the bubble dissolve, should we limit maximum community polygon size? | After first visualization |
| OQ-T5 | CLOSED: GPU KMeans unnecessary — 9,393 × 37 runs in seconds on CPU. |  |
| OQ-T6 | Should we add a type-to-type correspondence matrix to the comparison framework? (Shows which political types map to which nonpolitical types — many-to-many or clean mapping?) | During comparison analysis |
| OQ-T7 | Dasymetric refinement for areal interpolation: use Census block population as a weight surface instead of uniform area? Deferred improvement, not blocking. | Post-MVP |
| OQ-T8 | Historical poll data availability — does the Economist model repo include poll datasets? Need polls for FL/GA/AL state-level races to validate poll propagation. | Next research phase |
