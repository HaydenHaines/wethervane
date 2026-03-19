# Shift-Based Community Discovery — Design Spec

**Date:** 2026-03-18
**Status:** Approved
**Replaces:** NMF-on-demographics community detection (Stages 1-3 of proof-of-concept)

## Core Concept

Communities are defined by correlated electoral behavior, not demographics. A "community" is a contiguous geographic region where vote shifts move together across elections. Census and other data describe and border these communities after discovery — demographics are descriptive, not prescriptive.

This inverts the proof-of-concept approach (which used census demographics to discover community types via NMF, then validated against election data). The proof-of-concept confirmed the hypothesis that community structure predicts electoral covariance (R^2 ~ 0.66). This design builds the real model.

### Architectural Pivot: Two-Stage Separation

The proof-of-concept enforced "two-stage separation" — communities discovered from non-political data only, political data used only for validation. This was the right design for testing whether non-political community structure predicts political covariance. It did (R^2 ~ 0.66), confirming the hypothesis.

This design deliberately abandons two-stage separation. The hypothesis is now confirmed; the goal shifts from "test whether community structure predicts politics" to "find the actual communities that move together politically, then describe them." The falsifiability mechanism changes accordingly: instead of "do non-political communities predict votes?", it becomes "do communities discovered from past electoral shifts predict future shifts?" (temporal holdout). The CLAUDE.md and ASSUMPTIONS_LOG must be updated to reflect this pivot. See ADR-005 (to be written during implementation).

## Data Model: Shift Vectors

For each census tract, compute a shift vector per election type.

**Presidential (2 pairs from available data):**
- 2016 -> 2020: delta D share, delta R share, delta turnout
- 2020 -> 2024: delta D share, delta R share, delta turnout
- 6 dimensions total

**Midterm (1 pair from available data):**
- 2018 -> 2022: delta D share, delta R share, delta turnout
- 3 dimensions total

**Combined tract feature vector: 9 dimensions** (6 presidential + 3 midterm)

### Preprocessing

- Normalize each dimension to zero-mean, unit-variance
- Scale the midterm block by sqrt(2) so its total contribution equals the presidential block (compensates for 1 pair vs 2)
- D share and R share are nearly redundant in two-party races but both kept — elections with meaningful third-party vote (2016 had ~5%) carry signal in both

### County-level fallback

For 2022/2024 (MEDSL county-level data only), all tracts within a county receive the same county-level shift. The clustering will group these tracts together; finer data will differentiate them when sources expand.

### Alabama midterm gap

The 2018 AL gubernatorial race was uncontested, and 2022 AL data was excluded from the proof-of-concept back-calculation due to MEDSL quality issues. AL tracts (~2,300) receive zero for all three midterm shift dimensions (delta D, delta R, delta turnout). This is a structural zero, not a missing value — it means "no midterm signal available." The sqrt(2) midterm scaling still applies to FL+GA tracts; AL tracts contribute zero to the midterm block, which means their clustering is driven entirely by their 6 presidential dimensions. This is acceptable for MVP — AL midterm data is a target for the data expansion TODO.

## Spatial Community Discovery

### Algorithm: Hierarchical Agglomerative Clustering with Spatial Constraint

**Step 1 — Adjacency graph.** Queen contiguity from tract geometries via `libpysal.weights.Queen`. Tracts sharing even a single point are neighbors. Produces a sparse connectivity matrix. After construction, check `w.islands` for tracts with zero neighbors (coastal FL barrier islands, keys, isolated urban tracts). Assign island tracts to their nearest neighbor by centroid distance using `libpysal.weights.KNN(k=1)` and merge into the adjacency graph. Log island count.

**Step 2 — Clustering.** `sklearn.cluster.AgglomerativeClustering` with:
- `connectivity` = Queen adjacency matrix (guarantees spatial contiguity)
- `linkage` = "ward" (minimizes within-cluster shift variance)
- `distance_threshold` = tunable, no fixed K
- `n_clusters` = None (threshold determines count)

Set `compute_distances=True`. Build a scipy-format linkage matrix from `children_` and `distances_` (format: `[left, right, distance, count]` per merge) — sklearn does not produce this directly. Pickle the scipy linkage matrix, not the sklearn object. Every cut height on this dendrogram yields a different community map.

**Step 3 — Operating cut selection.** Sweep thresholds from fine to coarse:
1. At each threshold, compute weighted mean within-community shift variance
2. Plot variance vs. number of communities
3. Detect elbow using the `kneed` library (`KneeLocator` with `curve="convex"`, `direction="decreasing"`, `S=1.0`). Add `kneed` to pyproject.toml.
4. Expose a "resolution" parameter for finer/coarser exploration (manual override of the elbow)

Expected range: 50-500 communities for FL+GA+AL.

**Step 4 — Border gradient scoring.** For every pair of adjacent communities, compute mean shift-vector difference across shared boundary tracts. Produces a "border sharpness" score for visualization and demographic validation.

### Limitations accepted for MVP

- Hard assignment only (each tract in exactly one community, no soft/fuzzy membership)
- County-level data for 2022/2024 means intra-county differentiation depends on presidential shifts only
- Thin midterm signal (1 pair) — will strengthen with data expansion

## Census Overlay and Community Description

After community discovery, overlay descriptive data per community:

- **Aggregate ACS demographics:** The 12 features from `build_features.py` (racial composition, income, education, commute mode, housing tenure, occupation, age)
- **Population and land area** from tract geometries
- **Turnout profile by election type:** Mean turnout for presidential vs. midterm within each community

**Output:** Community profile table — one row per community with shift signature, demographic summary, geographic extent, and turnout-by-type profile.

### Border validation

For each community boundary, compare demographic profiles on either side. Shift-defined borders that align with demographic transitions = confirmation. Borders that don't align = interesting (electoral behavior grouping people demographics wouldn't predict).

### Comparison with proof-of-concept

- Take the K=7 NMF communities from the shelved approach
- For each shift-discovered community, compute NMF community type overlap
- Compare predictive accuracy: discover communities from pre-2024 data, predict 2024 shifts, measure error for both approaches
- Side-by-side output

## Temporal Validation

### Training/holdout split

- **Training:** Discover communities from 2016->2020 (presidential) + 2018->2022 (midterm) shifts
- **Holdout:** 2020->2024 presidential shift (unseen)

### Metrics

1. **Within-community shift consistency:** Variance of holdout shift across tracts within each community. Low = community correctly grouped co-moving tracts.
2. **Community-level prediction:** Use mean 2016->2020 shift as naive predictor of 2020->2024 shift direction. Measure correlation and MAE across communities.
3. **Comparison to NMF baseline:** Same holdout on NMF-derived assignments. Which grouping has lower within-community variance?

### Success criteria

- Shift-discovered communities have meaningfully lower within-community variance on holdout than (a) random spatial groupings of same size and (b) NMF-derived communities
- Community-level shift predictions correlate positively with 2024 actuals
- FL's structural rightward drift shows up as community-level pattern

### Failure criteria

- Within-community holdout variance no better than random spatial chunks = shift signal not stable, communities overfit to training period
- First diagnostic: training period too short (1 presidential pair) — data expansion TODO addresses this

## Pipeline Architecture

### New modules

```
Stage 1: Data Assembly (reuse + new)
  fetch_acs.py             # keep as-is
  build_features.py        # keep as-is
  fetch_vest*.py           # keep as-is
  fetch_2022/2024          # keep as-is
  build_community_weights.py  # keep (for future poll decomposition)
  NEW: build_shift_vectors.py  # per-tract shift vectors by election type

Stage 2: Community Discovery (replaces NMF)
  NEW: build_adjacency.py     # Queen contiguity from tract geometries
  NEW: cluster_communities.py  # hierarchical agglomerative, dendrogram, elbow
  NEW: score_borders.py        # boundary gradient sharpness

Stage 3: Community Description (replaces covariance estimation)
  NEW: describe_communities.py  # census overlay, demographic profiles
  NEW: compare_to_nmf.py        # side-by-side with shelved NMF approach

Stage 4: Validation
  NEW: validate_holdout.py  # temporal holdout on 2024

Stage 5: Prediction (post-MVP, poll decomposition)
  Deferred until community discovery validated
```

### Shelving strategy

Existing NMF/Stan/propagation code stays in `src/` untouched. New code goes alongside it. `compare_to_nmf.py` imports from old modules for comparison. Nothing deleted.

### Data outputs

```
data/
  shifts/
    tract_shifts.parquet            # 9-dim shift vectors per tract
  communities/
    community_assignments.parquet   # tract -> community_id mapping
    community_profiles.parquet      # per-community demographics + shifts
    dendrogram.pkl                  # full dendrogram for multi-scale exploration
    border_gradients.parquet        # boundary sharpness per community pair
  validation/
    holdout_2024_results.parquet    # per-community holdout metrics
    nmf_comparison.parquet          # NMF vs shift communities side-by-side
```

### Prerequisites

The following data files must exist before the new pipeline runs. They are produced by existing assembly scripts (gitignored, not committed):

```bash
# Stage 1 prerequisites — run these if data/assembled/ is empty:
python src/assembly/fetch_acs.py                    # -> data/assembled/acs_tracts_2022.parquet
python src/assembly/build_features.py               # -> data/assembled/tract_features.parquet
python src/assembly/fetch_vest_multi_year.py         # -> data/raw/vest_tracts_{2016,2018,2020}.parquet
python src/assembly/fetch_2022_governor.py           # -> data/raw/medsl_county_2022_governor.parquet
python src/assembly/fetch_2024_president.py          # -> data/raw/medsl_county_2024_president.parquet
```

Also required: tract geometries (shapefile or GeoJSON) for Queen contiguity. The existing `build_tract_geojson.py` in `src/viz/` produces this, or `geopandas` can fetch from Census TIGER/Line directly.

### Dependencies to add

- `libpysal` (spatial adjacency)
- `kneed` (elbow detection)
- `geopandas` (already used by viz)
- `scipy` (dendrogram utilities)
- scikit-learn (already available)

## Autonomous TODOs: Stage 1 Robustness Expansion

### TODO 1: Historical VEST Data Expansion

Pull VEST precinct-to-tract crosswalks for 2012 and 2014 elections. Expands shift vector from 9 to 15 dimensions:
- Presidential: 2012->2016, 2016->2020, 2020->2024 (3 pairs)
- Midterm: 2014->2018, 2018->2022 (2 pairs)

Must handle 2010->2020 census tract boundary crosswalk (Census relationship files). Output: tract-level D share, R share, turnout for 2012 and 2014, harmonized to 2020 census tracts, in same format as existing VEST data.

### TODO 2: Research Additional Data Sources

Evaluate data sources for border refinement and community description. Candidates:
- RCMS religious congregation data
- LODES commuting patterns (tract-level)
- IRS SOI migration data (county-level)
- FCC broadband access maps
- USDA rural-urban continuum codes
- School district boundaries
- Property value / Zillow ZTRAX
- Social Connectedness Index (Meta)

For each: availability, cost (free only for MVP), resolution, temporal coverage, likely signal beyond ACS. Output: `docs/DATA_SOURCES_EXPANSION.md`.

### TODO 3: Local Election Data Research

Research local/municipal election results at tract-compatible geographies:
- State election offices (FL, GA, AL)
- OpenElections project
- Academic datasets

Assess coverage, format, geography, crosswalk difficulty. Output: `docs/LOCAL_ELECTION_DATA.md`.

## Future: Poll Decomposition (Post-MVP)

Given the community mosaic + enough polls with crosstabs across elections (2000-2026), infer which communities the polling pool oversamples/undersamples. MVP assumption: identical respondent pool across pollsters. With enough polls, the pool becomes describable in community terms and reweightable to ground truth. Design deferred until community discovery is validated.
