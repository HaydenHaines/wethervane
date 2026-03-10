# Architecture Design Document: US Political Covariation Model

**Status:** Living document (last updated March 2026)
**Scope:** Full technical specification for the community-covariance political model
**Geography:** FL + GA + AL proof-of-concept (226 counties)
**Target:** Functional predictions by October 2026 midterms

---

## Table of Contents

1. [Overall Architecture and Philosophy](#1-overall-architecture-and-philosophy)
2. [Data Assembly Component](#2-data-assembly-component)
3. [Community Type Discovery](#3-community-type-discovery)
4. [Historical Covariance Estimation](#4-historical-covariance-estimation)
5. [Poll Ingestion and Community-Level Propagation](#5-poll-ingestion-and-community-level-propagation)
6. [Prediction and Interpretation](#6-prediction-and-interpretation)
7. [Validation Framework](#7-validation-framework)
8. [MVP Scope (Reduced First Milestone)](#8-mvp-scope-reduced-first-milestone)
9. [Identified Gaps](#9-identified-gaps)

---

## 1. Overall Architecture and Philosophy

### Design Principles

Four non-negotiable design principles govern every component of this system:

**1. Two-stage separation.** Community detection uses only non-political data (religion, class/occupation, neighborhood characteristics, social networks, migration, commuting). Political validation is a separate, downstream stage. These never leak. If non-political community structure fails to predict political covariance, the hypothesis fails cleanly and visibly. This is the core falsifiability mechanism.

**2. Loosely coupled components.** Python handles data assembly and community detection, exports artifacts to `data/`. R and Stan handle Bayesian modeling and MRP. Components communicate through files on disk (Parquet, JSON, CSV), not direct function calls or in-memory objects. Any stage can be re-run independently without re-running the full pipeline.

**3. Reproducibility.** Every intermediate output is saved to `data/` subdirectories. Random seeds are fixed and logged. All data transformations are scripted -- no manual steps between raw data and outputs. The pipeline can be re-run from scratch and produce identical results.

**4. Falsification built in.** A demographic baseline model runs alongside every experiment. If the community-type model does not beat the demographic baseline, the model is not contributing useful structure. Negative results are documented, not hidden.

### Pipeline Overview

```
[Data Assembly] --> [Community Detection] --> [Covariance Estimation] --> [Poll Propagation] --> [Prediction/Interpretation] --> [Validation]
     src/assembly/      src/detection/          src/covariance/           src/propagation/        src/prediction/              src/validation/
     Python              Python                  Python + Stan             R + Stan                Python                       Python + R
```

Components communicate through artifacts in `data/`:

```
data/
  raw/                  # Original downloaded files (never modified)
  assembled/            # Cleaned county-level Parquet + network edge files
  communities/          # Type assignments (W matrices, hierarchy JSON)
  covariance/           # Factor loadings, covariance matrices (Arrow/Parquet)
  polls/                # Cleaned poll data, crosstabs
  predictions/          # Model outputs with credible intervals
  validation/           # Holdout sets, metrics, comparison tables
```

### What "Community Type" Means

A community type is a **latent archetype** discovered from non-political data. It is not a geographic region, not a demographic category, and not a political party proxy. It is a pattern of co-occurring social characteristics -- religion, class structure, occupation mix, neighborhood form, migration patterns -- that recurs across multiple counties, potentially in different states.

Each county has a **probability distribution** over types (soft assignment, not hard clustering). A county is never "one thing." Types are **hierarchical**: fine-grained types (30-80 for FL+GA+AL) nest into blocs (8-15), which nest into mega-blocs (3-5). The number of types at each level is determined by the data, not pre-specified.

**Example:** Tulsa County, OK (if included in a national expansion) might be 35% suburban-professional, 20% urban-Black-institutional, 15% oil-patch-working-class, 10% university-adjacent, 20% mixed-other. Within FL+GA+AL, a county like Duval (Jacksonville) might be 30% military-suburban, 25% urban-Black-institutional, 20% New-South-professional, 15% coastal-retirement, 10% other.

The critical claim is that counties sharing a type will **covary politically**, even when geographically separated. Miami-Dade and Fulton County may share a type that behaves similarly in response to national political shifts, despite being in different states. This is the testable hypothesis.

### Dual-Output Model

Every community type produces **two parameters** per election:

1. **Turnout rate** -- the fraction of eligible voters who vote. Captures mobilization, enthusiasm, access, and suppression effects.
2. **Vote share conditional on turnout** -- the D/R split among those who actually vote. Captures persuasion, partisan lean, and issue positioning.

Covariance operates on **both dimensions jointly**, producing a 2K x 2K covariance matrix (where K is the number of types). This enables decomposition of apparent county-level shifts into three distinct mechanisms, following Grimmer & Hersh (2021, *Science Advances*):

- **Persuasion:** People who voted in both elections changed their vote choice.
- **Differential turnout:** Different people showed up -- some previous voters stayed home, some new voters participated.
- **Population change:** The county's residents changed through migration, aging, death, and new eligible voters.

Most election models treat the county-level vote margin as a single number. By modeling turnout and vote share as separate but correlated quantities at the type level, this model can distinguish between "Type X shifted 3 points toward Democrats because voters were persuaded" and "Type X appeared to shift 3 points because its turnout rate dropped differentially among Republican-leaning members."

---

## 2. Data Assembly Component

**Technology:** Python (pandas, geopandas, cenpy/pytidycensus equivalents)
**Source:** `src/assembly/`
**Output:** `data/assembled/`

### Data Sources

| Source | Resolution | Temporal Coverage | Access Method | Pipeline Step |
|--------|-----------|-------------------|---------------|---------------|
| **MEDSL county returns** | County | 2000-2024 (7 presidential + 5 midterm) | Download from [MIT Election Data + Science Lab](https://electionlab.mit.edu/data) | `download_medsl.py` |
| **Census / ACS** | Tract, block group, county | 2000, 2010, 2020 (decennial); 2005-2024 (ACS 5-year) | Census API via [NHGIS](https://www.nhgis.org/) or cenpy | `download_census.py` |
| **RCMS religion** | County | 2000, 2010, 2020 | Download from [ARDA](https://www.thearda.com/) (Religious Congregations and Membership Study) | `download_rcms.py` |
| **IRS SOI migration** | County-to-county | 2011-2022 (annual) | Download from [IRS SOI](https://www.irs.gov/statistics/soi-tax-stats-migration-data) | `download_irs_migration.py` |
| **Census LODES commuting** | County-to-county (block available) | 2002-2021 | Download from [LEHD](https://lehd.ces.census.gov/) | `download_lodes.py` |
| **Facebook SCI** | County-to-county, ZIP-to-ZIP | Snapshot (~2020) | Download from [HDX](https://data.humdata.org/dataset/social-connectedness-index) | `download_sci.py` |
| **BLS QCEW** | County | 2000-2024 (quarterly) | Download from [BLS](https://www.bls.gov/qcew/) | `download_qcew.py` |
| **Opportunity Insights social capital** | County, ZIP | 2022 | Download from [Opportunity Insights](https://opportunityinsights.org/) | `download_social_capital.py` |
| **CES/CCES** | Individual (geocoded) | 2006-2024 (biennial) | Download from [Harvard Dataverse](https://cces.gov.harvard.edu/) | `download_ces.py` |
| **538 poll archive** | National, state, district | 2000-2024 | GitHub: [fivethirtyeight/data](https://github.com/fivethirtyeight/data) | `download_polls.py` |
| **FL/GA/AL voter files** | Individual (geocoded) | Varies by state | State-specific (FL: public request; GA/AL: see Gaps) | `download_voter_files.py` |

### Feature Engineering

Census/ACS variables are organized into five domains following the OAC methodology (Singleton & Longley 2015):

1. **Demographic structure:** Age distribution, racial/ethnic composition, household size, foreign-born share
2. **Socioeconomic:** Education attainment, income distribution, occupation mix (QCEW), poverty rate
3. **Housing:** Owner/renter ratio, housing age, housing value, density, structure type
4. **Religious/cultural:** RCMS denominational shares (evangelical Protestant, mainline Protestant, Catholic, historically Black Protestant, other), Opportunity Insights social capital indices
5. **Connectivity:** SCI summary statistics per county, commuting self-containment, net migration rate

All features are standardized (range or z-score) before community detection. Correlations are checked; highly collinear features (r > 0.95) are reduced via PCA within domains.

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/assembled/counties.parquet` | Parquet | 226 rows (FL+GA+AL counties), all features across five domains |
| `data/assembled/sci_pairs.parquet` | Parquet | County-to-county SCI values (226 x 226 = 51,076 pairs, sparse) |
| `data/assembled/migration_pairs.parquet` | Parquet | County-to-county IRS migration flows |
| `data/assembled/commuting_pairs.parquet` | Parquet | County-to-county LODES commuting flows |
| `data/assembled/elections.parquet` | Parquet | County x election matrix (226 x ~12 elections), vote share + turnout |
| `data/assembled/feature_metadata.json` | JSON | Feature names, domains, transformations applied, sources |

---

## 3. Community Type Discovery

**Technology:** Python (leidenalg, graph-tool, scikit-learn NMF, infomap, python-igraph)
**Source:** `src/detection/`
**Output:** `data/communities/`

### Core Constraint

Community detection uses **only non-political data**. The `elections.parquet` file is never read by any code in `src/detection/`. This separation is enforced by convention and verified in tests.

### Two Parallel Approaches

Both approaches are run and compared. Neither is assumed to be correct a priori. The question is whether they discover similar structure.

#### Approach A: Multi-Layer Network Community Detection

Build a multi-layer (multiplex) network on the 226 FL+GA+AL counties with five edge layers:

| Layer | Source | Edge Weight | Normalization |
|-------|--------|------------|---------------|
| Social connectedness | Facebook SCI | SCI score | Log-transform + rank |
| Commuting flows | Census LODES | Bidirectional flow count | Log-transform + rank |
| Migration flows | IRS SOI | Net + gross flow | Log-transform + rank |
| Religious similarity | RCMS | Cosine similarity of denominational share vectors | Already [0, 1] |
| Demographic/class similarity | ACS + QCEW | Cosine similarity of standardized feature vectors | Already [0, 1] |

**Primary algorithm: Leiden with resolution sweep** ([leidenalg](https://github.com/vtraag/leidenalg), Traag et al. 2019).

```python
import leidenalg as la
import igraph as ig

# Multiplex partition optimization
layers, interslice_layer, G_full = la.time_slices_to_layers(
    [G_sci, G_commuting, G_migration, G_religion, G_demog],
    interslice_weight=0.1
)
partitions = [la.CPMVertexPartition(H, weights='weight',
              resolution_parameter=gamma) for H, gamma in zip(layers, gammas)]
interslice_partition = la.CPMVertexPartition(interslice_layer,
                        resolution_parameter=0, weights='weight')
optimiser = la.Optimiser()
optimiser.optimise_partition_multiplex(partitions + [interslice_partition])
```

Resolution parameter sweep: Run at 50-100 gamma values from 0.001 to 1.0, producing partitions ranging from 3 mega-blocs to 80+ fine types. Build a co-classification matrix C (how often each pair of counties lands in the same community across resolutions) and cluster C to find robust multi-scale structure (Jeub et al. 2018, *Scientific Reports*).

**Validation algorithms:**

- **Nested SBM** via [graph-tool](https://graph-tool.skewed.de/) (Peixoto 2014): Bayesian model selection, no resolution limit, automatic hierarchy. Provides a principled cross-check on the number of levels and communities.

```python
import graph_tool.all as gt
state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(ec=ec, layers=True))
levels = state.get_levels()  # Automatic hierarchy
```

- **Infomap** ([mapequation.org](https://www.mapequation.org/infomap/)): Flow-based community detection. Particularly appropriate for the commuting and migration layers, where random walks have a direct physical interpretation as movement of people.

#### Approach B: Graph-Regularized NMF

Treat the problem as matrix factorization. The county-by-feature matrix V (226 x ~60 features) is factored into:

```
V ~ W x H
```

where W (226 x K) gives type assignments (each row is a county's probability distribution over K types) and H (K x 60) gives type profiles (each row is a type's characteristic feature vector).

The graph regularizer penalizes assignments that violate the SCI network structure:

```
minimize ||V - WH||^2 + lambda * tr(W^T L W)
```

where L is the graph Laplacian of the SCI network and lambda controls the strength of network regularization. This encourages socially connected counties to have similar type assignments.

K is selected by reconstruction error plus stability analysis: run NMF at multiple K values with random restarts, measure how stable W is across restarts (using Amari distance or cophenetic correlation), and choose the K that balances reconstruction error and stability.

**Implementation:** `sklearn.decomposition.NMF` for the base factorization, with custom graph regularization term. For the full graph-regularized version, follow Cai et al. ([arXiv:1111.0885](https://arxiv.org/pdf/1111.0885)).

### Hierarchy Construction

The hierarchy is built bottom-up:

1. **Fine-grained types** (target: 30-80 for 226 counties): Discovered at high resolution. Each type should correspond to a recognizable community archetype.
2. **Blocs** (target: 8-15): Types that frequently co-occur or have similar feature profiles are merged. Discovered either by running Leiden at lower resolution or by agglomerative clustering on type profiles.
3. **Mega-blocs** (target: 3-5): Blocs that share broad structural characteristics (urban vs. rural, Black Belt vs. non-Black Belt, etc.).

Hierarchical consistency is enforced: every fine-grained type maps to exactly one bloc, every bloc maps to exactly one mega-bloc.

### Key References

- Bailey et al. (2018). "Social Connectedness: Measurement, Determinants, and Effects." *Journal of Economic Perspectives*. [AEA](https://www.aeaweb.org/articles?id=10.1257/jep.32.3.259). Already clustered SCI into cross-state social communities, finding FL+GA+AL form a natural grouping.
- Singleton & Longley (2015). "Creating the 2011 Area Classification for Output Areas." [UCL](https://discovery.ucl.ac.uk/id/eprint/1498873/). Open geodemographic classification methodology (OAC, UK).
- Spielman & Singleton. "North American Geodemographics." [Liverpool GDS](https://www.liverpool.ac.uk/geographic-data-science/research/understandingthemorphologyofcities/north-american-geodemographics/). First open US geodemographic classification from ACS data.
- Liang et al. (2025). Region2Vec-GAT. [GitHub](https://github.com/GeoDS/region2vec-GAT). GNN-based community detection on spatial networks with node attributes and interaction flows.
- Zhang (2022). "Improving Commuting Zones Using the Louvain Community Detection Algorithm." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165176522003093). Precedent for community detection on county-level flow networks.

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/communities/W_leiden.parquet` | Parquet | 226 x K soft assignment matrix (Leiden approach) |
| `data/communities/W_nmf.parquet` | Parquet | 226 x K soft assignment matrix (NMF approach) |
| `data/communities/H_nmf.parquet` | Parquet | K x F type profile matrix (NMF approach) |
| `data/communities/hierarchy.json` | JSON | Type-to-bloc-to-megabloc mapping |
| `data/communities/stability.json` | JSON | Cross-method agreement, resolution stability metrics |
| `data/communities/type_profiles.parquet` | Parquet | Descriptive statistics for each type (median demographics, etc.) |

---

## 4. Historical Covariance Estimation

**Technology:** Python (scikit-learn PCA, statsmodels), Stan via cmdstanpy
**Source:** `src/covariance/`
**Output:** `data/covariance/`

### The Problem

We have K community types and T elections (T ~ 12 for 2000-2024: 7 presidential, 5 midterm). We need a 2K x 2K joint covariance matrix for (vote share, turnout) across types. But 2K >> T (if K = 50, 2K = 100 >> 12), so direct sample covariance estimation is rank-deficient and unreliable.

### Method: Factor-Structured Covariance in Three Steps

#### Step 1: Aggregate County Results to Type Level

Using the soft assignment matrix W from community detection, aggregate county-level election results to type-level results:

```
theta_k^(t) = sum_i( w_ik * y_i^(t) ) / sum_i( w_ik )
```

where `w_ik` is county i's weight on type k, and `y_i^(t)` is county i's vote share (or turnout) in election t. This produces a K x T matrix for vote share and a K x T matrix for turnout. Stack them into a 2K x T matrix.

#### Step 2: PCA for Factor Discovery

Apply PCA to the 2K x T joint matrix. With T = 12, at most 12 components exist. Based on prior work on US election PCA (see [aidanem.com/us-presidential-elections-pca.html](https://www.aidanem.com/us-presidential-elections-pca.html)), expect 3-5 dominant factors:

- **Factor 1:** National swing (all types move together, ~30-40% of variance)
- **Factor 2:** Education polarization (college-town types vs. non-college types move in opposite directions, ~15-20%)
- **Factor 3:** Racial axis (types with high Black population share vs. others, ~10-15%)
- **Factor 4:** Urban/rural divergence (~5-10%)
- **Factor 5:** Turnout-specific variation (~5%)

These factors and their loadings provide the initial estimate of the covariance structure.

**Reference:** Research on county-level PCA consistently finds 2-4 factors capture most variance. The first PC (~34% of variance) captures a North/South cleavage; the second (~14%) captures Urban/Rural opposition. See the ScienceDirect paper on nationalization of county-level elections (1872-2020).

#### Step 3: Bayesian Factor Model in Stan

The PCA provides initial estimates. The full Bayesian factor model, implemented in Stan and called via cmdstanpy, estimates the factor structure with proper uncertainty:

```
theta_k^(t) = Lambda_k * f^(t) + epsilon_k
f^(t) ~ f^(t-1) + eta^(t)
eta^(t) ~ StudentT(nu, 0, sigma_eta)
```

where:
- `theta_k^(t)` is the 2-vector (vote share, turnout) for type k in election t
- `Lambda_k` is a 2 x D loading matrix for type k (D = number of factors, typically 3-5)
- `f^(t)` is the D-vector of latent factors in election t
- `eta^(t)` is the factor innovation, modeled with heavy tails (Student-t) to accommodate occasional large shifts (e.g., 2016 education realignment)
- `epsilon_k` is type-specific idiosyncratic noise

**Network regularization on loadings:** Types that are adjacent in the community similarity network should have similar factor loadings. This is implemented as a graph Laplacian penalty on Lambda: `tr(Lambda^T L Lambda)`, where L is the graph Laplacian of the type similarity network.

**Prior autocorrelation structure**, supported by the voter stability literature (Gelman et al. 2016, Kalla & Broockman 2018; see `research/voter-stability-evidence.md`):

| Timescale | Autocorrelation | Justification |
|-----------|----------------|---------------|
| Within election cycle (months) | > 0.95 | Most poll-to-poll variation is measurement noise, not genuine opinion change. Phantom swings. |
| Between consecutive elections (2-4 years) | 0.85-0.95 | Small but real drift. 1-3 percentage points per cycle for most types. |
| Over longer horizons (10+ years) | 0.50-0.80 | Cumulative drift from realignment. Education polarization, rural-urban sorting. |

**Heavy-tailed innovations:** The Student-t distribution on `eta` (with nu ~ 4-7 degrees of freedom) allows for occasional large shifts without inflating the baseline innovation variance. This handles events like the 2016 education realignment, where college-town types shifted sharply while non-college types shifted in the opposite direction.

### Four Types of Joint Covariance

The 2K x 2K covariance matrix decomposes into four blocks:

| Block | Dimension | What It Captures |
|-------|-----------|-----------------|
| Vote-vote | K x K | How types' vote shares covary across elections. The core "when Type A shifts right, Type B also shifts right" structure. |
| Turnout-turnout | K x K | How types' turnout rates covary. When Type A's turnout drops, does Type B's also drop? |
| Vote-turnout (within type) | K diagonal elements | For each type, how its vote share and turnout covary. Does higher turnout favor D or R for this type? |
| Vote-turnout (cross-type) | K x K off-diagonal | When Type A's turnout rises, does Type B's vote share shift? Captures differential mobilization spillovers. |

### Stability Testing

- **Leave-one-election-out:** For each of the 12 elections, hold it out, re-estimate the factor model from the remaining 11, and measure how well the held-out election's type-level results are predicted by the estimated covariance structure.
- **Rolling window:** Estimate the covariance from 2000-2012, predict 2016. Estimate from 2000-2016, predict 2020. Check whether the covariance structure is stable enough to predict forward.
- **Factor stability:** Compare the factor loadings estimated from 2000-2012 vs. 2000-2020. If the factors change dramatically, the covariance structure is not stable enough for forecasting.

### Key References

- PCA of US presidential elections: [aidanem.com](https://www.aidanem.com/us-presidential-elections-pca.html)
- Linzer (2013). "Dynamic Bayesian Forecasting of Presidential Elections in the States." *JASA*, 108(501), 124-134. [PDF](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf)
- TheEconomist/us-potus-model: [GitHub](https://github.com/TheEconomist/us-potus-model). Open-source Stan code for correlated state-level priors.
- markjrieke/2024-potus: [GitHub](https://github.com/markjrieke/2024-potus). Estimated covariance parameters for the 2024 cycle.
- Heidemanns, Gelman, Morris (2020). "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election." [HDSR](https://hdsr.mitpress.mit.edu/pub/nw1dzd02).

### Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `data/covariance/pca_loadings.parquet` | Parquet | 2K x D PCA loading matrix |
| `data/covariance/pca_explained_variance.json` | JSON | Variance explained per component |
| `data/covariance/factor_model_summary.json` | JSON | Stan model diagnostics, Rhat, ESS |
| `data/covariance/lambda.parquet` | Parquet | Bayesian factor loading matrix (posterior mean + CI) |
| `data/covariance/sigma_2k.parquet` | Parquet | Full 2K x 2K implied covariance matrix |
| `data/covariance/stability_metrics.json` | JSON | Leave-one-out and rolling window results |

---

## 5. Poll Ingestion and Community-Level Propagation

**Technology:** Python for scraping/cleaning, R + Stan for the Bayesian model (adapted from [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model)), cmdstanpy as the Python-Stan bridge
**Source:** `src/propagation/` (R + Stan), `src/assembly/` (poll cleaning, Python)
**Output:** `data/polls/`, `data/predictions/`

### Poll Sources

| Source | Type | Access | Notes |
|--------|------|--------|-------|
| 538 poll archive | National, state, district | [GitHub](https://github.com/fivethirtyeight/data) | Historical archive, structured CSV |
| RealClearPolitics | National, state | Web scraping | Current cycle polls |
| CES/CCES individual data | Individual-level (geocoded) | [Harvard Dataverse](https://cces.gov.harvard.edu/) | 50,000+ respondents per wave, with county FIPS. Gold standard for MRP. |
| State/district polls | State, CD | Various | Aggregated from 538 and RCP |
| Crosstab data | Demographic subgroups within polls | Extracted from poll reports | Age x race x education breakdowns when available |

### Poll Corrections

Following the Economist Stan model (Heidemanns, Gelman, Morris 2020), all poll observations are corrected for systematic biases before entering the propagation model. These corrections are estimated **within the Stan model** as parameters with priors, not applied as fixed pre-processing steps:

1. **Partisan non-response bias:** When one party's supporters are more enthusiastic, they answer polls at higher rates, creating phantom swings (Gelman et al. 2016). Corrected by poststratifying on party identification when individual-level data is available, and by estimating a time-varying non-response parameter in the model.

2. **Mode adjustment:** Phone polls, online polls, IVR polls, and in-person polls have systematic differences. Each mode gets an estimated bias parameter.

3. **Population adjustment:** Registered voter polls vs. likely voter polls vs. adult population polls differ systematically. Adjustment parameters are estimated per population type.

4. **House effects:** Each polling firm has a persistent lean. Estimated as firm-level random effects with a shrinkage prior (half-normal on the SD).

### The Propagation Model (Core Architecture)

This is the central inference engine. It is adapted from [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model), with the critical modification that the 51 state units are replaced by K community types.

#### Generative Model

**Prior (anchoring to previous elections + fundamentals):**

The prior for each type's election-day position is informed by:

- Previous election result for that type (strongest signal, per voter stability evidence)
- Demographic drift since the last election (ACS changes in education, age, race composition)
- Economic fundamentals (following Erikson & Wlezien 2012, *The Timeline of Presidential Elections*; Sides & Vavreck 2013, *The Gamble*): national GDP growth, unemployment, presidential approval
- The prior is **tight**, reflecting the voter stability literature: campaigns have minimal persuasive effects (Kalla & Broockman 2018), and most poll variation is noise

**Latent state (reverse random walk):**

Following Linzer (2013), the latent type-level political state is modeled as a **reverse random walk** anchored at Election Day and walking backward in time:

```
theta_k(t) = theta_k(t+1) + eta_k(t)
eta(t) ~ MVN(0, Sigma_innovation)
```

where `Sigma_innovation` is the innovation covariance derived from the factor model (Section 4). The reverse walk ensures that the model is anchored to the actual election outcome (for hindcasting) or to the prior (for forecasting), with polls progressively refining the estimate as they accumulate.

**Observation model (spectral unmixing):**

Each poll p observes a noisy mixture of type-level signals:

```
y_p = sum_k( w_pk * theta_k ) + bias_p + epsilon_p
```

where:
- `y_p` is the observed poll result
- `w_pk` is the weight of type k in the polled population (derived from the demographic composition of the poll's geographic area and the type assignment matrix W)
- `theta_k` is the latent political state of type k
- `bias_p` captures house effects, mode effects, and population adjustment
- `epsilon_p` is sampling noise (known from poll sample size)

**This is the linear spectral unmixing model from remote sensing** (Rasti et al. 2024; see [HySUPP](https://github.com/BehnoodRasti/HySUPP)). The "endmembers" are community types; the "abundances" are known population shares; the "observed spectrum" is the poll result. The key difference from remote sensing: our "abundances" (type population shares) are known from census data, so we are solving for the endmember values (type-level political states) from multiple mixed observations.

**Crosstab sub-model:**

When polls report demographic breakdowns (e.g., vote share among 18-29 year olds, or among Black respondents), these provide **tighter constraints** because the type-composition weights `w_pk` are more narrowly defined. A crosstab for "Black voters in Florida" has a type composition that is heavily concentrated on types with high Black population share, making the unmixing problem more determined.

```
y_p_crosstab = sum_k( w_pk_crosstab * theta_k ) + bias_p + epsilon_p
```

where `w_pk_crosstab` is the type composition of the specific demographic subgroup in the polled geography.

**CES integration via MRP:**

The Cooperative Election Study (CES) provides individual-level vote choice data with demographic covariates and geographic identifiers. This is integrated via MRP (Multilevel Regression and Poststratification):

```
P(vote_D | demographics_i, community_type_k) = logit^{-1}(
    alpha + X_i * beta + gamma_state + delta_type_k
)
```

where `delta_type_k` is a random effect for community type with a structured prior.

The structured prior on type random effects follows Gao et al. (2021, *Bayesian Analysis*): an ICAR (Intrinsic Conditional Autoregressive) prior where the adjacency structure is defined by type similarity in the community network. This provides spatial smoothing across types without estimating K free parameters.

**Implementation uses:**
- [ccesMRPprep](https://github.com/kuriwaki/ccesMRPprep) (Kuriwaki): R package for preparing CES data for MRP, including poststratification table construction
- [ccesMRPrun](https://github.com/kuriwaki/ccesMRPrun) (Kuriwaki): R package for running MRP on CES data with brms/rstanarm
- Structured priors from [alexgao09/structuredpriorsmrp_public](https://github.com/alexgao09/structuredpriorsmrp_public): Implementation of Gao et al. 2021

### Poll Accumulation Mechanism

The model's behavior changes over the election cycle:

| Phase | Polls Available | Model Behavior |
|-------|----------------|----------------|
| **Early cycle** (12+ months out) | Few/none | Prior dominates. Type-level estimates are essentially previous election + fundamentals + demographic drift. Wide credible intervals. |
| **Mid cycle** (6-12 months) | National + some state polls | Systematic deviations from prior begin to emerge. National swing factor identified. Type-level estimates begin to differentiate. |
| **Late cycle** (0-6 months) | Dense state + some district polls + crosstabs | Type-level shifts identifiable for major types. Crosstabs tighten estimates. Credible intervals narrow. |
| **Election night** | Actual results (partial, then full) | Results decomposed into type-level contributions. Full posterior on type-level parameters. |

### Key Repos to Adapt

| Repository | License | Language | What to Adapt |
|-----------|---------|----------|---------------|
| [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model) | MIT | R + Stan | Core state-space model structure, poll correction framework, reverse random walk. Replace 51 states with K community types. |
| [markjrieke/2024-potus](https://github.com/markjrieke/2024-potus) | MIT | R + Stan | Estimated covariance parameters, updated polling methodology for 2024 cycle. |
| [alexgao09/structuredpriorsmrp_public](https://github.com/alexgao09/structuredpriorsmrp_public) | -- | R + Stan | ICAR structured priors for MRP random effects. Apply to community type random effects. |
| [fonnesbeck/election_pycast](https://github.com/fonnesbeck/election_pycast) | -- | Python + PyMC | Dynamic Bayesian model in PyMC. Reference for Python-native implementation if cmdstanpy bridge proves unwieldy. |
| [kuriwaki/ccesMRPprep](https://github.com/kuriwaki/ccesMRPprep) | MIT | R | CES data preparation for MRP. |
| [kuriwaki/ccesMRPrun](https://github.com/kuriwaki/ccesMRPrun) | MIT | R | MRP execution with brms/rstanarm on CES data. |

---

## 6. Prediction and Interpretation

**Technology:** Python for aggregation and visualization, R for mapping
**Source:** `src/prediction/`, `src/viz/`
**Output:** `data/predictions/`

### Outputs

#### County-Level Predictions

For each of the 226 FL+GA+AL counties, the model produces:

- **Vote share** (two-party Democratic share) with 80% and 95% credible intervals
- **Turnout** (as fraction of voting-eligible population) with credible intervals
- **Type decomposition:** Which types contribute what fraction of the county's predicted vote share and turnout

#### Aggregated Predictions

County-level predictions are aggregated to:

- **Congressional districts** (using county-to-CD crosswalk; for split counties, use tract-level type assignments if available)
- **State-level** totals (FL, GA, AL)
- **Custom geographies** (media markets, MSAs)

#### Uncertainty Decomposition

Total prediction uncertainty decomposes into four sources:

| Source | Description | How Estimated |
|--------|------------|---------------|
| **Polling noise** | Sampling error in poll observations | Known from poll sample sizes; propagated through observation model |
| **Type assignment uncertainty** | Counties' type compositions are estimated, not known | Bootstrap or posterior from NMF/Leiden stability analysis |
| **Covariance estimation uncertainty** | Factor loadings and innovation variance are estimated from 12 elections | Posterior from Stan factor model |
| **Innovation uncertainty** | Future type-level shifts are stochastic | Innovation covariance from factor model, propagated through random walk |

### Shift Narratives (the Distinctive Output)

This is what distinguishes this model from standard election forecasts. Instead of saying "Duval County shifted 2 points toward Democrats," the model says:

**Type-level shift table:**

| Type | Weight in Duval | Vote Share Shift | Turnout Shift | Contribution to Duval Shift |
|------|----------------|-----------------|---------------|---------------------------|
| Military-suburban | 0.30 | +0.5 | -1.2 | +0.15 |
| Urban-Black-institutional | 0.25 | +1.0 | +3.5 | +0.25 |
| New-South-professional | 0.20 | +2.5 | +0.5 | +0.50 |
| Coastal-retirement | 0.15 | -1.0 | -0.5 | -0.15 |
| Other | 0.10 | +0.0 | +0.0 | +0.00 |
| **Total** | **1.00** | -- | -- | **+0.75** |

**Per-county decomposition of shifts into type contributions** -- this is the spectral unmixing output. Each county's observed shift is decomposed into the weighted sum of its constituent types' shifts.

**Turnout decomposition** following Grimmer & Hersh (2021):

| Mechanism | Contribution to Duval Shift |
|-----------|---------------------------|
| Persuasion (same voters, different choice) | +0.30 |
| Differential turnout (different voters showed up) | +0.35 |
| Population change (different people live there) | +0.10 |
| **Total apparent shift** | **+0.75** |

### Conditional Forecasting

The type-level structure enables mechanistic conditional forecasts:

- "If Cuban American communities shift as polls suggest (Type X moves +5 toward R), Florida shifts by Y points."
- "If Black turnout returns to 2012 levels in GA (Type Z turnout increases by 8 points), Georgia flips by probability P."
- "If suburban-professional types continue their 2016-2020 trend, FL-CD13 shifts by Z points."

These are **mechanistic scenarios**, not pure extrapolation. Each scenario specifies which type-level parameters change and by how much, and the model propagates those changes through the county composition structure.

---

## 7. Validation Framework

**Technology:** Python + R
**Source:** `src/validation/`
**Output:** `data/validation/`

### Three Baselines

Every model run is compared against three baselines. If the community-type model does not beat all three, it is not contributing useful structure.

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **1. Demographic linear model** | OLS regression of county vote share on ACS demographic variables (education, race, income, age, urbanicity) + state fixed effects. No community structure. | Python: scikit-learn `LinearRegression` with state dummies |
| **2. Uniform swing** | Apply the national popular vote swing uniformly to all counties' previous-election results. The simplest possible model. | Python: single scalar applied to all counties |
| **3. Demographic MRP** | Standard MRP (multilevel regression and poststratification) using CES data with demographic grouping variables but **no community type variable**. Uses ccesMRPprep + ccesMRPrun with `(1|state) + (1|age) + (1|race) + (1|education)` but not `(1|community_type)`. | R: brms or rstanarm via ccesMRPrun |

### Hindcast Validation

**Leave-one-election-out** for all elections 2000-2024:

For each election t in {2000, 2002, 2004, ..., 2024}:
1. Estimate community types from non-political data (this does not change across folds, since types use no political data).
2. Estimate covariance from all elections **except** t.
3. Set priors from elections before t.
4. Simulate a "polling environment" for election t using polls available before election day.
5. Generate predictions for election t.
6. Compare to actual results.

**Specific historical tests** (chosen because they stress-test the model's ability to capture known dynamics):

| Test Case | What It Tests |
|-----------|--------------|
| **FL 2000** (Bush v. Gore) | Can the model handle a razor-thin election with unusual turnout patterns? |
| **FL 2008/2012** (Obama coalition) | Does the model capture the Obama coalition's distinctive type composition? |
| **GA 2020** (Biden flip) | Can the model capture a state flip driven by suburban-professional + Black-urban type shifts? |
| **FL Cuban American shift 2016-2024** | Can the model detect and propagate a type-specific shift (Cuban American communities moving sharply toward R) across multiple elections? |
| **2022 midterm** | Does the model generalize from presidential to midterm elections (different turnout patterns, potentially different covariance structure)? |

### Metrics

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| **County RMSE (vote share)** | < demographic baseline | Average prediction error across counties |
| **County RMSE (turnout)** | < demographic baseline | Average turnout prediction error |
| **Differential swing correlation** | r > 0.5 between predicted and actual type-level swings | Does the model correctly predict *which types shift* and *in which direction*? |
| **Type-level shift stability** | Cross-election correlation of type shifts > 0.7 | Are the same types consistently behaving similarly across elections? |
| **Calibration** | 80% CI covers actual result 75-85% of the time | Are credible intervals well-calibrated? |
| **Information gain from polls** | Measurable reduction in RMSE as polls accumulate | Do polls actually improve predictions through the covariance structure? |

### Falsification Criteria

The model is considered **falsified** (hypothesis rejected) if any of the following hold:

| # | Criterion | What It Would Mean |
|---|-----------|-------------------|
| 1 | Community types do not beat demographics (Baseline 3 wins) | Non-political community structure adds no information beyond standard demographic predictors. The hypothesis that community types capture something beyond demographics is wrong. |
| 2 | Cross-border structure adds < 1% RMSE improvement over within-state-only types | The SCI-based cross-state community structure is not meaningfully different from state-level demographics. The "communities cross state borders" claim is not supported. |
| 3 | Type-level behavior is less stable across elections than county-level behavior | The types are not capturing stable behavioral patterns. The abstraction is losing signal rather than gaining it. |
| 4 | 2000-2016 covariance fails to predict 2020/2024 shifts | The covariance structure is not stable enough for forecasting. The factor model is fitting noise in historical data rather than capturing genuine structure. |
| 5 | Adding the turnout dimension does not improve vote share predictions | The joint modeling of turnout and vote share is not adding value. The additional complexity is not justified. |

### Variation Partitioning

Following metacommunity ecology methodology, use `vegan::varpart()` (R) to decompose county-level political variation into:

- **E (Environment):** Fraction explained by demographics alone (ACS variables)
- **S (Space/Structure):** Fraction explained by community structure alone (type assignments)
- **E intersection S:** Fraction explained by spatially structured demographics (demographics that covary with community structure)
- **Residual:** Unexplained variation

This directly answers: "How much does community structure add beyond demographics?" If E alone explains 90% and S adds only 1%, the community types are not pulling their weight.

**Implementation:** Use `vegan::varpart()` with county vote share as the response, ACS demographic variables as the E matrix, and community type dummy variables (or soft assignment vectors) as the S matrix. The partial R-squared values give the decomposition.

### Reference

Economist backtesting scripts: `final_2008.R`, `final_2012.R`, `final_2016.R` in [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model). These provide tested code for running the Economist model on historical elections with known outcomes.

---

## 8. MVP Scope (Reduced First Milestone)

**Target:** 3 months (by June 2026)

The MVP answers four specific questions before investing in the full Bayesian machinery:

1. Do the discovered types correspond to recognizable, interpretable communities?
2. Is the PCA factor structure meaningful and stable?
3. Does community structure beat demographics in explaining political variation?
4. Do cross-border communities exist in the SCI clustering of FL+GA+AL?

### What Is In the MVP

| Component | MVP Implementation | Full Model |
|-----------|-------------------|------------|
| **Geography** | FL + GA + AL (226 counties) | Same for v1; national expansion post-v1 |
| **Data sources** | MEDSL + ACS + RCMS + SCI + IRS migration | Add LODES, QCEW, Opportunity Insights, voter files |
| **Community detection** | Leiden only (single algorithm, resolution sweep) | Leiden + nested SBM + Infomap + NMF, consensus clustering |
| **Covariance estimation** | PCA only (sklearn), no Bayesian factor model | Full Bayesian factor model in Stan |
| **Poll propagation** | Simple Kalman filter via [FilterPy](https://filterpy.readthedocs.io/) with PCA-derived covariance | Full Stan state-space model adapted from Economist model |
| **MRP** | None | CES integration via ccesMRPprep + ccesMRPrun with structured priors |
| **Validation** | Demographic baseline + variation partitioning | All three baselines + full hindcast + falsification criteria |
| **Voter files** | FL only (publicly available) | FL + GA + AL |
| **Turnout model** | Turnout as separate univariate alongside vote share | Joint 2K x 2K covariance model |

### MVP Pipeline

```
[Download Data]           scripts/download_all.sh
       |
[Assemble Counties]       python -m src.assembly.run
       |
[Detect Communities]      python -m src.detection.run  (Leiden only)
       |
[PCA Covariance]          python -m src.covariance.run  (PCA only)
       |
[Kalman Propagation]      python -m src.propagation.run  (FilterPy)
       |
[Validate]                python -m src.validation.run
                          Rscript src/validation/varpart.R
```

### MVP Deliverables

1. A map of FL+GA+AL colored by discovered community types, with human-interpretable labels
2. PCA scree plot and factor loading visualization
3. Variation partitioning results: how much does community structure add beyond demographics?
4. Cross-border community analysis: which SCI-discovered communities span state borders?
5. Simple Kalman filter predictions for a held-out election, compared to demographic baseline

### What Is Deferred to Post-MVP

| Component | Why Deferred |
|-----------|-------------|
| Full Stan Bayesian factor model | Requires significant Stan development time; PCA provides a viable first approximation |
| MRP integration | Requires R infrastructure (renv, ccesMRPprep); adds complexity before core hypothesis is tested |
| Voter files for GA and AL | GA voter file requires institutional access; AL is restrictive (see Gaps) |
| Real-time poll scraping | Not needed for hindcasting; adds operational complexity |
| Turnout decomposition (Grimmer & Hersh) | Requires voter file data for composition/conversion separation |
| Shift narratives | Requires the full propagation model to be meaningful |
| LODES commuting data | Engineering complexity of processing LODES origin-destination files; IRS migration + SCI provide adequate network layers for MVP |
| Live 2026 prediction | Requires real-time infrastructure; MVP focuses on hindcasting validation |

---

## 9. Identified Gaps

Known gaps in the current design that require resolution before or during implementation:

| Gap | Severity | Description | Mitigation Strategy |
|-----|----------|-------------|---------------------|
| **GA voter file access** | Medium | Georgia voter file requires a fee and institutional affiliation for full access. Individual-level turnout history needed for Grimmer-Hersh decomposition. | Defer to post-MVP. Use aggregate county-level turnout from MEDSL for MVP. Explore academic data-sharing agreements. |
| **AL voter file access** | Medium | Alabama voter file access is restrictive. Less critical than GA because AL is less politically competitive. | Defer to post-MVP. AL primarily included for geographic contiguity and SCI network completeness, not as a primary prediction target. |
| **Tract-level community assignment** | Medium | Crosstab mapping requires knowing the type composition of demographic subgroups within counties. This requires tract-level type assignments, but MVP community detection operates at county level. | For MVP, approximate by assuming uniform type composition within counties. Post-MVP, run geodemographic classification at tract level within heterogeneous counties and construct tract-level type assignments. |
| **Soft assignment validation** | Low-Medium | There is no ground truth for soft type assignments. How do we validate that a county is "35% suburban-professional and 25% urban-Black-institutional"? | Validate indirectly: (a) compare NMF-derived soft assignments to Leiden-derived ones; (b) check that soft assignments correlate with known tract-level demographic variation within counties; (c) verify that using soft vs. hard assignments improves prediction. |
| **Midterm vs. presidential covariance** | Medium | The covariance structure may differ between presidential and midterm elections (different turnout patterns, different issue salience). With only 5 midterms in the 2000-2024 window, estimating a separate midterm covariance is infeasible. | Start by assuming the same factor structure with an election-type indicator. If midterm predictions are systematically worse, investigate a modified covariance for midterms. The factor model can include a midterm-specific loading adjustment. |
| **Turnout ground truth at type level** | Medium | Type-level turnout is not directly observed; it is inferred from county-level turnout via soft assignments. This inference is circular if the type assignments were partly determined by turnout-correlated features. | Mitigated by the two-stage separation: types are discovered from non-political data, so turnout does not influence type assignment. Validate by comparing inferred type-level turnout against voter-file-derived turnout estimates (FL voter file provides individual turnout history). |
| **Narrative generation** | Low | Shift narratives (Section 6) are currently conceived as manual interpretation of model outputs. Automated narrative generation would require NLG infrastructure. | Keep manual for MVP and v1. Consider LLM-based narrative generation post-v1 (cf. the EPJ Data Science paper on LLM-based geodemographic naming). |
| **LODES engineering complexity** | Low-Medium | Census LODES origin-destination files are large (block-level), require aggregation to county level, and have annual vintage changes. Processing pipeline is non-trivial. | Defer to post-MVP. IRS SOI migration data + Facebook SCI provide adequate network layers for initial community detection. Add LODES when the pipeline is stable. |
| **Model identifiability with K >> T** | Medium | With K = 50 types and T = 12 elections, the type-level parameters are not individually identifiable from election data alone. The factor model addresses this by reducing dimensionality, but individual type estimates may have wide posteriors. | Accept wide posteriors on individual type parameters. The model's value is in the covariance structure (factor loadings), not individual type point estimates. Validate that the covariance structure produces useful predictions despite individual-type uncertainty. |
| **Temporal alignment of data sources** | Low | ACS, RCMS, SCI, and election data have different temporal cadences. ACS is 5-year rolling; RCMS is decennial; SCI is a snapshot; elections are biennial. | Use temporally closest available data for each election. Log all temporal mismatches in `data/assembled/feature_metadata.json`. For MVP, use a single cross-section (most recent available) and note the approximation. |

---

## Appendix A: Key References (Consolidated)

### Election Modeling

- Linzer (2013). "Dynamic Bayesian Forecasting of Presidential Elections in the States." *JASA*, 108(501), 124-134. [PDF](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf)
- Heidemanns, Gelman, Morris (2020). "An Updated Dynamic Bayesian Forecasting Model for the US Presidential Election." [HDSR](https://hdsr.mitpress.mit.edu/pub/nw1dzd02)
- Ghitza & Gelman (2013). "Deep Interactions with MRP." *AJPS*. [PDF](https://sites.stat.columbia.edu/gelman/research/published/misterp.pdf)
- Gao et al. (2021). "Improving MRP with Structured Priors." *Bayesian Analysis*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203002/)
- Erikson & Wlezien (2012). *The Timeline of Presidential Elections*. University of Chicago Press.
- Sides & Vavreck (2013). *The Gamble*. Princeton University Press.

### Voter Stability

- Gelman, Goel, Rivers, Rothschild (2016). "The Mythical Swing Voter." *QJPS*, 11, 103-130.
- Kalla & Broockman (2018). "The Minimal Persuasive Effects of Campaign Contact in General Elections." *APSR*, 112(1), 148-166.
- Shirani-Mehr, Rothschild, Goel, Gelman (2018). "Disentangling Bias and Variance in Election Polls." *JASA*, 113(522), 607-614.

### Composition vs. Conversion

- Grimmer, Hersh, et al. (2021). "Not by Turnout Alone: Measuring the Sources of Electoral Change, 2012 to 2016." *Science Advances*.

### Community Detection and Geodemographics

- Traag, Waltman, van Eck (2019). "From Louvain to Leiden." *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-019-41695-z)
- Peixoto (2014). "Hierarchical Block Structures and High-Resolution Model Selection in Large Networks." *Physical Review X*.
- Bailey et al. (2018). "Social Connectedness: Measurement, Determinants, and Effects." *JEP*. [AEA](https://www.aeaweb.org/articles?id=10.1257/jep.32.3.259)
- Singleton & Longley (2015). "Creating the 2011 Area Classification for Output Areas." [UCL](https://discovery.ucl.ac.uk/id/eprint/1498873/)
- Jeub et al. (2018). "Multiresolution Consensus Clustering in Networks." *Scientific Reports*. [Nature](https://www.nature.com/articles/s41598-018-21352-7)
- Cai et al. "Graph Regularized Nonnegative Matrix Factorization." [arXiv:1111.0885](https://arxiv.org/pdf/1111.0885)

### Cross-Disciplinary Methods

- Rasti et al. (2024). "Image Processing and Machine Learning for Hyperspectral Unmixing." *IEEE TGRS*. [HySUPP](https://github.com/BehnoodRasti/HySUPP)
- Liang et al. (2025). Region2Vec-GAT. [GitHub](https://github.com/GeoDS/region2vec-GAT)
- Zhang (2022). "Improving Commuting Zones Using the Louvain Community Detection Algorithm." [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165176522003093)

### Factor Analysis of Election Data

- PCA of US Presidential Elections. [aidanem.com](https://www.aidanem.com/us-presidential-elections-pca.html)
- Partisanship & Nationalization (1872-2020). [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0261379421001050)

### Open-Source Election Models

- TheEconomist/us-potus-model: [GitHub](https://github.com/TheEconomist/us-potus-model) (MIT license, R + Stan)
- markjrieke/2024-potus: [GitHub](https://github.com/markjrieke/2024-potus) (MIT license, R + Stan)
- fonnesbeck/election_pycast: [GitHub](https://github.com/fonnesbeck/election_pycast) (Python + PyMC)
- alexgao09/structuredpriorsmrp_public: [GitHub](https://github.com/alexgao09/structuredpriorsmrp_public) (R + Stan)
- kuriwaki/ccesMRPprep: [GitHub](https://github.com/kuriwaki/ccesMRPprep) (MIT, R)
- kuriwaki/ccesMRPrun: [GitHub](https://github.com/kuriwaki/ccesMRPrun) (MIT, R)

### Ecology (Variation Partitioning)

- `vegan::varpart()`: [CRAN](https://cran.r-project.org/web/packages/vegan/). Standard tool for decomposing variation into environmental vs. spatial components.

## Appendix B: Technology Stack Summary

| Role | Primary Tool | Language | Backup |
|------|-------------|----------|--------|
| Data wrangling | pandas, geopandas | Python | -- |
| Census data access | cenpy, NHGIS downloads | Python | tidycensus (R) |
| Community detection (network) | leidenalg | Python | graph-tool (nSBM), infomap |
| Community detection (matrix) | sklearn NMF | Python | custom graph-regularized NMF |
| Graph construction | python-igraph | Python | networkx (prototyping only) |
| PCA / factor analysis | sklearn PCA | Python | statsmodels DynamicFactorMQ |
| Bayesian modeling | Stan via cmdstanpy | Python/Stan | PyMC v5 |
| State-space filtering (MVP) | FilterPy | Python | pykalman |
| MRP | brms, rstanarm via ccesMRPrun | R/Stan | PyMC MRP |
| Spatial analysis | PySAL (libpysal, esda) | Python | spdep (R) |
| Variation partitioning | vegan::varpart() | R | -- |
| Posterior diagnostics | ArviZ | Python | -- |
| Visualization | matplotlib, plotly | Python | ggplot2 (R) |
| Data format (intermediate) | Parquet (pyarrow) | -- | CSV |
| Data format (covariance) | Parquet or Arrow IPC | -- | NetCDF |
| Environment (Python) | pyproject.toml | -- | -- |
| Environment (R) | renv.lock | -- | -- |
