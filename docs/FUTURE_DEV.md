# Future Development Roadmap

**Status:** Living document (last updated March 2026)

---

## Table of Contents

1. [Second-Order Campaign Propagation](#1-second-order-campaign-propagation)
2. [Primary Elections as Candidate Adjustment Signals](#2-primary-elections-as-candidate-adjustment-signals)
3. [Convergence/Divergence Timeline](#3-convergencedivergence-timeline)
4. [What-If Scenario Machine](#4-what-if-scenario-machine)
5. [Visualization Engine](#5-visualization-engine)
6. [Architecture Gaps](#6-architecture-gaps)
7. [Computational Requirements and Hardware Planning](#7-computational-requirements-and-hardware-planning)
8. [Development Phases](#8-development-phases)

---

## 1. Second-Order Campaign Propagation

### The Problem

The current architecture treats all instances of a community type as equally connected. If "Cuban American urban" is a community type, a poll showing FL Cubans shifting automatically updates the estimate for NYC Cubans identically, because they share the same type-level parameter theta_k. But a FL-targeted campaign should affect FL Cubans more than NYC Cubans.

### The Missing Piece: Within-Type Decay

Decompose each type's behavior into two components:

- **National type component**: moves in lockstep across all instances (a national shift in Cuban American sentiment driven by national media/policy)
- **Local instance component**: independent, captures local effects (FL-specific campaigning, candidate-specific appeal, local issue salience)

The ratio between national and local is itself estimable from historical data. Some types are more nationally coherent (evangelical communities react to national media uniformly) while others are more local (union communities respond to local plant closings). The SCI data can calibrate this: two Cuban communities with high mutual SCI share more of their shift than two with low SCI.

### Implementation Path

This is a natural extension of the factor model. Factor 1 (national swing) loads on all instances equally. A "type-specific national factor" loads equally on all instances of one type. A "local factor" loads only on individual instances. The architecture supports this -- it needs an additional factor level in the hierarchy.

### Theoretical Grounding

The Axelrod model (from the adjacent fields research) provides the framework: interaction intensity determines convergence rate. Communities that interact more (higher SCI, shared media market) converge more tightly. Geschke et al. (2019) on calibrating opinion dynamics to empirical data provides the calibration framework for determining the decay rate.

### Example

FL Cuban targeted campaigning shifts FL Cuban communities by +5R. The model propagates:
- NYC Cuban communities: +3R (high SCI with FL Cubans, shared media consumption)
- TX Cuban communities: +2R (moderate SCI, different media market)
- Puerto Rican communities: +1R (correlated via factor structure but different type, weaker connection)
- Non-Hispanic communities: +0.1R (minimal cross-type propagation)

---

## 2. Primary Elections as Candidate Adjustment Signals

### Core Idea

Primary election results -- particularly upsets -- can inform the "candidate adjustment" parameter in the general election model. A young upstart knocking off an entrenched incumbent signals something about general election dynamics that should feed into the prior.

### What the Literature Says

**Hall (2015, APSR)** found that ideologically extreme primary winners lose ~9-13 percentage points in general elections (via regression discontinuity). But this finding is specifically about ideological extremism, not "upstart energy" -- a charismatic moderate unseating an incumbent would not trigger the same penalty.

**Abramowitz (1988, updated)** operationalizes candidate quality primarily through prior office-holding (+3-5 points for candidates who have held elected office). Primary performance is not a direct input.

**Sides, Tausanovitch, and Vavreck (2022)** argue that elections are increasingly "calcified" -- determined by fundamentals and identity rather than candidate-specific factors -- implying that primary signals should have **diminishing** predictive value for general elections. However, this calcification itself may vary by community type, which our model can test.

**Existing models (538, Economist, Cook, Sabato)** use primary results only indirectly, through expert ratings. No major model algorithmically incorporates primary data into general election forecasts.

### Why Primaries Matter for This Model Specifically

The community-covariance model can decompose primary results by community type in ways standard models cannot:

1. **Evangelical turnout surge in a GOP primary** --> signals about the evangelical community type's general election enthusiasm/mobilization
2. **Progressive challenger winning in an urban district** --> signals about the "urban gentrifying" type's intensity and potential turnout
3. **Incumbent upset driven by specific community types** --> reveals which communities are activated and in what direction

This decomposition is the unique value-add. Standard models see "candidate X won the primary by 8 points." Our model sees "candidate X won because Type A surged +15 in turnout while Type B stayed flat, and the margin came entirely from Type A precincts."

### Technical Implementation

Two approaches, starting simple:

#### Approach A: Prior Adjustment (simpler, recommended first)

Primary results adjust the candidate quality component of the fundamentals prior:

```
delta_q = f(primary_margin_vs_expectation, primary_turnout_vs_baseline)
```

For community-type decomposition, replace the scalar delta_q with a vector of community-type adjustments:

```
alpha_c = f(primary_turnout_deviation_in_type_c, primary_margin_in_type_c)
```

These alpha_c values adjust the prior mean for that community type's contribution to the general election.

#### Approach B: Additional Observation (more principled, harder)

Treat primary results as a noisy observation of the latent general election state with a different observation model:

```
y_primary = g(theta_general) + epsilon_primary
```

where g() is a non-identity mapping accounting for differences between primary and general electorates, and epsilon_primary has higher variance than poll observations. The function g() must account for: different turnout composition, different choice set, partisan vs. general electorate.

### Interaction with Reverse Random Walk

Primaries occur months before the general, when the reverse random walk has the **highest uncertainty** (widest credible intervals). This means primary results would have the most influence precisely when we know the least -- which is desirable. As polls accumulate closer to Election Day, the primary signal is naturally downweighted by Bayesian updating.

```
y_primary,d = mu_d(t_primary) + Z_d * alpha + epsilon_primary,d
```

where mu_d(t_primary) is the latent state at the primary date, Z_d maps community types to the district, and alpha is the vector of type-level enthusiasm signals.

### Key Complications

- **Open vs. closed primaries**: FL has closed primaries (registered party members only); GA and AL have open primaries (anyone can vote in either). Open primaries include strategic crossover voting, making the signal noisier.
- **Selection effects**: Only districts with contested primaries generate data. The decision to contest is itself informative.
- **Primary type matters**: Crowded fields vs. two-candidate races, presidential vs. congressional, with vs. without incumbent -- all affect signal quality.
- **Calibration**: The observation variance for primary signals should be estimated from historical data. Expect it to be large.

### Data Sources for Primary Results

| Source | Coverage | Granularity | Primary Data? | Access |
|--------|----------|-------------|---------------|--------|
| FL Division of Elections | 2000+ | Precinct | Yes | Free, downloadable |
| GA Secretary of State | 2000+ (partial) | County/Precinct | Yes | Free |
| AL Secretary of State | 2000+ (limited) | County | Yes | Free, may require direct request pre-2010 |
| OpenElections Project | Varies | Precinct (partial) | Yes | Free, GitHub repos |
| Dave Leip's Atlas | 1800s+ | County | Yes | ~$50/year subscription |
| MEDSL/Harvard Dataverse | Varies | County/Precinct | Partial | Free |
| Daily Kos/Downballot | 2012+ | Precinct | Some | Free Google Sheets |
| Snyder et al. "Primary Elections in the US, 1824-2020" | 1824-2020 | County | Yes | Harvard Dataverse |

### Key References

- Hall (2015). "What Happens When Extremists Win Primaries?" *APSR*. RDD finding: extremist primary winners lose 9-13 pts in general.
- Abramowitz (1988, updated). "An Improved Model for Predicting the Outcomes of House Elections." Fundamentals + prior office as quality proxy.
- Sides, Tausanovitch, Vavreck (2022). *The Bitter End*. Calcification thesis -- shrinking candidate effects.
- Kenney and Rice (1987). "The Relationship between Divisive Primaries and General Election Outcomes."
- Jacobson (1978, 1980, 1990). Campaign spending effects, challenger quality.
- Bonica (2014). "Mapping the Ideological Marketplace." DIME ideology estimates from finance data.
- Linzer (2013). "Dynamic Bayesian Forecasting." *JASA*. Reverse random walk framework.
- Gelman and King (1993). "Why Are American Presidential Election Campaign Polls So Variable When Votes Are So Predictable?"

---

## 3. Convergence/Divergence Timeline

### Concept

A timeline visualization showing when community types converged or diverged -- both politically and in identity.

### Vote-Choice Convergence

For any pair of types, compute rolling correlation of their election-to-election shifts. Rising correlation = converging politically. Dropping correlation = diverging.

With 7 presidential elections (2000-2024), there are 6 shift observations per type pair. Adding midterms gives ~12 observations. Thin for rolling windows, but sufficient for the basic analysis. Dave Leip's county data extends further back for deeper historical views.

### Identity Convergence

Track how non-political feature profiles change over time using annual ACS data (2005-present) and decennial Census. For each type pair, compute cosine similarity of feature profiles over time. Rising similarity = identity convergence.

The interesting cases are where identity is converging but politics is diverging (or vice versa).

### Implementation

This is a downstream analysis of the covariance estimation output -- not a gap in the architecture, but a dedicated viz module. The data is there; it needs a timeline visualization (x-axis: election years, heatmap or line chart of pairwise type correlations). Ideally interactive: click on a type pair and see both political and demographic trajectories.

---

## 4. What-If Scenario Machine

### Architecture

The Bayesian posterior IS the Monte Carlo simulation. The Stan model produces posterior draws: thousands of samples of (type-level vote share, type-level turnout) for every type and election. Each draw is one "possible world."

### How It Works

1. For each posterior draw, compute county results --> aggregate to district/state/national
2. The distribution of outcomes across draws gives probability distributions
3. "Florida goes R with probability 72%" = "72% of posterior draws show FL going R"

### User Interface

The user adjusts a constraint ("Cuban Americans shift +5R relative to prior"). The system filters or re-weights posterior draws to only those consistent with the constraint, then re-aggregates. This is **importance sampling** -- computationally trivial once posterior draws are stored.

### Storage Requirements

Pre-compute ~10,000 posterior draws: 10K draws x 500 types x 2 parameters = 10M floats = ~80MB in Parquet. The what-if adjustments are post-hoc array operations on these draws, fast enough for interactive response.

### Conditional Resampling

When the user fixes a type's behavior, resample all other types from the conditional posterior (respecting the covariance structure). If you fix Cuban Americans at +5R, the model shows what that implies for Puerto Ricans given their historical correlation. This is the key advantage over simple slider-based tools.

### Precedent

This is the same approach 538 used (40,000 simulations), but with community types as the atomic unit rather than state-level correlations.

---

## 5. Visualization Engine

### Why It Matters

538's success was largely viz-driven. The model's outputs are inherently spatial and hierarchical -- they demand good visualization to be interpretable.

### Core Visualizations

| Viz | Purpose | Interaction |
|-----|---------|-------------|
| **Community type map** | Geographic distribution of types (choropleth with mixture coloring) | Click county --> see type breakdown |
| **Shift map by type** | "Which communities moved?" -- community types overlaid on geography | Toggle between types, election years |
| **Covariance network** | Force-directed graph of types, edge weight = covariance | Highlight type --> see correlations |
| **Convergence timeline** | When did types converge/diverge? (Section 3) | Select type pairs, toggle political vs identity |
| **Poll accumulation tracker** | How uncertainty narrows as polls arrive | Animate through time |
| **What-if scenario builder** | Sliders for type-level shifts --> instant map update (Section 4) | Real-time posterior resampling |
| **Turnout decomposition** | Stacked bar: persuasion vs turnout vs composition | Drill down from state --> county --> type |
| **Baseline comparison** | Side-by-side: community model vs demographic model vs uniform swing | Toggle between models |

### Technology Options

| Stack | Strengths | Weaknesses | When to Use |
|-------|-----------|------------|-------------|
| **Streamlit/Panel** | Fast prototyping, Python-native | Limited polish, hard to customize | Research phase, MVP |
| **Plotly Dash** | Python backend, React frontend, good interactivity | Middle-of-road design quality | v1 dashboard |
| **Observable/D3.js** | Maximum design control, 538-quality output | Heavy frontend engineering | Public-facing v2+ |
| **React + deck.gl** | Best for map-heavy applications | Significant engineering investment | If maps are the primary view |

**Recommendation:** Streamlit for MVP, Plotly Dash for v1, evaluate Observable/D3 for public-facing v2.

---

## 6. Architecture Gaps

### 6.1 Candidate Effects in Downballot Races

The community model explains structural political behavior, but local/downballot elections have strong candidate-specific effects. The model needs a candidate-effect layer for anything below presidential -- a simple additive term estimated from candidate quality proxies (fundraising, incumbency, scandal indicators, primary performance per Section 2). The Economist and 538 models handle this with "candidate adjustment" parameters for Senate/Governor races.

### 6.2 Temporal Evolution of Community Types

Types are currently detected once and treated as static. But communities change: gentrification reshapes neighborhoods, factory closures transform labor communities. The architecture should eventually re-detect types periodically (using each ACS vintage) and track type evolution -- which types are growing, shrinking, splitting, or merging. This is a version of the **dynamic topic model** (Blei & Lafferty 2006).

### 6.3 Media Market Effects

Media environment is a plausible community-shaping force (Hopkins 2018 on nationalization of local politics) but isn't in the current detection pipeline. DMA boundaries are proprietary (Nielsen), but approximate shapefiles exist. Adding media market as a detection layer could reveal communities defined partly by shared information environment.

### 6.4 Real-Time Data Pipeline

For October 2026, polls need to flow in without manual curation. This means scraping or API access to polling aggregators, automated pollster rating lookup, and triggering model updates. The current design assumes manual poll ingestion.

### 6.5 National Scaling

FL+GA+AL is 226 counties. The full US is ~3,100. Community detection scales fine (Leiden handles millions of nodes). Data assembly scales linearly. The bottleneck is the **Stan state-space model** at national scale -- see Section 7.

### 6.6 Individual-Level Validation via Voter Files

Florida voter files provide individual-level turnout records that could validate type-level turnout estimates. The crosswalk from individual voters to community types (via geocoding to tracts, then tract-level type assignments) is a significant data engineering project, but it's the gold standard validation.

### 6.7 Causal Inference for Campaign Effects

The second-order campaign effect (Section 1) is ultimately a causal question: did the FL Cuban campaign *cause* NYC Cuban shifts, or did both respond independently to the same national stimulus? Establishing causality requires quasi-experimental designs (geographic regression discontinuity at campaign boundary, difference-in-differences around campaign timing). This is a research direction, not a feature.

---

## 7. Computational Requirements and Hardware Planning

### Reference Hardware

**Local machine:** Ryzen 5600 (6-core/12-thread), Radeon RX 7800XT (16GB VRAM, RDNA3), 64GB RAM

### GPU Compatibility (Critical Context)

**RX 7800XT (gfx1101) ROCm Status:**
- ROCm 6.4.1+: Official support on native Linux and WSL
- PyTorch: Supported (training + inference, Linux and Windows)
- JAX: Supported for inference only per official docs; training/sampling not officially supported on consumer Radeon
- Stan OpenCL: Works via standard OpenCL drivers (not ROCm-specific)
- scikit-learn: CPU only (NVIDIA cuML has no AMD equivalent)

**Key caveat:** JAX is listed as "inference only" on Radeon GPUs. This limits PyMC's JAX backend GPU sampling (NumPyro NUTS) -- may work but is not officially supported for sampling workloads.

### Task-by-Task Breakdown: FL+GA+AL (226 Counties)

| Task | RAM | Time | GPU Useful? | Verdict |
|------|-----|------|-------------|---------|
| Leiden community detection (5-layer multiplex, 100 gamma sweep) | <500 MB | 1-10 min | No | **LOCAL** |
| Graph-tool nested SBM | <1 GB | 1-10 min | No | **LOCAL** |
| Scikit-learn NMF (226 x 60, K=30-80) | <100 MB | <1 sec | No | **LOCAL** |
| PCA (100 x 12 matrix) | <10 MB | milliseconds | No | **LOCAL** |
| Stan Bayesian factor model (2K=100 params, D=5 factors, 4 chains x 2000 iter) | ~200-500 MB | 5-30 min | No (data below 20K element threshold) | **LOCAL** |
| Stan state-space model (K=50 types, reverse random walk, 4 chains) | 2-8 GB | 2-8 hours | Marginal (1.2-1.5x at best) | **LOCAL, slow** |
| MRP via brms (50K respondents, K=50 type random effects) | 2-8 GB | 30 min - 2 hr | **Yes** (bernoulli_logit_glm_lpmf is OpenCL-accelerated, 50K > 20K threshold, 2-5x speedup) | **LOCAL** |
| FilterPy Kalman filter (100-dim state, MVP) | <100 MB | milliseconds | No | **LOCAL** |

**Stan threading note:** Stan only benefits from physical cores, not hyperthreads. The Ryzen 5600 has 6 physical cores. Best configs: 4 chains x 1 thread, or 3 chains x 2 threads (with `reduce_sum` for within-chain parallelism).

### Task-by-Task Breakdown: National Scale (~3,100 Counties, K=200-500)

| Task | RAM | Time | GPU Useful? | Verdict |
|------|-----|------|-------------|---------|
| Leiden (3,100-node multiplex, 100 gamma sweep) | 1-4 GB | 2-10 min | No | **LOCAL** |
| Nested SBM (3,100 nodes) | 1-4 GB | 10-60 min | No | **LOCAL** |
| NMF (3,100 x 60, K=200-500) | <1 GB | 10-30 min | No | **LOCAL** |
| PCA (1,000 x 12) | <10 MB | milliseconds | No | **LOCAL** |
| Stan factor model (2K=400-1000, D=5 factors) | 2-8 GB | 1-6 hours | No | **LOCAL, slow** |
| **Stan state-space model (K=200-500, 200-day random walk)** | **10-30 GB** | **12-72+ hours** | **Marginal** | **CLOUD for K>200** |
| MRP (50K respondents, K=200-500 type random effects) | 4-16 GB | 2-8 hours | Yes (OpenCL, 2-5x on likelihood) | **LOCAL** |
| Kalman filter (1,000-dim state) | <1 GB | seconds to minutes | No | **LOCAL** |

### The Cloud Bottleneck: National-Scale State-Space Model

The Stan state-space model at national scale is the **only task that definitively requires cloud compute**. With K=500 types x T=200 days = 100,000 latent state parameters plus covariance structure:

- Autodiff expression graph: 10-30 GB
- Runtime: 24-72+ hours, with high risk of divergent transitions
- Each leapfrog step requires gradient computation through the entire latent state sequence

**Recommended cloud instances:**

| Instance | Specs | Cost/hr | Use Case |
|----------|-------|---------|----------|
| AWS c7a.16xlarge | 64 vCPU AMD EPYC, 128 GB RAM | ~$2.50 | Stan state-space, national scale |
| AWS c6i.8xlarge | 32 vCPU Intel, 64 GB RAM | ~$1.36 | Stan state-space, moderate K |
| AWS r7a.4xlarge | 16 vCPU, 128 GB RAM | ~$1.80 | Memory-bound models |

**Expected cloud costs:** A national-scale model run would take 12-48 hours on a 32-64 core instance, costing $15-120 per run. Budget for 5-10 runs during development (iteration on reparameterization, prior sensitivity): **$100-1,000 total for model development.**

### Reparameterization to Avoid Cloud

The cloud bottleneck can potentially be avoided by **reparameterizing the state-space model** to marginalize out latent states using a Kalman filter:

- The `walker` R package and `bssm` R package implement Kalman-filter-marginalized state-space models in Stan
- This avoids sampling the 100K latent state parameters directly
- Expected speedup: **50-100x**, potentially bringing national-scale runs back to local feasibility (1-6 hours on local hardware)
- Trade-off: requires reformulating the Stan model, and some model features (e.g., heavy-tailed innovations) are harder to implement with Kalman marginalization

**This reparameterization should be a high priority for national scaling.**

### GPU Acceleration Strategy

1. **Stan OpenCL for MRP** is the clearest GPU win. Uses standard OpenCL (not ROCm), should work with the 7800XT's OpenCL 2.0 support directly. Expected 2-5x speedup on the likelihood evaluation for models with >20K observations.

2. **PyMC + JAX on ROCm** is theoretically possible for GPU-accelerated NUTS sampling (~4-8x faster than CPU Stan on large models). Test carefully -- consumer Radeon stability is not guaranteed. If stable, this is the best path for all Bayesian models.

3. **CPU-only Stan is the safe default.** Use `parallel_chains=4` and `threads_per_chain` for within-chain parallelism via `reduce_sum`.

### Summary: What Runs Where

```
LOCAL (all 226-county work):
  Everything. No cloud needed for FL+GA+AL proof-of-concept.

LOCAL (national scale, most tasks):
  Community detection, NMF, PCA, factor model, MRP, Kalman filter

CLOUD (national scale, one task):
  Stan state-space model with K>200 types
  ...UNLESS reparameterized with Kalman marginalization (walker/bssm),
  in which case it may also run locally.
```

---

## 8. Development Phases

| Phase | Target | What It Adds |
|-------|--------|-------------|
| **MVP** | June 2026 | FL+GA+AL, Leiden detection, PCA covariance, Kalman filter, basic viz (Streamlit). All local. |
| **v1** | October 2026 | Full Stan model, MRP, live poll ingestion, Streamlit dashboard, 2026 midterm live test. All local. |
| **v1.5** | Early 2027 | Primary election integration (prior adjustment approach), candidate quality parameters, convergence timeline viz. All local. |
| **v2** | Mid 2027 | National scaling (3,100 counties), Kalman-marginalized state-space model OR cloud Stan runs, voter file validation (FL), what-if machine, Plotly Dash dashboard. Mostly local; cloud for state-space if not reparameterized. |
| **v3** | 2028 cycle | Within-type decay modeling (second-order propagation), temporal type evolution, media market layer, polished viz engine (D3/Observable), causal campaign effects research. Local + cloud model runs. |
| **v4** | Long-term | Primary modeling as full observation model (Approach B), custom polling integration, public-facing interactive tool, national real-time dashboard. |

---

## References (Consolidated for This Document)

### Second-Order Propagation
- Geschke et al. (2019). "The Triple-Filter Bubble: Using Agent-Based Modelling to Test a Meta-Theoretical Framework for the Emergence of Filter Bubbles and Echo Chambers." Calibrating opinion dynamics to empirical data.
- Axelrod (1997). "The Dissemination of Culture." Core model for interaction-driven convergence.

### Primary Elections and Candidate Quality
- Hall (2015). "What Happens When Extremists Win Primaries?" *APSR*.
- Abramowitz (1988, updated). "An Improved Model for Predicting House Elections."
- Sides, Tausanovitch, Vavreck (2022). *The Bitter End*.
- Kenney & Rice (1987). "Divisive Primaries and General Election Outcomes."
- Jacobson (1978, 1980, 1990). Campaign spending and challenger quality.
- Bonica (2014). "Mapping the Ideological Marketplace." DIME database.
- Gelman & King (1993). "Why Are Campaign Polls So Variable?"
- Rodden (2019). *Why Cities Lose*.
- Green, Palmquist, Schickler (2002). *Partisan Hearts and Minds*.

### Computational Methods
- Stan OpenCL GPU Guide: https://mc-stan.org/cmdstanr/articles/articles-online-only/opencl.html
- ROCm Compatibility Matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- walker R package (Kalman-marginalized state-space in Stan): https://cran.r-project.org/web/packages/walker/
- bssm R package (Bayesian state-space models): https://cran.r-project.org/web/packages/bssm/

### Visualization
- Hopkins (2018). *The Increasingly United States*. Nationalization of politics and media effects.
- Blei & Lafferty (2006). "Dynamic Topic Models." Framework for temporal type evolution.
