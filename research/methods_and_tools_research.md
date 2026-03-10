# Statistical Methods & Tools for Political Community-Covariance Modeling

## Research Summary — March 2026

**Problem statement:** ~5,000 political communities (county clusters), ~7–12 elections (2000–2024). Estimate covariance structure of political behavior, propagate sparse polls across it, produce community-level estimates with uncertainty. Core constraint: N_elections << N_communities, so direct covariance estimation is rank-deficient.

---

## 1. Factor Models for Political Data

### 1.1 The Approach

Factor models decompose the 5,000 x T vote-share matrix into a small number of latent factors (k << T) times community-specific loadings. This directly addresses the rank deficiency: if k=3 factors explain most variance, the implied covariance matrix is Λ Λ' + Ψ (factor loadings times their transpose plus idiosyncratic variance), which is well-defined even with T=12 observations.

### 1.2 What PCA/Factor Analysis of US Election Data Has Found

- **North/South vs. Urban/Rural:** PCA of county-level presidential vote shares across elections consistently finds that the first principal component (~34% of variance) captures a North/South cleavage, while the second (~14%) captures Urban/Rural or East/West opposition. Source: [US Presidential Elections PCA](https://www.aidanem.com/us-presidential-elections-pca.html).
- **Nationalization trend:** Research on county-level data across presidential, senatorial, and gubernatorial elections (1872–2020) finds increasing nationalization — meaning fewer independent dimensions of variation over time, and a dominant first PC. Source: Partisanship & nationalization paper via [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0261379421001050).
- **Health/economic correlates of swing:** PCA of county-level demographic and health variables found a single "unhealthy score" component explaining 68% of variance in those predictors, and that score strongly predicted 2012→2016 vote shift (22% shift per unit). Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5624580/).

**Implication for the model:** 2–4 factors likely capture most of the covariance in community-level vote shares. This makes factor-model-based covariance estimation feasible and well-motivated.

### 1.3 Implementations

| Package | Language | What It Does | Maturity | URL |
|---------|----------|-------------|----------|-----|
| `statsmodels.tsa.statespace.DynamicFactorMQ` | Python | Large-scale dynamic factor models with EM algorithm. Handles missing data, mixed frequencies, blocks of factors with different AR orders. Scales to hundreds of observed variables. | Production-ready | [statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ.html) |
| `sklearn.decomposition.PCA` / `FactorAnalysis` | Python | Standard PCA and factor analysis. Simple, fast, well-tested. No time-series structure. | Production-ready | [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) |
| `sklearn.decomposition.NMF` | Python | Non-negative matrix factorization — enforces non-negativity, yields more interpretable factors. | Production-ready | [scikit-learn](https://scikit-learn.org/stable/modules/decomposition.html) |
| `dfms` | R | Dynamic factor models via EM or two-step estimation. Handles missing data, assumes stationary VAR process for factors. Clean API. | Mature (rOpenSci) | [dfms](https://sebkrantz.github.io/dfms/) |

**Assessment for this project:**
- `DynamicFactorMQ` is the strongest option if you want to model the temporal dynamics of factors (how the latent political dimensions evolve across elections). It handles missing data natively, which matters if some communities have incomplete election data.
- For a simpler first pass, `sklearn.decomposition.PCA` on the 5,000 x 12 matrix of vote-share changes will immediately tell you how many factors you need and what they look like. With T=12, you can extract at most 12 components — the first 2–4 will likely dominate.
- The factor-implied covariance Λ Λ' + Ψ gives you a well-conditioned 5,000 x 5,000 covariance matrix from just a few parameters.

---

## 2. Bayesian Hierarchical Models

### 2.1 The Approach

Bayesian hierarchical models handle the N >> T problem by imposing structure via priors: communities within the same bloc share a group-level distribution, which provides partial pooling. The posterior covariance among communities is implied by the hierarchical structure rather than estimated directly.

### 2.2 Key Election Modeling Work

**Gelman et al. / The Economist model:**
- The Economist's 2020/2024 presidential election model (Gelman, Heidemanns, Morris) is the state of the art. It combines national polls, state polls, economic fundamentals, and a hierarchical Bayesian model with:
  - Correlated state-level random effects (correlation matrix estimated from past elections + demographic predictors like education)
  - Random-walk priors across time
  - Pollster house effects
  - Partial pooling across states
- The correlation matrix was estimated from past election results and state-level predictors, with off-diagonal elements < 0 set to zero, then scaled.
- Open-source Stan code: [TheEconomist/us-potus-model](https://github.com/TheEconomist/us-potus-model)
- Paper: [Harvard Data Science Review](https://hdsr.mitpress.mit.edu/pub/nw1dzd02)
- **Direct relevance:** This model does exactly the "propagate polls across a covariance structure" task, but at the state level (51 units). Scaling to 5,000 communities requires structural modifications (see below).

**Linzer (2013):**
- Dynamic Bayesian forecasting at the state level. Random-walk priors borrow strength across time; hierarchical specification borrows across states. Foundation for the Economist model.
- Paper: [JASA](https://votamatic.org/wp-content/uploads/2013/07/Linzer-JASA13.pdf)

**Ghitza & Gelman (2013) — "Deep Interactions with MRP":**
- Extended MRP to deeply interacted subgroups (age x race x income x state). Uses hierarchical Gaussian priors on random effects for smoothing.
- Replication data: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PZAOO6)
- Paper: [AJPS](https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12004)

### 2.3 Scaling Hierarchical Models to 5,000 Communities

The Economist model has 51 states and estimates a 51x51 correlation matrix. With 5,000 communities, you cannot estimate a full correlation matrix. Options:

1. **Block-diagonal structure:** Communities within the same bloc share a bloc-level mean; blocs are the unit of correlation estimation. If you have ~50 blocs, you estimate a 50x50 correlation matrix (feasible) and communities within a bloc share that bloc's trajectory plus idiosyncratic noise.

2. **Factor-structured prior:** Instead of a full covariance matrix, parameterize community random effects as θ_i = Λ_i f + ε_i where f is a low-dimensional factor and Λ_i are loadings. This is a Bayesian factor model.

3. **Spatial/network prior:** Use a CAR/ICAR prior (see Section 4) that borrows strength from neighboring communities without estimating a full covariance matrix.

### 2.4 Software Comparison

| Tool | Language | Strengths | Weaknesses | URL |
|------|----------|-----------|------------|-----|
| **Stan** (via `cmdstanpy`) | Python/R/etc. | Gold standard for HMC. Full Bayesian inference. Excellent diagnostics. Used by Gelman, Economist, etc. Multivariate priors, LKJ correlation prior, state-space models built in. | Slower than INLA for large spatial models. Custom modeling language. 5,000 group-level random effects will be slow (~hours). | [mc-stan.org](https://mc-stan.org/) |
| **PyMC** (v5.x) | Python | Pure Python. JAX/Numba backends. NUTS sampler. Good GP support (HSGP). Active development. | Slightly less flexible than raw Stan for custom models. Similar speed constraints for large hierarchical models. | [pymc.io](https://www.pymc.io/) |
| **R-INLA** | R | Orders of magnitude faster than MCMC for latent Gaussian models. BYM/BYM2 spatial models built in. Handles thousands of spatial units routinely. | Approximate (not full MCMC). R-only. Less flexible for non-standard models. | [r-inla.org](https://www.r-inla.org/) |
| **brms** | R (Stan backend) | Formula interface for Stan. Supports CAR, SAR, ICAR spatial structures. MRP-ready. | Same computational limits as Stan. | [CRAN](https://cran.r-project.org/web/packages/brms/) |
| **rstanarm** | R (Stan backend) | Pre-compiled Stan models for MRP. Fast for standard models. | Less flexible than brms or raw Stan. | [mc-stan.org/rstanarm](https://mc-stan.org/rstanarm/) |
| **ArviZ** | Python | Posterior analysis, diagnostics, model comparison. Works with Stan, PyMC, Pyro, NumPyro, emcee. | Not a modeling tool — it's for analyzing output. | [python.arviz.org](https://python.arviz.org/) |

**Assessment for this project:**
- **For prototyping:** Use `cmdstanpy` or `PyMC` to build a hierarchical model with bloc-level random effects and community-level noise. Start with a simplified structure (e.g., 50 blocs, each with a random effect, and within-bloc community variation).
- **For production at scale (5,000 communities):** R-INLA is likely necessary if you want spatial random effects (BYM2 model) on all 5,000 communities. It handles this scale routinely for disease mapping. Stan/PyMC will struggle with 5,000+ latent variables in a single model unless you use clever parameterizations.
- **For the Economist-style correlated prior:** Adapt the open-source Stan code, but replace the 51x51 state correlation matrix with a factor-structured or block-diagonal prior for 5,000 communities.

### 2.5 Stan Covariance Priors

Stan supports several priors for covariance/correlation matrices:
- **LKJ prior:** `lkj_corr_cholesky(eta)` — eta=1 is uniform on correlations, eta=2 favors smaller correlations. Standard for moderate dimensions.
- **Wishart / Inverse-Wishart:** Classical but often poorly calibrated.
- **Decomposed parameterization:** Break covariance into scales (half-normal/half-Cauchy on SDs) and a correlation matrix (LKJ). This is the recommended approach.
- For 5,000 communities, none of these are feasible for a full correlation matrix. You need structural assumptions (factors, blocks, or spatial priors).

Reference: [Stan User's Guide — Multivariate Priors for Hierarchical Models](https://mc-stan.org/docs/2_19/stan-users-guide/multivariate-hierarchical-priors-section.html)

---

## 3. Network-Regularized Covariance Estimation

### 3.1 The Approach

If you have a network connecting communities (shared demographics, geographic adjacency, or economic similarity), you can regularize the covariance/precision matrix estimation by penalizing deviations from the network structure. This is the graphical lasso with a non-uniform penalty: connected communities are allowed to have non-zero partial correlations, disconnected ones are penalized toward zero.

### 3.2 Methods

**Graphical Lasso (standard):**
- Estimates a sparse precision matrix (inverse covariance) by solving: min_Θ { -log det(Θ) + tr(S Θ) + α ||Θ||_1 }
- With T=12 observations and N=5,000 communities, the sample covariance S is rank 12. The L1 penalty makes the problem well-posed by forcing sparsity.
- However, the standard graphical lasso applies a uniform penalty — it doesn't know which communities should be connected.

**Network-informed graphical lasso:**
- Replace the scalar penalty α with a matrix P where P_ij is small when communities i and j are connected (allowing non-zero precision entries) and large when they are not (forcing zeros).
- This encodes the network topology into the covariance structure.

**Graph Laplacian regularization:**
- Penalize the precision matrix to be close to a target structured matrix derived from the network's graph Laplacian: L = D - A (degree matrix minus adjacency matrix).
- This enforces that communities connected in the network have similar random effects.
- Paper: [Bayesian Regularization via Graph Laplacian](https://projecteuclid.org/journals/bayesian-analysis/volume-9/issue-2/Bayesian-Regularization-via-Graph-Laplacian/10.1214/14-BA860.pdf)

### 3.3 Implementations

| Package | Language | What It Does | URL |
|---------|----------|-------------|-----|
| `sklearn.covariance.GraphicalLasso` | Python | Standard graphical lasso with uniform L1 penalty. Also `GraphicalLassoCV` with cross-validation. | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html) |
| `skggm` (QuicGraphicalLasso) | Python | **Matrix penalty** support — can specify a full penalty matrix P_ij. Scikit-learn compatible. This is the key package for network-informed penalties. Much faster than sklearn's implementation. | [skggm](https://skggm.github.io/skggm/tour), [GitHub](https://github.com/skggm/skggm) |
| `mdpeer` | R | Graph-based penalty combining graph Laplacian + ridge. Handles singularity in graph-originated penalty matrices. | [CRAN](https://cran.r-project.org/web/packages/mdpeer/) |

**Assessment for this project:**
- `skggm` is the critical package. It lets you encode community similarity as a penalty matrix: communities that share demographic/geographic features get lower penalties (allowed to covary), while dissimilar communities get higher penalties (forced toward conditional independence).
- However, with T=12 and N=5,000, even the graphical lasso may struggle — the sample covariance is rank 12, so the optimization landscape is very flat. You'd likely combine this with factor structure: first extract k factors, then apply graphical lasso to the residuals.
- The Bayesian version (Graph Laplacian prior on precision) is more principled and can be embedded in a Stan/PyMC model, but is computationally heavier.

---

## 4. Spatial Econometrics

### 4.1 The Approach

Spatial econometric models explicitly parameterize spatial dependence via a spatial weights matrix W. Three main variants:

- **Spatial Lag Model (SLM):** y = ρ W y + X β + ε — the outcome depends on neighbors' outcomes.
- **Spatial Error Model (SEM):** y = X β + u, where u = λ W u + ε — spatial correlation in the errors.
- **Spatial Durbin Model (SDM):** y = ρ W y + X β + W X γ + ε — both spatial lag and spatially lagged predictors.

### 4.2 Election Applications

- **County-level voting patterns:** Moran's I statistics computed on county-level vote shares show significant spatial autocorrelation. Clusters of counties shift together. Using rook contiguity for continental US counties (n=3,085) yields an average of 5.6 neighbors per county.
- **Bayesian spatial econometric analysis:** Lacombe (2025) used a Bayesian spatial probit model for presidential elections, finding that economic freedom variables exhibit spatial spillover effects. Source: [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/ajes.12616).
- **UK elections:** Multilevel spatial models combining Gaussian processes with Markov random fields have been applied to UK constituency data (2019, 2024). Source: [Springer](https://link.springer.com/article/10.1007/s12061-023-09563-6), [Oxford Academic](https://academic.oup.com/jrsssc/advance-article/doi/10.1093/jrsssc/qlaf055/8307260).

### 4.3 Implementations

| Package | Language | What It Does | Maturity | URL |
|---------|----------|-------------|----------|-----|
| **PySAL / spreg** | Python | Full spatial econometrics: SLM, SEM, SDM, GMM estimation, spatial diagnostics, spatial weights construction. Actively maintained. | Production-ready | [pysal.org/spreg](https://pysal.org/spreg/), [GitHub](https://github.com/pysal/spreg) |
| **PySAL / libpysal** | Python | Spatial weights matrices from shapefiles, contiguity, distance bands. Queen/rook contiguity for county data. | Production-ready | [pysal.org](https://pysal.org/pysal/) |
| **spdep** | R | Spatial dependence analysis: Moran's I, spatial weights, spatial regression interfaces. The canonical R package. | Production-ready | [r-spatial.github.io/spdep](https://r-spatial.github.io/spdep/) |
| **spatialreg** | R | Spatial regression models (SLM, SEM, SDM). Companion to spdep. | Production-ready | CRAN |

**Assessment for this project:**
- Spatial econometric models are useful for **modeling spatial spillovers** (does a shift in one community cause shifts in neighbors?) but they assume a fixed, known spatial weights matrix W.
- For your problem, the "communities" are clusters of counties — the spatial structure is already partially absorbed into the community definition. The remaining question is how communities relate to each other, which may not be purely geographic.
- **Best use case:** Use PySAL to compute spatial weights between communities and test for spatial autocorrelation in election results. If significant, incorporate a spatial lag or error term. But don't rely on spatial econometrics alone — it captures geographic proximity but not demographic/economic similarity.
- **Key limitation:** Standard spatial econometric models are cross-sectional (one election at a time). For panel data (multiple elections), you'd need spatial panel models, which are less well-supported.

---

## 5. Kalman Filters and State-Space Models

### 5.1 The Approach

Model each community's latent political state as an unobserved time series that evolves via a random walk or AR(1) process. Polls are noisy measurements of this state. As new polls arrive, the Kalman filter updates the state estimate and its uncertainty.

The key insight: by specifying a **cross-community covariance** in the state transition, the Kalman filter propagates information from polled communities to unpolled ones.

### 5.2 Election Modeling Applications

**Jim Savage's state-space poll aggregation in Stan:**
- Models each candidate's preference share as an unobserved state, with polls as noisy measurements. State moves via random walk with ~0.25% daily standard deviation.
- Blog post: [Gelman's blog](https://statmodeling.stat.columbia.edu/2016/08/06/state-space-poll-averaging-model/)
- Python/Stan replica: [eliflab/polling_model_py](https://github.com/eliflab/polling_model_py)

**The Economist / 538 approach:**
- Both essentially use state-space models. The latent state is the "true" voter preference in each state, which evolves over time. Polls are noisy observations. The hierarchical structure provides the cross-state covariance that allows information to flow between states.
- 538 model Python replica: [jseabold/538model](https://github.com/jseabold/538model)

**Stan's built-in state-space support:**
- `gaussian_dlm_obs` distribution computes the log-likelihood for Gaussian linear state-space models when system matrices are time-invariant.
- Full tutorial: [ssmodels-in-stan](https://github.com/jrnold/ssmodels-in-stan) by Jeffrey Arnold.

### 5.3 Ensemble Kalman Filter for High-Dimensional Problems

For 5,000 communities, the standard Kalman filter requires storing and updating a 5,000 x 5,000 covariance matrix — expensive but feasible. The **Ensemble Kalman Filter (EnKF)** provides an alternative:
- Represent the state distribution with an ensemble of M samples (e.g., M=100).
- The covariance is implicitly estimated from the ensemble.
- **Covariance localization** forces the empirical covariance to respect spatial/network structure, preventing spurious long-range correlations from small ensembles.
- Paper: [Penalized EnKF for high-dimensional systems](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0248046)

### 5.4 Implementations

| Package | Language | What It Does | URL |
|---------|----------|-------------|-----|
| **FilterPy** | Python | Kalman filter, Extended KF, Unscented KF, Ensemble KF, particle filters, smoothers. Companion book: "Kalman and Bayesian Filters in Python." | [filterpy.readthedocs.io](https://filterpy.readthedocs.io/en/latest/) |
| **pykalman** | Python | Kalman Filter, Smoother, EM algorithm for parameter learning. Dead-simple API. | [pypi.org/project/pykalman](https://pypi.org/project/pykalman/) |
| **ensemblefilters** | Python | Collection of ensemble square root Kalman filters with covariance localization. | [GitHub](https://github.com/mchoblet/ensemblefilters) |
| **Stan** (state-space) | Stan/Python/R | `gaussian_dlm_obs` for linear state-space. Full Bayesian inference including parameter uncertainty. | [Stan User's Guide](https://mc-stan.org/docs/) |
| **statsmodels** (state-space) | Python | `UnobservedComponents`, `DynamicFactorMQ` — state-space models with ML or EM estimation. | [statsmodels](https://www.statsmodels.org/) |

**Assessment for this project:**
- **For real-time poll updating:** This is the natural framework. As polls come in for a subset of communities, the Kalman filter (or its Bayesian equivalent in Stan) propagates that information to all communities via the covariance structure.
- **The covariance structure is the hard part.** The Kalman filter takes Q (state transition covariance) as given — estimating Q is where the factor model / graphical lasso / spatial structure from other sections comes in.
- **Recommended approach:** Use Stan's state-space capabilities to implement a Bayesian state-space model where the latent state is community-level vote intention, the transition covariance Q is structured (factor model or block-diagonal), and polls are observations with known measurement error. This gives you full posterior uncertainty and handles missing data naturally.
- **The Ensemble Kalman Filter** is a viable alternative if Stan is too slow. With 5,000 communities and an ensemble of 100–500 members, it's computationally efficient, but you lose full Bayesian uncertainty quantification.

---

## 6. MRP (Multilevel Regression and Poststratification)

### 6.1 The Approach

MRP estimates opinion in small areas by:
1. Fitting a multilevel model of individual survey response as a function of demographics and geography (with random effects for each grouping variable).
2. Poststratifying: weighting the model predictions by the known population composition of each area.

For your problem, MRP could estimate community-level opinion from national polls by treating "community" as a grouping variable.

### 6.2 Key Papers and Developments

- **Gelman & Little (1997):** Original MRP framework.
- **Lax & Phillips (2009):** Validated MRP for state-level opinion estimation.
- **Ghitza & Gelman (2013):** "Deep Interactions with MRP" — extended to deeply interacted subgroups (age x race x income x state). Published in AJPS. [Paper PDF](https://sites.stat.columbia.edu/gelman/research/published/misterp.pdf). [Replication data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PZAOO6).
- **Gao et al. (2021):** "Improving MRP with structured priors" — introduces autoregressive, random-walk, and spatial (CAR/ICAR) structured priors for ordinal and geographic grouping variables. This is directly relevant: it shows how to add spatial structure to MRP. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203002/).
- **Gelman et al. (MRPW):** Regression, poststratification, and small-area estimation with survey weights. [Working paper](https://sites.stat.columbia.edu/gelman/research/unpublished/weight_regression.pdf).

### 6.3 Computational Challenges with 5,000 Communities

Adding "community" as a random effect with 5,000 levels is computationally demanding but not unprecedented:
- **Stan/brms:** Will work but sampling will be slow (thousands of latent variables). Use non-centered parameterization.
- **R-INLA:** Much faster for this scale. INLA handles thousands of spatial random effects routinely.
- **Structured priors help:** An ICAR prior on the community random effects (communities that are geographic/demographic neighbors share information) dramatically reduces the effective number of parameters.

### 6.4 Adding Non-Standard Grouping Variables

To add "community" to a standard MRP model:
- Treat it as an additional random effect: `(1 | community)` in lme4/brms syntax.
- Optionally add a structured prior (autoregressive within blocs, or ICAR across neighboring communities).
- The poststratification table needs to include community membership, which you'd derive from census data + your community definitions.

### 6.5 Implementations

| Package | Language | What It Does | URL |
|---------|----------|-------------|-----|
| **rstanarm** | R (Stan) | Pre-compiled MRP models. `stan_glmer()` with random effects. Built-in MRP vignette. | [mc-stan.org/rstanarm/articles/mrp](https://mc-stan.org/rstanarm/articles/mrp.html) |
| **brms** | R (Stan) | Formula interface supporting CAR, SAR, ICAR spatial structures for random effects. More flexible than rstanarm. | [CRAN](https://cran.r-project.org/web/packages/brms/) |
| **PyMC** | Python | Fully Bayesian MRP. Port of Kastellec's MRP primer to PyMC3 exists. | [MRPyMC3](https://austinrochford.com/posts/2017-07-09-mrpymc3.html) |
| **shinymrp** | R | Interactive MRP interface on CRAN. | [CRAN](https://cran.r-project.org/web/packages/shinymrp/) |
| **R-INLA** | R | Fast approximate Bayesian inference for MRP with spatial random effects. Handles 5,000+ groups. | [r-inla.org](https://www.r-inla.org/) |
| **MRP case studies book** | R/Stan | Comprehensive tutorial with worked examples. | [bookdown](https://bookdown.org/jl5522/MRP-case-studies/) |

**Assessment for this project:**
- MRP is the standard approach for going from national/state polls to small-area estimates. It's well-validated for elections.
- **The challenge is 5,000 communities.** Standard MRP with `(1 | state)` (50 levels) is easy. With 5,000 community levels, you need either R-INLA (for speed) or structured priors (ICAR/autoregressive) to regularize.
- **Key recommendation:** Use the structured prior framework from Gao et al. (2021): put an ICAR prior on the community random effects, where the adjacency graph is defined by your community similarity structure. This gives you spatial smoothing across communities without estimating 5,000 free parameters.
- MRP and the Kalman filter approach (Section 5) can be combined: use MRP to estimate community-level baseline opinion, then use the state-space model to update as polls come in.

---

## 7. Collaborative Filtering / Matrix Completion

### 7.1 The Analogy

| Election Problem | Collaborative Filtering |
|-----------------|------------------------|
| Community | User |
| Election | Item |
| Vote share | Rating |
| Missing polls | Missing ratings |
| Community similarity | User similarity |

The problem is: given a partially observed 5,000 x 12 matrix (communities x elections), fill in the missing entries. In collaborative filtering, this is matrix completion.

### 7.2 Methods

**SVD / PCA:**
- Decompose the matrix into U Σ V'. Keep the top k singular values. This gives you a low-rank approximation that smooths out noise and fills in missing data.
- For your 5,000 x 12 matrix with T=12, this is essentially PCA — at most 12 components exist.

**NMF (Non-negative Matrix Factorization):**
- Like SVD but enforces non-negativity. More interpretable: each factor is a "type" of political behavior, and loadings represent how much each community exhibits that type.

**ALS (Alternating Least Squares):**
- Standard collaborative filtering algorithm. Iteratively solves for user factors and item factors. Handles missing data naturally.
- **Very relevant:** ALS treats the incomplete matrix exactly as your problem presents it — you observe some entries (past elections) and want to predict others (current election in unpolled communities).

**SoftImpute / Nuclear norm minimization:**
- Matrix completion by iterative soft-thresholded SVD. Finds the lowest-rank matrix consistent with the observed entries.
- Equivalent to nuclear norm minimization (convex relaxation of rank minimization).

### 7.3 Implementations

| Package | Language | What It Does | URL |
|---------|----------|-------------|-----|
| **implicit** | Python | ALS for implicit feedback. Multi-threaded CPU and CUDA GPU support. Very fast. | [GitHub](https://github.com/benfred/implicit) |
| **surprise** | Python | Explicit-feedback recommender library. SVD, NMF, KNN, baseline models. | [surpriselib.com](http://surpriselib.com/) |
| **fancyimpute** | Python | Matrix completion via SoftImpute, IterativeSVD, NuclearNormMinimization, MatrixFactorization. | [GitHub](https://github.com/iskandr/fancyimpute) |
| **softImpute** | R | Matrix completion via iterative soft-thresholded SVD. The original implementation. | [CRAN](https://cran.r-project.org/web/packages/softImpute/) |
| **sklearn.decomposition.NMF` / `TruncatedSVD** | Python | Standard matrix factorization. No missing data handling. | [scikit-learn](https://scikit-learn.org/) |

### 7.4 Applications to Political Data

No direct published applications of collaborative filtering to election prediction at the community/county level were found in the literature. However:
- **Voting Advice Applications (VAAs)** use collaborative filtering to match voters to parties, treating policy statements as items and voter responses as ratings. Source: [Frontiers](https://www.frontiersin.org/journals/political-science/articles/10.3389/fpos.2024.1286893/full).
- **Legislative vote prediction** uses matrix factorization: legislators are "users," bills are "items," votes are "ratings." This is structurally identical to your problem.

**Assessment for this project:**
- **This is a powerful framing** that hasn't been fully exploited for area-level election analysis. The 5,000 x 12 matrix is very "wide and short" — with only 12 elections, the low-rank structure is essentially given (at most 12 factors). The question is whether 2–3 factors capture enough of the covariance.
- **fancyimpute's SoftImpute** is the most directly applicable: give it the incomplete matrix (missing = current election polls), and it returns the completed matrix with predicted vote shares.
- **The key limitation:** Pure matrix completion doesn't give you uncertainty estimates. You'd need to combine it with a Bayesian approach (or bootstrap) for confidence intervals.
- **Hybrid approach:** Use matrix factorization to estimate the factor structure, then embed those factors as the covariance structure in a Bayesian model (Section 2) for proper uncertainty quantification.

---

## 8. Gaussian Process Models for Elections

### 8.1 The Approach

Gaussian processes define a distribution over functions: f(x) ~ GP(m(x), k(x, x')). For communities, x could encode community features (demographics, geography, economics), and the kernel k defines the covariance structure. GPs give you both predictions and uncertainty for free.

### 8.2 Key Work

**Flaxman, Wang, & Smola (2015):**
- "Who Supported Obama in 2012? Ecological Inference through Distribution Regression." Won Best Student Paper at KDD 2015.
- Used distribution regression (related to GPs) to infer voter support patterns from aggregate election results combined with individual-level covariates.
- [Seth Flaxman's papers page](https://sethrf.com/papers/)

**Flaxman et al. — Scalable Kronecker GPs:**
- Developed Fast Kronecker Inference in Gaussian Processes with non-Gaussian Likelihoods, enabling scalable GP inference for spatiotemporal count processes.
- Applied to crime forecasting and disease modeling with similar structural features (many spatial units, temporal evolution).
- [ICML 2015 paper](https://proceedings.mlr.press/v37/flaxman15.html)

**UK Election Spatial Models:**
- Multilevel spatial models for UK elections (2019, 2024) combine GPs with Markov random fields to capture varying degrees of spatial cohesion.
- Source: [Springer](https://link.springer.com/article/10.1007/s12061-023-09563-6), [JRSSC](https://academic.oup.com/jrsssc/advance-article/doi/10.1093/jrsssc/qlaf055/8307260)

### 8.3 Kernel Design for Community Similarity

For your problem, the kernel k(community_i, community_j) should capture:
- **Demographic similarity:** Communities with similar age/race/education/income distributions should have similar political behavior.
- **Geographic proximity:** Nearby communities may share media markets, labor markets, cultural features.
- **Economic structure:** Communities with similar industries/employment patterns.
- **Past voting patterns:** Historical vote share is the strongest predictor of future vote share.

A composite kernel could be: k = k_demographic + k_geographic + k_economic, or a product kernel: k = k_demographic * k_geographic.

### 8.4 Scalability for 5,000 Communities

Standard GPs are O(N^3) — for N=5,000, that's ~125 billion operations per evaluation. Solutions:

| Method | Complexity | Package |
|--------|-----------|---------|
| **Sparse GP (inducing points)** | O(N M^2) where M << N | GPyTorch, GPflow |
| **HSGP (Hilbert Space GP)** | O(N m) where m = number of basis functions | PyMC (`gp.HSGP`) |
| **Kronecker structure** | O(D N^{(D+1)/D}) for D-dimensional inputs | GPyTorch |
| **FITC / VFE** | O(N M^2) | GPflow |

### 8.5 Implementations

| Package | Language | Strengths | URL |
|---------|----------|-----------|-----|
| **GPyTorch** | Python (PyTorch) | Most scalable GP library. Sparse GP, inducing points, Kronecker methods, CUDA support. | [gpytorch.ai](https://docs.gpytorch.ai/), [GitHub](https://github.com/cornellius-gp/gpytorch) |
| **GPflow** | Python (TensorFlow) | Best multi-output GP support. Good balance of ease-of-use and flexibility. SVGP (Stochastic Variational GP) for large datasets. | [gpflow.github.io](https://www.gpflow.org/) |
| **PyMC HSGP** | Python | Hilbert Space GP approximation integrated into PyMC's probabilistic programming. Works with NUTS sampler. Best for 1–2D inputs. | [PyMC docs](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html) |
| **scikit-learn GaussianProcessRegressor** | Python | Simple API but not scalable beyond ~1,000 points. | [scikit-learn](https://scikit-learn.org/) |

**Assessment for this project:**
- GPs are attractive because they naturally handle the "predict with uncertainty" requirement and the "similar communities should have similar outcomes" assumption.
- **GPyTorch with sparse GPs** is the recommended implementation. With ~500 inducing points and a well-designed composite kernel, you can handle 5,000 communities.
- **PyMC's HSGP** is excellent if you want to integrate the GP into a larger Bayesian model (e.g., as the prior on community random effects in an MRP-like model). However, HSGP works best for low-dimensional inputs — if your community feature space is high-dimensional, GPyTorch is better.
- **GP vs. factor model trade-off:** Factor models are simpler and more interpretable for the covariance structure. GPs are more flexible but harder to interpret. For community-level election modeling, I'd start with factor models and graduate to GPs only if the factor structure is insufficient.

---

## 9. Recommended Architecture

Based on this research, here is a suggested architecture that combines the most relevant methods:

### Phase 1: Discover the Covariance Structure
1. **PCA / factor analysis** (`sklearn`) on the 5,000 x 12 matrix of community-level vote shares (or vote-share changes). Determine how many factors are needed (likely 2–4).
2. **Spatial autocorrelation tests** (`PySAL`) to verify that the factor structure aligns with geographic patterns.
3. **Matrix completion** (`fancyimpute.SoftImpute`) to fill in any missing election data and validate the low-rank assumption.

### Phase 2: Build the Hierarchical Model
4. **Bayesian hierarchical model** (`cmdstanpy` or `PyMC`) with:
   - Community-level random effects structured by the factor model (θ_i = Λ_i f + ε_i)
   - Bloc-level grouping for partial pooling
   - ICAR or CAR prior on community random effects for spatial smoothing (if factor structure alone is insufficient)
   - Adapt the Economist model's correlated-prior approach, replacing the 51x51 state correlation matrix with a factor-structured prior

### Phase 3: Real-Time Updating
5. **State-space model** (Stan's `gaussian_dlm_obs` or custom) where:
   - The latent state is community-level vote intention
   - The state transition covariance Q comes from the factor model (Phase 1)
   - Polls are noisy observations
   - As polls arrive, the filter updates all community estimates via the covariance structure

### Phase 4: Validate and Refine
6. **Leave-one-election-out cross-validation:** For each past election, hold it out, estimate the model from the remaining elections + simulated sparse polls, and check prediction accuracy.
7. **MRP** for integrating individual-level survey data (if available) with the community-level model.

### Key Packages Summary

| Role | Primary Tool | Backup |
|------|-------------|--------|
| Factor analysis | `sklearn.decomposition.PCA` / `FactorAnalysis` | `statsmodels.DynamicFactorMQ` |
| Bayesian modeling | `cmdstanpy` + custom Stan code | `PyMC` v5 |
| Spatial weights/tests | `PySAL` (libpysal + esda) | `spdep` (R) |
| Covariance regularization | `skggm` (network-informed graphical lasso) | `sklearn.covariance.LedoitWolf` |
| Shrinkage estimation | `sklearn.covariance.LedoitWolf` / `OAS` | Custom Ledoit-Wolf |
| Matrix completion | `fancyimpute` (SoftImpute) | `implicit` (ALS) |
| Kalman filtering | Stan state-space / `FilterPy` | `pykalman` |
| Scalable GPs | `GPyTorch` | PyMC `HSGP` |
| MRP | `brms` / `rstanarm` (R) | `PyMC` |
| Fast spatial Bayes | `R-INLA` | — |
| Posterior diagnostics | `ArviZ` | — |

---

## 10. Key References

### Directly Applied to Elections
- Linzer (2013). "Dynamic Bayesian Forecasting of Presidential Elections in the States." JASA.
- Heidemanns, Gelman, Morris (2020). "An Updated Dynamic Bayesian Forecasting Model for the U.S. Presidential Election." HDSR.
- Ghitza & Gelman (2013). "Deep Interactions with MRP." AJPS.
- Gao et al. (2021). "Improving MRP with Structured Priors." Bayesian Analysis.
- Flaxman, Wang, Smola (2015). "Who Supported Obama in 2012?" KDD.
- The Economist model: [GitHub](https://github.com/TheEconomist/us-potus-model)

### Methodological Foundations
- Besag, York, Mollié (1991). BYM model for areal data.
- Riebler et al. (2016). BYM2 reparameterization.
- Ledoit & Wolf (2004). Shrinkage covariance estimation.
- Mazumder, Hastie, Tibshirani (2010). SoftImpute for matrix completion.

### Open-Source Code
- Economist Stan model: https://github.com/TheEconomist/us-potus-model
- State-space models in Stan: https://github.com/jrnold/ssmodels-in-stan
- Poll aggregation in Stan/Python: https://github.com/eliflab/polling_model_py
- 538 model replica: https://github.com/jseabold/538model
- MRP case studies: https://bookdown.org/jl5522/MRP-case-studies/
- MRP in PyMC3: https://austinrochford.com/posts/2017-07-09-mrpymc3.html
- BYM model in Stan: https://mc-stan.org/learn-stan/case-studies/icar_stan.html
