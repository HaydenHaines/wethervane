# Cross-Disciplinary Methods for Political Community-Covariance Modeling

## Research Summary — March 2026

**The structural problem:** Detect ~5,000 communities of people who covary in political behavior, using non-political data (religion, class, neighborhood, migration, commuting, social networks). Then propagate sparse, noisy polling observations across the community network to estimate community-level political shifts. This problem has direct structural analogs in at least eight other fields.

---

## 1. Epidemiology / Disease Modeling

### The structural analog

Epidemiologists face an identical problem structure: they observe disease counts at geographic units (counties, districts), know something about the contact/mobility structure connecting those units, and need to infer the latent transmission dynamics and community structure from sparse, noisy spatial observations. The "communities" are groups of people who transmit disease to each other; the "polls" are case counts.

### Most transferable methods

#### 1a. The `surveillance` R package — Endemic-Epidemic (HHH4) Model

This is the single most transferable tool from epidemiology. The `hhh4` model decomposes count time series across areal units into three additive components:

- **Endemic component:** baseline rate driven by covariates (analogous to baseline partisanship driven by demographics)
- **Autoregressive component:** dependence on the unit's own past (analogous to partisan inertia)
- **Neighbor-driven epidemic component:** propagation from neighboring units weighted by a spatial coupling matrix W (analogous to political contagion/covariation across communities)

The transmission weights w_ji can be estimated parametrically as a function of adjacency order or derived from movement/network data. This is structurally identical to propagating poll signals across a community network.

- **Package:** `surveillance` (R), CRAN
- **URL:** https://surveillance.r-forge.r-project.org/
- **Key vignette:** [hhh4_spacetime](https://cran.r-project.org/web/packages/surveillance/vignettes/hhh4_spacetime.pdf)
- **Transfer to political model:** Replace disease counts with vote shares or poll observations. The endemic component captures baseline partisanship from demographics. The epidemic component captures how a political shift in one community propagates to linked communities. The spatial coupling matrix W encodes your community similarity network. The model handles multivariate areal time series natively and estimates the propagation parameters from data.

#### 1b. EpiModel — Network-Based Epidemic Simulation with ERGMs

EpiModel uses temporal Exponential Random Graph Models (ERGMs) to simulate epidemic spread on networks. The network is estimated from empirical data on contacts, then disease dynamics are simulated on top of it. The ERGM framework is directly relevant to estimating the community similarity network from observed data.

- **Package:** `EpiModel` (R), CRAN
- **URL:** https://www.epimodel.org/
- **GitHub:** https://github.com/statnet/EpiModel
- **Transfer:** The ERGM framework for estimating network structure from node attributes and observed interactions could be used to estimate the community similarity network. Rather than simulating disease, you'd simulate political influence propagation. The statnet ecosystem (on which EpiModel is built) provides network estimation tools that work independently of the epidemic simulation.

#### 1c. CARBayes — Bayesian Spatial Areal Models

CARBayes implements conditional autoregressive (CAR) models for areal data with Bayesian inference via MCMC. It supports BYM, Leroux, and localized models for univariate and multivariate spatial data. Disease mapping is its primary use case, but the underlying problem (borrowing strength across spatially linked areas with sparse observations) is identical.

- **Package:** `CARBayes` (R), CRAN
- **URL:** https://www.rdocumentation.org/packages/CARBayes/
- **Transfer:** The BYM model with a community-adjacency weight matrix provides spatial smoothing of poll estimates across linked communities. Supports Poisson, binomial, and Gaussian responses. Already used in small-area estimation contexts structurally identical to poll propagation.

#### 1d. Phylogeographic Methods — BEAST

BEAST (Bayesian Evolutionary Analysis by Sampling Trees) infers population structure from the spatial spread of genetic sequences. The structured coalescent models migration between discrete populations, estimating a migration matrix that relates spatial units. The MASCOT-Skyline extension integrates population dynamics with migration.

- **Package:** BEAST 2 (Java, cross-platform)
- **URL:** https://www.beast2.org/
- **Visualization:** SPREAD (https://beast.community/spread)
- **Transfer:** The structured coalescent's migration matrix estimation is conceptually similar to estimating the flow of political influence between communities. Less directly applicable than the surveillance package, but the Bayesian framework for inferring population structure from spatial observations is relevant methodologically.

---

## 2. Ecology / Biogeography

### The structural analog

Ecologists model communities of species across geographic sites. Species co-occur because of shared environmental requirements, biotic interactions, and dispersal — structurally identical to how demographic groups co-occur in geographic areas because of shared economic conditions, cultural affinity, and migration. The "species" are demographic/political types; the "sites" are counties; the question is what drives co-occurrence patterns and how to estimate the latent community structure.

### Most transferable methods

#### 2a. HMSC (Hierarchical Modelling of Species Communities) — The Most Direct Analog

HMSC is a joint species distribution model that estimates:
1. **Species responses to environmental covariates** (= community responses to demographic/economic features)
2. **Residual species-to-species association matrices** at multiple spatial scales (= the latent covariance structure among political communities after controlling for observables)
3. **Random effects** at multiple spatial scales capturing unmeasured environmental variation

The residual association matrix is the key output: it captures the covariance among species (communities) that is NOT explained by measured environmental variables. This is exactly the "unexplained political covariance" you want to estimate.

HMSC uses Bayesian inference with MCMC and can handle:
- Multiple response types (presence-absence, counts, continuous)
- Latent factor structure for the residual associations (avoids estimating a full NxN covariance matrix)
- Spatial random effects at multiple scales
- Species traits that predict responses to environment (= community features that predict political behavior)

- **Package:** `Hmsc` (R), CRAN
- **GitHub:** https://github.com/hmsc-r/HMSC
- **Key paper:** Tikhonov et al. (2020). "Joint species distribution modelling with the R-package Hmsc." Methods in Ecology and Evolution.
- **URL:** https://www.helsinki.fi/en/researchgroups/statistical-ecology/software/hmsc
- **Transfer to political model:** This is potentially the most directly transferable framework. Map the problem as follows:

| Ecology (HMSC) | Political model |
|----------------|-----------------|
| Sites | Counties or geographic units |
| Species occurrences | Community prevalence (% of county population in each community) |
| Environmental covariates | County-level demographics, economics, geography |
| Species traits | Community-level features (e.g., median income, education, urbanity) |
| Residual species associations | Latent covariance among communities not explained by observables |
| Spatial random effects | Geographic clustering of communities |
| Phylogeny | Community similarity hierarchy |

The latent factor approach in HMSC is particularly relevant: rather than estimating a full 5,000x5,000 covariance matrix, HMSC uses a small number of latent factors to capture the residual associations. This is mathematically identical to the factor-structured covariance approach in the main methods document (Section 1), but with the full Bayesian machinery for uncertainty quantification.

**Limitation:** HMSC is designed for ecological data scales (hundreds to thousands of sites, tens to hundreds of species). With 5,000 communities across 3,000+ counties, computational scaling may be challenging. The latent factor structure helps, but MCMC inference is inherently slow.

#### 2b. `bioregion` R Package — Bioregionalization as Community Detection

The `bioregion` package implements a comprehensive bioregionalization workflow: take a sites-by-species matrix, compute similarity, apply clustering (hierarchical, partitional, or network-based community detection), and evaluate the results. It includes Infomap and OSLOM algorithms from network theory.

- **Package:** `bioregion` (R), CRAN
- **URL:** https://cran.r-project.org/web/packages/bioregion/
- **Key paper:** Denelle et al. (2025). "Bioregionalization analyses with the bioregion R-package." Methods in Ecology and Evolution.
- **Transfer:** The bioregionalization workflow is directly applicable to detecting political communities. Replace the sites-by-species matrix with a counties-by-features matrix (demographic, economic, cultural features). The package's network-based community detection algorithms (especially Infomap on bipartite networks) would cluster counties into communities based on shared feature profiles. The evaluation framework (cluster validity metrics, comparison across methods) is directly useful.

#### 2c. Variation Partitioning in Metacommunity Ecology

Metacommunity ecology has developed a formal framework for partitioning variation in community composition into:
- **E:** Environmental filtering (demographics drive community composition)
- **S:** Spatial processes (geography/dispersal creates spatial autocorrelation)
- **E-S overlap:** Spatially structured environmental variation
- **Residual:** Unexplained variation

This decomposition, typically done with canonical analysis (RDA/CCA) and partial regression, could be applied to political data to understand how much of the political covariance structure is explained by demographics vs. spatial proximity vs. their interaction.

- **Package:** `vegan` (R), CRAN — function `varpart()`
- **Transfer:** Before building the full model, use variation partitioning to understand how much of county-level political variation is explained by demographics alone, by geography alone, and by their intersection. This tells you whether a purely demographic community definition is sufficient or whether you need spatial structure in the community definitions.

---

## 3. Marketing / Consumer Analytics

### The structural analog

Geodemographic segmentation systems (PRIZM, Mosaic, Tapestry) solve exactly the problem of grouping geographic areas into behaviorally similar communities. They then use these segments to predict behavior (purchasing, media consumption) for areas where direct observations are sparse. The "segment X in location A behaves like segment X in location B" assumption is precisely the covariation assumption in the political model.

### Most transferable methods

#### 3a. Output Area Classification (OAC) — Open-Source Geodemographic Segmentation

OAC is the UK's free, open-source geodemographic classification. It classifies small areas into a three-tier hierarchy (8 supergroups, 21 groups, 52 subgroups) using k-means clustering on census variables across five domains: demographic structure, household composition, housing, socio-economic indicators, and employment.

The methodology is fully documented and reproducible:
1. Select ~60 census variables
2. Standardize (range standardization)
3. Apply k-means clustering at three hierarchical levels
4. Evaluate using within-cluster variance

- **URL:** https://data.geods.ac.uk/dataset/output-area-classification-2021
- **Documentation:** https://apps.cdrc.ac.uk/static/OAC.pdf
- **2021 update paper:** Denelle et al., "A neighbourhood Output Area Classification from the 2021 Census"
- **Transfer:** The OAC methodology is a directly replicable template for building political communities from US Census/ACS data. Replace UK census variables with ACS variables across similar domains. The three-tier hierarchical structure (supergroups > groups > subgroups) maps onto the political model's need for nested community definitions. The key advantage is that OAC's methodology is fully open, peer-reviewed, and designed for exactly this type of geographic segmentation.

#### 3b. The "Segment Transfer" Problem and How Marketing Solves It

The marketing industry handles "people in segment X in location A behave like people in segment X in location B" through:

1. **National calibration panels:** A panel of consumers with known behavior (analogous to densely polled areas) calibrates the segment-level behavior model. This model is then applied to all locations based on their segment composition.
2. **Propensity scoring by segment:** Each segment gets a propensity score for each behavior (purchase likelihood, media preference). The location-level prediction is the population-weighted average of segment propensities. This is structurally identical to decomposing a county's expected vote share as a weighted average of community-level vote shares.
3. **Bayesian updating with local data:** When local data exists (analogous to a poll), it updates the segment-level propensities via Bayesian shrinkage toward the national estimate. Nielsen's methodology for local TV ratings does exactly this.

- **Transfer:** The propensity-by-segment architecture directly maps to the political model. Define communities (segments), estimate community-level political propensities from densely polled areas, then apply those propensities to all areas based on their community composition. When a new poll arrives for an area, update the community-level estimates via Bayesian shrinkage.

#### 3c. Market Basket Transformer — Transfer Learning Across Segments

The Market Basket Transformer (MBT) is a recent foundation model for retail data that learns general-purpose segment representations from transaction data, then transfers to downstream tasks. The key insight: behavioral patterns learned in one context (purchasing) transfer across locations when mediated by segment membership.

- **Paper:** Gabel & Ringel. "The Market Basket Transformer." SSRN (2023).
- **URL:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4335141
- **Transfer:** The transfer-learning architecture could apply to political data. Train a model on densely observed areas (many polls, past elections) to learn community-level political representations. Transfer those representations to sparsely observed areas based on community composition. The MBT was pretrained on 20 million baskets — for the political problem, the "baskets" would be election results from counties across multiple elections.

#### 3d. Open-Source Segmentation Tools

Several open-source tools replicate the commercial geodemographic segmentation workflow:

- **K-means on census data:** Available via scikit-learn (Python) or base R. The Geographic Data Service provides a tutorial: [Creating a Geodemographic Classification Using K-means Clustering in R](https://data.geods.ac.uk/dataset/creating-a-geodemographic-classification-using-k-means-clustering-in-r)
- **Customer segmentation libraries:** Multiple GitHub projects implement RFM analysis + k-means for customer segmentation (e.g., https://github.com/topics/customer-segmentation). The methodology transfers directly to geographic-unit segmentation.

---

## 4. Recommendation Systems / Collaborative Filtering

### The structural analog

The recommendation problem is: given a sparse matrix of user-item ratings, predict missing entries by exploiting structure (similar users rate similar items similarly). The political analog: given a sparse matrix of community-poll observations, predict missing entries by exploiting structure (similar communities shift similarly in elections).

### Most transferable methods

#### 4a. LightFM — Hybrid Matrix Factorization with Side Features

LightFM is the most directly applicable recommendation library. Unlike standard matrix factorization (which only uses the interaction matrix), LightFM learns embeddings as linear combinations of feature embeddings. This means:
- Each community's latent representation is a function of its observable features (demographics, geography, economics)
- This allows generalization to communities with no poll data at all (the "cold start" problem)
- Side features regularize the factorization, preventing overfitting to sparse observations

For the political problem:
- **Users = communities**
- **Items = elections/polls**
- **Ratings = vote shares**
- **User features = community demographics, economics, geography**
- **Item features = election characteristics (year, office, national environment)**

LightFM would learn community embeddings that predict vote shares, regularized by demographic similarity. When a new poll arrives, it updates the community's embedding (and by extension, all similar communities' predictions).

- **Package:** `lightfm` (Python)
- **GitHub:** https://github.com/lyst/lightfm
- **Transfer:** Train on historical election results (communities x elections matrix) with community features as side information. Predict current-election vote shares for unpolled communities. The WARP loss function is designed for implicit feedback (which election results resemble — you observe vote shares, not explicit "ratings"). The cold-start capability is critical: even for communities with no polls, LightFM can predict based on demographic similarity to polled communities.

#### 4b. Iterative Collaborative Filtering for Sparse Matrix Estimation

The theoretical framework of Borgs, Chayes, Lee, and Shah (2017) provides guarantees for sparse matrix completion by propagating observations through a similarity network. The key insight: instead of comparing users directly (which fails when the matrix is very sparse), compare users by their extended neighborhoods in the similarity graph. This connects to belief propagation and the non-backtracking operator.

- **Paper:** "Thy Friend is My Friend: Iterative Collaborative Filtering for Sparse Matrix Estimation." NeurIPS 2017.
- **URL:** https://arxiv.org/abs/1712.00710
- **Transfer:** This provides theoretical justification for propagating poll observations through the community similarity network. The iterative algorithm — compare communities not just by direct feature similarity but by the similarity of their neighborhoods' political behavior — is exactly the multi-hop propagation the political model needs. The paper also connects matrix completion to community detection in stochastic block models, providing a bridge between the segmentation and estimation problems.

#### 4c. Surprise Library — Simple Collaborative Filtering Baseline

Surprise provides clean implementations of SVD, NMF, KNN-based collaborative filtering, and baseline models with proper evaluation (cross-validation, RMSE, MAE).

- **Package:** `surprise` (Python)
- **URL:** https://surpriselib.com/
- **GitHub:** https://github.com/NicolasHug/Surprise
- **Transfer:** Use Surprise to quickly benchmark how well simple matrix factorization predicts held-out election results from the communities x elections matrix. This provides a baseline against which more complex models (LightFM, Bayesian hierarchical) can be compared.

---

## 5. Remote Sensing / Spectral Unmixing

### The structural analog

A mixed pixel in a satellite image contains light from multiple ground cover types (vegetation, water, soil, urban). The observed spectrum is a weighted sum of the pure spectra ("endmembers"), where the weights are the fractional abundances. Spectral unmixing recovers the endmember spectra and abundances from the mixed observations.

The political analog is exact: a county's aggregate poll result is a weighted sum of community-level political signals, where the weights are the community's share of the county population. "Unmixing" recovers the community-level signals from the aggregate observations.

**This is the most structurally precise analog across all eight fields.**

### Most transferable methods

#### 5a. Linear Spectral Unmixing and the County Decomposition Problem

The linear mixing model states: y = M * a + n, where:
- y = observed mixed spectrum (= observed county poll result)
- M = endmember matrix, columns are pure spectra (= community-level political behavior profiles)
- a = abundance vector, fractions of each endmember (= community population shares in the county)
- n = noise

If you know M (endmember spectra = community political profiles) and y (observed poll), solving for a (abundances = community shares) is straightforward. But in the political problem, you know a (community population shares from census/ACS) and y (poll results) and want to solve for M (community-level political behavior). This is the **inverse** unmixing problem: given known abundances and observations, recover the endmember spectra.

With K communities and one poll observation per county, the system is underdetermined (one equation, K unknowns). But across many counties with different community compositions, the system becomes determined — this is identical to the ecological inference problem.

#### 5b. Constrained Least Squares Approaches

Spectral unmixing uses two constraints:
1. **Abundance non-negativity constraint (ANC):** abundances >= 0 (community shares are non-negative: satisfied by construction)
2. **Abundance sum-to-one constraint (ASC):** abundances sum to 1 (community shares sum to 1: satisfied by construction)

Since your "abundances" (community population shares) are known, you're solving the simpler problem of estimating endmembers from known abundances and observations. This is a constrained linear regression that the unmixing literature has extensively studied.

- **Fully Constrained Least Squares (FCLS):** Standard approach. Solve M = Y * A^+ (pseudoinverse) subject to constraints.

#### 5c. HySUPP — Comprehensive Unmixing Toolkit

HySUPP is the first open-source Python package to include supervised, semi-supervised, and blind unmixing methods. It provides 20+ algorithms including:
- Vertex Component Analysis (VCA) for endmember extraction
- SUNSAL for sparse unmixing
- Deep learning-based unmixing (autoencoders, deep NMF)
- Evaluation metrics for unmixing quality

- **Package:** `HySUPP` (Python)
- **GitHub:** https://github.com/BehnoodRasti/HySUPP (also https://github.com/inria-thoth/HySUPP)
- **License:** MIT
- **Paper:** Rasti et al. (2024). "Image Processing and Machine Learning for Hyperspectral Unmixing." IEEE TGRS.
- **Transfer:** The blind unmixing algorithms are most relevant. Given county-level poll results and community population shares, blind unmixing would simultaneously discover the community-level political profiles AND the mixing structure. Even though you know the population shares, the blind methods could discover latent community structure that differs from your initial community definitions.

#### 5d. QLSU Plugin and Python `unmixing` Library

- **QLSU:** Open-source QGIS plugin for linear spectral unmixing. Supports unconstrained, partially constrained, and fully constrained least squares. URL: https://www.sciencedirect.com/science/article/abs/pii/S1364815223001688
- **`unmixing`:** Python library for spectral mixture analysis with parallel FCLS. URL: https://github.com/arthur-e/unmixing
- **Transfer:** These provide the computational infrastructure for the linear unmixing step. The `unmixing` library's parallel FCLS implementation could handle thousands of counties efficiently.

#### 5e. NMF for Political Unmixing

Non-negative Matrix Factorization (NMF) is widely used for hyperspectral unmixing because it enforces the physical constraint that both endmember spectra and abundances must be non-negative. For the political problem:

- The counties-by-elections matrix V is factored into W * H, where W (counties x K communities) gives community memberships and H (K communities x elections) gives community-level political profiles.
- Spatial regularization variants (graph-regularized NMF, spatial group sparsity NMF) incorporate geographic adjacency, which is directly relevant.
- The nonnegative spatial factorization (NSF) method from spatial genomics uses transformed Gaussian processes to encourage spatially smooth factorizations — this could produce geographically coherent political communities.

- **Graph-regularized NMF paper:** Cai et al. "Graph Regularized Nonnegative Matrix Factorization." https://arxiv.org/pdf/1111.0885
- **NSF paper:** Townes & Bhatt (2022). "Nonnegative spatial factorization." Nature Methods.
- **Implementations:** `sklearn.decomposition.NMF` (basic), `robust-nmf` (https://github.com/neel-dey/robust-nmf), multiple implementations at https://github.com/topics/nonnegative-matrix-factorization
- **Transfer:** Graph-regularized NMF with a county adjacency graph would simultaneously discover community structure AND community-level political profiles, with spatial smoothing. This is a strong candidate for the community detection step of the model.

---

## 6. Topic Modeling / Text Analysis

### The structural analog

In LDA (Latent Dirichlet Allocation):
- Documents are mixtures of topics
- Each topic is a distribution over words
- The observed data is a bag of words per document

In the political community model:
- Counties are mixtures of communities
- Each community has a distribution over observable features (demographics, occupations, housing types)
- The observed data is a feature profile per county

The mapping is exact: counties = documents, communities = topics, demographic features = words.

### Most transferable methods

#### 6a. Structural Topic Model (STM) — Topic Discovery with Covariates

The Structural Topic Model extends LDA to allow document-level covariates (metadata) to affect both topic prevalence and topic content. For the political problem:
- Document-level covariates = county-level geographic/economic variables
- Topic prevalence covariates = what predicts which communities are present in which counties
- Topic content covariates = how community feature profiles vary by region

This is more powerful than standard LDA because it can model, e.g., how "suburban professional" communities differ between the Sun Belt and the Rust Belt.

- **Package:** `stm` (R), CRAN
- **Key paper:** Roberts, Stewart, Tingley. "The Structural Topic Model and Applied Social Science." Harvard WCFIA Working Paper.
- **URL:** https://projects.iq.harvard.edu/files/wcfia/files/stmnips2013.pdf
- **Transfer:** Apply STM to the counties-by-features matrix where features are census/ACS variables (treated as "word counts"). Geographic region, state, or metro area serve as covariates affecting topic prevalence. The discovered "topics" are candidate political communities. The prevalence covariates tell you which communities are geographically concentrated. This could serve as the community detection step, producing soft community memberships (each county has a probability distribution over communities, not hard assignments).

#### 6b. Standard LDA Implementations

For a simpler first pass:
- **gensim** (Python): Optimized LDA implementation. Handles large corpora efficiently. URL: https://radimrehurek.com/gensim/
- **scikit-learn LDA:** `sklearn.decomposition.LatentDirichletAllocation`. URL: https://scikit-learn.org/
- **Transfer:** Convert county feature profiles into "documents" by treating each demographic/economic variable as a "word" with frequency proportional to its value. Run LDA with K=target number of communities. Each topic = a community defined by its characteristic feature profile. Each county gets a topic distribution = community composition.

**Important caveat:** LDA assumes exchangeability of words within documents, which doesn't perfectly match the structure of county feature profiles. The Structural Topic Model is better because it can incorporate geographic covariates. However, standard LDA can serve as a fast baseline for community detection.

#### 6c. BERTopic — Neural Topic Modeling

BERTopic uses transformer embeddings + UMAP + HDBSCAN for topic modeling. It's more flexible than LDA and doesn't require specifying the number of topics in advance.

- **Package:** `bertopic` (Python)
- **URL:** https://bertopic.com/
- **GitHub:** https://github.com/MaartenGr/BERTopic
- **Transfer:** If you embed county feature profiles using a suitable encoder (a learned embedding of demographic/economic features), BERTopic's clustering pipeline could discover community structure without pre-specifying K. The HDBSCAN component handles variable-density clusters. However, the transformer component is designed for text and would need adaptation for tabular feature profiles. The UMAP + HDBSCAN clustering pipeline alone (without the BERT component) is more directly applicable.

---

## 7. Sensor Networks / Data Fusion

### The structural analog

Sensor networks have sparse, noisy readings at fixed locations and need to estimate a continuous field (temperature, pollution, soil moisture) across the entire domain. The "sensors" are polls, the "field" is community-level political sentiment, and the "network" connecting sensors is the community similarity structure.

The key challenge is the same: how to propagate sparse observations across a network to produce estimates everywhere, with calibrated uncertainty.

### Most transferable methods

#### 7a. PyGSP — Graph Signal Processing

PyGSP implements graph signal processing operations including signal interpolation on graphs. Given a signal observed at a subset of nodes, PyGSP can interpolate to all nodes using the graph's spectral structure (eigenvectors of the graph Laplacian).

This is conceptually different from Euclidean kriging: instead of interpolating based on physical distance, it interpolates based on graph distance and spectral structure. Two communities that are far apart geographically but connected in the community similarity network will share information.

- **Package:** `PyGSP` (Python)
- **URL:** https://pygsp.readthedocs.io/
- **GitHub:** https://github.com/epfl-lts2/pygsp
- **Key features:** Graph construction (from data), spectral analysis, filtering, interpolation, visualization
- **Transfer:** Build a community similarity graph (nodes = communities, edges weighted by demographic/economic/geographic similarity). Observe poll results at a subset of communities. Use PyGSP's graph interpolation to estimate poll results at all communities. The spectral approach provides natural regularization: low-frequency graph signals (smooth political trends) are preserved while high-frequency noise is suppressed.

**Specific workflow:**
1. Build the community graph G using PyGSP's graph constructors
2. Define a graph signal s (poll results) observed at a subset of nodes
3. Use `pygsp.learning.interpolate()` or spectral methods to recover s at all nodes
4. The graph Laplacian's eigenvalues quantify the smoothness of the recovered signal

This is a fast, non-Bayesian alternative to the full state-space model. It provides point estimates but not full posterior uncertainty.

#### 7b. Gaussian Belief Propagation on Factor Graphs

Belief propagation (BP) passes messages between nodes in a graphical model to compute marginal distributions. Gaussian BP handles continuous variables and converges to the correct answer on trees (and often converges on loopy graphs too).

For the political problem, construct a factor graph where:
- **Variable nodes** = community-level political states (the unknowns)
- **Factor nodes** = polls (observed noisy mixtures of community states) and community similarity constraints
- **Messages** propagate poll information through the community network

- **Package:** `pgmpy` (Python) — supports belief propagation on Bayesian networks and Markov random fields
- **URL:** https://pgmpy.org/
- **GitHub:** https://github.com/pgmpy/pgmpy
- **Gaussian BP tutorial:** https://gaussianbp.github.io/
- **Transfer:** Model each poll as a factor that constrains the weighted sum of community states (the communities present in the polled area). Model community similarity as pairwise factors that constrain similar communities to have similar states. Run Gaussian BP to compute the posterior distribution of each community's political state given all observed polls. This handles the propagation problem naturally and scales well.

#### 7c. Kriging on Networks / Graph Kernels

Standard kriging assumes Euclidean distance. For the community network, you need kriging on graph distances. This can be done with Gaussian Processes using graph kernels:

- **Diffusion kernel:** k(i,j) = exp(-t * L) where L is the graph Laplacian and t controls the smoothness
- **Regularized Laplacian kernel:** k(i,j) = (I + sigma^2 * L)^{-1}

These kernels define covariance structures on graphs that respect the network topology.

- **Package:** GPyTorch (Python) with custom kernels, or GPflow
- **URLs:** https://docs.gpytorch.ai/, https://www.gpflow.org/
- **Paper:** Smola & Kondor (2003). "Kernels and Regularization on Graphs." COLT.
- **Transfer:** Define a GP with a graph kernel on the community network. Observe poll results at some communities. The GP posterior gives you estimates and uncertainties at all communities, with the graph kernel ensuring that information flows through the network structure. This is the Bayesian version of PyGSP's interpolation.

---

## 8. Social Physics / Computational Social Science

### The structural analog

Social physics models how opinions, behaviors, and cultural traits spread through social networks. The Axelrod model, bounded confidence models, and voter models all predict that interacting agents converge on shared states — forming communities of aligned opinion. The question is whether these models can be calibrated to real political data to predict how political shifts propagate through community networks.

### Most transferable methods

#### 8a. Calibrated Opinion Dynamics Models

The critical gap in social physics has been empirical calibration — most models are validated only at the phenomenological level (can they produce polarization?) rather than against real survey data. Recent work has begun to close this gap:

- Lorenz (2017) was one of the first to validate opinion dynamics models against empirical survey data.
- Johnson et al. developed an adapted genetic algorithm for calibrating the DeGroot model to limited empirical data.
- Geschke et al. (2019) calibrated an opinion dynamics model to empirical opinion distributions and transitions, testing whether the model could reproduce both the static distribution of opinions AND the dynamics of opinion change observed in panel surveys.

- **Key paper:** Geschke et al. (2019). "Calibrating an Opinion Dynamics Model to Empirical Opinion Distributions and Transitions." JASSS 26(4).
- **URL:** https://www.jasss.org/26/4/9.html
- **Transfer:** These calibration methods could be applied to the political community model. Define communities as interacting agents with multi-dimensional opinion states. Calibrate interaction parameters (how much communities influence each other) from historical election data. Use the calibrated model to predict how a political shock in one community propagates to others. The calibration framework addresses the key weakness of social physics models — grounding them in empirical data.

#### 8b. The Axelrod Model and Cultural Dissemination

The Axelrod model combines two mechanisms directly relevant to the political community model:
1. **Homophily:** Agents interact more with similar agents (communities with similar demographics interact more)
2. **Social influence:** Interaction makes agents more similar (political behavior converges within interacting communities)

Key findings from the literature:
- Despite local convergence, global polarization emerges: communities become internally homogeneous but distinct from other communities.
- Homophily dynamics outweigh network topology in determining macro outcomes.
- The model has been extended with repulsion mechanisms, layered influence, and cultural drift.

- **Implementations:** Multiple open-source implementations exist on GitHub (search "axelrod model simulation"). The JASSS (Journal of Artificial Societies and Social Simulation) archive has extensive model code.
- **Transfer:** The Axelrod model provides a theoretical framework for understanding WHY political communities covary — it's not just shared demographics but ongoing social interaction that produces convergence. This is more of a theoretical lens than a practical tool, but it motivates the design of the community similarity network: connections should reflect interaction potential (shared media markets, commuting patterns, social networks), not just static demographic similarity.

#### 8c. Region2vec — GeoAI Community Detection on Spatial Networks

Region2vec is a family of GeoAI-enhanced community detection methods based on Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN). It generates node embeddings that capture attribute similarity, geographic adjacency, and spatial interactions, then extracts communities using agglomerative clustering.

This is the most practical tool from computational social science for the political community detection problem:
- It handles multiple edge types (geographic adjacency AND interaction flows)
- It combines node attributes (demographics) with network structure
- It produces communities that maximize both attribute similarity and interaction intensity

- **Package:** Python, open source
- **GitHub:** https://github.com/GeoDS/region2vec-GAT (GAT version), https://github.com/GeoDS/Region2vec (GCN version)
- **Paper:** Liang et al. (2025). "GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning." Computers, Environment and Urban Systems.
- **Transfer:** Apply region2vec directly to the political community detection problem. Nodes = counties. Node attributes = demographic/economic features. Edges = geographic adjacency + commuting flows + migration flows. The learned embeddings capture multi-dimensional similarity. Agglomerative clustering on the embeddings produces political communities. The GAT version with attention mechanisms can learn which types of connections matter most for political covariation.

---

## 9. Network Community Detection (Cross-Cutting)

Several tools span multiple fields and deserve mention for the community detection component of the problem.

#### 9a. Infomap — Information-Theoretic Community Detection

Infomap finds communities by minimizing the description length of a random walk on the network. It supports:
- Multi-level (hierarchical) community detection
- Bipartite networks (counties x features)
- Multiplex (multilayer) networks
- Overlapping communities

- **Package:** `infomap` (Python, C++)
- **URL:** https://www.mapequation.org/infomap/
- **GitHub:** https://github.com/mapequation/infomap
- **PyPI:** `pip install infomap`
- **Transfer:** Build a bipartite network of counties and features (demographic/economic variables). Run Infomap to detect communities of counties that share feature profiles. The multi-level output gives you a natural hierarchy: broad political blocs > communities > sub-communities.

#### 9b. Stochastic Block Models via graph-tool

The Stochastic Block Model (SBM) is a generative model for networks with community structure. The Mixed-Membership SBM (MMSBM) allows nodes to belong to multiple communities — essential if counties are mixtures of political communities.

- **Package:** `graph-tool` (Python, C++ backend)
- **URL:** https://graph-tool.skewed.de/
- **Key function:** `minimize_blockmodel_dl()` — fits SBMs by minimizing description length
- **Documentation:** https://graph-tool.skewed.de/static/doc/demos/inference/inference.html
- **Transfer:** Fit a mixed-membership SBM to the community similarity network. The inferred block structure gives you communities, and the mixed-membership parameters give you soft community assignments for each county. Graph-tool is among the fastest implementations available and handles networks with millions of edges.

#### 9c. scikit-network

A scikit-learn-inspired library for graph analysis including community detection, embedding, and classification on large sparse graphs.

- **Package:** `scikit-network` (Python)
- **URL:** https://scikit-network.readthedocs.io/
- **Transfer:** Provides a clean API for spectral clustering, Louvain community detection, and graph embedding that integrates with the scikit-learn ecosystem. Good for rapid prototyping of community detection approaches.

---

## 10. Summary: Highest-Impact Transfer Opportunities

Ranked by directness of structural analog and practical usability:

### Tier 1: Direct structural analogs with production-ready tools

| Method | Source Field | Tool | Why It Transfers |
|--------|-------------|------|-----------------|
| **Linear spectral unmixing** | Remote sensing | HySUPP, `unmixing`, sklearn NMF | County poll = weighted sum of community signals. Unmixing recovers community-level signals from aggregate observations. Exact mathematical analog. |
| **HHH4 endemic-epidemic model** | Epidemiology | `surveillance` (R) | Decomposes areal time series into baseline + autoregressive + neighbor-propagation components. Replace disease counts with poll results. Handles the spatial propagation problem natively. |
| **HMSC joint species distribution model** | Ecology | `Hmsc` (R) | Estimates covariance among "species" (communities) across "sites" (counties) from environmental covariates. Produces the residual association matrix that IS the political covariance structure. |
| **LightFM hybrid collaborative filtering** | Recommendation systems | `lightfm` (Python) | Matrix factorization with side features. Handles cold-start (unpolled communities) via feature-based embeddings. Most practical for poll propagation. |
| **Graph signal interpolation** | Sensor networks | PyGSP (Python) | Interpolates sparse poll observations across community network using spectral graph theory. Fast, simple, principled. |

### Tier 2: Strong methodological analogs requiring adaptation

| Method | Source Field | Tool | Why It Transfers |
|--------|-------------|------|-----------------|
| **Structural Topic Model** | Text analysis | `stm` (R) | Counties = documents, features = words, communities = topics. Discovers community structure from feature profiles with geographic covariates. |
| **Region2vec** | Computational social science | Python (GitHub) | GNN-based community detection on spatial networks with node attributes. Directly applicable to political community detection. |
| **Graph-regularized NMF** | Remote sensing / ML | sklearn NMF + custom regularization | Simultaneously discovers community structure and political profiles with spatial smoothing. |
| **Bioregionalization** | Ecology | `bioregion` (R) | Complete workflow for clustering geographic areas by composition profiles. Includes Infomap, OSLOM, hierarchical methods, evaluation metrics. |
| **Gaussian belief propagation** | Sensor networks / ML | pgmpy (Python) | Propagates poll information through community network via message passing. Natural uncertainty quantification. |

### Tier 3: Theoretical frameworks that inform model design

| Method | Source Field | Insight |
|--------|-------------|---------|
| **Metacommunity variation partitioning** | Ecology | Decompose political variation into environment (demographics) vs. space (geography) vs. their interaction. Tells you whether demographic-only community definitions are sufficient. |
| **Geodemographic segmentation (OAC)** | Marketing | Open-source, peer-reviewed template for building geographic classifications from census data. Directly replicable with US data. |
| **Axelrod model / opinion dynamics** | Social physics | Theoretical basis for WHY communities covary: homophily + social influence. Motivates using interaction data (commuting, media, social networks) rather than just static demographics for community definitions. |
| **Calibrated opinion dynamics** | Social physics | Methods for grounding agent-based models in empirical survey data. Could validate the community-level propagation model. |
| **Iterative collaborative filtering theory** | Recommendation systems | Theoretical justification for multi-hop poll propagation through similarity networks. Connects matrix completion to community detection. |

---

## Key Cross-Disciplinary Insight

The most important insight from this cross-disciplinary review is that the political community-covariance problem is not one problem but two:

1. **Community detection** (discovering the ~5,000 communities from non-political data): Best addressed by ecology (bioregionalization, HMSC), topic modeling (STM, LDA), network science (Infomap, SBM, region2vec), or marketing (geodemographic segmentation). These fields have mature methods for discovering latent group structure from multivariate feature profiles.

2. **Signal propagation** (estimating community-level political shifts from sparse polls): Best addressed by epidemiology (hhh4 propagation model), remote sensing (spectral unmixing), recommendation systems (collaborative filtering with side features), sensor networks (graph signal processing), and computational social science (belief propagation).

The two problems connect through the community similarity network: the same network that defines community structure (Problem 1) serves as the substrate for signal propagation (Problem 2). This suggests a unified approach where community detection and signal estimation are performed jointly — which is exactly what NMF, HMSC, and LightFM do in their respective domains.

---

## Sources

### Epidemiology
- [epinet: Epidemic/Network-Related Tools](https://cran.r-project.org/web/packages/epinet/index.html)
- [EpiModel: An R Package for Mathematical Modeling of Infectious Disease over Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC5931789/)
- [EpiModel](https://www.epimodel.org/)
- [Spatio-Temporal Analysis of Epidemic Phenomena Using the R Package surveillance](https://arxiv.org/pdf/1411.0416)
- [surveillance R package](https://surveillance.r-forge.r-project.org/)
- [hhh4 spacetime vignette](https://cran.r-project.org/web/packages/surveillance/vignettes/hhh4_spacetime.pdf)
- [CARBayes: an R package for Bayesian spatial modeling](http://eprints.gla.ac.uk/108235/)
- [Bayesian disease mapping: Past, present, and future](https://pmc.ncbi.nlm.nih.gov/articles/PMC8769562/)
- [BEAST X for Bayesian phylogenetic, phylogeographic and phylodynamic inference](https://www.nature.com/articles/s41592-025-02751-x)
- [BEAST 2 phylogeography](https://www.beast2.org/2022/03/01/phylogeography.html)
- [SPREAD visualization](https://beast.community/spread)
- [MASCOT-Skyline for phylogeographic reconstructions](https://pubmed.ncbi.nlm.nih.gov/41004543/)

### Ecology / Biogeography
- [Hmsc: Joint species distribution modelling with the R-package Hmsc](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13345)
- [Hmsc GitHub](https://github.com/hmsc-r/HMSC)
- [HMSC at University of Helsinki](https://www.helsinki.fi/en/researchgroups/statistical-ecology/software/hmsc)
- [bioregion R package](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14496)
- [bioregion CRAN](https://cran.r-project.org/web/packages/bioregion/vignettes/bioregion.html)
- [bioregion arxiv](https://arxiv.org/abs/2404.15300)
- [Metacommunity ecology - Wikipedia](https://en.wikipedia.org/wiki/Metacommunity)
- [Disentangling the drivers of metacommunity structure across spatial scales](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000944/)

### Marketing / Consumer Analytics
- [Geodemographic segmentation - Wikipedia](https://en.wikipedia.org/wiki/Geodemographic_segmentation)
- [Output Area Classification 2021](https://rgs-ibg.onlinelibrary.wiley.com/doi/full/10.1111/geoj.12550)
- [OAC documentation](https://apps.cdrc.ac.uk/static/OAC.pdf)
- [UK OAC Dataset](https://public.cdrc.ac.uk/dataset/uk-output-area-classification-uk-oac)
- [Creating a Geodemographic Classification Using K-means Clustering in R](https://data.geods.ac.uk/dataset/creating-a-geodemographic-classification-using-k-means-clustering-in-r)
- [The Market Basket Transformer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4335141)
- [Claritas PRIZM](https://claritas360.claritas.com/mybestsegments/)

### Recommendation Systems / Collaborative Filtering
- [LightFM GitHub](https://github.com/lyst/lightfm)
- [Implicit library](https://github.com/benfred/implicit)
- [Surprise library](https://surpriselib.com/)
- [Iterative Collaborative Filtering for Sparse Matrix Estimation](https://arxiv.org/abs/1712.00710)
- [Matrix factorization (recommender systems) - Wikipedia](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

### Remote Sensing / Spectral Unmixing
- [HySUPP GitHub](https://github.com/BehnoodRasti/HySUPP)
- [HySUPP paper](https://arxiv.org/abs/2308.09375)
- [unmixing Python library](https://github.com/arthur-e/unmixing)
- [QLSU QGIS plugin](https://www.sciencedirect.com/science/article/abs/pii/S1364815223001688)
- [Graph Regularized NMF](https://arxiv.org/pdf/1111.0885)
- [Nonnegative spatial factorization](https://www.nature.com/articles/s41592-022-01687-w)
- [Spectral unmixing GitHub topic](https://github.com/topics/spectral-unmixing)

### Topic Modeling / Text Analysis
- [Structural Topic Model paper](https://projects.iq.harvard.edu/files/wcfia/files/stmnips2013.pdf)
- [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
- [BERTopic](https://bertopic.com/)
- [BERTopic GitHub](https://github.com/MaartenGr/BERTopic)
- [LDA - Wikipedia](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

### Sensor Networks / Data Fusion
- [PyGSP documentation](https://pygsp.readthedocs.io/en/stable/)
- [PyGSP GitHub](https://github.com/epfl-lts2/pygsp)
- [pgmpy - Belief Propagation](https://pgmpy.org/exact_infer/bp.html)
- [pgmpy GitHub](https://github.com/pgmpy/pgmpy)
- [Gaussian Belief Propagation tutorial](https://gaussianbp.github.io/)
- [GPflow](https://github.com/GPflow/GPflow)
- [Graph Signal Reconstruction for IoT](https://arxiv.org/pdf/2201.00378)
- [Data fusion of sparse sensor devices](https://www.cambridge.org/core/journals/environmental-data-science/article/data-fusion-of-sparse-heterogeneous-and-mobile-sensor-devices-using-adaptive-distance-attention/14F5F461955D8FCBBAF0EEA3D29E45D2)

### Social Physics / Computational Social Science
- [Calibrating an Opinion Dynamics Model to Empirical Opinion Distributions and Transitions](https://www.jasss.org/26/4/9.html)
- [Opinion Dynamics: A Comprehensive Overview](https://arxiv.org/html/2511.00401v1)
- [Dynamic Parameter Calibration Framework for Opinion Dynamics Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC9407186/)
- [Homophily dynamics outweigh network topology in Axelrod model](https://www.sciencedirect.com/science/article/pii/S0378437121003599)
- [Axelrod's Dissemination of Culture](https://journals.sagepub.com/doi/10.1177/0022002797041002001)
- [Region2vec-GAT GitHub](https://github.com/GeoDS/region2vec-GAT)
- [GeoAI-Enhanced Community Detection](https://www.sciencedirect.com/science/article/abs/pii/S0198971524001571)

### Network Community Detection
- [Infomap](https://www.mapequation.org/infomap/)
- [Infomap GitHub](https://github.com/mapequation/infomap)
- [graph-tool](https://graph-tool.skewed.de/)
- [graph-tool inference documentation](https://graph-tool.skewed.de/static/doc/demos/inference/inference.html)
- [Stochastic block model - Wikipedia](https://en.wikipedia.org/wiki/Stochastic_block_model)
- [scikit-network](https://link.springer.com/article/10.1007/s41109-019-0165-9)
- [Network community detection via neural embeddings](https://www.nature.com/articles/s41467-024-52355-w)

### Spatial Methods
- [PySAL spreg](https://pysal.org/spreg/)
- [Spatial Regression in Python tutorial](https://sustainability-gis.readthedocs.io/en/2022/lessons/L4/spatial_regression.html)
- [MRP with structured priors](https://pmc.ncbi.nlm.nih.gov/articles/PMC9203002/)
- [MRP case studies](https://bookdown.org/jl5522/MRP-case-studies/)
