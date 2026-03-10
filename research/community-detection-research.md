# Community Detection & Geodemographic Clustering Research

## Research Summary -- March 2026

**Problem:** Cluster ~3,100 U.S. counties into ~5,000 communities (some counties contain multiple sub-county communities) using non-political data: religion, class/occupation, neighborhood characteristics, migration flows, commuting patterns, and social network connections. Communities should be hierarchical (5,000 --> ~500 --> ~50), allowed to cross state/county borders, somewhat geographically coherent, and potentially overlapping.

---

## 1. Multi-Layer / Multiplex Network Community Detection

### 1.1 The Core Idea

Your data sources (social network ties, commuting flows, migration, feature similarity) each define a different network on the same set of nodes (counties or sub-county units). Rather than collapsing them into a single weighted graph (losing information about which layer contributed which edge), multiplex community detection algorithms operate on all layers simultaneously, finding communities that are coherent across multiple relationship types.

### 1.2 The Leiden Algorithm (Recommended Starting Point)

The Leiden algorithm (Traag, Waltman, van Eck 2019) is the successor to Louvain. It guarantees well-connected communities (Louvain can produce arbitrarily badly connected ones) and converges to a partition where all subsets of all communities are locally optimally assigned.

**Critical feature for your problem: native multiplex support.** The `leidenalg` package directly supports multiplex partition optimization. You create separate partition objects for each layer (e.g., one for commuting, one for SCI, one for feature similarity), each with its own resolution parameter, plus an "interslice" partition that couples the layers. The optimizer finds a single community assignment that balances all layers.

```python
import leidenalg as la
import igraph as ig

# Create graphs for each layer (same node set, different edges)
G_commuting = ig.Graph.Read(...)  # commuting flows
G_social = ig.Graph.Read(...)     # SCI connections
G_similarity = ig.Graph.Read(...) # feature similarity

# Convert to multiplex layers with inter-layer coupling
layers, interslice_layer, G_full = la.time_slices_to_layers(
    [G_commuting, G_social, G_similarity],
    interslice_weight=0.1  # coupling strength between layers
)

# Create partitions with different resolution parameters per layer
partitions = [
    la.CPMVertexPartition(H, node_sizes='node_size', weights='weight',
                          resolution_parameter=gamma)
    for H, gamma in zip(layers, [0.01, 0.05, 0.03])
]
interslice_partition = la.CPMVertexPartition(
    interslice_layer, resolution_parameter=0,
    node_sizes='node_size', weights='weight'
)

optimiser = la.Optimiser()
diff = optimiser.optimise_partition_multiplex(partitions + [interslice_partition])
```

The resolution parameter gamma directly controls granularity: higher gamma = more/smaller communities, lower gamma = fewer/larger communities. By sweeping gamma, you can produce the 50 --> 500 --> 5,000 hierarchy from a single algorithm.

**Package details:**

| Attribute | Value |
|-----------|-------|
| Name | `leidenalg` |
| Language | Python (C++ core) |
| Install | `pip install leidenalg` |
| GitHub | https://github.com/vtraag/leidenalg |
| Docs | https://leidenalg.readthedocs.io/en/stable/multiplex.html |
| Maintained | Yes, actively (V.A. Traag) |
| Relevance | **HIGH** -- native multiplex support, resolution parameter for hierarchy, Python API, scales well |

### 1.3 Infomap (Best for Flow-Based Communities)

Infomap uses information theory (the Map Equation) to find communities by compressing the description of a random walker's movements. It has three features that make it especially relevant:

1. **Native multilayer support:** Handles multiplex networks where modules can span or differ across layers. Can identify overlapping communities across layers that would not be found by analyzing layers separately or by aggregating into a single network.

2. **Hierarchical by default:** Unlike modularity-based methods, Infomap naturally produces hierarchical partitions (modules within modules) without a resolution parameter. The hierarchy depth is determined by the data.

3. **Flow-based:** Random walks on commuting/migration networks have a direct physical interpretation -- they model how people actually move. Infomap finds regions where these flows are contained.

**Package details:**

| Attribute | Value |
|-----------|-------|
| Name | `infomap` |
| Language | Python (C++ core) |
| Install | `pip install infomap` |
| GitHub | https://github.com/mapequation/infomap |
| Docs | https://www.mapequation.org/infomap/ |
| Maintained | Yes, actively (mapequation group, Umea University) |
| Relevance | **HIGH** -- ideal for commuting/migration flow networks, natural hierarchy, multilayer support |

### 1.4 graph-tool: Nested Stochastic Block Models (Most Principled)

The nested stochastic block model (nSBM) in graph-tool is the most statistically principled approach. Rather than optimizing a quality function (modularity, map equation), it performs Bayesian inference: what generative model best explains the observed network?

**Key advantages for your problem:**

1. **Hierarchical by construction:** The nested SBM clusters communities into groups, groups into supergroups, etc., recursively. You get the 5,000 --> 500 --> 50 hierarchy automatically.

2. **No resolution limit:** Standard modularity-based methods (including Louvain/Leiden) have a resolution limit: they cannot detect communities smaller than O(sqrt(N)). For 3,100 counties, that means communities smaller than ~56 counties may be missed. The nested SBM overcomes this by using a hierarchy of priors.

3. **Layered network support:** The `LayeredBlockState` class handles multiplex networks directly. Each edge type (commuting, social, similarity) becomes a layer, and the model infers a single block structure that accounts for all layers. Weighted edges and edge covariates are supported.

4. **Model selection:** You can compare different model variants (degree-corrected vs. not, layered vs. not) using description length, providing a principled way to decide if multiplex analysis actually helps.

```python
import graph_tool.all as gt

# Load graph with edge layers
g = gt.Graph()
# ... add nodes and edges ...
# ec = edge property map with layer labels

# Fit nested SBM with layers
state = gt.minimize_nested_blockmodel_dl(
    g, state_args=dict(ec=ec, layers=True)
)

# Access hierarchy levels
levels = state.get_levels()
# levels[0] = finest partition (your ~5,000 communities)
# levels[1] = next level (~500 blocs)
# levels[2] = coarsest (~50 mega-blocs)
```

**Package details:**

| Attribute | Value |
|-----------|-------|
| Name | `graph-tool` |
| Language | Python (C++/Boost core) |
| Install | conda or from source (no pip) |
| Website | https://graph-tool.skewed.de/ |
| GitLab | https://git.skewed.de/count0/graph-tool |
| Maintained | Yes, actively (Tiago Peixoto, v2.98 current) |
| Relevance | **VERY HIGH** -- most principled approach, native hierarchy, layered networks, no resolution limit, model selection |

**Caveat:** Installation is harder than pip packages (requires conda or compilation). Performance is excellent once installed.

### 1.5 CDlib (Meta-Library)

CDlib wraps 39+ community detection algorithms under a unified API. It includes overlapping methods (BigClam, ANGEL, DEMON, k-clique, Ego-splitting), crisp methods (Leiden, Louvain, Infomap, label propagation), and evaluation metrics (NMI, modularity, conductance).

**Especially useful for:** comparing multiple algorithms on your data, evaluating community quality, and accessing overlapping community detection methods.

| Attribute | Value |
|-----------|-------|
| Name | `cdlib` |
| Language | Python |
| Install | `pip install cdlib` |
| GitHub | https://github.com/GiulioRossetti/cdlib |
| Docs | https://cdlib.readthedocs.io/ |
| Maintained | Yes (Giulio Rossetti, v0.4.0) |
| Relevance | **MEDIUM-HIGH** -- excellent for algorithm comparison and overlapping detection; multiplex support still maturing |

### 1.6 Other Network Libraries

| Package | Language | Install | GitHub | Maintained | Assessment |
|---------|----------|---------|--------|------------|------------|
| `python-igraph` | Python (C core) | `pip install igraph` | https://github.com/igraph/python-igraph | Yes, actively | Foundation for leidenalg. Fast graph operations. Good for building and manipulating the county network. |
| `NetworKit` | Python (C++/OpenMP core) | `pip install networkit` | https://github.com/networkit/networkit | Yes (v11.2.1, Jan 2026) | Fastest for very large graphs (billions of edges). Parallel Louvain, PLM. Overkill for 3,100 nodes but excellent if you go to sub-county resolution. |
| `networkx` | Python | `pip install networkx` | https://github.com/networkx/networkx | Yes, actively | Slow but feature-complete. Good for prototyping and visualization. Not for production community detection. |
| `louvain-igraph` | Python | `pip install louvain` | https://github.com/vtraag/louvain-igraph | Superseded by leidenalg | Legacy. Use leidenalg instead. |

### 1.7 Multiplex-Specific Research Tools

| Package | Language | GitHub | Assessment |
|---------|----------|--------|------------|
| `multiplexcd` | Python | https://github.com/michaelsiemon/multiplexcd | Small research package for multiplex community detection. Not actively maintained. Use leidenalg or graph-tool instead. |
| `multinet` (R) | R | CRAN | R package for multilayer network analysis. Good if you prefer R. |

---

## 2. Hierarchical Community Detection

### 2.1 Getting 50 --> 500 --> 5,000 from the Same Algorithm

There are three main strategies:

**Strategy A: Resolution Parameter Sweep (Leiden/Louvain)**
- Run Leiden with the Constant Potts Model (CPM) at multiple resolution values.
- Low resolution (gamma=0.001) --> ~50 mega-blocs. Medium resolution (gamma=0.01) --> ~500 blocs. High resolution (gamma=0.1) --> ~5,000 communities.
- Advantage: simple, fast, can be done independently at each level.
- Disadvantage: the levels are not guaranteed to be hierarchically consistent (a community at the fine level may span two blocs at the medium level).
- **Fix:** Use consensus clustering across resolutions (see below).

**Strategy B: Nested Stochastic Block Models (graph-tool)**
- The hierarchy is intrinsic to the model. The finest level gives communities; coarser levels are automatically generated.
- The number of levels and communities per level are inferred from the data (no free parameters).
- Guarantee: the hierarchy is consistent by construction.
- This is the most principled approach.

**Strategy C: Hierarchical Infomap**
- Infomap naturally produces nested modules. You can specify `--two-level` for flat, or let it find the optimal hierarchy depth.
- The hierarchy corresponds to different scales of flow containment.

### 2.2 Consensus Clustering for Robust Multi-Scale Structure

A powerful approach (Lancichinetti & Fortunato 2012; Jeub et al. 2018) addresses the instability of single-resolution methods:

1. Run Leiden at many resolution values (e.g., 100 values from 0.001 to 1.0).
2. Build a "co-classification matrix" C where C_ij = fraction of runs where nodes i and j were in the same community.
3. Cluster C itself (using Leiden or spectral methods) to find robust communities.

This gives you communities that are stable across resolutions -- they represent genuine multi-scale structure rather than artifacts of a particular resolution setting.

**Paper:** "Multiresolution Consensus Clustering in Networks" (Jeub et al. 2018), Scientific Reports. https://www.nature.com/articles/s41598-018-21352-7

**Key innovation:** An "event sampling" strategy for the resolution parameter that directly exploits modularity's behavior to provide good coverage of different scales.

### 2.3 Dendrogram-Based Methods

Hierarchical agglomerative clustering (HAC) on the network produces a dendrogram that can be cut at any level:

- **Used by Bailey et al. (SCI paper):** They used HAC with the SCI as the similarity measure to cluster US counties into social communities. Cutting at different heights gives different numbers of communities.
- **Advantage:** Simple, deterministic, produces a full hierarchy.
- **Disadvantage:** Greedy -- once two communities merge, they never split. Can produce poor results for networks with multi-scale structure.

---

## 3. Spatial Clustering with Contiguity Constraints

### 3.1 PySAL / spopt (Recommended)

The `spopt` package (part of PySAL) provides regionalization algorithms that produce spatially contiguous clusters. These are directly relevant for ensuring geographic coherence.

**Algorithms available:**

| Algorithm | What It Does | When to Use |
|-----------|-------------|-------------|
| **Max-p** | Maximizes the number of regions subject to a minimum threshold (e.g., minimum population per region). Ensures contiguity. | When you want to maximize the number of communities while ensuring each has sufficient population for statistical analysis. |
| **SKATER** | Builds a minimum spanning tree on the spatial weights graph, then prunes edges to create regions that maximize internal homogeneity. | When you want regions that are both spatially contiguous and internally homogeneous on feature variables. |
| **AZP** (Automatic Zoning Procedure) | Iteratively reassigns spatial units between regions to optimize an objective function while maintaining contiguity. | When you want a specific number of regions with maximum internal homogeneity. |
| **WardSpatial** | Agglomerative clustering with a contiguity constraint. | When you want hierarchical clustering that respects spatial adjacency. |

**Important nuance for your problem:** You said communities should be "somewhat geographically coherent but not rigidly bounded." Pure regionalization algorithms enforce strict contiguity. You can relax this by:

1. **Weighted hybrid approach:** Run both a network-based method (Leiden on SCI/commuting) and a spatial method (SKATER on demographics). Use consensus clustering to combine them -- communities that appear in both will be geographically coherent, while network-only communities can cross borders.

2. **Soft contiguity via spatial weights:** In Leiden, add a spatial proximity layer (edges weighted by geographic adjacency) as one of several multiplex layers. This *encourages* but does not *require* geographic coherence.

3. **Post-hoc splitting:** Run Leiden without spatial constraints. For communities that are geographically disconnected, split them into connected components and check if the pieces are substantively different.

**Package details:**

| Attribute | Value |
|-----------|-------|
| Name | `spopt` (part of PySAL) |
| Language | Python |
| Install | `pip install spopt` |
| GitHub | https://github.com/pysal/spopt |
| Docs | https://pysal.org/spopt/ |
| Maintained | Yes (PySAL ecosystem, v0.7.0) |
| Relevance | **HIGH** for geographic coherence constraint |

**Related PySAL packages:**

| Package | Role | GitHub |
|---------|------|--------|
| `libpysal` | Spatial weights matrices (Queen/Rook contiguity, distance-based) | https://github.com/pysal/libpysal |
| `esda` | Exploratory spatial data analysis (Moran's I, LISA) | https://github.com/pysal/esda |
| `spreg` | Spatial regression | https://github.com/pysal/spreg |

### 3.2 scikit-learn Clustering with Spatial Constraints

scikit-learn's `AgglomerativeClustering` accepts a `connectivity` parameter -- a sparse matrix defining which pairs of samples can be merged. You can pass a spatial contiguity matrix to enforce geographic coherence.

```python
from sklearn.cluster import AgglomerativeClustering
from libpysal.weights import Queen

# Build contiguity matrix from county shapefile
w = Queen.from_dataframe(counties_gdf)
connectivity = w.sparse

# Cluster with contiguity constraint
model = AgglomerativeClustering(
    n_clusters=500,
    connectivity=connectivity,
    linkage='ward'
)
labels = model.fit_predict(feature_matrix)
```

For soft (non-strict) contiguity, use `SpectralClustering` with a precomputed affinity matrix that combines feature similarity and spatial proximity.

### 3.3 HDBSCAN with Soft Clustering (For Overlapping Communities)

HDBSCAN provides membership probability vectors: each data point gets a probability of belonging to each cluster. This directly addresses the "a county can belong to multiple communities" requirement.

```python
from hdbscan import HDBSCAN

clusterer = HDBSCAN(min_cluster_size=10, prediction_data=True)
clusterer.fit(feature_matrix)

# Soft membership probabilities
from hdbscan import all_points_membership_vectors
soft_clusters = all_points_membership_vectors(clusterer)
# soft_clusters[i] = probability vector over all clusters for county i
```

| Attribute | Value |
|-----------|-------|
| Name | `hdbscan` |
| Language | Python |
| Install | `pip install hdbscan` (also in scikit-learn 1.3+) |
| GitHub | https://github.com/scikit-learn-contrib/hdbscan |
| Maintained | Yes (also integrated into scikit-learn) |
| Relevance | **MEDIUM-HIGH** -- overlapping community detection via soft clustering; no spatial contiguity constraint built in |

---

## 4. Geodemographic Segmentation Methods

### 4.1 Commercial Systems: How They Actually Work

**Claritas PRIZM Premier (68 segments)**

Methodology: PRIZM abandoned traditional k-means clustering in favor of a proprietary method called **Multivariate Divisive Partitioning (MDP)**, which uses Classification and Regression Trees (CART). The process:

1. Start with all households in one segment.
2. Identify the variable and split point that maximally separates a target behavior.
3. Split the segment. Repeat recursively until segments fall below a size threshold.
4. The 68 segments are organized into 11 Lifestage Groups and 14 Social Groups (hierarchical).
5. Variables include income, education, occupation, home value, plus 10,000+ consumer behavior variables.
6. Available at ZIP+4, block group, ZIP code, and household levels.

**Key insight:** PRIZM is not a pure clustering method -- it is a supervised tree-based approach where the splits are guided by behavioral outcomes (purchasing, media consumption). This makes it very different from unsupervised geodemographic classification.

**Esri Tapestry (67 segments)**

Methodology: Uses **k-means clustering with median centroids** (rather than mean centroids), which makes it more robust to outliers. The distance metric is least absolute deviation rather than Euclidean distance.

1. Data sources: Census Bureau, ACS, and consumer surveys (MRI-Simmons).
2. Multiple clustering rounds with different initialization strategies.
3. 67 segments organized into 14 LifeMode groups and 6 Urbanization groups.
4. Available at block group and tract levels via ArcGIS.

**Experian Mosaic USA (~71 types)**

Methodology: Classic cluster analysis using 600+ variables from Experian's consumer data. Uses a combination of principal components analysis for dimensionality reduction followed by k-means or similar clustering. Organized into groups and types hierarchically.

**Shared limitations of commercial systems:**
- Proprietary methods -- not reproducible
- Expensive ($10,000+/year for full data)
- Fixed segment definitions -- cannot customize for your use case
- Optimized for consumer marketing, not political community structure
- Segment boundaries may not correspond to politically meaningful divisions

### 4.2 Open-Source Alternatives

**OAC (Output Area Classification) -- UK**

The gold standard for open geodemographic classification. Created by the UK Office for National Statistics and UCL researchers (Alex Singleton, Paul Longley).

- **2021 OAC:** 8 supergroups, 21 groups, 52 subgroups (hierarchical)
- **Method:** K-means clustering on standardized census variables, applied iteratively (first cluster into supergroups, then subdivide)
- **All data, code, and methods are published and reproducible**
- **2011 OAC code and tutorial:** https://geogale.github.io/2011OAC/

| Attribute | Value |
|-----------|-------|
| Name | Output Area Classification (OAC) |
| Country | UK |
| Data | https://data.geods.ac.uk/dataset/output-area-classification-2021 |
| Method paper | https://discovery.ucl.ac.uk/id/eprint/1498873/ |
| Relevance | **HIGH** as a methodological template. Directly transferable approach for US census tracts/block groups. |

**North American Geodemographic Classification (Spielman & Singleton)**

The first open geodemographic classification for the United States:

- Built from American Community Survey (ACS) data at the census tract level
- All methods, data inputs, and software are open source
- Organized around a conceptual framework of Concepts, Domains, and Measures
- Developed collaboratively between Seth Spielman (U. Colorado) and Alex Singleton (U. Liverpool)

| Attribute | Value |
|-----------|-------|
| Name | North American Geodemographics |
| Institution | Geographic Data Science Lab, University of Liverpool |
| Website | https://www.liverpool.ac.uk/geographic-data-science/research/understandingthemorphologyofcities/north-american-geodemographics/ |
| Relevance | **VERY HIGH** -- directly applicable to US census tracts, open source, academic |

**Python Tutorial: Building Geodemographic Classifications from Scratch**

A complete Jupyter notebook workflow from the Geographic Data Service:

1. Access Census/ACS data
2. Select and standardize variables
3. Correlation and variance analysis
4. Determine optimal k using clustergrams
5. Apply k-means hierarchically (supergroups first, then subdivide)
6. Visualize with Kepler.gl

| Attribute | Value |
|-----------|-------|
| Tutorial | https://geographicdataservice.github.io/geodem-python-training/creatinggeodem.html |
| Notebook | https://github.com/GeographicDataService/geodem-python-training |
| Relevance | **HIGH** -- ready-to-use Python code for building exactly the type of classification you need |

**GeodemCreator**

Cross-platform Java tool for building geodemographic classifications. Requires Java and R. Designed for non-experts but limited in flexibility.

| Attribute | Value |
|-----------|-------|
| Language | Java + R |
| Relevance | **LOW** -- use the Python tutorial above instead |

### 4.3 Recent Innovation: LLM-Based Geodemographic Naming

A 2024 paper in EPJ Data Science ("Segmentation using large language models: A new typology of American neighborhoods") uses LLMs to generate intuitive descriptions and names for high-dimensional clusters. The pipeline:

1. Standard k-means clustering on ACS block-group data
2. Feed cluster centroids to an LLM to generate descriptive names
3. Open-source and reproducible

This addresses a major usability issue: giving your 5,000 communities meaningful names automatically.

**Paper:** https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00466-1

---

## 5. Facebook Social Connectedness Index (SCI)

### 5.1 What It Is

The SCI measures the relative probability that two individuals across two locations are Facebook friends. For every pair of US counties (3,136 x 3,136 = ~9.8 million pairs), Meta provides a score proportional to:

```
SCI_{ij} = (FB_friends_{ij}) / (FB_users_i * FB_users_j)
```

This is then scaled to a 0--1,000,000,000 range and noise is added for privacy.

### 5.2 Resolution and Access

| Level | Coverage | Access |
|-------|----------|--------|
| US county to US county | All 3,136 counties | Public download |
| US ZIP to US ZIP | ~32,000 ZIPs | Public download |
| US county to foreign country | All countries | Public download |
| NUTS2/NUTS3 (Europe) | EU regions | Public download |

**Download:** https://data.humdata.org/dataset/social-connectedness-index

**Example scripts:** https://github.com/social-connectedness-index/example-scripts

**Methodology documentation:** https://dataforgood.facebook.com/dfg/docs/methodology-social-connectedness-index

### 5.3 Key Research Using SCI

**Bailey, Cao, Kuchler, Stroebel, Wong (2018). "Social Connectedness: Measurement, Determinants, and Effects." Journal of Economic Perspectives.**

- Foundational paper. Used hierarchical agglomerative clustering on the SCI to identify "social communities" of US counties.
- Found that SCI-based communities align with state borders but also reveal cross-state groupings (e.g., all West Coast states + Nevada form one community; Florida + Georgia + Alabama form another).
- Higher SCI between county pairs predicts: more trade, more migration, more patent citations, more correlated COVID spread.
- Paper: https://www.aeaweb.org/articles?id=10.1257/jep.32.3.259
- NBER working paper: https://www.nber.org/papers/w23608

**Johnston, Kuchler, Koenig, Stroebel (2024). "The Social Connectedness Index."**

- Updated and expanded SCI paper with more applications.
- Paper: https://pages.stern.nyu.edu/~jstroebe/PDF/JKKS_SCI.pdf

**Political applications:**
- Counties more socially connected on Facebook tend to vote more similarly, even controlling for geographic distance and demographics (Baruch College / Zicklin research).
- SCI predicts COVID spread patterns between counties (PMC 8675563).

### 5.4 Using SCI for Community Detection

**Direct approach:** Treat the county-to-county SCI matrix as a weighted adjacency matrix. Run Leiden/Infomap/nSBM on it. The communities are "social regions" -- groups of counties whose residents are disproportionately connected to each other.

**As one layer in a multiplex:** The SCI is your strongest single layer for social community detection. Combine it with commuting, migration, and feature similarity layers.

**Resolution control:** By thresholding the SCI matrix (only keep edges above a cutoff) or using different resolution parameters, you can control the number/size of communities.

### 5.5 Bias and Limitations

| Bias | Severity | Details |
|------|----------|---------|
| **Age bias** | Moderate | Facebook skews younger than the general population (especially for older rural populations). Older adults' social connections are underrepresented. |
| **Rural coverage** | Moderate-High | Counties and ZIP codes with few Facebook users are excluded or have noisy SCI values. Rural areas tend to have lower Facebook penetration. |
| **Urban-rural distance decay** | Moderate | The SCI captures different things in urban vs. rural areas. In rural counties, SCI may partly measure propensity to travel rather than genuine social ties. Distance decay is weaker in more urbanized counties. |
| **Privacy noise** | Low | Random noise is added and small-count locations are excluded. This reduces accuracy for sparsely populated areas. |
| **Platform-specific** | Moderate | Captures only Facebook friendships, not other social ties (family not on Facebook, work relationships, church communities, etc.). |
| **Temporal snapshot** | Low | SCI represents a point-in-time snapshot; social networks evolve. |

**Key paper on limitations:** The original Bailey et al. (2018) paper acknowledges these limitations. The SCI correlates at r=0.5--0.8 with ground-truth measures (e.g., IRS migration data), suggesting it captures real social structure but with substantial noise.

**Mitigation:** Use SCI as one of several data sources, not the only one. Commuting data (Census LODES) and migration data (ACS/IRS) provide complementary coverage that does not have the Facebook-specific biases.

---

## 6. Relevant Methods From Other Fields

### 6.1 Commuting Zone Delineation (Labor Economics)

The US Department of Agriculture Economic Research Service has defined **Commuting Zones (CZs)** since 1987 using exactly the approach you are pursuing: community detection on a county-level flow network.

**Current methodology (2020 CZs):**
- Build a county-to-county commuting flow matrix from Census LODES data
- Apply hierarchical cluster analysis to group counties into contiguous labor markets
- The 2020 delineation produces **598 commuting zones** from 3,222 counties
- Contiguity is enforced as a hard constraint

**Recent improvement using Louvain:**
Whitney Zhang (2022) showed that replacing hierarchical clustering with the **Louvain community detection algorithm** produces better commuting zones. The "TS Louvain" and "Sum Louvain" delineations better capture actual commuting flows and improve statistical precision in downstream economic analyses.

- Paper: https://www.sciencedirect.com/science/article/abs/pii/S0165176522003093
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3885325

**A separate 2020 paper (PMC 7190107)** addresses a technical challenge: counties with large self-loops (most workers commute within the county). Standard community detection algorithms ignore self-loops. The paper develops methods for demarcating geographic regions using community detection in commuting networks with significant self-loops.

**Direct relevance:** CZ delineation is the closest published precedent to your problem. The main differences are: (1) CZs use only commuting data; you want multiple data layers. (2) CZs produce ~600 zones; you want ~5,000 communities. (3) CZs enforce strict contiguity; you want soft contiguity.

| Resource | URL |
|----------|-----|
| USDA CZ data | https://www.ers.usda.gov/data-products/commuting-zones-and-labor-market-areas |
| 2020 CZ paper | https://www.nature.com/articles/s41597-024-03829-5 |
| Penn State CZ project | https://sites.psu.edu/psucz/data/ |

### 6.2 Bioregionalization (Ecology)

Ecologists solve an analogous problem: delineate biogeographic regions (groups of locations with similar species) from species occurrence data. This is structurally identical to "delineate communities from county-level demographic/cultural data."

**Methods used:**
- Build a bipartite network (locations <--> species; for you: counties <--> demographic features)
- Apply community detection (Infomap is popular in ecology) to find bioregions
- The network approach objectively identifies and quantifies regions without relying on arbitrary similarity metrics

**Key paper:** Vilhena & Antonelli (2015). "A network approach for identifying and delimiting biogeographical regions." Nature Communications. https://www.nature.com/articles/ncomms7848

**The bioregion R package** (2024) implements the full bioregionalization workflow:
- Dissimilarity/similarity computation
- Network construction from bipartite data
- Community detection (Infomap, OSLOM, Louvain)
- Cluster comparison and evaluation

| Attribute | Value |
|-----------|-------|
| Name | `bioregion` |
| Language | R |
| Paper | https://arxiv.org/abs/2404.15300 |
| Relevance | **MEDIUM** -- methodological template for network-based regionalization; R-only |

### 6.3 Region2Vec / GeoAI Community Detection (Geography)

A family of deep-learning methods specifically designed for community detection on spatial networks. Developed by the GeoDS Lab (University of Wisconsin-Madison).

**Region2Vec approach:**
1. Build a spatial network where nodes are geographic units (counties, tracts) with attribute vectors (demographics, economics)
2. Generate node embeddings using Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT) that encode attribute similarity, geographic adjacency, AND spatial interactions (commuting/migration flows)
3. Apply agglomerative clustering on the embeddings

**Why this matters for you:** Region2Vec is the only published method that jointly optimizes for (a) node attribute similarity, (b) geographic adjacency, and (c) spatial interaction intensity -- exactly the three criteria for your communities.

| Attribute | Value |
|-----------|-------|
| Name | Region2Vec (GCN version) |
| Language | Python |
| GitHub | https://github.com/GeoDS/Region2vec |
| Paper | https://arxiv.org/abs/2210.08041 |

| Attribute | Value |
|-----------|-------|
| Name | Region2Vec-GAT (improved version) |
| Language | Python |
| GitHub | https://github.com/GeoDS/region2vec-GAT |
| Paper | https://arxiv.org/abs/2411.15428 (2024) |
| Relevance | **HIGH** -- directly addresses multi-criteria spatial community detection |

### 6.4 Epidemiology: Disease Mapping and Spatial Clustering

Epidemiological disease mapping uses spatial clustering methods at the county level that are directly transferable:

- **SaTScan** (spatial scan statistic): Finds clusters of unusually high/low rates. Not community detection per se, but the spatial scanning approach could identify locally distinctive communities.
- **BYM/BYM2 model** (Besag-York-Mollie): Bayesian spatial model that decomposes variation into structured (spatially smooth) and unstructured components. Already referenced in the methods document for the covariance model.
- **SpatialEpi** (R package): Implements disease cluster detection methods including Kulldorff's scan statistic.

| Attribute | Value |
|-----------|-------|
| Name | `SpatialEpi` |
| Language | R |
| CRAN | https://cran.r-project.org/web/packages/SpatialEpi/ |
| Relevance | **LOW-MEDIUM** -- disease clustering methods can identify spatial anomalies but are not designed for comprehensive community detection |

### 6.5 Marketing: Geo-Marketing Segmentation

The marketing field has developed geo-marketing segmentation using deep learning on geographic data. A 2021 paper (MDPI) applied deep learning to geographic customer segmentation, combining spatial features with behavioral data.

**CARTO** provides a commercial platform for geographic segmentation using spatial clustering, but the underlying methods are standard (k-means, DBSCAN with geographic features).

**Assessment:** Marketing methods are either proprietary (PRIZM, Tapestry) or use standard clustering techniques. The open-source tools from spatial science (PySAL, Region2Vec) are more directly applicable.

---

## 7. Data Sources for Building the Multi-Layer Network

For completeness, here are the specific datasets that feed into each network layer:

| Layer | Data Source | Resolution | Access |
|-------|------------|------------|--------|
| **Social network** | Facebook SCI | County-county, ZIP-ZIP | Public: https://data.humdata.org/dataset/social-connectedness-index |
| **Commuting** | Census LODES/LEHD | County-county (block-level available) | Public: https://lehd.ces.census.gov/ |
| **Migration** | ACS County-to-County Flows | County-county | Public: https://www.census.gov/topics/population/migration/guidance/county-to-county-migration-flows.html |
| **Migration (tax)** | IRS SOI Migration Data | County-county | Public: https://www.irs.gov/statistics/soi-tax-stats-migration-data |
| **Religion** | ARDA / RCMS 2020 | County | Public: https://www.thearda.com/ |
| **Demographics** | ACS 5-year | Tract/block group | Public: Census API |
| **Occupation** | ACS / BLS QCEW | County/tract | Public |
| **Neighborhood** | ACS housing, density, urbanization | Tract/block group | Public |

---

## 8. Recommended Approach

### 8.1 Pipeline Overview

```
Phase 1: Build Multi-Layer County Network
  |-- Layer 1: SCI (social connections)
  |-- Layer 2: Commuting flows (Census LODES)
  |-- Layer 3: Migration flows (ACS + IRS)
  |-- Layer 4: Feature similarity (demographics, religion, occupation)
  |-- Layer 5: Geographic adjacency (Queen contiguity from libpysal)

Phase 2: Community Detection
  |-- Option A: Leiden multiplex (leidenalg) with resolution sweep
  |-- Option B: Nested SBM with layers (graph-tool)
  |-- Option C: Region2Vec-GAT (deep learning embeddings + clustering)
  |-- Evaluate all three; use consensus clustering to combine

Phase 3: Hierarchical Structuring
  |-- Fine level: ~5,000 communities (high resolution / bottom nSBM level)
  |-- Medium level: ~500 blocs (medium resolution / middle nSBM level)
  |-- Coarse level: ~50 mega-blocs (low resolution / top nSBM level)
  |-- Ensure hierarchical consistency (every community maps to exactly one bloc)

Phase 4: Sub-County Splitting
  |-- For large/heterogeneous counties, use tract-level data
  |-- Run geodemographic classification (k-means on ACS) within counties
  |-- Identify counties with multiple distinct communities
  |-- Split into sub-county units before re-running Phase 2

Phase 5: Overlapping Communities
  |-- HDBSCAN soft clustering for membership probabilities
  |-- Or: CDlib overlapping algorithms (BigClam, ANGEL)
  |-- Assign each county a probability vector over communities

Phase 6: Validation
  |-- Compare with known functional regions (CZs, MSAs, media markets)
  |-- Test community homogeneity on held-out variables
  |-- Verify geographic coherence (most communities should be ~contiguous)
```

### 8.2 Tool Recommendations by Priority

| Priority | Tool | Role | Why |
|----------|------|------|-----|
| 1 | `leidenalg` | Primary community detection | Native multiplex, resolution control, fast, well-maintained |
| 2 | `graph-tool` (nSBM) | Principled hierarchical detection | No resolution limit, automatic hierarchy, layered networks, model selection |
| 3 | `libpysal` + `spopt` | Spatial weights and regionalization | Queen contiguity, spatial constraints, integration with Python spatial ecosystem |
| 4 | `python-igraph` | Graph construction and manipulation | Foundation for leidenalg, fast I/O |
| 5 | `hdbscan` | Overlapping/soft community detection | Membership probabilities for multi-community counties |
| 6 | `cdlib` | Algorithm comparison and evaluation | Wraps many methods, provides community quality metrics |
| 7 | `infomap` | Flow-based community detection | Best for commuting/migration layers specifically |
| 8 | Region2Vec-GAT | Deep learning spatial communities | Joint optimization of attributes + adjacency + flows |

### 8.3 Key Methodological Decisions

1. **Unit of analysis:** Start with counties (3,100). For counties with population > X or intra-county heterogeneity above a threshold, split into census tracts first. This gets you closer to ~5,000 units before community detection.

2. **Edge weight construction:** For each layer, normalize edge weights so that layers contribute roughly equally. SCI values range 0 to 10^9; commuting flows range 0 to millions. Use rank-based normalization or log-transform + standardize.

3. **Layer coupling strength:** The `interslice_weight` parameter in leidenalg (or implicit layer coupling in nSBM) controls how much layers agree. Start with weak coupling (0.01--0.1) and increase until communities stabilize.

4. **Resolution calibration:** Calibrate the resolution parameter to produce the target number of communities by binary search or by using the relationship between gamma and community count.

5. **Contiguity as a soft constraint:** Add geographic adjacency as a low-weight layer rather than a hard constraint. This allows cross-border communities (e.g., the Kansas City metro spanning Kansas and Missouri) while encouraging coherence.

---

## 9. Complete Tool/Package Reference

| Package | Language | GitHub/URL | Maintained | Primary Use |
|---------|----------|------------|------------|-------------|
| `leidenalg` | Python | https://github.com/vtraag/leidenalg | Yes | Multiplex community detection with resolution control |
| `graph-tool` | Python | https://git.skewed.de/count0/graph-tool | Yes | Nested SBM, layered networks, principled inference |
| `infomap` | Python | https://github.com/mapequation/infomap | Yes | Flow-based hierarchical community detection |
| `cdlib` | Python | https://github.com/GiulioRossetti/cdlib | Yes | Meta-library: 39+ algorithms, overlapping detection |
| `python-igraph` | Python | https://github.com/igraph/python-igraph | Yes | Graph construction, manipulation, I/O |
| `networkit` | Python | https://github.com/networkit/networkit | Yes | Large-scale parallel community detection |
| `networkx` | Python | https://github.com/networkx/networkx | Yes | Prototyping, visualization (not for production) |
| `hdbscan` | Python | https://github.com/scikit-learn-contrib/hdbscan | Yes | Soft/overlapping clustering with membership probabilities |
| `spopt` | Python | https://github.com/pysal/spopt | Yes | Regionalization (Max-p, SKATER, AZP) |
| `libpysal` | Python | https://github.com/pysal/libpysal | Yes | Spatial weights matrices |
| `esda` | Python | https://github.com/pysal/esda | Yes | Spatial autocorrelation, LISA |
| Region2Vec | Python | https://github.com/GeoDS/Region2vec | Research | GCN-based spatial community detection |
| Region2Vec-GAT | Python | https://github.com/GeoDS/region2vec-GAT | Research | GAT-based spatial community detection |
| Geodem Training | Python | https://github.com/GeographicDataService/geodem-python-training | Tutorial | Open geodemographic classification from scratch |
| `bioregion` | R | CRAN / https://arxiv.org/abs/2404.15300 | Yes | Bioregionalization (transferable methods) |
| `SpatialEpi` | R | CRAN | Yes | Spatial cluster detection (epidemiology) |
| `multiplexcd` | Python | https://github.com/michaelsiemon/multiplexcd | No | Multiplex community detection (use leidenalg instead) |

---

## 10. Key References

### Community Detection Methods
- Traag, Waltman, van Eck (2019). "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports. https://www.nature.com/articles/s41598-019-41695-z
- Peixoto (2014). "Hierarchical block structures and high-resolution model selection in large networks." Physical Review X.
- Rosvall & Bergstrom (2008). "Maps of random walks on complex networks reveal community structure." PNAS.
- Jeub et al. (2018). "Multiresolution Consensus Clustering in Networks." Scientific Reports. https://www.nature.com/articles/s41598-018-21352-7

### Multiplex / Multi-Layer Networks
- Mucha et al. (2010). "Community Structure in Time-Dependent, Multiplex, and Other Multirelational Networks." Science.
- Peixoto (2015). "Inferring the mesoscale structure of layered, edge-valued, and time-varying networks." Physical Review E.
- De Domenico et al. (2015). "Identifying modular flows on multilayer networks reveals highly overlapping organization in interconnected systems." Physical Review X.

### Spatial Community Detection
- Gao et al. (2024). "GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning." Computers, Environment and Urban Systems.
- Gao et al. (2022). "Region2Vec: Community Detection on Spatial Networks Using Graph Embedding." GIScience.

### Commuting Zones
- Zhang (2022). "Improving Commuting Zones Using the Louvain Community Detection Algorithm." Economics Letters. https://www.sciencedirect.com/science/article/abs/pii/S0165176522003093
- Nelson & Rae (2016). "An Economic Geography of the United States: From Commutes to Megaregions." PLoS ONE.

### Facebook SCI
- Bailey, Cao, Kuchler, Stroebel, Wong (2018). "Social Connectedness: Measurement, Determinants, and Effects." JEP. https://www.aeaweb.org/articles?id=10.1257/jep.32.3.259
- Johnston, Kuchler, Koenig, Stroebel (2024). "The Social Connectedness Index." https://pages.stern.nyu.edu/~jstroebe/PDF/JKKS_SCI.pdf

### Geodemographic Classification
- Singleton & Longley (2015). "Creating the 2011 Area Classification for Output Areas." JOSIS. https://discovery.ucl.ac.uk/id/eprint/1498873/
- Spielman & Singleton. "North American Geodemographics." https://www.liverpool.ac.uk/geographic-data-science/research/understandingthemorphologyofcities/north-american-geodemographics/
- Esri Tapestry Methodology: https://downloads.esri.com/esri_content_doc/dbl/us/J9941_Tapestry_Segmentation_Methodology_2018.pdf

### Bioregionalization
- Vilhena & Antonelli (2015). "A network approach for identifying and delimiting biogeographical regions." Nature Communications. https://www.nature.com/articles/ncomms7848
