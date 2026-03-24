# HDBSCAN Clustering Experiment (P3.4)

**Date:** 2026-03-24
**Session:** S178
**Branch:** feat/hdbscan-experiment

## Objective

Test whether HDBSCAN (density-based hierarchical clustering) can find
non-convex or variably-dense electoral type clusters that KMeans misses,
and compare holdout r on the county-prior prediction task.

## Setup

- **Data:** 293 counties (FL+GA+AL), 39 training dims (2008+, presidential×2.5)
- **Holdout:** 2020→2024 presidential shifts (pres_d, pres_r, pres_turnout)
- **KMeans baseline:** J=43, n_init=10, random_state=42, T=10 soft membership
- **HDBSCAN:** metric=euclidean, cluster_selection_method=eom (Excess of Mass default)
- **min_cluster_size sweep:** [5, 10, 15, 20, 30]
- **Soft membership:** T=10 inverse-distance to cluster centroids (same as KMeans production)
- **Noise points (-1):** assigned to nearest centroid for soft membership computation
- **Holdout metric:** mean of per-column Pearson r across 3 holdout dimensions (same as spectral experiment)
- **Production baseline:** KMeans J=43, holdout r=0.8428 (direct 2020→2024 holdout)

## Results

### Holdout r — county-prior method

| min_cluster_size | n_clusters | n_noise | noise_pct | holdout_r | delta vs KMeans |
|-----------------|------------|---------|-----------|-----------|-----------------|
| 5               | 3          | 1       | 0.3%      | 0.5295    | -0.3133         |
| 10              | 3          | 1       | 0.3%      | 0.5295    | -0.3133         |
| 15              | 3          | 5       | 1.7%      | 0.5295    | -0.3133         |
| 20              | 3          | 7       | 2.4%      | 0.5294    | -0.3134         |
| 30              | 2          | 5       | 1.7%      | 0.5087    | -0.3341         |

**KMeans J=43 baseline:** holdout r = 0.8428
**Best HDBSCAN result:** min_cluster_size=5, holdout r = 0.5295 (delta = -0.3133)

### Key Structural Finding

HDBSCAN consistently finds exactly 3 clusters across all min_cluster_size settings:

- **Cluster 0:** 159 counties — 100% Georgia (FIPS prefix 13)
- **Cluster 1:** 67 counties — 100% Alabama (FIPS prefix 01)
- **Cluster 2:** 66 counties — 100% Florida (FIPS prefix 12)

HDBSCAN is detecting the state-level density structure of the shift space, not
electoral sub-types. The three states form three well-separated density blobs in
the 39-dimensional shift space, and HDBSCAN treats each state as one cluster.
This is structurally correct from a density perspective (three states, three blobs),
but useless for the electoral type discovery task — which requires finding
fine-grained types that **cross** state boundaries.

## Conclusion

**HDBSCAN does not beat KMeans J=43.** It performs dramatically worse:
- Best holdout r = 0.5295 vs KMeans 0.8428 (gap of -0.3133)
- HDBSCAN finds only 2-3 clusters (one per state) regardless of min_cluster_size
- This is not a parameter tuning problem — it reflects a fundamental mismatch between
  HDBSCAN's density-based approach and the structure of electoral shift data

## Why HDBSCAN Fails Here

HDBSCAN requires data with meaningfully denser sub-regions in feature space. The
FL+GA+AL electoral shift space has a different structure:

1. **State-level density dominates**: Counties within the same state share
   governor/Senate shift patterns, creating three strong state-level density peaks.
2. **Electoral types are not dense sub-clusters**: Types like "Rural White
   Conservative" or "Black Belt" are scattered continuously across the shift space
   — they do not form tight density islands. The types are better described as
   centroids in a continuous distribution, which is exactly what KMeans captures.
3. **Dimensionality**: At 39 dimensions, density estimation is challenging — the
   curse of dimensionality weakens HDBSCAN's core mechanism.

## Recommendation

**Do not adopt HDBSCAN. KMeans J=43 remains the production algorithm.**

HDBSCAN is categorically unsuited to this data structure. The spectral clustering
experiment (S175) showed no systematic advantage over KMeans; HDBSCAN shows a
severe disadvantage. The KMeans centroid-based approach is the correct algorithmic
choice for discovering fine-grained electoral types that cross state boundaries.

Future clustering experiments should focus on:
- Expanding the county coverage to all 50 states (national expansion)
- Tuning J for the national model once data is assembled
- Alternative soft membership schemes (Gaussian mixture model as a potential upgrade)
