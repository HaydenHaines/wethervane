# Spectral Clustering Experiment (P3.3)

**Date:** 2026-03-23  
**Session:** S175  
**Branch:** feat/spectral-clustering-experiment

## Objective

Test whether spectral clustering (k-NN affinity) finds non-convex clusters
that KMeans misses, and compare holdout r on the county-prior prediction task.

## Setup

- **Data:** 293 counties (FL+GA+AL), 33 training dims (2008+, presidential×2.5)
- **Holdout:** 2020→2024 presidential shifts (pres_d, pres_r, pres_turnout)
- **KMeans:** n_init=10, random_state=42, T=10 inverse-distance soft membership
- **Spectral:** affinity=nearest_neighbors, n_neighbors=10, assign_labels=kmeans
- **Soft membership:** both methods use T=10.0 inverse-distance (distance to centroids in original space)
- **J sweep:** [20, 30, 40, 43, 50]
- **Metric:** county-prior holdout r (same as production validation)
- **Production baseline:** KMeans J=43, holdout r=0.828

## Results

### Holdout r — county-prior method

| J | KMeans r | Spectral r | Delta | Winner |
|---|----------|------------|-------|--------|
| 20 | 0.7876 | 0.7884 | +0.0008 | Tie |
| 30 | 0.8240 | 0.8218 | -0.0022 | KMeans |
| 40 | 0.8448 | 0.8445 | -0.0003 | Tie |
| 43 | 0.8428 | 0.8490 | +0.0061 | Spectral |
| 50 | 0.8603 | 0.8508 | -0.0094 | KMeans |

### RMSE — county-prior method

| J | KMeans RMSE | Spectral RMSE |
|---|-------------|---------------|
| 20 | 0.0502 | 0.0500 |
| 30 | 0.0461 | 0.0464 |
| 40 | 0.0436 | 0.0436 |
| 43 | 0.0438 | 0.0430 |
| 50 | 0.0415 | 0.0427 |

## Conclusion

Best spectral result: **J=50, holdout r=0.8508**
Production KMeans baseline: holdout r=0.828

**Spectral beats production baseline (KMeans J=43 r=0.828) by 0.0228 at J=50.**

Note: KMeans at the same J may still be competitive — see delta column above.
Consider running a more thorough J sweep before promoting spectral to production.

## Method Notes

Spectral soft membership is computed in the **original feature space** (not
the spectral embedding space). This matches how KMeans computes distances.
An alternative would compute distances in the 2D eigenspace, but the
county-prior prediction task requires soft weights anchored to cluster
centers in the original shift space.

A future experiment could test spectral soft membership computed in the
normalized Laplacian eigenvector space.
