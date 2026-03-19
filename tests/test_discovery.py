"""Tests for community discovery pipeline (adjacency, clustering, borders).

Uses synthetic geometries and shift data — no real data files needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.discovery.build_adjacency import (
    build_queen_adjacency,
    handle_islands,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def grid_gdf():
    """4x4 grid of square tract polygons with geoids."""
    import geopandas as gpd
    geoids, geoms = [], []
    for r in range(4):
        for c in range(4):
            geoids.append(f"120010{r:02d}{c:02d}0")
            geoms.append(box(c, r, c + 1, r + 1))
    return gpd.GeoDataFrame({"tract_geoid": geoids, "geometry": geoms}, crs="EPSG:4326")


@pytest.fixture(scope="module")
def grid_with_island(grid_gdf):
    """Grid plus one disconnected island tract."""
    import geopandas as gpd
    island = gpd.GeoDataFrame({
        "tract_geoid": ["12001009990"],
        "geometry": [box(100, 100, 101, 101)],
    }, crs="EPSG:4326")
    result = pd.concat([grid_gdf, island], ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQueenAdjacency:
    def test_returns_sparse_matrix(self, grid_gdf):
        from scipy.sparse import issparse
        W, geoids = build_queen_adjacency(grid_gdf)
        assert issparse(W)

    def test_shape_matches_tract_count(self, grid_gdf):
        W, geoids = build_queen_adjacency(grid_gdf)
        assert W.shape == (16, 16)

    def test_corner_has_3_neighbors(self, grid_gdf):
        """Corner tract in 4x4 grid has 3 Queen neighbors (side + diagonal)."""
        W, geoids = build_queen_adjacency(grid_gdf)
        corner_idx = 0  # (0,0)
        assert W[corner_idx].nnz == 3

    def test_center_has_8_neighbors(self, grid_gdf):
        """Interior tract in 4x4 grid has 8 Queen neighbors."""
        W, geoids = build_queen_adjacency(grid_gdf)
        # (1,1) = row 1 * 4 + col 1 = index 5
        assert W[5].nnz == 8

    def test_symmetric(self, grid_gdf):
        W, geoids = build_queen_adjacency(grid_gdf)
        diff = W - W.T
        assert diff.nnz == 0


class TestIslandHandling:
    def test_island_gets_connected(self, grid_with_island):
        W, geoids = build_queen_adjacency(grid_with_island)
        W_fixed = handle_islands(W, grid_with_island)
        island_idx = len(geoids) - 1
        assert W_fixed[island_idx].nnz >= 1

    def test_no_islands_remain(self, grid_with_island):
        W, _ = build_queen_adjacency(grid_with_island)
        W_fixed = handle_islands(W, grid_with_island)
        # Every row should have at least 1 neighbor
        for i in range(W_fixed.shape[0]):
            assert W_fixed[i].nnz >= 1


# ── Task 3: Cluster Communities ────────────────────────────────────────────────

from src.discovery.cluster_communities import (
    cluster_at_threshold,
    build_linkage_matrix,
    find_elbow,
    normalize_shifts,
)


class TestNormalizeShifts:
    def test_output_zero_mean(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_presidential_cols_unit_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        assert np.allclose(normed[:, :6].var(axis=0), 1.0, atol=0.15)

    def test_midterm_cols_scaled_variance(self):
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(100, 9))
        normed = normalize_shifts(shifts, n_presidential_dims=6)
        assert np.allclose(normed[:, 6:].var(axis=0), 2.0, atol=0.3)


class TestClusterAtThreshold:
    def test_returns_labels(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels, model = cluster_at_threshold(shifts, W, threshold=5.0)
        assert len(labels) == n
        assert labels.min() >= 0

    def test_more_clusters_at_lower_threshold(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels_fine, _ = cluster_at_threshold(shifts, W, threshold=1.0)
        labels_coarse, _ = cluster_at_threshold(shifts, W, threshold=10.0)
        assert len(set(labels_fine)) >= len(set(labels_coarse))


class TestBuildLinkageMatrix:
    def test_linkage_shape(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        _, model = cluster_at_threshold(shifts, W, n_clusters=1)
        linkage = build_linkage_matrix(model)
        assert linkage.shape == (n - 1, 4)

    def test_distances_monotonic(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, _ = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        _, model = cluster_at_threshold(shifts, W, n_clusters=1)
        linkage = build_linkage_matrix(model)
        distances = linkage[:, 2]
        assert np.all(distances[1:] >= distances[:-1])


class TestFindElbow:
    def test_returns_valid_threshold(self):
        n_communities = np.array([200, 150, 100, 80, 60, 50, 45, 42, 40, 39])
        variances = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 1.10, 1.50])
        elbow_k = find_elbow(n_communities, variances)
        assert elbow_k is not None
        assert 40 <= elbow_k <= 200


# ── Task 4: Score Borders ──────────────────────────────────────────────────────

from src.discovery.score_borders import compute_border_gradients


class TestBorderGradients:
    def test_output_columns(self, grid_gdf):
        n = len(grid_gdf)
        rng = np.random.default_rng(42)
        shifts = rng.normal(size=(n, 9))
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert "community_a" in result.columns
        assert "community_b" in result.columns
        assert "gradient" in result.columns
        assert "n_boundary_pairs" in result.columns

    def test_identical_communities_have_zero_gradient(self, grid_gdf):
        n = len(grid_gdf)
        shifts = np.ones((n, 9))
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert (result["gradient"] == 0.0).all()

    def test_different_communities_have_positive_gradient(self, grid_gdf):
        n = len(grid_gdf)
        shifts = np.zeros((n, 9))
        shifts[:8, :] = 1.0
        shifts[8:, :] = -1.0
        W, geoids = build_queen_adjacency(grid_gdf)
        W = handle_islands(W, grid_gdf)
        labels = np.array([0]*8 + [1]*8)
        result = compute_border_gradients(labels, shifts, W, geoids)
        assert (result["gradient"] > 0.0).all()
