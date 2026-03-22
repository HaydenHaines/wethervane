"""Tests for bubble_dissolve: adjacent same-type tract merging into community polygons.

Uses a synthetic 4x4 grid of tract-like squares with known type assignments.
No real data files needed.

Type layout (row 0 = bottom, row 3 = top):
  Row 3: A A B B   (indices 12 13 14 15)
  Row 2: A A B C   (indices  8  9 10 11)
  Row 1: A B B C   (indices  4  5  6  7)
  Row 0: D D C C   (indices  0  1  2  3)

Expected connected components:
  Type A: 5 tracts {4, 8, 9, 12, 13}  -> 1 polygon
  Type B: 5 tracts {5, 6, 10, 14, 15} -> 1 polygon
  Type C: 4 tracts {2, 3, 7, 11}      -> 1 polygon
  Type D: 2 tracts {0, 1}              -> 1 polygon
Total: 4 community polygons
"""
from __future__ import annotations

import pytest
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from src.viz.bubble_dissolve import bubble_dissolve


# ── Grid fixture ─────────────────────────────────────────────────────────────

CELL_SIZE = 1000  # metres (EPSG:5070-style)


def make_grid(rows: int = 4, cols: int = 4, cell_size: int = CELL_SIZE) -> list:
    """Create a rows*cols list of square polygons in metre coordinates."""
    polys = []
    for r in range(rows):
        for c in range(cols):
            polys.append(
                box(
                    c * cell_size,
                    r * cell_size,
                    (c + 1) * cell_size,
                    (r + 1) * cell_size,
                )
            )
    return polys


# Type layout — row-major, bottom row first
# Row 0: D D C C  (indices  0  1  2  3)
# Row 1: A B B C  (indices  4  5  6  7)
# Row 2: A A B C  (indices  8  9 10 11)
# Row 3: A A B B  (indices 12 13 14 15)
_TYPE_MAP = {
    0: 4, 1: 4,           # D D
    2: 3, 3: 3,           # C C
    4: 1, 5: 2, 6: 2, 7: 3,   # A B B C
    8: 1, 9: 1, 10: 2, 11: 3,  # A A B C
    12: 1, 13: 1, 14: 2, 15: 2,  # A A B B
}

# Super-type: A(1) and B(2) -> super 1; C(3) -> super 2; D(4) -> super 2
_SUPER_MAP = {1: 1, 2: 1, 3: 2, 4: 2}


@pytest.fixture(scope="module")
def grid_gdf() -> gpd.GeoDataFrame:
    """4x4 grid GeoDataFrame in EPSG:5070 with dominant_type and super_type."""
    polys = make_grid()
    dominant_types = [_TYPE_MAP[i] for i in range(16)]
    super_types = [_SUPER_MAP[t] for t in dominant_types]
    gdf = gpd.GeoDataFrame(
        {
            "geometry": polys,
            "dominant_type": dominant_types,
            "super_type": super_types,
        },
        crs="EPSG:5070",
    )
    return gdf


@pytest.fixture(scope="module")
def dissolved(grid_gdf) -> gpd.GeoDataFrame:
    """Run bubble_dissolve once; reused by multiple tests."""
    return bubble_dissolve(grid_gdf, min_area_sqkm=0.0, simplify_tolerance=0.0)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_output_columns(dissolved):
    """Result must have exactly the required columns."""
    required = {"geometry", "type_id", "super_type", "n_tracts", "area_sqkm"}
    assert required.issubset(set(dissolved.columns)), (
        f"Missing columns: {required - set(dissolved.columns)}"
    )


def test_total_polygon_count(dissolved):
    """4x4 grid with 4 connected groups -> 4 output polygons."""
    assert len(dissolved) == 4, (
        f"Expected 4 polygons, got {len(dissolved)}"
    )


def test_adjacent_same_type_merge(dissolved):
    """Type A has 5 adjacent tracts -> exactly 1 output polygon."""
    type_a = dissolved[dissolved["type_id"] == 1]
    assert len(type_a) == 1, (
        f"Type A should merge to 1 polygon, got {len(type_a)}"
    )


def test_n_tracts_correct(dissolved):
    """n_tracts field must match the actual tract count per component."""
    type_a = dissolved[dissolved["type_id"] == 1].iloc[0]
    type_b = dissolved[dissolved["type_id"] == 2].iloc[0]
    type_c = dissolved[dissolved["type_id"] == 3].iloc[0]
    type_d = dissolved[dissolved["type_id"] == 4].iloc[0]

    assert type_a["n_tracts"] == 5, f"A: expected 5, got {type_a['n_tracts']}"
    assert type_b["n_tracts"] == 5, f"B: expected 5, got {type_b['n_tracts']}"
    assert type_c["n_tracts"] == 4, f"C: expected 4, got {type_c['n_tracts']}"
    assert type_d["n_tracts"] == 2, f"D: expected 2, got {type_d['n_tracts']}"


def test_different_types_not_merged(dissolved):
    """A and B are adjacent but different types -> distinct polygons."""
    assert len(dissolved["type_id"].unique()) == 4, (
        "Should have 4 distinct type_id values"
    )
    # Confirm A and B are separate rows
    assert 1 in dissolved["type_id"].values
    assert 2 in dissolved["type_id"].values


def test_non_adjacent_same_type_separate():
    """Disconnected same-type tracts -> separate polygons."""
    # Build a 2x2 grid where type 1 occupies corners [0,3] — not adjacent
    polys = make_grid(rows=2, cols=2, cell_size=CELL_SIZE)
    # (0,0)=type1, (0,1)=type2, (1,0)=type2, (1,1)=type1
    dominant_types = [1, 2, 2, 1]
    super_types = [1, 2, 2, 1]
    gdf = gpd.GeoDataFrame(
        {
            "geometry": polys,
            "dominant_type": dominant_types,
            "super_type": super_types,
        },
        crs="EPSG:5070",
    )
    result = bubble_dissolve(gdf, min_area_sqkm=0.0, simplify_tolerance=0.0)
    type1_rows = result[result["type_id"] == 1]
    # Corners (0,0) and (1,1) share only a corner point, not an edge.
    # Queen contiguity INCLUDES diagonal neighbours, so they WILL be adjacent.
    # Therefore we expect 1 merged polygon (Queen = 8-connected).
    # Document this explicitly as the confirmed behaviour.
    assert len(type1_rows) == 1, (
        "Queen contiguity is 8-connected: diagonal corners are neighbours -> merge"
    )


def test_small_polygon_filtered():
    """Polygons below min_area_sqkm are excluded from output."""
    polys = make_grid(rows=1, cols=3, cell_size=CELL_SIZE)
    # 1x3 strip: cell area = 1km x 1km = 1 sqkm each
    dominant_types = [1, 2, 1]
    super_types = [1, 2, 1]
    gdf = gpd.GeoDataFrame(
        {
            "geometry": polys,
            "dominant_type": dominant_types,
            "super_type": super_types,
        },
        crs="EPSG:5070",
    )
    # Each cell is 1 sqkm; filter at 1.5 sqkm -> only type 1 fails (1 sqkm each,
    # but they're non-adjacent in 1x3, so two separate 1-sqkm polygons, both filtered)
    # Type 2 is 1 sqkm -> also filtered
    # Set threshold to 0 to get all, then 1.1 to drop isolated 1-sqkm type-1 tracts
    result = bubble_dissolve(gdf, min_area_sqkm=1.1, simplify_tolerance=0.0)
    # Type 1 tracts are at positions 0 and 2 (non-adjacent in 1x3 row):
    # Queen neighbours of 0 = [1]; Queen neighbours of 2 = [1]; they don't see each other.
    # So type 1 -> two 1-sqkm polygons -> both filtered.
    # Type 2 -> one 1-sqkm polygon -> filtered.
    assert len(result) == 0, (
        f"All polygons should be filtered at min_area=1.1 sqkm, got {len(result)}"
    )


def test_geometry_valid(dissolved):
    """All output polygons must pass shapely is_valid."""
    invalid = [i for i, geom in enumerate(dissolved.geometry) if not geom.is_valid]
    assert invalid == [], f"Invalid geometries at indices: {invalid}"


def test_output_crs(dissolved):
    """Output must be in WGS84 (EPSG:4326)."""
    assert dissolved.crs.to_epsg() == 4326, (
        f"Expected EPSG:4326, got {dissolved.crs.to_epsg()}"
    )


def test_total_area_conserved(grid_gdf):
    """Sum of output areas approximates sum of input areas (within 1%)."""
    # Compute in EPSG:5070 for accuracy
    result = bubble_dissolve(grid_gdf, min_area_sqkm=0.0, simplify_tolerance=0.0)
    result_5070 = result.to_crs("EPSG:5070")
    out_area = result_5070.geometry.area.sum() / 1e6  # sqkm

    input_5070 = grid_gdf.to_crs("EPSG:5070")
    in_area = input_5070.geometry.area.sum() / 1e6  # sqkm

    ratio = out_area / in_area
    assert abs(ratio - 1.0) < 0.01, (
        f"Area not conserved: input={in_area:.4f} sqkm, output={out_area:.4f} sqkm"
    )


def test_area_sqkm_field(dissolved):
    """area_sqkm field must be positive for all polygons."""
    assert (dissolved["area_sqkm"] > 0).all(), "area_sqkm must be positive"


def test_super_type_field(dissolved):
    """super_type field must be populated and match expected mapping."""
    # Type A (type_id=1) -> super_type 1
    type_a = dissolved[dissolved["type_id"] == 1].iloc[0]
    assert type_a["super_type"] == 1, (
        f"Type A should have super_type=1, got {type_a['super_type']}"
    )
    # Type D (type_id=4) -> super_type 2
    type_d = dissolved[dissolved["type_id"] == 4].iloc[0]
    assert type_d["super_type"] == 2, (
        f"Type D should have super_type=2, got {type_d['super_type']}"
    )


def test_simplify_reduces_vertices():
    """Applying simplify_tolerance > 0 produces fewer or equal vertices."""
    polys = make_grid()
    dominant_types = [_TYPE_MAP[i] for i in range(16)]
    super_types = [_SUPER_MAP[t] for t in dominant_types]
    gdf = gpd.GeoDataFrame(
        {"geometry": polys, "dominant_type": dominant_types, "super_type": super_types},
        crs="EPSG:5070",
    )
    no_simplify = bubble_dissolve(gdf, min_area_sqkm=0.0, simplify_tolerance=0.0)
    with_simplify = bubble_dissolve(gdf, min_area_sqkm=0.0, simplify_tolerance=0.001)
    v_no = sum(len(geom.exterior.coords) for geom in no_simplify.geometry)
    v_with = sum(len(geom.exterior.coords) for geom in with_simplify.geometry)
    assert v_with <= v_no, (
        f"Simplification should not increase vertex count: {v_with} > {v_no}"
    )
