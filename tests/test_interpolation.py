"""Tests for precinct-to-tract vote interpolation via area overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box

from src.tracts.interpolate_precincts import interpolate_precincts_to_tracts


def _make_tract_gdf(geometries: list, geoids: list[str], crs="EPSG:5070"):
    """Helper: build a tract GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"GEOID": geoids},
        geometry=geometries,
        crs=crs,
    )


def _make_precinct_gdf(
    geometries: list,
    votes_dem: list[float],
    votes_rep: list[float],
    votes_total: list[float],
    crs="EPSG:5070",
):
    """Helper: build a precinct GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "votes_dem": votes_dem,
            "votes_rep": votes_rep,
            "votes_total": votes_total,
        },
        geometry=geometries,
        crs=crs,
    )


class TestFullOverlap:
    """Precinct entirely within one tract gets all votes assigned there."""

    def test_full_overlap(self):
        # Tract: 0-10 x 0-10, Precinct: 2-8 x 2-8 (fully inside)
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        precinct = _make_precinct_gdf([box(2, 2, 8, 8)], [100.0], [50.0], [160.0])

        result = interpolate_precincts_to_tracts(precinct, tract)

        assert len(result) == 1
        assert result.loc[result["GEOID"] == "01001", "votes_dem"].iloc[0] == pytest.approx(100.0)
        assert result.loc[result["GEOID"] == "01001", "votes_rep"].iloc[0] == pytest.approx(50.0)
        assert result.loc[result["GEOID"] == "01001", "votes_total"].iloc[0] == pytest.approx(160.0)


class TestPartialOverlap:
    """Precinct spanning two tracts splits votes by area fraction."""

    def test_partial_overlap_60_40(self):
        # Two tracts side by side: left 0-6, right 6-10 (both 0-10 in y)
        # Precinct: 0-10 x 0-10 — overlaps left 60%, right 40%
        tract_left = box(0, 0, 6, 10)   # area = 60
        tract_right = box(6, 0, 10, 10)  # area = 40
        tracts = _make_tract_gdf([tract_left, tract_right], ["LEFT", "RIGHT"])

        precinct = _make_precinct_gdf([box(0, 0, 10, 10)], [100.0], [200.0], [300.0])

        result = interpolate_precincts_to_tracts(precinct, tracts)

        left_row = result[result["GEOID"] == "LEFT"].iloc[0]
        right_row = result[result["GEOID"] == "RIGHT"].iloc[0]

        assert left_row["votes_dem"] == pytest.approx(60.0, abs=0.1)
        assert left_row["votes_rep"] == pytest.approx(120.0, abs=0.1)
        assert right_row["votes_dem"] == pytest.approx(40.0, abs=0.1)
        assert right_row["votes_rep"] == pytest.approx(80.0, abs=0.1)


class TestNoOverlap:
    """Precinct outside all tracts: tracts get zero votes."""

    def test_no_overlap(self):
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        # Precinct far away
        precinct = _make_precinct_gdf([box(100, 100, 110, 110)], [500.0], [300.0], [800.0])

        result = interpolate_precincts_to_tracts(precinct, tract)

        assert len(result) == 1
        assert result.iloc[0]["votes_dem"] == pytest.approx(0.0)
        assert result.iloc[0]["votes_rep"] == pytest.approx(0.0)
        assert result.iloc[0]["votes_total"] == pytest.approx(0.0)


class TestMultiplePrecinctsSumCorrectly:
    """Multiple precincts overlapping one tract sum votes."""

    def test_multiple_precincts_one_tract(self):
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["TRACT1"])

        # Two precincts fully inside the tract
        p1 = box(0, 0, 5, 5)   # quarter of tract
        p2 = box(5, 5, 10, 10)  # another quarter
        precincts = _make_precinct_gdf(
            [p1, p2],
            [100.0, 200.0],
            [50.0, 100.0],
            [160.0, 310.0],
        )

        result = interpolate_precincts_to_tracts(precincts, tract)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["votes_dem"] == pytest.approx(300.0)
        assert row["votes_rep"] == pytest.approx(150.0)
        assert row["votes_total"] == pytest.approx(470.0)


class TestOutputColumns:
    """Result has required columns."""

    def test_output_columns(self):
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        precinct = _make_precinct_gdf([box(0, 0, 10, 10)], [100.0], [50.0], [150.0])

        result = interpolate_precincts_to_tracts(precinct, tract)

        required = {"GEOID", "votes_dem", "votes_rep", "votes_total", "dem_share"}
        assert required.issubset(set(result.columns))


class TestDemShare:
    """dem_share is correctly computed."""

    def test_dem_share(self):
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        precinct = _make_precinct_gdf([box(0, 0, 10, 10)], [60.0], [40.0], [100.0])

        result = interpolate_precincts_to_tracts(precinct, tract)

        assert result.iloc[0]["dem_share"] == pytest.approx(0.6)


class TestVoteConservation:
    """Total votes in equals total votes out (within float tolerance)."""

    def test_vote_conservation(self):
        # Three tracts, two precincts that partially overlap
        t1 = box(0, 0, 5, 10)
        t2 = box(5, 0, 10, 10)
        t3 = box(10, 0, 15, 10)
        tracts = _make_tract_gdf([t1, t2, t3], ["T1", "T2", "T3"])

        p1 = box(0, 0, 7, 10)    # spans T1 fully + part of T2
        p2 = box(8, 0, 15, 10)   # spans part of T2 + T3 fully
        precincts = _make_precinct_gdf(
            [p1, p2],
            [700.0, 350.0],
            [300.0, 650.0],
            [1000.0, 1000.0],
        )

        result = interpolate_precincts_to_tracts(precincts, tracts)

        total_in_dem = 700.0 + 350.0
        total_in_rep = 300.0 + 650.0
        total_in_all = 1000.0 + 1000.0

        assert result["votes_dem"].sum() == pytest.approx(total_in_dem, abs=0.1)
        assert result["votes_rep"].sum() == pytest.approx(total_in_rep, abs=0.1)
        assert result["votes_total"].sum() == pytest.approx(total_in_all, abs=0.1)


class TestCRSHandling:
    """Input in EPSG:4326 is auto-reprojected."""

    def test_crs_handling(self):
        # Create data in EPSG:4326 (lon/lat)
        tract = _make_tract_gdf([box(-85.0, 30.0, -84.9, 30.1)], ["01001"], crs="EPSG:4326")
        precinct = _make_precinct_gdf(
            [box(-85.0, 30.0, -84.9, 30.1)],
            [100.0], [50.0], [150.0],
            crs="EPSG:4326",
        )

        result = interpolate_precincts_to_tracts(precinct, tract)

        # Should work without error and produce correct results
        assert len(result) == 1
        assert result.iloc[0]["votes_dem"] == pytest.approx(100.0, abs=1.0)
        assert result.iloc[0]["votes_total"] == pytest.approx(150.0, abs=1.0)


class TestZeroAreaPrecinctsSkipped:
    """Point geometries (zero area) are skipped to avoid division by zero."""

    def test_zero_area_skipped(self):
        from shapely.geometry import Point

        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        # Mix of polygon (has area) and point (zero area)
        precincts = _make_precinct_gdf(
            [box(0, 0, 5, 5), Point(5, 5)],
            [100.0, 999.0],
            [50.0, 999.0],
            [150.0, 9999.0],
            crs="EPSG:5070",
        )

        result = interpolate_precincts_to_tracts(precincts, tract)

        # Only the polygon precinct's votes should be allocated
        assert result.iloc[0]["votes_dem"] == pytest.approx(100.0)
        assert result.iloc[0]["votes_total"] == pytest.approx(150.0)


class TestDemShareZeroTotal:
    """dem_share is NaN when votes_total is zero (no division by zero error)."""

    def test_dem_share_zero_total(self):
        tract = _make_tract_gdf([box(0, 0, 10, 10)], ["01001"])
        # Precinct far away — tract gets zero votes
        precinct = _make_precinct_gdf([box(100, 100, 110, 110)], [0.0], [0.0], [0.0])

        result = interpolate_precincts_to_tracts(precinct, tract)

        assert np.isnan(result.iloc[0]["dem_share"])
