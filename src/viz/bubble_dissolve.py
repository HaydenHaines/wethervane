"""Bubble dissolve: merge adjacent same-type tracts into community polygons.

Produces the "stained glass" visualization at tract level — same-type tracts
that touch each other (Queen contiguity) merge into one polygon, creating
emergent community shapes.

CLI usage:
    python -m src.viz.bubble_dissolve \
        --input data/experiments/{name}/assignments.parquet \
        --tracts data/raw/tiger/ \
        --output data/experiments/{name}/dissolved_communities.geojson
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
from libpysal.weights import Queen
from shapely.ops import unary_union


def bubble_dissolve(
    tract_gdf: gpd.GeoDataFrame,
    min_area_sqkm: float = 0.1,
    simplify_tolerance: float = 0.001,
    super_type_names: dict[int, str] | None = None,
    type_demographics: "dict[int, dict[str, float]] | None" = None,
) -> gpd.GeoDataFrame:
    """Merge adjacent same-type tracts into community polygons.

    Parameters
    ----------
    tract_gdf : GeoDataFrame
        Must have columns: geometry, dominant_type, super_type.
    min_area_sqkm : float
        Minimum polygon area in square kilometres.  Smaller polygons are
        filtered out.  Pass 0.0 to keep all polygons.
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance applied after merging.
        Units are those of the output CRS (degrees for geographic, metres for
        projected).  Pass 0.0 to skip simplification.
    super_type_names : dict[int, str] | None
        Optional mapping of super_type id → human-readable display name.
        If omitted, names default to ``'Type {id}'``.
    type_demographics : dict[int, dict[str, float]] | None
        Optional mapping of type_id → demographic summary stats dict.
        Keys: median_hh_income, pct_ba_plus, pct_white_nh, pct_black,
              pct_hispanic, evangelical_share.
        These are embedded as flat properties on each output polygon.

    Returns
    -------
    GeoDataFrame with columns:
        geometry        : merged polygon (WGS84 / EPSG:4326)
        type_id         : int, the dominant_type value shared by the component
        super_type      : int, modal super_type across component tracts
        super_type_name : str, display name for super_type
        n_tracts        : int, number of tracts in the component
        area_sqkm       : float, polygon area in square kilometres (2 d.p.)
        + optional demographic columns if type_demographics provided
    """
    if super_type_names is None:
        super_type_names = {}
    if type_demographics is None:
        type_demographics = {}
    # 1. Work in EPSG:5070 (Albers Equal Area, metres) for area calculation
    gdf = tract_gdf.to_crs("EPSG:5070").copy()
    gdf = gdf.reset_index(drop=True)  # ensure 0-based integer index

    # 2. Build Queen contiguity weights (8-connected, includes diagonals)
    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)

    # 3. For each type, find connected components of adjacent same-type tracts
    communities: list[dict] = []

    for type_id in sorted(gdf["dominant_type"].unique()):
        type_mask = gdf["dominant_type"] == type_id
        type_indices = gdf[type_mask].index.tolist()
        type_set = set(type_indices)

        # Build subgraph restricted to same-type tracts
        G: nx.Graph = nx.Graph()
        G.add_nodes_from(type_indices)
        for i in type_indices:
            for j in w.neighbors[i]:
                if j in type_set:
                    G.add_edge(i, j)

        # Each connected component -> one candidate community polygon
        for component in nx.connected_components(G):
            component_list = list(component)
            component_tracts = gdf.loc[component_list]
            merged_geom = unary_union(component_tracts.geometry)
            area_sqkm = merged_geom.area / 1e6

            if area_sqkm < min_area_sqkm:
                continue

            if simplify_tolerance > 0:
                merged_geom = merged_geom.simplify(simplify_tolerance)

            super_type = int(component_tracts["super_type"].mode().iloc[0])
            record: dict = {
                "geometry": merged_geom,
                "type_id": int(type_id),
                "super_type": super_type,
                "super_type_name": super_type_names.get(super_type, f"Type {super_type}"),
                "n_tracts": len(component_list),
                "area_sqkm": round(area_sqkm, 2),
            }
            # Embed type-level demographic averages if provided
            if type_id in type_demographics:
                record.update(type_demographics[type_id])
            communities.append(record)

    if not communities:
        return gpd.GeoDataFrame(
            columns=["geometry", "type_id", "super_type", "super_type_name", "n_tracts", "area_sqkm"],
            crs="EPSG:4326",
        )

    result = gpd.GeoDataFrame(communities, crs=gdf.crs)
    # Reproject to WGS84 for GeoJSON output
    return result.to_crs("EPSG:4326")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dissolve adjacent same-type tracts into community polygons."
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to assignments.parquet (must have geoid/tract_id column + dominant_type + super_type).",
    )
    p.add_argument(
        "--tracts",
        required=True,
        type=Path,
        help="Path to TIGER tract shapefiles directory.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for dissolved_communities.geojson.",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=0.1,
        dest="min_area",
        help="Minimum polygon area in sqkm (default: 0.1).",
    )
    p.add_argument(
        "--simplify",
        type=float,
        default=0.001,
        dest="simplify",
        help="Douglas-Peucker simplification tolerance (default: 0.001).",
    )
    p.add_argument(
        "--super-types",
        type=Path,
        default=None,
        help="Path to super_types.parquet for display names. If omitted, names default to 'Type {id}'.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    import pandas as pd

    args = _build_parser().parse_args(argv)

    st_names: dict[int, str] = {}
    if args.super_types and args.super_types.exists():
        st_df = pd.read_parquet(args.super_types)
        st_names = dict(zip(st_df["super_type_id"], st_df["display_name"]))

    assignments = pd.read_parquet(args.input)

    # Load tract geometries from TIGER directory
    shp_files = list(args.tracts.glob("*.shp"))
    if not shp_files:
        print(f"ERROR: No shapefiles found in {args.tracts}", file=sys.stderr)
        sys.exit(1)

    tract_geoms = gpd.read_file(shp_files[0])
    for shp in shp_files[1:]:
        tract_geoms = gpd.pd.concat([tract_geoms, gpd.read_file(shp)], ignore_index=True)

    # Join assignments onto geometries
    join_col = "GEOID" if "GEOID" in tract_geoms.columns else tract_geoms.columns[0]
    tract_gdf = tract_geoms.merge(
        assignments[["geoid", "dominant_type", "super_type"]],
        left_on=join_col,
        right_on="geoid",
        how="inner",
    )

    result = bubble_dissolve(
        tract_gdf,
        min_area_sqkm=args.min_area,
        simplify_tolerance=args.simplify,
        super_type_names=st_names,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(args.output, driver="GeoJSON")
    print(
        f"Wrote {len(result)} community polygons to {args.output} "
        f"({result['type_id'].nunique()} types)"
    )


if __name__ == "__main__":
    main()
