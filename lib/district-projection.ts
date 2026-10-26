import { geoArea } from "d3-geo";
import type { Position } from "geojson";
import type { DistrictFeature } from "./district-geography.ts";

/**
 * D3 expects clockwise exterior rings and counterclockwise holes for districts
 * smaller than a hemisphere. Normalize each ring, including each island of a
 * MultiPolygon: an aggregate feature area cannot identify mixed winding.
 * This display copy preserves every coordinate and the authenticated source.
 * Convention: https://d3js.org/d3-geo
 */
export function districtForProjection(feature: DistrictFeature): DistrictFeature {
  const ring = (points: Position[], hole: boolean): Position[] => {
    const complement = geoArea({ type: "Polygon", coordinates: [points] }) > 2 * Math.PI;
    return complement !== hole ? [...points].reverse() : points;
  };
  const polygon = (rings: Position[][]): Position[][] =>
    rings.map((points, index) => ring(points, index > 0));
  const geometry = feature.geometry;
  return {
    ...feature,
    geometry: geometry.type === "Polygon"
      ? { ...geometry, coordinates: polygon(geometry.coordinates) }
      : { ...geometry, coordinates: geometry.coordinates.map(polygon) },
  };
}
