import type {
  Feature,
  FeatureCollection,
  MultiPolygon,
  Polygon,
} from "geojson";
import type { ExplorerRace } from "./explorer-types";

export interface DistrictRace {
  race: ExplorerRace;
  /** Canonical state postal code and two-digit district, including explicit at-large 00. */
  districtKey: string;
}

export interface DistrictGeographyIdentity {
  cycle: number;
  /** Exact edition identity; a matching label alone does not admit legal geometry. */
  mapVintage: string;
}

export interface DistrictGeographyMetadata extends DistrictGeographyIdentity {
  admission: "verified" | "unverified";
  /** Fingerprint of the geometry source bytes verified by the supplying adapter. */
  sourceSha256: string;
  sourceName?: string;
  sourceUrl?: string;
  sourceNote?: string;
}

export interface DistrictFeatureProperties {
  districtKey: string;
  [key: string]: unknown;
}
export type DistrictFeature = Feature<
  Polygon | MultiPolygon,
  DistrictFeatureProperties
>;
export type DistrictFeatureCollection = FeatureCollection<
  Polygon | MultiPolygon,
  DistrictFeatureProperties
>;
export interface DistrictGeography {
  metadata: DistrictGeographyMetadata;
  geojson: DistrictFeatureCollection;
}

export interface DistrictMatch {
  identity: DistrictGeographyIdentity & { districtKey: string };
  /** JSON tuple avoids delimiter collisions in cycle/vintage/district identity. */
  identityKey: string;
  race: ExplorerRace;
  feature: DistrictFeature;
}

export interface DistrictGeographyDiagnostic {
  code:
    | "invalid_identity"
    | "cycle_mismatch"
    | "vintage_mismatch"
    | "unverified_geometry"
    | "missing_source_digest"
    | "invalid_collection"
    | "invalid_district_key"
    | "duplicate_race_id"
    | "duplicate_race_key"
    | "duplicate_feature_key"
    | "race_identity_mismatch"
    | "invalid_geometry"
    | "missing_geometry"
    | "uncovered_geometry";
  message: string;
  keys?: string[];
}

export interface DistrictGeographyJoin {
  status: "ready" | "partial" | "unavailable" | "invalid";
  matches: DistrictMatch[];
  unmatchedFeatures: DistrictFeature[];
  missingDistrictKeys: string[];
  extraDistrictKeys: string[];
  diagnostics: DistrictGeographyDiagnostic[];
}

/** No padding, state conversion, district-01 substitution, or cross-vintage crosswalk is inferred. */
export function parseDistrictKey(
  value: unknown,
): { state: string; district: string } | null {
  if (typeof value !== "string" || !/^[A-Z]{2}-\d{2}$/.test(value)) return null;
  return { state: value.slice(0, 2), district: value.slice(3) };
}

export function districtIdentityKey(
  identity: DistrictGeographyIdentity,
  districtKey: string,
): string {
  return JSON.stringify([identity.cycle, identity.mapVintage, districtKey]);
}

const lexical = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
function validIdentity(
  identity: DistrictGeographyIdentity | undefined,
): boolean {
  return (
    !!identity &&
    Number.isInteger(identity.cycle) &&
    identity.cycle > 0 &&
    typeof identity.mapVintage === "string" &&
    identity.mapVintage.length > 0 &&
    identity.mapVintage === identity.mapVintage.trim()
  );
}
function validGeometry(feature: DistrictFeature): boolean {
  const geometry = feature.geometry;
  if (!geometry || !["Polygon", "MultiPolygon"].includes(geometry.type))
    return false;
  const ring = (points: unknown): boolean =>
    Array.isArray(points) &&
    points.length >= 4 &&
    points.every(
      (point) =>
        Array.isArray(point) &&
        point.length >= 2 &&
        Number.isFinite(point[0]) &&
        Number.isFinite(point[1]) &&
        Math.abs(point[0]) <= 180 &&
        Math.abs(point[1]) <= 90,
    ) &&
    points[0][0] === points[points.length - 1][0] &&
    points[0][1] === points[points.length - 1][1];
  const polygon = (rings: unknown): boolean =>
    Array.isArray(rings) && rings.length > 0 && rings.every(ring);
  return geometry.type === "Polygon"
    ? polygon(geometry.coordinates)
    : Array.isArray(geometry.coordinates) &&
        geometry.coordinates.length > 0 &&
        geometry.coordinates.every(polygon);
}

/** Admission is supplied by a trusted data adapter; this join does not certify publisher geometry. */
export function joinDistrictGeography(
  races: readonly DistrictRace[],
  identity: DistrictGeographyIdentity,
  geography: DistrictGeography,
): DistrictGeographyJoin {
  const diagnostics: DistrictGeographyDiagnostic[] = [];
  const empty = (
    status: DistrictGeographyJoin["status"],
  ): DistrictGeographyJoin => ({
    status,
    matches: [],
    unmatchedFeatures: [],
    missingDistrictKeys: [],
    extraDistrictKeys: [],
    diagnostics,
  });
  if (!validIdentity(identity) || !validIdentity(geography?.metadata)) {
    diagnostics.push({
      code: "invalid_identity",
      message:
        "A valid election cycle and exact map-vintage identity are required for both races and geometry.",
    });
    return empty("invalid");
  }
  if (identity.cycle !== geography.metadata.cycle)
    diagnostics.push({
      code: "cycle_mismatch",
      message: `Race cycle ${identity.cycle} does not match geometry cycle ${geography.metadata.cycle}.`,
    });
  if (identity.mapVintage !== geography.metadata.mapVintage)
    diagnostics.push({
      code: "vintage_mismatch",
      message: `Race map vintage ${identity.mapVintage} does not match geometry vintage ${geography.metadata.mapVintage}.`,
    });
  if (diagnostics.length) return empty("invalid");
  if (geography.metadata.admission !== "verified")
    diagnostics.push({
      code: "unverified_geometry",
      message:
        "This geography has not been admitted as verified geometry for the selected election and map vintage.",
    });
  if (!/^[a-fA-F0-9]{64}$/.test(geography.metadata.sourceSha256 || ""))
    diagnostics.push({
      code: "missing_source_digest",
      message:
        "A verified geometry source fingerprint is required; a publisher-vintage label alone is insufficient.",
    });
  if (diagnostics.length) return empty("unavailable");
  if (
    geography.geojson?.type !== "FeatureCollection" ||
    !Array.isArray(geography.geojson.features)
  ) {
    diagnostics.push({
      code: "invalid_collection",
      message:
        "District geometry must be an injected GeoJSON FeatureCollection.",
    });
    return empty("invalid");
  }
  const byRaceKey = new Map<string, DistrictRace>(),
    byFeatureKey = new Map<string, DistrictFeature>(),
    raceIds = new Set<string>();
  for (const entry of races) {
    const parsed = parseDistrictKey(entry.districtKey);
    if (!parsed) {
      diagnostics.push({
        code: "invalid_district_key",
        message:
          "A race has a noncanonical district key; use STATE-XX, including 00 for at-large districts.",
        keys: [String(entry.districtKey)],
      });
      continue;
    }
    if (byRaceKey.has(entry.districtKey))
      diagnostics.push({
        code: "duplicate_race_key",
        message:
          "Multiple races claim the same district; choose an explicit contest set before mapping.",
        keys: [entry.districtKey],
      });
    if (raceIds.has(entry.race.id))
      diagnostics.push({
        code: "duplicate_race_id",
        message: "Race identifiers must be unique across district entries.",
        keys: [entry.race.id],
      });
    if (
      entry.race.state !== parsed.state ||
      !entry.race.id.startsWith(
        `${identity.cycle}:${entry.districtKey}:house:`,
      ) ||
      !/^[\w-]+$/.test(
        entry.race.id.slice(
          `${identity.cycle}:${entry.districtKey}:house:`.length,
        ),
      )
    )
      diagnostics.push({
        code: "race_identity_mismatch",
        message:
          "A race's cycle, district identifier, chamber, or state conflicts with its declared geography identity.",
        keys: [entry.race.id, entry.districtKey],
      });
    raceIds.add(entry.race.id);
    byRaceKey.set(entry.districtKey, entry);
  }
  for (const feature of geography.geojson.features) {
    const key = feature?.properties?.districtKey;
    if (!parseDistrictKey(key)) {
      diagnostics.push({
        code: "invalid_district_key",
        message: "A geometry feature has a noncanonical district key.",
        keys: [String(key)],
      });
      continue;
    }
    if (byFeatureKey.has(key))
      diagnostics.push({
        code: "duplicate_feature_key",
        message:
          "Geometry district keys collide; duplicate polygons must be resolved in the source adapter.",
        keys: [key],
      });
    if (feature.type !== "Feature" || !validGeometry(feature))
      diagnostics.push({
        code: "invalid_geometry",
        message:
          "District geometry must contain finite, closed Polygon or MultiPolygon rings in longitude/latitude coordinates.",
        keys: [key],
      });
    byFeatureKey.set(key, feature);
  }
  if (diagnostics.length) return empty("invalid");
  const missingDistrictKeys = [...byRaceKey.keys()]
    .filter((key) => !byFeatureKey.has(key))
    .sort(lexical);
  const extraDistrictKeys = [...byFeatureKey.keys()]
    .filter((key) => !byRaceKey.has(key))
    .sort(lexical);
  if (missingDistrictKeys.length)
    diagnostics.push({
      code: "missing_geometry",
      message: `${missingDistrictKeys.length} district races have no matching geometry.`,
      keys: missingDistrictKeys,
    });
  if (extraDistrictKeys.length)
    diagnostics.push({
      code: "uncovered_geometry",
      message: `${extraDistrictKeys.length} geometry districts have no race data; they must remain uncolored.`,
      keys: extraDistrictKeys,
    });
  if (!races.length && !geography.geojson.features.length)
    diagnostics.push({
      code: "missing_geometry",
      message: "No district races or geometry are available.",
    });
  const matches = [...byRaceKey.keys()]
    .filter((key) => byFeatureKey.has(key))
    .sort(lexical)
    .map((key) => ({
      identity: { ...identity, districtKey: key },
      identityKey: districtIdentityKey(identity, key),
      race: byRaceKey.get(key)!.race,
      feature: byFeatureKey.get(key)!,
    }));
  return {
    status: diagnostics.length ? "partial" : "ready",
    matches,
    unmatchedFeatures: extraDistrictKeys.map((key) => byFeatureKey.get(key)!),
    missingDistrictKeys,
    extraDistrictKeys,
    diagnostics,
  };
}
