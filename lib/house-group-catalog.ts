import reviewedCatalogBindings from "../data/house-groups/bindings-721d5f0d7a1d0e90a76d68da3ba362223d47ef74f91dc3edfd7a041d55bf2a08.json" with { type: "json" };
import { parseHouseGroupView, type HouseGroupResult } from "./house-group-result.ts";
import { joinDistrictGeography, type DistrictGeography } from "./district-geography.ts";
import { groupRaceForMap } from "./house-group-presentation.ts";

export const HOUSE_GROUP_QUALIFICATION_SHA = "0f15deb174095565baba3e8bc76af9b7c39d1ba781fdd2d96afd4cd4111908f2";
export const HOUSE_GROUP_POLICY_SHA = "281055cecd26412e4e896817791d58974f01f3b5d848228c5b0a7830de07a5e3";
export const HOUSE_GROUP_SOURCE_CATALOG_SHA = "57b6b79614e1132297cb5519b67c6ffa48e9b73820552c7c8cb254fe948169f6";
/** Canonical, path-free public catalog, including every identity/hash binding and default. */
export const HOUSE_GROUP_PUBLIC_CATALOG_SHA = "721d5f0d7a1d0e90a76d68da3ba362223d47ef74f91dc3edfd7a041d55bf2a08";
export interface HouseGroupCatalogEntry {
  id: string; cycle: number; fundamentalsId: string; label: string;
  resultSha256: string; viewSha256: string; legacyRunId: string; legacyResultSha256: string;
  mapVintage: string; geographySha256: string; geographyContentSha256: string;
}
export interface HouseGroupCatalog {
  schemaVersion: 1; qualificationSha256: string; sourcePolicySha256: string; sourceCatalogSha256: string;
  initialRunId: string; runs: HouseGroupCatalogEntry[];
}
export interface HouseGroupPublication {
  schemaVersion: 1; status: "available"; family: "house_groups";
  catalog: HouseGroupCatalog; entry: HouseGroupCatalogEntry; result: HouseGroupResult; geography: DistrictGeography;
}
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
export const groupRunId = (v: unknown): v is string => typeof v === "string" && /^house_(history_refreshed|presidential_slope|relative_lean|relative_context)_v1-(2018|2020|2022|2024)-h14-full435-v1$/.test(v);
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
export function canonicalGroupJson(value: unknown): string {
  const normalize = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(normalize);
    if (record(v)) return Object.fromEntries(Object.keys(v).sort().map(key => [key, normalize(v[key])]));
    if (v === undefined || (typeof v === "number" && !Number.isFinite(v))) throw Error("Invalid canonical value.");
    return v;
  };
  return JSON.stringify(normalize(value));
}
export function parseHouseGroupCatalog(value: unknown): HouseGroupCatalog | null {
  try {
  if (!record(value) || value.schemaVersion !== 1 || value.qualificationSha256 !== HOUSE_GROUP_QUALIFICATION_SHA ||
    value.sourcePolicySha256 !== HOUSE_GROUP_POLICY_SHA || value.sourceCatalogSha256 !== HOUSE_GROUP_SOURCE_CATALOG_SHA ||
    !groupRunId(value.initialRunId) || !Array.isArray(value.runs) || value.runs.length !== 16) return null;
  const runs: HouseGroupCatalogEntry[] = [], ids = new Set<string>();
  for (const r of value.runs) {
    if (!record(r) || !groupRunId(r.id) || ids.has(r.id) || ![2018, 2020, 2022, 2024].includes(r.cycle as number) ||
      typeof r.fundamentalsId !== "string" || r.id !== `${r.fundamentalsId}-${r.cycle}-h14-full435-v1` ||
      typeof r.label !== "string" || !r.label.trim() || r.label.length > 100 || !hash(r.resultSha256) || !hash(r.viewSha256) ||
      r.legacyRunId !== r.id.replace(/-full435-v1$/, "") || !hash(r.legacyResultSha256) ||
      r.mapVintage !== `house_election_map_${r.cycle}` || !hash(r.geographySha256) || !hash(r.geographyContentSha256)) return null;
    ids.add(r.id);
    runs.push({ id: r.id, cycle: r.cycle as number, fundamentalsId: r.fundamentalsId, label: r.label,
      resultSha256: r.resultSha256, viewSha256: r.viewSha256, legacyRunId: r.legacyRunId as string,
      legacyResultSha256: r.legacyResultSha256, mapVintage: r.mapVintage, geographySha256: r.geographySha256, geographyContentSha256:r.geographyContentSha256 });
  }
  if (!ids.has(value.initialRunId)) return null;
  const parsed: HouseGroupCatalog = { schemaVersion: 1, qualificationSha256: HOUSE_GROUP_QUALIFICATION_SHA, sourcePolicySha256: HOUSE_GROUP_POLICY_SHA,
    sourceCatalogSha256: HOUSE_GROUP_SOURCE_CATALOG_SHA, initialRunId: value.initialRunId, runs };
  // Exact canonical equality enforces the pinned digest synchronously without custom browser crypto.
  if (canonicalGroupJson(parsed) !== canonicalGroupJson(reviewedCatalogBindings)) return null;
  return parsed;
  } catch { return null; }
}

async function canonicalDigest(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalGroupJson(value));
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(n => n.toString(16).padStart(2, "0")).join("");
}
/** Verify reconstructed content against its exact published fingerprint in the browser. */
export async function parseHouseGroupPublication(value: unknown): Promise<HouseGroupPublication | null> {
  try {
    if (!record(value) || value.schemaVersion !== 1 || value.status !== "available" || value.family !== "house_groups") return null;
    const catalog = parseHouseGroupCatalog(value.catalog), result = parseHouseGroupView(value.result);
    if (!catalog || !result || !record(value.entry)) return null;
    const entry = catalog.runs.find(run => run.id === result.id);
    if (!entry || canonicalGroupJson(value.entry) !== canonicalGroupJson(entry) || result.cycle !== entry.cycle ||
      result.configuration !== `${entry.fundamentalsId}-loading-1` || result.legacyRunId !== entry.legacyRunId ||
      result.provenance.sourcePolicySha256 !== catalog.sourcePolicySha256) return null;
    if (await canonicalDigest(catalog) !== HOUSE_GROUP_PUBLIC_CATALOG_SHA || await canonicalDigest(result) !== entry.viewSha256) return null;
    const geography = value.geography as DistrictGeography;
    if (geography?.metadata?.sourceSha256 !== entry.geographySha256 || geography.metadata.mapVintage !== entry.mapVintage || geography.geojson.features.length !== 435) return null;
    if (await canonicalDigest(geography.geojson) !== entry.geographyContentSha256) return null;
    const joined = joinDistrictGeography(result.races.map(race => ({ race: groupRaceForMap(race), districtKey: race.districtKey })),
      { cycle: result.cycle, mapVintage: entry.mapVintage }, geography);
    if (joined.status !== "ready" || joined.matches.length !== 435) return null;
    return { schemaVersion: 1, status: "available", family: "house_groups", catalog, entry, result, geography };
  } catch { return null; }
}
