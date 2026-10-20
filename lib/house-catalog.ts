import type { HousePublication, HouseResult } from "./house-result";
import { houseReadinessInput, parseHouseResult } from "./house-result.ts";
import { validateChamberReadiness } from "./chamber.ts";
import { joinDistrictGeography, type DistrictGeography } from "./district-geography.ts";

export const HOUSE_CAPABILITY_KEYS = [
  "historical_races", "polling", "district_geometry", "full_chamber",
  "chamber_control", "chamber_tipping", "current_2026",
] as const;
export type HouseCapabilities = Record<(typeof HOUSE_CAPABILITY_KEYS)[number], boolean>;
export type HouseCatalogRun = HouseResult["run"] & {
  resultSha256: string | null;
  raceCount: number;
  available: boolean;
  unavailableReason: string | null;
};
export interface HouseCatalog {
  schemaVersion: 1;
  defaultRunId: string;
  selectionNote: string;
  runs: HouseCatalogRun[];
  capabilities?: HouseCapabilities;
}
export type HousePublicationWithCatalog = HousePublication & { catalog?: HouseCatalog };

const record = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 1000 && value === value.trim();
export const validHouseRunId = (value: unknown): value is string =>
  typeof value === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}$/.test(value);
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);

/** Canonical presentation metadata only: no filenames, source paths or extra fields. */
export function parseHouseRun(value: unknown): HouseResult["run"] | null {
  if (!record(value) || !validHouseRunId(value.id) || !text(value.configurationId) ||
    !hash(value.configurationSha256) || !text(value.label) ||
    typeof value.year !== "number" || !Number.isInteger(value.year) || value.year < 1 ||
    typeof value.historical !== "boolean" || typeof value.experimental !== "boolean" ||
    !["retrospective_proxy_strictly_earlier_fitted_labels", "strict_asof"].includes(String(value.informationMode)) ||
    !text(value.evidenceLabel)) return null;
  return {
    id: value.id, configurationId: value.configurationId, configurationSha256: value.configurationSha256,
    label: value.label, year: value.year, historical: value.historical, experimental: value.experimental,
    informationMode: value.informationMode as HouseResult["run"]["informationMode"], evidenceLabel: value.evidenceLabel,
  };
}

export function houseRunMatches(first: HouseResult["run"], second: HouseResult["run"]): boolean {
  return (Object.keys(first) as (keyof HouseResult["run"])[])
    .every(key => first[key] === second[key]) && Object.keys(first).length === Object.keys(second).length;
}

/** Browser-safe, bounded, fail-closed parser. Unavailable choices never carry a result. */
export function parseHouseCatalog(value: unknown): HouseCatalog | null {
  if (!record(value) || value.schemaVersion !== 1 || !validHouseRunId(value.defaultRunId) ||
    !text(value.selectionNote) || !Array.isArray(value.runs) || value.runs.length < 1 || value.runs.length > 128) return null;
  const runs: HouseCatalogRun[] = [], ids = new Set<string>();
  for (const entry of value.runs) {
    const run = parseHouseRun(entry);
    if (!run || !record(entry) || ids.has(run.id) || typeof entry.available !== "boolean" ||
      typeof entry.raceCount !== "number" || !Number.isInteger(entry.raceCount) || entry.raceCount < 0 || entry.raceCount > 435 ||
      (entry.available ? !hash(entry.resultSha256) || entry.raceCount === 0 || entry.unavailableReason !== null
        : entry.resultSha256 !== null || !text(entry.unavailableReason))) return null;
    ids.add(run.id);
    runs.push({ ...run, resultSha256: entry.resultSha256 as string | null, raceCount: entry.raceCount,
      available: entry.available, unavailableReason: entry.unavailableReason as string | null });
  }
  if (!runs.find(run => run.id === value.defaultRunId)?.available) return null;
  let capabilities: HouseCapabilities | undefined;
  if (value.capabilities !== undefined) {
    if (!record(value.capabilities) || Object.keys(value.capabilities).length !== HOUSE_CAPABILITY_KEYS.length ||
      !HOUSE_CAPABILITY_KEYS.every(key => typeof (value.capabilities as Record<string, unknown>)[key] === "boolean")) return null;
    capabilities = Object.fromEntries(HOUSE_CAPABILITY_KEYS.map(key => [key, (value.capabilities as Record<string, unknown>)[key]])) as HouseCapabilities;
  }
  return { schemaVersion: 1, defaultRunId: value.defaultRunId, selectionNote: value.selectionNote, runs,
    ...(capabilities ? { capabilities } : {}) };
}

/** Validate an API response without importing server files or trusting its catalog label. */
export function parseHousePublicationWithCatalog(value: unknown): HousePublicationWithCatalog | null {
  try {
    if (!record(value) || value.schemaVersion !== 1) return null;
    if (value.status === "preparing") return value.catalog === undefined ? { schemaVersion: 1, status: "preparing" } : null;
    if (value.status !== "available" || !hash(value.resultSha256)) return null;
    const result = parseHouseResult(value.result);
    if (!result) return null;
    let geography: DistrictGeography | null = null;
    if (record(value.geography)) {
      try {
        const candidate = value.geography as unknown as DistrictGeography;
        const joined = joinDistrictGeography(result.races.map(race => ({ race, districtKey: race.district.districtKey })),
          { cycle: result.run.year, mapVintage: result.metadata.mapVintage }, candidate);
        if (!["invalid", "unavailable"].includes(joined.status) && !joined.missingDistrictKeys.length) geography = candidate;
      } catch {
        // Geography admission is independent of valid race-level results.
      }
    }
    const readiness = validateChamberReadiness(houseReadinessInput(result, geography));
    if (!readiness.canShowRaces || (result.metadata.full_chamber && !readiness.canShowControl)) return null;
    let catalog: HouseCatalog | undefined;
    if (value.catalog !== undefined) {
      catalog = parseHouseCatalog(value.catalog) ?? undefined;
      if (!catalog) return null;
      const advertised = catalog.runs.find(run => run.id === result.run.id);
      const run = parseHouseRun(result.run);
      if (!advertised?.available || !run || !houseRunMatches(parseHouseRun(advertised)!, run) ||
        advertised.resultSha256 !== value.resultSha256 || advertised.raceCount !== result.races.length) return null;
    }
    return { schemaVersion: 1, status: "available", result, resultSha256: value.resultSha256, geography,
      ...(catalog ? { catalog } : {}) };
  } catch {
    return null;
  }
}
