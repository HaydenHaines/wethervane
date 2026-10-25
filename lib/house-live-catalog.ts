/** Native saved-run catalog shape; authentication of every pin belongs to the loader. */
import { houseLiveConfigurationKey, houseLivePin, parseHouseLiveConfiguration,
  type HouseLiveConfiguration, type HouseLivePin } from "./house-live-result.ts";

export interface HouseLiveCatalogEntry {
  id: string;
  label: string;
  configuration: HouseLiveConfiguration;
  result: HouseLivePin;
}
export interface HouseLiveCatalog {
  schema: "HouseLiveForecastCatalogV1";
  cycle: 2026;
  default_id: "house-current-polling";
  entries: HouseLiveCatalogEntry[];
  source_policy: HouseLivePin;
  qualification: HouseLivePin;
}

const BASES = ["fundamentals", "recent_calendar_history", "house_history_refreshed_v1",
  "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1"] as const;
const expected = new Set<string>([
  ...BASES.flatMap(base => ["none", "standard"].map(polling => `${base}|${polling}|standard`)),
  ...["cycle-low", "cycle-high", "error-low", "error-high"].map(uncertainty =>
    `fundamentals|standard|${uncertainty}`),
]);
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && v.length > 0 &&
  v.length <= 4000 && v === v.trim() && !/[\u0000-\u001f\u007f]/.test(v);
const id = (v: unknown): v is string => text(v) && /^[A-Za-z0-9][A-Za-z0-9:._-]{0,199}$/.test(v);

export function parseHouseLiveCatalog(value: unknown): HouseLiveCatalog | null {
  if (!object(value) || value.schema !== "HouseLiveForecastCatalogV1" || value.cycle !== 2026 ||
    value.default_id !== "house-current-polling" || !houseLivePin(value.source_policy) ||
    !houseLivePin(value.qualification) || !Array.isArray(value.entries) || value.entries.length !== 16) return null;
  const ids = new Set<string>(), configurations = new Set<string>();
  for (const row of value.entries) {
    if (!object(row) || !id(row.id) || ids.has(row.id) || !text(row.label) ||
      !houseLivePin(row.result)) return null;
    const configuration = parseHouseLiveConfiguration(row.configuration);
    if (!configuration) return null;
    const key = houseLiveConfigurationKey(configuration);
    if (!expected.has(key) || configurations.has(key)) return null;
    ids.add(row.id);
    configurations.add(key);
    if (row.id === value.default_id && key !== "fundamentals|standard|standard") return null;
  }
  if (configurations.size !== expected.size || !ids.has(value.default_id)) return null;
  return value as unknown as HouseLiveCatalog;
}
