import type { HouseCatalog } from "./house-catalog.ts";
import { houseRunForYear } from "./house-selection.ts";

const HISTORICAL_YEARS = new Set([2018, 2020, 2022, 2024]);

/** Select the requested historical year using the catalog's default model family. */
export function houseHistoricalRunForYear(catalog: HouseCatalog, year: number): string | null {
  if (!HISTORICAL_YEARS.has(year)) return null;
  return houseRunForYear(catalog, catalog.defaultRunId, year);
}
