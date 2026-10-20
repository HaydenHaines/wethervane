import type { HouseCatalog } from "./house-catalog.ts";
import type { HouseResult } from "./house-result.ts";

/** Model families carry across years; district identity does not carry across maps. */
export function houseRunForYear(catalog: HouseCatalog, currentRunId: string, year: number): string | null {
  const candidates = catalog.runs.filter((run) => run.available && run.year === year);
  const current = catalog.runs.find((run) => run.id === currentRunId);
  const reference = catalog.runs.find((run) => run.id === catalog.defaultRunId);
  return candidates.find((run) => run.configurationId === current?.configurationId)?.id
    ?? candidates.find((run) => run.configurationId === reference?.configurationId)?.id
    ?? candidates[0]?.id ?? null;
}

export function houseSelectionUrl(result: HouseResult, raceId?: string | null, view?: string | null): string {
  const params = new URLSearchParams({ chamber: "house", run: result.run.id });
  if (raceId && result.races.some((race) => race.id === raceId)) params.set("race", raceId);
  if (result.run.historical && view === "actual") params.set("view", "actual");
  return `/?${params}`;
}

export function houseSelectionForResult(result: HouseResult, previous: HouseResult | null, search: string): string {
  const params = new URLSearchParams(search);
  const sameMap = !previous || (previous.run.year === result.run.year && previous.metadata.mapVintage === result.metadata.mapVintage);
  return houseSelectionUrl(result, sameMap ? params.get("race") : null, result.run.historical ? params.get("view") : null);
}
