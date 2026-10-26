import type { HouseLivePublication } from "./house-live-publication.ts";

export interface HouseLiveNavigation {
  runId: string | null;
  raceId: string | null;
  comparisonId: string | null;
}
export type HouseLiveNavigationResult =
  | { status: "valid"; selection: HouseLiveNavigation }
  | { status: "invalid"; reason: string };

export function houseLiveRunId(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9:._-]{0,199}$/.test(value);
}

/** Live links select model choices, not user-saved configurations or historical scores. */
export function parseHouseLiveNavigation(search: string): HouseLiveNavigationResult {
  const params = new URLSearchParams(search);
  const invalid = (reason: string): HouseLiveNavigationResult => ({ status: "invalid", reason });
  for (const name of ["chamber", "edition", "run", "race", "compare", "view", "year"])
    if (params.getAll(name).length > 1) return invalid(`This link repeats the ${name} selection.`);
  if (params.has("chamber") && params.get("chamber") !== "house")
    return invalid("This link selects a different chamber.");
  if (params.has("edition") && params.get("edition") !== "current")
    return invalid("This link does not select the current House edition.");
  if (params.has("year") && params.get("year") !== "2026")
    return invalid("This current House view forecasts the 2026 election.");
  if (params.has("view") && params.get("view") !== "forecast")
    return invalid("Actual 2026 winners are not available in this forecast.");
  const runId = params.get("run"), comparisonId = params.get("compare"), raceId = params.get("race");
  if ((runId !== null && !houseLiveRunId(runId)) || (comparisonId !== null && !houseLiveRunId(comparisonId)))
    return invalid("This link has an invalid House model selection.");
  if (raceId !== null && !/^2026:[A-Z]{2}-\d{2}:house:regular$/.test(raceId))
    return invalid("This link has an invalid current House district selection.");
  if (runId && comparisonId === runId)
    return invalid("Choose two different House models to compare.");
  return { status: "valid", selection: { runId, raceId, comparisonId } };
}

/** Called after loading the exact selected model and checking its catalog. */
export function houseLiveSelectionUrl(publication: HouseLivePublication,
  selection: { raceId?: string | null; comparisonId?: string | null } = {}): string {
  const { view, catalog } = publication;
  const params = new URLSearchParams({ chamber: "house", edition: "current", run: view.run.id });
  if (selection.raceId) {
    if (!view.races.some(race => race.id === selection.raceId))
      throw Error("This district is not present in the selected House forecast.");
    params.set("race", selection.raceId);
  }
  if (selection.comparisonId) {
    if (selection.comparisonId === view.run.id || !catalog.entries.some(entry => entry.id === selection.comparisonId))
      throw Error("This House comparison model is not available in the displayed collection.");
    params.set("compare", selection.comparisonId);
  }
  return `/?${params}`;
}
