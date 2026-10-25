import { validHouseRunId } from "./house-catalog.ts";
import { groupRunId as publishedGroupRunId } from "./house-reviewed-catalog.ts";
import { houseLiveRunId } from "./house-live-navigation.ts";

const HISTORICAL_YEARS = [2018, 2020, 2022, 2024] as const;
type HistoricalYear = typeof HISTORICAL_YEARS[number];

/**
 * Supported House URL selections:
 * - Current: `/?chamber=house&edition=current&run=<live-id>&race=<2026-race-id>&compare=<live-id>`
 * - Historical: `/?chamber=house&edition=historical&year=<2018|2020|2022|2024>&run=<saved-id>&view=<forecast|actual>`
 * - Legacy historical links without `edition` stay historical when their run/year identifies an old edition.
 * Bare `?chamber=house` requires an availability response before choosing an edition.
 */
export type HouseAvailabilityInput =
  | { state: "pending" }
  | { state: "error" }
  | { state: "response"; body: unknown };

export interface HouseEditionSelection {
  runId: string | null;
  year: 2026 | HistoricalYear | null;
  view: "forecast" | "actual" | null;
  raceId: string | null;
  comparisonId: string | null;
}
export type HouseEditionResolution =
  | { status: "selected"; edition: "current" | "historical"; selection: HouseEditionSelection }
  | { status: "pending" | "error" | "not_house"; reason: string }
  | { status: "invalid"; reason: string };

function historicalYearForRun(id: string): HistoricalYear | null {
  if (publishedGroupRunId(id)) {
    const match = /-(2018|2020|2022|2024)-h14-full435-(?:polls-)?v1$/.exec(id);
    return match ? Number(match[1]) as HistoricalYear : null;
  }
  const legacyId: unknown = id;
  if (!validHouseRunId(legacyId)) return null;
  for (const year of HISTORICAL_YEARS)
    if ((legacyId as string).endsWith(`-${year}-h14`)) return year;
  return null;
}

function validHistoricalRun(id: string): boolean {
  return publishedGroupRunId(id) || historicalYearForRun(id) !== null;
}

function availabilityChoice(input: HouseAvailabilityInput | undefined):
  | { state: "available" | "unavailable" }
  | { state: "pending" | "error"; reason: string } {
  if (!input || input.state === "pending")
    return { state: "pending", reason: "House edition availability is still being checked." };
  if (input.state === "error")
    return { state: "error", reason: "House edition availability could not be checked." };
  if (input.state !== "response" || !input.body || typeof input.body !== "object" || Array.isArray(input.body))
    return { state: "error", reason: "The House availability response is invalid." };
  const body = input.body as Record<string, unknown>;
  if (body.schema !== "HouseLiveAvailabilityV1" || typeof body.available !== "boolean" ||
    Object.keys(body).length !== 2 || !Object.hasOwn(body, "available"))
    return { state: "error", reason: "The House availability response is invalid." };
  return { state: body.available ? "available" : "unavailable" };
}

/** Resolve URL edition intent before any current or historical data load. A 503 must be passed as state:error. */
export function resolveHouseEdition(search: string, availability?: HouseAvailabilityInput): HouseEditionResolution {
  const params = new URLSearchParams(search);
  const invalid = (reason: string): HouseEditionResolution => ({ status: "invalid", reason });
  for (const name of ["chamber", "edition", "run", "race", "compare", "view", "year"])
    if (params.getAll(name).length > 1) return invalid(`This link repeats the ${name} selection.`);

  const chamber = params.get("chamber");
  const edition = params.get("edition");
  const runId = params.get("run");
  const raceId = params.get("race");
  const comparisonId = params.get("compare");
  const view = params.get("view");
  const yearText = params.get("year");
  if (chamber !== null && chamber !== "house") return invalid("This link selects a different chamber.");
  if (edition !== null && edition !== "current" && edition !== "historical") return invalid("This House edition is not recognized.");
  if (view !== null && view !== "forecast" && view !== "actual") return invalid("This House view is not recognized.");
  const year = yearText === null ? null : Number(yearText);
  if (yearText !== null && !/^(?:2018|2020|2022|2024|2026)$/.test(yearText))
    return invalid("This House year is not supported.");

  const hasHouseIntent = edition !== null || yearText !== null || runId !== null || raceId !== null || comparisonId !== null || view !== null;
  if (chamber === null && !hasHouseIntent) return { status: "not_house", reason: "This link does not select the House." };
  if (chamber === null && hasHouseIntent) return invalid("House edition selections need chamber=house.");

  if (runId !== null && !houseLiveRunId(runId) && !validHouseRunId(runId)) return invalid("This link has an invalid House model ID.");
  if (comparisonId !== null && !houseLiveRunId(comparisonId) && !validHouseRunId(comparisonId)) return invalid("This link has an invalid House comparison ID.");
  if (raceId !== null && !/^(?:2018|2020|2022|2024|2026):[A-Z]{2}-\d{2}:house:regular$/.test(raceId))
    return invalid("This link has an invalid House district selection.");
  if (runId && comparisonId === runId) return invalid("Choose two different House models to compare.");

  const inferredHistoricalYear = runId ? historicalYearForRun(runId) : null;
  const explicitHistoricalYear = year !== null && HISTORICAL_YEARS.includes(year as HistoricalYear) ? year as HistoricalYear : null;
  const explicitCurrent = edition === "current" || year === 2026;
  const explicitHistorical = edition === "historical" || explicitHistoricalYear !== null || inferredHistoricalYear !== null || view === "actual";
  if (explicitCurrent && explicitHistorical) return invalid("This link mixes current and historical House selections.");
  if (explicitHistoricalYear !== null && inferredHistoricalYear !== null && explicitHistoricalYear !== inferredHistoricalYear)
    return invalid("The year in this House link conflicts with its saved run.");
  if (edition === "historical" && year === 2026) return invalid("A historical House edition cannot select 2026.");
  if (edition === "current" && view === "actual") return invalid("Observed winners are not available for the current 2026 House forecast.");
  if (edition === "current" && runId !== null && validHistoricalRun(runId)) return invalid("A historical saved run cannot select the current House edition.");

  if (explicitCurrent) {
    if (raceId !== null && !raceId.startsWith("2026:")) return invalid("A historical district cannot select the current House edition.");
    if (comparisonId !== null && validHistoricalRun(comparisonId)) return invalid("A historical run cannot be compared as a current House model.");
    return { status: "selected", edition: "current", selection: {
      runId, year: 2026, view: view as "forecast" | null, raceId, comparisonId,
    } };
  }
  if (explicitHistorical) {
    if (runId !== null && !validHistoricalRun(runId)) return invalid("This saved House run does not identify a supported historical edition.");
    const chosenYear = explicitHistoricalYear ?? inferredHistoricalYear;
    if (comparisonId !== null && !validHistoricalRun(comparisonId)) return invalid("A current or unknown run cannot be compared as a historical House model.");
    const comparedYear = comparisonId === null ? null : historicalYearForRun(comparisonId);
    if (chosenYear !== null && comparedYear !== null && comparedYear !== chosenYear)
      return invalid("The comparison run belongs to a different historical House year.");
    if (raceId !== null && raceId.startsWith("2026:")) return invalid("A current district cannot select a historical House edition.");
    if (raceId !== null && chosenYear !== null && !raceId.startsWith(`${chosenYear}:`))
      return invalid("This district belongs to a different historical House year.");
    return { status: "selected", edition: "historical", selection: {
      runId, year: chosenYear, view: view as "forecast" | "actual" | null, raceId, comparisonId,
    } };
  }

  if (runId !== null || raceId !== null || comparisonId !== null || view !== null)
    return invalid("This House link does not identify a current or historical edition.");
  const choice = availabilityChoice(availability);
  if (choice.state === "pending" || choice.state === "error") return { status: choice.state, reason: choice.reason };
  if (choice.state === "available") return { status: "selected", edition: "current",
    selection: { runId: null, year: 2026, view: null, raceId: null, comparisonId: null } };
  return { status: "selected", edition: "historical",
    selection: { runId: null, year: null, view: null, raceId: null, comparisonId: null } };
}
