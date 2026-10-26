import type {
  ExplorerBundle,
  ExplorerRegistry,
  ExplorerRun,
  ExplorerRunEntry,
} from "./explorer-types";

export interface ExplorerSelection {
  runId: string;
  scenarioId: string | null;
  raceId: string | null;
  compare: boolean;
  baselineRunId?: string | null;
  mapView: "forecast" | "actual";
}

export interface SelectionResult {
  selection: ExplorerSelection;
  notice: string | null;
}

export interface RunChoice {
  run: ExplorerRunEntry;
  notice: string | null;
}

export function availableRuns(
  registry: ExplorerRegistry,
  year: number,
  dateId?: string,
): ExplorerRunEntry[] {
  return registry.runs.filter(
    (run) => run.year === year && (!dateId || run.dateId === dateId),
  );
}

export function defaultRun(
  registry: ExplorerRegistry,
  year: number = 2026,
): ExplorerRunEntry {
  const run = registry.runs.find((item) => item.id === registry.defaults[String(year)]);
  if (!run) throw new Error(`No default forecast is available for ${year}.`);
  return run;
}

export function parseSelection(
  search: string | URLSearchParams,
  registry: ExplorerRegistry,
): SelectionResult {
  const params = typeof search === "string" ? new URLSearchParams(search) : search;
  const requested = params.get("run");
  const run = registry.runs.find((item) => item.id === requested) ?? defaultRun(registry);
  return {
    selection: {
      runId: run.id,
      scenarioId: params.get("scenario") || null,
      raceId: params.get("race") || null,
      compare: params.get("compare") === "1",
      ...(params.get("against") ? { baselineRunId: registry.runs.find(r => r.id === params.get("against") && r.year === run.year && r.dateId === run.dateId && r.targetId === run.targetId && r.lawId === run.lawId)?.id || null } : {}),
      mapView: params.get("view") === "actual" && run.historical ? "actual" : "forecast",
    },
    notice: requested && requested !== run.id
      ? "That saved configuration is unavailable. Showing the default 2026 forecast."
      : null,
  };
}

export function serializeSelection(selection: ExplorerSelection): string {
  const params = new URLSearchParams({ run: selection.runId });
  if (selection.scenarioId) params.set("scenario", selection.scenarioId);
  if (selection.raceId) params.set("race", selection.raceId);
  if (selection.compare) params.set("compare", "1");
  if (selection.baselineRunId) params.set("against", selection.baselineRunId);
  if (selection.mapView === "actual") params.set("view", "actual");
  return `?${params.toString()}`;
}

/** Component intent may select another current edition; explicit dated requests use the exact resolver instead. */
export function selectRunForComponents(
  registry: ExplorerRegistry,
  currentRun: ExplorerRun,
  patch: Partial<ExplorerRun["components"]>,
): RunChoice {
  const desired = { ...currentRun.components, ...patch };
  const dated = availableRuns(registry, currentRun.year, currentRun.dateId);
  const fields = ["base", "polling", "uncertainty"] as const;
  const matches = (run: ExplorerRunEntry) => fields.every((field) => run.components[field] === desired[field]);
  const exact = dated.find(matches);
  if (exact) return { run: exact, notice: null };

  const candidates = availableRuns(registry, currentRun.year);
  const retained = candidates.find((run) => run.id === currentRun.id);
  if (!retained) throw new Error("The selected forecast is no longer in the catalog.");
  if (currentRun.year === 2026 && !currentRun.historical) {
    const otherEditions = candidates.filter(run => !run.historical && matches(run));
    if (otherEditions.length === 1) {
      const run = otherEditions[0];
      return { run, notice: `These components use the saved ${run.dateLabel || run.dateId} edition (${run.dateId}), rather than ${currentRun.dateLabel || currentRun.dateId}. Its original input cutoff is preserved.` };
    }
    if (otherEditions.length > 1) return { run: retained,
      notice: "These components have more than one saved edition. Choose a forecast date; your displayed stack is unchanged." };
  }
  return {
    run: retained,
    notice: "That exact combination has no saved forecast. Your displayed stack is unchanged.",
  };
}

/** A date choice is constrained to that date, even when it changes the available components. */
export function selectRunForDate(registry: ExplorerRegistry, currentRun: ExplorerRun, dateId: string): RunChoice {
  const candidates = availableRuns(registry, currentRun.year, dateId).sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
  const fields = ["base", "polling", "uncertainty"] as const;
  const run = candidates.find(r => fields.every(field => r.components[field] === currentRun.components[field])) ||
    candidates.find(r => r.components.base === currentRun.components.base && r.components.uncertainty === currentRun.components.uncertainty) ||
    candidates.find(r => r.components.base === currentRun.components.base) ||
    candidates.find(r => r.id === registry.defaults[String(currentRun.year)]) || candidates[0];
  if (run) return { run, notice: `Showing a saved forecast from the requested ${run.dateLabel || dateId} edition (${dateId}).` };
  const retained = registry.runs.find(r => r.id === currentRun.id);
  if (!retained) throw new Error("The selected forecast is no longer in the catalog.");
  return { run: retained, notice: "No saved forecast is available for the requested date. Your displayed stack is unchanged." };
}

export function selectRunForYear(
  registry: ExplorerRegistry,
  currentRun: ExplorerRun,
  year: number,
): RunChoice {
  const candidates = availableRuns(registry, year);
  const preferred = defaultRun(registry, year);
  const matching = candidates.find((run) =>
    run.components.base === currentRun.components.base &&
    run.components.polling === currentRun.components.polling &&
    run.components.uncertainty === preferred.components.uncertainty,
  );
  const run = matching ?? preferred;
  return {
    run,
    notice: year === currentRun.year ? null
      : run.components.base === currentRun.components.base &&
        run.components.polling === currentRun.components.polling
        ? `Showing ${year} with its standard uncertainty setting.`
        : `Showing the available default stack for ${year}.`,
  };
}

/** Structural/probability checks complement the downloaded file's SHA256. */
export function validateExplorerBundle(bundle: ExplorerBundle, entry: ExplorerRunEntry): boolean {
  try {
    const probability = (p: number) => Number.isFinite(p) && p >= 0 && p <= 1;
    const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
    if (
      bundle.schemaVersion !== 1 ||
      bundle.run.id !== entry.id ||
      bundle.run.year !== entry.year ||
      bundle.run.lawId !== entry.lawId ||
      bundle.run.dateId !== entry.dateId ||
      bundle.run.targetId !== entry.targetId ||
      bundle.run.components.base !== entry.components.base ||
      bundle.run.components.polling !== entry.components.polling ||
      bundle.run.components.uncertainty !== entry.components.uncertainty ||
      !bundle.races.length ||
      new Set(bundle.races.map((race) => race.id)).size !== bundle.races.length ||
      !bundle.scenarios.some((scenario) => scenario.id === bundle.defaultScenarioId)
    ) return false;
    for (const race of bundle.races) {
      if (
        !race.outcomes.length ||
        new Set(race.outcomes.map((outcome) => outcome.id)).size !== race.outcomes.length ||
        !race.outcomes.every((outcome) => probability(outcome.winProbability)) ||
        Math.abs(sum(race.outcomes.map((outcome) => outcome.winProbability)) - 1) > 1e-8 ||
        (race.observed && !race.outcomes.some((outcome) => outcome.id === race.observed?.outcomeId))
      ) return false;
      for (const outcome of race.outcomes) {
        if (!outcome.members.length || !["candidate", "group"].includes(outcome.kind)) return false;
        if (outcome.share) {
          const { mean, quantiles: q } = outcome.share;
          if (![mean, q.p05, q.p50, q.p95].every(probability) || q.p05 > q.p50 || q.p50 > q.p95) return false;
        }
      }
    }
    for (const scenario of bundle.scenarios) {
      const control = scenario.control;
      const values = [control.democraticMajorityProbability, control.republicanMajorityProbability, control.unresolvedOrganizingProbability];
      if (!values.every(probability) || Math.abs(sum(values) - 1) > 1e-8) return false;
      for (const party of ["D", "R", "U"] as const) {
        const distribution = scenario.seats[party];
        if (!distribution.pmf.every(probability) || Math.abs(sum(distribution.pmf) - 1) > 1e-8) return false;
        if (Math.abs(sum(distribution.pmf.map((p, seats) => p * seats)) - distribution.expectedSeats) > 1e-8) return false;
      }
      if (Math.abs(sum(Object.values(scenario.seats).map((seats) => seats.expectedSeats)) - 100) > 1e-8) return false;
      for (const party of ["D", "R"] as const) {
        const p = sum(scenario.seats[party].pmf.slice(scenario.thresholds[party]));
        const expected = party === "D" ? control.democraticMajorityProbability : control.republicanMajorityProbability;
        if (Math.abs(p - expected) > 1e-8) return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}
