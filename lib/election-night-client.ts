import type { ChamberId } from "./chamber";
import type { ExplorerBundle, ExplorerRegistry, ExplorerRunEntry } from "./explorer-types";
import { validateExplorerBundle } from "./explorer-state.ts";
import { loadHouseLive } from "./house-live-client.ts";
import { toHouseElectionNightForecast, toSenateElectionNightForecast, type ElectionNightForecast } from "./election-night-forecast.ts";

export interface ElectionNightChoice { id: string; label: string; resultSha256: string }
export interface ElectionNightLoaded {
  forecast: ElectionNightForecast;
  defaultRunId: string;
  choices: ElectionNightChoice[];
}
export interface ElectionNightRequest {
  chamber: ChamberId;
  runId: string | null;
  expectedResultSha256?: string;
}

/** Independent read-only selection. Never rewrites the visitor's forecast URL/settings. */
export async function loadElectionNightForecast(request: ElectionNightRequest, signal: AbortSignal): Promise<ElectionNightLoaded> {
  if (request.expectedResultSha256 && (!request.runId || !/^[a-f0-9]{64}$/.test(request.expectedResultSha256)))
    throw Error("The selected scorecard forecast reference is invalid.");
  if (request.chamber === "house") {
    const publication = await loadHouseLive(request, signal);
    return { forecast: toHouseElectionNightForecast(publication.view), defaultRunId: publication.catalog.defaultId,
      choices: publication.catalog.entries.map(entry => ({ id: entry.id, label: entry.label, resultSha256: entry.resultSha256 })) };
  }
  const response = await fetch("/data/explorer/registry.json", { signal, cache: "no-store" });
  if (!response.ok) throw Error("The published Senate model list could not be loaded.");
  const registry = await response.json() as ExplorerRegistry;
  if (registry?.schemaVersion !== 1 || !registry.defaults || !Array.isArray(registry.runs))
    throw Error("The published Senate model list could not be verified.");
  const entries = registry.runs.filter(entry => entry?.year === 2026);
  if (!entries.length || new Set(entries.map(entry => entry.id)).size !== entries.length ||
    entries.some(entry => typeof entry.id !== "string" || !/^[A-Za-z0-9_-]+$/.test(entry.id) ||
      entry.href !== `/data/explorer/${entry.id}.json` || !/^[a-f0-9]{64}$/.test(entry.sha256) ||
      typeof entry.label !== "string" || !entry.label.trim() || entry.historical !== false))
    throw Error("The current Senate forecast references could not be verified.");
  const defaultRunId = registry.defaults["2026"];
  if (!entries.some(entry => entry.id === defaultRunId)) throw Error("The standard Senate forecast is missing.");
  const entry = entries.find(entry => entry.id === (request.runId ?? defaultRunId));
  if (!entry) throw Error("That model is not available in the current Senate edition.");
  if (request.expectedResultSha256 && request.expectedResultSha256 !== entry.sha256)
    throw Error("The forecast edition changed. Close and reopen the scorecard to load the new edition.");
  const bundle = await loadSenateBundle(entry, signal);
  return { forecast: toSenateElectionNightForecast(bundle, entry), defaultRunId,
    choices: entries.map(run => ({ id: run.id, label: run.label, resultSha256: run.sha256 })) };
}

async function loadSenateBundle(entry: ExplorerRunEntry, signal: AbortSignal): Promise<ExplorerBundle> {
  const response = await fetch(entry.href, { signal, cache: "no-store" });
  if (!response.ok) throw Error("The selected Senate forecast could not be loaded.");
  const bytes = await response.arrayBuffer();
  if (!globalThis.crypto?.subtle) throw Error("A secure connection is required to verify the scorecard forecast.");
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");
  if (digest !== entry.sha256) throw Error("The forecast file does not match this edition. Reload before trying again.");
  const bundle = JSON.parse(new TextDecoder().decode(bytes)) as ExplorerBundle;
  if (!validateExplorerBundle(bundle, entry)) throw Error("The selected forecast data could not be verified.");
  return bundle;
}
