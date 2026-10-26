import type { ChamberId } from "./chamber";
import type { ExplorerBundle, ExplorerRegistry, ExplorerRunEntry } from "./explorer-types";
import { validateExplorerBundle } from "./explorer-state.ts";
import { loadHouseLive } from "./house-live-client.ts";
import { toHouseElectionNightForecast, toSenateElectionNightForecast, type ElectionNightForecast } from "./election-night-forecast.ts";
import { matchElectionNightReference, type ElectionNightReference } from "./election-night-reference.ts";
import { matchElectionNightCoreReference, supportsElectionNightCoreReference, type ElectionNightCoreReference } from "./election-night-core-reference.ts";

export interface ElectionNightChoice { id: string; label: string; resultSha256: string }
export interface ElectionNightLoaded {
  forecast: ElectionNightForecast;
  defaultRunId: string;
  choices: ElectionNightChoice[];
  reference: ElectionNightReference | null;
  referenceError: string | null;
  primaryReference: ElectionNightCoreReference | null;
  primaryReferenceError: string | null;
}
export interface ElectionNightReferencePublication {
  schemaVersion: number; href: string; sha256: string;
  editions: { chamber: string; runId: string; resultSha256: string }[];
}
export interface ElectionNightRequest {
  chamber: ChamberId;
  runId: string | null;
  expectedResultSha256?: string;
}

/** Independent read-only selection. Never rewrites the visitor's forecast URL/settings. */
export async function loadElectionNightForecast(request: ElectionNightRequest, signal: AbortSignal,
  publication?: ElectionNightReferencePublication, primaryPublication?: ElectionNightReferencePublication): Promise<ElectionNightLoaded> {
  if (request.expectedResultSha256 && (!request.runId || !/^[a-f0-9]{64}$/.test(request.expectedResultSha256)))
    throw Error("The selected scorecard forecast reference is invalid.");
  if (request.chamber === "house") {
    const house = await loadHouseLive(request, signal);
    return attachReferences({ forecast: toHouseElectionNightForecast(house.view), defaultRunId: house.catalog.defaultId,
      choices: house.catalog.entries.map(entry => ({ id: entry.id, label: entry.label, resultSha256: entry.resultSha256 })) }, signal, publication, primaryPublication);
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
  return attachReferences({ forecast: toSenateElectionNightForecast(bundle, entry), defaultRunId,
    choices: entries.map(run => ({ id: run.id, label: run.label, resultSha256: run.sha256 })) }, signal, publication, primaryPublication);
}

/** The build pins reference bytes; a missing or invalid optional reference never replaces the forecast. */
type ForecastOnly = Pick<ElectionNightLoaded, "forecast" | "defaultRunId" | "choices">;
async function attachReferences(value: ForecastOnly, signal: AbortSignal, publication?: ElectionNightReferencePublication,
  primaryPublication?: ElectionNightReferencePublication): Promise<ElectionNightLoaded> {
  const [full, primary] = await Promise.all([attachReference(value, signal, publication), attachPrimaryReference(value.forecast, signal, primaryPublication)]);
  return { ...full, ...primary };
}

async function attachReference(value: ForecastOnly,
  signal: AbortSignal, publication?: ElectionNightReferencePublication): Promise<Omit<ElectionNightLoaded, "primaryReference" | "primaryReferenceError">> {
  if (!publication) return { ...value, reference: null, referenceError: null };
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let forwardAbort: (() => void) | undefined;
  try {
    signal.throwIfAborted();
    if (publication.schemaVersion !== 1 || publication.href !== "/data/election-night/references.json" ||
      !/^[a-f0-9]{64}$/.test(publication.sha256) || !Array.isArray(publication.editions) ||
      publication.editions.some(e => !["senate", "house"].includes(e.chamber) ||
        typeof e.runId !== "string" || !/^[a-zA-Z0-9_-]+$/.test(e.runId) ||
        typeof e.resultSha256 !== "string" || !/^[a-f0-9]{64}$/.test(e.resultSha256)))
      throw Error("Invalid reference publication.");
    const editions = publication.editions.filter(e => e.chamber === value.forecast.chamber &&
      e.runId === value.forecast.run.id && e.resultSha256 === value.forecast.run.artifactSha256);
    if (editions.length === 0) return { ...value, reference: null, referenceError: null };
    if (editions.length !== 1) throw Error("Ambiguous reference publication.");
    const referenceController = new AbortController();
    forwardAbort = () => referenceController.abort(signal.reason);
    signal.addEventListener("abort", forwardAbort, { once: true });
    timeout = setTimeout(() => referenceController.abort(new DOMException("Reference request timed out.", "TimeoutError")), 8000);
    const response = await fetch(publication.href, { signal: referenceController.signal, cache: "no-store" });
    if (!response.ok || !globalThis.crypto?.subtle) throw Error("Reference unavailable.");
    const bytes = await response.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");
    signal.throwIfAborted();
    referenceController.signal.throwIfAborted();
    if (digest !== publication.sha256) throw Error("Reference bytes changed.");
    const reference = matchElectionNightReference(value.forecast, JSON.parse(new TextDecoder().decode(bytes)));
    if (!reference) throw Error("The declared reference does not match this forecast.");
    return { ...value, reference, referenceError: null };
  } catch {
    signal.throwIfAborted();
    return { ...value, reference: null, referenceError: "The expected-error reference could not be verified. The forecast remains available." };
  } finally {
    clearTimeout(timeout);
    if (forwardAbort) signal.removeEventListener("abort", forwardAbort);
  }
}

/** Independent optional fixed32 attachment: a failed35 fetch never borrows or suppresses this reference. */
async function attachPrimaryReference(forecast: ElectionNightForecast, signal: AbortSignal,
  publication?: ElectionNightReferencePublication): Promise<Pick<ElectionNightLoaded, "primaryReference" | "primaryReferenceError">> {
  const absent = { primaryReference: null, primaryReferenceError: null };
  if (!publication || !supportsElectionNightCoreReference(forecast)) return absent;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let forwardAbort: (() => void) | undefined;
  try {
    signal.throwIfAborted();
    if (publication.schemaVersion !== 1 || publication.href !== "/data/election-night/core-references.json" ||
      !/^[a-f0-9]{64}$/.test(publication.sha256) || !Array.isArray(publication.editions) ||
      publication.editions.some(e => e.chamber !== "senate" || typeof e.runId !== "string" || !/^[a-zA-Z0-9_-]+$/.test(e.runId) ||
        typeof e.resultSha256 !== "string" || !/^[a-f0-9]{64}$/.test(e.resultSha256))) throw Error("Invalid primary-reference publication.");
    const editions = publication.editions.filter(e => e.chamber === forecast.chamber && e.runId === forecast.run.id && e.resultSha256 === forecast.run.artifactSha256);
    if (editions.length === 0) return absent;
    if (editions.length !== 1) throw Error("Ambiguous primary-reference publication.");
    const controller = new AbortController();
    forwardAbort = () => controller.abort(signal.reason);
    signal.addEventListener("abort", forwardAbort, { once: true });
    timeout = setTimeout(() => controller.abort(new DOMException("Primary reference request timed out.", "TimeoutError")), 8000);
    const response = await fetch(publication.href, { signal: controller.signal, cache: "no-store" });
    if (!response.ok || !globalThis.crypto?.subtle) throw Error("Primary reference unavailable.");
    const bytes = await response.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2,"0")).join("");
    signal.throwIfAborted(); controller.signal.throwIfAborted();
    if (digest !== publication.sha256) throw Error("Primary reference bytes changed.");
    const primaryReference = matchElectionNightCoreReference(forecast, JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
    if (!primaryReference) throw Error("Primary reference does not match this forecast.");
    return { primaryReference, primaryReferenceError: null };
  } catch {
    signal.throwIfAborted();
    return { primaryReference: null, primaryReferenceError: "The 32-race expected-error reference could not be verified. The forecast remains available." };
  } finally {
    clearTimeout(timeout); if (forwardAbort) signal.removeEventListener("abort", forwardAbort);
  }
}

async function loadSenateBundle(entry: ExplorerRunEntry, signal: AbortSignal): Promise<ExplorerBundle> {
  const response = await fetch(entry.href, { signal, cache: "no-store" });
  if (!response.ok) throw Error("The selected Senate forecast could not be loaded.");
  const bytes = await response.arrayBuffer();
  if (!globalThis.crypto?.subtle) throw Error("A secure connection is required to verify the scorecard forecast.");
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), byte => byte.toString(16).padStart(2, "0")).join("");
  signal.throwIfAborted();
  if (digest !== entry.sha256) throw Error("The forecast file does not match this edition. Reload before trying again.");
  const bundle = JSON.parse(new TextDecoder().decode(bytes)) as ExplorerBundle;
  if (!validateExplorerBundle(bundle, entry)) throw Error("The selected forecast data could not be verified.");
  return bundle;
}
