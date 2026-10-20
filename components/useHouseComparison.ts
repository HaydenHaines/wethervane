"use client";

import { useEffect, useState } from "react";
import { parseHousePublicationWithCatalog, validHouseRunId, type HousePublicationWithCatalog } from "../lib/house-catalog";
import { houseComparisonAvailability, summarizeHouseComparison, verifyHouseComparison } from "../lib/house-comparison";
import { HOUSE_COMPARISONS } from "../lib/house-comparison-data";
import { houseSelectionUrl } from "../lib/house-selection";

export type AvailableHousePublication = Extract<HousePublicationWithCatalog, { status: "available" }>;
type Summary = ReturnType<typeof summarizeHouseComparison>;
type Loaded = { primarySha: string; requestedId: string; reference: AvailableHousePublication; summary: Summary };

export async function loadComparisonPublication(id: string, signal: AbortSignal): Promise<AvailableHousePublication> {
  if (!validHouseRunId(id)) throw Error("This comparison selection is not valid.");
  const response = await fetch(`/api/chambers/house?${new URLSearchParams({ run: id })}`, { signal, cache: "no-store" });
  if (!response.ok) throw Error("The comparison result could not be loaded. The selected model is still displayed.");
  const value = parseHousePublicationWithCatalog(await response.json());
  if (!value || value.status !== "available" || value.result.run.id !== id)
    throw Error("The returned comparison does not match the requested saved result.");
  return value;
}

export default function useHouseComparison(publication: AvailableHousePublication, seed?: AvailableHousePublication | null) {
  const [, refreshLocation] = useState(0);
  const requestedId = typeof window === "undefined" ? null : new URLSearchParams(location.search).get("compare") || null;
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [failure, setFailure] = useState<{ primarySha: string; requestedId: string; message: string } | null>(null);

  useEffect(() => {
    const restore = () => refreshLocation(value => value + 1);
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);

  useEffect(() => {
    if (!requestedId) { setLoaded(null); setFailure(null); return; }
    const controller = new AbortController();
    const binding = { primarySha: publication.resultSha256, requestedId };
    setFailure(null);
    setLoaded(null);
    async function resolve() {
      if (!HOUSE_COMPARISONS) throw Error("Reviewed comparison definitions are not available.");
      const availability = houseComparisonAvailability(HOUSE_COMPARISONS, publication.result.run.id, requestedId!);
      if (availability.status !== "eligible") throw Error(availability.reason || "These results cannot be compared.");
      const reference = seed?.result.run.id === requestedId ? seed : await loadComparisonPublication(requestedId!, controller.signal);
      const verified = await verifyHouseComparison(HOUSE_COMPARISONS, publication, reference);
      if (verified.status !== "comparable") throw Error(verified.reason || "These results do not use matching comparison definitions.");
      if (!controller.signal.aborted) setLoaded({ ...binding, reference, summary: summarizeHouseComparison(verified) });
    }
    resolve().catch(error => {
      if (!controller.signal.aborted) setFailure({ ...binding, message: error instanceof Error ? error.message : "The comparison could not be verified." });
    });
    return () => controller.abort();
  }, [publication, requestedId, attempt, seed]);

  const matches = (value: { primarySha: string; requestedId: string } | null) =>
    value?.primarySha === publication.resultSha256 && value.requestedId === requestedId;
  const ready = requestedId && matches(loaded) ? loaded : null;
  const error = requestedId && matches(failure) ? failure!.message : "";
  const busy = !!requestedId && !ready && !error;
  const manifest = HOUSE_COMPARISONS;
  const choices = manifest ? publication.catalog?.runs.filter(run =>
    houseComparisonAvailability(manifest, publication.result.run.id, run.id).status === "eligible") ?? [] : [];
  const requested = publication.catalog?.runs.find(run => run.id === requestedId);
  function choose(id: string) {
    const params = new URLSearchParams(location.search);
    history.pushState(null, "", houseSelectionUrl(publication.result, params.get("race"), params.get("view"), id || null));
    refreshLocation(value => value + 1);
    setLoaded(null);
    setFailure(null);
  }
  function retry() { setFailure(null); setAttempt(value => value + 1); }
  return { requestedId, requested, choices, ready, busy, error, choose, retry };
}

export type HouseComparisonState = ReturnType<typeof useHouseComparison>;
