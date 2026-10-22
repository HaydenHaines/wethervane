"use client";

import { useEffect, useRef, useState } from "react";
import type { HouseGroupPublication } from "../lib/house-group-catalog";
import { loadHouseGroup } from "../lib/house-group-client";
import { verifyHouseGroupComparison } from "../lib/house-group-comparison";
import { groupSelectionUrl } from "../lib/house-group-selection";

export default function useHouseGroupComparison(publication: HouseGroupPublication, seed?: HouseGroupPublication | null) {
  const [, refresh] = useState(0);
  const requested = useRef<{ runId: string; id: string | null }>({ runId: publication.result.id, id: null });
  const locationParams = typeof window === "undefined" ? null : new URLSearchParams(location.search);
  if (locationParams?.get("run") === publication.result.id) requested.current = { runId: publication.result.id, id: locationParams.get("compare") || null };
  const requestedId = requested.current.runId === publication.result.id ? requested.current.id : null;
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState<{ primarySha: string; requestedId: string; reference: HouseGroupPublication } | null>(null);
  const [failure, setFailure] = useState<{ primarySha: string; requestedId: string; message: string } | null>(null);
  useEffect(() => {
    const restore = () => refresh(n => n + 1);
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => {
    setLoaded(null); setFailure(null);
    if (!requestedId) return;
    const controller = new AbortController();
    const binding = { primarySha: publication.entry.viewSha256, requestedId };
    async function resolve() {
      const entry = publication.catalog.runs.find(run => run.id === requestedId);
      if (!entry || entry.cycle !== publication.result.cycle || entry.id === publication.result.id)
        throw Error("Choose a different group forecast for the same election year.");
      const reference = seed?.result.id === requestedId ? seed : await loadHouseGroup(requestedId!, controller.signal);
      const checked = await verifyHouseGroupComparison(publication, reference);
      if (checked.status !== "comparable") throw Error(checked.reason || "These forecasts cannot be compared.");
      if (!controller.signal.aborted) setLoaded({ ...binding, reference });
    }
    resolve().catch(reason => { if (!controller.signal.aborted) setFailure({ ...binding, message: reason instanceof Error ? reason.message : "The comparison could not be verified." }); });
    return () => controller.abort();
  }, [publication, requestedId, attempt, seed]);
  const matches = (value: { primarySha: string; requestedId: string } | null) => value?.primarySha === publication.entry.viewSha256 && value.requestedId === requestedId;
  const ready = requestedId && matches(loaded) ? loaded!.reference : null;
  const error = requestedId && matches(failure) ? failure!.message : "";
  function choose(id: string) {
    const params = new URLSearchParams(location.search);
    history.pushState(null, "", groupSelectionUrl(publication.result, params.get("race"), params.get("view"), id));
    refresh(n => n + 1);
  }
  return { requestedId, ready, error, busy: !!requestedId && !ready && !error,
    choices: publication.catalog.runs.filter(run => run.cycle === publication.result.cycle && run.id !== publication.result.id),
    choose, retry: () => setAttempt(n => n + 1) };
}
