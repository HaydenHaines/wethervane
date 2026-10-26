"use client";

import { useEffect, useState } from "react";
import type { ChamberId } from "../lib/chamber";
import { parseElectionNightEvaluationView, type ElectionNightEvaluationView } from "../lib/election-night-evaluation-view";

export interface ElectionNightEvaluationState {
  chamber: ChamberId;
  view: ElectionNightEvaluationView | null;
  loading: boolean;
  error: boolean;
}

/** The evaluation head is independent of the current forecast selection. */
export function useElectionNightEvaluation(chamber: ChamberId): ElectionNightEvaluationState {
  const [state, setState] = useState<ElectionNightEvaluationState>({ chamber, view: null, loading: true, error: false });
  useEffect(() => {
    let active: AbortController | null = null;
    let disposed = false;
    setState({ chamber, view: null, loading: true, error: false });
    const refresh = async () => {
      if (disposed || active) return;
      const controller = new AbortController(); active = controller;
      const timer = window.setTimeout(() => controller.abort(), 8_000);
      try {
        const response = await fetch(`/api/election-night/evaluation?chamber=${chamber}`,
          { signal: controller.signal, cache: "no-store" });
        const value = parseElectionNightEvaluationView(await response.json(), chamber);
        if (controller.signal.aborted) throw Error("Evaluation request timed out.");
        if (!value || (!response.ok && value.status !== "unavailable")) throw Error("Invalid evaluation response.");
        if (!disposed && !controller.signal.aborted) setState({ chamber, view: value, loading: false, error: false });
      } catch {
        if (!disposed) setState({ chamber, view: null, loading: false, error: true });
      } finally {
        window.clearTimeout(timer);
        if (active === controller) active = null;
      }
    };
    void refresh();
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void refresh(); }, 60_000);
    const onVisible = () => { if (document.visibilityState === "visible") void refresh(); };
    const onFocus = () => { void refresh(); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onFocus);
    return () => { disposed = true; active?.abort(); window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible); window.removeEventListener("focus", onFocus); };
  }, [chamber]);
  return state.chamber === chamber ? state : { chamber, view: null, loading: true, error: false };
}
