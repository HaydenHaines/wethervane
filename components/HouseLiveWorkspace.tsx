"use client";

import { useEffect, useRef, useState } from "react";
import type { HouseLivePublication } from "../lib/house-live-publication";
import { loadHouseLive, type HouseLiveRequest } from "../lib/house-live-client";
import { parseHouseLiveNavigation, houseLiveSelectionUrl, type HouseLiveNavigation } from "../lib/house-live-navigation";
import { compareHouseLiveViews, type HouseLiveComparison } from "../lib/house-live-comparison";
import { competitiveOrder } from "../lib/presentation";
import HouseLiveResults from "./HouseLiveResults";
import HouseLiveComparisonPanel from "./HouseLiveComparison";

type Request = HouseLiveRequest & { selection: HouseLiveNavigation; navigation: "initial" | "push" | "pop"; attempt: number };
type ComparisonState = { value: HouseLiveComparison | null; busy: boolean; error: string };

/** Activated by the chamber router only after a reviewed live publication is installed. */
export default function HouseLiveWorkspace({ onHistorical, onReturn }: { onHistorical: () => void; onReturn: () => void }) {
  const [publication, setPublication] = useState<HouseLivePublication | null>(null);
  const displayed = useRef<HouseLivePublication | null>(null);
  const attempt = useRef(0);
  const [request, setRequest] = useState<Request | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [builderExpanded, setBuilderExpanded] = useState(false);
  const [raceId, setRaceId] = useState("");
  const [comparisonId, setComparisonId] = useState<string | null>(null);
  const [comparisonAttempt, setComparisonAttempt] = useState(0);
  const [comparison, setComparison] = useState<ComparisonState>({ value: null, busy: false, error: "" });

  useEffect(() => {
    function restore(navigation: "initial" | "pop") {
      if (new URLSearchParams(location.search).get("chamber") !== "house") return;
      const parsed = parseHouseLiveNavigation(location.search);
      if (parsed.status === "invalid") {
        ++attempt.current;
        setRequest(null); setBusy(false); setError(parsed.reason); return;
      }
      setRequest({ runId: parsed.selection.runId, selection: parsed.selection, navigation, attempt: ++attempt.current });
    }
    restore("initial");
    const history = () => restore("pop");
    window.addEventListener("popstate", history);
    return () => window.removeEventListener("popstate", history);
  }, []);

  useEffect(() => {
    if (!request || request.attempt !== attempt.current) return;
    const controller = new AbortController();
    const isCurrent = () => !controller.signal.aborted && request.attempt === attempt.current;
    setBusy(true); setError("");
    async function resolve() {
      const current = displayed.current;
      const value = request!.navigation === "pop" && current?.view.run.id === request!.runId
        ? current : await loadHouseLive(request!, controller.signal);
      if (!isCurrent()) return;
      const selection = request!.selection;
      if (selection.raceId && !value.view.races.some(race => race.id === selection.raceId))
        throw Error("The district in this link is not part of the selected current House forecast.");
      if (selection.comparisonId && (selection.comparisonId === value.view.run.id ||
        !value.catalog.entries.some(entry => entry.id === selection.comparisonId)))
        throw Error("Choose a different model from this House collection for comparison.");
      const selectedRace = selection.raceId ?? [...value.view.races].sort(competitiveOrder)[0].id;
      const url = houseLiveSelectionUrl(value, { raceId: selectedRace, comparisonId: selection.comparisonId });
      if (request!.navigation === "push") history.pushState(null, "", url);
      else history.replaceState(null, "", url);
      displayed.current = value;
      setPublication(value); setRaceId(selectedRace); setComparisonId(selection.comparisonId); setBusy(false);
    }
    resolve().catch(reason => {
      if (isCurrent()) { setError(reason instanceof Error ? reason.message : "The current House forecast could not be loaded."); setBusy(false); }
    });
    return () => controller.abort();
  }, [request]);

  useEffect(() => {
    const controller = new AbortController();
    if (!publication || !comparisonId) {
      setComparison({ value: null, busy: false, error: "" });
      return () => controller.abort();
    }
    const entry = publication.catalog.entries.find(entry => entry.id === comparisonId);
    if (!entry || entry.id === publication.view.run.id) {
      setComparison({ value: null, busy: false, error: "Choose a different published House model to compare." });
      return () => controller.abort();
    }
    setComparison(previous => ({
      value: previous.value?.status === "comparable" &&
        previous.value.summary.primary.resultSha256 === publication.view.run.resultSha256 ? previous.value : null,
      busy: true, error: "",
    }));
    loadHouseLive({ runId: entry.id, expectedResultSha256: entry.resultSha256 }, controller.signal)
      .then(reference => {
        if (!controller.signal.aborted) setComparison({ value: compareHouseLiveViews(publication.view, reference.view), busy: false, error: "" });
      }).catch(reason => {
        if (!controller.signal.aborted) setComparison(previous => ({ ...previous, busy: false,
          error: reason instanceof Error ? reason.message : "The comparison could not be loaded." }));
      });
    return () => controller.abort();
  }, [publication, comparisonId, comparisonAttempt]);

  const loadDefault = () => setRequest({ runId: null, selection: { runId: null, raceId: null, comparisonId: null }, navigation: "push", attempt: ++attempt.current });
  const retry = () => request ? setRequest({ ...request, attempt: ++attempt.current }) : loadDefault();
  function chooseModel(id: string) {
    const entry = publication?.catalog.entries.find(entry => entry.id === id);
    if (!entry || busy) return;
    setRequest({ runId: id, expectedResultSha256: entry.resultSha256,
      selection: { runId: id, raceId, comparisonId: comparisonId === id ? null : comparisonId },
      navigation: "push", attempt: ++attempt.current });
  }
  function selectRace(id: string) {
    if (!publication) return;
    history.replaceState(null, "", houseLiveSelectionUrl(publication, { raceId: id, comparisonId }));
    setRaceId(id);
  }
  function selectComparison(id: string | null) {
    if (!publication || busy) return;
    history.pushState(null, "", houseLiveSelectionUrl(publication, { raceId, comparisonId: id }));
    setComparisonId(id);
  }

  if (!publication) return <main className="load-page" aria-busy={busy}>
    {error ? <div className="error-box"><h1>Current House forecast unavailable</h1><p role="alert">{error}</p>
      <button onClick={retry}>Try again</button><button onClick={loadDefault}>Load current default</button>
      <button onClick={onHistorical}>Explore historical House research</button><button onClick={onReturn}>Explore the Senate</button>
    </div> : <p role="status">Loading the current House forecast…</p>}
  </main>;
  return <HouseLiveResults view={publication.view} catalog={publication.catalog} geography={publication.geography}
    publicationLimitations={publication.publicationReview?.limitations}
    builderExpanded={builderExpanded} onBuilderExpandedChange={setBuilderExpanded} busy={busy} error={error}
    onSelect={chooseModel} onRetry={retry} onHistorical={onHistorical} selectedRaceId={raceId} onRaceSelect={selectRace}
    comparisonTools={<HouseLiveComparisonPanel comparison={comparison.value} selectedRaceId={raceId}
      catalog={publication.catalog} selectedRunId={publication.view.run.id} comparisonId={comparisonId}
      busy={busy || comparison.busy} error={comparison.error} onSelect={selectComparison}
      onRetry={() => setComparisonAttempt(value => value + 1)} />} />;
}
