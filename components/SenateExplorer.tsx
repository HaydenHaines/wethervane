"use client";

import { useEffect, useRef, useState } from "react";
import type {
  ExplorerBundle,
  ExplorerRegistry,
  ExplorerRunEntry,
} from "../lib/explorer-types";
import type { RaceForecast } from "../lib/site-types";
import {
  availableRuns,
  defaultRun,
  parseSelection,
  selectRunForYear,
  serializeSelection,
  validateExplorerBundle,
  type ExplorerSelection,
} from "../lib/explorer-state";
import { COLORS, competitiveOrder, dateLabel } from "../lib/presentation";
import SenateMap, { type SenateGeography } from "./SenateMap";
import TippingRibbon from "./TippingRibbon";
import StackControls from "./StackControls";
import RaceDetail from "./RaceDetail";
import RaceList from "./RaceList";
import ChamberControl from "./ChamberControl";
import PerformancePanel from "./PerformancePanel";
import HistoricalScoreboard from "./HistoricalScoreboard";
import type { CompositionCatalog, CompositionRequest, CompositionResolution, SavedView } from "../lib/composition-types";
import { requestForRun } from "../lib/composition-resolver";
import { createSavedView, parseSavedViews } from "../lib/saved-views";
import { compareScoreboards } from "../lib/scoreboard";
import OutlookRibbon from "./OutlookRibbon";
import { deriveRibbonLayout } from "../lib/ribbon-layout";

const bundleCache = new Map<string, Promise<ExplorerBundle>>();
async function loadBundle(entry: ExplorerRunEntry): Promise<ExplorerBundle> {
  const key = entry.href + ":" + entry.sha256;
  const cached = bundleCache.get(key);
  if (cached) return cached;
  const request = (async () => {
    const response = await fetch(entry.href);
    if (!response.ok) throw Error("The saved stack could not be loaded.");
    const bytes = await response.arrayBuffer();
    if (globalThis.crypto?.subtle && entry.sha256) {
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      )
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      if (hash !== entry.sha256)
        throw Error(
          "The stack fingerprint does not match the catalog. Reload to get a consistent snapshot.",
        );
    }
    const bundle = JSON.parse(
      new TextDecoder().decode(bytes),
    ) as ExplorerBundle;
    if (!validateExplorerBundle(bundle, entry))
      throw Error("This saved stack did not pass its consistency checks.");
    return bundle;
  })();
  bundleCache.set(key, request);
  request.catch(() => bundleCache.delete(key));
  return request;
}

/** Use the margin ribbon only when result data contains principal-pair shares and tipping statistics. */
function principalPair(bundle: ExplorerBundle): RaceForecast[] | null {
  if (!bundle.tipping) return null;
  const result: RaceForecast[] = [];
  for (const race of bundle.races) {
    if (
      race.outcomes.length !== 2 ||
      race.outcomes.some(
        (o) =>
          o.kind !== "candidate" ||
          !o.share ||
          !["D", "R", "I"].includes(o.party),
      )
    )
      return null;
    const [first, second] = race.outcomes;
    result.push({
      id: race.id,
      state: race.state,
      seatId: race.seatId,
      candidates: [first, second].map((o) => ({
        id: o.id,
        name: o.name,
        party: o.party,
        incumbent: o.members[0]?.incumbent || false,
        sourceUrls: o.members[0]?.sourceUrls || [],
      })) as RaceForecast["candidates"],
      firstWinProbability: first.winProbability,
      secondWinProbability: second.winProbability,
      firstShareMean: first.share!.mean,
      shareQuantiles: first.share!.quantiles,
      pollCount: race.pollCount,
      notes: race.notes,
    });
  }
  const ids = new Set(result.map((race) => race.id));
  for (const metadata of Object.values(bundle.tipping.scenarios)) {
    try {
      deriveRibbonLayout(result, metadata);
    } catch {
      return null;
    }
    for (const caucus of [metadata.D, metadata.R]) {
      if (caucus.guaranteedRaceIds.some((id) => !ids.has(id))) return null;
      if (
        Object.entries(caucus.eligibleCandidateByRace).some(
          ([raceId, candidateId]) =>
            !result
              .find((race) => race.id === raceId)
              ?.candidates.some((candidate) => candidate.id === candidateId),
        )
      )
        return null;
      if (
        result.some(
          (race) =>
            !Number.isFinite(caucus.frequencyByRace[race.id]) ||
            caucus.frequencyByRace[race.id] < 0 ||
            caucus.frequencyByRace[race.id] > 1,
        )
      )
        return null;
    }
  }
  return result;
}

function normalize(
  requested: ExplorerSelection,
  bundle: ExplorerBundle,
  previous?: ExplorerBundle,
) {
  const scenario =
    bundle.scenarios.find((s) => s.id === requested.scenarioId) ||
    bundle.scenarios.find((s) => s.id === bundle.defaultScenarioId)!;
  const former = previous?.races.find((r) => r.id === requested.raceId);
  const race =
    bundle.races.find((r) => r.id === requested.raceId) ||
    (former
      ? bundle.races
          .filter((r) => r.state === former.state)
          .sort(competitiveOrder)[0]
      : undefined) ||
    [...bundle.races].sort(competitiveOrder)[0];
  const notes = [];
  if (requested.scenarioId && requested.scenarioId !== scenario.id)
    notes.push(
      "The requested caucus scenario is unavailable; showing the saved default.",
    );
  if (requested.raceId && requested.raceId !== race.id)
    notes.push(
      "The requested election is unavailable; showing an available race.",
    );
  return {
    selection: {
      ...requested,
      runId: bundle.run.id,
      scenarioId: scenario.id,
      raceId: race.id,
      mapView: bundle.run.historical ? requested.mapView : "forecast",
    } as ExplorerSelection,
    notice: notes.join(" "),
  };
}

export default function SenateExplorer() {
  const [registry, setRegistry] = useState<ExplorerRegistry | null>(null),
    [bundle, setBundle] = useState<ExplorerBundle | null>(null),
    [selection, setSelection] = useState<ExplorerSelection | null>(null),
    [geo, setGeo] = useState<SenateGeography | null>(null);
  const [busy, setBusy] = useState(true),
    [error, setError] = useState(""),
    [notice, setNotice] = useState<string | null>(null),
    [baseline, setBaseline] = useState<ExplorerBundle | null>(null),
    [comparisonError, setComparisonError] = useState(""),
    [shareUrl, setShareUrl] = useState("");
  const [capabilities, setCapabilities] = useState<CompositionCatalog | null>(null);
  const [draft, setDraft] = useState<CompositionRequest | null>(null);
  const [resolution, setResolution] = useState<CompositionResolution | null>(null);
  const [savedViews, setSavedViews] = useState<SavedView[]>([]);
  const capabilityRef = useRef<CompositionCatalog | null>(null);
  const requestId = useRef(0),
    bundleRef = useRef<ExplorerBundle | null>(null),
    selectionRef = useRef<ExplorerSelection | null>(null);
  function write(next: ExplorerSelection, mode: "push" | "replace" = "push") {
    setSelection(next);
    setShareUrl("");
    selectionRef.current = next;
    const url = serializeSelection(next);
    if (location.search !== url) {
      if (mode === "replace") history.replaceState(null, "", url);
      else history.pushState(null, "", url);
    }
  }
  async function open(
    catalog: ExplorerRegistry,
    requested: ExplorerSelection,
    mode: "push" | "replace",
    message: string | null = null,
  ) {
    const entry = catalog.runs.find((r) => r.id === requested.runId);
    if (!entry) return;
    const ticket = ++requestId.current;
    setBusy(true);
    setError("");
    setNotice(message);
    try {
      const next = await loadBundle(entry);
      if (ticket !== requestId.current || new URLSearchParams(location.search).get("chamber") === "house") return;
      const resolved = normalize(
        requested,
        next,
        bundleRef.current || undefined,
      );
      const nextDraft = capabilityRef.current ? requestForRun(capabilityRef.current, entry) : null;
      bundleRef.current = next;
      setBundle(next);
      setResolution(null);
      if (nextDraft) setDraft(nextDraft);
      const compared = catalog.runs.find(r => r.id === resolved.selection.baselineRunId);
      if (compared && (compared.year !== next.run.year || compared.dateId !== next.run.dateId || compared.targetId !== next.run.targetId || compared.lawId !== next.run.lawId)) resolved.selection.baselineRunId = null;
      write(resolved.selection, mode);
      setNotice([message, resolved.notice].filter(Boolean).join(" ") || null);
    } catch (e) {
      if (ticket === requestId.current) {
        setError(
          e instanceof Error ? e.message : "This stack could not be loaded.",
        );
        if (bundleRef.current && selectionRef.current)
          history.replaceState(
            null,
            "",
            serializeSelection(selectionRef.current),
          );
      }
    } finally {
      if (ticket === requestId.current) setBusy(false);
    }
  }
  useEffect(() => {
    let active = true;
    fetch("/data/explorer/registry.json")
      .then((r) => {
        if (!r.ok)
          throw Error("The saved forecast catalog could not be loaded.");
        return r.json();
      })
      .then((catalog: ExplorerRegistry) => {
        if (!active) return;
        if (catalog.schemaVersion !== 1 || !catalog.runs?.length)
          throw Error("The forecast catalog is unavailable.");
        setRegistry(catalog);
        const parsed = parseSelection(location.search, catalog);
        void open(catalog, parsed.selection, "replace", parsed.notice);
      })
      .catch((e) => {
        if (active) {
          setError(e.message);
          setBusy(false);
        }
      });
    fetch("/api/configurations/catalog")
      .then(r => { if (!r.ok) throw Error("Component catalog unavailable"); return r.json(); })
      .then((catalog: CompositionCatalog) => {
        if (!active) return;
        if (catalog.schemaVersion !== 1 || !catalog.saved?.length || !catalog.slots?.length) throw Error("Component catalog unavailable");
        capabilityRef.current = catalog;
        setCapabilities(catalog);
        if (bundleRef.current) setDraft({catalogId:catalog.id,year:bundleRef.current.run.year,dateId:bundleRef.current.run.dateId,components:{...bundleRef.current.run.components}});
        try { setSavedViews(parseSavedViews(localStorage.getItem("ballotmancy.saved-views.v1"), catalog)); } catch { /* Browser storage can be disabled. */ }
      })
      .catch(() => { if (active) setNotice("Component catalog unavailable. Reload to enable component selection and saved configurations."); });
    fetch("/states-us.geojson")
      .then((r) => {
        if (!r.ok) throw Error("Map unavailable");
        return r.json();
      })
      .then((data) => {
        if (active) setGeo(data);
      })
      .catch(() => {});
    return () => {
      active = false;
      requestId.current++;
    };
  }, []);
  useEffect(() => {
    if (!registry) return;
    const back = () => {
      if (new URLSearchParams(location.search).get("chamber") === "house") {
        requestId.current++;
        return;
      }
      const parsed = parseSelection(location.search, registry);
      void open(registry, parsed.selection, "replace", parsed.notice);
    };
    window.addEventListener("popstate", back);
    return () => window.removeEventListener("popstate", back);
  }, [registry]);
  useEffect(() => {
    let active = true;
    setComparisonError("");
    if (!registry || !bundle || (!selection?.compare && !bundle.run.historical)) {
      setBaseline(null);
      return;
    }
    setBaseline(null);
    loadBundle(registry.runs.find(r => r.id === selection?.baselineRunId) || defaultRun(registry, bundle.run.year))
      .then((value) => {
        if (active) setBaseline(value);
      })
      .catch((e) => {
        if (active) setComparisonError(e.message);
      });
    return () => {
      active = false;
    };
  }, [registry, bundle, selection?.compare, selection?.baselineRunId]);
  const scenario = bundle?.scenarios.find(
      (s) => s.id === selection?.scenarioId,
    ),
    selected = bundle?.races.find((r) => r.id === selection?.raceId);
  const baselineScenario = baseline?.scenarios.find(s => s.id === scenario?.id);
  const matchedBaseline = bundle && scenario && baseline && baselineScenario &&
    (!bundle.run.historical || compareScoreboards(bundle, scenario, baseline, baselineScenario).status === "matched");
  const reference =
      selection?.compare && matchedBaseline && baseline?.run.year === bundle?.run.year
        ? baseline
        : undefined,
    referenceScenario = reference?.scenarios.find((s) => s.id === scenario?.id),
    pair = bundle ? principalPair(bundle) : null;
  function patch(values: Partial<ExplorerSelection>) {
    if (busy) return;
    if (selectionRef.current) {
      write({ ...selectionRef.current, ...values });
      setNotice(null);
    }
  }
  function selectRace(id: string) {
    if (busy) return;
    patch({ raceId: id });
    if (window.matchMedia("(max-width: 800px)").matches)
      requestAnimationFrame(() => {
        document
          .getElementById("race-detail-title")
          ?.focus({ preventScroll: true });
        document.getElementById("race-detail")?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? "instant"
            : "smooth",
          block: "start",
        });
      });
  }
  function choose(run: ExplorerRunEntry, message: string | null) {
    setResolution(null);
    if (capabilities) setDraft(requestForRun(capabilities, run));
    if (registry && selection)
      void open(
        registry,
        {
          ...selection,
          runId: run.id,
          mapView: run.historical ? selection.mapView : "forecast",
        },
        "push",
        message,
      );
  }
  async function resolveRequest(request: CompositionRequest) {
    if (!registry || !selection) return;
    const ticket = ++requestId.current;
    setDraft(request); setResolution(null); setBusy(true); setError(""); setNotice(null);
    try {
      const response = await fetch("/api/configurations/resolve", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
      if (!response.ok) throw Error("The selected configuration could not be checked. Try again or reset the stack.");
      const result = await response.json() as CompositionResolution;
      if (ticket !== requestId.current) return;
      setResolution(result); setDraft(result.request);
      if (result.state === "ready" && result.run) {
        await open(registry, {...selection,runId:result.run.id}, "push");
      } else setBusy(false);
    } catch (e) {
      if (ticket === requestId.current) {
        setError(e instanceof Error ? e.message : "Configuration check failed."); setBusy(false);
      }
    }
  }
  function components(values: Record<string,string>) {
    if (capabilities && draft) void resolveRequest({...draft,presetId:undefined,components:{...draft.components,...values}});
  }
  function preset(id: string) {
    const entry = capabilities?.presets.find(p => p.id === id);
    const run = registry?.runs.find(r => r.id === entry?.runId);
    if (capabilities && run) void resolveRequest({...requestForRun(capabilities,run),presetId:id});
  }
  const displayedConfiguration = capabilities?.saved.find(c => c.runId === bundle?.run.id) || null;
  function persistViews(views: SavedView[]) {
    try {
      localStorage.setItem("ballotmancy.saved-views.v1",JSON.stringify(views)); setSavedViews(views);
      return true;
    } catch { setNotice("Browser storage is unavailable. Use Share view to keep this configuration."); return false; }
  }
  function saveView(name: string) {
    if (!displayedConfiguration || !selection || busy || error || !name.trim() || !draft ||
      draft.year !== displayedConfiguration.year || draft.dateId !== displayedConfiguration.dateId ||
      Object.entries(displayedConfiguration.components).some(([key,value]) => draft.components[key] !== value)) return;
    if (persistViews([...savedViews.slice(-29), createSavedView(name,displayedConfiguration,selection)])) setNotice("Configuration saved in this browser, with its exact forecast result.");
  }
  function restoreView(id: string) {
    const view = savedViews.find(v => v.id === id);
    if (!view || !registry || !capabilities) return;
    const config = capabilities.saved.find(c => c.id === view.configurationId && c.runId === view.runId && c.resultSha256 === view.resultSha256);
    if (!config) {setNotice("This saved configuration no longer matches an available result.");return;}
    setResolution(null);
    const run = registry.runs.find(r => r.id === view.runId);
    if (run) setDraft(requestForRun(capabilities, run));
    void open(registry,{runId:view.runId,scenarioId:view.scenarioId,raceId:view.raceId,compare:view.compare,mapView:view.mapView,baselineRunId:view.baselineRunId},"push","Restored “"+view.name+"”.");
  }
  async function share() {
    setShareUrl(location.href);
    try {
      await navigator.clipboard.writeText(location.href);
      setNotice(
        "Link copied, including your stack, scenario, race, comparison, and map view.",
      );
    } catch {
      setNotice("Copy the shareable link below.");
    }
  }
  return (
    <>
      {!bundle || !selection || !registry || !scenario || !selected ? (
        <main className="load-page" aria-busy={busy}>
          <div className="error-box">
            {error ? (
              <>
                <h1>Snapshot unavailable</h1>
                <p>{error}</p>
                <button onClick={() => location.reload()}>Try again</button>
              </>
            ) : (
              <p>Preparing the Senate perspective…</p>
            )}
          </div>
        </main>
      ) : (
        <main>
          <section className="hero">
            <div>
              <div className="eyebrow">
                {bundle.run.year} United States Senate · {bundle.races.length}{" "}
                elections
              </div>
              <h1>
                The Senate,
                <br />
                in perspective.
              </h1>
              <p>
                Build a model stack. Explore the races and the chamber together.
                <br className="desktop-break" />
                {bundle.run.historical
                  ? "A historical reconstruction, with observed results and visible limitations."
                  : "A closer look at the races that could shape the chamber, with visible uncertainty."}
              </p>
            </div>
            <div className="edition">
              {bundle.run.historical
                ? "THE HISTORICAL EDITION"
                : "THE RESEARCH EDITION"}
              <strong>{dateLabel(bundle.run.forecastAt)}</strong>
              {bundle.run.historical
                ? String(bundle.run.horizonDays) + " days before the election"
                : "Evidence through " + dateLabel(bundle.run.cutoffAt)}
            </div>
          </section>
          <StackControls
            registry={registry}
            bundle={bundle}
            busy={busy}
            catalog={capabilities}
            request={draft}
            resolution={resolution}
            displayedConfiguration={displayedConfiguration}
            onPreset={preset}
            savedViews={savedViews}
            onSave={saveView}
            onRestore={restoreView}
            onDelete={(id) => {persistViews(savedViews.filter(v => v.id !== id));}}
            baselineRunId={selection.baselineRunId || null}
            onBaseline={(id) => patch({baselineRunId:id})}
            compare={selection.compare}
            notice={notice}
            onYear={(year) => {
              const c = selectRunForYear(registry, bundle.run, year);
              choose(c.run, c.notice);
            }}
            onDate={(id) => {
              const choices = availableRuns(registry, bundle.run.year, id);
              const run =
                choices.find(
                  (r) =>
                    JSON.stringify(r.components) ===
                    JSON.stringify(bundle.run.components),
                ) || choices[0];
              if (run)
                choose(run, "Showing an available saved stack for this date.");
            }}
            onComponents={components}
            onCompare={(compare) => patch({ compare })}
            onShare={() => void share()}
            onReset={() =>
              choose(
                defaultRun(registry, bundle.run.year),
                "Showing this year’s default saved stack.",
              )
            }
          />
          {bundle.run.historical && (
            <HistoricalScoreboard bundle={bundle} scenario={scenario}
              baselineBundle={baseline || undefined} baselineScenario={baselineScenario}
              onSelectRace={selectRace} />
          )}
          {shareUrl && (
            <div className="share-link">
              <label htmlFor="share-url">Shareable view</label>
              <input
                id="share-url"
                readOnly
                value={shareUrl}
                onFocus={(e) => e.target.select()}
              />
            </div>
          )}
          {error && (
            <div className="compare-banner" role="alert">
              <strong>Unable to change stacks</strong>
              <span>{error} The previous stack remains visible.</span>
            </div>
          )}
          {selection.compare && (
            <div className="compare-banner" role="status">
              <strong>Comparing with {selection.baselineRunId ? "a saved configuration" : `the ${bundle.run.year} default`}</strong>
              <span>
                {comparisonError
                  ? "Comparison unavailable: " + comparisonError
                  : reference
                    ? reference.run.label +
                      ". Deltas use the same race, outcome, and caucus scenario."
                    : "Loading the default comparison…"}
              </span>
              {reference && !referenceScenario && (
                <span>
                  No matching default caucus scenario; chamber and performance
                  deltas are unavailable.
                </span>
              )}
            </div>
          )}
          {bundle.historicalNote && (
            <p className="historical-context">{bundle.historicalNote}</p>
          )}
          <div className="ribbon-overview">
            <div className="toolbar">
              <span className="small">
                {bundle.run.label} ·{" "}
                {bundle.run.simulationCount.toLocaleString()} saved simulations
              </span>
              <div className="scenario">
                <label htmlFor="scenario" className="small">
                  Caucus context
                </label>
                <select
                  id="scenario"
                  value={scenario.id}
                  disabled={busy || bundle.scenarios.length === 1}
                  onChange={(e) => patch({ scenarioId: e.target.value })}
                >
                  {bundle.scenarios.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            {pair && bundle.tipping ? (
              <TippingRibbon
                summary={bundle.tipping}
                scenario={scenario.id}
                races={pair}
                selectedId={selected.id}
                onSelect={selectRace}
              />
            ) : (
              <OutlookRibbon
                races={bundle.races}
                scenario={scenario}
                selectedId={selected.id}
                onSelect={selectRace}
              />
            )}
          </div>
          <section
            className="workspace"
            aria-label="Senate race explorer"
            aria-busy={busy}
          >
            <div className="workspace-main">
              <div className="map-card">
                <div className="map-head">
                  <div>
                    <h2>A nation of distinct races</h2>
                    <div className="small">
                      Select a state to explore its election.
                    </div>
                  </div>
                  {bundle.run.historical ? (
                    <div
                      className="segmented"
                      role="group"
                      aria-label="Map view"
                    >
                      <button
                        aria-pressed={selection.mapView === "forecast"}
                        onClick={() => patch({ mapView: "forecast" })}
                      >
                        Forecast
                      </button>
                      <button
                        aria-pressed={selection.mapView === "actual"}
                        onClick={() => patch({ mapView: "actual" })}
                      >
                        Actual winners
                      </button>
                    </div>
                  ) : (
                    <span className="small">{bundle.run.year} Senate</span>
                  )}
                </div>
                {geo ? (
                  <SenateMap
                    geo={geo}
                    races={bundle.races}
                    selected={selected}
                    mapView={selection.mapView}
                    onSelect={(state) => {
                      const choices = bundle.races
                        .filter((r) => r.state === state)
                        .sort(competitiveOrder);
                      if (choices.length) selectRace(choices[0].id);
                    }}
                  />
                ) : (
                  <div className="map-loading">
                    Map unavailable. Explore every race in the list below.
                  </div>
                )}
                <div className="map-legend">
                  {(selection.mapView === "actual"
                    ? [
                        { label: "Democratic winner", color: COLORS.D },
                        { label: "Republican winner", color: COLORS.R },
                        { label: "Independent winner", color: COLORS.I },
                        { label: "Other winner", color: COLORS.O },
                      ]
                    : [
                        { label: "Safe D", color: COLORS.D },
                        { label: "Likely D", color: COLORS.likelyD },
                        { label: "Safe R", color: COLORS.R },
                        { label: "Likely R", color: COLORS.likelyR },
                        { label: "Safe I", color: COLORS.I },
                        { label: "Likely I", color: COLORS.likelyI },
                        ...(bundle.run.historical
                          ? [
                              { label: "Safe other", color: COLORS.O },
                              { label: "Likely other", color: COLORS.likelyO },
                            ]
                          : []),
                        { label: "Tossup", color: COLORS.competitive },
                      ]
                  ).map((x) => (
                    <span key={x.label}>
                      <i className="dot" style={{ color: x.color }} />
                      {x.label}
                    </span>
                  ))}
                </div>
                <div className="map-foot">
                  <span>
                    {selection.mapView === "actual"
                      ? "Observed winner party. For multiple contests, the selected race takes priority."
                      : "Safe >90% · Likely >60–90% · Tossup ≤60% leading outcome win chance (40–60% in two-outcome races)."}
                  </span>
                  <span>Gray: no election · Area ≠ votes</span>
                </div>
              </div>
              <ChamberControl
                scenario={scenario}
                baseline={referenceScenario}
              />
              <PerformancePanel
                bundle={bundle}
                scenario={scenario}
                baseline={referenceScenario}
              />
              <RaceList
                races={bundle.races}
                selectedId={selected.id}
                baseline={reference?.races}
                onSelect={selectRace}
              />
            </div>
            <RaceDetail
              bundle={bundle}
              race={selected}
              baseline={reference?.races.find((r) => r.id === selected.id)}
              onSelect={selectRace}
            />
          </section>
          <section id="methodology" className="methodology">
            <div>
              <div className="eyebrow">A little context goes a long way</div>
              <h2>Read the uncertainty.</h2>
              <p className="small">
                One saved stack. A coherent view of races and chamber control.
              </p>
              <a
                className="download"
                href={registry.runs.find((r) => r.id === bundle.run.id)!.href}
                download={bundle.run.id + ".json"}
              >
                Download this stack ↓
              </a>
            </div>
            <div className="method-content">
              <div>
                <h3>Build from saved evidence</h3>
                <p>
                  Components resolve to an exact saved result, a combination that
                  needs computation, or an explanation of what is unsupported.
                  Some changes require a joint refit. New fits cannot run on this
                  website yet. Only saved forecast dates are selectable.
                  Sensitivity settings describe assumptions rather than new
                  evidence.
                </p>
              </div>
              <div>
                <h3>Different years, different laws</h3>
                <p>
                  The 2026 preview models two principal contenders and
                  conditional shares, excluding third-party wins. Historical
                  views use complete winner laws; some outcomes group multiple
                  candidates of the same party. Group probabilities are not
                  individual probabilities. Historical sources include
                  retrospective proxies and do not establish strict as-of
                  performance.
                </p>
              </div>
              <div>
                <h3>Party and control are different</h3>
                <p>
                  Colors retain party identity. Caucus context sets organizing
                  assumptions, retained seats, and the year’s control
                  thresholds. It updates chamber probabilities and performance
                  together. Unresolved organizing outcomes remain explicit.
                </p>
              </div>
              <div>
                <h3>Compare like with like</h3>
                <p>
                  Configuration comparisons use the same year, matching outcomes, and the
                  same caucus scenario. Changes in win and control probabilities
                  are percentage points. Historical scores do not validate the
                  current model. Missing shares, margins, and tipping statistics
                  are never inferred.
                </p>
              </div>
            </div>
          </section>
          <footer className="footer">
            <span className="footer-brand">ballotmancy</span>
            <span>
              Independent research. Visible assumptions.
              <br />
              {bundle.run.year} Senate research preview ·{" "}
              {dateLabel(bundle.run.forecastAt)}
            </span>
          </footer>
        </main>
      )}
    </>
  );
}
