"use client";

import { useEffect, useRef, useState } from "react";
import type { ComponentStudyCatalog, ComponentStudyEntry, ComponentStudyRequest, ComponentStudyResolution, ComponentStudyResult } from "../lib/component-study";
import { studySelectionsMatch, validateComponentStudyResult } from "../lib/component-study";
import { comparableStudies, componentStudySlices, parseStudySavedViews, studyRaces, studyShareError, studySourceCopy, type StudySavedView } from "../lib/component-study-view";
import { deriveRaceScoreboard } from "../lib/scoreboard";
import { COLORS, STATE_NAMES, competitiveOrder } from "../lib/presentation";
import ModelSliceControls from "./ModelSliceControls";
import SenateMap, { type SenateGeography } from "./SenateMap";
import RaceList from "./RaceList";
import RaceProbabilityRibbon from "./RaceProbabilityRibbon";
import StudyRaceDetail from "./StudyRaceDetail";

const savedKey = "ballotmancy.component-study-views.v1";
const shortLabel = (id: string) => id === "ordinary-2018-f" ? "Without polls" : "With polls";
const cache = new Map<string, Promise<ComponentStudyResult>>();
async function loadStudy(entry: ComponentStudyEntry): Promise<ComponentStudyResult> {
  const key = `${entry.id}:${entry.sha256}`;
  if (cache.has(key)) return cache.get(key)!;
  const promise = (async () => {
    const response = await fetch(`/api/component-studies?result=${encodeURIComponent(entry.id)}`);
    if (!response.ok) throw Error("This study result could not be loaded. The previous result remains displayed.");
    const bytes = await response.arrayBuffer();
    if (globalThis.crypto?.subtle) {
      const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (b) => b.toString(16).padStart(2, "0")).join("");
      if (hash !== entry.sha256) throw Error("The result differs from its saved fingerprint. Reload to obtain matching files.");
    }
    const value: unknown = JSON.parse(new TextDecoder().decode(bytes));
    if (!validateComponentStudyResult(value) || value.id !== entry.id || value.configurationSha256 !== entry.configurationSha256 || value.artifactId !== entry.artifactId)
      throw Error("This study result is inconsistent and cannot be displayed.");
    return value;
  })();
  cache.set(key, promise);
  promise.catch(() => cache.delete(key));
  return promise;
}

export default function ComponentStudy({ initialId, onReturn }: { initialId: string; onReturn: () => void }) {
  const [catalog, setCatalog] = useState<ComponentStudyCatalog | null>(null);
  const [result, setResult] = useState<ComponentStudyResult | null>(null);
  const [reference, setReference] = useState<ComponentStudyResult | null>(null);
  const [geo, setGeo] = useState<SenateGeography | null>(null);
  const [request, setRequest] = useState<ComponentStudyRequest | null>(null);
  const [resolution, setResolution] = useState<ComponentStudyResolution | null>(null);
  const [raceId, setRaceId] = useState("");
  const [compare, setCompare] = useState(false);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [comparisonError, setComparisonError] = useState("");
  const [notice, setNotice] = useState("");
  const [shareUrl, setShareUrl] = useState("");
  const [views, setViews] = useState<StudySavedView[]>([]);
  const [viewName, setViewName] = useState("");
  const ticket = useRef(0);
  const current = useRef<ComponentStudyResult | null>(null);
  const state = useRef({ raceId: "", compare: false });

  function write(id: string, race: string, comparison: boolean, mode: "push" | "replace") {
    const query = new URLSearchParams({ study: id, race });
    if (comparison) query.set("compare", "1");
    const url = `/?${query}`;
    if (location.pathname + location.search !== url) history[mode === "push" ? "pushState" : "replaceState"](null, "", url);
    setRaceId(race); setCompare(comparison); state.current = { raceId: race, compare: comparison }; setShareUrl("");
  }
  async function open(nextCatalog: ComponentStudyCatalog, id: string, mode: "push" | "replace", desired?: { raceId: string; compare: boolean }) {
    const entry = nextCatalog.studies.find((study) => study.id === id);
    const token = ++ticket.current;
    setError(""); setBusy(true);
    if (!entry) { setError("This study link does not identify an available result. Return to the full Senate forecast or choose a saved study."); setBusy(false); return; }
    setRequest({ studyId: id, selections: entry.selections }); setResolution(null);
    try {
      const next = await loadStudy(entry);
      if (token !== ticket.current) return;
      const wanted = desired || state.current;
      const selected = next.races.find((race) => race.id === wanted.raceId)?.id || [...studyRaces(next)].sort(competitiveOrder)[0].id;
      current.current = next; setResult(next);
      write(id, selected, wanted.compare, mode);
    } catch (e) {
      if (token === ticket.current) setError(e instanceof Error ? e.message : "The study could not be loaded.");
    } finally { if (token === ticket.current) setBusy(false); }
  }
  useEffect(() => {
    let active = true;
    fetch("/api/component-studies").then((r) => { if (!r.ok) throw Error("The study catalog is unavailable."); return r.json(); })
      .then((value: ComponentStudyCatalog) => {
        if (!active) return;
        if (value.schemaVersion !== 1 || value.studies?.length !== 2 || !value.catalog?.roles?.length) throw Error("The study catalog is inconsistent.");
        setCatalog(value);
        try { setViews(parseStudySavedViews(localStorage.getItem(savedKey), value)); } catch { /* Storage is optional. */ }
        const query = new URLSearchParams(location.search);
        void open(value, query.get("study") || initialId, "replace", { raceId: query.get("race") || "", compare: query.get("compare") === "1" });
      }).catch((e) => { if (active) { setError(e.message); setBusy(false); } });
    fetch("/states-us.geojson").then((r) => r.ok ? r.json() : null).then((value) => { if (active) setGeo(value); }).catch(() => {});
    return () => { active = false; ticket.current++; };
    // The popstate handler restores subsequent study links without resetting the catalog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!catalog) return;
    const restore = () => {
      const query = new URLSearchParams(location.search);
      if (query.get("chamber") === "house" || !query.has("study")) { ticket.current++; return; }
      void open(catalog, query.get("study")!, "replace", { raceId: query.get("race") || "", compare: query.get("compare") === "1" });
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [catalog]);
  useEffect(() => {
    let active = true;
    setReference(null); setComparisonError("");
    if (compare && result && catalog) {
      const other = catalog.studies.find((entry) => entry.id !== result.id)!;
      loadStudy(other).then((value) => {
        if (!active) return;
        if (!comparableStudies(result, value)) throw Error("The studies do not describe matching races and outcomes.");
        setReference(value);
      }).catch((e) => { if (active) setComparisonError(e.message); });
    }
    return () => { active = false; };
  }, [compare, result, catalog]);

  async function change(instance: string, value: string) {
    if (!request || !catalog || busy) return;
    const provider = catalog.catalog.roles.flatMap((role) => role.providers).find((p) => p.default_instance_id === instance)!;
    const option = Object.keys(provider.options_schema.properties)[0];
    const next: ComponentStudyRequest = {
      studyId: instance === "poll" ? value === "identity" ? "ordinary-2018-f" : "ordinary-2018-fp" : request.studyId,
      selections: request.selections.map((s) => s.instance_id === instance ? { ...s, options: { ...s.options, [option]: value } } : s),
    };
    const token = ++ticket.current;
    setRequest(next); setResolution(null); setBusy(true); setError(""); setNotice("");
    try {
      const response = await fetch("/api/component-studies", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next) });
      if (!response.ok) throw Error("These choices could not be checked. The previous result remains displayed.");
      const resolved = await response.json() as ComponentStudyResolution;
      if (token !== ticket.current) return;
      setResolution(resolved);
      if (resolved.state === "ready" && resolved.resultId) await open(catalog, resolved.resultId, "push");
    } catch (e) { if (token === ticket.current) setError(e instanceof Error ? e.message : "The configuration check failed."); }
    finally { if (token === ticket.current) setBusy(false); }
  }
  function selectRace(id: string) {
    if (!result || busy) return;
    write(result.id, id, compare, "push");
    if (window.matchMedia("(max-width: 800px)").matches) requestAnimationFrame(() => {
      document.getElementById("race-detail-title")?.focus({ preventScroll: true });
      document.getElementById("race-detail")?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    });
  }
  function persist(next: StudySavedView[]) {
    try { localStorage.setItem(savedKey, JSON.stringify(next)); setViews(next); setNotice("Study view saved in this browser."); }
    catch { setNotice("Browser storage is unavailable. Use Share view to preserve this result."); }
  }
  async function share() {
    setShareUrl(location.href);
    try { await navigator.clipboard.writeText(location.href); setNotice("Link copied for the displayed study result, race and comparison."); }
    catch { setNotice("Copy the shareable link below."); }
  }
  if (!catalog || !result || !request) return <main className="load-page" aria-busy={busy}><div className="error-box">
    <h1>{error ? "Study unavailable" : "Preparing the 2018 study…"}</h1>{error && <p role="alert">{error}</p>}
    <button onClick={onReturn}>Return to the full Senate forecast</button>
  </div></main>;

  const races = studyRaces(result), selected = races.find((race) => race.id === raceId) || races[0];
  const sourceRace = result.races.find((race) => race.id === selected.id)!;
  const score = deriveRaceScoreboard(races), baseline = reference ? deriveRaceScoreboard(studyRaces(reference)) : null;
  const matches = request.studyId === result.id && studySelectionsMatch(catalog.catalog, request.selections, result.selections);
  const shareError = studyShareError(result);
  const candidateOff = request.selections.find((s) => s.instance_id === "candidate")?.options.candidate === "off";
  return <main>
    <section className="hero"><div><div className="eyebrow">2018 United States Senate · {races.length} study races</div>
      <h1>The <span className="chamber-name" data-chamber-outlook="unavailable" style={{ color: COLORS.unresolved }}>Senate</span>,<br />component by component.</h1>
      <p>Explore how individual model ingredients work together, with a historical result for each supported setup.</p></div>
      <div className="edition">THE COMPONENT STUDY<strong>October 23, 2018</strong>14 days before the election</div>
    </section>
    <details className="stack-builder collapsible-stack" aria-labelledby="stack-title" aria-busy={busy}>
      <summary className="stack-builder-summary"><span className="stack-summary-main"><span id="stack-title">Build your perspective</span><span className="stack-summary-displayed">2018 component study · {shortLabel(result.id)} · {races.length} races</span></span><span className="stack-summary-actions">{!matches && <span className="stack-summary-status">Requested result is not displayed</span>}<span className="stack-summary-closed">Expand to customize</span><span className="stack-summary-open">Collapse controls</span><span className="stack-summary-arrow" aria-hidden="true">⌄</span></span></summary>
      <div className="stack-builder-body">
        <div className="study-entry"><div><strong>2018 component study</strong><p className="small">This study covers regular Senate contests with Democratic and Republican candidates. It does not cover the whole chamber.</p></div><button onClick={onReturn}>Return to the full Senate forecast</button></div>
        <div className="stack-builder-top"><div className="composition-presets"><span>Saved study results</span><div role="group" aria-label="Saved study results">
          {catalog.studies.map((entry) => <button key={entry.id} disabled={busy} aria-pressed={matches && result.id === entry.id} onClick={() => { setNotice(""); void open(catalog, entry.id, "push"); }}>{shortLabel(entry.id)}</button>)}
        </div></div><div className="stack-actions">
          <label className="compare-toggle"><input type="checkbox" checked={compare} disabled={busy} onChange={(e) => write(result.id, raceId, e.target.checked, "push")} />Compare with the other study result</label>
          <button disabled={busy} onClick={() => void open(catalog, result.id, "replace")}>Reset to displayed result</button>
          <button onClick={() => void share()}>Share view ↗</button>
        </div></div>
        <div className={`composition-state${!matches ? " unresolved" : ""}`} role="status">
          <strong>{busy ? "Checking requested configuration…" : resolution?.state === "requires_computation" ? "Requires a new calculation" : resolution?.state === "incompatible" ? "Combination unavailable on this site" : matches && !error ? "Saved result ready" : "Requested result is not displayed"}</strong>
          <p>{matches ? "Displaying" : "Still displaying"}: <strong>{result.label}</strong>.</p>
          {!matches && <p>Requested: {shortLabel(request.studyId)}{candidateOff ? ", without the incumbency adjustment" : " with modified component choices"}.</p>}
          {resolution?.state === "requires_computation" && <p>No saved result exists for this setup. It needs a new model fit: estimating its settings together and checking new predictions. This site cannot run that calculation. No calculation has been scheduled.</p>}
          {resolution && resolution.state !== "ready" && <details><summary>Why this result is unavailable</summary><p>{resolution.reason}</p></details>}
          {notice && <p>{notice}</p>}{error && <p role="alert">{error}</p>}
        </div>
        <ModelSliceControls slices={componentStudySlices(catalog, request)} busy={busy} explain={(_slot, _id, fallback) => fallback} onChange={(id, value) => void change(id, value)} />
        <details className="composition-saved"><summary>Saved study views <span>({views.length}) · This browser</span></summary>
          <div className="save-view-form"><label htmlFor="study-view-name">Name this study view</label><div><input id="study-view-name" value={viewName} maxLength={80} onChange={(e) => setViewName(e.target.value)} /><button disabled={!matches || busy || !!error || !viewName.trim()} onClick={() => persist([...views.slice(-29), { name: viewName.trim(), studyId: result.id, sha256: catalog.studies.find((s) => s.id === result.id)!.sha256, raceId, compare }])}>Save view</button></div></div>
          {!matches && <p className="small">Only the displayed, completed configuration can be saved. The requested setup is unresolved.</p>}
          <ul className="saved-view-list">{views.map((view, index) => <li key={`${view.name}-${index}`}><button className="saved-view-restore" aria-label={`Restore ${view.name}`} disabled={busy} onClick={() => void open(catalog, view.studyId, "push", { raceId: view.raceId, compare: view.compare })}>{view.name}<span>{shortLabel(view.studyId)}</span></button><button className="saved-view-delete" aria-label={`Delete ${view.name}`} onClick={() => persist(views.filter((_, i) => i !== index))}>Delete</button></li>)}</ul>
        </details>
      </div>
    </details>
    {shareUrl && <div className="share-link"><label htmlFor="study-share">Shareable displayed result</label><input id="study-share" readOnly value={shareUrl} onFocus={(e) => e.target.select()} /></div>}
    <div className="study-scope"><strong>{races.length} regular Democratic–Republican races · Retrospective study</strong><p>These are reconstructed results, using some information assembled after the election. They do not establish what a forecast published in 2018 would have achieved. Chamber control, tipping points and predicted vote counts are unavailable.</p></div>
    <section className="historical-scoreboard study-scoreboard" aria-labelledby="study-score-title"><div className="scoreboard-heading"><div><div className="eyebrow">Displayed result · {shortLabel(result.id)}</div><h2 id="study-score-title" className="section-title">Historical study scorecard</h2><p>Scored on {score.coverage.scored} of {races.length} study races, against Democratic–Republican results.</p></div></div>
      <div className="scoreboard-cards">
        <div className="scoreboard-card"><h3>Two-party favorites right</h3><div className="scoreboard-value">{score.favorites?.correct}<span> / {score.favorites?.decisive}</span></div><p className="scoreboard-caption">The party given the higher chance received more of the two parties' votes.</p>{!!score.favorites?.tiedRaceIds.length && <p>{score.favorites.tiedRaceIds.length} tied forecasts excluded.</p>}</div>
        <div className="scoreboard-card"><h3>Probability error</h3><div className="scoreboard-value">{score.brier?.mean.toFixed(3)}</div><p className="scoreboard-caption">Lower is better. This score penalizes confident mistakes more than close calls.</p>{baseline?.brier && score.brier && <p className="scoreboard-comparison">{(score.brier.mean - baseline.brier.mean).toFixed(3)} versus {shortLabel(reference!.id).toLowerCase()}.</p>}</div>
        <div className="scoreboard-card"><h3>Average vote-share miss</h3><div className="scoreboard-value">{(100 * shareError).toFixed(1)}<span> pp</span></div><p className="scoreboard-caption">Average distance between the predicted Democratic two-party share and the observed share. One percentage point (pp) is the difference between two percentages.</p>{reference && <p className="scoreboard-comparison">{(100 * (shareError - studyShareError(reference))).toFixed(1)} pp versus {shortLabel(reference.id).toLowerCase()}.</p>}</div>
      </div>
      {compare && <p className="scoreboard-context" role="status">{comparisonError ? `Comparison unavailable: ${comparisonError}` : reference ? `Compared with ${shortLabel(reference.id).toLowerCase()}, on the same races, two-party outcomes and forecast date. Differences below zero mean less error.` : "Loading matching study comparison…"}</p>}
      <details className="scoreboard-raw"><summary>How these scores are calculated</summary><p>Probability error is the categorical Brier score: add the squared error for the Democratic and Republican probabilities in each race, then average across races. It ranges from 0 to 2; 0 is perfect. A 50–50 forecast for every race scores 0.5.</p><p>The vote-share measure averages absolute errors, so overestimates and underestimates do not cancel out. These are descriptive scores for this reconstruction, not an independent validation or a basis for changing the default forecast.</p></details>
      {!!score.favorites?.missRaceIds.length && <details className="scoreboard-misses"><summary>Races where the favorite fell short ({score.favorites.missRaceIds.length})</summary><div>{score.favorites.missRaceIds.map((id) => <button key={id} onClick={() => selectRace(id)}>{STATE_NAMES[races.find((race) => race.id === id)!.state]}</button>)}</div></details>}
    </section>
    <RaceProbabilityRibbon races={races} selectedId={selected.id} onSelect={selectRace} />
    <section className="workspace" aria-label="Senate component study explorer" aria-busy={busy}>
      <div className="workspace-main"><div className="map-card"><div className="map-head"><div><h2>The study's races</h2><p className="small">Select a state. Gray states are outside this study.</p></div></div>
        {geo ? <SenateMap geo={geo} races={races} selected={selected} onSelect={(state) => selectRace(races.find((race) => race.state === state)!.id)} absentLabel="not included in this study" /> : <p className="empty">Map unavailable. Explore the races in the list below.</p>}
        <p className="small study-map-key">Safe: over 90% chance · Likely: over 60% through 90% · Tossup: 40–60% Democratic chance. Blue indicates a Democratic favorite; red indicates a Republican favorite; purple indicates a tossup.</p>
      </div><RaceList races={races} selectedId={selected.id} onSelect={selectRace} baseline={reference ? studyRaces(reference) : undefined} /></div>
      <StudyRaceDetail result={result} race={sourceRace} />
    </section>
    <section id="methodology" className="methodology study-methodology"><h2>What this study can tell us</h2>
      <p>Each saved setup combines seven contributions within three model slices. Changing a contribution can require refitting the whole model. Empty slices are visible so you can see which parts of the eventual model are still being developed.</p>
      <p><strong>Experimental · Calculation checks passed · Forecast accuracy not independently established.</strong> A completed calculation and a historical score do not establish predictive reliability. These results preserve shared uncertainty across races, but do not include all the races and retained seats needed for a Senate-control forecast.</p>
      <h3>Why these polls?</h3><p>The polling setup uses the historical poll records admitted by this study's saved input package. Historical availability uses a date-based approximation; original publication dates and later revisions have not been verified. It should not be read as a complete collection of polls known on October 23, 2018. The no-poll setup excludes all polling evidence.</p>
      <details><summary>Exact configuration and source records</summary><p>These identifiers record the precise choices and saved output. File fingerprints check whether an artifact has changed; they are not evidence of forecast skill.</p><dl className="configuration-display"><dt>Configuration</dt><dd><code>{result.configurationSha256}</code></dd><dt>Saved output</dt><dd><code>{result.artifactId}</code></dd><dt>Input cutoff</dt><dd>{result.informationCutoff}</dd></dl>
        {result.sourceAttribution.map(studySourceCopy).map((source) => <p key={source.label}><strong>{source.label}:</strong> {source.description} {source.urls.map((url) => <a key={url} href={url} target="_blank" rel="noreferrer">Source documentation ↗</a>)}</p>)}
      </details>
    </section>
  </main>;
}
