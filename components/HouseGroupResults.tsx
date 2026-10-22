"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { HouseGroupPublication } from "../lib/house-group-catalog";
import type { HouseGroupSavedView } from "../lib/house-group-saved-views";
import { groupSelectionUrl } from "../lib/house-group-selection";
import { groupRaceForMap, houseFundamentalsLabel } from "../lib/house-group-presentation";
import { competitiveOrder, raceRating, STATE_NAMES } from "../lib/presentation";
import HouseGroupControls from "./HouseGroupControls";
import HouseGroupRibbon from "./HouseGroupRibbon";
import HouseGroupLegend from "./HouseGroupLegend";
import HouseGroupScoreboard from "./HouseGroupScoreboard";
import HouseGroupRaceDetails from "./HouseGroupRaceDetails";
import HouseGroupSavedViews from "./HouseGroupSavedViews";
import useHouseGroupComparison from "./useHouseGroupComparison";
import DistrictMap from "./DistrictMap";
import RaceListToggle from "./RaceListToggle";

export default function HouseGroupResults({ builderExpanded, onBuilderExpandedChange, publication, restoredComparison, busy, error, onChoose, onRetry, onRestore }: {
  builderExpanded: boolean; onBuilderExpandedChange: (expanded: boolean) => void;
  publication: HouseGroupPublication; restoredComparison: HouseGroupPublication | null;
  busy: boolean; error: string; onChoose: (id: string) => void; onRetry: () => void;
  onRestore: (view: HouseGroupSavedView) => void;
}) {
  const { result, entry } = publication;
  const comparison = useHouseGroupComparison(publication, restoredComparison);
  const mapRaces = useMemo(() => result.races.map(groupRaceForMap), [result]);
  const ordered = useMemo(() => [...mapRaces].sort(competitiveOrder), [mapRaces]);
  const mapInput = useMemo(() => mapRaces.map(race => ({ race, districtKey: race.seatId })), [mapRaces]);
  const mapIdentity = useMemo(() => ({ cycle: result.cycle, mapVintage: entry.mapVintage }), [result.cycle, entry.mapVintage]);
  const [selectedId, setSelectedId] = useState(ordered[0].id);
  const [mapView, setMapView] = useState<"forecast" | "actual">("forecast");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const detail = useRef<HTMLElement>(null);
  const selected = result.races.find(race => race.id === selectedId) || result.races.find(race => race.id === ordered[0].id)!;
  const selectedMap = mapRaces.find(race => race.id === selected.id)!;
  const label = houseFundamentalsLabel(entry.fundamentalsId);
  useEffect(() => {
    const restore = () => {
      const params = new URLSearchParams(location.search);
      if (params.get("chamber") !== "house" || params.get("run") !== result.id) return;
      setSelectedId(result.races.find(race => race.id === params.get("race"))?.id || ordered[0].id);
      setMapView(params.get("view") === "actual" ? "actual" : "forecast");
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [result, ordered]);
  useEffect(() => { setExpanded(false); setQuery(""); }, [result.id]);
  useEffect(() => { detail.current?.scrollTo({ top: 0 }); }, [selected.id]);
  function select(id: string, view = mapView, focusDetails = true) {
    history.pushState(null, "", groupSelectionUrl(result, id, view, comparison.requestedId));
    setSelectedId(id); setMapView(view);
    if (focusDetails && window.matchMedia("(max-width: 800px)").matches) requestAnimationFrame(() => {
      document.getElementById("house-detail-title")?.focus({ preventScroll: true });
      detail.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
    });
  }
  const matching = ordered.filter(race => {
    const groupRace = result.races.find(item => item.id === race.id)!;
    return `${race.seatId} ${STATE_NAMES[race.state]} ${groupRace.groups.flatMap(group => group.members.map(member => member.name)).join(" ")}`.toLowerCase().includes(query.trim().toLowerCase());
  });
  const visible = expanded ? matching : matching.slice(0, 10);
  const knownComparison = comparison.choices.some(run => run.id === comparison.requestedId);
  return <main>
    <section className="hero">
      <div>
        <div className="eyebrow">{result.cycle} United States House · All {result.races.length} districts</div>
        <h1>The <span className="chamber-name" data-chamber-outlook="unavailable" title="Chamber forecast unavailable">House</span>,<br />in perspective.</h1>
        <p>{label}. Explore historical winning chances for Democratic, Republican and Other candidate groups in every district.</p>
      </div>
      <div className="edition">THE HISTORICAL EDITION<strong>All-district group forecasts</strong>{result.races.length} of 435 districts · Experimental</div>
    </section>
    <HouseGroupControls selectionKey={`${selected.id}:${mapView}:${comparison.requestedId}`} expanded={builderExpanded} onExpandedChange={onBuilderExpandedChange} publication={publication} busy={busy} error={error} onChoose={onChoose} onRetry={onRetry} onLegacy={() => onChoose(entry.legacyRunId)}>
      <div className="house-comparison-controls stack-field">
        <label htmlFor="house-group-comparison">Compare with</label>
        <select id="house-group-comparison" value={comparison.requestedId || ""} disabled={busy} aria-describedby="house-group-comparison-help" onChange={event => comparison.choose(event.target.value)}>
          <option value="">No comparison</option>
          {comparison.requestedId && !knownComparison && <option value={comparison.requestedId} disabled>Unavailable comparison</option>}
          {comparison.choices.map(run => <option key={run.id} value={run.id}>{houseFundamentalsLabel(run.fundamentalsId)}</option>)}
        </select>
        <p id="house-group-comparison-help">Compare group forecasts for the same election year. The selected model stays in the charts. Changing year or winner outcome set clears the comparison.</p>
      </div>
      <HouseGroupSavedViews publication={publication} raceId={selected.id} mapView={mapView} comparison={comparison.ready} busy={busy || comparison.busy} error={error || comparison.error} onRestore={onRestore} />
    </HouseGroupControls>
    {comparison.busy && <div className="compare-banner" role="status"><strong>Loading group comparison…</strong><span>Still displaying {label} for {result.cycle}.</span></div>}
    {comparison.error && <div className="compare-banner" role="alert"><strong>Comparison unavailable</strong><span>{comparison.error} The selected forecast remains visible.</span><button onClick={comparison.retry}>Retry comparison</button><button onClick={() => comparison.choose("")}>Remove comparison</button></div>}
    {comparison.ready && <section className="house-comparison-note" aria-label="House model comparison"><strong>{label} compared with {houseFundamentalsLabel(comparison.ready.entry.fundamentalsId)} · {result.cycle}</strong><p>Both models predict the same groups in the same districts. Differences describe these saved forecasts; they do not measure how certain one model’s advantage is.</p></section>}
    <HouseGroupScoreboard publication={publication} comparison={comparison.ready} />
    <HouseGroupRibbon races={result.races} selectedRaceId={selected.id} onSelect={select} />
    <div className="house-results-layout">
      <div>
        <section className="map-card" aria-label="House map">
          <div className="map-head"><div><h2>District by district</h2><p className="small">All {result.races.length} districts have group forecasts.{mapView === "actual" && (result.cycle === 2018 ? " NC-09 is gray because its 2018 election had no certified winner." : " Actual winners are recorded separately from the forecasts.")}</p></div>
            <div className="segmented" role="group" aria-label="Map view">{(["forecast", "actual"] as const).map(view => <button key={view} aria-pressed={view === mapView} onClick={() => select(selected.id, view, false)}>{view === "forecast" ? "Forecast" : "Actual winners"}</button>)}</div>
          </div>
          <DistrictMap races={mapInput} identity={mapIdentity} geography={publication.geography} selectedRaceId={selected.id} onSelect={select} mapView={mapView} />
          <p className="small">D means Democratic; R means Republican; Other pools the remaining candidates and write-ins. Safe means the leading group has more than a 90% chance; Likely means more than 60% up to 90%; Tossup means no group exceeds 60%. These are winning chances, not vote shares.</p>
          <HouseGroupLegend mapView={mapView} unavailableLabel={mapView === "actual" && result.cycle === 2018 ? "No certified winner" : undefined} />
        </section>
        <section className="control-card house-control-unavailable" aria-label="House control availability">
          <h2 className="section-title">Who could control the chamber?</h2>
          <p>District forecasts are available, but the simulations have not met the study’s precision requirements for seat totals. House control and a tipping point remain unavailable.</p><span className="small">435 seats · 218 for a majority</span>
        </section>
        <section id="races" className="race-section" aria-labelledby="house-races-title">
          <div className="section-top"><h2 id="house-races-title" className="section-title">The districts</h2><span className="small">{visible.length} of {matching.length} {query.trim() ? "matching races" : "modeled races"}</span></div>
          <div className="race-tools"><input aria-label="Search districts or candidates" placeholder="Find a district, state, or candidate" value={query} onChange={event => { setQuery(event.target.value); setExpanded(false); }} /></div>
          <div className="house-race-list" id="house-race-list">{visible.map(race => <button key={race.id} aria-pressed={race.id === selected.id} onClick={() => select(race.id)}><strong>{race.seatId}</strong><span>{race.outcomes.map(outcome => outcome.name).join(" / ")}</span><b>{raceRating(race)}</b></button>)}</div>
          <RaceListToggle expanded={expanded} total={matching.length} controls="house-race-list" onToggle={() => setExpanded(!expanded)} />
          {!matching.length && <p>No matching districts. Try another district, state, or candidate.</p>}
        </section>
      </div>
      <aside ref={detail} tabIndex={0} id="house-race-detail" className="detail" aria-label="Selected House race detail" aria-live="polite">
        <div className="eyebrow">District in focus · {raceRating(selectedMap)}</div><h2 id="house-detail-title" tabIndex={-1}>{selected.districtKey}</h2><p className="small">{STATE_NAMES[selectedMap.state]} · {selectedMap.electionLabel}</p>
        <HouseGroupRaceDetails race={selected} comparisonRace={comparison.ready?.result.races.find(race => race.id === selected.id)} />
        <details><summary>Original two-party vote-share forecast</summary><p>Vote shares and winning chances answer different questions. The earlier share forecast measures Democratic votes as a fraction of Democratic plus Republican votes, excluding Other.</p>
          {selected.voteShareReferenceAvailable ? <button onClick={() => onChoose(entry.legacyRunId)}>View this district’s original forecast</button> : <p>This district was outside the original two-candidate study, so it has no earlier share forecast.</p>}
        </details>
      </aside>
    </div>
    <section className="methodology" id="methodology">
      <div><div className="eyebrow">Read the evidence</div><h2>Every district, with context.</h2></div>
      <div className="method-content"><div><h3>What these forecasts predict</h3><p>The model estimates the group of the eventual winner. It does not simulate individual ranked-choice transfers or runoff rounds. When a group contains several people, winning that group does not identify which person wins. Other remains pooled even when a source lists a single name.</p><h3>A reconstruction of past elections</h3><p>The nominal forecast date is fourteen days before the election, but some inputs use final candidate lists assembled later. These results are not a replay using only information available on that date. The test elections also informed model development.</p></div>
        <div><h3>What is included?</h3><p>All 435 districts have group forecasts in 2018, 2020, 2022 and 2024. The annulled 2018 NC-09 election has no certified result and is excluded from scoring; its later election is not substituted. The original candidate and vote-share forecasts remain available for their smaller study population.</p><p>These models use election history without polls. Population, turnout and separate polling components are not connected.</p>
          <details><summary>Publication fingerprints</summary><p>These file fingerprints identify the exact saved result and review used here. They check file identity, not forecasting accuracy.</p><p className="house-digest">Result: {entry.resultSha256}<br />Author qualification: {publication.catalog.qualificationSha256}</p><a href={`/api/chambers/house/groups?${new URLSearchParams({ run: result.id })}`}>View this saved result as structured data (JSON)</a></details>
        </div></div>
    </section>
    <footer className="footer"><span className="footer-brand">ballotmancy</span><span>Independent research. Visible assumptions.<br />House research preview</span></footer>
  </main>;
}
