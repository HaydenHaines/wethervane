"use client";

import { useEffect, useMemo, useRef, useState, type RefObject, type ReactNode } from "react";
import type { DistrictGeography } from "../lib/district-geography";
import type { HouseLiveCatalogView, HouseLiveRaceView, HouseLiveView } from "../lib/house-live-view";
import type { GroupRibbonRace } from "../lib/group-ribbon-layout";
import {
  COLORS,
  STATE_NAMES,
  chamberOutlook,
  competitiveOrder,
  outcomeColor,
  partyName,
  probability,
  raceRating,
} from "../lib/presentation";
import { percent } from "../lib/presentation";
import DistrictMap from "./DistrictMap";
import RaceListToggle from "./RaceListToggle";
import HouseLiveControls from "./HouseLiveControls";
import { houseLiveStatusLabel as statusLabel, houseLiveCutoffRole } from "../lib/house-live-copy";
import HouseGroupRibbon from "./HouseGroupRibbon";
import HouseLivePollingPolicy from "./HouseLivePollingPolicy";
import type { HouseLivePollingRun } from "../lib/house-live-polling";

export interface HouseLiveResultsProps {
  view: HouseLiveView;
  catalog: HouseLiveCatalogView;
  geography: DistrictGeography | null;
  publicationLimitations?: readonly string[];
  polling?: HouseLivePollingRun | null;
  builderExpanded: boolean;
  onBuilderExpandedChange: (expanded: boolean) => void;
  busy?: boolean;
  error?: string;
  onSelect: (runId: string) => void;
  onRetry: () => void;
  onHistorical: () => void;
  selectedRaceId?: string;
  onRaceSelect?: (raceId: string) => void;
  comparisonTools?: ReactNode;
}

function candidatePartyLabel(race: HouseLiveRaceView, outcomeId: string, party: string): string {
  if (party !== "O") return `${partyName(party)} (${party})`;
  const outcome = race.outcomes.find((item) => item.id === outcomeId);
  if (outcome?.kind === "group") return "Other (O)";
  const independent = outcome?.members.map((member) => race.candidates.find((candidate) => candidate.id === member.id)?.partyLabel)
    .find((label) => label && /independent/i.test(label));
  return independent ? `${independent} · Other (O)` : "Other (O)";
}

function HouseLiveRaceDetail({ race, detailRef, pollingEnabled }: { race: HouseLiveRaceView; detailRef: RefObject<HTMLElement | null>; pollingEnabled: boolean }) {
  const rule = ({
    plurality: "Plurality: the candidate with the most votes wins; the forecast reports that candidate’s group.",
    ranked_choice: "Ranked choice: voters rank candidates; the winner’s group is the forecast outcome.",
    majority_runoff: "Majority runoff: a candidate must meet the required majority or a runoff is held; the forecast reports the winner’s group.",
    top_two_plurality: "This general election follows a top-two primary; the forecast reports the winning candidate’s group.",
  } as const)[race.ruleFamily];
  const method = /pair/i.test(race.method)
    ? "A vote-share estimate is conditional on the two named candidates; the group win chances remain separate."
    : /group|categorical/i.test(race.method)
      ? "The forecast estimates which candidate group wins; it does not divide a group’s chance among its members."
      : "The saved district model estimates the winning candidate or group.";
  return <aside ref={detailRef} tabIndex={0} id="house-live-race-detail" className="detail" aria-label="Selected House race detail" aria-live="polite">
    <div className="eyebrow">District in focus · {raceRating(race)}</div>
    <h2 id="house-live-detail-title" tabIndex={-1}>{race.district.districtKey}</h2>
    <p className="small">{STATE_NAMES[race.state] || race.state} · {race.electionLabel}</p>
    <p className="detail-note">{pollingEnabled ? `${race.pollCount} selected local poll-question ${race.pollCount === 1 ? "result" : "results"}` : "Polling is switched off for this model"}. <a href="#house-live-polling">Why these polls?</a></p>
    <p className="detail-note">{rule} {method}</p>
    <h3 className="detail-label">Chance of winning</h3>
    <p className="detail-note">These are chances of each modeled candidate or group winning, not vote shares. A group probability is not allocated among its members.</p>
    {race.outcomes.map((outcome) => <div className="candidate" key={outcome.id}>
      <div className="candidate-top">
        <div>
          <div className="candidate-name">{outcome.name}</div>
          <span className="party-label">{candidatePartyLabel(race, outcome.id, outcome.party)}{outcome.kind === "group" ? " · Group outcome" : ""}</span>
        </div>
        <strong className="candidate-prob" style={{ color: outcomeColor(outcome) }}>{probability(outcome.winProbability)}</strong>
      </div>
      <div className="candidate-track"><span style={{ width: `${outcome.winProbability * 100}%`, background: outcomeColor(outcome) }} /></div>
      {outcome.kind === "group" && <p className="detail-note">Combined probability for {outcome.members.map((member) => member.name).join("; ")}. This probability is not split among the individual candidates.</p>}
      {outcome.members.map((member) => {
        const candidate = race.candidates.find((item) => item.id === member.id);
        return <p className="detail-note" key={member.id}>
          {member.name}{candidate?.partyLabel ? ` · ${candidate.partyLabel}` : ` · ${member.party}`}{member.incumbent ? " · currently holds this seat" : ""}
          {member.sourceUrls.map((url) => <span key={url}> · <a href={url} target="_blank" rel="noreferrer">Source</a></span>)}
        </p>;
      })}
    </div>)}
    {race.conditionalPair && <section aria-label="Conditional two-candidate vote share">
      <h3 className="detail-label">Conditional two-candidate share</h3>
      <p className="detail-note">Pair: {race.conditionalPair.firstCandidateName} and {race.conditionalPair.secondCandidateName}. First candidate average: {percent(race.conditionalPair.mean)}. Denominator: {race.conditionalPair.denominator}.</p>
      <p className="detail-note">Middle 90% of modeled shares: {percent(race.conditionalPair.quantiles.p05)}–{percent(race.conditionalPair.quantiles.p95)}. First candidate exceeds half: {probability(race.conditionalPair.firstExceedsHalf)}.</p>
      <p className="detail-note">This share estimate is conditional on the named pair; it is not the candidate or group win probability above.</p>
      {race.conditionalPair.qualifications.map((note, index) => <p className="detail-note" key={index}>{note}</p>)}
    </section>}
    {race.notes.map((note, index) => <p className="detail-note" key={index}>{note}</p>)}
  </aside>;
}

function asOfExplanation(strict: boolean) {
  return strict
    ? "Input records were checked against their stated cutoff dates; this alone does not show that the same data were available at a past date."
    : "This run does not establish that every input was available by the forecast cutoff.";
}

function HouseLiveChamberControl({ scenario }: { scenario: NonNullable<HouseLiveView["chamber"]> }) {
  const stats = [
    { label: "Democratic majority", probabilityValue: scenario.control.democraticMajorityProbability, color: COLORS.D },
    { label: "Republican majority", probabilityValue: scenario.control.republicanMajorityProbability, color: COLORS.R },
    { label: "Neither party reaches 218", probabilityValue: scenario.control.unresolvedOrganizingProbability, color: COLORS.unresolved },
  ];
  return <section className="control-card" aria-labelledby="house-live-control-title">
    <div className="section-top"><div><h2 id="house-live-control-title" className="section-title">Who could control the chamber?</h2>
      <div className="small">435 seats · 218 for a majority · Other seats remain unassigned</div></div></div>
    <div className="control-probs" aria-live="polite">{stats.map((item) => <div key={item.label} style={{ color: item.color }}>
      <span className="prob-label"><span className="dot" />{item.label}</span><span className="prob-number">{probability(item.probabilityValue)}</span>
    </div>)}</div>
    <div className="prob-track" aria-hidden="true">{stats.map((item) => <span key={item.label} style={{ width: `${item.probabilityValue * 100}%`, background: item.color }} />)}</div>
    <p className="control-note">{scenario.description}</p>
    <p className="control-note">“Neither party reaches 218” is the unresolved outcome. It does not mean a tied House, and this result does not model Speaker negotiations.</p>
    <div className="chamber-seat-summary">{(["D", "R", "U"] as const).map((party) => <span key={party}>
      {party === "U" ? "Unassigned O" : `${party} seats`}: <b>{scenario.seats[party].expectedSeats.toFixed(1)}</b> expected seats{" "}
      <span className="small">({scenario.seats[party].seatQuantiles.p05}–{scenario.seats[party].seatQuantiles.p95}, modeled 90% range)</span>
    </span>)}</div>
    <p className="detail-note">Expected seats are averages across simulated elections. O (Other) seats remain unassigned to either party; outcomes outside the modeled 90% range remain possible.</p>
  </section>;
}

export default function HouseLiveResults({ view, catalog, geography, builderExpanded, onBuilderExpandedChange,
  busy = false, error = "", onSelect, onRetry, onHistorical,
  selectedRaceId, onRaceSelect, comparisonTools, publicationLimitations = [], polling }: HouseLiveResultsProps) {
  const sorted = useMemo(() => [...view.races].sort(competitiveOrder), [view.races]);
  const ribbonRaces = useMemo<GroupRibbonRace[]>(() => view.races.map((race) => ({
    id: race.id,
    districtKey: race.district.districtKey,
    probabilities: {
      D: race.outcomes.find((outcome) => outcome.party === "D")?.winProbability ?? 0,
      R: race.outcomes.find((outcome) => outcome.party === "R")?.winProbability ?? 0,
      O: race.outcomes.find((outcome) => outcome.party === "O")?.winProbability ?? 0,
    },
  })), [view.races]);
  const [selectedId, setSelectedId] = useState(sorted[0]?.id ?? "");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const detail = useRef<HTMLElement>(null);
  const selected = view.races.find((race) => race.id === (selectedRaceId ?? selectedId)) || sorted[0];
  const outlook = chamberOutlook(view.chamber?.control);
  useEffect(() => {
    if (!view.races.some((race) => race.id === selectedId)) setSelectedId(sorted[0]?.id ?? "");
  }, [view.run.id, view.races, selectedId, sorted]);
  useEffect(() => { detail.current?.scrollTo({ top: 0 }); }, [selected?.id]);
  useEffect(() => { setExpanded(false); setQuery(""); }, [view.run.id]);
  function selectRace(id: string, focusDetail = true) {
    setSelectedId(id);
    onRaceSelect?.(id);
    if (focusDetail && window.matchMedia("(max-width: 800px)").matches)
      requestAnimationFrame(() => {
        document.getElementById("house-live-detail-title")?.focus({ preventScroll: true });
        document.getElementById("house-live-race-detail")?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
          block: "start",
        });
      });
  }
  const shown = useMemo(() => view.races.filter((race) =>
    `${race.district.districtKey} ${STATE_NAMES[race.state] || race.state} ${race.candidates.map((candidate) => `${candidate.name} ${candidate.partyLabel || ""}`).join(" ")} ${race.outcomes.map((outcome) => outcome.name).join(" ")}`
      .toLowerCase().includes(query.trim().toLowerCase())).sort(competitiveOrder), [view.races, query]);
  const visible = expanded ? shown : shown.slice(0, 10);

  if (!selected) return <main className="load-page"><p role="status">No verified current House races are available.</p><button onClick={onHistorical}>Explore historical House research</button></main>;
  return <main>
    <section className="hero">
      <div>
        <div className="eyebrow">2026 United States House · {view.races.length} current districts</div>
        <h1>The <span className="chamber-name" data-chamber-outlook={outlook.party} style={{ color: outlook.color }} title={outlook.party === "U" ? "Neither party reaches 218 seats in most simulated elections" : outlook.label}>House</span>,<br />in perspective.</h1>
        <p>{view.run.label}. Current forecast with its inputs, testing status, and limitations shown separately.</p>
        <details className="edition-inputs">
          <summary>Input and forecast times</summary>
          <p>Forecast cutoff: {new Date(view.run.cutoffAt).toUTCString()}</p>
          <p>Result issued: {new Date(view.run.issuedAt).toUTCString()}</p>
          {view.run.sourceCutoffs.map((cutoff, index) => <p key={`${cutoff.role}-${cutoff.at}-${index}`}>{houseLiveCutoffRole(cutoff.role)} inputs through: {new Date(cutoff.at).toUTCString()}</p>)}
          <p>{asOfExplanation(view.run.strictAsOf)}</p>
        </details>
      </div>
      <div className="edition">THE CURRENT EDITION<strong>{view.run.cycle} forecast · 435 seats · 218 for a majority</strong>Forecast cutoff {new Date(view.run.cutoffAt).toUTCString()}</div>
    </section>

    <HouseLiveControls view={view} catalog={catalog} expanded={builderExpanded} onExpandedChange={onBuilderExpandedChange}
      busy={busy} error={error} onSelect={onSelect} onRetry={onRetry} />
    {comparisonTools}

    <HouseGroupRibbon races={ribbonRaces} selectedRaceId={selected.id} onSelect={selectRace} />
    <div className="house-results-layout">
      <div>
        <section className="map-card" aria-label="House map">
          <div className="map-head"><div><h2>District by district</h2><p className="small">Forecast chances for all 435 current districts.</p></div><span className="small">{view.run.cycle} House</span></div>
          {geography ? <DistrictMap races={view.races.map((race) => ({ race, districtKey: race.district.districtKey }))}
            identity={{ cycle: view.run.cycle, mapVintage: view.mapVintage }} geography={geography} selectedRaceId={selected.id} onSelect={selectRace} />
            : <div className="district-map-unavailable"><h3>District map unavailable</h3><p>Current district boundaries have not been verified for display. Explore the races below.</p></div>}
          <p className="small">D and R identify Democratic and Republican modeled outcomes. O identifies an Other group and includes independently labeled candidates where applicable. Safe means above 90%; Likely means above 60% through 90%; Tossup means no outcome exceeds 60%. These are model chances, not guarantees.</p>
          <div className="map-legend">{[
            { label: "Safe D", color: COLORS.D }, { label: "Likely D", color: COLORS.likelyD },
            { label: "Tossup", color: COLORS.competitive }, { label: "Likely R", color: COLORS.likelyR },
            { label: "Safe R", color: COLORS.R }, { label: "Likely O", color: COLORS.likelyO },
            { label: "Safe O", color: COLORS.O },
          ].map((item) => <span key={item.label}><i className="dot" style={{ color: item.color }} />{item.label}</span>)}
          </div>
        </section>
        {view.chamber ? <HouseLiveChamberControl scenario={view.chamber} /> : <section className="control-card house-control-unavailable" aria-label="House control availability">
          <h2 className="section-title">Who could control the chamber?</h2><p>A qualified House control distribution is unavailable for this run.</p><span className="small">435 seats · 218 for a majority</span>
        </section>}
        <section id="races" className="race-section" aria-labelledby="house-live-races-title">
          <div className="section-top"><h2 id="house-live-races-title" className="section-title">The districts</h2><span className="small">{visible.length} of {shown.length} {query.trim() ? "matching races" : "races"}</span></div>
          <div className="race-tools"><input aria-label="Search districts or candidates" placeholder="Find a district, state, or candidate" value={query} onChange={(event) => { setQuery(event.target.value); setExpanded(false); }} /></div>
          <div className="house-race-list" id="house-live-race-list">{visible.map((race) => <button key={race.id} aria-pressed={race.id === selected.id} onClick={() => selectRace(race.id)}>
            <strong>{race.district.districtKey}</strong><span>{race.outcomes.map((outcome) => `${outcome.name} (${outcome.party})`).join(" / ")}</span><b>{raceRating(race)}</b>
          </button>)}</div>
          <RaceListToggle expanded={expanded} total={shown.length} controls="house-live-race-list" onToggle={() => setExpanded(!expanded)} />
          {!shown.length && <p>No matching districts. Try another district, state, or candidate.</p>}
        </section>
      </div>
      <HouseLiveRaceDetail race={selected} detailRef={detail} pollingEnabled={view.run.configuration.polling === "standard"} />
    </div>

    <HouseLivePollingPolicy polling={polling} view={view} selected={selected} onRaceSelect={id => selectRace(id, false)} />

    <section className="methodology" id="methodology" aria-labelledby="live-evidence-title">
      <div><div className="eyebrow">Current run evidence</div><h2 id="live-evidence-title">Inputs, computation, and review.</h2></div>
      <div className="method-content">
        <div><h3>Input status</h3><p>{statusLabel(view.qualification.inputStatus, "input")}. Source cutoffs are listed above.</p></div>
        <div><h3>Numerical checks</h3><p>{statusLabel(view.qualification.numericalStatus, "numerical")}. {view.chamberPrecision ? `${view.jointDraws.count.toLocaleString()} simulated elections across ${view.jointDraws.chainCount} independent sampling runs. The largest estimated standard errors are ${view.chamberPrecision.maxSeatMeanMcse.toFixed(2)} seats for average seats and ${(view.chamberPrecision.maxControlMcse * 100).toFixed(1)} percentage points for control chances. Standard error measures variation from a finite simulation; it is separate from uncertainty about the election.` : "A qualified chamber summary is unavailable for this run."}</p></div>
        <div><h3>Publication review and forecast skill</h3><p>{statusLabel(view.qualification.publicationStatus, "publication")}. Forecast skill: {statusLabel(view.qualification.forecastSkill, "skill")}. Numerical checks do not by themselves establish forecast accuracy.</p></div>
      </div>
      <h3>Model limitations</h3>
      <ul>{view.qualification.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>
      {publicationLimitations.length > 0 && <section aria-labelledby="house-live-publication-review-notes">
        <h3 id="house-live-publication-review-notes">Publication review notes</h3>
        <ul>{publicationLimitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>
      </section>}
    </section>

    <section className="methodology" aria-labelledby="historical-house-title">
      <div><div className="eyebrow">Separate historical research</div><h2 id="historical-house-title">Past elections remain a different edition.</h2></div>
      <div className="method-content"><p>Historical performance and its tested-race coverage are reported separately from this current forecast. This 2026 view has no observed winners or historical scorecard.</p><button className="house-return" onClick={onHistorical}>Explore historical House research <span aria-hidden="true">→</span></button></div>
    </section>
    <footer className="footer"><span className="footer-brand">ballotmancy</span><span>Independent research. Visible assumptions.<br />2026 House current forecast</span></footer>
  </main>;
}
