"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChamberId } from "../lib/chamber";
import { loadElectionNightForecast, type ElectionNightLoaded, type ElectionNightRequest } from "../lib/election-night-client";
import type { ElectionNightInputNotice, ElectionNightPair, ElectionNightRace } from "../lib/election-night-forecast";
import referencePublication from "../data/election-night-reference-publication.json";
import { probability, STATE_NAMES } from "../lib/presentation";

const signed = (value: number) => `${Math.abs(value) < .05 ? "" : value > 0 ? "+" : "−"}${Math.abs(value).toFixed(1)}`;
const date = (value: string) => new Date(value).toUTCString();
const label = (race: ElectionNightRace) => race.seatId.startsWith("US-SEN")
  ? `${STATE_NAMES[race.state] || race.state} · ${race.electionLabel}` : `${race.seatId} · ${STATE_NAMES[race.state] || race.state}`;
function pointPrediction(pair: ElectionNightPair): string {
  if (Math.abs(pair.meanMarginPp) < .05) return "Average margin rounds to 0.0 points";
  return `${pair.meanMarginPp > 0 ? pair.firstCandidateName : pair.secondCandidateName} leads by ${Math.abs(pair.meanMarginPp).toFixed(1)} points`;
}

function MarginRange({ pair, extent }: { pair: ElectionNightPair; extent: number }) {
  const x = (value: number) => 20 + (value + extent) / (2 * extent) * 280;
  return <div className="night-margin-range">
    <svg viewBox="0 0 320 44" role="img" aria-label={`Average margin ${signed(pair.meanMarginPp)} points; middle 90% from ${signed(pair.range90.lower)} to ${signed(pair.range90.upper)} points. Positive means ${pair.firstCandidateName} ahead.`}>
      <line x1="20" x2="300" y1="14" y2="14" className="night-range-axis" />
      <line x1="160" x2="160" y1="3" y2="26" className="night-range-zero" />
      <line x1={x(pair.range90.lower)} x2={x(pair.range90.upper)} y1="14" y2="14" className="night-range-interval" />
      <circle cx={x(pair.meanMarginPp)} cy="14" r="4" className="night-range-point" />
      <text x="20" y="40">−{extent}</text><text x="160" y="40" textAnchor="middle">0</text><text x="300" y="40" textAnchor="end">+{extent} points</text>
    </svg>
    <p>Middle 90%: <strong>{signed(pair.range90.lower)} to {signed(pair.range90.upper)} points</strong>.</p>
    <p className="small">Positive means {pair.firstCandidateName} ahead; negative means {pair.secondCandidateName} ahead. Dot: average margin.</p>
  </div>;
}

function InputNotice({ notice, detail = false }: { notice: ElectionNightInputNotice; detail?: boolean }) {
  return <aside className="night-input-notice" aria-label={`${notice.districtKey} forecast input notice`}>
    <strong>{notice.districtKey} inputs under review · issued {notice.issuedDate}</strong>
    <p>{notice.publicWording}</p>
    {detail && <><p>{notice.explanation}</p><ul>{notice.sources.map(source => <li key={source.url}>
      <a href={source.url} target="_blank" rel="noreferrer">{source.label}</a> · {source.date}
    </li>)}</ul></>}
  </aside>;
}

function RacePrediction({ race, extent, featured = false, notice, expectedError }: { race: ElectionNightRace; extent: number; featured?: boolean; notice?: ElectionNightInputNotice; expectedError?: number }) {
  const pair = race.pair;
  return <article className="night-race" data-scorecard-race={race.id}>
    <div className="night-race-heading"><h4>{label(race)}</h4></div>
    {notice && <InputNotice notice={notice} detail />}
    <dl className="night-prediction-facts">
      <div><dt>Predicted two-candidate margin</dt><dd>{pair ? pointPrediction(pair) : "No supported two-candidate margin"}</dd></div>
      <div><dt>Highest win chance</dt><dd>{race.modalPick.tiedOutcomeIds.length > 1 ? "Tied leading chances" : race.modalPick.name}{" · "}{probability(race.modalPick.probability)}{race.modalPick.kind === "group" && " for the group"}</dd></div>
      <div><dt>Final result / prediction error</dt><dd>Awaiting verified result</dd></div>
    </dl>
    {featured && pair && <MarginRange pair={pair} extent={extent} />}
    {expectedError !== undefined && <p className="night-race-expected-error">For this race, the model expects its predicted margin to miss by <strong>{expectedError.toFixed(1)} percentage points</strong> on average, ignoring the direction of the miss. This is a simulation-based expectation, not an error limit.</p>}
    {featured && pair?.closeWithin2Pp !== null && pair?.closeWithin2Pp !== undefined && <p className="night-close-chance">
      <strong>{probability(pair.closeWithin2Pp)}</strong> chance that these two candidates finish within 2 points of each other.
    </p>}
    <details className="night-race-definitions"><summary>Prediction details and what is covered</summary>
      <ul>{race.outcomes.map(outcome => <li key={outcome.id}>{outcome.name}: {probability(outcome.winProbability)}{outcome.kind === "group" ? " for this group, not for each member" : ""}.
        {outcome.kind === "group" && <> Members: {outcome.members.map(member => member.name).join("; ")}.</>}
      </li>)}</ul>
      {pair && <>
        <p>The margin compares {pair.firstCandidateName} with {pair.secondCandidateName}. Denominator: {pair.denominator}. It does not include votes outside that pair.</p>
        <p>Average margin: {signed(pair.meanMarginPp)} points. Median margin: {signed(pair.medianMarginPp)} points, with half of predictions on each side. Middle 90%: {signed(pair.range90.lower)} to {signed(pair.range90.upper)} points.</p>
        <p>{pair.firstCandidateName} exceeds half of the pair’s votes in {probability(pair.firstExceedsHalf)} of the model’s predictions. {pair.winnerRelationship === "same_event"
          ? "That is the winning event used by this model’s two-contender approximation."
          : "This pair event is separate from the winner chances above."}</p>
        {pair.closeWithin2Pp === null ? <p>The saved forecast does not report the chance of a margin within 2 points. The range alone cannot supply that probability.</p> : <>
          <p>Within 2 points: {probability(pair.closeWithin2Pp)}.</p>
          <p>A lead of at least 10 points: {pair.firstCandidateName} {probability(pair.firstAtLeast10Pp!)}; {pair.secondCandidateName} {probability(pair.secondAtLeast10Pp!)}.</p>
        </>}
        {pair.qualifications.length > 0 && <ul>{pair.qualifications.map((note, index) => <li key={index}>{note}</li>)}</ul>}
      </>}
      {race.notes.length > 0 && <ul>{race.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}
      {race.sourceUrls.length > 0 && <ul>{race.sourceUrls.map((url, index) => <li key={url}><a href={url} target="_blank" rel="noreferrer">Forecast input source {index + 1}</a></li>)}</ul>}
    </details>
  </article>;
}

/** Same-page preparation view. Final-result ingestion and an evaluation lock are separate contracts. */
export default function ElectionNightGuide({ chamber, onClose }: { chamber: ChamberId; onClose: () => void }) {
  const heading = useRef<HTMLHeadingElement>(null);
  const [loaded, setLoaded] = useState<ElectionNightLoaded | null>(null);
  const [request, setRequest] = useState<ElectionNightRequest>({ chamber, runId: null });
  const [attempt, setAttempt] = useState(0);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [watchIds, setWatchIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [allExpanded, setAllExpanded] = useState(false);
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    heading.current?.scrollIntoView({ block: "start", behavior: "instant" });
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError("");
    loadElectionNightForecast(request, controller.signal, referencePublication).then(value => {
      if (controller.signal.aborted) return;
      if (request.runId === null) setWatchIds([...value.forecast.races]
        .sort((a, b) => a.modalPick.probability - b.modalPick.probability || a.id.localeCompare(b.id))
        .slice(0, 6).map(race => race.id));
      setLoaded(value); setBusy(false);
    }).catch(reason => {
      if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : "The scorecard forecast could not be loaded."); setBusy(false); }
    });
    return () => controller.abort();
  }, [request, attempt]);
  const forecast = loaded?.forecast;
  const reference = loaded?.reference;
  const featured = useMemo(() => watchIds.flatMap(id => forecast?.races.find(race => race.id === id) ?? []), [watchIds, forecast]);
  const extent = Math.max(10, Math.ceil(Math.max(0, ...featured.flatMap(race => race.pair
    ? [Math.abs(race.pair.range90.lower), Math.abs(race.pair.range90.upper), Math.abs(race.pair.meanMarginPp)] : [])) / 5) * 5);
  const filtered = useMemo(() => forecast?.races.filter(race => `${label(race)} ${race.outcomes.map(outcome => outcome.name).join(" ")}`.toLowerCase().includes(query.toLowerCase().trim())) ?? [], [forecast, query]);
  const shown = allExpanded ? filtered : filtered.slice(0, 10);
  return <section id="election-night-scorecard" className="night-guide" data-scorecard-chamber={chamber} aria-labelledby="night-guide-title" aria-busy={busy}>
    <div className="night-guide-heading"><div><div className="eyebrow">An election-night watcher’s guide</div>
      <h2 id="night-guide-title" ref={heading} tabIndex={-1}>Election night scorecard</h2>
      <p>What the model predicts, how wrong it expects to be, and what the results can tell us.</p>
    </div><button type="button" className="night-close" onClick={onClose}>Close scorecard</button></div>
    {error && <div className="error-box" role="alert"><p>{error}</p>{loaded && <p>The previously verified forecast is still shown below.</p>}<button type="button" onClick={() => setAttempt(value => value + 1)}>Try again</button></div>}
    {!forecast || !loaded ? !error && <p role="status">Loading the standard {chamber === "house" ? "House" : "Senate"} forecast…</p> : <>
      <div className="night-edition"><strong>{forecast.chamber === "house" ? "House" : "Senate"} · {forecast.run.label}</strong>
        <span>{forecast.run.id === loaded.defaultRunId ? "Standard forecast" : "Alternative model"} · Dated snapshot</span>
        <p>Inputs through {date(forecast.run.cutoffAt)}. Forecast issued {date(forecast.run.issuedAt)}.</p>
        <p>This is a saved forecast edition. The final election-night evaluation edition and grading rules have not yet been locked.</p>
        <p>The chamber tabs above also change this scorecard. Its model selection is separate from the main forecast.</p>
      </div>
      {forecast.inputNotices.map(notice => <InputNotice key={notice.raceId} notice={notice} />)}
      <details className="night-models"><summary>Inspect other model stacks</summary>
        <label>Scorecard model <select disabled={busy} value={forecast.run.id} onChange={event => {
          const choice = loaded.choices.find(item => item.id === event.target.value);
          if (choice) setRequest({ chamber, runId: choice.id, expectedResultSha256: choice.resultSha256 });
        }}>{loaded.choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label}{choice.id === loaded.defaultRunId ? " · Standard" : ""}</option>)}</select></label>
        <p>Changes here affect only this scorecard. Opening it again starts with the standard forecast.</p>
      </details>
      {busy && <p role="status">Loading the selected model. The previously verified forecast remains visible.</p>}
      <section className="night-section" aria-labelledby="night-results-title">
        <div className="eyebrow">1 · Results and predictive power</div><h3 id="night-results-title">The predictions we can check</h3>
        <p>{forecast.chamber === "senate" ? "This Senate edition covers each race’s two named principal contenders, under the stated election-rule approximations." : "Winner chances cover the listed modeled candidates or groups. A group’s chance does not identify which member wins."}</p>
        <div className="night-summary"><div><strong>{forecast.summary.raceCount}</strong><span>modeled contests</span></div>
          <div><strong>{forecast.summary.pairRaceCount}</strong><span>with a two-candidate margin</span></div>
          <div><strong>Not scored</strong><span>verified election results have not been added</span></div></div>
        <h4>Races to watch</h4><p>These six examples have the least certain winners in the dated standard forecast. The same races stay here when you inspect another stack. Their margins show whether an even win chance also means a close expected race.</p>
        <div className="night-featured">{featured.map(race => <RacePrediction key={race.id} race={race} extent={extent} featured notice={forecast.inputNotices.find(note => note.raceId === race.id)} expectedError={reference?.perRace.find(row => row.raceId === race.id)?.expectedAbsoluteMarginErrorPp} />)}</div>
        <details className="night-all-races"><summary>All {forecast.summary.raceCount} predictions</summary>
          <label>Find a race or candidate <input value={query} onChange={event => { setQuery(event.target.value); setAllExpanded(false); }} /></label>
          <p className="small">Showing {shown.length} of {filtered.length} matching contests.</p>
          <div className="night-predictions">{shown.map(race => <RacePrediction key={race.id} race={race} extent={100} notice={forecast.inputNotices.find(note => note.raceId === race.id)} expectedError={reference?.perRace.find(row => row.raceId === race.id)?.expectedAbsoluteMarginErrorPp} />)}</div>
          {filtered.length > 10 && <button type="button" onClick={() => setAllExpanded(value => !value)}>{allExpanded ? "Show fewer predictions" : `See all ${filtered.length} predictions`}</button>}
          {!filtered.length && <p>No matching races or candidates.</p>}
        </details>
        <p>When verified results are added, a called winner and a final vote margin will be tracked separately. An incomplete count will not be scored as a final margin.</p>
      </section>
      <section className="night-section" aria-labelledby="night-uncertainty-title">
        <div className="eyebrow">2 · Uncertainty</div><h3 id="night-uncertainty-title">What the forecast expects to get wrong</h3>
        {loaded.referenceError && <div className="night-reference-error" role="status"><p>{loaded.referenceError}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Retry uncertainty reference</button></div>}
        <div className="night-uncertainty-grid"><article className="night-miss-reference"><h4>Wrong top choices</h4><div className="night-big-number">{forecast.summary.expectedModalMisses.toFixed(1)} <small>of {forecast.summary.raceCount}</small></div>
          <p>The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.</p>
          {reference && <p className="night-reference-band">Across its simulated elections, the middle 90% runs from <strong>{reference.summary.modalWinnerMissCount.range90.lower} to {reference.summary.modalWinnerMissCount.range90.upper} wrong choices</strong>. Shared errors let several races miss together.</p>}
          <p className="small">{forecast.chamber === "house" ? "A top choice may be a candidate group; its chance is not divided among the members." : "These choices cover the two named principal contenders under each race’s stated assumptions."} {forecast.summary.topTieRaceCount > 0 && `${forecast.summary.topTieRaceCount} contests have tied top chances; a fixed outcome-ID order breaks those ties for this count.`}</p>
        </article><article className="night-margin-reference"><h4>Expected average margin error</h4>{reference ? <>
          <div className="night-reference-value">{reference.summary.meanAbsoluteMarginError.mean.toFixed(1)} <small>percentage points</small></div>
          <p>The model expects its predicted margins to miss by this much on average across {reference.perRace.length} races, ignoring the direction of each miss. Every race counts equally.</p>
          <p className="night-reference-band">The middle 90% of simulated elections has an average error of <strong>{reference.summary.meanAbsoluteMarginError.range90.lower.toFixed(1)} to {reference.summary.meanAbsoluteMarginError.range90.upper.toFixed(1)} points</strong>.</p>
          <p>This is the model’s own expectation, not measured accuracy or a ±{reference.summary.meanAbsoluteMarginError.mean.toFixed(1)}-point range for each race. Individual races have different expected errors, shown with their predictions.</p>
        </> : <><p className="night-pending">No reviewed error reference for this edition</p><p>Individual margin ranges are shown above. A reviewed expected average margin error and shared range for the number of misses are not available for this exact model and date.</p></>}</article></div>
        {reference && <details className="night-reference-details"><summary>How these expected errors were calculated</summary>
          <p>Each saved simulation describes one possible election across all {reference.perRace.length} named candidate pairs. We compare its margins with the forecast’s predicted averages, then average the error sizes across races. The calculation keeps shared errors together; it does not treat races as independent.</p>
          <p>These are margins between the two named contenders under the forecast’s stated rules. They are not full-ballot shares or a national popular-vote margin.</p>
          <h4>How many race ranges could contain the result?</h4>
          <p>The model expects <strong>{reference.summary.interval90CoverageCount.mean.toFixed(1)} of {reference.perRace.length}</strong> race results inside their individual middle-90% margin ranges, on average. The 5th to 95th percentiles for that count are <strong>{reference.summary.interval90CoverageCount.range90.lower} to {reference.summary.interval90CoverageCount.range90.upper}</strong>.</p>
          <p>Those ranges are built from the same model, so this is an internal reference, not evidence that their real-world coverage is accurate. A range’s width and its misses must both be scored.</p>
          <h4>Could the misses lean in one direction?</h4>
          <p>For the average signed margin error, the middle 90% runs from <strong>{signed(reference.summary.meanSignedMarginError.range90.lower)} to {signed(reference.summary.meanSignedMarginError.range90.upper)} points</strong>. Positive means the first named contender does better than predicted; negative means the second does better. Each race’s prediction details name that order.</p>
          <p>The simulated average is centered near zero by construction. That does not establish that actual election errors will be unbiased.</p>
          <p>Reference calculated {date(reference.referenceIssuedAt)} for this exact forecast edition. It contains no election results and assigns no grade.</p>
        </details>}
        {forecast.chamberScenario && <article className="night-chamber"><h4>Shared uncertainty in the chamber</h4>
          <p>{forecast.chamberScenario.label}. {forecast.chamberScenario.description}</p>
          <dl>{(["D", "R", "U"] as const).map(party => {
            const seats = forecast.chamberScenario!.seats[party];
            return <div key={party}><dt>{party === "D" ? "Democratic seats" : party === "R" ? "Republican seats" : "Other / unassigned seats"}</dt>
              <dd>{seats.expectedSeats.toFixed(1)} average · {seats.seatQuantiles.p05}–{seats.seatQuantiles.p95} in the middle 90% of simulated elections</dd></div>;
          })}</dl>
          <p>These chamber ranges come from the saved election simulations. Races can miss in the same direction together; they are not independent tests of a national prediction.</p>
        </article>}
        <p><strong>50/50 does not tell us the margin.</strong> It can mean a narrow contest, or uncertainty about which candidate wins comfortably. Use the separate margin range and close-race chances where available.</p>
      </section>
      <section className="night-section night-rubric" aria-labelledby="night-rubric-title">
        <div className="eyebrow">The evaluation rubric</div><h3 id="night-rubric-title">How the forecast will be judged</h3>
        <p className="small">Grade bands under review</p>
        <div className="night-grade-scale" aria-label="A through F grade bands under review; no grade assigned">{["A", "B", "C", "D", "F"].map(grade => <span key={grade}>{grade}</span>)}</div>
        <p><strong>No letter grade assigned.</strong> The thresholds, comparison forecasts and rules for combining scores are still under review. They must be published before outcomes are scored.</p>
        <dl><div><dt>Specific predictions</dt><dd>Compare the named candidate margins with verified final results, using the same vote definition. Show the size and direction of each error.</dd></div>
          <div><dt>Useful probabilities</dt><dd>Score confident mistakes more heavily and compare performance with simple forecasts on the same races. An uninformative forecast must not pass as demonstrated skill. A 50/50 reference applies to two-outcome contests; contests with more outcomes need a reference covering those same outcomes.</dd></div>
          <div><dt>Honest ranges</dt><dd>Check both how often results fall inside the ranges and how wide those ranges were. Very broad ranges do not earn a high grade merely by containing the results.</dd></div>
          <div><dt>Shared misses</dt><dd>Check coordinated shifts against the model’s joint predictions. Many misses in one election are not many independent elections.</dd></div></dl>
        <p>A result having been possible does not excuse poor performance. One surprising result also does not settle whether the model is reliable across elections. Both the individual misses and the comparison scores need to stay visible.</p>
      </section>
      <details className="night-method"><summary>Edition, sources and evaluation definitions</summary>
        <p>Result fingerprint: <code>{forecast.run.artifactSha256}</code>. This identifies the saved forecast bytes; it is not a claim that the inputs are complete or the model is accurate.</p>
        <p>{forecast.run.strictAsOf ? "Inputs were checked against their stated cutoff." : "This edition does not establish that every input was available at its forecast cutoff."}</p>
        {forecast.run.sourceCutoffs.map((source, index) => <p key={index}>{source.label}: {date(source.at)}</p>)}
        <p>The expected wrong-choice count adds one minus each race’s highest modeled win chance. It needs no independence assumption and does not supply an uncertainty band. Exact ties use a fixed ordering of the outcome identifiers.</p>
        <ul>{forecast.qualification.map((note, index) => <li key={index}>{note}</li>)}</ul>
        {forecast.inputNotices.map(notice => <InputNotice key={notice.raceId} notice={notice} detail />)}
        {forecast.sourceNotes.length > 0 && <><h4>Source capture notes</h4><ul>{forecast.sourceNotes.map((source, index) => <li key={index}>
          <a href={source.url} target="_blank" rel="noreferrer">{source.label}</a><ul>{source.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>
        </li>)}</ul></>}
        <p>Methodological basis: <a href="https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf" target="_blank" rel="noreferrer">proper probability and interval scores</a>; <a href="https://sites.stat.columbia.edu/gelman/research/published/jdm200907b.pdf" target="_blank" rel="noreferrer">election-forecast evaluation and communication</a>.</p>
      </details>
      <button type="button" className="night-close" onClick={onClose}>Return to the forecast</button>
    </>}
  </section>;
}
