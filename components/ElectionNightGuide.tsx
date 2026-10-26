"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ChamberId } from "../lib/chamber";
import { loadPublishedElectionNightForecast, type ElectionNightLoaded, type ElectionNightRequest } from "../lib/election-night-client";
import type { ElectionNightInputNotice, ElectionNightPair, ElectionNightRace } from "../lib/election-night-forecast";
import { electionNightCalls } from "../lib/election-night-calls";
import type { ElectionNightReference } from "../lib/election-night-reference";
import type { HouseExpectedErrorReference } from "../lib/election-night-house-reference";
import ElectionNightEvaluation from "./ElectionNightEvaluation";
import ElectionNightRubric from "./ElectionNightRubric";
import { useElectionNightEvaluation } from "./useElectionNightEvaluation";
import senateResultPolicy from "../data/election-night-senate-result-policy.json";
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

function raceCallLabel(race: ElectionNightRace): string {
  const outcome = race.outcomes.find(item => item.id === race.modalPick.outcomeId)!;
  return outcome.kind === "candidate" ? outcome.name : outcome.members.length === 1
    ? `${outcome.members[0].name} (${outcome.name})` : `${outcome.name} (individual winner not distinguished: ${outcome.members.map(member => member.name).join(", ")})`;
}
function RaceCallRow({ race, notice }: { race: ElectionNightRace; notice?: ElectionNightInputNotice }) {
  const outcome = race.outcomes.find(item => item.id === race.modalPick.outcomeId)!;
  const call = raceCallLabel(race);
  const category = outcome.winProbability <= .6 ? "Tossup" : outcome.winProbability <= .9 ? "Likely" : "Safe";
  return <article className="night-call-row" data-call-race={race.id}>
    <div><span>{label(race)} · {race.modalPick.tiedOutcomeIds.length > 1 ? "Tied top chances · fixed-order call" : "Our call"}</span><strong>{call}</strong><small>{probability(outcome.winProbability)}{outcome.kind === "group" ? " chance for the group" : " chance to win"} · {category}</small></div>
    <div><span>Predicted two-candidate margin</span><strong>{race.pair ? pointPrediction(race.pair) : "Not available"}</strong>{race.pair?.winnerRelationship === "separate_conditional_event" && <small>This pair margin is separate from the winner-group call.</small>}</div>
    {notice && <InputNotice notice={notice} />}
  </article>;
}

function RacePrediction({ race, extent, featured = false, notice, reference, houseReference, resultStatus }: { race: ElectionNightRace; extent: number;
  featured?: boolean; notice?: ElectionNightInputNotice; reference?: ElectionNightReference["perRace"][number];
  houseReference?: HouseExpectedErrorReference["perRace"][number]; resultStatus: string }) {
  const pair = race.pair;
  const expectedError = reference?.expectedAbsoluteMarginErrorPp ?? houseReference?.expectedAbsoluteMarginErrorPp ?? undefined;
  const closeChance = reference?.closeWithin2Pp ?? pair?.closeWithin2Pp ?? null;
  const firstDecisive = reference?.firstAtLeast10Pp ?? pair?.firstAtLeast10Pp ?? null;
  const secondDecisive = reference?.secondAtLeast10Pp ?? pair?.secondAtLeast10Pp ?? null;
  return <article className="night-race" data-scorecard-race={race.id}>
    <div className="night-race-heading"><h4>{label(race)}</h4></div>
    {notice && <InputNotice notice={notice} detail />}
    <dl className="night-prediction-facts">
      <div><dt>Predicted two-candidate margin</dt><dd>{pair ? pointPrediction(pair) : "No supported two-candidate margin"}</dd></div>
      <div><dt>Highest win chance</dt><dd>{race.modalPick.tiedOutcomeIds.length > 1 ? "Tied leading chances" : race.modalPick.name}{" · "}{probability(race.modalPick.probability)}{race.modalPick.kind === "group" && " for the group"}</dd></div>
      <div><dt>Final result / prediction error</dt><dd>{resultStatus}</dd></div>
    </dl>
    {featured && pair && <MarginRange pair={pair} extent={extent} />}
    {expectedError !== undefined && <p className="night-race-expected-error">For this race, the model expects its predicted margin to miss by <strong>{expectedError.toFixed(1)} percentage points</strong> on average, ignoring the direction of the miss. This is the mean across saved simulation draws, not an error limit.</p>}
    {featured && pair && closeChance !== null && firstDecisive !== null && secondDecisive !== null && <div className="night-margin-chances">
      <p className="small">These margins count only votes for the two named candidates.</p>
      <p className="night-close-chance"><strong>{probability(closeChance)}</strong> chance that these two candidates finish within 2 points of each other.</p>
      <dl><div><dt>{pair.firstCandidateName} ahead by at least 10 points</dt><dd>{probability(firstDecisive)}</dd></div>
        <div><dt>{pair.secondCandidateName} ahead by at least 10 points</dt><dd>{probability(secondDecisive)}</dd></div></dl>
    </div>}
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
        {closeChance === null || firstDecisive === null || secondDecisive === null ? <p>The saved forecast does not report the chance of a margin within 2 points. The range alone cannot supply that probability.</p> : <>
          <p>Within 2 points: {probability(closeChance)}.</p>
          <p>A lead of at least 10 points: {pair.firstCandidateName} {probability(firstDecisive)}; {pair.secondCandidateName} {probability(secondDecisive)}.</p>
          <p>These thresholds include exactly 2 and 10 points. Margins between 2 and 10 points fall outside all three categories, so the chances need not add to 100%. A zero in the saved simulations does not mean an outcome is impossible.</p>
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
  const evaluation = useElectionNightEvaluation(chamber);
  const heading = useRef<HTMLHeadingElement>(null);
  const standardWatchEdition = useRef<string | null>(null);
  const standardWatchNeedsRefresh = useRef(false);
  const [loaded, setLoaded] = useState<ElectionNightLoaded | null>(null);
  const [request, setRequest] = useState<ElectionNightRequest>({ chamber, runId: null });
  const [attempt, setAttempt] = useState(0);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [watchIds, setWatchIds] = useState<string[]>([]);
  const [watchIsStandard, setWatchIsStandard] = useState(true);
  const [query, setQuery] = useState("");
  const [allExpanded, setAllExpanded] = useState(false);
  const [otherExpanded, setOtherExpanded] = useState(false);
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    heading.current?.scrollIntoView({ block: "start", behavior: "instant" });
  }, []);
  useEffect(() => {
    let active: AbortController | null = null;
    let disposed = false;
    const refresh = async (background: boolean) => {
      if (disposed || active) return;
      const controller = new AbortController(); active = controller;
      if (!background) { setBusy(true); setError(""); }
      try {
        // A dropdown SHA guards the immediate click. Later checks follow that
        // stable model ID if the site republishes it with new verified bytes.
        const current = background ? { chamber: request.chamber, runId: request.runId } : request;
        let value: ElectionNightLoaded;
        try { value = await loadPublishedElectionNightForecast(current, controller.signal); }
        catch (reason) {
          if (!background && request.expectedResultSha256 && reason instanceof Error &&
            /edition changed|published forecast changed while loading/i.test(reason.message))
            value = await loadPublishedElectionNightForecast({ chamber: request.chamber, runId: request.runId }, controller.signal);
          else throw reason;
        }
        if (controller.signal.aborted || disposed) return;
        const standard = value.choices.find(choice => choice.id === value.defaultRunId);
        const standardEdition = standard ? `${value.forecast.chamber}:${standard.id}:${standard.resultSha256}` : null;
        const standardChanged = standardEdition !== standardWatchEdition.current;
        standardWatchEdition.current = standardEdition;
        const showingStandard = value.forecast.run.id === value.defaultRunId;
        if (standardChanged && !showingStandard) standardWatchNeedsRefresh.current = true;
        const recompute = standardChanged || (showingStandard && standardWatchNeedsRefresh.current);
        if (showingStandard) standardWatchNeedsRefresh.current = false;
        if (recompute) setWatchIsStandard(showingStandard);
        setWatchIds(currentIds => {
          const roster = new Set(value.forecast.races.map(race => race.id));
          const kept = recompute ? [] : currentIds.filter(id => roster.has(id)).slice(0, 6);
          const ranked = [...value.forecast.races]
            .sort((a, b) => a.modalPick.probability - b.modalPick.probability || a.id.localeCompare(b.id))
            .map(race => race.id);
          for (const id of ranked) if (kept.length < 6 && !kept.includes(id)) kept.push(id);
          return kept.join("|") === currentIds.join("|") ? currentIds : kept;
        });
        setLoaded(value); setError(""); setBusy(false);
      } catch (reason) {
        if (!controller.signal.aborted && !disposed) {
          setError(reason instanceof Error ? reason.message : "The scorecard forecast could not be loaded.");
          setBusy(false);
        }
      } finally { if (active === controller) active = null; }
    };
    void refresh(false);
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void refresh(true); }, 60_000);
    const onVisible = () => { if (document.visibilityState === "visible") void refresh(true); };
    const onFocus = () => { void refresh(true); };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onFocus);
    return () => { disposed = true; active?.abort(); window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible); window.removeEventListener("focus", onFocus); };
  }, [request, attempt]);
  const displayed = loaded?.forecast.chamber === chamber ? loaded : null;
  const forecast = displayed?.forecast;
  const calls = useMemo(() => forecast ? electionNightCalls(forecast) : null, [forecast]);
  const reference = displayed?.reference;
  const primaryReference = displayed?.primaryReference;
  const houseReference = displayed?.houseReference;
  const evaluationHasResults = !!evaluation.view?.ledger?.recordCount;
  const recordedEvaluationRaces = new Set(evaluation.view?.rows.filter(row => row.sources.length > 0).map(row => row.raceId) ?? []);
  const resultStatus = (raceId: string) => evaluation.loading ? "Checking evaluation status" : evaluation.error || evaluation.view?.status === "unavailable"
    ? "Evaluation temporarily unavailable" : recordedEvaluationRaces.has(raceId) ? "See the locked evaluation above" : "Awaiting verified result";
  const senateScope = forecast?.chamber === "senate" && forecast.run.configuration.polling === "standard" && primaryReference &&
    primaryReference.sourcePins.policySha256 === senateResultPolicy.sourceHashes.policySha256 &&
    JSON.stringify(forecast.races.map(race => race.id)) === JSON.stringify(senateResultPolicy.fixedCohorts.full35) &&
    JSON.stringify(primaryReference.perRace.map(race => race.raceId)) === JSON.stringify(senateResultPolicy.fixedCohorts.pluralityCore32) &&
    JSON.stringify(primaryReference.excludedRaceIds) === JSON.stringify(senateResultPolicy.fixedCohorts.conditional3) &&
    (!evaluation.view?.lock || (JSON.stringify(evaluation.view.lock.cohort.displayRaceIds) === JSON.stringify(senateResultPolicy.fixedCohorts.full35) &&
      JSON.stringify(evaluation.view.lock.cohort.winnerRaceIds) === JSON.stringify(senateResultPolicy.fixedCohorts.pluralityCore32)))
    ? senateResultPolicy : null;
  const featured = useMemo(() => watchIds.flatMap(id => forecast?.races.find(race => race.id === id) ?? []), [watchIds, forecast]);
  const extent = Math.max(10, Math.ceil(Math.max(0, ...featured.flatMap(race => race.pair
    ? [Math.abs(race.pair.range90.lower), Math.abs(race.pair.range90.upper), Math.abs(race.pair.meanMarginPp)] : [])) / 5) * 5);
  const filtered = useMemo(() => forecast?.races.filter(race => `${label(race)} ${race.outcomes.map(outcome => outcome.name).join(" ")} ${forecast.inputNotices.find(notice => notice.raceId === race.id)?.publicWording ?? ""}`.toLowerCase().includes(query.toLowerCase().trim())) ?? [], [forecast, query]);
  const shown = allExpanded ? filtered : filtered.slice(0, 10);
  const otherCalls = useMemo(() => {
    const competitive = new Set(calls?.competitive.map(race => race.id) ?? []);
    return forecast?.races.filter(race => !competitive.has(race.id)).sort((a, b) => a.modalPick.probability - b.modalPick.probability || a.id.localeCompare(b.id)) ?? [];
  }, [calls, forecast]);
  return <section id="election-night-scorecard" className="night-guide" data-scorecard-chamber={chamber} aria-labelledby="night-guide-title" aria-busy={busy}>
    <div className="night-guide-heading"><div><div className="eyebrow">An election-night watcher’s guide</div>
      <h2 id="night-guide-title" ref={heading} tabIndex={-1}>Election night scorecard</h2>
      <p>What the model predicts, how wrong it expects to be, and what the results can tell us.</p>
    </div><div className="night-guide-actions"><button type="button" className="night-close" disabled={!forecast || busy} onClick={() => window.print()}>Print scorecard</button><button type="button" className="night-close" onClick={onClose}>Close scorecard</button></div></div>
    {error && <div className="error-box" role="alert"><p>{error}</p>{loaded && <p>The previously verified forecast is still shown below.</p>}<button type="button" onClick={() => setAttempt(value => value + 1)}>Try again</button></div>}
    {!forecast || !displayed ? !error && <p role="status">Loading the standard {chamber === "house" ? "House" : "Senate"} forecast…</p> : <>
      <div className="night-edition" data-scorecard-result-sha={forecast.run.artifactSha256}><strong>{forecast.chamber === "house" ? "House" : "Senate"} · {forecast.run.label}</strong>
        <span>{forecast.run.id === displayed.defaultRunId ? "Standard forecast" : "Alternative model"} · Dated snapshot</span>
        <p>Inputs through {date(forecast.run.cutoffAt)}. Forecast issued {date(forecast.run.issuedAt)}.</p>
        <p>Updates automatically from the published forecast while this scorecard is open.</p>
        <p>{evaluation.loading ? "Checking the published evaluation status." : evaluation.error || evaluation.view?.status === "unavailable"
          ? "The published evaluation status could not be verified." : evaluation.view?.lock
            ? "The locked evaluation is shown separately from this updating forecast."
            : "No locked election-night evaluation has been published on this site yet."}</p>
        <p>The chamber tabs above also change this scorecard. Its model selection is separate from the main forecast.</p>
      </div>
      {forecast.inputNotices.map(notice => <InputNotice key={notice.raceId} notice={notice} />)}
      <details className="night-models"><summary>Inspect other model stacks</summary>
        <label>Scorecard model <select disabled={busy} value={forecast.run.id} onChange={event => {
          const choice = displayed.choices.find(item => item.id === event.target.value);
          if (choice) setRequest({ chamber, runId: choice.id === displayed.defaultRunId ? null : choice.id,
            expectedResultSha256: choice.id === displayed.defaultRunId ? undefined : choice.resultSha256 });
        }}>{displayed.choices.map(choice => <option key={choice.id} value={choice.id}>{choice.label}{choice.id === displayed.defaultRunId ? " · Standard" : ""}</option>)}</select></label>
        <p>Changes here affect only this scorecard. Opening it again starts with the standard forecast.</p>
      </details>
      {busy && <p role="status">Loading the selected model. The previously verified forecast remains visible.</p>}
      <section className="night-print-only" aria-label="Printable scorecard">
        <div className="eyebrow">Ballotmancy · Election-night watcher’s guide</div>
        <h1>{forecast.chamber === "house" ? "House" : "Senate"} scorecard</h1>
        <p>{forecast.run.label} · {forecast.run.id === displayed.defaultRunId ? "Standard forecast" : "Alternative model"}<br />Inputs through {date(forecast.run.cutoffAt)} · Issued {date(forecast.run.issuedAt)}</p>
        <h2>Point prediction</h2>
        {calls?.seats && calls.control ? <p><strong>{calls.control === "U" ? "Control unresolved" : `${calls.control === "D" ? "Democrats" : "Republicans"} control the ${forecast.chamber === "house" ? "House" : "Senate"} with ${calls.seats[calls.control]} seats`}</strong><br />{calls.seats.D} Democratic · {calls.seats.R} Republican · {calls.seats.U} other / unassigned.</p> : <p>A complete chamber prediction is unavailable for this scenario.</p>}
        <p>Seat counts add the highest-chance outcome in each contest{forecast.chamber === "senate" ? " to the seats not up for election; the vice president breaks a 50–50 tie" : "; 218 seats are needed for a majority"}. Fixed ordering breaks exact ties. These are point calls, not guarantees or the most common simulated outcome.</p>
        {forecast.inputNotices.map(notice => <p key={notice.raceId}><strong>{notice.districtKey} inputs under review:</strong> {notice.publicWording}</p>)}
        <h2>Race predictions</h2>
        <p>Competitive contests appear first: a leading win chance of at most 90%, or a predicted two-candidate margin within five points. Group chances do not distinguish individual winners. A pair margin is a separate measure and, in House races, may not represent the winner-group event.</p>
        <table><thead><tr><th>Contest</th><th>Point call</th><th>Win chance</th><th>Two-candidate margin</th></tr></thead><tbody>
          {[...(calls?.competitive ?? []), ...otherCalls].map(race => <tr key={race.id}><td>{label(race)}</td><td>{raceCallLabel(race)}{race.modalPick.tiedOutcomeIds.length > 1 ? " (fixed-order tie)" : ""}</td><td>{probability(race.modalPick.probability)}{race.modalPick.kind === "group" ? " group" : ""}</td><td>{race.pair ? pointPrediction(race.pair) : "Unavailable"}</td></tr>)}
        </tbody></table>
        <h2>Uncertainty</h2>
        <p>The model expects {forecast.summary.expectedModalMisses.toFixed(1)} wrong top choices on average across {forecast.summary.raceCount} races. This is the model’s own expectation, not a guaranteed limit or a measured accuracy record.</p>
        {forecast.chamberScenario && <p>Middle 90% of simulated seat totals: Democrats {forecast.chamberScenario.seats.D.seatQuantiles.p05}–{forecast.chamberScenario.seats.D.seatQuantiles.p95}; Republicans {forecast.chamberScenario.seats.R.seatQuantiles.p05}–{forecast.chamberScenario.seats.R.seatQuantiles.p95}. Shared errors can move races together.</p>}
        <p>Forecasts update on the website. This printout records the dated edition above; the election-night evaluation needs a separately locked forecast and verified results.</p>
      </section>
      <section className="night-section" aria-labelledby="night-results-title">
        <div className="eyebrow">1 · Results and predictive power</div><h3 id="night-results-title">The predictions we can check</h3>
        <p>{forecast.chamber === "senate" ? "This Senate edition covers each race’s two named principal contenders, under the stated election-rule approximations." : "Winner chances cover the listed modeled candidates or groups. A group’s chance does not identify which member wins."}</p>
        <article className="night-hard-call" aria-label="Point prediction for chamber control">
          <div className="eyebrow">Our point prediction</div>
          {calls?.seats && calls.control ? <>
            <h4>{calls.control === "U" ? `Control of the ${forecast.chamber === "house" ? "House" : "Senate"} is unresolved` : `${calls.control === "D" ? "Democrats" : "Republicans"} control the ${forecast.chamber === "house" ? "House" : "Senate"} with ${calls.seats[calls.control]} seats`}</h4>
            <p><strong>{calls.seats.D} Democratic · {calls.seats.R} Republican{calls.seats.U > 0 ? ` · ${calls.seats.U} other / unassigned` : ""}</strong></p>
            <p>We pick the highest-chance outcome in each race and add those calls{forecast.chamber === "senate" ? " to the seats not up for election. A 50–50 Senate gives Republicans control under the current vice-presidential tie-break" : ". A House majority requires 218 seats"}. Independent and other winners remain unassigned in this scenario.</p>
            <p>This is one concrete prediction, not a guarantee or the most common simulated outcome. The model’s control chances and shared uncertainty are shown below. {forecast.summary.topTieRaceCount > 0 && `For ${forecast.summary.topTieRaceCount} exact top-chance ties, a fixed ordering selects the point call.`}</p>
          </> : <p>A complete seat-by-seat point prediction is unavailable for this forecast’s chamber scenario.</p>}
        </article>
        <div className="night-competitive" aria-labelledby="night-competitive-title">
          <h4 id="night-competitive-title">Every competitive race · {calls?.competitive.length ?? 0}</h4>
          <p>Includes every race where the leading outcome has at most a 90% win chance, plus any race with an estimated two-candidate margin of 5 points or less. Closest win chances come first.</p>
          <p>We make a call even in a tossup. A win chance measures uncertainty about the winner; the predicted margin is a separate estimate of the vote gap.</p>
          <div className="night-call-list">{calls?.competitive.map(race => <RaceCallRow key={race.id} race={race} notice={forecast.inputNotices.find(notice => notice.raceId === race.id)} />)}</div>
          {!calls?.competitive.length && <p>No races meet these criteria in this edition.</p>}
          {otherCalls.length > 0 && <><button type="button" aria-expanded={otherExpanded} onClick={() => setOtherExpanded(value => !value)}>{otherExpanded ? "Hide the other race calls" : `See all ${otherCalls.length} other race calls`}</button>
            {otherExpanded && <div className="night-call-list">{otherCalls.map(race => <RaceCallRow key={race.id} race={race} notice={forecast.inputNotices.find(notice => notice.raceId === race.id)} />)}</div>}</>}
        </div>
        {senateScope && <aside className="night-cohort-note" aria-label="Senate evaluation coverage">
          <strong>All {senateScope.fixedCohorts.full35.length} predictions stay visible. The primary grade covers {senateScope.fixedCohorts.pluralityCore32.length} candidate pairs.</strong>
          <p>Alaska and Maine may require ranked-choice rounds; Georgia may require a runoff. They remain on the scoreboard with their own status. This grouping is chosen before election results arrive.</p>
          <p>The grade tests the margin and which of the two named candidates finishes ahead. It does not score the chance of every possible office winner.</p>
          <details><summary>Why three races are tracked separately</summary>
            <p>For Alaska and Maine, the comparison requires an officially reported final round with exactly the two forecast candidates remaining. We will not invent transfers or substitute first-choice votes.</p>
            <p>For Georgia, the general-election count is used only if it decides the election without a runoff. Otherwise, the comparison waits for an actual runoff between the two forecast candidates.</p>
            <p>All comparisons concern the two named candidates’ votes. If another candidate wins the office, that is recorded separately; it does not erase known votes for the forecast pair. Certified tied pair counts can give a zero margin, but no winner-probability score.</p>
            <p>The primary grade needs eligible certified results for every one of its {senateScope.fixedCohorts.pluralityCore32.length} races. A missing or disputed result stays pending; it is not dropped to produce a grade.</p>
          </details>
        </aside>}
        <div className="night-summary"><div><strong>{forecast.summary.raceCount}</strong><span>modeled contests</span></div>
          <div><strong>{forecast.summary.pairRaceCount}</strong><span>with a two-candidate margin</span></div>
          <div><strong>{evaluationHasResults ? "See evaluation" : "Not scored"}</strong><span>{evaluationHasResults ? "locked forecast results below" : "no verified results shown"}</span></div></div>
        <ElectionNightEvaluation view={evaluation.view} currentResultSha256={forecast.run.artifactSha256} />
        <h4>Races to watch</h4><p>{watchIsStandard ? "These six examples have the least certain winners in the dated standard forecast. The same races stay here when you inspect another stack." : "These six examples have the least certain winners in the selected model. Choose Standard to see its latest set."} Their margins show whether an even win chance also means a close expected race.</p>
        <div className="night-featured">{featured.map(race => <RacePrediction key={race.id} race={race} extent={extent} featured notice={forecast.inputNotices.find(note => note.raceId === race.id)} reference={reference?.perRace.find(row => row.raceId === race.id)} houseReference={houseReference?.perRace.find(row => row.raceId === race.id)} resultStatus={resultStatus(race.id)} />)}</div>
        <details className="night-all-races"><summary>All {forecast.summary.raceCount} predictions</summary>
          <label>Find a race or candidate <input value={query} onChange={event => { setQuery(event.target.value); setAllExpanded(false); }} /></label>
          <p className="small">Showing {shown.length} of {filtered.length} matching contests.</p>
          <div className="night-predictions">{shown.map(race => <RacePrediction key={race.id} race={race} extent={100} notice={forecast.inputNotices.find(note => note.raceId === race.id)} reference={reference?.perRace.find(row => row.raceId === race.id)} houseReference={houseReference?.perRace.find(row => row.raceId === race.id)} resultStatus={resultStatus(race.id)} />)}</div>
          {filtered.length > 10 && <button type="button" onClick={() => setAllExpanded(value => !value)}>{allExpanded ? "Show fewer predictions" : `See all ${filtered.length} predictions`}</button>}
          {!filtered.length && <p>No matching races or candidates.</p>}
        </details>
        <p>A called winner and a final vote margin are tracked separately. An incomplete count is not scored as a final margin.</p>
      </section>
      <section className="night-section" aria-labelledby="night-uncertainty-title">
        <div className="eyebrow">2 · Uncertainty</div><h3 id="night-uncertainty-title">What the forecast expects to get wrong</h3>
        {displayed.referenceError && <div className="night-reference-error" role="status"><p>{displayed.referenceError}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Retry uncertainty reference</button></div>}
        {displayed.houseReferenceError && <div className="night-reference-error" role="status"><p>{displayed.houseReferenceError}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Retry House expected-error reference</button></div>}
        <h4>All {forecast.summary.raceCount} modeled contests</h4>
        {forecast.chamber === "house" && <p className="small">Winner or group top-choice misses cover all {forecast.summary.raceCount} districts. Margin errors and range coverage cover the {forecast.summary.pairRaceCount} districts where the model estimates a margin between two named contenders; the other {forecast.summary.raceCount - forecast.summary.pairRaceCount} have no such estimate.</p>}
        <div className="night-uncertainty-grid"><article className="night-miss-reference"><h4>Wrong top choices</h4><div className="night-big-number">{forecast.summary.expectedModalMisses.toFixed(1)} <small>of {forecast.summary.raceCount}</small></div>
          <p>The expected number of incorrect top choices under the model’s own probabilities. This is an average, not a limit or a measured accuracy record.</p>
          {(reference || houseReference) && <p className="night-reference-band">Across its simulated elections, the middle 90% runs from <strong>{reference?.summary.modalWinnerMissCount.range90.lower ?? houseReference?.summary.modalGroupMissCount.range90.lower} to {reference?.summary.modalWinnerMissCount.range90.upper ?? houseReference?.summary.modalGroupMissCount.range90.upper} wrong choices</strong>. Shared errors let several races miss together.</p>}
          <p className="small">{forecast.chamber === "house" ? "A top choice may be a candidate group; its chance is not divided among the members." : "These choices cover the two named principal contenders under each race’s stated assumptions."} {forecast.summary.topTieRaceCount > 0 && `${forecast.summary.topTieRaceCount} contests have tied top chances; ${forecast.chamber === "house" ? "the declared Democratic, Republican, Other order" : "a fixed outcome-ID order"} breaks those ties for this count.`}</p>
        </article><article className="night-margin-reference"><h4>Expected average margin error</h4>{reference || houseReference ? <>
          <div className="night-reference-value">{(reference?.summary.meanAbsoluteMarginError.mean ?? houseReference!.summary.meanAbsolutePairMarginError.mean).toFixed(1)} <small>percentage points</small></div>
          <p>The model expects its predicted margins to miss by this much on average across {reference ? reference.perRace.length : houseReference!.pairRaceCount} supported named pairs, ignoring the direction of each miss. Every pair counts equally.</p>
          <p className="night-reference-band">The middle 90% of simulated elections has an average error of <strong>{(reference?.summary.meanAbsoluteMarginError.range90.lower ?? houseReference!.summary.meanAbsolutePairMarginError.range90.lower).toFixed(1)} to {(reference?.summary.meanAbsoluteMarginError.range90.upper ?? houseReference!.summary.meanAbsolutePairMarginError.range90.upper).toFixed(1)} points</strong>.</p>
          <p>This is the model’s own expectation, not measured accuracy or an error limit for each race. Individual races have different expected errors, shown with their predictions where a named pair exists.</p>
        </> : <><p className="night-pending">No reviewed error reference for this edition</p><p>Individual margin ranges are shown above. A reviewed expected average margin error and shared range for the number of misses are not available for this exact model and date.</p></>}</article></div>
        {displayed.primaryReferenceError && <div className="night-reference-error" role="status"><p>{displayed.primaryReferenceError}</p><button type="button" onClick={() => setAttempt(value => value + 1)}>Retry grading-group reference</button></div>}
        {forecast.chamber === "senate" && !primaryReference && !displayed.primaryReferenceError &&
          <p className="night-pending">No reviewed 32-race expected-error reference is available for this exact forecast edition.</p>}
        {primaryReference && <article className="night-primary-reference" aria-label="Primary grading group expected errors" data-reference-run={primaryReference.runId}>
          <h4>The {primaryReference.perRace.length} candidate pairs used for the primary grade</h4>
          <p>This reference uses exactly the fixed grading group. Alaska, Maine and Georgia stay in the full forecast above and are tracked separately for evaluation.</p>
          <div className="night-uncertainty-grid"><div>
            <strong className="night-primary-margin-value">{primaryReference.summary.meanAbsoluteMarginError.mean.toFixed(1)} percentage points</strong>
            <p>Expected average margin error across these {primaryReference.perRace.length} races, ignoring the direction of each miss.</p>
            <p className="night-primary-margin-range">The middle 90% of simulated elections has an average error of {primaryReference.summary.meanAbsoluteMarginError.range90.lower.toFixed(1)} to {primaryReference.summary.meanAbsoluteMarginError.range90.upper.toFixed(1)} points.</p>
          </div><div>
            <strong className="night-primary-miss-value">{primaryReference.summary.modalWinnerMissCount.mean.toFixed(1)} wrong top choices out of {primaryReference.perRace.length}</strong>
            <p className="night-primary-miss-range">The middle 90% runs from {primaryReference.summary.modalWinnerMissCount.range90.lower} to {primaryReference.summary.modalWinnerMissCount.range90.upper} wrong choices in these same races.</p>
          </div></div>
          <p className="small">These are this model’s expectations, not observed errors or grades. The calculation preserves shared errors across races and uses this forecast’s own saved simulations. It does not combine simulations from different models.</p>
        </article>}
        {houseReference && <details className="night-reference-details night-house-reference-details"><summary>How the House expected errors were calculated</summary>
          <p>One saved simulation row covers all {houseReference.winnerRaceCount} winner or group outcomes and {houseReference.pairRaceCount} supported conditional named-pair margins. The other {houseReference.pairInapplicableRaceCount} contests have no pair error or range coverage. Each supported pair counts equally, and shared errors stay together within this model.</p>
          <p>The expected number of pair margins inside their own middle-90% ranges is <strong>{houseReference.summary.central90PairIntervalCoverageCount.mean.toFixed(1)} of {houseReference.pairRaceCount}</strong>. Across simulated elections, the middle 90% of that count runs from {houseReference.summary.central90PairIntervalCoverageCount.range90.lower} to {houseReference.summary.central90PairIntervalCoverageCount.range90.upper}.</p>
          <p>Average signed pair margin error has a simulated middle 90% from {signed(houseReference.summary.meanSignedPairMarginError.range90.lower)} to {signed(houseReference.summary.meanSignedPairMarginError.range90.upper)} points. Positive means the first named contender does better than predicted; negative means the second does better. This self-reference is centered near zero by construction; it does not establish real-world accuracy.</p>
          <p>These estimates come from a finite set of simulated elections. Numerical diagnostics are reported, but these error summaries were not separately qualified against a precision threshold. The middle-90% endpoints have no separate precision claim.</p>
          <details><summary>Technical precision note</summary><p>{houseReference.precisionScope}</p></details>
          <p>Reference calculated {date(houseReference.referenceIssuedAt)} for this exact forecast edition. It contains no observed results or grade.</p>
          <ul>{houseReference.qualifications.map((note, i) => <li key={i}>{note}</li>)}</ul>
        </details>}
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
      <ElectionNightRubric standardSenate={!!senateScope} forecastCutoffAt={forecast.run.cutoffAt} evaluation={evaluation} />
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
