"use client";

import type { HouseLiveComparison, HouseLiveComparisonValue } from "../lib/house-live-comparison";
import type { HouseLiveCatalogView } from "../lib/house-live-view";
import { houseLiveCutoffRole } from "../lib/house-live-copy";

const GROUP_NAMES = { D: "Democratic", R: "Republican", O: "Other or independent" } as const;
const SEAT_NAMES = { D: "Democratic", R: "Republican", U: "Other seats, unassigned" } as const;
const SLOT_NAMES = { base: "electoral history", polling: "polling", uncertainty: "uncertainty" } as const;
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const points = (value: number) => `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)} percentage points`;
const seats = (value: number) => value.toFixed(1);
const seatChange = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)} seats`;
const count = (value: number) => new Intl.NumberFormat("en-US").format(value);
const time = (value: string) => new Intl.DateTimeFormat("en-US", {
  month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  timeZone: "UTC", timeZoneName: "short",
}).format(new Date(value));

export default function HouseLiveComparison({ comparison, selectedRaceId, catalog, selectedRunId,
  comparisonId, busy, error, onSelect, onRetry }: {
  comparison: HouseLiveComparison | null;
  selectedRaceId: string;
  catalog: HouseLiveCatalogView;
  selectedRunId: string;
  comparisonId: string | null;
  busy: boolean;
  error: string;
  onSelect: (id: string | null) => void;
  onRetry: () => void;
}) {
  const choices = catalog.entries.filter(entry => entry.id !== selectedRunId);
  const knownChoice = choices.some(entry => entry.id === comparisonId);
  // A reference for a previous selected run must never be shown under the new run.
  const candidateSummary = comparison?.status === "comparable" ? comparison.summary : null;
  const selectedEntry = catalog.entries.find(entry => entry.id === selectedRunId);
  const referenceEntry = catalog.entries.find(entry => entry.id === candidateSummary?.reference.id);
  const summary = comparisonId && candidateSummary?.primary.id === selectedRunId &&
    candidateSummary.primary.resultSha256 === selectedEntry?.resultSha256 &&
    candidateSummary.reference.resultSha256 === referenceEntry?.resultSha256
    ? candidateSummary : null;
  const showingPrior = !!summary && summary.reference.id !== comparisonId;
  const chamber = summary?.chamber.status === "comparable" ? summary.chamber : null;
  const race = summary?.races.find(item => item.raceId === selectedRaceId);
  const changedPolls = summary?.contextDifferences.pollCounts.find(item => item.raceId === selectedRaceId);
  const changedMethod = summary?.contextDifferences.methods.find(item => item.raceId === selectedRaceId);

  return <section className="house-comparison-note" aria-label="Current House model comparison">
    <div className="house-comparison-controls stack-field">
      <label htmlFor="house-live-comparison">Compare with</label>
      <select id="house-live-comparison" value={comparisonId ?? ""} disabled={busy}
        aria-describedby="house-live-comparison-help"
        onChange={event => onSelect(event.target.value || null)}>
        <option value="">No comparison</option>
        {comparisonId && !knownChoice && <option value={comparisonId} disabled>Unavailable model</option>}
        {choices.map(entry => <option key={entry.id} value={entry.id}>{entry.label}</option>)}
      </select>
      <p id="house-live-comparison-help">Compare two current forecasts for the same districts and forecast time.</p>
      {comparisonId && <button type="button" disabled={busy} onClick={() => onSelect(null)}>Remove comparison</button>}
    </div>

    {busy && <div className="compare-banner" role="status">
      <strong>Loading comparison…</strong>
      <span>{summary ? `The last verified comparison with ${summary.reference.label} remains visible until the new model is verified.`
        : "The selected forecast remains visible while the comparison is verified."}</span>
    </div>}
    {error && <div className="compare-banner" role="alert">
      <strong>Comparison unavailable</strong>
      <span>{error} {summary ? "The last verified comparison remains visible." : "The selected forecast remains visible."}</span>
      <button type="button" onClick={onRetry}>Try comparison again</button>
    </div>}
    {comparison?.status === "incomparable" && !busy && !error && comparisonId &&
      <div className="compare-banner" role="alert"><strong>These forecasts cannot be compared</strong>
        <span>{comparison.reason}</span></div>}

    {summary && <>
      <h2>{summary.primary.label} compared with {summary.reference.label}</h2>
      {showingPrior && <p role="status">Showing the last verified comparison; the requested model has not replaced it.</p>}
      <p>Changes below are the selected forecast minus the compared forecast. The 2026 election has not happened, so these are differences between forecasts, not accuracy scores or evidence that one model is better.</p>
      <p>Both forecasts use the same election date, district map, candidate groups, election rules, and forecast cutoff ({time(summary.primary.cutoffAt)}).</p>
      <p className="small">On small screens, scroll the tables sideways to see both forecasts and their differences.</p>

      {chamber ? <>
        <h3>Whole House</h3>
        <p>Expected seats are the average across the model&apos;s simulated elections. A party needs {chamber.threshold} seats for a majority. Other seats remain unassigned to either party.</p>
        <div className="house-group-score-table-wrap" role="region" aria-label="Expected House seats comparison" tabIndex={0}>
        <table className="house-group-score-table"><caption>Expected seats by winner group</caption>
          <thead><tr><th scope="col">Seats</th><th scope="col">Selected</th><th scope="col">Compared</th><th scope="col">Change</th></tr></thead>
          <tbody>{(["D", "R", "U"] as const).map(party => {
            const item = chamber.expectedSeats[party];
            return <tr key={party}><th scope="row">{SEAT_NAMES[party]}</th>
              <td>{seats(item.primary)}</td><td>{seats(item.reference)}</td><td>{seatChange(item.difference)}</td></tr>;
          })}</tbody>
        </table>
        </div>
        <div className="house-group-score-table-wrap" role="region" aria-label="House control chance comparison" tabIndex={0}>
        <table className="house-group-score-table"><caption>Chance of a House majority or unresolved control</caption>
          <thead><tr><th scope="col">Outcome</th><th scope="col">Selected</th><th scope="col">Compared</th><th scope="col">Change</th></tr></thead>
          <tbody>{([
            ["D", "Democratic majority"], ["R", "Republican majority"],
            ["unresolved", "Neither party has a majority"],
          ] as const).map(([party, label]) => {
            const item: HouseLiveComparisonValue = chamber.control[party];
            return <tr key={party}><th scope="row">{label}</th>
              <td>{percent(item.primary)}</td><td>{percent(item.reference)}</td><td>{points(item.difference)}</td></tr>;
          })}</tbody>
        </table>
        </div>
      </> : <p>Whole House comparison unavailable: {summary.chamber.status === "unavailable" ? summary.chamber.reason : "No matching chamber result."}</p>}

      <h3>Selected district</h3>
      {race ? <div className="house-group-score-table-wrap" role="region" aria-label={`${race.districtKey} winner-group chance comparison`} tabIndex={0}>
      <table className="house-group-score-table"><caption>{race.districtKey} winner-group chances</caption>
        <thead><tr><th scope="col">Winner group</th><th scope="col">Selected</th><th scope="col">Compared</th><th scope="col">Change</th></tr></thead>
        <tbody>{race.outcomes.map(outcome => <tr key={outcome.party}>
          <th scope="row">{outcome.supported ? outcome.label : `${GROUP_NAMES[outcome.party]} (no listed contender)`}</th>
          <td>{percent(outcome.probability.primary)}</td><td>{percent(outcome.probability.reference)}</td>
          <td>{points(outcome.probability.difference)}</td>
        </tr>)}</tbody>
      </table></div> : <p>The selected district is not in this verified comparison.</p>}
      <p>These are chances that a candidate group wins the district. Candidate vote shares use a separate definition and are not compared here.</p>

      <h3>What differs between the models</h3>
      <ul>
        <li>Model choices: {summary.contextDifferences.modelSlots.length
          ? summary.contextDifferences.modelSlots.map(slot => SLOT_NAMES[slot]).join(", ")
          : "the listed model choices are the same"}.</li>
        <li>District polling inputs, summed across all races: {summary.primary.totalPollCount} selected; {summary.reference.totalPollCount} compared. A poll used in multiple districts can contribute in each district.
          {summary.contextDifferences.pollCounts.length > 0 && ` Counts differ in ${summary.contextDifferences.pollCounts.length} districts.`}
          {changedPolls && ` In ${race?.districtKey ?? "the selected district"}: ${changedPolls.primary} selected; ${changedPolls.reference} compared.`}</li>
        <li>Source cutoff times: {summary.contextDifferences.sourceCutoffsDiffer ? "different" : "the same"}.
          {summary.contextDifferences.sourceCutoffsDiffer && <details><summary>See each model&apos;s source cutoffs</summary>
            <p>Selected: {summary.primary.sourceCutoffs.map(item => `${houseLiveCutoffRole(item.role)}: ${time(item.at)}`).join("; ")}.</p>
            <p>Compared: {summary.reference.sourceCutoffs.map(item => `${houseLiveCutoffRole(item.role)}: ${time(item.at)}`).join("; ")}.</p>
          </details>}</li>
        <li>District calculation methods: {summary.contextDifferences.methods.length
          ? `different in ${summary.contextDifferences.methods.length} districts` : "the same"}.
          {changedMethod && <details><summary>Method in {race?.districtKey ?? "the selected district"}</summary>
            <p>Selected: {changedMethod.primary}</p><p>Compared: {changedMethod.reference}</p>
          </details>}</li>
      </ul>
      <details><summary>Evidence for each forecast</summary>
        <p>A file fingerprint identifies the exact saved set of simulated elections; it does not measure forecast quality.</p>
        <p>Selected: {count(summary.primary.jointDraws.count)} complete simulated House outcomes; evidence {summary.primary.jointDraws.sha256.slice(0, 12)}…</p>
        <p>Compared: {count(summary.reference.jointDraws.count)} complete simulated House outcomes; evidence {summary.reference.jointDraws.sha256.slice(0, 12)}…</p>
        <p>The two sets of simulated outcomes are separate. This comparison does not calculate a range for their difference.</p>
      </details>
    </>}
  </section>;
}
