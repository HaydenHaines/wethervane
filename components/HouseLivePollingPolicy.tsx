"use client";

import { useState } from "react";
import type { HouseLivePollingRun } from "../lib/house-live-polling";
import type { HouseLiveRaceView, HouseLiveView } from "../lib/house-live-view";
import { dateLabel } from "../lib/presentation";

const REASONS: Record<string, string> = {
  not_one_current_house_district: "Not tied to one current House district",
  not_2026_house_regular_general: "A different election or contest",
  outside_100_day_field_end_window: "Fieldwork outside the 100 days before Election Day",
  unresolved_or_departed_candidate_in_question: "A candidate name could not be matched to the current roster",
  earlier_same_pollster_race_sample: "A newer eligible sample from the same pollster and district was kept",
  reallocated_or_later_round_not_initial_pair_share: "Later-round or reallocated results, rather than initial candidate choices",
  outside_named_principal_pair_support: "The model does not support the district’s named candidate pair",
  alternative_question_same_sample: "Another eligible question from the same sample was kept",
  unavailable_at_cutoff_under_created_plus_one_day_rule: "Estimated availability after the polling cutoff",
  missouri_fieldwork_before_final_old_map_confirmation: "Missouri fieldwork began before the final map confirmation",
  question_does_not_include_current_principal_pair: "The question omits one of the two modeled candidates",
  hypothetical_question: "A hypothetical candidate matchup",
  tracking_sample_overlap: "A tracking poll excluded to avoid sample overlap",
  unsupported_population_or_subgroup: "An unsupported respondent group or subgroup",
  invalid_dates_or_election: "Missing or inconsistent fieldwork or election dates",
  invalid_nominal_sample_size: "An invalid reported sample size",
  duplicate_candidate_in_question: "A candidate appears more than once in the question",
  invalid_toplines_or_denominator: "Unusable candidate percentages or two-candidate total",
};
const reasonLabel = (reason: string) => reason.startsWith("conflicting_question_metadata:")
  ? "Conflicting information within the same question" : REASONS[reason] ?? "Selection reason unavailable";
const populationLabel = (population: string) => ({ LV: "Likely voters", RV: "Registered voters", A: "Adults" }[population] ?? population);

function DistrictPollRecords({ polling, race }: { polling: HouseLivePollingRun; race: HouseLiveRaceView }) {
  const [showAll, setShowAll] = useState(false);
  const included = polling.usedQuestions.filter(question => question.raceId === race.id);
  const excluded = polling.excluded.filter(question => question.districtKey === race.district.districtKey);
  const visible = showAll ? excluded : excluded.slice(0, 10);
  return <>
    {!polling.modelUsesSelectedQuestions ? <p>Polling is switched off for this model. No poll questions are used for {race.seatId}.</p>
      : included.length ? <p><strong>{included.length} selected {included.length === 1 ? "question result" : "question results"}</strong> for {race.seatId}.</p>
        : <p>No eligible local poll question was selected for {race.seatId}. This does not mean no poll was conducted. The district still receives a forecast and can be affected by polling information shared across districts.</p>}
    {included.length > 0 && <ul className="poll-record-list" aria-label={`Included poll questions for ${race.seatId}`}>
      {included.map(question => <li key={question.observationId}>
        <strong>{question.pollster}</strong><span className="poll-record-status included">Included</span>
        <p>Fieldwork: <time dateTime={question.startDate}>{dateLabel(question.startDate)}</time>–<time dateTime={question.fieldEnd}>{dateLabel(question.fieldEnd)}</time>.</p>
        <p>{populationLabel(question.population)} · {question.sampleSize === null ? "Sample size not reported" : `${question.sampleSize.toLocaleString()} respondents reported`}.</p>
        <p>Estimated availability: {dateLabel(question.availableAt)}. This is not a verified release date.</p>
        <p>{question.sourceUrls.map((url, index) => <span key={url}>{index > 0 && " · "}<a href={url} target="_blank" rel="noreferrer">{index === 0 ? "Poll source" : `Additional source ${index + 1}`}</a></span>)}</p>
      </li>)}
    </ul>}
    <details>
      <summary>{excluded.length} source question records left out for {race.seatId}</summary>
      <p>These records are tied to this district, but may concern a primary or another election. Each shows the first selection rule it failed. They are not all current general-election polls rejected for quality.</p>
      {excluded.length ? <>
        <ul className="poll-record-list" aria-label={`Excluded source questions for ${race.seatId}`}>
          {visible.map((question, index) => <li key={index}>
            <strong>{question.pollster}</strong><span className="poll-record-status">Left out</span>
            <p>Reported fieldwork end: {question.endDate || "Not reported"}.</p>
            <p>{reasonLabel(question.reason)}.</p>
          </li>)}
        </ul>
        {excluded.length > 10 && <button className="house-poll-expand" onClick={() => setShowAll(!showAll)} aria-expanded={showAll}>
          {showAll ? "Show first 10 records" : `See all ${excluded.length} records`}
        </button>}
      </> : <p>No district-specific exclusions were recorded in this source collection. That does not establish that every poll was found or reviewed.</p>}
    </details>
  </>;
}

export default function HouseLivePollingPolicy({ polling, view, selected, onRaceSelect }: {
  polling?: HouseLivePollingRun | null;
  view: HouseLiveView;
  selected: HouseLiveRaceView;
  onRaceSelect: (id: string) => void;
}) {
  const enabled = view.run.configuration.polling === "standard";
  const reasons = new Map<string, number>();
  for (const row of polling?.excludedReasonCounts ?? []) reasons.set(row.reason, (reasons.get(row.reason) ?? 0) + row.count);
  return <section className="polling-policy house-live-polling" id="house-live-polling" aria-labelledby="house-live-polling-title">
    <div className="eyebrow">Polling inputs</div>
    <h2 id="house-live-polling-title" className="section-title">Which polls are included?</h2>
    {!enabled ? <p className="house-poll-summary"><strong>Polling is switched off.</strong> This model uses no current poll measurements.</p>
      : polling ? <p className="house-poll-summary">This model uses <strong>{polling.usedQuestions.length.toLocaleString()} selected poll-question results</strong> across <strong>{polling.selectedPoolDistrictCount} of {view.races.length} districts</strong>.</p>
        : <p className="house-poll-summary">Polling is enabled. The selection details have not been published for this result.</p>}
    {!polling ? <p>District-level inclusion and exclusion records are unavailable here. A missing record does not establish that no polls were conducted.</p> : <>
      <p>Each result is one question used for one district. A survey covering several districts can supply several results; this is not a count of independent surveys.</p>
      {!enabled && <p>The eligible source collection contains {polling.selectedPoolQuestionCount.toLocaleString()} question results across {polling.selectedPoolDistrictCount} districts. They are not used by this model.</p>}
      <details>
        <summary>How polls are chosen</summary>
        <div className="poll-rule-grid">
          <div><h3>The right contest</h3><p>Questions must match the current district and named candidates for the November general election. Fieldwork must end within the final 100 days before Election Day. Hypothetical matchups, subgroup samples and later ranked-choice rounds are left out.</p></div>
          <div><h3>Avoid counting repeats</h3><p>Keep one eligible question per sample and district, preferring likely voters, then registered voters, then adults. When more than one eligible sample from a pollster covers a district, keep the latest fieldwork.</p></div>
          <div><h3>Available by the cutoff</h3><p>Polling cutoff: <time dateTime={polling.cutoffs.pollingAt}>{new Date(polling.cutoffs.pollingAt).toLocaleString("en-US", { timeZone: "UTC", dateStyle: "long", timeStyle: "short" })} UTC</time>. Estimated availability is the later of fieldwork ending and the day after the aggregator recorded the poll. The original publication time is not verified.</p></div>
        </div>
        <p>The source collection comes from NYT poll-question records. VoteHub is used as a coverage cross-check; matching surveys are not added a second time. This is not an exhaustive census of every poll or an audit of every questionnaire.</p>
        <p>The poll calculation compares the two named principal candidates. Each candidate’s share is measured among respondents choosing either of those two; other responses are left out of that denominator. This share is separate from the chance of winning the race.</p>
        <p>Missing sample sizes stay missing. Reported respondent counts are not added as a separate model weight: the error model is calibrated on historical polling errors, including sampling error.</p>
        <p>Polls can share errors across the country, within a district, from the same pollster, and across nearby fieldwork dates. The model accounts for these relationships. Unknown overlap among respondents can remain.</p>
      </details>
      <details className="poll-records">
        <summary>Poll details for {selected.seatId}</summary>
        <label htmlFor="house-live-poll-district">District to inspect</label>
        <select id="house-live-poll-district" value={selected.id} onChange={event => onRaceSelect(event.target.value)}>
          {view.races.map(race => <option key={race.id} value={race.id}>{race.seatId}</option>)}
        </select>
        <DistrictPollRecords key={`${view.run.resultSha256}:${selected.id}`} polling={polling} race={selected} />
      </details>
      <details>
        <summary>Why {polling.excluded.length.toLocaleString()} source question records were left out</summary>
        <p>The broader source inventory also includes other elections and records without a current district. These counts describe source filtering, not a number of current House polls rejected for quality. Each question is counted once under its first failing rule; repeated source rows within a question are not counted here.</p>
        <div className="house-poll-reasons" role="region" aria-label="Reasons source questions were left out" tabIndex={0}>
          <table><thead><tr><th scope="col">First reason</th><th scope="col">Question records</th></tr></thead>
            <tbody>{[...reasons].sort((a, b) => b[1] - a[1]).map(([reason, count]) => <tr key={reason}><th scope="row">{reasonLabel(reason)}</th><td>{count.toLocaleString()}</td></tr>)}</tbody>
          </table>
        </div>
      </details>
    </>}
  </section>;
}
