"use client";

import { useState } from "react";
import type { ElectionNightEvaluationRowView, ElectionNightEvaluationView } from "../lib/election-night-evaluation-view";
import { probability } from "../lib/presentation";

const date = (value: string) => new Date(value).toUTCString();

function pairPrediction(row: ElectionNightEvaluationRowView): string {
  const pair = row.forecast.pair;
  if (!pair) return "No supported pair margin";
  if (Math.abs(pair.meanMarginPp) < .05) return "Average named-pair margin rounded to 0.0 points";
  return `${pair.meanMarginPp > 0 ? pair.firstCandidateName : pair.secondCandidateName} by ${Math.abs(pair.meanMarginPp).toFixed(1)} points`;
}

function observed(row: ElectionNightEvaluationRowView): string {
  const value = row.winner.model.observation;
  if (!value) return "No verified winner result";
  const parties: Record<string, string> = { D: "Democratic", R: "Republican", O: "Other" };
  const label = value.label ?? (row.winner.model.observedValueKind === "party_group" && typeof value.value === "string"
    ? parties[value.value] ?? "Reported winner group" : "Reported pair leader");
  return `${label} · ${value.status}`;
}

function margin(row: ElectionNightEvaluationRowView): string {
  const value = row.margin.model.observation;
  if (!value || typeof value.value !== "number" || !row.forecast.pair) return "No verified pair margin";
  const pair = row.forecast.pair;
  const status = value.status === "called" ? "preliminary" : value.status;
  if (Math.abs(value.value) < .05) return `Margin rounds to 0.0 points · ${status}`;
  const lead = value.value > 0 ? pair.firstCandidateName : value.value < 0 ? pair.secondCandidateName : null;
  return `${lead ? `${lead} by ${Math.abs(value.value).toFixed(1)}` : "Tied at 0.0"} points · ${status}`;
}

function marginError(row: ElectionNightEvaluationRowView): string {
  const score = row.margin.model.marginScore;
  if (!score || (row.margin.model.status !== "scored" && !(row.displayOnly && row.margin.model.status === "not_in_grade_cohort"))) return "Not scored";
  return `${score.absoluteErrorPp.toFixed(1)} points from the ${score.pointRule === "mean" ? "average" : "median"}; ${score.interval90.covered ? "inside" : "outside"} the 90% range${row.displayOnly ? " · tracked separately" : ""}`;
}

/** The locked evaluation is deliberately separate from the changing current forecast. */
export default function ElectionNightEvaluation({ view, currentResultSha256 }: {
  view: ElectionNightEvaluationView | null; currentResultSha256: string;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!view || !view.lock || !view.ledger || !view.coverage || !view.availableMatchedRows || view.status === "unavailable") return null;
  const recorded = view.rows.filter(row => row.sources.length > 0);
  const shown = expanded ? recorded : recorded.slice(0, 10);
  const matchingCurrent = view.lock.model.resultSha256 === currentResultSha256;
  const winner = view.availableMatchedRows.winner, interval = view.availableMatchedRows.interval;
  const cohort = new Set(view.lock.cohort.winnerRaceIds);
  const scoredRows = view.rows.filter(row => cohort.has(row.raceId) && row.winner.model.status === "scored" &&
    row.winner.model.modalPickCorrect !== null);
  const correct = scoredRows.filter(row => row.winner.model.modalPickCorrect).length;
  const scored = scoredRows.length;
  return <section className="night-evaluation" aria-label="Evaluation of the locked forecast">
    <h4>Evaluation of the locked forecast</h4>
    <p>Forecast frozen {date(view.lock.frozenAt)}; inputs through {date(view.lock.model.cutoffAt)}.
      {matchingCurrent ? " This is the forecast currently shown above." : " The forecast shown above is a different edition; these results and scores belong to the locked edition."}</p>
    <p>Comparison: {view.lock.comparison.label}. {view.lock.comparison.mode === "explicit_earlier_benchmark"
      ? "The reference was issued earlier, so this comparison does not isolate the effect of polling."
      : "The comparison uses the declared matching forecast cutoff."}</p>
    <p><strong>{view.coverage.winner.final} of {view.coverage.winner.total}</strong> graded winner results and <strong>{view.coverage.interval.final} of {view.coverage.interval.total}</strong> graded margin results are final or certified. Missing and provisional results remain in those totals.</p>
    {scored > 0 && <p>For {scored} winner results eligible for scoring so far, the locked forecast&apos;s top {view.chamber === "house" ? "party or group" : "named candidate"} was correct in {correct}. This count alone does not evaluate the probabilities.</p>}
    {(winner.count > 0 || interval.count > 0) && <details><summary>Scores for the results available so far</summary>
      <p>These averages cover only results eligible for both forecasts. They are raw scores, not the letter grade; lower penalties are better.</p>
      {winner.count > 0 && winner.modelMeanLoss !== null && winner.baselineMeanLoss !== null && <p>Winner probability penalty, {winner.count} races: {winner.modelMeanLoss.toFixed(3)} for the locked forecast and {winner.baselineMeanLoss.toFixed(3)} for the reference.</p>}
      {interval.count > 0 && interval.modelMeanLoss !== null && interval.baselineMeanLoss !== null && <p>Margin-range penalty, {interval.count} races: {interval.modelMeanLoss.toFixed(1)} points for the locked forecast and {interval.baselineMeanLoss.toFixed(1)} points for the reference.</p>}
      {(winner.provisionalCount > 0 || interval.provisionalCount > 0) && <p>Some eligible scores use provisional or called results; they may change.</p>}
    </details>}
    {recorded.length > 0 && <details><summary>Results reported for the locked forecast ({recorded.length})</summary>
      <p>Winner and margin status are reported separately. A call or provisional count can change and does not establish a final margin.</p>
      <p className="small">On a narrow screen, scroll the table sideways to see all columns.</p>
      <div className="house-group-score-table-wrap" role="region" aria-label="Locked forecast results" tabIndex={0}>
        <table className="house-group-score-table"><thead><tr><th scope="col">Race</th><th scope="col">Locked top choice</th><th scope="col">Locked average pair margin</th><th scope="col">Winner or pair leader</th><th scope="col">Named-pair result margin</th><th scope="col">Margin error</th></tr></thead>
          <tbody>{shown.map(row => <tr key={row.raceId}><th scope="row">{row.forecast.seatId}{row.displayOnly && <><br /><small>Outside primary grade</small></>}</th>
            <td>{row.forecast.modalPick.name}{row.forecast.modalPick.kind === "group" ? " (group)" : ""} · {probability(row.forecast.modalPick.probability)}</td>
            <td>{pairPrediction(row)}</td><td>{observed(row)}</td><td>{margin(row)}</td><td>{marginError(row)}</td></tr>)}</tbody>
        </table>
      </div>
      {recorded.length > 10 && <button type="button" onClick={() => setExpanded(value => !value)}>{expanded ? "Show fewer results" : `See all ${recorded.length} results`}</button>}
      <p className="small">Margins compare the same two named candidates and exclude other votes. Scores use the admitted result version; corrections may change them.</p>
      <ul>{shown.flatMap(row => row.sources.map(source => <li key={`${row.raceId}:${source.recordSha256}`}>
        {row.forecast.seatId}: <a href={source.sourceUrl} target="_blank" rel="noreferrer">result source</a> · {date(source.publishedAt)}
      </li>))}</ul>
    </details>}
    {!recorded.length && <p>No verified results have been added for this evaluation yet.</p>}
  </section>;
}
