import type { HouseReviewedPublication } from "../lib/house-reviewed-catalog.ts";
import type { HouseGroupRaceScore, HouseGroupScoreStratum } from "../lib/house-group-result.ts";
import { summarizeHouseGroupResult } from "../lib/house-group-scoreboard.ts";

const stratumLabels: Record<HouseGroupScoreStratum, string> = {
  all_scored: "All scoreable districts",
  legacy_subset: "Districts also covered by earlier D/R share forecasts",
  newly_included: "Districts without earlier D/R share forecasts",
  competitive: "Competitive districts (earlier presidential share near 50%)",
  newly_included_competitive: "Newly included competitive districts",
};
const metric = (value: number | null | undefined) => value == null ? "—" : value.toFixed(3);
function difference(selected: number | null, compared: number | null) {
  if (selected === null || compared === null) return null;
  const delta = selected - compared;
  return `${delta > 0 ? "+" : ""}${delta.toFixed(3)} (selected minus compared)`;
}
function scoreText(score: HouseGroupRaceScore | undefined) { return score ? `${metric(score.categoricalBrier)} · ${score.n} districts` : "Not reported"; }
function nativeScore(result: HouseReviewedPublication["result"], subset: string) {
  if (!("polling" in result) || !result.polling) return null;
  const rows = result.nativeMetrics.race_scores;
  if (!Array.isArray(rows)) return null;
  const score = rows.find((row): row is Record<string, unknown> => !!row && typeof row === "object" &&
    !Array.isArray(row) && row.subset === subset);
  if (!score || typeof score.categorical_brier !== "number" || typeof score.n !== "number") return null;
  return { brier: score.categorical_brier, n: score.n };
}

export default function HouseGroupScoreboard({ publication, comparison }: { publication: HouseReviewedPublication; comparison?: HouseReviewedPublication | null }) {
  const result = publication.result, summary = summarizeHouseGroupResult(result);
  const reference = comparison ? summarizeHouseGroupResult(comparison.result) : null;
  const brierDifference = difference(summary.categoricalBrier, reference?.categoricalBrier ?? null);
  const polling = "polling" in result && result.polling ? result.pollingCoverage : null;
  const hasPollingComparison = !!comparison && (publication.entry.polling || comparison.entry.polling);
  const rows = hasPollingComparison
    ? result.raceScores.filter(score => ["all_scored", "legacy_subset", "newly_included"].includes(score.stratum))
    : result.raceScores;
  const polledScore = nativeScore(result, "polled_groups");
  const unpolledScore = nativeScore(result, "unpolled_groups");
  return <section className="historical-scoreboard house-group-scoreboard" aria-labelledby="house-group-scoreboard-title">
    <div className="scoreboard-heading"><div>
      <div className="eyebrow">{result.cycle} historical reconstruction · experimental research</div>
      <h2 className="section-title" id="house-group-scoreboard-title">How group forecasts matched past results</h2>
      <p>Target: which Democratic, Republican, or Other candidate group wins each district · research review complete</p>
    </div></div>
    <p className="scoreboard-context">These results use candidate lists and other information collected after the elections. They are reviewed research on past elections, not independent tests of forecasts made before election day.</p>
    <div className="scoreboard-cards house-scoreboard-cards">
      <article className="scoreboard-card">
        <h3>Unique favorites that won</h3>
        <div className="scoreboard-value">{summary.correctUniqueFavorites}<span> / {summary.uniqueFavorites}</span></div>
        <p>Among districts with one group at the highest chance. {summary.topChanceTies} exact top ties are excluded.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{reference.correctUniqueFavorites} / {reference.uniqueFavorites}</strong> unique favorites won; {reference.topChanceTies} exact ties excluded.</p>}
      </article>
      <article className="scoreboard-card">
        <h3>D/R/O group-win probability error</h3>
        <div className="scoreboard-value">{metric(summary.categoricalBrier)}</div>
        <p>This categorical Brier score checks the predicted chances for all three groups against the recorded winning group. Lower is better; 0 is perfect. It scores group wins, not D/R vote share or an individual candidate’s chance.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{metric(reference.categoricalBrier)}</strong>. Difference: {brierDifference}. Differences are descriptive; no precision claim is implied.</p>}
      </article>
      <article className="scoreboard-card">
        <h3>Scored coverage</h3>
        <div className="scoreboard-value">{summary.scoredRaces}<span> / {summary.forecastRaces}</span></div>
        <p>District forecasts with a recorded outcome that can be scored. {summary.unscoredRaces} {summary.unscoredRaces === 1 ? "district has" : "districts have"} no scoreable outcome.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{reference.scoredRaces} / {reference.forecastRaces}</strong> districts scored.</p>}
      </article>
      <article className="scoreboard-card">
        <h3>Local poll coverage</h3>
        {polling ? <>
          <div className="scoreboard-value">{polling.polledDistricts}<span> / {polling.totalDistricts}</span></div>
          <p>{polling.polls} selected polls were used. All 435 districts still receive a forecast; polls also inform a shared national estimate that can affect districts without local polls.</p>
        </> : <p>Polls are not included in this saved result.</p>}
      </article>
    </div>
    {summary.unscoredRaces > 0 && <p className="scoreboard-context">2018 NC-09 is unscored because the election was annulled and has no certified winner to compare with its probabilities.</p>}
    {comparison && <p className="scoreboard-context">Compared with {comparison.result.cycle} {comparison.entry.label}. Both results use the same D/R/O group-win target. Score differences describe these saved forecasts; they do not measure certainty or establish future accuracy.</p>}
    <details className="house-score-evidence">
      <summary>Group-win score by district group</summary>
      <p>Every score here checks which group won. “Earlier D/R share forecast” marks a subset with a separate two-party share estimate; it does not change what this table scores. These scores do not measure vote-share accuracy.</p>
      <div className="house-group-score-table-wrap" role="region" aria-label="Group-win scores by district category" tabIndex={0}><table className="house-group-score-table">
        <thead><tr><th scope="col">District group</th><th scope="col">Scored</th><th scope="col">Categorical Brier</th>{comparison && <><th scope="col">Compared</th><th scope="col">Difference</th></>}</tr></thead>
        <tbody>{rows.map(score => {
          const other = comparison?.result.raceScores.find(item => item.stratum === score.stratum);
          return <tr key={score.stratum}><th scope="row">{stratumLabels[score.stratum]}</th><td>{score.n}</td><td>{metric(score.categoricalBrier)}</td>{comparison && <><td>{scoreText(other)}</td><td>{difference(score.categoricalBrier, other?.categoricalBrier ?? null) || "—"}</td></>}</tr>;
        })}</tbody>
      </table></div>
      {hasPollingComparison
        ? <p>Only district groups with exactly the same members are compared here. The polling result does not report the same competitive-district breakdown, so those rows are unavailable for this comparison.</p>
        : <p>The first three rows distinguish districts with and without an earlier D/R share forecast. Competitive rows are subsets, not additional districts.</p>}
      {polling && (polledScore || unpolledScore) && <>
        <h3>Within the polling result: districts with and without local polls</h3>
        <p>These are separate district groups within this one result, not a before-and-after test of polling. The score remains the D/R/O group-win score.</p>
        <div className="house-group-score-table-wrap" role="region" aria-label="Group-win scores for locally polled and unpolled districts" tabIndex={0}><table className="house-group-score-table">
          <thead><tr><th scope="col">Districts in this result</th><th scope="col">Scored</th><th scope="col">Categorical Brier</th></tr></thead>
          <tbody>{[["With local polls", polledScore], ["Without local polls", unpolledScore]].map(([label, score]) => {
            const item = score as {brier:number;n:number}|null;
            return <tr key={label as string}><th scope="row">{label as string}</th><td>{item?.n ?? "—"}</td><td>{metric(item?.brier)}</td></tr>;
          })}</tbody>
        </table></div>
      </>}
    </details>
    <details className="house-score-evidence">
      <summary>Numerical review and model status</summary>
      <p>Research status: experimental. Author review: complete for the stated scope. Numerical checks for the district group probabilities passed. The largest estimated simulation sampling error was {(result.raceDiagnostics.maxGroupWinMcse * 100).toFixed(2)} percentage points. Sampling chains agreed (R-hat {result.raceDiagnostics.maxGroupWinRhat.toFixed(3)}; values near 1 indicate agreement). These checks describe the simulation calculations, not whether the forecasts are right.</p>
      <p>The precision check for adding expected seats across districts did not pass. This result does not report House control or a tipping district.</p>
    </details>
  </section>;
}
