import type { HouseGroupPublication } from "../lib/house-group-catalog.ts";
import type { HouseGroupRaceScore, HouseGroupScoreStratum } from "../lib/house-group-result.ts";
import { summarizeHouseGroupResult } from "../lib/house-group-scoreboard.ts";

const stratumLabels: Record<HouseGroupScoreStratum, string> = {
  all_scored: "All scored districts",
  legacy_subset: "Districts also covered by earlier D/R share forecasts",
  newly_included: "Districts without earlier D/R share forecasts",
  competitive: "Competitive districts",
  newly_included_competitive: "Newly included competitive districts",
};
const metric = (value: number | null | undefined) => value == null ? "—" : value.toFixed(3);
function difference(selected: number | null, compared: number | null) {
  if (selected === null || compared === null) return null;
  const delta = selected - compared;
  return `${delta > 0 ? "+" : ""}${delta.toFixed(3)} (selected minus compared)`;
}
function scoreText(score: HouseGroupRaceScore | undefined) { return score ? `${metric(score.categoricalBrier)} · ${score.n} districts` : "Not reported"; }

export default function HouseGroupScoreboard({ publication, comparison }: { publication: HouseGroupPublication; comparison?: HouseGroupPublication | null }) {
  const result = publication.result, summary = summarizeHouseGroupResult(result);
  const reference = comparison ? summarizeHouseGroupResult(comparison.result) : null;
  const brierDifference = difference(summary.categoricalBrier, reference?.categoricalBrier ?? null);
  const rows = result.raceScores;
  return <section className="historical-scoreboard house-group-scoreboard" aria-labelledby="house-group-scoreboard-title">
    <div className="scoreboard-heading"><div>
      <div className="eyebrow">{result.cycle} historical reconstruction · experimental research</div>
      <h2 className="section-title" id="house-group-scoreboard-title">How group forecasts matched past results</h2>
      <p>Which party or candidate group wins each district · research review complete</p>
    </div></div>
    <p className="scoreboard-context">These results use candidate lists and other information collected after the elections. They are reviewed research on past elections, not independent tests of forecasts made before election day.</p>
    <div className="scoreboard-cards house-scoreboard-cards">
      <article className="scoreboard-card">
        <h3>Unique favorites that won</h3>
        <div className="scoreboard-value">{summary.correctUniqueFavorites}<span> / {summary.uniqueFavorites}</span></div>
        <p>Counts districts with one group at the highest probability. {summary.topChanceTies} exact top ties are excluded.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{reference.correctUniqueFavorites} / {reference.uniqueFavorites}</strong> unique favorites won; {reference.topChanceTies} exact ties excluded.</p>}
      </article>
      <article className="scoreboard-card">
        <h3>D/R/O probability error</h3>
        <div className="scoreboard-value">{metric(summary.categoricalBrier)}</div>
        <p>Categorical Brier score is a measure of probability error across all three groups. Lower is better; 0 is perfect. It penalizes probabilities more when a confident forecast is wrong. This scores the winning group, a different outcome from earlier forecasts of D/R vote share or individual candidates.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{metric(reference.categoricalBrier)}</strong>. Difference: {brierDifference}. Differences are descriptive; no precision claim is implied.</p>}
      </article>
      <article className="scoreboard-card">
        <h3>Scored coverage</h3>
        <div className="scoreboard-value">{summary.scoredRaces}<span> / {summary.forecastRaces}</span></div>
        <p>Districts with an observed outcome that could be scored. {summary.unscoredRaces} forecast {summary.unscoredRaces === 1 ? "district has" : "districts have"} no scoreable outcome.</p>
        {reference && <p className="scoreboard-comparison">Compared result: <strong>{reference.scoredRaces} / {reference.forecastRaces}</strong> districts scored.</p>}
      </article>
    </div>
    {summary.unscoredRaces > 0 && <p className="scoreboard-context">2018 NC-09 is unscored because the election was annulled and this result has no certified winner to compare with its probabilities.</p>}
    {comparison && <p className="scoreboard-context">Compared with {comparison.result.cycle} {comparison.entry.label}. The compared result uses the same group-win target. Numeric differences are not evidence of meaningful precision or future accuracy.</p>}
    <details className="house-score-evidence">
      <summary>Score by district group</summary>
      <p>Categorical Brier scores compare D/R/O group probabilities with the recorded winning group. “Competitive” means the earlier presidential share was within five percentage points of 50%. These categories overlap; the last two rows are subsets and should not be added to the others.</p>
      <div className="house-group-score-table-wrap" role="region" aria-label="Group forecast scores by district category" tabIndex={0}><table className="house-group-score-table">
        <thead><tr><th scope="col">District group</th><th scope="col">Scored</th><th scope="col">Categorical Brier</th>{comparison && <><th scope="col">Compared</th><th scope="col">Difference</th></>}</tr></thead>
        <tbody>{rows.map(score => {
          const other = comparison?.result.raceScores.find(item => item.stratum === score.stratum);
          return <tr key={score.stratum}><th scope="row">{stratumLabels[score.stratum]}</th><td>{score.n}</td><td>{metric(score.categoricalBrier)}</td>{comparison && <><td>{scoreText(other)}</td><td>{difference(score.categoricalBrier, other?.categoricalBrier ?? null) || "—"}</td></>}</tr>;
        })}</tbody>
      </table></div>
      <p>The first three rows distinguish districts with and without an earlier D/R share forecast. Their scores here are all for the new D/R/O group-win target. Competitive rows are subsets.</p>
    </details>
    <details className="house-score-evidence">
      <summary>Numerical review and model status</summary>
      <p>Research status: experimental. Author review: complete for the stated scope. Numerical checks for the district group probabilities passed. The largest estimated simulation sampling error was {(result.raceDiagnostics.maxGroupWinMcse * 100).toFixed(2)} percentage points. Sampling chains agreed (R-hat {result.raceDiagnostics.maxGroupWinRhat.toFixed(3)}; values near 1 indicate agreement). These checks describe the simulation calculations, not whether the forecasts are right.</p>
      <p>The precision check for adding expected seats across districts did not pass. This result does not report House control or a tipping district.</p>
    </details>
  </section>;
}
