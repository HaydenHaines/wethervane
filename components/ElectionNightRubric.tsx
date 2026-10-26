import { ELECTION_NIGHT_RUBRIC } from "../lib/election-night-rubric";

const percent = (value: number) => `${Math.round(value * 100)}%`;

/** The displayed thresholds come from the same frozen configuration as the scorer. */
export default function ElectionNightRubric({ standardSenate }: { standardSenate: boolean }) {
  const bands = ELECTION_NIGHT_RUBRIC.bands;
  return <section className="night-section night-rubric" aria-labelledby="night-rubric-title">
    <div className="eyebrow">The evaluation rubric</div><h3 id="night-rubric-title">How the forecast will be judged</h3>
    <p>Two tests: did the winner probabilities provide useful information, and did the margin ranges balance precision with misses?</p>
    <div className="night-grade-scale" aria-label="A through F evaluation scale; no grade assigned">{bands.map(band => <span key={band.grade}>{band.grade}</span>)}</div>
    <p><strong>No letter grade assigned.</strong> These are the selected grading rules. The final forecast edition, comparison forecast, eligible races and result definitions still need to be locked before election results are scored.</p>
    <dl><div><dt>Winner probabilities</dt><dd>Compare the chance assigned to each possible winner with what happened. Confident mistakes cost more. The score uses every declared outcome, including Other in the House.</dd></div>
      <div><dt>Margin ranges</dt><dd>Charge for the width of each 90% range and add a penalty when the result falls outside it. Making every range very wide cannot earn a good score merely by covering the results.</dd></div></dl>
    <p>Each test gets a grade based on how much its scoring penalty improves on the declared basic electoral-history forecast. Lower penalties are better. The overall grade is the lower of the two grades.</p>
    <table className="night-rubric-table">
      <caption>Grade for each test · improvement over the reference forecast</caption>
      <thead><tr><th scope="col">Grade</th><th scope="col">Scoring penalty compared with the reference</th></tr></thead>
      <tbody>{bands.map((band, index) => {
        const minimum = band.minimumSkill;
        const previous = index ? bands[index - 1].minimumSkill : null;
        const description = minimum === null ? `More than ${percent(Math.abs(previous!))} higher`
          : minimum < 0 ? `More than 0%, up to ${percent(Math.abs(minimum))} higher`
          : index === 0 ? `At least ${percent(minimum)} lower`
          : `${percent(minimum)} to less than ${percent(previous!)} lower`;
        return <tr key={band.grade}><th scope="row">{band.grade}</th><td>{description}</td></tr>;
      })}</tbody>
    </table>
    <p className="small">These letter boundaries are editorial choices, not universal scientific standards. The raw scores and differences will remain visible alongside any grade.</p>
    <p><strong>The 50/50 check:</strong> if the winner probabilities do not beat a fixed coin-flip forecast, the overall grade can be no better than D. An F stays an F. For House races that reference assigns 50% to Democratic, 50% to Republican and 0% to Other, and still counts Other wins.</p>
    {standardSenate && <div className="night-baseline-note"><p><strong>Planned Senate reference:</strong> Basic electoral history — October 20 edition (earlier issued). We will compare it with the October 26 forecast on the same results. The forecasts had different information and lead times; this does not isolate the value of polling.</p>
      <p>The primary Senate grade uses the fixed 32-race group explained above. Its collective error must be compared with the reference for those same 32 races. The all-35 forecast reference remains separate.</p></div>}
    <details className="night-rubric-details"><summary>Exact scores, coverage and exceptions</summary>
      <p>The winner penalty is the Brier score divided by two: square the difference between each predicted chance and its result (1 for the winner, 0 otherwise), add those values, then divide by two. Average across the fixed race list.</p>
      <p>The interval penalty is the 90% range’s width plus 20 times the distance of a result outside its nearest boundary, in percentage points. A result exactly on a boundary is inside. Average across the fixed list of races with a supported two-candidate margin.</p>
      <p>Improvement is 1 minus the model’s average penalty divided by the reference’s average penalty. If the reference penalty is zero, that percentage is undefined: show the raw difference and no component letter.</p>
      <p>A final letter requires final or certified results for every race in each graded list. Partial results can show a scoreboard and how many races are scored, but cannot earn a final letter. Called winners and unfinished vote counts stay distinct.</p>
      <p>House party-winner scores use Democratic, Republican and Other for every race. We also show a separate reference assigning one-third to each group. A missing named candidate remains a separate input failure, even if the correct party was predicted. Senate scores compare the two named principal contenders; that is not necessarily the full-ballot office winner.</p>
      <p>Beating the one-third reference can simply mean correctly favoring a Democratic or Republican winner. It does not necessarily show that the model can choose between those two parties. We report both reference scores, including all Other wins in each comparison.</p>
      <p>Where no supported margin list exists, only a clearly labeled winner-probability grade is possible. We do not invent margins or remove surprising winners to improve a grade.</p>
      <p>The size and direction of individual margin errors remain on the scoreboard. Shared-error diagnostics remain separate: one election can shift many races together, and the model’s own simulated coverage does not establish real-world accuracy.</p>
    </details>
    <p>A result having been possible does not excuse poor performance. One surprising result also does not settle whether the model is reliable across elections. Both the individual misses and the comparison scores need to stay visible.</p>
  </section>;
}
