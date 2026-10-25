import {
  candidatePairComparisonReadiness, pairComparisonArms, pairComparisonArmLabels,
  type CandidatePairComparisonDraft, type PairArmView,
} from "../lib/candidate-pair-comparison-view.ts";

const percent = (value: number) => `${new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value * 100)}%`;
const shareMass = (value: number) => value === 0 ? "0%" : value === 1 ? "100%"
  : value < 0.001 ? "less than 0.1%" : value > 0.999 ? "more than 99.9%" : percent(value);
const loss = (value: number) => new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value * 100);

function Arm({ result, label }: { result: PairArmView; label: string }) {
  const d = result.distribution, e = result.evaluation;
  return <section className="count-quantity" aria-label={label}>
    <h4>{label}</h4>
    {d.status === "available" ? <>
      <span className="count-kind">Invented pair-share summary</span>
      <p className="count-value">{percent(d.summary.mean)} <span>average pair share (mean)</span></p>
      <p>Middle predicted share (median): {percent(d.summary.median)}.</p>
      <ul className="count-intervals">{d.summary.intervals.map(interval => <li key={interval.level}>
        {percent(interval.level)} predictive range: {percent(interval.lower)}–{percent(interval.upper)}
      </li>)}</ul>
    </> : <p className="count-unavailable"><strong>Pair share unavailable.</strong> {d.reason}
      {d.status === "undefined_share" ? <> The share cannot be calculated in {shareMass(d.undefinedMass)} of the predicted outcomes. Those outcomes stay in the comparison; the remaining outcomes do not get extra weight.</> : <> We do not know how often the share could be calculated. Missing does not mean a share of 0%.</>}
    </p>}
    <h5>Errors against the invented result</h5>
    {e.status === "scored" ? <dl className="count-outcomes">
      <div><dt>Full-distribution error (CRPS)</dt><dd style={{ whiteSpace: "nowrap" }}>{loss(e.score.crps)} points</dd></div>
      <div><dt>Range error (weighted interval score)</dt><dd style={{ whiteSpace: "nowrap" }}>{loss(e.score.weightedIntervalScore)} points</dd></div>
      <div><dt>Average-share miss</dt><dd style={{ whiteSpace: "nowrap" }}>{loss(e.score.absoluteMeanError)} points</dd></div>
    </dl> : <p><strong>Scores unavailable.</strong> {e.reason} No perfect score is substituted for missing results.</p>}
    <p className="small"><strong>Calculation precision unavailable.</strong> {result.precision.reason} A real simulation may give slightly different numbers when run again. Prediction ranges describe possible election outcomes, not that calculation error.</p>
  </section>;
}

/** Private invented-fixture component. No route, API, live input or owner DTO. */
export default function CandidatePairComparisonPreview({ view }: { view: CandidatePairComparisonDraft }) {
  const readiness = candidatePairComparisonReadiness(view);
  const banner = <div className="count-fixture-banner" role="note"><strong>Illustrative data only · Private comparison study</strong>
    <p>People, values and source records are invented. This preview is not connected to a real model or public forecast.</p></div>;
  if (!readiness.display) return <section className="count-preview" aria-label="Private candidate-pair comparison">
    {banner}<div className="count-warning"><strong>Comparison unavailable.</strong>
      <ul>{readiness.problems.map(problem => <li key={problem}>{problem}</li>)}</ul>
      <p>No pair-share estimates or scores are displayed for an incompatible draft.</p></div>
  </section>;
  const draft = readiness.view, c = readiness.coverage;
  const rows = new Map(draft.rows.map(row => [row.pairId, row]));
  return <section className="count-preview" aria-label={draft.label} style={{ overflowWrap: "anywhere" }}>
    {banner}
    <header><div className="eyebrow">Private frontend preparation</div><h2>{draft.label}</h2><p>{draft.provenance.label}</p></header>
    <div className="count-scope">
      <dl>
        <dt>Study status</dt><dd>Experimental: an interface example, not a tested real forecast</dd>
        <dt>Model checks</dt><dd>Not established here. These examples do not show that a real model passed its scientific or numerical checks.</dd>
        <dt>Ready for public use?</dt><dd>No. This study does not approve a forecast or change the public default.</dd>
        <dt>Training and input data</dt><dd>Different across these models</dd>
        <dt>Example input cutoff</dt><dd>{draft.binding.informationCutoff} · {draft.binding.horizonDays} days before the example election</dd>
      </dl>
      <p>{draft.qualification.explanation}</p>
      <p>Differences in errors cannot tell us what polling alone changed, because the models also differ in their data and assumptions. This example does not show which model would predict future elections better.</p>
    </div>
    <section className="count-section" aria-label="Fixed comparison coverage">
      <h3>Comparison coverage</h3>
      <p><strong>{c.fullyScoredPairs} of {c.expectedPairs} expected pairs have scores for all three models.</strong> Results are supplied for {c.suppliedPairs} of {c.expectedPairs} pairs.</p>
      <dl className="count-outcomes">{pairComparisonArms.map(arm => <div key={arm}><dt>{pairComparisonArmLabels[arm]}</dt><dd>{c.scoredByArm[arm]} of {c.expectedPairs} scored</dd></div>)}</dl>
      {c.missingPairIds.length > 0 && <p>Missing pairs remain in the expected total. No overall average error is shown using only the pairs that happened to have results.</p>}
      <p>This list does not cover all Senate contests. A candidate can lead this pair and still trail a third person. Pair shares do not establish who wins office or who controls the chamber.</p>
    </section>
    <section className="count-section" aria-label="How to read shares and errors">
      <h3>How to read these numbers</h3>
      <p>The mean is the average of possible pair shares, counting more likely outcomes more heavily. The median is the middle predicted share. Prediction ranges show how widely that share could vary.</p>
      <p><strong>Full-distribution error (CRPS)</strong> compares all the possible shares and the probabilities assigned to them with the observed share. <strong>Range error (weighted interval score)</strong> rewards narrower prediction ranges and penalizes ranges that miss the observed share. <strong>Average-share miss</strong> is the distance between the mean prediction and the observed share.</p>
      <p>All three errors are shown in percentage-point units. A mean prediction of 55% and an observed share of 52% differ by 3 percentage points. <strong>For all three, 0 is perfect and lower is better.</strong> The letters CRPS stand for continuous ranked probability score.</p>
    </section>
    {draft.requiredPairs.map(pair => {
      const row = rows.get(pair.id);
      return <article key={pair.id} className="count-contest" aria-label={`${pair.first.name} and ${pair.second.name}`}>
        <h3>{pair.first.name} and {pair.second.name}</h3>
        <p>Example election · {pair.electionDate}</p>
        <p className="count-denominator"><strong>Votes included in the share: {pair.first.name} plus {pair.second.name}.</strong> Each percentage divides {pair.first.name}’s votes by those two candidates’ combined votes. Other candidates stay outside this denominator. This is neither all-candidate share nor a probability of winning.</p>
        {!row ? <p className="count-unavailable"><strong>No result supplied for this expected pair.</strong> Its prediction and score are unavailable, not zero.</p> : <>
          <p>{row.observedFirstPairShare === null ? "Observed pair share unavailable in this invented example." : `Invented observed pair share: ${percent(row.observedFirstPairShare)}.`}</p>
          <div className="count-grid">{pairComparisonArms.map(arm => <Arm key={arm} result={row.arms[arm]} label={pairComparisonArmLabels[arm]} />)}</div>
        </>}
        <details className="count-checks"><summary>Candidate and election records · Invented</summary>
          <dl><dt>Race</dt><dd>{pair.raceId}</dd><dt>Physical event</dt><dd>{pair.eventId}</dd><dt>Election date</dt><dd>{pair.electionDate}</dd><dt>Contest</dt><dd>{pair.contestId}</dd>
            <dt>{pair.first.name}</dt><dd>Person {pair.first.personId}; source {pair.first.sourceCandidateId}; model {pair.first.modelCandidateId}; identity <code>{pair.first.identitySha256}</code></dd>
            <dt>{pair.second.name}</dt><dd>Person {pair.second.personId}; source {pair.second.sourceCandidateId}; model {pair.second.modelCandidateId}; identity <code>{pair.second.identitySha256}</code></dd>
          </dl>
        </details>
      </article>;
    })}
    <details className="count-checks"><summary>Technical source records · Invented declarations</summary>
      <p>These matching records test whether the example values stay linked to the same people and election. They are not evidence that real research data or candidate identities have been checked. The example format is not an owner-approved format for real results.</p>
      <dl><dt>Request</dt><dd><code>{draft.binding.requestSha256}</code></dd><dt>Comparison context</dt><dd><code>{draft.binding.comparisonContextSha256}</code></dd><dt>Pair specification</dt><dd><code>{draft.binding.pairSpecificationSha256}</code></dd>
        {pairComparisonArms.map(arm => <div key={arm}><dt>{pairComparisonArmLabels[arm]}</dt><dd>{draft.binding.sources[arm].id} · <code>{draft.binding.sources[arm].artifactSha256}</code></dd></div>)}
      </dl>
    </details>
  </section>;
}
