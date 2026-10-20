import { countViewReadiness, type CountQuantity, type CountViewDraft, type CountViewReadiness } from "../lib/count-view.ts";

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const percentage = (value: number) => `${number.format(value * 100)}%`;
const formatted = (value: number, quantity: CountQuantity) => quantity.unit === "ratio" ? percentage(value) : number.format(value);
const lifecycleLabel = {
  ready: "Saved fixture result",
  requires_computation: "A new calculation is required; none is scheduled",
  incompatible: "This configuration is unsupported in the fixture",
  failed: "The fixture calculation failed",
};

function Quantity({ quantity, readiness, label }: {
  quantity: CountQuantity | undefined; readiness: CountViewReadiness; label?: string;
}) {
  if (!quantity) return <div className="count-quantity"><h4>{label || "Quantity"}</h4><p>Not available: this quantity was not supplied.</p></div>;
  const gate = readiness.quantityGates[quantity.id];
  return <div className="count-quantity" data-quantity={quantity.id}>
    <h4>{label || quantity.label}</h4>
    {quantity.kind === "unavailable" || !gate?.display ? <p className="count-unavailable"><strong>Not available.</strong> {quantity.kind === "unavailable" ? quantity.reason : gate?.reason}</p> : <>
      <span className="count-kind">{quantity.kind === "observed_administrative" ? "Reported observation" : "Model estimate"}</span>
      <p className="count-value">{formatted(quantity.kind === "observed_administrative" ? quantity.value : quantity.mean, quantity)}{quantity.unit !== "ratio" && <span> {quantity.unit}</span>}</p>
      {quantity.kind === "backend_weighted_predictive" ? <>
        <p className="small">Average across predicted outcomes, allowing some outcomes to be more likely than others.</p>
        <ul className="count-intervals">{quantity.intervals.map((interval) => <li key={interval.level}>{percentage(interval.level)} predicted range: {formatted(interval.lower, quantity)}–{formatted(interval.upper, quantity)}</li>)}</ul>
      </> : <p className="small">Reference date: {quantity.referenceAt}<br />Published: {quantity.publicationAt}<br />Source qualification: {quantity.sourceQualification}<br />This report is an observation about participation, with its own measurement error.</p>}
    </>}
    {quantity.denominator.kind === "quantity" && <p className="count-denominator"><strong>Denominator: {quantity.denominator.label}.</strong> {quantity.denominator.interpretation}</p>}
  </div>;
}

/** Private fixture component. No production page or route imports this preview.
 * A future public view needs a separately reviewed, source-bound adapter.
 */
export default function CountForecastPreview({ view }: { view: CountViewDraft }) {
  const readiness = countViewReadiness(view);
  const quantities = new Map(view.quantities.map((quantity) => [quantity.id, quantity]));
  return <section className="count-preview" aria-label="Illustrative count and participation preview">
    <div className="count-fixture-banner" role="note"><strong>Illustrative data only — no public count forecast</strong><p>{view.provenance.label}. These values and qualification declarations are invented to test the interface.</p></div>
    <header><div className="eyebrow">Frontend preparation</div><h2>Population, participation & votes</h2><p>These are different quantities. Each keeps its own units, denominator and uncertainty.</p></header>
    <div className="count-scope">
      <strong>{view.scope.label}{readiness.missingEventIds.length || readiness.missingContestIds.length ? " · Partial coverage" : ""}</strong>
      <dl><dt>Physical election events supplied</dt><dd>{view.events.filter((event) => view.scope.requiredEventIds.includes(event.id)).length} of {view.scope.requiredEventIds.length} required in this fixture</dd>
        <dt>Applicable contests supplied</dt><dd>{view.contests.filter((contest) => contest.applicability === "present" && view.scope.requiredContestIds.includes(contest.id)).length}; {readiness.missingContestIds.length} required contest identities are missing</dd>
        <dt>Input cutoff</dt><dd>{view.bindings.informationCutoff}</dd>
        <dt>Calculation state</dt><dd>{lifecycleLabel[view.lifecycle]}</dd>
      </dl>
      {!readiness.canShowControl && <p><strong>No full-Senate control estimate is available.</strong> A partial result cannot establish the outcome of the whole chamber.</p>}
      <p>Calculation status is separate from evidence about sources, election rules and forecast accuracy.</p>
      <p><strong>Fixture evidence declaration:</strong> {view.audit.qualification}</p>
    </div>
    {readiness.problems.some((problem) => problem.code === "parent_mass") && <div className="count-warning" role="alert"><strong>Incomplete joint result: some population states failed or are missing.</strong><p>The predicted summaries are withheld. A population state is one possible population used by the model; its probability weight cannot be discarded or redistributed when a calculation fails. Separately supplied administrative reports keep their reported labels.</p></div>}
    {readiness.problems.length > 0 && <details className="count-checks"><summary>Why some results are unavailable</summary><ul>{readiness.problems.map((problem, index) => <li key={`${problem.code}-${index}`}>{problem.message}</li>)}</ul><p>These fixture checks test presentation boundaries. They do not verify an actual scientific result or the underlying simulation rows.</p></details>}

    <section className="count-section" aria-labelledby="count-population-title"><h3 id="count-population-title">1. People represented</h3>
      <p>Adult residents include noncitizens. Citizen-adult exposure is the uncertain population used by the working model; it is not an exact count of legal eligibility or a hard cap on participation.</p>
      <div className="count-grid">{view.quantities.filter((quantity) => ["adult_residents", "citizen_adult_exposure"].includes(quantity.quantityType)).map((quantity) => <Quantity key={quantity.id} quantity={quantity} readiness={readiness} label={quantity.quantityType === "adult_residents" ? "Adult residents" : "Citizen adults used as population exposure"} />)}</div>
      <details><summary>Which population and community groups are represented?</summary>
        <p><strong>{view.scope.populationCells.label}.</strong> {view.scope.populationCells.dimensions.join("; ")}. {view.scope.populationCells.limitations}</p>
        <p><strong>{view.scope.communityProfiles.label}.</strong> {view.scope.communityProfiles.dimensions.join("; ")}. {view.scope.communityProfiles.limitations}</p>
      </details>
    </section>

    <section className="count-section" aria-labelledby="count-events-title"><h3 id="count-events-title">2. Participation in one election event</h3>
      <p>People taking part in simultaneous contests share one physical election event. Adding those contests' vote totals would count some people more than once. A later runoff is a different event.</p>
      {view.events.map((event) => <article key={event.id} className="count-event" aria-label={`${event.state} election event on ${event.date}`}>
        <h4>{event.state} · {event.date}</h4>
        <div className="count-grid">
          <Quantity quantity={quantities.get(event.participantQuantityId)} readiness={readiness} label="People participating in this event" />
          {event.administrativeReportQuantityId && <Quantity quantity={quantities.get(event.administrativeReportQuantityId)} readiness={readiness} label="Administrative participation report" />}
          {event.turnoutQuantityId && <Quantity quantity={quantities.get(event.turnoutQuantityId)} readiness={readiness} label="Participation divided by citizen-adult exposure" />}
        </div>
        <p className="small">The modeled participation-to-exposure ratio can exceed 100%. It is different from a verified eligibility fraction or the model's underlying participation tendency before the event-wide adjustment.</p>
      </article>)}
    </section>

    <section className="count-section" aria-labelledby="count-contests-title"><h3 id="count-contests-title">3. Votes in one contest</h3>
      <p>Each named candidate stays separate. Blank ballots, invalid marks and other admitted noncandidate outcomes are different categories, each with its own meaning.</p>
      {view.contests.map((contest) => {
        const gate = readiness.contestGates[contest.id];
        const outcomes = contest.outcomes;
        return <article key={contest.id} className="count-contest" aria-label={contest.id}>
          <h4>{contest.id}</h4>
          {contest.applicability === "absent" ? <p><strong>No contest applies to this event.</strong> This is different from leaving an applicable contest blank. Participation may still be modeled for the event.</p> : <>
            {!gate?.counts ? <p className="count-unavailable"><strong>Contest counts unavailable.</strong> {gate?.reason}</p> : <>
              <div className="count-candidates">{contest.candidates.map((candidate) => <section key={candidate.id} aria-label={candidate.name}>
                <h5>{candidate.name} <span>{candidate.party === null ? "Party affiliation unknown" : `Party: ${candidate.party}`}</span></h5>
                <div className="count-grid"><Quantity quantity={quantities.get(candidate.countQuantityId)} readiness={readiness} label="Candidate votes" /><Quantity quantity={quantities.get(candidate.shareQuantityId)} readiness={readiness} label="Candidate share" /></div>
              </section>)}</div>
              <div className="count-grid">{contest.noncandidateCategories.map((category) => <Quantity key={category.id} quantity={quantities.get(category.countQuantityId)} readiness={readiness} label={category.label} />)}</div>
            </>}
            <h5>Possible election outcomes</h5>
            {gate?.outcomes && outcomes.kind === "backend_weighted_predictive" ? <>
              <dl className="count-outcomes">{contest.candidates.map((candidate) => <div key={candidate.id}><dt>{candidate.name} wins</dt><dd>{percentage(outcomes.candidateProbabilities[candidate.id])}</dd></div>)}
                <div><dt>Tied highest candidate vote count · Unresolved</dt><dd>{percentage(outcomes.tiedMaximumProbability)}</dd></div>
                <div><dt>No valid candidate votes · Unresolved</dt><dd>{percentage(outcomes.noValidCandidateVotesProbability)}</dd></div>
              </dl><p className="small">Unresolved chances remain in the total. Winners come from the same exact vote-count rows and election rule; rounded shares do not determine a winner.</p>
            </> : <p>Winner probabilities are unavailable. {contest.outcomes.kind === "unavailable" ? contest.outcomes.reason : gate?.reason}</p>}
          </>}
          <p className="small">Rule qualification: {contest.rule.qualification}</p>
        </article>;
      })}
    </section>

    {view.chamber && <section className="count-section" aria-labelledby="count-chamber-title"><h3 id="count-chamber-title">Chamber organizing scenario</h3><h4>{view.chamber.scenario.label}</h4><p>{view.chamber.scenario.organizingDescription}</p>
      {readiness.canShowControl ? <dl className="count-outcomes"><div><dt>Democratic control</dt><dd>{percentage(view.chamber.control.democratic)}</dd></div><div><dt>Republican control</dt><dd>{percentage(view.chamber.control.republican)}</dd></div><div><dt>Unresolved because of race outcomes</dt><dd>{percentage(view.chamber.control.unresolvedRace)}</dd></div><div><dt>Unresolved chamber organization</dt><dd>{percentage(view.chamber.control.unresolvedOrganizing)}</dd></div></dl> : <p>Control probabilities are unavailable until complete contest results, retained seats and the organizing scenario are supplied and qualified together.</p>}
    </section>}

    <section className="count-section count-contributions" aria-labelledby="count-contributions-title"><h3 id="count-contributions-title">Planned behavioral contributions</h3><p>These describe research in progress. They are not options you can switch on in this preview. Changing a contribution requires a supported model fit while keeping the population cells and source coverage intact.</p>
      {[{ label: "Demographics", text: "How turnout and candidate choice vary across age and racial or ethnic population groups." }, { label: "Education", text: "How turnout and candidate choice vary with educational attainment." }, { label: "Education interactions", text: "How education patterns differ across age and racial or ethnic groups, with their effects estimated together." }, { label: "Community", text: "Measured religion and attendance patterns beyond demographic and education differences. These associations do not establish causes." }].map((contribution) => <div key={contribution.label}><h4>{contribution.label} <span>Not available in this result</span></h4><p>{contribution.text}</p></div>)}
    </section>
    <details className="count-checks"><summary>Immutable result and weighting record · Fixture</summary><p>A real result must preserve the probability assigned to each population state and the conditional weights within it. These summaries cannot be reconstructed by treating every row as equally likely.</p>
      <dl><dt>Joint artifact</dt><dd><code>{view.bindings.jointArtifact.sha256}</code></dd><dt>Weighted summary</dt><dd><code>{view.bindings.weightedSummary.sha256}</code></dd><dt>Audit qualification</dt><dd>{view.audit.qualification}</dd></dl>
    </details>
  </section>;
}
