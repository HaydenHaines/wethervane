import { electionOutcomeReadiness, unresolvedOutcomeLabels, type ElectionOutcomeDraft, type PhaseOutcome } from "../lib/election-outcome-view.ts";

const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const percentage = (n: number) => `${number.format(n * 100)}%`;

/** Private invented-example component; no public route imports this module. */
export default function ElectionOutcomePreview({ view }: { view: ElectionOutcomeDraft }) {
  const readiness = electionOutcomeReadiness(view);
  const names = new Map(view.people.map(person => [person.id, person.name]));
  const outcomeLabel = (outcome: PhaseOutcome, winnerAction: string) => {
    if (outcome.kind === "named_winner") return `${names.get(outcome.personId)} ${winnerAction}`;
    if (outcome.kind === "runoff_required") return "A runoff is required";
    if (outcome.kind === "unidentified_leader") return "Unidentified person leads · Identity or eligibility unresolved";
    return `${unresolvedOutcomeLabels[outcome.reason]} · Unresolved`;
  };
  const distribution = (outcomes: PhaseOutcome[], winnerAction: string) => <dl className="count-outcomes">{outcomes.map((outcome, index) => <div key={index}><dt>{outcomeLabel(outcome, winnerAction)}</dt><dd>{percentage(outcome.probability)}</dd></div>)}</dl>;
  const hasUnknown = view.finalOutcomes.values.some(outcome => outcome.kind === "unidentified_leader");
  return <section className="count-preview election-outcome-preview" aria-label={view.label}>
    <div className="count-fixture-banner" role="note"><strong>Illustrative data only — no public count forecast</strong><p>{view.provenance.label}. The numbers, people and evidence declarations are invented to test the interface.</p></div>
    <header><div className="eyebrow">Frontend preparation</div><h2>{view.label}</h2><p>One election can involve more than one event. Advancing to another round is different from winning office.</p></header>
    <p className="small">{view.evidence.explanation}</p>
    {!readiness.display ? <div className="count-warning"><strong>Outcome estimates unavailable</strong><ul>{readiness.problems.map((problem, index) => <li key={index}>{problem}</li>)}</ul><p>No winner probabilities or participation estimates are displayed for this example.</p></div> : <>
      <section className="count-section" aria-label="Final election outcome">
        <h3>Final election outcome</h3><p>Across all simulated outcomes, after any required runoff or ranked count. Unresolved outcomes remain part of the total.</p>
        {distribution(view.finalOutcomes.values, "wins overall")}
        {hasUnknown && <p>An unidentified leader represents one of the distinct people in a simulation, whose identity or eligibility is unresolved. It is not a candidate named “Other,” a pooled write-in total or a known chamber affiliation.</p>}
      </section>
      <section className="count-section" aria-label="Election stages"><h3>How the election gets there</h3>
        {view.phases.map(phase => <article className="count-event" key={phase.id} aria-label={`${phase.kind === "general" ? "General election" : "Runoff"} on ${phase.date}`}>
          <h4>{phase.kind === "general" ? "General election" : "Runoff"} · {phase.date}</h4>
          {phase.kind === "runoff" && <p><strong>Chance a runoff occurs: {percentage(phase.activationProbability)}</strong></p>}
          {phase.activationProbability === 0 ? <p><strong>Not held in this example.</strong> No participant count is defined for a runoff that does not occur.</p> : <>
            <h5>{phase.kind === "general" ? "Outcome of the general election" : "Outcome if the runoff occurs"}</h5>
            {phase.outcomes && distribution(phase.outcomes.values, phase.kind === "general" ? "wins the general election" : "wins the runoff")}
            {phase.kind === "general" && view.advancement && <details className="count-checks"><summary>Who could advance to a runoff?</summary>
              <p>Chances across all simulated general elections. Two people advance in each resolved runoff branch, so these chances need not add to 100%. Advancing does not mean winning office.</p>
              <dl className="count-outcomes">{view.advancement.values.map(person => <div key={person.personId}><dt>{names.get(person.personId)} advances</dt><dd>{percentage(person.probability)}</dd></div>)}</dl>
            </details>}
            <div className="count-quantity">
              <h5>{phase.kind === "runoff" ? "Runoff participation, if this runoff occurs" : "Participation in the general election"}</h5>
              {phase.participants ? <><p className="count-value">{number.format(phase.participants.mean)} <span>people</span></p><p>{percentage(phase.participants.interval.level)} predicted range: {number.format(phase.participants.interval.lower)}–{number.format(phase.participants.interval.upper)} people.</p><p className="small">{phase.kind === "runoff" ? "Average among simulated outcomes where this runoff takes place. Outcomes with no runoff are excluded from this conditional average." : "Average across simulated outcomes for this event."} The mean can be fractional even though each simulated count is a whole number.</p></> : <p>Participation estimate not supplied.</p>}
            </div>
          </>}
        </article>)}
        <p>These events fill one election contest. A later runoff has its own participants; the two events are not two seats or two independent elections.</p>
      </section>
      {view.rankedExample && <details className="count-checks"><summary>Example ranked-choice count</summary>
        <p>Voters rank candidates. Each round shows how their ballots count at that stage, including ballots that are no longer active under the supplied rule.</p>
        <p><strong>One illustrative simulation, not the predicted sequence.</strong> {view.rankedExample.label} Counts below come from the supplied example; the interface does not transfer ballots or break ties.</p>
        {view.rankedExample.rounds.map(round => <section key={round.number} className="count-event" aria-label={`Example round ${round.number}`}>
          <h4>Round {round.number}</h4><dl className="count-outcomes">{round.activeVotes.map(person => <div key={person.personId}><dt>{names.get(person.personId)} · Active votes</dt><dd>{number.format(person.votes)}</dd></div>)}<div><dt>Ballots no longer active in this round</dt><dd>{number.format(round.inactiveBallots)}</dd></div></dl>
          <p>{round.inactiveExplanation}</p><p><strong>Supplied round outcome:</strong> {round.decision}</p>
        </section>)}
      </details>}
    </>}
    <p className="small">This private preview does not establish source or legal qualification, forecast accuracy, chamber control or the availability of a public count model.</p>
  </section>;
}
