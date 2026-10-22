/** Private presentation preparation. This is neither a backend contract nor an
 * artifact parser. Only invented fixtures may use it until a reviewed adapter exists. */
export type UnresolvedOutcomeReason =
  | "highest_count_tie" | "elimination_tie" | "finalist_boundary_tie"
  | "no_valid_votes" | "procedure_unresolved" | "identity_or_eligibility";
export type ElectionOutcome =
  | { kind: "named_winner"; personId: string; probability: number }
  | { kind: "unidentified_leader"; probability: number }
  | { kind: "unresolved"; reason: UnresolvedOutcomeReason; probability: number };
export type PhaseOutcome = ElectionOutcome | { kind: "runoff_required"; probability: number };
export interface OutcomeSummary<T> {
  jointSummaryId: string;
  conditioning: { kind: "all_election_outcomes" } | { kind: "event_occurs"; eventId: string };
  values: T[];
}
export interface ElectionOutcomeDraft {
  kind: "election_outcome_preparation";
  provenance: { kind: "invented_fixture"; label: string };
  label: string;
  binding: { jointSummaryId: string; jointArtifactSha256: string; targetId: string; ruleScenarioId: string; electionId: string; informationCutoff: string; derivation: "same_joint_rule_rows_distinct_people" };
  evidence: {
    completeParentMass: boolean;
    sourceAndRuleAdmitted: boolean;
    summaryQualified: boolean;
    explanation: string;
  };
  people: { id: string; name: string }[];
  finalOutcomes: OutcomeSummary<ElectionOutcome>;
  phases: {
    id: string;
    eventId: string;
    date: string;
    kind: "general" | "runoff";
    /** Later-event summaries are conditional on this event occurring. */
    activationProbability: number;
    participants: null | {
      jointSummaryId: string;
      basis: "event_occurs";
      eventId: string;
      mean: number;
      interval: { level: number; lower: number; upper: number };
    };
    outcomes: null | OutcomeSummary<PhaseOutcome>;
  }[];
  /** Unconditional chances. Two people advance in each resolved runoff branch. */
  advancement: null | OutcomeSummary<{ personId: string; probability: number }>;
  /** A single supplied example; never an average sequence assembled from marginals. */
  rankedExample: null | {
    label: string;
    phaseId: string;
    jointSummaryId: string;
    exampleRowId: string;
    rounds: {
      number: number;
      activeVotes: { personId: string; votes: number }[];
      inactiveBallots: number;
      inactiveExplanation: string;
      decision: string;
    }[];
  };
}

export const unresolvedOutcomeLabels: Record<UnresolvedOutcomeReason, string> = {
  highest_count_tie: "Tie for highest vote count",
  elimination_tie: "Tie over who is eliminated",
  finalist_boundary_tie: "Tie over who advances to the runoff",
  no_valid_votes: "No valid candidate votes",
  procedure_unresolved: "Outcome unresolved under the available rule",
  identity_or_eligibility: "Winner identity or eligibility unresolved",
};

const probability = (n: number) => Number.isFinite(n) && n >= 0 && n <= 1;
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-8;
const nonempty = (s: string) => typeof s === "string" && s.trim().length > 0;
const unique = (ids: string[]) => ids.every(nonempty) && new Set(ids).size === ids.length;
const outcomeKey = (o: PhaseOutcome) => o.kind === "named_winner" ? `named:${o.personId}` : o.kind === "unresolved" ? `unresolved:${o.reason}` : o.kind;

/** Checks the supplied summaries without tabulating ballots or computing a forecast. */
export function electionOutcomeReadiness(view: ElectionOutcomeDraft): {
  productionActivated: false; display: boolean; problems: string[];
} {
  const problems: string[] = [];
  const issue = (message: string) => problems.push(message);
  if (view.kind !== "election_outcome_preparation" || view.provenance.kind !== "invented_fixture" || !nonempty(view.provenance.label)) issue("Explicit invented-example provenance is required.");
  const b = view.binding;
  if (![b.jointSummaryId, b.targetId, b.ruleScenarioId, b.electionId].every(nonempty) || !Number.isFinite(Date.parse(b.informationCutoff)) || b.derivation !== "same_joint_rule_rows_distinct_people" || !/^[a-f0-9]{64}$/.test(b.jointArtifactSha256)) issue("The joint result and rule identities are incomplete.");
  if (view.evidence.completeParentMass !== true) issue("Some population states are missing or failed; their probability weight cannot be discarded.");
  if (view.evidence.sourceAndRuleAdmitted !== true) issue("Sources or election rules are awaiting review.");
  if (view.evidence.summaryQualified !== true) issue("The outcome summaries have not passed their required checks.");
  if (!nonempty(view.evidence.explanation) || !nonempty(view.label)) issue("The scope and evidence explanation are required.");
  const people = new Set(view.people.map(p => p.id));
  if (!unique(view.people.map(p => p.id)) || view.people.some(p => !nonempty(p.name))) issue("Named people need distinct identities and names.");
  const checkDistribution = (summary: OutcomeSummary<PhaseOutcome>, phase: boolean, eventId?: string) => {
    if (summary.jointSummaryId !== b.jointSummaryId) issue("All summaries must belong to the same joint result.");
    if (eventId ? summary.conditioning?.kind !== "event_occurs" || summary.conditioning.eventId !== eventId : summary.conditioning?.kind !== "all_election_outcomes") issue("Outcome summaries need an explicit denominator: all election outcomes, or the identified event occurring.");
    if (!summary.values.length || !unique(summary.values.map(outcomeKey)) || summary.values.some(o => !probability(o.probability)) || !close(summary.values.reduce((sum, o) => sum + o.probability, 0), 1)) issue("Outcome chances must cover all outcomes exactly once and total 100%, including unresolved cases.");
    for (const o of summary.values) {
      if (o.kind === "named_winner") { if (!people.has(o.personId)) issue("A winner refers to an unknown named person."); }
      else if (o.kind === "unresolved") { if (!Object.hasOwn(unresolvedOutcomeLabels, o.reason)) issue("An unresolved outcome needs a supported explanation."); }
      else if (o.kind !== "unidentified_leader" && !(phase && o.kind === "runoff_required")) issue("Only named people, unidentified-person outcomes and explicit unresolved cases can represent final winners.");
    }
  };
  checkDistribution(view.finalOutcomes, false);
  if (!unique(view.phases.map(p => p.id)) || !unique(view.phases.map(p => p.eventId))) issue("Each phase needs a distinct event and phase identity.");
  const general = view.phases[0], runoff = view.phases[1];
  if (view.phases.length < 1 || view.phases.length > 2 || general?.kind !== "general" || general?.activationProbability !== 1 || (runoff && runoff.kind !== "runoff")) issue("This preparation supports one general election and at most one later runoff.");
  for (const phase of view.phases) {
    if (!probability(phase.activationProbability) || !Number.isFinite(Date.parse(phase.date))) issue("Each event needs a valid date and chance of occurring.");
    if (phase.activationProbability === 0) {
      if (phase.participants !== null || phase.outcomes !== null) issue("An event that does not occur has no participation or outcome summary, rather than zero turnout.");
      continue;
    }
    if (phase.participants) {
      const p = phase.participants, i = p.interval;
      if (p.jointSummaryId !== b.jointSummaryId || p.basis !== "event_occurs" || p.eventId !== phase.eventId) issue("Event participation must come from the same joint result and be conditional on the identified event occurring.");
      if (![p.mean, i.lower, i.upper].every(n => Number.isFinite(n) && n >= 0) || i.lower > i.upper || !probability(i.level) || i.level === 0 || i.level === 1) issue("Participation needs nonnegative values and an ordered uncertainty interval.");
    }
    if (!phase.outcomes) issue("An active event is missing its conditional outcome summary.");
    else checkDistribution(phase.outcomes, phase.kind === "general", phase.eventId);
  }
  const runoffProbability = general?.outcomes?.values.find(o => o.kind === "runoff_required")?.probability ?? 0;
  if ((runoffProbability > 0 && !runoff) || (runoff && !close(runoff.activationProbability, runoffProbability))) issue("The later event chance must match the general election's runoff branch.");
  if (runoff && general && Date.parse(runoff.date) <= Date.parse(general.date)) issue("A runoff must be a later event.");
  if (view.advancement) {
    const a = view.advancement;
    if (a.jointSummaryId !== b.jointSummaryId || a.conditioning?.kind !== "all_election_outcomes" || !unique(a.values.map(p => p.personId)) || a.values.length !== people.size || a.values.some(p => !people.has(p.personId) || !probability(p.probability) || p.probability > runoffProbability + 1e-8) || !close(a.values.reduce((sum, p) => sum + p.probability, 0), 2 * runoffProbability)) issue("Runoff advancement must retain every named person and two finalists per resolved branch, using unconditional chances.");
  }
  // Reconcile supplied final probabilities against supplied event summaries. This
  // is an accounting check, not a new probability estimate or a rule tabulation.
  if (general?.outcomes && (!runoff || runoff.activationProbability === 0 || runoff.outcomes)) {
    const expected = new Map<string, number>();
    for (const [summary, weight] of [[general.outcomes, 1], [runoff?.outcomes, runoffProbability]] as const) {
      for (const o of summary?.values ?? []) if (o.kind !== "runoff_required") expected.set(outcomeKey(o), (expected.get(outcomeKey(o)) ?? 0) + weight * o.probability);
    }
    const supplied = new Map(view.finalOutcomes.values.map(o => [outcomeKey(o), o.probability]));
    if ([...new Set([...expected.keys(), ...supplied.keys()])].some(key => !close(expected.get(key) ?? 0, supplied.get(key) ?? 0))) issue("Final outcomes do not reconcile with the supplied general and conditional-runoff results.");
  }
  const ranked = view.rankedExample;
  if (ranked) {
    if (ranked.jointSummaryId !== b.jointSummaryId || !view.phases.some(p => p.id === ranked.phaseId && p.activationProbability > 0) || !nonempty(ranked.exampleRowId) || !nonempty(ranked.label) || !ranked.rounds.length) issue("The illustrative ranked count needs an identified example row, matching active event and joint result.");
    let ballots: number | undefined;
    ranked.rounds.forEach((round, index) => {
      const total = round.activeVotes.reduce((sum, p) => sum + p.votes, 0) + round.inactiveBallots;
      if (round.number !== index + 1 || !unique(round.activeVotes.map(p => p.personId)) || round.activeVotes.some(p => !people.has(p.personId) || !Number.isSafeInteger(p.votes) || p.votes < 0) || !Number.isSafeInteger(round.inactiveBallots) || round.inactiveBallots < 0 || !Number.isSafeInteger(total) || !nonempty(round.inactiveExplanation) || !nonempty(round.decision) || (ballots !== undefined && total !== ballots)) issue("Example rounds need ordered integer counts, distinct named candidates and unchanged total ballots including inactive ballots.");
      ballots = total;
    });
  }
  return { productionActivated: false, display: problems.length === 0, problems };
}
