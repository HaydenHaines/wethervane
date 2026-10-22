/** Invented interface examples only. No people, counts or chances are real. */
const summaryId = "invented-rule-summary";
const summary = (values, eventId) => ({ jointSummaryId: summaryId, conditioning: eventId ? { kind: "event_occurs", eventId } : { kind: "all_election_outcomes" }, values });
const winner = (personId, probability) => ({ kind: "named_winner", personId, probability });
const unresolved = (reason, probability) => ({ kind: "unresolved", reason, probability });
const participants = (mean, eventId) => ({ jointSummaryId: summaryId, basis: "event_occurs", eventId, mean, interval: { level: .8, lower: mean * .8, upper: mean * 1.2 } });

export function runoffFixture() {
  return {
    kind: "election_outcome_preparation",
    provenance: { kind: "invented_fixture", label: "Invented general-election and runoff example" },
    label: "One contest, two possible events",
    binding: { jointSummaryId: summaryId, jointArtifactSha256: "a".repeat(64), targetId: "invented-target", ruleScenarioId: "invented-runoff-rule", electionId: "invented-setting", informationCutoff: "2026-10-20T00:00:00Z", derivation: "same_joint_rule_rows_distinct_people" },
    evidence: { completeParentMass: true, sourceAndRuleAdmitted: true, summaryQualified: true, explanation: "Invented declarations test display behavior; they do not certify real sources, election rules or a model fit." },
    people: [{ id: "a", name: "Alex Example" }, { id: "b", name: "Blair Example" }, { id: "c", name: "Casey Example" }],
    finalOutcomes: summary([winner("a", .5), winner("b", .36), unresolved("highest_count_tie", .04), unresolved("finalist_boundary_tie", .1)]),
    phases: [
      { id: "general", eventId: "general-event", date: "2026-11-03", kind: "general", activationProbability: 1, participants: participants(1000, "general-event"), outcomes: summary([winner("a", .3), winner("b", .2), { kind: "runoff_required", probability: .4 }, unresolved("finalist_boundary_tie", .1)], "general-event") },
      { id: "runoff", eventId: "runoff-event", date: "2026-12-01", kind: "runoff", activationProbability: .4, participants: participants(750.5, "runoff-event"), outcomes: summary([winner("a", .5), winner("b", .4), unresolved("highest_count_tie", .1)], "runoff-event") },
    ],
    advancement: summary([{ personId: "a", probability: .4 }, { personId: "b", probability: .4 }, { personId: "c", probability: 0 }]),
    rankedExample: null,
  };
}

export function inactiveRunoffFixture() {
  const view = runoffFixture();
  view.label = "General election resolves the contest";
  view.provenance.label = "Invented inactive runoff example";
  view.finalOutcomes = summary([winner("a", 1)]);
  view.phases[0].outcomes = summary(structuredClone(view.finalOutcomes.values), "general-event");
  view.phases[1].activationProbability = 0;
  view.phases[1].participants = null;
  view.phases[1].outcomes = null;
  view.advancement = null;
  return view;
}

export function rankedFixture() {
  const view = inactiveRunoffFixture();
  view.label = "Ranked counts and final outcomes";
  view.provenance.label = "Invented ranked-count example";
  view.binding.ruleScenarioId = "invented-ranked-rule";
  view.phases = [view.phases[0]];
  view.finalOutcomes = summary([winner("a", .55), winner("b", .35), unresolved("elimination_tie", .1)]);
  view.phases[0].outcomes = summary(structuredClone(view.finalOutcomes.values), "general-event");
  view.rankedExample = {
    label: "This example ends with an unresolved elimination tie.", phaseId: "general", jointSummaryId: summaryId, exampleRowId: "invented-ranked-row",
    rounds: [{ number: 1, activeVotes: [{ personId: "a", votes: 40 }, { personId: "b", votes: 30 }, { personId: "c", votes: 30 }], inactiveBallots: 0, inactiveExplanation: "All 100 example ballots remain active.", decision: "Blair and Casey are tied for elimination. The supplied rule leaves this outcome unresolved." }],
  };
  return view;
}

export function unknownLeaderFixture() {
  const view = inactiveRunoffFixture();
  view.label = "An unidentified person can lead";
  view.provenance.label = "Invented distinct-person leader example";
  view.phases = [view.phases[0]];
  view.finalOutcomes = summary([winner("a", .5), { kind: "unidentified_leader", probability: .3 }, unresolved("no_valid_votes", .2)]);
  view.phases[0].outcomes = summary(structuredClone(view.finalOutcomes.values), "general-event");
  return view;
}

export function missingParentFixture() {
  const view = runoffFixture();
  view.label = "Incomplete simulation stays unavailable";
  view.provenance.label = "Invented missing population-state example";
  view.evidence.completeParentMass = false;
  return view;
}
