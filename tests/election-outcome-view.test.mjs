import test from "node:test";
import assert from "node:assert/strict";
import { electionOutcomeReadiness } from "../lib/election-outcome-view.ts";
import { runoffFixture, inactiveRunoffFixture, rankedFixture, unknownLeaderFixture, missingParentFixture } from "./fixtures/election-outcome.mjs";

test("independent event and full-election summaries reconcile without normalizing advancement", () => {
  for (const fixture of [runoffFixture, inactiveRunoffFixture, rankedFixture, unknownLeaderFixture]) {
    const ready = electionOutcomeReadiness(fixture());
    assert.equal(ready.display, true, ready.problems.join("; "));
    assert.equal(ready.productionActivated, false);
  }
  assert.equal(runoffFixture().advancement.values.reduce((sum, p) => sum + p.probability, 0), .8);
});

test("missing parents, source/rule admission or qualification suppress the complete result", () => {
  assert.equal(electionOutcomeReadiness(missingParentFixture()).display, false);
  for (const field of ["sourceAndRuleAdmitted", "summaryQualified"]) {
    const view = runoffFixture(); view.evidence[field] = false;
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
});

test("changing a joint identity, conditioning or rule binding cannot silently combine summaries", () => {
  for (const change of [
    v => v.finalOutcomes.jointSummaryId = "another-result",
    v => v.phases[1].participants.jointSummaryId = "another-result",
    v => v.phases[1].outcomes.jointSummaryId = "another-result",
    v => v.phases[1].participants.basis = "all_simulations",
    v => v.phases[1].participants.eventId = "general-event",
    v => v.phases[1].outcomes.conditioning = { kind: "all_election_outcomes" },
    v => v.phases[1].outcomes.conditioning.eventId = "general-event",
    v => v.finalOutcomes.conditioning = { kind: "event_occurs", eventId: "runoff-event" },
    v => v.advancement.conditioning = { kind: "event_occurs", eventId: "runoff-event" },
    v => v.binding.derivation = "independent_marginals",
    v => v.binding.electionId = "",
    v => v.binding.informationCutoff = "invalid",
  ]) {
    const view = runoffFixture(); change(view);
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
});

test("nonboolean evidence and prototype property names cannot pass the display gate", () => {
  for (const field of ["completeParentMass", "sourceAndRuleAdmitted", "summaryQualified"]) {
    const view = runoffFixture(); view.evidence[field] = "false";
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
  const view = runoffFixture();
  view.finalOutcomes.values[2].reason = "constructor";
  assert.equal(electionOutcomeReadiness(view).display, false);
});

test("zero-activation events require null summaries instead of zero participation", () => {
  const view = inactiveRunoffFixture();
  view.phases[1].participants = structuredClone(view.phases[0].participants);
  view.phases[1].participants.mean = 0;
  assert.equal(electionOutcomeReadiness(view).display, false);
  view.phases[1].participants = null;
  view.phases[1].outcomes = structuredClone(view.finalOutcomes);
  assert.equal(electionOutcomeReadiness(view).display, false);
});

test("branch mismatches and swapped conditional probabilities reject an apparently valid final total", () => {
  for (const change of [
    v => v.phases[1].activationProbability = .5,
    v => { v.phases[1].outcomes.values[0].probability = .4; v.phases[1].outcomes.values[1].probability = .5; },
    v => v.phases.pop(),
    v => v.phases[1].date = v.phases[0].date,
    v => v.phases[1].eventId = v.phases[0].eventId,
  ]) {
    const view = runoffFixture(); change(view);
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
});

test("invalid, duplicate and incomplete outcome mass stays unavailable", () => {
  for (const p of [NaN, Infinity, -.1, 1.1, .6]) {
    const view = runoffFixture(); view.finalOutcomes.values[0].probability = p;
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
  const duplicate = runoffFixture();
  duplicate.finalOutcomes.values[1].personId = "a";
  assert.equal(electionOutcomeReadiness(duplicate).display, false);
});

test("pooled residual totals cannot become a person or final winner bucket", () => {
  const view = unknownLeaderFixture();
  view.finalOutcomes.values[1].kind = "pooled_other_winner";
  assert.equal(electionOutcomeReadiness(view).display, false);
  view.finalOutcomes.values[1] = { kind: "named_winner", personId: "unbound-source-label", probability: .3 };
  assert.equal(electionOutcomeReadiness(view).display, false);
});

test("advancement retains two finalists per resolved branch without exceeding activation chance", () => {
  for (const change of [
    v => v.advancement.values[0].probability = .5,
    v => v.advancement.values[0].personId = "unknown",
    v => v.advancement.values.pop(),
    v => v.advancement.jointSummaryId = "another-result",
  ]) {
    const view = runoffFixture(); change(view);
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
});

test("ranked examples keep integer counts and inactive ballots in one supplied sequence", () => {
  const view = rankedFixture();
  const second = structuredClone(view.rankedExample.rounds[0]);
  second.number = 2;
  second.activeVotes = [{ personId: "a", votes: 53 }, { personId: "b", votes: 43 }];
  second.inactiveBallots = 4;
  second.inactiveExplanation = "Four ballots have no remaining usable preference in this supplied example.";
  second.decision = "Alex wins in this example.";
  view.rankedExample.rounds[0].activeVotes = [{ personId: "a", votes: 40 }, { personId: "b", votes: 35 }, { personId: "c", votes: 25 }];
  view.rankedExample.rounds[0].decision = "Casey is eliminated in this example.";
  view.rankedExample.rounds.push(second);
  assert.equal(electionOutcomeReadiness(view).display, true);
  second.inactiveBallots = 0;
  assert.equal(electionOutcomeReadiness(view).display, false);
  second.inactiveBallots = 4.1;
  assert.equal(electionOutcomeReadiness(view).display, false);
});

test("a ranked example cannot borrow another phase, result or unnamed source label", () => {
  for (const change of [
    v => v.rankedExample.phaseId = "unbound-phase",
    v => v.rankedExample.jointSummaryId = "another-result",
    v => v.rankedExample.rounds[0].activeVotes[0].personId = "unknown",
    v => v.rankedExample.rounds[0].number = 3,
  ]) {
    const view = rankedFixture(); change(view);
    assert.equal(electionOutcomeReadiness(view).display, false);
  }
});
