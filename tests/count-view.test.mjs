import assert from "node:assert/strict";
import test from "node:test";
import { countViewReadiness } from "../lib/count-view.ts";
import {
  partialTexasFixture,
  failedParentFixture,
  unsupportedRuleFixture,
  unresolvedRaceTieFixture,
  chamberTieScenarioFixture,
  absentContestFixture,
} from "./fixtures/count-view.mjs";

const quantity = (view, id) => view.quantities.find((q) => q.id === id);

test("partial Texas is explicitly illustrative, with separate coverage gates and no national control", () => {
  const v = partialTexasFixture(),
    r = countViewReadiness(v);
  assert.equal(r.valid, true);
  assert.equal(r.productionActivated, false);
  assert.equal(r.status, "partial");
  assert.equal(r.canShowPopulation, true);
  assert.equal(r.canShowParticipation, true);
  assert.equal(r.canShowRaces, true);
  assert.equal(r.canShowControl, false);
  assert.deepEqual(r.missingEventIds, ["XX-main"]);
  assert.deepEqual(r.missingContestIds, ["XX-senate"]);
  assert.equal(r.quantityGates["F1a-report"].display, true);
  assert.equal(r.quantityGates.participants.display, true);
  assert.notEqual(
    quantity(v, "F1a-report").value,
    quantity(v, "participants").mean,
  );
  assert.equal(v.scope.populationCells.count, 24);
  assert.equal(v.scope.communityProfiles.count, 12);
});

test("turnout ratios above one, fractional summaries and means of ratios are preserved", () => {
  const v = partialTexasFixture();
  assert.ok(quantity(v, "turnout").mean > 1);
  assert.notEqual(
    quantity(v, "turnout").mean,
    quantity(v, "participants").mean / quantity(v, "citizen-exposure").mean,
  );
  quantity(v, "adult-residents").mean = 130.25;
  quantity(v, "adult-residents").intervals = [
    { level: 0.95, lower: 100.5, upper: 160.75 },
  ];
  assert.equal(countViewReadiness(v).valid, true);
  assert.equal(quantity(v, "turnout").mean, 1.15);
});

test("missing is not zero and arbitrary interval probability levels stay explicit", () => {
  const v = partialTexasFixture();
  quantity(v, "D-count").mean = 0;
  quantity(v, "R-count").mean = 95;
  assert.equal(countViewReadiness(v).valid, true);
  const original = quantity(v, "D-count");
  v.quantities = v.quantities.map((q) =>
    q === original
      ? {
          id: q.id,
          label: q.label,
          quantityType: q.quantityType,
          unit: q.unit,
          denominator: q.denominator,
          kind: "unavailable",
          reason: "No qualified count summary.",
        }
      : q,
  );
  const r = countViewReadiness(v);
  assert.equal(r.quantityGates["D-count"].display, false);
  assert.match(r.quantityGates["D-count"].reason, /No qualified/);
  assert.equal(r.contestGates["TX-senate"].counts, false);
  quantity(v, "adult-residents").intervals = [
    { level: 0.73, lower: 100, upper: 150 },
  ];
  assert.equal(countViewReadiness(v).valid, true);
});

test("failed, missing, or redistributed parent mass suppresses predictions but preserves observed reports", () => {
  const cases = [
    failedParentFixture(),
    partialTexasFixture(),
    partialTexasFixture(),
    partialTexasFixture(),
  ];
  cases[1].audit.parents.pop();
  cases[2].audit.parents[0].representedMass = 1;
  cases[3].audit.sameRowAccounting = "failed";
  for (const v of cases) {
    const r = countViewReadiness(v);
    assert.equal(r.canShowParticipation, false);
    assert.equal(r.canShowRaces, false);
    assert.equal(r.canShowControl, false);
    assert.equal(r.canShowAdministrativeReports, true);
    assert.equal(r.quantityGates.participants.display, false);
    assert.equal(r.quantityGates["F1a-report"].display, true);
  }
});

test("immutable weighted artifact bindings are mandatory; marginals do not substitute", () => {
  for (const mutate of [
    (v) => (v.bindings.jointArtifact.sha256 = ""),
    (v) => (v.bindings.weightedSummary.jointArtifactSha256 = "b".repeat(64)),
    (v) => (v.bindings.weightedSummary.weighting = "equal_rows"),
    (v) => (quantity(v, "participants").weightedSummaryId = "other-summary"),
    (v) => v.bindings.axes.candidateIds.pop(),
  ]) {
    const v = partialTexasFixture();
    mutate(v);
    const r = countViewReadiness(v);
    assert.equal(r.valid, false);
    assert.equal(r.canShowRaces, false);
    assert.equal(r.canShowParticipation, false);
    assert.equal(r.quantityGates["F1a-report"].display, true);
  }
});

test("unsupported rule cannot force a winner, and absent contest is not an undervote", () => {
  const unsupported = countViewReadiness(unsupportedRuleFixture());
  assert.equal(unsupported.valid, true);
  assert.equal(unsupported.canShowParticipation, true);
  assert.equal(unsupported.canShowRaces, false);
  assert.equal(unsupported.contestGates["TX-senate"].outcomes, false);
  const v = unsupportedRuleFixture();
  v.contests[0].outcomes = partialTexasFixture().contests[0].outcomes;
  assert.equal(countViewReadiness(v).valid, false);
  const absent = absentContestFixture(),
    r = countViewReadiness(absent);
  assert.equal(r.valid, true);
  assert.equal(r.canShowParticipation, true);
  assert.equal(r.canShowRaces, false);
  assert.match(r.contestGates["TX-senate"].reason, /not an undervote/);
  absent.contests[0].noncandidateCategories = [
    {
      id: "blank",
      label: "Undervote",
      kind: "undervote",
      admitted: true,
      countQuantityId: "blank-count",
    },
  ];
  assert.equal(countViewReadiness(absent).valid, false);
});

test("tied maximum and no valid votes stay separate from resolved candidates and chamber organizing", () => {
  const v = unresolvedRaceTieFixture(),
    r = countViewReadiness(v);
  assert.equal(r.valid, true);
  assert.equal(r.contestGates["TX-senate"].outcomes, true);
  assert.equal(v.contests[0].outcomes.tiedMaximumProbability, 1);
  assert.equal(v.contests[0].outcomes.candidateProbabilities["named-D"], 0);
  v.contests[0].outcomes.tiedMaximumProbability = 0;
  v.contests[0].outcomes.noValidCandidateVotesProbability = 1;
  assert.equal(countViewReadiness(v).valid, true);
  v.contests[0].outcomes.candidateProbabilities["named-D"] = 1;
  assert.equal(countViewReadiness(v).valid, false);
  const chamber = chamberTieScenarioFixture();
  const ready = countViewReadiness(chamber);
  assert.equal(ready.canShowControl, true);
  assert.equal(ready.productionActivated, false);
  assert.equal(chamber.chamber.scenario.vicePresidentialTieBreakParty, "R");
  assert.equal(chamber.chamber.control.republican, 1);
  assert.equal(chamber.contests[0].outcomes.tiedMaximumProbability, 0);
});

test("full chamber needs event coverage, rule support, joint winner binding and complete fixed ledger", () => {
  for (const mutate of [
    (v) => v.scope.requiredEventIds.push("missing-event"),
    (v) => v.chamber.fixedSeatIds.pop(),
    (v) => (v.chamber.derivation = "independent_marginals"),
    (v) => (v.contests[0].rule.admission = "held"),
    (v) => (v.chamber.scenario.sourceSha256 = ""),
  ]) {
    const v = chamberTieScenarioFixture();
    mutate(v);
    assert.equal(countViewReadiness(v).canShowControl, false);
  }
});

test("named minor parties and unknown affiliations are preserved, pooled Other is not a candidate", () => {
  const v = partialTexasFixture();
  v.contests[0].candidates[0].party = "LIB";
  v.contests[0].candidates[1].party = null;
  assert.equal(countViewReadiness(v).valid, true);
  v.contests[0].candidates[0].name = "Other";
  assert.equal(countViewReadiness(v).valid, false);
});

test("negative counts, invalid intervals and hidden denominator omissions fail closed", () => {
  for (const mutate of [
    (v) => (quantity(v, "D-count").mean = -1),
    (v) => (quantity(v, "D-count").intervals[0].level = 1),
    (v) => (quantity(v, "D-share").denominator = { kind: "none" }),
    (v) => (quantity(v, "D-share").intervals[0].upper = 2),
    (v) => (quantity(v, "participants").intervals[0].upper = NaN),
  ]) {
    const v = partialTexasFixture();
    mutate(v);
    assert.equal(countViewReadiness(v).valid, false);
  }
});

test("preview readiness is not a provider-availability or production activation claim", () => {
  const v = chamberTieScenarioFixture();
  v.scope.populationCells.count = 6;
  v.scope.populationCells.label = "Six illustrative cells";
  v.scope.communityProfiles.count = 3;
  v.scope.communityProfiles.label = "Three illustrative profiles";
  const r = countViewReadiness(v);
  assert.equal(r.valid, true);
  assert.equal(r.productionActivated, false);
  v.lifecycle = "requires_computation";
  assert.equal(countViewReadiness(v).canShowParticipation, false);
});

test("simultaneous contests share one event rather than doubling participants", () => {
  const v = partialTexasFixture();
  const second = structuredClone(v.contests[0]);
  second.id = "TX-second-contest";
  second.candidates.forEach((candidate) => {
    candidate.id += "-second";
  });
  second.outcomes.candidateProbabilities = {
    "named-D-second": 0.65,
    "named-R-second": 0.3,
  };
  v.contests.push(second);
  v.events[0].contestIds.push(second.id);
  v.bindings.axes.contestIds.push(second.id);
  v.scope.requiredContestIds.push(second.id);
  v.bindings.axes.candidateIds.push(
    ...second.candidates.map((candidate) => candidate.id),
  );
  const result = countViewReadiness(v);
  assert.equal(result.valid, true);
  assert.deepEqual(result.displayableEventIds, ["TX-main"]);
  assert.deepEqual(result.displayableContestIds, [
    "TX-senate",
    "TX-second-contest",
  ]);
  assert.equal(quantity(v, "participants").mean, 100);
});

test("a conditional pair share or administrative report cannot fill the required predictive quantity", () => {
  for (const mutate of [
    (v) => (quantity(v, "D-share").denominator.quantityId = "D-count"),
    (v) => (v.events[0].participantQuantityId = "F1a-report"),
    (v) =>
      (quantity(v, "participants").quantityType =
        "administrative_participation"),
  ]) {
    const v = partialTexasFixture();
    mutate(v);
    const result = countViewReadiness(v);
    assert.equal(result.valid, false);
    assert.equal(result.canShowRaces, false);
    assert.equal(result.canShowAdministrativeReports, true);
  }
});

test("noncandidate ballots and responses retain meaning while candidate counts remain votes", () => {
  const v = partialTexasFixture();
  assert.equal(quantity(v, "blank-count").unit, "ballots");
  assert.equal(quantity(v, "invalid-count").unit, "ballots");
  assert.equal(countViewReadiness(v).valid, true);
  v.contests[0].noncandidateCategories[0].kind = "nonresponse";
  v.contests[0].noncandidateCategories[0].label =
    "Admitted same-event response category";
  quantity(v, "blank-count").unit = "responses";
  assert.equal(countViewReadiness(v).valid, true);
  for (const id of ["D-count", "valid-candidate-votes"]) {
    const changed = partialTexasFixture();
    quantity(changed, id).unit = "ballots";
    assert.equal(countViewReadiness(changed).valid, false);
  }
  for (const unit of ["votes", "responses"]) {
    const changed = partialTexasFixture();
    quantity(changed, "invalid-count").unit = unit;
    assert.equal(countViewReadiness(changed).valid, false);
  }
  v.contests[0].noncandidateCategories[0].admitted = false;
  assert.equal(countViewReadiness(v).canShowRaces, false);
});

test("event and contest coverage rejects identities outside the declared scope", () => {
  for (const change of [
    (v) => (v.scope.requiredEventIds = ["XX-main"]),
    (v) => (v.scope.requiredContestIds = ["XX-senate"]),
  ]) {
    const v = partialTexasFixture();
    change(v);
    const r = countViewReadiness(v);
    assert.equal(r.valid, false);
    assert.equal(r.canShowControl, false);
    assert.equal(r.canShowParticipation, false);
    assert.ok(r.problems.some((p) => p.code === "unexpected_scope"));
  }
});

test("administrative reports require independent source admission even when predictive audit passes", () => {
  const v = partialTexasFixture();
  quantity(v, "F1a-report").sourceAdmission = "held";
  quantity(v, "F1a-report").sourceQualification =
    "Invented report awaiting direct source review.";
  let r = countViewReadiness(v);
  assert.equal(r.valid, true);
  assert.equal(r.canShowParticipation, true);
  assert.equal(r.canShowAdministrativeReports, false);
  assert.equal(r.quantityGates["F1a-report"].display, false);
  assert.match(r.quantityGates["F1a-report"].reason, /own source review/);
  assert.ok(r.problems.some((p) => p.code === "administrative_source_held"));
  v.audit.parents[0].status = "failed";
  r = countViewReadiness(v);
  assert.equal(r.canShowAdministrativeReports, false);
  assert.equal(r.canShowParticipation, false);
  quantity(v, "F1a-report").sourceAdmission = "admitted";
  r = countViewReadiness(v);
  assert.equal(r.canShowAdministrativeReports, true);
  assert.equal(r.canShowParticipation, false);
  delete quantity(v, "F1a-report").sourceQualification;
  r = countViewReadiness(v);
  assert.equal(r.valid, false);
  assert.equal(r.canShowAdministrativeReports, false);
});
