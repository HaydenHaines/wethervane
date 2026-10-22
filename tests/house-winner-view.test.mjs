import test from "node:test";
import assert from "node:assert/strict";
import { houseWinnerView } from "../lib/house-winner-view.ts";
import { modeledWinnerFixture, samePartyWinnerFixture, residualWinnerFixture, unadmittedWinnerFixture, rankedChoiceWinnerFixture, uncertifiedWinnerFixture } from "./fixtures/house-winner.mjs";
test("modeled ultimate group distribution binds truth by group ID, not names", () => {
  const view = houseWinnerView(modeledWinnerFixture());
  assert.equal(view.stage, "ultimate_outcome"); assert.equal(view.resolution, "party_group");
  assert.equal(view.prediction.status, "modeled");
  assert.equal(view.prediction.probabilities.reduce((s, e) => s + e.probability, 0), 1);
  assert.equal(view.truth.groupId, "D");
  assert.notEqual(view.truth.winnerNames[0], view.groups[0].members[0].label);
  assert.equal(view.groups[0].members[0].winProbability, undefined);
  assert.equal(view.firstBallotShare, undefined); assert.equal(view.score, undefined); assert.equal(view.control, undefined);
});
test("same-party structural certainty never resolves the individual winner", () => {
  const view = houseWinnerView(samePartyWinnerFixture());
  assert.equal(view.prediction.status, "structural"); assert.equal(view.prediction.groupId, "D");
  assert.equal(view.groups[0].members.length, 2); assert.ok(view.groups[0].members.every(m => m.probability === undefined));
  assert.equal(view.prediction.probabilities, undefined);
  const invalid = modeledWinnerFixture(); invalid.prediction = { status: "structural", groupId: "D", reason: "Multiple admitted alternatives contradict structural certainty." };
  assert.equal(houseWinnerView(invalid).prediction.status, "unavailable");
});
test("singleton residual remains a category and source-reported names remain distinct from admitted people", async () => {
  const view = houseWinnerView(residualWinnerFixture());
  assert.deepEqual(view.groups[2].members, [{ id: "residual", label: "Scattered", kind: "pooled_or_residual" }]);
  const { sourceReportedNameWinnerFixture } = await import("./fixtures/house-winner.mjs");
  const sourceReported = houseWinnerView(sourceReportedNameWinnerFixture());
  assert.deepEqual(sourceReported.groups[2].members, [{ id: "reported-entry", label: "Source-listed entry 7", kind: "source_reported_name" }]);
  assert.equal(sourceReported.groups[2].resolvedCandidate, undefined);
  const sourceEligible = sourceReportedNameWinnerFixture();
  sourceEligible.groups[2].namedSingletonResolutionEligible = true;
  assert.throws(() => houseWinnerView(sourceEligible), TypeError);
  const unadmitted = houseWinnerView(unadmittedWinnerFixture());
  assert.equal(unadmitted.groups.length, 1); assert.equal(unadmitted.prediction.status, "unavailable");
  assert.equal(unadmitted.truth.status, "observed"); assert.equal(unadmitted.prediction.probability, undefined);
});
test("final RCV truth and uncertified truth stay separate from predictions", () => {
  const ranked = houseWinnerView(rankedChoiceWinnerFixture());
  assert.equal(ranked.ruleFamily, "ranked_choice"); assert.match(ranked.truth.sourceBasis, /final ranked-choice/);
  assert.equal(ranked.truth.groupId, "R"); assert.equal(ranked.prediction.probabilities[0].probability, .6);
  const uncertified = houseWinnerView(uncertifiedWinnerFixture());
  assert.equal(uncertified.prediction.status, "modeled"); assert.equal(uncertified.truth.status, "unavailable");
  assert.equal(uncertified.truth.groupId, undefined); assert.equal(uncertified.truth.winnerNames, undefined);
});
test("missing, nonfinite, unsupported, duplicate and unnormalized probabilities fail closed", () => {
  const distributions = [undefined, [], [{ groupId: "D", probability: 1 }], [{ groupId: "D", probability: .5 }, { groupId: "D", probability: .4 }, { groupId: "O", probability: .1 }],
    ...[NaN, Infinity, -.1, 1.1, undefined].map(probability => [{ groupId: "D", probability }, { groupId: "R", probability: .35 }, { groupId: "O", probability: .05 }]),
    [{ groupId: "D", probability: .5 }, { groupId: "R", probability: .35 }, { groupId: "O", probability: .05 }],
    [{ groupId: "X", probability: .6 }, { groupId: "R", probability: .35 }, { groupId: "O", probability: .05 }]];
  for (const probabilities of distributions) {
    const fixture = modeledWinnerFixture(); fixture.prediction.probabilities = probabilities;
    const view = houseWinnerView(fixture); assert.equal(view.prediction.status, "unavailable"); assert.equal(view.groups.length, 3); assert.equal(view.truth.status, "observed");
  }
});
test("invalid truth is unavailable without suppressing a valid prediction", () => {
  for (const truth of [undefined, { status: "observed", groupId: "X", sourceBasis: "Report" }, { status: "observed", groupId: "D", sourceBasis: "" }, { status: "observed", groupId: "D", sourceBasis: "Report", winnerNames: [null] }]) {
    const view = houseWinnerView({ ...modeledWinnerFixture(), truth }); assert.equal(view.truth.status, "unavailable"); assert.equal(view.prediction.status, "modeled");
  }
});
test("reject wrong provenance, target and ambiguous membership", () => {
  for (const override of [{ fixtureOnly: false }, { sourceStatus: "reviewed" }, { stage: "first_ballot" }, { resolution: "candidate" }, { ruleFamily: "unknown" }]) assert.throws(() => houseWinnerView({ ...modeledWinnerFixture(), ...override }), TypeError);
  for (const mutate of [f => f.groups.push(f.groups[0]), f => f.groups[1].members.push(f.groups[0].members[0]), f => delete f.groups[0].members[0].kind, f => f.groups[0].members = []]) {
    const fixture = modeledWinnerFixture(); mutate(fixture); assert.throws(() => houseWinnerView(fixture), TypeError);
  }
});
test("view snapshots evidence and strips undeclared candidate numerical claims", () => {
  const fixture = modeledWinnerFixture(); fixture.groups[0].members[0].probability = .6; fixture.control = .7;
  const view = houseWinnerView(fixture); fixture.groups[0].members[0].label = "Changed"; fixture.prediction.probabilities[0].probability = 0; fixture.truth.winnerNames[0] = "Changed";
  assert.equal(view.groups[0].members[0].label, "Dana Example"); assert.equal(view.prediction.probabilities[0].probability, .6); assert.equal(view.truth.winnerNames[0], "Example, Dana");
  assert.equal(view.groups[0].members[0].probability, undefined); assert.equal(view.control, undefined);
});

test("explicit eligible singleton resolves to exactly its validated group probability", async () => {
  const { eligibleSingletonWinnerFixture, eligibleStructuralWinnerFixture } = await import("./fixtures/house-winner.mjs");
  const view = houseWinnerView(eligibleSingletonWinnerFixture());
  assert.deepEqual(view.groups[0].resolvedCandidate, { id: "example-d", label: "Dana Example", probability: .6, basis: "modeled_group_probability" });
  assert.equal(view.groups[1].resolvedCandidate, undefined);
  assert.equal(view.groups[2].resolvedCandidate, undefined);
  assert.notEqual(view.groups[0].resolvedCandidate.label, view.truth.winnerNames[0]);
  const structural = houseWinnerView(eligibleStructuralWinnerFixture());
  assert.deepEqual(structural.groups[0].resolvedCandidate, { id: "example-d", label: "Dana Example", probability: 1, basis: "structural_group_support" });
});
test("eligibility alone or malformed prediction cannot resolve a candidate", async () => {
  const { eligibleSingletonWinnerFixture, eligibleUnavailableWinnerFixture } = await import("./fixtures/house-winner.mjs");
  const unavailable = houseWinnerView(eligibleUnavailableWinnerFixture());
  assert.equal(unavailable.groups[0].namedSingletonResolutionEligible, true);
  assert.equal(unavailable.groups[0].resolvedCandidate, undefined);
  for (const malformed of [undefined, NaN, -1, 1.1, .5]) {
    const fixture = eligibleSingletonWinnerFixture(); fixture.prediction.probabilities[0].probability = malformed;
    const view = houseWinnerView(fixture);
    assert.equal(view.prediction.status, "unavailable"); assert.equal(view.groups[0].resolvedCandidate, undefined);
  }
});
test("absent or false singleton eligibility preserves group-only results despite named truth", () => {
  const fixture = modeledWinnerFixture();
  assert.equal(houseWinnerView(fixture).groups[0].resolvedCandidate, undefined);
  fixture.groups[0].namedSingletonResolutionEligible = false;
  assert.equal(houseWinnerView(fixture).groups[0].resolvedCandidate, undefined);
});
test("explicit eligibility rejects pooled categories, multiple members and malformed flags", () => {
  for (const fixture of [residualWinnerFixture(), samePartyWinnerFixture()]) {
    fixture.groups[fixture.groups.length - 1].namedSingletonResolutionEligible = true;
    assert.throws(() => houseWinnerView(fixture), TypeError);
  }
  const malformed = modeledWinnerFixture(); malformed.groups[0].namedSingletonResolutionEligible = "true";
  assert.throws(() => houseWinnerView(malformed), TypeError);
});

test("Other-group singletons never resolve to an individual candidate", () => {
  for (const kind of ["named_person", "source_reported_name", "pooled_or_residual"]) {
    const fixture = modeledWinnerFixture();
    fixture.groups[2].members = [{ id: "other-singleton", label: "Other group entry", kind }];
    fixture.groups[2].namedSingletonResolutionEligible = true;
    assert.throws(() => houseWinnerView(fixture), TypeError);
  }
  const fixture = modeledWinnerFixture();
  fixture.groups[2].members = [{ id: "other-singleton", label: "Source-listed entry 7", kind: "source_reported_name" }];
  const view = houseWinnerView(fixture);
  assert.equal(view.groups[2].resolvedCandidate, undefined);
});
