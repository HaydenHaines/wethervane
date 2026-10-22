import test from "node:test";
import assert from "node:assert/strict";
import { housePartyShareView } from "../lib/house-party-share-view.ts";
import { partyShareFixture, samePartyFixture, structuralZeroFixture, zeroDenominatorFixture, invalidIntervalFixture } from "./fixtures/house-party-share.mjs";
test("modeled shares retain declared levels and separate winner capability", () => {
  const view = housePartyShareView(partyShareFixture());
  assert.equal(view.estimate.mean, .65);
  assert.equal(view.estimate.intervalKind, "central");
  assert.deepEqual(view.estimate.intervals.map(i => i.level), [.8, .9]);
  assert.equal(view.candidateWinner.status, "unavailable");
  assert.deepEqual(view.observed, { status: "reported", numerator: 65, denominator: 100, share: .65 });
  assert.equal(view.winner, undefined);
});
test("same-party aggregates and structural endpoints remain structural", () => {
  for (const fixture of [samePartyFixture(), structuralZeroFixture()]) {
    const view = housePartyShareView(fixture);
    assert.equal(view.estimate.status, "structural");
    assert.equal(view.estimate.value, fixture.estimate.value);
    assert.equal(view.estimate.mean, undefined);
    assert.equal(view.candidateWinner.status, "unavailable");
  }
  const invalid = structuralZeroFixture(); invalid.estimate.absentParty = "R";
  assert.equal(housePartyShareView(invalid).estimate.status, "unavailable");
  const missingDenominator = samePartyFixture(); delete missingDenominator.estimate.denominatorStatus;
  assert.equal(housePartyShareView(missingDenominator).estimate.status, "unavailable");
});
test("zero denominator never becomes an observed zero share", () => {
  const view = housePartyShareView(zeroDenominatorFixture());
  assert.equal(view.observed.status, "unavailable"); assert.equal(view.observed.share, undefined);
  assert.match(view.observed.reason, /undefined/);
});
test("missing and invalid numerical evidence is withheld without losing observations", () => {
  const cases = [undefined, null, { status: "modeled", intervalKind: "central", mean: .65, intervals: [] }, { status: "modeled", intervalKind: "central", intervals: [{ level: .9, lower: 0, upper: 1 }] }, invalidIntervalFixture().estimate,
    ...[NaN, Infinity, -1, 1.1].map(mean => ({ status: "modeled", intervalKind: "central", mean, intervals: [{ level: .9, lower: 0, upper: 1 }] })),
    { status: "modeled", intervalKind: "central", mean: .65, intervals: [{ level: .8, lower: .1, upper: .9 }, { level: .9, lower: .2, upper: .8 }] }];
  for (const estimate of cases) {
    const view = housePartyShareView({ ...partyShareFixture(), estimate });
    assert.equal(view.estimate.status, "unavailable"); assert.equal(view.observed.share, .65);
  }
  const fixture = partyShareFixture(); fixture.observed.democraticVotes = undefined;
  assert.equal(housePartyShareView(fixture).observed.status, "unavailable");
});
test("production-like inputs and unknown stages are rejected", () => {
  for (const override of [{ fixtureOnly: false }, { sourceStatus: "reviewed" }, { stage: "final" }, { units: "percent" }, { candidateWinner: { status: "available" } }])
    assert.throws(() => housePartyShareView({ ...partyShareFixture(), ...override }), TypeError);
});

test("structural fixture observations agree with absent-party semantics", () => {
  for (const fixture of [samePartyFixture(), structuralZeroFixture()]) {
    const absentVotes = fixture.estimate.absentParty === "D" ? fixture.observed.democraticVotes : fixture.observed.republicanVotes;
    assert.equal(absentVotes, 0);
    const view = housePartyShareView(fixture);
    assert.equal(view.observed.status, "reported");
    assert.ok(view.observed.denominator > 0);
    assert.equal(view.observed.share, view.estimate.value);
  }
});
test("modeled intervals require their central method to be declared", () => {
  for (const method of [undefined, "highest_density", null]) {
    const fixture = partyShareFixture();
    fixture.estimate.intervalKind = method;
    const view = housePartyShareView(fixture);
    assert.equal(view.estimate.status, "unavailable");
    assert.equal(view.observed.share, .65);
  }
});
