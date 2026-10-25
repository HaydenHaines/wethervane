import assert from "node:assert/strict";
import test from "node:test";
import { compareHouseLiveViews } from "../lib/house-live-comparison.ts";
import { houseLiveView } from "../lib/house-live-view.ts";
import { makeHouseLiveResult, SYNTHETIC_CONFIGURATIONS } from "./fixtures/house-live-fixtures.mjs";

const FIRST_HASH = "a".repeat(64);
const SECOND_HASH = "b".repeat(64);

function view(configuration, hash) {
  const result = makeHouseLiveResult(configuration);
  if (configuration.polling === "none") result.races[0].method = "Synthetic fundamentals-only election method";
  result.qualification.publication_status = "reviewed";
  result.joint_draws.sha256 = hash;
  result.chamber.joint_draws_sha256 = hash;
  return houseLiveView(result, {
    id: result.id,
    label: `Synthetic ${configuration.base} ${configuration.polling}`,
    configuration: structuredClone(configuration),
    result: { path: `synthetic/${result.id}.json`, sha256: hash },
  });
}

function pair() {
  return [view(SYNTHETIC_CONFIGURATIONS[1], FIRST_HASH),
    view(SYNTHETIC_CONFIGURATIONS[0], SECOND_HASH)];
}

function comparable(a, b) {
  const result = compareHouseLiveViews(a, b);
  assert.equal(result.status, "comparable", result.reason);
  return result.summary;
}

test("same current roster compares each D/R/O group chance and keeps run evidence separate", () => {
  const [primary, reference] = pair();
  // Simulate different, independently projected saved results; comparison
  // consumes their summaries and does not pair or blend joint rows.
  primary.races[0].outcomes[0].winProbability = 0.6;
  primary.races[0].outcomes[1].winProbability = 0.4;
  reference.races[0].pollCount = 1;
  reference.run.sourceCutoffs[1].at = "2026-10-19T12:00:00.000Z";

  const summary = comparable(primary, reference);
  assert.equal(summary.races.length, 435);
  assert.deepEqual(summary.races[0].outcomes.map(row => row.party), ["D", "R", "O"]);
  assert.deepEqual(summary.races[0].outcomes[0].probability,
    { primary: 0.6, reference: 0.5, difference: 0.09999999999999998 });
  assert.deepEqual(summary.races[0].outcomes[2].probability,
    { primary: 0, reference: 0, difference: 0 });
  assert.equal(summary.races[0].outcomes[2].supported, false);
  assert.deepEqual(summary.contextDifferences.modelSlots, ["polling"]);
  assert.equal(summary.contextDifferences.sourceCutoffsDiffer, true);
  assert.deepEqual(summary.contextDifferences.pollCounts,
    [{ raceId: primary.races[0].id, primary: 0, reference: 1 }]);
  assert.deepEqual(summary.contextDifferences.methods,
    [{ raceId: primary.races[0].id, primary: primary.races[0].method,
      reference: reference.races[0].method }]);
  assert.equal(summary.primary.jointDraws.sha256, FIRST_HASH);
  assert.equal(summary.reference.jointDraws.sha256, SECOND_HASH);
  assert.equal(summary.chamber.status, "comparable");
  assert.deepEqual(summary.chamber.expectedSeats.D,
    { primary: 217.75, reference: 217.75, difference: 0 });
  assert.deepEqual(summary.chamber.control.D,
    { primary: 0.5, reference: 0.5, difference: 0 });
  assert.equal(summary.conditionalPairs.status, "not_compared");
  assert.match(summary.interpretation, /unobserved.*do not score accuracy/);
  assert.equal("accuracy" in summary, false);
  assert.equal("differenceInterval" in summary, false);
});

test("different result identity, election, cutoff, and geography are required", () => {
  const [primary, reference] = pair();
  const check = (mutate, pattern) => {
    const copy = structuredClone(reference);
    mutate(copy);
    const result = compareHouseLiveViews(primary, copy);
    assert.equal(result.status, "incomparable");
    assert.match(result.reason, pattern);
  };
  check(copy => { copy.run.resultSha256 = FIRST_HASH; }, /same result artifact/);
  check(copy => { copy.run.electionDate = "2028-11-07"; }, /same 2026 House election/);
  check(copy => { copy.run.cutoffAt = "2026-10-19T00:00:00.000Z"; }, /cutoff/);
  check(copy => { copy.geographySha256 = "c".repeat(64); }, /geography/);
  check(copy => { copy.mapVintage = "old-map"; }, /map/);
  const equivalent = structuredClone(reference);
  equivalent.run.cutoffAt = "2026-10-20T00:00:00+00:00";
  equivalent.run.sourceCutoffs[0].at = "2026-10-19T00:00:00+00:00";
  assert.equal(comparable(primary, equivalent).contextDifferences.sourceCutoffsDiffer, false);
});

test("435 race identities, candidate groups, membership, and rules must match", () => {
  const [primary, reference] = pair();
  const check = (mutate, pattern) => {
    const copy = structuredClone(reference);
    mutate(copy);
    const result = compareHouseLiveViews(primary, copy);
    assert.equal(result.status, "incomparable");
    assert.match(result.reason, pattern);
  };
  check(copy => { copy.races.pop(); }, /435-district roster/);
  check(copy => { copy.races[0].id = "different-race"; }, /support/);
  check(copy => { copy.races[0].candidates[0].name = "Different candidate"; }, /candidates/);
  check(copy => { copy.races[0].outcomes[0].members.pop(); }, /support/);
  check(copy => { copy.races[0].ruleFamily = "ranked_choice"; }, /rules/);
  check(copy => { copy.races[0].observed = { outcomeId: "D", winnerName: "Invented", winnerParty: "D", sourceUrls: [] }; }, /roster/);
  const renamed = structuredClone(reference);
  renamed.races[0].outcomes[0].name = "Different display label for same group";
  assert.equal(comparable(primary, renamed).races[0].outcomes[0].label,
    primary.races[0].outcomes[0].name);
});

test("chamber comparison is withheld when either result or its scenario is incompatible", () => {
  const [primary, reference] = pair();
  const missing = structuredClone(reference);
  missing.chamber = null;
  missing.chamberPrecision = null;
  assert.equal(comparable(primary, missing).chamber.status, "unavailable");

  const different = structuredClone(reference);
  different.chamber.thresholds.D = 219;
  assert.match(comparable(primary, different).chamber.reason, /threshold/);

  const allocatedOther = structuredClone(reference);
  allocatedOther.chamber.retainedSeats.U = 1;
  assert.match(comparable(primary, allocatedOther).chamber.reason, /unassigned-seat/);
});

test("different joint hashes with equal marginals do not imply paired draws or a covariance result", () => {
  const [primary, reference] = pair();
  const summary = comparable(primary, reference);
  assert.notEqual(summary.primary.jointDraws.sha256, summary.reference.jointDraws.sha256);
  assert.equal(summary.chamber.status, "comparable");
  assert.equal(summary.races.every(race => race.outcomes.every(row => row.probability.difference === 0)), true);
  assert.equal("jointDifference" in summary, false);
  assert.equal("covariance" in summary, false);
  assert.match(summary.interpretation, /Each run keeps its own joint-draw evidence/);
});
