import assert from "node:assert/strict";
import test from "node:test";
import { COLORS, chamberOutlook, raceRating, raceColor, raceTextColor, visualParty } from "../lib/presentation.ts";

function categorical(chance, party = "D") {
  return {
    id: "race", state: "AA", seatId: "seat", electionLabel: "Regular election", notes: [],
    outcomes: [
      { id: party === "D" || party === "R" ? party : "other", party, name: "Leader", kind: "candidate", members: [], winProbability: chance },
      { id: party === "R" ? "D" : "R", party: party === "R" ? "D" : "R", name: "Second", kind: "candidate", members: [], winProbability: 1 - chance },
    ],
  };
}

function principalPair(chance, party = "D") {
  return {
    id: "race", state: "AA", seatId: "seat", notes: [],
    candidates: [{ id: "first", party }, { id: "second", party: "R" }],
    firstWinProbability: chance, secondWinProbability: 1 - chance,
    firstShareMean: 0.5, shareQuantiles: { p05: 0.4, p50: 0.5, p95: 0.6 },
  };
}

test("accepted leading-outcome thresholds include exactly 60% in Tossup and exactly 90% in Likely", () => {
  for (const build of [categorical, principalPair]) {
    for (const [chance, rating] of [
      [0.5, "Tossup"], [0.6, "Tossup"],
      [0.6000000001, "Likely"], [0.65, "Likely"], [0.9, "Likely"],
      [0.9000000001, "Safe"], [0.98, "Safe"], [1, "Safe"],
    ]) {
      assert.equal(raceRating(build(chance)), rating, `${chance} should be ${rating}`);
    }
  }
});

test("map fill follows the same threshold and leading party for all supported parties", () => {
  for (const [party, safe, likely] of [
    ["D", COLORS.D, COLORS.likelyD], ["R", COLORS.R, COLORS.likelyR],
    ["I", COLORS.I, COLORS.likelyI], ["IND", COLORS.I, COLORS.likelyI],
    ["LIB", COLORS.O, COLORS.likelyO],
  ]) {
    assert.equal(raceColor(categorical(0.6, party)), COLORS.competitive);
    assert.equal(raceColor(categorical(0.6000000001, party)), likely);
    assert.equal(raceColor(categorical(0.9, party)), likely);
    assert.equal(raceColor(categorical(0.9000000001, party)), safe);
    assert.equal(raceTextColor(categorical(0.6, party)), COLORS.competitive);
    assert.equal(raceTextColor(categorical(0.9, party)), safe);
  }
});

test("the second contender's winning probability controls legacy-pair color and rating", () => {
  const race = principalPair(0.1);
  race.secondWinProbability = 0.9;
  assert.equal(raceRating(race), "Likely");
  assert.equal(raceColor(race), COLORS.likelyR);
  race.firstWinProbability = 0.0999999999;
  race.secondWinProbability = 0.9000000001;
  assert.equal(raceRating(race), "Safe");
  assert.equal(raceColor(race), COLORS.R);
});

test("categorical ratings use the leading outcome rather than a majority-vote or first-outcome assumption", () => {
  const race = categorical(0.2);
  race.outcomes[1].winProbability = 0.8;
  assert.equal(raceRating(race), "Likely");
  assert.equal(raceColor(race), COLORS.likelyR);
  race.outcomes = [
    { id: "a", party: "IND", winProbability: 0.4 },
    { id: "b", party: "D", winProbability: 0.35 },
    { id: "c", party: "R", winProbability: 0.25 },
  ];
  assert.equal(raceRating(race), "Tossup");
  assert.equal(raceColor(race), COLORS.competitive);
});

test("party groups and independent candidates retain their identity through ratings", () => {
  const group = categorical(0.95, "DFL");
  group.outcomes[0].id = "D";
  group.outcomes[0].kind = "group";
  assert.equal(visualParty(group.outcomes[0]), "D");
  assert.equal(raceColor(group), COLORS.D);
  const independent = categorical(0.95, "IND");
  assert.equal(visualParty(independent.outcomes[0]), "I");
  assert.equal(raceColor(independent), COLORS.I);
});

test("chamber color follows unconditioned control probabilities and treats exactly 60% as tossup", () => {
  const control = (d, r, u = 0) => ({ democraticMajorityProbability: d, republicanMajorityProbability: r, unresolvedOrganizingProbability: u });
  assert.equal(chamberOutlook(control(0.6, 0.4)).color, COLORS.competitive);
  assert.equal(chamberOutlook(control(0.4, 0.6)).color, COLORS.competitive);
  assert.equal(chamberOutlook(control(0.6001, 0.3999)).color, COLORS.D);
  assert.equal(chamberOutlook(control(0.3999, 0.6001)).color, COLORS.R);
  assert.equal(chamberOutlook(control(0.55, 0.1, 0.35)).party, "tossup");
  assert.equal(chamberOutlook(control(0.1, 0.1, 0.8)).party, "U");
  assert.equal(chamberOutlook().color, undefined);
  assert.equal(chamberOutlook(control(NaN, 0.5)).color, undefined);
  assert.equal(chamberOutlook(control(0.9, 0.9)).color, undefined);
});
