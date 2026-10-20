import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { deriveScoreboard, compareScoreboards } from "../lib/scoreboard.ts";

const read = (name) => JSON.parse(readFileSync(
  new URL(`../public/data/explorer/${name}.json`, import.meta.url), "utf8",
));
const actual2018 = read("history-2018-FP-1");
const scenario = (bundle) => bundle.scenarios.find((item) => item.id === bundle.defaultScenarioId);
const near = (actual, expected, tolerance = 1e-12) => assert.ok(
  Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`,
);

function race(id, probabilities, observed = "a") {
  return {
    id, state: "AA", seatId: `seat:${id}`, electionLabel: "Regular election", notes: [],
    outcomes: probabilities.map((p, index) => {
      const outcomeId = String.fromCharCode(97 + index);
      return {
        id: outcomeId, name: `Outcome ${outcomeId}`, party: outcomeId,
        kind: "candidate", members: [{ id: `person:${outcomeId}`, name: `Outcome ${outcomeId}`, party: outcomeId, sourceUrls: [] }],
        winProbability: p,
      };
    }),
    ...(observed === null ? {} : {
      observed: { outcomeId: observed, winnerName: `Outcome ${observed}`, winnerParty: observed, sourceUrls: [] },
    }),
  };
}

function synthetic(races) {
  const bundle = structuredClone(actual2018);
  bundle.run = { ...bundle.run, id: "synthetic", year: 2042 };
  bundle.races = races;
  return bundle;
}

test("2018 standard polling correctly favors 31 of 35 outcome groups with raw Brier 0.135235", () => {
  const result = deriveScoreboard(actual2018, scenario(actual2018));
  assert.equal(result.status, "available");
  assert.deepEqual(result.coverage, { total: 35, observed: 35, scored: 35, missingRaceIds: [], invalidRaces: [] });
  assert.equal(result.favorites.correct, 31);
  assert.equal(result.favorites.decisive, 35);
  near(result.favorites.expectedCorrect, 31.2219125);
  near(result.brier.mean, 0.13523522369642857);
  assert.deepEqual(result.favorites.missRaceIds, [
    "2018:FL:senate:regular", "2018:ME:senate:regular", "2018:MT:senate:regular", "2018:NV:senate:regular",
  ]);
  assert.deepEqual(result.favorites.tiedRaceIds, []);
  assert.equal(result.seats.length, 3);
  assert.deepEqual(result.seats.map((item) => item.observed), [47, 53, 0]);
  const d = result.seats.find((item) => item.party === "D");
  near(d.error, d.expected - 47);
  assert.equal(d.median, scenario(actual2018).seats.D.seatQuantiles.p50);
  assert.equal(d.p05, scenario(actual2018).seats.D.seatQuantiles.p05);
  assert.equal(d.p95, scenario(actual2018).seats.D.seatQuantiles.p95);
  near(result.control.observedProbability, 0.9911125);
  assert.equal(result.control.observed, "R");
  assert.deepEqual(result.control.mostLikely, ["R"]);
});

test("more correct favorites does not imply a better probabilistic forecast", () => {
  const f = read("history-2018-F-1"), w = read("history-2018-W-1");
  const fundamentals = deriveScoreboard(f, scenario(f));
  const winner = deriveScoreboard(w, scenario(w));
  assert.equal(fundamentals.favorites.correct, 29);
  assert.equal(winner.favorites.correct, 27);
  near(fundamentals.brier.mean, 0.24178502738392857);
  near(winner.brier.mean, 0.2271712509107143);
  assert.ok(fundamentals.brier.mean > winner.brier.mean);
  const comparison = compareScoreboards(f, scenario(f), w, scenario(w));
  assert.equal(comparison.status, "matched");
  assert.equal(comparison.favoritesDifference, 2);
  assert.ok(comparison.brierDifference > 0);
});

test("2022 standard polling favors all 35 outcomes but its expected count remains below 35", () => {
  const bundle = read("history-2022-FP-1");
  const result = deriveScoreboard(bundle, scenario(bundle));
  assert.equal(result.favorites.correct, 35);
  assert.equal(result.favorites.decisive, 35);
  near(result.favorites.expectedCorrect, 32.0940375);
  near(result.brier.mean, 0.04655610560714286);
  assert.deepEqual(result.favorites.missRaceIds, []);
  assert.deepEqual(result.seats.map((item) => item.observed), [51, 49, 0]);
  assert.equal(result.control.observed, "D");
  near(result.control.observedProbability, 0.87895);
});

test("a correctly predicted party group does not become separate candidate forecasts", () => {
  const groupRace = race("group", [0.8, 0.2]);
  groupRace.outcomes[0].kind = "group";
  groupRace.outcomes[0].members.push({ id: "person:a2", name: "Other group member", party: "a", sourceUrls: [] });
  groupRace.observed.winnerName = "Other group member";
  const bundle = synthetic([groupRace]);
  const result = deriveScoreboard(bundle, scenario(bundle));
  assert.equal(result.favorites.correct, 1);
  assert.equal(result.favorites.decisive, 1);
  near(result.favorites.expectedCorrect, 0.8);
  near(result.brier.mean, 0.08);
  assert.equal(result.coverage.total, 1);
});

test("raw multiclass Brier is a sum over categories, ranging from zero to two", () => {
  for (const [probabilities, observed, expected] of [
    [[1, 0, 0], "a", 0],
    [[1, 0, 0], "b", 2],
    [[1 / 3, 1 / 3, 1 / 3], "c", 2 / 3],
  ]) {
    const bundle = synthetic([race("r", probabilities, observed)]);
    near(deriveScoreboard(bundle, scenario(bundle)).brier.mean, expected);
  }
});

test("exact favorite ties are excluded from counts and expected wins, but remain in Brier", () => {
  const bundle = synthetic([race("tie", [0.4, 0.4, 0.2], "c"), race("unique", [0.7, 0.3])]);
  const result = deriveScoreboard(bundle, scenario(bundle));
  assert.equal(result.favorites.correct, 1);
  assert.equal(result.favorites.decisive, 1);
  near(result.favorites.expectedCorrect, 0.7);
  assert.deepEqual(result.favorites.tiedRaceIds, ["tie"]);
  assert.deepEqual(result.favorites.missRaceIds, []);
  assert.equal(result.brier.count, 2);
  near(result.brier.mean, (0.96 + 0.18) / 2);
  const close = synthetic([race("close", [0.50000000001, 0.49999999999])]);
  assert.equal(deriveScoreboard(close, scenario(close)).favorites.decisive, 1);
});

test("missing and invalid observations have explicit coverage and matching denominators", () => {
  const bundle = synthetic([
    race("valid", [0.7, 0.3]),
    race("missing", [0.9, 0.1], null),
    race("nan", [NaN, 0.2]),
    race("outside-support", [0.5, 0.5], "z"),
    race("tie", [0.5, 0.5]),
  ]);
  const result = deriveScoreboard(bundle, scenario(bundle));
  assert.deepEqual({ total: result.coverage.total, observed: result.coverage.observed, scored: result.coverage.scored }, { total: 5, observed: 4, scored: 2 });
  assert.deepEqual(result.coverage.missingRaceIds, ["missing"]);
  assert.deepEqual(result.coverage.invalidRaces.map((item) => item.raceId), ["nan", "outside-support"]);
  assert.ok(result.coverage.invalidRaces.every((item) => item.reason.length > 0));
  assert.equal(result.favorites.correct, 1);
  assert.equal(result.favorites.decisive, 1);
  near(result.favorites.expectedCorrect, 0.7);
  near(result.brier.mean, 0.34);
});

test("invalid categorical distributions never produce a historical score", () => {
  for (const probabilities of [[-0.1, 1.1], [Infinity, 0], [NaN, 0.2], [0.3, 0.4], ["0.5", 0.5]]) {
    const bundle = synthetic([race("invalid", probabilities)]);
    const result = deriveScoreboard(bundle, scenario(bundle));
    assert.equal(result.status, "unavailable");
    assert.equal(result.brier, null);
    assert.equal(result.favorites, null);
    assert.equal(result.coverage.invalidRaces.length, 1);
  }
  const duplicate = synthetic([race("same", [1]), race("same", [1])]);
  assert.equal(deriveScoreboard(duplicate, scenario(duplicate)).coverage.invalidRaces.length, 2);
  const overlap = synthetic([race("overlap", [0.8, 0.2])]);
  overlap.races[0].outcomes[1].members[0].id = "person:a";
  assert.equal(deriveScoreboard(overlap, scenario(overlap)).status, "unavailable");
});

test("unobserved current forecasts have no favorite count, Brier, actual control, or seat error", () => {
  const bundle = read("current-polling");
  const result = deriveScoreboard(bundle, scenario(bundle));
  assert.equal(result.status, "unavailable");
  assert.match(result.reason, /Observed race outcomes are unavailable/);
  assert.equal(result.favorites, null);
  assert.equal(result.brier, null);
  assert.equal(result.control, null);
  assert.equal(result.coverage.scored, 0);
  assert.equal(result.coverage.missingRaceIds.length, bundle.races.length);
  assert.ok(result.seats.every((row) => row.observed === null && row.error === null));
});

test("most-likely chamber control preserves ties and highlights the actual category probability", () => {
  const bundle = synthetic([race("r", [0.8, 0.2])]);
  const selected = scenario(bundle);
  selected.control = { description: "Tie", democraticMajorityProbability: 0.4, republicanMajorityProbability: 0.4, unresolvedOrganizingProbability: 0.2 };
  selected.performance.observedControl = "U";
  const result = deriveScoreboard(bundle, selected);
  assert.deepEqual(result.control.mostLikely, ["D", "R"]);
  assert.equal(result.control.observed, "U");
  assert.equal(result.control.observedProbability, 0.2);
  selected.control.democraticMajorityProbability = NaN;
  const invalid = deriveScoreboard(bundle, selected);
  assert.equal(invalid.control, null);
  assert.ok(invalid.issues.some((issue) => /control probabilities/.test(issue)));
});

test("invalid seat distributions are disclosed instead of inventing summaries", () => {
  const bundle = synthetic([race("r", [1])]);
  const selected = scenario(bundle);
  selected.seats.D.pmf[0] = -1;
  const result = deriveScoreboard(bundle, selected);
  assert.equal(result.seats.find((row) => row.party === "D"), undefined);
  assert.ok(result.issues.some((issue) => /D seat distribution/.test(issue)));
});

test("matched comparisons ignore array order and use selected-minus-baseline score deltas", () => {
  const f = read("history-2018-F-1");
  const comparison = compareScoreboards(actual2018, scenario(actual2018), f, scenario(f));
  assert.equal(comparison.status, "matched");
  near(comparison.brierDifference, 0.13523522369642857 - 0.24178502738392857);
  assert.equal(comparison.favoritesDifference, 2);
  const reordered = structuredClone(actual2018);
  reordered.races.reverse();
  for (const item of reordered.races) {
    item.outcomes.reverse();
    for (const outcome of item.outcomes) outcome.members.reverse();
  }
  const same = compareScoreboards(actual2018, scenario(actual2018), reordered, scenario(reordered));
  assert.equal(same.status, "matched");
  near(same.brierDifference, 0);
});

test("date, target, law, and scenario differences cannot become matched skill claims", () => {
  const changes = [
    [(bundle) => bundle.run.year++, /years/],
    [(bundle) => bundle.run.dateId = "h30", /dates/],
    [(bundle) => bundle.run.forecastAt = "2018-10-22", /dates/],
    [(bundle) => bundle.run.cutoffAt = "2018-10-22", /cutoffs/],
    [(bundle) => bundle.run.targetId = "another-target", /targets/],
    [(bundle) => bundle.run.lawId = "another-law", /laws/],
    [(bundle) => scenario(bundle).id = "another-scenario", /scenarios/],
    [(bundle) => scenario(bundle).thresholds.D++, /thresholds/],
  ];
  for (const [change, reason] of changes) {
    const baseline = structuredClone(actual2018);
    const selected = baseline.scenarios[0];
    change(baseline);
    const comparison = compareScoreboards(actual2018, scenario(actual2018), baseline, selected);
    assert.equal(comparison.status, "unavailable");
    assert.match(comparison.reason, reason);
    assert.equal(comparison.brierDifference, null);
  }
});

test("different outcome or member support prevents comparison even if probabilities match", () => {
  for (const change of [
    (bundle) => bundle.races.pop(),
    (bundle) => bundle.races[0].outcomes[0].members[0].id = "different-person",
    (bundle) => bundle.races[0].outcomes[0].members[0].party = "different-party",
    (bundle) => bundle.races[0].seatId = "different-seat",
    (bundle) => bundle.races.find((item) => item.state === "CA").outcomes[0].members.pop(),
  ]) {
    const baseline = structuredClone(actual2018);
    change(baseline);
    const comparison = compareScoreboards(actual2018, scenario(actual2018), baseline, scenario(baseline));
    assert.equal(comparison.status, "unavailable");
    assert.match(comparison.reason, /support/);
  }
});

test("observed race and chamber truth must match, including the individual winner within a group", () => {
  for (const change of [
    (bundle) => bundle.races.find((item) => item.state === "CA").observed.winnerName = "Kevin De Leon",
    (bundle) => delete bundle.races[0].observed,
    (bundle) => scenario(bundle).performance.observedSeats.D++,
    (bundle) => scenario(bundle).performance.observedControl = "D",
  ]) {
    const baseline = structuredClone(actual2018);
    change(baseline);
    const comparison = compareScoreboards(actual2018, scenario(actual2018), baseline, scenario(baseline));
    assert.equal(comparison.status, "unavailable");
    assert.match(comparison.reason, /Observed/);
  }
});

test("invalid observed probability vectors are never silently excluded from a comparison", () => {
  const baseline = structuredClone(actual2018);
  baseline.races[0].outcomes[0].winProbability = NaN;
  const comparison = compareScoreboards(actual2018, scenario(actual2018), baseline, scenario(baseline));
  assert.equal(comparison.status, "unavailable");
  assert.match(comparison.reason, /Invalid observed/);
});

test("derivation does not mutate bundles or source observations", () => {
  const original = JSON.stringify(actual2018);
  deriveScoreboard(actual2018, scenario(actual2018));
  compareScoreboards(actual2018, scenario(actual2018), actual2018, scenario(actual2018));
  assert.equal(JSON.stringify(actual2018), original);
});
