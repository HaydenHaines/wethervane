import test from "node:test";
import assert from "node:assert/strict";
import { summarizeHouseGroupResult } from "../lib/house-group-scoreboard.ts";
import { parseHouseGroupResult } from "../lib/house-group-result.ts";
import { groupResultFixture, groupPolicyBinding } from "./fixtures/house-group-result.mjs";

test("2018 coverage scores 434 known results and excludes the unavailable NC-09 target", () => {
  const result = parseHouseGroupResult(groupResultFixture(2018), groupPolicyBinding);
  assert.ok(result);
  const summary = summarizeHouseGroupResult(result);
  assert.equal(summary.forecastRaces, 435);
  assert.equal(summary.scoredRaces, 434);
  assert.equal(summary.unscoredRaces, 1);
  assert.equal(summary.uniqueFavorites, 434);
  assert.equal(summary.correctUniqueFavorites, 434);
  assert.equal(summary.topChanceTies, 0);
  assert.ok(Math.abs(summary.categoricalBrier - .26) < 1e-10);
});

test("exact top-probability ties are excluded from favorite accuracy while all eligible races remain scored", () => {
  const result = parseHouseGroupResult(groupResultFixture(2020), groupPolicyBinding);
  assert.ok(result);
  result.races[0].probabilities = { D: .4, R: .4, O: .2 };
  const summary = summarizeHouseGroupResult(result);
  assert.equal(summary.scoredRaces, 435);
  assert.equal(summary.topChanceTies, 1);
  assert.equal(summary.uniqueFavorites, 434);
});
