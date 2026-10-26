import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { electionNightCalls, competitiveRaces } from "../lib/election-night-calls.ts";
import { toSenateElectionNightForecast, toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { houseLiveView } from "../lib/house-live-view.ts";

const read = relative => JSON.parse(readFileSync(new URL(`../${relative}`, import.meta.url), "utf8"));
const registry = read("public/data/explorer/registry.json");
const entry = registry.runs.find(r => r.id === "current-polling");
const senate = toSenateElectionNightForecast(read(`public${entry.href}`), entry);
const houseCatalog = read("data/house-live/catalog.json");
const houseEntry = houseCatalog.entries.find(r => r.id === "house-current-polling");
const house = toHouseElectionNightForecast(houseLiveView(read(`data/house-live/${houseEntry.result.path}`), houseEntry));

test("point seat totals sum every Senate and House race call under the published organizing scenarios", () => {
  for (const forecast of [senate, house]) {
    const result = electionNightCalls(forecast);
    const independent = { ...forecast.chamberScenario.retainedSeats };
    for (const race of forecast.races) {
      const winner = race.outcomes.find(outcome => outcome.id === race.modalPick.outcomeId);
      const bucket = winner.party === "D" || winner.party === "R" ? winner.party : "U";
      independent[bucket]++;
    }
    assert.deepEqual(result.seats, independent);
    assert.equal(Object.values(result.seats).reduce((a, b) => a + b), forecast.chamber === "senate" ? 100 : 435);
    assert.equal(new Set(result.competitive.map(r => r.id)).size, result.competitive.length);
    assert.ok(result.competitive.every(r => r.modalPick.probability <= .9 || r.pair && Math.abs(r.pair.meanMarginPp) <= 5));
    assert.ok(result.competitive.every((r, i, rows) => !i || rows[i - 1].modalPick.probability <= r.modalPick.probability));
  }
});

test("competitive boundaries include 90% and a five-point pair even with higher win chance", () => {
  const race = (id, probability, margin) => ({ id, modalPick: { probability }, pair: margin === null ? null : { meanMarginPp: margin } });
  const input = [race("a", .9, 20), race("b", .90001, -5), race("c", .90001, 5.01), race("d", .5, null)];
  assert.deepEqual(competitiveRaces(input).map(r => r.id), ["d", "a", "b"]);
  assert.deepEqual(input.map(r => r.id), ["a", "b", "c", "d"]);
});

test("unknown organizing assumptions, missing winners and incomplete seat totals withhold the chamber call", () => {
  const unknown = structuredClone(senate); unknown.chamberScenario.id = "unknown";
  assert.equal(electionNightCalls(unknown).seats, null);
  const party = structuredClone(senate); party.races[0].outcomes.find(o => o.id === party.races[0].modalPick.outcomeId).party = "unknown";
  assert.equal(electionNightCalls(party).seats, null);
  const incomplete = structuredClone(house); incomplete.races.pop();
  assert.equal(electionNightCalls(incomplete).seats, null);
});

test("updated forecasts immediately recompute calls and Senate 50–50 tie uses the supplied threshold", () => {
  const next = structuredClone(senate);
  const initial = electionNightCalls(next);
  const race = next.races.find(r => r.outcomes.find(o => o.id === r.modalPick.outcomeId).party === "R" && r.outcomes.some(o => o.party === "D"));
  const replacement = race.outcomes.find(o => o.party === "D");
  race.modalPick.outcomeId = replacement.id; race.modalPick.probability = .51;
  const updated = electionNightCalls(next);
  assert.equal(updated.seats.D, initial.seats.D + 1);
  assert.equal(updated.seats.R, initial.seats.R - 1);
  const tied = structuredClone(senate);
  tied.chamberScenario.retainedSeats = { D: 50, R: 50, U: 0 }; tied.races = [];
  assert.deepEqual(electionNightCalls(tied).seats, { D: 50, R: 50, U: 0 });
  assert.equal(electionNightCalls(tied).control, "R");
});
