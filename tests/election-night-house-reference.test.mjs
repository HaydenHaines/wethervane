import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { houseLiveView } from "../lib/house-live-view.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { matchHouseExpectedErrorReference, matchesHouseModalCall } from "../lib/election-night-house-reference.ts";
import { makeCompatibilityHouseReference } from "./fixtures/house-expected-reference.mjs";

const read = relative => JSON.parse(readFileSync(new URL(`../${relative}`, import.meta.url), "utf8"));
const compatibility = read("tests/fixtures/election-night/house-compatibility-sidecar.json");
const houseCatalog = read("data/house-live/catalog.json");
const entry = houseCatalog.entries.find(row => row.id === "house-current-fundamentals");
const native = read(`data/house-live/${entry.result.path}`);
assert.ok(parseHouseLiveResult(native));
const view = houseLiveView(native, entry);
const forecast = toHouseElectionNightForecast(view);
const h = digit => digit.repeat(64);
const fixture = () => makeCompatibilityHouseReference(forecast);

test("compatibility-only fixture proves 435 group calls and 412 exact conditional-pair errors remain separate", () => {
  const { catalog, qualification } = fixture();
  const result = matchHouseExpectedErrorReference(forecast, catalog, qualification);
  assert.ok(result);
  assert.equal(result.resultSha256, forecast.run.artifactSha256);
  assert.equal(result.perRace.length, 435);
  assert.equal(result.perRace.filter(row => row.availability === "available").length, 412);
  assert.equal(result.perRace.filter(row => row.availability === "inapplicable").length, 23);
  assert.ok(Math.abs(result.summary.modalGroupMissCount.mean - forecast.summary.expectedModalMisses) < 1e-8);
  assert.equal(result.summary.meanAbsolutePairMarginError.mean, compatibility.joint_summaries.mean_absolute_pair_margin_error_pp.mean);
  assert.equal(result.summary.central90PairIntervalCoverageCount.range90.lower, 281);
  assert.equal(result.perRace.find(row => row.raceId === "2026:CA-04:house:regular").availability, "inapplicable");
  assert.equal(forecast.races.find(row => row.id === "2026:CA-04:house:regular").modalPick.kind, "group");
  assert.equal(result.evaluationLock, false);
});

test("same run with changed result SHA, pair identity, group identity or draw rows cannot inherit expected errors", () => {
  for (const mutate of [
    ({ catalog }) => { catalog.entries[0].resultSha256 = h("2"); },
    ({ catalog }) => { catalog.entries[0].reference.race_expected_errors[forecast.races.find(r => r.pair).id].first_candidate_id = "wrong"; },
    ({ catalog }) => { catalog.entries[0].reference.modal_calls.find(c => c.kind === "group").member_candidate_ids.reverse(); },
    ({ catalog }) => { catalog.entries[0].reference.draw_layout.rows++; },
    ({ catalog }) => { catalog.entries[0].reference.draw_layout.pair_mask_in_winner_order[0] = false; },
    ({ qualification }) => { qualification.sourcePins.jointDrawsSha256 = h("3"); },
  ]) {
    const value = fixture(); mutate(value);
    assert.equal(matchHouseExpectedErrorReference(forecast, value.catalog, value.qualification), null);
  }
});

test("count PMFs, quantile boundaries and units must match finite joint rows", () => {
  for (const mutate of [
    metric => { metric.pmf[0] += 1 / compatibility.draw_layout.rows; },
    metric => { metric.quantiles.p05 += 1; },
    metric => { metric.units = "percentage_points"; },
    metric => { metric.pmf = metric.pmf.slice(1); },
    metric => { metric.pmf[4] = JSON.parse("1e400"); },
    metric => { metric.mean = Number.NaN; },
    metric => { metric.sd += 1; },
  ]) {
    const value = fixture(); mutate(value.catalog.entries[0].reference.joint_summaries.modal_group_miss_count);
    assert.equal(matchHouseExpectedErrorReference(forecast, value.catalog, value.qualification), null);
  }
});

test("House forced R/O tie chooses R before O; old lexical O call and reversed pair IDs fail", () => {
  const changedView = structuredClone(view), alaska = changedView.races.find(race => race.seatId === "AK-00");
  alaska.outcomes.forEach(outcome => { outcome.winProbability = outcome.party === "D" ? 0 : .5; });
  const changed = toHouseElectionNightForecast(changedView).races.find(race => race.seatId === "AK-00");
  assert.deepEqual(changed.modalPick.tiedOutcomeIds, [`${changed.id}:R`, `${changed.id}:O`]);
  assert.equal(changed.modalPick.outcomeId, `${changed.id}:R`);
  const republican = changed.outcomes.find(outcome => outcome.party === "R");
  const correct = { race_id: changed.id, outcome_id: "R", party_or_group: "R", kind: "candidate",
    member_candidate_ids: republican.members.map(member => member.id), tied_modal_outcomes: ["R", "O"] };
  assert.equal(matchesHouseModalCall(changed, correct), true);
  const oldLexical = { ...correct, outcome_id: "O", party_or_group: "O", kind: "group",
    member_candidate_ids: changed.outcomes.find(outcome => outcome.party === "O").members.map(member => member.id),
    tied_modal_outcomes: ["O", "R"] };
  assert.equal(matchesHouseModalCall(changed, oldLexical), false);
  const reversed = fixture(), pairRace = forecast.races.find(race => race.pair);
  const nativeRow = reversed.catalog.entries[0].reference.race_expected_errors[pairRace.id];
  [nativeRow.first_candidate_id, nativeRow.second_candidate_id] = [nativeRow.second_candidate_id, nativeRow.first_candidate_id];
  assert.equal(matchHouseExpectedErrorReference(forecast, reversed.catalog, reversed.qualification), null);
});

test("record key order is irrelevant, but changed issuance and absent pair status are rejected", () => {
  const value = fixture();
  value.catalog.entries[0].configuration = Object.fromEntries(Object.entries(value.catalog.entries[0].configuration).reverse());
  value.qualification.sourcePins = Object.fromEntries(Object.entries(value.qualification.sourcePins).reverse());
  assert.ok(matchHouseExpectedErrorReference(forecast, value.catalog, value.qualification));
  const late = fixture(); late.catalog.entries[0].referenceIssuedAt = "2020-01-01T00:00:00Z";
  late.qualification.referenceIssuedAt = late.catalog.entries[0].referenceIssuedAt;
  late.catalog.entries[0].reference.issued_at = late.catalog.entries[0].referenceIssuedAt;
  assert.equal(matchHouseExpectedErrorReference(forecast, late.catalog, late.qualification), null);
  const noPair = fixture(); noPair.catalog.entries[0].reference.race_expected_errors["2026:CA-04:house:regular"].availability = "available";
  assert.equal(matchHouseExpectedErrorReference(forecast, noPair.catalog, noPair.qualification), null);
});
