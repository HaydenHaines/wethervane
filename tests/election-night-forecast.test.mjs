import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import { toSenateElectionNightForecast, toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { houseLiveView } from "../lib/house-live-view.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { houseNy21Notice } from "../lib/house-ny21-notice.ts";

const read = relative => JSON.parse(readFileSync(new URL(`../${relative}`, import.meta.url), "utf8"));
const sha = relative => createHash("sha256").update(readFileSync(new URL(`../${relative}`, import.meta.url))).digest("hex");
const registry = read("public/data/explorer/registry.json");
const senateEntries = registry.runs.filter(r => r.year === 2026);
const houseCatalog = read("data/house-live/catalog.json");
function senate(entry = senateEntries.find(r => r.id === "current-polling")) {
  return { entry: structuredClone(entry), bundle: read(`public${entry.href}`) };
}
function house(entry = houseCatalog.entries.find(r => r.id === "house-current-polling")) {
  const native = read(`data/house-live/${entry.result.path}`);
  assert.ok(parseHouseLiveResult(native), "actual native result must satisfy its existing structural contract");
  return { native, view: houseLiveView(native, entry) };
}
function freeze(value) {
  if (value && typeof value === "object") { Object.freeze(value); Object.values(value).forEach(freeze); }
  return value;
}
function near(actual, expected, message) { assert.ok(Math.abs(actual - expected) < 1e-10, `${message}: ${actual} vs ${expected}`); }
function assertSummary(result, original) {
  const probabilities = original.map(r => r.outcomes.map(o => o.winProbability));
  const expected = probabilities.map(p => 1 - [...p].sort((a, b) => b - a)[0]).reduce((a, b) => a + b, 0);
  near(result.summary.expectedModalMisses, expected, "expected modal misses by linearity");
  assert.equal(result.summary.raceCount, original.length);
  assert.equal(result.summary.tiePolicy, result.chamber === "house" ? "house_declared_D_R_O_party_order" : "lexicographic_outcome_id");
  assert.equal(result.editionStatus, "dated_snapshot");
  assert.ok(!("evaluationLocked" in result));
  assert.ok(!("expectedMarginError" in result.summary));
  assert.ok(!("missInterval" in result.summary));
  assert.match(result.summary.interpretation, /not expected margin error or a predictive interval/);
}
function assertPair(actual, first, second, mean, quantiles) {
  assert.equal(actual.firstCandidateId, first.id);
  assert.equal(actual.secondCandidateId, second.id);
  assert.equal(actual.firstCandidateName, first.name);
  assert.equal(actual.secondCandidateName, second.name);
  assert.equal(actual.denominator, "Share of votes for the two named principal contenders");
  assert.equal(actual.units, "percentage_points");
  assert.equal(actual.orientation, "first_minus_second");
  // Independent algebra checks the share-to-margin transformation and both interval endpoints.
  near(actual.meanMarginPp, 200 * mean - 100, "pair mean");
  near(actual.medianMarginPp, 200 * quantiles.p50 - 100, "pair median");
  near(actual.range90.lower, 200 * quantiles.p05 - 100, "lower margin endpoint");
  near(actual.range90.upper, 200 * quantiles.p95 - 100, "upper margin endpoint");
}

test("all 16 actual current Senate artifacts preserve exact editions, winner vectors, margins and native default scenario without mutation", () => {
  assert.equal(senateEntries.length, 16);
  for (const sourceEntry of senateEntries) {
    const { bundle, entry } = senate(sourceEntry);
    assert.equal(sha(`public${entry.href}`), entry.sha256);
    const before = JSON.stringify({ bundle, entry }); freeze(bundle); freeze(entry);
    const result = toSenateElectionNightForecast(bundle, entry);
    assert.equal(result.chamber, "senate"); assert.equal(result.races.length, 35);
    assert.equal(result.summary.pairRaceCount, 35);
    assert.deepEqual(result.run, { id: entry.id, label: entry.label, year: 2026, configuration: entry.components,
      artifactSha256: entry.sha256, issuedAt: entry.issuedAt, cutoffAt: entry.cutoffAt, strictAsOf: entry.strictAsOf,
      sourceCutoffs: (entry.inputCutoffs ?? []).map(s => ({ label: s.label, at: s.at })), targetId: entry.targetId, lawId: entry.lawId });
    assert.deepEqual(result.sourceHashes, bundle.provenance.sources);
    assert.deepEqual(result.chamberScenario, bundle.scenarios.find(s => s.id === bundle.defaultScenarioId));
    assertSummary(result, bundle.races);
    result.races.forEach((r, i) => {
      const source = bundle.races[i], [first, second] = source.outcomes;
      assert.deepEqual(r.outcomes, source.outcomes); assert.deepEqual(r.notes, source.notes);
      assert.deepEqual(r.sourceUrls, [...new Set(source.outcomes.flatMap(o => o.members.flatMap(m => m.sourceUrls)))]);
      assertPair(r.pair, first, second, first.share.mean, first.share.quantiles);
      assert.equal(r.pair.firstExceedsHalf, first.winProbability);
      assert.equal(r.pair.winnerRelationship, "same_event");
      assert.equal(r.pair.closeWithin2Pp, null); assert.equal(r.pair.firstAtLeast10Pp, null); assert.equal(r.pair.secondAtLeast10Pp, null);
      assert.deepEqual(r.pair.qualifications, source.notes);
    });
    bundle.limitations.forEach(note => assert.ok(result.qualification.includes(note)));
    assert.equal(JSON.stringify({ bundle, entry }), before);
    // Returned data is detached as well as read-only with respect to source inputs.
    result.races[0].outcomes[0].members[0].name = "CHANGED OUTPUT ONLY";
    result.chamberScenario.seats.D.pmf[0] = -10;
    result.races[0].pair.qualifications.push("CHANGED OUTPUT ONLY");
    assert.equal(JSON.stringify({ bundle, entry }), before);
  }
});

const noPairDistricts = ["AZ-03", "CA-04", "CA-07", "CA-11", "CA-12", "CA-14", "CA-29", "CA-34", "CA-37", "CA-40",
  "FL-10", "LA-01", "LA-03", "LA-04", "LA-05", "LA-06", "MA-01", "MA-02", "MA-05", "MA-07", "NJ-08", "PA-03", "WI-02"];
test("all 16 actual House artifacts preserve 435 listed winner vectors, exact 412/23 pair support and saved bands without allocating group mass", () => {
  assert.equal(houseCatalog.entries.length, 16);
  for (const entry of houseCatalog.entries) {
    const { native, view } = house(entry);
    assert.equal(sha(`data/house-live/${entry.result.path}`), entry.result.sha256);
    const before = JSON.stringify(view); freeze(view);
    const result = toHouseElectionNightForecast(view);
    assert.equal(result.chamber, "house"); assert.equal(result.races.length, 435);
    assert.deepEqual(result.inputNotices, [houseNy21Notice]);
    assert.equal(result.summary.pairRaceCount, 412);
    assert.deepEqual(result.races.filter(r => r.pair === null).map(r => r.seatId), noPairDistricts);
    assert.equal(result.run.id, native.id); assert.deepEqual(result.run.configuration, native.configuration);
    assert.equal(result.run.artifactSha256, entry.result.sha256);
    assert.equal(result.run.issuedAt, native.issued_at); assert.equal(result.run.cutoffAt, native.forecast_cutoff);
    assert.equal(result.run.strictAsOf, native.strict_as_of);
    assert.deepEqual(result.run.sourceCutoffs, native.source_cutoffs.map(s => ({ label: s.role, at: s.at })));
    assert.equal(result.run.targetId, null); assert.equal(result.run.lawId, null);
    assert.deepEqual(result.chamberScenario, view.chamber);
    assert.equal(result.sourceHashes.jointDrawsSha256, native.joint_draws.sha256);
    assertSummary(result, view.races);
    result.races.forEach((r, i) => {
      const source = native.races[i], displayed = view.races[i], q = source.conditional_shares;
      assert.deepEqual(r.outcomes, displayed.outcomes);
      assert.deepEqual(r.notes, source.qualifications);
      assert.equal(r.ruleFamily, source.rule_family); assert.equal(r.method, source.method);
      for (const outcome of r.outcomes) {
        const original = source.outcomes.find(o => o.outcome_id === outcome.party);
        assert.equal(outcome.winProbability, original.p_win);
        assert.deepEqual(outcome.members.map(m => m.id), original.member_candidate_ids);
        outcome.members.forEach(m => { assert.ok(!("winProbability" in m)); assert.ok(!("share" in m)); });
        assert.ok(!("share" in outcome));
      }
      if (!q) return assert.equal(r.pair, null);
      const first = displayed.candidates.find(c => c.id === q.first_candidate_id), second = displayed.candidates.find(c => c.id === q.second_candidate_id);
      assertPair(r.pair, first, second, q.mean, q.quantiles);
      assert.equal(r.pair.firstExceedsHalf, q.p_first_exceeds_half);
      assert.equal(r.pair.winnerRelationship, "separate_conditional_event");
      assert.equal(r.pair.closeWithin2Pp, q.p_margin_within_2pp);
      assert.equal(r.pair.firstAtLeast10Pp, q.p_first_margin_at_least_10pp);
      assert.equal(r.pair.secondAtLeast10Pp, q.p_second_margin_at_least_10pp);
      assert.deepEqual(r.pair.qualifications, q.qualifications);
    });
    native.qualification.limitations.forEach(note => assert.ok(result.qualification.includes(note)));
    native.provenance.forEach(s => { assert.ok(result.sourceUrls.includes(s.url)); assert.equal(result.sourceHashes[s.id], s.sha256); });
    assert.ok(result.qualification.some(q => q.includes("listed modeled outcomes")));
    assert.equal(JSON.stringify(view), before);
    result.races[0].outcomes[0].members[0].name = "CHANGED OUTPUT ONLY";
    result.races[0].pair.qualifications.push("CHANGED OUTPUT ONLY");
    result.chamberScenario.seats.U.pmf[0] = -10;
    assert.equal(JSON.stringify(view), before);
  }
});

test("a replacement House artifact never inherits the old NY-21 input notice", () => {
  const { view } = house();
  view.run.resultSha256 = "a".repeat(64);
  assert.deepEqual(toHouseElectionNightForecast(view).inputNotices, []);
  const { bundle, entry } = senate();
  assert.deepEqual(toSenateElectionNightForecast(bundle, entry).inputNotices, []);
});

test("Senate outcome-ID and House declared D/R/O modal ties preserve groups without treating near ties as exact", () => {
  const { bundle, entry } = senate();
  const race = bundle.races[0]; race.outcomes.forEach(o => o.winProbability = 0.5);
  race.outcomes.reverse();
  let result = toSenateElectionNightForecast(bundle, entry);
  assert.equal(result.summary.topTieRaceCount, 1);
  assert.deepEqual(result.races[0].modalPick.tiedOutcomeIds, race.outcomes.map(o => o.id).sort());
  assert.equal(result.races[0].modalPick.outcomeId, race.outcomes.map(o => o.id).sort()[0]);
  race.outcomes[0].winProbability = 0.5000000001; race.outcomes[1].winProbability = 0.4999999999;
  result = toSenateElectionNightForecast(bundle, entry);
  assert.equal(result.summary.topTieRaceCount, 0);
  assert.deepEqual(result.races[0].modalPick.tiedOutcomeIds, [race.outcomes[0].id]);
  const { view } = house(); const alaska = view.races.find(r => r.seatId === "AK-00");
  alaska.outcomes.forEach(o => o.winProbability = o.party === "D" ? 0 : 0.5);
  const projected = toHouseElectionNightForecast(view).races.find(r => r.seatId === "AK-00");
  assert.equal(projected.modalPick.kind, "candidate"); assert.equal(projected.modalPick.outcomeId, `${alaska.id}:R`);
  assert.deepEqual(projected.modalPick.tiedOutcomeIds, [`${alaska.id}:R`, `${alaska.id}:O`]);
  assert.equal(projected.outcomes.find(o => o.party === "O").members.length, 2);
  assert.notEqual(projected.pair.firstExceedsHalf, projected.modalPick.probability);
});

test("Senate rejects historical, cross-bound metadata, malformed support and share identities", () => {
  const cases = [
    ["historical", (b, e) => { b.run.historical = e.historical = true; }],
    ["year", (b, e) => { b.run.year = e.year = 2022; }],
    ["entry mismatch", (_b, e) => e.label = "wrong entry"],
    ["target", (b, e) => b.run.targetId = e.targetId = "another target"],
    ["law", (b, e) => b.run.lawId = e.lawId = "another law"],
    ["path", (_b, e) => e.href = "/data/explorer/history-2022-FP-1.json"],
    ["hash", (_b, e) => e.sha256 = "bad"],
    ["observation", b => b.races[0].observed = { outcomeId: b.races[0].outcomes[0].id }],
    ["race ID", b => b.races[0].id = "2026:TX:house:regular"],
    ["seat state", b => b.races[0].seatId = "US-SEN-TX-CLASS-2"],
    ["duplicate race", b => b.races[1] = structuredClone(b.races[0])],
    ["probability mass", b => b.races[0].outcomes[0].winProbability = 1],
    ["negative probability", b => b.races[0].outcomes[0].winProbability = -0.1],
    ["member identity", b => b.races[0].outcomes[0].members[0].id = "not this candidate"],
    ["denominator", b => b.races[0].outcomes[0].share.denominator = "all votes"],
    ["quantiles", b => b.races[0].outcomes[0].share.quantiles.p05 = 0.9],
    ["source cutoff", (b, e) => { b.run.inputCutoffs = e.inputCutoffs = [{ label: "future", at: "2027-01-01T00:00:00Z" }]; }],
    ["default scenario", b => b.defaultScenarioId = "invented"],
    ["source identity", b => b.provenance.sources = {}],
  ];
  for (const [name, change] of cases) {
    const { bundle, entry } = senate(); change(bundle, entry);
    assert.throws(() => toSenateElectionNightForecast(bundle, entry), /Election-night forecast:/, name);
  }
});

test("House rejects wrong editions, districts, joint order, candidate allocation and unbound pair support", () => {
  const cases = [
    ["historical", v => v.run.historical = true], ["year", v => v.run.cycle = 2024],
    ["run ID", v => v.run.id = "house-current-unknown"], ["configuration", v => v.run.configuration.base = "senate_relative_lean_v1"],
    ["unreviewed", v => v.qualification.publicationStatus = "not_reviewed"],
    ["map", v => v.mapVintage = "house_election_map_2024"], ["hash", v => v.run.resultSha256 = "bad"],
    ["district", v => v.races[0].district.cycle = 2024],
    ["joint order", v => v.jointDraws.orderedRaceIds.reverse()],
    ["winner member", v => v.races[0].outcomes[0].members[0].name = "somebody else"],
    ["candidate label", v => v.races[0].outcomes[0].name = "somebody else"],
    ["pooled candidate claim", v => v.races[0].outcomes.find(o => o.kind === "group").kind = "candidate"],
    ["group share", v => v.races[0].outcomes[0].share = { mean: 0.5 }],
    ["pair identity", v => v.races[0].conditionalPair.firstCandidateId = "outside listed support"],
    ["implicit unavailable pair", v => delete v.races[0].conditionalPair],
    ["pair event", v => v.races[0].conditionalPair.marginWithinTwoPoints = 1.2],
    ["incompatible pair bands", v => v.races[0].conditionalPair.firstMarginAtLeastTenPoints = 1],
    ["observation", v => v.races[0].observed = {}],
    ["chamber", v => v.chamber.id = "independent_opposition_caucuses_D"],
  ];
  for (const [name, change] of cases) {
    const { view } = house(); change(view);
    assert.throws(() => toHouseElectionNightForecast(view), /Election-night forecast:/, name);
  }
});

test("unavailable qualified House chamber remains null and input issue/cutoff dates are never replaced", () => {
  const { view } = house(); view.chamber = null; view.chamberPrecision = null;
  const result = toHouseElectionNightForecast(freeze(view));
  assert.equal(result.chamberScenario, null);
  assert.equal(result.run.cutoffAt, "2026-10-25T15:48:30.658333+00:00");
  assert.equal(result.run.issuedAt, "2026-10-25T21:58:38.251767+00:00");
});
