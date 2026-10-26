import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { matchElectionNightReference } from "../lib/election-night-reference.ts";
import { toSenateElectionNightForecast, toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { houseLiveView } from "../lib/house-live-view.ts";

const read = path => JSON.parse(readFileSync(path, "utf8"));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const registry = read("public/data/explorer/registry.json");
const publication = read("data/election-night-reference-publication.json");
const catalogBytes = readFileSync("public/data/election-night/references.json");
const originalCatalog = JSON.parse(catalogBytes.toString("utf8"));
function fixture() {
  const entry = registry.runs.find(r => r.id === "current-polling");
  return { forecast: toSenateElectionNightForecast(read(`public${entry.href}`), entry), catalog: structuredClone(originalCatalog) };
}
function freeze(value) { if (value && typeof value === "object") { Object.freeze(value); Object.values(value).forEach(freeze); } return value; }
const near = (a, b) => assert.ok(Math.abs(a - b) <= 1e-10, `${a} vs ${b}`);

test("actual publication fingerprint and exact standard forecast expose the reviewed model-implied reference", () => {
  assert.equal(publication.href, "/data/election-night/references.json");
  assert.equal(sha(catalogBytes), publication.sha256);
  const { forecast, catalog } = fixture(), raw = catalog.entries[0].reference;
  const result = matchElectionNightReference(forecast, catalog);
  assert.ok(result); assert.equal(result.runId, "current-polling");
  assert.equal(result.resultSha256, forecast.run.artifactSha256);
  assert.equal(result.sourcePins.referenceSha256, "64df0b35a79e46d7d838b5aba70dbd678d110b19392ce30fa720c0ae3840a983");
  assert.equal(result.sourcePins.reviewSha256, "b5f793d959c484fc977e46f92afc63bc0375b8817708cd771fe9ae637e5f14e0");
  assert.equal(result.sourcePins.reviewManifestSha256, "b4cae1844dce11bc9bee0ed287cc0eb8e461142bce91359408207d3e6e0c6ab9");
  assert.equal(result.sourcePins.previousReferenceSha256, "48d36fdce404a74569553c859a8b8d66bb9790c0eef614950bfa98ca5da058db");
  assert.equal(result.sourcePins.previousReviewSha256, "4ba519f76c03852659c69eec96e84913c0fbcd00c8e8e3d82de6e9e7d845fe05");
  assert.equal(result.sourcePins.forecastSha256, forecast.sourceHashes.forecastSha256);
  assert.equal(result.sourcePins.jointDrawsSha256, forecast.sourceHashes.jointDrawsSha256);
  assert.equal(result.forecastIssuedAt, forecast.run.issuedAt); assert.equal(result.forecastCutoffAt, forecast.run.cutoffAt);
  assert.equal(result.structuralCutoffAt, "2026-10-20T15:21:23Z");
  assert.deepEqual(result.drawIdentity.shape, [4, 10000]); assert.equal(result.drawIdentity.rowCount, 40000);
  assert.equal(result.drawIdentity.weightPerRow, 1 / 40000); assert.equal(result.drawIdentity.sharedAcrossAllRaces, true);
  assert.equal(result.drawIdentity.resamplingOrIndependenceReconstruction, false);
  assert.equal(result.perRace.length, 35);
  for (let i = 0; i < 35; i++) {
    const pair = forecast.races[i].pair, row = result.perRace[i], native = raw.cohort[i];
    assert.equal(row.raceId, forecast.races[i].id);
    assert.equal(row.firstCandidateId, pair.firstCandidateId); assert.equal(row.secondCandidateId, pair.secondCandidateId);
    assert.equal(row.denominator, pair.denominator); assert.equal(row.nativeDenominator, native.denominator);
    near(row.meanMarginPp, pair.meanMarginPp); near(row.range90.lower, pair.range90.lower); near(row.range90.upper, pair.range90.upper);
    assert.equal(row.expectedAbsoluteMarginErrorPp, native.expected_absolute_margin_error_pp);
    assert.equal(row.modalOutcomeId, forecast.races[i].modalPick.outcomeId);
    assert.equal(row.marginEventDrawDenominator, 40000);
    assert.deepEqual(row.marginEventDrawCounts, native.margin_event_draw_counts);
    near(row.closeWithin2Pp, native.close_margin_probability_2pp);
    near(row.firstAtLeast10Pp, native.first_decisive_margin_probability_10pp);
    near(row.secondAtLeast10Pp, native.second_decisive_margin_probability_10pp);
    assert.equal(forecast.races[i].pair.closeWithin2Pp, null, "The dated public bundle has no tail probability field; the V2 reference supplies it");
  }
  // Independently reviewed native numerical anchors, not freshly calculated from displayed summaries.
  near(result.perRace[0].expectedAbsoluteMarginErrorPp, 4.729417948029316);
  near(result.summary.meanAbsoluteMarginError.mean, 5.621801760616599);
  near(result.summary.meanAbsoluteMarginError.range90.lower, 4.014618569894771);
  near(result.summary.meanAbsoluteMarginError.range90.upper, 7.929702663015093);
  assert.deepEqual(result.summary.modalWinnerMissCount.range90, { lower: 1, upper: 7 });
  near(result.summary.modalWinnerMissCount.mean, 3.83785);
  assert.deepEqual(result.summary.interval90CoverageCount.range90, { lower: 26, upper: 35 });
  near(result.summary.meanSignedMarginError.range90.lower, -5.5424942199247775);
  near(result.summary.meanSignedMarginError.range90.upper, 5.775931772163015);
  near(result.summary.meanSignedMarginError.mean, 0);
  assert.equal(result.evaluationLock, false); assert.equal(result.observedResultsConsumed, false);
  assert.deepEqual(result.qualifications, raw.qualifications);
  assert.match(result.interpretation, /not observed errors, empirical accuracy, calibration validation/);
  assert.ok(!("grade" in result)); assert.ok(!("actualError" in result));
});

const ownerRoot = process.env.BALLOTMANCY_RESEARCH_ROOT || "/home/hayden/projects/wethervane-redesign-20261006/integration";
const nativePath = join(ownerRoot, "research/senate-polling-refresh-20261026/standard-scorecard-reference-v2/reference.json");
const reviewPath = join(ownerRoot, "research/senate-polling-refresh-20261026/scorecard-tail-review-sol/review-v1.json");
const previousPath = join(ownerRoot, "research/senate-polling-refresh-20261026/standard-scorecard-reference-v1/reference.json");
const previousReviewPath = join(ownerRoot, "research/senate-polling-refresh-20261026/scorecard-review-sol/review-v1.json");
test("available actual owner V2 reference, tail review and preserved V1 diagnostics match the shipped catalog", { skip: !existsSync(nativePath) || !existsSync(reviewPath) || !existsSync(previousPath) || !existsSync(previousReviewPath) }, () => {
  assert.equal(sha(readFileSync(nativePath)), originalCatalog.entries[0].sourcePins.referenceSha256);
  assert.equal(sha(readFileSync(reviewPath)), originalCatalog.entries[0].sourcePins.reviewSha256);
  assert.equal(sha(readFileSync(previousPath)), originalCatalog.entries[0].sourcePins.previousReferenceSha256);
  assert.equal(sha(readFileSync(previousReviewPath)), originalCatalog.entries[0].sourcePins.previousReviewSha256);
  const native = read(nativePath), previous = read(previousPath), publicRef = originalCatalog.entries[0].reference;
  assert.deepEqual(publicRef.cohort, native.cohort);
  assert.deepEqual(publicRef.drawIdentity, native.draw_identity);
  assert.deepEqual(publicRef.metricDefinitions, native.metric_definitions);
  assert.deepEqual(publicRef.distributions, native.distributions);
  assert.deepEqual(publicRef.qualifications, native.qualifications);
  assert.equal(publicRef.cohortSha256, native.cohort_sha256);
  assert.deepEqual(publicRef.raceMarginProbabilityDefinitions, native.race_margin_probability_definitions);
  assert.deepEqual(native.metric_definitions, previous.metric_definitions);
  assert.deepEqual(native.distributions, previous.distributions);
  for (let i = 0; i < 35; i++) {
    for (const key of Object.keys(previous.cohort[i])) assert.deepEqual(native.cohort[i][key], previous.cohort[i][key]);
    for (const key of ["margin_event_draw_counts", "margin_event_draw_denominator", "close_margin_probability_2pp",
      "first_decisive_margin_probability_10pp", "second_decisive_margin_probability_10pp"])
      assert.deepEqual(publicRef.cohort[i][key], native.cohort[i][key]);
  }
});

test("only one of all16 actual current Senate models matches; House and historical/locked editions never borrow it", () => {
  let matches = 0;
  for (const entry of registry.runs.filter(r => r.year === 2026)) {
    const forecast = toSenateElectionNightForecast(read(`public${entry.href}`), entry);
    const result = matchElectionNightReference(forecast, originalCatalog);
    if (entry.id === "current-polling") { assert.ok(result); matches++; } else assert.equal(result, null, entry.id);
  }
  assert.equal(matches, 1);
  const hc = read("data/house-live/catalog.json"), he = hc.entries.find(e => e.id === hc.default_id);
  const house = toHouseElectionNightForecast(houseLiveView(read(`data/house-live/${he.result.path}`), he));
  assert.equal(matchElectionNightReference(house, originalCatalog), null);
  for (const change of [f => f.run.year = 2022, f => f.run.id = "history-2022-FP-1", f => f.editionStatus = "evaluation_locked",
    f => f.run.configuration.uncertainty = "error-high"]) {
    const { forecast, catalog } = fixture(); change(forecast); assert.equal(matchElectionNightReference(forecast, catalog), null);
  }
});

test("stale public or native hashes, dates, sources and lock/outcome claims reject exact-reference matching", () => {
  const changes = [
    (f, c) => c.entries[0].resultSha256 = "0".repeat(64),
    (f, c) => c.entries[0].runId = "current-error-high",
    (f, c) => c.entries[0].targetId = "other-target",
    (f, c) => c.entries[0].lawId = "other-law",
    (f, c) => c.entries[0].forecastIssuedAt = "2026-10-26T16:00:00Z",
    (f, c) => c.entries[0].forecastCutoffAt = "2026-10-20T16:21:44Z",
    (f, c) => c.entries[0].structuralCutoffAt = "2026-10-26T15:03:57.212405+00:00",
    (f, c) => c.entries[0].referenceIssuedAt = "2026-10-27T00:00:00Z",
    (f, c) => c.entries[0].evaluationLock = true,
    (f, c) => c.entries[0].observedResultsConsumed = true,
    (f, c) => c.entries[0].reference.cohortSha256 = "0".repeat(64),
    f => f.sourceHashes.forecastSha256 = "0".repeat(64),
    f => f.sourceHashes.jointDrawsSha256 = "0".repeat(64),
    f => f.run.sourceCutoffs.reverse().pop(),
  ];
  for (const key of Object.keys(originalCatalog.entries[0].sourcePins)) changes.push((_f, c) => c.entries[0].sourcePins[key] = "0".repeat(64));
  for (const change of changes) {
    const { forecast, catalog } = fixture(); change(forecast, catalog);
    assert.equal(matchElectionNightReference(forecast, catalog), null, String(change));
  }
});

test("ordered cohort, named candidates, mean/interval bounds, group support and conditional denominator cannot be substituted", () => {
  const changes = [
    (f, c) => c.entries[0].reference.cohort.reverse(), f => f.races.reverse(),
    (f, c) => c.entries[0].reference.cohort[1] = structuredClone(c.entries[0].reference.cohort[0]),
    (f, c) => c.entries[0].reference.cohort[0].included_in_native_diagnostic_cohort = false,
    (f, c) => c.entries[0].reference.cohort[0].first_candidate_id = "another-person",
    (f, c) => c.entries[0].reference.cohort[0].second_candidate_id = "another-person",
    (f, c) => c.entries[0].reference.cohort[0].target_id = "different-target",
    (f, c) => c.entries[0].reference.cohort[0].denominator = "all valid ballots",
    (f, c) => c.entries[0].reference.cohort[0].margin_sign = "second_candidate_minus_first_candidate",
    (f, c) => c.entries[0].reference.cohort[0].point_margin_mean_pp += 0.01,
    (f, c) => c.entries[0].reference.cohort[0].interval_90_lower_margin_pp += 0.01,
    (f, c) => c.entries[0].reference.cohort[0].interval_90_upper_margin_pp += 0.01,
    (f, c) => c.entries[0].reference.cohort[0].share_interpretation = "opposition_proxy",
    (f, c) => c.entries[0].reference.cohort[0].rule_proxy = "exact all-ballot RCV result",
    f => f.races[0].pair.denominator = "all valid ballots",
    f => f.races[0].pair.winnerRelationship = "separate_conditional_event",
    f => f.races[0].outcomes[0].kind = "group",
    f => f.races[0].outcomes.reverse(),
    f => f.races[0].outcomes[0].members.push(structuredClone(f.races[0].outcomes[1].members[0])),
    f => f.races[0].pair = null,
    f => f.races.pop(),
    f => f.summary.pairRaceCount = 34,
  ];
  for (const change of changes) {
    const { forecast, catalog } = fixture(); change(forecast, catalog);
    assert.equal(matchElectionNightReference(forecast, catalog), null, String(change));
  }
});

test("native first-candidate half-tie rule is accepted only when the guide's actual lexical modal choice agrees", () => {
  function tieFixture(index) {
    const f = fixture(), row = f.catalog.entries[0].reference.cohort[index], race = f.forecast.races[index];
    row.first_pair_win_probability = 0.5; row.forecast_probability_exact_half = true; row.modal_pair_winner_id = row.first_candidate_id;
    race.outcomes.forEach(o => o.winProbability = 0.5); race.pair.firstExceedsHalf = 0.5;
    const ties = race.outcomes.map(o => o.id).sort(), chosen = race.outcomes.find(o => o.id === ties[0]);
    race.modalPick = { outcomeId: chosen.id, name: chosen.name, kind: "candidate", probability: 0.5, tiedOutcomeIds: ties };
    const misses = f.forecast.races.reduce((sum, r) => sum + 1 - Math.max(...r.outcomes.map(o => o.winProbability)), 0);
    f.forecast.summary.expectedModalMisses = misses; f.forecast.summary.topTieRaceCount = 1;
    f.catalog.entries[0].reference.distributions.modal_pair_winner_miss_count.mean = misses;
    return f;
  }
  // Hypothetical software fixtures only; actual reviewed native data has zero exact-half ties.
  const rows = originalCatalog.entries[0].reference.cohort;
  assert.equal(rows.filter(r => r.forecast_probability_exact_half).length, 0);
  const agrees = rows.findIndex(r => r.first_candidate_id < r.second_candidate_id);
  const disagrees = rows.findIndex(r => r.first_candidate_id > r.second_candidate_id);
  assert.ok(agrees >= 0 && disagrees >= 0);
  const yes = tieFixture(agrees), no = tieFixture(disagrees);
  assert.ok(matchElectionNightReference(yes.forecast, yes.catalog));
  assert.equal(matchElectionNightReference(no.forecast, no.catalog), null);
  const bad = fixture(); bad.catalog.entries[0].reference.cohort[0].modal_pair_winner_id = "wrong-modal-id";
  assert.equal(matchElectionNightReference(bad.forecast, bad.catalog), null);
});

test("draw identity, metric definitions, numerical support and cross-summary identities fail closed", () => {
  const changes = [
    r => r.drawIdentity.shape = [8, 5000], r => r.drawIdentity.weight_per_row = 1 / 80000,
    r => r.drawIdentity.order = "sorted marginal draws", r => r.drawIdentity.shared_across_all_races = false,
    r => r.drawIdentity.resampling_or_independence_reconstruction = true,
    r => r.metricDefinitions.mean_signed_margin_error_pp = "predicted minus observed",
    r => r.distributions.mean_absolute_margin_error_pp.mean += 0.1,
    r => r.cohort[0].expected_absolute_margin_error_pp += 1,
    r => r.distributions.modal_pair_winner_miss_count.mean += 0.1,
    r => r.distributions.mean_signed_margin_error_pp.mean = 1,
    r => r.distributions.mean_absolute_margin_error_pp.standard_deviation = -1,
    r => r.distributions.mean_absolute_margin_error_pp.quantiles.p05 = 100,
    r => r.distributions.interval_90_coverage_count.quantiles.p99 = 36,
    r => r.distributions.modal_pair_winner_miss_count.quantiles.p01 = -1,
    r => r.distributions.mean_signed_margin_error_pp.mean = NaN,
    r => r.cohort[0].expected_absolute_margin_error_pp = Infinity,
    r => r.qualifications = [],
  ];
  for (const change of changes) {
    const { forecast, catalog } = fixture(); change(catalog.entries[0].reference);
    assert.equal(matchElectionNightReference(forecast, catalog), null, String(change));
  }
  for (const value of [null, {}, [], { schemaVersion: 2, entries: [] }, { schemaVersion: 1, entries: [] }]) {
    assert.equal(matchElectionNightReference(fixture().forecast, value), null);
  }
  const f = fixture(); f.catalog.entries.push(structuredClone(f.catalog.entries[0]));
  assert.equal(matchElectionNightReference(f.forecast, f.catalog), null);
});

test("V2 native event rows match count fractions and authenticated catalog bytes remain a separate boundary", () => {
  const { forecast, catalog } = fixture();
  const result = matchElectionNightReference(forecast, catalog); assert.ok(result);
  const first = result.perRace[0], raw = catalog.entries[0].reference.cohort[0];
  for (const key of ["closeWithin2Pp", "firstAtLeast10Pp", "secondAtLeast10Pp"])
    assert.equal(first[key], raw[{ closeWithin2Pp: "close_margin_probability_2pp", firstAtLeast10Pp: "first_decisive_margin_probability_10pp",
      secondAtLeast10Pp: "second_decisive_margin_probability_10pp" }[key]]);
  assert.equal(first.marginEventDrawDenominator, 40000);
  assert.equal(first.marginEventDrawCounts.close_margin_probability_2pp / 40000, first.closeWithin2Pp);
  // This changed catalog has no matching publication fingerprint even if a
  // caller's semantic matcher accepts extra fields; loaders authenticate bytes.
  catalog.entries[0].reference.cohort[0].unreviewedExtra = 0.88;
  assert.notEqual(sha(Buffer.from(JSON.stringify(catalog))), publication.sha256);
});

test("tail counts, probabilities, candidate ordering and foreign editions fail closed", () => {
  const changes = [
    (_f, c) => c.entries[0].reference.cohort[0].close_margin_probability_2pp += 0.01,
    (_f, c) => c.entries[0].reference.cohort[0].margin_event_draw_counts.close_margin_probability_2pp += 1,
    (_f, c) => c.entries[0].reference.cohort[0].margin_event_draw_denominator = 39999,
    (_f, c) => c.entries[0].reference.cohort[0].first_decisive_margin_probability_10pp = 0.99,
    (_f, c) => c.entries[0].reference.cohort[0].second_decisive_margin_probability_10pp = 0.99,
    (_f, c) => c.entries[0].reference.cohort[0].margin_event_draw_counts.close_margin_probability_2pp = Infinity,
    (_f, c) => c.entries[0].reference.raceMarginProbabilityDefinitions.partition = "exhaustive",
    (_f, c) => c.entries[0].sourcePins.previousReviewSha256 = "0".repeat(64),
  ];
  for (const change of changes) {
    const { forecast, catalog } = fixture(); change(forecast, catalog);
    assert.equal(matchElectionNightReference(forecast, catalog), null, String(change));
  }
  const { forecast, catalog } = fixture();
  forecast.races[0].pair.closeWithin2Pp = 0.5;
  assert.equal(matchElectionNightReference(forecast, catalog), null);
  const foreign = fixture(); foreign.forecast.run.id = "current-error-low";
  assert.equal(matchElectionNightReference(foreign.forecast, foreign.catalog), null);
});

test("matcher does not mutate inputs and normalized arrays, sources, ranges and qualifications are detached", () => {
  const f = fixture(), before = JSON.stringify(f); freeze(f);
  const result = matchElectionNightReference(f.forecast, f.catalog); assert.ok(result);
  assert.equal(JSON.stringify(f), before);
  result.sourcePins.reviewSha256 = "changed-output-only"; result.perRace[0].range90.lower = -99;
  result.summary.meanAbsoluteMarginError.quantiles.p05 = -99;
  result.drawIdentity.shape[0] = 99; result.qualifications.push("changed-output-only");
  assert.equal(JSON.stringify(f), before);
  assert.equal(JSON.parse(JSON.stringify(result)).observedResultsConsumed, false);
});
