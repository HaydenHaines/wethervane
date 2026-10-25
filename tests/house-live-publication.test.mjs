import assert from "node:assert/strict";
import test from "node:test";
import { parseHouseLivePublication } from "../lib/house-live-publication.ts";
import { makeHouseLivePublication } from "./fixtures/house-live-publication.mjs";

function publication() {
  return makeHouseLivePublication();
}

function geometry(view) {
  return { metadata: { cycle: 2026, mapVintage: view.mapVintage, admission: "verified",
    sourceSha256: view.geographySha256, sourceName: "Invented test grid; not district boundaries" },
    geojson: { type: "FeatureCollection", features: view.races.map((race, index) => {
      const x = -120 + index % 25, y = 26 + Math.floor(index / 25);
      return { type: "Feature", properties: { districtKey: race.seatId, discardedPrivateProperty: "drop me" },
        geometry: { type: "Polygon", coordinates: [[[x, y], [x + .5, y], [x + .5, y + .5], [x, y + .5], [x, y]]], discardedPrivateGeometryData: "drop too" } };
    }) } };
}

test("accepts a complete 435-district selected result and preserves only the public response shape", () => {
  const input = publication();
  input.view.privatePath = "/srv/forecast/private/result.json";
  input.view.sources[0].path = "private/source-panel.csv";
  input.view.races[0].private_model_draws = ["do not copy"];
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  assert.equal(output.view.races.length, 435);
  assert.equal(output.view.run.id, output.catalog.defaultId);
  assert.equal(output.view.run.resultSha256, output.catalog.entries.find(entry => entry.id === output.view.run.id).resultSha256);
  assert.equal("privatePath" in output.view, false);
  assert.equal("path" in output.view.sources[0], false);
  assert.equal("private_model_draws" in output.view.races[0], false);
  assert.equal(output.view.races[0].outcomes.every(outcome => !("share" in outcome)), true);
  assert.equal("publicationReview" in output, false);
});

test("preserves optional website publication notes separately from native limitations", () => {
  const input = publication();
  const nativeLimitations = [...input.view.qualification.limitations];
  input.publicationReview = { limitations: ["Invented operator review note.", "No observed 2026 result yet."] };
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  assert.deepEqual(output.publicationReview.limitations, input.publicationReview.limitations);
  assert.deepEqual(output.view.qualification.limitations, nativeLimitations);
  output.publicationReview.limitations.push("Display-only mutation.");
  assert.equal(input.publicationReview.limitations.length, 2);
});

test("rejects malformed or embellished website publication notes", () => {
  for (const publicationReview of [
    null, { limitations: "not a list" }, { limitations: [""] },
    { limitations: ["Valid note", 3] }, { limitations: [], privatePath: "/srv/private" },
  ]) {
    const input = publication();
    input.publicationReview = publicationReview;
    assert.equal(parseHouseLivePublication(input), null);
  }
});

test("requires all 16 exact allowed configurations and selected id/config/result digest binding", () => {
  const good = publication();
  const missing = structuredClone(good);
  missing.catalog.entries.pop();
  assert.equal(parseHouseLivePublication(missing), null);

  const duplicate = structuredClone(good);
  duplicate.catalog.entries[1].configuration = structuredClone(duplicate.catalog.entries[0].configuration);
  assert.equal(parseHouseLivePublication(duplicate), null);

  const unsafeId = structuredClone(good);
  unsafeId.catalog.entries[0].id = "bad run id";
  assert.equal(parseHouseLivePublication(unsafeId), null);

  for (const mutate of [
    value => { value.view.run.id = "another-saved-run"; },
    value => { value.view.run.configuration.polling = "none"; },
    value => { value.view.run.resultSha256 = "f".repeat(64); },
  ]) {
    const bad = structuredClone(good);
    mutate(bad);
    assert.equal(parseHouseLivePublication(bad), null);
  }
});

test("rejects wrong cycle, dates, cutoff order, map identity, missing current districts, and observed outcomes", () => {
  const good = publication();
  for (const mutate of [
    value => { value.view.run.cycle = 2024; },
    value => { value.view.run.issuedAt = "2026-02-30T12:00:00Z"; },
    value => { value.view.run.cutoffAt = "2026-10-22T00:00:00Z"; },
    value => { value.view.races[0].district.mapVintage = "other-vintage"; },
    value => { value.view.races.pop(); },
    value => { value.view.races[0].observed = { winnerName: "invented" }; },
    value => { value.view.qualification.forecastSkill = "forecasted_2026"; },
  ]) {
    const bad = structuredClone(good);
    mutate(bad);
    assert.equal(parseHouseLivePublication(bad), null);
  }
});

test("rejects malformed outcome probability simplex and candidate/group membership", () => {
  const good = publication();
  for (const mutate of [
    value => { value.view.races[3].outcomes[0].winProbability = 0.75; },
    value => { value.view.races[0].outcomes[0].members.pop(); },
    value => { value.view.races[0].outcomes[0].members[0].party = "R"; },
    value => { value.view.races[3].outcomes[0].members[0].id = "not-a-candidate"; },
    value => { value.view.races[3].outcomes[0].share = { mean: 0.8 }; },
  ]) {
    const bad = structuredClone(good);
    mutate(bad);
    assert.equal(parseHouseLivePublication(bad), null);
  }
});

test("preserves a singleton group outcome without relabeling it as an individual candidate chance", () => {
  const input = publication();
  const race = input.view.races[2];
  assert.equal(race.outcomes.length, 1);
  assert.equal(race.outcomes[0].kind, "candidate");
  race.outcomes[0].kind = "group";
  race.outcomes[0].name = "Invented at-large Democratic group";
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  const group = output.view.races[2].outcomes[0];
  assert.equal(group.kind, "group");
  assert.equal(group.winProbability, 1);
  assert.equal(group.members.length, 1);
  assert.equal("winProbability" in group.members[0], false);
  assert.equal("share" in group.members[0], false);
});

test("keeps conditional two-candidate shares separate from winner chances and checks named candidates", () => {
  const good = publication();
  const pairRace = good.view.races.find(race => race.conditionalPair);
  assert.ok(pairRace);
  const parsed = parseHouseLivePublication(good);
  assert.ok(parsed);
  const pair = parsed.view.races.find(race => race.conditionalPair).conditionalPair;
  assert.equal(pair.firstCandidateName, pairRace.candidates.find(candidate => candidate.id === pair.firstCandidateId).name);
  assert.equal(pair.secondCandidateName, pairRace.candidates.find(candidate => candidate.id === pair.secondCandidateId).name);
  assert.equal(pair.firstExceedsHalf, 0.45);
  assert.ok(parsed.view.races.find(race => race.conditionalPair).outcomes.every(outcome => !("share" in outcome)));

  const bad = structuredClone(good);
  bad.view.races.find(race => race.conditionalPair).conditionalPair.firstCandidateName = "wrong named candidate";
  assert.equal(parseHouseLivePublication(bad), null);
});

test("accepts unavailable chamber only with null precision and an explanatory limitation; tipping stays null", () => {
  const input = publication();
  input.view.chamber = null;
  input.view.chamberPrecision = null;
  input.view.qualification.limitations.push("Chamber control is unavailable in this invented test result.");
  input.view.tipping = null;
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  assert.equal(output.view.chamber, null);
  assert.equal(output.view.chamberPrecision, null);
  assert.equal(output.view.tipping, null);

  input.view.chamberPrecision = { maxSeatMeanMcse: 0, maxControlMcse: 0, evidenceSha256: "a".repeat(64) };
  assert.equal(parseHouseLivePublication(input), null);

  const partyResult = publication();
  partyResult.view.chamber.seats.O = structuredClone(partyResult.view.chamber.seats.U);
  assert.equal(parseHouseLivePublication(partyResult), null, "the public chamber shape carries O only as unassigned U seats");
});

test("accepts finite fractional seat quantiles from the native linear four-row example", () => {
  const input = publication();
  // Match the fractional values in the native joint-check's linear four-row case.
  input.view.chamber.seats.D.seatQuantiles = { p05: 1, p50: 217.5, p95: 434.85 };
  input.view.chamber.seats.R.seatQuantiles = { p05: 0, p50: 216.5, p95: 433.85 };
  input.view.chamber.seats.U.seatQuantiles = { p05: 0, p50: 0.5, p95: 1 };
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  assert.equal(output.view.chamber.seats.D.seatQuantiles.p50, 217.5);
});

test("source cutoffs and candidate source URLs stay bound to the projected provenance", () => {
  const lateSource = publication();
  lateSource.view.sources[0].retrieved_at = "2026-10-21T00:00:00.000Z";
  assert.equal(parseHouseLivePublication(lateSource), null, "source retrieval cannot extend past the forecast cutoff");

  const unbound = publication();
  unbound.view.races[0].candidates[0].sourceUrls = ["https://example.invalid/unlisted-source"];
  unbound.view.races[0].outcomes[0].members[0].sourceUrls = ["https://example.invalid/unlisted-source"];
  assert.equal(parseHouseLivePublication(unbound), null, "candidate sources must resolve against public provenance");

  const empty = publication();
  empty.view.races[0].candidates[0].sourceUrls = [];
  empty.view.races[0].outcomes[0].members[0].sourceUrls = [];
  assert.equal(parseHouseLivePublication(empty), null, "native candidate source references project to at least one URL");
});

test("joins optional geography to all 435 districts and drops unknown feature fields", () => {
  const input = publication();
  input.geography = geometry(input.view);
  const output = parseHouseLivePublication(input);
  assert.ok(output);
  assert.equal(output.geography.geojson.features.length, 435);
  assert.deepEqual(Object.keys(output.geography.geojson.features[0].properties), ["districtKey"]);
  assert.deepEqual(Object.keys(output.geography.geojson.features[0].geometry), ["type", "coordinates"]);
  assert.equal(parseHouseLivePublication({ ...input, geography: { ...input.geography, metadata: { ...input.geography.metadata, sourceSha256: "0".repeat(64) } } }), null);

  const bad = structuredClone(input);
  bad.geography.geojson.features.pop();
  assert.equal(parseHouseLivePublication(bad), null);
});

test("does not authenticate or imply review from status strings and rejects observation-bearing response branches", () => {
  const input = publication();
  assert.equal(input.view.qualification.forecastSkill, "not_established");
  const parsed = parseHouseLivePublication(input);
  assert.ok(parsed, "not_established is an honest current-skill status");
  assert.equal(parsed.view.qualification.forecastSkill, "not_established");
  input.view.qualification.forecastSkill = "unobserved_2026";
  const unobserved = parseHouseLivePublication(input);
  assert.ok(unobserved);
  assert.equal(unobserved.view.qualification.forecastSkill, "unobserved_2026");
  input.view.qualification.forecastSkill = "accurate_2026";
  assert.equal(parseHouseLivePublication(input), null, "known or asserted current-cycle accuracy is not yet measurable");
  input.view.qualification.forecastSkill = "not_established";
  input.view.qualification.publicationStatus = "not_for_publication";
  assert.equal(parseHouseLivePublication(input), null, "the browser view projection is only for reviewed publication status");
  input.view.qualification.publicationStatus = "reviewed";
  input.view.currentActualResults = { observed: true };
  assert.equal(parseHouseLivePublication(input), null);
});
