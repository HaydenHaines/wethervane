import assert from "node:assert/strict";
import test from "node:test";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import {
  makeHouseLiveCatalog,
  makeHouseLiveResult,
  SYNTHETIC_CONFIGURATIONS,
  syntheticCurrentDistrictCount,
} from "./fixtures/house-live-fixtures.mjs";

test("synthetic live result covers the 435 current districts and keeps outcomes distinct from conditional shares", () => {
  const value = makeHouseLiveResult();
  const parsed = parseHouseLiveResult(value);
  assert.ok(parsed, "the invented current-contract fixture should parse");
  assert.equal(value.cycle, 2026);
  assert.equal(value.races.length, 435);
  assert.equal(syntheticCurrentDistrictCount, 435);
  assert.equal(new Set(value.races.map((race) => race.race_id)).size, 435);
  assert.equal(new Set(value.races.map((race) => race.district_key)).size, 435);
  assert.ok(value.races.every((race) => race.map_vintage === value.map_vintage));
  assert.ok(value.races.every((race) => race.district_key.startsWith(`${race.state}-`)));
  assert.ok(value.races.every((race) => !("observed" in race)));

  const grouped = value.races.find((race) => race.outcomes.some((outcome) => outcome.kind === "group"));
  assert.ok(grouped);
  assert.equal(grouped.outcomes.find((outcome) => outcome.kind === "group").member_candidate_ids.length, 2);
  const independent = value.races.find((race) => race.outcomes.some((outcome) => outcome.party === "O"));
  assert.ok(independent);
  assert.equal(independent.outcomes.find((outcome) => outcome.party === "O").p_win, 0.5);
  const unopposed = value.races.find((race) => race.outcomes.length === 1);
  assert.equal(unopposed.outcomes[0].p_win, 1);

  const pair = value.races.find((race) => race.conditional_shares !== null);
  assert.ok(pair);
  assert.equal(pair.outcomes.find((outcome) => outcome.outcome_id === "D").p_win, 0.5);
  assert.equal(pair.conditional_shares.p_first_exceeds_half, 0.45);
  assert.notEqual(pair.conditional_shares.p_first_exceeds_half, pair.outcomes.find((outcome) => outcome.outcome_id === "D").p_win);
  assert.equal(value.chamber.majority.D, 0.5);
  assert.equal(value.chamber.majority.R, 0.5);
  assert.equal(value.tipping, null);
});

test("optional source capture metadata is validated without requiring it in older fixtures", () => {
  const legacy = makeHouseLiveResult();
  assert.ok(parseHouseLiveResult(legacy));
  const captured = makeHouseLiveResult();
  captured.provenance[0].capture_kind = "rendered_primary_source";
  captured.provenance[0].limitations = ["Invented captured text, not original PDF bytes."];
  assert.ok(parseHouseLiveResult(captured));
  for (const change of [
    source => { source.capture_kind = "unreviewed_capture"; },
    source => { source.limitations = ["Invalid\ncontrol character"]; },
    source => { source.limitations = "not an array"; },
  ]) {
    const invalid = structuredClone(captured);
    change(invalid.provenance[0]);
    assert.equal(parseHouseLiveResult(invalid), null);
  }
});

test("live catalog contains exactly the 16 explicitly allowed saved configurations", () => {
  const catalog = makeHouseLiveCatalog();
  const parsed = parseHouseLiveCatalog(catalog);
  assert.ok(parsed);
  assert.equal(parsed.cycle, 2026);
  assert.equal(parsed.entries.length, 16);
  assert.equal(parsed.default_id, "house-current-polling");
  assert.deepEqual(
    parsed.entries.map((entry) => entry.configuration),
    SYNTHETIC_CONFIGURATIONS,
  );
  assert.equal(new Set(parsed.entries.map((entry) => entry.id)).size, 16);

  for (const mutate of [
    (value) => { value.entries.pop(); },
    (value) => { value.entries[1].configuration.uncertainty = "cycle-high"; },
    (value) => { value.entries[0].configuration.polling = "standard"; },
    (value) => { value.entries[0].id = value.entries[1].id; },
    (value) => { value.default_id = "missing-default"; },
    (value) => { value.entries[0].result.sha256 = "bad"; },
  ]) {
    const invalid = structuredClone(catalog);
    mutate(invalid);
    assert.equal(parseHouseLiveCatalog(invalid), null);
  }
});

test("result must have the complete current roster, matching map identity, and no transplanted TX-23 incumbent", () => {
  for (const mutate of [
    (value) => { value.races.pop(); },
    (value) => { value.races[0].district_key = value.races[1].district_key; },
    (value) => { value.races[0].race_id = value.races[1].race_id; },
    (value) => { value.races[0].map_vintage = "house_election_map_2022"; },
    (value) => { value.geography.feature_count = 434; },
    (value) => { value.races.find((race) => race.district_key === "TX-23").candidates[0].name = "Tony Gonzales"; },
    (value) => { value.races.find((race) => race.district_key === "TX-23").candidates[0].candidate_id = "gonzales-tx23-2024"; },
  ]) {
    const invalid = makeHouseLiveResult();
    mutate(invalid);
    assert.equal(parseHouseLiveResult(invalid), null);
  }
});

test("candidate membership, outcome support, and group-vs-candidate kinds are checked", () => {
  const groupedIndex = 0;
  for (const mutate of [
    (value) => { value.races[groupedIndex].outcomes[0].member_candidate_ids[0] = "not-on-roster"; },
    (value) => { value.races[groupedIndex].outcomes[0].member_candidate_ids.push(value.races[groupedIndex].outcomes[0].member_candidate_ids[0]); },
    (value) => { value.races[groupedIndex].outcomes[0].kind = "candidate"; },
    (value) => { value.races[groupedIndex].outcomes[0].party = "R"; },
    (value) => { value.races[groupedIndex].outcomes[1].member_candidate_ids = value.races[groupedIndex].outcomes[0].member_candidate_ids; },
    (value) => { value.races[groupedIndex].outcomes[0].p_win = 0.6; },
  ]) {
    const invalid = makeHouseLiveResult();
    mutate(invalid);
    assert.equal(parseHouseLiveResult(invalid), null);
  }
});

test("nonfinite probabilities, non-simplex outcomes, unordered shares, and late cutoffs fail closed", () => {
  for (const mutate of [
    (value) => { value.races[3].outcomes[0].p_win = Number.NaN; },
    (value) => { value.races[3].outcomes[0].p_win = 0.8; },
    (value) => { value.races[3].conditional_shares.mean = Number.POSITIVE_INFINITY; },
    (value) => { value.races[3].conditional_shares.quantiles.p10 = 0.8; },
    (value) => { value.races[3].conditional_shares.p_margin_within_2pp = 1.1; },
    (value) => { value.source_cutoffs[0].at = "2026-10-21T00:00:00.000Z"; },
    (value) => { value.races[3].poll_count = -1; },
  ]) {
    const invalid = makeHouseLiveResult();
    mutate(invalid);
    assert.equal(parseHouseLiveResult(invalid), null);
  }
});

test("an election-day forecast cutoff is valid when issue time is later and every source is earlier", () => {
  const value = makeHouseLiveResult();
  value.forecast_cutoff = "2026-11-03T00:00:00.000Z";
  value.issued_at = "2026-11-03T12:00:00.000Z";
  value.source_cutoffs = [
    { role: "structural", at: "2026-11-02T12:00:00.000Z" },
    { role: "poll", at: "2026-11-02T23:59:00.000Z" },
  ];
  value.provenance[0].retrieved_at = "2026-11-02T23:59:00.000Z";
  assert.ok(parseHouseLiveResult(value));
});

test("impossible calendar dates are rejected in issue, forecast-cutoff, and source-cutoff timestamps", () => {
  for (const mutate of [
    (value) => { value.issued_at = "2026-02-30T12:00:00Z"; },
    (value) => { value.forecast_cutoff = "2026-02-30T12:00:00Z"; },
    (value) => { value.source_cutoffs[0].at = "2026-02-30T12:00:00Z"; },
  ]) {
    const invalid = makeHouseLiveResult();
    mutate(invalid);
    assert.ok(!parseHouseLiveResult(invalid), "an impossible date must not normalize into another day");
  }
});

test("poll count must match the number of distinct poll source IDs", () => {
  const invalid = makeHouseLiveResult();
  invalid.races[3].poll_source_ids = ["synthetic-poll-1"];
  invalid.races[3].poll_count = 0;
  assert.ok(!parseHouseLiveResult(invalid));

  const duplicate = makeHouseLiveResult();
  duplicate.races[3].poll_source_ids = ["synthetic-poll-1", "synthetic-poll-1"];
  duplicate.races[3].poll_count = 2;
  assert.ok(!parseHouseLiveResult(duplicate));
});

test("a current forecast cannot claim a known 2026 observed result", () => {
  const invalid = makeHouseLiveResult();
  invalid.races.find((race) => race.district_key === "TX-23").observed = {
    outcome_id: "R", winner_name: "Invented observed winner",
  };
  assert.ok(!parseHouseLiveResult(invalid), "the parser must reject any observed 2026 outcome");
});

test("joint-draw roster order and seat PMF summaries must remain internally coherent", () => {
  for (const mutate of [
    (value) => { value.joint_draws.ordered_race_ids[0] = value.joint_draws.ordered_race_ids[1]; },
    (value) => { value.joint_draws.ordered_race_ids.pop(); },
    (value) => { value.chamber.seats.D.expected += 1; },
    (value) => { value.chamber.seats.R.pmf[434] -= 0.25; value.chamber.seats.R.pmf[433] += 0.25; },
    (value) => { value.chamber.majority.D = 0.25; value.chamber.majority.R = 0.75; },
    (value) => { value.chamber.joint_draws_sha256 = "f".repeat(64); },
  ]) {
    const invalid = makeHouseLiveResult();
    mutate(invalid);
    assert.equal(parseHouseLiveResult(invalid), null);
  }
});

test("unqualified chamber and tipping statistics stay absent rather than being inferred", () => {
  const value = makeHouseLiveResult();
  value.chamber = null;
  value.tipping = null;
  value.qualification.limitations.push("Chamber calculations are not qualified for this invented result.");
  const parsed = parseHouseLiveResult(value);
  assert.ok(parsed);
  assert.equal(parsed.chamber, null);
  assert.equal(parsed.tipping, null);

  const invalid = makeHouseLiveResult();
  delete invalid.tipping;
  assert.equal(parseHouseLiveResult(invalid), null);

  const unexplained = makeHouseLiveResult();
  unexplained.chamber = null;
  assert.equal(parseHouseLiveResult(unexplained), null);
});
