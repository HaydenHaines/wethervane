import assert from "node:assert/strict";
import test from "node:test";
import { validateHouseLiveCollection } from "../lib/house-live-collection.ts";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { makeHouseLiveCatalog, makeHouseLiveResult, SYNTHETIC_CONFIGURATIONS, SYNTHETIC_SHA } from "./fixtures/house-live-fixtures.mjs";

function parsedCollection() {
  const catalog = parseHouseLiveCatalog(makeHouseLiveCatalog());
  assert.ok(catalog, "invented catalog fixture should parse");
  const results = SYNTHETIC_CONFIGURATIONS.map((configuration) => {
    const result = parseHouseLiveResult(makeHouseLiveResult(configuration));
    assert.ok(result, `invented ${configuration.base}/${configuration.polling}/${configuration.uncertainty} fixture should parse`);
    return result;
  });
  return { catalog, results };
}

function reparseResults(results) {
  return results.map((result) => {
    const parsed = parseHouseLiveResult(result);
    assert.ok(parsed, "mutated synthetic fixture should still satisfy its individual native parser");
    return parsed;
  });
}

test("all 16 parsed results share the current roster, candidate identity, source pins, and map artifacts", () => {
  const { catalog, results } = parsedCollection();
  assert.equal(results.length, 16);
  const validation = validateHouseLiveCollection(catalog, results);
  assert.deepEqual(validation, { valid: true, issues: [] });
  assert.ok(results.every((result) => result.races.length === 435));
  // Native fixtures are intentionally not publication-reviewed; collection
  // identity checks do not imply publication admission or scientific review.
  assert.ok(results.every((result) => result.qualification.publication_status === "not_for_publication"));
});

test("catalog and result ordering are ignored while each run's own joint race order is preserved", () => {
  const initial = parsedCollection();
  const catalog = parseHouseLiveCatalog({ ...initial.catalog, entries: [...initial.catalog.entries].reverse() });
  assert.ok(catalog);
  const results = initial.results.map((source) => {
    const result = structuredClone(source);
    result.races.reverse();
    result.joint_draws.ordered_race_ids.reverse();
    for (const race of result.races) {
      race.candidates.reverse();
      race.outcomes.reverse();
      for (const outcome of race.outcomes) outcome.member_candidate_ids.reverse();
      race.source_ids.reverse();
    }
    const parsed = parseHouseLiveResult(result);
    assert.ok(parsed, "row order changes preserve the individual parsed result identity");
    return parsed;
  }).reverse();
  const orderedBefore = results.map((result) => [...result.joint_draws.ordered_race_ids]);
  assert.equal(validateHouseLiveCollection(catalog, results).valid, true);
  assert.deepEqual(results.map((result) => result.joint_draws.ordered_race_ids), orderedBefore);
  for (const result of results)
    assert.deepEqual(result.joint_draws.ordered_race_ids, result.races.map((race) => race.race_id));
});

test("model-specific probabilities, pair shares, qualifications, methods, and polling inputs may vary", () => {
  const { catalog, results: original } = parsedCollection();
  const results = structuredClone(original);
  const polling = results.find((result) => result.configuration.polling === "standard");
  assert.ok(polling);
  const pairRace = polling.races.find((race) => race.conditional_shares);
  pairRace.outcomes[0].p_win = 0.6;
  pairRace.outcomes[1].p_win = 0.4;
  pairRace.conditional_shares.mean = 0.49;
  pairRace.qualifications.push("Different run-specific qualification note.");
  pairRace.method = "a distinct invented method description";
  pairRace.poll_count = 1;
  pairRace.poll_source_ids = ["synthetic-poll-specific-to-this-run"];
  const parsed = reparseResults(results);
  assert.deepEqual(validateHouseLiveCollection(catalog, parsed), { valid: true, issues: [] });
});

test("collection rejects missing, extra, or duplicate parsed results", () => {
  const { catalog, results } = parsedCollection();
  assert.equal(validateHouseLiveCollection(catalog, results.slice(1)).valid, false);
  assert.equal(validateHouseLiveCollection(catalog, [...results, results[0]]).valid, false);
  assert.equal(validateHouseLiveCollection(catalog, [...results.slice(1), { ...results[0], id: "uncataloged-live-run" }]).valid, false);
});

test("each result ID and exact model configuration must match its catalog entry", () => {
  const { catalog, results: original } = parsedCollection();
  const wrongConfiguration = structuredClone(original);
  wrongConfiguration[0].configuration = { ...wrongConfiguration[0].configuration, polling: "standard" };
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(wrongConfiguration)).valid, false);

  const wrongId = structuredClone(original);
  wrongId[0].id = "synthetic-unlisted-id";
  const parsed = wrongId.map((result) => parseHouseLiveResult(result));
  assert.ok(parsed.every(Boolean));
  assert.equal(validateHouseLiveCollection(catalog, parsed).valid, false);
});

test("collection rejects cross-run candidate roster drift in every identity field", () => {
  const { catalog, results: original } = parsedCollection();
  const changes = [
    (result) => { result.races[3].candidates[0].name = "A different invented candidate"; result.races[3].outcomes.find((outcome) => outcome.party === "D").label = "A different invented candidate"; },
    (result) => { result.races[3].candidates[0].party_label = "Changed invented party label"; },
    (result) => { result.races[3].candidates[0].incumbent = true; },
    (result) => { result.races[3].candidates[0].ballot_status = "declared_writein"; },
    (result) => { result.races[3].candidates[0].source_ids = ["synthetic-roster-source-v2"]; result.provenance.push({ id: "synthetic-roster-source-v2", label: "Invented alternative source", url: "https://example.invalid/roster-v2", retrieved_at: "2026-10-19T00:00:00.000Z", sha256: SYNTHETIC_SHA, role: "candidate-roster" }); },
  ];
  for (const mutate of changes) {
    const results = structuredClone(original);
    mutate(results[1]);
    assert.equal(validateHouseLiveCollection(catalog, reparseResults(results)).valid, false);
  }

  const changedParty = structuredClone(original);
  const race = changedParty[1].races[1];
  const rCandidate = race.candidates.find((candidate) => candidate.aggregation_party === "R");
  const oCandidate = race.candidates.find((candidate) => candidate.aggregation_party === "O");
  rCandidate.aggregation_party = "O";
  oCandidate.aggregation_party = "R";
  for (const candidate of race.candidates) {
    const outcome = race.outcomes.find((item) => item.member_candidate_ids.includes(candidate.candidate_id));
    outcome.outcome_id = candidate.aggregation_party;
    outcome.party = candidate.aggregation_party;
    outcome.label = candidate.name;
  }
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(changedParty)).valid, false);
});

test("candidate source IDs must remain stable and their referenced provenance pins must identify the same source", () => {
  const { catalog, results: original } = parsedCollection();
  const changedPin = structuredClone(original);
  changedPin[1].provenance[0].sha256 = "b".repeat(64);
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(changedPin)).valid, false);
});

test("collection rejects outcome support, outcome kind, rule, and race identity drift", () => {
  const { catalog, results: original } = parsedCollection();

  const wrongKind = structuredClone(original);
  const singletonRace = wrongKind[1].races[3];
  singletonRace.outcomes[0].kind = "group";
  singletonRace.outcomes[0].label = "Invented Democratic group";
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(wrongKind)).valid, false);

  const wrongRule = structuredClone(original);
  wrongRule[1].races[3].rule_family = "ranked_choice";
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(wrongRule)).valid, false);

  const wrongMembers = structuredClone(original);
  const groupRace = wrongMembers[1].races[0];
  const dGroup = groupRace.outcomes.find((outcome) => outcome.kind === "group");
  const rCandidate = groupRace.candidates.find((candidate) => candidate.aggregation_party === "R");
  rCandidate.aggregation_party = "D";
  dGroup.member_candidate_ids.push(rCandidate.candidate_id);
  groupRace.outcomes = [dGroup];
  dGroup.p_win = 1;
  assert.equal(validateHouseLiveCollection(catalog, reparseResults(wrongMembers)).valid, false);

  const wrongRaceId = structuredClone(original);
  wrongRaceId[1].races[0].race_id = "2026:ZZ-99:house:regular";
  wrongRaceId[1].joint_draws.ordered_race_ids[0] = "2026:ZZ-99:house:regular";
  const runtimeOnly = wrongRaceId.map((result) => parseHouseLiveResult(result) || result);
  assert.equal(validateHouseLiveCollection(catalog, runtimeOnly).valid, false);
});

test("map vintage, geography pin, and source receipt pin must remain identical across runs", () => {
  const { catalog, results: original } = parsedCollection();
  for (const mutate of [
    (result) => { result.races[0].map_vintage = "house_election_map_2024"; },
    (result) => { result.geography.path = "synthetic/alternate-geography.json"; },
    (result) => { result.geography.sha256 = "c".repeat(64); },
    (result) => { result.geography.source_receipt.path = "synthetic/alternate-receipt.json"; },
    (result) => { result.geography.source_receipt.sha256 = "d".repeat(64); },
  ]) {
    const results = structuredClone(original);
    mutate(results[1]);
    const parsed = results.map((result) => parseHouseLiveResult(result) || result);
    assert.equal(validateHouseLiveCollection(catalog, parsed).valid, false);
  }
});
