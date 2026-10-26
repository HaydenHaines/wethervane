import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { makeHouseLiveCatalog, makeHouseLiveResult } from "./fixtures/house-live-fixtures.mjs";
import {
  HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256,
  HOUSE_LIVE_POLL_EXCLUSION_LABELS,
  buildHouseLivePollingCollection, parseHouseLivePollingCollection,
  parseHouseLivePollingRun, projectHouseLivePollingRun,
} from "../lib/house-live-polling.ts";

const sha = value => createHash("sha256").update(`invented polling test ${value}`).digest("hex");
const path = (name, digest) => ({ path: `research/house-live-20261025/${name}`, sha256: digest });
const clone = value => structuredClone(value);

function fixture() {
  const catalog = makeHouseLiveCatalog();
  for (const entry of catalog.entries) entry.result.sha256 = sha(`result:${entry.id}`);
  const policySha = sha("policy"), admissionSha = sha("admission"),
    selectionSha = sha("selection"), ledgerSha = sha("ledger"), sourceSha = sha("poll discovery");
  catalog.source_policy.sha256 = policySha;
  const forecast = "2026-10-20T00:00:00.000Z", polling = "2026-10-19T12:00:00Z";
  const first = makeHouseLiveResult(catalog.entries[1].configuration);
  const pairRace = first.races.find(race => race.conditional_shares);
  const pair = pairRace.conditional_shares;
  const observationId = `nyt:invented:${pairRace.district_key}`;
  const resultFor = entry => {
    const result = makeHouseLiveResult(entry.configuration);
    result.id = entry.id;
    result.source_cutoffs = [{ role: "structural", at: forecast }, { role: "poll", at: polling }];
    result.qualification.evidence.push({ path: "evidence/admission-v1.json", sha256: admissionSha });
    result.provenance.push({ id: "nyt-house-discovery", label: "Invented polling discovery",
      url: "https://example.invalid/invented-discovery", retrieved_at: "2026-10-19T00:00:00Z",
      sha256: sourceSha, role: "poll discovery" });
    if (entry.configuration.polling === "standard") {
      const race = result.races.find(row => row.race_id === pairRace.race_id);
      race.poll_count = 1;
      race.poll_source_ids = [observationId];
    }
    return { result, resultSha256: entry.result.sha256 };
  };
  const policy = { schema: "HouseLiveSourcePolicyV1", forecast_cutoff: forecast,
    polling_cutoff: polling, strict_as_of: true };
  const admission = { schema: "HouseLiveAggregateInputAdmissionV1",
    status: "admitted_for_current_aggregate_comparison", forecast_cutoff: forecast,
    input_pins: [path("source-policy-v1.json", policySha), path("polls-v3/receipt.json", selectionSha),
      path("polls-v3/ledger.json", ledgerSha)] };
  const selection = { schema: "HouseLivePollSelectionV1", forecast_cutoff: polling,
    observations: 1, races: 1, outputs: [path("polls-v3/ledger.json", ledgerSha)],
    input_pins: [path("current_polls.py", HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256)] };
  const selected = { status: "selected", reason: "", poll_id: "invented-poll", question_id: "invented-question",
    district_key: pairRace.district_key, race_id: pairRace.race_id, observation_id: observationId,
    source_id: "nyt-house-discovery", source_sha256: sourceSha, pollster: "Invented Pollster",
    start_date: "2026-10-16", field_end: "2026-10-17", available_date: "2026-10-18T00:00:00Z",
    population: "LV", sample_size_status: "reported", nominal_n: 700,
    source_url: "https://example.invalid/invented-question",
    source_urls: ["https://example.invalid/invented-question"],
    first_candidate_id: pair.first_candidate_id, second_candidate_id: pair.second_candidate_id };
  const excluded = { status: "excluded", reason: "outside_100_day_field_end_window",
    poll_id: "invented-excluded", question_id: "invented-excluded-question",
    district_key: pairRace.district_key, source_id: "nyt-house-discovery", source_sha256: sourceSha,
    pollster: "Earlier Invented Pollster", end_date: "6/1/26" };
  return { input: { catalog, catalogSha256: sha("catalog"),
    results: catalog.entries.map(resultFor), sourcePolicy: policy, sourcePolicySha256: policySha,
    admission, admissionSha256: admissionSha, selection, selectionSha256: selectionSha,
    ledger: [selected, excluded], ledgerSha256: ledgerSha }, pairRace, observationId };
}

test("invented collection binds all 16 runs and keeps polling-off use separate from the selected pool", () => {
  const { input, pairRace, observationId } = fixture();
  const collection = buildHouseLivePollingCollection(input);
  assert.ok(parseHouseLivePollingCollection(collection));
  assert.equal(collection.runs.length, 16);
  assert.equal(collection.selected.length, 1);
  assert.equal(collection.excludedReasonCounts[0].count, 1);
  assert.deepEqual(Object.keys(collection.selected[0]).sort(), ["observationId", "raceId", "pollster",
    "startDate", "fieldEnd", "availableAt", "population", "sampleSize", "sourceId",
    "sourceSha256", "sourceUrls"].sort());
  assert.equal(collection.selected[0].observationId, observationId);
  assert.equal(collection.selected[0].raceId, pairRace.race_id);
  assert.equal(JSON.stringify(collection).includes("first_candidate_id"), false);
  assert.equal(JSON.stringify(collection).includes("publisher_names"), false);
  assert.equal(JSON.stringify(collection).includes("input_pins"), false);

  const on = collection.runs.find(run => run.polling === "standard");
  const off = collection.runs.find(run => run.polling === "none");
  const onView = projectHouseLivePollingRun(collection, on.id, on.resultSha256);
  const offView = projectHouseLivePollingRun(collection, off.id, off.resultSha256);
  assert.ok(parseHouseLivePollingRun(onView));
  assert.ok(parseHouseLivePollingRun(offView));
  assert.equal(onView.modelUsesSelectedQuestions, true);
  assert.equal(onView.usedQuestions.length, 1);
  assert.equal(offView.modelUsesSelectedQuestions, false);
  assert.equal(offView.usedQuestions.length, 0);
  assert.equal(offView.selectedPoolQuestionCount, 1);
  assert.equal(offView.selectedPoolDistrictCount, 1);
  assert.equal(projectHouseLivePollingRun(collection, on.id, off.resultSha256), null);
  const parsed = parseHouseLivePollingCollection(collection);
  parsed.selected[0].pollster = "Changed copy";
  assert.equal(collection.selected[0].pollster, "Invented Pollster");
  const parsedRun = parseHouseLivePollingRun(onView);
  parsedRun.usedQuestions[0].pollster = "Changed copy";
  assert.equal(onView.usedQuestions[0].pollster, "Invented Pollster");
});

test("strict parsers reject unsafe links, raw fields, inconsistent counts and invented reasons", () => {
  const { input } = fixture();
  const collection = buildHouseLivePollingCollection(input);
  for (const mutation of [
    value => { value.selected[0].sourceUrls = ["javascript:alert(1)"]; },
    value => { value.selected[0].publisher_names = ["raw data"]; },
    value => { value.excluded[0].reason = "invented_reason"; },
    value => { value.excludedReasonCounts[0].count = 2; },
    value => { value.runs[0].resultSha256 = value.runs[1].resultSha256; },
  ]) {
    const altered = clone(collection); mutation(altered);
    assert.equal(parseHouseLivePollingCollection(altered), null);
  }
  const on = collection.runs.find(run => run.polling === "standard");
  const off = collection.runs.find(run => run.polling === "none");
  const onView = projectHouseLivePollingRun(collection, on.id, on.resultSha256);
  const offView = projectHouseLivePollingRun(collection, off.id, off.resultSha256);
  const missing = clone(onView); missing.usedQuestions = [];
  assert.equal(parseHouseLivePollingRun(missing), null);
  const leaked = clone(offView); leaked.usedQuestions = clone(onView.usedQuestions);
  assert.equal(parseHouseLivePollingRun(leaked), null);
  const raw = clone(onView); raw.inputPins = [{ path: "/source", sha256: sha("x") }];
  assert.equal(parseHouseLivePollingRun(raw), null);
  const impossible = clone(offView);
  impossible.selectedPoolQuestionCount = 999;
  impossible.selectedPoolDistrictCount = 999;
  assert.equal(parseHouseLivePollingRun(impossible), null);
  assert.ok(HOUSE_LIVE_POLL_EXCLUSION_LABELS["conflicting_question_metadata:pollster"]);
});

test("builder fails closed on ledger, receipt, selector, native IDs and candidate-pair drift", () => {
  const mutations = [
    input => { input.selection.outputs[0].sha256 = sha("wrong ledger"); },
    input => { input.admission.input_pins[0].sha256 = sha("wrong policy"); },
    input => { input.selection.input_pins[0].sha256 = sha("changed selector"); },
    input => { input.ledger[0].reason = "invented_reason"; },
    input => { input.ledger[1].reason = "invented_reason"; },
    input => { input.ledger[0].source_sha256 = sha("wrong source"); },
    input => { input.ledger[0].first_candidate_id = "missing"; },
    input => { input.ledger[0].observation_id = "changed"; },
    input => { input.results[1].result.races.find(race => race.poll_count).poll_source_ids = ["wrong"]; },
    input => { input.results[1].result.qualification.evidence.pop(); },
    input => { input.results[1].result.source_cutoffs[1].at = "2026-10-18T00:00:00Z"; },
  ];
  for (const mutate of mutations) {
    const { input } = fixture(); mutate(input);
    assert.throws(() => buildHouseLivePollingCollection(input), /Live House polling disclosure/);
  }
});

test("only absolute web URLs and narrowly normalized bare domains enter disclosure", () => {
  for (const unsafe of ["//evil.example/path", "https://user:secret@example.invalid/poll",
    "javascript:alert(1)", "example.invalid:80/path", "example.invalid/has space"]) {
    const { input } = fixture();
    input.ledger[0].source_url = unsafe;
    input.ledger[0].source_urls = [unsafe];
    assert.throws(() => buildHouseLivePollingCollection(input), /Live House polling disclosure/);
  }
  const { input } = fixture();
  input.ledger[0].source_url = "polls.example.invalid/house/district";
  input.ledger[0].source_urls = [input.ledger[0].source_url];
  const disclosure = buildHouseLivePollingCollection(input);
  assert.deepEqual(disclosure.selected[0].sourceUrls, ["https://polls.example.invalid/house/district"]);
  const missingEnd = fixture().input;
  missingEnd.ledger[1].end_date = "unparseable";
  const withNull = buildHouseLivePollingCollection(missingEnd);
  assert.equal(withNull.excluded[0].endDate, null);
  assert.equal(withNull.excluded[0].reason, "outside_100_day_field_end_window");
  const isoEnd = fixture().input;
  isoEnd.ledger[1].end_date = "2026-06-01";
  assert.equal(buildHouseLivePollingCollection(isoEnd).excluded[0].endDate, "2026-06-01");
});

test("authenticated hashes bind copied artifacts without fixed local basenames", () => {
  const { input } = fixture();
  input.catalog.source_policy.path = "staged/opaque-policy.json";
  input.admission.input_pins[0].path = "research/opaque/policy-copy.json";
  input.admission.input_pins[1].path = "research/opaque/selection-copy.json";
  input.admission.input_pins[2].path = "research/opaque/ledger-copy.json";
  input.selection.outputs[0].path = "research/opaque/selected-records.json";
  for (const row of input.results) {
    const admission = row.result.qualification.evidence.find(pin => pin.sha256 === input.admissionSha256);
    admission.path = "staged/opaque-review-input.json";
  }
  assert.equal(buildHouseLivePollingCollection(input).selected.length, 1);
  input.catalog.source_policy.sha256 = sha("wrong catalog policy");
  assert.throws(() => buildHouseLivePollingCollection(input), /catalog, policy/);
});

test("non-district exclusions disclose their selection reason without exporting unused poll metadata", () => {
  const { input } = fixture();
  input.ledger[1].district_key = null;
  input.ledger[1].reason = "not_one_current_house_district";
  const disclosure = buildHouseLivePollingCollection(input);
  assert.equal(disclosure.excluded[0].pollster, "Record without a current district");
  assert.equal(disclosure.excluded[0].endDate, null);
  assert.equal(disclosure.excludedReasonCounts[0].count, 1);
  assert.equal(disclosure.excludedReasonCounts[0].districtKey, null);
  disclosure.excluded[0].pollster = "Unnecessary source metadata";
  assert.equal(parseHouseLivePollingCollection(disclosure), null);
});
