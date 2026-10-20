import assert from "node:assert/strict";
import test from "node:test";
import { createSavedView, parseSavedViews } from "../lib/saved-views.ts";

const configuration = {
  id: "configuration-a", runId: "run-a", resultSha256: "a".repeat(64),
  year: 2042, dateId: "date-a", components: { structure: "history" },
  targetId: "target-a", lawId: "law-a", outputs: ["races"], cutoffs: [], evidence: {},
};
const baseline = { ...configuration, id: "configuration-b", runId: "run-b", resultSha256: "b".repeat(64) };
const catalog = { schemaVersion: 1, saved: [configuration, baseline] };
const selection = {
  runId: "run-a", scenarioId: "scenario-a", raceId: "race-a", compare: true,
  baselineRunId: "run-b", mapView: "actual",
};
const view = (patch = {}) => ({
  id: "view-a", name: "My view", configurationId: configuration.id,
  runId: configuration.runId, resultSha256: configuration.resultSha256,
  scenarioId: selection.scenarioId, raceId: selection.raceId,
  compare: selection.compare, baselineRunId: selection.baselineRunId, mapView: selection.mapView,
  ...patch,
});

test("saved views retain their exact configuration, run, hash, scenario, and baseline binding", () => {
  const original = view();
  assert.deepEqual(parseSavedViews(JSON.stringify([original]), catalog), [original]);
  const created = createSavedView("  My created view  ", configuration, selection);
  assert.match(created.id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
  assert.equal(created.name, "My created view");
  assert.equal(created.configurationId, configuration.id);
  assert.equal(created.runId, configuration.runId);
  assert.equal(created.resultSha256, configuration.resultSha256);
  assert.equal(created.baselineRunId, baseline.runId);
  assert.deepEqual(parseSavedViews(JSON.stringify([created]), catalog), [created]);
  assert.notEqual(createSavedView("Another", configuration, selection).id, created.id);
});

test("stale or tampered bindings are rejected rather than rebound to an available result", () => {
  for (const patch of [
    { configurationId: "unknown-configuration" },
    { runId: baseline.runId },
    { resultSha256: baseline.resultSha256 },
    { configurationId: baseline.id, runId: baseline.runId },
    { resultSha256: "not-a-fingerprint" },
    { baselineRunId: "stale-baseline" },
  ]) {
    assert.deepEqual(parseSavedViews(JSON.stringify([view(patch)]), catalog), []);
  }
  const updatedCatalog = { saved: [{ ...configuration, resultSha256: "c".repeat(64) }, baseline] };
  assert.deepEqual(parseSavedViews(JSON.stringify([view()]), updatedCatalog), []);
  assert.deepEqual(parseSavedViews(JSON.stringify([view({ resultSha256: configuration.resultSha256.toUpperCase() })]), catalog), []);
});

test("malformed JSON, storage shapes, fields, and catalogs do not throw", () => {
  for (const input of [null, undefined, 42, {}, "", "null", "{", '"text"', "42", "{}", "[null,3,[],{}]"]) {
    assert.doesNotThrow(() => parseSavedViews(input, catalog));
    assert.deepEqual(parseSavedViews(input, catalog), []);
  }
  for (const malformedCatalog of [null, {}, { saved: null }, { saved: [null, false] }]) {
    assert.deepEqual(parseSavedViews(JSON.stringify([view()]), malformedCatalog), []);
  }
  for (const patch of [
    { id: "" }, { id: "x".repeat(513) }, { id: " view-a " },
    { name: " " }, { name: 4 }, { name: "line\nbreak" },
    { scenarioId: {} }, { scenarioId: "" }, { raceId: [] }, { raceId: 3 },
    { compare: "true" }, { compare: 1 }, { mapView: "other" },
    { baselineRunId: false }, { configurationId: " configuration-a" },
  ]) {
    assert.deepEqual(parseSavedViews(JSON.stringify([view(patch)]), catalog), []);
  }
});

test("canonical views drop unknown fields, trim display names, and retain nullable selections", () => {
  const original = view({ name: "  Nice view  ", scenarioId: null, raceId: null, extra: { components: "tampered" } });
  delete original.baselineRunId;
  const parsed = parseSavedViews(JSON.stringify([original]), catalog)[0];
  assert.equal(parsed.name, "Nice view");
  assert.equal(parsed.scenarioId, null);
  assert.equal(parsed.raceId, null);
  assert.equal(parsed.baselineRunId, null);
  assert.equal("extra" in parsed, false);
  assert.deepEqual(Object.keys(parsed).sort(), [
    "id", "name", "configurationId", "runId", "resultSha256", "scenarioId", "raceId", "compare", "baselineRunId", "mapView",
  ].sort());
});

test("stored collections and identifier collisions are bounded", () => {
  const many = Array.from({ length: 40 }, (_, index) => view({ id: `view-${index}` }));
  const parsed = parseSavedViews(JSON.stringify(many), catalog);
  assert.equal(parsed.length, 30);
  assert.equal(parsed.at(-1).id, "view-29");
  const duplicate = parseSavedViews(JSON.stringify([view(), view({ name: "Collision" }), view({ id: "view-b" })]), catalog);
  assert.equal(duplicate.length, 2);
  assert.equal(duplicate[0].name, "My view");
  assert.deepEqual(parseSavedViews(" ".repeat(128001), catalog), []);
});

test("names are capped at 80 UTF-16 characters without leaving a broken surrogate", () => {
  const exact = createSavedView("x".repeat(80), configuration, selection);
  const long = createSavedView("x".repeat(120), configuration, selection);
  assert.equal(exact.name.length, 80);
  assert.equal(long.name.length, 80);
  assert.equal(createSavedView(" ", configuration, selection).name, "Saved view");
  const unicode = createSavedView("x".repeat(79) + "🎉", configuration, selection);
  assert.equal(unicode.name, "x".repeat(79));
  assert.deepEqual(parseSavedViews(JSON.stringify([view({ name: "x".repeat(81) })]), catalog), []);
  assert.equal(parseSavedViews(JSON.stringify([view({ name: "x".repeat(80) })]), catalog).length, 1);
});

test("creation rejects a run/configuration mismatch and malformed typed selections", () => {
  assert.throws(() => createSavedView("View", configuration, { ...selection, runId: baseline.runId }), /does not match/);
  assert.throws(() => createSavedView("View", { ...configuration, resultSha256: "bad" }, selection), /bound result identity/);
  assert.throws(() => createSavedView("View", configuration, { ...selection, compare: "false" }), /invalid fields/);
  const result = createSavedView("View", configuration, { ...selection, baselineRunId: undefined });
  assert.equal(result.baselineRunId, null);
});

test("helpers never mutate the catalog or caller selection", () => {
  const catalogBefore = JSON.stringify(catalog), selectionBefore = JSON.stringify(selection);
  createSavedView("View", configuration, selection);
  parseSavedViews(JSON.stringify([view()]), catalog);
  assert.equal(JSON.stringify(catalog), catalogBefore);
  assert.equal(JSON.stringify(selection), selectionBefore);
});
