import assert from "node:assert/strict";
import test from "node:test";
import { makeHouseLiveCatalog } from "./fixtures/house-live-fixtures.mjs";
import { houseLiveCatalogView } from "../lib/house-live-view.ts";
import { houseLiveModelSlices } from "../lib/house-live-model-slices.ts";
import { resolveHouseLiveSelection } from "../lib/house-live-selection.ts";

const catalog = houseLiveCatalogView(makeHouseLiveCatalog());

test("all 16 exact saved House choices resolve to their own run", () => {
  assert.equal(catalog.entries.length, 16);
  for (const entry of catalog.entries) {
    const selected = structuredClone(entry.configuration);
    const resolution = resolveHouseLiveSelection(catalog, selected);
    assert.equal(resolution.state, "ready", entry.id);
    assert.equal(resolution.entry.id, entry.id);
    assert.deepEqual(selected, entry.configuration);
  }
});

test("changing one control never silently changes another to manufacture a saved result", () => {
  const selection = { base: "house_relative_lean_v1", polling: "standard", uncertainty: "error-low" };
  const before = structuredClone(selection);
  const resolution = resolveHouseLiveSelection(catalog, selection);
  assert.equal(resolution.state, "requires_computation");
  assert.equal(resolution.entry, null);
  assert.match(resolution.reason, /no saved result|new joint calculation/i);
  assert.deepEqual(selection, before);

  const noPollSensitivity = resolveHouseLiveSelection(catalog,
    { base: "fundamentals", polling: "none", uncertainty: "cycle-low" });
  assert.equal(noPollSensitivity.state, "requires_computation");
  assert.equal(noPollSensitivity.entry, null);
  assert.equal(resolveHouseLiveSelection(catalog,
    { base: "house_unknown", polling: "standard", uncertainty: "standard" }).state, "unsupported");
});

test("missing or ambiguous catalog bindings cannot become ready", () => {
  const chosen = catalog.entries[0];
  const missing = { ...catalog, entries: catalog.entries.filter(entry => entry.id !== chosen.id) };
  assert.equal(resolveHouseLiveSelection(missing, chosen.configuration).state, "requires_computation");
  const duplicated = { ...catalog, entries: [...catalog.entries, { ...chosen, id: "second-run" }] };
  assert.equal(resolveHouseLiveSelection(duplicated, chosen.configuration).state, "unsupported");
});

test("eight roles remain visible with exact settings and honest disconnected roles", () => {
  const selected = { base: "fundamentals", polling: "standard", uncertainty: "standard" };
  const slices = houseLiveModelSlices(catalog, selected);
  assert.deepEqual(slices.map(slice => slice.id), ["population", "community", "fundamentals",
    "environment", "measurement", "participation", "choice", "counts_rules"]);
  assert.equal(slices.find(slice => slice.id === "fundamentals").controls[0].options.length, 6);
  assert.equal(slices.find(slice => slice.id === "measurement").controls[0].toggle.onValue, "standard");
  assert.equal(slices.find(slice => slice.id === "environment").controls[0].options.length, 5);
  const uncertainty = slices.find(slice => slice.id === "environment").controls[0].options;
  assert.match(uncertainty.find(row => row.id === "error-low").label, /2\/3/);
  assert.match(uncertainty.find(row => row.id === "cycle-low").label, /0\.5/);
  for (const role of ["population", "community", "participation"]) {
    const slice = slices.find(item => item.id === role);
    assert.equal(slice.controls.length, 0);
    assert.match(slice.unavailableReason, /not|no /i);
  }
  for (const role of ["choice", "counts_rules"]) {
    assert.equal(slices.find(slice => slice.id === role).builtIn, true);
  }
});
