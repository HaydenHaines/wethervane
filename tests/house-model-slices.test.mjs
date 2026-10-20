import assert from "node:assert/strict";
import test from "node:test";
import { houseModelSlices } from "../lib/house-model-slices.ts";
import { makeHouseResult, fixtureSha256 } from "./fixtures/house.mjs";

const runEntry = result => ({ ...result.run, resultSha256: fixtureSha256(result), raceCount: result.races.length, available: true, unavailableReason: null });
function fixture() {
  const result = makeHouseResult({ year: 2022 });
  const alternative = structuredClone(result);
  alternative.run.id = "synthetic-alternative-2022";
  alternative.run.label = "Synthetic alternative House model";
  alternative.run.configurationId = "synthetic-alternative";
  const earlier = makeHouseResult({ year: 2018 });
  const catalog = { schemaVersion: 1, defaultRunId: result.run.id, selectionNote: "Synthetic fixed reference.",
    capabilities: { historical_races: true, polling: false, district_geometry: false, full_chamber: false, chamber_control: false, chamber_tipping: false, current_2026: false },
    runs: [result, alternative, earlier].map(runEntry) };
  return { result, alternative, earlier, catalog };
}

test("all eight House responsibilities are present, with only the saved fundamentals choice enabled", () => {
  const { result, catalog } = fixture();
  const slices = houseModelSlices(result, catalog);
  assert.deepEqual(slices.map(slice => slice.id), ["population", "community", "fundamentals", "environment", "measurement", "participation", "choice", "counts_rules"]);
  const enabled = slices.filter(slice => slice.controls.length);
  assert.equal(enabled.length, 1);
  assert.equal(enabled[0].id, "fundamentals");
  const fixed = slices.filter(slice => slice.builtIn);
  assert.deepEqual(fixed.map(slice => slice.id), ["environment", "choice", "counts_rules"]);
  assert.ok(fixed.every(slice => slice.unavailableReason === null && slice.builtInReason.length > 30));
  assert.equal(slices.filter(slice => !slice.controls.length && !slice.builtIn).length, 4);
  const control = enabled[0].controls[0];
  assert.equal(control.label, "House model");
  assert.equal(control.value, result.run.id);
  assert.equal(control.required, true);
  assert.equal(control.toggle, undefined);
  assert.equal(control.options.length, 2);
  assert.ok(control.options.every(option => option.years[0] === 2022 && option.availability === "saved"));
  for (const slice of slices.filter(slice => !slice.controls.length && !slice.builtIn)) {
    assert.ok(slice.unavailableReason.length > 30);
    assert.ok(!slice.unavailableReason.includes("Senate"));
  }
  assert.match(slices.find(slice => slice.id === "environment").builtInReason, /bundled into each saved House model/);
  assert.match(slices.find(slice => slice.id === "measurement").unavailableReason, /Polling is not connected/);
  assert.match(slices.find(slice => slice.id === "counts_rules").builtInReason, /does not model vote counts or cover all 435 districts/);
});

test("unavailable saved variants retain reasons and never become optional component toggles", () => {
  const { result, catalog } = fixture();
  catalog.runs[1] = { ...catalog.runs[1], available: false, resultSha256: null, unavailableReason: "Synthetic diagnostic failure; not reviewed for publication." };
  const control = houseModelSlices(result, catalog).find(slice => slice.id === "fundamentals").controls[0];
  assert.equal(control.options[1].availability, "unavailable");
  assert.match(control.options[1].description, /diagnostic failure/);
  assert.equal(control.toggle, undefined);
});

test("future capability flags and Senate-like offers cannot activate new House roles", () => {
  const { result, catalog } = fixture();
  for (const key of Object.keys(catalog.capabilities)) catalog.capabilities[key] = true;
  catalog.slots = [{ id: "polling", options: [{ id: "standard", availability: "saved", years: [2022] }] }];
  catalog.componentOffers = [{ role: "environment", availability: "available" }, { role: "measurement", availability: "available" }];
  const slices = houseModelSlices(result, catalog);
  assert.equal(slices.filter(slice => slice.controls.length).length, 1);
  assert.equal(slices.filter(slice => !slice.controls.length && !slice.builtIn).length, 4);
  assert.equal(slices.filter(slice => slice.builtIn).length, 3);
  assert.equal(slices.flatMap(slice => slice.controls).some(control => control.toggle), false);
  assert.match(slices.find(slice => slice.id === "measurement").unavailableReason, /No separately selectable House polling component/);
  result.metadata.full_chamber = true;
  result.metadata.scope = "full_chamber";
  const full = houseModelSlices(result, catalog);
  assert.equal(full.find(slice => slice.id === "counts_rules").builtIn, true);
  assert.match(full.find(slice => slice.id === "counts_rules").builtInReason, /Chamber calculations and supported race rules are fixed/);
  assert.doesNotMatch(full.find(slice => slice.id === "counts_rules").builtInReason, /coverage is missing/);
});

test("a single legacy result has fixed fundamentals without inventing a catalog selector", () => {
  const { result } = fixture();
  const control = houseModelSlices(result).find(slice => slice.id === "fundamentals").controls[0];
  assert.equal(control.options.length, 1);
  assert.equal(control.options[0].id, result.run.id);
  assert.equal(control.options[0].label, result.run.label);
  assert.equal(control.required, true);
});

test("selection support follows the displayed year and never rebinds a missing run", () => {
  const { result, earlier, catalog } = fixture();
  const before = structuredClone({ result, earlier, catalog });
  const earlierControl = houseModelSlices(earlier, catalog).find(slice => slice.id === "fundamentals").controls[0];
  assert.equal(earlierControl.value, earlier.run.id);
  assert.deepEqual(earlierControl.options.map(option => option.id), [earlier.run.id]);
  const missing = { ...catalog, runs: catalog.runs.filter(run => run.year === 2018) };
  const fixed = houseModelSlices(result, missing).find(slice => slice.id === "fundamentals").controls[0];
  assert.deepEqual(fixed.options.map(option => option.id), [result.run.id]);
  fixed.options[0].label = "Changed view copy";
  assert.deepEqual({ result, earlier, catalog }, before);
});
