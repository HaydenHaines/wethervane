import assert from "node:assert/strict";
import test from "node:test";
import {
  parseSelection,
  serializeSelection,
  selectRunForComponents,
  selectRunForYear,
} from "../lib/explorer-state.ts";

const run = (id, year, base, polling, uncertainty) => ({
  id, year, label: id, dateId: year === 2026 ? "2026-10-20" : "h14",
  components: { base, polling, uncertainty },
  historical: year !== 2026,
});
const runs = [
  run("current-polling", 2026, "fundamentals", "standard", "standard"),
  run("current-fundamentals", 2026, "fundamentals", "none", "standard"),
  run("current-error-high", 2026, "fundamentals", "standard", "error-high"),
  run("history-2022-FP-1", 2022, "fundamentals", "standard", "loading-1"),
  run("history-2022-W-1", 2022, "winner", "none", "loading-1"),
  run("history-2022-FP_persistent-1", 2022, "fundamentals", "persistent", "loading-1"),
  run("history-2018-FP-1", 2018, "fundamentals", "standard", "loading-1"),
];
const registry = {
  schemaVersion: 1,
  defaults: { 2026: "current-polling", 2022: "history-2022-FP-1", 2018: "history-2018-FP-1" },
  runs,
};

test("a shared historical configuration round-trips including special race and actual map", () => {
  const selection = {
    runId: "history-2022-FP-1", scenarioId: "asof_alignment_other_U",
    raceId: "2022:OK:senate:special", compare: true, mapView: "actual",
  };
  assert.deepEqual(parseSelection(serializeSelection(selection), registry), { selection, notice: null });
});

test("unknown configurations get an explicit default notice and no future actual-results mode", () => {
  const result = parseSelection("?run=missing&view=actual", registry);
  assert.equal(result.selection.runId, "current-polling");
  assert.equal(result.selection.mapView, "forecast");
  assert.match(result.notice, /unavailable/);
});

test("disabling polls drops a polling-only error sensitivity with a visible explanation", () => {
  const result = selectRunForComponents(registry, runs[2], { polling: "none" });
  assert.equal(result.run.id, "current-fundamentals");
  assert.match(result.notice, /uncertainty/);
});

test("unsupported combinations never borrow an unrelated forecast", () => {
  const result = selectRunForComponents(registry, runs[0], { polling: "persistent" });
  assert.equal(result.run.id, runs[0].id);
  assert.match(result.notice, /unchanged/);
});

test("changing the historical base to winner-only explicitly removes the polling component", () => {
  const result = selectRunForComponents(registry, runs[3], { base: "winner" });
  assert.equal(result.run.id, "history-2022-W-1");
  assert.match(result.notice, /polling/);
});

test("a year change resolves to that year's actual saved uncertainty law", () => {
  const result = selectRunForYear(registry, runs[2], 2022);
  assert.equal(result.run.id, "history-2022-FP-1");
  assert.match(result.notice, /standard uncertainty/);
});
