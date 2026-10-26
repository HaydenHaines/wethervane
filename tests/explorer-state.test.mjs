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

test("disabling polls never silently changes an unsupported uncertainty selection", () => {
  const result = selectRunForComponents(registry, runs[2], { polling: "none" });
  assert.equal(result.run.id, "current-error-high");
  assert.match(result.notice, /unchanged/);
});

test("unsupported combinations never borrow an unrelated forecast", () => {
  const result = selectRunForComponents(registry, runs[0], { polling: "persistent" });
  assert.equal(result.run.id, runs[0].id);
  assert.match(result.notice, /unchanged/);
});

test("changing the historical base never silently removes polling", () => {
  const result = selectRunForComponents(registry, runs[3], { base: "winner" });
  assert.equal(result.run.id, "history-2022-FP-1");
  assert.match(result.notice, /unchanged/);
});

test("a year change resolves to that year's actual saved uncertainty law", () => {
  const result = selectRunForYear(registry, runs[2], 2022);
  assert.equal(result.run.id, "history-2022-FP-1");
  assert.match(result.notice, /standard uncertainty/);
});


test("chosen comparison survives shared URL and rejects a different year", () => {
  const current = parseSelection("?run=history-2022-FP-1&compare=1&against=history-2022-W-1", registry).selection;
  assert.equal(current.baselineRunId, "history-2022-W-1");
  assert.equal(parseSelection(serializeSelection(current), registry).selection.baselineRunId, current.baselineRunId);
  assert.equal(parseSelection("?run=history-2022-FP-1&against=current-polling", registry).selection.baselineRunId, null);
});

import { selectRunForDate } from "../lib/explorer-state.ts";
import { mixedSenateEditionFixture } from "./fixtures/senate-mixed-edition.mjs";
import { requestForRun, resolveComposition } from "../lib/composition-resolver.ts";

test("every current saved component choice remains reachable across a 10 FP / 6 F edition split", () => {
  const { registry, catalog } = mixedSenateEditionFixture();
  const current = registry.runs.filter(r => r.year === 2026);
  assert.equal(current.length, 16);
  assert.equal(current.filter(r => r.dateId === "2026-10-26").length, 10);
  assert.equal(current.filter(r => r.dateId === "2026-10-20").length, 6);
  const original = JSON.stringify(registry);
  for (const from of current) for (const target of current) {
    const result = selectRunForComponents(registry, from, target.components);
    assert.equal(result.run.id, target.id, `${from.id} -> ${target.id}`);
    assert.equal(result.run.dateId, target.dateId);
    assert.equal(result.run.cutoffAt, target.cutoffAt);
    if (from.dateId !== target.dateId) assert.match(result.notice, /edition.*original input cutoff/);
    else assert.equal(result.notice, null);
    const exactRequest = requestForRun(catalog, result.run);
    assert.equal(exactRequest.dateId, target.dateId);
    assert.equal(resolveComposition(catalog, registry, exactRequest).run.id, target.id);
  }
  assert.equal(JSON.stringify(registry), original);
});

test("component switches preserve every F/FP pair's actual artifact date and uncertainty", () => {
  const { registry } = mixedSenateEditionFixture();
  const noPolls = registry.runs.filter(r => r.year === 2026 && r.components.polling === "none");
  for (const f of noPolls) {
    const on = selectRunForComponents(registry, f, { polling: "standard" });
    assert.equal(on.run.dateId, "2026-10-26"); assert.equal(on.run.components.base, f.components.base);
    const off = selectRunForComponents(registry, on.run, { polling: "none" });
    assert.equal(off.run.id, f.id); assert.equal(off.run.dateId, "2026-10-20");
  }
  const sensitivity = registry.runs.find(r => r.id === "current-error-high");
  assert.equal(selectRunForComponents(registry, sensitivity, { polling: "none" }).run.id, sensitivity.id);
});

test("explicit dates and old-date preset requests never borrow a newer saved edition", () => {
  const { registry, catalog } = mixedSenateEditionFixture();
  const current = registry.runs.find(r => r.id === "current-polling");
  const dated = selectRunForDate(registry, current, "2026-10-20");
  assert.equal(dated.run.id, "current-fundamentals"); assert.equal(dated.run.dateId, "2026-10-20");
  assert.match(dated.notice, /requested.*2026-10-20/);
  const next = selectRunForDate(registry, dated.run, "2026-10-26");
  assert.equal(next.run.id, "current-polling"); assert.equal(next.run.dateId, "2026-10-26");
  const absent = selectRunForDate(registry, current, "2026-10-19");
  assert.equal(absent.run.id, current.id); assert.match(absent.notice, /requested date.*unchanged/);
  const request = { ...requestForRun(catalog, current), dateId: "2026-10-20" };
  const resolution = resolveComposition(catalog, registry, request);
  assert.notEqual(resolution.state, "ready"); assert.equal(resolution.request.dateId, "2026-10-20");
  const preset = catalog.presets.find(p => p.runId === "current-polling");
  assert.equal(resolveComposition(catalog, registry, { ...request, presetId: preset.id }).state, "incompatible");
  assert.equal(resolveComposition(catalog, registry, { ...requestForRun(catalog, current), presetId: preset.id }).run.id, current.id);
});

test("historical component intent stays date constrained and ambiguous current editions require a date choice", () => {
  const { registry } = mixedSenateEditionFixture();
  const current = registry.runs.find(r => r.id === "current-polling");
  const old = registry.runs.find(r => r.id === "current-fundamentals");
  const second = structuredClone(old); second.id = "INVENTED-second-fundamentals-edition"; second.dateId = "2026-10-19";
  registry.runs.push(second);
  const ambiguous = selectRunForComponents(registry, current, { polling: "none" });
  assert.equal(ambiguous.run.id, current.id); assert.match(ambiguous.notice, /more than one saved edition/);
  assert.equal(selectRunForComponents(registry, old, { polling: "none" }).run.id, old.id); // Same-date choice still wins.
  const history = registry.runs.find(r => r.id === "history-2022-FP-1");
  const missing = structuredClone(history); missing.id = "INVENTED-later-history"; missing.dateId = "h7"; missing.components.polling = "INVENTED-polls";
  registry.runs.push(missing);
  assert.equal(selectRunForComponents(registry, history, { polling: "INVENTED-polls" }).run.id, history.id);
  const yearChange = selectRunForYear(registry, current, 2022);
  assert.equal(yearChange.run.id, "history-2022-FP-1");
});

test("comparison URLs reject different dates, laws and targets after mixed-edition component switches", () => {
  const { registry } = mixedSenateEditionFixture();
  const f = registry.runs.find(r => r.id === "current-comparison-senate_relative_lean_v1-f");
  const fp = registry.runs.find(r => r.id === "current-comparison-senate_relative_lean_v1-fp");
  assert.equal(parseSelection(`?run=${fp.id}&compare=1&against=${f.id}`, registry).selection.baselineRunId, null);
  const peer = registry.runs.find(r => r.id === "current-comparison-senate_relative_context_v1-fp");
  assert.equal(parseSelection(`?run=${fp.id}&compare=1&against=${peer.id}`, registry).selection.baselineRunId, peer.id);
  assert.equal(parseSelection(`?run=${fp.id}&against=current-polling`, registry).selection.baselineRunId, null);
});
