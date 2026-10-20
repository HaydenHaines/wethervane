import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { requestForRun, resolveComposition } from "../lib/composition-resolver.ts";

const catalog = JSON.parse(await readFile("public/data/explorer/capabilities.json", "utf8"));
const registry = JSON.parse(await readFile("public/data/explorer/registry.json", "utf8"));

function requestFor(configuration, changes = {}) {
  return {
    catalogId: catalog.id,
    year: configuration.year,
    dateId: configuration.dateId,
    components: { ...configuration.components },
    ...changes,
  };
}

test("all 38 exact saved component configurations resolve to their frozen run", () => {
  assert.equal(catalog.saved.length, 38);
  for (const configuration of catalog.saved) {
    const entry = registry.runs.find((run) => run.id === configuration.runId);
    assert.ok(entry, configuration.runId);
    const resolution = resolveComposition(catalog, registry, requestFor(configuration));
    assert.equal(resolution.state, "ready", configuration.runId);
    assert.equal(resolution.configurationId, configuration.id);
    assert.equal(resolution.resultId, entry.sha256);
    assert.equal(resolution.run.id, entry.id);
    assert.deepEqual(resolution.outputs, configuration.outputs);
  }
});

test("canonical presets resolve only to their exact server-owned saved runs", () => {
  for (const preset of catalog.presets) {
    const entry = registry.runs.find((run) => run.id === preset.runId);
    const request = requestForRun(catalog, entry);
    request.presetId = preset.id;
    const resolution = resolveComposition(catalog, registry, request);
    assert.equal(resolution.state, "ready");
    assert.equal(resolution.run.id, preset.runId);
    assert.equal(resolution.configuration.runId, preset.runId);
  }
  const preset = catalog.presets[0];
  const configuration = catalog.saved.find((item) => item.runId === preset.runId);
  const conflict = resolveComposition(catalog, registry, requestFor(configuration, {
    presetId: preset.id,
    components: { base: "fundamentals", polling: "standard", uncertainty: "standard" },
  }));
  assert.equal(conflict.state, "incompatible");
});

test("run request helper copies exact server metadata and rejects an unknown run", () => {
  const entry = registry.runs[0];
  assert.deepEqual(requestForRun(catalog, entry), {
    catalogId: catalog.id,
    year: entry.year,
    dateId: entry.dateId,
    components: entry.components,
  });
  assert.throws(() => requestForRun(catalog, { ...entry, sha256: "0".repeat(64) }));
});

test("stale or tampered hashes and scientific IDs never resolve ready", () => {
  const configuration = catalog.saved.find((item) => item.runId === "current-polling");
  const request = requestFor(configuration);

  const staleCatalog = structuredClone(catalog);
  staleCatalog.saved.find((item) => item.runId === "current-polling").resultSha256 = "0".repeat(64);
  assert.equal(resolveComposition(staleCatalog, registry, request).state, "incompatible");

  const invalidConfigurationId = structuredClone(catalog);
  invalidConfigurationId.saved.find((item) => item.runId === "current-polling").id = "current-polling";
  assert.equal(resolveComposition(invalidConfigurationId, registry, request).state, "incompatible");

  const tamperedRegistry = structuredClone(registry);
  tamperedRegistry.runs.find((run) => run.id === "current-polling").sha256 = "1".repeat(64);
  assert.equal(resolveComposition(catalog, tamperedRegistry, request).state, "incompatible");

  const changedTarget = structuredClone(registry);
  changedTarget.runs.find((run) => run.id === "current-polling").targetId = "other-target";
  assert.equal(resolveComposition(catalog, changedTarget, request).state, "incompatible");

  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, { catalogId: "old-catalog" })).state,
    "incompatible");
});

test("only the four declared 2026 aggregate families return metadata-only requirements", () => {
  assert.equal(catalog.computation.length, 4);
  for (const requirement of catalog.computation) {
    for (const polling of ["none", "standard"]) {
      const resolution = resolveComposition(catalog, registry, {
        catalogId: catalog.id,
        year: 2026,
        dateId: "2026-10-20",
        components: { base: requirement.baseId, polling, uncertainty: "standard" },
      });
      assert.equal(resolution.state, "requires_computation");
      assert.equal(resolution.configurationId, null);
      assert.equal(resolution.resultId, null);
      assert.deepEqual(resolution.outputs, ["race-probabilities"]);
      assert.equal(resolution.executionAvailable, false);
      assert.match(resolution.reason, /no custom chamber adapter/i);
    }
  }
});

test("unknown slots, unsupported combinations and missing choices are incompatible", () => {
  const configuration = catalog.saved.find((item) => item.runId === "current-fundamentals");
  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, {
    components: { ...configuration.components, community: "population_cut_v1" },
  })).state, "incompatible");
  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, {
    components: { base: "fundamentals", polling: "standard" },
  })).state, "incompatible");
  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, {
    components: { base: "senate_history_refreshed_v1", polling: "persistent", uncertainty: "standard" },
  })).state, "incompatible");
  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, {
    components: { base: "population_cut_v1", polling: "none", uncertainty: "standard" },
  })).state, "incompatible");
  assert.equal(resolveComposition(catalog, registry, requestFor(configuration, { presetId: "unknown" })).state,
    "incompatible");
});

test("catalog-defined extra slots resolve saved combinations and block undeclared computation", () => {
  const extendedCatalog = structuredClone(catalog);
  const extendedRegistry = structuredClone(registry);
  extendedCatalog.slots.push({
    id: "candidate",
    label: "Candidate adjustment",
    description: "Synthetic fourth slot for resolver extensibility coverage.",
    options: [
      { id: "signed-incumbency", label: "Incumbency", description: "Saved", years: [2026], availability: "saved", experimental: false },
      { id: "off", label: "Off", description: "Uncomputed", years: [2026], availability: "requires_computation", experimental: true },
    ],
  });
  const configuration = extendedCatalog.saved.find((item) => item.runId === "current-polling");
  const entry = extendedRegistry.runs.find((run) => run.id === "current-polling");
  configuration.components.candidate = "signed-incumbency";
  entry.components.candidate = "signed-incumbency";

  const saved = resolveComposition(extendedCatalog, extendedRegistry, requestFor(configuration));
  assert.equal(saved.state, "ready");
  assert.equal(saved.run.id, "current-polling");

  const undeclared = resolveComposition(extendedCatalog, extendedRegistry, {
    catalogId: extendedCatalog.id,
    year: 2026,
    dateId: "2026-10-20",
    components: {
      base: "senate_history_refreshed_v1",
      polling: "none",
      uncertainty: "standard",
      candidate: "off",
    },
  });
  assert.equal(undeclared.state, "incompatible");
});
