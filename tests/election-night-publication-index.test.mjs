import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { build, exportIndex, REPO, OUTPUT, FULL, PRIMARY, HOUSE, HREFS } from "../scripts/export_election_night_publication_index.mjs";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { houseLiveView } from "../lib/house-live-view.ts";
import { toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { makeCompatibilityHouseReference } from "./fixtures/house-expected-reference.mjs";

const json = path => JSON.parse(readFileSync(path, "utf8"));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");

test("runtime publication index is deterministic and pins exact catalog/source bytes", () => {
  const actual = build(REPO), saved = json(join(REPO, OUTPUT));
  assert.deepEqual(actual, saved);
  const folder = mkdtempSync(join(tmpdir(), "night-index-"));
  try {
    const output = join(folder, "index.json");
    exportIndex(REPO, output);
    const first = readFileSync(output);
    exportIndex(REPO, output);
    assert.deepEqual(readFileSync(output), first);
    for (const role of ["full", "primary"]) {
      const body = readFileSync(join(REPO, "public", HREFS[role].slice(1)));
      assert.equal(actual[role].sha256, sha(body));
      const catalog = JSON.parse(body);
      for (const edition of actual[role].editions) {
        const row = catalog.entries.find(item => item.runId === edition.runId);
        assert.equal(edition.qualification.sourcePins.forecastSha256, row.sourcePins.forecastSha256);
        assert.ok(edition.qualification.configuration);
        assert.equal(typeof edition.qualification.strictAsOf, "boolean");
      }
    }
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test("a repointed run cannot inherit old reference; changed catalog bytes fail", () => {
  const repo = mkdtempSync(join(tmpdir(), "night-index-"));
  try {
    for (const relative of [FULL, PRIMARY, "public/data/explorer/registry.json",
      "public/data/explorer/current-fundamentals.json", "public/data/explorer/current-polling.json",
      "public/data/election-night/references.json", "public/data/election-night/core-references.json"]) {
      const target = join(repo, relative);
      mkdirSync(dirname(target), { recursive: true });
      copyFileSync(join(REPO, relative), target);
    }
    const registryPath = join(repo, "public/data/explorer/registry.json"), registry = json(registryPath);
    registry.runs = registry.runs.filter(row => ["current-fundamentals", "current-polling"].includes(row.id));
    registry.runs.find(row => row.id === "current-polling").sha256 = "0".repeat(64);
    writeFileSync(registryPath, JSON.stringify(registry));
    const index = build(repo);
    assert.notEqual(index.full.editions[0].resultSha256,
      registry.runs.find(row => row.id === "current-polling").sha256);
    assert.equal(index.primary.editions[1].qualification.structuralCutoffAt,
      index.primary.editions[0].qualification.forecastCutoffAt);
    assert.equal(index.full.editions[0].qualification.configuration, undefined);
    const catalog = join(repo, "public/data/election-night/references.json");
    writeFileSync(catalog, Buffer.concat([readFileSync(catalog), Buffer.from(" ")]));
    assert.throws(() => build(repo), /Changed full catalog bytes/);
  } finally { rmSync(repo, { recursive: true, force: true }); }
});

test("optional House publication builds exact descriptors without editing the renderer or weakening Senate fields", () => {
  const repo = mkdtempSync(join(tmpdir(), "night-house-index-"));
  try {
    for (const relative of [FULL, PRIMARY, "public/data/explorer/registry.json",
      "public/data/explorer/current-fundamentals.json", "public/data/explorer/current-polling.json",
      "public/data/election-night/references.json", "public/data/election-night/core-references.json",
      "data/house-live/catalog.json", "data/house-live/results/house-current-fundamentals.json"]) {
      const target = join(repo, relative); mkdirSync(dirname(target), { recursive: true });
      copyFileSync(join(REPO, relative), target);
    }
    const current = parseHouseLiveCatalog(json(join(repo, "data/house-live/catalog.json")));
    const entry = current.entries.find(row => row.id === "house-current-fundamentals");
    const native = parseHouseLiveResult(json(join(repo, "data/house-live", entry.result.path)));
    const forecast = toHouseElectionNightForecast(houseLiveView(native, entry));
    const wrapped = makeCompatibilityHouseReference(forecast), body = Buffer.from(JSON.stringify(wrapped.catalog));
    const publicPath = join(repo, "public", HREFS.house.slice(1));
    mkdirSync(dirname(publicPath), { recursive: true }); writeFileSync(publicPath, body);
    const row = wrapped.catalog.entries[0];
    const { chamber, runId, resultSha256, reference, ...qualification } = row;
    const manifest = { schemaVersion: 1, href: HREFS.house, sha256: sha(body),
      editions: [{ chamber, runId, resultSha256, qualification }] };
    writeFileSync(join(repo, HOUSE), JSON.stringify(manifest));
    const index = build(repo);
    assert.deepEqual(index.house.editions[0], manifest.editions[0]);
    assert.deepEqual(index.full, build(REPO).full);
    assert.deepEqual(index.primary, build(REPO).primary);
    const houseCatalog = json(join(repo, "data/house-live/catalog.json"));
    houseCatalog.entries.find(row => row.id === entry.id).result.sha256 = "0".repeat(64);
    writeFileSync(join(repo, "data/house-live/catalog.json"), JSON.stringify(houseCatalog));
    const stale = build(repo);
    assert.equal(stale.house.editions[0].resultSha256, entry.result.sha256);
    writeFileSync(publicPath, Buffer.concat([body, Buffer.from(" ")]));
    assert.throws(() => build(repo), /Changed House catalog bytes/);
  } finally { rmSync(repo, { recursive: true, force: true }); }
});
