import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { build, exportIndex, REPO, OUTPUT, FULL, PRIMARY, HREFS } from "../scripts/export_election_night_publication_index.mjs";

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
