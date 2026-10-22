import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { attachHouseGeography } from "../scripts/attach_house_geography.mjs";
import { loadHousePublication } from "../lib/house-publication.ts";
import { makeHousePublication } from "./fixtures/house.mjs";
import { assertGeometryOnlyCatalogExtension } from "../scripts/build_house_comparisons.mjs";

const bytes = value => Buffer.from(JSON.stringify(value));
const hash = value => createHash("sha256").update(value).digest("hex");

test("comparison evidence accepts only a geometry addition, preserving all scientific catalog fields", async () => {
  const baseline = JSON.parse(await readFile(new URL("../data/house/catalog-82c63f9ba70bcd8a9fbbb2037d5bb48d3e9ee8190462f1583caa9a3c8ddb092b.json", import.meta.url)));
  const extended = structuredClone(baseline);
  extended.capabilities.district_geometry = true;
  extended.runs[0].geography = { file: "fixture.json", sha256: "0".repeat(64) };
  assert.doesNotThrow(() => assertGeometryOnlyCatalogExtension(extended, baseline));
  for (const mutate of [
    value => { value.runs[0].resultSha256 = "0".repeat(64); },
    value => { value.runs[0].raceCount--; },
    value => { value.defaultRunId = value.runs[1].id; },
    value => { value.capabilities.chamber_control = true; },
    value => { value.selectionNote = "Different scientific choice"; },
  ]) {
    const changed = structuredClone(extended); mutate(changed);
    assert.throws(() => assertGeometryOnlyCatalogExtension(changed, baseline), /scientific catalog changed/);
  }
});
async function fixture(fn) {
  const root = await mkdtemp(join(tmpdir(), "house-geography-attachment-"));
  const directory = join(root, "publication"), geography = join(root, "geography");
  await mkdir(directory); await mkdir(geography);
  try {
    // Synthetic polygons deliberately test the attachment boundary, not real election geography.
    const full = makeHousePublication({ eligibleCount: 435, year: 2022 });
    const partial = makeHousePublication({ eligibleCount: 16, year: 2022 });
    const resultBytes = bytes(partial.result), resultSha = hash(resultBytes);
    await writeFile(join(directory, "result.json"), resultBytes);
    const catalog = { schemaVersion: 1, defaultRunId: partial.result.run.id, selectionNote: "Synthetic attachment test.",
      capabilities: { historical_races: true, polling: false, district_geometry: false, full_chamber: false, chamber_control: false, chamber_tipping: false, current_2026: false },
      runs: [{ ...partial.result.run, resultSha256: resultSha, raceCount: 16, available: true, unavailableReason: null,
        result: { file: "result.json", sha256: resultSha } }] };
    const catalogBytes = bytes(catalog);
    await writeFile(join(directory, "catalog.json"), catalogBytes);
    const publicationBytes = bytes({ schemaVersion: 1, status: "available", catalog: { file: "catalog.json", sha256: hash(catalogBytes) } });
    await writeFile(join(directory, "publication.json"), publicationBytes);
    const geometryBytes = bytes(full.geography.geojson);
    await writeFile(join(geography, "districts.json"), geometryBytes);
    const manifest = { schemaVersion: 1, editions: [{ cycle: 2022, mapVintage: "house_election_map_2022", file: "districts.json", sha256: hash(geometryBytes), sourceName: "Synthetic test fixture", sourceUrl: "https://example.com/fixture", sourceNote: "Invented polygons, never a real map." }] };
    const manifestPath = join(geography, "manifest.json");
    await writeFile(manifestPath, bytes(manifest));
    await fn({ directory, geography, manifestPath, manifest, publicationBytes, resultBytes, geometry: full.geography.geojson });
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("a separate map preserves scientific bytes and enables only district geography", async () => fixture(async f => {
  const result = await attachHouseGeography({ publicationDirectory: f.directory, geographyManifest: f.manifestPath });
  assert.equal(result.attached[0].modeled, 16);
  assert.equal(result.attached[0].gray, 419);
  assert.deepEqual(await readFile(join(f.directory, "result.json")), f.resultBytes);
  const loaded = await loadHousePublication(f.directory);
  assert.equal(loaded.geography.geojson.features.length, 435);
  assert.equal(loaded.catalog.capabilities.district_geometry, true);
  for (const capability of ["full_chamber", "chamber_control", "chamber_tipping", "current_2026"])
    assert.equal(loaded.catalog.capabilities[capability], false);
  // Repeating the operation is idempotent and preserves all original references.
  assert.deepEqual(await attachHouseGeography({ publicationDirectory: f.directory, geographyManifest: f.manifestPath }), result);
}));

for (const [name, change, expected] of [
  ["wrong election year", f => { f.manifest.editions[0].cycle = 2020; f.manifest.editions[0].mapVintage = "house_election_map_2020"; }, /no published results/],
  ["wrong digest", f => { f.manifest.editions[0].sha256 = "0".repeat(64); }, /fingerprint mismatch/],
  ["duplicate edition", f => { f.manifest.editions.push(f.manifest.editions[0]); }, /duplicate geography/],
  ["wrong district support", async f => {
    f.geometry.features[0].properties.districtKey = "XX-00";
    const changed = bytes(f.geometry);
    await writeFile(join(f.geography, "districts.json"), changed);
    f.manifest.editions[0].sha256 = hash(changed);
  }, /coverage ledger/],
]) test(`${name} fails without changing the publication`, async () => fixture(async f => {
  await change(f); await writeFile(f.manifestPath, bytes(f.manifest));
  await assert.rejects(attachHouseGeography({ publicationDirectory: f.directory, geographyManifest: f.manifestPath }), expected);
  assert.deepEqual(await readFile(join(f.directory, "publication.json")), f.publicationBytes);
  assert.deepEqual(await readFile(join(f.directory, "result.json")), f.resultBytes);
}));
