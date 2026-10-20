import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { HOUSE_CAPABILITY_KEYS, parseHouseCatalog, parseHousePublicationWithCatalog } from "../lib/house-catalog.ts";
import { HouseRunSelectionError, loadHousePublication } from "../lib/house-publication.ts";
import { importHousePublication } from "../scripts/import_house_publication.mjs";
import { makeHousePublication, makeHouseResult } from "./fixtures/house.mjs";

const bytes = value => Buffer.from(JSON.stringify(value, null, 2) + "\n");
const hash = value => createHash("sha256").update(value).digest("hex");
const capabilities = Object.fromEntries(HOUSE_CAPABILITY_KEYS.map(key => [key, key === "historical_races"]));
const publicEntry = result => ({ ...result.run, resultSha256: hash(bytes(result)), raceCount: result.races.length, available: true, unavailableReason: null });
const unavailableEntry = () => ({ ...makeHouseResult({ year: 2020 }).run, resultSha256: null, raceCount: 16, available: false, unavailableReason: "Numerical diagnostics did not pass review." });
const catalogOf = results => ({ schemaVersion: 1, defaultRunId: results[0].run.id, selectionNote: "Explicit presentation reference; no outcome-based selection.", runs: [...results.map(publicEntry), unavailableEntry()], capabilities: { ...capabilities } });

async function temporary(fn) {
  const directory = await mkdtemp(join(tmpdir(), "house-catalog-"));
  try { await fn(directory); } finally { await rm(directory, { recursive: true, force: true }); }
}
async function writeCatalog(directory, results, mutate = () => {}) {
  const catalog = catalogOf(results);
  for (let index = 0; index < results.length; index++) {
    const filename = `result-${index}.json`;
    await writeFile(join(directory, filename), bytes(results[index]));
    catalog.runs[index].result = { file: filename, sha256: catalog.runs[index].resultSha256 };
  }
  mutate(catalog);
  await writeFile(join(directory, "catalog.json"), bytes(catalog));
  await writeFile(join(directory, "publication.json"), bytes({ schemaVersion: 1, status: "available", catalog: { file: "catalog.json", sha256: hash(bytes(catalog)) } }));
  return catalog;
}

test("browser catalog parser clones presentation fields and strips disk references", () => {
  const result = makeHouseResult();
  const raw = catalogOf([result]);
  raw.runs[0].result = { file: "/private/unpublished.json", sha256: "secret" };
  raw.privateSource = "/private";
  const parsed = parseHouseCatalog(raw);
  assert.ok(parsed);
  assert.equal(parsed.defaultRunId, result.run.id);
  assert.equal(parsed.runs[0].available, true);
  assert.equal(parsed.runs[1].available, false);
  assert.equal("result" in parsed.runs[0], false);
  assert.equal("privateSource" in parsed, false);
  parsed.runs[0].label = "Changed clone";
  assert.notEqual(parsed.runs[0].label, raw.runs[0].label);
});

test("malformed, duplicate, unavailable-default and unbounded catalogs fail closed", () => {
  for (const mutate of [
    value => { value.defaultRunId = value.runs[1].id; },
    value => { value.runs.push(value.runs[0]); },
    value => { value.runs[0].resultSha256 = "bad"; },
    value => { value.runs[1].resultSha256 = "a".repeat(64); },
    value => { value.runs[1].unavailableReason = null; },
    value => { value.runs[0].raceCount = 436; },
    value => { value.runs[0].configurationSha256 = "bad"; },
    value => { value.runs[0].year = 2022.5; },
    value => { value.runs[0].available = "running"; },
    value => { value.runs[0].id = "../escape"; },
    value => { value.capabilities.polling = "false"; },
    value => { value.runs = Array.from({ length: 129 }, () => value.runs[0]); },
  ]) {
    const value = catalogOf([makeHouseResult()]); mutate(value);
    assert.equal(parseHouseCatalog(value), null);
  }
  for (const value of [null, [], {}, "not-json"]) assert.equal(parseHouseCatalog(value), null);
});

test("browser publication parser preserves legacy/preparing and binds advertised result", () => {
  assert.deepEqual(parseHousePublicationWithCatalog({ schemaVersion: 1, status: "preparing" }), { schemaVersion: 1, status: "preparing" });
  const legacy = makeHousePublication();
  assert.equal(parseHousePublicationWithCatalog(legacy)?.status, "available");
  const value = { ...legacy, resultSha256: hash(bytes(legacy.result)), catalog: catalogOf([legacy.result]) };
  assert.equal(parseHousePublicationWithCatalog(value)?.catalog.defaultRunId, legacy.result.run.id);
  for (const mutate of [
    item => { item.catalog.runs[0].label = "Wrong model"; },
    item => { item.catalog.runs[0].configurationId = "Wrong configuration"; },
    item => { item.catalog.runs[0].resultSha256 = "0".repeat(64); },
    item => { item.catalog.runs[0].raceCount++; },
    item => { item.result.run.id = "not-advertised"; },
    item => { item.result.races[0].outcomes[0].winProbability = 2; },
    item => { item.result.metadata.full_chamber = true; },
  ]) {
    const copy = structuredClone(value); mutate(copy);
    assert.equal(parseHousePublicationWithCatalog(copy), null);
  }
  assert.equal(parseHousePublicationWithCatalog({ schemaVersion: 1, status: "preparing", catalog: value.catalog }), null);
  for (const input of [null, {}, [], "bad"]) assert.equal(parseHousePublicationWithCatalog(input), null);
});

test("browser retains valid race results while discarding missing or unadmitted geography", () => {
  for (const mutate of [
    item => { item.geography.metadata.mapVintage = "wrong-edition"; },
    item => { item.geography.metadata.cycle = 2018; },
    item => { item.geography.metadata.admission = "unverified"; },
    item => { item.geography.metadata.sourceSha256 = "bad"; },
    item => { item.geography.geojson.features = []; },
    item => { item.geography = { geojson: { features: [null] } }; },
    item => { delete item.geography; },
    item => { item.geography = null; },
  ]) {
    const value = makeHousePublication(); mutate(value);
    const parsed = parseHousePublicationWithCatalog(value);
    assert.equal(parsed?.status, "available");
    assert.equal(parsed.result.races.length, value.result.races.length);
    assert.equal(parsed.geography, null);
  }
});

test("pinned catalog serves explicit default and chosen run without publishing references", async () => temporary(async directory => {
  const results = [makeHouseResult({ year: 2022 }), makeHouseResult({ year: 2018, eligibleCount: 12 })];
  await writeCatalog(directory, results);
  const defaultPublication = await loadHousePublication(directory);
  assert.equal(defaultPublication.result.run.id, results[0].run.id);
  assert.equal(defaultPublication.catalog.defaultRunId, results[0].run.id);
  assert.equal(defaultPublication.catalog.runs.length, 3);
  assert.equal(JSON.stringify(defaultPublication.catalog).includes("result-0.json"), false);
  const chosen = await loadHousePublication(directory, results[1].run.id);
  assert.equal(chosen.result.run.year, 2018);
  assert.equal(chosen.result.races.length, 12);
  assert.equal(chosen.resultSha256, hash(bytes(results[1])));
  for (const id of ["unknown", "", "../escape", unavailableEntry().id]) {
    await assert.rejects(loadHousePublication(directory, id), HouseRunSelectionError);
  }
}));

test("catalog and every advertised result must match hashes, metadata and eligibility", async t => {
  for (const [name, mutate, expected] of [
    ["metadata mismatch", catalog => { catalog.runs[1].label = "Another model"; }, /metadata does not match/],
    ["race count mismatch", catalog => { catalog.runs[1].raceCount++; }, /metadata does not match/],
    ["result digest mismatch", catalog => { catalog.runs[1].resultSha256 = "0".repeat(64); }, /metadata does not match/],
    ["unavailable reference", catalog => { catalog.runs[2].result = catalog.runs[0].result; }, /must not have publication references/],
    ["traversal", catalog => { catalog.runs[0].result.file = "../escape.json"; }, /Invalid House artifact reference/],
  ]) await t.test(name, async () => temporary(async directory => {
    await writeCatalog(directory, [makeHouseResult({ year: 2022 }), makeHouseResult({ year: 2018 })], mutate);
    await assert.rejects(loadHousePublication(directory), expected);
  }));
  await temporary(async directory => {
    await writeCatalog(directory, [makeHouseResult({ year: 2022 }), makeHouseResult({ year: 2018 })]);
    await writeFile(join(directory, "result-1.json"), "{}");
    await assert.rejects(loadHousePublication(directory), /fingerprint mismatch/);
  });
  await temporary(async directory => {
    await writeCatalog(directory, [makeHouseResult()]);
    await writeFile(join(directory, "catalog.json"), "{}");
    await assert.rejects(loadHousePublication(directory), /fingerprint mismatch/);
  });
});

test("legacy preparing and single-result selections remain bounded", async () => temporary(async directory => {
  await writeFile(join(directory, "publication.json"), bytes({ schemaVersion: 1, status: "preparing" }));
  assert.deepEqual(await loadHousePublication(directory), { schemaVersion: 1, status: "preparing" });
  await assert.rejects(loadHousePublication(directory, "unknown"), HouseRunSelectionError);
  const result = makeHouseResult();
  await writeFile(join(directory, "result.json"), bytes(result));
  await writeFile(join(directory, "publication.json"), bytes({ schemaVersion: 1, status: "available", result: { file: "result.json", sha256: hash(bytes(result)) } }));
  assert.equal((await loadHousePublication(directory, result.run.id)).result.run.id, result.run.id);
  await assert.rejects(loadHousePublication(directory, "unknown"), HouseRunSelectionError);
}));

test("API distinguishes invalid selections from unavailable publication data", async () => temporary(async directory => {
  const publicationDirectory = join(directory, "data/house");
  await mkdir(publicationDirectory, { recursive: true });
  const results = [makeHouseResult({ year: 2022 }), makeHouseResult({ year: 2018 })];
  await writeCatalog(publicationDirectory, results);
  const routeUrl = new URL("../app/api/chambers/house/route.ts", import.meta.url).href;
  const request = query => JSON.parse(execFileSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", `
    import {GET} from ${JSON.stringify(routeUrl)};
    console.error = () => {};
    const response = await GET(new Request('http://example.invalid/api/chambers/house' + ${JSON.stringify(query)}));
    console.log(JSON.stringify({status:response.status, body:await response.json()}));
  `], { cwd: directory, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }));
  const selected = request(`?run=${results[1].run.id}`);
  assert.equal(selected.status, 200);
  assert.equal(selected.body.result.run.year, 2018);
  for (const query of ["?run=unknown", "?run=", "?run=../escape", "?run=a&run=b", "?unrecognized=yes"]) {
    const response = request(query);
    assert.equal(response.status, 400, query);
    assert.equal("result" in response.body, false);
  }
  await writeFile(join(publicationDirectory, "catalog.json"), "{}");
  const invalid = request("");
  assert.equal(invalid.status, 503);
  assert.deepEqual(Object.keys(invalid.body), ["error"]);
  assert.equal(JSON.stringify(invalid.body).includes(directory), false);
}));

async function sourceFixture(directory) {
  const sourceRoot = join(directory, "source"), outputDirectory = join(directory, "published");
  await mkdir(join(sourceRoot, "review"), { recursive: true });
  const pin = async (path, value) => { const data = bytes(value); await writeFile(join(sourceRoot, path), data); return { path, sha256: hash(data) }; };
  const inputs = {};
  for (const key of ["review_source", "panel", "scores"]) inputs[key] = await pin(`review/${key}.json`, { synthetic: true });
  const results = [makeHouseResult({ year: 2022 }), makeHouseResult({ year: 2018 }), makeHouseResult({ year: 2024 })];
  results[2].reviewStatus = "diagnostic_failure";
  const configurations = [], cells = [], outputs = [];
  for (let index = 0; index < results.length; index++) {
    const result = results[index], status = index === 2 ? "diagnostic_failure" : "passed";
    const plan = await pin(`review/plan-${index}.json`, { target_cycle: result.run.year, house_configuration_id: result.run.configurationId });
    result.run.configurationSha256 = plan.sha256;
    const resultPin = await pin(`review/result-${index}.json`, result); outputs.push(resultPin);
    configurations.push({ ...result.run, result: resultPin, race_count: result.races.length, publication_eligible: status === "passed" });
    cells.push({ cycle: result.run.year, configuration: result.run.configurationId, races: result.races.length,
      structural_and_score_audit_passed: true, numerical_status: status, fit_receipt: await pin(`review/fit-${index}.json`, { status, plan }) });
  }
  const catalog = { schema: "house-historical-capability-catalog-v1", configurations, default_selection: null,
    selection_note: "All prespecified laws retained; no winner selected from target outcomes.", capabilities };
  outputs.push(await pin("review/catalog.json", catalog));
  const receipt = { schema: "house-actual-output-review-v1", outputs, cells, ...inputs };
  await pin("review/review-receipt.json", receipt);
  return { sourceRoot, outputDirectory, catalogPath: "review/catalog.json", reviewReceiptPath: "review/review-receipt.json", defaultRunId: results[0].run.id, catalog, receipt, results };
}

test("importer stages reviewed outputs, retains unavailable metadata, and preserves source bytes", async () => temporary(async directory => {
  const fixture = await sourceFixture(directory);
  const original = await readFile(join(fixture.sourceRoot, fixture.catalogPath));
  const summary = await importHousePublication(fixture);
  assert.equal(summary.availableRuns, 2);
  assert.equal(summary.unavailableRuns, 1);
  const loaded = await loadHousePublication(fixture.outputDirectory);
  assert.equal(loaded.catalog.runs[2].available, false);
  assert.equal(loaded.catalog.runs[2].resultSha256, null);
  assert.equal((await readdir(fixture.outputDirectory)).filter(name => name.startsWith("result-")).length, 2);
  const unpublishedDigest = fixture.catalog.configurations[2].result.sha256;
  assert.equal((await readdir(fixture.outputDirectory)).some(name => name.includes(unpublishedDigest)), false);
  assert.deepEqual(await readFile(join(fixture.sourceRoot, fixture.catalogPath)), original);
  await importHousePublication(fixture); // Idempotent, content-addressed repeat.
}));

test("importer rejects stale review pins, unqualified defaults and source writes", async t => {
  for (const [name, mutate, expected] of [
    ["no explicit default", fixture => { fixture.defaultRunId = ""; }, /Explicit source/],
    ["unqualified default", fixture => { fixture.defaultRunId = fixture.results[2].run.id; }, /Explicit default/],
    ["stale output", async fixture => { await writeFile(join(fixture.sourceRoot, "review/result-0.json"), "{}"); }, /fingerprint mismatch/],
    ["source output path", fixture => { fixture.outputDirectory = join(fixture.sourceRoot, "new-output"); }, /outside read-only source root/],
    ["receipt source escape", async fixture => {
      fixture.receipt.outputs[0].path = "../outside.json";
      await writeFile(join(fixture.sourceRoot, fixture.reviewReceiptPath), bytes(fixture.receipt));
    }, /Invalid source fingerprint reference/],
    ["catalog metadata mismatch", async fixture => {
      fixture.catalog.configurations[0].label = "Incorrect model";
      const data = bytes(fixture.catalog);
      await writeFile(join(fixture.sourceRoot, fixture.catalogPath), data);
      fixture.receipt.outputs.at(-1).sha256 = hash(data);
      await writeFile(join(fixture.sourceRoot, fixture.reviewReceiptPath), bytes(fixture.receipt));
    }, /metadata or eligibility conflicts/],
  ]) await t.test(name, async () => temporary(async directory => {
    const fixture = await sourceFixture(directory); await mutate(fixture);
    await assert.rejects(importHousePublication(fixture), expected);
    if (name === "source output path") await assert.rejects(readFile(join(fixture.outputDirectory, "publication.json")), { code: "ENOENT" });
  }));
});

test("importer rejects source symlinks that escape the supplied root", async () => temporary(async directory => {
  const fixture = await sourceFixture(directory);
  const outside = join(directory, "outside.json");
  await writeFile(outside, await readFile(join(fixture.sourceRoot, fixture.catalogPath)));
  await symlink(outside, join(fixture.sourceRoot, "catalog-link.json"));
  fixture.catalogPath = "catalog-link.json";
  await assert.rejects(importHousePublication(fixture), /escapes source root/);
}));
