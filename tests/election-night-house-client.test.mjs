import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadPublishedElectionNightForecast } from "../lib/election-night-client.ts";
import { houseLiveCatalogView, houseLiveView } from "../lib/house-live-view.ts";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { makeCompatibilityHouseReference } from "./fixtures/house-expected-reference.mjs";

const read = relative => JSON.parse(readFileSync(new URL(`../${relative}`, import.meta.url), "utf8"));
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const rawCatalog = read("data/house-live/catalog.json");
const nativeCatalog = parseHouseLiveCatalog(rawCatalog);
const selected = nativeCatalog.entries.find(row => row.id === "house-current-fundamentals");
const native = parseHouseLiveResult(read(`data/house-live/${selected.result.path}`));
const view = houseLiveView(native, selected);
const forecast = toHouseElectionNightForecast(view);
const actualIndex = read("public/data/election-night/publication-index.json");
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
function setup() {
  const { catalog, qualification } = makeCompatibilityHouseReference(forecast);
  const body = Buffer.from(JSON.stringify(catalog));
  const index = structuredClone(actualIndex);
  index.house = { schemaVersion: 1, href: "/data/election-night/house-references.json", sha256: sha(body),
    editions: [{ chamber: "house", runId: forecast.run.id, resultSha256: forecast.run.artifactSha256,
      qualification }] };
  const publication = { schema: "HouseLivePublicationV1", status: "available",
    catalog: houseLiveCatalogView(nativeCatalog), view: structuredClone(view), geography: null };
  return { body, index, publication };
}
function serve(fixture) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push(url); options.signal.throwIfAborted();
    const body = String(url).startsWith("/api/chambers/house/live") ? JSON.stringify(fixture.publication)
      : url === "/data/election-night/publication-index.json" ? JSON.stringify(fixture.index)
      : url === "/data/election-night/house-references.json" ? fixture.body : "unexpected URL";
    return new Response(body, { status: body === "unexpected URL" ? 404 : 200 });
  };
  return calls;
}

test("the optional exact House catalog attaches 435 winner and 412 pair expectations", async () => {
  const fixture = setup(), calls = serve(fixture);
  const loaded = await loadPublishedElectionNightForecast({ chamber: "house", runId: selected.id }, new AbortController().signal);
  assert.equal(loaded.forecast.run.artifactSha256, selected.result.sha256);
  assert.equal(loaded.houseReference?.resultSha256, selected.result.sha256);
  assert.equal(loaded.houseReference?.perRace.length, 435);
  assert.equal(loaded.houseReference?.pairRaceCount, 412);
  assert.equal(loaded.houseReference?.pairInapplicableRaceCount, 23);
  assert.equal(loaded.houseReferenceError, null);
  assert.equal(loaded.reference, null); assert.equal(loaded.primaryReference, null);
  assert.ok(calls.includes("/data/election-night/house-references.json"));
});

test("same House run ID with new result SHA never borrows old errors, and malformed optional House block preserves forecast", async () => {
  const stale = setup(); stale.publication.view.run.resultSha256 = "2".repeat(64);
  stale.publication.catalog.entries.find(row => row.id === selected.id).resultSha256 = "2".repeat(64);
  const calls = serve(stale);
  const loaded = await loadPublishedElectionNightForecast({ chamber: "house", runId: selected.id }, new AbortController().signal);
  assert.equal(loaded.forecast.run.artifactSha256, "2".repeat(64));
  assert.equal(loaded.houseReference, null); assert.equal(loaded.houseReferenceError, null);
  assert.ok(!calls.includes("/data/election-night/house-references.json"));
  const malformed = setup(); malformed.index.house.editions[0].qualification.rows = -1;
  serve(malformed);
  const failure = await loadPublishedElectionNightForecast({ chamber: "house", runId: selected.id }, new AbortController().signal);
  assert.equal(failure.forecast.run.artifactSha256, selected.result.sha256);
  assert.equal(failure.houseReference, null);
  assert.match(failure.houseReferenceError, /list could not be verified/);
  assert.equal(failure.referenceError, null);
});

test("changed House catalog bytes and missing House field leave the authenticated forecast available", async () => {
  const changed = setup(); changed.body = Buffer.concat([changed.body, Buffer.from(" ")]);
  serve(changed);
  const loaded = await loadPublishedElectionNightForecast({ chamber: "house", runId: selected.id }, new AbortController().signal);
  assert.equal(loaded.houseReference, null);
  assert.match(loaded.houseReferenceError, /could not be verified/);
  const absent = setup(); delete absent.index.house;
  const calls = serve(absent);
  const noReference = await loadPublishedElectionNightForecast({ chamber: "house", runId: selected.id }, new AbortController().signal);
  assert.equal(noReference.forecast.run.artifactSha256, selected.result.sha256);
  assert.equal(noReference.houseReference, null); assert.equal(noReference.houseReferenceError, null);
  assert.ok(!calls.includes("/data/election-night/house-references.json"));
});
