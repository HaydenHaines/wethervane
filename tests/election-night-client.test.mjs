import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadElectionNightForecast } from "../lib/election-night-client.ts";

const root = new URL("../", import.meta.url);
const registryBytes = readFileSync(new URL("public/data/explorer/registry.json", root));
const registry = JSON.parse(registryBytes);
const entry = registry.runs.find(run => run.id === registry.defaults["2026"]);
const bytes = readFileSync(new URL(`public${entry.href}`, root));
const referencePublication = JSON.parse(readFileSync(new URL("data/election-night-reference-publication.json", root)));
const referenceBytes = readFileSync(new URL(`public${referencePublication.href}`, root));
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
function serve(list = registry, result = bytes, reference = referenceBytes) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.ok(options.signal instanceof AbortSignal);
    return new Response(url === "/data/explorer/registry.json" ? JSON.stringify(list) : url === referencePublication.href ? reference : result);
  };
  return calls;
}

test("the scorecard default follows the current standard registry with exact bytes and independent choices", async () => {
  const calls = serve();
  const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal);
  assert.equal(value.forecast.run.id, entry.id);
  assert.equal(value.forecast.run.artifactSha256, createHash("sha256").update(bytes).digest("hex"));
  assert.equal(value.defaultRunId, entry.id);
  assert.equal(value.forecast.editionStatus, "dated_snapshot");
  assert.equal(value.choices.length, 16);
  assert.equal(value.forecast.races.length, 35);
  assert.deepEqual(calls.map(call => call.url), ["/data/explorer/registry.json", entry.href]);
});

test("changed artifact bytes and selected-result revisions cannot silently replace a forecast", async () => {
  serve(registry, Buffer.concat([bytes, Buffer.from(" ")]));
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal), /does not match this edition/);
  const calls = serve();
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: entry.id, expectedResultSha256: "0".repeat(64) }, new AbortController().signal), /edition changed/);
  assert.equal(calls.length, 1);
});

test("a valid hash cannot make another model's payload become the requested model", async () => {
  const other = registry.runs.find(run => run.id === "current-fundamentals");
  const otherBytes = readFileSync(new URL(`public${other.href}`, root));
  const altered = structuredClone(registry);
  altered.runs.find(run => run.id === entry.id).sha256 = other.sha256;
  serve(altered, otherBytes);
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal), /could not be verified/);
});

test("untrusted registry paths, duplicate identities, and historical choices never fetch a forecast", async () => {
  for (const change of [
    list => { list.runs.find(run => run.id === entry.id).href = "https://example.invalid/forecast.json"; },
    list => { list.runs.push(structuredClone(entry)); },
    list => { list.defaults["2026"] = "history-2022-FP-1"; },
  ]) {
    const altered = structuredClone(registry); change(altered); const calls = serve(altered);
    await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal));
    assert.equal(calls.length, 1);
  }
  const calls = serve();
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: "history-2022-FP-1" }, new AbortController().signal), /not available/);
  assert.equal(calls.length, 1);
});

test("aborting a guide request propagates to its read and cannot manufacture a result", async () => {
  const controller = new AbortController();
  controller.abort();
  globalThis.fetch = async (_url, options) => { options.signal.throwIfAborted(); };
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, controller.signal), { name: "AbortError" });
});

test("the optional error reference authenticates actual bytes and binds only the exact reviewed edition", async () => {
  const calls = serve();
  const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal, referencePublication);
  assert.equal(value.referenceError, null);
  assert.equal(value.reference.runId, entry.id);
  assert.equal(value.reference.resultSha256, entry.sha256);
  assert.equal(value.reference.perRace.length, 35);
  assert.equal(value.reference.summary.meanAbsoluteMarginError.mean, 5.621801760616599);
  assert.equal(value.reference.evaluationLock, false);
  assert.equal(value.reference.observedResultsConsumed, false);
  assert.deepEqual(calls.map(call => call.url), ["/data/explorer/registry.json", entry.href, referencePublication.href]);
  const alternate = registry.runs.find(run => run.id === "current-fundamentals");
  const otherCalls = serve(registry, readFileSync(new URL(`public${alternate.href}`, root)), Buffer.from("not a reference"));
  const other = await loadElectionNightForecast({ chamber: "senate", runId: alternate.id }, new AbortController().signal, referencePublication);
  assert.equal(other.forecast.run.id, alternate.id);
  assert.equal(other.reference, null);
  assert.equal(other.referenceError, null);
  assert.deepEqual(otherCalls.map(call => call.url), ["/data/explorer/registry.json", alternate.href]);
});

test("changed or unavailable reference bytes preserve the forecast but never show unverified expectations", async () => {
  for (const reference of [Buffer.concat([referenceBytes, Buffer.from(" ")]), Buffer.from("{}")]) {
    serve(registry, bytes, reference);
    const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal, referencePublication);
    assert.equal(value.forecast.run.artifactSha256, entry.sha256);
    assert.equal(value.reference, null);
    assert.match(value.referenceError, /could not be verified/);
  }
  const calls = serve();
  const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal,
    { ...referencePublication, href: "https://example.invalid/references.json" });
  assert.equal(value.reference, null);
  assert.match(value.referenceError, /could not be verified/);
  assert.equal(calls.length, 2);
});

test("cancellation during reference loading remains cancellation, not a stale fulfilled forecast", async () => {
  const controller = new AbortController();
  const regular = serve();
  const source = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    if (url === referencePublication.href) { controller.abort(); options.signal.throwIfAborted(); }
    return source(url, options);
  };
  await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, controller.signal, referencePublication), { name: "AbortError" });
  assert.equal(regular.length, 2);
});

test("authenticated but semantically mismatched references are verification failures, not absent editions", async () => {
  const altered = JSON.parse(referenceBytes);
  altered.entries[0].reference.cohort[0].first_candidate_id = "not-the-named-contender";
  const result = Buffer.from(JSON.stringify(altered));
  serve(registry, bytes, result);
  const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal,
    { ...referencePublication, sha256: createHash("sha256").update(result).digest("hex") });
  assert.equal(value.forecast.run.artifactSha256, entry.sha256);
  assert.equal(value.reference, null);
  assert.match(value.referenceError, /could not be verified/);
});

test("cancellation during non-abortable reference hashing does not fulfill a stale result", async () => {
  serve();
  const controller = new AbortController();
  const originalDigest = crypto.subtle.digest;
  let calls = 0;
  crypto.subtle.digest = async function (...args) {
    const value = await originalDigest.apply(this, args);
    if (++calls === 2) controller.abort();
    return value;
  };
  try {
    await assert.rejects(loadElectionNightForecast({ chamber: "senate", runId: null }, controller.signal, referencePublication), { name: "AbortError" });
    assert.equal(calls, 2);
  } finally { crypto.subtle.digest = originalDigest; }
});

test("an optional reference deadline preserves the verified forecast instead of blocking forever", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  serve();
  const regular = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    if (url === referencePublication.href) {
      t.mock.timers.tick(8000);
      options.signal.throwIfAborted();
    }
    return regular(url, options);
  };
  const value = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal, referencePublication);
  assert.equal(value.forecast.run.artifactSha256, entry.sha256);
  assert.equal(value.reference, null);
  assert.match(value.referenceError, /could not be verified/);
});
