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
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
function serve(list = registry, result = bytes) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.ok(options.signal instanceof AbortSignal);
    return new Response(url === "/data/explorer/registry.json" ? JSON.stringify(list) : result);
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
