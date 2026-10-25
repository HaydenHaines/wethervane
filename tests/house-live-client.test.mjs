import test from "node:test";
import assert from "node:assert/strict";
import { loadHouseLive } from "../lib/house-live-client.ts";
import { makeHouseLivePublication } from "./fixtures/house-live-publication.mjs";

async function withResponse(response, check) {
  const previous = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return response; };
  try { await check(calls); } finally { globalThis.fetch = previous; }
}

test("default and explicit model loading preserve the requested identity and abort signal", async () => {
  const value = makeHouseLivePublication(), controller = new AbortController();
  await withResponse(Response.json(value), async calls => {
    const loaded = await loadHouseLive({ runId: null }, controller.signal);
    assert.equal(loaded.view.run.id, value.catalog.defaultId);
    assert.equal(calls[0].url, "/api/chambers/house/live");
    assert.equal(calls[0].options.signal, controller.signal);
    assert.equal(calls[0].options.cache, "no-store");
  });
  await withResponse(Response.json(value), async calls => {
    const loaded = await loadHouseLive({ runId: value.view.run.id, expectedResultSha256: value.view.run.resultSha256 }, controller.signal);
    assert.equal(loaded.view.run.resultSha256, value.view.run.resultSha256);
    assert.match(calls[0].url, /\?run=house-current-polling$/);
  });
});

test("selection failures never fall back to another result or the historical endpoint", async () => {
  const value = makeHouseLivePublication(), signal = new AbortController().signal;
  for (const response of [new Response(null, { status: 503 }), new Response(null, { status: 400 }), Response.json({ status: "available" })]) {
    await withResponse(response, async calls => {
      await assert.rejects(loadHouseLive({ runId: value.view.run.id }, signal));
      assert.equal(calls.length, 1);
      assert.match(calls[0].url, /^\/api\/chambers\/house\/live/);
    });
  }
  await withResponse(Response.json(value), async () => {
    await assert.rejects(loadHouseLive({ runId: value.catalog.entries[0].id }, signal), /does not match/);
  });
  await withResponse(Response.json(value), async () => {
    await assert.rejects(loadHouseLive({ runId: value.view.run.id, expectedResultSha256: "f".repeat(64) }, signal), /changed while loading/);
  });
});

test("invalid requests never cause a fetch", async () => {
  await withResponse(Response.json({}), async calls => {
    for (const request of [{ runId: "../../private" }, { runId: "" }, { runId: null, expectedResultSha256: "a".repeat(64) }, { runId: "house-current-polling", expectedResultSha256: "bad" }])
      await assert.rejects(loadHouseLive(request, new AbortController().signal));
    assert.equal(calls.length, 0);
  });
});
