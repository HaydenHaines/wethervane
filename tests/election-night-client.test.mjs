import test, { afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { loadElectionNightForecast, loadPublishedElectionNightForecast } from "../lib/election-night-client.ts";
import { parseElectionNightPublicationIndex } from "../lib/election-night-publication.ts";

const root = new URL("../", import.meta.url);
const registryBytes = readFileSync(new URL("public/data/explorer/registry.json", root));
const registry = JSON.parse(registryBytes);
const entry = registry.runs.find(run => run.id === registry.defaults["2026"]);
const bytes = readFileSync(new URL(`public${entry.href}`, root));
const referencePublication = JSON.parse(readFileSync(new URL("data/election-night-reference-publication.json", root)));
const referenceBytes = readFileSync(new URL(`public${referencePublication.href}`, root));
const corePublication = JSON.parse(readFileSync(new URL("data/election-night-core-reference-publication.json", root)));
const coreBytes = readFileSync(new URL(`public${corePublication.href}`, root));
const runtimeIndex = JSON.parse(readFileSync(new URL("public/data/election-night/publication-index.json", root)));
const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });
function serve(list = registry, result = bytes, reference = referenceBytes, core = coreBytes) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.ok(options.signal instanceof AbortSignal);
    return new Response(url === "/data/explorer/registry.json" ? JSON.stringify(list) : url === referencePublication.href ? reference : url === corePublication.href ? core : result);
  };
  return calls;
}
const sha = body => createHash("sha256").update(body).digest("hex");
function servePublished({ list = registry, result = bytes, reference = referenceBytes,
  core = coreBytes, index = runtimeIndex } = {}) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push(url);
    options.signal.throwIfAborted();
    const payload = url === "/data/explorer/registry.json" ? JSON.stringify(list)
      : url === "/data/election-night/publication-index.json" ? JSON.stringify(index)
      : url === referencePublication.href ? reference
      : url === corePublication.href ? core : result;
    return new Response(payload);
  };
  return calls;
}

test("runtime publication follows same run ID with a revised public SHA and exact new references", async () => {
  // Synthetic republication: same reviewed native rows, new public edition date,
  // separately published catalog bytes and their runtime qualification decision.
  const list = structuredClone(registry), next = list.runs.find(run => run.id === entry.id);
  const result = JSON.parse(bytes);
  const cutoff = "2026-10-27T12:00:00Z", issued = "2026-10-27T12:05:00Z";
  for (const run of [next, result.run]) {
    run.dateId = "2026-10-27"; run.dateLabel = "October 27, 2026";
    run.forecastAt = cutoff; run.cutoffAt = cutoff; run.issuedAt = issued;
    run.components.base = "recent_calendar_history";
    run.components.uncertainty = "reviewed_extended";
    run.inputCutoffs.find(row => row.label === "Polling inputs").at = cutoff;
  }
  const resultBytes = Buffer.from(JSON.stringify(result)); next.sha256 = sha(resultBytes);
  const full = JSON.parse(referenceBytes), fullEntry = full.entries.find(row => row.runId === entry.id);
  fullEntry.resultSha256 = next.sha256; fullEntry.forecastCutoffAt = cutoff; fullEntry.forecastIssuedAt = issued;
  const fullBytes = Buffer.from(JSON.stringify(full));
  const core = JSON.parse(coreBytes), coreEntry = core.entries.find(row => row.runId === entry.id);
  coreEntry.resultSha256 = next.sha256; coreEntry.forecastCutoffAt = cutoff; coreEntry.forecastIssuedAt = issued;
  coreEntry.reference.edition = "2026-10-27"; coreEntry.reference.input_cutoff_utc = cutoff;
  const coreResultBytes = Buffer.from(JSON.stringify(core));
  const index = structuredClone(runtimeIndex);
  index.full.sha256 = sha(fullBytes); index.primary.sha256 = sha(coreResultBytes);
  for (const publication of [index.full, index.primary]) {
    const qualified = publication.editions.find(row => row.runId === entry.id);
    qualified.resultSha256 = next.sha256;
    qualified.qualification.forecastIssuedAt = issued;
    qualified.qualification.forecastCutoffAt = cutoff;
    qualified.qualification.configuration = { base: "recent_calendar_history", polling: "standard", uncertainty: "reviewed_extended" };
    qualified.qualification.strictAsOf = false;
  }
  const staleCalls = servePublished({ list, result: resultBytes });
  const stale = await loadPublishedElectionNightForecast({ chamber: "senate", runId: entry.id }, new AbortController().signal);
  assert.equal(stale.forecast.run.artifactSha256, next.sha256);
  assert.equal(stale.reference, null); assert.equal(stale.primaryReference, null);
  assert.equal(stale.referenceError, null); assert.equal(stale.primaryReferenceError, null);
  assert.ok(!staleCalls.includes(referencePublication.href));
  const calls = servePublished({ list, result: resultBytes, reference: fullBytes, core: coreResultBytes, index });
  const fresh = await loadPublishedElectionNightForecast({ chamber: "senate", runId: entry.id }, new AbortController().signal);
  assert.equal(fresh.forecast.run.artifactSha256, next.sha256);
  assert.equal(fresh.reference?.resultSha256, next.sha256);
  assert.equal(fresh.primaryReference?.resultSha256, next.sha256);
  assert.equal(fresh.reference?.forecastCutoffAt, cutoff);
  assert.equal(fresh.primaryReference?.forecastCutoffAt, cutoff);
  assert.equal(fresh.reference?.perRace.length, 35);
  assert.equal(fresh.primaryReference?.perRace.length, 32);
  assert.equal(fresh.referenceError, null); assert.equal(fresh.primaryReferenceError, null);
  assert.ok(calls.includes(referencePublication.href)); assert.ok(calls.includes(corePublication.href));
});

test("runtime publication rejects missing role metadata, and optional malformed core manifest stays optional", async () => {
  for (const mutate of [
    value => { delete value.full.editions[0].qualification.structuralCutoffAt; },
    value => { delete value.primary.editions[1].qualification.cohortId; },
    value => { delete value.primary.editions[1].qualification.sourcePins.policySha256; },
    value => { value.full.editions[0].qualification.configuration = null; },
  ]) {
    const changed = structuredClone(runtimeIndex); mutate(changed);
    assert.equal(parseElectionNightPublicationIndex(changed), null);
  }
  serve();
  const loaded = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal,
    undefined, { ...corePublication, editions: null });
  assert.equal(loaded.forecast.run.artifactSha256, entry.sha256);
  assert.equal(loaded.primaryReference, null);
  assert.match(loaded.primaryReferenceError, /could not be verified/);
});

test("a newly qualified fixed32 policy/mapping pin is read from publication, not an old code constant", async () => {
  const catalog = JSON.parse(coreBytes), row = catalog.entries.find(item => item.runId === entry.id);
  const previousPolicy = row.sourcePins.policySha256, previousMapping = row.sourcePins.mappingSha256;
  row.sourcePins.policySha256 = "a".repeat(64); row.sourcePins.mappingSha256 = "b".repeat(64);
  row.reference.sources.find(source => source.sha256 === previousPolicy).sha256 = row.sourcePins.policySha256;
  row.reference.sources.find(source => source.sha256 === previousMapping).sha256 = row.sourcePins.mappingSha256;
  const core = Buffer.from(JSON.stringify(catalog));
  const index = structuredClone(runtimeIndex);
  index.primary.sha256 = sha(core);
  index.primary.editions.find(item => item.runId === entry.id).qualification.sourcePins = structuredClone(row.sourcePins);
  servePublished({ core, index });
  const loaded = await loadPublishedElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal);
  assert.equal(loaded.primaryReference?.resultSha256, entry.sha256);
  assert.equal(loaded.primaryReference?.sourcePins.policySha256, "a".repeat(64));
  assert.equal(loaded.primaryReference?.sourcePins.mappingSha256, "b".repeat(64));
  assert.equal(loaded.primaryReferenceError, null);
});

test("pre-aborted published load never starts an independent reference-index fetch", async () => {
  const calls = servePublished();
  const controller = new AbortController(); controller.abort();
  await assert.rejects(loadPublishedElectionNightForecast({ chamber: "senate", runId: null }, controller.signal),
    { name: "AbortError" });
  assert.ok(!calls.includes("/data/election-night/publication-index.json"));
});

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


test("qualified fixed32 and existing35 references attach independently to the actual FP edition", async () => {
  const calls = serve();
  const loaded = await loadElectionNightForecast({ chamber: "senate", runId: null }, new AbortController().signal, referencePublication, corePublication);
  assert.equal(loaded.forecast.races.length, 35); assert.equal(loaded.reference.perRace.length, 35); assert.equal(loaded.primaryReference.perRace.length, 32);
  assert.equal(loaded.referenceError, null); assert.equal(loaded.primaryReferenceError, null);
  assert.equal(loaded.primaryReference.summary.meanAbsoluteMarginError.mean, 5.725521323266725);
  assert.equal(loaded.reference.summary.meanAbsoluteMarginError.mean, 5.621801760616599);
  assert.deepEqual(calls.map(c=>c.url), ["/data/explorer/registry.json",entry.href,referencePublication.href,corePublication.href]);
  assert.notEqual(calls[2].options.signal,calls[3].options.signal);
});

test("fixed32 F reference preserves its October20 edition; ineligible saved choices have no error or fetch", async () => {
  for(const id of ["current-fundamentals","current-cycle-low"]){
    const e=registry.runs.find(e=>e.id===id), calls=serve(registry,readFileSync(new URL(`public${e.href}`,root)));
    const loaded=await loadElectionNightForecast({chamber:"senate",runId:id},new AbortController().signal,referencePublication,corePublication);
    assert.equal(loaded.reference,null);assert.equal(loaded.referenceError,null);assert.equal(loaded.primaryReferenceError,null);
    if(id==="current-fundamentals"){
      assert.equal(loaded.primaryReference.forecastCutoffAt,"2026-10-20T15:21:23Z");assert.equal(loaded.primaryReference.drawIdentity.rowCount,100000);
      assert.deepEqual(calls.map(c=>c.url),["/data/explorer/registry.json",e.href,corePublication.href]);
    }else{assert.equal(loaded.primaryReference,null);assert.equal(calls.length,2);}
  }
});

test("fixed32 verification failures and fresh retry cannot suppress or substitute the existing35 reference", async () => {
  serve(registry,bytes,referenceBytes,Buffer.concat([coreBytes,Buffer.from(" ")]));
  let loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,referencePublication,corePublication);
  assert.ok(loaded.reference);assert.equal(loaded.referenceError,null);assert.equal(loaded.primaryReference,null);assert.match(loaded.primaryReferenceError,/32-race.*could not be verified/);
  serve(registry,bytes,Buffer.from("invalid35"),coreBytes);
  loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,referencePublication,corePublication);
  assert.equal(loaded.reference,null);assert.match(loaded.referenceError,/could not be verified/);assert.ok(loaded.primaryReference);assert.equal(loaded.primaryReferenceError,null);
  serve();loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,referencePublication,corePublication);
  assert.ok(loaded.reference);assert.ok(loaded.primaryReference);assert.equal(loaded.referenceError,null);assert.equal(loaded.primaryReferenceError,null);
});

test("authenticated fixed32 bytes with a changed candidate mapping remain a visible optional verification failure", async () => {
  const changed=JSON.parse(coreBytes);changed.entries[1].reference.cohort[0].first_candidate.id="INVENTED wrong candidate";
  const body=Buffer.from(JSON.stringify(changed));serve(registry,bytes,referenceBytes,body);
  const loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,referencePublication,
    {...corePublication,sha256:createHash("sha256").update(body).digest("hex")});
  assert.ok(loaded.reference);assert.equal(loaded.primaryReference,null);assert.match(loaded.primaryReferenceError,/could not be verified/);
});

test("fixed32 publication paths and duplicate editions cannot cause arbitrary fetches", async () => {
  for(const p of [{...corePublication,href:"https://example.invalid/core.json"},{...corePublication,editions:[...corePublication.editions,corePublication.editions[1]]}]){
    const calls=serve();const loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,undefined,p);
    assert.equal(loaded.primaryReference,null);assert.match(loaded.primaryReferenceError,/could not be verified/);assert.equal(calls.length,2);
  }
  const calls=serve();const loaded=await loadElectionNightForecast({chamber:"senate",runId:null},new AbortController().signal,undefined,{...corePublication,editions:[]});
  assert.equal(loaded.primaryReference,null);assert.equal(loaded.primaryReferenceError,null);assert.equal(calls.length,2);
});

test("fixed32 has its own eight-second abort deadline and the authenticated forecast survives timeout", async t => {
  t.mock.timers.enable({apis:["setTimeout"]});serve();const regular=globalThis.fetch, parent=new AbortController();
  globalThis.fetch=async(url,options)=>{
    if(url===corePublication.href){assert.notEqual(options.signal,parent.signal);t.mock.timers.tick(8000);options.signal.throwIfAborted();}
    return regular(url,options);
  };
  const loaded=await loadElectionNightForecast({chamber:"senate",runId:null},parent.signal,undefined,corePublication);
  assert.equal(parent.signal.aborted,false);assert.equal(loaded.forecast.run.artifactSha256,entry.sha256);
  assert.equal(loaded.primaryReference,null);assert.match(loaded.primaryReferenceError,/could not be verified/);assert.equal(loaded.referenceError,null);
});

test("cancellation during fixed32 fetch or digest never fulfills a stale result", async () => {
  const controller=new AbortController();serve();const regular=globalThis.fetch;
  globalThis.fetch=async(url,options)=>{if(url===corePublication.href){controller.abort();options.signal.throwIfAborted();}return regular(url,options);};
  await assert.rejects(loadElectionNightForecast({chamber:"senate",runId:null},controller.signal,undefined,corePublication),{name:"AbortError"});
  serve();const duringDigest=new AbortController(),originalDigest=crypto.subtle.digest;let digests=0;
  crypto.subtle.digest=async function(...args){const result=await originalDigest.apply(this,args);if(++digests===2)duringDigest.abort();return result;};
  try{await assert.rejects(loadElectionNightForecast({chamber:"senate",runId:null},duringDigest.signal,undefined,corePublication),{name:"AbortError"});assert.equal(digests,2);}
  finally{crypto.subtle.digest=originalDigest;}
});
