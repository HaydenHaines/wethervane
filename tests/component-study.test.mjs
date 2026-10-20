import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { cp, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { canonicalStudySelections, studySelectionsMatch, validateComponentStudyResult } from "../lib/component-study.ts";
import { COMPONENT_STUDY_PINS, ComponentStudyInputError, loadComponentStudies, resolveComponentStudy } from "../lib/component-study-server.ts";

const deployment = new URL("../", import.meta.url);
const { catalog, results, resultBytes } = await loadComponentStudies(new URL("data", deployment).pathname);
const clone = value => structuredClone(value);
const digest = value => createHash("sha256").update(value).digest("hex");
const selection = (selections, id) => selections.find(value => value.instance_id === id);

test("the admitted catalog contains exactly two scoped, independently identified ready results", () => {
  assert.equal(catalog.catalog.activation, "metadata_only");
  assert.equal(catalog.executionAvailable, false);
  assert.equal(catalog.scheduled, false);
  assert.deepEqual(catalog.studies.map(study => study.id), ["ordinary-2018-f", "ordinary-2018-fp"]);
  for (const result of results) {
    assert.ok(validateComponentStudyResult(result));
    const study = catalog.studies.find(study => study.id === result.id);
    assert.equal(digest(resultBytes[result.id]), study.sha256);
    assert.equal(result.races.length, 30);
    assert.equal(result.status.evidence, "unassessed");
    assert.deepEqual(result.capabilities, { raceProbabilities: true, conditionalTwoPartyShares: true, chamberControl: false, tipping: false, counts: false });
  }
  assert.notEqual(results[0].sourceFingerprints.scientificPlan, results[1].sourceFingerprints.scientificPlan);
  assert.equal(results[0].sourceFingerprints.dataBundle, results[1].sourceFingerprints.dataBundle);
  assert.equal(JSON.stringify(catalog).includes('"file"'), false);
  assert.equal(JSON.stringify(catalog).includes("/home/"), false);
  assert.equal(JSON.stringify(catalog).includes('"scientific_plan"'), false);
});

test("exact saved requests and declared implicit defaults resolve to their original identities", () => {
  for (const study of catalog.studies) {
    const selections = clone(study.selections).reverse();
    selection(selections, "candidate").options.candidate = "signed_incumbency";
    const resolved = resolveComponentStudy(catalog, { studyId: study.id, selections });
    assert.equal(resolved.state, "ready");
    assert.equal(resolved.resultId, study.id);
    assert.equal(resolved.resultSha256, study.sha256);
    assert.equal(resolved.configurationSha256, study.configurationSha256);
    assert.equal(resolved.scheduled, false);
    assert.ok(studySelectionsMatch(catalog.catalog, selections, study.selections));
  }
  const changed = clone(catalog.studies[0].selections);
  selection(changed, "context").options = {}; // Default party_midterm is not the example's explicit off.
  assert.equal(studySelectionsMatch(catalog.catalog, changed, catalog.studies[0].selections), false);
  assert.equal(resolveComponentStudy(catalog, { studyId: catalog.studies[0].id, selections: changed }).state, "incompatible");
});

test("only the exact candidate-off boundaries claim requires_computation without results or scheduled work", () => {
  for (const study of catalog.studies) {
    const selections = clone(study.selections);
    selection(selections, "candidate").options = { candidate: "off" };
    const resolved = resolveComponentStudy(catalog, { studyId: study.id, selections });
    assert.equal(resolved.state, "requires_computation");
    assert.equal(resolved.resultId, null);
    assert.equal(resolved.resultSha256, null);
    assert.equal(resolved.configurationSha256, null);
    assert.equal(resolved.executionAvailable, false);
    assert.equal(resolved.scheduled, false);
    assert.match(resolved.reason, /No computation has been scheduled/);
    selection(selections, "structure").options.state_slope = "fixed_one";
    assert.equal(resolveComponentStudy(catalog, { studyId: study.id, selections }).state, "incompatible");
  }
});

test("other valid enum drafts and crossed F/FP bindings stay explicitly unsupported by this deployment", () => {
  for (const [instance, option, value] of [["structure", "state_slope", "fixed_one"], ["carryover", "carryover", "off"], ["context", "context", "party"]]) {
    const study = catalog.studies[0], selections = clone(study.selections);
    selection(selections, instance).options[option] = value;
    const resolved = resolveComponentStudy(catalog, { studyId: study.id, selections });
    assert.equal(resolved.state, "incompatible");
    assert.equal(resolved.resultId, null);
    assert.match(resolved.reason, /does not establish scientific impossibility/);
  }
  const crossed = resolveComponentStudy(catalog, { studyId: catalog.studies[0].id, selections: catalog.studies[1].selections });
  assert.equal(crossed.state, "incompatible");
  assert.equal(crossed.resultId, null);
});

test("malformed and injected selections are rejected rather than guessed or rebound", () => {
  for (const mutate of [
    value => { value.studyId = "unknown"; },
    value => { value.data_bundle = { path: "/browser/chosen" }; },
    value => { value.selections.pop(); },
    value => { value.selections[1] = value.selections[0]; },
    value => { value.selections[0].provider_id = "unknown-provider"; },
    value => { value.selections[0].version = "v2"; },
    value => { value.selections[0].options.unknown = "on"; },
    value => { value.selections[0].options.state_slope = "invented"; },
    value => { value.selections[0].options = []; },
    value => { value.selections[0].path = "/injected"; },
  ]) {
    const value = { studyId: catalog.studies[0].id, selections: clone(catalog.studies[0].selections) }; mutate(value);
    assert.throws(() => resolveComponentStudy(catalog, value), ComponentStudyInputError);
  }
  assert.equal(canonicalStudySelections(catalog.catalog, null), null);
});

test("result validation rejects false scope, interval, probability, identity and observed-count claims", () => {
  for (const mutate of [
    value => { value.capabilities.chamberControl = true; },
    value => { value.status.evidence = "prospective"; },
    value => { value.strictAsOf = true; },
    value => { value.contextId = "ordinary-senate-2018-fp"; },
    value => { value.races.pop(); },
    value => { value.races[1].id = value.races[0].id; },
    value => { value.races[0].candidates[0].winProbability = NaN; },
    value => { value.races[0].candidates[0].share.quantiles[0].level = 0.05; },
    value => { value.races[0].candidates[0].share.mean = 0.9; },
    value => { value.races[0].candidates[0].share.denominator = "All ballots"; },
    value => { value.races[0].observed.twoPartyVotes++; },
    value => { value.races[0].observed.democraticTwoPartyShare = 0.1; },
    value => { value.selections[1] = value.selections[0]; },
    value => { delete value.sourceFingerprints.scientificPlan; },
  ]) {
    const value = clone(results[0]); mutate(value);
    assert.equal(validateComponentStudyResult(value), false);
  }
  for (const value of [null, [], {}, "bad"]) assert.equal(validateComponentStudyResult(value), false);
});

async function temporaryPacket(fn) {
  const directory = await mkdtemp(join(tmpdir(), "component-study-"));
  try {
    await cp(new URL("data/component-results", deployment), join(directory, "component-results"), { recursive: true });
    await cp(new URL("data/model-components-v2.json", deployment), join(directory, "model-components-v2.json"));
    await fn(directory);
  } finally { await rm(directory, { recursive: true, force: true }); }
}
test("activation fails closed when catalog, index or any advertised result changes", async () => {
  for (const filename of ["model-components-v2.json", "component-results/index.json", "component-results/fp.json"]) {
    await temporaryPacket(async directory => {
      const path = join(directory, filename);
      await writeFile(path, (await readFile(path)).toString() + "\n");
      await assert.rejects(loadComponentStudies(directory), /fingerprint mismatch/);
    });
  }
  await temporaryPacket(async directory => {
    const path = join(directory, "component-results/f.json"), outside = join(directory, "outside.json");
    await writeFile(outside, await readFile(path)); await rm(path); await symlink(outside, path);
    await assert.rejects(loadComponentStudies(directory), /leaves its publication directory/);
  });
});

test("HTTP adapter returns exact result bytes and separates 400 from 503 lifecycle errors", async () => {
  const route = new URL("../app/api/component-studies/route.ts", import.meta.url).href;
  const script = `
    import {GET,POST} from ${JSON.stringify(route)};
    import {createHash} from 'node:crypto';
    console.error=()=>{};
    const base='http://example.invalid/api/component-studies';
    const list=await GET(new Request(base)), catalog=await list.json();
    const entry=catalog.studies[0];
    const result=await GET(new Request(base+'?result='+entry.id));
    const bytes=Buffer.from(await result.arrayBuffer());
    const ready=await POST(new Request(base,{method:'POST',body:JSON.stringify({studyId:entry.id,selections:entry.selections})}));
    const bad=await GET(new Request(base+'?result=unknown'));
    const injected=await POST(new Request(base,{method:'POST',body:JSON.stringify({studyId:entry.id,selections:entry.selections,bindings:{}})}));
    const oversized=await POST(new Request(base,{method:'POST',body:' '.repeat(32001)}));
    console.log(JSON.stringify({list:list.status,result:result.status,digest:createHash('sha256').update(bytes).digest('hex'),expected:entry.sha256,ready:await ready.json(),unknown:bad.status,injected:injected.status,oversized:oversized.status}));
  `;
  const output = JSON.parse(execFileSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", script], { cwd: new URL("../", import.meta.url), encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }));
  assert.equal(output.list, 200); assert.equal(output.result, 200); assert.equal(output.digest, output.expected);
  assert.equal(output.ready.state, "ready"); assert.equal(output.unknown, 400); assert.equal(output.injected, 400); assert.equal(output.oversized, 400);
  await temporaryPacket(async directory => {
    const invalid = JSON.parse(execFileSync(process.execPath, ["--experimental-strip-types", "--input-type=module", "-e", `
      import {GET} from ${JSON.stringify(route)};
      console.error=()=>{};
      const response=await GET(new Request('http://example.invalid/api/component-studies'));
      console.log(JSON.stringify({status:response.status,body:await response.json()}));
    `], { cwd: directory, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }));
    assert.equal(invalid.status, 503);
    assert.deepEqual(Object.keys(invalid.body), ["error"]);
    assert.equal(JSON.stringify(invalid.body).includes(directory), false);
  });
  assert.match(COMPONENT_STUDY_PINS.index, /^[a-f0-9]{64}$/);
});
