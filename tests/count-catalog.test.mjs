import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, symlink, writeFile, copyFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { COUNT_ROLES, parseCountCatalogView, resolveCountCatalogSelection } from "../lib/count-catalog.ts";
import { COUNT_CATALOG_SOURCE_PINS, loadCountCatalogView } from "../lib/count-catalog-server.ts";
import { importCountCatalog } from "../scripts/import_count_catalog.mjs";

const staged = resolve("data/count-catalog");
const owner = process.env.BALLOTMANCY_COUNT_CATALOG_OWNER_ROOT;
const names = Object.keys(COUNT_CATALOG_SOURCE_PINS);

async function tempStage(t) {
  const root = await mkdtemp(join(tmpdir(), "count-catalog-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const directory = join(root, "stage");
  await mkdir(directory);
  await Promise.all(names.map(name => copyFile(join(staged, name), join(directory, name))));
  return directory;
}

test("frozen native handoff projects eight research roles and only an uncomputed preparation", async () => {
  const view = await loadCountCatalogView(staged);
  assert.equal(view.schema, "CountCatalogViewV1");
  assert.deepEqual(view.roles.map(row => row.role), COUNT_ROLES);
  assert.equal(view.scope.stateFips, "48");
  assert.deepEqual(view.scope.trainingCycles, [2020, 2024]);
  assert.equal(view.scope.predictionElectionDate, "2026-11-03");
  assert.equal(view.roles.find(row => row.role === "choice").providers[0].providerId,
    "candidate_TX_ballot_role_profile_choice_v1");
  assert.equal(view.prepared.status, "requires_computation");
  assert.equal(view.prepared.independentRoleSwaps, false);
  assert.equal(view.prepared.configurationSha256,
    "bc3c7e317f9c8b938b1df0cb67eb2cfa948c5c5c5d84264fbabbb38f3662d460");
  assert.deepEqual(view.prepared.selections.map(row => row.instanceId).sort(), [...COUNT_ROLES].sort());
  assert.equal(view.statuses.inferenceEligibility, "not_established_for_current_execution");
  assert.equal(view.statuses.savedResult, "none_registered");
  assert.equal(view.statuses.scheduler, "unavailable");
  assert.equal(view.statuses.empiricalQualification, "unassessed");
  assert.equal(view.statuses.publication, "unavailable");
  assert.equal(JSON.stringify(view).includes("research/senate-stack"), false);
  assert.equal(JSON.stringify(view).includes("/home/hayden"), false);
  assert.deepEqual(parseCountCatalogView(view), view);
});

test("only exact whole-stack selection resolves; changes do not become saved or ready", async () => {
  const view = await loadCountCatalogView(staged);
  const draft = { targetId: view.targetId, scopeSourceProblemSha256: view.scope.sourceProblemSha256,
    roles: structuredClone(view.prepared.selections) };
  const exact = resolveCountCatalogSelection(view, draft);
  assert.equal(exact.state, "requires_computation");
  assert.equal(exact.configurationSha256, view.prepared.configurationSha256);
  assert.equal(resolveCountCatalogSelection(view, { ...draft, roles: [...draft.roles].reverse() }).state, "requires_computation");
  const mutated = structuredClone(draft);
  mutated.roles.find(row => row.instanceId === "choice").providerId = "invented_choice_v1";
  assert.equal(resolveCountCatalogSelection(view, mutated).state, "unsupported");
  mutated.roles = structuredClone(draft.roles);
  mutated.roles[0].options = { unauthorized: "on" };
  assert.equal(resolveCountCatalogSelection(view, mutated).state, "unsupported");
  assert.equal(resolveCountCatalogSelection(view, { ...draft, roles: draft.roles.slice(1) }).state, "unsupported");
  assert.equal(resolveCountCatalogSelection(view, { ...draft, targetId: "senate_aggregate_logit_empirical_mixture_v1" }).state, "unsupported");
  assert.equal(resolveCountCatalogSelection(view, { ...draft, scopeSourceProblemSha256: "0".repeat(64) }).state, "unsupported");
  const stale = structuredClone(view);
  stale.prepared.configurationSha256 = "0".repeat(64);
  assert.equal(resolveCountCatalogSelection(stale, draft).state, "stale");
});

test("browser parser rejects private fields, incomplete roles and upgraded statuses", async () => {
  const original = await loadCountCatalogView(staged);
  const invalid = structuredClone(original);
  invalid.privateSourcePath = "/tmp/raw-fit";
  assert.equal(parseCountCatalogView(invalid), null);
  delete invalid.privateSourcePath;
  invalid.roles.pop();
  assert.equal(parseCountCatalogView(invalid), null);
  invalid.roles = structuredClone(original.roles);
  invalid.statuses.savedResult = "ready";
  assert.equal(parseCountCatalogView(invalid), null);
  invalid.statuses = structuredClone(original.statuses);
  invalid.prepared.status = "ready";
  assert.equal(parseCountCatalogView(invalid), null);
  invalid.prepared.status = "requires_computation";
  invalid.roles[0].providers[0].label = "https://unreviewed.invalid/source";
  assert.equal(parseCountCatalogView(invalid), null);
  assert.equal(original.roles[0].providers[0].label.includes("https:"), false);
});

test("private stage rejects byte drift and symlinked metadata", async t => {
  const directory = await tempStage(t);
  await writeFile(join(directory, "catalog.json"), Buffer.concat([await readFile(join(directory, "catalog.json")), Buffer.from(" ")]));
  await assert.rejects(loadCountCatalogView(directory), /fingerprint mismatch/);
  await copyFile(join(staged, "catalog.json"), join(directory, "catalog.json"));
  await rm(join(directory, "scope.json"));
  await symlink(join(staged, "scope.json"), join(directory, "scope.json"));
  await assert.rejects(loadCountCatalogView(directory), /escapes the private stage/);
});

test("offline import verifies pinned execution, preparation and problem scope without copying those inputs",
  { skip: !owner }, async t => {
  const root = await mkdtemp(join(tmpdir(), "count-catalog-import-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const out = join(root, "stage");
  const result = await importCountCatalog(owner, out);
  assert.equal(result.scopeSha256, COUNT_CATALOG_SOURCE_PINS["scope.json"]);
  assert.deepEqual((await loadCountCatalogView(out)).scope, (await loadCountCatalogView(staged)).scope);
  assert.deepEqual((await readFile(join(out, "catalog.json"))),
    await readFile(join(owner, "research/senate-stack-integration-20261020/count-application-driver-astra-v1/catalog.json")));
  await assert.rejects(importCountCatalog(owner, out), /EEXIST/);
  });
