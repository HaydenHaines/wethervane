#!/usr/bin/env node
/** Freeze only the source-bound count metadata and a narrow scope receipt. No execution. */
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { isAbsolute, join, relative, resolve } from "node:path";

const native = "research/senate-stack-integration-20261020/count-application-driver-astra-v1";
const expected = Object.freeze({
  "catalog.json": "4b94b8a9e3be9bfc8db638a8d72fad6ec7b720c9fce35aaba3285b0056471f18",
  "request.json": "77a56140a55a06524f54d8c157c950f978e06e312d6411b0001d21f87f9d40ff",
  "resolution.json": "5f6f187ca675d65840bbe0ff80cbd83726677b2c3c0ccb691d533a4d57e9a852",
  "bundle.json": "c816c32ad1bf4ec6832591c32b47f90c393d5be17393ad69a9a8ede580388a35",
  "registration.json": "256c73e8731bf7b297fb70c5a6a2512a78f5f4b5252f945ab391d99379af6393",
});
const planPin = { path: "research/senate-stack-integration-20261020/partial-affine-completion-root-v1/plan.json",
  sha256: "adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1" };
const problemPin = { path: "research/senate-stack-integration-20261020/actual-pilot-preparation-sol-v1/prepared-v4/problem.json",
  sha256: "96f890fae93fec40caa0b58fb67b7df264259fc533b5d7a23eab70c8a4dc7034" };
const preparationPin = { path: "research/senate-stack-integration-20261020/actual-pilot-preparation-sol-v1/prepared-v4/plan.json",
  sha256: "0b35a7054b8ec24b2269cf3e06bf37f95b70a6ab1553290d5253067c217af723" };
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const fail = message => { throw Error(message); };

async function pinned(root, path, digest, limit) {
  if (!path || isAbsolute(path) || path.split("/").some(part => part === "" || part === "." || part === "..")) fail("Unsafe source path.");
  const base = await realpath(root), full = resolve(base, path), file = await realpath(full);
  const inside = relative(base, file);
  if (!inside || inside.startsWith("..") || isAbsolute(inside) || (await lstat(full)).isSymbolicLink()) fail("Source path escapes the owner root or is a symlink.");
  const stat = await lstat(file);
  if (!stat.isFile() || stat.size > limit) fail("Source metadata is missing or too large.");
  const bytes = await readFile(file);
  if (bytes.length !== stat.size || sha(bytes) !== digest) fail(`Source pin mismatch: ${path}`);
  return bytes;
}

export async function importCountCatalog(ownerRoot, outputDirectory) {
  if (!isAbsolute(ownerRoot) || !isAbsolute(outputDirectory)) fail("Use absolute owner and output directories.");
  const source = {}, parsed = {};
  for (const [name, digest] of Object.entries(expected)) {
    source[name] = await pinned(ownerRoot, `${native}/${name}`, digest, 100_000);
    parsed[name] = JSON.parse(source[name].toString("utf8"));
  }
  const catalog = parsed["catalog.json"], request = parsed["request.json"], resolution = parsed["resolution.json"],
    bundle = parsed["bundle.json"], registration = parsed["registration.json"];
  if (catalog.schema_version !== "composition-catalog-v1" || catalog.roles?.length !== 8 ||
    catalog.saved_presets?.length !== 0 || catalog.default_preset_id !== null || catalog.scheduler_available !== false ||
    !same(catalog.executable_targets, ["population_cut_v1"]) || request.schema_version !== "composition-request-v1" ||
    request.target_id !== "population_cut_v1" || request.information_cutoff !== "2026-10-20T22:13:33Z" ||
    request.output_contract_id !== "whole_parent_joint_candidate_counts_v1" || resolution.schema_version !== "composition-resolution-v1" ||
    resolution.status !== "requires_computation" || resolution.artifact !== null || resolution.record !== null || resolution.summary !== null ||
    !same(resolution.configuration?.request, request) || request.context?.execution_registration !== bundle.registration_id ||
    bundle.registration_id !== registration.registration_id || !same(bundle.scientific_plan, planPin) || !same(registration.plan, planPin) ||
    !same(resolution.configuration?.scientific_plan, planPin)) fail("Frozen metadata bindings changed.");
  const plan = JSON.parse((await pinned(ownerRoot, planPin.path, planPin.sha256, 100_000)).toString("utf8"));
  if (plan.schema !== "actual_TX_role_execution_plan_v2" || plan.purpose !== "actual_current" ||
    plan.cutoff_utc !== request.information_cutoff || plan.problem_identity !== resolution.configuration.support_id ||
    !same(plan.source_inputs?.problem, problemPin) || !same(plan.source_inputs?.prepared_plan, preparationPin) ||
    plan.provider?.eight_roles_independently_swappable !== false ||
    plan.forecast_qualified !== false || plan.publication_authorized !== false) fail("Frozen plan scope changed.");
  const preparation = JSON.parse((await pinned(ownerRoot, preparationPin.path, preparationPin.sha256, 150_000)).toString("utf8"));
  if (preparation.schema !== "ActualTXUpstreamPreparationPlanV2" ||
    preparation.scope !== "fixed_roster_conditional_research" || preparation.cutoff_utc !== request.information_cutoff ||
    !same(preparation.problem, problemPin) || preparation.publication_legal_support_qualified !== false)
    fail("Pinned prepared-plan scope changed.");
  const problem = JSON.parse((await pinned(ownerRoot, problemPin.path, problemPin.sha256, 3_000_000)).toString("utf8"));
  const events = problem.events?.map(({ cycle, date, role, state }) => ({ cycle, date, role, state }));
  const contests = problem.contests?.map(({ id, event_id, kind }) => ({ id, event_id, kind }));
  const expectedEvents = [
    { cycle: 2020, date: "2020-11-03", role: "training", state: "48" },
    { cycle: 2024, date: "2024-11-05", role: "training", state: "48" },
    { cycle: 2026, date: "2026-11-03", role: "prediction", state: "48" },
  ];
  if (problem.schema !== "senate_count_joint_problem_v1" || problem.purpose !== "actual_current" ||
    problem.cutoff_utc !== request.information_cutoff || !same(problem.states, ["48"]) ||
    !same(events, expectedEvents) || !same(contests, [
      { id: "contest_tx2020_regular", event_id: "event_tx2020_general", kind: "regular" },
      { id: "contest_tx2024_regular", event_id: "event_tx2024_general", kind: "regular" },
      { id: "contest_tx2026_regular", event_id: "event_tx2026_general", kind: "regular" },
    ])) fail("Pinned Texas event/contest scope changed.");
  const scope = {
    schema: "CountCatalogScopeReceiptV1", sourcePlanSha256: planPin.sha256,
    sourceProblemSha256: problemPin.sha256, state: "Texas", stateFips: "48",
    kind: "fixed_roster_conditional_research", trainingCycles: [2020, 2024],
    predictionCycle: 2026, predictionElectionDate: "2026-11-03",
    predictionContestIds: ["contest_tx2026_regular"],
  };
  const scopeBytes = Buffer.from(`${JSON.stringify(scope, null, 2)}\n`);
  await mkdir(outputDirectory, { recursive: false });
  for (const [name, bytes] of Object.entries(source)) await writeFile(join(outputDirectory, name), bytes, { flag: "wx" });
  await writeFile(join(outputDirectory, "scope.json"), scopeBytes, { flag: "wx" });
  return { staged: Object.fromEntries(Object.entries(source).map(([name, bytes]) => [name, sha(bytes)])),
    scopeSha256: sha(scopeBytes), outputDirectory };
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(new URL(import.meta.url).pathname)) {
  const args = process.argv.slice(2);
  if (args.length !== 4 || args[0] !== "--owner-root" || args[2] !== "--out") {
    console.error("Usage: node scripts/import_count_catalog.mjs --owner-root ABSOLUTE --out ABSOLUTE_NEW_DIRECTORY");
    process.exitCode = 2;
  } else {
    importCountCatalog(args[1], args[3]).then(result => process.stdout.write(`${JSON.stringify(result)}\n`), error => {
      console.error(`Count catalog import failed: ${error.message}`);
      process.exitCode = 1;
    });
  }
}
