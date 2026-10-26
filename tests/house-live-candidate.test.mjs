import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { appendFile, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { checkHouseLiveCandidate } from "../scripts/check_house_live_candidate.mjs";
import { makeHouseLiveResult } from "./fixtures/house-live-fixtures.mjs";
import { makeHouseLiveCombinedPlan } from "./fixtures/house-live-combined-plan.mjs";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");
async function put(root, path, value) {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 1)}\n\n`);
  await mkdir(dirname(join(root, path)), { recursive: true });
  await writeFile(join(root, path), bytes);
  return { path, sha256: digest(bytes) };
}
function fourRows() {
  const rows = [Array(435).fill(0), Array(435).fill(0), Array(435).fill(1), Array(435).fill(1)];
  rows[1][1] = rows[3][1] = 2;
  rows[2][2] = rows[3][2] = 0;
  return rows;
}
async function fixture(t, { kind = "original", mutatePlan, mutateReceipt, mutateResult } = {}) {
  const sourceRoot = await mkdtemp(join(tmpdir(), "house-live-offline-candidate-"));
  t.after(() => rm(sourceRoot, { recursive: true, force: true }));
  const result = makeHouseLiveResult();
  result.qualification.publication_status = "not_reviewed";
  result.qualification.numerical_status = "invented_native_status_preserved";
  await mkdir(join(sourceRoot, "private"));
  const archivePath = "private/candidate-joint.npz";
  const written = spawnSync("python3", [fileURLToPath(new URL("./fixtures/house-live-joint-archive.py", import.meta.url))], {
    input: JSON.stringify({ destination: join(sourceRoot, archivePath), raceIds: result.joint_draws.ordered_race_ids,
      rows: fourRows(), chains: 2, ...(kind === "combined" ? { sourceRunLengths: [2, 2] } : {}) }), encoding: "utf8",
  });
  assert.equal(written.status, 0, written.stderr);
  const archiveSha = digest(await readFile(join(sourceRoot, archivePath)));
  Object.assign(result.joint_draws, { path: archivePath, sha256: archiveSha });
  result.chamber.joint_draws_sha256 = archiveSha;
  const plan = makeHouseLiveCombinedPlan(result);
  if (kind === "original") {
    plan.schema = "HouseLivePredictionPlanV1";
    plan.settings = { chains: 2, draws: 2, warmup: 1, seed: 2000 };
    delete plan.source_runs;
    delete plan.combination;
  }
  mutatePlan?.(plan);
  const planPin = await put(sourceRoot, "local-evidence/frozen-law.json", plan);
  const receipt = { schema: "HouseLivePredictionReceiptV1", issued_at: result.issued_at,
    configuration: { id: result.id, ...result.configuration }, status: "passed",
    plan: { path: "research/deliberately-absent/plan.json", sha256: planPin.sha256 }, elapsed_seconds: 1,
    outputs: [{ path: "research/deliberately-absent/joint-draws.npz", sha256: archiveSha }] };
  mutateReceipt?.(receipt);
  const receiptPin = await put(sourceRoot, "local-evidence/prediction-receipt.json", receipt);
  const genericPin = await put(sourceRoot, "local-evidence/generic.json", { schema: "InventedCandidateEvidenceV1" });
  result.qualification.evidence = [genericPin, receiptPin, planPin];
  mutateResult?.(result);
  const resultPin = await put(sourceRoot, "native-candidate.json", result);
  return { sourceRoot, result, planPin, receiptPin, genericPin,
    options: { sourceRoot, result: resultPin, seatQuantileConvention: "inverse_cdf" } };
}
async function snapshot(root) {
  const rows = [];
  for (const path of (await readdir(root, { recursive: true, withFileTypes: true })).filter(row => row.isFile())) {
    const file = join(path.parentPath, path.name);
    rows.push([file, digest(await readFile(file))]);
  }
  return rows.sort((a, b) => a[0].localeCompare(b[0]));
}

for (const kind of ["original", "combined"]) test(`offline ${kind} verification binds explicit local evidence without changing native qualification or source files`, async t => {
  const site = await fixture(t, { kind });
  const before = await snapshot(site.sourceRoot), qualification = structuredClone(site.result.qualification);
  const checked = await checkHouseLiveCandidate(site.options);
  assert.equal(checked.schema, "HouseLiveCandidateVerificationV1");
  assert.equal(checked.publicationAdmission, false);
  assert.match(checked.notice, /does not admit the candidate for publication/);
  assert.deepEqual(checked.result, { id: site.result.id, sha256: site.options.result.sha256 });
  assert.deepEqual(checked.qualification, qualification);
  assert.equal(checked.qualification.publication_status, "not_reviewed");
  assert.deepEqual(checked.predictionEvidence, { kind, planSha256: site.planPin.sha256, receiptSha256: site.receiptPin.sha256 });
  assert.equal(checked.jointArchive.rowsChecked, 4);
  assert.equal(checked.jointArchive.archiveFormat, kind === "combined" ? "house_live_npz_combined_chain_major_v1" : "house_live_npz_chain_major_v1");
  assert.deepEqual(await snapshot(site.sourceRoot), before);
  assert.equal((await readdir(site.sourceRoot, { recursive: true })).some(path => path.includes("publication.json") || path.startsWith("research")), false);
});

test("candidate CLI requires explicit pins and reports combined checks without publication admission", async t => {
  const site = await fixture(t, { kind: "combined" });
  const cli = fileURLToPath(new URL("../scripts/check_house_live_candidate.mjs", import.meta.url));
  const invoked = spawnSync(process.execPath, ["--experimental-strip-types", cli,
    "--source-root", site.sourceRoot, "--result", site.options.result.path,
    "--result-sha256", site.options.result.sha256, "--seat-quantile-convention", "inverse_cdf"], { encoding: "utf8" });
  assert.equal(invoked.status, 0, invoked.stderr);
  const checked = JSON.parse(invoked.stdout);
  assert.equal(checked.predictionEvidence.kind, "combined");
  assert.equal(checked.qualification.publication_status, "not_reviewed");
  assert.equal(checked.publicationAdmission, false);
  const noConvention = spawnSync(process.execPath, ["--experimental-strip-types", cli,
    "--source-root", site.sourceRoot, "--result", site.options.result.path, "--result-sha256", site.options.result.sha256], { encoding: "utf8" });
  assert.notEqual(noConvention.status, 0);
  assert.match(noConvention.stderr, /explicit candidate seat-quantile convention/);
});

test("a native reviewed string never turns a single candidate consistency check into publication admission", async t => {
  const site = await fixture(t, { mutateResult: result => { result.qualification.publication_status = "reviewed"; } });
  const checked = await checkHouseLiveCandidate(site.options);
  assert.equal(checked.qualification.publication_status, "reviewed");
  assert.deepEqual(checked.qualification, site.result.qualification);
  assert.equal(checked.publicationAdmission, false);
});

test("candidate, local receipt, local plan, and raw archive bytes must match their explicit pins", async t => {
  for (const artifact of ["result", "receipt", "plan", "archive"]) {
    const site = await fixture(t, { kind: "combined" });
    const pin = { result: site.options.result, receipt: site.receiptPin, plan: site.planPin, archive: site.result.joint_draws }[artifact];
    await appendFile(join(site.sourceRoot, pin.path), " ");
    await assert.rejects(checkHouseLiveCandidate(site.options), /fingerprint mismatch|decoder failed/);
  }
});

test("candidate verification requires one receipt and plan pair, including for original archives", async t => {
  const mutations = [
    result => { result.qualification.evidence = result.qualification.evidence.slice(0, 1); },
    result => { result.qualification.evidence.pop(); },
    result => { result.qualification.evidence.splice(1, 1); },
    result => { result.qualification.evidence.push(result.qualification.evidence[2]); },
    result => { result.qualification.evidence.push(result.qualification.evidence[1]); },
  ];
  for (const mutateResult of mutations) {
    const site = await fixture(t, { mutateResult });
    await assert.rejects(checkHouseLiveCandidate(site.options), /requires an authenticated local prediction receipt and plan pair|missing or ambiguous/);
  }
  const missingFile = await fixture(t);
  await rm(join(missingFile.sourceRoot, missingFile.planPin.path));
  await assert.rejects(checkHouseLiveCandidate(missingFile.options), { code: "ENOENT" });
});

test("candidate evidence must bind the native configuration, plan hash, joint hash, and chain counts", async t => {
  const mutations = [
    { mutateReceipt: receipt => { receipt.plan.sha256 = "a".repeat(64); } },
    { mutateReceipt: receipt => { receipt.outputs[0].sha256 = "a".repeat(64); } },
    { mutateReceipt: receipt => { receipt.status = "diagnostic_failure"; } },
    { mutatePlan: plan => { plan.configuration.polling = "none"; } },
    { mutatePlan: plan => { plan.settings.draws = 3; } },
    { mutatePlan: plan => { plan.source_runs[1].joint_draw_offset = 0; } },
  ];
  for (const mutation of mutations) {
    const site = await fixture(t, { kind: "combined", ...mutation });
    await assert.rejects(checkHouseLiveCandidate(site.options), /prediction receipt|prediction plan|combined source/);
  }
  const wrongConvention = await fixture(t);
  await assert.rejects(checkHouseLiveCandidate({ ...wrongConvention.options, seatQuantileConvention: "linear" }), /quantiles do not match/);
  await assert.rejects(checkHouseLiveCandidate({ ...wrongConvention.options,
    result: { path: "../outside.json", sha256: wrongConvention.options.result.sha256 } }), /Invalid live House artifact reference/);
});
