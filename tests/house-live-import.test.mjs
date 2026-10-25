import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { appendFile, lstat, mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { importHouseLive } from "../scripts/import_house_live.mjs";
import { loadHouseLiveSitePublication, readHouseLiveAvailability } from "../lib/house-live-server.ts";
import { makeHouseLiveCatalog, makeHouseLiveResult, SYNTHETIC_CONFIGURATIONS } from "./fixtures/house-live-fixtures.mjs";
import { makeHouseLiveCombinedPlan } from "./fixtures/house-live-combined-plan.mjs";

// Independent invented receipt construction: no fixture below asserts real review.
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
async function put(root, path, value) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(`${JSON.stringify(value, null, 1)}\n\n`);
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
function geometry(results) {
  return { type: "FeatureCollection", features: results[0].races.map((race, index) => {
    const x = -120 + index % 25, y = 26 + Math.floor(index / 25);
    return { type: "Feature", properties: { districtKey: race.district_key },
      geometry: { type: "Polygon", coordinates: [[[x, y], [x + .5, y], [x + .5, y + .5], [x, y + .5], [x, y]]] } };
  }) };
}
async function fixture(t, { mutateResults, mutateCatalog, mutateReview, mutateGeography, mutateQualification,
  predictionPlan, mutatePredictionPlan, mutatePredictionReceipt } = {}) {
  const base = await mkdtemp(join(tmpdir(), "house-live-offline-import-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const sourceRoot = join(base, "native"), reviewRoot = join(base, "operator");
  await mkdir(sourceRoot); await mkdir(reviewRoot);
  const catalog = makeHouseLiveCatalog();
  const results = SYNTHETIC_CONFIGURATIONS.map(configuration => makeHouseLiveResult(configuration));
  const jointPath = "private/predictive.npz";
  await mkdir(join(sourceRoot, "private"));
  const written = spawnSync("python3", [fileURLToPath(new URL("./fixtures/house-live-joint-archive.py", import.meta.url))], {
    input: JSON.stringify({ destination: join(sourceRoot, jointPath), raceIds: results[0].joint_draws.ordered_race_ids,
      rows: fourRows(), chains: 2, ...(predictionPlan === "combined" ? { sourceRunLengths: [2, 2] } : {}) }), encoding: "utf8",
  });
  assert.equal(written.status, 0, written.stderr);
  const jointSha = digest(await readFile(join(sourceRoot, jointPath)));
  const geo = geometry(results);
  mutateGeography?.(geo);
  const geographyPin = await put(sourceRoot, results[0].geography.path, geo);
  const sourceReceipt = await put(sourceRoot, results[0].geography.source_receipt.path, { invented: true, source: "grid only" });
  catalog.source_policy = await put(sourceRoot, catalog.source_policy.path, Buffer.from("invented opaque source policy\n\0"));
  const numericalReview = await put(sourceRoot, "synthetic/numerical-review.json",
    { invented: true, note: "Opaque invented numerical-review bytes; no real scientific review." });
  const qualification = { schema: "HouseLivePublicationQualificationV1", status: "reviewed",
    numerical_review: numericalReview, seat_quantile_convention: "inverse_cdf", share_quantile_convention: "linear",
    limitations: [...results[0].qualification.limitations] };
  mutateQualification?.(qualification);
  catalog.qualification = await put(sourceRoot, catalog.qualification.path, qualification);
  const genericEvidence = await put(sourceRoot, "synthetic/qualification.json",
    { schema: "InventedGenericQualificationEvidenceV1", note: "Invented evidence, with no prediction receipt or plan." });
  const predictionEvidence = [];
  for (const [index, result] of results.entries()) {
    result.qualification.publication_status = "reviewed";
    Object.assign(result.geography, geographyPin);
    result.geography.source_receipt = { ...sourceReceipt };
    Object.assign(result.joint_draws, { path: jointPath, sha256: jointSha });
    result.chamber.joint_draws_sha256 = jointSha;
    result.qualification.evidence = [genericEvidence];
    if (predictionPlan) {
      const plan = makeHouseLiveCombinedPlan(result);
      if (predictionPlan === "original") {
        plan.schema = "HouseLivePredictionPlanV1";
        plan.settings = { chains: 2, draws: 2, warmup: 1, seed: 2000 };
        delete plan.source_runs;
        delete plan.combination;
      }
      mutatePredictionPlan?.(plan, result, index);
      const planPin = await put(sourceRoot, `evidence/local-${index}/frozen-law.json`, plan);
      const receipt = { schema: "HouseLivePredictionReceiptV1", issued_at: result.issued_at,
        configuration: { id: result.id, ...result.configuration }, status: "passed",
        plan: { path: `research/never-follow-this/${result.id}/plan.json`, sha256: planPin.sha256 },
        elapsed_seconds: 1, outputs: [{ path: `research/never-follow-this/${result.id}/joint-draws.npz`, sha256: jointSha }] };
      mutatePredictionReceipt?.(receipt, result, index);
      const receiptPin = await put(sourceRoot, `evidence/local-${index}/prediction-receipt.json`, receipt);
      result.qualification.evidence.push(receiptPin, planPin);
      predictionEvidence.push({ plan, planPin, receipt, receiptPin });
    }
  }
  mutateResults?.(results, catalog);
  for (const [index, result] of results.entries()) catalog.entries[index].result =
    await put(sourceRoot, catalog.entries[index].result.path, result);
  mutateCatalog?.(catalog);
  const catalogPin = await put(sourceRoot, "native-catalog.json", catalog);
  const review = {
    schema: "HouseLiveSiteReviewV1", decision: "publish",
    joint_format: "house_live_npz_chain_major_v1", seat_quantile_convention: "inverse_cdf",
    catalog_sha256: catalogPin.sha256, source_policy_sha256: catalog.source_policy.sha256,
    qualification_sha256: catalog.qualification.sha256, geography_sha256: geographyPin.sha256,
    geography_source_receipt_sha256: sourceReceipt.sha256,
    results: results.map((result, index) => ({ id: catalog.entries[index].id,
      result_sha256: catalog.entries[index].result.sha256, joint_draws_sha256: result.joint_draws.sha256 })),
    checks: { inputsVerified: true, calculationsReviewed: true, jointRowsVerified: true,
      crossModelRosterVerified: true, geographyVerified: true },
    limitations: ["Invented admission fixture; not a real forecast or scientific review."],
  };
  mutateReview?.(review);
  const reviewPin = await put(reviewRoot, "decisions/release.json", review);
  await writeFile(join(sourceRoot, "private/conditional-shares.npz"), "must never be copied");
  await writeFile(join(sourceRoot, "private/training-bank.npz"), "must never be copied");
  return { base, catalog, results, review, geographyPin, qualification, genericEvidence, predictionEvidence,
    options: { sourceRoot, catalog: catalogPin, reviewRoot, review: reviewPin,
      geographySource: { name: "Invented grid", url: "https://example.invalid/grid", note: "Synthetic geometry only." },
      outputDirectory: join(base, "new-release") } };
}
async function absent(path) { await assert.rejects(lstat(path), { code: "ENOENT" }); }

const installer = fileURLToPath(new URL("../scripts/install_house_live_release.py", import.meta.url));
function install(parent, source, destination) {
  return spawnSync("python3", [installer, "--parent", parent, "--source", source, "--destination", destination],
    { encoding: "utf8" });
}

test("atomic installer never replaces an existing empty directory, nonempty directory, or symlink", async t => {
  const parent = await mkdtemp(join(tmpdir(), "house-live-no-replace-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  await mkdir(join(parent, "staging"));
  await writeFile(join(parent, "staging", "checked.json"), "invented checked bytes");
  await mkdir(join(parent, "empty"));
  await mkdir(join(parent, "nonempty"));
  await writeFile(join(parent, "nonempty", "keep.txt"), "preserve external bytes");
  await symlink(join(parent, "nonempty"), join(parent, "linked"));
  for (const destination of ["empty", "nonempty", "linked"]) {
    const before = await lstat(join(parent, destination));
    const result = install(parent, "staging", destination);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /File exists/);
    const after = await lstat(join(parent, destination));
    assert.equal(after.ino, before.ino);
    assert.equal(after.dev, before.dev);
    assert.equal(after.isSymbolicLink(), before.isSymbolicLink());
    assert.equal(await readFile(join(parent, "staging", "checked.json"), "utf8"), "invented checked bytes");
  }
  assert.deepEqual(await readdir(join(parent, "empty")), []);
  assert.equal(await readFile(join(parent, "nonempty", "keep.txt"), "utf8"), "preserve external bytes");
  const result = install(parent, "staging", "new-release");
  assert.equal(result.status, 0, result.stderr);
  await absent(join(parent, "staging"));
  assert.equal(await readFile(join(parent, "new-release", "checked.json"), "utf8"), "invented checked bytes");
});

test("atomic installer has no mutation fallback on unsupported platforms or syscalls", async t => {
  const parent = await mkdtemp(join(tmpdir(), "house-live-unsupported-install-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  await mkdir(join(parent, "staging"));
  await writeFile(join(parent, "staging", "keep.txt"), "preserve source");
  const probes = [
    'module["sys"].platform = "darwin"',
    'module["ctypes"].CDLL = lambda *args, **kwargs: object()',
    'def unavailable(*args):\n    module["ctypes"].set_errno(38)\n    return -1\nclass Libc:\n    pass\nlibc = Libc()\nlibc.renameat2 = unavailable\nmodule["ctypes"].CDLL = lambda *args, **kwargs: libc',
  ];
  for (const probe of probes) {
    const script = `import runpy, sys\nmodule = runpy.run_path(sys.argv[1])\n${probe}\nmodule["install"](sys.argv[2], "staging", "new-release")\n`;
    const result = spawnSync("python3", ["-c", script, installer, parent], { encoding: "utf8" });
    assert.notEqual(result.status, 0);
    await absent(join(parent, "new-release"));
    assert.equal(await readFile(join(parent, "staging", "keep.txt"), "utf8"), "preserve source");
  }
});

test("imports all 16 invented native results byte for byte; site loads every choice without raw archives", async t => {
  const site = await fixture(t);
  const { options } = site;
  const before = await readFile(join(options.sourceRoot, options.catalog.path));
  assert.deepEqual(await importHouseLive(options), { outputDirectory: options.outputDirectory, results: 16,
    jointRowsChecked: 64, catalogSha256: options.catalog.sha256, reviewSha256: options.review.sha256 });
  const manifest = JSON.parse(await readFile(join(options.outputDirectory, "publication.json")));
  assert.deepEqual(manifest.catalog, options.catalog);
  assert.deepEqual(await readFile(join(options.outputDirectory, options.catalog.path)), before);
  assert.deepEqual(await readFile(join(options.sourceRoot, options.catalog.path)), before);
  for (const pin of [options.catalog, site.catalog.source_policy, site.catalog.qualification, site.qualification.numerical_review, site.geographyPin,
    site.results[0].geography.source_receipt, site.genericEvidence, ...site.catalog.entries.map(entry => entry.result)]) {
    assert.deepEqual(await readFile(join(options.outputDirectory, pin.path)), await readFile(join(options.sourceRoot, pin.path)));
  }
  assert.deepEqual(await readFile(join(options.outputDirectory, manifest.review.path)),
    await readFile(join(options.reviewRoot, options.review.path)));
  assert.deepEqual(await readHouseLiveAvailability(options.outputDirectory), { available: true });
  for (const entry of site.catalog.entries) {
    const publication = await loadHouseLiveSitePublication(options.outputDirectory, entry.id);
    assert.equal(publication.view.run.id, entry.id);
    assert.equal(publication.view.races.length, 435);
    assert.deepEqual(publication.publicationReview.limitations, site.review.limitations);
  }
  const paths = await readdir(options.outputDirectory, { recursive: true });
  assert.equal(paths.filter(path => path.endsWith(".npz")).length, 0);
  assert.equal(paths.includes("private"), false);
  assert.equal(paths.filter(path => /\.(?:geo)?json$/.test(path)).length, 25);
});

test("combined archives use the authenticated local plan selected by receipt SHA and retain its exact evidence bytes", async t => {
  const site = await fixture(t, { predictionPlan: "combined" });
  const imported = await importHouseLive(site.options);
  assert.equal(imported.results, 16);
  assert.equal(imported.jointRowsChecked, 64);
  for (const { planPin, receiptPin } of site.predictionEvidence) {
    for (const pin of [planPin, receiptPin]) assert.deepEqual(
      await readFile(join(site.options.outputDirectory, pin.path)), await readFile(join(site.options.sourceRoot, pin.path)));
  }
  for (const entry of site.catalog.entries) {
    const publication = await loadHouseLiveSitePublication(site.options.outputDirectory, entry.id);
    assert.equal(publication.view.run.id, entry.id);
    assert.deepEqual(await readFile(join(site.options.outputDirectory, entry.result.path)),
      await readFile(join(site.options.sourceRoot, entry.result.path)));
  }
  const paths = await readdir(site.options.outputDirectory, { recursive: true });
  assert.equal(paths.some(path => path.endsWith(".npz")), false);
  assert.equal(paths.includes("research"), false);
  assert.equal(paths.includes("private"), false);
  assert.deepEqual(await readHouseLiveAvailability(site.options.outputDirectory), { available: true });
});

test("original nine-member archives still accept an explicit authenticated original receipt and local plan", async t => {
  const site = await fixture(t, { predictionPlan: "original" });
  assert.equal((await importHouseLive(site.options)).results, 16);
  assert.deepEqual(await readHouseLiveAvailability(site.options.outputDirectory), { available: true });
});

test("prediction receipt and local plan mappings cannot be missing or ambiguous", async t => {
  const mutations = [
    evidence => evidence.slice(0, -1),
    evidence => [evidence[0], evidence[2]],
    evidence => [...evidence, evidence[2]],
    evidence => [...evidence, evidence[1]],
  ];
  for (const mutate of mutations) {
    const site = await fixture(t, { predictionPlan: "combined", mutateResults: results => {
      results[15].qualification.evidence = mutate(results[15].qualification.evidence);
    } });
    await assert.rejects(importHouseLive(site.options), /missing or ambiguous local prediction receipt\/plan mapping/);
    await absent(site.options.outputDirectory);
  }
  const original = await fixture(t, { predictionPlan: "original", mutateResults: results => {
    results[15].qualification.evidence.pop();
  } });
  await assert.rejects(importHouseLive(original.options), /missing or ambiguous local prediction receipt\/plan mapping/);
  await absent(original.options.outputDirectory);
});

test("combined receipt must bind the native configuration, passing status, original plan SHA and joint artifact SHA", async t => {
  const mutations = [
    receipt => { receipt.plan.sha256 = "a".repeat(64); },
    receipt => { receipt.outputs[0].sha256 = "a".repeat(64); },
    receipt => { receipt.outputs.push(receipt.outputs[0]); },
    receipt => { receipt.configuration.id = "invented-wrong-run"; },
    receipt => { receipt.status = "diagnostic_failure"; },
  ];
  for (const mutate of mutations) {
    const site = await fixture(t, { predictionPlan: "combined", mutatePredictionReceipt: (receipt, _result, index) => {
      if (index === 15) mutate(receipt);
    } });
    await assert.rejects(importHouseLive(site.options), /prediction receipt/);
    await absent(site.options.outputDirectory);
  }
});

test("combined plan bytes, configuration, chain counts and source layout are authenticated before raw checks", async t => {
  const mutations = [
    plan => { plan.configuration.polling = "none"; },
    plan => { delete plan.configuration.id; },
    plan => { plan.settings.chains = 4; },
    plan => { plan.settings.draws = 3; },
    plan => { plan.source_runs[1].joint_draw_offset = 0; },
    plan => { plan.source_runs[1].settings.seed = plan.source_runs[0].settings.seed; },
  ];
  for (const mutate of mutations) {
    const site = await fixture(t, { predictionPlan: "combined", mutatePredictionPlan: (plan, _result, index) => {
      if (index === 15) mutate(plan);
    } });
    await assert.rejects(importHouseLive(site.options), /prediction plan configuration or chain\/draw counts|combined source chain\/draw lineage/);
    await absent(site.options.outputDirectory);
  }
  const corrupted = await fixture(t, { predictionPlan: "combined" });
  await appendFile(join(corrupted.options.sourceRoot, corrupted.predictionEvidence[15].planPin.path), " ");
  await assert.rejects(importHouseLive(corrupted.options), /fingerprint mismatch/);
  await absent(corrupted.options.outputDirectory);
});

test("combined raw arrays cannot be admitted as an original archive when explicit prediction evidence is absent", async t => {
  const site = await fixture(t, { predictionPlan: "combined", mutateResults: results => {
    for (const result of results) result.qualification.evidence = result.qualification.evidence.slice(0, 1);
  } });
  await assert.rejects(importHouseLive(site.options), /decoder failed/);
  await absent(site.options.outputDirectory);
});

test("changed source bytes and changed operator-review bytes prevent a destination", async t => {
  for (const target of ["source", "review"]) {
    const { options } = await fixture(t);
    await appendFile(join(target === "source" ? options.sourceRoot : options.reviewRoot,
      target === "source" ? options.catalog.path : options.review.path), " ");
    await assert.rejects(importHouseLive(options), /fingerprint mismatch/);
    await absent(options.outputDirectory);
  }
});

test("review must bind source policy, every result identity, joint identity, and explicit export convention", async t => {
  const mutations = [
    review => { review.source_policy_sha256 = "a".repeat(64); },
    review => { review.results[15].result_sha256 = "a".repeat(64); },
    review => { review.results[15].joint_draws_sha256 = "a".repeat(64); },
    review => { review.results[15].id = "invented-unlisted-id"; },
    review => { delete review.seat_quantile_convention; },
    review => { review.joint_format = "invented-guess"; },
  ];
  for (const mutateReview of mutations) {
    const { options } = await fixture(t, { mutateReview });
    await assert.rejects(importHouseLive(options), /review|identity/);
    await absent(options.outputDirectory);
  }
});

test("native qualification must be reviewed, state valid conventions, and match the operator review", async t => {
  const mutations = [
    qualification => { qualification.status = "pending"; },
    qualification => { qualification.schema = "invented-unrecognized-qualification"; },
    qualification => { delete qualification.seat_quantile_convention; },
    qualification => { qualification.share_quantile_convention = "inverse_cdf"; },
    qualification => { qualification.seat_quantile_convention = "linear"; },
    qualification => { qualification.limitations = []; },
  ];
  for (const mutateQualification of mutations) {
    const { options } = await fixture(t, { mutateQualification });
    await assert.rejects(importHouseLive(options), /Native House qualification/);
    await absent(options.outputDirectory);
  }
  const { options, qualification } = await fixture(t);
  await appendFile(join(options.sourceRoot, qualification.numerical_review.path), " ");
  await assert.rejects(importHouseLive(options), /fingerprint mismatch/);
  await absent(options.outputDirectory);
  const omittedLimitations = await fixture(t, { mutateResults: results => {
    results[15].qualification.limitations = ["Another invented note omits the global limitation."];
  } });
  await assert.rejects(importHouseLive(omittedLimitations.options), /reviewed identity and qualification/);
  await absent(omittedLimitations.options.outputDirectory);
});

test("native catalog/configuration and cross-model candidate roster must match before staging", async t => {
  const wrongId = await fixture(t, { mutateResults: results => { results[15].id = "invented-wrong-result"; } });
  await assert.rejects(importHouseLive(wrongId.options), /reviewed identity/);
  await absent(wrongId.options.outputDirectory);
  const wrongConfiguration = await fixture(t, { mutateResults: results => { results[15].configuration = results[0].configuration; } });
  await assert.rejects(importHouseLive(wrongConfiguration.options), /reviewed identity/);
  await absent(wrongConfiguration.options.outputDirectory);
  const roster = await fixture(t, { mutateResults: results => {
    results[15].races[0].candidates[0].name = "Invented changed name";
    results[15].races[0].outcomes[0].label = "Invented changed name";
  } });
  await assert.rejects(importHouseLive(roster.options), /candidate, outcome, rule, or map identity/);
  await absent(roster.options.outputDirectory);
});

test("review booleans cannot admit wrong raw calculations or a guessed quantile convention", async t => {
  const calculations = await fixture(t, { mutateResults: results => {
    results[15].races[0].outcomes[0].p_win = .75;
    results[15].races[0].outcomes[1].p_win = .25;
  } });
  await assert.rejects(importHouseLive(calculations.options), /winner frequency differs/);
  await absent(calculations.options.outputDirectory);
  const convention = await fixture(t, {
    mutateQualification: qualification => { qualification.seat_quantile_convention = "linear"; },
    mutateReview: review => { review.seat_quantile_convention = "linear"; },
  });
  await assert.rejects(importHouseLive(convention.options), /quantiles do not match/);
  await absent(convention.options.outputDirectory);
});

test("invalid staged geography fails selected-response validation and cleans temporary staging", async t => {
  const { options, base } = await fixture(t, { mutateGeography: geography => { geography.features.pop(); } });
  await assert.rejects(importHouseLive(options), /selected-result or geography checks/);
  await absent(options.outputDirectory);
  assert.equal((await readdir(base)).some(path => path.startsWith(".house-live-import-")), false);
});

test("existing destination and symlink output parent are untouched", async t => {
  const { options, base } = await fixture(t);
  await mkdir(options.outputDirectory);
  const marker = join(options.outputDirectory, "existing.txt");
  await writeFile(marker, "keep existing release");
  await assert.rejects(importHouseLive(options), /destination already exists/);
  assert.equal(await readFile(marker, "utf8"), "keep existing release");
  assert.deepEqual(await readdir(options.outputDirectory), ["existing.txt"]);
  const symlinkParent = join(base, "linked-parent");
  const actualParent = join(base, "actual-parent");
  await mkdir(actualParent); await symlink(actualParent, symlinkParent);
  await assert.rejects(importHouseLive({ ...options, outputDirectory: join(symlinkParent, "new-release") }), /must not traverse symlinks/);
  assert.deepEqual(await readdir(actualParent), []);
  const outputLink = join(base, "linked-output");
  await symlink(actualParent, outputLink);
  await assert.rejects(importHouseLive({ ...options, outputDirectory: outputLink }), /destination already exists/);
  assert.equal((await lstat(outputLink)).isSymbolicLink(), true);
});

test("native publication manifest, review and file-parent collisions are rejected", async t => {
  for (const path of ["publication.json", "publication.json/child.json"]) {
    const { options } = await fixture(t, { mutateResults: (_results, catalog) => { catalog.entries[0].result.path = path; } });
    await assert.rejects(importHouseLive(options), /conflicts with the publication manifest/);
    await absent(options.outputDirectory);
  }
  for (const nested of [false, true]) {
    const { options } = await fixture(t);
    const nativeBytes = await readFile(join(options.sourceRoot, options.catalog.path));
    const reviewPath = `operator-review-${options.review.sha256}.json`;
    options.catalog = await put(options.sourceRoot, nested ? `${reviewPath}/catalog.json` : reviewPath, nativeBytes);
    await assert.rejects(importHouseLive(options), nested ? /file and parent paths conflict/ : /operator review path/);
    await absent(options.outputDirectory);
  }
});
