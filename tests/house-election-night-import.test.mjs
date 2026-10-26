import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rename, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { importHouseElectionNightReferences, normalizeHouseElectionNightReferences } from "../scripts/import_house_election_night_references.mjs";
import { makeHouseReferenceImportFixture, digest } from "./fixtures/house-election-night-import-fixture.mjs";
import { houseLiveView } from "../lib/house-live-view.ts";
import { toHouseElectionNightForecast } from "../lib/election-night-forecast.ts";
import { matchHouseExpectedErrorReference } from "../lib/election-night-house-reference.ts";

async function fixture(t, mutations) {
  const base = await mkdtemp(join(tmpdir(), "house-reference-import-test-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  return { base, ...await makeHouseReferenceImportFixture(base, mutations) };
}
async function rejectsMutation(t, mutations, pattern) {
  const { options } = await fixture(t, mutations);
  await assert.rejects(normalizeHouseElectionNightReferences(options), pattern);
}

test("all16 explicitly reviewed synthetic editions normalize losslessly and attach to actual runtime adapter", async t => {
  const f = await fixture(t), before = structuredClone(f);
  const { catalog, publication, verification } = await normalizeHouseElectionNightReferences(f.options);
  assert.equal(catalog.entries.length, 16);
  assert.equal(new Set(catalog.entries.map(e => e.forecastIssuedAt)).size, 16, "native sequential issue times must remain truthful");
  assert.equal(catalog.evaluationLock, false); assert.equal(catalog.observedResultsConsumed, false);
  assert.equal(verification.rawDrawRehash, false);
  assert.ok(verification.delegatedRawArchiveArtifacts > 0);
  assert.equal(publication.sha256, digest(Buffer.from(`${JSON.stringify(catalog, null, 2)}\n`)));
  for (const [i, native] of f.natives.entries()) {
    const entry = catalog.entries[i], descriptor = publication.editions[i];
    assert.deepEqual(entry.reference, f.rawSidecars[i], "entire precision, qualifications, modal calls and native pins survive");
    const forecast = toHouseElectionNightForecast(houseLiveView(native, f.nativeCatalog.entries[i]));
    const matched = matchHouseExpectedErrorReference(forecast, catalog, descriptor.qualification);
    assert.ok(matched, `${native.id} must be readable by the actual runtime matcher`);
    assert.equal(matched.winnerRaceCount, 435); assert.equal(matched.pairRaceCount, 412); assert.equal(matched.pairInapplicableRaceCount, 23);
    assert.equal(matched.perRace.filter(r => r.availability === "available").length, 412);
    assert.equal(matched.summary.modalGroupMissCount.mean, forecast.summary.expectedModalMisses);
    assert.equal(matched.summary.meanAbsolutePairMarginError.mean, 10);
    assert.deepEqual(matched.summary.central90PairIntervalCoverageCount.range90, { lower: 370, upper: 370 });
    assert.deepEqual(entry.reference.modal_calls[0].tied_modal_outcomes, ["D", "R"]);
    assert.equal(entry.sourcePins.operatorReviewSha256, f.options.review.sha256);
  }
  assert.deepEqual(f, before, "normalizer must not mutate source fixtures/options");
});

test("publication admission cannot come from producer selfclaim, missing review or stale review", async t => {
  const f = await fixture(t);
  await assert.rejects(normalizeHouseElectionNightReferences({ ...f.options, review: undefined }), /pin/);
  await assert.rejects(normalizeHouseElectionNightReferences({ ...f.options, reviewRoot: f.options.sourceRoot }), /separately/);
  await assert.rejects(normalizeHouseElectionNightReferences({ ...f.options, review: { ...f.options.review, sha256: "1".repeat(64) } }), /fingerprint/);
  await rejectsMutation(t, { review: r => { r.collectionSha256 = "2".repeat(64); } }, /operator review/);
  await rejectsMutation(t, { review: r => { r.status = "implementation_review_passed_actual_collection_pending"; } }, /operator review/);
});

test("actual audit and source receipt bytes are authenticated independently", async t => {
  const f = await fixture(t), catalog = JSON.parse(await readFile(join(f.options.collectionRoot, "catalog.json")));
  const audit = join(f.options.collectionRoot, catalog.entries[0].independent_audit.path);
  await writeFile(audit, '{"status":"passed","changed":true}\n');
  await assert.rejects(normalizeHouseElectionNightReferences(f.options), /fingerprint/);
  const g = await fixture(t), p = g.rawSidecars[0].source_receipt.path;
  await writeFile(join(g.options.sourceRoot, p), '{"status":"passed"}\n');
  await assert.rejects(normalizeHouseElectionNightReferences(g.options), /fingerprint/);
});

test("duplicate/incomplete/config-mismatched and stale packet collections fail closed", async t => {
  for (const mutation of [
    { catalog: c => { c.entries[1] = structuredClone(c.entries[0]); } },
    { catalog: c => { c.entries.pop(); } },
    { catalog: c => { c.entries[0].configuration.polling = "standard"; } },
    { ready: r => { r.catalog.sha256 = "3".repeat(64); } },
    { sidecar: (s, i) => { if (!i) s.native_result.sha256 = "4".repeat(64); } },
    { sidecar: (s, i) => { if (!i) s.forecast_issued_at = "2026-10-21T00:00:00Z"; } },
    { sidecar: (s, i) => { if (!i) s.issued_at = "2026-10-01T00:00:00Z"; } },
  ]) await rejectsMutation(t, mutation);
});

test("relative traversal and escaped symlinks are rejected before attachment", async t => {
  await rejectsMutation(t, { catalog: c => { c.entries[0].sidecar.path = "../untrusted.json"; } }, /pin/);
  const f = await fixture(t), c = JSON.parse(await readFile(join(f.options.collectionRoot, "catalog.json")));
  const file = join(f.options.collectionRoot, c.entries[0].independent_audit.path), outside = join(f.base, "escaped.json");
  await rename(file, outside); await symlink(outside, file);
  await assert.rejects(normalizeHouseElectionNightReferences(f.options), /escapes/);
  await assert.rejects(normalizeHouseElectionNightReferences({ ...f.options, nativeRoot: f.base }), /scoped separately/);
});

test("pair orientation/mask/inapplicable accounting and D/R/O modal tie ordering are bound", async t => {
  for (const mutation of [
    s => { const r = Object.values(s.race_expected_errors).find(r => r.availability === "available"); [r.first_candidate_id, r.second_candidate_id] = [r.second_candidate_id, r.first_candidate_id]; },
    s => { s.draw_layout.pair_mask_in_winner_order[0] = true; },
    s => { Object.values(s.race_expected_errors)[0].expected_absolute_margin_error_pp = 0; },
    s => { s.modal_calls[0].tied_modal_outcomes.reverse(); },
    s => { s.modal_calls[0].member_candidate_ids.reverse(); },
    s => { s.definitions.modal_tie_rule = "lexicographic"; },
    s => { s.precision_scope = ""; },
  ]) await rejectsMutation(t, { sidecar: (s, i) => { if (!i) mutation(s); } });
});

test("receipt input/output links and independent audit reconciliation cannot be replaced by passed status", async t => {
  for (const mutation of [
    { receipt: (r, i) => { if (!i) r.input_pins.shift(); } },
    { receipt: (r, i) => { if (!i) r.outputs.pop(); } },
    { audit: (a, i) => { if (!i) a.receipt.sha256 = "5".repeat(64); } },
    { audit: (a, i) => { if (!i) a.max_abs_per_race_expected_error_diff_pp = 1; } },
    { audit: (a, i) => { if (!i) a.exact_modal_call_tie_count--; } },
    { sourceReceipt: (r, i) => { if (!i) r.outputs.pop(); } },
    { sourcePlan: (p, i) => { if (!i) p.settings.chains = 3; } },
    { collectionReceipt: r => { r.outputs.pop(); } },
  ]) await rejectsMutation(t, mutation);
});

test("scientific and numerical reviews bind selection, native catalog, results and selected source audit", async t => {
  for (const mutation of [
    { science: s => { s.selection_sha256 = "6".repeat(64); } },
    { numerical: n => { n.input_pins.shift(); } },
    { numerical: n => { n.input_pins.pop(); } },
    { sidecar: (s, i) => { if (!i) s.selected_reviews = []; } },
    { native: (n, i) => { if (i === 1) n.races[0].candidates[0].name = "Changed edition roster"; } },
  ]) await rejectsMutation(t, mutation);
});

test("overflow, original integer-row PMFs, count quantiles and metric units fail closed", async t => {
  for (const mutation of [
    s => { s.draw_layout.chains = Number.MAX_SAFE_INTEGER; s.draw_layout.draws_per_chain = 2; },
    s => { s.draw_layout.rows = 5; },
    s => { s.draw_layout.source_runs = [{ changed: "original source row ordering" }]; },
    s => { s.joint_summaries.modal_group_miss_count.quantiles.p05--; },
    s => { s.joint_summaries.modal_group_miss_count.sd = 1; },
    s => { const m = s.joint_summaries.modal_group_miss_count; m.pmf[217] = .9; m.pmf[216] = .05; m.pmf[218] = .05; },
    s => { s.joint_summaries.mean_signed_pair_margin_error_pp.mean = 1; },
    s => { s.joint_summaries.central90_pair_interval_coverage_count.units = "percentage_points"; },
  ]) await rejectsMutation(t, { sidecar: (s, i) => { if (!i) mutation(s); } });
});

test("V2 segmented metadata selects the exact index and original-row law without decoding archives", async t => {
  const f = await fixture(t, { storageVersion: 2 });
  const result = await normalizeHouseElectionNightReferences(f.options);
  for (const [i, entry] of result.catalog.entries.entries()) {
    assert.equal(entry.reference.conditional_storage.schema, "ConditionalSegmentsManifestV1");
    assert.deepEqual(entry.reference.conditional_storage.manifest, entry.reference.conditional_shares);
    const forecast = toHouseElectionNightForecast(houseLiveView(f.natives[i], f.nativeCatalog.entries[i]));
    assert.ok(matchHouseExpectedErrorReference(forecast, result.catalog, result.publication.editions[i].qualification));
  }
  for (const change of [
    p => { p.conditional_storage.manifest.sha256 = "7".repeat(64); },
    p => { p.conditional_storage.weights = "equal_weight_per_segment"; },
    p => { p.conditional_storage.schema = "unknown"; },
  ]) await rejectsMutation(t, { storageVersion: 2, sourcePlan: (p, i) => { if (!i) change(p); } }, /conditional manifest/);
  await rejectsMutation(t, { sidecar: (s, i) => { if (!i) s.conditional_shares.path = "arrays/other.npz"; } });
});

test("private exports bind exact bytes and never overwrite or activate public/producer data", async t => {
  const f = await fixture(t), outputDirectory = join(f.base, "private-export");
  const result = await importHouseElectionNightReferences({ ...f.options, outputDirectory });
  const bytes = await readFile(join(outputDirectory, "house-references.json"));
  assert.equal(digest(bytes), result.publication.sha256);
  assert.deepEqual(JSON.parse(await readFile(join(outputDirectory, "publication.json"))), result.publication);
  await assert.rejects(importHouseElectionNightReferences({ ...f.options, outputDirectory }), /EEXIST/);
  assert.deepEqual(await readFile(join(outputDirectory, "house-references.json")), bytes);
  await assert.rejects(importHouseElectionNightReferences({ ...f.options, outputDirectory: join(f.options.sourceRoot, "public-copy") }), /outside producer/);
  await assert.rejects(importHouseElectionNightReferences({ ...f.options,
    outputDirectory: new URL("../public/unwanted-house-import", import.meta.url).pathname }), /active public/);
});
