import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { appendFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { loadHouseLiveSitePublication, readHouseLiveAvailability, HouseLiveSelectionError }
  from "../lib/house-live-server.ts";
import { makeHouseLiveCatalog, makeHouseLiveResult, SYNTHETIC_CONFIGURATIONS }
  from "./fixtures/house-live-fixtures.mjs";
import { buildHouseLivePollingCollection, HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256 }
  from "../lib/house-live-polling.ts";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");

function geometry(results) {
  return { type: "FeatureCollection", features: results[0].races.map((race, index) => {
    const x = -120 + index % 25, y = 26 + Math.floor(index / 25);
    return { type: "Feature", properties: { districtKey: race.district_key, private_note: "discard" },
      geometry: { type: "Polygon", coordinates: [[[x, y], [x + .5, y],
        [x + .5, y + .5], [x, y + .5], [x, y]]] } };
  }) };
}

async function put(directory, path, value) {
  const bytes = Buffer.from(JSON.stringify(value));
  await mkdir(dirname(join(directory, path)), { recursive: true });
  await writeFile(join(directory, path), bytes);
  return { path, sha256: digest(bytes) };
}

async function fixture(t, mutate, includePolling = false) {
  const base = await mkdtemp(join(tmpdir(), "ballotmancy-live-site-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  const directory = join(base, "release");
  await mkdir(directory);
  const catalog = makeHouseLiveCatalog();
  const results = SYNTHETIC_CONFIGURATIONS.map(configuration => makeHouseLiveResult(configuration));
  for (const result of results) result.qualification.publication_status = "reviewed";
  mutate?.({ catalog, results });
  const geographyPin = await put(directory, "stage/geography.geojson", geometry(results));
  const receiptPin = await put(directory, "stage/source-receipt.json", { synthetic: true, edition: "invented-grid" });
  const policy = includePolling ? { schema: "HouseLiveSourcePolicyV1", forecast_cutoff: results[0].forecast_cutoff,
    polling_cutoff: results[0].forecast_cutoff, strict_as_of: true } : { synthetic: true, policy: "invented" };
  const policyPin = await put(directory, "stage/source-policy.json", policy);
  let pollingInput;
  if (includePolling) {
    const race = results[0].races[0], observationId = "invented:selected:AL-01", sourceSha = digest("invented poll source");
    const ledger = [{ status: "selected", reason: "", poll_id: "invented-poll", question_id: "invented-question",
      district_key: race.district_key, race_id: race.race_id, observation_id: observationId,
      source_id: "invented-poll-source", source_sha256: sourceSha, pollster: "Invented Pollster",
      start_date: "2026-10-15", field_end: "2026-10-17", available_date: "2026-10-19T00:00:00Z",
      population: "LV", sample_size_status: "not_reported_not_imputed", nominal_n: null,
      source_url: "https://example.invalid/poll", source_urls: ["https://example.invalid/poll"],
      first_candidate_id: race.candidates[0].candidate_id, second_candidate_id: race.candidates[2].candidate_id }];
    const ledgerSha = digest(JSON.stringify(ledger));
    const selection = { schema: "HouseLivePollSelectionV1", forecast_cutoff: policy.polling_cutoff,
      observations: 1, races: 1, outputs: [{ path: "research/polls-v3/ledger.json", sha256: ledgerSha }],
      input_pins: [{ path: "research/reviewed-selector.py", sha256: HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256 }] };
    const selectionSha = digest(JSON.stringify(selection));
    const admission = { schema: "HouseLiveAggregateInputAdmissionV1", status: "admitted_for_current_aggregate_comparison",
      forecast_cutoff: policy.forecast_cutoff, input_pins: [
        { path: "research/source-policy-v1.json", sha256: policyPin.sha256 },
        { path: "research/polls-v3/receipt.json", sha256: selectionSha },
        { path: "research/polls-v3/ledger.json", sha256: ledgerSha }] };
    const admissionPin = await put(directory, "stage/admission-v1.json", admission);
    for (const result of results) {
      result.qualification.evidence = [admissionPin];
      result.source_cutoffs = [{ role: "structural", at: policy.forecast_cutoff }, { role: "poll", at: policy.polling_cutoff }];
      result.provenance.push({ id: "invented-poll-source", label: "Invented poll source", url: "https://example.invalid/poll",
        retrieved_at: "2026-10-19T00:00:00Z", sha256: sourceSha, role: "polling" });
      if (result.configuration.polling === "standard") Object.assign(result.races[0],
        { poll_count: 1, poll_source_ids: [observationId], source_ids: [...result.races[0].source_ids, "invented-poll-source"] });
    }
    pollingInput = { ledger, ledgerSha256: ledgerSha, selection, selectionSha256: selectionSha, admission, admissionSha256: admissionPin.sha256 };
  }
  const numericalReviewPin = await put(directory, "stage/numerical-review.json", { synthetic: true, review: "invented" });
  const qualificationPin = await put(directory, "stage/qualification.json", {
    schema: "HouseLivePublicationQualificationV1", status: "reviewed", numerical_review: numericalReviewPin,
    seat_quantile_convention: "inverse_cdf", share_quantile_convention: "linear",
    limitations: results[0].qualification.limitations,
  });
  const stagedResults = [];
  for (const [index, result] of results.entries()) {
    const id = catalog.entries[index].id;
    result.geography.sha256 = geographyPin.sha256;
    result.geography.source_receipt.sha256 = receiptPin.sha256;
    const artifact = await put(directory, `stage/results/${id}.json`, result);
    stagedResults.push({ id, artifact });
    catalog.entries[index].result.sha256 = artifact.sha256;
  }
  catalog.source_policy.sha256 = policyPin.sha256;
  catalog.qualification.sha256 = qualificationPin.sha256;
  const catalogPin = await put(directory, "stage/catalog.json", catalog);
  const review = {
    schema: "HouseLiveSiteReviewV1", decision: "publish",
    joint_format: "house_live_npz_chain_major_v1", seat_quantile_convention: "inverse_cdf",
    catalog_sha256: catalogPin.sha256,
    source_policy_sha256: policyPin.sha256,
    qualification_sha256: qualificationPin.sha256,
    geography_sha256: geographyPin.sha256,
    geography_source_receipt_sha256: receiptPin.sha256,
    results: results.map((result, index) => ({ id: catalog.entries[index].id,
      result_sha256: stagedResults[index].artifact.sha256,
      joint_draws_sha256: result.joint_draws.sha256 })),
    checks: { inputsVerified: true, calculationsReviewed: true, jointRowsVerified: true,
      crossModelRosterVerified: true, geographyVerified: true },
    limitations: ["Invented release fixture only; no real forecast."],
  };
  const reviewPin = await put(directory, "stage/site-review.json", review);
  const manifest = {
    schema: "HouseLiveSiteManifestV1", status: "available", cycle: 2026,
    catalog: catalogPin, review: reviewPin, sourcePolicy: policyPin,
    qualification: qualificationPin,
    geography: { artifact: geographyPin, sourceReceipt: receiptPin,
      sourceName: "Invented grid fixture", sourceUrl: "https://example.invalid/invented-grid",
      sourceNote: "Synthetic squares; not electoral districts." },
    results: stagedResults,
  };
  const pollingCollection = includePolling ? buildHouseLivePollingCollection({ catalog, catalogSha256: catalogPin.sha256,
    results: results.map((result, index) => ({ result, resultSha256: stagedResults[index].artifact.sha256 })),
    sourcePolicy: policy, sourcePolicySha256: policyPin.sha256, ...pollingInput }) : null;
  if (pollingCollection) manifest.pollingDisclosure = await put(directory, "stage/polling-disclosure.json", pollingCollection);
  await writeFile(join(directory, "publication.json"), JSON.stringify(manifest));
  const writeManifest = () => writeFile(join(directory, "publication.json"), JSON.stringify(manifest));
  const writeReview = async () => {
    manifest.review = await put(directory, "stage/site-review.json", review);
    await writeManifest();
  };
  const writePolling = async () => {
    manifest.pollingDisclosure = await put(directory, "stage/polling-disclosure.json", pollingCollection);
    await writeManifest();
  };
  return { base, directory, catalog, results, review, manifest, writeManifest, writeReview, pollingCollection, writePolling };
}

test("loads only a reviewed selected result as a safe browser DTO and reports complete availability", async t => {
  const site = await fixture(t);
  const publication = await loadHouseLiveSitePublication(site.directory);
  assert.equal(publication.status, "available");
  assert.equal(publication.view.run.id, site.catalog.default_id);
  assert.equal(publication.catalog.entries.length, 16);
  assert.equal(publication.view.races.length, 435);
  assert.equal(publication.geography.geojson.features.length, 435);
  assert.equal(publication.geography.metadata.sourceName, "Invented grid fixture");
  assert.deepEqual(publication.publicationReview.limitations,
    ["Invented release fixture only; no real forecast."]);
  assert.ok(!publication.view.qualification.limitations.includes(publication.publicationReview.limitations[0]));
  assert.equal("path" in publication.view.jointDraws, false);
  assert.equal("path" in publication.view.sources[0], false);
  assert.equal("private_note" in publication.geography.geojson.features[0].properties, false);
  assert.equal(publication.polling, null);
  assert.deepEqual(await readHouseLiveAvailability(site.directory), { available: true });

  const another = site.catalog.entries[0];
  const selected = await loadHouseLiveSitePublication(site.directory, another.id);
  assert.equal(selected.view.run.id, another.id);
  assert.equal(selected.view.run.resultSha256, another.result.sha256);
});

test("site review field is emitted even when operator notes are empty", async t => {
  const site = await fixture(t);
  site.review.limitations = [];
  await site.writeReview();
  const publication = await loadHouseLiveSitePublication(site.directory);
  assert.deepEqual(publication.publicationReview, { limitations: [] });
  assert.ok(publication.view.qualification.limitations.length > 0);
});

test("authenticated polling sidecar projects exact selected questions and keeps baseline polling usage empty", async t => {
  const site = await fixture(t, ({ results }) => {
    results[1].forecast_cutoff = "2026-10-20T00:00:00+00:00";
  }, true);
  const publication = await loadHouseLiveSitePublication(site.directory);
  assert.equal(publication.polling.modelUsesSelectedQuestions, true);
  assert.equal(publication.polling.usedQuestions.length, 1);
  assert.equal(publication.polling.usedQuestions[0].observationId, site.results[1].races[0].poll_source_ids[0]);
  assert.equal(publication.polling.usedQuestions[0].sampleSize, null);
  assert.equal(JSON.stringify(publication.polling).includes('"path"'), false);
  const baseline = await loadHouseLiveSitePublication(site.directory, site.catalog.entries[0].id);
  assert.equal(baseline.polling.modelUsesSelectedQuestions, false);
  assert.deepEqual(baseline.polling.usedQuestions, []);
  assert.equal(baseline.polling.selectedPoolQuestionCount, 1);
  assert.deepEqual(await readHouseLiveAvailability(site.directory), { available: true });
});

test("sidecar catalog, source-policy, admission, selection and ledger identities must bind", async t => {
  for (const field of ["catalogSha256", "sourcePolicySha256", "admissionSha256", "selectionSha256", "ledgerSha256"]) {
    const site = await fixture(t, null, true);
    site.pollingCollection.bindings[field] = "a".repeat(64);
    await site.writePolling();
    await assert.rejects(loadHouseLiveSitePublication(site.directory), /polling disclosure does not bind/);
  }
  const wrongRun = await fixture(t, null, true);
  wrongRun.pollingCollection.runs[15].resultSha256 = "a".repeat(64);
  await wrongRun.writePolling();
  await assert.rejects(readHouseLiveAvailability(wrongRun.directory), /polling disclosure does not bind/);
});

test("selected native observation IDs are checked even when changed sidecar bytes have a new valid pin", async t => {
  const site = await fixture(t, null, true);
  site.pollingCollection.selected[0].observationId = "invented:incorrect-native-row";
  await site.writePolling();
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /questions do not match the selected native result/);
  const policyCutoff = await fixture(t, null, true);
  policyCutoff.pollingCollection.cutoffs.pollingAt = "2026-10-19T00:00:00Z";
  await policyCutoff.writePolling();
  await assert.rejects(loadHouseLiveSitePublication(policyCutoff.directory), /does not bind/);
});

test("an advertised polling sidecar or admission that is missing or altered is a broken release", async t => {
  const missing = await fixture(t, null, true);
  await rm(join(missing.directory, missing.manifest.pollingDisclosure.path));
  await assert.rejects(readHouseLiveAvailability(missing.directory), { code: "ENOENT" });
  const altered = await fixture(t, null, true);
  await appendFile(join(altered.directory, altered.manifest.pollingDisclosure.path), " ");
  await assert.rejects(loadHouseLiveSitePublication(altered.directory), /fingerprint mismatch/);
  const admission = await fixture(t, null, true);
  await rm(join(admission.directory, admission.results[1].qualification.evidence[0].path));
  await assert.rejects(loadHouseLiveSitePublication(admission.directory), { code: "ENOENT" });
  const invalidPin = await fixture(t, null, true);
  invalidPin.manifest.pollingDisclosure.path = "../outside.json";
  await invalidPin.writeManifest();
  await assert.rejects(loadHouseLiveSitePublication(invalidPin.directory), /Invalid live House site manifest/);
});

test("absence alone means unpublished; bad selections have a distinct error", async t => {
  const base = await mkdtemp(join(tmpdir(), "ballotmancy-live-absent-"));
  t.after(() => rm(base, { recursive: true, force: true }));
  assert.deepEqual(await readHouseLiveAvailability(join(base, "missing")), { available: false });
  const empty = join(base, "empty");
  await mkdir(empty);
  assert.deepEqual(await readHouseLiveAvailability(empty), { available: false });

  const danglingDirectory = join(base, "dangling-release");
  await symlink(join(base, "missing-target"), danglingDirectory);
  await assert.rejects(readHouseLiveAvailability(danglingDirectory), /ENOENT/);

  const site = await fixture(t);
  await assert.rejects(loadHouseLiveSitePublication(site.directory, "../escape"), HouseLiveSelectionError);
  await assert.rejects(loadHouseLiveSitePublication(site.directory, "unlisted-run"), HouseLiveSelectionError);
});

test("review decision, check gates, selected publication status, and joint binding are required", async t => {
  const unreviewed = await fixture(t, ({ results, catalog }) => {
    results.find(result => result.id === catalog.default_id).qualification.publication_status = "pending";
  });
  await assert.rejects(loadHouseLiveSitePublication(unreviewed.directory), /reviewed release/);

  const wrongId = await fixture(t, ({ results, catalog }) => {
    results.find(result => result.id === catalog.default_id).id = "synthetic-mismatched-native-id";
  });
  await assert.rejects(loadHouseLiveSitePublication(wrongId.directory), /reviewed release/);

  const site = await fixture(t);
  delete site.review.seat_quantile_convention;
  await site.writeReview();
  await assert.rejects(readHouseLiveAvailability(site.directory), /site review/);
  site.review.seat_quantile_convention = "choose_whichever_matches";
  await site.writeReview();
  await assert.rejects(readHouseLiveAvailability(site.directory), /site review/);
  site.review.seat_quantile_convention = "inverse_cdf";
  site.review.checks.jointRowsVerified = false;
  await site.writeReview();
  await assert.rejects(readHouseLiveAvailability(site.directory), /site review/);
  site.review.checks.jointRowsVerified = true;
  site.review.results.find(row => row.id === site.catalog.default_id).joint_draws_sha256 = "f".repeat(64);
  await site.writeReview();
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /reviewed release/);
});

test("all 16 catalog, manifest, and review result fingerprints must bind", async t => {
  const site = await fixture(t);
  site.manifest.results[0].artifact.sha256 = "e".repeat(64);
  await site.writeManifest();
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /does not bind catalog result/);

  site.manifest.results[0].artifact.sha256 = site.catalog.entries[0].result.sha256;
  site.review.results.pop();
  await site.writeReview();
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /site review/);
});

test("existing release with missing or altered artifacts fails readiness, including nondefault results", async t => {
  const site = await fixture(t);
  await rm(join(site.directory, site.manifest.geography.sourceReceipt.path));
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /ENOENT/);
  await assert.rejects(readHouseLiveAvailability(site.directory), /ENOENT/);

  const another = await fixture(t);
  const nondefault = another.manifest.results.find(row => row.id !== another.catalog.default_id);
  await rm(join(another.directory, nondefault.artifact.path));
  // A selected load authenticates its own native result; readiness authenticates all 16.
  await loadHouseLiveSitePublication(another.directory);
  await assert.rejects(readHouseLiveAvailability(another.directory), /ENOENT/);

  const altered = await fixture(t);
  await appendFile(join(altered.directory, altered.manifest.sourcePolicy.path), " ");
  await assert.rejects(loadHouseLiveSitePublication(altered.directory), /fingerprint mismatch/);
  const changedGeo = await fixture(t);
  await appendFile(join(changedGeo.directory, changedGeo.manifest.geography.artifact.path), " ");
  await assert.rejects(loadHouseLiveSitePublication(changedGeo.directory), /fingerprint mismatch/);
});

test("rejects path traversal and an escaped fixed publication manifest", async t => {
  const site = await fixture(t);
  site.manifest.results[0].artifact.path = "../outside.json";
  await site.writeManifest();
  await assert.rejects(loadHouseLiveSitePublication(site.directory), /result references/);

  const escaped = await fixture(t);
  const outside = join(escaped.base, "outside-publication.json");
  await writeFile(outside, await readFile(join(escaped.directory, "publication.json")));
  await rm(join(escaped.directory, "publication.json"));
  await symlink(outside, join(escaped.directory, "publication.json"));
  await assert.rejects(readHouseLiveAvailability(escaped.directory), /inside its publication directory/);
});
