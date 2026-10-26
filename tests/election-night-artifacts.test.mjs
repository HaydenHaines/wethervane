import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync, readFileSync, writeFileSync, chmodSync, symlinkSync, mkdirSync, unlinkSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { ElectionNightArtifactStore, electionNightArtifactSha256, loadElectionNightEvaluationLock,
  loadElectionNightResultLedger, assertLoadedEvaluationLock, assertLoadedResultLedger } from "../lib/election-night-artifacts.ts";

// INVENTED bytes, people, reviews, result observations and trust roots only.
// A caller must independently admit real publication/review manifests in production.
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
const h = c => c.repeat(64);
const unique = pins => [...new Map(pins.map(p => [p.sha256, p])).values()];
function setup(t) {
  const root = mkdtempSync(join(tmpdir(), "election-night-artifacts-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const store = new ElectionNightArtifactStore(root);
  const reviews = Object.fromEntries(["comparison", "cohort", "rubric", "resultPolicy", "projection"].map(role =>
    [role, store.putJson({ schema: "INVENTED-reviewed-evidence", role })]));
  const cohort = store.putJson({ schema: "INVENTED-fixed-cohort", raceIds: ["INVENTED-A", "INVENTED-B"] });
  const rubric = store.putJson({ schema: "INVENTED-rubric-policy" });
  const resultPolicy = store.putJson({ schema: "INVENTED-result-policy", operator: "INVENTED-final-pair" });
  const source = side => store.put(Buffer.from(`INVENTED ${side} original public source bytes, not normalized JSON`));
  const modelSource = source("model"), baselineSource = source("baseline");
  const binding = (side, pin) => ({ chamber: "senate", runId: `INVENTED-${side}`, resultSha256: pin.sha256, nativeForecastSha256: h(side === "model" ? "a" : "b"),
    targetId: "INVENTED-pair-target", lawId: "INVENTED-pair-law", issuedAt: "2026-10-20T15:25:00Z", cutoffAt: "2026-10-20T15:00:00Z", horizonCalendarDays: 14 });
  const bindings = { model: binding("model", modelSource), baseline: binding("baseline", baselineSource) };
  const race = id => ({ id, state: "INVENTED", seatId: id, electionLabel: "INVENTED election", ruleFamily: "plurality", method: "INVENTED pair proxy", notes: [], sourceUrls: [],
    outcomes: ["first", "second"].map((name, i) => ({ id: `${id}:${name}`, name, party: i ? "R" : "D", kind: "candidate", winProbability: i ? 0.4 : 0.6,
      members: [{ id: `${id}:${name}`, name, party: i ? "R" : "D", sourceUrls: [] }] })),
    pair: { firstCandidateId: `${id}:first`, secondCandidateId: `${id}:second`, firstCandidateName: "first", secondCandidateName: "second",
      denominator: "INVENTED named-pair votes", units: "percentage_points", orientation: "first_minus_second", meanMarginPp: 2, medianMarginPp: 2,
      range90: { lower: -10, upper: 10 }, firstExceedsHalf: 0.6, winnerRelationship: "same_event", closeWithin2Pp: null, firstAtLeast10Pp: null,
      secondAtLeast10Pp: null, qualifications: [] }, modalPick: { outcomeId: `${id}:first`, name: "first", kind: "candidate", probability: 0.6, tiedOutcomeIds: [`${id}:first`] } });
  const forecast = side => ({ schema: "ElectionNightForecastV1", chamber: "senate", editionStatus: "dated_snapshot",
    run: { id: bindings[side].runId, label: `INVENTED ${side}`, year: 2026, configuration: {}, artifactSha256: bindings[side].resultSha256,
      issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt, strictAsOf: true, sourceCutoffs: [], targetId: bindings[side].targetId, lawId: bindings[side].lawId },
    qualification: [], sourceUrls: [], sourceHashes: { forecastSha256: bindings[side].nativeForecastSha256 }, sourceNotes: [], inputNotices: [],
    races: [race("INVENTED-A"), race("INVENTED-B")], chamberScenario: null,
    summary: { raceCount: 2, pairRaceCount: 2, expectedModalMisses: 0.8, topTieRaceCount: 0, tiePolicy: "lexicographic_outcome_id", interpretation: "INVENTED" } });
  const modelForecastValue = forecast("model"), baselineForecastValue = forecast("baseline");
  const winnerTarget = id => ({ raceId: id, eventId: "INVENTED-election", roundId: "INVENTED-final-pair", resultRuleId: "INVENTED-rule", denominator: "INVENTED named-pair votes",
    estimand: "senate_principal_pair_lead", outcomeIds: [`${id}:first`, `${id}:second`], firstCandidateId: `${id}:first`, secondCandidateId: `${id}:second` });
  const winnerCohort = modelForecastValue.races.map(r => winnerTarget(r.id));
  const intervalCohort = winnerCohort.map(t => { const { outcomeIds, ...rest } = t; return { ...rest, estimand: "named_pair_margin", units: "percentage_points", orientation: "first_minus_second", intervalLevel: 0.9 }; });
  const contract = { schema: "ElectionNightRubricComparisonV1", id: "INVENTED-comparison", sha256: h("c"), rubricDecisionSha256: rubric.sha256,
    comparisonMode: "matched_horizon", label: "INVENTED locked comparison", model: bindings.model, baseline: bindings.baseline,
    comparisonReview: { id: "INVENTED-review", sha256: reviews.comparison.sha256 }, cohortDefinition: { id: "INVENTED-cohort", sha256: cohort.sha256 },
    resultPolicy: { id: "INVENTED-policy", sha256: resultPolicy.sha256 }, earlierBenchmarkPins: null, houseMappingReviewSha256: null,
    winnerCohort, intervalCohort, intervalExclusions: [] };
  const protocol = side => ({ schema: "ElectionNightEvaluationProtocolV1", id: `INVENTED-${side}-protocol`,
    forecast: { chamber: "senate", runId: bindings[side].runId, artifactSha256: bindings[side].resultSha256, issuedAt: bindings[side].issuedAt, cutoffAt: bindings[side].cutoffAt },
    reviewedDefinition: { sha256: reviews.resultPolicy.sha256, url: "https://example.invalid/INVENTED-review" },
    races: winnerCohort.map(t => ({ raceId: t.raceId, winner: { estimand: "conditional_principal_pair_winner", eventId: t.eventId,
      roundId: t.roundId, ruleFamily: "plurality", outcomeIds: [...t.outcomeIds], minimumStatus: "final" },
      margin: { estimand: "named_pair_margin", eventId: t.eventId, roundId: t.roundId, ruleFamily: "plurality", firstCandidateId: t.firstCandidateId,
        secondCandidateId: t.secondCandidateId, denominator: t.denominator, units: "percentage_points", orientation: "first_minus_second", pointRule: "mean", minimumStatus: "final" } })) });
  const artifacts = { modelSource, baselineSource, modelForecast: store.putJson(modelForecastValue), baselineForecast: store.putJson(baselineForecastValue),
    modelProtocol: store.putJson(protocol("model")), baselineProtocol: store.putJson(protocol("baseline")), comparisonContract: store.putJson(contract), cohort, rubric, resultPolicy };
  const manifest = { schema: "ElectionNightEvaluationLockV1", id: "INVENTED-lock", frozenAt: "2026-10-26T12:00:00Z", forecastBindings: bindings, artifacts, reviews,
    projections: { model: { sourceSha256: modelSource.sha256, projectionSha256: artifacts.modelForecast.sha256, reviewSha256: reviews.projection.sha256 },
      baseline: { sourceSha256: baselineSource.sha256, projectionSha256: artifacts.baselineForecast.sha256, reviewSha256: reviews.projection.sha256 } } };
  function authority(subject, scope, lockSha256, reviewPins, publicationPins, { omit, admissionChange, publicationChange } = {}) {
    const admission = { schema: "ElectionNightArtifactAdmissionV1", id: "INVENTED-trusted-admission", scope, subject, evaluationLockSha256: lockSha256, reviews: unique(reviewPins) };
    admissionChange?.(admission);
    const trustedAdmission = store.putJson(admission);
    const publication = { schema: "ElectionNightArtifactPublicationV1", id: "INVENTED-trusted-publication", artifacts: unique([subject, trustedAdmission, ...reviewPins, ...publicationPins]).filter(p => p.sha256 !== omit) };
    publicationChange?.(publication);
    return { trustedAdmission, trustedPublication: store.putJson(publication) };
  }
  function lockRequest(value = manifest, options) {
    const lock = store.putJson(value);
    return { lock, ...authority(lock, "evaluation_lock", lock.sha256, Object.values(value.reviews), [...Object.values(value.artifacts), ...Object.values(value.reviews)], options) };
  }
  const request = lockRequest(), loadedLock = loadElectionNightEvaluationLock(store, request);
  function revision({ id = "INVENTED-revision-1", raceId = "INVENTED-A", recordId = `INVENTED-result-${raceId}`, recordRevision = "1", predecessor = null,
    supersedes = null, lockSha256 = loadedLock.lockSha256, marginPp = 1, sourceText = "INVENTED immutable result source 1", publishedAt = "2026-11-05T00:00:00Z" } = {}) {
    const source = store.put(Buffer.from(sourceText)), evidence = store.putJson({ schema: "INVENTED-result-evidence", id });
    const recordValue = { schema: "ElectionNightOutcomeRecordV1", id: recordId, revision: recordRevision, raceId,
      source: { sha256: source.sha256, url: "https://example.invalid/INVENTED-official-result", publishedAt, retrievedAt: "2026-11-07T00:00:00Z" },
      winner: { status: "final", estimand: "conditional_principal_pair_winner", eventId: "INVENTED-election", roundId: "INVENTED-final-pair", ruleFamily: "plurality",
        outcomeId: `${raceId}:first`, kind: "candidate", candidateId: `${raceId}:first`, reason: null },
      margin: { status: "final", estimand: "named_pair_margin", eventId: "INVENTED-election", roundId: "INVENTED-final-pair", ruleFamily: "plurality",
        firstCandidateId: `${raceId}:first`, secondCandidateId: `${raceId}:second`, denominator: "INVENTED named-pair votes", units: "percentage_points", orientation: "first_minus_second", marginPp, reason: null } };
    const record = store.putJson(recordValue), revisionValue = { schema: "ElectionNightResultRevisionV1", id, ledgerId: "INVENTED-ledger", evaluationLockSha256: lockSha256,
      predecessorSha256: predecessor, supersedesRecordSha256: supersedes, record, source, evidence: [evidence] };
    return { pin: store.putJson(revisionValue), manifest: revisionValue, record, recordValue, source, evidence };
  }
  function ledgerRequest(revisions, { head, ledgerChange, ...options } = {}) {
    const value = { schema: "ElectionNightResultLedgerV1", id: "INVENTED-ledger", evaluationLockSha256: loadedLock.lockSha256,
      chosenHeadSha256: head === undefined ? revisions.at(-1)?.pin.sha256 ?? null : head, revisions: revisions.map(r => r.pin) };
    ledgerChange?.(value);
    const ledger = store.putJson(value), extras = revisions.flatMap(r => [r.pin, r.record, r.source, r.evidence]);
    return { evaluationLock: loadedLock, ledger, ...authority(ledger, "result_ledger", loadedLock.lockSha256, [reviews.resultPolicy], extras, options) };
  }
  return { root, store, manifest, request, loadedLock, artifacts, reviews, contract, authority, lockRequest, revision, ledgerRequest };
}

test("store hashes actual bytes, canonical JSON is stable and manifests/blobs are write-once", t => {
  const f = setup(t), input = Buffer.from("INVENTED exact bytes\n"), before = Buffer.from(input), first = f.store.put(input);
  assert.equal(first.sha256, sha(input)); assert.equal(first.sha256, electionNightArtifactSha256(input)); assert.equal(first.byteLength, input.byteLength);
  assert.equal(first.path, `blobs/sha256/${sha(input)}`); assert.deepEqual(f.store.read(first), input); assert.deepEqual(input, before);
  const originalStat = readFileSync(join(f.root, first.path));
  assert.deepEqual(f.store.put(input), first); assert.deepEqual(readFileSync(join(f.root, first.path)), originalStat);
  const different = f.store.put(Buffer.from("INVENTED different bytes")); assert.notEqual(different.sha256, first.sha256); assert.deepEqual(f.store.read(first), before);
  assert.deepEqual(f.store.putJson({ z: [1, null], a: "INVENTED" }), f.store.putJson({ a: "INVENTED", z: [1, null] }));
  const json = f.store.readJson(f.store.putJson({ nested: { array: [1, 2] } })); assert.ok(Object.isFrozen(json.nested.array));
  assert.throws(() => json.nested.array.push(3)); assert.throws(() => f.store.putJson({ bad: Number.NaN }));
  assert.throws(() => f.store.putJson({ bad: undefined })); assert.throws(() => f.store.putJson(new Date()));
  const cyclic = {}; cyclic.self = cyclic; assert.throws(() => f.store.putJson(cyclic));
  const sparse = new Array(1); sparse.extra = 1; assert.throws(() => f.store.putJson(sparse), /Sparse/);
  assert.equal(readdirSync(join(f.root, "blobs/sha256")).some(name => name.startsWith(".incoming")), false);
});

test("corrupt/missing bytes, incorrect digest or size fail without overwriting prior blobs", t => {
  const f = setup(t), bytes = Buffer.from("INVENTED immutable blob"), pin = f.store.put(bytes), file = join(f.root, pin.path);
  const copy = f.store.read(pin); copy.fill(0); assert.deepEqual(f.store.read(pin), bytes);
  assert.throws(() => f.store.read({ ...pin, byteLength: pin.byteLength + 1 }), /Corrupt/);
  chmodSync(file, 0o644); writeFileSync(file, Buffer.from("INVENTED corrupted bytes"));
  assert.throws(() => f.store.read(pin), /Corrupt/); assert.throws(() => f.store.put(bytes), /Corrupt/);
  assert.equal(readFileSync(file, "utf8"), "INVENTED corrupted bytes");
  const missing = { sha256: h("f"), byteLength: 1, path: `blobs/sha256/${h("f")}` }; assert.throws(() => f.store.read(missing), /ENOENT/);
  const malformed = f.store.put(Buffer.from([0x7b, 0x22, 0x61, 0x22, 0x3a, 0x22, 0xff, 0x22, 0x7d])); assert.throws(() => f.store.readJson(malformed));
});

test("relative traversal, absolute paths, backslashes and symlink escapes are rejected", t => {
  const f = setup(t), pin = f.store.put(Buffer.from("INVENTED safe"));
  for (const path of ["../outside", "/etc/passwd", "blobs/../outside", "blobs\\sha256\\x", `blobs/sha256/${pin.sha256}/..`]) assert.throws(() => f.store.read({ ...pin, path }));
  const outside = mkdtempSync(join(tmpdir(), "election-night-outside-")); t.after(() => rmSync(outside, { recursive: true, force: true }));
  const outsideFile = join(outside, "blob"); writeFileSync(outsideFile, "INVENTED safe");
  const path = join(f.root, pin.path); unlinkSync(path); symlinkSync(outsideFile, path); assert.throws(() => f.store.read(pin), /symlink|regular/);
  assert.throws(() => f.store.put(Buffer.from("INVENTED safe")), /symlink|regular/); assert.equal(readFileSync(outsideFile, "utf8"), "INVENTED safe");
  const rootLink = join(outside, "store-link"); symlinkSync(f.root, rootLink); assert.throws(() => new ElectionNightArtifactStore(rootLink), /symlink/);
  const other = mkdtempSync(join(tmpdir(), "election-night-parent-")); t.after(() => rmSync(other, { recursive: true, force: true }));
  mkdirSync(join(other, "blobs")); symlinkSync(outside, join(other, "blobs", "sha256"));
  assert.throws(() => new ElectionNightArtifactStore(other).put(Buffer.from("INVENTED new")), /real in-store/);
});

test("authenticated lock distinguishes original public and projection bytes, preserving exact identities immutably", t => {
  const f = setup(t), lock = f.loadedLock;
  assert.doesNotThrow(() => assertLoadedEvaluationLock(lock));
  assert.equal(lock.modelForecast.run.artifactSha256, f.artifacts.modelSource.sha256);
  assert.notEqual(f.artifacts.modelSource.sha256, f.artifacts.modelForecast.sha256);
  assert.equal(lock.manifest.projections.model.projectionSha256, f.artifacts.modelForecast.sha256);
  assert.equal(lock.comparisonContract.sha256, h("c")); assert.notEqual(lock.comparisonContract.sha256, f.artifacts.comparisonContract.sha256);
  assert.equal(lock.lockSha256, sha(f.store.read(f.request.lock))); assert.equal(lock.publicationSha256, f.request.trustedPublication.sha256);
  assert.ok(Object.isFrozen(lock.manifest.forecastBindings.model)); assert.ok(Object.isFrozen(lock.modelForecast.races[0].outcomes));
  assert.throws(() => lock.modelForecast.races[0].outcomes[0].winProbability = 0);
  assert.throws(() => assertLoadedEvaluationLock({ ...lock })); assert.throws(() => assertLoadedEvaluationLock({ verified: true, admitted: true }));
});

test("lock loading rejects missing/untrusted/corrupt roles and mismatched admission without self-approval shortcuts", t => {
  const f = setup(t);
  const changes = [
    m => m.forecastBindings.model.resultSha256 = h("9"),
    m => m.projections.model.sourceSha256 = m.artifacts.modelForecast.sha256,
    m => m.projections.model.projectionSha256 = m.artifacts.modelSource.sha256,
    m => m.projections.model.reviewSha256 = h("9"),
    m => m.artifacts.cohort = m.artifacts.resultPolicy,
    m => m.artifacts.modelProtocol = m.artifacts.baselineProtocol,
    m => m.artifacts.modelForecast = m.artifacts.baselineForecast,
    m => m.reviews.comparison = m.reviews.cohort,
    m => m.artifacts.extra = m.artifacts.rubric,
  ];
  for (const change of changes) { const m = structuredClone(f.manifest); change(m); assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(m))); }
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(f.manifest, { omit: f.artifacts.modelSource.sha256 })), /Unpublished/);
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(f.manifest, { admissionChange: a => a.subject = f.artifacts.modelSource })), /does not authorize/);
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(f.manifest, { admissionChange: a => a.scope = "result_ledger" })), /does not authorize/);
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(f.manifest, { admissionChange: a => a.reviews.pop() })), /review bytes differ/);
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(f.manifest, { publicationChange: p => p.artifacts.push(p.artifacts[0]) })), /Duplicate/);
  const selfClaim = { lock: f.request.lock, verified: true, admitted: true }; assert.throws(() => loadElectionNightEvaluationLock(f.store, selfClaim));
  const file = join(f.root, f.artifacts.baselineSource.path); chmodSync(file, 0o644); writeFileSync(file, "INVENTED corrupted original source");
  assert.throws(() => loadElectionNightEvaluationLock(f.store, f.request), /Corrupt/);
});

test("result ledger uses only its explicit pinned head, preserves corrections and never orders by timestamps", t => {
  const f = setup(t), first = f.revision(), second = f.revision({ id: "INVENTED-revision-2", raceId: "INVENTED-B", predecessor: first.pin.sha256 });
  const corrected = f.revision({ id: "INVENTED-correction", recordRevision: "2", predecessor: second.pin.sha256, supersedes: first.record.sha256,
    marginPp: -3, sourceText: "INVENTED corrected source", publishedAt: "2026-11-04T00:00:00Z" });
  const request = f.ledgerRequest([first, second, corrected]), ledger = loadElectionNightResultLedger(f.store, request);
  assert.doesNotThrow(() => assertLoadedResultLedger(ledger)); assert.equal(ledger.history.length, 3); assert.equal(ledger.records.length, 2);
  assert.equal(ledger.records[0].revision, "2"); assert.equal(ledger.records[0].margin.marginPp, -3);
  assert.equal(ledger.history[0].record.margin.marginPp, 1); assert.equal(ledger.chosenHeadSha256, corrected.pin.sha256);
  const provenance = ledger.provenanceByRecordId[corrected.recordValue.id];
  assert.equal(provenance.recordSha256, corrected.record.sha256); assert.equal(provenance.sourceSha256, corrected.source.sha256);
  assert.equal(provenance.resultPolicySha256, f.artifacts.resultPolicy.sha256); assert.equal(provenance.revision, "2");
  assert.deepEqual(provenance.evidenceSha256, [corrected.evidence.sha256]); assert.equal(ledger.lockSha256, f.loadedLock.lockSha256);
  assert.ok(Object.isFrozen(ledger.records[0].margin)); assert.throws(() => ledger.records[0].margin.marginPp = 99);
  assert.throws(() => assertLoadedResultLedger({ ...ledger })); assert.throws(() => assertLoadedResultLedger({ authenticated: true }));
  const older = loadElectionNightResultLedger(f.store, f.ledgerRequest([first]));
  assert.equal(older.records[0].revision, "1"); assert.equal(older.history.length, 1); assert.equal(older.records.length, 1);
  assert.equal(older.manifest.revisions.length, 1); // Missing B remains missing for downstream unresolved coverage.
  assert.equal(ledger.records[0].revision, "2"); // Loading a historical head does not mutate newer history.
});

test("an admitted empty pre-result ledger selects null head and returns no placeholder observation", t => {
  const f = setup(t), request = f.ledgerRequest([]), ledger = loadElectionNightResultLedger(f.store, request);
  assert.equal(ledger.chosenHeadSha256, null);
  assert.deepEqual(ledger.manifest.revisions, []);
  assert.deepEqual(ledger.records, []);
  assert.deepEqual(ledger.history, []);
  assert.deepEqual(Object.keys(ledger.provenanceByRecordId), []);
  assert.equal(ledger.admissionSha256, request.trustedAdmission.sha256);
  assert.equal(ledger.publicationSha256, request.trustedPublication.sha256);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([], { head: f.request.lock.sha256 })), /Chosen head/);
  const first = f.revision();
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { head: null })), /Chosen head/);
});

test("missing/duplicate/branched/global or record revisions fail closed, even if their bytes were published", t => {
  const f = setup(t), first = f.revision();
  const branch = f.revision({ id: "INVENTED-branch", recordRevision: "2", predecessor: null, supersedes: first.record.sha256 });
  const missing = f.revision({ id: "INVENTED-missing-predecessor", recordRevision: "2", predecessor: h("9"), supersedes: first.record.sha256 });
  const duplicateRevision = f.revision({ id: "INVENTED-same-record-revision", predecessor: first.pin.sha256, supersedes: first.record.sha256, sourceText: "INVENTED changed source under same record revision" });
  const badRecordPredecessor = f.revision({ id: "INVENTED-wrong-record-predecessor", recordRevision: "2", predecessor: first.pin.sha256, supersedes: h("9") });
  const changedRecordIdentity = f.revision({ id: "INVENTED-changed-record-ID", recordId: "INVENTED-new-ID-same-race", predecessor: first.pin.sha256 });
  const wrongLock = f.revision({ id: "INVENTED-wrong-lock", predecessor: first.pin.sha256, lockSha256: h("9") });
  const outside = f.revision({ id: "INVENTED-outside-cohort", raceId: "INVENTED-outside", predecessor: first.pin.sha256 });
  for (const second of [branch, missing, duplicateRevision, badRecordPredecessor, changedRecordIdentity, wrongLock, outside])
    assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first, second])));
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first, first])), /Duplicate/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first, branch], { head: first.pin.sha256 })), /Chosen head/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { ledgerChange: l => l.revisions = [] })), /Chosen head/);
  const repeatedId = f.revision({ id: first.manifest.id, raceId: "INVENTED-B", predecessor: first.pin.sha256 });
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first, repeatedId])), /Duplicate/);
});

test("result source/evidence bytes, result-policy admission and all historical revisions are verified", t => {
  const f = setup(t), first = f.revision();
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { omit: first.source.sha256 })), /Unpublished/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { omit: first.evidence.sha256 })), /Unpublished/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { omit: first.record.sha256 })), /Unpublished/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { admissionChange: a => a.evaluationLockSha256 = h("9") })), /does not authorize/);
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([first], { admissionChange: a => a.reviews = [f.reviews.comparison], publicationChange: p => p.artifacts.push(f.reviews.comparison) })), /result-policy review/);
  const selfClaim = f.ledgerRequest([first]); selfClaim.evaluationLock = { ...f.loadedLock, admitted: true }; assert.throws(() => loadElectionNightResultLedger(f.store, selfClaim), /authenticated loader/);
  const correction = f.revision({ id: "INVENTED-correction", recordRevision: "2", predecessor: first.pin.sha256, supersedes: first.record.sha256, sourceText: "INVENTED replacement source" });
  const request = f.ledgerRequest([first, correction]), sourcePath = join(f.root, first.source.path);
  chmodSync(sourcePath, 0o644); writeFileSync(sourcePath, "INVENTED corrupt superseded source");
  assert.throws(() => loadElectionNightResultLedger(f.store, request), /Corrupt/); // Older evidence is not skipped at the chosen head.
});


test("an explicit predeclared scoring subset preserves the complete protocol/display cohort", t => {
  const f = setup(t), c = structuredClone(f.contract);
  c.displayRaceIds = ["INVENTED-A", "INVENTED-B"]; c.ungradedDisplayRaceIds = ["INVENTED-B"];
  c.winnerCohort.pop(); c.intervalCohort.pop();
  const m = structuredClone(f.manifest); m.artifacts.comparisonContract = f.store.putJson(c);
  const lock = loadElectionNightEvaluationLock(f.store, f.lockRequest(m));
  assert.equal(lock.modelForecast.races.length, 2); assert.equal(lock.modelProtocol.races.length, 2);
  assert.equal(lock.comparisonContract.winnerCohort.length, 1); assert.deepEqual(lock.comparisonContract.ungradedDisplayRaceIds, ["INVENTED-B"]);
  for (const change of [c => delete c.displayRaceIds, c => c.ungradedDisplayRaceIds = [], c => c.displayRaceIds.reverse(), c => c.winnerCohort[0].raceId = "INVENTED-outside"])
    { const bad = structuredClone(c); change(bad); const m2 = structuredClone(m); m2.artifacts.comparisonContract = f.store.putJson(bad); assert.throws(() => loadElectionNightEvaluationLock(f.store, f.lockRequest(m2))); }
});


test("source pin mismatches, missing physical history and corrupt evidence cannot enter loaded records", t => {
  const f = setup(t), first = f.revision();
  const otherSource = f.store.put(Buffer.from("INVENTED other source bytes"));
  const swappedValue = structuredClone(first.manifest); swappedValue.source = otherSource;
  const swapped = { ...first, pin: f.store.putJson(swappedValue), source: otherSource };
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([swapped])), /source-byte provenance/);
  const noEvidenceValue = structuredClone(first.manifest); noEvidenceValue.evidence = [];
  const noEvidence = { ...first, pin: f.store.putJson(noEvidenceValue) };
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([noEvidence])), /Invalid result evidence/);
  const duplicateEvidenceValue = structuredClone(first.manifest); duplicateEvidenceValue.evidence.push(first.evidence);
  const duplicateEvidence = { ...first, pin: f.store.putJson(duplicateEvidenceValue) };
  assert.throws(() => loadElectionNightResultLedger(f.store, f.ledgerRequest([duplicateEvidence])), /Duplicate result evidence/);
  const request = f.ledgerRequest([first]);
  const evidencePath = join(f.root, first.evidence.path); chmodSync(evidencePath, 0o644); writeFileSync(evidencePath, "INVENTED corrupted evidence");
  assert.throws(() => loadElectionNightResultLedger(f.store, request), /Corrupt/);
  unlinkSync(join(f.root, first.pin.path)); assert.throws(() => loadElectionNightResultLedger(f.store, request), /ENOENT/);
});
