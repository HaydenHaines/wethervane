import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { loadConfiguredElectionNightEvaluation } from "../lib/election-night-evaluation-server.ts";
import { parseElectionNightEvaluationView } from "../lib/election-night-evaluation-view.ts";
import { GET } from "../app/api/election-night/evaluation/route.ts";
import { makeElectionNightEvaluationFixture } from "./fixtures/election-night-evaluation-server.mjs";
import { makeElectionNightEvaluationViewFixtures } from "./fixtures/election-night-evaluation-views.mjs";

const H = char => char.repeat(64);
function temp(t) {
  const root = mkdtempSync(join(tmpdir(), "election-night-evaluation-api-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return root;
}
function writeConfig(dataRoot, config) {
  const directory = join(dataRoot, "election-night"); mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "evaluation-publication.json"), JSON.stringify(config));
}
function configFor(fixture, ledger) {
  return { schema: "ElectionNightEvaluationPublicationConfigV1", chambers: { [fixture.chamber]: {
    lock: { subject: fixture.lockPin, publication: fixture.lockRequest.trustedPublication, admission: fixture.lockRequest.trustedAdmission },
    ledger,
  } } };
}
function ledgerPins(fixture) {
  const req = fixture.makeLedger();
  return { request: req, pins: { subject: req.ledger, publication: req.trustedPublication, admission: req.trustedAdmission } };
}

test("missing publication config is an explicit no-lock state, and route accepts only a chamber selector", async t => {
  const dataRoot = temp(t);
  const view = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(view.status, "lock_not_configured"); assert.equal(view.lock, null); assert.equal(view.rubric, null);
  assert.equal(view.rows.length, 0); assert.equal(parseElectionNightEvaluationView(view, "senate")?.status, "lock_not_configured");
  const invalid = await GET(new Request("https://example.test/api/election-night/evaluation?chamber=senate&lock=../../secret"));
  assert.equal(invalid.status, 400); assert.doesNotMatch(await invalid.text(), /secret|path|sha/i);
  const badChamber = await GET(new Request("https://example.test/api/election-night/evaluation?chamber=senate&chamber=house"));
  assert.equal(badChamber.status, 400);
});

test("malformed or corrupted server trust configuration is unavailable without leaking filesystem details", async t => {
  const dataRoot = temp(t); writeConfig(dataRoot, { schema: "wrong", storeRoot: "/secret/path" });
  const malformed = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(malformed.status, "unavailable"); assert.equal(malformed.errorCode, "configuration_invalid");
  assert.deepEqual(malformed.rows, []); assert.equal(malformed.rubric, null);
  const fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  const ledger = ledgerPins(fixture);
  const corrupted = structuredClone(configFor(fixture, ledger.pins));
  corrupted.chambers.senate.lock.admission.sha256 = H("f");
  corrupted.chambers.senate.lock.admission.path = `blobs/sha256/${H("f")}`;
  writeConfig(dataRoot, corrupted);
  const failed = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(failed.status, "unavailable"); assert.equal(failed.errorCode, "lock_verification_failed");
  assert.doesNotMatch(JSON.stringify(failed), /secret|evaluation-artifacts|tmp/);
  assert.equal(parseElectionNightEvaluationView(failed, "senate")?.status, "unavailable");

  const brokenLedger = ledgerPins(fixture), badLedger = structuredClone(configFor(fixture, brokenLedger.pins));
  badLedger.chambers.senate.ledger.admission.sha256 = H("e");
  badLedger.chambers.senate.ledger.admission.path = `blobs/sha256/${H("e")}`;
  writeConfig(dataRoot, badLedger);
  const ledgerFailed = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(ledgerFailed.status, "unavailable"); assert.equal(ledgerFailed.errorCode, "ledger_verification_failed");
  assert.equal(ledgerFailed.lock, null); assert.equal(ledgerFailed.rows.length, 0);
  assert.equal(parseElectionNightEvaluationView(ledgerFailed, "senate")?.status, "unavailable");
});

test("an authenticated empty ledger is locked but has no results, scores or grade", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(view.status, "locked_no_results"); assert.equal(view.lock?.sha256, fixture.lockPin.sha256);
  assert.equal(view.ledger?.recordCount, 0); assert.equal(view.ledger?.chosenHeadSha256, null);
  assert.equal(view.rows.length, 2); assert.equal(view.coverage?.winner.pending, 2);
  assert.equal(view.availableMatchedRows?.winner.count, 0); assert.equal(view.rubricStatus, "not_graded"); assert.equal(view.rubric, null);
  assert.equal(view.rows[0].forecast.modalPick.name, "INVENTED First Candidate");
  assert.equal(parseElectionNightEvaluationView(view, "senate")?.status, "locked_no_results");
});

test("native House null target/law identities survive the authenticated lock and view boundary", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"),
    { chamber: "house", count: 1 });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("house", dataRoot);
  assert.equal(view.status, "locked_no_results");
  assert.equal(view.lock?.model.targetId, null); assert.equal(view.lock?.model.lawId, null);
  assert.equal(view.lock?.baseline.targetId, null); assert.equal(view.lock?.baseline.lawId, null);
  assert.equal(view.rows[0].forecast.seatId, "INVENTED-HOUSE-race-1");
  assert.equal(view.rubric, null);
  assert.equal(parseElectionNightEvaluationView(view, "house")?.status, "locked_no_results");
});

test("House group winner stays distinct from candidate-pair margin in a scored synthetic result", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"),
    { chamber: "house", count: 1 });
  fixture.addRecord({ index: 0, winnerStatus: "certified", marginStatus: "certified", marginPp: 11 });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("house", dataRoot);
  const row = view.rows[0];
  assert.equal(row.winner.model.status, "scored");
  assert.equal(row.winner.model.observedValueKind, "party_group");
  assert.equal(row.winner.model.observedValue, "D");
  assert.equal(row.winner.model.observation?.value, "D");
  assert.equal(row.winner.model.observation?.label, "Democratic party group");
  assert.equal(row.winner.model.modalPickCorrect, true);
  assert.equal(row.margin.model.status, "scored");
  assert.equal(row.margin.model.observedValueKind, null);
  assert.equal(row.margin.model.observation?.value, 11);
  assert.deepEqual(row.margin.model.marginScore, { pointRule: "mean", predictedMarginPp: 2,
    observedMarginPp: 11, absoluteErrorPp: 9, interval90: { lower: -5, upper: 9, covered: false } });
  assert.equal(row.forecast.modalPick.kind, "group");
  assert.notEqual(row.winner.model.observedValue, row.forecast.modalPick.name);
  assert.equal(parseElectionNightEvaluationView(view, "house")?.status, "graded");
});

test("certified Senate race outside the fixed scoring cohort remains visible and cannot enter grade denominators", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"),
    { chamber: "senate", count: 3, displayOnlyCount: 1 });
  fixture.addRecord({ index: 2, winnerStatus: "certified", marginStatus: "certified", marginPp: 11 });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(view.status, "partial_results"); assert.equal(view.rows.length, 3);
  assert.equal(view.lock?.cohort.displayRaceIds.length, 3); assert.equal(view.lock?.cohort.winnerRaceIds.length, 2);
  assert.equal(view.coverage?.winner.total, 2); assert.equal(view.coverage?.winner.pending, 2);
  assert.equal(view.coverage?.interval.total, 2); assert.equal(view.availableMatchedRows?.winner.count, 0);
  const excluded = view.rows[2];
  assert.equal(excluded.displayOnly, true);
  assert.equal(excluded.winner.model.status, "not_in_grade_cohort");
  assert.equal(excluded.winner.model.loss, null);
  assert.equal(excluded.winner.model.observation?.status, "certified");
  assert.equal(excluded.winner.model.modalPickCorrect, true);
  assert.equal(excluded.winner.model.diagnosticScope, "display_only");
  assert.equal(excluded.margin.model.status, "not_in_grade_cohort");
  assert.deepEqual(excluded.margin.model.marginScore, { pointRule: "mean", predictedMarginPp: 2,
    observedMarginPp: 11, absoluteErrorPp: 9, interval90: { lower: -5, upper: 9, covered: false } });
  assert.equal(excluded.margin.model.diagnosticScope, "display_only");
  assert.equal(parseElectionNightEvaluationView(view, "senate")?.status, "partial_results");
});

test("partial certified results stay paired to both forecasts while absent rows stay pending and no letter leaks", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  fixture.addRecord({ index: 0, winnerStatus: "certified", marginStatus: "certified", marginPp: -7 });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(view.status, "partial_results"); assert.equal(view.rubric, null); assert.equal(view.rubricStatus, "not_graded");
  assert.equal(view.lock?.cohort.winnerRaceIds.length, 2);
  assert.equal(view.coverage?.winner.total, 2); assert.equal(view.coverage?.winner.final, 1); assert.equal(view.coverage?.winner.pending, 1);
  assert.equal(view.availableMatchedRows?.winner.count, 1); assert.equal(view.rows[0].winner.model.finality, "certified");
  assert.equal(view.rows[0].winner.model.observation?.status, "certified");
  assert.equal(view.rows[0].winner.model.observedValueKind, "candidate");
  assert.equal(view.rows[0].winner.model.modalPickCorrect, true);
  assert.deepEqual(view.rows[0].margin.model.marginScore, { pointRule: "mean", predictedMarginPp: 2,
    observedMarginPp: -7, absoluteErrorPp: 9, interval90: { lower: -5, upper: 9, covered: false } });
  assert.equal(view.rows[1].winner.model.finality, "unreported"); assert.equal(view.rows[1].winner.model.loss, null);
  assert.equal(view.ledger?.provenance[0].sourceSha256, fixture.records[0].source.sha256);
  assert.equal(parseElectionNightEvaluationView(view, "senate")?.status, "partial_results");
});

test("called provisional observations are labeled separately from scoring losses", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  fixture.addRecord({ index: 0, winnerStatus: "called", marginStatus: "called" });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const view = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(view.status, "partial_results");
  assert.equal(view.rows[0].winner.model.finality, "called");
  assert.equal(view.rows[0].winner.model.observation?.status, "called");
  assert.equal(view.rows[0].winner.model.loss, null); assert.equal(view.rows[0].margin.model.loss, null);
  assert.equal(view.coverage?.winner.provisional, 0);
  assert.equal(view.availableMatchedRows?.winner.count, 0);
});

test("complete fixed synthetic cohort grades only through the frozen grader and correction selects a new immutable head", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  fixture.addRecord({ index: 0, marginPp: -7 }); fixture.addRecord({ index: 1, marginPp: 3 });
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const complete = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(complete.status, "graded"); assert.equal(complete.rubricStatus, "graded");
  assert.ok(["A", "B", "C", "D", "F"].includes(complete.rubric?.overall.grade));
  assert.equal(complete.coverage?.winner.final, complete.lock?.cohort.winnerRaceIds.length);
  assert.equal(complete.availableMatchedRows?.interval.count, 2);
  assert.equal(parseElectionNightEvaluationView(complete, "senate")?.status, "graded");

  fixture.addRecord({ index: 0, revision: "2", marginPp: 9 });
  const correctedLedger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, correctedLedger.pins));
  const corrected = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  assert.equal(corrected.status, "graded");
  assert.equal(corrected.ledger?.revisionCount, 3); assert.equal(corrected.ledger?.recordCount, 2);
  assert.notEqual(corrected.ledger?.chosenHeadSha256, complete.ledger?.chosenHeadSha256);
  assert.equal(corrected.rows[0].margin.model.observation?.value, 9);
  assert.equal(corrected.rows[0].sources[0].revision, "2");
});

test("browser parser rejects fabricated grades, incomplete coverage, numeric dates and policy substitution", async t => {
  const dataRoot = temp(t), fixture = makeElectionNightEvaluationFixture(join(dataRoot, "election-night", "evaluation-artifacts"));
  const ledger = ledgerPins(fixture); writeConfig(dataRoot, configFor(fixture, ledger.pins));
  const empty = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  const forged = structuredClone(empty); forged.status = "graded"; forged.rubricStatus = "graded";
  forged.rubric = { policyId: "ElectionNightRubricEditorialDecisionV3", policyDecisionSha256: H("f"), status: "graded", overall: { grade: "A" } };
  assert.equal(parseElectionNightEvaluationView(forged), null);
  const wrongCoverage = structuredClone(empty); wrongCoverage.coverage.winner.pending = 1;
  assert.equal(parseElectionNightEvaluationView(wrongCoverage), null);
  const numericDate = structuredClone(empty); numericDate.lock.model.issuedAt = 1793538000000;
  assert.equal(parseElectionNightEvaluationView(numericDate), null);
  const wrongChamber = structuredClone(empty); assert.equal(parseElectionNightEvaluationView(wrongChamber, "house"), null);
});

test("shared browser fixture set is valid and contains only invented no-lock, pending, partial and graded states", async t => {
  const fixture = await makeElectionNightEvaluationViewFixtures(); t.after(fixture.cleanup);
  assert.equal(fixture.views.noLock.status, "lock_not_configured");
  assert.equal(fixture.views.lockedNoResults.status, "locked_no_results");
  assert.equal(fixture.views.called.status, "partial_results");
  assert.equal(fixture.views.called.rows[0].winner.model.finality, "called");
  assert.equal(fixture.views.called.rubric, null);
  assert.equal(fixture.views.corrected.ledger.revisionCount, 2);
  assert.equal(fixture.views.corrected.rows[0].winner.model.finality, "certified");
  assert.equal(fixture.views.graded.status, "graded");
  assert.ok(fixture.views.graded.rows.every(row => row.forecast.seatId.startsWith("INVENTED-")));
  for (const name of ["noLock", "lockedNoResults", "called", "corrected", "graded"]) {
    const stored = JSON.parse(readFileSync(new URL(`./fixtures/election-night/${name}.json`, import.meta.url), "utf8"));
    assert.ok(parseElectionNightEvaluationView(stored, "senate"), `static ${name} fixture parses`);
  }
});
