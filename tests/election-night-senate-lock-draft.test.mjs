import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ElectionNightArtifactStore } from "../lib/election-night-artifacts.ts";
import { ELECTION_NIGHT_RUBRIC, gradeElectionNightRubric } from "../lib/election-night-rubric.ts";
import { prepareSenateLockDraft, readPinnedFile } from "../scripts/prepare_election_night_senate_lock.mjs";

const ownerAvailable = existsSync('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/policy-v1.json');

function stage(t) {
  const outputDir = mkdtempSync(join(tmpdir(), "senate-lock-draft-"));
  t.after(() => rmSync(outputDir, { recursive: true, force: true }));
  return { outputDir, draft: prepareSenateLockDraft({ outputDir }), store: new ElectionNightArtifactStore(outputDir) };
}

test("exact public F/FP forecasts and owner evidence yield a reproducible non-admitted artifact graph", { skip: !ownerAvailable }, t => {
  const a = stage(t), b = stage(t);
  assert.deepEqual(readFileSync(join(a.outputDir, "draft.json")), readFileSync(join(b.outputDir, "draft.json")));
  const d = a.draft, pins = d.lockArtifactCandidates;
  assert.equal(d.status, "awaiting_independent_review_and_admission");
  assert.equal(d.admissionIssued, false); assert.equal(d.actualResultsUsed, false); assert.equal(d.actualGradeAssigned, false);
  assert.equal(pins.modelSource.sha256, ELECTION_NIGHT_RUBRIC.earlierBenchmark.model.resultSha256);
  assert.equal(pins.baselineSource.sha256, ELECTION_NIGHT_RUBRIC.earlierBenchmark.baseline.resultSha256);
  assert.notEqual(pins.modelSource.sha256, pins.modelForecast.sha256);
  assert.notEqual(pins.baselineSource.sha256, pins.baselineForecast.sha256);
  assert.equal(pins.rubric.sha256, ELECTION_NIGHT_RUBRIC.decisionSha256);
  assert.equal(pins.resultPolicy.sha256, ELECTION_NIGHT_RUBRIC.earlierBenchmark.resultPolicySha256);
  assert.equal(d.ownerEvidence.coreReferenceCatalog.sha256, "8a63d2d3f8490b991a3da6b92ad686d2f9a5d0eb3dd03d51f0c644d8d3e16407");
  assert.equal(d.ownerEvidence.preservedFull35ReferenceCatalog.sha256, "9d2c289667d9f25fe38515eefb1503133a6f72b92cfcf65e5061540b235ccd5f");
  assert.notEqual(d.fixed32ReferenceBindings.model.jointDrawsSha256, d.fixed32ReferenceBindings.baseline.jointDrawsSha256);
  assert.equal(d.candidateLock.frozenAt, null); assert.equal(d.candidateLock.reviews, null);
  assert.equal(d.candidateLock.projections.model.reviewSha256, null);
  assert.equal(d.emptyLedgerProposal.evaluationLockSha256, null);
  assert.equal(d.emptyLedgerProposal.chosenHeadSha256, null);
  assert.deepEqual(d.emptyLedgerProposal.revisions, []);
  const ledgerEvidence = JSON.parse(readFileSync(join(a.outputDir, "empty-ledger-evidence.json")));
  assert.deepEqual(readFileSync(join(a.outputDir, "empty-ledger-evidence.json")),
    readFileSync(join(b.outputDir, "empty-ledger-evidence.json")));
  assert.deepEqual(ledgerEvidence.ledger, d.emptyLedgerProposal);
  assert.equal(ledgerEvidence.evidencePins.resultPolicy.sha256, pins.resultPolicy.sha256);
  assert.equal(ledgerEvidence.evidencePins.cohort.sha256, pins.cohort.sha256);
  assert.deepEqual(ledgerEvidence.actualSourcePins, []);
  assert.equal(ledgerEvidence.trustedAdmissionIssued, false);
  assert.ok(d.explicitlyMissing.includes("trusted publication/admission manifests"));
});

test("all 35 protocol selectors preserve special-round policy while exactly 32 pairs enter the primary grade", { skip: !ownerAvailable }, t => {
  const { draft, store } = stage(t), pins = draft.lockArtifactCandidates;
  const cohort = store.readJson(pins.cohort), model = store.readJson(pins.modelProtocol), baseline = store.readJson(pins.baselineProtocol);
  const modelForecast = store.readJson(pins.modelForecast), baselineForecast = store.readJson(pins.baselineForecast);
  const comparison = store.readJson(pins.comparisonContract);
  assert.equal(cohort.fullDisplayRaceIds.length, 35); assert.equal(cohort.primaryGradeRaceIds.length, 32);
  assert.deepEqual(cohort.ungradedDisplayRaceIds, ELECTION_NIGHT_RUBRIC.earlierBenchmark.conditionalRaceIds);
  assert.equal(cohort.senateObservationSelectors.length, 35);
  assert.deepEqual(model.races.map(row => row.raceId), cohort.fullDisplayRaceIds);
  assert.deepEqual(baseline.races.map(row => row.raceId), cohort.fullDisplayRaceIds);
  assert.deepEqual(comparison.winnerCohort.map(row => row.raceId), cohort.primaryGradeRaceIds);
  assert.deepEqual(comparison.intervalCohort.map(row => row.raceId), cohort.primaryGradeRaceIds);
  assert.deepEqual(comparison.displayRaceIds, cohort.fullDisplayRaceIds);
  assert.deepEqual(comparison.ungradedDisplayRaceIds, cohort.ungradedDisplayRaceIds);
  for (const [i, row] of cohort.senateObservationSelectors.entries()) {
    assert.equal(row.semanticRole, "predeclared_observation_selection_rule");
    assert.equal(row.policyRow.raceId, cohort.fullDisplayRaceIds[i]);
    assert.equal(row.selector.ruleFamily, null);
    for (const protocol of [model, baseline]) {
      const target = protocol.races[i];
      assert.equal(target.winner.eventId, row.selector.eventId);
      assert.equal(target.winner.roundId, row.selector.roundId);
      assert.equal(target.margin.resultRuleId, row.selector.resultRuleId);
      assert.equal(target.winner.resultRuleId, row.selector.resultRuleId);
      assert.equal(target.winner.minimumStatus, "certified");
      assert.equal(target.margin.minimumStatus, "certified");
      assert.equal(target.winner.ruleFamily, null); assert.equal(target.margin.ruleFamily, null);
      assert.equal(target.margin.firstCandidateId, row.policyRow.firstCandidateId);
      assert.equal(target.margin.secondCandidateId, row.policyRow.secondCandidateId);
    }
    assert.equal(modelForecast.races[i].id, row.policyRow.raceId);
    assert.equal(baselineForecast.races[i].id, row.policyRow.raceId);
  }
  const byState = Object.fromEntries(cohort.senateObservationSelectors.map(row => [row.policyRow.state, row]));
  for (const state of ["AK", "ME"]) {
    assert.equal(byState[state].policyRow.operatorFamily, "rcv_locked_final_two");
    assert.equal(byState[state].selector.roundId, "official_observed_exact_locked_final_two_v1");
  }
  assert.equal(byState.GA.policyRow.operatorFamily, "ga_certified_decisive_event");
  assert.equal(byState.GA.policyRow.runoffDate, "2026-12-01");
  assert.equal(byState.GA.selector.roundId, "certified_no_runoff_general_or_exact_pair_runoff_v1");
  assert.equal(byState.AL.selector.roundId, "certified_decisive_general_v1");
  assert.equal(model.reviewedDefinition, null); assert.equal(baseline.reviewedDefinition, null);
  const check = gradeElectionNightRubric({ contract: comparison, modelForecast: comparison.model,
    baselineForecast: comparison.baseline, winnerRows: [], intervalRows: [] });
  assert.deepEqual(check.issues, []); assert.equal(check.status, "not_graded");
  assert.equal(check.winner.coverage.total, 32); assert.equal(check.interval.coverage.total, 32);
});

test("pinned source reads reject changed bytes and escaped paths", t => {
  const root = mkdtempSync(join(tmpdir(), "senate-lock-source-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = join(root, "evidence"); writeFileSync(file, "INVENTED reviewed bytes");
  const digest = "c9c56e9da32794d8e6fa707bc7345e31611147215654134b9a5b34f34a0bce6d";
  assert.throws(() => readPinnedFile(root, "evidence", digest), /Source bytes changed/);
  assert.throws(() => readPinnedFile(root, "../outside", digest), /Unsafe source path/);
  assert.throws(() => readPinnedFile(root, file, digest), /Unsafe source path/);
  symlinkSync("/etc/hosts", join(root, "outside-link"));
  assert.throws(() => readPinnedFile(root, "outside-link", digest), /Unsafe source path/);
});
