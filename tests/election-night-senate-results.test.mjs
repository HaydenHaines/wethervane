import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { evaluateSenatePairResult, SENATE_PAIR_RESULT_POLICY } from "../lib/election-night-senate-results.ts";

const firstId = "synthetic:2099:first";
const secondId = "synthetic:2099:second";
const row = (operatorFamily = "plurality_named_pair") => ({
  raceId: "2099:ZZ:senate:regular", state: "ZZ", seatId: "SYN-SEN-ZZ-1", eventKind: "regular",
  generalDate: "2099-11-03", runoffDate: "2099-12-01", firstCandidateId: firstId,
  secondCandidateId: secondId, operatorFamily,
});
const pin = label => ({ path: `invented-fixture/${label}.json`, sha256: "a".repeat(64) });
const observation = overrides => ({
  sourcePin: pin("source"), sourceAdmissionReviewPin: pin("admission"),
  candidateCrosswalkReviewPin: pin("candidate-crosswalk"), eventRoundReviewPin: pin("event-round"),
  official: true, candidateCrosswalkReviewed: true, eventRoundReviewed: true,
  raceId: "2099:ZZ:senate:regular", seatId: "SYN-SEN-ZZ-1", eventKind: "regular",
  replacementOrIdentityChange: false, relevantRuleChange: false, stage: "general", eventDate: "2099-11-03",
  countsByNativeCandidateId: { [firstId]: 6123, [secondId]: 5877 }, officialOfficeWinnerId: firstId,
  certified: true, contestFinal: true, pendingRecount: false, pendingResultDispute: false,
  ...overrides,
});

test("certified plurality pair arithmetic is conditional, and an outside office winner stays a separate diagnostic", () => {
  const result = evaluateSenatePairResult(row(), observation({ officialOfficeWinnerId: "synthetic:2099:other" }));
  assert.equal(result.status, "eligible_certified_pair_margin");
  assert.equal(result.reason, null);
  assert.equal(result.firstShare, 6123 / 12000);
  assert.equal(result.marginPp, 100 * (6123 - 5877) / 12000);
  assert.equal(result.firstRelativeLead, true);
  assert.equal(result.pairTie, false);
  assert.equal(result.marginFinalScoreEligible, true);
  assert.equal(result.pairBinaryFinalScoreEligible, true);
  assert.equal(result.officialOfficeWinnerIdDiagnostic, "synthetic:2099:other");
  assert.equal(result.officeWinnerOutsideLockedPair, true);
  assert.equal(result.legalWinnerGradeSupported, false);
  assert.equal(result.admissionBoundary, "caller_assertions_only_source_authentication_not_performed");
});

test("a certified pair tie has margin zero but no binary relative-lead result", () => {
  const result = evaluateSenatePairResult(row(), observation({
    countsByNativeCandidateId: { [firstId]: 200, [secondId]: 200 }, officialOfficeWinnerId: null,
  }));
  assert.equal(result.status, "eligible_certified_pair_margin");
  assert.equal(result.reason, "certified_pair_tie_binary_unavailable");
  assert.equal(result.firstShare, 0.5);
  assert.equal(result.marginPp, 0);
  assert.equal(result.firstRelativeLead, null);
  assert.equal(result.pairTie, true);
  assert.equal(result.marginFinalScoreEligible, true);
  assert.equal(result.pairBinaryFinalScoreEligible, false);
  assert.equal(result.officeWinnerOutsideLockedPair, null);
});

test("RCV rows require the officially observed final two to equal the exact locked pair", () => {
  const locked = row("rcv_locked_final_two");
  const common = { stage: "general", officialObservedFinalRound: true, continuingCandidateIds: [secondId, firstId] };
  const eligible = evaluateSenatePairResult(locked, observation(common));
  assert.equal(eligible.status, "eligible_certified_pair_margin");
  assert.equal(eligible.firstRelativeLead, true);
  for (const [change, reason] of [
    [{ continuingCandidateIds: [firstId, secondId, "synthetic:2099:third"] }, "no_official_observed_exact_locked_two_final_pair"],
    [{ officialObservedFinalRound: false }, "no_official_observed_exact_locked_two_final_pair"],
    [{ eventDate: "2099-11-04" }, "wrong_rcv_event_date_or_stage"],
    [{ stage: "primary" }, "wrong_rcv_event_date_or_stage"],
  ]) {
    const result = evaluateSenatePairResult(locked, observation({ ...common, ...change }));
    assert.equal(result.status, "unavailable"); assert.equal(result.reason, reason);
    assert.equal(result.marginPp, null);
  }
});

test("Georgia general and runoff conditions remain explicit and date-bound", () => {
  const locked = row("ga_certified_decisive_event");
  const pending = evaluateSenatePairResult(locked, observation({ runoffRequired: true }));
  assert.equal(pending.status, "pending_runoff");
  assert.equal(pending.reason, "general_not_decisive_runoff_required");
  const unknown = evaluateSenatePairResult(locked, observation({ runoffRequired: undefined }));
  assert.equal(unknown.reason, "decisive_general_or_runoff_requirement_unverified");
  const runoff = { stage: "runoff", eventDate: "2099-12-01", runoffCandidateIds: [firstId, secondId] };
  assert.equal(evaluateSenatePairResult(locked, observation(runoff)).status, "eligible_certified_pair_margin");
  const changedDate = evaluateSenatePairResult(locked, observation({ ...runoff, eventDate: "2099-12-08" }));
  assert.equal(changedDate.status, "escalation");
  assert.equal(changedDate.reason, "runoff_date_changed_requires_new_policy_review");
  const otherFinalists = evaluateSenatePairResult(locked, observation({
    ...runoff, runoffCandidateIds: [firstId, "synthetic:2099:replacement"],
  }));
  assert.equal(otherFinalists.reason, "runoff_does_not_contain_exact_locked_principals");
});

test("official preliminary values stay descriptive and cannot become final-score eligible", () => {
  const result = evaluateSenatePairResult(row(), observation({ contestFinal: false }));
  assert.equal(result.status, "preliminary_descriptive_only");
  assert.equal(result.reason, "certification_finality_or_review_pending");
  assert.equal(result.marginPp, 2.05);
  assert.equal(result.marginFinalScoreEligible, false);
  assert.equal(result.pairBinaryFinalScoreEligible, false);
  for (const change of [{ certified: false }, { pendingRecount: true }, { pendingResultDispute: true }]) {
    const partial = evaluateSenatePairResult(row(), observation(change));
    assert.equal(partial.status, "preliminary_descriptive_only");
    assert.equal(partial.marginFinalScoreEligible, false);
  }
});

test("source-review, event, identity and count failures fail closed with owner reason codes", () => {
  const cases = [
    ["pin", o => { o.sourcePin.sha256 = "bad"; }, "missing_reviewed_source_pin"],
    ["official", o => { o.official = false; }, "source_not_official"],
    ["crosswalk", o => { o.candidateCrosswalkReviewed = false; }, "identity_or_event_crosswalk_unreviewed"],
    ["race", o => { o.raceId = "2099:YY:senate:regular"; }, "wrong_race_seat_or_regular_special_event"],
    ["seat", o => { o.seatId = "SYN-SEN-ZZ-2"; }, "wrong_race_seat_or_regular_special_event"],
    ["event kind", o => { o.eventKind = "special"; }, "wrong_race_seat_or_regular_special_event"],
    ["replacement", o => { o.replacementOrIdentityChange = true; }, "locked_identity_changed_or_unresolved"],
    ["changed rule", o => { o.relevantRuleChange = true; }, "rule_change_or_unknown_requires_new_policy_review", "escalation"],
    ["missing count", o => { delete o.countsByNativeCandidateId[firstId]; }, "missing_locked_candidate_counts"],
    ["negative count", o => { o.countsByNativeCandidateId[firstId] = -1; }, "counts_must_be_nonnegative_integers"],
    ["fractional count", o => { o.countsByNativeCandidateId[firstId] = 1.2; }, "counts_must_be_nonnegative_integers"],
    ["unsafe count", o => { o.countsByNativeCandidateId[firstId] = Number.MAX_SAFE_INTEGER + 1; }, "counts_must_be_nonnegative_integers"],
    ["zero denominator", o => { o.countsByNativeCandidateId = { [firstId]: 0, [secondId]: 0 }; }, "zero_pair_denominator"],
    ["stage", o => { o.stage = "runoff"; }, "wrong_plurality_general_event"],
    ["date", o => { o.eventDate = "2099-11-04"; }, "wrong_plurality_general_event"],
  ];
  for (const [label, change, reason, status = "unavailable"] of cases) {
    const value = observation({}); change(value);
    const result = evaluateSenatePairResult(row(), value);
    assert.equal(result.reason, reason, label);
    assert.equal(result.status, status, label);
    assert.equal(result.marginPp, null, label);
  }
  assert.equal(SENATE_PAIR_RESULT_POLICY.operatorSha256, "81cf4c04c974faab4388c6d623a93cb8ddfd1d6ce0aa7659e4c7cd330eb9830e");
});

const OWNER_OPERATOR = "/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-polling-refresh-20261026/result-policy-special-rules-sol-v1/pair_operator.py";
test("synthetic owner parity cases match the pinned Python operator when available", { skip: !existsSync(OWNER_OPERATOR) }, () => {
  const cases = [
    [row(), observation({ officialOfficeWinnerId: "synthetic:2099:other" })],
    [row(), observation({ countsByNativeCandidateId: { [firstId]: 200, [secondId]: 200 } })],
    [row(), observation({ contestFinal: false })],
    [row(), observation({ official: false })],
    [row(), observation({ relevantRuleChange: true })],
    [row(), observation({ countsByNativeCandidateId: { [firstId]: -1, [secondId]: 3 } })],
    [row("rcv_locked_final_two"), observation({ officialObservedFinalRound: true, continuingCandidateIds: [firstId, secondId] })],
    [row("rcv_locked_final_two"), observation({ officialObservedFinalRound: false, continuingCandidateIds: [firstId, secondId] })],
    [row("rcv_locked_final_two"), observation({ officialObservedFinalRound: true, continuingCandidateIds: [firstId, secondId, "synthetic:2099:third"] })],
    [row("ga_certified_decisive_event"), observation({ runoffRequired: true })],
    [row("ga_certified_decisive_event"), observation({ runoffRequired: false })],
    [row("ga_certified_decisive_event"), observation({
      stage: "runoff", eventDate: "2099-12-01", runoffCandidateIds: [firstId, secondId],
    })],
    [row("ga_certified_decisive_event"), observation({
      stage: "runoff", eventDate: "2099-12-08", runoffCandidateIds: [firstId, secondId],
    })],
  ];
  const payload = cases.map(([locked, current]) => ({
    locked: {
      race_id: locked.raceId, state: locked.state, seat_id: locked.seatId, event_kind: locked.eventKind,
      scheduled_general_event_date: locked.generalDate, scheduled_runoff_event_date: locked.runoffDate,
      first_candidate_id: locked.firstCandidateId, second_candidate_id: locked.secondCandidateId,
      operator_family: locked.operatorFamily,
    },
    observation: {
      source_pin: current.sourcePin,
      source_admission_review_pin: current.sourceAdmissionReviewPin,
      candidate_crosswalk_review_pin: current.candidateCrosswalkReviewPin,
      event_round_review_pin: current.eventRoundReviewPin,
      official: current.official,
      candidate_crosswalk_reviewed: current.candidateCrosswalkReviewed,
      event_round_reviewed: current.eventRoundReviewed,
      race_id: current.raceId, seat_id: current.seatId, event_kind: current.eventKind,
      replacement_or_identity_change: current.replacementOrIdentityChange,
      relevant_rule_change: current.relevantRuleChange, stage: current.stage,
      event_date: current.eventDate,
      official_observed_final_round: current.officialObservedFinalRound,
      continuing_candidate_ids: current.continuingCandidateIds,
      runoff_required: current.runoffRequired, runoff_candidate_ids: current.runoffCandidateIds,
      counts_by_native_candidate_id: current.countsByNativeCandidateId,
      official_office_winner_id: current.officialOfficeWinnerId,
      certified: current.certified, contest_final: current.contestFinal,
      pending_recount: current.pendingRecount, pending_result_dispute: current.pendingResultDispute,
    },
  }));
  const script = `import json,sys; sys.path.insert(0, ${JSON.stringify(OWNER_OPERATOR.replace(/\/pair_operator\.py$/, ""))}); from pair_operator import evaluate_pair; print(json.dumps([evaluate_pair(x['locked'],x['observation']) for x in json.load(sys.stdin)]))`;
  const child = spawnSync("python3", ["-B", "-c", script], { input: JSON.stringify(payload), encoding: "utf8" });
  assert.equal(child.status, 0, child.stderr);
  const pythonRows = JSON.parse(child.stdout);
  const nodeRows = cases.map(([locked, current]) => evaluateSenatePairResult(locked, current));
  for (let index = 0; index < nodeRows.length; index++) {
    const js = nodeRows[index], py = pythonRows[index];
    assert.equal(js.status, py.status);
    assert.equal(js.reason, py.reason);
    assert.equal(js.marginPp, py.margin_pp);
    assert.equal(js.firstShare, py.first_share);
    assert.equal(js.firstRelativeLead, py.first_relative_lead);
    assert.equal(js.pairTie, py.pair_tie);
    assert.equal(js.marginFinalScoreEligible, py.margin_final_score_eligible);
    assert.equal(js.pairBinaryFinalScoreEligible, py.pair_binary_final_score_eligible);
    assert.equal(js.officeWinnerOutsideLockedPair, py.office_winner_outside_locked_pair);
    assert.equal(js.legalWinnerGradeSupported, py.legal_winner_grade_supported);
  }
});
