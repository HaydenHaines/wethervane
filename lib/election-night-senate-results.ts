/**
 * Pure port of the pinned pre-result Senate pair operator.
 *
 * This projects an already prepared observation into the owner policy's pair
 * arithmetic and availability status. It does not fetch or authenticate source
 * bytes, prove the supplied review pins, admit a candidate/event crosswalk, or
 * grade a legal office winner. Boolean review declarations are conditions, not
 * evidence. The caller must separately establish every source and identity claim.
 */
export const SENATE_PAIR_RESULT_POLICY = Object.freeze({
  schema: "SenatePairResultProjectionV1",
  policySha256: "efe1bf193e2959284b73ea833af0a136f6c592001a2faf50505e3258e25cf0cd",
  mappingSha256: "244e9db017957d0a5361f5f1b7a2498f066c43d9b94a59f1277e8cff91b6afa3",
  operatorSha256: "81cf4c04c974faab4388c6d623a93cb8ddfd1d6ce0aa7659e4c7cd330eb9830e",
});

export interface SenateResultPolicyRow {
  raceId: string;
  state: string;
  seatId: string;
  eventKind: "regular" | "special";
  generalDate: string;
  runoffDate: string | null;
  firstCandidateId: string;
  secondCandidateId: string;
  operatorFamily: "plurality_named_pair" | "rcv_locked_final_two" | "ga_certified_decisive_event";
}

export interface SenateResultReviewPin { path: string; sha256: string }

/** Caller-admitted source data and reviews. This module only checks the declarations. */
export interface SenatePairObservation {
  sourcePin: SenateResultReviewPin;
  sourceAdmissionReviewPin: SenateResultReviewPin;
  candidateCrosswalkReviewPin: SenateResultReviewPin;
  eventRoundReviewPin: SenateResultReviewPin;
  official: boolean;
  candidateCrosswalkReviewed: boolean;
  eventRoundReviewed: boolean;
  raceId: string;
  seatId: string;
  eventKind: string;
  replacementOrIdentityChange: boolean;
  relevantRuleChange: boolean;
  stage: string;
  eventDate: string;
  officialObservedFinalRound?: boolean;
  continuingCandidateIds?: string[];
  runoffRequired?: boolean;
  runoffCandidateIds?: string[];
  countsByNativeCandidateId: Record<string, number>;
  officialOfficeWinnerId?: string | null;
  certified: boolean;
  contestFinal: boolean;
  pendingRecount: boolean;
  pendingResultDispute: boolean;
}

export type SenatePairResultStatus = "unavailable" | "pending_runoff" | "escalation" |
  "preliminary_descriptive_only" | "eligible_certified_pair_margin";

export interface SenatePairResultProjection {
  schema: typeof SENATE_PAIR_RESULT_POLICY.schema;
  policySha256: string;
  mappingSha256: string;
  operatorSha256: string;
  /** Passing caller-supplied booleans is not source authentication or admission. */
  admissionBoundary: "caller_assertions_only_source_authentication_not_performed";
  raceId: string;
  marginPp: number | null;
  firstShare: number | null;
  firstRelativeLead: boolean | null;
  pairTie: boolean | null;
  marginFinalScoreEligible: boolean;
  pairBinaryFinalScoreEligible: boolean;
  officialOfficeWinnerIdDiagnostic: string | null;
  officeWinnerOutsideLockedPair: boolean | null;
  legalWinnerGradeSupported: false;
  status: SenatePairResultStatus;
  reason: string | null;
}

const PIN_RE = /^[0-9a-f]{64}$/;
function validPin(value: unknown): value is SenateResultReviewPin {
  if (!value || typeof value !== "object") return false;
  const pin = value as Partial<SenateResultReviewPin>;
  return typeof pin.path === "string" && pin.path.length > 0 &&
    typeof pin.sha256 === "string" && PIN_RE.test(pin.sha256);
}

function baseResult(row: SenateResultPolicyRow, observation: SenatePairObservation): SenatePairResultProjection {
  return {
    schema: SENATE_PAIR_RESULT_POLICY.schema,
    policySha256: SENATE_PAIR_RESULT_POLICY.policySha256,
    mappingSha256: SENATE_PAIR_RESULT_POLICY.mappingSha256,
    operatorSha256: SENATE_PAIR_RESULT_POLICY.operatorSha256,
    admissionBoundary: "caller_assertions_only_source_authentication_not_performed",
    raceId: row.raceId,
    marginPp: null,
    firstShare: null,
    firstRelativeLead: null,
    pairTie: null,
    marginFinalScoreEligible: false,
    pairBinaryFinalScoreEligible: false,
    officialOfficeWinnerIdDiagnostic: observation.officialOfficeWinnerId ?? null,
    officeWinnerOutsideLockedPair: null,
    legalWinnerGradeSupported: false,
    status: "unavailable",
    reason: null,
  };
}

function stop(result: SenatePairResultProjection, reason: string,
  status: SenatePairResultStatus = "unavailable"): SenatePairResultProjection {
  return { ...result, status, reason };
}

function exactPair(values: unknown, pair: readonly string[]): values is string[] {
  return Array.isArray(values) && values.length === 2 &&
    new Set(values).size === 2 && pair.every(id => values.includes(id));
}

/** Evaluate only the declared two-candidate pair event; never infer legal office winner. */
export function evaluateSenatePairResult(
  locked: SenateResultPolicyRow,
  observation: SenatePairObservation,
): SenatePairResultProjection {
  const result = baseResult(locked, observation);
  for (const [key, value] of Object.entries({
    source_pin: observation.sourcePin,
    source_admission_review_pin: observation.sourceAdmissionReviewPin,
    candidate_crosswalk_review_pin: observation.candidateCrosswalkReviewPin,
    event_round_review_pin: observation.eventRoundReviewPin,
  })) {
    if (!validPin(value)) return stop(result, `missing_reviewed_${key}`);
  }
  if (observation.official !== true) return stop(result, "source_not_official");
  if (observation.candidateCrosswalkReviewed !== true || observation.eventRoundReviewed !== true)
    return stop(result, "identity_or_event_crosswalk_unreviewed");
  if (observation.raceId !== locked.raceId || observation.seatId !== locked.seatId || observation.eventKind !== locked.eventKind)
    return stop(result, "wrong_race_seat_or_regular_special_event");
  const pair = [locked.firstCandidateId, locked.secondCandidateId];
  if (new Set(pair).size !== 2) return stop(result, "invalid_locked_pair");
  if (observation.replacementOrIdentityChange !== false)
    return stop(result, "locked_identity_changed_or_unresolved");
  if (observation.relevantRuleChange !== false)
    return stop(result, "rule_change_or_unknown_requires_new_policy_review", "escalation");

  if (locked.operatorFamily === "rcv_locked_final_two") {
    if (observation.stage !== "general" || observation.eventDate !== locked.generalDate)
      return stop(result, "wrong_rcv_event_date_or_stage");
    if (observation.officialObservedFinalRound !== true || !exactPair(observation.continuingCandidateIds, pair))
      return stop(result, "no_official_observed_exact_locked_two_final_pair");
  } else if (locked.operatorFamily === "ga_certified_decisive_event") {
    if (observation.stage === "general") {
      if (observation.eventDate !== locked.generalDate) return stop(result, "wrong_general_event_date");
      if (observation.runoffRequired === true)
        return stop(result, "general_not_decisive_runoff_required", "pending_runoff");
      if (observation.runoffRequired !== false)
        return stop(result, "decisive_general_or_runoff_requirement_unverified");
    } else if (observation.stage === "runoff") {
      if (observation.eventDate !== locked.runoffDate)
        return stop(result, "runoff_date_changed_requires_new_policy_review", "escalation");
      if (!exactPair(observation.runoffCandidateIds, pair))
        return stop(result, "runoff_does_not_contain_exact_locked_principals");
    } else return stop(result, "wrong_ga_stage");
  } else if (locked.operatorFamily === "plurality_named_pair") {
    if (observation.stage !== "general" || observation.eventDate !== locked.generalDate)
      return stop(result, "wrong_plurality_general_event");
  } else return stop(result, "unsupported_operator_family");

  const counts = observation.countsByNativeCandidateId;
  if (!counts || typeof counts !== "object" || !pair.every(id => Object.hasOwn(counts, id)))
    return stop(result, "missing_locked_candidate_counts");
  const first = counts[pair[0]], second = counts[pair[1]];
  if (![first, second].every(value => Number.isSafeInteger(value) && value >= 0))
    return stop(result, "counts_must_be_nonnegative_integers");
  if (first + second <= 0) return stop(result, "zero_pair_denominator");

  result.firstShare = first / (first + second);
  result.marginPp = 100 * (first - second) / (first + second);
  result.firstRelativeLead = first === second ? null : first > second;
  result.pairTie = first === second;
  result.officeWinnerOutsideLockedPair = observation.officialOfficeWinnerId == null
    ? null : !pair.includes(observation.officialOfficeWinnerId);
  const final = observation.certified === true && observation.contestFinal === true &&
    observation.pendingRecount === false && observation.pendingResultDispute === false;
  if (!final) return stop(result, "certification_finality_or_review_pending", "preliminary_descriptive_only");
  return {
    ...result,
    status: "eligible_certified_pair_margin",
    reason: result.pairTie ? "certified_pair_tie_binary_unavailable" : null,
    marginFinalScoreEligible: true,
    pairBinaryFinalScoreEligible: !result.pairTie,
  };
}
