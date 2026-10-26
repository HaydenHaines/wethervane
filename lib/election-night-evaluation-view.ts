import { ELECTION_NIGHT_RUBRIC, type ElectionNightRubricResult } from "./election-night-rubric.ts";
import type { RubricTarget } from "./election-night-rubric.ts";

export type ElectionNightEvaluationViewStatus = "lock_not_configured" | "locked_no_results" |
  "partial_results" | "complete_score_not_gradeable" | "graded" | "unavailable";
export interface ElectionNightForecastBindingView {
  chamber: "senate" | "house"; runId: string; resultSha256: string; nativeForecastSha256: string;
  targetId: string | null; lawId: string | null; issuedAt: string; cutoffAt: string; horizonCalendarDays: number;
}
export interface ElectionNightEvaluationSourceView {
  recordId: string; revision: string; recordSha256: string; revisionSha256: string;
  sourceSha256: string; sourceUrl: string; publishedAt: string; retrievedAt: string;
  resultPolicySha256: string; evidenceSha256: string[];
}
export interface ElectionNightEvaluationPartView {
  status: "scored" | "unresolved" | "unsupported" | "mismatched" | "not_in_grade_cohort" | "not_applicable";
  finality: "unreported" | "provisional" | "called" | "final" | "certified" | "unresolved" | "unsupported" | null;
  loss: number | null; observedValue: string | number | null; reason: string | null;
  observedValueKind: "candidate" | "party_group" | null;
  modalPickCorrect: boolean | null;
  diagnosticScope: "graded_cohort" | "display_only" | null;
  observation: { value: string | number; label: string | null; status: "provisional" | "called" | "final" | "certified" } | null;
  marginScore: null | { pointRule: "mean" | "median"; predictedMarginPp: number; observedMarginPp: number;
    absoluteErrorPp: number; interval90: { lower: number; upper: number; covered: boolean } };
  target: RubricTarget | null;
}
export interface ElectionNightLockedRaceProjectionView {
  seatId: string; state: string; electionLabel: string;
  modalPick: { name: string; kind: "candidate" | "group"; probability: number };
  pair: null | { firstCandidateName: string; secondCandidateName: string; meanMarginPp: number;
    range90: { lower: number; upper: number } };
}
export interface ElectionNightEvaluationRowView {
  raceId: string; displayOnly: boolean; forecast: ElectionNightLockedRaceProjectionView;
  winner: { model: ElectionNightEvaluationPartView; baseline: ElectionNightEvaluationPartView };
  margin: { model: ElectionNightEvaluationPartView; baseline: ElectionNightEvaluationPartView };
  sources: ElectionNightEvaluationSourceView[];
}
export interface ElectionNightEvaluationCoverageView {
  total: number; final: number; provisional: number; pending: number; mismatched: number; unexpected: number;
}
export interface ElectionNightEvaluationRawSummaryView {
  scope: "available_matched_rows"; count: number; finalCount: number; provisionalCount: number;
  modelMeanLoss: number | null; baselineMeanLoss: number | null; meanPairedLossDifference: number | null;
}
export interface ElectionNightEvaluationView {
  schema: "ElectionNightEvaluationViewV1"; chamber: "senate" | "house";
  status: ElectionNightEvaluationViewStatus; errorCode: "lock_verification_failed" | "ledger_verification_failed" |
    "scoring_verification_failed" | "configuration_invalid" | null;
  lock: null | { id: string; sha256: string; frozenAt: string; model: ElectionNightForecastBindingView; baseline: ElectionNightForecastBindingView;
    comparison: { id: string; sha256: string; label: string; mode: "matched_horizon" | "explicit_earlier_benchmark" };
    rubricDecisionSha256: string; cohort: { displayRaceIds: string[]; winnerRaceIds: string[];
      intervalRaceIds: string[]; intervalExclusions: { raceId: string; reason: string }[] } };
  ledger: null | { id: string; sha256: string; chosenHeadSha256: string | null; revisionCount: number; recordCount: number;
    provenance: ElectionNightEvaluationSourceView[] };
  coverage: null | { winner: ElectionNightEvaluationCoverageView; interval: ElectionNightEvaluationCoverageView };
  availableMatchedRows: null | { winner: ElectionNightEvaluationRawSummaryView; interval: ElectionNightEvaluationRawSummaryView };
  rows: ElectionNightEvaluationRowView[];
  rubricStatus: "graded" | "not_graded" | null;
  /** Exposed only after the grader returns a complete overall grade. */
  rubric: ElectionNightRubricResult | null;
}

const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const statuses = new Set(["scored", "unresolved", "unsupported", "mismatched", "not_in_grade_cohort", "not_applicable"]);
const finalities = new Set(["unreported", "provisional", "called", "final", "certified", "unresolved", "unsupported"]);

function validSource(value: unknown): value is ElectionNightEvaluationSourceView {
  return object(value) && text(value.recordId) && text(value.revision) && hash(value.recordSha256) && hash(value.revisionSha256) &&
    hash(value.sourceSha256) && typeof value.sourceUrl === "string" && /^https?:\/\//.test(value.sourceUrl) &&
    typeof value.publishedAt === "string" && typeof value.retrievedAt === "string" &&
    Number.isFinite(Date.parse(value.publishedAt)) && Number.isFinite(Date.parse(value.retrievedAt)) &&
    Date.parse(String(value.publishedAt)) <= Date.parse(String(value.retrievedAt)) &&
    hash(value.resultPolicySha256) && Array.isArray(value.evidenceSha256) && value.evidenceSha256.every(hash);
}
function validMarginScore(value: unknown): value is ElectionNightEvaluationPartView["marginScore"] {
  if (!object(value) || !["mean", "median"].includes(String(value.pointRule)) ||
    !["predictedMarginPp", "observedMarginPp", "absoluteErrorPp"].every(key =>
      typeof value[key] === "number" && Number.isFinite(value[key])) || typeof value.absoluteErrorPp !== "number" ||
    value.absoluteErrorPp < 0 || !object(value.interval90) ||
    typeof value.interval90.lower !== "number" || !Number.isFinite(value.interval90.lower) ||
    typeof value.interval90.upper !== "number" || !Number.isFinite(value.interval90.upper) ||
    value.interval90.lower > value.interval90.upper || typeof value.interval90.covered !== "boolean") return false;
  return true;
}
function validPart(value: unknown): value is ElectionNightEvaluationPartView {
  return object(value) && statuses.has(String(value.status)) &&
    (value.finality === null || finalities.has(String(value.finality))) &&
    (value.loss === null || (typeof value.loss === "number" && Number.isFinite(value.loss))) &&
    (value.observedValue === null || typeof value.observedValue === "string" ||
      (typeof value.observedValue === "number" && Number.isFinite(value.observedValue))) &&
    (value.reason === null || typeof value.reason === "string") &&
    [null, "candidate", "party_group"].includes(value.observedValueKind as never) &&
    (value.modalPickCorrect === null || typeof value.modalPickCorrect === "boolean") &&
    [null, "graded_cohort", "display_only"].includes(value.diagnosticScope as never) &&
    (value.observation === null || object(value.observation) &&
      (typeof value.observation.value === "string" || Number.isFinite(value.observation.value)) &&
      (value.observation.label === null || typeof value.observation.label === "string") &&
      ["provisional", "called", "final", "certified"].includes(String(value.observation.status))) &&
    (value.marginScore === null || validMarginScore(value.marginScore)) &&
    (value.target === null || object(value.target)) &&
    (value.status === "scored" ? typeof value.loss === "number" && value.observedValue !== null && value.diagnosticScope === "graded_cohort" :
      value.loss === null) &&
    (value.diagnosticScope === "display_only" ? value.status === "not_in_grade_cohort" :
      value.status !== "not_in_grade_cohort" || value.diagnosticScope === null) &&
    ((value.marginScore !== null || value.modalPickCorrect !== null) ?
      (value.status === "scored" && value.diagnosticScope === "graded_cohort") ||
      (value.status === "not_in_grade_cohort" && value.diagnosticScope === "display_only") : value.diagnosticScope !== "display_only");
}
function validProjection(value: unknown): value is ElectionNightLockedRaceProjectionView {
  if (!object(value) || !text(value.seatId) || !text(value.state) || !text(value.electionLabel) || !object(value.modalPick) ||
    !text(value.modalPick.name) || !["candidate", "group"].includes(String(value.modalPick.kind)) ||
    typeof value.modalPick.probability !== "number" || !Number.isFinite(value.modalPick.probability) ||
    value.modalPick.probability < 0 || value.modalPick.probability > 1) return false;
  if (value.pair === null) return true;
  return object(value.pair) && text(value.pair.firstCandidateName) && text(value.pair.secondCandidateName) &&
    typeof value.pair.meanMarginPp === "number" && Number.isFinite(value.pair.meanMarginPp) && object(value.pair.range90) &&
    typeof value.pair.range90.lower === "number" && Number.isFinite(value.pair.range90.lower) &&
    typeof value.pair.range90.upper === "number" && Number.isFinite(value.pair.range90.upper) &&
    value.pair.range90.lower <= value.pair.range90.upper;
}
function validCoverage(value: unknown): value is ElectionNightEvaluationCoverageView {
  return object(value) && ["total", "final", "provisional", "pending", "mismatched", "unexpected"].every(key =>
    Number.isSafeInteger(value[key]) && Number(value[key]) >= 0) && Number(value.final) <= Number(value.total) &&
    Number(value.provisional) <= Number(value.total) && Number(value.pending) <= Number(value.total) &&
    Number(value.mismatched) <= Number(value.total) && Number(value.unexpected) >= 0;
}
function validBinding(value: unknown): value is ElectionNightForecastBindingView {
  if (!object(value) || !["senate", "house"].includes(String(value.chamber)) || !text(value.runId) ||
    !hash(value.resultSha256) || !hash(value.nativeForecastSha256) ||
    typeof value.issuedAt !== "string" || typeof value.cutoffAt !== "string" ||
    !Number.isFinite(Date.parse(value.issuedAt)) || !Number.isFinite(Date.parse(value.cutoffAt)) ||
    !Number.isSafeInteger(value.horizonCalendarDays) || Number(value.horizonCalendarDays) < 0) return false;
  const hasTypedIdentity = text(value.targetId) && text(value.lawId);
  const nativeHouseNullIdentity = value.chamber === "house" && value.targetId === null && value.lawId === null;
  if (!hasTypedIdentity && !nativeHouseNullIdentity) return false;
  return Date.parse(String(value.cutoffAt)) <= Date.parse(String(value.issuedAt));
}

/** Validate the small browser contract; this is not a trust/admission decision. */
export function parseElectionNightEvaluationView(value: unknown, expectedChamber?: "senate" | "house"): ElectionNightEvaluationView | null {
  try {
    if (!object(value)) return null;
    const v = value as unknown as ElectionNightEvaluationView;
    if (v.schema !== "ElectionNightEvaluationViewV1" || !["senate", "house"].includes(v.chamber) ||
      (expectedChamber && v.chamber !== expectedChamber) ||
      !["lock_not_configured", "locked_no_results", "partial_results", "complete_score_not_gradeable", "graded", "unavailable"].includes(v.status) ||
      ![null, "lock_verification_failed", "ledger_verification_failed", "scoring_verification_failed", "configuration_invalid"].includes(v.errorCode as never) ||
      ![null, "graded", "not_graded"].includes(v.rubricStatus as never) || !Array.isArray(v.rows) ||
      !v.rows.every(row => text(row?.raceId) && typeof row.displayOnly === "boolean" && validProjection(row.forecast) &&
        validPart(row.winner?.model) && validPart(row.winner?.baseline) && validPart(row.margin?.model) && validPart(row.margin?.baseline) &&
        Array.isArray(row.sources) && row.sources.every(validSource)) ||
      (v.rubric !== null && (v.rubric.policyId !== ELECTION_NIGHT_RUBRIC.id || v.rubric.status !== "graded" ||
        !hash(v.rubric.policyDecisionSha256) || v.rubric.policyDecisionSha256 !== ELECTION_NIGHT_RUBRIC.decisionSha256 ||
        !v.lock || v.rubric.policyDecisionSha256 !== v.lock.rubricDecisionSha256))) return null;
    if ((v.status === "lock_not_configured" || v.status === "unavailable") &&
      (v.rubric !== null || v.rubricStatus !== null || v.rows.length !== 0 || v.coverage !== null || v.availableMatchedRows !== null)) return null;
    if (v.lock !== null) {
      const lock = v.lock;
      if (!object(lock) || !text(lock.id) || !hash(lock.sha256) || typeof lock.frozenAt !== "string" || !Number.isFinite(Date.parse(lock.frozenAt)) ||
        !validBinding(lock.model) || !validBinding(lock.baseline) || lock.model.chamber !== v.chamber || lock.baseline.chamber !== v.chamber ||
        !object(lock.comparison) || !text(lock.comparison.id) || !hash(lock.comparison.sha256) || !text(lock.comparison.label) ||
        !["matched_horizon", "explicit_earlier_benchmark"].includes(String(lock.comparison.mode)) ||
        lock.rubricDecisionSha256 !== ELECTION_NIGHT_RUBRIC.decisionSha256 ||
        !object(lock.cohort) || !Array.isArray(lock.cohort.displayRaceIds) || !Array.isArray(lock.cohort.winnerRaceIds) ||
        !Array.isArray(lock.cohort.intervalRaceIds) || !Array.isArray(lock.cohort.intervalExclusions) ||
        ![lock.cohort.displayRaceIds, lock.cohort.winnerRaceIds, lock.cohort.intervalRaceIds].every((ids: unknown[]) =>
          ids.every(text) && new Set(ids).size === ids.length) ||
        !lock.cohort.intervalExclusions.every((row: unknown) => object(row) && text(row.raceId) && text(row.reason)) ||
        [lock.model, lock.baseline].some(binding => typeof binding.issuedAt !== "string" || typeof binding.cutoffAt !== "string" ||
          typeof binding.horizonCalendarDays !== "number")) return null;
    }
    if (v.ledger !== null) {
      const ledger = v.ledger;
      if (!object(ledger) || !text(ledger.id) || !hash(ledger.sha256) ||
        !(ledger.chosenHeadSha256 === null || hash(ledger.chosenHeadSha256)) ||
        !Number.isSafeInteger(ledger.revisionCount) || Number(ledger.revisionCount) < 0 ||
        !Number.isSafeInteger(ledger.recordCount) || Number(ledger.recordCount) < 0 || !Array.isArray(ledger.provenance) ||
        ledger.recordCount !== ledger.provenance.length || ledger.recordCount > ledger.revisionCount ||
        !ledger.provenance.every(validSource)) return null;
    }
    if (v.coverage !== null && (!object(v.coverage) || !validCoverage(v.coverage.winner) || !validCoverage(v.coverage.interval))) return null;
    if (v.availableMatchedRows !== null && (!object(v.availableMatchedRows) ||
      !(["winner", "interval"] as const).every(key => {
        const row = v.availableMatchedRows?.[key];
        return object(row) && row.scope === "available_matched_rows" &&
          ["count", "finalCount", "provisionalCount"].every(name => Number.isSafeInteger(row[name]) && Number(row[name]) >= 0) &&
          Number(row.finalCount) + Number(row.provisionalCount) === Number(row.count) &&
          ["modelMeanLoss", "baselineMeanLoss", "meanPairedLossDifference"].every(name => row[name] === null ||
            typeof row[name] === "number" && Number.isFinite(row[name]));
      }))) return null;
    if (v.lock) {
      const ids = v.lock.cohort.displayRaceIds;
      const winnerIds = v.lock.cohort.winnerRaceIds, intervalIds = v.lock.cohort.intervalRaceIds;
      const uniqueRows = new Set(v.rows.map(row => row.raceId));
      if (winnerIds.some(id => !ids.includes(id)) || intervalIds.some(id => !winnerIds.includes(id)) ||
        v.lock.cohort.intervalExclusions.some(row => !winnerIds.includes(row.raceId)) ||
        v.rows.length !== ids.length || uniqueRows.size !== ids.length ||
        v.rows.some((row, index) => row.raceId !== ids[index]) || v.coverage === null ||
        v.coverage.winner.total !== winnerIds.length || v.coverage.interval.total !== intervalIds.length ||
        v.coverage.winner.final + v.coverage.winner.provisional + v.coverage.winner.pending + v.coverage.winner.mismatched !== v.coverage.winner.total ||
        v.coverage.interval.final + v.coverage.interval.provisional + v.coverage.interval.pending + v.coverage.interval.mismatched !== v.coverage.interval.total) return null;
    } else if (v.coverage !== null || v.ledger !== null) return null;
    if ((v.status === "lock_not_configured" && (v.lock !== null || v.ledger !== null || v.rubric !== null || v.rows.length !== 0)) ||
      (v.status === "locked_no_results" && (!v.lock || !v.ledger || v.ledger.recordCount !== 0 ||
        v.ledger.chosenHeadSha256 !== null || v.rubric !== null || v.rubricStatus !== "not_graded" ||
        v.coverage?.winner.final !== 0 || v.coverage?.winner.provisional !== 0 || v.coverage?.winner.pending !== v.coverage?.winner.total ||
        v.coverage?.interval.final !== 0 || v.coverage?.interval.provisional !== 0 || v.coverage?.interval.pending !== v.coverage?.interval.total)) ||
      (v.status === "partial_results" && (!v.lock || !v.ledger || v.ledger.recordCount === 0 || v.rubric !== null || v.rubricStatus !== "not_graded")) ||
      (v.status === "complete_score_not_gradeable" && (!v.lock || !v.ledger || v.ledger.recordCount === 0 || v.rubric !== null ||
        v.rubricStatus !== "not_graded" || v.coverage?.winner.final !== v.coverage?.winner.total ||
        v.coverage?.interval.final !== v.coverage?.interval.total)) ||
      (v.status === "graded" && (!v.lock || !v.ledger || !v.rubric || v.rubricStatus !== "graded" ||
        !["A", "B", "C", "D", "F"].includes(String(v.rubric.overall.grade)) ||
        v.coverage?.winner.final !== v.coverage?.winner.total || v.coverage?.winner.provisional !== 0 ||
        v.coverage?.winner.pending !== 0 || v.coverage?.winner.mismatched !== 0 ||
        v.coverage?.interval.final !== v.coverage?.interval.total || v.coverage?.interval.provisional !== 0 ||
        v.coverage?.interval.pending !== 0 || v.coverage?.interval.mismatched !== 0)) ||
      (v.status === "unavailable" && !v.errorCode)) return null;
    return structuredClone(v);
  } catch { return null; }
}
