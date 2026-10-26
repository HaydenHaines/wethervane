/**
 * Pure evaluation of separately admitted forecast, protocol and result records.
 * Callers authenticate evidence bytes and scientifically admit the target/round
 * mapping before calling. A SHA-shaped reference or this projection is not approval.
 * No function here acquires results, locks a forecast, changes it, or assigns grades.
 */
import type { ElectionNightForecast, ElectionNightRace } from "./election-night-forecast.ts";

export interface EvaluationEvidence { sha256: string; url: string }
export interface EvaluationSource extends EvaluationEvidence { publishedAt: string; retrievedAt: string }
export interface EvaluationForecastBinding {
  chamber: "senate" | "house"; runId: string; artifactSha256: string; issuedAt: string; cutoffAt: string;
}
export type EvaluationResultStatus = "unreported" | "provisional" | "called" | "final" | "certified" | "unresolved" | "unsupported";
export type WinnerEstimand = "conditional_principal_pair_winner" | "listed_modeled_winner";
export interface EvaluationWinnerTarget {
  estimand: WinnerEstimand; eventId: string; roundId: string; ruleFamily: string | null;
  /** Exactly the listed forecast outcome IDs; no subset selection or renormalization. */
  outcomeIds: string[]; minimumStatus: "called" | "final" | "certified";
}
export interface EvaluationMarginTarget {
  estimand: "named_pair_margin"; eventId: string; roundId: string; ruleFamily: string | null;
  firstCandidateId: string; secondCandidateId: string; denominator: string;
  units: "percentage_points"; orientation: "first_minus_second";
  pointRule: "mean" | "median"; minimumStatus: "final" | "certified";
}
export interface ElectionNightEvaluationProtocol {
  schema: "ElectionNightEvaluationProtocolV1"; id: string;
  forecast: EvaluationForecastBinding;
  /** Reference only. Source authentication and scientific admission belong to the caller. */
  reviewedDefinition: EvaluationEvidence;
  races: { raceId: string; winner: EvaluationWinnerTarget; margin: EvaluationMarginTarget | null }[];
}
export interface EvaluationWinnerObservation {
  status: EvaluationResultStatus; estimand: string; eventId: string; roundId: string; ruleFamily: string | null;
  outcomeId: string | null; kind: "candidate" | "group" | null; candidateId: string | null;
  reason: string | null;
}
export interface EvaluationMarginObservation {
  status: EvaluationResultStatus; estimand: string; eventId: string; roundId: string; ruleFamily: string | null;
  firstCandidateId: string; secondCandidateId: string; denominator: string;
  units: string; orientation: string; marginPp: number | null; reason: string | null;
}
/** Already authenticated source record; this type and its fields do not establish verification. */
export interface VerifiedElectionNightOutcome {
  schema: "ElectionNightOutcomeRecordV1"; id: string; revision: string; raceId: string;
  source: EvaluationSource; winner: EvaluationWinnerObservation | null; margin: EvaluationMarginObservation | null;
}
/** Natural-log loss. Infinity is explicit and survives JSON; it is never clipped or encoded as a finite value. */
export type ElectionNightLogLoss = { status: "finite"; value: number } | { status: "infinite"; value: null };
export interface WinnerEvaluationScore {
  estimand: WinnerEstimand; eventId: string; roundId: string; ruleFamily: string | null;
  outcomeId: string; kind: "candidate" | "group"; candidateId: string | null; modalPickCorrect: boolean;
  /** Unnormalized multicategory Brier: sum_k (p_k - indicator_k)^2, range 0–2. Binary two-class sum is twice binary one-event Brier. */
  brierSum: number; logLoss: ElectionNightLogLoss;
}
export interface MarginEvaluationScore {
  estimand: "named_pair_margin"; eventId: string; roundId: string; ruleFamily: string | null;
  firstCandidateId: string; secondCandidateId: string; denominator: string; units: "percentage_points";
  orientation: "first_minus_second"; pointRule: "mean" | "median";
  predictedMarginPp: number; observedMarginPp: number;
  /** Observed first-minus-second margin minus predicted margin, in percentage points. */
  signedErrorPp: number; absoluteErrorPp: number;
  interval90: { level: 0.9; lower: number; upper: number; widthPp: number; covered: boolean; scorePp: number };
}
export interface EvaluationPart<T> {
  status: "scored" | "unresolved" | "unsupported" | "mismatched";
  reason: string | null; resultStatus: EvaluationResultStatus | null;
  /** Called-winner observations remain provisional even when permitted for provisional scores. */
  provisional: boolean; score: T | null;
}
export interface EvaluationCoverage {
  total: number; scored: number; unresolved: number; unsupported: number; mismatched: number;
  /** Scored subsets; called scores are never counted as final/certified. */
  scoredCalled: number; scoredFinal: number; scoredCertified: number;
}
export interface ElectionNightEvaluation {
  schema: "ElectionNightEvaluationV1"; protocolId: string; reviewedDefinition: EvaluationEvidence;
  forecast: EvaluationForecastBinding;
  rows: { raceId: string; records: { id: string; revision: string; source: EvaluationSource }[];
    winner: EvaluationPart<WinnerEvaluationScore>; margin: EvaluationPart<MarginEvaluationScore> }[];
  coverage: { winner: EvaluationCoverage; margin: EvaluationCoverage; resultRecords: number; unmatchedRecords: number };
  unmatchedRecords: { recordId: string; raceId: string; reason: string }[];
  protocolIssues: { raceId: string; reason: string }[];
  grade: "not_graded";
}
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown): v is string => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
const statuses: EvaluationResultStatus[] = ["unreported", "provisional", "called", "final", "certified", "unresolved", "unsupported"];
const rank = { called: 1, final: 2, certified: 3 };
function requireValue(v: unknown, message: string): asserts v { if (!v) throw Error(`Election-night evaluation: ${message}`); }
function evidence(v: EvaluationEvidence): boolean { return !!v && hash(v.sha256) && typeof v.url === "string" && /^https?:\/\//.test(v.url); }
function source(v: EvaluationSource): boolean {
  return evidence(v) && typeof v.publishedAt === "string" && typeof v.retrievedAt === "string" &&
    Number.isFinite(Date.parse(v.publishedAt)) && Number.isFinite(Date.parse(v.retrievedAt)) && Date.parse(v.publishedAt) <= Date.parse(v.retrievedAt);
}
function sameIds(a: string[], b: string[]): boolean {
  return Array.isArray(a) && a.length === b.length && new Set(a).size === a.length &&
    a.every(text) && [...a].sort().every((id, i) => id === [...b].sort()[i]);
}
function distribution(outcomes: readonly { id: string; winProbability: number }[], observedId: string): void {
  requireValue(Array.isArray(outcomes) && outcomes.length > 0 && new Set(outcomes.map(o => o.id)).size === outcomes.length &&
    outcomes.every(o => text(o.id) && finite(o.winProbability) && o.winProbability >= 0 && o.winProbability <= 1) &&
    Math.abs(outcomes.reduce((s, o) => s + o.winProbability, 0) - 1) <= 1e-8, "invalid categorical forecast distribution");
  requireValue(outcomes.some(o => o.id === observedId), "observed outcome is outside declared support");
}
export function multicategoryBrier(outcomes: readonly { id: string; winProbability: number }[], observedId: string): number {
  distribution(outcomes, observedId);
  return outcomes.reduce((s, o) => s + (o.winProbability - Number(o.id === observedId)) ** 2, 0);
}
export function categoricalLogLoss(outcomes: readonly { id: string; winProbability: number }[], observedId: string): ElectionNightLogLoss {
  distribution(outcomes, observedId);
  const p = outcomes.find(o => o.id === observedId)!.winProbability;
  return p === 0 ? { status: "infinite", value: null } : { status: "finite", value: p === 1 ? 0 : -Math.log(p) };
}
/** Central 90% interval score: width + (2/.1)*distance outside, inclusive coverage at endpoints. */
export function central90IntervalScore(lower: number, upper: number, observed: number): MarginEvaluationScore["interval90"] {
  requireValue([lower, upper, observed].every(finite) && lower <= upper, "invalid central-90% interval or observation");
  const widthPp = upper - lower;
  const scorePp = widthPp + 20 * Math.max(lower - observed, 0) + 20 * Math.max(observed - upper, 0);
  requireValue(finite(widthPp) && finite(scorePp), "interval score exceeds finite numeric representation");
  return { level: 0.9, lower, upper, widthPp, covered: observed >= lower && observed <= upper, scorePp };
}
function blocked<T>(status: EvaluationPart<T>["status"], reason: string, resultStatus: EvaluationResultStatus | null): EvaluationPart<T> {
  return { status, reason, resultStatus, provisional: resultStatus === "called", score: null };
}
function availability<T>(observation: { status: EvaluationResultStatus; reason: string | null } | null,
  minimum: "called" | "final" | "certified"): EvaluationPart<T> | null {
  if (!observation) return blocked("unresolved", "No separate observed-result field is available.", null);
  if (!statuses.includes(observation.status)) return blocked("mismatched", "Unknown observed-result status.", null);
  if (observation.status === "unsupported") return blocked("unsupported", observation.reason || "Source does not support this target.", observation.status);
  if (!(observation.status in rank) || rank[observation.status as keyof typeof rank] < rank[minimum])
    return blocked("unresolved", observation.reason || `Result has not reached the declared ${minimum} status.`, observation.status);
  return null;
}
function winnerScore(forecast: ElectionNightForecast, race: ElectionNightRace, target: EvaluationWinnerTarget,
  observed: EvaluationWinnerObservation | null): EvaluationPart<WinnerEvaluationScore> {
  const expectedEstimand = forecast.chamber === "senate" ? "conditional_principal_pair_winner" : "listed_modeled_winner";
  if (forecast.chamber === "senate" && (!race.pair || race.outcomes.length !== 2 ||
    !sameIds(race.outcomes.map(o => o.members.length === 1 && o.kind === "candidate" ? o.members[0].id : ""),
      [race.pair.firstCandidateId, race.pair.secondCandidateId])))
    return blocked("mismatched", "Senate conditional winner target requires its exact two named candidate outcomes.", observed?.status ?? null);
  if (!target || target.estimand !== expectedEstimand || !text(target.eventId) || !text(target.roundId) ||
    target.ruleFamily !== race.ruleFamily || !sameIds(target.outcomeIds, race.outcomes.map(o => o.id)) ||
    !["called", "final", "certified"].includes(target.minimumStatus))
    return blocked("mismatched", "Protocol winner target does not match the frozen listed outcome space/rule.", observed?.status ?? null);
  // Target identities are checked even for partial results; no finality is inferred from an event ID.
  if (observed && (observed.estimand !== target.estimand || observed.eventId !== target.eventId ||
    observed.roundId !== target.roundId || observed.ruleFamily !== target.ruleFamily))
    return blocked("mismatched", "Observed winner target, event, round or rule differs from the protocol.", observed.status);
  const unavailable = availability<WinnerEvaluationScore>(observed, target.minimumStatus);
  if (unavailable) return unavailable;
  const truth = observed!;
  const outcome = race.outcomes.find(o => o.id === truth.outcomeId);
  if (!outcome) return blocked("unsupported", "Observed winner is outside the frozen listed modeled outcomes; no probability is reassigned.", truth.status);
  if (truth.kind !== outcome.kind || (outcome.kind === "candidate" && truth.candidateId !== outcome.members[0].id) ||
    (truth.candidateId !== null && !outcome.members.some(m => m.id === truth.candidateId)))
    return blocked("mismatched", "Observed candidate/group identity does not match the frozen outcome membership.", truth.status);
  distribution(race.outcomes, outcome.id);
  const maximum = Math.max(...race.outcomes.map(o => o.winProbability));
  const modalId = race.outcomes.filter(o => o.winProbability === maximum).map(o => o.id).sort()[0];
  return { status: "scored", reason: null, resultStatus: truth.status, provisional: truth.status === "called",
    score: { estimand: target.estimand, eventId: target.eventId, roundId: target.roundId, ruleFamily: target.ruleFamily,
      outcomeId: outcome.id, kind: outcome.kind, candidateId: truth.candidateId, modalPickCorrect: modalId === outcome.id,
      brierSum: multicategoryBrier(race.outcomes, outcome.id), logLoss: categoricalLogLoss(race.outcomes, outcome.id) } };
}
function marginScore(race: ElectionNightRace, target: EvaluationMarginTarget | null,
  observed: EvaluationMarginObservation | null): EvaluationPart<MarginEvaluationScore> {
  const pair = race.pair;
  if (!pair) return blocked("unsupported", "The frozen forecast has no named-pair margin distribution.", observed?.status ?? null);
  if (!target) return blocked("unsupported", "No admitted pair-margin evaluation target is specified in this protocol.", observed?.status ?? null);
  if (target.estimand !== "named_pair_margin" || !text(target.eventId) || !text(target.roundId) || target.ruleFamily !== race.ruleFamily ||
    target.firstCandidateId !== pair.firstCandidateId || target.secondCandidateId !== pair.secondCandidateId || target.denominator !== pair.denominator ||
    target.units !== pair.units || target.orientation !== pair.orientation || !["mean", "median"].includes(target.pointRule) ||
    !["final", "certified"].includes(target.minimumStatus))
    return blocked("mismatched", "Protocol margin target does not match the frozen named pair, denominator, orientation, units or rule.", observed?.status ?? null);
  if (observed && (observed.estimand !== target.estimand || observed.eventId !== target.eventId || observed.roundId !== target.roundId ||
    observed.ruleFamily !== target.ruleFamily || observed.firstCandidateId !== target.firstCandidateId || observed.secondCandidateId !== target.secondCandidateId ||
    observed.denominator !== target.denominator || observed.units !== target.units || observed.orientation !== target.orientation))
    return blocked("mismatched", "Observed margin target, round, named candidates, denominator, orientation, units or rule differs from the protocol.", observed.status);
  // A winner call cannot authorize a margin score, even when a number is present.
  if (observed?.status === "called") return blocked("unresolved", "A called winner does not establish a final vote margin.", observed.status);
  const unavailable = availability<MarginEvaluationScore>(observed, target.minimumStatus);
  if (unavailable) return unavailable;
  const truth = observed!, point = target.pointRule === "mean" ? pair.meanMarginPp : pair.medianMarginPp;
  if (!finite(truth.marginPp) || Math.abs(truth.marginPp) > 100 || !finite(point) || Math.abs(point) > 100 ||
    ![pair.range90.lower, pair.range90.upper].every(v => finite(v) && Math.abs(v) <= 100) || pair.range90.lower > pair.range90.upper)
    return blocked("mismatched", "Named-pair margin or frozen prediction is invalid on its percentage-point scale.", truth.status);
  const signedErrorPp = truth.marginPp - point;
  return { status: "scored", reason: null, resultStatus: truth.status, provisional: false,
    score: { estimand: "named_pair_margin", eventId: target.eventId, roundId: target.roundId, ruleFamily: target.ruleFamily,
      firstCandidateId: target.firstCandidateId, secondCandidateId: target.secondCandidateId, denominator: target.denominator,
      units: "percentage_points", orientation: "first_minus_second", pointRule: target.pointRule,
      predictedMarginPp: point, observedMarginPp: truth.marginPp, signedErrorPp, absoluteErrorPp: Math.abs(signedErrorPp),
      interval90: central90IntervalScore(pair.range90.lower, pair.range90.upper, truth.marginPp) } };
}
function coverage(parts: EvaluationPart<unknown>[]): EvaluationCoverage {
  const value: EvaluationCoverage = { total: parts.length, scored: 0, unresolved: 0, unsupported: 0, mismatched: 0,
    scoredCalled: 0, scoredFinal: 0, scoredCertified: 0 };
  for (const p of parts) {
    value[p.status]++;
    if (p.status === "scored") {
      if (p.resultStatus === "called") value.scoredCalled++;
      else if (p.resultStatus === "final") value.scoredFinal++;
      else if (p.resultStatus === "certified") value.scoredCertified++;
    }
  }
  return value;
}
export function evaluateElectionNightForecast(forecast: ElectionNightForecast, protocol: ElectionNightEvaluationProtocol,
  records: readonly VerifiedElectionNightOutcome[]): ElectionNightEvaluation {
  const binding: EvaluationForecastBinding = { chamber: forecast.chamber, runId: forecast.run.id,
    artifactSha256: forecast.run.artifactSha256, issuedAt: forecast.run.issuedAt, cutoffAt: forecast.run.cutoffAt };
  requireValue(forecast.schema === "ElectionNightForecastV1" && protocol.schema === "ElectionNightEvaluationProtocolV1" && text(protocol.id) &&
    evidence(protocol.reviewedDefinition) && hash(binding.artifactSha256) && Object.entries(binding).every(([k, v]) =>
      protocol.forecast[k as keyof EvaluationForecastBinding] === v), "protocol does not bind the exact forecast edition");
  requireValue(Array.isArray(protocol.races) && Array.isArray(records) && new Set(forecast.races.map(r => r.id)).size === forecast.races.length,
    "invalid protocol, records or duplicate forecast races");
  const targets = new Map<string, ElectionNightEvaluationProtocol["races"]>();
  for (const t of protocol.races) targets.set(t.raceId, [...(targets.get(t.raceId) ?? []), t]);
  const byRace = new Map<string, VerifiedElectionNightOutcome[]>(), recordIds = new Map<string, number>();
  for (const r of records) {
    byRace.set(r.raceId, [...(byRace.get(r.raceId) ?? []), r]); recordIds.set(r.id, (recordIds.get(r.id) ?? 0) + 1);
  }
  const forecastIds = new Set(forecast.races.map(r => r.id));
  const protocolIssues: ElectionNightEvaluation["protocolIssues"] = [];
  for (const [raceId, matches] of targets) if (!forecastIds.has(raceId) || matches.length !== 1)
    protocolIssues.push({ raceId, reason: !forecastIds.has(raceId) ? "Protocol race is outside the frozen cohort." : "Duplicate protocol race targets." });
  const unmatchedRecords = records.filter(r => !forecastIds.has(r.raceId)).map(r => ({ recordId: r.id, raceId: r.raceId,
    reason: "Observed record is outside the frozen forecast cohort." }));
  const rows = forecast.races.map(race => {
    const found = byRace.get(race.id) ?? [], target = targets.get(race.id) ?? [];
    let problem: string | null = null;
    if (target.length !== 1) {
      problem = target.length ? "Duplicate protocol race targets." : "No explicit protocol target for this frozen race.";
      if (!target.length) protocolIssues.push({ raceId: race.id, reason: problem });
    } else if (found.length > 1) problem = "Multiple observed records for one race; select an explicit revision before evaluation.";
    else if (found.length === 1) {
      const r = found[0];
      if (r.schema !== "ElectionNightOutcomeRecordV1" || !text(r.id) || !text(r.revision) || !source(r.source) || recordIds.get(r.id) !== 1)
        problem = "Observed record identity, revision or source evidence is invalid/duplicated.";
    }
    const record = found[0] ?? null;
    return { raceId: race.id, records: found.map(r => ({ id: r.id, revision: r.revision, source: structuredClone(r.source) })),
      winner: problem ? blocked<WinnerEvaluationScore>("mismatched", problem, record?.winner?.status ?? null) :
        winnerScore(forecast, race, target[0].winner, record?.winner ?? null),
      margin: problem ? blocked<MarginEvaluationScore>("mismatched", problem, record?.margin?.status ?? null) :
        marginScore(race, target[0].margin, record?.margin ?? null) };
  });
  return { schema: "ElectionNightEvaluationV1", protocolId: protocol.id, reviewedDefinition: structuredClone(protocol.reviewedDefinition),
    forecast: binding, rows, coverage: { winner: coverage(rows.map(r => r.winner)), margin: coverage(rows.map(r => r.margin)),
      resultRecords: records.length, unmatchedRecords: unmatchedRecords.length }, unmatchedRecords, protocolIssues, grade: "not_graded" };
}
