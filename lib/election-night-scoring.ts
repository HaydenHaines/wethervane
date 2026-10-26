/**
 * Bridge two byte-authenticated forecast editions and one admitted result head to
 * the editorial rubric. Losses come only from the evaluator; no scalar loss enters
 * this API. The artifact store authenticates bytes and admits the pre-result lock.
 */
import type { ElectionNightForecast, ElectionNightRace } from "./election-night-forecast.ts";
import { central90IntervalScore, evaluateElectionNightForecast, type ElectionNightEvaluation,
  type ElectionNightEvaluationProtocol, type EvaluationPart, type VerifiedElectionNightOutcome,
  type WinnerEvaluationScore, type MarginEvaluationScore } from "./election-night-evaluation.ts";
import { assertLoadedEvaluationLock, assertLoadedResultLedger,
  type LoadedElectionNightEvaluationLock, type LoadedElectionNightResultLedger,
  type ElectionNightRecordProvenance } from "./election-night-artifacts.ts";
import { ELECTION_NIGHT_RUBRIC } from "./election-night-rubric.ts";
import type { ElectionNightRubricContract, ElectionNightRubricInput, PairedRubricLossRecord,
  RubricForecastBinding, RubricIntervalTarget, RubricLossRecord, RubricResultBinding,
  RubricTarget, RubricWinnerTarget } from "./election-night-rubric.ts";

const SENATE_NATIVE_DENOMINATOR = "sum_of_named_principal_candidate_votes_under_declared_rule_proxy";
const SENATE_DISPLAY_DENOMINATOR = "Share of votes for the two named principal contenders";
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
function requireValue(value: unknown, reason: string): asserts value {
  if (!value) throw Error(`Election-night scoring: ${reason}`);
}
function canonical(value: unknown): string {
  return JSON.stringify(value, (_, item) => item && typeof item === "object" && !Array.isArray(item)
    ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item);
}
const same = (a: unknown, b: unknown): boolean => canonical(a) === canonical(b);
const ids = (values: readonly string[]): boolean => values.every(text) && new Set(values).size === values.length;

export interface ElectionNightScoringBridge {
  lockSha256: string; ledgerSha256: string; chosenHeadSha256: string | null;
  /** Complete evaluator rows remain available even when a reviewed grade cohort is narrower. */
  modelEvaluation: ElectionNightEvaluation; baselineEvaluation: ElectionNightEvaluation;
  rubricInput: ElectionNightRubricInput;
  displayOnlyRaceIds: string[];
}

/** Owner's certified-only rule applies to every displayed row, including ungraded AK/GA/ME. */
export function assertEarlierSenateCertifiedMinimums(contract: ElectionNightRubricContract,
  modelForecast: ElectionNightForecast, baselineForecast: ElectionNightForecast,
  modelProtocol: ElectionNightEvaluationProtocol, baselineProtocol: ElectionNightEvaluationProtocol): void {
  if (contract.comparisonMode !== "explicit_earlier_benchmark" || contract.model.chamber !== "senate" ||
    contract.resultPolicy.sha256 !== ELECTION_NIGHT_RUBRIC.earlierBenchmark.resultPolicySha256) return;
  for (const [role, forecast, protocol] of [["model", modelForecast, modelProtocol],
    ["baseline", baselineForecast, baselineProtocol]] as const) {
    requireValue(forecast.races.length === ELECTION_NIGHT_RUBRIC.earlierBenchmark.cohortSize &&
      protocol.races.length === forecast.races.length &&
      forecast.races.every((race, i) => race.pair !== null && protocol.races[i]?.raceId === race.id &&
        protocol.races[i].winner?.minimumStatus === "certified" &&
        protocol.races[i].margin?.minimumStatus === "certified"),
    `${role} earlier Senate result policy requires certified winner and named-pair margin targets for all 35 displayed races.`);
  }
}

function verifyForecast(role: "model" | "baseline", lock: LoadedElectionNightEvaluationLock,
  forecast: ElectionNightForecast, protocol: ElectionNightEvaluationProtocol,
  binding: RubricForecastBinding): void {
  const artifacts = lock.manifest.artifacts, projection = lock.manifest.projections[role];
  const source = role === "model" ? artifacts.modelSource : artifacts.baselineSource;
  const projected = role === "model" ? artifacts.modelForecast : artifacts.baselineForecast;
  const protocolPin = role === "model" ? artifacts.modelProtocol : artifacts.baselineProtocol;
  const nativeSha = forecast.chamber === "senate" ? forecast.sourceHashes.forecastSha256 : forecast.run.artifactSha256;
  requireValue(forecast.schema === "ElectionNightForecastV1" && forecast.editionStatus === "dated_snapshot" &&
    forecast.chamber === binding.chamber && forecast.run.id === binding.runId &&
    forecast.run.artifactSha256 === binding.resultSha256 && nativeSha === binding.nativeForecastSha256 &&
    forecast.run.targetId === binding.targetId && forecast.run.lawId === binding.lawId &&
    forecast.run.issuedAt === binding.issuedAt && forecast.run.cutoffAt === binding.cutoffAt &&
    source.sha256 === projection.sourceSha256 && source.sha256 === binding.resultSha256 &&
    projected.sha256 === projection.projectionSha256 && projection.reviewSha256 === lock.manifest.reviews.projection.sha256 &&
    hash(projected.sha256) && hash(protocolPin.sha256) && same(lock.manifest.forecastBindings[role], binding),
  `${role} original/projection/native forecast identity differs from the admitted lock.`);
  requireValue(protocol.schema === "ElectionNightEvaluationProtocolV1" &&
    protocol.forecast.chamber === forecast.chamber && protocol.forecast.runId === forecast.run.id &&
    protocol.forecast.artifactSha256 === forecast.run.artifactSha256 &&
    protocol.forecast.issuedAt === forecast.run.issuedAt && protocol.forecast.cutoffAt === forecast.run.cutoffAt &&
    Array.isArray(protocol.races) && protocol.races.length === forecast.races.length,
  `${role} evaluator protocol does not bind every forecast race and exact source edition.`);
  const forecastIds = forecast.races.map(race => race.id), protocolIds = protocol.races.map(target => target.raceId);
  requireValue(ids(forecastIds) && ids(protocolIds) && same([...forecastIds].sort(), [...protocolIds].sort()),
    `${role} evaluator protocol race roster is incomplete or duplicated.`);
}

function admittedRecords(lock: LoadedElectionNightEvaluationLock,
  ledger: LoadedElectionNightResultLedger): { records: VerifiedElectionNightOutcome[];
  byRace: Map<string, { record: VerifiedElectionNightOutcome; provenance: ElectionNightRecordProvenance }> } {
  requireValue(ledger.lockSha256 === lock.lockSha256 && ledger.manifest.evaluationLockSha256 === lock.lockSha256 &&
    ledger.chosenHeadSha256 === ledger.manifest.chosenHeadSha256 &&
    Array.isArray(ledger.records) && ledger.provenanceByRecordId && typeof ledger.provenanceByRecordId === "object",
  "Result ledger does not bind the admitted evaluation lock and selected head.");
  const records = structuredClone(ledger.records) as VerifiedElectionNightOutcome[];
  const byRace = new Map<string, { record: VerifiedElectionNightOutcome; provenance: ElectionNightRecordProvenance }>();
  const usedIds = new Set<string>();
  for (const record of records) {
    const provenance = ledger.provenanceByRecordId[record.id];
    requireValue(record.schema === "ElectionNightOutcomeRecordV1" && text(record.id) && text(record.revision) && text(record.raceId) &&
      !usedIds.has(record.id) && !byRace.has(record.raceId) && provenance &&
      provenance.recordId === record.id && provenance.revision === record.revision &&
      hash(provenance.recordSha256) && provenance.sourceSha256 === record.source?.sha256 &&
      provenance.resultPolicySha256 === lock.comparisonContract.resultPolicy.sha256,
    "Current result head has a duplicate or unbound record, revision, source or policy.");
    usedIds.add(record.id); byRace.set(record.raceId, { record, provenance });
  }
  requireValue(Object.keys(ledger.provenanceByRecordId).length === usedIds.size,
    "Current result provenance differs from the selected records.");
  return { records, byRace };
}

function denomination(contract: ElectionNightRubricContract, target: RubricTarget, protocolDenominator: string): boolean {
  return target.denominator === protocolDenominator ||
    (contract.comparisonMode === "explicit_earlier_benchmark" && contract.model.chamber === "senate" &&
      target.denominator === SENATE_NATIVE_DENOMINATOR && protocolDenominator === SENATE_DISPLAY_DENOMINATOR);
}

function verifyWinnerTarget(contract: ElectionNightRubricContract, target: RubricWinnerTarget,
  race: ElectionNightRace, protocol: ElectionNightEvaluationProtocol): void {
  const selected = protocol.races.find(row => row.raceId === target.raceId)?.winner;
  requireValue(selected && target.raceId === race.id && selected.eventId === target.eventId &&
    selected.roundId === target.roundId && selected.ruleFamily === race.ruleFamily,
  `Winner target mapping is absent or different for ${target.raceId}.`);
  if (contract.model.chamber === "house") {
    requireValue(target.estimand === "house_official_party_winner" &&
      selected.estimand === "house_official_party_winner" &&
      same(target.outcomeIds, ["D", "R", "O"]) && same(selected.outcomeIds, ["D", "R", "O"]) &&
      target.firstCandidateId === null && target.secondCandidateId === null &&
      selected.resultRuleId === target.resultRuleId && selected.denominator === target.denominator &&
      selected.interpretationSha256 === contract.houseMappingReviewSha256,
    `House fixed D/R/O target or official result rule differs for ${target.raceId}.`);
  } else {
    const pair = race.pair;
    requireValue(target.estimand === "senate_principal_pair_lead" && selected.estimand === "conditional_principal_pair_winner" &&
      pair && target.firstCandidateId === pair.firstCandidateId && target.secondCandidateId === pair.secondCandidateId &&
      selected.resultRuleId === target.resultRuleId &&
      same(target.outcomeIds, [pair.firstCandidateId, pair.secondCandidateId]) &&
      same(selected.outcomeIds, target.outcomeIds) &&
      denomination(contract, target, pair.denominator),
    `Senate principal-pair lead target differs from its frozen named pair for ${target.raceId}.`);
  }
}

function verifyIntervalTarget(contract: ElectionNightRubricContract, target: RubricIntervalTarget,
  race: ElectionNightRace, protocol: ElectionNightEvaluationProtocol): void {
  const selected = protocol.races.find(row => row.raceId === target.raceId)?.margin, pair = race.pair;
  requireValue(selected && pair && target.raceId === race.id && target.estimand === "named_pair_margin" &&
    target.eventId === selected.eventId && target.roundId === selected.roundId && selected.ruleFamily === race.ruleFamily &&
    selected.resultRuleId === target.resultRuleId &&
    target.firstCandidateId === selected.firstCandidateId && target.secondCandidateId === selected.secondCandidateId &&
    target.firstCandidateId === pair.firstCandidateId && target.secondCandidateId === pair.secondCandidateId &&
    target.units === selected.units && target.orientation === selected.orientation && target.intervalLevel === 0.9 &&
    selected.pointRule === "mean" && denomination(contract, target, selected.denominator) &&
    selected.denominator === pair.denominator,
  `Named-pair interval target differs from its frozen forecast/result mapping for ${target.raceId}.`);
}

function resultBinding(part: EvaluationPart<unknown>, provenance: ElectionNightRecordProvenance | null,
  policySha256: string): RubricResultBinding | null {
  if (!provenance) return null;
  requireValue(provenance.resultPolicySha256 === policySha256,
    "Result revision does not bind the exact admitted result policy.");
  const status = part.resultStatus;
  const finality = status === "final" || status === "certified" || status === "called" || status === "provisional" ||
    status === "unsupported" ? status : "unresolved";
  return { recordId: provenance.recordId, revision: provenance.revision,
    recordSha256: provenance.recordSha256, sourceSha256: provenance.sourceSha256,
    resultPolicySha256: provenance.resultPolicySha256, finality };
}

function loss(part: EvaluationPart<WinnerEvaluationScore> | EvaluationPart<MarginEvaluationScore>,
  target: RubricTarget, forecastSha256: string, provenance: ElectionNightRecordProvenance | null,
  policySha256: string, winner: boolean): RubricLossRecord {
  const metric = winner ? "multicategory_brier_sum" : "central90_interval_score_pp";
  const score = part.score;
  requireValue((part.status === "scored") === (score !== null) &&
    (part.status !== "scored" || provenance !== null &&
      (part.resultStatus === "called" || part.resultStatus === "final" || part.resultStatus === "certified")),
  `Evaluator score/status is inconsistent for ${target.raceId}.`);
  if (!score) return { forecastSha256, target: structuredClone(target), status: part.status,
    metric, result: resultBinding(part, provenance, policySha256), loss: null, observedValue: null, reason: part.reason };
  if (winner) {
    const value = score as WinnerEvaluationScore;
    requireValue(finite(value.brierSum) && value.brierSum >= 0 && value.brierSum <= 2 &&
      value.eventId === target.eventId && value.roundId === target.roundId &&
      value.resultRuleId === target.resultRuleId &&
      value.outcomeId && (target as RubricWinnerTarget).outcomeIds.includes(value.outcomeId) &&
      value.estimand === ((target as RubricWinnerTarget).estimand === "house_official_party_winner"
        ? "house_official_party_winner" : "conditional_principal_pair_winner"),
    `Evaluator winner score differs from the frozen rubric target for ${target.raceId}.`);
    return { forecastSha256, target: structuredClone(target), status: "scored", metric,
      result: resultBinding(part, provenance, policySha256), loss: value.brierSum,
      observedValue: value.outcomeId, reason: null };
  }
  const value = score as MarginEvaluationScore;
  const interval = central90IntervalScore(value.interval90.lower, value.interval90.upper, value.observedMarginPp);
  requireValue(finite(value.interval90.scorePp) && value.interval90.scorePp >= 0 && value.interval90.scorePp <= 4000 &&
    value.interval90.level === 0.9 && value.eventId === target.eventId && value.roundId === target.roundId &&
    value.resultRuleId === target.resultRuleId &&
    value.firstCandidateId === (target as RubricIntervalTarget).firstCandidateId &&
    value.secondCandidateId === (target as RubricIntervalTarget).secondCandidateId &&
    finite(value.observedMarginPp) && Math.abs(value.observedMarginPp) <= 100 &&
    value.interval90.scorePp === interval.scorePp && value.interval90.widthPp === interval.widthPp &&
    value.interval90.covered === interval.covered,
  `Evaluator interval score differs from the frozen rubric target for ${target.raceId}.`);
  return { forecastSha256, target: structuredClone(target), status: "scored", metric,
    result: resultBinding(part, provenance, policySha256), loss: value.interval90.scorePp,
    observedValue: value.observedMarginPp, reason: null };
}

/** Derive one paired rubric row per declared target, keeping complete evaluator rows for display. */
export function buildElectionNightScoring(lock: LoadedElectionNightEvaluationLock,
  ledger: LoadedElectionNightResultLedger): ElectionNightScoringBridge {
  assertLoadedEvaluationLock(lock); assertLoadedResultLedger(ledger);
  const contract = structuredClone(lock.comparisonContract) as ElectionNightRubricContract;
  const model = structuredClone(lock.modelForecast) as ElectionNightForecast;
  const baseline = structuredClone(lock.baselineForecast) as ElectionNightForecast;
  const modelProtocol = structuredClone(lock.modelProtocol) as ElectionNightEvaluationProtocol;
  const baselineProtocol = structuredClone(lock.baselineProtocol) as ElectionNightEvaluationProtocol;
  requireValue(contract?.schema === "ElectionNightRubricComparisonV1" &&
    same(contract.model, lock.manifest.forecastBindings.model) &&
    same(contract.baseline, lock.manifest.forecastBindings.baseline) &&
    contract.resultPolicy.sha256 === lock.manifest.artifacts.resultPolicy.sha256,
  "Comparison contract, edition bindings or result policy differs from the admitted lock.");
  verifyForecast("model", lock, model, modelProtocol, contract.model);
  verifyForecast("baseline", lock, baseline, baselineProtocol, contract.baseline);
  assertEarlierSenateCertifiedMinimums(contract, model, baseline, modelProtocol, baselineProtocol);
  const modelIds = model.races.map(race => race.id), baselineIds = baseline.races.map(race => race.id);
  requireValue(ids(modelIds) && ids(baselineIds) && same([...modelIds].sort(), [...baselineIds].sort()) &&
    contract.model.chamber === contract.baseline.chamber && contract.model.chamber === model.chamber,
  "Model and baseline do not cover the same forecast race identities.");
  const declaredWinnerIds = contract.winnerCohort.map(target => target.raceId);
  requireValue(ids(declaredWinnerIds) && declaredWinnerIds.every(id => modelIds.includes(id)),
    "Winner cohort contains duplicate or nonforecast races.");
  const displayOnlyRaceIds = modelIds.filter(id => !declaredWinnerIds.includes(id));
  requireValue((contract.displayRaceIds === undefined || same(contract.displayRaceIds, modelIds)) &&
    (contract.ungradedDisplayRaceIds === undefined || same(contract.ungradedDisplayRaceIds, displayOnlyRaceIds)) &&
    (displayOnlyRaceIds.length === 0 || Array.isArray(contract.displayRaceIds) && Array.isArray(contract.ungradedDisplayRaceIds)),
  "The admitted contract must identify every displayed race excluded from its grade cohort.");
  const issuedHouse = model.chamber === "house" &&
    (model.run.id.startsWith("house-current-") || baseline.run.id.startsWith("house-current-"));
  if (issuedHouse) {
    requireValue(model.run.year === 2026 && baseline.run.year === 2026 &&
      model.run.id.startsWith("house-current-") && baseline.run.id.startsWith("house-current-") &&
      model.races.length === 435 && baseline.races.length === 435 &&
      contract.winnerCohort.length === 435 && contract.intervalCohort.length === 412 &&
      contract.intervalExclusions.length === 23 &&
      model.races.filter(race => race.pair !== null).length === 412 &&
      baseline.races.filter(race => race.pair !== null).length === 412,
    "Issued 2026 House lock must retain 435 party winners and the separate 412/23 pair split.");
    const excluded = new Set(contract.intervalExclusions.map(row => row.raceId));
    const marginIds = contract.intervalCohort.map(row => row.raceId);
    requireValue(ids(marginIds) && marginIds.every(id => modelIds.includes(id) && !excluded.has(id)),
      "Issued House margin targets are duplicated or outside the frozen eligible cohort.");
    requireValue(model.races.every(race => excluded.has(race.id) === (race.pair === null)) &&
      baseline.races.every(race => excluded.has(race.id) === (race.pair === null)),
    "Issued House interval exclusions do not match the frozen unavailable pairs.");
  }
  const { records, byRace } = admittedRecords(lock, ledger);
  requireValue(records.every(record => modelIds.includes(record.raceId)),
    "Selected result head contains a race outside both forecast editions.");
  const modelEvaluation = evaluateElectionNightForecast(model, modelProtocol, records);
  const baselineEvaluation = evaluateElectionNightForecast(baseline, baselineProtocol, records);
  requireValue(modelEvaluation.protocolIssues.length === 0 && baselineEvaluation.protocolIssues.length === 0 &&
    modelEvaluation.unmatchedRecords.length === 0 && baselineEvaluation.unmatchedRecords.length === 0,
  "Evaluator found an unmatched result or incomplete frozen protocol.");
  const mRows = new Map(modelEvaluation.rows.map(row => [row.raceId, row]));
  const bRows = new Map(baselineEvaluation.rows.map(row => [row.raceId, row]));
  const mRaces = new Map(model.races.map(race => [race.id, race]));
  const bRaces = new Map(baseline.races.map(race => [race.id, race]));
  const makeRows = (targets: readonly RubricTarget[], winner: boolean): PairedRubricLossRecord[] => targets.map(target => {
    const mRace = mRaces.get(target.raceId), bRace = bRaces.get(target.raceId);
    const mRow = mRows.get(target.raceId), bRow = bRows.get(target.raceId);
    requireValue(mRace && bRace && mRow && bRow, `Rubric race ${target.raceId} is missing from the evaluated forecasts.`);
    if (winner) {
      verifyWinnerTarget(contract, target as RubricWinnerTarget, mRace, modelProtocol);
      verifyWinnerTarget(contract, target as RubricWinnerTarget, bRace, baselineProtocol);
    } else {
      verifyIntervalTarget(contract, target as RubricIntervalTarget, mRace, modelProtocol);
      verifyIntervalTarget(contract, target as RubricIntervalTarget, bRace, baselineProtocol);
    }
    const current = byRace.get(target.raceId) ?? null;
    for (const row of [mRow, bRow]) requireValue(row.records.length === (current ? 1 : 0) &&
      (!current || row.records[0].id === current.record.id && row.records[0].revision === current.record.revision &&
        row.records[0].source.sha256 === current.provenance.sourceSha256),
    `Evaluator result record differs from the admitted head for ${target.raceId}.`);
    return { raceId: target.raceId, comparisonContractSha256: contract.sha256,
      cohortDefinitionSha256: contract.cohortDefinition.sha256,
      model: loss(winner ? mRow.winner : mRow.margin, target, contract.model.resultSha256,
        current?.provenance ?? null, contract.resultPolicy.sha256, winner),
      baseline: loss(winner ? bRow.winner : bRow.margin, target, contract.baseline.resultSha256,
        current?.provenance ?? null, contract.resultPolicy.sha256, winner) };
  });
  const rubricInput: ElectionNightRubricInput = { contract, modelForecast: structuredClone(contract.model),
    baselineForecast: structuredClone(contract.baseline),
    winnerRows: makeRows(contract.winnerCohort, true), intervalRows: makeRows(contract.intervalCohort, false) };
  return { lockSha256: lock.lockSha256, ledgerSha256: ledger.ledgerSha256,
    chosenHeadSha256: ledger.chosenHeadSha256, modelEvaluation, baselineEvaluation, rubricInput,
    displayOnlyRaceIds };
}
