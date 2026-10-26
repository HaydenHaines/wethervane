import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ElectionNightArtifactStore, loadElectionNightEvaluationLock, loadElectionNightResultLedger,
  type ElectionNightArtifactPin, type ElectionNightLockLoadRequest, type ElectionNightLedgerLoadRequest,
  type ElectionNightArtifactPublicationManifest,
  type LoadedElectionNightEvaluationLock, type LoadedElectionNightResultLedger } from "./election-night-artifacts.ts";
import { buildElectionNightScoring } from "./election-night-scoring.ts";
import { gradeElectionNightRubric } from "./election-night-rubric.ts";
import type { EvaluationPart, VerifiedElectionNightOutcome } from "./election-night-evaluation.ts";
import type { ElectionNightRubricContract, ElectionNightRubricResult, RubricForecastBinding, RubricLossRecord, RubricTarget } from "./election-night-rubric.ts";
import type { ElectionNightEvaluationCoverageView, ElectionNightEvaluationPartView, ElectionNightEvaluationRowView,
  ElectionNightEvaluationSourceView, ElectionNightEvaluationRawSummaryView, ElectionNightEvaluationView } from "./election-night-evaluation-view.ts";

export type EvaluationChamber = "senate" | "house";
type Pin = ElectionNightArtifactPin;
type LockedRace = LoadedElectionNightEvaluationLock["modelForecast"]["races"][number];
export interface ElectionNightEvaluationPublicationEntry {
  lock: { subject: Pin; publication: Pin; admission: Pin };
  ledger: { subject: Pin; publication: Pin; admission: Pin };
}
export interface ElectionNightEvaluationPublicationConfig {
  schema: "ElectionNightEvaluationPublicationConfigV1";
  chambers: Partial<Record<EvaluationChamber, ElectionNightEvaluationPublicationEntry>>;
}
export type ElectionNightEvaluationServiceResult = ElectionNightEvaluationView;

const unavailableCodes = ["configuration_invalid", "lock_verification_failed", "ledger_verification_failed", "scoring_verification_failed"] as const;
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === "string" && value.trim().length > 0;
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const pin = (value: unknown): value is Pin => object(value) && hash(value.sha256) && Number.isSafeInteger(value.byteLength) &&
  Number(value.byteLength) >= 0 && value.path === `blobs/sha256/${value.sha256}`;

function validPublicationConfig(value: unknown): value is ElectionNightEvaluationPublicationConfig {
  if (!object(value) || value.schema !== "ElectionNightEvaluationPublicationConfigV1" || !object(value.chambers) ||
    Object.keys(value).some(key => !["schema", "chambers"].includes(key)) ||
    Object.keys(value.chambers).some(key => key !== "senate" && key !== "house")) return false;
  for (const chamber of ["senate", "house"] as const) {
    const entry = value.chambers[chamber];
    if (entry === undefined) continue;
    if (!object(entry) || Object.keys(entry).some(key => !["lock", "ledger"].includes(key)) ||
      !object(entry.lock) || Object.keys(entry.lock).some(key => !["subject", "publication", "admission"].includes(key)) ||
      !object(entry.ledger) || Object.keys(entry.ledger).some(key => !["subject", "publication", "admission"].includes(key)) ||
      ![entry.lock.subject, entry.lock.publication, entry.lock.admission,
        entry.ledger.subject, entry.ledger.publication, entry.ledger.admission].every(pin)) return false;
  }
  return true;
}

function sourceView(record: VerifiedElectionNightOutcome, lock: LoadedElectionNightEvaluationLock,
  ledger: LoadedElectionNightResultLedger): ElectionNightEvaluationSourceView {
  const provenance = ledger.provenanceByRecordId[record.id];
  if (!provenance || !hash(provenance.recordSha256) || !hash(provenance.revisionSha256) ||
    !hash(provenance.sourceSha256) || !Array.isArray(provenance.evidenceSha256) || !provenance.evidenceSha256.every(hash) ||
    provenance.sourceSha256 !== record.source.sha256 ||
    provenance.resultPolicySha256 !== lock.manifest.artifacts.resultPolicy.sha256)
    throw Error("Result provenance does not match the verified head.");
  return { recordId: record.id, revision: record.revision, recordSha256: provenance.recordSha256,
    revisionSha256: provenance.revisionSha256, sourceSha256: provenance.sourceSha256, sourceUrl: record.source.url,
    publishedAt: record.source.publishedAt, retrievedAt: record.source.retrievedAt,
    resultPolicySha256: provenance.resultPolicySha256, evidenceSha256: [...provenance.evidenceSha256] };
}
function partView(loss: RubricLossRecord | undefined, evaluated: EvaluationPart<unknown> | undefined,
  target: RubricTarget | null, displayOnly: boolean, exclusion?: string,
  observation: ElectionNightEvaluationPartView["observation"] = null,
  modalPickCorrect: boolean | null = null,
  marginScore: ElectionNightEvaluationPartView["marginScore"] = null): ElectionNightEvaluationPartView {
  if (loss) return { status: loss.status, finality: loss.result?.finality ?? evaluated?.resultStatus ?? "unreported",
    loss: loss.loss, observedValue: loss.observedValue, reason: loss.reason,
    observedValueKind: loss.observedValue === null ? null :
      target?.estimand === "house_official_party_winner" ? "party_group" : target?.estimand === "senate_principal_pair_lead" ? "candidate" : null,
    modalPickCorrect, diagnosticScope: "graded_cohort", observation, marginScore, target: structuredClone(loss.target) };
  if (exclusion) return { status: "not_applicable", finality: evaluated?.resultStatus ?? "unreported",
    loss: null, observedValue: null, reason: exclusion, observedValueKind: null, modalPickCorrect: null,
    diagnosticScope: null, observation, marginScore: null, target: null };
  if (displayOnly) return { status: "not_in_grade_cohort", finality: evaluated?.resultStatus ?? "unreported",
    loss: null, observedValue: null, reason: "Displayed for evaluation tracking; excluded from the predeclared graded cohort.",
    observedValueKind: null, modalPickCorrect, diagnosticScope: modalPickCorrect !== null || marginScore !== null ? "display_only" : null,
    observation, marginScore, target: null };
  return { status: "unresolved", finality: evaluated?.resultStatus ?? "unreported",
    loss: null, observedValue: null, reason: evaluated?.reason ?? "No evaluated target row was produced.",
    observedValueKind: null, modalPickCorrect: null, diagnosticScope: null, observation, marginScore: null, target };
}
function pairedByRace(rows: readonly { raceId: string; model: RubricLossRecord; baseline: RubricLossRecord }[]) {
  return new Map(rows.map(row => [row.raceId, row]));
}
function coverage(value: ElectionNightRubricResult["winner"]["coverage"]): ElectionNightEvaluationCoverageView {
  return { total: value.total, final: value.final, provisional: value.provisional, pending: value.pending,
    mismatched: value.mismatched, unexpected: value.unexpected };
}
function availableMatchedRows(value: ElectionNightRubricResult["winner"]["raw"]): ElectionNightEvaluationRawSummaryView {
  return { scope: value.scope, count: value.count, finalCount: value.finalCount, provisionalCount: value.provisionalCount,
    modelMeanLoss: value.modelMeanLoss, baselineMeanLoss: value.baselineMeanLoss,
    meanPairedLossDifference: value.meanPairedLossDifference };
}
function exactMarginScore(evaluated: EvaluationPart<unknown> | undefined): ElectionNightEvaluationPartView["marginScore"] {
  if (evaluated?.status !== "scored" || !evaluated.score || typeof evaluated.score !== "object") return null;
  const score = evaluated.score as { pointRule?: unknown; predictedMarginPp?: unknown; observedMarginPp?: unknown;
    absoluteErrorPp?: unknown; interval90?: { lower?: unknown; upper?: unknown; covered?: unknown } };
  if (!["mean", "median"].includes(String(score.pointRule)) ||
    ![score.predictedMarginPp, score.observedMarginPp, score.absoluteErrorPp, score.interval90?.lower, score.interval90?.upper]
      .every(value => typeof value === "number" && Number.isFinite(value)) || typeof score.interval90?.covered !== "boolean") return null;
  return { pointRule: score.pointRule as "mean" | "median", predictedMarginPp: score.predictedMarginPp as number,
    observedMarginPp: score.observedMarginPp as number, absoluteErrorPp: score.absoluteErrorPp as number,
    interval90: { lower: score.interval90!.lower as number, upper: score.interval90!.upper as number, covered: score.interval90!.covered } };
}
function publicObservation(record: VerifiedElectionNightOutcome | undefined, kind: "winner" | "margin",
  evaluated: EvaluationPart<unknown> | undefined, race: LockedRace,
  chamber: "senate" | "house", target?: { estimand?: string }): ElectionNightEvaluationPartView["observation"] {
  if (!record || !evaluated || !target || evaluated.status === "mismatched" || evaluated.status === "unsupported") return null;
  if (kind === "winner") {
    const value = record.winner;
    if (!value || value.status !== evaluated.resultStatus || !["called", "provisional", "final", "certified"].includes(value.status)) return null;
    if (typeof value.outcomeId !== "string") return null;
    if (chamber === "house") {
      if (value.estimand !== "house_official_party_winner" || value.kind !== "group" || !["D", "R", "O"].includes(value.outcomeId)) return null;
      const labels = { D: "Democratic party group", R: "Republican party group", O: "Other party group" } as const;
      return { value: value.outcomeId, label: labels[value.outcomeId as keyof typeof labels], status: value.status as "called" | "provisional" | "final" | "certified" };
    }
    if (value.estimand !== "conditional_principal_pair_winner" || !race.pair ||
      ![race.pair.firstCandidateId, race.pair.secondCandidateId].includes(value.outcomeId)) return null;
    const candidate = race.outcomes.flatMap(outcome => outcome.members).find(member => member.id === value.outcomeId);
    return { value: value.outcomeId, label: candidate?.name ?? null,
      status: value.status as "called" | "provisional" | "final" | "certified" };
  }
  const value = record.margin;
  if (!value || value.status !== evaluated.resultStatus || !["called", "provisional", "final", "certified"].includes(value.status)) return null;
  if (value.estimand !== "named_pair_margin" || !Number.isFinite(value.marginPp) || !race.pair ||
    value.firstCandidateId !== race.pair.firstCandidateId || value.secondCandidateId !== race.pair.secondCandidateId ||
    value.units !== "percentage_points" || value.orientation !== "first_minus_second") return null;
  return { value: value.marginPp!, label: `${race.pair.firstCandidateName} minus ${race.pair.secondCandidateName} percentage points`,
    status: value.status as "called" | "provisional" | "final" | "certified" };
}
function lockedProjection(race: LockedRace) {
  return { seatId: race.seatId, state: race.state, electionLabel: race.electionLabel,
    modalPick: { name: race.modalPick.name, kind: race.modalPick.kind, probability: race.modalPick.probability },
    pair: race.pair ? { firstCandidateName: race.pair.firstCandidateName, secondCandidateName: race.pair.secondCandidateName,
      meanMarginPp: race.pair.meanMarginPp, range90: { lower: race.pair.range90.lower, upper: race.pair.range90.upper } } : null };
}
function lockView(lock: LoadedElectionNightEvaluationLock) {
  const contract = lock.comparisonContract;
  const forecastIds = lock.modelForecast.races.map(race => race.id);
  return { id: lock.manifest.id, sha256: lock.lockSha256, frozenAt: lock.manifest.frozenAt,
    model: structuredClone(lock.manifest.forecastBindings.model), baseline: structuredClone(lock.manifest.forecastBindings.baseline),
    comparison: { id: contract.id, sha256: contract.sha256, label: contract.label, mode: contract.comparisonMode },
    rubricDecisionSha256: contract.rubricDecisionSha256,
    cohort: { displayRaceIds: contract.displayRaceIds ? [...contract.displayRaceIds] : [...forecastIds],
      winnerRaceIds: contract.winnerCohort.map(row => row.raceId), intervalRaceIds: contract.intervalCohort.map(row => row.raceId),
      intervalExclusions: contract.intervalExclusions.map(row => ({ raceId: row.raceId, reason: row.reason })) } };
}
function incomplete(chamber: EvaluationChamber, status: ElectionNightEvaluationView["status"], errorCode: ElectionNightEvaluationView["errorCode"],
  lock: ElectionNightEvaluationView["lock"] = null): ElectionNightEvaluationView {
  return { schema: "ElectionNightEvaluationViewV1", chamber, status, errorCode, lock, ledger: null,
    coverage: null, availableMatchedRows: null, rows: [], rubricStatus: null, rubric: null };
}

/** Convert authenticated Node-only objects to the deliberately small browser contract. */
export function buildElectionNightEvaluationView(chamber: EvaluationChamber, lock: LoadedElectionNightEvaluationLock,
  ledger: LoadedElectionNightResultLedger): ElectionNightEvaluationView {
  try {
    const bridge = buildElectionNightScoring(lock, ledger);
    const rubric = gradeElectionNightRubric(bridge.rubricInput);
    const contract = lock.comparisonContract as ElectionNightRubricContract;
    const winnerRows = pairedByRace(bridge.rubricInput.winnerRows), intervalRows = pairedByRace(bridge.rubricInput.intervalRows);
    const exclusions = new Map(contract.intervalExclusions.map(row => [row.raceId, row.reason]));
    const modelEval = new Map(bridge.modelEvaluation.rows.map(row => [row.raceId, row]));
    const baselineEval = new Map(bridge.baselineEvaluation.rows.map(row => [row.raceId, row]));
    const records = new Map(ledger.records.map(record => [record.id, record]));
    const forecastRaces = new Map(lock.modelForecast.races.map(race => [race.id, race]));
    const displayIds = contract.displayRaceIds ?? lock.modelForecast.races.map(race => race.id);
    const displayOnly = new Set(bridge.displayOnlyRaceIds);
    const rows: ElectionNightEvaluationRowView[] = displayIds.map(raceId => {
      const m = modelEval.get(raceId), b = baselineEval.get(raceId), winner = winnerRows.get(raceId), interval = intervalRows.get(raceId);
      const forecast = forecastRaces.get(raceId);
      if (!forecast) throw Error("Locked display race is absent from the frozen model projection.");
      const recordIds = new Set([...(m?.records ?? []).map(row => row.id), ...(b?.records ?? []).map(row => row.id)]);
      const sources = [...recordIds].map(id => records.get(id)).filter((record): record is VerifiedElectionNightOutcome => !!record)
        .map(record => sourceView(record, lock, ledger));
      const currentRecord = m?.records[0] ? records.get(m.records[0].id) : undefined;
      const winnerTarget = winner?.model.target ?? null, marginTarget = interval?.model.target ?? null;
      const modelProtocolRace = lock.modelProtocol.races.find(row => row.raceId === raceId);
      const baselineProtocolRace = lock.baselineProtocol.races.find(row => row.raceId === raceId);
      const baselineForecast = lock.baselineForecast.races.find(row => row.id === raceId);
      if (!baselineForecast) throw Error("Locked baseline race is absent from the frozen display cohort.");
      const marginExclusion = exclusions.get(raceId);
      return { raceId, displayOnly: displayOnly.has(raceId), forecast: lockedProjection(forecast),
        winner: { model: partView(winner?.model, m?.winner, winnerTarget, displayOnly.has(raceId), undefined,
          publicObservation(currentRecord, "winner", m?.winner, forecast, chamber, winnerTarget ?? modelProtocolRace?.winner),
          (m?.winner?.score as { modalPickCorrect?: boolean } | null)?.modalPickCorrect ?? null),
          baseline: partView(winner?.baseline, b?.winner, winner?.baseline.target ?? null, displayOnly.has(raceId), undefined,
            publicObservation(currentRecord, "winner", b?.winner, baselineForecast, chamber, winner?.baseline.target ?? baselineProtocolRace?.winner),
            (b?.winner?.score as { modalPickCorrect?: boolean } | null)?.modalPickCorrect ?? null) },
        margin: { model: partView(interval?.model, m?.margin, marginTarget, displayOnly.has(raceId), marginExclusion,
          publicObservation(currentRecord, "margin", m?.margin, forecast, chamber, marginTarget ?? modelProtocolRace?.margin ?? undefined), null,
          marginExclusion ? null : exactMarginScore(m?.margin)),
          baseline: partView(interval?.baseline, b?.margin, interval?.baseline.target ?? null, displayOnly.has(raceId), marginExclusion,
            publicObservation(currentRecord, "margin", b?.margin, baselineForecast, chamber,
              interval?.baseline.target ?? baselineProtocolRace?.margin ?? undefined), null,
            marginExclusion ? null : exactMarginScore(b?.margin)) },
        sources };
    });
    const viewLedger = { id: ledger.manifest.id, sha256: ledger.ledgerSha256, chosenHeadSha256: ledger.chosenHeadSha256,
      revisionCount: ledger.history.length, recordCount: ledger.records.length,
      provenance: ledger.records.map(record => sourceView(record, lock, ledger)) };
    let status: ElectionNightEvaluationView["status"];
    if (ledger.chosenHeadSha256 === null && ledger.records.length === 0) status = "locked_no_results";
    else if (rubric.status === "graded") status = "graded";
    else if (rubric.winner.complete && rubric.interval.complete) status = "complete_score_not_gradeable";
    else status = "partial_results";
    return { schema: "ElectionNightEvaluationViewV1", chamber, status, errorCode: null,
      lock: lockView(lock), ledger: viewLedger,
      coverage: { winner: coverage(rubric.winner.coverage), interval: coverage(rubric.interval.coverage) }, rows,
      availableMatchedRows: { winner: availableMatchedRows(rubric.winner.raw), interval: availableMatchedRows(rubric.interval.raw) },
      rubricStatus: rubric.status, rubric: rubric.status === "graded" ? structuredClone(rubric) : null };
  } catch {
    return incomplete(chamber, "unavailable", "scoring_verification_failed");
  }
}

function pinFromParts(value: unknown): ElectionNightArtifactPin | null {
  if (!object(value) || !hash(value.sha256) || !Number.isSafeInteger(value.byteLength) || Number(value.byteLength) < 0 ||
    value.path !== `blobs/sha256/${value.sha256}`) return null;
  return value as unknown as ElectionNightArtifactPin;
}
function requestFromEntry(entry: ElectionNightEvaluationPublicationEntry): {
  lock: ElectionNightLockLoadRequest; ledger: Omit<ElectionNightLedgerLoadRequest, "evaluationLock">;
} | null {
  const lock = { lock: pinFromParts(entry.lock.subject), trustedPublication: pinFromParts(entry.lock.publication),
    trustedAdmission: pinFromParts(entry.lock.admission) };
  const ledger = { ledger: pinFromParts(entry.ledger.subject), trustedPublication: pinFromParts(entry.ledger.publication),
    trustedAdmission: pinFromParts(entry.ledger.admission) };
  if (Object.values(lock).some(value => !value) || Object.values(ledger).some(value => !value)) return null;
  return { lock: lock as ElectionNightLockLoadRequest, ledger: ledger as Omit<ElectionNightLedgerLoadRequest, "evaluationLock"> };
}
function isMissing(error: unknown): boolean { return (error as NodeJS.ErrnoException)?.code === "ENOENT"; }

/** Server entry point. Trust configuration and storage roots are fixed beneath repository data/. */
export async function loadConfiguredElectionNightEvaluation(chamber: EvaluationChamber,
  dataRoot = join(process.cwd(), "data")): Promise<ElectionNightEvaluationServiceResult> {
  let config: unknown;
  try {
    const bytes = await readFile(join(dataRoot, "election-night", "evaluation-publication.json"));
    config = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    if (isMissing(error)) return incomplete(chamber, "lock_not_configured", null);
    return incomplete(chamber, "unavailable", "configuration_invalid");
  }
  if (!validPublicationConfig(config)) return incomplete(chamber, "unavailable", "configuration_invalid");
  const entry = config.chambers[chamber];
  if (!entry) return incomplete(chamber, "lock_not_configured", null);
  const requests = requestFromEntry(entry);
  if (!requests) return incomplete(chamber, "unavailable", "configuration_invalid");
  const storePath = join(dataRoot, "election-night", "evaluation-artifacts");
  let store: ElectionNightArtifactStore;
  try { store = new ElectionNightArtifactStore(storePath); }
  catch { return incomplete(chamber, "unavailable", "lock_verification_failed"); }
  let lock: LoadedElectionNightEvaluationLock;
  try { lock = loadElectionNightEvaluationLock(store, requests.lock); }
  catch { return incomplete(chamber, "unavailable", "lock_verification_failed"); }
  if (lock.modelForecast.chamber !== chamber || lock.baselineForecast.chamber !== chamber)
    return incomplete(chamber, "unavailable", "lock_verification_failed");
  let ledger: LoadedElectionNightResultLedger;
  try { ledger = loadElectionNightResultLedger(store, { ...requests.ledger, evaluationLock: lock }); }
  catch { return incomplete(chamber, "unavailable", "ledger_verification_failed"); }
  return buildElectionNightEvaluationView(chamber, lock, ledger);
}

export const ELECTION_NIGHT_EVALUATION_ERROR_CODES = unavailableCodes;
