/** Node-only bridge from independently admitted source extraction to a proposed
 * immutable result revision. Raw report interpretation is an explicit external
 * review responsibility: matching hashes alone does not prove the vote counts.
 * This module never issues an admission, chooses a public ledger head or grades.
 */
import { ElectionNightArtifactStore, assertLoadedEvaluationLock, assertLoadedResultLedger,
  type ElectionNightArtifactPin as Pin, type ElectionNightArtifactPublicationManifest,
  type LoadedElectionNightEvaluationLock, type LoadedElectionNightResultLedger,
  type ElectionNightResultRevisionManifest } from './election-night-artifacts.ts';
import { evaluateSenatePairResult, SENATE_PAIR_RESULT_POLICY,
  type SenatePairObservation, type SenateResultPolicyRow } from './election-night-senate-results.ts';
import { senateResultSelector, type SenateResultSelector } from './election-night-senate-targets.ts';
import type { VerifiedElectionNightOutcome, EvaluationResultStatus } from './election-night-evaluation.ts';

export type SenateExtractedObservation = Omit<SenatePairObservation, 'sourcePin' | 'sourceAdmissionReviewPin' |
  'candidateCrosswalkReviewPin' | 'eventRoundReviewPin' | 'candidateCrosswalkReviewed' | 'eventRoundReviewed'>;
export interface SenateSourceExtraction {
  schema: 'SenateSourceExtractionV1'; id: string; preparedAt: string; evaluationLockSha256: string;
  source: { artifact: Pin; url: string; authority: string; publishedAt: string; retrievedAt: string };
  supportingArtifacts: Pin[];
  /** Actual report labels, distinct from the locked selection-policy identifiers. */
  sourceContest: { id: string | null; locator: string; roundId: string | null; roundLabel: string;
    stage: string; eventDate: string; officialObservedFinalRound: boolean | null;
    continuingCandidateIds: string[] | null; runoffRequired: boolean | null; runoffCandidateIds: string[] | null };
  /** Pair keys are exact native IDs; other diagnostic keys identify reviewed source people, not modeled outcomes. */
  candidateCrosswalk: { candidateKey: string; sourceCandidateId: string | null; sourceName: string; locator: string }[];
  /** Human-readable report locations used by the independent extraction review. */
  fieldLocations: { field: string; locator: string }[];
  observation: SenateExtractedObservation;
}
export type SenateObservationReviewRole = 'source' | 'candidateIdentity' | 'eventRound';
export interface SenateObservationReview {
  schema: 'SenateObservationReviewV1'; id: string; role: SenateObservationReviewRole;
  decision: 'admit_extraction'; reviewedAt: string; reviewer: string;
  evaluationLockSha256: string; extractionSha256: string; sourceSha256: string;
  supportingArtifactSha256: string[]; policySha256: string; mappingSha256: string;
}
export interface SenateObservationAdmission {
  schema: 'SenateObservationAdmissionV1'; id: string; evaluationLockSha256: string;
  extraction: Pin; source: Pin; supportingArtifacts: Pin[]; mapping: Pin;
  policySha256: string; mappingSha256: string;
  reviews: Record<SenateObservationReviewRole, Pin>;
}
export interface PrepareSenateResultRequest {
  extraction: Pin;
  /** Caller-selected trust roots; never accepted from the extraction itself. */
  trustedPublication: Pin; trustedAdmission: Pin;
  priorLedger: LoadedElectionNightResultLedger;
  recordRevision: string; revisionId: string;
}
export interface PreparedSenateResultRevision {
  schema: 'PreparedSenateResultRevisionV1'; evaluationLockSha256: string;
  priorLedgerSha256: string; priorHeadSha256: string | null;
  record: Pin; revision: Pin; operatorProjection: Pin;
  /** A separate review and publication decision must select the new ledger head. */
  resultLedgerAdmissionIssued: false; publicHeadChanged: false;
}
const roles: SenateObservationReviewRole[] = ['source', 'candidateIdentity', 'eventRound'];
const text = (v: unknown): v is string => typeof v === 'string' && v.trim().length > 0;
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const at = (v: unknown): v is string => typeof v === 'string' && Number.isFinite(Date.parse(v));
const samePin = (a: Pin, b: Pin) => !!a && !!b && a.sha256 === b.sha256 && a.byteLength === b.byteLength && a.path === b.path;
const sameStrings = (a: readonly string[], b: readonly string[]) => a.length === b.length && a.every((x,i) => x === b[i]);
const sorted = (a: readonly string[]) => [...a].sort();
function need(v: unknown, message: string): asserts v { if (!v) throw Error(`Senate result admission: ${message}`); }
function httpUrl(value: unknown): boolean {
  try { const url = new URL(String(value)); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password; }
  catch { return false; }
}
function uniquePins(pins: readonly Pin[]): Pin[] { return [...new Map(pins.map(pin => [pin.sha256,pin])).values()]; }

function lockedTarget(lock: LoadedElectionNightEvaluationLock, raceId: string): { row: SenateResultPolicyRow; selector: SenateResultSelector; denominator: string } {
  need(lock.modelForecast.chamber === 'senate' && lock.baselineForecast.chamber === 'senate' &&
    lock.manifest.artifacts.resultPolicy.sha256 === SENATE_PAIR_RESULT_POLICY.policySha256,
  'An admitted Senate lock with the exact reviewed policy is required.');
  const cohort = lock.cohort as { policySha256?: string; mappingSha256?: string; senateObservationSelectors?: unknown[] };
  need(cohort?.policySha256 === SENATE_PAIR_RESULT_POLICY.policySha256 &&
    cohort.mappingSha256 === SENATE_PAIR_RESULT_POLICY.mappingSha256 && Array.isArray(cohort.senateObservationSelectors),
  'The lock does not bind the reviewed observation-selector mapping.');
  const entries = cohort.senateObservationSelectors as { policyRow: SenateResultPolicyRow; selector: SenateResultSelector; semanticRole: string }[];
  need(sameStrings(entries.map(e => e?.policyRow?.raceId), lock.modelForecast.races.map(r => r.id)),
    'Observation selectors must preserve every ordered display race.');
  const entry = entries.find(e => e.policyRow.raceId === raceId);
  need(entry && entry.semanticRole === 'predeclared_observation_selection_rule', 'No locked observation selector for this race.');
  const row = entry.policyRow, selector = senateResultSelector(row);
  need(Object.keys(selector).every(k => entry.selector?.[k as keyof SenateResultSelector] === selector[k as keyof SenateResultSelector]),
    'The locked selector differs from the reviewed policy.');
  let denominator = '';
  for (const side of ['model','baseline'] as const) {
    const forecast = side === 'model' ? lock.modelForecast : lock.baselineForecast;
    const protocol = side === 'model' ? lock.modelProtocol : lock.baselineProtocol;
    const race = forecast.races.find(r => r.id === raceId), target = protocol.races.find(r => r.raceId === raceId);
    need(race?.pair && race.seatId === row.seatId && race.state === row.state && race.ruleFamily === null &&
      race.pair.firstCandidateId === row.firstCandidateId && race.pair.secondCandidateId === row.secondCandidateId && target?.margin,
    `${side} forecast does not contain the exact locked candidate pair.`);
    const winner = target.winner, margin = target.margin;
    need([winner,margin].every(t => t.eventId === selector.eventId && t.roundId === selector.roundId &&
      t.resultRuleId === selector.resultRuleId && t.ruleFamily === null && t.minimumStatus === 'certified') &&
      winner.estimand === 'conditional_principal_pair_winner' &&
      sameStrings(winner.outcomeIds,[row.firstCandidateId,row.secondCandidateId]) &&
      margin.estimand === 'named_pair_margin' && margin.firstCandidateId === row.firstCandidateId &&
      margin.secondCandidateId === row.secondCandidateId && margin.denominator === race.pair.denominator &&
      margin.units === 'percentage_points' && margin.orientation === 'first_minus_second' && margin.pointRule === 'mean',
    `${side} protocol differs from the frozen certified pair target.`);
    need(!denominator || denominator === margin.denominator, 'The two protocols use different pair denominators.');
    denominator = margin.denominator;
  }
  return { row, selector, denominator };
}

export function prepareSenateResultRevision(store: ElectionNightArtifactStore,
  lock: LoadedElectionNightEvaluationLock, request: PrepareSenateResultRequest): PreparedSenateResultRevision {
  assertLoadedEvaluationLock(lock); assertLoadedResultLedger(request?.priorLedger);
  const prior = request.priorLedger;
  need(prior.lockSha256 === lock.lockSha256 && text(request.recordRevision) && text(request.revisionId),
    'An authenticated prior ledger for this lock and explicit new revision identifiers are required.');
  need(!prior.history.some(h => h.manifest.id === request.revisionId), 'Revision ID already exists in the preserved history.');
  const publication = store.readJson<ElectionNightArtifactPublicationManifest>(request.trustedPublication);
  need(publication.schema === 'ElectionNightArtifactPublicationV1' && text(publication.id) && Array.isArray(publication.artifacts),
    'Invalid caller-trusted publication manifest.');
  need(new Set(publication.artifacts.map(p => p.sha256)).size === publication.artifacts.length, 'Duplicate published artifact identity.');
  function published(pin: Pin): void {
    need(publication.artifacts.some(p => samePin(p,pin)), 'Artifact is absent from the caller-trusted publication.');
    store.read(pin);
  }
  published(request.trustedAdmission); published(request.extraction);
  const admission = store.readJson<SenateObservationAdmission>(request.trustedAdmission);
  const extraction = store.readJson<SenateSourceExtraction>(request.extraction);
  need(admission.schema === 'SenateObservationAdmissionV1' && text(admission.id) &&
    admission.evaluationLockSha256 === lock.lockSha256 && samePin(admission.extraction,request.extraction) &&
    admission.policySha256 === SENATE_PAIR_RESULT_POLICY.policySha256 && admission.mappingSha256 === SENATE_PAIR_RESULT_POLICY.mappingSha256,
  'Caller-trusted admission does not bind this extraction, lock and reviewed policy.');
  need(extraction.schema === 'SenateSourceExtractionV1' && text(extraction.id) && at(extraction.preparedAt) &&
    extraction.evaluationLockSha256 === lock.lockSha256 && object(extraction.source) &&
    samePin(extraction.source.artifact,admission.source) && httpUrl(extraction.source.url) && text(extraction.source.authority) &&
    at(extraction.source.publishedAt) && at(extraction.source.retrievedAt) &&
    Date.parse(extraction.source.publishedAt) <= Date.parse(extraction.source.retrievedAt) &&
    Date.parse(extraction.source.retrievedAt) <= Date.parse(extraction.preparedAt),
  'Extraction source, times or exact source identity are invalid.');
  published(admission.source);
  need(admission.mapping?.sha256 === SENATE_PAIR_RESULT_POLICY.mappingSha256, 'Exact raw owner mapping bytes are required.');
  published(admission.mapping);
  need(Array.isArray(extraction.supportingArtifacts) && Array.isArray(admission.supportingArtifacts) &&
    sameStrings(sorted(extraction.supportingArtifacts.map(p => p.sha256)), sorted(admission.supportingArtifacts.map(p => p.sha256))) &&
    new Set(admission.supportingArtifacts.map(p => p.sha256)).size === admission.supportingArtifacts.length,
  'Supporting source/certification evidence differs from the admitted extraction.');
  for (const pin of [...extraction.supportingArtifacts,...admission.supportingArtifacts]) published(pin);
  need(object(admission.reviews) && sameStrings(sorted(Object.keys(admission.reviews)),sorted(roles)), 'All three independent review roles must be explicitly bound.');
  for (const role of roles) {
    const pin = admission.reviews[role]; published(pin);
    const review = store.readJson<SenateObservationReview>(pin);
    need(review.schema === 'SenateObservationReviewV1' && text(review.id) && review.role === role &&
      review.decision === 'admit_extraction' && text(review.reviewer) && at(review.reviewedAt) &&
      Date.parse(review.reviewedAt) >= Date.parse(extraction.preparedAt) &&
      review.evaluationLockSha256 === lock.lockSha256 && review.extractionSha256 === request.extraction.sha256 &&
      review.sourceSha256 === admission.source.sha256 && review.policySha256 === admission.policySha256 &&
      review.mappingSha256 === admission.mappingSha256 && Array.isArray(review.supportingArtifactSha256) &&
      sameStrings(sorted(review.supportingArtifactSha256), sorted(admission.supportingArtifacts.map(p => p.sha256))),
    `The ${role} review does not admit these exact bytes, policy and lock.`);
  }
  const observation = structuredClone(extraction.observation) as SenateExtractedObservation;
  need(object(observation) && text(observation.raceId), 'Missing extracted observation.');
  const { row, selector, denominator } = lockedTarget(lock, observation.raceId);
  const mapping = store.readJson<{ schema: string; races: Record<string, unknown>[] }>(admission.mapping);
  need(mapping.schema === 'LockedSenate35PreResultMappingV1' && Array.isArray(mapping.races), 'Invalid raw owner mapping.');
  const raw = mapping.races.find(r => r.race_id === row.raceId);
  need(raw && raw.state === row.state && raw.seat_id === row.seatId && raw.event_kind === row.eventKind &&
    raw.scheduled_general_event_date === row.generalDate && raw.scheduled_runoff_event_date === row.runoffDate &&
    raw.first_candidate_id === row.firstCandidateId && raw.second_candidate_id === row.secondCandidateId &&
    raw.operator_family === row.operatorFamily, 'Locked policy row differs from the authenticated owner mapping.');
  need(object(extraction.sourceContest) && (extraction.sourceContest.id === null || text(extraction.sourceContest.id)) &&
    (extraction.sourceContest.roundId === null || text(extraction.sourceContest.roundId)) &&
    text(extraction.sourceContest.locator) && text(extraction.sourceContest.roundLabel), 'Actual source contest and round need reviewed locations.');
  const observedRound = extraction.sourceContest;
  need(observedRound.stage === observation.stage && observedRound.eventDate === observation.eventDate &&
    observedRound.officialObservedFinalRound === (observation.officialObservedFinalRound ?? null) &&
    observedRound.runoffRequired === (observation.runoffRequired ?? null) &&
    JSON.stringify(observedRound.continuingCandidateIds) === JSON.stringify(observation.continuingCandidateIds ?? null) &&
    JSON.stringify(observedRound.runoffCandidateIds) === JSON.stringify(observation.runoffCandidateIds ?? null),
  'Actual report round mapping contradicts the observation supplied to the policy operator.');
  need(Array.isArray(extraction.candidateCrosswalk) && extraction.candidateCrosswalk.every(c => text(c.candidateKey) &&
    (c.sourceCandidateId === null || text(c.sourceCandidateId)) && text(c.sourceName) && text(c.locator)) &&
    new Set(extraction.candidateCrosswalk.map(c => c.candidateKey)).size === extraction.candidateCrosswalk.length &&
    [row.firstCandidateId,row.secondCandidateId].every(id => extraction.candidateCrosswalk.some(c => c.candidateKey === id)),
  'Both fixed candidates require explicit source identity locations; absent feed IDs must remain null.');
  const sourceIds = extraction.candidateCrosswalk.flatMap(c => c.sourceCandidateId === null ? [] : [c.sourceCandidateId]);
  need(new Set(sourceIds).size === sourceIds.length, 'Distinct native candidates cannot share one source candidate ID.');
  need(Array.isArray(extraction.fieldLocations) && extraction.fieldLocations.every(l => text(l.field) && text(l.locator)) &&
    ['contest','round','finality'].every(field => extraction.fieldLocations.some(l => l.field === field)),
  'Contest, round and finality extraction locations are required.');
  need(['official','replacementOrIdentityChange','relevantRuleChange','certified','contestFinal','pendingRecount','pendingResultDispute']
    .every(key => typeof observation[key as keyof SenateExtractedObservation] === 'boolean'), 'Explicit source/finality declarations are required.');
  need(!observation.certified || (at(`${observation.eventDate}T00:00:00Z`) &&
    Date.parse(extraction.source.publishedAt) >= Date.parse(`${observation.eventDate}T00:00:00Z`)),
  'A certified result report cannot precede its declared election date.');
  need(object(observation.countsByNativeCandidateId) && Object.values(observation.countsByNativeCandidateId)
    .every(value => Number.isSafeInteger(value) && value >= 0), 'Extracted counts must be nonnegative safe integers.');
  for (const [field,id] of [['firstCandidateCount',row.firstCandidateId],['secondCandidateCount',row.secondCandidateId]]) {
    if (Object.hasOwn(observation.countsByNativeCandidateId,id))
      need(extraction.fieldLocations.some(l => l.field === field), 'Every extracted pair count needs a reviewed source location.');
  }
  need(Object.keys(observation.countsByNativeCandidateId).every(id => extraction.candidateCrosswalk.some(c => c.candidateKey === id)),
    'Every extracted count needs an explicit source candidate crosswalk.');
  if (observation.officialOfficeWinnerId != null)
    need(text(observation.officialOfficeWinnerId) && extraction.candidateCrosswalk.some(c => c.candidateKey === observation.officialOfficeWinnerId) &&
      extraction.fieldLocations.some(l => l.field === 'officialOfficeWinner'), 'Office-winner diagnostic requires reviewed identity and report location.');
  const pairTotal = (observation.countsByNativeCandidateId[row.firstCandidateId] ?? 0) + (observation.countsByNativeCandidateId[row.secondCandidateId] ?? 0);
  need(Number.isSafeInteger(pairTotal), 'Pair denominator exceeds exact integer arithmetic.');
  const projection = evaluateSenatePairResult(row, { ...observation,
    sourcePin: admission.source, sourceAdmissionReviewPin: admission.reviews.source,
    candidateCrosswalkReviewPin: admission.reviews.candidateIdentity, eventRoundReviewPin: admission.reviews.eventRound,
    candidateCrosswalkReviewed: true, eventRoundReviewed: true });
  const previous = prior.records.find(r => r.raceId === row.raceId), recordId = previous?.id ?? `senate-result:${row.raceId}`;
  need(!prior.history.some(h => h.record.id === recordId && h.record.revision === request.recordRevision),
    'Record revision already exists in the preserved history.');
  const status: EvaluationResultStatus = projection.marginFinalScoreEligible ? 'certified'
    : projection.status === 'preliminary_descriptive_only' ? 'provisional' : 'unresolved';
  const candidateId = projection.firstRelativeLead === null ? null
    : projection.firstRelativeLead ? row.firstCandidateId : row.secondCandidateId;
  const record: VerifiedElectionNightOutcome = {
    schema: 'ElectionNightOutcomeRecordV1', id: recordId, revision: request.recordRevision, raceId: row.raceId,
    source: { sha256: admission.source.sha256, url: extraction.source.url,
      publishedAt: extraction.source.publishedAt, retrievedAt: extraction.source.retrievedAt },
    winner: { ...selector, status: projection.marginFinalScoreEligible && !projection.pairBinaryFinalScoreEligible ? 'unresolved' : status,
      estimand: 'conditional_principal_pair_winner', outcomeId: candidateId, kind: candidateId ? 'candidate' : null,
      candidateId, reason: projection.reason, denominator },
    margin: { ...selector, status, estimand: 'named_pair_margin', firstCandidateId: row.firstCandidateId,
      secondCandidateId: row.secondCandidateId, denominator, units: 'percentage_points', orientation: 'first_minus_second',
      marginPp: projection.marginPp, reason: projection.reason },
  };
  const projectionPin = store.putJson(projection), recordPin = store.putJson(record);
  const revision: ElectionNightResultRevisionManifest = {
    schema: 'ElectionNightResultRevisionV1', id: request.revisionId, ledgerId: prior.manifest.id,
    evaluationLockSha256: lock.lockSha256, predecessorSha256: prior.chosenHeadSha256,
    supersedesRecordSha256: previous ? prior.provenanceByRecordId[previous.id].recordSha256 : null,
    record: recordPin, source: admission.source,
    evidence: uniquePins([request.extraction,request.trustedAdmission,request.trustedPublication,
      ...Object.values(admission.reviews),...admission.supportingArtifacts,admission.mapping,projectionPin]),
  };
  return Object.freeze({ schema: 'PreparedSenateResultRevisionV1', evaluationLockSha256: lock.lockSha256,
    priorLedgerSha256: prior.ledgerSha256, priorHeadSha256: prior.chosenHeadSha256,
    record: recordPin, revision: store.putJson(revision), operatorProjection: projectionPin,
    resultLedgerAdmissionIssued: false, publicHeadChanged: false });
}
