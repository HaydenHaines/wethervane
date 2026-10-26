/**
 * Node-only immutable artifact storage and byte-authenticated evaluation inputs.
 * Callers provide trusted publication AND scientific-admission manifest pins.
 * Authentic bytes do not establish the scientific validity of an event mapping.
 * This is a local store, not a hostile multi-user filesystem service: its root
 * and parents must not be renamed concurrently by an untrusted process.
 */
import { createHash, randomBytes } from "node:crypto";
import { constants, closeSync, fsyncSync, linkSync, lstatSync, mkdirSync, openSync,
  readFileSync, realpathSync, unlinkSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import type { ElectionNightForecast } from "./election-night-forecast.ts";
import type { ElectionNightEvaluationProtocol, VerifiedElectionNightOutcome } from "./election-night-evaluation.ts";
import type { ElectionNightRubricContract, RubricForecastBinding } from "./election-night-rubric.ts";

export interface ElectionNightArtifactPin { sha256: string; byteLength: number; path: string }
export type DeepReadonly<T> = T extends object ? { readonly [P in keyof T]: DeepReadonly<T[P]> } : T;
export type ElectionNightLockArtifactRole = "modelSource" | "baselineSource" | "modelForecast" | "baselineForecast" |
  "modelProtocol" | "baselineProtocol" | "comparisonContract" | "cohort" | "rubric" | "resultPolicy";
export type ElectionNightLockReviewRole = "comparison" | "cohort" | "rubric" | "resultPolicy" | "projection";
export interface ElectionNightEvaluationLockManifest {
  schema: "ElectionNightEvaluationLockV1"; id: string; frozenAt: string;
  forecastBindings: { model: RubricForecastBinding; baseline: RubricForecastBinding };
  artifacts: Record<ElectionNightLockArtifactRole, ElectionNightArtifactPin>;
  reviews: Record<ElectionNightLockReviewRole, ElectionNightArtifactPin>;
  projections: {
    model: { sourceSha256: string; projectionSha256: string; reviewSha256: string };
    baseline: { sourceSha256: string; projectionSha256: string; reviewSha256: string };
  };
}
export interface ElectionNightArtifactPublicationManifest {
  schema: "ElectionNightArtifactPublicationV1"; id: string;
  /** Exactly pinned content allowed by this caller-trusted publication edition. */
  artifacts: ElectionNightArtifactPin[];
}
export interface ElectionNightArtifactAdmissionManifest {
  schema: "ElectionNightArtifactAdmissionV1"; id: string;
  scope: "evaluation_lock" | "result_ledger";
  subject: ElectionNightArtifactPin; evaluationLockSha256: string;
  /** Reviewed evidence, independently pinned by the trusted admission decision. */
  reviews: ElectionNightArtifactPin[];
}
export interface ElectionNightResultRevisionManifest {
  schema: "ElectionNightResultRevisionV1"; id: string; ledgerId: string; evaluationLockSha256: string;
  /** Global ledger predecessor. No timestamp-based ordering or head selection. */
  predecessorSha256: string | null;
  /** Previous immutable record bytes for this record ID; null only on first admission. */
  supersedesRecordSha256: string | null;
  record: ElectionNightArtifactPin; source: ElectionNightArtifactPin; evidence: ElectionNightArtifactPin[];
}
export interface ElectionNightResultLedgerManifest {
  schema: "ElectionNightResultLedgerV1"; id: string; evaluationLockSha256: string;
  /** Null selects the empty pre-result history; otherwise the final explicit revision. */
  chosenHeadSha256: string | null;
  /** Complete oldest-to-chosen-head chain, explicitly selected by the caller. */
  revisions: ElectionNightArtifactPin[];
}
export interface LoadedElectionNightEvaluationLock {
  lockSha256: string; publicationSha256: string; admissionSha256: string;
  manifest: DeepReadonly<ElectionNightEvaluationLockManifest>;
  modelForecast: DeepReadonly<ElectionNightForecast>; baselineForecast: DeepReadonly<ElectionNightForecast>;
  modelProtocol: DeepReadonly<ElectionNightEvaluationProtocol>; baselineProtocol: DeepReadonly<ElectionNightEvaluationProtocol>;
  comparisonContract: DeepReadonly<ElectionNightRubricContract>;
  cohort: unknown; rubric: unknown; resultPolicy: unknown;
}
export interface ElectionNightRecordProvenance {
  recordId: string; revision: string; recordSha256: string; sourceSha256: string; resultPolicySha256: string;
  revisionSha256: string; evidenceSha256: readonly string[];
}
export interface LoadedElectionNightResultLedger {
  lockSha256: string; ledgerSha256: string; publicationSha256: string; admissionSha256: string; chosenHeadSha256: string | null;
  manifest: DeepReadonly<ElectionNightResultLedgerManifest>;
  records: readonly VerifiedElectionNightOutcome[];
  provenanceByRecordId: Readonly<Record<string, ElectionNightRecordProvenance>>;
  /** Full immutable revisions/records, including all superseded results. */
  history: readonly { revisionSha256: string; manifest: DeepReadonly<ElectionNightResultRevisionManifest>;
    record: DeepReadonly<VerifiedElectionNightOutcome> }[];
}
export interface ElectionNightLockLoadRequest {
  lock: ElectionNightArtifactPin; trustedPublication: ElectionNightArtifactPin; trustedAdmission: ElectionNightArtifactPin;
}
export interface ElectionNightLedgerLoadRequest {
  evaluationLock: LoadedElectionNightEvaluationLock; ledger: ElectionNightArtifactPin;
  trustedPublication: ElectionNightArtifactPin; trustedAdmission: ElectionNightArtifactPin;
}

const hashPattern = /^[0-9a-f]{64}$/;
const text = (v: unknown): v is string => typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown): v is string => typeof v === "string" && hashPattern.test(v);
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
function need(value: unknown, message: string): asserts value { if (!value) throw Error(`Election-night artifacts: ${message}`); }
export function electionNightArtifactSha256(bytes: Uint8Array): string { return createHash("sha256").update(bytes).digest("hex"); }
function pinPath(sha256: string): string { return `blobs/sha256/${sha256}`; }
function validPin(v: unknown): v is ElectionNightArtifactPin {
  return object(v) && hash(v.sha256) && Number.isSafeInteger(v.byteLength) && (v.byteLength as number) >= 0 &&
    v.path === pinPath(v.sha256);
}
function samePin(a: ElectionNightArtifactPin, b: ElectionNightArtifactPin): boolean {
  return validPin(a) && validPin(b) && a.sha256 === b.sha256 && a.byteLength === b.byteLength && a.path === b.path;
}
function deepFreeze<T>(v: T): DeepReadonly<T> {
  if (v && typeof v === "object") { Object.values(v).forEach(deepFreeze); Object.freeze(v); }
  return v as DeepReadonly<T>;
}
function canonicalJson(v: unknown, seen = new Set<object>()): string {
  if (v === null || typeof v === "string" || typeof v === "boolean") return JSON.stringify(v);
  if (typeof v === "number") { need(Number.isFinite(v), "JSON numbers must be finite."); return JSON.stringify(v); }
  need(v && typeof v === "object" && (Array.isArray(v) || Object.getPrototypeOf(v) === Object.prototype || Object.getPrototypeOf(v) === null),
    "Only plain JSON objects/arrays and values can be stored as JSON.");
  need(!seen.has(v), "Cyclic JSON is not an artifact."); seen.add(v);
  let result: string;
  if (Array.isArray(v)) {
    need(Object.keys(v).length === v.length && Object.keys(v).every((key, i) => key === String(i)), "Sparse arrays or array properties are not JSON artifacts.");
    result = `[${v.map(x => canonicalJson(x, seen)).join(",")}]`;
  } else result = `{${Object.keys(v).sort().map(k => `${JSON.stringify(k)}:${canonicalJson((v as Record<string, unknown>)[k], seen)}`).join(",")}}`;
  seen.delete(v); return result;
}
const same = (a: unknown, b: unknown) => canonicalJson(a) === canonicalJson(b);
function pathError(error: unknown): string | undefined { return (error as NodeJS.ErrnoException)?.code; }

export class ElectionNightArtifactStore {
  readonly root: string;
  constructor(root: string) {
    need(text(root), "An existing local store directory is required.");
    const absolute = resolve(root), stat = lstatSync(absolute);
    need(stat.isDirectory() && !stat.isSymbolicLink() && realpathSync(absolute) === absolute,
      "Store root and ancestors must be real directories, not symlinks.");
    this.root = absolute;
    Object.freeze(this);
  }
  private localPath(relativePath: string, createParents = false): string {
    need(text(relativePath) && !isAbsolute(relativePath) && !relativePath.includes("\\") && !relativePath.includes("\0") &&
      relativePath.split("/").every(part => part !== "" && part !== "." && part !== ".."), "Unsafe relative artifact path.");
    const target = resolve(this.root, relativePath), rel = relative(this.root, target);
    need(rel !== "" && rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel), "Artifact path escapes the store.");
    const rootStat = lstatSync(this.root);
    need(rootStat.isDirectory() && !rootStat.isSymbolicLink() && realpathSync(this.root) === this.root, "Store root changed or became a symlink.");
    let parent = this.root;
    for (const part of relativePath.split("/").slice(0, -1)) {
      parent = join(parent, part);
      if (createParents) { try { mkdirSync(parent, { mode: 0o755 }); } catch (error) { if (pathError(error) !== "EEXIST") throw error; } }
      const stat = lstatSync(parent);
      need(stat.isDirectory() && !stat.isSymbolicLink() && realpathSync(parent) === parent, "Artifact parent is not a real in-store directory.");
    }
    return target;
  }
  /** Atomic create-if-absent. Existing corrupt bytes are an error and are never repaired or overwritten. */
  put(value: Uint8Array): ElectionNightArtifactPin {
    need(value instanceof Uint8Array, "Artifact bytes must be a Uint8Array.");
    const bytes = Buffer.from(value), sha256 = electionNightArtifactSha256(bytes);
    const pin = { sha256, byteLength: bytes.byteLength, path: pinPath(sha256) }, target = this.localPath(pin.path, true);
    try { lstatSync(target); need(this.read(pin).equals(bytes), "Existing content address has different bytes."); return deepFreeze(pin); } catch (error) { if (pathError(error) !== "ENOENT") throw error; }
    const temp = this.localPath(`blobs/sha256/.incoming-${randomBytes(16).toString("hex")}`);
    let descriptor: number | undefined;
    try {
      descriptor = openSync(temp, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o444);
      writeFileSync(descriptor, bytes); fsyncSync(descriptor); closeSync(descriptor); descriptor = undefined;
      try { linkSync(temp, target); } catch (error) { if (pathError(error) !== "EEXIST") throw error; }
      need(this.read(pin).equals(bytes), "Existing content address has different bytes.");
      const directory = openSync(join(this.root, "blobs", "sha256"), constants.O_RDONLY | constants.O_NOFOLLOW);
      try { fsyncSync(directory); } finally { closeSync(directory); }
      return deepFreeze(pin);
    } finally { if (descriptor !== undefined) closeSync(descriptor); try { unlinkSync(temp); } catch (error) { if (pathError(error) !== "ENOENT") throw error; } }
  }
  putJson(value: unknown): ElectionNightArtifactPin { return this.put(Buffer.from(canonicalJson(value), "utf8")); }
  read(pin: ElectionNightArtifactPin): Buffer {
    need(validPin(pin), "Invalid content-addressed artifact pin or unsafe relative path.");
    const path = this.localPath(pin.path), stat = lstatSync(path);
    need(stat.isFile() && !stat.isSymbolicLink(), "Artifact must be a regular file, never a symlink.");
    const descriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    let bytes: Buffer;
    try { bytes = readFileSync(descriptor); } finally { closeSync(descriptor); }
    need(bytes.byteLength === pin.byteLength && electionNightArtifactSha256(bytes) === pin.sha256,
      `Corrupt artifact bytes for ${pin.sha256}.`);
    return bytes;
  }
  readJson<T = unknown>(pin: ElectionNightArtifactPin): DeepReadonly<T> {
    const bytes = this.read(pin);
    // Buffer's permissive decoder must not turn malformed UTF-8 into different JSON text.
    const decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return deepFreeze(JSON.parse(decoded) as T);
  }
}

const loadedLocks = new WeakSet<object>();
const loadedLedgers = new WeakSet<object>();
export function assertLoadedEvaluationLock(value: unknown): asserts value is LoadedElectionNightEvaluationLock {
  need(object(value) && loadedLocks.has(value), "Evaluation lock did not come from the authenticated loader.");
}
export function assertLoadedResultLedger(value: unknown): asserts value is LoadedElectionNightResultLedger {
  need(object(value) && loadedLedgers.has(value), "Result ledger did not come from the authenticated loader.");
}
const artifactRoles: ElectionNightLockArtifactRole[] = ["modelSource", "baselineSource", "modelForecast", "baselineForecast",
  "modelProtocol", "baselineProtocol", "comparisonContract", "cohort", "rubric", "resultPolicy"];
const reviewRoles: ElectionNightLockReviewRole[] = ["comparison", "cohort", "rubric", "resultPolicy", "projection"];
function pins(values: unknown, label: string, nonempty = true): asserts values is ElectionNightArtifactPin[] {
  need(Array.isArray(values) && (!nonempty || values.length > 0) && values.every(validPin), `Invalid ${label} pins.`);
  need(new Set(values.map(p => p.sha256)).size === values.length, `Duplicate ${label} pin.`);
}
function rolePins(value: unknown, roles: string[], label: string): void {
  need(object(value) && same(Object.keys(value).sort(), [...roles].sort()) && roles.every(role => validPin(value[role])), `Invalid ${label} role bindings.`);
}
function published(index: Map<string, ElectionNightArtifactPin>, pin: ElectionNightArtifactPin): void {
  need(validPin(pin) && !!index.get(pin.sha256) && samePin(index.get(pin.sha256)!, pin), `Unpublished/untrusted artifact ${pin?.sha256 || "(missing)"}.`);
}
function authority(store: ElectionNightArtifactStore, subject: ElectionNightArtifactPin, publicationPin: ElectionNightArtifactPin,
  admissionPin: ElectionNightArtifactPin, scope: ElectionNightArtifactAdmissionManifest["scope"], lockSha256: string):
  { publication: Map<string, ElectionNightArtifactPin>; admission: DeepReadonly<ElectionNightArtifactAdmissionManifest> } {
  const publication = store.readJson<ElectionNightArtifactPublicationManifest>(publicationPin);
  need(publication?.schema === "ElectionNightArtifactPublicationV1" && text(publication.id), "Invalid trusted publication manifest.");
  pins(publication.artifacts, "publication");
  const index = new Map(publication.artifacts.map(p => [p.sha256, p]));
  published(index, subject); published(index, admissionPin);
  const admission = store.readJson<ElectionNightArtifactAdmissionManifest>(admissionPin);
  need(admission?.schema === "ElectionNightArtifactAdmissionV1" && text(admission.id) && admission.scope === scope &&
    samePin(admission.subject, subject) && admission.evaluationLockSha256 === lockSha256, "Trusted admission does not authorize this exact subject/lock/scope.");
  pins(admission.reviews, "admission review");
  for (const pin of admission.reviews) { published(index, pin); store.read(pin); }
  return { publication: index, admission };
}
function readPublished<T>(store: ElectionNightArtifactStore, index: Map<string, ElectionNightArtifactPin>, pin: ElectionNightArtifactPin): T {
  published(index, pin); return store.readJson<T>(pin) as T;
}
function validBinding(v: RubricForecastBinding): boolean {
  return !!v && ["senate", "house"].includes(v.chamber) && text(v.runId) && hash(v.resultSha256) && hash(v.nativeForecastSha256) &&
    text(v.targetId) && text(v.lawId) && Number.isFinite(Date.parse(v.issuedAt)) && Number.isFinite(Date.parse(v.cutoffAt)) &&
    Date.parse(v.cutoffAt) <= Date.parse(v.issuedAt) && Number.isSafeInteger(v.horizonCalendarDays) && v.horizonCalendarDays >= 0;
}
/**
 * Trust roots are explicit caller arguments, never taken from the subject's claims.
 * comparisonContract.sha256 is an admitted comparison identity, NOT a self-hash of
 * its JSON; artifacts.comparisonContract separately pins those exact serialized bytes.
 */
export function loadElectionNightEvaluationLock(store: ElectionNightArtifactStore, request: ElectionNightLockLoadRequest): LoadedElectionNightEvaluationLock {
  need(request && validPin(request.lock), "An exact lock pin is required.");
  const { publication, admission } = authority(store, request.lock, request.trustedPublication, request.trustedAdmission, "evaluation_lock", request.lock.sha256);
  const manifest = readPublished<ElectionNightEvaluationLockManifest>(store, publication, request.lock);
  need(manifest?.schema === "ElectionNightEvaluationLockV1" && text(manifest.id) && Number.isFinite(Date.parse(manifest.frozenAt)), "Invalid evaluation lock manifest.");
  rolePins(manifest.artifacts, artifactRoles, "lock artifact"); rolePins(manifest.reviews, reviewRoles, "lock review");
  for (const pin of [...Object.values(manifest.artifacts), ...Object.values(manifest.reviews)]) { published(publication, pin); store.read(pin); }
  need(same([...new Set(Object.values(manifest.reviews).map(p => p.sha256))].sort(), admission.reviews.map(p => p.sha256).sort()),
    "Lock review bytes differ from the caller-trusted scientific admission.");
  const contract = readPublished<ElectionNightRubricContract>(store, publication, manifest.artifacts.comparisonContract);
  need(contract?.schema === "ElectionNightRubricComparisonV1" && hash(contract.sha256) &&
    validBinding(manifest.forecastBindings?.model) && validBinding(manifest.forecastBindings?.baseline) &&
    same(contract.model, manifest.forecastBindings.model) && same(contract.baseline, manifest.forecastBindings.baseline), "Comparison and lock forecast bindings differ.");
  need(contract.cohortDefinition?.sha256 === manifest.artifacts.cohort.sha256 && contract.resultPolicy?.sha256 === manifest.artifacts.resultPolicy.sha256 &&
    contract.rubricDecisionSha256 === manifest.artifacts.rubric.sha256 && contract.comparisonReview?.sha256 === manifest.reviews.comparison.sha256,
    "Comparison contract does not bind the exact cohort, result policy, rubric and review bytes.");
  const forecasts = {} as Record<"model" | "baseline", ElectionNightForecast>;
  const protocols = {} as Record<"model" | "baseline", ElectionNightEvaluationProtocol>;
  for (const side of ["model", "baseline"] as const) {
    const binding = manifest.forecastBindings[side], source = manifest.artifacts[side === "model" ? "modelSource" : "baselineSource"],
      projectionPin = manifest.artifacts[side === "model" ? "modelForecast" : "baselineForecast"], projection = manifest.projections?.[side];
    need(binding.resultSha256 === source.sha256 && projection?.sourceSha256 === source.sha256 &&
      projection.projectionSha256 === projectionPin.sha256 && projection.reviewSha256 === manifest.reviews.projection.sha256,
      "Original forecast source, normalized projection and projection review pins differ.");
    const f = readPublished<ElectionNightForecast>(store, publication, projectionPin);
    need(f?.schema === "ElectionNightForecastV1" && f.editionStatus === "dated_snapshot" && f.chamber === binding.chamber &&
      f.run?.id === binding.runId && f.run.artifactSha256 === source.sha256 && f.run.issuedAt === binding.issuedAt && f.run.cutoffAt === binding.cutoffAt &&
      f.run.targetId === binding.targetId && f.run.lawId === binding.lawId && Array.isArray(f.races) && f.races.length > 0 &&
      f.races.every(r => r && text(r.id)) && new Set(f.races.map(r => r.id)).size === f.races.length,
      "Scorer projection does not match the pinned original forecast edition.");
    need(binding.nativeForecastSha256 === (f.chamber === "senate" ? f.sourceHashes?.forecastSha256 : source.sha256), "Native forecast provenance differs from the declared binding.");
    const protocol = readPublished<ElectionNightEvaluationProtocol>(store, publication, manifest.artifacts[side === "model" ? "modelProtocol" : "baselineProtocol"]);
    need(protocol?.schema === "ElectionNightEvaluationProtocolV1" && text(protocol.id) && same(protocol.forecast,
      { chamber: binding.chamber, runId: binding.runId, artifactSha256: source.sha256, issuedAt: binding.issuedAt, cutoffAt: binding.cutoffAt }) &&
      Array.isArray(protocol.races) && protocol.races.length === f.races.length && protocol.races.every((r, i) => r && r.raceId === f.races[i].id) &&
      hash(protocol.reviewedDefinition?.sha256) && Object.values(manifest.reviews).some(p => p.sha256 === protocol.reviewedDefinition.sha256),
      "Evaluation protocol edition, ordered race cohort or reviewed definition differs from the lock.");
    forecasts[side] = f; protocols[side] = protocol;
  }
  const displayIds = forecasts.model.races.map(r => r.id);
  need(same(displayIds, forecasts.baseline.races.map(r => r.id)) && Array.isArray(contract.winnerCohort) &&
    contract.winnerCohort.length > 0 && contract.winnerCohort.every(t => t && text(t.raceId)) &&
    new Set(contract.winnerCohort.map(t => t.raceId)).size === contract.winnerCohort.length,
    "Locked model/baseline display cohorts or declared scoring cohort differ.");
  const scoredIds = contract.winnerCohort.map(t => t.raceId), scored = new Set(scoredIds);
  need(same(scoredIds, displayIds.filter(id => scored.has(id))), "Scoring cohort must be an explicit ordered subset of the preserved display races.");
  const excluded = displayIds.filter(id => !scored.has(id));
  if (contract.displayRaceIds !== undefined || contract.ungradedDisplayRaceIds !== undefined || excluded.length > 0)
    need(same(contract.displayRaceIds, displayIds) && same(contract.ungradedDisplayRaceIds, excluded),
      "Every preserved display race outside the scoring cohort must be explicitly predeclared as ungraded.");
  const result: LoadedElectionNightEvaluationLock = {
    lockSha256: request.lock.sha256, publicationSha256: request.trustedPublication.sha256, admissionSha256: request.trustedAdmission.sha256,
    manifest: deepFreeze(manifest), modelForecast: forecasts.model, baselineForecast: forecasts.baseline,
    modelProtocol: protocols.model, baselineProtocol: protocols.baseline, comparisonContract: contract,
    cohort: readPublished(store, publication, manifest.artifacts.cohort), rubric: readPublished(store, publication, manifest.artifacts.rubric),
    resultPolicy: readPublished(store, publication, manifest.artifacts.resultPolicy),
  };
  deepFreeze(result); loadedLocks.add(result); return result;
}

/** Complete pinned ledger history is verified before any chosen-head records are returned. */
export function loadElectionNightResultLedger(store: ElectionNightArtifactStore, request: ElectionNightLedgerLoadRequest): LoadedElectionNightResultLedger {
  assertLoadedEvaluationLock(request?.evaluationLock);
  need(validPin(request.ledger), "An exact result ledger pin is required.");
  const lock = request.evaluationLock, { publication, admission } = authority(store, request.ledger, request.trustedPublication,
    request.trustedAdmission, "result_ledger", lock.lockSha256);
  const manifest = readPublished<ElectionNightResultLedgerManifest>(store, publication, request.ledger);
  need(manifest?.schema === "ElectionNightResultLedgerV1" && text(manifest.id) && manifest.evaluationLockSha256 === lock.lockSha256 &&
    (manifest.chosenHeadSha256 === null || hash(manifest.chosenHeadSha256)),
  "Ledger does not bind the authenticated evaluation lock and explicit chosen head.");
  pins(manifest.revisions, "result revision", false);
  need(manifest.revisions.length === 0 ? manifest.chosenHeadSha256 === null :
    manifest.revisions[manifest.revisions.length - 1].sha256 === manifest.chosenHeadSha256,
  "Chosen head is not the end of the complete explicit revision chain.");
  need(admission.reviews.some(p => samePin(p, lock.manifest.reviews.resultPolicy)), "Ledger admission must retain the locked result-policy review.");
  const current = new Map<string, { record: VerifiedElectionNightOutcome; pin: ElectionNightArtifactPin }>(), raceRecordId = new Map<string, string>();
  const seenIds = new Set<string>(), seenRecords = new Set<string>(), seenRevisions = new Set<string>();
  const provenanceByRecordId: Record<string, ElectionNightRecordProvenance> = Object.create(null);
  const history: LoadedElectionNightResultLedger["history"][number][] = [];
  let predecessor: string | null = null;
  const cohortIds = new Set(lock.modelForecast.races.map(r => r.id));
  for (const pin of manifest.revisions) {
    const revision = readPublished<ElectionNightResultRevisionManifest>(store, publication, pin);
    need(revision?.schema === "ElectionNightResultRevisionV1" && text(revision.id) && !seenIds.has(revision.id) &&
      revision.ledgerId === manifest.id && revision.evaluationLockSha256 === lock.lockSha256 && revision.predecessorSha256 === predecessor,
      "Duplicate, branched, missing or out-of-order result revision predecessor.");
    seenIds.add(revision.id);
    need(validPin(revision.record) && validPin(revision.source), "Invalid result record/source pins.");
    pins(revision.evidence, "result evidence");
    for (const evidence of [revision.source, ...revision.evidence]) { published(publication, evidence); store.read(evidence); }
    const record = readPublished<VerifiedElectionNightOutcome>(store, publication, revision.record);
    need(record?.schema === "ElectionNightOutcomeRecordV1" && text(record.id) && text(record.revision) && cohortIds.has(record.raceId) &&
      record.source?.sha256 === revision.source.sha256 && /^https?:\/\//.test(record.source.url) &&
      Number.isFinite(Date.parse(record.source.publishedAt)) && Number.isFinite(Date.parse(record.source.retrievedAt)) &&
      Date.parse(record.source.publishedAt) <= Date.parse(record.source.retrievedAt) &&
      (record.winner === null || object(record.winner)) && (record.margin === null || object(record.margin)),
      "Invalid result identity, frozen cohort or actual source-byte provenance.");
    const prior = current.get(record.id), recordRevision = canonicalJson([record.id, record.revision]);
    need(!seenRecords.has(revision.record.sha256) && !seenRevisions.has(recordRevision) &&
      revision.supersedesRecordSha256 === (prior?.pin.sha256 ?? null) && (!prior || prior.record.raceId === record.raceId) &&
      (!raceRecordId.has(record.raceId) || raceRecordId.get(record.raceId) === record.id),
      "Duplicate/branched record revision, changed race identity, or missing record predecessor.");
    seenRecords.add(revision.record.sha256); seenRevisions.add(recordRevision); raceRecordId.set(record.raceId, record.id);
    current.set(record.id, { record, pin: revision.record });
    provenanceByRecordId[record.id] = { recordId: record.id, revision: record.revision, recordSha256: revision.record.sha256,
      sourceSha256: revision.source.sha256, resultPolicySha256: lock.manifest.artifacts.resultPolicy.sha256,
      revisionSha256: pin.sha256, evidenceSha256: revision.evidence.map(p => p.sha256) };
    history.push({ revisionSha256: pin.sha256, manifest: revision, record }); predecessor = pin.sha256;
  }
  // Missing races stay absent: downstream evaluation counts unresolved locked targets.
  const records = lock.modelForecast.races.flatMap(r => { const id = raceRecordId.get(r.id); return id ? [current.get(id)!.record] : []; });
  const result: LoadedElectionNightResultLedger = { lockSha256: lock.lockSha256, ledgerSha256: request.ledger.sha256,
    publicationSha256: request.trustedPublication.sha256, admissionSha256: request.trustedAdmission.sha256,
    chosenHeadSha256: manifest.chosenHeadSha256, manifest: deepFreeze(manifest), records, provenanceByRecordId, history };
  deepFreeze(result); loadedLedgers.add(result); return result;
}
