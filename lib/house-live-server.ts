/** Server-only loader for an operator-reviewed staged 2026 House release. */
import { readFile, realpath, lstat, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { DistrictGeography } from "./district-geography";
import { readPinnedHouseLiveJson, verifyPinnedHouseLiveArtifact,
  type HouseLiveArtifactPin } from "./house-live-artifacts.ts";
import { parseHouseLiveCatalog } from "./house-live-catalog.ts";
import { parseHouseLiveResult, houseLiveConfigurationKey, houseLivePin } from "./house-live-result.ts";
import { houseLiveCatalogView, houseLiveView } from "./house-live-view.ts";
import { parseHouseLivePublication, type HouseLivePublication } from "./house-live-publication.ts";
import { houseLiveRunId } from "./house-live-navigation.ts";
import { parseHouseLivePublicationQualification } from "./house-live-native-qualification.ts";
import { parseHouseLivePollingCollection, projectHouseLivePollingRun, type HouseLivePollingRun } from "./house-live-polling.ts";

export interface HouseLiveSiteManifestV1 {
  schema: "HouseLiveSiteManifestV1";
  status: "available";
  cycle: 2026;
  catalog: HouseLiveArtifactPin;
  review: HouseLiveArtifactPin;
  sourcePolicy: HouseLiveArtifactPin;
  qualification: HouseLiveArtifactPin;
  pollingDisclosure?: HouseLiveArtifactPin;
  geography: {
    artifact: HouseLiveArtifactPin;
    sourceReceipt: HouseLiveArtifactPin;
    sourceName: string;
    sourceUrl: string;
    sourceNote?: string;
  };
  results: { id: string; artifact: HouseLiveArtifactPin }[];
}

export interface HouseLiveSiteReviewV1 {
  schema: "HouseLiveSiteReviewV1";
  decision: "publish";
  /** Common winner encoding; the offline importer also authenticates combined-source lineage when present. */
  joint_format: "house_live_npz_chain_major_v1";
  seat_quantile_convention: "inverse_cdf" | "linear";
  catalog_sha256: string;
  source_policy_sha256: string;
  qualification_sha256: string;
  geography_sha256: string;
  geography_source_receipt_sha256: string;
  results: { id: string; result_sha256: string; joint_draws_sha256: string }[];
  checks: {
    inputsVerified: true;
    calculationsReviewed: true;
    jointRowsVerified: true;
    crossModelRosterVerified: true;
    geographyVerified: true;
  };
  limitations: string[];
}

export class HouseLiveSelectionError extends Error {
  constructor(message: string) { super(message); this.name = "HouseLiveSelectionError"; }
}

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0 && value.length <= 4000 &&
  value === value.trim() && !/[\u0000-\u001f\u007f]/.test(value);
const hash = (value: unknown): value is string =>
  typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const exactKeys = (value: Record<string, unknown>, keys: readonly string[]) =>
  Object.keys(value).length === keys.length && keys.every(key => key in value);
const pin = (value: unknown): value is HouseLiveArtifactPin =>
  object(value) && exactKeys(value, ["path", "sha256"]) && hash(value.sha256) &&
  typeof value.path === "string" && /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(value.path) &&
  value.path.split("/").every(part => !!part && part !== "." && part !== "..");

function parseManifest(value: unknown): HouseLiveSiteManifestV1 {
  if (!object(value) || !exactKeys(value, ["schema", "status", "cycle", "catalog", "review",
    "sourcePolicy", "qualification", "geography", "results", ...(value.pollingDisclosure === undefined ? [] : ["pollingDisclosure"])]) ||
    value.schema !== "HouseLiveSiteManifestV1" || value.status !== "available" || value.cycle !== 2026 ||
    !pin(value.catalog) || !pin(value.review) || !pin(value.sourcePolicy) || !pin(value.qualification) ||
    !(value.pollingDisclosure === undefined || pin(value.pollingDisclosure)) ||
    !object(value.geography) ||
    !exactKeys(value.geography, ["artifact", "sourceReceipt", "sourceName", "sourceUrl", ...(value.geography.sourceNote === undefined ? [] : ["sourceNote"])]) ||
    !pin(value.geography.artifact) || !pin(value.geography.sourceReceipt) ||
    !text(value.geography.sourceName) || !text(value.geography.sourceUrl) ||
    !/^https?:\/\//i.test(value.geography.sourceUrl) ||
    !(value.geography.sourceNote === undefined || text(value.geography.sourceNote)) ||
    !Array.isArray(value.results) || value.results.length !== 16) throw Error("Invalid live House site manifest.");
  const seen = new Set<string>();
  for (const row of value.results) {
    if (!object(row) || !exactKeys(row, ["id", "artifact"]) || !houseLiveRunId(row.id) ||
      seen.has(row.id) || !pin(row.artifact)) throw Error("Invalid live House result references.");
    seen.add(row.id);
  }
  return value as unknown as HouseLiveSiteManifestV1;
}

export function parseHouseLiveSiteReview(value: unknown): HouseLiveSiteReviewV1 {
  if (!object(value) || !exactKeys(value, ["schema", "decision", "catalog_sha256",
    "source_policy_sha256", "qualification_sha256", "geography_sha256",
    "geography_source_receipt_sha256", "results", "checks", "limitations", "joint_format", "seat_quantile_convention"]) ||
    value.schema !== "HouseLiveSiteReviewV1" || value.decision !== "publish" ||
    value.joint_format !== "house_live_npz_chain_major_v1" ||
    !["inverse_cdf", "linear"].includes(String(value.seat_quantile_convention)) ||
    ![value.catalog_sha256, value.source_policy_sha256, value.qualification_sha256,
      value.geography_sha256, value.geography_source_receipt_sha256].every(hash) ||
    !object(value.checks) || !exactKeys(value.checks, ["inputsVerified", "calculationsReviewed",
      "jointRowsVerified", "crossModelRosterVerified", "geographyVerified"]) ||
    Object.values(value.checks).some(check => check !== true) ||
    !Array.isArray(value.limitations) || !value.limitations.every(text) ||
    !Array.isArray(value.results) || value.results.length !== 16)
    throw Error("Invalid or unpublished live House site review.");
  const seen = new Set<string>();
  for (const row of value.results) {
    if (!object(row) || !exactKeys(row, ["id", "result_sha256", "joint_draws_sha256"]) ||
      !houseLiveRunId(row.id) || seen.has(row.id) || !hash(row.result_sha256) ||
      !hash(row.joint_draws_sha256)) throw Error("Invalid live House site review results.");
    seen.add(row.id);
  }
  return value as unknown as HouseLiveSiteReviewV1;
}

async function readFixedManifest(directory: string): Promise<HouseLiveSiteManifestV1> {
  const root = await realpath(directory);
  const path = await realpath(join(root, "publication.json"));
  if (dirname(path) !== root || !(await stat(path)).isFile())
    throw Error("Live House publication manifest must be a file inside its publication directory.");
  const bytes = await readFile(path);
  const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  return parseManifest(value);
}

function bindRelease(manifest: HouseLiveSiteManifestV1, review: HouseLiveSiteReviewV1,
  catalog: NonNullable<ReturnType<typeof parseHouseLiveCatalog>>): void {
  if (review.catalog_sha256 !== manifest.catalog.sha256 ||
    review.source_policy_sha256 !== manifest.sourcePolicy.sha256 ||
    review.qualification_sha256 !== manifest.qualification.sha256 ||
    review.geography_sha256 !== manifest.geography.artifact.sha256 ||
    review.geography_source_receipt_sha256 !== manifest.geography.sourceReceipt.sha256 ||
    catalog.source_policy.sha256 !== manifest.sourcePolicy.sha256 ||
    catalog.qualification.sha256 !== manifest.qualification.sha256)
    throw Error("Live House site review, catalog, and staged artifact fingerprints disagree.");
  const staged = new Map(manifest.results.map(row => [row.id, row.artifact]));
  const reviewed = new Map(review.results.map(row => [row.id, row]));
  if (staged.size !== 16 || reviewed.size !== 16 || catalog.entries.length !== 16)
    throw Error("Live House release must bind all 16 saved results.");
  for (const entry of catalog.entries) {
    const artifact = staged.get(entry.id), decision = reviewed.get(entry.id);
    if (!artifact || !decision || artifact.sha256 !== entry.result.sha256 ||
      decision.result_sha256 !== artifact.sha256)
      throw Error(`Live House release does not bind catalog result ${entry.id}.`);
  }
  if ([...staged.keys()].some(id => !catalog.entries.some(entry => entry.id === id)) ||
    [...reviewed.keys()].some(id => !catalog.entries.some(entry => entry.id === id)))
    throw Error("Live House release contains an unlisted result.");
}

async function loadStaged(directory: string, runId?: string | null): Promise<{
  publication: HouseLivePublication; manifest: HouseLiveSiteManifestV1;
}> {
  if (runId != null && !houseLiveRunId(runId))
    throw new HouseLiveSelectionError("Invalid current House result selection.");
  const manifest = await readFixedManifest(directory);
  const catalog = parseHouseLiveCatalog(await readPinnedHouseLiveJson(directory, manifest.catalog));
  if (!catalog) throw Error("Invalid native live House catalog.");
  const review = parseHouseLiveSiteReview(await readPinnedHouseLiveJson(directory, manifest.review));
  bindRelease(manifest, review, catalog);
  const selectedId = runId ?? catalog.default_id;
  const entry = catalog.entries.find(row => row.id === selectedId);
  if (!entry) throw new HouseLiveSelectionError("This current House result is not in the published collection.");
  const staged = manifest.results.find(row => row.id === selectedId)!;
  const reviewResult = review.results.find(row => row.id === selectedId)!;
  // Runtime checks authenticate staged bytes. The checked-in operator review
  // records prior importer checks of raw joint rows and all-run roster identity;
  // neither this loader nor matching booleans re-establish those findings.
  const sourcePolicy = manifest.pollingDisclosure ? await readPinnedHouseLiveJson(directory, manifest.sourcePolicy) : null;
  if (!manifest.pollingDisclosure) await verifyPinnedHouseLiveArtifact(directory, manifest.sourcePolicy);
  const qualification = parseHouseLivePublicationQualification(await readPinnedHouseLiveJson(directory, manifest.qualification));
  if (!qualification || qualification.seat_quantile_convention !== review.seat_quantile_convention)
    throw Error("Native House qualification is unreviewed or disagrees with the reviewed export convention.");
  await verifyPinnedHouseLiveArtifact(directory, qualification.numerical_review);
  await verifyPinnedHouseLiveArtifact(directory, manifest.geography.sourceReceipt);
  const geojson = await readPinnedHouseLiveJson(directory, manifest.geography.artifact);
  const native = parseHouseLiveResult(await readPinnedHouseLiveJson(directory, staged.artifact));
  if (!native || native.id !== entry.id ||
    houseLiveConfigurationKey(native.configuration) !== houseLiveConfigurationKey(entry.configuration) ||
    native.geography.sha256 !== manifest.geography.artifact.sha256 ||
    native.geography.source_receipt.sha256 !== manifest.geography.sourceReceipt.sha256 ||
    native.joint_draws.sha256 !== reviewResult.joint_draws_sha256 ||
    native.qualification.publication_status !== "reviewed" ||
    qualification.limitations.some(note => !native.qualification.limitations.includes(note)))
    throw Error("Selected native live House result does not match its reviewed release.");
  let polling: HouseLivePollingRun | null = null;
  if (manifest.pollingDisclosure) {
    const disclosure = parseHouseLivePollingCollection(await readPinnedHouseLiveJson(directory, manifest.pollingDisclosure));
    if (!disclosure || disclosure.bindings.catalogSha256 !== manifest.catalog.sha256 ||
      disclosure.bindings.sourcePolicySha256 !== manifest.sourcePolicy.sha256 ||
      !object(sourcePolicy) || sourcePolicy.schema !== "HouseLiveSourcePolicyV1" ||
      disclosure.cutoffs.forecastAt !== sourcePolicy.forecast_cutoff ||
      disclosure.cutoffs.pollingAt !== sourcePolicy.polling_cutoff ||
      Date.parse(disclosure.cutoffs.forecastAt) !== Date.parse(native.forecast_cutoff) ||
      disclosure.runs.some(run => !catalog.entries.some(row => row.id === run.id &&
        row.result.sha256 === run.resultSha256 && row.configuration.polling === run.polling)))
      throw Error("Live House polling disclosure does not bind its catalog, source policy, or native cutoffs.");
    const admissions: { artifact: HouseLiveArtifactPin; value: Record<string, unknown> }[] = [];
    for (const artifact of native.qualification.evidence) {
      const value = await readPinnedHouseLiveJson(directory, artifact);
      if (object(value) && value.schema === "HouseLiveAggregateInputAdmissionV1") admissions.push({ artifact, value });
    }
    const admission = admissions[0];
    if (admissions.length !== 1 || admission.artifact.sha256 !== disclosure.bindings.admissionSha256 ||
      admission.value.status !== "admitted_for_current_aggregate_comparison" ||
      typeof admission.value.forecast_cutoff !== "string" ||
      Date.parse(admission.value.forecast_cutoff) !== Date.parse(disclosure.cutoffs.forecastAt) ||
      !Array.isArray(admission.value.input_pins) || !admission.value.input_pins.every(houseLivePin) ||
      ![disclosure.bindings.sourcePolicySha256, disclosure.bindings.selectionSha256, disclosure.bindings.ledgerSha256]
        .every(sha256 => (admission.value.input_pins as { sha256: string }[]).filter(row => row.sha256 === sha256).length === 1))
      throw Error("Live House polling disclosure does not bind its authenticated native input admission.");
    polling = projectHouseLivePollingRun(disclosure, native.id, staged.artifact.sha256);
    const observations = new Map(native.races.flatMap(race => race.poll_source_ids.map(id => [id, race.race_id] as const)));
    if (!polling || observations.size !== native.races.reduce((sum, race) => sum + race.poll_source_ids.length, 0) ||
      polling.modelUsesSelectedQuestions !== (native.configuration.polling === "standard") ||
      polling.usedQuestions.length !== observations.size || polling.usedQuestions.some(question =>
        observations.get(question.observationId) !== question.raceId))
      throw Error("Live House polling disclosure questions do not match the selected native result.");
  }
  const geography: DistrictGeography = {
    metadata: { cycle: 2026, mapVintage: native.map_vintage, admission: "verified",
      sourceSha256: manifest.geography.artifact.sha256,
      sourceName: manifest.geography.sourceName, sourceUrl: manifest.geography.sourceUrl,
      ...(manifest.geography.sourceNote ? { sourceNote: manifest.geography.sourceNote } : {}) },
    // Browser parser checks all 435 identities and geometry before returning a safe DTO.
    geojson: geojson as DistrictGeography["geojson"],
  };
  const projected = { schema: "HouseLivePublicationV1", status: "available",
    catalog: houseLiveCatalogView(catalog), view: houseLiveView(native, entry), geography, polling,
    publicationReview: { limitations: [...review.limitations] } };
  const publication = parseHouseLivePublication(projected);
  if (!publication || !publication.geography)
    throw Error("Live House browser publication did not pass selected-result or geography checks.");
  return { publication, manifest };
}

/** Load one selected native result as a path-free browser DTO. */
export async function loadHouseLiveSitePublication(directory: string, runId?: string | null): Promise<HouseLivePublication> {
  return (await loadStaged(directory, runId)).publication;
}

/** Absence alone means unpublished; a staged but incomplete release is an error. */
export async function readHouseLiveAvailability(directory: string): Promise<{ available: boolean }> {
  try {
    await lstat(directory);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { available: false };
    throw error;
  }
  // A configured directory entry that points to a missing target is a broken
  // release, not absence. Canonicalize outside the absence-only catches.
  const root = await realpath(directory);
  try {
    await lstat(join(root, "publication.json"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { available: false };
    throw error;
  }
  const { manifest } = await loadStaged(directory);
  // The catalog advertises 16 choices. Check every staged result's bytes before
  // declaring the release ready, including choices other than the default.
  for (const row of manifest.results) await verifyPinnedHouseLiveArtifact(directory, row.artifact);
  return { available: true };
}
