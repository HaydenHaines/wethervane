import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import {
  canonicalStudySelections, studySelectionsMatch, validateComponentStudyResult,
  type ComponentStudyCatalog, type ComponentStudyMetadata, type ComponentStudyRequest,
  type ComponentStudyResolution, type ComponentStudyResult,
} from "./component-study.ts";

/** Admission pins for the independently exported, scoped result packet. */
export const COMPONENT_STUDY_PINS = {
  metadata: "226501da329ee8dd079ae531917427c51a5747dfe24e14b7be58d5c2082e206a",
  index: "e9dd053c310e91d55f10bd569302ef0016e4d678f7e872f2c033a62b9b56a3f0",
} as const;
export class ComponentStudyInputError extends Error {}
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);

async function pinnedBytes(directory: string, filename: string, expected: string): Promise<Buffer> {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*\.json$/.test(filename) || !/^[a-f0-9]{64}$/.test(expected)) throw Error("Invalid component-study artifact reference.");
  const root = await realpath(directory), file = await realpath(join(root, filename));
  if (dirname(file) !== root) throw Error("Component-study artifact leaves its publication directory.");
  const bytes = await readFile(file);
  if (createHash("sha256").update(bytes).digest("hex") !== expected) throw Error("Component-study artifact fingerprint mismatch.");
  return bytes;
}
async function pinnedFile(directory: string, filename: string, expected: string): Promise<unknown> {
  return JSON.parse((await pinnedBytes(directory, filename, expected)).toString("utf8"));
}

/** Read and validate all advertised results before exposing any catalog entries. */
export async function loadComponentStudies(dataDirectory: string): Promise<{ catalog: ComponentStudyCatalog; results: ComponentStudyResult[]; resultBytes: Record<string, Buffer> }> {
  const metadata = await pinnedFile(dataDirectory, "model-components-v2.json", COMPONENT_STUDY_PINS.metadata) as ComponentStudyMetadata;
  const index = await pinnedFile(join(dataDirectory, "component-results"), "index.json", COMPONENT_STUDY_PINS.index);
  if (!object(index) || index.schemaVersion !== 1 || !Array.isArray(index.results) || index.results.length !== 2 || metadata.savedExamples.length !== 2) throw Error("Invalid component-study index.");
  const results: ComponentStudyResult[] = [], studies: ComponentStudyCatalog["studies"] = [], ids = new Set<string>(), resultBytes: Record<string, Buffer> = {};
  for (const entry of index.results) {
    if (!object(entry) || typeof entry.id !== "string" || ids.has(entry.id) || typeof entry.file !== "string" || typeof entry.sha256 !== "string") throw Error("Invalid component-study result entry.");
    const example = metadata.savedExamples.find(example => example.id === entry.id);
    const context = metadata.supportedContexts.find(context => context.id === example?.contextId);
    const bytes = await pinnedBytes(join(dataDirectory, "component-results"), entry.file, entry.sha256);
    const result: unknown = JSON.parse(bytes.toString("utf8"));
    if (!example || !context || !validateComponentStudyResult(result) || result.id !== entry.id || result.contextId !== entry.contextId || result.contextId !== context.id ||
      result.configurationSha256 !== entry.configurationSha256 || result.configurationSha256 !== example.configurationSha256 ||
      result.requestSha256 !== example.requestSha256 || result.artifactId !== example.artifactId ||
      result.year !== context.year || result.races.length !== context.raceCount ||
      JSON.stringify(result.races.map(race => race.id)) !== JSON.stringify(context.raceIds) ||
      result.targetId !== context.targetId || result.outputContractId !== context.outputContractId || result.informationCutoff !== context.informationCutoff ||
      result.sourceFingerprints.interfaceReceipt !== metadata.sourceFingerprints.interfaceReceipt || result.sourceFingerprints.readyReceipt !== metadata.sourceFingerprints.readyReceipt ||
      result.sourceFingerprints.readyResult !== example.readyFileSha256 || result.sourceFingerprints.summary !== example.summaryFileSha256 ||
      result.sourceFingerprints.resultIndex !== metadata.sourceFingerprints.resultIndex || result.qualification !== context.qualification ||
      result.sourceFingerprints.dataBundle !== context.dataBundleSha256 || result.sourceFingerprints.scientificPlan !== context.scientificPlanSha256 ||
      !studySelectionsMatch(metadata, result.selections, example.requestTemplate.selections)) throw Error("Component-study result does not match its reviewed context.");
    ids.add(result.id); results.push(result); resultBytes[result.id] = bytes;
    studies.push({ id: result.id, contextId: result.contextId, label: result.label, year: result.year, raceCount: result.races.length,
      resultId: result.id, sha256: entry.sha256, selections: result.selections, configurationSha256: result.configurationSha256,
      requestSha256: result.requestSha256, artifactId: result.artifactId });
  }
  return { catalog: { schemaVersion: 1, catalog: metadata, studies, executionAvailable: false, scheduled: false }, results, resultBytes };
}

export function resolveComponentStudy(catalog: ComponentStudyCatalog, value: unknown): ComponentStudyResolution {
  if (!object(value) || Object.keys(value).length !== 2 || !Object.hasOwn(value, "studyId") || !Object.hasOwn(value, "selections") || typeof value.studyId !== "string")
    throw new ComponentStudyInputError("Supply only a studyId and its seven component selections.");
  const study = catalog.studies.find(study => study.id === value.studyId);
  if (!study) throw new ComponentStudyInputError("Unknown component study.");
  const selections = canonicalStudySelections(catalog.catalog, value.selections);
  if (!selections) throw new ComponentStudyInputError("Selections must contain all seven known instances, exact provider versions, and declared option values only.");
  const request = value as unknown as ComponentStudyRequest;
  const base = { schemaVersion: 1 as const, studyId: study.id, contextId: study.contextId, canonicalSelections: selections,
    resultId: null, resultSha256: null, configurationSha256: null, executionAvailable: false as const, scheduled: false as const };
  if (studySelectionsMatch(catalog.catalog, request.selections, study.selections)) return {
    ...base, state: "ready", resultId: study.resultId, resultSha256: study.sha256, configurationSha256: study.configurationSha256,
    reason: "This exact source-bound component selection has a verified saved race-level result. Predictive evidence remains unassessed.",
  };
  const pending = catalog.catalog.lifecycleExamples.find(example => example.contextId === study.contextId && studySelectionsMatch(catalog.catalog, request.selections, example.requestTemplate.selections));
  if (pending) return { ...base, state: "requires_computation", reason: pending.reason + ". No computation has been scheduled on this site." };
  return { ...base, state: "incompatible", reason: "This deployment has no reviewed result or source-bound resolution for this exact draft in the selected study context. This does not establish scientific impossibility; the scientific resolver must check it before computation." };
}
