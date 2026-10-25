/** Browser-safe contracts for the two source-bound ordinary-race studies. */
export interface ComponentStudySelection {
  instance_id: string;
  provider_id: string;
  version: string;
  options: Record<string, string>;
}
export interface ComponentStudyProvider {
  provider_id: string;
  version: string;
  default_instance_id: string;
  role: string;
  label: string;
  availability: string;
  options_schema: {
    type: "object";
    additionalProperties: false;
    properties: Record<string, { type: "string"; enum: string[]; default: string; description?: string }>;
  };
  [key: string]: unknown;
}
export interface ComponentStudyContext {
  id: string;
  year: number;
  supportId: string;
  raceIds: string[];
  raceCount: number;
  targetId: string;
  informationCutoff: string;
  outputContractId: string;
  dataBundleSha256: string;
  scientificPlanSha256: string;
  qualification: string;
  strictAsOf: false;
  [key: string]: unknown;
}
export interface ComponentStudyExample {
  id: string;
  contextId: string;
  requestTemplate: { selections: ComponentStudySelection[]; [key: string]: unknown };
  requestSha256: string;
  configurationSha256: string;
  artifactId: string;
  readyFileSha256: string;
  summaryFileSha256: string;
  [key: string]: unknown;
}
export interface ComponentStudyMetadata {
  schemaVersion: 2;
  id: string;
  activation: "metadata_only";
  execution: { available: false; schedulerAvailable: false; reason: string };
  roles: { role: string; label: string; description: string; availability: string; providers: ComponentStudyProvider[] }[];
  supportedContexts: ComponentStudyContext[];
  savedExamples: ComponentStudyExample[];
  lifecycleExamples: { id: string; contextId: string; requestTemplate: { selections: ComponentStudySelection[]; [key: string]: unknown };
    sourceLifecycle: "requires_computation"; displayedResultId: null; scheduled: false; reason: string; sourceFingerprint: string }[];
  sourceFingerprints: { interfaceReceipt: string; readyReceipt: string; [key: string]: unknown };
  limitations: string[];
  [key: string]: unknown;
}
export interface ComponentStudyCandidate {
  id: "D" | "R";
  name: string;
  party: "D" | "R";
  incumbent: boolean;
  winProbability: number;
  share: { mean: number; denominator: string; quantiles: { level: number; value: number }[] };
}
export interface ComponentStudyRace {
  id: string;
  state: string;
  seatId: string;
  candidates: ComponentStudyCandidate[];
  observed: { democraticVotes: number; republicanVotes: number; twoPartyVotes: number; democraticTwoPartyShare: number };
  notes: string[];
}
export interface ComponentStudyResult {
  schemaVersion: 1;
  id: string;
  contextId: string;
  label: string;
  year: 2018;
  scope: "ordinary_dr_race_subset";
  informationCutoff: string;
  targetId: string;
  outputContractId: string;
  requestSha256: string;
  configurationSha256: string;
  artifactId: string;
  selections: ComponentStudySelection[];
  status: { lifecycle: "ready"; numerical: "passed"; evidence: "unassessed"; experimental: true };
  qualification: string;
  strictAsOf: false;
  capabilities: { raceProbabilities: true; conditionalTwoPartyShares: true; chamberControl: false; tipping: false; counts: false };
  quantileLevels: number[];
  races: ComponentStudyRace[];
  sourceAttribution: { label: string; description: string; urls: string[] }[];
  sourceFingerprints: Record<string, string>;
  limitations: string[];
}
export interface ComponentStudyEntry {
  id: string;
  contextId: string;
  label: string;
  year: 2018;
  raceCount: number;
  resultId: string;
  sha256: string;
  selections: ComponentStudySelection[];
  configurationSha256: string;
  requestSha256: string;
  artifactId: string;
}
export interface ComponentStudyCatalog {
  schemaVersion: 1;
  catalog: ComponentStudyMetadata;
  studies: ComponentStudyEntry[];
  executionAvailable: false;
  scheduled: false;
}
export interface ComponentStudyRequest {
  studyId: string;
  selections: ComponentStudySelection[];
}
export interface ComponentStudyResolution {
  schemaVersion: 1;
  state: "ready" | "requires_computation" | "incompatible";
  studyId: string;
  contextId: string;
  canonicalSelections: ComponentStudySelection[];
  resultId: string | null;
  resultSha256: string | null;
  configurationSha256: string | null;
  reason: string;
  executionAvailable: false;
  scheduled: false;
}

const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === "string" && value.length > 0 && value.length < 3000;
const hash = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
const probability = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(text);
const exactKeys = (value: Record<string, unknown>, keys: string[]) => Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
const close = (left: number, right: number) => Math.abs(left - right) <= 1e-8;

/** Fill only provider-declared defaults; template overrides remain substantive. */
export function canonicalStudySelections(metadata: ComponentStudyMetadata, value: unknown): ComponentStudySelection[] | null {
  if (!Array.isArray(value) || value.length !== 7) return null;
  const providers = metadata.roles.flatMap(role => role.providers);
  if (providers.length !== 7) return null;
  const instances = new Set<string>(), result: ComponentStudySelection[] = [];
  for (const selection of value) {
    if (!object(selection) || !exactKeys(selection, ["instance_id", "provider_id", "version", "options"]) ||
      !text(selection.instance_id) || instances.has(selection.instance_id) || !object(selection.options)) return null;
    const provider = providers.find(provider => provider.default_instance_id === selection.instance_id && provider.provider_id === selection.provider_id && provider.version === selection.version);
    if (!provider || provider.availability !== "available" || provider.options_schema.additionalProperties !== false) return null;
    const properties = provider.options_schema.properties, options: Record<string, string> = {};
    if (Object.keys(selection.options).some(key => !Object.hasOwn(properties, key))) return null;
    for (const [key, schema] of Object.entries(properties)) {
      const chosen = Object.hasOwn(selection.options, key) ? selection.options[key] : schema.default;
      if (schema.type !== "string" || typeof chosen !== "string" || !schema.enum.includes(chosen)) return null;
      options[key] = chosen;
    }
    instances.add(selection.instance_id);
    result.push({ instance_id: selection.instance_id, provider_id: provider.provider_id, version: provider.version, options });
  }
  return result.sort((a, b) => a.instance_id < b.instance_id ? -1 : a.instance_id > b.instance_id ? 1 : 0);
}
export function studySelectionsMatch(metadata: ComponentStudyMetadata, first: unknown, second: unknown): boolean {
  const left = canonicalStudySelections(metadata, first), right = canonicalStudySelections(metadata, second);
  return !!left && !!right && JSON.stringify(left) === JSON.stringify(right);
}

/** Reject incorrect scopes, intervals and categorical accounting before rendering. */
export function validateComponentStudyResult(value: unknown): value is ComponentStudyResult {
  if (!object(value) || value.schemaVersion !== 1 || !["ordinary-2018-f", "ordinary-2018-fp"].includes(String(value.id)) ||
    value.contextId !== `ordinary-senate-2018-${value.id === "ordinary-2018-f" ? "f" : "fp"}` ||
    !text(value.label) || value.year !== 2018 || value.scope !== "ordinary_dr_race_subset" ||
    value.informationCutoff !== "2018-10-23T00:00:00Z" || value.targetId !== "senate_aggregate_logit_empirical_mixture_v1" ||
    value.outputContractId !== "empirical_linear_fundamentals_joint_v1" || ![value.requestSha256, value.configurationSha256, value.artifactId].every(hash) ||
    !Array.isArray(value.selections) || value.selections.length !== 7 ||
    !object(value.status) || value.status.lifecycle !== "ready" || value.status.numerical !== "passed" || value.status.evidence !== "unassessed" || value.status.experimental !== true ||
    !text(value.qualification) || value.strictAsOf !== false || !object(value.capabilities) ||
    !exactKeys(value.capabilities, ["raceProbabilities", "conditionalTwoPartyShares", "chamberControl", "tipping", "counts"]) ||
    value.capabilities.raceProbabilities !== true || value.capabilities.conditionalTwoPartyShares !== true ||
    value.capabilities.chamberControl !== false || value.capabilities.tipping !== false || value.capabilities.counts !== false ||
    JSON.stringify(value.quantileLevels) !== JSON.stringify([0.025, 0.1, 0.5, 0.9, 0.975]) ||
    !Array.isArray(value.races) || value.races.length !== 30 || !strings(value.limitations) ||
    !object(value.sourceFingerprints) || !Object.values(value.sourceFingerprints).every(hash) ||
    !["interfaceReceipt", "readyReceipt", "readyResult", "resultIndex", "summary", "dataBundle", "targetFeatures", "observedOutcomeLabels", "panelReceipt", "scientificPlan", "trainingLaw", "recipe"].every(key => hash((value.sourceFingerprints as Record<string, unknown>)[key])) ||
    !Array.isArray(value.sourceAttribution) || !value.sourceAttribution.every(source => object(source) && text(source.label) && text(source.description) && Array.isArray(source.urls) && source.urls.every(url => typeof url === "string" && /^https?:\/\//.test(url)))) return false;
  const instances = new Set<string>();
  for (const selection of value.selections) {
    if (!object(selection) || !exactKeys(selection, ["instance_id", "provider_id", "version", "options"]) ||
      !["structure", "candidate", "carryover", "context", "cycle", "race", "poll"].includes(String(selection.instance_id)) ||
      instances.has(String(selection.instance_id)) || !text(selection.provider_id) || !text(selection.version) ||
      !object(selection.options) || !Object.values(selection.options).every(text)) return false;
    instances.add(String(selection.instance_id));
  }
  const ids = new Set<string>(), seats = new Set<string>();
  for (const race of value.races) {
    if (!object(race) || typeof race.state !== "string" || !/^[A-Z]{2}$/.test(race.state) || race.id !== `2018:${race.state}:senate:regular` || ids.has(String(race.id)) ||
      !text(race.seatId) || seats.has(race.seatId) || !strings(race.notes) || !Array.isArray(race.candidates) || race.candidates.length !== 2) return false;
    ids.add(String(race.id)); seats.add(race.seatId);
    const candidates: ComponentStudyCandidate[] = [];
    for (const candidate of race.candidates) {
      if (!object(candidate) || !["D", "R"].includes(String(candidate.id)) || candidate.id !== candidate.party || !text(candidate.name) ||
        typeof candidate.incumbent !== "boolean" || !probability(candidate.winProbability) || !object(candidate.share) ||
        !probability(candidate.share.mean) || candidate.share.denominator !== "Democratic and Republican candidate votes only" ||
        !Array.isArray(candidate.share.quantiles) || candidate.share.quantiles.length !== 5) return false;
      for (let index = 0; index < 5; index++) {
        const quantile = candidate.share.quantiles[index];
        if (!object(quantile) || quantile.level !== [0.025, 0.1, 0.5, 0.9, 0.975][index] || !probability(quantile.value) ||
          (index > 0 && quantile.value < candidate.share.quantiles[index - 1].value)) return false;
      }
      candidates.push(candidate as unknown as ComponentStudyCandidate);
    }
    if (candidates[0].id === candidates[1].id || !close(candidates[0].winProbability + candidates[1].winProbability, 1) ||
      !close(candidates[0].share.mean + candidates[1].share.mean, 1) ||
      candidates[0].share.quantiles.some((quantile, index) => !close(quantile.value + candidates[1].share.quantiles[4 - index].value, 1))) return false;
    const observed = race.observed;
    if (!object(observed) || ![observed.democraticVotes, observed.republicanVotes, observed.twoPartyVotes].every(value => typeof value === "number" && Number.isSafeInteger(value) && value >= 0) ||
      Number(observed.twoPartyVotes) <= 0 || Number(observed.democraticVotes) + Number(observed.republicanVotes) !== observed.twoPartyVotes ||
      !probability(observed.democraticTwoPartyShare) || !close(Number(observed.democraticVotes) / Number(observed.twoPartyVotes), observed.democraticTwoPartyShare)) return false;
  }
  return true;
}
