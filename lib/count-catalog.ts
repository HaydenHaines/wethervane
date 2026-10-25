/** Browser-safe description of one frozen count-application preparation. */
export const COUNT_ROLES = ["population", "community", "fundamentals", "environment",
  "measurement", "participation", "choice", "counts_rules"] as const;
export type CountRole = typeof COUNT_ROLES[number];

export const COUNT_PROVIDERS: Readonly<Record<CountRole, string>> = Object.freeze({
  population: "protected_full_parent_exposure_v1",
  community: "demographic_profile_drift_v1",
  fundamentals: "lean_carryover_context_incumbency_v1",
  environment: "separated_lkj2_cycle_student5_race_v1",
  measurement: "joint_CES_F1a_final_pair_total_error_v1",
  participation: "profile_turnout_response_observed_total_gamma_v1",
  choice: "candidate_TX_ballot_role_profile_choice_v1",
  counts_rules: "single_mark_category_allocation_poisson_v1",
});
/** Expected identities for this frozen handoff; source authentication is server-side. */
const FROZEN = Object.freeze({
  request: "dcb847b48065d5e8f3210b4ffc518a350130f17dd1dd5535a58461d65095afa7",
  configuration: "bc3c7e317f9c8b938b1df0cb67eb2cfa948c5c5c5d84264fbabbb38f3662d460",
  problem: "96f890fae93fec40caa0b58fb67b7df264259fc533b5d7a23eab70c8a4dc7034",
  bindings: {
    catalogSha256: "4b94b8a9e3be9bfc8db638a8d72fad6ec7b720c9fce35aaba3285b0056471f18",
    requestArtifactSha256: "77a56140a55a06524f54d8c157c950f978e06e312d6411b0001d21f87f9d40ff",
    resolutionSha256: "5f6f187ca675d65840bbe0ff80cbd83726677b2c3c0ccb691d533a4d57e9a852",
    bundleSha256: "c816c32ad1bf4ec6832591c32b47f90c393d5be17393ad69a9a8ede580388a35",
    registrationSha256: "256c73e8731bf7b297fb70c5a6a2512a78f5f4b5252f945ab391d99379af6393",
    planSha256: "adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1",
    scopeReceiptSha256: "f5d8b13948095253d3fc5970d9b1827576cb20e7791687d1dd6e5fe50e0da2c5",
  },
});

export interface CountChoice {
  instanceId: CountRole;
  providerId: string;
  version: "1";
  options: Record<string, string>;
}
export interface CountProviderView {
  providerId: string;
  version: "1";
  instanceId: CountRole;
  label: string;
  optionsSchema: { type: "object"; additionalProperties: false };
  researchStatus: "implemented_preparation";
  limitations: string[];
}
export interface CountCatalogView {
  schema: "CountCatalogViewV1";
  targetId: "population_cut_v1";
  scope: {
    state: "Texas";
    stateFips: "48";
    kind: "fixed_roster_conditional_research";
    trainingCycles: [2020, 2024];
    predictionCycle: 2026;
    predictionElectionDate: "2026-11-03";
    predictionContestIds: ["contest_tx2026_regular"];
    sourceProblemSha256: string;
  };
  prepared: {
    requestSha256: string;
    configurationSha256: string;
    informationCutoff: "2026-10-20T22:13:33Z";
    outputContractId: "whole_parent_joint_candidate_counts_v1";
    independentRoleSwaps: false;
    status: "requires_computation";
    reason: string;
    selections: CountChoice[];
  };
  roles: { role: CountRole; providers: [CountProviderView] }[];
  bindings: {
    catalogSha256: string;
    requestArtifactSha256: string;
    resolutionSha256: string;
    bundleSha256: string;
    registrationSha256: string;
    planSha256: string;
    scopeReceiptSha256: string;
  };
  statuses: {
    inferenceEligibility: "not_established_for_current_execution";
    savedResult: "none_registered";
    scheduler: "unavailable";
    empiricalQualification: "unassessed";
    publication: "unavailable";
  };
}
export interface CountCatalogDraft {
  targetId: string;
  scopeSourceProblemSha256: string;
  roles: CountChoice[];
}
export type CountCatalogResolution =
  | { state: "requires_computation"; configurationSha256: string; reason: string }
  | { state: "unsupported" | "stale"; configurationSha256: null; reason: string };

const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const exact = (v: Record<string, unknown>, keys: string[]) => Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
const safeText = (v: unknown): v is string => typeof v === "string" && v.length > 0 && v.length <= 1000 &&
  v === v.trim() && !/[\u0000-\u001f\u007f]/.test(v) && !v.includes("/");

function choices(value: unknown): value is CountChoice[] {
  if (!Array.isArray(value) || value.length !== COUNT_ROLES.length) return false;
  const seen = new Set<string>();
  for (const row of value) {
    if (!object(row) || !exact(row, ["instanceId", "providerId", "version", "options"]) ||
      !COUNT_ROLES.includes(row.instanceId as CountRole) || seen.has(String(row.instanceId)) ||
      row.providerId !== COUNT_PROVIDERS[row.instanceId as CountRole] || row.version !== "1" ||
      !object(row.options) || Object.keys(row.options).length !== 0) return false;
    seen.add(String(row.instanceId));
  }
  return true;
}

/** Reject unknown/private fields before a metadata DTO reaches presentation code. */
export function parseCountCatalogView(value: unknown): CountCatalogView | null {
  if (!object(value) || !exact(value, ["schema", "targetId", "scope", "prepared", "roles", "bindings", "statuses"]) ||
    value.schema !== "CountCatalogViewV1" || value.targetId !== "population_cut_v1" ||
    !object(value.scope) || !exact(value.scope, ["state", "stateFips", "kind", "trainingCycles", "predictionCycle", "predictionElectionDate", "predictionContestIds", "sourceProblemSha256"]) ||
    value.scope.state !== "Texas" || value.scope.stateFips !== "48" || value.scope.kind !== "fixed_roster_conditional_research" ||
    JSON.stringify(value.scope.trainingCycles) !== "[2020,2024]" || value.scope.predictionCycle !== 2026 ||
    value.scope.predictionElectionDate !== "2026-11-03" || JSON.stringify(value.scope.predictionContestIds) !== '["contest_tx2026_regular"]' ||
    value.scope.sourceProblemSha256 !== FROZEN.problem || !object(value.prepared) ||
    !exact(value.prepared, ["requestSha256", "configurationSha256", "informationCutoff", "outputContractId", "independentRoleSwaps", "status", "reason", "selections"]) ||
    value.prepared.requestSha256 !== FROZEN.request || value.prepared.configurationSha256 !== FROZEN.configuration ||
    value.prepared.informationCutoff !== "2026-10-20T22:13:33Z" || value.prepared.outputContractId !== "whole_parent_joint_candidate_counts_v1" ||
    value.prepared.independentRoleSwaps !== false ||
    value.prepared.status !== "requires_computation" || !safeText(value.prepared.reason) || !choices(value.prepared.selections) ||
    !Array.isArray(value.roles) || value.roles.length !== COUNT_ROLES.length) return null;
  const seen = new Set<string>();
  for (const row of value.roles) {
    if (!object(row) || !exact(row, ["role", "providers"]) || !COUNT_ROLES.includes(row.role as CountRole) ||
      seen.has(String(row.role)) || !Array.isArray(row.providers) || row.providers.length !== 1) return null;
    seen.add(String(row.role));
    const provider = row.providers[0];
    if (!object(provider) || !exact(provider, ["providerId", "version", "instanceId", "label", "optionsSchema", "researchStatus", "limitations"]) ||
      provider.providerId !== COUNT_PROVIDERS[row.role as CountRole] || provider.version !== "1" || provider.instanceId !== row.role ||
      !safeText(provider.label) || !object(provider.optionsSchema) || !exact(provider.optionsSchema, ["type", "additionalProperties"]) ||
      provider.optionsSchema.type !== "object" || provider.optionsSchema.additionalProperties !== false ||
      provider.researchStatus !== "implemented_preparation" || !Array.isArray(provider.limitations) ||
      provider.limitations.length === 0 || !provider.limitations.every(safeText)) return null;
  }
  const bindings = value.bindings;
  if (!object(bindings) || !exact(bindings, ["catalogSha256", "requestArtifactSha256", "resolutionSha256", "bundleSha256", "registrationSha256", "planSha256", "scopeReceiptSha256"]) ||
    !Object.entries(FROZEN.bindings).every(([key, expected]) => bindings[key] === expected) || !object(value.statuses) ||
    !exact(value.statuses, ["inferenceEligibility", "savedResult", "scheduler", "empiricalQualification", "publication"]) ||
    value.statuses.inferenceEligibility !== "not_established_for_current_execution" || value.statuses.savedResult !== "none_registered" ||
    value.statuses.scheduler !== "unavailable" || value.statuses.empiricalQualification !== "unassessed" ||
    value.statuses.publication !== "unavailable") return null;
  for (const selection of value.prepared.selections) {
    const role = value.roles.find(row => row.role === selection.instanceId);
    if (!role || role.providers[0].providerId !== selection.providerId || role.providers[0].version !== selection.version) return null;
  }
  return structuredClone(value) as unknown as CountCatalogView;
}

/** Only the exact prepared whole-stack choice has a recorded resolution. */
export function resolveCountCatalogSelection(view: CountCatalogView, draft: CountCatalogDraft): CountCatalogResolution {
  const catalog = parseCountCatalogView(view);
  if (!catalog) return { state: "stale", configurationSha256: null, reason: "The count research catalog failed validation." };
  if (!object(draft) || !exact(draft, ["targetId", "scopeSourceProblemSha256", "roles"]) ||
    draft.targetId !== catalog.targetId || draft.scopeSourceProblemSha256 !== catalog.scope.sourceProblemSha256 || !choices(draft.roles))
    return { state: "unsupported", configurationSha256: null, reason: "This target, scope, or provider set has no recorded prepared count configuration." };
  const key = (rows: CountChoice[]) => JSON.stringify(rows.map(row => [row.instanceId, row.providerId, row.version, row.options])
    .sort((a, b) => String(a[0]).localeCompare(String(b[0]))));
  if (key(draft.roles) !== key(catalog.prepared.selections))
    return { state: "unsupported", configurationSha256: null, reason: "This component set differs from the recorded preparation." };
  return { state: "requires_computation", configurationSha256: catalog.prepared.configurationSha256, reason: catalog.prepared.reason };
}
