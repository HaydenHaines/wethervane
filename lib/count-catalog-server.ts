/** Private, byte-pinned reader for one recorded native count preparation. */
import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import { isAbsolute, join, relative } from "node:path";
import { COUNT_PROVIDERS, COUNT_ROLES, parseCountCatalogView,
  type CountCatalogView, type CountChoice, type CountRole } from "./count-catalog.ts";

export const COUNT_CATALOG_SOURCE_PINS = Object.freeze({
  "catalog.json": "4b94b8a9e3be9bfc8db638a8d72fad6ec7b720c9fce35aaba3285b0056471f18",
  "request.json": "77a56140a55a06524f54d8c157c950f978e06e312d6411b0001d21f87f9d40ff",
  "resolution.json": "5f6f187ca675d65840bbe0ff80cbd83726677b2c3c0ccb691d533a4d57e9a852",
  "bundle.json": "c816c32ad1bf4ec6832591c32b47f90c393d5be17393ad69a9a8ede580388a35",
  "registration.json": "256c73e8731bf7b297fb70c5a6a2512a78f5f4b5252f945ab391d99379af6393",
  "scope.json": "f5d8b13948095253d3fc5970d9b1827576cb20e7791687d1dd6e5fe50e0da2c5",
});
const REQUEST_SHA = "dcb847b48065d5e8f3210b4ffc518a350130f17dd1dd5535a58461d65095afa7";
const CONFIGURATION_SHA = "bc3c7e317f9c8b938b1df0cb67eb2cfa948c5c5c5d84264fbabbb38f3662d460";
const PLAN_SHA = "adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1";
const PROBLEM_SHA = "96f890fae93fec40caa0b58fb67b7df264259fc533b5d7a23eab70c8a4dc7034";
const REGISTRATION = "reviewed_TX_eight_role_baseline_fixed_completion_v1";

const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const compare = (a: string, b: string) => a < b ? -1 : a > b ? 1 : 0;
const fail = (message: string): never => { throw Error(`Count catalog metadata invalid: ${message}`); };
const sha = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (object(value)) return Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => [key, sortKeys(item)]));
  return value;
}
const digest = (value: unknown) => sha(JSON.stringify(sortKeys(value)));

async function staged(directory: string, name: keyof typeof COUNT_CATALOG_SOURCE_PINS): Promise<Record<string, unknown>> {
  const root = await realpath(directory), file = join(root, name), resolved = await realpath(file);
  const inside = relative(root, resolved);
  if (!inside || inside.startsWith("..") || isAbsolute(inside) || (await lstat(file)).isSymbolicLink()) fail("artifact path escapes the private stage");
  const stat = await lstat(resolved);
  if (!stat.isFile() || stat.size > 100_000) fail("artifact is not a bounded regular file");
  const bytes = await readFile(resolved);
  if (bytes.length !== stat.size || sha(bytes) !== COUNT_CATALOG_SOURCE_PINS[name]) fail(`${name} fingerprint mismatch`);
  const value: unknown = JSON.parse(bytes.toString("utf8"));
  if (!object(value)) return fail(`${name} is not an object`);
  return value;
}

function preparedChoices(request: Record<string, unknown>, catalog: Record<string, unknown>): CountChoice[] {
  const selections = request.selections;
  if (!Array.isArray(selections) || selections.length !== 8 || !Array.isArray(catalog.roles) || catalog.roles.length !== 8)
    return fail("eight roles/selections are required");
  const seen = new Set<string>(), choices: CountChoice[] = [];
  for (const selection of selections) {
    if (!object(selection) || typeof selection.instance_id !== "string" || !COUNT_ROLES.includes(selection.instance_id as CountRole) ||
      seen.has(selection.instance_id) || selection.provider_id !== COUNT_PROVIDERS[selection.instance_id as CountRole] ||
      selection.version !== "1" || !object(selection.options) || Object.keys(selection.options).length !== 0) fail("unrecognized prepared selection");
    seen.add(selection.instance_id);
    choices.push({ instanceId: selection.instance_id as CountRole, providerId: String(selection.provider_id), version: "1", options: {} });
  }
  return choices;
}

export async function loadCountCatalogView(dataDirectory: string): Promise<CountCatalogView> {
  const [catalog, request, resolution, bundle, registration, scope] = await Promise.all(
    (Object.keys(COUNT_CATALOG_SOURCE_PINS) as (keyof typeof COUNT_CATALOG_SOURCE_PINS)[])
      .map(name => staged(dataDirectory, name)));
  if (catalog.schema_version !== "composition-catalog-v1" || !same(catalog.executable_targets, ["population_cut_v1"]) ||
    !same(catalog.saved_presets, []) || catalog.default_preset_id !== null || catalog.scheduler_available !== false ||
    request.schema_version !== "composition-request-v1" || request.target_id !== "population_cut_v1" ||
    request.information_cutoff !== "2026-10-20T22:13:33Z" || request.output_contract_id !== "whole_parent_joint_candidate_counts_v1" ||
    !object(request.context) || request.context.execution_registration !== REGISTRATION ||
    !object(request.data_bundle) || request.data_bundle.sha256 !== COUNT_CATALOG_SOURCE_PINS["bundle.json"] ||
    resolution.schema_version !== "composition-resolution-v1" || resolution.status !== "requires_computation" ||
    resolution.artifact !== null || resolution.record !== null || resolution.summary !== null ||
    !object(resolution.configuration) || !same(resolution.configuration.request, request) ||
    !object(bundle) || bundle.schema_version !== "count_application_bundle_v1" || bundle.registration_id !== REGISTRATION ||
    !object(registration) || registration.registration_id !== REGISTRATION ||
    !object(bundle.scientific_plan) || bundle.scientific_plan.sha256 !== PLAN_SHA ||
    !object(registration.plan) || registration.plan.sha256 !== PLAN_SHA ||
    !object(resolution.configuration.scientific_plan) || resolution.configuration.scientific_plan.sha256 !== PLAN_SHA ||
    !same(bundle.scientific_plan, registration.plan) || !same(bundle.scientific_plan, resolution.configuration.scientific_plan))
    fail("native metadata cross-binding");

  const configuration = resolution.configuration;
  if (!object(configuration)) return fail("missing prepared configuration");
  const req = { ...request, selections: [...(request.selections as unknown[])].sort((a, b) =>
    compare(String((a as Record<string, unknown>).instance_id), String((b as Record<string, unknown>).instance_id))) };
  const requestSha = digest(req);
  const cfg = { ...configuration, request: requestSha,
    dependencies: Array.isArray(configuration.dependencies) ? [...configuration.dependencies].sort((a, b) =>
      compare(String((a as Record<string, unknown>).path), String((b as Record<string, unknown>).path))) : fail("missing dependencies"),
    outputs: Array.isArray(configuration.outputs) ? [...configuration.outputs].sort((a, b) =>
      compare(String((a as Record<string, unknown>).port), String((b as Record<string, unknown>).port))) : fail("missing outputs") };
  const configurationSha = digest(cfg);
  if (requestSha !== REQUEST_SHA || resolution.request_sha256 !== REQUEST_SHA || configurationSha !== CONFIGURATION_SHA ||
    configuration.driver_id !== "scientific_count_application" || configuration.driver_version !== "1" ||
    !hash(configuration.law_sha256) || !hash(configuration.graph_sha256) ||
    !hash(configuration.support_id) || !Array.isArray(configuration.outputs) ||
    !configuration.outputs.some((out: unknown) => object(out) && out.contract_id === request.output_contract_id) ||
    typeof resolution.reason !== "string" || !resolution.reason || resolution.reason.includes("/")) fail("prepared identity or output contract");

  const selections = preparedChoices(request, catalog), seen = new Set<string>();
  const roles = (catalog.roles as unknown[]).map((row): CountCatalogView["roles"][number] => {
    if (!object(row)) return fail("invalid role inventory");
    if (typeof row.role !== "string" || !COUNT_ROLES.includes(row.role as CountRole) || seen.has(row.role) ||
      !Array.isArray(row.providers) || row.providers.length !== 1) return fail("invalid role inventory");
    seen.add(row.role);
    const offer = row.providers[0];
    if (!object(offer) || offer.provider_id !== COUNT_PROVIDERS[row.role as CountRole] || offer.version !== "1" ||
      offer.role !== row.role || offer.default_instance_id !== row.role || offer.availability !== "available" ||
      !same(offer.target_ids, ["population_cut_v1"]) || !object(offer.options_schema) ||
      !same(offer.options_schema, { type: "object", additionalProperties: false }) ||
      typeof offer.label !== "string" || !Array.isArray(offer.limitations) || !offer.limitations.length ||
      !offer.limitations.every(item => typeof item === "string") || !Array.isArray(offer.sources) || !offer.sources.length ||
      !Array.isArray(offer.outputs) || !offer.outputs.length) fail("provider offer no longer matches the frozen preparation");
    return { role: row.role as CountRole, providers: [{ providerId: offer.provider_id as string,
      version: "1", instanceId: row.role as CountRole, label: offer.label,
      optionsSchema: { type: "object", additionalProperties: false }, researchStatus: "implemented_preparation",
      limitations: offer.limitations as string[] }] };
  });
  if (seen.size !== COUNT_ROLES.length || !object(scope) || scope.schema !== "CountCatalogScopeReceiptV1" ||
    scope.sourcePlanSha256 !== PLAN_SHA || scope.sourceProblemSha256 !== PROBLEM_SHA || scope.state !== "Texas" ||
    scope.stateFips !== "48" || scope.kind !== "fixed_roster_conditional_research" ||
    !same(scope.trainingCycles, [2020, 2024]) || scope.predictionCycle !== 2026 ||
    scope.predictionElectionDate !== "2026-11-03" || !same(scope.predictionContestIds, ["contest_tx2026_regular"]))
    fail("scope receipt");

  const reason = resolution.reason;
  if (typeof reason !== "string") return fail("missing prepared reason");
  const candidate: CountCatalogView = {
    schema: "CountCatalogViewV1", targetId: "population_cut_v1",
    scope: { state: "Texas", stateFips: "48", kind: "fixed_roster_conditional_research",
      trainingCycles: [2020, 2024], predictionCycle: 2026, predictionElectionDate: "2026-11-03",
      predictionContestIds: ["contest_tx2026_regular"], sourceProblemSha256: PROBLEM_SHA },
    prepared: { requestSha256: requestSha, configurationSha256: configurationSha,
      informationCutoff: "2026-10-20T22:13:33Z", outputContractId: "whole_parent_joint_candidate_counts_v1",
      independentRoleSwaps: false, status: "requires_computation", reason, selections },
    roles,
    bindings: { catalogSha256: COUNT_CATALOG_SOURCE_PINS["catalog.json"],
      requestArtifactSha256: COUNT_CATALOG_SOURCE_PINS["request.json"],
      resolutionSha256: COUNT_CATALOG_SOURCE_PINS["resolution.json"], bundleSha256: COUNT_CATALOG_SOURCE_PINS["bundle.json"],
      registrationSha256: COUNT_CATALOG_SOURCE_PINS["registration.json"], planSha256: PLAN_SHA,
      scopeReceiptSha256: COUNT_CATALOG_SOURCE_PINS["scope.json"] },
    statuses: { inferenceEligibility: "not_established_for_current_execution", savedResult: "none_registered",
      scheduler: "unavailable", empiricalQualification: "unassessed", publication: "unavailable" },
  };
  return parseCountCatalogView(candidate) ?? fail("projected browser contract");
}
