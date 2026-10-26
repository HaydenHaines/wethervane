/** Derived polling disclosure only. Call the builder after authenticating all input bytes. */
import { parseHouseLiveCatalog, type HouseLiveCatalog } from "./house-live-catalog.ts";
import { HOUSE_LIVE_EXPECTED_DISTRICT_KEYS, houseLiveConfigurationKey, parseHouseLiveResult,
  type HouseLiveResult } from "./house-live-result.ts";

export const HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256 =
  "271268d44018e1eb89a2d51bc5a82e2c8159db44139d654c64b1d39fbf38c10a";
export const HOUSE_LIVE_POLL_RULESET = "house_question_selection_100day_v1" as const;

const CONFLICTING_FIELDS = ["state", "seat_number", "cycle", "office_type", "stage",
  "election_date", "start_date", "end_date", "created_at", "population", "sample_size", "pollster"] as const;
export type HouseLivePollExclusionReason =
  | "not_one_current_house_district" | "not_2026_house_regular_general"
  | "outside_100_day_field_end_window" | "unresolved_or_departed_candidate_in_question"
  | "earlier_same_pollster_race_sample" | "reallocated_or_later_round_not_initial_pair_share"
  | "outside_named_principal_pair_support" | "alternative_question_same_sample"
  | "unavailable_at_cutoff_under_created_plus_one_day_rule"
  | "missouri_fieldwork_before_final_old_map_confirmation"
  | "question_does_not_include_current_principal_pair" | "hypothetical_question"
  | "tracking_sample_overlap" | "unsupported_population_or_subgroup"
  | "invalid_dates_or_election" | "invalid_nominal_sample_size"
  | "duplicate_candidate_in_question" | "invalid_toplines_or_denominator"
  | `conflicting_question_metadata:${typeof CONFLICTING_FIELDS[number]}`;
export const HOUSE_LIVE_POLL_EXCLUSION_LABELS: Readonly<Record<HouseLivePollExclusionReason, string>> = Object.freeze({
  not_one_current_house_district: "Not one current House district",
  not_2026_house_regular_general: "Not the 2026 regular House general election",
  outside_100_day_field_end_window: "Fieldwork ended outside the 100-day window",
  unresolved_or_departed_candidate_in_question: "Candidate identity was unresolved or no longer current",
  earlier_same_pollster_race_sample: "An earlier sample from the same pollster in this district",
  reallocated_or_later_round_not_initial_pair_share: "Later-round or reallocated response",
  outside_named_principal_pair_support: "No supported named principal pair",
  alternative_question_same_sample: "Another question from the same sample was selected",
  unavailable_at_cutoff_under_created_plus_one_day_rule: "Not available by the polling cutoff",
  missouri_fieldwork_before_final_old_map_confirmation: "Missouri fieldwork preceded final map confirmation",
  question_does_not_include_current_principal_pair: "Question omitted a current principal contender",
  hypothetical_question: "Hypothetical question",
  tracking_sample_overlap: "Overlapping tracking sample",
  unsupported_population_or_subgroup: "Unsupported population or subgroup",
  invalid_dates_or_election: "Invalid fieldwork or election date",
  invalid_nominal_sample_size: "Invalid reported sample size",
  duplicate_candidate_in_question: "Duplicate candidate in the question",
  invalid_toplines_or_denominator: "Invalid toplines or pair denominator",
  ...Object.fromEntries(CONFLICTING_FIELDS.map(field => [
    `conflicting_question_metadata:${field}`, `Conflicting ${field.replaceAll("_", " ")} metadata`,
  ])),
} as Record<HouseLivePollExclusionReason, string>);

export interface HouseLivePollingSelectedQuestion {
  observationId: string; raceId: string; pollster: string;
  startDate: string; fieldEnd: string; availableAt: string;
  population: "LV" | "RV" | "A"; sampleSize: number | null;
  sourceId: string; sourceSha256: string; sourceUrls: string[];
}
export interface HouseLivePollingExcludedQuestion {
  districtKey: string | null; pollster: string; endDate: string | null;
  reason: HouseLivePollExclusionReason;
}
export interface HouseLivePollingReasonCount {
  districtKey: string | null; reason: HouseLivePollExclusionReason; count: number;
}
export interface HouseLivePollingBindings {
  catalogSha256: string; sourcePolicySha256: string; admissionSha256: string;
  selectionSha256: string; ledgerSha256: string;
}
export interface HouseLivePollingCutoffs { forecastAt: string; pollingAt: string }
export interface HouseLivePollingCollection {
  schema: "HouseLiveSitePollingDisclosureV1"; cycle: 2026;
  ruleset: typeof HOUSE_LIVE_POLL_RULESET;
  cutoffs: HouseLivePollingCutoffs; bindings: HouseLivePollingBindings;
  runs: { id: string; resultSha256: string; polling: "none" | "standard" }[];
  selected: HouseLivePollingSelectedQuestion[];
  excluded: HouseLivePollingExcludedQuestion[];
  excludedReasonCounts: HouseLivePollingReasonCount[];
}
export interface HouseLivePollingRun {
  schema: "HouseLivePollingRunV1"; runId: string; resultSha256: string;
  ruleset: typeof HOUSE_LIVE_POLL_RULESET;
  cutoffs: HouseLivePollingCutoffs; bindings: HouseLivePollingBindings;
  modelUsesSelectedQuestions: boolean;
  usedQuestions: HouseLivePollingSelectedQuestion[];
  selectedPoolQuestionCount: number; selectedPoolDistrictCount: number;
  excluded: HouseLivePollingExcludedQuestion[];
  excludedReasonCounts: HouseLivePollingReasonCount[];
}
export interface HouseLivePollingBuildInput {
  catalog: HouseLiveCatalog; catalogSha256: string;
  results: { result: HouseLiveResult; resultSha256: string }[];
  sourcePolicy: unknown; sourcePolicySha256: string;
  admission: unknown; admissionSha256: string;
  selection: unknown; selectionSha256: string;
  ledger: unknown; ledgerSha256: string;
}

const DISTRICTS = new Set(HOUSE_LIVE_EXPECTED_DISTRICT_KEYS);
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const keys = (value: Record<string, unknown>, names: readonly string[]) =>
  Object.keys(value).length === names.length && names.every(name => name in value);
const text = (value: unknown): value is string => typeof value === "string" &&
  value.length > 0 && value.length <= 4000 && value === value.trim() &&
  !/[\u0000-\u001f\u007f]/.test(value);
const hash = (value: unknown): value is string => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
const count = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const raceId = (value: unknown): value is string => typeof value === "string" &&
  /^2026:[A-Z]{2}-\d{2}:house:regular$/.test(value) && DISTRICTS.has(value.slice(5, 10));
const validUrl = (value: unknown): value is string => {
  if (!text(value)) return false;
  try {
    const parsed = new URL(value);
    return ["https:", "http:"].includes(parsed.protocol) && !!parsed.hostname &&
      !parsed.username && !parsed.password;
  } catch { return false; }
};
function disclosureUrl(value: unknown): string | null {
  if (validUrl(value)) return value;
  // The reviewed ledger has one domain/path URL without a scheme. Add HTTPS only
  // for a bare domain; never turn a relative path or an arbitrary scheme into a link.
  if (text(value) && /^[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:\/[^\s\\]*)?$/.test(value)) {
    const absolute = `https://${value}`;
    if (validUrl(absolute)) return absolute;
  }
  return null;
}
const date = (value: unknown): value is string => {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map(Number);
  return month >= 1 && month <= 12 && day >= 1 &&
    day <= new Date(Date.UTC(year, month, 0)).getUTCDate();
};
const timestamp = (value: unknown): value is string => {
  if (typeof value !== "string" ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$/.test(value) ||
    !date(value.slice(0, 10))) return false;
  const hour = Number(value.slice(11, 13)), minute = Number(value.slice(14, 16));
  const second = Number(value.slice(17, 19));
  return hour <= 23 && minute <= 59 && second <= 59 && Number.isFinite(Date.parse(value));
};
const sameInstant = (a: string, b: string) => Date.parse(a) === Date.parse(b);
const pin = (value: unknown): value is { path: string; sha256: string } =>
  object(value) && text(value.path) && !value.path.startsWith("/") &&
  !value.path.includes("\\") && !value.path.split("/").some(part => part === "" || part === "." || part === "..") &&
  hash(value.sha256);
const fail = (message: string): never => { throw Error(`Live House polling disclosure: ${message}`); };
function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) fail(message);
}
const reason = (value: unknown): value is HouseLivePollExclusionReason =>
  typeof value === "string" && Object.prototype.hasOwnProperty.call(HOUSE_LIVE_POLL_EXCLUSION_LABELS, value);
const selectedKeys = ["observationId", "raceId", "pollster", "startDate", "fieldEnd", "availableAt",
  "population", "sampleSize", "sourceId", "sourceSha256", "sourceUrls"];
const excludedKeys = ["districtKey", "pollster", "endDate", "reason"];
const countKeys = ["districtKey", "reason", "count"];

function selectedQuestion(value: unknown, cutoff: string): value is HouseLivePollingSelectedQuestion {
  if (!object(value) || !keys(value, selectedKeys) || !text(value.observationId) ||
    !raceId(value.raceId) || !text(value.pollster) || !date(value.startDate) ||
    !date(value.fieldEnd) || !timestamp(value.availableAt) ||
    !["LV", "RV", "A"].includes(String(value.population)) ||
    !(value.sampleSize === null || (count(value.sampleSize) && value.sampleSize > 0)) ||
    !text(value.sourceId) || !hash(value.sourceSha256) ||
    !Array.isArray(value.sourceUrls) || !value.sourceUrls.length ||
    !value.sourceUrls.every(validUrl) || new Set(value.sourceUrls).size !== value.sourceUrls.length) return false;
  return value.startDate <= value.fieldEnd &&
    (Date.UTC(2026, 10, 3) - Date.parse(`${value.fieldEnd}T00:00:00Z`)) / 86_400_000 >= 0 &&
    (Date.UTC(2026, 10, 3) - Date.parse(`${value.fieldEnd}T00:00:00Z`)) / 86_400_000 <= 100 &&
    Date.parse(`${value.fieldEnd}T00:00:00Z`) <= Date.parse(value.availableAt) &&
    Date.parse(value.availableAt) <= Date.parse(cutoff);
}
function excludedQuestion(value: unknown): value is HouseLivePollingExcludedQuestion {
  return object(value) && keys(value, excludedKeys) &&
    (value.districtKey === null || (typeof value.districtKey === "string" && DISTRICTS.has(value.districtKey))) &&
    text(value.pollster) && (value.endDate === null || date(value.endDate)) && reason(value.reason) &&
    (value.reason !== "not_one_current_house_district" || value.districtKey === null) &&
    (value.districtKey !== null || (value.pollster === "Record without a current district" && value.endDate === null));
}
function reasonCounts(excluded: readonly HouseLivePollingExcludedQuestion[]): HouseLivePollingReasonCount[] {
  const totals = new Map<string, HouseLivePollingReasonCount>();
  for (const row of excluded) {
    const key = `${row.districtKey ?? ""}|${row.reason}`;
    const prior = totals.get(key);
    if (prior) prior.count++;
    else totals.set(key, { districtKey: row.districtKey, reason: row.reason, count: 1 });
  }
  return [...totals.values()].sort((a, b) =>
    (a.districtKey ?? "").localeCompare(b.districtKey ?? "") || a.reason.localeCompare(b.reason));
}
function common(value: Record<string, unknown>): value is Record<string, unknown> &
  { cutoffs: HouseLivePollingCutoffs; bindings: HouseLivePollingBindings;
    excluded: HouseLivePollingExcludedQuestion[]; excludedReasonCounts: HouseLivePollingReasonCount[] } {
  if (!object(value.cutoffs) || !keys(value.cutoffs, ["forecastAt", "pollingAt"]) ||
    !timestamp(value.cutoffs.forecastAt) || !timestamp(value.cutoffs.pollingAt) ||
    Date.parse(value.cutoffs.pollingAt) > Date.parse(value.cutoffs.forecastAt) ||
    !object(value.bindings) || !keys(value.bindings,
      ["catalogSha256", "sourcePolicySha256", "admissionSha256", "selectionSha256", "ledgerSha256"]) ||
    !Object.values(value.bindings).every(hash) ||
    !Array.isArray(value.excluded) || !value.excluded.every(excludedQuestion) ||
    !Array.isArray(value.excludedReasonCounts) ||
    !value.excludedReasonCounts.every(row => object(row) && keys(row, countKeys) &&
      (row.districtKey === null || (typeof row.districtKey === "string" && DISTRICTS.has(row.districtKey))) &&
      reason(row.reason) && count(row.count) && row.count > 0)) return false;
  return JSON.stringify(value.excludedReasonCounts) === JSON.stringify(reasonCounts(value.excluded));
}

/** Strict browser-side shape check; hashes here are identities, not authentication. */
export function parseHouseLivePollingCollection(value: unknown): HouseLivePollingCollection | null {
  if (!object(value) || !keys(value, ["schema", "cycle", "ruleset", "cutoffs", "bindings", "runs",
    "selected", "excluded", "excludedReasonCounts"]) ||
    value.schema !== "HouseLiveSitePollingDisclosureV1" || value.cycle !== 2026 ||
    value.ruleset !== HOUSE_LIVE_POLL_RULESET || !common(value) ||
    !Array.isArray(value.runs) || value.runs.length !== 16 ||
    !Array.isArray(value.selected) ||
    !value.selected.every(row => selectedQuestion(row, value.cutoffs.pollingAt))) return null;
  const seenRuns = new Set<string>(), seenHashes = new Set<string>(), seenQuestions = new Set<string>();
  let pollingOff = 0, pollingOn = 0;
  for (const row of value.runs) {
    if (!object(row) || !keys(row, ["id", "resultSha256", "polling"]) || !text(row.id) ||
      !hash(row.resultSha256) || seenRuns.has(row.id) || seenHashes.has(row.resultSha256) ||
      !["none", "standard"].includes(String(row.polling))) return null;
    seenRuns.add(row.id); seenHashes.add(row.resultSha256);
    if (row.polling === "none") pollingOff++; else pollingOn++;
  }
  if (pollingOff !== 6 || pollingOn !== 10) return null;
  for (const row of value.selected) {
    if (seenQuestions.has(row.observationId)) return null;
    seenQuestions.add(row.observationId);
  }
  return structuredClone(value) as unknown as HouseLivePollingCollection;
}

/** Strict response parser. The server separately binds the run to its native view. */
export function parseHouseLivePollingRun(value: unknown): HouseLivePollingRun | null {
  if (!object(value) || !keys(value, ["schema", "runId", "resultSha256", "ruleset", "cutoffs",
    "bindings", "modelUsesSelectedQuestions", "usedQuestions", "selectedPoolQuestionCount",
    "selectedPoolDistrictCount", "excluded", "excludedReasonCounts"]) ||
    value.schema !== "HouseLivePollingRunV1" || !text(value.runId) || !hash(value.resultSha256) ||
    value.ruleset !== HOUSE_LIVE_POLL_RULESET || !common(value) ||
    typeof value.modelUsesSelectedQuestions !== "boolean" ||
    !count(value.selectedPoolQuestionCount) || !count(value.selectedPoolDistrictCount) ||
    value.selectedPoolDistrictCount > value.selectedPoolQuestionCount ||
    value.selectedPoolDistrictCount > 435 ||
    !Array.isArray(value.usedQuestions) ||
    !value.usedQuestions.every(row => selectedQuestion(row, value.cutoffs.pollingAt))) return null;
  const used = value.usedQuestions as HouseLivePollingSelectedQuestion[];
  if (new Set(used.map(row => row.observationId)).size !== used.length) return null;
  if (value.modelUsesSelectedQuestions) {
    if (used.length !== value.selectedPoolQuestionCount ||
      new Set(used.map(row => row.raceId)).size !== value.selectedPoolDistrictCount) return null;
  } else if (used.length !== 0) return null;
  return structuredClone(value) as unknown as HouseLivePollingRun;
}

function sourceHash(pins: unknown, sha: string): boolean {
  return Array.isArray(pins) && pins.every(pin) &&
    pins.filter(row => row.sha256 === sha).length === 1;
}
function rawEndDate(value: unknown): string | null {
  if (date(value)) return value;
  if (typeof value === "string") {
    const match = /^(\d{1,2})\/(\d{1,2})\/(\d{2})$/.exec(value);
    if (match) {
      const formatted = `20${match[3]}-${match[1].padStart(2, "0")}-${match[2].padStart(2, "0")}`;
      if (date(formatted)) return formatted;
    }
  }
  return null;
}

/** Build from already authenticated native artifacts; never copy raw polling rows into the output. */
export function buildHouseLivePollingCollection(input: HouseLivePollingBuildInput): HouseLivePollingCollection {
  const catalog = parseHouseLiveCatalog(input.catalog);
  requireValue(catalog && hash(input.catalogSha256) && Array.isArray(input.results) &&
    input.results.length === 16 &&
    catalog.source_policy.sha256 === input.sourcePolicySha256,
  "complete catalog, policy, and result hashes are required");
  const digests = [input.sourcePolicySha256, input.admissionSha256,
    input.selectionSha256, input.ledgerSha256];
  requireValue(digests.every(hash), "source evidence hashes are invalid");
  const policy = input.sourcePolicy, admission = input.admission, selection = input.selection;
  requireValue(object(policy) && policy.schema === "HouseLiveSourcePolicyV1" &&
    timestamp(policy.forecast_cutoff) && timestamp(policy.polling_cutoff) &&
    typeof policy.strict_as_of === "boolean" &&
    Date.parse(policy.polling_cutoff) <= Date.parse(policy.forecast_cutoff),
  "source policy cutoffs are invalid");
  requireValue(object(admission) && admission.schema === "HouseLiveAggregateInputAdmissionV1" &&
    admission.status === "admitted_for_current_aggregate_comparison" &&
    timestamp(admission.forecast_cutoff) && sameInstant(admission.forecast_cutoff, policy.forecast_cutoff) &&
    Array.isArray(admission.input_pins) && admission.input_pins.every(pin) &&
    sourceHash(admission.input_pins, input.sourcePolicySha256) &&
    sourceHash(admission.input_pins, input.selectionSha256) &&
    sourceHash(admission.input_pins, input.ledgerSha256),
  "admission does not bind the policy, selection, and ledger");
  requireValue(object(selection) && selection.schema === "HouseLivePollSelectionV1" &&
    timestamp(selection.forecast_cutoff) && sameInstant(selection.forecast_cutoff, policy.polling_cutoff) &&
    count(selection.observations) && count(selection.races) &&
    Array.isArray(selection.outputs) && selection.outputs.every(pin) &&
    sourceHash(selection.outputs, input.ledgerSha256) &&
    Array.isArray(selection.input_pins) && selection.input_pins.every(pin) &&
    selection.input_pins.filter(row => row.path.endsWith(".py")).length === 1 &&
    selection.input_pins.some(row => row.path.endsWith(".py") &&
      row.sha256 === HOUSE_LIVE_POLL_SELECTION_IMPLEMENTATION_SHA256),
  "poll selection receipt does not bind the reviewed ruleset and ledger");
  requireValue(Array.isArray(input.ledger), "poll selection ledger is not an array");

  const selected: HouseLivePollingSelectedQuestion[] = [];
  const excluded: HouseLivePollingExcludedQuestion[] = [];
  const selectedRaw: Record<string, unknown>[] = [];
  const seenQuestionPairs = new Set<string>();
  for (const row of input.ledger) {
    requireValue(object(row) && text(row.poll_id) && text(row.question_id) &&
      text(row.source_id) && hash(row.source_sha256), "ledger question identity is invalid");
    const pairKey = `${row.poll_id}|${row.question_id}`;
    requireValue(!seenQuestionPairs.has(pairKey), "ledger repeats a poll question");
    seenQuestionPairs.add(pairKey);
    if (row.status === "selected") {
      requireValue(row.reason === "" && text(row.observation_id) && raceId(row.race_id) &&
        row.district_key === row.race_id.slice(5, 10) && text(row.pollster) &&
        date(row.start_date) && date(row.field_end) && timestamp(row.available_date) &&
        ["LV", "RV", "A"].includes(String(row.population)) &&
        ["reported", "not_reported_not_imputed"].includes(String(row.sample_size_status)) &&
        ((row.sample_size_status === "reported" && count(row.nominal_n) && row.nominal_n > 0) ||
         (row.sample_size_status === "not_reported_not_imputed" && row.nominal_n === null)) &&
        text(row.first_candidate_id) && text(row.second_candidate_id) &&
        row.first_candidate_id !== row.second_candidate_id &&
        Array.isArray(row.source_urls) && row.source_urls.includes(row.source_url),
      "selected question metadata is invalid");
      const sourceUrls = (row.source_urls as unknown[]).map(disclosureUrl);
      requireValue(sourceUrls.every((url): url is string => url !== null) &&
        disclosureUrl(row.source_url) !== null,
      "selected question source URL is unsafe");
      const item = { observationId: row.observation_id as string, raceId: row.race_id as string,
        pollster: row.pollster as string, startDate: row.start_date as string,
        fieldEnd: row.field_end as string, availableAt: row.available_date as string,
        population: row.population as "LV" | "RV" | "A",
        sampleSize: row.nominal_n as number | null, sourceId: row.source_id as string,
        sourceSha256: row.source_sha256 as string, sourceUrls };
      requireValue(selectedQuestion(item, policy.polling_cutoff as string),
        "selected question is unavailable or unsafe for disclosure");
      selected.push(item); selectedRaw.push(row);
    } else if (row.status === "excluded") {
      requireValue(reason(row.reason) && text(row.pollster) &&
        (row.district_key === null ||
          (typeof row.district_key === "string" && DISTRICTS.has(row.district_key))),
      "excluded question has an unknown reason or district");
      const item = { districtKey: row.district_key as string | null,
        // Non-district records contribute only to source-scope counts. Their
        // pollster/date metadata has no display purpose and stays offline.
        pollster: row.district_key === null ? "Record without a current district" : row.pollster as string,
        endDate: row.district_key === null ? null : rawEndDate(row.end_date),
        reason: row.reason as HouseLivePollExclusionReason };
      requireValue(excludedQuestion(item), "excluded question is unsafe for disclosure");
      excluded.push(item);
    } else fail("ledger contains an unknown selection status");
  }
  requireValue(selected.length === selection.observations &&
    new Set(selected.map(row => row.raceId)).size === selection.races &&
    new Set(selected.map(row => row.observationId)).size === selected.length,
  "selection receipt counts or selected observation identities differ");

  const selectedByRace = new Map<string, HouseLivePollingSelectedQuestion[]>();
  for (const row of selected) selectedByRace.set(row.raceId, [...(selectedByRace.get(row.raceId) ?? []), row]);
  const runById = new Map<string, { id: string; resultSha256: string; polling: "none" | "standard" }>();
  const entryById = new Map(catalog!.entries.map(entry => [entry.id, entry]));
  for (const item of input.results) {
    requireValue(object(item) && hash(item.resultSha256) && parseHouseLiveResult(item.result),
      "one native result is invalid");
    const result = item.result, entry = entryById.get(result.id);
    requireValue(entry && !runById.has(result.id) && entry.result.sha256 === item.resultSha256 &&
      houseLiveConfigurationKey(entry.configuration) === houseLiveConfigurationKey(result.configuration) &&
      sameInstant(result.forecast_cutoff, policy.forecast_cutoff as string) &&
      result.strict_as_of === policy.strict_as_of &&
      result.qualification.evidence.filter(evidence => evidence.sha256 === input.admissionSha256).length === 1 &&
      result.source_cutoffs.some(cutoff => /^(?:poll|poll selection)$/i.test(cutoff.role) &&
        sameInstant(cutoff.at, policy.polling_cutoff as string)) &&
      result.source_cutoffs.some(cutoff => /^(?:structural|nonpoll inputs)$/i.test(cutoff.role) &&
        sameInstant(cutoff.at, policy.forecast_cutoff as string)),
    "native result identity, admission, or cutoffs differ");
    const provenance = new Map(result.provenance.map(source => [source.id, source.sha256]));
    for (const race of result.races) {
      const questions = selectedByRace.get(race.race_id) ?? [];
      if (result.configuration.polling === "none") {
        requireValue(race.poll_count === 0 && race.poll_source_ids.length === 0,
          "polling-off run unexpectedly uses selected questions");
      } else {
        requireValue(race.poll_count === questions.length &&
          JSON.stringify(race.poll_source_ids) === JSON.stringify(questions.map(question => question.observationId)),
        "polling-on run question IDs differ from selected ledger order");
        for (const question of questions) {
          const raw = selectedRaw.find(row => row.observation_id === question.observationId)!;
          requireValue(provenance.get(question.sourceId) === question.sourceSha256 &&
            result.qualification.evidence.length > 0 &&
            race.candidates.some(candidate => candidate.candidate_id === raw.first_candidate_id) &&
            race.candidates.some(candidate => candidate.candidate_id === raw.second_candidate_id) &&
            (race.conditional_shares === null ||
              (race.conditional_shares.first_candidate_id === raw.first_candidate_id &&
               race.conditional_shares.second_candidate_id === raw.second_candidate_id)),
          "selected source or named pair differs from native provenance");
        }
      }
    }
    runById.set(result.id, { id: result.id, resultSha256: item.resultSha256,
      polling: result.configuration.polling });
  }
  requireValue(runById.size === catalog!.entries.length,
    "polling disclosure does not bind all catalog results");
  const collection: HouseLivePollingCollection = {
    schema: "HouseLiveSitePollingDisclosureV1", cycle: 2026, ruleset: HOUSE_LIVE_POLL_RULESET,
    cutoffs: { forecastAt: policy.forecast_cutoff as string, pollingAt: policy.polling_cutoff as string },
    bindings: { catalogSha256: input.catalogSha256, sourcePolicySha256: input.sourcePolicySha256,
      admissionSha256: input.admissionSha256, selectionSha256: input.selectionSha256,
      ledgerSha256: input.ledgerSha256 },
    runs: catalog!.entries.map(entry => runById.get(entry.id)!),
    selected, excluded, excludedReasonCounts: reasonCounts(excluded),
  };
  return parseHouseLivePollingCollection(collection) ?? fail("derived collection failed strict validation");
}

/** Run binding is by both ID and authenticated result digest. */
export function projectHouseLivePollingRun(collection: HouseLivePollingCollection, runId: string,
  resultSha256: string): HouseLivePollingRun | null {
  const parsed = parseHouseLivePollingCollection(collection);
  if (!parsed) return null;
  const run = parsed.runs.find(row => row.id === runId && row.resultSha256 === resultSha256);
  if (!run) return null;
  const uses = run.polling === "standard";
  const projected: HouseLivePollingRun = {
    schema: "HouseLivePollingRunV1", runId, resultSha256, ruleset: parsed.ruleset,
    cutoffs: { ...parsed.cutoffs }, bindings: { ...parsed.bindings },
    modelUsesSelectedQuestions: uses,
    usedQuestions: uses ? parsed.selected.map(row => ({ ...row, sourceUrls: [...row.sourceUrls] })) : [],
    selectedPoolQuestionCount: parsed.selected.length,
    selectedPoolDistrictCount: new Set(parsed.selected.map(row => row.raceId)).size,
    excluded: parsed.excluded.map(row => ({ ...row })),
    excludedReasonCounts: parsed.excludedReasonCounts.map(row => ({ ...row })),
  };
  return parseHouseLivePollingRun(projected);
}
