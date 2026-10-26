/** Structural contract for saved 2026 House forecasts. This does not authenticate pins. */
export type HouseLiveParty = "D" | "R" | "O";
export type HouseLiveBase =
  | "fundamentals" | "recent_calendar_history" | "house_history_refreshed_v1"
  | "house_presidential_slope_v1" | "house_relative_lean_v1" | "house_relative_context_v1";
export type HouseLiveUncertainty = "standard" | "cycle-low" | "cycle-high" | "error-low" | "error-high";
export interface HouseLiveConfiguration {
  base: HouseLiveBase;
  polling: "none" | "standard";
  uncertainty: HouseLiveUncertainty;
}
export interface HouseLivePin { path: string; sha256: string }
export interface HouseLiveSource {
  id: string; label: string; url: string; retrieved_at: string; sha256: string; role: string;
  capture_kind?: "received" | "rendered_primary_source";
  limitations?: string[];
}
export interface HouseLiveCandidate {
  candidate_id: string; name: string; party_label: string | null;
  aggregation_party: HouseLiveParty; ballot_status: "printed" | "declared_writein";
  incumbent: boolean; source_ids: string[];
}
export interface HouseLiveOutcome {
  outcome_id: HouseLiveParty; label: string; kind: "candidate" | "group";
  party: HouseLiveParty; member_candidate_ids: string[]; p_win: number;
}
export interface HouseLiveConditionalPair {
  first_candidate_id: string; second_candidate_id: string;
  denominator: "Share of votes for the two named principal contenders";
  mean: number;
  quantiles: { p025: number; p05: number; p10: number; p50: number; p90: number; p95: number; p975: number };
  p_first_exceeds_half: number; p_margin_within_2pp: number;
  p_first_margin_at_least_10pp: number; p_second_margin_at_least_10pp: number;
  qualifications: string[];
}
export interface HouseLiveRace {
  race_id: string; district_key: string; state: string; district_number: number;
  map_vintage: "house_election_map_2026";
  rule_family: "plurality" | "ranked_choice" | "majority_runoff" | "top_two_plurality";
  method: string; candidates: HouseLiveCandidate[]; outcomes: HouseLiveOutcome[];
  conditional_shares: HouseLiveConditionalPair | null;
  poll_count: number; poll_source_ids: string[]; source_ids: string[]; qualifications: string[];
}
export interface HouseLiveSeatDistribution {
  expected: number;
  quantiles: { p05: number; p10: number; p50: number; p90: number; p95: number };
  pmf: number[];
}
export interface HouseLiveChamber {
  total_seats: 435; threshold: 218;
  retained_seats: Record<HouseLiveParty, 0>;
  scenario_id: "party_majority_others_unassigned"; description: string;
  seats: Record<HouseLiveParty, HouseLiveSeatDistribution>;
  majority: { D: number; R: number; unresolved: number };
  precision: { passed: true; max_seat_mean_mcse: number; max_control_mcse: number; evidence: HouseLivePin };
  joint_draws_sha256: string;
}
export interface HouseLiveResult {
  schema: "HouseLiveForecastResultV1"; id: string; cycle: 2026;
  configuration: HouseLiveConfiguration;
  issued_at: string; forecast_cutoff: string; election_date: "2026-11-03";
  source_cutoffs: { role: string; at: string }[]; strict_as_of: boolean;
  map_vintage: "house_election_map_2026";
  geography: HouseLivePin & { feature_count: 435; map_vintage: "house_election_map_2026"; source_receipt: HouseLivePin };
  races: HouseLiveRace[];
  joint_draws: HouseLivePin & { count: number; chain_count: number; ordered_race_ids: string[];
    columns: ["D", "R", "O"]; winner_encoding: { D: 0; R: 1; O: 2 }; identity_semantics: string };
  chamber: HouseLiveChamber | null; tipping: null;
  qualification: { input_status: string; numerical_status: string; publication_status: string;
    forecast_skill: string; limitations: string[]; evidence: HouseLivePin[] };
  provenance: HouseLiveSource[];
  /** Draft interpretation pending a versioned backend schema. */
  historical_evidence: { label: string; url: string; qualification: string }[];
}

const PARTIES = ["D", "R", "O"] as const;
const BASES = ["fundamentals", "recent_calendar_history", "house_history_refreshed_v1", "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1"] as const;
const UNCERTAINTIES = ["standard", "cycle-low", "cycle-high", "error-low", "error-high"] as const;
const SEATS: Record<string, number> = {
  AL: 7, AK: 1, AZ: 9, AR: 4, CA: 52, CO: 8, CT: 5, DE: 1, FL: 28,
  GA: 14, HI: 2, ID: 2, IL: 17, IN: 9, IA: 4, KS: 4, KY: 6, LA: 6,
  ME: 2, MD: 8, MA: 9, MI: 13, MN: 8, MS: 4, MO: 8, MT: 2, NE: 3,
  NV: 4, NH: 2, NJ: 12, NM: 3, NY: 26, NC: 14, ND: 1, OH: 15, OK: 5,
  OR: 6, PA: 17, RI: 2, SC: 7, SD: 1, TN: 9, TX: 38, UT: 4, VT: 1,
  VA: 11, WA: 10, WV: 2, WI: 8, WY: 1,
};
/** Canonical current-district roster shared with the browser publication boundary. */
export const HOUSE_LIVE_EXPECTED_DISTRICT_KEYS: readonly string[] = Object.freeze(
  Object.entries(SEATS).flatMap(([state, count]) => Array.from({ length: count }, (_, index) =>
    `${state}-${String(count === 1 ? 0 : index + 1).padStart(2, "0")}`)),
);
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && v.length > 0 && v.length <= 4000 && v === v.trim() && !/[\u0000-\u001f\u007f]/.test(v);
const id = (v: unknown): v is string => text(v) && /^[A-Za-z0-9][A-Za-z0-9:._-]{0,199}$/.test(v);
const hash = (v: unknown): v is string => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
const fraction = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
const nonnegative = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;
const integer = (v: unknown): v is number => Number.isSafeInteger(v) && (v as number) >= 0;
const list = (v: unknown, minimum = 0): v is string[] => Array.isArray(v) && v.length >= minimum && v.every(text);
const unique = (v: readonly unknown[]) => new Set(v).size === v.length;
const near = (a: number, b: number, tolerance = 1e-8) => Math.abs(a - b) <= tolerance;
const sum = (v: readonly number[]) => v.reduce((a, b) => a + b, 0);
const exactKeys = (v: Record<string, unknown>, names: readonly string[]) => Object.keys(v).length === names.length && names.every(name => name in v);
const timestamp = (v: unknown): v is string => {
  if (typeof v !== "string") return false;
  const parts = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|\+00:00)$/.exec(v);
  if (!parts) return false;
  const [, yearText, monthText, dayText, hourText, minuteText, secondText] = parts;
  const year = Number(yearText), month = Number(monthText), day = Number(dayText);
  const hour = Number(hourText), minute = Number(minuteText), second = Number(secondText);
  if (month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59) return false;
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const monthDays = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return day >= 1 && day <= monthDays[month - 1] && Number.isFinite(Date.parse(v));
};
export const houseLivePin = (v: unknown): v is HouseLivePin => object(v) &&
  text(v.path) && !v.path.startsWith("/") && !v.path.split("/").includes("..") &&
  !v.path.includes("\\") && !/^[a-z]+:/i.test(v.path) && hash(v.sha256);

export function parseHouseLiveConfiguration(v: unknown): HouseLiveConfiguration | null {
  if (!object(v) || !exactKeys(v, ["base", "polling", "uncertainty"]) ||
    !BASES.includes(v.base as HouseLiveBase) || !["none", "standard"].includes(String(v.polling)) ||
    !UNCERTAINTIES.includes(v.uncertainty as HouseLiveUncertainty)) return null;
  if (v.uncertainty !== "standard" && (v.base !== "fundamentals" || v.polling !== "standard")) return null;
  return v as unknown as HouseLiveConfiguration;
}
export function houseLiveConfigurationKey(v: HouseLiveConfiguration): string {
  return `${v.base}|${v.polling}|${v.uncertainty}`;
}

function validQuantiles(v: unknown, keys: readonly string[], maximum: number): boolean {
  if (!object(v) || !exactKeys(v, keys)) return false;
  const values = keys.map(key => v[key]);
  return values.every(value => nonnegative(value) && value <= maximum) &&
    values.every((value, index) => index === 0 || (values[index - 1] as number) <= (value as number));
}
function conditionalPair(v: unknown, candidates: Map<string, HouseLiveCandidate>): v is HouseLiveConditionalPair {
  if (!object(v) || !id(v.first_candidate_id) || !id(v.second_candidate_id) ||
    v.first_candidate_id === v.second_candidate_id || !candidates.has(v.first_candidate_id) ||
    !candidates.has(v.second_candidate_id) ||
    v.denominator !== "Share of votes for the two named principal contenders" || !fraction(v.mean) ||
    !validQuantiles(v.quantiles, ["p025", "p05", "p10", "p50", "p90", "p95", "p975"], 1) ||
    ![v.p_first_exceeds_half, v.p_margin_within_2pp, v.p_first_margin_at_least_10pp,
      v.p_second_margin_at_least_10pp].every(fraction) || !list(v.qualifications, 1)) return false;
  return (v.p_first_margin_at_least_10pp as number) + (v.p_second_margin_at_least_10pp as number) +
    (v.p_margin_within_2pp as number) <= 1 + 1e-8 &&
    (v.p_first_margin_at_least_10pp as number) <= (v.p_first_exceeds_half as number) + 1e-8 &&
    (v.p_second_margin_at_least_10pp as number) <= 1 - (v.p_first_exceeds_half as number) + 1e-8;
}
function race(v: unknown, provenanceIds: Set<string>, polling: HouseLiveConfiguration["polling"]): v is HouseLiveRace {
  if (!object(v) || ["observed", "observed_outcome", "actual_winner", "actual_result"].some(key => key in v) ||
    !text(v.district_key) || !/^([A-Z]{2})-(\d{2})$/.test(v.district_key) ||
    !id(v.race_id) || v.race_id !== `2026:${v.district_key}:house:regular` ||
    v.state !== v.district_key.slice(0, 2) || !integer(v.district_number) ||
    v.district_number !== Number(v.district_key.slice(3)) ||
    v.map_vintage !== "house_election_map_2026" ||
    !["plurality", "ranked_choice", "majority_runoff", "top_two_plurality"].includes(String(v.rule_family)) ||
    !text(v.method) || !Array.isArray(v.candidates) || !v.candidates.length ||
    !Array.isArray(v.outcomes) || !v.outcomes.length || v.outcomes.length > 3 ||
    !integer(v.poll_count) || !list(v.poll_source_ids) || !unique(v.poll_source_ids) ||
    v.poll_count !== v.poll_source_ids.length ||
    !list(v.source_ids, 1) || !unique(v.source_ids) || !v.source_ids.every((x: string) => provenanceIds.has(x)) ||
    !list(v.qualifications, 1) || (polling === "none" && (v.poll_count !== 0 || v.poll_source_ids.length !== 0))) return false;
  const candidates = new Map<string, HouseLiveCandidate>();
  for (const candidate of v.candidates) {
    if (!object(candidate) || !id(candidate.candidate_id) || candidates.has(candidate.candidate_id) ||
      !text(candidate.name) || !PARTIES.includes(candidate.aggregation_party as HouseLiveParty) ||
      !["printed", "declared_writein"].includes(String(candidate.ballot_status)) ||
      typeof candidate.incumbent !== "boolean" || !list(candidate.source_ids, 1) ||
      !unique(candidate.source_ids) || !candidate.source_ids.every((x: string) => provenanceIds.has(x)) ||
      !(text(candidate.party_label) || (candidate.party_label === null && candidate.ballot_status === "declared_writein"))) return false;
    if (v.district_key === "TX-23" &&
      /\b(?:tony|antonio)[- _]+gonzales\b/i.test(`${candidate.name} ${candidate.candidate_id}`)) return false;
    candidates.set(candidate.candidate_id, candidate as unknown as HouseLiveCandidate);
  }
  const parties = new Set<HouseLiveParty>(), covered = new Set<string>();
  let mass = 0;
  for (const outcome of v.outcomes) {
    if (!object(outcome) || ["share", "conditional_share", "observed"].some(key => key in outcome) ||
      !PARTIES.includes(outcome.outcome_id as HouseLiveParty) ||
      outcome.party !== outcome.outcome_id || parties.has(outcome.party as HouseLiveParty) ||
      !text(outcome.label) || !["candidate", "group"].includes(String(outcome.kind)) ||
      !list(outcome.member_candidate_ids, 1) || !unique(outcome.member_candidate_ids) || !fraction(outcome.p_win) ||
      (outcome.kind === "candidate" && (outcome.member_candidate_ids.length !== 1 ||
        outcome.label !== candidates.get(outcome.member_candidate_ids[0])?.name)) ||
      (outcome.kind === "group" && outcome.member_candidate_ids.length === 1 &&
        outcome.label === candidates.get(outcome.member_candidate_ids[0])?.name)) return false;
    for (const memberId of outcome.member_candidate_ids) {
      if (covered.has(memberId) || candidates.get(memberId)?.aggregation_party !== outcome.party) return false;
      covered.add(memberId);
    }
    parties.add(outcome.party as HouseLiveParty);
    mass += outcome.p_win;
  }
  if (covered.size !== candidates.size || !near(mass, 1) ||
    !PARTIES.every(party => parties.has(party) === [...candidates.values()].some(candidate => candidate.aggregation_party === party))) return false;
  return v.conditional_shares === null || conditionalPair(v.conditional_shares, candidates);
}
function seatDistribution(v: unknown): v is HouseLiveSeatDistribution {
  if (!object(v) || !nonnegative(v.expected) || v.expected > 435 ||
    !validQuantiles(v.quantiles, ["p05", "p10", "p50", "p90", "p95"], 435) ||
    !Array.isArray(v.pmf) || v.pmf.length !== 436 || !v.pmf.every(fraction) ||
    !near(sum(v.pmf), 1)) return false;
  return near(v.expected, sum(v.pmf.map((p: number, seats: number) => p * seats)), 1e-6);
}
function chamber(v: unknown, jointHash: string): v is HouseLiveChamber {
  if (!object(v) || v.total_seats !== 435 || v.threshold !== 218 ||
    !object(v.retained_seats) || !exactKeys(v.retained_seats, PARTIES) ||
    !PARTIES.every(party => (v.retained_seats as Record<string, unknown>)[party] === 0) ||
    v.scenario_id !== "party_majority_others_unassigned" || !text(v.description) ||
    !object(v.seats) || !exactKeys(v.seats, PARTIES) ||
    !PARTIES.every(party => seatDistribution((v.seats as Record<string, unknown>)[party])) ||
    !object(v.majority) || !exactKeys(v.majority, ["D", "R", "unresolved"]) ||
    ![v.majority.D, v.majority.R, v.majority.unresolved].every(fraction) ||
    !near(sum([v.majority.D, v.majority.R, v.majority.unresolved] as number[]), 1) ||
    !object(v.precision) || v.precision.passed !== true ||
    !nonnegative(v.precision.max_seat_mean_mcse) || !nonnegative(v.precision.max_control_mcse) ||
    !houseLivePin(v.precision.evidence) || v.joint_draws_sha256 !== jointHash) return false;
  const seats = v.seats as unknown as Record<HouseLiveParty, HouseLiveSeatDistribution>;
  return near(sum(PARTIES.map(party => seats[party].expected)), 435, 1e-6) &&
    near(v.majority.D as number, sum(seats.D.pmf.slice(218))) &&
    near(v.majority.R as number, sum(seats.R.pmf.slice(218))) &&
    near(v.majority.unresolved as number, 1 - (v.majority.D as number) - (v.majority.R as number));
}

/** Shape and internal semantics only; reviewed strings and SHA-shaped pins do not publish a run. */
export function parseHouseLiveResult(value: unknown): HouseLiveResult | null {
  if (!object(value) || value.schema !== "HouseLiveForecastResultV1" || !id(value.id) ||
    value.cycle !== 2026 || !parseHouseLiveConfiguration(value.configuration) ||
    !timestamp(value.issued_at) || !timestamp(value.forecast_cutoff) ||
    Date.parse(value.forecast_cutoff) > Date.parse(value.issued_at) ||
    value.election_date !== "2026-11-03" || !Array.isArray(value.source_cutoffs) ||
    !value.source_cutoffs.length || typeof value.strict_as_of !== "boolean" ||
    value.map_vintage !== "house_election_map_2026" || !object(value.geography) ||
    !houseLivePin(value.geography) || value.geography.feature_count !== 435 ||
    value.geography.map_vintage !== value.map_vintage || !houseLivePin(value.geography.source_receipt) ||
    !Array.isArray(value.races) || value.races.length !== 435 || !object(value.joint_draws) ||
    !houseLivePin(value.joint_draws) || !integer(value.joint_draws.count) || value.joint_draws.count === 0 ||
    !integer(value.joint_draws.chain_count) || value.joint_draws.chain_count === 0 ||
    (value.joint_draws.count as number) % (value.joint_draws.chain_count as number) !== 0 ||
    !Array.isArray(value.joint_draws.ordered_race_ids) || value.joint_draws.ordered_race_ids.length !== 435 ||
    JSON.stringify(value.joint_draws.columns) !== '["D","R","O"]' ||
    !object(value.joint_draws.winner_encoding) ||
    !exactKeys(value.joint_draws.winner_encoding, PARTIES) ||
    value.joint_draws.winner_encoding.D !== 0 || value.joint_draws.winner_encoding.R !== 1 ||
    value.joint_draws.winner_encoding.O !== 2 || !text(value.joint_draws.identity_semantics) ||
    value.tipping !== null || !object(value.qualification) ||
    !text(value.qualification.input_status) || !text(value.qualification.numerical_status) ||
    !text(value.qualification.publication_status) || !text(value.qualification.forecast_skill) ||
    !list(value.qualification.limitations, 1) || !Array.isArray(value.qualification.evidence) ||
    !value.qualification.evidence.length || !value.qualification.evidence.every(houseLivePin) ||
    !Array.isArray(value.provenance) || !value.provenance.length ||
    !Array.isArray(value.historical_evidence) || !value.historical_evidence.length ||
    !value.historical_evidence.every(row => object(row) && text(row.label) && text(row.qualification) &&
      text(row.url) && /^https?:\/\//i.test(row.url))) return null;
  const cutoff = Date.parse(value.forecast_cutoff);
  for (const row of value.source_cutoffs) {
    if (!object(row) || !text(row.role) || !timestamp(row.at) || Date.parse(row.at) > cutoff) return null;
  }
  const provenanceIds = new Set<string>();
  for (const row of value.provenance) {
    if (!object(row) || !id(row.id) || provenanceIds.has(row.id) || !text(row.label) ||
      !text(row.url) || !/^https?:\/\//i.test(row.url) || !timestamp(row.retrieved_at) ||
      Date.parse(row.retrieved_at) > cutoff || !hash(row.sha256) || !text(row.role) ||
      (row.capture_kind !== undefined && !["received", "rendered_primary_source"].includes(String(row.capture_kind))) ||
      (row.limitations !== undefined && (!list(row.limitations) || row.limitations.length > 20))) return null;
    provenanceIds.add(row.id);
  }
  const seen = new Set<string>();
  for (let index = 0; index < value.races.length; index++) {
    const row = value.races[index];
    if (!race(row, provenanceIds, (value.configuration as HouseLiveConfiguration).polling) ||
      seen.has(row.district_key) || value.joint_draws.ordered_race_ids[index] !== row.race_id) return null;
    seen.add(row.district_key);
  }
  if (seen.size !== HOUSE_LIVE_EXPECTED_DISTRICT_KEYS.length ||
    HOUSE_LIVE_EXPECTED_DISTRICT_KEYS.some(key => !seen.has(key))) return null;
  if (value.chamber === null && !value.qualification.limitations.some((note: string) =>
    /(?:chamber|control|seat total)/i.test(note) &&
    /(?:unavailable|unqualified|failed|withheld|not published|not qualified)/i.test(note))) return null;
  if (value.chamber !== null && !chamber(value.chamber, value.joint_draws.sha256)) return null;
  return value as unknown as HouseLiveResult;
}
