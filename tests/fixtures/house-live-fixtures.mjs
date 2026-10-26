import { createHash } from "node:crypto";

// Synthetic contract fixtures only. Names, probabilities, source pins, and draw
// identities below are invented and must never be placed in published data.
const STATE_SEATS = [
  ["AL", 7], ["AK", 1], ["AZ", 9], ["AR", 4], ["CA", 52], ["CO", 8],
  ["CT", 5], ["DE", 1], ["FL", 28], ["GA", 14], ["HI", 2], ["ID", 2],
  ["IL", 17], ["IN", 9], ["IA", 4], ["KS", 4], ["KY", 6], ["LA", 6],
  ["ME", 2], ["MD", 8], ["MA", 9], ["MI", 13], ["MN", 8], ["MS", 4],
  ["MO", 8], ["MT", 2], ["NE", 3], ["NV", 4], ["NH", 2], ["NJ", 12],
  ["NM", 3], ["NY", 26], ["NC", 14], ["ND", 1], ["OH", 15], ["OK", 5],
  ["OR", 6], ["PA", 17], ["RI", 2], ["SC", 7], ["SD", 1], ["TN", 9],
  ["TX", 38], ["UT", 4], ["VT", 1], ["VA", 11], ["WA", 10], ["WV", 2],
  ["WI", 8], ["WY", 1],
];

export const syntheticCurrentDistricts = STATE_SEATS.flatMap(([state, count]) =>
  Array.from({ length: count }, (_, index) => ({
    state,
    district_number: count === 1 ? 0 : index + 1,
    district_key: `${state}-${String(count === 1 ? 0 : index + 1).padStart(2, "0")}`,
  })),
);

export const SYNTHETIC_SHA = createHash("sha256")
  .update("invented fixture pin; not a production artifact")
  .digest("hex");

export const SYNTHETIC_CONFIGURATIONS = [
  ...[
    "fundamentals",
    "recent_calendar_history",
    "house_history_refreshed_v1",
    "house_presidential_slope_v1",
    "house_relative_lean_v1",
    "house_relative_context_v1",
  ].flatMap((base) => ["none", "standard"].map((polling) => ({
    base, polling, uncertainty: "standard",
  }))),
  ...["standard", "cycle-low", "cycle-high", "error-low", "error-high"].slice(1)
    .map((uncertainty) => ({ base: "fundamentals", polling: "standard", uncertainty })),
];

const configId = (configuration) =>
  configuration.base === "fundamentals" && configuration.polling === "standard" && configuration.uncertainty === "standard"
    ? "house-current-polling"
    : `house-current-${configuration.base}-${configuration.polling}-${configuration.uncertainty}`.replaceAll("_", "-");
const iso = "2026-10-20T00:00:00.000Z";

function candidate(id, party, extra = {}) {
  return {
    candidate_id: id,
    name: `Invented ${id}`,
    party_label: party === "O" ? "Independent" : party,
    aggregation_party: party,
    ballot_status: "printed",
    incumbent: false,
    source_ids: ["synthetic-roster-source"],
    ...extra,
  };
}

const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.ceil(q * values.length) - 1] ?? 0;
function seatDistribution(values) {
  const pmf = Array(436).fill(0);
  for (const value of values) pmf[value] += 1 / values.length;
  return {
    expected: values.reduce((sum, value) => sum + value, 0) / values.length,
    quantiles: {
      p05: quantile(values, 0.05), p10: quantile(values, 0.10),
      p50: quantile(values, 0.50), p90: quantile(values, 0.90),
      p95: quantile(values, 0.95),
    },
    pmf,
  };
}

function raceFor(district, index) {
  const raceId = `2026:${district.district_key}:house:regular`;
  const common = {
    race_id: raceId,
    district_key: district.district_key,
    state: district.state,
    district_number: district.district_number,
    map_vintage: "house_election_map_2026",
    rule_family: "plurality",
    method: "synthetic four-row categorical draw fixture",
    poll_count: 0,
    poll_source_ids: [],
    source_ids: ["synthetic-roster-source"],
    qualifications: ["Invented data for parser tests only."],
    conditional_shares: null,
  };
  if (index === 0) {
    return {
      ...common,
      candidates: [candidate(`${district.district_key}-D1`, "D"), candidate(`${district.district_key}-D2`, "D"), candidate(`${district.district_key}-R1`, "R")],
      outcomes: [
        { outcome_id: "D", label: "Democratic group", kind: "group", party: "D", member_candidate_ids: [`${district.district_key}-D1`, `${district.district_key}-D2`], p_win: 0.5 },
        { outcome_id: "R", label: `Invented ${district.district_key}-R1`, kind: "candidate", party: "R", member_candidate_ids: [`${district.district_key}-R1`], p_win: 0.5 },
      ],
    };
  }
  if (index === 1) {
    return {
      ...common,
      candidates: [candidate(`${district.district_key}-D1`, "D"), candidate(`${district.district_key}-R1`, "R"), candidate(`${district.district_key}-I1`, "O", { party_label: "Independent" })],
      outcomes: [
        { outcome_id: "D", label: `Invented ${district.district_key}-D1`, kind: "candidate", party: "D", member_candidate_ids: [`${district.district_key}-D1`], p_win: 0.25 },
        { outcome_id: "R", label: `Invented ${district.district_key}-R1`, kind: "candidate", party: "R", member_candidate_ids: [`${district.district_key}-R1`], p_win: 0.25 },
        { outcome_id: "O", label: `Invented ${district.district_key}-I1`, kind: "candidate", party: "O", member_candidate_ids: [`${district.district_key}-I1`], p_win: 0.5 },
      ],
      conditional_shares: null,
    };
  }
  if (index === 2) {
    return {
      ...common,
      candidates: [candidate(`${district.district_key}-D1`, "D")],
      outcomes: [{ outcome_id: "D", label: `Invented ${district.district_key}-D1`, kind: "candidate", party: "D", member_candidate_ids: [`${district.district_key}-D1`], p_win: 1 }],
    };
  }
  const d = candidate(`${district.district_key}-D1`, "D");
  const r = candidate(`${district.district_key}-R1`, "R");
  return {
    ...common,
    candidates: [d, r],
    outcomes: [
      { outcome_id: "D", label: d.name, kind: "candidate", party: "D", member_candidate_ids: [d.candidate_id], p_win: 0.5 },
      { outcome_id: "R", label: r.name, kind: "candidate", party: "R", member_candidate_ids: [r.candidate_id], p_win: 0.5 },
    ],
    conditional_shares: {
      first_candidate_id: d.candidate_id,
      second_candidate_id: r.candidate_id,
      denominator: "Share of votes for the two named principal contenders",
      mean: 0.5,
      quantiles: { p025: 0.2, p05: 0.3, p10: 0.35, p50: 0.5, p90: 0.65, p95: 0.7, p975: 0.8 },
      p_first_exceeds_half: 0.45,
      p_margin_within_2pp: 0.1,
      p_first_margin_at_least_10pp: 0.2,
      p_second_margin_at_least_10pp: 0.3,
      qualifications: ["Synthetic conditional pair; distinct from group win probabilities."],
    },
  };
}

function syntheticChamber() {
  // Rows are complete joint outcomes in roster order. D and R each control in
  // two of four rows; the independent/O race remains in the O seat PMF.
  const d = [435, 434, 1, 1];
  const r = [0, 0, 434, 433];
  const o = [0, 1, 0, 1];
  return {
    total_seats: 435,
    threshold: 218,
    retained_seats: { D: 0, R: 0, O: 0 },
    scenario_id: "party_majority_others_unassigned",
    description: "Invented fixture: 218+ D or R seats is a majority; other seats do not organize either party.",
    seats: { D: seatDistribution(d), R: seatDistribution(r), O: seatDistribution(o) },
    majority: { D: 0.5, R: 0.5, unresolved: 0 },
    precision: { passed: true, max_seat_mean_mcse: 0, max_control_mcse: 0, evidence: { path: "synthetic/precision.json", sha256: SYNTHETIC_SHA } },
    joint_draws_sha256: SYNTHETIC_SHA,
  };
}

export function makeHouseLiveResult(configuration = SYNTHETIC_CONFIGURATIONS[1]) {
  const races = syntheticCurrentDistricts.map(raceFor);
  return {
    schema: "HouseLiveForecastResultV1",
    id: configId(configuration),
    cycle: 2026,
    configuration: structuredClone(configuration),
    issued_at: "2026-10-21T12:00:00.000Z",
    forecast_cutoff: iso,
    election_date: "2026-11-03",
    source_cutoffs: [
      { role: "structural", at: "2026-10-19T00:00:00.000Z" },
      { role: "poll", at: "2026-10-20T00:00:00.000Z" },
    ],
    strict_as_of: true,
    map_vintage: "house_election_map_2026",
    geography: { path: "synthetic/geography.json", sha256: SYNTHETIC_SHA, feature_count: 435, map_vintage: "house_election_map_2026", source_receipt: { path: "synthetic/map-receipt.json", sha256: SYNTHETIC_SHA } },
    races,
    joint_draws: {
      path: "synthetic/joint-draws.json",
      sha256: SYNTHETIC_SHA,
      count: 4,
      chain_count: 2,
      ordered_race_ids: races.map((race) => race.race_id),
      columns: ["D", "R", "O"],
      winner_encoding: { D: 0, R: 1, O: 2 },
      identity_semantics: "Rows contain one exhaustive winner group for every ordered race; each row totals 435 seats.",
    },
    chamber: syntheticChamber(),
    tipping: null,
    qualification: {
      input_status: "synthetic",
      numerical_status: "synthetic",
      publication_status: "not_for_publication",
      forecast_skill: "not_established",
      limitations: ["Invented fixture; no 2026 forecast or empirical skill claim."],
      evidence: [{ path: "synthetic/qualification.json", sha256: SYNTHETIC_SHA }],
    },
    provenance: [{ id: "synthetic-roster-source", label: "Invented fixture source", url: "https://example.invalid/synthetic", retrieved_at: "2026-10-19T00:00:00.000Z", sha256: SYNTHETIC_SHA, role: "candidate-roster" }],
    historical_evidence: [{ label: "Unchanged historical benchmark reference", url: "https://example.invalid/synthetic/historical-evidence", qualification: "Synthetic link only; does not qualify this current result." }],
  };
}

export function makeHouseLiveCatalog() {
  const entries = SYNTHETIC_CONFIGURATIONS.map((configuration) => ({
    id: configId(configuration),
    label: `Invented ${configuration.base} ${configuration.polling} ${configuration.uncertainty}`,
    configuration: structuredClone(configuration),
    result: { path: `synthetic/results/${configId(configuration)}.json`, sha256: SYNTHETIC_SHA },
  }));
  return {
    schema: "HouseLiveForecastCatalogV1",
    cycle: 2026,
    default_id: entries[1].id,
    entries,
    source_policy: { path: "synthetic/source-policy.json", sha256: SYNTHETIC_SHA },
    qualification: { path: "synthetic/catalog-qualification.json", sha256: SYNTHETIC_SHA },
  };
}

export const syntheticCurrentDistrictCount = syntheticCurrentDistricts.length;
