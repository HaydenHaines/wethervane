/** Private explanatory excerpts. Authentication and source permission belong to the offline importer. */
import type { HouseLiveParty, HouseLiveRace, HouseLiveResult } from "./house-live-result.ts";

export type HouseContextParty = HouseLiveParty;
export interface HouseContextEdition {
  publicationSha256: string;
  catalogSha256: string;
  readyPacketSha256: string;
  runId: string;
  resultSha256: string;
  issuedAt: string;
  cutoffAt: string;
  mapVintage: string;
  geographySha256: string;
}
export interface HouseContextPair {
  firstCandidateId: string;
  secondCandidateId: string;
  denominator: string;
  firstShareMean: number;
}
export interface HouseContextRaceSide {
  pollCount: number;
  candidates: { id: string; name: string; partyLabel: string | null;
    aggregationParty: HouseContextParty; ballotStatus: string }[];
  winnerGroups: Record<HouseContextParty, { label: string; kind: "candidate" | "group";
    memberCandidateIds: string[]; probability: number } | null>;
  pair: HouseContextPair | null;
}
export interface HouseContextCase {
  districtKey: string;
  raceId: string;
  ruleFamily: string;
  cook: { rating: string; favoredParty: "D" | "R" | null; scale: "ordinal_no_probability" };
  current: HouseContextRaceSide;
  baseline: HouseContextRaceSide;
  candidateRecordsChanged: boolean;
  principalPairChanged: boolean;
  winnerGroupDelta: Record<HouseContextParty, number | null>;
  winnerGroupDeltaReason: string | null;
  pairComparison: { status: "comparable"; firstShareMeanDelta: number } |
    { status: "incompatible"; reason: string };
  baselineSourceFinding: string;
  interpretation: string;
}
export interface HouseContextComparisonCatalog {
  schema: "HouseContextComparisonCatalogV1";
  publicationQualified: false;
  scope: "private_selected_excerpts_only";
  selfComparison: boolean;
  cookSnapshot: string;
  sourceUrl: string;
  sourcePins: {
    handoffSha256: string; handoffReviewSha256: string; comparisonReceiptSha256: string;
    operatorApprovalSha256: string; sourcePolicyApprovalSha256: string;
  };
  currentEdition: HouseContextEdition;
  baselineEdition: HouseContextEdition;
  cases: HouseContextCase[];
  interpretation: string;
}

const PARTIES = ["D", "R", "O"] as const;
const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);
const near = (a: number, b: number, tolerance = 1.1e-6) => Math.abs(a - b) <= tolerance;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
function need(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
const text = (v: unknown): v is string => typeof v === "string" && !!v.trim() && v === v.trim() && v.length <= 4000;
const hash = (v: unknown): v is string => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);

function side(race: HouseLiveRace): HouseContextRaceSide {
  const groups = Object.fromEntries(PARTIES.map(party => {
    const outcome = race.outcomes.find(item => item.party === party);
    return [party, outcome ? { label: outcome.label, kind: outcome.kind,
      memberCandidateIds: [...outcome.member_candidate_ids], probability: outcome.p_win } : null];
  })) as HouseContextRaceSide["winnerGroups"];
  const p = race.conditional_shares;
  return { pollCount: race.poll_count,
    candidates: race.candidates.map(c => ({ id: c.candidate_id, name: c.name, partyLabel: c.party_label,
      aggregationParty: c.aggregation_party, ballotStatus: c.ballot_status })),
    winnerGroups: groups,
    pair: p ? { firstCandidateId: p.first_candidate_id, secondCandidateId: p.second_candidate_id,
      denominator: p.denominator, firstShareMean: p.mean } : null };
}

/** Refuses to create a delta when candidate support or the relevant grouped event changed. */
export function projectHouseContextComparisonCase(
  raw: Record<string, unknown>, current: HouseLiveResult, baseline: HouseLiveResult,
): HouseContextCase {
  need(text(raw.district_key) && /^([A-Z]{2})-\d{2}$/.test(raw.district_key), "Invalid selected district");
  const a = current.races.find(r => r.district_key === raw.district_key);
  const b = baseline.races.find(r => r.district_key === raw.district_key);
  need(a && b && a.race_id === b.race_id && raw.race_id === a.race_id &&
    a.map_vintage === b.map_vintage && a.rule_family === b.rule_family && raw.rule_family === a.rule_family,
  "Selected race, map, or counting rule differs");
  need(raw.current_model_id === current.id &&
    (raw.baseline_model_id === undefined || raw.baseline_model_id === baseline.id) &&
    raw.forecast_issued_at === current.issued_at && raw.forecast_cutoff === current.forecast_cutoff &&
    raw.baseline_issued_at === baseline.issued_at, "Selected forecast edition differs");
  need(text(raw.cook_rating) &&
    (/^(Solid|Likely|Lean) [DR]$/.test(raw.cook_rating) ? raw.cook_favored_party === raw.cook_rating.at(-1) :
      raw.cook_rating === "Toss Up" && raw.cook_favored_party === null) &&
    text(raw.baseline_source_finding) && text(raw.interpretation), "Missing ordinal source or case interpretation");
  need(Number.isSafeInteger(raw.poll_count) && raw.poll_count === a.poll_count &&
    Number.isSafeInteger(raw.baseline_poll_count) && raw.baseline_poll_count === b.poll_count,
  "Selected poll counts differ");
  const currentSide = side(a), baselineSide = side(b);
  const candidateChanged = !same(a.candidates, b.candidates);
  const pairIdentity = (race: HouseLiveRace) => race.conditional_shares ?
    [race.conditional_shares.first_candidate_id, race.conditional_shares.second_candidate_id,
      race.conditional_shares.denominator] : [null, null, null];
  const pairChanged = !same(pairIdentity(a), pairIdentity(b));
  need(raw.candidate_records_changed === candidateChanged && raw.principal_pair_changed === pairChanged &&
    same(raw.ordered_principal_ids, pairIdentity(a).slice(0, 2)) &&
    same(raw.baseline_ordered_principal_ids, pairIdentity(b).slice(0, 2)) &&
    raw.conditional_pair_denominator === pairIdentity(a)[2] &&
    (a.conditional_shares === null ? raw.first_candidate_pair_share === null :
      finite(raw.first_candidate_pair_share) && near(raw.first_candidate_pair_share, a.conditional_shares.mean, 1e-9)),
  "Candidate or named-pair identity differs from native results");
  const providedMasses = raw.party_win_probability as Record<string, unknown> | null;
  const providedDeltas = raw.delta_party_win_probability as Record<string, unknown> | null;
  need(providedMasses && providedDeltas && same(Object.keys(providedMasses).sort(), [...PARTIES].sort()) &&
    same(Object.keys(providedDeltas).sort(), [...PARTIES].sort()), "Expected exact D/R/O masses and deltas");
  for (const party of PARTIES) {
    const currentMass = currentSide.winnerGroups[party]?.probability ?? 0;
    const baselineMass = baselineSide.winnerGroups[party]?.probability ?? 0;
    need(finite(providedMasses[party]) && near(providedMasses[party], currentMass) &&
      finite(providedDeltas[party]) && near(providedDeltas[party], currentMass - baselineMass),
    `Winner-group mass or edition delta differs for ${party}`);
  }
  const groupSame = PARTIES.every(party => same(
    currentSide.winnerGroups[party] && { kind: currentSide.winnerGroups[party].kind,
      members: currentSide.winnerGroups[party].memberCandidateIds },
    baselineSide.winnerGroups[party] && { kind: baselineSide.winnerGroups[party].kind,
      members: baselineSide.winnerGroups[party].memberCandidateIds }));
  const groupReason = candidateChanged ? "Candidate records changed between editions." :
    !groupSame ? "Winner-group membership changed between editions." : null;
  const winnerGroupDelta = Object.fromEntries(PARTIES.map(party => [party,
    groupReason || !currentSide.winnerGroups[party] || !baselineSide.winnerGroups[party] ? null :
      currentSide.winnerGroups[party].probability - baselineSide.winnerGroups[party].probability])) as Record<HouseContextParty, number | null>;
  const pairComparison: HouseContextCase["pairComparison"] = pairChanged ?
    { status: "incompatible", reason: "The ordered named pair or denominator changed between editions." } :
    candidateChanged ? { status: "incompatible", reason: "Candidate records changed between editions." } :
    !currentSide.pair || !baselineSide.pair ?
      { status: "incompatible", reason: "A named pair is unavailable in one or both editions." } :
      { status: "comparable", firstShareMeanDelta: currentSide.pair.firstShareMean - baselineSide.pair.firstShareMean };
  const favoredParty = raw.cook_favored_party as "D" | "R" | null; // checked against the ordinal label above
  return { districtKey: a.district_key, raceId: a.race_id, ruleFamily: a.rule_family,
    cook: { rating: raw.cook_rating, favoredParty, scale: "ordinal_no_probability" },
    current: currentSide, baseline: baselineSide, candidateRecordsChanged: candidateChanged,
    principalPairChanged: pairChanged, winnerGroupDelta, winnerGroupDeltaReason: groupReason,
    pairComparison, baselineSourceFinding: raw.baseline_source_finding, interpretation: raw.interpretation };
}

/** Catalog bytes must first be authenticated by a trusted loader; this checks exact dated attachment. */
export function matchHouseContextComparison(
  catalog: HouseContextComparisonCatalog, current: HouseContextEdition, baseline: HouseContextEdition,
): HouseContextComparisonCatalog | null {
  const sourcePinNames = ["handoffSha256", "handoffReviewSha256", "comparisonReceiptSha256",
    "operatorApprovalSha256", "sourcePolicyApprovalSha256"] as const;
  if (!catalog || !current || !baseline || !catalog.sourcePins || !catalog.currentEdition ||
    !catalog.baselineEdition || catalog.schema !== "HouseContextComparisonCatalogV1" || catalog.publicationQualified !== false ||
    catalog.scope !== "private_selected_excerpts_only" || !Array.isArray(catalog.cases) ||
    catalog.cases.length < 1 || catalog.cases.length > 7 ||
    !same(Object.keys(catalog.sourcePins).sort(), [...sourcePinNames].sort()) ||
    !sourcePinNames.every(name => hash(catalog.sourcePins[name])) ||
    catalog.cases.some(c => !c || !/^([A-Z]{2})-\d{2}$/.test(c.districtKey) ||
      c.raceId !== `2026:${c.districtKey}:house:regular`) ||
    new Set(catalog.cases.map(c => c.districtKey)).size !== catalog.cases.length) return null;
  const keys: (keyof HouseContextEdition)[] = ["publicationSha256", "catalogSha256", "readyPacketSha256",
    "runId", "resultSha256", "issuedAt", "cutoffAt", "mapVintage", "geographySha256"];
  const matches = (saved: HouseContextEdition, seen: HouseContextEdition) =>
    keys.every(key => saved[key] === seen[key]) &&
    [seen.publicationSha256, seen.catalogSha256, seen.readyPacketSha256,
      seen.resultSha256, seen.geographySha256].every(hash);
  return matches(catalog.currentEdition, current) && matches(catalog.baselineEdition, baseline) ? catalog : null;
}
