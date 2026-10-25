/** Whitelisted presentation projection. Call only after authenticating the reviewed native artifacts. */
import type { ExplorerRace, ExplorerScenario } from "./explorer-types";
import type { DistrictIdentity } from "./chamber";
import type { HouseLiveCatalog, HouseLiveCatalogEntry } from "./house-live-catalog";
import { houseLiveConfigurationKey, type HouseLiveResult, type HouseLiveRace,
  type HouseLiveConditionalPair, type HouseLiveConfiguration } from "./house-live-result.ts";

export interface HouseLiveRunChoice {
  id: string; label: string; configuration: HouseLiveConfiguration; resultSha256: string;
}
export interface HouseLiveCatalogView {
  defaultId: string;
  entries: HouseLiveRunChoice[];
}
export function houseLiveCatalogView(catalog: HouseLiveCatalog): HouseLiveCatalogView {
  return { defaultId: catalog.default_id, entries: catalog.entries.map(entry => ({
    id: entry.id, label: entry.label, configuration: { ...entry.configuration }, resultSha256: entry.result.sha256,
  })) };
}

export interface HouseLiveRaceView extends ExplorerRace {
  district: DistrictIdentity;
  ruleFamily: HouseLiveRace["rule_family"];
  method: string;
  candidates: {
    id: string; name: string; partyLabel: string | null;
    aggregationParty: "D" | "R" | "O"; ballotStatus: "printed" | "declared_writein";
    incumbent: boolean; sourceUrls: string[];
  }[];
  /** A separate conditional estimand; never attached to unconditional winner outcomes. */
  conditionalPair: null | {
    firstCandidateId: string; secondCandidateId: string;
    firstCandidateName: string; secondCandidateName: string;
    denominator: string; mean: number; quantiles: HouseLiveConditionalPair["quantiles"];
    firstExceedsHalf: number; marginWithinTwoPoints: number;
    firstMarginAtLeastTenPoints: number; secondMarginAtLeastTenPoints: number;
    qualifications: string[];
  };
}

export interface HouseLiveView {
  schema: "HouseLiveDisplayV1";
  run: {
    id: string; label: string; cycle: 2026; historical: false;
    configuration: HouseLiveConfiguration; resultSha256: string;
    issuedAt: string; cutoffAt: string; electionDate: string; strictAsOf: boolean;
    sourceCutoffs: { role: string; at: string }[];
  };
  mapVintage: string;
  races: HouseLiveRaceView[];
  chamber: ExplorerScenario | null;
  /** No supported live tipping statistic has been admitted by this draft contract. */
  tipping: null;
  qualification: {
    inputStatus: string; numericalStatus: string; publicationStatus: string;
    forecastSkill: string; limitations: string[]; evidenceHashes: string[];
  };
  chamberPrecision: null | { maxSeatMeanMcse: number; maxControlMcse: number; evidenceSha256: string };
  jointDraws: { sha256: string; count: number; chainCount: number; orderedRaceIds: string[]; identitySemantics: string };
  geographySha256: string;
  sources: HouseLiveResult["provenance"];
  historicalEvidence: HouseLiveResult["historical_evidence"];
}

export function houseLiveView(result: HouseLiveResult, entry: HouseLiveCatalogEntry): HouseLiveView {
  if (entry.id !== result.id || houseLiveConfigurationKey(entry.configuration) !== houseLiveConfigurationKey(result.configuration))
    throw Error("Live House display selection does not match its reviewed result.");
  if (result.qualification.publication_status !== "reviewed")
    throw Error("Live House display requires reviewed publication status and separate artifact authentication.");
  const sources = new Map(result.provenance.map(source => [source.id, source]));
  const urls = (ids: string[]) => [...new Set(ids.map(id => {
    const source = sources.get(id);
    if (!source) throw Error("Live House display contains an unbound source.");
    return source.url;
  }))];
  const races: HouseLiveRaceView[] = result.races.map(race => {
    const candidates = race.candidates.map(candidate => ({
      id: candidate.candidate_id, name: candidate.name, partyLabel: candidate.party_label,
      aggregationParty: candidate.aggregation_party, ballotStatus: candidate.ballot_status,
      incumbent: candidate.incumbent, sourceUrls: urls(candidate.source_ids),
    }));
    const byId = new Map(candidates.map(candidate => [candidate.id, candidate]));
    const pair = race.conditional_shares;
    return {
      id: race.race_id, state: race.state, seatId: race.district_key,
      electionLabel: "2026 House election", pollCount: race.poll_count, notes: [...race.qualifications],
      district: { cycle: result.cycle, districtKey: race.district_key, mapVintage: race.map_vintage },
      ruleFamily: race.rule_family, method: race.method, candidates,
      outcomes: race.outcomes.map(outcome => ({
        id: `${race.race_id}:${outcome.outcome_id}`, name: outcome.label, party: outcome.party,
        kind: outcome.kind, winProbability: outcome.p_win,
        members: outcome.member_candidate_ids.map(id => {
          const candidate = byId.get(id);
          if (!candidate) throw Error("Live House outcome contains an unknown candidate.");
          return { id, name: candidate.name, party: candidate.aggregationParty,
            incumbent: candidate.incumbent, sourceUrls: candidate.sourceUrls };
        }),
      })),
      conditionalPair: pair ? {
        firstCandidateId: pair.first_candidate_id, secondCandidateId: pair.second_candidate_id,
        firstCandidateName: byId.get(pair.first_candidate_id)!.name,
        secondCandidateName: byId.get(pair.second_candidate_id)!.name,
        denominator: pair.denominator, mean: pair.mean, quantiles: { ...pair.quantiles },
        firstExceedsHalf: pair.p_first_exceeds_half, marginWithinTwoPoints: pair.p_margin_within_2pp,
        firstMarginAtLeastTenPoints: pair.p_first_margin_at_least_10pp,
        secondMarginAtLeastTenPoints: pair.p_second_margin_at_least_10pp,
        qualifications: [...pair.qualifications],
      } : null,
    };
  });
  const native = result.chamber;
  // Shared presentation calls unassigned seats U; this preserves native O without assigning a caucus.
  const seats = (party: "D" | "R" | "O") => {
    const distribution = native!.seats[party];
    return { expectedSeats: distribution.expected, pmf: [...distribution.pmf],
      seatQuantiles: { p05: distribution.quantiles.p05, p50: distribution.quantiles.p50, p95: distribution.quantiles.p95 } };
  };
  return {
    schema: "HouseLiveDisplayV1",
    run: { id: result.id, label: entry.label, cycle: result.cycle, historical: false,
      configuration: { ...result.configuration }, resultSha256: entry.result.sha256,
      issuedAt: result.issued_at, cutoffAt: result.forecast_cutoff, electionDate: result.election_date,
      strictAsOf: result.strict_as_of, sourceCutoffs: result.source_cutoffs.map(source => ({ role: source.role, at: source.at })) },
    mapVintage: result.map_vintage, races,
    chamber: native ? {
      id: native.scenario_id, label: "Party majority · others unassigned", description: native.description,
      thresholds: { D: native.threshold, R: native.threshold }, retainedSeats: { D: 0, R: 0, U: 0 },
      control: { description: native.description, democraticMajorityProbability: native.majority.D,
        republicanMajorityProbability: native.majority.R, unresolvedOrganizingProbability: native.majority.unresolved },
      seats: { D: seats("D"), R: seats("R"), U: seats("O") },
    } : null,
    tipping: null,
    qualification: { inputStatus: result.qualification.input_status, numericalStatus: result.qualification.numerical_status,
      publicationStatus: result.qualification.publication_status, forecastSkill: result.qualification.forecast_skill,
      limitations: [...result.qualification.limitations], evidenceHashes: result.qualification.evidence.map(pin => pin.sha256) },
    chamberPrecision: native ? { maxSeatMeanMcse: native.precision.max_seat_mean_mcse,
      maxControlMcse: native.precision.max_control_mcse, evidenceSha256: native.precision.evidence.sha256 } : null,
    jointDraws: { sha256: result.joint_draws.sha256, count: result.joint_draws.count,
      chainCount: result.joint_draws.chain_count, orderedRaceIds: [...result.joint_draws.ordered_race_ids],
      identitySemantics: result.joint_draws.identity_semantics },
    geographySha256: result.geography.sha256,
    sources: result.provenance.map(source => ({ id: source.id, label: source.label, url: source.url,
      retrieved_at: source.retrieved_at, sha256: source.sha256, role: source.role })),
    historicalEvidence: result.historical_evidence.map(evidence => ({ label: evidence.label, url: evidence.url, qualification: evidence.qualification })),
  };
}
