/** Cross-run identity checks for already-parsed live House artifacts.
 * This does not authenticate pins or establish scientific/publication review.
 */
import type { HouseLiveCatalog } from "./house-live-catalog.ts";
import { houseLiveConfigurationKey, type HouseLiveResult } from "./house-live-result.ts";

export interface HouseLiveCollectionValidation {
  valid: boolean;
  issues: string[];
}

function sorted(values: readonly string[]): string[] {
  return [...values].sort((left, right) => left.localeCompare(right));
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function geographyIdentity(result: HouseLiveResult) {
  return {
    mapVintage: result.map_vintage,
    geography: {
      path: result.geography.path,
      sha256: result.geography.sha256,
      featureCount: result.geography.feature_count,
      mapVintage: result.geography.map_vintage,
      sourceReceipt: {
        path: result.geography.source_receipt.path,
        sha256: result.geography.source_receipt.sha256,
      },
    },
  };
}

function sourcePinSignature(result: HouseLiveResult, sourceIds: readonly string[]): string[] {
  const byId = new Map(result.provenance.map((source) => [source.id, source]));
  return sorted(sourceIds).map((id) => {
    const source = byId.get(id);
    // Native result parsing guarantees candidate source references resolve.
    return JSON.stringify(source ? [source.id, source.url, source.sha256, source.role] : [id, "missing"]);
  });
}

function raceIdentity(result: HouseLiveResult) {
  return new Map(result.races.map((race) => {
    const candidates = [...race.candidates].map((candidate) => ({
      id: candidate.candidate_id,
      name: candidate.name,
      partyLabel: candidate.party_label,
      aggregationParty: candidate.aggregation_party,
      ballotStatus: candidate.ballot_status,
      incumbent: candidate.incumbent,
      sourceIds: sorted(candidate.source_ids),
      sourcePins: sourcePinSignature(result, candidate.source_ids),
    })).sort((a, b) => a.id.localeCompare(b.id));
    const outcomes = [...race.outcomes].map((outcome) => ({
      id: outcome.outcome_id,
      party: outcome.party,
      kind: outcome.kind,
      memberCandidateIds: sorted(outcome.member_candidate_ids),
    })).sort((a, b) => a.id.localeCompare(b.id));
    return [race.race_id, JSON.stringify({
      districtKey: race.district_key,
      state: race.state,
      districtNumber: race.district_number,
      mapVintage: race.map_vintage,
      ruleFamily: race.rule_family,
      candidates,
      outcomes,
    })] as const;
  }));
}

/**
 * Verify identity invariants shared by all saved configurations before staging.
 * Probabilities, polls, conditional shares, labels, and qualifications remain
 * run-specific. Array order is ignored except for each run's own draw ordering,
 * which is left untouched and belongs to that run's artifact contract.
 */
export function validateHouseLiveCollection(
  catalog: HouseLiveCatalog,
  results: readonly HouseLiveResult[],
): HouseLiveCollectionValidation {
  const issues: string[] = [];
  if (catalog.entries.length !== 16)
    issues.push(`Catalog must contain 16 saved configurations (found ${catalog.entries.length}).`);
  if (results.length !== 16)
    issues.push(`Collection must contain 16 parsed results (found ${results.length}).`);

  const entriesById = new Map(catalog.entries.map((entry) => [entry.id, entry]));
  if (entriesById.size !== catalog.entries.length)
    issues.push("Catalog contains duplicate result IDs.");
  const resultsById = new Map<string, HouseLiveResult>();
  for (const result of results) {
    if (resultsById.has(result.id)) issues.push(`Results contain duplicate ID ${result.id}.`);
    else resultsById.set(result.id, result);
  }

  for (const entry of catalog.entries) {
    const result = resultsById.get(entry.id);
    if (!result) {
      issues.push(`Catalog result ${entry.id} is missing from the parsed collection.`);
      continue;
    }
    if (result.id !== entry.id || houseLiveConfigurationKey(result.configuration) !== houseLiveConfigurationKey(entry.configuration))
      issues.push(`Result ${entry.id} does not match its catalog ID/configuration.`);
    if (result.cycle !== catalog.cycle)
      issues.push(`Result ${entry.id} is for cycle ${result.cycle}, not catalog cycle ${catalog.cycle}.`);
  }
  for (const result of results) {
    if (!entriesById.has(result.id)) issues.push(`Parsed result ${result.id} is not present in the catalog.`);
  }

  const first = results.find((result) => entriesById.has(result.id));
  if (first) {
    const firstGeography = geographyIdentity(first);
    const firstRaces = raceIdentity(first);
    if (first.races.length !== 435 || firstRaces.size !== 435)
      issues.push(`Result ${first.id} does not have 435 unique race identities.`);

    for (const result of results) {
      if (result === first) continue;
      if (!sameJson(geographyIdentity(result), firstGeography))
        issues.push(`Result ${result.id} changes the shared map/geography/source-receipt identity.`);
      const races = raceIdentity(result);
      if (races.size !== firstRaces.size || [...firstRaces.keys()].some((id) => !races.has(id))) {
        issues.push(`Result ${result.id} changes the 435-race ID/district roster.`);
        continue;
      }
      for (const [raceId, identity] of firstRaces) {
        if (races.get(raceId) !== identity) {
          issues.push(`Result ${result.id} changes candidate, outcome, rule, or map identity for ${raceId}.`);
        }
      }
    }
  }

  return { valid: issues.length === 0, issues };
}
