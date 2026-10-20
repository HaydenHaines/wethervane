import type {
  ChamberMetadata,
  ChamberReadinessInput,
  DistrictIdentity,
} from "./chamber";
import type { ExplorerRace } from "./explorer-types";
import type { DistrictGeography } from "./district-geography";

/** Frontend publication adapter, not a claim that the research exporter implements this format. */
export interface HouseResult {
  schemaVersion: 1;
  reviewStatus: "reviewed";
  run: {
    id: string;
    configurationId: string;
    configurationSha256: string;
    label: string;
    year: number;
    historical: boolean;
    experimental: boolean;
    informationMode:
      | "retrospective_proxy_strictly_earlier_fitted_labels"
      | "strict_asof";
    evidenceLabel: string;
  };
  metadata: ChamberMetadata & { chamber: "house"; mapVintage: string };
  races: (ExplorerRace & { district: DistrictIdentity })[];
  coverage: {
    raceId: string;
    districtKey: string;
    eligible: boolean;
    exclusionReason: string | null;
  }[];
  jointDraws?: ChamberReadinessInput["jointDraws"];
  chamberProof?: ChamberReadinessInput["chamberProof"];
  tippingProof?: ChamberReadinessInput["tippingProof"];
  limitations: string[];
}

export type HousePublication =
  | { schemaVersion: 1; status: "preparing" }
  | {
      schemaVersion: 1;
      status: "available";
      result: HouseResult;
      resultSha256: string;
      geography: DistrictGeography | null;
    };

const record = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const label = (v: unknown): v is string =>
  typeof v === "string" && v.trim().length > 0;
const hash = (v: unknown) => typeof v === "string" && /^[a-f0-9]{64}$/i.test(v);
const strings = (v: unknown) => Array.isArray(v) && v.every(label);
const links = (v: unknown) =>
  Array.isArray(v) &&
  v.every((x) => typeof x === "string" && /^https?:\/\//i.test(x));
const probability = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;
export const HOUSE_SHARE_DENOMINATOR =
  "Democratic and Republican candidate votes";

/** Validate presentation fields and a complete coverage ledger before any chart consumes them. */
export function parseHouseResult(value: unknown): HouseResult | null {
  if (
    !record(value) ||
    value.schemaVersion !== 1 ||
    value.reviewStatus !== "reviewed" ||
    !record(value.run) ||
    !record(value.metadata) ||
    !Array.isArray(value.races) ||
    !value.races.length ||
    !Array.isArray(value.coverage) ||
    value.coverage.length !== 435 ||
    !strings(value.limitations)
  )
    return null;
  const run = value.run,
    metadata = value.metadata;
  if (
    ![run.id, run.configurationId, run.label, run.evidenceLabel].every(label) ||
    !hash(run.configurationSha256) ||
    !Number.isInteger(run.year) ||
    typeof run.historical !== "boolean" ||
    typeof run.experimental !== "boolean" ||
    ![
      "retrospective_proxy_strictly_earlier_fitted_labels",
      "strict_asof",
    ].includes(String(run.informationMode)) ||
    metadata.chamber !== "house" ||
    metadata.chamber_size !== 435 ||
    metadata.majority_threshold !== 218 ||
    metadata.cycle !== run.year ||
    metadata.configurationId !== run.configurationId ||
    !label(metadata.mapVintage)
  )
    return null;
  const ledger = new Map<string, Record<string, unknown>>(),
    districts = new Set<string>();
  for (const entry of value.coverage) {
    if (
      !record(entry) ||
      !label(entry.raceId) ||
      typeof entry.districtKey !== "string" ||
      !/^[A-Z]{2}-\d{2}$/.test(entry.districtKey) ||
      districts.has(entry.districtKey) ||
      ledger.has(entry.raceId) ||
      entry.raceId !== `${run.year}:${entry.districtKey}:house:regular` ||
      typeof entry.eligible !== "boolean" ||
      (entry.eligible
        ? entry.exclusionReason !== null
        : !label(entry.exclusionReason))
    )
      return null;
    districts.add(entry.districtKey);
    ledger.set(entry.raceId, entry);
  }
  const seen = new Set<string>();
  for (const race of value.races) {
    if (
      !record(race) ||
      !label(race.id) ||
      seen.has(race.id) ||
      !record(race.district) ||
      !label(race.state) ||
      !label(race.seatId) ||
      !label(race.electionLabel) ||
      !strings(race.notes) ||
      !Array.isArray(race.outcomes) ||
      race.outcomes.length !== 2
    )
      return null;
    const entry = ledger.get(race.id);
    if (
      !entry?.eligible ||
      race.district.districtKey !== entry.districtKey ||
      race.district.cycle !== run.year ||
      race.district.mapVintage !== metadata.mapVintage
    )
      return null;
    seen.add(race.id);
    const parties = new Set<string>(),
      ids = new Set<string>();
    let total = 0;
    for (const outcome of race.outcomes) {
      if (
        !record(outcome) ||
        !label(outcome.id) ||
        ids.has(outcome.id) ||
        !label(outcome.name) ||
        !["D", "R"].includes(String(outcome.party)) ||
        parties.has(String(outcome.party)) ||
        outcome.kind !== "candidate" ||
        !probability(outcome.winProbability) ||
        !Array.isArray(outcome.members) ||
        outcome.members.length !== 1
      )
        return null;
      const member = outcome.members[0];
      if (
        !record(member) ||
        !label(member.id) ||
        member.name !== outcome.name ||
        member.party !== outcome.party ||
        !links(member.sourceUrls)
      )
        return null;
      parties.add(String(outcome.party));
      ids.add(outcome.id);
      total += outcome.winProbability;
      if (outcome.share !== undefined) {
        const share = outcome.share;
        if (
          !record(share) ||
          !probability(share.mean) ||
          share.denominator !== HOUSE_SHARE_DENOMINATOR ||
          !record(share.quantiles)
        )
          return null;
        const { p05, p50, p95 } = share.quantiles;
        if (
          ![p05, p50, p95].every(probability) ||
          Number(p05) > Number(p50) ||
          Number(p50) > Number(p95)
        )
          return null;
      }
    }
    if (Math.abs(total - 1) > 1e-8) return null;
    const [left, right] = race.outcomes as unknown as ExplorerRace["outcomes"];
    if (
      left.share &&
      right.share &&
      [
        left.share.mean + right.share.mean,
        left.share.quantiles.p05 + right.share.quantiles.p95,
        left.share.quantiles.p50 + right.share.quantiles.p50,
        left.share.quantiles.p95 + right.share.quantiles.p05,
      ].some((total) => Math.abs(total - 1) > 1e-8)
    )
      return null;
    if (
      race.observed !== undefined &&
      (!record(race.observed) ||
        !ids.has(String(race.observed.outcomeId)) ||
        !label(race.observed.winnerName) ||
        !parties.has(String(race.observed.winnerParty)) ||
        !links(race.observed.sourceUrls) ||
        race.outcomes.find(
          (o) => o.id === (race.observed as Record<string, unknown>).outcomeId,
        )?.party !== race.observed.winnerParty ||
        race.outcomes.find(
          (o) => o.id === (race.observed as Record<string, unknown>).outcomeId,
        )?.name !== race.observed.winnerName)
    )
      return null;
  }
  if (
    [...ledger.values()].some(
      (entry) => entry.eligible && !seen.has(String(entry.raceId)),
    )
  )
    return null;
  if (
    metadata.full_chamber &&
    (seen.size !== 435 || metadata.scope !== "full_chamber")
  )
    return null;
  if (!metadata.full_chamber && metadata.scope !== "eligible_race_subset")
    return null;
  return value as unknown as HouseResult;
}

export function houseReadinessInput(
  result: HouseResult,
  geography: DistrictGeography | null,
): ChamberReadinessInput {
  return {
    metadata: result.metadata,
    expectedRaceIds: result.coverage
      .filter((row) => row.eligible)
      .map((row) => row.raceId),
    races: result.races,
    jointDraws: result.jointDraws,
    chamberProof: result.chamberProof,
    // The UI describes this statistic specifically; other definitions need their own adapter.
    tippingProof:
      result.tippingProof?.method === "caucus_rank_threshold_v1"
        ? result.tippingProof
        : undefined,
    ...(geography
      ? {
          geography: {
            kind: "district" as const,
            ...geography.metadata,
            keys:
              geography.geojson?.features?.map(
                (feature) => feature.properties?.districtKey,
              ) || [],
          },
        }
      : {}),
  };
}
