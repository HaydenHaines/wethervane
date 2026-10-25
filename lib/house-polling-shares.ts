import { HOUSE_POLLING_POLICY_SHA, HOUSE_POLLING_AUDIT_SHA, HOUSE_POLLING_ADMISSION_SHA } from "./house-polling-result.ts";
import type {
  ChamberMetadata,
  ChamberReadinessInput,
  DistrictIdentity,
} from "./chamber.ts";
import type { ExplorerRace } from "./explorer-types.ts";
import type { DistrictGeography } from "./district-geography.ts";

/** Frontend publication adapter, not a claim that the research exporter implements this format. */
export interface HousePollingShareResult {
  schema: "HousePollingConditionalShareReviewedResultV1";
  currentElectionForecast: false;
  pollingCoverage: {totalDistricts:435;shareEligibleDistricts:number;polledDistricts:number;polls:number;horizonDays:14};
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
      | "retrospective_archive_ingestion_cutoff_proxy_strictly_earlier_fitted_labels";
    evidenceLabel: string;
  };
  metadata: ChamberMetadata & { chamber: "house"; mapVintage: string };
  races: (ExplorerRace & { district: DistrictIdentity; pollCount:number })[];
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

export type HousePollingSharePublication =
  | { schemaVersion: 1; status: "preparing" }
  | {
      schemaVersion: 1;
      status: "available";
      result: HousePollingShareResult;
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
export function parseHousePollingShares(value: unknown): HousePollingShareResult | null {
  if (
    !record(value) ||
    value.schema !== "HousePollingConditionalShareReviewedResultV1" || value.currentElectionForecast !== false ||
    !record(value.sourcePolicy) || value.sourcePolicy.sha256 !== HOUSE_POLLING_POLICY_SHA ||
    !record(value.sourceAdmission) || value.sourceAdmission.sha256 !== HOUSE_POLLING_ADMISSION_SHA ||
    !record(value.audit) || value.audit.sha256 !== HOUSE_POLLING_AUDIT_SHA || !record(value.pollingCoverage) ||
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
  if (value.run.historical !== true || value.run.experimental !== true ||
    !/^house_(history_refreshed|presidential_slope|relative_lean|relative_context)_v1-(2018|2020|2022|2024)-h14-polls-v1$/.test(String(value.run.id)) ||
    value.run.id !== `${String(value.run.configurationId).replace(/_polling_v1$/, "")}-${value.run.year}-h14-polls-v1`) return null;
  const run = value.run,
    metadata = value.metadata;
  if (
    ![run.id, run.configurationId, run.label, run.evidenceLabel].every(label) ||
    !hash(run.configurationSha256) ||
    !Number.isInteger(run.year) ||
    typeof run.historical !== "boolean" ||
    typeof run.experimental !== "boolean" ||
    ![
      "retrospective_archive_ingestion_cutoff_proxy_strictly_earlier_fitted_labels",
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
      !record(race) || !Number.isInteger(race.pollCount) || (race.pollCount as number) < 0 ||
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
        const quantileKeys = ["p025","p05","p10","p50","p90","p95","p975"];
        const quantiles = share.quantiles;
        const ordered = quantileKeys.map(key=>quantiles[key]);
        if (!ordered.every(probability) || ordered.some((q,i)=>i>0 && Number(q)<Number(ordered[i-1]))) return null;
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
    const first = race.outcomes[0], second = race.outcomes[1];
    if (!record(first.share) || !record(second.share) || !record(first.share.quantiles) || !record(second.share.quantiles)) return null;
    const keys=["p025","p05","p10","p50","p90","p95","p975"];
    const a=first.share.quantiles,b=second.share.quantiles;
    if(keys.some((key,i)=>Math.abs(Number(a[key])+Number(b[keys[keys.length-1-i]])-1)>1e-8))return null;
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
        !race.outcomes.some((o) => o.id === (race.observed as Record<string, unknown>).outcomeId &&
          (o.name === (race.observed as Record<string, unknown>).winnerName || o.sourceDisplayName === (race.observed as Record<string, unknown>).winnerName)))
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
  const coverage = value.pollingCoverage;
  if (coverage.totalDistricts !== 435 || coverage.horizonDays !== 14 ||
    coverage.shareEligibleDistricts !== value.races.length ||
    coverage.polledDistricts !== value.races.filter(r=>r.pollCount>0).length ||
    coverage.polls !== value.races.reduce((n,r)=>n+r.pollCount,0) || value.metadata.full_chamber !== false ||
    value.chamberProof !== undefined || value.tippingProof !== undefined) return null;
  if (!record(value.jointDraws) || value.jointDraws.id !== `${run.id}:joint` ||
    value.jointDraws.configurationId !== run.configurationId || value.jointDraws.cycle !== run.year ||
    !hash(value.jointDraws.sourceSha256) || !Number.isInteger(value.jointDraws.simulationCount) ||
    (value.jointDraws.simulationCount as number) < 1 || !Array.isArray(value.jointDraws.raceIds) ||
    JSON.stringify(value.jointDraws.raceIds) !== JSON.stringify(value.races.map(r=>r.id))) return null;
  return value as unknown as HousePollingShareResult;
}
