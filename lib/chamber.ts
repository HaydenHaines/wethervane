import type { ExplorerRace, ExplorerScenario } from "./explorer-types";

export type ChamberId = "senate" | "house";
export const CHAMBERS = {
  senate: {
    id: "senate",
    label: "Senate",
    chamberSize: 100,
    majorityThreshold: 51,
    geography: "state",
  },
  house: {
    id: "house",
    label: "House",
    chamberSize: 435,
    majorityThreshold: 218,
    geography: "district",
  },
} as const;

/** District identity is never transferred across elections or map vintages. */
export interface DistrictIdentity {
  cycle: number;
  mapVintage: string;
  districtKey: string;
}

export interface ChamberMetadata {
  chamber: ChamberId;
  chamber_size: number;
  /** Absolute majority; a Senate VP scenario may have a 50-seat control threshold. */
  majority_threshold: number;
  scope: "eligible_race_subset" | "full_chamber";
  full_chamber: boolean;
  cycle: number;
  mapVintage?: string;
  configurationId?: string;
}

export interface ChamberRace
  extends Pick<ExplorerRace, "id" | "state" | "seatId" | "outcomes"> {
  district?: DistrictIdentity;
}

export interface ChamberGeography {
  kind: "state" | "district";
  cycle?: number;
  mapVintage?: string;
  admission?: "verified" | "unverified";
  sourceSha256?: string;
  /** Exact state codes or backend-issued district keys; no frontend crosswalk. */
  keys: string[];
}

export interface JointDrawIdentity {
  id: string;
  configurationId: string;
  cycle: number;
  sourceSha256: string;
  simulationCount: number;
  raceIds: string[];
}

export interface FullChamberProof {
  /** Authenticated artifact binding, checked against downloaded bytes by its adapter. */
  sourceSha256: string;
  jointDrawsId: string;
  jointArtifactSha256: string;
  configurationId: string;
  cycle: number;
  expectedSeatIds: string[];
  retainedSeatIds: string[];
  scenarios: ExplorerScenario[];
}

export interface FullTippingProof {
  sourceSha256: string;
  jointDrawsId: string;
  jointArtifactSha256: string;
  configurationId: string;
  cycle: number;
  method: string;
  /** Explicit saved threshold-rank probabilities; never reconstructed from marginals. */
  scenarios: {
    id: string;
    frequencies: { D: Record<string, number>; R: Record<string, number> };
  }[];
}

export interface ChamberReadinessInput {
  metadata: ChamberMetadata;
  /** The authenticated eligible-race manifest, which can itself be a subset. */
  expectedRaceIds: string[];
  races: ChamberRace[];
  geography?: ChamberGeography;
  jointDraws?: JointDrawIdentity;
  chamberProof?: FullChamberProof;
  tippingProof?: FullTippingProof;
}

export interface ChamberReadiness {
  status:
    | "unavailable"
    | "invalid"
    | "race_subset"
    | "races_only"
    | "full_chamber";
  canShowRaces: boolean;
  canShowMap: boolean;
  canShowRaceRibbon: boolean;
  canShowControl: boolean;
  canShowTipping: boolean;
  coverage: {
    chamberSize: number;
    expectedRaces: number;
    availableRaces: number;
    missingRaceIds: string[];
    unexpectedRaceIds: string[];
    missingGeographyIds: string[];
  };
  issues: { code: string; message: string }[];
}

const PARTIES = ["D", "R", "U"] as const;
const TOLERANCE = 1e-8;
const text = (value: unknown): value is string =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= 512 &&
  value === value.trim();
const probability = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= 0 &&
  value <= 1;
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const sum = (values: number[]) =>
  values.reduce((total, value) => total + value, 0);
const strings = (values: unknown): values is string[] =>
  Array.isArray(values) &&
  values.every(text) &&
  new Set(values).size === values.length;
const sameSet = (first: string[], second: string[]) =>
  first.length === second.length &&
  new Set(first).size === first.length &&
  first.every((value) => second.includes(value));
const hash = (value: unknown) =>
  typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);

/** Collision-safe and precision-preserving; malformed identities have no join key. */
export function districtIdentityKey(value: DistrictIdentity): string | null {
  if (
    !value ||
    !Number.isInteger(value.cycle) ||
    value.cycle < 1 ||
    !text(value.mapVintage) ||
    !text(value.districtKey) ||
    !/^[A-Z]{2}-\d{2}$/.test(value.districtKey)
  )
    return null;
  return JSON.stringify([value.cycle, value.mapVintage, value.districtKey]);
}

function validDistribution(
  value: ExplorerScenario["seats"]["D"],
  size: number,
): boolean {
  if (
    !value ||
    !Array.isArray(value.pmf) ||
    value.pmf.length !== size + 1 ||
    !value.pmf.every(probability) ||
    Math.abs(sum(value.pmf) - 1) > TOLERANCE ||
    !finite(value.expectedSeats) ||
    Math.abs(
      sum(value.pmf.map((p, seats) => p * seats)) - value.expectedSeats,
    ) > TOLERANCE
  )
    return false;
  const q = value.seatQuantiles;
  return (
    !!q &&
    [q.p05, q.p50, q.p95].every(finite) &&
    0 <= q.p05 &&
    q.p05 <= q.p50 &&
    q.p50 <= q.p95 &&
    q.p95 <= size
  );
}

function validScenario(
  scenario: ExplorerScenario,
  metadata: ChamberMetadata,
  retained: number,
): boolean {
  if (!scenario || !text(scenario.id)) return false;
  const {
    chamber_size: size,
    majority_threshold: majority,
    chamber,
  } = metadata;
  if (
    !PARTIES.every((party) => validDistribution(scenario.seats?.[party], size))
  )
    return false;
  if (
    Math.abs(
      sum(PARTIES.map((party) => scenario.seats[party].expectedSeats)) - size,
    ) > TOLERANCE
  )
    return false;
  if (
    !scenario.retainedSeats ||
    !PARTIES.every(
      (party) =>
        Number.isInteger(scenario.retainedSeats[party]) &&
        scenario.retainedSeats[party] >= 0,
    ) ||
    sum(PARTIES.map((party) => scenario.retainedSeats[party])) !== retained
  )
    return false;
  const dThreshold = scenario.thresholds?.D,
    rThreshold = scenario.thresholds?.R;
  if (![dThreshold, rThreshold].every(Number.isInteger)) return false;
  if (
    chamber === "house"
      ? dThreshold !== majority || rThreshold !== majority
      : ![majority - 1, majority].includes(dThreshold) ||
        ![majority - 1, majority].includes(rThreshold) ||
        dThreshold + rThreshold <= size
  )
    return false;
  const control = scenario.control;
  const probabilities = [
    control?.democraticMajorityProbability,
    control?.republicanMajorityProbability,
    control?.unresolvedOrganizingProbability,
  ];
  if (
    !probabilities.every(probability) ||
    Math.abs(sum(probabilities) - 1) > TOLERANCE
  )
    return false;
  return (
    Math.abs(sum(scenario.seats.D.pmf.slice(dThreshold)) - probabilities[0]) <=
      TOLERANCE &&
    Math.abs(sum(scenario.seats.R.pmf.slice(rThreshold)) - probabilities[1]) <=
      TOLERANCE
  );
}

/**
 * Structural UI gates for already-authenticated artifacts, not a scientific
 * qualification or a replacement for checking their hashes and joint draws.
 * Existing Senate snapshots must be adapted explicitly; missing metadata never
 * causes an unknown artifact to be guessed to be a Senate forecast.
 */
export function validateChamberReadiness(
  input: ChamberReadinessInput,
): ChamberReadiness {
  const issues: ChamberReadiness["issues"] = [];
  const add = (code: string, message: string) => issues.push({ code, message });
  const metadata = input?.metadata;
  const known =
    metadata && (metadata.chamber === "senate" || metadata.chamber === "house")
      ? CHAMBERS[metadata.chamber]
      : null;
  const expected = Array.isArray(input?.expectedRaceIds)
    ? input.expectedRaceIds
    : [];
  const races = Array.isArray(input?.races) ? input.races : [];
  const ids = races.map((race) => race?.id);
  const coverage: ChamberReadiness["coverage"] = {
    chamberSize: known?.chamberSize || 0,
    expectedRaces: expected.length,
    availableRaces: races.length,
    missingRaceIds: expected.filter((id) => !ids.includes(id)),
    unexpectedRaceIds: ids.filter((id) => !expected.includes(id)),
    missingGeographyIds: [],
  };
  const result: ChamberReadiness = {
    status: "invalid",
    canShowRaces: false,
    canShowMap: false,
    canShowRaceRibbon: false,
    canShowControl: false,
    canShowTipping: false,
    coverage,
    issues,
  };
  if (
    !known ||
    metadata.chamber_size !== known.chamberSize ||
    metadata.majority_threshold !== known.majorityThreshold ||
    !Number.isInteger(metadata.cycle) ||
    metadata.cycle < 1 ||
    !["eligible_race_subset", "full_chamber"].includes(metadata.scope) ||
    typeof metadata.full_chamber !== "boolean" ||
    metadata.full_chamber !== (metadata.scope === "full_chamber")
  ) {
    add(
      "invalid_chamber_metadata",
      "Chamber size, majority, cycle, and declared forecast scope must agree.",
    );
    return result;
  }
  if (
    !strings(expected) ||
    !strings(ids) ||
    (races.length > 0 && expected.length === 0)
  ) {
    add(
      "invalid_race_manifest",
      "The eligible-race manifest and forecast identifiers must be explicit and unique.",
    );
    return result;
  }
  if (coverage.unexpectedRaceIds.length) {
    add(
      "unexpected_races",
      "Forecast races occur outside the declared eligible manifest.",
    );
    return result;
  }
  if (coverage.missingRaceIds.length)
    add("missing_races", "Some declared eligible races have no forecast.");
  const seatIds = races.map((race) => race.seatId);
  if (!strings(seatIds)) {
    add(
      "invalid_seat_identity",
      "Each forecast needs one distinct, nonempty seat identifier.",
    );
    return result;
  }
  for (const race of races) {
    if (
      !text(race.state) ||
      !Array.isArray(race.outcomes) ||
      !race.outcomes.length ||
      !strings(race.outcomes.map((outcome) => outcome?.id)) ||
      !race.outcomes.every((outcome) => probability(outcome.winProbability)) ||
      Math.abs(
        sum(race.outcomes.map((outcome) => outcome.winProbability)) - 1,
      ) > TOLERANCE
    ) {
      add(
        "invalid_race_probabilities",
        `Race ${race.id} has an invalid categorical forecast or identity.`,
      );
      return result;
    }
    if (
      metadata.chamber === "house" &&
      (!race.district ||
        !districtIdentityKey(race.district) ||
        race.district.cycle !== metadata.cycle ||
        race.district.mapVintage !== metadata.mapVintage ||
        !race.district.districtKey.startsWith(race.state + "-") ||
        ![
          `${metadata.cycle}:${race.district.districtKey}:house:regular`,
          `${metadata.cycle}:${race.district.districtKey}:house:special`,
        ].includes(race.id))
    ) {
      add(
        "invalid_district_identity",
        `Race ${race.id} does not match the forecast cycle and district-map vintage.`,
      );
      return result;
    }
  }
  if (!races.length) {
    result.status = "unavailable";
    add(
      "forecast_unavailable",
      "No race forecast artifact is available for this chamber.",
    );
    return result;
  }
  if (
    metadata.chamber === "house" &&
    (!text(metadata.mapVintage) ||
      new Set(races.map((race) => districtIdentityKey(race.district!))).size !==
        races.length)
  ) {
    add(
      "invalid_map_vintage",
      "House races need one exact cycle, map vintage, and district identity per seat.",
    );
    return result;
  }
  result.canShowRaces = result.canShowRaceRibbon = true;
  result.status =
    metadata.scope === "eligible_race_subset" ? "race_subset" : "races_only";
  const geography = input.geography;
  if (
    !geography ||
    geography.kind !== known.geography ||
    !strings(geography.keys) ||
    (geography.cycle !== undefined && geography.cycle !== metadata.cycle) ||
    (metadata.chamber === "house" &&
      (geography.cycle !== metadata.cycle ||
        geography.mapVintage !== metadata.mapVintage ||
        geography.admission !== "verified" ||
        !hash(geography.sourceSha256) ||
        !geography.keys.every((key) => /^[A-Z]{2}-\d{2}$/.test(key))))
  ) {
    add(
      "geography_unavailable",
      "Matching chamber geography, cycle, and map vintage are unavailable.",
    );
  } else {
    const required = races.map((race) =>
      metadata.chamber === "house" ? race.district!.districtKey : race.state,
    );
    coverage.missingGeographyIds = [
      ...new Set(required.filter((id) => !geography.keys.includes(id))),
    ];
    result.canShowMap = coverage.missingGeographyIds.length === 0;
    if (!result.canShowMap)
      add(
        "missing_geography",
        "Some forecast identities have no matching map feature.",
      );
  }
  const joint = input.jointDraws;
  const alignedJoint =
    !!joint &&
    text(joint.id) &&
    hash(joint.sourceSha256) &&
    text(metadata.configurationId) &&
    joint.configurationId === metadata.configurationId &&
    joint.cycle === metadata.cycle &&
    Number.isInteger(joint.simulationCount) &&
    joint.simulationCount > 0 &&
    strings(joint.raceIds) &&
    sameSet(joint.raceIds, ids);
  if (joint && !alignedJoint)
    add(
      "invalid_joint_identity",
      "Joint-draw identity does not cover these exact forecast races.",
    );
  if (!metadata.full_chamber) {
    add(
      "subset_has_no_control",
      "An eligible-race subset cannot establish full-chamber control or a chamber tipping point.",
    );
    return result;
  }
  const proof = input.chamberProof;
  if (
    !proof ||
    !hash(proof.sourceSha256) ||
    !alignedJoint ||
    proof.jointDrawsId !== joint!.id ||
    proof.jointArtifactSha256 !== joint!.sourceSha256 ||
    proof.configurationId !== joint!.configurationId ||
    proof.cycle !== metadata.cycle ||
    !strings(proof.expectedSeatIds) ||
    proof.expectedSeatIds.length !== known.chamberSize ||
    !strings(proof.retainedSeatIds) ||
    !sameSet([...seatIds, ...proof.retainedSeatIds], proof.expectedSeatIds) ||
    coverage.missingRaceIds.length > 0 ||
    (metadata.chamber === "house" && proof.retainedSeatIds.length > 0) ||
    !Array.isArray(proof.scenarios) ||
    !proof.scenarios.length ||
    !strings(proof.scenarios.map((scenario) => scenario?.id)) ||
    !proof.scenarios.every((scenario) =>
      validScenario(scenario, metadata, proof.retainedSeatIds.length),
    )
  ) {
    add(
      "full_chamber_proof_missing",
      "Full seat coverage and a matching joint-derived chamber distribution are not qualified for display.",
    );
    return result;
  }
  result.canShowControl = true;
  result.status = "full_chamber";
  const tipping = input.tippingProof;
  if (tipping) {
    const valid =
      hash(tipping.sourceSha256) &&
      text(tipping.method) &&
      tipping.jointDrawsId === joint!.id &&
      tipping.jointArtifactSha256 === joint!.sourceSha256 &&
      tipping.configurationId === joint!.configurationId &&
      tipping.cycle === metadata.cycle &&
      Array.isArray(tipping.scenarios) &&
      strings(tipping.scenarios.map((scenario) => scenario?.id)) &&
      sameSet(
        tipping.scenarios.map((scenario) => scenario.id),
        proof.scenarios.map((scenario) => scenario.id),
      ) &&
      tipping.scenarios.every((scenario) =>
        (["D", "R"] as const).every((party) => {
          const vector = scenario.frequencies?.[party];
          return (
            !!vector &&
            sameSet(Object.keys(vector), ids) &&
            Object.values(vector).every(probability) &&
            Math.abs(sum(Object.values(vector)) - 1) <= TOLERANCE
          );
        }),
      );
    result.canShowTipping = valid;
    if (!valid)
      add(
        "invalid_tipping_proof",
        "Tipping frequencies must be bound to the same complete joint artifact and race identities.",
      );
  }
  return result;
}
