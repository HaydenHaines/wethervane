/** Static tipping-point summaries derived from the saved joint forecast draws. */

export type TippingParty = "D" | "R";
export type CaucusScenarioId =
  | "campaign_pledge_unaligned"
  | "independent_opposition_caucuses_D"
  | "independent_opposition_caucuses_R";

export interface TippingCaucus {
  thresholdSeats: number;
  retainedSeats: number;
  /** Contests where both named candidates belong to this caucus; one seat is guaranteed. */
  guaranteedRaceIds: string[];
  /** Contests where exactly one named candidate belongs to this caucus. */
  eligibleRaceIds: string[];
  /** Candidate from this caucus in each eligible race, keyed by race ID. */
  eligibleCandidateByRace: Record<string, string>;
  requiredContestedWins: number;
  status: "reachable" | "unreachable" | "already_reached";
  statusReason?: string | null;
  projectedTippingRaceId: string | null;
  /** Projected tipping candidate margin in percentage points, from this caucus's perspective. */
  projectedMarginPp: number | null;
  /** Unconditional fraction of joint draws in which each race is the kth ordered margin. */
  frequencyByRace: Record<string, number>;
}

export interface TippingScenario {
  D: TippingCaucus;
  R: TippingCaucus;
}

export interface ModelTippingSummary {
  simulationCount: number;
  /** All races, sorted by first-contender mean share descending, then race ID ascending. */
  raceOrder: string[];
  scenarios: Record<string, TippingScenario>;
}

export interface TippingSnapshot {
  schemaVersion: 1;
  method: "caucus_rank_threshold_v1";
  issuedAt: string;
  cutoffAt: string;
  sources: {
    pollingForecastSha256: string;
    pollingJointDrawsSha256: string;
    fundamentalsForecastSha256: string;
    fundamentalsJointDrawsSha256: string;
  };
  models: {
    polling: ModelTippingSummary;
    fundamentals: ModelTippingSummary;
  };
}

export interface TippingProvenance {
  schemaVersion: 1;
  snapshotSha256: string;
  sources: TippingSnapshot["sources"];
  auditArtifacts: {
    pollingFrozenInputsSha256: string;
    fundamentalsFrozenInputsSha256: string;
    pollingChamberDrawsSha256: string;
  };
}
