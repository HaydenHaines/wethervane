/** Public, static forecast snapshot contract consumed by the SiteGround site. */

export type PartyCode = "D" | "R" | "I";
export type ChamberParty = "D" | "R" | "I";
export type CaucusParty = "D" | "R";

export interface ShareQuantiles {
  p05: number;
  p50: number;
  p95: number;
}

export interface SiteCandidate {
  id: string;
  name: string;
  party: PartyCode;
  incumbent: boolean;
  sourceUrls: string[];
}

export interface SiteRaceForecast {
  id: string;
  state: string;
  seatId: string;
  candidates: [SiteCandidate, SiteCandidate];
  /** Probability that candidates[0] wins under the forecast's two-principal-contender law. */
  firstWinProbability: number;
  /** Probability that candidates[1] wins under the same law. */
  secondWinProbability: number;
  /** Expected first-contender share on the declared two-principal share scale. */
  firstShareMean: number;
  /** Quantiles for the first contender's share. */
  shareQuantiles: ShareQuantiles;
  /** Direct admitted poll observations; absent for fundamentals-only forecasts. */
  pollCount?: number;
  /** Forecast rule proxy and omitted-option assumptions for this contest. */
  notes: string[];
}

export interface SeatDistribution {
  expectedSeats: number;
  seatQuantiles: ShareQuantiles;
  /** Index is the seat count; value is the probability of that count. */
  pmf: number[];
}

export interface ControlScenario {
  description: string;
  democraticMajorityProbability: number;
  republicanMajorityProbability: number;
  unresolvedOrganizingProbability: number;
}

export interface ModelSnapshot {
  targetId: string;
  issuedAt: string;
  cutoffAt: string;
  races: SiteRaceForecast[];
  /** Majority-control probabilities under the corresponding caucus assumption. */
  controls: Record<string, ControlScenario>;
  chamberDistributions: {
    /** Seat distributions by party label, regardless of caucus membership. */
    party: Record<ChamberParty, SeatDistribution>;
    /** Seat distributions under each declared caucus scenario. */
    caucus: Record<string, Record<CaucusParty, SeatDistribution>>;
  };
  limitations: string[];
}

export interface SiteForecastSnapshot {
  schemaVersion: 1;
  /** Issuance time for the current standard polling model. */
  issuedAt: string;
  /** Information cutoff for the current standard polling model. */
  cutoffAt: string;
  sources: {
    pollingSha256: string;
    fundamentalsSha256: string;
  };
  models: {
    polling: ModelSnapshot;
    fundamentals: ModelSnapshot;
  };
}

export interface SnapshotProvenance {
  schemaVersion: 1;
  snapshotSha256: string;
  sources: Record<"polling" | "fundamentals", {
    sha256: string;
    targetId: string;
    artifactLabel: string;
  }>;
}

// Short names for page/component imports.
export type Candidate = SiteCandidate;
export type RaceForecast = SiteRaceForecast;
export type ForecastSnapshot = SiteForecastSnapshot;
