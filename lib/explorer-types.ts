import type { ControlScenario, SeatDistribution, ShareQuantiles } from "./site-types";
import type { ModelTippingSummary } from "./tipping-types";

/** Every selectable stack resolves to a saved, internally coherent run. */
export interface ExplorerRun {
  id: string;
  year: number;
  label: string;
  targetId: string;
  lawId: string;
  dateId: string;
  dateLabel: string;
  forecastAt: string;
  issuedAt: string;
  cutoffAt: string;
  horizonDays: number | null;
  components: {
    base: "fundamentals" | "winner";
    polling: "none" | "standard" | "persistent";
    uncertainty: string;
  };
  uncertaintyLabel: string;
  uncertaintyDescription: string;
  historical: boolean;
  sensitivity: boolean;
  simulationCount: number;
  /** False for retrospective publication/roster proxies. */
  strictAsOf: boolean;
}

export interface ExplorerRunEntry extends ExplorerRun {
  href: string;
  sha256: string;
}

export interface ExplorerRegistry {
  schemaVersion: 1;
  defaults: Record<string, string>;
  runs: ExplorerRunEntry[];
}

export interface ExplorerMember {
  id: string;
  name: string;
  party: string;
  incumbent?: boolean;
  sourceUrls: string[];
}

export interface ExplorerOutcome {
  id: string;
  name: string;
  party: string;
  kind: "candidate" | "group";
  members: ExplorerMember[];
  winProbability: number;
  /** Present only when the saved run provides this exact outcome's share law. */
  share?: {
    mean: number;
    quantiles: ShareQuantiles;
    denominator: string;
  };
}

export interface ExplorerRace {
  id: string;
  state: string;
  seatId: string;
  electionLabel: string;
  outcomes: ExplorerOutcome[];
  pollCount?: number;
  notes: string[];
  observed?: {
    outcomeId: string;
    winnerName: string;
    winnerParty: string;
    sourceUrls: string[];
  };
}

export type ExplorerCaucus = "D" | "R" | "U";

export interface ExplorerPerformance {
  observedSeats: Record<ExplorerCaucus, number>;
  observedControl: ExplorerCaucus;
  dSeatCrps: number;
  rSeatCrps: number;
  /** Categorical D/R/U control Brier; lower is better. */
  controlBrier: number;
  controlLogScore: number | null;
  intervals: {
    party: ExplorerCaucus;
    level: number;
    lower: number;
    upper: number;
    covered: boolean;
  }[];
}

export interface ExplorerScenario {
  id: string;
  label: string;
  description: string;
  thresholds: { D: number; R: number };
  retainedSeats: Record<ExplorerCaucus, number>;
  control: ControlScenario;
  seats: Record<ExplorerCaucus, SeatDistribution>;
  performance?: ExplorerPerformance;
}

export interface ExplorerBundle {
  schemaVersion: 1;
  run: ExplorerRun;
  races: ExplorerRace[];
  scenarios: ExplorerScenario[];
  defaultScenarioId: string;
  /** Only the present principal-pair runs support the saved margin-rank statistic. */
  tipping: ModelTippingSummary | null;
  limitations: string[];
  historicalNote: string | null;
  provenance: {
    /** Public artifact labels and hashes, without private filesystem locations. */
    sources: Record<string, string>;
  };
}
