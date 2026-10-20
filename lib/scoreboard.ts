import type {
  ExplorerBundle,
  ExplorerCaucus,
  ExplorerOutcome,
  ExplorerRace,
  ExplorerScenario,
} from "./explorer-types";

const PARTIES: ExplorerCaucus[] = ["D", "R", "U"];
const MASS_TOLERANCE = 1e-8;

export interface ScoreboardCoverage {
  total: number;
  /** Races with an observed-outcome record, including any rejected records. */
  observed: number;
  scored: number;
  missingRaceIds: string[];
  invalidRaces: { raceId: string; reason: string }[];
}

export interface FavoriteScore {
  correct: number;
  /** Only observed races with a unique maximum probability. */
  decisive: number;
  /** Sum of unique-maximum probabilities on that same decisive observed set. */
  expectedCorrect: number;
  tiedRaceIds: string[];
  missRaceIds: string[];
}

export interface ScoreboardSeats {
  party: ExplorerCaucus;
  expected: number;
  median: number;
  p05: number;
  p95: number;
  observed: number | null;
  /** Forecast expectation minus observed seats. */
  error: number | null;
}

export interface Scoreboard {
  status: "available" | "unavailable";
  reason: string | null;
  coverage: ScoreboardCoverage;
  favorites: FavoriteScore | null;
  /** Raw multiclass Brier, sum over outcomes then mean over scored races; 0–2. */
  brier: { mean: number; count: number } | null;
  seats: ScoreboardSeats[];
  control: {
    observed: ExplorerCaucus;
    observedProbability: number;
    /** Multiple categories mean an exact tie at the stored probabilities. */
    mostLikely: ExplorerCaucus[];
  } | null;
  issues: string[];
}

export interface ScoreboardComparison {
  status: "matched" | "unavailable";
  reason: string | null;
  /** Selected minus baseline; negative is a better Brier score. */
  brierDifference: number | null;
  favoritesDifference: number | null;
  expectedCorrectDifference: number | null;
}

function probability(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function raceProblem(race: ExplorerRace): string | null {
  if (!Array.isArray(race.outcomes) || !race.outcomes.length)
    return "No outcome distribution is available.";
  const ids = race.outcomes.map((outcome) => outcome.id);
  if (ids.some((id) => typeof id !== "string" || !id) || new Set(ids).size !== ids.length)
    return "Outcome identifiers are missing or duplicated.";
  if (!race.outcomes.every((outcome) => probability(outcome.winProbability)))
    return "Outcome probabilities must be finite values between zero and one.";
  if (Math.abs(sum(race.outcomes.map((outcome) => outcome.winProbability)) - 1) > MASS_TOLERANCE)
    return "Outcome probabilities do not sum to one.";
  const memberIds: string[] = [];
  for (const outcome of race.outcomes) {
    if (!Array.isArray(outcome.members) || !outcome.members.length || !["candidate", "group"].includes(outcome.kind))
      return "Outcome membership is unavailable.";
    if (outcome.kind === "candidate" && outcome.members.length !== 1)
      return "An individual-candidate outcome has multiple members.";
    for (const member of outcome.members) {
      if (typeof member.id !== "string" || !member.id)
        return "An outcome member has no identifier.";
      memberIds.push(member.id);
    }
  }
  if (new Set(memberIds).size !== memberIds.length)
    return "Candidate membership overlaps outcome groups.";
  if (race.observed && !ids.includes(race.observed.outcomeId))
    return "The observed outcome is outside the forecast's support.";
  return null;
}

function deriveChamber(scenario: ExplorerScenario): Pick<Scoreboard, "seats" | "control" | "issues"> {
  const issues: string[] = [];
  const seats: ScoreboardSeats[] = [];
  for (const party of PARTIES) {
    const distribution = scenario.seats?.[party];
    const quantiles = distribution?.seatQuantiles;
    if (!distribution || !Array.isArray(distribution.pmf) || !distribution.pmf.length ||
      !distribution.pmf.every(probability) ||
      Math.abs(sum(distribution.pmf) - 1) > MASS_TOLERANCE ||
      !finite(distribution.expectedSeats) || !quantiles ||
      ![quantiles.p05, quantiles.p50, quantiles.p95].every(finite) ||
      quantiles.p05 > quantiles.p50 || quantiles.p50 > quantiles.p95 ||
      quantiles.p05 < 0 || quantiles.p95 > distribution.pmf.length - 1 ||
      Math.abs(sum(distribution.pmf.map((p, count) => p * count)) - distribution.expectedSeats) > MASS_TOLERANCE) {
      issues.push(`${party} seat distribution is invalid or unavailable.`);
      continue;
    }
    const actual = scenario.performance?.observedSeats?.[party];
    const observed = finite(actual) && Number.isInteger(actual) && actual >= 0 ? actual : null;
    if (scenario.performance && observed === null)
      issues.push(`${party} observed seat count is invalid or unavailable.`);
    seats.push({
      party,
      expected: distribution.expectedSeats,
      median: quantiles.p50,
      p05: quantiles.p05,
      p95: quantiles.p95,
      observed,
      error: observed === null ? null : distribution.expectedSeats - observed,
    });
  }
  const observed = scenario.performance?.observedControl;
  const values = [
    scenario.control?.democraticMajorityProbability,
    scenario.control?.republicanMajorityProbability,
    scenario.control?.unresolvedOrganizingProbability,
  ];
  let control: Scoreboard["control"] = null;
  if (observed && PARTIES.includes(observed)) {
    if (!values.every(probability) || Math.abs(sum(values) - 1) > MASS_TOLERANCE) {
      issues.push("Chamber control probabilities are invalid or unavailable.");
    } else {
      const maximum = Math.max(...values);
      control = {
        observed,
        observedProbability: values[PARTIES.indexOf(observed)],
        mostLikely: PARTIES.filter((_, index) => values[index] === maximum),
      };
    }
  } else if (scenario.performance) {
    issues.push("The observed control category is invalid or unavailable.");
  }
  return { seats, control, issues };
}

/** Score categorical forecast outcomes, including grouped outcomes, without splitting groups. */
export function deriveRaceScoreboard(races: ExplorerRace[]): Scoreboard {
  const coverage: ScoreboardCoverage = {
    total: races.length, observed: 0, scored: 0, missingRaceIds: [], invalidRaces: [],
  };
  const favorites: FavoriteScore = {
    correct: 0, decisive: 0, expectedCorrect: 0, tiedRaceIds: [], missRaceIds: [],
  };
  const counts = new Map<string, number>();
  for (const race of races) counts.set(race.id, (counts.get(race.id) || 0) + 1);
  let totalBrier = 0;
  for (const race of races) {
    if (!race.observed) {
      coverage.missingRaceIds.push(race.id);
      continue;
    }
    coverage.observed++;
    const problem = counts.get(race.id)! > 1 ? "The race identifier is duplicated." : raceProblem(race);
    if (problem) {
      coverage.invalidRaces.push({ raceId: race.id, reason: problem });
      continue;
    }
    coverage.scored++;
    totalBrier += sum(race.outcomes.map((outcome) =>
      (outcome.winProbability - (outcome.id === race.observed!.outcomeId ? 1 : 0)) ** 2,
    ));
    const maximum = Math.max(...race.outcomes.map((outcome) => outcome.winProbability));
    const leaders = race.outcomes.filter((outcome) => outcome.winProbability === maximum);
    if (leaders.length !== 1) {
      favorites.tiedRaceIds.push(race.id);
      continue;
    }
    favorites.decisive++;
    favorites.expectedCorrect += maximum;
    if (leaders[0].id === race.observed.outcomeId) favorites.correct++;
    else favorites.missRaceIds.push(race.id);
  }
  const available = coverage.scored > 0;
  return {
    status: available ? "available" : "unavailable",
    reason: available ? null : coverage.observed > 0
      ? "No observed race has a valid matching outcome distribution."
      : "Observed race outcomes are unavailable; no historical score can be calculated.",
    coverage,
    favorites: available ? favorites : null,
    brier: available ? { mean: totalBrier / coverage.scored, count: coverage.scored } : null,
    seats: [], control: null, issues: [],
  };
}

/** Chamber summaries require their own joint result, independently of race scores. */
export function deriveScoreboard(bundle: ExplorerBundle, scenario: ExplorerScenario): Scoreboard {
  return { ...deriveRaceScoreboard(bundle.races), ...deriveChamber(scenario) };
}

function outcomeSupport(outcome: ExplorerOutcome): unknown {
  return {
    id: outcome.id, kind: outcome.kind, party: outcome.party,
    denominator: outcome.share?.denominator ?? null,
    members: outcome.members.map((member) => ({ id: member.id, name: member.name, party: member.party }))
      .sort((a, b) => a.id.localeCompare(b.id)),
  };
}

function raceSupport(race: ExplorerRace): unknown {
  return {
    id: race.id, state: race.state, seatId: race.seatId,
    outcomes: [...race.outcomes].sort((a, b) => a.id.localeCompare(b.id)).map(outcomeSupport),
  };
}

function observedTruth(bundle: ExplorerBundle): unknown {
  return [...bundle.races].sort((a, b) => a.id.localeCompare(b.id)).map((race) => ({
    id: race.id,
    observed: race.observed ? {
      outcomeId: race.observed.outcomeId,
      winnerName: race.observed.winnerName,
      winnerParty: race.observed.winnerParty,
    } : null,
  }));
}

function mismatch(reason: string): ScoreboardComparison {
  return {
    status: "unavailable", reason,
    brierDifference: null, favoritesDifference: null, expectedCorrectDifference: null,
  };
}

/** Match the entire score target and observed truth before returning selected-minus-baseline deltas. */
export function compareScoreboards(
  bundle: ExplorerBundle,
  scenario: ExplorerScenario,
  baselineBundle: ExplorerBundle,
  baselineScenario: ExplorerScenario,
): ScoreboardComparison {
  if (bundle.run.year !== baselineBundle.run.year)
    return mismatch("Election years differ.");
  if (bundle.run.dateId !== baselineBundle.run.dateId ||
    bundle.run.forecastAt !== baselineBundle.run.forecastAt ||
    bundle.run.horizonDays !== baselineBundle.run.horizonDays)
    return mismatch("Forecast dates or horizons differ.");
  if (bundle.run.cutoffAt !== baselineBundle.run.cutoffAt)
    return mismatch("Information cutoffs differ.");
  if (bundle.run.targetId !== baselineBundle.run.targetId)
    return mismatch("Forecast targets differ.");
  if (bundle.run.lawId !== baselineBundle.run.lawId)
    return mismatch("Forecast laws differ.");
  if (scenario.id !== baselineScenario.id)
    return mismatch("Caucus scenarios differ.");
  if (PARTIES.some((party) => scenario.retainedSeats?.[party] !== baselineScenario.retainedSeats?.[party]) ||
    scenario.thresholds.D !== baselineScenario.thresholds.D || scenario.thresholds.R !== baselineScenario.thresholds.R)
    return mismatch("Retained seats or control thresholds differ.");
  const selected = deriveScoreboard(bundle, scenario);
  const baseline = deriveScoreboard(baselineBundle, baselineScenario);
  if (selected.coverage.invalidRaces.length || baseline.coverage.invalidRaces.length)
    return mismatch("Invalid observed outcome distributions prevent a matched comparison.");
  try {
    const support = (value: ExplorerBundle) => JSON.stringify([...value.races]
      .sort((a, b) => a.id.localeCompare(b.id)).map(raceSupport));
    if (support(bundle) !== support(baselineBundle))
      return mismatch("Race, outcome, or candidate-member support differs.");
  } catch {
    return mismatch("Race or candidate-member support is invalid.");
  }
  if (JSON.stringify(observedTruth(bundle)) !== JSON.stringify(observedTruth(baselineBundle)))
    return mismatch("Observed race outcomes differ.");
  const actual = scenario.performance;
  const baseActual = baselineScenario.performance;
  if (actual?.observedControl !== baseActual?.observedControl ||
    PARTIES.some((party) => actual?.observedSeats?.[party] !== baseActual?.observedSeats?.[party]))
    return mismatch("Observed chamber truth differs.");
  if (!selected.brier || !baseline.brier || !selected.favorites || !baseline.favorites)
    return mismatch("Matching observed outcomes are unavailable.");
  return {
    status: "matched", reason: null,
    brierDifference: selected.brier.mean - baseline.brier.mean,
    favoritesDifference: selected.favorites.correct - baseline.favorites.correct,
    expectedCorrectDifference: selected.favorites.expectedCorrect - baseline.favorites.expectedCorrect,
  };
}
