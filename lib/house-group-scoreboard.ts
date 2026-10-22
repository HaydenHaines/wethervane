import type { HouseGroupBloc, HouseGroupResult } from "./house-group-result.ts";

export interface HouseGroupScoreSummary {
  forecastRaces: number;
  scoredRaces: number;
  unscoredRaces: number;
  uniqueFavorites: number;
  correctUniqueFavorites: number;
  topChanceTies: number;
  categoricalBrier: number | null;
  strata: HouseGroupResult["raceScores"];
}

/** Derives only exact-top favorite counts; score metrics come from the verified result. */
export function summarizeHouseGroupResult(result: HouseGroupResult): HouseGroupScoreSummary {
  const scored = result.races.filter(race => race.groupScoreEligible && race.observed.status === "observed");
  let uniqueFavorites = 0, correctUniqueFavorites = 0, topChanceTies = 0;
  for (const race of scored) {
    const max = Math.max(race.probabilities.D, race.probabilities.R, race.probabilities.O);
    const leaders = (["D", "R", "O"] as const).filter((bloc: HouseGroupBloc) => race.probabilities[bloc] === max);
    if (leaders.length > 1) topChanceTies++;
    else {
      uniqueFavorites++;
      if (race.observed.status === "observed" && race.observed.groupId === leaders[0]) correctUniqueFavorites++;
    }
  }
  return {
    forecastRaces: result.races.length,
    scoredRaces: scored.length,
    unscoredRaces: result.races.length - scored.length,
    uniqueFavorites,
    correctUniqueFavorites,
    topChanceTies,
    categoricalBrier: result.raceScores.find(score => score.stratum === "all_scored")?.categoricalBrier ?? null,
    strata: result.raceScores,
  };
}
