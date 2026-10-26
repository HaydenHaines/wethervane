import type { ElectionNightForecast, ElectionNightRace } from "./election-night-forecast.ts";

/** A deterministic, auditable point call; it is not a joint simulation or a control probability. */
export function electionNightCalls(forecast: ElectionNightForecast) {
  const scenario = forecast.chamberScenario;
  const expectedScenario = forecast.chamber === "senate" ? "campaign_pledge_unaligned" : "party_majority_others_unassigned";
  const expectedTotal = forecast.chamber === "senate" ? 100 : 435;
  const seats = scenario && scenario.id === expectedScenario ? { ...scenario.retainedSeats } : null;
  if (seats) {
    for (const race of forecast.races) {
      const outcome = race.outcomes.find(item => item.id === race.modalPick.outcomeId);
      if (!outcome || !["D", "R", "I", "O"].includes(outcome.party)) return { seats: null, control: null, competitive: competitiveRaces(forecast.races) };
      const caucus = outcome.party === "D" || outcome.party === "R" ? outcome.party : "U";
      seats[caucus]++;
    }
    if (seats.D + seats.R + seats.U !== expectedTotal) return { seats: null, control: null, competitive: competitiveRaces(forecast.races) };
  }
  const control: "D" | "R" | "U" | null = seats && scenario ? seats.D >= scenario.thresholds.D ? "D" : seats.R >= scenario.thresholds.R ? "R" : "U" : null;
  return { seats, control, competitive: competitiveRaces(forecast.races) };
}

/** Include all likely/tossup races and any very narrow modeled pair, even if its win chance exceeds 90%. */
export function competitiveRaces(races: ElectionNightRace[]): ElectionNightRace[] {
  return races.filter(race => race.modalPick.probability <= .9 || (race.pair !== null && Math.abs(race.pair.meanMarginPp) <= 5))
    .sort((a, b) => a.modalPick.probability - b.modalPick.probability || a.id.localeCompare(b.id));
}
