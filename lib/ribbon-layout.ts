import type { RaceForecast } from "./site-types";
import type { TippingCaucus } from "./tipping-types";

export type CaucusMembership = Pick<
  TippingCaucus,
  | "thresholdSeats"
  | "retainedSeats"
  | "guaranteedRaceIds"
  | "eligibleCandidateByRace"
>;

export interface CaucusLayout {
  thresholdSeats: number;
  retainedSeats: number;
  guaranteedRaceIds: string[];
  eligibleRaceIds: string[];
  requiredContestedWins: number;
  status: "reachable" | "already_reached" | "unreachable";
  statusReason: string | null;
  projectedTippingRaceId: string | null;
  projectedMarginPp: number | null;
}

export interface RibbonLayout {
  raceOrder: string[];
  caucuses: { D: CaucusLayout; R: CaucusLayout };
}

const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/** Rank expected caucus-candidate shares, not winning probabilities or party labels. */
function caucusLayout(
  races: readonly RaceForecast[],
  membership: CaucusMembership,
): CaucusLayout {
  const { thresholdSeats, retainedSeats } = membership;
  if (
    ![thresholdSeats, retainedSeats].every(
      (value) => Number.isInteger(value) && value >= 0,
    )
  )
    throw new Error("Caucus seat counts must be nonnegative integers.");
  const present = new Set(races.map((race) => race.id));
  const guaranteedRaceIds = [...new Set(membership.guaranteedRaceIds)]
    .filter((id) => present.has(id))
    .sort(byId);
  const guaranteed = new Set(guaranteedRaceIds);
  const eligible = races
    .flatMap((race) => {
      if (guaranteed.has(race.id)) return [];
      const candidateId = membership.eligibleCandidateByRace[race.id];
      if (!candidateId) return [];
      const candidateIndex = race.candidates.findIndex(
        (candidate) => candidate.id === candidateId,
      );
      if (candidateIndex < 0)
        throw new Error(`Unknown eligible candidate in ${race.id}.`);
      return [
        {
          id: race.id,
          share:
            candidateIndex === 0
              ? race.firstShareMean
              : 1 - race.firstShareMean,
        },
      ];
    })
    .sort((a, b) => b.share - a.share || byId(a.id, b.id));
  const requiredContestedWins =
    thresholdSeats - retainedSeats - guaranteedRaceIds.length;
  const common = {
    thresholdSeats,
    retainedSeats,
    guaranteedRaceIds,
    eligibleRaceIds: eligible.map((race) => race.id),
    requiredContestedWins,
  };
  if (requiredContestedWins <= 0)
    return {
      ...common,
      status: "already_reached",
      statusReason:
        "Retained and guaranteed seats already reach the control threshold.",
      projectedTippingRaceId: null,
      projectedMarginPp: null,
    };
  if (requiredContestedWins > eligible.length)
    return {
      ...common,
      status: "unreachable",
      statusReason: `Needs ${requiredContestedWins} contested wins, but only ${eligible.length} eligible races are available.`,
      projectedTippingRaceId: null,
      projectedMarginPp: null,
    };
  const tipping = eligible[requiredContestedWins - 1];
  return {
    ...common,
    status: "reachable",
    statusReason: null,
    projectedTippingRaceId: tipping.id,
    projectedMarginPp: 100 * (2 * tipping.share - 1),
  };
}

/** Layout depends on the races and declared caucus membership. Draw frequencies are never inferred. */
export function deriveRibbonLayout(
  races: readonly RaceForecast[],
  scenario: { D: CaucusMembership; R: CaucusMembership },
): RibbonLayout {
  if (new Set(races.map((race) => race.id)).size !== races.length)
    throw new Error("Ribbon race IDs must be unique.");
  if (
    races.some(
      (race) =>
        !Number.isFinite(race.firstShareMean) ||
        race.firstShareMean < 0 ||
        race.firstShareMean > 1 ||
        race.candidates.length !== 2,
    )
  )
    throw new Error(
      "A margin ribbon requires two principal candidates and a valid mean share per race.",
    );
  return {
    raceOrder: [...races]
      .sort((a, b) => b.firstShareMean - a.firstShareMean || byId(a.id, b.id))
      .map((race) => race.id),
    caucuses: {
      D: caucusLayout(races, scenario.D),
      R: caucusLayout(races, scenario.R),
    },
  };
}

/** Fixed readable tiles and endcaps determine when the ribbon needs an internal scroller. */
export function ribbonMinimumWidth(
  raceCount: number,
  tileWidth = 22,
  endcapWidth = 49,
  gap = 3,
): number {
  if (!Number.isInteger(raceCount) || raceCount < 0)
    throw new Error("Race count must be a nonnegative integer.");
  return endcapWidth * 2 + raceCount * tileWidth + (raceCount + 1) * gap;
}

export function ribbonSeatLabel(
  race: Pick<RaceForecast, "id" | "seatId">,
): string {
  if (/special/i.test(race.id + race.seatId)) return "SP";
  const year = race.id.match(/^\d{4}(?=[:\-]|$)/)?.[0];
  return year ? `’${year.slice(-2)}` : "REG";
}
