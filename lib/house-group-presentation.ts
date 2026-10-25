import type { HouseGroupBloc, HouseGroupRace } from "./house-group-result";
import type { ExplorerRace } from "./explorer-types";

export const groupName = (id: HouseGroupBloc) => ({ D: "Democratic group", R: "Republican group", O: "Other candidates and write-ins" })[id];
export const houseFundamentalsLabel = (id: string) => ({ house_history_refreshed_v1: "Fixed history", house_presidential_slope_v1: "Presidential slope", house_relative_lean_v1: "Relative lean", house_relative_context_v1: "Relative context" })[id] || id;
export const groupFundamentalsId = (configuration: string) => configuration.replace(/-loading-1$/, "");

/** Map-only presentation; preserves group probabilities and never allocates them among members. */
export function groupRaceForMap(race: Pick<HouseGroupRace,"id"|"cycle"|"districtKey"|"groups"|"observed">): ExplorerRace {
  return {
    id: race.id, state: race.districtKey.slice(0, 2), seatId: race.districtKey,
    electionLabel: `${race.cycle} U.S. House — ${race.districtKey}`,
    outcomes: race.groups.map(group => ({
      id: group.id, party: group.id,
      name: group.individualCandidateProbability !== null ? group.members[0].name : groupName(group.id),
      kind: group.individualCandidateProbability !== null ? "candidate" : "group",
      members: [], winProbability: group.winProbability,
    })), notes: [],
    ...(race.observed.status === "observed" ? { observed: {
      outcomeId: race.observed.groupId, winnerName: race.observed.winnerNames.join("; ") || groupName(race.observed.groupId),
      winnerParty: race.observed.groupId, sourceUrls: [],
    } } : {}),
  };
}
