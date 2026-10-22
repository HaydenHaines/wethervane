import { houseWinnerView, type HouseWinnerFixture } from "./house-winner-view.ts";
import { layoutSerpentine, type SerpentineLayout } from "./serpentine-layout.ts";
export interface HouseGroupRibbonEntry {
  id: string;
  districtKey: string;
  leaders: ("D" | "R" | "O")[];
  probabilities: { D: number; R: number; O: number };
  probabilitySource: "modeled" | "structural";
  /** P(D) minus P(R), solely a display order; not a vote margin or tipping statistic. */
  balance: number;
}
export interface HouseGroupRibbonLayout {
  main: SerpentineLayout<HouseGroupRibbonEntry>;
  other: SerpentineLayout<HouseGroupRibbonEntry>;
  mixedTies: HouseGroupRibbonEntry[];
  unavailable: { id: string; districtKey: string; reason: string }[];
}
const compareDistrict = (a: { districtKey: string }, b: { districtKey: string }) => a.districtKey < b.districtKey ? -1 : a.districtKey > b.districtKey ? 1 : 0;
/** Private fixture adapter. Other-leading races and O top ties have no partisan rank. */
export function layoutHouseGroupRibbon(fixtures: readonly HouseWinnerFixture[], columns: number): HouseGroupRibbonLayout {
  // Validate columns even when every district has unavailable predictions.
  layoutSerpentine([], columns);
  const seen = new Set<string>(), main: HouseGroupRibbonEntry[] = [], other: HouseGroupRibbonEntry[] = [], mixedTies: HouseGroupRibbonEntry[] = [], unavailable: HouseGroupRibbonLayout["unavailable"] = [];
  for (const fixture of fixtures) {
    const view = houseWinnerView(fixture), districtKey = view.district;
    if (seen.has(districtKey)) throw new TypeError(`Duplicate House district key: ${districtKey}`);
    seen.add(districtKey);
    if (view.prediction.status === "unavailable") { unavailable.push({ id: districtKey, districtKey, reason: view.prediction.reason }); continue; }
    const probabilities = { D: 0, R: 0, O: 0 };
    // Only validated available distributions have complete declared support.
    for (const group of view.groups) probabilities[group.bloc] = view.prediction.status === "structural" ? 1 : view.prediction.probabilities.find(p => p.groupId === group.id)!.probability;
    const maximum = Math.max(...Object.values(probabilities));
    const leaders = (["D", "R", "O"] as const).filter(bloc => probabilities[bloc] === maximum);
    const entry: HouseGroupRibbonEntry = { id: districtKey, districtKey, leaders, probabilities, probabilitySource: view.prediction.status, balance: probabilities.D - probabilities.R };
    if (leaders.length === 1 && leaders[0] === "O") other.push(entry);
    else if (leaders.includes("O")) mixedTies.push(entry);
    else main.push(entry);
  }
  main.sort((a, b) => b.balance - a.balance || compareDistrict(a, b));
  other.sort((a, b) => b.probabilities.O - a.probabilities.O || compareDistrict(a, b));
  mixedTies.sort(compareDistrict); unavailable.sort(compareDistrict);
  return { main: layoutSerpentine(main, columns), other: layoutSerpentine(other, columns), mixedTies, unavailable };
}
