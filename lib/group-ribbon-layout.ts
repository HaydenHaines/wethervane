import type { HouseGroupBloc, HouseGroupRace } from "./house-group-result.ts";
import { layoutSerpentine, type SerpentineLayout } from "./serpentine-layout.ts";

export interface GroupRibbonEntry {
  id: string;
  districtKey: string;
  probabilities: { D: number; R: number; O: number };
  leaders: HouseGroupBloc[];
  /** Display ordering only. This is not a vote margin or a tipping statistic. */
  balance: number;
  probabilitySource: "modeled" | "structural";
}
export interface GroupRibbonLayout {
  main: SerpentineLayout<GroupRibbonEntry>;
  other: SerpentineLayout<GroupRibbonEntry>;
  mixedTies: GroupRibbonEntry[];
  unavailable: { id: string; districtKey: string; reason: string }[];
}

const blocs: HouseGroupBloc[] = ["D", "R", "O"];
const compareDistrict = (a: { districtKey: string }, b: { districtKey: string }) => a.districtKey.localeCompare(b.districtKey);
function validDistribution(value: unknown): value is { D: number; R: number; O: number } {
  if (!value || typeof value !== "object") return false;
  const p = value as Record<string, unknown>;
  return Object.keys(p).length === 3 && blocs.every(k => typeof p[k] === "number" && Number.isFinite(p[k]) && (p[k] as number) >= 0 && (p[k] as number) <= 1)
    && Math.abs((p.D as number) + (p.R as number) + (p.O as number) - 1) <= 1e-12;
}

/** Orders only explicitly supplied, normalized group probabilities; invalid/future rows stay unranked. */
export function layoutGroupRibbon(races: readonly HouseGroupRace[], columns: number): GroupRibbonLayout {
  layoutSerpentine([], columns);
  const seen = new Set<string>(), main: GroupRibbonEntry[] = [], other: GroupRibbonEntry[] = [], mixedTies: GroupRibbonEntry[] = [], unavailable: GroupRibbonLayout["unavailable"] = [];
  for (const race of races) {
    if (!race || typeof race.id !== "string" || typeof race.districtKey !== "string" || !race.id || !race.districtKey) throw new TypeError("Group ribbon rows need an id and district key.");
    if (seen.has(race.id)) throw new TypeError(`Duplicate group-ribbon id: ${race.id}`);
    seen.add(race.id);
    if (!validDistribution(race.probabilities)) {
      unavailable.push({ id: race.id, districtKey: race.districtKey, reason: "A valid group prediction is not available." });
      continue;
    }
    const probabilities = { D: race.probabilities.D, R: race.probabilities.R, O: race.probabilities.O };
    const maximum = Math.max(...blocs.map(b => probabilities[b]));
    const leaders = blocs.filter(b => probabilities[b] === maximum);
    const entry: GroupRibbonEntry = { id: race.id, districtKey: race.districtKey, probabilities, leaders, balance: probabilities.D - probabilities.R, probabilitySource: race.structural ? "structural" : "modeled" };
    if (leaders.length === 1 && leaders[0] === "O") other.push(entry);
    else if (leaders.includes("O")) mixedTies.push(entry);
    else main.push(entry);
  }
  main.sort((a, b) => b.balance - a.balance || compareDistrict(a, b));
  other.sort((a, b) => b.probabilities.O - a.probabilities.O || compareDistrict(a, b));
  mixedTies.sort(compareDistrict); unavailable.sort(compareDistrict);
  return { main: layoutSerpentine(main, columns), other: layoutSerpentine(other, columns), mixedTies, unavailable };
}
