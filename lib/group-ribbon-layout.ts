import type { HouseGroupBloc } from "./house-group-result.ts";
import { layoutSerpentine, type SerpentineLayout } from "./serpentine-layout.ts";

/** Minimal supplied result; presentation does not depend on a historical model family. */
export interface GroupRibbonRace {
  id: string;
  districtKey: string;
  probabilities: unknown;
  structural?: boolean;
}

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

export interface RibbonChamber {
  totalSeats: number;
  majorityThreshold: number;
}
export interface GroupRibbonMajorityPosition {
  id: string;
  districtKey: string;
  from: ("D" | "R")[];
}

/** Counts seats from each end of the main path; never estimates a tipping probability.
 * Other-leading and tied-Other districts remain outside both parties' counts.
 * An explicit full-chamber contract is required; partial/invalid fields have no marker.
 */
export function groupRibbonMajorityPositions(layout: GroupRibbonLayout, chamber?: RibbonChamber): GroupRibbonMajorityPosition[] {
  if (!chamber) return [];
  const { totalSeats, majorityThreshold } = chamber;
  if (!Number.isInteger(totalSeats) || totalSeats < 1 || !Number.isInteger(majorityThreshold) ||
    majorityThreshold <= totalSeats / 2 || majorityThreshold > totalSeats) return [];
  const ready = [...layout.main.rows, ...layout.other.rows, ...layout.mixedTies];
  if (layout.unavailable.length || ready.length !== totalSeats ||
    new Set(ready.map(row => row.districtKey)).size !== totalSeats || layout.main.rows.length < majorityThreshold) return [];
  const positions: GroupRibbonMajorityPosition[] = [];
  for (const from of ["D", "R"] as const) {
    const row = layout.main.rows[from === "D" ? majorityThreshold - 1 : layout.main.rows.length - majorityThreshold];
    const existing = positions.find(position => position.id === row.id);
    if (existing) existing.from.push(from);
    else positions.push({ id: row.id, districtKey: row.districtKey, from: [from] });
  }
  return positions;
}

const blocs: HouseGroupBloc[] = ["D", "R", "O"];
const compareDistrict = (a: { districtKey: string }, b: { districtKey: string }) => a.districtKey.localeCompare(b.districtKey);
function validDistribution(value: unknown): value is { D: number; R: number; O: number } {
  if (!value || typeof value !== "object") return false;
  const p = value as Record<string, unknown>;
  return Object.keys(p).length === 3 && blocs.every(k => typeof p[k] === "number" && Number.isFinite(p[k]) && (p[k] as number) >= 0 && (p[k] as number) <= 1)
    // Match native result rounding tolerance without renormalizing the supplied chances.
    && Math.abs((p.D as number) + (p.R as number) + (p.O as number) - 1) <= 1e-8;
}

/** Orders only explicitly supplied, normalized group probabilities; invalid/future rows stay unranked. */
export function layoutGroupRibbon(races: readonly GroupRibbonRace[], columns: number): GroupRibbonLayout {
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
