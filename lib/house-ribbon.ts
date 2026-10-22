import { layoutSerpentine } from "./serpentine-layout.ts";
import type { RibbonDirection } from "./serpentine-layout.ts";
export type { RibbonDirection } from "./serpentine-layout.ts";

export interface HouseRaceRibbonInput {
  id: string;
  districtKey: string;
  democraticWinProbability: number;
}

export interface HouseRibbonRow extends HouseRaceRibbonInput {
  /** Zero-based position in the probability-sorted race list. */
  index: number;
  /** Zero-based CSS grid row and column coordinates. */
  row: number;
  column: number;
  /** Horizontal reading direction for this row in the ribbon path. */
  direction: RibbonDirection;
  /** Row-major visual position; gaps in a partial reversed row are intentional. */
  visualIndex: number;
}

export interface HouseRibbonLayout {
  rowCount: number;
  columnCount: number;
  rows: HouseRibbonRow[];
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/**
 * Order House races by Democratic win probability and place them on a
 * row-major serpentine grid. `columns` is the maximum width of each row.
 * This layout makes no chamber-control or
 * majority/tipping inference; any such marker must be validated upstream.
 */
export function layoutHouseRibbon(
  races: readonly HouseRaceRibbonInput[],
  columns: number,
): HouseRibbonLayout {
  if (!Number.isInteger(columns) || columns < 1) {
    throw new RangeError("Ribbon columns must be a positive integer.");
  }
  const ids = new Set<string>();
  const districtKeys = new Set<string>();
  for (const race of races) {
    if (!race.id || !race.districtKey) {
      throw new TypeError("Every House race needs an ID and district key.");
    }
    if (ids.has(race.id))
      throw new TypeError(`Duplicate House race ID: ${race.id}`);
    if (districtKeys.has(race.districtKey))
      throw new TypeError(`Duplicate House district key: ${race.districtKey}`);
    if (
      !Number.isFinite(race.democraticWinProbability) ||
      race.democraticWinProbability < 0 ||
      race.democraticWinProbability > 1
    ) {
      throw new RangeError(
        `Invalid Democratic win probability for ${race.id}.`,
      );
    }
    ids.add(race.id);
    districtKeys.add(race.districtKey);
  }

  const ordered = [...races].sort(
    (left, right) =>
      right.democraticWinProbability - left.democraticWinProbability ||
      compareText(left.districtKey, right.districtKey) ||
      compareText(left.id, right.id),
  );
  return layoutSerpentine(ordered, columns);
}
