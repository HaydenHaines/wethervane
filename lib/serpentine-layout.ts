/** Coordinates for an already ordered sequence; no probability or ranking semantics. */
export type RibbonDirection = "left-to-right" | "right-to-left";
export interface SerpentinePosition {
  index: number;
  row: number;
  column: number;
  direction: RibbonDirection;
  visualIndex: number;
}
export interface SerpentineLayout<T> {
  rowCount: number;
  columnCount: number;
  rows: (T & SerpentinePosition)[];
}
export function layoutSerpentine<T extends object>(ordered: readonly T[], columns: number): SerpentineLayout<T> {
  if (!Number.isInteger(columns) || columns < 1) throw new RangeError("Ribbon columns must be a positive integer.");
  if (!ordered.length) return { rowCount: 0, columnCount: 0, rows: [] };
  return { rowCount: Math.ceil(ordered.length / columns), columnCount: Math.min(ordered.length, columns), rows: ordered.map((item, index) => {
    const row = Math.floor(index / columns), offset = index % columns;
    const direction: RibbonDirection = row % 2 === 0 ? "left-to-right" : "right-to-left";
    const column = direction === "left-to-right" ? offset : columns - 1 - offset;
    return { ...item, index, row, column, direction, visualIndex: row * columns + column };
  }) };
}
