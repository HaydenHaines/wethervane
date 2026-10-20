import assert from "node:assert/strict";
import test from "node:test";

import { layoutHouseRibbon } from "../lib/house-ribbon.ts";

function rows(count, probability = (index) => (count - index) / (count + 1)) {
  return Array.from({ length: count }, (_, index) => ({
    id: `race-${String(index).padStart(3, "0")}`,
    districtKey: `district-${String(index).padStart(3, "0")}`,
    democraticWinProbability: probability(index),
  }));
}

test("one race gets a single top-left cell", () => {
  const result = layoutHouseRibbon(rows(1), 6);
  assert.equal(result.rowCount, 1);
  assert.equal(result.columnCount, 1);
  assert.deepEqual(result.rows[0], {
    ...rows(1)[0],
    index: 0,
    row: 0,
    column: 0,
    direction: "left-to-right",
    visualIndex: 0,
  });
});

test("16 races form a two-row serpentine ribbon with explicit indexes", () => {
  const result = layoutHouseRibbon(rows(16), 8);
  assert.equal(result.rowCount, 2);
  assert.equal(result.columnCount, 8);
  assert.deepEqual(result.rows.slice(0, 8).map(({ row, column, direction, index, visualIndex }) =>
    ({ row, column, direction, index, visualIndex })), [
    ...Array.from({ length: 8 }, (_, index) => ({
      row: 0, column: index, direction: "left-to-right", index, visualIndex: index,
    })),
  ]);
  assert.deepEqual(result.rows.slice(8).map(({ row, column, direction, index, visualIndex }) =>
    ({ row, column, direction, index, visualIndex })), [
    ...Array.from({ length: 8 }, (_, offset) => ({
      row: 1, column: 7 - offset, direction: "right-to-left", index: 8 + offset,
      visualIndex: 8 + 7 - offset,
    })),
  ]);
});

test("435 races fit 16 columns and a partial reversed row is right-aligned", () => {
  const result = layoutHouseRibbon(rows(435), 16);
  assert.equal(result.rowCount, 28);
  assert.equal(result.columnCount, 16);
  assert.equal(result.rows.length, 435);
  assert.equal(result.rows[0].index, 0);
  assert.equal(result.rows[0].column, 0);
  assert.equal(result.rows[15].index, 15);
  assert.equal(result.rows[15].column, 15);
  assert.equal(result.rows[16].index, 16);
  assert.equal(result.rows[16].row, 1);
  assert.equal(result.rows[16].column, 15);
  assert.equal(result.rows[16].direction, "right-to-left");
  assert.equal(result.rows.at(-1).index, 434);
  assert.equal(result.rows.at(-1).row, 27);
  assert.equal(result.rows.at(-1).column, 13);
  assert.equal(result.rows.at(-1).visualIndex, 445);
  assert.ok(result.rows.every((row) => !("controlProbability" in row) && !("majorityIndex" in row)));
});

test("probabilities sort descending and ties break by district key then ID", () => {
  const input = [
    { id: "z", districtKey: "D2", democraticWinProbability: 0.7 },
    { id: "b", districtKey: "D1", democraticWinProbability: 0.7 },
    { id: "a", districtKey: "D3", democraticWinProbability: 0.7 },
    { id: "c", districtKey: "D4", democraticWinProbability: 0.9 },
  ];
  const result = layoutHouseRibbon(input, 4);
  assert.deepEqual(result.rows.map((row) => row.id), ["c", "b", "z", "a"]);
  assert.deepEqual(input.map((row) => row.id), ["z", "b", "a", "c"]);
});

test("rejects invalid columns, IDs, district keys and probabilities", () => {
  assert.throws(() => layoutHouseRibbon(rows(1), 0), RangeError);
  assert.throws(() => layoutHouseRibbon(rows(1), 1.5), RangeError);
  assert.throws(() => layoutHouseRibbon([
    { id: "same", districtKey: "D1", democraticWinProbability: 0.5 },
    { id: "same", districtKey: "D2", democraticWinProbability: 0.4 },
  ], 2), /Duplicate House race ID/);
  assert.throws(() => layoutHouseRibbon([
    { id: "a", districtKey: "D1", democraticWinProbability: 0.5 },
    { id: "b", districtKey: "D1", democraticWinProbability: 0.4 },
  ], 2), /Duplicate House district key/);
  for (const p of [NaN, Infinity, -0.01, 1.01]) {
    assert.throws(() => layoutHouseRibbon([
      { id: "bad", districtKey: "D1", democraticWinProbability: p },
    ], 1), RangeError);
  }
});
