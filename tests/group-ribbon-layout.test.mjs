import test from "node:test";
import assert from "node:assert/strict";
import { groupRibbonMajorityPositions, layoutGroupRibbon } from "../lib/group-ribbon-layout.ts";

function race(id, probabilities, structural = false) {
  return { id, districtKey: id, probabilities, structural };
}

test("main path orders by Democratic minus Republican probability and keeps D/R ties", () => {
  const result = layoutGroupRibbon([
    race("D-1", { D: .8, R: .1, O: .1 }),
    race("T-1", { D: .4, R: .4, O: .2 }),
    race("R-1", { D: .1, R: .8, O: .1 }),
  ], 2);
  assert.deepEqual(result.main.rows.map(row => row.id), ["D-1", "T-1", "R-1"]);
  assert.equal(result.main.rows[2].direction, "right-to-left");
  assert.equal(result.other.rows.length, 0);
});

test("strict O leads and ties involving O are outside the D/R rank", () => {
  const result = layoutGroupRibbon([
    race("O-strong", { D: .05, R: .15, O: .8 }),
    race("O-near", { D: .35, R: .3, O: .35 }),
    race("all-tie", { D: 1 / 3, R: 1 / 3, O: 1 / 3 }),
    race("D-leads", { D: .5, R: .3, O: .2 }),
  ], 4);
  assert.deepEqual(result.other.rows.map(row => row.id), ["O-strong"]);
  assert.deepEqual(result.mixedTies.map(row => row.id), ["all-tie", "O-near"]);
  assert.deepEqual(result.main.rows.map(row => row.id), ["D-leads"]);
});

test("invalid or absent probability values stay visible but unranked and never become zero", () => {
  const result = layoutGroupRibbon([
    race("bad-sum", { D: .7, R: .7, O: 0 }),
    race("future-missing", undefined),
    race("fixed-R", { D: 0, R: 1, O: 0 }, true),
  ], 3);
  assert.deepEqual(result.main.rows.map(row => row.id), ["fixed-R"]);
  assert.deepEqual(result.unavailable.map(row => row.id), ["bad-sum", "future-missing"]);
  assert.equal("probabilities" in result.unavailable[0], false);
  assert.equal(result.main.rows[0].probabilitySource, "structural");
});

test("435 rows stay accounted for and duplicate ids or invalid columns reject", () => {
  const races = Array.from({ length: 435 }, (_, i) => race(`R-${i}`, i % 2 ? { D: .55, R: .35, O: .1 } : { D: .1, R: .2, O: .7 }));
  const result = layoutGroupRibbon(races, 16);
  assert.equal(result.main.rows.length + result.other.rows.length + result.mixedTies.length + result.unavailable.length, 435);
  for (const segment of [result.main, result.other]) assert.equal(new Set(segment.rows.map(row => `${row.row}:${row.column}`)).size, segment.rows.length);
  assert.throws(() => layoutGroupRibbon([race("same", { D: .5, R: .3, O: .2 }), race("same", { D: .4, R: .4, O: .2 })], 2), /Duplicate/);
  assert.throws(() => layoutGroupRibbon([], 0), RangeError);
});

test("native rounding tolerance preserves probabilities without inventing normalization", () => {
  const probabilities = { D: .600000001, R: .3, O: .1 };
  const result = layoutGroupRibbon([
    race("rounding", probabilities),
    race("invalid", { D: .600001, R: .3, O: .1 }),
  ], 2);
  assert.deepEqual(result.main.rows[0].probabilities, probabilities);
  assert.deepEqual(result.unavailable.map(row => row.id), ["invalid"]);
});

const orderedRaces = count => Array.from({ length: count }, (_, i) =>
  race(`district-${String(i + 1).padStart(3, "0")}`, { D: 1 - i / count, R: i / count, O: 0 }));

test("majority markers follow supplied seats and ranking through every snake width", () => {
  for (const totalSeats of [1, 5, 16, 435]) {
    const races = orderedRaces(totalSeats), majorityThreshold = Math.floor(totalSeats / 2) + 1;
    for (const columns of [1, 3, 12, 48, 60]) {
      const layout = layoutGroupRibbon(races.toReversed(), columns);
      const positions = groupRibbonMajorityPositions(layout, { totalSeats, majorityThreshold });
      assert.equal(positions.length, totalSeats % 2 ? 1 : 2);
      for (const from of ["D", "R"]) {
        const index = from === "D" ? majorityThreshold - 1 : totalSeats - majorityThreshold;
        assert.equal(positions.find(row => row.from.includes(from)).id, races[index].id);
      }
      assert.ok(positions.every(row => !("probability" in row) && !("tipping" in row)));
    }
  }
});

test("Other leads and Other ties stay outside both party counts without moving the seat threshold", () => {
  for (const probabilities of [{ D: .1, R: .1, O: .8 }, { D: .5, R: 0, O: .5 }]) {
    const races = [...orderedRaces(4), race("other", probabilities)];
    const positions = groupRibbonMajorityPositions(layoutGroupRibbon(races, 3), { totalSeats: 5, majorityThreshold: 3 });
    assert.deepEqual(positions, [
      { id: "district-003", districtKey: "district-003", from: ["D"] },
      { id: "district-002", districtKey: "district-002", from: ["R"] },
    ]);
  }
});

test("no majority marker is inferred from partial, missing, duplicate, or insufficient main-path seats", () => {
  const chamber = { totalSeats: 5, majorityThreshold: 3 }, complete = layoutGroupRibbon(orderedRaces(5), 3);
  assert.deepEqual(groupRibbonMajorityPositions(complete), []);
  for (const races of [orderedRaces(4), orderedRaces(6), [...orderedRaces(4), race("missing", null)],
    [...orderedRaces(4), { ...race("duplicate-key", { D: 0, R: 1, O: 0 }), districtKey: "district-001" }],
    [...orderedRaces(2), ...["a", "b", "c"].map(id => race(id, { D: 0, R: 0, O: 1 }))]]) {
    assert.deepEqual(groupRibbonMajorityPositions(layoutGroupRibbon(races, 3), chamber), []);
  }
  for (const invalid of [{ totalSeats: 5, majorityThreshold: 2 }, { totalSeats: 5, majorityThreshold: 6 },
    { totalSeats: 5.1, majorityThreshold: 3 }, { totalSeats: 5, majorityThreshold: 3.1 }]) {
    assert.deepEqual(groupRibbonMajorityPositions(complete, invalid), []);
  }
});
