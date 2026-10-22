import test from "node:test";
import assert from "node:assert/strict";
import { layoutHouseGroupRibbon } from "../lib/house-group-ribbon.ts";
import { layoutHouseRibbon } from "../lib/house-ribbon.ts";
import { layoutSerpentine } from "../lib/serpentine-layout.ts";
import { modeledWinnerFixture, unadmittedWinnerFixture } from "./fixtures/house-winner.mjs";
function fixture(district, probabilities) {
  const f = modeledWinnerFixture(); f.district = district;
  f.groups = f.groups.filter(g => Object.hasOwn(probabilities, g.bloc));
  f.prediction.probabilities = Object.entries(probabilities).map(([groupId, probability]) => ({ groupId, probability }));
  f.truth = { status: "unavailable", reason: "No invented truth provided." }; return f;
}
test("D/R-only order and coordinates exactly match existing Democratic ribbon", () => {
  const fixtures = [.1, .9, .5, .7, .7].map((p, i) => fixture(`AA-0${i}`, { D: p, R: 1 - p }));
  const expected = layoutHouseRibbon(fixtures.map(f => ({ id: f.district, districtKey: f.district, democraticWinProbability: f.prediction.probabilities[0].probability })), 2);
  const result = layoutHouseGroupRibbon(fixtures, 2);
  assert.deepEqual(result.main.rows.map(({ id, index, row, column, direction, visualIndex }) => ({ id, index, row, column, direction, visualIndex })), expected.rows.map(({ id, index, row, column, direction, visualIndex }) => ({ id, index, row, column, direction, visualIndex })));
  assert.deepEqual(result.other.rows, []); assert.deepEqual(result.mixedTies, []);
});
test("third-group mass stays explicit and O leaders are separately ordered", () => {
  const result = layoutHouseGroupRibbon([fixture("AA-01", { D: .4, R: .35, O: .25 }), fixture("AA-02", { D: .2, R: .1, O: .7 }), fixture("AA-03", { D: .1, R: .1, O: .8 })], 2);
  assert.deepEqual(result.main.rows[0].probabilities, { D: .4, R: .35, O: .25 });
  assert.ok(Math.abs(result.main.rows[0].balance - .05) < 1e-12);
  assert.notEqual(result.main.rows[0].probabilities.R, 1 - result.main.rows[0].probabilities.D);
  assert.deepEqual(result.other.rows.map(r => r.id), ["AA-03", "AA-02"]);
  assert.deepEqual(result.other.rows[0].leaders, ["O"]);
});
test("absent Democratic structural Republican support differs from unavailable prediction", () => {
  const structural = fixture("AA-01", { R: 1 }); structural.prediction = { status: "structural", groupId: "R", reason: "Sole explicitly admitted group." };
  const unavailable = unadmittedWinnerFixture(); unavailable.district = "AA-02";
  const result = layoutHouseGroupRibbon([structural, unavailable], 4);
  assert.deepEqual(result.main.rows[0].probabilities, { D: 0, R: 1, O: 0 }); assert.equal(result.main.rows[0].probabilitySource, "structural");
  assert.equal(result.unavailable[0].districtKey, "AA-02"); assert.equal(result.unavailable[0].probabilities, undefined); assert.equal(result.unavailable[0].balance, undefined);
});
test("D/R top ties stay main while normalized O ties have no partisan rank", () => {
  const result = layoutHouseGroupRibbon([fixture("AA-01", { D: .4, R: .4, O: .2 }), fixture("AA-02", { D: 1 / 3, R: 1 / 3, O: 1 / 3 }), fixture("AA-03", { D: .5, O: .5 })], 3);
  assert.deepEqual(result.main.rows[0].leaders, ["D", "R"]);
  assert.deepEqual(result.mixedTies.map(r => r.leaders), [["D", "R", "O"], ["D", "O"]]);
  assert.ok(result.mixedTies.every(r => r.index === undefined && r.row === undefined));
});
test("invalid mass is unavailable and duplicate districts or invalid columns reject", () => {
  const invalid = fixture("AA-01", { D: .7, R: .7 });
  const result = layoutHouseGroupRibbon([invalid], 2); assert.equal(result.main.rows.length, 0); assert.equal(result.unavailable.length, 1); assert.equal(result.unavailable[0].probabilities, undefined);
  assert.throws(() => layoutHouseGroupRibbon([invalid, invalid], 2), /Duplicate House district/);
  assert.throws(() => layoutHouseGroupRibbon([], 0), RangeError);
});
test("all 435 invented districts remain accounted for with unique valid coordinates", () => {
  const fixtures = Array.from({ length: 435 }, (_, i) => {
    const district = `${String.fromCharCode(65 + Math.floor(i / 99))}A-${String(i % 99).padStart(2, "0")}`;
    const f = fixture(district, i % 4 === 0 ? { D: .6, R: .3, O: .1 } : i % 4 === 1 ? { D: .1, R: .2, O: .7 } : { D: 1 / 3, R: 1 / 3, O: 1 / 3 });
    if (i % 4 === 3) f.prediction = { status: "unavailable", reason: "Missing invented prediction." }; return f;
  });
  const result = layoutHouseGroupRibbon(fixtures, 16);
  const entries = [...result.main.rows, ...result.other.rows, ...result.mixedTies, ...result.unavailable];
  assert.equal(entries.length, 435); assert.equal(new Set(entries.map(r => r.id)).size, 435);
  for (const layout of [result.main, result.other]) {
    assert.equal(new Set(layout.rows.map(r => `${r.row}:${r.column}`)).size, layout.rows.length);
    assert.ok(layout.rows.every(r => r.column >= 0 && r.column < 16 && r.visualIndex === r.row * 16 + r.column));
  }
  assert.ok(entries.every(r => !["seat", "control", "tipping", "majorityIndex"].some(k => k in r)));
});
test("generic serpentine helper preserves caller order without semantic inputs", () => {
  const ordered = [{ label: "Z" }, { label: "A" }, { label: "M" }];
  const result = layoutSerpentine(ordered, 2);
  assert.deepEqual(result.rows.map(r => r.label), ["Z", "A", "M"]);
  assert.equal(result.rows[2].column, 1); assert.equal(result.rows[2].direction, "right-to-left");
  assert.deepEqual(ordered, [{ label: "Z" }, { label: "A" }, { label: "M" }]);
});
