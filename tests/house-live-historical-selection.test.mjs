import assert from "node:assert/strict";
import test from "node:test";
import { houseHistoricalRunForYear } from "../lib/house-live-historical-selection.ts";

function catalog() {
  return { defaultRunId: "house_fixed_2024", runs: [
    { id: "house_fixed_2024", year: 2024, configurationId: "fixed", available: true },
    { id: "house_fixed_2020", year: 2020, configurationId: "fixed", available: true },
    { id: "house_other_2020", year: 2020, configurationId: "other", available: true },
    { id: "house_unavailable_2018", year: 2018, configurationId: "fixed", available: false },
  ] };
}

test("year-only historical links carry the catalog default model family into the requested year", () => {
  const value = catalog();
  assert.equal(houseHistoricalRunForYear(value, 2020), "house_fixed_2020");
});

test("missing or unsupported historical years return no run instead of the global default", () => {
  const value = catalog();
  assert.equal(houseHistoricalRunForYear(value, 2018), null);
  assert.equal(houseHistoricalRunForYear(value, 2026), null);
  assert.equal(houseHistoricalRunForYear(value, 2016), null);
});
