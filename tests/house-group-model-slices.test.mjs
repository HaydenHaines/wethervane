import test from "node:test";
import assert from "node:assert/strict";
import { houseGroupModelSlices } from "../lib/house-group-model-slices.ts";

const models = ["house_history_refreshed_v1", "house_presidential_slope_v1", "house_relative_lean_v1", "house_relative_context_v1"];
function publication() {
  const runs = [2018, 2020].flatMap(cycle => models.map(model => ({
    id: `${model}-${cycle}-h14-full435-v1`, cycle, fundamentalsId: model, label: model,
  })));
  const entry = runs.find(run => run.id === "house_relative_lean_v1-2020-h14-full435-v1");
  return { catalog: { runs, initialRunId: "house_history_refreshed_v1-2018-h14-full435-v1" }, entry,
    result: { id: entry.id, cycle: 2020, configuration: "house_relative_lean_v1-loading-1", races: Array(435), reviewStatus: "reviewed_experimental_research" } };
}

test("group builder retains all eight roles and offers same-year fundamentals with current run selected", () => {
  const slices = houseGroupModelSlices(publication());
  assert.deepEqual(slices.map(slice => slice.id), ["population", "community", "fundamentals", "environment", "measurement", "participation", "choice", "counts_rules"]);
  const fundamentals = slices.find(slice => slice.id === "fundamentals");
  assert.equal(fundamentals.controls[0].value, "house_relative_lean_v1-2020-h14-full435-v1");
  assert.deepEqual(fundamentals.controls[0].options.map(option => option.id), models.map(model => `${model}-2020-h14-full435-v1`));
  assert.ok(fundamentals.controls[0].options.every(option => option.years[0] === 2020));
});

test("winner target is required, while map, approximation, and unavailable roles remain explicit", () => {
  const slices = houseGroupModelSlices(publication());
  const choice = slices.find(slice => slice.id === "choice");
  assert.equal(choice.controls[0].required, true);
  assert.deepEqual(choice.controls[0].options.map(option => option.id), ["legacy", "full435"]);
  assert.match(slices.find(slice => slice.id === "population").unavailableReason, /maps are available.*population or demographic estimates/);
  assert.equal(slices.find(slice => slice.id === "counts_rules").builtIn, true);
  assert.match(slices.find(slice => slice.id === "counts_rules").builtInReason, /does not simulate vote counts, ranked-choice rounds, or runoffs/);
});
