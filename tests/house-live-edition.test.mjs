import assert from "node:assert/strict";
import test from "node:test";
import { resolveHouseEdition } from "../lib/house-live-edition.ts";

const available = { state: "response", body: { schema: "HouseLiveAvailabilityV1", available: true } };
const unavailable = { state: "response", body: { schema: "HouseLiveAvailabilityV1", available: false } };
const legacyRun = "house_history_refreshed_v1-2020-h14";
const groupRun = "house_history_refreshed_v1-2024-h14-full435-v1";
const pollingGroupRun = "house_presidential_slope_v1-2022-h14-full435-polls-v1";

test("explicit edition intent is stable regardless of current publication availability", () => {
  const current = resolveHouseEdition("?chamber=house&edition=current&run=house-current-polling", unavailable);
  assert.equal(current.status, "selected");
  assert.equal(current.edition, "current");
  assert.equal(current.selection.runId, "house-current-polling");

  const historical = resolveHouseEdition("?chamber=house&edition=historical&year=2020", available);
  assert.equal(historical.status, "selected");
  assert.equal(historical.edition, "historical");
  assert.equal(historical.selection.year, 2020);
});

test("bare House route waits for valid availability and treats API errors as errors, not unavailable", () => {
  assert.deepEqual(resolveHouseEdition("?chamber=house", available), {
    status: "selected", edition: "current",
    selection: { runId: null, year: 2026, view: null, raceId: null, comparisonId: null },
  });
  assert.deepEqual(resolveHouseEdition("?chamber=house", unavailable), {
    status: "selected", edition: "historical",
    selection: { runId: null, year: null, view: null, raceId: null, comparisonId: null },
  });
  assert.equal(resolveHouseEdition("?chamber=house", { state: "pending" }).status, "pending");
  assert.equal(resolveHouseEdition("?chamber=house", { state: "error" }).status, "error");
  assert.equal(resolveHouseEdition("?chamber=house", {
    state: "response", body: { schema: "WrongSchema", available: false },
  }).status, "error");
});

test("published group IDs and legacy House IDs remain historical without edition parameters", () => {
  for (const [runId, year] of [[groupRun, 2024], [pollingGroupRun, 2022], [legacyRun, 2020]]) {
    const result = resolveHouseEdition(`?chamber=house&run=${encodeURIComponent(runId)}`);
    assert.equal(result.status, "selected", runId);
    assert.equal(result.edition, "historical", runId);
    assert.equal(result.selection.runId, runId);
    assert.equal(result.selection.year, year);
  }
});

test("explicit 2026 year selects current; supported historical years and actual view stay historical", () => {
  const current = resolveHouseEdition("?chamber=house&year=2026", unavailable);
  assert.equal(current.status, "selected");
  assert.equal(current.edition, "current");

  const historical = resolveHouseEdition(`?chamber=house&year=2018&view=actual&run=house_relative_lean_v1-2018-h14`);
  assert.equal(historical.status, "selected");
  assert.equal(historical.edition, "historical");
  assert.equal(historical.selection.view, "actual");
  assert.equal(historical.selection.year, 2018);
});

test("conflicting, repeated, unknown, and unsupported edition/year/view combinations fail explicitly", () => {
  for (const search of [
    "?chamber=house&edition=current&year=2024",
    "?chamber=house&edition=historical&year=2026",
    "?chamber=house&edition=current&view=actual",
    "?chamber=house&edition=unknown",
    "?chamber=house&year=2016",
    "?chamber=house&year=02018",
    "?chamber=house&view=score",
    "?chamber=house&edition=current&edition=historical",
    "?chamber=house&year=2018&run=house_history_refreshed_v1-2020-h14",
    "?chamber=house&edition=current&run=house_history_refreshed_v1-2020-h14",
    "?chamber=house&edition=historical&run=house-current-polling",
    "?chamber=house&edition=historical&run=unknown-safe-id",
    "?chamber=house&edition=historical&year=2020&race=2026%3AAL-01%3Ahouse%3Aregular",
    "?chamber=house&edition=historical&year=2020&compare=house-current-polling",
    "?chamber=house&edition=historical&year=2020&compare=house_history_refreshed_v1-2024-h14",
  ]) assert.equal(resolveHouseEdition(search).status, "invalid", search);
});

test("current selection preserves generated run, district, and comparison choices", () => {
  const result = resolveHouseEdition("?chamber=house&edition=current&run=house-current-polling&race=2026%3ATX-23%3Ahouse%3Aregular&compare=house-current-fundamentals-none-standard");
  assert.deepEqual(result, { status: "selected", edition: "current", selection: {
    runId: "house-current-polling", year: 2026, view: null,
    raceId: "2026:TX-23:house:regular", comparisonId: "house-current-fundamentals-none-standard",
  } });
});

test("race-only and comparison-only links do not infer an edition or year", () => {
  assert.equal(resolveHouseEdition("?chamber=house&race=2026%3AAL-01%3Ahouse%3Aregular", available).status, "invalid");
  assert.equal(resolveHouseEdition("?chamber=house&compare=house-current-polling", unavailable).status, "invalid");
});
