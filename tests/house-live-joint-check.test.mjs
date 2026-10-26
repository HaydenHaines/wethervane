import assert from "node:assert/strict";
import test from "node:test";
import { makeHouseLiveResult } from "./fixtures/house-live-fixtures.mjs";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { checkHouseLiveJointRows, HouseLiveJointCheckError } from "../lib/house-live-joint-check.ts";

const inverseCdf = { seatQuantileConvention: "inverse_cdf" };
const linear = { seatQuantileConvention: "linear" };

function fourJointRows() {
  const allD = Array(435).fill(0);
  const otherInSecond = Array(435).fill(0);
  otherInSecond[1] = 2;
  const allRExceptUnopposed = Array(435).fill(1);
  allRExceptUnopposed[2] = 0;
  const otherAndUnopposed = [...allRExceptUnopposed];
  otherAndUnopposed[1] = 2;
  return [allD, otherInSecond, allRExceptUnopposed, otherAndUnopposed];
}
function parsedResult() {
  const result = parseHouseLiveResult(makeHouseLiveResult());
  assert.ok(result);
  return result;
}

test("checks every invented joint row against races and full chamber summaries", async () => {
  async function* rows() { for (const row of fourJointRows()) yield row; }
  assert.deepEqual(await checkHouseLiveJointRows(parsedResult(), rows(), inverseCdf),
    { rowsChecked: 4, raceCount: 435, chamberChecked: true });
});

test("rejects missing, extra, malformed, and out-of-support rows without filtering", async () => {
  const result = parsedResult(), valid = fourJointRows();
  const cases = [
    valid.slice(0, 3),
    [...valid, valid[0]],
    valid.map((row, index) => index === 0 ? row.slice(1) : row),
    valid.map((row, index) => index === 0 ? row.map((value, race) => race === 0 ? 3 : value) : row),
    valid.map((row, index) => index === 0 ? row.map((value, race) => race === 0 ? 0.5 : value) : row),
    valid.map((row, index) => index === 0 ? row.map((value, race) => race === 2 ? 1 : value) : row),
  ];
  for (const rows of cases)
    await assert.rejects(checkHouseLiveJointRows(result, rows, inverseCdf), HouseLiveJointCheckError);
});

test("rejects a different row coupling even when all race winner marginals agree", async () => {
  const rows = fourJointRows();
  // Move the O win between two rows, exchanging it with R at the same race.
  // Per-race D/R/O frequencies stay identical; the chamber seat PMFs change.
  rows[1][1] = 1;
  rows[2][1] = 2;
  const result = parsedResult();
  await assert.rejects(checkHouseLiveJointRows(result, rows, inverseCdf), /seat PMF differs/);
});

test("detects mismatched race, seat mean, quantile, and control summaries", async () => {
  const mutators = [
    result => { result.races[0].outcomes[0].p_win = 0.25; },
    result => { result.chamber.seats.D.expected += 0.01; },
    result => { result.chamber.seats.D.quantiles.p50 += 0.01; },
    result => { result.chamber.majority.D = 0.25; },
  ];
  for (const mutate of mutators) {
    const result = parsedResult();
    mutate(result);
    await assert.rejects(checkHouseLiveJointRows(result, fourJointRows(), inverseCdf), HouseLiveJointCheckError);
  }
});

test("with an unqualified chamber, still checks all rows and race frequencies", async () => {
  const raw = makeHouseLiveResult();
  raw.chamber = null;
  raw.qualification.limitations.push("House chamber seat totals are not qualified for this result.");
  const result = parseHouseLiveResult(raw);
  assert.ok(result);
  assert.deepEqual(await checkHouseLiveJointRows(result, fourJointRows(), inverseCdf),
    { rowsChecked: 4, raceCount: 435, chamberChecked: false });
  const changed = fourJointRows();
  changed[0][0] = 1;
  await assert.rejects(checkHouseLiveJointRows(result, changed, inverseCdf), /winner frequency differs/);
});

test("seat quantiles require the explicitly reviewed empirical convention", async () => {
  const inverseResult = parsedResult();
  await assert.rejects(checkHouseLiveJointRows(inverseResult, fourJointRows(), linear), /quantiles do not match/);
  assert.equal((await checkHouseLiveJointRows(inverseResult, fourJointRows(), inverseCdf)).chamberChecked, true);

  const linearResult = parsedResult();
  linearResult.chamber.seats.D.quantiles = { p05: 1, p10: 1, p50: 217.5, p90: 434.7, p95: 434.85 };
  linearResult.chamber.seats.R.quantiles = { p05: 0, p10: 0, p50: 216.5, p90: 433.7, p95: 433.85 };
  linearResult.chamber.seats.O.quantiles = { p05: 0, p10: 0, p50: 0.5, p90: 1, p95: 1 };
  await assert.rejects(checkHouseLiveJointRows(linearResult, fourJointRows(), inverseCdf), /quantiles do not match/);
  assert.equal((await checkHouseLiveJointRows(linearResult, fourJointRows(), linear)).chamberChecked, true);
  await assert.rejects(checkHouseLiveJointRows(linearResult, fourJointRows()), /convention is required/);
});
