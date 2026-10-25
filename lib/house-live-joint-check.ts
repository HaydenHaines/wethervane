/** Numerical consistency of decoded joint winner rows; this does not authenticate their bytes. */
import type { HouseLiveParty, HouseLiveResult, HouseLiveSeatDistribution } from "./house-live-result.ts";

/** Native probability/simplex checks use 1e-8; seat means and quantiles may interpolate. */
export const HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE = 1e-8;
export const HOUSE_LIVE_JOINT_SEAT_TOLERANCE = 1e-6;
const PARTIES = ["D", "R", "O"] as const;
const QUANTILES = { p05: 0.05, p10: 0.10, p50: 0.50, p90: 0.90, p95: 0.95 } as const;

export class HouseLiveJointCheckError extends Error {}
export interface HouseLiveJointCheck {
  rowsChecked: number;
  raceCount: 435;
  chamberChecked: boolean;
}
export interface HouseLiveJointCheckOptions {
  /** Must come from the reviewed export specification, not be inferred from matching output. */
  seatQuantileConvention: "inverse_cdf" | "linear";
}

function requireCheck(condition: unknown, message: string): asserts condition {
  if (!condition) throw new HouseLiveJointCheckError(message);
}
function near(actual: number, expected: number, tolerance: number): boolean {
  return Number.isFinite(actual) && Number.isFinite(expected) &&
    Math.abs(actual - expected) <= tolerance;
}
function nth(histogram: Float64Array, index: number): number {
  let count = 0;
  for (let seats = 0; seats < histogram.length; seats++) {
    count += histogram[seats];
    if (count > index) return seats;
  }
  throw new HouseLiveJointCheckError("Seat histogram has fewer draws than declared.");
}
function inverseCdf(histogram: Float64Array, count: number, q: number): number {
  return nth(histogram, Math.max(0, Math.ceil(q * count) - 1));
}
function linearQuantile(histogram: Float64Array, count: number, q: number): number {
  const position = (count - 1) * q;
  const low = Math.floor(position), high = Math.ceil(position);
  return nth(histogram, low) + (position - low) * (nth(histogram, high) - nth(histogram, low));
}

/**
 * The caller must bind the quantile convention from a reviewed export. This
 * checks summaries, not full covariance; exact matrix identity requires pinned
 * source bytes. It never invents precision gates, tipping points, or chain order.
 */
export async function checkHouseLiveJointRows(
  result: HouseLiveResult,
  rows: AsyncIterable<readonly number[]> | Iterable<readonly number[]>,
  options: HouseLiveJointCheckOptions,
): Promise<HouseLiveJointCheck> {
  requireCheck(options?.seatQuantileConvention === "inverse_cdf" || options?.seatQuantileConvention === "linear",
    "A reviewed seat-quantile convention is required for joint row checking.");
  const raceCount = result.races.length;
  requireCheck(raceCount === 435 && result.joint_draws.ordered_race_ids.length === raceCount,
    "Joint check requires the complete ordered 435-race result.");
  const expectedRows = result.joint_draws.count;
  requireCheck(Number.isSafeInteger(expectedRows) && expectedRows > 0, "Invalid declared joint draw count.");
  const support = result.races.map((race, index) => {
    requireCheck(result.joint_draws.ordered_race_ids[index] === race.race_id,
      `Joint draw order does not match race ${index}.`);
    const parties = new Set(race.outcomes.map(outcome => outcome.outcome_id));
    requireCheck(parties.size === race.outcomes.length && [...parties].every(party => PARTIES.includes(party)),
      `Race ${race.race_id} has invalid exported winner support.`);
    return parties;
  });
  // Float64 counters exactly represent every integer through Number.MAX_SAFE_INTEGER.
  const raceWins = Array.from({ length: raceCount }, () => new Float64Array(3));
  const seatHistograms: Record<HouseLiveParty, Float64Array> = {
    D: new Float64Array(436), R: new Float64Array(436), O: new Float64Array(436),
  };
  const control = { D: 0, R: 0, unresolved: 0 };
  let rowsChecked = 0;
  for await (const row of rows) {
    requireCheck(rowsChecked < expectedRows, "Joint source contains more rows than declared.");
    requireCheck(Array.isArray(row) && row.length === raceCount,
      `Joint row ${rowsChecked} does not have 435 ordered winner values.`);
    const seats = [0, 0, 0];
    for (let index = 0; index < raceCount; index++) {
      const winner = row[index];
      requireCheck(Number.isInteger(winner) && winner >= 0 && winner <= 2,
        `Joint row ${rowsChecked} has an invalid winner encoding at race ${index}.`);
      const party = PARTIES[winner];
      requireCheck(support[index].has(party),
        `Joint row ${rowsChecked} selects ${party} outside race ${result.races[index].race_id}'s exported support.`);
      raceWins[index][winner]++;
      seats[winner]++;
    }
    requireCheck(seats[0] + seats[1] + seats[2] === 435,
      `Joint row ${rowsChecked} does not conserve 435 House seats.`);
    for (let party = 0; party < 3; party++) seatHistograms[PARTIES[party]][seats[party]]++;
    if (seats[0] >= 218) control.D++;
    else if (seats[1] >= 218) control.R++;
    else control.unresolved++;
    rowsChecked++;
  }
  requireCheck(rowsChecked === expectedRows, `Joint source has ${rowsChecked} rows; expected ${expectedRows}.`);

  for (let index = 0; index < raceCount; index++) {
    for (const outcome of result.races[index].outcomes) {
      const encoding = result.joint_draws.winner_encoding[outcome.outcome_id];
      requireCheck(near(raceWins[index][encoding] / rowsChecked, outcome.p_win,
        HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE),
        `Joint winner frequency differs from race ${result.races[index].race_id}, group ${outcome.outcome_id}.`);
    }
  }
  if (result.chamber === null) return { rowsChecked, raceCount: 435, chamberChecked: false };

  // The same decoded rows must reproduce every published marginal chamber summary.
  const summaries = PARTIES.map(party => ({ party, histogram: seatHistograms[party],
    published: result.chamber!.seats[party] }));
  for (const { party, histogram, published } of summaries) {
    checkSeatDistribution(party, histogram, published, rowsChecked);
  }
  const quantile = options.seatQuantileConvention === "inverse_cdf" ? inverseCdf : linearQuantile;
  requireCheck(summaries.every(({ histogram, published }) =>
    (Object.entries(QUANTILES) as [keyof typeof QUANTILES, number][]).every(([key, q]) =>
      near(published.quantiles[key], quantile(histogram, rowsChecked, q),
        HOUSE_LIVE_JOINT_SEAT_TOLERANCE))),
    `Published chamber seat quantiles do not match the reviewed ${options.seatQuantileConvention} convention.`);
  requireCheck(near(result.chamber.majority.D, control.D / rowsChecked,
      HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE) &&
    near(result.chamber.majority.R, control.R / rowsChecked,
      HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE) &&
    near(result.chamber.majority.unresolved, control.unresolved / rowsChecked,
      HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE),
  "Published House majority probabilities do not match the joint rows.");
  return { rowsChecked, raceCount: 435, chamberChecked: true };
}

function checkSeatDistribution(party: HouseLiveParty, histogram: Float64Array,
  published: HouseLiveSeatDistribution, count: number): void {
  requireCheck(published.pmf.length === 436, `Published ${party} seat PMF has the wrong length.`);
  let expected = 0;
  for (let seats = 0; seats <= 435; seats++) {
    const probability = histogram[seats] / count;
    requireCheck(near(published.pmf[seats], probability, HOUSE_LIVE_JOINT_PROBABILITY_TOLERANCE),
      `Published ${party} seat PMF differs from joint rows at ${seats} seats.`);
    expected += seats * probability;
  }
  requireCheck(near(published.expected, expected, HOUSE_LIVE_JOINT_SEAT_TOLERANCE),
    `Published ${party} expected seats differ from joint rows.`);
}
